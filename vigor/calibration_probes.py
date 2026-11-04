"""Per-mask Gaussian trace-probe geometry for per-trait moment uncertainty.

Rademacher probes estimate tr(Ai Ak Aj Al) for the residualized forms of each
distinct observation mask. Each kernel is aligned once; the probes of many
masks share wide float32 full-kernel GEMMs with the residual projection applied
on the fly, and trace reductions accumulate in float64. There is no per-mask
kernel copy, animal-scale eigensolve, factorization or dense matrix square. The tensors do not depend on
the fitted covariance, so one geometry serves every candidate covariance of a
training fold; :func:`probe_cache` persists it under the supervisor's
immutable data/fold identity. PSD animal kernels are an input contract.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import numbers
import pickle

import numpy as np
import pandas as pd
from scipy.linalg import qr

from .progress import report_progress

_scope=ContextVar('vigor_eb_probe_cache',default=None)


@contextmanager
def probe_cache(directory=None,key=None,max_bytes=2*1024**3):
    """Use the supervisor's immutable data/fold identity for probe geometry."""
    token=_scope.set(None if directory is None else (Path(directory),str(key),int(max_bytes)))
    try:yield
    finally:_scope.reset(token)


def _kernel_subset(kernel, index, indices):
    """Dense float64 observed block of a labeled/positional kernel; O(n_obs²)."""
    if isinstance(kernel, pd.DataFrame):
        if kernel.index.has_duplicates or kernel.columns.has_duplicates:
            raise ValueError("kernel animal labels must be unique")
        labels = index[indices]
        value = kernel.loc[labels, labels].to_numpy(dtype=np.float64)
    elif hasattr(kernel, "submatrix"):
        if tuple(kernel.shape) != (len(index), len(index)):
            raise ValueError("kernel shape must match the animal axis")
        value = np.asarray(kernel.submatrix(indices), dtype=np.float64)
    else:
        if tuple(kernel.shape) != (len(index), len(index)):
            raise ValueError("kernel shape must match the animal axis")
        if isinstance(kernel, np.ndarray):
            value = np.asarray(kernel[np.ix_(indices, indices)], dtype=np.float64)
        elif hasattr(kernel, "oindex"):
            value = np.asarray(kernel.oindex[indices, indices], dtype=np.float64)
        else:
            # One full row temporarily, rather than n_observed x n_full.
            value = np.empty((len(indices), len(indices)), dtype=np.float64)
            for row, i in enumerate(indices):
                value[row] = np.asarray(kernel[int(i), :], dtype=np.float64)[indices]
    if value.shape != (len(indices), len(indices)) or not np.isfinite(value).all():
        raise ValueError("observed kernel block must be finite and square")
    if not np.allclose(value, value.T, rtol=1e-6, atol=1e-8):
        raise ValueError("kernel must be symmetric")
    return (value + value.T) * 0.5


def _residual_basis(design):
    q, r, _ = qr(design, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    tolerance = (float(diagonal.max()) if diagonal.size else 0.0) * max(design.shape) * np.finfo(float).eps
    return q[:, :int(np.sum(diagonal > tolerance))]


def _residualize_kernel(kernel, fixed_basis):
    # Q K Q without forming Q K as two cubic dense GEMMs.
    u = fixed_basis
    ku = kernel @ u
    centered = kernel - ku @ u.T - u @ ku.T + u @ (u.T @ ku) @ u.T
    return (centered + centered.T) * 0.5


def _base(kernels,frame,design,n_probes,seed,block,kernel_rank='auto'):
    """Per-mask trace tensors for every distinct observation mask.

    For a mask m with residual projection Q_m, A_m x = Q_m K_mm Q_m x. With a
    retained kernel_rank (default 'auto' = 512) K is the shared grm2Us
    eigensystem K~ = U (s-b) U' + b I of vigor.kernel_eigen (computed once per
    kernel and reused by the grm2us sandwich and every hyper_opt fold), so
    K~_mm x = U_m (s-b) U_m'x + b x costs O(n_m r P) per mask. With
    kernel_rank='full' the kernel is aligned once (float32) and the probes of
    many masks are padded side by side into one n x (masks*probes) GEMM (block
    sets the column budget in units of 256 columns), O(n² P) per mask. No
    per-mask kernel copy is formed; trace reductions accumulate in float64.
    """
    values=frame.to_numpy(float);groups={}
    for j in range(frame.shape[1]):
        mask=np.isfinite(values[:,j]);groups.setdefault(mask.tobytes(),[]).append(j)
    output=[None]*frame.shape[1];c=len(kernels);d=c+1;n_all=len(frame)
    from .kernel_eigen import eigensystem,resolve_rank
    full=[None]*c;low=[None]*c;diagonals=[]
    for i,(name,kernel) in enumerate(kernels.items()):
        aligned=lambda kernel=kernel:_kernel_subset(kernel,frame.index,np.arange(n_all)).astype(np.float32)
        r=resolve_rank(kernel_rank,n_all,name,i)
        if r is None:
            full[i]=aligned();diagonals.append(np.diag(full[i]).astype(np.float64))
        else:
            # Same seed as the grm2us sandwich, so one decomposition serves both.
            eig=eigensystem(kernel,aligned,r,seed=113*i)
            low[i]=(eig['U'],(eig['values']-eig['bulk']).astype(np.float32),np.float32(eig['bulk']))
            diagonals.append(np.asarray(eig['diag'],float))
    masks=[]
    for mask_bytes,columns in groups.items():
        indices=np.flatnonzero(np.frombuffer(mask_bytes,dtype=bool));n=len(indices)
        u=_residual_basis(design[indices]);rank=u.shape[1]
        if n-rank<d:raise ValueError('insufficient observations for identifiable per-trait moments')
        scales=np.asarray([float(diag[indices].mean()) for diag in diagonals])
        if np.any(scales<=0):raise ValueError('kernel mean observed diagonals must be positive')
        # Reuse the same probes for shared masks and all covariance candidates.
        digest=hashlib.sha256(mask_bytes+np.asarray(design[indices]).tobytes()).digest()
        rng=np.random.default_rng(np.random.SeedSequence([seed,int.from_bytes(digest[:8],'little')]))
        z=(rng.integers(0,2,size=(n_probes,n),dtype=np.int8).T*2-1).astype(np.float32)
        masks.append({'indices':indices,'u':u.astype(np.float32),'columns':columns,'scales':scales,
                      'z':z,'n':n,'rank':rank})

    def project(m,x):
        return x-m['u']@(m['u'].T@x)

    def kernel_apply(i,items):
        """Q_m K_i Q_m x for every (mask, projected x) item through one padded GEMM."""
        if low[i] is not None:
            U,shrink,bulk=low[i]
            out=[]
            for m,x in items:
                Um=U[m['indices']]
                out.append(project(m,Um@(shrink[:,None]*(Um.T@x))+bulk*x))
            return out
        widths=[x.shape[1] for _,x in items]
        pad=np.zeros((n_all,sum(widths)),dtype=np.float32);start=0
        for (m,x),w in zip(items,widths):
            pad[m['indices'],start:start+w]=x;start+=w
        product=full[i]@pad;del pad
        out=[];start=0
        for (m,_),w in zip(items,widths):
            out.append(project(m,product[m['indices'],start:start+w]));start+=w
        return out

    budget=max(1,int(block))*256
    per_mask=d*n_probes
    step=max(1,budget//per_mask)
    done=0
    for first in range(0,len(masks),step):
        chunk=masks[first:first+step]
        qz=[project(m,m['z']) for m in chunk]
        # az[i][q] = A_i z for mask q; the identity form is the projection itself.
        az=[kernel_apply(i,list(zip(chunk,qz))) for i in range(c)]+[qz]
        # pair[k][i] = A_k A_i z; A_identity leaves projected vectors unchanged.
        pair=[]
        for k in range(c):
            flat=kernel_apply(k,[(m,az[i][q]) for q,m in enumerate(chunk) for i in range(d)])
            pair.append([[flat[q*d+i] for q in range(len(chunk))] for i in range(d)])
        pair.append(az)
        for q,m in enumerate(chunk):
            tensor=np.empty((d,d,d,d,n_probes),dtype=float)
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        for l in range(d):
                            # z' Ai Ak Aj Al z = (Ak Ai z)'(Aj Al z).
                            tensor[i,j,k,l]=2*np.sum(pair[k][i][q]*pair[j][l][q],axis=0,dtype=np.float64)
            for j in m['columns']:
                output[j]={'scales':m['scales'],'tensor':tensor,'n':m['n'],'rank':m['rank']}
        done+=len(chunk)
        report_progress(kind='calibration',phase='h² prior geometry',phase_key='calibration',
            trait_index=chunk[-1]['columns'][0]+1,trait_total=frame.shape[1],trait_completed=0,
            trait=str(frame.columns[chunk[-1]['columns'][0]]),n_observed=chunk[-1]['n'],
            calibration_step=f'masks {done}/{len(masks)}: Gaussian trace probes {n_probes}')
    return output


def probe_geometry(kernels,frame,*,n_probes=64,probe_seed=42,probe_block_size=16,design=None,kernel_rank='auto'):
    """Return per-trait trace tensors 2 z'Ai Ak Aj Al z, shared by masks.

    kernels is an ordered name->kernel mapping aligned to frame rows; frame is
    the animal x trait outcome table (NaN = unobserved). design defaults to an
    intercept. The tensor order is kernels..., identity. kernel_rank ('auto' =
    512, an integer, 'full', or per-kernel mapping/sequence) selects the shared
    grm2Us approximation, O(n² r) once per kernel then O(c² P n_m r) per mask,
    or the full kernel, O(c² P n²) per mask as wide shared GEMMs.
    Returns (list of per-trait dicts, cache_hit).
    """
    for name,value,minimum in [('n_probes',n_probes,8),('probe_seed',probe_seed,0),('probe_block_size',probe_block_size,1)]:
        if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
    design=np.ones((len(frame),1)) if design is None else np.asarray(design,float)
    context=_scope.get();path=None;hit=False
    if context is not None:
        directory,identity,budget=context
        tag=hashlib.sha256((identity+repr((n_probes,probe_seed,probe_block_size,tuple(map(str,frame.columns)),tuple(kernels),kernel_rank))).encode()+design.tobytes()).hexdigest()
        path=directory/'h2_prior_probes'/f'{tag}.pkl'
        if path.exists():
            with path.open('rb') as stream:base=pickle.load(stream)
            hit=True
    if not hit:
        base=_base(kernels,frame,design,int(n_probes),int(probe_seed),int(probe_block_size),kernel_rank)
        if path is not None:
            payload=pickle.dumps(base,protocol=5)
            used=sum(p.stat().st_size for p in directory.rglob('*.pkl')) if directory.exists() else 0
            if used+len(payload)<=budget:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(path)
    report_progress(kind='calibration',phase='h² prior geometry',phase_key='calibration',
                    calibration_step='reused fold trace geometry' if hit else 'trace geometry ready')
    return base,hit
