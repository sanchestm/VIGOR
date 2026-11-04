"""Scalable relationship operators and pedigree precision utilities (VIGOR 2.7).

The operator layer is compatible with the 2.6 covariance/mvBLUP acceleration.  It
avoids rebuilding standardized genotype blocks on every PCG matvec by caching the
bounded block representation after the first pass when ``cache_blocks=True``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .kinship import pedigree_inbreeding_sparse


def dense_grm_bytes(n: int, dtype=np.float32) -> int:
    """Construct dense genomic relationship matrix bytes.

    Parameters
    ----------
    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    object
        Computed result from ``int(n) * int(n) * np.dtype(dtype).itemsize``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return int(n) * int(n) * np.dtype(dtype).itemsize


def matrix_free_working_bytes(n: int, block_variants: int, n_rhs: int = 1, dtype=np.float32) -> int:
    """Return the matrix free working bytes result computed by ``int``.

    Parameters
    ----------
    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    block_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.

    n_rhs : int
        Input for the operation ``item * (n * block_variants + 3 * n * n_rhs + block_variants *
        n_rhs)``. See ``vigor.scalable.matrix_free_working_bytes`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``1``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    object
        Computed result from ``int(item * (n * block_variants + 3 * n * n_rhs + block_variants * n_rhs))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    item=np.dtype(dtype).itemsize
    return int(item*(n*block_variants + 3*n*n_rhs + block_variants*n_rhs))


class StreamingGRMOperator(LinearOperator):
    """Matrix-free K=ZZ'/m operator with optional bounded standardized-block cache.

    ``block_factory`` must return a fresh iterable of sample x variant blocks on
    first use.  With ``cache_blocks=True`` (the 2.7 default), standardized blocks
    are retained after that first pass so repeated PCG matvecs do not repeatedly
    decode and standardize the genome.  Set ``max_cache_bytes`` to enforce a hard
    cache ceiling.  The cache is used only when the *entire* standardized
    genome fits under the ceiling; otherwise every matvec streams all blocks
    again.  This avoids the silent prefix truncation bug present in 2.9.1.
    """
    def __init__(self, n_samples: int, block_factory: Callable[[], Iterable[np.ndarray]], *,
                 n_variants: int | None=None, dtype=np.float32, cache_blocks=True,
                 max_cache_bytes: int | None=None, fit_mask=None):
        """Initialize StreamingGRMOperator with the inputs and resource settings below.

        Parameters
        ----------
        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        block_factory : Callable[[], Iterable[np.ndarray]]
            Zero-argument factory returning a fresh iterable of sample × variant numerical blocks.
            All blocks use the same sample order; each traversal must reproduce the genotype stream
            for operator products.

        n_variants : int | None
            Number of variants in the logical source axis, including the unpadded extent of packed
            arrays. Default: ``None``. None delegates to the local fallback shown in this body; it
            is not an additional numeric/category value.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``.

        cache_blocks : bool
            Whether to perform the named persistence/materialization/refitting step. True enables
            it; False follows the bypass path in this routine. Refitting during CV must use training
            outcomes only. Default: ``True``.

        max_cache_bytes : int | None
            Memory budget used by the allocation/chunking guard. Units are bytes for names ending in
            bytes; maxmem uses the resource helper's GiB convention. None delegates to that helper's
            available-memory policy. Default: ``None``.

        fit_mask : one-dimensional bool array or None
            Mask selecting training animals for fitting frequencies, scaling or model parameters.
            Length must equal the animal axis; None uses the routine's full eligible training set.
            Keep evaluation outcomes outside this mask. Default: ``None``.
        """
        self.n_samples=int(n_samples); self.block_factory=block_factory
        self.n_variants=None if n_variants is None else int(n_variants)
        self.work_dtype=np.dtype(dtype); self.cache_blocks=bool(cache_blocks)
        self.max_cache_bytes=None if max_cache_bytes is None else int(max_cache_bytes)
        self.fit_mask=np.ones(self.n_samples,bool) if fit_mask is None else np.asarray(fit_mask,bool)
        if self.fit_mask.shape!=(self.n_samples,): raise ValueError('fit_mask shape mismatch')
        self._cache=[]; self._cache_complete=False; self._denom=0.0; self._diag_cache=None
        super().__init__(dtype=self.work_dtype,shape=(self.n_samples,self.n_samples))

    def _standardized_blocks(self):
        """Iterate genotype blocks standardized using training-only frequencies and missing-call policy.

        Yields
        ------
        object
            Successive blocks or values in iterator order, following the representation described above.
        """
        if self._cache_complete:
            yield from self._cache
            return
        cache_bytes=0
        denom=0.0
        built=[]
        all_blocks_cached=True
        for X0 in self.block_factory():
            X=np.asarray(X0,dtype=self.work_dtype)
            if X.ndim!=2 or X.shape[0]!=self.n_samples:
                raise ValueError('genotype block shape mismatch')
            fit=X[self.fit_mask]
            mu=np.nanmean(fit,axis=0)
            sd=np.nanstd(fit,axis=0,ddof=0)
            keep=np.isfinite(sd)&(sd>1e-8)
            if not keep.any():
                continue
            Z=(X[:,keep]-mu[keep][None,:])/sd[keep][None,:]
            Z[~np.isfinite(Z)]=0.0
            Z=np.ascontiguousarray(Z,dtype=self.work_dtype)
            denom += Z.shape[1]
            if self.cache_blocks:
                fits=(self.max_cache_bytes is None or cache_bytes+Z.nbytes<=self.max_cache_bytes)
                if fits and all_blocks_cached:
                    built.append(Z)
                    cache_bytes += Z.nbytes
                else:
                    # A partial cache is unsafe unless the operator also tracks
                    # and replays the uncached suffix.  Prefer a full cache or a
                    # fully streamed pass; correctness is more important than a
                    # misleading bounded prefix cache.
                    all_blocks_cached=False
                    built=[]
                    cache_bytes=0
            yield Z
        self._denom=max(float(denom),1.0)
        if self.cache_blocks and all_blocks_cached:
            self._cache=built
            self._cache_complete=True
        else:
            self._cache=[]
            self._cache_complete=False

    def _matmat(self, V):
        """Apply the operator to a matrix for the LinearOperator interface.

        Parameters
        ----------
        V : array-like
            Numerical input passed to ``np.zeros`` as ``dtype``; see that callable for required
            dimensions and matrix role. Passed to ``np.zeros`` as ``dtype``.
        """
        V=np.asarray(V,dtype=self.work_dtype); was1=V.ndim==1
        if was1: V=V[:,None]
        out=np.zeros((self.n_samples,V.shape[1]),dtype=np.result_type(V.dtype,np.float32)); denom=0
        for Z in self._standardized_blocks():
            out += Z@(Z.T@V); denom += Z.shape[1]
        den=self._denom if self._cache_complete else max(float(denom),1.0)
        out/=den
        return out[:,0] if was1 else out

    def _matvec(self,v): 
        """Apply the operator to a vector for the LinearOperator interface.

        Parameters
        ----------
        v : one-dimensional array-like
            Vector supplied by the linear operator/iterative-solver callback. Length equals the
            enclosing operator dimension; observed-cell operators use the prepared finite-cell
            order.
        """
        return self._matmat(v)
    def matmat(self,V): 
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        V : array-like
            Numerical operand in ``self._matmat(V)``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``self._matmat(V)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self._matmat(V)
    def diag(self):
        """Return the diagonal variances of the represented covariance/kernel.

        Returns
        -------
        object
            Computed result from ``self._diag_cache``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if self._diag_cache is not None:return self._diag_cache
        d=np.zeros(self.n_samples,float);den=0
        for Z in self._standardized_blocks(): d+=np.sum(Z*Z,axis=1);den+=Z.shape[1]
        self._diag_cache=d/(self._denom if self._cache_complete else max(den,1));return self._diag_cache


class LowRankGRMOperator(LinearOperator):
    """LowRankGRMOperator represents low rank genomic relationship matrix operator.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    def __init__(self,U,s,*,residual_eigenvalue=0.0,dtype=np.float32):
        """Initialize LowRankGRMOperator with the inputs and resource settings below.

        Parameters
        ----------
        U : array-like
            Numerical input passed to ``super().__init__`` as ``dtype``; see that callable for
            required dimensions and matrix role. Passed to ``super().__init__`` as ``dtype``.

        s : array-like
            Numerical operand in ``s - self.bulk``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.

        residual_eigenvalue : float
            Eigenvalue assigned to directions outside the retained basis. A nonnegative value
            supplies the bulk covariance contribution; zero discards that omitted-space
            contribution. Default: ``0.0``.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``.
        """
        U=np.asarray(U,dtype=dtype);s=np.asarray(s,dtype=dtype).reshape(-1)
        if U.ndim!=2 or U.shape[1]!=len(s):raise ValueError('U/s mismatch')
        self.U=U;self.s=s;self.bulk=float(residual_eigenvalue);self.delta=s-self.bulk
        super().__init__(dtype=U.dtype,shape=(U.shape[0],U.shape[0]))
    def _matvec(self,v): 
        """Apply the operator to a vector for the LinearOperator interface.

        Parameters
        ----------
        v : one-dimensional array-like
            Vector supplied by the linear operator/iterative-solver callback. Length equals the
            enclosing operator dimension; observed-cell operators use the prepared finite-cell
            order.
        """
        return self.matmat(np.asarray(v)[:,None])[:,0]
    def _matmat(self,V): 
        """Apply the operator to a matrix for the LinearOperator interface.

        Parameters
        ----------
        V : array-like
            Numerical operand in ``self.matmat(V)``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.
        """
        return self.matmat(V)
    def matmat(self,V):
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        V : array-like
            Numerical operand in ``self.U @ (self.delta[:, None] * (self.U.T @ V))``. Shape must
            conform to that matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        V=np.asarray(V);out=self.U@(self.delta[:,None]*(self.U.T@V))
        if self.bulk:out+=self.bulk*V
        return out
    def diag(self): 
        """Return the diagonal variances of the represented covariance/kernel.

        Returns
        -------
        object
            Computed result from ``self.U * self.U @ self.delta + self.bulk``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.U*self.U)@self.delta+self.bulk


class MultiTraitCovarianceOperator(LinearOperator):
    """Operator for sum_k K_k kron G_k + I kron E on full N x T arrays."""
    def __init__(self,kernels:dict[str,Any],trait_covariances:dict[str,np.ndarray],E):
        """Initialize MultiTraitCovarianceOperator with the inputs and resource settings below.

        Parameters
        ----------
        kernels : dict[str, Any]
            Named animal relationship kernels; each matrix/operator must use the same animal order.
            Names identify the matching trait covariance components.

        trait_covariances : dict[str, np.ndarray]
            Trait × trait genetic covariance for each named animal kernel. Every kernel name must
            have a corresponding component in the same trait order.

        E : array-like or covariance specification or None
            Residual trait covariance in observation-column order, including cross-trait residual
            dependence. Prediction can recover this component from a covariance-fit result when None
            is supplied; a separate genetic matrix requires an explicit compatible residual
            covariance.
        """
        self.kernels=kernels;self.G={k:np.asarray(trait_covariances[k],float) for k in kernels};self.E=np.asarray(E,float)
        self.n=next(iter(kernels.values())).shape[0];self.t=self.E.shape[0]
        super().__init__(dtype=np.dtype(float),shape=(self.n*self.t,self.n*self.t))
    def _matvec(self,v):
        """Apply the operator to a vector for the LinearOperator interface.

        Parameters
        ----------
        v : one-dimensional array-like
            Vector supplied by the linear operator/iterative-solver callback. Length equals the
            enclosing operator dimension; observed-cell operators use the prepared finite-cell
            order.
        """
        A=np.asarray(v).reshape(self.n,self.t);R=A@self.E
        for name,K in self.kernels.items():
            KA=K.matmat(A) if hasattr(K,'matmat') else np.asarray(K)@A
            R+=KA@self.G[name]
        return R.reshape(-1)


@dataclass
class PedigreePrecision:
    """PedigreePrecision represents pedigree precision.

    Attributes
    ----------
    matrix : sparse.csr_matrix
        Stored matrix; its axis/order follows the construction inputs.
    ids : list[str]
        Stored ids; its axis/order follows the construction inputs.
    inbreeding : pd.Series
        Stored inbreeding; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    matrix : sparse.csr_matrix
        Stored matrix consumed by ``to_zarr``/``to_xarray``; see those methods for the
        numerical/label contract. Required construction field.

    ids : list[str]
        Stored ids consumed by ``to_zarr``/``to_xarray``; see those methods for the
        numerical/label contract. Required construction field.

    inbreeding : pd.Series
        Stored inbreeding consumed by ``to_xarray``; see those methods for the numerical/label
        contract. Required construction field.
    """
    matrix: sparse.csr_matrix
    ids: list[str]
    inbreeding: pd.Series

    def to_zarr(self, destination, *, group="relationship/Ainv", overwrite=False, **kwargs):
        """Return the to zarr result computed by ``write_sparse_relationship_zarr``.

        Parameters
        ----------
        destination : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'relationship/Ainv'``. Passed to ``write_sparse_relationship_zarr`` as
            ``group``.

        overwrite : bool
            Whether an existing destination may be replaced. False protects the existing artifact;
            True permits this writer to recreate it. Default: ``False``. Passed to
            ``write_sparse_relationship_zarr`` as ``overwrite``.

        **kwargs : mapping
            Additional keyword arguments forwarded to ``write_sparse_relationship_zarr``. See those
            callables for accepted keys and values. Passed to ``write_sparse_relationship_zarr`` as
            ``**expanded mapping``.

        Returns
        -------
        object
            Computed result from ``write_sparse_relationship_zarr(self.matrix, destination, ids=self.ids, group=group, name='Ainv', overwrite=overwrite, attrs={'source': 'Henderson pedigree precision'}, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        from .sparse_relationship import write_sparse_relationship_zarr
        return write_sparse_relationship_zarr(self.matrix, destination, ids=self.ids,
                                               group=group, name="Ainv", overwrite=overwrite,
                                               attrs={"source":"Henderson pedigree precision"}, **kwargs)

    def to_xarray(self):
        """Compact in-memory xarray CSR view; never constructs a dense matrix.

        Returns
        -------
        object
            Computed result from ``xr.Dataset({'data': (('nnz',), M.data), 'indices': (('nnz',), M.indices.astype(np.int64)), 'indptr': (('sample_ptr',), M.indptr.astype(np.int64)), 'inbreeding': (('sample',), self.inbreeding.reinde...``; see the summary and the called implementation for its labeled/numerical representation.
        """
        try: import xarray as xr
        except ImportError as exc: raise ImportError("xarray is required for this view") from exc
        M=self.matrix.tocsr()
        return xr.Dataset({"data":(("nnz",),M.data),"indices":(("nnz",),M.indices.astype(np.int64)),
                           "indptr":(("sample_ptr",),M.indptr.astype(np.int64)),
                           "inbreeding":(("sample",),self.inbreeding.reindex(self.ids).to_numpy(float))},
                          coords={"sample":(("sample",),np.asarray(self.ids,dtype=str))},
                          attrs={"domino_sparse_format":"csr-v1","matrix_name":"Ainv","shape":M.shape})


def pedigree_relationship_inverse(pedigree: pd.DataFrame, *, parent1_col='dam', parent2_col='sire', offspring_col='rfid',
                                  parent_inbreeding=None, inbreeding_cache_pairs=200_000,
                                  return_container=False) -> sparse.csr_matrix | PedigreePrecision:
    """Sparse Henderson A^-1 with bounded-memory parental inbreeding.

    If ``parent_inbreeding`` is omitted, VIGOR computes exact pedigree
    inbreeding with a bounded pairwise-kinship cache rather than materializing
    the dense numerator relationship matrix.  The sparse precision path thus
    no longer has an implicit O(n^2) memory allocation.

    Parameters
    ----------
    pedigree : pd.DataFrame
        Pedigree table with offspring and parental identifiers. Parent-before-child/topological
        ordering and unknown-parent markers follow the selected pedigree algorithm; cycles and
        duplicate offspring may be rejected.

    parent1_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'dam'``.

    parent2_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'sire'``.

    offspring_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'rfid'``.

    parent_inbreeding : array-like or numeric scalar
        Input for the operation ``pd.Series(parent_inbreeding, index=ids, dtype=float)``. See
        ``vigor.scalable.pedigree_relationship_inverse`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    inbreeding_cache_pairs : int
        Input supplied to ``pedigree_inbreeding_sparse`` as ``max_pair_cache``. See that
        callable's documented ``max_pair_cache`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``200000``. Passed to
        ``pedigree_inbreeding_sparse`` as ``max_pair_cache``.

    return_container : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``PedigreePrecision(M, ids, F) if return_container else M``; see the summary and the called implementation for its labeled/numerical representation.
    """
    ped=pedigree[[offspring_col,parent1_col,parent2_col]].copy();ids=ped[offspring_col].astype(str).tolist();idx={x:i for i,x in enumerate(ids)}
    if len(idx)!=len(ids):raise ValueError('offspring IDs must be unique')
    if parent_inbreeding is None:
        F=pedigree_inbreeding_sparse(ped,parent1_col,parent2_col,offspring_col,max_pair_cache=inbreeding_cache_pairs)
    else:
        F=pd.Series(parent_inbreeding,index=ids,dtype=float) if not isinstance(parent_inbreeding,pd.Series) else parent_inbreeding.reindex(ids).astype(float)
        if F.isna().any():raise ValueError('parent_inbreeding is missing pedigree IDs')
    rows=[];cols=[];vals=[]
    def parent(v):
        """Resolve a pedigree parent index using the configured unknown-parent markers.

        Parameters
        ----------
        v : array-like
            Numerical operand in ``pd.isna(v)``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state. Source-recognized literal values: ``0``, ``None``,
            ``nan``. These select the corresponding branches below; comparison evidence alone does
            not establish an exhaustive enum.

        Returns
        -------
        object
            Computed result from ``idx.get(str(v))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if pd.isna(v) or str(v).strip() in {'','0','nan','None'}:return None
        return idx.get(str(v))
    for i,row in enumerate(ped.itertuples(index=False,name=None)):
        _,a,b=row;pa,pb=parent(a),parent(b)
        if pa is None and pb is None:d=1.0
        elif pa is None or pb is None:
            p0=pa if pa is not None else pb;d=0.75-0.25*float(F.iloc[p0])
        else:d=0.5-0.25*(float(F.iloc[pa])+float(F.iloc[pb]))
        if d<=0:raise ValueError(f'non-positive Mendelian sampling variance for {ids[i]}')
        inv=1.0/d
        coeff={i:1.0}
        if pa is not None:coeff[pa]=coeff.get(pa,0)-0.5
        if pb is not None:coeff[pb]=coeff.get(pb,0)-0.5
        for u,cu in coeff.items():
            for v,cv in coeff.items():rows.append(u);cols.append(v);vals.append(inv*cu*cv)
    M=sparse.coo_matrix((vals,(rows,cols)),shape=(len(ids),len(ids))).tocsr();M.sum_duplicates()
    return PedigreePrecision(M,ids,F) if return_container else M


def candidate_pair_grm(genotypes, *, threshold=0.05, sketch_size=512, random_state=0):
    """Sparse candidate relatedness screen followed by exact standardized dot products.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``0.05``.

    sketch_size : int
        Input for the operation ``min(int(sketch_size), Z.shape[1])``. See
        ``vigor.scalable.candidate_pair_grm`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``512``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``sparse.coo_matrix((vals, (rows, cols)), shape=(len(X), len(X))).tocsr()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X=np.asarray(genotypes,float);mu=np.nanmean(X,0);sd=np.nanstd(X,0);ok=sd>1e-8;Z=(X[:,ok]-mu[ok])/sd[ok];Z[~np.isfinite(Z)]=0
    rng=np.random.default_rng(random_state);q=min(int(sketch_size),Z.shape[1]);sel=rng.choice(Z.shape[1],q,replace=False);S=Z[:,sel]@Z[:,sel].T/q
    ii,jj=np.where(np.triu(np.abs(S)>=float(threshold),1));rows=list(ii)+list(jj)+list(range(len(X)));cols=list(jj)+list(ii)+list(range(len(X)))
    vals=[]
    for i,j in zip(ii,jj):vals.append(float(Z[i]@Z[j]/Z.shape[1]))
    vals=vals+vals+[float(Z[i]@Z[i]/Z.shape[1]) for i in range(len(X))]
    return sparse.coo_matrix((vals,(rows,cols)),shape=(len(X),len(X))).tocsr()


def solve_multitrait_pcg(operator, rhs, *, tol=1e-6, maxiter=500, preconditioner=None):
    """Solve multitrait preconditioned conjugate-gradient.

    Parameters
    ----------
    operator : kernel/linear operator or mapping
        Prepared matrix operator exposing the multiplication methods used below; shape and
        sample order must match the right-hand side or phenotype matrix.

    rhs : array-like or None
        Right-hand side of the prepared linear system. Its length must match the operator's
        observed-cell dimension; None uses the response stored in the solver plan when
        supported.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-06``. Passed to ``cg`` as ``rtol``.

    maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``500``. Passed to ``cg`` as ``maxiter``.

    preconditioner : object (see the cited input contract)
        Input supplied to ``cg`` as ``M``. See that callable's documented ``M`` contract for its
        type, alignment and value domain. This wrapper retains that contract. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    Returns
    -------
    object
        Computed result from ``(out[:, 0] if one else out, info)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    B=np.asarray(rhs,float);one=B.ndim==1
    if one:B=B[:,None]
    out=np.empty_like(B);info=[]
    for j in range(B.shape[1]):
        x,code=cg(operator,B[:,j],rtol=float(tol),atol=0,maxiter=int(maxiter),M=preconditioner);out[:,j]=x;info.append(code)
    return (out[:,0] if one else out),info

__all__=['StreamingGRMOperator','LowRankGRMOperator','MultiTraitCovarianceOperator','PedigreePrecision','candidate_pair_grm','pedigree_relationship_inverse','solve_multitrait_pcg','dense_grm_bytes','matrix_free_working_bytes']
