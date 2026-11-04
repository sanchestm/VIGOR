"""Genomic kernel experts for VIGOR mvBLUP unseen-animal prediction.

Architecture
------------
1. Keep whole-genome/chromosome mvBLUP as the stable baseline.
2. Screen global/chromosome kernels in the mvBLUP dual genetic subspace.
3. Run compact multivariate ``loco_mv_score_fast`` discovery through
   :mod:`mvgwas_fast` (no genome-wide univariate f_regression).
4. Every significant lead seeds a +/-1 Mb region; overlapping loci are merged
   with a recombination guard.
5. Selected loci use fast banded R2 and randomized SVD of centered ``1-R2`` to
   form local animal coordinates, plus LD-pruned high-score variants and an
   optional founder-haplotype feature callback.
6. Fit a low-rank genetic-factor grouped ridge correction; optionally compare
   ridge/elastic-net/LightGBM/XGBoost factor experts by inner OOF gain and
   compute SHAP only for the accepted final model.

All phenotype-dependent selection must be rerun inside each outer training
fold when benchmarking unseen animals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import math
import hashlib
import numpy as np
from scipy import linalg

__all__ = [
    "TraitFactorBasis", "KernelFeatureState", "GroupRidgeModel",
    "PrefilteredFactorModel", "GenomicKernelExpert", "GenomicKernelExpertScanFit",
    "trait_factor_basis", "dual_factor_signal", "landmark_nystrom_features",
    "spectral_feature_groups", "feature_subspace_score", "screen_kernel_bank",
    "fit_lowrank_group_ridge", "ld_distance_svd_features", "ld_components_from_r2", "ld_components_from_edges",
    "ld_group_svd_feature_bank", "ld_group_svd_feature_bank_packed", "ld_pruned_score_variants", "ld_pruned_score_variants_edges", "fit_prefiltered_factor_models",
    "fit_genomic_kernel_expert", "fit_genomic_kernel_expert_xarray",
    "estimate_large_scale", "crossfit_genomic_kernel_expert_xarray",
]


def _to_numpy(x: Any, dtype=None) -> np.ndarray:
    """Convert a table/array to a NumPy array with the requested element dtype.

    Parameters
    ----------
    x : Any
        Numerical operand in ``x.to_numpy()``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=dtype)


def _sym(a):
    """Return the arithmetic symmetrization (M + M.T)/2.

    Parameters
    ----------
    a : array-like
        Numerical operand in ``0.5 * (a + a.T)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.
    """
    a=np.asarray(a,float); return .5*(a+a.T)


def _mask(mask, n):
    """Resolve a sample selection mask and validate its length against the sample count.

    Parameters
    ----------
    mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.
    """
    if mask is None: return np.ones(n,bool)
    m=np.asarray(mask,bool)
    if m.shape!=(n,): raise ValueError("mask has wrong shape")
    return m


def _total_trait_cov(G: Any) -> np.ndarray:
    """Sum named genetic trait covariance components or normalize a single covariance.

    Parameters
    ----------
    G : Any
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.
    """
    if hasattr(G, "data_vars"):
        from .multivariate import _result_view
        G = _result_view(G)
    if isinstance(G, Mapping):
        if "G_components" in G:
            return _total_trait_cov(G["G_components"])
        mats=[]
        for v in G.values():
            try:
                a=_to_numpy(v,np.float64)
                if a.ndim==2 and a.shape[0]==a.shape[1]: mats.append(a)
            except Exception: pass
        if not mats: raise ValueError("G mapping contains no square covariance matrices")
        out=np.zeros_like(mats[0],float)
        for a in mats: out+=a
        return _sym(out)
    return _sym(_to_numpy(G,np.float64))


def _mean_trait_r2(Y,P,mask=None):
    """Average per-trait held-out predictive R-squared over supported observed cells.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value.
    """
    Y=_to_numpy(Y,float);P=_to_numpy(P,float);m=np.ones(len(Y),bool) if mask is None else np.asarray(mask,bool);vals=[]
    for j in range(Y.shape[1]):
        ok=m&np.isfinite(Y[:,j])&np.isfinite(P[:,j])
        if ok.sum()<3: continue
        y=Y[ok,j];p=P[ok,j];den=np.sum((y-y.mean())**2)
        if den>0: vals.append(1-np.sum((y-p)**2)/den)
    return float(np.mean(vals)) if vals else np.nan


@dataclass(frozen=True)
class TraitFactorBasis:
    """TraitFactorBasis represents trait factor basis.

    Attributes
    ----------
    vectors : np.ndarray
        Stored vectors; its axis/order follows the construction inputs.
    eigenvalues : np.ndarray
        Stored eigenvalues; its axis/order follows the construction inputs.
    rank : int
        Stored rank; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    vectors : np.ndarray
        Ordered retained eigenvectors/basis directions, with one column per factor. Required
        construction field.

    eigenvalues : np.ndarray
        Nonnegative retained eigenvalues in the same order as vector columns. Required
        construction field.

    rank : int
        Number of retained basis/factor columns. Required construction field.
    """
    vectors: np.ndarray
    eigenvalues: np.ndarray
    rank: int


def trait_factor_basis(G: Any, rank: int=10, eps: float=1e-10) -> TraitFactorBasis:
    """Return the trait factor basis result computed by ``TraitFactorBasis``.

    Parameters
    ----------
    G : Any
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    rank : int
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``10``.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-10``.

    Returns
    -------
    object
        Computed result from ``TraitFactorBasis(V[:, :r], w[:r], r)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    G=_total_trait_cov(G);w,V=np.linalg.eigh(G);o=np.argsort(w)[::-1];w=np.clip(w[o],0,None);V=V[:,o]
    if len(w) and w[0]>0:
        keep=w>w[0]*float(eps);w=w[keep];V=V[:,keep]
    r=max(1,min(int(rank),V.shape[1]));return TraitFactorBasis(V[:,:r],w[:r],r)


def dual_factor_signal(dual_A: Any,G: Any,*,rank:int=10,train_mask:Optional[np.ndarray]=None):
    """Project the BLUP dual state onto retained genetic trait factors.

    Parameters
    ----------
    dual_A : Any
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    G : Any
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    rank : int
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``10``.

    train_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``(D, basis)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    A=_to_numpy(dual_A,np.float64);basis=trait_factor_basis(G,rank);D=A@basis.vectors;fit=_mask(train_mask,len(A));D=D.copy();D[~fit]=0
    s=np.sqrt(np.mean(D[fit]*D[fit],axis=0));s[s<1e-12]=1;D/=s[None,:]
    return D,basis


@dataclass(frozen=True)
class KernelFeatureState:
    """KernelFeatureState represents kernel feature state.

    Attributes
    ----------
    landmarks : np.ndarray
        Stored landmarks; its axis/order follows the construction inputs.
    landmark_vectors : np.ndarray
        Stored landmark vectors; its axis/order follows the construction inputs.
    landmark_values : np.ndarray
        Stored landmark values; its axis/order follows the construction inputs.
    name : str
        Stored name; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    landmarks : np.ndarray
        Stored landmarks supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    landmark_vectors : np.ndarray
        Stored landmark vectors supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    landmark_values : np.ndarray
        Stored landmark values supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    name : str
        Stored name supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.
    """
    landmarks: np.ndarray
    landmark_vectors: np.ndarray
    landmark_values: np.ndarray
    name: str


def _kernel_slice(K,rows,cols):
    """Retrieve an aligned row/column submatrix from a supported animal kernel.

    Parameters
    ----------
    K : array-like
        Numerical operand in ``list(K.dims)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    rows : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering.

    cols : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.
    """
    if hasattr(K,"isel"):
        dims=list(K.dims);return np.asarray(K.isel({dims[0]:rows,dims[1]:cols}).values,float)
    A=np.asarray(K);return np.asarray(A[np.ix_(rows,cols)],float)


def _kernel_cols(K,cols):
    """Retrieve selected animal-kernel columns without requiring a full kernel copy.

    Parameters
    ----------
    K : array-like
        Numerical operand in ``np.asarray(K)[:, cols]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    cols : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.
    """
    if hasattr(K,"isel"):
        dims=list(K.dims);return np.asarray(K.isel({dims[1]:cols}).values,float)
    return np.asarray(K)[:,cols]


def landmark_nystrom_features(K:Any,*,train_mask:np.ndarray,rank:int=32,n_landmarks:int=192,random_state:int=0,name:str="kernel"):
    """Construct a landmark-based low-rank approximation to an animal kernel.

    Parameters
    ----------
    K : Any
        Numerical operand in ``_kernel_cols(K, L) @ (V / np.sqrt(w)[None, :])``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    train_mask : np.ndarray
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    rank : int
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``32``.

    n_landmarks : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``192``.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'kernel'``.

    Returns
    -------
    object
        Computed result from ``(np.asarray(F, np.float32), KernelFeatureState(L, V, w, str(name)))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    n=int(K.shape[0]);fit=np.flatnonzero(_mask(train_mask,n));rng=np.random.default_rng(random_state);q=min(int(n_landmarks),len(fit));L=np.sort(rng.choice(fit,size=q,replace=False))
    W=_sym(_kernel_slice(K,L,L));w,V=np.linalg.eigh(W);o=np.argsort(w)[::-1];r=min(max(1,int(rank)),q);w=w[o][:r];V=V[:,o][:,:r]
    keep=w>max(1e-10,(w[0] if len(w) else 0)*1e-10);w=w[keep];V=V[:,keep]
    F=_kernel_cols(K,L)@(V/np.sqrt(w)[None,:]) if len(w) else np.zeros((n,0))
    return np.asarray(F,np.float32),KernelFeatureState(L,V,w,str(name))


def spectral_feature_groups(F:np.ndarray,eigenvalues:np.ndarray,*,n_bands:int=6,prefix:str="spectral"):
    """Convert retained kernel eigenvectors into named feature groups.

    Parameters
    ----------
    F : np.ndarray
        Numerical operand in ``F.shape[1]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    eigenvalues : np.ndarray
        Eigenvalues associated with the columns of the retained eigenvector basis. Length must
        equal the basis rank; covariance uses nonnegative eigenvalues.

    n_bands : int
        Input for the operation ``np.array_split(np.arange(r), min(max(1, int(n_bands)), max(1,
        r)))``. See ``vigor.genomic_kernel_expert.spectral_feature_groups`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.
        Default: ``6``.

    prefix : str
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants. Default: ``'spectral'``.

    Returns
    -------
    object
        Computed result from ``{f'{prefix}::{i + 1}': F[:, ix] for i, ix in enumerate(parts) if len(ix)}``; see the summary and the called implementation for its labeled/numerical representation.
    """
    r=F.shape[1];parts=np.array_split(np.arange(r),min(max(1,int(n_bands)),max(1,r)));return {f"{prefix}::{i+1}":F[:,ix] for i,ix in enumerate(parts) if len(ix)}


def feature_subspace_score(F:Any,D:Any,*,train_mask:np.ndarray):
    """Measure how strongly a feature subspace explains the training factor signal.

    Parameters
    ----------
    F : Any
        Numerical operand in ``F[fit] - F[fit].mean(0)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    D : Any
        Input for the operation ``D[fit] ** 2``. See
        ``vigor.genomic_kernel_expert.feature_subspace_score`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    train_mask : np.ndarray
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    Returns
    -------
    dict
        Result fields: ``score``, ``captured``, ``rank``. Field arrays preserve the corresponding input axis order.
    """
    F=_to_numpy(F,float);D=_to_numpy(D,float);fit=np.asarray(train_mask,bool);X=F[fit]-F[fit].mean(0);Q,R=np.linalg.qr(X,mode="reduced")
    if R.size:
        d=np.abs(np.diag(R));Q=Q[:,d>max(1e-10,d.max()*1e-9)]
    den=np.sum(D[fit]**2);capt=float(np.sum((Q.T@D[fit])**2)/max(den,1e-30)) if Q.size else 0.;null=Q.shape[1]/max(fit.sum(),1);score=(capt-null)/max(1-null,1e-12)
    return {"score":float(score),"captured":capt,"rank":int(Q.shape[1])}


def screen_kernel_bank(kernels:Mapping[str,Any],D:np.ndarray,*,train_mask:np.ndarray,rank:int=10,n_landmarks:int=192,top_k:Optional[int]=6,random_state:int=0):
    """Rank candidate animal kernels by their alignment with training genetic signal.

    Parameters
    ----------
    kernels : Mapping[str, Any]
        Named animal relationship kernels; each matrix/operator must use the same animal order.
        Names identify the matching trait covariance components.

    D : np.ndarray
        Input for the operation ``feature_subspace_score(F, D, train_mask=train_mask)``. See
        ``vigor.genomic_kernel_expert.screen_kernel_bank`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    train_mask : np.ndarray
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Passed to ``landmark_nystrom_features`` as
        ``train_mask``.

    rank : int
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``10``. Passed to
        ``landmark_nystrom_features`` as ``rank``.

    n_landmarks : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``192``.
        Passed to ``landmark_nystrom_features`` as ``n_landmarks``.

    top_k : Optional[int]
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``6``.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``landmark_nystrom_features`` as ``random_state``.

    Returns
    -------
    object
        Computed result from ``([(a, b, c, d) for a, b, c, d, _ in sel], bank)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rows=[];bank={}
    for i,(name,K) in enumerate(kernels.items()):
        F,_=landmark_nystrom_features(K,train_mask=train_mask,rank=rank,n_landmarks=n_landmarks,random_state=random_state+i,name=str(name));sc=feature_subspace_score(F,D,train_mask=train_mask);rows.append((str(name),sc["score"],sc["captured"],sc["rank"],F))
    rows.sort(key=lambda z:z[1],reverse=True);sel=rows if top_k is None else rows[:int(top_k)]
    for z in sel: bank[z[0]]=z[4]
    return [(a,b,c,d) for a,b,c,d,_ in sel],bank


@dataclass(frozen=True)
class _GroupTransform:
    """_GroupTransform represents group transform.

    Attributes
    ----------
    mean : np.ndarray
        Stored mean; its axis/order follows the construction inputs.
    whitener : np.ndarray
        Stored whitener; its axis/order follows the construction inputs.
    out_slice : slice
        Stored out slice; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    mean : np.ndarray
        Stored mean supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    whitener : np.ndarray
        Stored whitener supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    out_slice : slice
        Stored out slice supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.
    """
    mean: np.ndarray
    whitener: np.ndarray
    out_slice: slice


def _apply_group_transform(features, state):
    """Apply group transform.

    Parameters
    ----------
    features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    F=np.asarray(features,np.float64)
    return (np.where(np.isfinite(F),F,state.mean)-state.mean)@state.whitener


def _whiten_groups(feature_bank:Mapping[str,np.ndarray],fit:np.ndarray):
    """Fit training feature-group centers and whitening transforms for grouped regression.

    Parameters
    ----------
    feature_bank : Mapping[str, np.ndarray]
        Prepared feature/annotation collection indexed by its named groups. Reuse the same
        fitted preprocessing, animal/variant order and metadata when transforming or predicting.

    fit : np.ndarray
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    pieces=[];states={};start=0
    for name,F0 in feature_bank.items():
        F=np.asarray(F0,np.float64)
        train=F[fit];observed=np.isfinite(train)
        mu=np.divide(np.where(observed,train,0.).sum(0),observed.sum(0),
                     out=np.zeros(F.shape[1]),where=observed.sum(0)>0)
        C=np.where(np.isfinite(F),F,mu)-mu
        S=_sym((C[fit].T@C[fit])/max(1,fit.sum()-1))
        w,V=np.linalg.eigh(S);o=np.argsort(w)[::-1];w=w[o];V=V[:,o]
        keep=w>max(1e-8,(w[0] if len(w) else 0)*1e-7)
        W=V[:,keep]/np.sqrt(w[keep])[None,:]
        Z=C@W;states[name]=_GroupTransform(mu,W,slice(start,start+Z.shape[1]))
        start+=Z.shape[1];pieces.append(Z)
    return (np.concatenate(pieces,1) if pieces else np.zeros((len(fit),0))),states


def _residual_factor_targets(Y,B,V,train_mask):
    """Project baseline prediction residuals into the genetic trait factor space.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    B : array-like
        Numerical operand in ``np.asarray(Y, float) - np.asarray(B, float)``. Shape must conform
        to that matrix/vector operation and the enclosing prepared state.

    V : array-like
        Numerical operand in ``V.shape[1]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.
    """
    R=np.asarray(Y,float)-np.asarray(B,float);n=len(R);r=V.shape[1];F=np.full((n,r),np.nan,float);valid=np.zeros(n,bool)
    for i in np.flatnonzero(train_mask):
        obs=np.isfinite(R[i])
        if obs.sum()<1: continue
        Vo=V[obs];ro=R[i,obs]
        A=Vo.T@Vo+1e-6*np.eye(r);b=Vo.T@ro
        try:F[i]=linalg.solve(A,b,assume_a="pos",check_finite=False);valid[i]=True
        except Exception: pass
    return F,valid


def _ridge_solve(X,Y,alpha,penalty=None):
    """Solve the ridge-penalized normal equations for multiple response columns.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.

    penalty : float or array-like
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    p=X.shape[1]
    if p==0:return np.zeros((0,Y.shape[1]))
    pen=np.ones(p) if penalty is None else np.asarray(penalty,float);A=(X.T@X)/max(len(X),1);A.flat[::p+1]+=float(alpha)*pen;B=(X.T@Y)/max(len(X),1)
    try:return linalg.solve(A,B,assume_a="pos",check_finite=False)
    except Exception:return np.linalg.pinv(A,rcond=1e-10)@B


@dataclass
class GroupRidgeModel:
    """GroupRidgeModel represents group ridge model.

    Attributes
    ----------
    trait_vectors : np.ndarray
        Stored trait vectors; its axis/order follows the construction inputs.
    beta : np.ndarray
        Stored beta; its axis/order follows the construction inputs.
    transforms : Dict[str, _GroupTransform]
        Stored transforms; its axis/order follows the construction inputs.
    groups : Tuple[str, ...]
        Stored groups; its axis/order follows the construction inputs.
    alpha : float
        Stored alpha; its axis/order follows the construction inputs.
    accepted : bool
        Stored accepted; its axis/order follows the construction inputs.
    cv_gain : float
        Stored cv gain; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    trait_vectors : np.ndarray
        Stored trait vectors consumed by ``correction``; see those methods for the
        numerical/label contract. Required construction field.

    beta : np.ndarray
        Stored beta consumed by ``correction``; see those methods for the numerical/label
        contract. Required construction field.

    transforms : Dict[str, _GroupTransform]
        Stored transforms consumed by ``transform``; see those methods for the numerical/label
        contract. Required construction field.

    groups : Tuple[str, ...]
        Stored groups consumed by ``transform``; see those methods for the numerical/label
        contract. Required construction field.

    alpha : float
        Stored alpha supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    accepted : bool
        Stored accepted supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``True``.

    cv_gain : float
        Stored cv gain supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``0.0``.
    """
    trait_vectors: np.ndarray
    beta: np.ndarray
    transforms: Dict[str,_GroupTransform]
    groups: Tuple[str,...]
    alpha: float
    accepted: bool = True
    cv_gain: float = 0.0

    def transform(self,feature_bank):
        """Apply the stored encoding/standardization or model transform to aligned inputs.

        Parameters
        ----------
        feature_bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.

        Returns
        -------
        object
            Computed result from ``np.concatenate(cols, 1) if cols else np.zeros((len(next(iter(feature_bank.values()))), 0))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        cols=[]
        for name in self.groups:
            F=np.asarray(feature_bank[name],float);st=self.transforms[name];cols.append(_apply_group_transform(F,st))
        return np.concatenate(cols,1) if cols else np.zeros((len(next(iter(feature_bank.values()))),0))
    def correction(self,feature_bank):
        """Compute the fitted genomic expert's residual correction in trait space.

        Parameters
        ----------
        feature_bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.

        Returns
        -------
        object
            Computed result from ``self.transform(feature_bank) @ self.beta @ self.trait_vectors.T``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.transform(feature_bank)@self.beta)@self.trait_vectors.T


def fit_lowrank_group_ridge(Y,baseline,feature_bank,G,*,train_mask:np.ndarray,trait_rank:int=10,alphas=(1e-3,1e-2,1e-1,1.,10.),cv:int=3,ard_iter:int=3,min_cv_gain:float=0.0,random_state:int=0):
    """Grouped ridge on genetic factors, gated by raw-trait OOF improvement.

    Hyperparameters are chosen on held-out training animals.  The correction is
    accepted only when it reduces squared error of observed trait residuals
    relative to the original BLUP baseline; otherwise coefficients are zero.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    feature_bank : mapping or feature/annotation bundle
        Prepared feature/annotation collection indexed by its named groups. Reuse the same
        fitted preprocessing, animal/variant order and metadata when transforming or predicting.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    train_mask : np.ndarray
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    trait_rank : int
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``.

    alphas : float, int or sequence
        Regularization search grid: l1_ratio mixes L1 versus L2 penalties on [0, 1]; alphas
        supplies candidate strengths or the number of strengths generated by the called CV
        estimator. See that estimator for automatic grid semantics. Default: ``(0.001, 0.01,
        0.1, 1.0, 10.0)``.

    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``.

    ard_iter : int
        Number of alternating/projection/relevance/eigenspace refinement iterations performed by
        the selected algorithm. Positive counts trade additional work for a more settled
        approximation. Default: ``3``.

    min_cv_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.0``.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``model``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Y=_to_numpy(Y,float);B=_to_numpy(baseline,float);n,T=Y.shape;fit0=_mask(train_mask,n);basis=trait_factor_basis(G,min(int(trait_rank),T));target,valid=_residual_factor_targets(Y,B,basis.vectors,fit0);fit=fit0&valid;X,states=_whiten_groups(feature_bank,fit);ids=np.flatnonzero(fit)
    if len(ids)<20:raise ValueError("too few phenotype-informative training animals")
    Rraw=Y-B
    rng=np.random.default_rng(random_state);folds=np.array_split(rng.permutation(ids),max(2,min(int(cv),len(ids)//30 if len(ids)>=60 else 2)))
    base_sse=0.;scores={};best_a=float(alphas[0]);best_sse=np.inf
    for va in folds:
        rr=Rraw[va];obs=np.isfinite(rr);base_sse+=float(np.sum(rr[obs]**2))
    # Fit imputation/whitening independently in each inner training fold.
    # The final all-training transform above is reserved for the deployment fit.
    cv_design = []
    for va in folds:
        tr = np.setdiff1d(ids, va, assume_unique=False)
        fold_mask = np.zeros(n, bool); fold_mask[tr] = True
        Xfold, _ = _whiten_groups(feature_bank, fold_mask)
        cv_design.append((tr, va, Xfold))
    for a0 in alphas:
        sse=0.
        for tr, va, Xfold in cv_design:
            coef0=_ridge_solve(Xfold[tr],target[tr],a0);C=(Xfold[va]@coef0)@basis.vectors.T;rr=Rraw[va];obs=np.isfinite(rr);sse+=float(np.sum((rr[obs]-C[obs])**2))
        gain=float(1-sse/max(base_sse,1e-30));scores[float(a0)]=gain
        if sse<best_sse:best_sse=float(sse);best_a=float(a0)
    cv_gain=float(1-best_sse/max(base_sse,1e-30));accepted=bool(np.isfinite(cv_gain) and cv_gain>=float(min_cv_gain))
    pen=np.ones(X.shape[1]);coef=_ridge_solve(X[fit],target[fit],best_a,pen) if accepted else np.zeros((X.shape[1],basis.rank),float)
    if accepted:
        for _ in range(max(0,int(ard_iter))):
            energy={name:float(np.sqrt(np.mean(coef[st.out_slice]**2))) if coef[st.out_slice].size else 0. for name,st in states.items()};pos=np.asarray([v for v in energy.values() if v>0]);floor=(np.median(pos) if len(pos) else 1.)*.05+1e-12
            for name,st in states.items():pen[st.out_slice]=1/(energy[name]+floor)
            med=np.median(pen[np.isfinite(pen)]);pen/=med if med>0 else 1.;coef=_ridge_solve(X[fit],target[fit],best_a,pen)
    model=GroupRidgeModel(basis.vectors,coef,states,tuple(feature_bank.keys()),best_a,accepted,cv_gain)
    model.cv_scores_=scores
    return model


def _mvgwas_fast_module():
    """Import the accelerated multivariate association implementation used for discovery.
        
    """
    try:
        from . import mvgwas_fast as mf
    except Exception:
        import mvgwas_fast as mf
    return mf


def _standardize_train_genotypes(X,train_mask):
    """Center and scale genotype features using training animals only.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Passed to ``np.divide`` as ``out``.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.
    """
    X=np.asarray(X,np.float32);fit=np.asarray(train_mask,bool);finite=np.isfinite(X);cnt=finite[fit].sum(0);mu=np.divide(np.nansum(X[fit],0,dtype=np.float64),cnt,out=np.zeros(X.shape[1]),where=cnt>0).astype(np.float32);Xi=np.where(finite,X,mu[None,:]);sd=Xi[fit].std(0);sd[sd<1e-6]=1.;return (Xi-mu[None,:])/sd[None,:],mu,sd


def ld_components_from_r2(r2, *, threshold=.20):
    """Connected components of the thresholded local LD graph.

    Parameters
    ----------
    r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``0.2``.

    Returns
    -------
    object
        Computed result from ``np.asarray(lab, np.int32)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    R=np.asarray(r2,np.float32);A=(R>=float(threshold)).astype(np.int8,copy=False);np.fill_diagonal(A,1)
    _,lab=connected_components(csr_matrix(A),directed=False,return_labels=True)
    return np.asarray(lab,np.int32)


def ld_components_from_edges(n_variants, edge_i, edge_j):
    """Connected components from a thresholded sparse LD edge list.

    Parameters
    ----------
    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    edge_i : array-like or numeric scalar
        Input for the operation ``np.asarray(edge_i, np.int64)``. See
        ``vigor.genomic_kernel_expert.ld_components_from_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    edge_j : array-like or numeric scalar
        Input for the operation ``np.asarray(edge_j, np.int64)``. See
        ``vigor.genomic_kernel_expert.ld_components_from_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    Returns
    -------
    object
        Computed result from ``np.asarray(lab, np.int32)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m=int(n_variants);i=np.asarray(edge_i,np.int64);j=np.asarray(edge_j,np.int64)
    if not len(i):return np.arange(m,dtype=np.int32)
    data=np.ones(len(i)*2,np.int8);rows=np.r_[i,j];cols=np.r_[j,i]
    A=coo_matrix((data,(rows,cols)),shape=(m,m)).tocsr();_,lab=connected_components(A,directed=False,return_labels=True)
    return np.asarray(lab,np.int32)


def _distance_svd_one_group(Z,r2,*,n_components,fit,random_state):
    """Project only defined, variable LD geometry; never turn NaN LD into0.

    Monomorphic/undefined diagonal variants get zero loading (excluded from
    geometry). A group with remaining unsupported pair distances produces no
    LD coordinates. Raw genotype experts can still use train-imputed calls.

    Parameters
    ----------
    Z : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.

    r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.

    fit : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Passed to
        ``randomized_svd`` as ``random_state``.
    """
    from sklearn.utils.extmath import randomized_svd
    Z=np.asarray(Z,np.float32);R=np.asarray(r2,np.float32);m=Z.shape[1]
    valid=np.isfinite(np.diag(R)) & (np.std(Z[fit],axis=0)>1e-7)
    ix=np.flatnonzero(valid);RR=R[np.ix_(ix,ix)];ZZ=Z[:,ix]
    if not len(ix) or not np.isfinite(RR).all():
        return np.zeros((len(Z),0),np.float32),np.zeros(0,np.float32),np.zeros((m,0),np.float32)
    if len(ix)==1:F=ZZ.copy();s=np.ones(1,np.float32);smallV=np.ones((1,1),np.float32)
    else:
        D=1.-RR;np.fill_diagonal(D,0.)
        Dc=D-D.mean(1,keepdims=True)-D.mean(0,keepdims=True)+D.mean()
        k=max(1,min(int(n_components),len(ix)-1));_,s,Vt=randomized_svd(Dc,n_components=k,n_iter=3,random_state=random_state,flip_sign=True)
        smallV=Vt.T.astype(np.float32,copy=False);F=(ZZ@smallV)/np.sqrt(len(ix))
    fm=F[fit].mean(0);fs=F[fit].std(0);fs[fs<1e-7]=1.;F=(F-fm)/fs
    V=np.zeros((m,smallV.shape[1]),np.float32);V[ix]=smallV
    return np.asarray(F,np.float32),np.asarray(s,np.float32),V


def ld_group_svd_feature_bank(
    X,positions,*,train_mask,scores=None,pvalues=None,genetic_cm=None,n_components=6,
    ld_component_r2=.20,max_groups=8,max_distance_bp=1_000_000,max_distance_cm=None,
    ld_sample_rows=2048,ld_block_variants=256,random_state=0):
    """Recombination/LD-aware local features using a sparse-first LD pass.

    Pass 1 builds only thresholded rolling-R2 edges and connected components;
    no region-wide m x m R2 matrix is created.  Components are selected by a
    mixture of statistical signal and genomic coverage/size (not p-value
    alone).  Pass 2 computes dense banded R2 only inside the retained LD
    components and performs SVD of centered ``1-R2`` to obtain animal-level
    local-kernel coordinates.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    scores : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    pvalues : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    genetic_cm : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.
        Passed to ``mf.fast_windowed_r2_edges`` as ``genetic_cm``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``6``. Passed to ``_distance_svd_one_group`` as ``n_components``.

    ld_component_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.2``. Passed to ``mf.fast_windowed_r2_edges`` as ``min_r2``.

    max_groups : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``8``.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``mf.fast_windowed_r2_edges`` as ``max_distance_bp``.

    max_distance_cm : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``. Passed to ``mf.fast_windowed_r2_edges`` as ``max_distance_cm``.

    ld_sample_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``mf.fast_windowed_r2_edges`` as
        ``sample_rows``.

    ld_block_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``256``. Passed to ``mf.fast_windowed_r2_edges``
        as ``block_variants``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``mf.fast_windowed_r2_edges`` as ``random_state``.

    Returns
    -------
    object
        Computed result from ``(bank, {'edges': edges, 'labels': lab, 'groups': meta, 'component_r2': dense_r2, 'genotype_mean': mu, 'genotype_scale': sd})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    mf=_mvgwas_fast_module();X=_to_numpy(X,np.float32);pos=np.asarray(positions,np.int64);fit=np.asarray(train_mask,bool)
    Z,mu,sd=_standardize_train_genotypes(X,fit)
    edges=mf.fast_windowed_r2_edges(
        X,positions=pos,genetic_cm=genetic_cm,fit_mask=fit,
        max_distance_bp=max_distance_bp,max_distance_cm=max_distance_cm,
        sample_rows=ld_sample_rows,block_variants=ld_block_variants,
        min_r2=ld_component_r2,random_state=random_state,return_xarray=False)
    lab=ld_components_from_edges(X.shape[1],edges["i"],edges["j"])
    sc=np.zeros(X.shape[1],float) if scores is None else np.nan_to_num(np.asarray(scores,float),nan=-np.inf)
    pv=np.full(X.shape[1],np.nan) if pvalues is None else np.asarray(pvalues,float)
    rec=[]
    for g in np.unique(lab):
        ix=np.flatnonzero(lab==g);span=int(pos[ix[-1]]-pos[ix[0]]) if len(ix)>1 else 0
        rec.append({"label":int(g),"indices":ix,"n":int(len(ix)),"span_bp":span,
                    "max_score":float(np.max(sc[ix])) if len(ix) else -np.inf,
                    "min_p":float(np.nanmin(pv[ix])) if len(ix) and np.isfinite(pv[ix]).any() else np.nan})
    # Do not let p-values alone determine which local structure is represented.
    # Take signal-rich components and the largest/most spatially extensive
    # components, then fill remaining slots by a stable combined rank.
    k=max(1,min(int(max_groups),len(rec))) if rec else 0
    chosen=[]
    if k:
        sig=sorted(range(len(rec)),key=lambda q:(rec[q]["min_p"] if np.isfinite(rec[q]["min_p"]) else np.inf,-rec[q]["max_score"]))
        cov=sorted(range(len(rec)),key=lambda q:(-rec[q]["n"],-rec[q]["span_bp"],rec[q]["label"]))
        nsig=max(1,int(math.ceil(.67*k)));ncov=max(1,k-nsig)
        for q in sig[:nsig]+cov[:ncov]:
            if q not in chosen:chosen.append(q)
        if len(chosen)<k:
            sig_rank={q:r for r,q in enumerate(sig)};cov_rank={q:r for r,q in enumerate(cov)}
            rest=sorted((q for q in range(len(rec)) if q not in chosen),key=lambda q:.7*sig_rank[q]+.3*cov_rank[q])
            chosen.extend(rest[:k-len(chosen)])
    selected=[rec[q] for q in chosen[:k]]
    bank={};meta=[];dense_r2={}
    for gi,g0 in enumerate(selected):
        ix=g0["indices"]
        # Dense R2 is now local to one retained connected component only.
        r2g=mf.fast_windowed_r2(
            X[:,ix],positions=pos[ix],genetic_cm=(None if genetic_cm is None else np.asarray(genetic_cm,float)[ix]),
            fit_mask=fit,max_distance_bp=max_distance_bp,max_distance_cm=max_distance_cm,
            sample_rows=ld_sample_rows,block_variants=ld_block_variants,random_state=random_state+7919+gi)
        F,sval,V=_distance_svd_one_group(Z[:,ix],r2g,n_components=n_components,fit=fit,random_state=random_state+gi)
        name=f"ldgroup::{gi+1}";bank[name]=F;dense_r2[int(g0["label"])]=r2g
        meta.append({"name":name,"label":int(g0["label"]),"indices":ix,"n_variants":int(len(ix)),
                     "span_bp":int(g0["span_bp"]),"max_score":float(g0["max_score"]),"min_p":float(g0["min_p"]),
                     "distance_singular_values":sval,"variant_loadings":V})
    return bank,{"edges":edges,"labels":lab,"groups":meta,"component_r2":dense_r2,
                 "genotype_mean":mu,"genotype_scale":sd}



def ld_group_svd_feature_bank_packed(
    prefix, variant_indices, positions, *, sample_ids, train_mask, scores=None, pvalues=None,
    n_components=6, ld_component_r2=.20, max_groups=8, max_distance_bp=1_000_000,
    max_offset=256, ld_sample_rows=2048, random_state=0, dense_group_max_pairs=4_000_000,
):
    """PLINK1 local LD/SVD features with packed-byte LD before genotype decoding.

    Rolling LD and connected-component discovery operate directly on BED 2-bit
    calls. Only variants in retained LD groups are decoded to float32 to form
    animal-level local features. The returned edge indices are local to
    ``variant_indices``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    variant_indices : one-dimensional integer array or None
        Zero-based variant positions to select from the current source. Values must refer to
        existing columns; None retains all eligible variants. Streaming readers may additionally
        require sorted indices.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    scores : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    pvalues : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``6``. Passed to ``_distance_svd_one_group`` as ``n_components``.

    ld_component_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.2``.

    max_groups : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``8``.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``virtual_offset_r2`` as ``max_distance_bp``.

    max_offset : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``256``.

    ld_sample_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``rng.choice`` as ``size``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``_distance_svd_one_group`` as ``random_state``.

    dense_group_max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``4000000``.

    Returns
    -------
    object
        Computed result from ``(bank, {'edges': {'i': ei, 'j': ej, 'r2': ev}, 'labels': lab, 'groups': meta, 'component_r2': dense_r2, 'packed_ld': True, 'decoded_variants': int(sum((len(g['indices']) for g in selected))), 'regi...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .packed_bed import virtual_offset_r2, virtual_ld_matrix, virtual_dosage, _metadata
    idx=np.asarray(variant_indices,np.int64); pos=np.asarray(positions,np.int64)
    if len(idx)!=len(pos): raise ValueError('variant_indices and positions must have equal length')
    order=np.argsort(idx,kind='stable'); idx=idx[order]; pos=pos[order]
    if scores is not None: scores=np.asarray(scores)[order]
    if pvalues is not None: pvalues=np.asarray(pvalues)[order]
    fam,_=_metadata(prefix); all_ids=fam['iid'].astype(str).to_numpy()
    wanted=np.asarray(sample_ids).astype(str); lookup={x:i for i,x in enumerate(all_ids)}
    rows=np.asarray([lookup[x] for x in wanted],np.int64)
    fit=np.asarray(train_mask,bool)
    if fit.shape!=(len(rows),): raise ValueError('train_mask must align to sample_ids')
    train_rows=rows[fit]; rng=np.random.default_rng(random_state)
    if ld_sample_rows is not None and len(train_rows)>int(ld_sample_rows):
        train_rows=np.sort(rng.choice(train_rows,size=int(ld_sample_rows),replace=False))
    smask=np.zeros(len(all_ids),bool); smask[train_rows]=True

    # Sparse rolling LD pass on packed bytes.  The virtual array is indexed to
    # the requested region before materialization; no genome-wide pair graph exists.
    off=max(1,int(max_offset)); roll=virtual_offset_r2(prefix,max_offset=off,
        max_distance_bp=max_distance_bp,sample_mask=smask,max_pairs=max(len(idx)*(off+1),1))
    Roff=np.asarray(roll.isel(snp=idx,offset=slice(1,off+1)).values,np.float32)
    local_of={int(g):i for i,g in enumerate(idx)}; ei=[]; ej=[]; ev=[]
    for li,gi in enumerate(idx):
        vals=Roff[li]
        good=np.flatnonzero(np.isfinite(vals)&(vals>=float(ld_component_r2)))
        for q in good:
            gj=int(gi)+int(q)+1; lj=local_of.get(gj)
            if lj is not None: ei.append(li); ej.append(lj); ev.append(float(vals[q]))
    ei=np.asarray(ei,np.int64); ej=np.asarray(ej,np.int64); ev=np.asarray(ev,np.float32)
    lab=ld_components_from_edges(len(idx),ei,ej)
    sc=np.zeros(len(idx),float) if scores is None else np.nan_to_num(np.asarray(scores,float),nan=-np.inf)
    pv=np.full(len(idx),np.nan) if pvalues is None else np.asarray(pvalues,float)
    rec=[]
    for g0 in np.unique(lab):
        ix=np.flatnonzero(lab==g0); span=int(pos[ix[-1]]-pos[ix[0]]) if len(ix)>1 else 0
        rec.append({'label':int(g0),'indices':ix,'n':int(len(ix)),'span_bp':span,
                    'max_score':float(np.max(sc[ix])) if len(ix) else -np.inf,
                    'min_p':float(np.nanmin(pv[ix])) if len(ix) and np.isfinite(pv[ix]).any() else np.nan})
    k=max(1,min(int(max_groups),len(rec))) if rec else 0; chosen=[]
    if k:
        sig=sorted(range(len(rec)),key=lambda q:(rec[q]['min_p'] if np.isfinite(rec[q]['min_p']) else np.inf,-rec[q]['max_score']))
        cov=sorted(range(len(rec)),key=lambda q:(-rec[q]['n'],-rec[q]['span_bp'],rec[q]['label']))
        nsig=max(1,int(math.ceil(.67*k))); ncov=max(1,k-nsig)
        for q in sig[:nsig]+cov[:ncov]:
            if q not in chosen: chosen.append(q)
        if len(chosen)<k:
            sr={q:r for r,q in enumerate(sig)}; cr={q:r for r,q in enumerate(cov)}
            rest=sorted((q for q in range(len(rec)) if q not in chosen),key=lambda q:.7*sr[q]+.3*cr[q])
            chosen.extend(rest[:k-len(chosen)])
    selected=[rec[q] for q in chosen[:k]]

    dose=virtual_dosage(prefix); bank={}; meta=[]; dense_r2={}
    for gi,g0 in enumerate(selected):
        ix=g0['indices']; gidx=idx[ix]
        # Dense LD exists only as a selected virtual view for this retained group.
        lim=max(int(dense_group_max_pairs),int(len(ix))*int(len(ix)))
        r2g=np.asarray(virtual_ld_matrix(prefix,sample_mask=smask,max_pairs=lim)
                       .isel(snp_i=gidx,snp_j=gidx).values,np.float32)
        X=np.asarray(dose.isel(iid=rows,snp=gidx).values,np.float32)
        Z,_,_=_standardize_train_genotypes(X,fit)
        F,sval,V=_distance_svd_one_group(Z,r2g,n_components=n_components,fit=fit,random_state=random_state+gi)
        name=f'ldgroup::{gi+1}'; bank[name]=F; dense_r2[int(g0['label'])]=r2g
        meta.append({'name':name,'label':int(g0['label']),'indices':ix,'n_variants':int(len(ix)),
                     'span_bp':int(g0['span_bp']),'max_score':float(g0['max_score']),'min_p':float(g0['min_p']),
                     'distance_singular_values':sval,'variant_loadings':V,'ld_geometry_status':'defined_variable_subset' if F.shape[1] else 'unsupported_pairwise_LD','n_geometry_variants':int(np.count_nonzero(np.any(V!=0,axis=1)))})
    return bank,{'edges':{'i':ei,'j':ej,'r2':ev},'labels':lab,'groups':meta,'component_r2':dense_r2,
                 'packed_ld':True,'decoded_variants':int(sum(len(g['indices']) for g in selected)),
                 'region_variants':int(len(idx))}

def ld_distance_svd_features(X,positions,*,train_mask,n_components=8,max_distance_bp=1_000_000,max_distance_cm=None,genetic_cm=None,ld_sample_rows=2048,ld_block_variants=256,random_state=0):
    """Backward-compatible single local LD-distance embedding.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Passed to ``ld_group_svd_feature_bank`` as
        ``train_mask``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``8``. Passed to ``ld_group_svd_feature_bank`` as ``n_components``.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``ld_group_svd_feature_bank`` as ``max_distance_bp``.

    max_distance_cm : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``. Passed to ``ld_group_svd_feature_bank`` as ``max_distance_cm``.

    genetic_cm : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.
        Passed to ``ld_group_svd_feature_bank`` as ``genetic_cm``.

    ld_sample_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``ld_group_svd_feature_bank`` as
        ``ld_sample_rows``.

    ld_block_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``256``. Passed to ``ld_group_svd_feature_bank``
        as ``ld_block_variants``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``ld_group_svd_feature_bank`` as ``random_state``.

    Returns
    -------
    object
        Computed result from ``(F, {'r2': r2, 'distance_singular_values': g.get('distance_singular_values', np.array([], np.float32)), 'variant_loadings': g.get('variant_loadings', np.empty((0, 0), np.float32)), 'genotype_mean':...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    bank,info=ld_group_svd_feature_bank(X,positions,train_mask=train_mask,genetic_cm=genetic_cm,n_components=n_components,ld_component_r2=0.0,max_groups=1,max_distance_bp=max_distance_bp,max_distance_cm=max_distance_cm,ld_sample_rows=ld_sample_rows,ld_block_variants=ld_block_variants,random_state=random_state)
    F=next(iter(bank.values())) if bank else np.zeros((len(X),0),np.float32);g=info["groups"][0] if info["groups"] else {}
    r2=info.get("component_r2",{}).get(g.get("label",-1),np.eye(X.shape[1],dtype=np.float32))
    return F,{"r2":r2,"distance_singular_values":g.get("distance_singular_values",np.array([],np.float32)),"variant_loadings":g.get("variant_loadings",np.empty((0,0),np.float32)),"genotype_mean":info["genotype_mean"],"genotype_scale":info["genotype_scale"]}


def ld_pruned_score_variants(scores,r2,*,max_variants=128,prune_r2=.95,eligible=None):
    """Return the linkage disequilibrium pruned score variants result computed by ``np.asarray``.

    Parameters
    ----------
    scores : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    max_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``128``.

    prune_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.95``.

    eligible : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``np.asarray(keep, np.int64)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    s=np.asarray(scores,float);R=np.asarray(r2,float);keep=[];order=np.argsort(np.nan_to_num(s,nan=-np.inf))[::-1]
    if eligible is not None:
        e=np.zeros(len(s),bool);e[np.asarray(eligible,np.int64)]=True;order=order[e[order]]
    for j in order:
        if not np.isfinite(s[j]):continue
        if keep and np.nanmax(R[j,keep])>=float(prune_r2):continue
        keep.append(int(j))
        if len(keep)>=int(max_variants):break
    return np.asarray(keep,np.int64)


def ld_pruned_score_variants_edges(scores, edge_i, edge_j, edge_r2, *, n_variants=None, max_variants=128, prune_r2=.95, eligible=None):
    """Score-rank variants while pruning redundant high-LD neighbors from a sparse edge list.

    Parameters
    ----------
    scores : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    edge_i : array-like or numeric scalar
        Input for the operation ``np.asarray(edge_i, np.int64)``. See
        ``vigor.genomic_kernel_expert.ld_pruned_score_variants_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    edge_j : array-like or numeric scalar
        Input for the operation ``np.asarray(edge_j, np.int64)``. See
        ``vigor.genomic_kernel_expert.ld_pruned_score_variants_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    edge_r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value.

    max_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``128``.

    prune_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.95``.

    eligible : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``np.asarray(chosen, np.int64)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from scipy.sparse import coo_matrix
    s=np.asarray(scores,float);m=len(s) if n_variants is None else int(n_variants)
    ei=np.asarray(edge_i,np.int64);ej=np.asarray(edge_j,np.int64);ev=np.asarray(edge_r2,float)
    hi=np.isfinite(ev)&(ev>=float(prune_r2));ei=ei[hi];ej=ej[hi]
    if len(ei):
        A=coo_matrix((np.ones(len(ei)*2,np.int8),(np.r_[ei,ej],np.r_[ej,ei])),shape=(m,m)).tocsr()
    else:
        A=None
    order=np.argsort(np.nan_to_num(s,nan=-np.inf))[::-1]
    if eligible is not None:
        ok=np.zeros(m,bool);ok[np.asarray(eligible,np.int64)]=True;order=order[ok[order]]
    chosen=[];chosen_mask=np.zeros(m,bool)
    for j in order:
        if not np.isfinite(s[j]):continue
        if A is not None:
            nb=A.indices[A.indptr[j]:A.indptr[j+1]]
            if len(nb) and np.any(chosen_mask[nb]):continue
        chosen.append(int(j));chosen_mask[j]=True
        if len(chosen)>=int(max_variants):break
    return np.asarray(chosen,np.int64)


def _region_feature_bank(genotypes,regions,state,*,sample_ids,train_mask,n_components,max_regions,max_region_variants,raw_variants_per_region,ld_max_distance_bp,ld_max_distance_cm,ld_sample_rows,ld_block_variants,ld_prune_r2,ld_component_r2,max_ld_groups_per_region,founder_feature_fn,random_state,scan_top_variants=None,scan_table=None):
    """Construct bounded local genomic feature groups for selected discovery regions.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    regions : array-like or numeric scalar
        Input for the operation ``np.argsort(np.asarray(regions['lead_p'].values,
        float))[:min(int(max_regions), regions.sizes['region'])]``. See
        ``vigor.genomic_kernel_expert._region_feature_bank`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Passed to ``mf.open_genotype_xarray`` as ``sample_ids``.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Passed to
        ``ld_group_svd_feature_bank_packed`` as ``train_mask``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Passed to ``ld_group_svd_feature_bank_packed`` as ``n_components``.

    max_regions : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy.

    max_region_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy.

    raw_variants_per_region : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Passed to
        ``ld_pruned_score_variants_edges`` as ``max_variants``.

    ld_max_distance_bp : int or None
        Maximum physical variant separation in base pairs for banded LD; positive bounds
        restrict distant pairs and None leaves this distance unrestricted where supported.

    ld_max_distance_cm : object (see the cited input contract)
        Input supplied to ``ld_group_svd_feature_bank`` as ``max_distance_cm``. See that
        callable's documented ``max_distance_cm`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Passed to ``ld_group_svd_feature_bank`` as
        ``max_distance_cm``.

    ld_sample_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Passed to ``ld_group_svd_feature_bank_packed`` as ``ld_sample_rows``.

    ld_block_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Passed to ``ld_group_svd_feature_bank_packed`` as
        ``max_offset``.

    ld_prune_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Passed to ``ld_pruned_score_variants_edges`` as ``prune_r2``.

    ld_component_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Passed to ``ld_group_svd_feature_bank_packed`` as ``ld_component_r2``.

    max_ld_groups_per_region : object (see the cited input contract)
        Input supplied to ``ld_group_svd_feature_bank_packed`` as ``max_groups``. See that
        callable's documented ``max_groups`` contract for its type, alignment and value domain.
        This wrapper retains that contract. Passed to ``ld_group_svd_feature_bank_packed`` as
        ``max_groups``.

    founder_feature_fn : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Passed to
        ``ld_group_svd_feature_bank_packed`` as ``random_state``.

    scan_top_variants : object (see the cited input contract)
        Input for the operation ``scan_top_variants['chrom']``. See
        ``vigor.genomic_kernel_expert._region_feature_bank`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    scan_table : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    mf=_mvgwas_fast_module();g=mf.open_genotype_xarray(genotypes,io_chunk_variants='auto',dtype=np.float32,use_memmap=True,sample_ids=sample_ids);g=mf._optimized_align_samples(g,sample_ids) if sample_ids is not None else g
    chrom=np.asarray(g.coords["chrom"].values).astype(str);pos=np.asarray(g.coords["pos"].values,np.int64);snps=np.asarray(g.coords["snp"].values).astype(str);cm_name=next((nm for nm in ("genetic_cm","cm","cM","genetic_pos_cm") if nm in g.coords),None);gcm=np.asarray(g.coords[cm_name].values,float) if cm_name else None;bank={};meta=[]
    from pathlib import Path
    prefix=genotypes if isinstance(genotypes,(str,Path)) else getattr(genotypes,'attrs',{}).get('prefix',getattr(getattr(genotypes,'reader',None),'prefix',getattr(genotypes,'prefix',None)))
    packed_ok=prefix is not None and Path(str(prefix)+'.bed').exists()
    scan_score={};scan_p={}
    if scan_table is not None and getattr(scan_table,'sizes',{}).get('variant',0) and 'variant_index' in scan_table.coords:
        vv=np.asarray(scan_table['variant_index'].values,np.int64); ss=np.asarray(scan_table['omnibus_stat'].values,float); pp=np.asarray(scan_table['omnibus_p'].values,float)
        scan_score={int(v):float(x) for v,x in zip(vv,ss)};scan_p={int(v):float(x) for v,x in zip(vv,pp)}
    if regions is None or regions.sizes.get("region",0)==0:return bank,meta
    order=np.argsort(np.asarray(regions["lead_p"].values,float))[:min(int(max_regions),regions.sizes["region"])]
    for kk,ri in enumerate(order):
        cc=str(regions["chrom"].values[ri]);lo=int(regions["start_pos"].values[ri]);hi=int(regions["end_pos"].values[ri]);idx_all=np.flatnonzero((chrom==cc)&(pos>=lo)&(pos<=hi))
        if len(idx_all)<2:continue
        idx=idx_all
        # The +/-1 Mb locus remains complete in the discovery results.  The cap
        # below limits only the local LD-SVD geometry and always preserves scan
        # priority variants before evenly filling the physical interval.
        if len(idx)>int(max_region_variants):
            priority=[]
            if "lead_variant_index" in regions.coords:priority.append(int(regions["lead_variant_index"].values[ri]))
            if scan_top_variants is not None and getattr(scan_top_variants,"sizes",{}).get("variant",0):
                tc=np.asarray(scan_top_variants["chrom"].values).astype(str);tp=np.asarray(scan_top_variants["pos"].values,np.int64);tv=np.asarray(scan_top_variants["variant_index"].values,np.int64);priority.extend(tv[(tc==cc)&(tp>=lo)&(tp<=hi)].tolist())
            pri=np.asarray(sorted(set(priority).intersection(set(idx.tolist()))),np.int64);room=max(0,int(max_region_variants)-len(pri));even=idx[np.unique(np.linspace(0,len(idx)-1,max(1,room),dtype=np.int64))] if room else np.array([],np.int64);idx=np.asarray(sorted(set(pri.tolist()+even.tolist())),np.int64)[:int(max_region_variants)]
        rstate=mf._resolve_score_state(state,cc); cm=None if gcm is None else gcm[idx]; X=None
        if scan_score and all(int(v) in scan_score for v in idx):
            z={"stat":np.asarray([scan_score[int(v)] for v in idx],float),"p":np.asarray([scan_p[int(v)] for v in idx],float)}
        else:
            X=mf._xarray_compute(g.isel(snp=idx),np.float32); z=mf.score_variant_block(X,rstate,return_factor_z=True)
        if packed_ok:
            ldbank,ld=ld_group_svd_feature_bank_packed(str(prefix),idx,pos[idx],sample_ids=sample_ids,train_mask=train_mask,
                scores=z["stat"],pvalues=z["p"],n_components=n_components,ld_component_r2=ld_component_r2,
                max_groups=max_ld_groups_per_region,max_distance_bp=ld_max_distance_bp,max_offset=max(64,int(ld_block_variants)),
                ld_sample_rows=ld_sample_rows,random_state=random_state+1009*kk)
        else:
            if X is None: X=mf._xarray_compute(g.isel(snp=idx),np.float32)
            ldbank,ld=ld_group_svd_feature_bank(X,pos[idx],train_mask=train_mask,scores=z["stat"],pvalues=z["p"],genetic_cm=cm,n_components=n_components,ld_component_r2=ld_component_r2,max_groups=max_ld_groups_per_region,max_distance_bp=ld_max_distance_bp,max_distance_cm=ld_max_distance_cm,ld_sample_rows=ld_sample_rows,ld_block_variants=ld_block_variants,random_state=random_state+1009*kk)
        base=f"mvregion::{cc}:{lo}-{hi}";selected_member=[]
        for gm in ld["groups"]:
            local_name=gm["name"];bank[f"{base}::{local_name}::ldsvd"]=ldbank[local_name];selected_member.extend(np.asarray(gm["indices"],np.int64).tolist())
        eligible=np.unique(np.asarray(selected_member,np.int64)) if selected_member else np.arange(len(idx),dtype=np.int64)
        # Raw ML variants are score-prefiltered, but pruning uses the sparse LD
        # edge graph so no dense region-wide R2 matrix is recreated.
        sel=ld_pruned_score_variants_edges(z["stat"],ld["edges"]["i"],ld["edges"]["j"],ld["edges"]["r2"],n_variants=len(idx),max_variants=raw_variants_per_region,prune_r2=ld_prune_r2,eligible=eligible)
        if len(sel):
            if packed_ok:
                from .packed_bed import virtual_dosage
                fam_ids=np.asarray(sample_ids).astype(str); Xsel=np.asarray(virtual_dosage(str(prefix)).sel(iid=fam_ids).isel(snp=idx[sel]).values,np.float32)
            else:
                if X is None: X=mf._xarray_compute(g.isel(snp=idx),np.float32)
                Xsel=X[:,sel]
            bank[base+"::variants"]=_standardize_train_genotypes(Xsel,train_mask)[0]
        if founder_feature_fn is not None:
            H=_to_numpy(founder_feature_fn(chrom=cc,start_pos=lo,end_pos=hi,sample_ids=sample_ids),np.float32);hm=H[train_mask].mean(0);hs=H[train_mask].std(0);hs[hs<1e-7]=1.;bank[base+"::founder"]=(H-hm)/hs
        groups_meta=[{"n_variants":g0["n_variants"],"max_score":g0["max_score"],"min_p":g0["min_p"],"distance_singular_values":g0["distance_singular_values"]} for g0 in ld["groups"]]
        meta.append({"name":base,"chrom":cc,"start_pos":lo,"end_pos":hi,"lead_p":float(regions["lead_p"].values[ri]),"n_region_variants_total":int(len(idx_all)),"n_region_variants_ld":int(len(idx)),"n_ld_groups":int(len(ld["groups"])),"ld_groups":groups_meta,"n_raw_variants":int(len(sel)),"packed_ld":bool(ld.get("packed_ld",False)),"decoded_ld_variants":int(ld.get("decoded_variants",len(idx) if X is not None else 0)),"raw_snps":snps[idx[sel]].tolist(),"raw_variant_scores":np.asarray(z["stat"])[sel].astype(float).tolist(),"raw_variant_p":np.asarray(z["p"])[sel].astype(float).tolist()})
    return bank,meta


@dataclass
class PrefilteredFactorModel:
    """PrefilteredFactorModel represents prefiltered factor model.

    Attributes
    ----------
    method : str
        Stored method; its axis/order follows the construction inputs.
    trait_vectors : np.ndarray
        Stored trait vectors; its axis/order follows the construction inputs.
    transforms : Dict[str, _GroupTransform]
        Stored transforms; its axis/order follows the construction inputs.
    groups : Tuple[str, ...]
        Stored groups; its axis/order follows the construction inputs.
    models : Any
        Stored models; its axis/order follows the construction inputs.
    cv_gain : float
        Stored cv gain; its axis/order follows the construction inputs.
    feature_names : Tuple[str, ...]
        Stored feature names; its axis/order follows the construction inputs.
    shap_importance : Optional[np.ndarray]
        Stored shap importance; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    method : str
        Stored method consumed by ``correction``; see those methods for the numerical/label
        contract. Required construction field.

    trait_vectors : np.ndarray
        Stored trait vectors consumed by ``correction``; see those methods for the
        numerical/label contract. Required construction field.

    transforms : Dict[str, _GroupTransform]
        Stored transforms consumed by ``_transform``; see those methods for the numerical/label
        contract. Required construction field.

    groups : Tuple[str, ...]
        Stored groups consumed by ``_transform``; see those methods for the numerical/label
        contract. Required construction field.

    models : Any
        Stored models consumed by ``correction``; see those methods for the numerical/label
        contract. Required construction field.

    cv_gain : float
        Stored cv gain supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    feature_names : Tuple[str, ...]
        Stored feature names supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    shap_importance : Optional[np.ndarray]
        Stored shap importance supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: ``None``.
    """
    method:str
    trait_vectors:np.ndarray
    transforms:Dict[str,_GroupTransform]
    groups:Tuple[str,...]
    models:Any
    cv_gain:float
    feature_names:Tuple[str,...]
    shap_importance:Optional[np.ndarray]=None
    def _transform(self,bank):
        """Return the transform result computed by ``np.concatenate``.

        Parameters
        ----------
        bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.
        """
        return np.concatenate([_apply_group_transform(bank[n],self.transforms[n]) for n in self.groups],1)
    def correction(self,bank):
        """Compute the fitted genomic expert's residual correction in trait space.

        Parameters
        ----------
        bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.

        Returns
        -------
        object
            Computed result from ``np.asarray(P) @ self.trait_vectors.T``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X=self._transform(bank);P=self.models.predict(X) if self.method in {"ridge","elasticnet"} else np.column_stack([m.predict(X) for m in self.models]);return np.asarray(P)@self.trait_vectors.T


def _fit_factor_method(method,X,Y,seed):
    """Fit factor method.

    Parameters
    ----------
    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Source-recognized literal values:
        ``elasticnet``, ``lightgbm``, ``ridge``, ``xgboost``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Passed to
        ``MultiTaskElasticNetCV`` as ``random_state``.
    """
    method=str(method).lower()
    if method=="ridge":
        from sklearn.linear_model import RidgeCV
        m=RidgeCV(alphas=np.logspace(-3,3,13));m.fit(X,Y);return m
    if method=="elasticnet":
        from sklearn.linear_model import MultiTaskElasticNetCV
        m=MultiTaskElasticNetCV(l1_ratio=(.05,.15,.3,.5,.7,.9),alphas=12,cv=3,max_iter=3000,tol=1e-4,random_state=seed,n_jobs=1);m.fit(X,Y);return m
    cls=None;pars={}
    if method=="lightgbm":
        from lightgbm import LGBMRegressor as cls
        pars=dict(n_estimators=120,num_leaves=15,max_depth=5,learning_rate=.04,min_child_samples=20,subsample=.9,colsample_bytree=.8,reg_lambda=.5,verbosity=-1,n_jobs=1)
    elif method=="xgboost":
        from xgboost import XGBRegressor as cls
        pars=dict(n_estimators=140,max_depth=5,learning_rate=.04,min_child_weight=10,subsample=.9,colsample_bytree=.8,reg_lambda=1.,tree_method="hist",n_jobs=1,verbosity=0)
    else:raise ValueError("method must be ridge, elasticnet, lightgbm, or xgboost")
    out=[]
    for j in range(Y.shape[1]):m=cls(random_state=seed+j,**pars);m.fit(X,Y[:,j]);out.append(m)
    return out


def _predict_factor_method(method,model,X):
    """Predict factor method.

    Parameters
    ----------
    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Source-recognized literal values:
        ``elasticnet``, ``ridge``. These select the corresponding branches below; comparison
        evidence alone does not establish an exhaustive enum.

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here.

    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.
    """
    return model.predict(X) if method in {"ridge","elasticnet"} else np.column_stack([m.predict(X) for m in model])


def fit_prefiltered_factor_models(Y,baseline,feature_bank,G,*,train_mask,trait_rank=10,methods=("ridge","elasticnet","lightgbm","xgboost"),cv=3,min_cv_gain=.005,compute_shap=True,shap_rows=256,random_state=0):
    """Fit prefiltered factor models.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    feature_bank : mapping or feature/annotation bundle
        Prepared feature/annotation collection indexed by its named groups. Reuse the same
        fitted preprocessing, animal/variant order and metadata when transforming or predicting.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    trait_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``.

    methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('ridge', 'elasticnet', 'lightgbm', 'xgboost')``.

    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``.

    min_cv_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.005``.

    compute_shap : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    shap_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``256``. Passed to ``rng.choice`` as ``size``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``(PrefilteredFactorModel(best, basis.vectors, states, tuple(feature_bank.keys()), model, gain, tuple(fn), imp), {'accepted': True, 'cv_gain': scores, 'best': best, 'best_gain': gain, 'feature_names'...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Y=_to_numpy(Y,float);B=_to_numpy(baseline,float);basis=trait_factor_basis(G,min(int(trait_rank),Y.shape[1]));target,valid=_residual_factor_targets(Y,B,basis.vectors,_mask(train_mask,len(Y)));fit=_mask(train_mask,len(Y))&valid;X,states=_whiten_groups(feature_bank,fit);ids=np.flatnonzero(fit);rng=np.random.default_rng(random_state);folds=np.array_split(rng.permutation(ids),max(2,min(int(cv),len(ids)//40 if len(ids)>=80 else 2)));Rraw=Y-B;base=0.;scores={}
    if X.shape[1]==0 or len(ids)<20 or not methods:
        return None,{"accepted":False,"cv_gain":{},"best":None,"best_gain":0.,"reason":"no informative features or too few training animals"}
    for va in folds:
        rr=Rraw[va];obs=np.isfinite(rr);base+=float(np.sum(rr[obs]**2))
    # Discovery/features may be a conditional reference for this CV, but
    # imputation/whitening itself must exclude each scored fold's animals.
    cv_design=[]
    for va in folds:
        tr=np.setdiff1d(ids,va,assume_unique=False);fold_fit=np.zeros(len(Y),bool);fold_fit[tr]=True
        _,fold_states=_whiten_groups(feature_bank,fold_fit)
        design=np.concatenate([_apply_group_transform(feature_bank[name],fold_states[name]) for name in feature_bank],axis=1)
        cv_design.append((tr,va,design))
    for method0 in methods:
        method=str(method0).lower();sse=0.
        for fi,(tr,va,design) in enumerate(cv_design):
            mdl=_fit_factor_method(method,design[tr],target[tr],random_state+100*fi);P=_predict_factor_method(method,mdl,design[va]);C=np.asarray(P)@basis.vectors.T;rr=Rraw[va];obs=np.isfinite(rr);sse+=float(np.sum((rr[obs]-C[obs])**2))
        scores[method]=float(1-sse/max(base,1e-30))
    best=max(scores,key=scores.get);gain=scores[best]
    if gain<float(min_cv_gain):return None,{"accepted":False,"cv_gain":scores,"best":best,"best_gain":gain}
    model=_fit_factor_method(best,X[fit],target[fit],random_state+999);fn=[]
    for name,st in states.items():fn += [f"{name}::{j+1}" for j in range(st.out_slice.stop-st.out_slice.start)]
    imp=None
    if compute_shap and X.shape[1]:
        rid=ids if len(ids)<=int(shap_rows) else np.sort(rng.choice(ids,size=int(shap_rows),replace=False));Xs=X[rid]
        if best in {"ridge","elasticnet"}:
            coef=np.asarray(model.coef_,float);coef=coef[None,:] if coef.ndim==1 else coef;imp=np.mean(np.abs(Xs[:,None,:]*coef[None,:,:]),axis=(0,1))
        else:
            try:
                import shap
                acc=np.zeros(X.shape[1]);
                for m in model:acc+=np.mean(np.abs(np.asarray(shap.TreeExplainer(m).shap_values(Xs),float)),0)
                imp=acc/max(1,len(model))
            except Exception:imp=np.mean([np.asarray(getattr(m,"feature_importances_",np.zeros(X.shape[1])),float) for m in model],axis=0)
    return PrefilteredFactorModel(best,basis.vectors,states,tuple(feature_bank.keys()),model,gain,tuple(fn),imp),{"accepted":True,"cv_gain":scores,"best":best,"best_gain":gain,"feature_names":fn,"shap_importance":imp}


@dataclass
class GenomicKernelExpert:
    """GenomicKernelExpert represents genomic kernel expert.

    Attributes
    ----------
    model : Any
        Stored model; its axis/order follows the construction inputs.
    selected_scores : list
        Stored selected scores; its axis/order follows the construction inputs.
    selected_groups : Tuple[str, ...]
        Stored selected groups; its axis/order follows the construction inputs.
    feature_bank : Dict[str, np.ndarray]
        Stored feature bank; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    model : Any
        Stored model consumed by ``predict``; see those methods for the numerical/label
        contract. Required construction field.

    selected_scores : list
        Stored selected scores supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    selected_groups : Tuple[str, ...]
        Stored selected groups supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    feature_bank : Dict[str, np.ndarray]
        Stored feature bank consumed by ``predict``; see those methods for the numerical/label
        contract. Required construction field.
    """
    model:Any
    selected_scores:list
    selected_groups:Tuple[str,...]
    feature_bank:Dict[str,np.ndarray]
    def predict(self,baseline,feature_bank=None):
        """Apply the fitted estimator/preprocessing state to aligned input observations.

        Parameters
        ----------
        baseline : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        feature_bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        Returns
        -------
        object
            Computed result from ``_to_numpy(baseline, float) + self.model.correction(bank)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        bank=self.feature_bank if feature_bank is None else feature_bank;return _to_numpy(baseline,float)+self.model.correction(bank)


def fit_genomic_kernel_expert(Y,baseline,G,*,train_mask,dual_A,global_kernel=None,candidate_kernels=None,trait_rank=10,global_rank=48,global_bands=6,kernel_rank=10,top_kernels=6,n_landmarks=192,min_cv_gain=0.0,random_state=0):
    """Fit genomic kernel expert.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    dual_A : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    global_kernel : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    candidate_kernels : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    trait_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``. Passed to ``dual_factor_signal`` as ``rank``.

    global_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``48``. Passed to ``landmark_nystrom_features`` as
        ``rank``.

    global_bands : int
        Input supplied to ``spectral_feature_groups`` as ``n_bands``. See that callable's
        documented ``n_bands`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``6``. Passed to ``spectral_feature_groups`` as
        ``n_bands``.

    kernel_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``.

    top_kernels : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``6``. Passed to ``screen_kernel_bank`` as ``top_k``.

    n_landmarks : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``192``.
        Passed to ``landmark_nystrom_features`` as ``n_landmarks``.

    min_cv_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.0``. Passed to ``fit_lowrank_group_ridge`` as
        ``min_cv_gain``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``landmark_nystrom_features`` as ``random_state``.

    Returns
    -------
    object
        Computed result from ``GenomicKernelExpert(model, scores, tuple(bank), bank)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Y0=_to_numpy(Y,float);B=_to_numpy(baseline,float);fit=_mask(train_mask,len(Y0));Gtot=_total_trait_cov(G);D,_=dual_factor_signal(dual_A,Gtot,rank=min(trait_rank,Y0.shape[1]),train_mask=fit);bank={};scores=[]
    if global_kernel is not None:
        F,st=landmark_nystrom_features(global_kernel,train_mask=fit,rank=global_rank,n_landmarks=max(n_landmarks,2*global_rank),random_state=random_state,name="global")
        for name,z in spectral_feature_groups(F,st.landmark_values,n_bands=global_bands,prefix="global_spec").items():bank[name]=z;sc=feature_subspace_score(z,D,train_mask=fit);scores.append((name,sc["score"],"spectral"))
    if candidate_kernels:
        rows,kb=screen_kernel_bank(candidate_kernels,D,train_mask=fit,rank=kernel_rank,n_landmarks=n_landmarks,top_k=top_kernels,random_state=random_state+100)
        for name,sc,_,_ in rows:scores.append((name,sc,"kernel"));bank[f"kernel::{name}"]=kb[name]
    if not bank:raise ValueError("no genomic features")
    model=fit_lowrank_group_ridge(Y0,B,bank,Gtot,train_mask=fit,trait_rank=trait_rank,min_cv_gain=min_cv_gain,random_state=random_state);return GenomicKernelExpert(model,scores,tuple(bank),bank)


@dataclass
class GenomicKernelExpertScanFit:
    """GenomicKernelExpertScanFit represents genomic kernel expert scan fit.

    Attributes
    ----------
    expert : GenomicKernelExpert
        Stored expert; its axis/order follows the construction inputs.
    scan : Any
        Stored scan; its axis/order follows the construction inputs.
    selected_regions : Any
        Stored selected regions; its axis/order follows the construction inputs.
    selected_chromosomes : Tuple[str, ...]
        Stored selected chromosomes; its axis/order follows the construction inputs.
    region_metadata : list
        Stored region metadata; its axis/order follows the construction inputs.
    local_model : Optional[PrefilteredFactorModel]
        Stored local model; its axis/order follows the construction inputs.
    local_model_diagnostics : Optional[dict]
        Stored local model diagnostics; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    expert : GenomicKernelExpert
        Stored expert consumed by ``predict``; see those methods for the numerical/label
        contract. Required construction field.

    scan : Any
        Stored scan supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    selected_regions : Any
        Stored selected regions consumed by ``selected_windows``; see those methods for the
        numerical/label contract. Required construction field.

    selected_chromosomes : Tuple[str, ...]
        Stored selected chromosomes supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    region_metadata : list
        Stored region metadata supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    local_model : Optional[PrefilteredFactorModel]
        Stored local model consumed by ``predict``; see those methods for the numerical/label
        contract. Default: ``None``.

    local_model_diagnostics : Optional[dict]
        Stored local model diagnostics supplied by the corresponding fitting/planning routine.
        See that routine for the result-axis/value contract. Default: ``None``.
    """
    expert:GenomicKernelExpert
    scan:Any
    selected_regions:Any
    selected_chromosomes:Tuple[str,...]
    region_metadata:list
    local_model:Optional[PrefilteredFactorModel]=None
    local_model_diagnostics:Optional[dict]=None
    @property
    def selected_windows(self):
        """Merge selected association leads into bounded physical discovery windows.

        Returns
        -------
        object
            Computed result from ``self.selected_regions``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.selected_regions
    def predict(self,baseline,feature_bank=None):
        """Apply the fitted estimator/preprocessing state to aligned input observations.

        Parameters
        ----------
        baseline : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        feature_bank : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        Returns
        -------
        object
            Computed result from ``B + (self.local_model.correction(bank) if self.local_model is not None else self.expert.model.correction(bank))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        bank=self.expert.feature_bank if feature_bank is None else feature_bank;B=_to_numpy(baseline,float);return B+(self.local_model.correction(bank) if self.local_model is not None else self.expert.model.correction(bank))


def fit_genomic_kernel_expert_xarray(
    Y,baseline,G,*,E,train_mask,dual_A,genotypes,sample_ids=None,global_kernel=None,candidate_kernels=None,selected_chromosomes=None,
    trait_rank=10,global_rank=48,global_bands=6,kernel_rank=10,top_kernels=6,local_components=8,top_regions=24,n_landmarks=192,
    io_chunk_variants='auto',score_chunk_variants=None,calibration_n=4096,scan_top_k=2048,region_p_threshold=1e-5,region_padding_bp=1_000_000,
    recombination_rate_cM_per_mb=.5,merge_recombination_cM=1.,scan_zarr_path=None,exact_rescore_fn=None,exact_rescore_p=5e-7,exact_rescore_top=64,
    max_region_variants=8192,raw_variants_per_region=96,ld_max_distance_bp=1_000_000,ld_max_distance_cm=None,ld_sample_rows=2048,ld_block_variants=256,ld_prune_r2=.95,ld_component_r2=.20,max_ld_groups_per_region=8,
    founder_feature_fn=None,loco_score_states=None,assume_complete=False,grouped_ridge_min_cv_gain=.001,local_model_methods=None,local_model_min_cv_gain=.005,compute_shap=True,random_state=0,discovery_train_mask=None,scan_progress_callback=None,precomputed_scan=None,scan_result_callback=None,maxmem=None,memory_fraction=None):
    """Fit genomic kernel expert xarray.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.

    dual_A : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``mf.open_genotype_xarray`` as ``sample_ids``.

    global_kernel : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    candidate_kernels : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    selected_chromosomes : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value. Passed to ``mf.scan_xarray`` as
        ``chromosomes``.

    trait_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``. Passed to ``dual_factor_signal`` as ``rank``.

    global_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``48``. Passed to ``landmark_nystrom_features`` as
        ``rank``.

    global_bands : int
        Input supplied to ``spectral_feature_groups`` as ``n_bands``. See that callable's
        documented ``n_bands`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``6``. Passed to ``spectral_feature_groups`` as
        ``n_bands``.

    kernel_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``.

    top_kernels : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``6``. Passed to ``screen_kernel_bank`` as ``top_k``.

    local_components : int
        Input supplied to ``_region_feature_bank`` as ``n_components``. See that callable's
        documented ``n_components`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``8``. Passed to ``_region_feature_bank`` as
        ``n_components``.

    top_regions : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``24``. Passed to ``mf.scan_xarray`` as ``max_regions``.

    n_landmarks : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``192``.
        Passed to ``landmark_nystrom_features`` as ``n_landmarks``.

    io_chunk_variants : int or 'auto'
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``'auto'`` (sized from the working-memory budget;
        integers are ceilings). Passed to ``mf.open_genotype_xarray`` as ``io_chunk_variants``.

    score_chunk_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value. Passed to
        ``mf.scan_xarray`` as ``score_chunk_variants``.

    calibration_n : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4096``. Passed to ``mf.scan_xarray`` as ``calibration_n``.

    scan_top_k : int
        Input supplied to ``mf.scan_xarray`` as ``top_k``. See that callable's documented
        ``top_k`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``2048``. Passed to ``mf.scan_xarray`` as ``top_k``.

    region_p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``1e-05``. Passed to ``mf.scan_xarray`` as ``region_p_threshold``.

    region_padding_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``mf.scan_xarray`` as ``region_padding_bp``.

    recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``. Passed to ``mf.scan_xarray`` as
        ``region_recombination_rate_cM_per_mb``.

    merge_recombination_cM : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``1.0``. Passed to ``mf.scan_xarray`` as ``region_merge_recombination_cM``.

    scan_zarr_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``. Passed to
        ``mf.scan_xarray`` as ``zarr_path``.

    exact_rescore_fn : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.
        Passed to ``mf.scan_xarray`` as ``exact_rescore_fn``.

    exact_rescore_p : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``5e-07``. Passed to ``mf.scan_xarray`` as ``exact_rescore_p``.

    exact_rescore_top : int
        Number of top-ranked association candidates selected for exact rescoring regardless of
        the p-value threshold; zero disables this top-count route. Default: ``64``. Passed to
        ``mf.scan_xarray`` as ``exact_rescore_top``.

    max_region_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``8192``. Passed to ``_region_feature_bank`` as ``max_region_variants``.

    raw_variants_per_region : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``96``. Passed to ``_region_feature_bank`` as ``raw_variants_per_region``.

    ld_max_distance_bp : int or None
        Maximum physical variant separation in base pairs for banded LD; positive bounds
        restrict distant pairs and None leaves this distance unrestricted where supported.
        Default: ``1000000``.

    ld_max_distance_cm : object (see the cited input contract)
        Input supplied to ``_region_feature_bank`` as ``ld_max_distance_cm``. See that
        callable's documented ``ld_max_distance_cm`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``_region_feature_bank`` as ``ld_max_distance_cm``.

    ld_sample_rows : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``_region_feature_bank`` as ``ld_sample_rows``.

    ld_block_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``256``. Passed to ``_region_feature_bank`` as
        ``ld_block_variants``.

    ld_prune_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.95``. Passed to ``_region_feature_bank`` as ``ld_prune_r2``.

    ld_component_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.2``. Passed to ``_region_feature_bank`` as ``ld_component_r2``.

    max_ld_groups_per_region : int
        Input supplied to ``_region_feature_bank`` as ``max_ld_groups_per_region``. See that
        callable's documented ``max_ld_groups_per_region`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``8``. Passed to
        ``_region_feature_bank`` as ``max_ld_groups_per_region``.

    founder_feature_fn : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.
        Passed to ``_region_feature_bank`` as ``founder_feature_fn``.

    loco_score_states : object (see the cited input contract)
        Input for the operation ``loco_score_states is None``. See
        ``vigor.genomic_kernel_expert.fit_genomic_kernel_expert_xarray`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``.

    assume_complete : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``. Passed to
        ``mf.scan_xarray`` as ``assume_complete``.

    grouped_ridge_min_cv_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.001``. Passed to ``fit_lowrank_group_ridge`` as
        ``min_cv_gain``.

    local_model_methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value. Passed to
        ``fit_prefiltered_factor_models`` as ``methods``.

    local_model_min_cv_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.005``. Passed to ``fit_prefiltered_factor_models`` as
        ``min_cv_gain``.

    compute_shap : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``. Passed to ``fit_prefiltered_factor_models``
        as ``compute_shap``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``landmark_nystrom_features`` as ``random_state``.

    discovery_train_mask : object (see the cited input contract)
        Input for the operation ``_mask(discovery_train_mask, len(Y0))``. See
        ``vigor.genomic_kernel_expert.fit_genomic_kernel_expert_xarray`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    scan_progress_callback : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.
        Passed to ``mf.scan_xarray`` as ``progress_callback``.

    precomputed_scan : mapping
        Input for the operation ``precomputed_scan.diagnostics.get('discovery_context_sha256')
        != context``. See ``vigor.genomic_kernel_expert.fit_genomic_kernel_expert_xarray`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    scan_result_callback : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.

    maxmem, memory_fraction : budget or None
        Working-memory budget for genotype I/O and score chunks; see
        ``vigor.resources.working_memory_bytes``. Default: ``None`` (80% of physical RAM).

    Returns
    -------
    object
        Computed result from ``GenomicKernelExpertScanFit(expert, scan, scan.top_regions, tuple([] if selected_chromosomes is None else map(str, selected_chromosomes)), meta, local, diag)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    mf=_mvgwas_fast_module();Y0=_to_numpy(Y,float);B0=_to_numpy(baseline,float);fit=_mask(train_mask,len(Y0));discovery=fit if discovery_train_mask is None else _mask(discovery_train_mask,len(Y0));Gtot=_total_trait_cov(G);D,_=dual_factor_signal(dual_A,Gtot,rank=min(trait_rank,Y0.shape[1]),train_mask=discovery);bank={};scores=[]
    if sample_ids is None and hasattr(Y,"index"):sample_ids=list(map(str,Y.index))
    if global_kernel is not None:
        F,st=landmark_nystrom_features(global_kernel,train_mask=discovery,rank=global_rank,n_landmarks=max(n_landmarks,2*global_rank),random_state=random_state,name="global")
        for name,z in spectral_feature_groups(F,st.landmark_values,n_bands=global_bands,prefix="global_spec").items():bank[name]=z;sc=feature_subspace_score(z,D,train_mask=discovery);scores.append((name,sc["score"],"spectral"))
    if selected_chromosomes is None and candidate_kernels:
        rows,kb=screen_kernel_bank(candidate_kernels,D,train_mask=discovery,rank=kernel_rank,n_landmarks=n_landmarks,top_k=top_kernels,random_state=random_state+100);selected_chromosomes=[r[0] for r in rows]
        for name,sc,_,_ in rows:scores.append((name,sc,"kernel"));bank[f"kernel::{name}"]=kb[name]
    elif selected_chromosomes is not None:selected_chromosomes=list(map(str,selected_chromosomes))
    lazy_genotypes=mf.open_genotype_xarray(genotypes,io_chunk_variants=io_chunk_variants,sample_ids=sample_ids,maxmem=maxmem,memory_fraction=memory_fraction)
    shared_state=mf.prepare_fast_score_state(dual_A,Gtot,E,rank=min(trait_rank,Y0.shape[1]),basis="canonical",train_mask=discovery)
    scan_state=shared_state if loco_score_states is None else loco_score_states
    # Cache reuse is allowed only for an identical file-backed discovery null,
    # allele/sample ordering and genotype source stat signature.
    prefix=lazy_genotypes.attrs.get('prefix',lazy_genotypes.attrs.get('plink_prefix'))
    h=hashlib.sha256()
    for a in (np.asarray(dual_A),Gtot,np.asarray(E),discovery,np.asarray(sample_ids).astype('U')):h.update(np.ascontiguousarray(a).tobytes())
    if prefix:
        from pathlib import Path
        for ext in ('.bed','.bim','.fam'):
            p=Path(str(prefix)+ext);st=p.stat();h.update(repr((str(p.resolve()),st.st_size,st.st_mtime_ns)).encode())
    context=h.hexdigest()
    if precomputed_scan is not None:
        if not prefix or precomputed_scan.diagnostics.get('discovery_context_sha256')!=context:
            raise ValueError('cached scan does not match file-backed genotype/discovery null context')
        scan=precomputed_scan
    else:
        scan=mf.scan_xarray(lazy_genotypes,scan_state,chromosomes=selected_chromosomes,sample_ids=sample_ids,io_chunk_variants=io_chunk_variants,score_chunk_variants=score_chunk_variants,calibration_n=calibration_n,top_k=scan_top_k,return_factor_z=True,region_p_threshold=region_p_threshold,region_padding_bp=region_padding_bp,region_recombination_rate_cM_per_mb=recombination_rate_cM_per_mb,region_merge_recombination_cM=merge_recombination_cM,max_regions=top_regions,exact_rescore_p=exact_rescore_p,exact_rescore_top=exact_rescore_top,exact_rescore_fn=exact_rescore_fn,zarr_path=scan_zarr_path,return_all=scan_zarr_path is not None,assume_complete=assume_complete,progress_callback=scan_progress_callback,maxmem=maxmem,memory_fraction=memory_fraction)
        scan.diagnostics['discovery_context_sha256']=context
    if scan_result_callback is not None:scan_result_callback(scan)

    region_states=scan.calibrated_states if scan.calibrated_states else {str(c):mf.calibrate_xarray_score_state(mf.open_genotype_xarray(genotypes,io_chunk_variants=io_chunk_variants,sample_ids=sample_ids,maxmem=maxmem,memory_fraction=memory_fraction),shared_state,chromosomes=[c],sample_ids=sample_ids,n_calibration=calibration_n,maxmem=maxmem,memory_fraction=memory_fraction) for c in (selected_chromosomes or [])}
    rbank,meta=_region_feature_bank(lazy_genotypes,scan.top_regions,region_states,sample_ids=sample_ids,train_mask=discovery,n_components=local_components,max_regions=top_regions,max_region_variants=max_region_variants,raw_variants_per_region=raw_variants_per_region,ld_max_distance_bp=ld_max_distance_bp,ld_max_distance_cm=ld_max_distance_cm,ld_sample_rows=ld_sample_rows,ld_block_variants=ld_block_variants,ld_prune_r2=ld_prune_r2,ld_component_r2=ld_component_r2,max_ld_groups_per_region=max_ld_groups_per_region,founder_feature_fn=founder_feature_fn,random_state=random_state+10000,scan_top_variants=scan.top_variants,scan_table=scan.table);bank.update(rbank)
    for z in meta:scores.append((z["name"],-math.log10(max(z["lead_p"],1e-300)),"mvscore_ld_region"))
    if not bank:raise ValueError("no genomic feature groups survived screening")
    ridge=fit_lowrank_group_ridge(Y0,B0,bank,Gtot,train_mask=fit,trait_rank=trait_rank,min_cv_gain=grouped_ridge_min_cv_gain,random_state=random_state);local=diag=None
    if local_model_methods:local,diag=fit_prefiltered_factor_models(Y0,B0,bank,Gtot,train_mask=fit,trait_rank=trait_rank,methods=local_model_methods,min_cv_gain=local_model_min_cv_gain,compute_shap=compute_shap,random_state=random_state+20000)
    expert=GenomicKernelExpert(ridge,scores,tuple(bank),bank);return GenomicKernelExpertScanFit(expert,scan,scan.top_regions,tuple([] if selected_chromosomes is None else map(str,selected_chromosomes)),meta,local,diag)


def estimate_large_scale(n_animals=10_000,n_traits=50,n_variants=5_500_000,trait_rank=10,io_chunk_variants='auto',score_chunk_variants='auto',top_regions=24,local_components=8,maxmem=None,memory_fraction=None):
    """Estimate large scale.

    Parameters
    ----------
    n_animals : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``10000``.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension. Default: ``50``.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``5500000``.

    trait_rank : int, str or None
        Retained rank of the corresponding kernel/trait covariance or genomic feature group.
        Rank must fit that matrix dimension; full/auto/None semantics follow the specific
        resolver invoked below. Default: ``10``.

    io_chunk_variants : int or 'auto'
        Variants per I/O block, resolved against this machine's working-memory budget
        (``maxmem``/``memory_fraction``) exactly as ``open_genotype_xarray`` does; integers are
        ceilings. Default: ``'auto'``.

    score_chunk_variants : int or 'auto'
        Accepted for API symmetry; not used by the estimate. Default: ``'auto'``.

    top_regions : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``24``.

    local_components : int
        Input for the operation ``n * int(top_regions) * int(local_components) * 4 / 1024 **
        2``. See ``vigor.genomic_kernel_expert.estimate_large_scale`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``8``.

    Returns
    -------
    dict
        Result fields: ``full_dense_genotypes_float32_gb_avoid``, ``io_chunk_variants``, ``io_chunk_float32_gb``, ``score_dual_mb``, ``phenotypes_mb``, ``retained_local_pc_mb``. Field arrays preserve the corresponding input axis order.
    """
    from .resources import resolve_variant_chunk
    n=int(n_animals);T=int(n_traits);M=int(n_variants)
    io=resolve_variant_chunk(io_chunk_variants,n,bytes_per_call=16,maxmem=maxmem,memory_fraction=memory_fraction,n_variants=M,alignment=1)  # same 16 B/call float32 I/O planner as open_genotype_xarray
    return {"full_dense_genotypes_float32_gb_avoid":n*M*4/1024**3,"io_chunk_variants":int(io),"io_chunk_float32_gb":n*int(io)*4/1024**3,"score_dual_mb":n*int(trait_rank)*4/1024**2,"phenotypes_mb":n*T*4/1024**2,"retained_local_pc_mb":n*int(top_regions)*int(local_components)*4/1024**2}


def crossfit_genomic_kernel_expert_xarray(
    traits, GRM, genotypes, validation_masks, *, sample_ids=None,
    covariance_kwargs=None, expert_kwargs=None, global_kernel=None, candidate_kernels=None,
    prediction_selection=None, random_state=0,
):
    """Leakage-safe outer-fold genomic-expert predictions.

    Every fold masks its validation phenotype cells *before* covariance/null
    estimation and before mvGWAS/region/LD/feature selection.  The genomic expert
    augments the genetic mean for both fully unseen and partially observed
    animals; the partial-trait residual conditional mean is then added back.

    ``validation_masks`` is an iterable of boolean sample x trait masks.  This
    makes the routine usable for both row-held-out and arbitrary cell-held-out
    benchmarks without imposing one missingness design.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model. Passed to
        ``fit_genomic_kernel_expert_xarray`` as ``genotypes``.

    validation_masks : object (see the cited input contract)
        Input for the operation ``enumerate(validation_masks)``. See
        ``vigor.genomic_kernel_expert.crossfit_genomic_kernel_expert_xarray`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    expert_kwargs : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Default: ``None``.

    global_kernel : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value. Passed to ``fit_genomic_kernel_expert_xarray`` as
        ``global_kernel``.

    candidate_kernels : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value. Passed to ``fit_genomic_kernel_expert_xarray`` as
        ``candidate_kernels``.

    prediction_selection : object (see the cited input contract)
        Input for the operation ``mv.apply_prediction_covariance_selection(cov,
        prediction_selection, 'partial_traits')``. See
        ``vigor.genomic_kernel_expert.crossfit_genomic_kernel_expert_xarray`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``fit_genomic_kernel_expert_xarray`` as ``random_state``.

    Returns
    -------
    dict
        Result fields: ``prediction``, ``folds``, ``validation_masks``, ``leakage_contract``. Field arrays preserve the corresponding input axis order.
    """
    import pandas as pd
    from . import multivariate as mv
    Y=traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits)
    ids=np.asarray(Y.index.astype(str) if sample_ids is None else list(map(str,sample_ids)))
    if len(ids)!=len(Y): raise ValueError("sample_ids must match trait rows")
    ckw={} if covariance_kwargs is None else dict(covariance_kwargs)
    ckw.setdefault("n_boot",0); ckw.setdefault("progress",False)
    ekw={} if expert_kwargs is None else dict(expert_kwargs)
    oof=np.full(Y.shape,np.nan,float); fold_results=[]
    for fold,vm0 in enumerate(validation_masks):
        vm=np.asarray(vm0,bool)
        if vm.shape!=Y.shape: raise ValueError("each validation mask must be sample x trait")
        Ytr=Y.mask(vm)
        cov=mv.genetic_varcov_PSD(GRM,Ytr,**ckw)
        # Prediction covariance policy can be preselected by an inner CV run.
        if prediction_selection is None:
            partial=mv.prediction_covariance_view(cov,"partial_traits")
            unseen=mv.prediction_covariance_view(cov,"unseen_animals")
        else:
            partial=mv.apply_prediction_covariance_selection(cov,prediction_selection,"partial_traits")
            unseen=mv.apply_prediction_covariance_selection(cov,prediction_selection,"unseen_animals")
        pp=mv.mvPredict(Ytr,GRM,partial["G"],E=partial["E"],accuracy="exact",kernel_rank="full")
        pu=mv.mvPredict(Ytr,GRM,unseen["G"],E=unseen["E"],accuracy="exact",kernel_rank="full")
        gen=np.asarray(pp["genetic_mean"],float).copy()
        empty=~Ytr.notna().any(axis=1).to_numpy()
        if np.any(empty): gen[empty]=np.asarray(pu["genetic_mean"],float)[empty]
        residual=np.asarray(pp["residual_prediction"],float)
        # The discovery null uses only outer-training observations and the
        # partial-information prediction geometry because phenotype-informed
        # training rows are the rows contributing to genomic-expert fitting.
        from .gwas.exact import prepare_score_null
        null=prepare_score_null(Ytr,GRM,partial["G"],partial["E"])
        dual_signal=np.zeros(Ytr.shape,dtype=np.float64)
        dual_signal.ravel()[null.observed]=null.py
        train_rows=Ytr.notna().any(axis=1).to_numpy()
        fit=fit_genomic_kernel_expert_xarray(
            Ytr,gen,partial["G"],E=partial["E"],train_mask=train_rows,
            dual_A=dual_signal,genotypes=genotypes,sample_ids=ids,
            global_kernel=global_kernel,candidate_kernels=candidate_kernels,
            random_state=int(random_state)+10007*fold,**ekw,
        )
        pred_gen=fit.predict(gen)
        pred=np.asarray(pred_gen,float)+residual
        oof[vm]=pred[vm]
        fold_results.append({"fold":fold,"fit":fit,"covariance":cov,"n_validation":int(vm.sum()),
                             "n_fully_unseen_rows":int(empty.sum())})
    return {"prediction":pd.DataFrame(oof,index=Y.index,columns=Y.columns),
            "folds":fold_results,"validation_masks":[np.asarray(x,bool) for x in validation_masks],
            "leakage_contract":"all phenotype-dependent covariance/null/scan/region/feature selection rerun after outer-fold masking"}
