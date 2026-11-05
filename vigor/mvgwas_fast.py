"""Fast multivariate GWAS discovery for VIGOR.

Designed for ~1e3-1e4 animals, 10-100s of traits and millions of variants.
The discovery scan projects the exact mvBLUP/LMM dual matrix onto a small
number of genetic trait axes and scans one genotype chunk with one GEMM::

    A_scan = A @ V_genetic
    U      = X.T @ A_scan

Only compact per-variant omnibus statistics (plus optional factor z-scores)
are retained.  Exact full-trait rescoring is deferred until the end of each
chromosome.  Candidate loci are significance-seeded +/- physical windows with
recombination-aware merging, not fixed SNP-count windows.

The preferred genotype input is an xarray.DataArray backed by npplink's lazy
PLINK/PGEN reader.  ``scan_xarray`` accepts either one score state or a
chromosome->state mapping/callable; the latter is the true LOCO path. Optional
output is sharded, consolidated Zarr v3.
"""
from __future__ import annotations

# Experimental calibrated feature-discovery helpers remain for genomic experts.
# Public scientific association entry points use vigor.gwas.exact exclusively.
from .io import open_bed_memmap

from dataclasses import dataclass, replace as _replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import inspect
import hashlib
import math
import os

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import chi2

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


__all__ = [
    "TraitScanBasis", "ScoreCalibration", "FastScoreState", "ScanResult",
    "trait_scan_basis", "prepare_fast_score_state", "state_from_vigor_null", "null_trait_duals", "calibrate_score_state",
    "calibrate_xarray_score_state", "score_variant_block",
    "open_genotype_xarray", "scan_xarray", "scan_xarray_recombination", "scan_packed_bed",
    "prepare_packed_score_tables", "calibrate_packed_score_state", "score_packed_bed_block",
    "candidate_regions_from_scan", "pairwise_r2_xarray", "pairwise_r2_xarray_lazy", "fast_windowed_r2", "fast_windowed_r2_edges", "open_scan_zarr3",
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


def _sym(a: np.ndarray) -> np.ndarray:
    """Return the arithmetic symmetrization (M + M.T)/2.

    Parameters
    ----------
    a : np.ndarray
        Numerical operand in ``0.5 * (a + a.T)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.
    """
    return 0.5 * (a + a.T)


def _project_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the positive-semidefinite cone by flooring eigenvalues.

    Parameters
    ----------
    a : np.ndarray
        Numerical operand in ``_sym(np.asarray(a, np.float64))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-10``.
    """
    a = _sym(np.asarray(a, np.float64))
    w, v = np.linalg.eigh(a)
    floor = max(float(eps), float(np.max(np.abs(w))) * 1e-10 if len(w) else float(eps))
    w = np.maximum(w, floor)
    return (v * w[None, :]) @ v.T


def _require_xarray():
    """Import and verify availability of xarray.
    """
    try:
        import xarray as xr
        return xr
    except Exception as exc:  # pragma: no cover
        raise ImportError("xarray is required for the optimized mvGWAS path") from exc


def _xarray_compute(a: Any, dtype=np.float32) -> np.ndarray:
    """Return the xarray compute result computed by ``np.asarray``.

    Parameters
    ----------
    a : Any
        Numerical operand in ``getattr(a, 'data', a)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.
    """
    data = getattr(a, "data", a)
    if hasattr(data, "compute"):
        data = data.compute()
    data = getattr(data, "values", data)
    return np.asarray(data, dtype=dtype, order="C")


@dataclass(frozen=True)
class TraitScanBasis:
    """TraitScanBasis represents trait scan basis.

    Attributes
    ----------
    vectors : np.ndarray
        Stored vectors; its axis/order follows the construction inputs.
    eigenvalues : np.ndarray
        Stored eigenvalues; its axis/order follows the construction inputs.
    mode : str
        Stored mode; its axis/order follows the construction inputs.
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

    mode : str
        Stored mode supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    rank : int
        Number of retained basis/factor columns. Required construction field.
    """
    vectors: np.ndarray
    eigenvalues: np.ndarray
    mode: str
    rank: int


@dataclass(frozen=True)
class ScoreCalibration:
    """ScoreCalibration represents score calibration.

    Attributes
    ----------
    covariance : np.ndarray
        Stored covariance; its axis/order follows the construction inputs.
    inverse : np.ndarray
        Stored inverse; its axis/order follows the construction inputs.
    rank : int
        Stored rank; its axis/order follows the construction inputs.
    lambda_gc : float
        Stored lambda gc; its axis/order follows the construction inputs.
    n_variants : int
        Stored n variants; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    covariance : np.ndarray
        Stored covariance supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    inverse : np.ndarray
        Stored inverse supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    rank : int
        Number of retained basis/factor columns. Required construction field.

    lambda_gc : float
        Stored lambda gc supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    n_variants : int
        Stored n variants supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.
    """
    covariance: np.ndarray
    inverse: np.ndarray
    rank: int
    lambda_gc: float
    n_variants: int


@dataclass(frozen=True)
class FastScoreState:
    """FastScoreState represents fast score state.

    Attributes
    ----------
    A : np.ndarray
        Stored A; its axis/order follows the construction inputs.
    basis : TraitScanBasis
        Stored basis; its axis/order follows the construction inputs.
    calibration : Optional[ScoreCalibration]
        Stored calibration; its axis/order follows the construction inputs.
    trait_names : Tuple[str, ...]
        Stored trait names; its axis/order follows the construction inputs.
    fit_mask : Optional[np.ndarray]
        Stored fit mask; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    A : np.ndarray
        Stored A supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    basis : TraitScanBasis
        Stored basis supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    calibration : Optional[ScoreCalibration]
        Stored calibration supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    trait_names : Tuple[str, ...]
        Stored trait names supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    fit_mask : Optional[np.ndarray]
        Stored fit mask supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``None``.

    null_A : Optional[np.ndarray]
        Basis-projected duals of simulated null traits, n x (reps * r), stacked by
        replicate. Calibration estimates the score covariance M and lambda from
        these, never from the real traits, so association signal cannot leak into
        the calibration. Default: ``None`` (calibration then raises).
    """
    A: np.ndarray                    # n x r, float32
    basis: TraitScanBasis
    calibration: Optional[ScoreCalibration]
    trait_names: Tuple[str, ...]
    fit_mask: Optional[np.ndarray] = None
    null_A: Optional[np.ndarray] = None


@dataclass
class ScanResult:
    """ScanResult represents scan result.

    Attributes
    ----------
    table : Any
        Stored table; its axis/order follows the construction inputs.
    top_variants : Any
        Stored top variants; its axis/order follows the construction inputs.
    top_windows : Any
        Stored top windows; its axis/order follows the construction inputs.
    diagnostics : Dict[str, Any]
        Stored diagnostics; its axis/order follows the construction inputs.
    calibrated_states : Optional[Dict[str, FastScoreState]]
        Stored calibrated states; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    table : Any
        Stored table supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    top_variants : Any
        Stored top variants supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    top_windows : Any
        Stored top windows consumed by ``top_regions``; see those methods for the
        numerical/label contract. Required construction field.

    diagnostics : Dict[str, Any]
        Stored diagnostics supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    calibrated_states : Optional[Dict[str, FastScoreState]]
        Stored calibrated states supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Default: ``None``.
    """
    table: Any
    top_variants: Any
    top_windows: Any                 # compatibility: contains candidate regions
    diagnostics: Dict[str, Any]
    calibrated_states: Optional[Dict[str, FastScoreState]] = None

    @property
    def top_regions(self):
        """Merge top association leads into selected physical discovery intervals.

        Returns
        -------
        object
            Computed result from ``self.top_windows``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.top_windows


def trait_scan_basis(
    G: Any,
    E: Optional[Any] = None,
    *,
    rank: Optional[int] = 10,
    mode: str = "canonical",
    jitter: float = 1e-8,
) -> TraitScanBasis:
    """Low-dimensional genetic trait basis for multivariate discovery.

    Parameters
    ----------
    G : Any
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : Optional[Any]
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    rank : Optional[int]
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``10``.

    mode : str
        Input for the operation ``str(mode).lower()``. See
        ``vigor.mvgwas_fast.trait_scan_basis`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``'canonical'``.
        Source-recognized literal values: ``eigen``, ``full``, ``g``, ``ge``, ``genetic``,
        ``genetic_canonical``, ``identity``, ``none``, ``pca``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    jitter : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-08``.

    Returns
    -------
    object
        Computed result from ``TraitScanBasis(V, w, resolved, V.shape[1])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    G = _sym(_to_numpy(G, np.float64))
    t = G.shape[0]
    if G.shape != (t, t):
        raise ValueError("G must be square")
    r = t if rank is None else max(1, min(int(rank), t))
    mode = str(mode).lower()
    if mode in {"full", "identity", "none"}:
        return TraitScanBasis(np.eye(t)[:, :r], np.ones(r), "full", r)
    if mode in {"canonical", "ge", "genetic_canonical"} and E is not None:
        E = _sym(_to_numpy(E, np.float64))
        scale = max(float(np.nanmedian(np.diag(E))), 1e-8)
        try:
            w, V = scipy_eigh(G, E + float(jitter) * scale * np.eye(t), check_finite=False)
            resolved = "canonical"
        except Exception:
            w, V = np.linalg.eigh(_project_psd(G)); resolved = "genetic_fallback"
    elif mode in {"genetic", "g", "pca", "eigen"} or E is None:
        w, V = np.linalg.eigh(_project_psd(G)); resolved = "genetic"
    else:
        raise ValueError("mode must be canonical, genetic, or full")
    order = np.argsort(w)[::-1][:r]
    w = np.clip(np.asarray(w)[order], 0.0, None)
    V = np.asarray(V)[:, order]
    nrm = np.linalg.norm(V, axis=0); nrm[nrm < 1e-15] = 1.0
    V = V / nrm[None, :]
    return TraitScanBasis(V, w, resolved, V.shape[1])


def prepare_fast_score_state(
    dual_A: Any,
    G: Any,
    E: Optional[Any] = None,
    *,
    rank: Optional[int] = 10,
    basis: str = "canonical",
    trait_names: Optional[Sequence[str]] = None,
    train_mask: Optional[np.ndarray] = None,
    null_A: Optional[Any] = None,
) -> FastScoreState:
    """Project the exact n x T mvBLUP/LMM dual to r genetic axes.

    ``null_A`` (reps x n x T, e.g. from :func:`null_trait_duals`) are duals of
    simulated null traits under the same model; they are projected to the same
    axes and used by every calibration routine.

    Parameters
    ----------
    dual_A : Any
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    G : Any
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : Optional[Any]
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    rank : Optional[int]
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``10``. Passed to ``trait_scan_basis`` as
        ``rank``.

    basis : str
        Trait/feature × retained-factor loading matrix or prepared basis. Column order
        identifies factor directions and must agree with the scores used for projection.
        Default: ``'canonical'``. Passed to ``trait_scan_basis`` as ``mode``.

    trait_names : Optional[Sequence[str]]
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    train_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``FastScoreState(np.asarray(As, np.float32, order='C'), b, None, names, None if fm is None else fm.copy())``; see the summary and the called implementation for its labeled/numerical representation.
    """
    A = _to_numpy(dual_A, np.float64)
    b = trait_scan_basis(G, E, rank=rank, mode=basis)
    if A.ndim != 2 or A.shape[1] != b.vectors.shape[0]:
        raise ValueError("dual_A must be n x traits and match G/E")
    As = A @ b.vectors
    fm = None
    if train_mask is not None:
        fm = np.asarray(train_mask, bool)
        if fm.shape != (len(A),):
            raise ValueError("train_mask has wrong shape")
        As = As.copy(); As[~fm] = 0.0
    names = tuple(map(str, trait_names)) if trait_names is not None else tuple(f"factor{i+1}" for i in range(As.shape[1]))
    Nn = None
    if null_A is not None:
        N0 = np.asarray(null_A, np.float64)
        N0 = N0[None] if N0.ndim == 2 else N0
        if N0.ndim != 3 or N0.shape[1:] != A.shape:
            raise ValueError("null_A must be reps x n x traits and match dual_A")
        Nn = np.concatenate([a @ b.vectors for a in N0], axis=1)
        if fm is not None:
            Nn[~fm] = 0.0
        Nn = np.asarray(Nn, np.float32, order="C")
    return FastScoreState(np.asarray(As, np.float32, order="C"), b, None, names, None if fm is None else fm.copy(), Nn)


def _score_components_missing(X: np.ndarray, A: np.ndarray, fit_mask: Optional[np.ndarray]):
    """Evaluate components missing.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    A : np.ndarray
        Numerical operand in ``X.T @ A``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.
    """
    X = np.asarray(X, np.float32, order="C").copy()
    n, m = X.shape
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    finite = np.isfinite(X)
    cnt = finite[fit].sum(0).astype(np.int32)
    mu = np.divide(np.nansum(X[fit], 0, dtype=np.float64), cnt, out=np.zeros(m), where=cnt > 0).astype(np.float32)
    X -= mu[None, :]; X[~finite] = 0.0
    Xtr = X[fit]
    ss = np.einsum("ij,ij->j", Xtr, Xtr, optimize=True, dtype=np.float64)
    U = np.asarray(X.T @ A, np.float64)
    return U, ss, cnt


def _score_components_complete(X: np.ndarray, A: np.ndarray, fit_mask: Optional[np.ndarray]):
    """Sufficient statistics when genotype calls are known complete/imputed.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    A : np.ndarray
        Numerical operand in ``np.asarray(A)[fit]``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.
    """
    X=np.asarray(X,np.float32,order="C");n,m=X.shape
    fit=np.ones(n,bool) if fit_mask is None else np.asarray(fit_mask,bool)
    if fit.shape!=(n,) or not np.any(fit):raise ValueError("fit_mask must match genotype rows and select >=1 animal")
    Xtr=X if fit.all() else X[fit];Af=A if fit.all() else np.asarray(A)[fit]
    # n<=O(1e4) and dosage range is small; float32 accumulation is accurate
    # enough here and roughly halves memory traffic. Compact outputs are then
    # promoted to float64 for the chi-square algebra.
    ntr=max(Xtr.shape[0],1);sx=np.sum(Xtr,axis=0,dtype=np.float32).astype(np.float64);mu=sx/ntr
    s2=np.einsum("ij,ij->j",Xtr,Xtr,optimize=True,dtype=np.float32).astype(np.float64);ss=np.maximum(s2-sx*sx/ntr,0.0)
    U=np.asarray(Xtr.T@Af,np.float64);U-=mu[:,None]*np.sum(Af,axis=0,dtype=np.float64)[None,:]
    return U,ss,np.full(m,ntr,np.int32)


def _score_components(X: np.ndarray, A: np.ndarray, fit_mask: Optional[np.ndarray]):
    """Exact sufficient-statistic fast path for complete genotype blocks.

    For complete X, avoid materializing centered genotypes:
      (X-mu)'A = X'A - mu outer sum(A)
      SS        = sum(X^2) - sum(X)^2/n

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    A : np.ndarray
        Numerical operand in ``_score_components_complete(X, A, fit)``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask.
    """
    X = np.asarray(X, np.float32, order="C")
    n, m = X.shape
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if fit.shape != (n,) or not np.any(fit):
        raise ValueError("fit_mask must match genotype rows and select >=1 animal")
    if np.isfinite(X).all():
        return _score_components_complete(X,A,fit)
    return _score_components_missing(X, A, fit)


def _null_view(state: FastScoreState) -> FastScoreState:
    """The state with its null-trait duals in place of the real dual (for calibration scores)."""
    if state.null_A is None:
        raise ValueError("score state has no null-trait duals: build it with state_from_vigor_null(...) "
                         "or prepare_fast_score_state(..., null_A=null_trait_duals(null))")
    return FastScoreState(state.null_A, state.basis, None, state.trait_names, state.fit_mask)


def _calibration_from_null(U0: np.ndarray, ss: np.ndarray, rank_r: int, lambda_gc: bool = True) -> ScoreCalibration:
    """Score covariance M and lambda from null-trait scores on the calibration variants.

    U0 holds X'A0 for every replicate (m x reps*r); with Z0 = U0 / sqrt(ss),
    M = E[Z0 Z0'] over variants and replicates (null scores have mean zero) and
    lambda = median(Z0' M^-1 Z0) / chi2_r median. Both use only simulated null
    traits, so real association signal on the calibration variants cannot
    deflate the statistics.
    """
    good = np.isfinite(ss) & (ss > 0) & np.all(np.isfinite(U0), axis=1)
    reps = max(1, U0.shape[1] // int(rank_r))
    Z = (U0[good].reshape(-1, reps, rank_r) / np.sqrt(ss[good])[:, None, None]).reshape(-1, rank_r)
    M = _project_psd(Z.T @ Z / len(Z)) if len(Z) > 1 else np.eye(rank_r)
    Mi = np.linalg.pinv(M, rcond=1e-10)
    rank = max(1, int(np.linalg.matrix_rank(M)))
    Q = np.einsum("ij,jk,ik->i", Z, Mi, Z, optimize=True)
    med0 = float(chi2.ppf(.5, rank))
    lam = max(float(np.median(Q) / med0), 1e-6) if lambda_gc and len(Q) and med0 > 0 else 1.0
    return ScoreCalibration(M, Mi, rank, lam, int(good.sum()))


def _cholesky_jitter(M: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Lower Cholesky factor of a PSD matrix, adding jitter * mean(diag) until it succeeds."""
    from scipy.linalg import cholesky
    M = _sym(np.asarray(M, np.float64))
    scale = max(float(np.mean(np.diag(M))), np.finfo(float).tiny)
    eye = np.eye(len(M))
    for k in range(6):
        try:
            return cholesky(M + jitter * (10 ** k) * scale * eye, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("matrix is not positive semidefinite within jitter")


def null_trait_duals(null: Any, *, reps: int = 4, seed: int = 0, jitter: float = 1e-6) -> np.ndarray:
    """Duals P y0 of simulated null traits under an exact score null (reps x n x t).

    y0 ~ N(0, sum_c K_c (x) G_c + I (x) E) is drawn with Cholesky factors
    (K = L L' with jitter; no eigendecomposition), restricted to the null's
    observed cells and projected with the same fixed effects and V as the real
    traits. One covariance solve per replicate.

    Parameters
    ----------
    null : vigor.gwas.exact.ExactScoreNull
        Null prepared with dense kernels (``prepare_score_null``).
    reps : int
        Independent null trait sets; 4 keeps lambda within a few percent.
    seed : int
        Random seed for the draws.
    jitter : float
        Relative diagonal jitter for the Cholesky factors.
    """
    rng = np.random.default_rng(seed)
    n, t = null.y.shape
    roots = []
    for name, kernel in null.kernels.items():
        if kernel.dense is None:
            raise ValueError("null-trait calibration needs dense kernels")
        roots.append((_cholesky_jitter(kernel.dense, jitter), _cholesky_jitter(null.covariances[name], jitter)))
    Er = _cholesky_jitter(null.residual + null.jitter * np.eye(t), jitter)
    out = np.zeros((int(reps), n, t))
    for k in range(int(reps)):
        y0 = rng.standard_normal((n, t)) @ Er.T
        for Lk, Lg in roots:
            y0 += Lk @ rng.standard_normal((n, t)) @ Lg.T
        flat = np.zeros(n * t)
        flat[null.observed] = null.project(y0.ravel()[null.observed])
        out[k] = flat.reshape(n, t)
    return out


def calibrate_score_state(state: FastScoreState, calibration_genotypes: Any, *, lambda_gc: bool = True) -> FastScoreState:
    """Return the calibrate score state result computed by ``FastScoreState``.

    Parameters
    ----------
    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    calibration_genotypes : Any
        Input for the operation ``_to_numpy(calibration_genotypes, np.float32)``. See
        ``vigor.mvgwas_fast.calibrate_score_state`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    lambda_gc : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``FastScoreState(state.A, state.basis, ScoreCalibration(M, Mi, rank, lam, int(len(q))), state.trait_names, state.fit_mask)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X = _to_numpy(calibration_genotypes, np.float32)
    U0, ss, _ = _score_components(X, _null_view(state).A, state.fit_mask)
    return _replace(state, calibration=_calibration_from_null(U0, ss, state.A.shape[1], lambda_gc))


def score_variant_block(X: Any, state: FastScoreState, *, return_factor_z: bool = False, assume_complete: bool = False) -> Dict[str, np.ndarray]:
    """Score one animals x variants block with one principal GEMM.

    Parameters
    ----------
    X : Any
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    return_factor_z : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    assume_complete : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if state.calibration is None:
        raise ValueError("state is not calibrated")
    X = _to_numpy(X, np.float32)
    if X.ndim != 2 or X.shape[0] != state.A.shape[0]:
        raise ValueError("X rows must match score state")
    U, ss, cnt = (_score_components_complete(X,state.A,state.fit_mask) if assume_complete else _score_components(X, state.A, state.fit_mask))
    root = np.sqrt(ss)
    Z = np.divide(U, root[:, None], out=np.zeros_like(U), where=root[:, None] > 0)
    cal = state.calibration
    Q = np.einsum("ij,jk,ik->i", Z, cal.inverse, Z, optimize=True) / float(cal.lambda_gc)
    p = chi2.sf(Q, cal.rank)
    out = {"stat": Q, "p": p, "neglog_p": -np.log10(np.clip(p, np.finfo(float).tiny, 1.0)), "nobs": cnt, "ss": ss}
    if return_factor_z:
        den = np.sqrt(np.clip(np.diag(cal.covariance)[None, :] * ss[:, None] * float(cal.lambda_gc), 1e-30, None))
        out["factor_z"] = np.divide(U, den, out=np.zeros_like(U), where=den > 0)
    return out


def open_genotype_xarray(
    genotypes: Any,
    *,
    npplink_module=None,
    io_chunk_variants: Any = "auto",
    dtype=np.float32,
    use_memmap: bool = True,
    sample_ids: Optional[Sequence[Any]] = None,
    sample_indices: Optional[Sequence[int]] = None,
    maxmem=None,
    memory_fraction=None,
    workers=1,
):
    """Open a lazy genotype view with an integer, memory-bounded I/O chunk.

    Resolve chunks before PLINK/PGEN reader dispatch. Source sample counts come
    from .fam/.psam metadata; opening never decodes genotype values. Packed BED
    dosage remains virtual, and Dask-backed arrays are rechunked lazily.

    Parameters
    ----------
    genotypes : xarray.DataArray, xarray.Dataset, str or pathlib.Path
        Genotypes with iid (sample) and snp (variant) dimensions, a Dataset
        containing genotypes or one variable, or a PLINK1/PLINK2 fileset prefix
        (optionally ending in .bed/.bim/.fam/.pgen/.pvar/.psam).
    npplink_module : module or None
        Default None uses the package npplink reader. Alternate adapters can
        supply packed PLINK, PLINK or PGEN xarray readers with integer chunks.
    io_chunk_variants : int, 'auto' or None
        Default 'auto'. Auto and None derive a positive integer from the source
        sample count and memory budget. Explicit positive integers are ceilings;
        every choice is capped by available working memory.
        Invalid strings, booleans, fractional and nonpositive values are rejected.
    dtype : NumPy dtype or str
        Default np.float32. Requested ordinary-reader dosage dtype and working
        buffer cost (at least 16 bytes/sample/variant). Packed virtual dosage
        retains its native reader dtype.
    use_memmap : bool
        Default True. Forwarded to ordinary adapters that support file-backed
        arrays; packed BED already uses its native virtual storage.
    sample_ids : sequence or None
        Default None retains source sample order. Otherwise select iid labels
        in this order on every backend; missing IDs raise ValueError.
    sample_indices : sequence of int or None
        Default None. Positional iid selection in requested order; mutually
        exclusive with sample_ids. Indices must be valid for the source axis.
    maxmem : int, float, unit string or None
        Default None uses available RAM times memory_fraction. Positive byte
        counts or B/KB/KiB/MB/MiB/GB/GiB/TB/TiB strings use the existing binary
        resource convention: 8GB and 8GiB both mean 8 * 1024**3 bytes. Explicit
        budgets are capped at 80% of available RAM and shared across workers.
        If one working variant cannot fit, raise MemoryError.
    memory_fraction : float or None
        Available RAM fraction for automatic planning, in (0, .95]. Default: ``None`` (80% of physical RAM).
    workers : int
        Positive concurrent worker count sharing the budget. Default: 1.

    Returns
    -------
    xarray.DataArray
        Lazy dosage with iid/snp axes, requested sample order, source metadata,
        and truthful io_chunk_variants, io_chunk_policy (memory_auto or
        memory_bounded_explicit) and io_memory_budget_bytes attributes.
    """
    from .resources import bounded_variant_chunk, working_memory_bytes
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if sample_ids is not None and sample_indices is not None:
        raise ValueError('supply sample_ids or sample_indices, not both')
    automatic = io_chunk_variants is None or (isinstance(io_chunk_variants, str)
                                               and io_chunk_variants.lower() == 'auto')
    if not automatic:
        if isinstance(io_chunk_variants, (bool, np.bool_)) or not isinstance(io_chunk_variants, (int, np.integer)) or io_chunk_variants < 1:
            raise ValueError("io_chunk_variants must be a positive integer, 'auto', or None")
    # Validate resource options even for pre-opened views; never ignore maxmem.
    budget = working_memory_bytes(maxmem, memory_fraction, workers)
    requested = 'auto' if automatic else int(io_chunk_variants)
    xr = _require_xarray()
    if isinstance(genotypes, (str, Path)) and (str(genotypes).rstrip("/\\").endswith(".zarr")
                                               or (Path(genotypes).is_dir() and (Path(genotypes) / "zarr.json").exists())):
        # VIGOR Zarr genotype store: canonical lazy sample x variant view, no PLINK reader.
        from .data_model import open_genotype_dataset
        genotypes = open_genotype_dataset(str(genotypes), source_format="zarr", maxmem=maxmem,
                                          memory_fraction=memory_fraction)
    if isinstance(genotypes, xr.DataArray):
        g = genotypes
    elif isinstance(genotypes, xr.Dataset):
        name = next((v for v in ("genotypes", "GT", "dosage", "DS") if v in genotypes.data_vars), None)
        if name is not None:
            g = genotypes[name]
        elif len(genotypes.data_vars) == 1:
            g = genotypes[next(iter(genotypes.data_vars))]
        else:
            raise ValueError("genotype Dataset must contain one genotype variable (genotypes, GT, dosage or DS)")
    elif isinstance(genotypes, (str, Path)):
        if npplink_module is None:
            try:
                from . import npplink as npplink_module  # type: ignore
            except Exception:
                import npplink as npplink_module  # type: ignore
        src = str(genotypes)
        if src.lower().endswith((".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam")):
            src = str(Path(src).with_suffix(""))
        pgen = Path(src + ".pgen").exists()
        metadata = Path(src + ('.psam' if pgen else '.fam'))
        with metadata.open() as handle:
            n_samples = sum(1 for line in handle if line.strip() and not line.startswith('#'))
        io_chunk_variants = bounded_variant_chunk(
            n_samples, requested=requested, maxmem=budget, memory_fraction=memory_fraction,
            bytes_per_call=max(16, np.dtype(dtype).itemsize * 4), alignment=1)

        names = (["load_pgen_xarray", "load_plink2_xarray", "load_genotypes_xarray", "load_plink_xarray"] if pgen
                 else ["load_plink_xarray", "load_genotypes_xarray", "load_pgen_xarray", "load_plink2_xarray"])
        last = None; g = None
        if not pgen and hasattr(npplink_module,'load_plink_packed_xarray'):
            packed=npplink_module.load_plink_packed_xarray(src,chunk_variants=io_chunk_variants)
            g=packed.domino_genomics.dosage()
            g.attrs['domino_reader']='npplink.load_plink_packed_xarray virtual dosage'
        for name in ([] if g is not None else names):
            fn = getattr(npplink_module, name, None)
            if fn is None:
                continue
            try:
                sig = inspect.signature(fn); kw = {}
                for key, val in {"prefix": src, "path": src, "plinkfile": src,
                                 "chunk_variants": io_chunk_variants, "dtype": dtype,
                                 "use_memmap": bool(use_memmap), "sample_ids": sample_ids,
                                 "maxmem": budget}.items():
                    if key in sig.parameters and val is not None:
                        kw[key] = val
                out = fn(**kw)
                if isinstance(out, xr.DataArray): g = out
                elif isinstance(out, xr.Dataset) and "genotypes" in out: g = out["genotypes"]
                if g is not None:
                    if "cm" not in g.coords and "genetic_cm" not in g.coords:
                        try:
                            reader = getattr(npplink_module, "read_fam_bim", None)
                            if reader is not None:
                                fb = reader(src)
                                bim = fb[1] if isinstance(fb, tuple) and len(fb) >= 2 else None
                                if bim is not None and "cm" in bim and len(bim) == g.sizes["snp"]:
                                    g = g.assign_coords(genetic_cm=("snp", np.asarray(bim["cm"], np.float64)))
                        except Exception:
                            pass
                    break
            except Exception as exc:
                last = exc
        if g is None:
            raise RuntimeError(f"no npplink xarray reader could open {src!r}") from last
    else:
        raise TypeError("genotypes must be xarray or PLINK/PGEN path")
    g = _canonical_genotype_view(g)
    if "iid" not in g.dims or "snp" not in g.dims:
        raise ValueError("genotype xarray must use iid/snp or sample/variant dimensions")
    if not isinstance(genotypes, (str, Path)):
        io_chunk_variants = bounded_variant_chunk(
            g.sizes['iid'], requested=requested, maxmem=budget,
            memory_fraction=memory_fraction, bytes_per_call=max(16, np.dtype(dtype).itemsize * 4), alignment=1)
    # Align every backend identically; repeated alignment is harmless for ports
    # that already honored the requested order. No genotype values are read.
    if sample_ids is not None:
        g = _align_samples(g, sample_ids)
    elif sample_indices is not None:
        g = g.isel(iid=np.asarray(sample_indices, np.int64))
    # Inspect the wrapper only: DataArray.data materializes BackendArray dosage.
    data = g.variable._data
    if str(g.attrs.get('physical_backend', '')) != 'plink1_bed_2bit' and getattr(data, 'chunks', None) is not None:
        g = g.chunk({'iid': -1, 'snp': io_chunk_variants})
    g.attrs = dict(g.attrs, io_chunk_variants=int(io_chunk_variants),
                   io_chunk_policy='memory_auto' if automatic else 'memory_bounded_explicit',
                   io_memory_budget_bytes=budget)
    return g


def _canonical_genotype_view(g):
    """iid/snp dimensions, chrom/pos coordinates and object-string labels for any genotype view.

    Accepts the VIGOR Zarr/data-model names (sample, variant, chromosome,
    position, variant_id) as well as npplink's (iid, snp, chrom, pos). Zarr v3
    StringDType/bytes coordinates become object str arrays. Only coordinate
    metadata is touched; genotype values stay lazy.
    """
    rename = {d: a for d, a in (("sample", "iid"), ("variant", "snp")) if d in g.dims and a not in g.dims}
    if rename:
        g = g.rename(rename)
    extra = {}
    for src, dst in (("chromosome", "chrom"), ("position", "pos")):
        if src in g.coords and dst not in g.coords:
            extra[dst] = (g.coords[src].dims, g.coords[src].values)
    if "variant_id" in g.coords and g.coords["variant_id"].dims == ("snp",):
        extra["snp"] = g.coords["variant_id"].values
    if extra:
        g = g.assign_coords(extra)
    for name in list(g.coords):
        values = np.asarray(g.coords[name].values)
        if values.dtype.kind in "OSUT" or "String" in type(values.dtype).__name__:
            g = g.assign_coords({name: (g.coords[name].dims, np.array([str(v) for v in values], dtype=object))})
    return g


def _align_samples(g, sample_ids):
    """Align samples.

    Parameters
    ----------
    g : object (see the cited input contract)
        Input for the operation ``g.coords['iid']``. See ``vigor.mvgwas_fast._align_samples``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
    """
    if sample_ids is None:
        return g
    ids = np.asarray(g.coords["iid"].values).astype(str)
    wanted = np.asarray(list(map(str, sample_ids)), dtype=object)
    loc = {x: i for i, x in enumerate(ids)}
    missing = [x for x in wanted if x not in loc]
    if missing:
        raise ValueError(f"{len(missing)} requested sample IDs absent from genotypes; first={missing[:3]}")
    return g.isel(iid=np.asarray([loc[x] for x in wanted], np.int64))

# compatibility with the previous development file
_optimized_align_samples = _align_samples


def _chrom_indices(g, chromosome: str) -> np.ndarray:
    """Select zero-based variant positions belonging to the requested chromosome.

    Parameters
    ----------
    g : object (see the cited input contract)
        Input for the operation ``g.sizes['snp']``. See ``vigor.mvgwas_fast._chrom_indices`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    chromosome : str
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    if "chrom" not in g.coords:
        return np.arange(g.sizes["snp"], dtype=np.int64)
    c = np.asarray(g.coords["chrom"].values).astype(str)
    return np.flatnonzero(c == str(chromosome))


def _even_calibration_indices(g, chromosomes, n_calibration: int) -> np.ndarray:
    """Choose deterministic evenly spaced variant indices for score calibration.

    Parameters
    ----------
    g : object (see the cited input contract)
        Input for the operation ``g.coords['chrom']``. See
        ``vigor.mvgwas_fast._even_calibration_indices`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine.

    n_calibration : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
    """
    if "chrom" in g.coords and chromosomes is not None:
        c = np.asarray(g.coords["chrom"].values).astype(str)
        mask = np.isin(c, np.asarray(list(map(str, chromosomes))))
        idx = np.flatnonzero(mask)
    else:
        idx = np.arange(g.sizes["snp"], dtype=np.int64)
    if len(idx) <= int(n_calibration):
        return idx
    return idx[np.unique(np.linspace(0, len(idx)-1, int(n_calibration), dtype=np.int64))]


def _chrom_calibration_indices(idx, n_calibration, io_chunk_variants, n_io_windows=4):
    """Sample calibration SNPs from a few distributed storage chunks.

    Evenly scattered SNPs can force Dask to decode essentially every PLINK/PGEN
    storage chunk before the real scan. This keeps calibration representative
    across the chromosome while touching only a small number of large chunks.

    Parameters
    ----------
    idx : object (see the cited input contract)
        Input for the operation ``idx // io``. See
        ``vigor.mvgwas_fast._chrom_calibration_indices`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    n_calibration : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.

    io_chunk_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size.

    n_io_windows : int
        Input for the operation ``max(1, min(int(n_io_windows), len(chunks), n))``. See
        ``vigor.mvgwas_fast._chrom_calibration_indices`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``4``.
    """
    idx=np.asarray(idx,np.int64);n=min(int(n_calibration),len(idx))
    if len(idx)<=n:return idx
    io=max(1,int(io_chunk_variants));labels=idx//io;chunks=np.unique(labels);nw=max(1,min(int(n_io_windows),len(chunks),n))
    chosen=chunks[np.unique(np.linspace(0,len(chunks)-1,nw,dtype=np.int64))];per=int(math.ceil(n/max(len(chosen),1)));parts=[]
    for ch in chosen:
        cand=idx[labels==ch]
        if len(cand):parts.append(cand[np.unique(np.linspace(0,len(cand)-1,min(per,len(cand)),dtype=np.int64))])
    out=np.unique(np.concatenate(parts)) if parts else idx[:n]
    if len(out)>n:out=out[np.unique(np.linspace(0,len(out)-1,n,dtype=np.int64))]
    return out


def _xarray_small_compute(ds):
    """Materialize only the bounded xarray block required by the scan.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.
    """
    try:
        return ds.compute()
    except Exception:
        return ds


def _packed_bed_score_components_xarray(block, state: FastScoreState):
    """Exact score sufficient statistics from virtual packed-BED xarray blocks.

    Returns ``None`` for non-packed sources.  For PLINK1 hard calls the BED
    bytes are reduced directly to U=X'A, nobs, sums, and sum-of-squares using
    precomputed 256-state byte tables; no iid x SNP dosage array is decoded.

    Parameters
    ----------
    block : object (see the cited input contract)
        Input for the operation ``block.coords['variant_index']``. See
        ``vigor.mvgwas_fast._packed_bed_score_components_xarray`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    attrs=getattr(block,"attrs",{})
    if str(attrs.get("physical_backend","")) != "plink1_bed_2bit":
        return None
    if "variant_index" not in block.coords or "sample_index" not in block.coords:
        return None
    prefix=attrs.get("prefix")
    if not prefix:
        return None
    vi=np.asarray(block.coords["variant_index"].values,np.int64).reshape(-1)
    si=np.asarray(block.coords["sample_index"].values,np.int64).reshape(-1)
    if len(si)!=state.A.shape[0]:
        raise ValueError("packed BED sample_index does not match score-state rows")
    nfull=int(attrs.get("n_full_samples", int(si.max())+1 if len(si) else 0))
    mfull=int(attrs.get("n_full_variants", int(vi.max())+1 if len(vi) else 0))
    bpv=(nfull+3)//4
    from .packed_bed import build_packed_score_tables, packed_score_components_fast
    # Cache the A-dependent byte tables on the chromosome-specific score state.
    # The key includes physical sample mapping and fit mask; variant selection does
    # not affect the tables, so every scan block reuses them.
    fit=np.ones(len(si),bool) if state.fit_mask is None else np.asarray(state.fit_mask,bool)
    key=(str(prefix),nfull,hash(si.tobytes()),hash(fit.tobytes()),state.A.shape,
         hashlib.blake2b(np.ascontiguousarray(state.A).view(np.uint8),digest_size=16).digest())
    cache=getattr(_packed_bed_score_components_xarray,"_cache",None)
    if cache is None:
        cache={}; setattr(_packed_bed_score_components_xarray,"_cache",cache)
    tables=cache.get(key)
    if tables is None:
        tables=build_packed_score_tables(state.A,nfull,sample_indices=si,fit_mask=fit,dtype=np.float32)
        # Score states are immutable; keep a small process-local table cache instead.
        if len(cache)>=32: cache.pop(next(iter(cache)))
        cache[key]=tables
    mm=open_bed_memmap(str(prefix)+".bed", n_variants=mfull)
    P=np.asarray(mm[vi],np.uint8)
    return packed_score_components_fast(P,tables)


def _xarray_score_components_complete(block, state: FastScoreState):
    """Lazy sufficient statistics for complete/imputed genotype blocks.

    Only O(m*r) score output is materialized.  The n x m genotype block stays
    lazy when backed by Dask/xarray.

    Parameters
    ----------
    block : object (see the cited input contract)
        Input for the operation ``block.sizes['iid']``. See
        ``vigor.mvgwas_fast._xarray_score_components_complete`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    xr = _require_xarray()
    if "iid" not in block.dims or "snp" not in block.dims:
        raise ValueError("genotype block must have iid and snp dimensions")
    A = state.A
    if block.sizes["iid"] != A.shape[0]:
        raise ValueError("genotype rows do not match score state")
    fit = np.ones(A.shape[0], bool) if state.fit_mask is None else np.asarray(state.fit_mask, bool)
    rows = np.flatnonzero(fit)
    X = block.isel(iid=rows).astype(np.float32)
    Af = A[rows]
    if not (hasattr(X.data,"chunks") and X.data.chunks is not None):
        return _score_components_complete(np.asarray(X.data,np.float32),Af,None)
    Ada = xr.DataArray(Af, dims=("iid", "factor"), coords={"iid": X.coords["iid"], "factor": np.arange(Af.shape[1])})
    sx = X.sum("iid", dtype=np.float64)
    s2 = (X * X).sum("iid", dtype=np.float64)
    Uraw = xr.dot(X, Ada, dim="iid")
    small = xr.Dataset({"sx": sx, "s2": s2, "Uraw": Uraw})
    small = _xarray_small_compute(small)
    sxv = np.asarray(small["sx"].values, np.float64)
    s2v = np.asarray(small["s2"].values, np.float64)
    U = np.asarray(small["Uraw"].values, np.float64)
    n = max(len(rows), 1)
    mu = sxv / n
    ss = np.maximum(s2v - sxv * sxv / n, 0.0)
    U -= mu[:, None] * np.sum(Af, axis=0, dtype=np.float64)[None, :]
    cnt = np.full(len(sxv), n, np.int32)
    return U, ss, cnt


def _xarray_score_components_missing(block, state: FastScoreState):
    """Exact lazy sufficient statistics with arbitrary per-call missingness.

    Missing genotypes are excluded variant-by-variant.  The only materialized
    objects are O(m) counts/sums and O(m*r) score terms; the iid x variant
    genotype block stays Dask/xarray-backed.

    Parameters
    ----------
    block : array-like or numeric scalar
        Input for the operation ``block.isel(iid=rows).astype(np.float32)``. See
        ``vigor.mvgwas_fast._xarray_score_components_missing`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    xr = _require_xarray()
    A = state.A
    fit = np.ones(A.shape[0], bool) if state.fit_mask is None else np.asarray(state.fit_mask, bool)
    rows = np.flatnonzero(fit)
    X = block.isel(iid=rows).astype(np.float32)
    Af = A[rows]
    if not (hasattr(X.data, "chunks") and X.data.chunks is not None):
        return _score_components_missing(np.asarray(X.data, np.float32), Af, None)
    Ada = xr.DataArray(Af, dims=("iid", "factor"), coords={"iid": X.coords["iid"], "factor": np.arange(Af.shape[1])})
    M = X.notnull().astype(np.float32)
    X0 = X.fillna(0.0).astype(np.float32)
    cnt = M.sum("iid", dtype=np.float32)
    sx = X0.sum("iid", dtype=np.float32)
    s2 = (X0 * X0).sum("iid", dtype=np.float32)
    Uraw = xr.dot(X0, Ada, dim="iid")
    Asum = xr.dot(M, Ada, dim="iid")
    small = _xarray_small_compute(xr.Dataset({"cnt": cnt, "sx": sx, "s2": s2, "Uraw": Uraw, "Asum": Asum}))
    cntv = np.asarray(small["cnt"].values, np.int32)
    sxv = np.asarray(small["sx"].values, np.float64)
    s2v = np.asarray(small["s2"].values, np.float64)
    U = np.asarray(small["Uraw"].values, np.float64)
    As = np.asarray(small["Asum"].values, np.float64)
    mu = np.divide(sxv, cntv, out=np.zeros_like(sxv), where=cntv > 0)
    ss = np.maximum(s2v - np.divide(sxv * sxv, cntv, out=np.zeros_like(sxv), where=cntv > 0), 0.0)
    U -= mu[:, None] * As
    return U, ss, cntv


def score_xarray_block_lazy(
    block: Any,
    state: Any,
    *,
    return_factor_z: bool = False,
    assume_complete: Any = "auto",
) -> Dict[str, np.ndarray]:
    """Score an xarray genotype block without materializing n x m when possible.

    ``assume_complete=True`` is the fast path for imputed PLINK/PGEN dosages.
    Set it False for raw genotype files that can contain missing calls; those
    blocks use the exact NumPy missing-data path.

    Parameters
    ----------
    block : Any
        Input for the operation ``block['genotypes']``. See
        ``vigor.mvgwas_fast.score_xarray_block_lazy`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    state : Any
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    return_factor_z : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``. Passed to ``score_variant_block`` as
        ``return_factor_z``.

    assume_complete : Any
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``'auto'``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr = _require_xarray()
    if state.calibration is None:
        raise ValueError("state is not calibrated")
    if isinstance(block, xr.Dataset):
        if "genotypes" in block: block = block["genotypes"]
        elif len(block.data_vars) == 1: block = block[next(iter(block.data_vars))]
        else: raise ValueError("genotype Dataset must contain one genotype variable")
    if not isinstance(block, xr.DataArray):
        arr=np.asarray(block,np.float32)
        complete=bool(np.isfinite(arr).all()) if isinstance(assume_complete,str) and assume_complete.lower()=="auto" else bool(assume_complete)
        return score_variant_block(arr, state, return_factor_z=return_factor_z, assume_complete=complete)
    packed=_packed_bed_score_components_xarray(block,state)
    if packed is not None:
        U,ss,cnt=packed
        root=np.sqrt(ss); Z=np.divide(U,root[:,None],out=np.zeros_like(U),where=root[:,None]>0)
        cal=state.calibration; Q=np.einsum("ij,jk,ik->i",Z,cal.inverse,Z,optimize=True)/float(cal.lambda_gc)
        p=chi2.sf(Q,cal.rank)
        out={"stat":Q,"p":p,"neglog_p":-np.log10(np.clip(p,np.finfo(float).tiny,1.0)),"nobs":cnt,"ss":ss}
        if return_factor_z:
            den=np.sqrt(np.clip(np.diag(cal.covariance)[None,:]*ss[:,None]*float(cal.lambda_gc),1e-30,None))
            out["factor_z"]=np.divide(U,den,out=np.zeros_like(U),where=den>0)
        return out
    # Eager xarray is only a labeled NumPy view; BLAS on the underlying array
    # is substantially faster than routing reductions through xarray.  The
    # lazy sufficient-statistic path is reserved for Dask-backed genotype data.
    if not (hasattr(block.data, "chunks") and block.data.chunks is not None):
        arr=np.asarray(block.data,np.float32)
        complete=bool(np.isfinite(arr).all()) if isinstance(assume_complete,str) and assume_complete.lower()=="auto" else bool(assume_complete)
        return score_variant_block(arr, state, return_factor_z=return_factor_z, assume_complete=complete)
    if isinstance(assume_complete,str) and assume_complete.lower()=="auto":
        # Safe automatic mode: if completeness is not explicitly declared by the
        # source, use the exact lazy missing-aware sufficient statistics in one graph.
        complete=(block.attrs.get("genotype_complete",None) is True)
    else:
        complete=bool(assume_complete)
    if complete:
        U, ss, cnt = _xarray_score_components_complete(block, state)
    else:
        U, ss, cnt = _xarray_score_components_missing(block, state)
    root = np.sqrt(ss)
    Z = np.divide(U, root[:, None], out=np.zeros_like(U), where=root[:, None] > 0)
    cal = state.calibration
    Q = np.einsum("ij,jk,ik->i", Z, cal.inverse, Z, optimize=True) / float(cal.lambda_gc)
    p = chi2.sf(Q, cal.rank)
    out = {"stat": Q, "p": p, "neglog_p": -np.log10(np.clip(p, np.finfo(float).tiny, 1.0)), "nobs": cnt, "ss": ss}
    if return_factor_z:
        den = np.sqrt(np.clip(np.diag(cal.covariance)[None, :] * ss[:, None] * float(cal.lambda_gc), 1e-30, None))
        out["factor_z"] = np.divide(U, den, out=np.zeros_like(U), where=den > 0)
    return out

def calibrate_xarray_score_state(
    genotypes: Any,
    state: FastScoreState,
    *,
    chromosomes: Optional[Sequence[Any]] = None,
    sample_ids: Optional[Sequence[Any]] = None,
    n_calibration: int = 4096,
    io_chunk_variants: Any = "auto",
    calibration_io_windows: int = 4,
    assume_complete: Any = "auto",
    maxmem=None,
    memory_fraction=None,
) -> FastScoreState:
    """Calibrate score covariance while touching only a few storage chunks.

    Parameters
    ----------
    genotypes : Any
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    state : FastScoreState
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    chromosomes : Optional[Sequence[Any]]
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    sample_ids : Optional[Sequence[Any]]
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``open_genotype_xarray`` as ``sample_ids``.

    n_calibration : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4096``.

    io_chunk_variants : int
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``'auto'`` (sized from the working-memory budget;
        integers are ceilings). Passed to ``open_genotype_xarray`` as ``io_chunk_variants``.

    calibration_io_windows : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4``.

    assume_complete : Any
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``'auto'``.

    maxmem, memory_fraction : budget or None
        Working-memory budget for the automatic I/O chunk; see ``open_genotype_xarray``.
        Default: ``None`` (80% of physical RAM, capped at available RAM).

    Returns
    -------
    object
        Computed result from ``calibrate_score_state(state, _xarray_compute(block, np.float32))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    g = open_genotype_xarray(genotypes,io_chunk_variants=io_chunk_variants,sample_ids=sample_ids,maxmem=maxmem,memory_fraction=memory_fraction)
    g = _align_samples(g, sample_ids)
    if "chrom" in g.coords and chromosomes is not None:
        cc=np.asarray(g.coords["chrom"].values).astype(str);idx=np.flatnonzero(np.isin(cc,np.asarray(list(map(str,chromosomes)))))
        _io=int(g.attrs.get("io_chunk_variants", g.variable._data.chunks[1][0] if getattr(g.variable._data,"chunks",None) else 16384))
        ci=_chrom_calibration_indices(idx,int(n_calibration),_io,int(calibration_io_windows))
    else:
        idx=np.arange(g.sizes["snp"],dtype=np.int64); _io=int(g.attrs.get("io_chunk_variants", g.variable._data.chunks[1][0] if getattr(g.variable._data,"chunks",None) else 16384)); ci=_chrom_calibration_indices(idx,int(n_calibration),_io,int(calibration_io_windows))
    block = g.isel(snp=ci)
    nv = _null_view(state)
    r = state.A.shape[1]
    packed=_packed_bed_score_components_xarray(block,nv)
    if packed is not None:
        U0,ss,_=packed
        return _replace(state, calibration=_calibration_from_null(U0, ss, r))
    complete=(block.attrs.get("genotype_complete",None) is True) if isinstance(assume_complete,str) and assume_complete.lower()=="auto" else bool(assume_complete)
    if complete:
        U0, ss, _ = _xarray_score_components_complete(block, nv)
        return _replace(state, calibration=_calibration_from_null(U0, ss, r))
    if isinstance(assume_complete,str) and assume_complete.lower()=="auto":
        U0,ss,_=_xarray_score_components_missing(block,nv)
        return _replace(state, calibration=_calibration_from_null(U0, ss, r))
    return calibrate_score_state(state, _xarray_compute(block, np.float32))


def _chrom_dataset(chromosome, names, pos, vidx, stat, p, nlp, nobs, factor_z, state, genetic_cm=None):
    """Subset the lazy genotype Dataset to one chromosome while preserving coordinates.

    Parameters
    ----------
    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Passed to ``xr.Dataset`` as ``attrs``.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    vidx : one-dimensional integer array or None
        Zero-based variant positions to select from the current source. Values must refer to
        existing columns; None retains all eligible variants. Streaming readers may additionally
        require sorted indices.

    stat : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset.

    p : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations.

    nlp : array-like or numeric scalar
        Input for the operation ``np.asarray(nlp, np.float32)``. See
        ``vigor.mvgwas_fast._chrom_dataset`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    nobs : array-like
        Pairwise weights or jointly observed counts, aligned to the trait/variant-pair axes of
        the associated covariance/LD matrix. Missing-data precision calculations must use the
        actual shared observations.

    factor_z : array-like or numeric scalar
        Input for the operation ``np.asarray(factor_z, np.float32)``. See
        ``vigor.mvgwas_fast._chrom_dataset`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes. Passed to ``xr.Dataset`` as ``attrs``.

    genetic_cm : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.
    """
    xr = _require_xarray()
    data = {
        "omnibus_stat": ("variant", np.asarray(stat, np.float32)),
        "omnibus_p": ("variant", np.asarray(p, np.float64)),
        "neglog_p": ("variant", np.asarray(nlp, np.float32)),
        "nobs": ("variant", np.asarray(nobs, np.int32)),
        "exact_rescored": ("variant", np.zeros(len(stat), bool)),
    }
    coords = {
        "variant": np.arange(len(stat), dtype=np.int64),
        "snp": ("variant", np.asarray(names).astype(str)),
        "chrom": ("variant", np.asarray([str(chromosome)] * len(stat), dtype=str)),
        "pos": ("variant", np.asarray(pos, np.int64)),
        "variant_index": ("variant", np.asarray(vidx, np.int64)),
    }
    if genetic_cm is not None:
        coords["genetic_cm"] = ("variant", np.asarray(genetic_cm, np.float64))
    if factor_z is not None:
        fz = np.asarray(factor_z, np.float32)
        data["factor_z"] = (("variant", "factor"), fz)
        coords["factor"] = np.asarray([f"genetic_factor_{i+1}" for i in range(fz.shape[1])]).astype(str)
    ds=xr.Dataset(data, coords=coords, attrs={
        "domino_schema":"mvGWAS_chromosome_scan","domino_schema_version":"1.0",
        "chromosome": str(chromosome), "score_basis": state.basis.mode,
        "score_rank": int(state.basis.rank), "score_dof": int(state.calibration.rank),
        "lambda_gc": float(state.calibration.lambda_gc),
        "calibration_variants": int(state.calibration.n_variants),
        "factor_eigenvalues": np.asarray(state.basis.eigenvalues, float).tolist(),
        "missingness_semantics":"nobs is variant-specific; missing genotype calls are excluded when assume_complete=False",
    })
    annotations={
        "omnibus_stat":("Multivariate omnibus score statistic","Calibrated chi-square-like score across retained genetic trait factors"),
        "omnibus_p":("Multivariate omnibus p-value","Survival probability using calibrated score degrees of freedom"),
        "neglog_p":("-log10 omnibus p-value","Negative base-10 logarithm of omnibus_p"),
        "nobs":("Observed genotype count","Number of phenotype-training animals with a non-missing genotype call for this variant"),
        "exact_rescored":("Exact rescore flag","True when the compact discovery statistic was replaced by exact locus rescoring"),
        "factor_z":("Genetic-factor score z","Per-variant standardized score on retained genetic trait axes"),
    }
    for name,(long_name,description) in annotations.items():
        if name in ds: ds[name].attrs.update({"long_name":long_name,"description":description})
    for name,desc in {"snp":"Variant identifier","chrom":"Chromosome label","pos":"Physical base-pair position","variant_index":"Zero-based index in the source genotype matrix","genetic_cm":"Genetic-map position in centimorgans","factor":"Retained genetic trait-factor label"}.items():
        if name in ds.coords: ds.coords[name].attrs["description"]=desc
    return ds


def _candidate_positions(ds, p_threshold, top_k):
    """Return the candidate positions result computed by ``np.asarray``.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy.
    """
    p = np.asarray(ds["omnibus_p"].values, float)
    take = np.flatnonzero(np.isfinite(p) & (p <= float(p_threshold))) if p_threshold is not None else np.array([], np.int64)
    finite = np.flatnonzero(np.isfinite(p)); k = min(max(0, int(top_k)), len(finite))
    if k:
        take = np.unique(np.r_[take, finite[np.argpartition(p[finite], k-1)[:k]]])
    return np.asarray(take, np.int64)


def _merge_exact(ds, exact_ds):
    """Merge exact.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    exact_ds : object (see the cited input contract)
        Input for the operation ``exact_ds.coords[ecoord]``. See
        ``vigor.mvgwas_fast._merge_exact`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.
    """
    xr = _require_xarray()
    if exact_ds is None:
        return ds
    if not isinstance(exact_ds, xr.Dataset):
        raise TypeError("exact_rescore_fn must return an xarray.Dataset")
    ecoord = "exact_variant_index" if "exact_variant_index" in exact_ds.coords else ("variant_index" if "variant_index" in exact_ds.coords else None)
    if ecoord is None:
        raise ValueError("exact rescore Dataset must expose exact_variant_index or variant_index")
    eidx = np.asarray(exact_ds.coords[ecoord].values, np.int64)
    target = np.asarray(ds["variant_index"].values, np.int64)
    lookup = {int(v): i for i, v in enumerate(target)}
    out = ds.copy(); flag = np.asarray(out["exact_rescored"].values, bool).copy()
    for j, v in enumerate(eidx):
        i = lookup.get(int(v))
        if i is None: continue
        flag[i] = True
        for src, dst in (("exact_omnibus_stat","omnibus_stat"),("exact_omnibus_p","omnibus_p"),("exact_neglog_p","neglog_p")):
            if src in exact_ds:
                a = np.asarray(out[dst].values).copy(); a[i] = np.asarray(exact_ds[src].values)[j]; out[dst] = ("variant", a)
    out["exact_rescored"] = ("variant", flag)
    return out


def _top_variants(ds, top_k, p_threshold):
    """Retain the strongest association variants under the configured top-count cutoff.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy.

    p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
    """
    xr = _require_xarray()
    if ds.sizes.get("variant", 0) == 0:
        return xr.Dataset()
    p = np.asarray(ds["omnibus_p"].values, float); stat = np.asarray(ds["omnibus_stat"].values, float)
    take = _candidate_positions(ds, p_threshold, top_k)
    if not len(take):
        return ds.isel(variant=slice(0, 0))
    take = take[np.lexsort((-np.nan_to_num(stat[take], nan=-np.inf), np.nan_to_num(p[take], nan=np.inf)))]
    return ds.isel(variant=take)


def candidate_regions_from_scan(
    ds: Any,
    *,
    lead_p: float = 1e-5,
    padding_bp: int = 1_000_000,
    recombination_rate_cM_per_mb: float = 0.5,
    merge_recombination_cM: Optional[float] = 1.0,
    max_regions: Optional[int] = None,
    fallback_top_k: int = 0,
):
    """All +/-padding loci around significant leads, merged recombination-aware.

    Parameters
    ----------
    ds : Any
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    lead_p : float
        Input supplied to ``xr.Dataset`` as ``attrs``. See that callable's documented ``attrs``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``1e-05``. Passed to ``xr.Dataset`` as ``attrs``.

    padding_bp : int
        Input supplied to ``xr.Dataset`` as ``attrs``. See that callable's documented ``attrs``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``1000000``. Passed to ``xr.Dataset`` as ``attrs``.

    recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``. Passed to ``xr.Dataset`` as ``attrs``.

    merge_recombination_cM : Optional[float]
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``1.0``. Passed to ``xr.Dataset`` as ``attrs``.

    max_regions : Optional[int]
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``None``.

    fallback_top_k : int
        Input for the operation ``int(fallback_top_k) > 0``. See
        ``vigor.mvgwas_fast.candidate_regions_from_scan`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``0``.
        Source comparisons: ``int(fallback_top_k) > 0``. See the rejection/fallback branches for
        enforcement.

    Returns
    -------
    object
        Computed result from ``xr.Dataset(data_vars={'lead_p': ('region', np.asarray([z['lead_p'] for z in out], np.float64)), 'lead_stat': ('region', np.asarray([z['lead_stat'] for z in out], np.float32)), 'n_leads': ('region',...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr = _require_xarray()
    if ds is None or not isinstance(ds, xr.Dataset) or ds.sizes.get("variant", 0) == 0:
        return xr.Dataset()
    p = np.asarray(ds["omnibus_p"].values, float); stat = np.asarray(ds["omnibus_stat"].values, float)
    pos = np.asarray(ds["position" if "position" in ds else "pos"].values, np.int64); chrom = np.asarray(ds["chromosome" if "chromosome" in ds else "chrom"].values).astype(str)
    snp = np.asarray(ds["variant_id" if "variant_id" in ds else "snp"].values).astype(str)
    vidx = np.asarray(ds["variant_index"].values, np.int64) if "variant_index" in ds.coords else np.arange(len(p), dtype=np.int64)
    cm_name = next((x for x in ("cm","cM","genetic_cm","genetic_pos_cm") if x in ds.coords), None)
    gpos = np.asarray(ds.coords[cm_name].values, float) if cm_name else pos.astype(float)/1e6*float(recombination_rate_cM_per_mb)
    lead = np.flatnonzero(np.isfinite(p) & (p <= float(lead_p)))
    if not len(lead) and int(fallback_top_k) > 0:
        finite = np.flatnonzero(np.isfinite(p)); k = min(int(fallback_top_k), len(finite))
        if k: lead = finite[np.argpartition(p[finite], k-1)[:k]]
    if not len(lead): return xr.Dataset()
    pad = int(padding_bp); pad_cm = pad/1e6*float(recombination_rate_cM_per_mb); rows=[]
    for cc in dict.fromkeys(chrom[lead].tolist()):
        ii = lead[chrom[lead] == cc]; ii = ii[np.argsort(pos[ii])]; cur=None
        for j in ii:
            rec={"chrom":str(cc),"start_pos":max(0,int(pos[j])-pad),"end_pos":int(pos[j])+pad,
                 "start_cm":float(gpos[j])-pad_cm,"end_cm":float(gpos[j])+pad_cm,"lead_indices":[int(j)]}
            if cur is None: cur=rec; continue
            prev=cur["lead_indices"][-1]
            genetic_ok = True if merge_recombination_cM is None else abs(float(gpos[j])-float(gpos[prev])) <= float(merge_recombination_cM)
            if rec["start_pos"] <= cur["end_pos"] and genetic_ok:
                cur["end_pos"] = max(cur["end_pos"], rec["end_pos"]); cur["start_cm"] = min(cur["start_cm"], rec["start_cm"]); cur["end_cm"] = max(cur["end_cm"], rec["end_cm"]); cur["lead_indices"].append(int(j))
            else: rows.append(cur); cur=rec
        if cur is not None: rows.append(cur)
    out=[]
    for r in rows:
        ii=np.asarray(r.pop("lead_indices"),np.int64); best=ii[np.nanargmin(p[ii])]
        out.append({**r,"lead_snp":snp[best],"lead_pos":int(pos[best]),"lead_variant_index":int(vidx[best]),
                    "lead_p":float(p[best]),"lead_stat":float(stat[best]),"lead_cm":float(gpos[best]),"n_leads":int(len(ii)),
                    "recombination_span_cm":float(r["end_cm"]-r["start_cm"])})
    out.sort(key=lambda z:(z["lead_p"],-z["lead_stat"]))
    if max_regions is not None: out=out[:max(0,int(max_regions))]
    return xr.Dataset(
        data_vars={"lead_p":("region",np.asarray([z["lead_p"] for z in out],np.float64)),
                   "lead_stat":("region",np.asarray([z["lead_stat"] for z in out],np.float32)),
                   "n_leads":("region",np.asarray([z["n_leads"] for z in out],np.int32)),
                   "recombination_span_cm":("region",np.asarray([z["recombination_span_cm"] for z in out],np.float32))},
        coords={"region":np.arange(len(out),dtype=np.int32),"chrom":("region",np.asarray([z["chrom"] for z in out]).astype(str)),
                "start_pos":("region",np.asarray([z["start_pos"] for z in out],np.int64)),"end_pos":("region",np.asarray([z["end_pos"] for z in out],np.int64)),
                "lead_snp":("region",np.asarray([z["lead_snp"] for z in out]).astype(str)),"lead_pos":("region",np.asarray([z["lead_pos"] for z in out],np.int64)),
                "lead_variant_index":("region",np.asarray([z["lead_variant_index"] for z in out],np.int64)),"lead_cm":("region",np.asarray([z["lead_cm"] for z in out],np.float64)),
                "start_cm":("region",np.asarray([z["start_cm"] for z in out],np.float64)),"end_cm":("region",np.asarray([z["end_cm"] for z in out],np.float64))},
        attrs={"lead_p_threshold":float(lead_p),"padding_bp":int(padding_bp),"recombination_rate_cM_per_mb":float(recombination_rate_cM_per_mb),
               "merge_recombination_cM":None if merge_recombination_cM is None else float(merge_recombination_cM),
               "region_definition":"significant lead +/- padding; recombination-aware merge"})



def pairwise_r2_xarray(
    X: Any, *, fit_mask: Optional[np.ndarray] = None, sample_rows: Optional[int] = None,
    random_state: int = 0, min_n: int = 3, max_compute_pairs: int = 4_000_000,
):
    """Virtual exact pairwise-observed LD matrix as a lazy xarray DataArray.

    The returned object has dimensions ``snp_i`` and ``snp_j`` but does not
    construct an m x m Dask graph and does not decode genotypes.  xarray label
    indexing is applied to the virtual backend first; only the requested SNP
    rows/columns are decoded when ``.compute()``, ``.load()`` or ``.values`` is
    requested.  A hard pair-count guard prevents accidental whole-genome LD
    materialization.

    Examples
    --------
    ``R2 = pairwise_r2_xarray(genotypes)``
    ``small = R2.sel(snp_i=locus_snps, snp_j=locus_snps).compute()``
    ``window = R2.isel(snp_i=slice(100, 180), snp_j=slice(100, 180)).compute()``

    Parameters
    ----------
    X : Any
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    sample_rows : Optional[int]
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``None``. Passed to ``rng.choice`` as ``size``.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    min_n : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``3``.

    max_compute_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``4000000``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_require_xarray()
    from xarray.backends import BackendArray
    from xarray.core import indexing
    if isinstance(X,xr.Dataset):
        if "genotypes" in X: X=X["genotypes"]
        elif len(X.data_vars)==1: X=X[next(iter(X.data_vars))]
        else: raise ValueError("genotype Dataset must contain one genotype variable")
    if not isinstance(X,xr.DataArray) or "iid" not in X.dims or "snp" not in X.dims:
        raise TypeError("X must be an xarray DataArray with iid and snp dimensions")
    rows=np.arange(X.sizes["iid"],dtype=np.int64)
    if fit_mask is not None:
        fit=np.asarray(fit_mask,bool)
        if fit.shape!=(len(rows),): raise ValueError("fit_mask has wrong shape")
        rows=rows[fit]
    if sample_rows is not None and len(rows)>int(sample_rows):
        rng=np.random.default_rng(random_state); rows=np.sort(rng.choice(rows,size=int(sample_rows),replace=False))
    source=X.isel(iid=rows)
    m=int(source.sizes["snp"]); guard=max(1,int(max_compute_pairs)); minobs=max(1,int(min_n))

    class _VirtualR2Backend(BackendArray):
        """_VirtualR2Backend represents virtual backend.

        See ``__init__`` for complete input types, defaults and construction constraints.
        """
        def __init__(self,src):
            """Initialize _VirtualR2Backend with the inputs and resource settings below.

            Parameters
            ----------
            src : path-like or genotype source
                Input file/store location or opened source consumed by this adapter. See the selected
                reader for physical format, sample order and supported block representations.
            """
            self.src=src; self._shape=(m,m); self._dtype=np.dtype(np.float32)
        @property
        def shape(self): 
            """Expose the logical array/operator dimensions without materializing its data.

            Returns
            -------
            object
                Computed result from ``self._shape``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return self._shape
        @property
        def dtype(self): 
            """Expose the numerical dtype of the represented lazy array/operator.

            Returns
            -------
            object
                Computed result from ``self._dtype``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return self._dtype
        def __getitem__(self,key):
            """Retrieve the requested entry from the indexed collection.

            Parameters
            ----------
            key : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.

            Returns
            -------
            object
                Computed result from ``indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._raw_indexing_method)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return indexing.explicit_indexing_adapter(key,self.shape,indexing.IndexingSupport.OUTER,self._raw_indexing_method)
        @staticmethod
        def _indices(k,n):
            """Normalize the requested array positions for the lazy genotype reader.

            Parameters
            ----------
            k : int or index array
                Local position, index selection or step used by the stated slice/iteration. Indices must
                refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
                the enclosing dimensions.

            n : int
                Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
                length, not an arbitrary truncation limit.
            """
            if isinstance(k,slice): return np.arange(n,dtype=np.int64)[k],False
            a=np.asarray(k)
            if a.ndim==0: return np.asarray([int(a)],np.int64),True
            return a.astype(np.int64,copy=False).reshape(-1),False
        def _raw_indexing_method(self,key):
            """Decode the requested genotype slice for xarray's explicit indexing protocol.

            Parameters
            ----------
            key : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.
            """
            ii,si=self._indices(key[0],m); jj,sj=self._indices(key[1],m)
            pairs=int(len(ii))*int(len(jj))
            if pairs>guard:
                raise MemoryError(
                    f"refusing to materialize {len(ii):,} x {len(jj):,} = {pairs:,} R2 pairs; "
                    f"slice the virtual DataArray first with .sel/.isel (max_compute_pairs={guard:,})"
                )
            both=np.r_[ii,jj]
            uniq,inv=np.unique(both,return_inverse=True); ni=len(ii); mi=inv[:ni]; mj=inv[ni:]
            Xu=_xarray_compute(self.src.isel(snp=uniq),np.float32)
            A=np.asarray(Xu[:,mi],np.float32,order="C"); B=np.asarray(Xu[:,mj],np.float32,order="C")
            if np.isfinite(Xu).all():
                A0=A-A.mean(0,dtype=np.float64).astype(np.float32); B0=B-B.mean(0,dtype=np.float64).astype(np.float32)
                na=np.sqrt(np.sum(A0*A0,axis=0,dtype=np.float64)); nb=np.sqrt(np.sum(B0*B0,axis=0,dtype=np.float64))
                den=np.outer(na,nb); R=np.divide(A0.T@B0,den,out=np.zeros((len(ii),len(jj)),np.float64),where=den>1e-20)
                out=np.clip(R*R,0.0,1.0).astype(np.float32,copy=False)
            else:
                out=_pairwise_complete_r2_block(A,B,min_n=minobs)
            if si and sj: return np.asarray(out[0,0],np.float32)
            if si: return out[0]
            if sj: return out[:,0]
            return out

    data=indexing.LazilyIndexedArray(_VirtualR2Backend(source))
    snp=np.asarray(source.coords["snp"].values).astype(str) if "snp" in source.coords else np.arange(m).astype(str)
    coords={"snp_i":snp,"snp_j":snp}
    for nm in ("pos","chrom","genetic_cm","cm","a0","a1","source_variant_index"):
        if nm in source.coords and source.coords[nm].dims==("snp",):
            vals=np.asarray(source.coords[nm].values)
            coords[f"{nm}_i"]=("snp_i",vals); coords[f"{nm}_j"]=("snp_j",vals)
    out=xr.DataArray(data,dims=("snp_i","snp_j"),coords=coords,name="r2")
    out.attrs.update({"long_name":"Exact pairwise-observed genotype R-squared","virtual":True,
                      "materialization":"selection_then_compute_only","n_samples":int(len(rows)),
                      "min_joint_observations":int(minobs),"max_compute_pairs":int(guard),
                      "missingness":"pairwise_observed_exact","source":"lazy genotype DataArray"})
    return out



def _pairwise_complete_r2_block(A, B, min_n=3):
    """Pairwise-complete R2 using BLAS sufficient statistics.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``np.isfinite(A)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    B : array-like
        Numerical operand in ``np.isfinite(B)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    min_n : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``3``.
    """
    from .gwas.ld import pairwise_complete
    return pairwise_complete(A, B, min_n=min_n, dtype=np.float32, statistic="r2")





# Backward-compatible spelling; both names return the same virtual xarray backend.
pairwise_r2_xarray_lazy = pairwise_r2_xarray

def fast_windowed_r2(
    X: Any,
    *,
    positions: Optional[Sequence[int]] = None,
    genetic_cm: Optional[Sequence[float]] = None,
    fit_mask: Optional[np.ndarray] = None,
    max_distance_bp: Optional[int] = 1_000_000,
    max_distance_cm: Optional[float] = None,
    max_distance_variants: Optional[int] = None,
    sample_rows: Optional[int] = 2048,
    block_variants: int = 256,
    min_r2: float = 0.0,
    random_state: int = 0,
) -> np.ndarray:
    """Fast banded R2 for selected loci using BLAS blocks.

    The xarray input is sliced lazily first; only the selected locus and at
    most ``sample_rows`` animals are materialized.  Pairs can be bounded by
    physical distance, genetic-map distance, variant distance, or any
    combination thereof.  Uncomputed pairs are treated as r2=0.

    Parameters
    ----------
    X : Any
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    positions : Optional[Sequence[int]]
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    genetic_cm : Optional[Sequence[float]]
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    max_distance_bp : Optional[int]
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.

    max_distance_cm : Optional[float]
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``.

    max_distance_variants : Optional[int]
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``None``.

    sample_rows : Optional[int]
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``rng.choice`` as ``size``.

    block_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``256``.

    min_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.0``. Source comparisons: ``min_r2 > 0``. See the rejection/fallback
        branches for enforcement.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``R2``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr = _require_xarray()
    if isinstance(X, xr.DataArray):
        rows = np.arange(X.sizes["iid"], dtype=np.int64)
        fit = np.ones(len(rows), bool) if fit_mask is None else np.asarray(fit_mask, bool)
        rows = rows[fit]
        if sample_rows is not None and len(rows) > int(sample_rows):
            rng = np.random.default_rng(random_state); rows = np.sort(rng.choice(rows, size=int(sample_rows), replace=False))
        X = _xarray_compute(X.isel(iid=rows), np.float32)
        fit_mask = None; sample_rows = None
    X = _to_numpy(X, np.float32); n, m = X.shape
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool); rows = np.flatnonzero(fit)
    if sample_rows is not None and len(rows) > int(sample_rows):
        rng = np.random.default_rng(random_state); rows = np.sort(rng.choice(rows, size=int(sample_rows), replace=False))
    Z = np.asarray(X[rows], np.float32, order="C"); finite = np.isfinite(Z); has_missing=not bool(finite.all())
    if not has_missing:
        mu=Z.mean(0,dtype=np.float64).astype(np.float32); Z=Z-mu[None,:]
        ss=np.sqrt(np.sum(Z*Z,axis=0,dtype=np.float64)).astype(np.float32);ss[ss<1e-7]=1.;Z/=ss[None,:]
    pos = None if positions is None else np.asarray(positions,np.int64)
    cm = None if genetic_cm is None else np.asarray(genetic_cm,np.float64)
    if pos is not None and np.any(np.diff(pos)<0): raise ValueError("positions must be sorted")
    if cm is not None and np.any(np.diff(cm)<-1e-12): raise ValueError("genetic_cm must be sorted")
    R2 = np.zeros((m,m),np.float32); np.fill_diagonal(R2,1.); bs=max(1,int(block_variants))
    for a0 in range(0,m,bs):
        a1=min(m,a0+bs)
        b1=m
        if pos is not None and max_distance_bp is not None:
            b1=min(b1,int(np.searchsorted(pos,pos[a1-1]+int(max_distance_bp),side="right")))
        if cm is not None and max_distance_cm is not None:
            b1=min(b1,int(np.searchsorted(cm,cm[a1-1]+float(max_distance_cm),side="right")))
        if max_distance_variants is not None:
            b1=min(b1,a1+int(max_distance_variants))
        RR=_pairwise_complete_r2_block(Z[:,a0:a1],Z[:,a0:b1]) if has_missing else np.asarray((Z[:,a0:a1].T@Z[:,a0:b1])**2,np.float32)
        for ia,gi in enumerate(range(a0,a1)):
            js=np.arange(gi,b1,dtype=np.int64)
            keep=np.ones(len(js),bool)
            if pos is not None and max_distance_bp is not None: keep &= (pos[js]-pos[gi]) <= int(max_distance_bp)
            if cm is not None and max_distance_cm is not None: keep &= (cm[js]-cm[gi]) <= float(max_distance_cm)
            if max_distance_variants is not None: keep &= (js-gi) <= int(max_distance_variants)
            jj=js[keep]; vals=RR[ia,(jj-a0)]
            if min_r2>0: vals=np.where(vals>=float(min_r2),vals,0.)
            R2[gi,jj]=vals; R2[jj,gi]=vals
    np.fill_diagonal(R2,1.)
    return R2


def fast_windowed_r2_edges(
    X: Any,
    *,
    positions: Optional[Sequence[int]] = None,
    genetic_cm: Optional[Sequence[float]] = None,
    fit_mask: Optional[np.ndarray] = None,
    max_distance_bp: Optional[int] = 1_000_000,
    max_distance_cm: Optional[float] = None,
    max_distance_variants: Optional[int] = None,
    sample_rows: Optional[int] = 2048,
    block_variants: int = 256,
    min_r2: float = 0.20,
    random_state: int = 0,
    return_xarray: bool = True,
):
    """Thresholded rolling-LD graph without materializing an m x m matrix.

    This is the preferred first pass for on-the-go fine mapping.  Only pairs
    inside the physical/recombination/variant band are evaluated and only
    edges with ``r2 >= min_r2`` are retained.  For Dask-backed xarray input,
    the requested locus/sample subset is sliced lazily before one local
    compute.  Dense R2 is then computed only *within surviving components*
    by :func:`genomic_kernel_expert.ld_group_svd_feature_bank`.

    Returns an xarray Dataset with one ``edge`` dimension by default, which is
    compact and convenient for Zarr/visualization; ``return_xarray=False``
    returns a small dictionary of NumPy arrays.

    Parameters
    ----------
    X : Any
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Passed to ``xr.Dataset`` as ``attrs``.

    positions : Optional[Sequence[int]]
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    genetic_cm : Optional[Sequence[float]]
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.

    fit_mask : Optional[np.ndarray]
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``pairwise_r2_xarray`` as ``fit_mask``.

    max_distance_bp : Optional[int]
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``xr.Dataset`` as ``attrs``.

    max_distance_cm : Optional[float]
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``. Passed to ``xr.Dataset`` as ``attrs``.

    max_distance_variants : Optional[int]
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``None``. Passed to ``xr.Dataset`` as ``attrs``.

    sample_rows : Optional[int]
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``2048``. Passed to ``pairwise_r2_xarray`` as ``sample_rows``.

    block_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``256``.

    min_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.2``.

    random_state : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``pairwise_r2_xarray`` as ``random_state``.

    return_xarray : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    dict
        Result fields: ``i``, ``j``, ``r2``, ``n_variants``, ``sample_rows``. Field arrays preserve the corresponding input axis order.
    """
    xr = _require_xarray()
    snp_names = None
    if isinstance(X, xr.DataArray):
        if "snp" in X.coords: snp_names=np.asarray(X.coords["snp"].values).astype(str)
        m=int(X.sizes["snp"]); pos=None if positions is None else np.asarray(positions,np.int64)
        cm=None if genetic_cm is None else np.asarray(genetic_cm,np.float64)
        if pos is not None and np.any(np.diff(pos)<0): raise ValueError("positions must be sorted")
        if cm is not None and np.any(np.diff(cm)<-1e-12): raise ValueError("genetic_cm must be sorted")
        # O(m) virtual metadata only. Each loop slices both SNP dimensions first;
        # .compute() therefore decodes just the requested rolling block.
        maxpairs=max(1,int(block_variants))*max(1,int(block_variants))*16
        Rview=pairwise_r2_xarray(X,fit_mask=fit_mask,sample_rows=sample_rows,random_state=random_state,
                                 max_compute_pairs=max(4_000_000,maxpairs))
        src_parts=[];dst_parts=[];val_parts=[];bs=max(1,int(block_variants));thr=float(min_r2)
        for a0 in range(0,m,bs):
            a1=min(m,a0+bs); b1=m
            if pos is not None and max_distance_bp is not None:b1=min(b1,int(np.searchsorted(pos,pos[a1-1]+int(max_distance_bp),side="right")))
            if cm is not None and max_distance_cm is not None:b1=min(b1,int(np.searchsorted(cm,cm[a1-1]+float(max_distance_cm),side="right")))
            if max_distance_variants is not None:b1=min(b1,a1+int(max_distance_variants))
            if b1<=a0:continue
            RR=np.asarray(Rview.isel(snp_i=slice(a0,a1),snp_j=slice(a0,b1)).compute().values,np.float32)
            for ia,gi in enumerate(range(a0,a1)):
                js=np.arange(gi+1,b1,dtype=np.int64)
                if not len(js):continue
                keep=np.ones(len(js),bool)
                if pos is not None and max_distance_bp is not None:keep&=(pos[js]-pos[gi])<=int(max_distance_bp)
                if cm is not None and max_distance_cm is not None:keep&=(cm[js]-cm[gi])<=float(max_distance_cm)
                if max_distance_variants is not None:keep&=(js-gi)<=int(max_distance_variants)
                vals=RR[ia,js-a0]; hit=keep&(vals>=thr)
                if np.any(hit):
                    jj=js[hit];src_parts.append(np.full(len(jj),gi,np.int32));dst_parts.append(jj.astype(np.int32,copy=False));val_parts.append(vals[hit])
        src=np.concatenate(src_parts) if src_parts else np.empty(0,np.int32); dst=np.concatenate(dst_parts) if dst_parts else np.empty(0,np.int32); val=np.concatenate(val_parts) if val_parts else np.empty(0,np.float32)
        if not return_xarray:return {"i":src,"j":dst,"r2":val,"n_variants":m,"sample_rows":int(Rview.attrs.get("n_samples",X.sizes["iid"]))}
        coords={"edge":np.arange(len(src),dtype=np.int64),"variant_i":("edge",src),"variant_j":("edge",dst)}
        if snp_names is not None:coords.update(snp_i=("edge",snp_names[src]),snp_j=("edge",snp_names[dst]))
        return xr.Dataset({"r2":("edge",val)},coords=coords,attrs={"n_variants":m,"sample_rows":int(Rview.attrs.get("n_samples",X.sizes["iid"])),
            "min_r2":thr,"missingness":"pairwise_observed_exact","engine":"virtual_xarray_selection_then_compute"})
    X = _to_numpy(X, np.float32)
    n, m = X.shape
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    rows = np.flatnonzero(fit)
    if sample_rows is not None and len(rows) > int(sample_rows):
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(rows, size=int(sample_rows), replace=False))
    Z = np.asarray(X[rows], np.float32, order="C")
    finite=np.isfinite(Z);has_missing=not bool(finite.all())
    if not has_missing:
        mu=Z.mean(0,dtype=np.float64).astype(np.float32);Z=Z-mu[None,:]
        norm=np.sqrt(np.sum(Z*Z,axis=0,dtype=np.float64)).astype(np.float32);norm[norm<1e-7]=1.0;Z/=norm[None,:]

    pos = None if positions is None else np.asarray(positions, np.int64)
    cm = None if genetic_cm is None else np.asarray(genetic_cm, np.float64)
    if pos is not None and np.any(np.diff(pos) < 0):
        raise ValueError("positions must be sorted")
    if cm is not None and np.any(np.diff(cm) < -1e-12):
        raise ValueError("genetic_cm must be sorted")

    src_parts, dst_parts, val_parts = [], [], []
    bs = max(1, int(block_variants)); thr = float(min_r2)
    for a0 in range(0, m, bs):
        a1 = min(m, a0 + bs); b1 = m
        if pos is not None and max_distance_bp is not None:
            b1 = min(b1, int(np.searchsorted(pos, pos[a1-1] + int(max_distance_bp), side="right")))
        if cm is not None and max_distance_cm is not None:
            b1 = min(b1, int(np.searchsorted(cm, cm[a1-1] + float(max_distance_cm), side="right")))
        if max_distance_variants is not None:
            b1 = min(b1, a1 + int(max_distance_variants))
        if b1 <= a0:
            continue
        if has_missing:
            RR=_pairwise_complete_r2_block(Z[:,a0:a1],Z[:,a0:b1])
        else:
            R=Z[:,a0:a1].T@Z[:,a0:b1];RR=np.asarray(R*R,np.float32)
        for ia, gi in enumerate(range(a0, a1)):
            js = np.arange(gi + 1, b1, dtype=np.int64)
            if not len(js):
                continue
            keep = np.ones(len(js), bool)
            if pos is not None and max_distance_bp is not None:
                keep &= (pos[js] - pos[gi]) <= int(max_distance_bp)
            if cm is not None and max_distance_cm is not None:
                keep &= (cm[js] - cm[gi]) <= float(max_distance_cm)
            if max_distance_variants is not None:
                keep &= (js - gi) <= int(max_distance_variants)
            vals = RR[ia, js - a0]
            hit = keep & (vals >= thr)
            if np.any(hit):
                jj = js[hit]
                src_parts.append(np.full(len(jj), gi, dtype=np.int32))
                dst_parts.append(jj.astype(np.int32, copy=False))
                val_parts.append(vals[hit])
    src = np.concatenate(src_parts) if src_parts else np.empty(0, np.int32)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, np.int32)
    val = np.concatenate(val_parts) if val_parts else np.empty(0, np.float32)
    if not return_xarray:
        return {"i": src, "j": dst, "r2": val, "n_variants": int(m), "sample_rows": int(len(rows))}
    coords = {"edge": np.arange(len(src), dtype=np.int64),
              "variant_i": ("edge", src), "variant_j": ("edge", dst)}
    if snp_names is not None and len(snp_names) == m:
        coords["snp_i"] = ("edge", snp_names[src])
        coords["snp_j"] = ("edge", snp_names[dst])
    return xr.Dataset(
        data_vars={"r2": ("edge", val)}, coords=coords,
        attrs={"n_variants": int(m), "sample_rows": int(len(rows)),
               "min_r2": thr,
               "max_distance_bp": None if max_distance_bp is None else int(max_distance_bp),
               "max_distance_cm": None if max_distance_cm is None else float(max_distance_cm),
               "max_distance_variants": None if max_distance_variants is None else int(max_distance_variants)})


def _zarr_encoding(ds, chunk_variants=8192, shard_variants=65536):
    """Construct per-variable Zarr v3 chunks/shards for the scan result schema.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``8192``.

    shard_variants : int or None
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy. Default:
        ``65536``.
    """
    c=max(1,int(chunk_variants));s=max(c,int(shard_variants));s=int(math.ceil(s/c)*c);enc={}
    for name,var in ds.variables.items():
        if getattr(var.dtype,"kind","O") in {"O","U","S"}: continue
        if "variant" in var.dims:
            j=var.dims.index("variant"); n=max(1,int(var.shape[j])); ch=list(var.shape); sh=list(var.shape)
            inner=min(c,n); outer=min(s,n)
            if outer > inner: outer=max(inner,(outer//inner)*inner)
            ch[j]=inner; sh[j]=outer; enc[name]={"chunks":tuple(ch),"shards":tuple(sh)}
    return enc


def _write_chrom_zarr(ds, path, chromosome, chunk_variants, shard_variants):
    """Write chrom zarr.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Passed to ``VigorStore(p,
        mode=mode).write_dataset`` as ``encoding``.

    shard_variants : int or None
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy. Passed to
        ``VigorStore(p, mode=mode).write_dataset`` as ``encoding``.
    """
    p=Path(path);mode="a" if p.exists() else "w"
    # The group name/attribute already identifies chromosome; do not store the
    # same chromosome string once per variant on disk.  Reconstruct it on read.
    store=ds.drop_vars("chrom") if "chrom" in ds.coords and ds["chrom"].dims==("variant",) else ds
    store=store.copy();store.attrs["chromosome"]=str(chromosome)
    from .store import VigorStore
    VigorStore(p,mode=mode).write_dataset(store,f"chromosomes/{chromosome}",
        mode="a",encoding=_zarr_encoding(store,chunk_variants,shard_variants),consolidated=False)


def open_scan_zarr3(path, chromosomes=None):
    """Open scan zarr3.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_require_xarray(); scans=[]
    import zarr
    root=zarr.open_group(str(path),mode="r",use_consolidated=False)
    if "variant" in root and "trait" in root and "stat" in root:
        from .gwas_dataset import open_gwas_zarr
        ds=open_gwas_zarr(path)
        if chromosomes is not None:
            key="chromosome" if "chromosome" in ds else "chrom"
            selected=np.isin(np.asarray(ds[key].values).astype(str),list(map(str,chromosomes)))
            ds=ds.isel(variant=np.flatnonzero(selected))
        return ds
    if chromosomes is None:
        import zarr
        root=zarr.open_group(str(path),mode="r")
        chromosomes=list(root["chromosomes"].group_keys()) if "chromosomes" in root else []
        def _ckey(x):
            """Construct the chromosome/selection cache key for this scan input.

            Parameters
            ----------
            x : array-like
                Numerical operand in ``str(x)``. Shape must conform to that matrix/vector operation and
                the enclosing prepared state.
            """
            sx=str(x); return (0,int(sx)) if sx.isdigit() else (1,sx)
        chromosomes=sorted(map(str,chromosomes),key=_ckey)
    for c in chromosomes:
        try: ds=xr.open_zarr(str(path),group=f"chromosomes/{c}",consolidated=True,zarr_format=3,chunks="auto")
        except Exception: ds=xr.open_zarr(str(path),group=f"chromosomes/{c}",consolidated=False,zarr_format=3,chunks="auto")
        if "chrom" not in ds.coords: ds=ds.assign_coords(chrom=("variant",np.asarray([str(c)] * ds.sizes.get("variant",0),dtype=str)))
        scans.append(ds)
    out=xr.concat(scans,dim="variant",data_vars="all",coords="minimal",compat="override",join="outer") if scans else xr.Dataset()
    if out.sizes.get("variant",0): out=out.assign_coords(variant=np.arange(out.sizes["variant"],dtype=np.int64))
    return out


def state_from_vigor_null(null: Any, G: Any = None, E: Any = None, *, rank: int = 10, basis: str = "canonical",
                          trait_names: Optional[Sequence[str]] = None, train_mask: Optional[np.ndarray] = None,
                          null_reps: int = 4, seed: int = 0) -> FastScoreState:
    """Build a calibrated-ready scan state from an exact VIGOR score null.

    ``null`` comes from :func:`vigor.gwas.exact.prepare_score_null` (use the
    LOCO kernel of the scanned chromosome). Its dual ``null.dual`` (P y,
    animals x traits) is projected to ``rank`` genetic axes, and ``null_reps``
    simulated null trait sets under the same model provide the calibration
    duals (:func:`null_trait_duals`).

    Parameters
    ----------
    null : ExactScoreNull
        Prepared exact null with dense kernels.
    G, E : array-like, optional
        Trait covariances defining the scan axes; default the null's total
        genetic covariance and its residual covariance.
    rank : int
        Number of scan axes. Default 10.
    basis : str
        Axis construction passed to :func:`trait_scan_basis` ('canonical',
        'genetic' or 'full').
    trait_names : sequence of str, optional
        Labels for the scan axes.
    train_mask : bool array, optional
        Animals contributing to the scores.
    null_reps : int
        Null trait sets for calibration (0 disables; the state then cannot be
        calibrated).
    seed : int
        Seed for the null draws.
    """
    from .gwas.exact import ExactScoreNull
    if not isinstance(null, ExactScoreNull):
        raise TypeError("null must be an ExactScoreNull from vigor.gwas.exact.prepare_score_null")
    if G is None:
        G = sum(np.asarray(v, np.float64) for v in null.covariances.values())
    if E is None:
        E = null.residual
    duals = null_trait_duals(null, reps=null_reps, seed=seed) if null_reps else None
    return prepare_fast_score_state(null.dual, G, E, rank=rank, basis=basis, trait_names=trait_names,
                                    train_mask=train_mask, null_A=duals)


def _resolve_score_state(state: Any, chromosome: str) -> FastScoreState:
    """Resolve score state.

    Parameters
    ----------
    state : Any
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    chromosome : str
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    if callable(state): st=state(str(chromosome))
    elif isinstance(state, Mapping):
        if str(chromosome) in state: st=state[str(chromosome)]
        elif chromosome in state: st=state[chromosome]
        else: raise KeyError(f"no score state for chromosome {chromosome}")
    else: st=state
    if not isinstance(st,FastScoreState): raise TypeError("score state/factory must yield FastScoreState")
    return st



def scores_to_xarray(names, chrom, pos, variant_index, z, state):
    """Compact xarray Dataset: one variant row plus optional factor scores.

    Parameters
    ----------
    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    variant_index : one-dimensional integer array or None
        Zero-based variant positions to select from the current source. Values must refer to
        existing columns; None retains all eligible variants. Streaming readers may additionally
        require sorted indices.

    z : object (see the cited input contract)
        Input for the operation ``z['stat']``. See ``vigor.mvgwas_fast.scores_to_xarray`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    Returns
    -------
    object
        Computed result from ``_chrom_dataset(cc, names, pos, variant_index, z['stat'], z['p'], z['neglog_p'], z['nobs'], z.get('factor_z'), state)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    c = np.asarray(chrom).astype(str)
    cc = c[0] if len(c) else ""
    return _chrom_dataset(cc, names, pos, variant_index, z["stat"], z["p"], z["neglog_p"], z["nobs"], z.get("factor_z"), state)


def iter_xarray_variant_chunks(genotypes, *, chromosomes=None, sample_ids=None, chunk_variants="auto", npplink_module=None, maxmem=None, memory_fraction=None):
    """Yield lazy contiguous xarray genotype chunks with genomic coordinates.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``open_genotype_xarray`` as ``sample_ids``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the working-memory budget; integers are ceilings). Passed to
        ``open_genotype_xarray`` as ``io_chunk_variants``.

    npplink_module : object (see the cited input contract)
        Input supplied to ``open_genotype_xarray`` as ``npplink_module``. See that callable's
        documented ``npplink_module`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value. Passed to
        ``open_genotype_xarray`` as ``npplink_module``.

    maxmem, memory_fraction : budget or None
        Working-memory budget for the automatic chunk; see ``open_genotype_xarray``.
        Default: ``None`` (80% of physical RAM, capped at available RAM).

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    g = open_genotype_xarray(genotypes, npplink_module=npplink_module, io_chunk_variants=chunk_variants, sample_ids=sample_ids, maxmem=maxmem, memory_fraction=memory_fraction)
    g = _align_samples(g, sample_ids); chunk_variants = int(g.attrs["io_chunk_variants"])  # budget-resolved ('auto' or ceiling)
    chrom = np.asarray(g.coords["chrom"].values).astype(str) if "chrom" in g.coords else np.full(g.sizes["snp"], "", object)
    use = list(dict.fromkeys(chrom.tolist())) if chromosomes is None else list(map(str, chromosomes))
    names = np.asarray(g.coords["snp"].values).astype(str)
    pos = np.asarray(g.coords["pos"].values, np.int64) if "pos" in g.coords else np.arange(g.sizes["snp"], dtype=np.int64)
    for cc in use:
        idx = np.flatnonzero(chrom == str(cc)) if "chrom" in g.coords else np.arange(g.sizes["snp"], dtype=np.int64)
        for a0 in range(0, len(idx), int(chunk_variants)):
            gi = idx[a0:a0+int(chunk_variants)]
            yield {"X": g.isel(snp=gi), "snp": names[gi], "pos": pos[gi], "variant_index": gi, "chrom": str(cc)}


def recombination_aware_regions(ds, *, seed_p=1e-5, flank_bp=1_000_000, flank_cM=None,
                                recombination_rate_cM_per_mb=.5, merge_recombination_cM=1.0,
                                max_regions=64):
    """Compatibility wrapper accepting compact ``stat/p`` or omnibus names.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    seed_p : float
        Input supplied to ``candidate_regions_from_scan`` as ``lead_p``. See that callable's
        documented ``lead_p`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``1e-05``. Passed to ``candidate_regions_from_scan`` as
        ``lead_p``.

    flank_bp : int
        Input supplied to ``candidate_regions_from_scan`` as ``padding_bp``. See that callable's
        documented ``padding_bp`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``1000000``. Passed to
        ``candidate_regions_from_scan`` as ``padding_bp``.

    flank_cM : object (see the cited input contract)
        Input for the operation ``float(flank_cM) * 2.0``. See
        ``vigor.mvgwas_fast.recombination_aware_regions`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``. Passed to ``candidate_regions_from_scan`` as
        ``recombination_rate_cM_per_mb``.

    merge_recombination_cM : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``1.0``.

    max_regions : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``64``. Passed to ``candidate_regions_from_scan`` as ``max_regions``.

    Returns
    -------
    object
        Computed result from ``candidate_regions_from_scan(z, lead_p=float(seed_p), padding_bp=int(flank_bp), recombination_rate_cM_per_mb=float(recombination_rate_cM_per_mb), merge_recombination_cM=merge_cm, max_regions=max_reg...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr = _require_xarray()
    if ds is None or ds.sizes.get("variant",0)==0: return xr.Dataset()
    z = ds
    ren = {}
    if "p" in z and "omnibus_p" not in z: ren["p"] = "omnibus_p"
    if "stat" in z and "omnibus_stat" not in z: ren["stat"] = "omnibus_stat"
    if "neglog_p" in z and "neglog_p" not in z: pass
    if ren: z = z.rename(ren)
    # +/-1 Mb remains the primary locus definition requested for prediction.
    # flank_cM is used as a recombination-aware merge guard when supplied.
    merge_cm = merge_recombination_cM if flank_cM is None else max(float(flank_cM)*2.0, 1e-12)
    return candidate_regions_from_scan(z, lead_p=float(seed_p), padding_bp=int(flank_bp),
        recombination_rate_cM_per_mb=float(recombination_rate_cM_per_mb),
        merge_recombination_cM=merge_cm, max_regions=max_regions, fallback_top_k=0)


def _write_scan_metadata(path, *, chromosomes, n_variants, chunk_variants, shard_variants, engine, assume_complete, consolidated, factor_ranks):
    """Write scan metadata.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.

    shard_variants : int or None
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation.

    assume_complete : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Source-recognized literal values:
        ``auto``. These select the corresponding branches below; comparison evidence alone does
        not establish an exhaustive enum.

    consolidated : bool or None
        Whether to request consolidated Zarr metadata. None lets the reader choose/fall back
        according to the store metadata and backend support.

    factor_ranks : mapping
        Input for the operation ``factor_ranks.items()``. See
        ``vigor.mvgwas_fast._write_scan_metadata`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    from datetime import datetime, timezone
    import json
    try:
        from .store import SCHEMA_VERSION
    except Exception:
        SCHEMA_VERSION="domino-zarr-v3-2"
    meta={
        "domino_schema":"mvGWAS_scan_store","domino_schema_version":SCHEMA_VERSION,
        "domino_artifact_type":"mvgwas_scan","domino_storage_format":"zarr3","domino_write_state":"complete",
        "zarr_format":3,"consolidated_metadata":bool(consolidated),"sharded":True,
        "chunk_variants":int(chunk_variants),"shard_variants":int(shard_variants),
        "chromosomes":list(map(str,chromosomes)),"n_variants_scanned":int(n_variants),
        "engine":str(engine),"factor_ranks":{str(k):int(v) for k,v in factor_ranks.items()},
        "genotype_missingness_policy":("auto_exact_lazy" if isinstance(assume_complete,str) and assume_complete.lower()=="auto" else ("assume_complete" if assume_complete else "exact_variantwise_observed")),
        "writer":"vigor.mvgwas_fast","created_utc":datetime.now(timezone.utc).isoformat(),
        "layout":"chromosomes/<chromosome> xarray Dataset groups; variant-major sharded Zarr v3",
    }
    try:
        import zarr
        root=zarr.open_group(str(path),mode="a"); root.attrs.update(meta)
    except Exception: pass
    sidecar=Path(str(path)+".metadata.json");sidecar.write_text(json.dumps(meta,indent=2,sort_keys=True))
    return meta


class Zarr3ScanWriter:
    """Per-chromosome buffered writer for sharded, consolidated Zarr v3.

    Only compact score datasets are buffered; genotype matrices are never held.
    This is intentionally optional so the module still imports without zarr.
    """
    def __init__(self, path, *, chunk_variants=8192, shard_variants=65536):
        """Initialize Zarr3ScanWriter with the inputs and resource settings below.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        chunk_variants : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``8192``.

        shard_variants : int or None
            Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
            compatible with the chunk shape; None uses the writer's shard policy. Default:
            ``65536``.
        """
        self.path = Path(path); self.chunk_variants=int(chunk_variants); self.shard_variants=int(shard_variants)
        self._chrom=None; self._parts=[]; self._written=[]
    def _flush(self):
        """Write the buffered association block to its preallocated Zarr region.
                
        """
        if not self._parts: return
        xr=_require_xarray(); ds=xr.concat(self._parts,dim="variant",data_vars="all",coords="minimal",compat="override",join="outer")
        ds=ds.assign_coords(variant=np.arange(ds.sizes["variant"],dtype=np.int64))
        _write_chrom_zarr(ds,self.path,self._chrom,self.chunk_variants,self.shard_variants); self._written.append(str(self._chrom)); self._parts=[]
    def append(self, ds):
        """Append one result block to the writer's bounded in-memory buffer.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Labeled data object containing the variables and dimension coordinates read below.
            Scientific axes/metadata must match the corresponding genotype, association or posterior
            schema.
        """
        if ds.sizes.get("variant",0)==0:return
        cc=str(np.asarray(ds["chrom"].values).astype(str)[0]) if "chrom" in ds.coords else "unknown"
        if self._chrom is None:self._chrom=cc
        if cc!=self._chrom:self._flush();self._chrom=cc
        self._parts.append(ds)
    def finalize(self, open_dataset=False):
        """Flush pending blocks and mark the persistent scan artifact complete.

        Parameters
        ----------
        open_dataset : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``False``.

        Returns
        -------
        object
            Computed result from ``open_scan_zarr3(self.path, self._written) if open_dataset else None``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self._flush()
        try:
            import zarr; zarr.consolidate_metadata(str(self.path))
        except Exception: pass
        return open_scan_zarr3(self.path,self._written) if open_dataset else None

def scan_xarray_recombination(
    genotypes: Any,
    state: Any,
    *,
    chromosomes: Optional[Sequence[Any]] = None,
    sample_ids: Optional[Sequence[Any]] = None,
    npplink_module=None,
    io_chunk_variants: Any = "auto",
    score_chunk_variants: Optional[int] = None,
    calibration_n: int = 4096,
    calibration_io_windows: int = 4,
    top_k: int = 2048,
    p_threshold: Optional[float] = None,
    return_factor_z: bool = True,
    region_p_threshold: float = 1e-5,
    region_padding_bp: int = 1_000_000,
    region_recombination_rate_cM_per_mb: float = 0.5,
    region_merge_recombination_cM: Optional[float] = 1.0,
    max_regions: Optional[int] = 64,
    region_fallback_top_k: int = 0,
    exact_rescore_p: Optional[float] = 5e-7,
    exact_rescore_top: int = 64,
    exact_rescore_fn: Optional[Callable[..., Any]] = None,
    zarr_path: Optional[Union[str, Path]] = None,
    zarr_chunk_variants: int = 8192,
    zarr_shard_variants: int = 65536,
    return_all: bool = True,
    assume_complete: Any = "auto",
    score_workers: Optional[int] = None,
    progress_callback=None,
    maxmem=None,
    memory_fraction=None,
) -> ScanResult:
    """High-throughput compact multivariate score scan.

    ``state`` may be one shared :class:`FastScoreState`, a chromosome->state
    mapping, or a callable returning a state.  Mapping/callable mode is the
    true LOCO path: each chromosome uses the exact VIGOR LOCO null dual.

    The Dask-backed genotype matrix remains lazy.  A storage chunk is reduced
    directly to X' A_factor, sums and sums-of-squares; no n x chunk NumPy
    matrix is materialized on the complete/imputed fast path. Exact rescoring
    happens only after the chromosome-wide fast scan has defined surviving
    +/-1 Mb recombination-aware loci.

    Parameters
    ----------
    genotypes : Any
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    state : Any
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    chromosomes : Optional[Sequence[Any]]
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    sample_ids : Optional[Sequence[Any]]
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``open_genotype_xarray`` as ``sample_ids``.

    npplink_module : array-like or numeric scalar
        Input supplied to ``open_genotype_xarray`` as ``npplink_module``. See that callable's
        documented ``npplink_module`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value. Passed to
        ``open_genotype_xarray`` as ``npplink_module``.

    io_chunk_variants : int or 'auto'
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``'auto'`` (sized from the working-memory
        budget; integers are ceilings).

    score_chunk_variants : Optional[int]
        Variants per scoring block. Default: ``None`` (the I/O chunk), then clipped so each
        score worker's block fits ``budget / score_workers``.

    calibration_n : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4096``. Passed to ``calibrate_xarray_score_state`` as ``n_calibration``.

    calibration_io_windows : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4``. Passed to ``calibrate_xarray_score_state`` as
        ``calibration_io_windows``.

    top_k : int
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``2048``.

    p_threshold : Optional[float]
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    return_factor_z : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    region_p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``1e-05``. Passed to ``candidate_regions_from_scan`` as ``lead_p``.

    region_padding_bp : int
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``candidate_regions_from_scan`` as ``padding_bp``.

    region_recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``. Passed to ``candidate_regions_from_scan`` as
        ``recombination_rate_cM_per_mb``.

    region_merge_recombination_cM : Optional[float]
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``1.0``. Passed to ``candidate_regions_from_scan`` as ``merge_recombination_cM``.

    max_regions : Optional[int]
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``64``.

    region_fallback_top_k : int
        Input supplied to ``candidate_regions_from_scan`` as ``fallback_top_k``. See that
        callable's documented ``fallback_top_k`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``0``. Passed to
        ``candidate_regions_from_scan`` as ``fallback_top_k``.

    exact_rescore_p : Optional[float]
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``5e-07``.

    exact_rescore_top : int
        Number of top-ranked association candidates selected for exact rescoring regardless of
        the p-value threshold; zero disables this top-count route. Default: ``64``. Source
        comparisons: ``int(exact_rescore_top) > 0``. See the rejection/fallback branches for
        enforcement.

    exact_rescore_fn : Optional[Callable[..., Any]]
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.

    zarr_path : Optional[Union[str, Path]]
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    zarr_chunk_variants : int
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``8192``. Passed to
        ``_write_scan_metadata`` as ``chunk_variants``.

    zarr_shard_variants : int
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy. Default:
        ``65536``. Passed to ``_write_scan_metadata`` as ``shard_variants``.

    return_all : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    assume_complete : Any
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``'auto'``. Passed to
        ``calibrate_xarray_score_state`` as ``assume_complete``.

    score_workers : Optional[int]
        Concurrent scoring threads. Default: ``None`` (``usable_cpu_count()``), reduced so every
        worker can hold a 256-variant block of ``budget / workers`` and to the number of score
        chunks. The packed-BED direct path always uses 1 (its kernel is already parallel).

    progress_callback : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Default: ``None``.

    maxmem, memory_fraction : budget or None
        Working-memory budget for I/O and score chunks; see ``open_genotype_xarray``.
        Default: ``None`` (80% of physical RAM, capped at available RAM).

    Returns
    -------
    object
        Computed result from ``ScanResult(table, top, regions, diag, calibrated_states)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .resources import resolve_variant_chunk, usable_cpu_count, working_memory_bytes
    xr=_require_xarray()
    g=open_genotype_xarray(genotypes,npplink_module=npplink_module,io_chunk_variants=io_chunk_variants,dtype=np.float32,use_memmap=True,sample_ids=sample_ids,maxmem=maxmem,memory_fraction=memory_fraction);g=_align_samples(g,sample_ids)
    global_index=np.arange(g.sizes["snp"],dtype=np.int64)  # genome-wide variant positions reported in outputs
    if chromosomes is not None and "chrom" in g.coords:
        # Restrict the lazy view to the requested chromosomes before any rechunk or compute;
        # a genome-wide graph is never built for a per-chromosome scan.
        global_index=np.flatnonzero(np.isin(np.asarray(g.coords["chrom"].values).astype(str),[str(c) for c in chromosomes]))
        _attrs=dict(g.attrs);g=g.isel(snp=global_index);g.attrs=_attrs
    io_chunk=int(g.attrs["io_chunk_variants"]);n_iid=int(g.sizes["iid"])
    _packed_direct=(str(g.attrs.get("physical_backend",""))=="plink1_bed_2bit" and "variant_index" in g.coords and "sample_index" in g.coords)
    # Score scratch per call: float32 decode + float32 missing mask + filled copy + squared temp + finiteness bool ~ 20 B.
    # Pool threads each hold one block, so workers are capped to those that can afford a 256-variant block of budget/workers.
    _bpc=20;req_workers=int(score_workers) if score_workers else usable_cpu_count()
    req_workers=1 if _packed_direct else max(1,min(req_workers,working_memory_bytes(maxmem,memory_fraction)//max(1,n_iid*_bpc*256)))
    score_chunk=resolve_variant_chunk(score_chunk_variants or io_chunk,n_iid,bytes_per_call=_bpc,maxmem=maxmem,memory_fraction=memory_fraction,workers=req_workers,n_variants=g.sizes["snp"])
    if score_chunk<io_chunk and getattr(g.variable._data,"chunks",None) is not None and not _packed_direct:
        # Keep Dask storage chunks no wider than a worker's score block so threads never pull budget-sized chunks each.
        io_chunk=score_chunk;g=g.chunk({"iid":-1,"snp":io_chunk});g.attrs["io_chunk_variants"]=io_chunk
    cc_all=np.asarray(g.coords["chrom"].values).astype(str) if "chrom" in g.coords else np.full(g.sizes["snp"],"",object)
    available=list(dict.fromkeys(cc_all.tolist()));use=available if chromosomes is None else [str(x) for x in chromosomes]
    top_parts=[];region_parts=[];all_parts=[];calibrated_states={};factor_ranks={};nscan=0;effective_workers=[]
    keep_p=float(region_p_threshold) if p_threshold is None else max(float(region_p_threshold),float(p_threshold))
    names_all=np.asarray(g.coords["snp"].values).astype(str);pos_all=np.asarray(g.coords["pos"].values,np.int64) if "pos" in g.coords else np.arange(g.sizes["snp"],dtype=np.int64)
    cm_name=next((nm for nm in ("genetic_cm","cm","cM","genetic_pos_cm") if nm in g.coords),None);cm_all=np.asarray(g.coords[cm_name].values,np.float64) if cm_name is not None else None
    null_mode="loco_mv_score_fast_xarray" if (callable(state) or isinstance(state,Mapping)) else "mv_score_fast_shared_null_xarray"
    total_variants=int(np.isin(cc_all,use).sum())
    for cc in use:
        idx=_chrom_indices(g,cc)
        if not len(idx): continue
        base_state=_resolve_score_state(state,cc)
        if g.sizes["iid"]!=base_state.A.shape[0]:raise ValueError(f"score state rows do not match genotype samples for chromosome {cc}")
        cstate=calibrate_xarray_score_state(g,_replace(base_state,calibration=None),chromosomes=[cc],sample_ids=None,n_calibration=calibration_n,io_chunk_variants=io_chunk,calibration_io_windows=calibration_io_windows,assume_complete=assume_complete,maxmem=maxmem,memory_fraction=memory_fraction)
        calibrated_states[str(cc)]=cstate;factor_ranks[str(cc)]=int(cstate.A.shape[1])
        names=[];poss=[];cms=[];vidxs=[];stats=[];ps=[];nlps=[];nobss=[];fzs=[]
        chunk_ids=[idx[a0:a0+score_chunk] for a0 in range(0,len(idx),score_chunk)]
        workers=max(1,min(req_workers,len(chunk_ids)))  # packed-direct stays at 1 (numba prange)
        effective_workers.append(workers)
        def _score_gi(gi):
            """Evaluate gi.

            Parameters
            ----------
            gi : object (see the cited input contract)
                Input supplied to ``g.isel`` as ``snp``. See that callable's documented ``snp`` contract
                for its type, alignment and value domain. This wrapper retains that contract. Passed to
                ``g.isel`` as ``snp``.
            """
            block=g.isel(snp=gi)
            return score_xarray_block_lazy(block,cstate,return_factor_z=return_factor_z,assume_complete=assume_complete)
        if workers==1:
            scored=map(_score_gi,chunk_ids)
            _ctx=None
        else:
            from concurrent.futures import ThreadPoolExecutor
            try:
                from threadpoolctl import threadpool_limits
                _ctx=threadpool_limits(limits=1)
                _ctx.__enter__()
            except Exception:
                _ctx=None
            _ex=ThreadPoolExecutor(max_workers=workers)
            scored=_ex.map(_score_gi,chunk_ids)
        local_scanned=0
        try:
            for gi,z in zip(chunk_ids,scored):
                local_scanned+=len(gi)
                if progress_callback is not None:progress_callback({"phase":"score_chunk","chromosome":cc,"processed_variants":nscan+local_scanned,"total_variants":total_variants,"chromosome_processed":local_scanned,"chromosome_total":len(idx)})
                stats.append(z["stat"]);ps.append(z["p"]);nlps.append(z["neglog_p"]);nobss.append(z["nobs"])
                if return_factor_z:fzs.append(z["factor_z"])
                names.append(names_all[gi]);poss.append(pos_all[gi]);vidxs.append(global_index[gi])
                if cm_all is not None:cms.append(cm_all[gi])
        finally:
            if workers>1:
                _ex.shutdown(wait=True)
                if _ctx is not None:_ctx.__exit__(None,None,None)
        names=np.concatenate(names);poss=np.concatenate(poss);gcm=np.concatenate(cms) if cms else None;vidxs=np.concatenate(vidxs);stat=np.concatenate(stats);pv=np.concatenate(ps);nlp=np.concatenate(nlps);nobs=np.concatenate(nobss);fz=np.concatenate(fzs,0) if fzs else None
        ds=_chrom_dataset(cc,names,poss,vidxs,stat,pv,nlp,nobs,fz,cstate,genetic_cm=gcm)
        # Define surviving loci first. Exact rescoring is limited to their leads
        # plus the strongest fast-score candidates, and happens once/chromosome.
        pretop=_top_variants(ds,top_k,keep_p);prereg=candidate_regions_from_scan(pretop,lead_p=region_p_threshold,padding_bp=region_padding_bp,recombination_rate_cM_per_mb=region_recombination_rate_cM_per_mb,merge_recombination_cM=region_merge_recombination_cM,max_regions=None,fallback_top_k=region_fallback_top_k)
        lead_local=np.array([],np.int64)
        if prereg.sizes.get("region",0) and "lead_variant_index" in prereg.coords:
            lead_global=np.asarray(prereg["lead_variant_index"].values,np.int64);lookup={int(v):i for i,v in enumerate(np.asarray(ds["variant_index"].values,np.int64))};lead_local=np.asarray([lookup[int(v)] for v in lead_global if int(v) in lookup],np.int64)
        sig=_candidate_positions(ds,exact_rescore_p,0);extra=np.setdiff1d(sig,lead_local,assume_unique=False)
        if int(exact_rescore_top)>0:
            budget=max(0,int(exact_rescore_top)-len(lead_local));pp=np.asarray(ds["omnibus_p"].values,float);extra=extra[np.argsort(pp[extra])[:budget]] if len(extra) else extra
        cand=np.unique(np.r_[lead_local,extra])
        if exact_rescore_fn is not None and len(cand):
            gi=np.asarray(ds["variant_index"].values,np.int64)[cand];Xex=_xarray_compute(g.isel(snp=np.searchsorted(global_index,gi)),np.float32);ex=exact_rescore_fn(Xex,np.asarray(ds["snp"].values).astype(str)[cand],np.asarray(ds["pos"].values,np.int64)[cand],gi,chromosome=str(cc));ds=_merge_exact(ds,ex)
        top=_top_variants(ds,top_k,keep_p);regions=candidate_regions_from_scan(top,lead_p=region_p_threshold,padding_bp=region_padding_bp,recombination_rate_cM_per_mb=region_recombination_rate_cM_per_mb,merge_recombination_cM=region_merge_recombination_cM,max_regions=None,fallback_top_k=region_fallback_top_k)
        if top.sizes.get("variant",0): top_parts.append(top)
        if regions.sizes.get("region",0): region_parts.append(regions)
        nscan+=len(ds.variant)
        if zarr_path is not None:_write_chrom_zarr(ds,zarr_path,cc,zarr_chunk_variants,zarr_shard_variants)
        elif return_all:all_parts.append(ds)
    if zarr_path is not None:
        try:
            import zarr
            # Root attrs are part of consolidated metadata, so write them before consolidation.
            _write_scan_metadata(zarr_path,chromosomes=use,n_variants=nscan,chunk_variants=zarr_chunk_variants,
                shard_variants=zarr_shard_variants,engine=null_mode,assume_complete=assume_complete,
                consolidated=True,factor_ranks=factor_ranks)
            zarr.consolidate_metadata(str(zarr_path));consolidated=True;cerr=None
        except Exception as exc:consolidated=False;cerr=str(exc)
        table=open_scan_zarr3(zarr_path,use) if return_all and consolidated else None
    elif return_all:
        table=xr.concat(all_parts,dim="variant",data_vars="all",coords="minimal",compat="override",join="outer") if all_parts else xr.Dataset()
        if table.sizes.get("variant",0):table=table.assign_coords(variant=np.arange(table.sizes["variant"],dtype=np.int64))
        consolidated=False;cerr=None
    else:table=None;consolidated=False;cerr=None
    top=xr.concat(top_parts,dim="variant",data_vars="all",coords="minimal",compat="override",join="outer") if top_parts else xr.Dataset()
    if top.sizes.get("variant",0):top=top.assign_coords(variant=np.arange(top.sizes["variant"],dtype=np.int64));top=_top_variants(top,top_k,keep_p)
    regions=xr.concat(region_parts,dim="region",data_vars="all",coords="minimal",compat="override",join="outer") if region_parts else xr.Dataset()
    if regions.sizes.get("region",0):
        order=np.argsort(np.asarray(regions["lead_p"].values,float));order=order if max_regions is None else order[:int(max_regions)];regions=regions.isel(region=order).assign_coords(region=np.arange(len(order),dtype=np.int32))
    diag={"engine":null_mode,"n_variants_scanned":int(nscan),"factor_ranks":factor_ranks,"chromosomes":use,"io_chunk_variants":io_chunk,"score_chunk_variants":score_chunk,"calibration_io_windows":int(calibration_io_windows),"score_workers":max(effective_workers,default=0),"requested_score_workers":int(score_workers or usable_cpu_count()),"lazy_xarray":True,"assume_complete":(str(assume_complete) if isinstance(assume_complete,str) else bool(assume_complete)),"region_p_threshold":float(region_p_threshold),"region_padding_bp":int(region_padding_bp),"n_regions":int(regions.sizes.get("region",0)),"zarr_path":None if zarr_path is None else str(zarr_path),"zarr_format":3 if zarr_path is not None else None,"zarr_sharded":bool(zarr_path is not None),"zarr_consolidated":consolidated}
    if cerr:diag["zarr_consolidation_error"]=cerr
    return ScanResult(table,top,regions,diag,calibrated_states)


scan_xarray = scan_xarray_recombination


# ---------------------------------------------------------------------------
# VIGOR 2.11 direct PLINK1 BED byte scorer
# ---------------------------------------------------------------------------
def prepare_packed_score_tables(state, prefix, *, sample_ids=None, precision='exact'):
    """Build fixed BED-byte lookup tables for a FastScoreState without decoding genotypes.

    Parameters
    ----------
    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes. Passed to ``build_packed_score_tables`` as ``fit_mask``.

    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    precision : str
        Input for the operation ``str(precision).lower()``. See
        ``vigor.mvgwas_fast.prepare_packed_score_tables`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'exact'``. Source-recognized literal values: ``float64``, ``high``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    Returns
    -------
    object
        Computed result from ``build_packed_score_tables(state.A, len(all_ids), sample_indices=si, fit_mask=state.fit_mask, dtype=dtype)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .npplink import read_fam_bim
    from .packed_bed import build_packed_score_tables
    fam,_=read_fam_bim(str(prefix)); all_ids=fam['iid'].astype(str).to_numpy()
    if sample_ids is None:
        if len(all_ids)!=state.A.shape[0]: raise ValueError('sample_ids are required when score-state rows are a genotype subset')
        si=np.arange(len(all_ids),dtype=np.int64)
    else:
        wanted=np.asarray(sample_ids).astype(str); lookup={x:i for i,x in enumerate(all_ids)}
        missing=[x for x in wanted if x not in lookup]
        if missing: raise KeyError(f'score samples absent from FAM: {missing[:5]}')
        si=np.asarray([lookup[x] for x in wanted],np.int64)
        if len(si)!=state.A.shape[0]: raise ValueError('sample_ids length does not match score-state rows')
    key=str(precision).lower(); dtype=np.float64 if key in {'exact','float64','high'} else np.float32
    return build_packed_score_tables(state.A,len(all_ids),sample_indices=si,fit_mask=state.fit_mask,dtype=dtype)


def score_packed_bed_block(packed, state, tables, *, return_factor_z=False, return_allele_frequency=False):
    """Score a packed variant block directly; exact missingness and no dosage matrix.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    tables : object (see the cited input contract)
        Input for the operation ``packed_score_components_fast(packed, tables,
        return_sx=bool(return_allele_frequency))``. See
        ``vigor.mvgwas_fast.score_packed_bed_block`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    return_factor_z : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    return_allele_frequency : bool
        Input supplied to ``packed_score_components_fast`` as ``return_sx``. See that callable's
        documented ``return_sx`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``False``. Passed to ``packed_score_components_fast`` as
        ``return_sx``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if state.calibration is None: raise ValueError('state is not calibrated')
    from .packed_bed import packed_score_components_fast
    q=packed_score_components_fast(packed,tables,return_sx=bool(return_allele_frequency))
    if return_allele_frequency: U,ss,cnt,sx=q
    else: U,ss,cnt=q
    root=np.sqrt(ss)
    Z=np.divide(U,root[:,None],out=np.zeros_like(U),where=root[:,None]>0)
    cal=state.calibration; Q=np.einsum('ij,jk,ik->i',Z,cal.inverse,Z,optimize=True)/float(cal.lambda_gc)
    p=chi2.sf(Q,cal.rank); out={'stat':Q,'p':p,'neglog_p':-np.log10(np.clip(p,np.finfo(float).tiny,1.0)),'nobs':cnt,'ss':ss}
    if return_factor_z:
        den=np.sqrt(np.clip(np.diag(cal.covariance)[None,:]*ss[:,None]*float(cal.lambda_gc),1e-30,None))
        out['factor_z']=np.divide(U,den,out=np.zeros_like(U),where=den>0)
    if return_allele_frequency:
        out['allele_frequency']=np.divide(np.asarray(sx,float),2.0*np.asarray(cnt,float),out=np.full(len(cnt),np.nan),where=np.asarray(cnt)>0)
    return out

def calibrate_packed_score_state(prefix, state, tables=None, *, chromosome=None, sample_ids=None,
                                 n_calibration=4096, io_chunk_variants='auto',
                                 calibration_io_windows=4, precision='exact'):
    """Calibrate factor-score covariance from packed BED variants only.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    tables : array-like or numeric scalar
        Input for the operation ``packed_score_components_fast(np.asarray(mm[ci], np.uint8),
        tables)``. See ``vigor.mvgwas_fast.calibrate_packed_score_state`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``prepare_packed_score_tables`` as ``sample_ids``.

    n_calibration : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4096``.

    io_chunk_variants : int or 'auto'
        Accepted for API compatibility; packed calibration reads its evenly spaced SNPs
        directly, so the value does not affect results. Default: ``'auto'``.

    calibration_io_windows : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4``.

    precision : str
        Input supplied to ``prepare_packed_score_tables`` as ``precision``. See that callable's
        documented ``precision`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``'exact'``. Passed to ``prepare_packed_score_tables``
        as ``precision``.

    Returns
    -------
    object
        Computed result from ``(FastScoreState(state.A, state.basis, ScoreCalibration(M, Mi, rank, lam, int(len(q))), state.trait_names, state.fit_mask), tables)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .npplink import read_fam_bim
    from .packed_bed import packed_score_components_fast
    fam,bim=read_fam_bim(str(prefix)); n=len(fam); m=len(bim); bpv=(n+3)//4
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m)
    idx=np.arange(m,dtype=np.int64)
    if chromosome is not None:
        chrom=np.asarray(bim['chrom']).astype(str); idx=np.flatnonzero(chrom==str(chromosome))
    # Packed BED random access is cheap and does not require Dask chunk locality.
    # Keep calibration SNPs invariant to scan chunk size so lambda/statistics do
    # not change when only the I/O tuning changes.
    if len(idx)<=int(n_calibration):
        ci=idx
    else:
        ci=idx[np.unique(np.linspace(0,len(idx)-1,int(n_calibration),dtype=np.int64))]
    if tables is None: tables=prepare_packed_score_tables(state,prefix,sample_ids=sample_ids,precision=precision)
    null_tables=prepare_packed_score_tables(_null_view(state),prefix,sample_ids=sample_ids,precision=precision)
    U0,ss,_=packed_score_components_fast(np.asarray(mm[ci],np.uint8),null_tables)
    return _replace(state,calibration=_calibration_from_null(U0,ss,state.A.shape[1])),tables

def scan_packed_bed(prefix, state, *, chromosomes=None, sample_ids=None, score_chunk_variants='auto',
                    calibration_n=4096, calibration_io_windows=4, precision='exact', min_maf=None, top_k=10000,
                    keep_p=1e-3, return_factor_z=True, region_p_threshold=1e-5,
                    region_padding_bp=1_000_000, region_recombination_rate_cM_per_mb=.5,
                    region_merge_recombination_cM=1.0, max_regions=64, region_fallback_top_k=8,
                    zarr_path=None, zarr_chunk_variants=8192, zarr_shard_variants=65536,
                    return_all=False, maxmem=None, memory_fraction=None):
    """Direct 2-bit PLINK1 mvGWAS discovery scan without genotype decoding.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    state : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``calibrate_packed_score_state`` as ``sample_ids``.

    score_chunk_variants : int or 'auto'
        Variants per packed scoring block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``'auto'`` (sized from the working-memory
        budget; integers are ceilings).

    calibration_n : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4096``. Passed to ``calibrate_packed_score_state`` as ``n_calibration``.

    calibration_io_windows : int
        Number of variants/reference cases used to calibrate the approximate association scan.
        Positive counts trade calibration work for improved agreement with exact scores.
        Default: ``4``. Passed to ``calibrate_packed_score_state`` as
        ``calibration_io_windows``.

    precision : str
        Input supplied to ``calibrate_packed_score_state`` as ``precision``. See that callable's
        documented ``precision`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``'exact'``. Passed to ``calibrate_packed_score_state``
        as ``precision``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value. Passed to ``score_packed_bed_block`` as
        ``return_allele_frequency``.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``10000``.

    keep_p : float
        Input for the operation ``_top_variants(ds, top_k, keep_p)``. See
        ``vigor.mvgwas_fast.scan_packed_bed`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``0.001``.

    return_factor_z : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``. Passed to ``score_packed_bed_block`` as
        ``return_factor_z``.

    region_p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``1e-05``. Passed to ``candidate_regions_from_scan`` as ``lead_p``.

    region_padding_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``candidate_regions_from_scan`` as ``padding_bp``.

    region_recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``. Passed to ``candidate_regions_from_scan`` as
        ``recombination_rate_cM_per_mb``.

    region_merge_recombination_cM : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``1.0``. Passed to ``candidate_regions_from_scan`` as ``merge_recombination_cM``.

    max_regions : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``64``.

    region_fallback_top_k : int
        Input supplied to ``candidate_regions_from_scan`` as ``fallback_top_k``. See that
        callable's documented ``fallback_top_k`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``8``. Passed to
        ``candidate_regions_from_scan`` as ``fallback_top_k``.

    zarr_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    zarr_chunk_variants : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``8192``. Passed to
        ``_write_scan_metadata`` as ``chunk_variants``.

    zarr_shard_variants : int or None
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy. Default:
        ``65536``. Passed to ``_write_scan_metadata`` as ``shard_variants``.

    return_all : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    maxmem, memory_fraction : budget or None
        Working-memory budget for score blocks; see ``vigor.resources.working_memory_bytes``.
        Default: ``None`` (80% of physical RAM, capped at available RAM).

    Returns
    -------
    object
        Computed result from ``ScanResult(table, top, regions, diag, calibrated_states)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .npplink import read_fam_bim
    from .resources import resolve_variant_chunk
    fam,bim=read_fam_bim(str(prefix)); n=len(fam); m=len(bim); bpv=(n+3)//4; used_chunks=[]
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m)
    chrom_all=np.asarray(bim['chrom']).astype(str); names_all=bim['snp'].astype(str).to_numpy(); pos_all=bim['pos'].to_numpy(np.int64)
    use=list(dict.fromkeys(chrom_all.tolist())) if chromosomes is None else list(map(str,chromosomes))
    all_parts=[]; top_parts=[]; region_parts=[]; calibrated_states={}; factor_ranks={}; nscan=0
    if zarr_path is not None:
        p=Path(zarr_path)
        if p.exists():
            import shutil; shutil.rmtree(p)
    for cc in use:
        idx=np.flatnonzero(chrom_all==str(cc))
        if not len(idx): continue
        base=_resolve_score_state(state,cc)
        cstate,tables=calibrate_packed_score_state(prefix,_replace(base,calibration=None),
            chromosome=cc,sample_ids=sample_ids,n_calibration=calibration_n,
            io_chunk_variants=score_chunk_variants,calibration_io_windows=calibration_io_windows,precision=precision)
        calibrated_states[str(cc)]=cstate; factor_ranks[str(cc)]=int(cstate.A.shape[1])
        stats=[]; ps=[]; nlps=[]; nobss=[]; fzs=[]; kept_idx=[]
        # No dosage decode: memmap read + active-byte gather ~ 0.5 B/call; U, Z, factor_z float64 (3r) + scalars per variant.
        sc=resolve_variant_chunk(score_chunk_variants,n,bytes_per_call=0.5,bytes_per_variant=8*(3*int(cstate.A.shape[1])+8),maxmem=maxmem,memory_fraction=memory_fraction,n_variants=len(idx)); used_chunks.append(sc)
        for a0 in range(0,len(idx),sc):
            gi=idx[a0:a0+sc]; z=score_packed_bed_block(np.asarray(mm[gi],np.uint8),cstate,tables,return_factor_z=return_factor_z,return_allele_frequency=min_maf is not None)
            if min_maf is not None:
                af=np.asarray(z['allele_frequency'],float); kk=np.isfinite(af)&(np.minimum(af,1.0-af)>=float(min_maf))
                gi=gi[kk]
                for k in ('stat','p','neglog_p','nobs'):
                    z[k]=np.asarray(z[k])[kk]
                if return_factor_z:z['factor_z']=np.asarray(z['factor_z'])[kk]
            if not len(gi):continue
            kept_idx.append(gi);stats.append(z['stat']);ps.append(z['p']);nlps.append(z['neglog_p']);nobss.append(z['nobs'])
            if return_factor_z:fzs.append(z['factor_z'])
        if not kept_idx:continue
        qidx=np.concatenate(kept_idx);stat=np.concatenate(stats);pv=np.concatenate(ps);nlp=np.concatenate(nlps);nobs=np.concatenate(nobss);fz=np.concatenate(fzs,0) if fzs else None
        ds=_chrom_dataset(cc,names_all[qidx],pos_all[qidx],qidx,stat,pv,nlp,nobs,fz,cstate)
        top=_top_variants(ds,top_k,keep_p)
        regions=candidate_regions_from_scan(top,lead_p=region_p_threshold,padding_bp=region_padding_bp,
            recombination_rate_cM_per_mb=region_recombination_rate_cM_per_mb,
            merge_recombination_cM=region_merge_recombination_cM,max_regions=None,fallback_top_k=region_fallback_top_k)
        if top.sizes.get('variant',0):top_parts.append(top)
        if regions.sizes.get('region',0):region_parts.append(regions)
        nscan+=len(ds.variant)
        if zarr_path is not None:_write_chrom_zarr(ds,zarr_path,cc,zarr_chunk_variants,zarr_shard_variants)
        elif return_all:all_parts.append(ds)

    consolidated=False; cerr=None
    if zarr_path is not None:
        try:
            import zarr
            _write_scan_metadata(zarr_path,chromosomes=use,n_variants=nscan,chunk_variants=zarr_chunk_variants,
                shard_variants=zarr_shard_variants,engine='packed_bed_lut_'+str(precision),assume_complete='auto',
                consolidated=True,factor_ranks=factor_ranks)
            zarr.consolidate_metadata(str(zarr_path)); consolidated=True
        except Exception as exc: cerr=str(exc)
        table=open_scan_zarr3(zarr_path,use) if return_all and consolidated else None
    elif return_all:
        xr=_require_xarray(); table=xr.concat(all_parts,dim='variant',data_vars='all',coords='minimal',compat='override',join='outer') if all_parts else xr.Dataset()
        if table.sizes.get('variant',0):table=table.assign_coords(variant=np.arange(table.sizes['variant'],dtype=np.int64))
    else: table=None
    xr=_require_xarray(); top=xr.concat(top_parts,dim='variant',data_vars='all',coords='minimal',compat='override',join='outer') if top_parts else xr.Dataset()
    if top.sizes.get('variant',0):top=top.assign_coords(variant=np.arange(top.sizes['variant'],dtype=np.int64));top=_top_variants(top,top_k,keep_p)
    regions=xr.concat(region_parts,dim='region',data_vars='all',coords='minimal',compat='override',join='outer') if region_parts else xr.Dataset()
    if regions.sizes.get('region',0):
        order=np.argsort(np.asarray(regions['lead_p'].values,float));order=order if max_regions is None else order[:int(max_regions)]
        regions=regions.isel(region=order).assign_coords(region=np.arange(len(order),dtype=np.int32))
    diag={'engine':'packed_bed_lut_'+str(precision),'n_variants_scanned':int(nscan),'factor_ranks':factor_ranks,
          'chromosomes':use,'score_chunk_variants':max(used_chunks,default=0),'packed_bed':True,
          'genotype_decoding':False,'zarr_path':None if zarr_path is None else str(zarr_path),
          'zarr_format':3 if zarr_path is not None else None,'zarr_consolidated':bool(consolidated),
          'precision':str(precision),'min_maf':None if min_maf is None else float(min_maf),'n_regions':int(regions.sizes.get('region',0))}
    if cerr:diag['zarr_consolidation_error']=cerr
    return ScanResult(table,top,regions,diag,calibrated_states)

# Temporary public alias for callers and historical pickle globals.
state_from_domino_null = state_from_vigor_null
__all__.append("state_from_domino_null")
