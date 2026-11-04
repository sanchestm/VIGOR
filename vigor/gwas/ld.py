"""Bounded-memory LD primitives: the single pairwise-complete kernel and its users.

``pairwise_complete`` is VIGOR's one implementation of pairwise-complete
genotype correlation/covariance.  ``npplink.R2_exact``, ``_pairwise_r2``,
``finemap.graph._correlation`` and the packed-BED BLAS LD engine are thin
wrappers with explicit settings.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

__all__ = ["pairwise_complete", "R2Block", "local_correlation_matrix"]

#: Statistics accepted by :func:`pairwise_complete`.
LD_STATISTICS = ("r", "r2", "cov", "D", "D2", "chord")
#: Global missing-call rate up to which pair counts/sums use the sparse-missing complement.
SPARSE_MISSING_RATE = 0.05


class _CenteredColumns:
    """Column-centered float working copy with float64 per-column reductions.

    ``X`` holds ``x - mean(x)`` over each column's observed entries, zero where
    missing (Fortran order, so a variant block is one contiguous slab).
    ``S``/``SS`` are float64 sums of the centered values and their squares,
    ``cnt``/``nmiss`` the observed/missing counts and ``mean`` the centering
    means (needed for allele frequencies).
    """

    def __init__(self, A, dtype):
        X = np.array(A, dtype=dtype, order="F", copy=True)
        if X.ndim != 2:
            raise ValueError("pairwise_complete expects sample x variant matrices")
        miss = ~np.isfinite(X)
        np.copyto(X, 0, where=miss)
        self.nmiss = miss.sum(axis=0).astype(np.int64)
        self.cnt = X.shape[0] - self.nmiss
        self.mean = np.divide(X.sum(axis=0, dtype=np.float64), self.cnt,
                              out=np.zeros(X.shape[1]), where=self.cnt > 0)
        X -= self.mean.astype(dtype)
        np.copyto(X, 0, where=miss)
        self.S = X.sum(axis=0, dtype=np.float64)
        self.SS = np.einsum("ij,ij->j", X, X, dtype=np.float64)
        self.X = X
        self.miss = miss
        self.n, self.p = X.shape
        self.M = None

    def sparse(self, dtype):
        """Build the CSC missing-call indicator once (sparse-missing complement path)."""
        if self.M is None:
            from scipy import sparse
            self.M = sparse.csc_matrix(self.miss, dtype=dtype)
        return self.M

    def observed(self, a, b, dtype):
        """Dense observed-call indicator for columns ``a:b`` (dense-missing path)."""
        return np.asfortranarray(~self.miss[:, a:b], dtype=dtype)


def _ld_block_size(n, p, q, itemsize, block_size):
    """Variant tile width: the widest square tile the working-memory budget affords.

    One tile holds the float GEMM output plus ~9 float64 pair-statistic arrays
    (~80 bytes per variant pair) and, per block column, squared/indicator copies
    over the samples (~4 working-dtype values per sample).
    """
    width = max(int(p), int(q), 1)
    if block_size is not None and str(block_size).lower() != "auto":
        b = int(block_size)
        if b < 1:
            raise ValueError("block_size must be positive or 'auto'")
        return min(b, width)
    from ..resources import working_memory_bytes
    budget = float(working_memory_bytes())
    a = 80.0
    c = 4.0 * float(itemsize) * max(int(n), 1)
    b = int((-c + math.sqrt(c * c + 4.0 * a * budget)) / (2.0 * a))
    if b < 1:
        raise MemoryError("LD budget cannot fit one pairwise statistic tile")
    return min(b, width)


def pairwise_complete(A, B=None, *, statistic="r2", min_n=3, nan_policy="pairwise", ddof=0,
                      dtype=np.float32, return_counts=False, block_size="auto"):
    """Pairwise-complete LD statistics between the variant columns of ``A`` and ``B``.

    Every variant column is centered by its own observed mean (float64 means,
    stored in ``dtype``), so the heavy contraction over samples is one GEMM of
    centered data per ``(block_i, block_j)`` tile. Pair-specific counts and sums
    over the jointly observed samples come from the missing-call complement,
    ``n_ij = n - miss_i - miss_j + Miss_i' Miss_j`` and
    ``sum_x = S_i - X_i' Miss_j``, evaluated with sparse products when
    missingness is sparse (<= ``SPARSE_MISSING_RATE``) and with indicator GEMMs
    otherwise; complete tiles need no correction at all. All pairwise
    reductions (centered cross-products, variances, divisions) run in float64.

    Parameters
    ----------
    A : array-like, shape (n, p)
        Samples in rows, variants in columns; NaN/inf mark missing calls.
    B : array-like, shape (n, q) or None
        Second variant block on the same sample rows. None computes the
        symmetric ``A`` versus ``A`` matrix (upper tiles only, mirrored).
    statistic : {'r2', 'r', 'cov', 'D', 'D2', 'chord'}
        ``'r'`` signed Pearson correlation, ``'r2'`` its square, ``'cov'`` the
        covariance ``C_xy / (n_ij - ddof)``, ``'D'``/``'D2'`` the correlation
        rescaled by ``sqrt(p_x q_x p_y q_y)`` with pair-specific allele
        frequencies ``mean/2`` (clipped to [1e-12, 1-1e-12]), ``'chord'``
        ``sqrt(2 (1 - r))``. Default ``'r2'``.
    min_n : int
        Pairs with fewer than ``min_n`` jointly observed samples are NaN
        (``nan_policy='mean'``: either column with fewer than ``min_n`` observed
        calls). Default 3.
    nan_policy : {'pairwise', 'mean'}
        ``'pairwise'`` uses the jointly observed samples of each pair (pairwise
        deletion). ``'mean'`` mean-imputes each column and uses all n samples.
    ddof : int
        Delta degrees of freedom for ``statistic='cov'``. Default 0.
    dtype : {numpy.float32, numpy.float64}
        Working copy, GEMM and output precision. Default float32.
    return_counts : bool
        Also return the jointly observed sample counts (int32, shape (p, q)).
    block_size : int or 'auto'
        Variant tile width; 'auto' uses the widest tile the memory budget affords.

    Returns
    -------
    numpy.ndarray, shape (p, q), dtype ``dtype``
        Statistic matrix; NaN for pairs below ``min_n``, zero-variance pairs and
        undefined values. ``(values, counts)`` when ``return_counts``.
    """
    stat = str(statistic)
    if stat not in LD_STATISTICS:
        raise ValueError(f"statistic has to be in {list(LD_STATISTICS)}")
    policy = str(nan_policy).lower()
    if policy in {"mean_impute", "impute"}:
        policy = "mean"
    if policy not in {"pairwise", "mean"}:
        raise ValueError("nan_policy must be 'pairwise' or 'mean'")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    if isinstance(min_n, (bool, np.bool_)) or not isinstance(min_n, (int, np.integer)) or min_n < 1:
        raise ValueError("min_n must be a positive integer")
    min_n = int(min_n)
    ddof = float(ddof)
    if not np.isfinite(ddof) or ddof < 0:
        raise ValueError("ddof must be finite and nonnegative")
    x = _CenteredColumns(A, dtype)
    symmetric = B is None
    y = x if symmetric else _CenteredColumns(B, dtype)
    if y.n != x.n:
        raise ValueError("A and B must have the same number of rows (samples)")
    n, p, q = x.n, x.p, y.p
    out = np.empty((p, q), dtype=dtype)
    counts = np.empty((p, q), dtype=np.int32) if return_counts else None
    if p == 0 or q == 0:
        return (out, counts) if return_counts else out
    bs = _ld_block_size(n, p, q, dtype.itemsize, block_size)
    calls = n * (p if symmetric else p + q)
    total_missing = int(x.nmiss.sum()) + (0 if symmetric else int(y.nmiss.sum()))
    sparse_missing = total_missing <= SPARSE_MISSING_RATE * max(calls, 1)
    need_pair = policy == "pairwise"
    need_n = need_pair or return_counts
    tol = 16.0 * float(np.finfo(dtype).eps)
    for i0 in range(0, p, bs):
        i1 = min(p, i0 + bs)
        Xi = x.X[:, i0:i1]
        mi = x.nmiss[i0:i1]
        Xi_sq = Oi = None
        for j0 in range(i0 if symmetric else 0, q, bs):
            j1 = min(q, j0 + bs)
            Yj = y.X[:, j0:j1]
            mj = y.nmiss[j0:j1]
            sxy = np.asarray(Xi.T @ Yj, dtype=np.float64)  # the one contraction over samples
            Si, Sj = x.S[i0:i1, None], y.S[None, j0:j1]
            SSi, SSj = x.SS[i0:i1, None], y.SS[None, j0:j1]
            nn = float(n)
            sx, sy, sxx, syy = Si, Sj, SSi, SSj
            if need_n and (mi.any() or mj.any()):
                if Xi_sq is None and need_pair:
                    Xi_sq = Xi * Xi
                Yj_sq = (Yj * Yj) if need_pair else None
                if sparse_missing:
                    Mi = x.sparse(dtype)[:, i0:i1]
                    Mj = y.sparse(dtype)[:, j0:j1]
                    nn = (float(n) - mi[:, None] - mj[None, :]
                          + np.asarray((Mi.T @ Mj).toarray(), dtype=np.float64))
                    if need_pair:
                        sx = Si - np.asarray(Mj.T @ Xi, dtype=np.float64).T
                        sy = Sj - np.asarray(Mi.T @ Yj, dtype=np.float64)
                        sxx = SSi - np.asarray(Mj.T @ Xi_sq, dtype=np.float64).T
                        syy = SSj - np.asarray(Mi.T @ Yj_sq, dtype=np.float64)
                else:
                    if Oi is None:
                        Oi = x.observed(i0, i1, dtype)
                    Oj = y.observed(j0, j1, dtype)
                    nn = np.asarray(Oi.T @ Oj, dtype=np.float64)
                    if need_pair:
                        sx = np.asarray(Xi.T @ Oj, dtype=np.float64)
                        sy = np.asarray(Oi.T @ Yj, dtype=np.float64)
                        sxx = np.asarray(Xi_sq.T @ Oj, dtype=np.float64)
                        syy = np.asarray(Oi.T @ Yj_sq, dtype=np.float64)
            if need_pair:
                nsafe = np.maximum(nn, 1.0)
                cxy = sxy - sx * sy / nsafe
                cxx = sxx - sx * sx / nsafe
                cyy = syy - sy * sy / nsafe
                valid = np.asarray(nn) >= min_n
            else:
                nsafe = float(n)
                cxy, cxx, cyy = sxy, SSi, SSj
                valid = (x.cnt[i0:i1, None] >= min_n) & (y.cnt[None, j0:j1] >= min_n)
            with np.errstate(divide="ignore", invalid="ignore"):
                if stat == "cov":
                    den = nsafe - ddof
                    val = cxy / den
                    bad = ~(valid & (np.asarray(den) > 0))
                else:
                    zero = (cxx <= tol * np.maximum(sxx, 0.0)) | (cyy <= tol * np.maximum(syy, 0.0))
                    val = np.clip(cxy / np.sqrt(np.maximum(cxx, 0.0) * np.maximum(cyy, 0.0)), -1.0, 1.0)
                    bad = zero | ~valid
                    if stat == "r2":
                        val = val * val
                    elif stat == "chord":
                        val = np.sqrt(2.0 * (1.0 - val))
                    elif stat in {"D", "D2"}:
                        if need_pair:
                            px = np.clip((x.mean[i0:i1, None] + sx / nsafe) * 0.5, 1e-12, 1 - 1e-12)
                            py = np.clip((y.mean[None, j0:j1] + sy / nsafe) * 0.5, 1e-12, 1 - 1e-12)
                        else:
                            px = np.clip(x.mean[i0:i1, None] * 0.5, 1e-12, 1 - 1e-12)
                            py = np.clip(y.mean[None, j0:j1] * 0.5, 1e-12, 1 - 1e-12)
                        scale = px * (1 - px) * py * (1 - py)
                        val = val * np.sqrt(scale) if stat == "D" else val * val * scale
            val = np.broadcast_to(val, (i1 - i0, j1 - j0)).astype(dtype, copy=True)
            val[np.broadcast_to(bad, val.shape)] = np.nan
            out[i0:i1, j0:j1] = val
            if symmetric and j0 != i0:
                out[j0:j1, i0:i1] = val.T
            if counts is not None:
                cnt = np.rint(np.broadcast_to(nn, val.shape)).astype(np.int32)
                counts[i0:i1, j0:j1] = cnt
                if symmetric and j0 != i0:
                    counts[j0:j1, i0:i1] = cnt.T
    return (out, counts) if return_counts else out


@dataclass
class R2Block:
    """R2Block represents block.

    Attributes
    ----------
    i : np.ndarray
        Stored i; its axis/order follows the construction inputs.
    j : np.ndarray
        Stored j; its axis/order follows the construction inputs.
    r2 : np.ndarray
        Stored r2; its axis/order follows the construction inputs.
    n_pair : np.ndarray
        Stored n pair; its axis/order follows the construction inputs.
    distance_bp : np.ndarray | None
        Stored distance bp; its axis/order follows the construction inputs.
    distance_cm : np.ndarray | None
        Stored distance cm; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    i : np.ndarray
        Zero-based first-variant positions for each emitted pair. Required construction field.

    j : np.ndarray
        Zero-based second-variant positions for each emitted pair. Required construction field.

    r2 : np.ndarray
        Pairwise squared correlations aligned with i/j. Required construction field.

    n_pair : np.ndarray
        Jointly observed sample counts for the corresponding variant pairs. Required
        construction field.

    distance_bp : np.ndarray | None
        Absolute pair separation in base pairs; None when no physical coordinates are supplied.
        Default: ``None``.

    distance_cm : np.ndarray | None
        Absolute pair separation in centimorgans; None when no genetic map is supplied. Default:
        ``None``.
    """
    i: np.ndarray
    j: np.ndarray
    r2: np.ndarray
    n_pair: np.ndarray
    distance_bp: np.ndarray | None = None
    distance_cm: np.ndarray | None = None


def _debiased_r2(r2, n):
    """Remove the finite-sample null expectation 1/(n-1), pair by pair.

    Parameters
    ----------
    r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    n : array-like
        Jointly observed sample count for each variant pair, aligned to r2. Counts determine the
        finite-sample null correlation expectation 1/(n-1); values are floored through
        max(n-1,1).
    """
    r2 = np.asarray(r2, dtype=float)
    n = np.asarray(n, dtype=float)
    null = 1.0 / np.maximum(n - 1.0, 1.0)
    return np.maximum((r2 - null) / np.maximum(1.0 - null, np.finfo(float).eps), 0.0)


def _pairwise_r2(A, B, min_n=3, dtype=np.float32):
    """Pairwise-complete Pearson r2 and joint counts (wrapper of :func:`pairwise_complete`).

    Parameters
    ----------
    A : two-dimensional array-like
        Sample × variant dosage block. Both blocks must share the same sample rows; finite masks
        retain pairwise-complete observations and pairs with fewer than ``min_n`` joint calls
        yield NaN.

    B : two-dimensional array-like
        Sample × variant dosage block on the same sample rows as ``A``.

    min_n : int
        Minimum jointly observed samples per pair. Default: ``3``.

    dtype : NumPy floating dtype
        Working/GEMM precision of the shared kernel (centered data, float64 reductions).
        Default: ``np.float32``.

    Returns
    -------
    tuple
        ``(r2, n)``: r2 in [0, 1] with NaN for undefined pairs, and int32 joint counts.
    """
    return pairwise_complete(A, B, statistic="r2", min_n=min_n, nan_policy="pairwise",
                             dtype=dtype, return_counts=True)


def _iter_off_diagonal_r2(snps, *, block_size=200, max_variant_lag=None,
                          min_variant_lag=0, positions=None, max_bp=None,
                          genetic_map=None, max_cm=None, backend="exact",
                          upper_only=True, return_pair_counts=False, statistic="r2"):
    """Yield local pairwise LD blocks without constructing an M by M matrix.

    ``snps`` is sample by variant. Pairwise-complete exact LD from
    :func:`pairwise_complete` is the only implemented calculation; ``legacy``
    is an explicit compatibility alias for it, not a separate approximation
    backend. ``statistic`` ('r2' default, or 'r' for signed correlation) selects
    the value stored in ``R2Block.r2``.

    Parameters
    ----------
    snps : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``200``.

    max_variant_lag : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``None``.

    min_variant_lag : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``0``.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    max_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``None``.

    genetic_map : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.

    max_cm : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``.

    backend : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'exact'``. Source-recognized literal
        values: ``legacy``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    upper_only : bool
        Whether to emit only i<j off-diagonal variant pairs. False includes both directions
        while still excluding the diagonal. Default: ``True``.

    return_pair_counts : bool
        Compatibility flag accepted by the LD iterator. R2Block already contains n_pair; this
        current implementation does not branch on the flag. Default: ``False``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    if backend not in {"exact", "legacy"}:
        raise ValueError("backend must be 'exact' or 'legacy'")
    if statistic not in {"r2", "r"}:
        raise ValueError("statistic must be 'r2' or 'r'")
    X = np.asarray(snps)
    if X.ndim != 2: raise ValueError("snps must be sample x variant")
    m = X.shape[1]; bs = max(1, int(block_size))
    pos = None if positions is None else np.asarray(positions, dtype=float)
    cm = None if genetic_map is None else np.asarray(genetic_map, dtype=float)
    if pos is not None and len(pos) != m: raise ValueError("positions length mismatch")
    if cm is not None and len(cm) != m: raise ValueError("genetic_map length mismatch")
    lag = m - 1 if max_variant_lag is None else int(max_variant_lag)
    if max_bp is not None and pos is None: raise ValueError("max_bp requires positions")
    if max_cm is not None and cm is None: raise ValueError("max_cm requires genetic_map")
    for a0 in range(0, m, bs):
        a1 = min(m, a0 + bs)
        # only blocks that could contain allowed lags
        bmax = min(m, a1 + lag)
        if max_bp is not None:
            bmax = min(bmax, int(np.searchsorted(pos, pos[a1 - 1] + float(max_bp), side="right")))
        if max_cm is not None:
            bmax = min(bmax, int(np.searchsorted(cm, cm[a1 - 1] + float(max_cm), side="right")))
        for b0 in range(a0, bmax, bs):
            b1 = min(m, b0 + bs, bmax)
            rr, nn = pairwise_complete(X[:, a0:a1], X[:, b0:b1], statistic=statistic,
                                       min_n=3, return_counts=True)
            ii, jj = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
            keep = jj - ii >= int(min_variant_lag)
            if upper_only: keep &= jj > ii
            else: keep &= jj != ii
            keep &= np.abs(jj - ii) <= lag
            dbp = None if pos is None else np.abs(pos[jj] - pos[ii])
            dcm = None if cm is None else np.abs(cm[jj] - cm[ii])
            if max_bp is not None: keep &= dbp <= float(max_bp)
            if max_cm is not None: keep &= dcm <= float(max_cm)
            keep &= np.isfinite(rr)
            if np.any(keep):
                yield R2Block(ii[keep], jj[keep], rr[keep], nn[keep],
                              None if dbp is None else dbp[keep],
                              None if dcm is None else dcm[keep])


def local_correlation_matrix(X):
    """PSD local marker correlation after mean imputation and scaling.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Returns
    -------
    object
        Computed result from ``(R, good)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X = np.asarray(X, dtype=np.float64)
    means = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, means)
    X -= X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    good = np.isfinite(sd) & (sd > 0)
    Z = np.zeros_like(X); Z[:, good] = X[:, good] / sd[good]
    R = Z.T @ Z / max(X.shape[0] - 1, 1)
    np.fill_diagonal(R, good.astype(float))
    return R, good
