"""Calibrated fast score information for the observed-space score core.

The score ``u = D' P y`` is computed exactly from the prepared null's dual
``P y`` (one GEMM per block). Only the information ``D' P D``, which needs a
covariance solve per design column when phenotypes are missing, is replaced by
a calibrated ratio approximation (the multivariate GRAMMAR-gamma / BOLT-LMM
idea):

    D'PD[(c,j),(d,k)] ~= sqrt(g_cj s_cj) sqrt(g_dk s_dk) * rho_cd * R_jk

with ``s_cj`` the centered genotype sum of squares over animals observed for
trait j, ``g_cj`` a per-contrast/trait ratio, ``rho_cd`` the per-variant
correlation between design contrasts (additive vs dominance) and ``R`` a trait
correlation of the information. ``g`` and ``R`` are fitted once per null
against the exact ``D'PD`` of a few calibration variants (default 32, spread
along the scanned group); every other variant costs O(n c t + c t^2).

Outputs keep the exact engine's schema (beta, z, p, joint/conditional,
omnibus), so the fast and exact results are drop-in comparable.
"""
from __future__ import annotations

import numpy as np

from .exact import (_center_designs, _finalize_scores, _generic_scores,
                    _spectral_scores, _spectral_state, _sym, chi2_logsf)

__all__ = ["FastInformation", "calibrate_fast_information", "calibrate_group",
           "fast_design_scores", "calibration_indices"]


class FastInformation:
    """Per-null calibration of the fast information approximation.

    Attributes
    ----------
    gamma : (c, t) ndarray
        Ratio of exact information diagonal to masked genotype sum of squares.
    trait_correlation : (t, t) ndarray
        Pooled correlation of the exact information across traits.
    contrasts : int
        Number of design contrasts the calibration was fitted for.
    n_variants : int
        Calibration variants used.
    diagnostics : dict
        In-sample agreement with the exact statistics on the calibration
        variants (optimistic; the out-of-sample error is somewhat larger).
    """

    def __init__(self, gamma, trait_correlation, contrasts, n_variants, observed, diagnostics=None):
        self.gamma = np.asarray(gamma, float)
        self.trait_correlation = np.asarray(trait_correlation, float)
        self.contrasts = int(contrasts)
        self.n_variants = int(n_variants)
        self.observed = np.asarray(observed, float)
        self.diagnostics = dict(diagnostics or {})
        R = _sym(self.trait_correlation)
        w, V = np.linalg.eigh(R)
        largest = max(float(np.max(np.abs(w))), np.finfo(float).tiny)
        tol = max(len(w), 1) * np.finfo(float).eps * largest * 100
        good = w > tol
        self.trait_rank = int(good.sum())
        self.trait_inverse = (V * np.where(good, 1.0 / np.where(good, w, 1.0), 0.0)) @ V.T
        self.trait_projector = (V ** 2) @ good.astype(float)

    def metadata(self):
        """JSON-safe summary for result attributes."""
        out = {"calibration_variants": self.n_variants, "trait_rank": self.trait_rank,
               "gamma_median": float(np.nanmedian(self.gamma)),
               "gamma_min": float(np.nanmin(self.gamma)), "gamma_max": float(np.nanmax(self.gamma))}
        out.update({k: float(v) for k, v in self.diagnostics.items()})
        # NaN breaks JSON round-trips and the per-group metadata equality check.
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()}


def _observed_matrix(null):
    """Animal x trait 0/1 matrix of observed phenotype cells."""
    n, t = null.y.shape
    o = np.zeros(n * t)
    o[null.observed] = 1.0
    return o.reshape(n, t)


def _as_designs(designs, n):
    x = np.asarray(designs, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3 or x.shape[0] != n or np.isinf(x).any():
        raise ValueError("designs must be sample-by-variant(-by-contrast), finite or NaN")
    return x


def calibration_indices(n_variants, n_calibration=32):
    """Evenly spaced positions (unique, sorted) used to pick calibration variants."""
    n_variants = int(n_variants)
    k = max(0, min(int(n_calibration), n_variants))
    if k == 0:
        return np.zeros(0, np.int64)
    return np.unique(np.linspace(0, n_variants - 1, k).round().astype(np.int64))


def calibrate_fast_information(null, designs, *, min_maf=0.05, attach=True):
    """Fit the fast information approximation against exact D'PD.

    Parameters
    ----------
    null : ExactScoreNull
        Prepared null (use the LOCO kernel of the group being scanned).
    designs : (n, m) or (n, m, c) array
        Calibration genotypes/designs in ``null.sample_ids`` order, ideally
        spread along the scanned group. 16 to 64 variants suffice; each costs
        one exact information evaluation.
    min_maf : float
        Additive-dosage MAF floor for calibration variants (contrast 0 is
        read as a 0/1/2 dosage). Falls back to all informative variants.
    attach : bool
        Cache the calibration on ``null`` so later fast scoring reuses it.

    Returns
    -------
    FastInformation
    """
    n, t = null.y.shape
    x = _as_designs(designs, n)
    xc, finite = _center_designs(x, null.work_dtype)
    O = _observed_matrix(null)
    _, m, c = xc.shape
    s = np.einsum("nmc,nt->mct", xc * xc, O, optimize=True)
    good = (s.sum(axis=2) > 0).all(axis=1)
    dosage = x[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.nanmean(dosage, axis=0) / 2.0
    maf = np.minimum(freq, 1.0 - freq)
    keep = good & (np.nan_to_num(maf) >= float(min_maf))
    if keep.sum() < min(8, int(good.sum())):
        keep = good
    if not keep.any():
        # Only monomorphic variants: every statistic is uninformative anyway.
        fast = FastInformation(np.ones((c, t)), np.eye(t), c, 0, O,
                               {"calibration_skipped_no_polymorphic_variants": 1.0})
        if attach:
            null._fast_information = fast
        return fast
    xc, finite, s = xc[:, keep], finite[:, keep], s[keep]
    mk = int(keep.sum())
    b = np.eye(t)
    spectral = _spectral_state(null)
    if spectral is not None:
        u, info, nobs = _spectral_scores(spectral, xc, finite, b)
    else:
        u, info, nobs = _generic_scores(null, xc, finite, b)
    info = 0.5 * (info + info.transpose(0, 2, 1))
    diag = np.einsum("mii->mi", info).reshape(mk, c, t)
    ssum = s.sum(axis=0)
    gamma = np.divide(diag.sum(axis=0), ssum, out=np.full((c, t), np.nan), where=ssum > 0)
    gamma = np.where(np.isfinite(gamma) & (gamma > 0), gamma, np.nan)
    if np.isnan(gamma).any():
        fill = np.nanmedian(gamma) if np.isfinite(gamma).any() else 1.0
        gamma = np.where(np.isnan(gamma), fill, gamma)
    blocks = np.stack([info[:, k * t:(k + 1) * t, k * t:(k + 1) * t] for k in range(c)], axis=1)
    dv = np.sqrt(np.maximum(np.einsum("mcii->mci", blocks), 0.0))
    ok = dv > np.maximum(dv.max(axis=2, keepdims=True) * 1e-6, np.finfo(float).tiny)
    pair = ok[..., :, None] & ok[..., None, :]
    denom = np.where(pair, dv[..., :, None] * dv[..., None, :], 1.0)
    corr = np.where(pair, blocks / denom, 0.0)
    count = pair.sum(axis=(0, 1))
    R = np.divide(corr.sum(axis=(0, 1)), count, out=np.zeros((t, t)), where=count > 0)
    R = _sym(R)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 0.0, None)) @ V.T
    d = np.sqrt(np.maximum(np.diag(R), np.finfo(float).tiny))
    R = R / d[:, None] / d[None, :]
    fast = FastInformation(gamma, R, c, mk, O)
    # In-sample agreement with the exact statistics on the calibration variants.
    exact_res, exact_omni = _finalize_scores(u, info, nobs)
    fast_res, fast_omni = _fast_finalize(fast, u.reshape(mk, c, t), s, xc, nobs.reshape(mk, c, t))
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = fast_res["stat"] / exact_res["stat"]
        cr = np.abs(np.log(cr[np.isfinite(cr) & (cr > 0) & (exact_res["stat"] > 1e-8)]))
        orat = fast_omni["stat"] / exact_omni["stat"]
        orat = orat[np.isfinite(orat) & (orat > 0)]
    fast.diagnostics = {
        "coordinate_stat_median_abs_log_ratio": float(np.median(cr)) if cr.size else np.nan,
        "coordinate_stat_p90_abs_log_ratio": float(np.quantile(cr, .9)) if cr.size else np.nan,
        "omnibus_stat_median_ratio": float(np.median(orat)) if orat.size else np.nan,
    }
    if attach:
        null._fast_information = fast
    return fast


def _fast_finalize(fast, u, s, xc, nobs):
    """Schema-identical statistics from exact scores and approximate information.

    u, s, nobs: (m, c, t). xc: centered designs (n, m, c) for contrast correlation.
    """
    m, c, t = u.shape
    d2 = fast.gamma[None] * s
    largest = np.maximum(d2.reshape(m, -1).max(axis=1), np.finfo(float).tiny)
    informative = d2 > (largest * 1e-12)[:, None, None]
    safe = np.where(informative, d2, 1.0)
    root = np.sqrt(safe)
    res = {"nobs": nobs}
    res["beta"] = np.where(informative, u / safe, np.nan)
    res["beta_se"] = np.where(informative, 1.0 / root, np.nan)
    z = np.where(informative, u / root, np.nan)
    res["z"] = z
    res["stat"] = z ** 2
    res["log_p"] = chi2_logsf(res["stat"], 1)
    res["p"] = np.exp(res["log_p"])
    res["neglog_p"] = -res["log_p"] / np.log(10)
    # Contrast correlation per variant (c x c), zero rows for constant contrasts.
    if c == 1:
        has = (np.einsum("nm,nm->m", xc[:, :, 0], xc[:, :, 0]) > 0).astype(float)
        Pinv = has[:, None, None]
        projP = has[:, None]
        rankP = has
    else:
        gram = np.einsum("nmc,nmd->mcd", xc, xc, optimize=True)
        dg = np.sqrt(np.maximum(np.einsum("mcc->mc", gram), 0.0))
        live = dg > 0
        P = np.where(live[:, :, None] & live[:, None, :],
                     gram / np.where(live[:, :, None] & live[:, None, :], dg[:, :, None] * dg[:, None, :], 1.0), 0.0)
        pw, pv = np.linalg.eigh(P)
        ptol = c * np.finfo(float).eps * np.maximum(np.abs(pw).max(axis=1), np.finfo(float).tiny) * 100
        pgood = pw > np.maximum(ptol[:, None], 1e-10)
        Pinv = (pv * np.where(pgood, 1.0 / np.where(pgood, pw, 1.0), 0.0)[:, None, :]) @ pv.transpose(0, 2, 1)
        projP = np.einsum("mij,mj->mi", pv ** 2, pgood.astype(float))
        rankP = pgood.sum(axis=1).astype(float)
    zz = np.where(informative, z, 0.0)
    W = zz @ fast.trait_inverse
    y = np.einsum("mcd,mdt->mct", Pinv, W, optimize=True)
    stat = np.maximum(np.einsum("mct,mct->m", zz, y, optimize=True), 0.0)
    rank = rankP * fast.trait_rank
    stat = np.where(rank > 0, stat, np.nan)
    joint = y / root
    joint_se = np.sqrt(np.maximum(np.einsum("mcc->mc", Pinv)[:, :, None]
                                  * np.diag(fast.trait_inverse)[None, None, :], 0.0)) / root
    estimable = informative & (projP[:, :, None] * fast.trait_projector[None, None, :] > 1 - 1e-7) & (joint_se > 0)
    res["joint_beta"] = np.where(estimable, joint, np.nan)
    res["joint_beta_se"] = np.where(estimable, joint_se, np.nan)
    res["conditional_stat"] = np.where(estimable, (joint / np.where(estimable, joint_se, 1.0)) ** 2, np.nan)
    res["conditional_log_p"] = chi2_logsf(res["conditional_stat"], 1)
    res["conditional_neglog_p"] = -res["conditional_log_p"] / np.log(10)
    log_p = chi2_logsf(stat, rank)
    omnibus = {"stat": stat, "dof": rank, "log_p": log_p, "neglog_p": -log_p / np.log(10), "p": np.exp(log_p)}
    order = ("beta", "beta_se", "z", "stat", "p", "log_p", "neglog_p", "nobs", "joint_beta",
             "joint_beta_se", "conditional_stat", "conditional_log_p", "conditional_neglog_p")
    return {k: res[k].reshape(m, c * t) for k in order}, omnibus


def _dosage_designs(dosage, model):
    if model == "additive":
        return dosage[:, :, None]
    observed = np.isfinite(dosage)
    if np.any(observed & (dosage != np.round(dosage))):
        raise ValueError("add-dom heterozygote tests require hardcalls, not fractional dosages")
    dom = np.where(observed, dosage == 1, np.nan).astype(np.float32)
    return np.stack([dosage, dom], axis=2)


def calibrate_group(null, source, vm, samples, group, *, n_calibration=32, block_size="auto",
                    model="additive", min_maf=0.05):
    """Calibrate ``null`` on variants spread evenly along one scan group.

    Decodes the group's genotype blocks once (no scoring) and keeps a strided
    pool of 4 x n_calibration variants, then n_calibration with MAF >= min_maf
    evenly spaced within the pool. Stops reading after the last pooled variant.
    The calibration is cached on ``null`` and returned.
    """
    from .scan import genotype_blocks
    keep = vm.chromosome.astype(str).to_numpy() == str(group)
    total = int(keep.sum())
    if total == 0:
        raise ValueError(f"no variants in scan group {group!r}")
    pool = calibration_indices(total, 4 * int(n_calibration))
    last = int(pool[-1])
    pieces, seen = [], 0
    blocks = genotype_blocks(source, vm, samples, group, block_size=block_size)
    try:
        for X, _ in blocks:
            w = X.shape[1]
            local = pool[(pool >= seen) & (pool < seen + w)] - seen
            if len(local):
                pieces.append(np.asarray(X.to_numpy(np.float32)[:, local]))
            seen += w
            if seen > last:
                break
    finally:
        blocks.close()
    dosage = np.concatenate(pieces, axis=1).astype(np.float64)
    with np.errstate(invalid="ignore"):
        freq = np.nanmean(dosage, axis=0) / 2.0
    maf = np.nan_to_num(np.minimum(freq, 1.0 - freq))
    ok = np.flatnonzero(maf >= float(min_maf))
    if len(ok) < min(8, int(np.count_nonzero(maf > 0))):
        ok = np.flatnonzero(maf > 0)
    if not len(ok):
        ok = np.arange(dosage.shape[1])
    chosen = ok[calibration_indices(len(ok), n_calibration)]
    return calibrate_fast_information(null, _dosage_designs(dosage[:, chosen], model), min_maf=min_maf)


def fast_design_scores(null, xc, finite, fast=None):
    """Fast statistics for centered designs (n, m, c); returns (result, omnibus, fast).

    Without a supplied/cached calibration, one is fitted on up to 32 evenly
    spaced variants of this block and cached on ``null``.
    """
    n, m, c = xc.shape
    t = null.y.shape[1]
    if fast is None:
        fast = getattr(null, "_fast_information", None)
    if fast is None or fast.contrasts != c:
        idx = calibration_indices(m, 32)
        # xc is already centered; centering again is a no-op on observed calls.
        designs = np.where(finite[:, idx], xc[:, idx], np.nan)
        fast = calibrate_fast_information(null, designs)
        fast.diagnostics["calibrated_on_first_block"] = 1.0
    dt = np.dtype(null.work_dtype)
    X2 = xc.reshape(n, m * c)
    u = np.asarray(X2.T.astype(dt) @ null.dual.astype(dt), np.float64).reshape(m, c, t)
    O = fast.observed
    s = np.asarray((X2 * X2).T.astype(dt) @ O.astype(dt), np.float64).reshape(m, c, t)
    nobs = (finite.reshape(n, m * c).T.astype(dt) @ O.astype(dt)).astype(np.float64).reshape(m, c, t)
    result, omnibus = _fast_finalize(fast, u, s, xc, nobs)
    return result, omnibus, fast
