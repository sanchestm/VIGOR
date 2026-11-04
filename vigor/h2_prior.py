"""Empirical-Bayes heritability prior inside the PSD/FA moment objective.

The covariance fit minimizes

    f_MoM(C) + sum_j kappa_j (h2_j(C) - mu_g(j))^2,   h2_j = sum_c G_c[jj] / (sum_c G_c[jj] + E[jj]),

where mu_g is the kappa-weighted group mean, profiled exactly at every
objective evaluation, so the prior mean is jointly estimated by L-BFGS.
kappa_j = strength * omega_j / tau2_g with omega_j = a_j se_j^2 / 2, where a_j
is the curvature of the trait's diagonal moment system along h2 (total
variance fixed) and se_j the Gaussian sandwich SE of the closure-augmented
per-trait moment h2. For decoupled traits this reproduces the normal-normal
EB weight tau2 / (tau2 + se_j^2) at strength 1. tau2_g is the
DerSimonian-Laird between-trait variance of the pilot estimates. Strength,
like every penalty weight, is not identified by training loss and is chosen
by held-out prediction (hyper_opt).

Intervals for shrunk h2 and marginal variances use the robust EBCI of
Armstrong, Kolesar & Plagborg-Moller (2022, Econometrica 90:2567): the
critical value bounds worst-case non-coverage given the estimated second
moment of the normalized shrinkage bias. Their guarantee is average coverage
across the traits of a group under approximately normal estimates, not
coverage of each trait; for this coupled FA estimator it is an approximation
checked by validation/coverage_h2_prior.py.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import numbers

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TAU2_FLOOR = 1e-4
MIN_GROUP_SIZE = 3


def resolve_groups(groups, columns):
    """Return integer group codes, labels and per-trait labels for h2_prior_groups.

    groups is None/'global' (one prior) or a label per trait: a Series/mapping
    keyed by trait label, or a positional sequence. Groups with fewer than
    three traits share one pooled group; a pooled group that is still too
    small joins the largest group. O(t).
    """
    columns = pd.Index(columns)
    T = len(columns)
    if T < MIN_GROUP_SIZE:
        raise ValueError(f'the h2 prior needs at least {MIN_GROUP_SIZE} traits')
    if groups is None or isinstance(groups, str) and groups.lower() in {'global', 'all', 'pooled'}:
        labels = np.repeat('global', T).astype(object)
    elif isinstance(groups, str):
        raise ValueError("h2_prior_groups must be None/'global' or one label per trait")
    else:
        if isinstance(groups, Mapping):
            groups = pd.Series(groups)
        if isinstance(groups, pd.Series):
            series = groups.copy()
            series.index = series.index.map(str)
            if series.index.has_duplicates:
                raise ValueError('h2_prior_groups labels must be unique')
            labels = series.reindex(columns.map(str)).to_numpy(dtype=object)
        else:
            labels = np.asarray(groups, dtype=object)
            if labels.shape != (T,):
                raise ValueError('positional h2_prior_groups must have one label per trait')
        if pd.isna(labels).any():
            raise ValueError('h2_prior_groups must label every trait')
        labels = np.asarray([str(x) for x in labels], dtype=object)
    counts = pd.Series(labels).value_counts()
    small = set(counts.index[counts < MIN_GROUP_SIZE])
    if small:
        pooled = np.isin(labels, list(small))
        labels = labels.copy()
        if pooled.sum() >= MIN_GROUP_SIZE or pooled.sum() == T:
            labels[pooled] = '__pooled_small_groups__'
        else:
            largest = counts[counts >= MIN_GROUP_SIZE].idxmax()
            labels[pooled] = largest
    codes, names = pd.factorize(pd.Series(labels), sort=False)
    return codes.astype(np.int64), [str(x) for x in names], labels


def pilot_uncertainty(kernels, frame, moments, ridge, closure_weight, *,
                      n_probes=64, probe_seed=42, probe_block_size=16, kernel_rank='auto',
                      method='analytic'):
    """Pilot h2, Gaussian sandwich SE and h2 curvature of each trait's diagonal system.

    method='analytic' (default) uses the Gaussian null approximation of the
    moment covariance, Cov(B_a, B_b) = 2 tr(A_a V A_b V) with V ~ Vp I on the
    trait's own mask, so Cov(B) ~ 2 Vp^2 A_jj. A_jj (the raw trace matrix
    tr(A_a A_b), closure removed) is already in the moments, so this costs
    O(t c^3), with no probes or kernel products. It is the multivariate form of the
    standard method-of-moments SE approximation (Visscher et al. 2014), most
    accurate at modest h2. The SEs only set the prior weights. method='probes'
    uses the Rademacher trace-probe geometry below for the full Gaussian
    covariance.

    kernels is the ordered kernel mapping matching the genetic moment
    components; moments are the (closure-augmented) fit moments; ridge has one
    value per component including E; closure_weight is the soft-closure weight
    actually added to A (zeros when off). The plug-in covariance is the
    nonnegative part of the unconstrained per-trait moment solution, which is
    available before fitting. kernel_rank ('auto' = 512, 'full', integer or per
    kernel) selects the shared grm2Us kernel approximation of
    vigor.kernel_eigen. Cost: one O(n² r) eigensystem per kernel (shared with
    the grm2us sandwich and hyper_opt folds) then O(c² P n_m r) per distinct
    mask, or O(c² P n²) per mask for the full kernel; plus O(t c⁴ P).
    """
    A = np.asarray(moments['A'], float)
    B = np.asarray(moments['B'], float)
    p = A.shape[0]
    T = A.shape[-1]
    if len(kernels) != p - 1:
        raise ValueError('kernel mapping does not match the moment components')
    if method not in {'analytic', 'probes'}:
        raise ValueError("h2_prior_pilot must be 'analytic' or 'probes'")
    ridge = np.asarray(ridge, float).reshape(-1)
    lam = np.zeros(T) if closure_weight is None else np.asarray(closure_weight, float).reshape(-1)
    hit = False
    if method == 'probes':
        from .calibration_probes import probe_geometry
        base, hit = probe_geometry(kernels, frame, n_probes=n_probes, probe_seed=probe_seed,
                                   probe_block_size=probe_block_size, kernel_rank=kernel_rank)
    h_raw = np.full(T, np.nan); se = np.full(T, np.nan); curvature = np.full(T, np.nan)
    mc = np.full(T, np.nan)
    for j in range(T):
        H = 0.5 * (A[:, :, j, j] + A[:, :, j, j].T) + np.diag(ridge)
        Hinv = np.linalg.pinv(H, rcond=1e-12)
        theta = Hinv @ B[:, j, j]
        total_raw = float(theta.sum())
        if np.isfinite(total_raw) and total_raw > 0:
            h_raw[j] = float(theta[:-1].sum()) / total_raw
        model = np.maximum(theta, 0.0)
        vg = float(model[:-1].sum()); ve = float(model[-1]); total = vg + ve
        if not np.isfinite(total) or total <= 0:
            continue
        closure_j = lam[j] if np.isfinite(lam[j]) and lam[j] > 0 else 0.0
        if method == 'analytic':
            # Raw traces tr(A_a A_b) on this trait's mask: the closure added lam to every entry.
            traces = 0.5 * (A[:, :, j, j] + A[:, :, j, j].T) - closure_j
            samples = (2.0 * total * total * traces)[:, :, None]
            dof = float(traces[-1, -1])  # tr(Q) = n_j - rank(fixed effects)
        else:
            item = base[j]
            samples = np.einsum('abklp,k,l->abp', item['tensor'], model, model, optimize=True)
            samples = 0.5 * (samples + samples.transpose(1, 0, 2))
            dof = float(item['n'] - item['rank'])
        # Closure adds lam * Vp to every component moment, Vp = y'Qy/(n-1).
        L = np.eye(p)
        if closure_j > 0:
            L[:, -1] += closure_j / max(dof, 1.0)
        M = Hinv @ L
        gradient = np.r_[np.repeat(ve, p - 1), -vg] / total**2
        direction = M.T @ gradient
        values = np.einsum('i,ijp,j->p', direction, samples, direction, optimize=True)
        variance = float(values.mean())
        if not np.isfinite(variance) or variance <= 0:
            continue
        se[j] = np.sqrt(variance)
        mc[j] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if values.size > 1 else 0.0
        share = model[:-1] / vg if vg > 0 else np.full(p - 1, 1.0 / (p - 1))
        u = total * np.r_[share, -1.0]
        curvature[j] = float(2.0 * u @ H @ u)
    return {'h2_pilot': h_raw, 'se_pilot': se, 'curvature': curvature,
            'h2_variance_mc_se': mc, 'n_probes': int(n_probes) if method == 'probes' else 0,
            'probe_seed': int(probe_seed), 'cache_hit': bool(hit), 'method': method}


def _dersimonian_laird(h, se):
    w = 1.0 / se**2
    mu = float(np.sum(w * h) / np.sum(w))
    q = float(np.sum(w * (h - mu) ** 2))
    c = float(np.sum(w) - np.sum(w * w) / np.sum(w))
    tau2 = max(0.0, (q - (len(h) - 1)) / c) if c > 0 else 0.0
    return mu, tau2


def build_penalty(strength, groups, columns, pilot):
    """Assemble the fit penalty and its diagnostics from pilot uncertainty; O(t)."""
    if isinstance(strength, bool) or not isinstance(strength, numbers.Real) or not np.isfinite(strength) or strength < 0:
        raise ValueError('h2_prior_strength must be a finite nonnegative number')
    codes, names, labels = resolve_groups(groups, columns)
    h = np.asarray(pilot['h2_pilot'], float); se = np.asarray(pilot['se_pilot'], float)
    a = np.asarray(pilot['curvature'], float)
    valid = np.isfinite(h) & np.isfinite(se) & (se > 0) & np.isfinite(a) & (a > 0)
    if valid.sum() < 2:
        raise ValueError('the h2 prior needs pilot uncertainty for at least two traits')
    global_mu, global_tau2 = _dersimonian_laird(h[valid], se[valid])
    tau2_group = np.empty(len(names)); mu_group = np.empty(len(names))
    for g in range(len(names)):
        sel = valid & (codes == g)
        if sel.sum() >= 2:
            mu_group[g], tau2_group[g] = _dersimonian_laird(h[sel], se[sel])
        else:
            mu_group[g], tau2_group[g] = global_mu, global_tau2
    tau2 = tau2_group[codes]
    precision = float(strength) / np.maximum(tau2, TAU2_FLOOR)
    omega = np.where(valid, 0.5 * a * se**2, 0.0)
    kappa = precision * omega
    weight = np.where(valid, 1.0 / (1.0 + precision * np.where(valid, se, 0.0) ** 2), 1.0)
    return {'strength': float(strength), 'kappa': kappa, 'codes': codes, 'group_names': names,
            'group_labels': labels, 'tau2': tau2, 'tau2_group': tau2_group, 'mu_pilot_group': mu_group,
            'tau2_floor': TAU2_FLOOR, 'omega': omega, 'weight': weight, 'valid': valid,
            'h2_pilot': h, 'se_pilot': se, 'curvature': a,
            'h2_variance_mc_se': pilot.get('h2_variance_mc_se'),
            'n_probes': pilot.get('n_probes'), 'probe_seed': pilot.get('probe_seed'),
            'pilot_method': pilot.get('method', 'probes'),
            'cache_hit': pilot.get('cache_hit')}


def _group_mean(values, kappa, codes, n_groups):
    num = np.bincount(codes, weights=kappa * values, minlength=n_groups)
    den = np.bincount(codes, weights=kappa, minlength=n_groups)
    return np.divide(num, den, out=np.zeros(n_groups), where=den > 0)


def penalty_terms(diagonals, penalty):
    """Penalty value and gradients with respect to each component diagonal.

    diagonals is a list of per-component diagonal vectors (genetic..., E).
    mu is profiled at its kappa-weighted optimum, so by the envelope theorem
    it contributes no extra gradient term. O(c t).
    """
    vg = np.sum(diagonals[:-1], axis=0); ve = np.asarray(diagonals[-1], float)
    total = vg + ve
    good = total > 0
    h = np.divide(vg, total, out=np.zeros_like(total), where=good)
    kappa = np.where(good, penalty['kappa'], 0.0); codes = penalty['codes']
    mu = _group_mean(h, kappa, codes, len(penalty['group_names']))
    residual = h - mu[codes]
    value = float(np.sum(kappa * residual**2))
    dh = 2.0 * kappa * residual
    total2 = np.where(good, total * total, 1.0)
    dvg = np.where(good, dh * ve / total2, 0.0)
    dve = np.where(good, -dh * vg / total2, 0.0)
    return value, [dvg] * (len(diagonals) - 1) + [dve], h, mu


def penalty_hessian_apply(delta, vg, ve, penalty):
    """Gauss-Newton penalty Hessian (profiled mu) applied to diagonal perturbations.

    delta has shape (c+1, ..., t) for components genetic..., E. Returns the
    same shape. This is D' K D with D=dh/d(diagonals) and K the kappa matrix
    projected orthogonally to each group mean. O(c t) per column.
    """
    delta = np.asarray(delta, float)
    total = vg + ve
    good = total > 0
    total2 = np.where(good, total * total, 1.0)
    dg = np.where(good, ve / total2, 0.0); de = np.where(good, -vg / total2, 0.0)
    kappa = np.where(good, penalty['kappa'], 0.0); codes = penalty['codes']
    n_groups = len(penalty['group_names'])
    dh = np.sum(delta[:-1], axis=0) * dg + delta[-1] * de
    flat = dh.reshape(-1, dh.shape[-1])
    projected = np.empty_like(flat)
    for row in range(flat.shape[0]):
        mean = _group_mean(flat[row], kappa, codes, n_groups)
        projected[row] = kappa * (flat[row] - mean[codes])
    kd = projected.reshape(dh.shape)
    out = np.empty_like(delta)
    out[:-1] = kd * dg
    out[-1] = kd * de
    return out


def penalty_curvature_apply(delta, vg, ve, penalty):
    """Residual curvature sum_j kappa_j (h_j - mu_g) d²h_j of the half penalty.

    The Gauss-Newton term (:func:`penalty_hessian_apply`) plus this term is the
    exact Hessian of the half penalty with mu profiled (envelope theorem).
    delta has shape (c+1, ..., t); returns the same shape. O(c t) per column.
    """
    delta = np.asarray(delta, float)
    total = vg + ve
    good = total > 0
    kappa = np.where(good, penalty['kappa'], 0.0); codes = penalty['codes']
    h = np.divide(vg, total, out=np.zeros_like(total), where=good)
    mu = _group_mean(h, kappa, codes, len(penalty['group_names']))
    weight = kappa * (h - mu[codes])
    cube = np.where(good, total**3, 1.0)
    hgg = np.where(good, -2.0 * ve / cube, 0.0)
    hee = np.where(good, 2.0 * vg / cube, 0.0)
    hge = np.where(good, (vg - ve) / cube, 0.0)
    dvg = np.sum(delta[:-1], axis=0); dve = delta[-1]
    out = np.empty_like(delta)
    out[:-1] = weight * (hgg * dvg + hge * dve)
    out[-1] = weight * (hge * dvg + hee * dve)
    return out


def _dl_gradient(h, se):
    """d tau2_DL / d h of the DerSimonian-Laird estimate; zero when truncated at 0."""
    w = 1.0 / se**2
    mu = float(np.sum(w * h) / np.sum(w))
    c = float(np.sum(w) - np.sum(w * w) / np.sum(w))
    if c <= 0:
        return np.zeros_like(h)
    raw = (float(np.sum(w * (h - mu) ** 2)) - (len(h) - 1)) / c
    return 2.0 * w * (h - mu) / c if raw > 0 else np.zeros_like(h)


def prior_sensitivity(penalty, moments, ridge, diagonals):
    """Response of the half-penalty gradient to moment perturbations.

    The pilot h2 (unconstrained per-trait moment solution), each group's
    DerSimonian-Laird tau2 and hence kappa_j and the profiled mean are
    functions of the moments, so they move with every sandwich draw.
    diagonals are the fitted component diagonals (genetic..., E). Returns
    apply(dd, dlam): dd (b, c+1, t) are diagonal moment perturbations (closure
    moment included), dlam (b, t) the closure-weight perturbation; the result
    (b, c+1, t) is the change of each component's diagonal half gradient.
    Pilot SEs and curvatures are held at their data values (they enter
    through second-order terms). Setup O(t c³), O(b t c²) per call.
    """
    A = np.asarray(moments['A'], float); Bdata = np.asarray(moments['B'], float)
    p = A.shape[0]; T = A.shape[-1]; idx = np.arange(T)
    ridge = np.asarray(ridge, float).reshape(-1)
    H = np.transpose(A[:, :, idx, idx], (2, 0, 1))
    H = 0.5 * (H + np.swapaxes(H, 1, 2)) + np.diag(ridge)[None]
    Hinv = np.linalg.pinv(H, rcond=1e-12)
    theta = np.einsum('jab,bj->ja', Hinv, Bdata[:, idx, idx])
    total_pilot = theta.sum(axis=1)
    target = np.asarray(moments.get('variance_closure_target', np.zeros(T)), float)
    shift = np.where(np.isfinite(target), total_pilot - target, 0.0)
    h_pilot = np.asarray(penalty['h2_pilot'], float); se = np.asarray(penalty['se_pilot'], float)
    valid = np.asarray(penalty['valid'], bool) & (total_pilot > 0)
    coef = np.zeros((T, p))
    coef[valid, :-1] = ((1.0 - h_pilot[valid]) / total_pilot[valid])[:, None]
    coef[valid, -1] = -h_pilot[valid] / total_pilot[valid]
    dh_map = np.einsum('ja,jab->jb', coef, Hinv)
    codes = np.asarray(penalty['codes']); n_groups = len(penalty['group_names'])
    tau2_group = np.asarray(penalty['tau2_group'], float)
    base = np.asarray(penalty['valid'], bool)
    global_grad = np.zeros(T)
    global_grad[base] = _dl_gradient(h_pilot[base], se[base])
    dl = np.zeros((n_groups, T))
    for g in range(n_groups):
        sel = base & (codes == g)
        if tau2_group[g] <= TAU2_FLOOR:
            continue
        if sel.sum() >= 2:
            dl[g, sel] = _dl_gradient(h_pilot[sel], se[sel])
        else:
            dl[g] = global_grad
    vg = np.sum(diagonals[:-1], axis=0); ve = np.asarray(diagonals[-1], float)
    total = vg + ve; good = total > 0
    kappa = np.where(good, penalty['kappa'], 0.0)
    h = np.divide(vg, total, out=np.zeros_like(total), where=good)
    mu = _group_mean(h, kappa, codes, n_groups)
    residual = h - mu[codes]
    kappa_sum = np.bincount(codes, weights=kappa, minlength=n_groups)
    onehot = np.zeros((T, n_groups)); onehot[idx, codes] = 1.0
    tau2 = np.maximum(np.asarray(penalty['tau2'], float), TAU2_FLOOR)
    total2 = np.where(good, total * total, 1.0)
    dg = np.where(good, ve / total2, 0.0); de = np.where(good, -vg / total2, 0.0)

    def apply(dd, dlam):
        # The closure weight also sits in each pilot system's A: d(A theta) = dlam 1 (1'theta).
        dd = np.asarray(dd, float) - (dlam * shift)[:, None, :]
        dh = np.einsum('jb,nbj->nj', dh_map, dd)
        dkappa = -kappa * (dh @ dl.T)[:, codes] / tau2
        dmu = np.divide((dkappa * residual) @ onehot, kappa_sum,
                        out=np.zeros((dd.shape[0], n_groups)), where=kappa_sum > 0)
        c = dkappa * residual - kappa * dmu[:, codes]
        out = np.empty((dd.shape[0], p, T))
        out[:, :-1] = (c * dg)[:, None, :]
        out[:, -1] = c * de
        return out
    return apply


def penalty_from_fit(fit):
    """Rebuild the fitted penalty (and its pilot state) from a stored result; O(t)."""
    prior = fit.get('h2_prior', {})
    if not (isinstance(prior, Mapping) and prior.get('active')):
        return None
    names = [str(x) for x in prior['groups']]
    lookup = {name: g for g, name in enumerate(names)}
    codes = np.asarray([lookup[str(x)] for x in prior['trait_group']], np.int64)
    tau2 = np.asarray(fit['h2_prior_tau2'], float)
    valid = np.asarray(prior['pilot_valid'], bool)
    return {'strength': float(prior['strength']), 'kappa': np.asarray(fit['h2_prior_kappa'], float),
            'codes': codes, 'group_names': names,
            'group_labels': list(prior['trait_group']), 'tau2': tau2,
            'tau2_group': np.asarray(prior['group_tau2'], float), 'tau2_floor': TAU2_FLOOR,
            'weight': np.asarray(fit['h2_prior_weight'], float), 'valid': valid,
            'h2_pilot': np.asarray(fit['h2_pilot'], float), 'se_pilot': np.asarray(fit['h2_pilot_se'], float)}


def _r0(t, chi):
    s = np.sqrt(t)
    return norm.cdf(-chi - s) + norm.sf(chi - s)


def _r0_dt(t, chi):
    s = np.sqrt(t)
    return (norm.pdf(chi - s) - norm.pdf(chi + s)) / (2.0 * s)


def _rho(m2, chi):
    """Worst-case non-coverage of h ± chi*se with E[bias²/se²] = m2 (AKP 2022)."""
    r_zero = float(_r0(0.0, chi))
    s = np.linspace(1e-4, chi + 10.0, 600)
    t = s * s
    g = _r0(t, chi) - r_zero - t * _r0_dt(t, chi)
    change = np.flatnonzero((g[:-1] < 0) & (g[1:] >= 0))
    if change.size == 0:
        return float(_r0(m2, chi))
    lo, hi = t[change[0]], t[change[0] + 1]
    t0 = brentq(lambda x: _r0(x, chi) - r_zero - x * _r0_dt(x, chi), lo, hi, xtol=1e-12)
    if m2 >= t0:
        return float(_r0(m2, chi))
    return float(r_zero + m2 / t0 * (_r0(t0, chi) - r_zero))


def _critical_value(m2, alpha):
    z = float(norm.isf(alpha / 2.0))
    if not np.isfinite(m2):
        return np.nan
    if m2 <= 1e-12:
        return z
    hi = np.sqrt(m2) + z + 2.0
    while _rho(m2, hi) > alpha:
        hi *= 1.5
    return float(brentq(lambda chi: _rho(m2, chi) - alpha, z, hi, xtol=1e-8))


@lru_cache(maxsize=8)
def _critical_table(alpha):
    grid = np.logspace(-6, 4, 161)
    return np.log(grid), np.array([_critical_value(m, alpha) for m in grid])


def robust_critical_values(m2, ci_level=0.95):
    """Robust EBCI critical values for normalized bias second moments m2; O(t)."""
    alpha = 1.0 - float(ci_level)
    if not 0 < alpha < 1:
        raise ValueError('ci_level must lie in (0, 1)')
    m2 = np.asarray(m2, float)
    out = np.full(m2.shape, np.nan)
    z = float(norm.isf(alpha / 2.0))
    logs, table = _critical_table(round(alpha, 12))
    finite = np.isfinite(m2)
    tiny = finite & (m2 <= 1e-6)
    out[tiny] = z
    mid = finite & (m2 > 1e-6) & (m2 <= 1e4)
    out[mid] = np.interp(np.log(m2[mid]), logs, table)
    for i in np.flatnonzero(finite & (m2 > 1e4)):
        out.flat[i] = _critical_value(float(m2.flat[i]), alpha)
    return out


def robust_intervals(h2, h2_se, vg, vg_se, ve, ve_se, penalty, ci_level=0.95):
    """Robust EBCI for shrunk h2 and marginal variances; O(t).

    The shrinkage bias of trait j has second moment (1-w_j)² tau2_j on the h2
    scale, and T_j² times that for vG and vE (closure fixes the total
    T_j=vG+vE to first order). Intervals are clipped to the parameter space.
    """
    w = np.asarray(penalty['weight'], float); tau2 = np.asarray(penalty['tau2'], float)
    bias_sd = (1.0 - w) * np.sqrt(np.maximum(tau2, 0.0))
    total = np.asarray(vg, float) + np.asarray(ve, float)

    def interval(value, se, sd, upper):
        se = np.asarray(se, float); value = np.asarray(value, float)
        m2 = np.divide(sd**2, se**2, out=np.full(se.shape, np.nan), where=se > 0)
        m2[(se > 0) & (sd == 0)] = 0.0
        cva = robust_critical_values(m2, ci_level)
        lo = np.maximum(value - cva * se, 0.0)
        hi = value + cva * se
        if upper is not None:
            hi = np.minimum(hi, upper); lo = np.minimum(lo, upper)
        return np.column_stack([lo, hi]), cva, m2

    h_ci, h_cva, h_m2 = interval(h2, h2_se, bias_sd, 1.0)
    g_ci, _, _ = interval(vg, vg_se, bias_sd * total, None)
    e_ci, _, _ = interval(ve, ve_se, bias_sd * total, None)
    return {'h2_ci95': h_ci, 'G_variance_ci95': g_ci, 'E_variance_ci95': e_ci,
            'h2_ci_critical_value': h_cva, 'h2_ci_bias_m2': h_m2}
