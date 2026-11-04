"""Final PSD/FA moment sandwich on heterogeneous masks with the fit's own metric.

The fitted covariance minimizes the half objective

    sum_pairs C' A~ C / 2 - B'C (+ half h2 prior + soft variance closure)

over the FA/sparse-FA parameterization, where A~ already contains ridge and
the closure moment. Its first-order response to the data is the Newton
(implicit-function) linearization

    dphi = (J'(H + P)J + R2)^-1 J' dB_eff,   dC = J dphi,

with H the pairwise moment Hessian (coupling all components), P the exact
Hessian of the half prior (Gauss-Newton part plus residual curvature, group
mean profiled), R2 the residual curvature of the loading parameterization
(2 R L on the support, R the fitted half gradient) and J the joint tangent
dF F' + F dF' on the selected support plus active specific variances. dB_eff
is the moment perturbation plus the movement of every learned data-dependent
weight: the closure precision (n-1)/(2 Vp²) and, through the pilot h2 and
DerSimonian-Laird tau2, the prior weights kappa and their profiled mean.
Pilot SEs/curvatures are held at their data values.

Cov(dB) comes from one of three kernel backends:
  'paired'   factorization-free bilinear probes. For independent masked
             Rademacher U, W, E[(u'A V w)(u'V B w)] = tr(A V B V), so the pair
             xi = U'K V(W), eta = V(U)'K W gives an unbiased covariance from
             kernel products only (V(X) = sum_c K_c X C_c + X E), retaining
             the full supplied kernel. Estimates can be negative with few
             probes; those are reported unavailable, never clipped.
  'grm2us'   Gaussian draws from vigor.npplink.grm2Us randomized eigensystems
             plus a trace-matched isotropic remainder (approximate kernel).
  'cholesky' Gaussian draws from one exact Cholesky factor per kernel; the
             O(n³/3) small-data reference.
Probes are added in blocks until every monitored SE (marginal variances and
h2) reaches the requested relative Monte Carlo error or the cap. All draws of
a block are projected together by batched PCG stopped on the energy-norm
error of the projected draws; non-convergence raises. Uncertainty is
conditional on rank, sparse support and h2-prior strength/grouping.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np
from scipy.linalg import cholesky, eigh

from .progress import operation_progress

KERNEL_FACTORS = ('paired', 'grm2us', 'cholesky')
PAIRED_CHUNK = 4      # probes (or probe pairs) sharing one wide kernel GEMM
MC_QUANTILE = 0.90    # stopping quantile of the per-SE Monte Carlo errors


def _dense_kernel(op):
    from .multivariate import DenseKernelOperator
    return op.K if isinstance(op, DenseKernelOperator) else op.submatrix(np.arange(op.shape[0]))


class _CholeskyKernelSampler:
    """Exact N(0, K) sampler from one dense factorization L L' = K (+ recorded jitter)."""

    def __init__(self, op):
        dense = np.asarray(_dense_kernel(op), dtype=np.float64)
        n = dense.shape[0]
        scale = max(float(np.mean(np.diag(dense))), 1e-12)
        self.n = n
        self.factor = None
        for relative in (0.0, 1e-10, 1e-8, 1e-6):
            work = np.array(dense, dtype=np.float64, copy=True)
            if relative:
                work.flat[::n + 1] += relative * scale
            try:
                self.factor = cholesky(work, lower=True, overwrite_a=True, check_finite=False)
                self.approximation = 'exact_cholesky' if relative == 0.0 else f'cholesky_relative_jitter_{relative:g}'
                self.jitter_relative = relative
                break
            except np.linalg.LinAlgError:
                del work
        if self.factor is None:
            # Not PSD: draw from the PSD eigen-projection of the kernel.
            values, vectors = eigh(np.array(dense, copy=True), overwrite_a=True, check_finite=False)
            negative = float(-values.min()) if values.min() < 0 else 0.0
            vectors *= np.sqrt(np.maximum(values, 0.0))[None, :]
            self.factor = vectors
            self.approximation = 'psd_eigen_projection'
            self.jitter_relative = negative / scale
        del dense
        # Draws only need single precision; the factorization above is float64.
        # Fortran order lets BLAS trmm use the factor without a copy.
        self.factor = np.asfortranarray(self.factor, dtype=np.float32)
        # With K = L L' exactly (no jitter), the moment products K Y0 can use L as well.
        self.triangular = self.approximation == 'exact_cholesky'

    def sample(self, rng, ncols):
        m = int(ncols)
        if m <= 0:
            return np.zeros((self.n, 0), float)
        Z = rng.standard_normal(size=(self.factor.shape[1], m), dtype=np.float32)
        if self.triangular:
            return np.asarray(_trmm(self.factor, Z, transpose=False), dtype=np.float64)
        return np.asarray(self.factor @ Z, dtype=np.float64)

    def half_kernel(self, Y32):
        """L' Y for float32 Y, so Y'K Y = (L'Y)'(L'Y); requires triangular."""
        return _trmm(self.factor, Y32, transpose=True)


def _trmm(L, X, *, transpose):
    """L X or L' X for lower-triangular float32 L: half the flops of a general product."""
    from scipy.linalg.blas import strmm
    return strmm(1.0, L, np.asfortranarray(X, dtype=np.float32), side=0, lower=1,
                 trans_a=1 if transpose else 0, diag=0, overwrite_b=1)


class _Grm2UsKernelSampler:
    """N(0, K~) draws from a grm2Us randomized eigensystem plus a trace-matched bulk.

    K~ = U diag(s) U' + b (I - U U') with b = (tr K - sum s)/(n - r), so every
    animal keeps its kernel variance on average; the omitted spectrum is
    approximated as isotropic. The eigensystem comes from vigor.kernel_eigen,
    so it is shared with the h2 prior pilot of the same fit (and with every
    hyper_opt fold). Randomized work O(n² r) once, storage O(n r).
    """

    def __init__(self, op, rank, seed, kernel=None, name=None, position=0):
        from .multivariate import LowRankKernelOperator
        n = int(op.shape[0])
        self.n = n
        if isinstance(op, LowRankKernelOperator):
            self.U = np.asarray(op.U, np.float32)
            self.root = np.sqrt(np.maximum(np.asarray(op.values, float), 0.0))
            self.bulk = max(float(op.bulk), 0.0)
            self.approximation = 'supplied_lowrank'
            self.retained_trace_fraction = np.nan
            return
        from .kernel_eigen import eigensystem, resolve_rank
        r = resolve_rank(rank, n, name, position)
        if r is None:
            raise ValueError("analytic_kernel_factor='grm2us' needs a retained rank; use 'cholesky' for the full kernel")
        eig = eigensystem(op if kernel is None else kernel, lambda: _dense_kernel(op), r, seed=int(seed))
        self.U = eig['U']
        self.root = np.sqrt(np.maximum(np.asarray(eig['values'], float), 0.0))
        self.bulk = float(eig['bulk'])
        self.approximation = f'grm2us_randomized_rank{r}_trace_matched'
        self.retained_trace_fraction = float(eig['retained_trace_fraction'])

    def sample(self, rng, ncols):
        m = int(ncols)
        if m <= 0:
            return np.zeros((self.n, 0), float)
        out = self.U @ (self.root[:, None] * rng.normal(size=(len(self.root), m)))
        if self.bulk > 0:
            Z = rng.normal(size=(self.n, m))
            Z -= self.U @ (self.U.T @ Z)
            out += math.sqrt(self.bulk) * Z
        return np.asarray(out, float)

    def apply(self, X):
        """K~ X = U (s - b) U'X + b X for the same approximate kernel the draws use; O(n r m)."""
        X = np.asarray(X, np.float32)
        shrink = (self.root**2 - self.bulk).astype(np.float32)
        return self.U @ (shrink[:, None] * (self.U.T @ X)) + np.float32(self.bulk) * X


class _ComponentTangent:
    """Batched tangent J, adjoint J' and loading curvature of one FA/PCA/sparse/full component."""

    def __init__(self, spec, *, boundary_tol, eps):
        F = np.asarray(spec.factor, float)
        T = F.shape[0]
        psi = np.asarray(spec.diag_specific, float)
        support = getattr(spec, 'support_mask', None)
        support = np.ones(F.shape, bool) if support is None else np.asarray(support, bool)
        if support.shape != F.shape:
            raise ValueError('sparse FA support mask shape does not match factor loadings')
        self.T = T
        self.F = np.where(support, F, 0.0)
        self.support = support
        self.free = bool(F.shape[1] >= T and support.all())
        diag = np.maximum(np.sum(self.F * self.F, axis=1) + np.maximum(psi, 0.0), 0.0)
        scale = max(float(np.nanmedian(diag)) if diag.size else 0.0, float(eps), 1e-12)
        threshold = max(float(eps) * 10.0, float(boundary_tol) * scale)
        self.active = np.array([], dtype=int) if self.free else np.flatnonzero(np.isfinite(psi) & (psi > threshold))
        specific = str(getattr(spec, 'structure', 'fa')) in {'fa', 'sparse_fa'} and not self.free
        self.boundary_count = int(T - self.active.size) if specific else 0
        self.k = F.shape[1]
        self.size = T * T if self.free else T * self.k + self.active.size

    def apply(self, x):
        P = x.shape[0]
        T = self.T
        if self.free:
            S = x.reshape(P, T, T)
            return 0.5 * (S + np.swapaxes(S, 1, 2))
        # One (P*T x k) GEMM instead of P small batched products.
        L = (x[:, :T * self.k].reshape(P, T, self.k) * self.support[None]).reshape(P * T, self.k)
        out = (L @ self.F.T).reshape(P, T, T)
        out = out + np.swapaxes(out, 1, 2)
        if self.active.size:
            out[:, self.active, self.active] += x[:, T * self.k:]
        return out

    def adjoint(self, V):
        P = V.shape[0]
        T = self.T
        V = 0.5 * (V + np.swapaxes(V, 1, 2))
        if self.free:
            return V.reshape(P, T * T)
        VF = (V.reshape(P * T, T) @ self.F).reshape(P, T, self.k)
        parts = [(2.0 * VF * self.support[None]).reshape(P, T * self.k)]
        if self.active.size:
            parts.append(V[:, self.active, self.active])
        return np.concatenate(parts, axis=1)

    def curvature(self, x, R):
        """Residual curvature <R, d²(FF')> = 2 R L on the support; specific variances are linear."""
        P = x.shape[0]
        out = np.zeros_like(x)
        if self.free or not self.k:
            return out
        T = self.T
        L = x[:, :T * self.k].reshape(P, T, self.k) * self.support[None]
        RL = (R @ L.transpose(1, 0, 2).reshape(T, P * self.k)).reshape(T, P, self.k).transpose(1, 0, 2)
        out[:, :T * self.k] = (2.0 * RL * self.support[None]).reshape(P, T * self.k)
        return out

    def preconditioner(self, h):
        """Diagonal of J'HJ for this component's own Hessian block h (T x T)."""
        T = self.T
        if self.free:
            d = 0.5 * h.copy()
            np.fill_diagonal(d, np.diag(h))
            return d.reshape(T * T)
        F2 = self.F * self.F
        load = 2.0 * (h @ F2) + 2.0 * np.diag(h)[:, None] * F2
        load = np.where(self.support, load, 1.0)
        return np.concatenate([load.reshape(T * self.k), np.diag(h)[self.active]])


def _center(X, obs, nobs):
    X = np.where(obs, X, 0.0)
    mean = np.divide(X.sum(axis=0), nobs, out=np.zeros(X.shape[1]), where=nobs > 0)
    return np.where(obs, X - mean[None, :], 0.0)


def _moments(kops, knames, left, right, obs, nobs, same=False):
    """sym(L0' K_c R0) for every kernel and the identity: float32 kernel products, float64 reductions."""
    L0 = _center(left, obs, nobs)
    R0 = L0 if same else _center(right, obs, nobs)
    R32 = R0.astype(np.float32)
    T = L0.shape[1]
    out = np.empty((len(knames) + 1, T, T))
    for k, nm in enumerate(knames):
        out[k] = L0.T @ np.asarray(kops[nm].matmat(R32), dtype=np.float64)
    out[-1] = L0.T @ R0
    return 0.5 * (out + np.swapaxes(out, 1, 2))


def _apply_model(kops, knames, cov, X):
    """V(X) = sum_c K_c X C_c + X E of the fitted model; X is animals x traits."""
    out = X @ cov['E']
    for nm in knames:
        out += np.asarray(kops[nm].matmat((X @ cov[nm]).astype(np.float32)), dtype=np.float64)
    return out


def _paired_moments(kops, knames, cov, U, W, obs, nobs, KM, chunk=4):
    """Paired moment draws xi = sym(U0'K V(W)0), eta = sym(V(U)0'K W0) for a batch of probes.

    U, W are (m, n, t) masked Rademacher probes. Only K U and K W are formed:
    K (W C) = (K W) C gives V(W), K U0 = K U - (K M) diag(mean U) gives the
    centered product (M the observation mask, K M precomputed once), and
    U0'K V(W)0 = (K U0)'V(W)0 by symmetry. That is 2c kernel products per probe,
    as for a Gaussian draw, and chunk probes share one wide float32 GEMM.
    Returns xi, eta of shape (m, c+1, t, t).
    """
    m, n, T = U.shape
    p = len(knames) + 1
    xi = np.empty((m, p, T, T)); eta = np.empty((m, p, T, T))
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        block = np.concatenate([np.concatenate([U[j], W[j]], axis=1) for j in range(start, stop)], axis=1)
        products = {nm: np.asarray(kops[nm].matmat(block.astype(np.float32)), dtype=np.float64) for nm in knames}
        for j in range(start, stop):
            col = 2 * T * (j - start)
            u, w = U[j], W[j]
            mu_u = np.divide(u.sum(axis=0), nobs, out=np.zeros(T), where=nobs > 0)
            mu_w = np.divide(w.sum(axis=0), nobs, out=np.zeros(T), where=nobs > 0)
            u0 = np.where(obs, u - mu_u, 0.0); w0 = np.where(obs, w - mu_w, 0.0)
            ku = {nm: products[nm][:, col:col + T] for nm in knames}
            kw = {nm: products[nm][:, col + T:col + 2 * T] for nm in knames}
            vw = w @ cov['E']; vu = u @ cov['E']
            for nm in knames:
                vw += kw[nm] @ cov[nm]
                vu += ku[nm] @ cov[nm]
            vw0 = _center(vw, obs, nobs); vu0 = _center(vu, obs, nobs)
            for k, nm in enumerate(knames):
                ku0 = ku[nm] - KM[nm] * mu_u[None, :]
                kw0 = kw[nm] - KM[nm] * mu_w[None, :]
                xi[j, k] = ku0.T @ vw0
                eta[j, k] = vu0.T @ kw0
            xi[j, -1] = u0.T @ vw0
            eta[j, -1] = vu0.T @ w0
    xi = 0.5 * (xi + np.swapaxes(xi, 2, 3))
    eta = 0.5 * (eta + np.swapaxes(eta, 2, 3))
    return xi, eta


def _covariance_root(C):
    """R with R R' = C for a PSD trait covariance (negative rounding clipped)."""
    values, vectors = np.linalg.eigh(0.5 * (C + C.T))
    return vectors * np.sqrt(np.maximum(values, 0.0))[None, :]


def _gaussian_moments(samplers, kops, knames, roots, obs, nobs, rng, m, chunk=4):
    """Moments of m Gaussian phenotype draws y = sum_c K_c^1/2 Z_c R_c' + Z_E R_E'.

    R_c R_c' = C_c, so each kernel needs one n x t root product per draw rather
    than separate loading and specific products. chunk draws share one wide
    product for the kernel roots and one for the moments (float32, float64
    reductions). With an exact Cholesky factor (K = L L') both are triangular
    trmm products, L Z and L'Y0 with Y0'K Y0 = (L'Y0)'(L'Y0), at half the cost
    of a general product. A grm2us sampler forms the moments with its own
    low-rank-plus-bulk kernel, so draws and moments describe the same
    approximate kernel at O(n r t) per draw. Returns (m, c+1, t, t).
    """
    n, T = obs.shape
    p = len(knames) + 1
    B = np.empty((m, p, T, T))
    for start in range(0, m, int(chunk)):
        b = min(int(chunk), m - start)
        Y = [rng.standard_normal(size=(n, T)) @ roots['E'].T for _ in range(b)]
        for nm in knames:
            draw = samplers[nm].sample(rng, b * T)
            for j in range(b):
                Y[j] += draw[:, j * T:(j + 1) * T] @ roots[nm].T
        Y0 = [_center(y, obs, nobs) for y in Y]
        stack = np.concatenate(Y0, axis=1).astype(np.float32)
        for k, nm in enumerate(knames):
            if hasattr(samplers[nm], 'apply'):
                KY = np.asarray(samplers[nm].apply(stack), dtype=np.float64)
                for j in range(b):
                    B[start + j, k] = Y0[j].T @ KY[:, j * T:(j + 1) * T]
            elif getattr(samplers[nm], 'triangular', False):
                H = np.asarray(samplers[nm].half_kernel(stack), dtype=np.float64)
                for j in range(b):
                    Hj = H[:, j * T:(j + 1) * T]
                    B[start + j, k] = Hj.T @ Hj
            else:
                KY = np.asarray(kops[nm].matmat(stack), dtype=np.float64)
                for j in range(b):
                    B[start + j, k] = Y0[j].T @ KY[:, j * T:(j + 1) * T]
        for j in range(b):
            B[start + j, -1] = Y0[j].T @ Y0[j]
    return 0.5 * (B + np.swapaxes(B, 2, 3))


class _IndefiniteHessian(ValueError):
    """The observed (Newton) sandwich Hessian is not positive definite on the tangent."""


def _batched_pcg(normal, rhs, precond, rtol, maxiter, delay=10):
    """PCG over rows of rhs stopped on the relative energy-norm error (Hestenes-Stiefel delay sum).

    The energy norm ||J(x - x*)||_H is the error of the projected draw itself,
    so rotation-gauge and weakly identified loading directions that barely
    move J x do not hold the solve hostage. Converged rows leave the batch.
    """
    npr = rhs.shape[0]
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual / precond
    rz = np.sum(residual * direction, axis=1)
    active = rz > 0
    energy = np.zeros(npr)
    window = deque(maxlen=delay)
    relative = np.zeros(npr)
    iterations = 0
    for iterations in range(1, int(maxiter) + 1):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        d = direction[idx]
        Ad = normal(d)
        curvature = np.sum(d * Ad, axis=1)
        if np.any(curvature <= 0):
            raise _IndefiniteHessian('FA sandwich Hessian is not positive definite on the fitted tangent')
        alpha = rz[idx] / curvature
        x[idx] += alpha[:, None] * d
        residual[idx] -= alpha[:, None] * Ad
        step = np.zeros(npr)
        step[idx] = alpha * rz[idx]
        energy += step
        window.append(step)
        if len(window) == delay:
            relative = np.sqrt(np.divide(np.sum(window, axis=0), energy, out=np.zeros(npr), where=energy > 0))
            active &= relative > float(rtol)
        z = residual[idx] / precond
        rz_new = np.sum(residual[idx] * z, axis=1)
        exact = rz_new <= 0
        active[idx[exact]] = False
        relative[idx[exact]] = 0.0
        if not active.any():
            break
        keep = active[idx]
        beta = rz_new[keep] / rz[idx[keep]]
        direction[idx[keep]] = z[keep] + beta[:, None] * d[keep]
        rz[idx[keep]] = rz_new[keep]
    if active.any():
        raise ValueError(f'FA tangent projection did not converge in {int(maxiter)} iterations '
                         f'(max relative energy error {float(relative[active].max()):.3g} > {float(rtol):g}); '
                         'increase fa_se_cg_maxiter or relax fa_se_cg_rtol. No approximate SE is substituted.')
    return x, iterations, float(relative.max()) if npr else 0.0


class _CrossAccumulator:
    """Running covariance of linear functionals from Gaussian draws or paired probes.

    Gaussian: left = right = draw, shifted by the first block mean (shift
    invariant, avoids cancellation) and centered at the end. Paired: E[left]
    = E[right] = 0 and Cov(a, b) = E[a_l b_r + a_r b_l].
    """

    def __init__(self, paired, stats):
        self.paired = paired
        self.stats = stats
        self.P = 0
        self.shift = None
        self.cross = {}
        self.linear = {}
        self.mc = {}

    def add(self, left, right):
        if not self.paired:
            if self.shift is None:
                self.shift = {k: v.mean(axis=0, keepdims=True) for k, v in left.items()}
            left = {k: v - self.shift[k] for k, v in left.items()}
            right = left
            for k, v in left.items():
                self.linear[k] = self.linear.get(k, 0.0) + v.sum(axis=0, keepdims=True)
        for name, (fa, fb) in self.stats.items():
            if self.paired:
                al, ar, bl, br = fa(left), fa(right), fb(left), fb(right)
                value = 0.5 * np.sum(al * br + ar * bl, axis=0)
            else:
                # Gaussian draws: left is right, so the symmetrized product is a single one.
                value = np.einsum('p...,p...->...', *np.broadcast_arrays(fa(left), fb(left)))
            self.cross[name] = self.cross.get(name, 0.0) + value
        for name in ('g', 'e', 'h'):
            a, b = left[name], right[name]
            c = a * b
            s = self.mc.setdefault(name, [0.0, 0.0, 0.0, 0.0])
            if self.paired:
                s[0] = s[0] + c.sum(axis=0); s[1] = s[1] + (c * c).sum(axis=0)
            else:
                s[0] = s[0] + a.sum(axis=0); s[1] = s[1] + (a**2).sum(axis=0)
                s[2] = s[2] + (a**3).sum(axis=0); s[3] = s[3] + (a**4).sum(axis=0)
        self.P += next(iter(left.values())).shape[0]

    def covariance(self, name):
        fa, fb = self.stats[name]
        if self.paired:
            return 2.0 * self.cross[name] / self.P
        mean_a = fa(self.linear)[0] / self.P
        mean_b = fb(self.linear)[0] / self.P
        return (self.cross[name] - self.P * mean_a * mean_b) / max(self.P - 1, 1)

    def variance_and_mc(self, name):
        """Variance estimate of a monitored scalar and the relative MC error of its SE."""
        s = self.mc[name]
        P = self.P
        if self.paired:
            m = s[0] / P
            spread = np.maximum(s[1] / P - m * m, 0.0)
            variance = 2.0 * m
            rel_var = np.divide(np.sqrt(spread / P), np.abs(m), out=np.full(m.shape, np.inf), where=m != 0)
        else:
            e1, e2, e3, e4 = (x / P for x in s)
            m2 = np.maximum(e2 - e1 * e1, 0.0)
            m4 = np.maximum(e4 - 4 * e1 * e3 + 6 * e1 * e1 * e2 - 3 * e1**4, 0.0)
            variance = m2 * P / max(P - 1, 1)
            rel_var = np.divide(np.sqrt(np.maximum(m4 - m2 * m2, 0.0) / P), m2,
                                out=np.full(m2.shape, np.inf), where=m2 > 0)
        return variance, 0.5 * rel_var


@operation_progress("analytic SE")
def final_fa_sandwich_se(kernels, Y, moments, specs, *, ridge, h2_penalty=None, n_probes=64,
                         max_probes=None, mc_rtol=None, kernel_factor='paired', kernel_rank='auto',
                         random_state=0, eps=1e-8, boundary_tol=1e-8, cg_rtol=1e-3,
                         cg_maxiter=2000, ci_level=0.95, probe_block=32, progress=None):
    """Final FA/sparse-FA Newton sandwich for G, E, h2 and rG on heterogeneous masks.

    kernels is the ordered genetic kernel mapping, Y the animal x trait
    outcomes, moments the fitted (closure-augmented) pairwise moments, specs the
    fitted component specs in moment order (genetic..., 'E'), ridge one value
    per component. Exactly n_probes probes (pairs for 'paired') are used and
    every SE's relative Monte Carlo error is reported. Only when mc_rtol is
    given are probes added in blocks of probe_block, up to max_probes (default
    1024), until the 90th percentile of the monitored SEs' errors (marginal G/E
    variances and h2 of non-boundary traits) is <= mc_rtol and no monitored
    variance is negative. Cost per probe: 2c kernel products of n x t blocks
    for every backend; Gaussian backends add one O(n³/3) ('cholesky') or
    O(n² r) ('grm2us') kernel factorization. CG work
    O(I P c t² r). Memory O(c n t) per probe plus kernel factors.
    Raises ValueError when the tangent solve fails; negative paired variance
    estimates give NaN SEs, never clipped values.
    """
    from .multivariate import as_kernel_operator, _analytic_delta_summaries, operation_stage, current_work
    from .h2_prior import penalty_hessian_apply, penalty_curvature_apply, prior_sensitivity, penalty_terms
    if kernel_factor not in KERNEL_FACTORS:
        raise ValueError(f"analytic_kernel_factor must be one of {KERNEL_FACTORS}")
    paired = kernel_factor == 'paired'
    values = Y.to_numpy(float) if hasattr(Y, 'to_numpy') else np.asarray(Y, float)
    obs = np.isfinite(values)
    n, T = values.shape
    knames = list(kernels)
    names = knames + ['E']
    p = len(names)
    A = np.asarray(moments['A'], float)
    if A.shape[:2] != (p, p):
        raise ValueError(f'moment system has shape {A.shape[:2]}, expected {(p, p)}')
    Bdata = np.asarray(moments['B'], float)
    ridge = np.asarray(ridge, float).reshape(-1)
    if ridge.size != p:
        raise ValueError('ridge needs one value per covariance component')
    npr_min = int(n_probes)
    if npr_min < 8:
        raise ValueError('analytic_se_probes must be at least 8')
    adaptive = mc_rtol is not None
    if adaptive and not (np.isfinite(mc_rtol) and mc_rtol > 0):
        raise ValueError('analytic_se_mc_rtol must be finite and positive, or None for exactly analytic_se_probes')
    # Without a Monte Carlo target exactly n_probes are used.
    npr_max = (int(max_probes) if max_probes is not None else 1024) if adaptive else npr_min
    if npr_max < npr_min:
        raise ValueError('analytic_se_max_probes must be at least analytic_se_probes')
    block = max(1, int(probe_block))
    bar = current_work()
    Atilde = A.copy()
    for r in range(p):
        Atilde[r, r] += ridge[r]
    Atilde = 0.5 * (Atilde + np.swapaxes(Atilde, 2, 3))
    closure = np.zeros(T)
    if 'variance_closure_weight' in moments:
        closure = np.nan_to_num(np.asarray(moments['variance_closure_weight'], float), nan=0.0)
        closure = np.where(closure > 0, closure, 0.0)
    target = np.asarray(moments.get('variance_closure_target', np.full(T, np.nan)), float)
    closure = np.where(np.isfinite(target) & (target > 0), closure, 0.0)
    nobs = obs.sum(axis=0)
    idx = np.arange(T)

    point = [specs[nm].materialize() for nm in names]
    cov = {nm: np.asarray(point[c], float) for c, nm in enumerate(names)}
    diagonals = [np.diag(c) for c in point]
    vg = np.sum(diagonals[:-1], axis=0)
    ve = diagonals[-1]
    S_total = vg + ve
    # Fitted half gradient R_r = sum_q A~_rq C_q - B_r (+ half prior gradient): residual curvature.
    Rfit = [sum(Atilde[r, q] * point[q] for q in range(p)) - Bdata[r] for r in range(p)]
    if h2_penalty is not None:
        _, pgrads, _, _ = penalty_terms(diagonals, h2_penalty)
        for r in range(p):
            Rfit[r] = Rfit[r] + np.diag(0.5 * pgrads[r])
        prior_apply = prior_sensitivity(h2_penalty, moments, ridge, diagonals)
    Rfit = [0.5 * (R + R.T) for R in Rfit]

    operation_stage('analytic SE: kernel preparation')
    kops = {nm: as_kernel_operator(kernels[nm], dtype=np.float32) for nm in knames}
    samplers = {}
    if kernel_factor == 'cholesky':
        samplers = {nm: _CholeskyKernelSampler(kops[nm]) for nm in knames}
    elif kernel_factor == 'grm2us':
        # Seed 113*k matches the h2 prior pilot, so both reuse one eigensystem per kernel.
        samplers = {nm: _Grm2UsKernelSampler(kops[nm], kernel_rank, 113 * k, kernel=kernels[nm], name=nm, position=k)
                    for k, nm in enumerate(knames)}

    operation_stage('analytic SE: tangent prep')
    tangents = [_ComponentTangent(specs[nm], boundary_tol=boundary_tol, eps=eps) for nm in names]
    sizes = [t.size for t in tangents]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def jac(x):
        return [tangents[r].apply(x[:, offsets[r]:offsets[r + 1]]) for r in range(p)]

    def jac_adjoint(V):
        return np.concatenate([tangents[r].adjoint(V[r]) for r in range(p)], axis=1)

    # Observed (Newton) Hessian when it is positive definite on the tangent;
    # otherwise the expected-information (Gauss-Newton) Hessian. Sparse FA can
    # have first-order null directions of J (e.g. a loading with one or two
    # nonzeros trading against its trait's specific variance) along which only
    # the residual curvature acts; with a negative residual there the observed
    # Hessian is indefinite although those directions do not move the fitted
    # covariance. The residual term has zero expectation under the model, so the
    # Gauss-Newton bread is the standard Fisher-type sandwich. The choice is
    # recorded in the result and warned about, never silent.
    mode = {'newton': True, 'reason': None, 'reset': False}

    def hessian(V):
        out = [sum(Atilde[r, q][None] * V[q] for q in range(p)) for r in range(p)]
        if h2_penalty is not None:
            delta = np.stack([V[r][:, idx, idx] for r in range(p)])
            extra = penalty_hessian_apply(delta, vg, ve, h2_penalty)
            if mode['newton']:
                extra = extra + penalty_curvature_apply(delta, vg, ve, h2_penalty)
            for r in range(p):
                out[r][:, idx, idx] += extra[r]
        return out

    precond = np.concatenate([tangents[r].preconditioner(Atilde[r, r]) for r in range(p)])
    positive = precond[precond > 0]
    floor = (float(np.median(positive)) if positive.size else 1.0) * 1e-10
    jitter = floor
    precond = np.maximum(precond, floor) + jitter

    def normal(x):
        out = jac_adjoint(hessian(jac(x))) + jitter * x
        if mode['newton']:
            for r in range(p):
                out[:, offsets[r]:offsets[r + 1]] += tangents[r].curvature(x[:, offsets[r]:offsets[r + 1]], Rfit[r])
        return out

    def effective(B):
        """Closure moment, closure precision and prior-weight movement for a batch of moment draws."""
        B = np.array(B, dtype=float, copy=True)
        dlam = np.zeros((B.shape[0], T))
        if np.any(closure > 0):
            vp = np.divide(B[:, -1, idx, idx], nobs - 1.0, out=np.zeros((B.shape[0], T)), where=nobs > 1)
            B[:, :, idx, idx] += (closure * vp)[:, None, :]
            # lambda = strength (n-1) / (2 Vp²): d lambda = -2 lambda dVp / Vp.
            dlam = np.divide(-2.0 * closure * vp, target, out=np.zeros_like(vp), where=closure > 0)
        out = B.copy()
        out[:, :, idx, idx] -= (dlam * (S_total - np.nan_to_num(target)))[:, None, :]
        if h2_penalty is not None:
            out[:, :, idx, idx] -= prior_apply(B[:, :, idx, idx], dlam)
        return out

    def project(B):
        dB = effective(B)
        rhs = jac_adjoint([dB[:, r] for r in range(p)])
        try:
            x, iterations, energy = _batched_pcg(normal, rhs, precond, cg_rtol, cg_maxiter)
        except _IndefiniteHessian as exc:
            if not mode['newton']:
                raise ValueError('FA sandwich Gauss-Newton Hessian is not positive definite on the fitted tangent; '
                                 'analytical SE unavailable') from exc
            import warnings
            mode.update(newton=False, reason=str(exc), reset=True)
            warnings.warn('FA sandwich: observed Hessian not positive definite on the sparse/FA tangent; '
                          'using the expected-information (Gauss-Newton) Hessian. See analytic_se["hessian"].',
                          RuntimeWarning, stacklevel=2)
            x, iterations, energy = _batched_pcg(normal, rhs, precond, cg_rtol, cg_maxiter)
        return jac(x), iterations, energy

    # Linearized reported quantities for each projected draw.
    good_total = S_total > max(float(eps), 1e-15)
    total2 = np.where(good_total, S_total**2, 1.0)
    dG = np.where(good_total, ve / total2, 0.0)
    dE = np.where(good_total, -vg / total2, 0.0)
    gpos = vg > max(float(eps), 1e-15)
    Gfit = np.sum(point[:-1], axis=0)
    root = np.sqrt(np.where(gpos, vg, 1.0))
    Rg = Gfit / np.outer(root, root)
    rg_ok = np.outer(gpos, gpos)
    dxy = np.where(rg_ok, 1.0 / np.outer(root, root), 0.0)
    inv_g = np.where(gpos, 1.0 / np.where(gpos, vg, 1.0), 0.0)

    def derived(draws):
        Gx = np.sum(draws[:-1], axis=0)
        Ex = draws[-1]
        g = Gx[:, idx, idx]; e = Ex[:, idx, idx]
        h = dG * g + dE * e
        rg = dxy * Gx - 0.5 * Rg * ((g * inv_g)[:, :, None] + (g * inv_g)[:, None, :])
        rg = np.where(rg_ok, rg, 0.0)
        out = {'G': Gx, 'E': Ex, 'g': g, 'e': e, 'h': h, 'rg': rg}
        for c in range(p):
            out[f'C{c}'] = draws[c]
        return out

    stats = {'Gvar': (lambda d: d['G'], lambda d: d['G']),
             'Evar': (lambda d: d['E'], lambda d: d['E']),
             'ts_tt': (lambda d: d['G'], lambda d: d['g'][:, :, None]),
             'ts_ss': (lambda d: d['G'], lambda d: d['g'][:, None, :]),
             'diagdiag': (lambda d: d['g'][:, :, None], lambda d: d['g'][:, None, :]),
             'GE': (lambda d: d['g'], lambda d: d['e']),
             'h': (lambda d: d['h'], lambda d: d['h']),
             'rg': (lambda d: d['rg'], lambda d: d['rg'])}
    for c in range(p):
        stats[f'C{c}'] = (lambda d, c=c: d[f'C{c}'], lambda d, c=c: d[f'C{c}'])
    acc = _CrossAccumulator(paired, stats)
    rel_floor = 1e-6
    monitor = {'g': vg > rel_floor * S_total, 'e': ve > rel_floor * S_total}
    monitor['h'] = monitor['g'] & monitor['e']

    rng = np.random.default_rng(int(random_state) + 285025)
    roots = {nm: _covariance_root(cov[nm]) for nm in names}
    KM = {}
    if paired:
        # K M (M the observation mask) turns K U into K U0 for every probe.
        mask32 = obs.astype(np.float32)
        KM = {nm: np.asarray(kops[nm].matmat(mask32), dtype=np.float64) for nm in knames}
    cg_iterations = 0
    typical = np.inf
    cg_energy = 0.0
    worst = np.inf
    met = False
    mc = {}
    while acc.P < npr_max:
        # The first block is the minimum probe count; later blocks top it up.
        b = min(npr_min if acc.P == 0 else block, npr_max - acc.P)
        if bar is not None:
            bar.add_total(b)
        left = []; right = []
        if paired:
            for start in range(0, b, PAIRED_CHUNK):
                stop = min(start + PAIRED_CHUNK, b)
                operation_stage(f'analytic SE: probe {acc.P + stop}/{acc.P + b}')
                shape = (stop - start, n, T)
                U = np.where(obs[None], rng.integers(0, 2, size=shape, dtype=np.int8) * 2.0 - 1.0, 0.0)
                W = np.where(obs[None], rng.integers(0, 2, size=shape, dtype=np.int8) * 2.0 - 1.0, 0.0)
                xi, eta = _paired_moments(kops, knames, cov, U, W, obs, nobs, KM, chunk=PAIRED_CHUNK)
                left.extend(xi); right.extend(eta)
                if bar is not None:
                    bar.update(stop - start)
        else:
            for start in range(0, b, PAIRED_CHUNK):
                stop = min(start + PAIRED_CHUNK, b)
                operation_stage(f'analytic SE: probe {acc.P + stop}/{acc.P + b}')
                left.extend(_gaussian_moments(samplers, kops, knames, roots, obs, nobs, rng,
                                              stop - start, chunk=PAIRED_CHUNK))
                if bar is not None:
                    bar.update(stop - start)
        operation_stage(f'analytic SE: tangent solve, probes {acc.P + 1}-{acc.P + b}')
        batch = np.stack(left + right)
        draws, iterations, energy = project(batch)
        if mode['reset']:
            # Every probe must use the same Hessian: drop draws projected with the observed one.
            acc = _CrossAccumulator(paired, stats)
            mode['reset'] = False
        cg_iterations = max(cg_iterations, iterations)
        cg_energy = max(cg_energy, energy)
        if paired:
            acc.add(derived([d[:b] for d in draws]), derived([d[b:] for d in draws]))
        else:
            acc.add(derived(draws), None)
        errors = []
        positive = True
        for key in ('g', 'e', 'h'):
            variance, rel = acc.variance_and_mc(key)
            mc[key] = rel
            sel = monitor[key]
            if sel.any():
                positive &= bool(np.all(variance[sel] > 0))
                errors.append(np.where(variance[sel] > 0, rel[sel], np.inf))
        errors = np.concatenate(errors) if errors else np.zeros(1)
        worst = float(np.max(errors))
        typical = float(np.quantile(errors, MC_QUANTILE))
        if adaptive and positive and typical <= float(mc_rtol):
            met = True
            break

    operation_stage('analytic SE: delta/rG/h2')
    Gvar = acc.covariance('Gvar'); Evar = acc.covariance('Evar')
    component_var = [acc.covariance(f'C{c}') for c in range(p)]
    negative = 0
    h_var = acc.covariance('h')
    rg_var = acc.covariance('rg')
    if paired:
        # A negative paired estimate means too few probes: report it unavailable.
        negative = int(np.sum(Gvar < 0) + np.sum(Evar < 0))
        Gvar = np.where(Gvar < 0, np.nan, Gvar)
        Evar = np.where(Evar < 0, np.nan, Evar)
        component_var = [np.where(v < 0, np.nan, v) for v in component_var]
        bad_h = ~(h_var >= 0)
        bad_rg = ~(rg_var >= 0) & rg_ok
        np.fill_diagonal(bad_rg, False)
    else:
        # Gaussian sample covariances are PSD; only rounding can dip below zero.
        component_var = [np.maximum(v, 0.0) for v in component_var]
        bad_h = np.zeros(T, bool)
        bad_rg = np.zeros((T, T), bool)
    summary = _analytic_delta_summaries(Gfit, point[-1], Gvar, Evar, acc.covariance('ts_tt'),
                                        acc.covariance('ts_ss'), acc.covariance('diagdiag'),
                                        acc.covariance('GE'), eps=eps, ci_level=ci_level)
    if bad_h.any():
        summary['h2_se'] = np.where(bad_h, np.nan, summary['h2_se'])
        summary['h2_ci95'] = np.where(bad_h[:, None], np.nan, summary['h2_ci95'])
    if bad_rg.any():
        for key in ('rg_se', 'rg_ci95_low', 'rg_ci95_high'):
            summary[key] = np.where(bad_rg, np.nan, summary[key])
    negative += int(bad_h.sum() + bad_rg.sum() // 2)
    packed = np.packbits(obs, axis=1)
    summary.update({
        'method': 'masked_mom_final_fa_newton_sandwich',
        'kernel_backend': kernel_factor,
        'probe_estimator': 'paired_bilinear_rademacher' if paired else 'gaussian_draws',
        'n_probes': int(acc.P), 'min_probes': npr_min, 'max_probes': npr_max,
        'mc_rtol': float(mc_rtol) if adaptive else None,
        'mc_target_met': bool(met) if adaptive else None,
        'se_relative_mc_error': {k: v for k, v in mc.items()},
        'se_relative_mc_error_max': float(worst),
        f'se_relative_mc_error_q{int(MC_QUANTILE * 100)}': float(typical),
        'mc_stopping': (f'{int(MC_QUANTILE * 100)}th percentile of monitored per-SE relative MC error <= mc_rtol '
                        'and no negative monitored variance' if adaptive else 'fixed probe count (analytic_se_probes)'),
        'negative_variance_estimates': int(negative),
        'mask_pattern_count': int(len(np.unique(packed, axis=0))),
        'common_trait_mask': bool((obs == obs[:, :1]).all()),
        'joint_kernel_covariance_retained': True,
        'kernel_names': tuple(knames), 'n_kernels': len(knames),
        'kernel_factor': {nm: ('matrix_free_full_kernel' if paired else samplers[nm].approximation) for nm in knames},
        'kernel_retained_trace_fraction': {nm: getattr(samplers.get(nm), 'retained_trace_fraction', 1.0) for nm in knames},
        'component_se': {nm: np.sqrt(component_var[c]) for c, nm in enumerate(names)},
        'component_var': {nm: component_var[c] for c, nm in enumerate(names)},
        'tangent_dimensions': {nm: int(tangents[c].size) for c, nm in enumerate(names)},
        'fa_boundary_counts': {nm: int(tangents[c].boundary_count) for c, nm in enumerate(names)},
        'cg_iterations': int(cg_iterations), 'cg_max_relative_energy_error': float(cg_energy),
        'cg_stopping': 'energy_norm_hestenes_stiefel_delay10',
        'hessian': ('newton (Gauss-Newton + loading residual curvature + exact h2-prior curvature)' if mode['newton'] else
                    'gauss_newton (expected information; observed Hessian not positive definite on the tangent)'),
        'observed_hessian_indefinite': not mode['newton'],
        'variance_closure_in_draws': bool(np.any(closure > 0)),
        'closure_precision_propagated': bool(np.any(closure > 0)),
        'h2_prior_in_sandwich': h2_penalty is not None,
        'h2_prior_weights_propagated': h2_penalty is not None,
        'conditional_on_rank': True, 'conditional_on_regularization_policy': True,
        'constraint_note': ('First-order linearization (Hessian as recorded in hessian) of the fitted FA/sparse-FA moment estimator '
                            'in its own pairwise metric, including the data dependence of the closure precision '
                            'and of the h2-prior pilot h2, DerSimonian-Laird tau2 and kappa (pilot SEs held fixed). '
                            'Specific variances at the boundary and zero loadings outside the selected support are '
                            'held fixed; rank, support, prior strength and grouping are conditioned on. Finite-probe '
                            'Monte Carlo error is reported per SE.'),
    })
    return summary
