"""Observed-space multivariate GLS score tests with checked block PCG.

There is no phenotype whitening shortcut or empirical score calibration here.
The model is V = sum_c K_c (x) G_c + I (x) E on the observed phenotype cells.
Every design information matrix is D' P D, where P projects all trait-specific
intercepts and supplied fixed effects in the V^-1 metric. ``exact`` describes
this statistical model, not infinite-precision arithmetic or an estimated G/E.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh, qr
from scipy.special import gammaincc, gammaln
from scipy.sparse import coo_matrix, csr_matrix


def _sym(a):
    return (a + a.T) * 0.5


def _labels(index, name):
    out = pd.Index(np.asarray(index).astype(str))
    if out.has_duplicates:
        raise ValueError(f"{name} labels must be unique")
    return out


def _matrix(value, labels=None, name="matrix", dtype=np.float64):
    if isinstance(value, pd.DataFrame):
        rows = _labels(value.index, name)
        cols = _labels(value.columns, name)
        if labels is not None:
            if set(rows) != set(labels) or set(cols) != set(labels):
                raise ValueError(f"{name} labels must match the modeled axis")
            value = value.copy().set_axis(rows, axis=0).set_axis(cols, axis=1).loc[labels, labels]
        value = value.to_numpy()
    elif hasattr(value, "materialize"):
        value = value.materialize()
    elif isinstance(value, Mapping) and "matrix" in value:
        value = value["matrix"]
    elif hasattr(value, "dims") and hasattr(value, "to_pandas"):
        return _matrix(value.to_pandas(), labels, name, dtype=dtype)
    a = np.asarray(value, dtype=dtype)
    if a.dtype.kind not in "fc":
        a = a.astype(np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite square matrix")
    scale = max(float(np.max(np.abs(a), initial=0)), 1.0)
    if not np.allclose(a, a.T, rtol=1e-8, atol=1e-10 * scale):
        raise ValueError(f"{name} must be symmetric")
    return _sym(a)


def _psd_inverse(a, *, require_positive=False):
    """Return a checked pseudoinverse, estimable rank, and range projector."""
    a = _sym(np.asarray(a, dtype=np.float64))
    w, u = eigh(a, check_finite=True)
    largest = max(float(np.max(np.abs(w), initial=0)), np.finfo(float).tiny)
    tol = max(a.shape[0], 1) * np.finfo(float).eps * largest * 100
    if np.min(w, initial=0) < -max(1e-8 * largest, tol):
        raise RuntimeError("score information/covariance is not positive semidefinite")
    good = w > tol
    if require_positive and not np.all(good):
        raise ValueError("residual covariance must be strictly positive definite")
    inv = (u[:, good] / w[good]) @ u[:, good].T
    proj = u[:, good] @ u[:, good].T
    return _sym(inv), int(good.sum()), proj


def chi2_logsf(statistic, dof):
    """Compute log chi-square upper tails without underflowing significance.

    Parameters
    ----------
    statistic, dof : broadcastable array-like
        Nonnegative finite statistics and strictly positive degrees of freedom.
        Invalid pairs return NaN; +infinity statistics return -infinity.

    Returns
    -------
    ndarray
        Natural logarithms of upper-tail probabilities. ``exp(result)`` may
        underflow to zero while the log probability remains finite.

    Complexity
    ----------
    O(N I) time and O(N) space, with at most I=10000 continued-fraction steps
    only for probabilities below the floating-point range.
    """
    s, d = np.broadcast_arrays(np.asarray(statistic, float), np.asarray(dof, float))
    out = np.full(s.shape, np.nan)
    valid = (s >= 0) & (d > 0) & np.isfinite(d)
    out[valid & np.isposinf(s)] = -np.inf
    finite = valid & np.isfinite(s)
    a = d[finite] / 2
    x = s[finite] / 2
    p = gammaincc(a, x)
    result = np.full(p.shape, -np.inf)
    positive = p > 0
    result[positive] = np.log(p[positive])
    # Lentz's continued fraction for the regularized upper incomplete gamma.
    # Underflow happens in the upper tail, where x > a + 1 and this is stable.
    for j in np.flatnonzero(~positive):
        aa, xx = float(a[j]), float(x[j])
        b = xx + 1 - aa
        floor = np.finfo(float).tiny / np.finfo(float).eps
        c = 1 / floor
        dd = 1 / max(b, floor)
        h = dd
        for i in range(1, 10001):
            an = -i * (i - aa)
            b += 2
            dd = an * dd + b
            if abs(dd) < floor:
                dd = np.copysign(floor, dd)
            c = b + an / c
            if abs(c) < floor:
                c = np.copysign(floor, c)
            dd = 1 / dd
            delta = dd * c
            h *= delta
            if abs(delta - 1) < 2e-14:
                break
        else:
            raise RuntimeError("chi-square log-tail continued fraction did not converge")
        if h <= 0 or not np.isfinite(h):
            raise RuntimeError("chi-square log-tail continued fraction was unstable")
        result[j] = -xx + aa * np.log(xx) - gammaln(aa) + np.log(h)
    out[finite] = np.minimum(result, 0)
    return out


class _Kernel:
    """Exact supplied kernel with explicit float64 residual multiplication."""

    def __init__(self, value, ids, name):
        self.dense = None
        self.lowrank = None
        self.source = value
        if isinstance(value, (pd.DataFrame, np.ndarray, list, tuple)) or hasattr(value, "dims"):
            self.dense = _matrix(value, ids, name, dtype=None)
        elif hasattr(value, "K"):
            self.dense = _matrix(value.K, None, name, dtype=None)
        elif all(hasattr(value, key) for key in ("U", "values", "bulk")):
            self.lowrank = (np.asarray(value.U, float), np.asarray(value.values, float), float(value.bulk))
            u, w, bulk = self.lowrank
            if u.shape != (len(ids), len(w)) or not np.isfinite(u).all() or not np.isfinite(w).all():
                raise ValueError(f"{name} has invalid spectral factors")
            if not np.allclose(u.T @ u, np.eye(len(w)), atol=2e-5):
                raise ValueError(f"{name} spectral eigenvectors must be orthonormal")
            if np.min(w, initial=0) < 0 or bulk < 0:
                raise ValueError(f"{name} must have nonnegative eigenvalues")
        elif not hasattr(value, "row_block"):
            raise TypeError(f"{name} operator needs row_block for float64 residual verification")
        if hasattr(value, "sample_ids"):
            if not _labels(value.sample_ids, name).equals(ids):
                raise ValueError(f"{name} operator sample order must match traits")
        shape = self.dense.shape if self.dense is not None else getattr(value, "shape", None)
        if shape != (len(ids), len(ids)):
            raise ValueError(f"{name} shape must equal the trait sample axis")
        self.shape = shape
        if self.dense is not None:
            # A full eigenvalue check is also used for the eigen preconditioner.
            self.diag = np.diag(self.dense).copy()
        elif self.lowrank is not None:
            u, w, bulk = self.lowrank
            self.diag = (u * u) @ (w - bulk) + bulk
        else:
            self.diag = np.asarray(value.diag(), float)
        if self.diag.shape != (len(ids),) or not np.isfinite(self.diag).all() or np.any(self.diag < 0):
            raise ValueError(f"{name} has an invalid kernel diagonal")

    def dot(self, x, dtype):
        if self.lowrank is not None:
            u, w, bulk = self.lowrank
            u = u.astype(dtype, copy=False)
            a = np.asarray(x, dtype=dtype)
            return u @ ((w - bulk).astype(dtype)[:, None] * (u.T @ a)) + bulk * a
        a = np.asarray(x, dtype=dtype)
        if self.dense is not None:
            # Avoid a persistent second n-by-n kernel copy in mixed precision.
            out = np.empty((self.shape[0], a.shape[1]), dtype=dtype)
            for start in range(0, self.shape[0], 512):
                stop = min(start + 512, self.shape[0])
                out[start:stop] = self.dense[start:stop].astype(dtype, copy=False) @ a
            return out
        out = np.empty((self.shape[0], a.shape[1]), dtype=dtype)
        for start in range(0, self.shape[0], 512):
            stop = min(start + 512, self.shape[0])
            block = np.asarray(self.source.row_block(start, stop), dtype=dtype)
            if block.shape != (stop - start, self.shape[0]) or not np.isfinite(block).all():
                raise ValueError("kernel row_block returned invalid values or dimensions")
            out[start:stop] = block @ a
        return out


def _block_pcg(apply, precondition, b, *, verify, tol, maxiter):
    """Rank-revealing, reliably restarted block PCG; never returns failed solves."""
    b = np.asarray(b, float)
    if b.ndim != 2 or not np.isfinite(b).all():
        raise ValueError("block-PCG RHS must be a finite matrix")
    if not b.size:
        return np.zeros_like(b), {"iterations": 0, "relative_residual": 0.0}
    # Solving an independent basis handles identical/zero RHS without singular
    # block Gram systems. The coefficient reconstruction retains all originals.
    q, r, _ = qr(b, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    cutoff = max(b.shape) * np.finfo(float).eps * max(diagonal.max(initial=0), np.finfo(float).tiny)
    rank = int((diagonal > cutoff).sum())
    if rank == 0:
        return np.zeros_like(b), {"iterations": 0, "relative_residual": 0.0}
    q = q[:, :rank]
    coefficients = q.T @ b
    x = np.zeros_like(q)
    active = np.arange(rank)
    residual = q.copy()
    z = precondition(residual)
    direction = z.copy()
    gram = _sym(residual.T @ z)
    iterations = 0
    for iterations in range(1, maxiter + 1):
        ad = apply(direction)
        curvature = _sym(direction.T @ ad)
        try:
            ci, crank, _ = _psd_inverse(curvature)
        except RuntimeError:
            # A near-dependent block can expose float32 asymmetry. Recheck
            # its curvature in the same float64 operator used by acceptance.
            ad = verify(direction)
            curvature = _sym(direction.T @ ad)
            ci, crank, _ = _psd_inverse(curvature)
        if crank == 0:
            # Rebuild from an independently verified residual instead of using
            # a fabricated ridge in a singular search-direction system.
            residual = q[:, active] - verify(x[:, active])
            if np.max(np.linalg.norm(residual, axis=0), initial=0) <= tol:
                active = active[:0]
                break
            z = precondition(residual)
            direction = z.copy()
            gram = _sym(residual.T @ z)
            continue
        step = ci @ gram
        x[:, active] += direction @ step
        residual -= ad @ step
        apparent = np.linalg.norm(residual, axis=0)
        # Frequent double-precision residual replacement controls drift in
        # float32 GEMMs and provides iterative refinement of the supplied V.
        if iterations % 8 == 0 or np.min(apparent, initial=np.inf) <= tol:
            residual = q[:, active] - verify(x[:, active])
            remaining = np.linalg.norm(residual, axis=0) > tol * 0.25
            if not remaining.any():
                active = active[:0]
                break
            active = active[remaining]
            residual = residual[:, remaining]
            z = precondition(residual)
            direction = z.copy()
            gram = _sym(residual.T @ z)
            continue
        z = precondition(residual)
        new_gram = _sym(residual.T @ z)
        gi, _, _ = _psd_inverse(gram)
        direction = z + direction @ (gi @ new_gram)
        gram = new_gram
    result = x @ coefficients
    final = b - verify(result)
    norm = np.linalg.norm(b, axis=0)
    error = np.linalg.norm(final, axis=0) / np.maximum(norm, np.finfo(float).tiny)
    maximum = float(np.max(error, initial=0))
    if not np.isfinite(result).all() or maximum > tol:
        raise RuntimeError(f"exact-score block PCG did not converge: relative residual={maximum:.3g}, tolerance={tol:.3g}, iterations={iterations}")
    return result, {"iterations": iterations, "relative_residual": maximum, "independent_rhs": rank}


@dataclass
class ExactScoreNull:
    """Prepared observed covariance, fixed effects, and reusable precision projector.

    Use :func:`prepare_score_null` to construct this state. Changing sample or
    trait axes, missingness, kernels, covariance, or covariates requires a new
    state. No phenotype cell outside ``observed`` enters the null solve.
    """

    y: np.ndarray
    sample_ids: pd.Index
    trait_names: pd.Index
    observed: np.ndarray
    kernels: dict
    covariances: dict
    residual: np.ndarray
    fixed: csr_matrix
    solver_tol: float
    solver_maxiter: int
    work_dtype: np.dtype
    rhs_block_size: int
    jitter: float
    preconditioner_name: str

    def apply(self, rhs, *, precision=None):
        """Multiply V on observed cells; ``precision='float64'`` verifies solves."""
        b = np.asarray(rhs)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.observed):
            raise ValueError("covariance RHS must have one row per observed phenotype cell")
        dt = self.work_dtype if precision is None else np.dtype(precision)
        n, t = self.y.shape
        r = b.shape[1]
        full = np.zeros((n * t, r), dtype=dt)
        full[self.observed] = b
        full = full.reshape(n, t, r)
        flat_traits = full.transpose(0, 2, 1).reshape(n * r, t)
        result = (flat_traits @ self.residual.astype(dt)).reshape(n, r, t).transpose(0, 2, 1).astype(float)
        if self.jitter:
            result += self.jitter * full
        for name, kernel in self.kernels.items():
            animal = kernel.dot(full.reshape(n, t * r), dt).reshape(n, t, r)
            term = animal.transpose(0, 2, 1).reshape(n * r, t) @ self.covariances[name].astype(dt)
            result += term.reshape(n, r, t).transpose(0, 2, 1)
        out = result.reshape(n * t, r)[self.observed]
        return out[:, 0] if vector else out

    def solve(self, rhs):
        """Solve V X=rhs in bounded RHS blocks with float64 residual checks.

        Returns ``(X, diagnostics)``; failures raise RuntimeError. This is true
        block PCG with small dense cross-RHS Gram systems, not a serial CG loop.
        """
        b = np.asarray(rhs, float)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.observed):
            raise ValueError("solve RHS must have one row per observed phenotype cell")
        result = np.empty_like(b)
        diagnostics = []
        for start in range(0, b.shape[1], self.rhs_block_size):
            sl = slice(start, start + self.rhs_block_size)
            result[:, sl], info = _block_pcg(self.apply, self._precondition, b[:, sl],
                verify=lambda x: self.apply(x, precision="float64"),
                tol=self.solver_tol, maxiter=self.solver_maxiter)
            diagnostics.append(info)
        return (result[:, 0] if vector else result), {"blocks": diagnostics,
            "relative_residual": max((d["relative_residual"] for d in diagnostics), default=0.0)}

    def project(self, rhs):
        """Apply P=V^-1 - V^-1 C(C'V^-1 C)^+ C'V^-1 to one/many RHS."""
        out, _ = self.solve(rhs)
        weights = self.fixed_information_inverse @ (self.fixed.T @ out)
        correction, _ = self.solve(self.fixed @ weights)
        return out - correction

    def score_block(self, genotypes, **kwargs):
        """Score one sample-by-variant dosage block; see :func:`score_block`."""
        return score_block(self, genotypes, **kwargs)


def _prepare_preconditioner(null):
    n, t = null.y.shape
    if null.preconditioner_name not in {"kronecker", "diag"}:
        raise ValueError("preconditioner must be 'kronecker' or 'diag'")
    diagonal = np.broadcast_to(np.diag(null.residual) + null.jitter, (n, t)).copy()
    for name, kernel in null.kernels.items():
        diagonal += kernel.diag[:, None] * np.diag(null.covariances[name])[None, :]
    diagonal = diagonal.ravel()[null.observed]
    if np.any(diagonal <= 0):
        raise ValueError("observed covariance must have positive diagonal")
    if null.preconditioner_name == "diag":
        null._precondition = lambda x: x / diagonal[:, None]
        null.preconditioner_details = "observed marginal-variance diagonal"
        return
    # The selected kernel is only a preconditioner; every supplied kernel stays
    # unchanged in the actual V multiplication and double-precision checks.
    name = max(null.kernels, key=lambda k: float(np.mean(null.kernels[k].diag)))
    kernel = null.kernels[name]
    e = null.residual.copy() + null.jitter * np.eye(t)
    for other, component in null.kernels.items():
        if other != name:
            e += float(np.mean(component.diag)) * null.covariances[other]
    ew, eu = eigh(e)
    if np.min(ew) <= 0:
        raise ValueError("Kronecker preconditioner requires positive-definite residual covariance")
    inverse_root = (eu / np.sqrt(ew)) @ eu.T
    gw, gu = eigh(_sym(inverse_root @ null.covariances[name] @ inverse_root))
    transform = inverse_root @ gu
    if kernel.lowrank is not None:
        ku, kw, bulk = kernel.lowrank
    elif kernel.dense is not None:
        kw, ku = eigh(kernel.dense)
        bulk = None
    else:
        raise ValueError("Kronecker preconditioning needs a dense or spectral kernel; use preconditioner='diag' for row-block-only operators")
    if np.min(kw, initial=0) < -1e-8 * max(np.max(np.abs(kw), initial=0), 1.0):
        raise ValueError("animal kernel must be positive semidefinite")
    if np.min(gw, initial=0) < -1e-8 * max(np.max(np.abs(gw), initial=0), 1.0):
        raise ValueError("genetic covariance must be positive semidefinite")
    denominator = 1 + np.maximum(kw, 0)[:, None] * np.maximum(gw, 0)[None, :]
    # The preconditioner is approximate arithmetic even when its mathematical
    # form is the exact complete-data inverse. Keep its large GEMMs in the
    # requested working precision; only the residual verifier defines success.
    dt = null.work_dtype
    ku = ku.astype(dt, copy=False)
    transform = transform.astype(dt, copy=False)
    denominator = denominator.astype(dt, copy=False)
    def precondition(rhs):
        r = rhs.shape[1]
        full = np.zeros((n * t, r), dtype=dt)
        full[null.observed] = rhs
        full = full.reshape(n, t, r)
        transformed = np.einsum("ntr,tf->nfr", full, transform, optimize=True)
        reduced = (ku.T @ transformed.reshape(n, t * r)).reshape(len(kw), t, r)
        if bulk is None:
            solved = ku @ (reduced / denominator[:, :, None]).reshape(len(kw), t * r)
            solved = solved.reshape(n, t, r)
        else:
            base_denominator = (1 + bulk * np.maximum(gw, 0)).astype(dt)
            solved = transformed / base_denominator[None, :, None]
            correction = reduced * (1 / denominator - 1 / base_denominator[None, :])[:, :, None]
            solved += (ku @ correction.reshape(len(kw), t * r)).reshape(n, t, r)
        result = np.einsum("nfr,tf->ntr", solved, transform, optimize=True)
        return result.reshape(n * t, r)[null.observed].astype(np.float64)
    null._precondition = precondition
    exact = len(null.kernels) == 1 and len(null.observed) == n * t
    null.preconditioner_details = ("complete single-kernel Kronecker inverse" if exact else
        "restricted full-space Kronecker inverse (approximate preconditioner; exact V retained)")


def prepare_score_null(traits, kernels, G, E, *, covariates=None, solver_tol=2e-6,
                       solver_maxiter=400, work_dtype="float32", rhs_block_size=16,
                       preconditioner="kronecker", jitter=0.0):
    """Prepare an exact observed-space multivariate score null.

    Parameters
    ----------
    traits : pandas.DataFrame or (n, t) array
        Gaussian phenotypes; NaN means unobserved. Labeled samples/traits must
        be unique. Infinite values and traits with fewer than two cells fail.
    kernels : (n, n) array/DataFrame, kernel operator, or mapping
        Supplied animal kernels, without implicit rank approximation. Operators
        need row_block/diag/shape or spectral U/values/bulk; labeled operators
        must already follow the phenotype sample order.
    G : (t, t) array/DataFrame/specification or mapping
        PSD genetic covariance matching each named kernel. A factor covariance
        specification may expose materialize(). Named keys must match exactly.
    E : (t, t) array/DataFrame/specification
        Strictly positive-definite residual covariance.
    covariates : (n, p) array/DataFrame, optional
        Finite sample covariates; a separate coefficient is fitted for each
        trait. Trait-specific intercepts are always included. Redundant columns
        are removed by a rank-revealing QR on observed cells.
    solver_tol : positive float
        Maximum float64 verified relative residual for every RHS, default 2e-6.
    solver_maxiter, rhs_block_size : positive int
        Block-PCG iteration limit and maximum simultaneous RHS, respectively.
    work_dtype : 'float32' or 'float64'
        Float32 GEMMs by default; reductions, fixed effects, information and
        residual verification use float64. No dense observed V is assembled.
    preconditioner : 'kronecker' or 'diag'
        The Kronecker eigen inverse is exact only for complete single-kernel
        models. Under missingness/multiple kernels it is an SPD preconditioner.
        Dense eigen setup is O(n^3); row-block-only kernels require 'diag'.
    jitter : nonnegative float
        Explicit residual variance added to V. Zero leaves the supplied model
        unchanged; its value is always recorded in output metadata.

    Returns
    -------
    ExactScoreNull
        Reusable observed-space model with solve/project/score_block methods.
        Null construction performs checked solves; nonconvergence raises.

    Complexity
    ----------
    With k dense kernels, per block-PCG iteration time is O(k n² t b +
    k n t² b + N b²), space O(k n² + N b + N p + q²); N observed cells,
    b RHS width, p covariates per trait, q fixed-effect rank. Fixed-effect
    projection uses a second solve to avoid storing an N-by-q inverse design. Kronecker setup adds O(n³+t³) time and
    O(n²+t²) space. Information for c contrasts/r basis effects needs c*r RHS
    per variant; exact inference is intentionally more costly than calibration.
    """
    if hasattr(traits, "to_pandas") and not isinstance(traits, pd.DataFrame):
        if tuple(traits.dims) != ("sample", "trait") or not {"sample", "trait"}.issubset(traits.coords):
            raise ValueError("phenotype DataArray requires labeled ('sample', 'trait') dimensions")
        traits = traits.to_pandas()
    y = traits.to_numpy(float, copy=True) if isinstance(traits, pd.DataFrame) else np.array(traits, float, copy=True)
    if y.ndim != 2 or not y.shape[0] or not y.shape[1] or np.isinf(y).any():
        raise ValueError("traits must be a nonempty sample-by-trait matrix with finite values or NaN")
    n, t = y.shape
    if np.any(np.isfinite(y).sum(axis=0) < 2):
        raise ValueError("each trait requires at least two observed samples")
    ids = _labels(traits.index if isinstance(traits, pd.DataFrame) else np.arange(n), "sample")
    names = _labels(traits.columns if isinstance(traits, pd.DataFrame) else np.arange(t), "trait")
    if not np.isfinite(solver_tol) or solver_tol <= 0:
        raise ValueError("solver_tol must be finite and positive")
    for label, value in (("solver_maxiter", solver_maxiter), ("rhs_block_size", rhs_block_size)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    dt = np.dtype(work_dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("work_dtype must be float32 or float64")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    components = dict(kernels) if isinstance(kernels, Mapping) else {"genetic": kernels}
    if not components:
        raise ValueError("at least one animal kernel is required")
    if isinstance(G, Mapping) and set(G) == set(components):
        genetic = dict(G)
    elif len(components) == 1:
        genetic = {next(iter(components)): G}
    else:
        raise ValueError("multi-kernel G keys must match the kernel names exactly")
    kops = {name: _Kernel(value, ids, str(name)) for name, value in components.items()}
    gcov = {name: _matrix(value, names, f"G[{name}]") for name, value in genetic.items()}
    residual = _matrix(E, names, "E")
    for name, a in list(gcov.items()) + [("E", residual)]:
        if a.shape != (t, t):
            raise ValueError(f"{name} shape must equal the trait axis")
        _psd_inverse(a, require_positive=(name == "E"))
    observed = np.flatnonzero(np.isfinite(y.ravel()))
    # Center in float64 before float32 GEMMs. Intercepts remain in the GLS
    # projection, so Euclidean centering cannot substitute for fixed effects.
    y -= np.nanmean(y, axis=0)[None, :]
    if covariates is None:
        x = np.ones((n, 1))
    else:
        if isinstance(covariates, pd.DataFrame):
            ci = _labels(covariates.index, "covariate sample")
            if set(ci) != set(ids):
                raise ValueError("covariate sample IDs must match traits")
            covariates = covariates.copy().set_axis(ci, axis=0).loc[ids]
        x = np.asarray(covariates, float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != n or not np.isfinite(x).all():
            raise ValueError("covariates must be finite sample-by-covariate values")
        x = np.column_stack([np.ones(n), x - x.mean(axis=0)])
    sample, trait = np.unravel_index(observed, (n, t))
    # Each trait has disjoint observed rows. Build its small fixed-effect QR
    # independently and retain a sparse global basis, never an Nobs-by-t array.
    lookup = np.full(n * t, -1, dtype=np.int64)
    lookup[observed] = np.arange(len(observed))
    row_parts, column_parts, value_parts = [], [], []
    fixed_rank = 0
    for j in range(t):
        samples = np.flatnonzero(np.isfinite(y[:, j]))
        rows = lookup[samples * t + j]
        cq, cr, _ = qr(x[samples], mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(cr))
        threshold = max(x[samples].shape) * np.finfo(float).eps * max(diagonal.max(initial=0), np.finfo(float).tiny)
        local_rank = int((diagonal > threshold).sum())
        if local_rank >= len(samples):
            raise ValueError("fixed effects consume all observed cells for at least one trait")
        row_parts.append(np.repeat(rows, local_rank))
        column_parts.append(np.tile(np.arange(fixed_rank, fixed_rank + local_rank), len(rows)))
        value_parts.append(cq[:, :local_rank].ravel())
        fixed_rank += local_rank
    fixed = coo_matrix((np.concatenate(value_parts),
        (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(len(observed), fixed_rank)).tocsr()
    null = ExactScoreNull(y, ids, names, observed, kops, gcov, residual, fixed,
        float(solver_tol), int(solver_maxiter), dt, int(rhs_block_size), float(jitter), str(preconditioner))
    _prepare_preconditioner(null)
    fixed_information = np.empty((fixed_rank, fixed_rank))
    null.fixed_solve_info = []
    for start in range(0, fixed_rank, null.rhs_block_size):
        stop = min(start + null.rhs_block_size, fixed_rank)
        solved, info = null.solve(fixed[:, start:stop].toarray())
        fixed_information[:, start:stop] = fixed.T @ solved
        null.fixed_solve_info.append(info)
    null.fixed_information_inverse, _, _ = _psd_inverse(fixed_information)
    null.py = null.project(y.ravel()[observed])
    null.observed_sample = sample
    null.observed_trait = trait
    return null


def score_design_block(null, designs, *, basis=None, contrast_names=None,
                       trait_names=None, variant_indices=None, variant_names=None):
    """Score per-locus multi-contrast designs with exact D'PD information.

    Parameters
    ----------
    null : ExactScoreNull
        Prepared on the same phenotype/sample axes as the design.
    designs : (n, m, c) finite-or-NaN array
        c contrasts for m loci, e.g. F-1 founder probabilities or additive and
        dominance dosages. Missing values are mean-imputed per contrast/locus
        after float64 centering. All-missing contrasts contain zero information.
    basis : (t, r) array/DataFrame, optional
        Effect restriction beta_trait=basis @ beta_basis. None uses identity.
        DataFrame rows are aligned to phenotype traits; r named columns define
        tested factors. The basis may be rank deficient; omnibus df is rank(I).
    contrast_names : length-c sequence, optional
        Contrast labels; c>1 outputs ``contrast:basis`` coordinate labels.
    trait_names : length-r sequence, optional
        Explicit names for basis directions, overriding DataFrame columns.
    variant_indices, variant_names : length-m sequence, optional
        Global unique integer variant indices and display labels for streaming.

    Returns
    -------
    xarray.Dataset
        variant x trait arrays for beta, beta_se, z, stat, p, log_p, neglog_p,
        nobs, joint_beta, joint_beta_se, conditional_stat/log_p/neglog_p;
        variant-only omnibus_stat/dof/log_p/neglog_p/p arrays. Coordinate score
        tests constrain other effects to zero. Joint estimates/conditional tests
        use the full information inverse; nonestimable coordinates are NaN.
        All log probabilities use natural logs; neglog_p is -log10(p).

    Complexity
    ----------
    O(m c r) covariance RHS solves plus O(m (c r)^3) small information work;
    peak working memory O(N b + (c r)^2 + m (c r)), not O(N m c r). Kernels
    and the fixed-effect state are reused. Genotype blocks stay caller-bounded.
    """
    import xarray as xr
    x = np.asarray(designs, dtype=np.float64)
    n, t = null.y.shape
    if x.ndim != 3 or x.shape[0] != n or np.isinf(x).any():
        raise ValueError("designs must be sample-by-variant-by-contrast, finite or NaN")
    _, m, c = x.shape
    if c < 1:
        raise ValueError("at least one design contrast is required")
    if basis is None:
        b = np.eye(t)
        labels = null.trait_names
    elif isinstance(basis, pd.DataFrame):
        bi = _labels(basis.index, "basis trait")
        if set(bi) != set(null.trait_names):
            raise ValueError("basis trait labels must match the null")
        basis = basis.copy().set_axis(bi, axis=0).loc[null.trait_names]
        b = basis.to_numpy(float)
        labels = _labels(basis.columns, "basis")
    else:
        b = np.asarray(basis, float)
        labels = _labels(np.arange(b.shape[1]) if b.ndim == 2 else [], "basis")
    if b.ndim != 2 or b.shape[0] != t or not b.shape[1] or not np.isfinite(b).all():
        raise ValueError("basis must be a finite trait-by-effect matrix")
    if trait_names is not None:
        labels = _labels(trait_names, "basis")
    r = b.shape[1]
    if len(labels) != r:
        raise ValueError("trait_names length must match basis columns")
    contrast_names = _labels(np.arange(c) if contrast_names is None else contrast_names, "contrast")
    if len(contrast_names) != c:
        raise ValueError("contrast_names length must match design contrasts")
    output_names = labels if c == 1 else _labels([f"{cn}:{tn}" for cn in contrast_names for tn in labels], "effect")
    indices = np.arange(m, dtype=np.int64) if variant_indices is None else np.asarray(variant_indices)
    if indices.shape != (m,) or indices.dtype.kind not in "iu" or np.any(indices < 0) or len(np.unique(indices)) != m:
        raise ValueError("variant_indices must be unique nonnegative integers")
    names = indices.astype(str) if variant_names is None else np.asarray(variant_names).astype(str)
    if names.shape != (m,):
        raise ValueError("variant_names length must match variants")
    q = c * r
    fields = ("beta", "beta_se", "z", "stat", "p", "log_p", "neglog_p", "nobs",
              "joint_beta", "joint_beta_se", "conditional_stat", "conditional_log_p", "conditional_neglog_p")
    result = {key: np.full((m, q), np.nan) for key in fields}
    omnibus = {key: np.full(m, np.nan) for key in ("stat", "dof", "log_p", "neglog_p", "p")}
    si, ti = null.observed_sample, null.observed_trait
    for variant in range(m):
        xx = x[:, variant, :].copy()
        finite = np.isfinite(xx)
        counts = finite.sum(axis=0)
        mean = np.divide(np.where(finite, xx, 0).sum(axis=0), counts,
                         out=np.zeros(c), where=counts > 0)
        xx = np.where(finite, xx - mean, 0.0)
        # Centered genotypes may be stored/streamed as float32, but the exact
        # information contractions and score reductions accumulate in float64.
        xx = xx.astype(null.work_dtype).astype(np.float64)
        def contract(values):
            # D' values from per-trait grouped sums, without materializing D.
            vector = values.ndim == 1
            vv = values[:, None] if vector else values
            out = np.empty((q, vv.shape[1]))
            for contrast in range(c):
                sums = np.empty((t, vv.shape[1]))
                for col in range(vv.shape[1]):
                    sums[:, col] = np.bincount(ti, weights=xx[si, contrast] * vv[:, col], minlength=t)
                out[contrast * r:(contrast + 1) * r] = b.T @ sums
            return out[:, 0] if vector else out
        u = contract(null.py)
        information = np.empty((q, q))
        for start in range(0, q, null.rhs_block_size):
            stop = min(start + null.rhs_block_size, q)
            columns = np.arange(start, stop)
            contrast = columns // r
            effect = columns % r
            design = xx[si[:, None], contrast[None, :]] * b[ti[:, None], effect[None, :]]
            information[:, start:stop] = contract(null.project(design))
        information = _sym(information)
        inverse, rank, projector = _psd_inverse(information)
        diagonal = np.diag(information)
        max_info = np.max(diagonal, initial=0)
        informative = diagonal > max(max_info * 1e-12, np.finfo(float).tiny)
        result["beta"][variant, informative] = u[informative] / diagonal[informative]
        result["beta_se"][variant, informative] = 1 / np.sqrt(diagonal[informative])
        result["z"][variant, informative] = u[informative] / np.sqrt(diagonal[informative])
        result["stat"][variant] = result["z"][variant] ** 2
        lp = chi2_logsf(result["stat"][variant], 1)
        result["log_p"][variant] = lp
        result["p"][variant] = np.exp(lp)
        result["neglog_p"][variant] = -lp / np.log(10)
        # nobs counts genuinely observed genotypes and phenotype cells with a
        # nonzero tested loading; mean imputation does not inflate this count.
        for contrast in range(c):
            for effect in range(r):
                result["nobs"][variant, contrast * r + effect] = np.count_nonzero(
                    finite[si, contrast] & (b[ti, effect] != 0))
        estimable = np.diag(projector) > 1 - 1e-7
        joint = inverse @ u
        joint_se = np.sqrt(np.maximum(np.diag(inverse), 0))
        estimable &= joint_se > 0
        result["joint_beta"][variant, estimable] = joint[estimable]
        result["joint_beta_se"][variant, estimable] = joint_se[estimable]
        result["conditional_stat"][variant, estimable] = (joint[estimable] / joint_se[estimable]) ** 2
        clp = chi2_logsf(result["conditional_stat"][variant], 1)
        result["conditional_log_p"][variant] = clp
        result["conditional_neglog_p"][variant] = -clp / np.log(10)
        omnibus["dof"][variant] = rank
        if rank:
            statistic = max(float(u @ inverse @ u), 0.0)
            lp = float(chi2_logsf(statistic, rank))
            omnibus["stat"][variant] = statistic
            omnibus["log_p"][variant] = lp
            omnibus["neglog_p"][variant] = -lp / np.log(10)
            omnibus["p"][variant] = np.exp(lp)
    data = {name: (("variant", "trait"), value) for name, value in result.items()}
    data.update({f"omnibus_{name}": (("variant",), value) for name, value in omnibus.items()})
    ds = xr.Dataset(data, coords={"variant": indices.astype(np.int64), "trait": np.asarray(output_names, str),
        "variant_id": ("variant", names)}, attrs={"engine": "exact_multivariate_score",
        "covariance_model": "supplied full kernels and G/E; observed cells only",
        "fixed_effects": "trait-specific intercepts and supplied covariates",
        "fixed_effect_rank": int(null.fixed.shape[1]), "work_dtype": str(null.work_dtype),
        "reduction_dtype": "float64", "solver_tolerance": null.solver_tol,
        "preconditioner": null.preconditioner_details, "jitter": null.jitter,
        "coordinate_test": "one-direction score; all other tested effects fixed to zero",
        "joint_estimates": "full information inverse; nonestimable coordinates are NaN",
        "null_covariance_uncertainty": "conditional on supplied G/E estimates"})
    ds["nobs"].attrs["description"] = "Observed phenotype cells with an observed genotype and nonzero tested loading"
    from .. import __version__
    ds.attrs["vigor_version"] = __version__
    for name in ("log_p", "conditional_log_p", "omnibus_log_p"):
        ds[name].attrs["units"] = "natural log probability"
    for name in ("neglog_p", "conditional_neglog_p", "omnibus_neglog_p"):
        ds[name].attrs["units"] = "-log10 probability"
    return ds


def score_block(null, genotypes, *, basis=None, trait_names=None,
                variant_indices=None, variant_names=None):
    """Score an aligned sample-by-variant dosage block using score_design_block.

    DataFrame rows are matched by unique sample ID (exact same sample set);
    xarray DataArrays must have sample then variant dimensions and labeled
    sample coordinates. Unlabeled arrays must already follow null.sample_ids.
    Returns the canonical variant-by-trait Dataset described in
    :func:`score_design_block`, with one genotype contrast per variant.
    """
    if hasattr(genotypes, "dims") and hasattr(genotypes, "to_pandas"):
        if tuple(genotypes.dims) != ("sample", "variant"):
            raise ValueError("genotype DataArray dimensions must be ('sample', 'variant')")
        if "sample" not in genotypes.coords:
            raise ValueError("genotype DataArray needs explicit sample coordinates")
        genotypes = genotypes.to_pandas()
    if isinstance(genotypes, pd.DataFrame):
        gi = _labels(genotypes.index, "genotype sample")
        if set(gi) != set(null.sample_ids):
            raise ValueError("genotype sample IDs must match the null")
        genotypes = genotypes.copy().set_axis(gi, axis=0).loc[null.sample_ids]
        if variant_names is None:
            variant_names = genotypes.columns.astype(str)
        x = genotypes.to_numpy(float)
    else:
        x = np.asarray(genotypes)
    if x.ndim != 2:
        raise ValueError("genotypes must be sample-by-variant")
    return score_design_block(null, x[:, :, None], basis=basis, trait_names=trait_names,
        variant_indices=variant_indices, variant_names=variant_names)


__all__ = ["ExactScoreNull", "prepare_score_null", "score_block", "score_design_block", "chi2_logsf"]
