"""Variance-component estimation in a shared LOCO eigenbasis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from .results import result_dataset, maybe_save

_SCORE_VAR_ATTRS = {
    "genetic_covariance": {"long_name": "SCORE genetic trait covariance", "description": "PSD-projected moment estimate"},
    "residual_covariance": {"long_name": "SCORE residual trait covariance", "description": "PSD-projected moment estimate"},
    "information": {"long_name": "two-component moment information matrix", "description": "rows/columns ordered (sigma_g2, sigma_e2)"},
}


def _apply_eigen_operator(
    x,
    U,
    s,
    h2,
    scale=1.0,
    power=-0.5,
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Apply ``[scale * (h2 G + (1-h2) I)]**power`` to arrays.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``U.T @ x``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    U : array-like
        Numerical operand in ``U.T @ x``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    s : array-like
        Numerical operand in ``scale * (h2 * s + (1.0 - h2))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``1.0``.

    power : array-like or numeric scalar
        Input for the operation ``U @ (utx * (d_top ** power - d_res ** power)[:, None])``. See
        ``vigor.vc._apply_eigen_operator`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape. Default: ``-0.5``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.
    """
    x = np.asarray(x, dtype=np.float64)
    was_vector = x.ndim == 1
    if was_vector:
        x = x[:, None]
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    d_top = np.maximum(scale * (h2 * s + (1.0 - h2)), tol)
    d_res = max(
        scale * (h2 * float(residual_eigenvalue) + (1.0 - h2)), tol
    )
    utx = U.T @ x
    result = U @ (utx * ((d_top ** power) - d_res ** power)[:, None])
    result += x * (d_res ** power)
    return result[:, 0] if was_vector else result


def covariance_matrix(
    U,
    s,
    h2,
    scale=1.0,
    power=-0.5,
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Materialize an eigen-operator for compatibility and small test cases.

    Parameters
    ----------
    U : array-like
        Numerical operand in ``np.asarray(U).shape[0]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``_apply_eigen_operator(identity, U, s, h2, scale, power, tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``1.0``.

    power : float
        Input for the operation ``_apply_eigen_operator(identity, U, s, h2, scale, power, tol,
        residual_eigenvalue=residual_eigenvalue)``. See ``vigor.vc.covariance_matrix`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``-0.5``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``_apply_eigen_operator`` as
        ``residual_eigenvalue``.

    Returns
    -------
    object
        Computed result from ``_apply_eigen_operator(identity, U, s, h2, scale, power, tol, residual_eigenvalue=residual_eigenvalue)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    identity = np.eye(np.asarray(U).shape[0], dtype=np.float64)
    return _apply_eigen_operator(
        identity,
        U,
        s,
        h2,
        scale,
        power,
        tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def inverse_covariance_weights(
    s, h2, yvar, residual_eigenvalue=0.0, tol=1e-10
):
    """Return complement weight and retained-eigenspace corrections for V^-1.

    Parameters
    ----------
    s : array-like
        Numerical operand in ``yvar * (h2 * s + 1.0 - h2)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    yvar : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    Returns
    -------
    object
        Computed result from ``(w_residual, 1.0 / top - w_residual)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    top = np.maximum(yvar * (h2 * s + 1.0 - h2), tol)
    residual = max(
        yvar * (h2 * float(residual_eigenvalue) + 1.0 - h2), tol
    )
    w_residual = 1.0 / residual
    return w_residual, 1.0 / top - w_residual


@dataclass
class ProfileREMLContext:
    """Cached profile ML/REML algebra for one phenotype and fixed design.

    Constructor fields
    ------------------
    y : np.ndarray
        Stored y consumed by ``__post_init__``; see those methods for the numerical/label
        contract. Required construction field.

    U : np.ndarray
        Stored U consumed by ``__post_init__``; see those methods for the numerical/label
        contract. Required construction field.

    s : np.ndarray
        Stored s consumed by ``__post_init__``/``products``; see those methods for the
        numerical/label contract. Required construction field.

    C : np.ndarray
        Stored C consumed by ``__post_init__``; see those methods for the numerical/label
        contract. Required construction field.

    method : str
        Stored method consumed by ``__post_init__``/``objective``/``fit``; see those methods for
        the numerical/label contract. Default: ``'REML'``.

    tol : float
        Stored tol consumed by ``products``/``objective``/``fit``; see those methods for the
        numerical/label contract. Default: ``1e-10``.

    residual_eigenvalue : float
        Stored residual eigenvalue consumed by ``products``; see those methods for the
        numerical/label contract. Default: ``0.0``.

    precomputed : object
        Stored precomputed consumed by ``__post_init__``; see those methods for the
        numerical/label contract. Default: ``None``.
    """

    y: np.ndarray
    U: np.ndarray
    s: np.ndarray
    C: np.ndarray
    method: str = "REML"
    tol: float = 1e-10
    residual_eigenvalue: float = 0.0
    precomputed: object = None

    def __post_init__(self):
        """Validate/coerce the constructed dataclass fields for subsequent model operations.
                
        """
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.U = np.asarray(self.U, dtype=np.float64)
        self.s = np.maximum(np.asarray(self.s, dtype=np.float64), 0.0)
        self.C = np.asarray(self.C, dtype=np.float64)
        if self.C.ndim == 1:
            self.C = self.C[:, None]
        self.method = self.method.upper()
        n = self.y.size
        if self.method not in {"ML", "REML"}:
            raise ValueError("method must be 'ML' or 'REML'")
        if self.C.shape[0] != n or self.U.shape != (n, len(self.s)):
            raise ValueError("y, U, s, and C dimensions do not agree")
        if not np.isfinite(self.y).all() or not np.isfinite(self.C).all():
            raise ValueError("y and covar must be finite")
        self.rank_c = int(np.linalg.matrix_rank(self.C))
        if self.rank_c == 0 or self.rank_c >= n:
            raise ValueError("fixed-effect design must have rank between 1 and n-1")
        self.n = n
        self.df = n - self.rank_c if self.method == "REML" else n
        self.k_residual = max(n - len(self.s), 0)
        if self.precomputed is None:
            self.uty = self.U.T @ self.y
            self.utc = self.U.T @ self.C
            self.yty = float(self.y @ self.y)
            self.cty = self.C.T @ self.y
            self.ctc = self.C.T @ self.C
        else:
            self.uty = self.precomputed["uty"]
            self.utc = self.precomputed["utc"]
            self.yty = float(self.precomputed["yty"])
            self.cty = self.precomputed["cty"]
            self.ctc = self.precomputed["ctc"]
        self.n_evaluations = 0

    def products(self, h2):
        """Compute the profile-likelihood quadratic/cross-product terms for one variance ratio.

        Parameters
        ----------
        h2 : float or one-dimensional array
            Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
            working/reference heritability when the routine builds a null precision rather than
            estimating the final fraction.

        Returns
        -------
        object
            Computed result from ``(d_top, d_residual, ywy, cwy, cwc)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        d_top = np.maximum(h2 * self.s + 1.0 - h2, self.tol)
        d_residual = max(
            h2 * float(self.residual_eigenvalue) + 1.0 - h2, self.tol
        )
        w_residual = 1.0 / d_residual
        delta = 1.0 / d_top - w_residual
        ywy = w_residual * self.yty + float(np.sum(delta * self.uty ** 2))
        cwy = w_residual * self.cty + self.utc.T @ (delta * self.uty)
        cwc = w_residual * self.ctc + self.utc.T @ (delta[:, None] * self.utc)
        return d_top, d_residual, ywy, cwy, cwc

    def objective(self, h2):
        """Evaluate the optimization objective for the current parameter vector.

        Parameters
        ----------
        h2 : float or one-dimensional array
            Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
            working/reference heritability when the routine builds a null precision rather than
            estimating the final fraction.

        Returns
        -------
        object
            Computed result from ``0.5 * value``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self.n_evaluations += 1
        d_top, d_residual, ywy, cwy, cwc = self.products(h2)
        sign, logdet_information = np.linalg.slogdet(cwc)
        if sign <= 0:
            return np.inf
        beta = np.linalg.solve(cwc, cwy)
        q = max(float(ywy - cwy @ beta), self.tol)
        sigma2 = max(q / self.df, self.tol)
        logdet = float(
            np.log(d_top).sum() + self.k_residual * np.log(d_residual)
        )
        value = self.df * np.log(sigma2) + logdet
        if self.method == "REML":
            value += logdet_information
        return 0.5 * value

    def fit(self):
        """Fit the estimator using the enclosing model configuration and aligned training inputs.

        Returns
        -------
        dict
            Result fields: ``h2``, ``h2_se``, ``h2_ci95``, ``boundary``, ``yvar``, ``converged``, ``objective``, ``n_evaluations``, ``estimator``. Field arrays preserve the corresponding input axis order.
        """
        result = minimize_scalar(
            self.objective, bounds=(1e-4, 1.0 - 1e-4), method="bounded"
        )
        h2 = float(result.x)
        _, _, ywy, cwy, cwc = self.products(h2)
        beta = np.linalg.solve(cwc, cwy)
        sigma2 = max(float(ywy - cwy @ beta) / self.df, self.tol)
        # Profile-likelihood curvature with a clipped 95% Wald interval.
        eps_h = max(1e-4, min(2e-2, 0.02 * max(h2, 1.0-h2)))
        lo_h, hi_h = max(1e-6, h2-eps_h), min(1-1e-6, h2+eps_h)
        f0 = self.objective(h2); fl = self.objective(lo_h); fh = self.objective(hi_h)
        denom = max(((hi_h-lo_h)/2.0)**2, 1e-12)
        curvature = max((fl - 2.0*f0 + fh) / denom, 0.0)
        h2_se = float(1.0/np.sqrt(curvature)) if curvature > 0 else np.nan
        boundary = bool(h2 <= 1.1e-4 or h2 >= 1.0-1.1e-4)
        # Wald CI is diagnostic; clipped because h2 is bounded.
        h2_ci = (max(0.0, h2-1.96*h2_se), min(1.0, h2+1.96*h2_se)) if np.isfinite(h2_se) else (np.nan,np.nan)
        return {
            "h2": h2,
            "h2_se": h2_se,
            "h2_ci95": h2_ci,
            "boundary": boundary,
            "yvar": sigma2,
            "converged": bool(result.success),
            "objective": float(result.fun),
            "n_evaluations": int(self.n_evaluations),
            "estimator": f"profile_{self.method.lower()}",
        }


def estimate_h2(
    y,
    U,
    s,
    method="REML",
    tol=1e-10,
    covar=None,
    residual_eigenvalue=0.0,
):
    """Profile ML/REML estimate with all eigenspace products cached.

    Parameters
    ----------
    y : array-like
        Numerical operand in ``np.asarray(y, dtype=np.float64).ravel()``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state.

    U : array-like
        Numerical operand in ``ProfileREMLContext(y, U, s, C, method=method, tol=tol,
        residual_eigenvalue=residual_eigenvalue).fit()``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``ProfileREMLContext(y, U, s, C, method=method, tol=tol,
        residual_eigenvalue=residual_eigenvalue).fit()``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'REML'``. Passed to
        ``ProfileREMLContext`` as ``method``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``ProfileREMLContext`` as ``tol``.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``ProfileREMLContext`` as
        ``residual_eigenvalue``.

    Returns
    -------
    object
        Computed result from ``ProfileREMLContext(y, U, s, C, method=method, tol=tol, residual_eigenvalue=residual_eigenvalue).fit()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    C = np.ones((len(y), 1), dtype=np.float64) if covar is None else covar
    return ProfileREMLContext(
        y,
        U,
        s,
        C,
        method=method,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    ).fit()


def estimate_h2_many(
    Y,
    U,
    s,
    covar=None,
    method="REML",
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Fit independent profile models sharing one LOCO eigenbasis.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    U : array-like
        Numerical operand in ``U.T @ Y``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    s : array-like
        Numerical operand in ``ProfileREMLContext(Y[:, column], U, s, C, method=method, tol=tol,
        residual_eigenvalue=residual_eigenvalue, precomputed={'uty': UTY[:, column], 'utc': UTC,
        'yty': YTY[column], 'cty': CTY[:, column], 'ctc': CTC})``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'REML'``. Passed to
        ``ProfileREMLContext`` as ``method``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``ProfileREMLContext`` as ``tol``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``ProfileREMLContext`` as
        ``residual_eigenvalue``.

    Returns
    -------
    object
        Computed result from ``fits``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    C = np.ones((len(Y), 1), dtype=np.float64) if covar is None else np.asarray(covar, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    U = np.asarray(U, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    UTY = U.T @ Y
    UTC = U.T @ C
    YTY = np.sum(Y * Y, axis=0)
    CTY = C.T @ Y
    CTC = C.T @ C
    fits = []
    for column in range(Y.shape[1]):
        context = ProfileREMLContext(
            Y[:, column],
            U,
            s,
            C,
            method=method,
            tol=tol,
            residual_eigenvalue=residual_eigenvalue,
            precomputed={
                "uty": UTY[:, column],
                "utc": UTC,
                "yty": YTY[column],
                "cty": CTY[:, column],
                "ctc": CTC,
            },
        )
        fits.append(context.fit())
    return fits


def _covariate_residual_basis(C):
    """Construct an orthonormal basis orthogonal to the fixed-effect design.

    Parameters
    ----------
    C : array-like
        Numerical operand in ``C[:, None]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state. Source comparisons: ``rank < 1``. See the
        rejection/fallback branches for enforcement.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    rank = int(np.linalg.matrix_rank(C))
    if rank < 1 or rank >= len(C):
        raise ValueError("fixed-effect design must have rank between 1 and n-1")
    left, _, _ = np.linalg.svd(C, full_matrices=False)
    return left[:, :rank], rank


def _nearest_psd(matrix, floor=0.0):
    """Project a symmetric covariance to the nearest supported positive-semidefinite representation.

    Parameters
    ----------
    matrix : array-like or numeric scalar
        Input for the operation ``(np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T)
        / 2.0``. See ``vigor.vc._nearest_psd`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    floor : array-like or numeric scalar
        Input for the operation ``np.maximum(values, floor)``. See ``vigor.vc._nearest_psd`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``0.0``.
    """
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def score_variance_components(
    Y,
    U,
    s,
    covar=None,
    residual_eigenvalue=0.0,
    full_covariance=False,
    tol=1e-10,
    save_path=None,
):
    """Covariate-adjusted two-component SCORE/moment estimator.

    The equations are evaluated in the residual subspace of ``covar``.  The
    independent path is O(ntr); ``full_covariance=True`` additionally computes
    the t by t genetic and residual covariance matrices and projects both to
    the positive-semidefinite cone.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    U : array-like
        Numerical operand in ``U.T @ Qc``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    s : array-like
        Numerical operand in ``np.eye(len(s), dtype=np.float64) - utc @ utc.T``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.

    full_covariance : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``_nearest_psd`` as ``floor``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``information`` (parameter_i, parameter_j) and, when
        ``full_covariance=True``, ``genetic_covariance``/``residual_covariance``
        (trait_i, trait_j; absent otherwise, so use ``ds.get``). Attrs: ``fits`` (list of
        per-trait fit dicts), ``fixed_effect_rank`` plus title/software/parameters provenance.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, n_traits = Y.shape
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    if U.shape != (n, len(s)):
        raise ValueError("Y and eigensystem dimensions do not agree")
    C = np.ones((n, 1), dtype=np.float64) if covar is None else covar
    Qc, rank_c = _covariate_residual_basis(C)
    residual_df = n - rank_c
    Yp = Y - Qc @ (Qc.T @ Y)
    utc = U.T @ Qc
    H = np.eye(len(s), dtype=np.float64) - utc @ utc.T
    bulk = float(residual_eigenvalue)
    spectral_difference = s - bulk
    trace_a = float(np.sum(spectral_difference * np.diag(H)))
    trace_a2 = float(
        np.sum(
            (spectral_difference[:, None] * spectral_difference[None, :])
            * (H * H)
        )
    )
    trace_k = trace_a + bulk * residual_df
    trace_k2 = trace_a2 + 2.0 * bulk * trace_a + bulk * bulk * residual_df
    information = np.array(
        [[trace_k2, trace_k], [trace_k, float(residual_df)]], dtype=np.float64
    )
    determinant = float(np.linalg.det(information))
    if not np.isfinite(determinant) or determinant <= tol:
        raise np.linalg.LinAlgError("SCORE variance-component information is singular")
    uty = U.T @ Yp

    if full_covariance:
        B2 = Yp.T @ Yp
        B1 = uty.T @ (spectral_difference[:, None] * uty) + bulk * B2
        genetic = (information[1, 1] * B1 - information[0, 1] * B2) / determinant
        residual = (information[0, 0] * B2 - information[1, 0] * B1) / determinant
        genetic = _nearest_psd(genetic)
        residual = _nearest_psd(residual, floor=tol)
        genetic_diag = np.diag(genetic)
        residual_diag = np.diag(residual)
    else:
        B2 = np.sum(Yp * Yp, axis=0)
        B1 = np.sum(spectral_difference[:, None] * uty * uty, axis=0) + bulk * B2
        genetic_diag = (information[1, 1] * B1 - information[0, 1] * B2) / determinant
        residual_diag = (information[0, 0] * B2 - information[1, 0] * B1) / determinant
        genetic_diag = np.maximum(genetic_diag, 0.0)
        residual_diag = np.maximum(residual_diag, tol)
        genetic = residual = None

    total = np.maximum(genetic_diag + residual_diag, tol)
    h2 = np.clip(genetic_diag / total, 0.0, 1.0 - 1e-8)
    # Cheap delta-method uncertainty from the two-component moment information.
    # The 2*V^2 I^{-1} sandwich is exact under the spherical null and is used as
    # a clearly-labelled approximation away from it; cluster/parametric bootstrap
    # remains preferable for strongly related cohorts.
    info_inv = np.linalg.inv(information)
    fits = []
    for index in range(n_traits):
        sg=float(genetic_diag[index]); se=float(residual_diag[index]); tv=float(total[index]); hh=float(h2[index])
        cov_theta = 2.0 * (tv * tv) * info_inv
        grad=np.array([se/(tv*tv), -sg/(tv*tv)],float)
        vh=float(grad @ cov_theta @ grad); hse=float(np.sqrt(max(vh,0.0)))
        fits.append({
            "h2": hh, "h2_se": hse,
            "h2_ci95": (max(0.0,hh-1.96*hse),min(1.0,hh+1.96*hse)),
            "boundary": bool(hh <= 1e-6 or hh >= 1-1e-6),
            "uncertainty_method": "moment_information_delta_approx",
            "yvar": tv, "sigma_g2": sg, "sigma_e2": se,
            "converged": True, "estimator": "score",
        })
    ds = result_dataset(
        {
            "fits": fits,
            "genetic_covariance": genetic,
            "residual_covariance": residual,
            "information": information,
            "fixed_effect_rank": rank_c,
        },
        dims={"genetic_covariance": ("trait_i", "trait_j"), "residual_covariance": ("trait_i", "trait_j"),
              "information": ("parameter_i", "parameter_j")},
        coords={"parameter_i": ["sigma_g2", "sigma_e2"], "parameter_j": ["sigma_g2", "sigma_e2"]},
        var_attrs=_SCORE_VAR_ATTRS, title="SCORE variance components",
        software="vigor.vc.score_variance_components",
        parameters={"residual_eigenvalue": float(residual_eigenvalue),
                    "full_covariance": bool(full_covariance), "tol": float(tol)},
    )
    return maybe_save(ds, save_path)


def multivariate_score_transform(
    Y, U, s, covar=None, residual_eigenvalue=0.0, tol=1e-10, save_path=None
):
    """Diagonalize SCORE genetic/residual trait covariance components.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    U : array-like
        Numerical operand in ``score_variance_components(Y, U, s, covar=covar,
        residual_eigenvalue=residual_eigenvalue, full_covariance=True, tol=tol)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``score_variance_components(Y, U, s, covar=covar,
        residual_eigenvalue=residual_eigenvalue, full_covariance=True, tol=tol)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``. Passed to
        ``score_variance_components`` as ``covar``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``score_variance_components`` as
        ``residual_eigenvalue``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``score_variance_components`` as ``tol``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``Y`` (sample, component; transformed phenotypes), ``transform``
        (trait, component), ``inverse_transform`` (component, trait),
        ``generalized_eigenvalues`` (component) and the SCORE ``genetic_covariance``,
        ``residual_covariance`` (trait_i, trait_j) and ``information``
        (parameter_i, parameter_j). Attrs: ``fits`` (per-component), ``original_fits``
        (per-trait), ``fixed_effect_rank`` plus title/software/parameters provenance.
    """
    score = score_variance_components(
        Y,
        U,
        s,
        covar=covar,
        residual_eigenvalue=residual_eigenvalue,
        full_covariance=True,
        tol=tol,
    )
    genetic = score["genetic_covariance"].values
    residual = score["residual_covariance"].values
    original_fits = score.attrs["fits"]
    scale = max(float(np.trace(residual)) / max(len(residual), 1), 1.0)
    residual_regularized = residual + np.eye(len(residual)) * (tol * scale)
    eigenvalues, transform = eigh(genetic, residual_regularized, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    inverse_transform = np.linalg.inv(transform)
    transformed = np.asarray(Y, dtype=np.float64) @ transform
    fits = [
        {
            "h2": float(value / (value + 1.0)),
            "yvar": float(value + 1.0),
            "sigma_g2": float(value),
            "sigma_e2": 1.0,
            "converged": True,
            "estimator": "score_multivariate_component",
        }
        for value in eigenvalues
    ]
    ds = result_dataset(
        {
            **{name: score[name] for name in score.data_vars},
            "fixed_effect_rank": score.attrs["fixed_effect_rank"],
            "Y": transformed,
            "transform": transform,
            "inverse_transform": inverse_transform,
            "generalized_eigenvalues": eigenvalues,
            "original_fits": original_fits,
            "fits": fits,
        },
        dims={"Y": ("sample", "component"), "transform": ("trait", "component"),
              "inverse_transform": ("component", "trait"), "generalized_eigenvalues": ("component",)},
        var_attrs={
            "Y": {"long_name": "phenotypes in the generalized-eigenvector basis", "description": "Y @ transform"},
            "transform": {"long_name": "trait-to-component transform", "description": "generalized eigenvectors of (genetic, residual)"},
            "inverse_transform": {"long_name": "component-to-trait inverse transform"},
            "generalized_eigenvalues": {"long_name": "generalized genetic/residual eigenvalues", "description": "component h2 = value / (value + 1)"},
        },
        title="Multivariate SCORE component transform",
        software="vigor.vc.multivariate_score_transform",
        parameters={"residual_eigenvalue": float(residual_eigenvalue), "tol": float(tol)},
    )
    return maybe_save(ds, save_path)


def whiten_y(
    y, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    """Return the whiten y result computed by ``_apply_eigen_operator``.

    Parameters
    ----------
    y : array-like
        Numerical operand in ``_apply_eigen_operator(y, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    U : array-like
        Numerical operand in ``_apply_eigen_operator(y, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``_apply_eigen_operator(y, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    yvar : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``_apply_eigen_operator`` as ``tol``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``_apply_eigen_operator`` as
        ``residual_eigenvalue``.

    Returns
    -------
    object
        Computed result from ``_apply_eigen_operator(y, U, s, h2, yvar, power=-0.5, tol=tol, residual_eigenvalue=residual_eigenvalue)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return _apply_eigen_operator(
        y,
        U,
        s,
        h2,
        yvar,
        power=-0.5,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def whiten_matrix(
    x, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    """Return the whiten matrix result computed by ``_apply_eigen_operator``.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``_apply_eigen_operator(x, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    U : array-like
        Numerical operand in ``_apply_eigen_operator(x, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``_apply_eigen_operator(x, U, s, h2, yvar, power=-0.5, tol=tol,
        residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    yvar : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``. Passed to ``_apply_eigen_operator`` as ``tol``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``_apply_eigen_operator`` as
        ``residual_eigenvalue``.

    Returns
    -------
    object
        Computed result from ``_apply_eigen_operator(x, U, s, h2, yvar, power=-0.5, tol=tol, residual_eigenvalue=residual_eigenvalue)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return _apply_eigen_operator(
        x,
        U,
        s,
        h2,
        yvar,
        power=-0.5,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def blup_resid(
    y, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    """GRAMMAR residual retained for compatibility.

    Parameters
    ----------
    y : array-like
        Numerical operand in ``y - y.mean()``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    U : array-like
        Numerical operand in ``centered - np.asarray(U) @ (weights * (np.asarray(U).T @
        centered))``. Shape must conform to that matrix/vector operation and the enclosing
        prepared state.

    s : array-like
        Numerical operand in ``h2 * s + 1.0 - h2``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction.

    yvar : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.

    Returns
    -------
    object
        Computed result from ``centered - np.asarray(U) @ (weights * (np.asarray(U).T @ centered))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    centered = y - y.mean()
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    denominator = h2 * s + 1.0 - h2
    weights = (h2 * (s - residual_eigenvalue)) / np.maximum(denominator, tol)
    return centered - np.asarray(U) @ (weights * (np.asarray(U).T @ centered))
