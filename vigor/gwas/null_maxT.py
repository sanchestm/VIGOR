"""Vectorized empirical maximum-score calibration."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm


def residualize(A, C, *, add_intercept=True):
    """Remove fixed-covariate projection from each simulated/null response vector.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``A - C @ (np.linalg.pinv(C) @ A)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    C : array-like
        Numerical operand in ``A - C @ (np.linalg.pinv(C) @ A)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    add_intercept : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``A - C @ (np.linalg.pinv(C) @ A)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if C is None:
        C = np.ones((len(A),1)) if add_intercept else None
    if C is None: return A
    C = np.asarray(C, float)
    if C.ndim == 1: C = C[:, None]
    if add_intercept: C = np.column_stack([np.ones(len(C)), C])
    return A - C @ (np.linalg.pinv(C) @ A)


def prepare_score_genotypes(X, covariates=None, grm=None, variance_components=None):
    """Return standardized genotypes in OLS or fixed LMM whitening geometry.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    variance_components : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``(X, good, C)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X = np.asarray(X, float)
    C, invsqrt = _score_geometry(len(X), covariates, grm, variance_components)
    X, good = _standardized_score_block(X, C, invsqrt)
    return X, good, C


def _score_geometry(n_samples, covariates=None, grm=None, variance_components=None):
    """Construct the fixed null-model geometry without touching genotypes.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    variance_components : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    C = np.empty((n_samples, 0)) if covariates is None else np.asarray(covariates, float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2 or C.shape[0] != n_samples:
        raise ValueError("covariates shape mismatch")
    C = np.column_stack([np.ones(n_samples), C])
    invsqrt = None
    if grm is not None:
        K = np.asarray(grm, float)
        if K.shape != (n_samples, n_samples):
            raise ValueError("grm shape mismatch")
        if variance_components is None:
            raise ValueError("variance_components=(sg,se) is required when grm is supplied; VIGOR no longer assumes h2=0.5")
        sg, se = variance_components
        if sg < 0 or se <= 0:
            raise ValueError("variance components must have sg >= 0 and se > 0")
        w, U = eigh((K + K.T) * 0.5)
        variance = sg * w + se
        if np.any(variance <= 0):
            raise ValueError("variance components do not define a positive covariance")
        invsqrt = (U * (1.0 / np.sqrt(variance))) @ U.T
        C = invsqrt @ C
    return C, invsqrt


def _standardized_score_block(X, C, invsqrt):
    """Whiten, residualize, and normalize one bounded variant block.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    C : array-like
        Numerical operand in ``residualize(X, C, add_intercept=False)``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state.

    invsqrt : array-like or numeric scalar
        Input for the operation ``invsqrt @ X``. See
        ``vigor.gwas.null_maxT._standardized_score_block`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    X = np.asarray(X, float)
    means = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, means)
    if invsqrt is not None:
        X = invsqrt @ X
    X = residualize(X, C, add_intercept=False)
    norms = np.sqrt(np.sum(X * X, axis=0))
    good = np.isfinite(norms) & (norms > np.finfo(float).eps)
    X[:, good] /= norms[good]
    X[:, ~good] = 0.0
    return X, good




def _score_source_shape(X):
    """Evaluate source shape.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.
    """
    if hasattr(X, "iter_blocks") and hasattr(X, "sample_ids"):
        return len(X.sample_ids), len(X.variant_metadata)
    if getattr(X, "ndim", None) == 2:
        return X.shape
    raise ValueError("X must be sample x variant or a VIGOR genotype source")


def _iter_score_blocks(X, block_size):
    """Yield bounded sample-by-variant dosage blocks from arrays or sources.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Passed to ``X.iter_blocks``
        as ``block_size``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    if hasattr(X, "iter_blocks") and hasattr(X, "sample_ids"):
        for block in X.iter_blocks(representation="dosage", block_size=int(block_size)):
            data = getattr(block, "data", block[1] if isinstance(block, tuple) else block)
            yield np.asarray(data, float)
        return
    n_variants = X.shape[1]
    for st in range(0, n_variants, int(block_size)):
        yield np.asarray(X[:, st:st + int(block_size)], float)


def adaptive_maxT(X, *, alpha=(0.05,), covariates=None, grm=None,
                  variance_components=None, min_replicates=200,
                  max_replicates=5000, batch_replicates=100,
                  target_relative_se=0.03, random_state=42,
                  variant_block_size='auto', maxmem=None, memory_fraction=None):
    """Simulate null residuals and retain maximum absolute score per replicate.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``(0.05,)``.

    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    variance_components : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    min_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``200``.

    max_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``5000``.

    batch_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``100``.

    target_relative_se : float
        Positive target relative Monte Carlo standard error for the adaptive null-threshold
        estimate. Smaller values request more simulation precision. Default: ``0.03``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``42``.

    variant_block_size : int or 'auto'
        Variants per I/O/scoring/LD work block; positive counts bound loaded genotype columns
        and intermediate matrix size. Default: ``'auto'`` (sized from the memory budget;
        integers are ceilings).

    maxmem : int, float, unit string or None
        Working-memory budget; see ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``(pthr, maxima, zthr, threshold_se, threshold_ci, good_count)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    n_samples, n_variants = _score_source_shape(X)
    transformed_covariates, invsqrt = _score_geometry(
        n_samples, covariates, grm, variance_components)
    from ..resources import resolve_variant_chunk
    # float64 per call: source block, imputed copy, invsqrt@X, residualized X, X*X (+ mask);
    # per variant: z and |z| over one replicate batch
    bs = resolve_variant_chunk(variant_block_size, n_samples, bytes_per_call=41,
                               bytes_per_variant=16 * max(1, int(batch_replicates)),
                               maxmem=maxmem, memory_fraction=memory_fraction, n_variants=n_variants)
    # Standardized blocks are deliberately not retained: source-backed and
    # memmap inputs therefore use O(N * block_size + N^2) resident memory.
    good_count = 0
    for Xblock in _iter_score_blocks(X, bs):
        _, good = _standardized_score_block(Xblock, transformed_covariates, invsqrt)
        good_count += int(good.sum())
    if good_count == 0:
        raise ValueError("no polymorphic variants remain after null-model projection")
    rng = np.random.default_rng(random_state); maxima = []
    alphas = np.asarray(alpha, float)
    if np.any((alphas <= 0) | (alphas >= 1)): raise ValueError("alpha must lie in (0,1)")
    threshold_se = {float(a): np.inf for a in alphas}; threshold_ci = {}
    while len(maxima) < int(max_replicates):
        b = min(int(batch_replicates), int(max_replicates) - len(maxima))
        # In whitened space null residuals are iid N(0,1); project the same
        # (possibly whitened) covariates used for genotype residualization.
        E = residualize(rng.standard_normal((n_samples, b)), transformed_covariates, add_intercept=False)
        mx = np.zeros(b)
        for Xblock in _iter_score_blocks(X, bs):
            Xs, _ = _standardized_score_block(Xblock, transformed_covariates, invsqrt)
            z = Xs.T @ E
            mx = np.maximum(mx, np.max(np.abs(z), axis=0))
        maxima.extend(mx.tolist())
        if len(maxima) >= int(min_replicates):
            arr = np.asarray(maxima); stop = True
            for a in alphas:
                q = np.quantile(arr, 1-a)
                # Quantile bootstrap avoids a fragile density estimate.
                brng=np.random.default_rng(int(random_state)+len(arr)*1009+int(a*1e6))
                qb = np.array([np.quantile(brng.choice(arr, len(arr), replace=True), 1-a) for _ in range(200)])
                se = float(qb.std(ddof=1)); threshold_se[float(a)] = se
                threshold_ci[float(a)] = tuple(np.quantile(qb,[.025,.975]).astype(float))
                stop &= se / max(abs(q), 1e-12) <= float(target_relative_se)
            if stop: break
    maxima = np.asarray(maxima)
    zthr = {float(a): float(np.quantile(maxima, 1-a)) for a in alphas}
    # two-sided p corresponding to max |Z|
    pthr = {a: float(2 * norm.sf(z)) for a, z in zthr.items()}
    return pthr, maxima, zthr, threshold_se, threshold_ci, good_count
