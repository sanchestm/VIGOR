"""Best linear unbiased predictions from VIGOR covariance models.

The primary ``run_blup`` entry point fits a full-genome additive GRM model and
writes one prediction row per animal and quantitative trait. A lower-level
component function also supports additive and dominance BLUPs when both GRMs
and their estimated variance components are supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from threadpoolctl import threadpool_limits

from .grm import compute_grm
from .io import PlinkReader
from .pipeline import _full_rank_design, _load_pheno, _prepare_covariates
from .vc import estimate_h2_many


def _phenotype_matrix(phenotypes, sample_ids=None, trait_names=None):
    """Coerce phenotype inputs into an animal-aligned numerical response matrix and labels.

    Parameters
    ----------
    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.
    """
    if isinstance(phenotypes, pd.DataFrame):
        frame = phenotypes.copy()
        frame.index = frame.index.astype(str)
        if frame.index.has_duplicates:
            dup = frame.index[frame.index.duplicated()][0]
            raise ValueError(f"duplicate phenotype sample ID: {dup}")
        if sample_ids is None:
            sample_ids = frame.index.tolist()
        else:
            sample_ids = [str(x) for x in sample_ids]
            missing = pd.Index(sample_ids).difference(frame.index)
            if len(missing):
                raise ValueError(f"phenotypes missing requested sample IDs: {list(missing[:5])}")
            frame = frame.reindex(sample_ids)
        if trait_names is None:
            trait_names = frame.columns.astype(str).tolist()
        else:
            trait_names = [str(x) for x in trait_names]
            missing_traits = pd.Index(trait_names).difference(frame.columns.astype(str))
            if len(missing_traits):
                raise ValueError(f"phenotypes missing requested traits: {list(missing_traits[:5])}")
            frame.columns = frame.columns.astype(str)
            frame = frame.loc[:, trait_names]
        values = frame.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(phenotypes, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if sample_ids is None:
            sample_ids = [str(index) for index in range(values.shape[0])]
        if trait_names is None:
            trait_names = [f"trait_{index + 1}" for index in range(values.shape[1])]
    if values.ndim != 2:
        raise ValueError("phenotypes must be a vector or two-dimensional matrix")
    if len(sample_ids) != values.shape[0]:
        raise ValueError("sample_ids does not match the phenotype rows")
    if len(trait_names) != values.shape[1]:
        raise ValueError("trait_names does not match the phenotype columns")
    if not np.isfinite(values).all():
        raise ValueError("phenotypes must be complete and finite")
    return values, [str(value) for value in sample_ids], [str(value) for value in trait_names]


def _fixed_design(covar, n_samples, sample_ids=None):
    """Construct and align the fixed-effect covariate design to the requested animals.

    Parameters
    ----------
    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.
    """
    if covar is None:
        return np.ones((n_samples, 1), dtype=np.float64)
    if isinstance(covar, (pd.DataFrame, pd.Series)) and sample_ids is not None:
        obj = covar.copy()
        obj.index = obj.index.astype(str)
        if obj.index.has_duplicates:
            raise ValueError("covariate index contains duplicate sample IDs")
        missing = pd.Index([str(x) for x in sample_ids]).difference(obj.index)
        if len(missing):
            raise ValueError(f"covariates missing requested sample IDs: {list(missing[:5])}")
        obj = obj.reindex([str(x) for x in sample_ids])
        values = obj.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(covar, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != n_samples:
        raise ValueError("covar does not match the phenotype rows")
    if not np.isfinite(values).all():
        raise ValueError("covar must be complete and finite")
    return values


def _variance_fits_by_trait(variance_fits, trait_names):
    """Align fitted variance-component results to the requested trait order.

    Parameters
    ----------
    variance_fits : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.
    """
    if isinstance(variance_fits, dict) and all(
        trait in variance_fits for trait in trait_names
    ):
        fits = [variance_fits[trait] for trait in trait_names]
    elif isinstance(variance_fits, dict) and "h2" in variance_fits:
        if len(trait_names) != 1:
            raise ValueError("one variance-component fit is required per trait")
        fits = [variance_fits]
    else:
        fits = list(variance_fits)
    if len(fits) != len(trait_names):
        raise ValueError("one variance-component fit is required per trait")
    for fit in fits:
        if "h2" not in fit or "yvar" not in fit:
            raise ValueError("each variance-component fit must contain h2 and yvar")
    return fits


def _apply_retained_operator(x, U, retained_values, bulk_value):
    """Apply retained operator.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``U.T @ x``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    U : array-like
        Numerical operand in ``U.T @ x``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    retained_values : one-dimensional array-like
        Eigenvalues associated with the columns of the retained eigenvector basis. Length must
        equal the basis rank; covariance uses nonnegative eigenvalues.

    bulk_value : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution.
    """
    projection = U.T @ x
    return bulk_value * x + U @ ((retained_values - bulk_value) * projection)


def _assemble_output(
    Y,
    sample_ids,
    trait_names,
    fixed,
    additive,
    dominance,
    fits,
    model,
):
    """Return the assemble output result computed by ``pd.DataFrame``.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    fixed : object (see the cited input contract)
        Input for the operation ``fixed + total_genetic``. See ``vigor.blup._assemble_output``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    additive : array-like or numeric scalar
        Input for the operation ``additive + np.nan_to_num(dominance, nan=0.0)``. See
        ``vigor.blup._assemble_output`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.

    dominance : array-like or numeric scalar
        Input for the operation ``additive + np.nan_to_num(dominance, nan=0.0)``. See
        ``vigor.blup._assemble_output`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.

    fits : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here.
    """
    n_samples, n_traits = Y.shape
    total_genetic = additive + np.nan_to_num(dominance, nan=0.0)
    fitted = fixed + total_genetic
    residual = Y - fitted
    h2_values = []
    sigma_a2_values = []
    sigma_e2_values = []
    for fit in fits:
        if "h2" in fit:
            h2_value = float(fit["h2"])
        else:
            total = (
                float(fit["sigma_a2"])
                + float(fit.get("sigma_d2", 0.0))
                + float(fit["sigma_e2"])
            )
            h2_value = float(fit["sigma_a2"]) / max(total, 1e-30)
        h2_values.append(h2_value)
        sigma_a2_values.append(
            float(fit["sigma_a2"])
            if "sigma_a2" in fit
            else h2_value * float(fit["yvar"])
        )
        sigma_e2_values.append(
            float(fit["sigma_e2"])
            if "sigma_e2" in fit
            else (1.0 - h2_value) * float(fit["yvar"])
        )
    h2 = np.asarray(h2_values, dtype=float)
    sigma_a2 = np.asarray(sigma_a2_values, dtype=float)
    sigma_d2 = np.asarray([fit.get("sigma_d2", np.nan) for fit in fits], dtype=float)
    sigma_e2 = np.asarray(sigma_e2_values, dtype=float)
    return pd.DataFrame(
        {
            "iid": np.repeat(np.asarray(sample_ids, dtype=object), n_traits),
            "trait": np.tile(np.asarray(trait_names, dtype=object), n_samples),
            "phenotype": Y.ravel(order="C"),
            "fixed_prediction": fixed.ravel(order="C"),
            "additive_blup": additive.ravel(order="C"),
            "dominance_blup": dominance.ravel(order="C"),
            "total_genetic_value": total_genetic.ravel(order="C"),
            "fitted_value": fitted.ravel(order="C"),
            "residual": residual.ravel(order="C"),
            "h2_additive": np.tile(h2, n_samples),
            "sigma_additive2": np.tile(sigma_a2, n_samples),
            "sigma_dominance2": np.tile(sigma_d2, n_samples),
            "sigma_residual2": np.tile(sigma_e2, n_samples),
            "blup_model": model,
        }
    )


def calculate_blups(
    phenotypes,
    U,
    s,
    variance_fits,
    covar=None,
    sample_ids=None,
    trait_names=None,
    residual_eigenvalue=0.0,
    tol=1e-10,
):
    """Calculate additive BLUPs from a fitted additive GRM eigensystem.

    Returns
    -------
    pandas.DataFrame
        One row per animal and trait. ``dominance_blup`` is missing because the
        current VIGOR variance model contains no dominance GRM.

    Parameters
    ----------
    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    U : array-like
        Numerical operand in ``U.shape != (len(Y), len(s))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``h2 * s + 1.0 - h2``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    variance_fits : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.
    """
    Y, ids, names = _phenotype_matrix(phenotypes, sample_ids, trait_names)
    C = _fixed_design(covar, len(Y), ids)
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    if U.shape != (len(Y), len(s)):
        raise ValueError("U, s, and phenotype dimensions do not agree")
    fits = _variance_fits_by_trait(variance_fits, names)

    fixed = np.empty_like(Y)
    additive = np.empty_like(Y)
    coefficients = []
    for index, fit in enumerate(fits):
        h2 = float(fit["h2"])
        if not 0.0 <= h2 <= 1.0:
            raise ValueError("h2 must be between zero and one")
        retained_covariance = np.maximum(h2 * s + 1.0 - h2, tol)
        bulk_covariance = max(
            h2 * float(residual_eigenvalue) + 1.0 - h2,
            tol,
        )
        inverse_retained = 1.0 / retained_covariance
        inverse_bulk = 1.0 / bulk_covariance
        inverse_y = _apply_retained_operator(
            Y[:, index], U, inverse_retained, inverse_bulk
        )
        inverse_c = np.column_stack(
            [
                _apply_retained_operator(C[:, column], U, inverse_retained, inverse_bulk)
                for column in range(C.shape[1])
            ]
        )
        information = C.T @ inverse_c
        beta = np.linalg.pinv(information) @ (C.T @ inverse_y)
        fixed[:, index] = C @ beta
        conditional = Y[:, index] - fixed[:, index]

        retained_shrinkage = h2 * s / retained_covariance
        bulk_shrinkage = h2 * float(residual_eigenvalue) / bulk_covariance
        additive[:, index] = _apply_retained_operator(
            conditional,
            U,
            retained_shrinkage,
            bulk_shrinkage,
        )
        coefficients.append(beta.tolist())

    dominance = np.full_like(additive, np.nan)
    result = _assemble_output(
        Y,
        ids,
        names,
        fixed,
        additive,
        dominance,
        fits,
        model="additive_grm",
    )
    result.attrs["fixed_effect_coefficients"] = dict(zip(names, coefficients))
    return result


def calculate_component_blups(
    phenotypes,
    additive_grm,
    variance_components,
    dominance_grm=None,
    covar=None,
    sample_ids=None,
    trait_names=None,
    tol=1e-10,
):
    """Calculate exact additive and optional dominance BLUPs from dense GRMs.

    The caller must supply externally estimated ``sigma_a2``, ``sigma_e2``,
    and, when ``dominance_grm`` is present, ``sigma_d2`` for every trait.
    This function calculates predictions only; it does not estimate the
    multi-GRM variance components.

    Parameters
    ----------
    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    additive_grm : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine.

    variance_components : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    dominance_grm : array-like, operator or mapping
        Animal kernel(s) used for the stated background/genetic component. Each matrix is square
        on the same aligned animal axis; operator forms must implement the
        multiplication/diagonal interface used by this routine. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Y, ids, names = _phenotype_matrix(phenotypes, sample_ids, trait_names)
    C = _fixed_design(covar, len(Y), ids)
    def _align_kernel(kernel, label):
        """Align kernel.

        Parameters
        ----------
        kernel : array-like, operator or mapping
            Animal kernel(s) used for the stated background/genetic component. Each matrix is square
            on the same aligned animal axis; operator forms must implement the
            multiplication/diagonal interface used by this routine.

        label : str, index or sequence
            Axis/column/feature selector used to identify the input/output quantities in this
            operation. Named selectors must exist in the labeled input; integer selectors are valid
            zero-based positions on that axis.
        """
        if isinstance(kernel, pd.DataFrame):
            k = kernel.copy()
            k.index = k.index.astype(str); k.columns = k.columns.astype(str)
            if k.index.has_duplicates or k.columns.has_duplicates:
                raise ValueError(f"{label} contains duplicate sample IDs")
            want = pd.Index(ids)
            missing = want.difference(k.index).union(want.difference(k.columns))
            if len(missing):
                raise ValueError(f"{label} missing phenotype sample IDs: {list(missing[:5])}")
            return k.reindex(index=ids, columns=ids).to_numpy(dtype=np.float64)
        return np.asarray(kernel, dtype=np.float64)

    K_add = _align_kernel(additive_grm, "additive_grm")
    if K_add.shape != (len(Y), len(Y)):
        raise ValueError("additive_grm must be square with one row per sample")
    K_dom = None if dominance_grm is None else _align_kernel(dominance_grm, "dominance_grm")
    if K_dom is not None and K_dom.shape != K_add.shape:
        raise ValueError("dominance_grm must have the same shape as additive_grm")

    if isinstance(variance_components, dict) and all(
        trait in variance_components for trait in names
    ):
        fits = [variance_components[trait] for trait in names]
    elif isinstance(variance_components, dict) and "sigma_a2" in variance_components:
        if len(names) != 1:
            raise ValueError("one variance-component mapping is required per trait")
        fits = [variance_components]
    else:
        fits = list(variance_components)
    if len(fits) != len(names):
        raise ValueError("one variance-component mapping is required per trait")

    fixed = np.empty_like(Y)
    additive = np.empty_like(Y)
    dominance = np.full_like(Y, np.nan)
    coefficients = []
    identity = np.eye(len(Y), dtype=np.float64)
    for index, fit in enumerate(fits):
        sigma_a2 = float(fit["sigma_a2"])
        sigma_e2 = float(fit["sigma_e2"])
        sigma_d2 = float(fit.get("sigma_d2", 0.0))
        if min(sigma_a2, sigma_d2, sigma_e2) < 0.0 or sigma_e2 <= tol:
            raise ValueError("variance components must be nonnegative with sigma_e2 > 0")
        if K_dom is None and sigma_d2 != 0.0:
            raise ValueError("dominance_grm is required when sigma_d2 is nonzero")

        covariance = sigma_a2 * K_add + sigma_e2 * identity
        if K_dom is not None:
            covariance = covariance + sigma_d2 * K_dom
        factor = cho_factor(covariance, lower=True, check_finite=False)
        inverse_y = cho_solve(factor, Y[:, index], check_finite=False)
        inverse_c = cho_solve(factor, C, check_finite=False)
        beta = np.linalg.pinv(C.T @ inverse_c) @ (C.T @ inverse_y)
        fixed[:, index] = C @ beta
        conditional = Y[:, index] - fixed[:, index]
        alpha = cho_solve(factor, conditional, check_finite=False)
        additive[:, index] = sigma_a2 * (K_add @ alpha)
        if K_dom is not None:
            dominance[:, index] = sigma_d2 * (K_dom @ alpha)
        coefficients.append(beta.tolist())

    normalized_fits = [
        {
            **fit,
            "h2": float(fit["sigma_a2"])
            / max(
                float(fit["sigma_a2"])
                + float(fit.get("sigma_d2", 0.0))
                + float(fit["sigma_e2"]),
                tol,
            ),
        }
        for fit in fits
    ]
    result = _assemble_output(
        Y,
        ids,
        names,
        fixed,
        additive,
        dominance,
        normalized_fits,
        model="additive_dominance_grm" if K_dom is not None else "additive_grm_dense",
    )
    result.attrs["fixed_effect_coefficients"] = dict(zip(names, coefficients))
    return result


def write_blups(blups, output):
    """Write a BLUP table as Parquet, CSV, or tab-separated text.

    Parameters
    ----------
    blups : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    output : str or os.PathLike
        Input for the operation ``Path(output)``. See ``vigor.blup.write_blups`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.

    Returns
    -------
    object
        Computed result from ``path``; see the summary and the called implementation for its labeled/numerical representation.
    """
    path = Path(output)
    if not path.suffix:
        path = Path(f"{path}.blup.parquet")
    suffix = path.suffix.lower()
    if suffix not in {".parquet", ".csv", ".tsv", ".txt"}:
        raise ValueError("BLUP output must end in .parquet, .csv, .tsv, or .txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial{suffix}")
    if suffix == ".parquet":
        blups.to_parquet(temporary, index=False)
    else:
        blups.to_csv(temporary, index=False, sep="\t" if suffix in {".tsv", ".txt"} else ",")
    os.replace(temporary, path)
    return path


def run_blup(
    bfile,
    pheno,
    traits=None,
    covar=None,
    grm_bfile=None,
    method="REML",
    block_size='auto',
    out=None,
    n_jobs=1,
    verbose=True,
):
    """Fit full-genome additive GRM models and calculate per-animal BLUPs.

    Traits sharing a complete-case mask reuse one GRM eigendecomposition.
    The output is additive because VIGOR's fitted background covariance model
    currently contains an additive GRM only.

    Parameters
    ----------
    bfile : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    pheno : path-like or pandas.DataFrame
        Phenotype table, with unique animal IDs and numeric trait columns. File inputs are
        parsed/aligned by _load_pheno; NaN represents unobserved trait values.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``.

    grm_bfile : object (see the cited input contract)
        Input for the operation ``PlinkReader(grm_bfile)``. See ``vigor.blup.run_blup`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'REML'``. Source-recognized
        literal values: ``ML``. These select the corresponding branches below; comparison
        evidence alone does not establish an exhaustive enum. Passed to ``estimate_h2_many`` as
        ``method``.

    block_size : int or 'auto'
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings). Passed to ``compute_grm`` as
        ``block_size``.

    out : object (see the cited input contract)
        Input for the operation ``write_blups(result, out)``. See ``vigor.blup.run_blup`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    n_jobs : int or str
        Worker-process count for bootstrap; a positive integer fixes concurrency, while 'auto'
        resolves a bounded process count from CPU availability and the replicate workload.
        Default: ``1``. Passed to ``threadpool_limits`` as ``limits``.

    verbose : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    reader = PlinkReader(bfile)
    grm_reader = reader if grm_bfile is None else PlinkReader(grm_bfile)
    try:
        phenotype = _load_pheno(pheno)
        phenotype.index = phenotype.index.astype(str)
        if phenotype.index.has_duplicates:
            raise ValueError("phenotype IIDs must be unique")
        if traits is None:
            traits = [column for column in phenotype.columns if column != "FID"]
        traits = list(traits)
        method = method.upper()
        if method not in {"REML", "ML"}:
            raise ValueError("method must be REML or ML")

        fam_iid = reader.fam["iid"].astype(str).to_numpy()
        common = [iid for iid in fam_iid if iid in set(phenotype.index)]
        sub = phenotype.loc[common, traits].apply(pd.to_numeric, errors="coerce")
        binary = [trait for trait in traits if sub[trait].dropna().nunique() == 2]
        if binary:
            raise ValueError(
                f"binary phenotypes are not supported by the quantitative-trait model: {binary}"
            )
        covariate_frame = _prepare_covariates(covar, sub.index)
        covariate_complete = np.isfinite(
            covariate_frame.to_numpy(dtype=float)
        ).all(axis=1)
        groups = {}
        for trait in traits:
            mask = sub[trait].notna().to_numpy() & covariate_complete
            groups.setdefault(mask.tobytes(), {"mask": mask, "traits": []})[
                "traits"
            ].append(trait)

        fam_position = pd.Index(fam_iid)
        grm_position = pd.Index(grm_reader.fam["iid"].astype(str))
        frames = []
        metadata_groups = []
        with threadpool_limits(limits=None if n_jobs == -1 else n_jobs):
            for group_index, group in enumerate(groups.values()):
                mask = group["mask"]
                group_traits = group["traits"]
                analysis_ids = sub.index[mask]
                if len(analysis_ids) < 5:
                    raise ValueError(
                        f"fewer than five complete samples for traits {group_traits}"
                    )
                sample_index = fam_position.get_indexer(analysis_ids)
                grm_sample_index = grm_position.get_indexer(analysis_ids)
                if np.any(sample_index < 0) or np.any(grm_sample_index < 0):
                    raise ValueError("not all phenotype samples are present in the genotype files")
                Y = sub.loc[analysis_ids, group_traits].to_numpy(dtype=np.float64)
                covariate_values = covariate_frame.loc[analysis_ids].to_numpy(dtype=float)
                C = _full_rank_design(covariate_values)

                if verbose:
                    print(
                        f"BLUP group {group_index + 1}: {len(Y)} samples x "
                        f"{len(group_traits)} traits"
                    )
                grm = compute_grm(
                    grm_reader,
                    sample_index=grm_sample_index,
                    block_size=block_size,
                    compute_dtype="float64",
                )
                s, U = eigh(grm, check_finite=False, overwrite_a=True)
                order = np.argsort(s)[::-1]
                s = np.maximum(s[order], 0.0)
                U = U[:, order]
                fits = estimate_h2_many(Y, U, s, covar=C, method=method)
                frame = calculate_blups(
                    Y,
                    U,
                    s,
                    fits,
                    covar=C,
                    sample_ids=analysis_ids,
                    trait_names=group_traits,
                )
                frames.append(frame)
                metadata_groups.append(
                    {
                        "group": group_index,
                        "traits": group_traits,
                        "n_samples": len(Y),
                        "fixed_effect_rank": int(C.shape[1]),
                        "variance_components": dict(zip(group_traits, fits)),
                        "fixed_effect_coefficients": frame.attrs[
                            "fixed_effect_coefficients"
                        ],
                    }
                )
        result = pd.concat(frames, ignore_index=True)
        metadata = {
            "model": "full_genome_additive_grm_blup",
            "bfile": str(Path(bfile).resolve()),
            "grm_bfile": (
                str(Path(grm_bfile).resolve()) if grm_bfile is not None else None
            ),
            "traits": traits,
            "method": method,
            "groups": metadata_groups,
            "dominance_blup_available": False,
            "dominance_note": (
                "A dominance BLUP requires a dominance GRM and an estimated "
                "dominance variance component."
            ),
        }
        result.attrs["blup_metadata"] = metadata
        if out is not None:
            output_path = write_blups(result, out)
            metadata_path = output_path.with_suffix(".metadata.json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            result.attrs["output_path"] = str(output_path)
            result.attrs["metadata_path"] = str(metadata_path)
        return result
    finally:
        if grm_reader is not reader:
            grm_reader.close()
        reader.close()


def _names(value):
    """Parse a comma-separated identifier list, trimming empty entries.

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation ``value.split(',')``. See ``vigor.blup._names`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.
    """
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def main(argv=None):
    """Parse command-line options and execute the requested VIGOR workflow.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line argument tokens. None reads the process argument vector; accepted switches
        and choices are defined by the corresponding parser. Default: ``None``.
    """
    parser = argparse.ArgumentParser(
        prog="vigor-blup",
        description="Calculate full-genome additive GRM BLUPs for quantitative traits",
    )
    from . import __version__
    parser.add_argument("--version", action="version", version=f"VIGOR {__version__}")
    parser.add_argument("--bfile", required=True, help="PLINK 1 prefix")
    parser.add_argument("--grm-bfile", help="Optional separate relationship-panel prefix")
    parser.add_argument("--pheno", required=True, type=Path, help="Tab-separated phenotype table")
    parser.add_argument("--id-column", default="iid")
    parser.add_argument("--traits", required=True, help="Comma-separated quantitative traits")
    parser.add_argument("--covariates", help="Comma-separated numeric covariates")
    parser.add_argument("--categorical-covariates", help="Comma-separated categorical covariates")
    parser.add_argument("--method", choices=["REML", "ML"], default="REML")
    parser.add_argument("--block-size", type=lambda v: v if str(v).lower() == "auto" else int(v), default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", required=True, help="Output prefix or table path")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.pheno, sep="\t", dtype={args.id_column: "string"})
    frame[args.id_column] = frame[args.id_column].str.strip()
    frame = frame.set_index(args.id_column)
    traits = _names(args.traits)
    numeric = _names(args.covariates)
    categorical = _names(args.categorical_covariates)
    requested = traits + numeric + categorical
    missing = sorted(set(requested) - set(frame.columns))
    if missing:
        raise ValueError(f"columns missing from phenotype table: {missing}")
    if numeric:
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    covar = frame[numeric + categorical] if numeric or categorical else None
    result = run_blup(
        args.bfile,
        frame[traits],
        traits=traits,
        covar=covar,
        grm_bfile=args.grm_bfile,
        method=args.method,
        block_size=args.block_size,
        out=args.out,
        n_jobs=args.n_jobs,
    )
    print(f"Wrote {len(result)} BLUP rows to {result.attrs['output_path']}")


if __name__ == "__main__":
    main()


def genomic_pcs_from_grm(grm, n_components=20, *, fit_mask=None, sample_ids=None,
                          center_scores=True, eigenvalue_tol=1e-10):
    """Return labeled genomic PC scores from a GRM, with leakage-safe Nyström projection.

    When ``fit_mask`` is supplied, the eigensystem is estimated on the training
    animals only. Scores for all animals are then obtained from ``K[:, train]``
    by Nyström projection, so held-out animals do not influence the axes.

    Parameters
    ----------
    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``20``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    center_scores : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    eigenvalue_tol : float
        Input for the operation ``float(eigenvalue_tol) * max(float(w[0]) if len(w) else 1.0,
        1.0)``. See ``vigor.blup.genomic_pcs_from_grm`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``1e-10``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(grm, pd.DataFrame):
        if not grm.index.equals(grm.columns):
            raise ValueError("labeled GRM must have identical row/column IDs in the same order")
        ids = grm.index.astype(str)
        K = grm.to_numpy(dtype=np.float64)
    else:
        K = np.asarray(grm, dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("grm must be square")
        ids = pd.Index([str(i) for i in range(K.shape[0])] if sample_ids is None else list(map(str, sample_ids)))
        if len(ids) != K.shape[0]:
            raise ValueError("sample_ids length does not match GRM")
    n = K.shape[0]
    fit = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if fit.shape != (n,) or fit.sum() < 2:
        raise ValueError("fit_mask must select at least two GRM rows")
    Ktt = (K[np.ix_(fit, fit)] + K[np.ix_(fit, fit)].T) * 0.5
    w, U = np.linalg.eigh(Ktt)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    keep = w > float(eigenvalue_tol) * max(float(w[0]) if len(w) else 1.0, 1.0)
    w, U = w[keep], U[:, keep]
    k = min(int(n_components), len(w))
    if k < 1:
        raise ValueError("GRM has no retained positive eigenvalues")
    w, U = w[:k], U[:, :k]
    # Training scores are U sqrt(lambda). Nyström scores for all animals are
    # K[:,train] U / sqrt(lambda), which reproduces those scores on train.
    scores = K[:, fit] @ (U / np.sqrt(w)[None, :])
    if center_scores:
        scores -= scores[fit].mean(axis=0, keepdims=True)
    cols = [f"GRM_PC{i+1}" for i in range(k)]
    out = pd.DataFrame(scores, index=ids, columns=cols)
    out.attrs.update({"fit_mask_used": fit_mask is not None, "eigenvalues": w.tolist(),
                      "projection": "nystrom" if fit_mask is not None else "full_eigendecomposition"})
    return out


def crossfit_nongenetic_predictions(phenotypes, covariates, *, n_splits=5,
                                     groups=None, random_state=0, ridge=1e-6, save_path=None):
    """Cross-fit covariate-only phenotype predictions without using genomic data.

    This helper is intended for leakage-safe benchmarking of non-genetic
    prediction. Every returned row is predicted by a model that did not use
    that row's phenotype. Numeric missing covariates are filled from training
    fold means; categorical covariates are one-hot encoded before splitting.

    Parameters
    ----------
    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``5``. Passed to ``KFold`` as ``n_splits``.

    groups : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.

    ridge : float or array-like
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``1e-06``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``prediction`` (sample, trait; covariate-only cross-fitted prediction)
        and ``fold`` (sample; validation fold index, -1 when unassigned). ``attrs`` hold
        ``method`` and ``n_splits`` (folds actually used).
    """
    from sklearn.model_selection import KFold, GroupKFold
    import xarray as xr
    from .results import maybe_save, result_dataset
    if isinstance(phenotypes, xr.DataArray):
        phenotypes = phenotypes.to_pandas()
    if isinstance(covariates, xr.DataArray):
        covariates = covariates.to_pandas()
    Y = phenotypes.copy() if isinstance(phenotypes, pd.DataFrame) else pd.DataFrame(phenotypes)
    if not Y.index.is_unique:
        raise ValueError("phenotype sample IDs must be unique")
    C = covariates.copy() if isinstance(covariates, pd.DataFrame) else pd.DataFrame(covariates, index=Y.index)
    if isinstance(covariates, pd.DataFrame):
        C.index = C.index.astype(str); Y.index = Y.index.astype(str)
        if not C.index.is_unique:
            raise ValueError("covariate sample IDs must be unique")
        missing = Y.index.difference(C.index)
        if len(missing):
            raise ValueError(f"covariates missing phenotype sample IDs: {list(missing[:5])}")
        C = C.reindex(Y.index)
    C = pd.get_dummies(C, dummy_na=True, drop_first=False).apply(pd.to_numeric, errors="coerce")
    X = C.to_numpy(dtype=np.float64)
    n = len(Y); pred = np.full(Y.shape, np.nan, dtype=np.float64); fold_id = np.full(n, -1, dtype=int)
    if groups is None:
        splitter = KFold(n_splits=min(int(n_splits), n), shuffle=True, random_state=random_state)
        splits = splitter.split(np.arange(n))
    else:
        g = np.asarray(groups)
        if len(g) != n: raise ValueError("groups length must equal phenotype rows")
        ns = min(int(n_splits), len(np.unique(g)))
        if ns < 2: raise ValueError("groups must contain at least two groups")
        splits = GroupKFold(ns).split(np.arange(n), groups=g)
    for f, (tr, va) in enumerate(splits):
        fold_id[va] = f
        mu = np.nanmean(X[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], mu); Xva = np.where(np.isfinite(X[va]), X[va], mu)
        Xtr = np.column_stack([np.ones(len(tr)), Xtr]); Xva = np.column_stack([np.ones(len(va)), Xva])
        P = np.eye(Xtr.shape[1]) * float(ridge); P[0, 0] = 0.0
        for j in range(Y.shape[1]):
            y = pd.to_numeric(Y.iloc[tr, j], errors="coerce").to_numpy(float)
            ok = np.isfinite(y)
            if ok.sum() < max(3, min(Xtr.shape[1] + 1, len(tr))):
                continue
            beta = np.linalg.pinv(Xtr[ok].T @ Xtr[ok] + P) @ (Xtr[ok].T @ y[ok])
            pred[va, j] = Xva @ beta
    ds = result_dataset(
        {"prediction": pd.DataFrame(pred, index=Y.index, columns=Y.columns),
         "fold": pd.Series(fold_id, index=Y.index, name="fold"),
         "method": "crossfit_covariate_ridge", "n_splits": int(np.max(fold_id) + 1)},
        dims={"prediction": ("sample", "trait"), "fold": ("sample",)},
        var_attrs={"prediction": {"long_name": "cross-fitted covariate-only phenotype prediction",
                                  "units": "phenotype units",
                                  "description": "each row predicted by a ridge model that excluded its phenotypes"},
                   "fold": {"long_name": "cross-fitting validation fold", "units": "index",
                            "description": "-1 marks rows never assigned to a validation fold"}},
        title="Cross-fitted non-genetic predictions",
        software="vigor.blup.crossfit_nongenetic_predictions",
        parameters={"n_splits": int(n_splits), "random_state": random_state, "ridge": float(ridge),
                    "grouped": groups is not None})
    return maybe_save(ds, save_path)
