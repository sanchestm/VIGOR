"""One-call, resource-bounded dominance-aware GWAS driver."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as _t
from threadpoolctl import threadpool_limits

from ._utils import neglog10p
from .assoc import iter_chromosome_gls_eigen
from .backend import resolve_backend_name
from .classify import classify_inheritance
from .grm import iter_loco_grms
from .io import PlinkReader
from .output import GWASZarrWriter
from .progress import StageProgress
from .resources import PeakRSSMonitor, ResourceConfig, plan_execution
from .vc import (
    estimate_h2_many,
    multivariate_score_transform,
    score_variance_components,
)


def _load_pheno(pheno):
    """Load pheno.

    Parameters
    ----------
    pheno : path-like or pandas.DataFrame
        Phenotype table, with unique animal IDs and numeric trait columns. File inputs are
        parsed/aligned by _load_pheno; NaN represents unobserved trait values.
    """
    if isinstance(pheno, pd.DataFrame):
        return pheno.copy()
    frame = pd.read_csv(pheno, sep=None, engine="python")
    columns = {column.lower(): column for column in frame.columns}
    id_column = columns.get("iid") or columns.get("id") or frame.columns[1]
    return frame.set_index(id_column)


def _prepare_covariates(covar, ids):
    """Prepare covariates.

    Parameters
    ----------
    covar : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
    """
    if covar is None:
        return pd.DataFrame(index=ids)
    if isinstance(covar, pd.DataFrame):
        covar = covar.copy()
        covar.index = covar.index.astype(str)
        frame = covar.loc[ids].copy()
        categorical = frame.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns
        numeric = frame.columns.difference(categorical, sort=False)
        parts = []
        if len(numeric):
            parts.append(frame[numeric].apply(pd.to_numeric, errors="coerce"))
        if len(categorical):
            parts.append(pd.get_dummies(frame[categorical], drop_first=True, dtype=float))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ids)
    values = np.asarray(covar, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(ids):
        raise ValueError("covariate array must have one row per aligned phenotype sample")
    return pd.DataFrame(values, index=ids)


def _full_rank_design(covariates):
    """Return the full rank design result computed by ``np.column_stack``.

    Parameters
    ----------
    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.
    """
    columns = [np.ones(len(covariates), dtype=float)]
    rank = 1
    for column in np.asarray(covariates, dtype=float).T:
        candidate = np.column_stack(columns + [column])
        new_rank = np.linalg.matrix_rank(candidate)
        if new_rank > rank:
            columns.append(column)
            rank = new_rank
    return np.column_stack(columns)


def _resolved_rank(n_samples, n_components, decomposition):
    """Resolve an eigensystem rank from sample count and decomposition policy.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Source comparisons: ``n_samples <= 4000``.
        See the rejection/fallback branches for enforcement.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.

    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver.
    """
    method = decomposition.lower()
    if method == "auto":
        method = "exact" if n_samples <= 4000 else "randomized"
    if n_components is None:
        return n_samples if method == "exact" else min(2000, max(n_samples - 1, 1))
    return min(int(n_components), n_samples)


def _matrix_by_trait(frame, value, traits, variants):
    """Return the matrix by trait result computed by ``frame.pivot(index='variant_index', columns='trait', values=value).reindex(index=variants, columns=traits).to_numpy``.

    Parameters
    ----------
    frame : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    value : object (see the cited input contract)
        Input supplied to ``frame.pivot`` as ``values``. See that callable's documented
        ``values`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Passed to ``frame.pivot`` as ``values``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Passed to ``frame.pivot(index='variant_index', columns='trait',
        values=value).reindex`` as ``columns``.

    variants : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader. Passed to ``frame.pivot(index='variant_index', columns='trait',
        values=value).reindex`` as ``index``.
    """
    return (
        frame.pivot(index="variant_index", columns="trait", values=value)
        .reindex(index=variants, columns=traits)
        .to_numpy(dtype=float)
    )


def _recover_multivariate_block(
    frame,
    component_names,
    original_traits,
    inverse_transform,
    n_samples,
    fixed_rank,
    model,
    da_thresholds,
    stability_z,
):
    """Recover original-trait effects and cross-trait Wald tests.

    Parameters
    ----------
    frame : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    component_names : object (see the cited input contract)
        Input for the operation ``len(component_names)``. See
        ``vigor.pipeline._recover_multivariate_block`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    original_traits : pandas.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.

    inverse_transform : array-like or numeric scalar
        Input for the operation ``np.asarray(inverse_transform, dtype=np.float64)``. See
        ``vigor.pipeline._recover_multivariate_block`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    fixed_rank : object (see the cited input contract)
        Input for the operation ``n_samples - fixed_rank - 1``. See
        ``vigor.pipeline._recover_multivariate_block`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Source-recognized literal values: ``add-dom``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    da_thresholds : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Passed to ``classify_inheritance`` as ``thresholds``.

    stability_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Passed to ``classify_inheritance`` as ``stability_z``.
    """
    variants = frame["variant_index"].drop_duplicates().tolist()
    base = (
        frame.drop_duplicates("variant_index")
        .set_index("variant_index")
        .reindex(variants)
    )
    t = len(component_names)
    inverse = np.asarray(inverse_transform, dtype=np.float64)
    inverse_squared = inverse * inverse

    def matrix(column):
        """Pivot the selected association metric into variant × trait order.

        Parameters
        ----------
        column : str, index or sequence
            Axis/column/feature selector used to identify the input/output quantities in this
            operation. Named selectors must exist in the labeled input; integer selectors are valid
            zero-based positions on that axis.

        Returns
        -------
        object
            Computed result from ``_matrix_by_trait(frame, column, component_names, variants)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _matrix_by_trait(frame, column, component_names, variants)

    beta_add_marginal_star = matrix("beta_additive_raw")
    variance_add_marginal_star = matrix("se_additive_raw") ** 2
    beta_add_marginal = beta_add_marginal_star @ inverse
    variance_add_marginal = variance_add_marginal_star @ inverse_squared
    se_add_marginal = np.sqrt(np.maximum(variance_add_marginal, 0.0))
    stat_add_marginal = beta_add_marginal / se_add_marginal
    df_additive = n_samples - fixed_rank - 1

    m = len(variants)
    repeated = {
        "variant_index": np.repeat(np.asarray(variants, dtype=np.int64), t),
        "snp": np.repeat(base["snp"].to_numpy(), t),
        "chrom": np.repeat(base["chrom"].to_numpy(), t),
        "pos": np.repeat(base["pos"].to_numpy(), t),
        "trait": np.tile(np.asarray(original_traits, dtype=object), m),
    }
    result = pd.DataFrame(repeated)
    result["beta_additive_raw"] = beta_add_marginal.ravel(order="C")
    result["se_additive_raw"] = se_add_marginal.ravel(order="C")
    result["stat_additive"] = stat_add_marginal.ravel(order="C")
    result["neglog_p_additive"] = neglog10p(
        2.0 * _t.sf(np.abs(stat_add_marginal), df=df_additive)
    ).ravel(order="C")

    add_scale = np.divide(
        matrix("beta_additive"),
        beta_add_marginal_star,
        out=np.full_like(beta_add_marginal_star, np.nan),
        where=np.abs(beta_add_marginal_star) > np.finfo(float).eps,
    )
    marker_add_scale = np.nanmedian(add_scale, axis=1)
    result["beta_additive"] = (
        beta_add_marginal * marker_add_scale[:, None]
    ).ravel(order="C")

    invariant_columns = [
        "n_obs",
        "count_AA",
        "count_AB",
        "count_BB",
        "maf",
        "genotype_filter_pass",
    ]
    for column in invariant_columns:
        result[column] = np.repeat(base[column].to_numpy(), t)
    result["dominance_class"] = np.where(
        result["genotype_filter_pass"].to_numpy(), "NA", "FILTERED"
    )
    result["inheritance_mode"] = result["dominance_class"]

    additive_multivariate = np.nansum(
        beta_add_marginal_star ** 2 / variance_add_marginal_star, axis=1
    )
    multi_statistics = {
        "stat_additive_multivariate": additive_multivariate,
        "neglog_p_additive_multivariate": neglog10p(
            chi2.sf(additive_multivariate, t)
        ),
        "df_additive_multivariate": np.full(m, t, dtype=float),
    }

    if model == "add-dom":
        beta_dom_marginal_star = matrix("beta_dominance_marginal_raw")
        variance_dom_marginal_star = matrix("se_dominance_marginal_raw") ** 2
        beta_dom_marginal = beta_dom_marginal_star @ inverse
        variance_dom_marginal = variance_dom_marginal_star @ inverse_squared
        se_dom_marginal = np.sqrt(np.maximum(variance_dom_marginal, 0.0))
        stat_dom_marginal = beta_dom_marginal / se_dom_marginal
        result["beta_dominance_marginal_raw"] = beta_dom_marginal.ravel(order="C")
        result["se_dominance_marginal_raw"] = se_dom_marginal.ravel(order="C")
        result["stat_dominance_marginal"] = stat_dom_marginal.ravel(order="C")
        result["neglog_p_dominance_marginal"] = neglog10p(
            2.0 * _t.sf(np.abs(stat_dom_marginal), df=df_additive)
        ).ravel(order="C")

        add_star = matrix("beta_add_joint_raw")
        dom_star = matrix("beta_dom_joint_raw")
        var_add_star = matrix("se_add_joint_raw") ** 2
        var_dom_star = matrix("se_dom_joint_raw") ** 2
        covariance_star = matrix("cov_add_dom_joint_raw")
        add = add_star @ inverse
        dom = dom_star @ inverse
        var_add = var_add_star @ inverse_squared
        var_dom = var_dom_star @ inverse_squared
        covariance = covariance_star @ inverse_squared
        se_add = np.sqrt(np.maximum(var_add, 0.0))
        se_dom = np.sqrt(np.maximum(var_dom, 0.0))
        stat_add = add / se_add
        stat_dom = dom / se_dom
        df_joint = n_samples - fixed_rank - 2
        fstat = stat_dom * stat_dom
        signed, magnitude, coarse, mode = classify_inheritance(
            add,
            dom,
            se_add,
            thresholds=da_thresholds,
            stability_z=stability_z,
        )
        valid = result["genotype_filter_pass"].to_numpy().reshape(m, t)
        coarse[~valid] = "FILTERED"
        mode[~valid] = "FILTERED"
        values = {
            "beta_add_joint_raw": add,
            "se_add_joint_raw": se_add,
            "beta_dom_joint_raw": dom,
            "se_dom_joint_raw": se_dom,
            "cov_add_dom_joint_raw": covariance,
            "stat_add_joint": stat_add,
            "stat_dom_joint": stat_dom,
            "neglog_p_add_joint": neglog10p(2.0 * _t.sf(np.abs(stat_add), df=df_joint)),
            "neglog_p_dom_joint": neglog10p(2.0 * _t.sf(np.abs(stat_dom), df=df_joint)),
            "f_avsad": fstat,
            "neglog_p_avsad": neglog10p(chi2.sf(fstat, 1)),
            "degree_of_dominance": signed,
            "degree_of_dominance_abs": magnitude,
            "dominance_class": coarse,
            "inheritance_mode": mode,
        }
        for column, value in values.items():
            result[column] = np.asarray(value).ravel(order="C")

        additive_joint_multivariate = np.nansum(add_star ** 2 / var_add_star, axis=1)
        dominance_joint_multivariate = np.nansum(dom_star ** 2 / var_dom_star, axis=1)
        determinant = var_add_star * var_dom_star - covariance_star ** 2
        component_joint = np.divide(
            var_dom_star * add_star ** 2
            - 2.0 * covariance_star * add_star * dom_star
            + var_add_star * dom_star ** 2,
            determinant,
            out=np.full_like(determinant, np.nan),
            where=determinant > np.finfo(float).eps,
        )
        add_dom_multivariate = np.nansum(component_joint, axis=1)
        multi_statistics.update(
            {
                "stat_add_joint_multivariate": additive_joint_multivariate,
                "neglog_p_add_joint_multivariate": neglog10p(
                    chi2.sf(additive_joint_multivariate, t)
                ),
                "df_add_joint_multivariate": np.full(m, t, dtype=float),
                "stat_dom_joint_multivariate": dominance_joint_multivariate,
                "neglog_p_dom_joint_multivariate": neglog10p(
                    chi2.sf(dominance_joint_multivariate, t)
                ),
                "df_dom_joint_multivariate": np.full(m, t, dtype=float),
                "stat_add_dom_multivariate": add_dom_multivariate,
                "neglog_p_add_dom_multivariate": neglog10p(
                    chi2.sf(add_dom_multivariate, 2 * t)
                ),
                "df_add_dom_multivariate": np.full(m, 2 * t, dtype=float),
            }
        )

    for column in multi_statistics:
        result[column] = np.nan
    result["multivariate_df"] = np.nan
    multivariate = result.iloc[::t].copy().reset_index(drop=True)
    multivariate["trait"] = "__joint__"
    for column in result.columns:
        if column not in {"variant_index", "snp", "chrom", "pos", "trait", *invariant_columns}:
            multivariate[column] = np.nan
    for column, values in multi_statistics.items():
        multivariate[column] = values
    multivariate["multivariate_df"] = t
    multivariate["dominance_class"] = "NA"
    multivariate["inheritance_mode"] = "NA"
    return pd.concat([result, multivariate], ignore_index=True, sort=False)


def _write_json_atomic(path, value):
    """Write json atomic.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    value : object (see the cited input contract)
        Input for the operation ``temporary.write_text(json.dumps(value, indent=2),
        encoding='utf-8')``. See ``vigor.pipeline._write_json_atomic`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_covariance_zarr(path, *, trait_names=None, **arrays):
    """Write covariance zarr.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value. Passed to ``xr.Dataset`` as ``coords``.

    **arrays : dict
        Input for the operation ``xr.Dataset({name: (('trait_i', 'trait_j') if
        np.asarray(array).ndim == 2 else tuple((f'{name}_axis{i}' for i in
        range(np.asarray(array).ndim))), np.asarray(array)) for name, array in arrays.items()},
        coords={} if trait_names is None else...``. See
        ``vigor.pipeline._write_covariance_zarr`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    import xarray as xr
    from .store import VigorStore
    ds=xr.Dataset({name:(('trait_i','trait_j') if np.asarray(array).ndim==2 else tuple(f'{name}_axis{i}' for i in range(np.asarray(array).ndim)),np.asarray(array))
                   for name,array in arrays.items()},
                   coords={} if trait_names is None else {'trait_i':list(map(str,trait_names)),'trait_j':list(map(str,trait_names))})
    VigorStore(path,mode='w').write_dataset(ds,'covariance')


def run_gwas(bfile, pheno, traits=None, covar=None, model="add-dom", chroms=None,
             block_size=128, n_components=None, y_correction="ystar", out=None,
             per_trait_missing=True, grm_bfile=None, return_results=True,
             max_return_rows=2_000_000, verbose=True, trait_mode="multivariate",
             variance_estimator="auto", decomposition="exact", compute_dtype="float32",
             memory_budget_mb=None, grm_cache=None, n_jobs=1, resource_config=None,
             backend="cpu", progress=None, covariance_kwargs=None,
             solver_tol=2e-6, solver_maxiter=400, rhs_block_size=16,
             variant_chunk=1024, shard_variants=16384, score_engine="fast",
             fast_calibration_variants=32, **retired):
    """Run LOCO Gaussian GWAS through the common observed-space score core.

    Parameters
    ----------
    bfile, grm_bfile : path-like
        PLINK1 test and optional background filesets. Animal IDs align by IID.
    pheno : DataFrame or delimited path
        Unique animal rows, numeric quantitative trait columns, NaN missing cells.
    traits : sequence of str, optional
        Selected phenotype columns. None uses columns except FID.
    covar : DataFrame, array or None
        Additional fixed effects; intercepts are always projected for every trait.
    model : {'additive','add-dom'}
        Additive dosage alone or a joint dosage/heterozygote design per locus.
        A+D results have contrast:trait coordinates and a full joint omnibus.
    chroms : sequence, optional
        Chromosomes to test; all chromosomes still contribute to LOCO backgrounds.
    block_size : positive int or 'auto'
        Variant decoding/scoring block ceiling.
    n_components : None
        Rank truncation is retired: the supplied LOCO covariance is not approximated.
    y_correction : {'ystar', None}
        GLS score null or OLS covariance; no residualized association shortcut.
    out : path-like, optional
        Stream sharded Zarr v3 and return a lazy Dataset.
    per_trait_missing : bool
        True retains actual trait-specific masks; False explicitly uses complete cases.
    return_results : bool
        False requires out. Output remains a lazy Dataset when out is present.
    max_return_rows : int or None
        Maximum score cells retained when out is absent.
    verbose, progress : bool
        Enable fastprogress-compatible stage reporting.
    trait_mode : {'multivariate','independent'}
        Joint G/E null or independent per-trait diagonal covariance.
    variance_estimator : {'auto','score','reml','profile_reml','ml','profile_ml'}
        Multivariate mode uses the joint covariance moment fit. Independent mode
        supports per-trait score or profile likelihood variance estimates.
    decomposition : {'exact'}
        Exact full background kernel. Randomized null-covariance substitution retired.
    compute_dtype : {'float32','float64'}
        Dense matrix-product dtype. Core reductions and small solves use float64.
    memory_budget_mb, resource_config : optional
        ResourceConfig memory guard and block limit. Dense matrices must fit.
    grm_cache : path-like, optional
        Existing canonical chromosome GRM store, read with labeled LOCO kernels.
    n_jobs : positive int
        Local BLAS worker limit; restored on exit.
    backend : {'cpu'}
        Current exact core backend.
    covariance_kwargs : mapping, optional
        genetic_varcov_PSD controls for multivariate LOCO fits.
    solver_tol, solver_maxiter, rhs_block_size : float, int, int
        Strict PCG tolerance, iteration cap and simultaneous RHS count.
    variant_chunk, shard_variants : positive int
        Persisted chunk and enclosing shard lengths.
    score_engine : {'fast','exact'}
        'fast' (default): exact score X'Py with the calibrated D'PD ratio
        approximation of vigor.gwas.fast, fitted per chromosome against exact
        information on ``fast_calibration_variants`` spread variants; O(n c t)
        per variant. 'exact': full D'PD (block PCG per variant column when
        phenotypes are missing). Same output schema; per-chromosome calibration
        diagnostics are in ``attrs['scan_group_metadata']``.
    fast_calibration_variants : positive int
        Exact-information variants per chromosome for the fast calibration.
    **retired : mapping
        Unsupported legacy controls fail with a migration message.

    Returns
    -------
    xarray.Dataset
        Exact null score tests, per-locus omnibus, counted allele and physical
        variant labels. No dominance classification is inferred from null scores.

    Complexity
    ----------
    Background reconstruction O(n²m) per chromosome without a persisted store.
    Dense eig setup O(n³); score work is O(iterations*m*t*(n²t+nt²)), bounded
    by RHS blocks. With out, result memory is O(one output block).
    """
    from .gwas.scan import (scan_inputs, background_for_group, genotype_blocks,
                             attach_variant_metadata, finish_scan, covariance_matrices)
    from .gwas.exact import prepare_score_null, score_block, score_design_block
    from .multivariate import _genetic_varcov_PSD_raw
    from .genotype_source import open_genotypes
    from .gwas.fast import calibrate_group
    if score_engine not in {"fast", "exact"}:
        raise ValueError("score_engine must be 'fast' or 'exact'")
    if retired:
        raise TypeError("retired pipeline controls: " + ", ".join(sorted(retired)) +
                        "; see docs/ROUND3_MIGRATION.md")
    if model not in {"additive", "add-dom"}:
        raise ValueError("model must be additive or add-dom")
    if decomposition != "exact" or n_components is not None:
        raise ValueError("exact score GWAS requires decomposition='exact', n_components=None")
    if trait_mode not in {"independent", "multivariate"}:
        raise ValueError("trait_mode must be independent or multivariate")
    if backend != "cpu":
        raise ValueError("the exact score core currently supports backend='cpu'")
    if y_correction not in {"ystar", None}:
        raise ValueError("y_correction must be ystar or None")
    allowed = {"auto", "score", "reml", "profile_reml", "ml", "profile_ml"}
    if variance_estimator not in allowed:
        raise ValueError("unknown variance_estimator")
    if trait_mode == "multivariate" and variance_estimator not in {"auto", "score"}:
        raise ValueError("multivariate mode requires variance_estimator='score' or 'auto'")
    if not return_results and out is None:
        raise ValueError("out is required when return_results=False")
    config = (resource_config or ResourceConfig(memory_budget_mb=memory_budget_mb,
                                                n_jobs=n_jobs, backend=backend)).resolved()
    if config.backend != "cpu":
        raise ValueError("resource_config backend must be cpu")
    phenotype = _load_pheno(pheno)
    phenotype.index = phenotype.index.astype(str)
    columns = list(traits) if traits is not None else [c for c in phenotype if str(c).lower() != "fid"]
    sub = phenotype.loc[:, columns].apply(pd.to_numeric, errors="raise")
    Cframe = _prepare_covariates(covar, sub.index)
    complete_cov = np.isfinite(Cframe.to_numpy(float)).all(axis=1)
    rows = complete_cov & (sub.notna().any(axis=1).to_numpy() if per_trait_missing else sub.notna().all(axis=1).to_numpy())
    sub, Cframe = sub.loc[rows], Cframe.loc[rows]
    Y, source, vm = scan_inputs(sub, bfile, maxmem=config.memory_budget_mb * 1024**2)
    if any(Y[c].dropna().nunique() == 2 for c in Y):
        raise ValueError("binary outcomes are not supported by the Gaussian score model")
    C = _full_rank_design(Cframe.to_numpy(float))
    fitY = Y.copy()
    for c in fitY:
        mask = fitY[c].notna().to_numpy()
        fitY.loc[mask, c] -= C[mask] @ np.linalg.lstsq(C[mask], fitY.loc[mask, c], rcond=None)[0]
    labels = list(dict.fromkeys(vm.chromosome.astype(str))) if chroms is None else list(map(str, chroms))
    if not set(labels).issubset(set(vm.chromosome.astype(str))):
        raise ValueError("requested chromosome absent from test genotypes")
    bgsource = source if grm_bfile is None else open_genotypes(grm_bfile, maxmem=config.memory_budget_mb * 1024**2)
    bgvm = bgsource.variant_metadata.copy()
    reporter = StageProgress(desc=f"{score_engine.capitalize()} GWAS", enabled=bool(verbose) if progress is None else bool(progress), unit="block")
    def blocks():
        try:
            with threadpool_limits(limits=config.n_jobs):
                for chrom in labels:
                    reporter.stage(f"chr {chrom}: null covariance")
                    dense_bytes = len(Y)**2 * np.dtype(compute_dtype).itemsize * 4
                    if dense_bytes >= config.memory_budget_mb * 1024**2:
                        raise MemoryError("exact dense LOCO null and eig workspace exceed memory budget")
                    if y_correction is None:
                        K = np.eye(len(Y), dtype=compute_dtype)
                        G = np.zeros((Y.shape[1], Y.shape[1]))
                        E = np.diag(np.nanvar(fitY.to_numpy(float), axis=0, ddof=C.shape[1]))
                    else:
                        K = background_for_group(bgsource, bgvm, Y.index, chrom, kernels=grm_cache,
                                                  block_size=block_size, compute_dtype=compute_dtype)
                        if trait_mode == "multivariate":
                            kw = dict(covariance_kwargs or {})
                            kw.setdefault("n_boot", 0); kw.setdefault("progress", False)
                            G, E = covariance_matrices(_genetic_varcov_PSD_raw(K, fitY, **kw))
                        else:
                            Uvalues, U = np.linalg.eigh(np.asarray(K, dtype=np.float64))
                            fits = []
                            for c in Y:
                                mask = Y[c].notna().to_numpy()
                                # Re-diagonalize the actual observed subkernel; slicing U
                                # would not be an eigensystem under heterogeneous masks.
                                if mask.all():
                                    vals, vecs = Uvalues, U
                                else:
                                    vals, vecs = np.linalg.eigh(np.asarray(K)[np.ix_(mask, mask)].astype(float))
                                yy = Y.loc[mask, [c]].to_numpy(float)
                                if variance_estimator == "score":
                                    fitted = score_variance_components(yy, vecs, vals, covar=C[mask]).attrs["fits"][0]
                                else:
                                    fitted = estimate_h2_many(yy, vecs, vals, covar=C[mask],
                                        method="ML" if variance_estimator in {"ml", "profile_ml"} else "REML")[0]
                                fits.append(fitted)
                            G = np.diag([f["h2"] * f["yvar"] for f in fits])
                            E = np.diag([(1-f["h2"]) * f["yvar"] for f in fits])
                    null = prepare_score_null(Y, K, G, E, covariates=Cframe,
                        solver_tol=solver_tol, solver_maxiter=solver_maxiter,
                        rhs_block_size=rhs_block_size, work_dtype=compute_dtype)
                    group_meta = None
                    if score_engine == "fast":
                        reporter.stage(f"chr {chrom}: fast calibration")
                        fast = calibrate_group(null, source, vm, Y.index, chrom,
                                               n_calibration=fast_calibration_variants,
                                               block_size=block_size, model=model)
                        group_meta = {str(chrom): {"score_engine": "fast", **fast.metadata()}}
                    for X, meta in genotype_blocks(source, vm, Y.index, chrom, block_size=block_size):
                        reporter.stage(f"chr {chrom}: {score_engine} score / save")
                        if model == "additive":
                            ds = score_block(null, X, engine=score_engine)
                        else:
                            dosage = X.to_numpy(np.float32)
                            observed = np.isfinite(dosage)
                            if np.any(observed & (dosage != np.round(dosage))):
                                raise ValueError("add-dom heterozygote tests require hardcalls, not fractional dosages")
                            dom = np.where(observed, dosage == 1, np.nan).astype(np.float32)
                            ds = score_design_block(null, np.stack([dosage, dom], axis=2),
                                                    contrast_names=["additive", "dominance"],
                                                    engine=score_engine)
                        ds = attach_variant_metadata(ds, meta, group=chrom)
                        ds.attrs.update(trait_mode=trait_mode, model=model,
                            variance_estimator=variance_estimator, background="LOCO" if y_correction else "OLS")
                        if group_meta is not None:
                            ds.attrs["scan_group_metadata"] = group_meta
                        reporter.update()
                        yield ds
        finally:
            reporter.close()
            if hasattr(source, "close"):
                source.close()
            if bgsource is not source and hasattr(bgsource, "close"):
                bgsource.close()
    return finish_scan(blocks(), out=out, max_return_rows=max_return_rows,
                        variant_chunk=variant_chunk, shard_variants=shard_variants)
