"""Internal Optuna CV for genetic_varcov_PSD prediction hyperparameters.

The optimizer searches a common genetic/residual factor rank as well as
covariance regularization/geometry. It conservatively retains the explicit
FA/auto incumbent unless a candidate improves prediction across CV folds.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd


def _meta(result):
    """Return the scalar/diagnostic mapping of a fit or prediction result.

    Parameters
    ----------
    result : xarray.Dataset or mapping
        Covariance-fit or prediction result. Dataset results keep scalar fields such as
        success, message and cg_info in ``attrs``; legacy dict/Series results are returned
        unchanged so ``.get`` lookups work for both.
    """
    return result.attrs if hasattr(result, "data_vars") and hasattr(result, "attrs") else result


def _as_dataframe(traits):
    """Convert inputs to dataframe.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.
    """
    return traits if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)


def validate_cv_count(value: int) -> int:
    """Validate the HPO fold count: zero disables search and integers at least two enable CV.

    Parameters
    ----------
    value : int
        Input for the operation ``value < 0``. See ``vigor.hyperparameter_cv.validate_cv_count``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Source comparisons: ``value < 0``; ``value == 1``. See the
        rejection/fallback branches for enforcement.

    Returns
    -------
    object
        Computed result from ``value``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("optimize_hyperparameters_cv must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("optimize_hyperparameters_cv must be 0 or >=2")
    if value == 1:
        raise ValueError("optimize_hyperparameters_cv=1 is not cross-validation; use 0 or >=2")
    return value


def _make_cv_masks(Y: pd.DataFrame, n_splits: int, seed: int):
    """Construct cv masks.

    Parameters
    ----------
    Y : pd.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.

    seed : int
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.
    """
    values = Y.to_numpy(float)
    observed = np.isfinite(values)
    eligible_rows = np.flatnonzero(observed.any(axis=1))
    if len(eligible_rows) < n_splits:
        raise ValueError(
            f"optimize_hyperparameters_cv={n_splits} requires at least {n_splits} "
            "animals with one observed trait"
        )

    rng = np.random.default_rng(int(seed))
    rows = eligible_rows.copy()
    rng.shuffle(rows)
    row_fold = np.full(len(Y), -1, dtype=int)
    for fold, chunk in enumerate(np.array_split(rows, n_splits)):
        row_fold[chunk] = fold

    # Every eligible partial row retains an observation for conditioning.
    # Distribute traits by row, rather than risking empty random cell folds.
    cell_fold = np.full(values.shape, -1, dtype=int)
    for i in eligible_rows:
        cols = rng.permutation(np.flatnonzero(observed[i]))
        cell_fold[i, cols] = (np.arange(len(cols)) + i) % n_splits

    folds = []
    for fold in range(n_splits):
        unseen_rows = row_fold == fold
        partial = observed & (cell_fold == fold) & ~unseen_rows[:, None]
        for i in np.flatnonzero(partial.any(axis=1)):
            obs = np.flatnonzero(observed[i])
            if len(obs) and partial[i, obs].all():
                partial[i, obs[0]] = False

        train = Y.copy()
        train.iloc[unseen_rows, :] = np.nan
        train = train.mask(partial)
        if (train.notna().sum() < 2).any():
            raise ValueError("prediction CV requires at least two training observations per trait in every fold")
        if not partial.any():
            raise ValueError("prediction CV requires partial-cell holdouts in every fold; supply more jointly observed traits/animals")

        kind = np.full(values.shape, "train", dtype=object)
        kind[~observed] = "missing"
        kind[partial] = "partial_cell"
        kind[unseen_rows[:, None] & observed] = "unseen_animal"
        folds.append((train, kind.astype(str)))
    return folds


def _standardized_losses(reference: pd.DataFrame, prediction, kind, train: pd.DataFrame):
    """Evaluate held-out squared prediction errors using training-only trait scales.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields.

    prediction : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    kind : str
        Edge/evidence/input interpretation selected by this constructor/helper. See the local
        branches and the source-derived option descriptions below; each builder admits only its
        own subset, not a package-wide enum.

    train : pd.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.
    """
    truth = reference.to_numpy(float)
    pred = np.asarray(prediction, float)
    sd = train.std(ddof=0).to_numpy(float).copy()  # read-only view under pandas>=3 copy-on-write
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    sq = ((pred - truth) / sd[None, :]) ** 2
    out = {}
    cells = {}
    for regime in ("unseen_animal", "partial_cell"):
        active = (kind == regime) & np.isfinite(truth)
        per_trait = []
        cells[regime] = []
        for j in range(truth.shape[1]):
            m = active[:, j]
            if np.any(m):
                if not np.isfinite(sq[m, j]).all():
                    raise FloatingPointError("non-finite held-out prediction loss")
                per_trait.append(float(np.mean(sq[m, j])))
                cells[regime].append(sq[m, j].astype(np.float64, copy=False))
        out[regime] = float(np.mean(per_trait)) if per_trait else np.nan
    vals = [out[r] for r in ("unseen_animal", "partial_cell") if np.isfinite(out[r])]
    out["balanced"] = float(np.mean(vals)) if vals else np.inf
    return out, cells


def _config_key(config):
    """Return the config key result computed by ``json.dumps``.

    Parameters
    ----------
    config : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def incumbent_config(*, rank_G="full", rank_E="full"):
    """Return the conservative FA configuration with explicitly recorded ranks.

    Parameters
    ----------
    rank_G, rank_E : positive int, 'auto', 'full', or mapping for G
        Requested ranks for the baseline. Defaults preserve the full-rank
        incumbent; callers can retain a previously selected rank explicitly.

    Returns
    -------
    dict
        Result fields: ``covariance_structure``, ``rank_G``, ``rank_E``, ``rank_selection``, ``shrink_G``, ``shrink_E``, ``E_structure``, ``min_E_pair_overlap``, ``ridge_G``, ``ridge_E``, ``factor_structure_G``, ``factor_structure_E``, ``factor_rotation_G``, ``factor_rotation_E``, ``factor_lambda_G``, ``factor_lambda_E``, ``factor_group_lambda_G``, ``factor_group_lambda_E``, ``factor_adaptive``, ``factor_relaxed_refit``, ``factor_auto_fraction``, ``variance_closure``, ``variance_closure_strength``. Field arrays preserve the corresponding input axis order.
    """
    return {
        "covariance_structure": "fa",
        "rank_G": rank_G,
        "rank_E": rank_E,
        "rank_selection": "none",
        "shrink_G": "auto",
        "shrink_E": 0.5263454565339929,
        "E_structure": "auto",
        "min_E_pair_overlap": 2,
        "ridge_G": 0.0,
        "ridge_E": 0.0,
        "factor_structure_G": "fa",
        "factor_structure_E": "fa",
        "factor_rotation_G": None,
        "factor_rotation_E": None,
        "factor_lambda_G": None,
        "factor_lambda_E": None,
        "factor_group_lambda_G": 0.0,
        "factor_group_lambda_E": 0.0,
        "factor_adaptive": True,
        "factor_relaxed_refit": True,
        "factor_auto_fraction": 0.05,
        "variance_closure": "auto",
        "variance_closure_strength": 1.0,
    }


def _rank_choices(n_traits):
    """Return stable categorical ranks bounded by the number of supplied traits."""
    if isinstance(n_traits, bool) or not isinstance(n_traits, (int, np.integer)) or n_traits < 1:
        raise ValueError("n_traits must be a positive integer")
    return [r for r in (1, 2, 4, 8, 10, 20, 40, 80, 160) if r < n_traits] + ["full"]


def _trial_config(trial, n_traits=None):
    """Draw one rank/retention/factor-geometry configuration.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial providing suggest_* methods for this training-only hyperparameter
        search. Trial state records the chosen configuration and fit/evaluation diagnostics.

    n_traits : positive int or None
        Number of traits determines the categorical common G/E rank grid.
        None retains full rank for compatibility with private historical callers;
        the actual optimizer always supplies the observed trait count.
    """
    rank = ("full" if n_traits is None else
            trial.suggest_categorical("factor_rank", _rank_choices(n_traits)))
    g_mode = trial.suggest_categorical("G_retention_mode", ["auto", "numeric"])
    shrink_g = "auto" if g_mode == "auto" else float(
        trial.suggest_float("G_retention", 0.25, 1.0)
    )

    e_structure = trial.suggest_categorical(
        "E_structure", ["auto", "fa", "full", "auto_supported"]
    )
    # The historical supported-block policy accepts scalar retention only.
    # Separate names keep Optuna categorical distributions fixed across trials.
    if e_structure == "auto_supported":
        e_mode = trial.suggest_categorical(
            "E_supported_retention_mode", ["auto", "numeric", "legacy_auto"]
        )
        e_numeric_name = "E_supported_retention"
    else:
        e_mode = trial.suggest_categorical(
            "E_retention_mode",
            ["auto", "numeric", "pairwise", "pairwise_full", "fisher_eb_full", "legacy_auto"],
        )
        e_numeric_name = "E_retention"
    shrink_e = (
        float(trial.suggest_float(e_numeric_name, 0.0, 1.0))
        if e_mode == "numeric"
        else e_mode
    )

    factor_choices = ["fa", "pca", "sparse_fa", "sparse_pca"]
    fg = trial.suggest_categorical("factor_structure_G", factor_choices)
    fe = trial.suggest_categorical("factor_structure_E", factor_choices)

    rg_mode = trial.suggest_categorical("ridge_G_mode", ["zero", "positive"])
    re_mode = trial.suggest_categorical("ridge_E_mode", ["zero", "positive"])
    ridge_g = 0.0 if rg_mode == "zero" else float(
        trial.suggest_float("ridge_G", 1e-8, 2.0, log=True)
    )
    ridge_e = 0.0 if re_mode == "zero" else float(
        trial.suggest_float("ridge_E", 1e-8, 2.0, log=True)
    )

    min_overlap = (
        int(trial.suggest_categorical("min_E_pair_overlap", [2, 4, 8, 16, 32, 64]))
        if e_structure in {"auto", "auto_supported"}
        else 2
    )

    sparse_g = fg in {"sparse_fa", "sparse_pca"}
    sparse_e = fe in {"sparse_fa", "sparse_pca"}
    if sparse_g:
        rot_g = trial.suggest_categorical("factor_rotation_G", [None, "varimax"])
        group_g = trial.suggest_categorical("factor_group_lambda_G", [0.0, "auto"])
    else:
        rot_g, group_g = None, 0.0
    if sparse_e:
        rot_e = trial.suggest_categorical("factor_rotation_E", [None, "varimax"])
        group_e = trial.suggest_categorical("factor_group_lambda_E", [0.0, "auto"])
    else:
        rot_e, group_e = None, 0.0

    if sparse_g or sparse_e:
        auto_fraction = float(
            trial.suggest_float("factor_auto_fraction", 0.002, 0.30, log=True)
        )
        adaptive = bool(trial.suggest_categorical("factor_adaptive", [True, False]))
        relaxed = bool(trial.suggest_categorical("factor_relaxed_refit", [True, False]))
    else:
        auto_fraction, adaptive, relaxed = 0.05, True, True

    closure = trial.suggest_categorical("variance_closure", ["auto", "off"])
    closure_strength = (
        float(trial.suggest_float("variance_closure_strength", 0.03, 10.0, log=True))
        if closure == "auto"
        else 1.0
    )

    return {
        "covariance_structure": "fa",
        "rank_G": rank,
        "rank_E": rank,
        "rank_selection": "none",
        "shrink_G": shrink_g,
        "shrink_E": shrink_e,
        "E_structure": e_structure,
        "min_E_pair_overlap": min_overlap,
        "ridge_G": ridge_g,
        "ridge_E": ridge_e,
        "factor_structure_G": fg,
        "factor_structure_E": fe,
        "factor_rotation_G": rot_g,
        "factor_rotation_E": rot_e,
        "factor_lambda_G": "auto" if sparse_g else None,
        "factor_lambda_E": "auto" if sparse_e else None,
        "factor_group_lambda_G": group_g,
        "factor_group_lambda_E": group_e,
        "factor_adaptive": adaptive,
        "factor_relaxed_refit": relaxed,
        "factor_auto_fraction": auto_fraction,
        "variance_closure": closure,
        "variance_closure_strength": closure_strength,
    }


def _anchors():
    """Build fixed baseline/search configurations for reproducible HPO comparison.
        
    """
    return [
        {
            "factor_rank": "full",
            "G_retention_mode": "auto",
            "E_retention_mode": "auto",
            "E_structure": "auto",
            "factor_structure_G": "fa",
            "factor_structure_E": "fa",
            "ridge_G_mode": "zero",
            "ridge_E_mode": "zero",
            "min_E_pair_overlap": 2,
            "variance_closure": "auto",
            "variance_closure_strength": 1.0,
        },
        {
            "factor_rank": "full",
            "G_retention_mode": "numeric",
            "G_retention": 0.52,
            "E_retention_mode": "pairwise_full",
            "E_structure": "auto",
            "factor_structure_G": "fa",
            "factor_structure_E": "pca",
            "ridge_G_mode": "positive",
            "ridge_G": 3.5e-5,
            "ridge_E_mode": "positive",
            "ridge_E": 9.0e-6,
            "min_E_pair_overlap": 4,
            "variance_closure": "off",
        },
        {
            "factor_rank": "full",
            "G_retention_mode": "numeric",
            "G_retention": 1.0,
            "E_retention_mode": "numeric",
            "E_retention": 1.0,
            "E_structure": "full",
            "factor_structure_G": "pca",
            "factor_structure_E": "pca",
            "ridge_G_mode": "zero",
            "ridge_E_mode": "zero",
            "variance_closure": "off",
        },
        {
            "factor_rank": "full",
            "G_retention_mode": "auto",
            "E_retention_mode": "fisher_eb_full",
            "E_structure": "auto",
            "factor_structure_G": "sparse_fa",
            "factor_structure_E": "sparse_fa",
            "ridge_G_mode": "zero",
            "ridge_E_mode": "zero",
            "min_E_pair_overlap": 8,
            "factor_rotation_G": "varimax",
            "factor_rotation_E": "varimax",
            "factor_group_lambda_G": "auto",
            "factor_group_lambda_E": "auto",
            "factor_auto_fraction": 0.05,
            "factor_adaptive": True,
            "factor_relaxed_refit": True,
            "variance_closure": "auto",
            "variance_closure_strength": 1.0,
        },
    ]


class _InvalidTrial(RuntimeError):
    """A failed fit/prediction must be an Optuna FAIL, never a completed trial."""


def _trial_budget(cv):
    # Private helper so bounded tests can exercise all four anchors cheaply.
    """Return the trial budget result computed by ``max``.

    Parameters
    ----------
    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
    """
    return max(24, min(64, 8 * cv))


def _paired_cell_bootstrap(candidate, incumbent, seed, n_boot=1000):
    """Paired cell resampling stratified by fold, prediction state and trait.

    Equal fold/state/trait weights match the CV objective. The interval is
    conditional on these folds and the candidate search, not an outer-test CI.

    Parameters
    ----------
    candidate : object (see the cited input contract)
        Input for the operation ``1.0 / len(candidate) / 2.0 / len(cand)``. See
        ``vigor.hyperparameter_cv._paired_cell_bootstrap`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    incumbent : object (see the cited input contract)
        Input for the operation ``zip(candidate, incumbent)``. See
        ``vigor.hyperparameter_cv._paired_cell_bootstrap`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    n_boot : int
        Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
        estimate sampling variation by the resampling scheme used here. Default: ``1000``.
    """
    strata = []
    for cand_fold, inc_fold in zip(candidate, incumbent):
        for regime in ("unseen_animal", "partial_cell"):
            cand, inc = cand_fold[regime], inc_fold[regime]
            for c, i in zip(cand, inc):
                weight = 1.0 / len(candidate) / 2.0 / len(cand)
                strata.append((c - i, weight))
    rng = np.random.default_rng(seed)
    means = np.zeros(n_boot)
    for delta, weight in strata:
        # One resample at a time bounds memory for large held-out cohorts.
        for b in range(n_boot):
            means[b] += weight * np.mean(delta[rng.integers(0, len(delta), len(delta))])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def fit_with_hyperparameter_cv(fit_fn, predict_fn, call_arguments):
    """Small public-estimator adapter; all recursive fits explicitly disable HPO.

    Parameters
    ----------
    fit_fn : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Passed to
        ``optimize_covariance_hyperparameters`` as ``fit_fn``.

    predict_fn : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation. Passed to
        ``optimize_covariance_hyperparameters`` as ``predict_fn``.

    call_arguments : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Passed to ``optimize_covariance_hyperparameters`` as ``traits``.

    Returns
    -------
    xarray.Dataset or mapping
        The selected full-data fit in fit_fn's container with ``optimize_hyperparameters_cv``
        and ``hyperparameter_optimization`` set (in ``attrs`` for a Dataset). The public
        genetic_varcov_PSD passes its undecorated estimator, so its own decorator converts the
        returned fit to an ``xarray.Dataset`` once.
    """
    cv = call_arguments["optimize_hyperparameters_cv"]
    legacy = dict(call_arguments["legacy_kwargs"])
    grm = call_arguments["grm"]
    if legacy.get("environment") is not None or (
        isinstance(grm, Mapping)
        and any("gxe" in str(name).lower() for name in grm)
    ):
        raise ValueError("optimize_hyperparameters_cv is not supported for environment/GxE routes")
    if str(call_arguments["moment_estimator"]).lower() not in {"mom", "score"}:
        raise ValueError("optimize_hyperparameters_cv supports MoM/score only; MINQUE is excluded")
    if str(call_arguments["solver"]).lower() not in {"lbfgs", "l-bfgs-b", "factor"}:
        raise ValueError("optimize_hyperparameters_cv requires solver='lbfgs' to fit all searched factor geometries")
    if str(call_arguments["covariance_structure"]).lower().replace("-", "_") not in {
        "fa", "factor", "factor_analytic", "psd", "pca", "eigen", "varimax_fa",
        "sparse_fa", "sparse_pca", "sparse_varimax_fa",
    }:
        raise ValueError("optimize_hyperparameters_cv requires the standard factor covariance path")
    legacy.pop("_runtime_progress", None)
    kwargs = {}
    for name, param in inspect.signature(fit_fn).parameters.items():
        if param.kind != inspect.Parameter.KEYWORD_ONLY:
            continue
        # The raw estimator may be wrapped with additional opt-in controls.
        # Those appear in its public signature but are absent from the inner
        # estimator's locals() because the wrapper already consumed them.
        if name in call_arguments:
            kwargs[name] = call_arguments[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise TypeError(f"missing required covariance CV argument: {name}")
    kwargs.update(legacy)
    # Persistence belongs to the outer public call; folds and the refit never write.
    kwargs.pop("save_path", None)
    bar = call_arguments["legacy_kwargs"].get("_runtime_progress")
    if bar is not None:
        bar.stage("Covariance: prediction hyperparameter CV")
    selected, metadata = optimize_covariance_hyperparameters(
        fit_fn=fit_fn, predict_fn=predict_fn, grm=grm,
        traits=call_arguments["traits"], cv=cv, base_fit_kwargs=kwargs,
        random_state=call_arguments["random_state"],
    )
    kwargs.update(selected)
    kwargs["optimize_hyperparameters_cv"] = 0
    if bar is not None:
        bar.stage("Covariance: selected full-data fit")
    result = fit_fn(grm, call_arguments["traits"], **kwargs)
    # Dataset fits keep these non-array fields in attrs; legacy containers by key.
    target = _meta(result)
    target["optimize_hyperparameters_cv"] = cv
    target["hyperparameter_optimization"] = metadata
    return result


def optimize_covariance_hyperparameters(
    *, fit_fn: Callable, predict_fn: Callable, grm, traits, cv: int,
    base_fit_kwargs: dict[str, Any], random_state: int | None,
):
    """Select covariance hyperparameters with training-only folds and conservative incumbent promotion.

    Parameters
    ----------
    fit_fn : Callable
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation.

    predict_fn : Callable
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Source comparisons: ``cv == 0``. See the rejection/fallback branches for enforcement.

    base_fit_kwargs : dict[str, Any]
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration.

    random_state : int | None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    Returns
    -------
    object
        Computed result from ``(selected, metadata)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    cv = validate_cv_count(cv)
    if cv == 0:
        return incumbent_config(), None
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optimize_hyperparameters_cv>=2 requires Optuna. Install the HPO extra "
            "with python -m pip install 'VIGOR[hpo]' (optuna>=4,<6)."
        ) from exc

    Y = _as_dataframe(traits)
    seed = 0 if random_state is None else int(random_state)
    folds = _make_cv_masks(Y, cv, seed + 9187)
    n_trials = max(len(_anchors()), int(_trial_budget(cv)))
    common = dict(base_fit_kwargs)
    common.update(
        optimize_hyperparameters_cv=0, progress=False, n_boot=0, analytic_se=False,
        rank_selection="none",
        # Caller warm starts can encode all supplied outcomes, including holdouts.
        warm_start=None, random_state=seed,
        structural_plots=False,
    )
    # Inner fits keep any h2-prior settings, so ranks/geometry are compared for
    # the estimator that will be refitted.
    common.pop("_runtime_progress", None)
    common.pop("save_path", None)
    cache = {}

    def evaluate(config):
        """Fit/evaluate the proposed configuration on the enclosing training-only validation folds.

        Parameters
        ----------
        config : mapping or None
            Named settings or metadata forwarded to the specific operation below. See that
            operation's argument/schema contract for accepted fields; None uses its default/empty
            configuration.

        Returns
        -------
        object
            Computed result from ``evaluated``; see the summary and the called implementation for its labeled/numerical representation.
        """
        key = _config_key(config)
        if key in cache:
            return cache[key]
        fold_scores, cell_losses, runtimes = [], [], []
        regime_scores = {"unseen_animal": [], "partial_cell": []}
        for train, kind in folds:
            kwargs = dict(common)
            kwargs.update(config)
            started = time.perf_counter()
            fit = fit_fn(grm, train, **kwargs)
            info = _meta(fit)
            result = info.get("result")
            if (not bool(info.get("success", False))
                    or not bool(info.get("optimizer_converged", True))
                    or (result is not None and not bool(result.success))):
                raise _InvalidTrial(f"covariance fit did not converge: {info.get('message')}")
            pred = predict_fn(
                train, grm, fit, accuracy="exact", solver_tol=1e-6,
                solver_maxiter=1200, preconditioner="diag",
                blas_threads=base_fit_kwargs.get("blas_threads", "auto"),
            )
            if _meta(pred).get("cg_info", 0) != 0 or not bool(_meta(pred).get("success", True)):
                raise _InvalidTrial("prediction solver did not converge")
            losses, cells = _standardized_losses(
                Y, pred["phenotype_prediction"], kind, train,
            )
            if not np.isfinite(losses["balanced"]):
                raise _InvalidTrial("non-finite prediction CV score")
            fold_scores.append(losses["balanced"])
            cell_losses.append(cells)
            for regime in regime_scores:
                regime_scores[regime].append(losses[regime])
            runtimes.append(time.perf_counter() - started)
        evaluated = {
            "score": float(np.mean(fold_scores)), "fold_scores": fold_scores,
            "regime_scores": {k: float(np.mean(v)) for k, v in regime_scores.items()},
            "cell_losses": cell_losses, "mean_seconds": float(np.mean(runtimes)),
            "config": dict(config),
        }
        cache[key] = evaluated
        return evaluated

    incumbent = incumbent_config(
        rank_G=base_fit_kwargs.get("rank_G", "full"),
        rank_E=base_fit_kwargs.get("rank_E", "full"),
    )
    try:
        incumbent_eval = evaluate(incumbent)
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"prediction CV incumbent could not be scored: {exc}") from exc

    sampler = optuna.samplers.TPESampler(seed=seed + 101, n_startup_trials=8)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for params in _anchors():
        study.enqueue_trial(params, skip_if_exists=True)

    def objective(trial):
        """Evaluate the optimization objective for the current parameter vector.

        Parameters
        ----------
        trial : optuna.Trial
            Active Optuna trial providing suggest_* methods for this training-only hyperparameter
            search. Trial state records the chosen configuration and fit/evaluation diagnostics.

        Returns
        -------
        object
            Computed result from ``ev['score']``; see the summary and the called implementation for its labeled/numerical representation.
        """
        config = _trial_config(trial, Y.shape[1])
        trial.set_user_attr("config", config)
        try:
            ev = evaluate(config)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            raise _InvalidTrial(str(exc)) from exc
        trial.set_user_attr("fold_scores", ev["fold_scores"])
        trial.set_user_attr("regime_scores", ev["regime_scores"])
        trial.set_user_attr("mean_seconds", ev["mean_seconds"])
        return ev["score"]

    study.optimize(objective, n_trials=n_trials, catch=(_InvalidTrial,), show_progress_bar=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None and np.isfinite(t.value)]
    best_trial = min(complete, key=lambda t: float(t.value)) if complete else None
    best_config = best_trial.user_attrs["config"] if best_trial else incumbent
    best_eval = evaluate(best_config)
    relative_gain = ((incumbent_eval["score"] - best_eval["score"])
                     / max(abs(incumbent_eval["score"]), 1e-12))
    wins = int(np.sum(np.asarray(best_eval["fold_scores"])
                      < np.asarray(incumbent_eval["fold_scores"])))
    required_wins = math.ceil(2 * cv / 3)
    ci = _paired_cell_bootstrap(best_eval["cell_losses"], incumbent_eval["cell_losses"], seed + 8181)
    promoted = bool(best_trial is not None and relative_gain >= 0.005
                    and wins >= required_wins and ci[1] < 0)
    selected = dict(best_config if promoted else incumbent)
    rows = [{
        "number": int(t.number), "state": t.state.name,
        "value": float(t.value) if t.value is not None and np.isfinite(t.value) else None,
        "params": dict(t.params), "config": t.user_attrs.get("config"),
        "fold_scores": t.user_attrs.get("fold_scores"),
        "regime_scores": t.user_attrs.get("regime_scores"),
        "mean_seconds": t.user_attrs.get("mean_seconds"),
        "failure": t.user_attrs.get("failure"),
    } for t in study.trials]
    metadata = {
        "enabled": True, "cv": cv, "seed": seed,
        "objective": "mean across folds of equal-state, macro-trait training-standardized MSE for unseen-animal and partial-cell prediction",
        "rank_policy": "common rank_G=rank_E searched on a bounded categorical grid; rank_selection=none inside folds",
        "rank_candidates": _rank_choices(Y.shape[1]),
        "n_trials": len(rows), "successful_trials": len(complete),
        "incumbent": incumbent, "incumbent_score": incumbent_eval["score"],
        "incumbent_fold_scores": incumbent_eval["fold_scores"],
        "incumbent_regime_scores": incumbent_eval["regime_scores"],
        "best_candidate": best_config, "best_candidate_score": best_eval["score"],
        "best_candidate_fold_scores": best_eval["fold_scores"],
        "best_candidate_regime_scores": best_eval["regime_scores"],
        "relative_gain": float(relative_gain), "fold_wins": wins,
        "required_fold_wins": required_wins, "paired_cell_loss_ci95": ci,
        "bootstrap_description": "1000 paired cell resamples stratified by fold/state/trait; conditional on CV search, not outer-test uncertainty",
        "min_relative_gain": 0.005, "promoted": promoted,
        "selected": selected, "trials": rows,
        "fold_holdout_counts": [{r: int(np.sum(kind == r))
                                 for r in ("unseen_animal", "partial_cell")}
                                for _, kind in folds],
    }
    return selected, metadata
