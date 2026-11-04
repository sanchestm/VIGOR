"""Leakage-safe sparse G+E covariance selection for high-level VIGOR prediction.

The input phenotype matrix contains *training observations only*; all external
test values must already be NaN. Selection holds out entire training animals,
compares genetic_mean predictions to masked training phenotypes, and freezes
parameters before the final fit. This is not a causal-genetic-value validation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RANKS = (3, 5, 8, 12, "full")


def _rank_grid(rank_candidates, traits):
    """Validate sparse factor ranks and retain a full-rank fallback.
    
    Parameters
    ----------
    rank_candidates : tuple or list
        Requested sparse covariance factor ranks or the full-rank option.
    traits : int
        Number of observed trait columns constraining permitted factor ranks.
    
    Returns
    -------
    list
        Deduplicated allowable integer ranks and an optional full marker.
    
    Complexity
    ----------
    TIME O(h²); SPACE O(h), for h supplied ranks; deduplication uses a list membership scan. 'full' is appended if absent; integer ranks above trait count are skipped.
    """
    if not isinstance(rank_candidates, (tuple, list)) or not rank_candidates:
        raise ValueError("covariance_cv_ranks must be a nonempty tuple or list")
    ranks = []
    for r in rank_candidates:
        if r == "full":
            v = "full"
        elif isinstance(r, (int, np.integer)) and not isinstance(r, (bool, np.bool_)) and 1 <= r <= traits:
            v = int(r)
        else:
            if isinstance(r, int) and r > traits:
                continue
            raise ValueError("covariance_cv_ranks must be positive integer ranks or 'full'")
        if v not in ranks:
            ranks.append(v)
    if "full" not in ranks:
        ranks.append("full")  # unconditionally retain a full-rank sparse baseline
    return ranks


def _folds(Y, n_splits, seed):
    """Yield training-only whole-animal folds without external phenotypes.
    
    Parameters
    ----------
    Y : pandas.DataFrame
        Training animal-by-trait phenotypes with unavailable cells as NaN.
    n_splits : int
        Number of animal folds used to calculate unseen-animal validation error.
    seed : int
        Reproducible random seed used for permutation of training animal rows.
    
    Yields
    ------
    tuple
        Fold-specific masked training frame and heldout indicator matrix.
    
    Complexity
    ----------
    TIME O(f nt); SPACE O(nt) while yielding one fold, or O(f nt) when all yields are retained by the caller. n is animals, t traits, f folds; only phenotype-observed animals are split.
    """
    observed = np.isfinite(Y.to_numpy(float))
    eligible = np.flatnonzero(observed.any(axis=1))
    if n_splits < 2 or n_splits > len(eligible):
        raise ValueError("covariance_cv requires >=2 folds and enough phenotype-observed animals")
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    for hold in np.array_split(eligible, n_splits):
        mask = np.zeros(len(Y), dtype=bool)
        mask[hold] = True
        train = Y.copy()
        train.iloc[hold, :] = np.nan
        if (train.notna().sum(axis=0) < 2).any():
            raise ValueError("each trait needs >=2 training observations in every CV fold")
        yield train, observed & mask[:, None]


def _prediction_error(true_y, train_y, heldout, pred):
    """Calculate equal-trait genetic-mean predictive error for heldout animals.
    
    Parameters
    ----------
    true_y : pandas.DataFrame
        Original training phenotype values withheld during the given inner fold.
    train_y : pandas.DataFrame
        Fold-restricted phenotype observations used to estimate trait scales.
    heldout : numpy.ndarray
        Boolean animal-by-trait mask identifying fully unseen validation entries.
    pred : pandas.DataFrame or numpy.ndarray
        Genetic-mean predictions for all animals using masked fold training data.
    
    Returns
    -------
    float
        Equal-trait mean training-standardized squared predictive error.
    
    Complexity
    ----------
    TIME O(nt); SPACE O(nt) for coercions and per-trait masks, n animals and t traits. It measures standardized phenotype error, not error against known genetic values.
    """
    sd = train_y.std(axis=0, ddof=0).to_numpy(float)
    mean = train_y.mean(axis=0).to_numpy(float)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    y = true_y.to_numpy(float)
    p = np.asarray(pred, float)
    if p.shape != y.shape:
        raise ValueError("predictions must match input animal and trait order")
    individual = []
    for j in range(y.shape[1]):
        mask = heldout[:, j]
        if np.any(mask):
            if not np.isfinite(p[mask, j]).all():
                raise FloatingPointError("nonfinite fully unseen predictions")
            individual.append(float(np.mean(((p[mask, j] - y[mask, j]) / sd[j]) ** 2)))
    if not individual:
        raise ValueError("no evaluable unseen-animal traits in CV fold")
    return float(np.mean(individual))


def fit_sparse_prediction_covariance(
    traits, grm, *, covariance_cv=2, optimize_hyperparameters_cv=0,
    covariance_cv_ranks=DEFAULT_RANKS, covariance_kwargs=None,
    hpo_trials=12, random_state=20261015, blas_threads="auto",
):
    """Fit sparse-FA G and E with training-only CV prioritizing unseen animals.
    
    Parameters
    ----------
    traits : pandas.DataFrame or 2D array, shape (n, t)
        Training traits in GRM animal order; NaN marks unavailable/outer holdout
        cells. Each trait needs at least two observations in every inner fold.
    grm : dense array or supported single-kernel operator, shape (n, n)
        Relationship kernel in trait row order. Named multi-kernel mappings raise
        NotImplementedError; no phenotype from unseen rows is used.
    covariance_cv : int
        Number of training-only animal folds (default 2), even without Optuna.
    optimize_hyperparameters_cv : int
        0 uses a deterministic sparse rank grid. >=2 also searches sparse
        regularization/rank configurations with Optuna; the requested value is
        the training-only fold count.
    covariance_cv_ranks : sequence
        Sparse-FA factor-rank candidates; includes full-rank fallback.
        Default: ``DEFAULT_RANKS``.
    covariance_kwargs : dict
        Project-level estimator options (e.g. maxiter, ridge_E, shrink_E).
        Reserved rank/structure/CV keys are managed by this function.
        Default: ``None``.
    hpo_trials : int
        Number of Optuna configurations, if enabled (>=1). Default: ``12``.
    random_state : int
        Reproducible fold assignment, optimizer and Optuna sampling.
        Default: ``20261015``.
    blas_threads : int or 'auto'
        BLAS thread limit passed to fit and prediction.
    
    Returns
    -------
    xarray.Dataset
        genetic_varcov_PSD fit (same Dataset layout) with the prediction_sparse_cv
        candidate/failure report, prediction_scale and inference_scale in attrs.
        G_prediction/E_prediction are restored to original input trait units;
        inference G/E/native factors remain standardized fitting units.
    
    Complexity
    ----------
    TIME O(H f (C_fit+C_pred+nt)+a C_fit); SPACE O(f nt+S_fit+S_pred+Hf+t²), retaining all fold frames. n is animals, t traits, H evaluated candidates (rank grid plus enabled Optuna trials), f folds, a attempted final refits. C_fit/S_fit are genetic_varcov_PSD work/peak storage; C_pred/S_pred are mvPredict work/peak storage for one fold (dense kernels require O(n²)). Candidates run serially; convergence changes actual evaluation counts. All fold scaling and ranking are training-only; no outer truth is accepted.
    """
    # Inner folds use the undecorated estimator/predictor (legacy containers);
    # the selected fit is converted to an xarray.Dataset once at the end.
    from .multivariate import (_genetic_varcov_PSD_raw as genetic_varcov_PSD, _mvPredict_raw as mvPredict,
                               _covariance_fit_dataset, _as_pandas_input)
    from .hyperparameter_cv import validate_cv_count
    traits = _as_pandas_input(traits)
    y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    if y.ndim != 2 or y.shape[1] < 1:
        raise ValueError("traits must be an animal by trait matrix")
    if not np.isfinite(y.to_numpy(float)).any():
        raise ValueError("no training phenotypes available")
    if not isinstance(covariance_cv, (int, np.integer)) or isinstance(covariance_cv, (bool, np.bool_)):
        raise TypeError("covariance_cv must be integer")
    cv_opt = validate_cv_count(optimize_hyperparameters_cv)
    cv = cv_opt if cv_opt else int(covariance_cv)
    if cv < 2:
        raise ValueError("sparse covariance selection requires covariance_cv>=2")
    if isinstance(grm, Mapping):
        # Existing multi-kernel paths may use a different genotype design;
        # fail visibly rather than silently treating a named kernel as additive.
        raise NotImplementedError("automatic sparse prediction selection currently requires a single GRM; supply a pre-fitted multi-kernel covariance")
    if not isinstance(hpo_trials, int) or isinstance(hpo_trials, bool) or hpo_trials < 1:
        raise ValueError("hpo_trials must be a positive integer")
    ranks = _rank_grid(covariance_cv_ranks, y.shape[1])
    opts = dict(covariance_kwargs or {})
    managed = {"rank_G", "rank_E", "factor_structure_G", "factor_structure_E",
               "factor_lambda_G", "factor_lambda_E", "optimize_hyperparameters_cv",
               "rank_selection", "n_boot", "analytic_se", "progress", "random_state"}
    bad = managed & opts.keys()
    if bad:
        raise ValueError(f"managed sparse prediction selection options cannot be overridden in covariance_kwargs: {sorted(bad)}")
    seed = int(random_state)
    base = dict(shrink_E=0.5263454565339929, shrink_G="auto",
                covariance_structure="fa", solver="lbfgs", maxiter=1500,
                factor_rotation_G="varimax", factor_rotation_E="varimax",
                factor_auto_fraction=0.05, factor_adaptive=True,
                factor_relaxed_refit=True, E_structure="auto")
    base.update(opts)
    common = dict(base, optimize_hyperparameters_cv=0,
                  rank_selection="none", n_boot=0, analytic_se=False,
                  progress=False, random_state=seed, blas_threads=blas_threads)
    folds = list(_folds(y, cv, seed))
    history = []
    cache = {}

    def evaluate(config):
        """Evaluate one sparse candidate on all training-only animal folds.

        Parameters
        ----------
        config : dict
            Explicit sparse factor ranks and regularization settings to evaluate.

        Returns
        -------
        dict
            Average heldout unseen-animal loss and individual fold losses.
        """
        key = repr(sorted(config.items()))
        if key in cache:
            return cache[key]
        errors = []
        for Ytrain, held in folds:
            # No heldout phenotypes enter this fit. The encoded Ytrain is
            # independently re-standardized on its observed training cells.
            mean = Ytrain.mean()
            sd = Ytrain.std(ddof=0).replace(0, 1).fillna(1.0)
            ztrain = (Ytrain - mean) / sd
            zref = (y - mean) / sd
            kw = dict(common, **config)
            fit = genetic_varcov_PSD(grm, ztrain, **kw)
            if not bool(fit.get("optimizer_converged", fit.get("success", False))):
                raise RuntimeError("sparse covariance fit did not converge: " + str(fit.get("message")))
            pred = mvPredict(ztrain, grm, fit["G_prediction"],
                             E=fit["E_prediction"], accuracy="exact",
                             solver_tol=1e-6, solver_maxiter=1200,
                             preconditioner="diag", blas_threads=blas_threads)
            if int(pred.get("cg_info", 0)) != 0:
                raise RuntimeError("prediction solver did not converge")
            errors.append(_prediction_error(zref, ztrain, held, pred["genetic_mean"]))
        return dict(score=float(np.mean(errors)), fold_scores=errors)

    def candidate(rank, **extras):
        """Construct a sparse G+E factor-analytic candidate.

        Parameters
        ----------
        rank : int or str
            Common genomic and residual covariance factor rank for this candidate.
        **extras : dict
            Optional additional covariance regularization requests for a trial.

        Returns
        -------
        dict
            Complete sparse covariance geometry and rank candidate settings.
        """
        return dict(rank_G=rank, rank_E=rank,
                    factor_structure_G="sparse_fa", factor_structure_E="sparse_fa",
                    factor_lambda_G="auto", factor_lambda_E="auto", **extras)

    def record(config, source):
        """Score and audit a candidate without treating failed fits as successful.

        Parameters
        ----------
        config : dict
            Candidate sparse-FA settings to fit on the recorded training folds.
        source : str
            Selection-stage origin, either rank-grid initialization or Optuna.

        Returns
        -------
        dict
            Valid score or an explicit failure record for the candidate.
        """
        try:
            result = evaluate(config)
            row = dict(state="PASS", source=source, config=dict(config), **result)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row = dict(state="FAIL", source=source, config=dict(config), error=str(exc))
        history.append(row)
        return row

    for rank in ranks:
        record(candidate(rank), "rank_grid")
    success = [row for row in history if row["state"] == "PASS"]
    if not success:
        raise RuntimeError("all sparse G+E ranks failed CV; no silent dense-FA fallback")
    best = min(success, key=lambda row: row["score"])
    optuna_status = "disabled"
    if cv_opt:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("project-specific Optuna requires pip install 'VIGOR[hpo]'") from exc
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        choices = [row["config"]["rank_G"] for row in success]
        def objective(trial):
            """Evaluate the Optuna proposal against heldout training-only animals.

            Parameters
            ----------
            trial : optuna.Trial
                Optuna suggestion object for rank and regularization candidates.

            Returns
            -------
            float
                Mean heldout genetic-mean predictive squared error.
            """
            rank = trial.suggest_categorical("factor_rank", choices)
            e = trial.suggest_float("residual_retention", 0.1, 0.9)
            g = trial.suggest_float("genetic_retention", 0.3, 1.0)
            frac = trial.suggest_float("sparsity_fraction", 0.01, 0.18, log=True)
            row = record(candidate(rank, shrink_E=e, shrink_G=g, factor_auto_fraction=frac), "optuna")
            trial.set_user_attr("candidate", row)
            if row["state"] == "FAIL":
                raise optuna.TrialPruned(row["error"])
            return row["score"]
        study.optimize(objective, n_trials=hpo_trials, show_progress_bar=False)
        success = [row for row in history if row["state"] == "PASS"]
        best = min(success, key=lambda row: row["score"])
        optuna_status = "completed"
    mean = y.mean()
    sd = y.std(ddof=0).replace(0, 1).fillna(1.0)
    # Freeze candidate ranking using *training CV only*. A selected sparse
    # candidate may fail on a full-data refit: try the next CV-ranked sparse
    # candidate, recording failures, never silently switching to dense FA.
    refit_attempts = []
    fit = None
    chosen = None
    for row in sorted((r for r in history if r["state"] == "PASS"), key=lambda r: r["score"]):
        cfg = dict(row["config"])
        try:
            proposal = genetic_varcov_PSD(grm, (y-mean)/sd, **dict(common, **cfg))
            if not bool(proposal.get("optimizer_converged", proposal.get("success", False))):
                raise RuntimeError(str(proposal.get("message", "no optimizer convergence")))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            refit_attempts.append({"config": cfg, "state": "FAIL", "error": str(exc)})
            continue
        fit = proposal
        chosen = cfg
        refit_attempts.append({"config": cfg, "state": "PASS"})
        break
    if fit is None:
        raise RuntimeError("all CV-successful sparse candidates failed final refit; no silent dense fallback")
    # Predictions use training-standardized covariance, so convert back to the
    # original trait scale before returning the public fit.
    # Scaled covariances are required because mvBLUP/mvPredict receive raw Y.
    G = np.asarray(fit["G_prediction"], dtype=float)
    E = np.asarray(fit["E_prediction"], dtype=float)
    scale = sd.to_numpy(float)
    fit["G_prediction"] = pd.DataFrame(G*scale[:,None]*scale[None,:],
                                      index=y.columns, columns=y.columns)
    fit["E_prediction"] = pd.DataFrame(E*scale[:,None]*scale[None,:],
                                      index=y.columns, columns=y.columns)
    # The public predictor chooses the factor specs over G_prediction/E_prediction
    # when handed a covariance-fit Series. Update BOTH representations so a
    # reused fit always produces the same posterior as an explicit G/E call.
    if isinstance(fit, pd.Series):
        gmat = fit["G_prediction"]
        emat = fit["E_prediction"]
        fit["component_specs_prediction"] = {"genetic": gmat}
        fit["E_spec_prediction"] = emat
        fit["G_components_prediction"] = {"genetic": gmat}
    # The inference-facing fitted MoM G/E and analytical-SE objects remain
    # in training-standardized trait units. Document units explicitly rather
    # than inadvertently changing inferential scientific estimands.
    fit["prediction_scale"] = "original_trait_units"
    fit["inference_scale"] = "inner_train_standardized_trait_units"
    fit["prediction_sparse_cv"] = dict(
        status="COMPLETE", objective="unseen_animal_genetic_mean_standardized_MSE",
        cv=cv, training_only=True, chosen=chosen, ranks=ranks,
        candidates=history, optuna=optuna_status,
        refit_attempts=refit_attempts,
        trait_means={str(k):float(v) for k,v in mean.items()},
        trait_sds={str(k):float(v) for k,v in sd.items()},
        note="Objective is prediction of phenotype from genetic mean, not measured true breeding value.",
    )
    return _covariance_fit_dataset(fit, dict(covariance_cv=covariance_cv, optimize_hyperparameters_cv=optimize_hyperparameters_cv,
                                             covariance_cv_ranks=covariance_cv_ranks, hpo_trials=hpo_trials,
                                             random_state=random_state, blas_threads=blas_threads),
                                   software="vigor.prediction_sparse_cv.fit_sparse_prediction_covariance",
                                   title="Sparse factor-analytic covariance selected for unseen-animal prediction")
