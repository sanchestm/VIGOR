"""Opt-in prediction-based rank selection for the covariance fitter.

Heritability calibration is part of the PSD/FA objective itself (see
:mod:`vigor.h2_prior` and ``genetic_varcov_PSD(h2_prior_strength=...)``);
this module only selects a common G/E factor rank by hidden-phenotype
prediction before the final fit.
"""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from .progress import report_progress


def _traits_frame(traits):
    frame = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    if frame.ndim != 2 or not len(frame) or not len(frame.columns):
        raise ValueError("traits must be a nonempty animal x trait matrix")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("animal and trait labels must be unique")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("traits may contain NaN, but not infinity")
    return frame


def _prediction_frame(result, reference):
    prediction = result["phenotype_prediction"]
    if hasattr(prediction, "to_pandas"):
        prediction = prediction.to_pandas()
    if isinstance(prediction, pd.DataFrame):
        return prediction.loc[reference.index, reference.columns]
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != reference.shape:
        raise ValueError("prediction shape must match traits")
    return pd.DataFrame(values, index=reference.index, columns=reference.columns)


def select_prediction_rank(
    grm, traits, *, fit_fn, predict_fn, covariance_kwargs=None,
    rank_candidates=(10, 20, 40, 80, "full"), validation_fraction=0.20,
    random_state=0, min_validation=15,
):
    """Choose a common G/E factor rank using hidden phenotype predictions.

    Parameters
    ----------
    grm, traits : kernels and animal x trait outcomes
        Same aligned inputs supplied to the covariance estimator. Only observed
        cells are candidates; at least one phenotype remains on every hidden row.
    fit_fn, predict_fn : callable
        ``fit_fn(grm, train, **kwargs)`` and
        ``predict_fn(train, grm, fit, **kwargs)``. A raw mapping or xarray result
        is supported. Predictions must contain ``phenotype_prediction``.
    covariance_kwargs : mapping, optional
        Fixed fitting settings. Warm starts, persistence, nested HPO/rank CV,
        bootstrap and analytic SEs are disabled in inner fits. The h2 prior
        settings are kept, so ranks are compared for the model being deployed.
    rank_candidates : sequence of positive int or 'full'
        Common rank for every named G component and E, capped at trait count.
        Duplicate effective ranks are removed. Equal scores retain the first.
    validation_fraction : float in (0, 1), default .2
        Fraction of rows with at least two observed traits eligible for hiding;
        half their observed traits are hidden, using one shared split for ranks.
    random_state : int or None, default 0
        Split seed; None uses a nondeterministic NumPy generator.
    min_validation : int >= 3, default 15
        Minimum held-out cells per scored trait. Traits are fixed across ranks.

    Returns
    -------
    dict
        ``rank_G``, ``rank_E``, ``selected_rank``, per-rank mean phenotype
        correlation ``scores``, heldout counts and scored trait labels. A final
        all-data fit is the caller's responsibility. This is an inner selection
        score, not an unbiased estimate of final prediction performance.

    Complexity
    ----------
    One covariance fit plus one prediction per candidate rank. Memory is the
    callback working memory plus O(n t) masked outcomes and predictions.
    """
    frame = _traits_frame(traits)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if isinstance(min_validation, bool) or not isinstance(min_validation, numbers.Integral) or min_validation < 3:
        raise ValueError("min_validation must be an integer >=3")
    candidates = []
    for rank in rank_candidates:
        if rank == "full":
            value = frame.shape[1]
        elif isinstance(rank, numbers.Integral) and not isinstance(rank, bool) and rank > 0:
            value = min(int(rank), frame.shape[1])
        else:
            raise ValueError("rank candidates must be positive integers or 'full'")
        if value not in candidates:
            candidates.append(value)
    if not candidates:
        raise ValueError("rank_candidates cannot be empty")
    rng = np.random.default_rng(random_state)
    observed = np.isfinite(frame.to_numpy(dtype=np.float64))
    eligible = np.flatnonzero(observed.sum(axis=1) >= 2)
    if not len(eligible):
        raise ValueError("prediction rank CV requires rows with two observed traits")
    rows = rng.choice(eligible, max(1, int(validation_fraction * len(eligible))), replace=False)
    hidden = np.zeros(frame.shape, dtype=bool)
    for i in rows:
        available = np.flatnonzero(observed[i])
        hidden[i, rng.choice(available, len(available) // 2, replace=False)] = True
    train = frame.mask(hidden)
    if (train.notna().sum() < 3).any():
        raise ValueError("prediction rank CV requires three training observations per trait")
    scored = hidden.sum(axis=0) >= min_validation
    truth = frame.to_numpy(dtype=np.float64)
    # Constant validation traits cannot define a correlation. Eligibility is
    # determined once, before seeing any candidate prediction.
    for j in np.flatnonzero(scored):
        if np.std(truth[hidden[:, j], j]) == 0:
            scored[j] = False
    if not scored.any():
        raise ValueError("no trait has enough nonconstant held-out outcomes; increase validation_fraction or provide more observations")
    kwargs = dict(covariance_kwargs or {})
    for key in ("save_path", "_runtime_progress"):
        kwargs.pop(key, None)
    kwargs.update(rank_selection="none", optimize_hyperparameters_cv=0,
                  warm_start=None, analytic_se=False, n_boot=0, progress=False,
                  structural_plots=False)
    scores, diagnostics = {}, {}
    for candidate_index,rank in enumerate(candidates,1):
        report_progress(kind='rank_cv',phase='Rank CV',phase_key='rank_cv',
                        candidate_index=candidate_index,candidate_total=len(candidates),rank=rank)
        local = dict(kwargs, rank_G=rank, rank_E=rank)
        fit = fit_fn(grm, train, **local)
        info = fit.attrs if hasattr(fit, "data_vars") else fit
        if not bool(info.get("success", True)) or not bool(info.get("optimizer_converged", True)):
            raise RuntimeError(f"rank {rank}: covariance fit did not converge")
        prediction = predict_fn(train, grm, fit, accuracy="exact",
                                solver_tol=1e-6, solver_maxiter=1200,
                                blas_threads=kwargs.get("blas_threads", "auto"))
        prediction_info = prediction.attrs if hasattr(prediction, "data_vars") else prediction
        if prediction_info.get("cg_info", 0) != 0 or not bool(prediction_info.get("success", True)):
            raise RuntimeError(f"rank {rank}: prediction solver did not converge")
        predicted = _prediction_frame(prediction, frame).to_numpy(dtype=np.float64)
        correlations = []
        for j in np.flatnonzero(scored):
            p = predicted[hidden[:, j], j]
            y = truth[hidden[:, j], j]
            if not np.isfinite(p).all():
                raise FloatingPointError(f"rank {rank}: non-finite held-out predictions")
            # A constant prediction carries no ranking information; use zero
            # instead of silently dropping a difficult trait for that rank.
            correlations.append(0.0 if np.std(p) == 0 else float(np.corrcoef(p, y)[0, 1]))
        scores[rank] = float(np.mean(correlations))
        diagnostics[rank] = correlations
    best = max(scores, key=scores.get)
    report_progress(kind='rank_cv',phase='Rank CV complete',phase_key='rank_cv',finished=True,
                    candidate_index=len(candidates),candidate_total=len(candidates),rank=best)
    return {"rank_G": best, "rank_E": best, "selected_rank": best,
            "scores": scores, "trait_correlations": diagnostics,
            "scored_traits": frame.columns[scored].tolist(),
            "heldout_per_trait": hidden.sum(axis=0).tolist(),
            "hidden_cells": int(hidden.sum()), "random_state": random_state,
            "criterion": "mean_heldout_phenotype_correlation",
            "uncertainty": "conditional inner-split selection; use an outer split for performance evaluation"}



__all__ = ["select_prediction_rank"]


def prediction_rank_options(function):
    """Add opt-in training-only prediction rank CV around the raw covariance fitter.

    rank_selection='prediction_cv' selects a common rank by hidden-phenotype
    prediction (:func:`select_prediction_rank`) and then runs the final fit
    with that rank. prediction_rank_kwargs are forwarded to the selector.
    Wrapper cost is O(1) when rank CV is not requested.
    """
    import functools
    import inspect
    signature = inspect.signature(function)

    @functools.wraps(function)
    def extended(*args, prediction_rank_kwargs=None, **kwargs):
        use_rank_cv = kwargs.get("rank_selection", "none") == "prediction_cv"
        if prediction_rank_kwargs and not use_rank_cv:
            raise ValueError("prediction_rank_kwargs require rank_selection='prediction_cv'")
        if not use_rank_cv:
            return function(*args, **kwargs)
        if kwargs.get("compatibility") is not None or kwargs.get("environment") is not None:
            raise ValueError("rank prediction CV requires the standard named-kernel covariance route")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        legacy = params.pop("legacy_kwargs", {})
        params.update(legacy)
        grm, traits = params.pop("grm"), params.pop("traits")
        from . import multivariate as mv
        grm = mv._kernel_input(grm)
        traits = _traits_frame(mv._as_pandas_input(traits))
        params.pop("save_path", None)
        if params.get("optimize_hyperparameters_cv", 0):
            raise ValueError("choose prediction rank CV or Optuna in one fit; nested searches are not implicit")
        rank_kw = dict(prediction_rank_kwargs or {})
        rank_kw.setdefault("rank_candidates", params["rank_candidates"])
        rank_kw.setdefault("validation_fraction", params["rank_validation_fraction"])
        rank_kw.setdefault("random_state", params.get("random_state"))
        selection = select_prediction_rank(grm, traits, fit_fn=extended,
            predict_fn=mv._mvPredict_raw, covariance_kwargs=params, **rank_kw)
        params.update(rank_G=selection["rank_G"], rank_E=selection["rank_E"], rank_selection="none")
        out = dict(function(grm, traits, **params))
        out["prediction_rank_selection"] = selection
        return out

    additions = [inspect.Parameter("prediction_rank_kwargs", inspect.Parameter.KEYWORD_ONLY, default=None)]
    parameters = list(signature.parameters.values())
    position = next((i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD), len(parameters))
    extended.__signature__ = signature.replace(parameters=parameters[:position] + additions + parameters[position:])
    extended.__doc__ = (function.__doc__ or "") + "\n\nRank selection extension\n------------------------\nrank_selection='prediction_cv' selects a common G/E rank by training-only hidden-phenotype prediction before the final fit; prediction_rank_kwargs are forwarded to select_prediction_rank.\n"
    return extended
