"""Prediction-oriented mvBLUP residual experts for VIGOR 2.12.1.

This module preserves the successful gdec 3.1.3 residual-expert design while
making the prediction target explicit:

* animals with no phenotypes -> genomic-only residual expert;
* partially observed animals -> phenotype-assisted or matrix-completion expert;
* fully observed rows -> diagnostic application of the same partial-row expert.

Cross-fitting uses the current 2.12.1 covariance/mvBLUP engine rather than the
archived compatibility estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import multivariate as _compat
from .multivariate import genetic_varcov_PSD, mvBLUP
from .residual_matrix import fit_residual_matrix_expert, predict_residual_matrix_expert
from .residual_matrix import (_ST, _TT, _call_parameters, _mapping_dataset, _pandas,
                              _result_dict)
from .results import maybe_save


def _as_frame(x, like):
    """Convert inputs to frame.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x.reindex(index=like.index, columns=like.columns)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    like : array-like, table or fitted object
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields. Passed to ``x.reindex`` as ``index``.
    """
    x = _pandas(x)
    if isinstance(x, pd.DataFrame):
        return x.reindex(index=like.index, columns=like.columns)
    a = np.asarray(x, dtype=float)
    if a.shape != like.shape:
        raise ValueError("BLUP shape does not match traits")
    return pd.DataFrame(a, index=like.index, columns=like.columns)


def _genetic_mean(traits, GRM, G, E=None, **kwargs):
    """Phenotype-scale baseline: fitted training intercept plus breeding value.

    Explicit caller-supplied ``blup`` arrays must use this same genetic-mean
    target. Public mvBLUP itself intentionally continues to return breeding values.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``. Passed to ``mvBLUP`` as ``E``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``mvBLUP``. See those callables for accepted
        keys and values. Passed to ``mvBLUP`` as ``**expanded mapping``.
    """
    # The public mvBLUP returns a Dataset; experts need the labeled breeding-value table.
    breeding=_compat._mvBLUP_raw(traits,GRM,G,E=E,**kwargs)
    means=traits.mean(axis=0).fillna(0.)
    return _as_frame(breeding,traits)+means


def _conditional_mean(traits, GRM, G, E=None, *, solver_tol=1e-7,
                      solver_maxiter=500, preconditioner='diag', **_kwargs):
    """Compute observed-cell Gaussian conditional phenotype means for residual-expert fitting.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-07``. Passed to ``_compat.mvPredict`` as ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``500``. Passed to ``_compat.mvPredict`` as ``solver_maxiter``.

    preconditioner : str
        Input supplied to ``_compat.mvPredict`` as ``preconditioner``. See that callable's
        documented ``preconditioner`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``'diag'``.

    **_kwargs : dict
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.
    """
    return _pandas(_compat.mvPredict(traits,GRM,G,E,accuracy='exact',kernel_rank='full',
        solver_tol=solver_tol,solver_maxiter=solver_maxiter,
        preconditioner=preconditioner)['phenotype_prediction'])


def _resolve_E(G, E):
    """Resolve E.

    Parameters
    ----------
    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance.
    """
    if E is not None:
        return E
    if isinstance(G, pd.Series) and "E" in G:
        return G.get("E_prediction", G["E"])
    if hasattr(G, "data_vars") and "E" in G.data_vars:
        return _pandas(G["E_prediction"] if "E_prediction" in G.data_vars else G["E"])
    return E


def fit_blup_residual_experts(
    traits, GRM, G, E=None, *, blup=None, n_splits=3,
    methods=("pcr", "pls", "elasticnet", "lightgbm"), n_components=8,
    random_state=0, stack_min_gain=.02, genomic_min_gain=.03,
    phenotype_min_gain=.005, genomic_min_z=2.33, phenotype_min_z=1.96,
    genomic_features=None, model_kwargs=None, solver="pcg", solver_tol=2e-6,
    solver_maxiter=500, preconditioner="diag", crossfit_refit_covariance=True,
    covariance_kwargs=None, allow_blup_only_nonlinear=False, store_oof=False,
    fit_genomic_per_trait=True, fit_phenotype_per_trait=True, selection_cv=3,
    partial_baseline='genetic_mean',fit_multitask_genomic=True, save_path=None,
):
    """Cross-fitted 3.1-style residual experts using the 2.12.1 covariance engine.

    Baselines and explicitly supplied ``blup`` use intercept + breeding value.
    Row-held-out folds train the genomic-only expert used for fully unseen
    animals. Cell-held-out folds train the phenotype-assisted expert. When
    ``crossfit_refit_covariance=True`` every fold refits covariance with the
    held-out phenotypes removed using :func:`genetic_varcov_PSD` from 2.12.1.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    blup : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``.

    methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('pcr', 'pls', 'elasticnet', 'lightgbm')``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``8``. Passed to ``_compat._expert_features`` as ``n_ge_components``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``_compat._fit_global_residual_models`` as ``random_state``.

    stack_min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.02``. Passed to
        ``_compat._fit_global_residual_models`` as ``stack_min_gain``.

    genomic_min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.03``. Passed to
        ``_compat._fit_global_residual_models`` as ``model_min_gain``.

    phenotype_min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.005``. Passed to
        ``_compat._fit_global_residual_models`` as ``model_min_gain``.

    genomic_min_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``2.33``. Passed to ``_compat._fit_global_residual_models`` as ``model_min_z``.

    phenotype_min_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``1.96``. Passed to ``_compat._fit_global_residual_models`` as ``model_min_z``.

    genomic_features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value. Passed to
        ``_compat._expert_features`` as ``genomic_features``.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    solver : str
        Input supplied to ``_genetic_mean`` as ``solver``. See that callable's documented
        ``solver`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``'pcg'``.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``2e-06``. Passed to ``_genetic_mean`` as ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``500``. Passed to ``_genetic_mean`` as ``solver_maxiter``.

    preconditioner : str
        Input supplied to ``_genetic_mean`` as ``preconditioner``. See that callable's
        documented ``preconditioner`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``'diag'``.

    crossfit_refit_covariance : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    allow_blup_only_nonlinear : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    store_oof : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    fit_genomic_per_trait : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    fit_phenotype_per_trait : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    selection_cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``_compat._fit_global_residual_models`` as ``selection_cv``.

    partial_baseline : str
        Partial-trait baseline: 'conditional' includes observed-cell conditional prediction. See
        the fitting routine for the other baseline policies it explicitly supports. Default:
        ``'genetic_mean'``.

    fit_multitask_genomic : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``blup`` (sample, trait) and ``G``/``E`` (trait_i, trait_j). ``attrs``
        hold ``G_components``, the echoed model inputs ``covariance_model``, ``E_model``,
        ``GRM_model`` and ``genomic_features``, the fitted ``genomic_models``,
        ``phenotype_models`` and ``multitask_genomic`` (in memory; saved as repr),
        ``methods``, ``n_splits``, ``diagnostics``, ``n_components``, ``version``,
        ``feature_names_genomic``, ``feature_names_phenotype``, ``metadata``,
        ``model_kwargs``, ``partial_baseline``, solver settings and, with ``store_oof``,
        ``oof``. ``predict_blup_residual_experts`` accepts it.
    """
    params = _call_parameters(fit_blup_residual_experts, locals())
    traits, GRM, G, E = _pandas(traits), _pandas(GRM), _pandas(G), _pandas(E)
    genomic_features = _pandas(genomic_features)
    Y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    if partial_baseline not in {'genetic_mean','conditional'}:
        raise ValueError('partial_baseline must be genetic_mean or conditional')
    n, t = Y.shape
    E_model = _resolve_E(G, E)
    Gf, Ef, Gcomponents = _compat._residual_expert_covariances(G, E_model)
    covariance_kwargs = {} if covariance_kwargs is None else dict(covariance_kwargs)
    covariance_kwargs.pop("n_boot", None)
    mk = {} if model_kwargs is None else {k: ({} if v is None else dict(v)) for k, v in model_kwargs.items()}

    # Selected LightGBM performs outcome-dependent SNP selection within each
    # residual-model CV fold. Additive and dominance raw-panel features are
    # both candidates; BLUP/canonical/GRM-PC features are protected.
    if any(str(m).lower() in {"lightgbm_selected", "lightgbm_topk", "ridge_selected", "selected_ridge"} for m in methods) and isinstance(genomic_features, pd.DataFrame):
        base = t + min(int(n_components), t)
        gnames = list(map(str, genomic_features.columns))
        cand_local = np.asarray([i for i, x in enumerate(gnames) if x.startswith("snp_") or x.startswith("dom_")], int)
        total_base = base + len(gnames)
        candidate = base + cand_local
        protect = np.setdiff1d(np.arange(total_base), candidate)
        for method in methods:
            if str(method).lower() in {"lightgbm_selected", "lightgbm_topk", "ridge_selected", "selected_ridge"}:
                z = mk.setdefault(method, {})
                z.setdefault("candidate_indices", candidate)
                z.setdefault("protect_indices", protect)

    B = (_genetic_mean(
        Y, GRM, G, E=E_model, missing="exact", solver=solver,
        solver_tol=solver_tol, solver_maxiter=solver_maxiter,
        preconditioner=preconditioner,
    ) if blup is None else _as_frame(blup, Y))

    def fold_blup(Yk, phenotype=False):
        """Fit training-fold covariance and produce fold-safe genetic/conditional predictions.

        Parameters
        ----------
        Yk : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        phenotype : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``False``.

        Returns
        -------
        object
            Computed result from ``(_as_frame(Bk, Y), Gcan, Ecan)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if crossfit_refit_covariance:
            ge = genetic_varcov_PSD(GRM, Yk, n_boot=0, **covariance_kwargs)
            Gcan, Ecan, _ = _compat._residual_expert_covariances(ge, None)
            cov,em=ge,None
        else:
            Gcan, Ecan = Gf, Ef
            cov,em=G,E_model
        if phenotype and partial_baseline == 'conditional':
            state=_compat.mvPredict(Yk,GRM,cov,em,accuracy='exact',kernel_rank='full',
                solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner)
            return _pandas(state['genetic_mean']),Gcan,Ecan,_pandas(state['phenotype_prediction'])
        Bk = _genetic_mean(
                Yk, GRM, cov, E=em, missing="exact", solver=solver,
                solver_tol=solver_tol, solver_maxiter=solver_maxiter,
                preconditioner=preconditioner,
            )
        if phenotype:return _as_frame(Bk,Y),Gcan,Ecan,_as_frame(Bk,Y)
        return _as_frame(Bk, Y),Gcan,Ecan

    row_fold = _compat._split_rows(n, n_splits, random_state)
    gx = [[] for _ in range(t)]; gy = [[] for _ in range(t)]
    Xo = None; Ro = np.full((n, t), np.nan)
    for k in range(int(n_splits)):
        val = row_fold == k
        Yk = Y.copy(); Yk.iloc[val, :] = np.nan
        Bk, Gk, Ek = fold_blup(Yk)
        Xg = _compat._expert_features(
            Yk, Bk, Gk, Ek, phenotype_assisted=False,
            genomic_features=genomic_features, n_ge_components=n_components,
        )
        if Xo is None:
            Xo = np.full((n, Xg.shape[1]), np.nan)
        Xo[val] = Xg[val]
        Ro[val] = Y.to_numpy(float)[val] - Bk.to_numpy(float)[val]
        for j in range(t):
            rows = np.flatnonzero(val & Y.iloc[:, j].notna().to_numpy())
            if len(rows):
                gx[j].append(Xg[rows])
                gy[j].append(Y.iloc[rows, j].to_numpy(float) - Bk.iloc[rows, j].to_numpy(float))

    px = [[] for _ in range(t)]; py = [[] for _ in range(t)]
    cell_fold = None
    if fit_phenotype_per_trait:
        cell_fold = _compat._split_cells(Y.to_numpy(float), n_splits, random_state + 101)
        for k in range(int(n_splits)):
            hold = cell_fold == k
            Yk = Y.mask(hold)
            Bk, Gk, Ek, Pk = fold_blup(Yk, phenotype=True)
            for j in range(t):
                rows = np.flatnonzero(hold[:, j])
                if len(rows):
                    Xp = _compat._expert_features(
                        Yk, Bk, Gk, Ek, target=j, phenotype_assisted=True,
                        genomic_features=genomic_features, n_ge_components=n_components,
                    )
                    px[j].append(Xp[rows])
                    py[j].append(Y.iloc[rows, j].to_numpy(float) - Pk.iloc[rows, j].to_numpy(float))

    gx = [np.vstack(z) if z else None for z in gx]
    gy = [np.concatenate(z) if z else None for z in gy]
    px = [np.vstack(z) if z else None for z in px]
    py = [np.concatenate(z) if z else None for z in py]
    cols = list(Y.columns)

    if fit_genomic_per_trait and (genomic_features is not None or allow_blup_only_nonlinear):
        gm, gdiag = _compat._fit_global_residual_models(
            gx, gy, methods, random_state=random_state, n_components=n_components,
            model_min_gain=genomic_min_gain, model_min_z=genomic_min_z,
            stack_min_gain=stack_min_gain, model_kwargs=mk, selection_cv=selection_cv,
        )
        genmodels = {cols[j]: gm.get(j, {"mode":"blup", "models":{}, "weights":{}, "global":True}) for j in range(t)}
    else:
        gdiag = {"mode":"blup", "weights":{}, "scores":{}, "best_single":"blup",
                 "reason":"no additional genomic features beyond BLUP"}
        genmodels = {c: {"mode":"blup", "models":{}, "weights":{}, "global":True} for c in cols}

    if fit_phenotype_per_trait:
        pm, pdiag = _compat._fit_global_residual_models(
            px, py, methods, random_state=random_state + 1000, n_components=n_components,
            model_min_gain=phenotype_min_gain, model_min_z=phenotype_min_z,
            stack_min_gain=stack_min_gain, model_kwargs=mk, selection_cv=selection_cv,
        )
        phenmodels = {cols[j]: pm.get(j, {"mode":"blup", "models":{}, "weights":{}, "global":True}) for j in range(t)}
    else:
        pdiag = {"mode":"blup", "weights":{}, "scores":{}, "best_single":"blup",
                 "reason":"phenotype-assisted per-trait experts disabled"}
        phenmodels = {c: {"mode":"blup", "models":{}, "weights":{}, "global":True} for c in cols}

    mt = None
    try:
        ok = np.isfinite(Xo).all(1) & np.isfinite(Ro).any(1)
        if fit_multitask_genomic and (genomic_features is not None or allow_blup_only_nonlinear) and ok.sum() > 30:
            mt = _compat.MaskedMultiTaskElasticNet().fit(Xo[ok], Ro[ok])
    except Exception:
        pass

    out = {
        "blup": B, "G": pd.DataFrame(Gf, index=cols, columns=cols),
        "E": pd.DataFrame(Ef, index=cols, columns=cols),
        "G_components": {k: pd.DataFrame(v, index=cols, columns=cols) for k, v in Gcomponents.items()},
        "covariance_model": G, "E_model": E_model, "GRM_model": GRM,
        "genomic_models": genmodels, "phenotype_models": phenmodels,
        "multitask_genomic": mt, "methods": tuple(methods), "n_splits": int(n_splits),
        "diagnostics": {"genomic_global": gdiag, "phenotype_global": pdiag},
        "genomic_features": genomic_features, "n_components": int(n_components),
        "version": __import__("vigor").__version__,
        "feature_names_genomic": _compat._expert_feature_names(
            Y, Gf, Ef, phenotype_assisted=False, genomic_features=genomic_features,
            n_ge_components=n_components,
        ),
        "feature_names_phenotype": {
            str(c): _compat._expert_feature_names(
                Y, Gf, Ef, target=j, phenotype_assisted=True,
                genomic_features=genomic_features, n_ge_components=n_components,
            ) for j, c in enumerate(Y.columns)
        },
        "metadata": {
            "covariance_engine": "domino_" + __import__("vigor").__version__,
            "row_crossfit_for_genomic_unseen": True,
            "cell_crossfit_for_phenotype_assisted": True,
            "crossfit_refit_covariance": bool(crossfit_refit_covariance),
        },
        "model_kwargs": mk,
        "partial_baseline": partial_baseline,
        "solver_tol":solver_tol,"solver_maxiter":solver_maxiter,"preconditioner":preconditioner,
    }
    if store_oof:
        out["oof"] = {"genomic_X": gx, "genomic_y": gy, "phenotype_X": px, "phenotype_y": py,
                      "row_fold": row_fold, "cell_fold": cell_fold,
                      "genomic_feature_matrix": Xo, "genomic_residual_matrix": Ro}
    ds = _mapping_dataset(
        out, dims={"blup": _ST, "G": _TT, "E": _TT},
        keep_attrs=("covariance_model", "E_model", "GRM_model", "genomic_features"),
        var_attrs={
            "blup": {"long_name": "phenotype-scale genetic-mean baseline", "units": "phenotype units",
                     "description": "training trait intercept plus mvBLUP breeding value"},
            "G": {"long_name": "total genetic trait covariance used for expert features",
                  "units": "phenotype units squared"},
            "E": {"long_name": "residual trait covariance used for expert features",
                  "units": "phenotype units squared"},
        },
        title="BLUP residual experts fit",
        software="vigor.prediction_experts.fit_blup_residual_experts", parameters=params)
    return maybe_save(ds, save_path)


def predict_blup_residual_experts(fit, traits, *, blup=None, GRM=None,
                                  genomic_features=None, include_multitask=False, save_path=None):
    """Return genomic-only, phenotype-assisted and automatic hybrid predictions.

    Parameters
    ----------
    fit : xarray.Dataset or mapping
        Fit returned by ``fit_blup_residual_experts`` (arrays read from its data variables,
        fitted models and settings from ``attrs``), or a legacy mapping with the same keys,
        for the same animal and trait axes. Passed to ``_compat._expert_features`` as
        ``n_ge_components``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    blup : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    genomic_features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value.

    include_multitask : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``prediction``, ``genomic_only``, ``phenotype_assisted``, ``blup``,
        ``partial_blup`` and, when requested and fitted, ``multitask_genomic`` (all sample,
        trait), plus ``has_phenotype`` (sample). An absent multitask prediction is recorded as
        ``attrs['multitask_genomic'] = None``.
    """
    params = _call_parameters(predict_blup_residual_experts, locals())
    fit = _result_dict(fit); traits = _pandas(traits); GRM = _pandas(GRM)
    genomic_features = _pandas(genomic_features)
    Y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits, columns=fit["blup"].columns)
    Gcan = fit["G"].to_numpy(); Ecan = fit["E"].to_numpy()
    if blup is not None:
        B = _as_frame(blup, Y)
    elif GRM is None:
        B = fit["blup"].reindex(index=Y.index, columns=Y.columns)
    else:
        cov = fit.get("covariance_model", fit["G"]); Em = fit.get("E_model", fit["E"])
        B = _as_frame(_genetic_mean(Y, GRM, cov, E=Em, missing="exact", solver="pcg", preconditioner="diag"), Y)
    gf = genomic_features if genomic_features is not None else fit.get("genomic_features")
    Xg = _compat._expert_features(Y, B, Gcan, Ecan, phenotype_assisted=False,
                                 genomic_features=gf, n_ge_components=fit["n_components"])
    if fit.get('partial_baseline') == 'conditional':
        kernel=GRM if GRM is not None else fit['GRM_model']
        P=_conditional_mean(Y,kernel,fit.get('covariance_model',fit['G']),fit.get('E_model',fit['E']),solver_tol=fit.get('solver_tol',1e-7),solver_maxiter=fit.get('solver_maxiter',500),preconditioner=fit.get('preconditioner','diag'))
    else:
        P=B
    gen = B.copy(); phen = P.copy()
    for j, c in enumerate(Y.columns):
        if c in fit["genomic_models"]:
            gen[c] = B[c].to_numpy() + _compat._predict_stacked_target(fit["genomic_models"][c], Xg)
        if c in fit["phenotype_models"]:
            Xp = _compat._expert_features(Y, B, Gcan, Ecan, target=j, phenotype_assisted=True,
                                         genomic_features=gf, n_ge_components=fit["n_components"])
            phen[c] = P[c].to_numpy() + _compat._predict_stacked_target(fit["phenotype_models"][c], Xp)
    mt = None
    if include_multitask and fit.get("multitask_genomic") is not None:
        mt = pd.DataFrame(B.to_numpy() + fit["multitask_genomic"].predict(Xg), index=Y.index, columns=Y.columns)
    has = Y.notna().any(axis=1)
    hybrid = gen.copy(); hybrid.loc[has] = phen.loc[has]
    out = {"prediction": hybrid, "genomic_only": gen, "phenotype_assisted": phen,
           "blup": B, "partial_blup": P, "multitask_genomic": mt, "has_phenotype": has}
    unit = {"units": "phenotype units"}
    ds = _mapping_dataset(
        out, dims={**{k: _ST for k in ("prediction", "genomic_only", "phenotype_assisted", "blup",
                                       "partial_blup", "multitask_genomic")},
                   "has_phenotype": ("sample",)},
        var_attrs={
            "prediction": {"long_name": "hybrid residual-expert prediction", **unit,
                           "description": "phenotype-assisted rows where any trait is observed, genomic-only otherwise"},
            "genomic_only": {"long_name": "genomic-only residual-expert prediction", **unit},
            "phenotype_assisted": {"long_name": "phenotype-assisted residual-expert prediction", **unit},
            "blup": {"long_name": "genetic-mean baseline", **unit},
            "partial_blup": {"long_name": "partial-row baseline", **unit,
                             "description": "conditional phenotype posterior when partial_baseline='conditional'"},
            "multitask_genomic": {"long_name": "multitask elastic-net genomic prediction", **unit},
            "has_phenotype": {"long_name": "animal has at least one observed trait"},
        },
        title="BLUP residual experts prediction",
        software="vigor.prediction_experts.predict_blup_residual_experts", parameters=params)
    return maybe_save(ds, save_path)


def _best_single_rmse(diag):
    """Extract the best supported single-model prediction error from fit diagnostics.

    Parameters
    ----------
    diag : mapping
        Input for the operation ``diag.get('scores', {})``. See
        ``vigor.prediction_experts._best_single_rmse`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    scores = diag.get("scores", {}) if isinstance(diag, dict) else {}
    base = scores.get("blup", {}).get("rmse", np.inf)
    best = diag.get("best_single", "blup") if isinstance(diag, dict) else "blup"
    return float(scores.get(best, {"rmse": base}).get("rmse", base)), best



def _screened_ridge_fit_matrix(X, R, feature_names, *, top_k=256, alpha=100.0,
                               n_splits=3, random_state=0, min_gain=.005,
                               min_z=1.96):
    """Leakage-safe vectorized SNP/dominance screening + per-trait ridge.

    ``X`` must be row-cross-fitted expert features and ``R`` the matching
    row-cross-fitted residual targets. Candidate raw genotype columns are
    identified by the ``snp_``/``dom_`` prefixes; independently fitted genotype-only spectral coordinates use ``genomic_spectral_``. Screening is recomputed
    inside every validation fold using only that fold's training residuals.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    R : array-like
        Numerical input passed to ``np.divide`` as ``out``; see that callable for required
        dimensions and matrix role. Passed to ``np.divide`` as ``out``.

    feature_names : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``256``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``100.0``. Passed to ``Ridge`` as ``alpha``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``KFold`` as ``n_splits``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.

    min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.005``.

    min_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``1.96``.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    X=np.asarray(X,float); R=np.asarray(R,float); names=list(map(str,feature_names))
    if X.ndim!=2 or R.ndim!=2 or X.shape[0]!=R.shape[0]:
        raise ValueError('screened-ridge X/R shapes do not align')
    cand=np.asarray([i for i,n in enumerate(names) if n.startswith('snp_') or n.startswith('dom_') or n.startswith('genomic_spectral_')],int)
    protect=np.setdiff1d(np.arange(X.shape[1],dtype=int),cand,assume_unique=False)
    rows=np.flatnonzero(np.isfinite(X).any(1) & np.isfinite(R).any(1))
    if len(rows)<max(20,int(n_splits)*4) or not len(cand):
        return {'mode':'blup','models':{},'candidate_indices':cand,'protect_indices':protect,
                'gain':0.0,'gain_z':-np.inf,'rmse':np.nan,'base_rmse':np.nan}
    # Keep missing calls until each inner fold learns its imputation values.
    C_raw=X[:,cand].copy()
    P_raw=X[:,protect].copy() if len(protect) else np.zeros((len(X),0))
    def impute_columns(A, fit_rows):
        """Fill feature missingness using medians estimated on training rows only.

        Parameters
        ----------
        A : array-like
            Numerical operand in ``A[fit_rows]``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        fit_rows : one-dimensional bool array or None
            Mask selecting training animals for fitting frequencies, scaling or model parameters.
            Length must equal the animal axis; None uses the routine's full eligible training set.
            Keep evaluation outcomes outside this mask.

        Returns
        -------
        object
            Computed result from ``(np.where(np.isfinite(A), A, median), median)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        median=np.nanmedian(np.where(np.isfinite(A[fit_rows]),A[fit_rows],np.nan),axis=0)
        median=np.where(np.isfinite(median),median,0.0)
        return np.where(np.isfinite(A),A,median),median
    pred=np.full_like(R,np.nan,dtype=float)
    row_fold=np.full(len(X),-1,int)
    kf=KFold(n_splits=max(2,min(int(n_splits),len(rows)//10)),shuffle=True,random_state=int(random_state))
    for fold,(tr0,va0) in enumerate(kf.split(rows)):
        tr=rows[tr0]; va=rows[va0]; row_fold[va]=fold
        C,_=impute_columns(C_raw,tr); P,_=impute_columns(P_raw,tr)
        cmean=C[tr].mean(0); csd=C[tr].std(0); csd=np.where(csd>1e-8,csd,1.0)
        Ct=(C[tr]-cmean)/csd; Cv=(C[va]-cmean)/csd
        if P.shape[1]:
            pmean=P[tr].mean(0); psd=P[tr].std(0); psd=np.where(psd>1e-8,psd,1.0)
            Pt=(P[tr]-pmean)/psd; Pv=(P[va]-pmean)/psd
        else: Pt=np.zeros((len(tr),0)); Pv=np.zeros((len(va),0))
        Rt=R[tr]; mask=np.isfinite(Rt); mu=np.divide(np.nansum(Rt,0),mask.sum(0),out=np.zeros(R.shape[1]),where=mask.sum(0)>0)
        Y0=np.where(mask,Rt-mu[None,:],0.0)
        # One BLAS-3 screen for all traits. Normalization by observed count is
        # enough for ranking because candidate columns are standardized.
        denom=np.maximum(mask.sum(0),1)
        score=np.abs(Ct.T@Y0)/denom[None,:]
        for j in range(R.shape[1]):
            obs=np.isfinite(Rt[:,j])
            if obs.sum()<10: continue
            k=min(max(1,int(top_k)),len(cand)); chosen=np.argpartition(score[:,j],-k)[-k:]
            Zt=np.c_[Pt[obs],Ct[obs][:,chosen]]; Zv=np.c_[Pv,Cv[:,chosen]]
            mdl=Ridge(alpha=float(alpha),solver='lsqr',tol=1e-5,max_iter=2000).fit(Zt,Rt[obs,j])
            pred[va,j]=mdl.predict(Zv)
    # Gate RAW OOF predictions. Scaling from other OOF folds is not nested:
    # their fitted predictions can carry this scoring fold's outcomes indirectly.
    # All-OOF calibration is reserved for deployment after the raw gate passes.
    from .residual_matrix import _final_oof_calibration
    calibration=_final_oof_calibration(R,pred,min_points=5,max_scale=1.0)
    pred_cal=pred.copy()
    ok=np.isfinite(R)&np.isfinite(pred_cal); y=R[ok]; q=pred_cal[ok]
    if not len(y):
        return {'mode':'blup','models':{},'candidate_indices':cand,'protect_indices':protect,
                'gain':0.0,'gain_z':-np.inf,'rmse':np.nan,'base_rmse':np.nan,
                'calibration':calibration}
    base=np.zeros_like(y); gain,z=_compat._paired_rmse_gain(y,base,q)
    base_rmse=float(np.sqrt(np.mean(y*y))); rmse=float(np.sqrt(np.mean((y-q)**2)))
    if gain<float(min_gain) or z<float(min_z):
        return {'mode':'blup','models':{},'candidate_indices':cand,'protect_indices':protect,
                'gain':float(gain),'gain_z':float(z),'rmse':rmse,'base_rmse':base_rmse,
                'oof_prediction':pred_cal,'calibration':calibration}
    # Final fit, again screening only on the training residuals supplied here.
    C,cm=impute_columns(C_raw,rows); P,pm=impute_columns(P_raw,rows)
    cmean=C[rows].mean(0); csd=C[rows].std(0); csd=np.where(csd>1e-8,csd,1.0); Ct=(C[rows]-cmean)/csd
    if P.shape[1]:
        pmean=P[rows].mean(0); psd=P[rows].std(0); psd=np.where(psd>1e-8,psd,1.0); Pt=(P[rows]-pmean)/psd
    else:
        pmean=psd=np.zeros(0); Pt=np.zeros((len(rows),0))
    mask=np.isfinite(R[rows]); mu=np.divide(np.nansum(R[rows],0),mask.sum(0),out=np.zeros(R.shape[1]),where=mask.sum(0)>0)
    Y0=np.where(mask,R[rows]-mu[None,:],0.0); score=np.abs(Ct.T@Y0)/np.maximum(mask.sum(0),1)[None,:]
    models={}
    for j in range(R.shape[1]):
        obs=mask[:,j]
        if obs.sum()<10: continue
        k=min(max(1,int(top_k)),len(cand)); chosen=np.argpartition(score[:,j],-k)[-k:]
        Z=np.c_[Pt[obs],Ct[obs][:,chosen]]
        mdl=Ridge(alpha=float(alpha),solver='lsqr',tol=1e-5,max_iter=2000).fit(Z,R[rows][obs,j])
        models[j]={'model':mdl,'chosen_local':chosen}
    return {'mode':'screened_ridge','models':models,'candidate_indices':cand,'protect_indices':protect,
            'candidate_median':cm,'protect_median':(pm if P.shape[1] else np.zeros(0)),
            'candidate_mean':cmean,'candidate_sd':csd,'protect_mean':pmean,'protect_sd':psd,
            'gain':float(gain),'gain_z':float(z),'rmse':rmse,'base_rmse':base_rmse,
            'oof_prediction':pred_cal,'calibration':calibration,'top_k':int(top_k),'alpha':float(alpha)}


def _screened_ridge_predict(fit, X, n_traits):
    """Apply traitwise screened ridge models to the aligned genomic feature matrix.

    Parameters
    ----------
    fit : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension.
    """
    X=np.asarray(X,float)
    if fit is None or fit.get('mode')!='screened_ridge': return np.zeros((len(X),int(n_traits)))
    cand=fit['candidate_indices']; protect=fit['protect_indices']
    C=X[:,cand].copy(); bad=~np.isfinite(C)
    if bad.any(): C[bad]=np.take(fit['candidate_median'],np.where(bad)[1])
    C=(C-fit['candidate_mean'])/fit['candidate_sd']
    if len(protect):
        P=X[:,protect].copy(); bad=~np.isfinite(P)
        if bad.any(): P[bad]=np.take(fit['protect_median'],np.where(bad)[1])
        P=(P-fit['protect_mean'])/fit['protect_sd']
    else: P=np.zeros((len(X),0))
    out=np.zeros((len(X),int(n_traits)))
    cal=np.asarray(fit.get('calibration',np.ones(int(n_traits))),float)
    for j,z in fit['models'].items():
        jj=int(j); out[:,jj]=cal[jj]*z['model'].predict(np.c_[P,C[:,z['chosen_local']]])
    return out

def _nested_screened_genomic_fit(Y,GRM,G,E,feature,*,covariance_kwargs,refit,n_splits,
        random_state,genomic_features,n_components,top_k,alpha,min_gain,min_z,
        selection_cv,solver_tol,solver_maxiter,preconditioner):
    """Outer row gates exclude scored labels from every inner baseline fit.

    Pooled OOF regressors alone are insufficient: another fold's covariance or
    BLUP feature could encode the gate fold. Rebuild inner pairs after masking
    the entire scored outer fold. Final training retains genuine OOF targets.

    Parameters
    ----------
    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance.

    feature : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options.

    refit : object (see the cited input contract)
        Input supplied to ``fit_blup_residual_experts`` as ``crossfit_refit_covariance``. See
        that callable's documented ``crossfit_refit_covariance`` contract for its type,
        alignment and value domain. This wrapper retains that contract. Passed to
        ``fit_blup_residual_experts`` as ``crossfit_refit_covariance``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Passed to ``fit_blup_residual_experts`` as ``n_splits``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Passed to
        ``fit_blup_residual_experts`` as ``random_state``.

    genomic_features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Passed to ``fit_blup_residual_experts`` as
        ``genomic_features``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Passed to ``fit_blup_residual_experts`` as ``n_components``.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Passed to
        ``_screened_ridge_fit_matrix`` as ``top_k``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Passed to ``_screened_ridge_fit_matrix`` as ``alpha``.

    min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Passed to ``_screened_ridge_fit_matrix`` as ``min_gain``.

    min_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Passed to ``_screened_ridge_fit_matrix`` as ``min_z``.

    selection_cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Passed to ``_screened_ridge_fit_matrix`` as ``n_splits``.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Passed to ``_genetic_mean`` as ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Passed to
        ``_genetic_mean`` as ``solver_maxiter``.

    preconditioner : object (see the cited input contract)
        Input supplied to ``_genetic_mean`` as ``preconditioner``. See that callable's
        documented ``preconditioner`` contract for its type, alignment and value domain. This
        wrapper retains that contract.
    """
    oof=feature['oof'];X=oof['genomic_feature_matrix'];R=oof['genomic_residual_matrix']
    folds=oof['row_fold'];prediction=np.full(R.shape,np.nan)
    kw=dict(covariance_kwargs or {});kw.pop('n_boot',None)
    for k in range(n_splits):
        va=folds==k;masked=Y.copy();masked.iloc[va]=np.nan
        cov=genetic_varcov_PSD(GRM,masked,n_boot=0,**kw) if refit else G
        em=None if refit else E
        base=_genetic_mean(masked,GRM,cov,em,missing='exact',solver='pcg',solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner)
        inner=fit_blup_residual_experts(masked,GRM,cov,em,blup=base,methods=(),
            n_splits=n_splits,random_state=random_state+1009*(k+1),genomic_features=genomic_features,
            crossfit_refit_covariance=refit,covariance_kwargs=kw,n_components=n_components,
            fit_genomic_per_trait=False,fit_phenotype_per_trait=False,fit_multitask_genomic=False,store_oof=True,
            solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner)
        io=inner.attrs['oof']
        model=_screened_ridge_fit_matrix(io['genomic_feature_matrix'],io['genomic_residual_matrix'],
            feature['feature_names_genomic'],top_k=top_k,alpha=alpha,n_splits=selection_cv,
            random_state=random_state+7919+k,min_gain=min_gain,min_z=min_z)
        prediction[va]=_screened_ridge_predict(model,X[va],R.shape[1])
    valid=np.isfinite(R)&np.isfinite(prediction);gain,_=_compat._paired_rmse_gain(R[valid],np.zeros(valid.sum()),prediction[valid])
    improvement=np.where(valid,R**2-(R-prediction)**2,0.).sum(1);count=valid.sum(1);rows=count>0;d=improvement[rows]/count[rows]
    z=float(d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))) if len(d)>1 and d.std()>0 else -np.inf
    rmse=float(np.sqrt(np.mean((R[valid]-prediction[valid])**2))) if valid.any() else np.inf
    base_rmse=float(np.sqrt(np.mean(R[valid]**2))) if valid.any() else np.inf
    if gain<min_gain or z<min_z:
        out={'mode':'blup','models':{},'gain':gain,'gain_z':z,'rmse':rmse,'base_rmse':base_rmse}
    else:
        out=_screened_ridge_fit_matrix(X,R,feature['feature_names_genomic'],top_k=top_k,alpha=alpha,n_splits=selection_cv,
            random_state=random_state+4242,min_gain=-np.inf,min_z=-np.inf)
        from .residual_matrix import _final_oof_calibration
        out['calibration']=_final_oof_calibration(R,prediction,min_points=5,max_scale=1.)
        out.update(gain=gain,gain_z=z,rmse=rmse,base_rmse=base_rmse)
    out['nested_oof_prediction']=prediction;out['selection']='nested row baseline/features/targets; animal-grouped gate'
    return out


def fit_blup_prediction_experts(
    traits, GRM, G, E=None, *, blup=None, genomic_features=None,
    feature_methods=("pcr", "pls", "elasticnet", "lightgbm"),
    matrix_methods=("iterative_ridge", "softimpute"), n_splits=3,
    random_state=0, crossfit_refit_covariance=True, covariance_kwargs=None,
    model_kwargs=None, min_matrix_gain=.01, family_tie_tolerance=.002,
    fit_genomic_per_trait=True, fit_phenotype_per_trait=True, selection_cv=3,
    genomic_strategy="per_trait", genomic_screen_top_k=256, genomic_screen_alpha=100.0,
    genomic_screen_min_gain=.005, genomic_screen_min_z=1.96,
    matrix_calibrate_oof=True, matrix_calibration_min_points=8,
    matrix_calibration_max_scale=1.0,
    partial_baseline='genetic_mean', save_path=None,
    **feature_kwargs,
):
    """Fit target-specific residual experts around one mvBLUP model.

    ``blup``, when supplied, must be the phenotype-scale genetic mean
    (training trait intercept + mvBLUP breeding value), not raw mvBLUP output.
    Omit it to compute that mean here. Rejected experts retain this genetic
    mean for partial rows too; they do not silently add mvPredict's conditional
    residual posterior. Migrate old callers with ``mvPredict(...)["genetic_mean"]``
    using matching prediction covariance, or add training means to mvBLUP.

    Set ``partial_baseline='conditional'`` to fit and gate partial-animal
    corrections around mvPredict's conditional phenotype posterior. Rejection
    then preserves that posterior. Fully unseen rows still use the genetic
    mean, since an identity residual kernel supplies no cross-animal E signal.
    The default genetic_mean policy is retained for compatibility.

    The returned model *always* keeps the genomic-only expert available for
    completely unseen animals.  For rows with phenotype context it selects
    between the phenotype-assisted feature expert and matrix completion using
    cross-fitted residual RMSE. This fixes the historical ``best expert``
    behavior where choosing matrix completion globally could discard a useful
    genomic expert for new animals.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Passed to ``fit_residual_matrix_expert`` as ``GRM``.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Passed to ``fit_residual_matrix_expert`` as ``G``.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    blup : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    genomic_features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value. Passed to
        ``fit_blup_residual_experts`` as ``genomic_features``.

    feature_methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('pcr', 'pls', 'elasticnet', 'lightgbm')``. Passed to
        ``fit_blup_residual_experts`` as ``methods``.

    matrix_methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('iterative_ridge', 'softimpute')``. Passed to
        ``fit_residual_matrix_expert`` as ``methods``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``fit_blup_residual_experts`` as ``n_splits``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``fit_blup_residual_experts`` as ``random_state``.

    crossfit_refit_covariance : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``. Passed to ``fit_blup_residual_experts`` as
        ``crossfit_refit_covariance``.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to
        ``fit_blup_residual_experts`` as ``covariance_kwargs``.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to
        ``fit_blup_residual_experts`` as ``model_kwargs``.

    min_matrix_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.01``. Passed to ``fit_residual_matrix_expert`` as
        ``min_gain``.

    family_tie_tolerance : float
        Input for the operation ``pr * (1.0 - float(family_tie_tolerance))``. See
        ``vigor.prediction_experts.fit_blup_prediction_experts`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``0.002``.

    fit_genomic_per_trait : bool
        Input supplied to ``fit_blup_residual_experts`` as ``fit_genomic_per_trait``. See that
        callable's documented ``fit_genomic_per_trait`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``True``. Passed to
        ``fit_blup_residual_experts`` as ``fit_genomic_per_trait``.

    fit_phenotype_per_trait : bool
        Input supplied to ``fit_blup_residual_experts`` as ``fit_phenotype_per_trait``. See that
        callable's documented ``fit_phenotype_per_trait`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``True``. Passed to
        ``fit_blup_residual_experts`` as ``fit_phenotype_per_trait``.

    selection_cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``fit_blup_residual_experts`` as ``selection_cv``.

    genomic_strategy : str
        Input for the operation ``str(genomic_strategy).lower() in {'screened_ridge',
        'screened', 'ridge_screen'}``. See
        ``vigor.prediction_experts.fit_blup_prediction_experts`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``'per_trait'``.

    genomic_screen_top_k : int
        Input supplied to ``_nested_screened_genomic_fit`` as ``top_k``. See that callable's
        documented ``top_k`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``256``. Passed to ``_nested_screened_genomic_fit`` as
        ``top_k``.

    genomic_screen_alpha : float
        Input supplied to ``_nested_screened_genomic_fit`` as ``alpha``. See that callable's
        documented ``alpha`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``100.0``. Passed to ``_nested_screened_genomic_fit`` as
        ``alpha``.

    genomic_screen_min_gain : float
        Input supplied to ``_nested_screened_genomic_fit`` as ``min_gain``. See that callable's
        documented ``min_gain`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``0.005``. Passed to ``_nested_screened_genomic_fit`` as
        ``min_gain``.

    genomic_screen_min_z : float
        Input supplied to ``_nested_screened_genomic_fit`` as ``min_z``. See that callable's
        documented ``min_z`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``1.96``. Passed to ``_nested_screened_genomic_fit`` as
        ``min_z``.

    matrix_calibrate_oof : bool
        Input supplied to ``fit_residual_matrix_expert`` as ``calibrate_oof``. See that
        callable's documented ``calibrate_oof`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``True``. Passed to
        ``fit_residual_matrix_expert`` as ``calibrate_oof``.

    matrix_calibration_min_points : int
        Input supplied to ``fit_residual_matrix_expert`` as ``calibration_min_points``. See that
        callable's documented ``calibration_min_points`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``8``. Passed to
        ``fit_residual_matrix_expert`` as ``calibration_min_points``.

    matrix_calibration_max_scale : float
        Input supplied to ``fit_residual_matrix_expert`` as ``calibration_max_scale``. See that
        callable's documented ``calibration_max_scale`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``1.0``. Passed to
        ``fit_residual_matrix_expert`` as ``calibration_max_scale``.

    partial_baseline : str
        Partial-trait baseline: 'conditional' includes observed-cell conditional prediction. See
        the fitting routine for the other baseline policies it explicitly supports. Default:
        ``'genetic_mean'``. Passed to ``fit_blup_residual_experts`` as ``partial_baseline``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    **feature_kwargs : dict
        Input supplied to ``fit_blup_residual_experts`` as ``**expanded mapping``. See that
        callable's documented ``**expanded mapping`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Passed to ``fit_blup_residual_experts`` as
        ``**expanded mapping``.

    Returns
    -------
    xarray.Dataset
        Data variable ``blup`` (sample, trait). ``attrs`` hold ``feature_fit`` (the
        ``fit_blup_residual_experts`` Dataset), ``matrix_fit`` (the
        ``fit_residual_matrix_expert`` Dataset), ``screened_genomic_fit`` (mapping or None),
        ``unseen_mode``, ``partial_mode``, ``scores``, ``version`` and ``partial_baseline``.
        Nested fits and estimators live in memory and are saved as repr.
        ``predict_blup_prediction_experts`` accepts it.
    """
    params = _call_parameters(fit_blup_prediction_experts, locals())
    traits, GRM, G, E = _pandas(traits), _pandas(GRM), _pandas(G), _pandas(E)
    genomic_features = _pandas(genomic_features)
    Y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    E_model = _resolve_E(G, E)
    B = _as_frame(blup, Y) if blup is not None else _as_frame(
        _genetic_mean(Y, GRM, G, E=E_model, missing="exact", solver="pcg", preconditioner="diag"), Y)
    screened = str(genomic_strategy).lower() in {"screened_ridge","screened","ridge_screen"}
    feature_kwargs.setdefault("fit_multitask_genomic",not screened)
    feature_ds = fit_blup_residual_experts(
        Y, GRM, G, E_model, blup=B, n_splits=n_splits, random_state=random_state,
        genomic_features=genomic_features, methods=feature_methods,
        crossfit_refit_covariance=crossfit_refit_covariance,
        covariance_kwargs=covariance_kwargs, model_kwargs=model_kwargs,
        fit_genomic_per_trait=(fit_genomic_per_trait and not screened),
        fit_phenotype_per_trait=fit_phenotype_per_trait, selection_cv=selection_cv,
        store_oof=screened, **feature_kwargs,
        partial_baseline=partial_baseline,
    )
    feature = _result_dict(feature_ds)
    screened_fit = None
    if screened:
        screened_fit=_nested_screened_genomic_fit(Y,GRM,G,E_model,feature,
            covariance_kwargs=covariance_kwargs,refit=crossfit_refit_covariance,n_splits=n_splits,
            random_state=random_state,genomic_features=genomic_features,n_components=feature['n_components'],
            top_k=genomic_screen_top_k,alpha=genomic_screen_alpha,min_gain=genomic_screen_min_gain,
            min_z=genomic_screen_min_z,selection_cv=selection_cv,
            solver_tol=feature.get('solver_tol',2e-6),solver_maxiter=feature.get('solver_maxiter',500),
            preconditioner=feature.get('preconditioner','diag'))
    matrix_ds = fit_residual_matrix_expert(
        Y, B, GRM=GRM, G=G, E=E_model, methods=matrix_methods, n_splits=n_splits,
        random_state=random_state + 3000, min_gain=min_matrix_gain,
        model_kwargs=model_kwargs, crossfit_refit_covariance=crossfit_refit_covariance,
        covariance_kwargs=covariance_kwargs, calibrate_oof=matrix_calibrate_oof,
        calibration_min_points=matrix_calibration_min_points,
        calibration_max_scale=matrix_calibration_max_scale,
        prediction_target=('phenotype_prediction' if partial_baseline == 'conditional' else 'genetic_mean'),
    )
    matrix = _result_dict(matrix_ds)
    pr, pbest = _best_single_rmse(feature["diagnostics"]["phenotype_global"])
    mr = float(matrix["scores"].get(matrix["method"], matrix["scores"]["blup"]).get("rmse", np.inf))
    p_mode = feature["diagnostics"]["phenotype_global"].get("mode", "blup")
    if matrix["method"] != "blup" and mr < pr * (1.0 - float(family_tie_tolerance)):
        partial_mode = "matrix"
    elif p_mode != "blup":
        partial_mode = "phenotype_assisted"
    else:
        partial_mode = "blup"
    if screened:
        g_mode = "screened_ridge" if screened_fit is not None and screened_fit.get("mode") == "screened_ridge" else "blup"
    else:
        g_mode = "genomic_expert" if feature["diagnostics"]["genomic_global"].get("mode", "blup") != "blup" else "blup"
    out = {
        "feature_fit": feature_ds, "matrix_fit": matrix_ds, "screened_genomic_fit": screened_fit, "blup": B,
        "unseen_mode": g_mode, "partial_mode": partial_mode,
        "scores": {"phenotype_feature_rmse": pr, "phenotype_feature_best": pbest,
                   "matrix_rmse": mr},
        "version": __import__("vigor").__version__,
        "partial_baseline": partial_baseline,
    }
    ds = _mapping_dataset(
        out, dims={"blup": _ST},
        var_attrs={"blup": {"long_name": "phenotype-scale genetic-mean baseline", "units": "phenotype units",
                            "description": "training trait intercept plus mvBLUP breeding value"}},
        title="BLUP prediction experts fit",
        software="vigor.prediction_experts.fit_blup_prediction_experts", parameters=params)
    return maybe_save(ds, save_path)


def predict_blup_prediction_experts(fit, traits, *, blup=None, GRM=None, genomic_features=None,
                                    save_path=None):
    """Predict using the correct expert family for each animal's information state.

    Parameters
    ----------
    fit : xarray.Dataset or mapping
        Fit returned by ``fit_blup_prediction_experts`` (nested fits, modes and settings are
        read from ``attrs``), or a legacy mapping with the same keys, for the same animal and
        trait axes. Passed to ``_compat._expert_features`` as ``n_ge_components``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    blup : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``predict_blup_residual_experts`` as ``blup``.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value. Passed to
        ``predict_blup_residual_experts`` as ``GRM``.

    genomic_features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value. Passed to
        ``predict_blup_residual_experts`` as ``genomic_features``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``prediction``, ``blup``, ``partial_blup``, ``genomic_only``,
        ``genomic_phenotype_prediction``, ``phenotype_assisted`` and, when used,
        ``screened_genomic`` and ``matrix`` (all sample, trait), plus ``row_status`` and
        ``prediction_mode`` (sample). ``attrs`` hold ``unseen_mode``, ``partial_mode``,
        ``genomic_prediction_interpretation`` and ``None`` for unused optional predictions.
    """
    params = _call_parameters(predict_blup_prediction_experts, locals())
    fit = _result_dict(fit); traits = _pandas(traits); GRM = _pandas(GRM)
    genomic_features = _pandas(genomic_features)
    feature = _result_dict(predict_blup_residual_experts(
        fit["feature_fit"], traits, blup=blup, GRM=GRM, genomic_features=genomic_features,
    ))
    Y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits, columns=feature["blup"].columns)
    B = feature["blup"]
    matrix = (predict_residual_matrix_expert(fit["matrix_fit"], Y, feature['partial_blup'], GRM=GRM)
              if fit["partial_mode"] == "matrix" else None)
    nobs = Y.notna().sum(axis=1)
    unseen = nobs.eq(0); full = nobs.eq(Y.shape[1]); partial = ~(unseen | full)
    out = B.copy(); out.loc[~unseen]=feature['partial_blup'].loc[~unseen]
    screened_prediction = None
    if fit.get("unseen_mode") == "screened_ridge":
        ff=_result_dict(fit["feature_fit"])
        Gcan=ff["G"].to_numpy(); Ecan=ff["E"].to_numpy()
        gf=genomic_features if genomic_features is not None else ff.get("genomic_features")
        Xg=_compat._expert_features(Y,B,Gcan,Ecan,phenotype_assisted=False,genomic_features=gf,
                                    n_ge_components=ff["n_components"])
        add=_screened_ridge_predict(fit.get("screened_genomic_fit"),Xg,Y.shape[1])
        screened_prediction=pd.DataFrame(B.to_numpy()+add,index=Y.index,columns=Y.columns)
        out.loc[unseen]=screened_prediction.loc[unseen]
    elif fit["unseen_mode"] == "genomic_expert":
        out.loc[unseen] = feature["genomic_only"].loc[unseen]
    if fit["partial_mode"] == "matrix":
        out.loc[~unseen] = matrix.loc[~unseen]
    elif fit["partial_mode"] == "phenotype_assisted":
        out.loc[~unseen] = feature["phenotype_assisted"].loc[~unseen]
    status = pd.Series("partially_observed", index=Y.index, dtype="object")
    status.loc[unseen] = "fully_unseen"
    status.loc[full] = "fully_observed_diagnostic"
    mode = pd.Series(fit["partial_mode"], index=Y.index, dtype="object")
    mode.loc[unseen] = fit["unseen_mode"]
    genomic_mean=(screened_prediction if screened_prediction is not None else feature["genomic_only"])
    # Row-held-out discovery/gating supports unseen prediction. It does not
    # authorize a partial-cell correction. The E posterior is the scientifically
    # correct fallback until a separate partial gate has been supplied.
    genomic_phenotype=feature['partial_blup'].copy()
    genomic_phenotype.loc[unseen]=genomic_mean.loc[unseen]
    partial_gate=fit.get('genomic_partial_gate',{})
    if partial_gate.get('accepted',False):
        genomic_phenotype.loc[~unseen]+=genomic_mean.loc[~unseen]-B.loc[~unseen]
    result = {
        "prediction": out, "blup": B, "partial_blup": feature['partial_blup'],
        "genomic_only": genomic_mean,
        "genomic_phenotype_prediction":genomic_phenotype,
        "genomic_prediction_interpretation":"genomic_only is intercept+genotype-based mean; conditional phenotype adds E posterior, with separate partial gate required for genomic correction",

        "screened_genomic": screened_prediction,
        "phenotype_assisted": feature["phenotype_assisted"], "matrix": matrix,
        "row_status": status, "prediction_mode": mode,
        "unseen_mode": fit["unseen_mode"], "partial_mode": fit["partial_mode"],
    }
    unit = {"units": "phenotype units"}
    ds = _mapping_dataset(
        result, dims={**{k: _ST for k in ("prediction", "blup", "partial_blup", "genomic_only",
                                          "genomic_phenotype_prediction", "screened_genomic",
                                          "phenotype_assisted", "matrix")},
                      "row_status": ("sample",), "prediction_mode": ("sample",)},
        var_attrs={
            "prediction": {"long_name": "information-state routed expert prediction", **unit},
            "blup": {"long_name": "genetic-mean baseline", **unit},
            "partial_blup": {"long_name": "partial-row baseline", **unit},
            "genomic_only": {"long_name": "intercept plus genotype-based mean", **unit},
            "genomic_phenotype_prediction": {"long_name": "conditional phenotype prediction with gated genomic correction", **unit},
            "screened_genomic": {"long_name": "screened-ridge genomic prediction", **unit},
            "phenotype_assisted": {"long_name": "phenotype-assisted feature-expert prediction", **unit},
            "matrix": {"long_name": "matrix-completion expert prediction", **unit},
            "row_status": {"long_name": "animal information state",
                           "description": "fully_unseen, partially_observed or fully_observed_diagnostic"},
            "prediction_mode": {"long_name": "expert family used for the animal"},
        },
        title="BLUP prediction experts prediction",
        software="vigor.prediction_experts.predict_blup_prediction_experts", parameters=params)
    return maybe_save(ds, save_path)


__all__ = [
    "fit_blup_residual_experts", "predict_blup_residual_experts",
    "fit_blup_prediction_experts", "predict_blup_prediction_experts",
]
