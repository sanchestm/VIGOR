"""Leakage-safe cross-fitting for matrix residual experts.

The historical two-argument API is retained for compatibility.  Leakage-safe
model selection requires ``GRM`` so that every validation cell is removed
before its fold-specific BLUP is calculated.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import multivariate as _compat
from .multivariate import genetic_varcov_PSD, mvBLUP, mvPredict
from .results import _is_array_like, maybe_save, result_dataset

_ST = ("sample", "trait")
_TT = ("trait_i", "trait_j")
_PROVENANCE = ("title", "software", "vigor_version", "created", "parameters")


def _pandas(value):
    """Return pandas data for a labeled ``xarray.DataArray`` and other inputs unchanged.

    Parameters
    ----------
    value : object
        Phenotype, baseline, covariance, feature or kernel input. A ``DataArray`` taken from a
        VIGOR result Dataset is converted with ``to_pandas()`` so its sample/trait labels are
        kept; pandas, NumPy, Dataset and operator inputs are returned as supplied.
    """
    import xarray as xr
    return value.to_pandas() if isinstance(value, xr.DataArray) else value


def _result_dict(result):
    """Return a fit/prediction result as a plain mapping under its historical key names.

    Parameters
    ----------
    result : xarray.Dataset or mapping
        Result produced by a VIGOR fit or predict function. Dataset data variables are
        converted with ``to_pandas()`` and non-array members are read from ``attrs`` under the
        same names (provenance attributes excluded); mappings are shallow-copied unchanged.
    """
    import xarray as xr
    if not isinstance(result, xr.Dataset):
        return dict(result)
    out = {k: (v.to_pandas() if v.ndim <= 2 else v) for k, v in result.data_vars.items()}
    out.update({k: v for k, v in result.attrs.items() if k not in _PROVENANCE})
    return out


def _call_parameters(function, values):
    """Collect scalar, string and boolean call parameters for ``attrs['parameters']``.

    Parameters
    ----------
    function : callable
        Public function whose signature names the parameters to record; ``save_path`` is
        excluded because it describes the output location rather than the analysis.
    values : mapping
        Local namespace of that function (``locals()``); only names in its signature whose
        values are None, str, bool, int, float or NumPy scalars are recorded.
    """
    import inspect
    names = [k for k in inspect.signature(function).parameters if k != "save_path"]
    return {k: values[k] for k in names if k in values
            and isinstance(values[k], (str, bool, int, float, np.generic, type(None)))}


def _mapping_dataset(values, *, dims=None, var_attrs=None, keep_attrs=(), title=None,
                     software=None, parameters=None):
    """Build a result Dataset that keeps every historical key, including ``None`` members.

    Parameters
    ----------
    values : mapping
        Named fit/prediction outputs under their historical key names. Array-like members
        (pandas, NumPy with ``dims``, xarray) become data variables; every other member,
        including ``None`` placeholders, is stored in ``attrs`` under the same key.
    dims : mapping or None
        Dimension names per array key, forwarded to ``result_dataset``; required for
        multi-axis NumPy arrays. Default: ``None``.
    var_attrs : mapping or None
        Per-variable ``long_name``/``units``/``description`` attributes forwarded to
        ``result_dataset``. Default: ``None``.
    keep_attrs : sequence of str
        Keys stored in ``attrs`` even when array-like: echoed model inputs (kernels, input
        covariances, feature matrices) whose labels need not align with the result axes.
        Default: ``()``.
    title : str or None
        Human-readable Dataset title recorded in ``attrs['title']``. Default: ``None``.
    software : str or None
        Producing VIGOR function recorded in ``attrs['software']``. Default: ``None``.
    parameters : mapping or None
        Scalar call parameters stored in ``attrs['parameters']`` for provenance. Default:
        ``None``.
    """
    arrays = {k: v for k, v in values.items()
              if k not in keep_attrs and v is not None and _is_array_like(v)}
    return result_dataset(arrays, dims=dims, var_attrs=var_attrs,
                          attrs={k: v for k, v in values.items() if k not in arrays},
                          title=title, software=software, parameters=parameters)


def _matrix_fit_dataset(out, parameters):
    """Package a residual-matrix fit mapping as a labeled result Dataset.

    Parameters
    ----------
    out : mapping
        Fit members under their historical names (``blup``, ``G``, ``E``, ``mode``,
        ``model``, ``scores``, ...). Sample-by-trait and trait-by-trait arrays become data
        variables; estimators, the echoed ``GRM`` and diagnostics stay in ``attrs``.
    parameters : mapping
        Scalar call parameters of ``fit_residual_matrix_expert`` recorded in
        ``attrs['parameters']`` for provenance.
    """
    return _mapping_dataset(
        out, dims={"blup": _ST, "genetic_mean": _ST, "_observed_traits": _ST,
                   "calibration": ("trait",), "G": _TT, "E": _TT},
        keep_attrs=("GRM",),
        var_attrs={
            "blup": {"long_name": "residual-expert baseline prediction", "units": "phenotype units",
                     "description": "genetic mean, or conditional phenotype prediction when prediction_target='phenotype_prediction'"},
            "genetic_mean": {"long_name": "intercept plus breeding value", "units": "phenotype units"},
            "_observed_traits": {"long_name": "training phenotypes cached for conditional prediction",
                                 "units": "phenotype units"},
            "calibration": {"long_name": "deployment amplitude calibration per trait", "units": "1"},
            "G": {"long_name": "genetic trait covariance used by the baseline", "units": "phenotype units squared"},
            "E": {"long_name": "residual trait covariance used by the baseline", "units": "phenotype units squared"},
        },
        title="Residual matrix expert fit",
        software="vigor.residual_matrix.fit_residual_matrix_expert",
        parameters=parameters)


def _matrix_model(method, random_state, model_kwargs):
    """Use the release factory, falling back to its complete archived module.

    Parameters
    ----------
    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options.
    """
    return _compat._v31_matrix_model(method, random_state, model_kwargs)


def _frame(value, *, like=None):
    """Compute frame.

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation ``value.copy()``. See ``vigor.residual_matrix._frame`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.

    like : array-like, table or fitted object
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``out.reindex`` as ``index``.
    """
    value = _pandas(value)
    if isinstance(value, pd.DataFrame):
        out = value.copy()
        if like is not None:
            out = out.reindex(index=like.index, columns=like.columns)
        return out
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("traits and blup must be two-dimensional")
    if like is None:
        return pd.DataFrame(array)
    if array.shape != like.shape:
        raise ValueError("blup must have the same shape as traits")
    return pd.DataFrame(array, index=like.index, columns=like.columns)


def _covariances(result):
    """Extract genetic/residual covariance components from a compatible fit result.

    Parameters
    ----------
    result : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.
    """
    import xarray as xr
    try:
        # mvBLUP needs the complete covariance result to associate each named
        # GRM with its own trait covariance.  The aggregate G is valid only for
        # a single kernel.
        view = _result_dict(result) if isinstance(result, xr.Dataset) else result
        components = view.get("G_components_prediction", view.get("G_components", {}))
        if len(components) > 1:
            return result, None
        return view.get("G_prediction", view["G"]), view.get("E_prediction", view["E"])
    except (KeyError, TypeError) as exc:
        raise ValueError("covariance fit must return mappings named 'G' and 'E'") from exc


def _blup(Y, GRM, G, E, solver_tol, solver_maxiter, preconditioner, prediction_target='genetic_mean'):
    # Full-rank float64 is intentional here: cross-fit selection must not vary
    # because an automatic kernel approximation happened to be selected.
    """Compute the configured genetic/conditional baseline prediction for a residual-matrix expert.

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

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Passed to ``mvPredict`` as ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Passed to
        ``mvPredict`` as ``solver_maxiter``.

    preconditioner : object (see the cited input contract)
        Input supplied to ``mvPredict`` as ``preconditioner``. See that callable's documented
        ``preconditioner`` contract for its type, alignment and value domain. This wrapper
        retains that contract.

    prediction_target : str
        Baseline target: 'genetic_mean' uses intercept plus genetic prediction;
        'phenotype_prediction' includes the conditional residual prediction where supported. The
        residual expert is trained against the selected baseline. Default: ``'genetic_mean'``.
    """
    if prediction_target == 'phenotype_prediction':
        return mvPredict(Y,GRM,G,E,accuracy='exact',kernel_rank='full',
                         solver_tol=solver_tol,solver_maxiter=solver_maxiter,
                         preconditioner=preconditioner)['phenotype_prediction']
    if prediction_target != 'genetic_mean':
        raise ValueError('prediction_target must be genetic_mean or phenotype_prediction')
    breeding = _compat._mvBLUP_raw(
        Y, GRM, G, E, missing="exact", dtype=np.float64,
        work_dtype=np.float64, kernel_rank="full", accuracy="exact", solver_tol=solver_tol,
        solver_maxiter=solver_maxiter, preconditioner=preconditioner,
    )
    return breeding + Y.mean(axis=0).fillna(0.)




def _final_oof_calibration(target, prediction, *, min_points=8, max_scale=1.0):
    """Deployment-only amplitude; selection must score raw OOF corrections.

    Parameters
    ----------
    target : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    prediction : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    min_points : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``8``.

    max_scale : array-like or numeric scalar
        Input for the operation ``np.clip(float(y[ok, j] @ q[ok, j]) / den if den > 1e-12 else
        0.0, 0.0, max_scale)``. See ``vigor.residual_matrix._final_oof_calibration`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``1.0``.
    """
    y=np.asarray(target,float); q=np.asarray(prediction,float)
    beta=np.zeros(y.shape[1],float)
    for j in range(y.shape[1]):
        ok=np.isfinite(y[:,j])&np.isfinite(q[:,j])
        if ok.sum()<min_points: continue
        den=float(q[ok,j]@q[ok,j])
        beta[j]=np.clip(float(y[ok,j]@q[ok,j])/den if den>1e-12 else 0.,0.,max_scale)
    return beta


def _crossfit_calibrate_oof(target, prediction, folds, *, min_points=8, max_scale=1.0):
    """Diagnostic leave-one-OOF-fold-out amplitude calibration (not nested).

    A residual expert can correlate with held-out residuals while overshooting
    their amplitude and therefore worsening RMSE.  For every validation fold
    and trait, estimate one non-negative shrinkage coefficient using *other*
    OOF folds only, then apply it to that fold.  The final coefficient stored
    for deployment is estimated from all OOF predictions after model selection.

    Other OOF predictions can indirectly carry scored-fold outcomes into scales.
    Do not use these diagnostic predictions for model acceptance; gates score raw
    OOF predictions and deployment uses _final_oof_calibration instead.

    Parameters
    ----------
    target : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    prediction : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    folds : array-like or numeric scalar
        Input for the operation ``np.asarray(folds)``. See
        ``vigor.residual_matrix._crossfit_calibrate_oof`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    min_points : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``8``.

    max_scale : array-like or numeric scalar
        Input for the operation ``np.clip(beta, 0.0, float(max_scale))``. See
        ``vigor.residual_matrix._crossfit_calibrate_oof`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``1.0``.
    """
    y=np.asarray(target,float); q=np.asarray(prediction,float); f=np.asarray(folds)
    if y.shape!=q.shape or y.shape!=f.shape:
        raise ValueError('target, prediction and folds must have identical shapes')
    t=y.shape[1]; out=np.full_like(q,np.nan,float); uniq=np.unique(f[f>=0])
    fold_scale={}
    for fold in uniq:
        sf=np.zeros(t,float)
        for j in range(t):
            train=(f[:,j]>=0)&(f[:,j]!=fold)&np.isfinite(y[:,j])&np.isfinite(q[:,j])
            if int(train.sum())<int(min_points):
                beta=0.0
            else:
                den=float(np.dot(q[train,j],q[train,j]))
                beta=float(np.dot(y[train,j],q[train,j])/den) if den>1e-12 else 0.0
            sf[j]=float(np.clip(beta,0.0,float(max_scale)))
            val=(f[:,j]==fold)&np.isfinite(q[:,j])
            out[val,j]=sf[j]*q[val,j]
        fold_scale[int(fold)]=sf
    final=np.zeros(t,float)
    for j in range(t):
        ok=(f[:,j]>=0)&np.isfinite(y[:,j])&np.isfinite(q[:,j])
        if int(ok.sum())<int(min_points):
            continue
        den=float(np.dot(q[ok,j],q[ok,j]))
        beta=float(np.dot(y[ok,j],q[ok,j])/den) if den>1e-12 else 0.0
        final[j]=float(np.clip(beta,0.0,float(max_scale)))
    return out, final, fold_scale

def fit_residual_matrix_expert(
    traits, blup=None, *, GRM=None, G=None, E=None,
    methods=("iterative_ridge", "softimpute"), n_splits=3,
    random_state=0, min_gain=.01, stack_min_gain=.02,
    model_kwargs=None, crossfit_refit_covariance=False,
    covariance_kwargs=None, solver_tol=1e-7, solver_maxiter=500,
    preconditioner="diag", store_oof=False,
    calibrate_oof=True, calibration_min_points=8, calibration_max_scale=1.0,
    prediction_target='genetic_mean', save_path=None,
):
    """Fit a matrix-completion expert using honest held-out residuals.

    Baselines are genetic means (training intercept + breeding value); explicit
    ``blup`` arrays must use the same phenotype scale. With ``GRM``, each fold masks validation phenotypes before covariance
    fitting (when requested), BLUP, and residual-model fitting.  Supplying
    fixed ``G``/``E`` treats them as externally estimated, fixed quantities;
    this assumption is reported in ``metadata``.

    Calls without ``GRM`` delegate to the historical API.  Its scores are
    conditional on the supplied BLUP and are not leakage-safe when that BLUP
    was fitted using the held-out phenotype cells.

    Parameters
    ----------
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

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('iterative_ridge', 'softimpute')``. Passed to
        ``_compat.fit_residual_matrix_expert`` as ``methods``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Source comparisons: ``int(n_splits) < 2``. See the rejection/fallback
        branches for enforcement. Passed to ``_compat.fit_residual_matrix_expert`` as
        ``n_splits``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``_compat.fit_residual_matrix_expert`` as ``random_state``.

    min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.01``. Passed to
        ``_compat.fit_residual_matrix_expert`` as ``min_gain``.

    stack_min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.02``. Passed to
        ``_compat.fit_residual_matrix_expert`` as ``stack_min_gain``.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to
        ``_compat.fit_residual_matrix_expert`` as ``model_kwargs``.

    crossfit_refit_covariance : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``False``. Passed to ``fit_conditional_residual_expert`` as
        ``crossfit_refit_covariance``.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-07``. Passed to ``fit_conditional_residual_expert`` as
        ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``500``. Passed to ``fit_conditional_residual_expert`` as ``solver_maxiter``.

    preconditioner : str
        Input supplied to ``fit_conditional_residual_expert`` as ``preconditioner``. See that
        callable's documented ``preconditioner`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``'diag'``.

    store_oof : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``. Passed to
        ``fit_conditional_residual_expert`` as ``store_oof``.

    calibrate_oof : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction. Default: ``True``. Passed to
        ``fit_conditional_residual_expert`` as ``calibrate_oof``.

    calibration_min_points : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction. Default: ``8``. Passed to ``fit_conditional_residual_expert``
        as ``calibration_min_points``.

    calibration_max_scale : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction. Default: ``1.0``. Passed to
        ``fit_conditional_residual_expert`` as ``calibration_max_scale``.

    prediction_target : str
        Baseline target: 'genetic_mean' uses intercept plus genetic prediction;
        'phenotype_prediction' includes the conditional residual prediction where supported. The
        residual expert is trained against the selected baseline. Default: ``'genetic_mean'``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``blup`` (sample, trait), ``G`` and ``E`` (trait_i, trait_j) when they
        are matrices, ``calibration`` (trait), and on the conditional path ``genetic_mean``
        and ``_observed_traits`` (sample, trait). ``attrs`` hold ``mode``, ``model`` (fitted
        estimator, in memory), ``method``, ``scores``, ``metadata``, ``GRM`` (echoed kernel),
        solver settings, ``prediction_target`` and, with ``store_oof``, ``oof``; ``None``
        members are kept in ``attrs``. ``predict_residual_matrix_expert`` accepts it.
    """
    params = _call_parameters(fit_residual_matrix_expert, locals())
    if prediction_target not in {'genetic_mean','phenotype_prediction'}:
        raise ValueError('prediction_target must be genetic_mean or phenotype_prediction')
    if GRM is None and prediction_target != 'genetic_mean':
        raise ValueError('conditional phenotype baselines require GRM and covariance')
    GRM, G, E = _pandas(GRM), _pandas(G), _pandas(E)
    Y = _frame(traits)
    if GRM is None:
        if blup is None:
            raise ValueError("blup is required when GRM is not supplied")
        warnings.warn(
            "Without GRM, residual-matrix scores are conditional on the supplied "
            "BLUP and are not leakage-safe if it used held-out phenotype cells.",
            UserWarning, stacklevel=2,
        )
        out = _result_dict(_compat.fit_residual_matrix_expert(
            Y, _frame(blup, like=Y), methods=methods, n_splits=n_splits,
            random_state=random_state, min_gain=min_gain,
            stack_min_gain=stack_min_gain, model_kwargs=model_kwargs,
        ))
        out["metadata"] = {
            "leakage_safe": False,
            "score_interpretation": "conditional_on_supplied_blup",
        }
        return maybe_save(_matrix_fit_dataset(out, params), save_path)

    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    if crossfit_refit_covariance and (G is not None or E is not None):
        warnings.warn(
            "G and E are ignored because crossfit_refit_covariance=True.",
            UserWarning, stacklevel=2,
        )
    if not crossfit_refit_covariance and (G is None or E is None):
        raise ValueError("G and E are required unless crossfit_refit_covariance=True")

    methods = tuple(methods)
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    cov_kwargs = {} if covariance_kwargs is None else dict(covariance_kwargs)
    if crossfit_refit_covariance and cov_kwargs.get("warm_start") is not None:
        raise ValueError(
            "covariance_kwargs['warm_start'] is not allowed during cross-fitting; "
            "it may contain information from held-out phenotypes"
        )
    if prediction_target == 'phenotype_prediction':
        from .conditional_residual import fit_conditional_residual_expert
        out=_result_dict(fit_conditional_residual_expert(Y,GRM,G,E,methods=methods,
            n_splits=n_splits,random_state=random_state,min_gain=min_gain,
            model_kwargs=model_kwargs,crossfit_refit_covariance=crossfit_refit_covariance,
            covariance_kwargs=cov_kwargs,solver_tol=solver_tol,solver_maxiter=solver_maxiter,
            preconditioner=preconditioner,store_oof=store_oof,calibrate_oof=calibrate_oof,
            calibration_min_points=calibration_min_points,calibration_max_scale=calibration_max_scale))
        out['_observed_traits']=Y.copy()
        return maybe_save(_matrix_fit_dataset(out, params), save_path)
    # Reject misspellings before doing expensive fold-specific BLUP fits.
    for number, method in enumerate(methods):
        _matrix_model(method, random_state + number, kwargs.get(method))
    folds = _compat._split_cells(Y.to_numpy(dtype=np.float64), n_splits, random_state)
    oof = {method: np.full(Y.shape, np.nan, dtype=np.float64) for method in methods}
    target = np.full(Y.shape, np.nan, dtype=np.float64)
    failures = {method: [] for method in methods}

    for fold in range(int(n_splits)):
        hold = folds == fold
        Y_fold = Y.mask(hold)
        if crossfit_refit_covariance:
            cov = genetic_varcov_PSD(GRM, Y_fold, **cov_kwargs)
            G_fold, E_fold = _covariances(cov)
        else:
            G_fold, E_fold = G, E
        B_fold = _frame(
            _blup(Y_fold, GRM, G_fold, E_fold, solver_tol,
                  solver_maxiter, preconditioner, prediction_target), like=Y,
        )
        residual = Y_fold.to_numpy(dtype=np.float64) - B_fold.to_numpy(dtype=np.float64)
        target[hold] = (Y.to_numpy(dtype=np.float64) - B_fold.to_numpy(dtype=np.float64))[hold]
        for number, method in enumerate(methods):
            try:
                model = _matrix_model(
                    method, random_state + 100 * fold + number, kwargs.get(method),
                )
                completed = np.asarray(model.fit_transform(residual), dtype=np.float64)
                oof[method][hold] = completed[hold]
            except Exception as exc:
                failures[method].append({"fold": fold, "error": repr(exc)})

    scored = np.isfinite(target)
    scores = {"blup": _compat._corr_rmse(target[scored], np.zeros(scored.sum()))}
    viable = []
    calibrated_oof = {}
    calibration = {}
    calibration_by_fold = {}
    for method in methods:
        qraw=oof[method]
        if calibrate_oof:
            beta = _final_oof_calibration(target,qraw,
                min_points=calibration_min_points,max_scale=calibration_max_scale)
            qscore=qraw.copy(); beta_fold={}
        else:
            qscore=qraw.copy(); beta=np.ones(Y.shape[1],float); beta_fold={}
        calibrated_oof[method]=qscore; calibration[method]=beta; calibration_by_fold[method]=beta_fold
        ok = scored & np.isfinite(qscore)
        complete = bool(np.array_equal(ok, scored))
        if ok.any():
            score = _compat._corr_rmse(target[ok], qscore[ok])
            gain, z_score = _compat._paired_rmse_gain(
                target[ok], np.zeros(ok.sum()), qscore[ok],
            )
            raw_ok=scored & np.isfinite(qraw)
            raw_score=_compat._corr_rmse(target[raw_ok],qraw[raw_ok]) if raw_ok.any() else {"rmse":np.inf,"r":np.nan,"n":0}
        else:
            score = {"n": 0, "r": np.nan, "rmse": np.inf}
            raw_score={"n":0,"r":np.nan,"rmse":np.inf}
            gain, z_score = -np.inf, -np.inf
        score.update(gain_vs_blup=gain, gain_z=z_score,
                     complete_heldout_coverage=complete,
                     n_expected=int(scored.sum()),
                     raw_rmse=float(raw_score.get("rmse",np.inf)),
                     raw_r=float(raw_score.get("r",np.nan)),
                     calibration_mean=float(np.mean(beta)),
                     calibration_nonzero=int(np.sum(beta>0)))
        scores[method] = score
        if complete and gain >= float(min_gain) and z_score >= 1.96:
            viable.append(method)

    # Full-data fitting happens only after OOF selection is complete.
    if crossfit_refit_covariance:
        final_cov = genetic_varcov_PSD(GRM, Y, **cov_kwargs)
        final_G, final_E = _covariances(final_cov)
        covariance_source = "crossfit_refit"
    else:
        final_G, final_E = G, E
        covariance_source = "external_fixed"
    B = _frame(_blup(Y, GRM, final_G, final_E, solver_tol,
                     solver_maxiter, preconditioner, prediction_target), like=Y)
    selected = min(viable, key=lambda name: scores[name]["rmse"]) if viable else "blup"
    model = None
    mode = "blup"
    if selected != "blup":
        full_residual = Y.to_numpy(dtype=np.float64) - B.to_numpy(dtype=np.float64)
        model = _matrix_model(
            selected, random_state + 9000, kwargs.get(selected),
        ).fit(full_residual)
        mode = "single"
    selected_calibration=(calibration.get(selected,np.zeros(Y.shape[1],float))
                          if selected != "blup" else np.zeros(Y.shape[1],float))
    out = {
        "mode": mode, "model": model, "method": selected, "scores": scores,
        "calibration": np.asarray(selected_calibration,float),
        "blup": B, "GRM": GRM, "G": final_G, "E": final_E,
        "solver_tol": solver_tol, "solver_maxiter": solver_maxiter,
        "preconditioner": preconditioner,
        "prediction_target": prediction_target,
        "metadata": {
            "leakage_safe": True, "selection": "cell_crossfit",
            "covariance_source": covariance_source,
            "fixed_covariance_assumption": not crossfit_refit_covariance,
            "leakage_scope": ("phenotype-label-safe conditional on externally supplied G/E"
                              if not crossfit_refit_covariance else "phenotype-label-safe"),
            "model_failures": failures,
            "n_splits": int(n_splits),
            "oof_amplitude_calibration": False,
            "selection_amplitude": "raw_oof_conservative_gate",
            "deployment_amplitude_calibration": bool(calibrate_oof),
            "calibration_min_points": int(calibration_min_points),
            "calibration_max_scale": float(calibration_max_scale),
        },
    }
    if store_oof:
        out["oof"] = {"fold": folds.copy(), "target": target.copy(),
                      "prediction_raw": {k: v.copy() for k, v in oof.items()},
                      "prediction": {k: v.copy() for k, v in calibrated_oof.items()},
                      "calibration": {k: np.asarray(v,float).copy() for k,v in calibration.items()},
                      "calibration_by_fold": calibration_by_fold}
    return maybe_save(_matrix_fit_dataset(out, params), save_path)


def predict_residual_matrix_expert(fit, traits, blup=None, *, GRM=None):
    """Predict with a fitted residual matrix expert.

    Parameters
    ----------
    fit : xarray.Dataset or mapping
        Fit returned by ``fit_residual_matrix_expert`` (its Dataset; arrays are read from data
        variables and estimators/settings from ``attrs``), or a legacy mapping with the same
        key names, for the same animal and trait axes. Passed to ``mvPredict`` as ``solver_tol``.

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

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(result, index=Y.index, columns=Y.columns)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    fit = _result_dict(fit); traits = _pandas(traits); GRM = _pandas(GRM)
    columns = fit["blup"].columns if isinstance(fit.get("blup"), pd.DataFrame) else None
    Y = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits, columns=columns)
    conditional=fit.get('prediction_target')=='phenotype_prediction'
    state=None
    if conditional and blup is None:
        cached=fit.get('_observed_traits')
        if GRM is None and isinstance(cached,pd.DataFrame) and cached.equals(Y):
            B=fit['blup'].reindex(index=Y.index,columns=Y.columns)
            genetic=fit['genetic_mean'].reindex(index=Y.index,columns=Y.columns)
        else:
            kernel=fit['GRM'] if GRM is None else GRM
            state=mvPredict(Y,kernel,fit['G'],fit['E'],accuracy='exact',kernel_rank='full',
                solver_tol=fit.get('solver_tol',1e-7),solver_maxiter=fit.get('solver_maxiter',500),
                preconditioner=fit.get('preconditioner','diag'))
            B=_frame(state['phenotype_prediction'],like=Y);genetic=_frame(state['genetic_mean'],like=Y)
    elif blup is not None:
        B=_frame(blup,like=Y)
    elif GRM is not None:
        B=_frame(_blup(Y,GRM,fit['G'],fit['E'],fit.get('solver_tol',1e-7),
            fit.get('solver_maxiter',500),fit.get('preconditioner','diag'),
            fit.get('prediction_target','genetic_mean')),like=Y)
    else:
        B=fit['blup'].reindex(index=Y.index,columns=Y.columns)
    if fit['mode']=='blup':return B.copy()
    if conditional:
        if blup is not None:
            kernel=fit['GRM'] if GRM is None else GRM
            state=mvPredict(Y,kernel,fit['G'],fit['E'],accuracy='exact',kernel_rank='full',
                solver_tol=fit.get('solver_tol',1e-7),solver_maxiter=fit.get('solver_maxiter',500),
                preconditioner=fit.get('preconditioner','diag'))
            genetic=_pandas(state['genetic_mean'])
        residual=Y.to_numpy(float)-genetic.to_numpy(float)
    else:residual=Y.to_numpy(float)-B.to_numpy(float)
    addition = np.asarray(fit["model"].transform(residual), dtype=np.float64)
    cal=np.asarray(fit.get("calibration",np.ones(Y.shape[1])),float).reshape(1,-1)
    if cal.shape[1] != Y.shape[1]:
        raise ValueError("matrix-expert calibration does not match the trait dimension")
    addition = addition * cal
    result = B.to_numpy(dtype=np.float64) + addition
    no_phenotype = ~np.isfinite(Y.to_numpy(dtype=np.float64)).any(axis=1)
    result[no_phenotype] = B.to_numpy(dtype=np.float64)[no_phenotype]
    return pd.DataFrame(result, index=Y.index, columns=Y.columns)


__all__ = ["fit_residual_matrix_expert", "predict_residual_matrix_expert"]
