"""Nested, supervised corrections to conditional phenotype predictions.

Available-trait features are innovations relative to the genetic mean. Targets
are held-out innovations relative to the conditional phenotype posterior. These
are different quantities: observed conditional posterior residuals are zero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def residual_features(residual, target):
    """Observed genetic innovations and their observation indicators.

    The target trait is explicitly excluded. Its outcome must also have been
    masked before calculating the supplied genetic predictions.

    Parameters
    ----------
    residual : str
        Residual covariance model: 'diag' uses independent trait noise, 'lkj' uses an LKJ
        correlation prior, and 'fa'/'factor' uses low-rank factors plus specific variances.
        Check add_trait/build for scalar-model restrictions.

    target : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    Returns
    -------
    object
        Computed result from ``np.column_stack([np.where(seen, r, 0.0), seen.astype(float)])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    r=np.asarray(residual,float).copy()
    r[:,int(target)]=np.nan
    seen=np.isfinite(r)
    return np.column_stack([np.where(seen,r,0.),seen.astype(float)])


def _paired_animal_gain_z(target,prediction):
    """Group dependent trait-cell squared-error gains by observed animal.

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
    """
    y=np.asarray(target,float);p=np.asarray(prediction,float);ok=np.isfinite(y)&np.isfinite(p)
    count=ok.sum(1);active=count>0
    gains=np.where(ok,y*y-(y-p)**2,0.).sum(1)[active]/count[active]
    if len(gains)<2 or gains.std(ddof=1)<=0:return float('-inf')
    return float(gains.mean()/(gains.std(ddof=1)/np.sqrt(len(gains))))


class ConditionalResidualModel:
    """Per-trait regression fitted to OOF features and OOF correction targets."""
    def __init__(self,method='oof_ridge',random_state=0,kwargs=None):
        """Initialize ConditionalResidualModel with the inputs and resource settings below.

        Parameters
        ----------
        method : callable or str
            Algorithm/estimator selected for the operation described above. See the source-derived
            choices and called model factory for the accepted methods; callable methods must obey
            that factory's fit/predict or objective contract. Default: ``'oof_ridge'``.

        random_state : int or None
            Seed controlling randomized sampling, folds or projections for reproducibility. None
            uses the selected random-number backend without a fixed seed. Default: ``0``.

        kwargs : mapping
            Additional keyword settings forwarded to the selected estimator/writer. See the called
            implementation for its accepted names and value domains; these are not unrestricted
            scientific options. Default: ``None``. None delegates to the local fallback shown in
            this body; it is not an additional numeric/category value.
        """
        self.method=method;self.random_state=random_state;self.kwargs=kwargs or {}
        self.models={};self.training_target=None

    def fit_pairs(self, features, targets):
        """Fit pairs.

        Parameters
        ----------
        features : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        targets : str, index or sequence
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering.

        Returns
        -------
        object
            Computed result from ``self``; see the summary and the called implementation for its labeled/numerical representation.
        """
        t=targets.shape[1];self.n_traits=t;self.training_target=np.array(targets,copy=True)
        for j in range(t):
            X=features[j];y=targets[:,j];ok=np.isfinite(y)&np.isfinite(X).all(1)
            if ok.sum()<8:continue
            if self.method in {'oof_ridge','iterative_ridge'}:
                model=make_pipeline(StandardScaler(),Ridge(alpha=float(self.kwargs.get('alpha',10.))))
            elif self.method=='oof_lightgbm':
                from lightgbm import LGBMRegressor
                options=dict(n_estimators=96,num_leaves=7,min_child_samples=30,
                             learning_rate=.05,reg_lambda=10.,n_jobs=1,verbosity=-1)
                options.update(self.kwargs)
                model=LGBMRegressor(random_state=self.random_state+j,**options)
            else:raise ValueError('conditional residual methods: oof_ridge, oof_lightgbm')
            model.fit(X[ok],y[ok]);self.models[j]=model
        return self

    def transform(self,residual):
        """Apply the stored encoding/standardization or model transform to aligned inputs.

        Parameters
        ----------
        residual : str
            Residual covariance model: 'diag' uses independent trait noise, 'lkj' uses an LKJ
            correlation prior, and 'fa'/'factor' uses low-rank factors plus specific variances.
            Check add_trait/build for scalar-model restrictions.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        r=np.asarray(residual,float);out=np.zeros(r.shape,float)
        for j,model in self.models.items():out[:,j]=model.predict(residual_features(r,j))
        out[~np.isfinite(r).any(1)]=0.
        return out


def fit_conditional_residual_expert(Y, GRM, G, E, *, methods, n_splits,
        random_state, min_gain, model_kwargs, crossfit_refit_covariance,
        covariance_kwargs, solver_tol, solver_maxiter, preconditioner, store_oof,
        calibrate_oof, calibration_min_points, calibration_max_scale):
    """Gate with outer cell folds; train each candidate on inner OOF targets.

    The scored outer cells enter neither covariance fits nor any inner feature,
    target, model or scale. Final training uses collected OOF pairs. Covariance
    supplied as fixed is explicitly treated as an independent external input.

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

    methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    min_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options.

    crossfit_refit_covariance : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only.

    covariance_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence.

    preconditioner : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller.

    store_oof : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary.

    calibrate_oof : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction.

    calibration_min_points : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction. Passed to ``_final_oof_calibration`` as ``min_points``.

    calibration_max_scale : bool, int or float
        OOF calibration control: calibrate_oof enables training-only affine calibration,
        min_points requires sufficient held-out support and max_scale bounds the fitted
        multiplicative correction. Passed to ``_final_oof_calibration`` as ``max_scale``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from . import multivariate as mv
    from .residual_matrix import _covariances,_final_oof_calibration
    methods=tuple(methods)
    for method in methods:
        if method not in {'oof_ridge','iterative_ridge','oof_lightgbm'}:
            raise ValueError('conditional residual methods: oof_ridge, oof_lightgbm')
    kw={} if covariance_kwargs is None else dict(covariance_kwargs)
    if kw.get('warm_start') is not None:
        raise ValueError('conditional crossfit cannot use a phenotype-informed warm_start')
    kw.setdefault('n_boot',0);kw.setdefault('progress',False)
    modelkw={} if model_kwargs is None else dict(model_kwargs)
    array=Y.to_numpy(float);n,t=array.shape

    def state(frame):
        """Fit the training-fold covariance and prepare its conditional prediction state.

        Parameters
        ----------
        frame : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        Returns
        -------
        object
            Computed result from ``(pred, g, e)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if crossfit_refit_covariance:
            fit=mv.genetic_varcov_PSD(GRM,frame,**kw);g,e=_covariances(fit)
        else:g,e=G,E
        pred=mv.mvPredict(frame,GRM,g,e,accuracy='exact',kernel_rank='full',
            solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner)
        return pred,g,e

    def pairs(frame,seed):
        """Construct training-only residual-learning pairs for the enclosing crossfit procedure.

        Parameters
        ----------
        frame : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        seed : int or None
            Seed controlling randomized sampling, folds or projections for reproducibility. None
            uses the selected random-number backend without a fixed seed.

        Returns
        -------
        object
            Computed result from ``(features, target, folds)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        folds=mv._split_cells(frame.to_numpy(float),n_splits,seed)
        features=[np.full((n,2*t),np.nan) for _ in range(t)]
        target=np.full((n,t),np.nan)
        for k in range(int(n_splits)):
            hold=folds==k;masked=frame.mask(hold);pred,_,_=state(masked)
            r=masked.to_numpy(float)-np.asarray(pred['genetic_mean'],float)
            delta=frame.to_numpy(float)-np.asarray(pred['phenotype_prediction'],float)
            for j in range(t):features[j][hold[:,j]]=residual_features(r,j)[hold[:,j]]
            target[hold]=delta[hold]
        return features,target,folds

    folds=mv._split_cells(array,n_splits,random_state)
    target=np.full((n,t),np.nan);oof={m:np.full((n,t),np.nan) for m in methods}
    final_features=[np.full((n,2*t),np.nan) for _ in range(t)]
    failures={m:[] for m in methods}
    if methods:
        for k in range(int(n_splits)):
            hold=folds==k;masked=Y.mask(hold);pred,_,_=state(masked)
            r=masked.to_numpy(float)-np.asarray(pred['genetic_mean'],float)
            target[hold]=(array-np.asarray(pred['phenotype_prediction'],float))[hold]
            for j in range(t):final_features[j][hold[:,j]]=residual_features(r,j)[hold[:,j]]
            train_features,train_target,_=pairs(masked,random_state+1009*(k+1))
            for number,method in enumerate(methods):
                try:
                    model=ConditionalResidualModel(method,random_state+100*k+number,modelkw.get(method))
                    model.fit_pairs(train_features,train_target)
                    addition=model.transform(r);oof[method][hold]=addition[hold]
                except Exception as exc:failures[method].append({'fold':k,'error':repr(exc)})
    scored=np.isfinite(target)
    scores={'blup':mv._corr_rmse(target[scored],np.zeros(scored.sum()))}
    viable=[];calibration={}
    for method in methods:
        p=oof[method];ok=scored&np.isfinite(p);complete=bool(np.array_equal(ok,scored))
        if ok.any():
            score=mv._corr_rmse(target[ok],p[ok]);gain,_=mv._paired_rmse_gain(target[ok],np.zeros(ok.sum()),p[ok]);z=_paired_animal_gain_z(target,p)
        else:score={'rmse':np.inf,'n':0,'r':np.nan};gain=z=-np.inf
        score.update(gain_vs_blup=gain,gain_z=z,gain_z_unit='animal; dependent trait cells grouped',complete_heldout_coverage=complete,n_expected=int(scored.sum()))
        scores[method]=score
        if complete and gain>=float(min_gain) and z>=1.96:viable.append(method)
        calibration[method]=(_final_oof_calibration(target,p,min_points=calibration_min_points,
                           max_scale=calibration_max_scale) if calibrate_oof else np.ones(t))
    selected=min(viable,key=lambda m:scores[m]['rmse']) if viable else 'blup'
    model=None
    if selected!='blup':
        model=ConditionalResidualModel(selected,random_state+9000,modelkw.get(selected))
        model.fit_pairs(final_features,target)
    final,g,e=state(Y)
    out={'mode':'single' if model is not None else 'blup','method':selected,'model':model,
         'scores':scores,'blup':final['phenotype_prediction'],'genetic_mean':final['genetic_mean'],
         'GRM':GRM,'G':g,'E':e,'prediction_target':'phenotype_prediction',
         'calibration':calibration.get(selected,np.zeros(t)),
         'solver_tol':solver_tol,'solver_maxiter':solver_maxiter,'preconditioner':preconditioner,
         'metadata':{'leakage_safe':True,'selection':'nested_cell_crossfit',
                     'features':'observed genetic-mean innovations plus mask',
                     'target':'OOF conditional phenotype innovations',
                     'final_training':'OOF features and OOF targets',
                     'covariance_source':'crossfit_refit' if crossfit_refit_covariance else 'external_fixed',
                     'fixed_covariance_assumption':not crossfit_refit_covariance,
                     'model_failures':failures,'selection_amplitude':'raw_oof_conservative_gate',
                     'gate_dependence':'animal-grouped paired squared-error gains; kinship cluster uncertainty remains external','deployment_amplitude_calibration':bool(calibrate_oof)}}
    if store_oof:
        out['oof']={'fold':folds,'target':target,'features':final_features,
                    'prediction_raw':oof,'prediction':oof,'calibration':calibration}
    return out
