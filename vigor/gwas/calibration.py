"""Retrospective GRM summaries and grouped out-of-sample calibration.

Threshold accuracy alone does not prove FWER calibration. Models remain
unvalidated unless the caller also supplies independent null-FWER evidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .effective_tests import effective_tests, meff_to_threshold
from .thresholds import read_legacy_thresholds


def calibration_table_from_projects(projects, *, alpha=.05):
    """Build rows from mappings with project, grm, and threshold_file keys.

    Optional cohort/sex/sample/QC identifiers are retained. Each ``grm`` is a
    matrix or chromosome-to-matrix mapping; caller controls its normalization.

    Parameters
    ----------
    projects : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.05``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(rows)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rows=[]
    for project in projects:
        threshold=read_legacy_thresholds(project['threshold_file']).loc[float(alpha)]
        result=effective_tests(method='grm_spectral',grm=project['grm'],cache=False)
        row={k:v for k,v in project.items() if k not in {'grm','threshold_file'}}
        row.update({'alpha':float(alpha),'empirical_threshold':float(threshold.p_threshold),
                    'empirical_equivalent_meff':float(threshold.empirical_equivalent_meff)})
        frame=result.chromosome_results
        for column in frame.columns:
            if column != 'chromosome' and pd.api.types.is_numeric_dtype(frame[column]):
                row[column]=float(frame[column].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _fit(x,y,kind):
    """Fit the enclosing calibration model on training calibration points.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x @ y / (x @ x)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    y : array-like
        Numerical operand in ``x @ y / (x @ x)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    kind : str
        Edge/evidence/input interpretation selected by this constructor/helper. See the local
        branches and the source-derived option descriptions below; each builder admits only its
        own subset, not a package-wide enum. Source-recognized literal values: ``linear``,
        ``power``, ``through_origin``. These select the corresponding branches below; comparison
        evidence alone does not establish an exhaustive enum.
    """
    if kind=='through_origin':
        return 0.,float(x@y/(x@x)),1.
    if kind=='linear':
        a,b=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
        return float(a),float(b),1.
    if kind=='power':
        a,b=np.linalg.lstsq(np.column_stack([np.ones(len(x)),np.log(x)]),np.log(y),rcond=None)[0]
        return 0.,float(np.exp(a)),float(b)
    raise ValueError('kind must be linear, through_origin, or power')


def _predict(x,params):
    """Apply the fitted calibration model to new statistic values.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``a + b * np.asarray(x) ** p``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    params : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration.
    """
    a,b,p=params
    return a+b*np.asarray(x)**p


def fit_grm_calibration(table, *, feature='participation_rank', group_col='project',
                        kind='through_origin', training_population, grm_normalization,
                        fwer_evidence=None, median_threshold_error=.05,
                        p95_threshold_error=.15):
    """Fit a simple model and leave each project/cohort/stratum out in turn.

    ``fwer_evidence`` is a mapping containing ``independent=True``, ``alpha``,
    ``false_positive_replicates`` and ``n_replicates`` for a held-out null
    experiment using the predicted threshold. A binomial confidence interval
    must include nominal alpha before the calibration is marked validated.
    No software routine can verify the provenance of user-supplied evidence.

    Parameters
    ----------
    table : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    feature : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``'participation_rank'``.

    group_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'project'``.

    kind : str
        Edge/evidence/input interpretation selected by this constructor/helper. See the local
        branches and the source-derived option descriptions below; each builder admits only its
        own subset, not a package-wide enum. Default: ``'through_origin'``. Source-recognized
        literal values: ``linear``, ``power``. These select the corresponding branches below;
        comparison evidence alone does not establish an exhaustive enum.

    training_population : object (see the cited input contract)
        Input for the operation ``str(training_population)``. See
        ``vigor.gwas.calibration.fit_grm_calibration`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    grm_normalization : object (see the cited input contract)
        Input for the operation ``str(grm_normalization)``. See
        ``vigor.gwas.calibration.fit_grm_calibration`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    fwer_evidence : object (see the cited input contract)
        Input for the operation ``dict(fwer_evidence)``. See
        ``vigor.gwas.calibration.fit_grm_calibration`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    median_threshold_error : array-like or numeric scalar
        Input for the operation ``np.median(errors) < median_threshold_error``. See
        ``vigor.gwas.calibration.fit_grm_calibration`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.05``.

    p95_threshold_error : array-like or numeric scalar
        Input for the operation ``np.quantile(errors, 0.95) < p95_threshold_error``. See
        ``vigor.gwas.calibration.fit_grm_calibration`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.15``.

    Returns
    -------
    dict
        Result fields: ``model``, ``held_out``, ``metrics``. Field arrays preserve the corresponding input axis order.
    """
    frame=pd.DataFrame(table).copy()
    required={feature,group_col,'alpha','empirical_threshold','empirical_equivalent_meff'}
    if not required.issubset(frame):
        raise ValueError(f'missing calibration columns: {sorted(required-set(frame))}')
    if frame[group_col].isna().any() or frame[group_col].nunique()<3:
        raise ValueError('at least three independently held-out groups are required')
    if frame.alpha.nunique()!=1:
        raise ValueError('fit separate calibrations for each alpha')
    alpha=float(frame.alpha.iloc[0])
    x=frame[feature].to_numpy(float); y=frame.empirical_equivalent_meff.to_numpy(float)
    if np.any(~np.isfinite(x)) or np.any(x<=0) or np.any(~np.isfinite(y)) or np.any(y<=0):
        raise ValueError('calibration features and empirical Meff must be finite and positive')
    pred=np.full(len(frame),np.nan)
    groups=frame[group_col].to_numpy()
    for group in pd.unique(groups):
        train=groups!=group
        if kind in {'linear','power'} and np.unique(x[train]).size<2:
            raise ValueError('held-out training design lacks feature variation')
        pred[~train]=_predict(x[~train],_fit(x[train],y[train],kind))
    if np.any(pred<=0) or np.any(~np.isfinite(pred)):
        raise ValueError('held-out calibration predicts nonpositive or invalid Meff')
    p=np.array([meff_to_threshold(v,(alpha,))[alpha] for v in pred])
    actual=frame.empirical_threshold.to_numpy(float)
    if np.any(~np.isfinite(actual)) or np.any((actual<=0)|(actual>=1)):
        raise ValueError('empirical thresholds must lie in (0,1)')
    errors=np.abs(-np.log10(p)+np.log10(actual))
    threshold_pass=bool(np.median(errors)<median_threshold_error and np.quantile(errors,.95)<p95_threshold_error)
    fwer_pass=False; fwer_summary=None
    if fwer_evidence is not None:
        from scipy.stats import binomtest
        evidence=dict(fwer_evidence)
        nr=int(evidence['n_replicates']); nf=int(evidence['false_positive_replicates'])
        if nr<1 or not 0<=nf<=nr:
            raise ValueError('invalid null-FWER replicate counts')
        ci=binomtest(nf,nr).proportion_ci()
        fwer_pass=bool(evidence.get('independent',False) and float(evidence['alpha'])==alpha
                       and nr>=1000 and ci.low<=alpha<=ci.high)
        fwer_summary={**evidence,'observed_fwer':nf/nr,'ci_low':ci.low,'ci_high':ci.high}
    params=_fit(x,y,kind)
    denom=np.sum((y-y.mean())**2)
    metrics={'r2':float(1-np.sum((pred-y)**2)/denom) if denom else None,
             'rmse_meff':float(np.sqrt(np.mean((pred-y)**2))),
             'median_abs_neglog10_error':float(np.median(errors)),
             'p95_abs_neglog10_error':float(np.quantile(errors,.95)),
             'median_relative_error':float(np.median(np.abs(pred/y-1)))}
    model={'kind':kind,'feature':feature,'intercept':params[0],'coefficient':params[1],'power':params[2],
           'validated':bool(threshold_pass and fwer_pass),'training_population':str(training_population),
           'grm_normalization':str(grm_normalization),'alpha':alpha,
           'validation_relative_error':metrics['median_relative_error'],
           'validation':metrics,'validation_group':group_col,'n_groups':int(frame[group_col].nunique()),
           'feature_range':[float(x.min()),float(x.max())], 'fwer_evidence':fwer_summary,
           'calibration_version':1}
    frame['predicted_meff']=pred; frame['predicted_threshold']=p; frame['abs_neglog10_error']=errors
    return {'model':model,'held_out':frame,'metrics':metrics}
