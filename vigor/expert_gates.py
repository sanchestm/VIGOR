"""Independent validation gates for genomic corrections to public predictions."""
import numpy as np
import pandas as pd

from .results import maybe_save, result_dataset


def _model_and_gates(fit):
    """Return the fitted genomic-correction model and its frozen deployment gates.

    Parameters
    ----------
    fit : xarray.Dataset or fitted object
        Dataset returned by ``gate_genomic_kernel_expert`` (model in ``attrs['model']`` and
        gates in ``attrs['deployment_gates']``) or a fitted estimator exposing ``predict`` and,
        optionally, a ``deployment_gates`` attribute.
    """
    import xarray as xr
    if isinstance(fit, xr.Dataset):
        return fit.attrs["model"], fit.attrs.get("deployment_gates")
    return fit, getattr(fit, 'deployment_gates', None)


def gate_genomic_kernel_expert(fit, targets, baseline, validation_mask, context_traits,
                              *, min_relative_gain=.005, min_z=1.96, save_path=None):
    """Freeze separate unseen/partial gates using independent inner outcomes.

    The model/discovery must have been fitted without these validation outcomes.
    Baseline is the conditional phenotype prediction from the declared context.
    The correction is genomic; the gate does not create residual extrapolation.

    Parameters
    ----------
    fit : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    targets : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    validation_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    context_traits : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    min_relative_gain : float
        Minimum held-out prediction improvement required for promotion. Positive values require
        a gain relative to the baseline; the evaluator defines whether its loss reduction is
        relative or absolute. Default: ``0.005``.

    min_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``1.96``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables on dimension ``gate`` (``unseen``, ``partial``): ``accepted``,
        ``relative_rmse_gain``, ``paired_animal_z``, ``n_cells`` and ``n_animals``. ``attrs``
        hold ``model`` (the fitted estimator, which also receives ``deployment_gates``; in
        memory only), ``deployment_gates`` (the per-gate mapping) and ``dependence_note``.
        ``predict_genomic_kernel_expert`` accepts this Dataset.
    """
    from .multivariate import _paired_rmse_gain
    fit, _ = _model_and_gates(fit)
    Y=np.asarray(targets,float);B=np.asarray(baseline,float);M=np.asarray(validation_mask,bool)
    if Y.shape!=B.shape or M.shape!=Y.shape:raise ValueError('aligned target/baseline/mask required')
    C=np.asarray(context_traits,float)
    correction=np.asarray(fit.predict(np.zeros_like(B)),float)
    unseen=~np.isfinite(C).any(1);gates={}
    for name,rows in [('unseen',unseen),('partial',~unseen)]:
        mask=M&rows[:,None]&np.isfinite(Y)&np.isfinite(B)&np.isfinite(correction)
        if mask.sum()<8:gain=z=float('-inf')
        else:
            gain,_=_paired_rmse_gain(Y[mask],B[mask],(B+correction)[mask])
            improvement=np.where(mask,(Y-B)**2-(Y-B-correction)**2,0.).sum(1)
            counts=mask.sum(1);valid=counts>0;d=improvement[valid]/counts[valid]
            z=float(d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))) if len(d)>1 and d.std()>0 else float('-inf')
        gates[name]={'accepted':bool(gain>=min_relative_gain and z>=min_z),
                     'relative_rmse_gain':float(gain),'paired_animal_z':float(z),'n_cells':int(mask.sum()),'n_animals':int(mask.any(1).sum()),'dependence_note':'trait errors grouped by animal; kinship uncertainty requires external cluster evaluation'}
    fit.deployment_gates=gates
    names=list(gates); fields=('accepted','relative_rmse_gain','paired_animal_z','n_cells','n_animals')
    ds=result_dataset(
        {k:np.asarray([gates[g][k] for g in names]) for k in fields},
        dims={k:('gate',) for k in fields}, coords={'gate':names},
        var_attrs={'accepted':{'long_name':'gate accepted for deployment'},
                   'relative_rmse_gain':{'long_name':'relative RMSE gain over the baseline','units':'1'},
                   'paired_animal_z':{'long_name':'animal-paired improvement z statistic','units':'1'},
                   'n_cells':{'long_name':'validation cells scored','units':'count'},
                   'n_animals':{'long_name':'validation animals scored','units':'count'}},
        attrs={'model':fit,'deployment_gates':gates,'dependence_note':gates['unseen']['dependence_note']},
        title='Genomic kernel expert deployment gates',
        software='vigor.expert_gates.gate_genomic_kernel_expert',
        parameters={'min_relative_gain':float(min_relative_gain),'min_z':float(min_z)})
    return maybe_save(ds, save_path)


def predict_genomic_kernel_expert(fit, genetic_mean, phenotype_baseline, context_traits,
                                  *, save_path=None):
    """Return gated genomic correction and conditional phenotype prediction.

    Genetic correction is distinct from E posterior. A partial gate changes
    phenotype prediction only; it is not evidence of genetic truth improvement.

    Parameters
    ----------
    fit : xarray.Dataset or fitted object
        Dataset returned by ``gate_genomic_kernel_expert`` (model and gates read from
        ``attrs``), or a fitted estimator exposing ``predict`` and optionally
        ``deployment_gates``, for the same animal and trait axes.

    genetic_mean : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    phenotype_baseline : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    context_traits : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``genomic_correction``, ``genetic_mean`` and ``phenotype_prediction``
        (sample, trait; labeled when ``context_traits`` is labeled). ``attrs`` hold ``gates``
        (None when the model was never gated) and ``interpretation``.
    """
    import xarray as xr
    if isinstance(context_traits,xr.DataArray):context_traits=context_traits.to_pandas()
    fit,gates=_model_and_gates(fit)
    B=np.asarray(phenotype_baseline,float);GM=np.asarray(genetic_mean,float)
    correction=np.asarray(fit.predict(np.zeros_like(B)),float)
    unseen=~np.isfinite(np.asarray(context_traits,float)).any(1)
    if gates is not None:
        correction[unseen]*=float(gates['unseen']['accepted'])
        correction[~unseen]*=float(gates['partial']['accepted'])
    def frame(x):
        """Coerce the supplied values to a labeled DataFrame for the enclosing model.

        Parameters
        ----------
        x : array-like
            Numerical operand in ``pd.DataFrame(x, index=context_traits.index,
            columns=context_traits.columns)``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``pd.DataFrame(x, index=context_traits.index, columns=context_traits.columns) if isinstance(context_traits, pd.DataFrame) else x``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return pd.DataFrame(x,index=context_traits.index,columns=context_traits.columns) if isinstance(context_traits,pd.DataFrame) else x
    unit={'units':'phenotype units'}
    ds=result_dataset({'genomic_correction':frame(correction),'genetic_mean':frame(GM+correction),
            'phenotype_prediction':frame(B+correction)},
        dims={k:('sample','trait') for k in ('genomic_correction','genetic_mean','phenotype_prediction')},
        var_attrs={'genomic_correction':{'long_name':'gated genomic correction',**unit},
                   'genetic_mean':{'long_name':'genetic mean plus gated genomic correction',**unit},
                   'phenotype_prediction':{'long_name':'conditional phenotype prediction plus gated genomic correction',**unit}},
        attrs={'gates':gates,
               'interpretation':'fixed genomic correction; residual posterior remains within-animal'},
        title='Genomic kernel expert prediction',
        software='vigor.expert_gates.predict_genomic_kernel_expert',parameters={'gated':gates is not None})
    return maybe_save(ds, save_path)
