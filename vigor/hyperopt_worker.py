"""Scientific work executed inside supervised, spawn-safe HPO processes."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd

from .hyperopt_cv import load_masks, preprocess, score_predictions, aggregate
from .hyperopt_runtime import worker_context, atomic_pickle


def load_data(directory):
    """Open read-only shared phenotype/kernel arrays and labeled metadata.

    directory is a supervisor-owned input directory. Dense arrays are mmap'd;
    supported operator pickles are trusted internal state. O(metadata) plus
    operator-specific deserialization, without serializing dense kernels per job.
    """
    directory=Path(directory);meta=json.loads((directory/'data.json').read_text())
    values=np.load(directory/'traits.npy',mmap_mode='r',allow_pickle=False)
    kernels={}
    for name,item in meta['kernels'].items():
        if item['kind']=='array':kernels[name]=np.load(directory/item['file'],mmap_mode='r',allow_pickle=False)
        else:
            with (directory/item['file']).open('rb') as stream:kernels[name]=pickle.load(stream)
    return meta,values,kernels


def covariance_view(fit):
    """Return the deployed prediction component matrices and E.

    The h2 prior, when selected, is already part of the fitted covariance, so
    there is a single deployed view. O(c*t^2) representation conversion,
    preserving named kernels.
    """
    return fit.get('G_components_prediction',fit['G_components']),fit.get('E_prediction',fit['E'])


def fit_diagnostics(fit):
    """Extract compact convergence/rank/loading metadata; no T-by-T attrs.

    Time O(number of components), plus precomputed scalar result metadata.
    Missing diagnostics remain null rather than fabricated iteration counts.
    """
    result=fit.get('result')
    def clean(value):
        """Normalize compact scalar/dict diagnostics, preserving unavailable nulls."""
        if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    return clean({'success':bool(fit.get('success',False)), 'optimizer_converged':bool(fit.get('optimizer_converged',True)),
        'message':str(fit.get('message','')),'nit':getattr(result,'nit',fit.get('nit')),
        'nfev':getattr(result,'nfev',None),'objective':fit.get('fun'),
        'gradient_norm':None if getattr(result,'jac',None) is None else float(np.linalg.norm(result.jac)),
        'requested_rank_G':fit.get('rank_G_requested'),'requested_rank_E':fit.get('rank_E_requested'),
        'achieved_rank_G':fit.get('rank_G'),'achieved_rank_E':fit.get('rank_E'),
        'nonzero_loading_columns':fit.get('factor_effective_ranks'),'loading_density':fit.get('factor_loading_density')})


def residual_condition(fit):
    """Smallest eigenvalue of the prediction residual correlation matrix.

    V = K (x) G + I (x) E with PSD terms, so lambda_min(V) >= lambda_min(E):
    a near-singular E bounds the mvBLUP conditioning from below. O(t^3) on a
    t x t matrix, far cheaper than a PCG solve.
    """
    _,e=covariance_view(fit)
    E=np.asarray(e,float);sd=np.sqrt(np.maximum(np.diag(E),1e-30))
    return float(np.linalg.eigvalsh(E/np.outer(sd,sd))[0])


def check_fit(fit,require_converged=False):
    """Reject unusable covariance fits; O(t^2).

    A fit that stopped at its iteration/evaluation cap is still a valid
    candidate: held-out prediction scores exactly the matrices it produced, so a
    poorly converged fit loses on its score instead of failing the trial.
    Rejecting such fits made every factor model fail on large trait panels
    (281 traits, sparse FA with a 120-iteration proximal cap), leaving only the
    trivially converged diagonal (rank 0) control. Convergence stays in
    fit_diagnostics; require_converged=True restores the strict rule.
    """
    result=fit.get('result')
    converged=(bool(fit.get('success',False)) and bool(fit.get('optimizer_converged',True))
               and (result is None or bool(result.success)))
    if require_converged and not converged:
        raise RuntimeError('covariance fit did not converge: '+str(fit.get('message')))
    g,e=covariance_view(fit)
    matrices=[np.asarray(m,float) for m in (g.values() if isinstance(g,Mapping) else [g])]+[np.asarray(e,float)]
    if not all(np.isfinite(m).all() for m in matrices):
        raise RuntimeError('covariance fit produced non-finite matrices: '+str(fit.get('message')))


def predict(train,kernels,fit,numerics):
    """Use the named full-kernel prediction view and reject inaccurate solves.

    train is already nuisance-adjusted. numerics controls tolerance/iterations,
    never kernel compression. Returns a raw prediction mapping.
    """
    from .multivariate import _mvPredict_raw
    g,e=covariance_view(fit)
    result=_mvPredict_raw(train,kernels,g,E=e,center=False,accuracy='exact',kernel_rank='full',
                          work_dtype='float64',progress=False,**numerics)
    if result.get('cg_info',0)!=0 or not bool(result.get('success',True)):
        raise RuntimeError('prediction did not converge')
    residual=result.get('cg_relative_residual')
    if residual is None or not np.isfinite(residual) or residual>max(5*numerics['solver_tol'],1e-10):
        raise RuntimeError(f'prediction residual failed stricter HPO gate: {residual}')
    return result


def covariance_diagnostics(fit,kernels,values):
    """Summarize the fitted inference covariance and its h2; O(c*t^3).

    h2 uses the fit's definition sum_c G_c[jj]/(sum_c G_c[jj]+E[jj]).
    Eigenvalue diagnostics are exact trait-matrix decompositions.
    """
    components=fit['G_components'];genetic=np.zeros(values.shape[1]);summary={}
    for name in kernels:
        matrix=np.asarray(components[name],float);eigen=np.linalg.eigvalsh(matrix)
        genetic+=np.diag(matrix)
        summary[name]={'trace':float(np.trace(matrix)),'min_eigenvalue':float(eigen[0]),
            'max_eigenvalue':float(eigen[-1]),'numerical_rank':int(np.sum(eigen>max(abs(eigen).max(),1)*1e-10)),
            'off_diagonal_norm':float(np.linalg.norm(matrix-np.diag(np.diag(matrix)))),
            'zero_fraction':float(np.mean(abs(matrix)<1e-12)),'diagonal':np.diag(matrix).tolist()}
    e_matrix=np.asarray(fit['E'],float);residual=np.diag(e_matrix);total=genetic+residual
    e_eigen=np.linalg.eigvalsh(e_matrix)
    summary['E']={'trace':float(np.trace(e_matrix)),'min_eigenvalue':float(e_eigen[0]),'max_eigenvalue':float(e_eigen[-1]),
        'off_diagonal_norm':float(np.linalg.norm(e_matrix-np.diag(residual))),'diagonal':residual.tolist()}
    h=np.divide(genetic,total,out=np.full_like(total,np.nan),where=total>0)
    observed=np.nanvar(values,axis=0)
    report={'fit':{'components':summary,'genetic_variance':genetic.tolist(),'residual_variance':residual.tolist(),
                   'total_variance':total.tolist(),'h2':[None if not np.isfinite(x) else float(x) for x in h],
                   'closure_error':(total-observed).tolist(),'boundary':(genetic<=0).tolist()}}
    prior=fit.get('h2_prior',{})
    if isinstance(prior,dict) and prior.get('active'):
        report['h2_prior']={k:prior.get(k) for k in ('strength','groups','group_mean','group_tau2','penalty_value')}
    return report


def connectivity_metrics(kernels,train_mask,test,values,predicted,fixed,scales,scenario,projects,units,traits,options,training_means):
    """Genotype-only connection strata using a fixed sample of training animals.

    Dense kernels use squared normalized kernel entries, not phenotype-derived
    similarity. Operators without cheap entries report unavailable. Sampling
    limits work to O(n_test * reference_animals); no full pairwise copy.
    """
    name=options.get('connectivity_kernel',next(iter(kernels)))
    if name not in kernels:raise ValueError('unknown connectivity_kernel')
    matrix=kernels[name]
    if not isinstance(matrix,np.ndarray):return [],{'status':'unavailable_for_operator_without_entry_access','kernel':name}
    count=options.get('connectivity_reference_animals',64)
    if isinstance(count,bool) or not isinstance(count,int) or count<1:raise ValueError('connectivity_reference_animals must be positive integer')
    eligible=np.flatnonzero(train_mask.any(1));rows=np.flatnonzero(test.any(1));rng=np.random.default_rng(9187+scenario['id'])
    reference=rng.choice(eligible,min(count,len(eligible)),replace=False)
    diagonal=np.diag(matrix);denominator=np.maximum(diagonal[rows,None]*diagonal[None,reference],1e-30)
    scores=np.mean(np.asarray(matrix[np.ix_(rows,reference)],float)**2/denominator,axis=1)
    cut=np.quantile(scores,[1/3,2/3]);bands=np.searchsorted(cut,scores,side='right');records=[]
    for band,label in enumerate(['low','middle','high']):
        mask=np.zeros_like(test);mask[rows[bands==band]]=test[rows[bands==band]]
        local=dict(scenario,role='connectivity_diagnostic')
        result,_=score_predictions(values,predicted,fixed,scales,mask,local,projects,units,traits,training_means=training_means)
        records += [dict(r,connectivity_band=label) for r in result]
    return records,{'kernel':name,'measure':'mean squared normalized kernel entry to fixed training reference sample',
                    'reference_animals':len(reference),'quantile_cuts':cut.tolist(),'status':'complete'}


def _representation(payload,fit,train,kernels,values,meta,test,fixed,scenario,numerics):
    """Evaluate source-frozen genetic embeddings with a separate target probe.

    Source fitting excludes every target trait. Representation SVD uses source
    training rows only; target calibration outcomes enter only the linear probe.
    Association mode reports correlations only, never a zero-shot prediction
    score or naive kinship-uncorrected association p-value.
    """
    prediction=predict(train,kernels,fit,numerics)
    features=np.asarray(prediction['breeding_value'],float);source_rows=train.notna().any(axis=1).to_numpy()
    mean=features[source_rows].mean(0);_,_,vt=np.linalg.svd(features[source_rows]-mean,full_matrices=False)
    rank=min(int(payload['cv_options'].get('embedding_rank',10)),len(vt));basis=vt[:rank].T
    embedding=(features-mean)@basis
    nuisance=np.load(Path(payload['cv_dir'])/'design.npy',allow_pickle=False)
    design=np.column_stack([nuisance,embedding])
    targets=scenario['target_columns'];cal=np.asarray(scenario['calibration_rows'],int)
    output=np.full_like(values,np.nan);records=[]
    ridge=float(payload['cv_options'].get('probe_ridge',1e-6))
    if not np.isfinite(ridge) or ridge<0:raise ValueError('probe_ridge must be finite/nonnegative')
    for j in targets:
        if scenario.get('association_only'):
            use=test[:,j];y=values[use,j]
            correlations=[None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1]) for x in embedding[use].T]
            records.append({'trait':meta['traits'][j],'source_frozen_phenotype_correlations':correlations,
                            'scenario':scenario['id'],'project':scenario.get('project'),'fold':scenario['fold'],'embedding_rank':rank,
                            'interpretation':'representation association; not rG, not zero-shot trait prediction; no p-value claimed'})
        else:
            use=cal[np.isfinite(values[cal,j])];x=design[use]
            penalty=np.diag([0.]*nuisance.shape[1]+[ridge]*rank)
            beta=np.linalg.solve(x.T@x+penalty,x.T@values[use,j]);output[:,j]=design@beta
    return output,records


def execute(payload,emit,cancel):
    """Run one fold, preflight, or final fit task; bounded by parent.

    payload contains trusted on-disk inputs, settings and optional scenario.
    Result mappings are atomically persisted by the process entry point.
    """
    from .multivariate import _genetic_varcov_PSD_raw
    if payload['action']=='paired_interval':
        from .hyperopt_cv import paired_cluster_interval
        with open(payload['candidate_file'],'rb') as stream:a=pickle.load(stream)
        with open(payload['baseline_file'],'rb') as stream:b=pickle.load(stream)
        emit({'phase':'paired_cluster_interval'})
        return paired_cluster_interval(a['groups'],b['groups'],payload['weights'],seed=payload['seed'],n_boot=payload['n_boot'])
    if payload['action']=='transfer_report':
        from .hyperopt_cv import transfer_report
        with open(payload['candidate_file'],'rb') as stream:a=pickle.load(stream)
        manifest=json.loads(Path(payload['manifest']).read_text())
        emit({'phase':'paired_transfer_report'})
        return transfer_report(a,manifest,seed=payload['seed'],n_boot=payload['n_boot'])
    if payload['action']=='persist_final':
        from .multivariate import _covariance_fit_dataset
        from .results import save_result
        import zarr
        with open(payload['point_path'],'rb') as stream:point=pickle.load(stream)
        dataset=_covariance_fit_dataset(point);dataset.attrs['domino_write_state']='writing'
        target=Path(payload['model_path']);temporary=target.with_name(target.stem+'.incomplete.zarr')
        emit({'phase':'model_persistence'})
        reopened=save_result(dataset,temporary,artifact_type='hyperopt_covariance_fit')
        reopened.close()
        state=zarr.open_group(temporary,mode='r',use_consolidated=False).attrs.get('domino_write_state')
        if state!='complete':raise RuntimeError('model writer did not finalize the Zarr artifact')
        temporary.replace(target)
        return {'model_path':str(target),'write_state':'complete'}
    meta,values,kernels=load_data(payload['data_dir']);index=pd.Index(meta['samples']);all_traits=pd.Index(meta['traits'])
    design=np.load(Path(payload['cv_dir'])/'design.npy',allow_pickle=False)
    projects=np.load(Path(payload['cv_dir'])/'projects.npy',allow_pickle=False)
    units=np.load(Path(payload['cv_dir'])/'units.npy',allow_pickle=False)
    candidate=dict(payload['candidate'])
    final=payload['action']=='final'
    if final: train_mask=np.isfinite(values);test=np.zeros(values.shape,bool);columns=np.arange(values.shape[1]);scenario=None
    else: scenario=payload['scenario'];train_mask,test,columns=load_masks(scenario)
    raw=np.asarray(values[:,columns],float)
    training_means=np.asarray([np.mean(raw[train_mask[:,column],j]) for j,column in enumerate(columns)])
    residual,fixed,coeff,scales,design_ranks=preprocess(raw,train_mask[:,columns],design)
    train=pd.DataFrame(residual,index=index,columns=all_traits[columns])
    kwargs=dict(payload['fit_kwargs']);kwargs.update(candidate)
    kwargs.update(optimize_hyperparameters_cv=0,rank_selection='none',warm_start=None,
                  progress=False,structural_refine=False,structural_plots=False)
    # 'project' prior domains are the declared trait projects of this fold's traits.
    groups=kwargs.get('h2_prior_groups')
    if isinstance(groups,str) and groups=='project':
        kwargs['h2_prior_groups']=pd.Series(np.asarray(projects)[columns].astype(str),index=all_traits[columns])
    elif isinstance(groups,str) and groups=='global':
        kwargs['h2_prior_groups']=None
    cache_key=payload['fingerprint']+':'+('final' if final else scenario['mask_sha256'])
    fit_started=time.monotonic()
    if payload['action']=='outer_predict':
        with open(payload['point_path'],'rb') as stream:fit=pickle.load(stream)
        if fit.get('hpo_covariance_view')!='retention_adjusted_prediction':raise ValueError('outer model has no deployable selected view')
        trained=np.asarray(fit['hpo_fixed_coefficients'],float)
        if trained.shape!=coeff.shape or not np.allclose(trained,coeff,rtol=1e-10,atol=1e-12):raise ValueError('outer nuisance mapping differs from the inner full-training refit')
    else:
        emit({'phase':'covariance_fit','scenario':None if scenario is None else scenario['id']})
        def checkpoint(fit):
            """Commit point estimates before uncertainty can fail or time out."""
            import xarray as xr
            point=dict(fit)
            point['hpo_fixed_coefficients']=xr.DataArray(coeff,dims=('fixed_effect','trait'),
                coords={'fixed_effect':meta['covariate_names'],'trait':all_traits.astype(str)})
            point['hpo_preprocessing']={'method':'training-only least-squares nuisance adjustment; no outcome imputation',
                'fixed_effect_names':meta['covariate_names'],'design_rank':design_ranks,'prediction_helper':'vigor.predict_hyperopt'}
            atomic_pickle(payload['point_path'],point)
            emit({'phase':'point_fit_committed','point_fit_committed':True})
        from .calibration_probes import probe_cache
        from .fit_cache import fit_cache
        # h2-prior pilot geometry depends only on the fold's masks, so it is
        # shared by every candidate fitted on the same fold; covariance fits are
        # shared by candidates that differ only in post-fit (prediction) settings.
        fit_identity=cache_key+':'+repr(sorted(payload['optimizer_limits'].items()))
        # Deterministic warm start: every candidate starts from the incumbent's
        # (anchor 0) fit of this same fold, re-factored to the candidate rank.
        # Anchor 0 always completes its folds first, so the start never depends
        # on trial order; non-finite fits are rejected by check_fit (iteration-capped
        # fits are kept and scored unless require_converged is set).
        warm_path=None
        if scenario is not None and payload.get('cache_dir') and payload['action'] in {'fold','preflight'}:
            warm_path=Path(payload['cache_dir'])/'warm_starts'/(scenario['mask_sha256']+'.pkl')
            if payload.get('use_warm') and warm_path.exists():
                with open(warm_path,'rb') as stream:kwargs['warm_start']=pickle.load(stream)
                emit({'phase':'warm_start','source':'incumbent fold fit'})

        def fit_with(extra):
            """One covariance fit; post-fit settings in extra reuse the cached fit."""
            with worker_context(cache_dir=payload.get('cache_dir'),cache_key=cache_key,
                                limits=payload['optimizer_limits'],emit=emit,cancel=cancel,
                                point_sink=checkpoint if final else None,cache_max_bytes=payload.get('cache_max_bytes',2*1024**3)), \
                 probe_cache(payload.get('cache_dir'),cache_key,payload.get('cache_max_bytes',2*1024**3)), \
                 fit_cache(payload.get('cache_dir'),fit_identity,payload.get('cache_max_bytes',2*1024**3)):
                result=_genetic_varcov_PSD_raw(kernels,train,**{**kwargs,**extra})
            check_fit(result,payload.get('require_converged',False))
            return result
        grid=payload.get('prediction_grid') or [{}]
        fit=fit_with(grid[0])
        if payload.get('store_warm') and warm_path is not None:
            atomic_pickle(warm_path,{**{name:spec for name,spec in fit['component_specs'].items()},'E':fit['E_spec']})
    fit_seconds=time.monotonic()-fit_started
    if final:
        import xarray as xr
        fit=dict(fit)
        fit['hpo_fixed_coefficients']=xr.DataArray(coeff,dims=('fixed_effect','trait'),
            coords={'fixed_effect':meta['covariate_names'],'trait':all_traits.astype(str)})
        fit['hpo_preprocessing']={'method':'training-only least-squares nuisance adjustment; no outcome imputation',
                                  'fixed_effect_names':meta['covariate_names'],'design_rank':design_ranks,
                                  'prediction_helper':'vigor.predict_hyperopt'}
        return {'fit':fit,'fit_diagnostics':fit_diagnostics(fit),
                'covariance_diagnostics':covariance_diagnostics(fit,kernels,residual)}
    if payload['action']=='sensitivity':
        return {'fit_diagnostics':fit_diagnostics(fit),'covariance_diagnostics':covariance_diagnostics(fit,kernels,residual),
                'traits':train.columns.astype(str).tolist()}
    def restrict(fit):
        """Apply the scenario's cross-project G restriction to a prediction view."""
        if not scenario.get('disable_cross_project_G'):return fit
        g,e=covariance_view(fit);mask=projects[columns,None]==projects[None,columns]
        g={name:np.asarray(matrix)*mask for name,matrix in g.items()}
        fit=dict(fit);fit['G_components_prediction']=g;fit['E_prediction']=e
        return fit
    fit=restrict(fit)
    emit({'phase':'prediction','scenario':scenario['id']})
    numerics=payload['prediction'];started=time.monotonic()
    if scenario['role']=='representation':
        predictions,representation=_representation(payload,fit,train,kernels,np.asarray(values),meta,test,fixed,scenario,numerics)
        if scenario.get('association_only'):return {'rows':[],'groups':[],'representation':representation,'fit_diagnostics':fit_diagnostics(fit)}
        # Probe scales and baseline are estimated only from target calibration.
        fixed=np.zeros_like(values);scales=np.ones(values.shape[1])
        for j in scenario['target_columns']:
            rows=np.asarray(scenario['calibration_rows']);y=values[rows,j];y=y[np.isfinite(y)]
            fixed[:,j]=np.mean(y);scales[j]=np.std(y)
            if scales[j]<=1e-12:raise ValueError('constant probe calibration outcome')
        metrics,groups=score_predictions(np.asarray(values),predictions,fixed,scales,test,scenario,projects,units,all_traits)
        return {'rows':metrics,'groups':groups,'fit_diagnostics':fit_diagnostics(fit)}
    partial=scenario['design'] in {'partial_cell','trait_block'}
    def score(prediction):
        """Genetic prediction (intercept + breeding value) is the primary objective for every
        design. Where the animal's other traits are observed, the total phenotype prediction
        (genetic plus conditional residual) is also scored, as secondary 'objective_total' rows."""
        genetic=np.asarray(prediction['genetic_mean'],float)+fixed
        rows,genetic_groups=score_predictions(raw,genetic,fixed,scales,test[:,columns],scenario,projects[columns],units,
                                              all_traits[columns],training_means=training_means)
        if partial:
            total=np.asarray(prediction['phenotype_prediction'],float)+fixed
            extra,_=score_predictions(raw,total,fixed,scales,test[:,columns],scenario,projects[columns],units,
                                      all_traits[columns],training_means=training_means)
            rows=rows+[dict(r,role='objective_total') for r in extra if r['role']=='objective']
        return rows,genetic_groups,genetic
    prediction=predict(train,kernels,fit,numerics)
    metrics,groups,predicted=score(prediction)
    output={'rows':metrics,'groups':groups,'fit_diagnostics':fit_diagnostics(fit),
            'prediction_diagnostics':{key:prediction.get(key) for key in ('cg_info','cg_iterations','cg_relative_residual','preconditioner','resolved_work_dtype')},
            'prediction_seconds':time.monotonic()-started,'covariance_view':'retention_adjusted_prediction',
            'design_rank':design_ranks,'stage_seconds':{'fit':fit_seconds,'prediction':time.monotonic()-started}}
    if len(grid)>1:
        # Inner loop over post-fit prediction settings on the same cached fit:
        # each variant is scored on this fold; the controller picks one variant
        # per candidate from the aggregate over its screening folds.
        variants=[{'variant':0,'settings':grid[0],'rows':metrics,'groups':groups}]
        floor=payload.get('condition_floor')
        for index,extra in enumerate(grid[1:],1):
            emit({'phase':'prediction_variant','scenario':scenario['id'],'variant':index})
            variant_fit=restrict(fit_with(extra))
            if floor:
                condition=residual_condition(variant_fit)
                if not condition>=float(floor):
                    # Analytical gate: skip the PCG solve for a near-singular residual matrix.
                    variants.append({'variant':index,'settings':extra,'rows':[],'groups':[],'skipped':True,
                                     'reason':f'residual correlation lambda_min {condition:.3g} < {float(floor):g}'})
                    continue
            variant_prediction=predict(train,kernels,variant_fit,numerics)
            rows,variant_groups,_=score(variant_prediction)
            variants.append({'variant':index,'settings':extra,'rows':rows,'groups':variant_groups})
        output['variants']=variants
        output['stage_seconds']['prediction']=time.monotonic()-started
    if 'connectivity_stratified' in payload['cv_options'].get('designs',[]):
        extra,connection=connectivity_metrics(kernels,train_mask[:,columns],test[:,columns],raw,predicted,fixed,scales,scenario,
            projects[columns],units,all_traits[columns],payload['cv_options'],training_means)
        output['rows']+=extra;output['connectivity']=connection
    if payload['action']=='preflight':
        emit({'phase':'prediction_verify_preflight'})
        verified=predict(train,kernels,fit,payload['prediction_verify'])
        tight_rows,_,reference=score(verified)
        discrepancies=np.abs(predicted-reference)/scales[None,:]
        active=test[:,columns];nmse_a=np.mean([r['nmse'] for r in metrics]);nmse_b=np.mean([r['nmse'] for r in tight_rows])
        output['preflight']={'max_prediction_delta_sd':float(discrepancies[active].max()),
            'mean_prediction_delta_sd':float(discrepancies[active].mean()),'mean_trait_nmse_delta':float(nmse_a-nmse_b),
            'screen_residual':prediction['cg_relative_residual'],'verify_residual':verified['cg_relative_residual'],
            'screen_iterations':prediction['cg_iterations'],'verify_iterations':verified['cg_iterations']}
    return output
