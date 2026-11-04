"""Compound Bayesian covariance search with durable supervised execution.

Inner scores select settings; untouched outer evaluation is a separate API.
No fitted covariance SE includes hyperparameter-selection uncertainty.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import functools
import hashlib
import inspect
import json
import math
import numbers
import os
import pickle
import sys
import time
import uuid
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .hyperopt_cv import (positive_int, aligned_labels, prepare_cv, aggregate,
                          paired_cluster_interval, load_masks)
from .hyperopt_runtime import (atomic_json, atomic_pickle, supervised, WorkerFailure,
                               acquire_study_lock, release_study_lock)
from .hyperopt_space import STATISTICAL, PREDICTION_ONLY, DEFAULT_PREDICTION_GRID, prepare_space, suggest, encode

OPTIONS = {'sampler','seed','n_trials','search_timeout_s','trial_timeout_s','phase_timeout_s',
           'final_refit_timeout_s','uncertainty_timeout_s','n_jobs','cv',
           'search_space','prediction_screen','prediction_verify','fit_screen','fit_verify',
           'early_stopping','on_interrupt','checkpoint_dir','resume','progress','heartbeat_s',
           'memory_limit_bytes','cancel_grace_s','finalists','verification_reserve_fraction',
           'selection','guardrails','checkpoint_only','final_refit_only','stop_file',
           'study_name','max_consecutive_failures','final_refit_retries','preflight',
           'optimizer_limits',
           'cache_max_bytes','diagnostic_timeout_s'}
OPTIONS.add('sampler_startup_trials')
OPTIONS |= {'search_preset','prediction_grid','pruning','sampler_multivariate','total_prediction_weight',
            'prediction_condition_floor','prediction_grid_drop','require_converged',
            'warm_start_from','warm_start_top','warm_start_incumbent'}


def _load_optuna():
    """Load the optional Optuna 4/5 API or raise an actionable version error."""
    try:import optuna
    except ImportError as exc:raise ImportError("hyper_opt requires python -m pip install 'VIGOR[hpo]' (optuna>=4,<6)") from exc
    if str(optuna.__version__).split('.')[0] not in {'4','5'}:
        raise RuntimeError('supported Optuna major versions are 4 and 5 (optuna>=4,<6)')
    return optuna


def _verification_shortlist(records, finalists):
    """Best screened candidates admitted to bounded full verification."""
    complete=sorted((r for r in records if 'screen' in r), key=lambda r:r['screen']['score'])
    return complete[:finalists]


def serializable(value):
    """Normalize small metadata to strict JSON, replacing nonfinite numbers.

    Arrays/Series are allowed only for input vectors/diagnostics, not covariance
    matrices in trial attrs. O(metadata size); scientific matrices stay labeled.
    """
    if isinstance(value,Mapping):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,pd.Index)):return [serializable(v) for v in value]
    if isinstance(value,pd.DataFrame):return {'index':serializable(value.index),'columns':serializable(value.columns),'values':serializable(value.to_numpy())}
    if isinstance(value,pd.Series):return {'index':serializable(value.index),'values':serializable(value.to_numpy())}
    if isinstance(value,np.ndarray):return serializable(value.tolist())
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    if isinstance(value,type):return value.__module__+'.'+value.__name__
    if isinstance(value,np.dtype):return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'unsupported checkpoint metadata type {type(value).__name__}')


def digest(value):
    """Hash canonical metadata bytes; O(metadata bytes), returns SHA256 hex."""
    return hashlib.sha256(json.dumps(serializable(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def hash_array(value):
    """Hash array contents in bounded row blocks; O(bytes), <=16 MiB copy.

    NaNs retain their binary representation; this hash is a resume identity,
    not a promise of equivalence across differently encoded input datasets.
    """
    array=np.asarray(value);h=hashlib.sha256(str((array.shape,str(array.dtype))).encode())
    if array.ndim==0:h.update(array.tobytes())
    else:
        step=max(1,16*1024**2//max(array[0:1].nbytes,1))
        for start in range(0,len(array),step):h.update(np.ascontiguousarray(array[start:start+step]).tobytes())
    return h.hexdigest()


def normalise_options(options,params):
    """Validate public controller options and finite budgets before any search.

    Returns an independent mapping with defaults. Unknown options and parallel
    trial requests fail explicitly. Numerical controls are not Optuna targets.
    """
    opts=dict(options or {});unknown=set(opts)-OPTIONS
    if unknown:raise TypeError(f'unknown hyper_opt_kwargs: {sorted(unknown)}')
    defaults={'sampler':'tpe','seed':params.get('random_state') or 0,'n_trials':30,
        'search_timeout_s':3600.,'trial_timeout_s':300.,'phase_timeout_s':300.,
        'final_refit_timeout_s':900.,'uncertainty_timeout_s':600.,
        'n_jobs':1,'cv':{},'search_space':{},'prediction_screen':{'solver_tol':1e-4,'solver_maxiter':500,'preconditioner':'diag'},
        'prediction_verify':{'solver_tol':1e-6,'solver_maxiter':1200,'preconditioner':'diag'},
        'fit_screen':{'gtol':1e-5,'ftol':1e-7},'fit_verify':{},'early_stopping':{'min_trials':12,'patience':10,'min_verified':3,'min_relative_gain':.005},
        'on_interrupt':'checkpoint_and_refit_best','resume':False,'progress':params.get('progress',True),
        'heartbeat_s':5.,'memory_limit_bytes':None,'cancel_grace_s':2.,'finalists':3,
        'verification_reserve_fraction':.25,'selection':{'min_relative_gain':.005,'require_cluster_ci':True,'n_boot':200},
        'guardrails':{},'checkpoint_only':False,'final_refit_only':False,'stop_file':None,
        'study_name':'vigor_compound_hpo','max_consecutive_failures':5,'final_refit_retries':0,
        'preflight':{'max_prediction_delta_sd':.001,'max_nmse_delta':.001},
        'optimizer_limits':{'maxfun':10000,'maxls':30},'cache_max_bytes':2*1024**3,
        'diagnostic_timeout_s':300.,'sampler_startup_trials':8,
        'search_preset':'compact','prediction_grid':None,'sampler_multivariate':True,'total_prediction_weight':.1,
        'prediction_condition_floor':1e-3,'prediction_grid_drop':True,'require_converged':False,
        'pruning':{'enabled':True,'startup_trials':8,'warmup_folds':0},
        'warm_start_from':None,'warm_start_top':5,'warm_start_incumbent':True}
    for key,value in opts.items():
        defaults[key]={**defaults[key],**value} if key in {'early_stopping','selection','preflight','optimizer_limits','prediction_screen','prediction_verify','fit_screen','fit_verify','pruning'} else value
    cfg=defaults
    for name in ('resume','progress','checkpoint_only','final_refit_only'):
        if not isinstance(cfg[name],(bool,np.bool_)):raise TypeError(f'{name} must be bool')
        cfg[name]=bool(cfg[name])
    if not isinstance(cfg['selection']['require_cluster_ci'],(bool,np.bool_)):raise TypeError('require_cluster_ci must be bool')
    positive_int(cfg['n_jobs'],'n_jobs')
    if cfg['sampler']!='tpe':raise ValueError('supervised HPO supports sampler=tpe')
    # n_jobs runs the folds of one candidate in parallel workers (explicit BLAS
    # thread counts are divided between them); trials stay sequential.
    for name in ('sampler_multivariate','warm_start_incumbent','require_converged'):
        if not isinstance(cfg[name],(bool,np.bool_)):raise TypeError(f'{name} must be bool')
        cfg[name]=bool(cfg[name])
    if cfg['search_preset'] not in {'compact','extended'}:raise ValueError("search_preset must be 'compact' or 'extended'")
    if set(cfg['pruning'])-{'enabled','startup_trials','warmup_folds'}:raise ValueError('pruning accepts enabled, startup_trials, warmup_folds')
    if not isinstance(cfg['pruning']['enabled'],(bool,np.bool_)):raise TypeError('pruning.enabled must be bool')
    positive_int(cfg['pruning']['startup_trials'],'pruning.startup_trials',0)
    positive_int(cfg['pruning']['warmup_folds'],'pruning.warmup_folds',0)
    if cfg['prediction_grid'] is not None:
        grid=cfg['prediction_grid']
        if not isinstance(grid,(list,tuple)) or not grid or any(not isinstance(v,Mapping) or set(v)-PREDICTION_ONLY for v in grid):
            raise ValueError(f'prediction_grid must be a nonempty list of mappings over {sorted(PREDICTION_ONLY)}')
        cfg['prediction_grid']=[dict(v) for v in grid]
    positive_int(cfg['warm_start_top'],'warm_start_top')
    if not isinstance(cfg['prediction_grid_drop'],(bool,np.bool_)):raise TypeError('prediction_grid_drop must be bool')
    cfg['prediction_grid_drop']=bool(cfg['prediction_grid_drop'])
    floor=cfg['prediction_condition_floor']
    if floor is not None and (isinstance(floor,(bool,np.bool_)) or not isinstance(floor,numbers.Real) or not 0<=floor<1):
        raise ValueError('prediction_condition_floor must be None or in [0, 1)')
    weight=cfg['total_prediction_weight']
    if isinstance(weight,(bool,np.bool_)) or not isinstance(weight,numbers.Real) or not np.isfinite(weight) or weight<0:
        raise ValueError('total_prediction_weight must be a finite nonnegative number')
    cfg['total_prediction_weight']=float(weight)
    source=cfg['warm_start_from']
    if source is not None and not isinstance(source,(str,os.PathLike,list,tuple)):
        raise TypeError('warm_start_from is a previous checkpoint_dir or a list of configuration mappings')
    positive_int(cfg['sampler_startup_trials'],'sampler_startup_trials',0)
    if cfg['on_interrupt']!='checkpoint_and_refit_best':raise ValueError('on_interrupt must be checkpoint_and_refit_best')
    for name in ['n_trials','finalists','max_consecutive_failures','cache_max_bytes']:cfg[name]=positive_int(cfg[name],name)
    if cfg['n_trials']<3:raise ValueError('n_trials must include at least three explicitly counted anchors')
    cfg['seed']=positive_int(cfg['seed'],'seed',0)
    cfg['final_refit_retries']=positive_int(cfg['final_refit_retries'],'final_refit_retries',0)
    if cfg['final_refit_retries']>1:raise ValueError('at most one numerical final-refit recovery is supported')
    for name in ['search_timeout_s','trial_timeout_s','phase_timeout_s','final_refit_timeout_s','uncertainty_timeout_s',
                 'heartbeat_s','cancel_grace_s','diagnostic_timeout_s']:
        if isinstance(cfg[name],(bool,np.bool_)) or not np.isfinite(cfg[name]) or cfg[name]<=0:raise ValueError(f'{name} must be finite positive seconds')
    if cfg['memory_limit_bytes'] is not None:cfg['memory_limit_bytes']=positive_int(cfg['memory_limit_bytes'],'memory_limit_bytes')
    if not 0<cfg['verification_reserve_fraction']<1:raise ValueError('verification_reserve_fraction must be in (0,1)')
    for stage in ['prediction_screen','prediction_verify']:
        if set(cfg[stage])-{'solver_tol','solver_maxiter','preconditioner','blas_threads'}:raise ValueError('prediction fidelity accepts solver_tol,solver_maxiter,preconditioner,blas_threads')
        tol=cfg[stage]['solver_tol']
        if isinstance(tol,(bool,np.bool_)) or not np.isfinite(tol) or not 0<tol<1:raise ValueError('solver_tol must lie in (0,1), not bool')
        cfg[stage]['solver_maxiter']=positive_int(cfg[stage]['solver_maxiter'],'solver_maxiter')
        cfg[stage].setdefault('blas_threads',params.get('blas_threads','auto'))
    if cfg['prediction_verify']['solver_tol']>cfg['prediction_screen']['solver_tol']:raise ValueError('verification must be at least as accurate as screening')
    for stage in ['fit_screen','fit_verify']:
        if set(cfg[stage])-{'maxiter','gtol','ftol','factor_sparse_maxiter','factor_sparse_tol'}:raise ValueError('fit fidelity only accepts iteration/convergence budgets')
        for name,value in cfg[stage].items():
            if name in {'maxiter','factor_sparse_maxiter'}:positive_int(value,name)
            elif not np.isfinite(value) or value<=0:raise ValueError('convergence tolerances must be finite positive')
    if set(cfg['optimizer_limits'])!={'maxfun','maxls'}:raise ValueError('optimizer_limits must contain maxfun and maxls')
    for k in cfg['optimizer_limits']:positive_int(cfg['optimizer_limits'][k],k)
    if set(cfg['early_stopping'])-{'min_trials','patience','min_verified','min_relative_gain'}:raise ValueError('unknown early_stopping controls')
    for k in ('min_trials','patience','min_verified'):positive_int(cfg['early_stopping'][k],k)
    if set(cfg['selection'])-{'min_relative_gain','require_cluster_ci','n_boot'}:raise ValueError('unknown selection controls')
    positive_int(cfg['selection']['n_boot'],'selection.n_boot',20)
    if set(cfg['preflight'])!={'max_prediction_delta_sd','max_nmse_delta'}:raise ValueError('unknown preflight controls')
    for mapping in [cfg['preflight'],cfg['guardrails'],{'selection_gain':cfg['selection']['min_relative_gain'],'early_gain':cfg['early_stopping']['min_relative_gain']}]:
        if any(not np.isfinite(v) or v<0 for v in mapping.values()):raise ValueError('gains/guardrails/preflight thresholds must be finite nonnegative')
    return cfg


class Reporter:
    """Durable JSONL events plus fastprogress/structured headless heartbeats."""
    def __init__(self,path,budget,enabled):
        """Open an owned event stream and optional parent display; O(1)."""
        from .progress import WorkflowProgress
        self.display=WorkflowProgress(trials=budget,enabled=enabled)
        self.path=Path(path);self.bar=self.display.progress
        self.enabled=enabled;self.current={};self.previous={};self.last_text=0.;self.best=None;self.budget=budget
        import threading
        self.lock=threading.Lock()
    def emit(self,event):
        """Persist real events; print throttled live state, never maxiter ETA."""
        with self.lock:
            event=serializable(dict(event,time_unix=time.time(),trial_budget=self.budget));self.current.update(event)
            with self.path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
            self.display.observe(dict(event,best_score=self.best))
    def candidate(self,number,params,fidelity):
        """Report numeric deltas only on matching coordinates; categorical changes."""
        changed={k:{'before':self.previous.get(k),'after':v} for k,v in params.items() if self.previous.get(k)!=v}
        for k,item in changed.items():
            a,b=item['before'],item['after']
            if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):item['delta']=b-a
        self.previous=dict(params);self.current={}
        self.emit({'kind':'candidate_start','phase':'candidate_start','trial':number,'fidelity':fidelity,
                   'workflow_label':'Search / '+fidelity,'parameters':params,'parameter_changes':changed})
    def close(self):
        """Close only UI resources; every event was already flushed."""
        self.display.close(self.current.get('workflow_status','stopped'))


def _prepare_input(grm,traits,cv,root,params,cfg):
    """Align/hash inputs and place read-only kernel arrays once per study.

    Returns fingerprint and dataset metadata. Dense hashing/placement O(c*n^2)
    time, bounded hashing copies; named operator inputs remain separate and
    require trusted spawn-picklable representations. Nothing sums kernels.
    """
    from .multivariate import _kernel_input,_as_pandas_input
    grm=_kernel_input(grm);traits=_as_pandas_input(traits)
    Y=traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits)
    if not len(Y) or not len(Y.columns) or Y.index.has_duplicates or Y.columns.has_duplicates:raise ValueError('nonempty unique sample/trait axes are required')
    Y.index=Y.index.astype(str);Y.columns=Y.columns.astype(str)
    if Y.index.has_duplicates or Y.columns.has_duplicates:raise ValueError('stringified sample/trait axes must remain unique')
    if np.isinf(Y.to_numpy(float)).any():raise ValueError('traits may contain NaN, not infinity')
    cv=dict(cv)
    for name in ('sample_metadata','covariates'):
        if isinstance(cv.get(name),pd.DataFrame):
            cv[name]=cv[name].copy();cv[name].index=cv[name].index.astype(str)
    for name in ('trait_projects','trait_blocks'):
        if isinstance(cv.get(name),pd.Series):cv[name]=cv[name].rename(index=str)
        elif isinstance(cv.get(name),dict):cv[name]={str(k):v for k,v in cv[name].items()}
    protected=cv.pop('protected_mask',None)
    if protected is not None:
        if isinstance(protected,pd.DataFrame):
            protected=protected.copy();protected.index=protected.index.astype(str);protected.columns=protected.columns.astype(str)
            protected=protected.reindex(index=Y.index,columns=Y.columns)
        a=np.asarray(protected)
        if a.shape!=Y.shape or a.dtype.kind!='b':raise ValueError('protected_mask must be a fully aligned boolean phenotype mask')
        Y=Y.mask(a)
    values=Y.to_numpy(dtype=float)
    if np.any(np.isfinite(values).sum(0)<3):raise ValueError('each authorized trait needs >=3 observations')
    if isinstance(grm,Mapping) and not any(k in grm for k in ('type','matrix','U','values')):
        kernels={str(k):v for k,v in grm.items()}
    else:kernels={'genetic':grm}
    if len(kernels)==0:raise ValueError('at least one kernel is required')
    identity={'samples':Y.index.tolist(),'traits':Y.columns.tolist(),'phenotypes':hash_array(values),'kernels':{},
              'cv':cv,'fit':params,'search_space':cfg['search_space'],
              'seed':cfg['seed'],'prediction_screen':cfg['prediction_screen'],'prediction_verify':cfg['prediction_verify'],
              'fit_screen':cfg['fit_screen'],'fit_verify':cfg['fit_verify'],'preflight':cfg['preflight'],
              'selection':cfg['selection'],'guardrails':cfg['guardrails']}
    identity['sampler_startup_trials']=cfg['sampler_startup_trials']
    for name in ('search_preset','prediction_grid','sampler_multivariate','pruning','warm_start_incumbent','total_prediction_weight',
                 'prediction_condition_floor','prediction_grid_drop','require_converged'):identity[name]=cfg[name]
    identity['optuna_version']=cfg.get('optuna_version')
    identity['protected_mask_sha256']=None if protected is None else hash_array(np.asarray(protected))
    prepared={}
    for name,kernel in kernels.items():
        if isinstance(kernel,pd.DataFrame):
            kernel=kernel.copy(deep=False);kernel.index=kernel.index.astype(str);kernel.columns=kernel.columns.astype(str)
            if kernel.index.has_duplicates or kernel.columns.has_duplicates:raise ValueError('kernel labels must be unique')
            kernel=kernel.loc[Y.index,Y.index].to_numpy()
        if isinstance(kernel,np.ndarray):
            if kernel.shape!=(len(Y),len(Y)) or kernel.dtype.kind not in 'fi':raise ValueError(f'{name}: dense kernel must be square/animal aligned')
            for start in range(0,len(Y),256):
                block=kernel[start:start+256]
                if not np.isfinite(block).all() or not np.allclose(block,kernel[:,start:start+256].T,rtol=1e-5,atol=1e-7):raise ValueError(f'{name}: finite symmetric kernel required')
            identity['kernels'][name]=hash_array(kernel);prepared[name]=('array',kernel)
        else:
            from .multivariate import as_kernel_operator
            kernel=as_kernel_operator(kernel,dtype=params.get('block_dtype',np.float32))
            if kernel.shape!=(len(Y),len(Y)):raise ValueError('kernel operator must be sample aligned')
            encoded=pickle.dumps(kernel,protocol=5)
            identity['kernels'][name]=hashlib.sha256(encoded).hexdigest();prepared[name]=('operator',encoded)
    source={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.rglob('*.py')}
    identity['source']=source
    fingerprint=digest(identity);state_path=root/'identity.json'
    if state_path.exists():
        prior=json.loads(state_path.read_text())
        if prior['fingerprint']!=fingerprint:raise ValueError('checkpoint identity mismatch: source/data/masks/settings changed; use a new checkpoint_dir')
        if not cfg['resume']:raise FileExistsError('checkpoint exists; set resume=True or choose a new directory')
        return Y,cv,fingerprint,prior['data']
    root.mkdir(parents=True,exist_ok=True);data_dir=root/'input';setup_path=root/'setup.json'
    if data_dir.exists():
        setup=json.loads(setup_path.read_text()) if setup_path.exists() else {}
        if not cfg['resume'] or setup.get('fingerprint')!=fingerprint or setup.get('write_state')!='writing':
            raise FileExistsError('uncommitted input directory is not owned by this resumable setup; use another checkpoint directory')
        import shutil
        shutil.rmtree(data_dir)
        if (root/'cv').exists():shutil.rmtree(root/'cv')
    atomic_json(setup_path,{'write_state':'writing','fingerprint':fingerprint});data_dir.mkdir()
    np.save(data_dir/'traits.npy',values)
    metadata={'samples':Y.index.tolist(),'traits':Y.columns.tolist(),'kernels':{}}
    for i,(name,(kind,value)) in enumerate(prepared.items()):
        file=f'kernel_{i}.npy' if kind=='array' else f'kernel_{i}.pkl'
        if kind=='array':np.save(data_dir/file,value)
        else:(data_dir/file).write_bytes(value)
        metadata['kernels'][name]={'kind':kind,'file':file}
    manifest=prepare_cv(Y,cv,root/'cv',cfg['seed']+9187,allow_outer_only=cfg.get('allow_outer_only',False))
    metadata['covariate_names']=manifest['covariate_names'];atomic_json(data_dir/'data.json',metadata)
    atomic_json(state_path,{'fingerprint':fingerprint,'source_sha256':source,'data':metadata,
                           'inputs':serializable({k:v for k,v in identity.items() if k not in {'fit','cv','source'}})})
    atomic_json(setup_path,{'write_state':'complete','fingerprint':fingerprint})
    return Y,cv,fingerprint,metadata


def _screen_schedule(manifest):
    """Choose one prespecified representative fold per regime/target project.

    Prefer the largest transfer budget; all finalists use the full schedule.
    Selection uses metadata only, never candidate scores.
    """
    selected={}
    for scenario in manifest['scenarios']:
        if scenario['role']!='objective':continue
        key=(scenario['design'],scenario.get('project'))
        if key not in selected or (scenario.get('budget') or 0)>(selected[key].get('budget') or 0):selected[key]=scenario
    return sorted(selected.values(),key=lambda s:s['id'])


def _seed_configs(source,top):
    """Configurations to enqueue first: a list of mappings, or the best screened
    candidates (and the selection) of a previous checkpoint directory."""
    if source is None:return []
    if isinstance(source,(list,tuple)):return [dict(c) for c in source]
    root=Path(source)
    if not (root/'trials.json').exists():raise FileNotFoundError(f'warm_start_from has no trials.json: {root}')
    records=json.loads((root/'trials.json').read_text())
    ranked=sorted((r for r in records if r.get('screen') and r.get('config')),key=lambda r:r['screen']['score'])
    configs=[r['config'] for r in ranked[:int(top)]]
    report=root/'report.json'
    if report.exists():
        selected=json.loads(report.read_text()).get('selected')
        if selected and selected not in configs:configs.insert(0,selected)
    return configs


def _objective(rows,weights,total_weight):
    """Genetic prediction first, total prediction second.

    The compound genetic-prediction score (role 'objective': intercept plus
    breeding value against held-out phenotypes, every design) is primary. Where
    designs also scored the total phenotype prediction ('objective_total':
    genetic plus conditional residual from the animal's other traits), its
    compound score is added with weight total_weight. by_regime/by_project and
    the paired cluster intervals stay on the genetic objective.
    """
    summary=aggregate(rows,weights)
    summary['genetic_score']=summary['score']
    total=[dict(r,role='objective') for r in rows if r['role']=='objective_total']
    if total_weight>0 and total:
        regimes={r['regime'] for r in total if r['comparison']=='all_donors'}
        mass=sum(weights[k] for k in regimes if k in weights)
        if mass>0:
            secondary=aggregate(total,{k:weights[k]/mass for k in regimes if k in weights})
            summary['total_score']=secondary['score'];summary['total_by_regime']=secondary['by_regime']
            summary['score']=summary['genetic_score']+total_weight*secondary['score']
    return summary


def _partial_score(outputs):
    """Mean objective NMSE over the folds done so far (best prediction variant).

    Every candidate runs the same screening folds in the same order, so values
    at one step are comparable for pruning; the final score still uses the
    full compound aggregation.
    """
    best=None
    count=max((len(x['variants']) if 'variants' in x else 1) for x in outputs)
    for index in range(count):
        rows=[r for x in outputs for r in (x['variants'][index]['rows'] if 'variants' in x else x.get('rows',[]))
              if r['role']=='objective' and r['comparison']=='all_donors']
        if rows:
            value=float(np.mean([r['nmse'] for r in rows]))
            best=value if best is None else min(best,value)
    return best


def run_search(grm,traits,params,options):
    """Select/verify/refit covariance settings with one durable TPE controller.

    Arguments follow genetic_varcov_PSD plus validated options. Returns a raw
    fit mapping (public wrapper converts once) or explicit status-only mapping
    if no valid full-data fit exists. Runtime is bounded by configured phase,
    trial, search and separately reserved final budgets, excluding initial
    input hashing/mask placement. Process cleanup is always bounded.
    """
    cfg=normalise_options(options,params)
    if params.get('rank_selection','none')!='none':raise ValueError('hyper_opt owns rank selection; set rank_selection=none')
    if params.get('compatibility') is not None:raise ValueError('hyper_opt excludes compatibility routes')
    if params.get('environment') is not None or isinstance(grm,Mapping) and any('gxe' in str(name).lower() for name in grm):
        raise ValueError('hyper_opt excludes environment/GxE routes')
    if params.get('structural_refine'):raise ValueError('structural SEM selection is a separate scientific task')
    if str(params.get('moment_estimator','mom')).lower() not in {'mom','score'}:raise ValueError('hyper_opt excludes MINQUE; use MoM/score')
    if str(params.get('solver','lbfgs')).lower() not in {'lbfgs','l-bfgs-b','factor'}:raise ValueError("hyper_opt requires solver='lbfgs'")
    if str(params.get('covariance_structure','fa')).lower().replace('-','_') not in {'fa','factor','factor_analytic','psd','pca','eigen','varimax_fa','sparse_fa','sparse_pca','sparse_varimax_fa'}:
        raise ValueError('hyper_opt requires the standard factor covariance path')
    optuna=_load_optuna()
    cfg['optuna_version']=str(optuna.__version__)
    legacy_count=int(params.get('optimize_hyperparameters_cv',0))
    params=dict(params);params.pop('optimize_hyperparameters_cv',None);params.pop('save_path',None)
    params.pop('_runtime_progress',None);params.pop('hyper_opt',None);params.pop('hyper_opt_kwargs',None)
    if params.get('warm_start') is not None:params['warm_start']=None
    params['random_state']=cfg['seed']+53
    groups=params.get('h2_prior_groups')
    if groups is None:params['h2_prior_groups']='global'
    elif not (isinstance(groups,str) and groups in {'global','project'}):
        raise ValueError("with hyper_opt, h2_prior_groups must be 'global' or 'project' (cv['trait_projects']); search other domains through search_space")
    root=Path(cfg.get('checkpoint_dir') or Path.cwd()/'vigor_hpo'/uuid.uuid4().hex[:12]).resolve()
    root.mkdir(parents=True,exist_ok=True);lock=acquire_study_lock(root/'supervisor.lock')
    reporter=None
    try:
        Y,cv,fingerprint,data=_prepare_input(grm,traits,cfg['cv'],root,params,cfg)
        manifest=json.loads((root/'cv/manifest.json').read_text())
        space=prepare_space(cfg['search_space'],Y.shape[1],list(data['kernels']),preset=cfg['search_preset'],fixed=params)
        if set(cfg['guardrails'])-set(manifest['weights']):raise ValueError('guardrails must name declared inner objective regimes')
        atomic_pickle(root/'configuration.pkl',{'config':cfg,'fit_params':params})
        atomic_json(root/'search_space.json',serializable(space))
        atomic_json(root/'configuration.json',serializable(cfg))
        reporter=Reporter(root/'events.jsonl',cfg['n_trials'],cfg['progress'])
        database='sqlite:///'+str(root/'study.sqlite3');study_name=cfg['study_name']
        import warnings
        from concurrent.futures import ThreadPoolExecutor
        def make_sampler(seed):
            """Multivariate/group TPE models parameter interactions and conditional branches."""
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',optuna.exceptions.ExperimentalWarning)
                return optuna.samplers.TPESampler(seed=seed,n_startup_trials=cfg['sampler_startup_trials'],
                    multivariate=cfg['sampler_multivariate'],group=cfg['sampler_multivariate'])
        def make_pruner():
            """Median pruning on the running screening score, compared fold by fold."""
            rule=cfg['pruning']
            if not rule['enabled']:return optuna.pruners.NopPruner()
            return optuna.pruners.MedianPruner(n_startup_trials=rule['startup_trials'],n_warmup_steps=rule['warmup_folds'])
        study=optuna.create_study(direction='minimize',sampler=make_sampler(cfg['seed']+101),pruner=make_pruner(),
                                  storage=database,study_name=study_name,load_if_exists=cfg['resume'])
        abort=root/'jobs'/'ABORT';abort.unlink(missing_ok=True)
        grid=cfg['prediction_grid'] if cfg['prediction_grid'] is not None else DEFAULT_PREDICTION_GRID
        seeds=_seed_configs(cfg['warm_start_from'],cfg['warm_start_top'])
        GRID_MARGIN=.02
        def variants_for(config):
            """Prediction settings still free for a candidate; TPE- or anchor-set keys stay fixed.

            After the start-up trials, settings that never won and never came within
            GRID_MARGIN of a candidate's best are dropped for later candidates. That
            only saves work; every candidate's chosen setting is still verified.
            """
            out=[]
            for variant in grid:
                free={k:v for k,v in variant.items() if k not in config}
                if free not in out:out.append(free)
            stats=state.get('variant_stats',{})
            if cfg['prediction_grid_drop'] and state.get('variant_trials',0)>=cfg['pruning']['startup_trials'] and len(out)>1:
                kept=[v for v in out if digest(v) not in stats or stats[digest(v)]['close']>0]
                if kept:out=kept
            return out if len(out)>1 else [{}]
        # A forced kill can leave RUNNING SQLite trials. They have no completed
        # common-schedule result and are never promoted on resumption.
        for old in study.get_trials(deepcopy=False):
            if old.state==optuna.trial.TrialState.RUNNING:study.tell(old.number,state=optuna.trial.TrialState.FAIL)
        state_path=root/'search_state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {'records':[],'preflight':None,'screening':cfg['prediction_screen'],'started_unix':time.time()}
        records=state['records'];interrupted=False;termination='completed';consecutive_failures=0
        for record in records:
            if record['state']=='RUNNING':record['state']='INTERRUPTED'
        start=time.monotonic();deadline=start+cfg['search_timeout_s'];screen_deadline=start+cfg['search_timeout_s']*(1-cfg['verification_reserve_fraction'])
        base=dict(params);base.update(n_boot=0,analytic_se=False,optimize_hyperparameters_cv=0,rank_selection='none',warm_start=None,
                                      structural_refine=False,structural_plots=False,progress=False)
        incumbent={k:params[k] for k in STATISTICAL if k in params}
        incumbent.update(covariance_structure='fa',rank_selection='none')
        incumbent.setdefault('h2_prior_strength',0.);incumbent.setdefault('h2_prior_groups','global')
        for name in ('factor_structure_G','factor_structure_E'):
            if incumbent.get(name) is None:incumbent[name]='fa'
        for name in ('rank_G','rank_E'):
            if incumbent.get(name)=='auto':incumbent[name]=min(Y.shape[1],max(10,min(100,math.ceil(2*math.sqrt(Y.shape[1])))))
        # Controls without the h2 prior keep every search comparable to the unpenalized estimator.
        anchors=[incumbent,{**incumbent,'rank_G':0,'rank_E':0,'factor_structure_G':'fa','factor_structure_E':'fa',
                  'factor_lambda_G':None,'factor_lambda_E':None,'h2_prior_strength':0.,'h2_prior_groups':'global'},
                 {**incumbent,'shrink_G':1.,'shrink_E':1.,'h2_prior_strength':0.,'h2_prior_groups':'global'}]
        screen=_screen_schedule(manifest)
        verify=[s for s in manifest['scenarios'] if s['role'] in {'objective','paired_diagnostic'}]
        def checkpoint():
            """Commit parent state/trial table after every fold and transition."""
            state['records']=records;atomic_json(state_path,serializable(state))
            atomic_json(root/'trials.json',serializable(records))
        def payload(config,action,scenario=None,fidelity='verify',variants=None,anchor=None):
            """Build a path-only scientific job with a reproducible fold seed."""
            fit_kw={**base,**cfg['fit_'+fidelity]};fit_kw['random_state']=cfg['seed']+53+(0 if scenario is None else 17*scenario['id'])
            prediction=dict(state['screening'] if fidelity=='screen' else cfg['prediction_verify'])
            if cfg['n_jobs']>1 and action in {'fold','preflight'}:
                # Parallel fold workers share the node: divide explicit BLAS thread counts.
                for target in (fit_kw,prediction):
                    threads=target.get('blas_threads')
                    if isinstance(threads,numbers.Integral) and not isinstance(threads,bool):target['blas_threads']=max(1,int(threads)//cfg['n_jobs'])
            message={'data_dir':str(root/'input'),'cv_dir':str(root/'cv'),'cv_options':cv,'candidate':config,
                'fit_kwargs':fit_kw,'action':action,'scenario':scenario,'fingerprint':fingerprint,
                'cache_dir':str(root/'moments'),'cache_max_bytes':cfg['cache_max_bytes'],
                'optimizer_limits':cfg['optimizer_limits'],'prediction':prediction,
                'prediction_verify':cfg['prediction_verify'],'require_converged':cfg['require_converged']}
            if action=='fold' and variants and len(variants)>1:
                message['prediction_grid']=variants;message['condition_floor']=cfg['prediction_condition_floor']
            if action in {'fold','preflight'} and cfg['warm_start_incumbent']:
                message['store_warm']=action=='fold' and anchor==0;message['use_warm']=anchor!=0
            return message
        def job(message,limit,job_deadline=None,phase_limit=None):
            """Run one uniquely named supervised job with shared input placement."""
            return supervised(message,root/'jobs'/uuid.uuid4().hex,timeout_s=limit,
                phase_timeout_s=phase_limit or cfg['phase_timeout_s'],heartbeat_s=cfg['heartbeat_s'],
                memory_limit_bytes=cfg['memory_limit_bytes'],cancel_grace_s=cfg['cancel_grace_s'],
                emit=reporter.emit,stop_file=[f for f in (cfg['stop_file'],abort) if f] if message['action'] in {'fold','preflight','sensitivity'} else None,deadline=job_deadline,
                stage_budgets={'point':cfg['final_refit_timeout_s'],'uncertainty':cfg['uncertainty_timeout_s']} if message['action']=='final' else None)
        def evaluate(record,fidelity,trial=None):
            """Complete common-fidelity coverage or leave an ineligible diagnostic.

            Folds run in waves of n_jobs parallel workers. During screening a
            non-anchor candidate reports its running score after every wave and
            is pruned when the median rule says it cannot catch up; its free
            prediction settings are chosen from the inner variant grid.
            """
            evaluation_started=time.monotonic();reporter.candidate(record['number'],record['config'],fidelity)
            for prior in records:
                if prior is record or digest(prior['config'])!=digest(record['config']) or fidelity not in prior:continue
                record[fidelity]=dict(prior[fidelity]);record[fidelity+'_file']=prior[fidelity+'_file']
                record[fidelity+'_cache_hit']=True
                if fidelity=='verify':record['eligible']=True
                checkpoint();return record[fidelity]
            schedule=screen if fidelity=='screen' else verify;path=root/f'trial_{record["number"]:05d}_{fidelity}.pkl'
            progress_path=root/f'trial_{record["number"]:05d}_{fidelity}_folds.json'
            complete=json.loads(progress_path.read_text()) if progress_path.exists() else []
            outputs=[];covered={x['scenario']:x['file'] for x in complete}
            end=min(deadline,time.monotonic()+cfg['trial_timeout_s'])
            variants=variants_for(record['config']) if fidelity=='screen' else [{}]
            prunable=(trial is not None and fidelity=='screen' and cfg['pruning']['enabled'] and record.get('anchor') is None)
            def run(scenario):
                return job(payload(record['config'],'fold',scenario,fidelity,variants=variants,anchor=record.get('anchor')),
                           end-time.monotonic(),end)
            done=0
            for start in range(0,len(schedule),cfg['n_jobs']):
                wave=list(enumerate(schedule,1))[start:start+cfg['n_jobs']]
                for fold_index,scenario in wave:
                    reporter.emit({'kind':'fold_start','phase':'fold_start','trial':record['number'],
                        'fidelity':fidelity,'fold_index':fold_index,'fold_total':len(schedule),
                        'design':scenario['design'],'scenario':scenario['id'],'project':scenario.get('project','all projects')})
                todo=[scenario for _,scenario in wave if scenario['id'] not in covered]
                if todo and time.monotonic()>=end:raise WorkerFailure('trial wall-time budget exhausted')
                if len(todo)<=1:fetched=[run(scenario) for scenario in todo]
                else:
                    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                        futures=[pool.submit(run,scenario) for scenario in todo]
                        try:fetched=[future.result() for future in futures]
                        except BaseException:
                            # Stop the sibling workers of this wave before re-raising.
                            abort.parent.mkdir(parents=True,exist_ok=True);abort.touch()
                            for future in futures:
                                try:future.result()
                                except BaseException:pass
                            abort.unlink(missing_ok=True)
                            raise
                for scenario,result in zip(todo,fetched):
                    result_path=root/f'fold_{record["number"]:05d}_{fidelity}_{scenario["id"]:05d}.pkl'
                    atomic_pickle(result_path,result);complete.append({'scenario':scenario['id'],'file':str(result_path)})
                    covered[scenario['id']]=str(result_path)
                atomic_json(progress_path,complete)
                if todo:record['last_completed_fold']=todo[-1]['id'];checkpoint()
                for fold_index,scenario in wave:
                    with open(covered[scenario['id']],'rb') as stream:outputs.append(pickle.load(stream))
                    reporter.emit({'kind':'fold_complete','phase':'fold_complete',
                                   'fold_completed':fold_index,'fold_total':len(schedule)})
                done+=len(wave)
                if prunable and done<len(schedule):
                    value=_partial_score(outputs)
                    if value is not None:
                        trial.report(value,done)
                        if trial.should_prune():
                            record['pruned_after_folds']=done;record['pruned_partial_score']=value
                            raise optuna.TrialPruned(f'median rule after {done} screening folds')
            if len(variants)>1 and all('variants' in x for x in outputs):
                choices=[]
                for index in range(len(variants)):
                    if any(x['variants'][index].get('skipped') for x in outputs):
                        choices.append(None);continue
                    choices.append(_objective([r for x in outputs for r in x['variants'][index]['rows']],manifest['weights'],cfg['total_prediction_weight']))
                valid=[i for i,c in enumerate(choices) if c is not None]
                best=min(valid,key=lambda i:choices[i]['score'])
                stats=state.setdefault('variant_stats',{});state['variant_trials']=state.get('variant_trials',0)+1
                for i,choice in enumerate(choices):
                    entry=stats.setdefault(digest(variants[i]),{'settings':serializable(variants[i]),'seen':0,'wins':0,'close':0,'skipped':0})
                    entry['seen']+=1
                    if choice is None:entry['skipped']+=1;continue
                    entry['wins']+=int(i==best)
                    entry['close']+=int(choice['score']<=choices[best]['score']*(1+GRID_MARGIN))
                rows=[r for x in outputs for r in x['variants'][best]['rows']]
                groups=[g for x in outputs for g in x['variants'][best]['groups']]
                summary=choices[best]
                summary['prediction_variants']=[{'settings':variants[i],'skipped':choices[i] is None,
                                                 'score':None if choices[i] is None else choices[i]['score']} for i in range(len(variants))]
                record['prediction_variant']=serializable(variants[best])
                record['config']=serializable({**record['config'],**variants[best]})
            else:
                rows=[r for x in outputs for r in x.get('rows',[])];groups=[r for x in outputs for r in x.get('groups',[])]
                summary=_objective(rows,manifest['weights'],cfg['total_prediction_weight'])
            summary.update(coverage=[s['id'] for s in schedule],fidelity=fidelity,
                fits=[x.get('fit_diagnostics') for x in outputs],solves=[x.get('prediction_diagnostics') for x in outputs],
                elapsed_s=time.monotonic()-evaluation_started,fold_seconds=[x.get('stage_seconds') for x in outputs])
            atomic_pickle(path,{'rows':rows,'groups':groups,'summary':summary})
            atomic_json(root/f'trial_{record["number"]:05d}_{fidelity}_metrics.json',serializable(rows))
            record[fidelity]=summary;record[fidelity+'_file']=str(path)
            if fidelity=='verify':record['eligible']=True
            reporter.emit({'phase':'evaluation_complete','trial':record['number'],
                           'fidelity':fidelity,'score':summary['score']})
            checkpoint();return summary
        if state['preflight'] is None and not cfg['final_refit_only']:
            reporter.emit({'kind':'workflow_stage','phase':'Preflight','workflow_label':'Preflight solve-fidelity checks'})
            # Same covariance, two tolerances. If the relaxed solve is unstable,
            # all candidates use the tighter solve, never a silent approximation.
            diagnostics=[]
            try:
                for scenario in screen:
                    try:
                        result=job(payload(incumbent,'preflight',scenario,'screen'),cfg['trial_timeout_s'],deadline)
                        diagnostics.append(result['preflight'])
                    except WorkerFailure as exc:
                        diagnostics.append({'failure':str(exc),'fallback':'verification fidelity for all screening'})
            except KeyboardInterrupt:
                interrupted=True;termination='interrupted';screen_deadline=time.monotonic();deadline=screen_deadline
            passed=bool(diagnostics) and all('failure' not in d and d['max_prediction_delta_sd']<=cfg['preflight']['max_prediction_delta_sd'] and abs(d['mean_trait_nmse_delta'])<=cfg['preflight']['max_nmse_delta'] for d in diagnostics)
            state['preflight']={'passed':passed,'comparisons':diagnostics,'screening_tightened':not passed}
            if not passed:state['screening']=cfg['prediction_verify']
            checkpoint()
        reporter.emit({'kind':'workflow_stage','phase':'Search','workflow_label':'Screen / verify trials'})
        try:
            while len(records)<cfg['n_trials'] and time.monotonic()<screen_deadline and not cfg['final_refit_only']:
                # Re-seeding each ask from the committed trial number makes
                # sequential resume independent of an unserialized sampler RNG.
                study=optuna.load_study(study_name=study_name,storage=database,
                    sampler=make_sampler(cfg['seed']+101+len(study.trials)),pruner=make_pruner())
                if seeds and len(records)>=len(anchors) and not state.get('seeds_enqueued'):
                    # Enqueued after the anchors so TPE learns from the seeded scores.
                    for seed_config in seeds:
                        encoded=encode(seed_config,space)
                        if encoded:study.enqueue_trial(encoded,skip_if_exists=True)
                    state['seeds_enqueued']=len(seeds);checkpoint()
                trial=study.ask();number=trial.number
                try:
                    config=anchors[len(records)] if len(records)<len(anchors) else suggest(trial,space)
                except (ValueError,TypeError) as exc:
                    study.tell(trial,state=optuna.trial.TrialState.FAIL)
                    records.append({'number':number,'config':{},'state':'FAIL','eligible':False,'failure':'parameter decode: '+str(exc)})
                    consecutive_failures+=1;checkpoint()
                    if consecutive_failures>=cfg['max_consecutive_failures']:termination='repeated_failures';break
                    continue
                record={'number':number,'config':serializable(config),'state':'RUNNING','eligible':False,'anchor':len(records) if len(records)<3 else None}
                records.append(record);trial.set_user_attr('active_model',serializable(config));checkpoint()
                try:
                    if record['anchor']==0:evaluate(record,'verify')
                    result=evaluate(record,'screen',trial)
                    # The chosen prediction variant joins the config; Optuna rejects
                    # attribute updates once a trial is told, so record it first.
                    trial.set_user_attr('active_model',serializable(record['config']))
                    study.tell(trial,values=result['score']);record['state']='COMPLETE'
                    consecutive_failures=0;reporter.best=min(r['screen']['score'] for r in records if 'screen' in r)
                    reporter.bar.update();reporter.emit({'phase':'trial_complete','trial':number,'score':result['score'],
                        'attempted_trials':len(records),'completed_trials':sum(r['state']=='COMPLETE' for r in records),
                        'failed_trials':sum(r['state']=='FAIL' for r in records),'verified_trials':sum(r.get('eligible',False) for r in records),
                        'delta_incumbent':None if 'screen' not in records[0] else result['score']-records[0]['screen']['score']})
                except optuna.TrialPruned as exc:
                    study.tell(trial,state=optuna.trial.TrialState.PRUNED);record.update(state='PRUNED',pruning=str(exc));consecutive_failures=0
                    reporter.bar.update();reporter.emit({'phase':'trial_pruned','trial':number,'reason':str(exc),
                        'attempted_trials':len(records),'pruned_trials':sum(r['state']=='PRUNED' for r in records)})
                except WorkerFailure as exc:
                    study.tell(trial,state=optuna.trial.TrialState.FAIL);record.update(state='FAIL',failure=str(exc),latest=serializable(exc.latest));consecutive_failures+=1
                    reporter.bar.update();reporter.emit({'phase':'trial_failed','trial':number,'reason':str(exc),
                        'attempted_trials':len(records),'failed_trials':sum(r['state']=='FAIL' for r in records)})
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    study.tell(trial,state=optuna.trial.TrialState.FAIL);record.update(state='FAIL',failure=type(exc).__name__+': '+str(exc));consecutive_failures+=1
                    reporter.bar.update();reporter.emit({'phase':'trial_failed','trial':number,'reason':str(exc),
                        'attempted_trials':len(records),'failed_trials':sum(r['state']=='FAIL' for r in records)})
                checkpoint()
                if consecutive_failures>=cfg['max_consecutive_failures']:termination='repeated_failures';break
                early=cfg['early_stopping'];complete=[r for r in records if 'screen' in r]
                if len(complete)>=early['min_trials'] and sum(r.get('eligible',False) for r in records)>=early['min_verified']:
                    prior=complete[:-early['patience']];recent=complete[-early['patience']:]
                    if prior:
                        a=min(r['screen']['score'] for r in prior);b=min(r['screen']['score'] for r in recent)
                        if (a-b)/max(abs(a),1e-12)<early['min_relative_gain']:termination='early_stopped';break
                # Periodic complete-schedule verification creates eligible
                # alternatives before a user interrupts a long search.
                if len(complete)%max(3,cfg['finalists'])==0:
                    shortlist=_verification_shortlist(complete,cfg['finalists'])
                    for candidate in shortlist:
                        if not candidate.get('eligible') and time.monotonic()<deadline:
                            try:evaluate(candidate,'verify')
                            except WorkerFailure as exc:candidate['verification_failure']=str(exc);checkpoint()
            if time.monotonic()>=screen_deadline and termination=='completed':termination='budget_exhausted'
            if not cfg['final_refit_only']:
                shortlist=_verification_shortlist(records,cfg['finalists'])
                for candidate in shortlist:
                    if candidate.get('eligible') or time.monotonic()>=deadline:continue
                    try:evaluate(candidate,'verify')
                    except WorkerFailure as exc:candidate['verification_failure']=str(exc);checkpoint()
        except KeyboardInterrupt:
            interrupted=True;termination='interrupted';checkpoint()
            reporter.emit({'phase':'search_stopped','reason':'first_interrupt; recovering eligible completed candidates'})
        if termination=='repeated_failures':
            warnings.warn(f"hyper_opt stopped after {cfg['max_consecutive_failures']} consecutive failed trials; "
                          "only candidates verified before then can be selected (see report['failure_reasons']).",
                          RuntimeWarning,stacklevel=2)
        reporter.emit({'kind':'workflow_stage','phase':'Selection','workflow_label':'Select verified model'})
        all_verified=[r for r in records if r.get('eligible') and 'verify' in r]
        verified=all_verified
        unrestricted_best=min(all_verified,key=lambda r:r['verify']['score']) if all_verified else None
        baseline=next((r for r in verified if r.get('anchor')==0),None)
        numerical=min(verified,key=lambda r:r['verify']['score']) if verified else None
        selected=numerical;reason='best complete verified score' if numerical is not None else 'no fully verified candidate completed'
        interval=None
        if numerical is not None and baseline is not None and numerical is not baseline:
            selected=baseline;reason='incumbent retained by material-gain, regime guardrail or cluster-interval promotion rule'
            for candidate in sorted(verified,key=lambda r:r['verify']['score']):
                if candidate is baseline:continue
                gain=(baseline['verify']['score']-candidate['verify']['score'])/max(abs(baseline['verify']['score']),1e-12)
                violates=any(candidate['verify']['by_regime'][name]>baseline['verify']['by_regime'][name]*(1+limit) for name,limit in cfg['guardrails'].items())
                if gain<cfg['selection']['min_relative_gain'] or violates:continue
                try:
                    ci=job({'action':'paired_interval','candidate_file':candidate['verify_file'],'baseline_file':baseline['verify_file'],
                            'weights':manifest['weights'],'seed':cfg['seed']+8181,'n_boot':cfg['selection']['n_boot']},cfg['diagnostic_timeout_s'])
                except WorkerFailure as exc:ci={'ci95':None,'reason':'bounded interval calculation failed: '+str(exc)}
                except KeyboardInterrupt:
                    interrupted=True;termination='interrupted';reason='selection interrupted; retain fully verified incumbent';break
                ci_ok=not cfg['selection']['require_cluster_ci'] or ci['ci95'] is not None and ci['ci95'][1]<0
                if ci_ok:selected=candidate;interval=ci;reason='best verified candidate passing material-gain, guardrail and cluster-interval rules';break
        elif numerical is not None and baseline is None:
            if cfg['guardrails']:selected=None;reason='no fully verified incumbent for required deterioration constraints'
            else:reason='best verified candidate; incumbent failed full verification'
        report={'study_name':study_name,'fingerprint':fingerprint,'started_unix':state['started_unix'],'finished_unix':time.time(),
            'termination':termination,'seed':cfg['seed'],
            'sampler':(f'Optuna {optuna.__version__} '+('multivariate/group ' if cfg['sampler_multivariate'] else '')+
                       f'TPE, deterministic per-ask seeds; sequential trials, {cfg["n_jobs"]} parallel fold worker(s)'),
            'n_trials':len(records),'trial_counts':{key:sum(r['state']==key for r in records) for key in ['COMPLETE','PRUNED','FAIL','RUNNING']},
            'eligible_verified':len(verified),
            'pruning_policy':('median rule on the running screening score after each fold wave '
                              f"(startup {cfg['pruning']['startup_trials']} trials, warm-up {cfg['pruning']['warmup_folds']} folds); "
                              'anchors and verification are never pruned' if cfg['pruning']['enabled'] else 'disabled'),
            'pruned_trials':sum(r['state']=='PRUNED' for r in records),
            'failure_reasons':dict(Counter(str(r.get('failure',''))[:160] for r in records if r['state']=='FAIL')),
            'search_preset':cfg['search_preset'],'prediction_grid':grid,
            'prediction_variant_stats':list(state.get('variant_stats',{}).values()),
            'prediction_condition_floor':cfg['prediction_condition_floor'],
            'seeded_from':None if cfg['warm_start_from'] is None else str(cfg['warm_start_from']) if not isinstance(cfg['warm_start_from'],(list,tuple)) else 'configurations',
            'warm_start':('every fold fit starts from the incumbent (anchor 0) fit of the same fold, re-factored to the candidate rank'
                          if cfg['warm_start_incumbent'] else 'cold starts'),'preflight':state['preflight'],'objective':(f"genetic prediction first: training-standardized NMSE of intercept + breeding value (scenario then trait then project then regime) "
                         f"+ {cfg['total_prediction_weight']:g} x the same score for the total phenotype prediction on partially observed animals"),
            'weights':manifest['weights'],'comparison_set':[s['id'] for s in verify],
            'numerical_best':None if numerical is None else numerical['config'],'selected':None if selected is None else selected['config'],
            'selection_reason':reason,'paired_cluster_interval':interval,'bootstrap_unit':manifest['bootstrap_unit'],
            'unrestricted_best':None if unrestricted_best is None else unrestricted_best['config'],
            'h2_prior_comparison':{mode:{'verified':sum((r['config'].get('h2_prior_strength',0)>0)==active for r in all_verified),
                'best_verified_score':min((r['verify']['score'] for r in all_verified if (r['config'].get('h2_prior_strength',0)>0)==active),default=None)}
                for mode,active in (('without_prior',False),('with_prior',True))},
            'uncertainty':'inner-selection conditional; analytic SEs condition on the selected rank/support/h2-prior strength and exclude nuisance-fit estimation uncertainty',
            'outer_evaluation':'not run by search; use evaluate_hyperopt_outer for independently nested outer evaluation',
            'checkpoint':str(root),'trials':str(root/'trials.json'),'cv_manifest':str(root/'cv/manifest.json'),
            'resume_schedule':'equivalent sequential completed schedules use deterministic ask seeds; killed/incomplete trials are recorded FAIL and may change later suggestions',
            'genomic_preprocessing':manifest['genotypes'],
            'cache':'fold-specific base moments, h2-pilot geometry, kernel eigensystems and covariance fits (shared by candidates differing only in post-fit settings)'}
        report['input_summary']={'n_animals':len(Y),'n_traits':Y.shape[1],'observed_per_trait':Y.notna().sum().tolist(),
            'traits':list(Y.columns),'kernel_names':list(data['kernels']),
            'sample_metadata_fields':list(cv['sample_metadata'].columns) if isinstance(cv.get('sample_metadata'),pd.DataFrame) else []}
        report['scenario_roles']={'screening':[s['id'] for s in screen],'full_verification':[s['id'] for s in verify],
            'separate_outer_or_representation':[s['id'] for s in manifest['scenarios'] if s['role'] in {'outer','representation'}]}
        report['not_estimated']=['selection-adjusted covariance uncertainty','predictive intervals/log scores',
                                 'known-truth h2/rG bias or coverage','loading support stability across independent initializations']
        if selected is not None:
            try:
                transfers=job({'action':'transfer_report','candidate_file':selected['verify_file'],'manifest':str(root/'cv/manifest.json'),
                               'seed':cfg['seed']+8181,'n_boot':cfg['selection']['n_boot']},cfg['diagnostic_timeout_s'])
                atomic_json(root/'transfer_gains.json',serializable(transfers));report['transfer_gains']=str(root/'transfer_gains.json')
            except WorkerFailure as exc:report['transfer_gains_status']='failed: '+str(exc)
            except KeyboardInterrupt:interrupted=True;termination='interrupted';report['transfer_gains_status']='interrupted'
        atomic_json(root/'report.json',serializable(report));checkpoint()
        result={'success':False,'hpo_status':'no_valid_candidate','hpo_final_refit_complete':False}
        config=None if selected is None else selected['config']
        if selected is not None and not cfg['checkpoint_only']:
            reporter.emit({'kind':'workflow_stage','phase':'final_refit','workflow_label':'Final full-training covariance fit','parameters':config})
            final_message=payload(config,'final');final_kwargs={**params,**cfg['fit_verify']}
            final_kwargs.update(optimize_hyperparameters_cv=0,rank_selection='none',warm_start=None,
                                random_state=cfg['seed']+53,progress=False,structural_refine=False,structural_plots=False)
            if final_kwargs.get('h2_prior_strength',0)>0 and final_kwargs.get('n_boot',0):final_kwargs['n_boot']=0
            final_message['fit_kwargs']=final_kwargs
            final_message['point_path']=str(root/'point_fits'/(uuid.uuid4().hex+'.pkl'))
            final_result=None;failure=None
            for attempt in range(1+cfg['final_refit_retries']):
                try:
                    final_result=job(final_message,cfg['final_refit_timeout_s']+cfg['uncertainty_timeout_s'],
                                     phase_limit=max(cfg['phase_timeout_s'],cfg['final_refit_timeout_s']))
                    break
                except KeyboardInterrupt:failure='refit_interrupted';break
                except WorkerFailure as exc:
                    failure=str(exc)
                    if Path(final_message['point_path']).exists():break
                    if attempt<cfg['final_refit_retries']:
                        final_message['fit_kwargs']['maxiter']=2*int(final_kwargs.get('maxiter',1000))
                        reporter.emit({'phase':'final_refit_recovery','changed_numerics':{'maxiter':final_message['fit_kwargs']['maxiter']},'reason':failure})
            if final_result is None and Path(final_message['point_path']).exists():
                # Point fit checkpoint precedes optional uncertainty. Reattach
                # preprocessing without re-entering the covariance optimizer.
                with open(final_message['point_path'],'rb') as stream:point=pickle.load(stream)
                final_result={'fit':point,'covariance_diagnostics':None}
                final_result['fit']['hpo_uncertainty_status']='unavailable_after_'+str(failure)
            if final_result is not None:
                result=dict(final_result['fit']);result['hpo_final_refit_complete']=True
                converged=(final_result.get('fit_diagnostics') or {}).get('optimizer_converged')
                result['hpo_final_optimizer_converged']=converged
                if converged is False:
                    warnings.warn('the selected model final refit stopped at its iteration cap; raise maxiter / '
                                  'factor_sparse_maxiter for the final fit (screening fits may stay capped).',
                                  RuntimeWarning,stacklevel=2)
                ase=result.get('analytic_se',{})
                requested=bool(final_kwargs.get('analytic_se') or final_kwargs.get('n_boot'))
                unavailable=isinstance(ase,dict) and ase.get('available') is False
                incomplete_boot=int(result.get('n_boot_success',0))<int(final_kwargs.get('n_boot',0))
                uncertainty='unavailable_or_partial' if unavailable or incomplete_boot else 'requested_raw_fit_uncertainty_completed' if requested else 'not_requested'
                result['hpo_uncertainty_status']=result.get('hpo_uncertainty_status',uncertainty)
                atomic_pickle(final_message['point_path'],result)
                result['hpo_status']='interrupted_refit_complete' if interrupted else termination if termination in {'early_stopped','budget_exhausted','repeated_failures'} else 'completed'
                if failure=='refit_interrupted':result['hpo_status']='refit_interrupted'
                report['covariance_diagnostics']=final_result.get('covariance_diagnostics')
            else:result.update(hpo_status='refit_interrupted' if failure=='refit_interrupted' else 'refit_failed',hpo_refit_error=failure)
            result['hpo_final_refit_params']=serializable({**config,**{k:final_message['fit_kwargs'][k] for k in ['maxiter','gtol','ftol','analytic_se','n_boot','blas_threads'] if k in final_message['fit_kwargs']}})
            if result.get('hpo_final_refit_complete') and manifest.get('project_addition_sensitivity') and not interrupted and failure!='refit_interrupted':
                sensitivity=[];diagnostic_end=time.monotonic()+cfg['diagnostic_timeout_s']
                labels=np.asarray(manifest['trait_projects']);observed=Y.notna().to_numpy()
                for project in pd.unique(labels):
                    if time.monotonic()>=diagnostic_end:
                        sensitivity.append({'status':'budget_exhausted'});break
                    columns=np.flatnonzero(labels==project);mask_file=root/f'sensitivity_{len(sensitivity):05d}.npz'
                    with mask_file.open('wb') as stream:
                        np.savez_compressed(stream,train=np.packbits(observed.ravel()),test=np.packbits(np.zeros_like(observed).ravel()),
                                            shape=np.array(Y.shape),columns=columns)
                    scenario={'id':100000+len(sensitivity),'mask_file':str(mask_file),'mask_sha256':digest({'columns':columns.tolist(),'fingerprint':fingerprint}),
                              'design':'project_addition_sensitivity','project':str(project),'role':'diagnostic'}
                    try:
                        summary=job(payload(config,'sensitivity',scenario),diagnostic_end-time.monotonic(),diagnostic_end)
                        sensitivity.append({'project':str(project),'status':'complete',**summary})
                    except (WorkerFailure,KeyboardInterrupt) as exc:
                        sensitivity.append({'project':str(project),'status':'failed_or_interrupted','error':str(exc)})
                        if isinstance(exc,KeyboardInterrupt):break
                atomic_json(root/'project_addition_sensitivity.json',serializable({'focal_only':sensitivity,'all_project':report.get('covariance_diagnostics'),
                    'interpretation':'covariance sensitivity, not a prediction objective or an unbiased h2 validation'}))
                report['project_addition_sensitivity']=str(root/'project_addition_sensitivity.json')
        elif selected is not None:result['hpo_status']='checkpoint_only'
        result.update(hyper_opt=True,optimize_hyperparameters_cv=legacy_count,
            hpo_best_params=None if numerical is None else numerical['config'],hpo_selected_params=config,
            hpo_selection_reason=reason,hpo_score_by_regime=None if selected is None else selected['verify']['by_regime'],
            hpo_cv_manifest=str(root/'cv/manifest.json'),hpo_trial_table=str(root/'trials.json'),hpo_checkpoint=str(root),
            hpo_covariance_view='retention_adjusted_prediction' if result.get('hpo_final_refit_complete') else 'unavailable',
            hpo_uncertainty_contract=report['uncertainty'],hyperparameter_optimization=report)
        result['hpo_final_prediction_params']=serializable(cfg['prediction_verify'])
        report.update(termination=termination,finished_unix=time.time(),elapsed_s=time.monotonic()-start,
                      final_point_status=result.get('hpo_status'),uncertainty_status=result.get('hpo_uncertainty_status'))
        atomic_json(root/'report.json',serializable(dict(report,status=result['hpo_status'],final_refit_complete=result['hpo_final_refit_complete'])))
        atomic_pickle(root/'returned_result.pkl',result)
        if result.get('hpo_final_refit_complete'):
            try:
                reporter.emit({'kind':'workflow_stage','phase':'Save model','workflow_label':'Save selected model'})
                model_path=root/'models'/(uuid.uuid4().hex+'.zarr')
                persisted=job({'action':'persist_final','point_path':str(root/'returned_result.pkl'),'model_path':str(model_path)},cfg['diagnostic_timeout_s'])
                result['hpo_model_path']=persisted['model_path'];result['hpo_persistence_status']='complete'
                atomic_json(root/'model_manifest.json',{'write_state':'complete','path':persisted['model_path'],'fingerprint':fingerprint,
                                                       'covariance_view':result['hpo_covariance_view']})
            except (WorkerFailure,KeyboardInterrupt) as exc:
                result['hpo_persistence_status']='failed_or_interrupted';result['hpo_persistence_error']=str(exc)
            atomic_pickle(root/'returned_result.pkl',result)
            atomic_json(root/'report.json',serializable(dict(report,persistence_status=result.get('hpo_persistence_status'))))
        reporter.emit({'kind':'workflow_complete','phase':'Complete','workflow_status':result['hpo_status']})
        return result
    finally:
        if reporter is not None:reporter.close()
        release_study_lock(lock)


def hyperopt_options(function):
    """Add explicit hyper_opt entry points outside the rank-CV/runtime wrappers.

    hyper_opt defaults False; hyper_opt_kwargs defaults None. Legacy
    optimize_hyperparameters_cv=k maps to the two-regime compound preset with
    k folds. Unknown simultaneous controllers are rejected. O(1) when disabled.
    """
    signature=inspect.signature(function)
    @functools.wraps(function)
    def extended(*args,hyper_opt=False,hyper_opt_kwargs=None,**kwargs):
        """Run ordinary fit or a single supervised search controller."""
        from .hyperparameter_cv import validate_cv_count
        legacy=validate_cv_count(kwargs.get('optimize_hyperparameters_cv',0))
        if not isinstance(hyper_opt,bool):raise TypeError('hyper_opt must be bool')
        if hyper_opt and legacy:raise ValueError('choose hyper_opt=True or optimize_hyperparameters_cv=k, not both')
        if not hyper_opt and not legacy:
            if hyper_opt_kwargs is not None:raise ValueError('hyper_opt_kwargs requires hyper_opt=True')
            return function(*args,**kwargs)
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();params=dict(bound.arguments)
        extra=params.pop('legacy_kwargs',{});params.update(extra)
        unsupported=set(extra)-{'compatibility','environment','progress_backend','progress_style'}
        if unsupported:raise TypeError(f'unsupported hyper_opt covariance arguments: {sorted(unsupported)}')
        grm=params.pop('grm');traits=params.pop('traits')
        options=dict(hyper_opt_kwargs or {})
        if legacy:
            if hyper_opt_kwargs is not None:raise ValueError('use hyper_opt=True for custom compound configuration')
            options.update(cv={'designs':['unseen_animal','partial_cell'],'inner_folds':legacy},
                           n_trials=max(24,min(64,8*legacy)))
        result=run_search(grm,traits,params,options)
        result['hyper_opt']=True
        result['optimize_hyperparameters_cv']=legacy
        return result
    additions=[inspect.Parameter('hyper_opt',inspect.Parameter.KEYWORD_ONLY,default=False),
               inspect.Parameter('hyper_opt_kwargs',inspect.Parameter.KEYWORD_ONLY,default=None)]
    values=list(signature.parameters.values());position=next((i for i,p in enumerate(values) if p.kind==p.VAR_KEYWORD),len(values))
    extended.__signature__=signature.replace(parameters=values[:position]+additions+values[position:])
    extended.__doc__=(function.__doc__ or '')+'''\n\nCompound HPO\n------------\nhyper_opt : bool, default False\n    Enable supervised training-only TPE with independent G/E and named-component ranks,\n    compound metadata-based CV, prediction-retention and h2-prior search, durable checkpoints,\n    hard time/memory ceilings and separately budgeted final refit.\nhyper_opt_kwargs : mapping or None, default None\n    See docs/HYPER_OPT.md for supported designs, conditional spaces, budgets,\n    resume/interruption semantics and the explicit predict_hyperopt deployment view.\n    Cost is the sum of fold candidate fits/predictions plus full-training refit and\n    requested uncertainty. Outer test outcomes must remain protected.\n'''
    return extended


def predict_hyperopt(traits, grm, covariance_fit, *, covariates=None, **prediction_kwargs):
    """Predict using the covariance view and nuisance mapping verified by HPO.

    Parameters
    ----------
    traits : pandas.DataFrame, shape (n,t)
        Authorized conditioning observations, with validation/test cells NaN.
        Trait order is aligned to the fit. Sample order must match grm.
    grm : kernel or named kernels
        Same kernel definitions/scaling as training, aligned to prediction rows.
    covariance_fit : xarray.Dataset or raw mapping
        Successful genetic_varcov_PSD(hyper_opt=True) result. A missing selected
        selected view is an error, never an implicit substitution.
    covariates : pandas.DataFrame or None
        Fixed nuisance columns used during training, including their original
        encoding, but excluding the intercept. Required when the fit used them.
    **prediction_kwargs : mapping
        Optional prediction tolerance, iterations, preconditioner and BLAS
        overrides. Full kernel/float64/exact conditioning and center=False are
        fixed to preserve the declared deployment model.

    Returns
    -------
    xarray.Dataset
        Native prediction result, with frozen fixed predictions restored and
        hpo_covariance_view labeled. Genetic deviations retain their meaning.

    Complexity
    ----------
    O(ntp) nuisance adjustment plus full-kernel mvPredict PCG complexity.
    No covariance fitting or HPO occurs here.
    """
    from . import multivariate as mv
    from .hyperopt_worker import covariance_view
    import xarray as xr
    view=mv._result_view(covariance_fit)
    if not view.get('hpo_final_refit_complete',False):raise ValueError('no complete HPO full-training point fit')
    mode=view.get('hpo_covariance_view')
    if mode!='retention_adjusted_prediction':raise ValueError('selected HPO covariance view unavailable')
    Y=mv._as_pandas_input(traits)
    if not isinstance(Y,pd.DataFrame):raise TypeError('predict_hyperopt requires a labeled trait DataFrame')
    Y=Y.copy();Y.index=Y.index.astype(str);Y.columns=Y.columns.astype(str)
    if Y.index.has_duplicates or Y.columns.has_duplicates:raise ValueError('sample/trait labels must be unique')
    coefficients=view['hpo_fixed_coefficients']
    if hasattr(coefficients,'dims'):
        labels=coefficients['trait'].values.astype(str);names=coefficients['fixed_effect'].values.astype(str)
        coefficients=np.asarray(coefficients.transpose('fixed_effect','trait'),float)
    elif isinstance(coefficients,pd.DataFrame):
        labels=coefficients.columns.astype(str);names=coefficients.index.astype(str);coefficients=coefficients.to_numpy(float)
    else:raise TypeError('labeled HPO nuisance coefficients are required')
    Y=Y.loc[:,labels]
    if len(names)>1:
        if not isinstance(covariates,pd.DataFrame):raise ValueError('training nuisance covariates are required for prediction')
        covariates=covariates.copy();covariates.index=covariates.index.astype(str);covariates.columns=covariates.columns.astype(str)
        if covariates.index.has_duplicates or covariates.columns.has_duplicates:raise ValueError('covariate labels must be unique')
        matrix=covariates.loc[Y.index,list(names[1:])].to_numpy(float)
        if not np.isfinite(matrix).all():raise ValueError('prediction covariates must be finite')
        design=np.column_stack([np.ones(len(Y)),matrix])
    else:
        if covariates is not None:raise ValueError('fit used intercept only; prediction covariates would change the model')
        design=np.ones((len(Y),1))
    fixed=pd.DataFrame(design@coefficients,index=Y.index,columns=Y.columns)
    conditioning=Y-fixed;g,e=covariance_view(view)
    allowed={'solver_tol','solver_maxiter','preconditioner','blas_threads','return_alpha','return_state','progress','jitter'}
    if set(prediction_kwargs)-allowed:raise ValueError('HPO prediction fixes exact/full-kernel/float64/uncentered inference; unsupported override')
    controls={**dict(view.get('hpo_final_prediction_params',{})),**prediction_kwargs}
    output=mv.mvPredict(conditioning,grm,g,E=e,center=False,accuracy='exact',work_dtype='float64',kernel_rank='full',**controls)
    fixed_array=xr.DataArray(fixed.to_numpy(),dims=('sample','trait'),coords={'sample':output['sample'],'trait':output['trait']})
    for key in ('genetic_mean','phenotype_prediction'):
        if key in output:output[key]=output[key]+fixed_array
    output['fixed_prediction']=fixed_array
    output.attrs.update(hpo_covariance_view=mode,hpo_checkpoint=view.get('hpo_checkpoint'),
                        hpo_prediction_contract='frozen training nuisance mapping and explicitly verified covariance view')
    return output


def evaluate_hyperopt_outer(grm, traits, *, outer_cv, hyper_opt_kwargs=None,
                           covariance_kwargs=None, out='vigor_outer_cv'):
    """Run genuinely nested outer evaluation, keeping outer truth from HPO.

    Parameters
    ----------
    grm, traits : kernels and labeled phenotype DataFrame
        Genotypes are transductive; nuisance/HPO/h2-prior estimation see training cells
        only. Sample/trait axes must match. No outputs from a previous full-data
        search initialize an outer model.
    outer_cv : mapping
        CV metadata/designs as in prepare_cv. Family/generation/forward/site,
        same-trait project holdout, few-shot and source-frozen probe/association
        designs are supported. All resulting masks are outer evaluation masks.
    hyper_opt_kwargs : mapping or None
        Inner HPO options. A separate checkpoint directory is assigned per outer
        scenario. Inner design metadata may be specified independently; nuisance
        covariates must match outer_cv['covariates'] when supplied.
    covariance_kwargs : mapping or None
        Prespecified ordinary genetic_varcov_PSD controls; no search controller.
    out : path
        Owned outer-run directory for masks, nested studies, scores and report.

    Returns
    -------
    pandas.DataFrame
        Per-trait outer metrics, never inner selection scores. Association-only
        representations are reported separately and are not prediction accuracy.

    Complexity
    ----------
    One independent complete HPO/refit per outer scenario plus supervised outer
    prediction/probe work. Costs multiply; no outer evaluation is run implicitly.
    """
    from .multivariate import genetic_varcov_PSD,_result_view
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=True)
    params=dict(covariance_kwargs or {})
    if any(k in params for k in ('hyper_opt','hyper_opt_kwargs')) or any(params.get(k) for k in ('optimize_hyperparameters_cv','warm_start')):raise ValueError('outer covariance kwargs cannot contain a prior search/warm start')
    inner=dict(hyper_opt_kwargs or {});inner_cfg=normalise_options(inner,params)
    # Input placement uses a fresh identity; a completed outer manifest is the
    # durable owner of the per-fold studies, not a model selected on outer truth.
    cv=dict(outer_cv);cv.pop('weights',None)
    input_cfg={**inner_cfg,'cv':cv,'resume':bool(inner.get('resume',False)),'allow_outer_only':True}
    Y,cv,fingerprint,data=_prepare_input(grm,traits,cv,root,params,input_cfg)
    manifest=json.loads((root/'cv/manifest.json').read_text());all_rows=[];diagnostics=[]
    for scenario in manifest['scenarios']:
        train_mask,_,columns=load_masks(scenario);train=Y.iloc[:,columns].mask(~train_mask[:,columns])
        local=dict(inner);local['checkpoint_dir']=str(root/f'inner_{scenario["id"]:05d}')
        local_cv=dict(local.get('cv',{}))
        for key in ('trait_projects','trait_blocks'):
            if key not in local_cv and key in cv:
                labels=aligned_labels(cv[key],Y.columns,key);local_cv[key]=pd.Series(labels[columns],index=train.columns)
            elif key in local_cv:
                labels=aligned_labels(local_cv[key],Y.columns,key);local_cv[key]=pd.Series(labels[columns],index=train.columns)
        for key in ('sample_metadata','covariates'):
            if key not in local_cv and key in cv:local_cv[key]=cv[key]
        if ('covariates' in local_cv)!=('covariates' in cv):raise ValueError('inner and outer nuisance designs must match')
        if 'covariates' in cv:
            candidate=local_cv['covariates'].copy();candidate.index=candidate.index.astype(str)
            if not candidate.reindex(Y.index).equals(cv['covariates'].reindex(Y.index)):
                raise ValueError('inner and outer nuisance covariates must be identical')
        local_cv.pop('protected_mask',None);local['cv']=local_cv
        fit=genetic_varcov_PSD(grm,train,hyper_opt=True,hyper_opt_kwargs=local,**params)
        result=_result_view(fit)
        if not result.get('hpo_final_refit_complete') or result.get('hpo_covariance_view')=='unavailable':
            raise RuntimeError(f'outer scenario {scenario["id"]}: no deployable inner fit; see {local["checkpoint_dir"]}')
        message={'data_dir':str(root/'input'),'cv_dir':str(root/'cv'),'cv_options':cv,
            'candidate':result['hpo_selected_params'],'fit_kwargs':{},'action':'outer_predict',
            'scenario':scenario,'fingerprint':fingerprint,'optimizer_limits':inner_cfg['optimizer_limits'],
            'prediction':inner_cfg['prediction_verify'],'point_path':str(Path(local['checkpoint_dir'])/'returned_result.pkl')}
        output=supervised(message,root/'outer_jobs'/uuid.uuid4().hex,timeout_s=inner_cfg['trial_timeout_s'],
            phase_timeout_s=inner_cfg['phase_timeout_s'],heartbeat_s=inner_cfg['heartbeat_s'],
            memory_limit_bytes=inner_cfg['memory_limit_bytes'],cancel_grace_s=inner_cfg['cancel_grace_s'])
        all_rows+=output.get('rows',[]);diagnostics+=output.get('representation',[])
        atomic_json(root/'outer_metrics.json',serializable(all_rows));atomic_json(root/'representation_associations.json',serializable(diagnostics))
    frame=pd.DataFrame(all_rows)
    frame.to_csv(root/'outer_metrics.csv',index=False)
    atomic_json(root/'outer_report.json',{'status':'complete','scope':'independently nested outer evaluation; no selection on outer truth',
        'manifest':str(root/'cv/manifest.json'),'metrics':str(root/'outer_metrics.csv'),'representation_associations':str(root/'representation_associations.json')})
    return frame


def resume_hyperopt(checkpoint_dir, *, final_refit_only=False, **budget_overrides):
    """Resume trusted owned checkpoint state without reconstructing CV metadata.

    Parameters
    ----------
    checkpoint_dir : path
        Existing HPO directory produced by this source version. It contains
        trusted internal pickles; never use an untrusted downloaded checkpoint.
    final_refit_only : bool, default False
        Recover eligible candidates and retry full-training refit, skipping
        search/preflight. Selection and protected training masks are retained.
    **budget_overrides : mapping
        n_trials, search/trial/phase/final/uncertainty timeout seconds,
        progress, memory_limit_bytes, heartbeat_s or stop_file only. Statistical
        settings/data/source changes require a new study instead.

    Returns
    -------
    xarray.Dataset
        Same labeled fit/status contract as genetic_varcov_PSD(hyper_opt=True).
    Complexity: input hashing plus remaining scheduled work/refit; completed
    fold results are reused only with identical fingerprint and fidelity.
    """
    from .hyperopt_worker import load_data
    from .multivariate import _covariance_fit_dataset
    allowed={'n_trials','search_timeout_s','trial_timeout_s','phase_timeout_s','final_refit_timeout_s',
             'uncertainty_timeout_s','progress','memory_limit_bytes','heartbeat_s','stop_file'}
    if set(budget_overrides)-allowed:raise ValueError('resume only accepts resource/display budgets; statistical changes need a new study')
    root=Path(checkpoint_dir).resolve()
    with (root/'configuration.pkl').open('rb') as stream:configuration=pickle.load(stream)
    meta,values,kernels=load_data(root/'input')
    config={k:v for k,v in configuration['config'].items() if k in OPTIONS}
    config.update(budget_overrides,checkpoint_dir=str(root),resume=True,final_refit_only=final_refit_only)
    if final_refit_only:config['checkpoint_only']=False
    params=dict(configuration['fit_params'])
    traits=pd.DataFrame(np.asarray(values),index=meta['samples'],columns=meta['traits'])
    result=run_search(kernels,traits,params,config)
    return _covariance_fit_dataset(result)
