"""Outcome-independent compound CV masks and project-balanced prediction metrics.

Masks are committed before candidates are generated. Large masks live packed
on disk, one scenario at a time. Unknown or impossible designs fail explicitly.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numbers

import numpy as np
import pandas as pd

from .hyperopt_runtime import atomic_json

DESIGNS = {'unseen_animal', 'partial_cell', 'trait_block', 'few_shot_project',
           'donor_ablation', 'family_block', 'generation_block', 'forward_generation',
           'batch_or_site_block', 'unseen_project_same_trait',
           'frozen_embedding_probe', 'frozen_embedding_association',
           'project_addition_sensitivity'}
DESIGNS.add('connectivity_stratified')
CV_KEYS = {'designs', 'trait_projects', 'trait_blocks', 'sample_metadata',
           'robustness_blocks', 'target_calibration_fractions', 'inner_folds',
           'weights', 'average', 'min_train', 'min_validation', 'min_calibration',
           'partial_fraction', 'protected_mask', 'covariates', 'generation_order',
           'batch_column', 'project_column', 'bootstrap_group', 'target_projects',
           'disable_cross_project_baseline', 'embedding_rank', 'probe_ridge'}
CV_KEYS.update({'connectivity_kernel','connectivity_reference_animals'})


def positive_int(value, name, minimum=1):
    """Validate an integer >= minimum, rejecting bool; O(1), returns int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def aligned_labels(value, axis, name, default=None):
    """Align a labeled Series/mapping or positional vector; O(axis length).

    Missing/duplicate labels are errors. None uses the explicitly supplied
    scalar default, never inferred projects or families from row order.
    """
    if value is None:
        if default is None: raise ValueError(f'{name} is required')
        return np.repeat(str(default), len(axis))
    if isinstance(value, dict): value = pd.Series(value)
    if isinstance(value, pd.Series):
        if value.index.has_duplicates: raise ValueError(f'duplicate {name} labels')
        value = value.reindex(axis).to_numpy()
    value = np.asarray(value)
    if value.shape != (len(axis),) or pd.isna(value).any(): raise ValueError(f'{name} must align without missing labels')
    return value.astype(str)


def prepare_cv(traits, config, directory, seed, *, allow_outer_only=False):
    """Persist deterministic masks and return a compact CV manifest.

    traits is a unique-labeled animal x trait frame, NaNs allowed. config
    contains only CV_KEYS. directory is an owned checkpoint path. seed is int.
    Time O(s*n*t), space O(n*t) plus compressed masks, for s scenarios.
    All eligibility is frozen before any predictions. No phenotype values
    determine masks, except finite-observation availability.
    """
    cfg = dict(config); unknown = set(cfg)-CV_KEYS
    if unknown: raise TypeError(f'unknown cv options: {sorted(unknown)}')
    if cfg.get('average', 'project_then_trait') != 'project_then_trait': raise ValueError('average must be project_then_trait')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    observed = np.isfinite(traits.to_numpy(float)); n,t = observed.shape
    folds = positive_int(cfg.get('inner_folds', 2), 'inner_folds', 2)
    min_train = positive_int(cfg.get('min_train', 3), 'min_train', 3)
    min_test = positive_int(cfg.get('min_validation', 3), 'min_validation', 2)
    min_cal = positive_int(cfg.get('min_calibration', 3), 'min_calibration', 3)
    projects = aligned_labels(cfg.get('trait_projects'), traits.columns, 'trait_projects', 'all')
    metadata = cfg.get('sample_metadata')
    if metadata is None: metadata = pd.DataFrame(index=traits.index)
    if not isinstance(metadata, pd.DataFrame) or metadata.index.has_duplicates: raise TypeError('sample_metadata must be a uniquely indexed DataFrame')
    metadata = metadata.reindex(traits.index)
    designs = list(cfg.get('designs', ['unseen_animal', 'partial_cell']))
    if not designs or len(designs)!=len(set(designs)) or set(designs)-DESIGNS: raise ValueError('designs must be unique supported design names')
    family_available='family' in metadata and not metadata['family'].isna().any() and metadata['family'].nunique()>=folds
    robustness = list(cfg.get('robustness_blocks', ['family'] if family_available else []))
    if set(robustness)-{'family','generation'}: raise ValueError('robustness_blocks supports family/generation')
    for field in robustness:
        if field not in metadata or metadata[field].isna().any(): raise ValueError(f'robustness block {field} requires complete sample_metadata')
    rng = np.random.default_rng(seed)
    rows_with_data = observed.any(axis=1)
    if rows_with_data.sum() < folds: raise ValueError('not enough observed animals for inner folds')
    schedule=[]; exclusions=[]
    def assignments(field=None):
        """Assign animals or entire metadata groups to frozen folds."""
        group = np.arange(n).astype(str) if field is None else aligned_labels(metadata[field],traits.index,field)
        unique = pd.unique(group[rows_with_data]).copy(); rng.shuffle(unique)
        if len(unique)<folds: raise ValueError(f'not enough {field or "animal"} groups for folds')
        return [np.isin(group, part) for part in np.array_split(unique, folds)]
    default_splits=assignments()
    block_splits={field:assignments(field) for field in robustness}
    def commit(design, fold, train, test, *, project=None, budget=None,
               comparison='all_donors', block=None, role='objective', columns=None, extra=None):
        """Validate and atomically describe one predeclared mask pair."""
        train=train & observed; test=test & observed
        if np.any(train & test): raise AssertionError('train/test leakage in CV mask')
        active = observed.any(0) if columns is None else np.isin(np.arange(t), columns)
        targets = np.flatnonzero(test.any(0))
        bad = [str(traits.columns[j]) for j in targets if test[:,j].sum()<min_test or (role!='representation' and train[:,j].sum()<min_train)]
        if bad: raise ValueError(f'{design}, fold={fold}, project={project}, budget={budget}: insufficient prespecified train/test counts for {bad[:8]}')
        active &= train.sum(0)>=min_train
        if not len(targets) or not active.any(): raise ValueError(f'{design} has no identifiable test targets')
        if role!='representation' and not np.all(active[targets]): raise ValueError('target trait absent from training')
        if columns is not None:
            dropped=np.flatnonzero(np.isin(np.arange(t),columns)&~active)
            if len(dropped): exclusions.append({'design':design,'fold':fold,'source_traits':traits.columns[dropped].astype(str).tolist(),'reason':'insufficient training observations, fixed before search'})
        train[:,~active]=False
        index=len(schedule); file=directory/f'mask_{index:05d}.npz'
        packed_train=np.packbits(train.ravel());packed_test=np.packbits(test.ravel())
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,train=packed_train,test=packed_test,shape=np.array([n,t]),columns=np.flatnonzero(active))
        file.with_suffix('.tmp').replace(file)
        digest=hashlib.sha256(packed_train.tobytes()+packed_test.tobytes()+np.flatnonzero(active).tobytes()).hexdigest()
        schedule.append({'id':index,'design':design,'fold':fold,'project':project,'budget':budget,
                         'comparison':comparison,'block':block,'role':role,'mask_file':str(file.resolve()),
                         'mask_sha256':digest,'n_train':int(train.sum()),'n_test':int(test.sum()),
                         'target_traits':traits.columns[targets].astype(str).tolist(),**(extra or {})})
    base = observed.copy()
    for design in designs:
        if design in {'donor_ablation','project_addition_sensitivity','connectivity_stratified'}: continue
        if design in {'unseen_animal','partial_cell','trait_block'}:
            schedules=[(None,default_splits)]+list(block_splits.items())
            for block,splits in schedules:
                for fold,rows in enumerate(splits):
                    train=base.copy();test=np.zeros_like(base)
                    if design=='unseen_animal': test[rows]=base[rows];train[rows]=False
                    elif design=='partial_cell':
                        fraction=cfg.get('partial_fraction',.5)
                        if not 0<fraction<1: raise ValueError('partial_fraction must be in (0,1)')
                        for i in np.flatnonzero(rows & (base.sum(1)>=2)):
                            cols=rng.permutation(np.flatnonzero(base[i]));count=max(1,min(len(cols)-1,int(fraction*len(cols))))
                            test[i,cols[:count]]=True;train[i,cols[:count]]=False
                    else:
                        groups=aligned_labels(cfg.get('trait_blocks',cfg.get('trait_projects')),traits.columns,'trait_blocks')
                        for label in pd.unique(groups):
                            local=base.copy();held=base & rows[:,None] & (groups==label)[None,:]
                            eligible=(base[:,groups!=label].sum(1)>0)
                            held &= eligible[:,None];local[held]=False
                            commit(design,fold,local,held,project=str(label),block=block)
                        continue
                    commit(design,fold,train,test,block=block)
        elif design in {'family_block','generation_block','batch_or_site_block'}:
            field={'family_block':'family','generation_block':'generation','batch_or_site_block':cfg.get('batch_column','batch')}[design]
            if field not in metadata: raise ValueError(f'{design} requires sample_metadata[{field!r}]')
            for fold,rows in enumerate(assignments(field)):
                train=base.copy();test=base & rows[:,None];train[rows]=False
                commit(design,fold,train,test,block=field)
        elif design=='forward_generation':
            field='generation'
            if field not in metadata or metadata[field].isna().any(): raise ValueError('forward_generation requires complete generation labels')
            labels=metadata[field].to_numpy();order=cfg.get('generation_order')
            if order is None:
                if not pd.api.types.is_numeric_dtype(metadata[field]): raise ValueError('non-numeric generations require explicit generation_order')
                order=sorted(pd.unique(labels))
            if set(order)!=set(pd.unique(labels)) or len(order)!=len(set(order)) or len(order)<3: raise ValueError('generation_order must uniquely cover >=3 observed generations')
            for fold in range(1,len(order)):
                train=base & np.isin(labels,order[:fold])[:,None];test=base & (labels==order[fold])[:,None]
                commit(design,fold-1,train,test,block=field,role='outer')
        elif design=='unseen_project_same_trait':
            field=cfg.get('project_column','project')
            if field not in metadata: raise ValueError('unseen_project_same_trait requires sample project/cohort metadata')
            groups=aligned_labels(metadata[field],traits.index,field)
            for fold,label in enumerate(pd.unique(groups)):
                rows=groups==label;train=base.copy();train[rows]=False
                commit(design,fold,train,base & rows[:,None],project=str(label),role='outer')
        elif design in {'few_shot_project','frozen_embedding_probe','frozen_embedding_association'}:
            if len(pd.unique(projects))<2: raise ValueError(f'{design} requires >=2 trait projects')
            budgets=sorted(set(cfg.get('target_calibration_fractions',[.05,.10,.25])))
            if not budgets or any(not 0<x<=1 for x in budgets): raise ValueError('calibration fractions must be in (0,1]')
            if design=='frozen_embedding_association':budgets=[None]
            targets=cfg.get('target_projects',pd.unique(projects).tolist())
            if not set(map(str,targets))<=set(projects): raise ValueError('unknown target project')
            for project in map(str,targets):
                target=projects==project;eligible=base[:,target].any(1)
                for block,splits in [(None,default_splits)]+list(block_splits.items()):
                    for fold,rows in enumerate(splits):
                        test=base & rows[:,None] & target[None,:]
                        allowed=np.flatnonzero(eligible & ~rows);permutation=rng.permutation(allowed)
                        for budget in budgets:
                            count=0 if budget is None else max(1,int(np.ceil(budget*len(permutation))));cal=permutation[:count]
                            counts=base[cal][:,target].sum(0)
                            if budget is not None and np.any(counts<min_cal):
                                raise ValueError(f'{design} target={project} budget={budget} fold={fold}: fewer than min_calibration={min_cal} per target trait; increase budget explicitly')
                            train=base.copy();train[rows]=False;train[:,target]=False
                            train[np.ix_(cal,np.flatnonzero(target))]=base[np.ix_(cal,np.flatnonzero(target))]
                            variants=[('all_donors',np.arange(t))]
                            if design=='few_shot_project' and 'donor_ablation' in designs:
                                variants += [('target_only',np.flatnonzero(target))]
                                variants += [(f'donor:{donor}',np.flatnonzero(target|(projects==donor))) for donor in pd.unique(projects) if donor!=project]
                            for comparison,columns in variants:
                                role='objective' if comparison=='all_donors' else 'paired_diagnostic'
                                extra={'calibration_rows':cal.tolist()}
                                if design.startswith('frozen_embedding'):
                                    # The source representation never sees target trait outcomes.
                                    train[:,target]=False;columns=np.flatnonzero(~target);role='representation'
                                    extra.update(target_columns=np.flatnonzero(target).tolist(),
                                                 association_only=design=='frozen_embedding_association')
                                commit(design,fold,train,test,project=project,budget=budget,comparison=comparison,
                                       block=block,role=role,columns=columns,extra=extra)
                            if design=='few_shot_project' and cfg.get('disable_cross_project_baseline',True):
                                commit(design,fold,train,test,project=project,budget=budget,
                                       comparison='no_cross_project_G',block=block,role='paired_diagnostic',extra={'disable_cross_project_G':True})
        else: raise ValueError(f'unsupported design {design}')
    objective_designs=list(dict.fromkeys(s['design'] for s in schedule if s['role']=='objective'))
    if not objective_designs and not allow_outer_only: raise ValueError('no inner objective scenarios; outer/representation designs cannot select hyperparameters')
    weights=dict(cfg.get('weights',{x:1/len(objective_designs) for x in objective_designs}))
    if set(weights)!=set(objective_designs) or any(not np.isfinite(x) or x<=0 for x in weights.values()): raise ValueError('weights must be finite positive and cover exactly the inner objective designs')
    total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
    group_field=cfg.get('bootstrap_group','family' if 'family' in metadata else 'generation' if 'generation' in metadata else None)
    units=aligned_labels(metadata[group_field],traits.index,group_field) if group_field else traits.index.astype(str).to_numpy()
    covariates=cfg.get('covariates')
    if covariates is None: design=np.ones((n,1));cov_names=['intercept']
    else:
        if not isinstance(covariates,pd.DataFrame) or covariates.index.has_duplicates: raise TypeError('cv.covariates must be a uniquely indexed DataFrame')
        covariates=covariates.reindex(traits.index)
        if not np.isfinite(covariates.to_numpy(float)).all(): raise ValueError('covariates must be finite/sample aligned')
        design=np.column_stack([np.ones(n),covariates.to_numpy(float)]);cov_names=['intercept']+list(map(str,covariates.columns))
    manifest={'seed':int(seed),'inner_folds':folds,'weights':weights,'average':'scenario_then_trait_then_project_then_regime',
              'robustness_blocks':robustness,'family_robustness_available':bool(family_available),
              'family_robustness_opted_out':bool(family_available and 'robustness_blocks' in cfg and 'family' not in robustness),
              'scenarios':schedule,'exclusions':exclusions,'trait_projects':projects.tolist(),
              'bootstrap_unit':group_field or 'animal','covariate_names':cov_names,
              'genotypes':'transductive; supplied kernel fixed across folds, no outcome-dependent genomic preprocessing',
              'inner_intervals':'conditional on this search and these masks; not untouched outer-test evidence'}
    manifest['project_addition_sensitivity'] = ('project_addition_sensitivity' in designs)
    manifest['connectivity_stratified'] = ('connectivity_stratified' in designs)
    np.save(directory/'design.npy',design);np.save(directory/'units.npy',units.astype(str));np.save(directory/'projects.npy',projects)
    atomic_json(directory/'manifest.json',manifest)
    return manifest


def load_masks(scenario):
    """Read one packed mask pair; returns train,test,column indices; O(n*t)."""
    with np.load(scenario['mask_file'],allow_pickle=False) as z:
        shape=tuple(z['shape']);count=int(np.prod(shape))
        return (np.unpackbits(z['train'],count=count).reshape(shape).astype(bool),
                np.unpackbits(z['test'],count=count).reshape(shape).astype(bool),z['columns'])


def preprocess(values, train_mask, design):
    """Fit nuisance coefficients/scales only on training cells; O(n*t*p^2).

    Returns residual phenotype frame values, fixed predictions, coefficients,
    training-only standard deviations and design ranks. Zero variance fails.
    """
    residual=np.full(values.shape,np.nan);fixed=np.empty_like(values);coeff=[];scales=[];ranks=[]
    for j in range(values.shape[1]):
        use=train_mask[:,j];x=design[use];y=values[use,j]
        beta,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
        if len(y)-rank<2: raise ValueError('insufficient residual training observations')
        fixed[:,j]=design@beta;residual[use,j]=y-fixed[use,j]
        scale=float(np.std(y,ddof=0))
        if not np.isfinite(scale) or scale<=1e-12: raise ValueError('constant training trait cannot define NMSE')
        coeff.append(beta);scales.append(scale);ranks.append(int(rank))
    return residual,fixed,np.asarray(coeff).T,np.asarray(scales),ranks


def score_predictions(values, predictions, fixed, scales, test, scenario, projects, units, traits, *, training_means=None):
    """Return per-trait metrics and cluster loss sums; O(held-out cells).

    Correlation/calibration are secondary, null when undefined. genetic-only
    predictions are selected by the caller for unseen regimes. No predictive
    interval coverage is fabricated from posterior means.
    """
    rows=[];groups=[]
    for j in np.flatnonzero(test.any(0)):
        use=test[:,j];truth=values[use,j];pred=predictions[use,j];baseline=fixed[use,j]
        if not np.isfinite(pred).all(): raise FloatingPointError('non-finite validation prediction')
        error=pred-truth;sq=(error/scales[j])**2
        reference=baseline if training_means is None else training_means[j]
        den=float(np.sum((truth-reference)**2));nuisance_den=float(np.sum((truth-baseline)**2))
        corr=None if np.std(truth)==0 or np.std(pred)==0 else float(np.corrcoef(truth,pred)[0,1])
        slope=None;intercept=None
        if np.std(pred)>0:
            b=np.linalg.lstsq(np.column_stack([np.ones(len(pred)),pred]),truth,rcond=None)[0];intercept=float(b[0]);slope=float(b[1])
        row={'scenario':scenario['id'],'regime':scenario['design'],'project':str(projects[j]),'trait':str(traits[j]),
             'budget':scenario.get('budget'),'comparison':scenario['comparison'],'role':scenario['role'],
             'fold':scenario['fold'],'n':int(use.sum()),'nmse':float(sq.mean()),'rmse':float(np.sqrt(np.mean(error**2))),
             'mae':float(np.mean(abs(error))),'correlation':corr,'baseline_r2':None if den==0 else float(1-np.sum(error**2)/den),
             'baseline_kind':'training_trait_mean' if training_means is not None else 'fixed_training_baseline',
             'nuisance_baseline_r2':None if nuisance_den==0 else float(1-np.sum(error**2)/nuisance_den),
             'calibration_intercept':intercept,'calibration_slope':slope}
        rows.append(row)
        for unit in pd.unique(units[use]):
            active=units[use]==unit
            groups.append({'scenario':scenario['id'],'regime':scenario['design'],'project':str(projects[j]),
                           'trait':str(traits[j]),'comparison':scenario['comparison'],'role':scenario['role'],
                           'unit':str(unit),'n':int(active.sum()),'sum_sq':float(sq[active].sum())})
    return rows,groups


def aggregate(rows, weights):
    """Balance scenarios/budgets, traits, projects, then regimes; O(rows).

    Every required regime must be present. Extra paired/outer diagnostics do
    not enter the primary objective and cannot improve it by being omitted.
    """
    active=[r for r in rows if r['role']=='objective' and r['comparison']=='all_donors']
    trait=defaultdict(list)
    for r in active: trait[(r['regime'],r['project'],r['trait'])].append(r['nmse'])
    project=defaultdict(list)
    for (regime,label,_),loss in trait.items(): project[(regime,label)].append(float(np.mean(loss)))
    regime=defaultdict(list)
    project_scores={}
    for (name,label),loss in project.items():
        value=float(np.mean(loss));regime[name].append(value);project_scores[name+':'+label]=value
    if set(regime)!=set(weights): raise ValueError('incomplete regime coverage is ineligible')
    scores={name:float(np.mean(loss)) for name,loss in regime.items()}
    score=sum(weights[k]*scores[k] for k in weights)
    if not np.isfinite(score): raise FloatingPointError('non-finite compound score')
    return {'score':float(score),'by_regime':scores,'by_project':project_scores}


def paired_cluster_interval(candidate, incumbent, weights, *, seed=0, n_boot=200):
    """Paired family/cohort/animal bootstrap, never independent-cell resampling.

    Inputs are group loss sums from identical scenarios. Resamples whole units
    consistently across traits/scenarios and recomputes macro aggregation.
    Replicates omitting a required stratum are rejected with a finite attempt
    cap. Returns CI/null plus reason; O(n_boot * grouped held-out records).
    """
    keys=lambda r:(r['scenario'],r['trait'],r['unit'],r['comparison'])
    a={keys(r):r for r in candidate if r['role']=='objective'}
    b={keys(r):r for r in incumbent if r['role']=='objective'}
    if set(a)!=set(b): raise ValueError('paired intervals require identical cell/group coverage')
    units=sorted({k[2] for k in a})
    if len(units)<3:return {'ci95':None,'reason':'fewer than three independent resampling units'}
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(n_boot*10):
        counts=dict(zip(*np.unique(rng.choice(units,len(units),replace=True),return_counts=True)))
        summaries=[defaultdict(lambda:[0.,0,{}]),defaultdict(lambda:[0.,0,{}])]
        for dest,source in zip(summaries,[a,b]):
            for key,r in source.items():
                c=counts.get(key[2],0);entry=dest[(r['scenario'],r['trait'])]
                entry[0]+=c*r['sum_sq'];entry[1]+=c*r['n'];entry[2]=r
        try:
            scores=[]
            for summary in summaries:
                if any(n==0 for _,n,_ in summary.values()):raise ValueError('empty bootstrap stratum')
                rows=[dict(r,nmse=s/n) for s,n,r in summary.values()]
                scores.append(aggregate(rows,weights)['score'])
            draws.append(scores[0]-scores[1])
        except ValueError:pass
        if len(draws)==n_boot:break
    if len(draws)<n_boot:return {'ci95':None,'reason':'insufficient complete cluster-bootstrap replicates'}
    return {'ci95':np.quantile(draws,[.025,.975]).tolist(),'n_boot':n_boot,'units':len(units),
            'interpretation':'conditional inner-selection paired loss difference; not an outer-test CI'}


def transfer_report(evaluated,manifest,*,seed=0,n_boot=200):
    """Summarize paired donor gains and learning curves on identical test masks.

    evaluated contains rows/groups from a full verification schedule. Reports
    target-only, single-donor and genetic-sharing-disabled comparisons against
    all donors. Intervals resample the same units across budgets/traits.
    Complexity O(n_boot * relevant grouped records), conditional on selection.
    """
    rows=evaluated['rows'];groups=evaluated['groups'];scenarios={s['id']:s for s in manifest['scenarios']}
    output=[]
    projects=sorted({r['project'] for r in rows if r['regime']=='few_shot_project'})
    for project in projects:
        available=[r for r in rows if r['regime']=='few_shot_project' and r['project']==project and r['role'] in {'objective','paired_diagnostic'}]
        budgets=sorted({r['budget'] for r in available})
        for budget in budgets:
            comparators=sorted({r['comparison'] for r in available if r['comparison']!='all_donors'})
            for comparison in comparators:
                def select(comparator,collection):
                    """Map paired scenario IDs to the same frozen target split."""
                    result=[]
                    for r in collection:
                        if r['regime']!='few_shot_project' or r['project']!=project or r['comparison']!=comparator or r['role'] not in {'objective','paired_diagnostic'}:continue
                        scenario=scenarios[r['scenario']]
                        if scenario.get('budget')!=budget:continue
                        paired_id=json.dumps([scenario['fold'],scenario.get('block'),scenario.get('budget'),scenario.get('project')])
                        result.append(dict(r,scenario=paired_id,comparison='all_donors',role='objective'))
                    return result
                a=select('all_donors',rows);b=select(comparison,rows)
                if not a or not b:raise ValueError('missing prescribed paired transfer comparison')
                weights={'few_shot_project':1.}
                all_loss=aggregate(a,weights)['score'];base_loss=aggregate(b,weights)['score']
                interval=paired_cluster_interval(select('all_donors',groups),select(comparison,groups),weights,seed=seed,n_boot=n_boot)
                ci=None if interval['ci95'] is None else [-interval['ci95'][1],-interval['ci95'][0]]
                output.append({'project':project,'calibration_fraction':budget,'comparator':comparison,
                    'comparator_nmse':base_loss,'all_donor_nmse':all_loss,'donor_gain_nmse':base_loss-all_loss,
                    'relative_gain':(base_loss-all_loss)/max(abs(base_loss),1e-12),'donor_gain_ci95':ci,
                    'bootstrap_unit':manifest['bootstrap_unit'],'uncertainty':interval})
    return output
