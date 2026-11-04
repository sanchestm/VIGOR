"""Conditional TPE search over supported covariance/prediction parameters."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numbers

import numpy as np

STATISTICAL = {'rank_G','rank_E','factor_structure_G','factor_structure_E',
               'factor_lambda_G','factor_lambda_E','factor_group_lambda_G','factor_group_lambda_E',
               'factor_auto_fraction','factor_adaptive','factor_relaxed_refit',
               'factor_rotation_G','factor_rotation_E','shrink_G','shrink_E','E_structure',
               'min_E_pair_overlap','ridge_G','ridge_E','variance_closure','variance_closure_strength',
               'h2_prior_strength','h2_prior_groups'}
GEOMETRIES={'fa','pca','sparse_fa','sparse_pca'}
# Post-fit prediction controls: they never change the covariance fit, so the
# controller evaluates them in a cheap inner loop on each cached fit instead of
# spending TPE trials on them (unless the user puts them in search_space).
PREDICTION_ONLY={'shrink_G','shrink_E','E_structure','min_E_pair_overlap'}
# Residual (E) shrinkage chosen per candidate on its cached fit, strongest first:
# per-pair overlap/(overlap+s) for s = 100, 400 (the 3.1.3 'auto') and 1600; global
# off-diagonal retention 0.25/0.5/0.75; the data-driven per-pair Fisher-z EB without
# rank preservation; the closed-form Schafer-Strimmer linear intensity; the model-based
# per-pair EB (sampling variances from the fit's MoM precision) with multiplier
# c in {0.5, 1, 2, 4}; and unshrunk.
# Badly conditioned variants are skipped before prediction and variants that never
# compete are dropped after the start-up trials (see hyperopt.py).
DEFAULT_PREDICTION_GRID=[{'shrink_G':'auto','shrink_E':'auto@100'},
                         {'shrink_G':'auto','shrink_E':'auto'},
                         {'shrink_G':'auto','shrink_E':'auto@1600'},
                         {'shrink_G':'auto','shrink_E':.25},{'shrink_G':'auto','shrink_E':.5},
                         {'shrink_G':'auto','shrink_E':.75},
                         {'shrink_G':'auto','shrink_E':'fisher_eb_full'},
                         {'shrink_G':'auto','shrink_E':'linear_analytic'},
                         {'shrink_G':'auto','shrink_E':'model_eb@0.5'},{'shrink_G':'auto','shrink_E':'model_eb'},
                         {'shrink_G':'auto','shrink_E':'model_eb@2'},{'shrink_G':'auto','shrink_E':'model_eb@4'},
                         {'shrink_G':1.0,'shrink_E':1.0}]
EXTENDED={
    'rank_G':[4,8,16,32,64,100,160,'full'],'rank_E':[4,8,16,32,64,100,160,'full'],
    'factor_structure_G':['fa','sparse_fa'],'factor_structure_E':['fa','sparse_fa'],
    'factor_lambda_G':['auto'],'factor_lambda_E':['auto'],
    'factor_group_lambda_G':[0.,'auto'],'factor_group_lambda_E':[0.,'auto'],
    'factor_auto_fraction':{'low':.002,'high':.10,'log':True},
    'factor_adaptive':[True,False],'factor_relaxed_refit':[True],
    'factor_rotation_G':[None,'varimax'],'factor_rotation_E':[None,'varimax'],
    'shrink_G':['auto',1.0,{'low':.25,'high':1.}],
    'shrink_E':[.5263454565339929,'auto',{'low':0.,'high':1.}],
    'E_structure':['auto','fa','full','auto_supported'],
    'min_E_pair_overlap':[2,4,8,16,32,64],
    'ridge_G':{'low':1e-8,'high':.01,'log':True,'zero':True},
    'ridge_E':{'low':1e-8,'high':.01,'log':True,'zero':True},
    'variance_closure':['auto','off'],'variance_closure_strength':{'low':.1,'high':10.,'log':True},
    'h2_prior_strength':[0.,.5,1.,2.],'h2_prior_groups':['global','project'],
}
# Compact default: few dimensions for tens of trials. Ranks are integer log
# ranges (TPE learns their ordering); everything else is fixed at the user's
# genetic_varcov_PSD arguments unless named in search_space.
COMPACT_KEYS=('rank_G','rank_E','ridge_G','ridge_E','factor_auto_fraction','h2_prior_strength','h2_prior_groups')


def rank_choices(value, n_traits):
    """Normalize a rank grid or an integer rank range.

    value is a nonempty sequence of ints >=0 or 'full' (categorical grid), or
    {'low','high'[,'log']} integers (an ordered range that TPE models as one
    integer; draws >= n_traits mean 'full'). O(grid length).
    Zero explicitly means diagonal FA; full is encoded consistently as 'full'.
    """
    if isinstance(value,Mapping) and 'low' in value:
        if set(value)-{'low','high','log'} or 'high' not in value:raise ValueError('rank ranges use low/high[/log]')
        low=value['low'];high=n_traits if value['high']=='full' else value['high']
        for bound in (low,high):
            if isinstance(bound,(bool,np.bool_)) or not isinstance(bound,numbers.Integral) or bound<0:raise ValueError('rank range bounds must be integers >=0 or full')
        low=int(low);high=min(int(high),int(n_traits))
        if high<low or (value.get('log') and low<1):raise ValueError('invalid rank range')
        return {'low':low,'high':high,'log':bool(value.get('log',False)),'rank_range':True,'n_traits':int(n_traits)}
    if not isinstance(value,(tuple,list)) or not value:raise ValueError('rank grids must be nonempty sequences')
    result=[]
    for rank in value:
        if rank=='full':effective='full'
        elif isinstance(rank,numbers.Integral) and not isinstance(rank,(bool,np.bool_)) and rank>=0:
            effective='full' if rank>=n_traits else int(rank)
        else:raise ValueError('ranks must be integers >=0 or full')
        if effective not in result:result.append(effective)
    return result


def prepare_space(space, n_traits, kernel_names, preset='compact', fixed=None):
    """Validate a supported search space; return normalized descriptors.

    preset 'compact' (default) searches integer log-range ranks
    {'low': 2, 'high': n_traits, 'log': True}, ridge_G/E, factor_auto_fraction
    and the h2 prior; 'extended' searches the full historical space. Keys in
    space override/extend the preset. Searched keys appear in the result; the
    rest are fixed at ``fixed`` (the user's fit arguments), which the decoder
    reads for conditional choices. Numeric descriptors use {low,high,log[,zero]};
    categorical descriptors are nonempty lists. Named G ranks/geometry accept
    component->descriptor maps.
    """
    provided=dict(space or {});unknown=set(provided)-STATISTICAL
    if unknown:raise TypeError(f'unsupported statistical search options: {sorted(unknown)}')
    if preset not in {'compact','extended'}:raise ValueError("search_preset must be 'compact' or 'extended'")
    defaults=dict(EXTENDED)
    if preset=='compact':
        defaults={k:EXTENDED[k] for k in COMPACT_KEYS}
        defaults['rank_G']={'low':2,'high':'full','log':True};defaults['rank_E']={'low':2,'high':'full','log':True}
    defaults.update(provided)
    for name in ('rank_G','rank_E'):
        value=defaults[name]
        if isinstance(value,Mapping) and 'low' not in value:
            if name!='rank_G' or set(value)!=set(kernel_names):raise ValueError('named rank_G grid must cover all kernel names')
            defaults[name]={k:rank_choices(v,n_traits) for k,v in value.items()}
        else:defaults[name]=rank_choices(value,n_traits)
    def validate(value,name):
        """Reject malformed descriptor values before starting the study."""
        if isinstance(value,Mapping) and value.get('rank_range'):return
        if isinstance(value,Mapping):
            if 'low' not in value:
                if name not in {'rank_G','factor_structure_G','factor_lambda_G','factor_group_lambda_G','ridge_G','shrink_G','factor_rotation_G'} or set(value)!=set(kernel_names):raise ValueError(f'unsupported component mapping for {name}')
                for child in value.values():validate(child,name)
                return
            if set(value)-{'low','high','log','zero','step'} or 'high' not in value:raise ValueError(f'bad numeric descriptor {name}')
            low,high=float(value['low']),float(value['high'])
            if not np.isfinite([low,high]).all() or high<low or (value.get('log') and low<=0):raise ValueError(f'invalid numeric range {name}')
        elif isinstance(value,(tuple,list)) and value:
            for item in value:
                if isinstance(item,Mapping):validate(item,name)
                elif item is not None and not isinstance(item,(str,int,float,bool)):raise ValueError(f'invalid choice for {name}')
                elif isinstance(item,float) and not math.isfinite(item):raise ValueError(f'nonfinite choice for {name}')
        else:raise ValueError(f'{name} needs a list or numeric descriptor')
    for name,value in defaults.items():validate(value,name)
    def leaves(value):
        """Yield numeric/string leaves from conditional descriptors."""
        if isinstance(value,Mapping):
            if 'low' in value:yield value['low'];yield value['high']
            else:
                for child in value.values():yield from leaves(child)
        elif isinstance(value,(list,tuple)):
            for child in value:yield from leaves(child)
        else:yield value
    for name,value in defaults.items():
        for leaf in leaves(value):
            if name in {'factor_adaptive','factor_relaxed_refit'} and not isinstance(leaf,bool):raise ValueError(f'{name} choices must be bool')
            if name=='min_E_pair_overlap' and (isinstance(leaf,bool) or not isinstance(leaf,numbers.Integral) or leaf<1):raise ValueError('overlap choices must be positive integers')
            if name in {'shrink_G','shrink_E'}:
                allowed={'auto','legacy_auto','analytic_global','analytic_pairwise'}|({'pairwise','pairwise_full','fisher_eb_full','eb','linear_analytic','model_eb'} if name=='shrink_E' else set())
                if isinstance(leaf,str) and leaf.split('@')[0] not in allowed or not isinstance(leaf,str) and (isinstance(leaf,bool) or not isinstance(leaf,numbers.Real) or not 0<=leaf<=1):raise ValueError(f'invalid prediction retention {leaf}')
            if name in {'ridge_G','ridge_E','factor_lambda_G','factor_lambda_E','factor_group_lambda_G','factor_group_lambda_E','variance_closure_strength','h2_prior_strength','factor_auto_fraction'}:
                if leaf is None and name.startswith('factor_lambda_'):continue
                if leaf=='auto' and name.startswith(('factor_lambda_','factor_group_lambda_')):continue
                if isinstance(leaf,bool) or not isinstance(leaf,numbers.Real) or leaf<0 or name=='factor_auto_fraction' and leaf<=0:raise ValueError(f'{name} requires supported nonnegative numerical controls')
            if name.startswith('factor_rotation_') and leaf not in {None,'varimax'}:raise ValueError('rotation choices must be None/varimax')
    if 'variance_closure' in defaults and set(defaults['variance_closure'])-{'auto','off'}:raise ValueError('closure choices must be auto/off')
    for name in ('factor_structure_G','factor_structure_E'):
        if name not in defaults:continue
        values=defaults[name].values() if isinstance(defaults[name],Mapping) else [defaults[name]]
        if any(set(v)-GEOMETRIES for v in values):raise ValueError(f'invalid geometry choices for {name}')
    if 'h2_prior_groups' in defaults and set(defaults['h2_prior_groups'])-{'global','project'}:raise ValueError("h2_prior_groups choices are 'global'/'project'")
    defaults['__fixed']={k:v for k,v in dict(fixed or {}).items() if k in STATISTICAL and k not in defaults}
    return defaults


def rank_value(draw, descriptor):
    """Map an integer rank draw to the fit argument ('full' at or above the trait count)."""
    return 'full' if int(draw)>=descriptor['n_traits'] else int(draw)


def suggest(trial, space):
    """Decode one conditional Optuna trial into active model controls.

    trial exposes suggest_categorical/float/int. space is prepare_space output.
    Inactive sparse controls and the prior domain at zero prior strength are
    fixed, not suggested. Numeric zero has its own branch, not a log(0)
    endpoint. Space/memory O(active parameter count).
    """
    fixed=space.get('__fixed',{})
    def draw(name,descriptor):
        """Draw one fixed-distribution categorical, numeric or component choice."""
        if isinstance(descriptor,Mapping) and descriptor.get('rank_range'):
            low,high=descriptor['low'],descriptor['high']
            return rank_value(low if low==high else trial.suggest_int(name,low,high,log=descriptor['log']),descriptor)
        if isinstance(descriptor,Mapping) and 'low' not in descriptor:
            return {key:draw(name+'::'+str(key),value) for key,value in descriptor.items()}
        if isinstance(descriptor,Mapping):
            if descriptor.get('zero') and trial.suggest_categorical(name+'__zero',[True,False]):return 0.
            low,high=descriptor['low'],descriptor['high']
            if low==high:return low
            if 'min_E_pair_overlap' in name:
                return trial.suggest_int(name,int(low),int(high),log=descriptor.get('log',False))
            return trial.suggest_float(name,float(low),float(high),log=descriptor.get('log',False),
                                       **({'step':descriptor['step']} if 'step' in descriptor else {}))
        choices=list(descriptor)
        ranges=[v for v in choices if isinstance(v,Mapping)]
        if ranges:
            options=list(range(len(choices)));index=options[0] if len(options)==1 else trial.suggest_categorical(name+'__branch',options)
            value=choices[index]
            return draw(name+'__numeric_'+str(index),value) if isinstance(value,Mapping) else value
        return choices[0] if len(choices)==1 else trial.suggest_categorical(name,choices)
    def take(name,default=None):
        """Searched keys are drawn; the others keep the user's fixed argument."""
        if name in space:return draw(name,space[name])
        return fixed.get(name,default)
    result={}
    for k,default in [('rank_G',None),('rank_E',None),('factor_structure_G','fa'),('factor_structure_E','fa'),
                      ('ridge_G',0.),('ridge_E',0.),('variance_closure','auto'),('h2_prior_strength',0.)]:
        value=take(k,default)
        if k in space or k in fixed:result[k]=value
    # The prior domain is only meaningful when the prior is active.
    strength=result.get('h2_prior_strength',0.) or 0.
    result['h2_prior_groups']=take('h2_prior_groups','global') if strength>0 else 'global'
    # Prediction-only controls are searched only when named; otherwise the
    # controller's inner prediction grid chooses them on the cached fit.
    for name in ('shrink_G','shrink_E','E_structure'):
        if name in space:result[name]=draw(name,space[name])
    result.update(covariance_structure='fa',rank_selection='none')
    e_structure=result.get('E_structure',fixed.get('E_structure','auto'))
    if 'min_E_pair_overlap' in space:
        result['min_E_pair_overlap']=draw('min_E_pair_overlap',space['min_E_pair_overlap']) if e_structure in {'auto','auto_supported'} else 2
    if e_structure=='auto_supported' and result.get('shrink_E') in {'pairwise','pairwise_full','fisher_eb_full'}:
        raise ValueError('auto_supported requires scalar/auto residual retention')
    any_sparse=False;any_auto=False
    for suffix in ['G','E']:
        structure=result.get('factor_structure_'+suffix,fixed.get('factor_structure_'+suffix,'fa')) or 'fa'
        sparse=any('sparse' in str(v) for v in structure.values()) if isinstance(structure,Mapping) else 'sparse' in str(structure)
        any_sparse |= sparse
        for prefix,default in [('factor_lambda_',None),('factor_group_lambda_',0.),('factor_rotation_',None)]:
            key=prefix+suffix
            if key not in space:
                value=fixed.get(key,default)
            else:
                descriptor=space[key]
                if suffix=='G' and isinstance(structure,Mapping) and isinstance(descriptor,Mapping) and 'low' not in descriptor:
                    value={name:draw(key+'::'+name,descriptor[name]) if 'sparse' in kind else default for name,kind in structure.items()}
                else:value=draw(key,descriptor) if sparse else default
                result[key]=value
            any_auto |= sparse and (any(v=='auto' for v in value.values()) if isinstance(value,Mapping) else value=='auto')
    if 'factor_auto_fraction' in space and any_auto:result['factor_auto_fraction']=draw('factor_auto_fraction',space['factor_auto_fraction'])
    if 'factor_adaptive' in space and any_sparse:result['factor_adaptive']=draw('factor_adaptive',space['factor_adaptive'])
    if 'factor_relaxed_refit' in space and any_sparse:result['factor_relaxed_refit']=draw('factor_relaxed_refit',space['factor_relaxed_refit'])
    if 'variance_closure_strength' in space and result.get('variance_closure',fixed.get('variance_closure','auto'))!='off':
        result['variance_closure_strength']=draw('variance_closure_strength',space['variance_closure_strength'])
    return result


def encode(config, space):
    """Optuna parameter values that make suggest() reproduce config, for enqueue_trial.

    Only searched keys are encoded; values outside the space are skipped, so a
    seeded trial falls back to TPE for those parameters. O(searched keys).
    """
    params={}
    def put(name,descriptor,value):
        if isinstance(descriptor,Mapping) and descriptor.get('rank_range'):
            draw=descriptor['high'] if value=='full' else value
            if isinstance(draw,numbers.Integral) and descriptor['low']<=draw<=descriptor['high'] and descriptor['low']!=descriptor['high']:
                params[name]=int(draw)
            return
        if isinstance(descriptor,Mapping) and 'low' not in descriptor:
            if isinstance(value,Mapping):
                for key,child in descriptor.items():
                    if key in value:put(name+'::'+str(key),child,value[key])
            return
        if isinstance(descriptor,Mapping):
            if descriptor.get('zero'):
                params[name+'__zero']=bool(value==0)
                if value==0:return
            if descriptor['low']==descriptor['high'] or isinstance(value,bool) or not isinstance(value,numbers.Real):return
            if descriptor['low']<=value<=descriptor['high']:
                params[name]=int(value) if 'min_E_pair_overlap' in name else float(value)
            return
        choices=list(descriptor)
        ranges=[i for i,v in enumerate(choices) if isinstance(v,Mapping)]
        if not ranges:
            if len(choices)>1 and value in choices:params[name]=value
            return
        for index,choice in enumerate(choices):
            if not isinstance(choice,Mapping) and choice==value:
                params[name+'__branch']=index;return
        for index in ranges:
            child=choices[index]
            if isinstance(value,numbers.Real) and not isinstance(value,bool) and child['low']<=value<=child['high']:
                params[name+'__branch']=index;put(name+'__numeric_'+str(index),child,value);return
    for key,descriptor in space.items():
        if key.startswith('__') or key not in config:continue
        put(key,descriptor,config[key])
    return params
