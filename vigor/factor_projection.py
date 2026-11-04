"""Within-animal factor projections and their independent persisted state.

These projections are used by phenome analysis, not by exact factor GWAS.
"""
import numpy as np
import pandas as pd

def _metadata_safe(x):
    """Recursively convert result metadata to equality/JSON-safe Python values.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x.item()``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.
    """
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,bytes): return x.hex()
    if isinstance(x,pd.DataFrame):
        return {"index":list(map(str,x.index)),"columns":list(map(str,x.columns)),"values":x.to_numpy().tolist()}
    if isinstance(x,pd.Series): return {"index":list(map(str,x.index)),"values":x.to_list()}
    if isinstance(x,dict): return {str(_metadata_safe(k)):_metadata_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_metadata_safe(v) for v in x]
    if x is None or isinstance(x,(str,int,float,bool)): return x
    return str(x)


def _score_pattern_metadata(patterns, mode='summary'):
    """Evaluate pattern metadata.

    Parameters
    ----------
    patterns : str, sequence or mapping
        Pattern/filter selectors applied to the resource/table search. Use patterns matching the
        required filenames/header or existing metadata fields.

    mode : str
        Input for the operation ``mode == 'full'``. See
        ``vigor.factor_gwas_api._score_pattern_metadata`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'summary'``.
    """
    if mode == 'full':
        return patterns
    if mode != 'summary':
        raise ValueError("score_pattern_metadata must be 'summary' or 'full'")
    counts = {}
    for value in patterns.values():
        observed = str(value['n_observed'])
        counts[observed] = counts.get(observed, 0) + int(value['n_rows'])
    return {'n_patterns': len(patterns), 'n_scored_rows': sum(counts.values()),
            'rows_by_observed_traits': counts,
            'detail': 'summary; request score_pattern_metadata=full for individual masks'}


def _write_factor_state(path, chromosome, traits, scores, loadings, specific_var, noise_cov, score_method):
    """Persist the actual LOCO posterior targets for downstream locus analysis.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Passed to ``xr.Dataset`` as ``attrs``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Passed to ``xr.Dataset`` as ``coords``.

    scores : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body. Passed to ``xr.Dataset`` as ``coords``.

    loadings : array-like or fitted basis
        Trait/feature × retained-factor loading matrix or prepared basis. Column order
        identifies factor directions and must agree with the scores used for projection.

    specific_var : array-like
        Loading-support mask or nonnegative trait-specific diagonal variance, as named. Masks
        match trait × factor loadings; specific variances have one value per trait and augment
        the low-rank covariance.

    noise_cov : float or array-like
        Variance/scale or trait covariance specified for the selected prior/simulation
        component. Scalar scales must be nonnegative (positive where a prior requires it);
        matrices match the trait axis and must be positive semidefinite.

    score_method : str
        Projection provenance string written as metadata. These targets are independent
        exploratory outputs and are not the original-trait exact-score null outcomes.
    """
    import xarray as xr
    ds = xr.Dataset(
        {'scores': (('sample', 'factor'), scores.to_numpy(np.float64)),
         'loadings': (('trait', 'factor'), np.asarray(loadings, np.float64)),
         'specific_variance': ('trait', np.asarray(specific_var, np.float32)),
         'noise_diagonal': ('trait', np.diag(noise_cov).astype(np.float32)),
         'n_observed': ('sample', traits.notna().sum(axis=1).to_numpy(np.int32))},
        coords={'sample': traits.index.astype(str).to_numpy(),
                'trait': traits.columns.astype(str).to_numpy(),
                'factor': scores.columns.astype(str).to_numpy()},
        attrs={'score_method': str(score_method), 'loco_group': str(chromosome),
               'targets': 'exploratory posterior projection targets; not outcomes in the exact association null',
               'noise_diagonal': 'summary of nuisance covariance, not an independence assumption'})
    encoding = {}
    for name, array in ds.data_vars.items():
        chunks = tuple(min(ds.sizes[d], 1024 if d == 'sample' else 256 if d == 'trait' else 32) for d in array.dims)
        encoding[name] = {'chunks': chunks, 'shards': tuple(2*c if ds.sizes[d] >= 2*c else c for d, c in zip(array.dims, chunks))}
    from .store import VigorStore
    VigorStore(path,mode='a').write_dataset(ds,f'factor_state/{chromosome}',mode='a',
        consolidated=False,encoding=encoding)


def _factor_scores_gls(traits,L,noise_cov,min_observed=None,method='gls',noise_spec=None):
    """Missing-pattern factor scores using GLS or Gaussian posterior means.

    ``method='posterior'`` assumes latent factors have unit prior covariance,
    consistent with C=L L' + Psi.  The posterior mean
        (I + L_o' C_o^-1 L_o)^-1 L_o' C_o^-1 y_o
    remains identified with a single observed trait and automatically shrinks
    low-information animals toward zero.  ``gls`` retains the historical
    unregularized score and normally requires at least ``rank(L)`` observations.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    L : array-like
        Numerical operand in ``L.shape[1]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    noise_cov : float or array-like
        Variance/scale or trait covariance specified for the selected prior/simulation
        component. Scalar scales must be nonnegative (positive where a prior requires it);
        matrices match the trait axis and must be positive semidefinite.

    min_observed : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'gls'``. Source-recognized
        literal values: ``bayes``, ``posterior``, ``regression``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    noise_spec : array-like or numeric scalar
        Input for the operation ``np.asarray(noise_spec['factor'], float)[obs]``. See
        ``vigor.factor_gwas_api._factor_scores_gls`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.
    """
    Y=traits.to_numpy(float) if isinstance(traits,pd.DataFrame) else np.asarray(traits,float);n,t=Y.shape;L=np.asarray(L,float);r=L.shape[1]
    mu=np.nanmean(Y,axis=0);Y=Y-mu;out=np.full((n,r),np.nan);info={};mode=str(method).lower()
    if mode not in {'gls','posterior','bayes','regression'}:raise ValueError("factor score method must be 'gls' or 'posterior'")
    min_obs=(max(r,1) if mode=='gls' else 1) if min_observed is None else int(min_observed);masks=np.isfinite(Y);I=np.eye(r)
    patterns,inverse=np.unique(masks,axis=0,return_inverse=True)
    order=np.argsort(inverse,kind='stable');bounds=np.r_[0,np.cumsum(np.bincount(inverse,minlength=len(patterns)))]
    for group,mask in enumerate(patterns):
        key=mask.tobytes();rows=order[bounds[group]:bounds[group+1]];obs=np.flatnonzero(mask)
        if len(obs)<min_obs:continue
        Lo=L[obs]
        if noise_spec is not None:
            U=np.asarray(noise_spec['factor'],float)[obs]
            d=np.asarray(noise_spec['diag'],float)[obs]
            if np.any(d<=0): raise ValueError('Woodbury factor noise requires positive specific variance')
            DinvL=Lo/d[:,None];DinvU=U/d[:,None]
            CinvL=DinvL-DinvU@np.linalg.solve(np.eye(U.shape[1])+U.T@DinvU,U.T@DinvL)
        else:
            C=np.asarray(noise_cov,float)[np.ix_(obs,obs)]
            CinvL=np.linalg.pinv(C,rcond=1e-10)@Lo
        J=Lo.T@CinvL
        if mode=='gls':
            H=np.linalg.pinv(J,rcond=1e-10)@CinvL.T;post=np.linalg.pinv(J,rcond=1e-10)
        else:
            post=np.linalg.pinv(I+J,rcond=1e-10);H=post@CinvL.T
        out[rows]=Y[np.ix_(rows,obs)]@H.T
        rel=np.clip(1.0-np.diag(post),0.0,1.0) if mode!='gls' else np.full(r,np.nan)
        info[key]={'n_rows':len(rows),'n_observed':len(obs),'method':mode,'posterior_reliability':rel.tolist()}
    cols=[f'factor_{i+1}' for i in range(r)];return pd.DataFrame(out,index=traits.index if isinstance(traits,pd.DataFrame) else None,columns=cols),info


