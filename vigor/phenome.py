"""Small, labelled helpers for large-phenome notebooks.

Opening a completed fit never starts an optimizer or compares its parameters
with a proposed new fit. Uncertainty, embeddings and discovery are explicit
operations. These helpers support a single additive genetic kernel plus E.
"""
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from . import multivariate as mv
from .progress import operation_progress, operation_stage
from .store import sharded_dataset_encoding, finalize_zarr_artifact


def phenotype_digest(traits):
    """Fingerprint a labelled phenotype frame, including row order and NaNs.
    
    Parameters
    ----------
    traits : pandas.DataFrame, shape (n, t)
        Numeric animal-by-trait observations. The index and stringified trait labels enter the digest; NaN patterns are retained. Arrays without columns are unsupported.
    
    Returns
    -------
    str
        64-character SHA256 hex digest of pandas row hashes plus ordered trait labels.
    
    Complexity
    ----------
    TIME O(nt); SPACE O(n+t) beyond the input for row hashes and labels. n is animals and t traits. This is a pandas-version-dependent fingerprint, not a cross-format file checksum.
    """
    h=hashlib.sha256(pd.util.hash_pandas_object(traits,index=True).values.tobytes())
    h.update(json.dumps(list(map(str,traits.columns))).encode())
    return h.hexdigest()


def save_analysis_dataset(dataset, path, *, overwrite=False):
    """Write an annotated xarray Dataset as consolidated sharded Zarr v3.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Variables and labelled coordinates to serialize; string-like coordinates are normalized to ordinary strings for codec compatibility.
    
    path : str or pathlib.Path
        Destination Zarr directory. Parent directories are created.
    
    overwrite : bool
        Default: ``False``. False raises FileExistsError for an existing destination; True recreates it with mode w.
    
    Returns
    -------
    pathlib.Path
        Destination store path after the write and metadata finalization complete.
    
    Complexity
    ----------
    TIME O(v+b); SPACE O(v+c), where v is total coordinate elements, b serialized array elements/bytes processed, and c the largest writer/codec buffer. Dask-backed data are computed during writing; native codec allocations are additional.
    """
    path=Path(path)
    if path.exists() and not overwrite: raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    ds=dataset.copy()
    # NumPy StringDType coordinates are not yet supported by every Zarr codec.
    for key,a in ds.coords.items():
        if a.dtype.kind in 'OTU': ds=ds.assign_coords({key:(a.dims,np.asarray(a.values,dtype=object).astype(str))})
    ds.to_zarr(path,mode='w',zarr_format=3,consolidated=False,encoding=sharded_dataset_encoding(ds))
    finalize_zarr_artifact(path,metadata=dict(ds.attrs))
    return path


_SAVED_MATRICES=('G','E','G_unshrunk','E_unshrunk','G_prediction','E_prediction','Rg','G_se','E_se','rg_se',
                 'rg_ci95_low','rg_ci95_high')
_SAVED_VECTORS=('h2','h2_se','G_variance_se','E_variance_se','cov_GE_diag','h2_ci_critical_value','h2_ci_bias_m2',
                'h2_prior_mean','h2_prior_tau2','h2_prior_weight','h2_prior_kappa','h2_pilot','h2_pilot_se')
_SAVED_INTERVALS=('h2_ci95','G_variance_ci95','E_variance_ci95')


def save_covariance_fit(fit, traits, path, *, parameters=None, overwrite=False):
    """Save a completed single-kernel covariance fit with native factors and phenotype provenance.
    
    Parameters
    ----------
    fit : xarray.Dataset, mapping or pandas.Series
        genetic_varcov_PSD result (Dataset, or a legacy dict/Series) with exactly one component_specs entry, E_spec, and G/E matrices (t,t). Native factors have shapes (t,r_G)/(t,r_E), and specific variances (t,). Available SEs, h2 and prediction profiles are retained.
    
    traits : pandas.DataFrame, shape (n, t)
        Original standardized fitting input, including identical labels/order/NaNs. Its digest is recorded; values are not stored.
    
    path : str or pathlib.Path
        Destination covariance Zarr v3 directory.
    
    parameters : dict or None
        Default: ``None``. Configuration/provenance metadata serialized as JSON; None records an empty object. This does not trigger a refit.
    
    overwrite : bool
        Default: ``False``. Refuse existing destination unless explicitly True.
    
    Returns
    -------
    pathlib.Path
        Completed consolidated, sharded Zarr v3 store (written through vigor.results.save_result), including inference matrices, prediction profiles, native factor support and phenotype digest.
    
    Complexity
    ----------
    TIME O(nt+t²+tR+b); SPACE O(t²+tR+n+c) beyond the supplied fit/traits. R=r_G+r_E, b is codec byte work and c the largest write buffer. Matrix/profile materialization is quadratic in trait count.
    
    Notes
    -----
    Raises ValueError for multiple genetic components. The saved digest checks phenotypes, not GRM/genotype provenance; callers must retain the kernel/source manifest separately.
    """
    fit=mv._result_view(fit); traits=mv._as_pandas_input(traits)
    if len(fit['component_specs'])!=1: raise ValueError('one genetic component is required')
    data={}
    for k in _SAVED_MATRICES:
        if fit.get(k) is not None: data[k]=( ('trait_i','trait_j'),np.asarray(fit[k]))
    specs={'G':next(iter(fit['component_specs'].values())),'E':fit['E_spec']}
    for component,spec in specs.items():
        dims=('trait_i','factor_'+component)
        data[component+'_loadings']=(dims,np.asarray(spec['factor']))
        data[component+'_specific']=('trait_i',np.asarray(spec['diag']))
        if spec.get('support_mask') is not None:
            data[component+'_support']=(dims,np.asarray(spec['support_mask'],bool))
    for k in _SAVED_VECTORS:
        if fit.get(k) is not None: data[k]=('trait_i',np.asarray(fit[k],float))
    for k in _SAVED_INTERVALS:
        if fit.get(k) is not None: data[k]=(('trait_i','bound'),np.asarray(fit[k],float))
    for profile in ('unseen_animals','partial_traits'):
        view=mv.prediction_covariance_view(fit,profile)
        for c in ('G','E'): data[profile+'_'+c]=( ('trait_i','trait_j'),np.asarray(view[c]))
    cols=traits.columns.astype(str)
    ds=xr.Dataset(data,coords={'trait_i':cols,'trait_j':cols,'bound':['lower','upper']},attrs={
        'h2_prior':json.dumps(fit.get('h2_prior') or {'active':False},default=str),
        'title':'Saved covariance fit','software':'vigor.phenome.save_covariance_fit',
        'schema':'vigor.phenome.covariance.v1','phenotype_digest':phenotype_digest(traits),
        'parameters':json.dumps(parameters or {},default=str),'uncertainty_method':str(fit.get('uncertainty_method','not computed')),
        'analytic_se_metadata':json.dumps({k:fit.get('analytic_se',{}).get(k) for k in (
            'method','kernel_backend','n_probes','mc_target_met','se_relative_mc_error_max',
            'common_trait_mask','kernel_factor','conditional_on_rank','conditional_on_sparse_support',
            'includes_support_selection_uncertainty')},default=str),
        'purpose':'completed covariance fit; inference and prediction are distinct'})
    from .results import save_result
    path=Path(path)
    if path.exists() and not overwrite: raise FileExistsError(path)
    save_result(ds,path,overwrite=True,artifact_type='vigor.phenome.covariance.v1')
    return path


def open_covariance_fit(path):
    """Read a completed covariance checkpoint without fitting or validating a new configuration.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Consolidated covariance Zarr directory from save_covariance_fit or compatible historical mega checkpoint. Trait axes must match and be unique.
    
    Returns
    -------
    xarray.Dataset
        genetic_varcov_PSD layout: covariance variables (G, E, G_unshrunk, E_unshrunk, G_prediction, E_prediction, Rg, G_se, E_se, rg_se) on ('trait_i','trait_j') and h2/h2_se on 'trait'; attrs hold native component_specs/E_spec, prediction_profiles, saved_parameters, analytic_se, phenotype_digest, uncertainty_method, legacy_support_recovered and source_path. Missing legacy support masks are inferred from exact nonzero loadings and marked legacy_support_recovered.
    
    Complexity
    ----------
    TIME O(t²+tR+b); SPACE O(t²+tR), for t traits, total factor rank R and serialized bytes b. This eagerly loads saved arrays but never reads the n-animal phenotype frame.
    
    Notes
    -----
    Raises ValueError for duplicate/mismatched trait axes. Loading a fit is not approval of its scientific provenance; add_covariance_se separately checks the original phenotype digest.
    """
    with xr.open_zarr(path,consolidated=True,chunks=None) as ds:
        cols=pd.Index(np.asarray(ds.trait_i.values,dtype=object).astype(str))
        if not cols.is_unique or not np.array_equal(cols,np.asarray(ds.trait_j.values,dtype=object).astype(str)):
            raise ValueError('covariance trait axes differ or contain duplicates')
        fit={k:pd.DataFrame(ds[k].values,index=cols,columns=cols) for k in
             _SAVED_MATRICES if k in ds}
        for c in ('G','E'):
            factor=ds[c+'_loadings'].values
            support=ds[c+'_support'].values if c+'_support' in ds else factor!=0
            spec=dict(factor=factor,diag=ds[c+'_specific'].values,support_mask=support,structure='sparse_fa')
            if c=='G': fit['component_specs']={'genetic':spec}
            else: fit['E_spec']=spec
        fit['prediction_profiles']={p:{c:pd.DataFrame(ds[p+'_'+c].values,index=cols,columns=cols)
              for c in ('G','E')} for p in ('unseen_animals','partial_traits') if all(p+'_'+c in ds for c in ('G','E'))}
        for k in _SAVED_VECTORS:
            if k in ds: fit[k]=pd.Series(ds[k].values,index=cols)
        for k in _SAVED_INTERVALS:
            if k in ds: fit[k]=pd.DataFrame(ds[k].values,index=cols,columns=['lower','upper'])
        fit['h2_prior']=json.loads(ds.attrs.get('h2_prior','{"active": false}'))
        fit['saved_parameters']=json.loads(ds.attrs.get('parameters','{}'))
        fit['analytic_se']=json.loads(ds.attrs.get('analytic_se_metadata','{}'))
        fit['phenotype_digest']=ds.attrs.get('phenotype_digest')
        fit['uncertainty_method']=ds.attrs.get('uncertainty_method','not recorded')
        fit['legacy_support_recovered']=not all(c+'_support' in ds for c in ('G','E'))
        fit['source_path']=str(Path(path).resolve())
    return mv._covariance_fit_dataset(fit, software='vigor.phenome.open_covariance_fit',
                                      title='Saved covariance fit (vigor.phenome.covariance.v1)')


@operation_progress('Post-fit covariance SE')
def add_covariance_se(grm, traits, fit, *, parameters=None, probes=64, trace_probes=16,
                      max_probes=None, mc_rtol=None, kernel_factor='paired', ci_level=0.95,
                      cg_rtol=1e-3, cg_maxiter=2000,
                      allow_legacy_provenance=False, random_state=42, blas_threads=8, progress=True,
                      save_path=None):
    """Recover fixed-support analytic covariance SEs from exact original fitting inputs.
    
    Parameters
    ----------
    grm : ndarray, shape (n,n)
        Original animal relationship matrix in the exact trait row order. This
        helper checks shape, not genotype/GRM identity; retain the source manifest.
    traits : pandas.DataFrame, shape (n,t)
        Exact original standardized fitting phenotypes, labels, row order and NaNs.
        Trait columns must match fit['G'].columns and phenotype digest must match.
    fit : xarray.Dataset, mapping or pandas.Series
        Completed single-genetic-component lbfgs FA fit with native component_specs
        and E_spec matching inference G/E (t,t). genetic_varcov_PSD Datasets and
        checkpoints from open_covariance_fit are supported.
    parameters : dict or None
        Default: ``None``. Overrides saved fitting configuration for moment
        recomputation: mode, block_size, variance_closure, variance_closure_strength,
        ridge_G, ridge_E and eps. moment_estimator must be 'mom'/'score'; solver
        must be 'lbfgs'. These must describe the actual original estimator.
    probes : int
        Default: ``64``. Probe count of the final FA sandwich (``analytic_se_probes`` of
        genetic_varcov_PSD); exactly this many unless mc_rtol is set.
    trace_probes : int
        Default: ``16``. Unused by the 2.12.1 sandwich; kept in the recorded parameters.
    max_probes, mc_rtol : int or None, float or None
        Defaults: ``None``. Optional probe cap and Monte Carlo target
        (``analytic_se_max_probes``/``analytic_se_mc_rtol``); None uses exactly ``probes``.
    kernel_factor : {'paired', 'grm2us', 'cholesky'}
        Default: ``'paired'``. Moment-covariance backend (``analytic_kernel_factor``).
    ci_level, cg_rtol, cg_maxiter : float, float, int
        Defaults: ``0.95``, ``1e-3``, ``2000``. Interval coverage and tangent-CG controls
        (``analytic_ci_level``, ``fa_se_cg_rtol``, ``fa_se_cg_maxiter``).
    allow_legacy_provenance : bool
        Default: ``False``. True permits an absent legacy phenotype digest with
        a warning; a present mismatching digest always fails. The caller must
        establish original phenotype/kernel identity independently for legacy fits.
    random_state : int or None
        Default: ``42``. Seed for analytic probes; None uses the backend RNG policy.
    blas_threads : int or None
        Default: ``8``. Positive BLAS thread cap; None preserves thread settings.
    progress : bool
        Default: ``True``. Report moment, variance-closure and analytic-SE stages.
    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded
        Zarr v3 and returned reopened lazily. Default: ``None``.
    
    Returns
    -------
    xarray.Dataset
        The fit in genetic_varcov_PSD layout with G_se/E_se/rg_se, rg_ci95_low/high on
        ('trait_i','trait_j'), h2/h2_se/h2_ci95 and G/E variance SEs and intervals on
        'trait' (robust empirical-Bayes h2 and variance intervals when the fit used the
        h2 prior), and attrs phenotype_digest,
        uncertainty_method and analytic_se probe/trace metadata. Rank and native
        sparse support stay fixed; support/graph/rank-selection uncertainty is
        excluded. Genetic correlation SEs are NaN at a zero-genetic-variance boundary.
    
    Complexity
    ----------
    TIME O(n²t+nt²)+C_SE; SPACE O(n²+nt+t²+S_SE) for n animals and t traits.
    C_SE/S_SE are vigor.fa_sandwich.final_fa_sandwich_se work/storage: with the default
    'paired' backend four O(n²t) kernel products per probe and the batched tangent CG,
    with no animal-scale factorization. This helper may copy
    GRM to float32 and recomputes moments; it does not rerun the covariance optimizer.
    A blocked contraction limits row-block temporaries, not dense GRM storage.
    """
    from threadpoolctl import threadpool_limits
    from .results import maybe_save
    fit=mv._result_view(fit); traits=mv._as_pandas_input(traits)
    cols=pd.Index(fit['G'].columns)
    if not traits.columns.equals(cols): raise ValueError('trait order must match saved fit')
    if np.shape(grm)!=(len(traits),len(traits)): raise ValueError('GRM size must match animals')
    digest=fit.get('phenotype_digest')
    if digest is not None and digest!=phenotype_digest(traits): raise ValueError('phenotypes differ from saved fit')
    if digest is None:
        if not allow_legacy_provenance: raise ValueError('legacy checkpoint lacks phenotype digest; explicitly allow original-input recovery')
        warnings.warn('Legacy input digest unavailable: use the exact original standardized traits and GRM.',RuntimeWarning)
    cfg=dict(fit.get('saved_parameters',{})); cfg.update(parameters or {})
    if cfg.get('moment_estimator','mom') not in ('mom','score'): raise ValueError('post-fit recovery supports MoM only')
    solver=str(cfg.get('solver','lbfgs')).lower()
    if solver!='lbfgs': raise ValueError('this recovery helper supports lbfgs SparseFA fits; compute SEs in the original solver for other fits')
    gs=mv.as_covariance_spec(next(iter(fit['component_specs'].values())),name='genetic')
    es=mv.as_covariance_spec(fit['E_spec'],name='E')
    for c,s in (('G',gs),('E',es)):
        np.testing.assert_allclose(s.materialize(),np.asarray(fit.get(c+'_unshrunk',fit[c])),rtol=1e-6,atol=1e-7,
                                   err_msg='native factors must match inference covariance')
    with threadpool_limits(blas_threads):
        operation_stage('post-fit SE: moment recompute')
        moments=mv._single_moments(np.asarray(grm,dtype=np.float32),traits,
                mode=cfg.get('mode','auto'),block_size=cfg.get('block_size',512),
                block_dtype=np.float32,accum_dtype=np.float64,square_dtype=None,dense_threshold_gb=4.)
        operation_stage('post-fit SE: variance closure')
        closure=cfg.get('variance_closure','auto')
        if closure not in (False,None,'off','none','false','diagnostic'):
            cm=mv._phenotypic_variance_moment(traits,eps=1e-8)
            moments=mv._augment_moments_variance_closure(moments,cm['variance'],cm['precision']*cfg.get('variance_closure_strength',1.))
        from .fa_sandwich import final_fa_sandwich_se
        from .h2_prior import penalty_from_fit, robust_intervals
        # The stored penalty carries its pilot state, so the learned weights move with the draws.
        penalty=penalty_from_fit(fit)
        ase=final_fa_sandwich_se({'genetic':np.asarray(grm,dtype=np.float32)},traits,moments,{'genetic':gs,'E':es},
              ridge=[float(cfg.get('ridge_G',0.)),float(cfg.get('ridge_E',0.))],h2_penalty=penalty,
              n_probes=probes,max_probes=max_probes,mc_rtol=mc_rtol,kernel_factor=kernel_factor,
              cg_rtol=cg_rtol,cg_maxiter=cg_maxiter,ci_level=ci_level,
              random_state=random_state,eps=cfg.get('eps',1e-8))
        if penalty is not None:
            ase.update(robust_intervals(ase['h2'],ase['h2_se'],np.diag(gs.materialize()),ase['G_variance_se'],
                                        np.diag(es.materialize()),ase['E_variance_se'],penalty,ci_level=ci_level))
    out=dict(fit)
    for k in ('G_se','E_se','rg_se','rg_ci95_low','rg_ci95_high'):
        out[k]=pd.DataFrame(ase[k],index=cols,columns=cols)
    for k in ('h2','h2_se','G_variance_se','E_variance_se'): out[k]=pd.Series(ase[k],index=cols)
    for k in ('h2_ci95','G_variance_ci95','E_variance_ci95'):
        out[k]=pd.DataFrame(ase[k],index=cols,columns=['lower','upper'])
    for k in ('h2_ci_critical_value','h2_ci_bias_m2'):
        if k in ase: out[k]=pd.Series(ase[k],index=cols)
    out['cov_GE_diag']=pd.Series(ase['cov_GE_diag'],index=cols)
    # Genetic correlations are undefined on a zero-variance boundary.
    bad=np.diag(np.asarray(fit['G']))<=1e-8
    for k in ('rg_se','rg_ci95_low','rg_ci95_high'):
        out[k]=out[k].mask(bad[:,None]|bad[None,:])
    out['uncertainty_method']=ase['method']
    out['analytic_se']={k:ase[k] for k in ('method','kernel_backend','probe_estimator','n_probes','mc_target_met',
        'se_relative_mc_error_max','negative_variance_estimates','common_trait_mask','kernel_factor',
        'kernel_retained_trace_fraction','cg_iterations','cg_max_relative_energy_error',
        'h2_prior_weights_propagated','closure_precision_propagated','conditional_on_rank') if k in ase}
    out['analytic_se']['h2_interval']=('robust empirical-Bayes interval (Armstrong, Kolesar and Plagborg-Moller 2022)'
                                       if penalty is not None else 'logit-scale Wald interval')
    out['analytic_se'].update(conditional_on_sparse_support=True,includes_support_selection_uncertainty=False)
    out['phenotype_digest']=phenotype_digest(traits)
    ds=mv._covariance_fit_dataset(out, dict(probes=probes, trace_probes=trace_probes, random_state=random_state,
                                            blas_threads=blas_threads, allow_legacy_provenance=allow_legacy_provenance),
                                  software='vigor.phenome.add_covariance_se',
                                  title='Covariance fit with post-fit analytic standard errors')
    return maybe_save(ds, save_path)


def native_factor_loadings(fit, component='G'):
    """Return saved native factor loadings without rotating or refitting them.
    
    Parameters
    ----------
    fit : xarray.Dataset, mapping or pandas.Series
        Completed/opened covariance fit containing G.columns and native component_specs/E_spec.
    
    component : str
        Default: ``'G'``. G selects the first genetic component; E selects residual factors; another string selects a named genetic component or raises KeyError.
    
    Returns
    -------
    pandas.DataFrame, shape (t, r)
        Loadings indexed by saved traits; columns use component_FA1 etc. Attributes identify the actual component. These are loadings, not animal scores.
    
    Complexity
    ----------
    TIME O(t+r); SPACE O(t+r) metadata when the factor array is shared, up to O(tr) if pandas copies it. t is traits and r selected component rank.
    """
    fit=mv._result_view(fit)
    if component=='E':spec=fit['E_spec'];name='E'
    else:
        name=next(iter(fit['component_specs'])) if component=='G' else str(component)
        if name not in fit['component_specs']:raise KeyError('unknown genetic covariance component '+name)
        spec=fit['component_specs'][name]
    cols=fit['G'].columns
    out=pd.DataFrame(spec['factor'],index=cols,columns=[f'{component}_FA{i+1}' for i in range(np.shape(spec['factor'])[1])])
    out.attrs['covariance_component']=name;out.attrs['total_genetic_factor']=False
    return out


def _joint_animal_factor_projection(traits, fit, *, center, trait_means,
                                    progress, blas_threads, batch_size):
    """Compute grouped within-animal Gaussian native-factor posteriors.
    
    Parameters
    ----------
    traits : pandas.DataFrame
        Labelled animals by observed trait subset; NaN cells are omitted.
    fit : mapping
        Native component_specs/E_spec and ordered trait_labels or G.columns.
    center : 'auto' or bool
        Zero or training centering; True permits input-cohort means as fallback.
    trait_means : array-like, pandas.Series or None
        Complete fitted-axis training means; None checks fit['trait_mean'].
    progress : bool
        Enable one counted bar for unique observation masks.
    blas_threads : int or None
        Positive thread cap, or None to retain the current thread configuration.
    batch_size : int
        Positive maximum number of animals per temporary response batch.
    
    Returns
    -------
    xarray.Dataset
        Native factor posterior means/SDs and observed counts in input row order.
    
    Complexity
    ----------
    TIME O(nt log n + sum_g(s_g³+m_g R s_g+n_g m_g R)); SPACE O(nt+nR+tR+max_g(m_g R+s_g²+b m_g)), with n animals, t input traits, R total G/E factors, g observation masks, m_g observed traits, n_g animals per mask, batch cap b, and s_g=min(m_g,R) when noise is positive. Zero specific variance forces s_g=m_g. Grouping/packing/sorting adds O(nt log n) as a safe upper bound. Returned factor means/SDs require O(nR); no kinship matrix is used.
    """
    from scipy.linalg import cho_factor, cho_solve
    from threadpoolctl import threadpool_limits
    from .progress import StageProgress
    if not isinstance(traits, pd.DataFrame):
        raise TypeError('traits must be a labelled pandas DataFrame')
    cols = traits.columns
    if cols.hasnans or not cols.is_unique:
        raise ValueError('trait labels must be unique and non-null')
    if len(fit['component_specs']) != 1:
        raise ValueError('within-animal projection requires one genetic component')
    genetic_name = next(iter(fit['component_specs']))
    labels = fit.get('trait_labels', fit.get('trait_names'))
    if labels is None and fit.get('G') is not None:
        labels = fit['G'].columns
    if labels is None:
        raise ValueError('lean fit requires ordered trait_labels')
    labels = pd.Index(labels)
    if labels.hasnans or not labels.is_unique:
        raise ValueError('fitted trait labels must be unique and non-null')
    loc = labels.get_indexer(cols)
    if np.any(loc < 0):
        raise ValueError('traits contain labels absent from fitted trait axis')
    factors = []; diagonals = []
    for spec in (fit['component_specs'][genetic_name], fit['E_spec']):
        factor = np.asarray(spec['factor'] if isinstance(spec, Mapping) else spec.factor, float)
        diag = np.asarray(spec.get('diag', np.zeros(len(labels))) if isinstance(spec, Mapping) else spec.diag_specific, float)
        if factor.ndim != 2 or factor.shape[0] != len(labels) or diag.shape != (len(labels),):
            raise ValueError('native factors and specific variances must match fitted trait axis')
        if not np.isfinite(factor).all() or not np.isfinite(diag).all() or np.any(diag < 0):
            raise ValueError('native factors must be finite and specific variances nonnegative')
        factors.append(factor[loc]); diagonals.append(diag[loc])
    D = diagonals[0] + diagonals[1]
    if center != 'auto' and not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be 'auto' or bool")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if blas_threads is not None and (isinstance(blas_threads, bool) or not isinstance(blas_threads, int) or blas_threads < 1):
        raise ValueError('blas_threads must be a positive integer or None')
    if isinstance(center, np.bool_): center = bool(center)
    values = traits.to_numpy(copy=False)
    if values.dtype.kind not in 'fiu':
        raise TypeError('traits must contain numeric values with NaN for missing cells')
    if np.isinf(values).any():
        raise ValueError('traits must be finite or NaN')
    obs = np.isfinite(values)
    means = np.zeros(len(cols)); centering = 'zero standardized response'
    supplied = trait_means if trait_means is not None else fit.get('trait_mean')
    if center is not False and supplied is not None:
        if isinstance(supplied, pd.Series):
            if supplied.index.hasnans or not supplied.index.is_unique or len(supplied) != len(labels) or np.any(supplied.index.get_indexer(labels) < 0):
                raise ValueError('training means must uniquely cover the fitted trait axis')
            supplied = supplied.reindex(labels).to_numpy(float)
        supplied = np.asarray(supplied, float)
        if supplied.shape != (len(labels),) or not np.isfinite(supplied).all():
            raise ValueError('training means must be finite on the complete fitted trait axis')
        means = supplied[loc]; centering = 'supplied training means'
    elif center is True:
        # Empty columns use zero; avoid all-NaN mean warnings.
        means = np.divide(np.sum(values, axis=0, where=obs, initial=0, dtype=float), obs.sum(axis=0),
                          out=means, where=obs.sum(axis=0) > 0)
        centering = 'observed input cohort means'
    L = np.concatenate(factors, axis=1); rg = factors[0].shape[1]; rank = L.shape[1]
    scores = np.zeros((len(traits), rank)); sd = np.ones_like(scores)
    patterns, inverse = np.unique(np.packbits(obs, axis=1), axis=0, return_inverse=True)
    # Sort group indices once, rather than scanning n rows for each unique mask.
    order = np.argsort(inverse, kind='stable')
    bounds = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(patterns)))]
    # Choose the smaller exact Gaussian conditioning system separately for
    # each missingness pattern. Full-rank FA can have >2,000 latent factors
    # even when an animal has only a handful of observed traits: solving the
    # rank-sized system per unique mask would then be prohibitively expensive.
    solve_counts = {'observed_dual': 0, 'latent_primal': 0, 'all_missing': 0,
                    'zero_rank_observed': 0}
    with threadpool_limits(limits=blas_threads, user_api='blas'), StageProgress(
            total=len(patterns), desc='Animal factor projection', enabled=progress, unit='mask') as bar:
        for group in range(len(patterns)):
            rows = order[bounds[group]:bounds[group+1]]
            observed = np.flatnonzero(obs[rows[0]])
            m = len(observed)
            if m and rank:
                Lo = L[observed]
                if m < rank or np.any(D[observed] == 0):
                    # Woodbury dual: C_o = L_o L_o' + diag(Psi_G+Psi_E).
                    # E(z|y_o) = L_o' C_o^-1 (y_o-mu_o). Unlike a truncated
                    # factor projection, this retains ALL nuisance factors.
                    C = Lo @ Lo.T
                    C.flat[::m+1] += D[observed]
                    try:
                        ch = cho_factor(C, lower=True, check_finite=False)
                    except np.linalg.LinAlgError as exc:
                        raise ValueError('observed trait covariance must be positive definite for joint projection') from exc
                    projection = cho_solve(ch, Lo, check_finite=False)
                    posterior_var = 1.0 - np.sum(Lo * projection, axis=0)
                    solve_counts['observed_dual'] += 1
                else:
                    weighted = Lo / D[observed, None]
                    ch = cho_factor(np.eye(rank) + Lo.T @ weighted,
                                    lower=True, check_finite=False)
                    projection = cho_solve(ch, weighted.T, check_finite=False).T
                    posterior_var = np.diag(
                        cho_solve(ch, np.eye(rank), check_finite=False))
                    solve_counts['latent_primal'] += 1
                uncertainty = np.sqrt(np.clip(posterior_var, 0.0, 1.0))
                sd[rows] = uncertainty
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start+batch_size]
                    scores[batch] = (values[np.ix_(batch, observed)] - means[observed]) @ projection
            elif m:
                if np.any(D[observed] == 0):
                    raise ValueError('observed trait covariance must be positive definite for joint projection')
                solve_counts['zero_rank_observed'] += 1
            else:
                solve_counts['all_missing'] += 1
            bar.update()
    coords = {'sample': ('sample', traits.index.astype(str).to_numpy()),
              'factor_G': [f'G_FA{i+1}' for i in range(rg)],
              'factor_E': [f'E_FA{i+1}' for i in range(rank-rg)]}
    data = {'n_observed': ('sample', obs.sum(axis=1))}
    for c, sl in (('G', slice(0, rg)), ('E', slice(rg, rank))):
        data[c+'_trait_projection'] = (('sample', 'factor_'+c), scores[:, sl])
        data[c+'_trait_projection_sd'] = (('sample', 'factor_'+c), sd[:, sl])
    return xr.Dataset(data, coords=coords, attrs={
        'schema': 'vigor.animal_factors.v2', 'genetic_component': genetic_name,
        'projection_note': 'within-animal joint Gaussian factor posterior; ignores kinship; not BLUPs',
        'centering': centering, 'mask_pattern_count': len(patterns),
        'posterior_solver': 'adaptive_exact_gaussian', 'solver_pattern_counts': solve_counts,
        'uncertainty_note': 'conditional marginal SD; fitted loadings/variances held fixed',
        'causal_note': 'descriptive fitted projections, not independent causal observations'})


def animal_factor_embeddings(traits, fit, prediction=None, *, center='auto',
                             trait_means=None, progress=False, blas_threads=1,
                             batch_size=1024):
    """Project observed traits onto native genetic and residual latent factors.
    
    With prediction=None, each animal has the joint Gaussian posterior
    z|y_obs with L=[L_G,L_E], D=diag(psi_G+psi_E) and observed covariance
    C_obs=L_obs @ L_obs.T + D_obs. Posterior means are
    L_obs.T @ solve(C_obs,y_obs-mean_obs), and covariance is
    I-L_obs.T @ solve(C_obs,L_obs). Missing traits are omitted. These within-animal
    projections ignore kinship and are NOT BLUPs. No GRM, mixed-model solve,
    or full trait predictions are constructed. Identical masks share a Cholesky
    solve in the smaller observed or latent space when noise is positive;
    zero specific noise uses the observed covariance, which must be positive
    definite. Response batches are bounded by batch_size.
    All-missing animals have zero prior means and unit posterior SDs.
    
    Supplying prediction preserves the legacy posterior-component API and
    its separately centered trait projections, including NaN empty rows.
    
    Parameters
    ----------
    traits : pandas.DataFrame
        Animals by available traits, with unique non-null column labels. A
        reordered subset of fitted traits is accepted without prediction.
        NaN denotes missing data; infinities are rejected. Index order is kept.
    fit : mapping or xarray.Dataset
        Native component_specs (one genetic component) and E_spec, each with
        factor and nonnegative diag. Lean fits need trait_labels (or trait_names)
        in factor-row order; complete fits can use G.columns. No G/E matrices
        are needed without prediction. Specific variances may be zero if each
        observed covariance block remains positive definite; singular blocks fail
        explicitly without adding an undocumented noise floor.
    prediction : mapping, xarray.Dataset or None
        Default None selects within-animal projection. Otherwise supply the
        matching mixed-model prediction (an mvPredict Dataset or dict) on the complete
        fitted trait axis; legacy G_genetic/E_residual keys retain their original meanings.
    center : 'auto' or bool
        Default 'auto' uses supplied trait_means, then fit['trait_mean'], then
        zero (the fitted standardized-response convention). True uses those
        training means if available, otherwise observed means of this input
        cohort. False uses zero. Never infer training means from a new cohort
        for prediction: supply the original training means explicitly.
        Only applies when prediction is None; legacy centering is preserved.
    trait_means : pandas.Series or array-like or None
        Default None. Training-scope means on the complete fitted trait axis;
        labelled means must cover that axis uniquely, and arrays use factor
        row order. Subset selection follows traits.columns. Applies only to
        prediction-free projection.
    progress : bool
        Default False. Report deterministic completed mask-group counts.
    blas_threads : int or None
        Default 1 bounds BLAS threads for small rank solves; None leaves BLAS
        unchanged. Applies to the prediction-free path.
    batch_size : int
        Default 1024; positive maximum animals in each temporary response batch.
    
    Returns
    -------
    xarray.Dataset
        G_trait_projection/E_trait_projection (sample by native factor),
        matching *_sd posterior marginal uncertainties, and n_observed for
        prediction-free calls. Prior factors have unit variance. Factor names
        G_FA1/E_FA1 etc. preserve fit column order. Supplying prediction returns
        legacy G_genetic/E_residual and trait-projection keys. Sample IDs are
        strings, preserving input order. Attributes record centering and scope.
    
    Complexity
    ----------
    For prediction=None, TIME O(nt log n + sum_g(s_g³+m_g R s_g+n_g m_g R)); SPACE O(nt+nR+tR+max_g(m_g R+s_g²+b m_g)). n is animals, t input traits, R combined native factors, g mask groups with n_g animals/m_g observed traits; b=batch_size and s_g=min(m_g,R), except zero noise forces observed-space s_g=m_g. The legacy prediction-supplied branch materializes trait covariances/pseudoinverses: TIME O(t³+nt²+ntR), SPACE O(t²+nt+nR), plus the per-mask GLS projection. These are projection costs, not full mixed-model fitting costs.
    """
    fit=mv._result_view(fit); traits=mv._as_pandas_input(traits)
    if prediction is not None: prediction=mv._result_view(prediction)
    if prediction is None:
        return _joint_animal_factor_projection(
            traits, fit, center=center, trait_means=trait_means, progress=progress,
            blas_threads=blas_threads, batch_size=batch_size)
    from .factor_projection import _factor_scores_gls
    genetic_name=next(iter(fit['component_specs']))
    gs=mv.as_covariance_spec(fit['component_specs'][genetic_name])
    es=mv.as_covariance_spec(fit['E_spec'])
    data={}; coords={'sample': ('sample', traits.index.astype(str).to_numpy())}
    for c,s,other,key,outkey in [('G',gs,es,'breeding_value','G_genetic'),('E',es,gs,'residual_prediction','E_residual')]:
        names=[f'{c}_FA{i+1}' for i in range(s.rank)]; coords['factor_'+c]=names
        if c=='G' and len(fit['component_specs'])>1:
            components=prediction.get('components',prediction.get('genetic_components',{}))
            if genetic_name not in components:raise ValueError('multi-kernel factor embeddings require matching posterior genetic components')
            value=components[genetic_name]
        else:value=prediction[key]
        if c=='G' and 'G_matrices' in prediction and genetic_name in prediction['G_matrices']:
            if not np.allclose(np.asarray(prediction['G_matrices'][genetic_name]),s.materialize(),rtol=1e-6,atol=1e-8):
                raise ValueError('prediction and native factor covariance differ; use inference covariance prediction')
        if c=='E' and 'E_matrix' in prediction and not np.allclose(prediction['E_matrix'],s.materialize(),rtol=1e-6,atol=1e-8):
            raise ValueError('prediction and native residual factor covariance differ; use inference covariance prediction')
        if isinstance(value,pd.DataFrame):
            if (not value.index.astype(str).equals(traits.index.astype(str))
                    or not value.columns.astype(str).equals(traits.columns.astype(str))):
                raise ValueError('prediction labels/order must match traits')
        component=np.asarray(value,float)
        if component.shape!=traits.shape: raise ValueError('prediction shape must match traits')
        data[outkey]=( ('sample','factor_'+c),component@np.linalg.pinv(s.materialize(),rcond=1e-10)@s.factor)
        noise=other.materialize()+np.diag(s.diag_specific)
        proj,_=_factor_scores_gls(traits,s.factor,noise,method='posterior')
        data[c+'_trait_projection']=( ('sample','factor_'+c),proj.to_numpy())
    return xr.Dataset(data,coords=coords,attrs={'schema':'vigor.animal_factors.v1',
         'genetic_component':genetic_name,'full_embedding':'concatenate named genetic-component and E_residual factors; total BV is not projected through a single component',
         'projection_note':'trait projections ignore kinship; posterior components reuse the mixed model',
         'causal_note':'embeddings are descriptive fitted estimates, not independent causal observations'})


def read_gcta_grm(prefix, *, weights=False):
    """Expand a lower-triangular GCTA binary GRM in its recorded animal order.
    
    Parameters
    ----------
    prefix : str or pathlib.Path
        GCTA prefix with companions .grm.id and .grm.bin; trailing .grm or .grm.bin is accepted. The second ID column supplies unique animal labels.
    
    weights : bool
        Default: ``False``. True additionally reads the pairwise marker-count matrix from .grm.N.bin; it is not a weight vector input.
    
    Returns
    -------
    tuple
        (ids, K), or (ids, K, N) if weights=True. ids has shape (n,), K/N are symmetric float32 arrays (n,n). Files must contain n(n+1)/2 float32 entries.
    
    Complexity
    ----------
    TIME O(n²); SPACE O(n²) for one or two expanded matrices plus O(n) IDs. Triangular inputs are read-only mmap; expansion avoids quadratic index arrays.
    
    Notes
    -----
    Raises ValueError for duplicate IDs or invalid triangular length. File assembly, sample inclusion and allele/kernel semantics remain caller provenance requirements.
    """
    prefix=str(prefix).removesuffix('.grm.bin').removesuffix('.grm')
    ids=pd.read_csv(prefix+'.grm.id',sep=r'\s+',header=None,dtype=str)[1].to_numpy()
    if len(set(ids))!=len(ids): raise ValueError('duplicate GRM animal IDs')
    n=len(ids)
    def expand(ext):
        """Expand the stored triangular GCTA relationship data into a symmetric matrix.

        Parameters
        ----------
        ext : object (see the cited input contract)
            Input for the operation ``prefix + ext``. See ``vigor.phenome.read_gcta_grm.expand`` and
            its caller for the scientific representation; the shown operation constrains the
            required type/shape.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        flat=np.memmap(prefix+ext,dtype=np.float32,mode='r')
        if len(flat)!=n*(n+1)//2: raise ValueError('invalid triangular GRM')
        out=np.empty((n,n),dtype=np.float32); offset=0
        for i in range(n):
            out[i,:i+1]=flat[offset:offset+i+1];out[:i+1,i]=out[i,:i+1];offset+=i+1
        return out
    K=expand('.grm.bin')
    return (ids,K,expand('.grm.N.bin')) if weights else (ids,K)


class GCTALOCO(Mapping):
    """Materialize one weighted chromosome exclusion at a time, aligned to All.

    Holds the All matrix and its SNP-count matrix; does not cache chromosome
    exclusions. Caller should release each exclusion before asking for another.
    """
    def __init__(self, matrix, ids, folder, chromosomes):
        """Configure lazy weighted chromosome exclusions without reading genotype data.
        
        Parameters
        ----------
        matrix : numpy.ndarray, shape (n, n)
            All-chromosome GRM in the exact order of ids; stored by reference.
        
        ids : sequence of str, length n
            All-chromosome animal order; chromosome matrices are aligned to these labels.
        
        folder : str or pathlib.Path
            GCTA directory containing AllchrGRM and <chrom>chrGRM files with IDs and pairwise marker counts.
        
        chromosomes : iterable of str or int
            Allowed exclusion keys, converted to strings in the supplied order; None is unsupported.
        
        Returns
        -------
        None
            Initializes the mapping; exclusion matrices are computed by __getitem__.
        
        Complexity
        ----------
        TIME O(n+k); SPACE O(n+k) metadata beyond matrix, for k chromosomes. No All/chr matrices are read during construction.
        """
        self.matrix=matrix; self.ids=pd.Index(ids);self.folder=Path(folder)
        self.chromosomes=tuple(map(str,chromosomes));self.weights=None
    def __iter__(self): 
        """Iterate through the collection in its stored order.
        """
        return iter(self.chromosomes)
    def __len__(self): 
        """Return the number of entries in this collection.
        """
        return len(self.chromosomes)
    def __getitem__(self, chrom):
        """Return the requested count-weighted LOCO GRM aligned to All-chromosome animals.
        
        Parameters
        ----------
        chrom : str or int
            Chromosome key present in the configured chromosomes; unknown keys raise KeyError.
        
        Returns
        -------
        numpy.ndarray, shape (n, n)
            (All*All_N - chr*chr_N)/(All_N-chr_N), float32 in All animal order. The caller owns the result; exclusions are not cached.
        
        Complexity
        ----------
        TIME O(n²+q²); SPACE O(n²+q²), where q is chromosome-file animal count before alignment and n requested animals. All marker counts are cached after the first access; each exclusion is re-read.
        
        Notes
        -----
        Raises ValueError for inconsistent All IDs, missing chromosome animals, or nonpositive remaining pairwise marker counts. This requires commensurate marker-count-normalized GRMs; arbitrary differently scaled kernels cannot be mixed.
        """
        chrom=str(chrom)
        if chrom not in self.chromosomes: raise KeyError(chrom)
        if self.weights is None:
            ids,unused,W=read_gcta_grm(self.folder/'AllchrGRM',weights=True)
            if not np.array_equal(ids,self.ids): raise ValueError('All GRM order differs')
            self.weights=W; del unused
        ids,C,W=read_gcta_grm(self.folder/f'{chrom}chrGRM',weights=True)
        ix=pd.Index(ids).get_indexer(self.ids)
        if np.any(ix<0): raise ValueError('chromosome GRM missing animals')
        if not np.array_equal(ix,np.arange(len(ix))) or len(ids)!=len(ix): C=C[np.ix_(ix,ix)];W=W[np.ix_(ix,ix)]
        den=self.weights-W
        if np.any(den<=0): raise ValueError('no variants remain in LOCO background')
        out=self.matrix*self.weights-C*W
        np.divide(out,den,out=out)
        return out


def factor_scan_peaks(scan, *, total_variants=None, half_window_bp=3_000_000, fallback_logp=5.3, alpha=.05):
    """Greedy independent lead peaks from omnibus OR marginal-factor p-values.

    Bonferroni counts M*(r+1) tests: r factors and one omnibus per variant.
    Without a known genome-wide M, use the requested descriptive 5.3 cutoff.
    Suppress further leads within +/-half_window_bp; overlapping retained windows
    are still allowed. Targets are factors significant anywhere inside a window;
    omnibus-only peaks retain the strongest marginal factor at the lead.

    Parameters
    ----------
    scan : array-like or numeric scalar
        Input for the operation ``np.asarray(scan.chrom.values, dtype=object).astype(str)``. See
        ``vigor.phenome.factor_scan_peaks`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    total_variants : object (see the cited input contract)
        Input for the operation ``alpha / (int(total_variants) * (z.shape[1] + 1))``. See
        ``vigor.phenome.factor_scan_peaks`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    half_window_bp : int
        Input for the operation ``(chrom == chrom[i]) & (abs(pos - pos[i]) <= half_window_bp)``.
        See ``vigor.phenome.factor_scan_peaks`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``3000000``.

    fallback_logp : float
        Input for the operation ``10.0 ** (-fallback_logp)``. See
        ``vigor.phenome.factor_scan_peaks`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``5.3``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.05``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(rows, columns=['chrom', 'lead_snp', 'lead_pos', 'lead_p', 'start_pos', 'end_pos', 'factors', 'p_threshold', 'threshold_method'])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if scan.sizes.get('variant',0)==0:
        return pd.DataFrame(columns=['chrom','lead_snp','lead_pos','lead_p','start_pos','end_pos','factors','p_threshold','threshold_method'])
    from .gwas.exact import chi2_logsf
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    if half_window_bp < 0 or not np.isfinite(fallback_logp):
        raise ValueError('window width must be nonnegative and fallback_logp finite')
    axis='trait' if 'trait' in scan.coords else 'factor'
    z=np.asarray(scan['z' if 'z' in scan else 'factor_z'].transpose('variant',axis),float)
    lp=np.asarray(scan.omnibus_log_p,float) if 'omnibus_log_p' in scan else np.log(np.asarray(scan.omnibus_p,float))
    flp=np.asarray(scan.log_p.transpose('variant',axis),float) if 'log_p' in scan else chi2_logsf(z*z,1)
    combined=np.minimum(np.where(np.isnan(lp),np.inf,lp),np.min(np.where(np.isnan(flp),np.inf,flp),axis=1))
    if total_variants is not None and (isinstance(total_variants,bool) or int(total_variants)!=total_variants or total_variants<len(z) or total_variants<1):
        raise ValueError('genome-wide variant count must be a positive integer at least as large as the scan')
    cutoff=alpha/(int(total_variants)*(z.shape[1]+1)) if total_variants is not None else 10.**(-fallback_logp)
    chrom=np.asarray(scan['chromosome' if 'chromosome' in scan else 'chrom'].values,dtype=object).astype(str)
    pos=np.asarray(scan['position' if 'position' in scan else 'pos'].values,int)
    snp=np.asarray(scan['variant_id' if 'variant_id' in scan else 'snp'].values,dtype=object).astype(str)
    factors=np.asarray(scan[axis].values,dtype=object).astype(str)
    pmin=np.exp(combined)
    log_cutoff=np.log(alpha)-np.log(int(total_variants))-np.log(z.shape[1]+1) if total_variants is not None else -fallback_logp*np.log(10.)
    p=np.exp(lp)
    suppressed=np.zeros(len(p),bool); rows=[]
    for i in np.argsort(combined):
        if np.isnan(combined[i]) or combined[i]>log_cutoff: break
        if suppressed[i]: continue
        use=(chrom==chrom[i])&(abs(pos-pos[i])<=half_window_bp)
        targets=np.flatnonzero(np.any(flp[use]<=log_cutoff,axis=0))
        if not len(targets): targets=np.array([int(np.nanargmax(abs(z[i])))])
        rows.append(dict(chrom=chrom[i],lead_snp=snp[i],lead_pos=int(pos[i]),lead_p=float(pmin[i]),
            start_pos=max(1,int(pos[i])-half_window_bp),end_pos=int(pos[i])+half_window_bp,
            factors=tuple(factors[targets]),p_threshold=cutoff,
            threshold_method='Bonferroni M*(factors+1)' if total_variants is not None else 'fallback -log10p=5.3'))
        suppressed|=use
    return pd.DataFrame(rows,columns=['chrom','lead_snp','lead_pos','lead_p','start_pos','end_pos','factors','p_threshold','threshold_method'])
