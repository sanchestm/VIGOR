"""LOCO factor effect-subspace scans through the exact multivariate score core."""
import warnings
import numpy as np
import pandas as pd
from . import multivariate as mv
# Projection compatibility exports are separate from the association core.
from .factor_projection import _factor_scores_gls, _score_pattern_metadata, _write_factor_state
from .resources import factor_scan_chunk as _factor_scan_chunk

def _canonicalize_loadings(L):
    """Deterministic factor order/sign without a genome-wide alignment target.

    Parameters
    ----------
    L : array-like
        Numerical operand in ``L * L``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.
    """
    L=np.asarray(L,float).copy();order=np.argsort(np.sum(L*L,axis=0))[::-1];L=L[:,order]
    for j in range(L.shape[1]):
        k=int(np.argmax(np.abs(L[:,j])))
        if L[k,j]<0:L[:,j]*=-1
    return L


def _loadings_from_component(cov,rank,method='fa',standardize=False):
    """Decompose one trait covariance and canonicalize the retained factor loadings.

    Parameters
    ----------
    cov : array-like, mapping or fitted result
        Covariance quantity or component collection consumed by the stated transformation.
        Square matrices use a common trait order; uncertainty matrices match their corresponding
        covariance entries. See the caller for whether this is genetic, residual or total
        covariance.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Passed to ``mv.decompose_covariance`` as ``rank``.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'fa'``. Passed to
        ``mv.decompose_covariance`` as ``method``.

    standardize : bool
        Whether to enable the named graph-model transformation, deterministic tracking or
        constraint. True follows its explicit construction branch; False omits it. See that
        branch for how model dimensions/priors are affected. Default: ``False``. Passed to
        ``mv.decompose_covariance`` as ``standardize``.
    """
    dec=mv.decompose_covariance.__wrapped__(cov,rank=rank,method=method,standardize=standardize,preserve_diag=True)
    return _canonicalize_loadings(dec['loadings']),np.asarray(dec['specific_var']),dec


def _native_loadings_from_fit(fit,component,rank):
    """Use the factor parameterization actually fitted by genetic_varcov_PSD.

    This preserves FA/sparseFA support instead of re-decomposing the fitted
    covariance matrix.  The factor order/sign is canonicalized locally so LOCO
    fits remain independent of the genome-wide axes.

    Parameters
    ----------
    fit : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.
        Source-recognized literal values: ``e``, ``environmental``, ``g``, ``genetic``,
        ``residual``. These select the corresponding branches below; comparison evidence alone
        does not establish an exhaustive enum.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine.
    """
    fit=mv._result_view(fit)
    if not hasattr(fit,'get'):
        raise ValueError('factor_method="native" requires a genetic_varcov_PSD result')
    key=str(component).lower()
    if key in {'g','genetic'}:
        specs=fit.get('component_specs',{})
        if 'genetic' in specs: spec=specs['genetic']
        elif len(specs)==1: spec=next(iter(specs.values()))
        else: spec=fit.get('G_spec')
    elif key in {'e','environmental','residual'}:
        spec=fit.get('E_spec')
    else:
        spec=fit.get('component_specs',{}).get(str(component))
        if spec is None: spec=fit.get('component_specs',{}).get(key)
    if not isinstance(spec,dict) or 'factor' not in spec:
        raise ValueError(f'no native factor specification is available for component {component!r}')
    L=np.asarray(spec['factor'],float)
    if L.ndim!=2 or L.shape[1]<1:
        raise ValueError(f'native component {component!r} has no estimable factor axis')
    requested=int(rank); available=int(L.shape[1]); use=min(requested,available)
    if available<requested:
        warnings.warn(f'native component {component!r} has effective rank {available}, below requested factors={requested}; using {use} factor(s) for this LOCO group',RuntimeWarning,stacklevel=2)
    L=_canonicalize_loadings(L)[:,:use]
    psi=np.asarray(spec.get('diag',np.zeros(L.shape[0])),float)
    dec={'method':'native','rank':int(L.shape[1]),'requested_rank':int(rank),'loadings':L,'specific_var':psi,
         'structure':spec.get('structure'),'rotation':spec.get('rotation'),
         'support_mask':spec.get('support_mask'),'loading_density':spec.get('loading_density'),
         'native_parameterization':spec.get('parameterization','factor_analytic')}
    return L,psi,dec


def factorGWAS(traits, GRM=None, decomposition=None, genotypes=None, *,
               component="G", covariance=None, factors=None, background="loco",
               method="exact", score_method="direct", heterogeneity=True,
               loco_factor="refit", covariance_rank_G=100, covariance_rank_E=50,
               covariance_kwargs=None, factor_method="native", factor_standardize=False,
               marker_coding="additive", groups=None, background_kernels=None,
               block_size=128, chromosomes=None, random_state=0,
               null_solver_kwargs=None, covariates=None, out=None, projection_out=None,
               max_return_rows=2_000_000, variant_chunk=1024, shard_variants=16384,
               maxmem=None, **retired):
    """Test LOCO factor effect directions in the exact full-trait Gaussian null.

    Parameters
    ----------
    traits : DataFrame (n,t)
        Labeled Gaussian phenotypes with NaN missing cells.
    GRM : kernel, named kernels, or canonical chromosome store, optional
        Supplied background; group-keyed inputs should use background_kernels.
    decomposition, covariance : fitted result or matrix, optional
        Axes used only with explicit loco_factor='fixed' sensitivity mode.
    genotypes : DataFrame, source or path
        Sample-by-variant counted-allele dosage.
    component : str
        Genetic/residual or named covariance component defining factor directions.
    factors : positive int, optional
        Number of tested axes, default min(10,t). Rank must stay equal across groups.
    background : {'loco','provided'}
        Construct leave-group-out backgrounds or require supplied kernels.
    method : {'exact'}
        All variants use exact information; calibration/rescoring modes retired.
    score_method : {'direct','subspace'}
        Restrict beta_trait=L beta_factor in the full original-trait null.
    heterogeneity : bool
        Also report nested full-trait minus factor-subspace score statistic.
        This requires full-trait exact information and can be expensive.
    loco_factor : {'refit','fixed'}
        Refit axes per excluded chromosome. Fixed axes are an explicit sensitivity
        analysis, with covariance/null still fitted per exclusion group.
    covariance_rank_G, covariance_rank_E : int or 'full'
        Default covariance ranks unless covariance_kwargs specifies them.
    covariance_kwargs : mapping, optional
        Covariance fitter controls; no genome-wide warm start in refit mode.
    factor_method : {'native','fa','pca'}
        Use fitted factor directions or decompose the selected component.
    factor_standardize : bool
        Standardize only for non-native decomposition.
    marker_coding : {'additive','dominance'}
        Dosage or observed hardcall heterozygosity indicator.
    groups, chromosomes : sequences, optional
        Per-variant exclusion labels and labels to scan.
    background_kernels : mapping, optional
        Exclusion-group -> background kernel(s), aligned by animal labels.
    block_size : positive int or 'auto'
        Source/scoring block length.
    random_state : int
        Covariance fit seed.
    null_solver_kwargs : mapping, optional
        prepare_score_null options; e.g. solver_tol, solver_maxiter, rhs_block_size.
    covariates : sample by covariate frame/array, optional
        Additional fixed effects; per-trait intercepts are mandatory.
    out : path, optional
        Stream sharded Zarr v3, returning its lazy Dataset.
    projection_out : path, optional
        Separate artifact for within-animal posterior factor projections used in
        exploratory fine mapping. These projected targets never replace the
        original-trait association null; no animal-kinship likelihood is claimed
        for downstream regressions on these targets. None creates no projections.
    max_return_rows : positive int or None
        In-memory score-cell limit.
    variant_chunk, shard_variants : positive int
        Persisted chunk and shard lengths.
    maxmem : memory size, optional
        Genotype source block budget.
    **retired : mapping
        Unknown/retired controls fail explicitly; see ROUND3_MIGRATION.md.

    Returns
    -------
    xarray.Dataset
        Variant by factor coordinate tests and joint factor omnibus. Q_SNP uses
        nested score projections under the same null, not posterior factor scores.
        Inference conditions on the chosen covariance and factor directions.

    Complexity
    ----------
    Per-group covariance/eigendecomposition plus exact O(m*r) RHS solves for r
    factors; heterogeneity adds O(m*t) solves. Result memory is one block with out.
    """
    from .gwas.scan import (scan_inputs, background_for_group, genotype_blocks,
                            attach_variant_metadata, finish_scan, covariance_matrices,
                            scan_covariates)
    from .gwas.exact import prepare_score_null, score_block, chi2_logsf
    from pathlib import Path
    if retired:
        raise TypeError("retired factorGWAS controls: " + ", ".join(sorted(retired)) +
                        "; see docs/ROUND3_MIGRATION.md")
    if method != "exact" or score_method not in {"direct", "subspace"}:
        raise ValueError("use method='exact', score_method='direct'; posterior/fast scans are retired")
    if loco_factor not in {"refit", "fixed"}:
        raise ValueError("loco_factor must be refit or fixed")
    if marker_coding not in {"additive", "dominance"}:
        raise ValueError("marker_coding must be additive or dominance")
    if factor_method not in {"native", "fa", "pca"}:
        raise ValueError("factor_method must be native, fa or pca")
    if factor_method == "native" and factor_standardize:
        raise ValueError("native factor directions cannot be restandardized")
    if genotypes is None:
        raise ValueError("genotypes are required")
    if projection_out is not None and out is not None:
        association_path = str(out) if str(out).endswith('.zarr') else str(out) + '.zarr'
        if Path(projection_out).resolve() == Path(association_path).resolve():
            raise ValueError("projection_out must be separate from association out")
    Y, source, vm = scan_inputs(traits, genotypes, groups=groups, maxmem=maxmem)
    covariates = scan_covariates(covariates, Y.index)
    if decomposition is not None and hasattr(decomposition, "dims"):
        axes = [dim for dim in ("trait", "trait_i", "trait_j") if dim in decomposition.dims]
        for dim in axes:
            if dim not in decomposition.coords:
                raise ValueError("fixed factor decompositions require labeled trait axes")
            decomposition = decomposition.assign_coords({dim: decomposition[dim].values.astype(str)}).sel({dim: Y.columns})
    if isinstance(covariance, pd.DataFrame):
        covariance = covariance.copy()
        covariance.index = covariance.index.astype(str); covariance.columns = covariance.columns.astype(str)
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise ValueError("fixed covariance trait labels must be unique")
        covariance = covariance.loc[Y.columns, Y.columns]
    elif loco_factor == "fixed" and decomposition is not None and not hasattr(decomposition, "dims"):
        raise TypeError("fixed decomposition must be a labeled covariance-fit Dataset; use covariance= for an explicit positional matrix")
    nfac = min(10, Y.shape[1]) if factors is None else factors
    if isinstance(nfac, bool) or int(nfac) != nfac or nfac < 1 or nfac > Y.shape[1]:
        raise ValueError("factors must be an integer in [1, number of traits]")
    nfac = int(nfac)
    labels = list(dict.fromkeys(vm.chromosome.astype(str))) if chromosomes is None else list(map(str, chromosomes))
    if not set(labels).issubset(set(vm.chromosome.astype(str))):
        raise ValueError("requested exclusion group absent from variants")
    if len(set(labels)) != len(labels):
        raise ValueError("requested exclusion groups must be unique")
    if loco_factor == "fixed":
        warnings.warn("fixed factor axes may include the tested chromosome; sensitivity analysis only", RuntimeWarning, stacklevel=2)
        if decomposition is None and covariance is None:
            raise ValueError("fixed mode requires decomposition or covariance")
    def blocks():
        try:
            for group in labels:
                K = background_for_group(source, vm, Y.index, group, kernels=GRM,
                    background_kernels=background_kernels, background=background, block_size=block_size,
                    compute_dtype=dict(null_solver_kwargs or {}).get('work_dtype', 'float32'))
                kw = dict(covariance_kwargs or {})
                kw.setdefault("rank_G", covariance_rank_G); kw.setdefault("rank_E", covariance_rank_E)
                kw.setdefault("n_boot", 0); kw.setdefault("random_state", int(random_state))
                kw.setdefault("progress", False)
                if loco_factor == "refit":
                    kw["warm_start"] = None
                fitY = Y.copy()
                if covariates is not None:
                    C = covariates.to_numpy(float)
                    if C.ndim == 1:
                        C = C[:, None]
                    if C.shape[0] != len(Y) or not np.isfinite(C).all():
                        raise ValueError("covariates must be finite and sample aligned")
                    C = np.column_stack([np.ones(len(Y)), C])
                    for col in Y:
                        mask = Y[col].notna().to_numpy()
                        fitY.loc[mask, col] -= C[mask] @ np.linalg.lstsq(C[mask], Y.loc[mask, col], rcond=None)[0]
                fit = mv._genetic_varcov_PSD_raw(K, fitY, **kw)
                G, E = covariance_matrices(fit)
                view = mv._result_view(fit)
                axes = mv._result_view(decomposition) if loco_factor == "fixed" and decomposition is not None else view
                if factor_method == "native":
                    L, psi, _ = _native_loadings_from_fit(axes, component, nfac)
                else:
                    if loco_factor == "fixed" and covariance is not None:
                        Caxis = covariance
                    elif component.lower() in {"g", "genetic"}:
                        Caxis = axes["G"]
                    elif component.lower() in {"e", "residual", "environmental"}:
                        Caxis = axes["E"]
                    else:
                        Caxis = axes["G_components"][component]
                    L, psi, _ = _loadings_from_component(Caxis, nfac, method=factor_method, standardize=factor_standardize)
                if L.shape[1] != nfac:
                    raise ValueError("a LOCO fit has fewer factor axes than requested; lower factors explicitly")
                if projection_out is not None:
                    key = str(component).lower()
                    nuisance = np.diag(np.maximum(psi, 0.0))
                    if key in {'e', 'residual', 'environmental'}:
                        nuisance += np.asarray(view['G'])
                    else:
                        nuisance += np.asarray(view['E'])
                        if key not in {'g', 'genetic'}:
                            components = view.get('G_components', {})
                            nuisance += sum((np.asarray(value) for name, value in components.items()
                                             if str(name).lower() != key), np.zeros_like(nuisance))
                    projected, _ = _factor_scores_gls(fitY, L, nuisance, method='posterior')
                    projected.columns = [f'factor{i+1}' for i in range(nfac)]
                    _write_factor_state(projection_out, group, fitY, projected, L, psi,
                                        nuisance, 'posterior_exploratory_separate_from_score_null')
                null = prepare_score_null(Y, K, G, E, covariates=covariates, **dict(null_solver_kwargs or {}))
                basis = pd.DataFrame(L, index=Y.columns, columns=[f"factor{i+1}" for i in range(nfac)])
                for X, meta in genotype_blocks(source, vm, Y.index, group, block_size=block_size):
                    if marker_coding == "dominance":
                        a = X.to_numpy(np.float32)
                        finite = np.isfinite(a)
                        if np.any(finite & (a != np.round(a))):
                            raise ValueError("dominance requires hardcalls")
                        X = pd.DataFrame(np.where(finite, a == 1, np.nan), index=X.index, columns=X.columns)
                    ds = score_block(null, X, basis=basis)
                    if heterogeneity:
                        full = score_block(null, X)
                        dof = full.omnibus_dof.values - ds.omnibus_dof.values
                        stat = full.omnibus_stat.values - ds.omnibus_stat.values
                        tolerance = 1e-5 * np.maximum(full.omnibus_stat.values, 1)
                        if np.any(stat < -tolerance):
                            raise RuntimeError("nested heterogeneity score exceeds numerical tolerance")
                        stat = np.where(dof > 0, np.maximum(stat, 0), np.nan)
                        logp = chi2_logsf(stat, dof)
                        ds["q_snp"] = ("variant", stat)
                        ds["q_snp_dof"] = ("variant", dof)
                        ds["q_snp_log_p"] = ("variant", logp)
                        ds["q_snp_neglog_p"] = ("variant", -logp / np.log(10))
                    ds = attach_variant_metadata(ds, meta, group=group)
                    ds.attrs.update(component=str(component), loco_factor=loco_factor,
                        factor_method=factor_method, marker_coding=marker_coding,
                        scan_group_metadata={str(group): {"factor_loadings": L.tolist(),
                            "factor_specific_variance": psi.tolist(), "factor_trait_names": list(Y.columns)}},
                        heterogeneity="nested full-trait minus factor score" if heterogeneity else "not requested")
                    yield ds
        finally:
            if not hasattr(genotypes, "iter_blocks") and hasattr(source, "close"):
                source.close()
    result = finish_scan(blocks(), out=out, max_return_rows=max_return_rows,
                         variant_chunk=variant_chunk, shard_variants=shard_variants)
    if projection_out is not None:
        from .store import finalize_zarr_artifact
        finalize_zarr_artifact(projection_out, artifact_type='exploratory_factor_projection')
        result.attrs['projection_path'] = str(projection_out)
        if out is not None:
            import zarr
            association_path=str(out) if str(out).endswith('.zarr') else str(out)+'.zarr'
            group=zarr.open_group(association_path,mode='a',use_consolidated=False)
            group.attrs['projection_path']=str(projection_out)
            group.attrs['projection_scope']='exploratory posterior targets; separate from exact association null'
            zarr.consolidate_metadata(association_path)
    return result


__all__ = ["factorGWAS"]
