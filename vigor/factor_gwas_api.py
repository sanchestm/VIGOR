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


def _residualize(Y, covariates):
    """Least-squares covariate adjustment per trait on its observed cells."""
    fitY = Y.copy()
    if covariates is None:
        return fitY
    C = covariates.to_numpy(float)
    if C.ndim == 1:
        C = C[:, None]
    if C.shape[0] != len(Y) or not np.isfinite(C).all():
        raise ValueError("covariates must be finite and sample aligned")
    C = np.column_stack([np.ones(len(Y)), C])
    for col in Y:
        mask = Y[col].notna().to_numpy()
        fitY.loc[mask, col] -= C[mask] @ np.linalg.lstsq(C[mask], Y.loc[mask, col], rcond=None)[0]
    return fitY


def _group_dual(fitY, K, G, E, null_solver_kwargs):
    """Observed-space mvBLUP dual V^-1(y - mu) for one background, animals x traits.

    Zero on unobserved cells. ``A @ L`` gives the exact factor-score directions:
    the score of loading vector l for a centered genotype x is x'(A l).
    """
    opts = dict(null_solver_kwargs or {})
    names = fitY.columns
    frame = lambda m: pd.DataFrame(np.asarray(m, float), index=names, columns=names)
    Gf = {k: frame(v) for k, v in G.items()} if isinstance(G, dict) else frame(G)
    state = mv._mvPredict_raw(fitY, K, Gf, frame(E), return_state=True,
                              solver_tol=float(opts.get("solver_tol", 1e-6)),
                              solver_maxiter=int(opts.get("solver_maxiter", 1000)), progress=False)
    return state["A"].reindex(index=fitY.index, columns=names).to_numpy(float)


def _grammar_gamma(null, calibration, active, sigma, L):
    """Per-factor gamma = median(exact information / (x'x sigma_kk)) over calibration variants."""
    from .gwas.exact import score_design_block
    if len(calibration) < 10:
        warnings.warn(f"only {len(calibration)} polymorphic calibration variants; gamma set to 1",
                      RuntimeWarning, stacklevel=3)
        return np.ones(L.shape[1]), len(calibration)
    X = np.column_stack(calibration).astype(np.float64)
    exact = score_design_block(null, X[:, :, None], basis=L)
    info = 1.0 / exact["beta_se"].values ** 2
    a = X[active]
    finite = np.isfinite(a)
    counts = finite.sum(axis=0)
    mean = np.divide(np.where(finite, a, 0.0).sum(axis=0), counts, out=np.zeros(a.shape[1]), where=counts > 0)
    xc = np.where(finite, a - mean, 0.0)
    ols = np.einsum("ij,ij->j", xc, xc)[:, None] * np.diag(sigma)[None, :]
    ratio = np.where(np.isfinite(info) & (ols > 0), info / np.where(ols > 0, ols, 1.0), np.nan)
    gamma = np.nanmedian(np.where(ratio > 0, ratio, np.nan), axis=0)
    return np.where(np.isfinite(gamma) & (gamma > 0), gamma, 1.0), X.shape[1]


def _factor_dataset(result, omnibus, meta, cols):
    """Variant-by-factor Dataset with the exact engine's schema."""
    import xarray as xr
    from . import __version__
    data = {name: (("variant", "trait"), value) for name, value in result.items()}
    data.update({f"omnibus_{name}": (("variant",), value) for name, value in omnibus.items()})
    ds = xr.Dataset(data, coords={"variant": np.arange(len(meta), dtype=np.int64),
                                  "trait": np.asarray(cols, str),
                                  "variant_id": ("variant", meta.variant_id.astype(str).to_numpy())},
                    attrs={"vigor_version": __version__,
                           "coordinate_test": "one-direction score; all other tested effects fixed to zero",
                           "null_covariance_uncertainty": "conditional on supplied G/E estimates"})
    for name in ("log_p", "conditional_log_p", "omnibus_log_p"):
        ds[name].attrs["units"] = "natural log probability"
    for name in ("neglog_p", "conditional_neglog_p", "omnibus_neglog_p"):
        ds[name].attrs["units"] = "-log10 probability"
    return ds


def factorGWAS(traits, GRM=None, decomposition=None, genotypes=None, *,
               component="G", covariance=None, factors=None, background="loco",
               method="grammar_gamma", score_method="direct", heterogeneity=None,
               loco_factor=None, covariance_rank_G=100, covariance_rank_E=50,
               covariance_kwargs=None, factor_method="native", factor_standardize=False,
               marker_coding="additive", groups=None, background_kernels=None,
               block_size=128, chromosomes=None, random_state=0,
               null_solver_kwargs=None, covariates=None, out=None, projection_out=None,
               max_return_rows=2_000_000, variant_chunk=1024, shard_variants=16384,
               maxmem=None, gamma_snps=200, gamma_scope="auto", rescore_p=1e-5,
               rescore_max=5000, **retired):
    """Test LOCO factor effect directions in the full-trait Gaussian null.

    The default ``method='grammar_gamma'`` is the fast genome-wide mode. Per
    exclusion group it computes the observed-space mvBLUP dual A = V^-1(y - mu)
    with the LOCO kernel, so the factor scores u = x'(A L) are exact (including
    missing phenotype cells); only the information x'Px is approximated by
    x'x * S, where S is the factor-score covariance rescaled per factor by
    gamma = median(exact information / x'x S_kk) over ``gamma_snps`` strided
    variants. Variants whose approximate joint test passes ``rescore_p`` are
    rescored with the exact engine (``exact_rescored`` marks them). Covariance
    and axes are estimated once (``loco_factor='fixed'``) unless 'refit' is set.
    ``method='exact'`` scores every variant with the exact engine.

    Parameters
    ----------
    traits : DataFrame (n,t)
        Labeled Gaussian phenotypes with NaN missing cells.
    GRM : kernel, named kernels, or canonical chromosome store, optional
        Supplied background; group-keyed inputs should use background_kernels.
    decomposition, covariance : fitted result or matrix, optional
        Fitted model whose axes are tested with loco_factor='fixed'. In the
        grammar_gamma mode a supplied decomposition also provides G and E, so no
        covariance is fitted at all; otherwise one fit on the first LOCO kernel
        is reused for every group.
    genotypes : DataFrame, source or path
        Sample-by-variant counted-allele dosage.
    component : str
        Genetic/residual or named covariance component defining factor directions.
    factors : positive int, optional
        Number of tested axes, default min(10,t). Rank must stay equal across groups.
    background : {'loco','provided'}
        Construct leave-group-out backgrounds or require supplied kernels.
    method : {'grammar_gamma', 'exact'}
        'grammar_gamma' (default): exact scores, gamma-calibrated information and
        exact rescoring of hits. 'exact': exact information for every variant.
    score_method : {'direct','subspace'}
        Restrict beta_trait=L beta_factor in the full original-trait null.
    heterogeneity : bool, optional
        Also report nested full-trait minus factor-subspace score statistic.
        This requires full-trait exact information and can be expensive. Default
        True for 'exact'; False for 'grammar_gamma', where True reports it for the
        exact-rescored variants only.
    loco_factor : {'refit','fixed'}, optional
        'refit' re-estimates covariance and axes per excluded chromosome. 'fixed'
        keeps one set of axes. Default 'fixed' for 'grammar_gamma' (one covariance
        fit) and 'refit' for 'exact'. With 'exact' + 'fixed' the null covariance
        is still fitted per exclusion group (sensitivity analysis).
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
    gamma_snps : positive int
        Variants per group, taken by stride, used to estimate gamma. Default 200.
    gamma_scope : {'auto', 'group', 'first'}
        Estimate gamma per group ('group'), once on the first group ('first'),
        or per group only when the exact information is cheap, i.e. complete
        phenotypes and one dense kernel ('auto').
    rescore_p : float
        Approximate joint p-value below which variants are rescored exactly.
    rescore_max : positive int
        Maximum candidate genotype columns cached per group for rescoring.
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
    grammar_gamma: one mvBLUP solve per group, one O(n m r) GEMM per block and
    exact scoring of ``gamma_snps`` plus the rescored variants; per-group results
    (m x r) are held until gamma is known. exact: per-group covariance fit plus
    exact information for every variant (O(m r) solves, or one GEMM per block
    for complete phenotypes and one dense kernel); heterogeneity adds O(m t).
    """
    from .gwas.scan import (scan_inputs, background_for_group, genotype_blocks,
                            attach_variant_metadata, finish_scan, covariance_matrices,
                            scan_covariates)
    from .gwas.exact import (prepare_score_null, score_block, score_design_block, chi2_logsf,
                             _finalize_scores, _spectral_state)
    from pathlib import Path
    if retired:
        raise TypeError("retired factorGWAS controls: " + ", ".join(sorted(retired)) +
                        "; see docs/ROUND3_MIGRATION.md")
    if method not in {"grammar_gamma", "exact"} or score_method not in {"direct", "subspace"}:
        raise ValueError("method must be 'grammar_gamma' or 'exact' and score_method 'direct'")
    fast = method == "grammar_gamma"
    if loco_factor is None:
        loco_factor = "fixed" if fast else "refit"
    if heterogeneity is None:
        heterogeneity = not fast
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
    if gamma_scope not in {"auto", "group", "first"}:
        raise ValueError("gamma_scope must be 'auto', 'group' or 'first'")
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
    if loco_factor == "fixed" and not fast:
        warnings.warn("fixed factor axes may include the tested chromosome; sensitivity analysis only", RuntimeWarning, stacklevel=2)
        if decomposition is None and covariance is None:
            raise ValueError("fixed mode requires decomposition or covariance")
    # Covariate-adjusted phenotypes for covariance fitting and the fast dual; the
    # exact null keeps covariates as GLS fixed effects instead.
    fitY = _residualize(Y, covariates)
    kw = dict(covariance_kwargs or {})
    kw.setdefault("rank_G", covariance_rank_G); kw.setdefault("rank_E", covariance_rank_E)
    kw.setdefault("n_boot", 0); kw.setdefault("random_state", int(random_state))
    kw.setdefault("progress", False)
    if loco_factor == "refit":
        kw["warm_start"] = None
    shared = {}

    def model_for_group(K):
        """Null covariance (G, E), factor loadings and the fit view for one exclusion group."""
        if loco_factor == "fixed" and "model" in shared:
            return shared["model"]
        if loco_factor == "fixed" and fast and decomposition is not None:
            fit = decomposition   # one supplied fit: G, E and axes for every group
        else:
            fit = mv._genetic_varcov_PSD_raw(K, fitY, **kw)
        G, E = covariance_matrices(fit)
        view = mv._result_view(fit)
        axes_view = mv._result_view(decomposition) if loco_factor == "fixed" and decomposition is not None else view
        if factor_method == "native":
            L, psi, _ = _native_loadings_from_fit(axes_view, component, nfac)
        else:
            if loco_factor == "fixed" and covariance is not None:
                Caxis = covariance
            elif component.lower() in {"g", "genetic"}:
                Caxis = axes_view["G"]
            elif component.lower() in {"e", "residual", "environmental"}:
                Caxis = axes_view["E"]
            else:
                Caxis = axes_view["G_components"][component]
            L, psi, _ = _loadings_from_component(Caxis, nfac, method=factor_method, standardize=factor_standardize)
        if L.shape[1] != nfac:
            raise ValueError("a LOCO fit has fewer factor axes than requested; lower factors explicitly")
        model = (G, E, L, psi, view)
        if loco_factor == "fixed":
            shared["model"] = model
        return model

    def coded(X):
        if marker_coding != "dominance":
            return X
        a = X.to_numpy(np.float32)
        finite = np.isfinite(a)
        if np.any(finite & (a != np.round(a))):
            raise ValueError("dominance requires hardcalls")
        return pd.DataFrame(np.where(finite, a == 1, np.nan), index=X.index, columns=X.columns)

    def group_attrs(ds, group, L, psi, extra=None):
        ds.attrs.update(component=str(component), loco_factor=loco_factor, method=method,
            factor_method=factor_method, marker_coding=marker_coding,
            scan_group_metadata={str(group): {"factor_loadings": L.tolist(),
                "factor_specific_variance": psi.tolist(), "factor_trait_names": list(Y.columns),
                **(extra or {})}},
            heterogeneity=("nested full-trait minus factor score" + (" (exact-rescored variants only)" if fast else ""))
                          if heterogeneity else "not requested")
        return ds

    def add_heterogeneity(ds, full):
        dof = full.omnibus_dof.values - ds.omnibus_dof.values
        stat = full.omnibus_stat.values - ds.omnibus_stat.values
        tolerance = 1e-5 * np.maximum(full.omnibus_stat.values, 1)
        if np.any(stat < -tolerance):
            raise RuntimeError("nested heterogeneity score exceeds numerical tolerance")
        stat = np.where(dof > 0, np.maximum(stat, 0), np.nan)
        return stat, dof, chi2_logsf(stat, dof)

    def projection_state(group, L, psi, view):
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

    def exact_group(group, K, G, E, L, psi):
        null = prepare_score_null(Y, K, G, E, covariates=covariates, **dict(null_solver_kwargs or {}))
        basis = pd.DataFrame(L, index=Y.columns, columns=[f"factor{i+1}" for i in range(nfac)])
        for X, meta in genotype_blocks(source, vm, Y.index, group, block_size=block_size):
            X = coded(X)
            ds = score_block(null, X, basis=basis)
            if heterogeneity:
                stat, dof, logp = add_heterogeneity(ds, score_block(null, X))
                ds["q_snp"] = ("variant", stat)
                ds["q_snp_dof"] = ("variant", dof)
                ds["q_snp_log_p"] = ("variant", logp)
                ds["q_snp_neglog_p"] = ("variant", -logp / np.log(10))
            yield group_attrs(attach_variant_metadata(ds, meta, group=group), group, L, psi)

    def grammar_group(group, K, G, E, L, psi):
        """GRAMMAR-Gamma factor scan of one exclusion group (see factorGWAS Notes)."""
        names = pd.Index(Y.columns)
        cols = [f"factor{i+1}" for i in range(nfac)]
        dual = _group_dual(fitY, K, G, E, null_solver_kwargs)
        active = Y.notna().any(axis=1).to_numpy()
        observed = Y.notna().to_numpy()[active].astype(float)
        Z = dual[active] @ L                                     # exact factor score directions
        Zc = Z - Z.mean(axis=0)
        sigma = Zc.T @ Zc / max(int(active.sum()), 1)
        sigma_inv = np.linalg.pinv(sigma, rcond=1e-10)
        loaded = (L != 0).astype(float)
        n_group = int((vm.chromosome.astype(str).to_numpy() == str(group)).sum())
        stride = max(1, n_group // max(int(gamma_snps), 1))
        stored, calibration, candidates = [], [], []
        seen = 0
        for X, meta in genotype_blocks(source, vm, Y.index, group, block_size=block_size):
            X = coded(X)
            raw = X.to_numpy(np.float32)
            a = raw[active].astype(np.float64)
            finite = np.isfinite(a)
            counts = finite.sum(axis=0)
            mean = np.divide(np.where(finite, a, 0.0).sum(axis=0), counts, out=np.zeros(a.shape[1]), where=counts > 0)
            xc = np.where(finite, a - mean, 0.0)
            u = xc.T @ Zc
            xx = np.einsum("ij,ij->j", xc, xc)
            nobs = (finite.astype(float).T @ observed) @ loaded
            pick = np.flatnonzero(((seen + np.arange(a.shape[1])) % stride == 0) & (xx > 0))
            for j in pick[:max(int(gamma_snps) - len(calibration), 0)]:
                calibration.append(raw[:, j].copy())
            # Provisional statistic with gamma = 1 keeps a bounded cache of
            # candidate genotypes for exact rescoring once gamma is known.
            provisional = np.einsum("mi,ij,mj->m", u, sigma_inv, u) / np.where(xx > 0, xx, np.inf)
            for j in np.flatnonzero(provisional >= rescore_cache_stat):
                candidates.append((provisional[j], len(stored), j, raw[:, j].copy()))
            if len(candidates) > rescore_max:
                candidates.sort(key=lambda c: -c[0])
                del candidates[rescore_max:]
            stored.append((meta, u, xx, nobs))
            seen += a.shape[1]
        if not stored:
            return
        # gamma: per group when exact information is cheap (spectral null); otherwise
        # estimated on the first group and reused (it varies little across LOCO groups).
        null = None
        if "gamma" in shared:
            gamma, gamma_n = shared["gamma"], shared["gamma_n"]
        else:
            null = prepare_score_null(Y, K, G, E, covariates=covariates, **dict(null_solver_kwargs or {}))
            gamma, gamma_n = _grammar_gamma(null, calibration, active, sigma, L)
            if gamma_scope == "first" or (gamma_scope == "auto" and _spectral_state(null) is None):
                shared.update(gamma=gamma, gamma_n=gamma_n)
        root = np.sqrt(gamma)
        S = sigma * root[:, None] * root[None, :]
        Sinv = np.linalg.pinv(S, rcond=1e-10)
        threshold = rescore_stat_threshold(nfac)
        rescore = {}
        final = []
        for _, bi, row, column in candidates:
            u_row, xx_row = stored[bi][1][row], stored[bi][2][row]
            if xx_row > 0 and float(u_row @ Sinv @ u_row) / xx_row >= threshold:
                final.append((bi, row, column))
        if final:
            if null is None:
                null = prepare_score_null(Y, K, G, E, covariates=covariates, **dict(null_solver_kwargs or {}))
            Xh = np.column_stack([c[2] for c in final]).astype(np.float64)
            exact = score_design_block(null, Xh[:, :, None], basis=pd.DataFrame(L, index=names, columns=cols))
            full = score_design_block(null, Xh[:, :, None]) if heterogeneity else None
            for k, (bi, row, _) in enumerate(final):
                rescore.setdefault(bi, []).append((row, k))
        for bi, (meta, u, xx, nobs) in enumerate(stored):
            info = xx[:, None, None] * S[None]
            result, omnibus = _finalize_scores(u, info, nobs)
            ds = _factor_dataset(result, omnibus, meta, cols)
            ds["exact_rescored"] = ("variant", np.zeros(len(meta), bool))
            if heterogeneity:
                for name in ("q_snp", "q_snp_dof", "q_snp_log_p", "q_snp_neglog_p"):
                    ds[name] = ("variant", np.full(len(meta), np.nan))
            for row, k in rescore.get(bi, []):
                for name in ds.data_vars:
                    if name in exact.data_vars:
                        ds[name].values[row] = exact[name].values[k]
                ds["exact_rescored"].values[row] = True
                if heterogeneity:
                    stat, dof, logp = add_heterogeneity(exact.isel(variant=[k]), full.isel(variant=[k]))
                    ds["q_snp"].values[row] = stat[0]; ds["q_snp_dof"].values[row] = dof[0]
                    ds["q_snp_log_p"].values[row] = logp[0]; ds["q_snp_neglog_p"].values[row] = -logp[0] / np.log(10)
            ds.attrs.update(engine="grammar_gamma_factor_score",
                            information="x'x * S, S = factor-score covariance rescaled per factor by gamma "
                                        "(median exact/OLS information ratio); variants above the rescore "
                                        "threshold use the exact engine")
            yield group_attrs(attach_variant_metadata(ds, meta, group=group), group, L, psi,
                              {"gamma": np.asarray(gamma).tolist(), "gamma_snps": int(gamma_n),
                               "rescore_p": float(rescore_p),
                               "exact_rescored": int(sum(len(v) for v in rescore.values()))})

    def rescore_stat_threshold(dof):
        from scipy.stats import chi2
        return float(chi2.isf(rescore_p, dof))

    rescore_cache_stat = None

    def blocks():
        nonlocal rescore_cache_stat
        from scipy.stats import chi2
        # Cache candidates generously (gamma is not yet known); rescore the strict set.
        rescore_cache_stat = float(chi2.isf(min(rescore_p * 1e3, 1.0), nfac))
        try:
            for group in labels:
                K = background_for_group(source, vm, Y.index, group, kernels=GRM,
                    background_kernels=background_kernels, background=background, block_size=block_size,
                    compute_dtype=dict(null_solver_kwargs or {}).get('work_dtype', 'float32'))
                G, E, L, psi, view = model_for_group(K)
                if projection_out is not None:
                    projection_state(group, L, psi, view)
                yield from (grammar_group if fast else exact_group)(group, K, G, E, L, psi)
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
