"""LOCO factor effect-subspace scans through the exact multivariate score core."""
import warnings
from collections.abc import Mapping
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


def _group_plan(fitY, K, G, E, null_solver_kwargs):
    """Prepared observed-space mvBLUP plan for one background (same settings as mvPredict).

    The plan is reused for the real dual and every null-trait dual, so the
    operator and preconditioner are built once per exclusion group.
    """
    names = fitY.columns
    frame = lambda m: pd.DataFrame(np.asarray(m, float), index=names, columns=names)
    Gf = {k: frame(v) for k, v in G.items()} if isinstance(G, dict) else frame(G)
    accuracy = "fast" if len(fitY) >= 1024 else "exact"
    return mv.prepare_mvBLUP_plan(fitY, K, Gf, frame(E), center=True, accuracy=accuracy,
                                  kernel_rank="full", progress=False)


def _plan_dual(plan, frame, null_solver_kwargs, *, check_against_plan=False):
    """Dual V^-1(y - ybar) on observed cells (animals x traits, zero elsewhere).

    ``frame`` must have the plan's observed-cell pattern. Cells follow the plan's
    sample-major ``flat`` order, the order of its operator and solution.
    """
    opts = dict(null_solver_kwargs or {})
    tol = float(opts.get("solver_tol", 1e-6))
    Yv = frame.to_numpy(float)
    means = np.nanmean(Yv, axis=0)
    Yv = Yv - np.where(np.isfinite(means), means, 0.0)[None, :]
    flat = plan.obs_plan["flat"]
    rhs = np.nan_to_num(Yv.ravel()[flat])
    if check_against_plan and "y" in plan.obs_plan and not np.allclose(np.asarray(plan.obs_plan["y"], float), rhs,
                                               rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(rhs).max()))):
        raise RuntimeError("mvBLUP plan cell order differs from the flat (animal, trait) order")
    alpha, info = plan.solve(y=rhs, rtol=tol, maxiter=int(opts.get("solver_maxiter", 1000)))
    if info["cg_info"] != 0 or info["cg_relative_residual"] > max(20 * tol, 2e-5):
        raise RuntimeError(f"factorGWAS null PCG did not converge: {info}")
    A = np.zeros(plan.n * plan.t)
    A[flat] = alpha
    return A.reshape(plan.n, plan.t)


def _psd_root(M):
    """Symmetric PSD square root (handles singular trait covariances without jitter)."""
    M = np.asarray(M, float)
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def _kernel_root(K, jitter=1e-8):
    """Lower Cholesky factor (returned as float32) of a dense kernel, raising jitter until it succeeds.

    Centred GRMs are exactly singular (the 1-vector is in their null space), so the
    factorization runs in float64, where a tiny relative jitter suffices; float32
    round-off would force jitter that adds spurious iid genetic variance to the null
    draws and inflates S. A warning reports jitter above 1e-4 of the mean diagonal.
    """
    from scipy.linalg import cholesky
    base = np.asarray(getattr(K, "values", K))
    if base.ndim != 2 or base.shape[0] != base.shape[1] or base.dtype == object:
        raise ValueError("null-trait calibration needs a dense square background kernel; "
                         "materialize the GRM or use method='exact'")
    scale = max(float(np.mean(np.diag(base))), np.finfo(np.float64).tiny)
    for k in range(8):
        added = jitter * 10 ** k * scale
        work = np.array(base, dtype=np.float64, copy=True)
        work[np.diag_indices_from(work)] += added
        try:
            root = cholesky(work, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError:
            continue
        if added > 1e-4 * scale:
            warnings.warn(f"background kernel needed jitter {added / scale:.1e} x mean diagonal "
                          "for the null-trait draws; S may be slightly conservative",
                          RuntimeWarning, stacklevel=3)
        return root.astype(np.float32)
    raise np.linalg.LinAlgError("background kernel is not positive semidefinite within jitter")


def _null_trait_frames(fitY, K, G, E, covariates, reps, seed):
    """Simulated null phenotype sets y0 ~ N(0, sum_k K_k (x) G_k + I (x) E).

    Drawn on the full grid, then masked to ``fitY``'s observed cells and
    covariate-residualized exactly like the real phenotypes.
    """
    rng = np.random.default_rng(seed)
    n, t = fitY.shape
    if isinstance(G, dict):
        if not isinstance(K, Mapping):
            raise ValueError("multi-component G requires named background kernels")
        pairs = [(K[name], G[name]) for name in G]
    else:
        if isinstance(K, Mapping):
            # covariance_matrices returns a plain G for one component; unwrap a
            # one-entry named background (GRM={'genetic': K}) to match.
            if len(K) != 1:
                raise ValueError("a single genetic covariance needs exactly one background kernel")
            K = next(iter(K.values()))
        pairs = [(K, G)]
    roots = [(_kernel_root(Kk), _psd_root(Gk).astype(np.float32)) for Kk, Gk in pairs]
    Er = _psd_root(E).astype(np.float32)
    observed = fitY.notna().to_numpy()
    for _ in range(int(reps)):
        y0 = rng.standard_normal((n, t), dtype=np.float32) @ Er
        for Lk, Rg in roots:
            y0 += Lk @ (rng.standard_normal((n, t), dtype=np.float32) @ Rg)
        y0 = np.where(observed, y0.astype(np.float64), np.nan)
        yield _residualize(pd.DataFrame(y0, index=fitY.index, columns=fitY.columns), covariates)


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
               maxmem=None, null_reps=4, rescore_p=None, rescore_max=5000, **retired):
    """Test LOCO factor effect directions in the full-trait Gaussian null.

    The default ``method='grammar_gamma'`` is the fast genome-wide mode. Per
    exclusion group it computes the observed-space mvBLUP dual A = V^-1(y - mu)
    with the LOCO kernel, so the factor scores u = x'(A L) are exact (including
    missing phenotype cells); only the information x'Px is approximated by
    x'x * S. The r x r matrix S is calibrated in factor space from ``null_reps``
    simulated null trait sets y0 ~ N(0, K (x) G + I (x) E): their duals A0 are
    solved with the same prepared plan, and their factor scores u0 = x'(A0 L),
    computed in the same GEMM as the real scores, have covariance x'P_L x. So
    S = sum u0 u0' / sum x'x over every variant of the group; no exact
    information model is built. A held-out replicate gives a lambda check
    (``null_lambda`` in ``scan_group_metadata``). Optional exact rescoring of
    hits (``rescore_p``) marks ``exact_rescored``. Covariance and axes are
    estimated once (``loco_factor='fixed'``) unless 'refit' is set.
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
        'grammar_gamma' (default): exact scores and information x'x S with S
        calibrated from null-trait duals. 'exact': exact information for every
        variant.
    score_method : {'direct','subspace'}
        Restrict beta_trait=L beta_factor in the full original-trait null.
    heterogeneity : bool, optional
        Also report nested full-trait minus factor-subspace score statistic.
        This requires full-trait exact information and can be expensive. Default
        True for 'exact'; False for 'grammar_gamma', where True reports it for the
        exact-rescored variants only and therefore requires ``rescore_p``.
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
        Covariance fit seed; also seeds the null trait draws (per group).
    null_solver_kwargs : mapping, optional
        Solver options; solver_tol and solver_maxiter apply to the grammar_gamma
        mvBLUP duals, all keys to prepare_score_null (exact mode and rescoring).
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
    null_reps : int >= 1
        Simulated null trait sets per group for calibrating S (grammar_gamma).
        Each costs one extra PCG solve with the group's prepared plan and widens
        the scan GEMM by r columns. Default 4; with >= 2 one replicate is held
        out to report ``null_lambda``.
    rescore_p : float or None
        Opt-in exact rescoring: approximate joint p-value below which variants
        are rescored with the exact engine (builds an exact null per group with
        hits). Default None (no rescoring).
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
    grammar_gamma: one plan and 1 + null_reps PCG solves per group, one Cholesky
    of the LOCO kernel (O(n^3/3)) for the null draws, one O(n m r (1+null_reps))
    GEMM per block; per-group results (m x r(1+null_reps)) are held until S is
    known. Automatic LOCO kernels use one genome-wide pass plus each group's own
    variants. Rescoring, when requested, adds exact work per hit. exact: per-group covariance fit plus
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
                        ". gamma_snps/gamma_scope were replaced by null_reps (null-trait "
                        "calibration of S); exact rescoring is opt-in via rescore_p")
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
    if isinstance(null_reps, bool) or int(null_reps) != null_reps or null_reps < 1:
        raise ValueError("null_reps must be a positive integer")
    null_reps = int(null_reps)
    if rescore_p is not None and not 0 < float(rescore_p) < 1:
        raise ValueError("rescore_p must be None or in (0, 1)")
    if fast and heterogeneity and rescore_p is None:
        raise ValueError("grammar_gamma reports heterogeneity only for exact-rescored variants; "
                         "set rescore_p or use method='exact'")
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
        """GRAMMAR-Gamma factor scan of one exclusion group with null-trait calibration of S."""
        from scipy.stats import chi2
        names = pd.Index(Y.columns)
        cols = [f"factor{i+1}" for i in range(nfac)]
        plan = _group_plan(fitY, K, G, E, null_solver_kwargs)
        dual = _plan_dual(plan, fitY, null_solver_kwargs, check_against_plan=True)
        seed = np.random.SeedSequence([int(random_state), labels.index(group)])
        null_duals = [_plan_dual(plan, y0, null_solver_kwargs)
                      for y0 in _null_trait_frames(fitY, K, G, E, covariates, null_reps, seed)]
        active = Y.notna().any(axis=1).to_numpy()
        # Per-factor observation counts: (finite' observed) loaded == finite' (observed loaded),
        # an n x r product instead of an n x t one.
        obs_loaded = (Y.notna().to_numpy()[active].astype(np.float32)
                      @ (L != 0).astype(np.float32))
        # Real and null factor-score directions share one GEMM per block.
        Zall = np.concatenate([d[active] @ L for d in [dual, *null_duals]], axis=1)
        Zall -= Zall.mean(axis=0)
        Zc = Zall[:, :nfac]
        Zall32 = np.asarray(Zall, np.float32)
        sigma = Zc.T @ Zc / max(int(active.sum()), 1)
        sigma_inv = np.linalg.pinv(sigma, rcond=1e-10)
        R = null_reps
        num = np.zeros((R, nfac, nfac))
        den = 0.0
        stored, candidates = [], []
        for X, meta in genotype_blocks(source, vm, Y.index, group, block_size=block_size):
            X = coded(X)
            raw = X.to_numpy(np.float32)
            a = raw[active]
            finite = np.isfinite(a)
            counts = finite.sum(axis=0)
            mean = np.divide(np.where(finite, a, 0).sum(axis=0, dtype=np.float64), counts,
                             out=np.zeros(a.shape[1]), where=counts > 0).astype(np.float32)
            xc = np.where(finite, a - mean[None, :], np.float32(0))
            scores = np.asarray(xc.T @ Zall32, np.float64)
            u = scores[:, :nfac]
            u0 = scores[:, nfac:].reshape(len(meta), R, nfac)
            xx = np.einsum("ij,ij->j", xc, xc, dtype=np.float64)
            nobs = np.asarray(finite.T.astype(np.float32) @ obs_loaded, np.float64)
            good = xx > 0
            num += np.einsum("mki,mkj->kij", u0[good], u0[good], optimize=True)
            den += float(xx[good].sum())
            if rescore_p is not None:
                # Provisional statistic keeps a bounded candidate cache. The running S
                # estimate (num/den so far) is on the final scale; sigma = Z'Z/n is only a
                # fallback before any polymorphic variant, since LOCO usually makes S < sigma.
                prov_inv = (np.linalg.pinv(num.mean(axis=0) / den, rcond=1e-10)
                            if den > 0 else sigma_inv)
                provisional = np.einsum("mi,ij,mj->m", u, prov_inv, u) / np.where(good, xx, np.inf)
                for j in np.flatnonzero(provisional >= rescore_cache_stat):
                    candidates.append((provisional[j], len(stored), j, raw[:, j].copy()))
                if len(candidates) > rescore_max:
                    candidates.sort(key=lambda c: -c[0])
                    del candidates[rescore_max:]
            stored.append((meta, u, xx, nobs, u0))
        if not stored:
            return
        if den <= 0:
            # Every variant is monomorphic among the analysed animals: x'x = 0 gives
            # rank-0, NaN statistics; one such group must not abort the whole scan.
            warnings.warn(f"factorGWAS group {group!r} has no polymorphic variants; its "
                          "statistics are NaN", RuntimeWarning, stacklevel=2)
            S_reps = np.zeros_like(num)
        else:
            S_reps = num / den
        S = S_reps.mean(axis=0)
        Sinv = np.linalg.pinv(S, rcond=1e-10)
        # Held-out check: S from the other replicates, lambda on the left-out one.
        lambdas = []
        if R >= 2 and den > 0:
            for k in range(R):
                S_out = np.delete(S_reps, k, axis=0).mean(axis=0)
                w = np.linalg.eigvalsh(S_out)
                rank = max(1, int(np.sum(w > w.max() * 1e-10)))
                Si = np.linalg.pinv(S_out, rcond=1e-10)
                Q = np.concatenate([np.einsum("mi,ij,mj->m", u0b[xxb > 0, k], Si, u0b[xxb > 0, k]) / xxb[xxb > 0]
                                    for _, _, xxb, _, u0b in stored])
                lambdas.append(float(np.median(Q) / chi2.ppf(.5, rank)))
        rescore = {}
        if rescore_p is not None:
            threshold = rescore_stat_threshold(nfac)
            final = [(bi, row, column) for _, bi, row, column in candidates
                     if stored[bi][2][row] > 0
                     and float(stored[bi][1][row] @ Sinv @ stored[bi][1][row]) / stored[bi][2][row] >= threshold]
            if final:
                null = prepare_score_null(Y, K, G, E, covariates=covariates, **dict(null_solver_kwargs or {}))
                Xh = np.column_stack([c[2] for c in final]).astype(np.float64)
                exact = score_design_block(null, Xh[:, :, None], basis=pd.DataFrame(L, index=names, columns=cols))
                full = score_design_block(null, Xh[:, :, None]) if heterogeneity else None
                for k, (bi, row, _) in enumerate(final):
                    rescore.setdefault(bi, []).append((row, k))
        extra = {"calibration": "null-trait duals in factor space", "null_reps": int(R),
                 "S_diagonal": np.diag(S).tolist(),
                 "null_lambda": float(np.median(lambdas)) if lambdas else None,
                 "null_lambda_by_rep": lambdas,
                 "rescore_p": None if rescore_p is None else float(rescore_p),
                 "exact_rescored": int(sum(len(v) for v in rescore.values()))}
        for bi, (meta, u, xx, nobs, _) in enumerate(stored):
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
                            information="x'x * S; S = sum u0 u0' / sum x'x over the group's variants, "
                                        "u0 = null-trait factor scores with covariance x'P_L x")
            yield group_attrs(attach_variant_metadata(ds, meta, group=group), group, L, psi, extra)

    def rescore_stat_threshold(dof):
        from scipy.stats import chi2
        return float(chi2.isf(rescore_p, dof))

    rescore_cache_stat = None
    # One genome pass builds every automatic LOCO kernel when several groups are scanned.
    loco_cache = {} if len(labels) > 1 else None

    def blocks():
        nonlocal rescore_cache_stat
        from scipy.stats import chi2
        # Cache candidates generously (S is not yet known); rescore the strict set.
        if rescore_p is not None:
            rescore_cache_stat = float(chi2.isf(min(float(rescore_p) * 1e3, 1.0), nfac))
        try:
            for group in labels:
                K = background_for_group(source, vm, Y.index, group, kernels=GRM,
                    background_kernels=background_kernels, background=background, block_size=block_size,
                    compute_dtype=dict(null_solver_kwargs or {}).get('work_dtype', 'float32'),
                    loco_cache=loco_cache)
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
