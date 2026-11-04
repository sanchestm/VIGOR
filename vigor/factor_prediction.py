"""GRM-aware posterior animal genetic-factor embeddings.

Native factor scores are conditional expectations, not causal factors.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits
from .multivariate import prepare_mvBLUP_plan, _as_pandas_input, _call_parameters, _result_view


def _factor_prediction_dataset(out, traits, arguments):
    """Convert the mvFactorPredict mapping into a labeled ('sample','factor') Dataset.

    Parameters
    ----------
    out : dict
        Historical result: per-mode score tables (one DataFrame, or one per genetic
        component), loadings per component, optional uncertainty_sd per mode, diagnostics
        and metadata.
    traits : pandas.DataFrame
        Input animals x traits whose index/columns label the sample and trait axes.
    arguments : dict
        Scalar call options recorded in ``ds.attrs['parameters']``.

    Returns
    -------
    xarray.Dataset
        ``assisted``/``genomic_only`` on ('sample','factor'), ``loadings`` on
        ('trait','factor'), optional ``uncertainty_sd`` on ('mode','sample','factor');
        the ``factor_component`` coordinate names each factor's genetic component.
        ``diagnostics`` and ``metadata`` are attributes.

    Complexity
    ----------
    TIME O(n r + t r); SPACE O(n r m + t r) for n animals, t traits, r factors, m modes.
    """
    import xarray as xr
    from .results import result_dataset

    def frame(x):
        """Concatenate component-keyed factor tables along columns (factor axis).

        Parameters
        ----------
        x : pandas.DataFrame or dict of pandas.DataFrame
            One table, or one table per genetic component sharing the row axis.

        Returns
        -------
        pandas.DataFrame
            Factor columns of all components in component order.
        """
        return pd.concat(list(x.values()), axis=1) if isinstance(x, dict) else x

    loadings = frame(out["loadings"])
    component = [str(nm) for nm, f in out["loadings"].items() for _ in f.columns]
    values, dims = {"loadings": loadings}, {"loadings": ("trait", "factor")}
    for key in ("assisted", "genomic_only"):
        if key in out:
            values[key] = frame(out[key]).reindex(columns=loadings.columns)
            dims[key] = ("sample", "factor")
    if "uncertainty_sd" in out:
        modes = list(out["uncertainty_sd"])
        stack = np.stack([frame(out["uncertainty_sd"][m]).reindex(columns=loadings.columns).to_numpy(float) for m in modes])
        values["uncertainty_sd"] = xr.DataArray(stack, dims=("mode", "sample", "factor"),
                                                coords={"mode": np.asarray(modes, dtype=str)})
    values["diagnostics"] = out["diagnostics"]
    values["metadata"] = out["metadata"]
    va = {"assisted": {"long_name": "posterior genetic factor score (phenotype-assisted)", "units": "factor SD units"},
          "genomic_only": {"long_name": "posterior genetic factor score (target phenotypes masked)", "units": "factor SD units"},
          "loadings": {"long_name": "native genetic factor loadings", "units": "trait SD units"},
          "uncertainty_sd": {"long_name": "posterior marginal SD of the genetic factor score",
                             "units": "factor SD units", "description": "Conditional on the fitted covariance."}}
    ds = result_dataset(values, dims=dims, var_attrs={k: v for k, v in va.items() if k in values},
                        title="GRM-aware posterior genetic factor scores",
                        software="vigor.factor_prediction.mvFactorPredict", parameters=_call_parameters(arguments))
    return ds.assign_coords(factor_component=("factor", np.asarray(component, dtype=str)))


def _means(traits, fit, targets, supplied, center):
    """Select training means and report the centering provenance.
    
    Parameters
    ----------
    traits : pandas.DataFrame
        Trait response table whose columns define the fitted trait order.
    fit : mapping
        Fitted covariance result potentially containing training trait_mean values.
    targets : pandas.Index or None
        Held-out animal labels excluded from fallback reference means.
    supplied : pandas.Series, array, or None
        Explicit training-only fixed mean for every fitted trait.
    center : bool or str
        False denotes already-centered response; otherwise use training means.
    
    Returns
    -------
    tuple
        Labeled mean vector and string describing the source of the means.
    
    Complexity
    ----------
    TIME O(nt); SPACE O(t) for n animals and t traits.
    """
    if center is False:
        return pd.Series(0.0, index=traits.columns), "already_centered"
    x = supplied if supplied is not None else fit.get("trait_mean", None)
    if x is not None:
        if isinstance(x, pd.Series):
            if x.index.has_duplicates or not traits.columns.isin(x.index).all():
                raise ValueError("trait means do not cover the fitted trait labels")
            x = x.reindex(traits.columns)
        else:
            x = np.asarray(x, float)
            if x.shape != (traits.shape[1],):
                raise ValueError("trait_means array shape mismatch")
            x = pd.Series(x, index=traits.columns)
        scope = "fitted"
    else:
        x = (traits if targets is None else traits.loc[~traits.index.isin(targets)]).mean()
        scope = "cohort" if targets is None else "reference_only"
        if targets is None:
            warnings.warn("Input-cohort mean: supply training means for held-out evaluation",
                          RuntimeWarning, stacklevel=3)
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("No training mean is available for one or more traits")
    return x.astype(float), scope


def _scatter(plan, alpha):
    """Scatter observed dual coefficients into a full animal-trait buffer.
    
    Parameters
    ----------
    plan : PreparedMultivariatePlan
        Prepared VIGOR conditional Gaussian model with observed cell indices.
    alpha : numpy.ndarray
        Flattened Gaussian dual coefficients ordered by observed cell indices.
    
    Returns
    -------
    numpy.ndarray
        Two-dimensional dual state with zeros on unobserved phenotype cells.
    
    Complexity
    ----------
    TIME O(nt); SPACE O(nt) for animals n and traits t.
    """
    A = np.zeros((plan.n, plan.t), dtype=plan.dtype)
    A.ravel()[plan.obs_plan["flat"]] = alpha
    return A


def _dual(plan, tol, maxiter):
    """Apply each genomic kernel to observed dual coefficients in latent space.
    
    Parameters
    ----------
    plan : PreparedMultivariatePlan
        Full fitted conditioning plan with native factor loading specifications.
    tol : float
        Relative convergence tolerance for the observed-space CG solver.
    maxiter : int
        Maximum conjugate gradient iterations before failure is reported.
    
    Returns
    -------
    tuple
        Component-specific animal factor scores and numerical convergence diagnostics.
    
    Complexity
    ----------
    TIME O(I*n*n*t+n*n*r); SPACE O(nt+nr) for dense kernels.
    """
    alpha, info = plan.solve(rtol=tol, maxiter=maxiter)
    if info["cg_info"] != 0 or info["cg_relative_residual"] > max(tol*20, 2e-5):
        raise RuntimeError("Observed-cell CG did not converge: " + repr(info))
    A = _scatter(plan, alpha)
    scores = {nm: K.matmat(A @ plan.G_specs[nm].factor)
              for nm, K in plan.kernels.items()}
    return scores, {**info, "solver_used": "observed_"+plan.accuracy}


def _latent(plan, Y, tol, maxiter):
    """Condition animal genomic factors directly with exact latent precision.
    
    Parameters
    ----------
    plan : PreparedMultivariatePlan
        Fitted model, animal kernels, and factor-residual covariance specifications.
    Y : pandas.DataFrame or numpy.ndarray
        Centered animal-by-trait response with arbitrary missing phenotype cells.
    tol : float
        Relative convergence tolerance for the latent-space conjugate gradient.
    maxiter : int
        Maximum latent CG iterations, raising an error on nonconvergence.
    
    Returns
    -------
    tuple
        Dictionary of component factor scores and validated solver convergence metadata.
    
    Complexity
    ----------
    TIME O(k*n^3 + I*k*n^2*r + n*t*r^2); SPACE O(kn^2+nr^2).
    Requires zero genetic uniqueness, positive residual specificity, and PD kernels.
    """
    names = list(plan.G_specs)
    ranks = [plan.G_specs[nm].rank for nm in names]
    if not sum(ranks):
        raise ValueError("No fitted genetic factors")
    if any(np.max(np.abs(plan.G_specs[nm].diag_specific), initial=0) > 1e-12
           for nm in names):
        raise ValueError("latent solver requires zero genetic-specific variance")
    d = np.asarray(plan.E_spec.diag_specific, float)
    if np.any(d <= 0):
        raise ValueError("latent solver requires strictly positive residual uniqueness")
    H = np.asarray(plan.E_spec.factor, float)
    L = np.concatenate([plan.G_specs[nm].factor for nm in names], axis=1)
    n, t = Y.shape
    chol = []
    offsets = np.cumsum([0]+ranks)
    slices = [slice(offsets[i], offsets[i+1]) for i in range(len(ranks))]
    for nm in names:
        K = plan.kernels[nm]
        if not hasattr(K, "K"):
            raise ValueError("latent solver requires dense full-rank GRMs")
        raw = np.asarray(K.K, dtype=float)
        try:
            chol.append(cho_factor((raw+raw.T)/2, check_finite=False))
        except np.linalg.LinAlgError as exc:
            raise ValueError("GRM is not positive definite") from exc
    r = L.shape[1]
    S = np.zeros((n,r,r), float)
    b = np.zeros((n,r), float)
    for i in range(n):
        mask = np.isfinite(Y[i])
        if not mask.any():
            continue
        Lo, Ho, di = L[mask], H[mask], d[mask]
        W = Lo/di[:, None]
        if H.shape[1]:
            VH = Ho/di[:, None]
            C = np.eye(H.shape[1]) + Ho.T@VH
            W -= VH@cho_solve(cho_factor(C, check_finite=False),
                             Ho.T@W, check_finite=False)
        S[i] = Lo.T @ W
        b[i] = Y[i, mask] @ W
    def mv(vec):
        """Apply exact factor-posterior precision to a flattened factor vector.
        
        Parameters
        ----------
        vec : numpy.ndarray
            Animal-major factor vector of length n times total genetic factor rank.
        
        Returns
        -------
        numpy.ndarray
            Precision-matrix product in the same flattened factor ordering.
        
        Complexity
        ----------
        TIME O(k*n*n*r+n*r*r); SPACE O(n*r) temporary work.
        """
        x = vec.reshape(n,r)
        z = np.einsum("ijk,ik->ij", S, x)
        for sl, cf in zip(slices, chol):
            z[:,sl] += cho_solve(cf,x[:,sl],check_finite=False)
        return z.ravel()
    op = LinearOperator((n*r,n*r),matvec=mv,dtype=float)
    # Jacobi approximation to the prior precision, exact S diagonal.
    dp = np.zeros((n,r))
    for nm, sl in zip(names,slices):
        dp[:,sl] = 1/np.maximum(plan.kernels[nm].diag()[:,None],1e-12)
    inv_diag = 1/np.maximum(dp+np.diagonal(S,axis1=1,axis2=2),1e-12)
    pre = LinearOperator(op.shape,matvec=lambda v: inv_diag.ravel()*v,dtype=float)
    nit = [0]
    def cb(_):
        """Count actual conjugate-gradient iterations for solver diagnostics.
        
        Parameters
        ----------
        _ : numpy.ndarray
            Current latent solver iteration vector; never modified by callback.
        
        Returns
        -------
        None
            Updates the enclosed iteration counter without changing solver state.
        """
        nit[0]+=1
    f, status = cg(op,b.ravel(),M=pre,rtol=tol,atol=0,maxiter=maxiter,callback=cb)
    relative = np.linalg.norm(op@f-b.ravel())/max(np.linalg.norm(b.ravel()),1e-30)
    if status or relative>max(tol*20,2e-5):
        raise RuntimeError(f"Latent CG did not converge: status={status}, relres={relative}")
    f=f.reshape(n,r)
    return {nm:f[:,sl] for nm,sl in zip(names,slices)}, {
        "solver_used":"latent_exact", "cg_info":int(status),
        "cg_iterations":int(nit[0]),"cg_relative_residual":float(relative)}


def _factor_uncertainty(plan, widths, tol, maxiter, probes, random_state):
    """Estimate posterior marginal genetic-factor standard deviations.
    
    Parameters
    ----------
    plan : PreparedMultivariatePlan
        Gaussian conditioning plan with fitted trait and animal covariance components.
    widths : mapping
        Component names mapped to counts of returned native genetic factors.
    tol : float
        Relative tolerance enforced on each posterior uncertainty linear solve.
    maxiter : int
        Maximum observed-cell CG iterations for every uncertainty linear solve.
    probes : int
        Rademacher probe count when the output exceeds exact-basis threshold.
    random_state : int
        Fixed seed allowing reproducible approximate uncertainty diagonal estimation.
    
    Returns
    -------
    tuple
        Component-specific conditional standard-deviation arrays and method metadata.
    
    Complexity
    ----------
    TIME O(Q * cost(observed CG)); SPACE O(nt+nr), Q=probes or number of cells.
    The stochastic estimate may require variance clipping; fitted covariance held fixed.
    """
    names=list(plan.G_specs)
    sizes = {nm:(plan.n,widths[nm]) for nm in names}
    prior={nm:np.broadcast_to(plan.kernels[nm].diag()[:,None],sizes[nm]).copy()
           for nm in names}
    reduce={nm:np.zeros(sizes[nm]) for nm in names}
    total=sum(a.size for a in prior.values())
    exact=total<=64
    rng=np.random.default_rng(random_state)
    niter=total if exact else probes
    for it in range(niter):
        z={nm:np.zeros(sizes[nm]) for nm in names}
        if exact:
            j=it
            for nm in names:
                if j<z[nm].size:
                    z[nm].flat[j]=1
                    break
                j-=z[nm].size
        else:
            z={nm:rng.choice([-1.,1.],size=sizes[nm]) for nm in names}
        Ctz=np.zeros((plan.n,plan.t),float)
        for nm in names:
            Ctz+=plan.kernels[nm].matmat(z[nm])@plan.G_specs[nm].factor[:,:widths[nm]].T
        w,info=plan.solve(y=Ctz.ravel()[plan.obs_plan["flat"]],
                          rtol=tol,maxiter=maxiter)
        if info["cg_info"]!=0 or info["cg_relative_residual"]>max(tol*20,2e-5):
            raise RuntimeError("Uncertainty CG did not converge")
        A=_scatter(plan,w)
        for nm in names:
            v=plan.kernels[nm].matmat(A@plan.G_specs[nm].factor)[:,:widths[nm]]
            reduce[nm]+=z[nm]*v
    if not exact:
        reduce={nm:v/probes for nm,v in reduce.items()}
    sd={nm:np.sqrt(np.clip(prior[nm]-reduce[nm],0,None)) for nm in names}
    return sd,{"method":"exact_basis" if exact else "hutchinson",
               "probes":niter,"conditional_on_fitted_covariance":True}


def mvFactorPredict(traits, GRM, covariance_result, *, n_factors=None,
                    mode="both", target_ids=None, solver="auto", accuracy="exact",
                    work_dtype="float64", preconditioner="diag", kernel_rank="full",
                    solver_tol=1e-6, solver_maxiter=5000, blas_threads=10,
                    trait_means=None, center="auto", return_uncertainty=False,
                    uncertainty_probes=32, random_state=0, progress=False, save_path=None):
    """Predict GRM-aware genetic latent factor means directly.

    Parameters
    ----------
    traits : pandas.DataFrame
        Animals x traits; NaN marks any unobserved cell.
    GRM : matrix, labeled DataFrame, operator or named mapping
        Animal genomic kernels, compatible with the fitted component names.
    covariance_result : VIGOR genetic_varcov_PSD result (xarray.Dataset or mapping)
        Fitted genetic and residual factor specifications (not refitted).
    n_factors : positive int or None
        Default: None.
        Return first n factors per component; condition on ALL fitted factors.
    mode : 'both', 'assisted', 'genomic_only'
        Default: 'both'.
        Assisted uses target's own observed traits and its relatives.
        Genomic-only masks target_ids' own traits; reference traits remain.
    target_ids : iterable of animal labels, optional
        Required for genomic_only/both. Default: None. For all-animal unbiased predictions,
        run whole-animal cross-fitting outside this function.
    solver : 'auto', 'observed', 'latent'
        Default: 'auto'.
        Observed solves all observed cells, then K@(A@L). Latent is an exact
        factor-space precision solve only with zero genetic uniqueness and
        positive residual uniqueness; otherwise raises or auto-falls back.
    work_dtype : str
        Floating point precision used in matrix operations. Default: 'float64'.
    accuracy : 'exact' or 'fast'
        Default: 'exact'.
        VIGOR covariance conditioning precision. Exact means float64, full K.
    preconditioner : str
        Selects observed-cell conjugate-gradient preconditioning. Default: ``'diag'``.
    kernel_rank : str or int
        Animal kernel approximation rank; full retains exact GRM. Default: ``'full'``.
    solver_tol : float
        Maximum relative residual tolerance required of conditioning CG. Default: 1e-06.
    solver_maxiter : int
        Positive cap on conjugate gradient iterations per solve. Default: 5000.
    blas_threads : int
        Positive BLAS compute thread limit during solver operations. Default: 10.
    random_state : int
        Random number seed for stochastic posterior uncertainty probes. Default: 0.
    trait_means : Series or array, optional
        Default: None.
        Training-only means strongly recommended. Otherwise from fitted
        result if available, else from reference animals.
    center : 'auto', True or False
        Default: 'auto'.
        False indicates externally centered traits. Otherwise fixed means.
    return_uncertainty : bool
        Default: False.
        Optional exact (small) or stochastic posterior marginal factor SDs.
        Does not incorporate covariance fitting uncertainty.
    uncertainty_probes : int
        Default: 32.
        Hutchinson probe count for large problems.
    progress : bool
        Enable optional progress feedback in compatible wrappers. Default: False.
    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded
        Zarr v3 and returned reopened lazily. Default: None.

    Returns
    -------
    xarray.Dataset
        Data variables ``assisted`` and/or ``genomic_only`` on ('sample','factor')
        (factors of all genetic components concatenated; the ``factor_component``
        coordinate maps each factor, e.g. ``additive_FA1``, to its component),
        ``loadings`` on ('trait','factor') and, with return_uncertainty,
        ``uncertainty_sd`` on ('mode','sample','factor'). Attributes ``diagnostics``
        (per mode) and ``metadata`` (targets, centering, trait_mean, modes, model).

    Complexity
    ----------
    Observed CG: O(iter*k*(n^2*t+n*t*r)) dense-kernel worst case, plus
    O(k*n^2*r) factor propagation, O(nt+n*r) work arrays.
    Latent CG: O(k*n^3) Cholesky setup, O(iter*k*n^2*r+n*r^2)
    work. No full n x t trait-prediction matrix is generated.
    """
    from .results import maybe_save
    traits=_as_pandas_input(traits)
    if not isinstance(traits,pd.DataFrame) or traits.index.has_duplicates or traits.columns.has_duplicates:
        raise ValueError("traits must be a DataFrame with unique animal/trait labels")
    if traits.index.hasnans or traits.columns.hasnans:
        raise ValueError("animal/trait labels must be non-missing")
    Yraw=traits.to_numpy(dtype=float)
    if np.isinf(Yraw).any():
        raise ValueError("infinite traits not supported")
    if mode not in ("both","genomic_only","assisted"):
        raise ValueError("invalid mode")
    if solver not in ("auto","observed","latent"):
        raise ValueError("invalid solver")
    if n_factors is not None and (not isinstance(n_factors,(int,np.integer)) or n_factors<=0):
        raise ValueError("n_factors must be a positive integer")
    if return_uncertainty and uncertainty_probes<1:
        raise ValueError("uncertainty_probes must be positive")
    if center not in ("auto",True,False):
        raise ValueError("invalid center")
    fit_view=_result_view(covariance_result)
    fitted_cov=fit_view.get("G") if hasattr(fit_view,"get") else None
    if isinstance(fitted_cov,pd.DataFrame) and (not fitted_cov.index.equals(traits.columns)
                                            or not fitted_cov.columns.equals(traits.columns)):
        raise ValueError("traits.columns must exactly match fitted covariance order")
    if target_ids is None:
        if mode!="assisted":
            raise ValueError("target_ids required for genomic_only/both; use animal-held-out folds")
        targets=None
    else:
        targets=pd.Index(target_ids)
        if targets.has_duplicates or len(targets)==0 or not targets.isin(traits.index).all():
            raise ValueError("target_ids must be distinct and contained in traits.index")
        if len(targets)==len(traits) and mode!="assisted":
            raise ValueError("no reference animals remain")
    means,scope=_means(traits,fit_view,targets,trait_means,center)
    centered=traits.astype(float).sub(means,axis=1)
    modes=["assisted","genomic_only"] if mode=="both" else [mode]
    out={"metadata":{"uses_grm":True,"target_ids":[] if targets is None else list(targets),
                     "centering":scope,"trait_mean":means,
                     "modes":modes,"model":"genetic_factor_posterior",
                     "covariance_refitted":False},
         "diagnostics":{}}
    for key in modes:
        Y=centered.copy()
        if key=="genomic_only":Y.loc[targets,:]=np.nan
        with threadpool_limits(limits=blas_threads,user_api="blas"):
            plan=prepare_mvBLUP_plan(
                Y,GRM,covariance_result,center=False,accuracy=accuracy,
                work_dtype=work_dtype,kernel_rank=kernel_rank,
                preconditioner=preconditioner,blas_threads=blas_threads)
            widths={nm:min(sp.rank,n_factors or sp.rank) for nm,sp in plan.G_specs.items()}
            all_rank=sum(sp.rank for sp in plan.G_specs.values())
            eligible=(plan.n<=1500 and all_rank<=32 and plan.t>=4*max(1,all_rank)
                      and plan.E_spec.rank<=64
                      and np.all(np.asarray(plan.E_spec.diag_specific)>0)
                      and all(np.max(np.abs(s.diag_specific),initial=0)<=1e-12
                              for s in plan.G_specs.values())
                      and all(hasattr(k,"K") for k in plan.kernels.values()))
            F=None;reason=None
            if solver=="latent" or (solver=="auto" and eligible):
                try:F,diag=_latent(plan,Y.to_numpy(dtype=float),solver_tol,solver_maxiter)
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    if solver=="latent":raise
                    reason=str(exc)
            if F is None:
                F,diag=_dual(plan,solver_tol,solver_maxiter)
                if reason:diag["latent_fallback_reason"]=reason
            data={nm:pd.DataFrame(np.asarray(F[nm])[:,:widths[nm]],
                                  index=traits.index,
                                  columns=[f"{nm}_FA{i+1}" for i in range(widths[nm])])
                  for nm in F}
            if "loadings" not in out:
                out["loadings"]={nm:pd.DataFrame(plan.G_specs[nm].factor[:,:widths[nm]],
                                                index=traits.columns,columns=data[nm].columns)
                                 for nm in data}
            # A reference animal contributes its OWN phenotype in this solve:
            # its score is not genomic-only. Expose genomic-only targets only.
            if key=="genomic_only":
                for frame in data.values():
                    frame.loc[~frame.index.isin(targets),:]=np.nan
            out[key]=next(iter(data.values())) if len(data)==1 else data
            if return_uncertainty:
                sd,meta=_factor_uncertainty(plan,widths,solver_tol,solver_maxiter,
                                            uncertainty_probes,random_state)
                dfs={nm:pd.DataFrame(sd[nm],index=traits.index,columns=data[nm].columns)
                     for nm in data}
                out.setdefault("uncertainty_sd",{})[key]=(
                    next(iter(dfs.values())) if len(dfs)==1 else dfs)
                diag["uncertainty"]=meta
            diag.update(n_observed=int(plan.obs_plan["m"]),n_animals=plan.n,
                        n_traits=plan.t,component_ranks=widths,
                        target_phenotypes_masked=(key=="genomic_only"))
            out["diagnostics"][key]=diag
    params=dict(n_factors=n_factors,mode=mode,solver=solver,accuracy=accuracy,work_dtype=work_dtype,
                preconditioner=preconditioner,kernel_rank=kernel_rank,solver_tol=solver_tol,
                solver_maxiter=solver_maxiter,blas_threads=blas_threads,center=center,
                return_uncertainty=return_uncertainty,uncertainty_probes=uncertainty_probes,
                random_state=random_state)
    return maybe_save(_factor_prediction_dataset(out,traits,params),save_path)


__all__=["mvFactorPredict"]
