"""Additive+dominance fine mapping, colocalization and mediation (VIGOR 2.7)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import multivariate as _mv
from .results import result_dataset, maybe_save
from .attribution import _table_vars, _result_table

_project_psd=_mv._project_psd
_logbf_mvn_matrix=_mv._logbf_mvn_matrix
credible_set_idx=_mv.credible_set_idx
compute_variant_ld=_mv.compute_variant_ld
trait_genetic_canonical_basis=_mv.trait_genetic_canonical_basis
_linear_shap_from_effects=_mv._linear_shap_from_effects
_fit_single_residual_model=_mv._fit_single_residual_model
_tree_pred_contrib=_mv._tree_pred_contrib
_tree_interactions=_mv._tree_interactions
mvPredict=_mv.mvPredict
multitrait_elasticnet_finemap=_mv.multitrait_elasticnet_finemap
lightgbm_multitrait_finemap=_mv.lightgbm_multitrait_finemap
multitrait_sparse_group_elasticnet_finemap=_mv.multitrait_sparse_group_elasticnet_finemap
cluster_variant_ld=_mv.cluster_variant_ld
summarize_ld_clusters=_mv.summarize_ld_clusters
evidence_prior_weights=_mv.evidence_prior_weights
_normalize_snp_id=_mv._normalize_snp_id
fine_map_region=_mv.fine_map_region

# ============================================================================
# v3.1.4 additive + dominance fine mapping
# ============================================================================

V3_1_NATIVE_FINEMAP_AD_VERSION = '2.9.4-native-finemap-ad'


def _resolve_ad_trait_covariances(G=None, G_additive=None, G_dominance=None, E=None, heritability=None, n_traits=None):
    """Resolve additive/dominance trait covariance matrices from named or explicit inputs.

    Parameters
    ----------
    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.
    """
    if isinstance(G, dict):
        low={str(k).lower():v for k,v in G.items()}
        if G_additive is None:
            G_additive=low.get('additive',low.get('a',low.get('genetic')))
        if G_dominance is None:
            G_dominance=low.get('dominance',low.get('d'))
    elif G is not None and G_additive is None:
        G_additive=G
    if G_additive is None:
        if heritability is None:
            if n_traits is None: raise ValueError('provide G_additive/G or heritability')
            G_additive=np.eye(int(n_traits))
        else:
            h=np.asarray(heritability,float).ravel();G_additive=np.diag(np.clip(h,1e-8,None))
    GA=_project_psd(np.asarray(G_additive,float),eps=1e-10);t=GA.shape[0]
    if G_dominance is None:
        GD=np.zeros_like(GA)
    else: GD=_project_psd(np.asarray(G_dominance,float),eps=0.0)
    if E is None:
        # If no E was supplied, use the remaining unit-scale variance as a conservative diagonal approximation.
        diag=np.clip(1.-np.diag(GA)-np.diag(GD),1e-3,None);E0=np.diag(diag)
    else:E0=_project_psd(np.asarray(E,float),eps=1e-10)
    if GA.shape!=(t,t) or GD.shape!=(t,t) or E0.shape!=(t,t):raise ValueError('G_additive, G_dominance and E must have identical trait dimensions')
    return GA,GD,E0


def _prepare_ad_finemap_xy(genotypes, traits, *, allele_frequencies=None):
    """Prepare Vitezica A/D marker designs and complete multivariate phenotype rows.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    allele_frequencies : one-dimensional array-like or None
        Frequency of the coded allele for each genotype variant, on [0, 1]. Length must equal
        the variant axis; None estimates frequencies from the designated training animals.
        Default: ``None``.
    """
    X0=genotypes.to_numpy(float) if isinstance(genotypes,pd.DataFrame) else np.asarray(genotypes,float)
    Y0=traits.to_numpy(float) if isinstance(traits,pd.DataFrame) else np.asarray(traits,float)
    if Y0.ndim==1:Y0=Y0[:,None]
    rows=np.isfinite(Y0).all(1)
    if rows.sum()<max(20,Y0.shape[1]+5):raise ValueError('too few complete phenotype rows for A+D fine mapping')
    X=X0[rows];valid=np.isfinite(X);counts=valid.sum(0)
    p=np.asarray(allele_frequencies,float) if allele_frequencies is not None else np.divide(np.nansum(X,0),2*counts,out=np.full(X.shape[1],np.nan),where=counts>0)
    p=np.clip(np.nan_to_num(p,nan=.5),1e-6,1-1e-6);q=1-p
    A=X-2*p[None,:];A[~valid]=0.
    D=np.where(X==0,-2*p[None,:]**2,np.where(X==1,2*p[None,:]*q[None,:],np.where(X==2,-2*q[None,:]**2,0.)));D[~valid]=0.
    # Unit-SD scaling improves numerical comparability while retaining Vitezica centering.
    As=np.sqrt(np.mean(A*A,axis=0));Ds=np.sqrt(np.mean(D*D,axis=0));As=np.where(As>1e-8,As,1.);Ds=np.where(Ds>1e-8,Ds,1.)
    Az=A/As[None,:];Dz=D/Ds[None,:]
    Ym=Y0[rows].mean(0);Y=Y0[rows]-Ym
    names=list(map(str,genotypes.columns)) if isinstance(genotypes,pd.DataFrame) else [f'v{j}' for j in range(X0.shape[1])]
    tnames=list(map(str,traits.columns)) if isinstance(traits,pd.DataFrame) else [f'trait{j}' for j in range(Y0.shape[1])]
    return {'A':Az,'D':Dz,'A_raw':A,'D_raw':D,'Y':Y,'rows':rows,'snps':names,'traits':tnames,'allele_frequency':p,
            'A_scale':As,'D_scale':Ds,'y_mean':Ym}


def _ad_prior_covariances(C, prior_scales=(.01,.03,.1,.3), include_iid=True, max_axes=4, eps=1e-10):
    """Construct additive/dominance multivariate prior covariance mixtures.

    Parameters
    ----------
    C : array-like
        Numerical operand in ``C * (t / max(tr, eps))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    prior_scales : float or sequence
        Positive prior effect-variance/scale setting for Bayesian inclusion. Scalar settings
        apply across variants; mixtures/arrays follow the corresponding prior-component order.
        Default: ``(0.01, 0.03, 0.1, 0.3)``.

    include_iid : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``True``.

    max_axes : int
        Input for the operation ``range(min(int(max_axes), t))``. See
        ``vigor.multivariate_ad._ad_prior_covariances`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``4``.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-10``.
    """
    C=_project_psd(np.asarray(C,float),eps=0.0);t=C.shape[0];tr=float(np.trace(C));shape=C*(t/max(tr,eps)) if tr>eps else np.eye(t);out=[];names=[]
    for s in prior_scales:
        s=float(s);out.append(s*shape);names.append(f'genetic_{s:g}')
        if include_iid:out.append(s*np.eye(t));names.append(f'iid_{s:g}')
    w,V=np.linalg.eigh(shape);order=np.argsort(w)[::-1]
    for k in range(min(int(max_axes),t)):
        v=V[:,order[k]];out.append(float(np.median(prior_scales))*np.outer(v,v));names.append(f'axis{k+1}')
    return out,names


def _mixture_abf_posterior_mean(bhat,V,Wlist,log_weights=None):
    """Average effect posterior means over approximate-Bayes-factor mixture weights.

    Parameters
    ----------
    bhat : array-like or numeric scalar
        Input for the operation ``np.asarray(bhat, float).reshape(1, -1)``. See
        ``vigor.multivariate_ad._mixture_abf_posterior_mean`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    V : array-like
        Numerical operand in ``W @ np.linalg.solve(V + W, bh[0])``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    Wlist : object (see the cited input contract)
        Input for the operation ``enumerate(Wlist)``. See
        ``vigor.multivariate_ad._mixture_abf_posterior_mean`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    log_weights : array-like or mapping
        Model parameter/prior-weight values in the fitting layout consumed here. Weights must
        align prior mixture states and parameters the declared coefficient slices; use the
        matching preparation routine. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.
    """
    from scipy.special import logsumexp
    bh=np.asarray(bhat,float).reshape(1,-1);Q=len(Wlist);lw=np.full(Q,-np.log(Q)) if log_weights is None else np.asarray(log_weights,float);lbf=np.empty(Q);means=[]
    for q,W in enumerate(Wlist):
        W=np.asarray(W,float);lbf[q]=_logbf_mvn_matrix(bh,V,W)[0];means.append(W@np.linalg.solve(V+W,bh[0]))
    z=lbf+lw;den=logsumexp(z);gamma=np.exp(z-den);mean=np.sum(np.asarray(means)*gamma[:,None],axis=0)
    return float(den),mean,gamma


def multitrait_bayes_finemap_ad(genotypes,traits,*,G=None,G_additive=None,G_dominance=None,E=None,heritability=None,
                                 L=10,coverage=.95,prior_scales=(.01,.03,.1,.3),prior_weights=None,
                                 state_prior=(.40,.25,.35),max_iter=100,tol=1e-5,ld_r=None,ld_r2=None,
                                 ld_clusters=None,allele_frequencies=None,return_effects=True,min_cs_abs_r=.5,drop_impure=True):
    """Native A+D multi-trait IBSS fine mapping with additive, dominance and joint states.

    Each single effect chooses one variant and one state S in {A,D,A+D}. Additive and
    dominance effect-vector priors are informed separately by G_A and G_D, while the common
    likelihood is E-whitened using the canonical basis of G_A+G_D versus E.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    L : array-like
        Numerical operand in ``alpha[:, :, 0].sum(0) / max(int(L), 1)``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state. Default: ``10``.

    coverage : float
        Posterior probability mass targeted by a credible set, on (0, 1]. Larger coverage
        retains more variants; inclusion is conditional on the fitted locus model. Default:
        ``0.95``.

    prior_scales : float or sequence
        Positive prior effect-variance/scale setting for Bayesian inclusion. Scalar settings
        apply across variants; mixtures/arrays follow the corresponding prior-component order.
        Default: ``(0.01, 0.03, 0.1, 0.3)``.

    prior_weights : mapping or array-like or None
        Weights for the corresponding covariance/evidence/prior components. Match the documented
        component names or variant order; None applies the routine's uniform/default weighting.
        Default: ``None``.

    state_prior : array-like or mapping
        Model parameter/prior-weight values in the fitting layout consumed here. Weights must
        align prior mixture states and parameters the declared coefficient slices; use the
        matching preparation routine. Default: ``(0.4, 0.25, 0.35)``.

    max_iter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``100``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-05``.

    ld_r : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    ld_r2 : array-like or None
        Variant × variant linkage-disequilibrium matrix in the genotype-column order. Use
        correlation or squared correlation according to ld_kind/the called fitter; None requests
        computation where supported. Default: ``None``.

    ld_clusters : sequence or None
        LD-group assignment for each genotype column. Equal labels group linked variants; None
        asks the fitter to infer groups when supported. Default: ``None``.

    allele_frequencies : one-dimensional array-like or None
        Frequency of the coded allele for each genotype variant, on [0, 1]. Length must equal
        the variant axis; None estimates frequencies from the designated training animals.
        Default: ``None``. Passed to ``_prepare_ad_finemap_xy`` as ``allele_frequencies``.

    return_effects : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    min_cs_abs_r : float
        Input for the operation ``csdf['cs_min_abs_r'] >= float(min_cs_abs_r)``. See
        ``vigor.multivariate_ad.multitrait_bayes_finemap_ad`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.5``.

    drop_impure : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from scipy.special import logsumexp
    z=_prepare_ad_finemap_xy(genotypes,traits,allele_frequencies=allele_frequencies);A,D,Y=z['A'],z['D'],z['Y'];n,p=A.shape;t=Y.shape[1]
    GA,GD,E0=_resolve_ad_trait_covariances(G,G_additive,G_dominance,E,heritability,t);Gtot=_project_psd(GA+GD,eps=1e-10)
    cb=trait_genetic_canonical_basis(Gtot,E0,n_traits=t);T,Tinv=cb['transform'],cb['inverse_transform'];Yc=Y@T;Ecan=_project_psd(T.T@E0@T,eps=1e-10)
    GAcan=_project_psd(T.T@GA@T,eps=0.0);GDcan=_project_psd(T.T@GD@T,eps=0.0)
    WA,WAn=_ad_prior_covariances(GAcan,prior_scales);WD,WDn=_ad_prior_covariances(GDcan if np.trace(GDcan)>1e-10 else np.eye(t)*1e-3,prior_scales)
    q=min(len(WA),len(WD));WA,WD=WA[:q],WD[:q];WJ=[np.block([[WA[i],np.zeros((t,t))],[np.zeros((t,t)),WD[i]]]) for i in range(q)];lw=np.full(q,-np.log(q))
    pi=np.ones(p)/p if prior_weights is None else np.asarray(prior_weights,float);pi=np.clip(pi,0,None);pi=pi/pi.sum();lpi=np.log(np.clip(pi,1e-300,None))
    sp=np.asarray(state_prior,float);sp=np.clip(sp,1e-8,None);sp=sp/sp.sum();lsp=np.log(sp)
    alpha=np.full((int(L),p,3),1/(3*p),float);mA=np.zeros((int(L),p,3,t));mD=np.zeros_like(mA);mu=np.zeros((int(L),n,t));trace=[]
    for it in range(int(max_iter)):
        old=alpha.copy();fitted=mu.sum(0);obj=0.
        for ell in range(int(L)):
            base=fitted-mu[ell];R=Yc-base;logbf=np.empty((p,3));ma=np.zeros((p,3,t));md=np.zeros_like(ma)
            for j in range(p):
                xa=A[:,j];xd=D[:,j];ssa=float(xa@xa);ssd=float(xd@xd)
                if ssa<=1e-10:ssa=1.;
                if ssd<=1e-10:ssd=1.;
                bha=(xa@R)/ssa; bhd=(xd@R)/ssd
                logbf[j,0],ma[j,0],_= _mixture_abf_posterior_mean(bha,Ecan/ssa,WA,lw)
                logbf[j,1],md[j,1],_= _mixture_abf_posterior_mean(bhd,Ecan/ssd,WD,lw)
                Zj=np.column_stack([xa,xd]);XtX=Zj.T@Zj+1e-8*np.eye(2);invXt=np.linalg.inv(XtX);Bhat=invXt@Zj.T@R;bv=Bhat.reshape(-1);Vj=np.kron(invXt,Ecan)
                lb,mj,_=_mixture_abf_posterior_mean(bv,Vj,WJ,lw);logbf[j,2]=lb;ma[j,2]=mj[:t];md[j,2]=mj[t:]
            la=lpi[:,None]+lsp[None,:]+logbf;den=logsumexp(la);alpha[ell]=np.exp(la-den);mA[ell]=ma;mD[ell]=md
            ea=np.sum(alpha[ell,:,:,None]*ma,axis=1);ed=np.sum(alpha[ell,:,:,None]*md,axis=1);mu[ell]=A@ea+D@ed;fitted=base+mu[ell];obj+=float(den)
        trace.append(obj)
        if np.max(np.abs(alpha-old))<float(tol):break
    pa=alpha[:,:,0]+alpha[:,:,2];pd_=alpha[:,:,1]+alpha[:,:,2];pany=alpha.sum(2)
    pipA=1-np.prod(1-pa,axis=0);pipD=1-np.prod(1-pd_,axis=0);pip=1-np.prod(1-pany,axis=0)
    BAcan=np.sum(alpha[:,:,:,None]*mA,axis=(0,2));BDcan=np.sum(alpha[:,:,:,None]*mD,axis=(0,2));BA=BAcan@Tinv;BD=BDcan@Tinv
    pipdf=pd.DataFrame({'snp':z['snps'],'pip':pip,'pip_additive':pipA,'pip_dominance':pipD,'posterior_state_additive':alpha[:,:,0].sum(0)/max(int(L),1),'posterior_state_dominance':alpha[:,:,1].sum(0)/max(int(L),1),'posterior_state_joint':alpha[:,:,2].sum(0)/max(int(L),1)}).sort_values('pip',ascending=False)
    cs=[];along=[]
    for ell in range(int(L)):
        av=pany[ell];idx=credible_set_idx(av,coverage);coverage_actual=float(av[idx].sum());
        for j in range(p):along.append((ell+1,z['snps'][j],alpha[ell,j,0],alpha[ell,j,1],alpha[ell,j,2],av[j]))
        for rank,j in enumerate(idx,1):cs.append((ell+1,z['snps'][j],rank,av[j],alpha[ell,j,0],alpha[ell,j,1],alpha[ell,j,2],coverage_actual))
    csdf=pd.DataFrame(cs,columns=['effect','snp','cs_rank','alpha','alpha_additive','alpha_dominance','alpha_joint','coverage_actual']);csdf_all=csdf.copy()
    adf=pd.DataFrame(along,columns=['effect','snp','alpha_additive','alpha_dominance','alpha_joint','alpha_any'])
    if ld_r is None or ld_r2 is None:
        ldz=_mv._compute_variant_ld_raw(pd.DataFrame(np.asarray(genotypes)[z['rows']] if not isinstance(genotypes,pd.DataFrame) else genotypes.loc[np.asarray(genotypes.index)[z['rows']]],columns=z['snps']),statistic='both');ld_r=ldz['r'];ld_r2=ldz['r2']
    if len(csdf):
        rr=pd.DataFrame(ld_r,index=z['snps'],columns=z['snps']) if not isinstance(ld_r,pd.DataFrame) else ld_r
        r2=pd.DataFrame(ld_r2,index=z['snps'],columns=z['snps']) if not isinstance(ld_r2,pd.DataFrame) else ld_r2
        purity=[]
        for e,g in csdf.groupby('effect'):
            ids=g.snp.tolist();sub=np.abs(rr.loc[ids,ids].to_numpy());sub2=r2.loc[ids,ids].to_numpy();mask=~np.eye(len(ids),dtype=bool);purity.append((e,float(np.nanmin(sub[mask])) if mask.any() else 1.,float(np.nanmedian(sub[mask])) if mask.any() else 1.,float(np.nanmin(sub2[mask])) if mask.any() else 1.))
        pur=pd.DataFrame(purity,columns=['effect','cs_min_abs_r','cs_median_abs_r','cs_min_r2']);csdf=csdf.merge(pur,on='effect',how='left');csdf['credible_set_pure']=csdf['cs_min_abs_r']>=float(min_cs_abs_r);csdf_all=csdf.copy();csdf=csdf[csdf['credible_set_pure']].copy() if drop_impure else csdf
    out={'pip':pipdf,'alpha':adf,'credible_sets':csdf,'credible_sets_all':csdf_all,'credible_set_purity_threshold':float(min_cs_abs_r),'coef_additive':pd.DataFrame(BA,index=z['snps'],columns=z['traits']),
         'coef_dominance':pd.DataFrame(BD,index=z['snps'],columns=z['traits']),'canonical_basis':cb,'G_additive':GA,'G_dominance':GD,'E':E0,
         'prior_weights':pd.Series(pi,index=z['snps']),'state_prior':pd.Series(sp,index=['additive','dominance','joint']),
         'elbo_trace':np.asarray(trace),'converged':len(trace)<int(max_iter),'n_iter':len(trace),'allele_frequency':pd.Series(z['allele_frequency'],index=z['snps']),
         'version':V3_1_NATIVE_FINEMAP_AD_VERSION}
    return out


def _fit_ad_ridge_core(A,D,Y,GA,GD,E,alpha,*,dominance_penalty_multiplier=1.,max_alt=30,tol=1e-7):
    """Alternating closed-form G_A/E and G_D/E ridge updates.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``Y - A @ BA``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    D : array-like or numeric scalar
        Input for the operation ``Y - D @ BD``. See ``vigor.multivariate_ad._fit_ad_ridge_core``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    GA : array-like or numeric scalar
        Input for the operation ``update(A, Y - D @ BD, GA, alpha)``. See
        ``vigor.multivariate_ad._fit_ad_ridge_core`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    GD : array-like or numeric scalar
        Input for the operation ``update(D, Y - A @ BA, GD if np.trace(GD) > 1e-10 else
        np.eye(t) * 0.001, float(alpha) * float(dominance_penalty_multiplier))``. See
        ``vigor.multivariate_ad._fit_ad_ridge_core`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.

    dominance_penalty_multiplier : float
        Input for the operation ``float(alpha) * float(dominance_penalty_multiplier)``. See
        ``vigor.multivariate_ad._fit_ad_ridge_core`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``1.0``.

    max_alt : int
        Input for the operation ``range(int(max_alt))``. See
        ``vigor.multivariate_ad._fit_ad_ridge_core`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``30``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-07``.
    """
    p=A.shape[1];t=Y.shape[1];BA=np.zeros((p,t));BD=np.zeros((p,t));I=np.eye(p)
    def update(X,R,Gc,penalty):
        """Advance the progress counter by the requested increment.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        R : array-like
            Numerical operand in ``R @ T``. Shape must conform to that matrix/vector operation and
            the enclosing prepared state.

        Gc : object (see the cited input contract)
            Input for the operation ``trait_genetic_canonical_basis(_project_psd(Gc, eps=1e-10), E,
            n_traits=t)``. See ``vigor.multivariate_ad._fit_ad_ridge_core.update`` and its caller
            for the scientific representation; the shown operation constrains the required
            type/shape.

        penalty : float or array-like
            Regularization strength added to the fitted objective or linear system. Nonnegative
            values describe a penalty; zero removes that penalty. Componentwise arrays must match
            the component order.

        Returns
        -------
        object
            Computed result from ``Bc @ Tinv``; see the summary and the called implementation for its labeled/numerical representation.
        """
        cb=trait_genetic_canonical_basis(_project_psd(Gc,eps=1e-10),E,n_traits=t);T,Tinv,lam=cb['transform'],cb['inverse_transform'],cb['genetic_to_environment_ratio'];Rc=R@T;XtX=X.T@X;XtR=X.T@Rc;Bc=np.column_stack([np.linalg.solve(XtX+(float(penalty)/max(lam[q],1e-8))*I,XtR[:,q]) for q in range(t)]);return Bc@Tinv
    for it in range(int(max_alt)):
        oldA=BA.copy();oldD=BD.copy();BA=update(A,Y-D@BD,GA,alpha);BD=update(D,Y-A@BA,GD if np.trace(GD)>1e-10 else np.eye(t)*1e-3,float(alpha)*float(dominance_penalty_multiplier));delta=max(np.max(np.abs(BA-oldA)),np.max(np.abs(BD-oldD)))
        if delta<float(tol):break
    return BA,BD,it+1


def multitrait_ridge_finemap_ad(genotypes,traits,*,G=None,G_additive=None,G_dominance=None,E=None,heritability=None,
                                 alphas=None,cv=5,random_state=0,dominance_penalty_multiplier=1.,compute_shap=True,
                                 shap_mode='interventional',shap_nsamples=512,save_path=None):
    """Joint A+D ridge fine mapping with separate G_A and G_D trait-space penalties.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    alphas : float, int or sequence
        Regularization search grid: l1_ratio mixes L1 versus L2 penalties on [0, 1]; alphas
        supplies candidate strengths or the number of strengths generated by the called CV
        estimator. See that estimator for automatic grid semantics. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value. Passed to ``pd.Series`` as ``index``.

    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``5``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.

    dominance_penalty_multiplier : float
        Input supplied to ``_fit_ad_ridge_core`` as ``dominance_penalty_multiplier``. See that
        callable's documented ``dominance_penalty_multiplier`` contract for its type, alignment
        and value domain. This wrapper retains that contract. Default: ``1.0``. Passed to
        ``_fit_ad_ridge_core`` as ``dominance_penalty_multiplier``.

    compute_shap : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    shap_mode : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default:
        ``'interventional'``. Passed to ``_linear_shap_from_effects`` as ``mode``.

    shap_nsamples : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default: ``512``. Passed
        to ``_linear_shap_from_effects`` as ``nsamples``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``coef_additive``, ``coef_dominance`` (snp, trait); the former
        ``ranking`` table as per-snp variables in genotype-column order (``ridge_ad_l2``,
        ``ridge_additive_l2``, ``ridge_dominance_l2`` and, with SHAP,
        ``mean_abs_shap_additive``, ``mean_abs_shap_dominance``, ``mean_abs_shap``; sort by
        ``ridge_ad_l2`` for the ranking order); ``cv_mse`` (ridge_alpha); ``G_additive``,
        ``G_dominance``, ``E`` (trait_i, trait_j); and the former ``shap`` mapping as
        ``shap_values`` (sample, feature, trait), ``shap_expected_value`` (trait) and
        ``shap_mean_abs_shap`` (feature). Attrs: ``alpha``, ``n_iter``, ``version``,
        ``shap_mode`` plus title/software/parameters provenance.
    """
    from sklearn.model_selection import KFold
    import xarray as xr
    genotypes=genotypes.to_pandas() if isinstance(genotypes,xr.DataArray) else genotypes
    traits=traits.to_pandas() if isinstance(traits,xr.DataArray) else traits
    z=_prepare_ad_finemap_xy(genotypes,traits);A,D,Y=z['A'],z['D'],z['Y'];t=Y.shape[1];GA,GD,E0=_resolve_ad_trait_covariances(G,G_additive,G_dominance,E,heritability,t);alphas=np.logspace(-3,3,15) if alphas is None else np.asarray(alphas,float);kf=KFold(min(int(cv),max(2,len(Y)//20)),shuffle=True,random_state=random_state);scores=[]
    for a in alphas:
        ss=0.;nn=0
        for tr,va in kf.split(A):
            BA,BD,_=_fit_ad_ridge_core(A[tr],D[tr],Y[tr],GA,GD,E0,a,dominance_penalty_multiplier=dominance_penalty_multiplier,max_alt=15,tol=3e-6);R=Y[va]-A[va]@BA-D[va]@BD;ss+=float(np.sum(R*R));nn+=R.size
        scores.append(ss/max(nn,1))
    alpha=float(alphas[int(np.argmin(scores))]);BA,BD,nit=_fit_ad_ridge_core(A,D,Y,GA,GD,E0,alpha,dominance_penalty_multiplier=dominance_penalty_multiplier,max_alt=50,tol=1e-7);ea=np.linalg.norm(BA,axis=1);ed=np.linalg.norm(BD,axis=1);et=np.sqrt(ea*ea+ed*ed);rank=pd.DataFrame({'snp':z['snps'],'ridge_ad_l2':et,'ridge_additive_l2':ea,'ridge_dominance_l2':ed}).sort_values('ridge_ad_l2',ascending=False);shap=None
    if compute_shap:
        features=np.concatenate([A,D],axis=1);B=np.concatenate([BA,BD],axis=0);fn=[f'{x}|A' for x in z['snps']]+[f'{x}|D' for x in z['snps']];shap=_linear_shap_from_effects(features,B,feature_names=fn,trait_names=z['traits'],mode=shap_mode,nsamples=shap_nsamples,random_state=random_state);sv=shap['shap_values'];sA=np.mean(np.abs(sv[:,:len(z['snps'])]),axis=(0,2));sD=np.mean(np.abs(sv[:,len(z['snps']):]),axis=(0,2));rank=rank.merge(pd.DataFrame({'snp':z['snps'],'mean_abs_shap_additive':sA,'mean_abs_shap_dominance':sD,'mean_abs_shap':sA+sD}),on='snp',how='left')
    snps,trs=z['snps'],z['traits'];tm=lambda M:pd.DataFrame(np.asarray(M,float),index=trs,columns=trs)
    values={'coef_additive':pd.DataFrame(BA,index=snps,columns=trs),'coef_dominance':pd.DataFrame(BD,index=snps,columns=trs),
            **_table_vars(rank.set_index('snp').reindex(snps),'ranking','snp'),
            'alpha':alpha,'cv_mse':pd.Series(scores,index=alphas),'G_additive':tm(GA),'G_dominance':tm(GD),'E':tm(E0),'n_iter':nit,'version':V3_1_NATIVE_FINEMAP_AD_VERSION}
    if shap is not None:
        fn=[f'{x}|A' for x in snps]+[f'{x}|D' for x in snps]
        values.update({'shap_values':xr.DataArray(shap['shap_values'],dims=('sample','feature','trait'),coords={'feature':fn,'trait':trs},
                                                  attrs={'long_name':'linear SHAP values','description':'complete phenotype rows x A/D marker features x traits'}),
                       'shap_expected_value':shap['expected_value'],'shap_mode':shap['mode'],
                       **_table_vars(shap['ranking'].set_index('snp').reindex(fn),'shap_ranking','feature',prefix='shap_')})
    tvar={'long_name':'trait covariance used as ridge penalty geometry'}
    ds=result_dataset(values,dims={'coef_additive':('snp','trait'),'coef_dominance':('snp','trait'),'cv_mse':('ridge_alpha',),
                                   'G_additive':('trait_i','trait_j'),'G_dominance':('trait_i','trait_j'),'E':('trait_i','trait_j'),'shap_expected_value':('trait',)},
                      var_attrs={'coef_additive':{'long_name':'additive (Vitezica) ridge effect','description':'per unit-SD additive coding'},
                                 'coef_dominance':{'long_name':'dominance (Vitezica) ridge effect','description':'per unit-SD dominance coding'},
                                 'cv_mse':{'long_name':'cross-validated mean squared error'},'G_additive':tvar,'G_dominance':tvar,
                                 'E':{'long_name':'residual trait covariance'},'shap_expected_value':{'long_name':'SHAP expected value'}},
                      title='Multitrait A+D ridge fine mapping',software='vigor.multivariate_ad.multitrait_ridge_finemap_ad',
                      parameters={'cv':int(cv),'random_state':random_state,'dominance_penalty_multiplier':float(dominance_penalty_multiplier),
                                  'compute_shap':bool(compute_shap),'shap_mode':str(shap_mode),'shap_nsamples':int(shap_nsamples)})
    return maybe_save(ds,save_path)


def multitrait_elasticnet_finemap_ad(genotypes,traits,*,G=None,G_additive=None,G_dominance=None,E=None,heritability=None,
                                      l1_ratio=(.2,.5,.8,.95),alphas=50,cv=5,random_state=0,max_iter=10000,
                                      compute_shap=True,shap_mode='interventional',shap_nsamples=512):
    """A+D multitrait ElasticNet in the total-genetic G_A+G_D versus E canonical basis.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    l1_ratio : float, int or sequence
        Regularization search grid: l1_ratio mixes L1 versus L2 penalties on [0, 1]; alphas
        supplies candidate strengths or the number of strengths generated by the called CV
        estimator. See that estimator for automatic grid semantics. Default: ``(0.2, 0.5, 0.8,
        0.95)``. Passed to ``multitrait_elasticnet_finemap`` as ``l1_ratio``.

    alphas : float, int or sequence
        Regularization search grid: l1_ratio mixes L1 versus L2 penalties on [0, 1]; alphas
        supplies candidate strengths or the number of strengths generated by the called CV
        estimator. See that estimator for automatic grid semantics. Default: ``50``. Passed to
        ``multitrait_elasticnet_finemap`` as ``alphas``.

    cv : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``5``. Passed to ``multitrait_elasticnet_finemap`` as ``cv``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``multitrait_elasticnet_finemap`` as ``random_state``.

    max_iter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``10000``. Passed to ``multitrait_elasticnet_finemap`` as ``max_iter``.

    compute_shap : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``. Passed to ``multitrait_elasticnet_finemap``
        as ``compute_shap``.

    shap_mode : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default:
        ``'interventional'``. Passed to ``multitrait_elasticnet_finemap`` as ``shap_mode``.

    shap_nsamples : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default: ``512``. Passed
        to ``multitrait_elasticnet_finemap`` as ``shap_nsamples``.

    Returns
    -------
    object
        Computed result from ``fit``; see the summary and the called implementation for its labeled/numerical representation.
    """
    z=_prepare_ad_finemap_xy(genotypes,traits);GA,GD,E0=_resolve_ad_trait_covariances(G,G_additive,G_dominance,E,heritability,z['Y'].shape[1]);p=len(z['snps']);fn=[f'{x}|A' for x in z['snps']]+[f'{x}|D' for x in z['snps']];F=pd.DataFrame(np.concatenate([z['A'],z['D']],axis=1),columns=fn);Y=pd.DataFrame(z['Y'],columns=z['traits']);fit=multitrait_elasticnet_finemap(F,Y,G=_project_psd(GA+GD,eps=1e-10),E=E0,l1_ratio=l1_ratio,alphas=alphas,cv=cv,random_state=random_state,max_iter=max_iter,compute_shap=compute_shap,shap_mode=shap_mode,shap_nsamples=shap_nsamples);B=fit['coef'].to_numpy();BA=B[:p];BD=B[p:];ea=np.linalg.norm(BA,axis=1);ed=np.linalg.norm(BD,axis=1);rank=pd.DataFrame({'snp':z['snps'],'elasticnet_ad_l2':np.sqrt(ea*ea+ed*ed),'elasticnet_additive_l2':ea,'elasticnet_dominance_l2':ed})
    if fit.get('shap') is not None:
        S=fit['shap']['shap_values'];sA=np.mean(np.abs(S[:,:p]),axis=(0,2));sD=np.mean(np.abs(S[:,p:]),axis=(0,2));rank['mean_abs_shap_additive']=sA;rank['mean_abs_shap_dominance']=sD;rank['mean_abs_shap']=sA+sD
    rank=rank.sort_values('elasticnet_ad_l2',ascending=False);fit.update({'coef_additive':pd.DataFrame(BA,index=z['snps'],columns=z['traits']),'coef_dominance':pd.DataFrame(BD,index=z['snps'],columns=z['traits']),'ranking_variant':rank,'G_additive':GA,'G_dominance':GD,'version':V3_1_NATIVE_FINEMAP_AD_VERSION});return fit


def lightgbm_multitrait_finemap_ad(genotypes,traits,*,G=None,G_additive=None,G_dominance=None,E=None,heritability=None,
                                    feature_mode='ad',n_splits=5,random_state=0,model_kwargs=None,interactions=False):
    """A+D-aware LightGBM fine mapping using raw genotype or explicit A/D features.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    feature_mode : str
        Genotype feature coding: '012'/'genotype'/'raw' retains dosage features;
        'additive_dominance' augments additive dosage with dominance contrasts where this
        feature builder supports it. Default: ``'ad'``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``5``. Passed to ``lightgbm_multitrait_finemap`` as ``n_splits``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``lightgbm_multitrait_finemap`` as ``random_state``.

    model_kwargs : mapping
        Additional keyword settings forwarded to the selected estimator/writer. See the called
        implementation for its accepted names and value domains; these are not unrestricted
        scientific options. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to
        ``lightgbm_multitrait_finemap`` as ``model_kwargs``.

    interactions : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default: ``False``. Passed
        to ``lightgbm_multitrait_finemap`` as ``interactions``.

    Returns
    -------
    object
        Computed result from ``fit``; see the summary and the called implementation for its labeled/numerical representation.
    """
    z=_prepare_ad_finemap_xy(genotypes,traits);GA,GD,E0=_resolve_ad_trait_covariances(G,G_additive,G_dominance,E,heritability,z['Y'].shape[1]);p=len(z['snps']);mode=str(feature_mode).lower()
    if mode in {'raw','genotype','012'}:
        X=genotypes.loc[np.asarray(genotypes.index)[z['rows']]] if isinstance(genotypes,pd.DataFrame) else pd.DataFrame(np.asarray(genotypes)[z['rows']],columns=z['snps']);fn=z['snps'];split=False
    elif mode in {'ad','additive_dominance'}:
        fn=[f'{x}|A' for x in z['snps']]+[f'{x}|D' for x in z['snps']];X=pd.DataFrame(np.concatenate([z['A'],z['D']],axis=1),columns=fn);split=True
    else:raise ValueError("feature_mode must be 'raw' or 'ad'")
    Y=pd.DataFrame(z['Y'],columns=z['traits']);fit=lightgbm_multitrait_finemap(X,Y,G=_project_psd(GA+GD,eps=1e-10),E=E0,n_splits=n_splits,random_state=random_state,model_kwargs=model_kwargs,interactions=interactions,covariance_mode='canonical');S=fit['oof_shap']
    if split:
        sA=np.nanmean(np.abs(S[:,:p]),axis=(0,2));sD=np.nanmean(np.abs(S[:,p:]),axis=(0,2));rank=pd.DataFrame({'snp':z['snps'],'mean_abs_oof_shap':sA+sD,'mean_abs_oof_shap_additive':sA,'mean_abs_oof_shap_dominance':sD}).sort_values('mean_abs_oof_shap',ascending=False)
    else:rank=pd.DataFrame({'snp':z['snps'],'mean_abs_oof_shap':np.nanmean(np.abs(S),axis=(0,2))}).sort_values('mean_abs_oof_shap',ascending=False)
    fit.update({'ranking_variant':rank,'feature_mode':mode,'G_additive':GA,'G_dominance':GD,'version':V3_1_NATIVE_FINEMAP_AD_VERSION});return fit


_fine_map_region_additive_only = fine_map_region

def fine_map_region(genotypes,traits,*,G=None,E=None,heritability=None,G_additive=None,G_dominance=None,genetic_model='auto',
                    methods=('bayes','ridge','elasticnet','sparse_group','lightgbm'),coding=None,eqtl=None,xqtl=None,phewas=None,noe=None,
                    alphagenome=None,use_evidence_priors=True,evidence_weights=None,L=10,coverage=.95,random_state=0,
                    lightgbm_kwargs=None,lightgbm_ad_feature_mode='ad',shap_mode='ld',compute_linear_shap=True,ld_threshold=.2,
                    hdbscan=True,bayes_prior_scales=(.01,.03,.1,.3),sparse_group_kwargs=None):
    """Unified fine mapper; automatically uses A+D methods when dominance covariance is supplied.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to
        ``_fine_map_region_additive_only`` as ``G``.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as ``E``.

    heritability : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value. Passed to
        ``_fine_map_region_additive_only`` as ``heritability``.

    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    genetic_model : str
        Input for the operation ``str(genetic_model).lower()``. See
        ``vigor.multivariate_ad.fine_map_region`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'auto'``. Source-recognized literal values: ``a``, ``a+d``, ``ad``, ``additive``,
        ``additive_dominance``. These select the corresponding branches below; comparison
        evidence alone does not establish an exhaustive enum.

    methods : sequence of str
        Candidate estimator names compared using held-out prediction error. Names must be
        supported by the called model factory; selection/refitting must occur within each
        training fold. Default: ``('bayes', 'ridge', 'elasticnet', 'sparse_group',
        'lightgbm')``. Passed to ``_fine_map_region_additive_only`` as ``methods``.

    coding : str
        Genotype coding: 'additive'/'a' uses centered allele dosage; 'dominance'/'d' uses the
        frequency-adjusted heterozygosity contrast. Each component requires frequencies fitted
        on the designated training animals. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value. Passed to
        ``_fine_map_region_additive_only`` as ``coding``.

    eqtl : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as
        ``eqtl``.

    xqtl : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as
        ``xqtl``.

    phewas : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as
        ``phewas``.

    noe : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as
        ``noe``.

    alphagenome : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``. Passed to ``_fine_map_region_additive_only`` as
        ``alphagenome``.

    use_evidence_priors : bool
        Input supplied to ``_fine_map_region_additive_only`` as ``use_evidence_priors``. See
        that callable's documented ``use_evidence_priors`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``True``. Passed to
        ``_fine_map_region_additive_only`` as ``use_evidence_priors``.

    evidence_weights : mapping or array-like or None
        Weights for the corresponding covariance/evidence/prior components. Match the documented
        component names or variant order; None applies the routine's uniform/default weighting.
        Default: ``None``. Passed to ``_fine_map_region_additive_only`` as ``evidence_weights``.

    L : array-like
        Numerical input passed to ``_fine_map_region_additive_only`` as ``L``; see that callable
        for required dimensions and matrix role. Default: ``10``. Passed to
        ``_fine_map_region_additive_only`` as ``L``.

    coverage : float
        Posterior probability mass targeted by a credible set, on (0, 1]. Larger coverage
        retains more variants; inclusion is conditional on the fitted locus model. Default:
        ``0.95``. Passed to ``_fine_map_region_additive_only`` as ``coverage``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``_fine_map_region_additive_only`` as ``random_state``.

    lightgbm_kwargs : object (see the cited input contract)
        Input supplied to ``_fine_map_region_additive_only`` as ``lightgbm_kwargs``. See that
        callable's documented ``lightgbm_kwargs`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``_fine_map_region_additive_only`` as ``lightgbm_kwargs``.

    lightgbm_ad_feature_mode : str
        Input supplied to ``lightgbm_multitrait_finemap_ad`` as ``feature_mode``. See that
        callable's documented ``feature_mode`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``'ad'``. Passed to
        ``lightgbm_multitrait_finemap_ad`` as ``feature_mode``.

    shap_mode : int, str or bool
        SHAP explanation control: sample counts bound approximation work, mode selects the
        feature-dependence/LD explanation policy, and interactions/oof_shap request pair or
        held-out explanations. See the selected explainer's contract. Default: ``'ld'``. Passed
        to ``_fine_map_region_additive_only`` as ``shap_mode``.

    compute_linear_shap : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``. Passed to
        ``_fine_map_region_additive_only`` as ``compute_linear_shap``.

    ld_threshold : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.2``.

    hdbscan : bool
        Whether to cluster supported LD distances with HDBSCAN when available; unknown/missing
        LD distances are never fabricated to run clustering. Default: ``True``.

    bayes_prior_scales : float or sequence
        Positive prior effect-variance/scale setting for Bayesian inclusion. Scalar settings
        apply across variants; mixtures/arrays follow the corresponding prior-component order.
        Default: ``(0.01, 0.03, 0.1, 0.3)``. Passed to ``_fine_map_region_additive_only`` as
        ``bayes_prior_scales``.

    sparse_group_kwargs : object (see the cited input contract)
        Input supplied to ``_fine_map_region_additive_only`` as ``sparse_group_kwargs``. See
        that callable's documented ``sparse_group_kwargs`` contract for its type, alignment and
        value domain. This wrapper retains that contract. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``_fine_map_region_additive_only`` as ``sparse_group_kwargs``.

    Returns
    -------
    dict
        Result fields: ``ranking``, ``fits``, ``evidence``, ``ld``, ``ld_clusters``, ``ld_cluster_summary``, ``errors``, ``genetic_model``, ``G_additive``, ``G_dominance``, ``E``, ``version``. Field arrays preserve the corresponding input axis order.
    """
    model=str(genetic_model).lower();has_d=G_dominance is not None or (isinstance(G,dict) and any(str(k).lower() in {'dominance','d'} for k in G))
    if model=='auto':model='ad' if has_d else 'additive'
    if model in {'additive','a'}:
        return _fine_map_region_additive_only(genotypes,traits,G=G,E=E,heritability=heritability,methods=methods,coding=coding,eqtl=eqtl,xqtl=xqtl,phewas=phewas,noe=noe,alphagenome=alphagenome,use_evidence_priors=use_evidence_priors,evidence_weights=evidence_weights,L=L,coverage=coverage,random_state=random_state,lightgbm_kwargs=lightgbm_kwargs,shap_mode=shap_mode,compute_linear_shap=compute_linear_shap,ld_threshold=ld_threshold,hdbscan=hdbscan,bayes_prior_scales=bayes_prior_scales,sparse_group_kwargs=sparse_group_kwargs)
    if model not in {'ad','a+d','additive_dominance'}:raise ValueError("genetic_model must be 'auto', 'additive', or 'ad'")
    X=genotypes.copy() if isinstance(genotypes,pd.DataFrame) else pd.DataFrame(genotypes);Y=traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits);GA,GD,E0=_resolve_ad_trait_covariances(G,G_additive,G_dominance,E,heritability,Y.shape[1]);ldz=_mv._compute_variant_ld_raw(X,statistic='both');clusters=cluster_variant_ld(ldz['r2'],threshold=ld_threshold,use_hdbscan=hdbscan);evidence=evidence_prior_weights(X.columns,coding=coding,eqtl=eqtl,xqtl=xqtl,phewas=phewas,noe=noe,alphagenome=alphagenome,weights=evidence_weights) if use_evidence_priors else None;ranks=pd.DataFrame(index=X.columns.map(_normalize_snp_id));fits={};errors={}
    pw=None if evidence is None else evidence['prior_weights'].reindex(ranks.index).to_numpy()
    if 'bayes' in methods or 'mvsusie' in methods:
        try:
            z=multitrait_bayes_finemap_ad(X,Y,G_additive=GA,G_dominance=GD,E=E0,prior_weights=pw,L=L,coverage=coverage,prior_scales=bayes_prior_scales,ld_r=ldz['r'],ld_r2=ldz['r2'],ld_clusters=clusters);fits['bayes']=z;r=z['pip'].assign(snp=lambda d:d.snp.map(_normalize_snp_id)).set_index('snp');ranks['bayes_pip']=r.pip;ranks['bayes_pip_additive']=r.pip_additive;ranks['bayes_pip_dominance']=r.pip_dominance
        except Exception as exc:errors['bayes']=str(exc)
    if 'ridge' in methods:
        try:
            z=multitrait_ridge_finemap_ad(X,Y,G_additive=GA,G_dominance=GD,E=E0,random_state=random_state,compute_shap=compute_linear_shap,shap_mode=shap_mode);fits['ridge']=z;r=_result_table(z,'ranking').rename_axis('snp').reset_index().assign(snp=lambda d:d.snp.map(_normalize_snp_id)).set_index('snp');ranks['ridge']=r.ridge_ad_l2;ranks['ridge_additive']=r.ridge_additive_l2;ranks['ridge_dominance']=r.ridge_dominance_l2
            if 'mean_abs_shap' in r:ranks['ridge_shap']=r.mean_abs_shap
        except Exception as exc:errors['ridge']=str(exc)
    if 'elasticnet' in methods:
        try:
            z=multitrait_elasticnet_finemap_ad(X,Y,G_additive=GA,G_dominance=GD,E=E0,random_state=random_state,compute_shap=compute_linear_shap,shap_mode=shap_mode);fits['elasticnet']=z;r=z['ranking_variant'].assign(snp=lambda d:d.snp.map(_normalize_snp_id)).set_index('snp');ranks['elasticnet']=r.elasticnet_ad_l2;ranks['elasticnet_additive']=r.elasticnet_additive_l2;ranks['elasticnet_dominance']=r.elasticnet_dominance_l2
            if 'mean_abs_shap' in r:ranks['elasticnet_shap']=r.mean_abs_shap
        except Exception as exc:errors['elasticnet']=str(exc)
    if 'sparse_group' in methods:
        try:
            zz=_prepare_ad_finemap_xy(X,Y);fn=[f'{x}|A' for x in zz['snps']]+[f'{x}|D' for x in zz['snps']];F=pd.DataFrame(np.concatenate([zz['A'],zz['D']],axis=1),columns=fn);YY=pd.DataFrame(zz['Y'],columns=zz['traits']);lc=pd.concat([clusters.assign(_suffix='A'),clusters.assign(_suffix='D')]);lc.index=[f'{i}|A' for i in clusters.index]+[f'{i}|D' for i in clusters.index];sk={} if sparse_group_kwargs is None else dict(sparse_group_kwargs);z=multitrait_sparse_group_elasticnet_finemap(F,YY,G=_project_psd(GA+GD,eps=1e-10),E=E0,ld_clusters=lc,random_state=random_state,compute_shap=compute_linear_shap,shap_mode=shap_mode,**sk);fits['sparse_group']=z;rr=z['ranking'].copy();rr['variant']=rr.snp.str.rsplit('|',n=1).str[0];agg=rr.groupby('variant').sparse_group_l2.apply(lambda v:float(np.sqrt(np.sum(np.asarray(v,float)**2))));ranks['sparse_group']=agg.reindex(ranks.index)
        except Exception as exc:errors['sparse_group']=str(exc)
    if 'lightgbm' in methods:
        try:
            z=lightgbm_multitrait_finemap_ad(X,Y,G_additive=GA,G_dominance=GD,E=E0,feature_mode=lightgbm_ad_feature_mode,random_state=random_state,model_kwargs=lightgbm_kwargs);fits['lightgbm']=z;r=z['ranking_variant'].assign(snp=lambda d:d.snp.map(_normalize_snp_id)).set_index('snp');ranks['lightgbm_shap']=r.mean_abs_oof_shap
        except Exception as exc:errors['lightgbm']=str(exc)
    ctab=clusters.copy();ctab.index=ctab.index.map(_normalize_snp_id);ranks=ranks.join(ctab[['ld_component','hdbscan_cluster','hdbscan_confidence','ld_group']],how='left');num=ranks.select_dtypes(include=np.number).drop(columns=['ld_component','hdbscan_cluster','hdbscan_confidence'],errors='ignore');norm=num.rank(pct=True).fillna(0.);stat=norm.mean(1) if len(norm.columns) else pd.Series(0.,index=ranks.index)
    if evidence is not None:
        ev=evidence['evidence_score'].rank(pct=True).reindex(ranks.index).fillna(0.);ranks=ranks.join(evidence['components'],how='left').join(evidence['prior_weights'].rename('evidence_prior'),how='left');ranks['consensus_score']=.85*stat+.15*ev
    else:ranks['consensus_score']=stat
    return {'ranking':ranks.sort_values('consensus_score',ascending=False),'fits':fits,'evidence':evidence,'ld':ldz,'ld_clusters':clusters,'ld_cluster_summary':summarize_ld_clusters(clusters,ldz['r2']),'errors':errors,'genetic_model':'additive+dominance','G_additive':GA,'G_dominance':GD,'E':E0,'version':V3_1_NATIVE_FINEMAP_AD_VERSION}


# ============================================================================
# Fine-map colocalization and efficient A+D genetic mediation
# ============================================================================

def finemap_colocalization(trait_finemap, mediator_finemap, *, ld_clusters=None, pip_col='pip', save_path=None):
    """Approximate shared-signal evidence from two native fine-mapping results.

    This is a fast colocalization screen, not a replacement for a full multi-signal coloc model.
    Variant-level shared mass is proportional to PIP_trait * PIP_mediator. LD-group aggregation
    is also returned because causal variants can be statistically indistinguishable within LD.

    Parameters
    ----------
    trait_finemap : object (see the cited input contract)
        Input for the operation ``pip_table(trait_finemap)``. See
        ``vigor.multivariate_ad.finemap_colocalization`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    mediator_finemap : object (see the cited input contract)
        Input for the operation ``pip_table(mediator_finemap)``. See
        ``vigor.multivariate_ad.finemap_colocalization`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    ld_clusters : sequence or None
        LD-group assignment for each genotype column. Equal labels group linked variants; None
        asks the fitter to infer groups when supported. Default: ``None``.

    pip_col : str
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``'pip'``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: the former ``variant`` table along ``snp`` (descending
        ``shared_product`` order: ``trait_pip``, ``mediator_pip``, ``shared_product``,
        ``shared_pip_product_normalized``) and, when ``ld_clusters`` is given, the former
        ``ld_group`` table along ``ld_group`` (``ld_group_trait_pip``,
        ``ld_group_mediator_pip``, ``ld_group_shared_product``). Attrs:
        ``shared_signal_score`` plus title/software/parameters provenance. Fine-map inputs may
        be DataFrames, mappings with a ``pip`` table, or Datasets with a ``pip_col`` variable.
    """
    import xarray as xr
    if isinstance(ld_clusters,xr.Dataset):ld_clusters=ld_clusters.to_dataframe()
    def pip_table(x):
        """Assemble aligned variant posterior inclusion probabilities and locus annotations.

        Parameters
        ----------
        x : array-like
            Numerical operand in ``x['pip']``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``d[['snp', pip_col]].assign(snp=lambda q: q.snp.map(_normalize_snp_id)).drop_duplicates('snp').set_index('snp')[pip_col]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if isinstance(x,pd.DataFrame):d=x.copy()
        elif isinstance(x,xr.Dataset) and pip_col in x.data_vars:d=x[[pip_col]].to_dataframe()
        elif isinstance(x,dict) and 'pip' in x:d=x['pip'].copy()
        elif isinstance(x,dict) and 'fits' in x and 'bayes' in x['fits']:d=x['fits']['bayes']['pip'].copy()
        else:raise ValueError('fine-map result must provide a PIP table')
        if 'snp' not in d.columns:d=d.reset_index().rename(columns={d.index.name or 'index':'snp'})
        if pip_col not in d.columns:raise ValueError(f"PIP table does not contain '{pip_col}'")
        return d[['snp',pip_col]].assign(snp=lambda q:q.snp.map(_normalize_snp_id)).drop_duplicates('snp').set_index('snp')[pip_col]
    pt,pm=pip_table(trait_finemap),pip_table(mediator_finemap);idx=pt.index.union(pm.index);tab=pd.DataFrame({'trait_pip':pt.reindex(idx).fillna(0.),'mediator_pip':pm.reindex(idx).fillna(0.)});tab['shared_product']=tab.trait_pip*tab.mediator_pip;mass=float(tab.shared_product.sum());tab['shared_pip_product_normalized']=tab.shared_product/max(mass,1e-300)
    group=None
    if ld_clusters is not None:
        lc=ld_clusters.copy();lc.index=lc.index.map(_normalize_snp_id);lab=lc.reindex(idx).ld_group.fillna(pd.Series([f'singleton:{x}' for x in idx],index=idx));group=tab.assign(ld_group=lab).groupby('ld_group').agg(trait_pip=('trait_pip','sum'),mediator_pip=('mediator_pip','sum'),shared_product=('shared_product','sum')).sort_values('shared_product',ascending=False)
    ln={'trait_pip':'trait posterior inclusion probability','mediator_pip':'mediator posterior inclusion probability',
        'shared_product':'PIP_trait x PIP_mediator','shared_pip_product_normalized':'shared PIP product normalized to unit mass'}
    values=_table_vars(tab.sort_values('shared_product',ascending=False).rename_axis('snp'),'variant','snp',long_names=ln)
    if group is not None:values.update(_table_vars(group,'ld_group','ld_group',prefix='ld_group_',long_names=ln))
    values['shared_signal_score']=mass
    ds=result_dataset(values,title='Fine-map colocalization screen',software='vigor.multivariate_ad.finemap_colocalization',
                      parameters={'pip_col':str(pip_col),'ld_clusters_supplied':ld_clusters is not None})
    return maybe_save(ds,save_path)


def _ridge_with_unpenalized_prefix(X,y,n_unpenalized=0,ridge=1e-3):
    """Return the ridge with unpenalized prefix result computed by ``np.linalg.solve``.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    y : array-like
        Numerical operand in ``X.T @ y``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    n_unpenalized : int
        Input for the operation ``P[:int(n_unpenalized), :int(n_unpenalized)]``. See
        ``vigor.multivariate_ad._ridge_with_unpenalized_prefix`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``0``.

    ridge : float or array-like
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``0.001``.
    """
    X=np.asarray(X,float);y=np.asarray(y,float);P=np.eye(X.shape[1])*float(ridge);P[:int(n_unpenalized),:int(n_unpenalized)]=0.;return np.linalg.solve(X.T@X+P,X.T@y)


def mixed_model_mediation_ad(mediator,traits,cis_genotypes,*,background_grm=None,background_covariance=None,
                              n_splits=5,random_state=0,first_stage_ridge=1.,second_stage_ridge=1.,
                              direct_genotype=True,mediation_mode='classical',allele_frequencies=None,solver_tol=1e-6,solver_maxiter=500,
                              preconditioner='diag',blas_threads='auto',sample_ids=None,save_path=None):
    """Cross-fitted genotype -> expression -> trait mediation with A+D local effects.

    If ``background_grm`` and a covariance fit are supplied, the combined mediator/trait matrix
    is first residualized for genome-wide additive/dominance polygenic background using mvPredict.
    The focal cis A/D marker design is used in the mediator equation. With
    ``mediation_mode='classical'`` the observed mediator enters the outcome equation together
    with optional direct A/D genotype paths, matching classical mediation. With ``'iv'`` the
    cross-fitted genetically predicted mediator enters the outcome equation and direct focal
    genotype paths are disallowed because they are collinear with the instrumented mediator
    without additional excluded instruments.

    For SNP j and mediator m, indirect additive effect on trait t is a_A[j,m] * b[m,t]; the
    dominance indirect effect is a_D[j,m] * b[m,t]. Colocalization should be checked separately
    before interpreting these path products causally.

    Parameters
    ----------
    mediator : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    cis_genotypes : object (see the cited input contract)
        Input for the operation ``cis_genotypes.copy()``. See
        ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    background_grm : object (see the cited input contract)
        Input for the operation ``mvPredict(combined, background_grm, Gbg, Ebg,
        solver_tol=solver_tol, solver_maxiter=solver_maxiter, preconditioner=preconditioner,
        blas_threads=blas_threads)``. See ``vigor.multivariate_ad.mixed_model_mediation_ad`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    background_covariance : object (see the cited input contract)
        Input for the operation ``background_covariance is not None``. See
        ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``5``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.

    first_stage_ridge : array-like or numeric scalar
        Input for the operation ``F[tr].T @ F[tr] + float(first_stage_ridge) *
        np.eye(F.shape[1])``. See ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``1.0``.

    second_stage_ridge : array-like or numeric scalar
        Input for the operation ``np.eye(Z.shape[1]) * float(second_stage_ridge)``. See
        ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``1.0``.

    direct_genotype : array-like or numeric scalar
        Input for the operation ``indirectA + (np.nan_to_num(directA) if direct_genotype else
        0.0)``. See ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``True``.

    mediation_mode : str
        Input for the operation ``str(mediation_mode).lower()``. See
        ``vigor.multivariate_ad.mixed_model_mediation_ad`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'classical'``. Source-recognized literal values: ``instrumental``, ``iv``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    allele_frequencies : one-dimensional array-like or None
        Frequency of the coded allele for each genotype variant, on [0, 1]. Length must equal
        the variant axis; None estimates frequencies from the designated training animals.
        Default: ``None``.

    solver_tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-06``. Passed to ``mvPredict`` as ``solver_tol``.

    solver_maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``500``. Passed to ``mvPredict`` as ``solver_maxiter``.

    preconditioner : str
        Input supplied to ``mvPredict`` as ``preconditioner``. See that callable's documented
        ``preconditioner`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``'diag'``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``mvPredict`` as ``blas_threads``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``first_stage_additive``, ``first_stage_dominance`` and their ``_se``
        (snp, mediator); ``mediator_to_trait``, ``mediator_to_trait_se`` (mediator, trait);
        ``direct_additive``, ``direct_dominance`` and their ``_se``, ``indirect_additive``,
        ``indirect_dominance``, ``total_additive``, ``total_dominance`` (snp, trait);
        ``oof_mediator_prediction`` (sample, mediator); ``second_stage_r2``,
        ``second_stage_n`` (trait); ``instrument_strength_oof_r2`` (mediator);
        ``allele_frequency`` (snp); the former ``path_table`` along ``path`` with every
        column prefixed ``path_`` (``path_snp``, ``path_mediator``, ``path_trait``,
        ``path_a_additive``, ..., ``path_indirect_additive_se``, ...). Attrs:
        ``uncertainty_method``, ``background_prediction`` (mvPredict result or None),
        ``background_residualized``, ``mediation_mode``, ``note`` plus
        title/software/parameters provenance.
    """
    from sklearn.model_selection import KFold
    import xarray as xr
    mediator,traits,cis_genotypes=[v.to_pandas() if isinstance(v,xr.DataArray) else v for v in (mediator,traits,cis_genotypes)]
    M=mediator.to_frame() if isinstance(mediator,pd.Series) else (mediator.copy() if isinstance(mediator,pd.DataFrame) else pd.DataFrame(mediator))
    Y=traits.to_frame() if isinstance(traits,pd.Series) else (traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits))
    x_is_df=isinstance(cis_genotypes,pd.DataFrame)
    if x_is_df:
        X=cis_genotypes.copy()
    else:
        xa=np.asarray(cis_genotypes)
        if xa.ndim!=2: raise ValueError('cis_genotypes must be a 2D array/DataFrame')
        if sample_ids is None and (not isinstance(M.index,pd.RangeIndex) or not isinstance(Y.index,pd.RangeIndex)):
            raise ValueError('sample_ids are required when cis_genotypes is an ndarray and mediator/traits carry sample labels')
        idx0=pd.Index(sample_ids) if sample_ids is not None else pd.RangeIndex(len(xa))
        if len(idx0)!=len(xa):raise ValueError('sample_ids length must equal cis_genotypes rows')
        X=pd.DataFrame(xa,index=idx0)
    if len(M)!=len(Y) and isinstance(M.index,pd.RangeIndex) and isinstance(Y.index,pd.RangeIndex):raise ValueError('mediator and traits row counts differ')
    if not M.index.is_unique or not Y.index.is_unique or not X.index.is_unique:raise ValueError('sample IDs must be unique')
    idx=M.index.intersection(Y.index,sort=False).intersection(X.index,sort=False)
    if len(idx)==0:raise ValueError('no shared sample IDs among mediator, traits and cis_genotypes')
    M=M.loc[idx];Y=Y.loc[idx];X=X.loc[idx]
    M.columns=M.columns.map(str);Y.columns=Y.columns.map(str);X.columns=X.columns.map(str);combined=pd.concat([M,Y],axis=1)
    background_prediction=None;work=combined.copy()
    if background_grm is not None and background_covariance is not None:
        cf=background_covariance
        if isinstance(cf,pd.Series):
            Gbg=cf.get('G_components_prediction',cf.get('G_prediction',cf.get('G_components',cf.get('G'))));Ebg=cf.get('E_prediction',cf.get('E'))
        elif isinstance(cf,dict):Gbg=cf.get('G_components_prediction',cf.get('G_prediction',cf.get('G_components',cf.get('G'))));Ebg=cf.get('E_prediction',cf.get('E'))
        elif isinstance(cf,xr.Dataset):
            # Unstack the 3-D component matrices back to {component: DataFrame} for mvPredict.
            cf=_mv._result_view(cf)
            Gbg=cf.get('G_components_prediction',cf.get('G_prediction',cf.get('G_components',cf.get('G'))));Ebg=cf.get('E_prediction',cf.get('E'))
        else:raise ValueError('background_covariance must be a genetic_varcov_PSD result/dict')
        background_prediction=mvPredict(combined,background_grm,Gbg,Ebg,solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner,blas_threads=blas_threads)
        # mvPredict returns breeding values in the input row/column order (DataFrame or labeled DataArray).
        work=combined-pd.DataFrame(np.asarray(background_prediction['breeding_value'],float),index=combined.index,columns=combined.columns)
    Mw=work[M.columns];Yw=work[Y.columns];raw=X.to_numpy(float);valid=np.isfinite(raw);fitrows=Mw.notna().all(1).to_numpy();counts=np.isfinite(raw[fitrows]).sum(0);p=np.asarray(allele_frequencies,float) if allele_frequencies is not None else np.divide(np.nansum(raw[fitrows],0),2*counts,out=np.full(raw.shape[1],np.nan),where=counts>0);p=np.clip(np.nan_to_num(p,nan=.5),1e-6,1-1e-6);q=1-p;A=raw-2*p[None,:];A[~valid]=0.;D=np.where(raw==0,-2*p[None,:]**2,np.where(raw==1,2*p[None,:]*q[None,:],np.where(raw==2,-2*q[None,:]**2,0.)));D[~valid]=0.;sa=np.sqrt(np.mean(A[fitrows]**2,axis=0));sd=np.sqrt(np.mean(D[fitrows]**2,axis=0));sa=np.where(sa>1e-8,sa,1.);sd=np.where(sd>1e-8,sd,1.);A=A/sa;D=D/sd;F=np.concatenate([A,D],axis=1);nm=M.shape[1];nt=Y.shape[1];ns=X.shape[1]
    midx=np.flatnonzero(fitrows);k=min(int(n_splits),max(2,len(midx)//25));kf=KFold(k,shuffle=True,random_state=random_state);Mhat=np.full((len(X),nm),np.nan);Mobs=Mw.to_numpy(float)
    for tr0,va0 in kf.split(midx):
        tr=midx[tr0];va=midx[va0];coef=np.linalg.solve(F[tr].T@F[tr]+float(first_stage_ridge)*np.eye(F.shape[1]),F[tr].T@Mobs[tr]);Mhat[va]=F[va]@coef
    # Full first-stage coefficients are used for path decomposition; OOF predictions are used in stage 2.
    # HC1 sandwich influence functions are retained so indirect-effect delta SEs
    # include the cross-stage covariance term omitted by a classical Sobel SE.
    F1=F[midx]; P1=float(first_stage_ridge)*np.eye(F.shape[1]); inv1=np.linalg.pinv(F1.T@F1+P1)
    coef1=inv1@(F1.T@Mobs[midx]); aA=coef1[:ns]; aD=coef1[ns:]
    res1=Mobs[midx]-F1@coef1; base_if1=F1@inv1.T
    df1=max(int(len(midx)-np.linalg.matrix_rank(F1)),1); hc1_1=float(len(midx)/df1)
    var1=np.empty((F.shape[1],nm),float)
    for _mi in range(nm):
        var1[:,_mi]=hc1_1*((base_if1*base_if1).T@(res1[:,_mi]*res1[:,_mi]))
    se1=np.sqrt(np.maximum(var1,0.0)); aA_se=se1[:ns]; aD_se=se1[ns:]
    b=np.full((nm,nt),np.nan); b_se=np.full((nm,nt),np.nan)
    directA=np.full((ns,nt),np.nan); directD=np.full((ns,nt),np.nan)
    directA_se=np.full((ns,nt),np.nan); directD_se=np.full((ns,nt),np.nan)
    n2=np.zeros(nt,int);r2=np.full(nt,np.nan); second_stage_info=[None]*nt
    mode=str(mediation_mode).lower()
    if mode not in {'classical','iv','instrumental'}:raise ValueError("mediation_mode must be 'classical' or 'iv'")
    if mode in {'iv','instrumental'} and direct_genotype:
        raise ValueError("IV mediation with the same cis variants cannot also include those variants as direct outcome predictors; use direct_genotype=False or classical mode")
    for t in range(nt):
        y=Yw.iloc[:,t].to_numpy(float)
        Mexposure=Mobs if mode=='classical' else Mhat
        ok=np.isfinite(y)&np.isfinite(Mexposure).all(1);n2[t]=int(ok.sum())
        if ok.sum()<max(25,nm+5):continue
        if direct_genotype:Z=np.concatenate([Mexposure[ok],A[ok],D[ok]],axis=1);nu=nm
        else:Z=Mexposure[ok];nu=nm
        # Fit stage 2 and retain HC1 influence geometry. The penalty is excluded
        # for mediator coefficients, matching _ridge_with_unpenalized_prefix.
        P2=np.eye(Z.shape[1])*float(second_stage_ridge); P2[:nu,:nu]=0.0
        inv2=np.linalg.pinv(Z.T@Z+P2); coef2=inv2@(Z.T@y[ok]); b[:,t]=coef2[:nm]
        if direct_genotype:directA[:,t]=coef2[nm:nm+ns];directD[:,t]=coef2[nm+ns:nm+2*ns]
        pred=Z@coef2; resid2=y[ok]-pred; r2[t]=1-float(np.sum(resid2**2))/max(float(np.sum((y[ok]-y[ok].mean())**2)),1e-12)
        df2=max(int(ok.sum()-np.linalg.matrix_rank(Z)),1); hc1_2=float(ok.sum()/df2); base_if2=Z@inv2.T
        var2=hc1_2*((base_if2*base_if2).T@(resid2*resid2)); se2=np.sqrt(np.maximum(var2,0.0))
        b_se[:,t]=se2[:nm]
        if direct_genotype:
            directA_se[:,t]=se2[nm:nm+ns]; directD_se[:,t]=se2[nm+ns:nm+2*ns]
        second_stage_info[t]=(np.flatnonzero(ok),base_if2,resid2,hc1_2)
    indirectA=aA@b;indirectD=aD@b;totalA=indirectA+(np.nan_to_num(directA) if direct_genotype else 0.);totalD=indirectD+(np.nan_to_num(directD) if direct_genotype else 0.)
    def df(B):
        """Return the df result computed by ``pd.DataFrame``.

        Parameters
        ----------
        B : array-like
            Numerical operand in ``pd.DataFrame(B, index=X.columns, columns=Y.columns)``. Shape must
            conform to that matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``pd.DataFrame(B, index=X.columns, columns=Y.columns)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return pd.DataFrame(B,index=X.columns,columns=Y.columns)
    medpaths=[]; zcrit=1.959963984540054
    mpos=np.full(len(X),-1,int); mpos[midx]=np.arange(len(midx))
    from scipy.stats import norm as _norm
    for j,snp in enumerate(X.columns):
        for mi,mn in enumerate(M.columns):
            for ti,tn in enumerate(Y.columns):
                aa=float(aA[j,mi]); ad=float(aD[j,mi]); bb=float(b[mi,ti]) if np.isfinite(b[mi,ti]) else np.nan
                va=float(var1[j,mi]); vd=float(var1[ns+j,mi]); vb=float(b_se[mi,ti]**2) if np.isfinite(b_se[mi,ti]) else np.nan
                cova=covd=np.nan
                info=second_stage_info[ti]
                if info is not None and np.isfinite(bb):
                    okidx,base2,resid2,hc1_2=info; common=(mpos[okidx]>=0)
                    if np.any(common):
                        rr=mpos[okidx[common]]
                        ib=base2[common,mi]*resid2[common]
                        ia=base_if1[rr,j]*res1[rr,mi]; idd=base_if1[rr,ns+j]*res1[rr,mi]
                        scale=(hc1_1*hc1_2)**0.5
                        cova=float(scale*np.sum(ia*ib)); covd=float(scale*np.sum(idd*ib))
                pa=aa*bb; pdm=ad*bb
                vara=(bb*bb*va + aa*aa*vb + 2*aa*bb*cova) if np.isfinite(vb) and np.isfinite(cova) else np.nan
                vard=(bb*bb*vd + ad*ad*vb + 2*ad*bb*covd) if np.isfinite(vb) and np.isfinite(covd) else np.nan
                sea=np.sqrt(max(vara,0.0)) if np.isfinite(vara) else np.nan; sed=np.sqrt(max(vard,0.0)) if np.isfinite(vard) else np.nan
                za=pa/sea if np.isfinite(sea) and sea>0 else np.nan; zd=pdm/sed if np.isfinite(sed) and sed>0 else np.nan
                medpaths.append((snp,mn,tn,aa,aA_se[j,mi],ad,aD_se[j,mi],bb,b_se[mi,ti],pa,sea,pa-zcrit*sea if np.isfinite(sea) else np.nan,pa+zcrit*sea if np.isfinite(sea) else np.nan,2*_norm.sf(abs(za)) if np.isfinite(za) else np.nan,pdm,sed,pdm-zcrit*sed if np.isfinite(sed) else np.nan,pdm+zcrit*sed if np.isfinite(sed) else np.nan,2*_norm.sf(abs(zd)) if np.isfinite(zd) else np.nan,cova,covd))
    path_cols=['snp','mediator','trait','a_additive','a_additive_se','a_dominance','a_dominance_se','b_mediator_trait','b_mediator_trait_se','indirect_additive','indirect_additive_se','indirect_additive_ci95_low','indirect_additive_ci95_high','indirect_additive_p','indirect_dominance','indirect_dominance_se','indirect_dominance_ci95_low','indirect_dominance_ci95_high','indirect_dominance_p','cov_a_additive_b','cov_a_dominance_b']
    values={'first_stage_additive':pd.DataFrame(aA,index=X.columns,columns=M.columns),'first_stage_dominance':pd.DataFrame(aD,index=X.columns,columns=M.columns),
            'first_stage_additive_se':pd.DataFrame(aA_se,index=X.columns,columns=M.columns),'first_stage_dominance_se':pd.DataFrame(aD_se,index=X.columns,columns=M.columns),
            'mediator_to_trait':pd.DataFrame(b,index=M.columns,columns=Y.columns),'mediator_to_trait_se':pd.DataFrame(b_se,index=M.columns,columns=Y.columns),
            'direct_additive':df(directA),'direct_dominance':df(directD),'direct_additive_se':df(directA_se),'direct_dominance_se':df(directD_se),
            'indirect_additive':df(indirectA),'indirect_dominance':df(indirectD),'total_additive':df(totalA),'total_dominance':df(totalD),
            **_table_vars(pd.DataFrame(medpaths,columns=path_cols),'path_table','path',prefix='path_'),
            'uncertainty_method':'HC1 delta method with estimated cross-stage covariance from stacked influence functions',
            'oof_mediator_prediction':pd.DataFrame(Mhat,index=X.index,columns=M.columns),'second_stage_r2':pd.Series(r2,index=Y.columns),
            'second_stage_n':pd.Series(n2,index=Y.columns),'instrument_strength_oof_r2':pd.Series([1-float(np.nansum((Mobs[:,j]-Mhat[:,j])**2))/max(float(np.nansum((Mobs[:,j]-np.nanmean(Mobs[:,j]))**2)),1e-12) for j in range(nm)],index=M.columns),'allele_frequency':pd.Series(p,index=X.columns),
            'background_residualized':background_prediction is not None,'mediation_mode':mode,'note':'OOF mediator R2 is reported as a predictive instrument-strength diagnostic, not as a classical partial-F/Stock-Yogo statistic. Classical mode estimates conditional path coefficients; IV mode requires the exclusion restriction. Indirect-effect SEs use an HC1 delta method including the estimated a-path/b-path covariance. Causal interpretation also requires compatible trait/eQTL fine mapping and confounding assumptions.'}
    sm,mt,st=('snp','mediator'),('mediator','trait'),('snp','trait')
    dims={**{k:sm for k in ('first_stage_additive','first_stage_dominance','first_stage_additive_se','first_stage_dominance_se')},
          'mediator_to_trait':mt,'mediator_to_trait_se':mt,
          **{k:st for k in ('direct_additive','direct_dominance','direct_additive_se','direct_dominance_se','indirect_additive','indirect_dominance','total_additive','total_dominance')},
          'oof_mediator_prediction':('sample','mediator'),'second_stage_r2':('trait',),'second_stage_n':('trait',),
          'instrument_strength_oof_r2':('mediator',),'allele_frequency':('snp',)}
    la={'first_stage_additive':'additive cis effect on mediator (a path)','first_stage_dominance':'dominance cis effect on mediator (a path)',
        'mediator_to_trait':'mediator effect on trait (b path)','direct_additive':'direct additive genotype effect on trait',
        'direct_dominance':'direct dominance genotype effect on trait','indirect_additive':'indirect additive effect a_A b',
        'indirect_dominance':'indirect dominance effect a_D b','total_additive':'total additive effect','total_dominance':'total dominance effect'}
    var_attrs={**{k:{'long_name':v} for k,v in la.items()},**{k+'_se':{'long_name':'HC1 standard error of '+v} for k,v in la.items() if k+'_se' in dims},
               'oof_mediator_prediction':{'long_name':'cross-fitted genetically predicted mediator'},
               'second_stage_r2':{'long_name':'second-stage in-sample R2'},'second_stage_n':{'long_name':'second-stage complete observations'},
               'instrument_strength_oof_r2':{'long_name':'out-of-fold mediator R2 (instrument strength diagnostic)'},
               'allele_frequency':{'long_name':'coded-allele frequency'}}
    ds=result_dataset(values,dims=dims,var_attrs=var_attrs,attrs={'background_prediction':background_prediction},
                      title='A+D genetic mediation',software='vigor.multivariate_ad.mixed_model_mediation_ad',
                      parameters={'n_splits':int(n_splits),'random_state':random_state,'first_stage_ridge':float(first_stage_ridge),
                                  'second_stage_ridge':float(second_stage_ridge),'direct_genotype':bool(direct_genotype),
                                  'mediation_mode':mode,'solver_tol':float(solver_tol),'solver_maxiter':int(solver_maxiter)})
    return maybe_save(ds,save_path)

