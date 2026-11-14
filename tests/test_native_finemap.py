"""Behavioral tests at the native fine-mapper seam; no mocked inference."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import null_space
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import pandas as pd
import pytest

from vigor.multivariate import multitrait_bayes_finemap


def test_single_effect_matches_independent_observation_space_gaussian_oracle():
    # Integrate out b in observation space, independently of normal-means ABF code.
    rng = np.random.default_rng(741)
    X = rng.normal(size=(24, 3))
    Y = rng.normal(size=(24, 2)) + X[:, [1]] @ np.array([[.7, -.4]])
    E = np.array([[1.1, -.72], [-.72, .9]])
    G = np.array([[.8, .2], [.2, .6]])
    fit = multitrait_bayes_finemap(
        X, Y, G=G, E=E, L=1, prior_scales=(.15,),
        prior_weights=[.2, .5, .3], estimate_residual_scale=False,
        residual_covariance='provided', max_iter=8, tol=1e-9)
    T = fit['canonical_basis']['transform']
    Ecan = T.T @ E @ T
    # Project away the intercept: likelihood lives on n-1 independent rows.
    P = null_space(np.ones((1, len(X))))
    Z = P.T @ ((X-X.mean(0))/X.std(0))
    y = (P.T @ Y @ T).ravel()
    base = np.kron(np.eye(len(X)-1), Ecan)
    null_logpdf = multivariate_normal.logpdf(y, cov=base)
    W = list(fit['prior_covariances'].values())
    lq = np.log(fit['prior_mixture_weights'].to_numpy())
    bf = []
    for j in range(X.shape[1]):
        bf.append(logsumexp([
            multivariate_normal.logpdf(y, cov=base + np.kron(np.outer(Z[:, j], Z[:, j]), w))
            - null_logpdf + weight for w, weight in zip(W, lq)]))
    la = np.log([.2, .5, .3])+bf
    expected = np.exp(la-logsumexp(la))
    assert_allclose(fit['alpha'].iloc[0], expected, rtol=2e-10, atol=2e-12)
    assert_allclose(fit['component_diagnostics'].conditional_logbf,
                    [logsumexp(la)], rtol=2e-10)
    assert_allclose(fit['residual_covariance_canonical'], Ecan, atol=1e-10)


def test_first_covariance_update_matches_dense_posterior_oracle():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(25, 3))
    Y = rng.normal(size=(25, 2)) + X[:, [0]] @ [[1.2, -.6]]
    fit = multitrait_bayes_finemap(X, Y, L=1, max_iter=1, covariance_shrinkage=3)
    T = fit['canonical_basis']['transform']
    P = null_space(np.ones((1, len(X))))
    Z = P.T @ ((X-X.mean(0))/X.std(0))
    y = (P.T @ Y @ T).ravel()
    E0 = fit['residual_covariance_trace'][0]
    noise = np.kron(np.eye(len(X)-1), E0)
    predictions, variances, evidence = [], [], []
    # Integrate SNP choice and mixture in the FULL observation space.
    for j in range(3):
        for W, qw in zip(fit['prior_covariances'].values(), fit['prior_mixture_weights']):
            K = np.kron(np.outer(Z[:, j], Z[:, j]), W)
            total = noise + K
            predictions.append(K @ np.linalg.solve(total, y))
            variances.append(K-K @ np.linalg.solve(total, K))
            evidence.append(multivariate_normal.logpdf(y, cov=total)+np.log(qw/3))
    weights = np.exp(evidence-logsumexp(evidence))
    mean = sum(w*m for w, m in zip(weights, predictions))
    covariance = sum(w*(V+np.outer(m, m)) for w, V, m in zip(weights, variances, predictions))-np.outer(mean, mean)
    uncertainty = sum(covariance[2*i:2*i+2, 2*i:2*i+2] for i in range(len(X)-1))
    residual = (y-mean).reshape(-1, 2)
    crossproduct = residual.T@residual+uncertainty
    expected = (crossproduct+3*fit['covariance_target'])/(len(X)-1+3)
    assert_allclose(fit['posterior_predictive_uncertainty'], uncertainty, rtol=1e-9, atol=1e-10)
    assert_allclose(fit['expected_residual_crossproduct'], crossproduct, rtol=1e-9)
    assert_allclose(fit['residual_covariance_canonical'], expected, rtol=1e-9)


@pytest.mark.parametrize('t,sign,strength', [(1,1,0), (1,1,1.5), (2,1,.15), (2,1,1.5), (2,-1,1.5), (3,-1,.8)])
def test_finite_psd_normalized_and_monotone(t, sign, strength):
    rng = np.random.default_rng(102+t)
    X = rng.normal(size=(130, 45))
    E = np.eye(t)
    if t>=2:E[0,1]=E[1,0]=-.85
    Y = rng.multivariate_normal(np.zeros(t), E, len(X))
    Y += X[:, [2]] @ (strength*np.array([1]+[sign]*(t-1)))[None,:]
    fit = multitrait_bayes_finemap(X, Y, L=3, max_iter=80, tol=1e-6)
    assert np.isfinite(fit['coef']).all().all()
    assert_allclose(fit['alpha'].sum(axis=1), 1, atol=1e-12)
    assert_allclose(fit['pip'].set_index('snp').reindex(fit['alpha'].columns).pip,
                    1-np.prod(1-fit['alpha'].to_numpy(), axis=0), atol=1e-12)
    assert np.linalg.eigvalsh(fit['residual_covariance_trace']).min()>0
    objectives=np.array([v['regularized_elbo'] for v in fit['objective_trace']])
    assert np.diff(objectives).min(initial=0)>-1e-8
    assert np.linalg.eigvalsh(fit['posterior_predictive_uncertainty']).min()>-1e-8
    if strength>=1.5:
        assert fit['pip'].iloc[0].snp=='v2'
        assert len(fit['selected_credible_set_indices'])>=1


def test_order_scale_prior_and_complete_case_invariances():
    # Invariances compare candidate to itself; independent oracle tests above
    # are the independent mathematical evidence.
    rng=np.random.default_rng(653)
    X=pd.DataFrame(rng.normal(size=(100, 24)), columns=[f's{i}' for i in range(24)])
    X.iloc[1,3]=np.nan
    Y=pd.DataFrame(rng.normal(size=(100,2))+X.fillna(0).iloc[:,[5]].to_numpy() @ [[1.,-.5]])
    Y.iloc[4,0]=np.nan;Y.iloc[6,1]=np.nan
    w=pd.Series(np.arange(1,25),index=X.columns,dtype=float);w.iloc[3]=0
    settings=dict(L=2,max_iter=80,tol=1e-7)
    a=multitrait_bayes_finemap(X,Y,prior_weights=w,**settings)
    order=rng.permutation(24)
    b=multitrait_bayes_finemap(X.iloc[:,order],Y,prior_weights=w*123.,**settings)
    assert_allclose(a['alpha'],b['alpha'].reindex(columns=X.columns),atol=1e-10)
    c=multitrait_bayes_finemap(X.loc[Y.notna().all(1)],Y.dropna(),prior_weights=w,**settings)
    assert_allclose(a['alpha'],c['alpha'],atol=1e-12)
    d=multitrait_bayes_finemap(X,Y*np.array([100.,.03]),prior_weights=w,**settings)
    assert_allclose(a['alpha'],d['alpha'],atol=1e-9)
    assert_allclose(a['coef']*np.array([100.,.03]),d['coef'],atol=1e-8)
    assert (a['alpha']['s3']==0).all()
    assert a['n_complete']==98
    repeat=multitrait_bayes_finemap(X,Y,prior_weights=w,**settings)
    assert_allclose(a['alpha'],repeat['alpha'],atol=0,rtol=0)


def test_p_greater_n_near_perfect_ld_constants_low_rank_and_nonconvergence():
    rng=np.random.default_rng(891)
    X=rng.normal(size=(45,100));X[:,1]=X[:,0];X[:,2]=0
    Y=rng.normal(size=(45,2))+X[:,[0]] @ [[2.,2.]]
    fit=multitrait_bayes_finemap(X,Y,G=np.ones((2,2)),E=np.ones((2,2)),L=2,max_iter=1)
    assert not fit['converged']
    assert np.isfinite(fit['alpha']).all().all()
    assert_allclose(fit['alpha'].iloc[:,0],fit['alpha'].iloc[:,1],atol=1e-9)
    assert np.linalg.eigvalsh(fit['residual_covariance']).min()>0
    # A constant SNP has BF=1 and zero predicted contribution, not fabricated x'x=1.
    z=multitrait_bayes_finemap(np.ones((45,1)),Y,L=1,max_iter=5)
    assert_allclose(z['coef'],0,atol=0)
    assert_allclose(z['component_diagnostics'].conditional_logbf,0,atol=1e-12)
    assert z['selected_credible_sets'].empty


@pytest.mark.parametrize('bad', ['all_missing_x','disjoint_traits','bad_weights','negative_weights','bad_rows','bad_L','bad_mode','bad_scale'])
def test_rejects_invalid_or_unsupported_inputs(bad):
    rng=np.random.default_rng(56)
    X=rng.normal(size=(30,4));Y=rng.normal(size=(30,2));kw={}
    if bad=='all_missing_x':X[:,0]=np.nan
    if bad=='disjoint_traits':Y[:15,0]=np.nan;Y[15:,1]=np.nan
    if bad=='bad_weights':kw['prior_weights']=[0]*4
    if bad=='negative_weights':kw['prior_weights']=[1,1,-1,1]
    if bad=='bad_rows':Y=Y[:29]
    if bad=='bad_L':kw['L']=0
    if bad=='bad_mode':kw['residual_covariance']='oracle'
    if bad=='bad_scale':kw['prior_scales']=[-1]
    with pytest.raises(ValueError):multitrait_bayes_finemap(X,Y,**kw)


def test_global_null_permutation_and_raw_sets_are_distinct():
    rng=np.random.default_rng(350)
    X=rng.normal(size=(65,20));Y=rng.normal(size=(65,2))
    fit=multitrait_bayes_finemap(X,Y,L=2,max_iter=30,null_permutations=19,random_state=7)
    assert len(fit['credible_set_indices'])==2
    assert fit['selected_credible_sets'].empty
    assert fit['selection_calibration']=='exchangeable_global_null_permutation'
    assert len(fit['null_max_logbf'])==19
    assert (fit['component_diagnostics'].global_null_pvalue>=.05).all()
