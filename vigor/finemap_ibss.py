"""Numerical IBSS in a fixed trait basis, including posterior covariance ECM.

No annotation, truth, data loading or trait whitening occurs here. Rows must be
independent under the likelihood; centered responses use n-1 row degrees of freedom.
"""
import numpy as np
from scipy.special import logsumexp


def _sym(a):
    return (a + a.T) * .5


def _floor_covariance(cov, target_diag, floor):
    """Constrain eigenvalues after scaling by a fixed null diagonal target."""
    sd = np.sqrt(target_diag)
    w, v = np.linalg.eigh(_sym(cov / sd[:, None] / sd[None, :]))
    return _sym(((v * np.maximum(w, floor)) @ v.T) * sd[:, None] * sd[None, :])


def _gaussian_factors(x2, E, priors):
    """Cache shared small solves; no inverse of a possibly singular prior."""
    positive = np.flatnonzero(x2 > 1e-12)
    if not len(positive):
        return []
    ref = float(np.median(x2[positive]))
    if np.max(np.abs(x2[positive] / ref - 1)) <= 1e-10:
        groups = [(positive, ref)]
    else:
        groups = [(np.flatnonzero(x2 == value), value) for value in np.unique(x2[positive])]
    factors = []
    for indices, value in groups:
        V = E / value
        invV = np.linalg.solve(V, np.eye(len(E)))
        qf = []
        for W in priors:
            VW = V + W
            invVW = np.linalg.solve(VW, np.eye(len(E)))
            A = W @ invVW
            S = _sym(W - A @ W)
            logdet = .5 * (np.linalg.slogdet(V)[1] - np.linalg.slogdet(VW)[1])
            qf.append((logdet, _sym(invV - invVW), A, S))
        factors.append((indices, qf))
    return factors


def _single_effect(bhat, x2, factors, priors, logpi, logqw):
    p, t = bhat.shape
    q = len(priors)
    logbfq = np.zeros((p, q))
    means = np.zeros((q, p, t))
    for indices, qf in factors:
        b = bhat[indices]
        for k, (logdet, precision, A, _) in enumerate(qf):
            logbfq[indices, k] = logdet + .5 * np.sum((b @ precision) * b, axis=1)
            means[k, indices] = b @ A.T
    joint = logbfq + logqw[None, :]
    logbf = logsumexp(joint, axis=1)
    gamma = np.exp(joint - logbf[:, None])
    evidence = float(logsumexp(logpi + logbf))
    alpha = np.exp(logpi + logbf - evidence)
    mean = np.einsum('jq,qjt->jt', gamma, means, optimize=True)
    # E[B_l' X'X B_l]: SNP and mixture uncertainty are included before
    # subtracting the fitted-mean crossproduct at the sweep level.
    second = np.zeros((t, t))
    for indices, qf in factors:
        for k, (_, _, _, S) in enumerate(qf):
            weight = alpha[indices] * gamma[indices, k] * x2[indices]
            m = means[k, indices]
            second += weight.sum() * S + m.T @ (weight[:, None] * m)
    return alpha, mean, _sym(second), evidence


def fit_ibss(X, Y, priors, pi, qw, *, L, max_iter, tol, residual_mode,
             initial_covariance, covariance_shrinkage, covariance_floor,
             residual_update_interval):
    """Fit centered complete observations; return moments and objective diagnostics.

    The covariance penalty is k/2*(logdet(E)+tr(D inv(E))), with fixed
    diagonal D from the intercept-only model and k pseudo-observations.
    Covariance flooring is a constrained maximization in D-scaled units.
    """
    n, p = X.shape
    t = Y.shape[1]
    df = n - 1
    x2 = np.sum(X * X, axis=0)
    target_diag = np.maximum(np.sum(Y * Y, axis=0) / df, 1e-12)
    target = np.diag(target_diag)
    k = float(t + 1 if covariance_shrinkage is None else covariance_shrinkage)
    if residual_mode == 'fit':
        E = _floor_covariance((Y.T @ Y + k * target) / (df + k), target_diag, covariance_floor)
    elif residual_mode == 'diagonal':
        E = np.diag(np.maximum(target_diag, 1e-6))
    else:
        E = np.array(initial_covariance, float, copy=True)
    logpi = np.full(p, -np.inf)
    np.log(pi, out=logpi, where=pi > 0)
    logqw = np.full(len(qw), -np.inf)
    np.log(qw, out=logqw, where=qw > 0)
    alpha = np.tile(pi, (L, 1))
    means = np.zeros((L, p, t))
    mu = np.zeros((L, n, t))
    second = np.zeros((L, t, t))
    kl = np.zeros(L)
    conditional_trace, convergence_trace, objective_trace = [], [], []
    covariance_trace = [E.copy()]
    converged = False
    factors = _gaussian_factors(x2, E, priors)
    for iteration in range(max_iter):
        old_alpha = alpha.copy()
        old_effect = np.sum(alpha[:, :, None] * means, axis=0)
        old_prediction = mu.sum(0)
        old_E = E.copy()
        fitted = old_prediction.copy()
        invE = np.linalg.solve(E, np.eye(t))
        conditional_sum = 0.
        for ell in range(L):
            other = fitted - mu[ell]
            residual = Y - other
            bhat = np.divide(X.T @ residual, x2[:, None],
                             out=np.zeros((p, t)), where=x2[:, None] > 1e-12)
            alpha[ell], means[ell], second[ell], evidence = _single_effect(
                bhat, x2, factors, priors, logpi, logqw)
            mu[ell] = X @ (alpha[ell, :, None] * means[ell])
            fitted = other + mu[ell]
            # log q/prior = log likelihood ratio - log integrated BF.
            # This identity remains valid for singular Gaussian priors.
            kl[ell] = (np.sum((mu[ell] @ invE) * residual)
                       - .5 * np.trace(invE @ second[ell]) - evidence)
            conditional_sum += evidence
        uncertainty = _sym(sum(second[ell] - mu[ell].T @ mu[ell] for ell in range(L)))
        residual = Y - fitted
        crossproduct = _sym(residual.T @ residual + uncertainty)
        update_due = (iteration + 1) % residual_update_interval == 0
        if residual_mode == 'fit' and update_due:
            E = _floor_covariance((crossproduct + k * target) / (df + k),
                                  target_diag, covariance_floor)
            factors = _gaussian_factors(x2, E, priors)
        invE = np.linalg.solve(E, np.eye(t))
        logdet = np.linalg.slogdet(E)[1]
        elbo = -.5 * (df * (t * np.log(2 * np.pi) + logdet)
                     + np.trace(invE @ crossproduct)) - kl.sum()
        penalty = -.5 * k * (logdet + np.trace(invE @ target)) if residual_mode == 'fit' else 0.
        objective = float(elbo + penalty)
        effect = np.sum(alpha[:, :, None] * means, axis=0)
        rel = lambda a, b: float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-12))
        changes = dict(alpha_delta=float(np.max(np.abs(alpha - old_alpha))),
                       effect_relative_delta=rel(effect, old_effect),
                       prediction_relative_delta=rel(fitted, old_prediction),
                       covariance_relative_delta=rel(E, old_E),
                       objective_relative_delta=(abs(objective-objective_trace[-1]['regularized_elbo'])
                           / max(1., abs(objective)) if objective_trace else np.inf),
                       conditional_logbf_relative_delta=(abs(conditional_sum-conditional_trace[-1])
                           / max(1., abs(conditional_sum)) if conditional_trace else np.inf))
        convergence_trace.append(changes)
        conditional_trace.append(conditional_sum)
        objective_trace.append(dict(elbo=float(elbo), covariance_penalty=float(penalty),
                                    regularized_elbo=objective, kl=float(kl.sum())))
        covariance_trace.append(E.copy())
        check = [v for key, v in changes.items() if key != 'conditional_logbf_relative_delta']
        if max(check) < tol and (residual_mode != 'fit' or update_due):
            converged = True
            break
    # Evaluate final conditional evidence at the returned E and other means.
    component_logbf = []
    fitted = mu.sum(0)
    for ell in range(L):
        bhat = np.divide(X.T @ (Y-fitted+mu[ell]), x2[:, None],
                         out=np.zeros((p, t)), where=x2[:, None] > 1e-12)
        component_logbf.append(_single_effect(bhat, x2, factors, priors, logpi, logqw)[3])
    return dict(alpha=alpha, postmean=means, fitted=fitted, residual_covariance=E,
                residual_covariance_trace=np.asarray(covariance_trace),
                posterior_predictive_uncertainty=uncertainty,
                expected_residual_crossproduct=crossproduct,
                covariance_target=target, covariance_pseudocount=k if residual_mode == 'fit' else 0.,
                conditional_logbf_trace=np.asarray(conditional_trace),
                component_logbf=np.asarray(component_logbf), convergence_trace=convergence_trace,
                objective_trace=objective_trace, converged=converged, n_iter=iteration+1)
