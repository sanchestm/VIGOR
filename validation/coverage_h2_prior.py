"""Simulation check of final-sandwich SEs and h2/vG/rG interval coverage.

Run with an installed VIGOR checkout:

    python validation/coverage_h2_prior.py --out DIR [--reps 100] [--animals 600]

Each replicate simulates correlated traits from a factor genetic and residual
covariance on one relatedness kernel, applies project-wise heterogeneous
missingness, fits genetic_varcov_PSD with the h2 prior off and on, and records
whether each interval contains the true value. With the prior on, the robust
EBCI target is average coverage across a group's traits, so both the pooled and
the per-trait coverage are reported. A second table compares the 'paired',
'grm2us' and 'cholesky' kernel backends on one replicate at matched Monte Carlo
precision. Outputs CSV files only; nothing is plotted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import vigor
from vigor.multivariate import _result_view


def simulate(rng, n, t, projects, missing):
    """Kernel, phenotypes and the true G, E for one replicate."""
    x = rng.binomial(2, rng.uniform(.05, .5, 2000), size=(n, 2000)).astype(float)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)
    K = x @ x.T / x.shape[1]
    K /= np.mean(np.diag(K))
    load_g = rng.normal(size=(t, 3)) * .45
    load_e = rng.normal(size=(t, 3)) * .45
    G = load_g @ load_g.T + np.diag(rng.uniform(.05, .2, t))
    E = load_e @ load_e.T + np.diag(rng.uniform(.2, .5, t))
    Lk = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    y = Lk @ rng.normal(size=(n, t)) @ np.linalg.cholesky(G).T + rng.normal(size=(n, t)) @ np.linalg.cholesky(E).T
    index = [f'a{i}' for i in range(n)]
    columns = [f'p{j % projects}_t{j}' for j in range(t)]
    Y = pd.DataFrame(y, index=index, columns=columns)
    # Each project measures its own random subset of animals, plus scattered cell loss.
    for p in range(projects):
        absent = rng.random(n) < missing
        cols = [c for c in columns if c.startswith(f'p{p}_')]
        Y.loc[absent, cols] = np.nan
    Y = Y.mask(rng.random(Y.shape) < .05)
    return pd.DataFrame(K, index=index, columns=index), Y, G, E


def covered(low, high, truth):
    low = np.asarray(low, float); high = np.asarray(high, float)
    ok = np.isfinite(low) & np.isfinite(high)
    return np.where(ok, (low <= truth) & (truth <= high), np.nan)


def replicate(seed, args, prior):
    rng = np.random.default_rng(seed)
    K, Y, G, E = simulate(rng, args.animals, args.traits, args.projects, args.missing)
    groups = pd.Series({c: c.split('_', 1)[0] for c in Y.columns})
    fit = _result_view(vigor.genetic_varcov_PSD(
        K, Y, solver='lbfgs', rank_G=args.rank, rank_E=args.rank, n_boot=0, progress=False,
        analytic_se='final', random_state=seed, blas_threads=args.threads,
        h2_prior_strength=1.0 if prior else 0.0, h2_prior_groups=groups if prior else None))
    gt, et = np.diag(G), np.diag(E)
    h2_true = gt / (gt + et)
    rg_true = G / np.sqrt(np.outer(gt, gt))
    h2_ci = np.asarray(fit['h2_ci95'], float)
    gv_ci = np.asarray(fit['G_variance_ci95'], float)
    h2 = np.asarray(fit['h2'], float); h2_se = np.asarray(fit['h2_se'], float)
    iu = np.triu_indices(args.traits, 1)
    rg_cov = covered(np.asarray(fit['rg_ci95_low'], float)[iu], np.asarray(fit['rg_ci95_high'], float)[iu], rg_true[iu])
    meta = fit.get('analytic_se', {})
    rows = []
    for j, trait in enumerate(Y.columns):
        rows.append({'seed': seed, 'prior': prior, 'trait': trait, 'group': groups[trait],
                     'h2_true': h2_true[j], 'h2': h2[j], 'h2_se': h2_se[j],
                     'z': (h2[j] - h2_true[j]) / h2_se[j] if h2_se[j] > 0 else np.nan,
                     'h2_covered': covered(h2_ci[j, 0], h2_ci[j, 1], h2_true[j]),
                     'vG_covered': covered(gv_ci[j, 0], gv_ci[j, 1], gt[j]),
                     'n_probes': meta.get('n_probes'), 'mc_target_met': meta.get('mc_target_met')})
    return pd.DataFrame(rows), float(np.nanmean(rg_cov))


def backend_comparison(args):
    rng = np.random.default_rng(args.seed + 999)
    K, Y, _, _ = simulate(rng, args.animals, args.traits, args.projects, args.missing)
    out = {}
    for backend in ('cholesky', 'grm2us', 'paired'):
        fit = _result_view(vigor.genetic_varcov_PSD(
            K, Y, solver='lbfgs', rank_G=args.rank, rank_E=args.rank, n_boot=0, progress=False,
            analytic_se='final', analytic_kernel_factor=backend, analytic_se_mc_rtol=.05,
            random_state=args.seed, blas_threads=args.threads))
        out[backend] = pd.Series(np.asarray(fit['h2_se'], float), index=Y.columns)
        out[backend + '_vG_se'] = pd.Series(np.asarray(fit['G_variance_se'], float), index=Y.columns)
    table = pd.DataFrame(out)
    for backend in ('grm2us', 'paired'):
        table[backend + '_over_cholesky'] = table[backend] / table['cholesky']
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--animals', type=int, default=600)
    parser.add_argument('--traits', type=int, default=12)
    parser.add_argument('--projects', type=int, default=3)
    parser.add_argument('--missing', type=float, default=.35)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2120)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    traits, rg = [], []
    for r in range(args.reps):
        for prior in (False, True):
            table, rg_cov = replicate(args.seed + r, args, prior)
            traits.append(table)
            rg.append({'seed': args.seed + r, 'prior': prior, 'rg_coverage': rg_cov})
    traits = pd.concat(traits, ignore_index=True)
    traits.to_csv(args.out / 'coverage_by_trait_replicate.csv', index=False)
    pooled = traits.groupby('prior').agg(h2_coverage=('h2_covered', 'mean'), vG_coverage=('vG_covered', 'mean'),
                                         z_sd=('z', 'std'), z_mean=('z', 'mean'),
                                         mc_target_met=('mc_target_met', 'mean'))
    pooled = pooled.join(pd.DataFrame(rg).groupby('prior')['rg_coverage'].mean())
    pooled.to_csv(args.out / 'coverage_pooled.csv')
    per_trait = traits.groupby(['prior', 'trait']).agg(h2_true=('h2_true', 'mean'), coverage=('h2_covered', 'mean'),
                                                       z_sd=('z', 'std'))
    per_trait.to_csv(args.out / 'coverage_per_trait.csv')
    backend_comparison(args).to_csv(args.out / 'backend_comparison.csv')
    print(pooled.to_string())
    print('\nNominal 0.95. Under the prior, the robust EBCI targets average coverage across a group;'
          ' see coverage_per_trait.csv for traits far from their group mean.')


if __name__ == '__main__':
    main()
