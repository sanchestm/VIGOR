"""Bootstrap utilities preserving VIGOR 2.6 weighted/no-GRM-copy acceleration.

VIGOR 2.7 adds family/cluster resampling by translating cluster draws into animal
multiplicity vectors.  It deliberately does not materialise K[idx, idx].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class GeneticVarCovBootstrapCache:
    """GeneticVarCovBootstrapCache represents genetic var covariance bootstrap cache.

    Attributes
    ----------
    grm : Any
        Stored grm; its axis/order follows the construction inputs.
    traits : pd.DataFrame
        Stored traits; its axis/order follows the construction inputs.
    block_size : int
        Stored block size; its axis/order follows the construction inputs.
    cache_indices : bool
        Stored cache indices; its axis/order follows the construction inputs.
    _counts : list | None
        Stored  counts; its axis/order follows the construction inputs.
    _key : tuple | None
        Stored  key; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    grm : Any
        Stored grm supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    traits : pd.DataFrame
        Stored traits consumed by ``bootstrap_counts``; see those methods for the
        numerical/label contract. Required construction field.

    block_size : int
        Stored block size supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: ``512``.

    cache_indices : bool
        Stored cache indices consumed by ``bootstrap_counts``; see those methods for the
        numerical/label contract. Default: ``True``.

    _counts : list | None
        Stored  counts consumed by ``bootstrap_counts``; see those methods for the
        numerical/label contract. Default: ``None``. Internal field (init=False), excluded from
        the generated constructor.

    _key : tuple | None
        Stored  key consumed by ``bootstrap_counts``; see those methods for the numerical/label
        contract. Default: ``None``. Internal field (init=False), excluded from the generated
        constructor.
    """
    grm: Any
    traits: pd.DataFrame
    block_size: int = 512
    cache_indices: bool = True
    _counts: list | None = field(default=None, init=False, repr=False)
    _key: tuple | None = field(default=None, init=False, repr=False)

    def bootstrap_counts(self, n_boot: int, random_state=0, *, groups=None):
        """Bootstrap counts.

        Parameters
        ----------
        n_boot : int
            Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
            estimate sampling variation by the resampling scheme used here.

        random_state : int or None
            Seed controlling randomized sampling, folds or projections for reproducibility. None
            uses the selected random-number backend without a fixed seed. Default: ``0``.

        groups : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if isinstance(n_boot, (bool, np.bool_)) or int(n_boot) != n_boot or n_boot < 0:
            raise ValueError("n_boot must be a nonnegative integer")
        gkey = None if groups is None else tuple(np.asarray(groups).astype(str))
        # None requests a fresh stochastic draw; do not reuse a previous draw.
        cacheable = self.cache_indices and random_state is not None
        key = (int(n_boot), random_state, gkey)
        if cacheable and self._counts is not None and self._key == key:
            return self._counts
        rng=np.random.default_rng(random_state); n=len(self.traits)
        if groups is None:
            out=[np.bincount(rng.integers(0,n,size=n),minlength=n).astype(np.int32) for _ in range(int(n_boot))]
        else:
            g=np.asarray(groups)
            if len(g)!=n: raise ValueError("groups must have one label per animal")
            ug,inv=np.unique(g.astype(str),return_inverse=True)
            out=[]
            for _ in range(int(n_boot)):
                gc=np.bincount(rng.integers(0,len(ug),size=len(ug)),minlength=len(ug))
                out.append(gc[inv].astype(np.int32))
        if cacheable: self._counts,self._key=out,key
        return out

    def bootstrap_indices(self,n_boot:int,random_state=0,*,groups=None):
        # Compatibility helper.  The production covariance bootstrap consumes counts.
        """Bootstrap indices.

        Parameters
        ----------
        n_boot : int
            Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
            estimate sampling variation by the resampling scheme used here.

        random_state : int or None
            Seed controlling randomized sampling, folds or projections for reproducibility. None
            uses the selected random-number backend without a fixed seed. Default: ``0``.

        groups : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value. Passed to ``self.bootstrap_counts`` as ``groups``.

        Returns
        -------
        object
            Computed result from ``[np.repeat(np.arange(len(c)), c) for c in self.bootstrap_counts(n_boot, random_state, groups=groups)]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return [np.repeat(np.arange(len(c)), c) for c in self.bootstrap_counts(n_boot,random_state,groups=groups)]

def fit_psd_from_moments(moments, *, rank_G='auto', rank_E='auto', **kwargs):
    """Fit positive-semidefinite covariance from moments.

    Parameters
    ----------
    moments : mapping
        Method-of-moments tensors: A contains component-pair × trait-pair coefficients and B
        contains component × trait-pair response moments. Component and trait ordering must
        agree with ranks, names and covariance specifications.

    rank_G : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``'auto'``.

    rank_E : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``'auto'``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``fit_mom_factor_analytic``. See those
        callables for accepted keys and values. Passed to ``fit_mom_factor_analytic`` as
        ``**expanded mapping``.

    Returns
    -------
    object
        Result from ``fit_mom_factor_analytic`` after translating the named
        genetic/residual ranks into its component-ordered ``ranks`` argument.
    """
    from .multivariate import fit_mom_factor_analytic, _rank_value, _v31_auto_trait_rank
    A = np.asarray(moments['A'])
    B = np.asarray(moments['B'])
    if A.ndim != 4 or A.shape[0] != A.shape[1] or A.shape[2] != A.shape[3] or B.shape != (A.shape[0], A.shape[2], A.shape[3]):
        raise ValueError("moments require A=(c,c,t,t), B=(c,t,t)")
    c, _, t, _ = A.shape
    names = kwargs.pop('names', list(moments.get('kernel_names', [f'G{i}' for i in range(c-1)])) + ['E'])
    if len(names) != c or names[-1] != 'E' or len(set(names)) != c:
        raise ValueError("names must uniquely label every component, with E last")
    def rank(value):
        """Resolve one component rank; auto uses the common trait-rank policy."""
        return _v31_auto_trait_rank(t) if value == 'auto' else _rank_value(value, t)
    genetic = [rank(rank_G.get(name, 'full') if isinstance(rank_G, dict) else rank_G) for name in names[:-1]]
    if 'ranks' in kwargs:
        raise TypeError("use rank_G/rank_E; ranks is constructed from the named moment components")
    return fit_mom_factor_analytic(moments, ranks=genetic + [rank(rank_E)], names=names, **kwargs)

def bootstrap_genetic_varcov_psd(grm,traits,*,n_boot=100,random_state=0,groups=None,bootstrap_unit='auto',
                                 retain_factor_draws='auto',bootstrap_factors=None,n_jobs='auto',**fit_kwargs):
    """Efficient FA covariance bootstrap with optional family/cluster resampling.

    This is a thin public wrapper around the 2.6 engine.  Both individual and cluster
    resampling are encoded as count weights and reuse the original GRM/operator.

    Parameters
    ----------
    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    n_boot : int
        Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
        estimate sampling variation by the resampling scheme used here. Default: ``100``. Passed
        to ``genetic_varcov_PSD`` as ``n_boot``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``genetic_varcov_PSD`` as ``random_state``.

    groups : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value. Passed to ``genetic_varcov_PSD`` as
        ``bootstrap_groups``.

    bootstrap_unit : str
        Resampling unit: 'auto' chooses clusters when bootstrap_groups is supplied and
        individuals otherwise; 'individual' samples animals separately; 'cluster' samples whole
        groups and requires one group label per animal. Default: ``'auto'``.

    retain_factor_draws : bool or str
        Whether to retain aligned factor draws. In bootstrap_factor_analytic, 'auto' checks the
        complete loading-draw allocation against 512 MiB; True retains and False omits draws.
        Other draw helpers follow their local boolean/count contract. Default: ``'auto'``.
        Passed to ``genetic_varcov_PSD`` as ``bootstrap_factor_draws``.

    bootstrap_factors : mapping or None
        Optional map from G/E or named component labels to post-fit factor decomposition
        settings. Each setting can specify rank, method, standardize and preserve_diag; None
        skips these additional aligned bootstrap factor targets. Default: ``None``.

    n_jobs : int or str
        Worker-process count for bootstrap; a positive integer fixes concurrency, while 'auto'
        resolves a bounded process count from CPU availability and the replicate workload.
        Default: ``'auto'``. Passed to ``genetic_varcov_PSD`` as ``bootstrap_n_jobs``.

    **fit_kwargs : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Passed to ``genetic_varcov_PSD`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``genetic_varcov_PSD(grm, traits, n_boot=int(n_boot), random_state=random_state, bootstrap_strategy='weighted', bootstrap_groups=groups, bootstrap_unit=bootstrap_unit, bootstrap_n_jobs=n_jobs, bootst...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .multivariate import genetic_varcov_PSD
    return genetic_varcov_PSD(
        grm, traits, n_boot=int(n_boot), random_state=random_state,
        bootstrap_strategy='weighted', bootstrap_groups=groups, bootstrap_unit=bootstrap_unit,
        bootstrap_n_jobs=n_jobs, bootstrap_factor_draws=retain_factor_draws,
        bootstrap_factors=bootstrap_factors, **fit_kwargs
    )

__all__=['GeneticVarCovBootstrapCache','bootstrap_genetic_varcov_psd','fit_psd_from_moments']
