"""Mechanistic genotype-to-phenotype simulation for VIGOR.

The simulator deliberately does *not* draw phenotypes from a pre-specified
multivariate ``G`` or ``E`` covariance matrix.  Instead it generates (or accepts)
individual-level genotypes and constructs phenotypes from explicit causal
variants, dominance deviations, epistatic pairs, gene-by-environment effects,
latent environmental covariates, penetrance modulation and residual shocks.
Trait covariance matrices are calculated *after* simulation from the realized
components.  This makes the simulator suitable for benchmarking covariance
estimators without giving a low-rank/factor model the same structure as the
simulation truth.

Coding follows the conventions used by VIGOR's PLINK/GCTA-style GRM code:

``A_ij = g_ij - 2 p_j``

and Vitezica dominance coding

``D_ij = -2 p_j^2`` for ``g=0``, ``2 p_j q_j`` for ``g=1`` and
``-2 q_j^2`` for ``g=2``.

The high-level :func:`simulate_genomic_dataset` function is the recommended
entry point.  :func:`simulate_traits_from_genotypes` can instead be used with a
real or externally simulated genotype matrix, and :func:`simulate_from_plink1`
uses a PLINK1 BED/BIM/FAM source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np
import pandas as pd

from .results import maybe_save, result_dataset, table_vars, to_dataarray


_EPS = 1e-12
_VARIANT_INFO_NAMES = {"chrom": "chromosome", "pos": "base-pair position",
                       "af": "alternate allele frequency", "maf": "minor allele frequency"}


def _rng(seed=None):
    """Return the rng result computed by ``np.random.default_rng``.

    Parameters
    ----------
    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``None``.
    """
    return np.random.default_rng(seed)


def _as_frame(x, index=None, columns=None, prefix="trait"):
    """Convert inputs to frame.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x.copy()``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    index : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    columns : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants. Default: ``'trait'``.
    """
    if hasattr(x, "to_pandas") and hasattr(x, "dims"):
        # Labeled DataArray (e.g. ``simulate_genotypes(...)['genotypes']``).
        x = x.to_pandas().copy()
        x.index.name = x.columns.name = None
        return x
    if isinstance(x, pd.DataFrame):
        return x.copy()
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError("expected a 2D matrix")
    if index is None:
        index = [f"ID{i:05d}" for i in range(a.shape[0])]
    if columns is None:
        columns = [f"{prefix}{j+1}" for j in range(a.shape[1])]
    return pd.DataFrame(a, index=index, columns=columns)


def _cov_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Coerce a trait covariance to a labeled frame in the requested trait order.

    Parameters
    ----------
    x : pd.DataFrame
        Numerical operand in ``pd.DataFrame(c, index=x.columns, columns=x.columns)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.
    """
    a = np.asarray(x, dtype=float)
    c = np.cov(a, rowvar=False, ddof=1)
    if np.ndim(c) == 0:
        c = np.asarray([[float(c)]])
    return pd.DataFrame(c, index=x.columns, columns=x.columns)


def _safe_standardize_cols(x, center=True):
    """Center and scale numeric columns, protecting constant/nonfinite scales.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``np.asarray(x, dtype=float)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``.
    """
    a = np.asarray(x, dtype=float)
    if center:
        a = a - np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > _EPS), sd, 1.0)
    return np.nan_to_num(a / sd[None, :], nan=0.0), sd


def _sigmoid(x):
    """Evaluate the logistic transform for variance-coordinate derivatives.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``1.0 / (1.0 + np.exp(-x))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.
    """
    x = np.clip(np.asarray(x, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    """Return the logit result computed by ``np.log``.

    Parameters
    ----------
    p : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _effect_draw(rng, shape, distribution="student_t", df=3.0):
    """Return the effect draw result computed by ``np.asarray``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.

    shape : tuple of int or None
        Requested array/distribution extents in axis order. Nonnegative sizes must agree with
        the associated data; None uses the backend's inferred scalar/default shape. Passed to
        ``rng.normal`` as ``size``.

    distribution : str
        Effect/noise distribution: 'normal'/'gaussian' gives normal draws,
        'laplace'/'double_exponential' heavier-tailed Laplace draws, and 'student'/'t' Student-t
        draws with the configured degrees of freedom where supported. Default: ``'student_t'``.

    df : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Default: ``3.0``. Source comparisons: ``df > 2``. See the
        rejection/fallback branches for enforcement.
    """
    distribution = str(distribution).lower()
    if distribution in {"normal", "gaussian"}:
        z = rng.normal(size=shape)
    elif distribution in {"laplace", "double_exponential"}:
        z = rng.laplace(size=shape) / np.sqrt(2.0)
    elif distribution in {"student", "student_t", "t"}:
        df = float(df)
        z = rng.standard_t(df, size=shape)
        if df > 2:
            z = z / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError("effect_distribution must be 'normal', 'laplace', or 'student_t'")
    return np.asarray(z, dtype=float)


def _allocate_counts(total, n_groups):
    """Allocate a total simulation effect count across the requested component proportions.

    Parameters
    ----------
    total : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported.

    n_groups : object (see the cited input contract)
        Input for the operation ``divmod(int(total), int(n_groups))``. See
        ``vigor.simulate._allocate_counts`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.
    """
    q, r = divmod(int(total), int(n_groups))
    out = np.full(int(n_groups), q, dtype=int)
    out[:r] += 1
    return out


def _variant_names(chrom, pos):
    """Construct ordered variant identifiers for a simulated genomic layout.

    Parameters
    ----------
    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.
    """
    return [f"{c}:{int(p)}" for c, p in zip(chrom, pos)]


def _sample_founder_alleles(rng, n_founders, n_variants, beta=(0.8, 0.8), min_mac=1):
    """Sample fixed founder alleles while avoiding monomorphic founder sites.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.

    n_founders : object (see the cited input contract)
        Input for the operation ``int(n_founders) - f.sum(axis=0)``. See
        ``vigor.simulate._sample_founder_alleles`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Passed to ``rng.beta`` as ``size``.

    beta : array-like
        Effect or path coefficients, ordered by the model's predictor/trait axes and on the
        scale used by its fit. Default: ``(0.8, 0.8)``.

    min_mac : int
        Input supplied to ``rng.choice`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``1``. Passed to ``rng.choice`` as ``size``.
    """
    a, b = map(float, beta)
    p = np.clip(rng.beta(a, b, size=n_variants), 1e-4, 1.0 - 1e-4)
    f = (rng.random((int(n_founders), int(n_variants))) < p[None, :]).astype(np.int8)
    min_mac = max(1, int(min_mac))
    for _ in range(100):
        mac = np.minimum(f.sum(axis=0), int(n_founders) - f.sum(axis=0))
        bad = mac < min_mac
        if not np.any(bad):
            break
        f[:, bad] = (rng.random((int(n_founders), int(bad.sum()))) < p[bad][None, :]).astype(np.int8)
    mac = np.minimum(f.sum(axis=0), int(n_founders) - f.sum(axis=0))
    if np.any(mac < min_mac):
        # Deterministic fallback for the very rare stubborn sites.
        for j in np.flatnonzero(mac < min_mac):
            f[:, j] = 0
            alt = rng.choice(int(n_founders), size=min_mac, replace=False)
            f[alt, j] = 1
    return f


def simulate_genotypes(
    n_samples: int = 1000,
    n_variants: int = 5000,
    *,
    model: str = "hs_founder_mosaic",
    n_chromosomes: int = 20,
    n_founders: int = 8,
    chromosome_length_mb: float = 100.0,
    recombination_rate_cM_per_mb: float = 0.5,
    generations: float = 100.0,
    founder_af_beta: tuple[float, float] = (0.8, 0.8),
    min_founder_mac: int = 1,
    missing_rate: float = 0.0,
    seed: int | None = 0,
    return_ancestry: bool = False,
    save_path=None,
):
    """Simulate diploid genotypes without imposing a trait covariance model.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``1000``. Source comparisons: ``n <
        2``. See the rejection/fallback branches for enforcement.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``5000``. Source comparisons: ``m < 2``. See the rejection/fallback
        branches for enforcement.

    model : str
        {'hs_founder_mosaic', 'hwe_independent'} ``hs_founder_mosaic`` generates two recombinant
        founder mosaics per animal from fixed inbred-founder haplotypes.  Local founder ancestry
        switches according to physical distance, recombination rate and the requested number of
        generations.  This creates LD/local ancestry from the finite founder panel rather than
        from a low-rank trait model. ``hwe_independent`` is a simpler Hardy-Weinberg control
        simulation. Default: ``'hs_founder_mosaic'``. Source-recognized literal values:
        ``founder_mosaic``, ``hs``, ``hwe``, ``independent``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    n_chromosomes : int
        Input for the operation ``int(n_chromosomes)``. See
        ``vigor.simulate.simulate_genotypes`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``20``. Source
        comparisons: ``C < 1``. See the rejection/fallback branches for enforcement.

    n_founders : int
        Input for the operation ``int(n_founders) - 1``. See
        ``vigor.simulate.simulate_genotypes`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``8``.

    chromosome_length_mb : float
        Physical span in megabases controlling the region, exclusion or sliding-window step
        named by this argument. Positive values determine extent; convert consistently with the
        variant base-pair coordinates. Default: ``100.0``.

    recombination_rate_cM_per_mb : float
        Physical-to-genetic map rate in centimorgans per megabase, used when approximating
        linkage/recombination from physical positions; positive values specify a nondegenerate
        map. Default: ``0.5``.

    generations : float
        Number of generations over which recombination attenuates linkage; nonnegative values
        scale the map-derived probability. Default: ``100.0``.

    founder_af_beta : tuple[float, float]
        Input for the operation ``_sample_founder_alleles(rng, n_founders, m, founder_af_beta,
        min_founder_mac)``. See ``vigor.simulate.simulate_genotypes`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``(0.8, 0.8)``.

    min_founder_mac : int
        Input for the operation ``_sample_founder_alleles(rng, n_founders, m, founder_af_beta,
        min_founder_mac)``. See ``vigor.simulate.simulate_genotypes`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``1``.

    missing_rate : float
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``0.0``. Source comparisons: ``float(missing_rate) > 0``.
        See the rejection/fallback branches for enforcement.

    seed : int | None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    return_ancestry : bool
        bool When True, include the two local founder-state matrices.  This can be large and is
        therefore off by default. Default: ``False``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables ``genotypes`` (sample, variant), ``founders`` (founder, variant; founder
        mosaic model only) and, with ``return_ancestry``, ``ancestry_hap0``/``ancestry_hap1``
        (sample, variant). The ``variant_info`` table is stored column-wise
        (``vigor.results.table_vars``) as ``chrom``, ``pos``, ``af`` and ``maf`` along
        ``variant``; ``vigor.results.result_table(ds, 'variant_info')`` rebuilds it. ``attrs``
        hold ``model`` and ``metadata``.
    """
    rng = _rng(seed)
    n = int(n_samples); m = int(n_variants); C = int(n_chromosomes)
    if n < 2 or m < 2 or C < 1:
        raise ValueError("n_samples>=2, n_variants>=2 and n_chromosomes>=1 are required")
    per = _allocate_counts(m, C)
    chrom = np.repeat(np.arange(1, C + 1).astype(str), per)
    positions = np.concatenate([
        np.linspace(1, max(2, int(chromosome_length_mb * 1e6)), k, dtype=np.int64)
        for k in per
    ])
    names = _variant_names(chrom, positions)
    model = str(model).lower()

    founders = None; ancestry0 = ancestry1 = None
    if model in {"hs", "hs_founder_mosaic", "founder_mosaic"}:
        founders = _sample_founder_alleles(rng, n_founders, m, founder_af_beta, min_founder_mac)
        hap = np.empty((2 * n, m), dtype=np.int8)
        ancestry = np.empty((2 * n, m), dtype=np.int16) if return_ancestry else None
        offset = 0
        for k in per:
            sl = slice(offset, offset + k)
            pos_mb = positions[sl].astype(float) / 1e6
            # Adjacent recombination probability in one meiosis; after G
            # generations the ancestry-break probability is 1-exp(-G*r).
            d_morgan = np.diff(pos_mb) * float(recombination_rate_cM_per_mb) / 100.0
            p_switch = 1.0 - np.exp(-float(generations) * np.maximum(d_morgan, 0.0))
            state = rng.integers(0, int(n_founders), size=2 * n)
            hap[:, offset] = founders[state, offset]
            if ancestry is not None:
                ancestry[:, offset] = state
            for local in range(1, k):
                sw = rng.random(2 * n) < p_switch[local - 1]
                if np.any(sw):
                    old = state[sw]
                    new = rng.integers(0, int(n_founders) - 1, size=int(sw.sum()))
                    new = new + (new >= old)
                    state[sw] = new
                j = offset + local
                hap[:, j] = founders[state, j]
                if ancestry is not None:
                    ancestry[:, j] = state
            offset += k
        G = hap[0::2].astype(np.float32) + hap[1::2].astype(np.float32)
        if ancestry is not None:
            ancestry0, ancestry1 = ancestry[0::2], ancestry[1::2]
    elif model in {"hwe", "hwe_independent", "independent"}:
        p = np.clip(rng.beta(*map(float, founder_af_beta), size=m), 0.01, 0.99)
        G = rng.binomial(2, p[None, :], size=(n, m)).astype(np.float32)
    else:
        raise ValueError("model must be 'hs_founder_mosaic' or 'hwe_independent'")

    if float(missing_rate) > 0:
        miss = rng.random(G.shape) < float(missing_rate)
        G = G.astype(np.float32, copy=True)
        G[miss] = np.nan

    ids = [f"ID{i:05d}" for i in range(n)]
    geno = pd.DataFrame(G, index=ids, columns=names)
    p_real = np.nanmean(G, axis=0) / 2.0
    maf = np.minimum(p_real, 1.0 - p_real)
    vi = pd.DataFrame({
        "snp": names, "chrom": chrom, "pos": positions,
        "af": p_real, "maf": maf,
    }).set_index("snp")
    out = {
        "genotypes": geno,
        "variant_info": vi,
        "model": model,
        "metadata": {
            "n_samples": n, "n_variants": m, "n_chromosomes": C,
            "n_founders": int(n_founders) if founders is not None else None,
            "generations": float(generations) if founders is not None else None,
            "recombination_rate_cM_per_mb": float(recombination_rate_cM_per_mb),
            "seed": seed,
        },
    }
    if founders is not None:
        out["founders"] = pd.DataFrame(founders, index=[f"F{i+1}" for i in range(int(n_founders))], columns=names)
    if return_ancestry:
        out["ancestry_hap0"] = pd.DataFrame(ancestry0, index=ids, columns=names)
        out["ancestry_hap1"] = pd.DataFrame(ancestry1, index=ids, columns=names)
    out.update(table_vars(out.pop("variant_info"), "variant_info", "variant", long_names=_VARIANT_INFO_NAMES))
    sv = ("sample", "variant")
    ds = result_dataset(
        out, dims={"genotypes": sv, "founders": ("founder", "variant"),
                   "ancestry_hap0": sv, "ancestry_hap1": sv},
        var_attrs={
            "genotypes": {"long_name": "genotype dosage", "units": "alternate-allele count",
                          "description": "0/1/2 dosage; NaN marks simulated missing calls"},
            "founders": {"long_name": "inbred founder alleles", "units": "allele indicator"},
            "ancestry_hap0": {"long_name": "local founder ancestry, haplotype 0", "units": "founder index"},
            "ancestry_hap1": {"long_name": "local founder ancestry, haplotype 1", "units": "founder index"},
        },
        title="Simulated genotypes", software="vigor.simulate.simulate_genotypes",
        parameters={"n_samples": n, "n_variants": m, "model": model, "n_chromosomes": C,
                    "n_founders": int(n_founders), "chromosome_length_mb": float(chromosome_length_mb),
                    "recombination_rate_cM_per_mb": float(recombination_rate_cM_per_mb),
                    "generations": float(generations), "min_founder_mac": int(min_founder_mac),
                    "missing_rate": float(missing_rate), "seed": seed,
                    "return_ancestry": bool(return_ancestry)})
    return maybe_save(ds, save_path)


def gcta_codings(genotypes, allele_frequency=None, *, min_maf=1e-6):
    """Return PLINK/GCTA-style additive and Vitezica dominance codings.

    Missing genotypes are mean-coded as zero after centering, exactly as in the
    GRM accumulation routines.  The returned coding matrices are *not*
    variance-standardized per SNP; effect sizes therefore remain interpretable
    on the PLINK dosage scale.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    allele_frequency : array-like or numeric scalar
        Input for the operation ``np.asarray(allele_frequency, dtype=float)``. See
        ``vigor.simulate.gcta_codings`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``1e-06``.

    Returns
    -------
    object
        Computed result from ``(A, D, p)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    G = np.asarray(genotypes, dtype=float)
    if G.ndim != 2:
        raise ValueError("genotypes must be a 2D matrix")
    if allele_frequency is None:
        cnt = np.isfinite(G).sum(axis=0)
        p = np.divide(np.nansum(G, axis=0), 2.0 * cnt,
                      out=np.full(G.shape[1], np.nan), where=cnt > 0)
    else:
        p = np.asarray(allele_frequency, dtype=float)
        if p.shape != (G.shape[1],):
            raise ValueError("allele_frequency length must equal number of variants")
    p = np.clip(p, float(min_maf), 1.0 - float(min_maf)); q = 1.0 - p
    valid = np.isfinite(G)
    A = G - 2.0 * p[None, :]
    A[~valid] = 0.0
    D = np.where(
        G == 0, -2.0 * p[None, :] ** 2,
        np.where(G == 1, 2.0 * p[None, :] * q[None, :],
                 np.where(G == 2, -2.0 * q[None, :] ** 2, 0.0)),
    )
    D[~valid] = 0.0
    return A, D, p


def _causal_effect_matrix(rng, m, T, n_causal, p, *, trait_hit_rate=0.25,
                          effect_distribution="student_t", effect_df=3.0,
                          maf_effect_alpha=-0.25, candidate=None):
    """Draw and scale simulated causal variant effects for the selected trait components.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.

    m : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    T : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension.

    n_causal : object (see the cited input contract)
        Input supplied to ``rng.choice`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Passed to ``rng.choice`` as ``size``.

    p : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations.

    trait_hit_rate : float or pair of float
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``0.25``.

    effect_distribution : str
        Effect/noise distribution: 'normal'/'gaussian' gives normal draws,
        'laplace'/'double_exponential' heavier-tailed Laplace draws, and 'student'/'t' Student-t
        draws with the configured degrees of freedom where supported. Default: ``'student_t'``.

    effect_df : float
        Input for the operation ``_effect_draw(rng, (n_causal, T), effect_distribution,
        effect_df)``. See ``vigor.simulate._causal_effect_matrix`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``3.0``.

    maf_effect_alpha : float
        Input for the operation ``maf_var ** float(maf_effect_alpha)``. See
        ``vigor.simulate._causal_effect_matrix`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``-0.25``.

    candidate : object (see the cited input contract)
        Input for the operation ``min(max(0, int(n_causal)), len(candidate))``. See
        ``vigor.simulate._causal_effect_matrix`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.
    """
    candidate = np.arange(m) if candidate is None else np.asarray(candidate, dtype=int)
    n_causal = min(max(0, int(n_causal)), len(candidate))
    idx = rng.choice(candidate, size=n_causal, replace=False) if n_causal else np.empty(0, dtype=int)
    B = np.zeros((m, T), dtype=float)
    if n_causal:
        mask = rng.random((n_causal, T)) < float(trait_hit_rate)
        # Every selected variant and every trait should have a chance to matter.
        empty = np.flatnonzero(~mask.any(axis=1))
        if len(empty):
            mask[empty, rng.integers(0, T, size=len(empty))] = True
        for t in range(T):
            if not np.any(mask[:, t]):
                mask[rng.integers(0, n_causal), t] = True
        z = _effect_draw(rng, (n_causal, T), effect_distribution, effect_df)
        maf_var = np.maximum(2.0 * p[idx] * (1.0 - p[idx]), 1e-5)
        scale = maf_var ** float(maf_effect_alpha)
        B[idx] = z * scale[:, None] * mask
    return B, idx


def _penetrance_gate(rng, n, n_terms, H, base_range=(0.65, 1.0), hidden_strength=0.8):
    """Generate locus-level incomplete penetrance modulated by hidden covariates.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.

    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    n_terms : object (see the cited input contract)
        Input supplied to ``rng.uniform`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Source comparisons: ``n_terms <= 0``. See the rejection/fallback branches for
        enforcement. Passed to ``rng.uniform`` as ``size``.

    H : array-like or numeric scalar
        Input supplied to ``rng.normal`` as ``scale``. See that callable's documented ``scale``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Passed to ``rng.normal`` as ``scale``.

    base_range : tuple
        Input for the operation ``map(float, base_range)``. See
        ``vigor.simulate._penetrance_gate`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``(0.65, 1.0)``.

    hidden_strength : array-like or numeric scalar
        Input supplied to ``rng.normal`` as ``scale``. See that callable's documented ``scale``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``0.8``. Passed to ``rng.normal`` as ``scale``.
    """
    if n_terms <= 0:
        return np.zeros((n, 0)), np.zeros(0), np.zeros((0, H.shape[1]))
    lo, hi = map(float, base_range)
    if not (0 < lo <= hi <= 1):
        raise ValueError("penetrance_range must satisfy 0 < low <= high <= 1")
    base = rng.uniform(lo, hi, size=n_terms)
    gamma = rng.normal(scale=float(hidden_strength) / np.sqrt(max(H.shape[1], 1)),
                       size=(n_terms, H.shape[1])) if H.shape[1] else np.zeros((n_terms, 0))
    logits = _logit(base)[None, :] + (H @ gamma.T if H.shape[1] else 0.0)
    prob = _sigmoid(logits)
    gate = (rng.random((n, n_terms)) < prob).astype(float)
    return gate, base, gamma


def _rescale_component_matrix(components, weights, eps=1e-12):
    """Scale component columns to requested marginal variance shares.

    No cross-trait covariance target is imposed.  Scaling is per trait and per
    named component only; all between-trait covariance therefore arises from
    shared causal variants, LD, shared environments and interactions.

    Parameters
    ----------
    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names.

    weights : array-like or int
        Weights or multiplicities used in this accumulation/resampling step. Align vector
        entries with the indexed sample/variant axis; scalar counts specify the number of
        consecutive items.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-12``. Passed to ``np.divide`` as ``where``.
    """
    keys = list(components)
    if not keys:
        return components, {}
    T = np.asarray(next(iter(components.values()))).shape[1]
    w = {k: float(weights.get(k, 0.0)) for k in keys}
    s = sum(max(v, 0.0) for v in w.values())
    if s <= 0:
        return {k: np.zeros_like(v, dtype=float) for k, v in components.items()}, {k: np.zeros(T) for k in keys}
    w = {k: max(v, 0.0) / s for k, v in w.items()}
    out = {}; scales = {}
    for k, x in components.items():
        a = np.asarray(x, dtype=float)
        sd = np.std(a, axis=0, ddof=1)
        target_sd = np.sqrt(w[k]) * np.ones(T)
        sc = np.divide(target_sd, sd, out=np.zeros(T), where=sd > eps)
        out[k] = a * sc[None, :]
        scales[k] = sc
    return out, scales


def _effect_table(names, traits, BA, BD, pbase_A=None, pbase_D=None):
    """Return the effect table result computed by ``pd.DataFrame``.

    Parameters
    ----------
    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    BA : object (see the cited input contract)
        Input for the operation ``BA[j, t]``. See ``vigor.simulate._effect_table`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Source comparisons: ``BA != 0``; ``a != 0``; ``a == 0``; ``a == 0.0``. See
        the rejection/fallback branches for enforcement.

    BD : object (see the cited input contract)
        Input for the operation ``BD[j, t]``. See ``vigor.simulate._effect_table`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Source comparisons: ``d == 0.0``. See the rejection/fallback branches for
        enforcement.

    pbase_A : array-like or numeric scalar
        Input for the operation ``pbase_A[np.searchsorted(np.flatnonzero(np.any(BA != 0,
        axis=1)), j)]``. See ``vigor.simulate._effect_table`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    pbase_D : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.
    """
    rows = []
    for j, name in enumerate(names):
        for t, trait in enumerate(traits):
            a = float(BA[j, t]); d = float(BD[j, t])
            if a == 0.0 and d == 0.0:
                continue
            rows.append({
                "snp": name, "variant_index": j, "trait": trait,
                "additive_effect_raw": a, "dominance_effect_raw": d,
                "dominance_ratio_raw": (d / abs(a)) if a != 0 else np.nan,
                "additive_penetrance": np.nan if pbase_A is None or a == 0 else float(pbase_A[np.searchsorted(np.flatnonzero(np.any(BA != 0, axis=1)), j)]) if j in set(np.flatnonzero(np.any(BA != 0, axis=1))) else np.nan,
                "dominance_penetrance": np.nan,
            })
    return pd.DataFrame(rows)


@dataclass
class GenomicSimulationResult:
    """Container returned by :func:`simulate_traits_from_genotypes`.

    :func:`simulate_genomic_dataset` and :func:`simulate_from_plink1` return
    :meth:`to_dataset` of this container (an ``xarray.Dataset``).

    Constructor fields
    ------------------
    genotypes : pd.DataFrame
        Stored genotypes consumed by ``write_plink``; see those methods for the numerical/label
        contract. Required construction field.

    phenotypes : pd.DataFrame
        Stored phenotypes supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    components : dict[str, pd.DataFrame]
        Stored components supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    trait_covariances : dict[str, pd.DataFrame]
        Stored trait covariances supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    variant_info : pd.DataFrame
        Stored variant info consumed by ``write_plink``; see those methods for the
        numerical/label contract. Required construction field.

    variant_effects : pd.DataFrame
        Stored variant effects supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    epistasis_effects : pd.DataFrame
        Stored epistasis effects supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    environment : pd.DataFrame
        Stored environment supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    grms : dict[str, pd.DataFrame]
        Stored grms supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Default: a fresh value from ``dict()`` for each
        instance.

    metadata : dict[str, Any]
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.

    founders : pd.DataFrame | None
        Stored founders supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``None``.

    truth : dict[str, Any]
        Stored truth supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: a fresh value from ``dict()`` for each
        instance.
    """
    genotypes: pd.DataFrame
    phenotypes: pd.DataFrame
    components: dict[str, pd.DataFrame]
    trait_covariances: dict[str, pd.DataFrame]
    variant_info: pd.DataFrame
    variant_effects: pd.DataFrame
    epistasis_effects: pd.DataFrame
    environment: pd.DataFrame
    grms: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    founders: pd.DataFrame | None = None
    truth: dict[str, Any] = field(default_factory=dict)

    def write_plink(self, prefix):
        """Write the simulated genotypes as a PLINK1 BED/BIM/FAM fileset.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        Returns
        -------
        object
            Computed result from ``p``; see the summary and the called implementation for its labeled/numerical representation.
        """
        from .testutils import write_plink
        p = str(prefix)
        vi = self.variant_info.reindex(self.genotypes.columns)
        write_plink(p, self.genotypes.to_numpy(float), list(map(str, self.genotypes.index)),
                    vi["chrom"].astype(str).to_numpy(), vi["pos"].to_numpy(int))
        return p

    def to_dataset(self, *, software=None, parameters=None):
        """Return the simulation as one labeled ``xarray.Dataset`` keeping the field names.

        Parameters
        ----------
        software : str or None
            Producing VIGOR function recorded in ``attrs['software']``; None records
            ``vigor.simulate.simulate_traits_from_genotypes``. Default: ``None``.

        parameters : mapping or None
            Scalar call parameters stored in ``attrs['parameters']``; None records no call
            parameters. Default: ``None``.

        Returns
        -------
        xarray.Dataset
            Data variables ``genotypes`` (sample, variant), ``phenotypes`` (sample, trait),
            ``components`` (component, sample, trait), ``trait_covariances`` (covariance,
            trait_i, trait_j), ``environment`` (sample, environment_covariate) and, when present,
            ``grms`` (kernel, sample_i, sample_j) and ``founders`` (founder, variant). Tables
            are stored column-wise (``vigor.results.table_vars``): ``variant_info`` as
            ``chrom``/``pos``/``af``/``maf`` along ``variant``, ``variant_effects`` as
            ``variant_effects_<column>`` along ``variant_effect`` and ``epistasis_effects``
            as ``epistasis_effects_<column>`` along ``epistasis_pair``; rebuild any of them
            with ``vigor.results.result_table(ds, '<table>')``. ``attrs`` hold ``metadata``
            and ``truth``.
        """
        import xarray as xr

        def stacked(frames, dim, dims):
            """Stack equally labeled matrices along a new named leading dimension.

            Parameters
            ----------
            frames : mapping of str to pandas.DataFrame
                Named matrices sharing row and column labels, stacked in mapping order; an
                empty mapping yields None so the variable is omitted.
            dim : str
                Name of the new leading dimension whose coordinate holds the mapping keys.
            dims : tuple of str
                Dimension names given to each matrix's rows and columns.
            """
            if not frames:
                return None
            return xr.concat([to_dataarray(v, dims=dims) for v in frames.values()],
                             dim=dim).assign_coords({dim: list(frames)})

        vi = self.variant_info
        if not vi.index.equals(self.genotypes.columns):
            vi = vi.reindex(self.genotypes.columns)
        values = {
            "genotypes": self.genotypes, "phenotypes": self.phenotypes,
            "components": stacked(self.components, "component", ("sample", "trait")),
            "trait_covariances": stacked(self.trait_covariances, "covariance", ("trait_i", "trait_j")),
            "environment": self.environment,
            "grms": stacked(self.grms, "kernel", ("sample_i", "sample_j")),
            "founders": self.founders,
            **table_vars(vi, "variant_info", "variant", long_names=_VARIANT_INFO_NAMES),
            **table_vars(self.variant_effects, "variant_effects", "variant_effect",
                         prefix="variant_effects_"),
            **table_vars(self.epistasis_effects, "epistasis_effects", "epistasis_pair",
                         prefix="epistasis_effects_"),
        }
        pu = "phenotype units"
        return result_dataset(
            values,
            dims={"genotypes": ("sample", "variant"), "phenotypes": ("sample", "trait"),
                  "environment": ("sample", "environment_covariate"), "founders": ("founder", "variant")},
            var_attrs={
                "genotypes": {"long_name": "genotype dosage", "units": "alternate-allele count",
                              "description": "0/1/2 dosage; NaN marks missing calls"},
                "phenotypes": {"long_name": "simulated phenotype", "units": pu,
                               "description": "sum of realized genetic and environmental components"},
                "components": {"long_name": "realized mechanistic trait components", "units": pu},
                "trait_covariances": {"long_name": "empirical trait covariance of realized components",
                                      "units": pu + " squared"},
                "environment": {"long_name": "hidden and observed environmental covariates",
                                "units": "standardized units"},
                "grms": {"long_name": "genomic relationship matrices",
                         "description": "PLINK/GCTA additive and Vitezica dominance GRMs"},
                "founders": {"long_name": "inbred founder alleles", "units": "allele indicator"},
            },
            attrs={"metadata": self.metadata, "truth": self.truth},
            title="Mechanistic genomic simulation",
            software=software or "vigor.simulate.simulate_traits_from_genotypes",
            parameters=parameters)


def simulate_traits_from_genotypes(
    genotypes,
    *,
    n_traits: int = 12,
    trait_names: Sequence[str] | None = None,
    n_additive: int = 120,
    n_dominance: int = 60,
    dominance_overlap: float = 0.6,
    n_epistatic_pairs: int = 40,
    epistasis_mix: Mapping[str, float] | None = None,
    n_gxe: int = 40,
    n_hidden_covariates: int = 6,
    n_observed_environments: int = 1,
    heritability: float | Sequence[float] = 0.45,
    genetic_weights: Mapping[str, float] | None = None,
    environmental_weights: Mapping[str, float] | None = None,
    trait_hit_rate: float = 0.30,
    effect_distribution: str = "student_t",
    effect_df: float = 3.0,
    maf_effect_alpha: float = -0.25,
    penetrance_range: tuple[float, float] = (0.65, 1.0),
    penetrance_hidden_strength: float = 0.8,
    gxe_hidden_fraction: float = 0.6,
    hidden_nonlinear_fraction: float = 0.25,
    residual_distribution: str = "normal",
    residual_df: float = 5.0,
    orthogonalize_epistasis: bool = False,
    seed: int | None = 0,
    compute_grms: bool = True,
) -> GenomicSimulationResult:
    """Simulate quantitative traits from explicit causal genotype architecture.

    The simulation never samples from a target trait covariance matrix.  It
    instead creates the following individual-level truth components:

    ``A``
        additive main effects from causal variants;
    ``D``
        Vitezica-coded dominance effects, with configurable overlap with the
        additive causal set and a broad distribution of dominance ratios;
    ``AA``, ``AD``, ``DD``
        pairwise epistasis;
    ``GxE``
        genotype-by-observed/hidden-environment modulation, including the
        deviation caused by incomplete penetrance;
    ``E_hidden``
        main effects of latent continuous/nonlinear environmental covariates;
    ``E_noise``
        trait-specific independent Gaussian or Student-t shocks.

    ``G_total`` and ``E_total`` covariance matrices are empirical covariances
    of the *realized* total genetic and residual values.  This automatically
    includes LD-induced cross-component covariance; it is deliberately not the
    sum of pre-specified low-rank matrices.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension. Default: ``12``. Source comparisons: ``T < 1``. See the
        rejection/fallback branches for enforcement.

    trait_names : Sequence[str] | None
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    n_additive : int
        Input for the operation ``_causal_effect_matrix(rng, m, T, n_additive, p,
        trait_hit_rate=trait_hit_rate, effect_distribution=effect_distribution,
        effect_df=effect_df, maf_effect_alpha=maf_effect_alpha, candidate=candidate)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``120``.

    n_dominance : int
        Input for the operation ``min(max(0, int(n_dominance)), len(candidate))``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``60``.

    dominance_overlap : float
        Input for the operation ``nd * float(dominance_overlap)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.6``.

    n_epistatic_pairs : int
        Input for the operation ``range(int(n_epistatic_pairs))``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``40``.

    epistasis_mix : Mapping[str, float] | None
        Input for the operation ``dict(epistasis_mix)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    n_gxe : int
        Input supplied to ``rng.choice`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``40``. Source comparisons: ``int(n_gxe) > 0``. See the rejection/fallback
        branches for enforcement. Passed to ``rng.choice`` as ``size``.

    n_hidden_covariates : int
        Input supplied to ``rng.normal`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``6``. Passed to ``rng.normal`` as ``size``.

    n_observed_environments : int
        Input supplied to ``rng.normal`` as ``size``. See that callable's documented ``size``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``1``. Passed to ``rng.normal`` as ``size``.

    heritability : float | Sequence[float]
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``0.45``.

    genetic_weights : Mapping[str, float] | None
        Input for the operation ``_rescale_component_matrix(source_total, genetic_weights)``.
        See ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    environmental_weights : Mapping[str, float] | None
        Input for the operation ``_rescale_component_matrix(raw_env, environmental_weights)``.
        See ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    trait_hit_rate : float
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``0.3``. Passed to ``_causal_effect_matrix`` as
        ``trait_hit_rate``.

    effect_distribution : str
        Effect/noise distribution: 'normal'/'gaussian' gives normal draws,
        'laplace'/'double_exponential' heavier-tailed Laplace draws, and 'student'/'t' Student-t
        draws with the configured degrees of freedom where supported. Default: ``'student_t'``.
        Passed to ``_causal_effect_matrix`` as ``effect_distribution``.

    effect_df : float
        Input supplied to ``_causal_effect_matrix`` as ``effect_df``. See that callable's
        documented ``effect_df`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``3.0``. Passed to ``_causal_effect_matrix`` as
        ``effect_df``.

    maf_effect_alpha : float
        Input supplied to ``_causal_effect_matrix`` as ``maf_effect_alpha``. See that callable's
        documented ``maf_effect_alpha`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``-0.25``. Passed to ``_causal_effect_matrix``
        as ``maf_effect_alpha``.

    penetrance_range : tuple[float, float]
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``(0.65, 1.0)``.

    penetrance_hidden_strength : float
        Input supplied to ``rng.normal`` as ``scale``. See that callable's documented ``scale``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``0.8``. Passed to ``rng.normal`` as ``scale``.

    gxe_hidden_fraction : float
        Input for the operation ``rng.random() < float(gxe_hidden_fraction)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.6``.

    hidden_nonlinear_fraction : float
        Input for the operation ``H.shape[1] * float(hidden_nonlinear_fraction)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.25``. Source comparisons: ``float(hidden_nonlinear_fraction) > 0``. See the
        rejection/fallback branches for enforcement.

    residual_distribution : str
        Input for the operation ``residual_distribution.lower() in {'normal', 'gaussian'}``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'normal'``. Source-recognized literal values: ``student``, ``student_t``, ``t``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    residual_df : float
        Input for the operation ``_effect_draw(rng, (n, T), 'student_t', residual_df)``. See
        ``vigor.simulate.simulate_traits_from_genotypes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``5.0``.

    orthogonalize_epistasis : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    seed : int | None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    compute_grms : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``GenomicSimulationResult(genotypes=Gdf, phenotypes=phenotypes, components=components, trait_covariances=covs, variant_info=vi, variant_effects=variant_effects, epistasis_effects=epistasis_effects, e...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rng = _rng(seed)
    Gdf = _as_frame(genotypes, prefix="v")
    X = Gdf.to_numpy(float); n, m = X.shape; T = int(n_traits)
    if T < 1:
        raise ValueError("n_traits must be >=1")
    traits = list(trait_names) if trait_names is not None else [f"trait{t+1}" for t in range(T)]
    if len(traits) != T:
        raise ValueError("trait_names length must equal n_traits")
    Acode, Dcode, p = gcta_codings(X)
    maf = np.minimum(p, 1.0 - p)
    candidate = np.flatnonzero(maf >= 0.01)
    if len(candidate) < 2:
        raise ValueError("not enough polymorphic variants for simulation")

    # Hidden and observed environments are generated as primitive causes; E is
    # derived from their realized trait effects rather than sampled as a matrix.
    H = rng.normal(size=(n, int(n_hidden_covariates))) if n_hidden_covariates else np.zeros((n, 0))
    if H.shape[1] and float(hidden_nonlinear_fraction) > 0:
        k = max(1, int(round(H.shape[1] * float(hidden_nonlinear_fraction))))
        H[:, :k] = (H[:, :k] ** 2 - 1.0) / np.sqrt(2.0)
    O = rng.normal(size=(n, int(n_observed_environments))) if n_observed_environments else np.zeros((n, 0))
    H, _ = _safe_standardize_cols(H) if H.shape[1] else (H, None)
    O, _ = _safe_standardize_cols(O) if O.shape[1] else (O, None)

    BA, ia = _causal_effect_matrix(rng, m, T, n_additive, p, trait_hit_rate=trait_hit_rate,
                                   effect_distribution=effect_distribution, effect_df=effect_df,
                                   maf_effect_alpha=maf_effect_alpha, candidate=candidate)
    # Dominance loci partly overlap additive loci.  On overlapping locus/trait
    # cells, a broad dominance ratio converts |a| to d.  Remaining cells get an
    # independent heavy-tailed dominance effect.
    nd = min(max(0, int(n_dominance)), len(candidate))
    nover = min(nd, int(round(nd * float(dominance_overlap))), len(ia))
    overlap = rng.choice(ia, size=nover, replace=False) if nover else np.empty(0, dtype=int)
    rem = np.setdiff1d(candidate, overlap)
    extra = rng.choice(rem, size=min(nd - nover, len(rem)), replace=False) if nd > nover else np.empty(0, dtype=int)
    idom = np.unique(np.concatenate([overlap, extra])).astype(int)
    BD = np.zeros((m, T), dtype=float)
    if len(idom):
        hit = rng.random((len(idom), T)) < float(trait_hit_rate)
        empty = np.flatnonzero(~hit.any(axis=1))
        if len(empty): hit[empty, rng.integers(0, T, size=len(empty))] = True
        raw = _effect_draw(rng, (len(idom), T), effect_distribution, effect_df)
        vscale = np.maximum(2*p[idom]*(1-p[idom]), 1e-5) ** float(maf_effect_alpha)
        BD[idom] = raw * vscale[:, None] * hit
        # Overlapping additive effects receive interpretable dominance ratios.
        ratio_levels = np.asarray([-1.5, -1.0, -0.5, 0.25, 0.5, 1.0, 1.5, 2.0])
        for j in overlap:
            tmask = BA[j] != 0
            if np.any(tmask):
                ratios = rng.choice(ratio_levels, size=int(tmask.sum()))
                signs = rng.choice([-1.0, 1.0], size=int(tmask.sum()))
                BD[j, tmask] = abs(BA[j, tmask]) * ratios * signs

    # Locus-level penetrance gates.  We decompose gate*effect as base-penetrance
    # main effect + (gate-base)*effect, placing the latter in GxE/penetrance.
    ia_active = np.flatnonzero(np.any(BA != 0, axis=1))
    id_active = np.flatnonzero(np.any(BD != 0, axis=1))
    gateA, pA, gammaA = _penetrance_gate(rng, n, len(ia_active), H, penetrance_range, penetrance_hidden_strength)
    gateD, pD, gammaD = _penetrance_gate(rng, n, len(id_active), H, penetrance_range, penetrance_hidden_strength)
    A_main = np.zeros((n, T)); D_main = np.zeros((n, T))
    A_pen = np.zeros((n, T)); D_pen = np.zeros((n, T))
    for k, j in enumerate(ia_active):
        locus = Acode[:, j:j+1] * BA[j:j+1, :]
        A_main += pA[k] * locus
        A_pen += (gateA[:, k:k+1] - pA[k]) * locus
    for k, j in enumerate(id_active):
        locus = Dcode[:, j:j+1] * BD[j:j+1, :]
        D_main += pD[k] * locus
        D_pen += (gateD[:, k:k+1] - pD[k]) * locus

    # Epistasis is generated from explicit variant pairs, not latent factors.
    mix = {"AA": 0.6, "AD": 0.2, "DD": 0.2} if epistasis_mix is None else dict(epistasis_mix)
    mk = ["AA", "AD", "DD"]; probs = np.asarray([max(0., float(mix.get(k, 0.))) for k in mk])
    probs = probs / probs.sum() if probs.sum() else np.asarray([1., 0., 0.])
    epi_components = {k: np.zeros((n, T), dtype=float) for k in mk}
    epi_penetrance = {k: np.zeros((n, T), dtype=float) for k in mk}
    epi_rows = []
    for pair_id in range(int(n_epistatic_pairs)):
        j, k = rng.choice(candidate, size=2, replace=False)
        kind = str(rng.choice(mk, p=probs))
        c1 = Acode[:, j] if kind[0] == "A" else Dcode[:, j]
        c2 = Acode[:, k] if kind[1] == "A" else Dcode[:, k]
        z = c1 * c2
        if orthogonalize_epistasis:
            Q = np.column_stack([np.ones(n), Acode[:, j], Acode[:, k], Dcode[:, j], Dcode[:, k]])
            z = z - Q @ np.linalg.lstsq(Q, z, rcond=None)[0]
        hit = rng.random(T) < float(trait_hit_rate)
        if not np.any(hit): hit[rng.integers(0, T)] = True
        beta = _effect_draw(rng, T, effect_distribution, effect_df) * hit
        base_pen = float(rng.uniform(*map(float, penetrance_range)))
        gamma = rng.normal(scale=float(penetrance_hidden_strength)/np.sqrt(max(H.shape[1], 1)), size=H.shape[1]) if H.shape[1] else np.zeros(0)
        prob = _sigmoid(_logit(base_pen) + (H @ gamma if H.shape[1] else 0.0))
        gate = (rng.random(n) < prob).astype(float)
        locus = z[:, None] * beta[None, :]
        epi_components[kind] += base_pen * locus
        epi_penetrance[kind] += (gate[:, None] - base_pen) * locus
        epi_rows.append({"pair_id": pair_id, "kind": kind,
                         "snp1": Gdf.columns[j], "snp2": Gdf.columns[k],
                         "variant1": int(j), "variant2": int(k),
                         "base_penetrance": base_pen,
                         **{f"effect_{traits[t]}": float(beta[t]) for t in range(T)}})

    # Explicit GxE effects.  A fraction uses hidden environments; the remainder
    # uses observed environments.  Effects are sparse across traits.
    GxE_explicit = np.zeros((n, T), dtype=float)
    GxE_explicit_hidden = np.zeros((n, T), dtype=float)
    GxE_explicit_observed = np.zeros((n, T), dtype=float)
    gxe_rows = []
    if int(n_gxe) > 0 and (H.shape[1] + O.shape[1]) > 0:
        gxe_idx = rng.choice(candidate, size=min(int(n_gxe), len(candidate)), replace=False)
        for j in gxe_idx:
            use_hidden = H.shape[1] > 0 and (O.shape[1] == 0 or rng.random() < float(gxe_hidden_fraction))
            Earr = H if use_hidden else O
            eidx = int(rng.integers(0, Earr.shape[1]))
            coding = "additive" if rng.random() < 0.75 else "dominance"
            code = Acode[:, j] if coding == "additive" else Dcode[:, j]
            hit = rng.random(T) < float(trait_hit_rate)
            if not np.any(hit): hit[rng.integers(0, T)] = True
            beta = _effect_draw(rng, T, effect_distribution, effect_df) * hit
            term = (code * Earr[:, eidx])[:, None] * beta[None, :]
            GxE_explicit += term
            if use_hidden:
                GxE_explicit_hidden += term
            else:
                GxE_explicit_observed += term
            gxe_rows.append({"snp": Gdf.columns[j], "variant_index": int(j),
                             "environment": f"hidden{eidx+1}" if use_hidden else f"observed{eidx+1}",
                             "hidden": bool(use_hidden), "coding": coding,
                             **{f"effect_{traits[t]}": float(beta[t]) for t in range(T)}})

    # Hidden environment main effects.  Independent coefficients across traits
    # ensure no factor covariance is specified as truth; shared covariates cause
    # whatever cross-trait residual covariance emerges.
    BH = _effect_draw(rng, (H.shape[1], T), effect_distribution, effect_df) if H.shape[1] else np.zeros((0, T))
    E_hidden = H @ BH if H.shape[1] else np.zeros((n, T))
    if residual_distribution.lower() in {"normal", "gaussian"}:
        E_noise = rng.normal(size=(n, T))
    elif residual_distribution.lower() in {"student", "student_t", "t"}:
        E_noise = _effect_draw(rng, (n, T), "student_t", residual_df)
    else:
        raise ValueError("residual_distribution must be 'normal' or 'student_t'")

    # Scale each biological mechanism together with its penetrance deviation.
    # This preserves the exact gate identity after calibration.  The requested
    # weight for A/D/AA/AD/DD refers to that complete source mechanism; the
    # explicit GxE term has its own weight.
    source_main = {"A": A_main, "D": D_main, **epi_components}
    source_pen = {"A": A_pen, "D": D_pen, **epi_penetrance}
    source_total = {k: source_main[k] + source_pen[k] for k in source_main}
    source_total["GxE"] = GxE_explicit
    if genetic_weights is None:
        genetic_weights = {"A": .58, "D": .10, "AA": .08, "AD": .04, "DD": .03, "GxE": .17}
    scaled_source, gscales = _rescale_component_matrix(source_total, genetic_weights)
    genetic = {}
    GxE_pen_scaled = np.zeros((n, T), dtype=float)
    for k in ["A", "D", "AA", "AD", "DD"]:
        sc = gscales[k]
        genetic[k] = source_main[k] * sc[None, :]
        GxE_pen_scaled += source_pen[k] * sc[None, :]
    gxe_pen_scaled = GxE_pen_scaled
    gxe_hidden_scaled = GxE_explicit_hidden * gscales["GxE"][None, :]
    gxe_observed_scaled = GxE_explicit_observed * gscales["GxE"][None, :]
    genetic["GxE"] = gxe_pen_scaled + gxe_hidden_scaled + gxe_observed_scaled
    raw_env = {"E_hidden": E_hidden, "E_noise": E_noise}
    if environmental_weights is None:
        environmental_weights = {"E_hidden": .35, "E_noise": .65}
    envc, escales = _rescale_component_matrix(raw_env, environmental_weights)

    # Set overall genetic/environmental marginal variance ratio per trait, while
    # preserving emergent cross-trait covariance and component correlations.
    Gtot0 = sum(genetic.values(), np.zeros((n, T)))
    Etot0 = sum(envc.values(), np.zeros((n, T)))
    h = np.asarray(heritability if np.ndim(heritability) else [heritability] * T, dtype=float)
    if h.shape != (T,):
        raise ValueError("heritability must be scalar or one value per trait")
    h = np.clip(h, 1e-4, 0.9999)
    vg = np.var(Gtot0, axis=0, ddof=1); ve = np.var(Etot0, axis=0, ddof=1)
    # Keep environment scale fixed; scale all genetic components equally within
    # trait to target Var(G)/(Var(G)+Var(E)).
    desired_vg = ve * h / (1.0 - h)
    overall_g_scale = np.sqrt(np.divide(desired_vg, vg, out=np.zeros(T), where=vg > _EPS))
    genetic = {k: v * overall_g_scale[None, :] for k, v in genetic.items()}
    gxe_pen_scaled = gxe_pen_scaled * overall_g_scale[None, :]
    gxe_hidden_scaled = gxe_hidden_scaled * overall_g_scale[None, :]
    gxe_observed_scaled = gxe_observed_scaled * overall_g_scale[None, :]
    Gtotal = sum(genetic.values(), np.zeros((n, T)))
    Etotal = sum(envc.values(), np.zeros((n, T)))
    Y = Gtotal + Etotal

    idx = Gdf.index
    components = {k: pd.DataFrame(v, index=idx, columns=traits) for k, v in {**genetic, **envc}.items()}
    components["GxE_penetrance"] = pd.DataFrame(gxe_pen_scaled, index=idx, columns=traits)
    components["GxE_hidden"] = pd.DataFrame(gxe_hidden_scaled, index=idx, columns=traits)
    components["GxE_observed"] = pd.DataFrame(gxe_observed_scaled, index=idx, columns=traits)
    components["G_total"] = pd.DataFrame(Gtotal, index=idx, columns=traits)
    components["E_total"] = pd.DataFrame(Etotal, index=idx, columns=traits)
    phenotypes = pd.DataFrame(Y, index=idx, columns=traits)

    covs = {k: _cov_frame(v) for k, v in components.items()}
    covs["P"] = _cov_frame(phenotypes)
    # Sum of component covariances is intentionally reported separately because
    # LD and shared environments can create nonzero cross-component covariance.
    covs["G_component_sum"] = sum((covs[k] for k in genetic), pd.DataFrame(0.0, index=traits, columns=traits))
    covs["E_component_sum"] = covs["E_hidden"] + covs["E_noise"]
    covs["GE_cross"] = covs["P"] - covs["G_total"] - covs["E_total"]

    # Variant effect table with explicit penetrance and dominance levels.
    rows = []
    posA = {j:k for k,j in enumerate(ia_active)}; posD = {j:k for k,j in enumerate(id_active)}
    for j in np.unique(np.concatenate([ia_active, id_active])):
        for t, tr in enumerate(traits):
            if BA[j, t] == 0 and BD[j, t] == 0:
                continue
            rows.append({
                "snp": Gdf.columns[j], "variant_index": int(j), "trait": tr,
                "af": float(p[j]), "maf": float(min(p[j], 1-p[j])),
                "additive_effect_raw": float(BA[j, t]),
                "dominance_effect_raw": float(BD[j, t]),
                "dominance_ratio_raw": float(BD[j,t]/abs(BA[j,t])) if BA[j,t] != 0 else np.nan,
                "additive_base_penetrance": float(pA[posA[j]]) if j in posA else np.nan,
                "dominance_base_penetrance": float(pD[posD[j]]) if j in posD else np.nan,
                "additive_effect_final_mean": (
                    float(BA[j,t] * pA[posA[j]] * gscales["A"][t] * overall_g_scale[t])
                    if j in posA and BA[j,t] != 0 else 0.0
                ),
                "dominance_effect_final_mean": (
                    float(BD[j,t] * pD[posD[j]] * gscales["D"][t] * overall_g_scale[t])
                    if j in posD and BD[j,t] != 0 else 0.0
                ),
            })
    variant_effects = pd.DataFrame(rows)
    epistasis_effects = pd.DataFrame(epi_rows)
    if len(epistasis_effects):
        for t, tr in enumerate(traits):
            epistasis_effects[f"effect_final_mean_{tr}"] = [
                float(r[f"effect_{tr}"] * r["base_penetrance"] * gscales[r["kind"]][t] * overall_g_scale[t])
                for r in epi_rows
            ]
    gxe_effects = pd.DataFrame(gxe_rows)
    if len(gxe_effects):
        for t, tr in enumerate(traits):
            gxe_effects[f"effect_final_{tr}"] = [
                float(r[f"effect_{tr}"] * gscales["GxE"][t] * overall_g_scale[t]) for r in gxe_rows
            ]

    env_names = [f"hidden{i+1}" for i in range(H.shape[1])] + [f"observed{i+1}" for i in range(O.shape[1])]
    environment = pd.DataFrame(np.column_stack([H, O]) if env_names else np.zeros((n,0)), index=idx, columns=env_names)

    grms = {}
    if compute_grms:
        denomA = float(np.sum(2*p*(1-p)))
        denomD = float(np.sum((2*p*(1-p))**2))
        grms["additive"] = pd.DataFrame((Acode @ Acode.T) / max(denomA, _EPS), index=idx, columns=idx)
        grms["dominance"] = pd.DataFrame((Dcode @ Dcode.T) / max(denomD, _EPS), index=idx, columns=idx)

    realized_h2 = np.diag(covs["G_total"]) / np.maximum(np.diag(covs["G_total"] + covs["E_total"]), _EPS)
    truth = {
        "causal_additive_indices": ia_active,
        "causal_dominance_indices": id_active,
        "additive_effect_matrix_raw": BA,
        "dominance_effect_matrix_raw": BD,
        "penetrance_base_additive": pA,
        "penetrance_base_dominance": pD,
        "penetrance_hidden_coeff_additive": gammaA,
        "penetrance_hidden_coeff_dominance": gammaD,
        "genetic_component_scales": gscales,
        "environment_component_scales": escales,
        "overall_genetic_scale": overall_g_scale,
        "penetrance_deviation_raw": {"A": A_pen, "D": D_pen, **epi_penetrance},
        "gxe_effects": gxe_effects,
        "hidden_environment_effects_raw": BH,
        "hidden_environment_effects_final": BH * escales["E_hidden"][None, :],
    }
    metadata = {
        "seed": seed, "n_samples": n, "n_variants": m, "n_traits": T,
        "n_additive": int(len(ia_active)), "n_dominance": int(len(id_active)),
        "n_epistatic_pairs": int(n_epistatic_pairs), "n_gxe": int(n_gxe),
        "n_hidden_covariates": int(H.shape[1]), "n_observed_environments": int(O.shape[1]),
        "heritability_target": h.tolist(), "heritability_realized_GE_ratio": realized_h2.tolist(),
        "genetic_weights": dict(genetic_weights), "environmental_weights": dict(environmental_weights),
        "effect_distribution": effect_distribution, "residual_distribution": residual_distribution,
        "maf_effect_alpha": float(maf_effect_alpha), "penetrance_range": tuple(map(float, penetrance_range)),
        "truth_definition": "covariances are empirical covariances of realized mechanistic components; no G/E covariance matrix was supplied to the simulator",
    }
    vi = pd.DataFrame({"af": p, "maf": np.minimum(p,1-p)}, index=Gdf.columns)
    return GenomicSimulationResult(
        genotypes=Gdf, phenotypes=phenotypes, components=components,
        trait_covariances=covs, variant_info=vi,
        variant_effects=variant_effects, epistasis_effects=epistasis_effects,
        environment=environment, grms=grms, metadata=metadata, truth=truth,
    )


def simulate_genomic_dataset(
    *,
    n_samples: int = 1000,
    n_variants: int = 5000,
    n_traits: int = 12,
    genotype_model: str = "hs_founder_mosaic",
    genotype_kwargs: Mapping[str, Any] | None = None,
    trait_kwargs: Mapping[str, Any] | None = None,
    seed: int | None = 0,
    save_path=None,
):
    """High-level mechanistic simulation of genotypes and multivariate traits.

    ``seed`` is split deterministically between genotype and phenotype
    generation so changing phenotype architecture does not silently change the
    simulated genotype population.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``1000``. Passed to
        ``simulate_genotypes`` as ``n_samples``.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``5000``. Passed to ``simulate_genotypes`` as ``n_variants``.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension. Default: ``12``. Passed to
        ``simulate_traits_from_genotypes`` as ``n_traits``.

    genotype_model : str
        Input supplied to ``simulate_genotypes`` as ``model``. See that callable's documented
        ``model`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``'hs_founder_mosaic'``. Passed to ``simulate_genotypes`` as
        ``model``.

    genotype_kwargs : Mapping[str, Any] | None
        Input supplied to ``simulate_genotypes`` as ``**expanded mapping``. See that callable's
        documented ``**expanded mapping`` contract for its type, alignment and value domain.
        This wrapper retains that contract. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value. Passed to
        ``simulate_genotypes`` as ``**expanded mapping``.

    trait_kwargs : Mapping[str, Any] | None
        Input supplied to ``simulate_traits_from_genotypes`` as ``**expanded mapping``. See that
        callable's documented ``**expanded mapping`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``simulate_traits_from_genotypes`` as ``**expanded mapping``.

    seed : int | None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        ``GenomicSimulationResult.to_dataset()``: data variables ``genotypes`` (sample,
        variant), ``phenotypes`` (sample, trait), ``components`` (component, sample, trait),
        ``trait_covariances`` (covariance, trait_i, trait_j), ``environment`` (sample,
        environment_covariate), ``grms`` (kernel, sample_i, sample_j; when computed) and ``founders``
        (founder, variant); column-wise tables ``chrom``/``pos``/``af``/``maf`` (variant,
        table ``variant_info``), ``variant_effects_*`` (variant_effect) and
        ``epistasis_effects_*`` (epistasis_pair), each rebuilt by
        ``vigor.results.result_table(ds, '<table>')``; ``attrs`` ``metadata`` and ``truth``.
    """
    genotype_kwargs = dict(genotype_kwargs or {})
    trait_kwargs = dict(trait_kwargs or {})
    ss = np.random.SeedSequence(seed)
    sg, st = ss.spawn(2)
    gseed = int(sg.generate_state(1)[0]); tseed = int(st.generate_state(1)[0])
    gout = simulate_genotypes(n_samples=n_samples, n_variants=n_variants,
                              model=genotype_model, seed=gseed, **genotype_kwargs)
    res = simulate_traits_from_genotypes(gout["genotypes"], n_traits=n_traits,
                                         seed=tseed, **trait_kwargs)
    vi = pd.DataFrame({c: gout[c].to_numpy() for c in ("chrom", "pos", "af", "maf")},
                      index=pd.Index(res.genotypes.columns, name="snp"))
    for c in res.variant_info.columns:
        vi[c] = res.variant_info[c].reindex(vi.index)
    res.variant_info = vi
    res.founders = _as_frame(gout["founders"]) if "founders" in gout else None
    res.metadata["genotype"] = gout.attrs["metadata"]
    res.metadata["master_seed"] = seed
    return maybe_save(res.to_dataset(
        software="vigor.simulate.simulate_genomic_dataset",
        parameters={"n_samples": int(n_samples), "n_variants": int(n_variants), "n_traits": int(n_traits),
                    "genotype_model": str(genotype_model), "seed": seed}), save_path)


def simulate_from_plink1(prefix, *, sample_ids=None, n_traits=12, seed=0,
                         max_variants=None, trait_kwargs=None, save_path=None):
    """Generate mechanistic phenotypes from a PLINK1 genotype source.

    This is useful when one wants realistic LD/relatedness from an observed
    genotype panel while keeping the causal architecture fully known.  The
    selected genotype matrix is materialized because the simulator must retain
    the causal variants and truth components; for very large datasets use
    ``max_variants`` to take a deterministic evenly spaced panel.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension. Default: ``12``. Passed to
        ``simulate_traits_from_genotypes`` as ``n_traits``.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``simulate_traits_from_genotypes`` as ``seed``.

    max_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``None``.

    trait_kwargs : object (see the cited input contract)
        Input supplied to ``simulate_traits_from_genotypes`` as ``**expanded mapping``. See that
        callable's documented ``**expanded mapping`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Passed to ``simulate_traits_from_genotypes`` as ``**expanded mapping``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        ``GenomicSimulationResult.to_dataset()`` with the same variables (``chrom``/``pos``
        from the BIM; ``af``/``maf`` from the simulation) and attrs as
        :func:`simulate_genomic_dataset`; ``attrs['metadata']`` also records
        ``plink_source`` and ``plink_variant_panel_size``.
    """
    from .io import PlinkReader
    reader = PlinkReader(str(prefix), dtype=np.float32)
    all_ids = reader.fam["iid"].astype(str).to_numpy()
    if sample_ids is None:
        rows = np.arange(reader.n_samples)
        ids = all_ids
    else:
        ids = np.asarray(sample_ids, dtype=str)
        rows = pd.Index(all_ids).get_indexer(ids)
        if np.any(rows < 0):
            raise KeyError(f"PLINK FAM is missing requested IID(s): {ids[rows<0][:5].tolist()}")
    if max_variants is None or int(max_variants) >= reader.n_variants:
        vidx = np.arange(reader.n_variants)
    else:
        vidx = np.unique(np.linspace(0, reader.n_variants-1, int(max_variants), dtype=int))
    blocks = []
    # Use the reader's sparse variant-index path so evenly spaced panels do not
    # force decoding the huge span between selected markers.
    for _meta, block in reader.iter_blocks(block_size='auto', variant_index=vidx, sample_index=rows):
        blocks.append(np.asarray(block, dtype=np.float32))
    X = np.column_stack(blocks)
    bim = reader.bim.iloc[vidx].copy()
    names = bim["snp"].astype(str).to_list()
    Gdf = pd.DataFrame(X, index=ids, columns=names)
    res = simulate_traits_from_genotypes(Gdf, n_traits=n_traits, seed=seed, **dict(trait_kwargs or {}))
    vi = pd.DataFrame({"chrom": bim["chrom"].astype(str).to_numpy(), "pos": bim["pos"].to_numpy(int)}, index=names)
    for c in res.variant_info.columns:
        vi[c] = res.variant_info[c]
    res.variant_info = vi
    res.metadata["plink_source"] = str(prefix)
    res.metadata["plink_variant_panel_size"] = int(len(vidx))
    return maybe_save(res.to_dataset(
        software="vigor.simulate.simulate_from_plink1",
        parameters={"prefix": str(prefix), "n_traits": int(n_traits), "seed": seed,
                    "max_variants": None if max_variants is None else int(max_variants)}), save_path)


def write_simulation_plink(sim, prefix):
    """Write a simulation's genotypes as a PLINK1 BED/BIM/FAM fileset.

    Parameters
    ----------
    sim : xarray.Dataset or GenomicSimulationResult
        Result of ``simulate_genomic_dataset``/``simulate_from_plink1`` (Dataset with
        ``genotypes`` on (sample, variant) and the ``variant_info`` table), or the
        legacy dataclass.
    prefix : str or os.PathLike
        Fileset prefix without the .bed/.bim/.fam suffix.

    Returns
    -------
    str
        The prefix written.
    """
    if isinstance(sim, GenomicSimulationResult):
        return sim.write_plink(prefix)
    from .results import result_table
    from .testutils import write_plink
    g = sim["genotypes"].transpose("sample", "variant")
    variants = [str(v) for v in g["variant"].values]
    vi = result_table(sim, "variant_info")
    vi.index = vi.index.astype(str)
    vi = vi.reindex(variants)
    write_plink(str(prefix), np.asarray(g.values, float), [str(s) for s in g["sample"].values],
                vi["chrom"].astype(str).to_numpy(), vi["pos"].to_numpy(int))
    return str(prefix)


def simulation_scenario(name: str) -> dict[str, Any]:
    """Return a documented mechanistic architecture preset.

    The presets change causal mechanisms and marginal variance allocation only;
    none of them specify a truth covariance matrix.  They are intended for
    robustness benchmarks rather than as claims about any one biological trait.

    Parameters
    ----------
    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.
        Source-recognized literal values: ``additive_sparse``, ``complex``, ``dominance_rich``,
        ``epistasis_rich``, ``gxe_penetrance``, ``polygenic_additive``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    Returns
    -------
    object
        Computed result from ``dict(presets[key])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    key = str(name).lower().replace("-", "_")
    presets = {
        "polygenic_additive": dict(
            n_additive=800, n_dominance=0, n_epistatic_pairs=0, n_gxe=0,
            n_hidden_covariates=5, penetrance_range=(0.95, 1.0),
            genetic_weights={"A": 1.0, "D": 0.0, "AA": 0.0, "AD": 0.0, "DD": 0.0, "GxE": 0.0},
        ),
        "additive_sparse": dict(
            n_additive=100, n_dominance=10, n_epistatic_pairs=4, n_gxe=6,
            n_hidden_covariates=5, penetrance_range=(0.85, 1.0),
            genetic_weights={"A": .88, "D": .03, "AA": .02, "AD": .01, "DD": .01, "GxE": .05},
        ),
        "dominance_rich": dict(
            n_additive=90, n_dominance=80, dominance_overlap=.75, n_epistatic_pairs=10, n_gxe=8,
            n_hidden_covariates=5, penetrance_range=(0.75, 1.0),
            genetic_weights={"A": .48, "D": .35, "AA": .05, "AD": .04, "DD": .03, "GxE": .05},
        ),
        "epistasis_rich": dict(
            n_additive=80, n_dominance=35, n_epistatic_pairs=90, n_gxe=10,
            epistasis_mix={"AA": .55, "AD": .25, "DD": .20},
            n_hidden_covariates=6, penetrance_range=(0.70, .98),
            genetic_weights={"A": .40, "D": .08, "AA": .22, "AD": .10, "DD": .08, "GxE": .12},
        ),
        "gxe_penetrance": dict(
            n_additive=100, n_dominance=45, n_epistatic_pairs=20, n_gxe=80,
            n_hidden_covariates=8, n_observed_environments=1,
            penetrance_range=(0.35, .85), penetrance_hidden_strength=1.4, gxe_hidden_fraction=.75,
            genetic_weights={"A": .45, "D": .08, "AA": .04, "AD": .03, "DD": .02, "GxE": .38},
            environmental_weights={"E_hidden": .55, "E_noise": .45},
        ),
        "complex": dict(
            n_additive=120, n_dominance=60, dominance_overlap=.60, n_epistatic_pairs=50, n_gxe=45,
            n_hidden_covariates=7, n_observed_environments=1,
            penetrance_range=(0.55, .98), penetrance_hidden_strength=1.0, gxe_hidden_fraction=.65,
            genetic_weights={"A": .55, "D": .10, "AA": .09, "AD": .05, "DD": .04, "GxE": .17},
            environmental_weights={"E_hidden": .40, "E_noise": .60},
        ),
    }
    if key not in presets:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(presets)}")
    return dict(presets[key])


__all__ = [
    "GenomicSimulationResult", "simulate_genotypes", "gcta_codings",
    "simulate_traits_from_genotypes", "simulate_genomic_dataset",
    "simulate_from_plink1", "simulation_scenario",
]
