"""Validated founder-probability kernels and exact per-locus joint score tests."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _founder_array(value):
    P = np.asarray(value, dtype=np.float32)
    if P.ndim != 3 or min(P.shape) < 1 or P.shape[2] < 2:
        raise ValueError("founder_probability must be sample x variant x >=2 founders")
    finite = np.isfinite(P)
    complete = finite.all(axis=2)
    if np.isinf(P).any() or np.any(finite.any(axis=2) & ~complete):
        raise ValueError("founder calls must be all finite or all NaN per sample/locus")
    if np.any(finite & ((P < 0) | (P > 1))):
        raise ValueError("founder probabilities must lie in [0,1]")
    if not complete.any() or np.any(np.abs(P.sum(axis=2)[complete] - 1) > 5e-3):
        raise ValueError("observed founder probabilities must sum to one")
    return P


def founder_kernel(founder_probability, *, center=True, dtype=np.float32):
    """Build a mean-imputed founder-ancestry relationship kernel.

    Parameters
    ----------
    founder_probability : array (n,m,f)
        Probabilities in [0,1] summing to one; whole-locus missing calls are NaN.
    center : bool
        Subtract observed per-locus founder means before mean imputation.
    dtype : float32 or float64
        Dense GEMM and output type; normalization reductions use float64.

    Returns
    -------
    ndarray (n,n)
        Ancestry feature covariance, scaled by summed founder variances.

    Complexity
    ----------
    O(n²mf) time, O(nmf+n²) memory. Use only background loci for GWAS.
    """
    P = _founder_array(founder_probability)
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    observed = np.isfinite(P)
    count = observed.sum(axis=0)
    means = np.divide(np.where(observed, P, 0).sum(axis=0, dtype=np.float64), count,
                      out=np.zeros(P.shape[1:]), where=count > 0)
    centered = np.where(observed, P - means[None], 0)
    variances = np.divide(np.sum(centered**2, axis=0, dtype=np.float64), count,
                          out=np.zeros_like(means), where=count > 0)
    den = float(variances.sum())
    if not np.isfinite(den) or den <= 0:
        raise ValueError("founder background contains no variable ancestry")
    X = (centered if center else np.where(observed, P, means[None])).reshape(P.shape[0], -1).astype(dtype)
    return (X @ X.T / den).astype(dtype, copy=False)


def run_haplotype_gwas(founder_probability, traits, *, G=None, E=None, kernel=None,
                       groups=None, block_size=64, out=None, covariates=None,
                       max_return_rows=2_000_000, variant_chunk=1024,
                       shard_variants=16384, **null_kwargs):
    """Jointly test all F-1 founder contrasts at each locus in the exact core.

    Parameters
    ----------
    founder_probability : ndarray or DataArray (n,m,f)
        Validated probabilities; labeled DataArray axes must be sample, variant,
        founder. The final founder is the reference contrast.
    traits : DataFrame (n,t)
        Labeled Gaussian phenotypes, NaN missing cells.
    G, E : trait covariance matrices or named genetic components
        Both required. They remain fixed throughout the scan.
    kernel : independent animal kernel or named kernels, optional
        Supplied background in phenotype row order. If absent, groups is required
        and founder kernels exclude every locus in the tested group.
    groups : sequence (m,), optional
        Exclusion label per locus (typically chromosome).
    block_size : positive int
        Loci per returned/written block.
    out : path, optional
        Stream sharded Zarr v3 and return its lazy Dataset.
    covariates : sample by covariate table/array, optional
        Additional fixed effects; intercepts are always projected.
    max_return_rows : int or None
        In-memory score-cell ceiling.
    variant_chunk, shard_variants : positive int
        Persisted chunk and shard lengths.
    **null_kwargs : mapping
        prepare_score_null controls, e.g. solver_tol and rhs_block_size.

    Returns
    -------
    xarray.Dataset
        One variant row per locus, contrast:trait coordinates, and a single
        joint omnibus using the rank of all founder-by-trait contrasts.

    Complexity
    ----------
    O(m*(F-1)*t) exact covariance RHS solves plus per-locus information solves.
    Probability input is caller-resident; output is one block in memory with out.
    """
    from .gwas.exact import prepare_score_null, score_design_block
    from .gwas.scan import align_kernel, finish_scan
    if not isinstance(traits, pd.DataFrame):
        raise TypeError("traits must be a labeled DataFrame")
    Y = traits.copy()
    Y.index = Y.index.astype(str); Y.columns = Y.columns.astype(str)
    if Y.index.has_duplicates or Y.columns.has_duplicates:
        raise ValueError("trait sample/column labels must be unique")
    names = None
    founders = None
    if hasattr(founder_probability, "dims"):
        if tuple(founder_probability.dims) != ("sample", "variant", "founder"):
            raise ValueError("founder DataArray dimensions must be sample, variant, founder")
        ids = pd.Index(founder_probability.sample.values.astype(str))
        if ids.has_duplicates or set(ids) != set(Y.index):
            raise ValueError("founder sample labels must match traits uniquely")
        founder_probability = founder_probability.assign_coords(sample=ids).sel(sample=Y.index)
        names = founder_probability.variant.values.astype(str)
        founders = founder_probability.founder.values.astype(str)
    P = _founder_array(founder_probability)
    if P.shape[0] != len(Y):
        raise ValueError("founder sample count must match traits")
    if G is None or E is None:
        raise ValueError("both G and E covariance estimates are required")
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    m = P.shape[1]
    labels = np.repeat("provided", m) if groups is None else np.asarray(groups).astype(str)
    if labels.shape != (m,):
        raise ValueError("groups must contain one label per locus")
    if kernel is None and len(np.unique(labels)) < 2:
        raise ValueError("provide an independent kernel or at least two background exclusion groups")
    names = np.array([f"locus{i}" for i in range(m)]) if names is None else names
    founders = np.array([f"founder{i}" for i in range(P.shape[2])]) if founders is None else founders
    if len(np.unique(names)) != m or len(np.unique(founders)) != P.shape[2]:
        raise ValueError("locus and founder labels must be unique")
    def blocks():
        for group in dict.fromkeys(labels):
            K = align_kernel(kernel, Y.index) if kernel is not None else founder_kernel(P[:, labels != group, :])
            null = prepare_score_null(Y, K, G, E, covariates=covariates, **null_kwargs)
            idx = np.flatnonzero(labels == group)
            for start in range(0, len(idx), int(block_size)):
                selected = idx[start:start+int(block_size)]
                ds = score_design_block(null, P[:, selected, :-1],
                    contrast_names=founders[:-1], variant_indices=selected, variant_names=names[selected])
                ds = ds.assign_coords(loco_group=("variant", labels[selected]))
                ds.attrs.update(reference_founder=str(founders[-1]),
                    tested_design="joint F-1 founder contrasts per locus",
                    background="provided independent kernel" if kernel is not None else "leave-group-out founder kernel")
                yield ds
    return finish_scan(blocks(), out=out, max_return_rows=max_return_rows,
                       variant_chunk=variant_chunk, shard_variants=shard_variants)


__all__ = ["founder_kernel", "run_haplotype_gwas"]
