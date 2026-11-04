"""Shared source, LOCO and streaming adapters for the exact score engine.

These adapters never substitute a genome-wide kernel for a missing LOCO kernel.
Phenotype rows are the canonical sample order. Arrays without labels are
positional; labeled matrices are explicitly aligned before numerical conversion.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import numpy as np
import pandas as pd


def scan_inputs(traits, genotypes, *, groups=None, maxmem=None):
    """Validate phenotypes and open a bounded-memory genotype source.

    Parameters
    ----------
    traits : pandas.DataFrame (n, t)
        Unique sample/trait labels, numeric Gaussian outcomes, NaN missing cells.
    genotypes : DataFrame, path, or GenotypeSource
        Sample by variant dosages, or an existing physical source.
    groups : sequence (m,), optional
        Exclusion group per physical variant; replaces chromosome metadata.
    maxmem : bytes or unit string, optional
        Genotype block memory ceiling.

    Returns
    -------
    (DataFrame, GenotypeSource, DataFrame)
        Validated phenotypes, source, and physical variant metadata.

    Complexity
    ----------
    O(nt + m + n_source) time/space for validation and metadata, excluding lazy
    genotype bytes. DataFrame genotypes already reside in caller memory.
    """
    from ..genotype_source import ArrayGenotypeSource, open_genotypes
    if not isinstance(traits, pd.DataFrame):
        raise TypeError("traits must be a labeled pandas DataFrame")
    Y = traits.copy()
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)
    if Y.index.has_duplicates or Y.columns.has_duplicates or Y.empty:
        raise ValueError("sample and trait labels must be unique and nonempty")
    Y = Y.apply(pd.to_numeric, errors="raise")
    if np.isinf(Y.to_numpy(float)).any() or (Y.notna().sum() < 2).any():
        raise ValueError("traits require at least two finite observations per column")
    if isinstance(genotypes, pd.DataFrame):
        X = genotypes.copy(deep=False)
        ids = X.index.astype(str)
        if ids.has_duplicates:
            raise ValueError("genotype sample IDs must be unique")
        vm = pd.DataFrame({"variant_id": X.columns.astype(str),
                           "chromosome": "all", "position": np.arange(X.shape[1])})
        source = ArrayGenotypeSource(X.to_numpy(), sample_ids=ids, variant_metadata=vm)
    else:
        source = open_genotypes(genotypes, maxmem=maxmem)
    if not pd.Index(source.sample_ids.astype(str)).is_unique:
        raise ValueError("genotype sample IDs must be unique")
    absent = Y.index.difference(pd.Index(source.sample_ids.astype(str)))
    if len(absent):
        raise ValueError(f"phenotype samples absent from genotypes: {list(absent[:5])}")
    vm = source.variant_metadata.copy().reset_index(drop=True)
    vm["variant_index"] = np.arange(len(vm), dtype=np.int64)
    if groups is not None:
        if len(groups) != len(vm):
            raise ValueError("groups must have one label per physical variant")
        vm["chromosome"] = np.asarray(groups).astype(str)
    if "chromosome" not in vm:
        vm["chromosome"] = "all"
    if "variant_id" not in vm or vm.variant_id.astype(str).duplicated().any():
        raise ValueError("variant IDs must be present and unique")
    vm.attrs["_physical_variant_lookup"] = pd.Index(vm.variant_id.astype(str))
    return Y, source, vm


def align_kernel(kernel, samples):
    """Align n by n labeled kernels, preserving operator/positional arrays.

    Returns the aligned kernel or a name-to-kernel mapping. O(n²) data movement
    for dense reindexing; operators are not materialized. Missing labels fail.
    """
    if isinstance(kernel, pd.DataFrame):
        K = kernel.copy(deep=False)
        K.index = K.index.astype(str)
        K.columns = K.columns.astype(str)
        if K.index.has_duplicates or K.columns.has_duplicates:
            raise ValueError("kernel labels must be unique")
        return K.loc[samples, samples]
    if isinstance(kernel, Mapping):
        return {str(k): align_kernel(v, samples) for k, v in kernel.items()}
    if getattr(kernel, "shape", None) != (len(samples), len(samples)):
        raise ValueError("positional kernel must match the phenotype sample axis")
    return kernel


def scan_covariates(covariates, samples):
    """Normalize and align finite covariates once; preserve numeric ID matches.

    Returns a DataFrame or None. O(nq) time/space, n samples and q covariates.
    Labels must be unique after conversion to strings; no outcome imputation.
    """
    if covariates is None:
        return None
    if isinstance(covariates, pd.DataFrame):
        C = covariates.copy()
        C.index = C.index.astype(str)
        if C.index.has_duplicates:
            raise ValueError("covariate sample IDs must be unique")
        C = C.loc[samples]
    else:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C[:, None]
        if C.ndim != 2 or C.shape[0] != len(samples):
            raise ValueError("covariates must be sample by covariate")
        C = pd.DataFrame(C, index=samples)
    if not np.isfinite(C.to_numpy(float)).all():
        raise ValueError("covariates must be finite")
    return C


def physical_variant_indices(block, vm):
    """Map block IDs to physical rows instead of trusting a source's local index.

    Returns unique positions (b,); unknown/duplicated IDs fail. O(m+b) lookup
    time and O(m) metadata indexing memory, excluding numerical genotype data.
    """
    if "variant_id" not in block.variants:
        raise ValueError("genotype blocks require variant_id metadata")
    ids = pd.Index(block.variants.variant_id.astype(str))
    if ids.has_duplicates or len(ids) != block.data.shape[1]:
        raise ValueError("block variant IDs must be unique and match data width")
    lookup = vm.attrs.get("_physical_variant_lookup")
    if lookup is None:
        lookup = pd.Index(vm.variant_id.astype(str))
        vm.attrs["_physical_variant_lookup"] = lookup
    indices = lookup.get_indexer(ids)
    if np.any(indices < 0):
        raise ValueError("genotype block contains unknown variant IDs")
    return indices


def validated_source_blocks(source, vm, samples, *, block_size="auto"):
    """Yield blocks/physical indices after sample, duplicate and coverage checks.

    Custom source metadata may reset its row indices. IDs must still enumerate
    the declared variant set exactly once. O(m) boolean metadata and O(nb)
    genotype-block memory; every source pass validates complete coverage.
    """
    seen = np.zeros(len(vm), bool)
    for block in source.iter_blocks(samples=samples, block_size=block_size):
        if not np.array_equal(np.asarray(block.sample_ids).astype(str), np.asarray(samples).astype(str)):
            raise ValueError("genotype source did not preserve requested sample order")
        if np.ndim(block.data) != 2 or block.data.shape[0] != len(samples):
            raise ValueError("genotype source block has an invalid sample axis")
        idx = physical_variant_indices(block, vm)
        if seen[idx].any():
            raise ValueError("genotype source repeated variants across blocks")
        seen[idx] = True
        yield block, idx
    if not seen.all():
        raise ValueError("genotype source omitted declared variants")


def covariance_matrices(result):
    """Extract component G and E from a raw or xarray covariance fit.

    Returns a component mapping (multi-kernel) or a single matrix and E. O(ct²)
    space/time for c component conversions. Fits with an h2 prior already contain
    it in their inference covariance, which is used as returned.
    """
    from ..multivariate import _result_view
    view = _result_view(result)
    components = view.get("G_components")
    if components is not None and len(components) > 1:
        G = {str(k): np.asarray(v) for k, v in components.items()}
    else:
        G = np.asarray(view["G"])
    return G, np.asarray(view["E"])


def background_for_group(source, vm, samples, group, *, kernels=None,
                         background_kernels=None, background="loco",
                         block_size="auto", min_maf=1e-4, compute_dtype=np.float32):
    """Resolve a declared background or construct exact leave-group-out GRM.

    ``background_kernels`` maps exclusion groups to kernels (or named kernel
    collections); ``kernels`` is a common explicitly supplied background. For
    automatic construction all variants of the tested group are excluded.
    A canonical chromosome GRM store is supported via ``kernels=path``.

    Returns
    -------
    ndarray, operator or mapping
        Aligned background in phenotype row order.

    Complexity
    ----------
    Automatic construction O(n² m_background) time, O(n²+nb) memory. It rereads
    background variants for each group; persisted chromosome stores avoid this.
    """
    if background not in {"loco", "provided"}:
        raise ValueError("background must be 'loco' or 'provided'")
    if background_kernels is not None:
        if callable(background_kernels):
            return align_kernel(background_kernels(group), samples)
        if group not in background_kernels and str(group) not in background_kernels:
            raise KeyError(f"no background kernel for exclusion group {group!r}")
        return align_kernel(background_kernels.get(group, background_kernels.get(str(group))), samples)
    if kernels is not None:
        if isinstance(kernels, (str, Path)):
            from ..grm_store import open_grm
            K = open_grm(kernels, loco=str(group), materialize=True, labeled=True)
            K = pd.DataFrame(K.values, index=K.sample_i.values.astype(str),
                             columns=K.sample_j.values.astype(str))
            return align_kernel(K, samples)
        return align_kernel(kernels, samples)
    if background == "provided":
        raise ValueError("background='provided' requires a kernel")
    keep = vm.chromosome.astype(str).to_numpy() != str(group)
    if not keep.any():
        raise ValueError("LOCO has no background variants; supply an independent kernel")
    from ..grm_backend import grm_from_blocks
    def blocks():
        for block, physical in validated_source_blocks(source, vm, samples, block_size=block_size):
            mask = keep[physical]
            if mask.any():
                yield np.asarray(block.data[:, mask], dtype=np.float32)
    out, _, _ = grm_from_blocks(blocks(), len(samples), method="van",
                          components=("additive",), min_maf=min_maf,
                          dtype=compute_dtype, output_dtype=compute_dtype)
    return out["additive"]


def genotype_blocks(source, vm, samples, group, *, block_size="auto"):
    """Yield sample-aligned dosage frames and metadata for one exclusion group.

    O(nm) decoding time and O(nb) memory. Physical indices are retained across
    source blocks, including an explicit custom grouping independent of chroms.
    """
    # Filtering physical rows also supports custom groups not matching source chroms.
    keep = vm.chromosome.astype(str).to_numpy() == str(group)
    for block, idx in validated_source_blocks(source, vm, samples, block_size=block_size):
        selected = idx[keep[idx]]
        if not len(selected):
            continue
        meta = vm.iloc[selected].copy()
        data = np.asarray(block.data[:, keep[idx]], dtype=np.float32)
        yield pd.DataFrame(data, index=samples, columns=meta.variant_id.astype(str)), meta


def attach_variant_metadata(ds, meta, *, group):
    """Attach physical variant coordinates and counted-allele metadata to a block.

    Returns the Dataset with O(b) metadata, leaving O(bt) numerical arrays intact.
    """
    ds = ds.assign_coords(variant=meta.variant_index.to_numpy(np.int64))
    ds = ds.assign_coords(variant_id=("variant", meta.variant_id.astype(str).to_numpy()),
                          chromosome=("variant", meta.chromosome.astype(str).to_numpy()),
                          loco_group=("variant", np.repeat(str(group), len(meta))))
    for name in ("position", "effect_allele", "other_allele", "ref", "alt"):
        if name in meta:
            ds = ds.assign_coords({name: ("variant", meta[name].to_numpy())})
    ds.attrs["effect_allele_convention"] = ("effect_allele coordinate declares counted allele"
        if "effect_allele" in meta else "counted allele not declared by source")
    return ds


def finish_scan(blocks, *, out=None, max_return_rows=2_000_000,
                variant_chunk=1024, shard_variants=16384):
    """Consume a block iterator, streaming to Zarr when out is given.

    With out, O(one block) numerical memory plus writer buffers. Otherwise the
    caller explicitly requests an in-memory result, capped at max_return_rows
    score cells. Exceptions propagate without marking partial stores complete.
    """
    raw = iter(blocks)
    def ordinal_blocks():
        offset = 0
        for ds in raw:
            # Scan order need not be physical order (e.g. interleaved custom
            # groups). Keep the physical coordinate explicitly and use an
            # increasing ordinal axis for reliable streamed append validation.
            n = ds.sizes["variant"]
            ds = ds.assign_coords(variant_index=("variant", ds.variant.values.copy()),
                                  variant=np.arange(offset, offset+n, dtype=np.int64))
            offset += n
            yield ds
    ordered = ordinal_blocks()
    try:
        if out is not None:
            from .streaming import write_score_stream
            path = str(out) if str(out).endswith(".zarr") else str(out) + ".zarr"
            return write_score_stream(ordered, path, variant_chunk=variant_chunk,
                                      shard_variants=shard_variants)
        import xarray as xr
        retained = []
        cells = 0
        group_metadata = {}
        for ds in ordered:
            cells += ds.sizes["variant"] * ds.sizes["trait"]
            if max_return_rows is not None and cells > int(max_return_rows):
                raise MemoryError("score results exceed max_return_rows; supply out= for streaming")
            retained.append(ds)
            for group, values in ds.attrs.get("scan_group_metadata", {}).items():
                if group in group_metadata and group_metadata[group] != values:
                    raise ValueError(f"scan group metadata changed within group {group}")
                group_metadata[group] = values
        if not retained:
            raise ValueError("no variants matched the requested groups")
        result = xr.concat(retained, dim="variant", data_vars="all", coords="minimal",
                           compat="equals", join="exact")
        if group_metadata:
            result.attrs["scan_group_metadata"] = group_metadata
        return result
    finally:
        ordered.close()
        closer = getattr(raw, "close", None)
        if closer is not None:
            closer()


def exact_mv_gwas(traitdf, genotypes="genotypes/genotypes", *, GRM=None, G=None,
                  E=None, background_kernels=None, background="loco", groups=None,
                  chromosomes=None, covariance_kwargs=None, covariates=None,
                  block_size=128, out=None, max_return_rows=2_000_000,
                  solver_tol=2e-6, solver_maxiter=400, work_dtype="float32",
                  rhs_block_size=16, preconditioner="kronecker", jitter=0.0,
                  min_maf=1e-4, maxmem=None,
                  variant_chunk=1024, shard_variants=16384):
    """Run exact multivariate score GWAS from array, BED, PGEN or Zarr sources.

    Parameters
    ----------
    traitdf : DataFrame (n,t)
        Numeric Gaussian phenotypes with missing cells and unique animal labels.
    genotypes : DataFrame, path or source
        Sample by variant counted-allele dosage; blocks are decoded lazily.
    GRM, background_kernels : kernel or mapping, optional
        Common supplied kernel, or group-to-background mapping. Absent inputs
        construct leave-group-out additive kernels. Multi-kernel inputs require
        matching G components. A common supplied kernel's exclusion is the
        caller's responsibility and is recorded in the result.
    G, E : covariance matrices or mappings, optional
        Both supplied together, or estimated anew per exclusion group.
    background : {'loco','provided'}
        Automatic exclusion policy or explicit-only background.
    groups, chromosomes : sequences, optional
        Per-variant exclusion labels and selected labels to scan.
    covariance_kwargs : mapping, optional
        Arguments for a per-group genetic_varcov_PSD fit.
    covariates : DataFrame or array (n,q), optional
        Fixed effects in addition to mandatory per-trait intercepts.
    block_size, rhs_block_size : positive int
        Source variants and simultaneous PCG right-hand sides, respectively.
    out : path, optional
        Stream into sharded Zarr v3; otherwise return bounded in-memory data.
    max_return_rows : int or None
        In-memory score-cell limit.
    solver_tol, solver_maxiter : positive float, int
        Verified relative residual threshold and iteration cap.
    work_dtype : {'float32','float64'}
        Matrix product precision; reductions/small solves use float64.
    preconditioner : {'kronecker','diag'}
        Solver-only inverse approximation; row-block-only operators require diag.
    jitter : nonnegative float
        Explicit covariance stabilization, default zero, recorded by the core.
    min_maf : float in [0,.5]
        Automatic background-kernel variant filter.
    maxmem : byte budget or unit string, optional
        Genotype I/O memory ceiling.
    variant_chunk, shard_variants : positive int
        On-disk score chunk and enclosing shard lengths.

    Returns
    -------
    xarray.Dataset
        Variant by trait score statistics and per-variant exact omnibus tests.
        The null is exact conditional on supplied/fitted covariance, to the
        recorded solver tolerance; it does not integrate covariance uncertainty.

    Complexity
    ----------
    For dense kernels, each covariance operator application costs O(n²t+nt²)
    per RHS, plus the eig preconditioner setup described in gwas.exact. Exact
    per-SNP information requires t RHS; total scan cost depends on PCG iterations.
    """
    from .exact import prepare_score_null, score_block
    from ..multivariate import _genetic_varcov_PSD_raw
    Y, source, vm = scan_inputs(traitdf, genotypes, groups=groups, maxmem=maxmem)
    covariates = scan_covariates(covariates, Y.index)
    if (G is None) != (E is None):
        raise ValueError("G and E must be supplied together")
    labels = list(dict.fromkeys(vm.chromosome.astype(str)))
    if chromosomes is not None:
        requested = list(map(str, chromosomes))
        if not set(requested).issubset(labels):
            raise ValueError("requested chromosome/exclusion group is absent")
        labels = requested
    if len(set(labels)) != len(labels):
        raise ValueError("requested exclusion groups must be unique")
    def scan_blocks():
        for group in labels:
            K = background_for_group(source, vm, Y.index, group, kernels=GRM,
                background_kernels=background_kernels, background=background,
                block_size=block_size, min_maf=min_maf, compute_dtype=work_dtype)
            if G is None:
                kw = dict(covariance_kwargs or {})
                kw.setdefault("n_boot", 0)
                kw.setdefault("progress", False)
                # Fit covariance on fixed-effect residuals; the score null still
                # projects the complete fixed-effect design with its GLS metric.
                fit_traits = Y
                if covariates is not None:
                    C = covariates.to_numpy(float)
                    if C.ndim == 1:
                        C = C[:, None]
                    if C.shape[0] != len(Y) or not np.isfinite(C).all():
                        raise ValueError("covariates must be finite and sample aligned")
                    C = np.column_stack([np.ones(len(Y)), C])
                    fit_traits = Y.copy()
                    for column in Y:
                        mask = Y[column].notna().to_numpy()
                        fit_traits.loc[mask, column] -= C[mask] @ np.linalg.lstsq(C[mask], Y.loc[mask, column], rcond=None)[0]
                fit = _genetic_varcov_PSD_raw(K, fit_traits, **kw)
                Gc, Ec = covariance_matrices(fit)
            else:
                Gc, Ec = G, E
            null = prepare_score_null(Y, K, Gc, Ec, covariates=covariates,
                solver_tol=solver_tol, solver_maxiter=solver_maxiter,
                work_dtype=work_dtype, rhs_block_size=rhs_block_size,
                preconditioner=preconditioner, jitter=jitter)
            for X, meta in genotype_blocks(source, vm, Y.index, group, block_size=block_size):
                ds = attach_variant_metadata(score_block(null, X), meta, group=group)
                ds.attrs["background_source"] = "group_mapping" if background_kernels is not None else "supplied_common" if GRM is not None else "automatic_leave_group_out"
                yield ds
    def blocks():
        try:
            yield from scan_blocks()
        finally:
            if not hasattr(genotypes, "iter_blocks") and hasattr(source, "close"):
                source.close()
    return finish_scan(blocks(), out=out, max_return_rows=max_return_rows,
                       variant_chunk=variant_chunk, shard_variants=shard_variants)


def exact_local_gwas(traitdf, genotypes, *, window_mb=10.0, exclusion_mb=None,
                     chromosomes=None, block_size=128, min_maf=1e-4, **kwargs):
    """Run exact leave-window-out tests over nonoverlapping physical windows.

    Parameters
    ----------
    traitdf, genotypes
        Labeled phenotypes and dosage source, as in exact_mv_gwas.
    window_mb : positive float
        Width of disjoint test windows along each chromosome.
    exclusion_mb : nonnegative float, optional
        Flank removed on either side of each test window (default window_mb).
        All other chromosomes remain in the background.
    chromosomes : sequence, optional
        Chromosomes to test, without removing others from the background.
    block_size, min_maf, **kwargs
        Exact scan source and null controls. GRM/background_kernels/groups and
        background overrides are unavailable because this routine constructs the
        precise leave-window-out background.

    Returns
    -------
    Dataset
        Exact score results; each physical variant is tested once.

    Complexity
    ----------
    O(n²*m_background) kernel work per test window plus exact score solves.
    Only one background kernel is resident; this correctness path is costly.
    """
    forbidden = {"GRM", "background_kernels", "groups", "background"}.intersection(kwargs)
    if forbidden:
        raise TypeError("local scan constructs its own background; unsupported: " + ", ".join(sorted(forbidden)))
    flank = window_mb if exclusion_mb is None else exclusion_mb
    if not np.isfinite(window_mb) or window_mb <= 0 or not np.isfinite(flank) or flank < 0:
        raise ValueError("window_mb must be positive and exclusion_mb nonnegative")
    Y, source, vm = scan_inputs(traitdf, genotypes)
    if "position" not in vm:
        raise ValueError("physical variant positions are required for local scans")
    pos = pd.to_numeric(vm.position, errors="raise").to_numpy(float)
    chrom = vm.chromosome.astype(str).to_numpy()
    width = float(window_mb) * 1e6
    if not np.isfinite(pos).all() or np.any(pos < 0):
        raise ValueError("physical positions must be finite and nonnegative")
    window = np.floor(pos / width).astype(np.int64)
    groups = np.array([f"{c}:window{w}" for c, w in zip(chrom, window)])
    selected = list(dict.fromkeys(groups))
    if chromosomes is not None:
        wanted = set(map(str, chromosomes))
        if not wanted.issubset(set(chrom)):
            raise ValueError("requested chromosome is absent")
        selected = [g for g in selected if chrom[np.flatnonzero(groups == g)[0]] in wanted]
    from ..grm_backend import grm_from_blocks
    def background(group):
        first = np.flatnonzero(groups == group)[0]
        start = window[first] * width - float(flank) * 1e6
        end = (window[first] + 1) * width + float(flank) * 1e6
        keep = (chrom != chrom[first]) | (pos < start) | (pos >= end)
        if not keep.any():
            raise ValueError(f"no background variants outside exclusion window {group}")
        def blocks():
            for block, idx in validated_source_blocks(source, vm, Y.index, block_size=block_size):
                mask = keep[idx]
                if mask.any():
                    yield block.data[:, mask]
        kernels, _, _ = grm_from_blocks(blocks(), len(Y), method="van",
            components=("additive",), min_maf=min_maf)
        return kernels["additive"]
    try:
        return exact_mv_gwas(Y, source, groups=groups, chromosomes=selected,
            background_kernels=background, block_size=block_size, min_maf=min_maf, **kwargs)
    finally:
        if not hasattr(genotypes, "iter_blocks") and hasattr(source, "close"):
            source.close()
