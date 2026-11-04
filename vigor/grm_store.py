"""Persistent chromosome GRMs and LOCO operators for VIGOR 2.9.

Chromosome matrices are cached once together with the exact normalization
weights used by VIGOR's additive and Vitezica-dominance definitions. Genome-
wide and LOCO kernels are weighted reconstructions of those components.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator

from .genotype_source import open_genotypes
from .store import VigorStore, open_store, sample_hash, array_hash
from .zarr_io import _create_array, _unicode_dtype
from .grm_methods import GRM_SCHEMA_VERSION

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


def _thread_ctx(n):
    """Construct a BLAS thread-limit context for the requested numerical workload.

    Parameters
    ----------
    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Source-recognized literal values: ``auto``.
        These select the corresponding branches below; comparison evidence alone does not
        establish an exhaustive enum. Source comparisons: ``n in {None, 'auto', 0}``. See the
        rejection/fallback branches for enforcement. Passed to ``threadpool_limits`` as
        ``limits``.
    """
    if threadpool_limits is None or n in {None, "auto", 0}: return nullcontext()
    return threadpool_limits(limits=int(n), user_api="blas")


def _source_signature(src):
    """Return the source signature result computed by ``h.hexdigest``.

    Parameters
    ----------
    src : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.
    """
    h=hashlib.sha256()
    h.update(sample_hash(src.sample_ids).encode())
    vm=src.variant_metadata
    for col in ("variant_id","chromosome","position"):
        if col in vm:
            h.update("\n".join(vm[col].astype(str)).encode()); h.update(b"\0")
    if hasattr(src,'data'):
        h.update(array_hash(src.data).encode())
    elif hasattr(src,'ds'):
        from dask.base import tokenize
        h.update(tokenize(src.ds).encode())
        if getattr(src,'_path',None):
            import zarr
            zg=zarr.open_group(src._path,mode='r',use_consolidated=False)
            attrs=dict(zg['genotype'].attrs) if 'genotype' in zg else dict(zg.attrs)
            h.update(json.dumps(attrs,sort_keys=True,default=str).encode())
    prefix=getattr(getattr(src,"reader",None),"prefix",getattr(src,"prefix",None))
    if prefix is not None:
        for ext in (".bed",".bim",".fam",".pgen",".pvar",".psam"):
            p=Path(str(prefix)+ext)
            if p.exists():
                st=p.stat(); h.update(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _ordered_chromosomes(vm):
    """Return the ordered chromosomes result computed by ``list``.

    Parameters
    ----------
    vm : object (see the cited input contract)
        Input for the operation ``vm['chromosome']``. See
        ``vigor.grm_store._ordered_chromosomes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    return list(dict.fromkeys(vm["chromosome"].astype(str).tolist()))


def _compute_chromosome_grms(src, chrom, *, sample_ids=None, fit_mask=None,
                              include=("additive","dominance"), block_size="auto",
                              min_maf=1e-4, compute_dtype=np.float32,
                              accum_dtype=np.float32, blas_threads="auto", progress=False,
                              maxmem=None, memory_fraction=None, method="vitezica"):
    """Accumulate additive/dominance kernels separately for each chromosome.

    Parameters
    ----------
    src : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Passed to ``src.iter_blocks`` as ``chrom``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    include : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default:
        ``('additive', 'dominance')``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``src.iter_blocks`` as ``block_size``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    compute_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.

    maxmem : int, float, unit string or None
        Working-memory budget for ``block_size='auto'``; None falls back to the source's
        ``maxmem`` and then to ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.
    """
    from .grm_backend import grm_from_blocks
    from .grm_methods import validate_grm_method
    from .resources import resolve_variant_chunk
    method, components = validate_grm_method(method, include)
    source_ids = np.asarray(src.sample_ids, dtype=str)
    out_ids = source_ids if sample_ids is None else np.asarray(sample_ids, dtype=str)
    if not pd.Index(source_ids).is_unique or not len(out_ids) or not pd.Index(out_ids).is_unique:
        raise ValueError("source and requested sample identifiers must be unique and nonempty")
    idx = pd.Index(source_ids).get_indexer(out_ids)
    if np.any(idx < 0):
        raise KeyError(f"genotype source is missing IID(s): {out_ids[idx < 0][:5].tolist()}")
    n = len(out_ids)
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if fit.shape != (n,) or fit.sum() < 2:
        raise ValueError("fit_mask must select at least two requested animals")
    arithmetic = np.dtype(accum_dtype)
    block_size = resolve_variant_chunk(block_size, n,
        bytes_per_call=80+12*max(0, arithmetic.itemsize-4),
        maxmem=getattr(src, "maxmem", None) if maxmem is None else maxmem,
        memory_fraction=memory_fraction,
        reserve_bytes=n*n*len(components)*(2*arithmetic.itemsize+
            (12 if method in {"npplink", "gcta0"} else 0)))
    iterator = src.iter_blocks(samples=out_ids, chrom=str(chrom), block_size=block_size)
    from .progress import StageProgress
    m = int((src.variant_metadata.chromosome.astype(str) == str(chrom)).sum())
    bar = StageProgress(total=(m+block_size-1)//block_size,
                        desc=f"chr{chrom} A+D GRMs", enabled=progress, unit="block")
    def blocks():
        for block in iterator:
            yield block.data
            bar.update(1)
    try:
        kernels, weights, diagnostics = grm_from_blocks(blocks(), n,
            method=method, components=components, fit_mask=fit,
            min_maf=min_maf, dtype=arithmetic, output_dtype=arithmetic,
            blas_threads=blas_threads)
    finally:
        bar.close()
    diagnostics["chromosome"] = str(chrom)
    return {c: (kernels[c], weights[c]) for c in components}, diagnostics



def _ensure_grm_schema(store, chromosomes, sample_ids, *, chunks=(1,1024,1024), dtype=np.float32,
                       components=("additive","dominance"), signature=None, overwrite=False,
                       method="vitezica", n_autosomes=20):
    """Validate/coerce genomic relationship matrix schema.

    Parameters
    ----------
    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Passed to ``_create_array`` as ``dtype``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Passed to ``_create_array`` as ``dtype``.

    chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``(1, 1024, 1024)``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``_create_array`` as ``dtype``.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive', 'dominance')``.

    signature : object (see the cited input contract)
        Input for the operation ``oldsig != signature``. See
        ``vigor.grm_store._ensure_grm_schema`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``False``.
    """
    from .grm_methods import validate_grm_method
    from .grm_api import _chromosome_class
    method, components = validate_grm_method(method, components)
    pairwise = method in {"npplink", "gcta0"}
    root = store.group
    gg = root.require_group("grm")
    if "chromosome" in gg and overwrite:
        del gg["chromosome"]
    cg = gg.require_group("chromosome")
    chromosomes = np.asarray(chromosomes, dtype=str)
    sample_ids = np.asarray(sample_ids, dtype=str)
    if not len(chromosomes) or not len(sample_ids) or not pd.Index(chromosomes).is_unique or not pd.Index(sample_ids).is_unique:
        raise ValueError("chromosome/sample coordinates must be nonempty and unique")
    n, c = len(sample_ids), len(chromosomes)
    c0, c1 = min(max(1, int(chunks[1])), n), min(max(1, int(chunks[2])), n)
    if "chromosome" in cg:
        if cg.attrs.get("grm_schema") != GRM_SCHEMA_VERSION:
            raise ValueError("legacy GRM cache requires overwrite=True to rebuild the canonical schema")
        if (cg.attrs.get("scientific_signature") != signature or
                cg.attrs.get("grm_method") != method or
                not np.array_equal(cg["chromosome"][:], chromosomes) or
                not np.array_equal(cg["sample_i"][:], sample_ids) or
                not np.array_equal(cg["sample_j"][:], sample_ids)):
            raise ValueError("existing chromosome GRM cache has incompatible signature or coordinates")
    else:
        for key, values, dims in (("chromosome", chromosomes, ("chromosome",)),
                                  ("sample_i", sample_ids, ("sample_i",)),
                                  ("sample_j", sample_ids, ("sample_j",))):
            a = _create_array(cg, key, shape=values.shape, chunks=(min(len(values), 4096),),
                dtype=_unicode_dtype(values), dimension_names=dims)
            a[:] = values
        _create_array(cg, "n_variants", shape=(c,), chunks=(c,), dtype=np.int64,
                      fill_value=0, dimension_names=("chromosome",))
        autosomal = _create_array(cg, "autosome", shape=(c,), chunks=(c,), dtype=bool,
                      fill_value=False, dimension_names=("chromosome",))
        autosomal[:] = [_chromosome_class(ch, n_autosomes) == "autosome" for ch in chromosomes]
        dosage = _create_array(cg, "dosage_status", shape=(c,), chunks=(c,),
                      dtype="<U48", dimension_names=("chromosome",))
        dosage[:] = ["weighted_autosomal_total" if ch == "All" else "not_recoded" for ch in chromosomes]
    for comp in components:
        if comp not in cg:
            _create_array(cg, comp, shape=(c,n,n), chunks=(1,c0,c1), dtype=dtype,
                fill_value=np.nan, dimension_names=("chromosome","sample_i","sample_j"))
            _create_array(cg, f"complete_{comp}", shape=(c,), chunks=(c,), dtype=bool,
                fill_value=False, dimension_names=("chromosome",))
            if pairwise:
                for key, dt in (("weight_"+comp, np.float32), ("numerator_"+comp, dtype),
                                ("pairwise_observed_counts_"+comp, np.float32)):
                    _create_array(cg, key, shape=(c,n,n), chunks=(1,c0,c1), dtype=dt,
                        fill_value=0, dimension_names=("chromosome","sample_i","sample_j"))
            else:
                _create_array(cg, "weight_"+comp, shape=(c,), chunks=(c,), dtype=np.float64,
                    fill_value=0, dimension_names=("chromosome",))
    cg.attrs.update(grm_schema=GRM_SCHEMA_VERSION, grm_method=method,
        grm_complete=False, n_autosomes=int(n_autosomes), scientific_signature=signature,
        normalization="pairwise_component_eligible" if pairwise else "global_HWE",
        created_or_updated_unix=float(time.time()))
    return cg



def build_grms(source, *, store, components=("additive","dominance"), by_chromosome=True,
               chromosomes=None, sample_ids=None, fit_mask=None, block_size="auto",
               min_maf=1e-4, compute_dtype=np.float32, accum_dtype=np.float32,
               output_dtype=np.float32, grm_chunks=(1,1024,1024), blas_threads="auto",
               progress=True, reuse=True, overwrite=False, persist_genomewide=True,
               maxmem=None, memory_fraction=None, save_path=None,
               method="vitezica", n_autosomes=20):
    """Build chromosome GRMs with the canonical GRMs() dataset schema.

    Supports npplink/gcta0 pairwise and van/vitezica global normalization.
    Named chromosomes and an autosomal All aggregate share one dataset; raw
    pairwise numerators/counts are retained so LOCO applies n-1 only once.
    The dataset lives under grm/chromosome in a VigorStore envelope; standalone
    GRMs() uses the same variable/coordinate schema at its root. Legacy cached
    layouts remain readable but require overwrite=True for canonical rebuilding.
    Time is O(n²m), storage O(c*n²), and working memory O(n²+n*block_size).


    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive', 'dominance')``. Passed to ``_ensure_grm_schema`` as ``components``.

    by_chromosome : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``_compute_chromosome_grms`` as ``block_size``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``_compute_chromosome_grms`` as ``min_maf``.

    compute_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``_compute_chromosome_grms`` as
        ``compute_dtype``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``_compute_chromosome_grms`` as
        ``accum_dtype``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``_ensure_grm_schema`` as
        ``dtype``.

    grm_chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``(1, 1024, 1024)``. Passed to ``_ensure_grm_schema`` as
        ``chunks``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``_compute_chromosome_grms`` as ``blas_threads``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``True``.

    reuse : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``False``. Passed to
        ``_ensure_grm_schema`` as ``overwrite``.

    persist_genomewide : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    maxmem : int, float, unit string or None
        Working-memory budget for ``block_size='auto'``; see ``resources.working_memory_bytes``.
        Default: ``None``. Passed to ``_compute_chromosome_grms``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    save_path : str or os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    method : str
        npplink (empirical sample SD, pairwise n-1), gcta0 (HWE, pairwise n),
        van or vitezica (global HWE denominators). Default vitezica.
    n_autosomes : int
        Number of autosomes used to classify chromosome labels for All/LOCO.
        Default 20 for rat; sex chromosomes are retained as separate rows.

    Returns
    -------
    xarray.Dataset
        Build summary (the GRMs themselves live in ``store``). Data variable ``sample_ids``
        (sample). ``attrs`` hold ``store`` (the open VigorStore; in memory, saved as repr),
        ``chromosomes`` (processed order), ``scientific_signature`` (hash of source, samples,
        fit mask, MAF filter and compute/accum/output dtypes) and ``diagnostics`` (one mapping
        per chromosome computed in this call).
    """
    from .results import maybe_save, result_dataset
    from .grm_methods import validate_grm_method
    method, components = validate_grm_method(method, components)
    if isinstance(n_autosomes, bool) or int(n_autosomes) != n_autosomes or n_autosomes < 1:
        raise ValueError("n_autosomes must be a positive integer")
    pairwise = method in {"npplink", "gcta0"}
    if not by_chromosome: raise ValueError("persistent GRM builder currently requires by_chromosome=True")
    st=store if isinstance(store,VigorStore) else open_store(store,mode="a",lazy=False)
    src=st.genotypes() if source is None else (open_genotypes(source) if not hasattr(source,"iter_blocks") else source)
    owned=source is None or not hasattr(source,'iter_blocks')
    try:
        ids=np.asarray(src.sample_ids,dtype=str) if sample_ids is None else np.asarray(sample_ids,dtype=str)
        fit=np.ones(len(ids),bool) if fit_mask is None else np.asarray(fit_mask,bool)
        if fit.shape!=(len(ids),): raise ValueError("fit_mask shape mismatch")
        chroms=_ordered_chromosomes(src.variant_metadata) if chromosomes is None else [str(x) for x in chromosomes]
        if "All" in chroms:
            raise ValueError("All is a reserved synthetic chromosome label")
        store_chroms = chroms+["All"]
        base_sig={
            "method":method,"n_autosomes":int(n_autosomes),"schema":GRM_SCHEMA_VERSION,
            "source":_source_signature(src),"samples":sample_hash(ids),"fit_mask":array_hash(fit.astype(np.uint8)),
            "min_maf":float(min_maf),"compute_dtype":np.dtype(compute_dtype).name,"accum_dtype":np.dtype(accum_dtype).name,"output_dtype":np.dtype(output_dtype).name,"normalization":"vitezica_global_variant_denominator",
        }
        sig=hashlib.sha256(json.dumps(base_sig,sort_keys=True).encode()).hexdigest()
        cg=_ensure_grm_schema(st,store_chroms,ids,chunks=grm_chunks,dtype=output_dtype,
            components=components,signature=sig,overwrite=overwrite,method=method,n_autosomes=n_autosomes)
        st.mark_writing()
        iterator=enumerate(chroms)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator=tqdm(list(iterator),desc="Chromosome A+D GRMs",unit="chr")
            except Exception: iterator=enumerate(chroms)
        diagnostics=[]
        for ci,chrom in iterator:
            needed=[comp for comp in components if not (reuse and bool(cg[f"complete_{comp}"][ci]))]
            if not needed: continue
            out,diag=_compute_chromosome_grms(src,chrom,sample_ids=ids,fit_mask=fit,include=needed,
                                              block_size=block_size,min_maf=min_maf,compute_dtype=compute_dtype,
                                              accum_dtype=accum_dtype,blas_threads=blas_threads,progress=False,
                                              maxmem=maxmem,memory_fraction=memory_fraction,method=method)
            cg["n_variants"][ci]=diag["n_variants_used"]
            for comp,(K,w) in out.items():
                cg[comp][ci,:,:]=np.asarray(K,dtype=output_dtype)
                if pairwise:
                    cg[f"weight_{comp}"][ci,:,:] = w
                    cg[f"numerator_{comp}"][ci,:,:] = diag["numerators"][comp]
                    cg[f"pairwise_observed_counts_{comp}"][ci,:,:] = diag["pairwise_observed_counts_by_component"][comp]
                else:
                    cg[f"weight_{comp}"][ci] = float(w)
                cg[f"complete_{comp}"][ci] = True
            diagnostics.append({key: value for key, value in diag.items()
                if key not in {"numerators", "pairwise_observed_counts_by_component", "pairwise_observed_counts"}})
        # Store All in the same chromosome schema used by GRMs(). Aggregate
        # only autosomes and normalize pairwise counts after summing raw moments.
        auto = np.flatnonzero(np.asarray(cg["autosome"][:], bool))
        if not len(auto):
            raise ValueError("no autosomal chromosomes remain for the All GRM")
        all_index = len(chroms)
        cg["n_variants"][all_index] = sum(int(cg["n_variants"][int(i)]) for i in auto)
        for comp in components:
            if not np.asarray(cg[f"complete_{comp}"][:], bool)[auto].all():
                raise ValueError(f"incomplete autosomal {comp} GRM")
            op = ChromosomeStatisticsGRMOperator(cg, comp, auto, dtype=output_dtype,
                                                row_chunk=max(1, int(grm_chunks[1])))
            for start in range(0, len(ids), op.row_chunk):
                stop = min(start+op.row_chunk, len(ids))
                cg[comp][all_index,start:stop,:] = op.row_block(start, stop)
                if pairwise:
                    numerator = np.zeros((stop-start,len(ids)), dtype=np.float64)
                    counts = np.zeros_like(numerator)
                    for ci in auto:
                        numerator += np.asarray(cg[f"numerator_{comp}"][int(ci),start:stop,:],float)
                        counts += np.asarray(cg[f"pairwise_observed_counts_{comp}"][int(ci),start:stop,:],float)
                    if np.any(counts > 2**24):
                        raise ValueError("pairwise float32 counts support at most 2**24 variants")
                    cg[f"numerator_{comp}"][all_index,start:stop,:] = numerator
                    cg[f"pairwise_observed_counts_{comp}"][all_index,start:stop,:] = counts
                    cg[f"weight_{comp}"][all_index,start:stop,:] = np.maximum(counts-1,0) if method=="npplink" else counts
            if not pairwise:
                cg[f"weight_{comp}"][all_index] = op.den
            cg[f"complete_{comp}"][all_index] = True
        cg.attrs["grm_complete"] = True
        if persist_genomewide:
            for comp in components: merge_chromosome_grms(st,component=comp,persist=True,dtype=output_dtype)
        st.mark_complete(grm_scientific_signature=sig)
        ds=result_dataset({"store":st,"chromosomes":chroms,"sample_ids":ids,"scientific_signature":sig,"diagnostics":diagnostics},
                          dims={"sample_ids":("sample",)},
                          var_attrs={"sample_ids":{"long_name":"GRM sample identifiers",
                                                   "description":"row/column order of every stored GRM"}},
                          title="Chromosome GRM build",software="vigor.grm_store.build_grms",
                          parameters={"components":",".join(map(str,components)),"min_maf":float(min_maf),
                                      "compute_dtype":np.dtype(compute_dtype).name,"accum_dtype":np.dtype(accum_dtype).name,
                                      "output_dtype":np.dtype(output_dtype).name,"block_size":str(block_size),
                                      "reuse":bool(reuse),"overwrite":bool(overwrite),"persist_genomewide":bool(persist_genomewide),
                                      "fit_mask_used":fit_mask is not None})
        return maybe_save(ds,save_path)
    finally:
        if owned and hasattr(src,'close'):src.close()


class ZarrArrayKernelOperator:
    """Chunked kernel operator over a persisted 2-D or chromosome slice Zarr array."""
    def __init__(self, array, *, chromosome_index=None, row_chunk=1024, dtype=np.float32):
        """Initialize ZarrArrayKernelOperator with the inputs and resource settings below.

        Parameters
        ----------
        array : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        chromosome_index : int or None
            Index of the chromosome term in the stored chromosome-kernel stack. Exclude identifies
            the component omitted in a LOCO sum; None leaves all chromosome terms included where
            supported. Default: ``None``.

        row_chunk : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``1024``.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``.
        """
        self.array=array; self.chromosome_index=chromosome_index; self.row_chunk=int(row_chunk); self.dtype=np.dtype(dtype)
        shape=array.shape[-2:]; self.shape=(int(shape[0]),int(shape[1]))
        if self.shape[0]!=self.shape[1]: raise ValueError("GRM must be square")
    def _rows(self,a,b):
        """Resolve requested kernel rows from slice/integer indexing.

        Parameters
        ----------
        a : array-like
            Numerical operand in ``self.array[a:b, :]``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.

        b : array-like
            Numerical operand in ``self.array[a:b, :]``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.
        """
        if self.chromosome_index is None:return np.asarray(self.array[a:b,:],dtype=self.dtype)
        return np.asarray(self.array[int(self.chromosome_index),a:b,:],dtype=self.dtype)
    def matmat(self,X):
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support. Passed to ``np.empty`` as ``dtype``.

        Returns
        -------
        object
            Computed result from ``out[:, 0] if one else out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X=np.asarray(X); one=X.ndim==1
        if one:X=X[:,None]
        out=np.empty((self.shape[0],X.shape[1]),dtype=np.result_type(X.dtype,self.dtype))
        for a in range(0,self.shape[0],self.row_chunk):
            b=min(a+self.row_chunk,self.shape[0]); out[a:b]=self._rows(a,b)@X
        return out[:,0] if one else out
    def diag(self):
        """Return the diagonal variances of the represented covariance/kernel.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        out=np.empty(self.shape[0],dtype=self.dtype)
        for a in range(0,self.shape[0],self.row_chunk):
            b=min(a+self.row_chunk,self.shape[0]); B=self._rows(a,b); out[a:b]=B[np.arange(b-a),np.arange(a,b)]
        return out
    def row_block(self,start,stop): 
        """Return the row block result computed by ``self._rows``.

        Parameters
        ----------
        start : int or None
            Boundary index or genomic coordinate for the requested interval. Use the coordinate
            convention of this reader/query; None leaves that boundary unrestricted where supported.

        stop : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        Returns
        -------
        object
            Computed result from ``self._rows(int(start), int(stop))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self._rows(int(start),int(stop))
    def submatrix(self,rows):
        """Retrieve the requested animal-index square/cross kernel block.

        Parameters
        ----------
        rows : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Returns
        -------
        object
            Computed result from ``full[np.ix_(rows, rows)]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        rows=np.asarray(rows,dtype=int); full=self.row_block(0,self.shape[0]); return full[np.ix_(rows,rows)]
    def aslinearoperator(self):
        """Return the aslinearoperator result computed by ``LinearOperator``.

        Returns
        -------
        object
            Computed result from ``LinearOperator(self.shape, matvec=lambda x: self.matmat(x), matmat=self.matmat, dtype=self.dtype)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return LinearOperator(self.shape,matvec=lambda x:self.matmat(x),matmat=self.matmat,dtype=self.dtype)


class WeightedChromosomeGRMOperator:
    """Exact weighted genome-wide/LOCO operator from chromosome GRMs."""
    def __init__(self,array,weights,*,exclude=None,row_chunk=1024,dtype=np.float32):
        """Initialize WeightedChromosomeGRMOperator with the inputs and resource settings below.

        Parameters
        ----------
        array : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        weights : array-like or int
            Weights or multiplicities used in this accumulation/resampling step. Align vector
            entries with the indexed sample/variant axis; scalar counts specify the number of
            consecutive items.

        exclude : int or None
            Index of the chromosome term in the stored chromosome-kernel stack. Exclude identifies
            the component omitted in a LOCO sum; None leaves all chromosome terms included where
            supported. Default: ``None``.

        row_chunk : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``1024``.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``.
        """
        self.array=array; self.weights=np.asarray(weights,float); self.exclude=None if exclude is None else int(exclude)
        self.row_chunk=int(row_chunk); self.dtype=np.dtype(dtype); n=int(array.shape[-1]); self.shape=(n,n)
        mask=np.ones(len(self.weights),bool)
        if self.exclude is not None: mask[self.exclude]=False
        self.keep=np.flatnonzero(mask & (self.weights>0)); self.den=float(self.weights[self.keep].sum())
        if self.den<=0: raise ValueError("no positive chromosome GRM weights remain")
    def matmat(self,X):
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support. Passed to ``np.empty`` as ``dtype``.

        Returns
        -------
        object
            Computed result from ``out[:, 0] if one else out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X=np.asarray(X);one=X.ndim==1
        if one:X=X[:,None]
        out=np.empty((self.shape[0],X.shape[1]),dtype=np.result_type(X.dtype,self.dtype))
        for a in range(0,self.shape[0],self.row_chunk):
            b=min(a+self.row_chunk,self.shape[0]); R=np.zeros((b-a,X.shape[1]),dtype=np.float64)
            for ci in self.keep:R += self.weights[ci]*(np.asarray(self.array[int(ci),a:b,:],dtype=self.dtype)@X)
            out[a:b]=R/self.den
        return out[:,0] if one else out
    def diag(self):
        """Return the diagonal variances of the represented covariance/kernel.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        out=np.empty(self.shape[0],float)
        for a in range(0,self.shape[0],self.row_chunk):
            b=min(a+self.row_chunk,self.shape[0]); d=np.zeros(b-a,float)
            cols=np.arange(a,b)
            for ci in self.keep:
                B=np.asarray(self.array[int(ci),a:b,:],dtype=self.dtype); d += self.weights[ci]*B[np.arange(b-a),cols]
            out[a:b]=d/self.den
        return out
    def row_block(self,start,stop):
        """Retrieve the requested contiguous animal rows of the represented kernel.

        Parameters
        ----------
        start : int or None
            Boundary index or genomic coordinate for the requested interval. Use the coordinate
            convention of this reader/query; None leaves that boundary unrestricted where supported.

        stop : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        Returns
        -------
        object
            Computed result from ``R / self.den``; see the summary and the called implementation for its labeled/numerical representation.
        """
        start,stop=int(start),int(stop); R=np.zeros((stop-start,self.shape[1]),float)
        for ci in self.keep:R += self.weights[ci]*np.asarray(self.array[int(ci),start:stop,:],dtype=self.dtype)
        return R/self.den
    def submatrix(self,rows):
        """Retrieve the requested animal-index square/cross kernel block.

        Parameters
        ----------
        rows : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Returns
        -------
        object
            Computed result from ``self.row_block(0, self.shape[0])[np.ix_(rows, rows)]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        rows=np.asarray(rows,dtype=int); return self.row_block(0,self.shape[0])[np.ix_(rows,rows)]
    def aslinearoperator(self): 
        """Return the aslinearoperator result computed by ``LinearOperator``.

        Returns
        -------
        object
            Computed result from ``LinearOperator(self.shape, matvec=lambda x: self.matmat(x), matmat=self.matmat, dtype=self.dtype)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return LinearOperator(self.shape,matvec=lambda x:self.matmat(x),matmat=self.matmat,dtype=self.dtype)



class ChromosomeStatisticsGRMOperator:
    """Read exact chromosome sums from the shared GRM dataset schema.

    Parameters are a chromosome Zarr group, a component, selected chromosome
    indices, an output floating dtype and a positive row chunk. Global methods
    aggregate weighted kernels; pairwise methods aggregate raw numerators and
    counts before applying n or n-1 once. Construction uses O(c+n) metadata;
    each row block uses O(row_chunk*n) scratch and O(c*row_chunk*n) I/O.
    """
    def __init__(self, group, component, indices, *, dtype=np.float32, row_chunk=1024):
        from .grm_methods import validate_grm_method
        self.group = group
        self.method, _ = validate_grm_method(group.attrs.get("grm_method", "vitezica"), (component,))
        self.component = component
        self.keep = np.asarray(indices, dtype=np.int64)
        self.dtype = np.dtype(dtype)
        self.row_chunk = int(row_chunk)
        if self.row_chunk < 1 or self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("row_chunk must be positive and dtype floating32/64")
        n = int(group[component].shape[-1]); self.shape = (n, n)
        if not len(self.keep):
            raise ValueError("no chromosomes remain for the requested GRM")
        if self.method in {"npplink", "gcta0"}:
            for key in ("numerator_"+component, "pairwise_observed_counts_"+component):
                if key not in group or group[key].shape != group[component].shape:
                    raise ValueError(f"pairwise GRM store requires aligned raw statistic {key}")
        else:
            weights = group["weight_"+component]
            if len(weights.shape) != 1:
                raise ValueError("global GRM method requires one scalar weight per chromosome")
            self.weights = np.asarray(weights[:], dtype=np.float64)
            if not np.isfinite(self.weights[self.keep]).all() or np.any(self.weights[self.keep] < 0):
                raise ValueError("GRM weights must be finite and nonnegative")
            self.den = float(self.weights[self.keep].sum())
            if self.den <= 0:
                raise ValueError("no positive chromosome GRM weight remains")

    def row_block(self, start, stop):
        """Return [start:stop, :] with exact chromosome normalization."""
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= self.shape[0]:
            raise ValueError("invalid row interval")
        out = np.zeros((stop-start, self.shape[1]), dtype=np.float64)
        if self.method in {"npplink", "gcta0"}:
            counts = np.zeros_like(out)
            for ci in self.keep:
                out += np.asarray(self.group["numerator_"+self.component][int(ci), start:stop, :], dtype=np.float64)
                counts += np.asarray(self.group["pairwise_observed_counts_"+self.component][int(ci), start:stop, :], dtype=np.float64)
            if not np.isfinite(counts).all() or np.any(counts < 0):
                raise ValueError("stored pairwise counts must be finite and nonnegative")
            denominator = counts-1 if self.method == "npplink" else counts
            valid = denominator > 0
            np.divide(out, denominator, out=out, where=valid)
            out[~valid] = np.nan
        else:
            for ci in self.keep:
                out += self.weights[ci]*np.asarray(self.group[self.component][int(ci), start:stop, :], dtype=np.float64)
            out /= self.den
        return out.astype(self.dtype, copy=False)

    def matmat(self, X):
        """Apply bounded row blocks to one or multiple sample right-hand sides."""
        X = np.asarray(X)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        if X.ndim != 2 or X.shape[0] != self.shape[1]:
            raise ValueError("right-hand side does not match the GRM sample axis")
        out = np.empty((self.shape[0], X.shape[1]), dtype=np.result_type(self.dtype, X.dtype))
        for start in range(0, self.shape[0], self.row_chunk):
            stop = min(start+self.row_chunk, self.shape[0])
            out[start:stop] = self.row_block(start, stop) @ X
        return out[:, 0] if one else out

    def diag(self):
        """Return the diagonal with bounded sample-row scratch."""
        out = np.empty(self.shape[0], dtype=self.dtype)
        for start in range(0, self.shape[0], self.row_chunk):
            stop = min(start+self.row_chunk, self.shape[0])
            block = self.row_block(start, stop)
            out[start:stop] = block[np.arange(stop-start), np.arange(start, stop)]
        return out

    def submatrix(self, rows):
        """Return one selected sample square without materializing unrelated rows."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), len(rows)), dtype=self.dtype)
        for i, row in enumerate(rows):
            out[i] = self.row_block(row, row+1)[0, rows]
        return out

    def aslinearoperator(self):
        """Expose the exact operator to SciPy iterative solvers."""
        return LinearOperator(self.shape, matvec=self.matmat, matmat=self.matmat, dtype=self.dtype)


def _open_standalone_grm(group, *, component, chromosome, loco, representation,
                         materialize, dtype, row_chunk, labeled):
    """Read a root-level chromosome dataset using the shared statistics schema."""
    if representation not in {"auto", "dense", "operator"}:
        raise ValueError("standalone chromosome stores support exact dense/operator representations")
    if group.attrs.get("grm_complete") is False:
        raise ValueError("chromosome GRM store is incomplete")
    if labeled and not materialize:
        raise ValueError("labeled=True requires materialize=True")
    names = np.asarray(group["chromosome"][:], dtype=str)
    rows = np.asarray(group["sample_i"][:], dtype=str)
    cols = np.asarray(group["sample_j"][:], dtype=str)
    if (not pd.Index(names).is_unique or not pd.Index(rows).is_unique or
            not np.array_equal(rows, cols) or group[component].shape != (len(names), len(rows), len(cols))):
        raise ValueError("stored chromosome GRM coordinates are not unique aligned square labels")
    if loco is not None or str(chromosome).lower() == "all":
        if loco is None and "All" in names:
            indices = np.flatnonzero(names == "All")
        else:
            if "autosome" in group:
                keep = np.asarray(group["autosome"][:], dtype=bool)
            else:
                from .grm_api import _chromosome_class
                keep = np.asarray([_chromosome_class(c, group.attrs.get("n_autosomes", 20)) == "autosome" for c in names])
            if loco is not None:
                if str(loco) not in names:
                    raise KeyError(f"chromosome {loco!r} absent from GRM store")
                keep &= names != str(loco)
            indices = np.flatnonzero(keep)
    else:
        indices = np.flatnonzero(names == str(chromosome))
        if len(indices) != 1:
            raise KeyError(f"chromosome {chromosome!r} absent from GRM store")
    if f"complete_{component}" in group and not np.asarray(group[f"complete_{component}"][:], bool)[indices].all():
        raise ValueError("requested chromosome GRMs are incomplete")
    dt = np.dtype(group[component].dtype if dtype is None else dtype)
    op = ChromosomeStatisticsGRMOperator(group, component, indices, dtype=dt, row_chunk=row_chunk)
    if not materialize:
        return op
    values = op.row_block(0, len(rows))
    if not labeled:
        return values
    import xarray as xr
    return xr.DataArray(values, dims=("sample_i", "sample_j"),
                        coords={"sample_i": rows, "sample_j": cols}, name=component)


def _store_obj(store): 
    """Normalize a store path or opened store into the persistence interface.

    Parameters
    ----------
    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.
    """
    return store if isinstance(store,VigorStore) else open_store(store,mode="a",lazy=False)


def merge_chromosome_grms(store, *, component="additive", persist=True, dtype=np.float32, row_chunk=512):
    """Merge chromosome grms.

    Parameters
    ----------
    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``.

    persist : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to
        ``WeightedChromosomeGRMOperator`` as ``dtype``.

    row_chunk : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``512``.
        Passed to ``WeightedChromosomeGRMOperator`` as ``row_chunk``.

    Returns
    -------
    object
        Computed result from ``ZarrArrayKernelOperator(out, row_chunk=row_chunk, dtype=dtype)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    st=_store_obj(store); cg=st.group["grm/chromosome"]; arr=cg[component]
    complete=np.asarray(cg[f"complete_{component}"][:],bool)
    if not np.all(complete): raise ValueError(f"not all chromosome {component} GRMs are complete")
    canonical = cg.attrs.get("grm_schema") == GRM_SCHEMA_VERSION
    if canonical:
        names = np.asarray(cg["chromosome"][:], dtype=str)
        indices = np.flatnonzero(names == "All")
        if len(indices) != 1:
            raise ValueError("canonical chromosome store requires one All aggregate")
        op = ChromosomeStatisticsGRMOperator(cg, component, indices,
                                             row_chunk=row_chunk, dtype=dtype)
    else:
        w=np.asarray(cg[f"weight_{component}"][:],float)
        op=WeightedChromosomeGRMOperator(arr,w,row_chunk=row_chunk,dtype=dtype)
    if not persist:return op
    gg=st.group["grm"].require_group("genomewide"); n=op.shape[0]; c=min(int(row_chunk),n)
    for dim in ('sample_i','sample_j'):
        coordinate=_create_array(gg,dim,shape=(n,),chunks=(min(n,4096),),dtype=cg[dim].dtype,dimension_names=(dim,))
        coordinate[:]=cg[dim][:]
    out=_create_array(gg,component,shape=(n,n),chunks=(c,c),dtype=dtype,fill_value=np.nan,
                      dimension_names=("sample_i","sample_j"),overwrite=True)
    for a in range(0,n,c):
        b=min(a+c,n); out[a:b,:]=op.row_block(a,b).astype(dtype,copy=False)
    if not canonical:
        gg.attrs[f"weight_{component}"] = float(w.sum())
    gg.attrs[f"source_signature_{component}"] = cg.attrs.get("scientific_signature")
    return ZarrArrayKernelOperator(out,row_chunk=row_chunk,dtype=dtype)


def _chrom_index(cg, chrom):
    """Return the chrom index result computed by ``int``.

    Parameters
    ----------
    cg : object (see the cited input contract)
        Input for the operation ``cg['chromosome'][:]``. See ``vigor.grm_store._chrom_index``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    vals=np.asarray(cg["chromosome"][:],dtype=str); hit=np.flatnonzero(vals==str(chrom))
    if len(hit)!=1: raise KeyError(f"chromosome {chrom!r} is absent or ambiguous")
    return int(hit[0])


def cache_grm_spectral(store, *, component="additive", chromosome="all", rank=256,
                       random_state=0, dtype=np.float32, probe_vectors=4):
    """Cache genomic relationship matrix spectral.

    Parameters
    ----------
    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``. Passed to ``open_grm`` as ``component``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``'all'``. Passed to ``open_grm`` as
        ``chromosome``.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``256``. Passed to
        ``LowRankKernelOperator.from_dense`` as ``rank``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``LowRankKernelOperator.from_dense`` as ``random_state``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to
        ``LowRankKernelOperator.from_dense`` as ``dtype``.

    probe_vectors : array-like
        Sparse/sketch indexing or random-probe values used by the operator. CSR indptr indexes
        row offsets, sign supplies signed sketch weights and probes have the operator's leading
        dimension; preserve the matching index layout. Default: ``4``. Passed to
        ``LowRankKernelOperator.from_dense`` as ``probe_vectors``.

    Returns
    -------
    object
        Computed result from ``op``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .multivariate import LowRankKernelOperator
    st=_store_obj(store)
    K=open_grm(st,component=component,chromosome=chromosome,materialize=True,representation="dense",dtype=np.float64)
    op=LowRankKernelOperator.from_dense(K,rank=rank,dtype=dtype,random_state=random_state,probe_vectors=probe_vectors)
    scope="all" if str(chromosome)=="all" else f"chr{chromosome}"
    sg=st.group["grm"].require_group("spectral").require_group(scope).require_group(component)
    U=_create_array(sg,"vectors",shape=op.U.shape,chunks=(min(op.U.shape[0],4096),min(op.U.shape[1],256)),dtype=dtype,
                    dimension_names=("sample","rank"),overwrite=True); U[:]=op.U
    v=_create_array(sg,"values",shape=op.values.shape,chunks=(min(len(op.values),4096),),dtype=dtype,
                    dimension_names=("rank",),overwrite=True); v[:]=op.values
    sg.attrs.update({"bulk":float(op.bulk),"rank":int(op.U.shape[1]),
                     "diagnostics":json.dumps(op.diagnostics),
                     "source_signature":st.group["grm/chromosome"].attrs.get("scientific_signature")})
    return op


def open_grm(store, *, component="additive", chromosome="all", loco=None,
             representation="auto", materialize=True, dtype=None, row_chunk=1024,
             labeled=False):
    """Open genomic relationship matrix.

    Parameters
    ----------
    store : VigorStore or Zarr group
        Opened persistent store/group holding the artifact required here. Its schema and
        sample/trait identifiers must match the requesting workflow.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``'all'``.

    loco : object (see the cited input contract)
        Input for the operation ``_chrom_index(cg, loco)``. See ``vigor.grm_store.open_grm`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    representation : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Default: ``'auto'``. Source-recognized literal
        values: ``low_rank``, ``lowrank``, ``spectral``. These select the corresponding branches
        below; comparison evidence alone does not establish an exhaustive enum.

    materialize : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    row_chunk : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``1024``.
        Passed to ``WeightedChromosomeGRMOperator`` as ``row_chunk``.

    labeled : bool
        If True, return an xarray.DataArray with validated stored sample_i and
        sample_j coordinates. Requires materialize=True; default False retains
        the ndarray/operator contract. Labels are never inferred from positions.

    Returns
    -------
    object
        Computed result from ``op.row_block(0, op.shape[0]).astype(dt, copy=False) if materialize else op``; see the summary and the called implementation for its labeled/numerical representation.
    """
    comp = str(component)
    import zarr
    root = store.group if isinstance(store, VigorStore) else (store if hasattr(store, "attrs") and hasattr(store, "array_keys") else zarr.open_group(str(store), mode="r"))
    rep = str(representation).lower()
    if "chromosome" in root and "sample_i" in root and comp in root:
        return _open_standalone_grm(root, component=comp, chromosome=chromosome,
            loco=loco, representation=rep, materialize=materialize,
            dtype=dtype, row_chunk=row_chunk, labeled=labeled)
    if "grm/chromosome" in root:
        cg = root["grm/chromosome"]
        if cg.attrs.get("grm_schema") == GRM_SCHEMA_VERSION and rep not in {"spectral", "lowrank", "low_rank"}:
            return _open_standalone_grm(cg, component=comp, chromosome=chromosome,
                loco=loco, representation=rep, materialize=materialize,
                dtype=dtype, row_chunk=row_chunk, labeled=labeled)
    st=_store_obj(store)
    if labeled and not materialize:
        raise ValueError("labeled=True requires materialize=True")
    def finish(op):
        if not materialize:
            return op
        value = op.row_block(0, op.shape[0]).astype(dt, copy=False)
        if not labeled:
            return value
        import xarray as xr
        cg = st.group["grm/chromosome"]
        rows = np.asarray(cg["sample_i"][:], dtype=str)
        cols = np.asarray(cg["sample_j"][:], dtype=str)
        if (value.shape != (len(rows), len(cols)) or
                not pd.Index(rows).is_unique or not pd.Index(cols).is_unique or
                not np.array_equal(rows, cols)):
            raise ValueError("stored GRM coordinates are not unique aligned square sample labels")
        return xr.DataArray(value, dims=("sample_i", "sample_j"),
                            coords={"sample_i": rows, "sample_j": cols}, name=comp)
    cg = st.group["grm/chromosome"]
    if f"complete_{comp}" in cg:
        complete = np.asarray(cg[f"complete_{comp}"][:], dtype=bool)
        include = np.ones(len(complete), dtype=bool)
        if loco is not None:
            include[_chrom_index(cg, loco)] = False
        elif str(chromosome) != "all":
            include[:] = False
            include[_chrom_index(cg, chromosome)] = True
        if not complete[include].all():
            raise ValueError(f"requested {comp} GRM contains incomplete chromosomes")
    if loco is not None:
        cg=st.group["grm/chromosome"]; ci=_chrom_index(cg,loco); arr=cg[comp]; w=np.asarray(cg[f"weight_{comp}"][:],float)
        dt=np.dtype(arr.dtype if dtype is None else dtype)
        op=WeightedChromosomeGRMOperator(arr,w,exclude=ci,row_chunk=row_chunk,dtype=dt)
        return finish(op)
    scope="all" if str(chromosome)=="all" else f"chr{chromosome}"
    rep=str(representation).lower()
    # Approximation is now opt-in.  A cached spectral representation must not
    # silently change the mathematics or return type of an otherwise exact
    # open_grm(..., representation="auto", materialize=True) call.
    if rep in {"spectral","lowrank","low_rank"}:
        if "spectral" not in st.group["grm"]:
            raise KeyError(f"no cached spectral GRM for {scope}/{comp}")
        specroot=st.group["grm/spectral"]
        if scope not in specroot or comp not in specroot[scope]:
            raise KeyError(f"no cached spectral GRM for {scope}/{comp}")
        from .multivariate import LowRankKernelOperator
        sg=specroot[scope][comp]
        expected = cg.attrs.get("scientific_signature")
        if expected is not None and sg.attrs.get("source_signature") != expected:
            raise ValueError("cached spectral GRM does not match chromosome scientific signature")
        dt=np.dtype(sg["vectors"].dtype if dtype is None else dtype)
        op=LowRankKernelOperator(np.asarray(sg["vectors"][:],dtype=dt),np.asarray(sg["values"][:],dtype=dt),
                                 bulk=float(sg.attrs.get("bulk",0.0)),dtype=dt)
        return finish(op)
    if str(chromosome)=="all":
        grm_group=st.group["grm"]; gg=grm_group["genomewide"] if "genomewide" in grm_group else None
        if gg is not None and comp in gg:
            expected = cg.attrs.get("scientific_signature")
            actual = gg.attrs.get(f"source_signature_{comp}")
            if expected is not None and actual != expected:
                raise ValueError("genomewide GRM cache does not match chromosome scientific signature")
            arr=gg[comp]; dt=np.dtype(arr.dtype if dtype is None else dtype); op=ZarrArrayKernelOperator(arr,row_chunk=row_chunk,dtype=dt)
        else:
            cg=st.group["grm/chromosome"]; dt=np.dtype(cg[comp].dtype if dtype is None else dtype)
            op=merge_chromosome_grms(st,component=comp,persist=False,dtype=dt,row_chunk=row_chunk)
    else:
        cg=st.group["grm/chromosome"]; ci=_chrom_index(cg,chromosome); arr=cg[comp]; dt=np.dtype(arr.dtype if dtype is None else dtype); op=ZarrArrayKernelOperator(arr,chromosome_index=ci,row_chunk=row_chunk,dtype=dt)
    return finish(op)


__all__=["build_grms","merge_chromosome_grms","open_grm","cache_grm_spectral",
         "ZarrArrayKernelOperator","WeightedChromosomeGRMOperator"]


# ---------------------------------------------------------------------------
# VIGOR 2.11 exact no-cache packed-BED GRM modes
# ---------------------------------------------------------------------------
def build_grms_in_memory(source, *, components=('additive','dominance'), sample_ids=None,
                         fit_mask=None, chromosome=None, min_maf=1e-4, block_size='auto',
                         memory_fraction=None, maxmem=None, output_dtype=np.float32,
                         blas_threads='auto', progress=True, method='vitezica'):
    """Build exact A/D GRMs in RAM without creating a GRM store.
    
    Candidate methods npplink/gcta0 use bounded packed float32 ssyrk.
    npplink uses sample SD (ddof=1), component-specific pair counts and
    n_ij-1 normalization; invalid pair denominators produce NaN. maxmem
    reserves numerical matrices and scratch, with metadata/library/I/O RSS
    additional. blas_threads is scoped; oversized explicit blocks are capped.
    See docs/GRM_API_2_11_7.md for the full scientific/resource contract.
    
    
    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.
    
    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive', 'dominance')``. Passed to ``dense_grm_from_bed`` as ``components``.
    
    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``dense_grm_from_bed`` as ``sample_ids``.
    
    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``dense_grm_from_bed`` as ``fit_mask``.
    
    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``. Passed to
        ``dense_grm_from_bed`` as ``chromosome``.
    
    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``dense_grm_from_bed`` as ``min_maf``.
    
    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``dense_grm_from_bed`` as ``chunk_variants``.
    
    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to ``dense_grm_from_bed`` as
        ``memory_fraction``.
    
    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``dense_grm_from_bed`` as
        ``maxmem``.
    
    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``dense_grm_from_bed`` as
        ``output_dtype``.
    
    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``dense_grm_from_bed`` as ``blas_threads``.
    
    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``True``. Passed to ``dense_grm_from_bed`` as
        ``progress``.
    
    method : str
        Estimator identifier: 'npplink' uses sample-SD pairwise n-minus-one normalization; 'gcta0' uses per-marker HWE scaling; 'van' and 'vitezica' use global HWE weights. Default: ``'vitezica'``.
    
    Returns
    -------
    object
        Computed result from ``dense_grm_from_bed(prefix, components=components, sample_ids=sample_ids, fit_mask=fit_mask, chromosome=chromosome, min_maf=min_maf, chunk_variants=block_size, memory_fraction=memory_fraction, maxme...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    prefix=source
    if not isinstance(source,(str,Path)):
        prefix=getattr(getattr(source,'reader',None),'prefix',getattr(source,'prefix',None))
    if prefix is None or not Path(str(prefix)+'.bed').exists():
        raise ValueError('build_grms_in_memory currently requires a PLINK1 BED source/prefix')
    from .packed_bed import dense_grm_from_bed
    return dense_grm_from_bed(prefix,components=components,sample_ids=sample_ids,fit_mask=fit_mask,
        chromosome=chromosome,min_maf=min_maf,chunk_variants=block_size,memory_fraction=memory_fraction,
        maxmem=maxmem,output_dtype=output_dtype,blas_threads=blas_threads,progress=progress,method=method)


def stream_loco_grm(source, genomewide_result, chromosome, *, components=None, sample_ids=None,
                    fit_mask=None, min_maf=1e-4, block_size='auto', memory_fraction=None,
                    maxmem=None, output_dtype=np.float32, blas_threads='auto', progress=False):
    """Exact LOCO from one stored/in-memory genomewide GRM plus one temporary chromosome GRM.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    genomewide_result : array-like or numeric scalar
        Input for the operation ``stream_loco_from_bed(prefix, genomewide_result, chromosome,
        components=components, sample_ids=sample_ids, fit_mask=fit_mask, min_maf=min_maf,
        chunk_variants=block_size, memory_fraction=memory_fraction, maxmem=maxmem,
        output_dtype=output_...``. See ``vigor.grm_store.stream_loco_grm`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``. Passed to ``stream_loco_from_bed`` as ``components``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``stream_loco_from_bed`` as ``sample_ids``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``stream_loco_from_bed`` as ``fit_mask``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``stream_loco_from_bed`` as ``min_maf``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``stream_loco_from_bed`` as ``chunk_variants``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to ``stream_loco_from_bed`` as
        ``memory_fraction``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``stream_loco_from_bed``
        as ``maxmem``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``stream_loco_from_bed`` as
        ``output_dtype``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``stream_loco_from_bed`` as ``blas_threads``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``. Passed to ``stream_loco_from_bed`` as
        ``progress``.

    Returns
    -------
    object
        Computed result from ``stream_loco_from_bed(prefix, genomewide_result, chromosome, components=components, sample_ids=sample_ids, fit_mask=fit_mask, min_maf=min_maf, chunk_variants=block_size, memory_fraction=memory_fract...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    prefix=source
    if not isinstance(source,(str,Path)):
        prefix=getattr(getattr(source,'reader',None),'prefix',getattr(source,'prefix',None))
    if prefix is None: raise ValueError('stream_loco_grm requires a PLINK1 BED source/prefix')
    from .packed_bed import stream_loco_from_bed
    return stream_loco_from_bed(prefix,genomewide_result,chromosome,components=components,
        sample_ids=sample_ids,fit_mask=fit_mask,min_maf=min_maf,chunk_variants=block_size,
        memory_fraction=memory_fraction,maxmem=maxmem,output_dtype=output_dtype,
        blas_threads=blas_threads,progress=progress)
