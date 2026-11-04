"""Zarr v3 genotype ingest and labeled persistent I/O for VIGOR 2.9.

The writer preallocates the final arrays once and fills variant regions. This
avoids the repeated resize/metadata work of append-per-block xarray writes.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import numpy as np

from .genotype_source import open_genotypes, Plink1GenotypeSource, Plink2GenotypeSource
from .store import SCHEMA_VERSION, sample_hash, canonical_provenance, finalize_zarr_artifact


def _require_zarr():
    """Import and verify availability of zarr.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Zarr v3 ingest requires `pip install 'VIGOR[io]'`.") from exc
    return zarr


def _source_fingerprint(src):
    """Return the source fingerprint result computed by ``h.hexdigest``.

    Parameters
    ----------
    src : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.
    """
    h=hashlib.sha256()
    prefix=getattr(getattr(src, "reader", None), "prefix", getattr(src, "prefix", None))
    if prefix is not None:
        for ext in (".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam"):
            p=Path(str(prefix)+ext)
            if p.exists():
                st=p.stat(); h.update(str(p.resolve()).encode()); h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    else:
        h.update(np.asarray(src.sample_ids, dtype=str).astype("U").tobytes())
        vm=getattr(src, "variant_metadata", None)
        if vm is not None: h.update("\n".join(vm.astype(str).agg("|".join, axis=1)).encode())
    return h.hexdigest()


def _unicode_dtype(values, minimum=1):
    """Return the unicode dtype result computed by ``np.dtype``.

    Parameters
    ----------
    values : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    minimum : int
        Input for the operation ``[int(minimum)] + [len(x) for x in values.tolist()]``. See
        ``vigor.zarr_io._unicode_dtype`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape. Default: ``1``.
    """
    values=np.asarray(values, dtype=str)
    width=max([int(minimum)]+[len(x) for x in values.tolist()])
    return np.dtype(f"U{width}")


def _default_chunks(n_samples, n_variants):
    """Choose genotype chunk extents from the sample and variant axis sizes.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Source comparisons: ``n_samples <= 5000``.
        See the rejection/fallback branches for enforcement.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.
    """
    if n_samples <= 5000:
        return (max(1, n_samples), min(16384, max(1, n_variants)))
    return (min(4096, n_samples), min(8192, max(1, n_variants)))


def _create_array(group, name, *, shape, chunks, dtype, fill_value=None, dimension_names=None,
                  overwrite=True, compressor=None, shards=None):
    """Create a Zarr v3 array with dimensions, chunks and optional sharding/compression.

    Parameters
    ----------
    group : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    shape : tuple of int or None
        Requested array/distribution extents in axis order. Nonnegative sizes must agree with
        the associated data; None uses the backend's inferred scalar/default shape.

    chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision.

    fill_value : object (see the cited input contract)
        Input for the operation ``fill_value is not None``. See ``vigor.zarr_io._create_array``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``.

    dimension_names : sequence of str or None
        Ordered names of array dimensions for persistent metadata. Length must equal array rank;
        None omits explicit dimension-name metadata. Default: ``None``.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``True``.

    compressor : codec specification or None
        Compression codec/settings passed to the storage backend. None uses its
        uncompressed/default path; codec support depends on the installed Zarr/Parquet backend.
        Default: ``None``.

    shards : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``None``.
    """
    kwargs=dict(name=name, shape=tuple(map(int, shape)), chunks=tuple(map(int, chunks)), dtype=np.dtype(dtype), overwrite=overwrite)
    if fill_value is not None: kwargs["fill_value"] = fill_value
    if dimension_names is not None: kwargs["dimension_names"] = tuple(dimension_names)
    # zarr v3 names this argument `compressors` in modern releases. Keep a
    # compatibility fallback because VIGOR supports a range of zarr>=3.
    if compressor is not None: kwargs["compressors"] = compressor
    if shards is None and dimension_names is not None and any(d in {'sample','sample_i','sample_j','variant','factor','entry','nonzero'} for d in dimension_names) and len(shape)>1:
        shards=tuple(int(ch)*2 for ch in chunks)
    if shards is not None:
        norm=[]
        for sh,ch,n in zip(shards,chunks,shape):
            ch=max(1,int(ch)); target=min(max(ch,int(sh)),int(n))
            outer=max(ch,(target//ch)*ch); norm.append(int(outer))
        kwargs["shards"] = tuple(norm)
    try:
        arr=group.create_array(**kwargs)
    except TypeError:
        if "compressors" in kwargs:
            kwargs["compressor"] = kwargs.pop("compressors")
        arr=group.create_array(**kwargs)
    # Zarr v2/xarray compatibility metadata is harmless under v3 and makes
    # older xarray readers understand dimensionality.
    if dimension_names is not None: arr.attrs["_ARRAY_DIMENSIONS"] = list(dimension_names)
    return arr


def ingest_genotypes(source, destination, *, source_format="auto", encoding="auto", block_size="auto",
                     chunks=None, shards=None, compressor=None, overwrite=False, progress=False,
                     zarr_format=3, maxmem=None, memory_fraction=None):
    """Convert a canonical genotype source to a transactional Zarr v3 store.

    PLINK hard calls are persisted as int8 ``0/1/2`` with ``-1`` missing by
    default. Other sources are stored as float32 dosage. Numerical kernels
    convert only the current block to floating point.

    Parameters
    ----------
    source : path-like or genotype source
        PLINK1/PLINK2 prefix, supported array/xarray source or an object exposing iter_blocks
        and sample_ids/variant_metadata. A nonempty stable full-sample row set and complete
        ordered variant stream are required. Blocks contain sample × variant calls/dosages with
        NaN missing.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    source_format : str
        Physical genotype backend: 'auto' identifies the source and rejects ambiguous
        PLINK1/PLINK2 prefixes; 'plink1'/'bed'/'bfile' selects BED, 'plink2'/'pgen'/'pfile'
        selects PGEN, and store/array support follows open_genotypes. Zarr uses its canonical
        schema. Default: ``'auto'``. Passed to ``open_genotypes`` as ``source_format``.

    encoding : str
        'auto' uses 'hardcall_int8' for PLINK hardcalls and 'dosage_float32' otherwise.
        'hardcall_int8' stores 0/1/2 and -1 missing, rejecting noninteger finite dosage;
        'dosage_float32' stores dosage with NaN missing. These are the only encoding choices
        after lowercasing. Default: ``'auto'``.

    block_size : int or 'auto'
        Variants read and converted per ingest block (in-memory only; on-disk chunk shapes are
        set by ``chunks``/``shards``). Default: ``'auto'`` (sized from the working-memory budget
        and aligned to the storage chunk width; integers are ceilings). Passed to
        ``src.iter_blocks`` as ``block_size``.

    chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``None``. Passed to ``_create_array`` as ``chunks``.

    shards : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``None``. Passed to ``_create_array`` as ``shards``.

    compressor : codec specification or None
        Compression codec/settings passed to the storage backend. None uses its
        uncompressed/default path; codec support depends on the installed Zarr/Parquet backend.
        Default: ``None``. Passed to ``_create_array`` as ``compressor``.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``False``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.

    zarr_format : int
        Canonical on-disk format version. Only int(zarr_format)==3 is accepted; older Zarr
        format versions raise ValueError. Default: ``3``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    memory_fraction : float or None
        RAM fraction used when maxmem is None; None means 80%. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``str(dest)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if int(zarr_format) != 3: raise ValueError("VIGOR 2.9 canonical stores require zarr_format=3")
    zarr=_require_zarr()
    src=open_genotypes(source, source_format=source_format) if not hasattr(source, "iter_blocks") else source
    dest=Path(destination); tmp=dest.with_name(dest.name+".incomplete")
    if dest.exists() and not overwrite: raise FileExistsError(f"destination already exists: {dest}")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    n=len(src.sample_ids); vm=src.variant_metadata.reset_index(drop=True); m=len(vm)
    if m == 0 or n == 0: raise ValueError("genotype source is empty")
    if chunks is None: chunks=_default_chunks(n,m)
    chunks=(min(int(chunks[0]),n), min(int(chunks[1]),m))
    if shards is not None: shards=(min(int(shards[0]),n), min(int(shards[1]),m))

    enc=str(encoding).lower()
    hard_source=isinstance(src,(Plink1GenotypeSource,Plink2GenotypeSource))
    if enc == "auto": enc="hardcall_int8" if hard_source else "dosage_float32"
    if enc not in {"hardcall_int8","dosage_float32"}: raise ValueError("encoding must be auto, hardcall_int8, or dosage_float32")

    root=zarr.open_group(str(tmp), mode="w", zarr_format=3)
    root.attrs.update(canonical_provenance("genotype_store",write_state="writing",
        domino_version=__import__("vigor").__version__,sample_hash=sample_hash(src.sample_ids),
        source_fingerprint=_source_fingerprint(src)))
    g=root.require_group("genotype")
    samples=np.asarray(src.sample_ids,dtype=str); variants=vm["variant_id"].astype(str).to_numpy()
    chrom=vm.get("chromosome", np.repeat("",m)).astype(str).to_numpy()
    pos=np.asarray(vm.get("position", np.arange(m)+1), dtype=np.int64)

    sa=_create_array(g,"sample",shape=(n,),chunks=(min(n,4096),),dtype=_unicode_dtype(samples),dimension_names=("sample",))
    va=_create_array(g,"variant",shape=(m,),chunks=(min(m,65536),),dtype=_unicode_dtype(variants),dimension_names=("variant",))
    ca=_create_array(g,"chromosome",shape=(m,),chunks=(min(m,65536),),dtype=_unicode_dtype(chrom),dimension_names=("variant",))
    pa=_create_array(g,"position",shape=(m,),chunks=(min(m,65536),),dtype=np.int64,dimension_names=("variant",))
    sa[:]=samples; va[:]=variants; ca[:]=chrom; pa[:]=pos
    sa.attrs["role"]="coordinate"; va.attrs["role"]="coordinate"; ca.attrs["role"]="coordinate"; pa.attrs["role"]="coordinate"

    if enc == "hardcall_int8":
        data=_create_array(g,"hardcall",shape=(n,m),chunks=chunks,dtype=np.int8,fill_value=-1,
                           dimension_names=("sample","variant"),compressor=compressor,shards=shards)
        data.attrs.update({"representation":"hardcall", "missing_value":-1, "coordinates":"chromosome position"})
    else:
        data=_create_array(g,"dosage",shape=(n,m),chunks=chunks,dtype=np.float32,fill_value=np.nan,
                           dimension_names=("sample","variant"),compressor=compressor,shards=shards)
        data.attrs.update({"representation":"dosage", "coordinates":"chromosome position"})

    from .resources import resolve_variant_chunk
    # Read block: decode (<=float64 8 B) + finite/isin/bad masks (3 B) + gathered values/int8 or float32 out (~9 B) ~ 20 B/call.
    # Aligned to the storage write unit so each block fills whole chunks/shards; storage layout itself is unchanged.
    block_size=resolve_variant_chunk(block_size,n,bytes_per_call=20,maxmem=maxmem,memory_fraction=memory_fraction,n_variants=m,
                                     alignment=int(shards[1]) if shards is not None else 2*int(chunks[1]))  # _create_array default shard = 2 chunks
    iterator=src.iter_blocks(block_size=int(block_size))
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator=tqdm(iterator,total=(m+int(block_size)-1)//int(block_size),desc="PLINK/Zarr ingest",unit="block")
        except Exception: pass
    offset=0
    for block in iterator:
        X=np.asarray(block.data)
        width=X.shape[1]
        if X.shape[0] != n: raise ValueError("ingest requires a stable full-sample row set")
        if offset+width > m: raise ValueError("source yielded more variants than its metadata")
        if enc == "hardcall_int8":
            finite=np.isfinite(X)
            bad=finite & ~np.isin(X,(0,1,2))
            if np.any(bad):
                raise ValueError("hardcall_int8 requested but source contains non-hardcall dosages")
            out=np.full(X.shape,-1,dtype=np.int8); out[finite]=X[finite].astype(np.int8)
        else:
            out=np.asarray(X,dtype=np.float32)
        data[:,offset:offset+width]=out
        offset += width
    if offset != m: raise ValueError(f"source yielded {offset} variants but metadata describes {m}")

    g.attrs.update({"encoding":enc,"n_samples":int(n),"n_variants":int(m),"chunk_shape":list(map(int,chunks))})
    root.attrs["domino_write_state"]="complete"
    try:
        zarr.consolidate_metadata(str(tmp)); root.attrs["domino_consolidated_metadata"]=True
        zarr.consolidate_metadata(str(tmp))
    except Exception as exc:
        root.attrs["domino_consolidated_metadata"]=False; root.attrs["domino_consolidation_error"]=str(exc)
    if dest.exists(): shutil.rmtree(dest)
    os.replace(tmp,dest)
    finalize_zarr_artifact(str(dest),artifact_type="genotype_store",metadata={"n_samples":int(n),"n_variants":int(m),"encoding":enc})
    return str(dest)



__all__=["ingest_genotypes"]


def consolidate_genotypes(source, destination, **kwargs):
    """Backward-compatible alias for :func:`ingest_genotypes`.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``ingest_genotypes``. See those callables for
        accepted keys and values. Passed to ``ingest_genotypes`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``ingest_genotypes(source, destination, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return ingest_genotypes(source, destination, **kwargs)
