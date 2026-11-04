"""Canonical xarray/Zarr data model for VIGOR 2.12.1.

Public dense scientific objects are xarray Datasets. BED/PGEN/Zarr are physical
backends. Numerical kernels may unwrap bounded blocks to NumPy internally.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from .store import canonical_provenance, finalize_zarr_artifact, sharded_dataset_encoding


def _require_xarray():
    """Import and verify availability of xarray.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("VIGOR's canonical data model requires xarray; install VIGOR[io]") from exc
    return xr


def _require_dask():
    """Import and verify availability of dask.
    """
    try:
        import dask.array as da
        from dask import delayed
    except ImportError as exc:
        raise ImportError("Lazy BED/PGEN xarray access requires dask; install VIGOR[io]") from exc
    return da, delayed


def _strip_prefix(path):
    """Remove a recognized genotype companion-file suffix to recover the fileset prefix.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.
    """
    s = str(path)
    for ext in (".pvar.zst", ".pgen.pgi", ".pgen", ".pvar", ".psam", ".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[:-len(ext)]
    return s


def _source_kind(source, source_format="auto"):
    """Identify a supported PLINK/Zarr backend and reject ambiguous or absent genotype sources.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    source_format : str
        Physical genotype backend: 'auto' identifies the source and rejects ambiguous
        PLINK1/PLINK2 prefixes; 'plink1'/'bed'/'bfile' selects BED, 'plink2'/'pgen'/'pfile'
        selects PGEN, and store/array support follows open_genotypes. Zarr uses its canonical
        schema. Default: ``'auto'``.
    """
    p = Path(str(source)); prefix = _strip_prefix(source); fmt = str(source_format).lower()
    has1 = all(Path(prefix + ext).exists() for ext in (".bed", ".bim", ".fam"))
    has2 = all(Path(prefix + ext).exists() for ext in (".pgen", ".pvar", ".psam"))
    if fmt == "auto" and has1 and has2:
        raise ValueError("both PLINK1 and PLINK2 files exist; specify source_format")
    if fmt in {"plink1", "bed", "bfile"} or (fmt == "auto" and has1): return "plink1", prefix
    if fmt in {"plink2", "pgen", "pfile"} or (fmt == "auto" and has2): return "plink2", prefix
    if p.exists() and (p.suffix == ".zarr" or p.is_dir()): return "zarr", str(p)
    raise FileNotFoundError(f"could not identify genotype source {source!r}")


def _read_plink1_block(prefix, start, count):
    """Read plink1 block.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    count : array-like or int
        Weights or multiplicities used in this accumulation/resampling step. Align vector
        entries with the indexed sample/variant axis; scalar counts specify the number of
        consecutive items.
    """
    from .io import PlinkReader
    with PlinkReader(prefix, dtype=np.float32) as r:
        return r.read_block(int(start), int(count))


def _read_pgen_block(prefix, start, count):
    """Read PLINK2 genotype block.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    count : array-like or int
        Weights or multiplicities used in this accumulation/resampling step. Align vector
        entries with the indexed sample/variant axis; scalar counts specify the number of
        consecutive items.
    """
    from .genotype_source import Plink2GenotypeSource
    src = Plink2GenotypeSource(prefix)
    try:
        cols=[]
        for v in range(int(start), int(start)+int(count)):
            buf=np.empty(len(src.sample_ids),dtype=np.int8)
            src.reader.read(int(v),buf)
            x=buf.astype(np.float32); x[x<0]=np.nan; cols.append(x)
        return np.column_stack(cols)
    finally:
        src.close()


def _lazy_plink_array(prefix, kind, n_samples, n_variants, variant_chunk):
    """Construct a lazy sample-by-variant Dask array from bounded PLINK read tasks.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    kind : str
        Edge/evidence/input interpretation selected by this constructor/helper. See the local
        branches and the source-derived option descriptions below; each builder admits only its
        own subset, not a package-wide enum. Source-recognized literal values: ``plink1``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Passed to ``da.from_delayed`` as ``shape``.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    variant_chunk : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.
    """
    da, delayed = _require_dask()
    pieces=[]
    reader = _read_plink1_block if kind == "plink1" else _read_pgen_block
    for st in range(0, int(n_variants), int(variant_chunk)):
        ct=min(int(variant_chunk), int(n_variants)-st)
        d=delayed(reader)(prefix, st, ct)
        pieces.append(da.from_delayed(d, shape=(int(n_samples), ct), dtype=np.float32))
    return da.concatenate(pieces, axis=1) if len(pieces)>1 else pieces[0]


def _unpack2_numpy(packed, n_variants):
    """Decode four canonical two-bit genotype calls per byte and trim padded variants.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.
    """
    p=np.asarray(packed,dtype=np.uint8)
    out=np.empty((p.shape[0], p.shape[1]*4),dtype=np.float32)
    for shift in range(4):
        out[:,shift::4]=((p >> (2*shift)) & 3).astype(np.float32)
    out=out[:,:int(n_variants)]
    out[out==3]=np.nan
    return out


def _unpack2_dask(packed, n_variants):
    """Decode canonical packed hardcalls blockwise while retaining Dask laziness.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.
    """
    try:
        import dask.array as da
    except ImportError:
        return _unpack2_numpy(np.asarray(packed), n_variants)
    if not isinstance(packed, da.Array):
        return _unpack2_numpy(packed, n_variants)
    # Each packed byte expands to four variants; map complete packed chunks.
    chunks=(packed.chunks[0], tuple(c*4 for c in packed.chunks[1]))
    out=da.map_blocks(_unpack2_numpy, packed, int(n_variants), dtype=np.float32, chunks=chunks)
    return out[:,:int(n_variants)]


def open_genotype_dataset(source, *, source_format="auto", variant_chunk="auto", chunks="auto",
                          maxmem=None, memory_fraction=None, workers=1):
    """Open BED, PGEN or VIGOR Zarr as one canonical lazy xarray Dataset.

    Variables use Variant Call Format-style names where practical:
    ``GT`` hard-call dosage, ``DS`` dosage, ``PDS`` phased dosage, ``GP``
    genotype probabilities, ``HP`` haplotype/founder probabilities, and ``GL``
    log-likelihoods. The public logical hard-call dimensions are always
    ``sample × variant`` even when the Zarr backend stores packed 2-bit bytes.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    source_format : str
        Physical genotype backend: 'auto' identifies the source and rejects ambiguous
        PLINK1/PLINK2 prefixes; 'plink1'/'bed'/'bfile' selects BED, 'plink2'/'pgen'/'pfile'
        selects PGEN, and store/array support follows open_genotypes. Zarr uses its canonical
        schema. Default: ``'auto'``.

    variant_chunk : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``bounded_variant_chunk`` as ``requested``.

    chunks : str, mapping or None
        Chunk settings passed to the xarray/Zarr reader. 'auto' requests automatic lazy chunking
        for stores; mappings specify dimensions. None requests unchunked arrays where supported.
        PLINK inputs are already built from delayed variant blocks controlled by variant_chunk.
        Default: ``'auto'``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``bounded_variant_chunk``
        as ``maxmem``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to ``bounded_variant_chunk`` as
        ``memory_fraction``.

    workers : int or str
        Concurrency count for the local worker pool. Positive integers select workers; auto,
        where supported, derives a bounded count from hardware/workload. More workers increase
        memory demand. Default: ``1``. Passed to ``bounded_variant_chunk`` as ``workers``.

    Returns
    -------
    object
        Computed result from ``ds``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_require_xarray(); kind, loc=_source_kind(source, source_format)
    if kind in {"plink1","plink2"}:
        from .genotype_source import Plink1GenotypeSource, Plink2GenotypeSource
        src=Plink1GenotypeSource(loc) if kind=="plink1" else Plink2GenotypeSource(loc)
        try:
            samples=np.asarray(src.sample_ids,dtype=str)
            vm=src.variant_metadata.reset_index(drop=True)
            n,m=len(samples),len(vm)
        finally:
            src.close()
        from .resources import bounded_variant_chunk
        variant_chunk=bounded_variant_chunk(n,requested=variant_chunk,maxmem=maxmem,
                                            memory_fraction=memory_fraction,workers=workers)
        X=_lazy_plink_array(loc,kind,n,m,variant_chunk)
        coords={
            "sample":samples,
            "variant":vm["variant_id"].astype(str).to_numpy(),
            "chromosome":("variant",vm["chromosome"].astype(str).to_numpy()),
            "position":("variant",pd.to_numeric(vm["position"],errors="coerce").to_numpy(np.int64)),
        }
        for name in ("ref","alt"):
            if name in vm: coords[name]=("variant",vm[name].astype(str).to_numpy())
        ds=xr.Dataset({"GT":(("sample","variant"),X)},coords=coords)
        ds["GT"].attrs.update({"representation":"biallelic_hardcall_dosage","missing":"NaN","physical_backend":kind})
        ds.attrs.update({"domino_data_model":"2.11.0","genotype_backend":kind,"genotype_source":str(loc)})
        return ds

    # Zarr v3 canonical genotype store.
    try:
        ds0=xr.open_zarr(loc,group="genotype",chunks={} if chunks=="auto" else chunks,
                         consolidated=None,create_default_indexes=False)
    except TypeError:
        ds0=xr.open_zarr(loc,group="genotype",chunks={} if chunks=="auto" else chunks,consolidated=None)
    if "GT_packed2" in ds0:
        nvar=int(ds0.attrs.get("logical_n_variants", ds0.sizes.get("variant",0)))
        gt=_unpack2_dask(ds0["GT_packed2"].data,nvar)
        ds=ds0.drop_vars("GT_packed2")
        ds["GT"]=(("sample","variant"),gt)
        ds["GT"].attrs.update({"representation":"packed2_decoded","physical_backend":"zarr3"})
    else:
        ds=ds0.rename({old:new for old,new in (("hardcall","GT"),("dosage","DS")) if old in ds0 and new not in ds0})
        if "GT" in ds:
            gt=ds["GT"].astype(np.float32)
            ds["GT"]=gt.where(gt>=0)
    ds.attrs["domino_data_model"]="2.11.0"
    ds.attrs["genotype_backend"]="zarr3"
    return ds


def _pack2_block(X):
    """Return the pack2 block result computed by ``(code[:, :, 0] | code[:, :, 1] << 2 | code[:, :, 2] << 4 | code[:, :, 3] << 6).astype``.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.
    """
    x=np.asarray(X)
    finite=np.isfinite(x)
    bad=finite & ~np.isin(x,(0,1,2))
    if np.any(bad): raise ValueError("packed2 requires biallelic hard calls 0/1/2/missing")
    code=np.full(x.shape,3,dtype=np.uint8); code[finite]=x[finite].astype(np.uint8)
    pad=(-code.shape[1])%4
    if pad: code=np.pad(code,((0,0),(0,pad)),constant_values=3)
    code=code.reshape(code.shape[0],-1,4)
    return (code[:,:,0] | (code[:,:,1]<<2) | (code[:,:,2]<<4) | (code[:,:,3]<<6)).astype(np.uint8)


def write_genotype_zarr(dataset, destination, *, pack_hardcalls=True, variant_chunk=16384,
                        float_probability_dtype="float16", log_likelihood_dtype="float32",
                        shards=None, compressor=None, overwrite=False, consolidated=True):
    """Write canonical genotype Dataset to consolidated/sharded Zarr v3.

    Hard calls are physically packed to two bits/call by default. Probability
    arrays are written using explicit precision policies while preserving xarray
    dimension names and coordinates.

    Parameters
    ----------
    dataset : xarray.Dataset
        Canonical genotype Dataset with sample and variant dimensions and matching coordinates.
        GT represents sample × variant 0/1/2 hardcalls with NaN missing; DS/PDS/GP/HP/GL retain
        their documented dosage/probability dimensions. Variable names and existing schema keys
        are preserved.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    pack_hardcalls : bool
        Physically pack GT to two bits per call when True; codes 0/1/2 are hardcalls and 3 marks
        missing, with padding trimmed using the logical variant count. False writes the explicit
        hardcall variable without packed storage. Default: ``True``.

    variant_chunk : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``16384``.

    float_probability_dtype : NumPy dtype or str
        Storage dtype for probability variables such as GP/HP. Floating NumPy dtypes are
        accepted by the writer; float16 reduces disk space at the cost of precision in stored
        probabilities. Default: ``'float16'``.

    log_likelihood_dtype : NumPy dtype or str
        Storage dtype for log-likelihood variables such as GL. Use a floating NumPy dtype that
        preserves the required dynamic range; float32 is the default likelihood precision.
        Default: ``'float32'``.

    shards : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``None``. Passed to ``g.create_array`` as ``shards``.

    compressor : codec specification or None
        Compression codec/settings passed to the storage backend. None uses its
        uncompressed/default path; codec support depends on the installed Zarr/Parquet backend.
        Default: ``None``.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``False``.

    consolidated : bool or None
        Whether to request consolidated Zarr metadata. None lets the reader choose/fall back
        according to the store metadata and backend support. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``str(dest)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_require_xarray()
    if not isinstance(dataset,xr.Dataset): raise TypeError("dataset must be an xarray.Dataset")
    if "sample" not in dataset.dims or "variant" not in dataset.dims: raise ValueError("requires sample and variant dimensions")
    dest=Path(destination)
    if dest.exists() and not overwrite: raise FileExistsError(dest)
    if dest.exists() and overwrite:
        import shutil; shutil.rmtree(dest)
    try: import zarr
    except ImportError as exc: raise ImportError("Zarr v3 output requires VIGOR[io]") from exc
    root=zarr.open_group(str(dest),mode="w",zarr_format=3)
    root.attrs.update(canonical_provenance("genotype_store",write_state="writing",domino_data_model="2.11.0"))
    g=root.require_group("genotype")
    n=int(dataset.sizes["sample"]); m=int(dataset.sizes["variant"])
    # Coordinates/metadata first; xarray writes these with dimension metadata.
    coord_vars={}
    for c in ("sample","variant","chromosome","position","ref","alt"):
        if c in dataset.coords: coord_vars[c]=dataset.coords[c]
    cd=xr.Dataset(coords=coord_vars)
    cd.to_zarr(str(dest),group="genotype",mode="a",zarr_format=3,consolidated=False,
               encoding=sharded_dataset_encoding(cd))
    if "GT" in dataset and pack_hardcalls:
        nbytes=(m+3)//4; cbytes=max(1,(int(variant_chunk)+3)//4)
        arr=g.create_array("GT_packed2",shape=(n,nbytes),chunks=(min(n,4096),min(nbytes,cbytes)),dtype="u1",
                           dimension_names=("sample","genotype_byte"),fill_value=255,overwrite=True,
                           shards=shards if shards is not None else (min(n,4096)*2,min(nbytes,cbytes)*4))
        arr.attrs.update({"_ARRAY_DIMENSIONS":["sample","genotype_byte"],"logical_n_variants":m,"bits_per_call":2,"missing_code":3})
        gt=dataset["GT"]
        for st in range(0,m,int(variant_chunk)):
            en=min(m,st+int(variant_chunk)); a=(st//4)*4; b=min(m,math.ceil(en/4)*4)
            block=np.asarray(gt.isel(variant=slice(a,b)).compute() if hasattr(gt.data,"compute") else gt.isel(variant=slice(a,b)).values)
            packed=_pack2_block(block)
            arr[:,a//4:a//4+packed.shape[1]]=packed
        g.attrs.update({"hardcall_encoding":"packed2","logical_n_variants":m})
    elif "GT" in dataset:
        gt=dataset[["GT"]].copy()
        gt["GT"]=gt["GT"].fillna(-1).astype(np.int8)
        gt.to_zarr(str(dest),group="genotype",mode="a",zarr_format=3,consolidated=False,
                   encoding={v:sharded_dataset_encoding(gt)[v] for v in gt.data_vars})
    # Rich variables. Keep GT out because it may be packed.
    dtype_map={"DS":np.dtype(float_probability_dtype),"PDS":np.dtype(float_probability_dtype),
               "GP":np.dtype(float_probability_dtype),"HP":np.dtype(float_probability_dtype),
               "GL":np.dtype(log_likelihood_dtype)}
    rich=[v for v in dtype_map if v in dataset]
    if rich:
        rd=dataset[rich].copy()
        for v in rich:
            rd[v]=rd[v].astype(dtype_map[v])
        enc={v:spec for v,spec in sharded_dataset_encoding(rd).items() if v in rd.data_vars}
        rd=rd.chunk({d:min(rd.sizes[d], max(enc[v]['chunks'][rd[v].dims.index(d)]
                                         for v in rich if d in rd[v].dims)) for d in rd.dims})
        rd.to_zarr(str(dest),group="genotype",mode="a",zarr_format=3,consolidated=False,encoding=enc,align_chunks=True)
    g.attrs.update({"logical_n_variants":m,"float_probability_dtype":str(float_probability_dtype),"log_likelihood_dtype":str(log_likelihood_dtype)})
    if consolidated:
        finalize_zarr_artifact(str(dest),artifact_type="genotype_store",metadata={"logical_n_variants":int(m),"n_samples":int(n)})
    else:
        root.attrs["domino_write_state"]="complete"
    return str(dest)


__all__=["open_genotype_dataset","write_genotype_zarr"]
