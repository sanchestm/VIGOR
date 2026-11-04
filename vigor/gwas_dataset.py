"""Canonical dense xarray representation for VIGOR mvGWAS results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .store import annotate_dataset, finalize_zarr_artifact


def _xr():
    """Import xarray for the canonical dense association data model.
        
    """
    try: import xarray as xr
    except ImportError as exc: raise ImportError("GWAS xarray output requires VIGOR[io]") from exc
    return xr


def as_gwas_dataset(result):
    """Normalize an internal association result to variant × trait xarray.

    This is an internal/public canonicalizer, not a legacy public result mode.
    Variant metadata are stored once along the variant dimension.

    Parameters
    ----------
    result : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes.

    Returns
    -------
    object
        Computed result from ``ds``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_xr()
    if isinstance(result,xr.Dataset):
        if "variant" not in result.dims or "trait" not in result.dims:
            raise ValueError("GWAS Dataset requires variant and trait dimensions")
        return result
    if not isinstance(result,pd.DataFrame): raise TypeError("association result must be an xarray Dataset")
    required={"variant_index","trait"}
    if not required.issubset(result.columns): raise ValueError(f"internal GWAS table missing {required-result.columns.to_series().index.to_series().index if False else required-set(result.columns)}")
    frame=result.copy()
    variants=pd.Index(frame["variant_index"].drop_duplicates().sort_values(),name="variant")
    traits=pd.Index(frame["trait"].astype(str).drop_duplicates(),name="trait")
    base=frame.drop_duplicates("variant_index").set_index("variant_index").reindex(variants)
    coords={"variant":variants.to_numpy(np.int64),"trait":traits.to_numpy(str)}
    for src,dst in (("snp","variant_id"),("chrom","chromosome"),("pos","position"),("ref","ref"),("alt","alt")):
        if src in base:
            vals=base[src].to_numpy()
            if src in {"snp","chrom","ref","alt"}: vals=vals.astype(str)
            coords[dst]=("variant",vals)
    data={}
    skip={"variant_index","trait","snp","chrom","pos","ref","alt"}
    for col in frame.columns:
        if col in skip: continue
        # dense numeric/statistical variables only; categorical labels are encoded.
        if isinstance(frame[col].dtype, pd.CategoricalDtype) or frame[col].dtype.kind in "OUSb":
            cats=pd.Index(pd.Series(frame[col].astype(str)).drop_duplicates())
            code_map={v:i for i,v in enumerate(cats)}
            tmp=frame[["variant_index","trait",col]].copy(); tmp[col]=tmp[col].astype(str).map(code_map).astype(np.int16)
            mat=tmp.pivot(index="variant_index",columns="trait",values=col).reindex(index=variants,columns=traits).to_numpy()
            data[col]=(("variant","trait"),mat)
        else:
            mat=frame.pivot(index="variant_index",columns="trait",values=col).reindex(index=variants,columns=traits).to_numpy()
            data[col]=(("variant","trait"),mat)
    ds=xr.Dataset(data,coords=coords)
    # Category mappings as attrs for variables that were encoded.
    for col in frame.columns:
        if col in skip or col not in ds: continue
        if isinstance(frame[col].dtype, pd.CategoricalDtype) or frame[col].dtype.kind in "OUSb":
            cats=pd.Index(pd.Series(frame[col].astype(str)).drop_duplicates()).tolist()
            ds[col].attrs["categories"]=cats
    from .store import annotate_dataset
    ds=annotate_dataset(ds,"gwas_result",result_type="dense_multivariate_gwas")
    return ds


def write_gwas_zarr(ds, path, *, variant_chunk=8192, trait_chunk=8, shard_variants=65536, consolidated=True, mode="w"):
    """Write canonical variant x trait GWAS output as sharded/consolidated Zarr v3.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    variant_chunk : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``8192``.

    trait_chunk : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``8``.

    shard_variants : int or None
        Variants per outer Zarr v3 shard, grouping storage chunks. Positive extents must be
        compatible with the chunk shape; None uses the writer's shard policy. Default:
        ``65536``.

    consolidated : bool or None
        Whether to request consolidated Zarr metadata. None lets the reader choose/fall back
        according to the store metadata and backend support. Default: ``True``.

    mode : str
        Input supplied to ``ds.to_zarr`` as ``mode``. See that callable's documented ``mode``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``'w'``.

    Returns
    -------
    object
        Computed result from ``str(path)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_xr()
    if not isinstance(ds,xr.Dataset): raise TypeError("ds must be xarray.Dataset")
    if set(("variant","trait"))-set(ds.dims): raise ValueError("GWAS Dataset requires variant and trait dimensions")
    encoding={}
    for name,da in ds.data_vars.items():
        chunks=[]
        for d in da.dims:
            if d=="variant": chunks.append(min(int(variant_chunk),ds.sizes[d]))
            elif d=="trait": chunks.append(min(int(trait_chunk),ds.sizes[d]))
            else: chunks.append(ds.sizes[d])
        enc={"chunks":tuple(chunks)}
        if "variant" in da.dims and shard_variants is not None:
            sh=list(chunks); j=da.dims.index("variant"); n=int(ds.sizes["variant"]); inner=int(chunks[j])
            target=min(max(inner,int(shard_variants)),n)
            outer=inner if target<=inner else max(inner,(target//inner)*inner)
            sh[j]=outer; enc["shards"]=tuple(sh)
        encoding[name]=enc
    from .store import annotate_dataset
    ds=annotate_dataset(ds,"gwas_result",result_type="dense_multivariate_gwas")
    ds=annotate_dataset(ds,"mvgwas_result",domino_data_model="2.11.0")
    ds.to_zarr(path,mode=mode,zarr_format=3,consolidated=False,encoding=encoding)
    if consolidated: finalize_zarr_artifact(path,artifact_type="mvgwas_result")
    return str(path)


def open_gwas_zarr(path, *, chunks="auto", allow_incomplete=False):
    """Open an association store, retaining numeric zeros and checking completion.

    allow_incomplete : bool
        Default False rejects explicitly writing/failed stores. True is only for
        inspecting partial artifacts. Unmarked legacy stores remain readable.
    Complexity: O(metadata) setup; numerical arrays remain lazy until requested.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage. Default: ``'auto'``.

    Returns
    -------
    object
        Computed result from ``xr.open_zarr(path, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_xr()
    import zarr
    state=zarr.open_group(str(path),mode='r',use_consolidated=False).attrs.get('domino_write_state')
    if state is not None and state != 'complete' and not allow_incomplete:
        raise ValueError(f"association store is incomplete (state={state}); use allow_incomplete=True only for inspection")
    kwargs=dict(chunks={} if chunks=="auto" else chunks,consolidated=None,mask_and_scale=False)
    try: kwargs["create_default_indexes"]=False; return xr.open_zarr(path,**kwargs)
    except TypeError: kwargs.pop("create_default_indexes",None); return xr.open_zarr(path,**kwargs)


def export_gwas_parquet(ds, path, *, metric=None, threshold=None, traits=None, chromosomes=None,
                        variant_chunk=65536, compression="zstd"):
    """Export a filtered/long view from canonical dense xarray GWAS results.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema. Passed to ``ds.isel`` as ``variant``.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    metric : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value.
        Source-recognized literal values: ``p``, ``p_value``, ``pvalue``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to ``ds.sel`` as
        ``trait``.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    variant_chunk : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``65536``. Passed
        to ``bounded_variant_chunk`` as ``requested``.

    compression : codec specification or None
        Compression codec/settings passed to the storage backend. None uses its
        uncompressed/default path; codec support depends on the installed Zarr/Parquet backend.
        Default: ``'zstd'``. Passed to ``pq.ParquetWriter`` as ``compression``.

    Returns
    -------
    object
        Computed result from ``str(path)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr=_xr()
    if not isinstance(ds,xr.Dataset): raise TypeError("ds must be xarray.Dataset")
    if traits is not None:
        if 'trait' not in ds.xindexes:ds=ds.set_xindex('trait')
        ds=ds.sel(trait=list(map(str,traits)))
    if chromosomes is not None:
        keep=np.isin(ds["chromosome"].astype(str),np.asarray(list(map(str,chromosomes))))
        ds=ds.isel(variant=np.flatnonzero(np.asarray(keep)))
    try: import pyarrow as pa, pyarrow.parquet as pq
    except ImportError as exc: raise ImportError("Parquet export requires VIGOR[io]") from exc
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); writer=None
    from .resources import bounded_variant_chunk
    dtype_bytes=sum(max(8,da.dtype.itemsize) for da in ds.data_vars.values())
    variant_chunk=bounded_variant_chunk(ds.sizes['trait'],requested=variant_chunk,
                                       bytes_per_call=max(16,dtype_bytes*3),memory_fraction=None,max_chunk=int(variant_chunk))
    try:
        for st in range(0,max(1,ds.sizes["variant"]),int(variant_chunk)):
            block=ds.isel(variant=slice(st,min(ds.sizes["variant"],st+int(variant_chunk)))).compute()
            for dim in ('variant','trait'):
                if dim not in block.xindexes:block=block.set_xindex(dim)
            frame=block.to_dataframe().reset_index()
            if metric is not None and threshold is not None:
                vals=pd.to_numeric(frame[metric],errors="coerce")
                # p/neglog naming decides direction; explicit raw p uses <=.
                keep=vals<=float(threshold) if str(metric).lower() in {"p","pvalue","p_value"} or str(metric).lower().startswith("p_") else vals>=float(threshold)
                frame=frame.loc[keep]
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(path,table.schema,compression=compression)
            else:
                if table.schema!=writer.schema: table=table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    return str(path)


def select_hits(ds, *, metric=None, threshold=5.0, traits=None, chromosomes=None):
    """Select hits.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    metric : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``None``, choosing canonical
        ``neglog_p`` when present, then legacy ``neglog_p_additive``.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``5.0``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to ``ds.sel`` as
        ``trait``.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``selected.where(selected[metric] >= float(threshold))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if traits is not None:
        if 'trait' not in ds.xindexes:ds=ds.set_xindex('trait')
        ds=ds.sel(trait=list(map(str,traits)))
    if chromosomes is not None:
        keep=np.isin(np.asarray(ds["chromosome"].astype(str)),np.asarray(list(map(str,chromosomes))))
        ds=ds.isel(variant=np.flatnonzero(keep))
    if metric is None:
        metric='neglog_p' if 'neglog_p' in ds else 'neglog_p_additive'
    mask=ds[metric]>=float(threshold)
    indexers={dim:np.flatnonzero(np.asarray(mask.any(dim=[d for d in mask.dims if d!=dim]).compute())) for dim in mask.dims}
    selected=ds.isel(indexers)
    return selected.where(selected[metric]>=float(threshold))


__all__=["as_gwas_dataset","write_gwas_zarr","open_gwas_zarr","export_gwas_parquet","select_hits"]
