"""Canonical labeled persistent storage for VIGOR 2.12.1.

The storage layer intentionally stops at the boundary of the numerical kernels:
Zarr/xarray own persistence, coordinates, chunking and alignment while
NumPy/SciPy own BLAS, eigensolvers, PCG and optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

SCHEMA_VERSION = "domino-zarr-v3-2"



def canonical_provenance(artifact_type, *, write_state="writing", **metadata):
    """Construct canonical writer/schema/version metadata without renaming serialized keys.

    Parameters
    ----------
    artifact_type : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options.

    write_state : str
        Input for the operation ``str(write_state)``. See ``vigor.store.canonical_provenance``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``'writing'``.

    **metadata : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import platform, time
    from . import __version__
    out={"domino_schema_version":SCHEMA_VERSION,"domino_storage_format":"zarr3",
         "domino_artifact_type":str(artifact_type),"domino_write_state":str(write_state),
         "domino_package_version":__version__,
         "created_unix":float(time.time()),"python_version":platform.python_version()}
    out.update({k:v for k,v in metadata.items() if v is not None})
    return out

def annotate_dataset(ds, artifact_type, **metadata):
    """Attach canonical artifact provenance to a labeled scientific Dataset.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    artifact_type : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options.

    **metadata : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Passed to ``canonical_provenance`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    out=ds.copy(deep=False); attrs=dict(getattr(out,"attrs",{}));
    attrs.update(canonical_provenance(artifact_type,write_state="complete",**metadata)); out.attrs=attrs; return out

def _json_safe(value):
    """Convert common NumPy/path metadata to JSON-safe values.

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation ``value.item()``. See ``vigor.store._json_safe`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.
    """
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json_safe(v) for v in value]
    try: json.dumps(value); return value
    except Exception: return str(value)


def finalize_zarr_artifact(path, *, artifact_type=None, metadata=None, write_sidecar=True):
    """Mark a Zarr v3 artifact complete, consolidate, and write a JSON manifest.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    artifact_type : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    metadata : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Default: ``None``.

    write_sidecar : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``root``; see the summary and the called implementation for its labeled/numerical representation.
    """
    zarr=_require_zarr(); path=str(path); root=zarr.open_group(path,mode="a",zarr_format=3,use_consolidated=False)
    if artifact_type is not None: root.attrs["domino_artifact_type"]=str(artifact_type)
    if metadata:
        for k,v in metadata.items(): root.attrs[str(k)]=_json_safe(v)
    from . import __version__
    import time
    root.attrs["finalized_unix"]=time.time()
    root.attrs["domino_package_version"]=__version__
    root.attrs["domino_schema_version"]=SCHEMA_VERSION
    root.attrs["domino_storage_format"]="zarr3"
    root.attrs["domino_write_state"]="writing"
    try:
        root.attrs["domino_write_state"]="complete"
        zarr.consolidate_metadata(path); root.attrs["domino_consolidated_metadata"]=True
        zarr.consolidate_metadata(path)
    except Exception as exc:
        root.attrs["domino_consolidated_metadata"]=False
        root.attrs["domino_consolidation_error"]=str(exc)
        root.attrs["domino_write_state"]="failed"
        raise RuntimeError(f"could not consolidate Zarr artifact {path}") from exc
    if write_sidecar:
        groups=[]; arrays=[]
        def walk(g,prefix=""):
            """Traverse the enclosing graph/store in the operation-defined order.

            Parameters
            ----------
            g : object (see the cited input contract)
                Input for the operation ``g.groups()``. See ``vigor.store.finalize_zarr_artifact.walk``
                and its caller for the scientific representation; the shown operation constrains the
                required type/shape.

            prefix : str or os.PathLike
                PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
                PLINK2 adapter). Companion metadata and genotype files must describe the same samples
                and variants. Default: ``''``.
            """
            for name,child in g.groups():
                q=f"{prefix}{name}"; groups.append(q); walk(child,q+"/")
            for name,arr in g.arrays(): arrays.append({"path":f"{prefix}{name}","shape":list(map(int,arr.shape)),"dtype":str(arr.dtype),"chunks":list(map(int,arr.chunks)) if getattr(arr,"chunks",None) else None,
                "shards":list(map(int,arr.shards)) if getattr(arr,"shards",None) else None,
                "dimensions":list(getattr(arr.metadata,"dimension_names",None) or arr.attrs.get("_ARRAY_DIMENSIONS",[]))})
        walk(root)
        manifest={"path":path,"schema_version":root.attrs.get("domino_schema_version"),
                  "storage_format":"zarr3","write_state":root.attrs.get("domino_write_state"),
                  "artifact_type":root.attrs.get("domino_artifact_type"),
                  "consolidated_metadata":bool(root.attrs.get("domino_consolidated_metadata",False)),
                  "groups":groups,"arrays":arrays,"attrs":_json_safe(dict(root.attrs))}
        Path(path+".metadata.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    return root


def sharded_dataset_encoding(ds, *, chunk_bytes=8*2**20, shard_bytes=32*2**20):
    """Bound chunks by byte size; shards reduce object count without eager loads.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Labeled data object containing the variables and dimension coordinates read below.
        Scientific axes/metadata must match the corresponding genotype, association or posterior
        schema.

    chunk_bytes : int
        Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
        extent as named. Allocation planners multiply these costs by the relevant dimensions;
        use nonnegative costs consistent with the actual representation. Default: ``8 * 2 **
        20``.

    shard_bytes : int
        Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
        extent as named. Allocation planners multiply these costs by the relevant dimensions;
        use nonnegative costs consistent with the actual representation. Default: ``32 * 2 **
        20``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    out={}
    preferred={"sample":1024,"sample_i":1024,"sample_j":1024,"variant":4096,
               "variant_i":1024,"variant_j":1024,"trait":128,"trait_i":128,"trait_j":128,
               "factor":32,"component":1,"chromosome":1,"fold":1,"feature":256,
               "snp":4096,"pair":4096,"edge":4096,"node":1024,"replicate":64}
    for name,da in ds.variables.items():
        if da.ndim==0 or any(int(n)==0 for n in da.shape):continue
        chunks=[min(int(n),preferred.get(d,1024)) for d,n in zip(da.dims,da.shape)]
        itemsize=max(8 if da.dtype.kind in "OUS" else da.dtype.itemsize,1)
        while int(np.prod(chunks))*itemsize>chunk_bytes:
            j=max(range(len(chunks)),key=lambda k:chunks[k]);chunks[j]=max(1,chunks[j]//2)
        shards=list(chunks)
        # Coordinates remain compact, explicitly named arrays. Numeric data are sharded.
        spec={"chunks":tuple(chunks)}
        if name in ds.data_vars:
            for j in sorted(range(len(chunks)),key=lambda k:int(da.shape[k]),reverse=True):
                ratio=max(1,min(int(da.shape[j])//chunks[j],shard_bytes//max(1,int(np.prod(shards))*itemsize)))
                shards[j]*=ratio
            spec["shards"]=tuple(shards)
        out[name]=spec
    return out


def _require_zarr():
    """Import and verify availability of zarr.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("VIGOR Zarr storage requires `pip install 'VIGOR[io]'` (zarr>=3, xarray).") from exc
    return zarr


def _require_xarray():
    """Import and verify availability of xarray.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VIGOR labeled storage requires xarray; install VIGOR[io].") from exc
    return xr


def sample_hash(ids) -> str:
    """Sample hash.

    Parameters
    ----------
    ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.

    Returns
    -------
    object
        Computed result from ``h.hexdigest()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    values = np.asarray(ids, dtype=str)
    h = hashlib.sha256()
    h.update(str(values.shape).encode("ascii"))
    for value in values:
        h.update(value.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def array_hash(values) -> str:
    """Return the array hash result computed by ``h.hexdigest``.

    Parameters
    ----------
    values : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    Returns
    -------
    object
        Computed result from ``h.hexdigest()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    a = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii")); h.update(str(a.dtype).encode("ascii")); h.update(a.view(np.uint8))
    return h.hexdigest()


@dataclass
class StoreSummary:
    """StoreSummary represents store summary.

    Attributes
    ----------
    path : str
        Stored path; its axis/order follows the construction inputs.
    schema_version : str | None
        Stored schema version; its axis/order follows the construction inputs.
    write_state : str | None
        Stored write state; its axis/order follows the construction inputs.
    groups : tuple[str, ...]
        Stored groups; its axis/order follows the construction inputs.
    arrays : tuple[str, ...]
        Stored arrays; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    path : str
        Stored path supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    schema_version : str | None
        Serialized artifact schema identifier; retain it for reader compatibility. Required
        construction field.

    write_state : str | None
        Stored write state supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    groups : tuple[str, ...]
        Stored groups supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    arrays : tuple[str, ...]
        Stored arrays supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.
    """
    path: str
    schema_version: str | None
    write_state: str | None
    groups: tuple[str, ...]
    arrays: tuple[str, ...]


class VigorStore:
    """Versioned Zarr v3 store used as VIGOR's persistent data/cache plane."""

    def __init__(self, path, *, mode="a", lazy=True, storage_options=None):
        """Initialize VigorStore with the inputs and resource settings below.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        mode : str
            Input for the operation ``mode in {'a', 'w', 'w-', 'x'}``. See
            ``vigor.store.VigorStore.__init__`` and its caller for the scientific representation;
            the shown operation constrains the required type/shape. Default: ``'a'``.

        lazy : bool
            Whether store-backed values remain lazily loaded until their blocks are accessed; eager
            mode materializes the arrays requested by the reader. Default: ``True``.

        storage_options : mapping or None
            Named settings or metadata forwarded to the specific operation below. See that
            operation's argument/schema contract for accepted fields; None uses its default/empty
            configuration. Default: ``None``.
        """
        zarr = _require_zarr()
        self.path = str(path)
        self.mode = str(mode)
        self.lazy = bool(lazy)
        self.storage_options = {} if storage_options is None else dict(storage_options)
        # Local paths are currently the dominant VIGOR use case. Passing a
        # fsspec store object remains possible through zarr itself.
        # Mutable stores must bypass consolidated metadata while a transaction
        # is open; otherwise a previously consolidated hierarchy can hide groups
        # created later in the same store. Read-only stores may use consolidation.
        _mutable = self.mode in {"a", "w", "w-", "x", "r+"}
        self.group = zarr.open_group(self.path, mode=self.mode, zarr_format=3,
                                     use_consolidated=False if _mutable else None)
        if mode in {"a", "w", "w-", "x"}:
            self.group.attrs.update({k:v for k,v in canonical_provenance("domino_store").items() if k not in self.group.attrs})

    @property
    def attrs(self):
        """Read the root persistent-store metadata attributes.

        Returns
        -------
        object
            Computed result from ``self.group.attrs``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.group.attrs

    def mark_writing(self):
        # Refresh against live metadata at the beginning of every write
        # transaction. This is required after a prior mark_complete() has
        # consolidated a Zarr v3 hierarchy.
        """Mark the store artifact as incomplete while its scientific arrays are being written.
                
        """
        zarr = _require_zarr()
        self.group = zarr.open_group(self.path, mode="a", zarr_format=3, use_consolidated=False)
        self.group.attrs["domino_write_state"] = "writing"

    def mark_complete(self, **metadata):
        """Mark the store artifact complete and finalize its provenance/sidecar metadata.

        Parameters
        ----------
        **metadata : mapping or None
            Named settings or metadata forwarded to the specific operation below. See that
            operation's argument/schema contract for accepted fields; None uses its default/empty
            configuration. Passed to ``finalize_zarr_artifact`` as ``metadata``.
        """
        self.group=finalize_zarr_artifact(self.path,artifact_type=self.group.attrs.get("domino_artifact_type","domino_store"),metadata=metadata)

    def validate(self, *, require_complete=True):
        """Validate the persistent artifact's schema, completeness and coordinate contracts.

        Parameters
        ----------
        require_complete : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``True``.

        Returns
        -------
        object
            Computed result from ``True``; see the summary and the called implementation for its labeled/numerical representation.
        """
        schema = self.group.attrs.get("domino_schema_version")
        if schema not in {SCHEMA_VERSION, None}:
            raise ValueError(f"unsupported VIGOR store schema {schema!r}; expected {SCHEMA_VERSION!r}")
        state = self.group.attrs.get("domino_write_state")
        if require_complete and state != "complete":
            raise ValueError(f"VIGOR store is not complete (write_state={state!r})")
        return True

    def summary(self) -> StoreSummary:
        """Return the summary result computed by ``StoreSummary``.

        Returns
        -------
        object
            Computed result from ``StoreSummary(path=self.path, schema_version=self.group.attrs.get('domino_schema_version'), write_state=self.group.attrs.get('domino_write_state'), groups=tuple(groups), arrays=tuple(arrays))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        groups=[]; arrays=[]
        def walk(g, prefix=""):
            """Traverse the enclosing graph/store in the operation-defined order.

            Parameters
            ----------
            g : object (see the cited input contract)
                Input for the operation ``g.groups()``. See ``vigor.store.VigorStore.summary.walk`` and
                its caller for the scientific representation; the shown operation constrains the
                required type/shape.

            prefix : str or os.PathLike
                PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
                PLINK2 adapter). Companion metadata and genotype files must describe the same samples
                and variants. Default: ``''``.
            """
            for name, child in g.groups():
                p=f"{prefix}{name}"; groups.append(p); walk(child, p+"/")
            for name, _arr in g.arrays(): arrays.append(f"{prefix}{name}")
        walk(self.group)
        return StoreSummary(
            path=self.path,
            schema_version=self.group.attrs.get("domino_schema_version"),
            write_state=self.group.attrs.get("domino_write_state"),
            groups=tuple(groups), arrays=tuple(arrays),
        )

    def open_dataset(self, group=None, *, lazy=None, **kwargs):
        """Open dataset.

        Parameters
        ----------
        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        lazy : bool
            Whether store-backed values remain lazily loaded until their blocks are accessed; eager
            mode materializes the arrays requested by the reader. Default: ``None``. None delegates
            to the local fallback shown in this body; it is not an additional numeric/category
            value.

        **kwargs : mapping
            Additional keyword arguments consumed by the compatibility/callback logic in this body.
            See the explicit keyword lookups below for accepted keys; unused keys are subject to
            that compatibility policy.

        Returns
        -------
        object
            Computed result from ``xr.open_zarr(self.path, **open_kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr = _require_xarray()
        lazy = self.lazy if lazy is None else bool(lazy)
        open_kwargs = dict(consolidated=False, create_default_indexes=False)
        open_kwargs.update(kwargs)
        if "chunks" not in open_kwargs:
            open_kwargs["chunks"] = {} if lazy else None
        if group is not None: open_kwargs["group"] = str(group)
        return xr.open_zarr(self.path, **open_kwargs)

    def write_dataset(self, ds, group, *, mode="a", region=None, compute=True,
                      encoding=None, consolidated=True, **kwargs):
        """Write labeled, sharded Zarr v3 and finalize only after data are written.

        ``compute=False`` returns one delayed write/finalization operation.
        Region updates and full writes to existing arrays retain their physical
        encoding; only new arrays receive a creation encoding. This permits
        repeat computations in the same group without stale array values.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Labeled data object containing the variables and dimension coordinates read below.
            Scientific axes/metadata must match the corresponding genotype, association or posterior
            schema.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.

        mode : str
            Input for the operation ``dict(mode=mode, group=str(group), zarr_format=3,
            consolidated=False, compute=compute)``. See ``vigor.store.VigorStore.write_dataset`` and
            its caller for the scientific representation; the shown operation constrains the
            required type/shape. Default: ``'a'``. Source-recognized literal values: ``a-``, ``r+``.
            These select the corresponding branches below; comparison evidence alone does not
            establish an exhaustive enum.

        region : pair of int or None
            Lower and upper genomic positions defining the selected interval. Positions refer to the
            chromosome's physical coordinate system; None uses the full eligible region. Default:
            ``None``.

        compute : bool
            Input for the operation ``dict(mode=mode, group=str(group), zarr_format=3,
            consolidated=False, compute=compute)``. See ``vigor.store.VigorStore.write_dataset`` and
            its caller for the scientific representation; the shown operation constrains the
            required type/shape. Default: ``True``.

        encoding : str
            Persistent genotype encoding: 'auto' chooses hardcall_int8 for PLINK hard calls and
            dosage_float32 otherwise; 'hardcall_int8' stores 0/1/2 with -1 missing; 'dosage_float32'
            stores floating dosage with NaN missing. Non-hardcall values cannot be written as
            hardcall_int8. Default: ``None``. None delegates to the local fallback shown in this
            body; it is not an additional numeric/category value.

        consolidated : bool or None
            Whether to request consolidated Zarr metadata. None lets the reader choose/fall back
            according to the store metadata and backend support. Default: ``True``.

        **kwargs : mapping
            Additional keyword arguments consumed by the compatibility/callback logic in this body.
            See the explicit keyword lookups below for accepted keys; unused keys are subject to
            that compatibility policy.

        Returns
        -------
        object
            Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
        """
        _require_xarray()
        if self.mode=="r":raise PermissionError("cannot write through a read-only VigorStore")
        self.mark_writing()
        options=dict(mode=mode,group=str(group),zarr_format=3,consolidated=False,compute=compute)
        if region is not None:
            options["region"]=region
        else:
            planned=sharded_dataset_encoding(ds)
            for name,spec in (encoding or {}).items():planned.setdefault(name,{}).update(spec)
            for name,spec in planned.items():
                if 'shards' in spec:
                    spec['shards']=tuple(max(int(ch),(int(sh)//int(ch))*int(ch)) for ch,sh in zip(spec['chunks'],spec['shards']))
            # xarray rejects a creation encoding for an already stored array.
            # Reuse that array's physical layout for a full group update, and
            # still create new variables with the requested sharding. Opening
            # live metadata avoids a stale consolidated view between writes.
            existing={}
            if mode in {"a","a-","r+"}:
                try:
                    target=self.group[str(group)]
                    existing={name:array for name,array in target.arrays()}
                except KeyError:
                    pass
            options["encoding"]={name:spec for name,spec in planned.items() if name not in existing}
            # Align lazy write tasks with the planned storage chunks.
            ds=ds.copy(deep=False)
            for name in ds.data_vars:
                if getattr(ds[name].data,"chunks",None) is not None and name in planned:
                    chunks=existing[name].chunks if name in existing else planned[name]["chunks"]
                    ds[name]=ds[name].chunk(dict(zip(ds[name].dims,chunks)))
        options.update(kwargs)
        ds=annotate_dataset(ds,ds.attrs.get("domino_artifact_type",str(group)))
        result=ds.to_zarr(self.path,**options)
        if not compute:
            from dask import delayed
            @delayed
            def finish(_):
                """Record the final status of the monitored computation.

                Parameters
                ----------
                _ : object (see the cited input contract)
                    Compatibility/callback input accepted but not consumed by this body. Any object is
                    locally accepted; see the calling protocol for the supplied type.

                Returns
                -------
                object
                    Computed result from ``self.path``; see the summary and the called implementation for its labeled/numerical representation.
                """
                if consolidated:self.mark_complete()
                return self.path
            return finish(result)
        if consolidated:self.mark_complete()
        return result

    def consolidate(self):
        """Finalize a multi-group transaction and its metadata sidecar.

        Returns
        -------
        object
            Computed result from ``self.path``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self.mark_complete()
        return self.path

    def genotypes(self, **kwargs):
        """Return the genotypes result computed by ``XarrayGenotypeSource``.

        Parameters
        ----------
        **kwargs : mapping
            Additional keyword arguments forwarded to ``XarrayGenotypeSource``. See those callables
            for accepted keys and values. Passed to ``XarrayGenotypeSource`` as ``**expanded
            mapping``.

        Returns
        -------
        object
            Computed result from ``XarrayGenotypeSource(self.path, group='genotype', lazy=self.lazy, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        from .genotype_source import XarrayGenotypeSource
        return XarrayGenotypeSource(self.path, group="genotype", lazy=self.lazy, **kwargs)

    def grm(self, component="additive", *, chromosome="all", loco=None,
            representation="auto", materialize=True, dtype=None):
        """Return the genomic relationship matrix result computed by ``open_grm``.

        Parameters
        ----------
        component : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required. Default:
            ``'additive'``. Passed to ``open_grm`` as ``component``.

        chromosome : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported. Default: ``'all'``. Passed to ``open_grm`` as
            ``chromosome``.

        loco : array-like or numeric scalar
            Input supplied to ``open_grm`` as ``loco``. See that callable's documented ``loco``
            contract for its type, alignment and value domain. This wrapper retains that contract.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value. Passed to ``open_grm`` as ``loco``.

        representation : str
            Returned/stored container or kernel representation. The source-derived choices below
            select dense arrays, labeled tables or lazy/operator representations; see the local
            dispatcher for which are supported. Default: ``'auto'``. Passed to ``open_grm`` as
            ``representation``.

        materialize : bool
            Whether to perform the named persistence/materialization/refitting step. True enables
            it; False follows the bypass path in this routine. Refitting during CV must use training
            outcomes only. Default: ``True``. Passed to ``open_grm`` as ``materialize``.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``None``. None delegates to the local fallback shown in
            this body; it is not an additional numeric/category value. Passed to ``open_grm`` as
            ``dtype``.

        Returns
        -------
        object
            Computed result from ``open_grm(self, component=component, chromosome=chromosome, loco=loco, representation=representation, materialize=materialize, dtype=dtype)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        from .grm_store import open_grm
        return open_grm(self, component=component, chromosome=chromosome, loco=loco,
                        representation=representation, materialize=materialize, dtype=dtype)


    def write_phenotypes(self, phenotypes, *, covariates=None, group="phenotype", mode="a"):
        """Persist phenotype values/mask with explicit sample/trait coordinates.

        Parameters
        ----------
        phenotypes : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        covariates : array-like or pandas.DataFrame or None
            Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
            animal order. None uses the routine's intercept-only or no-additional-covariate path;
            rank-deficient designs may be rejected or reduced. Default: ``None``.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'phenotype'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode, encoding=encoding)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray(); import pandas as pd
        if isinstance(phenotypes,pd.DataFrame):
            ids=phenotypes.index.astype(str).to_numpy(); traits=phenotypes.columns.astype(str).to_numpy(); Y=phenotypes.to_numpy(float)
        else:
            Y=np.asarray(phenotypes,float)
            if Y.ndim!=2: raise ValueError("phenotypes must be sample x trait")
            ids=np.arange(Y.shape[0]).astype(str); traits=np.arange(Y.shape[1]).astype(str)
        ds=xr.Dataset(
            {"values":(("sample","trait"),Y),"observed":(("sample","trait"),np.isfinite(Y))},
            coords={"sample":ids,"trait":traits},
        )
        if covariates is not None:
            if isinstance(covariates,pd.DataFrame):
                C=covariates.reindex(ids).to_numpy(float); names=covariates.columns.astype(str).to_numpy()
            else:
                C=np.asarray(covariates,float); names=np.arange(C.shape[1]).astype(str)
            if C.shape[0]!=Y.shape[0]: raise ValueError("covariates must have one row per phenotype sample")
            ds["covariates"]=(("sample","covariate"),C); ds=ds.assign_coords(covariate=names)
        ds=annotate_dataset(ds,"phenotype")
        encoding={"values":{"chunks":(min(Y.shape[0],4096),min(Y.shape[1],128))},
                  "observed":{"chunks":(min(Y.shape[0],4096),min(Y.shape[1],128))}}
        return self.write_dataset(ds,group,mode=mode,encoding=encoding)

    def write_covariance_result(self, result, *, trait_names=None, group="covariance", mode="a"):
        """Persist fitted covariance/SE outputs as labeled xarray variables.

        Parameters
        ----------
        result : mapping or fitted object
            Previously prepared model/result or solver state consumed by this operation. Use the
            object returned by the corresponding fitting/preparation routine with the same animal
            and trait axes.

        trait_names : sequence or pandas.Index
            Trait labels in matrix-column order, used to label or align the resulting trait
            quantities. Default: ``None``. None delegates to the local fallback shown in this body;
            it is not an additional numeric/category value.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'covariance'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray()
        from .multivariate import _result_view
        result=_result_view(result)
        G=result.get("G"); E=result.get("E")
        if E is None: raise ValueError("result must contain E")
        Earr=np.asarray(E,float); t=Earr.shape[0]
        if trait_names is None:
            if hasattr(E,"index"): trait_names=np.asarray(E.index.astype(str))
            else: trait_names=np.arange(t).astype(str)
        trait_names=np.asarray(trait_names,dtype=str)
        data={"E":(("trait_i","trait_j"),Earr)}
        if isinstance(result.get("G_components"),dict):
            names=list(result["G_components"]); stack=np.stack([np.asarray(result["G_components"][k],float) for k in names])
            data["G"]=(("component","trait_i","trait_j"),stack); coords={"component":np.asarray(names,dtype=str),"trait_i":trait_names,"trait_j":trait_names}
            gse=result.get("G_component_se")
            if isinstance(gse,dict) and all(k in gse for k in names): data["G_se"]=(("component","trait_i","trait_j"),np.stack([np.asarray(gse[k],float) for k in names]))
        else:
            Garr=np.asarray(G,float) if G is not None else None; coords={"trait_i":trait_names,"trait_j":trait_names}
            if Garr is not None:data["G"]=(("trait_i","trait_j"),Garr)
            if result.get("G_se") is not None:data["G_se"]=(("trait_i","trait_j"),np.asarray(result["G_se"],float))
        if result.get("E_se") is not None:data["E_se"]=(("trait_i","trait_j"),np.asarray(result["E_se"],float))
        if result.get("Rg") is not None:data["rg"]=(("trait_i","trait_j"),np.asarray(result["Rg"],float))
        elif result.get("rg") is not None:data["rg"]=(("trait_i","trait_j"),np.asarray(result["rg"],float))
        if result.get("rg_se") is not None:data["rg_se"]=(("trait_i","trait_j"),np.asarray(result["rg_se"],float))
        for key in ("h2","h2_se","h2_model","h2_phenotype","phenotypic_variance_observed","phenotypic_variance_model","variance_closure_ratio","variance_closure_z"):
            if result.get(key) is not None:
                arr=np.asarray(result[key],float).reshape(-1)
                if len(arr)==t:data[key]=(("trait",),arr)
        coords["trait"]=trait_names
        ds=annotate_dataset(xr.Dataset(data,coords=coords),"covariance")
        return self.write_dataset(ds,group,mode=mode)

    def write_predictions(self, values, *, name="mvblup", sample_ids=None, trait_names=None, group="prediction", mode="a"):
        """Write predictions.

        Parameters
        ----------
        values : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required. Default:
            ``'mvblup'``.

        sample_ids : sequence of str or None
            Animal identifiers in the required row order. IDs must align with the genotype/phenotype
            source; None uses source identifiers or the routine's positional indexing convention.
            Default: ``None``. Passed to ``xr.Dataset`` as ``coords``.

        trait_names : sequence or pandas.Index
            Trait labels in matrix-column order, used to label or align the resulting trait
            quantities. Default: ``None``. None delegates to the local fallback shown in this body;
            it is not an additional numeric/category value. Passed to ``xr.Dataset`` as ``coords``.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'prediction'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray(); import pandas as pd
        if isinstance(values,pd.DataFrame):
            sample_ids=values.index.astype(str).to_numpy(); trait_names=values.columns.astype(str).to_numpy(); A=values.to_numpy(float)
        else:
            A=np.asarray(values,float)
            if A.ndim!=2: raise ValueError("prediction values must be sample x trait")
            if sample_ids is None: sample_ids=np.arange(A.shape[0]).astype(str)
            if trait_names is None: trait_names=np.arange(A.shape[1]).astype(str)
        ds=annotate_dataset(xr.Dataset({str(name):(("sample","trait"),A)},coords={"sample":np.asarray(sample_ids,dtype=str),"trait":np.asarray(trait_names,dtype=str)}),"prediction")
        return self.write_dataset(ds,group,mode=mode)


    def write_decomposition(self, *, loadings=None, specific_variance=None, scores=None,
                            component_names=None, trait_names=None, factor_names=None,
                            sample_ids=None, group="decomposition", mode="a"):
        """Persist FA/PCA decomposition outputs with explicit dimensions.

        Parameters
        ----------
        loadings : array-like or fitted basis
            Trait/feature × retained-factor loading matrix or prepared basis. Column order
            identifies factor directions and must agree with the scores used for projection.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        specific_variance : array-like
            Loading-support mask or nonnegative trait-specific diagonal variance, as named. Masks
            match trait × factor loadings; specific variances have one value per trait and augment
            the low-rank covariance. Default: ``None``. None delegates to the local fallback shown
            in this body; it is not an additional numeric/category value.

        scores : array-like, pandas.DataFrame or mapping
            Association/evaluation quantities used to rank or report candidates. Match identifiers
            to the corresponding variants, traits or feature groups; required fields are selected
            explicitly in the body. Default: ``None``. None delegates to the local fallback shown in
            this body; it is not an additional numeric/category value.

        component_names : object (see the cited input contract)
            Input supplied to ``coords.update`` as ``component``. See that callable's documented
            ``component`` contract for its type, alignment and value domain. This wrapper retains
            that contract. Default: ``None``. None delegates to the local fallback shown in this
            body; it is not an additional numeric/category value. Passed to ``coords.update`` as
            ``component``.

        trait_names : sequence or pandas.Index
            Trait labels in matrix-column order, used to label or align the resulting trait
            quantities. Default: ``None``. None delegates to the local fallback shown in this body;
            it is not an additional numeric/category value. Passed to ``coords.update`` as
            ``trait``.

        factor_names : str, index or sequence
            Axis/column/feature selector used to identify the input/output quantities in this
            operation. Named selectors must exist in the labeled input; integer selectors are valid
            zero-based positions on that axis. Default: ``None``. None delegates to the local
            fallback shown in this body; it is not an additional numeric/category value. Passed to
            ``coords.update`` as ``factor``.

        sample_ids : sequence of str or None
            Animal identifiers in the required row order. IDs must align with the genotype/phenotype
            source; None uses source identifiers or the routine's positional indexing convention.
            Default: ``None``.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'decomposition'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray(); data={}; coords={}
        if loadings is not None:
            L=np.asarray(loadings,float)
            if L.ndim==2: L=L[None,:,:]
            if L.ndim!=3: raise ValueError("loadings must be trait x factor or component x trait x factor")
            c,t,r=L.shape
            component_names=np.arange(c).astype(str) if component_names is None else np.asarray(component_names,dtype=str)
            trait_names=np.arange(t).astype(str) if trait_names is None else np.asarray(trait_names,dtype=str)
            factor_names=np.arange(r).astype(str) if factor_names is None else np.asarray(factor_names,dtype=str)
            data["loadings"]=(("component","trait","factor"),L); coords.update(component=component_names,trait=trait_names,factor=factor_names)
        if specific_variance is not None:
            P=np.asarray(specific_variance,float)
            if P.ndim==1:P=P[None,:]
            if P.ndim!=2: raise ValueError("specific_variance must be trait or component x trait")
            data["specific_variance"]=(("component","trait"),P)
            coords.setdefault("component",np.arange(P.shape[0]).astype(str)); coords.setdefault("trait",np.arange(P.shape[1]).astype(str))
        if scores is not None:
            S=np.asarray(scores,float)
            if S.ndim!=2: raise ValueError("scores must be sample x factor")
            data["scores"]=(("sample","factor"),S)
            coords.setdefault("sample",np.arange(S.shape[0]).astype(str) if sample_ids is None else np.asarray(sample_ids,dtype=str))
            coords.setdefault("factor",np.arange(S.shape[1]).astype(str) if factor_names is None else np.asarray(factor_names,dtype=str))
        if not data: raise ValueError("provide loadings, specific_variance, and/or scores")
        ds=annotate_dataset(xr.Dataset(data,coords=coords),"decomposition")
        return self.write_dataset(ds,group,mode=mode)

    def write_expert_features(self, features, *, feature_metadata=None, sample_ids=None,
                              group="experts/candidate_features", mode="a", chunks=(4096,1024)):
        """Persist phenotype-independent expert candidate features.

        Outcome-dependent selected features are deliberately not promoted to a
        global candidate matrix; selection must remain inside CV folds.

        Parameters
        ----------
        features : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        feature_metadata : object (see the cited input contract)
            Input for the operation ``pd.DataFrame(feature_metadata).reset_index(drop=True)``. See
            ``vigor.store.VigorStore.write_expert_features`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        sample_ids : sequence of str or None
            Animal identifiers in the required row order. IDs must align with the genotype/phenotype
            source; None uses source identifiers or the routine's positional indexing convention.
            Default: ``None``. Passed to ``xr.Dataset`` as ``coords``.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'experts/candidate_features'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        chunks : tuple, mapping or None
            Chunk/shard extents per persisted array axis. Positive extents must follow the array's
            dimension order; None lets the writer select extents. Shards group multiple chunks for
            Zarr v3 storage. Default: ``(4096, 1024)``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode, encoding=encoding)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray(); import pandas as pd
        if isinstance(features,pd.DataFrame):
            sample_ids=features.index.astype(str).to_numpy(); names=features.columns.astype(str).to_numpy(); X=features.to_numpy(np.float32)
        else:
            X=np.asarray(features,dtype=np.float32)
            if X.ndim!=2: raise ValueError("features must be sample x feature")
            if sample_ids is None: sample_ids=np.arange(X.shape[0]).astype(str)
            names=np.arange(X.shape[1]).astype(str)
        ds=xr.Dataset({"values":(("sample","feature"),X)},coords={"sample":np.asarray(sample_ids,dtype=str),"feature":names})
        if feature_metadata is not None:
            meta=pd.DataFrame(feature_metadata).reset_index(drop=True)
            if len(meta)!=X.shape[1]: raise ValueError("feature_metadata must have one row per feature")
            for col in meta.columns:
                vals=meta[col].to_numpy()
                if vals.dtype.kind in "OUS": vals=vals.astype(str)
                ds[str(col)]=(("feature",),vals)
        ds=annotate_dataset(ds,"expert_features")
        encoding={"values":{"chunks":(min(X.shape[0],int(chunks[0])),min(X.shape[1],int(chunks[1])))}}
        return self.write_dataset(ds,group,mode=mode,encoding=encoding)

    def write_expert_selection(self, selected_feature_ids, coefficients, *, fold_ids=None, trait_names=None,
                               group="experts/selection", mode="a"):
        """Persist fold-local selected feature IDs and coefficients for auditability.

        Parameters
        ----------
        selected_feature_ids : array-like or numeric scalar
            Input for the operation ``np.asarray(selected_feature_ids)``. See
            ``vigor.store.VigorStore.write_expert_selection`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        coefficients : array-like
            Effect or path coefficients, ordered by the model's predictor/trait axes and on the
            scale used by its fit.

        fold_ids : array-like or numeric scalar
            Input for the operation ``np.asarray(fold_ids)``. See
            ``vigor.store.VigorStore.write_expert_selection`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        trait_names : sequence or pandas.Index
            Trait labels in matrix-column order, used to label or align the resulting trait
            quantities. Default: ``None``. None delegates to the local fallback shown in this body;
            it is not an additional numeric/category value.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``'experts/selection'``.

        mode : str
            Input supplied to ``self.write_dataset`` as ``mode``. See that callable's documented
            ``mode`` contract for its type, alignment and value domain. This wrapper retains that
            contract. Default: ``'a'``.

        Returns
        -------
        object
            Computed result from ``self.write_dataset(ds, group, mode=mode)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=_require_xarray()
        ids=np.asarray(selected_feature_ids)
        coef=np.asarray(coefficients,float)
        if ids.shape!=coef.shape or ids.ndim!=3:
            raise ValueError("selected_feature_ids and coefficients must be fold x trait x selected_feature with identical shapes")
        f,t,k=ids.shape
        coords={"fold":np.arange(f) if fold_ids is None else np.asarray(fold_ids),
                "trait":np.arange(t).astype(str) if trait_names is None else np.asarray(trait_names,dtype=str),
                "selected_feature":np.arange(k)}
        ds=annotate_dataset(xr.Dataset({"feature_id":(("fold","trait","selected_feature"),ids.astype(str)),
                       "coefficient":(("fold","trait","selected_feature"),coef)},coords=coords),"expert_selection")
        return self.write_dataset(ds,group,mode=mode)

    def json_manifest(self):
        """Return the json manifest result computed by ``json.dumps``.

        Returns
        -------
        object
            Computed result from ``json.dumps({'path': s.path, 'schema_version': s.schema_version, 'write_state': s.write_state, 'groups': list(s.groups), 'arrays': list(s.arrays)}, indent=2)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        s=self.summary()
        return json.dumps({
            "path":s.path, "schema_version":s.schema_version, "write_state":s.write_state,
            "groups":list(s.groups), "arrays":list(s.arrays),
        }, indent=2)


def open_store(path, *, mode="a", lazy=True, storage_options=None) -> VigorStore:
    """Open store.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    mode : str
        Input supplied to ``VigorStore`` as ``mode``. See that callable's documented ``mode``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``'a'``.

    lazy : bool
        Whether store-backed values remain lazily loaded until their blocks are accessed; eager
        mode materializes the arrays requested by the reader. Default: ``True``. Passed to
        ``VigorStore`` as ``lazy``.

    storage_options : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Default: ``None``. Passed to ``VigorStore`` as ``storage_options``.

    Returns
    -------
    object
        Computed result from ``VigorStore(path, mode=mode, lazy=lazy, storage_options=storage_options)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return VigorStore(path, mode=mode, lazy=lazy, storage_options=storage_options)


DominoStore = VigorStore  # Temporary public alias; also resolves historical pickle globals.

__all__ = ["SCHEMA_VERSION", "StoreSummary", "VigorStore", "DominoStore", "open_store", "sample_hash", "array_hash", "canonical_provenance", "annotate_dataset", "finalize_zarr_artifact"]
