"""Bounded-memory append-only Zarr v3 writing for exact score blocks."""
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def write_score_stream(blocks: Iterable, path, *, n_variants=None, variant_chunk=256,
                       trait_chunk=32, shard_variants=4096, consolidated=True):
    """Append canonical score blocks into a genuinely sharded Zarr v3 store.

    Parameters
    ----------
    blocks : iterable of xarray.Dataset
        Same-schema results from score_block/score_design_block. Every variable
        is variant-by-trait or variant-only; variant indices must be strictly
        increasing globally. Trait coordinates and metadata schema stay fixed.
        The iterator is consumed once. No scan-wide concatenation is performed.
    path : str or pathlib.Path
        New local store directory. Existing destinations are rejected; an
        interrupted store remains marked ``domino_write_state='writing'``.
    n_variants : nonnegative int or None
        Optional expected count checked before finalization. No variant data is
        preallocated. A mismatch leaves an explicitly incomplete store.
    variant_chunk, trait_chunk : positive int
        Inner Zarr chunk extents; the final chunk may be partial.
    shard_variants : positive int
        Outer variant shard extent, an exact multiple of variant_chunk. Zarr's
        create_array(shards=...) is used directly; sharding is never ignored.
    consolidated : bool
        Consolidate final v3 metadata before opening the lazy Dataset.

    Returns
    -------
    xarray.Dataset
        Lazy open_zarr view of the completed result. Fixed-size Unicode is not
        used: all string coordinates are variable-length UTF-8, so later blocks
        cannot truncate longer SNP IDs, alleles, chromosome or trait labels.

    Complexity
    ----------
    O(total output cells) write work and O(one input block + shard buffering)
    memory. The writer stores only trait coordinates and one block in Python.
    Partial-shard appends may rewrite that shard; feed shard-sized blocks when
    practical. It never buffers the complete result or all variant labels.
    """
    import xarray as xr
    import zarr
    from ..store import canonical_provenance

    vc = _positive_integer(variant_chunk, "variant_chunk")
    tc = _positive_integer(trait_chunk, "trait_chunk")
    sv = _positive_integer(shard_variants, "shard_variants")
    if sv % vc:
        raise ValueError("shard_variants must be an exact multiple of variant_chunk")
    if n_variants is not None and (isinstance(n_variants, bool) or
            not isinstance(n_variants, (int, np.integer)) or n_variants < 0):
        raise ValueError("n_variants must be a nonnegative integer or None")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"score destination already exists: {destination}")
    iterator = iter(blocks)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("score stream yielded no blocks; supply an explicit empty Dataset to write an empty scan") from exc
    if not isinstance(first, xr.Dataset) or "variant" not in first.dims or "trait" not in first.dims:
        raise TypeError("score blocks must be xarray Datasets with variant/trait dimensions")
    if "variant" not in first.coords or "trait" not in first.coords:
        raise ValueError("score blocks need explicit variant and trait coordinates")
    if len(first.trait) < 1:
        raise ValueError("score blocks need at least one tested effect")
    traits = np.asarray(first.trait.values).astype(str)
    if len(np.unique(traits)) != len(traits):
        raise ValueError("score trait coordinates must be unique")
    schema = {name: tuple(variable.dims) for name, variable in first.variables.items()}
    kinds = {name: variable.dtype.kind for name, variable in first.variables.items()}
    for name, dims in schema.items():
        if dims not in (("variant", "trait"), ("variant",), ("trait",)):
            raise ValueError(f"unsupported streamed variable dimensions for {name}: {dims}")
        if dims == ("trait",) and name != "trait":
            raise ValueError("only the trait dimension coordinate may be trait-only")
        if kinds[name] not in "biufUSO":
            raise TypeError(f"unsupported streamed dtype for {name}: {first[name].dtype}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(destination), mode="w-", zarr_format=3)
    attrs = _jsonable(dict(first.attrs))
    attrs.update(canonical_provenance("mvgwas_result", write_state="writing"))
    attrs.update({"expected_variant_count": None if n_variants is None else int(n_variants),
                  "written_variant_count": 0, "streamed": True,
                  "inner_variant_chunk": vc, "outer_variant_shard": sv})
    coordinate_names = [name for name in first.coords if name not in {"variant", "trait"}]
    if coordinate_names:
        attrs["coordinates"] = " ".join(coordinate_names)
    group.attrs.update(attrs)
    arrays = {}
    for name, dims in schema.items():
        variable = first[name]
        dtype = "str" if kinds[name] in "USO" else variable.dtype
        shape = tuple(0 if dim == "variant" else len(traits) for dim in dims)
        chunks = tuple(vc if dim == "variant" else min(tc, len(traits)) for dim in dims)
        kwargs = {"shape": shape, "dtype": dtype, "chunks": chunks,
                  "dimension_names": dims}
        if "variant" in dims:
            kwargs["shards"] = tuple(sv if dim == "variant" else chunk for dim, chunk in zip(dims, chunks))
        # Use no fill-value masking: legitimate zero statistics/indices and empty
        # string values must not turn into missing coordinates in xarray.
        kwargs["fill_value"] = None
        array = group.create_array(name, **kwargs)
        array.attrs.update(_jsonable(dict(variable.attrs)))
        array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
        arrays[name] = array
        if dims == ("trait",):
            array[:] = traits
    written = 0
    last_index = -1

    def append(block):
        nonlocal written, last_index
        if not isinstance(block, xr.Dataset) or {name: tuple(v.dims) for name, v in block.variables.items()} != schema:
            raise ValueError("score stream schema changed between blocks")
        if not np.array_equal(np.asarray(block.trait.values).astype(str), traits):
            raise ValueError("score stream trait coordinates changed between blocks")
        # LOCO factor directions can differ between groups. Persist each group's
        # declared axes, rather than copying only the first chromosome's attrs.
        incoming_metadata = _jsonable(block.attrs.get("scan_group_metadata", {}))
        existing_metadata = dict(group.attrs.get("scan_group_metadata", {}))
        for key, value in incoming_metadata.items():
            if key in existing_metadata and existing_metadata[key] != value:
                raise ValueError(f"scan group metadata changed within group {key}")
            existing_metadata[key] = value
        if existing_metadata:
            group.attrs["scan_group_metadata"] = existing_metadata
        indices = np.asarray(block.variant.values)
        if indices.dtype.kind not in "iu" or np.any(indices < 0):
            raise ValueError("streamed variant indices must be nonnegative integers")
        if len(indices) and (indices[0] <= last_index or np.any(np.diff(indices) <= 0)):
            raise ValueError("streamed variant indices must be strictly increasing globally")
        stop = written + len(indices)
        if n_variants is not None and stop > n_variants:
            raise ValueError("score stream exceeded expected variant count")
        # Validate the entire incoming block before extending any stored array.
        values = {}
        for name, dims in schema.items():
            if "variant" not in dims:
                continue
            incoming = np.asarray(block[name].values)
            if kinds[name] in "USO":
                incoming = incoming.astype(str)
            elif not np.can_cast(incoming.dtype, arrays[name].dtype, casting="safe"):
                raise TypeError(f"score stream dtype widened for {name}")
            values[name] = incoming
        for name, incoming in values.items():
            array = arrays[name]
            shape = list(array.shape)
            axis = schema[name].index("variant")
            shape[axis] = stop
            array.resize(tuple(shape))
            region = [slice(None)] * len(shape)
            region[axis] = slice(written, stop)
            array[tuple(region)] = incoming
        written = stop
        if len(indices):
            last_index = int(indices[-1])
        group.attrs["written_variant_count"] = written

    append(first)
    for block in iterator:
        append(block)
    if n_variants is not None and written != n_variants:
        raise ValueError(f"score stream wrote {written} variants, expected {n_variants}; store remains incomplete")
    group.attrs["domino_write_state"] = "complete"
    group.attrs["written_variant_count"] = written
    try:
        if consolidated:
            zarr.consolidate_metadata(str(destination))
    except Exception:
        group.attrs["domino_write_state"] = "writing"
        raise
    return xr.open_zarr(str(destination), consolidated=bool(consolidated), chunks={},
                        mask_and_scale=False, create_default_indexes=False)


__all__ = ["write_score_stream"]
