"""Labeled ``xarray.Dataset`` results and consolidated, sharded Zarr v3 persistence.

Every terminal VIGOR function returns an ``xarray.Dataset`` (the
``npplink.mvGWAS_xarray`` convention):

* arrays are named data variables on labeled dimensions (``sample``, ``trait``,
  ``trait_i``/``trait_j``, ``snp``, ``component``, ...), each carrying
  ``long_name``/``units``/``description`` attributes;
* scalars, diagnostics, run parameters and provenance live in ``ds.attrs``;
* non-array members (fitted estimators, operators, nested specifications) stay
  in ``ds.attrs`` in memory and are written as JSON (or their ``repr``) on save.

``save_result`` writes a Dataset as Zarr v3 with sharded chunks and
consolidated metadata, finalizes the artifact manifest, and returns the store
reopened lazily with ``xr.open_zarr(path, consolidated=True)``. Appending along
a dimension (block-wise writers) reuses the layout chosen by the first write.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np

__all__ = ["result_dataset", "to_dataarray", "table_vars", "result_table", "as_frame", "save_result", "open_result", "maybe_save"]

_SCALAR = (str, bytes, bool, int, float, complex, np.generic, type(None))


def _xr():
    import xarray as xr
    return xr


def to_dataarray(value, dims=None, name=None, attrs=None):
    """Convert a pandas/NumPy/xarray value into a labeled ``DataArray``.

    DataFrames use ``dims`` (default: their index/column names, else
    ``('row', 'column')``) and keep index/column labels as coordinates; Series
    and 1-D arrays become one-dimensional; NumPy arrays require ``dims`` when
    they have more than one axis.
    """
    import pandas as pd
    xr = _xr()
    if isinstance(value, xr.DataArray):
        da = value if dims is None else value.rename(dict(zip(value.dims, dims)))
    elif isinstance(value, pd.DataFrame):
        d = tuple(dims) if dims is not None else (value.index.name or "row", value.columns.name or "column")
        da = xr.DataArray(value.to_numpy(), dims=d,
                          coords={d[0]: np.asarray(value.index).astype(str) if value.index.dtype == object else value.index.to_numpy(),
                                  d[1]: np.asarray(value.columns).astype(str) if value.columns.dtype == object else value.columns.to_numpy()})
    elif isinstance(value, pd.Series):
        d = tuple(dims) if dims is not None else (value.index.name or "row",)
        da = xr.DataArray(value.to_numpy(), dims=d,
                          coords={d[0]: np.asarray(value.index).astype(str) if value.index.dtype == object else value.index.to_numpy()})
    else:
        arr = np.asarray(value)
        if dims is None:
            if arr.ndim > 1:
                raise ValueError(f"{name or 'array'} has {arr.ndim} axes; pass dims")
            dims = ("row",) if arr.ndim == 1 else ()
        da = xr.DataArray(arr, dims=tuple(dims))
    if name is not None:
        da = da.rename(name)
    if attrs:
        da = da.assign_attrs(attrs)
    return da


def _is_array_like(value):
    import pandas as pd
    xr = _xr()
    if isinstance(value, (pd.DataFrame, pd.Series, xr.DataArray)):
        return True
    return isinstance(value, np.ndarray) and value.ndim >= 1 and value.dtype != object


def result_dataset(values, *, dims=None, var_attrs=None, coords=None, attrs=None,
                   title=None, software=None, parameters=None):
    """Build a labeled result Dataset from a mapping of named outputs.

    Parameters
    ----------
    values : mapping
        Named outputs. Arrays (pandas, NumPy with ``dims`` given, xarray) become
        data variables; scalars and everything else (dicts, lists, fitted
        objects) go to ``ds.attrs`` under the same name. ``None`` is dropped.
    dims : mapping or None
        Per-name dimension names, required for multi-axis NumPy arrays and
        optional for pandas (overrides index/column names).
    var_attrs : mapping or None
        Per-variable attributes such as ``long_name``, ``units``, ``description``.
    coords : mapping or None
        Extra coordinates (e.g. per-snp ``chrom``/``pos``), passed to ``assign_coords``.
    attrs : mapping or None
        Additional dataset attributes (diagnostics, settings).
    title, software : str or None
        Recorded in ``ds.attrs`` with the creation time and package version.
    parameters : mapping or None
        Call parameters, stored under ``ds.attrs['parameters']``.

    Returns
    -------
    xarray.Dataset
    """
    xr = _xr()
    from . import __version__
    dims = dict(dims or {}); var_attrs = dict(var_attrs or {})
    data, extra = {}, {}
    for key, val in dict(values).items():
        if val is None:
            continue
        if _is_array_like(val):
            if isinstance(val, np.ndarray) and val.ndim > 1 and key not in dims:
                raise ValueError(f"result_dataset: {key!r} is a {val.ndim}-D array; give its dims")
            data[str(key)] = to_dataarray(val, dims=dims.get(key), name=str(key), attrs=var_attrs.get(key))
        elif isinstance(val, (np.generic, np.ndarray)) and np.ndim(val) == 0:
            extra[str(key)] = np.asarray(val).item()
        else:
            extra[str(key)] = val
    ds = xr.Dataset(data)
    if coords:
        ds = ds.assign_coords(coords)
    meta = {"title": title, "software": software, "vigor_version": __version__,
            "created": datetime.now().isoformat(timespec="seconds")}
    ds.attrs.update({k: v for k, v in meta.items() if v is not None})
    if parameters is not None:
        ds.attrs["parameters"] = dict(parameters)
    ds.attrs.update(extra)
    if attrs:
        ds.attrs.update(attrs)
    return ds


def table_vars(df, table, dim, *, prefix="", long_names=None):
    """Long table -> ``{name: DataArray}`` along ``dim``, one variable per column.

    Use for edge lists, rankings and other heterogeneous tables. Each variable
    records ``table`` (the former result key) and ``column`` in its attrs so
    :func:`result_table` rebuilds the table. A named or non-integer index
    becomes the ``dim`` coordinate; positional integer indexes are dropped.
    """
    import pandas as pd
    xr = _xr()
    df = pd.DataFrame(df); idx = df.index; ln = dict(long_names or {})
    keep = idx.name is not None or not pd.api.types.is_integer_dtype(idx.dtype)
    coords = {dim: np.asarray(idx).astype(str) if idx.dtype == object else idx.to_numpy()} if keep else None
    return {f"{prefix}{c}": xr.DataArray(df[c].to_numpy(), dims=(dim,), coords=coords,
                                         attrs={"long_name": ln.get(c, str(c).replace("_", " ")),
                                                "table": str(table), "column": str(c)})
            for c in df.columns}


def result_table(ds, table):
    """Rebuild the long table stored by :func:`table_vars` under the former key ``table``."""
    import pandas as pd
    names = [k for k, v in ds.data_vars.items() if v.attrs.get("table") == table and v.ndim == 1]
    if not names:
        return pd.DataFrame()
    frame = ds[names].to_dataframe()[names]
    frame.columns = [ds[k].attrs.get("column", k) for k in names]
    return frame if ds[names[0]].dims[0] in ds.indexes else frame.reset_index(drop=True)


_table_vars, _result_table = table_vars, result_table


def as_frame(da):
    """1-D/2-D ``DataArray`` -> pandas, restoring row labels kept as coordinates.

    Non-dimension coordinates along the first (row) dimension, e.g. ``trait1``/
    ``trait2`` on a positional ``trait_pair`` axis, become the index (a
    MultiIndex when there are several), as in the original pandas table.
    """
    import pandas as pd
    obj = da.to_pandas()
    row = da.dims[0]
    extra = [c for c, v in da.coords.items() if c not in da.dims and v.dims == (row,)]
    if extra:
        arrays = [np.asarray(da[c].values) for c in extra]
        obj.index = (pd.MultiIndex.from_arrays(arrays, names=extra) if len(extra) > 1
                     else pd.Index(arrays[0], name=extra[0]))
    return obj


def _json_attrs(attrs):
    from .store import _json_safe
    return {str(k): _json_safe(v) for k, v in dict(attrs).items() if not callable(v)}


def _storable(ds):
    """Copy with JSON-safe attrs and fixed-width string coordinates for Zarr."""
    out = ds.copy(deep=False)
    out.attrs = _json_attrs(out.attrs)
    for name in list(out.variables):
        out[name].attrs = _json_attrs(out[name].attrs)
    for key, a in out.coords.items():
        if a.dtype.kind in "OTU":
            out = out.assign_coords({key: (a.dims, np.asarray(a.values, dtype=object).astype(str), a.attrs)})
    for key, a in out.data_vars.items():
        if a.dtype.kind in "OT":
            out[key] = (a.dims, np.asarray(a.values, dtype=object).astype(str), a.attrs)
    return out


def _encoding(ds, chunks, shards):
    """Per-variable ``chunks``/``shards``; mappings are keyed by dimension name."""
    from .store import sharded_dataset_encoding
    enc = sharded_dataset_encoding(ds)
    if chunks is None and shards is None:
        return enc
    chunks, shards = dict(chunks or {}), dict(shards or {})
    for name, da in ds.variables.items():
        if not da.dims:
            continue
        c = tuple(min(int(chunks.get(d, enc[name]["chunks"][i])), int(n)) or 1
                  for i, (d, n) in enumerate(zip(da.dims, da.shape)))
        spec = {"chunks": c}
        if name in ds.data_vars:
            base = enc[name].get("shards", c)
            # Shards must be whole multiples of the chunk along every axis.
            spec["shards"] = tuple(-(-int(shards.get(d, base[i])) // ci) * ci for i, (d, ci) in enumerate(zip(da.dims, c)))
        enc[name] = spec
    return enc


def save_result(ds, path, *, chunks=None, shards=None, mode="w", append_dim=None,
                overwrite=True, artifact_type=None):
    """Write ``ds`` as consolidated, sharded Zarr v3 and return it reopened lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Result Dataset; attrs are written as JSON (objects as their ``repr``).
    path : str or os.PathLike
        Destination ``.zarr`` directory.
    chunks, shards : mapping or None
        Optional ``{dim: size}`` overrides; defaults come from
        ``store.sharded_dataset_encoding`` (~8 MiB chunks, ~32 MiB shards).
        Shards are rounded up to whole chunks.
    mode : {'w', 'a'}
        ``'w'`` creates (or replaces, when ``overwrite``) the store; ``'a'`` with
        ``append_dim`` appends a block along that dimension using the existing layout.
    append_dim : str or None
        Dimension to append along when ``mode='a'``.
    overwrite : bool
        Replace an existing store when ``mode='w'``.
    artifact_type : str or None
        Recorded in the store manifest; defaults to ``ds.attrs['title']``.

    Returns
    -------
    xarray.Dataset
        ``xr.open_zarr(path, consolidated=True)``.
    """
    xr = _xr()
    from .store import finalize_zarr_artifact
    path = Path(path)
    if mode == "w" and path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _storable(ds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        if mode == "a":
            if append_dim is None:
                raise ValueError("mode='a' requires append_dim")
            out.to_zarr(path, mode="a", append_dim=append_dim, zarr_format=3, consolidated=False)
        elif mode == "w":
            out.to_zarr(path, mode="w", zarr_format=3, consolidated=False, encoding=_encoding(out, chunks, shards))
        else:
            raise ValueError("mode must be 'w' or 'a'")
    finalize_zarr_artifact(path, artifact_type=artifact_type or out.attrs.get("title"), metadata=out.attrs)
    return xr.open_zarr(path, consolidated=True)


def maybe_save(ds, save_path, **kwargs):
    """Return ``save_result(ds, save_path, **kwargs)`` when ``save_path`` is given, else ``ds``."""
    return ds if save_path is None else save_result(ds, save_path, **kwargs)


def open_result(path, *, chunks=None):
    """Open a saved VIGOR result lazily (consolidated Zarr v3)."""
    return _xr().open_zarr(path, consolidated=True, chunks=chunks)
