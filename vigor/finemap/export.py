"""Typed, labelled, sharded Zarr3 export of integrated predictive attribution."""
from pathlib import Path
import json
import os
import shutil
import numpy as np
import pandas as pd


def write_joint_finemap_zarr(model, attribution, path, *, producer_provenance=None,
                             serializer_provenance=None, chunk_rows=256, shard_rows=4096, io_workers=4):
    """Export graph, true bubble DAG, LD, and conditional Owen tables.

    Scalar columns keep numeric/bool/string dtypes and explicit null masks.
    Lists use recursive CSR-style offsets and typed leaf values (never repr).
    Every array has named dimensions for xarray. Tables are row-oriented;
    reference columns retain original sample/variant/bubble/trait labels.
    NaN means unsupported/undefined, not a zero effect or independent LD.
    Provenance distinguishes numerical producer from metadata serializer.

    Parameters
    ----------
    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Passed to ``root.attrs.update`` as ``model_metadata``.

    attribution : mapping
        Input supplied to ``root.attrs.update`` as ``value_function``. See that callable's
        documented ``value_function`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Passed to ``root.attrs.update`` as ``value_function``.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    producer_provenance : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value. Passed to ``root.attrs.update`` as
        ``producer_provenance``.

    serializer_provenance : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value. Passed to ``root.attrs.update`` as
        ``serializer_provenance``.

    chunk_rows : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``256``.
        Source comparisons: ``chunk_rows < 1``. See the rejection/fallback branches for
        enforcement. Passed to ``write_haplotype_graph_zarr`` as ``chunk_rows``.

    shard_rows : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``4096``.
        Passed to ``write_haplotype_graph_zarr`` as ``shard_rows``.

    io_workers : int or str
        Concurrency count for the local worker pool. Positive integers select workers; auto,
        where supported, derives a bounded count from hardware/workload. More workers increase
        memory demand. Default: ``4``. Source comparisons: ``io_workers < 1``. See the
        rejection/fallback branches for enforcement.

    Returns
    -------
    object
        Computed result from ``str(dest)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import zarr
    from ..zarr_io import _create_array, _unicode_dtype
    from .graph import write_haplotype_graph_zarr
    from .integrated import bubble_nomenclature
    if io_workers<1:raise ValueError('io_workers must be positive')
    if chunk_rows<1 or shard_rows<chunk_rows or shard_rows%chunk_rows:
        raise ValueError('shard_rows must be a positive multiple of chunk_rows')
    dest=Path(path);temp=dest.with_name(dest.name+'.incomplete')
    if temp.exists():shutil.rmtree(temp)
    temp.parent.mkdir(parents=True,exist_ok=True)
    write_haplotype_graph_zarr(model.graph,temp,chunk_rows=chunk_rows,shard_rows=shard_rows)
    root=zarr.open_group(str(temp),mode='a',zarr_format=3)
    root.attrs.update(schema='domino_joint_radix_finemap_v1',producer_provenance=producer_provenance or {},
        serializer_provenance=serializer_provenance or {},model_metadata=model.metadata,
        value_function={k:attribution.get(k) for k in ['interpretation','uncertainty','hierarchy']},
        ragged_encoding='recursive zero-based offsets per nesting level, typed leaf values; masks distinguish null from empty list',
        write_state='writing',predictive_attribution_not_causal_PIP=True)
    def array(g,name,values,dim):
        """Write one labeled numerical array to the enclosing export destination.

        Parameters
        ----------
        g : array-like or numeric scalar
            Input for the operation ``_create_array(g, name, shape=vals.shape, chunks=(chunk_rows,),
            shards=(shard_rows,), dtype=dtype, dimension_names=(dim,), fill_value=np.nan if
            dtype.kind == 'f' else None)``. See
            ``vigor.finemap.export.write_joint_finemap_zarr.array`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        values : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        dim : str, index or sequence
            Axis/column/feature selector used to identify the input/output quantities in this
            operation. Named selectors must exist in the labeled input; integer selectors are valid
            zero-based positions on that axis. Passed to ``_create_array`` as ``dimension_names``.

        Returns
        -------
        object
            Computed result from ``a``; see the summary and the called implementation for its labeled/numerical representation.
        """
        vals=np.asarray(values)
        if vals.dtype.kind in 'OU':vals=np.asarray(values,dtype=str);dtype=_unicode_dtype(vals)
        else:dtype=vals.dtype
        a=_create_array(g,name,shape=vals.shape,chunks=(chunk_rows,),shards=(shard_rows,),dtype=dtype,
            dimension_names=(dim,),fill_value=np.nan if dtype.kind=='f' else None)
        a[:]=vals
        return a
    def ragged(g,values,dim,depth=0):
        """Serialize variable-length membership/path lists using flat values and offsets.

        Parameters
        ----------
        g : object (see the cited input contract)
            Input for the operation ``g.attrs['encoding']``. See
            ``vigor.finemap.export.write_joint_finemap_zarr.ragged`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        values : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        dim : str, index or sequence
            Axis/column/feature selector used to identify the input/output quantities in this
            operation. Named selectors must exist in the labeled input; integer selectors are valid
            zero-based positions on that axis.

        depth : int
            Input for the operation ``depth + 1``. See
            ``vigor.finemap.export.write_joint_finemap_zarr.ragged`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.
            Default: ``0``.
        """
        offsets=[0];flat=[]
        for v in values:
            flat.extend(v);offsets.append(len(flat))
        array(g,'offsets',np.asarray(offsets,np.int64),dim+'_boundary')
        nested=any(isinstance(v,(tuple,list,np.ndarray)) for v in flat)
        if nested:
            if not all(isinstance(v,(tuple,list,np.ndarray)) for v in flat):raise ValueError('inconsistent ragged nesting')
            ragged(g.require_group('items'),flat,dim+'_item',depth+1)
        else:array(g,'values',flat,dim+'_item')
        g.attrs['encoding']='ragged_offsets';g.attrs['nesting_level']=depth
    def table(name,frame):
        """Serialize the selected tabular result in the enclosing export format.

        Parameters
        ----------
        name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        frame : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation. Passed to ``g.attrs.update`` as ``columns``.
        """
        frame=frame.copy();g=root.require_group('attribution/'+name)
        g.attrs.update(columns=list(map(str,frame.columns)),n_rows=len(frame),index_names=list(map(str,frame.index.names)))
        array(g,'row',np.arange(len(frame),dtype=np.int64),'row')
        index=frame.index
        if isinstance(index,pd.MultiIndex):
            for j in range(index.nlevels):array(g,'index_'+str(j),index.get_level_values(j).astype(str),'row')
        else:array(g,'index',index.astype(str),'row')
        def column_write(column):
            """Persist one table column with its corresponding dimension metadata.

            Parameters
            ----------
            column : str, index or sequence
                Axis/column/feature selector used to identify the input/output quantities in this
                operation. Named selectors must exist in the labeled input; integer selectors are valid
                zero-based positions on that axis.
            """
            series=frame[column];values=series.tolist();cg=g.require_group(str(column))
            missing=np.asarray([v is None or v is pd.NA or (isinstance(v,(float,np.floating)) and np.isnan(v)) for v in values],bool)
            array(cg,'valid',~missing,'row')
            if any(isinstance(v,(tuple,list,np.ndarray)) for v in values):
                ragged(cg,[[] if missing[j] else v for j,v in enumerate(values)],'row')
            elif pd.api.types.is_bool_dtype(series.dtype):array(cg,'values',series.fillna(False).to_numpy(bool),'row')
            elif pd.api.types.is_numeric_dtype(series.dtype):array(cg,'values',series.to_numpy(),'row')
            else:array(cg,'values',['' if missing[j] else str(v) for j,v in enumerate(values)],'row')
        # Distinct column groups have disjoint keys. Parallel I/O amortizes
        # restricted event-loop wakeup latency; no numerical work is threaded.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            list(pool.map(column_write,frame.columns))
    table('bubble_hierarchy',bubble_nomenclature(model.bubbles))
    for name,frame in attribution.items():
        if isinstance(frame,pd.DataFrame):table(name,frame)
    lg=root.require_group('signed_ld')
    names=model.ld['r'].index.astype(str).to_numpy()
    array(lg,'variant_i',names,'variant_i');array(lg,'variant_j',names,'variant_j')
    for name in ['r','nobs','valid']:
        values=model.ld[name].to_numpy();a=_create_array(lg,name,shape=values.shape,
            chunks=(chunk_rows,chunk_rows),shards=(shard_rows,shard_rows),dtype=values.dtype,
            dimension_names=('variant_i','variant_j'),fill_value=np.nan if values.dtype.kind=='f' else None);a[:]=values
    lg.attrs.update(allele_orientation='input labelled dosage alleles',psd_policy=model.ld['psd_policy'])
    root.attrs['write_state']='complete';zarr.consolidate_metadata(str(temp))
    if dest.exists():shutil.rmtree(dest)
    os.replace(temp,dest)
    return str(dest)
