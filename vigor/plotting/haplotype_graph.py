"""Genome-ordered hvPlot/Datashader views of a local radix DAG."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class HaplotypeGraphPlotData:
    """HaplotypeGraphPlotData represents haplotype graph plot data.

    Attributes
    ----------
    nodes : pd.DataFrame
        Stored nodes; its axis/order follows the construction inputs.
    transitions : pd.DataFrame
        Stored transitions; its axis/order follows the construction inputs.
    ld_pairs : pd.DataFrame
        Stored ld pairs; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    nodes : pd.DataFrame
        Stored nodes supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    transitions : pd.DataFrame
        Stored transitions supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    ld_pairs : pd.DataFrame
        Stored ld pairs supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Required
        construction field.
    """
    nodes: pd.DataFrame
    transitions: pd.DataFrame
    ld_pairs: pd.DataFrame
    metadata: dict


def prepare_haplotype_graph_plot(graph, *, layout_depth=4, include_ld=True):
    """Prepare bounded plot columns without copying animal/path membership lists.

    X is genomic Mb. Y is a deterministic nested display lane from radix layer
    labels, truncated to layout_depth. Shared suffixes use the mean of their
    distinct lanes. This is a layout, not ancestry, recombination or an effect.
    include_ld=False avoids materializing the LD-pair display table for network layouts.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    layout_depth : int
        Input supplied to ``meta.update`` as ``layout_depth``. See that callable's documented
        ``layout_depth`` contract for its type, alignment and value domain. This wrapper retains
        that contract. Default: ``4``. Source comparisons: ``layout_depth < 1``. See the
        rejection/fallback branches for enforcement. Passed to ``meta.update`` as
        ``layout_depth``.

    include_ld : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``HaplotypeGraphPlotData(nodes, transitions, ld_pairs, meta)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if graph.graph.get('schema')!='domino.haplotype_radix_dag.v1':
        raise ValueError('expected a VIGOR haplotype radix DAG')
    if not isinstance(layout_depth,int) or layout_depth<1:
        raise ValueError('layout_depth must be a positive integer')
    v=graph.graph['variants'];anchor=graph.graph.get('radix_anchor_variants')
    anchored=anchor is not None and len(v)>anchor
    depths={};contexts={}
    for node,a in graph.nodes(data=True):
        layers=a.get('nesting_layers') or ('1',)
        keys=set()
        for label in layers:
            pieces=str(label).split('.',layout_depth+2)
            key=tuple(int(x) for x in pieces[(2 if anchored else 1):][:layout_depth])
            keys.add(key)
            for i,child in enumerate(key):depths[key[:i]]=max(depths.get(key[:i],0),child)
        contexts[node]=keys
    def center(key):
        """Find the display center of the selected haplotype graph node.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``lo + width / 2``; see the summary and the called implementation for its labeled/numerical representation.
        """
        lo,width=0.,1.
        for i,child in enumerate(key):
            width/=max(depths.get(key[:i],1),1);lo+=(child-1)*width
        return lo+width/2
    rows=[]
    for node,a in graph.nodes(data=True):
        lanes=[center(key) for key in contexts[node]]
        start,end=int(a['start']),int(a['end'])
        rows.append({'node':node,'x_mb':(start+max(start,end))/2e6,
            'lane':float(np.mean(lanes)),'lane_min':min(lanes),'lane_max':max(lanes),
            'start':start,'end':end,'n_samples':int(a['n_samples']),
            'n_founders':int(a.get('n_founders',0)),'n_variants':len(a['variants']),
            'variants':', '.join(map(str,a['variants'][:4]))+(' …' if len(a['variants'])>4 else ''),
            'genes':', '.join(map(str,a['genes'][:6])),
            'ld_groups':', '.join(map(str,a['ld_groups'][:4]))+(' …' if len(a['ld_groups'])>4 else ''),
            'layers':', '.join(map(str,a['nesting_layers'][:4]))+(' …' if len(a['nesting_layers'])>4 else ''),
            'n_layer_contexts':len(a['nesting_layers']),
            'sequence_status':a['sequence_status']})
    nodes=pd.DataFrame(rows).set_index('node')
    source,target=zip(*graph.edges()) if graph.number_of_edges() else ([],[])
    source=np.asarray(source);target=np.asarray(target)
    # NaN-delimited polylines never connect unrelated edges.
    x=np.full(3*len(source),np.nan);y=np.full_like(x,np.nan)
    if len(source):
        x[0::3]=nodes.loc[source,'x_mb'];x[1::3]=nodes.loc[target,'x_mb']
        y[0::3]=nodes.loc[source,'lane'];y[1::3]=nodes.loc[target,'lane']
    transitions=pd.DataFrame({'x_mb':x,'lane':y})
    if include_ld:
        pairs=graph.graph['ld_edges'];positions=v.pos.to_numpy(float)/1e6
        ld_pairs=pd.DataFrame({'x_mb':positions[pairs.i.to_numpy(int)],
            'y_mb':positions[pairs.j.to_numpy(int)],'r2':pairs.r2.to_numpy(float)})
    else:ld_pairs=pd.DataFrame(columns=['x_mb','y_mb','r2'])
    meta={k:graph.graph[k] for k in ('chrom','start','end','n_samples','path_kind','ld_scope','ld_threshold')}
    meta.update(n_nodes=len(nodes),n_edges=len(source),n_ld_pairs=len(ld_pairs),
                layout_depth=layout_depth,layout_interpretation='nested radix display lanes; not effects or ancestry')
    return HaplotypeGraphPlotData(nodes,transitions,ld_pairs,meta)


def plot_haplotype_radix_dag(graph, *, width=1200, height=480, ld_height=300,
                            layout_depth=4, hover_max_nodes=1500, show_ld=True,
                            dynamic=True, title=None, save_path=None):
    """Datashade all DAG edges/nodes and separately display retained LD pairs.

    Node color is max animal count per pixel, on a fixed linear scale. Edge
    brightness is overlapping transition density. At most hover_max_nodes raw
    nodes are sent to the browser for detailed hover; every node is still shaded.
    Adaptive re-shading needs a live notebook/Panel Python session. A standalone
    HTML export retains pan/zoom/hover on the initially rendered raster.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``1200``. Source comparisons: ``width < 100``. See
        the rejection/fallback branches for enforcement. Passed to ``hover.hvplot.scatter`` as
        ``width``.

    height : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``480``. Source comparisons: ``height < 100``. See
        the rejection/fallback branches for enforcement. Passed to ``hover.hvplot.scatter`` as
        ``height``.

    ld_height : int
        Input supplied to ``data.ld_pairs.hvplot.scatter`` as ``height``. See that callable's
        documented ``height`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``300``. Source comparisons: ``ld_height < 100``. See
        the rejection/fallback branches for enforcement. Passed to
        ``data.ld_pairs.hvplot.scatter`` as ``height``.

    layout_depth : int
        Input supplied to ``prepare_haplotype_graph_plot`` as ``layout_depth``. See that
        callable's documented ``layout_depth`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``4``. Passed to
        ``prepare_haplotype_graph_plot`` as ``layout_depth``.

    hover_max_nodes : int
        Input for the operation ``hover_max_nodes // 4``. See
        ``vigor.plotting.haplotype_graph.plot_haplotype_radix_dag`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``1500``. Source comparisons: ``hover_max_nodes < 0``. See the
        rejection/fallback branches for enforcement.

    show_ld : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    dynamic : bool
        Input supplied to ``data.ld_pairs.hvplot.scatter`` as ``dynamic``. See that callable's
        documented ``dynamic`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``True``. Passed to ``data.ld_pairs.hvplot.scatter`` as
        ``dynamic``.

    title : str
        Displayed title, status or descriptive text attached to the visualization/progress
        output. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``plot``; see the summary and the called implementation for its labeled/numerical representation.
    """
    try:
        import hvplot.pandas  # noqa: F401
        import holoviews as hv
        import datashader as ds
        from bokeh.resources import INLINE
    except ImportError as exc:
        raise ImportError('install VIGOR[plot], including datashader, for graph figures') from exc
    if width<100 or height<100 or ld_height<100 or hover_max_nodes<0:
        raise ValueError('invalid plot dimensions or hover limit')
    data=prepare_haplotype_graph_plot(graph,layout_depth=layout_depth)
    nodes=data.nodes.reset_index();m=data.metadata
    limits=(m['start']/1e6,m['end']/1e6)
    shared=dict(x='x_mb',y='lane',width=width,height=height,xlim=limits,
                ylim=(-.025,1.025),dynamic=dynamic)
    edges=data.transitions.hvplot.line(**shared,datashade=True,aggregator=ds.count(),
        cmap=['#d8e2ea','#647a90','#223f56'],cnorm='log',line_width=1)
    counts=nodes.hvplot.scatter(**shared,datashade=True,aggregator=ds.max('n_samples'),
        cmap='viridis',clim=(0,max(m['n_samples'],1)),cnorm='linear',dynspread=True,max_px=2)
    plot=edges*counts
    if hover_max_nodes:
        if len(nodes)>hover_max_nodes:
            # Coverage extremes plus a deterministic spread of all nodes.
            major=nodes.nlargest(max(1,hover_max_nodes//4),'n_samples').index.to_numpy()
            spread=np.linspace(0,len(nodes)-1,max(1,hover_max_nodes-len(major)),dtype=int)
            hover=nodes.loc[np.unique(np.r_[major,spread])[:hover_max_nodes]]
        else:hover=nodes
        hoverplot=hover.hvplot.scatter(x='x_mb',y='lane',width=width,height=height,
            color='n_samples',cmap='viridis',clim=(0,max(m['n_samples'],1)),colorbar=True,
            size=5,alpha=.65,hover_cols=['node','start','end','n_samples','n_founders',
                'n_variants','variants','genes','ld_groups','layers','n_layer_contexts','sequence_status'])
        plot=plot*hoverplot
    label=title or f"chr{m['chrom']} • {m['path_kind']} • {m['n_samples']:,} animals • {m['n_nodes']:,} nodes"
    plot=plot.opts(title=label,xlabel=f"Chromosome {m['chrom']} position (Mb)",
        ylabel='Radix branch lane (display)',toolbar='above',bgcolor='white',
        show_grid=False,fontsize={'title':13,'labels':11,'ticks':10})
    if show_ld and len(data.ld_pairs):
        ld=data.ld_pairs.hvplot.scatter(x='x_mb',y='y_mb',datashade=True,
            aggregator=ds.max('r2'),cmap='magma',clim=(m['ld_threshold'],1),cnorm='linear',
            width=width,height=ld_height,xlim=limits,ylim=limits,dynamic=dynamic,
            title=f"Retained LD pairs • {m['ld_scope']} • r² ≥ {m['ld_threshold']:.2f} • {m['n_ld_pairs']:,} pairs",
            xlabel='Variant i position (Mb)',ylabel='Variant j position (Mb)')
        plot=(plot+ld).cols(1).opts(shared_axes=False)
    if save_path:
        save_path=Path(save_path)
        if save_path.suffix.lower()!='.html':raise ValueError('save_path must be an HTML file')
        hv.save(plot,str(save_path),resources=INLINE)
    return plot
