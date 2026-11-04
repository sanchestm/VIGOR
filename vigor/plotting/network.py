from __future__ import annotations
import pandas as pd
def plot_network_embedding(embedding:pd.DataFrame,*,edges=None,node_metadata=None,label_nodes=False,width=800,height=650,save_path=None,**kwargs):
    """Plot network embedding.

    Parameters
    ----------
    embedding : pd.DataFrame
        Input for the operation ``embedding.iloc[:, :2]``. See
        ``vigor.plotting.network.plot_network_embedding`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation. Default:
        ``None``.

    node_metadata : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    label_nodes : bool
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``False``.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``800``. Passed to ``d.hvplot.scatter`` as
        ``width``.

    height : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``650``. Passed to ``d.hvplot.scatter`` as
        ``height``.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    **kwargs : mapping
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.

    Returns
    -------
    object
        Computed result from ``p``; see the summary and the called implementation for its labeled/numerical representation.
    """
    try:import hvplot.pandas  # noqa:F401
    except ImportError as e:raise ImportError('install VIGOR[plot] for network figures') from e
    d=embedding.iloc[:,:2].copy();d.columns=['x','y'];d['node']=d.index.astype(str);p=d.hvplot.scatter(x='x',y='y',hover_cols=['node'],width=width,height=height)
    if save_path:
        import holoviews as hv;hv.save(p,str(save_path))
    return p
