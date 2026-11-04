from __future__ import annotations
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from .sparse import edge_table_to_csr

def embed_graph_nodes(graph,*,node_features=None,dimensions=2,method='spectral',umap=False,n_neighbors=15,random_state=0):
    """Return the embed graph nodes result computed by ``pd.DataFrame``.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    node_features : object (see the cited input contract)
        Input for the operation ``node_features.reindex(nodes).apply(pd.to_numeric,
        errors='coerce').fillna(0)``. See ``vigor.network.embedding.embed_graph_nodes`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    dimensions : int
        Input supplied to ``umap_module.UMAP`` as ``n_components``. See that callable's
        documented ``n_components`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``2``. Passed to ``umap_module.UMAP`` as
        ``n_components``.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'spectral'``.

    umap : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    n_neighbors : int
        Input supplied to ``umap_module.UMAP`` as ``n_neighbors``. See that callable's
        documented ``n_neighbors`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``15``. Passed to ``umap_module.UMAP`` as
        ``n_neighbors``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``TruncatedSVD`` as ``random_state``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(coords, index=nodes, columns=[f'dim{i + 1}' for i in range(int(dimensions))])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    edges=graph.edges if hasattr(graph,'edges') and isinstance(graph.edges,pd.DataFrame) else graph
    A,nodes,idx=edge_table_to_csr(edges);raw=A
    if node_features is not None:
        f=node_features.reindex(nodes).apply(pd.to_numeric,errors='coerce').fillna(0);raw=sparse.csr_matrix(f.to_numpy(float))
    k=min(int(dimensions),max(1,min(raw.shape)-1));coords=TruncatedSVD(n_components=k,random_state=random_state).fit_transform(raw)
    if coords.shape[1]<dimensions:coords=np.pad(coords,((0,0),(0,dimensions-coords.shape[1])))
    if umap:
        try:import umap as umap_module
        except ImportError as e:raise ImportError('install VIGOR[embedding] for UMAP') from e
        coords=umap_module.UMAP(n_components=int(dimensions),n_neighbors=min(int(n_neighbors),max(2,len(nodes)-1)),random_state=random_state,n_jobs=1).fit_transform(coords)
    return pd.DataFrame(coords,index=nodes,columns=[f'dim{i+1}' for i in range(int(dimensions))])
