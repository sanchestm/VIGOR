"""Sparse typed-graph helpers."""
from __future__ import annotations
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

def edge_table_to_csr(edges,*,nodes=None,cost_col='cost',directed=False):
    """Construct aligned sparse adjacency from a table of weighted node edges.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    nodes : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value. Passed to ``sparse.coo_matrix`` as ``shape``.

    cost_col : str
        Input for the operation ``float(getattr(x, cost_col, 1.0))``. See
        ``vigor.network.sparse.edge_table_to_csr`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'cost'``.

    directed : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``(A, nodes, idx)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=edges.copy();
    if nodes is None:nodes=sorted(set(d.source.astype(str))|set(d.target.astype(str)))
    nodes=list(map(str,nodes));idx={n:i for i,n in enumerate(nodes)};r=[];c=[];v=[]
    for x in d.itertuples(index=False):
        a,b=str(x.source),str(x.target)
        if a not in idx or b not in idx:continue
        cost=float(getattr(x,cost_col,1.0));r.append(idx[a]);c.append(idx[b]);v.append(cost)
        if not directed:r.append(idx[b]);c.append(idx[a]);v.append(cost)
    A=sparse.coo_matrix((v,(r,c)),shape=(len(nodes),len(nodes))).tocsr();return A,nodes,idx

def network_proximity(edges,set_a,set_b,*,n_perm=1000,random_state=0,degree_matched=True):
    """Estimate graph distance enrichment between selected node sets under matched null samples.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    set_a : object (see the cited input contract)
        Input for the operation ``map(str, set_a)``. See
        ``vigor.network.sparse.network_proximity`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    set_b : object (see the cited input contract)
        Input for the operation ``map(str, set_b)``. See
        ``vigor.network.sparse.network_proximity`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    n_perm : int
        Number of random permutations used to estimate the null/evidence quantity. Nonnegative
        counts control Monte Carlo work; zero disables permutation where supported. Default:
        ``1000``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    degree_matched : bool
        Input for the operation ``bool(degree_matched)``. See
        ``vigor.network.sparse.network_proximity`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``True``.

    Returns
    -------
    dict
        Result fields: ``observed``, ``null_mean``, ``null_sd``, ``z``, ``empirical_p``, ``degree_matched``. Field arrays preserve the corresponding input axis order.
    """
    A,nodes,idx=edge_table_to_csr(edges);ia=[idx[x] for x in map(str,set_a) if x in idx];ib=[idx[x] for x in map(str,set_b) if x in idx]
    if not ia or not ib:raise ValueError('both node sets need graph nodes')
    D=dijkstra(A,indices=ia,directed=False);obs=float(np.nanmean(np.nanmin(D[:,ib],axis=1)))
    rng=np.random.default_rng(random_state);deg=np.diff(A.indptr);null=[]
    for _ in range(int(n_perm)):
        if degree_matched:
            samp=[]
            for j in ib:
                q=np.flatnonzero((np.abs(deg-deg[j])<=max(1,int(.1*max(deg[j],1)))) & (np.arange(len(deg))!=j))
                if len(q)==0: q=np.flatnonzero(np.arange(len(deg))!=j)
                samp.append(int(rng.choice(q)))
        else:samp=rng.choice(len(nodes),len(ib),replace=False)
        null.append(float(np.nanmean(np.nanmin(D[:,samp],axis=1))))
    null=np.asarray(null);z=(obs-null.mean())/max(null.std(ddof=1),1e-12);p=(1+np.sum(null<=obs))/(len(null)+1)
    return {'observed':obs,'null_mean':float(null.mean()),'null_sd':float(null.std(ddof=1)),'z':float(z),'empirical_p':float(p),'degree_matched':bool(degree_matched)}
