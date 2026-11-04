"""Sparse PCST/PCSF prioritization with explicit null calibration."""
from __future__ import annotations
from dataclasses import dataclass,field
import warnings
import numpy as np,pandas as pd
from scipy.sparse.csgraph import dijkstra
from .sparse import edge_table_to_csr

@dataclass
class NetworkPrioritizationResult:
    """NetworkPrioritizationResult represents network prioritization result.

    Attributes
    ----------
    nodes : pd.DataFrame
        Stored nodes; its axis/order follows the construction inputs.
    edges : pd.DataFrame
        Stored edges; its axis/order follows the construction inputs.
    objective : float
        Stored objective; its axis/order follows the construction inputs.
    method : str
        Stored method; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.
    warnings : list[str]
        Stored warnings; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    nodes : pd.DataFrame
        Stored nodes supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    edges : pd.DataFrame
        Stored edges supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    objective : float
        Stored objective supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    method : str
        Stored method supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.

    warnings : list[str]
        Warnings retained from failed/unsupported optional model stages. Default: a fresh value
        from ``list()`` for each instance.
    """
    nodes: pd.DataFrame
    edges: pd.DataFrame
    objective: float
    method: str
    metadata: dict=field(default_factory=dict)
    warnings: list[str]=field(default_factory=list)

def _transform_prizes(prizes,mode='softplus_cap',cap_quantile=.995):
    """Transform/cap node relevance prizes before network selection.

    Parameters
    ----------
    prizes : array-like or numeric scalar
        Input for the operation ``pd.Series(prizes, dtype=float).replace([np.inf, -np.inf],
        np.nan).fillna(0.0)``. See ``vigor.network.pcsf._transform_prizes`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    mode : str
        Input for the operation ``mode in {'softplus', 'softplus_cap'}``. See
        ``vigor.network.pcsf._transform_prizes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'softplus_cap'``. Source-recognized literal values: ``log1p``, ``none``. These select
        the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    cap_quantile : array-like or numeric scalar
        Input for the operation ``np.quantile(out, float(cap_quantile))``. See
        ``vigor.network.pcsf._transform_prizes`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.995``.
    """
    s=pd.Series(prizes,dtype=float).replace([np.inf,-np.inf],np.nan).fillna(0.0);raw=s.to_numpy(float)
    if mode in {None,'none'}:out=np.clip(raw,0,None)
    elif mode in {'softplus','softplus_cap'}:
        med=np.nanmedian(raw);mad=np.nanmedian(np.abs(raw-med));scale=max(1.4826*mad,1e-8);z=(raw-med)/scale;out=np.log1p(np.exp(np.clip(z,-40,40)))
    elif mode=='log1p':out=np.log1p(np.clip(raw,0,None))
    else:raise ValueError('unknown prize_transform')
    if 'cap' in str(mode):
        cap=np.quantile(out,float(cap_quantile)) if len(out) else 0.;out=np.minimum(out,cap)
    return pd.Series(out,index=s.index),{'transform':mode,'cap_quantile':cap_quantile,'raw_max':float(np.nanmax(raw)) if len(raw) else 0,'transformed_max':float(np.nanmax(out)) if len(out) else 0}

def _greedy_path_union(edges,prizes):
    """Approximate a prize-collecting subnetwork by combining high-value paths.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : object (see the cited input contract)
        Input for the operation ``prizes.reindex(nodes).fillna(0.0)``. See
        ``vigor.network.pcsf._greedy_path_union`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    A,nodes,idx=edge_table_to_csr(edges);ps=prizes.reindex(nodes).fillna(0.0);term=np.flatnonzero(ps.to_numpy()>0);selected=set(term.tolist());sel_edges=set()
    # Connect profitable terminal pairs by shortest paths; transparent baseline only.
    for ai,i in enumerate(term[:-1]):
        dist,pred=dijkstra(A,indices=i,directed=False,return_predecessors=True)
        for j in term[ai+1:]:
            if not np.isfinite(dist[j]):continue
            benefit=float(ps.iloc[i]+ps.iloc[j]-dist[j])
            if benefit<=0:continue
            cur=int(j);path=[]
            while cur!=i and cur>=0:
                prev=int(pred[cur]);
                if prev<0:break
                path.append((prev,cur));cur=prev
            if cur==i:
                selected.update([x for e in path for x in e]);sel_edges.update(tuple(sorted(e)) for e in path)
    nd=pd.DataFrame({'node':[nodes[i] for i in sorted(selected)],'prize':[float(ps.iloc[i]) for i in sorted(selected)]})
    pair={(min(idx[str(r.source)],idx[str(r.target)]),max(idx[str(r.source)],idx[str(r.target)])):r for r in edges.itertuples(index=False) if str(r.source) in idx and str(r.target) in idx}
    er=[]
    for e in sel_edges:
        r=pair.get(e);er.append({'source':nodes[e[0]],'target':nodes[e[1]],'cost':float(getattr(r,'cost',1.0)) if r is not None else 1.0})
    ed=pd.DataFrame(er);cost=float(ed.cost.sum()) if len(ed) else 0.;objective=float(nd.prize.sum()-cost)
    return nd,ed,objective

def _native_pcst(edges,prizes,num_clusters=1,pruning='strong'):
    """Run the optional native prize-collecting Steiner-tree solver on the prepared graph.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : object (see the cited input contract)
        Input for the operation ``prizes.reindex(nodes).fillna(0.0).to_numpy(float)``. See
        ``vigor.network.pcsf._native_pcst`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    num_clusters : array-like or numeric scalar
        Input for the operation ``pcst_fast(np.asarray(pairs, np.int64), p, np.asarray(costs,
        float), -1, int(num_clusters), str(pruning), 0)``. See
        ``vigor.network.pcsf._native_pcst`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``1``.

    pruning : array-like or numeric scalar
        Input for the operation ``pcst_fast(np.asarray(pairs, np.int64), p, np.asarray(costs,
        float), -1, int(num_clusters), str(pruning), 0)``. See
        ``vigor.network.pcsf._native_pcst`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``'strong'``.
    """
    try:from pcst_fast import pcst_fast
    except ImportError as e:raise ImportError('pcst_fast is required for method="pcst_fast"; install VIGOR[network]') from e
    A,nodes,idx=edge_table_to_csr(edges);d=edges.copy();pairs=[];costs=[]
    seen={}
    for r in d.itertuples(index=False):
        a,b=idx.get(str(r.source)),idx.get(str(r.target));
        if a is None or b is None or a==b:continue
        k=tuple(sorted((a,b)));c=float(getattr(r,'cost',1.0));
        if k not in seen or c<seen[k]:seen[k]=c
    for k,c in seen.items():pairs.append(k);costs.append(c)
    p=prizes.reindex(nodes).fillna(0.0).to_numpy(float);verts,eidx=pcst_fast(np.asarray(pairs,np.int64),p,np.asarray(costs,float),-1,int(num_clusters),str(pruning),0)
    nd=pd.DataFrame({'node':[nodes[i] for i in verts],'prize':p[verts]});er=[]
    for k in eidx:
        a,b=pairs[int(k)];er.append({'source':nodes[a],'target':nodes[b],'cost':costs[int(k)]})
    ed=pd.DataFrame(er);return nd,ed,float(nd.prize.sum()-(ed.cost.sum() if len(ed) else 0))

def prize_collecting_steiner_forest(edges,prizes,*,method='pcst_fast',prize_transform='softplus_cap',prize_cap_quantile=.995,**kwargs):
    """Return the prize collecting steiner forest result computed by ``NetworkPrioritizationResult``.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : array-like or numeric scalar
        Input for the operation ``pd.Series(prizes, dtype=float)``. See
        ``vigor.network.pcsf.prize_collecting_steiner_forest`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'pcst_fast'``.
        Source-recognized literal values: ``greedy``, ``legacy``, ``native``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    prize_transform : str
        Input for the operation ``_transform_prizes(ps, prize_transform, prize_cap_quantile)``.
        See ``vigor.network.pcsf.prize_collecting_steiner_forest`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``'softplus_cap'``.

    prize_cap_quantile : float
        Input for the operation ``_transform_prizes(ps, prize_transform, prize_cap_quantile)``.
        See ``vigor.network.pcsf.prize_collecting_steiner_forest`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``0.995``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``_native_pcst``. See those callables for
        accepted keys and values. Passed to ``_native_pcst`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``NetworkPrioritizationResult(nd, ed, obj, method, {'prize_transform': meta}, warns)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    ps=pd.Series(prizes,dtype=float);ps2,meta=_transform_prizes(ps,prize_transform,prize_cap_quantile)
    if method in {'pcst_fast','native'}:nd,ed,obj=_native_pcst(edges,ps2,**kwargs)
    elif method in {'greedy','legacy'}:nd,ed,obj=_greedy_path_union(edges,ps2)
    else:raise ValueError("method must be 'pcst_fast' or 'greedy'")
    warns=[]
    if meta['raw_max']>10*max(np.nanmedian(np.abs(ps.to_numpy())),1e-12):warns.append('extreme raw prizes were saturated before optimization')
    return NetworkPrioritizationResult(nd,ed,obj,method,{'prize_transform':meta},warns)

def prioritize_network(*args,**kwargs):
    """Return the prioritize network result computed by ``prize_collecting_steiner_forest``.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments forwarded, in order, to
        ``prize_collecting_steiner_forest``. See that signature for required scientific inputs
        and their order.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``prize_collecting_steiner_forest``. See those
        callables for accepted keys and values. Passed to ``prize_collecting_steiner_forest`` as
        ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``prize_collecting_steiner_forest(*args, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return prize_collecting_steiner_forest(*args,**kwargs)

def pcsf_stability(edges,prizes,*,n_boot=100,noise_sd=.1,random_state=0,**kwargs):
    """Estimate selected node/edge stability under prize/noise resampling.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : array-like or numeric scalar
        Input for the operation ``pd.Series(prizes, dtype=float)``. See
        ``vigor.network.pcsf.pcsf_stability`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    n_boot : int
        Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
        estimate sampling variation by the resampling scheme used here. Default: ``100``.

    noise_sd : float
        Input supplied to ``rng.normal`` as ``scale``. See that callable's documented ``scale``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``0.1``. Passed to ``rng.normal`` as ``scale``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``prize_collecting_steiner_forest``. See those
        callables for accepted keys and values. Passed to ``prize_collecting_steiner_forest`` as
        ``**expanded mapping``.

    Returns
    -------
    dict
        Result fields: ``node_frequency``, ``objective``, ``kind``. Field arrays preserve the corresponding input axis order.
    """
    rng=np.random.default_rng(random_state);p=pd.Series(prizes,dtype=float);cnt={str(x):0 for x in p.index};obj=[]
    for _ in range(int(n_boot)):
        pp=np.maximum(0,p.to_numpy()+rng.normal(scale=float(noise_sd),size=len(p)));r=prize_collecting_steiner_forest(edges,pd.Series(pp,index=p.index),**kwargs);obj.append(r.objective)
        for n in r.nodes.node.astype(str):cnt[n]=cnt.get(n,0)+1
    return {'node_frequency':pd.Series({k:v/int(n_boot) for k,v in cnt.items()}).sort_values(ascending=False),'objective':np.asarray(obj),'kind':'robustness_not_null'}

def network_null_test(edges,prizes,*,n_perm=1000,random_state=0,method='pcst_fast',null='prize_permutation',
                      degeneracy_tol=1e-12,min_distinct=3,max_rewire_attempts=20,**kwargs):
    """Empirical objective null with explicit degeneracy and rewire diagnostics.

    A failed degree-preserving rewire is never silently replaced by the observed
    graph. Degenerate null distributions return ``p``/``z`` as NaN rather than
    manufacturing a z-score from an arbitrary epsilon denominator.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : array-like or numeric scalar
        Input for the operation ``pd.Series(prizes, dtype=float)``. See
        ``vigor.network.pcsf.network_null_test`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    n_perm : int
        Number of random permutations used to estimate the null/evidence quantity. Nonnegative
        counts control Monte Carlo work; zero disables permutation where supported. Default:
        ``1000``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'pcst_fast'``. Passed to
        ``prize_collecting_steiner_forest`` as ``method``.

    null : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes. Default: ``'prize_permutation'``. Source-recognized literal values:
        ``degree_rewire``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    degeneracy_tol : float
        Input for the operation ``sd <= float(degeneracy_tol)``. See
        ``vigor.network.pcsf.network_null_test`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``1e-12``.

    min_distinct : int
        Input for the operation ``nd < min(int(min_distinct), len(vals))``. See
        ``vigor.network.pcsf.network_null_test`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``3``.

    max_rewire_attempts : int
        Input for the operation ``range(max(1, int(max_rewire_attempts)))``. See
        ``vigor.network.pcsf.network_null_test`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``20``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``prize_collecting_steiner_forest``. See those
        callables for accepted keys and values. Passed to ``prize_collecting_steiner_forest`` as
        ``**expanded mapping``.

    Returns
    -------
    dict
        Result fields: ``observed``, ``null_objective``, ``empirical_p``, ``z``, ``null``, ``null_sd``, ``null_n_distinct``, ``n_valid``, ``failed_rewires``, ``degenerate``, ``warning``. Field arrays preserve the corresponding input axis order.
    """
    rng=np.random.default_rng(random_state);p=pd.Series(prizes,dtype=float)
    obs=prize_collecting_steiner_forest(edges,p,method=method,**kwargs)
    vals=[];failed_rewires=0
    if null not in {'prize_permutation','degree_rewire'}:raise ValueError('null must be prize_permutation or degree_rewire')
    for b in range(int(n_perm)):
        if null=='prize_permutation':
            pe=pd.Series(rng.permutation(p.to_numpy()),index=p.index);ee=edges
        else:
            import networkx as nx
            G=nx.from_pandas_edgelist(edges,'source','target',edge_attr=True,create_using=nx.Graph)
            ee=None
            for _attempt in range(max(1,int(max_rewire_attempts))):
                H=G.copy()
                try:
                    nx.double_edge_swap(H,nswap=max(1,2*H.number_of_edges()),
                                        max_tries=max(100,20*H.number_of_edges()),
                                        seed=int(rng.integers(2**31-1)))
                    # A successful call can still leave a tiny graph unchanged.
                    if set(H.edges()) != set(G.edges()):
                        ee=nx.to_pandas_edgelist(H); break
                except Exception:
                    pass
            if ee is None:
                failed_rewires += 1
                continue
            pe=p
        vals.append(prize_collecting_steiner_forest(ee,pe,method=method,**kwargs).objective)
    vals=np.asarray(vals,float)
    if len(vals)==0:
        return {'observed':obs,'null_objective':vals,'empirical_p':np.nan,'z':np.nan,'null':null,
                'null_sd':np.nan,'null_n_distinct':0,'n_valid':0,'failed_rewires':int(failed_rewires),
                'degenerate':True,'warning':'no valid null replicates'}
    sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
    nd=int(np.unique(np.round(vals,12)).size)
    degenerate=(not np.isfinite(sd)) or sd <= float(degeneracy_tol) or nd < min(int(min_distinct),len(vals))
    if degenerate:
        emp=np.nan; z=np.nan
        warning='degenerate null objective distribution; significance is undefined'
    else:
        emp=(1+np.sum(vals>=obs.objective))/(len(vals)+1)
        z=(obs.objective-vals.mean())/sd
        warning=None
    return {'observed':obs,'null_objective':vals,'empirical_p':float(emp) if np.isfinite(emp) else np.nan,
            'z':float(z) if np.isfinite(z) else np.nan,'null':null,'null_sd':sd,
            'null_n_distinct':nd,'n_valid':int(len(vals)),'failed_rewires':int(failed_rewires),
            'degenerate':bool(degenerate),'warning':warning}

def diffuse_scores(edges,prizes,*,alpha=.85,maxiter=200,tol=1e-10):
    """Return the diffuse scores result computed by ``pd.Series``.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    prizes : object (see the cited input contract)
        Input for the operation ``pd.Series(prizes).reindex(nodes).fillna(0).to_numpy(float)``.
        See ``vigor.network.pcsf.diffuse_scores`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.85``.

    maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``200``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.

    Returns
    -------
    object
        Computed result from ``pd.Series(x, index=nodes, name='diffused_score')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    A,nodes,idx=edge_table_to_csr(edges);A=A.copy();A.data=np.ones_like(A.data);deg=np.asarray(A.sum(1)).ravel();P=A.multiply(1/np.maximum(deg,1)[:,None]);p=pd.Series(prizes).reindex(nodes).fillna(0).to_numpy(float);x=p.copy()
    for _ in range(int(maxiter)):
        xn=(1-alpha)*p+alpha*np.asarray(P@x).ravel()
        if np.linalg.norm(xn-x)<=tol*max(np.linalg.norm(x),1):x=xn;break
        x=xn
    return pd.Series(x,index=nodes,name='diffused_score')

__all__=['NetworkPrioritizationResult','prize_collecting_steiner_forest','prioritize_network','pcsf_stability','network_null_test','diffuse_scores']
