"""Joint graph-feature prediction and empirical conditional haplotype attribution.

Predictive values are not Bayesian causal inclusion probabilities. Donors are
whole observed paths; no independently shuffled SNP mosaics are manufactured.
"""
from __future__ import annotations
import hashlib,time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from .graph import build_haplotype_radix_dag, _correlation


def signed_ld(genotypes, *, block_variants=256, max_variants=4096):
    """Signed pairwise-complete LD, overlap and validity; no forced PSD claim.

    Alleles follow labelled input dosage orientation. Pairwise deletion can
    yield an indefinite matrix. Invalid/monomorphic correlations remain NaN.
    A downstream Gaussian model must choose and disclose its own PSD policy.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    block_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``256``.

    max_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``4096``.

    Returns
    -------
    dict
        Result fields: ``r``, ``nobs``, ``valid``, ``seconds``, ``allocated_result_bytes``, ``psd_policy``. Field arrays preserve the corresponding input axis order.
    """
    if hasattr(genotypes,'dims'):
        sd=next(d for d in genotypes.dims if d in ('iid','sample','animal'))
        vd=next(d for d in genotypes.dims if d!=sd);m=genotypes.sizes[vd]
        names=pd.Index(genotypes[vd].values.astype(str))
        read=lambda a,b:np.asarray(genotypes.isel({vd:slice(a,b)}).transpose(sd,vd).values,float)
    else:
        X=np.asarray(genotypes,float);m=X.shape[1]
        names=genotypes.columns if isinstance(genotypes,pd.DataFrame) else pd.RangeIndex(m)
        read=lambda a,b:X[:,a:b]
    if m>max_variants:raise MemoryError('bounded-locus signed LD only; use streamed local windows')
    if not names.is_unique:raise ValueError('variant labels must be unique')
    start=time.perf_counter();R=np.full((m,m),np.nan,np.float32);N=np.zeros((m,m),np.int32)
    for a in range(0,m,block_variants):
        aa=min(m,a+block_variants);A=read(a,aa)
        for b in range(a,m,block_variants):
            bb=min(m,b+block_variants);r,n=_correlation(A,A if a==b else read(b,bb))
            R[a:aa,b:bb]=r;R[b:bb,a:aa]=r.T;N[a:aa,b:bb]=n;N[b:bb,a:aa]=n.T
    return {'r':pd.DataFrame(R,index=names,columns=names),'nobs':pd.DataFrame(N,index=names,columns=names),
            'valid':pd.DataFrame(np.isfinite(R),index=names,columns=names),'seconds':time.perf_counter()-start,
            'allocated_result_bytes':R.nbytes+N.nbytes+R.size,
            'psd_policy':'pairwise-complete correlation; PSD not guaranteed; invalid pairs retained'}


def radix_bubbles(graph):
    """Single-entry/single-exit branching regions, with explicit path membership.

    Suffix node interning alone is not a bubble. Closure is checked against the
    directed graph. Nesting is set containment and may have multiple parents.
    Stable IDs derive from genomic alleles/boundaries, not insertion node IDs.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata. Passed to
        ``bubble_nomenclature`` as ``graph``.

    Returns
    -------
    object
        Computed result from ``bubble_nomenclature(frame.sort_values(['start', 'end', 'bubble_id']).reset_index(drop=True), graph=graph)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('radix bubbles require a DAG')
    H=graph.copy();terminal='__bubble_terminal__';H.add_node(terminal)
    for x in list(graph):
        if graph.out_degree(x)==0:H.add_edge(x,terminal)
    post=nx.immediate_dominators(H.reverse(copy=False),terminal)
    rows=[];paths=graph.graph['paths'];v=graph.graph['variants']
    for source in graph:
        if graph.out_degree(source)<2:continue
        sink=post.get(source)
        if sink is None:continue
        inside=({source}|nx.descendants(H,source))&({sink}|nx.ancestors(H,sink))
        middle=inside-{source,sink}
        if any(any(y not in inside for y in H.predecessors(x)) or
               any(y not in inside for y in H.successors(x)) for x in middle):continue
        if any(y not in inside for y in graph.successors(source)):continue
        variants=tuple(x for x in v.index if any(x in graph.nodes[n]['variants'] for n in middle))
        if not variants:continue
        def boundary(n):
            """Resolve the physical/graph boundary used by the current radix interval.

            Parameters
            ----------
            n : int
                Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
                length, not an arbitrary truncation limit.

            Returns
            -------
            object
                Computed result from ``[list(a['variants']), list(a['allele_states'])]``; see the summary and the called implementation for its labeled/numerical representation.
            """
            a=graph.nodes[n] if n in graph else {'variants':(),'allele_states':()};return [list(a['variants']),list(a['allele_states'])]
        signature=repr((graph.graph['chrom'],variants,boundary(source),boundary(sink)))
        bid='B'+hashlib.sha256(signature.encode()).hexdigest()[:16]
        members=[];path_strings=[]
        for row in paths.itertuples():
            seq=list(row.nodes)+[terminal]
            if source in seq and sink in seq and seq.index(source)<seq.index(sink):
                members.append(str(row.Index));path_strings.append(tuple(seq[seq.index(source):seq.index(sink)+1]))
        rows.append(dict(bubble_id=bid,entry=source,exit=sink,nodes=tuple(sorted(inside,key=str)),
                         variants=variants,start=int(v.loc[list(variants),'pos'].min()),end=int(v.loc[list(variants),'pos'].max()),
                         path_ids=tuple(members),path_membership=tuple(path_strings),n_paths=len(members)))
    frame=pd.DataFrame(rows)
    if frame.empty:return bubble_nomenclature(pd.DataFrame(columns=['bubble_id','entry','exit','nodes','variants','start','end','path_ids','path_membership','n_paths','parents','children','level','outer_context_variants']))
    sets={r.bubble_id:set(r.nodes) for r in frame.itertuples()};vr={r.bubble_id:set(r.variants) for r in frame.itertuples()}
    parents={x:[y for y in sets if sets[x]<sets[y] and not any(sets[x]<sets[z]<sets[y] for z in sets)] for x in sets}
    levels={}
    def level(x):
        """Find the selected node's depth in the enclosing radix graph ordering.

        Parameters
        ----------
        x : array-like
            Numerical operand in ``1 + max((level(y) for y in parents[x]))``. Shape must conform to
            that matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``levels[x]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if x not in levels:levels[x]=0 if not parents[x] else 1+max(level(y) for y in parents[x])
        return levels[x]
    frame['parents']=[tuple(sorted(parents[x])) for x in frame.bubble_id]
    frame['children']=[tuple(sorted(y for y in sets if x in parents[y])) for x in frame.bubble_id]
    frame['level']=[level(x) for x in frame.bubble_id]
    def ancestors(b):
        """Collect ancestral node identifiers for the selected radix path.

        Parameters
        ----------
        b : array-like
            Numerical operand in ``parents[b]``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        out=set(parents[b]);stack=list(out)
        while stack:
            for parent in parents[stack.pop()]:
                if parent not in out:out.add(parent);stack.append(parent)
        return out
    frame['outer_context_variants']=[tuple(x for x in v.index if x in set().union(*(vr[y] for y in ancestors(b)))-vr[b]) for b in frame.bubble_id]
    return bubble_nomenclature(frame.sort_values(['start','end','bubble_id']).reset_index(drop=True),graph=graph)


def _arrays(genotypes):
    """Align joint graph-model inputs into numerical arrays and labels.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.
    """
    if isinstance(genotypes,pd.DataFrame):return genotypes.to_numpy(float)[:,None,:],genotypes.index,genotypes.columns,False
    if not hasattr(genotypes,'dims'):raise TypeError('labelled DataFrame or DataArray required')
    sd=next(d for d in genotypes.dims if d in ('iid','sample','animal'))
    hd=next((d for d in genotypes.dims if d in ('haplotype','phase')),None)
    vd=next(d for d in genotypes.dims if d not in (sd,hd))
    A=np.asarray(genotypes.transpose(*([sd,hd,vd] if hd else [sd,vd])).values,float)
    return A if hd else A[:,None,:],pd.Index(genotypes[sd].values.astype(str)),pd.Index(genotypes[vd].values.astype(str)),bool(hd)


@dataclass
class JointRadixModel:
    """JointRadixModel represents joint radix model.

    Attributes
    ----------
    graph : object
        Stored graph; its axis/order follows the construction inputs.
    bubbles : pd.DataFrame
        Stored bubbles; its axis/order follows the construction inputs.
    ld : dict
        Stored ld; its axis/order follows the construction inputs.
    nodes : tuple
        Stored nodes; its axis/order follows the construction inputs.
    variant_names : tuple
        Stored variant names; its axis/order follows the construction inputs.
    phased : bool
        Stored phased; its axis/order follows the construction inputs.
    mean : np.ndarray
        Stored mean; its axis/order follows the construction inputs.
    scale : np.ndarray
        Stored scale; its axis/order follows the construction inputs.
    model : object
        Stored model; its axis/order follows the construction inputs.
    background : np.ndarray
        Stored background; its axis/order follows the construction inputs.
    background_prediction : np.ndarray
        Stored background prediction; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    graph : object
        Stored graph consumed by ``features``; see those methods for the numerical/label
        contract. Required construction field.

    bubbles : pd.DataFrame
        Stored bubbles supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    ld : dict
        Stored ld supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    nodes : tuple
        Stored nodes consumed by ``features``; see those methods for the numerical/label
        contract. Required construction field.

    variant_names : tuple
        Stored variant names consumed by ``features``/``predict``; see those methods for the
        numerical/label contract. Required construction field.

    phased : bool
        Stored phased consumed by ``features``/``predict``; see those methods for the
        numerical/label contract. Required construction field.

    mean : np.ndarray
        Stored mean consumed by ``predict``; see those methods for the numerical/label contract.
        Required construction field.

    scale : np.ndarray
        Stored scale consumed by ``predict``; see those methods for the numerical/label
        contract. Required construction field.

    model : object
        Stored model consumed by ``predict``; see those methods for the numerical/label
        contract. Required construction field.

    background : np.ndarray
        Stored background supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    background_prediction : np.ndarray
        Stored background prediction supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Required
        construction field.
    """
    graph:object
    bubbles:pd.DataFrame
    ld:dict
    nodes:tuple
    variant_names:tuple
    phased:bool
    mean:np.ndarray
    scale:np.ndarray
    model:object
    background:np.ndarray
    background_prediction:np.ndarray
    metadata:dict

    def features(self,A):
        """Return the features result computed by ``np.column_stack``.

        Parameters
        ----------
        A : array-like
            Numerical operand in ``A[:, 0, :]``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``np.column_stack(blocks)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        dosage=A.sum(1) if self.phased else A[:,0,:]
        blocks=[dosage];index={x:i for i,x in enumerate(self.variant_names)}
        for n in self.nodes:
            node=self.graph.nodes[n];ix=[index[x] for x in node['variants']];states=np.asarray(node['allele_states'])
            match=np.all(A[:,:,ix]==states[None,None,:],axis=2).sum(1)
            blocks.append(match[:,None])
        return np.column_stack(blocks)

    def predict(self,genotypes):
        """Return the predict result computed by ``pd.DataFrame``.

        Parameters
        ----------
        genotypes : array-like or pandas.DataFrame
            Genotype values with animals in rows and variants in columns, or the genotype source
            accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
            labels and the coding expected by the selected kernel or association model.

        Returns
        -------
        object
            Computed result from ``pd.DataFrame(np.asarray(self.model.predict((F - self.mean) / self.scale)).reshape(len(F), -1), index=ids, columns=self.metadata['traits'])``; see the summary and the called implementation for its labeled/numerical representation.
        """
        A,ids,names,phased=_arrays(genotypes)
        if phased!=self.phased or tuple(names)!=self.variant_names:raise ValueError('phase/allele-aligned variant columns required')
        F=self.features(A);F=np.where(np.isfinite(F),F,self.mean)
        return pd.DataFrame(np.asarray(self.model.predict((F-self.mean)/self.scale)).reshape(len(F),-1),index=ids,columns=self.metadata['traits'])


class _ObservedCellRidge:
    """Shared feature basis/common penalty, trait-specific observed-cell fits."""
    def __init__(self, alpha):
        """Initialize _ObservedCellRidge with the inputs and resource settings below.

        Parameters
        ----------
        alpha : float
            Regularization coefficient, mixing weight or statistical significance threshold selected
            by this estimator; the local validation and objective determine its precise scale.
        """
        self.alpha = alpha

    def fit(self, X, Y):
        """Fit the estimator using the enclosing model configuration and aligned training inputs.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Y : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        Returns
        -------
        object
            Computed result from ``self``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self.models = []
        for column in np.asarray(Y, float).T:
            observed = np.isfinite(column)
            if observed.sum() < 2:
                raise ValueError('each inner training trait needs at least2 observed responses')
            self.models.append(Ridge(alpha=self.alpha).fit(X[observed], column[observed]))
        return self

    def predict(self, X):
        """Return the predict result computed by ``np.column_stack``.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``np.column_stack([model.predict(X) for model in self.models])``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.column_stack([model.predict(X) for model in self.models])


def fit_radix_joint_finemap(genotypes,traits,*,variants,alphas=(.1,1.,10.,100.),
                            min_path_samples=8,max_node_features=256,random_state=0,
                            response_policy="complete",**graph_kwargs):
    """One train-only joint SNP+radix-path ridge model with shared LD/indexing.

    Graph/HDBSCAN and model tuning use supplied training inputs only. This is a
    predictive haplotype-feature model, not a Bayesian bubble inclusion model.
    Phased paths are honoured when a haplotype dimension exists; ordinary PLINK
    dosage retains its explicit phase ambiguity. Default ``response_policy="complete"``
    requires20 complete rows. ``"observed_cells"`` keeps rows with any response,
    requires20 observations per trait, and fits separate ridge coefficients on
    observed cells using a shared graph feature basis and common inner-selected
    penalty. It is not a joint Gaussian residual likelihood. Neither policy
    imputes response targets; imputation applies only to genotype features.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    variants : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader. Passed to ``build_haplotype_radix_dag`` as ``variants``.

    alphas : float, int or sequence
        Regularization search grid: l1_ratio mixes L1 versus L2 penalties on [0, 1]; alphas
        supplies candidate strengths or the number of strengths generated by the called CV
        estimator. See that estimator for automatic grid semantics. Default: ``(0.1, 1.0, 10.0,
        100.0)``.

    min_path_samples : int
        Input for the operation ``a['n_samples'] >= min_path_samples``. See
        ``vigor.finemap.integrated.fit_radix_joint_finemap`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``8``.

    max_node_features : int
        Input for the operation ``sorted(nodes, key=lambda n: (-g.nodes[n]['n_samples'],
        repr(g.nodes[n]['variants']), repr(g.nodes[n]['allele_states'])))[:max_node_features]``.
        See ``vigor.finemap.integrated.fit_radix_joint_finemap`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``256``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.

    response_policy : str
        Input for the operation ``response_policy not in ('complete', 'observed_cells')``. See
        ``vigor.finemap.integrated.fit_radix_joint_finemap`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'complete'``.

    **graph_kwargs : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Passed to ``build_haplotype_radix_dag`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``z``; see the summary and the called implementation for its labeled/numerical representation.
    """
    A,ids,names,phased=_arrays(genotypes)
    Y=traits.reindex(ids);observed=np.isfinite(np.asarray(Y,float))
    if response_policy not in ('complete','observed_cells'):
        raise ValueError('response_policy must be complete or observed_cells')
    ok=observed.all(1) if response_policy=='complete' else observed.any(1)
    if response_policy=='complete' and ok.sum()<20:
        raise ValueError('at least20 complete joint-response training rows required')
    if response_policy=='observed_cells' and (observed.sum(0)<20).any():
        raise ValueError('observed_cells requires at least20 observed responses per trait')
    model_type=Ridge if response_policy=='complete' else _ObservedCellRidge
    if not names.is_unique or not ids.is_unique:raise ValueError('unique sample/variant labels required')
    sd=None if isinstance(genotypes,pd.DataFrame) else next(d for d in genotypes.dims if d in ('iid','sample','animal'))
    selected=genotypes.loc[ids[ok]] if isinstance(genotypes,pd.DataFrame) else genotypes.isel({sd:np.flatnonzero(ok)})
    if 'ldmatrix' in graph_kwargs or 'ld_kind' in graph_kwargs:
        raise ValueError('joint workflow computes its shared signed LD; do not supply a second LD matrix/kind')
    dosage=A[ok].sum(1) if phased else A[ok,0,:]
    ld=signed_ld(pd.DataFrame(dosage,index=ids[ok],columns=names))
    g=build_haplotype_radix_dag(selected,variants=variants,ldmatrix=ld['r'],ld_kind='r',**graph_kwargs)
    if len(g.graph['ld_edges']):
        labels=g.graph['variants'].index
        edges=g.graph['ld_edges'];edges['nobs']=[int(ld['nobs'].loc[labels[i],labels[j]]) for i,j in zip(edges.i,edges.j)]
    g.graph['ld_observation_counts']='shared signed LD pairwise-complete animal counts'
    bubbles=radix_bubbles(g)
    nodes=[n for n,a in g.nodes(data=True) if a['variants'] and 3 not in a['allele_states'] and a['n_samples']>=min_path_samples]
    nodes=tuple(sorted(nodes,key=lambda n:(-g.nodes[n]['n_samples'],repr(g.nodes[n]['variants']),repr(g.nodes[n]['allele_states'])))[:max_node_features])
    z=JointRadixModel(g,bubbles,ld,nodes,tuple(names),phased,None,None,None,A[ok],None,{})
    F=z.features(A[ok]);y=np.asarray(Y,float)[ok];scores={};folds=KFold(3,shuffle=True,random_state=random_state)
    fold_features=[]
    for tr,va in folds.split(F):
        # Discovery of node/path features is fold-local too. Validation animals
        # cannot create the factor/graph basis used to choose regularization.
        sg=selected.iloc[tr] if isinstance(selected,pd.DataFrame) else selected.isel({sd:tr})
        fld=signed_ld(pd.DataFrame(dosage[tr],index=ids[ok][tr],columns=names))
        fg=build_haplotype_radix_dag(sg,variants=variants,ldmatrix=fld['r'],ld_kind='r',**graph_kwargs)
        fn=[n for n,a in fg.nodes(data=True) if a['variants'] and 3 not in a['allele_states'] and a['n_samples']>=min_path_samples]
        fn=tuple(sorted(fn,key=lambda n:(-fg.nodes[n]['n_samples'],repr(fg.nodes[n]['variants']),repr(fg.nodes[n]['allele_states'])))[:max_node_features])
        fm=JointRadixModel(fg,None,None,fn,tuple(names),phased,None,None,None,None,None,{})
        FF=fm.features(A[ok]);mu=np.nan_to_num(np.nanmean(FF[tr],0));ss=np.nanstd(FF[tr],0);ss=np.where(ss>1e-8,ss,1.)
        XX=(np.where(np.isfinite(FF),FF,mu)-mu)/ss
        fold_features.append((tr,va,XX))
    for alpha in alphas:
        sse=0.
        for tr,va,XX in fold_features:
            model=model_type(alpha=alpha).fit(XX[tr],y[tr])
            prediction=np.asarray(model.predict(XX[va])).reshape(len(va),-1)
            mask=np.isfinite(y[va]);sse+=float(np.sum((y[va][mask]-prediction[mask])**2))
        scores[str(alpha)]=sse
    alpha=min(alphas,key=lambda a:scores[str(a)]);z.mean=np.nan_to_num(np.nanmean(F,0));z.scale=np.nanstd(F,0);z.scale=np.where(z.scale>1e-8,z.scale,1.)
    z.model=model_type(alpha=alpha).fit((np.where(np.isfinite(F),F,z.mean)-z.mean)/z.scale,y)
    z.background_prediction=np.asarray(z.model.predict((np.where(np.isfinite(F),F,z.mean)-z.mean)/z.scale)).reshape(len(F),-1)
    z.metadata={'traits':list(Y.columns),'n_input':len(ids),'n_joint_complete':int(observed.all(1).sum()),'n_retained':int(ok.sum()),
                'response_policy':response_policy,'observed_responses_per_trait':dict(zip(map(str,Y.columns),observed[ok].sum(0).tolist())),
                'n_observed_response_cells':int(observed[ok].sum()),'alpha':alpha,'inner_sse':scores,
                'inner_basis':'graph/path support, imputation and scaling refit in each inner training fold','path_kind':g.graph['path_kind'],'pip_available':False,'fit':'joint SNP + genuine radix node-path features',
                'complete_response_policy':'default requires20 complete rows; observed_cells requires20 observed responses per trait; no response imputation',
                'response_fit':'joint multi-output ridge' if response_policy=='complete' else 'shared graph basis and common penalty; trait-specific observed-cell ridge, no joint residual likelihood',
                'causal_identification':'none; predictive effects only','LD_interpretation':ld['psd_policy'],
                'LD_integration':'this exact labelled signed LD is passed to graph dependency edges/HDBSCAN as ld_kind=r; HDBSCAN distance is 1-r^2; graph nesting is retained during reconciliation'}
    return z


def _bubble_routing(bubbles, owners):
    """Canonical containment forest; retain non-tree parents as audit metadata.

    Multiple-parent partial orders need an explicit coalition choice. Lexically
    first stable parent is used; incomparable-owner SNPs stay junction players.
    We do not claim an overlapping partial order defines a unique Owen game.

    Parameters
    ----------
    bubbles : object (see the cited input contract)
        Input for the operation ``bubbles.itertuples()``. See
        ``vigor.finemap.integrated._bubble_routing`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    owners : object (see the cited input contract)
        Input for the operation ``set(owners.values())``. See
        ``vigor.finemap.integrated._bubble_routing`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    parents={str(r.bubble_id):(sorted(r.parents)[0] if r.parents else None) for r in bubbles.itertuples()}
    for x in set(owners.values()):parents.setdefault(x,None)
    tree={None:[]}
    for child,parent in parents.items():tree.setdefault(parent,[]).append(child);tree.setdefault(child,[])
    for children in tree.values():children.sort()
    return parents,tree


def _recursive_owen_order(players,names,owners,clusters,tree,rng):
    """Randomly order sibling subtrees, direct-owner LD groups, then SNP leaves.

    Parameters
    ----------
    players : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names.

    owners : object (see the cited input contract)
        Input for the operation ``owners[names[p]]``. See
        ``vigor.finemap.integrated._recursive_owen_order`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    clusters : object (see the cited input contract)
        Input for the operation ``clusters.loc[names[p], 'ld_group']``. See
        ``vigor.finemap.integrated._recursive_owen_order`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    tree : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.
    """
    own={}
    for p in players:own.setdefault(owners[names[p]],{}).setdefault(str(clusters.loc[names[p],'ld_group']),[]).append(p)
    def walk(node):
        """Traverse the enclosing graph/store in the operation-defined order.

        Parameters
        ----------
        node : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        units=[('child',c) for c in tree.get(node,[])]+[('ld',g) for g in sorted(own.get(node,{}))]
        out=[]
        for index in rng.permutation(len(units)):
            kind,key=units[index]
            out.extend(walk(key) if kind=='child' else rng.permutation(own[node][key]).tolist())
        return out
    return walk(None)


def conditional_radix_shap(model,genotypes,*,n_permutations=128,min_context_support=8,random_state=0):
    """Recursive conditional empirical Owen values on observed allele-aligned paths.

    Players are unique SNP columns. Bubble aggregation assigns every variant
    to its unique innermost bubble; incomparable nesting gets a junction player.
    Global values E[f(X)|X_S=x_S] reconstruct f(x)-E[f(X)]. Inner-bubble local
    values additionally fix outer context, and reconstruct E[f(X)|owned players, context]-E[f(X)|context].
    These two decompositions have distinct baselines and must not be added.
    Low-support contexts return unsupported values, not invented conditional
    samples. Monte Carlo SE describes permutation uncertainty only.

    Parameters
    ----------
    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here.

    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    n_permutations : int
        Number of random permutations used to estimate the null/evidence quantity. Nonnegative
        counts control Monte Carlo work; zero disables permutation where supported. Default:
        ``128``. Source comparisons: ``n_permutations < 2``. See the rejection/fallback branches
        for enforcement.

    min_context_support : int
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``8``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    dict
        Result fields: ``path``, ``ld_graph_reconciliation``, ``snp``, ``bubble``, ``inner_conditional``, ``inner_bubble``, ``ownership``, ``bubble_subtree``, ``support``, ``hierarchy``, ``interpretation``, ``uncertainty``. Field arrays preserve the corresponding input axis order.
    """
    A,ids,names,phase=_arrays(genotypes)
    if tuple(names)!=model.variant_names or phase!=model.phased:raise ValueError('shared allele/phase indexing required')
    if n_permutations<2:raise ValueError('at least2 permutations required for Monte Carlo SE')
    bg=model.background;BP=model.background_prediction;m=len(names);t=BP.shape[1]
    rng=np.random.default_rng(random_state);pred=np.asarray(model.predict(genotypes));bubbles=model.bubbles
    owners={}
    for x in names:
        cand=[r for r in bubbles.itertuples() if x in r.variants]
        inner=[r for r in cand if not any(set(q.nodes)<set(r.nodes) for q in cand)]
        owners[x]=inner[0].bubble_id if len(inner)==1 else ('junction' if inner else 'locus')
    parents,tree=_bubble_routing(bubbles,owners)
    descendants={b:set(nx.descendants(nx.DiGraph([(p,c) for p,cs in tree.items() if p is not None for c in cs]),b)) if tree.get(b) else set() for b in parents}
    # Equality here must match decompose: NaN unresolved states compare equal.
    # np.unique(axis=0) treats those rows as distinct; pandas handles equal NaNs.
    background_diversity=len(pd.DataFrame(bg.reshape(len(bg),-1)).drop_duplicates())
    outputs=[];local=[];group_uncertainty={};local_groups=[];subtree_rows=[];support_rows=[]
    def decompose(i,players,context,matches):
        """Decompose a fitted covariance into the factor basis used by the joint graph model.

        Parameters
        ----------
        i : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        players : object (see the cited input contract)
            Input for the operation ``value(players)``. See
            ``vigor.finemap.integrated.conditional_radix_shap.decompose`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        context : float or str
            Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
            descriptive text identifying the operation in an allocation error.

        matches : object (see the cited input contract)
            Input for the operation ``matching & matches[:, p]``. See
            ``vigor.finemap.integrated.conditional_radix_shap.decompose`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        Returns
        -------
        object
            Computed result from ``(mean, se, base, final, float(np.max(np.abs(mean.sum(0) - (final - base)))) if supported else np.nan, min(counts, default=nb), supported, samples)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        samples=[];counts=[];cache={}
        def value(S,matching=None):
            """Extract the selected numerical quantity using the enclosing prepared state.

            Parameters
            ----------
            S : array-like
                Numerical operand in ``set(S) | set(context)``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.

            matching : object (see the cited input contract)
                Input for the operation ``matching is not None``. See
                ``vigor.finemap.integrated.conditional_radix_shap.decompose.value`` and its caller for
                the scientific representation; the shown operation constrains the required type/shape.
                Default: ``None``.

            Returns
            -------
            object
                Computed result from ``(v, count)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            key=tuple(sorted(set(S)|set(context)))
            if key in cache:return cache[key]
            ix=np.asarray(key,int)
            match=matching if matching is not None else (np.all(matches[:,ix],axis=1) if len(ix) else np.ones(len(bg),bool))
            count=int(match.sum());v=BP[match].mean(0) if count>=min_context_support else np.full(t,np.nan)
            cache[key]=(v,count);return v,count
        base,nb=value([])
        final,nf=value(players)
        # Exact matching support is monotone as coalitions grow. An unsupported
        # endpoint makes this entire conditional game undefined. Certify that
        # immediately instead of spending O(permutations * variants^2 * donors)
        # on an undefined high-dimensional game. Never smooth or invent donors.
        if min(nb,nf)<min_context_support:
            unavailable=np.full((len(players),t),np.nan)
            return unavailable,unavailable.copy(),base,final,np.nan,min(nb,nf),False,np.full((n_permutations,len(players),t),np.nan)
        for _ in range(n_permutations):
            order=_recursive_owen_order(players,names,owners,model.graph.graph['ld_clusters'],tree,rng)
            matching=np.all(matches[:,context],axis=1) if context else np.ones(len(bg),bool)
            S=[];old,n=value(S,matching);terms=np.zeros((len(players),t));mp={p:k for k,p in enumerate(players)}
            for p in order:
                matching=matching & matches[:,p]
                S.append(int(p));new,n=value(S,matching);counts.append(n);terms[mp[p]]=new-old;old=new
            samples.append(terms)
        samples=np.asarray(samples);supported=np.isfinite(samples).all()
        mean=samples.mean(0);se=samples.std(0,ddof=1)/np.sqrt(n_permutations)
        final,_=value(players)
        return mean,se,base,final,float(np.max(np.abs(mean.sum(0)-(final-base)))) if supported else np.nan,min(counts,default=nb),supported,samples
    for i,sample in enumerate(ids):
        matches=np.all((bg==A[i][None,:,:])|(np.isnan(bg)&np.isnan(A[i][None,:,:])),axis=1)
        values,se,base,final,error,support,supported,draws=decompose(i,list(range(m)),[],matches)
        support_rows.append(dict(sample=str(sample),supported=bool(supported),min_coalition_support=support,n_background=len(bg),
            n_unique_background_paths=background_diversity,
            query_observed_genotype_fraction=float(np.isfinite(A[i]).mean()),
            query_fully_unresolved_variants=int(np.isnan(A[i]).all(axis=0).sum()),
            missingness_interpretation='equal unresolved states enter empirical conditioning; missing-state compatibility is not observed allele evidence',conditional_endpoint_max_abs_model_difference=float(np.max(np.abs(final-pred[i]))) if np.isfinite(final).all() else np.nan))
        for row in bubbles.itertuples():
            subtree={row.bubble_id}|descendants[row.bubble_id]
            ix=[j for j,x in enumerate(names) if owners[x] in subtree]
            if not ix:continue
            total_draws=draws[:,ix,:].sum(1)
            for k,trait in enumerate(model.metadata['traits']):
                subtree_rows.append(dict(sample=str(sample),bubble_id=row.bubble_id,canonical_parent=parents[row.bubble_id],parents=row.parents,
                    children=row.children,level=row.level,trait=trait,variants=tuple(names[j] for j in ix),
                    value=values[ix,k].sum(),mc_se=float(total_draws[:,k].std(ddof=1)/np.sqrt(n_permutations)),supported=bool(supported),
                    interpretation='global recursive Owen subtree sum; ancestor/descendant summaries overlap and must not be added'))
        for owner in sorted(set(owners.values())):
            ix=[j for j,x in enumerate(names) if owners[x]==owner]
            total_draws=draws[:,ix,:].sum(1)
            for k,trait in enumerate(model.metadata['traits']):group_uncertainty[(str(sample),owner,trait)]=float(total_draws[:,k].std(ddof=1)/np.sqrt(n_permutations))
        for j,x in enumerate(names):
            for k,trait in enumerate(model.metadata['traits']):outputs.append(dict(sample=str(sample),variant=str(x),bubble_owner=owners[x],trait=trait,value=values[j,k],mc_se=se[j,k],baseline=base[k],prediction=pred[i,k],reconstruction_error=error,min_support=support,supported=supported))
        for row in bubbles.itertuples():
            players=[j for j,x in enumerate(names) if owners[x] in ({row.bubble_id}|descendants[row.bubble_id])];context=[j for j,x in enumerate(names) if x in row.outer_context_variants]
            if not players:continue
            v,se,base,final,error,support,supported,draws=decompose(i,players,context,matches)
            totals=draws.sum(1)
            for k,trait in enumerate(model.metadata['traits']):local_groups.append(dict(sample=str(sample),bubble_id=row.bubble_id,trait=trait,value=v[:,k].sum(),
                mc_se=float(totals[:,k].std(ddof=1)/np.sqrt(n_permutations)),context_baseline=base[k],coalition_value=final[k],
                reconstruction_error=error,min_support=support,supported=supported,interpretation='inner owned-player contrast at fixed outer context; distinct from global bubble allocation'))
            for jj,j in enumerate(players):
                for k,trait in enumerate(model.metadata['traits']):local.append(dict(sample=str(sample),bubble_id=row.bubble_id,variant=str(names[j]),trait=trait,value=v[jj,k],mc_se=se[jj,k],context_baseline=base[k],coalition_value=final[k],reconstruction_error=error,min_support=support,supported=supported))
    snps=pd.DataFrame(outputs);aggregation=snps.groupby(['sample','bubble_owner','trait'],as_index=False).value.sum(min_count=1)
    aggregation['mc_se']=[group_uncertainty[(r.sample,r.bubble_owner,r.trait)] for r in aggregation.itertuples()]
    # Index each leaf group once. Repeated full-table Boolean scans here cost
    # O(samples^2 * owners * traits^2 * variants) at real multitrait loci.
    # Keep Series.sum's original leaf summation convention for path values.
    grouped=snps.groupby(['sample','bubble_owner','trait'],sort=False)
    support_lookup=grouped.supported.all().to_dict()
    path_value_lookup=grouped.value.agg(lambda leaf:leaf.sum(min_count=1)).to_dict()
    aggregation['supported']=[bool(support_lookup[(r.sample,r.bubble_owner,r.trait)]) for r in aggregation.itertuples()]
    reconcile=pd.DataFrame([{'variant':str(x),'bubble_owner':owners[x],'ld_group':str(model.graph.graph['ld_clusters'].loc[x,'ld_group']),'policy':'intersect LD groups with unique bubble ownership; preserve graph nesting'} for x in names])
    # Active path-pair values aggregate the same non-overlapping SNP players;
    # they are not a second set of effects to add to the global decomposition.
    # A phased animal has a pair of paths; no arbitrary haplotype allocation.
    path_rows=[]
    for i,sample in enumerate(ids):
        for owner in sorted(set(owners.values())):
            ix=[j for j,x in enumerate(names) if owners[x]==owner]
            signature=repr((owner,tuple(names[j] for j in ix),A[i,:,ix].tolist()))
            pid='P'+hashlib.sha256(signature.encode()).hexdigest()[:16]
            match=np.all((bg[:,:,ix]==A[i][:,ix][None,:,:])|(np.isnan(bg[:,:,ix])&np.isnan(A[i][:,ix][None,:,:])),axis=(1,2))
            for trait in model.metadata['traits']:
                key=(str(sample),owner,trait)
                path_rows.append(dict(sample=str(sample),bubble_owner=owner,path_state_id=pid,trait=trait,
                    value=path_value_lookup[key],mc_se=group_uncertainty[key],supported=bool(support_lookup[key]),n_background_path_pairs=int(match.sum()),
                    phase='phased_path_pair' if phase else 'unphased_dosage_compatibility',variant_membership=tuple(names[j] for j in ix),
                    allele_states=repr(A[i,:,ix].tolist()),interpretation='active path-state aggregation of unique SNP Owen players; do not add to SNP/bubble totals'))
    return {'path':pd.DataFrame(path_rows),'ld_graph_reconciliation':reconcile,'snp':snps,'bubble':aggregation,'inner_conditional':pd.DataFrame(local),'inner_bubble':pd.DataFrame(local_groups),'ownership':owners,
            'bubble_subtree':pd.DataFrame(subtree_rows),'support':pd.DataFrame(support_rows),
            'hierarchy':{'canonical_parent':parents,'children':tree,'multiple_parent_policy':'lexically first stable parent defines coalition forest; retain all graph parents; no unique attribution for overlapping partial order'},
            'interpretation':'predictive empirical conditional recursive Owen values; canonical ancestor bubble subtrees then owner LD groups then SNPs; not unrestricted SNP Shapley; no PIP; whole observed paths only',
            'uncertainty':'Monte Carlo permutation SE; finite background and model uncertainty excluded'}


def bubble_nomenclature(bubbles, *, graph=None):
    """Deterministic readable labels for a containment DAG, without tree claims.

    Canonical parent is the lexically first immutable ID, exactly as in the
    recursive Owen routing. Siblings are numbered by genomic boundaries/ID.
    All graph parents and root-to-node ID paths remain available separately.
    Labels are phenotype independent; immutable IDs should be used as keys.

    Parameters
    ----------
    bubbles : object (see the cited input contract)
        Input for the operation ``bubbles.copy()``. See
        ``vigor.finemap.integrated.bubble_nomenclature`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    Returns
    -------
    object
        Computed result from ``frame``; see the summary and the called implementation for its labeled/numerical representation.
    """
    frame=bubbles.copy()
    records={str(r.bubble_id):r for r in frame.itertuples()}
    parents={b:tuple(sorted(map(str,r.parents))) for b,r in records.items()}
    if any(p not in records for ps in parents.values() for p in ps):
        raise ValueError('bubble parent references must be present')
    canonical={b:ps[0] if ps else None for b,ps in parents.items()}
    tree={None:[]}
    for b,p in canonical.items():tree.setdefault(p,[]).append(b)
    order=lambda b:(records[b].start,records[b].end,b)
    for ps in tree.values():ps.sort(key=order)
    labels={};paths={};visiting=set()
    def label(b):
        """Construct the displayed identifier for the selected path/node quantity.

        Parameters
        ----------
        b : array-like
            Numerical operand in ``prefix + str(tree[p].index(b) + 1)``. Shape must conform to that
            matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``labels[b]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if b in labels:return labels[b]
        if b in visiting:raise ValueError('bubble parents must form a DAG')
        visiting.add(b);p=canonical[b];prefix=label(p)+'.' if p else 'bubble'
        labels[b]=prefix+str(tree[p].index(b)+1);visiting.remove(b);return labels[b]
    def all_paths(b):
        """Enumerate the radix DAG paths used by the enclosing joint selection operation.

        Parameters
        ----------
        b : array-like
            Numerical operand in ``tuple(q) + (b,)``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``paths[b]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if b not in paths:
            paths[b]=tuple(sorted(tuple(q)+(b,) for p in parents[b] for q in all_paths(p))) if parents[b] else ((b,),)
        return paths[b]
    for b in records:label(b)
    frame['bubble_label']=[labels[str(b)] for b in frame.bubble_id]
    frame['canonical_parent']=[canonical[str(b)] for b in frame.bubble_id]
    def canonical_path(b):
        """Choose the deterministic representative for a radix path equivalence class.

        Parameters
        ----------
        b : array-like
            Numerical operand in ``canonical_path(canonical[b]) + (b,)``. Shape must conform to that
            matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``canonical_path(canonical[b]) + (b,) if canonical[b] else (b,)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return canonical_path(canonical[b])+(b,) if canonical[b] else (b,)
    frame['canonical_path']=[canonical_path(str(b)) for b in frame.bubble_id]
    frame['ancestor_id_paths']=[all_paths(str(b)) for b in frame.bubble_id]
    frame['parent_labels']=[tuple(labels[p] for p in parents[str(b)]) for b in frame.bubble_id]
    frame['nomenclature_policy']='canonical lexically first stable parent; genomic-ordered siblings; all DAG paths retained'
    if graph is not None:
        memberships={node:set() for node in graph}
        for row in frame.itertuples():
            for node in row.nodes:
                if node in memberships:memberships[node].add((row.bubble_id,row.bubble_label))
        for node,pairs in memberships.items():
            ordered=sorted(pairs)
            graph.nodes[node]['bubble_ids']=[b for b,label in ordered]
            graph.nodes[node]['bubble_labels']=[label for b,label in ordered]
    return frame
