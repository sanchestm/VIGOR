"""Trainable CPU GCN/R-GCN with CSR message passing and NumPy Adam.

Two learned message-passing layers. No torch/torch-geometric dependency. This
full-batch implementation exposes its transductive graph assumptions explicitly.
"""
from dataclasses import dataclass,field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax,logsumexp
from .sparse import edge_table_to_csr

def memory_limit(value=None):
 """Check/resolve the memory allowance for graph-model feature/message allocations.

 Parameters
 ----------
 value : object (see the cited input contract)
     Input for the operation ``str(value).lower()``. See ``vigor.network.gnn.memory_limit``
     and its caller for the scientific representation; the shown operation constrains the
     required type/shape. Default: ``None``. None delegates to the local fallback shown in
     this body; it is not an additional numeric/category value.

 Returns
 -------
 object
     Computed result from ``(int(float(v[:-len(key)]) * u[key]) if key else int(value), rss)``; see the summary and the called implementation for its labeled/numerical representation.
 """
 import psutil
 rss=psutil.Process().memory_info().rss
 if value is None:return psutil.virtual_memory().available+rss,rss
 u={"kib":2**10,"mib":2**20,"gib":2**30}; v=str(value).lower(); key=next((k for k in u if v.endswith(k)),None)
 return (int(float(v[:-len(key)])*u[key]) if key else int(value)),rss

class _Graph:
 """_Graph represents graph.

 See ``__init__`` for complete input types, defaults and construction constraints.
 """
 def __init__(self,ids,e,directed=False):
     """Initialize _Graph with the inputs and resource settings below.

     Parameters
     ----------
     ids : sequence of str or None
         Animal identifiers in the required row order. IDs must align with the genotype/phenotype
         source; None uses source identifiers or the routine's positional indexing convention.

     e : object (see the cited input contract)
         Input for the operation ``'relation' in e``. See ``vigor.network.gnn._Graph.__init__``
         and its caller for the scientific representation; the shown operation constrains the
         required type/shape.

     directed : bool
         Context/control value read by the enclosing operation; its contract follows the explicit
         local use and the caller. Default: ``False``.
     """
     self.node_ids=pd.Index(ids).astype(str);self.edge_table=e;self.directed=directed;self.relations=list(pd.unique(e.relation.astype(str))) if "relation" in e else []
 def csr(self,weight=None,relation=None,directed=False):
  """Assemble the sparse relation adjacency in compressed-row format.

  Parameters
  ----------
  weight : object (see the cited input contract)
      Input supplied to ``edge_table_to_csr`` as ``cost_col``. See that callable's documented
      ``cost_col`` contract for its type, alignment and value domain. This wrapper retains
      that contract. Default: ``None``. None delegates to the local fallback shown in this
      body; it is not an additional numeric/category value. Passed to ``edge_table_to_csr`` as
      ``cost_col``.

  relation : object (see the cited input contract)
      Input for the operation ``e.relation.astype(str).eq(str(relation)).to_numpy()``. See
      ``vigor.network.gnn._Graph.csr`` and its caller for the scientific representation; the
      shown operation constrains the required type/shape. Default: ``None``. None delegates to
      the local fallback shown in this body; it is not an additional numeric/category value.

  directed : bool
      Input supplied to ``edge_table_to_csr`` as ``directed``. See that callable's documented
      ``directed`` contract for its type, alignment and value domain. This wrapper retains
      that contract. Default: ``False``. Passed to ``edge_table_to_csr`` as ``directed``.

  Returns
  -------
  object
      Computed result from ``a``; see the summary and the called implementation for its labeled/numerical representation.
  """
  e=self.edge_table;use=np.ones(len(e),bool) if relation is None else e.relation.astype(str).eq(str(relation)).to_numpy();a,_,_=edge_table_to_csr(e.loc[use],nodes=self.node_ids,cost_col=weight if weight in e else "__missing__",directed=directed or self.directed);return a
def as_sparse_graph(graph,node_ids=None):
 """Convert inputs to sparse graph.

 Parameters
 ----------
 graph : networkx.Graph or networkx.DiGraph
     Graph whose nodes and edges define the network/path relationships processed by this
     operation. Node identifiers must match the supplied metadata.

 node_ids : sequence, index or scalar
     Axis/node identifiers or positions selected by the local lookup below. Values must refer
     to the supplied object and preserve the requested output ordering. Default: ``None``.
     None delegates to the local fallback shown in this body; it is not an additional
     numeric/category value.

 Returns
 -------
 object
     Computed result from ``_Graph(ids, pd.DataFrame({'source': ids.take(a.row), 'target': ids.take(a.col), 'confidence': a.data}))``; see the summary and the called implementation for its labeled/numerical representation.
 """
 if hasattr(graph,"csr"):return graph
 if isinstance(graph,pd.DataFrame):
  e=graph.copy();e.source=e.source.astype(str);e.target=e.target.astype(str);ids=pd.Index(node_ids).astype(str) if node_ids is not None else pd.Index(pd.unique(pd.concat([e.source,e.target],ignore_index=True)));return _Graph(ids,e)
 a=sparse.coo_matrix(graph);ids=pd.Index(node_ids if node_ids is not None else np.arange(a.shape[0])).astype(str);return _Graph(ids,pd.DataFrame({"source":ids.take(a.row),"target":ids.take(a.col),"confidence":a.data}))


def require_gnn_dependencies():
    """GCN and R-GCN use VIGOR's core NumPy/SciPy dependencies.

    Returns
    -------
    dict
        Result fields: ``backend``, ``architectures``. Field arrays preserve the corresponding input axis order.
    """
    return {'backend':'numpy_csr','architectures':['gcn','rgcn']}


def validate_gnn_split(split,train_mask=None,val_mask=None,test_mask=None,*,transductive=False):
    """Validate gnn split.

    Parameters
    ----------
    split : object (see the cited input contract)
        Input for the operation ``sparse.csr_matrix(split)``. See
        ``vigor.network.gnn.validate_gnn_split`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    train_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    val_mask : object (see the cited input contract)
        Input for the operation ``any((x is None for x in (train_mask, val_mask, test_mask)))``.
        See ``vigor.network.gnn.validate_gnn_split`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    test_mask : object (see the cited input contract)
        Input for the operation ``any((x is None for x in (train_mask, val_mask, test_mask)))``.
        See ``vigor.network.gnn.validate_gnn_split`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    transductive : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    dict
        Result fields: ``train``, ``validation``, ``test``. Field arrays preserve the corresponding input axis order.
    """
    if train_mask is not None or val_mask is not None or test_mask is not None:
        if any(x is None for x in (train_mask,val_mask,test_mask)):
            raise ValueError('all split masks are required')
        A=sparse.csr_matrix(split);masks=[np.asarray(x,bool) for x in (train_mask,val_mask,test_mask)];n=A.shape[0]
        if A.shape!=(n,n) or any(m.shape!=(n,) for m in masks):raise ValueError('split masks must match adjacency')
        if any(np.any(masks[i]&masks[j]) for i,j in ((0,1),(0,2),(1,2))):raise ValueError('split masks overlap')
        def masked(keep):
            """Return the masked result computed by ``(M @ A @ M).tocsr``.

            Parameters
            ----------
            keep : array-like or numeric scalar
                Input for the operation ``sparse.diags(np.asarray(keep, bool).astype(float))``. See
                ``vigor.network.gnn.validate_gnn_split.masked`` and its caller for the scientific
                representation; the shown operation constrains the required type/shape.

            Returns
            -------
            object
                Computed result from ``(M @ A @ M).tocsr()``; see the summary and the called implementation for its labeled/numerical representation.
            """
            M=sparse.diags(np.asarray(keep,bool).astype(float));return (M@A@M).tocsr()
        if transductive:return {'train':A,'validation':A,'test':A}
        return {'train':masked(masks[0]),'validation':masked(masks[0]|masks[1]),'test':masked(masks[0]|masks[1]|masks[2])}
    if not isinstance(split,dict) or not (split.get('held_out_loci') or split.get('held_out_chromosomes')):
        raise ValueError('GNN evaluation requires held_out_loci or held_out_chromosomes')
    if split.get('random_edge_split',False):raise ValueError('random edge splits alone are invalid for GWAS evaluation')


def _messages(graph,architecture,relations=None):
    """Aggregate relation-specific neighbor messages for the graph neural network layer.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    architecture : str
        Graph neural-network architecture; 'rgcn' uses relation-specific graph convolution.
        Other names are accepted only if explicitly implemented by this network builder.
        Source-recognized literal values: ``gcn``. These select the corresponding branches
        below; comparison evidence alone does not establish an exhaustive enum.

    relations : object (see the cited input contract)
        Input for the operation ``labels != relations``. See ``vigor.network.gnn._messages`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    g=as_sparse_graph(graph);n=len(g.node_ids)
    if architecture=='gcn':
        a=g.csr(None,directed=False)+sparse.eye(n,format='csr')
        inv=1/np.sqrt(np.asarray(a.sum(1)).ravel())
        return [sparse.diags(inv)@a@sparse.diags(inv)],['all']
    if architecture!='rgcn':raise ValueError('architecture must be gcn or rgcn')
    labels=list(g.relations) if g.relations else ['all']
    if relations is not None and labels!=relations:raise ValueError('relation vocabulary/order differs from trained model')
    matrices=[sparse.eye(n,format='csr')]
    for rel in labels:
        a=g.csr('confidence',relation=rel,directed=g.directed).T.tocsr() if g.relations else g.csr('confidence',directed=g.directed).T.tocsr()
        degree=np.asarray(a.sum(1)).ravel();inv=np.divide(1.,degree,out=np.zeros(n),where=degree>0)
        matrices.append((sparse.diags(inv)@a).tocsr())
    return matrices,labels


def _feature_array(features,ids,names=None):
    """Align node features to graph node order and coerce to a numerical matrix.

    Parameters
    ----------
    features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.
    """
    if isinstance(features,pd.DataFrame):
        frame=features.copy();frame.index=frame.index.astype(str)
        if not frame.index.is_unique or (frame.index.get_indexer(ids)<0).any():raise ValueError('features must cover every graph node uniquely')
        if names is not None:frame=frame.loc[:,names]
        return frame.loc[ids].to_numpy(float),list(frame.columns)
    x=np.asarray(features,float)
    if x.ndim!=2 or x.shape[0]!=len(ids):raise ValueError('features must be nodes × features')
    if names is not None and len(names)!=x.shape[1]:raise ValueError('feature count differs')
    return x,list(range(x.shape[1])) if names is None else names


def _forward(matrices,x,parameters):
    """Evaluate graph-network layers and the configured output prediction transform.

    Parameters
    ----------
    matrices : array-like, mapping or fitted result
        Covariance quantity or component collection consumed by the stated transformation.
        Square matrices use a common trait order; uncertainty matrices match their corresponding
        covariance entries. See the caller for whether this is genetic, residual or total
        covariance.

    x : array-like
        Numerical operand in ``sum((a @ x @ w for a, w in zip(matrices, W)), np.zeros((len(x),
        W.shape[-1]))) + b``. Shape must conform to that matrix/vector operation and the
        enclosing prepared state.

    parameters : array-like or mapping
        Model parameter/prior-weight values in the fitting layout consumed here. Weights must
        align prior mixture states and parameters the declared coefficient slices; use the
        matching preparation routine.
    """
    W,V,b,c=parameters
    z=sum((a@x@w for a,w in zip(matrices,W)),np.zeros((len(x),W.shape[-1])))+b
    h=np.maximum(z,0)
    out=sum((a@h@v for a,v in zip(matrices,V)),np.zeros((len(x),V.shape[-1])))+c
    return out,h,z


@dataclass
class GNNResult:
    """GNNResult represents gnn result.

    Attributes
    ----------
    parameters : list
        Stored parameters; its axis/order follows the construction inputs.
    mean : np.ndarray
        Stored mean; its axis/order follows the construction inputs.
    scale : np.ndarray
        Stored scale; its axis/order follows the construction inputs.
    feature_names : list
        Stored feature names; its axis/order follows the construction inputs.
    architecture : str
        Stored architecture; its axis/order follows the construction inputs.
    task : str
        Stored task; its axis/order follows the construction inputs.
    classes : list
        Stored classes; its axis/order follows the construction inputs.
    relations : list
        Stored relations; its axis/order follows the construction inputs.
    predictions : pd.DataFrame
        Stored predictions; its axis/order follows the construction inputs.
    history : pd.DataFrame
        Stored history; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    parameters : list
        Stored parameters consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    mean : np.ndarray
        Stored mean consumed by ``predict``/``save``; see those methods for the numerical/label
        contract. Required construction field.

    scale : np.ndarray
        Stored scale consumed by ``predict``/``save``; see those methods for the numerical/label
        contract. Required construction field.

    feature_names : list
        Stored feature names consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    architecture : str
        Stored architecture consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    task : str
        Stored task consumed by ``predict``/``save``; see those methods for the numerical/label
        contract. Required construction field.

    classes : list
        Stored classes consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    relations : list
        Stored relations consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    predictions : pd.DataFrame
        Stored predictions consumed by ``predict``/``save``; see those methods for the
        numerical/label contract. Required construction field.

    history : pd.DataFrame
        Stored history consumed by ``save``; see those methods for the numerical/label contract.
        Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.
    """
    parameters: list
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list
    architecture: str
    task: str
    classes: list
    relations: list
    predictions: pd.DataFrame
    history: pd.DataFrame
    metadata: dict=field(default_factory=dict)

    def predict(self,graph,features):
        """Return the predict result computed by ``pd.DataFrame``.

        Parameters
        ----------
        graph : networkx.Graph or networkx.DiGraph
            Graph whose nodes and edges define the network/path relationships processed by this
            operation. Node identifiers must match the supplied metadata.

        features : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``pd.DataFrame(values, index=g.node_ids, columns=self.classes if self.task == 'classification' else self.predictions.columns)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        g=as_sparse_graph(graph,features.index if isinstance(features,pd.DataFrame) else None);x,_=_feature_array(features,g.node_ids,self.feature_names)
        x=np.where(np.isfinite(x),x,self.mean);x=(x-self.mean)/self.scale
        matrices,_=_messages(g,self.architecture,self.relations if self.architecture=='rgcn' else None)
        logits,_,_=_forward(matrices,x,self.parameters)
        values=softmax(logits,axis=1) if self.task=='classification' else logits
        return pd.DataFrame(values,index=g.node_ids,columns=self.classes if self.task=='classification' else self.predictions.columns)

    def save(self,path):
        """Return the save result computed by ``str``.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        Returns
        -------
        object
            Computed result from ``str(p)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        p=Path(path);p.mkdir(parents=True,exist_ok=True)
        import xarray as xr
        from ..store import VigorStore
        ds=xr.Dataset({'W':(('message','feature','hidden'),self.parameters[0]),
            'V':(('message','hidden','output'),self.parameters[1]),
            'b':(('hidden',),self.parameters[2]),'c':(('output',),self.parameters[3]),
            'mean':(('feature',),self.mean),'scale':(('feature',),self.scale)},
            coords={'feature':list(map(str,self.feature_names)),
                    'output':list(map(str,self.predictions.columns))})
        VigorStore(p/'parameters.zarr',mode='w').write_dataset(ds,'model')
        manifest={'feature_names':self.feature_names,'architecture':self.architecture,'task':self.task,'classes':self.classes,'relations':self.relations,'output_columns':list(self.predictions.columns),'metadata':self.metadata}
        (p/'model.json').write_text(json.dumps(manifest,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
        self.history.to_csv(p/'history.csv',index=False)
        return str(p)

    @classmethod
    def load(cls,path):
        """Return the load result computed by ``cls``.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        Returns
        -------
        object
            Computed result from ``cls(parameters, mean, scale, d['feature_names'], d['architecture'], d['task'], d['classes'], d['relations'], pd.DataFrame(columns=d['output_columns']), pd.read_csv(p / 'history.csv'), d['metadata'])``; see the summary and the called implementation for its labeled/numerical representation.
        """
        p=Path(path);d=json.loads((p/'model.json').read_text())
        if (p/'parameters.zarr').exists():
            import xarray as xr
            with xr.open_zarr(p/'parameters.zarr',group='model',consolidated=True) as a:
                parameters=[a[k].values.copy() for k in ['W','V','b','c']];mean=a['mean'].values.copy();scale=a['scale'].values.copy()
        else:
            with np.load(p/'parameters.npz',allow_pickle=False) as a:
                parameters=[a[k].copy() for k in ['W','V','b','c']];mean=a['mean'].copy();scale=a['scale'].copy()
        return cls(parameters,mean,scale,d['feature_names'],d['architecture'],d['task'],d['classes'],d['relations'],pd.DataFrame(columns=d['output_columns']),pd.read_csv(p/'history.csv'),d['metadata'])


def fit_heterogeneous_gnn(graph,features,labels,*,split,node_loci=None,node_chromosomes=None,
                          architecture='rgcn',task='classification',hidden_dim=16,epochs=200,
                          learning_rate=.01,weight_decay=1e-4,patience=30,random_state=0,
                          transductive=False,maxmem=None):
    """Train a two-layer GCN/R-GCN, with explicit locus/chromosome holdout.

    Held-out labels never enter the loss, preprocessing or early stopping.
    Optional validation_nodes must be separate from test/held-out groups.
    Default inductive training removes all edges incident to test/validation
    nodes. Prediction uses the full graph; report this inference graph policy.
    Features must exclude outcome-derived or held-out association evidence.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    features : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    labels : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    split : object (see the cited input contract)
        Input for the operation ``split[key]``. See ``vigor.network.gnn.fit_heterogeneous_gnn``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    node_loci : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    node_chromosomes : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    architecture : str
        Graph neural-network architecture; 'rgcn' uses relation-specific graph convolution.
        Other names are accepted only if explicitly implemented by this network builder.
        Default: ``'rgcn'``.

    task : str
        Input for the operation ``task not in ('classification', 'regression')``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'classification'``.

    hidden_dim : int
        Input for the operation ``rss + 12 * 8 * (n * (x.shape[1] + hidden_dim + outputs) +
        len(matrices) * (x.shape[1] * hidden_dim + hidden_dim * outputs)) + 3 *
        sum((a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in matrices))``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``16``.

    epochs : int
        Input for the operation ``int(epochs) + 1``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``200``.

    learning_rate : array-like or numeric scalar
        Input for the operation ``learning_rate * (first[j] / (1 - 0.9 ** epoch)) /
        (np.sqrt(second[j] / (1 - 0.999 ** epoch)) + 1e-08)``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.01``. Source comparisons: ``learning_rate <= 0``. See the rejection/fallback
        branches for enforcement.

    weight_decay : float
        Input for the operation ``grad + weight_decay * p``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``0.0001``. Source comparisons: ``weight_decay < 0``. See the rejection/fallback
        branches for enforcement.

    patience : int
        Input for the operation ``min(hidden_dim, epochs, patience) < 1``. See
        ``vigor.network.gnn.fit_heterogeneous_gnn`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``30``.
        Source comparisons: ``stale >= patience``. See the rejection/fallback branches for
        enforcement.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    transductive : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``GNNResult(parameters, mean, scale, names, architecture, task, classes, relations, pd.DataFrame(values, index=ids, columns=columns), pd.DataFrame(history), metadata)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    validate_gnn_split(split)
    if task not in ('classification','regression'):raise ValueError('unknown task')
    if min(hidden_dim,epochs,patience)<1 or learning_rate<=0 or weight_decay<0:raise ValueError('invalid training parameters')
    g=as_sparse_graph(graph,features.index if isinstance(features,pd.DataFrame) else None);ids=g.node_ids;n=len(ids);x,names=_feature_array(features,ids)
    hold=np.zeros(n,bool)
    for key,annotation in [('held_out_loci',node_loci),('held_out_chromosomes',node_chromosomes)]:
        if split.get(key):
            if annotation is None:raise ValueError(f'{key} requires matching node annotations')
            a=annotation.copy() if isinstance(annotation,pd.Series) else pd.Series(annotation,index=ids)
            a.index=a.index.astype(str);a=a.reindex(ids)
            if a.isna().any():raise ValueError('group annotations must cover every graph node')
            hold|=a.astype(str).isin(list(map(str,split[key]))).to_numpy()
    def mask(key):
        """Resolve the node selection mask used for training/evaluation.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        out=np.zeros(n,bool);idx=ids.get_indexer(list(map(str,split.get(key,[]))))
        if (idx<0).any():raise ValueError('unknown nodes in '+key)
        out[idx]=True;return out
    test=hold|mask('test_nodes');validation=mask('validation_nodes')
    train=mask('train_nodes') if 'train_nodes' in split else ~(test|validation)
    if not hold.any():raise ValueError('declared held-out groups select no graph nodes')
    if (train&(test|validation)).any() or (test&validation).any():raise ValueError('training/validation/test nodes overlap')
    if isinstance(labels,(pd.DataFrame,pd.Series)):
        y=labels.copy();y.index=y.index.astype(str)
        if not y.index.is_unique:raise ValueError('duplicate label node IDs')
        y=y.reindex(ids)
    else:y=pd.DataFrame(np.asarray(labels),index=ids)
    if task=='classification':
        y=np.asarray(y).reshape(-1)
        if len(y)!=n:raise ValueError('classification requires one label per node')
        labeled=pd.notna(y);classes=list(pd.unique(y[train&labeled]))
        if len(classes)<2:raise ValueError('training requires at least two classes')
        target=pd.Index(classes).get_indexer(y);labeled=target>=0;outputs=len(classes);columns=classes
    else:
        columns=list(y.columns) if isinstance(y,pd.DataFrame) else ['prediction']
        target=np.asarray(y,float)
        if target.ndim==1:target=target[:,None]
        labeled=np.isfinite(target).any(1);outputs=target.shape[1];classes=[]
    train&=labeled;validation&=labeled;test&=labeled
    if train.sum()<2 or not test.any():raise ValueError('need labeled training and held-out test nodes')
    finite=np.isfinite(x[train]);counts=finite.sum(0)
    mean=np.divide(np.where(finite,x[train],0).sum(0),counts,out=np.zeros(x.shape[1]),where=counts>0)
    x=np.where(np.isfinite(x),x,mean);scale=x[train].std(0);scale[scale<1e-8]=1.;x=(x-mean)/scale
    matrices,relations=_messages(g,architecture)
    budget,rss=memory_limit(maxmem)
    estimated=rss+12*8*(n*(x.shape[1]+hidden_dim+outputs)+len(matrices)*(x.shape[1]*hidden_dim+hidden_dim*outputs))+3*sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in matrices)
    if estimated>.85*budget:raise MemoryError('full-batch GNN does not fit maxmem; train on a selected subgraph')
    train_matrices=matrices
    validation_matrices=matrices
    if not transductive:
        # Training may not aggregate from validation or test nodes. Model
        # selection may use validation nodes, but must still hide test nodes.
        visible_train=~(test|validation); Mtrain=sparse.diags(visible_train.astype(float))
        train_matrices=[(Mtrain@a@Mtrain).tocsr() for a in matrices]
        visible_validation=~test; Mval=sparse.diags(visible_validation.astype(float))
        validation_matrices=[(Mval@a@Mval).tocsr() for a in matrices]
    rng=np.random.default_rng(random_state);R=len(matrices)
    parameters=[rng.normal(0,np.sqrt(2/(x.shape[1]*R)),(R,x.shape[1],hidden_dim)),rng.normal(0,np.sqrt(2/(hidden_dim*R)),(R,hidden_dim,outputs)),np.zeros(hidden_dim),np.zeros(outputs)]
    first=[np.zeros_like(p) for p in parameters];second=[np.zeros_like(p) for p in parameters]
    def loss_and_grad(out,selected):
        """Evaluate the graph-network training loss and parameter gradients.

        Parameters
        ----------
        out : array-like or numeric scalar
            Input for the operation ``out - np.nan_to_num(target)``. See
            ``vigor.network.gnn.fit_heterogeneous_gnn.loss_and_grad`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        selected : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Returns
        -------
        object
            Computed result from ``(loss, grad)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        grad=np.zeros_like(out)
        if task=='classification':
            z=out[selected];p=softmax(z,axis=1);loss=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),target[selected]]))
            p[np.arange(len(z)),target[selected]]-=1;grad[selected]=p/len(z)
        else:
            valid=selected[:,None]&np.isfinite(target);delta=np.where(valid,out-np.nan_to_num(target),0)
            loss=float(np.sum(delta*delta)/(2*valid.sum()));grad=delta/valid.sum()
        return loss,grad
    history=[];best=np.inf;best_parameters=None;stale=0
    for epoch in range(1,int(epochs)+1):
        if memory_limit(maxmem)[1]>budget:raise MemoryError('GNN exceeded maxmem')
        out,h,z=_forward(train_matrices,x,parameters);loss,dout=loss_and_grad(out,train)
        W,V,b,c=parameters;dV=np.empty_like(V);dh=np.zeros_like(h)
        for r,a in enumerate(train_matrices):
            back=a.T@dout;dV[r]=h.T@back;dh+=back@V[r].T
        dz=dh*(z>0);dW=np.array([x.T@(a.T@dz) for a in train_matrices])
        gradients=[dW,dV,dz.sum(0),dout.sum(0)]
        for j,(p,grad) in enumerate(zip(parameters,gradients)):
            if j<2:grad=grad+weight_decay*p
            first[j]=.9*first[j]+.1*grad;second[j]=.999*second[j]+.001*grad*grad
            p-=learning_rate*(first[j]/(1-.9**epoch))/(np.sqrt(second[j]/(1-.999**epoch))+1e-8)
        if validation.any():
            valout,_,_=_forward(validation_matrices,x,parameters);monitor,_=loss_and_grad(valout,validation)
        else:monitor=loss
        history.append({'epoch':epoch,'train_loss':loss,'validation_loss':monitor if validation.any() else np.nan})
        if monitor<best-1e-8:best=monitor;best_parameters=[p.copy() for p in parameters];stale=0
        else:stale+=1
        if validation.any() and stale>=patience:break
    parameters=best_parameters if validation.any() else parameters
    out,_,_=_forward(matrices,x,parameters);test_loss,_=loss_and_grad(out,test)
    values=softmax(out,axis=1) if task=='classification' else out
    metric=float(np.mean(values[test].argmax(1)==target[test])) if task=='classification' else float(np.sqrt(2*test_loss))
    metadata={'backend':'numpy_csr','architecture':architecture,'task':task,'seed':random_state,'transductive_training':transductive,'model_selection_graph':('full supplied graph' if transductive else 'test-masked graph'),'inference_graph':'full supplied graph','test_metric':'accuracy' if task=='classification' else 'rmse','test_score':metric,'train_nodes':list(ids[train]),'validation_nodes':list(ids[validation]),'test_nodes':list(ids[test]),'split':split,'epochs':len(history),'estimated_peak_bytes':int(estimated),'maxmem_bytes':budget,'feature_leakage_contract':'caller excludes outcome-derived/test-association features and edges'}
    return GNNResult(parameters,mean,scale,names,architecture,task,classes,relations,pd.DataFrame(values,index=ids,columns=columns),pd.DataFrame(history),metadata)


__all__=['require_gnn_dependencies','validate_gnn_split','fit_heterogeneous_gnn','GNNResult']
