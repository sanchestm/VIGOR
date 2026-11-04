from __future__ import annotations
from dataclasses import dataclass,field
import pandas as pd
OPTIMUSKG_DEFAULT='OptimusKG'
@dataclass
class KnowledgeGraphBundle:
    """KnowledgeGraphBundle represents knowledge graph bundle.

    Attributes
    ----------
    edges : pd.DataFrame
        Stored edges; its axis/order follows the construction inputs.
    nodes : pd.DataFrame
        Stored nodes; its axis/order follows the construction inputs.
    provenance : dict
        Stored provenance; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    edges : pd.DataFrame
        Stored edges supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    nodes : pd.DataFrame
        Stored nodes supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: a fresh value from ``pd.DataFrame()`` for
        each instance.

    provenance : dict
        Stored provenance supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: a fresh value from ``dict()`` for
        each instance.
    """
    edges: pd.DataFrame
    nodes: pd.DataFrame=field(default_factory=pd.DataFrame)
    provenance: dict=field(default_factory=dict)
def build_typed_knowledge_graph(edges,*,source_version=None,knowledge_source=None):
    """Build typed knowledge graph.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    source_version : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    knowledge_source : object (see the cited input contract)
        Input for the operation ``isinstance(knowledge_source, str)``. See
        ``vigor.integration.knowledge.build_typed_knowledge_graph`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``KnowledgeGraphBundle(d.reset_index(drop=True), provenance=prov)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=edges.copy() if isinstance(edges,pd.DataFrame) else pd.DataFrame(edges)
    if not {'source','target'}.issubset(d):raise ValueError('edge table requires source,target')
    if 'relation' not in d:d['relation']='related_to'
    prov={'knowledge_source':source_version if source_version is not None else (knowledge_source if isinstance(knowledge_source,str) else 'provided_table')}
    return KnowledgeGraphBundle(d.reset_index(drop=True),provenance=prov)
def load_optimuskg(path,*,columns=None,filters=None):
    # Prefer columnar predicate pushdown when parquet is available.
    """Load optimuskg.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    columns : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value. Passed to ``pd.read_parquet`` as
        ``columns``.

    filters : str, sequence or mapping
        Pattern/filter selectors applied to the resource/table search. Use patterns matching the
        required filenames/header or existing metadata fields. Default: ``None``. None delegates
        to the local fallback shown in this body; it is not an additional numeric/category
        value. Passed to ``pd.read_parquet`` as ``filters``.

    Returns
    -------
    object
        Computed result from ``pd.read_parquet(path, columns=columns, filters=filters)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return pd.read_parquet(path,columns=columns,filters=filters)
def load_knowledge_source(path,**kwargs):
    """Load knowledge source.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``load_optimuskg``. See those callables for
        accepted keys and values. Passed to ``load_optimuskg`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``load_optimuskg(path, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return load_optimuskg(path,**kwargs)
def fetch_knowledge_source(*args,**kwargs):
    """Download and normalize the requested biological interaction/evidence source.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments supplied by the enclosing callback or context protocol;
        their order and types follow that protocol.

    **kwargs : mapping
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.
    """
    raise NotImplementedError('remote knowledge fetch is intentionally explicit; download a versioned source and call load_knowledge_source')
