from __future__ import annotations
import numpy as np,pandas as pd

def build_proximal_gene_edges(gene_table,*,max_distance_bp=1_000_000,min_distance_bp=0,distance_scale_bp=250_000,species=None,assembly=None,max_edges_per_gene=None):
    """Build proximal gene edges.

    Parameters
    ----------
    gene_table : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.

    min_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``0``.

    distance_scale_bp : int
        Input for the operation ``float(distance_scale_bp)``. See
        ``vigor.integration.proximity.build_proximal_gene_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``250000``.

    species : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.

    assembly : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.

    max_edges_per_gene : object (see the cited input contract)
        Input for the operation ``sorted(cand)[:int(max_edges_per_gene)]``. See
        ``vigor.integration.proximity.build_proximal_gene_edges`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(rows)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=gene_table.copy();req={'gene_id','chrom','start','end'}
    if not req.issubset(d):raise ValueError(f'gene_table needs {sorted(req)}')
    rows=[];scale=float(distance_scale_bp)
    for chrom,g in d.groupby('chrom'):
        g=g.sort_values('start').reset_index(drop=True);starts=g.start.to_numpy(int)
        for i,left in g.iterrows():
            stop=np.searchsorted(starts,int(left.end)+int(max_distance_bp),side='right');cand=[]
            for j in range(i+1,stop):
                right=g.iloc[j];dist=max(0,int(right.start)-int(left.end))
                if int(min_distance_bp)<=dist<=int(max_distance_bp):cand.append((dist,str(right.gene_id)))
            if max_edges_per_gene is not None:cand=sorted(cand)[:int(max_edges_per_gene)]
            for dist,rid in cand:
                conf=float(np.exp(-dist/scale));rows.append({'source':f'gene:{left.gene_id}','target':f'gene:{rid}','relation':'gene_proximity','confidence':conf,'cost':float(-np.log(max(conf,1e-12))),'distance_bp':dist,'linkage_context':True,'causal_evidence':False,'species':species,'assembly':assembly})
    return pd.DataFrame(rows)
