"""Indexed variant-gene candidate construction."""
from __future__ import annotations
import numpy as np,pandas as pd
from dataclasses import dataclass

@dataclass
class VariantGeneEvidenceSet:
    """VariantGeneEvidenceSet represents variant gene evidence set.

    Attributes
    ----------
    edges : pd.DataFrame
        Stored edges; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    edges : pd.DataFrame
        Stored edges supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Required
        construction field.
    """
    edges: pd.DataFrame
    metadata: dict

def build_variant_gene_evidence(variants,genes,*,window_bp=1_000_000,variant_id='snp',chrom='chrom',pos='pos',genome_build=None):
    """Build variant gene evidence.

    Parameters
    ----------
    variants : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader.

    genes : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    window_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.

    variant_id : str
        Input for the operation ``rows.append({'variant': str(getattr(r, variant_id)), 'gene':
        str(gr['gene_id']), 'chrom': str(c), 'distance_bp': int(dist), 'linkage_context': True,
        'causal_evidence': False, 'genome_build': genome_build})``. See
        ``vigor.integration.variant_gene.build_variant_gene_evidence`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``'snp'``.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``'chrom'``.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``'pos'``.

    genome_build : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``VariantGeneEvidenceSet(pd.DataFrame(rows), {'window_bp': int(window_bp), 'genome_build': genome_build})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    v=variants.copy();g=genes.copy();rows=[]
    if 'gene_id' not in g:raise ValueError('genes needs gene_id')
    for c,vg in v.groupby(chrom):
        gg=g[g[chrom].astype(str)==str(c)].sort_values('start');starts=gg.start.to_numpy(int);ends=gg.end.to_numpy(int);records=gg.to_dict('records')
        for r in vg.itertuples(index=False):
            p=int(getattr(r,pos));lo=np.searchsorted(starts,p-int(window_bp),side='left');hi=np.searchsorted(starts,p+int(window_bp),side='right')
            for j in range(lo,hi):
                gr=records[j];dist=0 if int(gr['start'])<=p<=int(gr['end']) else min(abs(p-int(gr['start'])),abs(p-int(gr['end'])))
                if dist<=int(window_bp):rows.append({'variant':str(getattr(r,variant_id)),'gene':str(gr['gene_id']),'chrom':str(c),'distance_bp':int(dist),'linkage_context':True,'causal_evidence':False,'genome_build':genome_build})
    return VariantGeneEvidenceSet(pd.DataFrame(rows),{'window_bp':int(window_bp),'genome_build':genome_build})

def summarize_variant_gene_evidence(evidence):
    """Summarize variant gene evidence.

    Parameters
    ----------
    evidence : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source.

    Returns
    -------
    object
        Computed result from ``d.groupby('gene', as_index=False).agg(n_variants=('variant', 'nunique'), min_distance_bp=('distance_bp', 'min')) if len(d) else pd.DataFrame(columns=['gene', 'n_variants', 'min_distance_bp'])``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=evidence.edges if isinstance(evidence,VariantGeneEvidenceSet) else evidence
    return d.groupby('gene',as_index=False).agg(n_variants=('variant','nunique'),min_distance_bp=('distance_bp','min')) if len(d) else pd.DataFrame(columns=['gene','n_variants','min_distance_bp'])
