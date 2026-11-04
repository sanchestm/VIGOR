"""Genome annotation I/O with explicit assembly provenance and bounded HTTP behavior."""
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
import time,json
import pandas as pd

@dataclass
class GenomeAnnotationBundle:
    """GenomeAnnotationBundle represents genome annotation bundle.

    Attributes
    ----------
    genes : pd.DataFrame
        Stored genes; its axis/order follows the construction inputs.
    transcripts : pd.DataFrame
        Stored transcripts; its axis/order follows the construction inputs.
    exons : pd.DataFrame
        Stored exons; its axis/order follows the construction inputs.
    accession : str | None
        Stored accession; its axis/order follows the construction inputs.
    species : str | None
        Stored species; its axis/order follows the construction inputs.
    provenance : dict
        Stored provenance; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    genes : pd.DataFrame
        Stored genes supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    transcripts : pd.DataFrame
        Stored transcripts supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: a fresh value from
        ``pd.DataFrame()`` for each instance.

    exons : pd.DataFrame
        Stored exons supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: a fresh value from ``pd.DataFrame()`` for
        each instance.

    accession : str | None
        Stored accession supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: ``None``.

    species : str | None
        Stored species supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``None``.

    provenance : dict
        Stored provenance supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: a fresh value from ``dict()`` for
        each instance.
    """
    genes: pd.DataFrame
    transcripts: pd.DataFrame=field(default_factory=pd.DataFrame)
    exons: pd.DataFrame=field(default_factory=pd.DataFrame)
    accession: str|None=None
    species: str|None=None
    provenance: dict=field(default_factory=dict)

def read_sequence_report(path):
    """Read sequence report.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    Returns
    -------
    object
        Computed result from ``pd.read_csv(path, sep='\\t', comment='#')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return pd.read_csv(path,sep='\t',comment='#')

def read_gtf(path):
    """Read gtf.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    Returns
    -------
    object
        Computed result from ``d``; see the summary and the called implementation for its labeled/numerical representation.
    """
    cols=['seqname','source','feature','start','end','score','strand','frame','attribute'];d=pd.read_csv(path,sep='\t',comment='#',names=cols)
    def attr(s,key):
        """Extract one quoted GTF attribute into a string-valued annotation column.

        Parameters
        ----------
        s : array-like
            Numerical operand in ``s.astype(str).str.extract(f'{key}[ ="]+([^";]+)')[0]``. Shape
            must conform to that matrix/vector operation and the enclosing prepared state.

        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``m``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m=s.astype(str).str.extract(fr'{key}[ ="]+([^";]+)')[0];return m
    d['gene_id']=attr(d.attribute,'gene_id');d['gene_name']=attr(d.attribute,'gene_name');d['transcript_id']=attr(d.attribute,'transcript_id');d['chrom']=d.seqname.astype(str).str.replace('chr','',regex=False)
    return d

def normalize_annotation_tables(gtf,*,accession=None,species=None):
    """Normalize annotation tables.

    Parameters
    ----------
    gtf : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    accession : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.

    species : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``GenomeAnnotationBundle(genes, tx, ex, accession, species, {'coordinate_system': '1-based closed', 'source': 'GTF'})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=read_gtf(gtf) if not isinstance(gtf,pd.DataFrame) else gtf.copy();
    genes=d[d.feature.eq('gene')][['gene_id','gene_name','chrom','start','end','strand']].copy();tx=d[d.feature.eq('transcript')].copy();ex=d[d.feature.eq('exon')].copy()
    return GenomeAnnotationBundle(genes,tx,ex,accession,species,{'coordinate_system':'1-based closed','source':'GTF'})

def load_genome_annotation(path,*,accession=None,species=None):
    """Load genome annotation.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    accession : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.
        Passed to ``normalize_annotation_tables`` as ``accession``.

    species : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.
        Passed to ``normalize_annotation_tables`` as ``species``.

    Returns
    -------
    object
        Computed result from ``normalize_annotation_tables(path, accession=accession, species=species)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return normalize_annotation_tables(path,accession=accession,species=species)

def fetch_genome_annotation(url,destination,*,timeout=30,retries=3,backoff=1.5,accession=None,species=None):
    """Download a genome annotation with retries and normalize its gene/transcript tables.

    Parameters
    ----------
    url : str
        Resource URL fetched by this downloader. It must point to the required genome
        annotation/resource format.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    timeout : float
        Maximum wait in seconds for each network/file request, as passed to the request backend.
        Default: ``30``. Passed to ``requests.get`` as ``timeout``.

    retries : int
        Maximum number of request attempts; positive integers permit retry after a failed
        resource transfer. Default: ``3``.

    backoff : float
        Delay multiplier used between repeated resource-transfer attempts; positive values
        control exponential waiting. Default: ``1.5``.

    accession : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.
        Passed to ``load_genome_annotation`` as ``accession``.

    species : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.
        Passed to ``load_genome_annotation`` as ``species``.

    Returns
    -------
    object
        Computed result from ``bundle``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import requests
    dest=Path(destination);last=None
    for i in range(int(retries)):
        try:
            r=requests.get(url,timeout=float(timeout));r.raise_for_status();dest.write_bytes(r.content);break
        except Exception as e:
            last=e
            if i+1>=int(retries):raise
            time.sleep(float(backoff)**i)
    bundle=load_genome_annotation(dest,accession=accession,species=species);bundle.provenance.update({'url':url,'fetched_at':pd.Timestamp.utcnow().isoformat(),'timeout_s':timeout,'retries':retries})
    return bundle

def query_genes(bundle_or_table,chrom,start,end,*,assembly=None):
    """Return the query genes result computed by ``g[(g.chrom.astype(str).str.replace('chr', '', regex=False) == c) & (g.end >= int(start)) & (g.start <= int(end))].copy``.

    Parameters
    ----------
    bundle_or_table : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    assembly : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``g[(g.chrom.astype(str).str.replace('chr', '', regex=False) == c) & (g.end >= int(start)) & (g.start <= int(end))].copy()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(bundle_or_table,GenomeAnnotationBundle):
        if assembly is not None and bundle_or_table.accession is not None and str(assembly)!=str(bundle_or_table.accession):raise ValueError(f'assembly mismatch: {assembly} != {bundle_or_table.accession}')
        g=bundle_or_table.genes
    else:g=bundle_or_table
    c=str(chrom).replace('chr','');return g[(g.chrom.astype(str).str.replace('chr','',regex=False)==c)&(g.end>=int(start))&(g.start<=int(end))].copy()

def pack_gene_track(genes):
    """Pack gene track.

    Parameters
    ----------
    genes : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    Returns
    -------
    object
        Computed result from ``genes.sort_values(['chrom', 'start', 'end']).reset_index(drop=True)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return genes.sort_values(['chrom','start','end']).reset_index(drop=True)

def normalize_orthology(table):
    """Normalize orthology.

    Parameters
    ----------
    table : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    Returns
    -------
    object
        Computed result from ``d.drop_duplicates(list(req)).reset_index(drop=True)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d=table.copy();req={'source_gene','target_gene'}
    if not req.issubset(d):raise ValueError('orthology table needs source_gene,target_gene')
    return d.drop_duplicates(list(req)).reset_index(drop=True)
def attach_orthology(genes,orthology):
    """Attach orthology.

    Parameters
    ----------
    genes : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    orthology : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene.

    Returns
    -------
    object
        Computed result from ``genes.merge(normalize_orthology(orthology), left_on='gene_id', right_on='source_gene', how='left')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return genes.merge(normalize_orthology(orthology),left_on='gene_id',right_on='source_gene',how='left')
