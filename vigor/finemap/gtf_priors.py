"""Truth-blind GTF feature priors and uncalibrated SNP-to-gene summaries.

GTF overlaps are genomic position annotations, never regulatory target proof.
"""
from __future__ import annotations

import gzip
from pathlib import Path
import re

import numpy as np
import pandas as pd

_FEATURES = ('intergenic', 'noncoding_gene', 'gene_body', 'exon', 'CDS')
_DEFAULT_WEIGHTS = {'intergenic': 1.0, 'noncoding_gene': 1.2,
                    'gene_body': 1.8, 'exon': 3.0, 'CDS': 5.0}
_GTF_ATTR = re.compile(r'(\w+)\s+"([^"]*)"')


def gtf_variant_priors(variants, gtf_path, *, variant_assembly=None,
                       annotation_assembly=None, contig_aliases=None,
                       allow_unverified_assembly=False, feature_weights=None):
    """Read GTF and compute variant priors from exon/CDS/gene-body overlap.

    Parameters
    ----------
    variants : pandas.DataFrame
        One row per variant, indexed by unique SNP identifier, with chrom and
        pos (or position) columns. Optional ref allele supplies variant span.
        Variants must be 1-based, with assembly coordinates matching the GTF.
    gtf_path : str or pathlib.Path
        Plain or gzip-compressed GTF with 1-based inclusive coordinates.
    variant_assembly, annotation_assembly : str or None
        Explicit assembly identifiers. Nonidentical names are rejected.
        Both must be known unless allow_unverified_assembly=True.
    contig_aliases : mapping or None
        Explicit map from input chromosome names to GTF seqid labels.
    allow_unverified_assembly : bool
        Explicit override for experimental unmatched provenance; not verification.
    feature_weights : mapping or None
        Positive weights for all five _FEATURES. These are assumptions,
        not learned functional evidence. Defaults (1, 1.2, 1.8, 3, 5).

    Returns
    -------
    pandas.DataFrame
        Input index order with chrom, pos, gtf_feature, overlapping_genes,
        prior_weight (normalized), and assembly_verified (based on supplied
        identifiers only). Gene annotations are not causal-gene probabilities.

    Complexity
    ----------
    Time O(F + F_span * P); memory O(P + O), with F streamed GTF rows,
    F_span the gene/exon/CDS rows overlapping the variants' span on their contig,
    P variants and O overlapping hits. A regional window keeps F_span small; the
    full GTF is still read once per call.
    """
    if not isinstance(variants, pd.DataFrame) or not variants.index.is_unique or len(variants)==0:
        raise ValueError('variants must be a nonempty unique-indexed DataFrame')
    if 'chrom' not in variants or not {'pos', 'position'}.intersection(variants.columns):
        raise ValueError('variants require chrom and pos/position columns')
    v=variants.copy()
    poscol='pos' if 'pos' in v else 'position'
    p=pd.to_numeric(v[poscol],errors='raise').to_numpy(float)
    if not np.isfinite(p).all() or np.any(p<1) or np.any(p!=np.floor(p)):
        raise ValueError('GTF variant positions must be positive 1-based integers')
    p=p.astype(np.int64)
    refs=v['ref'].fillna('').astype(str).to_numpy() if 'ref' in v else ['']*len(v)
    ends=p+np.maximum([len(s) for s in refs],1)-1
    chrom=v.chrom.astype(str).to_numpy()
    if not np.all(pd.Index(chrom).notna()):
        raise ValueError('chrom names must be present')
    matched=variant_assembly is not None and annotation_assembly is not None and str(variant_assembly)==str(annotation_assembly)
    if variant_assembly is not None and annotation_assembly is not None and not matched:
        raise ValueError('variant and annotation genome assemblies differ')
    if not matched and not allow_unverified_assembly:
        raise ValueError('explicit matched assemblies required, or allow_unverified_assembly=True')
    aliases={} if contig_aliases is None else {str(k):str(val) for k,val in contig_aliases.items()}
    target=np.asarray([aliases.get(c,c) for c in chrom],object)
    if np.any([not c for c in target]):
        raise ValueError('contig names cannot be blank')
    wc=_DEFAULT_WEIGHTS.copy()
    if feature_weights is not None:wc.update(feature_weights)
    if set(wc)!=set(_FEATURES) or not all(np.isfinite(x) and x>0 for x in wc.values()):
        raise ValueError('all five GTF feature weights must be finite and positive')
    code=np.zeros(len(v),dtype=np.uint8)
    symbols=[set() for _ in range(len(v))]
    used_contigs=set()
    target_contigs=set(target)
    span={c:(int(p[target==c].min()),int(ends[target==c].max())) for c in target_contigs}
    opener=gzip.open if str(gtf_path).endswith('.gz') else open
    with opener(Path(gtf_path),'rt') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=9:raise ValueError('GTF must contain nine tab-separated columns')
            contig,kind=fields[0],fields[2]
            if contig not in target_contigs:continue
            used_contigs.add(contig)
            if kind not in {'gene','exon','CDS'}:continue
            a,b=int(fields[3]),int(fields[4])
            if a<1 or b<a:raise ValueError('malformed GTF position')
            lo,hi=span[contig]
            if b<lo or a>hi:continue  # feature outside every variant on this contig
            hits=np.flatnonzero((target==contig)&(p<=b)&(ends>=a))
            if not len(hits):continue
            attrs=dict(_GTF_ATTR.findall(fields[8]))
            sym=attrs.get('gene') or attrs.get('gene_name') or attrs.get('gene_id') or ''
            if kind=='gene':
                level=2 if attrs.get('gene_biotype')=='protein_coding' else 1
                for j in hits:
                    code[j]=max(code[j],level)
                    if sym:symbols[j].add(sym)
            else:
                level=4 if kind=='CDS' else 3
                code[hits]=np.maximum(code[hits],level)
                if sym:
                    for j in hits:symbols[j].add(sym)
    if not set(target).issubset(used_contigs):
        raise ValueError('GTF lacks requested contigs: '+str(sorted(set(target)-used_contigs)))
    raw=np.array([wc[_FEATURES[int(x)]] for x in code],dtype=float)
    raw/=raw.sum()
    v['gtf_feature']=[_FEATURES[int(x)] for x in code]
    v['overlapping_genes']=[';'.join(sorted(x)) for x in symbols]
    v['prior_weight']=raw
    v['assembly_verified']=bool(matched)
    v['chrom']=chrom
    v['pos']=p
    return v


def gene_interval_scores(pips, annotations):
    """Aggregate variant PIPs by GTF-overlapping gene, without calling sums posteriors.

    Parameters
    ----------
    pips : pandas.Series
        SNP posterior inclusion probabilities indexed by variant name.
    annotations : pandas.DataFrame
        Output of gtf_variant_priors, aligned by the SNP index.

    Returns
    -------
    pandas.DataFrame
        Per-gene sum_pip, max_pip, number of SNPs and descending ranks.
        These summaries are NOT calibrated gene causal posterior probabilities.

    Complexity
    ----------
    Time O(P + O + G log G), memory O(P + O + G); O variant-to-gene overlap
    entries and G distinct gene symbols.
    """
    if not isinstance(pips,pd.Series) or not pips.index.is_unique:
        raise ValueError('pips must be uniquely SNP-indexed Series')
    vals=pips.reindex(annotations.index)
    if vals.isna().any() or not np.isfinite(vals).all() or ((vals<0)|(vals>1)).any():
        raise ValueError('PIPs must cover every annotated variant and lie in [0,1]')
    accum={}
    for pv,s in zip(vals.to_numpy(float),annotations.overlapping_genes.astype(str)):
        for g in s.split(';'):
            if not g:continue
            a=accum.setdefault(g,[])
            a.append(pv)
    if not accum:
        return pd.DataFrame(columns=['gene','n_variants','sum_pip','max_pip','sum_rank','max_rank'])
    z=pd.DataFrame([{'gene':g,'n_variants':len(v),'sum_pip':float(sum(v)),
                     'max_pip':float(max(v))} for g,v in accum.items()])
    z['sum_rank']=z.sum_pip.rank(ascending=False,method='min').astype(int)
    z['max_rank']=z.max_pip.rank(ascending=False,method='min').astype(int)
    z=z.sort_values(['sum_rank','gene']).reset_index(drop=True)
    z.attrs['interpretation']='uncalibrated_gene_level_SNP_PIP_summaries'
    return z


def equal_ld_group_prior(prior_weights, labels):
    """Equal prior mass per observed LD group, conditional SNP weights within.

    Returns
    -------
    numpy.ndarray
        Probability vector of length P, grouped mass exactly 1/G when no
        group has zero prior weight. This is an explicit modeling prior.
    Complexity
    ----------
    Time O(P), memory O(P), for P SNPs.
    """
    x=np.asarray(prior_weights,float)
    labels=pd.Index(labels)
    if x.ndim!=1 or len(x)!=len(labels) or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('invalid SNP priors or group labels')
    if labels.isna().any():raise ValueError('LD group labels must be complete')
    groups=pd.Series(np.arange(len(x)),index=np.arange(len(x))).groupby(labels,sort=False)
    out=np.zeros_like(x)
    for _,sub in groups:
        ix=sub.to_numpy(int)
        w=x[ix]
        if w.sum()<=0:raise ValueError('each LD group needs positive prior mass')
        out[ix]=w/w.sum()
    return out/out.sum()
