"""Export native fine-mapping SNP effects and variant-to-gene links to PCSF.

Variant PIP, posterior-mean beta and optional conditional-on-inclusion beta
are distinct quantities. GTF overlaps are *candidate* gene links only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..finemap.gtf_priors import gene_interval_scores


def finemap_pcsf_inputs(finemap_result, genotypes, variant_metadata, gene_annotations,
                       *, genotype_dosage_allele="A2", link_cost=0.05,
                       require_selected=True, min_pip=0.0, variant_prefix="variant:",
                       additional_links=None):
    """Build signed-beta variant attributes, PCSF edges and nonnegative prizes.

    Parameters
    ----------
    finemap_result : mapping
        Return from multitrait_bayes_finemap, or fine_map_region containing
        fits["bayes"]. Returned beta is per standardized dosage unit.
    genotypes : pandas.DataFrame
        The same SNP dosages and row order used for the fit, before centered
        genotype normalization. Missing dosages are mean-imputed on complete
        phenotype rows using the same fit contract.
    variant_metadata : pandas.DataFrame
        SNP-indexed table with A1/A2 allele fields and optional chrom/position.
        The source benchmark's PLINK1 two-bit decoding is **A2 allele dosage**.
    gene_annotations : pandas.DataFrame
        SNP-indexed result of gtf_variant_priors, with overlapping_genes field.
        Overlaps do not imply causal gene linkage.
    genotype_dosage_allele : {"A1","A2"}
        Which BIM allele is counted by numeric dosages in genotypes; required
        to report effect allele orientation correctly.
    link_cost : float
        Explicit positive, user-tunable heuristic cost of a GTF overlap link
        in the PCSF objective. NOT a probability or biological confidence.
    require_selected : bool
        Retain only SNPs belonging to evidence-filtered selected credible sets.
        If no selected sets, return no variant prizes/edges. When False, retain
        all SNPs with PIP >= min_pip.
    min_pip : float
        Nonnegative PIP threshold applied within selected CS when
        require_selected=True, or across all SNPs otherwise.
    variant_prefix : str
        Prefix variant nodes to prevent collisions with existing gene-network
        node identifiers. Gene nodes use literal annotated gene names.
    additional_links : pandas.DataFrame or None
        Independent SNP-to-gene links (eQTL, sQTL, regulatory links, or
        explicitly identified positional candidates) with required columns
        snp, gene, evidence_type, cost. Costs must be positive finite PCSF
        edge costs chosen by the caller, not derived from SNP beta. These
        links do not alter fine-mapping PIPs or gene ranks.

    Returns
    -------
    dict
        variant_nodes: one row per retained SNP with PIP, credible set IDs,
        allele orientation and trait-specific signed betas.
        variant_gene_edges: one row per GTF SNP-gene overlap with signed betas.
        pcsf_edges: source/target/cost, compatible with
        vigor.network.prize_collecting_steiner_forest.
        pcsf_prizes: Series named by "variant:SNP", with PIP as nonnegative prize.
        gene_ranks: gene PIP summary (NOT causal-gene posterior).
        metadata: effect scaling and evidence provenance.

    Complexity
    ----------
    O(n p + L p + number of returned gene overlaps) time for data validation,
    genotype scaling and filtering; O(n p + returned overlaps) transient memory.
    """
    f=(finemap_result['fits']['bayes'] if 'fits' in finemap_result else finemap_result)
    if not isinstance(genotypes,pd.DataFrame):
        raise TypeError('genotypes must be a labeled DataFrame')
    if not genotypes.columns.is_unique:
        raise ValueError('SNP names must be unique')
    snps=genotypes.columns.astype(str)
    if len(set(snps))!=len(snps):raise ValueError('SNP name strings must be unique')
    if genotype_dosage_allele not in {'A1','A2'}:
        raise ValueError('genotype_dosage_allele must be A1 or A2')
    if not np.isfinite(link_cost) or link_cost<=0:
        raise ValueError('link_cost must be finite and positive')
    if not np.isfinite(min_pip) or min_pip<0 or min_pip>1:
        raise ValueError('min_pip must lie in [0,1]')
    if not isinstance(variant_metadata,pd.DataFrame) or not isinstance(gene_annotations,pd.DataFrame):
        raise TypeError('variant_metadata and gene_annotations must be DataFrames')
    meta=variant_metadata.copy()
    ann=gene_annotations.copy()
    meta.index=meta.index.astype(str);ann.index=ann.index.astype(str)
    if not meta.index.is_unique or not ann.index.is_unique:
        raise ValueError('variant/annotation SNP identifiers must be unique')
    if not all(key in meta.columns for key in ('A1','A2')):
        raise ValueError('allele metadata must have A1 and A2')
    if 'overlapping_genes' not in ann:
        raise ValueError('gene_annotations requires overlapping_genes')
    if not set(snps).issubset(meta.index) or not set(snps).issubset(ann.index):
        raise ValueError('all genotype SNPs require aligned variant metadata and annotation')
    meta=meta.reindex(snps);ann=ann.reindex(snps)
    if meta[['A1','A2']].isna().any().any() or (meta.A1.astype(str)==meta.A2.astype(str)).any():
        raise ValueError('allele labels must be nonmissing and distinct')
    pip=f['pip'].set_index('snp').pip.reindex(snps).astype(float)
    beta=f['coef'].reindex(snps)
    if pip.isna().any() or beta.isna().any().any() or len(f['coef'])!=len(snps):
        raise ValueError('fine mapping SNP/trait labels or coefficients are incomplete')
    if not pip.between(0,1).all():
        raise ValueError('invalid posterior inclusion probabilities')
    rows=np.asarray(f['complete_rows'],bool)
    if rows.shape!=(len(genotypes),):
        raise ValueError('complete row mask must match the original genotype rows')
    if not np.any(rows):
        raise ValueError('no complete rows')
    # Exactly match _prepare_finemap_xy: impute means and use ddof=0 scale.
    x=genotypes.to_numpy(dtype=float)[rows]
    if np.isinf(x).any():raise ValueError('infinite dosages')
    with np.errstate(invalid='ignore'):
        mu=np.nanmean(x,axis=0)
    if not np.isfinite(mu).all():raise ValueError('entirely unobserved variant')
    x=np.where(np.isfinite(x),x,mu[None,:])
    sd=x.std(axis=0,ddof=0)
    sd=np.where(sd>1e-8,sd,1.)
    selected=f['selected_credible_sets']
    membership=selected.groupby('snp').effect.agg(lambda a:','.join(str(int(i)) for i in sorted(set(a)))) if len(selected) else pd.Series(dtype=str)
    retained=(pip.to_numpy()>=min_pip)
    if require_selected:
        retained &= np.isin(snps,membership.index)
    ids=np.flatnonzero(retained)
    traits=list(map(str,beta.columns))
    allelename=genotype_dosage_allele
    other='A1' if allelename=='A2' else 'A2'
    std=beta.to_numpy(dtype=float)
    dosage=std/sd[:,None]
    conditional=np.divide(dosage,pip.to_numpy()[:,None],
        out=np.zeros_like(dosage),where=pip.to_numpy()[:,None]>0)
    attrs=[]
    for j in ids:
        sid=snps[j]
        record={'snp':sid,'node':f'{variant_prefix}{sid}',
                'pip':float(pip.iloc[j]),
                'selected_effects':membership.get(sid,''),
                'effect_allele':str(meta.iloc[j][allelename]),
                'other_allele':str(meta.iloc[j][other]),
                'dosage_allele_field':allelename,'genotype_sd':float(sd[j]),
                'gtf_feature':str(ann.iloc[j].get('gtf_feature','')),
                'overlapping_genes':str(ann.iloc[j].overlapping_genes)}
        if 'chrom' in meta:record['chrom']=str(meta.iloc[j].chrom)
        if 'position' in meta:record['position']=int(meta.iloc[j].position)
        if 'pos' in meta:record['position']=int(meta.iloc[j].pos)
        for k,t in enumerate(traits):
            record[f'beta_standardized__{t}']=float(std[j,k])
            record[f'beta_per_{allelename}_dosage__{t}']=float(dosage[j,k])
            record[f'beta_per_{other}_dosage__{t}']=float(-dosage[j,k])
            record[f'beta_given_inclusion_per_{allelename}__{t}']=float(conditional[j,k])
        attrs.append(record)
    nodes=pd.DataFrame(attrs)
    ed=[]
    for record in attrs:
        for gene in record['overlapping_genes'].split(';'):
            gene=gene.strip()
            if not gene:continue
            row={'source':record['node'],'target':gene,'cost':float(link_cost),
                 'evidence_type':'gtf_interval_overlap_not_causal',
                 'snp':record['snp'],'gene':gene,'pip':record['pip'],
                 'selected_effects':record['selected_effects'],
                 'effect_allele':record['effect_allele']}
            for k in traits:
                row[f'beta_per_{allelename}_dosage__{k}']=record[f'beta_per_{allelename}_dosage__{k}']
                row[f'beta_given_inclusion_per_{allelename}__{k}']=record[f'beta_given_inclusion_per_{allelename}__{k}']
            ed.append(row)
    if additional_links is not None:
        if not isinstance(additional_links,pd.DataFrame) or not {'snp','gene','evidence_type','cost'}.issubset(additional_links.columns):
            raise ValueError('additional_links requires snp, gene, evidence_type, cost columns')
        # Only links to already selected credible-set variants are exported.
        by_snp={r['snp']:r for r in attrs}
        for link in additional_links.itertuples(index=False):
            sid=str(link.snp)
            if sid not in by_snp:continue
            gene=str(link.gene).strip();etype=str(link.evidence_type).strip()
            cost=float(link.cost)
            if not gene or gene.lower()=='nan' or not etype or etype.lower()=='nan' or not np.isfinite(cost) or cost<=0:
                raise ValueError('additional_links must have valid gene, provenance and positive costs')
            rec=by_snp[sid]
            entry={'source':rec['node'],'target':gene,'cost':cost,
                   'evidence_type':etype,'snp':sid,'gene':gene,
                   'pip':rec['pip'],'selected_effects':rec['selected_effects'],
                   'effect_allele':rec['effect_allele']}
            for trait in traits:
                entry[f'beta_per_{allelename}_dosage__{trait}']=rec[f'beta_per_{allelename}_dosage__{trait}']
                entry[f'beta_given_inclusion_per_{allelename}__{trait}']=rec[f'beta_given_inclusion_per_{allelename}__{trait}']
            ed.append(entry)
    edge_columns=['source','target','cost','evidence_type','snp','gene','pip',
                  'selected_effects','effect_allele']
    edges=pd.DataFrame(ed,columns=edge_columns+[f'beta_per_{allelename}_dosage__{k}' for k in traits]+[f'beta_given_inclusion_per_{allelename}__{k}' for k in traits])
    pcsf_edges=edges[['source','target','cost']].copy()
    prize=pd.Series({record['node']:record['pip'] for record in attrs},dtype=float,name='prize')
    # Ranks summarize ALL SNPs. PCSF evidence retains only selected variants.
    gene_ranks=gene_interval_scores(pip,ann)
    return {'variant_nodes':nodes,'variant_gene_edges':edges,
            'pcsf_edges':pcsf_edges,'pcsf_prizes':prize,
            'gene_ranks':gene_ranks,
            'metadata':{'beta_standardized':'posterior mean effect per 1 SD of mean-imputed genotype',
                        'beta_per_dosage':'posterior mean effect per counted allele; DO NOT multiply by PIP again',
                        'beta_given_inclusion':'posterior mean divided by PIP, not an extra posterior probability',
                        'effect_allele_source':allelename,
                        'phenotype_units':'same units as supplied phenotype columns',
                        'gene_links':'GTF overlap candidates, not causal targets',
                        'gene_ranks':'uncalibrated aggregation of SNP PIPs, subject to LD/gene-length bias',
                        'prize_policy':'SNP PIP, gene node prize=0; no double counting',
                        'edge_cost_policy':f'uniform user-specified {link_cost}; not a biological evidence likelihood',
                        'requires_selected_sets':bool(require_selected),
                        'min_pip':float(min_pip),'n_retained_variants':int(len(nodes)),
                        'n_variant_gene_edges':int(len(edges)),
                        'n_independent_annotation_edges':int((edges.evidence_type!='gtf_interval_overlap_not_causal').sum()),
                        'trait_names':traits}}
