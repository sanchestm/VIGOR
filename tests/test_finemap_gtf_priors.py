"""Focused regression tests for opt-in positional GTF priors and LD-group prior."""
from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from vigor.finemap.gtf_priors import (gtf_variant_priors, gene_interval_scores,
                                      equal_ld_group_prior)
from vigor.multivariate import fine_map_region


def _tiny_gtf(path):
    lines=[
        'NC_test.1\tNCBI\tgene\t100\t190\t.\t+\t.\tgene_id "id_A"; gene "GeneA"; gene_biotype "protein_coding";',
        'NC_test.1\tNCBI\texon\t120\t160\t.\t+\t.\tgene_id "id_A"; gene "GeneA";',
        'NC_test.1\tNCBI\tCDS\t130\t149\t.\t+\t0\tgene_id "id_A"; gene "GeneA";',
        'NC_test.1\tNCBI\tgene\t180\t240\t.\t+\t.\tgene_id "id_B"; gene "GeneB"; gene_biotype "lncRNA";',
        'NC_test.1\tNCBI\texon\t200\t205\t.\t+\t.\tgene_id "id_B"; gene "GeneB";',
    ]
    with gzip.open(path,'wt') as f:f.write('\n'.join(lines)+'\n')


def test_gtf_feature_priors_and_ambiguous_gene_overlap(tmp_path):
    path=tmp_path/'tiny.gtf.gz';_tiny_gtf(path)
    names=['z4','z2','z5','z1','z3','z0']
    poss=[300,125,185,135,202,99]
    v=pd.DataFrame({'chrom':['12']*6,'pos':poss},index=names)
    a=gtf_variant_priors(v,path,variant_assembly='test',annotation_assembly='test',
                           contig_aliases={'12':'NC_test.1'})
    assert a.index.tolist()==names
    assert a.gtf_feature.tolist()==['intergenic','exon','gene_body','CDS','exon','intergenic']
    assert a.loc['z5','overlapping_genes']=='GeneA;GeneB'
    assert a.loc['z1','overlapping_genes']=='GeneA'
    assert a.loc['z4','overlapping_genes']==''
    assert a.assembly_verified.all()
    expected=np.array([1.,3.,1.8,5.,3.,1.])
    assert_allclose(a.prior_weight.to_numpy(),expected/expected.sum())
    gene=gene_interval_scores(pd.Series([.01,.1,.02,.4,.05,.01],index=names),a)
    assert gene.set_index('gene').loc['GeneA','sum_pip']==pytest.approx(.52)
    assert gene.set_index('gene').loc['GeneB','sum_pip']==pytest.approx(.07)
    assert gene.attrs['interpretation'].startswith('uncalibrated')


def test_gtf_assembly_contig_and_metadata_policy(tmp_path):
    path=tmp_path/'tiny.gtf.gz';_tiny_gtf(path)
    v=pd.DataFrame({'chrom':['12'],'pos':[125]},index=['12:125'])
    with pytest.raises(ValueError,match='assemblies'):
        gtf_variant_priors(v,path,contig_aliases={'12':'NC_test.1'})
    with pytest.raises(ValueError,match='differ'):
        gtf_variant_priors(v,path,variant_assembly='rn7',annotation_assembly='GRCr8',
                           contig_aliases={'12':'NC_test.1'})
    with pytest.raises(ValueError,match='requested contigs'):
        gtf_variant_priors(v,path,allow_unverified_assembly=True)
    a=gtf_variant_priors(v,path,allow_unverified_assembly=True,
                         contig_aliases={'12':'NC_test.1'})
    assert a.assembly_verified.tolist()==[False]


def test_equal_group_prior_is_explicit_and_normalized():
    p=equal_ld_group_prior([2,4,8,1],['a','a','b','c'])
    assert_allclose([sum(p[:2]),p[2],p[3]],[1/3,1/3,1/3])
    assert_allclose(p[:2],[1/9,2/9])
    with pytest.raises(ValueError,match='group labels'):
        equal_ld_group_prior([1,1],['a',None])


def test_region_uses_gtf_prior_and_scores_without_claiming_gene_posterior(tmp_path):
    path=tmp_path/'tiny.gtf.gz';_tiny_gtf(path)
    rng=np.random.default_rng(13)
    n=80; positions=np.arange(110,210,5)
    x=pd.DataFrame(rng.binomial(2,.26,size=(n,len(positions))).astype(float),
      index=[f'rat{i}' for i in range(n)],columns=[f'12:{p}' for p in positions])
    y=pd.DataFrame({'t1':rng.normal(size=n)+.6*(x.iloc[:,4]-x.iloc[:,4].mean()),
                    't2':rng.normal(size=n)-.4*(x.iloc[:,4]-x.iloc[:,4].mean())},index=x.index)
    out=fine_map_region(x,y,methods=('bayes',),L=2,
        gtf_path=path,gtf_contig_aliases={'12':'NC_test.1'},
        allow_unverified_assembly=True,bayes_kwargs={'max_iter':100,'tol':1e-4},
        use_evidence_priors=False,hdbscan=False)
    assert 'bayes' in out['fits']
    assert out['gene_scores'] is not None
    assert set(out['gene_scores'].gene)=={'GeneA','GeneB'}
    assert out['gtf_assembly_unverified'] is True
    assert_allclose(out['fits']['bayes']['prior_weights'].to_numpy(),
                    out['gene_annotations']['prior_weight'].to_numpy(),atol=1e-12)
