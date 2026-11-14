"""Native fine-map → PCSF graph evidence tests with allele orientation."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from vigor.integration.finemap_pcsf import finemap_pcsf_inputs


def _example():
    snps=['12:101','12:102','12:201','12:999']
    data=np.array([[0,1,2,0],[1,0,1,0],[2,1,0,1],[0,2,2,1],
                   [1,1,1,0],[2,0,0,2],[1,2,1,2]],dtype=float)
    X=pd.DataFrame(data,columns=snps)
    X.iloc[2,1]=np.nan  # mean-imputation of genotype on complete rows
    X.iloc[6,0]=np.nan  # excluded response row
    pip=np.array([.7,.15,.02,.13])
    beta=np.array([[.4,-.2],[.2,.1],[.05,-.02],[.08,.03]])
    Y=['RT','FA']
    fit={'pip':pd.DataFrame({'snp':snps,'pip':pip}),
         'coef':pd.DataFrame(beta,index=snps,columns=Y),
         'complete_rows':np.array([True,True,True,True,True,True,False]),
         'selected_credible_sets':pd.DataFrame({'effect':[1,1,2],
                                               'snp':snps[:3]})}
    metadata=pd.DataFrame({'A1':['A','C','A','T'],
                           'A2':['G','T','C','G'],
                           'chrom':['12']*4,'position':[101,102,201,999]},index=snps)
    ann=pd.DataFrame({'overlapping_genes':['GENEA','GENEA;GENEB','GENEB',''],
                      'gtf_feature':['CDS','gene_body','gene_body','intergenic']},index=snps)
    return X,fit,metadata,ann


def test_export_correct_posterior_betas_per_counted_allele_and_conditional():
    X,fit,meta,ann=_example()
    graph=finemap_pcsf_inputs(fit,X,meta,ann,genotype_dosage_allele='A2')
    v=graph['variant_nodes'].set_index('snp')
    assert v.index.tolist()==['12:101','12:102','12:201']
    assert len(graph['variant_gene_edges'])==4
    assert graph['pcsf_edges'].columns.tolist()==['source','target','cost']
    assert set(graph['pcsf_edges'].target)=={'GENEA','GENEB'}
    assert graph['pcsf_prizes']['variant:12:101']==pytest.approx(.7)
    complete=X.iloc[:6]
    mu=complete.mean(axis=0).fillna(0)
    xx=complete.fillna(mu)
    sd=xx.std(axis=0,ddof=0)
    b=fit['coef']
    for k in b.columns:
        observed=v[f'beta_per_A2_dosage__{k}'].to_numpy()
        expected=(b[k]/sd).loc[v.index].to_numpy()
        assert_allclose(observed,expected,atol=1e-14)
        assert_allclose(v[f'beta_per_A1_dosage__{k}'],-observed,atol=1e-14)
        assert_allclose(v[f'beta_given_inclusion_per_A2__{k}']*
                        v.pip,observed,atol=1e-14)
        allbetas=(b[k]/sd)
        dosage_prediction=(xx-mu)@allbetas
        standard_prediction=((xx-mu)/sd)@b[k]
        assert_allclose(dosage_prediction,standard_prediction,atol=1e-14)
    # No 'PIP*beta' double penalty and both gene links carry same allele beta.
    edges=graph['variant_gene_edges']
    ix=edges.snp.eq('12:102')
    assert edges.loc[ix,'beta_per_A2_dosage__RT'].nunique()==1
    assert all(edges.loc[ix,'effect_allele']=='T')
    assert graph['metadata']['gene_links'].startswith('GTF overlap')


def test_graph_gene_nodes_reuse_real_names_for_pcsf_and_no_double_counting():
    X,fit,meta,ann=_example()
    g=finemap_pcsf_inputs(fit,X,meta,ann)
    assert 'GENEA' in set(g['pcsf_edges'].target)
    assert g['pcsf_prizes'].index.str.startswith('variant:').all()
    assert all(g['pcsf_edges'].cost>0)
    assert g['gene_ranks'].set_index('gene').loc['GENEA','sum_pip']==pytest.approx(.85)
    assert g['gene_ranks'].set_index('gene').loc['GENEB','sum_pip']==pytest.approx(.17)


def test_no_credible_sets_yields_no_false_pcsf_node_prizes():
    X,fit,meta,ann=_example()
    fit['selected_credible_sets']=pd.DataFrame(columns=['snp','effect'])
    g=finemap_pcsf_inputs(fit,X,meta,ann)
    assert len(g['pcsf_edges'])==len(g['variant_nodes'])==len(g['pcsf_prizes'])==0
    loose=finemap_pcsf_inputs(fit,X,meta,ann,require_selected=False,min_pip=.1)
    assert len(loose['variant_nodes'])==3


def test_explicit_allele_source_and_alignment_are_required():
    X,fit,meta,ann=_example()
    with pytest.raises(ValueError,match='A1 or A2'):
        finemap_pcsf_inputs(fit,X,meta,ann,genotype_dosage_allele='unknown')
    with pytest.raises(ValueError,match='aligned'):
        finemap_pcsf_inputs(fit,X,meta.iloc[:-1],ann)
    with pytest.raises(ValueError,match='positive'):
        finemap_pcsf_inputs(fit,X,meta,ann,link_cost=0)
    with pytest.raises(ValueError,match='row mask'):
        changed=dict(fit,complete_rows=np.ones(3,bool))
        finemap_pcsf_inputs(changed,X,meta,ann)


def test_high_level_finemap_region_opt_in_graph_export(tmp_path):
    import gzip
    from vigor.multivariate import fine_map_region
    gtf=tmp_path/'test.gtf.gz'
    with gzip.open(gtf,'wt') as f:
        f.write('NC_test.1\tRefSeq\tgene\t100\t230\t.\t+\t.\tgene_id "G1"; gene "GENEA"; gene_biotype "protein_coding";\n')
    rng=np.random.default_rng(52);n=88
    positions=np.arange(100,240,10)
    X=pd.DataFrame(rng.binomial(2,.30,size=(n,len(positions))).astype(float),
       columns=[f'12:{p}' for p in positions],index=[f'rat{i}' for i in range(n)])
    Y=pd.DataFrame({'rt':rng.normal(size=n)+.7*X.iloc[:,2].to_numpy(),
                    'err':rng.normal(size=n)-.5*X.iloc[:,2].to_numpy()},index=X.index)
    v=pd.DataFrame({'chrom':'12','pos':positions,'A1':'A','A2':'G'},index=X.columns)
    r=fine_map_region(X,Y,methods=('bayes',),L=2,
      gtf_path=gtf,gtf_variant_metadata=v,gtf_contig_aliases={'12':'NC_test.1'},
      allow_unverified_assembly=True,use_evidence_priors=False,hdbscan=False,
      pcsf_export=True,pcsf_genotype_dosage_allele='A2',pcsf_link_cost=.1,
      bayes_kwargs={'max_iter':70,'tol':1e-4})
    export=r['pcsf_inputs']
    assert export is not None
    assert set(export['pcsf_edges'].columns)=={'source','target','cost'}
    assert set(export['variant_nodes'].effect_allele)=={'G'}
    assert set(export['pcsf_edges'].target)=={'GENEA'}
    assert set(export['gene_ranks'].gene)=={'GENEA'}
    assert set(export['variant_nodes'].snp).issubset(set(X.columns))
    with pytest.raises(ValueError,match='A1 or A2'):
        fine_map_region(X,Y,methods=('bayes',),gtf_path=gtf,gtf_variant_metadata=v,
         gtf_contig_aliases={'12':'NC_test.1'},allow_unverified_assembly=True,
         pcsf_export=True)


def test_external_xqtl_links_connect_intergenic_variants_without_changing_betas():
    X,fit,meta,ann=_example()
    links=pd.DataFrame({'snp':['12:101','12:201','12:999'],
       'gene':['OTHER_GENE','OTHER_GENE','OUTSIDE'],
       'evidence_type':['eqtl','sqtl','proximity_not_causal'],
       'cost':[.02,.04,.10]})
    standard=finemap_pcsf_inputs(fit,X,meta,ann)
    out=finemap_pcsf_inputs(fit,X,meta,ann,additional_links=links)
    ed=out['variant_gene_edges']
    assert set(ed.loc[ed.gene=='OTHER_GENE','evidence_type'])=={'eqtl','sqtl'}
    assert not (ed.gene=='OUTSIDE').any() # SNP not selected
    assert out['metadata']['n_independent_annotation_edges']==2
    assert_allclose(out['variant_nodes']['beta_per_A2_dosage__RT'],
                    standard['variant_nodes']['beta_per_A2_dosage__RT'])
    assert out['pcsf_prizes'].equals(standard['pcsf_prizes'])
    assert 'OTHER_GENE' not in set(out['gene_ranks'].gene) # GTF ranks untouched
    with pytest.raises(ValueError,match='requires snp'):
        finemap_pcsf_inputs(fit,X,meta,ann,additional_links=links.drop(columns=['cost']))
