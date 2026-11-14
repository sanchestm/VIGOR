"""Regression tests for native PCST, corrected greedy fallback and causal G/E output."""
from pathlib import Path
import json
import numpy as np,pandas as pd,networkx as nx,pytest
from numpy.testing import assert_allclose
from vigor.network.pcsf import prize_collecting_steiner_forest
from vigor.causal import (genetic_covariance_mediation,fit_multisource_covariance_sem,
                          pc_from_covariance,pc_from_data)
from vigor.results import open_result

def test_greedy_does_not_return_unconnected_positive_prizes():
    edges=pd.DataFrame({'source':['a'],'target':['b'],'cost':[10.]})
    prizes=pd.Series({'a':1.,'b':1.})
    fit=prize_collecting_steiner_forest(edges,prizes,
           method='greedy',prize_transform='none')
    assert fit.objective==pytest.approx(1.)
    assert len(fit.nodes)==1 and len(fit.edges)==0
    assert any('approximation' in w for w in fit.warnings)
    with pytest.raises(ValueError,match='multi-cluster'):
        prize_collecting_steiner_forest(edges,prizes,method='greedy',
            prize_transform='none',num_clusters=2)

def test_greedy_connects_profitable_steiner_gene_but_never_negative():
    edges=pd.DataFrame({'source':['x','G1','G2'],'target':['G1','G2','y'],
                        'cost':[.10,.1,.10]})
    prizes=pd.Series({'x':.7,'y':.8,'G1':0.,'G2':0.})
    fit=prize_collecting_steiner_forest(edges,prizes,method='greedy',
                                           prize_transform='none')
    assert fit.objective==pytest.approx(1.2)
    assert set(fit.nodes.node)=={'x','G1','G2','y'}
    graph=nx.from_pandas_edgelist(fit.edges,'source','target')
    assert nx.is_tree(graph)
    assert fit.objective==pytest.approx(fit.nodes.prize.sum()-fit.edges.cost.sum())

@pytest.mark.parametrize('method',['greedy','pcst_fast'])
def test_native_or_fallback_matches_simple_exact_optimum(method):
    if method=='pcst_fast':pytest.importorskip('pcst_fast')
    edges=pd.DataFrame({'source':['a','b'],'target':['b','c'],'cost':[1.,1.]})
    prize=pd.Series({'a':4.,'b':0.,'c':3.})
    out=prize_collecting_steiner_forest(edges,prize,method=method,
                                    prize_transform='none')
    assert out.objective==pytest.approx(5.)

def test_result_zarr_pc_from_data_preserves_conditioning_sets(tmp_path):
    rng=np.random.default_rng(102)
    a=rng.normal(size=120)
    b=.8*a+rng.normal(size=120)
    c=.5*b+rng.normal(size=120)
    Y=pd.DataFrame({'a':a,'b':b,'c':c})
    raw=pc_from_data(Y,max_cond_set=1,n_eff=120)
    target=tmp_path/'pc_from_data.zarr'
    saved=pc_from_data(Y,max_cond_set=1,n_eff=120,save_path=target)
    reread=open_result(target)
    assert set(raw.data_vars)==set(saved.data_vars)==set(reread.data_vars)
    conditioned=list(reread['conditioned'].values)
    assert conditioned and all(isinstance(x,str) for x in conditioned)
    parsed=[json.loads(x) for x in conditioned]
    assert all(isinstance(x,list) for x in parsed)
    assert_allclose(raw['partial_r'],reread['partial_r'],equal_nan=True)

def test_causal_sem_and_mediate_known_two_source_G_E():
    B=np.zeros((3,3));B[1,0]=.52;B[2,1]=-.40
    A=np.linalg.inv(np.eye(3)-B)
    G=A@np.diag([.50,.30,.15])@A.T
    E=A@np.diag([.80,.50,.65])@A.T
    names=['M','Y','Z']
    med=genetic_covariance_mediation(G,['M'],outcomes=['Y'],E=E,names=names)
    sources=list(np.asarray(med['source']).astype(str))
    f=list(np.asarray(med['variance_fraction_accounted'],float))
    for source,frac in zip(sources,f):
        C=G if source=='additive' else E
        expected=C[0,1]**2/(C[0,0]*C[1,1])
        assert frac==pytest.approx(expected,abs=1e-7)
    sem=fit_multisource_covariance_sem({'G':G,'E':E},names=names,
             order=names,edge_mask=np.abs(B)>0,maxiter=700,tol=1e-9,
             evaluation_covariances={'G':G,'E':E})
    assert sem.attrs['success']
    assert_allclose(sem['B'],B,atol=1e-4)

def test_pc_covariance_skeleton_not_direction_identification():
    B=np.zeros((3,3));B[1,0]=.5;B[2,1]=-.6
    A=np.linalg.inv(np.eye(3)-B)
    G=A@np.eye(3)@A.T
    pc=pc_from_covariance(G,names=['a','b','c'],alpha=.01,
            n_eff=100000,max_cond_set=1,multiple_testing='none')
    adj=np.asarray(pc['adjacency'],bool)
    assert adj[0,1] and adj[1,2] and not adj[0,2]
