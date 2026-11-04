"""Portable process-isolated covariance/prediction benchmark for VIGOR 2.12.1.

Synthetic G/E truth and real-data distance to a comparator are different metrics.
Each method runs in a fresh process; timing is a single-run diagnostic.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'comparators'))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT.parent/'benchmark_results')
    p.add_argument('--samples',type=int,default=160);p.add_argument('--traits',type=int,default=6)
    p.add_argument('--rank',type=int,default=2);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=214)
    p.add_argument('--phenotypes',type=Path,help='real trait CSV, sample IDs in first column')
    p.add_argument('--truth',type=Path,help='optional real withheld phenotype truth CSV')
    p.add_argument('--grm-store',type=Path);p.add_argument('--grm-group',default='grm/genomewide')
    p.add_argument('--grm-variable',default='additive')
    p.add_argument('--methods',default='gdec313,pca,sparse_pca,fa,sparse_fa,spectral_fa')
    p.add_argument('--raw-covariances',action='store_true',help='use unregularized inference G/E instead of package prediction defaults')
    p.add_argument('--worker',help=argparse.SUPPRESS)
    return p.parse_args()


def load_data(args):
    import numpy as np,pandas as pd,xarray as xr
    rng=np.random.default_rng(args.seed)
    if args.phenotypes:
        if not args.grm_store:raise ValueError('--phenotypes requires --grm-store')
        frame=pd.read_csv(args.phenotypes,index_col=0,usecols=list(range(args.traits+1)))
        if len(frame)>args.samples:
            frame=frame.iloc[np.sort(rng.choice(len(frame),args.samples,replace=False))]
        frame.index=frame.index.astype(str);frame=frame.iloc[:,:args.traits]
        truth=pd.read_csv(args.truth,index_col=0) if args.truth else frame.copy()
        truth.index=truth.index.astype(str);truth=truth.reindex(index=frame.index,columns=frame.columns)
        with xr.open_zarr(args.grm_store,group=args.grm_group,consolidated=None) as ds:
            array=ds[args.grm_variable]
            sample_dims=[d for d in array.dims if d.startswith('sample')]
            if len(sample_dims)!=2:raise ValueError('GRM needs two labeled sample dimensions')
            if all(d in array.coords for d in sample_dims):
                K=np.asarray(array.sel({d:frame.index.tolist() for d in sample_dims}),float)
            else:
                # Historical genomewide groups keep coordinates in the chromosome group.
                with xr.open_zarr(args.grm_store,group='grm/chromosome',consolidated=None) as meta:
                    positions=pd.Index(np.asarray(meta.sample_i,dtype=object).astype(str)).get_indexer(frame.index)
                if np.any(positions<0):raise KeyError('phenotype sample absent from GRM')
                K=np.asarray(array.isel({d:positions for d in sample_dims}),float)
        G0=E0=None;name='real'
    else:
        n,t=args.samples,args.traits
        X=rng.normal(size=(n,max(40,n//2)));K=X@X.T/X.shape[1]
        L=rng.normal(0,.4,(t,args.rank));M=rng.normal(0,.25,(t,args.rank))
        G0=L@L.T+.2*np.eye(t);E0=M@M.T+.6*np.eye(t)
        v,u=np.linalg.eigh(K);half=u*np.sqrt(np.maximum(v,0))
        Y=half@rng.normal(size=(n,t))@np.linalg.cholesky(G0).T+rng.normal(size=(n,t))@np.linalg.cholesky(E0).T
        frame=pd.DataFrame(Y,index=[f'i{i}' for i in range(n)],columns=[f't{i}' for i in range(t)])
        truth=frame.copy();name='synthetic_gaussian'
    valid=frame.notna().sum()>=3
    frame=frame.loc[:,valid];truth=truth.loc[:,valid]
    if frame.shape[1]<args.rank:raise ValueError('not enough observed traits for requested rank')
    train=frame.mask(rng.random(frame.shape)<.25);train.iloc[:max(1,len(train)//10)]=np.nan
    for column in train:
        if train[column].notna().sum()<2:
            rows=frame.index[frame[column].notna()][:2];train.loc[rows,column]=frame.loc[rows,column]
    return name,train,truth,K,G0,E0


def metrics(P,train,truth):
    import numpy as np
    P=np.asarray(P,float);Y=truth.to_numpy(float);mask=train.isna().to_numpy()&np.isfinite(Y)&np.isfinite(P)
    scores=[];errs=[]
    for j in range(Y.shape[1]):
        valid=mask[:,j]
        if valid.sum()<5:continue
        y,p=Y[valid,j],P[valid,j];den=((y-y.mean())**2).sum()
        if den>0:scores.append(1-((y-p)**2).sum()/den)
        errs.append(np.sqrt(((y-p)**2).mean()))
    return dict(r2=float(np.mean(scores)) if scores else None,rmse=float(np.mean(errs)) if errs else None,n_traits_evaluated=len(scores),n_holdout_values=int(mask.sum()))


def worker(args):
    import numpy as np,vigor
    from vigor.resources import PeakRSSMonitor
    from threadpoolctl import threadpool_limits
    name,train,truth,K,G0,E0=load_data(args)
    row=dict(package_version=vigor.__version__,dataset=name,method=args.worker,n=len(train),traits=train.shape[1],rank=args.rank,seed=args.seed,prediction_geometry='raw_inference' if args.raw_covariances else 'package_prediction_defaults')
    mon=PeakRSSMonitor(.01).start();start=time.perf_counter()
    try:
        kw=dict(rank_G=args.rank,rank_E=args.rank,n_boot=0,maxiter=500,blas_threads=args.threads,progress=False,random_state=args.seed)
        with threadpool_limits(args.threads):
            if args.worker=='gdec313':
                import genome_decomposition_v3_1_3 as gdec
                fit=gdec.genetic_varcov_PSD(K,train,**{k:v for k,v in kw.items() if k!='progress'})
            else:
                if args.raw_covariances:kw.update(shrink_G=1.,shrink_E=1.,variance_closure='off')
                if args.worker.startswith('spectral'):kw['solver']=args.worker
                else:kw.update(solver='lbfgs',factor_structure_G=args.worker,factor_structure_E=args.worker)
                fit=vigor.genetic_varcov_PSD(K,train,**kw)
        row['fit_s']=time.perf_counter()-start;row['covariance_success']=bool(getattr(fit,'attrs',fit).get('success',True))
        row.update({f'{c}_rmse_truth':None if target is None else float(np.sqrt(np.mean((np.asarray(fit[c])-target)**2))) for c,target in [('G',G0),('E',E0)]})
        from vigor.store import VigorStore
        VigorStore(args.output/f'{args.worker}_fit.zarr',mode='w').write_covariance_result(fit,trait_names=train.columns)
        for label,function in [('mvblup',vigor.mvBLUP),('mvpredict',vigor.mvPredict)]:
            timer=time.perf_counter()
            try:
                result=function(train,K,fit['G'],fit['E'],blas_threads=args.threads) if args.raw_covariances or args.worker=='gdec313' else function(train,K,fit,blas_threads=args.threads)
                prediction=result['total'] if label=='mvblup' else result.get('phenotype_prediction')
                if prediction is None:raise KeyError('prediction result missing phenotype_prediction')
                row.update({f'{label}_{k}':v for k,v in metrics(prediction,train,truth).items()})
                row[f'{label}_success']=True
            except Exception as exc:row[f'{label}_success']=False;row[f'{label}_error']=str(exc)
            row[f'{label}_s']=time.perf_counter()-timer
        row['status']='complete'
    except Exception as exc:row.update(status='failed',error=str(exc))
    finally:row.update(mon.stop())
    (args.output/f'{args.worker}.json').write_text(json.dumps(row,indent=2,default=str))
    print(json.dumps(row,default=str))


def main():
    args=arguments();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    import pandas as pd
    records=[]
    for method in args.methods.split(','):
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker',method]
        result=subprocess.run(command,capture_output=True,text=True)
        (args.output/f'{method}.log').write_text(result.stdout+result.stderr)
        path=args.output/f'{method}.json'
        records.append(json.loads(path.read_text()) if result.returncode==0 and path.exists() else dict(method=method,status='failed',error=result.stderr[-1500:]))
        pd.DataFrame(records).to_csv(args.output/'benchmark.csv',index=False)
        print(method,records[-1]['status'],flush=True)
    # Distance to a real-data reference is explicitly not covariance truth.
    if 'gdec313' in args.methods.split(',') and (args.output/'gdec313_fit.zarr').exists():
        import xarray as xr,numpy as np
        with xr.open_zarr(args.output/'gdec313_fit.zarr',group='covariance',consolidated=True) as ref:
            for row in records:
                p=args.output/f"{row['method']}_fit.zarr"
                if not p.exists():continue
                with xr.open_zarr(p,group='covariance',consolidated=True) as ds:
                    for c in ['G','E']:row[f'{c}_rmse_gdec_reference']=float(np.sqrt(np.mean((ds[c].values-ref[c].values)**2)))
        pd.DataFrame(records).to_csv(args.output/'benchmark.csv',index=False)
    if any(row.get('status')=='failed' for row in records):raise SystemExit(1)


if __name__=='__main__':main()
