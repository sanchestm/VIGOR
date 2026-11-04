"""Shared, read-only-input workflows for the three real26 notebook entrypoints.

Set VIGOR_REAL26_FIXTURE, VIGOR_PLINK_PREFIX, VIGOR_GDEC and VIGOR_OUTPUT on a
server. Defaults resolve the existing XPS layout; outputs stay below the current
directory. Scientific implementations live in vigor, not in notebook copies.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Windows has no resource module
    resource = None
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


GDEC_SHA256 = 'e006df3134dd445ca17a9fe489298db827ad615d3c8fb53e1ce90c07936f0f83'
GRM_SIGNATURE = 'f313a0f164f3aa175e568fc5e67f82568453a6f20f3269749bd1af8ef8081785'
FROZEN_INPUT_SHA256 = {
    'train.parquet': 'cb0548fa831abb1b12d35b3b21350dfa85af17bc433abd95f8c675213122fee0',
    'truth.parquet': '61f9a8634f548897cd7acb6f16cff183406c2a68d114c09818a90ce1b6549c95',
    'holdout_masks.npz': '78e59d46caf122ec60cb49f917bd9b5a75574cf6688bdbf55650391d3c51c870',
    'additive_kernel.npy': 'af5d5a9c1fea0eb6b0f4651435d1befc52936004e79fd78ca5e36828c0f27a98',
}
PINNED_PANEL = {
    'fam_sha256': 'c33a991319bc5133b1e1abf33fbe5b2dc591e92b8b2e16f2c645784e7431a7f1',
    'bim_sha256': 'eff7b28f8b9a806c8cc965712a14268d932042601b4c78b1daf9d35bf0d41219',
    'consumed_bed_sha256': '363b25e4e65f4f67f0476a4536c81e50ae79661fc93da6f2986f589879dbd8b8',
    'bed_bytes': 47750224503,
    'source_samples': 24239,
    'source_variants': 7879575,
}


def file_sha256(path):
    """Hash all bytes of a file without loading it into memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing regular file opened read-only.

    Returns
    -------
    str
        Lowercase 64-character SHA256 digest.

    Complexity: O(b) TIME and O(c) SPACE for b bytes and an 8 MiB block c.
    """
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def configuration(label):
    """Resolve fixture locations, safe fresh output, CPU limits and optional stages.

    Parameters
    ----------
    label : str
        Notebook-specific relative output directory name; no parent traversal.

    Returns
    -------
    dict
        Paths, seed 42, at most 20 CPU threads, 64 GiB budget and booleans.
        VIGOR_RUN_EXTENDED defaults false; VIGOR_RUN_COMPARATORS defaults true.
        A new output directory is required so existing evidence is preserved.

    Raises
    ------
    ValueError
        Invalid label, thread count, or output overlapping an input/source tree.
    FileExistsError
        The selected run directory already contains evidence.

    Complexity: O(d) TIME and SPACE for path depth d; no scientific data read.
    """
    if Path(label).name != label or label in ('.', '..'):
        raise ValueError('label must be a single directory name')
    data = Path(os.environ.get('VIGOR_DATA_ROOT', Path.home()/'Documents'/'codex')).resolve()
    default = data/'VIGOR'/'validation_20261008_request'/'validation_20261012_sol_high'/'results'/'real26_prediction_comparison_20261012'
    fixture = Path(os.environ.get('VIGOR_REAL26_FIXTURE', default)).resolve()
    prefix = Path(os.environ.get('VIGOR_PLINK_PREFIX', data/'Domino'/'r11.2.1')).resolve()
    output = Path(os.environ.get('VIGOR_OUTPUT', Path.cwd()/'vigor_outputs'))/label
    output = output.resolve()
    source = Path(__file__).resolve().parents[1]
    protected = (fixture, prefix.parent, source)
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('output overlaps a protected input/source tree')
    threads = int(os.environ.get('VIGOR_THREADS', min(2, os.cpu_count() or 1)))
    if not 1 <= threads <= 20:
        raise ValueError('VIGOR_THREADS must be between 1 and 20')
    output.mkdir(parents=True, exist_ok=False)
    return dict(fixture=fixture, prefix=prefix, output=output, seed=42,
                threads=threads, memory_gib=64,
                gdec=Path(os.environ.get('VIGOR_GDEC', source/'comparators'/'genome_decomposition_v3_1_3.py')),
                extended=os.environ.get('VIGOR_RUN_EXTENDED','0') == '1',
                comparators=os.environ.get('VIGOR_RUN_COMPARATORS','1') == '1')


def prepare_training(train):
    """Standardize only observed training cells while retaining labels and NaNs.

    Parameters
    ----------
    train : pandas.DataFrame, shape (n, t)
        Unique animal index and trait columns; NaN means unavailable. Never pass
        outer test outcomes. Each trait needs two observations and positive SD.

    Returns
    -------
    tuple
        Standardized frame (n, t), training mean Series (t,), population-SD
        Series (t,). Original row and column order is preserved.

    Raises
    ------
    ValueError
        Duplicate labels, infinity, insufficient observations or constant trait.

    Complexity: O(nt) TIME and SPACE for n animals and t traits.
    """
    if not train.index.is_unique or not train.columns.is_unique:
        raise ValueError('animal and trait labels must be unique')
    if np.isinf(train.to_numpy(float)).any():
        raise ValueError('infinite phenotype')
    mean = train.mean()
    sd = train.std(ddof=0)
    if (train.notna().sum() < 2).any() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError('each trait needs two observations and positive training SD')
    return (train-mean)/sd, mean, sd


def load_fixture(config):
    """Verify and open the frozen real26 cohort without modifying its cache.

    Parameters
    ----------
    config : dict
        configuration() result; fixture must contain fixture.json, raw truth,
        masked training data, dual masks and ordered additive_kernel.npy.

    Returns
    -------
    dict
        Raw and standardized train/truth, K (n,n) read-only mmap, masks, scales,
        manifest and sample-order fingerprint. n=7616, t=26 in this fixture.

    Raises
    ------
    ValueError
        Stale hashes, wrong shape/order, overlapping masks or leaked test cells.

    Complexity: O(b+n²+nt) TIME and O(n²+nt) peak SPACE, where b is verified
    file bytes. K remains memory-mapped, but symmetry checking allocates n².
    The input manifest pins a transductive r11.2 kernel; no r11.3 cache is mixed.
    """
    base = config['fixture']
    manifest = json.loads((base/'fixture.json').read_text())
    for name in ('train.parquet','truth.parquet','holdout_masks.npz','additive_kernel.npy'):
        digest = file_sha256(base/name)
        if digest != manifest['sha256'][name] or digest != FROZEN_INPUT_SHA256[name]:
            raise ValueError(f'stale fixture: {name}')
    train = pd.read_parquet(base/'train.parquet')
    truth = pd.read_parquet(base/'truth.parquet')
    z, mean, sd = prepare_training(train)
    if train.shape != (7616,26) or not truth.index.equals(train.index) or not truth.columns.equals(train.columns):
        raise ValueError('frozen real26 shape or animal/trait order differs')
    if list(train.columns) != manifest['trait_names']:
        raise ValueError('trait manifest order differs')
    K = np.load(base/'additive_kernel.npy', mmap_mode='r', allow_pickle=False)
    if K.shape != (len(train),len(train)) or not np.isfinite(K).all() or np.max(abs(K-K.T)) > 1e-10:
        raise ValueError('invalid ordered relationship matrix')
    with np.load(base/'holdout_masks.npz', allow_pickle=False) as stored:
        masks = {name:stored[name] for name in ('unseen_animals','missing_traits')}
    if any(m.dtype != bool or m.shape != train.shape for m in masks.values()):
        raise ValueError('invalid holdout mask dimensions or dtype')
    a,b = masks.values()
    if (a & b).any() or not np.isnan(train.to_numpy()[a|b]).all():
        raise ValueError('holdout overlap or leaked test values')
    seen = train.notna().any(axis=1).to_numpy()
    if a[seen].any() or b[~seen].any():
        raise ValueError('whole-animal/within-animal split semantics differ')
    # Reconstruct the deterministic mask from truth, independent of saved masks.
    rng = np.random.default_rng(manifest['seed'])
    rows = np.zeros(len(train),bool)
    rows[rng.choice(len(train),round(.3*len(train)),replace=False)] = True
    held = np.isfinite(truth.to_numpy()) & (rows[:,None] | (rng.random(train.shape)<.1))
    if not np.array_equal(held,a|b) or not np.allclose(train.to_numpy(),truth.mask(held).to_numpy(),equal_nan=True):
        raise ValueError('fixture does not reproduce the frozen seed-42 split')
    order_hash = hashlib.sha256(json.dumps(list(train.index.astype(str))).encode()).hexdigest()
    return dict(train=train, truth=truth, ztrain=z, ztruth=(truth-mean)/sd,
                mean=mean, sd=sd, K=K, masks=masks, manifest=manifest,
                sample_order_sha256=order_hash)


def prediction_metrics(truth, train, prediction, masks):
    """Score labeled heldout phenotypes against the training-mean reference.

    Parameters
    ----------
    truth, train, prediction : pandas.DataFrame, shape (n,t)
        Identically ordered animals/traits. truth is used only for scoring;
        training NaNs mark withheld/unavailable cells. Predictions must be finite.
    masks : mapping of str to bool ndarray, shape (n,t)
        Named pre-frozen, observed truth cells withheld from training.

    Returns
    -------
    pandas.DataFrame
        Per split and trait n, MSE, predictive R2 versus training-mean MSE, and
        squared correlation (distinct from predictive R2). R2 is NaN when the
        denominator is zero; correlation is NaN for constant/singleton vectors.

    Raises
    ------
    ValueError
        Labels/order differ, a mask leaks training observations or prediction
        is nonfinite on a scored entry.

    Complexity: O(nt s) TIME and O(nt+st) SPACE for s split masks.
    """
    for frame in (train,prediction):
        if not truth.index.equals(frame.index) or not truth.columns.equals(frame.columns):
            raise ValueError('animal/trait order must match exactly')
    rows = []
    y,p = truth.to_numpy(float),prediction.to_numpy(float)
    for name,mask in masks.items():
        if mask.shape != y.shape or mask.dtype != bool or not np.isnan(train.to_numpy()[mask]).all():
            raise ValueError('holdout mask dimensions or training exclusion invalid')
        for j,trait in enumerate(truth.columns):
            m = mask[:,j] & np.isfinite(y[:,j]); yy,pp = y[m,j],p[m,j]
            if not np.isfinite(pp).all():
                raise ValueError('nonfinite heldout predictions')
            mse = float(np.mean((yy-pp)**2)) if len(yy) else np.nan
            reference = float(np.mean((yy-train.iloc[:,j].mean())**2)) if len(yy) else np.nan
            corr2 = float(np.corrcoef(yy,pp)[0,1]**2) if len(yy)>1 and np.std(yy)>0 and np.std(pp)>0 else np.nan
            rows.append(dict(split=name,trait=trait,n=int(m.sum()),mse=mse,
                             predictive_r2=1-mse/reference if reference>0 else np.nan,corr2=corr2))
    return pd.DataFrame(rows)


def write_json(path, value):
    """Write JSON atomically; values must be JSON serializable (no raw logs).

    Parameters
    ----------
    path : pathlib.Path
        Destination in the private run directory.
    value : dict or list
        Summary metadata; NumPy scalars are converted to Python scalars.

    Returns
    -------
    None
        Replaces only the selected report, leaving scientific checkpoints intact.

    Complexity: O(b) TIME and SPACE for b serialized bytes.
    """
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False)+'\n')
    pending.replace(path)


def run_prediction(data, config):
    """Run current production automatic prediction with masked real26 inputs.

    Parameters
    ----------
    data : dict
        Verified load_fixture() result. Only ztrain and K enter fitting.
    config : dict
        Threads/output from configuration(); seed 42 for reproducibility.

    Returns
    -------
    dict
        Actual mvPredict result including covariance_fit and sparse-CV failures.
        Saves per-trait metrics and convergence/PSD/resource summaries privately.

    Complexity: delegates to mvPredict including its default sparse-CV fitting;
    dense K storage is O(n²), predictions O(nt); iterative cost depends on
    convergence. See vigor's inspected docstrings for fit/operator bounds.
    """
    import vigor
    start = time.perf_counter()
    with threadpool_limits(config['threads']):
        result = vigor.mvPredict(data['ztrain'],data['K'],G=None,
                                 accuracy='exact',solver_maxiter=3000,
                                 blas_threads=config['threads'])
    fit = result.attrs['covariance_fit']
    if not bool(fit.attrs.get('optimizer_converged',fit.attrs.get('success',False))) or int(result.attrs.get('cg_info',0)):
        raise RuntimeError('covariance/prediction did not converge')
    eigs = {name:float(np.linalg.eigvalsh(np.asarray(fit[name])).min()) for name in ('G_prediction','E_prediction')}
    if min(eigs.values()) < -1e-8:
        raise ValueError('non-PSD fitted prediction covariance')
    unseen = ~data['ztrain'].notna().any(axis=1)
    if not np.array_equal(result['genetic_mean'].to_pandas().loc[unseen],result['phenotype_prediction'].to_pandas().loc[unseen]):
        raise ValueError('conditional residual is nonzero for entirely unseen animals')
    tables = []
    for target in ('genetic_mean','phenotype_prediction'):
        frame = result[target].to_pandas()
        if not np.isfinite(np.asarray(frame)).all():
            raise ValueError('nonfinite prediction')
        table = prediction_metrics(data['ztruth'],data['ztrain'],frame,data['masks'])
        table['target'] = target; table['model'] = 'VIGOR'; tables.append(table)
    metrics = pd.concat(tables,ignore_index=True)
    metrics.to_csv(config['output']/'vigor_metrics.csv',index=False)
    write_json(config['output']/'prediction.json',dict(state='COMPLETE',
        shape=list(data['ztrain'].shape),version=vigor.__version__,
        seconds=time.perf_counter()-start,peak_rss_mib=None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cg_info=int(result.attrs.get('cg_info',0)),optimizer_converged=True,min_eigenvalues=eigs,
        sample_order_sha256=data['sample_order_sha256'],
        selection=result.attrs['prediction_sparse_cv'],
        macro_predictive_r2=metrics.groupby(['target','split']).predictive_r2.mean().to_dict().__str__(),
        caveat='Frozen once-inspected phenotypes; transductive GRM; upstream regression/family independence unvalidated.'))
    return result


def run_comparators(data, config, prediction):
    """Run genuine GDEC 3.1.3 and native R rrBLUP on identical training inputs.

    Parameters
    ----------
    data, config : dict
        Verified fixture and output/thread configuration. VIGOR_GDEC can point
        to the supplied comparator; identity/version are checked before import.
    prediction : dict
        Current VIGOR result; its covariance is used only for solver parity.

    Returns
    -------
    pandas.DataFrame
        Per-trait heldout metrics. Missing optional comparator is explicitly
        SKIPPED; available comparator failure is recorded and raised. GDEC
        shared-G/E parity is distinct from independently fitted comparison.

    Complexity: GDEC uses its blocked moment fit plus iterative prediction;
    rrBLUP per trait uses O(o³+n o²) TIME and O(n²+o²+nt) SPACE for at most o
    observed training animals. Full binary K copy is O(n²) disk space.
    """
    import vigor
    records, tables = [], []
    if not config['comparators']:
        write_json(config['output']/'comparators.json',[dict(state='SKIPPED',reason='disabled')])
        return pd.DataFrame()
    active_model = 'GDEC 3.1.3'
    try:
        gpath = config['gdec']
        if not gpath.is_file():
            records.append(dict(model='GDEC 3.1.3',state='SKIPPED',reason='source not supplied'))
        else:
            digest = file_sha256(gpath)
            if digest != GDEC_SHA256:
                raise ValueError('GDEC source differs from supplied genuine 3.1.3 comparator')
            spec = importlib.util.spec_from_file_location('gdec313_comparator',gpath)
            gdec = importlib.util.module_from_spec(spec); spec.loader.exec_module(gdec)
            if gdec.V3_1_FINAL_VERSION != '3.1.3':
                raise ValueError('unexpected GDEC version')
            start = time.perf_counter()
            with threadpool_limits(config['threads']):
                fit = gdec.genetic_varcov_PSD(data['K'],data['ztrain'],rank_G=None,rank_E=None,
                    n_boot=0,random_state=config['seed'],blas_threads=config['threads'],maxiter=1500,
                    gtol=1e-7,ftol=1e-10,mode='blocked',block_size=512,
                    shrink_G='auto',shrink_E='auto',E_structure='auto')
                if not bool(fit.attrs.get('optimizer_converged',fit.attrs.get('success',False))):
                    raise RuntimeError('GDEC fit did not converge')
                outputs = gdec.mvPredict(data['ztrain'],data['K'],fit['G'].to_pandas(),E=fit['E'].to_pandas(),
                    solver_tol=1e-7,solver_maxiter=3000,work_dtype='float64',blas_threads=config['threads'])
                shared = prediction.attrs['covariance_fit']
                parity = gdec.mvPredict(data['ztrain'],data['K'],shared['G_prediction'],
                    E=shared['E_prediction'],solver_tol=1e-7,solver_maxiter=3000,
                    work_dtype='float64',blas_threads=config['threads'])
            # Native GDEC names the intercept separately; expose the same target.
            for native in (outputs,parity):
                native['genetic_mean'] = native['breeding_value'] + native['trait_mean']
            delta = float(np.max(abs(np.asarray(parity['genetic_mean'].to_pandas())-np.asarray(prediction['genetic_mean']))))
            np.testing.assert_allclose(parity['genetic_mean'].to_pandas(),prediction['genetic_mean'],rtol=2e-5,atol=2e-5)
            for target in ('genetic_mean','phenotype_prediction'):
                frame = pd.DataFrame(np.asarray(outputs[target]),index=data['ztrain'].index,columns=data['ztrain'].columns)
                table = prediction_metrics(data['ztruth'],data['ztrain'],frame,data['masks'])
                table['model']='GDEC 3.1.3';table['target']=target;tables.append(table)
            records.append(dict(model='GDEC 3.1.3',state='COMPLETE',source_sha256=digest,
                                seconds=time.perf_counter()-start,shared_covariance_max_abs=delta))
        active_model = 'rrBLUP'
        rscript = Path(os.environ.get('VIGOR_RSCRIPT',Path(sys.executable).parent/'Rscript'))
        if not rscript.is_file():
            located = shutil.which('Rscript');rscript = Path(located) if located else rscript
        if not rscript.is_file():
            records.append(dict(model='rrBLUP',state='SKIPPED',reason='Rscript unavailable'))
        else:
            probe = subprocess.run([str(rscript),'--vanilla','-e',
                'if(!requireNamespace("rrBLUP",quietly=TRUE))quit(status=3);cat(as.character(packageVersion("rrBLUP")))'],
                capture_output=True,text=True,timeout=60)
            if probe.returncode:
                records.append(dict(model='rrBLUP',state='SKIPPED',reason='rrBLUP unavailable in selected R library'))
            else:
                start = time.perf_counter(); out = config['output']
                data['ztrain'].rename_axis('IID').to_csv(out/'rr_train.csv',na_rep='NA')
                np.asarray(data['K'],dtype='<f8').tofile(out/'rr_K.bin')
                command = [str(rscript),'--vanilla',str(Path(__file__).with_name('rrblup_real26.R')),
                           str(out/'rr_train.csv'),str(out/'rr_K.bin'),str(out/'rr_prediction.csv')]
                with (out/'rr_execution.log').open('w') as log:
                    proc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                if proc.returncode:
                    raise RuntimeError(f'rrBLUP exited {proc.returncode}; private log retained')
                frame = pd.read_csv(out/'rr_prediction.csv',index_col=0,dtype={'IID':str})
                if not frame.index.equals(data['ztrain'].index) or not frame.columns.equals(data['ztrain'].columns):
                    raise ValueError('rrBLUP output order differs')
                table=prediction_metrics(data['ztruth'],data['ztrain'],frame,data['masks'])
                table['model']='rrBLUP '+probe.stdout.strip();table['target']='genetic_mean';tables.append(table)
                records.append(dict(model='rrBLUP',state='COMPLETE',version=probe.stdout.strip(),seconds=time.perf_counter()-start))
    except Exception as exc:
        records.append(dict(model=active_model,state='FAILED',error_type=type(exc).__name__,error=str(exc)[:500]))
        write_json(config['output']/'comparators.json',records)
        raise
    write_json(config['output']/'comparators.json',records)
    all_metrics = pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    all_metrics.to_csv(config['output']/'comparator_metrics.csv',index=False)
    return all_metrics


def bounded_real_panel(data, config, n_samples=512, variants_per_chromosome=64):
    """Read an assembly-pinned real genotype panel in verified phenotype order.

    Parameters
    ----------
    data, config : dict
        Verified frozen real26 data/configuration; the default r11.2.1 source
        must match its GRM lineage. r11.3 inputs require a separate rebuilt fixture.
    n_samples : int
        Default 512; seeded permutation of all cohort rows, independent of
        phenotype values and availability. No trait columns are removed.
    variants_per_chromosome : int
        Only 64 is supported for the byte-pinned panel; 64 markers on each of
        chromosomes 1 and 2 are selected without
        phenotype screening. This is an I/O/scientific function example, not GWAS.

    Returns
    -------
    tuple
        Dosage DataFrame (n,p) with NaN hardcall missingness, masked standardized
        real26 frame (n,26), chromosome string array (p,), and source provenance.

    Raises
    ------
    ValueError
        Genotype release mismatch, duplicate/missing animal IDs, incomplete BED,
        or source panel bounds incompatible with the requested subset.

    Complexity: O(B+np) TIME and O(N+np+c) SPACE for B BIM bytes scanned, N FAM
    animals, n retained animals, p selected variants, c BIM chunk size. BED is
    read-only mmap; decoding touches selected variant byte rows only.
    """
    from vigor.npplink import read_bed_chunk_subset
    prefix = config['prefix']
    if prefix.name != 'r11.2.1':
        raise ValueError('frozen GRM is r11.2.1; r11.3 requires its own fixture and GRM')
    if variants_per_chromosome != 64:
        raise ValueError('the provenance-pinned panel requires 64 variants per chromosome')
    observed_metadata = {suffix: file_sha256(Path(str(prefix)+ext))
                         for suffix, ext in [('fam_sha256','.fam'),('bim_sha256','.bim')]}
    if any(observed_metadata[key] != PINNED_PANEL[key] for key in observed_metadata):
        raise ValueError('genotype provenance differs from the pinned r11.2.1 source metadata')
    fam = pd.read_csv(str(prefix)+'.fam',sep=r'\s+',header=None,usecols=[1],dtype=str)[1]
    if len(fam) != PINNED_PANEL['source_samples'] or not fam.is_unique:
        raise ValueError('duplicate genotype animal IDs')
    if not 1 <= n_samples <= len(data['ztrain']):
        raise ValueError('n_samples must fit the complete cohort')
    selected = np.sort(np.random.default_rng(config['seed']).permutation(len(data['ztrain']))[:n_samples])
    Y = data['ztrain'].iloc[selected]
    indices = pd.Index(fam).get_indexer(Y.index.astype(str))
    if (indices<0).any():
        raise ValueError('genotype IDs do not cover selected real26 animals')
    panels,counts = [],{'1':0,'2':0}
    offset = 0
    for block in pd.read_csv(str(prefix)+'.bim',sep=r'\s+',header=None,
                             usecols=[0,1,3],dtype={0:str,1:str,3:np.int64},chunksize=131072):
        block.index = np.arange(offset,offset+len(block));offset+=len(block)
        for chrom in counts:
            part=block[block[0].eq(chrom)].head(variants_per_chromosome-counts[chrom])
            if len(part):panels.append(part);counts[chrom]+=len(part)
        if min(counts.values())==variants_per_chromosome:break
    if min(counts.values())!=variants_per_chromosome:
        raise ValueError('insufficient variants on required chromosomes')
    bim=pd.concat(panels).sort_index();bedpath=Path(str(prefix)+'.bed')
    with bedpath.open('rb') as f:
        if f.read(3)!=b'\x6c\x1b\x01':raise ValueError('BED is not SNP-major PLINK1')
    stride=(len(fam)+3)//4
    if bedpath.stat().st_size != PINNED_PANEL['bed_bytes'] or bedpath.stat().st_size != 3+stride*PINNED_PANEL['source_variants']:
        raise ValueError('BED payload geometry differs from pinned source metadata')
    bed=np.memmap(bedpath,mode='r',dtype=np.uint8)
    X=np.column_stack([read_bed_chunk_subset(bed,len(fam),int(i),1,indices)[:,0] for i in bim.index])
    X=pd.DataFrame(X,index=Y.index,columns=bim[1].to_numpy())
    # Hash exact consumed bytes plus full authoritative FAM/BIM metadata.
    h=hashlib.sha256()
    for i in bim.index:h.update(bed[3+int(i)*stride:3+(int(i)+1)*stride].tobytes())
    if h.hexdigest() != PINNED_PANEL['consumed_bed_sha256']:
        raise ValueError('genotype provenance differs in the consumed BED byte rows')
    meta=dict(genotype_release='r11.2.1',historical_GRM_signature=GRM_SIGNATURE,
              sample_order_sha256=hashlib.sha256(json.dumps(list(Y.index.astype(str))).encode()).hexdigest(),
              **observed_metadata,
              consumed_bed_sha256=h.hexdigest(),bed_bytes=bedpath.stat().st_size,
              variant_indices=bim.index.tolist(),n_samples=len(Y),n_variants=X.shape[1],
              provenance_checked='Exact frozen fixture plus expected FAM/BIM and consumed BED bytes; full source BED geometry checked',
              limitation='Two small windows; unconsumed BED bytes and independent biological assembly correctness not authenticated by this panel check.')
    return X,Y,bim[0].to_numpy(str),meta


def run_extended(data, config, prediction):
    """Execute bounded real26 factorGWAS, fine mapping and covariance exploration.

    Parameters
    ----------
    data, config : dict
        Verified full real26 fixture and configuration. extended=False records
        SKIPPED without counting the branch as scientific execution.
    prediction : dict
        Completed full-cohort VIGOR prediction containing its covariance fit.

    Returns
    -------
    dict
        Per-stage execution summaries/provenance; outputs saved privately.
        A 512-animal, 128-marker real panel retains all 26 traits. Fine mapping
        uses 24 variants and one trait's observed training rows, without imputation
        of heldout outcomes. Covariance mediation/PC are exploratory diagnostics.

    Complexity: factorGWAS delegates covariance fits and marker operators; fine
    mapping delegates a bounded locus. Neither small-panel timings nor results
    establish whole-genome calibration; see the numerical entrypoint contracts.
    """
    import vigor
    if not config['extended']:
        result=dict(state='SKIPPED',reason='VIGOR_RUN_EXTENDED=0',stages=['factorGWAS','fine_map_region','covariance_mediation','pc'])
        write_json(config['output']/'extended.json',result);return result
    start=time.perf_counter(); X,Y,groups,provenance=bounded_real_panel(data,config)
    write_json(config['output']/'panel_provenance.json',provenance)
    # The bounded demonstration uses direct observed-trait score tests. The
    # posterior-score route failed its null residual gate on this sparse panel;
    # that execution is retained in the review evidence, not silently relaxed.
    stages = {}
    def stage(name, operation):
        """Record a bounded stage outcome while preserving other stage evidence."""
        started = time.perf_counter()
        try:
            details = operation()
            stages[name] = dict(state='COMPLETE', seconds=time.perf_counter()-started, **details)
        except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as exc:
            stages[name] = dict(state='FAILED', seconds=time.perf_counter()-started,
                                error_type=type(exc).__name__, error=str(exc)[:500])
        write_json(config['output']/'extended_stages.json',stages)

    def scan_stage():
        scan=vigor.factorGWAS(Y,genotypes=X,groups=groups,background='loco',
            factors=2,method='exact',score_method='direct',loco_factor='refit',factor_method='native',
            heterogeneity=False,
            covariance_rank_G=26,covariance_rank_E=26,
            covariance_kwargs=dict(maxiter=1500,analytic_se=False,
                                   blas_threads=config['threads'],progress=False),
            random_state=config['seed'],
            null_solver_kwargs=dict(work_dtype='float64',preconditioner='kronecker',solver_maxiter=8000))
        if scan.sizes['variant']!=X.shape[1] or not np.isfinite(scan['stat'].values).all():
            raise ValueError('invalid bounded factorGWAS result')
        scan.to_dataframe().reset_index().to_csv(config['output']/'bounded_factor_gwas.csv',index=False)
        return dict(tests=scan.sizes['variant']*scan.sizes['trait'],score_method='direct observed-trait exact')

    def fine_stage():
        trait=Y.notna().sum().idxmax()
        fine=vigor.fine_map_region(X.iloc[:,:24],Y[[trait]],methods=('bayes','ridge'),
            L=2,hdbscan=False,compute_linear_shap=False,random_state=config['seed'])
        fine.ranking.to_csv(config['output']/'bounded_finemap_ranking.csv')
        # Posterior inclusion must be finite; optional evidence columns can be unavailable.
        pip_columns=[c for c in fine.ranking if 'pip' in str(c).lower()]
        if not pip_columns or not np.isfinite(fine.ranking[pip_columns].to_numpy(float)).all():
            raise ValueError('missing/nonfinite Bayesian inclusion probabilities')
        return dict(complete_samples=fine.metadata['n_complete_samples'],markers=24,
                    selected_traits=[str(trait)],pip_columns=list(map(str,pip_columns)))

    fit=prediction.attrs['covariance_fit'];G=np.asarray(fit['G_prediction'].to_pandas());E=np.asarray(fit['E_prediction'].to_pandas())
    def mediation_stage():
        mediated=vigor.genetic_covariance_mediation(G,[0],outcomes=[1,2],E=E,names=list(data['ztrain'].columns))
        summary_cols=['source','outcome','marginal_variance','conditional_variance','variance_fraction_accounted']
        mediated[summary_cols].to_dataframe().to_csv(config['output']/'exploratory_mediation.csv',index=False)
        return dict(sources=int(mediated.sizes['component']),outcomes=2)

    def pc_stage():
        pc=vigor.pc_from_covariance(G,n_eff=int(data['ztrain'].notna().sum().min()),
                                   names=list(data['ztrain'].columns),alpha=.01,max_cond_set=1)
        pc[['source','target','orientation']].to_dataframe().to_csv(config['output']/'exploratory_pc_edges.csv',index=False)
        return dict(graph_arcs=int(pc.attrs['graph'].number_of_edges()),
                    caveat='Approximate effective N for fitted covariance; exploratory only')

    for name, operation in [('factorGWAS',scan_stage),('fine_map_region',fine_stage),
                            ('covariance_mediation',mediation_stage),('PC',pc_stage)]:
        stage(name,operation)
    result=dict(state='COMPLETE' if all(x['state']=='COMPLETE' for x in stages.values()) else 'PARTIAL',
        shape=list(Y.shape),markers=X.shape[1],seconds=time.perf_counter()-start,stages=stages,
        limitation='Bounded two-window LOCO demonstration; no genome-wide result, causal identification or breeding-value truth. Failed stages remain failed.')
    write_json(config['output']/'extended.json',result);return result
