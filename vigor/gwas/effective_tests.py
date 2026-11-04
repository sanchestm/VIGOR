"""Effective number of GWAS tests and family-wise threshold calibration.

LD estimators are phenotype independent. ``null_maxT`` is the empirical
reference. Raw GRM effective ranks are diagnostics and only produce a GWAS
threshold when an explicit, validated calibration is supplied.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib, json, time, tempfile, os, weakref, functools
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits

from .ld import _debiased_r2, _iter_off_diagonal_r2, local_correlation_matrix, _pairwise_r2
from .null_maxT import adaptive_maxT


@dataclass
class EffectiveTestsResult:
    """EffectiveTestsResult represents effective tests result.

    Attributes
    ----------
    method : str
        Stored method; its axis/order follows the construction inputs.
    n_samples : int
        Stored n samples; its axis/order follows the construction inputs.
    n_variants : int
        Stored n variants; its axis/order follows the construction inputs.
    n_variants_evaluated : int | None
        Stored n variants evaluated; its axis/order follows the construction inputs.
    meff : float | None
        Stored meff; its axis/order follows the construction inputs.
    tau_ld : float | None
        Stored tau ld; its axis/order follows the construction inputs.
    p_threshold : dict
        Stored p threshold; its axis/order follows the construction inputs.
    neglog10_threshold : dict
        Stored neglog10 threshold; its axis/order follows the construction inputs.
    chromosome_results : pd.DataFrame | None
        Stored chromosome results; its axis/order follows the construction inputs.
    diagnostics : dict
        Stored diagnostics; its axis/order follows the construction inputs.
    runtime_seconds : float
        Stored runtime seconds; its axis/order follows the construction inputs.
    cache_key : str | None
        Stored cache key; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    method : str
        Stored method supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    n_samples : int
        Stored n samples supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    n_variants : int
        Stored n variants supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    n_variants_evaluated : int | None
        Stored n variants evaluated supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    meff : float | None
        Effective association-test count, or None when the method provides diagnostics only.
        Required construction field.

    tau_ld : float | None
        LD accumulation/inflation factor used by the effective-test estimator, when available.
        Required construction field.

    p_threshold : dict
        Mapping of familywise alpha levels to calibrated p-value thresholds. Required
        construction field.

    neglog10_threshold : dict
        Corresponding negative log10 familywise p-value cutoffs. Required construction field.

    chromosome_results : pd.DataFrame | None
        Per-chromosome effective-test/rank diagnostics, or None when not computed. Required
        construction field.

    diagnostics : dict
        Stored diagnostics supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    runtime_seconds : float
        Elapsed computation time in seconds. Required construction field.

    cache_key : str | None
        Deterministic input/settings fingerprint for cache lookup, or None when not cached.
        Default: ``None``.
    """
    method: str
    n_samples: int
    n_variants: int
    n_variants_evaluated: int | None
    meff: float | None
    tau_ld: float | None
    p_threshold: dict
    neglog10_threshold: dict
    chromosome_results: pd.DataFrame | None
    diagnostics: dict
    runtime_seconds: float
    cache_key: str | None = None


def meff_to_threshold(meff, alpha=(0.10, 0.05), correction="sidak"):
    """Convert an effective test count to familywise p-value thresholds.

    Parameters
    ----------
    meff : float or None
        Effective number of association tests, finite and positive when a threshold is computed.
        None/nonfinite/nonpositive values yield no threshold rather than fabricating a
        calibrated test count. Source comparisons: ``meff <= 0``. See the rejection/fallback
        branches for enforcement.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``(0.1, 0.05)``.

    correction : str
        Familywise threshold correction: 'sidak' uses 1-(1-alpha)^(1/meff); 'bonferroni' uses
        alpha/meff. Unsupported correction names raise in the threshold helper. Default:
        ``'sidak'``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if meff is None or not np.isfinite(meff) or meff <= 0: return {}
    out = {}
    for a in np.atleast_1d(alpha).astype(float):
        if not 0 < a < 1: raise ValueError("alpha must lie in (0,1)")
        p = (-np.expm1(np.log1p(-a) / meff) if correction == "sidak"
             else a / meff if correction == "bonferroni" else None)
        if p is None: raise ValueError("correction must be 'sidak' or 'bonferroni'")
        out[float(a)] = float(p)
    return out


def empirical_threshold_to_meff(p_threshold, alpha):
    """Invert a Sidak familywise threshold to an effective test count.

    Parameters
    ----------
    p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Source comparisons: ``0 < p < 1``. See the rejection/fallback branches for enforcement.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Source comparisons: ``0 < a < 1``. See the rejection/fallback branches for enforcement.

    Returns
    -------
    object
        Computed result from ``float(np.log1p(-a) / np.log1p(-p))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    p = float(p_threshold); a = float(alpha)
    if not 0 < p < 1 or not 0 < a < 1: raise ValueError("p_threshold and alpha must lie in (0,1)")
    return float(np.log1p(-a) / np.log1p(-p))


def _materialize(genotypes, block_size):
    """Normalize arrays/DataFrames or VIGOR genotype sources for this release.

    Source blocks are concatenated chromosome-wise. Callers must select a block
    size/max memory suitable for one chromosome; genome-wide pair matrices are
    never created.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Passed to
        ``genotypes.iter_blocks`` as ``block_size``.
    """
    if isinstance(genotypes, pd.DataFrame):
        return genotypes.to_numpy(), None
    if isinstance(genotypes, np.ndarray): return np.asarray(genotypes), None
    if hasattr(genotypes, "iter_blocks"):
        meta=[]; n=len(genotypes.sample_ids); m=len(genotypes.variant_metadata)
        blocks=iter(genotypes.iter_blocks(representation="dosage", block_size=block_size))
        try: first=next(blocks)
        except StopIteration: raise ValueError("genotype source contains no variants")
        first_data=np.asarray(first.data)
        # Disk-backed staging preserves bounded resident memory for source adapters.
        path=tempfile.NamedTemporaryFile(prefix="domino_effective_tests_",suffix=".dat",delete=False).name
        out=np.memmap(path,mode="w+",dtype=first_data.dtype,shape=(n,m)); st=0
        for b,a in ((first,first_data),):
            out[:,st:st+a.shape[1]]=a
            st+=a.shape[1]; meta.append(b.variants)
        for b in blocks:
            a=np.asarray(b.data)
            out[:,st:st+a.shape[1]]=a
            st+=a.shape[1]; meta.append(b.variants)
        out.flush()
        weakref.finalize(out, lambda p=path: os.path.exists(p) and os.unlink(p))
        return out, pd.concat(meta, ignore_index=True)
    raise TypeError("genotypes must be sample-by-variant array/DataFrame or a VIGOR genotype source")


def _groups(chromosomes, m):
    """Group variant indices by chromosome while preserving the genomic input order.

    Parameters
    ----------
    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine.

    m : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.
    """
    ch = np.repeat("1", m) if chromosomes is None else np.asarray(chromosomes).astype(str)
    if len(ch) != m: raise ValueError("chromosomes length mismatch")
    # preserve genomic input ordering
    return ch, [(c, np.flatnonzero(ch == c)) for c in pd.unique(ch)]


def _variant_subset(X, ix):
    """Use a view for contiguous chromosomes; copy only interleaved inputs.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    ix : int or index array
        Local position, index selection or step used by the stated slice/iteration. Indices must
        refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
        the enclosing dimensions.
    """
    if len(ix) == 0:
        return X[:, :0]
    if ix[-1] - ix[0] + 1 == len(ix):
        return X[:, int(ix[0]):int(ix[-1]) + 1]
    return X[:, ix]


def _iact_one(X, positions, cm, *, block_size, max_bp, max_cm,
              max_variant_lag, debias, backend):
    """Estimate an effective variant count from accumulated local off-diagonal LD.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Passed to
        ``_iter_off_diagonal_r2`` as ``positions``.

    cm : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Passed to
        ``_iter_off_diagonal_r2`` as ``genetic_map``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Passed to
        ``_iter_off_diagonal_r2`` as ``block_size``.

    max_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Passed to
        ``_iter_off_diagonal_r2`` as ``max_bp``.

    max_cm : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Passed to
        ``_iter_off_diagonal_r2`` as ``max_cm``.

    max_variant_lag : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Passed to ``_iter_off_diagonal_r2`` as ``max_variant_lag``.

    debias : bool
        Whether to remove the finite-sample null expectation from pairwise R-squared before
        LD/effective-test accumulation.

    backend : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Passed to ``_iter_off_diagonal_r2`` as
        ``backend``.
    """
    s=0.0; pairs=0
    for b in _iter_off_diagonal_r2(X, block_size=block_size,
            max_variant_lag=max_variant_lag, positions=positions, max_bp=max_bp,
            genetic_map=cm, max_cm=max_cm, backend=backend):
        vals = _debiased_r2(b.r2, b.n_pair) if debias else b.r2
        s += float(np.nansum(vals)); pairs += len(vals)
    tau = 1 + 2*s/max(X.shape[1], 1)
    return X.shape[1]/tau, tau, s, pairs


def _pair_bin_counts(x, edges):
    """Count genomic variant pairs within each distance bin without enumerating all pairs.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x + edges[k + 1]``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation. Passed
        to ``np.searchsorted`` as ``side``.
    """
    x=np.asarray(x, float); total=np.zeros(len(edges)-1, dtype=np.int64)
    for k in range(len(edges)-1):
        # Half-open bins avoid counting an edge in both adjacent bins; only
        # the final upper edge is inclusive, matching the LD sample binning.
        hi=np.searchsorted(x, x+edges[k+1], side="right" if k==len(edges)-2 else "left")
        lo=np.searchsorted(x, x+edges[k], side="left")
        if edges[k] == 0: lo=np.maximum(lo, np.arange(len(x))+1)
        total[k]=np.maximum(hi-lo, 0).sum()
    return total


def _ld_decay_one(X, coord, *, nref, bins, debias, rng, block_size=200):
    """Estimate effective tests from a bounded reference panel and distance-binned LD decay.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support. Passed to ``np.memmap`` as ``shape``.

    coord : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    nref : int
        Maximum reference variants sampled for LD-decay estimation; positive values bound the
        systematic genomic reference panel size.

    bins : int or array-like
        Calibration bin count or ordered bin edges for the statistic/distance being summarized.
        Counts are positive integers; explicit edges must be ordered and cover the intended
        scale. Passed to ``_iter_off_diagonal_r2`` as ``max_bp``.

    debias : bool
        Whether to remove the finite-sample null expectation from pairwise R-squared before
        LD/effective-test accumulation.

    rng : numpy.random.Generator
        Random-number generator used for draws; its current state is consumed by sampling.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``200``. Passed to
        ``_iter_off_diagonal_r2`` as ``block_size``.
    """
    m=X.shape[1]; take=min(m, int(nref))
    # Position-stratified systematic sample avoids density-dominated random draws.
    ix=np.unique(np.linspace(0, m-1, take).round().astype(int))
    sums=np.zeros(len(bins)-1); nums=np.zeros(len(bins)-1,dtype=np.int64); pairs=0
    co=coord[ix]; staged_path=None
    if len(ix) == m:
        xx=X
    else:
        # Fancy indexing a source-backed genome can allocate N*nref in RAM.
        # Stage the systematic reference panel to disk in bounded chunks.
        staged_path=tempfile.NamedTemporaryFile(prefix="domino_ld_decay_",suffix=".dat",delete=False).name
        xx=np.memmap(staged_path,mode="w+",dtype=np.float64,shape=(X.shape[0],len(ix)))
        for st in range(0,len(ix),max(1,int(block_size))):
            xx[:,st:st+block_size]=X[:,ix[st:st+block_size]]
        xx.flush()
    try:
        for rb in _iter_off_diagonal_r2(xx,block_size=block_size,positions=co,max_bp=float(bins[-1])):
            v=_debiased_r2(rb.r2,rb.n_pair) if debias else rb.r2
            bi=np.searchsorted(bins,rb.distance_bp,side="right")-1
            bi[rb.distance_bp == bins[-1]]=len(bins)-2
            for k in range(len(sums)):
                z=v[bi==k]; sums[k]+=np.nansum(z); nums[k]+=np.isfinite(z).sum()
            pairs+=len(v)
    finally:
        if staged_path is not None:
            del xx
            if os.path.exists(staged_path): os.unlink(staged_path)
    means=np.divide(sums,nums,out=np.zeros_like(sums),where=nums>0)
    counts=_pair_bin_counts(coord,bins); S=float(counts@means)
    tau=1+2*S/max(m,1)
    return m/tau,tau,S,pairs,means,counts,len(ix)


def _spectral_one(X, max_block_snps, estimator, boundary_snps=100, boundary_r2=.05):
    """Accumulate blockwise eigenvalue effective-test estimates with LD-aware boundary merging.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    max_block_snps : int
        Variant count per spectral/LD block or boundary panel. Positive counts bound matrix
        dimensions and determine how neighboring blocks are joined.

    estimator : str
        Spectral effective-test estimator: 'li_ji' sums min(eigenvalue,1), 'participation_ratio'
        uses sum(eigenvalue)^2/sum(eigenvalue^2), and 'galwey' uses
        sum(sqrt(eigenvalue))^2/sum(eigenvalue).

    boundary_snps : int
        Variant count per spectral/LD block or boundary panel. Positive counts bound matrix
        dimensions and determine how neighboring blocks are joined. Default: ``100``.

    boundary_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.05``.
    """
    sums={"li_ji":0.,"participation_ratio":0.,"galwey":0.}; neg=0.; clipped=0; blocks=0
    cap=int(max_block_snps); initial=max(2,cap//2)
    spans=[[st,min(X.shape[1],st+initial)] for st in range(0,X.shape[1],initial)]
    merged=[]; k=0
    while k<len(spans):
        left=spans[k]
        if k+1<len(spans) and spans[k+1][1]-left[0] <= cap:
            right=spans[k+1]; q=min(int(boundary_snps),left[1]-left[0],right[1]-right[0])
            rr,nn=_pairwise_r2(X[:,left[1]-q:left[1]],X[:,right[0]:right[0]+q])
            excess=_debiased_r2(rr,nn)
            if np.nanmax(excess,initial=0.) >= float(boundary_r2):
                left=[left[0],right[1]]; k+=1
        merged.append(left); k+=1
    for st,en in merged:
        R,good=local_correlation_matrix(X[:,st:en])
        ev=eigvalsh(R); neg += float(-ev[ev<0].sum()); clipped += int((ev<0).sum()); ev=np.clip(ev,0,None)
        sums["li_ji"] += float(np.minimum(ev,1).sum())
        sums["participation_ratio"] += float(ev.sum()**2/max(np.sum(ev*ev),1e-15))
        sums["galwey"] += float(np.sqrt(ev).sum()**2/max(ev.sum(),1e-15)); blocks += 1
    if estimator not in sums: raise ValueError("unknown spectral_estimator")
    return sums[estimator],sums,neg,clipped,blocks


def _grm_summary(grms, levels):
    """Return the genomic relationship matrix summary result computed by ``pd.DataFrame``.

    Parameters
    ----------
    grms : mapping
        Named animal relationship kernels; each matrix/operator must use the same animal order.
        Names identify the matching trait covariance components.

    levels : sequence of float
        Cumulative eigenvalue/variance fractions in (0, 1] for effective-rank diagnostics. Each
        probability selects the rank needed to explain that fraction.
    """
    if isinstance(grms, dict): items=list(grms.items())
    else: items=[("genome",grms)]
    rows=[]
    for name,K in items:
        a=np.asarray(K,float)
        if a.ndim==1:
            if len(a)==0 or np.any(~np.isfinite(a)): raise ValueError("GRM eigenvalues must be nonempty and finite")
            raw=np.sort(a)[::-1]; provenance="precomputed"
        else:
            if a.ndim!=2 or a.shape[0]!=a.shape[1] or np.any(~np.isfinite(a)): raise ValueError("GRM must be a finite square matrix or eigenvalue vector")
            raw=eigvalsh((a+a.T)/2)[::-1]; provenance="computed"
        negative_count=int(np.sum(raw<0)); negative_sum=float(-raw[raw<0].sum())
        ev=np.clip(raw,0,None)
        sm=ev.sum(); p=ev/sm if sm>0 else np.zeros_like(ev)
        row={"chromosome":str(name),"participation_rank":float(sm*sm/max(ev@ev,1e-15)),
             "entropy_rank":float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
             "largest_eigenvalue_fraction":float(p[0] if len(p) else np.nan),
             "eigenvalue_provenance":provenance,"negative_eigenvalues_clipped":negative_count,
             "negative_eigenvalue_sum":negative_sum}
        cs=np.cumsum(ev)/sm if sm>0 else np.zeros_like(ev)
        for q in levels: row[f"rank_{str(q).replace('.','')}" ]=int(np.searchsorted(cs,q)+1) if sm>0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _hash_array(h, value):
    """Update the fingerprint hash with values, shape and dtype in deterministic order.

    Parameters
    ----------
    h : hash object or numeric array
        Accumulator/operand consumed by the explicit operation below; for fingerprint helpers
        supply a hashlib hash object, and for numerical helpers use the aligned numeric quantity
        from the caller.

    value : object (see the cited input contract)
        Input for the operation ``value[k]``. See ``vigor.gwas.effective_tests._hash_array`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.
    """
    if value is None: h.update(b"<none>"); return
    if isinstance(value,dict):
        for k in sorted(value,key=str): h.update(str(k).encode()); _hash_array(h,value[k])
        return
    a=np.asarray(value); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
    if a.dtype.kind in "OUS":
        for item in np.nditer(a, flags=["refs_ok", "zerosize_ok"], order="C"):
            encoded=str(item.item()).encode("utf-8")
            h.update(len(encoded).to_bytes(8,"little")); h.update(encoded)
        return
    # nditer buffering hashes non-contiguous arrays without flattening the
    # entire input into a temporary resident-memory copy.
    for chunk in np.nditer(a, flags=["external_loop", "buffered", "zerosize_ok"],
                           order="C", buffersize=(1 << 20) // max(a.dtype.itemsize,1)):
        h.update(np.ascontiguousarray(chunk).view(np.uint8))


def _cheap_source_signature(source):
    """Cheap, stable cache identity for file/source-backed genotypes.

    We intentionally hash labels/coordinates and backing-file stat metadata,
    not every genotype byte.  This makes a cache hit O(number of labels) rather
    than O(n_samples*n_variants) and avoids staging a multi-hundred-GB matrix
    merely to discover that a cached threshold already exists.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.
    """
    vm=pd.DataFrame(source.variant_metadata)
    sig={
        "class":type(source).__name__,
        "shape":[int(len(source.sample_ids)),int(len(vm))],
        "sample_ids":np.asarray(source.sample_ids,dtype=str),
    }
    for col in ("variant_id","snp","chromosome","chrom","position","pos"):
        if col in vm.columns:
            sig[col]=vm[col].to_numpy()
    paths=[]
    prefix=getattr(source,"prefix",None)
    if prefix is None and hasattr(source,"reader"):
        prefix=getattr(source.reader,"prefix",None)
    if prefix is not None:
        prefix=str(prefix)
        for ext in (".bed",".bim",".fam",".pgen",".pvar",".psam"):
            q=Path(prefix+ext)
            if q.exists():
                st=q.stat(); paths.append((str(q.resolve()),int(st.st_size),int(st.st_mtime_ns)))
    zpath=getattr(source,"_path",None)
    if zpath is not None:
        q=Path(str(zpath))
        if q.exists():
            st=q.stat(); paths.append((str(q.resolve()),int(st.st_size),int(st.st_mtime_ns)))
    sig["backing_stats"]=paths
    # An in-memory source has no external stat identity; hash its data exactly.
    if hasattr(source,"data") and not paths:
        sig["resident_data"]=np.asarray(source.data)
    return sig


def _fingerprint(X, method, config, extra=()):
    """Hash genotype inputs and estimator settings to identify a reproducible cache entry.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract.

    config : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration.

    extra : tuple
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``()``.
    """
    h=hashlib.sha256()
    if isinstance(X,dict):
        _hash_array(h,X)
    else:
        h.update(str(X.shape).encode()); h.update(str(X.dtype).encode()); _hash_array(h,X)
    for value in extra: _hash_array(h,value)
    h.update(method.encode()); h.update(json.dumps(config,sort_keys=True,default=str).encode()); return h.hexdigest()


def _limit_blas_threads(func):
    """Limit BLAS threads while executing the wrapped effective-test estimator.

    Parameters
    ----------
    func : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """Execute the wrapped callable with the enclosing compatibility/progress policy.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments forwarded, in order, to ``func``. See that signature for
            required scientific inputs and their order.

        **kwargs : mapping
            Additional keyword arguments forwarded to ``func``. See those callables for accepted
            keys and values. Passed to ``threadpool_limits`` as ``limits``.

        Returns
        -------
        object
            Computed result from ``func(*args, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        with threadpool_limits(limits=max(1, int(kwargs.get("threads", 1)))):
            return func(*args, **kwargs)
    return wrapped


@_limit_blas_threads
def effective_tests(genotypes=None, method="auto", *, positions=None, chromosomes=None,
                    genetic_map=None, alpha=(0.10,0.05), correction="sidak",
                    max_bp="auto", max_cm=None, max_variant_lag=None,
                    r2_backend="exact", debias_r2=True, n_reference_snps=100_000,
                    block_size=200, threads=1, random_state=42, grm=None,
                    covariates=None, null_model=None, cache=True, cache_dir=None,
                    distance_bins="auto", spectral_estimator="li_ji",
                    max_block_snps=None, variance_levels=(.90,.95,.99,.999,.9999),
                    calibration=None, min_replicates=200, max_replicates=5000,
                    batch_replicates=100, target_relative_se=.03,
                    null_backend="score", refit_null=False,
                    boundary_snps=100, boundary_r2=.05, **kwargs):
    """Estimate effective GWAS tests using one of five complementary methods.

    ``grm_spectral`` without ``calibration`` deliberately returns ``meff=None``:
    sample-space GRM rank is not an identified SNP-wise FWER test count.
    ``refit_null=True`` and external-GCTA are reserved but unsupported here.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    method : callable or str
        Algorithm/estimator selected for the operation described above. See the source-derived
        choices and called model factory for the accepted methods; callable methods must obey
        that factory's fit/predict or objective contract. Default: ``'auto'``. Source-recognized
        literal values: ``block_spectral``, ``grm_spectral``, ``ld_decay``, ``ld_iact``,
        ``null_maxt``. These select the corresponding branches below; comparison evidence alone
        does not establish an exhaustive enum. Passed to ``EffectiveTestsResult`` as ``method``.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value. Passed to ``_fingerprint`` as ``extra``.

    chromosomes : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``. Passed to ``_fingerprint`` as ``extra``.

    genetic_map : one-dimensional array-like or None
        Genetic-map coordinates in centimorgans aligned to variants. None omits genetic-distance
        filtering or uses the routine's physical-to-genetic approximation. Default: ``None``.
        Passed to ``_fingerprint`` as ``extra``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``(0.1, 0.05)``. Passed to ``adaptive_maxT`` as ``alpha``.

    correction : str
        Familywise threshold correction: 'sidak' uses 1-(1-alpha)^(1/meff); 'bonferroni' uses
        alpha/meff. Unsupported correction names raise in the threshold helper. Default:
        ``'sidak'``.

    max_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``'auto'``.
        Passed to ``_iact_one`` as ``max_bp``.

    max_cm : float or None
        Genetic-distance bound in centimorgans for pairs or locus merging. Nonnegative values
        restrict recombination distance; None disables this optional distance filter. Default:
        ``None``. Passed to ``_iact_one`` as ``max_cm``.

    max_variant_lag : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``None``. Passed to ``_iact_one`` as ``max_variant_lag``.

    r2_backend : str or callable
        Null/LD backend used for empirical calibration. See the local dispatcher for implemented
        models; callable backends must return the simulated score/LD representation consumed
        here. Default: ``'exact'``. Passed to ``_iact_one`` as ``backend``.

    debias_r2 : bool
        Whether to remove the finite-sample null expectation from pairwise R-squared before
        LD/effective-test accumulation. Default: ``True``. Passed to ``_iact_one`` as
        ``debias``.

    n_reference_snps : int
        Maximum reference variants sampled for LD-decay estimation; positive values bound the
        systematic genomic reference panel size. Default: ``100000``. Passed to
        ``_ld_decay_one`` as ``nref``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``200``. Source
        comparisons: ``int(block_size) < 1``. See the rejection/fallback branches for
        enforcement. Passed to ``_materialize`` as ``block_size``.

    threads : int
        Input for the operation ``int(threads) < 1``. See
        ``vigor.gwas.effective_tests.effective_tests`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``1``.
        Source comparisons: ``int(threads) < 1``. See the rejection/fallback branches for
        enforcement.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``42``. Passed to
        ``adaptive_maxT`` as ``random_state``.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value. Passed to ``_fingerprint`` as ``extra``.

    covariates : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced. Default: ``None``. Passed to
        ``_fingerprint`` as ``extra``.

    null_model : str or callable
        Null/LD backend used for empirical calibration. See the local dispatcher for implemented
        models; callable backends must return the simulated score/LD representation consumed
        here. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value.

    cache : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    cache_dir : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    distance_bins : int or array-like
        Calibration bin count or ordered bin edges for the statistic/distance being summarized.
        Counts are positive integers; explicit edges must be ordered and cover the intended
        scale. Default: ``'auto'``.

    spectral_estimator : str
        Spectral effective-test estimator: 'li_ji' sums min(eigenvalue,1), 'participation_ratio'
        uses sum(eigenvalue)^2/sum(eigenvalue^2), and 'galwey' uses
        sum(sqrt(eigenvalue))^2/sum(eigenvalue). Default: ``'li_ji'``.

    max_block_snps : int
        Variant count per spectral/LD block or boundary panel. Positive counts bound matrix
        dimensions and determine how neighboring blocks are joined. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    variance_levels : sequence of float
        Cumulative eigenvalue/variance fractions in (0, 1] for effective-rank diagnostics. Each
        probability selects the rank needed to explain that fraction. Default: ``(0.9, 0.95,
        0.99, 0.999, 0.9999)``.

    calibration : object (see the cited input contract)
        Input for the operation ``required - set(calibration)``. See
        ``vigor.gwas.effective_tests.effective_tests`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    min_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``200``. Source comparisons: ``int(min_replicates) < 1``. See the
        rejection/fallback branches for enforcement. Passed to ``adaptive_maxT`` as
        ``min_replicates``.

    max_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``5000``. Passed to ``adaptive_maxT`` as ``max_replicates``.

    batch_replicates : int
        Minimum/maximum null simulation replicate count or incremental simulation batch size.
        Positive counts control adaptive maxT work; the maximum must not be below the minimum.
        Default: ``100``. Source comparisons: ``int(batch_replicates) < 1``. See the
        rejection/fallback branches for enforcement. Passed to ``adaptive_maxT`` as
        ``batch_replicates``.

    target_relative_se : float
        Positive target relative Monte Carlo standard error for the adaptive null-threshold
        estimate. Smaller values request more simulation precision. Default: ``0.03``. Passed to
        ``adaptive_maxT`` as ``target_relative_se``.

    null_backend : str or callable
        Null/LD backend used for empirical calibration. See the local dispatcher for implemented
        models; callable backends must return the simulated score/LD representation consumed
        here. Default: ``'score'``.

    refit_null : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``False``.

    boundary_snps : int
        Variant count per spectral/LD block or boundary panel. Positive counts bound matrix
        dimensions and determine how neighboring blocks are joined. Default: ``100``.

    boundary_r2 : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.05``.

    **kwargs : mapping
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    t0=time.perf_counter(); method=str(method).lower()
    if kwargs: raise TypeError("unexpected effective_tests options: "+", ".join(sorted(kwargs)))
    if int(threads) < 1: raise ValueError("threads must be at least 1")
    if correction not in {"sidak", "bonferroni"}:
        raise ValueError("correction must be 'sidak' or 'bonferroni'")
    # Validation found deterministic LD heuristics can be anti-conservative for
    # smooth LD. Until population-specific validation exists, empirical maxT is
    # the only defensible automatic significance calibration.
    if method == "auto": method = "null_maxt"
    allowed={"ld_decay","ld_iact","block_spectral","null_maxt","grm_spectral"}
    if method not in allowed: raise ValueError(f"method must be one of {sorted(allowed)}")
    if method == "null_maxt" and null_backend != "score": raise NotImplementedError("only null_backend='score' is implemented")
    if refit_null: raise NotImplementedError("per-replicate null-model refitting is a validation backend, not implemented")
    alphas=np.atleast_1d(alpha).astype(float)
    if np.any(~np.isfinite(alphas)) or np.any((alphas <= 0) | (alphas >= 1)): raise ValueError("alpha must lie in (0,1)")
    if int(block_size)<1 or int(min_replicates)<1 or int(max_replicates)<int(min_replicates) or int(batch_replicates)<1:
        raise ValueError("block_size, min_replicates, and batch_replicates must be positive and max_replicates >= min_replicates")
    if null_model is not None:
        unknown=set(null_model)-{"variance_components"}
        if unknown: raise TypeError("unexpected null_model options: "+", ".join(sorted(unknown)))
    X=meta=None; source_backed=False; source_signature=None
    if genotypes is not None:
        source_backed=hasattr(genotypes,"iter_blocks") and hasattr(genotypes,"sample_ids") and hasattr(genotypes,"variant_metadata")
        if source_backed:
            meta=pd.DataFrame(genotypes.variant_metadata).reset_index(drop=True)
            if positions is None and "position" in meta: positions=meta.position.to_numpy()
            if chromosomes is None and "chromosome" in meta: chromosomes=meta.chromosome.to_numpy()
            n,m=int(len(genotypes.sample_ids)),int(len(meta))
            source_signature=_cheap_source_signature(genotypes)
            # null_maxT now streams the source directly and never stages the
            # complete sample x variant matrix. Other LD estimators still need
            # random-access chromosome matrices and materialize on cache miss.
            X=genotypes if method=="null_maxt" else None
        else:
            X,meta=_materialize(genotypes, block_size=max(block_size,1024))
            if meta is not None:
                if positions is None and "position" in meta: positions=meta.position.to_numpy()
                if chromosomes is None and "chromosome" in meta: chromosomes=meta.chromosome.to_numpy()
            n,m=X.shape
    elif method=="grm_spectral":
        n,m=np.asarray(next(iter(grm.values())) if isinstance(grm,dict) else grm).shape[0],0
    else:
        n=m=0
    if method != "grm_spectral" and genotypes is None: raise ValueError(f"genotypes are required for {method}")
    if method == "grm_spectral" and grm is None: raise ValueError("grm is required for grm_spectral")
    if method != "grm_spectral" and (n < 2 or m < 1):
        raise ValueError("genotypes must contain at least two samples and one variant")
    config={"method":method,"max_bp":max_bp,"max_cm":max_cm,"lag":max_variant_lag,
            "r2_backend":r2_backend,"debias":debias_r2,"nref":n_reference_snps,
            "block":block_size,"threads":threads,"seed":random_state,
            "distance_bins":distance_bins,"spectral":spectral_estimator,
            "max_block_snps":max_block_snps,"variance_levels":variance_levels,
            "boundary_snps":boundary_snps,"boundary_r2":boundary_r2,
            "alpha":list(np.atleast_1d(alpha).astype(float)),"correction":correction,
            "min_replicates":min_replicates,"max_replicates":max_replicates,
            "batch_replicates":batch_replicates,"target_relative_se":target_relative_se,
            "null_backend":null_backend,"refit_null":refit_null,
            "null_model":null_model,"calibration":calibration}
    fingerprint_input=(source_signature if source_signature is not None else
                       (X if X is not None else np.empty((n,0))))
    key=_fingerprint(fingerprint_input, method, config,
                     extra=(positions,chromosomes,genetic_map,grm,covariates))
    cpath=Path(cache_dir or ".domino_cache/effective_tests")/key if cache else None
    if cpath is not None and (cpath/"summary.json").exists():
        d=json.loads((cpath/"summary.json").read_text()); df=pd.DataFrame(d.pop("chromosome_results")) if d.get("chromosome_results") is not None else None
        d["p_threshold"]={float(k):v for k,v in d["p_threshold"].items()}
        d["neglog10_threshold"]={float(k):v for k,v in d["neglog10_threshold"].items()}
        d.pop("chromosome_results",None); d["chromosome_results"]=df; return EffectiveTestsResult(**d)
    # For non-maxT source-backed LD methods, defer the large staging pass until
    # after the cheap cache lookup. This preserves compatibility without paying
    # hundreds of GB of scratch I/O on cache hits.
    if source_backed and method not in {"null_maxt","grm_spectral"}:
        X,_meta2=_materialize(genotypes, block_size=max(block_size,1024))
    diag={"limitations":[]}; rows=[]; meff=tau=None; evaluated=None; empirical_p=None
    if source_backed:
        diag["genotype_access"]=("streamed_source" if method=="null_maxt" else "source_staged_after_cache_miss")
        diag["cache_fingerprint"]="labels+coordinates+backing-file-stats"
    if method == "grm_spectral":
        df=_grm_summary(grm,variance_levels); rows=df.to_dict("records")
        sums={c:float(df[c].sum()) for c in df.select_dtypes(include=[np.number]).columns}
        diag["grm_spectral"]={"by_chromosome":rows,**{"sum_"+k:v for k,v in sums.items()},"calibration":calibration}
        if calibration is not None:
            if not calibration.get("validated",False): raise ValueError("GRM calibration must declare validated=True")
            required={"training_population","grm_normalization","validation_relative_error","alpha"}
            missing=required-set(calibration)
            if missing: raise ValueError("validated GRM calibration lacks provenance: "+", ".join(sorted(missing)))
            feature=calibration.get("feature","participation_rank"); x=float(df[feature].sum())
            trained_alpha=np.atleast_1d(calibration["alpha"]).astype(float)
            requested_alpha=np.atleast_1d(alpha).astype(float)
            if any(not np.any(np.isclose(a,trained_alpha)) for a in requested_alpha):
                raise ValueError("requested alpha is outside the GRM calibration alpha set")
            if "feature_range" in calibration:
                lo,hi=map(float,calibration["feature_range"])
                if not lo <= x <= hi: raise ValueError("GRM spectral feature is outside the calibration applicability range")
            kind=calibration.get("kind","linear"); a=float(calibration.get("intercept",0)); b=float(calibration.get("coefficient",1)); power=float(calibration.get("power",1))
            meff=(a+b*x if kind=="linear" else b*x if kind=="through_origin" else b*x**power if kind=="power" else None)
            if meff is None: raise ValueError("unknown calibration kind")
            if not np.isfinite(meff) or meff <= 0: raise ValueError("calibration predicted a nonpositive/nonfinite Meff")
        else: diag["limitations"].append("Uncalibrated GRM ranks are sample-space diagnostics and do not define a GWAS threshold.")
    elif method == "null_maxt":
        nm={} if null_model is None else dict(null_model)
        empirical_p,maxima,zthr,se,ci,ngood=adaptive_maxT(X,alpha=alpha,covariates=covariates,grm=grm,
            variance_components=nm.get("variance_components"),min_replicates=min_replicates,max_replicates=max_replicates,
            batch_replicates=batch_replicates,target_relative_se=target_relative_se,random_state=random_state,
            variant_block_size=max(block_size,256))
        eq={a:empirical_threshold_to_meff(empirical_p[a],a) for a in empirical_p}; meff=float(np.median(list(eq.values())))
        diag.update({"empirical_equivalent_meff":eq,"n_replicates_used":len(maxima),"max_abs_z_threshold":zthr,
                     "threshold_z_se":se,"threshold_z_ci":ci,"score_variants":ngood,"empirical_threshold_primary":True,
                     "limitations":["Adaptive quantile precision is estimated by bootstrap; validate final FWER on independent outer null replicates."]})
        evaluated=m
    else:
        ch,groups=_groups(chromosomes,m); pos=None if positions is None else np.asarray(positions,float); cm=None if genetic_map is None else np.asarray(genetic_map,float)
        if pos is not None and len(pos)!=m: raise ValueError("positions length mismatch")
        if cm is not None and len(cm)!=m: raise ValueError("genetic_map length mismatch")
        if pos is not None and np.any(~np.isfinite(pos)): raise ValueError("positions must be finite")
        if cm is not None and np.any(~np.isfinite(cm)): raise ValueError("genetic_map must be finite")
        if max_bp == "auto":
            # A requested genetic window takes precedence over the physical
            # fallback rather than silently intersecting two distance systems.
            max_bp = None if pos is None or max_cm is not None else 1_000_000.
        total=0.; weighted_tau=0.; neval=0
        for cname,ix in groups:
            xp=_variant_subset(X,ix); pp=None if pos is None else pos[ix]; cc=None if cm is None else cm[ix]
            if pp is not None and np.any(np.diff(pp)<0): raise ValueError(f"positions must be sorted within chromosome {cname}")
            if cc is not None and np.any(np.diff(cc)<0): raise ValueError(f"genetic_map must be sorted within chromosome {cname}")
            if method == "ld_iact":
                e,t,s,pairs=_iact_one(xp,pp,cc,block_size=block_size,max_bp=max_bp,max_cm=max_cm,max_variant_lag=max_variant_lag,debias=debias_r2,backend=r2_backend)
                row={"chromosome":cname,"n_variants":len(ix),"meff":e,"tau_ld":t,"sum_excess_r2":s,"pairs_evaluated":pairs}; neval+=len(ix)
            elif method == "ld_decay":
                coord=cc if max_cm is not None and cc is not None else pp if pp is not None else np.arange(len(ix),dtype=float)
                # With no map, a bounded variant-lag window is essential: using
                # the whole chromosome would turn a 100k reference panel into
                # an O(100k^2) calculation. This fallback is an approximation
                # and remains explicit in diagnostics.
                fallback_lag = min(1000, max(len(ix) - 1, 1))
                lim=max_cm if max_cm is not None else max_bp if max_bp is not None else max_variant_lag if max_variant_lag is not None else fallback_lag
                if max_cm is None and pp is None and max_variant_lag is None:
                    diag["limitations"].append(
                        "ld_decay used the automatic 1000-variant LD window because no physical or genetic map was supplied.")
                bins=np.linspace(0,float(lim),33) if isinstance(distance_bins,str) and distance_bins=="auto" else np.asarray(distance_bins,float)
                if bins.ndim!=1 or len(bins)<2 or np.any(~np.isfinite(bins)) or np.any(np.diff(bins)<=0): raise ValueError("distance_bins must be finite and strictly increasing")
                e,t,s,pairs,means,counts,nr=_ld_decay_one(xp,coord,nref=max(2,round(n_reference_snps*len(ix)/m)),bins=bins,debias=debias_r2,rng=np.random.default_rng(random_state),block_size=block_size)
                row={"chromosome":cname,"n_variants":len(ix),"meff":e,"tau_ld":t,"sum_excess_r2":s,"pairs_evaluated":pairs,"reference_snps":nr}; neval+=nr
                diag.setdefault("ld_decay_bins",{})[cname]={"edges":bins.tolist(),"mean_excess_r2":means.tolist(),"pair_counts":counts.tolist()}
            else:
                cap=int(max_block_snps or min(max(n-1,2),2000)); e,allest,neg,clip,nb=_spectral_one(xp,cap,spectral_estimator,boundary_snps,boundary_r2)
                row={"chromosome":cname,"n_variants":len(ix),"meff":e,"blocks":nb,"negative_eigenvalue_sum":neg,"eigenvalues_clipped":clip,**allest}; neval+=len(ix)
            rows.append(row); total+=e; weighted_tau+=len(ix)*row.get("tau_ld",len(ix)/max(e,1e-15))
        meff=float(np.clip(total,1,m)); tau=float(weighted_tau/max(m,1)); evaluated=neval
        if method in {"ld_iact","ld_decay"}:
            diag["limitations"].append("The positive-part finite-sample r2 correction retains positive residual bias and can lower Meff; LD-sum Meff is a heuristic requiring FWER validation against null_maxT.")
            if len(groups)>1: diag["limitations"].append("Chromosome-wise Meff additivity assumes negligible cross-chromosome test correlation and is approximate under population structure.")
        if method == "block_spectral": diag["limitations"].append("Adaptive local blocks reduce but cannot eliminate cross-boundary LD; inspect block-size and boundary-r2 sensitivity.")
    pthr=empirical_p if empirical_p is not None else meff_to_threshold(meff,alpha,correction)
    result=EffectiveTestsResult(method="null_maxT" if method=="null_maxt" else method,n_samples=n,n_variants=m,n_variants_evaluated=evaluated,
        meff=None if meff is None else float(meff),tau_ld=tau,p_threshold=pthr,
        neglog10_threshold={a:float(-np.log10(p)) for a,p in pthr.items()},
        chromosome_results=None if not rows else pd.DataFrame(rows),diagnostics=diag,
        runtime_seconds=time.perf_counter()-t0,cache_key=key)
    if cpath is not None:
        cpath.mkdir(parents=True,exist_ok=True); d=asdict(result); d["chromosome_results"]=None if result.chromosome_results is None else result.chromosome_results.to_dict("records")
        payload=json.dumps(d,indent=2,default=str)
        tmp=cpath/("summary.json.tmp-"+str(os.getpid()))
        tmp.write_text(payload); os.replace(tmp,cpath/"summary.json")
    return result
