"""Experimental packed-BED kernels for VIGOR 2.12.1.

CPU paths avoid materializing a genome-wide float genotype matrix.  They are
kept behind explicit/auto engines until benchmarked on the host hardware.
"""
from __future__ import annotations
from .io import open_bed_memmap
import numpy as np


def _metadata(prefix):
    """Return the metadata result computed by ``f``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.
    """
    from .packed_bed import _metadata as f
    return f(prefix)


def _resolve(prefix, sample_ids=None):
    """Resolve PLINK metadata and requested animal IDs into ordered sample positions.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.
    """
    fam,bim=_metadata(prefix)
    ids=fam['iid'].astype(str).to_numpy()
    if sample_ids is None:
        sel=np.arange(len(ids),dtype=np.int64); out_ids=ids
    else:
        wanted=np.asarray(sample_ids).astype(str); d={x:i for i,x in enumerate(ids)}
        sel=np.asarray([d[x] for x in wanted],dtype=np.int64); out_ids=wanted
    return fam,bim,sel,out_ids


def _variant_p(prefix, sel, min_maf=1e-4, chromosome=None):
    """Estimate training-sample allele frequencies for the selected packed variants.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sel : int or index array
        Local position, index selection or step used by the stated slice/iteration. Indices must
        refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
        the enclosing dimensions.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.
    """
    from .packed_bed import packed_variant_stats, sample_mask4
    fam,bim=_metadata(prefix); n=len(fam); bpv=(n+3)//4
    chrom=bim['chrom'].astype(str).to_numpy()
    idx=np.arange(len(bim),dtype=np.int64) if chromosome is None else np.flatnonzero(chrom==str(chromosome))
    full=np.zeros(n,bool); full[sel]=True; mask4=sample_mask4(n,full)
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=len(bim))
    cnt=np.empty(len(idx),np.int32); sx=np.empty(len(idx),np.int32)
    bs=65536
    for a in range(0,len(idx),bs):
        gi=idx[a:a+bs]; c,s,_=packed_variant_stats(np.asarray(mm[gi],np.uint8),mask4); cnt[a:a+len(gi)]=c; sx[a:a+len(gi)]=s
    p=np.divide(sx,2.0*cnt,out=np.full(len(idx),np.nan),where=cnt>0)
    maf=np.minimum(p,1-p); keep=np.isfinite(p)&(maf>=float(min_maf))
    return mm,idx[keep],p[keep],cnt[keep]


def packed_countsketch_factor(prefix, rank=512, *, sample_ids=None, chromosome=None,
                              min_maf=1e-4, random_state=0, output_dtype=np.float32):
    """Stream BED directly into a sparse-JL/CountSketch genomic factor.

    Returns F (n x rank) with E[F F.T] equal to the additive GRM under the
    same centered-mean-imputation convention.  No dense n x m genotype matrix
    or n x n GRM is created.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``512``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    dict
        Result fields: ``factor``, ``sample_ids``, ``rank``, ``weight``, ``n_variants_used``, ``representation``, ``source_representation``, ``random_state``. Field arrays preserve the corresponding input axis order.
    """
    from .npplink import decode_bed_chunk_subset
    fam,bim,sel,ids=_resolve(prefix,sample_ids); nfull=len(fam)
    mm,vidx,p,cnt=_variant_p(prefix,sel,min_maf,chromosome)
    rng=np.random.default_rng(random_state); r=int(rank)
    bucket=rng.integers(0,r,size=len(vidx),dtype=np.int32)
    sign=rng.choice(np.asarray([-1.0,1.0],np.float32),size=len(vidx))
    F=np.zeros((len(sel),r),np.float32); weight=0.0
    # Moderate blocks keep only a small temporary float matrix; aggregation is
    # by bucket, so the expensive output is n x rank rather than n x n.
    bs=8192
    for a in range(0,len(vidx),bs):
        gi=vidx[a:a+bs]; pp=p[a:a+bs].astype(np.float32); bb=bucket[a:a+bs]; ss=sign[a:a+bs]
        P=np.asarray(mm[gi],np.uint8)
        X=decode_bed_chunk_subset(P.reshape(-1),nfull,len(gi),sel,dtype=np.float32)
        valid=np.isfinite(X); X-=2*pp[None,:]; X[~valid]=0.0
        # sort by bucket; each bucket gets one BLAS-1 reduction over its SNPs
        order=np.argsort(bb,kind='stable'); bso=bb[order]; Xo=X[:,order]; sso=ss[order]
        cuts=np.r_[0,1+np.flatnonzero(bso[1:]!=bso[:-1]),len(bso)]
        for u,v in zip(cuts[:-1],cuts[1:]):
            F[:,int(bso[u])] += Xo[:,u:v] @ sso[u:v]
        weight += float(np.sum(2*pp*(1-pp)))
    F /= np.sqrt(max(weight,1e-30))
    return {'factor':F.astype(output_dtype,copy=False),'sample_ids':ids,'rank':r,
            'weight':weight,'n_variants_used':int(len(vidx)),'representation':'countsketch_additive',
            'source_representation':'plink1_bed_2bit','random_state':int(random_state)}


def lowrank_from_countsketch(prefix, rank=512, *, oversample=0, sample_ids=None,
                             chromosome=None,min_maf=1e-4,random_state=0,engine='direct'):
    """Return U,s from the streamed CountSketch factor without a dense GRM.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``512``.

    oversample : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``0``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``fn`` as ``sample_ids``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``. Passed to ``fn`` as
        ``chromosome``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``fn`` as ``min_maf``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``fn`` as ``random_state``.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'direct'``. Source-recognized
        literal values: ``2bit``, ``packed``. These select the corresponding branches below;
        comparison evidence alone does not establish an exhaustive enum.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rr=int(rank)+int(oversample)
    fn=packed_countsketch_factor_direct if str(engine).lower() in {'direct','packed','2bit'} else packed_countsketch_factor
    out=fn(prefix,rr,sample_ids=sample_ids,chromosome=chromosome,
           min_maf=min_maf,random_state=random_state)
    F=np.asarray(out['factor'],np.float32)
    # Economy SVD is on n x r, never n x n.
    U,sv,_=np.linalg.svd(F,full_matrices=False)
    k=min(int(rank),len(sv)); out.update({'U':U[:,:k].astype(np.float32),'s':(sv[:k]**2).astype(np.float32)})
    return out


def grm_from_factor(factor, dtype=np.float32):
    """Return the genomic relationship matrix from factor result computed by ``(F @ F.T).astype``.

    Parameters
    ----------
    factor : array-like or fitted basis
        Trait/feature × retained-factor loading matrix or prepared basis. Column order
        identifies factor directions and must agree with the scores used for projection.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    object
        Computed result from ``(F @ F.T).astype(dtype, copy=False)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    F=np.asarray(factor,np.float32)
    return (F@F.T).astype(dtype,copy=False)

_NUMBA_SKETCH = None
def _numba_sketch_kernel():
    """Construct the compiled packed-genotype CountSketch accumulation kernel.
        
    """
    global _NUMBA_SKETCH
    if _NUMBA_SKETCH is not None: return _NUMBA_SKETCH
    import numba
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def kern(P, vidx, sel, p, order, indptr, sign):
        """Execute the packed-genotype sketch accumulation kernel.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        vidx : one-dimensional integer array or None
            Zero-based variant positions to select from the current source. Values must refer to
            existing columns; None retains all eligible variants. Streaming readers may additionally
            require sorted indices.

        sel : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        p : array-like
            Probability or p-value quantities transformed or tested by this routine. Finite
            probability inputs are on [0, 1]; preserve alignment with the associated variants or
            observations.

        order : sequence or None
            Topological trait/node ordering restricting paths to earlier-to-later edges. None lets
            the graph routine choose or infer a compatible acyclic ordering.

        indptr : array-like
            Sparse/sketch indexing or random-probe values used by the operator. CSR indptr indexes
            row offsets, sign supplies signed sketch weights and probes have the operator's leading
            dimension; preserve the matching index layout.

        sign : array-like
            Sparse/sketch indexing or random-probe values used by the operator. CSR indptr indexes
            row offsets, sign supplies signed sketch weights and probes have the operator's leading
            dimension; preserve the matching index layout.

        Returns
        -------
        object
            Computed result from ``F``; see the summary and the called implementation for its labeled/numerical representation.
        """
        n=sel.shape[0]; r=indptr.shape[0]-1
        F=np.zeros((n,r),np.float32)
        for b in numba.prange(r):
            a=indptr[b]; z=indptr[b+1]
            for tt in range(a,z):
                j=order[tt]; mu=np.float32(2.0)*p[j]; sg=sign[j]
                for ii in range(n):
                    s=sel[ii]; code=(P[vidx[j],s>>2] >> ((s & 3)*2)) & 3
                    if code != 1:  # PLINK BED: 01 is missing
                        g=np.float32(0.0)
                        if code==2: g=np.float32(1.0)
                        elif code==3: g=np.float32(2.0)
                        F[ii,b] += sg*(g-mu)
        return F
    _NUMBA_SKETCH=kern
    return kern


def packed_countsketch_factor_direct(prefix, rank=512, *, sample_ids=None, chromosome=None,
                                     min_maf=1e-4, random_state=0, output_dtype=np.float32):
    """End-to-end 2-bit BED -> CountSketch factor; no dosage block is decoded.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``512``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    dict
        Result fields: ``factor``, ``sample_ids``, ``rank``, ``weight``, ``n_variants_used``, ``representation``, ``source_representation``, ``random_state``. Field arrays preserve the corresponding input axis order.
    """
    fam,bim,sel,ids=_resolve(prefix,sample_ids)
    mm,vidx,p,cnt=_variant_p(prefix,sel,min_maf,chromosome)
    P=mm  # keep the full BED as a memmap; vidx is applied inside the compiled kernel
    rng=np.random.default_rng(random_state); r=int(rank)
    bucket=rng.integers(0,r,size=len(vidx),dtype=np.int32)
    sign=rng.choice(np.asarray([-1.0,1.0],np.float32),size=len(vidx)).astype(np.float32)
    order=np.argsort(bucket,kind='stable').astype(np.int64); bs=bucket[order]
    counts=np.bincount(bs,minlength=r).astype(np.int64); indptr=np.empty(r+1,np.int64); indptr[0]=0; np.cumsum(counts,out=indptr[1:])
    F=_numba_sketch_kernel()(P,np.asarray(vidx,np.int64),np.asarray(sel,np.int64),np.asarray(p,np.float32),order,indptr,sign)
    weight=float(np.sum(2*p*(1-p))); F/=np.sqrt(max(weight,1e-30))
    return {'factor':F.astype(output_dtype,copy=False),'sample_ids':ids,'rank':r,'weight':weight,
            'n_variants_used':int(len(vidx)),'representation':'countsketch_additive_direct_2bit',
            'source_representation':'plink1_bed_2bit','random_state':int(random_state)}
