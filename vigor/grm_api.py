"""Labeled chromosome GRMs and sharded consolidated Zarr-v3 persistence.

Design: a numerical chromosome is fitted at a time; previous chromosome
matrices are persisted immediately.  The All chromosome is an exactly
denominator-weighted aggregate, not an unweighted chromosome average.
"""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from .grm_methods import validate_grm_method, GRM_SCHEMA_VERSION


def _prefix(source):
    """Inputs: source.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        source : object
            Source of genotype dosages or PLINK1 prefix used to recover the correct genotype and metadata inputs.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    if isinstance(source, (str, os.PathLike)):
        s = str(source)
    else:
        s = getattr(source, "prefix", getattr(getattr(source, "reader",None),"prefix",None))
    if s is None:
        raise ValueError("grm/GRMs currently requires PLINK1 BED or a PLINK1 reader")
    if s.endswith((".bed",".bim",".fam")):s=s[:-4]
    if not all(Path(s+ext).is_file() for ext in (".bed",".bim",".fam")):
        raise FileNotFoundError(f"missing PLINK1 .bed/.bim/.fam for {s!r}")
    return s


def _is_xarray_source(source):
    """Return True for a Zarr genotype store path, an xarray genotype object or an opened xarray source.

    Parameters
    ----------
    source : object
        Genotype input given to ``grm``/``GRMs``: a VIGOR Zarr store directory written by
        ``write_genotype_zarr``, an ``xarray.Dataset``/``DataArray`` such as one returned by
        ``open_genotype_dataset``, an opened ``XarrayGenotypeSource``, or a PLINK1 prefix.

    Returns
    -------
    bool
        Whether the input is routed through the bounded ``build_grms`` block reader.
    """
    if isinstance(source,(xr.Dataset,xr.DataArray)):return True
    if hasattr(source,"iter_blocks") and hasattr(source,"ds"):return True
    if isinstance(source,(str,os.PathLike)):
        p=Path(str(source))
        return p.is_dir() and (p.suffix==".zarr" or any((p/f).exists() for f in ("zarr.json",".zgroup",".zattrs")))
    return False


def _object_strings(a):
    """Coerce identifiers (including Zarr v3 StringDType and bytes) to an object array of Python str.

    Parameters
    ----------
    a : array-like
        One-dimensional sample or variant labels read from a genotype store.

    Returns
    -------
    numpy.ndarray
        Object-dtype array of ``str`` with the input order preserved.
    """
    return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in np.asarray(a).ravel()],dtype=object)


def _coerce_string_coords(ds):
    """Replace string-like 1-D sample/variant variables (StringDType, bytes, object) by object str arrays.

    Parameters
    ----------
    ds : xarray.Dataset
        Canonical genotype Dataset with ``sample`` and ``variant`` dimensions.

    Returns
    -------
    xarray.Dataset
        Same Dataset with string coordinates/variables coerced; numeric arrays are untouched
        and genotype arrays stay lazy.
    """
    co,dv={},{}
    for name,v in ds.variables.items():
        if v.ndim==1 and v.dims[0] in ("sample","variant") and v.dtype.kind in "TSO":
            (co if name in ds.coords else dv)[name]=(v.dims,_object_strings(v.values))
    if co:ds=ds.assign_coords(co)
    if dv:ds=ds.assign(dv)
    return ds


def _open_xarray_genotypes(source):
    """Open a Zarr/xarray genotype input as a lazy ``XarrayGenotypeSource`` without materialising calls.

    Parameters
    ----------
    source : object
        Zarr store path, ``xarray.Dataset``/``DataArray`` (dimensions ``sample``/``variant`` or
        ``iid``/``snp``) or an already opened ``XarrayGenotypeSource``.

    Returns
    -------
    tuple
        ``(src, owned)`` where ``owned`` tells the caller to close ``src`` when finished.
    """
    from .genotype_source import XarrayGenotypeSource
    if hasattr(source,"iter_blocks") and hasattr(source,"ds"):
        return source,False
    if isinstance(source,(xr.Dataset,xr.DataArray)):
        dims={k:v for k,v in (("iid","sample"),("snp","variant")) if k in source.dims and v not in source.dims}
        ds=source.rename(dims) if dims else source
        if isinstance(ds,xr.DataArray):ds=ds.to_dataset(name="GT")
        owned=False
    else:
        from .data_model import open_genotype_dataset
        ds=open_genotype_dataset(str(source));owned=True
    src=XarrayGenotypeSource(_coerce_string_coords(ds))
    src.sample_ids=_object_strings(src.sample_ids).astype(str)
    return src,owned


def _xarray_sample_metadata(src, ids):
    """Return FID and sex labels for requested animals from optional ``sample`` variables.

    Parameters
    ----------
    src : XarrayGenotypeSource
        Opened genotype source whose Dataset may carry ``fid``/``family_id`` and ``sex`` on ``sample``.
    ids : numpy.ndarray
        Requested animal identifiers in kernel order.

    Returns
    -------
    tuple
        ``(fid, sex)`` string arrays; FID defaults to IID and sex to ``'0'`` (unknown).
    """
    pos=pd.Index(np.asarray(src.sample_ids,dtype=str)).get_indexer(ids)
    def pick(names, default):
        for nm in names:
            if nm in src.ds.variables and src.ds[nm].dims==("sample",):
                v=np.asarray(src.ds[nm].values)[pos]
                if v.dtype.kind in "iub":return v.astype(np.int64).astype(str)
                if v.dtype.kind=="f":return np.where(np.isfinite(v),np.nan_to_num(v).astype(np.int64).astype(str),"0")
                return _object_strings(v).astype(str)
        return default
    return (pick(("fid","family_id","FID"),np.asarray(ids,dtype=str)),
            pick(("sex","SEX","sexes"),np.full(len(ids),"0")))


class _ScaledDosageSource:
    """Genotype-source view multiplying the dosage block rows of selected animals by fixed factors.

    Used for confidently detected male-X 0/1 haploid coding on Zarr/xarray sources, mirroring
    the PLINK ``dosage_scale`` path; every other attribute is forwarded to ``src``.
    """
    def __init__(self, src, scale_by_id):
        """Store the wrapped source and the ``{IID: factor}`` mapping (absent IDs use 1)."""
        self._src=src;self._scale=scale_by_id
        self.sample_ids=src.sample_ids;self.variant_metadata=src.variant_metadata
        self.maxmem=getattr(src,"maxmem",None);self.memory_fraction=getattr(src,"memory_fraction",None)

    def iter_blocks(self, representation="dosage", samples=None, **kwargs):
        """Yield the wrapped source's blocks with row-wise dosage factors applied."""
        ids=np.asarray(self.sample_ids if samples is None else samples,dtype=str)
        f=np.asarray([self._scale.get(x,1.) for x in ids],np.float32)[:,None]
        for b in self._src.iter_blocks(representation,samples=samples,**kwargs):
            b.data=np.asarray(b.data,np.float32)*f
            yield b


def _sample_selection(reader, sample_ids):
    """Inputs: reader, sample_ids.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        reader : object
            Open PLINK1 reader supplying validated variant calls and sample metadata in their original aligned order.
        sample_ids : object
            Ordered unique PLINK FAM individual identifiers to include in the resulting kernel row and column axes.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    fam = reader.fam
    iid = fam.iid.astype(str).to_numpy()
    if len(set(iid)) != len(iid):
        raise ValueError("FAM IID entries are not unique; subset requires unambiguous IDs")
    requested = iid if sample_ids is None else np.asarray(sample_ids,dtype=str)
    if len(set(requested))!=len(requested):
        raise ValueError("sample_ids must be unique")
    idxmap={x:i for i,x in enumerate(iid)}
    if any(x not in idxmap for x in requested):
        raise ValueError("sample_ids contains identifiers absent from FAM")
    idx=np.fromiter((idxmap[x] for x in requested),dtype=np.int64,count=len(requested))
    sexes = fam.sex.to_numpy()[idx]
    return idx, requested, fam.fid.astype(str).to_numpy()[idx], sexes


def _chromosome_class(chrom, n_autosomes=20):
    """Inputs: chrom, n_autosomes.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        chrom : object
            Chromosome label matched against the BIM chromosome axis, including autosomal and named sex chromosomes. Source-recognized literal choices: 'M', 'MITO', 'MT', 'X', 'XY', 'Y'.
        n_autosomes : object
            Count of labeled autosomes before the rat X, Y, XY and mitochondrial chromosome conventions. Default: ``20``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    key=str(chrom).upper().replace("CHR","")
    if key.isdigit():
        number=int(key)
        if 1<=number<=n_autosomes:return "autosome"
        if number==n_autosomes+1:return "X"
        if number==n_autosomes+2:return "Y"
        if number==n_autosomes+3:return "XY"
        if number==n_autosomes+4:return "MT"
    if key in {"X","Y","XY","MT","M","MITO"}:
        return "MT" if key in {"MT","M","MITO"} else key
    return "other"


def _detect_x_encoding(reader, sample_index, sexes, variant_index, max_variants=10000):
    """Inputs: reader, sample_index, sexes, variant_index, max_variants.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        reader : object
            Open PLINK1 reader supplying validated variant calls and sample metadata in their original aligned order.
        sample_index : object
            Zero-based ordered positions in PLINK FAM, which are used to preserve sample selection and its metadata.
        sexes : object
            Recorded PLINK FAM sex fields, where one denotes male and two denotes female for sex-chromosome dosage checks.
        variant_index : object
            Integer BIM variant positions restricted to this chromosome and examined without phenotype-driven choices.
        max_variants : object
            Maximum number of chromosome variants to sample for male haploid allele coding quality assessment. Default: ``10000``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    male=np.flatnonzero(np.isin(np.asarray(sexes).astype(str),["1","M","m"]))
    female=np.flatnonzero(np.isin(np.asarray(sexes).astype(str),["2","F","f"]))
    if len(male)<20 or len(female)<20 or len(variant_index)<100:
        return "skipped_insufficient_samples"
    take=np.asarray(variant_index[:max_variants],np.int64)
    observed=np.zeros(3,np.int64)
    # Only male call counts decide whether rescaling is safe.
    for _,X in reader.iter_blocks(block_size=256,variant_index=take,
                                  sample_index=np.asarray(sample_index)[male]):
        for i in range(3):observed[i]+=int(np.sum(np.asarray(X)==i))
    return _classify_male_x_calls(observed)


def _detect_x_encoding_source(src, ids, sexes, chrom, max_variants=10000):
    """Zarr/xarray analogue of ``_detect_x_encoding`` reading bounded male blocks from ``src.iter_blocks``.

    Parameters
    ----------
    src : XarrayGenotypeSource
        Opened lazy genotype source.
    ids : numpy.ndarray
        Requested animal identifiers aligned with ``sexes``.
    sexes : numpy.ndarray
        Sex labels, where 1/M denotes male and 2/F female.
    chrom : str
        X chromosome label in the source variant metadata.
    max_variants : int
        Maximum number of leading chromosome variants examined. Default: ``10000``.

    Returns
    -------
    str
        Encoding status with the same vocabulary as ``_detect_x_encoding``.
    """
    sx=np.asarray(sexes).astype(str)
    male=np.flatnonzero(np.isin(sx,["1","M","m"]));female=np.flatnonzero(np.isin(sx,["2","F","f"]))
    m=int((src.variant_metadata.chromosome.astype(str)==str(chrom)).sum())
    if len(male)<20 or len(female)<20 or m<100:
        return "skipped_insufficient_samples"
    observed=np.zeros(3,np.int64);seen=0
    for b in src.iter_blocks(samples=np.asarray(ids,dtype=str)[male],chrom=str(chrom),block_size=256):
        X=np.asarray(b.data)[:,:max_variants-seen];seen+=X.shape[1]
        for i in range(3):observed[i]+=int(np.sum(X==i))
        if seen>=max_variants:break
    return _classify_male_x_calls(observed)


def _classify_male_x_calls(observed):
    """Classify male X coding from counts of 0/1/2 calls (shared by PLINK and Zarr paths)."""
    alt=int(observed[1]+observed[2])
    if alt<100:return "skipped_insufficient_alternate_calls"
    if observed[1]>0 and observed[2]>0:return "skipped_mixed_diploid_or_unclear"
    if observed[1]>0 and observed[2]==0:return "male_haploid_01_double"
    if observed[1]==0 and observed[2]>0:return "already_diploid_02"
    return "skipped_unknown_encoding"


def _sex_chromosome_embedding(matrix, *, max_n=5000, n_neighbors=100,
                             min_cluster_size=1000):
    """Inputs: matrix, max_n, n_neighbors, min_cluster_size.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute optional Y/MT UMAP and HDBSCAN, or return a reason for skipping.
        
        Parameters
        ----------
        matrix : object
            Scientifically typed argument controlling validated genotype data handling and reproducible output behavior.
        max_n : object
            Scientifically typed argument controlling validated genotype data handling and reproducible output behavior. Default: ``5000``.
        n_neighbors : object
            Scientifically typed argument controlling validated genotype data handling and reproducible output behavior. Default: ``100``.
        min_cluster_size : object
            Scientifically typed argument controlling validated genotype data handling and reproducible output behavior. Default: ``1000``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    n=len(matrix)
    out=(np.full(n,np.nan,dtype=np.float32),
         np.full(n,np.nan,dtype=np.float32),
         np.full(n,-1,dtype=np.int32))
    if n<1000:return (*out,"skipped_below_1000_animals")
    if n>max_n:return (*out,"skipped_memory_5000_limit")
    K=np.asarray(matrix,np.float64)
    if not np.isfinite(K).all():return (*out,"skipped_nonfinite_kernel")
    diag=np.diag(K)
    distance=np.sqrt(np.maximum(diag[:,None]+diag[None,:]-2*K,0.))
    np.fill_diagonal(distance,0.)
    try:
        import umap
        from sklearn.cluster import HDBSCAN
        embedded=umap.UMAP(n_components=2, n_neighbors=min(n_neighbors,n-1),
                            metric="precomputed", random_state=103).fit_transform(
                                np.asarray(distance,np.float32))
        labels=HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(embedded)
    except (ImportError,ValueError,MemoryError,RuntimeError) as exc:
        return (*out,"skipped_"+type(exc).__name__)
    return (np.asarray(embedded[:,0],np.float32),
            np.asarray(embedded[:,1],np.float32),
            np.asarray(labels,np.int32),"completed")


def _grm_payload(source, chromosome, *, sample_ids=None, components=("additive","dominance"),
                 method="npplink", fit_mask=None, min_maf=1e-4, block_size='auto',
                 output_dtype=np.float32, blas_threads='auto', dosage_compensation=False,
                 n_autosomes=20, engine="auto", progress=False, memory_fraction=None, maxmem=None):
    """Inputs: source, chromosome, sample_ids, components, method, fit_mask, min_maf, block_size, output_dtype, blas_threads, dosage_compensation, n_autosomes, engine.
    Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
    Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
    Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
    
    Compute and validate chromosome relationship matrices with documented inputs.
    
    Parameters
    ----------
    source : object
        Source of genotype dosages or PLINK1 prefix used to recover the correct genotype and metadata inputs.
    chromosome : object
        Chromosome label matched literally against BIM variant coordinates, without implicit reordering of sample IDs.
    sample_ids : object
        Ordered unique PLINK FAM individual identifiers to include in the resulting kernel row and column axes. Default: ``None``.
    components : object
        Requested genetic components, additive and/or dominance, estimated with component-specific valid denominator weights. Source-recognized literal choices: 'additive'. Default: ``('additive', 'dominance')``.
    method : object
        GRM normalization choice: npplink is empirical standardized with pairwise missingness; gcta0 per-marker scaled; van VanRaden I global; vitezica legacy A and D. Source-recognized literal choices: 'van', 'vitezica'. Default: ``'npplink'``.
    fit_mask : object
        Boolean mask identifying allele-frequency estimation animals, avoiding use of excluded animals for frequency fitting. Default: ``None``.
    min_maf : object
        Minor allele frequency threshold applied to chromosome dosage calls before kernel accumulation. Default: ``0.0001``.
    block_size : object
        Bounded count of packed BED variant calls loaded and decoded per processing block of animals; 'auto' sizes it from the memory budget, integers are ceilings. Default: ``'auto'``.
    output_dtype : object
        NumPy floating-point output dtype controlling the numerical resolution and storage of GRM kernel values. Default: ``np.float32``.
    blas_threads : object
        Positive BLAS worker-thread limit used for small matrix and dense block multiplications; 'auto' uses every CPU available to the process. Default: ``'auto'``.
    dosage_compensation : object
        Enable conservative inferred male-X recoding only when haploid 0/1 encoding can be established. Default: ``False``.
    n_autosomes : object
        Count of labeled autosomes before the rat X, Y, XY and mitochondrial chromosome conventions. Default: ``20``.
    engine : object
        Select packed bitserial additive or bounded float-BLAS backend, with explicit unsupported component checks. Source-recognized literal choices: 'bitserial'. Default: ``'auto'``.
    
    progress : bool
        Whether to report completed packed variant blocks during computation; disabling progress leaves the numerical results and block-size plan identical. Default: ``False``.
    memory_fraction : float or None
        Fraction of currently available RAM used when maxmem is absent, strictly above zero and at most 0.95; dense numerical matrices and scratch share the budget. Default: ``None`` (80% of physical RAM).
    maxmem : int, unit string or None
        Incremental numerical working-memory ceiling in bytes or binary units such as 8GiB; None uses memory_fraction. Metadata, mapped pages and library/I/O RSS are additional. Default: ``None``.
    
    Returns
    -------
    object
        Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    method,components=validate_grm_method(method,components)
    if _is_xarray_source(source):
        return _grm_payload_xarray(source,chromosome,sample_ids=sample_ids,components=components,
            method=method,fit_mask=fit_mask,min_maf=min_maf,block_size=block_size,
            output_dtype=output_dtype,blas_threads=blas_threads,dosage_compensation=dosage_compensation,
            n_autosomes=n_autosomes,engine=engine,progress=progress,
            memory_fraction=memory_fraction,maxmem=maxmem)
    from .io import PlinkReader
    from .packed_bed import dense_grm_from_bed
    prefix=_prefix(source)
    with PlinkReader(prefix,dtype=np.float32) as reader:
        sel,ids,fid,sexes=_sample_selection(reader,sample_ids)
        chrom=str(chromosome)
        bim=reader.bim
        variant_index=np.flatnonzero(bim.chrom.astype(str).to_numpy()==chrom)
        if len(variant_index)==0:
            raise ValueError(f"chromosome {chrom!r} has no variants")
        fit = np.ones(len(ids),bool) if fit_mask is None else np.asarray(fit_mask,bool)
        if fit.shape!=(len(ids),) or not fit.any():
            raise ValueError("fit_mask must be length sample_ids, with >=1 selected")
        dosage="not_applicable"
        if _chromosome_class(chrom,n_autosomes)=="X" and dosage_compensation:
            dosage=_detect_x_encoding(reader,sel,sexes,variant_index)
        dosage_scale = None
        if dosage == 'male_haploid_01_double':
            dosage_scale = np.where(np.isin(np.asarray(sexes).astype(str), ['1', 'M', 'm']), 2., 1.)
        r = dense_grm_from_bed(prefix, components=components, sample_ids=ids,
            fit_mask=fit, chromosome=chrom, min_maf=min_maf, chunk_variants=block_size,
            output_dtype=output_dtype, blas_threads=blas_threads, progress=progress,
            engine=engine, method=method, memory_fraction=memory_fraction,
            maxmem=maxmem, dosage_scale=dosage_scale)
        kernels = {c: np.asarray(r['kernels'][c]) for c in components}
        weights = r['weights']
        diag = r['diagnostics']
        return kernels, weights, diag, ids, fid, sexes, dosage


def _grm_payload_xarray(source, chromosome, *, sample_ids=None, components=("additive","dominance"),
                        method="npplink", fit_mask=None, min_maf=1e-4, block_size='auto',
                        output_dtype=np.float32, blas_threads='auto', dosage_compensation=False,
                        n_autosomes=20, engine="auto", progress=False, memory_fraction=None, maxmem=None):
    """Zarr/xarray counterpart of ``_grm_payload`` using the ``build_grms`` block reader.

    Genotypes are read as bounded sample × variant blocks through
    ``grm_store._compute_chromosome_grms`` (lazy Dask/Zarr slices, packed ``GT_packed2``
    decoded per block), so the full genotype matrix is never materialised. The arithmetic is
    the float-BLAS ``grm_from_blocks`` accumulator used for PLINK float engines; the packed
    bitserial engine is PLINK-only. Parameters and return value match ``_grm_payload``.
    """
    from .grm_store import _compute_chromosome_grms
    if str(engine).lower() not in {"auto","float","blas"}:
        raise ValueError(f"engine={engine!r} requires PLINK1 BED; Zarr/xarray sources use the float-BLAS block reader (engine='auto')")
    src,owned=_open_xarray_genotypes(source)
    try:
        source_ids=np.asarray(src.sample_ids,dtype=str)
        ids=source_ids if sample_ids is None else _object_strings(sample_ids).astype(str)
        if len(set(ids))!=len(ids):raise ValueError("sample_ids must be unique")
        if len(set(source_ids))!=len(source_ids):raise ValueError("genotype sample IDs are not unique; subset requires unambiguous IDs")
        if np.any(pd.Index(source_ids).get_indexer(ids)<0):
            raise ValueError("sample_ids contains identifiers absent from the genotype store")
        fid,sexes=_xarray_sample_metadata(src,ids)
        chrom=str(chromosome)
        m=int((src.variant_metadata.chromosome.astype(str)==chrom).sum())
        if m==0:raise ValueError(f"chromosome {chrom!r} has no variants")
        fit=np.ones(len(ids),bool) if fit_mask is None else np.asarray(fit_mask,bool)
        if fit.shape!=(len(ids),) or not fit.any():
            raise ValueError("fit_mask must be length sample_ids, with >=1 selected")
        dosage="not_applicable"
        if _chromosome_class(chrom,n_autosomes)=="X" and dosage_compensation:
            dosage=_detect_x_encoding_source(src,ids,sexes,chrom)
        reader=src
        if dosage=='male_haploid_01_double':
            male=np.isin(np.asarray(sexes).astype(str),['1','M','m'])
            reader=_ScaledDosageSource(src,{x:2. for x in ids[male]})
        out,diag=_compute_chromosome_grms(reader,chrom,sample_ids=ids,fit_mask=fit,include=components,
            block_size=block_size,min_maf=min_maf,compute_dtype=np.float32,accum_dtype=np.float32,
            blas_threads=blas_threads,progress=progress,maxmem=maxmem,
            memory_fraction=memory_fraction,method=method)
        kernels={c:np.asarray(out[c][0]) for c in components}
        weights={c:out[c][1] for c in components}
        return kernels,weights,diag,ids,fid,sexes,dosage
    finally:
        if owned and hasattr(src,"close"):src.close()


def _source_description(genotypes):
    """Return ``(source_location, genotype_source_type)`` attributes for a PLINK or Zarr/xarray input."""
    if not _is_xarray_source(genotypes):
        return str(Path(_prefix(genotypes)).resolve()),"PLINK1_2bit"
    if isinstance(genotypes,(str,os.PathLike)):
        return str(Path(str(genotypes)).resolve()),"zarr3"
    ds=getattr(genotypes,"ds",genotypes)
    path=getattr(genotypes,"_path",None) or getattr(ds,"attrs",{}).get("genotype_source")
    return (str(path) if path else "in_memory_xarray"),"xarray"


def grm(genotypes, *, components=("additive","dominance"), chromosome=None, method="npplink",
        sample_ids=None, fit_mask=None, min_maf=1e-4, block_size='auto',
        output_dtype=np.float32, blas_threads='auto', progress=False,
        dosage_compensation=False, n_autosomes=20, engine="auto",
        memory_fraction=None, maxmem=None):
    """Inputs: genotypes, components, chromosome, method, sample_ids, fit_mask, min_maf, block_size, output_dtype, blas_threads, progress, dosage_compensation, n_autosomes, engine.
    
    Candidate methods npplink/gcta0 use bounded packed float32 ssyrk.
    npplink uses sample SD (ddof=1), component-specific pair counts and
    n_ij-1 normalization; invalid pair denominators produce NaN. maxmem
    reserves numerical matrices and scratch, with metadata/library/I/O RSS
    additional. blas_threads is scoped; oversized explicit blocks are capped.
    See docs/GRM_API_2_11_7.md for the full scientific/resource contract.
    
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        genotypes : object
            PLINK1 BED prefix, a VIGOR Zarr genotype store path (``write_genotype_zarr``), or an
            ``xarray.Dataset``/``DataArray`` with ``sample``/``variant`` (or ``iid``/``snp``)
            dimensions such as returned by ``open_genotype_dataset``. Zarr/xarray inputs are read
            lazily in bounded variant blocks through the ``build_grms`` block reader (float-BLAS
            engine); Zarr v3 StringDType identifiers are coerced to Python strings. Optional
            ``fid``/``sex`` variables on ``sample`` supply FID and sex (defaults: IID and ``'0'``).
        components : object
            Requested genetic components, additive and/or dominance, estimated with component-specific valid denominator weights. Default: ``('additive', 'dominance')``.
        chromosome : object
            Chromosome label matched literally against BIM variant coordinates, without implicit reordering of sample IDs. Default: ``None``.
        method : object
            GRM normalization choice: npplink is empirical standardized with pairwise missingness; gcta0 per-marker scaled; van VanRaden I global; vitezica legacy A and D. Default: ``'npplink'``.
        sample_ids : object
            Ordered unique PLINK FAM individual identifiers to include in the resulting kernel row and column axes. Default: ``None``.
        fit_mask : object
            Boolean mask identifying allele-frequency estimation animals, avoiding use of excluded animals for frequency fitting. Default: ``None``.
        min_maf : object
            Minor allele frequency threshold applied to chromosome dosage calls before kernel accumulation. Default: ``0.0001``.
        block_size : object
            Bounded count of packed BED variant calls loaded and decoded per processing block of animals; 'auto' sizes it from the memory budget, integers are ceilings. Default: ``'auto'``.
        output_dtype : object
            NumPy floating-point output dtype controlling the numerical resolution and storage of GRM kernel values. Default: ``np.float32``.
        blas_threads : object
            Positive BLAS worker-thread limit used for small matrix and dense block multiplications; 'auto' uses every CPU available to the process. Default: ``'auto'``.
        progress : object
            Whether to show an optional chromosome or variant processing progress indicator during the fit. Default: ``False``.
        dosage_compensation : object
            Enable conservative inferred male-X recoding only when haploid 0/1 encoding can be established. Default: ``False``.
        n_autosomes : object
            Count of labeled autosomes before the rat X, Y, XY and mitochondrial chromosome conventions. Default: ``20``.
        engine : object
            Select packed bitserial additive or bounded float-BLAS backend, with explicit unsupported component checks. Default: ``'auto'``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    
    Parameters
    ----------
    memory_fraction : float or None
        Fraction of currently available RAM used when maxmem is absent, strictly above zero and at most 0.95; dense numerical matrices and scratch share the budget. Default: ``None`` (80% of physical RAM).
    maxmem : int, unit string or None
        Incremental numerical working-memory ceiling in bytes or binary units such as 8GiB; None uses memory_fraction. Metadata, mapped pages and library/I/O RSS are additional. Default: ``None``.
    
    """
    method,components=validate_grm_method(method,components)
    if chromosome is None:
        raise ValueError("grm requires chromosome; use GRMs for all chromosomes")
    kernels, weights, diag, ids, fid, sexes, dosage=_grm_payload(
        genotypes,chromosome,sample_ids=sample_ids,components=components,
        method=method,fit_mask=fit_mask,min_maf=min_maf,block_size=block_size,
        output_dtype=output_dtype,blas_threads=blas_threads,
        dosage_compensation=dosage_compensation,n_autosomes=n_autosomes,engine=engine,
        progress=progress,memory_fraction=memory_fraction,maxmem=maxmem)
    chrom=str(chromosome)
    n=len(ids)
    coords={"chromosome":np.asarray([chrom],dtype="<U16"),
            "sample_i":np.asarray(ids,dtype=str),
            "sample_j":np.asarray(ids,dtype=str)}
    variables={}
    for c in components:
        variables[c]=(("chromosome","sample_i","sample_j"),
                      np.asarray(kernels[c],dtype=output_dtype)[None,:,:])
        w=weights[c]
        if np.isscalar(w):
            variables["weight_"+c]=(("chromosome",),np.asarray([w],np.float64))
        else:
            variables["weight_"+c]=(("chromosome","sample_i","sample_j"),
                                      np.asarray(w)[None,:,:])
    if method in {'npplink', 'gcta0'}:
        for c in components:
            variables['numerator_'+c] = (("chromosome", "sample_i", "sample_j"),
                                         diag['numerators'][c][None, :, :])
            variables['pairwise_observed_counts_'+c] = (("chromosome", "sample_i", "sample_j"),
                diag['pairwise_observed_counts_by_component'][c][None, :, :])
        variables['pairwise_observed_counts'] = variables['pairwise_observed_counts_'+components[0]]
    variables["n_variants"]=(("chromosome",),np.array([diag["n_variants_used"]],np.int64))
    variables["autosome"]=(("chromosome",),np.array([_chromosome_class(chrom,n_autosomes)=="autosome"],bool))
    variables["dosage_status"]=(("chromosome",),np.asarray([dosage],dtype="<U48"))
    ds=xr.Dataset(variables,coords=coords)
    ds=ds.assign_coords(fid=("sample_i",np.asarray(fid,dtype=str)),
                        sex=("sample_i",np.asarray(sexes,dtype=str)))
    location,source_type=_source_description(genotypes)
    ds.attrs.update({"grm_schema":GRM_SCHEMA_VERSION,"grm_method":method,"normalization":diag["normalization"] if "normalization" in diag else "global_HWE",
                     "n_variants_total":int(diag["n_variants_total"]),
                     "n_variants_used":int(diag["n_variants_used"]),
                     "chromosome_type":_chromosome_class(chrom,n_autosomes),
                     "source_prefix":location,
                     "genotype_source_type":source_type,
                     "missingness_policy":diag.get("missingness","mean_imputed"),
                     "dosage_compensation_status":dosage,
                     "n_autosomes":int(n_autosomes)})
    return ds


def _encoding(ds, chunks=256, shard_factor=2):
    """Inputs: ds, chunks, shard_factor.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        ds : object
            Scientifically typed argument controlling validated genotype data handling and reproducible output behavior.
        chunks : object
            Number of animals along each square kernel chunk used to create compressed Zarr version three arrays. Default: ``256``.
        shard_factor : object
            Count of numerical chunks combined along each sample dimension into a physical Zarr version three shard. Default: ``2``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    sizes=ds.sizes;n=sizes.get("sample_i",1)
    c=min(int(chunks),n);sh=min(int(shard_factor)*c,n)
    enc={}
    for k,var in ds.data_vars.items():
        if len(var.dims)==3 and var.dims==("chromosome","sample_i","sample_j"):
            enc[k]={"chunks":(1,c,c),"shards":(1,sh,sh)}
    return enc


def GRMs(genotypes, output_file="grm/grm.zarr", dosage_compensation=True,
         components=("additive","dominance"), method="npplink", *,
         sample_ids=None, fit_mask=None, min_maf=1e-4,
         block_size='auto', output_dtype=np.float32, blas_threads='auto',
         n_autosomes=20, chunks=256, shard_factor=2, progress=True,
         overwrite=False, sex_embedding=False, engine="auto",
         memory_fraction=None, maxmem=None):
    """Inputs: genotypes, output_file, dosage_compensation, components, method, sample_ids, fit_mask, min_maf, block_size, output_dtype, blas_threads, n_autosomes, chunks, shard_factor, progress, overwrite, sex_embedding, engine.
    
    Candidate methods npplink/gcta0 use bounded packed float32 ssyrk.
    npplink uses sample SD (ddof=1), component-specific pair counts and
    n_ij-1 normalization; invalid pair denominators produce NaN. maxmem
    reserves numerical matrices and scratch, with metadata/library/I/O RSS
    additional. blas_threads is scoped; oversized explicit blocks are capped.
    See docs/GRM_API_2_11_7.md for the full scientific/resource contract.
    
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        genotypes : object
            PLINK1 BED prefix, a VIGOR Zarr genotype store path or an ``xarray.Dataset``/``DataArray``
            genotype object (see ``grm``). Zarr/xarray inputs are opened once, lazily, and every
            chromosome is accumulated from bounded blocks via the ``build_grms`` block reader.
        output_file : object
            Filesystem path of the Zarr version three chromosome archive, with overwrite safety enforced. Default: ``'grm/grm.zarr'``.
        dosage_compensation : object
            Enable conservative inferred male-X recoding only when haploid 0/1 encoding can be established. Default: ``True``.
        components : object
            Requested genetic components, additive and/or dominance, estimated with component-specific valid denominator weights. Default: ``('additive', 'dominance')``.
        method : object
            GRM normalization choice: npplink is empirical standardized with pairwise missingness; gcta0 per-marker scaled; van VanRaden I global; vitezica legacy A and D. Default: ``'npplink'``.
        sample_ids : object
            Ordered unique PLINK FAM individual identifiers to include in the resulting kernel row and column axes. Default: ``None``.
        fit_mask : object
            Boolean mask identifying allele-frequency estimation animals, avoiding use of excluded animals for frequency fitting. Default: ``None``.
        min_maf : object
            Minor allele frequency threshold applied to chromosome dosage calls before kernel accumulation. Default: ``0.0001``.
        block_size : object
            Bounded count of packed BED variant calls loaded and decoded per processing block of animals; 'auto' sizes it from the memory budget, integers are ceilings. Default: ``'auto'``.
        output_dtype : object
            NumPy floating-point output dtype controlling the numerical resolution and storage of GRM kernel values. Default: ``np.float32``.
        blas_threads : object
            Positive BLAS worker-thread limit used for small matrix and dense block multiplications; 'auto' uses every CPU available to the process. Default: ``'auto'``.
        n_autosomes : object
            Count of labeled autosomes before the rat X, Y, XY and mitochondrial chromosome conventions. Default: ``20``.
        chunks : object
            Number of animals along each square kernel chunk used to create compressed Zarr version three arrays. Default: ``256``.
        shard_factor : object
            Count of numerical chunks combined along each sample dimension into a physical Zarr version three shard. Default: ``2``.
        progress : object
            Whether to show an optional chromosome or variant processing progress indicator during the fit. Default: ``True``.
        overwrite : object
            Permit deliberate replacement of an existing complete destination only after a new staged store validates. Default: ``False``.
        sex_embedding : object
            Optional UMAP and density cluster inference on sex-chromosome GRM distances, not raw GRM similarities. Default: ``False``.
        engine : object
            Select packed bitserial additive or bounded float-BLAS backend, with explicit unsupported component checks. Default: ``'auto'``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    
    Parameters
    ----------
    memory_fraction : float or None
        Fraction of currently available RAM used when maxmem is absent, strictly above zero and at most 0.95; dense numerical matrices and scratch share the budget. Default: ``None`` (80% of physical RAM).
    maxmem : int, unit string or None
        Incremental numerical working-memory ceiling in bytes or binary units such as 8GiB; None uses memory_fraction. Metadata, mapped pages and library/I/O RSS are additional. Default: ``None``.
    
    """
    import zarr
    method,components=validate_grm_method(method,components)
    out=Path(output_file)
    if out.exists() and not overwrite:raise FileExistsError(f"{out} exists; use overwrite=True")
    xsrc,owned=_open_xarray_genotypes(genotypes) if _is_xarray_source(genotypes) else (None,False)
    try:
        return _GRMs_impl(genotypes,xsrc,out,zarr,dosage_compensation=dosage_compensation,
            components=components,method=method,sample_ids=sample_ids,fit_mask=fit_mask,
            min_maf=min_maf,block_size=block_size,output_dtype=output_dtype,blas_threads=blas_threads,
            n_autosomes=n_autosomes,chunks=chunks,shard_factor=shard_factor,progress=progress,
            overwrite=overwrite,sex_embedding=sex_embedding,engine=engine,
            memory_fraction=memory_fraction,maxmem=maxmem)
    finally:
        if owned and hasattr(xsrc,"close"):xsrc.close()


def _GRMs_impl(genotypes, xsrc, out, zarr, *, dosage_compensation, components, method, sample_ids,
               fit_mask, min_maf, block_size, output_dtype, blas_threads, n_autosomes, chunks,
               shard_factor, progress, overwrite, sex_embedding, engine, memory_fraction, maxmem):
    """Body of ``GRMs``; ``xsrc`` is the opened lazy Zarr/xarray source or None for PLINK1 input."""
    if xsrc is None:
        prefix=_prefix(genotypes)
        from .io import PlinkReader
        with PlinkReader(prefix,dtype=np.float32) as reader:
            chrom_col=reader.bim.chrom.astype(str)
    else:
        prefix=xsrc  # grm() recognises the opened source and reuses it chromosome by chromosome
        chrom_col=pd.Series(_object_strings(xsrc.variant_metadata.chromosome)).astype(str)
    location,source_type=_source_description(genotypes)
    available=chrom_col.unique().tolist()
    autosomes=[c for c in available if _chromosome_class(c,n_autosomes)=='autosome']
    if method in {'npplink','gcta0'} and int(chrom_col.isin(autosomes).sum()) > 2**24:
        raise ValueError('pairwise float32 counts support at most 2**24 autosomal variants')
    chroms=sorted(available,key=lambda x:
        (0,int(x)) if _chromosome_class(x,n_autosomes)=="autosome" else
        (1,{"X":0,"Y":1,"XY":2,"MT":3}.get(_chromosome_class(x,n_autosomes),99),str(x)))
    if not chroms:raise ValueError("source contains no chromosomes")
    warn=("Y/MT UMAP/HDBSCAN is optional and skipped when not requested, "
          "not supported by installed dependencies, or n < 1000 / n > 5000")
    # An atomic swap prevents exposing a partial run as completed output.
    # Keep parent path under the user's explicit destination.
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=out.parent/(out.name+".incomplete")
    if stage.exists(): raise FileExistsError(f"incomplete staging path {stage}; inspect before retry")
    try:
        for i,chrom in enumerate(chroms):
            ds=grm(prefix,chromosome=chrom,components=components,method=method,
                sample_ids=sample_ids,fit_mask=fit_mask,min_maf=min_maf,
                block_size=block_size,output_dtype=output_dtype,
                blas_threads=blas_threads,dosage_compensation=dosage_compensation,
                n_autosomes=n_autosomes,engine=engine,progress=progress,
                memory_fraction=memory_fraction,maxmem=maxmem)
            n_rows=ds.sizes["sample_i"]
            special=_chromosome_class(chrom,n_autosomes) in ("Y","MT")
            if bool(sex_embedding) and special:
                u1,u2,clusters,emb_status=_sex_chromosome_embedding(
                    ds["additive"].isel(chromosome=0).values
                    if "additive" in ds else ds["dominance"].isel(chromosome=0).values)
            else:
                u1=np.full(n_rows,np.nan,dtype=np.float32)
                u2=np.full(n_rows,np.nan,dtype=np.float32)
                clusters=np.full(n_rows,-1,dtype=np.int32)
                emb_status="not_requested" if not sex_embedding else "not_sex_chromosome"
            ds["umap1"]=(("chromosome","sample_i"),u1[None,:])
            ds["umap2"]=(("chromosome","sample_i"),u2[None,:])
            ds["hdbscan"]=(("chromosome","sample_i"),clusters[None,:])
            ds["embedding_status"]=(("chromosome",),
                                    np.asarray([emb_status],dtype="<U64"))
            # Only static attributes persist on the root; chromosome QC lives in variables.
            ds.attrs={"grm_schema":GRM_SCHEMA_VERSION,"grm_complete":False,
                      "grm_method":method,"genotype_source_type":source_type,
                      "source_prefix":location,
                      "n_autosomes":n_autosomes,"zarr_format":3,
                      "aggregation":"sum(raw_numerator_chr) / component_denominator_from_summed_counts" if method in {"npplink","gcta0"} else "sum(K_chr * weight_chr) / sum(weight_chr)",
                      "dosage_rule":"male_X_haploid_01_double_only_if_confident; otherwise skip",
                      "sex_embedding_note":warn}
            if i==0:
                ds.to_zarr(stage,mode="w",zarr_format=3,consolidated=False,
                           encoding=_encoding(ds,chunks,shard_factor))
            else:
                ds.to_zarr(stage,mode="a",append_dim="chromosome",
                           zarr_format=3,consolidated=False)
            ds.close()
            del ds  # Release eager n² statistics before allocating the next chromosome.
        all_ds=xr.open_zarr(stage,zarr_format=3,consolidated=False)
        include=all_ds["autosome"].values.astype(bool)
        if not include.any():raise ValueError("no autosomal chromosomes; cannot build All GRM")
        aggregate={}
        selected=all_ds.isel(chromosome=np.flatnonzero(include))
        for comp in components:
            ww=selected["weight_"+comp]
            if method in {'npplink', 'gcta0'}:
                numerator=selected['numerator_'+comp].sum(dim='chromosome', skipna=False)
                counts=selected['pairwise_observed_counts_'+comp].sum(dim='chromosome')
                denominator=xr.where(counts>1,counts-1,0.) if method=='npplink' else counts
                new=(numerator/denominator.where(denominator>0)).where(denominator>0)
                aggregate['numerator_'+comp]=numerator.expand_dims(chromosome=['All'])
                aggregate['pairwise_observed_counts_'+comp]=counts.expand_dims(chromosome=['All'])
            else:
                numerator=(selected[comp].astype(np.float64)*ww).sum(dim='chromosome')
                denominator=ww.sum(dim='chromosome')
                new=xr.where(denominator>0,numerator/denominator,0.)
            aggregate[comp]=new.astype(output_dtype).expand_dims(chromosome=np.asarray(['All'],dtype='<U16'))
            aggregate['weight_'+comp]=denominator.expand_dims(chromosome=np.asarray(['All'],dtype='<U16'))
        if method in {'npplink', 'gcta0'}:
            aggregate['pairwise_observed_counts']=aggregate['pairwise_observed_counts_'+components[0]]
        agg=xr.Dataset(aggregate)
        agg["n_variants"]=xr.DataArray([int(selected.n_variants.sum())],dims=("chromosome",),coords={"chromosome":["All"]})
        agg["autosome"]=xr.DataArray([False],dims=("chromosome",),coords={"chromosome":["All"]})
        agg["dosage_status"]=xr.DataArray(np.asarray(["weighted_autosomal_total"],dtype="<U48"),
                                         dims=("chromosome",),coords={"chromosome":["All"]})
        agg=agg.assign_coords(sample_i=all_ds.sample_i.values,sample_j=all_ds.sample_j.values,
                              fid=("sample_i",all_ds.fid.values),sex=("sample_i",all_ds.sex.values))
        agg=agg.assign_coords(chromosome=np.asarray(["All"],dtype="<U16"))
        # Aggregating via Dask changes its chunk boundaries.  Explicitly align
        # to Zarr shard-aligned blocks to prevent overlapping parallel writes.
        ds_chunk=min(int(chunks)*int(shard_factor),len(all_ds.sample_i))
        agg=agg.chunk({"chromosome":1,"sample_i":ds_chunk,"sample_j":ds_chunk})
        n_rows=all_ds.sizes["sample_i"]
        agg["umap1"]=(("chromosome","sample_i"),np.full((1,n_rows),np.nan,np.float32))
        agg["umap2"]=(("chromosome","sample_i"),np.full((1,n_rows),np.nan,np.float32))
        agg["hdbscan"]=(("chromosome","sample_i"),np.full((1,n_rows),-1,np.int32))
        agg["embedding_status"]=(("chromosome",),
                                  np.asarray(["not_applicable_autosomal_total"],dtype="<U64"))
        agg.attrs = dict(all_ds.attrs)
        agg.to_zarr(stage,mode="a",append_dim="chromosome",zarr_format=3,consolidated=False,align_chunks=True)
        all_ds.close()
        zarr.open_group(str(stage),mode="a").attrs["grm_complete"] = True
        zarr.consolidate_metadata(str(stage))
        reopened=xr.open_zarr(stage,zarr_format=3,consolidated=True)
        assert "All" in reopened.chromosome.values and reopened.attrs["grm_method"]==method
        reopened.close()
        if out.exists():
            if overwrite:
                import shutil
                shutil.rmtree(out)
            else:raise FileExistsError(out)
        stage.rename(out)
    except Exception:
        # Preserve incomplete data for diagnostic inspection. Never silently
        # expose the staged path as the completed requested GRM.
        raise
    ds = xr.open_zarr(out,zarr_format=3,consolidated=True)
    # Convenience projections: retain the complete chromosome stack including All.
    for name in components:
        ds[f"all_{name}"] = ds[name].sel(chromosome="All", drop=True)
    return ds
