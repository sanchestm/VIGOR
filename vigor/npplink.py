import numpy as np
try:
    import dask.array as da
    import dask
    import dask.dataframe as dd
except ImportError:
    da = dask = dd = None
try:
    import xarray as xr
except ImportError:
    xr = None
import pandas as pd
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    class _NumbaFallback:
        """_NumbaFallback represents numba fallback.
        """
        @staticmethod
        def njit(*args, **kwargs):
            """Provide a no-op Numba decorator when the optional compiler is unavailable.

            Parameters
            ----------
            *args : tuple
                Additional positional arguments supplied by the enclosing callback or context protocol;
                their order and types follow that protocol.

            **kwargs : mapping
                Additional keyword arguments consumed by the compatibility/callback logic in this body.
                See the explicit keyword lookups below for accepted keys; unused keys are subject to
                that compatibility policy.

            Returns
            -------
            object
                Computed result from ``deco``; see the summary and the called implementation for its labeled/numerical representation.
            """
            def deco(func): 
                """Return the original function unchanged in the no-Numba fallback.

                Parameters
                ----------
                func : callable or None
                    Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
                    order shown by its call below and return the representation consumed there; None
                    disables an optional callback or uses the default implementation.

                Returns
                -------
                object
                    Computed result from ``func``; see the summary and the called implementation for its labeled/numerical representation.
                """
                return func
            return deco
        @staticmethod
        def prange(*args): 
            """Return Python range iteration as the fallback for Numba parallel range.

            Parameters
            ----------
            *args : tuple
                Additional positional arguments forwarded, in order, to ``range``. See that signature
                for required scientific inputs and their order.

            Returns
            -------
            object
                Computed result from ``range(*args)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return range(*args)
    numba = _NumbaFallback()
    njit = numba.njit
    prange = numba.prange
from sklearn.utils.extmath import randomized_svd
from scipy.stats import t as scipyt
from scipy.stats import chi2
from scipy.special import stdtr, erfc
from tqdm import tqdm
tqdm.pandas()
import itertools
import json
from typing import Literal
from scipy.linalg import blas
import os
from collections import defaultdict


# try:
#     import cupy as np
#     _ = np.cuda.runtime.getDeviceCount() > 0
#     GPU_AVAILABLE = True
#     print('running npplink on the GPU')
# except Exception:
#     import numpy as np
#     GPU_AVAILABLE = False

def _require_xarray():
    """Import and verify availability of xarray.
    """
    if xr is None:
        raise ImportError("xarray is required for this operation; install VIGOR[xarray]")
    return xr

def _require_dask():
    """Import and verify availability of dask.
    """
    if da is None or dask is None:
        raise ImportError("dask is required for lazy PLINK loading; install VIGOR[scale]")

def read_fam_bim(prefix):
    """Read authoritative PLINK1 metadata or a size/mtime-validated sidecar cache.
    
    Parameters
    ----------
    prefix : str or pathlib.Path
        PLINK1 prefix without suffix; .fam/.bim must exist. Optional .fam.parquet/.bim.parquet/.plinkmeta.json are used only when recorded text sizes and nanosecond mtimes match.
    
    Returns
    -------
    tuple of pandas.DataFrame
        fam has n rows and fid/iid/father/mother/gender/trait; bim has m rows and chrom/snp/cm/pos/a0/a1/i. File order and string animal/variant IDs are retained. Duplicate IID raises ValueError.
    
    Complexity
    ----------
    TIME O(B+n+m); SPACE O(n+m), where B is metadata bytes parsed/decompressed, n animals and m variants. Both full tables are materialized.
    
    Notes
    -----
    Cache validation uses size/mtime, not cryptographic content identity; immutable scientific provenance needs separate hashes. This function never writes sidecars or reads BED.
    """
    from .io import read_fam, read_bim
    fam = read_fam(prefix).rename(columns={"sex": "gender", "pheno": "trait"})
    bim = read_bim(prefix)
    bim["i"] = np.arange(len(bim), dtype=np.int64)
    return fam, bim


def write_plink_metadata_cache(prefix):
    """Create Parquet sidecars from PLINK1 text metadata and record validation stamps.
    
    Parameters
    ----------
    prefix : str or pathlib.Path
        PLINK1 prefix with existing .fam/.bim. Writes/replaces adjacent .fam.parquet, .bim.parquet and .plinkmeta.json; use only where such writes are authorized.
    
    Returns
    -------
    dict
        Text-file sizes/mtime_ns and row counts used by read_fam_bim to validate caches.
    
    Complexity
    ----------
    TIME O(B+n+m); SPACE O(n+m+c), for B input/output bytes, n animals, m markers and codec buffer c. Full text tables are loaded; a Parquet engine is required.
    
    Notes
    -----
    Cache creation does not validate IID uniqueness. Never call on protected read-only research inputs during candidate validation.
    """
    import json as _json
    from pathlib import Path as _Path
    pre=str(prefix);ff=_Path(pre+'.fam');bf=_Path(pre+'.bim')
    # Bypass the cache on creation to make the text PLINK metadata authoritative.
    fam=pd.read_csv(ff,sep=r'\s+',header=None,low_memory=True,engine='c',names=['fid','iid','father','mother','gender','trait'],dtype={'fid':str,'iid':str,'father':str,'mother':str})
    bim=pd.read_csv(bf,sep=r'\s+',header=None,low_memory=True,engine='c',names=['chrom','snp','cm','pos','a0','a1'],dtype={'chrom':str,'snp':str,'cm':float,'pos':int,'a0':str,'a1':str});bim['i']=bim.index.copy()
    fam.to_parquet(pre+'.fam.parquet',index=False);bim.to_parquet(pre+'.bim.parquet',index=False)
    fs=ff.stat();bs=bf.stat();meta={'fam_size':fs.st_size,'bim_size':bs.st_size,'fam_mtime_ns':fs.st_mtime_ns,'bim_mtime_ns':bs.st_mtime_ns,'rows_fam':len(fam),'rows_bim':len(bim)}
    _Path(pre+'.plinkmeta.json').write_text(_json.dumps(meta,indent=2));return meta




def memmap_bed_file(bed_file):
    """Open a read-only byte mapping of a BED file including its three-byte header.
    
    Parameters
    ----------
    bed_file : str or pathlib.Path
        Existing PLINK1 BED filename with matching BIM/FAM. The shared opener validates magic, SNP-major mode and exact payload geometry.
    
    Returns
    -------
    numpy.memmap, shape (b,), dtype uint8
        Read-only entire-file byte view; no calls are decoded.
    
    Complexity
    ----------
    TIME O(1) mapping setup; SPACE O(1) Python storage and O(b) virtual address mapping for b file bytes. Resident pages depend on later access.
    """
    from .io import open_bed_memmap
    return open_bed_memmap(bed_file, include_header=True)

def decode_bed_chunk(buffer, num_samples, variants_per_chunk, dtype=np.float32):
    """Decode a consecutive SNP-major PLINK1 variant block.
    
    Parameters
    ----------
    buffer : 1D uint8 ndarray or numpy.memmap
        Header-free SNP-major bytes for v variants, length v*ceil(n/4).
    
    num_samples : int
        Total source FAM animal count n, not the subset size. Must agree with BED stride.
    
    variants_per_chunk : int
        Number v of consecutive variants to decode, nonnegative and compatible with the byte buffer.
    
    dtype : NumPy floating dtype
        Default: ``np.float32``. Output/lookup precision; a floating type is required to represent missing calls as NaN.
    
    Returns
    -------
    numpy.ndarray, shape (n, v)
        Dosage codes 0/1/2, with BED missing code mapped to NaN. Rows preserve FAM order and columns BED variant order; no allele reorientation.
    
    Complexity
    ----------
    TIME O(nv); SPACE O(nv) beyond the source mapping. n is source animals, s selected animals and v decoded variants. Mapped pages may add resident memory; no whole BED copy is made.
    
    Notes
    -----
    The low-level decoder assumes SNP-major geometry. The caller validates magic bytes, variant bounds, and which BIM allele each dosage counts.
    """
    # Use the precomputed lookup table: each byte becomes 4 genotype calls.
    from .io import decode_bed
    v = int(variants_per_chunk)
    if v < 0 or v != variants_per_chunk:
        raise ValueError("variants_per_chunk must be a nonnegative integer")
    P = np.asarray(buffer)
    bpv = (int(num_samples) + 3) // 4
    if P.size != v * bpv:
        raise ValueError("BED chunk byte count does not match requested variants")
    return decode_bed(P.reshape(v, bpv), num_samples, dtype=dtype)

def read_bed_chunk(bed_mem, num_samples, variants_chunk_start, variants_per_chunk, dtype=np.float32):
    """Read and decode a consecutive SNP-major PLINK1 variant block.
    
    Parameters
    ----------
    bed_mem : 1D uint8 ndarray or numpy.memmap
        Whole SNP-major BED bytes including the three-byte header.
    
    num_samples : int
        Total source FAM animal count n, not the subset size. Must agree with BED stride.
    
    variants_chunk_start : int
        Zero-based first variant index in the BED; caller must ensure the complete block is within file bounds.
    
    variants_per_chunk : int
        Number v of consecutive variants to decode, nonnegative and compatible with the byte buffer.
    
    dtype : NumPy floating dtype
        Default: ``np.float32``. Output/lookup precision; a floating type is required to represent missing calls as NaN.
    
    Returns
    -------
    numpy.ndarray, shape (n, v)
        Dosage codes 0/1/2, with BED missing code mapped to NaN. Rows preserve FAM order and columns BED variant order; no allele reorientation.
    
    Complexity
    ----------
    TIME O(nv); SPACE O(nv) beyond the source mapping. n is source animals, s selected animals and v decoded variants. Mapped pages may add resident memory; no whole BED copy is made.
    
    Notes
    -----
    The low-level decoder assumes SNP-major geometry. The caller validates magic bytes, variant bounds, and which BIM allele each dosage counts.
    """
    from .io import BED_MAGIC
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    if not isinstance(variants_chunk_start, (int, np.integer)) or variants_chunk_start < 0:
        raise ValueError("variants_chunk_start must be a nonnegative integer")
    if not isinstance(variants_per_chunk, (int, np.integer)) or variants_per_chunk < 0:
        raise ValueError("variants_per_chunk must be a nonnegative integer")
    if bytes(np.asarray(bed_mem[:3], dtype=np.uint8)) != BED_MAGIC:
        raise ValueError("expected SNP-major PLINK1 BED header")
    bpv = (num_samples+3)//4
    offset = 3+variants_chunk_start*bpv
    size = variants_per_chunk*bpv
    if offset+size > len(bed_mem):
        raise IndexError("requested BED chunk exceeds payload")
    buffer = bed_mem[offset:offset+size]
    return decode_bed_chunk(buffer, num_samples, variants_per_chunk, dtype=dtype)

def decode_bed_chunk_subset(buffer, num_samples, variants_per_chunk, sample_indices, dtype=np.float32):
    """Decode selected animals from a consecutive SNP-major PLINK1 variant block.
    
    Parameters
    ----------
    buffer : 1D uint8 ndarray or numpy.memmap
        Header-free SNP-major bytes for v variants, length v*ceil(n/4).
    
    num_samples : int
        Total source FAM animal count n, not the subset size. Must agree with BED stride.
    
    variants_per_chunk : int
        Number v of consecutive variants to decode, nonnegative and compatible with the byte buffer.
    
    sample_indices : 1D integer array, length s
        Zero-based FAM positions in requested output order; None is unsupported. Duplicates are retained, empty selection gives shape (0,v); out-of-range positions raise IndexError.
    
    dtype : NumPy floating dtype
        Default: ``np.float32``. Output/lookup precision; a floating type is required to represent missing calls as NaN.
    
    Returns
    -------
    numpy.ndarray, shape (s, v)
        Dosage codes 0/1/2, with BED missing code mapped to NaN. Rows preserve sample_indices order and columns BED variant order; no allele reorientation.
    
    Complexity
    ----------
    TIME O(sv+s); SPACE O(sv+s) beyond the source mapping. n is source animals, s selected animals and v decoded variants. Mapped pages may add resident memory; no whole BED copy is made.
    
    Notes
    -----
    The low-level decoder assumes SNP-major geometry. The caller validates magic bytes, variant bounds, and which BIM allele each dosage counts.
    """
    from .io import decode_bed
    v = int(variants_per_chunk)
    if v < 0 or v != variants_per_chunk:
        raise ValueError("variants_per_chunk must be a nonnegative integer")
    P = np.asarray(buffer)
    bpv = (int(num_samples) + 3) // 4
    if P.size != v * bpv:
        raise ValueError("BED chunk byte count does not match requested variants")
    if sample_indices is None:
        raise TypeError("sample_indices is required for the subset decoder")
    return decode_bed(P.reshape(v, bpv), num_samples, sample_indices, dtype=dtype)


def read_bed_chunk_subset(bed_mem,num_samples,variants_chunk_start,variants_per_chunk,sample_indices,dtype=np.float32):
    """Read and decode selected animals from a consecutive SNP-major PLINK1 variant block.
    
    Parameters
    ----------
    bed_mem : 1D uint8 ndarray or numpy.memmap
        Whole SNP-major BED bytes including the three-byte header.
    
    num_samples : int
        Total source FAM animal count n, not the subset size. Must agree with BED stride.
    
    variants_chunk_start : int
        Zero-based first variant index in the BED; caller must ensure the complete block is within file bounds.
    
    variants_per_chunk : int
        Number v of consecutive variants to decode, nonnegative and compatible with the byte buffer.
    
    sample_indices : 1D integer array, length s
        Zero-based FAM positions in requested output order; None is unsupported. Duplicates are retained, empty selection gives shape (0,v); out-of-range positions raise IndexError.
    
    dtype : NumPy floating dtype
        Default: ``np.float32``. Output/lookup precision; a floating type is required to represent missing calls as NaN.
    
    Returns
    -------
    numpy.ndarray, shape (s, v)
        Dosage codes 0/1/2, with BED missing code mapped to NaN. Rows preserve sample_indices order and columns BED variant order; no allele reorientation.
    
    Complexity
    ----------
    TIME O(sv+s); SPACE O(sv+s) beyond the source mapping. n is source animals, s selected animals and v decoded variants. Mapped pages may add resident memory; no whole BED copy is made.
    
    Notes
    -----
    The low-level decoder assumes SNP-major geometry. The caller validates magic bytes, variant bounds, and which BIM allele each dosage counts.
    """
    from .io import BED_MAGIC
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    if not isinstance(variants_chunk_start, (int, np.integer)) or variants_chunk_start < 0:
        raise ValueError("variants_chunk_start must be a nonnegative integer")
    if not isinstance(variants_per_chunk, (int, np.integer)) or variants_per_chunk < 0:
        raise ValueError("variants_per_chunk must be a nonnegative integer")
    if bytes(np.asarray(bed_mem[:3], dtype=np.uint8)) != BED_MAGIC:
        raise ValueError("expected SNP-major PLINK1 BED header")
    bpv = (num_samples+3)//4
    offset = 3+variants_chunk_start*bpv
    size = variants_per_chunk*bpv
    if offset+size > len(bed_mem):
        raise IndexError("requested BED chunk exceeds payload")
    buffer = bed_mem[offset:offset+size]
    return decode_bed_chunk_subset(buffer, num_samples, variants_per_chunk, sample_indices, dtype=dtype)


def _memory_bytes(value):
    """Parse a memory budget; numeric values are bytes, strings accept KB/MB/GB/TB.

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation ``value.lower() == 'auto'``. See ``vigor.npplink._memory_bytes``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.
    """
    from .resources import parse_memory_bytes
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    return parse_memory_bytes(value)


def available_memory_bytes():
    """Compute available memory bytes.

    Returns
    -------
    object
        Computed result from ``4 * 1024 ** 3``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .resources import available_memory_bytes as _available  # container-aware; one implementation
    return _available()


def plan_variant_chunk_size(n_samples, n_variants=None, *, dtype=np.float32, maxmem=None,
                            memory_fraction=None, working_copies=3.0, reserve_bytes=512*1024**2,
                            min_variants=512, align_variants=256):
    """Choose the largest genotype chunk that fits the working-memory budget (default 80% of RAM).

    ``working_copies`` accounts for decoded X, standardized Z and temporary
    masks/reductions.  ``maxmem`` can be used to deliberately increase/decrease
    the planner's ceiling; otherwise it tracks live available RAM.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM).

    working_copies : int
        Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
        extent as named. Allocation planners multiply these costs by the relevant dimensions;
        use nonnegative costs consistent with the actual representation. Default: ``3.0``.

    reserve_bytes : int
        Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
        extent as named. Allocation planners multiply these costs by the relevant dimensions;
        use nonnegative costs consistent with the actual representation. Default: ``512 * 1024
        ** 2``.

    min_variants : int
        Retained for API compatibility; the working-memory budget alone sizes the block, so no
        floor above what the budget affords is applied. Default: ``512``.

    align_variants : int
        Positive block-alignment quantum in variant columns. Affordable chunk sizes at least
        this large are rounded down to a multiple of this count. Default: ``256``.

    Returns
    -------
    object
        Computed result from ``int(raw)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .resources import bounded_variant_chunk, working_memory_bytes
    n=max(1,int(n_samples)); item=np.dtype(dtype).itemsize; mm=_memory_bytes(maxmem)
    # Reserve never eats more than half the budget, so small hosts still get a block.
    reserve=min(int(reserve_bytes),working_memory_bytes(mm,memory_fraction)//2)
    return bounded_variant_chunk(n,maxmem=mm,memory_fraction=memory_fraction,bytes_per_call=float(working_copies)*item,
                                 reserve_bytes=reserve,alignment=max(1,int(align_variants)),n_variants=n_variants)


def _resolve_chunk_variants(chunk_variants,n_samples,n_variants,dtype,maxmem=None,working_copies=3.0,memory_fraction=None):
    """Resolve chunk variants.

    Parameters
    ----------
    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Source-recognized literal
        values: ``auto``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    n_variants : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Passed to ``plan_variant_chunk_size`` as ``dtype``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to
        ``plan_variant_chunk_size`` as ``maxmem``.

    working_copies : int
        Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
        extent as named. Allocation planners multiply these costs by the relevant dimensions;
        use nonnegative costs consistent with the actual representation. Default: ``3.0``.
        Passed to ``plan_variant_chunk_size`` as ``working_copies``.

    memory_fraction : float or None
        RAM fraction when maxmem is None; None means 80%. Default: ``None``.

    Returns
    -------
    int
        'auto'/None sizes from the budget; an integer is a ceiling still clipped to the budget.
    """
    cv=plan_variant_chunk_size(n_samples,n_variants,dtype=dtype,maxmem=maxmem,memory_fraction=memory_fraction,working_copies=working_copies)
    if chunk_variants is None or (isinstance(chunk_variants,str) and chunk_variants.lower()=="auto"): return cv
    return max(1,min(int(chunk_variants),cv))


def load_plink(prefix, chunk_variants="auto", dtype=np.float32, use_memmap=True, sample_indices=None, maxmem=None):
    """Lazy PLINK BED reader with optional packed-byte sample subsetting.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``dask.delayed(fn)`` as
        ``dtype``.

    use_memmap : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Default:
        ``None``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to
        ``_resolve_chunk_variants`` as ``maxmem``.

    Returns
    -------
    object
        Computed result from ``(bim, fam_out, da.concatenate(delayed, axis=1))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    _require_dask(); fam,bim=read_fam_bim(prefix)
    bed_file=f'{prefix}.bed'; num_samples_full=len(fam); num_variants=len(bim)
    if sample_indices is None:
        si=None; fam_out=fam; nout=num_samples_full
    else:
        si=np.asarray(sample_indices,dtype=np.int64)
        if si.ndim!=1 or (len(si) and (si.min()<0 or si.max()>=num_samples_full)):
            raise IndexError('invalid sample_indices')
        fam_out=fam.iloc[si].reset_index(drop=True); nout=len(si)
    cv=_resolve_chunk_variants(chunk_variants,nout,num_variants,dtype,maxmem=maxmem,working_copies=2.5)
    from .io import open_bed_memmap
    validated = open_bed_memmap(prefix, n_samples=num_samples_full, n_variants=num_variants, include_header=True)
    bed_mem = validated if use_memmap else None
    if not use_memmap:
        validated._mmap.close()
    delayed=[]
    for offset in range(0,num_variants,cv):
        nv=min(cv,num_variants-offset)
        if use_memmap:
            fn=read_bed_chunk if si is None else read_bed_chunk_subset
            args=(bed_mem,num_samples_full,offset,nv)
            if si is None: chunk=dask.delayed(fn)(*args,dtype=dtype)
            else: chunk=dask.delayed(fn)(*args,si,dtype=dtype)
        else:
            def read_chunk(bed_file=bed_file,offset=offset,nv=nv,si=si):
                """Read chunk.

                Parameters
                ----------
                bed_file : str or os.PathLike
                    Input for the operation ``open(bed_file, 'rb')``. See
                    ``vigor.npplink.load_plink.read_chunk`` and its caller for the scientific
                    representation; the shown operation constrains the required type/shape. Default:
                    ``bed_file``.

                offset : object (see the cited input contract)
                    Input for the operation ``3 + offset * bpv``. See
                    ``vigor.npplink.load_plink.read_chunk`` and its caller for the scientific
                    representation; the shown operation constrains the required type/shape. Default:
                    ``offset``.

                nv : object (see the cited input contract)
                    Input for the operation ``nv * bpv``. See ``vigor.npplink.load_plink.read_chunk`` and
                    its caller for the scientific representation; the shown operation constrains the
                    required type/shape. Default: ``nv``.

                si : array-like or numeric scalar
                    Input for the operation ``decode_bed_chunk_subset(buf, num_samples_full, nv, si,
                    dtype=dtype)``. See ``vigor.npplink.load_plink.read_chunk`` and its caller for the
                    scientific representation; the shown operation constrains the required type/shape.
                    Default: ``si``.

                Returns
                -------
                object
                    Computed result from ``decode_bed_chunk(buf, num_samples_full, nv, dtype=dtype) if si is None else decode_bed_chunk_subset(buf, num_samples_full, nv, si, dtype=dtype)``; see the summary and the called implementation for its labeled/numerical representation.
                """
                bpv=(num_samples_full+3)//4; off=3+offset*bpv; nbyte=nv*bpv
                with open(bed_file,'rb') as f: f.seek(off); buf=np.frombuffer(f.read(nbyte),dtype=np.uint8)
                return decode_bed_chunk(buf,num_samples_full,nv,dtype=dtype) if si is None else decode_bed_chunk_subset(buf,num_samples_full,nv,si,dtype=dtype)
            chunk=dask.delayed(read_chunk)()
        delayed.append(da.from_delayed(chunk,shape=(nout,nv),dtype=dtype))
    return bim, fam_out, (da.concatenate(delayed, axis=1) if delayed else da.empty((nout, 0), dtype=dtype))


def load_plink_xarray(prefix, chunk_variants="auto", dtype=np.float32, use_memmap=True, sample_indices=None, sample_ids=None, maxmem=None):
    """Lazy xarray PLINK reader; sample filtering occurs during packed BED decoding.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    use_memmap : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Default:
        ``None``. Passed to ``load_plink`` as ``sample_indices``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``load_plink`` as
        ``maxmem``.

    Returns
    -------
    object
        Computed result from ``g``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if sample_ids is not None:
        if sample_indices is not None: raise ValueError('use sample_indices or sample_ids, not both')
        fam0,_=read_fam_bim(prefix); lookup=pd.Series(np.arange(len(fam0)),index=fam0['iid'].astype(str))
        wanted=np.asarray(sample_ids).astype(str); missing=np.setdiff1d(wanted,lookup.index.values)
        if len(missing): raise KeyError(f'{len(missing)} requested sample IDs are absent from FAM')
        sample_indices=lookup.loc[wanted].to_numpy(dtype=np.int64)
    bim,fam,genotype_dask=load_plink(prefix,chunk_variants,dtype,use_memmap,sample_indices=sample_indices,maxmem=maxmem)
    src_rows=np.arange(len(fam),dtype=np.int64) if sample_indices is None else np.asarray(sample_indices,dtype=np.int64)
    g=_require_xarray().DataArray(genotype_dask,dims=('iid','snp'),coords={
        'iid':fam['iid'].astype(str).to_numpy(),'snp':bim['snp'].astype(str).to_numpy(),
        'source_iid_index':('iid',src_rows),'source_variant_index':('snp',np.arange(len(bim),dtype=np.int64)),
        'sex':('iid',fam['gender'].to_numpy()),'fid':('iid',fam['fid'].astype(str).to_numpy()),
        'father':('iid',fam['father'].astype(str).to_numpy()),'mother':('iid',fam['mother'].astype(str).to_numpy()),
        'chrom':('snp',bim['chrom'].astype(str).to_numpy()),'pos':('snp',bim['pos'].to_numpy()),
        'a0':('snp',bim['a0'].astype(str).to_numpy()),'a1':('snp',bim['a1'].astype(str).to_numpy()),
        'other_allele':('snp',bim['a0'].astype(str).to_numpy()),
        'effect_allele':('snp',bim['a1'].astype(str).to_numpy())},name='genotypes')
    g.attrs.update({'plink_prefix':str(prefix),'plink_format':'bed','packed_source':True,
                    'chunk_variants':int(genotype_dask.chunks[1][0]) if getattr(genotype_dask,'chunks',None) else None,
                    'effect_allele_convention':'BIM_column_6_counted','genotype_missingness':'unknown','domino_reader':'npplink.load_plink_xarray'})
    return g


@numba.njit(parallel=True, fastmath=False)
def king_robust_kinship_numba(G):
    """Compute conservative between-family KING-robust kinship (Manichaikul Eq. 11).

    Parameters
    ----------
    G : numpy.ndarray, shape (n, m)
        Autosomal diploid hardcalls 0/1/2, with NaN for missing. The caller
        selects autosomes. Fractional dosages and infinite values are rejected.

    Returns
    -------
    numpy.ndarray, shape (n, n), dtype float64
        Symmetric kinship. Pairs with zero jointly observed heterozygosity are
        NaN; negative estimates are retained. Informative diagonal entries are
        0.5. The between-family formula is
        0.5 - sum((g_i-g_j)^2)/(4*min(H_i,H_j)), where H counts heterozygotes
        only among jointly called markers. This includes both het-het sharing
        and opposite homozygotes; it is not an IBS0-only statistic.

    Complexity
    ----------
    Time O(n^2*m), space O(n*m+n^2) for the NumPy path and O(n^2) beyond input
    for the compiled path.

    Notes
    -----
    Manichaikul et al. (2010), doi:10.1093/bioinformatics/btq559, Eq. 11.
    The within-family Eq. 9 is a different estimator; no family labels are
    inferred here. No inbreeding adjustment or clipping is applied.
    """
    if G.ndim != 2:
        raise ValueError("G must have shape (samples, markers)")
    n, m = G.shape
    for i in range(n):
        for s in range(m):
            a = G[i, s]
            if not np.isnan(a) and (not np.isfinite(a) or (a != 0 and a != 1 and a != 2)):
                raise ValueError("KING requires hardcalls 0/1/2 or NaN")
    kin = np.full((n, n), np.nan, dtype=np.float64)
    for i in numba.prange(n):
        for j in range(i, n):
            hi = 0
            hj = 0
            distance = 0.0
            for s in range(m):
                a = G[i, s]
                b = G[j, s]
                if np.isnan(a) or np.isnan(b):
                    continue
                if a == 1:
                    hi += 1
                if b == 1:
                    hj += 1
                distance += (a-b)*(a-b)
            denom = 4 * min(hi, hj)
            if denom > 0:
                value = 0.5 - distance / denom
                kin[i, j] = value
                kin[j, i] = value
    return kin


def genomic_kernel_operators(prefix, *, include=("additive",), **kwargs):
    """Modern PLINK-backed A/D GRM operators; no GRM file is generated.

    This is intentionally distinct from legacy :func:`GRM`, whose pairwise
    standardized normalization is preserved for backward compatibility.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    include : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default:
        ``('additive',)``. Passed to ``genomic_kernel_operators_from_plink1`` as ``include``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``genomic_kernel_operators_from_plink1``. See
        those callables for accepted keys and values. Passed to
        ``genomic_kernel_operators_from_plink1`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``genomic_kernel_operators_from_plink1(prefix, include=include, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .grm import genomic_kernel_operators_from_plink1
    return genomic_kernel_operators_from_plink1(prefix,include=include,**kwargs)


def genomic_grm_xarray(prefix, *, component="additive", include=None, max_compute_pairs=4_000_000, **kwargs):
    """Virtual GRM DataArray; select iid_i/iid_j before ``compute()``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``.

    include : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    max_compute_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``4000000``. Passed to ``bundle.xarray`` as
        ``max_compute_pairs``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``PlinkGenomicKernelBundle``. See those
        callables for accepted keys and values. Passed to ``PlinkGenomicKernelBundle`` as
        ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``bundle.xarray(component, max_compute_pairs=max_compute_pairs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .grm import PlinkGenomicKernelBundle
    inc=(component,) if include is None else include
    bundle=PlinkGenomicKernelBundle(prefix,include=inc,**kwargs)
    return bundle.xarray(component,max_compute_pairs=max_compute_pairs)



def GRM(X, scale = True, return_weights= False, nan_policy = 'ignore', correlation_matrix= False, savefile = None):
    """Compute genotype-derived relationship matrices with the selected coding and missingness normalization.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``.

    return_weights : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    nan_policy : str
        Only 'ignore' is retained: empirical sample-SD normalization with the
        common engine's pairwise n-minus-one denominator. Legacy 'mean' and
        'per_iid' modes raise; select an explicit modern GRM method instead.

    correlation_matrix : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs. Default: ``False``.

    savefile : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    Returns
    -------
    dict
        Result fields: ``zzt``, ``w``, ``grm``. Field arrays preserve the corresponding input axis order.
    """
    return GRM_lowmem(X, scale=scale, return_weights=return_weights,
        nan_policy=nan_policy, correlation_matrix=correlation_matrix,
        dtype=np.float32, savefile=savefile)


def GRM_lowmem(X,*, scale: bool = True, return_weights: bool = False, nan_policy: str = "ignore", correlation_matrix: bool = False, dtype=np.float32,  savefile = None):
    """Accumulate canonical npplink relationship statistics in bounded blocks.

    This wrapper uses sample SD (ddof=1), the common npplink engine, and excludes
    monomorphic markers before pair counting. The earlier duplicate routine used
    population SD and counted constant markers; its output is intentionally retired.
    TIME O(n^2*m); SPACE O(n^2+n*b), for n animals, m variants and block width b.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    scale : bool
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``.

    return_weights : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    nan_policy : str
        Only 'ignore' is retained: empirical sample-SD normalization with the
        common engine's pairwise n-minus-one denominator. Legacy 'mean' and
        'per_iid' modes raise; select an explicit modern GRM method instead.

    correlation_matrix : bool
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs. Default: ``False``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    savefile : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    Returns
    -------
    dict
        Result fields: ``zzt``, ``w``, ``grm``. Field arrays preserve the corresponding input axis order.
    """
    from .grm_backend import grm_from_blocks
    from .resources import bounded_variant_chunk
    if scale is not True or nan_policy != "ignore":
        raise ValueError("legacy GRM scale/nan_policy modes are retired; use vigor.grm(method='npplink'/'van'/'gcta0'/'vitezica')")
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("X must have shape (samples, variants)")
    n, m = shape
    bs = bounded_variant_chunk(n, bytes_per_call=16, n_variants=m)
    def blocks():
        for first in range(0, m, bs):
            block = X.iloc[:, first:first+bs] if isinstance(X, pd.DataFrame) else X[:, first:first+bs]
            yield np.asarray(block, dtype=dtype)
    kernels, weights, info = grm_from_blocks(blocks(), n, method="npplink",
        components=("additive",), min_maf=0.0, dtype=dtype, output_dtype=dtype)
    grm = kernels["additive"]
    w = weights["additive"]
    zzt = info["numerators"]["additive"]
    if correlation_matrix:
        sd = np.sqrt(np.diag(grm))
        den = sd[:, None] * sd[None, :]
        np.divide(grm, den, out=grm, where=den > 0)
        grm[den <= 0] = np.nan
    ids = X.index if isinstance(X, pd.DataFrame) else None
    if savefile is not None:
        if ids is not None:
            pd.DataFrame({"fid": ids, "iid": ids}).to_csv(str(savefile)+".grm.id", sep="\t", index=False, header=False)
        # Stream rows to avoid two n^2 integer index arrays.
        with open(str(savefile)+".grm.bin", "wb") as gf, open(str(savefile)+".grm.N.bin", "wb") as wf:
            for i in range(n):
                np.asarray(grm[i, :i+1], np.float32).tofile(gf)
                np.asarray(w[i, :i+1], np.float32).tofile(wf)
    if ids is not None:
        xr = _require_xarray()
        def labeled(A):
            # Plain label arrays: a named pandas Index would be read as its own dimension.
            labels = np.asarray(ids)
            return xr.DataArray(A, dims=("sample_0", "sample_1"), coords={"sample_0": labels, "sample_1": labels})
        return {"zzt": labeled(zzt), "w": labeled(w), "grm": labeled(grm)} if return_weights else labeled(grm)
    return {"zzt": zzt, "w": w, "grm": grm} if return_weights else grm

def plink2GRM(plinkfile:str, n_autosomes:int=None, downsample_snps:float = None, downsample_stategy:str = 'equidistant', rfids = None, chrs_subset = None,
              double_male_x:bool = False, double_y:bool = False, double_mt:bool = False,save_grms_path:bool = None, decompose_grm:bool = False, include_non_autosomal_in_all:bool = False):
    """Build genomic relationship matrices from PLINK input with the configured chromosome/output policy.

    Parameters
    ----------
    plinkfile : str
        Input for the operation ``plinkfile.loc[:, ['sex']]``. See ``vigor.npplink.plink2GRM``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    n_autosomes : int
        Input for the operation ``n_autosomes + 1``. See ``vigor.npplink.plink2GRM`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    downsample_snps : float
        Input supplied to ``plink2df`` as ``downsample_snps``. See that callable's documented
        ``downsample_snps`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to ``plink2df`` as
        ``downsample_snps``.

    downsample_stategy : str
        Input supplied to ``plink2df`` as ``downsample_stategy``. See that callable's documented
        ``downsample_stategy`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``'equidistant'``. Passed to ``plink2df`` as
        ``downsample_stategy``.

    rfids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``plink2df`` as ``rfids``.

    chrs_subset : object (see the cited input contract)
        Input for the operation ``list(map(str, chrs_subset))``. See ``vigor.npplink.plink2GRM``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    double_male_x : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    double_y : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    double_mt : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    save_grms_path : bool
        Input for the operation ``save_grms_path.rstrip('/')``. See ``vigor.npplink.plink2GRM``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    decompose_grm : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    include_non_autosomal_in_all : bool
        Input for the operation ``json.dump({'included_components': [str(x) for x in
        allgrms.index[allgrms.in_all_background]], 'include_non_autosomal_in_all':
        bool(include_non_autosomal_in_all)}, fh, indent=2)``. See ``vigor.npplink.plink2GRM``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``allgrms``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(plinkfile, pd.DataFrame):
        load_snps_from_df = True
        if 'sex' in plinkfile.columns: 
            fam = plinkfile.loc[:, ['sex']].rename({'sex':'gender'}, axis = 1).reset_index(names = 'iid')
        elif 'gender' in plinkfile.columns: 
            fam = plinkfile.loc[:, ['gender']].reset_index(names = 'iid')
        else: 
            print('sex|gender data not provided assigning all individuals to male')
            double_male_x = False
            fam = pd.DataFrame(data = {'gender':1, 'iid': plinkfile.index})
        bim = plinkfile.columns[plinkfile.columns.str.contains(':')].to_series()\
                  .str.split(':', expand = True).set_axis(['chrom', 'pos'], axis = 1)\
                  .astype({'pos': int}) 
    else:
        bim, fam, gen = load_plink(plinkfile) if isinstance(plinkfile,str) else plinkfile 
        load_snps_from_df = False
    if save_grms_path is not None:
        save_grms_path = save_grms_path.rstrip('/')
        if not os.path.isdir(save_grms_path): os.makedirs(save_grms_path, exist_ok=True)
    allgrms = pd.DataFrame(columns = ['grm', 'w', 'zzt'])
    loopchr = bim.chrom.unique()
    if chrs_subset is not None:
        chrs_subset = list(map(str, chrs_subset))
        loopchr     = [x for x in loopchr if str(x) in chrs_subset]
    for c in tqdm(loopchr, desc = 'making grm'):
        if load_snps_from_df: 
            snps = plinkfile.filter(regex = f'^{c}:')
            if rfids is not None: snps = snps.loc[list(rfids)]
        else: snps = plink2df((bim, fam, gen), c = c, downsample_snps=downsample_snps, rfids = rfids,
                                downsample_stategy=downsample_stategy)
        if n_autosomes is not None:
            num2xymt = lambda x: x if not (z:=defaultdict(int, {n_autosomes+1:'x',n_autosomes+2:'y',
                                                                n_autosomes+3:'xy', n_autosomes+4:'mt'})[int(x)]) else z
            if (c2 := str(c).replace('chr', '').lower()).isdigit(): 
                c2 = num2xymt(c2)
            if c2 in ['x', 'y']: 
                maleiids = fam[fam.gender.isin(['M', 'm', 1, '1'])].iid
                if   c2 == 'x' and double_male_x: snps.loc[maleiids, :] *= 2
                elif c2 == 'y':  
                    snps = snps.loc[maleiids, :]
                    if double_y: snps *= 2
                elif c2 == 'mt' and double_mt: snps *= 2
        else: c2 = c
        filename = f'{save_grms_path}/{c2}chrGRM' if (save_grms_path is not None) else None
        allgrms.loc[str(c2)] = GRM(snps, return_weights=True, savefile=filename)
        
    allgrms['isnum'] = allgrms.index.astype(str).str.isnumeric()
    # Historical ``All`` is autosomal.  Optionally include non-autosomal GRMs
    # only when their sample axes exactly match the autosomal background.
    included=allgrms['isnum'].copy()
    if include_non_autosomal_in_all and included.any():
        ref=allgrms.loc[included,'grm'].iloc[0]
        ref_shape=np.asarray(ref).shape
        ref_ids=tuple(map(str, getattr(ref, 'coords', {}).get('sample_0', []))) if hasattr(ref,'coords') else None
        for idx,row in allgrms.loc[~included].iterrows():
            g=row.grm
            same=np.asarray(g).shape==ref_shape
            if same and ref_ids is not None and hasattr(g,'coords'):
                same=tuple(map(str,g.coords['sample_0'].values))==ref_ids
            if same:
                included.loc[idx]=True
    allgrms['in_all_background']=included.astype(bool)
    if (save_grms_path is not None) and (chrs_subset is None):
        allzzt = allgrms.loc[allgrms.in_all_background,'zzt'].sum()
        allw = allgrms.loc[allgrms.in_all_background,'w'].sum()    
        allgrmf = allzzt/allw
        allgrmf['grm'].coords['sample_0'].to_dataframe().to_csv(f"{save_grms_path}/AllchrGRM.grm.id", header = None, sep = '\t')
        # fam[['iid', 'iid']].to_csv(f"{save_grms_path}/AllchrGRM.grm.id", index = False, header = None, sep = '\t')
        idxs = np.tril_indices_from(allgrmf)
        allgrmf.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.bin") 
        allw.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.N.bin") 
        allgrms['subtracted_grm'] = allgrms.progress_apply(
            lambda x: ((allzzt-x.zzt)/(allw-x.w)).to_pandas() if x.in_all_background else allgrmf.to_pandas(), axis=1)
        allgrms.loc['All', ['grm','w','zzt','subtracted_grm','isnum','in_all_background']] = (allgrmf,allw,allzzt,allgrmf,False,False)
        with open(f"{save_grms_path}/AllchrGRM.domino.json",'w') as fh:
            json.dump({'included_components':[str(x) for x in allgrms.index[allgrms.in_all_background]],
                       'include_non_autosomal_in_all':bool(include_non_autosomal_in_all)},fh,indent=2)
    if decompose_grm and (chrs_subset is None):
        allgrms[['U', 's']] = allgrms.progress_apply(lambda x: grm2Us(x.subtracted_grm.values, n_components=2000)[:2], 
                                                         axis=1, result_type="expand")
    allgrms = allgrms.sort_index(key = lambda idx : idx.str.lower().map({str(i): int(i) for i in range(1000)}|\
                                  {i: int(i) for i in range(1000)}|\
                                  {'all': -1000, 'x':1001, 'y' : 1002, 'mt': 1003, 'm': 1003}))
    return allgrms


def king_robust_kinship(G):
    """Compute conservative between-family KING-robust kinship (Manichaikul Eq. 11).

    Parameters
    ----------
    G : array-like, shape (n, m)
        Autosomal diploid hardcalls 0/1/2, with NaN for missing. The caller
        selects autosomes. Fractional dosages and infinite values are rejected.

    Returns
    -------
    numpy.ndarray, shape (n, n), dtype float64
        Symmetric kinship. Pairs with zero jointly observed heterozygosity are
        NaN; negative estimates are retained. Informative diagonal entries are
        0.5. The between-family formula is
        0.5 - sum((g_i-g_j)^2)/(4*min(H_i,H_j)), where H counts heterozygotes
        only among jointly called markers. This includes both het-het sharing
        and opposite homozygotes; it is not an IBS0-only statistic.

    Complexity
    ----------
    Time O(n^2*m), space O(n*m+n^2) for the NumPy path and O(n^2) beyond input
    for the compiled path.

    Notes
    -----
    Manichaikul et al. (2010), doi:10.1093/bioinformatics/btq559, Eq. 11.
    The within-family Eq. 9 is a different estimator; no family labels are
    inferred here. No inbreeding adjustment or clipping is applied.
    """
    X = np.asarray(G)
    if X.ndim != 2:
        raise ValueError("G must have shape (samples, markers)")
    valid = ~np.isnan(X)
    if np.any(valid & (~np.isfinite(X) | ((X != 0) & (X != 1) & (X != 2)))):
        raise ValueError("KING requires hardcalls 0/1/2 or NaN")
    # Float64 numeric indicators avoid NumPy's boolean-semiring dot product
    # and keep exact integer counts for all practical marker counts.
    M = valid.astype(np.float64)
    A = (X == 0).astype(np.float64)
    B = (X == 2).astype(np.float64)
    H = (X == 1).astype(np.float64)
    opposite = A @ B.T
    opposite += opposite.T.copy()
    hi = H @ M.T
    distance = 4 * opposite - 2 * (H @ H.T) + hi + hi.T
    denom = 4 * np.minimum(hi, hi.T)
    kin = np.full(denom.shape, np.nan, dtype=np.float64)
    np.divide(distance, denom, out=kin, where=denom > 0)
    kin = 0.5 - kin
    return kin

def recodeSNP(s, a1, a2):
    """Return the recodeSNP result computed by ``np.where(np.isnan(s), 'NA', np.where(s == 0, a1 + a1, np.where(s == 1, a1 + a2, np.where(s == 2, a2 + a2, '??')))).astype``.

    Parameters
    ----------
    s : array-like
        Numerical operand in ``np.where(np.isnan(s), 'NA', np.where(s == 0, a1 + a1, np.where(s
        == 1, a1 + a2, np.where(s == 2, a2 + a2, '??')))).astype('U2')``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state. Source comparisons: ``s
        == 0``; ``s == 1``; ``s == 2``. See the rejection/fallback branches for enforcement.

    a1 : object (see the cited input contract)
        Input for the operation ``a1 + a1``. See ``vigor.npplink.recodeSNP`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    a2 : object (see the cited input contract)
        Input for the operation ``a1 + a2``. See ``vigor.npplink.recodeSNP`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    Returns
    -------
    object
        Computed result from ``np.where(np.isnan(s), 'NA', np.where(s == 0, a1 + a1, np.where(s == 1, a1 + a2, np.where(s == 2, a2 + a2, '??')))).astype('U2')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return np.where(np.isnan(s), 'NA',
                    np.where(s == 0, a1+a1,
                    np.where(s == 1, a1+a2,
                    np.where(s == 2, a2+a2, '??')))).astype('U2')

def recodeSNPs(arr, a0, a1):
    """Recode allele/hardcall columns into the requested genotype representation.

    Parameters
    ----------
    arr : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Source comparisons: ``arr == 0``; ``arr == 1``; ``arr == 2``.
        See the rejection/fallback branches for enforcement.

    a0 : array-like or numeric scalar
        Input for the operation ``np.char.add(a0, a0)``. See ``vigor.npplink.recodeSNPs`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    a1 : array-like or numeric scalar
        Input for the operation ``np.char.add(a0, a1)``. See ``vigor.npplink.recodeSNPs`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    a0, a1 = np.asarray(a0, dtype='U1'), np.asarray(a1, dtype='U1')
    hom_ref, het, hom_alt = np.char.add(a0, a0), np.char.add(a0, a1), np.char.add(a1, a1)
    out = np.full(arr.shape, 'NA', dtype='U2')
    rows, cols = np.indices(arr.shape)
    out[(arr == 0)] = hom_ref[cols[(arr == 0)]]
    out[(arr == 1)] = het[cols[(arr == 1)]]
    out[(arr == 2)] = hom_alt[cols[(arr == 2)]]
    if isinstance(arr, pd.DataFrame):
        out = pd.DataFrame(out, index = arr.index, columns = arr.columns).astype('string[pyarrow]')
    return out

def plink2df(plinkpath: str, rfids: list = None, snplist: list = None, c: int = None, pos_start: int = None, pos_end: int = None,
             sex:  Literal['M', 'F', 'ALL', 'UNK', 'NOUNK'] = 'ALL', 
             downsample_snps: int = None, downsample_stategy: Literal['random', 'equidistant'] = 'random', recodeACGT: bool = False) -> pd.DataFrame:
    """Decode a selected PLINK genotype region into a labeled pandas DataFrame.

    Parameters
    ----------
    plinkpath : str
        Input for the operation ``type(plinkpath) == str``. See ``vigor.npplink.plink2df`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    rfids : list
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    snplist : list
        Input for the operation ``snplist is not None``. See ``vigor.npplink.plink2df`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``None``.

    c : int
        Input for the operation ``int(c)``. See ``vigor.npplink.plink2df`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    pos_start : int
        Input for the operation ``pos_start is not None``. See ``vigor.npplink.plink2df`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``None``.

    pos_end : int
        Input for the operation ``pos_end is not None``. See ``vigor.npplink.plink2df`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``None``.

    sex : Literal['M', 'F', 'ALL', 'UNK', 'NOUNK']
        Sex filter/classification using pedigree metadata: 'F' female, 'M' male, 'UNK' unknown
        and 'NOUNK' excludes unknown sex where the pedigree routine supports those selectors.
        Default: ``'ALL'``.

    downsample_snps : int
        Input for the operation ``downsample_snps * sset.shape[0]``. See
        ``vigor.npplink.plink2df`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.
        Source comparisons: ``0 < downsample_snps <= 1``. See the rejection/fallback branches
        for enforcement.

    downsample_stategy : Literal['random', 'equidistant']
        Input for the operation ``downsample_stategy == 'random'``. See
        ``vigor.npplink.plink2df`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``'random'``. Source-recognized
        literal values: ``equidistant``. These select the corresponding branches below;
        comparison evidence alone does not establish an exhaustive enum.

    recodeACGT : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(recodeSNPs(gen[col.i.values][:, sset.i.values].astype(np.float32), sset.a0.values, sset.a1.values), index=col.index.values, columns=sset.snp.values).astype('string[pyarrow]').replace('...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if type(plinkpath) == str:  snps, iid, gen = load_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    try:
        snps.chrom = snps.chrom.astype(int)
        if c is not None: c = int(c)
    except: 
        if c is not None: c = str(c)
        print('non numeric chromosomes: using str(c) in this case')
    
    query_sentence = []
    if c is not None: query_sentence += ['chrom == @c']
    if (pos_start is not None) and (pos_end is not None): query_sentence += ['pos.between(@pos_start,@pos_end )']
    elif pos_start is not None: query_sentence += ['pos >= @pos_start']
    elif pos_end is not None: query_sentence += ['pos <= @pos_end']
    if snplist is not None: query_sentence += ['snp.isin(@snplist)']
    query_sentence = ' and '.join(query_sentence)
    sset = snps.query(query_sentence) if len(query_sentence) else snps
    if downsample_snps is not None:
        if (0<downsample_snps<=1): downsample_snps= int(downsample_snps*sset.shape[0])
        if downsample_stategy == 'random': sset = sset.sample(min(int(downsample_snps), sset.shape[0])).sort_values('i')
        elif downsample_stategy == 'equidistant': sset = sset[::max(1,sset.shape[0]//int(downsample_snps))]
        else: print('''downsample_stategy not in ['random', 'equidistant'], ignoring downsampling''')

    iiid = iid.assign(i = iid.index).set_index('iid')
    query_sentence_iid = []
    if sex != 'ALL':
        if sex == 'M': query_sentence_iid += ['gender.isin(["M", 1, "1"])']
        if sex == 'F': query_sentence_iid += ['gender.isin(["F", 2, "2"])']
        if sex == 'ALL': query_sentence_iid += ['gender.isin(["F", 2, "2","M", 1, "1"])']
        if sex == 'UNK': query_sentence_iid += ['gender.isin(["0", 0])']
        if sex == 'NOUNK': query_sentence_iid += ['~gender.isin(["0", 0])']    
    query_sentence_iid = ' and '.join(query_sentence_iid)
    if len(query_sentence_iid): iiid = iiid.query(query_sentence_iid) 
    if rfids is not None: 
        if len(y:= set(rfids)-set(iiid.index)):  
            print(f'iids {"|".join(y)} not in rfid genotype file, sample will be dropped')
        rfids = [x for x in rfids if x in iiid.index]
    else: rfids = iiid.index
    col = iiid.loc[rfids]
    
    if not recodeACGT:
        return pd.DataFrame(gen[col.i.values ][:, sset.i.values].astype(np.float32),
                            index = col.index.values, columns = sset.snp.values )
    return pd.DataFrame(recodeSNPs(gen[col.i.values ][:, sset.i.values].astype(np.float32), 
                                   sset.a0.values, sset.a1.values),
                        index = col.index.values, columns = sset.snp.values )\
             .astype('string[pyarrow]').replace('NA', np.nan)

def whiten_data(U, D_inv_sqrt, X):
    """Transform trait data by the supplied covariance/eigenbasis whitening operator.

    Parameters
    ----------
    U : array-like
        Numerical operand in ``np.einsum('ij,jk->ik', U.T, X, optimize=True)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    D_inv_sqrt : array-like or numeric scalar
        Input for the operation ``np.diag(D_inv_sqrt)``. See ``vigor.npplink.whiten_data`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    d = np.diag(D_inv_sqrt)
    temp = np.einsum('ij,jk->ik', U.T, X,optimize=True)
    temp = d[:, None] * temp
    result = np.einsum('ij,jk->ik', U, temp, optimize=True)
    return result

def subblock_svd_from_full(U, s, obs, n_components=50):
    """Restrict a supplied full eigensystem to the requested animal subset.

    Parameters
    ----------
    U : array-like
        Numerical operand in ``U[obs, :] * np.sqrt(s)[None, :]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``U[obs, :] * np.sqrt(s)[None, :]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    obs : mapping
        Observed-cell plan containing animal/trait indices, trait slices and finite-response
        values. Obtain it from the observation-plan helper for the same phenotype matrix. Passed
        to ``grm2Us`` as ``n_components``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``50``. Passed to ``grm2Us`` as ``n_components``.

    Returns
    -------
    object
        Computed result from ``(Usub, ssub ** 2)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    Usub, ssub = grm2Us(U[obs, :] * np.sqrt(s)[None, :]  ,  n_components=min(obs.sum(), n_components))
    return Usub, ssub**2

def _read_bin_url(path):
    """Read bin url.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.
    """
    import urllib
    if path.startswith(('http://', 'https://')):
        with urllib.request.urlopen(path) as resp: raw = resp.read()
    else:
        with open(path, 'rb') as f: raw = f.read()
    return raw

def read_grm(path):
    """Read genomic relationship matrix.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    Returns
    -------
    dict
        Result fields: ``grm``, ``w``, ``path``. Field arrays preserve the corresponding input axis order.
    """
    ids = pd.read_csv(f"{path}.id",  sep=r'\s+', header=None, 
                      names=['fid','iid'], dtype={'fid': str, 'iid': str})
    n = len(ids)
    n_elements = (n * (n + 1)) // 2
    grm_values = np.frombuffer(_read_bin_url(f"{path}.bin"), dtype=np.float32, count=n_elements)
    grm = np.zeros((n, n), dtype=np.float64)
    grm[np.tril_indices(n)] = grm_values
    grm += np.tril(grm,-1).T
    grm = _require_xarray().DataArray(grm, dims=["sample_0", "sample_1"], 
                 coords={"sample_0":ids.iid, "sample_1": ids.iid, 
                         'iid': ('sample_1', ids.iid), 'fid': ('sample_1', ids.fid)})
    w = np.zeros_like(grm, dtype=np.float64) 
    w[np.tril_indices(n)] = np.frombuffer(_read_bin_url(f"{path}.N.bin"), dtype=np.float32, count=n_elements)
    w += np.tril(w, -1).T
    w = _require_xarray().DataArray(w, dims=["sample_0", "sample_1"], coords={"sample_0":grm.sample_0, "sample_1": grm.sample_1} )
    return {'grm':grm,'w': w, 'path': path}

def grm2Us(G, n_components=None, *, decomposition='exact', random_state=0):
    """Decompose a GRM through the canonical symmetric eigenvalue engine.

    Inputs: finite symmetric G (n,n); n_components positive integer or None;
    decomposition exact/randomized; random_state seed for randomized projection.
    Returns (U, eigenvalues) in descending order, clipping negative eigenvalues
    to zero with the canonical warning; never reflects them by absolute value.
    Complexity: exact O(n^3) time/O(n^2) memory; retained randomized work follows
    vigor.grm._eig and changes the represented kernel only when explicitly asked.
    """
    from .grm import _eig
    return _eig(G, n_components=n_components, decomposition=decomposition, random_state=random_state)

def load_all_grms(paths, decompose_grm=True, save_all_grm_if_missing: bool=True, include_non_autosomal_in_all: bool=False):
    """Load all grms.

    Parameters
    ----------
    paths : object (see the cited input contract)
        Input for the operation ``paths[0]``. See ``vigor.npplink.load_all_grms`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.

    decompose_grm : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    save_all_grm_if_missing : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    include_non_autosomal_in_all : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``allgrms``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from glob import glob
    if isinstance(paths, str): paths = glob(paths)
    allgrms = pd.DataFrame.from_records([read_grm(x.replace('.N', '').replace('.grm.bin', '.grm')) for x in paths])
    #allgrms = pd.DataFrame.from_records([read_grm_w(x) for x in glob(r'grm/*.grm.bin')])
    allgrms['path'] = allgrms['path'].str.extract(r'([\d\w]+)chrGRM.')
    allgrms['isnum'] = allgrms['path'].astype(str).str.isnumeric()
    allgrms = allgrms.set_index('path')
    included=allgrms['isnum'].copy()
    if include_non_autosomal_in_all and included.any():
        ref=np.asarray(allgrms.loc[included,'grm'].iloc[0])
        for idx,g in allgrms.loc[~included,'grm'].items():
            if np.asarray(g).shape==ref.shape:
                included.loc[idx]=True
    allgrms['in_all_background']=included.astype(bool)
    if 'All' not in allgrms.index:  
        allw = allgrms.loc[allgrms.in_all_background,'w'].sum()    
        allgrmf = ( allgrms.loc[allgrms.in_all_background,'grm'] * allgrms.loc[allgrms.in_all_background,'w'] ).sum() / allw
        allgrms.loc['All', ['grm','w','subtracted_grm','isnum','in_all_background']] = (allgrmf,allw,allgrmf,False,False)
        if save_all_grm_if_missing:
            save_grms_path = os.path.dirname(paths[0]) if isinstance(paths,(list,tuple)) else os.path.dirname(str(paths))
            allgrms.loc['All','grm'].index.to_series().reset_index(name= '___')\
                   .to_csv(f"{save_grms_path}/AllchrGRM.grm.id", index = False, header = None, sep = '\t')
            idxs = np.tril_indices_from(allgrmf)
            allgrmf.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.bin") 
            allw.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.N.bin")
        
    # If a VIGOR sidecar exists, it is authoritative about which chromosome
    # GRMs contributed to ``All``.  Historical files without a sidecar are
    # conservatively treated as autosomal backgrounds.
    try:
        any_path=str(next(iter(paths)))
        meta_path=os.path.join(os.path.dirname(any_path),'AllchrGRM.domino.json')
        if os.path.exists(meta_path):
            with open(meta_path) as fh: meta=json.load(fh)
            comps=set(map(str,meta.get('included_components',[])))
            allgrms['in_all_background']=[str(i) in comps for i in allgrms.index]
            if 'All' in allgrms.index: allgrms.loc['All','in_all_background']=False
    except Exception:
        pass
    allgrms_weighted = allgrms.loc["All", 'grm']*allgrms.loc["All", 'w']
    allgrms['subtracted_grm'] = allgrms.progress_apply(
        lambda x: (((allgrms_weighted-x.grm*x.w)/(allgrms.loc["All",'w']-x.w)).to_pandas()
                   if bool(x.in_all_background) else allgrms.loc["All",'grm'].to_pandas()), axis=1)
    allgrms['loco_applied']=allgrms['in_all_background'].astype(bool)
    if 'All' in allgrms.index: allgrms.loc['All','loco_applied']=False
    if decompose_grm:
        allgrms[['U', 's']] = allgrms.progress_apply(lambda x: grm2Us(x.subtracted_grm.values, n_components=2000, random_state=0)[:2], 
                                                     axis=1,  result_type="expand")
    allgrms = allgrms.sort_index(key = lambda idx : idx.str.lower().map({str(i): int(i) for i in range(1000)}|\
                              {i: int(i) for i in range(1000)}|\
                              {'all': -1000, 'x':1001, 'y' : 1002, 'mt': 1003, 'm': 1003}))
    return allgrms


def H2SVD(y, grm=None,s=None, U=None, l='REML', n_components = None, return_SVD = False, tol = 1e-8):
    """Estimate trait heritability in the supplied retained relationship eigenspace.

    Parameters
    ----------
    y : array-like
        Numerical operand in ``np.dot(U.T, y) ** 2``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    grm : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    s : array-like
        Numerical operand in ``sg * s + se``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state. Default: ``None``. None delegates to the local
        fallback shown in this body; it is not an additional numeric/category value.

    U : array-like
        Numerical operand in ``U ** 2``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    l : str
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``'REML'``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``. Passed to ``grm2Us`` as ``n_components``.

    return_SVD : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-08``.

    Returns
    -------
    dict
        Result fields: ``U``, ``s``, ``h2``, ``L``, ``quad``, ``log_det``, ``se``, ``pval``. Field arrays preserve the corresponding input axis order.
    """
    from scipy.optimize import minimize
    from scipy.linalg import solve_triangular
    y = np.array(y).flatten()
    notnan = ~np.isnan(y)
    obs = np.where(notnan)[0]
    sp, m, N = np.nanvar(y, ddof=1), np.nanmean(y), notnan.sum()
    y = y[obs]
    y -= m
    if s is None and U is None and grm is not None:
        grm = np.array(grm)
        grm = grm[np.ix_(obs, obs)]
        if n_components is None: n_components =grm.shape[0]
        U, s = grm2Us(grm, n_components=n_components)
    elif s is not None and U is not None and grm is None: 
        #Ue, se = subblock_svd_from_full(U, s, obs, n_components = 700)
        U = U[obs, :]
        s *= np.sum(U**2, axis=0) #* s
        if n_components is not None:
            U, s = U[:, :n_components], s[:n_components] 
        # U[:, :len(se)] = Ue
        # s[:len(se)]    = se
        
    else: raise ValueError('cannot submit both grm and U s at the same time')
    Ur2 = np.dot(U.T, y)**2
    s = np.maximum(s, tol)        
    def _L(h2_arr):
        """Resolve the enclosing relationship/LD factor representation for matrix operations.

        Parameters
        ----------
        h2_arr : object (see the cited input contract)
            Input for the operation ``h2_arr[0]``. See ``vigor.npplink.H2SVD._L`` and its caller for
            the scientific representation; the shown operation constrains the required type/shape.
        """
        h2 = h2_arr[0]
        sg = h2 * sp
        se = sp - sg 
        sgsse = sg * s + se
        log_det = np.sum(np.log(sgsse))
        quad = np.sum(Ur2/sgsse)
        log_likelihood = -0.5 * (quad + log_det + N * np.log(2 * np.pi))
        if l == 'REML':
            Xt_cov_inv_X = np.sum(1.0 / sgsse)
            log_likelihood -= 0.5 * np.log(Xt_cov_inv_X)
        return -log_likelihood
        
    result = minimize(fun=_L, x0=[0.5], bounds=[(0., 1.)])
    h2, likelihood = result.x[0], result.fun
    if return_SVD: 
        sg, se = h2 * sp, sp*(1- h2)
        sgsse = sg * s + se
        log_det = np.sum(np.log(sgsse))
        quad = np.sum(Ur2/sgsse)
        eps = 1e-5
        f0, fph, fmh = _L([h2]),  _L([min(h2 + eps, 1.0)]),  _L([max(h2 - eps, 0.0)])
        second_deriv = (fph - 2*f0 + fmh) / eps**2
        se_hat = np.sqrt(1.0 / second_deriv)
        return {'U':U, 's': s, 'h2': result.x[0], 'L': likelihood,
                'quad':quad, 'log_det': log_det, 'se': se_hat, 
               'pval' : 1 - chi2.cdf(2 * (_L([0.0]) - likelihood), df=1)}
    return h2

def remove_relatedness_transformation(G=None, U=None, s=None, h2=0.5, yvar=1, tol=1e-8, random_state=None, n_components=None, return_eigen = False, **kws):
    """Undo the stored relatedness-removal trait transform.

    Parameters
    ----------
    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    U : array-like
        Numerical operand in ``U @ D_inv_sqrt @ U.T``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.

    s : array-like
        Numerical operand in ``yvar * (h2 * s + (1 - h2))``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    h2 : float or one-dimensional array
        Genetic variance fraction in [0, 1], scalar or ordered by trait as required. This is the
        working/reference heritability when the routine builds a null precision rather than
        estimating the final fraction. Default: ``0.5``.

    yvar : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``1``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-08``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``None``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``. Passed to ``grm2Us`` as ``n_components``.

    return_eigen : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    **kws : dict
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    Returns
    -------
    object
        Computed result from ``W``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if s is None and U is None and G is not None:
        G = yvar * (h2 * G + (1 - h2) * np.eye(G.shape[0]))
        if n_components is None: n_components = G.shape[0]
        U, s  = grm2Us(G, n_components=n_components)
    elif s is not None and U is not None and G is None: pass
    else: raise ValueError('cannot submit both G and U s at the same time')
    eigs_fG = yvar * (h2 * s + (1 - h2))
    eigs_fG[eigs_fG < tol] = tol
    D_inv_sqrt = np.diag(1 / np.sqrt(eigs_fG))
    if return_eigen: return U, D_inv_sqrt
    W = U @ D_inv_sqrt @ U.T
    return W
    
def rm_relatedness(c, trait, df, allgrms, n_components = None,return_eigen=True, svd_input = True, ):
    """Transform phenotypes to reduce the fitted animal relatedness contribution.

    Parameters
    ----------
    c : object (see the cited input contract)
        Input supplied to ``pd.Series`` as ``name``. See that callable's documented ``name``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Passed to ``pd.Series`` as ``name``.

    trait : object (see the cited input contract)
        Input supplied to ``pd.Series`` as ``name``. See that callable's documented ``name``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Passed to ``pd.Series`` as ``name``.

    df : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    allgrms : object (see the cited input contract)
        Input supplied to ``H2SVD`` as ``s``. See that callable's documented ``s`` contract for
        its type, alignment and value domain. This wrapper retains that contract. Passed to
        ``H2SVD`` as ``s``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``. Passed to ``H2SVD`` as ``n_components``.

    return_eigen : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``True``.

    svd_input : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``True``.

    Returns
    -------
    dict
        Result fields: ``transformed``, ``blup``, ``U``, ``D_inv_sqrt``, ``c``, ``trait``, ``h2``, ``n_components``. Field arrays preserve the corresponding input axis order.
    """
    grm_c = allgrms.loc[str(c),'subtracted_grm']
    trait_ = df.loc[grm_c.index, trait]
    navals = ~trait_.isna()
    yvar = np.nanvar(trait_)
    if navals.sum()< 10: 
        print('not enough samples')
        return None
    if svd_input:
        h2_res = H2SVD(y = trait_, s = allgrms.loc[str(c),'s'], U = allgrms.loc[str(c),'U'],
                   return_SVD=True, n_components = n_components) 
    else: 
        h2_res = H2SVD(y = trait_, grm = grm_c, return_SVD=True, n_components = n_components) 
    U, D_inv_sqrt = remove_relatedness_transformation(yvar= yvar,**h2_res, return_eigen=True)
    if U.shape[0] == sum(navals): 
        trait_vec = trait_.loc[navals]#.to_frame()
        idx = navals[navals].index
    else: 
        trait_vec = trait_.copy()
        idx = navals.index
    uttrait = U.T @ trait_vec
    transformed = U @ (D_inv_sqrt @ uttrait)
    #transformed = whiten_data(U,D_inv_sqrt,trait_vec)
    weights = (h2_res['h2'] * h2_res['s']) / (h2_res['h2'] * h2_res['s'] + (1.0 - h2_res['h2']))
    blup_vec = (U @ (weights * uttrait))
    if return_eigen:
        return {'transformed':pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}'),
                'blup': pd.Series(blup_vec, index=idx, name=f'{trait}__blup{c}'),
                'U': U, 'D_inv_sqrt': D_inv_sqrt, 'c': c, 'trait': trait,
                'h2': h2_res['h2'], 'n_components':n_components}
    return pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}')

def scale_with_mask(X, precision = np.float32, center = True, scale = True):
    """Scale with mask.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    precision : array-like or numeric scalar
        Input supplied to ``np.ones`` as ``dtype``. See that callable's documented ``dtype``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``np.float32``. Passed to ``np.ones`` as ``dtype``.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``(np.ascontiguousarray(X_scaled, precision), std.astype(precision), np.ascontiguousarray(M, precision))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X = np.asarray(X, dtype=precision)
    M = ~np.isnan(X)
    if center:
        X_centered = X - np.nanmean(X, axis=0)
        X_centered[~M] = 0.0
    else: 
        X_centered = X.copy() 
        X_centered[~M] = 0.0
    if not scale: return np.ascontiguousarray(X_centered, precision), np.ones(X.shape[1], dtype=precision), np.ascontiguousarray(M, precision)
    sum_sq = np.einsum("ij,ij->j", X_centered, X_centered, optimize=True)
    std = np.sqrt(sum_sq / M.sum(axis=0))
    std[std == 0] = np.nan
    X_scaled = X_centered / std
    return np.ascontiguousarray(X_scaled, precision), std.astype(precision), np.ascontiguousarray(M, precision)

def regression_with_einsum(ssnps, straits, snps_mask, traits_mask,dof='correct', stat = 'ttest', sided = 'two-sided', center = True,):
    """Compute blockwise genotype/trait regression statistics using Einstein summation.

    Parameters
    ----------
    ssnps : object (see the cited input contract)
        Input for the operation ``ssnps.shape[1]``. See ``vigor.npplink.regression_with_einsum``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    straits : object (see the cited input contract)
        Input for the operation ``straits * traits_mask``. See
        ``vigor.npplink.regression_with_einsum`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    snps_mask : array-like or numeric scalar
        Input for the operation ``np.einsum('ij,ik,ik->jk', snps_mask, phen, phen, out=ssr,
        optimize=True)``. See ``vigor.npplink.regression_with_einsum`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    traits_mask : object (see the cited input contract)
        Input for the operation ``straits * traits_mask``. See
        ``vigor.npplink.regression_with_einsum`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    dof : str or float
        Degrees-of-freedom policy for association calibration. 'correct' uses the
        implementation's effective sample/design degrees of freedom; numeric settings or other
        named policies are accepted only by the called regression helper. Default:
        ``'correct'``. Source-recognized literal values: ``incorrect``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    stat : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'ttest'``.

    sided : int
        Number of hypothesis-test tails (one or two), used when converting an effect statistic
        to a p-value. Default: ``'two-sided'``. Source-recognized literal values: ``one-sided``.
        These select the corresponding branches below; comparison evidence alone does not
        establish an exhaustive enum.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``(beta, se_beta, stats, p_values, df)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if sided not in ['two-sided','one-sided']: raise ValueError("sided must be 'two-sided' or 'one-sided'")
    if stat  not in ['ttest', 'wald', 'score']: raise ValueError("stat must be 'ttest' or 'wald'")
    XtY      = np.empty((ssnps.shape[1],straits.shape[1]), dtype=float)
    diag     = np.empty_like(XtY)
    ssr      = np.empty_like(XtY)
    df       = np.empty_like(XtY)
    np.einsum("ij,ik->jk", ssnps, straits, out=XtY, optimize=True)
    np.einsum("ij,ij,ik->jk", ssnps, ssnps, traits_mask, out=diag,  optimize=True)
    phen = straits * traits_mask   
    np.einsum("ij,ik,ik->jk", snps_mask, phen, phen, out=ssr, optimize=True)
    if dof!='incorrect': 
        np.einsum("ij,ik->jk", snps_mask, traits_mask, out=df,  optimize=True)
        df -= 2 if center else 1
    else: df = np.broadcast_to(traits_mask.sum(axis=0) - (2 if center else 1), (ssnps.shape[1], traits_mask.shape[1])).copy()
    df[df <= 0] = np.nan
    beta = XtY
    beta /= diag
    if stat != 'score': ssr -= beta * beta * diag
    np.divide(ssr, df*diag, out=ssr)
    np.sqrt(ssr, out=ssr)
    se_beta = ssr
    stats = beta/se_beta
    if stat == 'ttest':
        p_values = scipyt.sf(np.abs(stats), df=df)
        if sided == 'two-sided': p_values*=2
    elif stat in ['wald', 'score']:
        np.abs(stats, out = stats)
        p_values = erfc(stats/np.sqrt(2))
        if sided == 'one-sided': p_values*=.5
        np.square(stats, out = stats)
    np.log10(p_values, out=p_values)
    np.negative(p_values, out=p_values)
    return beta, se_beta, stats, p_values, df


def regression_with_blas(ssnps, straits, snps_mask, traits_mask, dof='correct', stat='ttest', sided='two-sided', center=True):
    """Compute the same genotype/trait regression terms using matrix multiplication.

    Parameters
    ----------
    ssnps : object (see the cited input contract)
        Input for the operation ``ssnps.shape[1]``. See ``vigor.npplink.regression_with_blas``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    straits : object (see the cited input contract)
        Input for the operation ``straits.shape[1]``. See ``vigor.npplink.regression_with_blas``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    snps_mask : array-like or numeric scalar
        Input for the operation ``np.result_type(ssnps.dtype, straits.dtype, snps_mask.dtype,
        traits_mask.dtype)``. See ``vigor.npplink.regression_with_blas`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    traits_mask : object (see the cited input contract)
        Input for the operation ``traits_mask.sum(axis=0) - (2 if center else 1)``. See
        ``vigor.npplink.regression_with_blas`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    dof : str or float
        Degrees-of-freedom policy for association calibration. 'correct' uses the
        implementation's effective sample/design degrees of freedom; numeric settings or other
        named policies are accepted only by the called regression helper. Default:
        ``'correct'``. Source-recognized literal values: ``incorrect``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    stat : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'ttest'``.

    sided : int
        Number of hypothesis-test tails (one or two), used when converting an effect statistic
        to a p-value. Default: ``'two-sided'``. Source-recognized literal values: ``one-sided``.
        These select the corresponding branches below; comparison evidence alone does not
        establish an exhaustive enum.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``(beta, se_beta, stats, p_values, df)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if sided not in ['two-sided', 'one-sided']: raise ValueError("sided must be 'two-sided' or 'one-sided'")
    if stat not in ['ttest', 'wald', 'score']: raise ValueError("stat must be 'ttest', 'wald', or 'score'")
    acc_dtype = np.result_type(ssnps.dtype, straits.dtype, snps_mask.dtype, traits_mask.dtype)
    XtY  = np.empty((ssnps.shape[1], straits.shape[1]), dtype=acc_dtype, order='C')
    diag = np.empty_like(XtY, order='C')
    ssr  = np.empty_like(XtY, order='C')
    df   = np.empty_like(XtY, order='C')
    ssnps2 = np.empty_like(ssnps, dtype=acc_dtype, order='C')
    phen2  = np.empty_like(straits, dtype=acc_dtype, order='C')
    np.dot(ssnps.T, straits, out=XtY)
    np.square(ssnps, out=ssnps2)
    np.dot(ssnps2.T, traits_mask, out=diag)
    np.multiply(straits, traits_mask, out=phen2)
    np.square(phen2, out=phen2)
    np.dot(snps_mask.T, phen2, out=ssr)
    if dof != 'incorrect':
        np.dot(snps_mask.T, traits_mask, out=df)
        df -= (2 if center else 1)
    else: df = np.broadcast_to( traits_mask.sum(axis=0) - (2 if center else 1), (ssnps.shape[1], straits.shape[1])).astype(float, copy=True)
    df[df <= 0] = np.nan
    diag[diag <= 0] = np.nan
    beta = XtY
    np.divide(beta, diag, out=beta)
    if stat != 'score': ssr -= (beta * beta) * diag
    np.divide(ssr, df * diag, out=ssr)
    np.sqrt(ssr, out=ssr)
    se_beta = ssr
    stats = np.empty_like(beta, order='C')
    np.divide(beta, se_beta, out=stats)
    if stat == 'ttest':
        p_values = scipyt.sf(np.abs(stats), df=df)
        if sided == 'two-sided': p_values *= 2
    elif stat in ['wald', 'score']:
        np.abs(stats, out=stats)
        p_values = erfc(stats / np.sqrt(2))
        if sided == 'one-sided': p_values *= 0.5
        np.square(stats, out=stats)
    np.log10(p_values, out=p_values)
    np.negative(p_values, out=p_values)
    return beta, se_beta, stats, p_values, df

def GWA(traits, snps, dtype = 'pandas_highmem', precision=np.float32, stat = 'ttest', sided = 'two-sided',dof='correct', center =True, scale = True, regression_mode = 'blas' ):
    """Run a genotype-block association calculation with the selected regression/calibration backend.

    Parameters
    ----------
    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    snps : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``'pandas_highmem'``. Source-recognized literal values:
        ``pandas``, ``tuple``, ``xarray``, ``xarray_dataset``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    precision : object (see the cited input contract)
        Input supplied to ``scale_with_mask`` as ``precision``. See that callable's documented
        ``precision`` contract for its type, alignment and value domain. This wrapper retains
        that contract. Default: ``np.float32``. Passed to ``scale_with_mask`` as ``precision``.

    stat : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'ttest'``. Passed to ``regression_with_blas`` as ``stat``.

    sided : int
        Number of hypothesis-test tails (one or two), used when converting an effect statistic
        to a p-value. Default: ``'two-sided'``. Passed to ``regression_with_blas`` as ``sided``.

    dof : str or float
        Degrees-of-freedom policy for association calibration. 'correct' uses the
        implementation's effective sample/design degrees of freedom; numeric settings or other
        named policies are accepted only by the called regression helper. Default:
        ``'correct'``. Passed to ``regression_with_blas`` as ``dof``.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``. Passed to ``scale_with_mask`` as
        ``center``.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``. Passed to ``scale_with_mask`` as
        ``scale``.

    regression_mode : str
        Dense genotype/trait contraction backend: 'blas' uses matrix multiplication, 'einsum'
        explicit Einstein summation where supported. The local scan validates available
        contraction paths. Default: ``'blas'``.

    Returns
    -------
    object
        Computed result from ``results``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(snps, tuple) and len(snps)==3: ssnps, snps_std, snps_mask = snps
    else: ssnps, snps_std, snps_mask = scale_with_mask(snps, precision = precision, center = center, scale = scale)
    if isinstance(traits, tuple) and len(traits)==3: straits, traits_std, traits_mask = traits
    else: straits, traits_std, traits_mask = scale_with_mask(traits, precision = precision, center = center, scale = scale)
    if regression_mode == 'blas':
        results = regression_with_blas(ssnps, straits, snps_mask, traits_mask, dof = dof, stat = stat, sided = sided, center = center)
    if regression_mode == 'einsum':
        results = regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof = dof, stat = stat, sided = sided, center = center)
    metrics = ['beta', 'beta_se', 'stat', 'neglog_p', 'dof']
    snp_names, trait_names   = list(snps.columns), [t.split('__subtractgrm')[0] for t in traits.columns]
    if dtype not in ['tuple', 'xarray', 'pandas', 'pandas_highmem', 'xarray_dataset']: raise ValueError(" dtype has to be in ['tuple', 'xarray', 'pandas']")
    if dtype == 'tuple':  return results
    elif dtype == 'xarray_dataset':
        return _require_xarray().Dataset( { m: (('snp', 'trait'), arr) for m, arr in zip(metrics, results) },
            coords={'snp': snp_names, 'trait': trait_names})
    elif dtype in ['xarray', 'pandas_highmem']:
        xar =  _require_xarray().DataArray( np.stack(results, axis=0),dims=["metric", "snp", "trait"],
             coords={"metric": np.array(['beta', 'beta_se', 'stat', 'neglog_p', 'dof']),
                     "snp":   snp_names, "trait": trait_names} )
        if dtype == 'xarray': return xar
        return xar.to_dataset(dim="metric").to_dataframe().reset_index()
    elif dtype == 'pandas': 
        snp_col   = np.repeat(snp_names, len(traits.columns))
        trait_col = np.tile(trait_names, len(snps.columns))
        return  pd.DataFrame({ 'snp':   snp_col, 'trait': trait_col,
            **{ metric: arr.ravel(order='C') for metric, arr in zip(metrics, results)}})
    return results

def GWAS(traitdf, genotypes = 'genotypes/genotypes', grms_folder = 'grm', save = True , y_correction  = 'ystar',
         save_path = 'results/gwas/', return_table = True, stat = 'ttest', dtype = 'pandas_highmem',dof='correct'):
    """Scan PLINK chromosomes using the retained npplink trait-correction workflow.

    Parameters
    ----------
    traitdf : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model. Default:
        ``'genotypes/genotypes'``.

    grms_folder : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'grm'``.

    save : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``True``.

    y_correction : str
        Phenotype preprocessing before the legacy association scan: 'blup_resid' removes the
        BLUP genetic prediction; 'ystar' uses the whitened/transformed phenotype. See the
        selected scan for accepted correction choices. Default: ``'ystar'``.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'results/gwas/'``.

    return_table : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    stat : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'ttest'``. Passed to ``GWA`` as ``stat``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``'pandas_highmem'``. Passed to ``GWA`` as ``dtype``.

    dof : str or float
        Degrees-of-freedom policy for association calibration. 'correct' uses the
        implementation's effective sample/design degrees of freedom; numeric settings or other
        named policies are accepted only by the called regression helper. Default:
        ``'correct'``. Passed to ``GWA`` as ``dof``.

    Returns
    -------
    object
        Computed result from ``pd.concat(res)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    res = []
    read_gen = load_plink(genotypes)
    chrunique =[str(x) for x in read_gen[0].chrom.unique()]
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = grms.index[~grms.index.str.contains('^All$')].to_list()
    for C in tqdm(chrs2run,position=0, desc = 'running Chr'):
        if y_correction=='blup_resid':
            traits =  tdf - pd.concat([rm_relatedness(C,_t,tdf,grms, svd_input=False)['blup'] for _t in tdf.columns], axis = 1).rename(lambda x: x.split('__')[0], axis = 1)
        elif y_correction=='ystar': 
            traits = pd.concat([rm_relatedness(C,_t,tdf,grms, svd_input=False)['transformed'] for _t in tdf.columns], axis = 1)
        elif y_correction is None:  traits = tdf
        else: raise ValueError(" y_correction has to be in ['blup_resid', 'ystar', None]")
        if str(C) not in chrunique: chr_alias = read_gen[0].loc[read_gen[0].snp.str.lower().str.startswith(f'{C}:').idxmax(), 'chrom']
        else: chr_alias = C
        snps = plink2df(read_gen, c=chr_alias, rfids=traits.index)
        if not (snps.index == traits.index).all(): 
            print('reordering snps to align with traits')
            snps =snps.loc[traits.index]
        if return_table: 
            res += [GWA(traits, snps, dtype=dtype, stat=stat, dof=dof)]
            if save: res[-1].to_parquet(f'{save_path}gwas{C}.parquet.gz', compression='gzip', engine = 'pyarrow',  compression_level=1, use_dictionary=True)
        else: 
            if save: GWA(traits, snps, dtype=dtype, stat=stat, dof=dof)\
                           .to_parquet(f'{save_path}gwas{C}.parquet.gz', compression='gzip',  engine = 'pyarrow',  compression_level=1, use_dictionary=True) #use_byte_stream_split=True
    if return_table: 
        if 'pandas' not in dtype: return res
        return pd.concat(res)
    return



def describe_trait_chr(traitdf, grms_folder = 'grm', return_allchrs=False, include_cols = ['transformed', 'blup', 'U', 'D_inv_sqrt', 'h2', 'n_components']):
    """Return the describe trait chr result computed by ``pd.DataFrame((pd.Series(rm_relatedness(_c, _t, tdf, grms, svd_input=False)).loc[include_cols] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave=True, desc='describing: ' + '|'.join(include_cols)))).set_index``.

    Parameters
    ----------
    traitdf : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    grms_folder : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'grm'``.

    return_allchrs : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    include_cols : list
        Input supplied to ``tqdm`` as ``desc``. See that callable's documented ``desc`` contract
        for its type, alignment and value domain. This wrapper retains that contract. Default:
        ``['transformed', 'blup', 'U', 'D_inv_sqrt', 'h2', 'n_components']``. Passed to ``tqdm``
        as ``desc``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame((pd.Series(rm_relatedness(_c, _t, tdf, grms, svd_input=False)).loc[include_cols] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave=True, desc='describing: ' + '|...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    include_cols = list(set(include_cols+['c', 'trait'] ))
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.DataFrame((pd.Series(rm_relatedness(_c,_t,tdf,grms,  svd_input=False)).loc[include_cols] \
                                for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), 
                                                   leave = True, desc = 'describing: ' + '|'.join(include_cols) )))\
           .set_index(['c', 'trait'])

def BLUP(traitdf, grms_folder = 'grm', return_allchrs=False):
    """Return the breeding-value prediction result computed by ``pd.concat``.

    Parameters
    ----------
    traitdf : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    grms_folder : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'grm'``.

    return_allchrs : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``pd.concat([rm_relatedness(_c, _t, tdf, grms, svd_input=False)['blup'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave=True, desc='calculating BLUP')], axis=1)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    grms = load_all_grms(f'{grms_folder}/{"All" if not return_allchrs else ""}*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.concat([rm_relatedness(_c,_t,tdf,grms, svd_input=False)['blup'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave = True, desc = 'calculating BLUP')], axis = 1)

def ySTAR(traitdf, grms_folder = 'grm', return_allchrs=False):
    """Return the ySTAR result computed by ``pd.concat``.

    Parameters
    ----------
    traitdf : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    grms_folder : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'grm'``.

    return_allchrs : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``pd.concat([rm_relatedness(_c, _t, tdf, grms, svd_input=False)['transformed'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave=True, desc='whittening y')], axis=1)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.concat([rm_relatedness(_c,_t,tdf,grms, svd_input=False)['transformed'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave = True, desc = 'whittening y')], axis = 1)

def heritability(traitdf, grms_folder = 'grm', return_allchrs=False, svd_input=False, n_components = None, GRM = None):
    """Estimate phenotype heritability using stored or supplied relationship eigensystems.

    Parameters
    ----------
    traitdf : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    grms_folder : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``'grm'``.

    return_allchrs : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    svd_input : bool
        Input for the operation ``params | (dict(s=allgrms.loc[str(_c), 's'],
        U=allgrms.loc[str(_c), 'U']) if svd_input else dict(grm=grm_c))``. See
        ``vigor.npplink.heritability`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape. Default: ``False``.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``.

    GRM : array-like, kernel operator, or mapping
        Animal relationship kernel with matching row/column animal order. A dense kernel is
        n_animals × n_animals; supported operator representations apply the same matrix without
        materialization. Named multi-kernel inputs require corresponding trait covariance
        components. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``pd.concat(res, axis=1).T``; see the summary and the called implementation for its labeled/numerical representation.
    """
    chrs2run, res = ['All'], []
    if GRM is None:
        if return_allchrs: 
            allgrms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
            chrs2run += allgrms.index[~allgrms.index.str.contains('^All$')].to_list()
        else: 
            allgrms = load_all_grms(f'{grms_folder}/All*.grm.bin', decompose_grm=False)
    for _c, _t in tqdm(list(itertools.product(chrs2run, traitdf.columns)), leave = True, desc = 'calculating heritability'):
        if GRM is None: grm_c = allgrms.loc[str(_c),'grm'].to_pandas()
        else: grm_c = GRM
        _y = traitdf.loc[grm_c.index, _t]
        params = dict(y = _y, return_SVD=True, n_components = n_components) 
        params = params | (dict(s = allgrms.loc[str(_c),'s'], U = allgrms.loc[str(_c),'U']) if svd_input else dict(grm = grm_c))
        idxname = f'{_t}_{_c}' if return_allchrs else f'{_t}'
        res += [pd.Series(H2SVD(**params) | dict(trait = _t, chrom = _c, n = _y.count()), name = idxname).drop(['U', 's'])]
    return pd.concat(res, axis=1).T

def shuffle_replicates(df,col ,n=500):
    """Generate columnwise permutation replicates for the enclosing null calculation.

    Parameters
    ----------
    df : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    col : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``500``.

    Returns
    -------
    object
        Computed result from ``reps``; see the summary and the called implementation for its labeled/numerical representation.
    """
    reps = df[[col]*n]
    for i in range(1, n):  reps.iloc[:, i] = np.random.permutation(reps.iloc[:, i].values)
    reps.columns = [f'{col}_ORIGINAL'] + [f'{col}_SHUFFLE{str(i).zfill(3)}' for i in range(1, n)]  
    return reps

def shuffle_replicates_normal(df,col ,n=500):
    """Generate normal-reference shuffled replicates for the selected phenotype columns.

    Parameters
    ----------
    df : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Passed to ``r.normal`` as ``size``.

    col : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.

    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``500``. Passed to ``r.normal`` as
        ``size``.

    Returns
    -------
    object
        Computed result from ``reps``; see the summary and the called implementation for its labeled/numerical representation.
    """
    napct = df[col].isna().mean()
    r = np.random.RandomState(42)
    reps =  pd.DataFrame(r.normal(size = (df.shape[0], n)), \
                         columns = [f'{col}_SHUFFLE{str(i).zfill(3)}' for i in range( n)], \
                         index = df.index )
    reps  *=  r.choice([1, np.nan],size =  reps.shape ,   p = [1-napct, napct])
    return reps


def r2toD(r2, snp1maf,snp2maf):
    """Return the r2toD result computed by ``np.sqrt``.

    Parameters
    ----------
    r2 : array-like
        Pairwise linkage/correlation quantities aligned to the corresponding variant pairs.
        Signed correlations are on [-1, 1]; squared correlations on [0, 1], with missing values
        representing unavailable pairs.

    snp1maf : object (see the cited input contract)
        Input for the operation ``r2 * snp1maf * (1 - snp1maf) * snp2maf * (1 - snp2maf)``. See
        ``vigor.npplink.r2toD`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape.

    snp2maf : object (see the cited input contract)
        Input for the operation ``r2 * snp1maf * (1 - snp1maf) * snp2maf * (1 - snp2maf)``. See
        ``vigor.npplink.r2toD`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape.

    Returns
    -------
    object
        Computed result from ``np.sqrt(r2 * snp1maf * (1 - snp1maf) * snp2maf * (1 - snp2maf))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return np.sqrt(r2*snp1maf*(1-snp1maf)*snp2maf*(1-snp2maf))

def R2(X, Y= None, return_named = True, return_square = True, statistic = 'r2', dtype = np.float32):
    """Compute labeled pairwise genotype correlation or squared correlation.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    return_named : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    return_square : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    statistic : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'r2'``. Source-recognized literal values: ``D``, ``D2``, ``chord``,
        ``cov``. These select the corresponding branches below; comparison evidence alone does
        not establish an exhaustive enum.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``.

    Returns
    -------
    object
        Computed result from ``res``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return R2_exact(X, Y, return_named=return_named, return_square=return_square,
                    statistic=statistic, dtype=dtype, block_size="auto")

def R2_exact_blocked(X, Y=None, return_named=True, return_square=True,
       statistic='r2', dtype=np.float32):

    """Compute pairwise-observed genotype LD in bounded variant blocks.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    return_named : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    return_square : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    statistic : str
        Association/LD statistic selection. 'ttest' uses a t calibration, 'score' a null score
        statistic and 'wald' effect/SE calibration where supported; 'r' is signed correlation
        and 'r2' squared correlation in LD routines. See the selected helper for its accepted
        subset. Default: ``'r2'``. Source-recognized literal values: ``D``, ``D2``, ``chord``,
        ``cov``. These select the corresponding branches below; comparison evidence alone does
        not establish an exhaustive enum.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    object
        Computed result from ``res``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return R2_exact(X, Y, return_named=return_named, return_square=return_square,
                    statistic=statistic, dtype=dtype, block_size="auto")

def R2_exact(X, Y=None, return_named=True, return_square=True,
             statistic='r2', dtype=np.float32, block_size=1024):
    """Compute pairwise-complete LD with the canonical centered-block kernel.

    Parameters
    ----------
    X, Y : array-like, shape (samples, variants)
        Aligned dosage columns, NaN/inf missing; Y=None computes X against X.
    return_named, return_square : bool
        Return a labeled DataFrame (using input columns) or an ndarray.
        A named non-square return is a long table with bp1/bp2/value.
    statistic : {'r2', 'r', 'cov', 'D', 'D2', 'chord'}
        Requested statistic; covariance uses population normalization.
    dtype : numpy.float32 or numpy.float64
        Working/GEMM/output dtype. Small sufficient-statistic arithmetic is float64.
    block_size : int or 'auto'
        Variant tile width. None/auto derives it from the shared memory budget.

    Returns
    -------
    ndarray or pandas.DataFrame
        Variant-by-variant statistics, or a long table when requested.
        Undefined pairs and pairs with fewer than two joint calls are NaN.

    Complexity
    ----------
    Time O(n*p*q), space O(n*(p+q)+p*q+b^2) for n samples, p/q variants and
    tile width b. The dense result is materialized; use a windowed/virtual API
    for genome-wide LD.
    """
    from .gwas.ld import pairwise_complete
    res = pairwise_complete(X, Y, statistic=statistic, min_n=2, dtype=dtype,
                            nan_policy="pairwise", block_size=block_size)
    if not return_named:
        return res
    ix = X.columns if isinstance(X, pd.DataFrame) else np.arange(res.shape[0])
    other = X if Y is None else Y
    iy = other.columns if isinstance(other, pd.DataFrame) else np.arange(res.shape[1])
    frame = pd.DataFrame(res, index=ix, columns=iy)
    if return_square:
        return frame
    result = frame.rename_axis("bp1").reset_index().melt(id_vars="bp1", var_name="bp2")
    # Positional coordinates are optional: generic marker names remain valid.
    left = result.bp1.astype(str).str.rsplit(":", n=1, expand=True)
    right = result.bp2.astype(str).str.rsplit(":", n=1, expand=True)
    if left.shape[1] == 2 and right.shape[1] == 2:
        lp = pd.to_numeric(left[1], errors="coerce")
        rp = pd.to_numeric(right[1], errors="coerce")
        same = left[0].eq(right[0])
        result["c"] = left[0]
        result["distance"] = (lp-rp).abs().where(same)
    return result


@njit(parallel=True, fastmath=False)
def _cityblock_distance(X, Y, axis = 'rows', dtype = np.float32, scale = True):
    """Compute pairwise Manhattan distances over the requested row/column axis.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    axis : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``'rows'``. Source-recognized literal
        values: ``columns``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``np.full`` as ``dtype``.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``.
    """
    if axis == 'columns': 
        if Y is None: return _cityblock_distance(X.T, Y = None, axis = 'rows')
        return _cityblock_distance(X.T, Y.T, axis = 'rows')
    nx, n = X.shape
    if Y is not None:
        ny = Y.shape[0]
        out = np.full((nx, ny), np.nan, dtype=dtype)
        for i in prange(nx):
            for j in range(ny):
                s = 0.0
                c = 0
                for k in range(n):
                    a,b = X[i, k], Y[j, k]
                    if not np.isnan(a) and not np.isnan(b):
                        s += abs(a - b)
                        c += 1
                if c > 0: out[i, j] = s if not scale else s/c   
        return out
    out = np.full((nx, nx), np.nan, dtype=dtype)
    for i in prange(nx):
        for j in range(nx):
            if i>j: 
                s = 0.0
                c = 0
                for k in range(n):
                    a,b = X[i, k], X[j, k]
                    if not np.isnan(a) and not np.isnan(b):
                        s += abs(a - b)
                        c += 1
                if c > 0: 
                    out[i, j] = s if not scale else s/c  
                    out[j, i] = out[i, j] 
            if i == j: 
                out[i, j] = 0
    return out

def cityblock(X, Y=None,  axis = 'rows', dtype = np.float32, scale = True):
    """Return the cityblock result computed by ``pd.DataFrame``.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    axis : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``'rows'``. Source-recognized literal
        values: ``columns``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum. Passed to ``_cityblock_distance`` as
        ``axis``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``_cityblock_distance`` as
        ``dtype``.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``. Passed to ``_cityblock_distance`` as
        ``scale``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(_cityblock_distance(X.values if isinstance(X, pd.DataFrame) else X, Y.values if isinstance(Y, pd.DataFrame) else Y, axis=axis, dtype=dtype, scale=scale), index=rindex, columns=rcolumns)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if axis == 'columns':
        rindex = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.columns
        elif isinstance(Y, pd.DataFrame): rcolumns = Y.columns
        elif Y is None: rcolumns = list(range(X.shape[1]))
        else:  rcolumns = list(range(Y.shape[1]))
    if axis == 'rows':
        rindex = X.index if isinstance(X, pd.DataFrame) else list(range(X.shape[0]))
        if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.index
        elif isinstance(Y, pd.DataFrame): rcolumns = Y.index
        elif Y is None: rcolumns = list(range(X.shape[0]))
        else:  rcolumns = list(range(Y.shape[0]))
    return pd.DataFrame(_cityblock_distance(X.values if isinstance(X, pd.DataFrame) else X, 
                                            Y.values  if isinstance(Y, pd.DataFrame) else Y, 
                                            axis = axis, dtype = dtype, scale = scale), 
                        index = rindex, columns = rcolumns)







    

# ---------------------------------------------------------------------------
# VIGOR 2.11 packed-BED virtual/xarray fast paths
# ---------------------------------------------------------------------------
def load_plink_packed_xarray(prefix, chunk_variants='auto'):
    """Open PLINK1 BED as lazy uint8 SNP x sample-byte xarray data.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    chunk_variants : int or 'auto'
        Preferred variants per decode block. Default: ``'auto'`` (sized from the working-memory
        budget; integers are ceilings). Passed to ``open_packed_bed_xarray`` as
        ``chunk_variants``.

    Returns
    -------
    object
        Computed result from ``open_packed_bed_xarray(prefix, chunk_variants=chunk_variants)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .packed_bed import open_packed_bed_xarray
    return open_packed_bed_xarray(prefix,chunk_variants=chunk_variants)


def R2_packed(prefix, *, layout='virtual', max_offset=256, max_distance_bp=1_000_000,
              sample_mask=None, max_pairs=2_000_000):
    """Exact pairwise-observed R2 directly from packed BED without float decoding.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    layout : str or sequence
        LD layout: 'virtual'/'dense'/'matrix' gives full-coordinate LD;
        'offset'/'rolling'/'band'/'window' gives offset/banded LD.
        Default: ``'virtual'``.

    max_offset : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``256``. Passed to ``virtual_offset_r2`` as ``max_offset``.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``virtual_offset_r2`` as ``max_distance_bp``.

    sample_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value. Passed to ``virtual_ld_matrix`` as
        ``sample_mask``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``2000000``. Passed to ``virtual_ld_matrix`` as
        ``max_pairs``.

    Returns
    -------
    object
        Computed result from ``virtual_offset_r2(prefix, max_offset=max_offset, max_distance_bp=max_distance_bp, sample_mask=sample_mask, max_pairs=max_pairs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .packed_bed import virtual_ld_matrix, virtual_offset_r2
    key=str(layout).lower()
    if key in {'virtual','dense','matrix'}:
        return virtual_ld_matrix(prefix,sample_mask=sample_mask,max_pairs=max_pairs)
    if key in {'offset','rolling','band','window'}:
        return virtual_offset_r2(prefix,max_offset=max_offset,max_distance_bp=max_distance_bp,
                                 sample_mask=sample_mask,max_pairs=max_pairs)
    raise ValueError("layout must be 'virtual' or 'offset'")


def GRM_packed(prefix, *, component='additive', materialize=False, components=None,
               sample_ids=None, fit_mask=None, min_maf=1e-4, chunk_variants='auto',
               memory_fraction=None, maxmem=None, max_pairs=1_000_000,
               output_dtype=np.float32, blas_threads='auto', progress=False):
    """Virtual or exact in-memory VIGOR GRM from PLINK1 BED.

    ``materialize=False`` returns a full-coordinate virtual xarray DataArray.
    ``materialize=True`` builds exact dense A/D kernel(s) in RAM with adaptive
    genotype block sizing and no separate GRM file.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``. Passed to ``virtual_grm`` as ``component``.

    materialize : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``False``.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``dense_grm_from_bed`` as ``sample_ids``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to ``virtual_grm``
        as ``fit_mask``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``virtual_grm`` as ``min_maf``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``virtual_grm`` as ``chunk_variants``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to ``dense_grm_from_bed`` as
        ``memory_fraction``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``dense_grm_from_bed`` as
        ``maxmem``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``1000000``. Passed to ``virtual_grm`` as ``max_pairs``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``dense_grm_from_bed`` as
        ``output_dtype``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``dense_grm_from_bed`` as ``blas_threads``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``. Passed to ``dense_grm_from_bed`` as
        ``progress``.

    Returns
    -------
    object
        Computed result from ``dense_grm_from_bed(prefix, components=comps, sample_ids=sample_ids, fit_mask=fit_mask, min_maf=min_maf, chunk_variants=chunk_variants, memory_fraction=memory_fraction, maxmem=maxmem, output_dtype=o...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .packed_bed import virtual_grm, dense_grm_from_bed
    if not materialize:
        if sample_ids is not None: raise ValueError('virtual GRM currently exposes the full FAM sample axis; subset with xarray .sel/.isel')
        return virtual_grm(prefix,component=component,fit_mask=fit_mask,min_maf=min_maf,
                           chunk_variants='auto' if chunk_variants is None or str(chunk_variants).lower()=='auto' else int(chunk_variants),max_pairs=max_pairs)  # backend sizes 'auto' from the budget
    comps=(component,) if components is None else tuple(components)
    return dense_grm_from_bed(prefix,components=comps,sample_ids=sample_ids,fit_mask=fit_mask,
        min_maf=min_maf,chunk_variants=chunk_variants,memory_fraction=memory_fraction,maxmem=maxmem,
        output_dtype=output_dtype,blas_threads=blas_threads,progress=progress)
