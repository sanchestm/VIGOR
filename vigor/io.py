"""PLINK1 ``.bed/.bim/.fam`` I/O: the one BED opener, decoder and metadata parser.

Genotypes are decoded on demand from the memory-mapped ``.bed`` file in
blocks of variants, so the full genotype matrix is never materialised.

Allele convention (all VIGOR PLINK1 readers): the decoded dosage counts the
allele in BIM column 6 (``a1``, labelled ``effect_allele``); BIM column 5
(``a0``) is the ``other_allele``. BED codes 00/10/11 decode to 0/1/2 copies of
``effect_allele``; 01 is missing (NaN).
"""
from pathlib import Path

import numpy as np
import pandas as pd

#: PLINK1 SNP-major BED header.
BED_MAGIC = bytes((0x6C, 0x1B, 0x01))
#: FAM columns in file order.
FAM_COLUMNS = ["fid", "iid", "father", "mother", "sex", "pheno"]
#: BIM columns in file order; a0 = column 5 (other_allele), a1 = column 6 (effect_allele, counted).
BIM_COLUMNS = ["chrom", "snp", "cm", "pos", "a0", "a1"]

# BED 2-bit code -> dosage of BIM column 6 (uint8; 255 marks a missing call).
_BED_DOSAGE_U8 = np.array([0, 255, 1, 2], dtype=np.uint8)
# One BED byte -> its four sample dosages (256 x 4 uint8; first sample in the low bits).
_BED_BYTE_LUT_U8 = _BED_DOSAGE_U8[(np.arange(256, dtype=np.uint16)[:, None]
                                   >> (2 * np.arange(4, dtype=np.uint16))[None, :]) & 3]


def _strip_plink_suffix(prefix):
    s = str(prefix)
    for ext in (".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[:-len(ext)]
    return s


def _count_records(path):
    """Number of non-blank lines (records) of a FAM/BIM text file."""
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def open_bed_memmap(prefix, *, n_samples=None, n_variants=None, include_header=False):
    """Open a SNP-major PLINK1 BED payload as a validated read-only ``(m, bpv)`` uint8 memmap.

    Parameters
    ----------
    prefix : str or pathlib.Path
        Fileset prefix (a trailing ``.bed/.bim/.fam`` is stripped).
    n_samples, n_variants : int or None
        Nonnegative FAM/BIM record counts when already known; None counts the text records.
    include_header : bool
        False returns the (variant, packed byte) payload. True returns a flat
        entire-file mapping for legacy byte-offset readers, after identical validation.

    Returns
    -------
    numpy.memmap, shape (n_variants, ceil(n_samples / 4)), dtype uint8
        Header-free payload, one row of packed sample bytes per variant (an empty
        in-memory array when either axis is empty).

    Raises
    ------
    ValueError
        Bad magic bytes, individual-major BED, or a payload size that differs from
        ``3 + n_variants * ceil(n_samples / 4)`` bytes.
    """
    pre = _strip_plink_suffix(prefix)
    bed = Path(pre + ".bed")
    n = _count_records(pre + ".fam") if n_samples is None else int(n_samples)
    m = _count_records(pre + ".bim") if n_variants is None else int(n_variants)
    for label, value in (("n_samples", n_samples), ("n_variants", n_variants)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))):
            raise TypeError(f"{label} must be a nonnegative integer or None")
    if n < 0 or m < 0:
        raise ValueError("sample and variant counts must be nonnegative")
    bpv = (n + 3) // 4
    with open(bed, "rb") as fh:
        head = fh.read(3)
    if len(head) < 3 or head[:2] != BED_MAGIC[:2]:
        raise ValueError(f"{bed}: bad magic bytes (not a PLINK1 .bed)")
    if head[2] != BED_MAGIC[2]:
        raise ValueError(f"{bed}: only SNP-major .bed is supported")
    size = bed.stat().st_size
    expected = 3 + m * bpv
    if size != expected:
        raise ValueError(f"{bed}: {size} bytes, expected {expected} for {n} samples x {m} variants")
    if include_header:
        return np.memmap(bed, mode="r", dtype=np.uint8)
    if m == 0 or bpv == 0:
        return np.zeros((m, bpv), dtype=np.uint8)
    return np.memmap(bed, mode="r", dtype=np.uint8, offset=3, shape=(m, bpv))


def decode_bed(packed2d, n_full, sample_indices=None, dtype=np.float32, out=None):
    """Decode packed SNP-major BED rows to a ``sample x variant`` dosage matrix.

    All samples use the 256 x 4 byte lookup table; sample subsets shift and mask
    the gathered bytes. Temporaries are one byte per call (codes, missing mask);
    the floating output is the only full-size allocation (or ``out``).

    Parameters
    ----------
    packed2d : array-like of uint8, shape (v, ceil(n_full / 4)) or flat
        Header-free packed rows (memmap slices stay mapped; a flat buffer is
        reshaped to whole variants).
    n_full : int
        Number of samples in the source FAM (sets bytes per variant).
    sample_indices : one-dimensional integer array or None
        Source sample positions in output order (duplicates kept); None decodes all.
    dtype : NumPy floating dtype
        Output precision. Default ``np.float32``.
    out : numpy.ndarray or None
        Optional preallocated ``(s, v)`` floating output.

    Returns
    -------
    numpy.ndarray, shape (s, v)
        Copies of the BIM column-6 allele (``effect_allele``) 0/1/2, NaN missing.
    """
    if isinstance(n_full, (bool, np.bool_)) or not isinstance(n_full, (int, np.integer)):
        raise TypeError("n_full must be a nonnegative integer")
    n_full = int(n_full)
    if n_full < 0:
        raise ValueError("n_full must be nonnegative")
    bpv = (n_full + 3) // 4
    P = np.asarray(packed2d)
    if P.dtype != np.uint8:
        raise TypeError("packed BED rows must have dtype uint8")
    if P.ndim == 1:
        if bpv == 0 or P.size % bpv:
            raise ValueError("flat BED buffer is not a whole number of variants")
        P = P.reshape(-1, bpv)
    if P.ndim != 2 or P.shape[1] != bpv:
        raise ValueError(f"packed BED rows must have {bpv} bytes per variant")
    v = P.shape[0]
    dtype = np.dtype(dtype if out is None else out.dtype)
    if dtype.kind != "f":
        raise TypeError("decode_bed needs a floating dtype to represent missing calls as NaN")
    si = None
    if sample_indices is None:
        s = n_full
    else:
        si = np.asarray(sample_indices)
        if si.ndim != 1:
            raise ValueError("sample_indices must be one-dimensional")
        if si.dtype.kind not in "iu":
            raise TypeError("sample_indices must be integer positions, not floats or masks")
        si = si.astype(np.intp, copy=False)
        s = si.size
        if s and (si.min() < 0 or si.max() >= n_full):
            raise IndexError("sample_indices out of bounds")
    if out is None:
        out = np.empty((s, v), dtype=dtype)
    elif not isinstance(out, np.ndarray) or not out.flags.writeable:
        raise TypeError("out must be a writable NumPy array")
    elif out.shape != (s, v):
        raise ValueError(f"out must have shape {(s, v)}")
    if s == 0 or v == 0:
        return out
    if si is None:
        codes = np.empty((v, bpv, 4), dtype=np.uint8)
        rows = max(1, (1 << 21) // bpv)  # bounds the intp index copy numpy makes for the gather
        for r0 in range(0, v, rows):
            r1 = min(v, r0 + rows)
            np.take(_BED_BYTE_LUT_U8, P[r0:r1], axis=0, out=codes[r0:r1], mode="clip")
        dos = codes.reshape(v, 4 * bpv)[:, :n_full]
        missing = dos == 255
    else:
        dos = P[:, si >> 2]
        dos >>= ((si & 3) << 1).astype(np.uint8)
        dos &= np.uint8(3)
        missing = dos == 1
        np.subtract(dos, dos > 1, out=dos, casting="unsafe")  # BED 10 -> 1, 11 -> 2
    out[...] = dos.T
    np.copyto(out, np.nan, where=missing.T)
    return out


def plink1_metadata_stamp(prefix):
    """Size/mtime stamp of the authoritative FAM/BIM text files."""
    pre = _strip_plink_suffix(prefix)
    fs = Path(pre + ".fam").stat()
    bs = Path(pre + ".bim").stat()
    return {"fam_size": int(fs.st_size), "bim_size": int(bs.st_size),
            "fam_mtime_ns": int(fs.st_mtime_ns), "bim_mtime_ns": int(bs.st_mtime_ns)}


def plink1_metadata_cache_valid(prefix):
    """True when ``.fam.parquet``/``.bim.parquet`` sidecars match the ``.plinkmeta.json`` stamp."""
    import json
    pre = _strip_plink_suffix(prefix)
    paths = [Path(pre + ext) for ext in (".fam.parquet", ".bim.parquet", ".plinkmeta.json")]
    if not all(p.exists() for p in paths):
        return False
    try:
        meta = json.loads(paths[2].read_text())
        stamp = plink1_metadata_stamp(pre)
        return all(int(meta.get(k, -1)) == v for k, v in stamp.items())
    except Exception:
        return False


def _read_cached_table(pre, ext):
    if not plink1_metadata_cache_valid(pre):
        return None
    try:
        return pd.read_parquet(pre + ext)
    except Exception:
        return None


def _check_fam(fam):
    iid = fam["iid"].astype(str)
    if iid.duplicated().any():
        dup = fam.loc[iid.duplicated(keep=False), "iid"].astype(str).unique()[:5]
        raise ValueError(f"duplicate IID(s) in FAM: {list(dup)}; VIGOR aligns by IID, so use unique IIDs or precompose FID:IID")
    if fam[["fid", "iid"]].astype(str).duplicated().any():
        raise ValueError("duplicate (FID, IID) rows in FAM")
    return fam


def read_bim(prefix):
    """Read PLINK1 variant metadata in physical BED-column order.

    A validated Parquet sidecar (``.bim.parquet`` with a matching
    ``.plinkmeta.json`` size/mtime stamp) is used when present; otherwise the
    text BIM is parsed.

    Parameters
    ----------
    prefix : str or pathlib.Path
        Fileset prefix without suffix.

    Returns
    -------
    pandas.DataFrame, shape (p, 8)
        ``chrom, snp, cm, pos, a0, a1`` in file order plus ``other_allele`` (= a0,
        column 5) and ``effect_allele`` (= a1, column 6, the allele every VIGOR
        PLINK1 dosage counts). chrom/snp/alleles are strings, cm float, pos int.
    """
    pre = _strip_plink_suffix(prefix)
    bim = _read_cached_table(pre, ".bim.parquet")
    if bim is not None:
        bim = bim.drop(columns=[c for c in ("i",) if c in bim.columns])
        if list(bim.columns[:6]) != BIM_COLUMNS:
            bim = None
    if bim is None:
        bim = pd.read_csv(pre + ".bim", sep=r"\s+", header=None, names=BIM_COLUMNS, engine="c",
                          dtype={"chrom": str, "snp": str, "cm": float, "pos": np.int64, "a0": str, "a1": str})
    bim = bim[BIM_COLUMNS].reset_index(drop=True)
    bim["other_allele"] = bim["a0"]
    bim["effect_allele"] = bim["a1"]
    return bim


def read_fam(prefix):
    """Read PLINK1 sample metadata and reject ambiguous IID-based alignment.

    Parameters
    ----------
    prefix : str or pathlib.Path
        Fileset prefix; a validated ``.fam.parquet`` sidecar is used when present.

    Returns
    -------
    pandas.DataFrame, shape (n, 6)
        ``fid, iid, father, mother, sex, pheno`` in source order; first four are strings.

    Raises
    ------
    ValueError
        Repeated IID strings (even across FIDs) or repeated FID/IID pairs.
    """
    pre = _strip_plink_suffix(prefix)
    fam = _read_cached_table(pre, ".fam.parquet")
    if fam is not None:
        fam = fam.rename(columns={"gender": "sex", "trait": "pheno"})
        if list(fam.columns[:6]) != FAM_COLUMNS:
            fam = None
    if fam is None:
        fam = pd.read_csv(pre + ".fam", sep=r"\s+", header=None, names=FAM_COLUMNS, engine="c",
                          dtype={"fid": str, "iid": str, "father": str, "mother": str})
    return _check_fam(fam[FAM_COLUMNS].reset_index(drop=True))


def read_plink1_metadata(prefix):
    """Return ``(fam, bim)`` from :func:`read_fam` and :func:`read_bim`."""
    return read_fam(prefix), read_bim(prefix)


class PlinkReader:
    """Memory-map SNP-major PLINK1 calls with bounded dosage block decoding.
    
    The reader exposes fam/bim metadata in physical file order. Missing BED calls
    are NaN; codes decode to 0/NaN/1/2 and retain BIM allele convention. Use a context
    manager or close explicitly; decoded blocks own their memory.
    
    Parameters
    ----------
    prefix : str or pathlib.Path
        Common BED/BIM/FAM prefix without extension.
    dtype : NumPy floating dtype
        Default float32 for decoded blocks; must represent NaN.
    
    Notes
    -----
    Constructor TIME O(B+F), SPACE O(B+F) for eager BIM/FAM bytes; BED is read-only
    mapped through :func:`open_bed_memmap` (magic, SNP-major mode and payload size are
    validated). Dosages count BIM column 6 (``effect_allele``). Block costs are
    documented on methods. Exposes no write operation.
    """

    def __init__(self, prefix, dtype=np.float32):
        """Open metadata and read-only BED mapping; validate PLINK magic/mode.
        
        Parameters
        ----------
        prefix : str or pathlib.Path
            Fileset prefix containing compatible .fam,.bim,.bed files.
        dtype : NumPy floating dtype
            Default: ``np.float32``; applied on decoded output only.
        
        Returns
        -------
        None
            Sets prefix,fam,bim,n_samples,n_variants,dtype and the BED mapping.
        
        Raises
        ------
        ValueError
            Invalid magic, individual-major BED, a payload size inconsistent with
            FAM/BIM, or duplicate FAM IIDs.
        OSError
            Missing files.
        
        Complexity
        ----------
        TIME O(B+F); SPACE O(B+F) for metadata bytes B/F, plus virtual BED mapping.
        """
        self.prefix = prefix
        self.fam = read_fam(prefix)
        self.bim = read_bim(prefix)
        self.n_samples = len(self.fam)
        self.n_variants = len(self.bim)
        self.dtype = dtype
        self._bpv = (self.n_samples + 3) // 4
        # Validated (n_variants, bytes_per_variant) payload view; magic, mode and size checked.
        self._bed = open_bed_memmap(prefix, n_samples=self.n_samples, n_variants=self.n_variants)

    def read_block(self, start, count):
        """Decode a contiguous variant span for every source sample.
        
        Parameters
        ----------
        start : int
            Zero-based first variant; expected 0<=start<=n_variants.
        count : int
            Number of variants; expected nonnegative and start+count<=n_variants.
            Bounds are not checked explicitly.
        
        Returns
        -------
        numpy.ndarray, shape (n_samples,count), dtype self.dtype
            Owned C-contiguous dosage matrix; NaN denotes BED missing code. Sample
            padding in the final packed byte is discarded. count=0 yields zero columns.
        
        Raises
        ------
        ValueError
            Truncated payload/bounds prevent reshape or dtype cannot retain values.
        
        Complexity
        ----------
        TIME O(nc); SPACE O(nc) output plus one-byte code/missing temporaries, n
        source samples and c requested variants.
        """
        start = int(start)
        return decode_bed(self._bed[start:start + int(count)], self.n_samples, dtype=self.dtype)

    def read_block_samples(self, start, count, sample_index):
        """Decode selected samples directly from a contiguous packed variant span.
        
        Parameters
        ----------
        start, count : int
            Zero-based start and number of variants within the BED payload; variant
            bounds are assumed. count=0 and empty sample selections are supported.
        sample_index : one-dimensional integer array-like, shape (s,)
            Physical FAM row positions, in desired output order; duplicates are allowed.
            Converted to int64; nonintegral floats would be truncated, so pass integers.
        
        Returns
        -------
        numpy.ndarray, shape (s,count), dtype self.dtype
            Owned C-contiguous dosages with NaNs, preserving selected order/duplicates.
        
        Raises
        ------
        IndexError
            A sample position is negative or >=n_samples.
        ValueError
            Variant payload cannot reshape or sample indices cannot convert.
        
        Complexity
        ----------
        TIME O(sc); SPACE O(sc) output plus one-byte code/missing temporaries,
        beyond a read-only packed span view of O(c ceil(n/4)) virtual bytes.
        """
        sidx = np.asarray(sample_index, dtype=np.int64)
        if np.any(sidx < 0) or np.any(sidx >= self.n_samples):
            raise IndexError("sample_index is outside the PLINK sample range")
        start = int(start)
        return decode_bed(self._bed[start:start + int(count)], self.n_samples,
                          sample_indices=sidx, dtype=self.dtype)

    def iter_blocks(self, block_size='auto', variant_index=None, sample_index=None,
                    maxmem=None, memory_fraction=None):
        """Yield metadata and dosage blocks for sorted source variant positions.

        Parameters
        ----------
        block_size : int or 'auto'
            Default 'auto' (sized from the memory budget), else a positive ceiling on
            variants per yielded block; still clipped to what the budget affords.
        variant_index : one-dimensional sorted integer array-like or None
            Default None visits all variants. Sorted positions may repeat. Bounds and
            integrality are assumed. Empty selection yields nothing. A selected block
            is decoded from one gather of its packed rows (a mapped slice when the
            positions are contiguous), so memory follows the block, not its span.
        sample_index : one-dimensional integer array-like or None
            Default None decodes all FAM samples; otherwise delegates selected positions
            to read_block_samples. Output preserves requested order.
        maxmem, memory_fraction : budget or None
            Default None; working-memory budget for block_size, see
            resources.working_memory_bytes (None fraction means 80% of RAM).

        Yields
        ------
        tuple
            (BIM DataFrame with original row index, dosage ndarray (s,c)), c<=block_size.
            Blocks are owned arrays; retaining them all forfeits bounded memory.
        
        Raises
        ------
        ValueError
            Unsorted/non-vector variants, zero block size or malformed BED span.
        IndexError
            Sample/metadata positions are out of bounds. Non-positive block_size
            raises ValueError; MemoryError if one variant exceeds the budget.
        
        Complexity
        ----------
        For p requested variants and s retained samples: TIME O(sp+q); SPACE O(sb)
        for block size b and q index/metadata work. Source BIM/FAM storage remains
        O(B+F); OS paging cost depends on physical locality.
        """
        from .resources import resolve_variant_chunk
        it = np.dtype(self.dtype).itemsize
        # decode scratch per call: output (itemsize) + caller copy (itemsize) + one-byte
        # codes and missing mask + gathered packed rows (1/4 byte)
        block_size = resolve_variant_chunk(
            block_size, self.n_samples if sample_index is None else len(sample_index),
            bytes_per_call=2 * it + 3, maxmem=maxmem, memory_fraction=memory_fraction,
            n_variants=self.n_variants if variant_index is None else len(variant_index))
        sidx = None if sample_index is None else np.asarray(sample_index, dtype=np.int64)
        if variant_index is None:
            for s in range(0, self.n_variants, block_size):
                c = min(block_size, self.n_variants - s)
                yield self.bim.iloc[s:s + c], decode_bed(self._bed[s:s + c], self.n_samples,
                                                         sample_indices=sidx, dtype=self.dtype)
        else:
            variant_index = np.asarray(variant_index, dtype=np.int64)
            if variant_index.ndim != 1 or (len(variant_index) and np.any(np.diff(variant_index) < 0)):
                raise ValueError("variant_index must be a sorted one-dimensional index")
            for s in range(0, len(variant_index), block_size):
                idx = variant_index[s:s + block_size]
                lo, hi = int(idx[0]), int(idx[-1])
                # One row gather of the selected packed variants (a mapped slice when contiguous).
                packed = self._bed[lo:hi + 1] if hi - lo + 1 == len(idx) else self._bed[idx]
                yield self.bim.iloc[idx], decode_bed(packed, self.n_samples, sample_indices=sidx,
                                                     dtype=self.dtype)

    def chrom_variant_index(self, chrom):
        """Find physical variant positions with an exact chromosome-string match.
        
        Parameters
        ----------
        chrom : str or int
            Converted to str; no chr-prefix normalization is performed.
        
        Returns
        -------
        numpy.ndarray of int, shape (k,)
            Increasing zero-based BIM row positions; empty when the label is absent.
        
        Complexity
        ----------
        TIME O(p); SPACE O(p+k), with p metadata rows and k matches.
        """
        return np.where(self.bim["chrom"].values == str(chrom))[0]

    @property
    def chroms(self):
        """Return first-seen distinct BIM chromosome strings.
        
        Returns
        -------
        list of str
            Physical metadata appearance order, not numeric chromosome sort order.
        
        Complexity
        ----------
        TIME O(p); SPACE O(k), p variants and k distinct chromosomes (hashing assumed).
        """
        return list(dict.fromkeys(self.bim["chrom"].values))

    def close(self):
        """Release the BED mapping, retaining already decoded arrays and metadata.
        
        Returns
        -------
        None
            Clears the internal BED reference. Repeated calls are harmless; reading
            afterward is unsupported. External views into the mapping must not be used.
        
        Complexity
        ----------
        TIME O(1) Python work plus OS unmapping cost dependent on mapped pages;
        SPACE O(1) additional memory. This is a read-only mapping, with no data flush.
        """
        bed = getattr(self, "_bed", None)
        mmap = getattr(bed, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._bed = None

    def __enter__(self):
        """Return this opened reader for a with statement.
        
        Returns
        -------
        PlinkReader
            The same instance; no reopen or copy occurs.
        
        Complexity
        ----------
        TIME O(1); SPACE O(1).
        """
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Close the mapped file on context exit and propagate any exception.
        
        Parameters
        ----------
        exc_type, exc, traceback : exception type, exception, traceback or None
            Standard context manager exception triplet; accepted and unused.
        
        Returns
        -------
        bool
            Always False, so exceptions from the context body propagate.
        
        Complexity
        ----------
        TIME O(1) Python work plus PlinkReader.close OS unmapping; SPACE O(1).
        """
        self.close()
        return False

    def __del__(self):
        """Release an extant BED mapping during finalization.
        
        Returns
        -------
        None
            Delegates to close; explicit context management is more deterministic.
        
        Complexity
        ----------
        TIME O(1) Python work plus close OS unmapping; SPACE O(1).
        """
        self.close()
