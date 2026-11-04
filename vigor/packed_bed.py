"""Bit-packed PLINK1 BED operations with virtual xarray LD views.

The BED payload remains uint8 (SNP x sample-byte). Pairwise operations reduce
2-bit genotype codes directly. Float dosage decoding is an explicit derived
view for algorithms that actually need sample x SNP arithmetic.
"""
from __future__ import annotations
from .io import open_bed_memmap
from functools import lru_cache
import numpy as np


def _require_xarray():
    """Import and verify availability of xarray.
    """
    import xarray as xr
    from xarray.backends import BackendArray
    from xarray.core import indexing
    return xr, BackendArray, indexing


@lru_cache(maxsize=8)
def _metadata_cached(prefix, fam_size, fam_mtime_ns, bim_size, bim_mtime_ns):
    """Cache validated by the authoritative FAM/BIM file identity.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    fam_size : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    fam_mtime_ns : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    bim_size : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    bim_mtime_ns : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.
    """
    from .io import read_fam, read_bim
    fam = read_fam(prefix).rename(columns={'sex': 'gender', 'pheno': 'trait'})
    bim = read_bim(prefix)
    bim['i'] = np.arange(len(bim), dtype=np.int64)
    return fam, bim

def _metadata(prefix):
    """Return the metadata result computed by ``_metadata_cached``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.
    """
    from pathlib import Path
    pre=str(Path(prefix).expanduser().resolve()); fs=Path(pre+'.fam').stat(); bs=Path(pre+'.bim').stat()
    return _metadata_cached(pre,int(fs.st_size),int(fs.st_mtime_ns),int(bs.st_size),int(bs.st_mtime_ns))


@lru_cache(maxsize=4)
def _compact_metadata_cached(prefix, fam_size, fam_mtime_ns, bim_size, bim_mtime_ns):
    """Read only packed-view coordinates into compact fixed-width NumPy arrays.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    fam_size : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    fam_mtime_ns : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    bim_size : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.

    bim_mtime_ns : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.
    """
    from pathlib import Path
    pre=str(prefix); fp=Path(pre+'.fam.parquet'); bp=Path(pre+'.bim.parquet')
    from .io import plink1_metadata_cache_valid
    if plink1_metadata_cache_valid(pre):
        try:
            import pyarrow.parquet as pq
            def read_cols(path, columns, text_cols):
                """Read selected Parquet columns in bounded row batches with string/numeric conversion.

                Parameters
                ----------
                path : str or os.PathLike
                    Filesystem location read or written by this routine. Use a path to the documented
                    file/store format; destination parents and overwrite behavior follow the writer's
                    contract.

                columns : sequence or pandas.Index
                    Trait labels in matrix-column order, used to label or align the resulting trait
                    quantities. Passed to ``pf.iter_batches`` as ``columns``.

                text_cols : object (see the cited input contract)
                    Input supplied to ``pf.iter_batches`` as ``columns``. See that callable's documented
                    ``columns`` contract for its type, alignment and value domain. This wrapper retains that
                    contract. Passed to ``pf.iter_batches`` as ``columns``.

                Returns
                -------
                object
                    Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
                """
                pf=pq.ParquetFile(path); n=pf.metadata.num_rows
                widths={c:1 for c in text_cols}
                for batch in pf.iter_batches(batch_size=65536,columns=list(text_cols)):
                    for j,c in enumerate(text_cols):
                        vals=batch.column(j).to_pylist()
                        if vals: widths[c]=max(widths[c],max(len(str(x)) for x in vals if x is not None))
                out={c:np.empty(n,dtype=f'<U{widths[c]}') for c in text_cols}
                for c in columns:
                    if c not in text_cols: out[c]=np.empty(n,dtype=np.int64)
                off=0
                for batch in pf.iter_batches(batch_size=65536,columns=columns):
                    m=batch.num_rows
                    for j,c in enumerate(columns):
                        if c in text_cols: out[c][off:off+m]=np.asarray(batch.column(j).to_pylist(),dtype=out[c].dtype)
                        else: out[c][off:off+m]=np.asarray(batch.column(j),dtype=np.int64)
                    off+=m
                return out
            fam=read_cols(fp,['iid'],{'iid'}); bim=read_cols(bp,['snp','chrom','pos'],{'snp','chrom'})
            return fam['iid'],bim['snp'],bim['chrom'],bim['pos']
        except Exception:
            pass

    # Text fallback is chunked so only a bounded pandas object table is live.
    import pandas as pd
    fam_parts=[]
    for q in pd.read_csv(pre+'.fam',sep=r'\s+',header=None,usecols=[1],names=['iid'],dtype=str,chunksize=131072):
        vals=q.iid.astype(str).tolist(); w=max([1]+[len(x) for x in vals]); fam_parts.append(np.asarray(vals,dtype=f'<U{w}'))
    snp_parts=[]; chr_parts=[]; pos_parts=[]
    for q in pd.read_csv(pre+'.bim',sep=r'\s+',header=None,usecols=[0,1,3],names=['chrom','snp','pos'],
                         dtype={'chrom':str,'snp':str,'pos':np.int64},chunksize=131072):
        sv=q.snp.astype(str).tolist(); cv=q.chrom.astype(str).tolist()
        sw=max([1]+[len(x) for x in sv]); cw=max([1]+[len(x) for x in cv])
        snp_parts.append(np.asarray(sv,dtype=f'<U{sw}')); chr_parts.append(np.asarray(cv,dtype=f'<U{cw}')); pos_parts.append(q.pos.to_numpy(np.int64))
    return np.concatenate(fam_parts),np.concatenate(snp_parts),np.concatenate(chr_parts),np.concatenate(pos_parts)


def _compact_metadata(prefix):
    """Return the compact metadata result computed by ``_compact_metadata_cached``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.
    """
    from pathlib import Path
    pre=str(Path(prefix).expanduser().resolve()); fs=Path(pre+'.fam').stat(); bs=Path(pre+'.bim').stat()
    return _compact_metadata_cached(pre,int(fs.st_size),int(fs.st_mtime_ns),int(bs.st_size),int(bs.st_mtime_ns))


@lru_cache(maxsize=4)
def _lean_metadata_cached(prefix, fam_size, fam_mtime_ns, bim_size, bim_mtime_ns, bed_size, bed_mtime_ns):
    """Packed-view metadata without materializing genome-wide SNP ID strings.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    fam_size : mapping
        Input for the operation ``int(z.get('fam_size', -1)) == fam_size``. See
        ``vigor.packed_bed._lean_metadata_cached`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    fam_mtime_ns : mapping
        Input for the operation ``int(z.get('fam_mtime_ns', -1)) == fam_mtime_ns``. See
        ``vigor.packed_bed._lean_metadata_cached`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    bim_size : mapping
        Input for the operation ``int(z.get('bim_size', -1)) == bim_size``. See
        ``vigor.packed_bed._lean_metadata_cached`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    bim_mtime_ns : mapping
        Input for the operation ``int(z.get('bim_mtime_ns', -1)) == bim_mtime_ns``. See
        ``vigor.packed_bed._lean_metadata_cached`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    bed_size : object (see the cited input contract)
        Input for the operation ``(bed_size - 3) // bpv``. See
        ``vigor.packed_bed._lean_metadata_cached`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    bed_mtime_ns : object (see the cited input contract)
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type.
    """
    from pathlib import Path
    import json
    pre=str(prefix); fp=Path(pre+'.fam.parquet'); bp=Path(pre+'.bim.parquet'); mp=Path(pre+'.plinkmeta.json')
    valid_cache=False
    if fp.exists() and bp.exists() and mp.exists():
        try:
            z=json.loads(mp.read_text())
            valid_cache=(int(z.get('fam_size',-1))==fam_size and int(z.get('fam_mtime_ns',-1))==fam_mtime_ns and
                         int(z.get('bim_size',-1))==bim_size and int(z.get('bim_mtime_ns',-1))==bim_mtime_ns)
        except Exception: valid_cache=False
    if valid_cache:
        import pyarrow.compute as pc, pyarrow.parquet as pq
        ftab=pq.read_table(fp,columns=['iid']); vals=ftab.column('iid').combine_chunks().to_pylist()
        iw=max([1]+[len(str(x)) for x in vals]); ids=np.asarray(vals,dtype=f'<U{iw}')
        pf=pq.ParquetFile(bp); m=int(pf.metadata.num_rows); pos=np.empty(m,np.int64); codes=np.empty(m,np.int16)
        levels=[]; lut={}; off=0
        for batch in pf.iter_batches(batch_size=131072,columns=['chrom','pos']):
            enc=pc.dictionary_encode(batch.column(0)); labs=[str(x) for x in enc.dictionary.to_pylist()]
            remap=np.empty(len(labs),np.int16)
            for j,lab in enumerate(labs):
                if lab not in lut: lut[lab]=len(levels); levels.append(lab)
                remap[j]=lut[lab]
            q=np.asarray(enc.indices,dtype=np.int64); n=batch.num_rows
            codes[off:off+n]=remap[q]; pos[off:off+n]=np.asarray(batch.column(1),dtype=np.int64); off+=n
        return ids,codes,tuple(levels),pos
    import pandas as pd
    f=pd.read_csv(pre+'.fam',sep=r'\s+',header=None,usecols=[1],dtype=str); vals=f.iloc[:,0].astype(str).tolist()
    iw=max([1]+[len(x) for x in vals]); ids=np.asarray(vals,dtype=f'<U{iw}')
    from .io import _count_records
    m = _count_records(pre+'.bim')
    pos=np.empty(m,np.int64); codes=np.empty(m,np.int16)
    levels=[]; lut={}; off=0
    for q in pd.read_csv(pre+'.bim',sep=r'\s+',header=None,usecols=[0,3],dtype={0:str,3:np.int64},chunksize=131072):
        cv=q.iloc[:,0].astype(str).to_numpy(); n=len(q); local=np.empty(n,np.int16)
        for lab in np.unique(cv):
            if lab not in lut: lut[lab]=len(levels); levels.append(lab)
            local[cv==lab]=lut[lab]
        codes[off:off+n]=local; pos[off:off+n]=q.iloc[:,1].to_numpy(np.int64); off+=n
    return ids,codes,tuple(levels),pos


def _lean_metadata(prefix):
    """Return the lean metadata result computed by ``_lean_metadata_cached``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.
    """
    from pathlib import Path
    pre=str(Path(prefix).expanduser().resolve()); fs=Path(pre+'.fam').stat(); bs=Path(pre+'.bim').stat(); ds=Path(pre+'.bed').stat()
    return _lean_metadata_cached(pre,int(fs.st_size),int(fs.st_mtime_ns),int(bs.st_size),int(bs.st_mtime_ns),int(ds.st_size),int(ds.st_mtime_ns))

def _packed_coordinate_metadata(prefix, label_threshold=500_000):
    """Return compact packed-view coordinates; genome-wide SNP IDs stay on disk for large files.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    label_threshold : int
        Input for the operation ``m <= int(label_threshold)``. See
        ``vigor.packed_bed._packed_coordinate_metadata`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``500000``.
    """
    ids,codes,levels,pos=_lean_metadata(prefix); m=len(pos)
    lw=max([1]+[len(str(x)) for x in levels]); lev=np.asarray(levels,dtype=f'<U{lw}')
    chrom=lev[np.asarray(codes,np.int64)]
    labels_loaded=bool(m<=int(label_threshold))
    if labels_loaded:
        _,names,_,_=_compact_metadata(prefix)
    else:
        names=np.arange(m,dtype=np.int64)
    return ids,names,chrom,pos,labels_loaded


def sample_mask4(n_samples, sample_mask=None):
    """One 4-bit inclusion mask per BED byte, including final-byte padding.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    sample_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    n=int(n_samples); nb=(n+3)//4
    keep=np.ones(n,bool) if sample_mask is None else np.asarray(sample_mask,bool)
    if keep.shape != (n,): raise ValueError("sample_mask must have length n_samples")
    out=np.zeros(nb,np.uint8)
    for i in np.flatnonzero(keep): out[i >> 2] |= np.uint8(1 << (i & 3))
    return out




def _as_indices(key, size):
    """Convert inputs to indices.

    Parameters
    ----------
    key : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    size : array-like or numeric scalar
        Input for the operation ``np.arange(*key.indices(size), dtype=np.int64)``. See
        ``vigor.packed_bed._as_indices`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.
    """
    if isinstance(key,slice):
        return np.arange(*key.indices(size),dtype=np.int64), False
    a=np.asarray(key)
    if a.ndim==0: return np.asarray([int(a)],np.int64), True
    if a.dtype==bool: a=np.flatnonzero(a)
    return np.asarray(a,np.int64).reshape(-1), False



@lru_cache(maxsize=1)
def bed_category_mask_lut():
    """Per BED byte, return 4-bit masks for observed, dosage==1 and dosage==2.

    Returns
    -------
    object
        Computed result from ``(observed, g1, g2)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    b=np.arange(256,dtype=np.uint16)[:,None]
    codes=(b >> (2*np.arange(4,dtype=np.uint8))[None,:]) & 3
    bits=(1 << np.arange(4,dtype=np.uint8))[None,:]
    observed=np.sum(np.where(codes!=1,bits,0),axis=1,dtype=np.uint16).astype(np.uint8)
    g1=np.sum(np.where(codes==2,bits,0),axis=1,dtype=np.uint16).astype(np.uint8)
    g2=np.sum(np.where(codes==3,bits,0),axis=1,dtype=np.uint16).astype(np.uint8)
    return observed,g1,g2





def _decode_masked_packed_variants(P,m4):
    """Decode only sample lanes selected by mask4 for a retained SNP block.

    Parameters
    ----------
    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    m4 : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.
    """
    from .io import decode_bed
    P = np.asarray(P)
    m4 = np.asarray(m4, np.uint8)
    if m4.ndim != 1 or P.ndim != 2 or P.shape[1] != len(m4) or np.any(m4 > 15):
        raise ValueError("mask4 must specify four sample lanes per packed byte")
    lanes = np.arange(4 * len(m4), dtype=np.int64)
    keep = ((m4[:, None] >> np.arange(4, dtype=np.uint8)[None, :]) & 1).reshape(-1).astype(bool)
    # Source geometry here includes padded lanes; mask4 excludes source padding.
    return decode_bed(P, 4 * len(m4), lanes[keep], dtype=np.float32)


def _packed_r2_blas_exact(A,B,m4):
    """Exact pairwise-complete R2 after decoding only the requested packed block.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``_decode_masked_packed_variants(A, m4)``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state.

    B : array-like
        Numerical operand in ``A.shape == B.shape``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    m4 : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.
    """
    from .gwas.ld import pairwise_complete
    X = _decode_masked_packed_variants(A, m4)
    Y = None if A is B else _decode_masked_packed_variants(B, m4)
    return pairwise_complete(X, Y, statistic="r2", min_n=3, dtype=np.float32,
                             block_size="auto")


def _packed_r2_blas_mean(A,B,m4):
    """Fast dense R2 after per-SNP mean imputation; intended for local fine mapping.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``_decode_masked_packed_variants(A, m4).astype(np.float32,
        copy=False)``. Shape must conform to that matrix/vector operation and the enclosing
        prepared state.

    B : array-like
        Numerical operand in ``A.shape == B.shape``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    m4 : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.
    """
    from .gwas.ld import pairwise_complete
    X = _decode_masked_packed_variants(A, m4)
    Y = None if A is B else _decode_masked_packed_variants(B, m4)
    return pairwise_complete(X, Y, statistic="r2", nan_policy="mean", min_n=3, dtype=np.float32)


def _packed_r2_selected(packed_i, packed_j, mask4, engine='auto', missing='pairwise'):
    """Exact pairwise-complete R2 on packed BED calls.

    All engine spellings are compatibility aliases for the shared centered-block
    LD kernel after bounded selected-block decoding. No genome-wide genotype
    matrix is decoded; select a local view before materializing.

    Parameters
    ----------
    packed_i : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    packed_j : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    mask4 : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'auto'``. Source-recognized literal
        values: ``blas``, ``compiled``, ``numba``, ``numpy``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.

    missing : str
        Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
        size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
        'impute' treats missing values as latent, and 'row' uses complete rows where supported.
        'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
        it. See the local validator for the accepted subset. Default: ``'pairwise'``.
    """
    A = np.asarray(packed_i)
    B = np.asarray(packed_j)
    m4 = np.asarray(mask4, np.uint8)
    e = str(engine).lower()
    if e not in {"auto", "compiled", "numba", "numpy", "blas"}:
        raise ValueError("unknown LD engine")
    miss = str(missing).lower()
    if miss not in {"pairwise", "mean", "mean_impute", "impute"}:
        raise ValueError("missing must be 'pairwise' or 'mean'")
    # Old engine spellings remain adapter aliases. All use the same validated
    # decoder and centered-block floating kernel; there is no separate estimator.
    if miss != "pairwise":
        return _packed_r2_blas_mean(A, B, m4)
    return _packed_r2_blas_exact(A, B, m4)

try:
    from xarray.backends import BackendArray
    from xarray.core import indexing as _xr_indexing
except Exception:  # module remains importable without xarray
    BackendArray=object; _xr_indexing=None


class PackedBedByteBackend(BackendArray):
    """Lazy uint8 BED payload; only requested variant/sample-byte blocks are read."""
    __slots__=("prefix","n_variants","bytes_per_variant","_packed")
    def __init__(self,prefix,n_variants,bytes_per_variant):
        """Initialize PackedBedByteBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_variants : int
            Number of variants in the logical source axis, including the unpadded extent of packed
            arrays.

        bytes_per_variant : int
            Byte cost for one decoded call, per-variant overhead, work-buffer multiplier or storage
            extent as named. Allocation planners multiply these costs by the relevant dimensions;
            use nonnegative costs consistent with the actual representation.
        """
        self.prefix=str(prefix); self.n_variants=int(n_variants); self.bytes_per_variant=int(bytes_per_variant)
        self._packed=open_bed_memmap(self.prefix+'.bed', n_variants=self.n_variants)
    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(self.n_variants, self.bytes_per_variant)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.n_variants,self.bytes_per_variant)
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.uint8)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.uint8)
    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _xr_indexing.explicit_indexing_adapter(key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)
    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],self.n_variants); jj,sj=_as_indices(key[1],self.bytes_per_variant)
        out=np.asarray(self._packed[np.ix_(ii,jj)],np.uint8)
        if si and sj:return np.asarray(out[0,0],np.uint8)
        if si:return out[0]
        if sj:return out[:,0]
        return out
class PackedBedLDBackend(BackendArray):
    """Virtual m x m LD array; bytes are read only after xarray indexing."""
    __slots__=("prefix","n_samples","n_variants","bytes_per_variant","_packed","_mask4","max_pairs","engine","missing")
    def __init__(self,prefix,n_samples,n_variants,*,sample_mask=None,max_pairs=2_000_000,engine="auto",missing="pairwise"):
        """Initialize PackedBedLDBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        n_variants : int
            Number of variants in the logical source axis, including the unpadded extent of packed
            arrays.

        sample_mask : bool or numeric mask array
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
            not an additional numeric/category value.

        max_pairs : int
            Maximum number of pair/group/path work items retained or evaluated by this guard.
            Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
            silently approximate. Default: ``2000000``.

        engine : str
            Execution/backend selection for this operation. See the source-derived literal
            descriptions below and the called resolver; automatic selection uses the available
            implementation and input representation. Default: ``'auto'``.

        missing : str
            Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
            size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
            'impute' treats missing values as latent, and 'row' uses complete rows where supported.
            'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
            it. See the local validator for the accepted subset. Default: ``'pairwise'``.
        """
        self.prefix=str(prefix); self.n_samples=int(n_samples); self.n_variants=int(n_variants); self.engine=str(engine); self.missing=str(missing).lower()
        self.bytes_per_variant=(self.n_samples+3)//4; self.max_pairs=int(max_pairs)
        self._packed=open_bed_memmap(self.prefix+".bed", n_variants=self.n_variants)
        self._mask4=sample_mask4(self.n_samples,sample_mask)

    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(self.n_variants, self.n_variants)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.n_variants,self.n_variants)
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.float32)

    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if _xr_indexing is None: raise ImportError("xarray is required for PackedBedLDBackend")
        return _xr_indexing.explicit_indexing_adapter(
            key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)

    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],self.n_variants); jj,sj=_as_indices(key[1],self.n_variants)
        npairs=int(len(ii))*int(len(jj))
        if npairs > self.max_pairs:
            raise MemoryError(
                f"virtual LD request contains {npairs:,} SNP pairs; select a smaller region before .load()/.values "
                f"or explicitly raise max_pairs (current {self.max_pairs:,})")
        A=np.asarray(self._packed[ii],np.uint8); B=np.asarray(self._packed[jj],np.uint8)
        out=_packed_r2_selected(A,B,self._mask4,engine=self.engine,missing=self.missing)
        if si and sj: return np.asarray(out[0,0],np.float32)
        if si: return out[0]
        if sj: return out[:,0]
        return out


def virtual_ld_matrix(prefix, *, sample_mask=None, max_pairs=2_000_000, engine="auto", missing="pairwise"):
    """Return a full-coordinate *virtual* xarray LD matrix.

    Merely constructing, printing, slicing, or label-selecting the DataArray does
    not calculate LD. Materializing a selected view invokes exact packed-BED R2.
    Whole-genome materialization is blocked by ``max_pairs``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sample_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value. Passed to ``PackedBedLDBackend`` as
        ``sample_mask``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``2000000``. Passed to ``PackedBedLDBackend`` as
        ``max_pairs``.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'auto'``. Passed to
        ``PackedBedLDBackend`` as ``engine``.

    missing : str
        Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
        size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
        'impute' treats missing values as latent, and 'row' uses complete rows where supported.
        'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
        it. See the local validator for the accepted subset. Default: ``'pairwise'``. Passed to
        ``PackedBedLDBackend`` as ``missing``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr,_,indexing=_require_xarray(); ids,names,chrom,pos,labels_loaded=_packed_coordinate_metadata(prefix)
    backend=PackedBedLDBackend(prefix,len(ids),len(pos),sample_mask=sample_mask,max_pairs=max_pairs,engine=engine,missing=missing)
    data=indexing.LazilyIndexedArray(backend)
    var=xr.Variable(('snp_i','snp_j'),data,attrs={
        'statistic':'r2','pairwise_complete':str(missing).lower()=='pairwise','missing_strategy':str(missing).lower(),'source_representation':'plink1_bed_2bit',
        'virtual':True,'materialization_guard_pairs':int(max_pairs),'engine':str(engine)})
    cmap={'snp_i':('snp_i',names),'snp_j':('snp_j',names),
        'chrom_i':('snp_i',chrom),'chrom_j':('snp_j',chrom),
        'pos_i':('snp_i',pos),'pos_j':('snp_j',pos)}
    coords=xr.Coordinates(cmap,indexes={} if not labels_loaded else None)
    out=xr.DataArray(var,coords=coords,name='r2')
    out.attrs['snp_label_index_materialized']=bool(labels_loaded)
    out.attrs['variant_id_mode']='bim_exact' if labels_loaded else 'integer_index_with_chrom_pos'
    return out


class PackedBedLDSubsetBackend(BackendArray):
    """Virtual LD over preselected source-variant indices only."""
    __slots__=("prefix","n_samples","source_n_variants","indices_i","indices_j","bytes_per_variant","_packed","_mask4","max_pairs","engine","missing")
    def __init__(self,prefix,n_samples,source_n_variants,indices_i,indices_j,*,sample_mask=None,max_pairs=2_000_000,engine='auto',missing='pairwise'):
        """Initialize PackedBedLDSubsetBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        source_n_variants : int
            Logical/packed source extent needed to decode or trim padded genotype storage. Use the
            actual source metadata count, including byte rounding only where requested.

        indices_i : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        indices_j : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        sample_mask : bool or numeric mask array
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
            not an additional numeric/category value.

        max_pairs : int
            Maximum number of pair/group/path work items retained or evaluated by this guard.
            Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
            silently approximate. Default: ``2000000``.

        engine : str
            Execution/backend selection for this operation. See the source-derived literal
            descriptions below and the called resolver; automatic selection uses the available
            implementation and input representation. Default: ``'auto'``.

        missing : str
            Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
            size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
            'impute' treats missing values as latent, and 'row' uses complete rows where supported.
            'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
            it. See the local validator for the accepted subset. Default: ``'pairwise'``.
        """
        self.prefix=str(prefix); self.n_samples=int(n_samples); self.source_n_variants=int(source_n_variants)
        self.indices_i=np.asarray(indices_i,np.int64); self.indices_j=np.asarray(indices_j,np.int64)
        self.bytes_per_variant=(self.n_samples+3)//4; self.max_pairs=int(max_pairs); self.engine=str(engine); self.missing=str(missing).lower()
        self._packed=open_bed_memmap(self.prefix+'.bed', n_variants=self.source_n_variants)
        self._mask4=sample_mask4(self.n_samples,sample_mask)
    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(len(self.indices_i), len(self.indices_j))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (len(self.indices_i),len(self.indices_j))
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.float32)
    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _xr_indexing.explicit_indexing_adapter(key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)
    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],len(self.indices_i)); jj,sj=_as_indices(key[1],len(self.indices_j))
        npairs=int(len(ii))*int(len(jj))
        if npairs>self.max_pairs: raise MemoryError(f'local LD request contains {npairs:,} SNP pairs; raise max_pairs explicitly to materialize it')
        gi=self.indices_i[ii]; gj=self.indices_j[jj]
        out=_packed_r2_selected(np.asarray(self._packed[gi],np.uint8),np.asarray(self._packed[gj],np.uint8),self._mask4,engine=self.engine,missing=self.missing)
        if si and sj:return np.asarray(out[0,0],np.float32)
        if si:return out[0]
        if sj:return out[:,0]
        return out

def virtual_ld_subset(prefix, indices_i, indices_j=None, *, sample_mask=None, max_pairs=2_000_000,
                      engine='auto', missing='pairwise', n_samples=None, source_n_variants=None,
                      names=None, chrom=None, pos=None):
    """Selection-first virtual LD that never allocates whole-genome pair coordinates.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    indices_i : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.

    indices_j : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value.

    sample_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value. Passed to ``PackedBedLDSubsetBackend`` as
        ``sample_mask``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``2000000``. Passed to ``PackedBedLDSubsetBackend`` as
        ``max_pairs``.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'auto'``. Passed to
        ``PackedBedLDSubsetBackend`` as ``engine``.

    missing : str
        Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
        size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
        'impute' treats missing values as latent, and 'row' uses complete rows where supported.
        'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
        it. See the local validator for the accepted subset. Default: ``'pairwise'``. Passed to
        ``PackedBedLDSubsetBackend`` as ``missing``.

    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Default: ``None``. None delegates to the
        local fallback shown in this body; it is not an additional numeric/category value.

    source_n_variants : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value. Passed to ``xr.Variable`` as ``attrs``.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``xr.DataArray(var, coords={'snp_i': ni, 'snp_j': nj, 'chrom_i': ('snp_i', np.asarray(all_chrom)[ii].astype(str, copy=False)), 'chrom_j': ('snp_j', np.asarray(all_chrom)[jj].astype(str, copy=False)),...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    ii=np.asarray(indices_i,np.int64); jj=ii if indices_j is None else np.asarray(indices_j,np.int64)
    if n_samples is None or source_n_variants is None or names is None or chrom is None or pos is None:
        fam,bim=_metadata(prefix); n_samples=len(fam); source_n_variants=len(bim)
        all_names=bim['snp'].astype(str).to_numpy(); all_chrom=bim['chrom'].astype(str).to_numpy(); all_pos=bim['pos'].to_numpy(np.int64)
    else:
        all_names=np.asarray(names); all_chrom=np.asarray(chrom); all_pos=np.asarray(pos,np.int64)
    backend=PackedBedLDSubsetBackend(prefix,n_samples,source_n_variants,ii,jj,sample_mask=sample_mask,max_pairs=max_pairs,engine=engine,missing=missing)
    xr,_,indexing=_require_xarray(); data=indexing.LazilyIndexedArray(backend)
    aa=np.asarray(all_names)
    if np.issubdtype(aa.dtype,np.integer):
        ci=np.asarray(all_chrom)[ii].astype(str); cj=np.asarray(all_chrom)[jj].astype(str)
        ni=np.char.add(np.char.add(ci,':'),np.asarray(all_pos)[ii].astype(str))
        nj=np.char.add(np.char.add(cj,':'),np.asarray(all_pos)[jj].astype(str))
    else:
        ni=aa[ii].astype(str,copy=False); nj=aa[jj].astype(str,copy=False)
    var=xr.Variable(('snp_i','snp_j'),data,attrs={'statistic':'r2','pairwise_complete':str(missing).lower()=='pairwise',
        'missing_strategy':str(missing).lower(),'source_representation':'plink1_bed_2bit','virtual':True,
        'selection_first':True,'source_n_variants':int(source_n_variants),'materialization_guard_pairs':int(max_pairs),'engine':str(engine)})
    return xr.DataArray(var,coords={'snp_i':ni,'snp_j':nj,
        'chrom_i':('snp_i',np.asarray(all_chrom)[ii].astype(str,copy=False)),
        'chrom_j':('snp_j',np.asarray(all_chrom)[jj].astype(str,copy=False)),
        'pos_i':('snp_i',np.asarray(all_pos)[ii]),'pos_j':('snp_j',np.asarray(all_pos)[jj])},name='r2')
class PackedBedOffsetLDBackend(BackendArray):
    """Virtual rolling LD array with dimensions (snp, offset)."""
    __slots__=("prefix","n_samples","n_variants","bytes_per_variant","_packed","_mask4",
               "max_offset","chrom","pos","max_distance_bp","max_pairs")
    def __init__(self,prefix,n_samples,n_variants,chrom,pos,*,max_offset=256,
                 max_distance_bp=1_000_000,sample_mask=None,max_pairs=2_000_000):
        """Initialize PackedBedOffsetLDBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        n_variants : int
            Number of variants in the logical source axis, including the unpadded extent of packed
            arrays.

        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported.

        pos : one-dimensional array-like
            Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
            genomic order is required for searchsorted/banded interval algorithms.

        max_offset : int or None
            Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
            count columns, not base pairs; None leaves the upper separation unrestricted where
            supported. Default: ``256``.

        max_distance_bp : int or None
            Physical distance/window bound in base pairs for selecting nearby variants/genes.
            Positive bounds restrict the region; None leaves the distance unrestricted where
            supported. Window width versus radius is specified by the summary. Default: ``1000000``.

        sample_mask : bool or numeric mask array
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
            not an additional numeric/category value.

        max_pairs : int
            Maximum number of pair/group/path work items retained or evaluated by this guard.
            Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
            silently approximate. Default: ``2000000``.
        """
        self.prefix=str(prefix); self.n_samples=int(n_samples); self.n_variants=int(n_variants)
        self.bytes_per_variant=(self.n_samples+3)//4; self.max_offset=int(max_offset)
        self.chrom=np.asarray(chrom).astype(str); self.pos=np.asarray(pos,np.int64)
        self.max_distance_bp=None if max_distance_bp is None else int(max_distance_bp)
        self.max_pairs=int(max_pairs); self._mask4=sample_mask4(self.n_samples,sample_mask)
        self._packed=open_bed_memmap(self.prefix+".bed", n_variants=self.n_variants)

    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(self.n_variants, self.max_offset + 1)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.n_variants,self.max_offset+1)
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.float32)

    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _xr_indexing.explicit_indexing_adapter(
            key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)

    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],self.n_variants); oo,so=_as_indices(key[1],self.max_offset+1)
        npairs=int(len(ii))*int(len(oo))
        if npairs > self.max_pairs:
            raise MemoryError(f"virtual rolling-LD request contains {npairs:,} pairs; select before materializing")
        out=np.full((len(ii),len(oo)),np.nan,np.float32)
        # Batch neighboring rows. A single row formerly launched a threaded
        # packed kernel for each SNP. Small rectangular BLAS tiles retain the
        # exact pairwise-deletion sums while amortizing decoding and dispatch.
        # The rectangle is bounded independently of the virtual genome size.
        for first in range(0,len(ii),32):
            rows=ii[first:first+32];jj=rows[:,None]+oo[None,:]
            valid=(jj>=0)&(jj<self.n_variants)
            safe=np.clip(jj,0,self.n_variants-1)
            valid &= self.chrom[safe]==self.chrom[rows,None]
            if self.max_distance_bp is not None:
                valid &= (self.pos[safe]-self.pos[rows,None])<=self.max_distance_bp
            targets=np.unique(jj[valid])
            if not len(targets):continue
            values=_packed_r2_selected(self._packed[rows],self._packed[targets],self._mask4,engine='blas')
            a,b=np.nonzero(valid)
            out[first+a,b]=values[a,np.searchsorted(targets,jj[a,b])]
        if si and so: return np.asarray(out[0,0],np.float32)
        if si: return out[0]
        if so: return out[:,0]
        return out


def virtual_offset_r2(prefix, *, max_offset=256, max_distance_bp=1_000_000,
                      sample_mask=None, max_pairs=2_000_000):
    """Virtual ``R2[snp, offset]`` view for rolling-window LD.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    max_offset : int or None
        Bound on variant-index separation in a local pair/LD scan. Nonnegative integer distances
        count columns, not base pairs; None leaves the upper separation unrestricted where
        supported. Default: ``256``. Passed to ``PackedBedOffsetLDBackend`` as ``max_offset``.

    max_distance_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.
        Passed to ``PackedBedOffsetLDBackend`` as ``max_distance_bp``.

    sample_mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
        not an additional numeric/category value. Passed to ``PackedBedOffsetLDBackend`` as
        ``sample_mask``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``2000000``. Passed to ``PackedBedOffsetLDBackend`` as
        ``max_pairs``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr,_,indexing=_require_xarray(); ids,names,chrom,pos,labels_loaded=_packed_coordinate_metadata(prefix)
    backend=PackedBedOffsetLDBackend(prefix,len(ids),len(pos),chrom,pos,max_offset=max_offset,
        max_distance_bp=max_distance_bp,sample_mask=sample_mask,max_pairs=max_pairs)
    data=indexing.LazilyIndexedArray(backend)
    var=xr.Variable(('snp','offset'),data,attrs={'statistic':'r2','pairwise_complete':True,
        'source_representation':'plink1_bed_2bit','virtual':True,
        'layout':'r2[snp, offset] = R2(snp, snp+offset)','max_distance_bp':max_distance_bp})
    cmap={'snp':('snp',names),'offset':('offset',np.arange(max_offset+1,dtype=np.int32)),
        'chrom':('snp',chrom),'pos':('snp',pos)}
    coords=xr.Coordinates(cmap,indexes={} if not labels_loaded else None)
    out=xr.DataArray(var,coords=coords,name='r2');out.attrs['snp_label_index_materialized']=bool(labels_loaded)
    out.attrs['variant_id_mode']='bim_exact' if labels_loaded else 'integer_index_with_chrom_pos'
    return out


@lru_cache(maxsize=1)
def bed_variant_summary_luts():
    """Byte-level observed-count, genotype-sum and squared-sum LUTs.

    Returns
    -------
    object
        Computed result from ``(cnt, sx, sxx)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    gmap=np.asarray([0,-1,1,2],dtype=np.int16); shifts=2*np.arange(4,dtype=np.uint8)
    G=gmap[(np.arange(256,dtype=np.uint16)[:,None] >> shifts[None,:]) & 3]
    cnt=np.zeros((16,256),np.uint8); sx=np.zeros_like(cnt); sxx=np.zeros_like(cnt)
    for mask in range(16):
        inc=((mask >> np.arange(4)) & 1).astype(bool)[None,:]
        valid=inc & (G>=0); X=np.where(valid,G,0)
        cnt[mask]=valid.sum(1,dtype=np.uint8)
        sx[mask]=X.sum(1,dtype=np.int16).astype(np.uint8)
        sxx[mask]=(X*X).sum(1,dtype=np.int16).astype(np.uint8)
    return cnt,sx,sxx


@lru_cache(maxsize=1)
def _numba_variant_stats_kernel():
    """Compile a direct BED-byte variant-statistics reducer lazily.
    """
    import numba
    @numba.njit(parallel=True)
    def kernel(P,m4):
        """Execute the captured compiled numerical kernel on aligned input buffers.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        m4 : int
            Logical/packed source extent needed to decode or trim padded genotype storage. Use the
            actual source metadata count, including byte rounding only where requested.

        Returns
        -------
        object
            Computed result from ``(cnt, sx, sxx)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m,w=P.shape
        cnt=np.zeros(m,np.int32); sx=np.zeros(m,np.int32); sxx=np.zeros(m,np.int32)
        for j in numba.prange(m):
            n=a=b=0
            for k in range(w):
                mm=m4[k]
                if mm==0: continue
                x=P[j,k]
                for q in range(4):
                    if ((mm>>q)&1)==0: continue
                    c=(x>>(2*q))&3
                    if c==1: continue
                    g=0 if c==0 else (1 if c==2 else 2)
                    n+=1; a+=g; b+=g*g
            cnt[j]=n; sx[j]=a; sxx[j]=b
        return cnt,sx,sxx
    return kernel


def packed_variant_stats(packed, mask4, engine='auto'):
    """Variant-wise N, sum(g), sum(g^2) without decoding genotypes.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    mask4 : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'auto'``. Source-recognized literal
        values: ``compiled``, ``numba``. These select the corresponding branches below;
        comparison evidence alone does not establish an exhaustive enum.

    Returns
    -------
    object
        Computed result from ``(cnt, sx, sxx)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    P=np.asarray(packed,np.uint8); m4=np.asarray(mask4,np.uint8)
    active=np.flatnonzero(m4)
    if len(active)!=len(m4): P=P[:,active]; m4=m4[active]
    e=str(engine).lower()
    if e in {'auto','compiled','numba'} and (e!='auto' or P.size>=50_000):
        try: return _numba_variant_stats_kernel()(P,m4)
        except Exception:
            if e in {'compiled','numba'}: raise
    cnt_lut,sx_lut,sxx_lut=bed_variant_summary_luts()
    cnt=cnt_lut[m4[None,:],P].sum(1,dtype=np.int32)
    sx=sx_lut[m4[None,:],P].sum(1,dtype=np.int32)
    sxx=sxx_lut[m4[None,:],P].sum(1,dtype=np.int32)
    return cnt,sx,sxx


def recommended_grm_variant_chunk(n_samples, *, components=('additive',), memory_fraction=None,
                                  maxmem=None, min_chunk=1024, max_chunk=None, requested='auto',
                                  n_full=None, reserve_bytes=0):
    """Choose a dense-GRM genotype block from the working-memory budget, not a fixed constant.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive',)``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to ``bounded_variant_chunk`` as
        ``memory_fraction``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to ``bounded_variant_chunk``
        as ``maxmem``.

    min_chunk : int
        Compatibility/callback input accepted but not consumed by this body. Any object is
        locally accepted; see the calling protocol for the supplied type. Default: ``1024``.

    max_chunk : int
        Input supplied to ``bounded_variant_chunk`` as ``max_chunk``. See that callable's
        documented ``max_chunk`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``None`` (memory budget only). Passed to ``bounded_variant_chunk`` as
        ``max_chunk``.

    requested : int or 'auto'
        Requested variants per block; integers are ceilings. Default: ``'auto'``.

    n_full : int or None
        Animals in the source BED (sets packed bytes per variant); None assumes n_samples.
        Default: ``None``.

    reserve_bytes : int
        Bytes held by resident GRM accumulators/outputs, removed from the budget first.
        Default: ``0``.

    Returns
    -------
    object
        Variants per block.
    """
    from .resources import bounded_variant_chunk
    n=int(n_samples);comps={str(x).lower() for x in components}
    bpv=(int(n if n_full is None else n_full)+3)//4
    # Additive: decode uint8 temporaries (3) + float32 X (4) + valid/~valid masks (2) + slack.
    # Dominance adds D (4) and the nested np.where float32/bool temporaries (~16).
    return bounded_variant_chunk(n,requested=requested,maxmem=maxmem,memory_fraction=memory_fraction,
                                 bytes_per_call=10+(20 if 'dominance' in comps else 0),
                                 bytes_per_variant=2*bpv,max_chunk=max_chunk,
                                 reserve_bytes=reserve_bytes,alignment=1)

def _resolve_sample_indices(fam, sample_ids=None):
    """Resolve sample indices.

    Parameters
    ----------
    fam : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.
    """
    all_ids=fam['iid'].astype(str).to_numpy()
    if sample_ids is None: return np.arange(len(all_ids),dtype=np.int64), all_ids
    wanted=np.asarray(sample_ids).astype(str); lookup={x:i for i,x in enumerate(all_ids)}
    missing=[x for x in wanted if x not in lookup]
    if missing: raise KeyError(f"requested sample IDs absent from FAM: {missing[:5]}")
    return np.asarray([lookup[x] for x in wanted],np.int64), wanted


def _fit_mask4_full(n_full, sample_indices, fit_mask=None):
    """Fit mask4 full.

    Parameters
    ----------
    n_full : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.
    """
    fit=np.ones(len(sample_indices),bool) if fit_mask is None else np.asarray(fit_mask,bool)
    if fit.shape!=(len(sample_indices),): raise ValueError("fit_mask shape mismatch")
    full=np.zeros(int(n_full),bool); full[np.asarray(sample_indices,np.int64)[fit]]=True
    return sample_mask4(n_full,full), fit


def _decode_selected_packed(P, n_full, sample_indices, dtype=np.float32):
    """Decode the selected packed block with the canonical BED decoder.

    Parameters
    ----------
    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    n_full : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``decode_bed_chunk_subset`` as
        ``dtype``.
    """
    from .io import decode_bed
    return decode_bed(P, n_full, sample_indices, dtype=dtype)

def _dense_grm_float_from_bed(prefix, *, components=('additive',), sample_ids=None, fit_mask=None,
                       chromosome=None, min_maf=1e-4, chunk_variants='auto', memory_fraction=None,
                       maxmem=None, accum_dtype=np.float32, output_dtype=np.float32,
                       blas_threads='auto', progress=False, method='vitezica'):
    """Exact VIGOR A/D GRM in RAM from packed BED with RAM-adaptive blocks.

    Genotype calls remain packed until the current block. Allele counts/QC are
    reduced directly from BED bytes. ``chromosome=None`` builds genome-wide and
    records chromosome normalization weights for exact streamed LOCO.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive',)``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to
        ``recommended_grm_variant_chunk`` as ``memory_fraction``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to
        ``recommended_grm_variant_chunk`` as ``maxmem``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``. Passed to ``np.zeros`` as ``dtype``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Source comparisons: ``blas_threads not in {None, 'auto', 0}``. See the
        rejection/fallback branches for enforcement. Passed to ``threadpool_limits`` as
        ``limits``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.
    """
    from .grm_backend import packed_grm_from_bed
    return packed_grm_from_bed(prefix, method=method, components=components,
        sample_ids=sample_ids, fit_mask=fit_mask, chromosome=chromosome,
        min_maf=min_maf, chunk_variants=chunk_variants, memory_fraction=memory_fraction,
        maxmem=maxmem, output_dtype=output_dtype, compute_dtype=accum_dtype,
        blas_threads=blas_threads, progress=progress)


@lru_cache(maxsize=1)
def _numba_bitserial_gram_kernel():
    """Exact raw dosage Gram kernel: 64 variants/word, three popcounts/word.
    """
    import numba
    @numba.njit(inline='always')
    def pc64(x):
        """Count set bits in one 64-bit genotype-mask word.

        Parameters
        ----------
        x : array-like
            Numerical operand in ``x - (x >> np.uint64(1) & np.uint64(6148914691236517205))``. Shape
            must conform to that matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``x * np.uint64(72340172838076673) >> np.uint64(56)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        x=x-((x>>np.uint64(1))&np.uint64(0x5555555555555555))
        x=(x&np.uint64(0x3333333333333333))+((x>>np.uint64(2))&np.uint64(0x3333333333333333))
        x=(x+(x>>np.uint64(4)))&np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x*np.uint64(0x0101010101010101))>>np.uint64(56)
    @numba.njit(parallel=True,fastmath=True)
    def gram(lo,hi):
        """Compute a Gram cross-product of the enclosing packed/feature representation.

        Parameters
        ----------
        lo : object (see the cited input contract)
            Input for the operation ``lo[i, k]``. See
            ``vigor.packed_bed._numba_bitserial_gram_kernel.gram`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        hi : object (see the cited input contract)
            Input for the operation ``hi[i, k]``. See
            ``vigor.packed_bed._numba_bitserial_gram_kernel.gram`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        n,w=lo.shape; out=np.empty((n,n),np.int32)
        for i in numba.prange(n):
            for j in range(i,n):
                z=np.uint64(0)
                for k in range(w):
                    a0=lo[i,k]; a1=hi[i,k]; b0=lo[j,k]; b1=hi[j,k]
                    # genotype bits 01 and 10 are mutually exclusive, so both
                    # 1x2 cross-products can be OR-combined before popcount.
                    z += pc64(a0&b0) + (pc64((a0&b1)|(a1&b0))<<np.uint64(1)) + (pc64(a1&b1)<<np.uint64(2))
                out[i,j]=np.int32(z); out[j,i]=np.int32(z)
        return out
    return gram


def _decode_uint8_selected_packed(P,n_full,sample_indices):
    """Decode one BED block to 0/1/2 uint8 plus a missing mask (no float expansion).

    Parameters
    ----------
    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    n_full : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.
    """
    P=np.asarray(P,np.uint8); sel=np.asarray(sample_indices,np.int64)
    bi=sel>>2; sh=((sel&3)*2).astype(np.uint8)
    codes=((P[:,bi]>>sh[None,:])&np.uint8(3)).T
    missing=(codes==1)
    lut=np.asarray([0,0,1,2],np.uint8)
    return lut[codes],missing


def _bitplanes_uint64(X0):
    """Pack a sample x variant uint8 0/1/2 block into two uint64 bit planes.

    Parameters
    ----------
    X0 : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.
    """
    lo=np.packbits(X0&np.uint8(1),axis=1,bitorder='little')
    hi=np.packbits((X0>>np.uint8(1))&np.uint8(1),axis=1,bitorder='little')
    pad=(-lo.shape[1])%8
    if pad:
        lo=np.pad(lo,((0,0),(0,pad))); hi=np.pad(hi,((0,0),(0,pad)))
    return np.ascontiguousarray(lo).view(np.uint64),np.ascontiguousarray(hi).view(np.uint64)


def dense_additive_grm_bitserial(prefix, *, sample_ids=None, fit_mask=None, chromosome=None,
                                 min_maf=1e-4, chunk_variants='auto', memory_fraction=None,
                                 maxmem=None, accum_dtype=np.float32, output_dtype=np.float32,
                                 blas_threads='auto', progress=False):
    """Exact additive GRM using 2-bit dosage products plus sparse missing-call correction.

    The expensive raw ``X X.T`` term uses three population counts per 64 variants.
    Centering is applied algebraically. Missing calls are mean-imputed exactly in
    centered space (zero contribution), with a sparse correction; no genome-wide
    float genotype matrix is formed.

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

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM).

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``. Passed to ``np.zeros`` as ``dtype``.

    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Source comparisons: ``blas_threads not in {None, 'auto', 0}``. See the
        rejection/fallback branches for enforcement. Passed to ``threadpool_limits`` as
        ``limits``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.

    Returns
    -------
    dict
        Result fields: ``kernels``, ``weights``, ``chromosome_weights``, ``sample_ids``, ``diagnostics``. Field arrays preserve the corresponding input axis order.
    """
    from contextlib import nullcontext
    from scipy import sparse
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits=None
    fam,bim=_metadata(prefix); n_full=len(fam); m_all=len(bim); bpv=(n_full+3)//4
    sel,ids=_resolve_sample_indices(fam,sample_ids); mask4,fit=_fit_mask4_full(n_full,sel,fit_mask)
    n=len(sel); chrom_all=bim['chrom'].astype(str).to_numpy()
    indices=(np.arange(m_all,dtype=np.int64) if chromosome is None else np.flatnonzero(chrom_all==str(chromosome)))
    if len(indices)==0: raise ValueError(f"no variants found for chromosome {chromosome!r}")
    from .resources import bounded_variant_chunk
    # Per call: codes + X0 + missing mask + three bit-op temporaries + packed planes.
    # Reserved: K, and per block the int32 Gram, its cast, CX and CC (all n x n),
    # the final K/wa cast, plus the fixed n x 4096 float32 slab used for X0 @ mu.
    acc=np.dtype(accum_dtype).itemsize
    bs=bounded_variant_chunk(n,requested=chunk_variants,maxmem=maxmem,memory_fraction=memory_fraction,
                             bytes_per_call=6,bytes_per_variant=2*bpv,
                             reserve_bytes=n*n*(2*acc+12)+n*4096*4,n_variants=len(indices),alignment=64)
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m_all)
    K=np.zeros((n,n),dtype=accum_dtype); wa=0.0; used=0; n_missing=0
    chrom_weights={c:{'additive':0.0,'dominance':0.0,'n_variants':0} for c in dict.fromkeys(chrom_all[indices])}
    starts=range(0,len(indices),bs)
    if progress:
        try:
            from tqdm.auto import tqdm; starts=tqdm(starts,total=(len(indices)+bs-1)//bs,desc='2-bit additive GRM',unit='block')
        except Exception: pass
    ctx=(threadpool_limits(limits=int(blas_threads)) if threadpool_limits is not None and blas_threads not in {None,'auto',0} else nullcontext())
    gram_kernel=_numba_bitserial_gram_kernel()
    with ctx:
        for k0 in starts:
            gi=indices[k0:min(len(indices),k0+bs)]; P=np.asarray(mm[gi],np.uint8)
            cnt,sx,_=packed_variant_stats(P,mask4); p=np.divide(sx,2.0*cnt,out=np.full(len(P),np.nan),where=cnt>0)
            maf=np.minimum(p,1-p); keep=np.isfinite(p)&(maf>=float(min_maf))
            if not np.any(keep): continue
            P=P[keep]; p=np.clip(p[keep],float(min_maf),1-float(min_maf)); mu=np.asarray(2*p,np.float32)
            ckeep=chrom_all[gi][keep]; wa_vec=2*p*(1-p); wblock=float(wa_vec.sum()); wa+=wblock; used+=len(p)
            for cc in np.unique(ckeep):
                cm=(ckeep==cc); chrom_weights[str(cc)]['n_variants']+=int(cm.sum()); chrom_weights[str(cc)]['additive']+=float(wa_vec[cm].sum())
            X0,missing=_decode_uint8_selected_packed(P,n_full,sel); rows,cols=np.nonzero(missing); n_missing+=len(rows)
            lo,hi=_bitplanes_uint64(X0); raw=gram_kernel(lo,hi).astype(accum_dtype)
            # X0 @ mu without creating an n x m float array.
            r=np.zeros(n,np.float64); sub=4096
            for j0 in range(0,X0.shape[1],sub):
                j1=min(X0.shape[1],j0+sub)
                r += X0[:,j0:j1].astype(np.float32) @ mu[j0:j1]
            block=raw; block-=r[:,None]; block-=r[None,:]; block+=float(np.dot(mu.astype(np.float64),mu.astype(np.float64)))
            if len(rows):
                vals=mu[cols]
                C=sparse.csr_matrix((vals,(rows,cols)),shape=X0.shape,dtype=np.float32)
                CX=np.asarray(C @ X0.T,dtype=np.float32)
                t=np.bincount(rows,weights=np.asarray(vals,np.float64)**2,minlength=n)
                CC=(C@C.T).toarray()
                block += CX; block += CX.T; block -= t[:,None]; block -= t[None,:]; block += CC
            K += block
    xr,_,_=_require_xarray(); coords={'sample_i':ids,'sample_j':ids}
    KA=xr.DataArray((K/max(wa,1e-30)).astype(output_dtype),dims=('sample_i','sample_j'),coords=coords)
    return {'kernels':{'additive':KA},'weights':{'additive':wa,'dominance':0.0},'chromosome_weights':chrom_weights,'sample_ids':ids,
            'diagnostics':{'chromosome':None if chromosome is None else str(chromosome),'n_variants_total':int(len(indices)),
                           'n_variants_used':int(used),'chunk_variants':int(bs),'storage':'in_memory','engine':'bitserial',
                           'missing_calls':int(n_missing),'missing_rate':float(n_missing/max(1,n*used)),
                           'source_representation':'plink1_bed_2bit','normalization':'vitezica_global_variant_denominator'}}


def dense_grm_from_bed(prefix, *, components=('additive',), sample_ids=None, fit_mask=None,
                       chromosome=None, min_maf=1e-4, chunk_variants='auto', memory_fraction=None,
                       maxmem=None, accum_dtype=np.float32, output_dtype=np.float32,
                       blas_threads='auto', progress=False, engine='auto', method='vitezica',
                       dosage_scale=None):
    """Build exact A/D GRMs from BED using float-BLAS or exact 2-bit arithmetic.
    
    Candidate methods npplink/gcta0 use bounded packed float32 ssyrk.
    npplink uses sample SD (ddof=1), component-specific pair counts and
    n_ij-1 normalization; invalid pair denominators produce NaN. maxmem
    reserves numerical matrices and scratch, with metadata/library/I/O RSS
    additional. blas_threads is scoped; oversized explicit blocks are capped.
    See docs/GRM_API_2_11_7.md for the full scientific/resource contract.
    
    
    ``engine='bitserial'`` is currently additive-only. ``auto`` retains the
    benchmark-proven float-BLAS path; bitserial is an explicit low-memory backend
    until hardware-specific timing shows it is faster.
    
    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.
    
    components : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('additive',)``. Passed to ``_dense_grm_float_from_bed`` as ``components``.
    
    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``dense_additive_grm_bitserial`` as ``sample_ids``.
    
    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``dense_additive_grm_bitserial`` as ``fit_mask``.
    
    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``. Passed to
        ``dense_additive_grm_bitserial`` as ``chromosome``.
    
    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``dense_additive_grm_bitserial`` as ``min_maf``.
    
    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``dense_additive_grm_bitserial`` as ``chunk_variants``.
    
    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM). Passed to
        ``dense_additive_grm_bitserial`` as ``memory_fraction``.
    
    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``. Passed to
        ``dense_additive_grm_bitserial`` as ``maxmem``.
    
    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``dense_additive_grm_bitserial``
        as ``accum_dtype``.
    
    output_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``dense_additive_grm_bitserial``
        as ``output_dtype``.
    
    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Passed to ``dense_additive_grm_bitserial`` as ``blas_threads``.
    
    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``. Passed to
        ``dense_additive_grm_bitserial`` as ``progress``.
    
    engine : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'auto'``. Source-recognized literal
        values: ``bitserial``, ``blas``, ``float``, ``packed``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.
    
    method : str
        Estimator identifier: 'npplink' uses sample-SD pairwise n-minus-one normalization; 'gcta0' uses per-marker HWE scaling; 'van' and 'vitezica' use global HWE weights. Default: ``'vitezica'``.
    dosage_scale : one-dimensional array or None
        Optional per-selected-animal dosage multiplier applied to each decoded block for confidently detected male-X recoding; None retains original BED dosages. Default: ``None``.
    
    Returns
    -------
    object
        Computed result from ``_dense_grm_float_from_bed(prefix, components=components, sample_ids=sample_ids, fit_mask=fit_mask, chromosome=chromosome, min_maf=min_maf, chunk_variants=chunk_variants, memory_fraction=memory_frac...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .grm_methods import validate_grm_method
    method, components = validate_grm_method(method, components)
    if method in {'npplink', 'gcta0'} or dosage_scale is not None:
        if str(engine).lower() not in {'auto', 'float', 'blas'}:
            raise ValueError('pairwise/dosage GRMs require the float BLAS engine')
        from .grm_backend import packed_grm_from_bed
        return packed_grm_from_bed(prefix, method=method, components=components,
            sample_ids=sample_ids, fit_mask=fit_mask, chromosome=chromosome,
            min_maf=min_maf, chunk_variants=chunk_variants, memory_fraction=memory_fraction,
            maxmem=maxmem, output_dtype=output_dtype, compute_dtype=accum_dtype, blas_threads=blas_threads,
            progress=progress, dosage_scale=dosage_scale)
    # Fail fast if the dense matrices cannot fit; each engine then sizes its own
    # blocks from what remains (explicit requests stay ceilings).
    from .resources import working_memory_bytes
    fam, _ = _metadata(prefix)
    selected_n = len(fam) if sample_ids is None else len(sample_ids)
    budget = working_memory_bytes(maxmem, memory_fraction)
    matrix_bytes = selected_n**2 * (len(components) *
        (np.dtype(accum_dtype).itemsize + np.dtype(output_dtype).itemsize) + 12)
    if budget <= matrix_bytes:
        raise MemoryError(f'GRM matrices need {matrix_bytes} bytes plus block scratch; budget is {budget}')
    if blas_threads in (None, 'auto', 0):
        # Blocks run one at a time, so the GEMM/SYRK may use every available core.
        from .resources import usable_cpu_count
        blas_threads = usable_cpu_count()
    comps={str(x).lower() for x in components}; e=str(engine).lower()
    if e not in {'auto','bitserial','packed','float','blas'}: raise ValueError("unknown GRM engine")
    if e in {'bitserial','packed'}:
        if comps!={'additive'}:
            if e!='auto': raise ValueError("bitserial GRM currently supports additive component only")
        else:
            return dense_additive_grm_bitserial(prefix,sample_ids=sample_ids,fit_mask=fit_mask,chromosome=chromosome,
                min_maf=min_maf,chunk_variants=chunk_variants,memory_fraction=memory_fraction,maxmem=maxmem,
                accum_dtype=accum_dtype,output_dtype=output_dtype,blas_threads=blas_threads,progress=progress)
    return _dense_grm_float_from_bed(prefix,components=components,method=method,sample_ids=sample_ids,fit_mask=fit_mask,
        chromosome=chromosome,min_maf=min_maf,chunk_variants=chunk_variants,memory_fraction=memory_fraction,
        maxmem=maxmem,accum_dtype=accum_dtype,output_dtype=output_dtype,blas_threads=blas_threads,progress=progress)



def _qc_variant_chunk(n_full, n_variants, *, maxmem=None, memory_fraction=None):
    """Variants per allele-frequency QC block: packed rows plus per-byte count/sum temporaries."""
    from .resources import bounded_variant_chunk
    return bounded_variant_chunk(n_full, maxmem=maxmem, memory_fraction=memory_fraction,
                                 bytes_per_call=13/4, n_variants=n_variants, alignment=1)


@lru_cache(maxsize=1)
def _numba_packed_additive_matmat_kernel():
    """Compile exact centered packed-BED K@V block operation lazily.
    """
    import numba
    @numba.njit(parallel=True,fastmath=True)
    def kernel(P,mu,byte_index,shift,V):
        """Execute the captured compiled numerical kernel on aligned input buffers.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        mu : object (see the cited input contract)
            Input for the operation ``g - mu[j]``. See
            ``vigor.packed_bed._numba_packed_additive_matmat_kernel.kernel`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        byte_index : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        shift : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        V : array-like
            Numerical operand in ``z * V[i, q]``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``Y``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m=P.shape[0]; n=len(byte_index); r=V.shape[1]
        T=np.zeros((m,r),np.float32)
        for j in numba.prange(m):
            muj=mu[j]
            for i in range(n):
                c=(P[j,byte_index[i]]>>shift[i])&3
                if c==1: continue
                g=0.0 if c==0 else (1.0 if c==2 else 2.0)
                z=g-muj
                for q in range(r): T[j,q]+=z*V[i,q]
        Y=np.zeros((n,r),np.float32)
        for i in numba.prange(n):
            bi=byte_index[i]; sh=shift[i]
            for j in range(m):
                c=(P[j,bi]>>sh)&3
                if c==1: continue
                g=0.0 if c==0 else (1.0 if c==2 else 2.0)
                z=g-mu[j]
                for q in range(r): Y[i,q]+=z*T[j,q]
        return Y
    return kernel


@lru_cache(maxsize=1)
def _numba_packed_additive_matmat_kernel64():
    """Float64-accumulation packed K@V kernel for certified exact solves.
    """
    import numba
    @numba.njit(parallel=True,fastmath=False)
    def kernel(P,mu,byte_index,shift,V):
        """Execute the captured compiled numerical kernel on aligned input buffers.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        mu : object (see the cited input contract)
            Input for the operation ``g - mu[j]``. See
            ``vigor.packed_bed._numba_packed_additive_matmat_kernel64.kernel`` and its caller for
            the scientific representation; the shown operation constrains the required type/shape.

        byte_index : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        shift : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        V : array-like
            Numerical operand in ``z * V[i, q]``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``Y``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m=P.shape[0]; n=len(byte_index); r=V.shape[1]
        T=np.zeros((m,r),np.float64)
        for j in numba.prange(m):
            muj=mu[j]
            for i in range(n):
                c=(P[j,byte_index[i]]>>shift[i])&3
                if c==1: continue
                g=0.0 if c==0 else (1.0 if c==2 else 2.0)
                z=g-muj
                for q in range(r): T[j,q]+=z*V[i,q]
        Y=np.zeros((n,r),np.float64)
        for i in numba.prange(n):
            bi=byte_index[i]; sh=shift[i]
            for j in range(m):
                c=(P[j,bi]>>sh)&3
                if c==1: continue
                g=0.0 if c==0 else (1.0 if c==2 else 2.0)
                z=g-mu[j]
                for q in range(r): Y[i,q]+=z*T[j,q]
        return Y
    return kernel


@lru_cache(maxsize=1)
def _numba_packed_additive_diag_kernel():
    """Construct the compiled packed-genotype additive-diagonal accumulation kernel.
        
    """
    import numba
    @numba.njit(parallel=True,fastmath=True)
    def kernel(P,mu,byte_index,shift):
        """Execute the captured compiled numerical kernel on aligned input buffers.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        mu : object (see the cited input contract)
            Input for the operation ``g - mu[j]``. See
            ``vigor.packed_bed._numba_packed_additive_diag_kernel.kernel`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.

        byte_index : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        shift : int or one-dimensional integer array
            Packed byte positions or bit shifts selecting the requested sample calls. Positions must
            be inside the packed axis; BED shifts are 0, 2, 4 or 6.

        Returns
        -------
        object
            Computed result from ``d``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m=P.shape[0]; n=len(byte_index); d=np.zeros(n,np.float64)
        for i in numba.prange(n):
            bi=byte_index[i]; sh=shift[i]; zsum=0.0
            for j in range(m):
                c=(P[j,bi]>>sh)&3
                if c==1: continue
                g=0.0 if c==0 else (1.0 if c==2 else 2.0)
                z=g-mu[j]; zsum+=z*z
            d[i]=zsum
        return d
    return kernel


def additive_grm_operator_from_bed(prefix, *, sample_ids=None, fit_mask=None, chromosome=None,
                                   min_maf=1e-4, chunk_variants='auto', progress=False, dtype=np.float32,
                                   maxmem=None, memory_fraction=None):
    """Full-rank exact additive GRM LinearOperator directly over PLINK BED bytes.

    The operator preserves every genomic eigen-direction and computes ``K @ V``
    as two packed BED traversals per variant block.  It never forms a dense
    genotype matrix or an n x n GRM.  Allele frequencies/QC are precomputed once
    with the compiled packed-statistics reducer.

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

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized from the memory budget).

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``.

    Returns
    -------
    object
        Computed result from ``PackedAdditiveGRMOperator()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from scipy.sparse.linalg import LinearOperator
    fam,bim=_metadata(prefix); n_full=len(fam); m_all=len(bim); bpv=(n_full+3)//4
    sel,ids=_resolve_sample_indices(fam,sample_ids); mask4,fit=_fit_mask4_full(n_full,sel,fit_mask); n=len(sel)
    chrom_all=bim['chrom'].astype(str).to_numpy()
    indices=(np.arange(m_all,dtype=np.int64) if chromosome is None else np.flatnonzero(chrom_all==str(chromosome)))
    if len(indices)==0: raise ValueError(f"no variants found for chromosome {chromosome!r}")
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m_all)
    work_dtype=np.dtype(dtype)
    if work_dtype not in {np.dtype(np.float32),np.dtype(np.float64)}:
        raise ValueError('dtype must be float32 or float64')
    kept=[]; mus=[]; wa=0.0; stat_bs=_qc_variant_chunk(n_full,len(indices),maxmem=maxmem,memory_fraction=memory_fraction)
    starts=range(0,len(indices),stat_bs)
    if progress:
        try:
            from tqdm.auto import tqdm; starts=tqdm(starts,total=(len(indices)+stat_bs-1)//stat_bs,desc='packed GRM operator QC',unit='block')
        except Exception: pass
    for k0 in starts:
        gi=indices[k0:min(len(indices),k0+stat_bs)]; P=np.asarray(mm[gi],np.uint8)
        cnt,sx,_=packed_variant_stats(P,mask4); p=np.divide(sx,2.0*cnt,out=np.full(len(P),np.nan),where=cnt>0)
        keep=np.isfinite(p)&(np.minimum(p,1-p)>=float(min_maf))
        if np.any(keep):
            pp=np.clip(p[keep],float(min_maf),1-float(min_maf)); kept.append(gi[keep]); mus.append((2*pp).astype(work_dtype)); wa+=float(np.sum(2*pp*(1-pp)))
    kept=np.concatenate(kept) if kept else np.empty(0,np.int64); mu=np.concatenate(mus) if mus else np.empty(0,dtype=work_dtype)
    if not len(kept): raise ValueError('no variants remain after MAF filtering')
    from .resources import bounded_variant_chunk
    # Gathered packed rows plus a (variants x probe) workspace; probes are budgeted at <=256 columns.
    bs=bounded_variant_chunk(n_full,requested=chunk_variants,maxmem=maxmem,memory_fraction=memory_fraction,
                             bytes_per_call=0.25,bytes_per_variant=256*work_dtype.itemsize,n_variants=len(kept))
    byte_index=(sel>>2).astype(np.int64); shift=((sel&3)*2).astype(np.uint8)
    mat_kernel=(_numba_packed_additive_matmat_kernel64() if work_dtype==np.dtype(np.float64) else _numba_packed_additive_matmat_kernel())
    diag_kernel=_numba_packed_additive_diag_kernel()
    class PackedAdditiveGRMOperator(LinearOperator):
        """PackedAdditiveGRMOperator represents packed additive genomic relationship matrix operator.

        See ``__init__`` for complete input types, defaults and construction constraints.
        """
        def __init__(self):
            """Initialize PackedAdditiveGRMOperator with the inputs and resource settings below.
            """
            self.sample_ids=ids; self.n_variants=int(len(kept)); self.weight=float(wa); self._diag_cache=None
            self.domino_exact_full_rank=True; self.supports_submatrix=False
            self.diagnostics={'engine':'packed_fullrank_operator','n_samples':int(n),'n_variants_used':int(len(kept)),
                'chunk_variants':int(bs),'formed_dense_grm':False,'source_representation':'plink1_bed_2bit',
                'normalization':'vitezica_global_variant_denominator','chromosome':None if chromosome is None else str(chromosome),
                'exact_full_rank':True,'dtype':str(work_dtype)}
            super().__init__(dtype=work_dtype,shape=(n,n))
        def _matmat(self,V):
            """Apply the operator to a matrix for the LinearOperator interface.

            Parameters
            ----------
            V : array-like
                Numerical operand in ``V[:, None]``. Shape must conform to that matrix/vector operation
                and the enclosing prepared state.
            """
            V=np.asarray(V,dtype=work_dtype); one=V.ndim==1
            if one: V=V[:,None]
            if V.shape[0]!=n: raise ValueError('RHS sample dimension mismatch')
            Y=np.zeros((n,V.shape[1]),dtype=work_dtype)
            for k0 in range(0,len(kept),bs):
                k1=min(len(kept),k0+bs); P=np.asarray(mm[kept[k0:k1]],np.uint8)
                Y += mat_kernel(P,mu[k0:k1],byte_index,shift,V)
            Y/=max(float(wa),1e-30)
            return Y[:,0] if one else Y
        def _matvec(self,v): 
            """Apply the operator to a vector for the LinearOperator interface.

            Parameters
            ----------
            v : one-dimensional array-like
                Vector supplied by the linear operator/iterative-solver callback. Length equals the
                enclosing operator dimension; observed-cell operators use the prepared finite-cell
                order.
            """
            return self._matmat(v)
        def _rmatvec(self,v): 
            """Return the rmatvec result computed by ``self._matmat``.

            Parameters
            ----------
            v : array-like
                Numerical operand in ``self._matmat(v)``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.
            """
            return self._matmat(v)
        def matmat(self,V): 
            """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

            Parameters
            ----------
            V : array-like
                Numerical operand in ``self._matmat(V)``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.

            Returns
            -------
            object
                Computed result from ``self._matmat(V)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return self._matmat(V)
        def diag(self):
            """Return the diagonal variances of the represented covariance/kernel.

            Returns
            -------
            object
                Computed result from ``self._diag_cache``; see the summary and the called implementation for its labeled/numerical representation.
            """
            if self._diag_cache is None:
                d=np.zeros(n,np.float64)
                for k0 in range(0,len(kept),bs):
                    k1=min(len(kept),k0+bs); P=np.asarray(mm[kept[k0:k1]],np.uint8)
                    d += diag_kernel(P,mu[k0:k1],byte_index,shift)
                self._diag_cache=(d/max(float(wa),1e-30)).astype(work_dtype)
            return self._diag_cache
    return PackedAdditiveGRMOperator()

def lowrank_grm_from_bed(prefix, *, rank=200, oversample=16, n_power=0, sample_ids=None,
                         fit_mask=None, chromosome=None, min_maf=1e-4, chunk_variants='auto',
                         memory_fraction=None, maxmem=None, random_state=0, blas_threads='auto', progress=False):
    """Direct packed-BED randomized eigensolver; never forms the n x n GRM.

    It applies K=ZZ.T/w to a block of vectors by streaming centered genotype
    blocks.  This is useful when rank << n; high ranks are intentionally allowed
    so callers can test whether lower-eigenvalue genomic signal is important.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine. Default: ``200``.

    oversample : int
        Extra sketch directions or selected landmark/anchor count for the approximation.
        Positive counts control retained information and computational cost. Default: ``16``.

    n_power : int
        Input for the operation ``range(max(0, int(n_power)))``. See
        ``vigor.packed_bed.lowrank_grm_from_bed`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``0``.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM).

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    blas_threads : int or str
        BLAS thread limit during numerical operations. Positive integers select an explicit
        thread count; auto uses the routine's workload-dependent thread policy. Default:
        ``'auto'``. Source comparisons: ``blas_threads not in {None, 'auto', 0}``. See the
        rejection/fallback branches for enforcement. Passed to ``threadpool_limits`` as
        ``limits``.

    progress : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``False``.

    Returns
    -------
    dict
        Result fields: ``U``, ``s``, ``weight``, ``sample_ids``, ``diagnostics``. Field arrays preserve the corresponding input axis order.
    """
    from contextlib import nullcontext
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits=None
    fam,bim=_metadata(prefix); n_full=len(fam); m_all=len(bim); bpv=(n_full+3)//4
    sel,ids=_resolve_sample_indices(fam,sample_ids); mask4,fit=_fit_mask4_full(n_full,sel,fit_mask); n=len(sel)
    r=min(int(rank),n); l=min(n,r+max(0,int(oversample)))
    chrom_all=bim['chrom'].astype(str).to_numpy(); indices=(np.arange(m_all,dtype=np.int64) if chromosome is None else np.flatnonzero(chrom_all==str(chromosome)))
    mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m_all)
    # First light pass: exact allele frequencies/QC, retained as compact vectors.
    kept=[]; ps=[]; wa=0.0; stat_bs=_qc_variant_chunk(n_full,len(indices),maxmem=maxmem,memory_fraction=memory_fraction)
    for k0 in range(0,len(indices),stat_bs):
        gi=indices[k0:min(len(indices),k0+stat_bs)]; P=np.asarray(mm[gi],np.uint8)
        cnt,sx,_=packed_variant_stats(P,mask4); p=np.divide(sx,2.0*cnt,out=np.full(len(P),np.nan),where=cnt>0)
        keep=np.isfinite(p)&(np.minimum(p,1-p)>=float(min_maf))
        if np.any(keep):
            pp=np.clip(p[keep],float(min_maf),1-float(min_maf)); kept.append(gi[keep]); ps.append(pp.astype(np.float32)); wa+=float(np.sum(2*pp*(1-pp)))
    kept=np.concatenate(kept) if kept else np.empty(0,np.int64); pvec=np.concatenate(ps) if ps else np.empty(0,np.float32)
    if not len(kept): raise ValueError('no variants remain after MAF filtering')
    from .resources import bounded_variant_chunk
    # Per call: decode uint8 temporaries (3) + float32 Z (4) + valid/~valid (2); per variant the
    # gathered packed row and T=Z.T@Q; reserved: Omega/Y/Q/KQ/U and QR workspace (n x l each).
    bs=bounded_variant_chunk(n,requested=chunk_variants,maxmem=maxmem,memory_fraction=memory_fraction,
                             bytes_per_call=9,bytes_per_variant=bpv+4*l,reserve_bytes=6*4*n*l,n_variants=len(kept))
    ctx=(threadpool_limits(limits=int(blas_threads)) if threadpool_limits is not None and blas_threads not in {None,'auto',0} else nullcontext())
    def apply_K(Q):
        """Apply K.

        Parameters
        ----------
        Q : array-like or numeric scalar
            Input for the operation ``Z.T @ Q``. See
            ``vigor.packed_bed.lowrank_grm_from_bed.apply_K`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        Returns
        -------
        object
            Computed result from ``Y``; see the summary and the called implementation for its labeled/numerical representation.
        """
        Y=np.zeros((n,Q.shape[1]),np.float32); starts=range(0,len(kept),bs)
        if progress:
            try:
                from tqdm.auto import tqdm; starts=tqdm(starts,total=(len(kept)+bs-1)//bs,desc='packed K @ Q',unit='block',leave=False)
            except Exception: pass
        with ctx:
            for k0 in starts:
                gi=kept[k0:min(len(kept),k0+bs)]; pp=pvec[k0:min(len(kept),k0+bs)]
                P=np.asarray(mm[gi],np.uint8); Z=_decode_selected_packed(P,n_full,sel,np.float32)
                valid=np.isfinite(Z); Z-=2*pp[None,:]; Z[~valid]=0.0
                T=Z.T@Q; Y += Z@T
        Y/=max(wa,1e-30); return Y
    rng=np.random.default_rng(random_state); Omega=rng.standard_normal((n,l),dtype=np.float32)
    Y=apply_K(Omega); Q,_=np.linalg.qr(Y,mode='reduced'); Q=Q.astype(np.float32,copy=False)
    for _ in range(max(0,int(n_power))):
        Y=apply_K(Q); Q,_=np.linalg.qr(Y,mode='reduced'); Q=Q.astype(np.float32,copy=False)
    KQ=apply_K(Q); B=np.asarray(Q.T@KQ,np.float64); B=(B+B.T)*0.5
    evals,V=np.linalg.eigh(B); order=np.argsort(evals)[::-1][:r]; svals=np.maximum(evals[order],0.0); U=np.asarray(Q@V[:,order],np.float32)
    xr,_,_=_require_xarray()
    return {'U':xr.DataArray(U,dims=('sample','component'),coords={'sample':ids,'component':np.arange(r)}),
            's':xr.DataArray(svals,dims=('component',),coords={'component':np.arange(r)}),
            'weight':float(wa),'sample_ids':ids,
            'diagnostics':{'engine':'randomized_operator','rank':int(r),'oversample':int(l-r),'n_power':int(n_power),
                           'n_variants_used':int(len(kept)),'chunk_variants':int(bs),'formed_dense_grm':False,
                           'source_representation':'plink1_bed_2bit','normalization':'vitezica_global_variant_denominator'}}

def _selected_grm_block(prefix, rows_i, rows_j, *, component='additive', fit_mask=None,
                        min_maf=1e-4, chunk_variants='auto'):
    """Return the selected genomic relationship matrix block result computed by ``(num / max(weight, 1e-30)).astype``.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    rows_i : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.

    rows_j : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``. Source-recognized literal values: ``dominance``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized from the memory budget).
    """
    fam,bim=_metadata(prefix); n=len(fam); m=len(bim); bpv=(n+3)//4
    ii=np.asarray(rows_i,np.int64); jj=np.asarray(rows_j,np.int64)
    union=np.unique(np.r_[ii,jj]); where={int(x):k for k,x in enumerate(union)}
    ri=np.asarray([where[int(x)] for x in ii]); rj=np.asarray([where[int(x)] for x in jj])
    mask4=sample_mask4(n,fit_mask); mm=open_bed_memmap(str(prefix)+'.bed', n_variants=m)
    num=np.zeros((len(ii),len(jj)),np.float64); weight=0.0; comp=str(component).lower()
    from .resources import bounded_variant_chunk
    # Packed row + LUT temporaries per source sample, decoded X/valid/D per selected sample.
    bs=bounded_variant_chunk(len(union),requested=chunk_variants,bytes_per_call=13,
                             bytes_per_variant=4*bpv,n_variants=m,alignment=1)
    for a0 in range(0,m,bs):
        P=np.asarray(mm[a0:min(m,a0+bs)],np.uint8); cnt,sx,_=packed_variant_stats(P,mask4)
        p=np.divide(sx,2.0*cnt,out=np.full(len(P),np.nan),where=cnt>0); maf=np.minimum(p,1-p)
        keep=np.isfinite(p)&(maf>=float(min_maf))
        if not np.any(keep): continue
        P=P[keep]; p=np.clip(p[keep],float(min_maf),1-float(min_maf)); q=1-p
        X=_decode_selected_packed(P,n,union,np.float32); valid=np.isfinite(X)
        if comp=='additive':
            X -= np.asarray(2*p,np.float32)[None,:]; X[~valid]=0.0
            num += X[ri]@X[rj].T; weight += float(np.sum(2*p*q))
        elif comp=='dominance':
            pp=np.asarray(p,np.float32); qq=np.asarray(q,np.float32)
            D=np.where(X==0,-2*pp[None,:]**2,np.where(X==1,2*pp[None,:]*qq[None,:],np.where(X==2,-2*qq[None,:]**2,0.0))).astype(np.float32)
            D[~valid]=0.0; num += D[ri]@D[rj].T; weight += float(np.sum((2*p*q)**2))
        else: raise ValueError("component must be additive or dominance")
    return (num/max(weight,1e-30)).astype(np.float32)


class PackedBedGRMBackend(BackendArray):
    """Virtual exact VIGOR GRM; only selected sample blocks are materialized."""
    __slots__=("prefix","n_samples","component","fit_mask","min_maf","chunk_variants","max_pairs")
    def __init__(self,prefix,n_samples,*,component='additive',fit_mask=None,min_maf=1e-4,
                 chunk_variants='auto',max_pairs=1_000_000):
        """Initialize PackedBedGRMBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        component : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required. Default:
            ``'additive'``.

        fit_mask : one-dimensional bool array or None
            Mask selecting training animals for fitting frequencies, scaling or model parameters.
            Length must equal the animal axis; None uses the routine's full eligible training set.
            Keep evaluation outcomes outside this mask. Default: ``None``.

        min_maf : float
            Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
            below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

        chunk_variants : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized from the memory budget).

        max_pairs : int
            Maximum number of pair/group/path work items retained or evaluated by this guard.
            Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
            silently approximate. Default: ``1000000``.
        """
        self.prefix=str(prefix); self.n_samples=int(n_samples); self.component=str(component)
        self.fit_mask=None if fit_mask is None else np.asarray(fit_mask,bool)
        self.min_maf=float(min_maf); self.chunk_variants='auto' if chunk_variants is None or str(chunk_variants).lower()=='auto' else int(chunk_variants); self.max_pairs=int(max_pairs)
    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(self.n_samples, self.n_samples)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.n_samples,self.n_samples)
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.float32)
    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _xr_indexing.explicit_indexing_adapter(key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)
    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],self.n_samples); jj,sj=_as_indices(key[1],self.n_samples)
        if len(ii)*len(jj)>self.max_pairs:
            raise MemoryError(f"virtual GRM request contains {len(ii)*len(jj):,} pairs; select a smaller block before materializing")
        out=_selected_grm_block(self.prefix,ii,jj,component=self.component,fit_mask=self.fit_mask,
                                min_maf=self.min_maf,chunk_variants=self.chunk_variants)
        if si and sj:return np.asarray(out[0,0],np.float32)
        if si:return out[0]
        if sj:return out[:,0]
        return out

def virtual_grm(prefix, *, component='additive', fit_mask=None, min_maf=1e-4,
                chunk_variants='auto', max_pairs=1_000_000):
    """Full-coordinate virtual GRM DataArray; computation occurs after indexing.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'additive'``. Passed to ``PackedBedGRMBackend`` as ``component``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``PackedBedGRMBackend`` as ``fit_mask``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``PackedBedGRMBackend`` as ``min_maf``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized from the memory budget). Passed
        to ``PackedBedGRMBackend`` as ``chunk_variants``.

    max_pairs : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``1000000``. Passed to ``PackedBedGRMBackend`` as
        ``max_pairs``.

    Returns
    -------
    object
        Computed result from ``xr.DataArray(var, coords={'sample_i': ids, 'sample_j': ids}, name=f'grm_{component}')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr,_,indexing=_require_xarray(); fam,_=_metadata(prefix); ids=fam['iid'].astype(str).to_numpy()
    backend=PackedBedGRMBackend(prefix,len(fam),component=component,fit_mask=fit_mask,
        min_maf=min_maf,chunk_variants=chunk_variants,max_pairs=max_pairs)
    data=indexing.LazilyIndexedArray(backend)
    var=xr.Variable(('sample_i','sample_j'),data,attrs={'kernel_component':str(component),
        'normalization':'vitezica_global_variant_denominator','source_representation':'plink1_bed_2bit',
        'virtual':True,'materialization_guard_pairs':int(max_pairs)})
    return xr.DataArray(var,coords={'sample_i':ids,'sample_j':ids},name=f'grm_{component}')


def open_packed_bed_xarray(prefix, *, chunk_variants='auto'):
    """Direct BED-byte xarray Dataset; ``packed`` is SNP x sample_byte uint8.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants. Passed to ``xr.Dataset`` as ``attrs``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized from the memory budget). Passed
        to ``xr.Variable`` as ``attrs``.

    Returns
    -------
    object
        Computed result from ``ds``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr,_,indexing=_require_xarray(); ids,names,chrom,pos,labels_loaded=_packed_coordinate_metadata(prefix); n=len(ids); m=len(pos); bpv=(n+3)//4
    backend=PackedBedByteBackend(prefix,m,bpv); packed=indexing.LazilyIndexedArray(backend)
    from .resources import bounded_variant_chunk
    # Preferred chunk = what a float32 decode plus two working copies affords.
    chunk_variants=bounded_variant_chunk(n,requested=chunk_variants,bytes_per_call=12,n_variants=m)
    coord_map={'snp':('snp',names),'sample_byte':('sample_byte',np.arange(bpv,dtype=np.int64)),
               'iid':('iid',ids),'chrom':('snp',chrom),'pos':('snp',pos)}
    coords=xr.Coordinates(coord_map,indexes={} if m>500_000 else None)
    pvar=xr.Variable(('snp','sample_byte'),packed,attrs={'preferred_variant_chunk':int(chunk_variants),'physical_backend':'plink1_bed_2bit','effect_allele_convention':'BIM_column_6_counted'})
    ds=xr.Dataset({'packed':pvar,'sample_byte_mask':(('sample_byte',),sample_mask4(n))},coords=coords,
        attrs={'representation':'plink1_bed_2bit','effect_allele_convention':'BIM_column_6_counted','prefix':str(prefix),'n_samples':int(n),'n_variants':int(m),
               'bytes_per_variant':int(bpv),'domino_lazy_packed':True,'snp_label_index_materialized':bool(labels_loaded),
               'variant_id_mode':'bim_exact' if labels_loaded else 'integer_index_with_chrom_pos',
               'preferred_variant_chunk':int(chunk_variants),'dask_wrapped_at_open':False})
    if not labels_loaded: ds=ds.set_xindex('iid')
    return ds

try:
    import xarray as _xr
    @_xr.register_dataset_accessor('vigor_genomics')
    @_xr.register_dataset_accessor('domino_genomics')
    class PackedBedAccessor:
        """Virtual packed-genotype operations attached to a packed BED Dataset."""
        def __init__(self,obj):
            """Initialize PackedBedAccessor with the inputs and resource settings below.

            Parameters
            ----------
            obj : array-like, table or labeled object
                Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
                described in the summary; the local coercion/selection expression below defines the
                accepted representation.
            """
            if obj.attrs.get('representation')!='plink1_bed_2bit':
                raise AttributeError('domino_genomics accessor requires a packed PLINK1 BED Dataset')
            self._obj=obj; self.prefix=str(obj.attrs['prefix'])
        def _region_indices(self, *, snp_indices=None, chrom=None, start=None, end=None, snps=None):
            """Resolve requested variants by indices, IDs or a chromosome-position interval.

            Parameters
            ----------
            snp_indices : one-dimensional integer array or None
                Zero-based variant positions to select from the current source. Values must refer to
                existing columns; None retains all eligible variants. Streaming readers may additionally
                require sorted indices. Default: ``None``.

            chrom : str or int or None
                Chromosome identifier matched against source metadata as a string. None leaves the
                chromosome unrestricted where supported. Default: ``None``.

            start : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``.

            end : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``.

            snps : array-like or pandas.DataFrame
                Genotype values with animals in rows and variants in columns, or the genotype source
                accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
                labels and the coding expected by the selected kernel or association model. Default:
                ``None``. None delegates to the local fallback shown in this body; it is not an
                additional numeric/category value.
            """
            n=int(self._obj.attrs['n_variants'])
            if snp_indices is not None:
                return np.asarray(snp_indices,np.int64)
            cc=np.asarray(self._obj.coords['chrom'].data).astype(str); pp=np.asarray(self._obj.coords['pos'].data,np.int64)
            if chrom is not None:
                c=str(chrom); mask=(cc==c)
                if not mask.any() and c.lower().startswith('chr'): mask=(cc==c[3:])
                if start is not None: mask &= pp>=int(start)
                if end is not None: mask &= pp<=int(end)
                return np.flatnonzero(mask)
            if snps is None: return None
            req=list(map(str,snps)); names=np.asarray(self._obj.coords['snp'].data).astype(str,copy=False)
            parsed=[]
            for q in req:
                x=q[3:] if q.lower().startswith('chr') else q
                if ':' not in x: parsed=[]; break
                a,b=x.rsplit(':',1)
                try: parsed.append((a,int(b)))
                except ValueError: parsed=[]; break
            if len(parsed)==len(req):
                out=[]
                for q,(c,bp) in zip(req,parsed):
                    m=np.flatnonzero((cc==c)&(pp==bp))
                    if len(m)==0: raise KeyError(f'SNP {q!r} not found')
                    exact=m[names[m]==q]; out.append(int(exact[0] if len(exact) else m[0]))
                return np.asarray(out,np.int64)
            if n>500_000:
                raise ValueError('For >500k variants, non-chrom:pos label lookup is disabled to avoid a multi-GB pandas index; use snp_indices or chrom/start/end.')
            import pandas as pd
            z=pd.Index(names).get_indexer(req)
            if np.any(z<0): raise KeyError(f'{int(np.sum(z<0))} requested SNP labels were not found')
            return z.astype(np.int64)
        def ld_matrix(self, *, snp_indices=None, indices_i=None, indices_j=None, chrom=None, start=None, end=None, snps=None, **kwargs):
            """Compute LD for the selected packed-genotype variant coordinates.

            Parameters
            ----------
            snp_indices : one-dimensional integer array or None
                Zero-based variant positions to select from the current source. Values must refer to
                existing columns; None retains all eligible variants. Streaming readers may additionally
                require sorted indices. Default: ``None``.

            indices_i : sequence or integer array
                Identifiers or zero-based positions of selected/protected items on the corresponding
                input axis. Entries must exist; paired index arrays define the row/column pairs to
                compute. Default: ``None``. None delegates to the local fallback shown in this body; it
                is not an additional numeric/category value.

            indices_j : sequence or integer array
                Identifiers or zero-based positions of selected/protected items on the corresponding
                input axis. Entries must exist; paired index arrays define the row/column pairs to
                compute. Default: ``None``. None delegates to the local fallback shown in this body; it
                is not an additional numeric/category value.

            chrom : str or int or None
                Chromosome identifier matched against source metadata as a string. None leaves the
                chromosome unrestricted where supported. Default: ``None``. Passed to
                ``self._region_indices`` as ``chrom``.

            start : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``. Passed to ``self._region_indices`` as ``start``.

            end : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``. Passed to ``self._region_indices`` as ``end``.

            snps : array-like or pandas.DataFrame
                Genotype values with animals in rows and variants in columns, or the genotype source
                accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
                labels and the coding expected by the selected kernel or association model. Default:
                ``None``. None delegates to the local fallback shown in this body; it is not an
                additional numeric/category value. Passed to ``self._region_indices`` as ``snps``.

            **kwargs : mapping
                Additional keyword arguments forwarded to ``virtual_ld_matrix``, ``virtual_ld_subset``.
                See those callables for accepted keys and values. Passed to ``virtual_ld_matrix`` as
                ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``virtual_ld_subset(self.prefix, np.asarray(ii, np.int64), jj, n_samples=int(self._obj.attrs['n_samples']), source_n_variants=int(self._obj.attrs['n_variants']), names=np.asarray(self._obj.coords['sn...``; see the summary and the called implementation for its labeled/numerical representation.
            """
            ii=indices_i if indices_i is not None else snp_indices
            if ii is None: ii=self._region_indices(chrom=chrom,start=start,end=end,snps=snps)
            if ii is None:
                import warnings
                if int(self._obj.attrs.get('n_variants',0))>500_000:
                    warnings.warn('Full-coordinate LD duplicates very large SNP label indexes; prefer selection-first ld_matrix(...) or ld_region(...).',RuntimeWarning,stacklevel=2)
                return virtual_ld_matrix(self.prefix,**kwargs)
            jj=np.asarray(indices_j,np.int64) if indices_j is not None else np.asarray(ii,np.int64)
            return virtual_ld_subset(self.prefix,np.asarray(ii,np.int64),jj,n_samples=int(self._obj.attrs['n_samples']),source_n_variants=int(self._obj.attrs['n_variants']),
                names=np.asarray(self._obj.coords['snp'].data),chrom=np.asarray(self._obj.coords['chrom'].data),pos=np.asarray(self._obj.coords['pos'].data),**kwargs)
        def ld_region(self, *, snp_indices=None, chrom=None, start=None, end=None, snps=None, **kwargs):
            """Return the linkage disequilibrium region result computed by ``self.ld_matrix``.

            Parameters
            ----------
            snp_indices : one-dimensional integer array or None
                Zero-based variant positions to select from the current source. Values must refer to
                existing columns; None retains all eligible variants. Streaming readers may additionally
                require sorted indices. Default: ``None``. Passed to ``self._region_indices`` as
                ``snp_indices``.

            chrom : str or int or None
                Chromosome identifier matched against source metadata as a string. None leaves the
                chromosome unrestricted where supported. Default: ``None``. Passed to
                ``self._region_indices`` as ``chrom``.

            start : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``. Passed to ``self._region_indices`` as ``start``.

            end : int or None
                Boundary index or genomic coordinate for the requested interval. Use the coordinate
                convention of this reader/query; None leaves that boundary unrestricted where supported.
                Default: ``None``. Passed to ``self._region_indices`` as ``end``.

            snps : array-like or pandas.DataFrame
                Genotype values with animals in rows and variants in columns, or the genotype source
                accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
                labels and the coding expected by the selected kernel or association model. Default:
                ``None``. None delegates to the local fallback shown in this body; it is not an
                additional numeric/category value. Passed to ``self._region_indices`` as ``snps``.

            **kwargs : mapping
                Additional keyword arguments forwarded to ``self.ld_matrix``. See those callables for
                accepted keys and values. Passed to ``self.ld_matrix`` as ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``self.ld_matrix(snp_indices=ii, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            ii=self._region_indices(snp_indices=snp_indices,chrom=chrom,start=start,end=end,snps=snps)
            if ii is None: raise ValueError('ld_region requires snp_indices, snps, or chrom with optional start/end')
            return self.ld_matrix(snp_indices=ii,**kwargs)
        def offset_r2(self,**kwargs): 
            """Return the offset squared correlation result computed by ``virtual_offset_r2``.

            Parameters
            ----------
            **kwargs : mapping
                Additional keyword arguments forwarded to ``virtual_offset_r2``. See those callables for
                accepted keys and values. Passed to ``virtual_offset_r2`` as ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``virtual_offset_r2(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return virtual_offset_r2(self.prefix,**kwargs)
        def grm(self,representation='virtual',**kwargs):
            """Return the requested dense, virtual or low-rank packed-genotype relationship representation.

            Parameters
            ----------
            representation : str
                Returned/stored container or kernel representation. The source-derived choices below
                select dense arrays, labeled tables or lazy/operator representations; see the local
                dispatcher for which are supported. Default: ``'virtual'``. Source-recognized literal
                values: ``factor``, ``lowrank``, ``matrix``, ``sketch``. These select the corresponding
                branches below; comparison evidence alone does not establish an exhaustive enum.

            **kwargs : mapping
                Additional keyword arguments forwarded to ``lowrank_from_countsketch``,
                ``dense_grm_from_bed``, ``virtual_grm``. See those callables for accepted keys and
                values. Passed to ``lowrank_from_countsketch`` as ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``virtual_grm(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            rep=str(representation).lower()
            if rep == 'lowrank':
                return lowrank_grm_from_bed(self.prefix,**kwargs)
            if rep in {'factor','sketch'}:
                from .packed_experimental import lowrank_from_countsketch
                return lowrank_from_countsketch(self.prefix,**kwargs)
            if rep in {'dense','matrix'}: return dense_grm_from_bed(self.prefix,**kwargs)
            return virtual_grm(self.prefix,**kwargs)
        def dense_grm(self,**kwargs): 
            """Construct dense genomic relationship matrix.

            Parameters
            ----------
            **kwargs : mapping
                Additional keyword arguments forwarded to ``dense_grm_from_bed``. See those callables
                for accepted keys and values. Passed to ``dense_grm_from_bed`` as ``**expanded
                mapping``.

            Returns
            -------
            object
                Computed result from ``dense_grm_from_bed(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return dense_grm_from_bed(self.prefix,**kwargs)
        def lowrank_grm(self,**kwargs): 
            """Construct low-rank genomic relationship matrix.

            Parameters
            ----------
            **kwargs : mapping
                Additional keyword arguments forwarded to ``lowrank_grm_from_bed``. See those callables
                for accepted keys and values. Passed to ``lowrank_grm_from_bed`` as ``**expanded
                mapping``.

            Returns
            -------
            object
                Computed result from ``lowrank_grm_from_bed(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return lowrank_grm_from_bed(self.prefix,**kwargs)
        def grm_operator(self,**kwargs): 
            """Return the genomic relationship matrix operator result computed by ``additive_grm_operator_from_bed``.

            Parameters
            ----------
            **kwargs : mapping
                Additional keyword arguments forwarded to ``additive_grm_operator_from_bed``. See those
                callables for accepted keys and values. Passed to ``additive_grm_operator_from_bed`` as
                ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``additive_grm_operator_from_bed(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return additive_grm_operator_from_bed(self.prefix,**kwargs)
        def sketch_factor(self,engine='direct',**kwargs):
            """Construct a packed genotype CountSketch factor using the selected decoding engine.

            Parameters
            ----------
            engine : str
                Execution/backend selection for this operation. See the source-derived literal
                descriptions below and the called resolver; automatic selection uses the available
                implementation and input representation. Default: ``'direct'``. Source-recognized
                literal values: ``2bit``, ``packed``. These select the corresponding branches below;
                comparison evidence alone does not establish an exhaustive enum.

            **kwargs : mapping
                Additional keyword arguments forwarded to ``packed_countsketch_factor_direct``,
                ``packed_countsketch_factor``. See those callables for accepted keys and values. Passed
                to ``packed_countsketch_factor_direct`` as ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``packed_countsketch_factor(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            if str(engine).lower() in {'direct','packed','2bit'}:
                from .packed_experimental import packed_countsketch_factor_direct
                return packed_countsketch_factor_direct(self.prefix,**kwargs)
            from .packed_experimental import packed_countsketch_factor
            return packed_countsketch_factor(self.prefix,**kwargs)
        def countsketch_grm(self,**kwargs):
            """Construct low-rank genomic relationship matrix.

            Parameters
            ----------
            **kwargs : mapping
                Additional keyword arguments forwarded to ``lowrank_from_countsketch``. See those
                callables for accepted keys and values. Passed to ``lowrank_from_countsketch`` as
                ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``lowrank_from_countsketch(self.prefix, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            from .packed_experimental import lowrank_from_countsketch
            return lowrank_from_countsketch(self.prefix,**kwargs)
except Exception:
    PackedBedAccessor=None


__all__=['open_packed_bed_xarray','virtual_ld_matrix','virtual_ld_subset','virtual_offset_r2','virtual_grm',
         'dense_grm_from_bed','dense_additive_grm_bitserial','lowrank_grm_from_bed','additive_grm_operator_from_bed','recommended_grm_variant_chunk','packed_variant_stats',
         'PackedBedLDBackend','PackedBedOffsetLDBackend','PackedBedGRMBackend']


def stream_loco_from_bed(prefix, genomewide, chromosome, *, components=None, sample_ids=None,
                         fit_mask=None, min_maf=1e-4, chunk_variants='auto', memory_fraction=None,
                         maxmem=None, output_dtype=np.float32, blas_threads='auto', progress=False):
    """Build one exact temporary chromosome GRM and subtract it from genome-wide.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    genomewide : object (see the cited input contract)
        Input for the operation ``genomewide['weights'][c] - chrom['weights'][c]``. See
        ``vigor.packed_bed.stream_loco_from_bed`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Passed to ``dense_grm_from_bed`` as
        ``chromosome``.

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
        Keep evaluation outcomes outside this mask. Default: ``None``. Passed to
        ``dense_grm_from_bed`` as ``fit_mask``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.
        Passed to ``dense_grm_from_bed`` as ``min_maf``.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``. Passed
        to ``dense_grm_from_bed`` as ``chunk_variants``.

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
    dict
        Result fields: ``kernels``, ``weights``, ``chromosome``, ``chromosome_result``, ``storage``. Field arrays preserve the corresponding input axis order.
    """
    comps=list(genomewide['kernels']) if components is None else list(components)
    chrom=dense_grm_from_bed(prefix,components=comps,sample_ids=sample_ids,fit_mask=fit_mask,
        chromosome=str(chromosome),min_maf=min_maf,chunk_variants=chunk_variants,
        memory_fraction=memory_fraction,maxmem=maxmem,output_dtype=output_dtype,
        blas_threads=blas_threads,progress=progress)
    out={}
    for comp in comps:
        wall=float(genomewide['weights'][comp]); wc=float(chrom['weights'][comp])
        if wall<=wc or wc<=0: raise ValueError(f"invalid LOCO weights for {comp}: all={wall}, chr={wc}")
        Kall=np.asarray(genomewide['kernels'][comp],np.float64); Kc=np.asarray(chrom['kernels'][comp],np.float64)
        K=((wall*Kall-wc*Kc)/(wall-wc)).astype(output_dtype)
        ref=genomewide['kernels'][comp]
        xr,_,_=_require_xarray(); out[comp]=xr.DataArray(K,dims=ref.dims,coords=ref.coords,
            attrs={'loco_chromosome':str(chromosome),'normalization':'vitezica_global_variant_denominator'})
    return {'kernels':out,'weights':{c:float(genomewide['weights'][c]-chrom['weights'][c]) for c in comps},
            'chromosome':str(chromosome),'chromosome_result':chrom,'storage':'temporary_in_memory'}


class PackedScoreTables:
    """Per-active-BED-byte lookup tables for direct BED -> X' A reductions.

    ``active_bytes`` contains only source BED bytes with >=1 fitted sample.  The
    score kernels therefore skip bytes that are guaranteed to contribute zero,
    which matters when the PLINK cohort is much larger than the analysis set.
    """
    def __init__(self,u_lut,aobs_lut,mask4,n_full,sample_indices,rank,*,active_bytes=None,source_bpv=None):
        """Initialize PackedScoreTables with the inputs and resource settings below.

        Parameters
        ----------
        u_lut : numpy.ndarray
            Lookup table for retained-eigenvector contributions of packed genotype codes, prepared
            for the same sample selection and eigensystem. Its byte/code layout must match
            PackedScoreTables.

        aobs_lut : numpy.ndarray
            Packed-code lookup table for observed-cell additive score terms, prepared with the same
            null masks and byte layout as the score tables.

        mask4 : bool or numeric mask array
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis.

        n_full : int
            Logical/packed source extent needed to decode or trim padded genotype storage. Use the
            actual source metadata count, including byte rounding only where requested.

        sample_indices : one-dimensional integer array or None
            Zero-based sample positions to retain, in the requested output order. Values must be
            valid positions in the source; None selects the source's full sample axis.

        rank : int, str, sequence or mapping
            Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
            or trait dimension. Full and automatic policies depend on the decomposition; see the
            rank resolver used by this routine.

        active_bytes : one-dimensional integer array
            Indices of source BED sample bytes needed by the selected animal rows. Values must refer
            to the full source bytes-per-variant extent. Default: ``None``. None delegates to the
            local fallback shown in this body; it is not an additional numeric/category value.

        source_bpv : int or None
            Number of bytes per variant in the source BED fileset. None derives ceil(n_full/4);
            supply the source-wide count when scoring a selected sample subset. Default: ``None``.
        """
        self.u_lut=u_lut; self.aobs_lut=aobs_lut; self.mask4=mask4
        self.n_full=int(n_full); self.sample_indices=np.asarray(sample_indices,np.int64); self.rank=int(rank)
        self.active_bytes=(np.arange(len(mask4),dtype=np.int64) if active_bytes is None else np.asarray(active_bytes,np.int64))
        self.source_bpv=int(source_bpv if source_bpv is not None else (self.n_full+3)//4)


def build_packed_score_tables(A, n_full, *, sample_indices=None, fit_mask=None, dtype=np.float32):
    """Precompute 256-state byte tables for a fixed multivariate score dual A.

    Parameters
    ----------
    A : array-like
        Numerical operand in ``A[fit]``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    n_full : int
        Logical/packed source extent needed to decode or trim padded genotype storage. Use the
        actual source metadata count, including byte rounding only where requested.

    sample_indices : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Default:
        ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``. Passed to ``np.zeros`` as ``dtype``.

    Returns
    -------
    object
        Computed result from ``PackedScoreTables(u, ao, mask4, n_full, si, r, active_bytes=active, source_bpv=bpv)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    A=np.asarray(A,dtype=dtype); nout,r=A.shape; n_full=int(n_full)
    si=np.arange(n_full,dtype=np.int64) if sample_indices is None else np.asarray(sample_indices,np.int64)
    if len(si)!=nout: raise ValueError('sample_indices must map every row of A')
    fit=np.ones(nout,bool) if fit_mask is None else np.asarray(fit_mask,bool)
    if fit.shape!=(nout,): raise ValueError('fit_mask shape mismatch')
    Afull=np.zeros((n_full,r),dtype=dtype); Afull[si[fit]]=A[fit]
    bpv=(n_full+3)//4; pad=np.zeros((bpv*4,r),dtype=dtype); pad[:n_full]=Afull; A4=pad.reshape(bpv,4,r)
    fullfit=np.zeros(n_full,bool); fullfit[si[fit]]=True; mask4_full=sample_mask4(n_full,fullfit)
    active=np.flatnonzero(mask4_full).astype(np.int64); A4=A4[active]; mask4=mask4_full[active]
    gmap=np.asarray([0,-1,1,2],dtype=np.int8); shifts=2*np.arange(4,dtype=np.uint8)
    G=gmap[(np.arange(256,dtype=np.uint16)[:,None] >> shifts[None,:]) & 3]
    obs=(G>=0).astype(dtype); G0=np.where(G>=0,G,0).astype(dtype)
    u=np.einsum('ck,bkr->bcr',G0,A4,optimize=True).astype(dtype,copy=False)
    ao=np.einsum('ck,bkr->bcr',obs,A4,optimize=True).astype(dtype,copy=False)
    return PackedScoreTables(u,ao,mask4,n_full,si,r,active_bytes=active,source_bpv=bpv)

def _active_score_packed(packed, tables):
    """Gather only packed sample bytes needed by the current observed-score plan.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    tables : object (see the cited input contract)
        Input for the operation ``P[:, tables.active_bytes]``. See
        ``vigor.packed_bed._active_score_packed`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.
    """
    P=np.asarray(packed,np.uint8)
    if P.ndim!=2: raise ValueError('packed BED block must be variants x bytes')
    if P.shape[1]==len(tables.mask4): return P
    if P.shape[1]==int(tables.source_bpv): return np.asarray(P[:,tables.active_bytes],np.uint8)
    raise ValueError('packed BED block byte dimension does not match source or compact score tables')


def packed_score_components(packed, tables, *, byte_block='auto', accum_dtype=np.float64):
    """Exact missing-aware score sufficient statistics directly from BED bytes.

    Returns ``U=sum (g-mu)A``, centered genotype sum-of-squares and nobs for
    every variant. No sample x variant dosage matrix is constructed.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    tables : object (see the cited input contract)
        Input for the operation ``tables.u_lut[bb[None, :], codes]``. See
        ``vigor.packed_bed.packed_score_components`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    byte_block : int or 'auto'
        Packed sample bytes (4 animals each) gathered per stripe. 'auto' sizes the stripe so the
        two (variants x bytes x rank) gather temporaries use about 64 MiB, never fewer than 64
        bytes. Default: ``'auto'``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``. Passed to ``np.zeros`` as ``dtype``.

    Returns
    -------
    object
        Computed result from ``(U, ss, cnt)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    P=_active_score_packed(packed,tables)
    cnt,sx,s2=packed_variant_stats(P,tables.mask4); m=len(P); r=tables.rank
    Uraw=np.zeros((m,r),dtype=accum_dtype); As=np.zeros((m,r),dtype=accum_dtype)
    if str(byte_block).lower()=='auto':
        # The two LUT gathers materialize (m, bs, r) temporaries; a fixed 64-byte stripe
        # meant thousands of tiny NumPy passes at biobank n, so size it from that cost.
        item=max(np.dtype(tables.u_lut.dtype).itemsize,np.dtype(accum_dtype).itemsize)
        bs=max(64,(64*1024**2)//max(1,2*m*r*item))
    else: bs=max(1,int(byte_block))
    for b0 in range(0,P.shape[1],bs):
        b1=min(P.shape[1],b0+bs); bb=np.arange(b0,b1,dtype=np.int64); codes=P[:,b0:b1]
        # Gather only a bounded byte stripe; this avoids an n x m decoded matrix.
        Uraw += tables.u_lut[bb[None,:],codes].sum(axis=1,dtype=accum_dtype)
        As += tables.aobs_lut[bb[None,:],codes].sum(axis=1,dtype=accum_dtype)
    # Per-variant moments are O(m); keep them in float64 to avoid cancellation in s2 - sx^2/n.
    cntf=cnt.astype(np.float64); mu=np.divide(sx,cntf,out=np.zeros(m),where=cnt>0)
    ss=np.maximum(s2.astype(np.float64)-np.divide(sx.astype(np.float64)**2,cntf,out=np.zeros(m),where=cnt>0),0.0)
    U=(Uraw-mu[:,None]*As).astype(accum_dtype,copy=False); ss=ss.astype(accum_dtype,copy=False)
    return U,ss,cnt




@lru_cache(maxsize=1)
def _packed_score_numba_kernel():
    """Compile the direct byte-LUT reducer lazily; VIGOR import stays lightweight.
    """
    import numba
    @numba.njit(parallel=True,fastmath=False,cache=False)
    def kernel(P,u,ao,mask4,cntl,sxl,s2l):
        """Execute the captured compiled numerical kernel on aligned input buffers.

        Parameters
        ----------
        P : array-like or pandas.DataFrame
            Baseline predicted values in the response animal/trait order, used for conditioning,
            residual construction or evaluation. Supply predictions from the corresponding fit
            without including held-out outcomes in training.

        u : object (see the cited input contract)
            Input for the operation ``u.shape[2]``. See
            ``vigor.packed_bed._packed_score_numba_kernel.kernel`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        ao : object (see the cited input contract)
            Input for the operation ``ao[b, code, q]``. See
            ``vigor.packed_bed._packed_score_numba_kernel.kernel`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        mask4 : bool or numeric mask array
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis.

        cntl : object (see the cited input contract)
            Input for the operation ``cntl[mk, code]``. See
            ``vigor.packed_bed._packed_score_numba_kernel.kernel`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        sxl : object (see the cited input contract)
            Input for the operation ``sxl[mk, code]``. See
            ``vigor.packed_bed._packed_score_numba_kernel.kernel`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        s2l : object (see the cited input contract)
            Input for the operation ``s2l[mk, code]``. See
            ``vigor.packed_bed._packed_score_numba_kernel.kernel`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        Returns
        -------
        object
            Computed result from ``(U, ss, cnt, sxout)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        m,bpv=P.shape; r=u.shape[2]
        U=np.zeros((m,r),np.float64); ss=np.zeros(m,np.float64); cnt=np.zeros(m,np.int32); sxout=np.zeros(m,np.int32)
        for v in numba.prange(m):
            sx=0.0; s2=0.0; nn=0; ur=np.zeros(r,np.float64); ar=np.zeros(r,np.float64)
            for b in range(bpv):
                code=P[v,b]; mk=mask4[b]; nn+=cntl[mk,code]; sx+=sxl[mk,code]; s2+=s2l[mk,code]
                for q in range(r):
                    ur[q]+=u[b,code,q]; ar[q]+=ao[b,code,q]
            cnt[v]=nn; sxout[v]=int(sx)
            if nn>0:
                mu=sx/nn; ss[v]=max(s2-sx*sx/nn,0.0)
                for q in range(r): U[v,q]=ur[q]-mu*ar[q]
        return U,ss,cnt,sxout
    return kernel


def packed_score_components_fast(packed, tables, *, return_sx=False):
    """Compiled exact-missingness BED score reduction without dosage decoding.

    JIT compilation is deliberately lazy. If the runtime environment cannot
    compile (for example because the process current directory was removed),
    fall back to the exact NumPy byte-LUT reducer rather than decoding dosages
    or failing the scan.

    Parameters
    ----------
    packed : array-like of uint8
        Packed genotype bytes. Axis layout must follow the selected BED/canonical packed
        representation; each byte encodes four 2-bit calls, and missing-code interpretation
        follows that format.

    tables : object (see the cited input contract)
        Input for the operation ``_active_score_packed(packed, tables)``. See
        ``vigor.packed_bed.packed_score_components_fast`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    return_sx : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``(U, ss, cnt, sx)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    cntl,sxl,s2l=bed_variant_summary_luts()
    P=_active_score_packed(packed,tables)
    try:
        # Float32 LUT storage does not relax the float64 accumulation contract.
        kern=_packed_score_numba_kernel()
        z=kern(P,tables.u_lut,tables.aobs_lut,tables.mask4,cntl,sxl,s2l)
        return z if return_sx else z[:3]
    except Exception:
        U,ss,cnt=packed_score_components(P,tables)
        if not return_sx:return U,ss,cnt
        _,sx,_=packed_variant_stats(P,tables.mask4)
        return U,ss,cnt,sx


class PackedBedDosageBackend(BackendArray):
    """Virtual sample x SNP dosage view over the packed BED payload."""
    __slots__=("prefix","n_samples","n_variants","bytes_per_variant","_packed")
    def __init__(self,prefix,n_samples,n_variants):
        """Initialize PackedBedDosageBackend with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        n_samples : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.

        n_variants : int
            Number of variants in the logical source axis, including the unpadded extent of packed
            arrays.
        """
        self.prefix=str(prefix); self.n_samples=int(n_samples); self.n_variants=int(n_variants)
        self.bytes_per_variant=(self.n_samples+3)//4
        self._packed=open_bed_memmap(self.prefix+".bed", n_variants=self.n_variants)
    @property
    def shape(self): 
        """Expose the logical array/operator dimensions without materializing its data.

        Returns
        -------
        object
            Computed result from ``(self.n_samples, self.n_variants)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return (self.n_samples,self.n_variants)
    @property
    def dtype(self): 
        """Return the dtype result computed by ``np.dtype``.

        Returns
        -------
        object
            Computed result from ``np.dtype(np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.dtype(np.float32)
    def __getitem__(self,key):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``_xr_indexing.explicit_indexing_adapter(key, self.shape, _xr_indexing.IndexingSupport.OUTER, self._raw_getitem)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return _xr_indexing.explicit_indexing_adapter(
            key,self.shape,_xr_indexing.IndexingSupport.OUTER,self._raw_getitem)
    def _raw_getitem(self,key):
        """Decode the requested packed-genotype selection into a bounded numerical block.

        Parameters
        ----------
        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        ii,si=_as_indices(key[0],self.n_samples); jj,sj=_as_indices(key[1],self.n_variants)
        P=np.asarray(self._packed[jj],np.uint8)
        from .io import decode_bed
        out=decode_bed(P,self.n_samples,ii,dtype=np.float32)
        if si and sj: return np.asarray(out[0,0],np.float32)
        if si: return out[0]
        if sj: return out[:,0]
        return out

def virtual_dosage(prefix):
    """Return a virtual xarray dosage matrix; decode happens after iid/SNP indexing.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants. Passed to ``xr.Variable`` as ``attrs``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    xr,_,indexing=_require_xarray(); ids,names,chrom,pos,labels_loaded=_packed_coordinate_metadata(prefix)
    backend=PackedBedDosageBackend(prefix,len(ids),len(pos))
    data=indexing.LazilyIndexedArray(backend)
    var=xr.Variable(('iid','snp'),data,attrs={'representation':'plink1_bed_2bit_virtual_decode',
        'physical_backend':'plink1_bed_2bit','prefix':str(prefix),
        'n_full_samples':int(len(ids)),'n_full_variants':int(len(pos)),
        'missing_value':np.nan,'virtual':True,'genotype_complete':False})
    cmap={'iid':('iid',ids),'snp':('snp',names),
        'sample_index':('iid',np.arange(len(ids),dtype=np.int64)),
        'variant_index':('snp',np.arange(len(names),dtype=np.int64)),
        'chrom':('snp',chrom),'pos':('snp',pos)}
    coords=xr.Coordinates(cmap,indexes={} if not labels_loaded else None)
    out=xr.DataArray(var,coords=coords,name='genotypes')
    if not labels_loaded: out=out.set_xindex('iid')
    out.attrs['snp_label_index_materialized']=bool(labels_loaded)
    out.attrs['variant_id_mode']='bim_exact' if labels_loaded else 'integer_index_with_chrom_pos'
    return out

if PackedBedAccessor is not None:
    def _packed_accessor_dosage(self): 
        """Return the packed accessor dosage result computed by ``virtual_dosage``.
        """
        return virtual_dosage(self.prefix)
    PackedBedAccessor.dosage=_packed_accessor_dosage

__all__ += ['virtual_dosage','PackedBedDosageBackend']
