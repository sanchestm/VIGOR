"""Sparse relationship/precision matrices in Zarr v3 with labeled xarray views.

The on-disk representation is CSR (data/indices/indptr + sample IDs).  It is
appropriate for Henderson A^-1 and externally computed single-step H/H^-1
matrices and never materializes an N x N dense array.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from .store import canonical_provenance, finalize_zarr_artifact


def _zarr():
    """Import the Zarr backend required for sparse relationship persistence.
        
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Sparse Zarr storage requires `pip install 'VIGOR[io]'`.") from exc
    return zarr


def _unicode_dtype(values):
    """Return the unicode dtype result computed by ``np.dtype``.

    Parameters
    ----------
    values : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.
    """
    a=np.asarray(values,dtype=str)
    width=max([1]+[len(x) for x in a.tolist()])
    return np.dtype(f"U{width}")


def _create(g,name,*,shape,chunks,dtype,dimension_names,overwrite=True):
    """Create a sparse relationship backing array with the requested chunk/storage metadata.

    Parameters
    ----------
    g : object (see the cited input contract)
        Input for the operation ``g.create_array(**kw)``. See
        ``vigor.sparse_relationship._create`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    shape : tuple of int or None
        Requested array/distribution extents in axis order. Nonnegative sizes must agree with
        the associated data; None uses the backend's inferred scalar/default shape.

    chunks : tuple, mapping or None
        Chunk/shard extents per persisted array axis. Positive extents must follow the array's
        dimension order; None lets the writer select extents. Shards group multiple chunks for
        Zarr v3 storage.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision.

    dimension_names : sequence of str or None
        Ordered names of array dimensions for persistent metadata. Length must equal array rank;
        None omits explicit dimension-name metadata.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``True``.
    """
    kw=dict(name=name,shape=tuple(map(int,shape)),chunks=tuple(map(int,chunks)),
            dtype=np.dtype(dtype),overwrite=bool(overwrite),dimension_names=tuple(dimension_names),
            shards=tuple(int(c)*4 for c in chunks))
    try:
        a=g.create_array(**kw)
    except TypeError:  # zarr versions before dimension_names support
        kw.pop("dimension_names",None); a=g.create_array(**kw)
    a.attrs["_ARRAY_DIMENSIONS"]=list(dimension_names)
    return a


def write_sparse_relationship_zarr(matrix, destination, *, ids=None, group="relationship/Hinv",
                                   name="Hinv", overwrite=False, data_chunk=1_000_000,
                                   pointer_chunk=65_536, zarr_format=3, attrs=None):
    """Write a scipy sparse relationship matrix as canonical CSR Zarr v3.

    ``matrix`` can be A, A^-1, H, H^-1, kinship, or another labeled symmetric
    relationship object. The representation is generic; ``name`` records the
    scientific meaning without changing the storage layout.

    Parameters
    ----------
    matrix : object (see the cited input contract)
        Input for the operation ``sparse.csr_matrix(matrix)``. See
        ``vigor.sparse_relationship.write_sparse_relationship_zarr`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``. Passed to ``_create`` as ``dtype``.

    group : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.
        Default: ``'relationship/Hinv'``. Passed to ``SparseZarrCSR`` as ``group``.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``'Hinv'``. Passed to ``canonical_provenance`` as ``matrix_name``.

    overwrite : bool
        Whether an existing destination may be replaced. False protects the existing artifact;
        True permits this writer to recreate it. Default: ``False``.

    data_chunk : array-like or numeric scalar
        Input supplied to ``_create`` as ``chunks``. See that callable's documented ``chunks``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``1000000``. Passed to ``_create`` as ``chunks``.

    pointer_chunk : array-like or numeric scalar
        Input supplied to ``_create`` as ``chunks``. See that callable's documented ``chunks``
        contract for its type, alignment and value domain. This wrapper retains that contract.
        Default: ``65536``. Passed to ``_create`` as ``chunks``.

    zarr_format : int
        Input supplied to ``zarr.open_group`` as ``zarr_format``. See that callable's documented
        ``zarr_format`` contract for its type, alignment and value domain. This wrapper retains
        that contract. Default: ``3``. Passed to ``zarr.open_group`` as ``zarr_format``.

    attrs : mapping or None
        Named settings or metadata forwarded to the specific operation below. See that
        operation's argument/schema contract for accepted fields; None uses its default/empty
        configuration. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``SparseZarrCSR(destination, group=group)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    zarr=_zarr(); M=sparse.csr_matrix(matrix)
    if M.shape[0]!=M.shape[1]: raise ValueError("relationship matrix must be square")
    M.sum_duplicates(); M.sort_indices()
    n=M.shape[0]
    if ids is None: ids=np.asarray([str(i) for i in range(n)],dtype=str)
    else: ids=np.asarray(ids,dtype=str)
    if len(ids)!=n: raise ValueError("ids length must match matrix dimension")
    if len(np.unique(ids))!=n: raise ValueError("relationship IDs must be unique")
    from .store import canonical_provenance
    root=zarr.open_group(str(destination),mode="a",zarr_format=int(zarr_format))
    for k,v in canonical_provenance("sparse_relationship",write_state="writing").items():
        if k not in root.attrs: root.attrs[k]=v
    root.attrs.update({k:v for k,v in canonical_provenance("relationship_store",write_state="writing").items() if k not in root.attrs})
    parts=[x for x in str(group).strip("/").split("/") if x]
    g=root
    for part in parts[:-1]: g=g.require_group(part)
    leaf=parts[-1] if parts else str(name)
    if leaf in g:
        if not overwrite: raise FileExistsError(f"Zarr group already exists: {group}")
        del g[leaf]
    g=g.require_group(leaf)
    nnz=int(M.nnz)
    da=_create(g,"data",shape=(nnz,),chunks=(max(1,min(nnz,int(data_chunk))),),dtype=M.data.dtype,dimension_names=("nnz",))
    ia=_create(g,"indices",shape=(nnz,),chunks=(max(1,min(nnz,int(data_chunk))),),dtype=np.int64,dimension_names=("nnz",))
    pa=_create(g,"indptr",shape=(n+1,),chunks=(max(1,min(n+1,int(pointer_chunk))),),dtype=np.int64,dimension_names=("sample_ptr",))
    sa=_create(g,"sample",shape=(n,),chunks=(max(1,min(n,int(pointer_chunk))),),dtype=_unicode_dtype(ids),dimension_names=("sample",))
    if nnz:
        da[:]=M.data; ia[:]=M.indices.astype(np.int64,copy=False)
    pa[:]=M.indptr.astype(np.int64,copy=False); sa[:]=ids
    g.attrs.update({"domino_sparse_format":"csr-v1","matrix_name":str(name),
                    "shape":[int(n),int(n)],"nnz":nnz,"symmetric":True,
                    "sample_dimension":"sample","storage":"scipy_csr"})
    if attrs: g.attrs.update(dict(attrs))
    g.attrs.update(canonical_provenance("sparse_relationship",write_state="complete",matrix_name=str(name)))
    finalize_zarr_artifact(str(destination),artifact_type="sparse_relationship",metadata={"matrix_name":str(name),"n_samples":int(n),"nnz":int(nnz)})
    return SparseZarrCSR(destination,group=group)


@dataclass
class SparseZarrCSR:
    """Lazy row-block operator over a CSR matrix stored in Zarr v3.

    Constructor fields
    ------------------
    destination : str | Path
        Stored destination consumed by ``__post_init__``; see those methods for the
        numerical/label contract. Required construction field.

    group : str
        Stored group consumed by ``__post_init__``; see those methods for the numerical/label
        contract. Default: ``'relationship/Hinv'``.

    row_chunk : int
        Stored row chunk consumed by ``matmat``; see those methods for the numerical/label
        contract. Default: ``8192``.
    """
    destination: str | Path
    group: str="relationship/Hinv"
    row_chunk: int=8192

    def __post_init__(self):
        """Validate/coerce the constructed dataclass fields for subsequent model operations.
                
        """
        zarr=_zarr(); self.root=zarr.open_group(str(self.destination),mode="r")
        g=self.root
        for part in [x for x in str(self.group).strip("/").split("/") if x]: g=g[part]
        self.g=g
        if str(g.attrs.get("domino_sparse_format",""))!="csr-v1": raise ValueError("not a VIGOR CSR relationship group")
        self.shape=tuple(map(int,g.attrs["shape"])); self.dtype=np.dtype(g["data"].dtype)
        self.ids=np.asarray(g["sample"][:],dtype=str)
        # O(N), not O(N^2); keeping pointers resident makes row-block access cheap.
        self.indptr=np.asarray(g["indptr"][:],dtype=np.int64)

    @property
    def nnz(self): 
        """Return the nnz result computed by ``int``.

        Returns
        -------
        object
            Computed result from ``int(self.g.attrs.get('nnz', int(self.indptr[-1])))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return int(self.g.attrs.get("nnz",int(self.indptr[-1])))

    def row_block(self,start,stop):
        """Return the row block result computed by ``sparse.csr_matrix``.

        Parameters
        ----------
        start : int or None
            Boundary index or genomic coordinate for the requested interval. Use the coordinate
            convention of this reader/query; None leaves that boundary unrestricted where supported.
            Passed to ``sparse.csr_matrix`` as ``shape``.

        stop : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions. Passed to ``sparse.csr_matrix`` as ``shape``.

        Returns
        -------
        object
            Computed result from ``sparse.csr_matrix((data, indices, ptr), shape=(stop - start, self.shape[1]))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        start=max(0,int(start)); stop=min(int(stop),self.shape[0])
        if stop<start: stop=start
        lo=int(self.indptr[start]); hi=int(self.indptr[stop])
        data=np.asarray(self.g["data"][lo:hi])
        indices=np.asarray(self.g["indices"][lo:hi],dtype=np.int64)
        ptr=self.indptr[start:stop+1]-lo
        return sparse.csr_matrix((data,indices,ptr),shape=(stop-start,self.shape[1]))

    def matmat(self,X):
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support. Passed to ``np.empty`` as ``dtype``.

        Returns
        -------
        object
            Computed result from ``out[:, 0] if one else out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X=np.asarray(X); one=X.ndim==1
        if one: X=X[:,None]
        if X.shape[0]!=self.shape[1]: raise ValueError("right-hand side dimension mismatch")
        out=np.empty((self.shape[0],X.shape[1]),dtype=np.result_type(self.dtype,X.dtype,np.float32))
        for a in range(0,self.shape[0],max(1,int(self.row_chunk))):
            b=min(self.shape[0],a+max(1,int(self.row_chunk)))
            out[a:b]=self.row_block(a,b)@X
        return out[:,0] if one else out

    def matvec(self,x): 
        """Apply the enclosing covariance/kernel operator to one vector.

        Parameters
        ----------
        x : one-dimensional array-like
            Vector supplied by the linear operator/iterative-solver callback. Length equals the
            enclosing operator dimension; observed-cell operators use the prepared finite-cell
            order.

        Returns
        -------
        object
            Computed result from ``self.matmat(x)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.matmat(x)

    def to_scipy(self):
        """Return the to scipy result computed by ``sparse.csr_matrix``.

        Returns
        -------
        object
            Computed result from ``sparse.csr_matrix((data, indices, self.indptr.copy()), shape=self.shape)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        data=np.asarray(self.g["data"][:]); indices=np.asarray(self.g["indices"][:],dtype=np.int64)
        return sparse.csr_matrix((data,indices,self.indptr.copy()),shape=self.shape)

    def to_xarray(self,lazy=True):
        """Return an xarray Dataset exposing the sparse CSR arrays, not dense N x N.

        Parameters
        ----------
        lazy : bool
            Whether store-backed values remain lazily loaded until their blocks are accessed; eager
            mode materializes the arrays requested by the reader. Default: ``True``.

        Returns
        -------
        object
            Computed result from ``ds``; see the summary and the called implementation for its labeled/numerical representation.
        """
        try: import xarray as xr
        except ImportError as exc: raise ImportError("xarray view requires `pip install xarray`.") from exc
        if lazy:
            try:
                import dask.array as da
                data=da.from_zarr(self.g["data"]); indices=da.from_zarr(self.g["indices"])
                indptr=da.from_zarr(self.g["indptr"])
            except Exception:
                data=self.g["data"]; indices=self.g["indices"]; indptr=self.g["indptr"]
        else:
            data=np.asarray(self.g["data"][:]); indices=np.asarray(self.g["indices"][:]); indptr=self.indptr.copy()
        ds=xr.Dataset({"data":(("nnz",),data),"indices":(("nnz",),indices),"indptr":(("sample_ptr",),indptr)},
                      coords={"sample":(("sample",),self.ids)})
        ds.attrs.update(dict(self.g.attrs)); ds.attrs["sparse_matrix_note"]="CSR arrays; use SparseZarrCSR for matvec/row access without densification"
        return ds


def open_sparse_relationship_zarr(destination, *, group="relationship/Hinv", row_chunk=8192):
    """Open sparse relationship zarr.

    Parameters
    ----------
    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    group : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.
        Default: ``'relationship/Hinv'``. Passed to ``SparseZarrCSR`` as ``group``.

    row_chunk : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``8192``.
        Passed to ``SparseZarrCSR`` as ``row_chunk``.

    Returns
    -------
    object
        Computed result from ``SparseZarrCSR(destination, group=group, row_chunk=row_chunk)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return SparseZarrCSR(destination,group=group,row_chunk=row_chunk)


__all__=["write_sparse_relationship_zarr","open_sparse_relationship_zarr","SparseZarrCSR"]
