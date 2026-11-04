"""Position-aware radix DAGs for local genotype/haplotype fine mapping.

The sequence DAG and the statistical LD graph are separate. Every observed path
is retained after suffix merging; reachability alone does not imply an observed
haplotype. Unphased dosages produce a *genotype-state* DAG, not invented phase.
Coordinates are 1-based closed. No phenotype is used to construct this graph.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

__all__ = ["build_haplotype_radix_dag", "haplotype_graph_tables",
           "haplotype_graph_variants", "haplotype_graph_features",
           "write_haplotype_graph_zarr"]


def _chrom(value):
    """Canonicalize a chromosome identifier for annotation/window matching.

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation ``str(value)``. See ``vigor.finemap.graph._chrom`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.
    """
    s = str(value)
    return s[3:] if s.lower().startswith("chr") else s


class _Genotypes:
    """Metadata-only adapter; materialize selected variant tiles on demand."""
    def __init__(self, data, phased, sample_dim, variant_dim, haplotype_dim):
        """Initialize _Genotypes with the inputs and resource settings below.

        Parameters
        ----------
        data : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        phased : bool or None
            Whether the corresponding input represents phased 0/1 haplotypes. None infers phase from
            dimensions/metadata; ordinary 0/1/2 dosage is retained as unphased, never assigned
            fabricated phase.

        sample_dim : str or None
            Name of the corresponding xarray dimension. None infers a recognized dimension from the
            input; explicit names must exist in that DataArray.

        variant_dim : str or None
            Name of the corresponding xarray dimension. None infers a recognized dimension from the
            input; explicit names must exist in that DataArray.

        haplotype_dim : str or None
            Name of the corresponding xarray dimension. None infers a recognized dimension from the
            input; explicit names must exist in that DataArray.
        """
        self.data = data
        self.xr = not isinstance(data, pd.DataFrame)
        if not self.xr:
            if not data.columns.is_unique or not data.index.is_unique:
                raise ValueError("genotype row and column labels must be unique")
            self.ids = pd.Index(data.columns.map(str), name="variant")
            self.ns = len(data)
            self.nh = 1
            self.phased = bool(phased)
            self.samples = list(map(str, data.index))
            self.haps = ["0"] * self.ns
            if isinstance(data.index, pd.MultiIndex):
                if data.index.nlevels != 2 or phased is not True:
                    raise ValueError("MultiIndex rows require phased=True and (sample, haplotype)")
                self.samples = list(map(str, data.index.get_level_values(0)))
                self.haps = list(map(str, data.index.get_level_values(1)))
        else:
            import xarray as xr
            if not isinstance(data, xr.DataArray):
                raise TypeError("genotypes must be pandas.DataFrame or xarray.DataArray")
            vd = variant_dim or next((x for x in ("variant", "snp") if x in data.dims), None)
            sd = sample_dim or next((x for x in ("iid", "sample", "animal", "founder") if x in data.dims), None)
            if vd is None or sd is None:
                raise ValueError("set sample_dim and variant_dim for nonstandard xarray dimensions")
            other = [d for d in data.dims if d not in (sd, vd)]
            hd = haplotype_dim or (other[0] if len(other) == 1 else None)
            if len(other) > 1 or (other and hd not in other):
                raise ValueError("expected samples x variants, optionally with one haplotype dimension")
            if other and phased is False:
                raise ValueError("a haplotype dimension cannot be declared unphased")
            self.phased = bool(other) if phased is None else bool(phased)
            order = (sd, hd, vd) if other else (sd, vd)
            self.data = data.transpose(*order)
            self.vd = vd
            self.ns = data.sizes[sd]
            self.nh = data.sizes[hd] if other else 1
            self.ids = pd.Index(data.coords[vd].values.astype(str), name="variant")
            sample_ids = data.coords[sd].values.astype(str)
            if len(set(sample_ids)) != self.ns:
                raise ValueError("sample labels must be unique")
            self.samples = np.repeat(sample_ids, self.nh).tolist()
            hs = data.coords[hd].values.astype(str) if other else ["0"]
            self.haps = np.tile(hs, self.ns).tolist()
        if not self.ids.is_unique:
            raise ValueError("variant identifiers must be unique after string conversion")
        if len(set(zip(self.samples, self.haps))) != len(self.samples):
            raise ValueError("sample/haplotype identifiers must be unique")

    @property
    def npaths(self):
        """Return the npaths result computed by ``len``.

        Returns
        -------
        object
            Computed result from ``len(self.samples)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return len(self.samples)

    def prepare_window(self, indices):
        """Cull unrelated lazy tasks once instead of for every selected tile.

        Parameters
        ----------
        indices : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.
        """
        if self.xr and hasattr(self.data.data, "__dask_graph__"):
            import dask
            selected = self.data.isel({self.vd: np.asarray(indices, dtype=int)})
            self._prepared = selected.copy(data=dask.optimize(selected.data)[0])
            self._window_indices = {int(original): i for i, original in enumerate(indices)}

    def read(self, indices):
        """Read the requested bounded data block from the enclosing input source.

        Parameters
        ----------
        indices : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        Returns
        -------
        object
            Computed result from ``values``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if self.xr:
            if hasattr(self, "_prepared"):
                local = np.asarray([self._window_indices[int(i)] for i in indices], dtype=int)
                block = self._prepared.isel({self.vd: local})
            else:
                block = self.data.isel({self.vd: np.asarray(indices, dtype=int)})
            values = block.data
            if hasattr(values, "compute"):
                values = values.compute()
            values = np.asarray(values, dtype=np.float32).reshape(self.npaths, -1)
        else:
            values = self.data.iloc[:, indices].to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        limit = 1 if self.phased else 2
        if np.any(finite & ((values < 0) | (values > limit))):
            raise ValueError(f"genotype values must be in [0,{limit}] or missing")
        return values

    def read_ld(self, indices):
        """LD among animals' dosages, including for phased diploid inputs.

        Parameters
        ----------
        indices : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        Returns
        -------
        object
            Computed result from ``np.asarray([np.sum(values[ix], axis=0) for ix in groups.values()], np.float32)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        values = self.read(indices)
        if not self.phased or len(set(self.samples)) == len(self.samples):
            return values
        groups = defaultdict(list)
        for i,sample in enumerate(self.samples):
            groups[sample].append(i)
        return np.asarray([np.sum(values[ix],axis=0) for ix in groups.values()],np.float32)

    def metadata(self, variants):
        """Extract ordered variant/sample coordinates from the enclosing genotype source.

        Parameters
        ----------
        variants : pandas.DataFrame or metadata mapping
            Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
            fields must describe the same fileset; extra annotation columns are retained according
            to the reader.

        Returns
        -------
        object
            Computed result from ``v``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if variants is not None:
            v = variants.copy()
            v.index = v.index.map(str)
            if not v.index.is_unique or not self.ids.isin(v.index).all():
                raise ValueError("variants metadata must uniquely cover genotype columns")
            v = v.reindex(self.ids)
        elif not self.xr and isinstance(self.data.attrs.get("variants"), pd.DataFrame):
            return self.metadata(self.data.attrs["variants"])
        else:
            v = pd.DataFrame(index=self.ids)
            aliases = {"chrom": ("chrom", "chromosome", "chr"),
                       "pos": ("pos", "position", "bp"),
                       "ref": ("ref", "REF"), "alt": ("alt", "ALT")}
            if self.xr:
                for out, keys in aliases.items():
                    key = next((k for k in keys if k in self.data.coords
                                and self.data.coords[k].dims == (self.vd,)), None)
                    if key is not None:
                        v[out] = self.data.coords[key].values
            parsed = self.ids.str.split(":")
            if "pos" not in v or "chrom" not in v:
                if not all(len(x) in (2, 4) and x[1].isdigit() for x in parsed):
                    raise ValueError("supply variants metadata or chrom/pos coordinates; rs IDs do not encode position")
                v["chrom"] = [x[0] for x in parsed]
                v["pos"] = [int(x[1]) for x in parsed]
            if "ref" not in v and all(len(x) == 4 for x in parsed):
                v["ref"] = [x[2] for x in parsed]
                v["alt"] = [x[3] for x in parsed]
        if "pos" not in v or "chrom" not in v:
            raise ValueError("variants metadata needs chrom and pos columns")
        v["chrom"] = v.chrom.map(_chrom)
        positions = pd.to_numeric(v.pos, errors="raise").to_numpy(float)
        if not np.isfinite(positions).all() or np.any(positions < 1) or np.any(positions != np.floor(positions)):
            raise ValueError("variant positions must be positive 1-based integers")
        v["pos"] = positions.astype(np.int64)
        for col in ("ref", "alt"):
            if col not in v:
                v[col] = ""
            v[col] = v[col].fillna("").astype(str).str.upper()
        return v


def _reference(path, chrom, start, end, contig_aliases=None):
    """Fetch just the requested interval, using .fai when present.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.
        Source comparisons: ``offset >= end``. See the rejection/fallback branches for
        enforcement.

    contig_aliases : mapping or None
        Mapping from input chromosome/contig identifiers to the names in the
        annotation/reference source. None uses the routine's built-in matching convention.
        Default: ``None``.
    """
    if path is None:
        return None
    chrom = _chrom((contig_aliases or {}).get(chrom, chrom))
    path = Path(path)
    fai = Path(str(path) + ".fai")
    if fai.exists() and path.suffix != ".gz":
        with fai.open() as handle:
            entries = [line.rstrip().split("\t") for line in handle]
        matches = [r for r in entries if _chrom(r[0]) == chrom]
        if len(matches) != 1:
            raise ValueError("reference chromosome alias is absent or ambiguous")
        r = matches[0]
        length, offset, bases, width = map(int, r[1:5])
        if end > length:
            raise ValueError("window exceeds reference chromosome length")
        chunks = []
        with path.open("rb") as handle:
            p = start - 1
            while p < end:
                count = min(end - p, bases - p % bases)
                handle.seek(offset + (p // bases) * width + p % bases)
                chunks.append(handle.read(count).decode("ascii"))
                p += count
        return "".join(chunks).upper()
    opener = gzip.open if path.suffix == ".gz" else open
    chunks, active, offset = [], False, 0
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if active:
                    break
                active = _chrom(line[1:].split()[0]) == chrom
                offset = 0
                continue
            if not active:
                continue
            line = line.strip()
            lo, hi = max(0, start - 1 - offset), min(len(line), end - offset)
            if lo < hi:
                chunks.append(line[lo:hi])
            offset += len(line)
            if offset >= end:
                break
    seq = "".join(chunks).upper()
    if len(seq) != end - start + 1:
        raise ValueError("reference chromosome missing or window exceeds its length")
    return seq


def _genes(path, chrom, start, end, contig_aliases=None):
    """Stream a GTF/GFF gene track; do not download or infer missing symbols.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    contig_aliases : mapping or None
        Mapping from input chromosome/contig identifiers to the names in the
        annotation/reference source. None uses the routine's built-in matching convention.
        Default: ``None``.
    """
    if path is None:
        return []
    chrom = _chrom((contig_aliases or {}).get(chrom, chrom))
    opener = gzip.open if str(path).endswith(".gz") else open
    records, fallback = [], defaultdict(list)
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9:
                raise ValueError("GTF/GFF records must have nine tab-separated columns")
            if _chrom(fields[0]) != chrom:
                continue
            a, b = int(fields[3]), int(fields[4])
            if a > b:
                raise ValueError("annotation interval has start > end")
            if a > end or b < start:
                continue
            attrs = dict(re.findall(r'(\w+)\s+"([^"]*)"', fields[8]))
            if not attrs:
                attrs = dict(part.split("=", 1) for part in fields[8].split(";") if "=" in part)
            symbol = attrs.get("gene_name") or attrs.get("gene_symbol") or attrs.get("gene")
            if fields[2] == "gene":
                symbol = symbol or attrs.get("Name")
                if symbol:
                    records.append((a, b, symbol))
            elif symbol:
                fallback[(attrs.get("gene_id", symbol), symbol)].append((a, b))
    if not records:
        records = [(min(a for a, b in spans), max(b for a, b in spans), symbol)
                   for (_, symbol), spans in fallback.items()]
    return sorted(set(records))


def _gene_symbols(records, start, end):
    """Return the gene symbols result computed by ``tuple``.

    Parameters
    ----------
    records : object (see the cited input contract)
        Input for the operation ``tuple(sorted({s for a, b, s in records if a <= end and b >=
        start}))``. See ``vigor.finemap.graph._gene_symbols`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.
    """
    return tuple(sorted({s for a, b, s in records if a <= end and b >= start}))


def _correlation(A, B):
    """Pairwise-complete, signed correlation (wrapper of ``vigor.gwas.ld.pairwise_complete``).

    Parameters
    ----------
    A : array-like
        Sample × variant tile; NaN marks missing calls.

    B : array-like
        Sample × variant tile on the same sample rows as ``A``.

    Returns
    -------
    tuple
        ``(r, n)``: float32 signed r (NaN when fewer than 3 joint calls or zero
        variance) and int32 joint counts.
    """
    from ..gwas.ld import pairwise_complete
    return pairwise_complete(A, B, statistic="r", min_n=3, nan_policy="pairwise",
                             dtype=np.float32, return_counts=True)


def _ld(reader, indices, v, matrix, kind, threshold, tile, max_distance, max_lag,
        mode, max_bytes, hdbscan, min_cluster, min_samples, dense_max_variants):
    """Resolve supplied or computed LD into dense/banded graph linkage evidence.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    indices : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.

    v : array-like
        Numerical operand in ``v.pos.iloc[b] - v.pos.iloc[aa[-1]]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    matrix : object (see the cited input contract)
        Input for the operation ``matrix.coords[dim]``. See ``vigor.finemap.graph._ld`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.

    kind : str
        Edge/evidence/input interpretation selected by this constructor/helper. See the local
        branches and the source-derived option descriptions below; each builder admits only its
        own subset, not a package-wide enum. Source-recognized literal values: ``r``, ``r2``.
        These select the corresponding branches below; comparison evidence alone does not
        establish an exhaustive enum.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain.

    tile : object (see the cited input contract)
        Input for the operation ``a + tile``. See ``vigor.finemap.graph._ld`` and its caller for
        the scientific representation; the shown operation constrains the required type/shape.

    max_distance : object (see the cited input contract)
        Input for the operation ``v.pos.iloc[b] - v.pos.iloc[aa[-1]] > max_distance``. See
        ``vigor.finemap.graph._ld`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape.

    max_lag : object (see the cited input contract)
        Input for the operation ``b - aa[-1] > max_lag``. See ``vigor.finemap.graph._ld`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    mode : str
        Input for the operation ``mode not in {'auto', 'dense', 'sparse'}``. See
        ``vigor.finemap.graph._ld`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape.

    max_bytes : int or float or None
        Memory budget used by the allocation/chunking guard. Units are bytes for names ending in
        bytes; maxmem uses the resource helper's GiB convention. None delegates to that helper's
        available-memory policy.

    hdbscan : bool
        Whether to cluster supported LD distances with HDBSCAN when available; unknown/missing
        LD distances are never fabricated to run clustering.

    min_cluster : object (see the cited input contract)
        Input supplied to ``HDBSCAN`` as ``min_cluster_size``. See that callable's documented
        ``min_cluster_size`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Passed to ``HDBSCAN`` as ``min_cluster_size``.

    min_samples : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Passed to ``HDBSCAN`` as ``min_samples``.

    dense_max_variants : object (see the cited input contract)
        Input for the operation ``m <= dense_max_variants``. See ``vigor.finemap.graph._ld`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.
    """
    m = len(v)
    supplied = matrix is not None
    if kind not in {"r", "r2"}:
        raise ValueError("ld_kind must be 'r' or 'r2'")
    if mode not in {"auto", "dense", "sparse"}:
        raise ValueError("ld_mode must be auto, dense or sparse")
    dense_bytes = m*m*4
    use_dense = mode == "dense" or (mode == "auto" and m <= dense_max_variants and dense_bytes <= max_bytes // 4)
    if use_dense and dense_bytes > max_bytes // 4:
        raise MemoryError("dense LD exceeds its memory allowance; use ld_mode='sparse'")
    rs, pairs, counts = [], [], []
    dense = None
    if supplied:
        if isinstance(matrix, pd.DataFrame):
            mat = matrix.copy(deep=False)
            mat.index = mat.index.map(str); mat.columns = mat.columns.map(str)
            if not mat.index.is_unique or not mat.columns.is_unique:
                raise ValueError("LD labels must be unique")
            if not v.index.isin(mat.index).all() or not v.index.isin(mat.columns).all():
                raise ValueError("LD labels do not cover selected variants")
            if dense_bytes > max_bytes // 4:
                raise MemoryError("labelled dense LD exceeds allowance; supply sparse LD")
            R = mat.loc[v.index, v.index].to_numpy(dtype=np.float32)
        elif hasattr(matrix, "dims"):
            if matrix.ndim != 2:
                raise ValueError("xarray LD must have two dimensions")
            if dense_bytes > max_bytes // 4:
                raise MemoryError("dense LD exceeds allowance; supply a sparse matrix")
            selectors = {}
            for dim in matrix.dims:
                if dim in matrix.coords:
                    labels = pd.Index(matrix.coords[dim].values.astype(str))
                    if not labels.is_unique:
                        raise ValueError("LD labels must be unique")
                    take = labels.get_indexer(v.index)
                    if np.any(take < 0):
                        raise ValueError("LD labels do not cover selected variants")
                    selectors[dim] = take
                else:
                    if matrix.sizes[dim] != len(reader.ids):
                        raise ValueError("unlabelled LD shape must match original genotype variants")
                    selectors[dim] = indices
            R = np.asarray(matrix.isel(selectors).values, np.float32)
        elif sparse.issparse(matrix):
            if matrix.shape != (len(reader.ids), len(reader.ids)):
                raise ValueError("unlabelled LD shape must match original genotype variants")
            R = matrix.tocsr()[indices][:, indices].astype(np.float32)
        else:
            if getattr(matrix, "shape", None) != (len(reader.ids), len(reader.ids)):
                raise ValueError("unlabelled LD shape must match original genotype variants")
            if dense_bytes > max_bytes // 4:
                raise MemoryError("dense LD exceeds allowance; supply a sparse matrix")
            R = np.asarray(matrix, np.float32)[np.ix_(indices, indices)]
        values = R.data if sparse.issparse(R) else R
        if np.isinf(values).any() or (sparse.issparse(R) and np.isnan(values).any()) or np.any(values > 1.00001) or np.any(values < (-1.00001 if kind == "r" else 0)):
            raise ValueError("LD entries must be within the declared r/r2 range; infinity and sparse undefined entries are unsupported")
        if not sparse.issparse(R) and not np.array_equal(np.isnan(R),np.isnan(R.T)):
            raise ValueError('LD matrix must have symmetric undefined-pair masks')
        asym = R - R.T
        if (abs(asym).max() if sparse.issparse(asym) else np.max(np.nan_to_num(np.abs(asym)))) > 1e-5:
            raise ValueError("LD matrix must be symmetric")
        R2 = R.multiply(R) if kind == "r" and sparse.issparse(R) else (R*R if kind == "r" else R)
        if sparse.issparse(R2):
            E = sparse.triu(R2, k=1).tocoo()
            keep = E.data >= threshold
            ei, ej, r2 = E.row[keep], E.col[keep], E.data[keep]
            r = np.asarray(R.tocsr()[ei, ej]).ravel() if kind == "r" else np.full(len(ei), np.nan)
        else:
            ei, ej = np.where(np.triu(R2 >= threshold, k=1))
            r2 = R2[ei, ej]
            r = R[ei, ej] if kind == "r" else np.full(len(ei), np.nan)
        if use_dense and not sparse.issparse(R2):
            dense = R2.toarray() if sparse.issparse(R2) else R2
        edges = pd.DataFrame({"i": ei, "j": ej, "r": r, "r2": r2, "nobs": -1})
        scope = "supplied_sparse" if sparse.issparse(R) else "supplied_dense"
    else:
        full = use_dense
        if full:
            dense = np.eye(m, dtype=np.float32)
        for a in range(0, m, tile):
            aa = np.arange(a, min(a+tile, m))
            A = reader.read_ld(indices[aa])
            for b in range(a, m, tile):
                bb = np.arange(b, min(b+tile, m))
                if not full:
                    if max_lag is not None and b-aa[-1] > max_lag:
                        break
                    if max_distance is not None and v.pos.iloc[b]-v.pos.iloc[aa[-1]] > max_distance:
                        break
                B = A if b == a else reader.read_ld(indices[bb])
                r, n = _correlation(A, B)
                r2 = r*r
                if full:
                    dense[np.ix_(aa, bb)] = r2
                    dense[np.ix_(bb, aa)] = r2.T
                ii, jj = np.meshgrid(aa, bb, indexing="ij")
                keep = (jj > ii) & (r2 >= threshold)
                if not full:
                    if max_lag is not None:
                        keep &= jj-ii <= max_lag
                    if max_distance is not None:
                        keep &= np.abs(v.pos.to_numpy()[jj]-v.pos.to_numpy()[ii]) <= max_distance
                u, w = np.where(keep)
                if sum(len(x) for x in pairs)+len(u) > max_bytes // 256:
                    raise MemoryError("LD edge table exceeds allowance; increase ld_threshold or shorten window")
                pairs.append(np.column_stack([aa[u], bb[w]])); rs.append(np.column_stack([r[u,w], r2[u,w]])); counts.append(n[u,w])
        ij = np.concatenate(pairs) if pairs else np.empty((0,2), int)
        rr = np.concatenate(rs) if rs else np.empty((0,2), float)
        nn = np.concatenate(counts) if counts else np.empty(0, int)
        edges = pd.DataFrame({"i": ij[:,0].astype(int), "j": ij[:,1].astype(int),
                              "r": rr[:,0], "r2": rr[:,1], "nobs": nn})
        scope = "computed_full" if full else "computed_banded"
    ei, ej = edges.i.to_numpy(int), edges.j.to_numpy(int)
    adjacency = sparse.coo_matrix((np.ones(2*len(edges), np.int8),
                                   (np.r_[ei,ej], np.r_[ej,ei])), shape=(m,m)).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    clusters = pd.DataFrame({"ld_component": labels, "hdbscan_cluster": -1,
                             "hdbscan_confidence": 0., "hdbscan_status": "disabled"}, index=v.index)
    group_number = 0
    for component in np.unique(labels):
        ix = np.flatnonzero(labels == component)
        if not hdbscan:
            continue
        if len(ix) < max(3, min_cluster):
            clusters.loc[v.index[ix], "hdbscan_status"] = "too_small"
            continue
        # Exact distance requires all pairs; a thresholded sparse adjacency is not a distance matrix.
        local_dense = dense
        if dense is None:
            if scope == 'computed_banded' and len(ix) <= dense_max_variants:
                # Fill actual within-component distances; thresholded edges
                # alone do not define the HDBSCAN distance matrix.
                local_dense = np.eye(len(ix),dtype=np.float32)
                for a in range(0,len(ix),tile):
                    aa = np.arange(a,min(a+tile,len(ix)))
                    A = reader.read_ld(indices[ix[aa]])
                    for b in range(a,len(ix),tile):
                        bb = np.arange(b,min(b+tile,len(ix)))
                        B = A if b==a else reader.read_ld(indices[ix[bb]])
                        r,_ = _correlation(A,B)
                        local_dense[np.ix_(aa,bb)] = r*r
                        local_dense[np.ix_(bb,aa)] = (r*r).T
            else:
                clusters.loc[v.index[ix], "hdbscan_status"] = "requires_full_ld"
                continue
        if len(ix)**2*24 > max_bytes // 4:
            clusters.loc[v.index[ix], "hdbscan_status"] = "memory_limit"
            continue
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            try:
                from hdbscan import HDBSCAN
            except ImportError:
                clusters.loc[v.index[ix], "hdbscan_status"] = "dependency_unavailable"
                continue
        Rloc = local_dense if dense is None else dense[np.ix_(ix,ix)]
        if not np.isfinite(Rloc).all():
            clusters.loc[v.index[ix], "hdbscan_status"] = "undefined_pairwise_ld"
            continue
        D = np.asarray(1-Rloc, dtype=np.float64)
        np.fill_diagonal(D, 0)
        fit = HDBSCAN(metric="precomputed", min_cluster_size=min_cluster,
                      min_samples=min(min_samples, len(ix)-1), allow_single_cluster=True).fit(D)
        lab = fit.labels_
        mapped = np.where(lab >= 0, lab+group_number, -1)
        group_number += int(lab.max()+1)
        clusters.loc[v.index[ix], "hdbscan_cluster"] = mapped
        clusters.loc[v.index[ix], "hdbscan_confidence"] = fit.probabilities_
        clusters.loc[v.index[ix], "hdbscan_status"] = "fit"
    clusters["ld_group"] = [f"H{h}" if h >= 0 else f"C{c}" for c,h in
                             zip(clusters.ld_component, clusters.hdbscan_cluster)]
    return clusters, edges, scope, dense


def _radix(codes, max_nodes, max_visits, max_variant_states):
    """Compressed prefix trie followed by exact suffix-subgraph interning.

    Parameters
    ----------
    codes : object (see the cited input contract)
        Input for the operation ``codes.shape[1] - stop``. See ``vigor.finemap.graph._radix``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    max_nodes : int
        Maximum allowed graph-node count, at least two; exceeding the guard stops graph
        materialization.

    max_visits : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate.

    max_variant_states : object (see the cited input contract)
        Input for the operation ``segment_variants > max_variant_states``. See
        ``vigor.finemap.graph._radix`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.
    """
    representatives, members, lookup = [], [], {}
    for row in range(len(codes)):
        digest = hashlib.blake2b(codes[row].tobytes(), digest_size=20).digest()
        found = lookup.get(digest)
        if found is not None and not np.array_equal(codes[row], codes[representatives[found]]):
            raise RuntimeError("path hash collision")
        if found is None:
            found = len(representatives); lookup[digest] = found
            representatives.append(row); members.append([])
        members[found].append(row)
    trie = {0: {"span": (0,0), "states": (), "parent": None, "layer": "1", "children": []}}
    stack = [(np.arange(len(representatives)), 0, 0)]
    leaves, visits, segment_variants = {}, 0, 0
    while stack:
        rows, depth, parent = stack.pop()
        rep = np.asarray(representatives)[rows]
        stop = depth
        while stop < codes.shape[1]:
            end = min(stop+64, codes.shape[1])
            block = codes[np.ix_(rep, np.arange(stop,end))]
            same = np.all(block == block[:1], axis=0)
            bad = np.flatnonzero(~same)
            stop = stop+int(bad[0]) if len(bad) else end
            if len(bad):
                break
        current = parent
        if stop > depth:
            segment_variants += stop-depth
            if segment_variants > max_variant_states:
                raise MemoryError("radix segment annotations exceed memory allowance; shorten window or increase budget")
            current = len(trie)
            if current >= max_nodes:
                raise MemoryError("radix graph exceeds max_nodes; shorten the window or increase the budget")
            child_no = len(trie[parent]["children"])+1
            layer = trie[parent]["layer"] if len(rows) == len(representatives) else f'{trie[parent]["layer"]}.{child_no}'
            trie[current] = {"span": (depth,stop), "states": tuple(codes[rep[0],depth:stop].tolist()),
                             "parent": parent, "layer": layer, "children": []}
            trie[parent]["children"].append(current)
        if stop == codes.shape[1]:
            for row in rows:
                leaves[int(row)] = current
        else:
            groups = [rows[codes[rep,stop] == state] for state in np.unique(codes[rep,stop])]
            # Reverse push gives deterministic ref / het / alt / missing ordering.
            stack.extend((group, stop, current) for group in groups[::-1])
    # Prefix compression can hide a shared suffix inside differently sized
    # segments. Discover suffix branch boundaries before interning, so suffixes
    # can merge even when their original prefix-trie segment boundaries differ.
    cuts = {0, codes.shape[1]}
    reverse = codes[:, ::-1]
    stack = [(np.arange(len(representatives)), 0)]
    while stack:
        rows, depth = stack.pop()
        rep = np.asarray(representatives)[rows]
        stop = depth
        while stop < codes.shape[1]:
            end = min(stop+64, codes.shape[1])
            block = reverse[np.ix_(rep, np.arange(stop,end))]
            bad = np.flatnonzero(~np.all(block == block[:1],axis=0))
            stop = stop+int(bad[0]) if len(bad) else end
            if len(bad):
                break
        if stop < codes.shape[1]:
            cuts.add(codes.shape[1]-stop)
            stack.extend((rows[reverse[rep,stop] == state],stop)
                         for state in np.unique(reverse[rep,stop]))
    old_trie, old_leaves = trie, leaves
    trie, ends = {}, {}
    cuts = np.asarray(sorted(cuts))
    for old,rec in old_trie.items():
        a,b = rec['span']
        boundaries = [a,*cuts[(cuts>a)&(cuts<b)].tolist(),b] if b>a else [a,b]
        parent = ends[rec['parent']] if rec['parent'] is not None else None
        for lo,hi in zip(boundaries,boundaries[1:]):
            node = len(trie)
            if node >= max_nodes:
                raise MemoryError('suffix-refined radix DAG exceeds max_nodes')
            trie[node] = {'span':(lo,hi),'states':rec['states'][lo-a:hi-a],
                          'parent':parent,'layer':rec['layer'],'children':[]}
            if parent is not None:
                trie[parent]['children'].append(node)
            parent = node
        ends[old] = parent
    leaves = {rep:ends[leaf] for rep,leaf in old_leaves.items()}
    canonical, signatures, interned = {}, {}, {}
    # Insertion order is parent before child, so reverse order is a topological pass.
    for old in reversed(list(trie)):
        rec = trie[old]
        signature = (rec["span"], rec["states"], tuple(sorted(canonical[c] for c in rec["children"])))
        new = signatures.get(signature)
        if new is None:
            new = len(interned)
            signatures[signature] = new
            interned[new] = {"span": rec["span"], "states": rec["states"], "layers": []}
        canonical[old] = new
        interned[new]["layers"].append(rec["layer"])
    paths = [None]*len(codes)
    for rep, leaf in leaves.items():
        chain = []
        while leaf is not None:
            chain.append(canonical[leaf]); leaf = trie[leaf]["parent"]
        chain = tuple(chain[::-1])
        visits += len(chain)*len(members[rep])
        if visits > max_visits:
            raise MemoryError("path memberships exceed max_path_visits")
        for row in members[rep]:
            paths[row] = chain
    edges = {(canonical[p], canonical[c]) for p, rec in trie.items() for c in rec["children"]}
    return interned, edges, paths, canonical[0]


_IUPAC = {frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("CG"): "S",
          frozenset("AT"): "W", frozenset("GT"): "K", frozenset("AC"): "M"}


def _node_sequence(rec, v, reference, start, end):
    """Extract the reference/allelic sequence represented by a graph node interval.

    Parameters
    ----------
    rec : object (see the cited input contract)
        Input for the operation ``rec['span']``. See ``vigor.finemap.graph._node_sequence`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    v : array-like
        Numerical operand in ``int(v.pos.iloc[a]) - 1``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    reference : array-like, table or fitted object
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.
    """
    a,b = rec["span"]
    if a == b:
        if a > 0:
            return int(v.pos.iloc[a]), int(v.pos.iloc[a])-1, "", "anchor"
        hi = int(v.pos.iloc[0])-1
        return start, hi, (reference[:hi-start+1] if reference is not None else None), "reference" if reference is not None else "unavailable"
    lo = int(v.pos.iloc[a])
    hi = int(v.pos.iloc[b])-1 if b < len(v) else end
    if reference is None:
        return lo, hi, None, "unavailable"
    chunks, status = [], "exact"
    for j,state in zip(range(a,b), rec["states"]):
        row = v.iloc[j]
        stop = int(v.pos.iloc[j+1])-1 if j+1 < len(v) else end
        ref, alt = row.ref, row.alt
        if state == 0:
            allele = ref
        elif state == 2 and alt and re.fullmatch("[ACGTN]+", alt):
            allele = alt
        elif state == 1 and len(ref) == len(alt) == 1:
            allele = _IUPAC.get(frozenset((ref,alt)), "N"); status = "unphased_ambiguity"
        else:
            return lo, hi, None, "unresolved_allele"
        chunks.append(allele)
        gap_lo = int(row.pos)+len(ref)
        if gap_lo <= stop:
            chunks.append(reference[gap_lo-start:stop-start+1])
    return lo, hi, "".join(chunks), status


def _anchored_radix(codes, max_nodes, max_visits, max_states, anchor_variants):
    """Build bounded radix-path tiles and join them across selected variant anchors.

    Parameters
    ----------
    codes : object (see the cited input contract)
        Input for the operation ``codes.shape[1]``. See ``vigor.finemap.graph._anchored_radix``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    max_nodes : int
        Maximum allowed graph-node count, at least two; exceeding the guard stops graph
        materialization.

    max_visits : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate.

    max_states : object (see the cited input contract)
        Input for the operation ``max_states - state_count``. See
        ``vigor.finemap.graph._anchored_radix`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    anchor_variants : object (see the cited input contract)
        Input for the operation ``a + anchor_variants``. See
        ``vigor.finemap.graph._anchored_radix`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Source
        comparisons: ``anchor_variants < 1``. See the rejection/fallback branches for
        enforcement.
    """
    if anchor_variants is None or anchor_variants >= codes.shape[1]:
        return _radix(codes,max_nodes,max_visits,max_states)
    if anchor_variants < 1:
        raise ValueError('radix_anchor_variants must be positive or None')
    records,links,paths = {},set(),[[] for _ in range(len(codes))]
    state_count,visit_count = 0,0
    for a in range(0,codes.shape[1],anchor_variants):
        b = min(a+anchor_variants,codes.shape[1])
        rec,edges,local_paths,root = _radix(codes[:,a:b],max_nodes-len(records),max_visits,max_states-state_count)
        offset = len(records)
        for node,data in rec.items():
            data['span'] = (data['span'][0]+a,data['span'][1]+a)
            data['layers'] = [f'1.{a//anchor_variants+1}'+layer[1:] for layer in data['layers']]
            state_count += len(data['states'])
            records[node+offset] = data
        links.update((s+offset,t+offset) for s,t in edges)
        for i,chain in enumerate(local_paths):
            translated = tuple(n+offset for n in chain)
            if paths[i]:
                links.add((paths[i][-1],translated[0]))
            paths[i].extend(translated)
            visit_count += len(translated)
        if len(records)>max_nodes or visit_count>max_visits or state_count>max_states:
            raise MemoryError('anchored radix graph exceeds graph/path/annotation allowance')
        if a == 0:
            global_root = root+offset
    return records,links,[tuple(x) for x in paths],global_root


def build_haplotype_radix_dag(
    genotypes, founder_genotypes=None, ldmatrix=None, gtf_path=None,
    reference_sequence_path=None, *, variants=None, phased=None, founder_phased=None,
    sample_dim=None, variant_dim=None, haplotype_dim=None, lead_variant=None,
    window_bp=6_000_000, window_bounds=None, ld_kind="r2", ld_mode="auto",
    ld_threshold=.20, ld_max_distance_bp=1_000_000, ld_max_variant_lag=256,
    ld_dense_max_variants=1024, radix_anchor_variants=64,
    hdbscan=True, hdbscan_min_cluster_size=20, hdbscan_min_samples=5,
    chunk_variants=256, max_memory_bytes=1_073_741_824, max_nodes=200_000,
    max_path_visits=10_000_000, dosage_tolerance=0.10, assembly=None, contig_aliases=None,
    return_type="graph", variant_format="pandas", return_variant_table=False,
):
    """Build an annotated radix DAG for one physical fine-mapping window.

    Inputs are samples x variants DataFrames/DataArrays. A phased DataArray is
    samples x haplotypes x variants (0/1 alleles); phased DataFrame rows may be
    a (sample, haplotype) MultiIndex. Ordinary 0/1/2 dosages remain unphased.
    Missing or uncertain calls are retained as unresolved states, never phased.
    ``variants`` is a variant-indexed frame with chrom, pos, optionally ref/alt;
    alternatively xarray coordinates or chr:pos[:ref:alt] IDs supply metadata.

    With a lead, window_bp is TOTAL width, clipped at chromosome start. Without
    a lead, the supplied window is used; window_bounds overrides either choice.
    Only selected tiles of a lazy genotype input are computed. A bounded uint8
    path matrix and the NetworkX graph are necessarily eager. Dense LD is exact
    for small windows; auto mode uses explicitly labelled banded sparse LD for
    large windows. HDBSCAN is never run on fabricated missing LD distances.

    Default returns nx.DiGraph. return_type='tables' returns nodes/edges/paths;
    'variants' returns the combined variant table; 'bundle' returns all of them
    with the graph. Formats are pandas, xarray.Dataset or Dask DataFrame.
    return_variant_table=True returns (graph, variant_table) in graph mode.

    Parameters
    ----------
    genotypes : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    founder_genotypes : array-like or None
        Founder-reference allele/dosage matrix with the same variant axis as the target
        genotypes. None omits founder path annotation; phased founders use haplotype-aware
        dimensions. Default: ``None``.

    ldmatrix : array-like or None
        Variant × variant linkage-disequilibrium matrix in the genotype-column order. Use
        correlation or squared correlation according to ld_kind/the called fitter; None requests
        computation where supported. Default: ``None``.

    gtf_path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value. Passed to ``G.graph.update`` as
        ``gtf_path``.

    reference_sequence_path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value. Passed to ``G.graph.update`` as
        ``reference_sequence_path``.

    variants : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    phased : bool or None
        Whether the corresponding input represents phased 0/1 haplotypes. None infers phase from
        dimensions/metadata; ordinary 0/1/2 dosage is retained as unphased, never assigned
        fabricated phase. Default: ``None``.

    founder_phased : bool or None
        Whether the corresponding input represents phased 0/1 haplotypes. None infers phase from
        dimensions/metadata; ordinary 0/1/2 dosage is retained as unphased, never assigned
        fabricated phase. Default: ``None``.

    sample_dim : str or None
        Name of the corresponding xarray dimension. None infers a recognized dimension from the
        input; explicit names must exist in that DataArray. Default: ``None``.

    variant_dim : str or None
        Name of the corresponding xarray dimension. None infers a recognized dimension from the
        input; explicit names must exist in that DataArray. Default: ``None``.

    haplotype_dim : str or None
        Name of the corresponding xarray dimension. None infers a recognized dimension from the
        input; explicit names must exist in that DataArray. Default: ``None``.

    lead_variant : str or int or None
        Lead variant identifier/index defining the physical window center; None uses the
        supplied region rather than requiring a lead. Default: ``None``.

    window_bp : int
        Total physical window width in base pairs, at least one; the centered interval is
        clipped at chromosome start and window_bounds overrides it. Default: ``6000000``. Source
        comparisons: ``window_bp < 1``. See the rejection/fallback branches for enforcement.

    window_bounds : pair of int or None
        Lower and upper genomic positions defining the selected interval. Positions refer to the
        chromosome's physical coordinate system; None uses the full eligible region. Default:
        ``None``.

    ld_kind : str
        'r2' interprets LD as squared correlation and 'r' as signed correlation. The LD helper
        validates exactly these two strings and converts the selected statistic appropriately
        for graph linkage thresholds. Default: ``'r2'``.

    ld_mode : str
        'auto' uses dense exact LD when the selected variant count is at most
        ld_dense_max_variants, otherwise bounded banded sparse LD. 'dense' forces dense
        allocation subject to the memory guard; 'sparse' forces banded LD. No other strings are
        accepted. Default: ``'auto'``.

    ld_threshold : float
        Minimum LD strength linking graph anchors, strictly greater than zero and no greater
        than one. Default: ``0.2``. Source comparisons: ``0 < ld_threshold <= 1``. See the
        rejection/fallback branches for enforcement.

    ld_max_distance_bp : int or None
        Maximum physical variant separation in base pairs for banded LD; positive bounds
        restrict distant pairs and None leaves this distance unrestricted where supported.
        Default: ``1000000``.

    ld_max_variant_lag : int
        Maximum variant-index separation examined in banded LD, bounding local pair count and
        memory. Default: ``256``.

    ld_dense_max_variants : int
        Maximum window variant count for automatic dense LD; larger windows use explicitly
        labeled banded sparse LD. Default: ``1024``.

    radix_anchor_variants : int
        Maximum variants used as radix-path anchors in the selected window; positive counts
        control graph detail. Default: ``64``.

    hdbscan : bool
        Whether to cluster supported LD distances with HDBSCAN when available; unknown/missing
        LD distances are never fabricated to run clustering. Default: ``True``.

    hdbscan_min_cluster_size : int
        Minimum HDBSCAN cluster size, at least two. Default: ``20``. Source comparisons:
        ``hdbscan_min_cluster_size < 2``. See the rejection/fallback branches for enforcement.

    hdbscan_min_samples : int
        HDBSCAN neighborhood support count, at least one. Default: ``5``. Source comparisons:
        ``hdbscan_min_samples < 1``. See the rejection/fallback branches for enforcement.

    chunk_variants : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``256``. Source
        comparisons: ``chunk_variants < 1``. See the rejection/fallback branches for
        enforcement.

    max_memory_bytes : int or float or None
        Memory budget used by the allocation/chunking guard. Units are bytes for names ending in
        bytes; maxmem uses the resource helper's GiB convention. None delegates to that helper's
        available-memory policy. Default: ``1073741824``. Source comparisons: ``max_memory_bytes
        < 1``. See the rejection/fallback branches for enforcement. Passed to ``G.graph.update``
        as ``max_memory_bytes``.

    max_nodes : int
        Maximum allowed graph-node count, at least two; exceeding the guard stops graph
        materialization. Default: ``200000``. Source comparisons: ``max_nodes < 2``. See the
        rejection/fallback branches for enforcement.

    max_path_visits : int
        Maximum radix path visits, at least one; bounds graph construction work. Default:
        ``10000000``. Source comparisons: ``max_path_visits < 1``. See the rejection/fallback
        branches for enforcement.

    dosage_tolerance : float
        Distance from an integer hardcall allowed before marking dosage unresolved; must satisfy
        0 <= dosage_tolerance < 0.5. Default: ``0.1``. Source comparisons: ``dosage_tolerance <
        0``; ``dosage_tolerance >= 0.5``. See the rejection/fallback branches for enforcement.

    assembly : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Default: ``None``.
        Passed to ``G.graph.update`` as ``assembly``.

    contig_aliases : mapping or None
        Mapping from input chromosome/contig identifiers to the names in the
        annotation/reference source. None uses the routine's built-in matching convention.
        Default: ``None``.

    return_type : str
        Result representation: 'graph' returns the directed radix DAG; 'tables' returns
        nodes/edges/paths; 'variants' returns variant annotations; 'bundle' returns graph and
        tables together. Default: ``'graph'``.

    variant_format : str
        Variant-table container: 'pandas' returns an eager DataFrame; 'xarray' returns a labeled
        Dataset; 'dask' returns a Dask DataFrame. Default: ``'pandas'``.

    return_variant_table : bool
        Whether graph-mode output is paired with the variant annotation table; other return_type
        choices already define their own container. Default: ``False``.

    Returns
    -------
    dict
        Result fields: ``graph``, ``variants``. Field arrays preserve the corresponding input axis order.

    Examples
    --------
        graph = build_haplotype_radix_dag(window_dosage, variants=variant_metadata,
                                          ld_mode='auto', hdbscan=False)
        # variant_metadata is indexed by variant ID with chrom and pos columns.
        # Dosage remains unphased; uncertain/missing calls remain unresolved.
    """
    if return_type not in {"graph", "tables", "variants", "bundle"}:
        raise ValueError("return_type must be graph, tables, variants or bundle")
    if variant_format not in {"pandas", "xarray", "dask"}:
        raise ValueError("variant_format must be pandas, xarray or dask")
    if not 0 < ld_threshold <= 1 or dosage_tolerance < 0 or dosage_tolerance >= .5:
        raise ValueError("invalid LD threshold or dosage tolerance")
    if chunk_variants < 1 or max_memory_bytes < 1 or max_nodes < 2 or max_path_visits < 1:
        raise ValueError("memory, chunk and graph budgets must be positive")
    if hdbscan_min_cluster_size < 2 or hdbscan_min_samples < 1:
        raise ValueError("invalid HDBSCAN settings")
    reader = _Genotypes(genotypes, phased, sample_dim, variant_dim, haplotype_dim)
    meta = reader.metadata(variants)
    if len(meta) == 0 or reader.npaths == 0:
        raise ValueError("genotypes must contain samples and variants")
    if lead_variant is not None:
        lead_variant = str(lead_variant)
        if lead_variant not in meta.index:
            raise ValueError("lead_variant not in genotype columns")
        chromosome = meta.loc[lead_variant, "chrom"]
        center = int(meta.loc[lead_variant, "pos"])
        if window_bp < 1:
            raise ValueError("window_bp must be positive")
        start = max(1, center-int(window_bp)//2)
        end = center+(int(window_bp)-1)//2
    else:
        if meta.chrom.nunique() != 1:
            raise ValueError("one chromosome is required unless lead_variant selects it")
        chromosome = meta.chrom.iloc[0]
        start, end = int(meta.pos.min()), int(meta.pos.max())
    if window_bounds is not None:
        start,end = map(int, window_bounds)
    if start < 1 or end < start:
        raise ValueError("window bounds must be 1-based closed with start <= end")
    selected = meta.chrom.eq(chromosome) & meta.pos.between(start,end)
    indices = np.flatnonzero(selected.to_numpy())
    indices = indices[np.argsort(meta.pos.iloc[indices].to_numpy(), kind="stable")]
    v = meta.iloc[indices].copy()
    if not len(v):
        raise ValueError("selected window contains no variants")
    if v.pos.duplicated().any():
        raise ValueError("overlapping/multiallelic sites require normalization to nonoverlapping biallelic records")
    if window_bounds is None and lead_variant is None:
        end = max(end, int(v.pos.iloc[-1])+max(len(v.ref.iloc[-1]),1)-1)
    aliases = {_chrom(k):str(val) for k,val in (contig_aliases or {}).items()}
    reference = _reference(reference_sequence_path, chromosome, start,end,aliases)
    if reference is not None:
        for j in range(len(v)):
            pos = int(v.pos.iloc[j])
            ref = v.ref.iloc[j] or reference[pos-start]
            if reference[pos-start:pos-start+len(ref)] != ref:
                raise ValueError(f"reference allele mismatch at {v.index[j]}; check assembly and allele orientation")
            if j+1 < len(v) and pos+len(ref) > v.pos.iloc[j+1]:
                raise ValueError("overlapping reference alleles are not supported")
            v.iloc[j,v.columns.get_loc("ref")] = ref
    genes = _genes(gtf_path, chromosome, start,end,aliases)
    founders = None
    if founder_genotypes is not None:
        founders = _Genotypes(founder_genotypes, founder_phased, None, None, None)
        missing = ~v.index.isin(founders.ids)
        if missing.any():
            raise ValueError("founder genotypes must cover all selected variant IDs")
        fi = founders.ids.get_indexer(v.index)
    total = reader.npaths + (founders.npaths if founders else 0)
    code_bytes = total*len(v)
    if code_bytes > max_memory_bytes // 3:
        raise MemoryError("window path matrix exceeds budget; shorten window or raise max_memory_bytes")
    tile = min(int(chunk_variants), max(1, int(max_memory_bytes)//max(256*total,1)))
    reader.prepare_window(indices)
    if founders:
        founders.prepare_window(fi)
    codes = np.empty((total,len(v)), np.uint8)
    unresolved = 0
    for source, src_indices, offset in [(reader, indices, 0)] + ([(founders, fi, reader.npaths)] if founders else []):
        for a in range(0,len(v),tile):
            block = source.read(src_indices[a:a+tile])
            rounded = np.rint(np.nan_to_num(block, nan=0))
            valid = np.isfinite(block) & (np.abs(block-rounded) <= dosage_tolerance)
            values = rounded * (2 if source.phased else 1)
            state = np.where(valid, values, 3).astype(np.uint8)
            unresolved += int((~valid).sum())
            codes[offset:offset+source.npaths,a:a+tile] = state
    clusters, ld_edges, scope, dense = _ld(reader, indices, v, ldmatrix, ld_kind,
        ld_threshold, tile, ld_max_distance_bp, ld_max_variant_lag, ld_mode,
        max_memory_bytes, hdbscan, hdbscan_min_cluster_size, hdbscan_min_samples, ld_dense_max_variants)
    records, links, paths, root = _anchored_radix(codes, max_nodes, max_path_visits,
        max_memory_bytes//64,radix_anchor_variants)
    if reference is not None:
        sequence_bytes = sum((int(v.pos.iloc[b])-1 if b<len(v) else end)-int(v.pos.iloc[a])+1
                             for rec in records.values() for a,b in [rec['span']] if b>a)
        if sequence_bytes > max_memory_bytes//2:
            raise MemoryError('expanded node sequences exceed allowance; increase memory or reduce radix_anchor_variants')
    G = nx.DiGraph()
    samples = reader.samples + (founders.samples if founders else [])
    haps = reader.haps + (founders.haps if founders else [])
    path_rows = []
    members = defaultdict(list)
    for row,nodes in enumerate(paths):
        is_founder = row >= reader.npaths
        kind = "founder" if is_founder else "animal"
        path_rows.append({"path_id": f"{kind}:{json.dumps([samples[row],haps[row]], separators=(',',':'))}",
                          "sample_id": samples[row], "haplotype": haps[row],
                          "is_founder": is_founder, "nodes": nodes})
        for node in nodes:
            members[node].append(row)
    for node,rec in records.items():
        a,b = rec["span"]
        lo,hi,seq,status = _node_sequence(rec,v,reference,start,end)
        rows = members[node]
        animals = sorted({samples[i] for i in rows if i < reader.npaths})
        founder_ids = sorted({samples[i] for i in rows if i >= reader.npaths})
        variant_ids = tuple(v.index[a:b])
        groups = tuple(sorted(set(clusters.ld_group.iloc[a:b])))
        layers = tuple(sorted(set(rec["layers"])))
        G.add_node(node, animals=animals, n_samples=len(animals), founders=founder_ids,
                   n_founders=len(founder_ids), n_paths=len(rows), sequence=seq,
                   sequence_status=status, variants=variant_ids, allele_states=rec["states"],
                   chrom=chromosome, start=lo, end=hi, position_range=(lo,hi),
                   genes=_gene_symbols(genes,lo,hi) if hi>=lo else (),
                   nesting_layer=layers[0] if len(layers)==1 else None,
                   nesting_layers=layers, nesting_depth=min(x.count(".")+1 for x in layers),
                   ld_groups=groups, variant_start=a, variant_stop=b)
        G.nodes[node]['has_unphased_ambiguity'] = 1 in rec['states']
        G.nodes[node]['has_unresolved_calls'] = 3 in rec['states']
    edge_members = defaultdict(list)
    for row,path in enumerate(paths):
        for edge in zip(path,path[1:]):
            edge_members[edge].append(row)
    for source,target in sorted(links):
        rows = edge_members[(source,target)]
        animals = sorted({samples[i] for i in rows if i < reader.npaths})
        fs = sorted({samples[i] for i in rows if i >= reader.npaths})
        G.add_edge(source,target, edge_type="sequence_adjacency", animals=animals,
                   n_samples=len(animals), founders=fs, n_paths=len(rows),
                   recombination_status="not_inferred")
    path_table = pd.DataFrame(path_rows).set_index("path_id")
    G.graph.update(schema="domino.haplotype_radix_dag.v1", assembly=assembly,
        chrom=chromosome, start=start, end=end, coordinate_system="1-based closed",
        root=root, n_samples=len(set(reader.samples)), n_founders=len(set(founders.samples)) if founders else 0,
        phased=reader.phased, path_kind="haplotype" if reader.phased else "unphased_genotype_state",
        phase_warning=None if reader.phased else "paths are genotype patterns; heterozygous phase is unresolved",
        paths=path_table, variants=v, ld_clusters=clusters, ld_edges=ld_edges,
        ld_scope=scope, ld_threshold=float(ld_threshold), ld_r2=dense,
        ld_max_distance_bp=ld_max_distance_bp, ld_max_variant_lag=ld_max_variant_lag,
        unresolved_calls=unresolved, dosage_tolerance=float(dosage_tolerance),
        gtf_path=str(gtf_path) if gtf_path else None,
        reference_sequence_path=str(reference_sequence_path) if reference_sequence_path else None,
        contig_aliases=aliases,
        max_memory_bytes=int(max_memory_bytes), implementation="2.11.4 graph extension")
    G.graph['radix_anchor_variants'] = radix_anchor_variants
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("internal error: sequence graph is not acyclic")
    if return_type == "graph":
        return (G, haplotype_graph_variants(G, format=variant_format)) if return_variant_table else G
    if return_type == "variants":
        return haplotype_graph_variants(G, format=variant_format)
    tables = haplotype_graph_tables(G, format=variant_format)
    if return_type == "tables":
        return tables
    return {"graph": G, **tables, "variants": haplotype_graph_variants(G, format=variant_format)}


def _convert(frame, format, dimension):
    """Convert graph result tables to the requested pandas/xarray/Dask container.

    Parameters
    ----------
    frame : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    format : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Source-recognized literal values: ``dask``,
        ``pandas``, ``xarray``. These select the corresponding branches below; comparison
        evidence alone does not establish an exhaustive enum.

    dimension : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis.
    """
    if format == "pandas":
        return frame
    # JSON strings avoid object-array codecs and retain all ragged memberships.
    flat = frame.copy()
    for column in flat:
        if flat[column].dtype == object:
            flat[column] = flat[column].map(lambda x: json.dumps(x, separators=(",",":"))
                if isinstance(x,(tuple,list,dict)) else ("" if x is None else str(x)))
    if format == "dask":
        import dask.dataframe as dd
        return dd.from_pandas(flat, npartitions=max(1,min(16,len(flat)//10000+1)), sort=False)
    if format == "xarray":
        import xarray as xr
        flat.index = flat.index.rename(dimension)
        ds = xr.Dataset.from_dataframe(flat)
        ds.attrs.update(frame.attrs)
        ds.attrs["ragged_columns_encoding"] = "JSON strings"
        return ds
    raise ValueError("format must be pandas, xarray or dask")


def haplotype_graph_tables(graph, *, format="pandas"):
    """Export graph nodes, edges and observed paths, without Cartesian joins.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    format : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Default: ``'pandas'``.

    Returns
    -------
    dict
        Result fields: ``nodes``, ``edges``, ``paths``. Field arrays preserve the corresponding input axis order.
    """
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)),orient="index").rename_axis("node")
    edges = pd.DataFrame([{"source":s,"target":t,**d} for s,t,d in graph.edges(data=True)])
    if not len(edges):
        edges = pd.DataFrame(columns=["source","target","edge_type","animals","n_samples","founders","n_paths","recombination_status"])
    edges.index.name = "edge"
    paths = graph.graph["paths"].copy()
    for frame in (nodes,edges,paths):
        frame.attrs.update({k:graph.graph[k] for k in ("schema","coordinate_system","path_kind","ld_scope")})
    return {"nodes":_convert(nodes,format,"node"), "edges":_convert(edges,format,"edge"),
            "paths":_convert(paths,format,"path_id")}


def haplotype_graph_variants(graph, *, format="pandas", include_memberships=False):
    """One row per original variant, retaining all node/allele contexts.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    format : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Default: ``'pandas'``.

    include_memberships : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    Returns
    -------
    object
        Computed result from ``_convert(v, format, 'variant')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    v = graph.graph["variants"].copy().join(graph.graph["ld_clusters"])
    contexts = defaultdict(list)
    for node,attrs in graph.nodes(data=True):
        for variant,state in zip(attrs["variants"],attrs["allele_states"]):
            context = {"node":node,"state":int(state), "n_samples":attrs["n_samples"],
                "founders":attrs["founders"],"nesting_layers":attrs["nesting_layers"]}
            if include_memberships:
                context["animals"] = attrs["animals"]
            contexts[variant].append(context)
    v["node_ids"] = [tuple(d["node"] for d in contexts[x]) for x in v.index]
    v["node_contexts"] = [contexts[x] for x in v.index]
    v["nesting_layers"] = [tuple(sorted({z for d in contexts[x] for z in d["nesting_layers"]})) for x in v.index]
    v["n_samples"] = [len({s for d in contexts[x] for s in graph.nodes[d["node"]]["animals"]}) for x in v.index]
    v["n_called_samples"] = [len({s for d in contexts[x] if d["state"] != 3 for s in graph.nodes[d["node"]]["animals"]}) for x in v.index]
    v["founders"] = [tuple(sorted({s for d in contexts[x] for s in d["founders"]})) for x in v.index]
    # SNP-level gene labels use the allele span, not a long compressed-node span.
    genes = _genes(graph.graph["gtf_path"],graph.graph["chrom"],graph.graph["start"],graph.graph["end"],graph.graph.get('contig_aliases'))
    v["genes"] = [_gene_symbols(genes,int(row.pos),int(row.pos)+max(len(row.ref),1)-1) for row in v.itertuples()]
    v.index.name = "variant"
    v.attrs.update({k:graph.graph[k] for k in ("schema","coordinate_system","path_kind","ld_scope")})
    return _convert(v,format,"variant")


def haplotype_graph_features(graph, *, min_samples=2, format="csr"):
    """Observed node-path copy counts for experts; excludes constant/unresolved nodes.

    Phased diploid inputs yield 0/1/2 haplotype copy counts. Unphased inputs
    yield genotype-pattern indicators and are explicitly labelled as such.
    This is a feature adapter, not a fitted causal or fine-mapping model.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    min_samples : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``2``.

    format : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Default: ``'csr'``. Source-recognized literal
        values: ``pandas``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    Returns
    -------
    object
        Computed result from ``(X, ids, metadata)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    paths = graph.graph["paths"]
    paths = paths.loc[~paths.is_founder]
    ids = pd.Index(pd.unique(paths.sample_id), name="iid")
    nodes = [n for n,a in graph.nodes(data=True) if min_samples <= a["n_samples"]
             and (graph.graph['phased'] or a['n_samples'] < len(ids))
             and a["variants"] and 3 not in a["allele_states"]]
    index = {n:j for j,n in enumerate(nodes)}
    samples = {s:i for i,s in enumerate(ids)}
    rows,cols = [],[]
    for row in paths.itertuples():
        for node in row.nodes:
            if node in index:
                rows.append(samples[row.sample_id]);cols.append(index[node])
    X = sparse.coo_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(ids),len(nodes))).tocsr()
    if graph.graph['phased'] and nodes:
        # A haplotype carried by every animal can still have variable 1/2-copy
        # dosage. Presence alone cannot determine whether it is constant.
        variable = (X.max(axis=0).toarray()-X.min(axis=0).toarray()).ravel() > 0
        X = X[:,variable]
        nodes = [node for node,keep in zip(nodes,variable) if keep]
    metadata = pd.DataFrame([{"node":n,"feature":f"radix_node_{n}",
        "n_samples":graph.nodes[n]["n_samples"],"ld_groups":graph.nodes[n]["ld_groups"],
        "nesting_layers":graph.nodes[n]["nesting_layers"]} for n in nodes])
    if not len(metadata):
        metadata = pd.DataFrame(columns=["node","feature","n_samples","ld_groups","nesting_layers"])
    metadata.attrs["path_kind"] = graph.graph["path_kind"]
    if format == "pandas":
        return pd.DataFrame.sparse.from_spmatrix(X,index=ids,columns=metadata.feature), metadata
    if format != "csr":
        raise ValueError("feature format must be csr or pandas")
    return X, ids, metadata


def write_haplotype_graph_zarr(graph, path, *, chunk_rows=4096, shard_rows=65536):
    """Write annotated graph tables as consolidated, sharded Zarr v3 groups.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    chunk_rows : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``4096``.
        Source comparisons: ``chunk_rows < 1``. See the rejection/fallback branches for
        enforcement.

    shard_rows : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``65536``.

    Returns
    -------
    object
        Computed result from ``path``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import xarray as xr
    import zarr
    if chunk_rows < 1 or shard_rows < chunk_rows or shard_rows % chunk_rows:
        raise ValueError("shard_rows must be a positive multiple of chunk_rows")
    path = str(path)
    root = zarr.open_group(path, mode="w", zarr_format=3)
    metadata = {k:graph.graph[k] for k in ("schema","assembly","chrom","start","end",
        "coordinate_system","path_kind","phased","ld_scope","ld_threshold",
        "unresolved_calls","dosage_tolerance","implementation","contig_aliases")}
    root.attrs.update(metadata)
    tables = haplotype_graph_tables(graph,format="xarray")
    tables["variants"] = haplotype_graph_variants(graph,format="xarray")
    tables["ld_edges"] = _convert(graph.graph["ld_edges"].rename_axis("ld_edge"),"xarray","ld_edge")
    for name,ds in tables.items():
        enc = {key:{"chunks":(chunk_rows,),"shards":(shard_rows,)}
               for key,value in ds.variables.items() if value.ndim == 1}
        ds.to_zarr(path,group=name,mode="a",zarr_format=3,consolidated=False,encoding=enc)
    zarr.consolidate_metadata(path)
    return path
