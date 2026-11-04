"""Bounded-memory physical genotype backends for VIGOR 2.12.1."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .io import PlinkReader


@dataclass
class GenotypeBlock:
    """GenotypeBlock represents genotype block.

    Attributes
    ----------
    data : np.ndarray
        Stored data; its axis/order follows the construction inputs.
    variants : pd.DataFrame
        Stored variants; its axis/order follows the construction inputs.
    sample_ids : np.ndarray
        Stored sample ids; its axis/order follows the construction inputs.
    representation : str
        Stored representation; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    data : np.ndarray
        Sample × variant genotype block with its representation-specific missing-call encoding.
        Required construction field.

    variants : pd.DataFrame
        Variant metadata rows in the data block column order. Required construction field.

    sample_ids : np.ndarray
        Animal identifiers in the data block row order. Required construction field.

    representation : str
        Genotype block representation, normally 'dosage'; interpretation follows the source
        adapter. Default: ``'dosage'``.
    """
    data: np.ndarray
    variants: pd.DataFrame
    sample_ids: np.ndarray
    representation: str = "dosage"


class ArrayGenotypeSource:
    """ArrayGenotypeSource represents array genotype source.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    def __init__(self, data, *, sample_ids=None, variant_metadata=None):
        """Initialize ArrayGenotypeSource with the inputs and resource settings below.

        Parameters
        ----------
        data : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        sample_ids : sequence of str or None
            Animal identifiers in the required row order. IDs must align with the genotype/phenotype
            source; None uses source identifiers or the routine's positional indexing convention.
            Default: ``None``.

        variant_metadata : pandas.DataFrame or metadata mapping
            Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
            fields must describe the same fileset; extra annotation columns are retained according
            to the reader. Default: ``None``. None delegates to the local fallback shown in this
            body; it is not an additional numeric/category value.
        """
        X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError("genotype array must be sample x variant")
        self.data = X
        self.sample_ids = np.asarray(sample_ids if sample_ids is not None else np.arange(X.shape[0]).astype(str), dtype=str)
        if len(self.sample_ids) != X.shape[0]:
            raise ValueError("sample_ids must have one entry per sample")
        if len(np.unique(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique")
        if variant_metadata is None:
            variant_metadata = pd.DataFrame({"variant_id": [f"v{i}" for i in range(X.shape[1])], "chromosome": "1", "position": np.arange(X.shape[1]) + 1})
        self.variant_metadata = pd.DataFrame(variant_metadata).reset_index(drop=True)
        if len(self.variant_metadata) != X.shape[1]:
            raise ValueError("variant_metadata must have one row per variant")
        self.founder_ids = None
        self.capabilities = frozenset({"dosage"})

    def iter_blocks(self, representation="dosage", samples=None, chrom=None, region=None, block_size='auto', execution_plan=None):
        """Iterate over blocks.

        Parameters
        ----------
        representation : str
            Returned/stored container or kernel representation. The source-derived choices below
            select dense arrays, labeled tables or lazy/operator representations; see the local
            dispatcher for which are supported. Default: ``'dosage'``.

        samples : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``None``.
            None delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported. Default: ``None``.

        region : pair of int or None
            Lower and upper genomic positions defining the selected interval. Positions refer to the
            chromosome's physical coordinate system; None uses the full eligible region. Default:
            ``None``.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
            from the memory budget; integers are ceilings).

        execution_plan : ExecutionPlan or ResourceConfig or None
            Resolved resource/chunking plan for this workload. None lets the resource planner derive
            a plan; explicit settings must fit the available memory and worker budget. Default:
            ``None``.

        Yields
        ------
        object
            Successive blocks or values in iterator order, following the representation described above.
        """
        block_size = _resolve_block_size(self, block_size, execution_plan, representation)
        if representation != "dosage":
            raise ValueError("array source provides dosage only")
        sidx = np.arange(len(self.sample_ids)) if samples is None else pd.Index(self.sample_ids).get_indexer(pd.Index(samples).astype(str))
        if np.any(sidx < 0):
            raise KeyError("requested sample is absent")
        vm = self.variant_metadata
        mask = np.ones(len(vm), bool)
        if chrom is not None and "chromosome" in vm:
            mask &= vm["chromosome"].astype(str).to_numpy() == str(chrom)
        if region is not None and "position" in vm:
            lo, hi = region
            pos = pd.to_numeric(vm["position"], errors="coerce").to_numpy()
            mask &= (pos >= lo) & (pos <= hi)
        vidx = np.flatnonzero(mask)
        for st in range(0, len(vidx), int(block_size)):
            j = vidx[st:st + int(block_size)]
            yield GenotypeBlock(np.asarray(self.data[np.ix_(sidx, j)]), vm.iloc[j].copy(), self.sample_ids[sidx])


class Plink1GenotypeSource:
    """Plink1GenotypeSource represents plink genotype source.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    def __init__(self, prefix):
        """Initialize Plink1GenotypeSource with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.
        """
        self.reader = PlinkReader(str(prefix))
        self.sample_ids = self.reader.fam["iid"].astype(str).to_numpy()
        vm = self.reader.bim.rename(columns={"snp": "variant_id", "chrom": "chromosome", "pos": "position"}).copy()
        self.variant_metadata = vm
        self.founder_ids = None
        self.capabilities = frozenset({"dosage"})

    def iter_blocks(self, representation="dosage", samples=None, chrom=None, region=None, block_size='auto', execution_plan=None):
        """Iterate over blocks.

        Parameters
        ----------
        representation : str
            Returned/stored container or kernel representation. The source-derived choices below
            select dense arrays, labeled tables or lazy/operator representations; see the local
            dispatcher for which are supported. Default: ``'dosage'``.

        samples : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``None``.
            None delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported. Default: ``None``.

        region : pair of int or None
            Lower and upper genomic positions defining the selected interval. Positions refer to the
            chromosome's physical coordinate system; None uses the full eligible region. Default:
            ``None``.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
            from the memory budget; integers are ceilings). Passed
            to ``self.reader.iter_blocks`` as ``block_size``.

        execution_plan : ExecutionPlan or ResourceConfig or None
            Resolved resource/chunking plan for this workload. None lets the resource planner derive
            a plan; explicit settings must fit the available memory and worker budget. Default:
            ``None``.

        Yields
        ------
        object
            Successive blocks or values in iterator order, following the representation described above.
        """
        block_size = _resolve_block_size(self, block_size, execution_plan, representation)
        if representation != "dosage":
            raise ValueError("PLINK1 source provides dosage only")
        sidx = None
        out_ids = self.sample_ids
        if samples is not None:
            sidx = pd.Index(self.sample_ids).get_indexer(pd.Index(samples).astype(str))
            if np.any(sidx < 0): raise KeyError("requested sample is absent")
            out_ids = self.sample_ids[sidx]
        vm = self.variant_metadata
        mask = np.ones(len(vm), bool)
        if chrom is not None: mask &= vm["chromosome"].astype(str).to_numpy() == str(chrom)
        if region is not None:
            lo, hi = region; p = vm["position"].to_numpy(); mask &= (p >= lo) & (p <= hi)
        idx = np.flatnonzero(mask)
        for meta, X in self.reader.iter_blocks(block_size=int(block_size), variant_index=idx, sample_index=sidx,
                                               maxmem=getattr(self,'maxmem',None), memory_fraction=getattr(self,'memory_fraction',None)):
            meta = meta.rename(columns={"snp":"variant_id", "chrom":"chromosome", "pos":"position"})
            yield GenotypeBlock(X, meta.copy(), out_ids)

    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        self.reader.close()




def _read_plink2_table(path, header_token):
    """Read plink2 table.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    header_token : str, sequence or mapping
        Pattern/filter selectors applied to the resource/table search. Use patterns matching the
        required filenames/header or existing metadata fields.
    """
    import gzip
    path=str(path)
    opener=gzip.open if path.endswith('.gz') else open
    header=None; line_no=None
    with opener(path,'rt') as fh:
        for i,line in enumerate(fh):
            if line.startswith(header_token):
                header=line.rstrip('\n').split('\t') if '\t' in line else line.split(); line_no=i; break
            if line.strip() and not line.lstrip().startswith('#'): break
    if header is None: raise ValueError(f"missing {header_token} header in {path}")
    header=[x.lstrip('#') for x in header]
    return pd.read_csv(path,sep=r'\s+',names=header,header=None,skiprows=line_no+1,compression='infer',low_memory=False)


class Plink2GenotypeSource:
    """Bounded-memory PGEN adapter using the reference optional ``pgenlib`` backend."""
    def __init__(self,prefix):
        """Initialize Plink2GenotypeSource with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.
        """
        try: import pgenlib
        except ImportError as exc:
            raise ImportError("PLINK2/PGEN input requires pgenlib; install VIGOR[io] or use canonical Zarr/PLINK1") from exc
        prefix=_strip_prefix(prefix); self.prefix=prefix; self._pgenlib=pgenlib
        pvar=prefix+'.pvar' if Path(prefix+'.pvar').exists() else prefix+'.pvar.zst'
        if pvar.endswith('.zst'):
            raise ImportError("compressed .pvar.zst requires conversion to .pvar in the pgenlib adapter")
        self.psam=_read_plink2_table(prefix+'.psam','#IID') if not any(line.startswith('#FID') for line in open(prefix+'.psam','rt')) else _read_plink2_table(prefix+'.psam','#FID')
        self.pvar=_read_plink2_table(pvar,'#CHROM')
        iid='IID' if 'IID' in self.psam else self.psam.columns[0]; self.sample_ids=self.psam[iid].astype(str).to_numpy()
        if len(np.unique(self.sample_ids))!=len(self.sample_ids): raise ValueError('duplicate IID(s) in PSAM')
        self.variant_metadata=pd.DataFrame({
            'variant_id':self.pvar['ID'].astype(str), 'chromosome':self.pvar['CHROM'].astype(str),
            'position':pd.to_numeric(self.pvar['POS'],errors='raise').astype(np.int64),
            'ref':self.pvar.get('REF',pd.Series('.',index=self.pvar.index)).astype(str),
            'alt':self.pvar.get('ALT',pd.Series('.',index=self.pvar.index)).astype(str),
        }).reset_index(drop=True)
        if self.variant_metadata['alt'].str.contains(',', regex=False).any():
            raise ValueError('PGEN dosage adapter requires biallelic variants; multiallelic ALT is unsupported')
        # pgenlib read() defaults to allele_idx=1: the first ALT allele. This
        # adapter is explicitly biallelic, so each 0/1/2 dosage counts ALT.
        self.variant_metadata['effect_allele'] = self.variant_metadata['alt']
        self.variant_metadata['other_allele'] = self.variant_metadata['ref']
        self.effect_allele_convention = 'ALT_counted_biallelic_pgen'
        self.reader=pgenlib.PgenReader(bytes(prefix+'.pgen','utf8'), raw_sample_ct=len(self.sample_ids))
        self.founder_ids=None; self.capabilities=frozenset({'dosage'})
    def iter_blocks(self,representation='dosage',samples=None,chrom=None,region=None,block_size='auto',execution_plan=None):
        """Iterate over blocks.

        Parameters
        ----------
        representation : str
            Returned/stored container or kernel representation. The source-derived choices below
            select dense arrays, labeled tables or lazy/operator representations; see the local
            dispatcher for which are supported. Default: ``'dosage'``.

        samples : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``None``.
            None delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported. Default: ``None``.

        region : pair of int or None
            Lower and upper genomic positions defining the selected interval. Positions refer to the
            chromosome's physical coordinate system; None uses the full eligible region. Default:
            ``None``.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
            from the memory budget; integers are ceilings).

        execution_plan : ExecutionPlan or ResourceConfig or None
            Resolved resource/chunking plan for this workload. None lets the resource planner derive
            a plan; explicit settings must fit the available memory and worker budget. Default:
            ``None``.

        Yields
        ------
        object
            Successive blocks or values in iterator order, following the representation described above.
        """
        block_size = _resolve_block_size(self, block_size, execution_plan, representation)
        if representation!='dosage': raise ValueError('PGEN adapter exposes biallelic hardcall dosage')
        sidx=np.arange(len(self.sample_ids)) if samples is None else pd.Index(self.sample_ids).get_indexer(pd.Index(samples).astype(str))
        if np.any(sidx<0): raise KeyError('requested sample is absent')
        vm=self.variant_metadata; mask=np.ones(len(vm),bool)
        if chrom is not None: mask &= vm.chromosome.astype(str).to_numpy()==str(chrom)
        if region is not None:
            lo,hi=region; pp=vm.position.to_numpy(); mask &= (pp>=lo)&(pp<=hi)
        vidx=np.flatnonzero(mask); buf=np.empty(len(self.sample_ids),dtype=np.int8)
        for st in range(0,len(vidx),int(block_size)):
            j=vidx[st:st+int(block_size)]; out=np.empty((len(sidx),len(j)),dtype=np.float32)
            for k,v in enumerate(j):
                self.reader.read(int(v),buf); x=buf.astype(np.float32); x[x<0]=np.nan; out[:,k]=x[sidx]
            yield GenotypeBlock(out,vm.iloc[j].copy(),self.sample_ids[sidx])
    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        try:self.reader.close()
        except Exception:pass


class XarrayGenotypeSource:
    """XarrayGenotypeSource represents xarray genotype source.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    def __init__(self, obj, *, dosage_variable="dosage", hardcall_variable="hardcall", founder_variable="founder_probability", group=None, lazy=False):
        """Initialize XarrayGenotypeSource with the inputs and resource settings below.

        Parameters
        ----------
        obj : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        dosage_variable : str
            Name of the corresponding genotype data variable in the xarray Dataset; it must expose
            the sample/variant (and founder when applicable) dimensions expected by this adapter.
            Default: ``'dosage'``.

        hardcall_variable : str
            Name of the corresponding genotype data variable in the xarray Dataset; it must expose
            the sample/variant (and founder when applicable) dimensions expected by this adapter.
            Default: ``'hardcall'``.

        founder_variable : str
            Name of the corresponding genotype data variable in the xarray Dataset; it must expose
            the sample/variant (and founder when applicable) dimensions expected by this adapter.
            Default: ``'founder_probability'``.

        group : sequence, str or None
            Grouping labels used to select a store subgroup or partition observations. For
            resampling, give one cluster label per animal so a cluster is sampled as a unit.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value. Passed to ``xr.open_zarr`` as ``group``.

        lazy : bool
            Whether store-backed values remain lazily loaded until their blocks are accessed; eager
            mode materializes the arrays requested by the reader. Default: ``False``.
        """
        try: import xarray as xr
        except ImportError as exc: raise ImportError("xarray is required for xarray/Zarr genotype sources") from exc
        self._path = None if isinstance(obj, xr.Dataset) else str(obj)
        self._group = group
        if isinstance(obj, xr.Dataset):
            self.ds = obj
        else:
            open_kwargs = dict(consolidated=False, chunks={} if lazy else None)
            try:
                open_kwargs["create_default_indexes"] = False
                self.ds = xr.open_zarr(obj, group=group, **open_kwargs)
            except (TypeError, ValueError, KeyError):
                open_kwargs.pop("create_default_indexes", None)
                # Canonical 2.9 stores keep genotypes in /genotype; old stores
                # remain readable from the root.
                try:
                    self.ds = xr.open_zarr(obj, group=group, **open_kwargs)
                except Exception:
                    if group is not None: raise
                    self.ds = xr.open_zarr(obj, group="genotype", **open_kwargs)
                    self._group = "genotype"
        # Decode canonical packed GT lazily; use the same logical schema for all backends.
        if "GT_packed2" in self.ds:
            if self._path is None:
                from .data_model import _unpack2_dask
                self.ds["GT"] = (("sample","variant"), _unpack2_dask(self.ds["GT_packed2"].data, self.ds.sizes["variant"]))
            else:
                from .data_model import open_genotype_dataset
                self.ds.close()
                self.ds = open_genotype_dataset(self._path)
        if dosage_variable not in self.ds and "DS" in self.ds: dosage_variable="DS"
        if hardcall_variable not in self.ds and "GT" in self.ds: hardcall_variable="GT"
        if founder_variable not in self.ds and "HP" in self.ds: founder_variable="HP"
        self.dosage_variable = dosage_variable
        self.hardcall_variable = hardcall_variable
        self.founder_variable = founder_variable
        self._direct_arrays = {}
        if self._path is not None:
            try:
                import zarr
                zg = zarr.open_group(self._path, mode="r")
                if self._group is not None:
                    for part in str(self._group).strip("/").split("/"):
                        if part: zg = zg[part]
                for name in (dosage_variable, hardcall_variable, founder_variable, "phased_allele", "diplotype_probability"):
                    try:
                        if name in zg: self._direct_arrays[name] = zg[name]
                    except Exception:
                        pass
            except Exception:
                self._direct_arrays = {}
        if "sample" not in self.ds.dims or "variant" not in self.ds.dims:
            raise ValueError("canonical xarray source requires sample and variant dimensions")
        sample_coord = self.ds.coords.get("sample")
        self.sample_ids = np.asarray(
            sample_coord.values if sample_coord is not None else np.arange(self.ds.sizes["sample"]), dtype=object
        ).astype(str)
        if len(np.unique(self.sample_ids)) != len(self.sample_ids): raise ValueError("sample IDs must be unique")
        self.founder_ids = np.asarray(self.ds.coords["founder"].values, dtype=object).astype(str) if "founder" in self.ds.coords else None
        caps = set()
        if dosage_variable in self.ds or hardcall_variable in self.ds: caps.add("dosage")
        if hardcall_variable in self.ds: caps.add("hardcall")
        if founder_variable in self.ds: caps.add("founder_probability")
        if "phased_allele" in self.ds: caps.add("phased_calls")
        if "diplotype_probability" in self.ds: caps.add("diplotypes")
        self.capabilities = frozenset(caps)
        def _coord(name, default):
            """Return the coord result computed by ``np.asarray``.

            Parameters
            ----------
            name : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.

            default : array-like or numeric scalar
                Input for the operation ``np.asarray(coord.values if coord is not None else default,
                dtype=object)``. See ``vigor.genotype_source.XarrayGenotypeSource.__init__._coord`` and
                its caller for the scientific representation; the shown operation constrains the
                required type/shape.
            """
            coord = self.ds.coords.get(name)
            if coord is None: coord = self.ds.variables.get(name)
            return np.asarray(coord.values if coord is not None else default, dtype=object)
        self.variant_metadata = pd.DataFrame({
            "variant_id": _coord("variant_id", _coord("variant", np.arange(self.ds.sizes["variant"]))).astype(str),
            "chromosome": _coord("chromosome", np.repeat("", self.ds.sizes["variant"])).astype(str),
            "position": _coord("position", np.arange(self.ds.sizes["variant"])),
        })
        for name in ("ref", "alt", "a0", "a1", "effect_allele", "other_allele"):
            if name in self.ds.variables:
                values = self.ds[name]
                if values.dims != ("variant",):
                    raise ValueError(f"{name} metadata must be one-dimensional on variant")
                self.variant_metadata[name] = np.asarray(values.values).astype(str)
        # REF/ALT alone do not establish which allele this producer counted.
        # Prefer explicit effect/other coordinates, then a declared convention
        # on the selected dosage/hardcall variable or containing Dataset.
        vm = self.variant_metadata
        attrs = dict(self.ds.attrs)
        call_variable = dosage_variable if dosage_variable in self.ds else hardcall_variable
        if call_variable in self.ds:
            attrs.update(self.ds[call_variable].attrs)
        declared = None
        for key in ("counted_allele", "dosage_allele", "genotype_counted_allele", "effect_allele_convention"):
            value = attrs.get(key)
            if value is not None:
                declared = str(value).strip().lower()
                break
        pairs = {
            "alt": ("alt", "ref"), "alt_counted": ("alt", "ref"),
            "alt_counted_biallelic_pgen": ("alt", "ref"),
            "ref": ("ref", "alt"), "ref_counted": ("ref", "alt"),
            "a1": ("a1", "a0"), "bim_column_6_counted": ("a1", "a0"),
            "a0": ("a0", "a1"), "bim_column_5_counted": ("a0", "a1"),
        }
        pair = pairs.get(declared)
        if pair is not None and all(name in vm for name in pair):
            if any(vm[name].str.contains(",", regex=False).any() for name in pair):
                raise ValueError("counted-allele derivation requires biallelic metadata")
            if "effect_allele" not in vm:
                vm["effect_allele"] = vm[pair[0]]
            if "other_allele" not in vm:
                vm["other_allele"] = vm[pair[1]]
        self.effect_allele_convention = declared or (
            "explicit_effect_allele" if "effect_allele" in vm else "unspecified")
        if any(name in vm for name in ("ref", "alt", "a0", "a1")) and "effect_allele" not in vm:
            import warnings
            warnings.warn("genotype REF/ALT metadata retained, but counted allele is unspecified; "
                          "provide effect_allele/other_allele coordinates or a counted_allele attribute",
                          RuntimeWarning, stacklevel=2)

    def iter_blocks(self, representation="dosage", samples=None, chrom=None, region=None, block_size='auto', execution_plan=None):
        """Iterate over blocks.

        Parameters
        ----------
        representation : str
            Returned/stored container or kernel representation. The source-derived choices below
            select dense arrays, labeled tables or lazy/operator representations; see the local
            dispatcher for which are supported. Default: ``'dosage'``.

        samples : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering. Default: ``None``.
            None delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported. Default: ``None``.

        region : pair of int or None
            Lower and upper genomic positions defining the selected interval. Positions refer to the
            chromosome's physical coordinate system; None uses the full eligible region. Default:
            ``None``.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
            from the memory budget; integers are ceilings).

        execution_plan : ExecutionPlan or ResourceConfig or None
            Resolved resource/chunking plan for this workload. None lets the resource planner derive
            a plan; explicit settings must fit the available memory and worker budget. Default:
            ``None``.

        Yields
        ------
        object
            Successive blocks or values in iterator order, following the representation described above.
        """
        block_size = _resolve_block_size(self, block_size, execution_plan, representation)
        var = {"dosage": self.dosage_variable, "hardcall": self.hardcall_variable,
               "founder_probability": self.founder_variable, "phased_calls":"phased_allele",
               "diplotypes":"diplotype_probability"}.get(representation, representation)
        decode_hardcall = representation == "dosage" and var not in self.ds and self.hardcall_variable in self.ds
        if decode_hardcall: var = self.hardcall_variable
        if var not in self.ds: raise ValueError(f"representation {representation!r} is unavailable")
        sidx = np.arange(len(self.sample_ids)) if samples is None else pd.Index(self.sample_ids).get_indexer(pd.Index(samples).astype(str))
        if np.any(sidx < 0): raise KeyError("requested sample is absent")
        vm = self.variant_metadata; mask = np.ones(len(vm), bool)
        if chrom is not None: mask &= vm.chromosome.astype(str).to_numpy() == str(chrom)
        if region is not None:
            lo, hi = region; p = pd.to_numeric(vm.position, errors="coerce").to_numpy(); mask &= (p >= lo) & (p <= hi)
        vidx = np.flatnonzero(mask)
        for st in range(0, len(vidx), int(block_size)):
            j = vidx[st:st+int(block_size)]
            # only the bounded block is materialized
            # Canonicalize storage layouts (sample x variant); Zarr stores may
            # legally carry dimensions in the opposite order.
            da = self.ds[var]
            if not {"sample", "variant"}.issubset(da.dims):
                raise ValueError(f"{var!r} must have sample and variant dimensions")
            # Fast path for the dominant VIGOR access pattern: all samples ×
            # a contiguous variant range. This bypasses xarray/Dask dispatch
            # while keeping xarray for coordinates and alignment.
            direct = self._direct_arrays.get(var)
            full_samples = len(sidx) == len(self.sample_ids) and np.array_equal(sidx, np.arange(len(self.sample_ids)))
            contiguous = len(j) == 0 or np.all(np.diff(j) == 1)
            if direct is not None and da.dims[:2] == ("sample", "variant") and full_samples and contiguous and len(j):
                arr = np.asarray(direct[:, int(j[0]):int(j[-1])+1])
            else:
                arr = da.transpose("sample", "variant", ...).isel(sample=sidx, variant=j).load().values
            if decode_hardcall:
                arr=np.asarray(arr,dtype=np.float32); arr[arr < 0]=np.nan
            yield GenotypeBlock(arr, vm.iloc[j].copy(), self.sample_ids[sidx], representation)


    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        self.ds.close()


def _resolve_block_size(source, block_size, execution_plan, representation):
    """Resolve block size.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations. Passed to
        ``bounded_variant_chunk`` as ``maxmem``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Source-recognized literal
        values: ``auto``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    execution_plan : ExecutionPlan or ResourceConfig or None
        Resolved resource/chunking plan for this workload. None lets the resource planner derive
        a plan; explicit settings must fit the available memory and worker budget.

    representation : str
        Returned/stored container or kernel representation. The source-derived choices below
        select dense arrays, labeled tables or lazy/operator representations; see the local
        dispatcher for which are supported. Source-recognized literal values: ``diplotypes``,
        ``founder_probability``, ``phased_calls``. These select the corresponding branches
        below; comparison evidence alone does not establish an exhaustive enum.
    """
    from .resources import bounded_variant_chunk
    auto = block_size is None or str(block_size).lower() == 'auto'
    ceiling = execution_plan.variant_block_size if execution_plan is not None else ('auto' if auto else block_size)
    if not auto and execution_plan is not None and str(ceiling).lower() != 'auto':
        ceiling = min(int(block_size), int(ceiling))
    elif not auto:
        ceiling = int(block_size)
    multiplier=1
    if hasattr(source,'ds'):
        for v in (getattr(source,'founder_variable',''), 'phased_allele', 'diplotype_probability'):
            if representation in {'founder_probability','phased_calls','diplotypes'} and v in source.ds:
                multiplier=max(multiplier,int(np.prod([source.ds[v].sizes[d] for d in source.ds[v].dims if d not in {'sample','variant'}])))
    # float32 block + one transpose/load copy + decode scratch, scaled by founder/allele axes
    return bounded_variant_chunk(len(source.sample_ids), requested='auto' if ceiling is None else ceiling,
                                 maxmem=getattr(source,'maxmem',None), memory_fraction=getattr(source,'memory_fraction',None),
                                 bytes_per_call=16*multiplier)


def _strip_prefix(path):
    """Remove a recognized genotype companion-file suffix to recover the fileset prefix.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.
    """
    s = str(path)
    for ext in (".pvar.zst", ".pgen.pgi", ".pgen", ".pvar", ".psam", ".bed", ".bim", ".fam"):
        if s.endswith(ext): return s[:-len(ext)]
    return s


def open_genotypes(source, *, source_format="auto", maxmem=None, **kwargs):
    """Open a canonical genotype source without silently selecting a competing format.

    If both PLINK1 and PLINK2 files exist for the same prefix, ``source_format='auto'``
    raises instead of silently preferring one representation.

    Parameters
    ----------
    source : path-like or genotype source
        Input file/store location or opened source consumed by this adapter. See the selected
        reader for physical format, sample order and supported block representations.

    source_format : str
        Physical genotype backend: 'auto' identifies the source and rejects ambiguous
        PLINK1/PLINK2 prefixes; 'plink1'/'bed'/'bfile' selects BED, 'plink2'/'pgen'/'pfile'
        selects PGEN, and store/array support follows open_genotypes. Zarr uses its canonical
        schema. Default: ``'auto'``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``ArrayGenotypeSource``,
        ``XarrayGenotypeSource``. See those callables for accepted keys and values. Passed to
        ``ArrayGenotypeSource`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if hasattr(source, "iter_blocks") and hasattr(source, "sample_ids"):
        result=source
    elif isinstance(source,np.ndarray):
        result=ArrayGenotypeSource(source,**kwargs)
    else:
        import xarray as xr
        if isinstance(source,xr.Dataset):
            result=XarrayGenotypeSource(source,**kwargs)
        else:
            from .data_model import _source_kind
            kind,loc=_source_kind(source,source_format)
            if kind=='plink1': result=Plink1GenotypeSource(loc)
            elif kind=='plink2': result=Plink2GenotypeSource(loc)
            else: result=XarrayGenotypeSource(loc,group="genotype",**kwargs)
    if maxmem is not None:
        from .resources import parse_memory_bytes
        result.maxmem=parse_memory_bytes(maxmem)
    return result

__all__ = ["GenotypeBlock","ArrayGenotypeSource","Plink1GenotypeSource","Plink2GenotypeSource","XarrayGenotypeSource","open_genotypes"]
