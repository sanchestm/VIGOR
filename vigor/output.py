"""Direct dense GWAS Zarr output for VIGOR 2.12.1."""
from __future__ import annotations
from .store import canonical_provenance, finalize_zarr_artifact
class GWASZarrWriter:
    """Direct region writer for canonical dense ``variant × trait`` GWAS Zarr.

    The writer consumes internal long DataFrame blocks but never exposes that
    representation publicly. Variant metadata are stored once; statistical
    arrays are chunked by variant and trait.
    """
    _KNOWN_CATEGORIES = {
        "dominance_class": ["FILTERED","NA","A","PD","D","OD","UNSTABLE"],
        "inheritance_mode": ["FILTERED","NA","additive","partial_dominant_A1","partial_dominant_A0",
                             "complete_dominant_A1","complete_dominant_A0","overdominant_high",
                             "underdominant_low","additive_near_zero"],
        "test_scope": ["trait","multivariate_joint"],
    }
    def __init__(self, path, *, variant_metadata, trait_names, variant_chunk=65536, trait_chunk=8,
                 overwrite=True):
        """Initialize GWASZarrWriter with the inputs and resource settings below.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        variant_metadata : pandas.DataFrame or metadata mapping
            Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
            fields must describe the same fileset; extra annotation columns are retained according
            to the reader.

        trait_names : sequence or pandas.Index
            Trait labels in matrix-column order, used to label or align the resulting trait
            quantities.

        variant_chunk : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``65536``.

        trait_chunk : int
            Rows/traits per processing or storage tile, along the axis named by the routine.
            Positive counts bound the corresponding intermediate allocation. Default: ``8``.

        overwrite : bool
            Whether an existing destination may be replaced. False protects the existing artifact;
            True permits this writer to recreate it. Default: ``True``.
        """
        try: import zarr
        except ImportError as exc: raise ImportError("GWAS Zarr output requires VIGOR[io]") from exc
        import numpy as np
        import pandas as pd
        from pathlib import Path
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists() and overwrite:
            import shutil; shutil.rmtree(self.path)
        from .store import canonical_provenance
        self.root=zarr.open_group(str(self.path),mode="w",zarr_format=3)
        self.root.attrs.update(canonical_provenance("gwas_result",write_state="writing",
                                result_type="dense_multivariate_gwas",domino_data_model="2.11.0"))
        vm=pd.DataFrame(variant_metadata).copy()
        if "variant_index" in vm.columns:
            global_variant=np.asarray(vm["variant_index"],dtype=np.int64)
        else:
            global_variant=np.asarray(vm.index,dtype=np.int64)
            vm["variant_index"]=global_variant
        vm=vm.reset_index(drop=True); self.nv=len(vm)
        self.global_variant_index=np.asarray(global_variant,dtype=np.int64)
        self.variant_map={int(v):i for i,v in enumerate(self.global_variant_index)}
        if len(self.variant_map)!=self.nv: raise ValueError("variant_index must be unique in GWAS writer metadata")
        self.traits=np.asarray(list(map(str,trait_names)),dtype=str); self.nt=len(self.traits)
        self.trait_map={v:i for i,v in enumerate(self.traits)}
        self.variant_chunk=min(int(variant_chunk),max(1,self.nv)); self.trait_chunk=min(int(trait_chunk),max(1,self.nt))
        def create1(name, vals, dim="variant"):
            """Create a preallocated labeled output array in the enclosing destination.

            Parameters
            ----------
            name : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.

            vals : array-like or numeric scalar
                Input supplied to ``self.root.create_array`` as ``shape``. See that callable's
                documented ``shape`` contract for its type, alignment and value domain. This wrapper
                retains that contract. Passed to ``self.root.create_array`` as ``shape``.

            dim : str, index or sequence
                Axis/column/feature selector used to identify the input/output quantities in this
                operation. Named selectors must exist in the labeled input; integer selectors are valid
                zero-based positions on that axis. Default: ``'variant'``. Passed to
                ``self.root.create_array`` as ``dimension_names``.

            Returns
            -------
            object
                Computed result from ``a``; see the summary and the called implementation for its labeled/numerical representation.
            """
            vals=np.asarray(vals)
            if vals.dtype.kind in "OUS":
                width=max(1,max((len(str(x)) for x in vals),default=1)); dtype=f"U{width}"; vals=vals.astype(dtype)
            else: dtype=vals.dtype
            ch=(min(len(vals),65536),)
            a=self.root.create_array(name,shape=(len(vals),),chunks=ch,dtype=dtype,dimension_names=(dim,),overwrite=True)
            a.attrs["_ARRAY_DIMENSIONS"]=[dim]; a[:]=vals; return a
        create1("variant",self.global_variant_index)
        create1("trait",self.traits,dim="trait")
        for src,dst in (("snp","variant_id"),("variant_id","variant_id"),("chrom","chromosome"),("chromosome","chromosome"),("pos","position"),("position","position"),("ref","ref"),("alt","alt")):
            if src in vm and dst not in self.root:
                create1(dst,vm[src].to_numpy())
        self.arrays={}; self.rows_written=0; self.blocks_written=0

    def _ensure_array(self,name,series):
        """Validate/coerce array.

        Parameters
        ----------
        name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        series : array-like or numeric scalar
            Input for the operation ``pd.api.types.is_numeric_dtype(series.dtype)``. See
            ``vigor.output.GWASZarrWriter._ensure_array`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.
        """
        import numpy as np, pandas as pd
        if name in self.arrays: return self.arrays[name]
        if name in self._KNOWN_CATEGORIES:
            a=self.root.create_array(name,shape=(self.nv,self.nt),chunks=(self.variant_chunk,self.trait_chunk),dtype=np.int16,
                                     fill_value=-1,dimension_names=("variant","trait"),overwrite=True)
            a.attrs.update({"_ARRAY_DIMENSIONS":["variant","trait"],"categories":self._KNOWN_CATEGORIES[name]})
        elif pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            dtype=np.float64 if series.dtype.kind=="f" else np.float64
            a=self.root.create_array(name,shape=(self.nv,self.nt),chunks=(self.variant_chunk,self.trait_chunk),dtype=dtype,
                                     fill_value=np.nan,dimension_names=("variant","trait"),overwrite=True)
            a.attrs["_ARRAY_DIMENSIONS"]=["variant","trait"]
        else:
            return None
        self.arrays[name]=a; return a

    def write(self,frame):
        """Write one completed scientific result block to the configured destination.

        Parameters
        ----------
        frame : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.
        """
        import numpy as np, pandas as pd
        if frame is None or len(frame)==0:return
        meta={"variant_index","trait","snp","chrom","pos","variant_id","chromosome","position","ref","alt"}
        for col in frame.columns:
            if col in meta:continue
            arr=self._ensure_array(col,frame[col])
            if arr is None:continue
            for trait,sub in frame[["variant_index","trait",col]].groupby("trait",sort=False):
                t=self.trait_map.get(str(trait));
                if t is None: continue
                sub=sub.sort_values("variant_index"); gidx=sub["variant_index"].to_numpy(np.int64)
                if len(gidx)==0:continue
                try: idx=np.asarray([self.variant_map[int(i)] for i in gidx],dtype=np.int64)
                except KeyError as exc: raise KeyError(f"GWAS block contains variant_index absent from writer metadata: {exc.args[0]}") from exc
                vals=sub[col]
                if col in self._KNOWN_CATEGORIES:
                    mp={v:i for i,v in enumerate(self._KNOWN_CATEGORIES[col])}; out=vals.astype(str).map(mp).fillna(-1).to_numpy(np.int16)
                else: out=pd.to_numeric(vals,errors="coerce").to_numpy(float)
                if np.all(np.diff(idx)==1): arr[int(idx[0]):int(idx[-1])+1,t]=out
                else:
                    for i,v in zip(idx,out): arr[int(i),t]=v
        self.rows_written+=len(frame); self.blocks_written+=1

    def close(self, *, attrs=None, consolidated=True):
        """Release resources retained by this reader, operator or progress object.

        Parameters
        ----------
        attrs : mapping or None
            Named settings or metadata forwarded to the specific operation below. See that
            operation's argument/schema contract for accepted fields; None uses its default/empty
            configuration. Default: ``None``.

        consolidated : bool or None
            Whether to request consolidated Zarr metadata. None lets the reader choose/fall back
            according to the store metadata and backend support. Default: ``True``.
        """
        self.root.attrs["domino_write_state"]="complete"
        if attrs:
            for k,v in attrs.items():
                try:self.root.attrs[k]=v
                except Exception:self.root.attrs[k]=str(v)
        if consolidated:
            self.root=finalize_zarr_artifact(str(self.path),artifact_type="mvgwas_result",metadata={"rows_written":int(self.rows_written),"blocks_written":int(self.blocks_written)})

    def abort(self):
        """Close/discard pending output after an interrupted or failed result-writing operation.
                
        """
        self.root.attrs["domino_write_state"]="aborted"
