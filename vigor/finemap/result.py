"""Versioned, lossless fine-mapping result container."""
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Mapping
import json, re
import numpy as np
import pandas as pd


def _software_version():
    """Read the installed VIGOR distribution version for fine-mapping provenance.
        
    """
    try:
        from .. import __version__
        return str(__version__)
    except ImportError:return 'unknown'


def _safe(s):
    """Return the safe result computed by ``re.sub``.

    Parameters
    ----------
    s : array-like
        Numerical operand in ``re.sub('[^A-Za-z0-9_.-]+', '_', str(s))``. Shape must conform to
        that matrix/vector operation and the enclosing prepared state.
    """
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s))


def _write_frame(df,path:Path):
    """Write frame.

    Parameters
    ----------
    df : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    path : Path
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.
    """
    try:
        df.to_parquet(path.with_suffix('.parquet'),index=True);return {'type':'dataframe','file':path.with_suffix('.parquet').name,'format':'parquet'}
    except (ImportError,ModuleNotFoundError,ValueError):
        df.to_pickle(path.with_suffix('.pkl'));return {'type':'dataframe','file':path.with_suffix('.pkl').name,'format':'pickle'}


def _read_frame(base:Path,spec):
    """Read frame.

    Parameters
    ----------
    base : Path
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields.

    spec : covariance specification or importlib.ModuleSpec
        Prepared matrix representation for numerical helpers, or importlib specification in the
        Domino import facade. Use the object supplied by the corresponding normalizer/import
        protocol, with unchanged shape/module identity.
    """
    p=base/spec['file']
    return pd.read_parquet(p) if spec.get('format')=='parquet' else pd.read_pickle(p)


def _save_value(base:Path,key:str,value):
    """Persist value.

    Parameters
    ----------
    base : Path
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields.

    key : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    value : object (see the cited input contract)
        Input for the operation ``_write_frame(value, stem)``. See
        ``vigor.finemap.result._save_value`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.
    """
    stem=base/_safe(key)
    if isinstance(value,pd.DataFrame):return _write_frame(value,stem)
    if isinstance(value,pd.Series):
        spec=_write_frame(value.to_frame('__series__'),stem);spec['type']='series';spec['name']=value.name;return spec
    if isinstance(value,np.ndarray):
        import xarray as xr
        from ..store import VigorStore
        f=stem.with_suffix('.zarr')
        dims=tuple(f'axis_{i}' for i in range(value.ndim))
        VigorStore(f,mode='w').write_dataset(xr.Dataset({'values':(dims,value)}),'array')
        return {'type':'ndarray_zarr','file':f.name,'group':'array'}
    if isinstance(value,Mapping):
        return {'type':'mapping','items':{str(k):_save_value(base,f'{key}__{k}',v) for k,v in value.items()}}
    if isinstance(value,(list,tuple)):
        try:return {'type':'json','value':value}
        except Exception:pass
    if value is None or isinstance(value,(str,int,float,bool,np.integer,np.floating)):
        if isinstance(value,(np.integer,np.floating)):value=value.item()
        return {'type':'json','value':value}
    return {'type':'repr','value':repr(value)}


def _load_value(base:Path,spec):
    """Load value.

    Parameters
    ----------
    base : Path
        Reference input or fitted state supplying alignment, initialization or baseline
        quantities. Match the axes/component order of the operation; see the corresponding
        fit/preparation routine for required fields.

    spec : covariance specification or importlib.ModuleSpec
        Prepared matrix representation for numerical helpers, or importlib specification in the
        Domino import facade. Use the object supplied by the corresponding normalizer/import
        protocol, with unchanged shape/module identity. Passed to ``xr.open_zarr`` as ``group``.
    """
    typ=spec['type']
    if typ=='dataframe':return _read_frame(base,spec)
    if typ=='series':
        d=_read_frame(base,spec)['__series__'];d.name=spec.get('name');return d
    if typ=='ndarray':return np.load(base/spec['file'],allow_pickle=False)  # historical input
    if typ=='ndarray_zarr':
        import xarray as xr
        with xr.open_zarr(base/spec['file'],group=spec['group'],consolidated=True) as ds:
            return ds['values'].values
    if typ=='mapping':return {k:_load_value(base,v) for k,v in spec['items'].items()}
    if typ=='json':return spec.get('value')
    return spec.get('value')


@dataclass
class FineMapResult:
    """FineMapResult represents fine map result.

    Attributes
    ----------
    ranking : pd.DataFrame
        Stored ranking; its axis/order follows the construction inputs.
    fits : dict[str, Any]
        Stored fits; its axis/order follows the construction inputs.
    evidence : dict[str, Any] | None
        Stored evidence; its axis/order follows the construction inputs.
    ld : dict[str, Any] | None
        Stored ld; its axis/order follows the construction inputs.
    ld_clusters : pd.DataFrame | None
        Stored ld clusters; its axis/order follows the construction inputs.
    metadata : dict[str, Any]
        Stored metadata; its axis/order follows the construction inputs.
    schema_version : str
        Stored schema version; its axis/order follows the construction inputs.
    software_version : str
        Stored software version; its axis/order follows the construction inputs.
    input_hashes : dict[str, str]
        Stored input hashes; its axis/order follows the construction inputs.
    warnings : list[str]
        Stored warnings; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    ranking : pd.DataFrame
        Variant ranking table with the fit/evidence fields retained by the selected models.
        Default: a fresh value from ``pd.DataFrame()`` for each instance.

    fits : dict[str, Any]
        Named fitted locus/expert models and their estimates/diagnostics. Default: a fresh value
        from ``dict()`` for each instance.

    evidence : dict[str, Any] | None
        Stored evidence consumed by ``to_legacy``/``to_parquet``; see those methods for the
        numerical/label contract. Default: ``None``.

    ld : dict[str, Any] | None
        Stored ld consumed by ``to_legacy``/``to_parquet``; see those methods for the
        numerical/label contract. Default: ``None``.

    ld_clusters : pd.DataFrame | None
        Stored ld clusters consumed by ``to_legacy``/``to_parquet``; see those methods for the
        numerical/label contract. Default: ``None``.

    metadata : dict[str, Any]
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.

    schema_version : str
        Serialized artifact schema identifier; retain it for reader compatibility. Default:
        ``'2.0'``.

    software_version : str
        Writer software version stored as provenance. Default: a fresh value from
        ``_software_version()`` for each instance.

    input_hashes : dict[str, str]
        Input fingerprints used for reproducibility/provenance, keyed by input source. Default:
        a fresh value from ``dict()`` for each instance.

    warnings : list[str]
        Warnings retained from failed/unsupported optional model stages. Default: a fresh value
        from ``list()`` for each instance.
    """
    ranking: pd.DataFrame=field(default_factory=pd.DataFrame)
    fits: dict[str,Any]=field(default_factory=dict)
    evidence: dict[str,Any]|None=None
    ld: dict[str,Any]|None=None
    ld_clusters: pd.DataFrame|None=None
    metadata: dict[str,Any]=field(default_factory=dict)
    schema_version: str='2.0'
    software_version: str=field(default_factory=_software_version)
    input_hashes: dict[str,str]=field(default_factory=dict)
    warnings: list[str]=field(default_factory=list)

    def __post_init__(self):
        """Validate/coerce the constructed dataclass fields for subsequent model operations.
                
        """
        self.ranking=self.ranking.copy() if isinstance(self.ranking,pd.DataFrame) else pd.DataFrame(self.ranking)
        self.fits=dict(self.fits or {});self.metadata=dict(self.metadata or {});self.input_hashes=dict(self.input_hashes or {});self.warnings=list(self.warnings or [])

    @classmethod
    def from_legacy(cls,value:Mapping[str,Any]):
        """Return the from legacy result computed by ``cls``.

        Parameters
        ----------
        value : Mapping[str, Any]
            Input supplied to ``cls`` as ``ranking``. See that callable's documented ``ranking``
            contract for its type, alignment and value domain. This wrapper retains that contract.
            Passed to ``cls`` as ``ranking``.

        Returns
        -------
        object
            Computed result from ``cls(ranking=value.get('ranking', pd.DataFrame()), fits=value.get('fits', {}), evidence=value.get('evidence'), ld=value.get('ld'), ld_clusters=value.get('ld_clusters'), metadata=meta, software_versi...``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if not isinstance(value,Mapping):raise TypeError('fine-map output must be a mapping')
        known={'ranking','fits','evidence','ld','ld_clusters'};meta={k:v for k,v in value.items() if k not in known}
        version=value.get('software_version',_software_version())
        return cls(ranking=value.get('ranking',pd.DataFrame()),fits=value.get('fits',{}),evidence=value.get('evidence'),ld=value.get('ld'),ld_clusters=value.get('ld_clusters'),metadata=meta,software_version=str(version))

    def to_legacy(self):
        """Convert the structured fine-mapping result to the historical mapping representation.

        Returns
        -------
        dict
            Result fields: ``ranking``, ``fits``, ``evidence``, ``ld``, ``ld_clusters``, ``software_version``. Field arrays preserve the corresponding input axis order.
        """
        return {'ranking':self.ranking,'fits':self.fits,'evidence':self.evidence,'ld':self.ld,'ld_clusters':self.ld_clusters,**self.metadata,'software_version':self.software_version}

    def to_parquet(self,path):
        """Persist every serialisable result field and return the manifest path.

        DataFrames use Parquet when an engine is available; the manifest records
        a pickle fallback for minimal environments.  LD matrices and ndarray
        diagnostics use annotated, sharded, consolidated Zarr v3. Historical
        manifests containing ``.npy`` arrays remain readable.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        Returns
        -------
        object
            Computed result from ``f``; see the summary and the called implementation for its labeled/numerical representation.
        """
        base=Path(path);base.mkdir(parents=True,exist_ok=True)
        payload={
            'ranking':_save_value(base,'ranking',self.ranking),
            'fits':_save_value(base,'fits',self.fits),
            'evidence':_save_value(base,'evidence',self.evidence),
            'ld':_save_value(base,'ld',self.ld),
            'ld_clusters':_save_value(base,'ld_clusters',self.ld_clusters),
            'metadata':_save_value(base,'metadata',self.metadata),
        }
        manifest={'schema_version':self.schema_version,'software_version':_software_version(),
                  'input_hashes':self.input_hashes,'warnings':self.warnings,'payload':payload}
        f=base/'finemap_manifest.json';f.write_text(json.dumps(manifest,indent=2,default=str));return f

    @classmethod
    def from_parquet(cls,path):
        """Return the from parquet result computed by ``cls``.

        Parameters
        ----------
        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract.

        Returns
        -------
        object
            Computed result from ``cls(ranking=_load_value(base, pl['ranking']), fits=_load_value(base, pl['fits']) or {}, evidence=_load_value(base, pl['evidence']), ld=_load_value(base, pl['ld']), ld_clusters=_load_value(base, pl[...``; see the summary and the called implementation for its labeled/numerical representation.
        """
        base=Path(path);manifest=json.loads((base/'finemap_manifest.json').read_text());pl=manifest['payload']
        return cls(ranking=_load_value(base,pl['ranking']),fits=_load_value(base,pl['fits']) or {},
                   evidence=_load_value(base,pl['evidence']),ld=_load_value(base,pl['ld']),
                   ld_clusters=_load_value(base,pl['ld_clusters']),metadata=_load_value(base,pl['metadata']) or {},
                   schema_version=manifest.get('schema_version','2.0'),software_version=manifest.get('software_version',_software_version()),
                   input_hashes=manifest.get('input_hashes',{}),warnings=manifest.get('warnings',[]))
