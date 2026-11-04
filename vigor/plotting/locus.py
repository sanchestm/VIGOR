"""Interactive hvPlot regional association plots from canonical xarray GWAS."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np,pandas as pd

@dataclass
class LocusPlotData:
    """LocusPlotData represents locus plot data.

    Attributes
    ----------
    associations : object
        Stored associations; its axis/order follows the construction inputs.
    genes : pd.DataFrame | None
        Stored genes; its axis/order follows the construction inputs.
    evidence : pd.DataFrame | None
        Stored evidence; its axis/order follows the construction inputs.
    genome_build : str | None
        Stored genome build; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    associations : object
        Stored associations supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    genes : pd.DataFrame | None
        Stored genes supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``None``.

    evidence : pd.DataFrame | None
        Stored evidence supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Default: ``None``.

    genome_build : str | None
        Stored genome build supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: ``None``.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.
    """
    associations: object
    genes: pd.DataFrame|None=None
    evidence: pd.DataFrame|None=None
    genome_build: str|None=None
    metadata: dict=field(default_factory=dict)

def _require_hvplot():
    """Import and verify availability of hvplot.
    """
    try:
        import hvplot.xarray  # noqa:F401
        import hvplot.pandas  # noqa:F401
    except ImportError as e:raise ImportError('install VIGOR[plot] for hvPlot/HoloViews figures') from e

def prepare_locus_plot(associations, *, trait=None, metric='neglog_p_additive', chromosome=None,
                       start=None,end=None, genes=None,evidence=None,genome_build=None,annotation_build=None,
                       p_col=None,**kwargs):
    """Prepare one regional association panel from canonical xarray or a small DataFrame.

    The public result model remains xarray.  The DataFrame branch is a plotting
    compatibility adapter for a single trait and is intentionally local to the
    plotting layer; it does not restore the old long-table GWAS result contract.

    Parameters
    ----------
    associations : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    trait : array-like or numeric scalar
        Input supplied to ``xr.Dataset`` as ``coords``. See that callable's documented
        ``coords`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value. Passed to ``xr.Dataset`` as ``coords``.

    metric : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``'neglog_p_additive'``.

    chromosome : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Default: ``None``.

    start : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.
        Default: ``None``. Passed to ``ds.isel`` as ``variant``.

    end : int or None
        Boundary index or genomic coordinate for the requested interval. Use the coordinate
        convention of this reader/query; None leaves that boundary unrestricted where supported.
        Default: ``None``. Passed to ``ds.isel`` as ``variant``.

    genes : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    evidence : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``.

    genome_build : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    annotation_build : object (see the cited input contract)
        Input for the operation ``str(genome_build) != str(annotation_build)``. See
        ``vigor.plotting.locus.prepare_locus_plot`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    p_col : object (see the cited input contract)
        Input for the operation ``KeyError(metric if p_col is None else p_col)``. See
        ``vigor.plotting.locus.prepare_locus_plot`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    **kwargs : mapping
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.

    Returns
    -------
    object
        Computed result from ``LocusPlotData(ds, genes, evidence, genome_build, {'metric': metric, 'trait': str(trait)})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    try: import xarray as xr
    except ImportError as exc: raise ImportError('locus plotting requires VIGOR[io]') from exc
    if genome_build and annotation_build and str(genome_build)!=str(annotation_build):
        raise ValueError('genome-build mismatch between associations and annotation')
    if isinstance(associations,pd.DataFrame):
        frame=associations.reset_index(drop=True).copy()
        chrom_col=next((c for c in ('chromosome','chrom','chr') if c in frame),None)
        pos_col=next((c for c in ('position','pos','bp') if c in frame),None)
        if chrom_col is None or pos_col is None: raise ValueError('DataFrame locus input requires chromosome/chrom and position/pos')
        value_col=str(p_col) if p_col is not None else (metric if metric in frame else None)
        if value_col is None or value_col not in frame: raise KeyError(metric if p_col is None else p_col)
        # p_col is a raw p-value convenience alias used only by this adapter.
        if p_col is not None:
            y=np.asarray(pd.to_numeric(frame[value_col],errors='coerce'),float)
            metric_name='neglog_p'; vals=-np.log10(np.clip(y,np.finfo(float).tiny,1.0))
        else:
            metric_name=str(metric); vals=np.asarray(pd.to_numeric(frame[value_col],errors='coerce'),float)
        vid=(frame['variant_id'].astype(str).to_numpy() if 'variant_id' in frame else
             frame['snp'].astype(str).to_numpy() if 'snp' in frame else np.arange(len(frame)).astype(str))
        ds=xr.Dataset({metric_name:(('variant','trait'),vals[:,None])},
            coords={'variant':np.arange(len(frame)), 'trait':[str(trait or '__single__')],
                    'chromosome':('variant',frame[chrom_col].astype(str).to_numpy()),
                    'position':('variant',pd.to_numeric(frame[pos_col],errors='coerce').to_numpy(np.int64)),
                    'variant_id':('variant',vid)})
        trait=str(trait or '__single__'); metric=metric_name
    elif isinstance(associations,xr.Dataset):
        ds=associations
        if trait is None:
            if 'trait' not in ds.dims or ds.sizes['trait']!=1:
                raise ValueError('trait is required when association Dataset contains multiple traits')
            trait=str(ds.coords['trait'].values[0])
    else:
        raise TypeError('associations must be an xarray.Dataset or a single-trait pandas.DataFrame')
    ds=ds.sel(trait=str(trait))
    if chromosome is not None:
        keep=np.asarray(ds['chromosome'].astype(str).values)==str(chromosome).replace('chr','')
        if not keep.any():keep=np.asarray(ds['chromosome'].astype(str).values)==str(chromosome)
        ds=ds.isel(variant=np.flatnonzero(keep))
    if start is not None: ds=ds.isel(variant=np.flatnonzero(np.asarray(ds['position'])>=int(start)))
    if end is not None: ds=ds.isel(variant=np.flatnonzero(np.asarray(ds['position'])<=int(end)))
    if metric not in ds:raise KeyError(metric)
    return LocusPlotData(ds,genes,evidence,genome_build,{'metric':metric,'trait':str(trait)})

def plot_locus(data:LocusPlotData,*,save_path=None,width=900,height=320):
    """Plot locus.

    Parameters
    ----------
    data : LocusPlotData
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Passed to ``ds[metric].hvplot.scatter`` as ``title``.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``900``. Passed to ``ds[metric].hvplot.scatter`` as
        ``width``.

    height : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``320``. Passed to ``ds[metric].hvplot.scatter`` as
        ``height``.

    Returns
    -------
    object
        Computed result from ``obj``; see the summary and the called implementation for its labeled/numerical representation.
    """
    _require_hvplot();ds=data.associations;metric=data.metadata['metric']
    p=ds[metric].hvplot.scatter(x='position',hover_cols=['variant_id','chromosome'],width=width,height=height,title=f"Regional association: {data.metadata['trait']}")
    plots=[p]
    if data.evidence is not None and len(data.evidence):
        pos='position' if 'position' in data.evidence else data.evidence.columns[0]
        y=next((c for c in ('score','pip','weight') if c in data.evidence),data.evidence.select_dtypes('number').columns[-1])
        plots.append(data.evidence.hvplot.scatter(x=pos,y=y,width=width,height=max(180,height//2),hover_cols='all',title='Evidence'))
    obj=plots[0]
    for q in plots[1:]:obj=obj+q
    if save_path:
        # Panel/Param consult os.getcwd() while resolving theme resources. Tests,
        # notebooks, or workflow managers can legitimately remove a temporary
        # working directory before saving. Anchor the save in the destination
        # directory so an invalid caller CWD cannot break an otherwise absolute
        # output path.
        import os
        from pathlib import Path
        import holoviews as hv
        target=Path(save_path)
        if not target.is_absolute():
            try: target=Path.cwd()/target
            except FileNotFoundError: target=Path.home()/target
        target=target.absolute(); target.parent.mkdir(parents=True,exist_ok=True)
        try: oldcwd=os.getcwd()
        except FileNotFoundError: oldcwd=None
        os.chdir(target.parent)
        try: hv.save(obj,str(target))
        finally:
            if oldcwd is not None and Path(oldcwd).exists(): os.chdir(oldcwd)
    return obj

def make_phewas_figure_single(associations,**kwargs):
    """Construct phewas figure single.

    Parameters
    ----------
    associations : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``prepare_locus_plot``. See those callables
        for accepted keys and values. Passed to ``prepare_locus_plot`` as ``**expanded
        mapping``.

    Returns
    -------
    object
        Computed result from ``plot_locus(prepare_locus_plot(associations, **kwargs), save_path=save_path)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    save_path=kwargs.pop('save_path',None);return plot_locus(prepare_locus_plot(associations,**kwargs),save_path=save_path)
