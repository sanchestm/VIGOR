"""Association-to-locus construction from canonical xarray GWAS datasets."""
from __future__ import annotations
import numpy as np, pandas as pd


def canonical_variant_id(chrom,pos,ref=None,alt=None):
    """Construct a consistent chromosome-position-allele variant identifier.

    Parameters
    ----------
    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    pos : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    ref : object (see the cited input contract)
        Input for the operation ``ref is None``. See
        ``vigor.integration.loci.canonical_variant_id`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    alt : object (see the cited input contract)
        Input for the operation ``alt is None``. See
        ``vigor.integration.loci.canonical_variant_id`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``.

    Returns
    -------
    object
        Computed result from ``base if ref is None or alt is None else f'{base}:{ref}:{alt}'``; see the summary and the called implementation for its labeled/numerical representation.
    """
    c=str(chrom).lower().replace('chr','');base=f'{c}:{int(pos)}';return base if ref is None or alt is None else f'{base}:{ref}:{alt}'


def build_loci(associations, *, metric=None, threshold=None, traits=None,
               window_bp=1_000_000, ld_r2=None, ld_threshold=.6, merge_traits=True,
               genome_build=None):
    """Build proximity/LD loci from a canonical ``variant × trait`` GWAS Dataset.

    Parameters
    ----------
    associations : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    metric : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default None chooses canonical neglog_p, then legacy neglog_p_additive.

    threshold : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value. Passed to ``ds.sel`` as
        ``trait``.

    window_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.

    ld_r2 : array-like or None
        Variant × variant linkage-disequilibrium matrix in the genotype-column order. Use
        correlation or squared correlation according to ld_kind/the called fitter; None requests
        computation where supported. Default: ``None``.

    ld_threshold : float
        LD/correlation cutoff used in the named grouping/pruning/link stage. Use [0, 1] for
        squared correlation; the selected LD statistic determines whether sign is retained.
        Default: ``0.6``.

    merge_traits : bool or str
        Annotation/report combination policy read by this routine. Boolean settings enable the
        named validation/formatting action; aggregation names select the explicit
        table-combination branch below. Default: ``True``.

    genome_build : str or mapping
        Format/build/provenance identifier attached to this artifact. Preserve the serialized
        schema/version names used by readers; these identify storage contracts rather than
        numerical options. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(loci)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    try: import xarray as xr
    except ImportError as exc: raise ImportError('build_loci requires VIGOR[io]') from exc
    if not isinstance(associations,xr.Dataset): raise TypeError('associations must be an xarray.Dataset')
    if 'variant' not in associations.dims or 'trait' not in associations.dims: raise ValueError('GWAS Dataset requires variant and trait dimensions')
    ds=associations
    if metric is None: metric='neglog_p' if 'neglog_p' in ds else 'neglog_p_additive'
    if metric not in ds:raise KeyError(metric)
    if traits is not None:
        if 'trait' not in ds.xindexes:ds=ds.set_xindex('trait')
        ds=ds.sel(trait=list(map(str,traits)))
    from ..resources import bounded_variant_chunk
    # Construct only retained hits, never the full variant × trait long table.
    chunk=bounded_variant_chunk(ds.sizes['trait'],bytes_per_call=16,max_chunk=65536)
    trait_labels=np.asarray(ds.trait.values,dtype=object).astype(str)
    frames=[]
    for start in range(0,ds.sizes['variant'],chunk):
        block=ds.isel(variant=slice(start,start+chunk))
        values=np.asarray(block[metric].transpose('variant','trait').values,float)
        keep=np.isfinite(values) if threshold is None else np.isfinite(values)&(values>=float(threshold))
        i,j=np.nonzero(keep)
        if len(i)==0:continue
        data={'variant':np.asarray(block.variant.values)[i],'trait':trait_labels[j],metric:values[i,j]}
        for c in ('variant_id','chromosome','position'):
            if c in block.coords:data[c]=np.asarray(block[c].values)[i]
        frames.append(pd.DataFrame(data))
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:return pd.DataFrame(columns=['locus_id','chrom','start','end','n_variants','variants','traits','genome_build'])
    if 'variant_id' not in frame: frame['variant_id']=frame['variant'].astype(str)
    if 'chromosome' not in frame or 'position' not in frame: raise ValueError('GWAS Dataset needs chromosome and position coordinates')
    a=frame.rename(columns={'chromosome':'_chrom','position':'_pos'}).copy()
    a['_chrom']=a['_chrom'].astype(str).str.replace('chr','',regex=False).str.lower()
    a['_pos']=pd.to_numeric(a['_pos'],errors='raise').astype(np.int64)
    a['variant_id']=a['variant_id'].astype(str)
    comp=np.empty(len(a),object)
    for chrom,ix in a.groupby('_chrom').groups.items():
        order=sorted(list(ix),key=lambda i:(a.at[i,'_pos'],a.at[i,'variant_id'])); c=0; last=None
        for i in order:
            pos=int(a.at[i,'_pos'])
            if last is None or pos-last>int(window_bp): c+=1
            comp[a.index.get_loc(i)]=f'{chrom}:{c}'; last=pos
    a['_component']=comp
    if ld_r2 is not None:
        comps=a['_component'].unique().tolist(); parent={c:c for c in comps}
        def root(x):
            """Find the union-find representative of the requested locus/group.

            Parameters
            ----------
            x : array-like
                Numerical operand in ``parent[x]``. Shape must conform to that matrix/vector operation
                and the enclosing prepared state.

            Returns
            -------
            object
                Computed result from ``x``; see the summary and the called implementation for its labeled/numerical representation.
            """
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(x,y):
            """Merge two locus/group components in the enclosing union-find structure.

            Parameters
            ----------
            x : array-like
                Numerical operand in ``root(x)``. Shape must conform to that matrix/vector operation and
                the enclosing prepared state.

            y : array-like
                Numerical operand in ``root(y)``. Shape must conform to that matrix/vector operation and
                the enclosing prepared state.
            """
            rx,ry=root(x),root(y)
            if rx!=ry: parent[max(rx,ry)]=min(rx,ry)
        v2c=dict(zip(a['variant_id'],a['_component']));v2chr=dict(zip(a['variant_id'],a['_chrom']))
        if isinstance(ld_r2,pd.DataFrame):
            common=[v for v in v2c if v in ld_r2.index and v in ld_r2.columns]
            for i,x in enumerate(common):
                for y,r2 in ld_r2.loc[x,common[i+1:]].items():
                    if v2chr[x]==v2chr[str(y)] and np.isfinite(r2) and float(r2)>=float(ld_threshold):union(v2c[x],v2c[str(y)])
        a['_component']=a['_component'].map(root)
    keys=['_chrom','_component'] if merge_traits else ['_chrom','trait','_component']
    loci=[]
    for _,g in a.groupby(keys,sort=True):
        loci.append({'locus_id':f'L{len(loci)+1:05d}','chrom':g['_chrom'].iloc[0],'start':int(g['_pos'].min()),'end':int(g['_pos'].max()),
                     'n_variants':int(g['variant_id'].nunique()),'variants':tuple(g['variant_id'].unique()),
                     'traits':tuple(g['trait'].dropna().astype(str).unique()),'genome_build':genome_build})
    return pd.DataFrame(loci)
