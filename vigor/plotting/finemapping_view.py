"""Aligned regional fine-mapping tracks and bundled annotated radix networks."""
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd


def _colors(groups):
    """Assign deterministic display colors to the supplied group labels.

    Parameters
    ----------
    groups : sequence, str or None
        Grouping labels used to select a store subgroup or partition observations. For
        resampling, give one cluster label per animal so a cluster is sampled as a unit.
    """
    names=sorted(set(map(str,groups)))
    return {g:'#%02x%02x%02x'%tuple(round(255*c) for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.62,.78)) for i,g in enumerate(names)}


def _save(plot,path):
    """Persist the rendered visualization in the format selected by its file suffix.

    Parameters
    ----------
    plot : object (see the cited input contract)
        Input for the operation ``hv.save(plot, str(path), resources=INLINE)``. See
        ``vigor.plotting.finemapping_view._save`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.
    """
    if path is not None:
        import holoviews as hv
        from bokeh.resources import INLINE
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix.lower() not in ('.html','.htm'):raise ValueError('use an HTML save path')
        hv.save(plot,str(path),resources=INLINE)
    return plot


def _association_frame(associations, variants, traits=None):
    """Normalize a local long table or compact factorGWAS scan without dense LD.

    Parameters
    ----------
    associations : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    variants : pandas.DataFrame or metadata mapping
        Sample/variant metadata aligned to the genotype axes. IDs, chromosome and position
        fields must describe the same fileset; extra annotation columns are retained according
        to the reader. Passed to ``associations.isel`` as ``variant``.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.
    """
    if isinstance(associations,pd.DataFrame):
        out=associations.copy().rename(columns={'snp':'variant','SNP':'variant','bp':'pos','position':'pos'})
        if 'variant' not in out:raise ValueError('associations need variant/snp/SNP')
        if 'trait' not in out:out['trait']='association'
        if 'neglog_p' not in out:
            p='p' if 'p' in out else 'pval'
            out['neglog_p']=-np.log10(np.clip(out[p].to_numpy(float),np.finfo(float).tiny,1.))
    else:
        from ..gwas.exact import chi2_logsf
        id_key='variant_id' if 'variant_id' in associations else 'snp'
        ids=np.asarray(associations[id_key].values,dtype=object).astype(str)
        ds=associations.isel(variant=np.flatnonzero(np.isin(ids,variants.index.astype(str))))
        axis='trait' if 'trait' in ds.coords else 'factor'
        fs=np.asarray(ds[axis].values,dtype=object).astype(str)
        if 'neglog_p' in ds:
            evidence=np.asarray(ds.neglog_p.transpose('variant',axis),float)
        else:
            z=np.asarray(ds['z' if 'z' in ds else 'factor_z'].transpose('variant',axis),float)
            evidence=-chi2_logsf(z*z,1)/np.log(10.)
        out=pd.DataFrame({'variant':np.repeat(ds[id_key].values,len(fs)),
             'trait':np.tile(fs,len(ds[id_key])),'neglog_p':evidence.ravel()})
    out=out.reset_index(drop=True)
    out['variant']=out.variant.astype(str);out['trait']=out.trait.astype(str)
    if traits is not None:out=out.loc[out.trait.isin(list(map(str,traits)))]
    if out.empty:raise ValueError('no association rows in the graph window')
    out=out.drop(columns=['pos'],errors='ignore').merge(variants[['pos','ld_group']],left_on='variant',right_index=True,how='inner',validate='many_to_one')
    return out


def plot_finemapping_locus(graph, associations, *, finemap=None, variant_info=None,
                           traits=None, genes=None, evidence=None, annotations=None,
                           p_threshold=None, width=1100, association_height=230,
                           datashade=True, dynamic=True, title=None, save_path=None):
    """Aligned locus panels: associations, joint CS/PIP, LD groups, triangle, genes.

    All panels use physical Mb and the same LD-group colors. Datashader shades
    dense background points/LD pairs; top associations and credible-set variants
    remain vector glyphs with hover. CSs are taken from the supplied joint fit,
    never constructed from p-values. Evidence must be supplied as a local table
    with variant, track, and neglog_p or score. No external evidence is inferred.
    Optional annotations are variant-indexed impact/consequence/effect records.
    The LD triangle displays only computed retained pairs, labelled by scope.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata. Passed to ``ld.opts`` as
        ``title``.

    associations : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    finemap : mapping
        Input for the operation ``finemap.fits.get('bayes', {})``. See
        ``vigor.plotting.finemapping_view.plot_finemapping_locus`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    variant_info : object (see the cited input contract)
        Input for the operation ``variant_info.copy()``. See
        ``vigor.plotting.finemapping_view.plot_finemapping_locus`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    traits : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    genes : pandas.DataFrame or annotation bundle
        Gene/annotation records with the identifiers and coordinate columns read below. Genomic
        coordinates and assembly must agree with the queried variant region; orthology tables
        map source_gene to target_gene. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    evidence : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source. Default: ``None``.

    annotations : object (see the cited input contract)
        Input for the operation ``v.join(annotations, rsuffix='_annotation')``. See
        ``vigor.plotting.finemapping_view.plot_finemapping_locus`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    p_threshold : float
        P-value threshold on [0, 1] for exact rescoring or region retention. Smaller values
        focus work on more significant candidates; larger values broaden the candidate set.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``1100``. Passed to
        ``v.reset_index().hvplot.scatter`` as ``width``.

    association_height : int
        Input for the operation ``dict(x='x_mb', y='neglog_p', height=association_height,
        xlim=limits, width=width)``. See
        ``vigor.plotting.finemapping_view.plot_finemapping_locus`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``230``.

    datashade : bool
        Input supplied to ``triangle.hvplot.scatter`` as ``rasterize``. See that callable's
        documented ``rasterize`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``True``. Passed to ``triangle.hvplot.scatter`` as
        ``rasterize``.

    dynamic : bool
        Input supplied to ``a.hvplot.scatter`` as ``dynamic``. See that callable's documented
        ``dynamic`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``True``. Passed to ``a.hvplot.scatter`` as ``dynamic``.

    title : str
        Displayed title, status or descriptive text attached to the visualization/progress
        output. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value. Passed to ``p.opts`` as ``title``.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``_save(hv.Layout(panels).cols(1).opts(shared_axes=True), save_path)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import hvplot.pandas  # noqa
    import holoviews as hv
    import datashader as ds
    from ..finemap.graph import haplotype_graph_variants, _genes
    hv.extension('bokeh')
    v=haplotype_graph_variants(graph) if variant_info is None else variant_info.copy()
    v.index=v.index.astype(str);v.index.name='variant'
    v['x_mb']=v.pos/1e6
    for field in ['genes','founders','nesting_layers']:
        if field in v:
            v[field]=v[field].map(lambda x: ', '.join(map(str,x[:12]))+(' …' if len(x)>12 else '') if isinstance(x,(tuple,list,np.ndarray)) else str(x))
    palette=_colors(v.ld_group)
    v['color']=v.ld_group.map(palette)
    if annotations is not None:
        if not annotations.index.is_unique:raise ValueError('annotation variant index must be unique')
        v=v.join(annotations,rsuffix='_annotation')
    association=_association_frame(associations,v,traits)
    hover_columns=[c for c in ['x_mb','color','genes','founders','nesting_layers','hdbscan_confidence','impact','consequence'] if c in v]
    association=association.merge(v[hover_columns].reset_index(),on='variant',how='left',validate='many_to_one')
    if association.empty:raise ValueError('no association rows in the graph window')
    limits=(graph.graph['start']/1e6,graph.graph['end']/1e6)
    panels=[]
    common=dict(width=width,xlim=limits,toolbar='above',show_grid=False,fontsize={'title':12,'labels':11,'ticks':9})
    lead=graph.graph.get('lead_variant')
    # Overlay a modest vector layer for useful hover, even over datashaded scans.
    for trait,a in association.groupby('trait',sort=False):
        opts=dict(x='x_mb',y='neglog_p',height=association_height,xlim=limits,width=width)
        if datashade:
            background=a.hvplot.scatter(**opts,datashade=True,dynamic=dynamic,cmap=['#c2cad3','#465970'])
        else:background=a.hvplot.scatter(**opts,c='color',size=5,alpha=.6)
        hover=a.nlargest(min(1800,len(a)),'neglog_p')
        point=hover.hvplot.scatter(**opts,c='color',size=7,line_color='#333333',line_width=.25,
            hover_cols=[c for c in ['variant','trait','ld_group','hdbscan_confidence','genes','founders','nesting_layers','neglog_p','impact','consequence'] if c in hover])
        p=background*point
        if p_threshold is not None:p=p*hv.HLine(-np.log10(p_threshold)).opts(color='#c94747',line_dash='dashed',line_width=1)
        # Always distinguish the largest association in this trait panel.
        top=a.nlargest(1,'neglog_p')
        p=p*top.hvplot.scatter(x='x_mb',y='neglog_p',marker='star',size=19,c='color',line_color='black',hover_cols=['variant','trait','neglog_p'])
        panels.append(p.opts(**common,title=(title+' — ' if title else '')+trait,ylabel='−log₁₀(p)',xlabel='',xaxis=None))
    bayes=finemap.fits.get('bayes',{}) if finemap is not None else {}
    if bayes:
        pip=bayes['pip'].rename(columns={'snp':'variant'}).merge(v.reset_index(),on='variant',how='inner')
        panels.append(pip.hvplot.scatter(x='x_mb',y='pip',c='color',size=5+20*pip['pip'],
            hover_cols=['variant','pip','genes','ld_group'],height=150).opts(**common,title='Joint inclusion probabilities',ylabel='Joint PIP',xaxis=None,xlabel=''))
        cs=bayes.get('credible_sets',pd.DataFrame())
        if len(cs):
            cs=cs.rename(columns={'snp':'variant'}).merge(v.reset_index(),on='variant',how='inner')
            cs['CS']=cs.effect.map(lambda x:f'Joint CS {x}')
            cs['point_size']=6+24*np.sqrt(cs.alpha)
            panels.append(cs.hvplot.scatter(x='x_mb',y='CS',c='color',size='point_size',line_color='black',line_width=.7,
                hover_cols=['variant','alpha','coverage','cs_size','cs_min_r2','genes','founders'],height=max(100,min(240,40*cs.effect.nunique()))).opts(**common,title='Credible sets across selected responses',ylabel='CS',xaxis=None,xlabel=''))
    # LD group runs, plus original sequence transitions summarized to LD lanes.
    groups=sorted(v.ld_group.unique(),key=lambda g:v.loc[v.ld_group.eq(g),'pos'].min())
    lane={g:i for i,g in enumerate(groups)}
    v['lane']=v.ld_group.map(lane);v['point_size']=4+7*v.hdbscan_confidence.fillna(0.)
    ldpoints=v.reset_index().hvplot.scatter(x='x_mb',y='lane',c='color',size='point_size',
        hover_cols=['variant','ld_group','hdbscan_cluster','hdbscan_status','hdbscan_confidence','founders','genes','nesting_layers'],
        width=width,height=max(180,min(480,22*len(groups))))
    runs=[];left=0;ordered=v.sort_values('pos')
    vals=ordered.ld_group.to_numpy();x=ordered.x_mb.to_numpy()
    for right in range(1,len(ordered)+1):
        if right==len(ordered) or vals[right]!=vals[left]:
            runs.append([(x[left],lane[vals[left]]),(x[right-1],lane[vals[left]])]);left=right
    ldpoints=hv.Path(runs,kdims=['x_mb','lane']).opts(color='#4d5358',line_width=1)*ldpoints
    # Show only genuine retained sequence-adjacency transitions; never optimize new links from missing LD.
    links=[];seen=set()
    for s,t,a in sorted(graph.edges(data=True),key=lambda e:e[2]['n_samples'],reverse=True):
        u,w=graph.nodes[s],graph.nodes[t]
        if not u['variants'] or not w['variants']:continue
        va,vb=u['variants'][-1],w['variants'][0]
        ga,gb=v.loc[va,'ld_group'],v.loc[vb,'ld_group']
        key=(va,vb)
        if ga!=gb and key not in seen:
            links.append([(v.loc[va,'x_mb'],lane[ga]),(v.loc[vb,'x_mb'],lane[gb])]);seen.add(key)
        if len(links)>=250:break
    if links:ldpoints=hv.Path(links,kdims=['x_mb','lane']).opts(color='#222222',line_width=.7,alpha=.35)*ldpoints
    panels.append(ldpoints.opts(**common,title='LD / HDBSCAN groups and observed sequence adjacency',ylabel='LD group',xaxis=None,xlabel='',yticks=[(i,g) for g,i in lane.items()]))
    pairs=graph.graph['ld_edges']
    if len(pairs):
        pos=graph.graph['variants'].pos.to_numpy(float)/1e6
        xi=pos[pairs.i.to_numpy(int)];xj=pos[pairs.j.to_numpy(int)]
        triangle=pd.DataFrame({'x_mb':(xi+xj)/2,'distance_mb':-abs(xj-xi)/2,'r2':pairs.r2.to_numpy()})
        ld=triangle.hvplot.scatter(x='x_mb',y='distance_mb',c='r2',cmap='magma',clim=(0,1),
            rasterize=datashade and len(pairs)>10_000,dynamic=dynamic,
            aggregator=ds.max('r2') if datashade and len(pairs)>10_000 else None,
            size=7,colorbar=True,
            width=width,height=220)
        panels.append(ld.opts(**common,title=f"LD triangle — retained pairs ({graph.graph['ld_scope']})",ylabel='LD distance (Mb)',xaxis=None,xlabel=''))
    if evidence is not None and len(evidence):
        e=evidence.copy().rename(columns={'snp':'variant'})
        if not {'variant','track'}.issubset(e):raise ValueError('evidence requires variant and track')
        value='neglog_p' if 'neglog_p' in e else 'score'
        e=e.merge(v[['x_mb']],left_on='variant',right_index=True,how='inner')
        for track,t in e.groupby('track',sort=False):
            t=t.copy();t['evidence_score']=t[value]
            t['point_size']=6+14*np.sqrt(np.maximum(t[value],0)/max(t[value].max(),1.))
            panels.append(t.hvplot.scatter(x='x_mb',y='track',c='evidence_score',size='point_size',cmap='Blues',clim=(0,float(t[value].max())),colorbar=True,
                hover_cols=list(t.columns),width=width,height=95).opts(**common,title=str(track),ylabel='',xaxis=None,xlabel=''))
    if genes is None and graph.graph.get('gtf_path'):
        records=_genes(graph.graph['gtf_path'],graph.graph['chrom'],graph.graph['start'],graph.graph['end'],graph.graph.get('contig_aliases'))
        genes=pd.DataFrame(records,columns=['start','end','symbol'])
    if genes is not None and len(genes):
        g=genes.copy().sort_values('start');ends=[];rows=[]
        for r in g.itertuples():
            gap=(limits[1]-limits[0])*1e6*.035
            use=next((i for i,x in enumerate(ends) if x+gap<r.start),len(ends))
            if use==len(ends):ends.append(r.end)
            else:ends[use]=r.end
            rows.append(dict(x_mb=r.start/1e6,x_end_mb=r.end/1e6,y0=-use,y1=-use,symbol=r.symbol,label_x=(r.start+r.end)/2e6,label_y=-use-.2))
        g=pd.DataFrame(rows)
        body=hv.Segments(g,kdims=['x_mb','y0','x_end_mb','y1'],vdims=['symbol']).opts(color='#244f69',line_width=5,tools=['hover'])
        labels=hv.Labels(g,kdims=[hv.Dimension('label_x',label='x_mb'),'label_y'],vdims=['symbol']).opts(text_font_size='9pt',text_color='#244f69')
        panels.append((body*labels).opts(**common,height=max(120,min(300,38*len(ends)+50)),ylabel='Genes',yaxis=None,title='Gene intervals',xlabel=f"Chromosome {graph.graph['chrom']} position (Mb)"))
    else:
        panels[-1]=panels[-1].opts(xaxis='bottom',xlabel=f"Chromosome {graph.graph['chrom']} position (Mb)")
    return _save(hv.Layout(panels).cols(1).opts(shared_axes=True),save_path)


def plot_haplotype_network(graph, *, node_ids=None, max_nodes=700, width=850,height=800,
                            bundle=True, datashade_edges=False, seed=42, k=1,
                            layout='spring', position_anchor='first', layout_depth=4,
                            labels=False, save_path=None):
    """Bundled spring-layout HoloViews Graph, with bounded, annotated node hover.

    Small graphs display every node. For larger graphs, an explicitly labelled
    representative-path view is selected: paths with greatest mean node support
    are added while whole paths fit the node budget. No new edges are inferred.
    Pass node_ids to choose a local branch/node subgraph. The full graph stays
    intact. This bounded display avoids running spring_layout on 200,000 nodes.
    Bundled curves are visual routing, not biological recombination estimates.
    Only bundle=True needs the optional scikit-image dependency (vigor[bundle]).
    layout='genomic' fixes x to physical SNP position (Mb); y uses deterministic
    nested radix display lanes, not ancestry/effect. Multi-SNP nodes use the
    first SNP by default; position_anchor='midpoint'/'last' uses the first/last
    SNP midpoint or last SNP. Hover retains the full node interval. Reference
    root nodes without SNPs use the window start. No spring solver is run in
    genomic mode. Bundling may curve edges but never moves the node positions.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        Graph whose nodes and edges define the network/path relationships processed by this
        operation. Node identifiers must match the supplied metadata. Passed to
        ``view.add_node`` as ``size``.

    node_ids : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering. Default: ``None``.
        None delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    max_nodes : int
        Maximum allowed graph-node count, at least two; exceeding the guard stops graph
        materialization. Default: ``700``. Source comparisons: ``max_nodes < 2``. See the
        rejection/fallback branches for enforcement.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``850``. Passed to ``datashade`` as ``width``.

    height : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``800``. Passed to ``datashade`` as ``height``.

    bundle : mapping or feature/annotation bundle
        Prepared feature/annotation collection indexed by its named groups. Reuse the same
        fitted preprocessing, animal/variant order and metadata when transforming or predicting.
        Default: ``True``.

    datashade_edges : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``42``. Passed to
        ``nx.spring_layout`` as ``seed``.

    k : int or index array
        Local position, index selection or step used by the stated slice/iteration. Indices must
        refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
        the enclosing dimensions. Default: ``1``. Passed to ``nx.spring_layout`` as ``k``.

    layout : str or sequence
        LD array layout or packed optimizer-coordinate slices. LD spellings 'dense'/'matrix' use a
        square matrix; 'offset'/'band'/'window'/'rolling' use banded offsets where supported.
        Optimization layouts must come from the same parameter-layout helper. Default:
        ``'spring'``. Source-recognized literal values: ``genomic``. These select the
        corresponding branches below; comparison evidence alone does not establish an exhaustive
        enum.

    position_anchor : str
        Input for the operation ``position_anchor not in ('first', 'midpoint', 'last')``. See
        ``vigor.plotting.finemapping_view.plot_haplotype_network`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``'first'``.

    layout_depth : int
        Input supplied to ``prepare_haplotype_graph_plot`` as ``layout_depth``. See that
        callable's documented ``layout_depth`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``4``. Passed to
        ``prepare_haplotype_graph_plot`` as ``layout_depth``.

    labels : str, index or sequence
        Axis/column/feature selector used to identify the input/output quantities in this
        operation. Named selectors must exist in the labeled input; integer selectors are valid
        zero-based positions on that axis. Default: ``False``.

    save_path : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``_save(result, save_path)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import networkx as nx
    import holoviews as hv
    from holoviews.operation.datashader import bundle_graph, datashade
    hv.extension('bokeh')
    if layout not in ('spring','genomic'):raise ValueError('layout must be spring or genomic')
    if position_anchor not in ('first','midpoint','last'):raise ValueError('position_anchor must be first, midpoint or last')
    if max_nodes<2:raise ValueError('max_nodes must be at least two')
    if node_ids is not None:selected=set(node_ids)
    elif graph.number_of_nodes()<=max_nodes:selected=set(graph)
    else:
        selected=set();paths=graph.graph['paths']
        scores=[(float(np.mean([graph.nodes[n]['n_samples'] for n in row.nodes])),i) for i,row in enumerate(paths.itertuples()) if not row.is_founder]
        for _,i in sorted(scores,reverse=True):
            path=set(paths.iloc[i].nodes)
            if len(selected|path)<=max_nodes:selected|=path
            if len(selected)>=max_nodes*.95:break
        if not selected:raise ValueError('one path exceeds max_nodes; select a narrower node_ids view or increase the display budget')
    if len(selected)>max_nodes:raise ValueError('selected graph exceeds max_nodes')
    if not selected.issubset(graph):raise ValueError('unknown selected node')
    from ..finemap.graph import haplotype_graph_variants
    palette=_colors(haplotype_graph_variants(graph).ld_group)
    view=nx.DiGraph()
    for n in sorted(selected):
        a=graph.nodes[n];group=a['ld_groups'][0] if a['ld_groups'] else 'root'
        def preview(values,limit=12):
            """Format a bounded preview of labels for the fine-mapping report.

            Parameters
            ----------
            values : array-like, table or labeled object
                Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
                described in the summary; the local coercion/selection expression below defines the
                accepted representation.

            limit : int or None
                Count limit for the corresponding sampled/retained/displayed items. Positive values
                bound work/output size; None uses all eligible items or the local default where
                supported. Default: ``12``.

            Returns
            -------
            object
                Computed result from ``', '.join(map(str, values[:limit])) + (' …' if len(values) > limit else '')``; see the summary and the called implementation for its labeled/numerical representation.
            """
            return ', '.join(map(str,values[:limit]))+(' …' if len(values)>limit else '')
        view.add_node(n,size=7+23*np.sqrt(a['n_samples']/max(1,graph.graph['n_samples'])),color=palette.get(group,'#656c76'),
            label=str(n),n_samples=a['n_samples'],n_founders=a['n_founders'],position_start_bp=a['start'],position_end_bp=a['end'],
            genes=preview(a['genes']),founders=preview(a['founders']),animals=preview(a['animals']),
            variants=preview(a['variants']),layers=preview(a['nesting_layers']),ld_groups=preview(a['ld_groups']),
            bubble_ids=preview(a.get('bubble_ids',[])),bubble_labels=preview(a.get('bubble_labels',[])),
            sequence=(a['sequence'] or '')[:120],sequence_status=a['sequence_status'])
    for s,t,a in graph.edges(data=True):
        if s in selected and t in selected:view.add_edge(s,t,n_samples=a['n_samples'],n_paths=a['n_paths'],edge_type=a['edge_type'])
    if layout=='genomic':
        from .haplotype_graph import prepare_haplotype_graph_plot
        lanes=prepare_haplotype_graph_plot(graph.subgraph(selected),layout_depth=layout_depth,include_ld=False).nodes.lane
        variant_positions=graph.graph['variants'].pos
        positions={}
        for node in view:
            variants=graph.nodes[node]['variants']
            if variants:
                first,last=variant_positions.loc[variants[0]],variant_positions.loc[variants[-1]]
                x=first if position_anchor=='first' else last if position_anchor=='last' else (int(first)+int(last))/2
            else:x=graph.graph['start']
            # Comparable coordinate scales keep HoloViews arrowhead geometry well behaved.
            positions[node]=(float(x)/1e6,float(lanes.loc[node])*max((graph.graph['end']-graph.graph['start'])/1e6,1e-6))
    else:positions=nx.spring_layout(view,k=k,seed=seed,iterations=80)
    hvg=hv.Graph.from_networkx(view,positions)
    if bundle and view.number_of_edges():
        try:import skimage  # noqa: F401; required by Datashader hammer_bundle
        except ImportError as exc:
            raise ImportError('Edge bundling requires scikit-image; install vigor[bundle] or use bundle=False') from exc
        hvg=bundle_graph(hvg)
    title=f"Annotated radix network — {len(view):,}/{graph.number_of_nodes():,} nodes"
    if len(view)<graph.number_of_nodes():title+=' (representative paths)'
    if layout=='genomic':title+=f' — genomic, {position_anchor} SNP anchor'
    opts=dict(width=width,height=height,xaxis=None,yaxis=None,shared_axes=False,bgcolor='white',title=title,
        node_size='size',node_color='color',node_line_color='black',edge_line_width=1,edge_alpha=.25,
        tools=['hover','tap'],inspection_policy='nodes',directed=True,arrowhead_length=.01)
    if layout=='genomic':
        span=max((graph.graph['end']-graph.graph['start'])/1e6,1e-6)
        opts.update(xaxis='bottom',yaxis='left',
            xlabel=f"Chromosome {graph.graph['chrom']} position (Mb)",
            ylabel='Radix branch lane (display)',
            xlim=(graph.graph['start']/1e6,graph.graph['end']/1e6),ylim=(-.03*span,1.03*span),yticks=[(f*span,str(f)) for f in (0,.25,.5,.75,1)])
    hvg=hvg.opts(**opts)
    if datashade_edges:
        # Preserve hoverable nodes; rasterized edges have no individual hover/arrows.
        result=datashade(hvg,width=width,height=height,dynamic=False,cmap=['#a5adb6','#33475c'])*hvg.nodes.opts(
            size='size',color='color',line_color='black',tools=['hover'],width=width,height=height,xaxis=None,yaxis=None)
    else:result=hvg
    if datashade_edges and layout=='genomic':
        result=result.opts(xaxis='bottom',yaxis='left',xlabel=opts['xlabel'],ylabel=opts['ylabel'],xlim=opts['xlim'],ylim=opts['ylim'],yticks=opts['yticks'],title=title)
    if labels and len(view)<=80:result=result*hv.Labels(hvg.nodes,['x','y'],'index').opts(text_font_size='8pt')
    return _save(result,save_path)
