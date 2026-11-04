"""hvPlot/HoloViews visualizations for covariance-factor stability."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _hv():
    """Import HoloViews and initialize the optional plotting backend.
        
    """
    try:
        import hvplot.pandas  # noqa: F401
        import holoviews as hv
        return hv
    except ImportError as exc:
        raise ImportError("factor plots require VIGOR[plot] (hvplot/holoviews/bokeh)") from exc

def _stability_table(stability, component=None):
    """Return the stability table result computed by ``pd.DataFrame``.

    Parameters
    ----------
    stability : mapping, array-like or None
        Resampling support/factor-stability estimates aligned to the displayed or selected
        nodes/components. None omits stability annotation where supported.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.
    """
    if component is not None and component in stability: stability=stability[component]
    mean=np.asarray(stability['loading_mean']); se=np.asarray(stability['loading_se'])
    rows=[]
    for j in range(mean.shape[1]):
        for i in range(mean.shape[0]): rows.append((i,f'FA{j+1}',mean[i,j],se[i,j]))
    return pd.DataFrame(rows,columns=['trait_index','factor','loading','se'])

def plot_factor_stability(stability, factor=None, *, component=None, trait_names=None):
    """Interactive loading estimate/SE plot for one component or factor.

    Parameters
    ----------
    stability : mapping, array-like or None
        Resampling support/factor-stability estimates aligned to the displayed or selected
        nodes/components. None omits stability annotation where supported.

    factor : array-like or fitted basis
        Trait/feature × retained-factor loading matrix or prepared basis. Column order
        identifies factor directions and must agree with the scores used for projection.
        Default: ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``d.hvplot.scatter(x='trait', y='loading', by='factor', hover_cols=['se'], rot=90, height=420, width=900, title='Factor loading stability')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    _hv(); d=_stability_table(stability,component)
    if trait_names is not None:
        names=list(map(str,trait_names)); d['trait']=d.trait_index.map(lambda i:names[int(i)])
    else:d['trait']=d.trait_index.astype(str)
    if factor is not None:
        f=f'FA{int(factor)}' if isinstance(factor,(int,np.integer)) else str(factor); d=d[d.factor==f]
    return d.hvplot.scatter(x='trait',y='loading',by='factor',hover_cols=['se'],rot=90,height=420,width=900,title='Factor loading stability')

def plot_factor_congruence(stability, *, component=None):
    """Interactive bootstrap congruence distribution for aligned factors.

    Parameters
    ----------
    stability : mapping, array-like or None
        Resampling support/factor-stability estimates aligned to the displayed or selected
        nodes/components. None omits stability annotation where supported.

    component : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``d.hvplot.line(x='replicate', y='congruence', by='factor', height=350, width=850, ylim=(0, 1.02), title='Bootstrap factor congruence')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    _hv(); st=stability[component] if component is not None and component in stability else stability
    cc=np.asarray(st.get('congruence',[]),float)
    if cc.ndim!=2:return pd.DataFrame(columns=['replicate','factor','congruence']).hvplot.scatter(x='replicate',y='congruence')
    d=pd.DataFrame(cc,columns=[f'FA{i+1}' for i in range(cc.shape[1])]).reset_index(names='replicate').melt('replicate',var_name='factor',value_name='congruence')
    return d.hvplot.line(x='replicate',y='congruence',by='factor',height=350,width=850,ylim=(0,1.02),title='Bootstrap factor congruence')

__all__=['plot_factor_stability','plot_factor_congruence']
