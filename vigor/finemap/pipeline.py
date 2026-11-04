"""Stable fine-mapping wrappers with labelled joint-trait selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .result import FineMapResult


def _frame(value, kind):
    """Normalize a bounded scientific table to a uniquely labelled DataFrame.
    
    Parameters
    ----------
    value : DataFrame, 2D DataArray or array-like
        Animal-by-feature input (n,m). A 1D array is promoted to (n,1) only for
        kind='traits'. DataArrays need an iid/sample/animal dimension; the other
        dimension supplies columns. Values are materialized and labels stringified.
    kind : str
        Diagnostic input name, normally 'traits' or 'genotypes'; only 'traits'
        permits 1D arrays. No closed enum is otherwise enforced.
    
    Returns
    -------
    tuple
        (frame, labelled), with frame shape (n,m) and labelled=True for pandas or
        xarray input, False for positional arrays. Duplicate row/column labels fail.
    
    Complexity: O(nm) TIME and O(nm+n+m) SPACE for n animals and m features.
    """
    if isinstance(value,pd.DataFrame):
        frame=value.copy()
        labelled=True
    elif hasattr(value,'dims'):
        import xarray as xr
        if not isinstance(value,xr.DataArray) or value.ndim!=2:
            raise TypeError(f'{kind} must be a two-dimensional DataArray')
        sd=next((d for d in ('iid','sample','animal') if d in value.dims),None)
        if sd is None:raise ValueError(f'{kind} xarray needs an iid/sample/animal dimension')
        vd=next(d for d in value.dims if d!=sd)
        ordered=value.transpose(sd,vd)
        frame=pd.DataFrame(ordered.values,index=ordered[sd].values,columns=ordered[vd].values)
        labelled=True
    else:
        array=np.asarray(value)
        if array.ndim==1 and kind=='traits':array=array[:,None]
        if array.ndim!=2:raise ValueError(f'{kind} must be two-dimensional')
        frame=pd.DataFrame(array);labelled=False
    frame.index=frame.index.map(str);frame.columns=frame.columns.map(str)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f'{kind} row and column labels must be unique')
    return frame,labelled


def _select_covariance(value, names, selected, name):
    """Select a response covariance in the requested labelled trait order.
    
    Parameters
    ----------
    value : DataFrame, 2D DataArray, ndarray or None
        Full covariance (t,t), selected covariance (s,s), or None. Labelled
        matrices must cover every selected trait on both axes. Arrays are
        interpreted in full names order when shape is (t,t), otherwise selected
        order when shape is (s,s). Equal-sized arrays use full names order.
    names : sequence of str, length t
        Complete original response-column order, used for positional matrices.
    selected : sequence of str, length s
        Requested distinct response labels in output order, validated by caller.
    name : str
        Diagnostic covariance label, normally 'G' or 'E'; affects error messages.
    
    Returns
    -------
    ndarray or None
        Float covariance (s,s) in selected order, or None if value is None.
        Missing labels/invalid shapes raise ValueError.
    
    Complexity: O(t²+s²) TIME and SPACE; labelled matrices are copied before
    subsetting. None returns in O(1). t is all traits and s selected traits.
    """
    if value is None:return None
    if isinstance(value,pd.DataFrame):
        matrix=value.copy();matrix.index=matrix.index.map(str);matrix.columns=matrix.columns.map(str)
        if not matrix.index.is_unique or not matrix.columns.is_unique:
            raise ValueError(f'{name} covariance labels must be unique')
        if not pd.Index(selected).isin(matrix.index).all() or not pd.Index(selected).isin(matrix.columns).all():
            raise ValueError(f'{name} covariance labels do not cover selected traits')
        return matrix.loc[selected,selected].to_numpy(float)
    if hasattr(value,'dims'):
        if value.ndim!=2:raise ValueError(f'{name} covariance must have two dimensions')
        if all(d in value.coords for d in value.dims):
            matrix=pd.DataFrame(value.values,index=value[value.dims[0]].values,
                                columns=value[value.dims[1]].values)
            return _select_covariance(matrix,names,selected,name)
    matrix=np.asarray(value,float)
    if matrix.shape==(len(names),len(names)):
        ix=pd.Index(names).get_indexer(selected)
        return matrix[np.ix_(ix,ix)]
    if matrix.shape==(len(selected),len(selected)):return matrix
    raise ValueError(f'{name} covariance shape must match full or selected traits')


def fine_map_region(genotypes,traits,*,trait_subset=None,**kwargs):
    """Align labelled samples and jointly fine-map selected traits in a bounded locus.
    
    Parameters
    ----------
    genotypes : DataFrame, 2D DataArray or array, shape (n, m)
        Dosage matrix with samples in rows and locus variants in columns. Xarray uses an iid/sample/animal dimension and is materialized. Labels must be unique; arrays align positionally.
    
    traits : DataFrame, 2D DataArray or array, shape (n, t)
        Responses; a 1D array is promoted to one column. Labelled rows are reordered to genotype rows, which must all be covered. Fitters retain complete selected-response rows; heldout values stay NaN.
    
    trait_subset : str, sequence or None
        Default: ``None``. Select all response columns, or explicit distinct column names (converted to strings), preserving order. This is label selection, not integer positional indexing.
    
    **kwargs : dict
        Forwarded to multivariate.fine_map_region. methods may select bayes/ridge/elasticnet/sparse_group/lightgbm, with mvsusie a bayes alias; L controls single effects. G/E and heritability are subset/reordered to selected traits. Bound locus size before calling; SHAP and graph controls may add large work.
    
    Returns
    -------
    FineMapResult
        Fitted joint responses, per-variant ranking/evidence and metadata selected_traits/n_input_samples/n_complete_samples. Bayesian joint inclusion is not colocalization.
    
    Complexity
    ----------
    Wrapper TIME O(n(m+t)+t²); SPACE O(n(m+t)+t²). Fitter TIME is method-dependent: dense LD costs O(nm²), SPACE O(m²); ridge/elastic/Bayesian iterations add their own iteration budgets. n is samples, m variants, t supplied traits. Optional model selection/SHAP are additional.
    
    Notes
    -----
    Raises ValueError for unknown/duplicate traits, missing labelled samples or incompatible covariance/heritability axes. No genome-wide scaling or calibration guarantee is implied.
    """
    from .. import multivariate as mv
    X,xlabel=_frame(genotypes,'genotypes');Y,ylabel=_frame(traits,'traits')
    if len(X)!=len(Y) and not (xlabel and ylabel):
        raise ValueError('genotypes and traits must have the same rows')
    if xlabel and ylabel:
        if not X.index.isin(Y.index).all():
            raise ValueError('trait animal labels do not cover genotype rows')
        Y=Y.reindex(X.index)
    elif xlabel:Y.index=X.index
    elif ylabel:X.index=Y.index
    names=Y.columns.tolist()
    selected=names if trait_subset is None else ([str(trait_subset)] if isinstance(trait_subset,str) else list(map(str,trait_subset)))
    if not selected or len(set(selected))!=len(selected):
        raise ValueError('trait_subset must contain distinct trait names')
    if not pd.Index(selected).isin(Y.columns).all():raise ValueError('unknown trait in trait_subset')
    for name in ('G','E'):
        if name in kwargs:kwargs[name]=_select_covariance(kwargs[name],names,selected,name)
    h=kwargs.get('heritability')
    if h is not None:
        if isinstance(h,pd.Series):
            h=h.copy();h.index=h.index.map(str)
            if not h.index.is_unique or not pd.Index(selected).isin(h.index).all():raise ValueError('heritability labels do not cover selected traits')
            h=h.reindex(selected).to_numpy(float)
        else:
            h=np.asarray(h,float).ravel()
            if len(h)==len(names):h=h[pd.Index(names).get_indexer(selected)]
            elif len(h)!=len(selected):raise ValueError('heritability length must match full or selected traits')
        kwargs['heritability']=h
    Y=Y[selected]
    result=mv.fine_map_region(X,Y,**kwargs)
    result=result if isinstance(result,FineMapResult) else FineMapResult.from_legacy(result)
    result.metadata.update(selected_traits=selected,joint_multitrait=len(selected)>1,
        n_input_samples=len(Y),n_complete_samples=int(np.isfinite(Y.to_numpy(float)).all(axis=1).sum()),
        inclusion_interpretation='joint response-vector inclusion; not a colocalization probability')
    return result


def fine_map_region_legacy(*args,**kwargs):
    """Return the fine map region legacy result computed by ``mv.fine_map_region``.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments forwarded, in order, to ``mv.fine_map_region``. See that
        signature for required scientific inputs and their order.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``mv.fine_map_region``. See those callables
        for accepted keys and values. Passed to ``mv.fine_map_region`` as ``**expanded
        mapping``.

    Returns
    -------
    object
        Computed result from ``mv.fine_map_region(*args, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .. import multivariate as mv
    return mv.fine_map_region(*args,**kwargs)
