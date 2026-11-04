from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np,pandas as pd
from sklearn.cluster import AgglomerativeClustering

@dataclass
class TraitIntegrationResult:
    """TraitIntegrationResult represents trait integration result.

    Attributes
    ----------
    similarity : pd.DataFrame
        Stored similarity; its axis/order follows the construction inputs.
    clusters : pd.Series
        Stored clusters; its axis/order follows the construction inputs.
    stability : pd.Series | None
        Stored stability; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    similarity : pd.DataFrame
        Stored similarity supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    clusters : pd.Series
        Stored clusters supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    stability : pd.Series | None
        Stored stability supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Default: ``None``.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Default: a
        fresh value from ``dict()`` for each instance.
    """
    similarity: pd.DataFrame
    clusters: pd.Series
    stability: pd.Series|None=None
    metadata: dict=field(default_factory=dict)

def integrate_traits(*,G_additive=None,G_dominance=None,G_gxe=None,E=None,weights=None,n_clusters=8,bootstrap_covariances=None,random_state=0):
    """Return the integrate traits result computed by ``TraitIntegrationResult``.

    Parameters
    ----------
    G_additive : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_gxe : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    weights : array-like or int
        Weights or multiplicities used in this accumulation/resampling step. Align vector
        entries with the indexed sample/variant axis; scalar counts specify the number of
        consecutive items. Default: ``None``. None delegates to the local fallback shown in this
        body; it is not an additional numeric/category value. Passed to ``integrate_traits`` as
        ``weights``.

    n_clusters : int
        Input supplied to ``AgglomerativeClustering`` as ``n_clusters``. See that callable's
        documented ``n_clusters`` contract for its type, alignment and value domain. This
        wrapper retains that contract. Default: ``8``. Passed to ``AgglomerativeClustering`` as
        ``n_clusters``.

    bootstrap_covariances : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``TraitIntegrationResult(pd.DataFrame(S, index=names, columns=names), pd.Series(lab, index=names, name='cluster'), stab, {'views': list(views)})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    views={'G_additive':G_additive,'G_dominance':G_dominance,'G_gxe':G_gxe,'E':E};views={k:v for k,v in views.items() if v is not None}
    if not views:raise ValueError('at least one covariance view is required')
    names=next((list(v.index) for v in views.values() if isinstance(v,pd.DataFrame)),[str(i) for i in range(np.asarray(next(iter(views.values()))).shape[0])])
    w={k:1.0 for k in views};w.update({} if weights is None else weights);S=np.zeros((len(names),len(names)),float);den=0
    for k,C in views.items():
        A=np.asarray(C,float);d=np.sqrt(np.clip(np.diag(A),1e-12,None));R=A/d[:,None]/d[None,:];S+=float(w[k])*R;den+=float(w[k])
    S/=max(den,1e-12);D=np.clip(1-S,-0,2);np.fill_diagonal(D,0);lab=AgglomerativeClustering(n_clusters=min(int(n_clusters),len(names)),metric='precomputed',linkage='average').fit_predict(D)
    stab=None
    if bootstrap_covariances:
        rng=np.random.default_rng(random_state);agree=np.zeros(len(names));B=0
        # bootstrap_covariances is iterable of dicts of re-estimated covariance views.
        for draw in bootstrap_covariances:
            Bd=integrate_traits(**draw,weights=weights,n_clusters=n_clusters,bootstrap_covariances=None).clusters.to_numpy();agree += (Bd==lab);B+=1
        if B:stab=pd.Series(agree/B,index=names,name='cluster_stability')
    return TraitIntegrationResult(pd.DataFrame(S,index=names,columns=names),pd.Series(lab,index=names,name='cluster'),stab,{'views':list(views)})
