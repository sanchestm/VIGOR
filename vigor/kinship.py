import numpy as np
import scipy.sparse as sps
from scipy.sparse import identity
from tqdm import tqdm
import numpy as np
import itertools
from scipy.sparse import csr_matrix, identity
from collections import defaultdict
import pandas as pd
try:
    import hvplot.pandas  # noqa: F401
    import holoviews as hv
except ImportError:  # plotting remains optional; sps2fig reports a useful error
    hv = None
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.sparse.csgraph import shortest_path


def sps2fig(sparse_matrix, ids):
    """Render sparse relationship matrix density/entries for inspection.

    Parameters
    ----------
    sparse_matrix : object (see the cited input contract)
        Input for the operation ``pd.DataFrame(sparse_matrix.todense(), columns=ids,
        index=ids)``. See ``vigor.kinship.sps2fig`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.

    Returns
    -------
    object
        Computed result from ``(hm := res.hvplot.heatmap(cmap='Blues', colorbar=False).opts(invert_yaxis=True)) * hv.Labels(hm).opts(padding=0, text_color='red', text_font_size='5pt', frame_width=500, frame_height=500, xrotation...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if hv is None:
        raise ImportError("sps2fig requires VIGOR[plot] (hvplot/holoviews)")
    res = pd.DataFrame(sparse_matrix.todense(), columns=ids, index = ids )
    res = res.loc[~res.index.str.contains('^rndgenerated@'), ~res.columns.str.contains('^rndgenerated@')]
    return (hm:=res.hvplot.heatmap(cmap = 'Blues',colorbar = False ).opts(invert_yaxis = True)) \
            * hv.Labels(hm).opts(padding=0, text_color = 'red', text_font_size = '5pt', frame_width = 500, frame_height = 500, xrotation =90)

def pedigree2sparse(pedigree: pd.DataFrame,parent1_col = 'dam', parent2_col = 'sire', offspring_col = 'rfid', 
                    na_ids = ['NaN', 'NA', 'na', 0, 'nan', 'NAN', 'n/a', 'N/A'] ):
    """Convert pedigree parent links to a sparse transmission/relationship representation.

    Parameters
    ----------
    pedigree : pd.DataFrame
        Pedigree table with offspring and parental identifiers. Parent-before-child/topological
        ordering and unknown-parent markers follow the selected pedigree algorithm; cycles and
        duplicate offspring may be rejected.

    parent1_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'dam'``. Passed to
        ``pedigree.melt`` as ``value_vars``.

    parent2_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'sire'``. Passed to
        ``pedigree.melt`` as ``value_vars``.

    offspring_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'rfid'``. Passed to
        ``pedigree.melt`` as ``id_vars``.

    na_ids : sequence
        Markers interpreted as unknown/missing parent identifiers. Use the values actually
        present in the pedigree; these do not denote real animals. Default: ``['NaN', 'NA',
        'na', 0, 'nan', 'NAN', 'n/a', 'N/A']``.

    Returns
    -------
    object
        Computed result from ``(sped, melted, id2num)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    cols = [offspring_col, parent1_col, parent2_col]
    ped = pedigree.loc[:, cols].copy()
    if ped[offspring_col].isna().any():
        raise ValueError("offspring identifiers cannot be missing")
    ped[offspring_col] = ped[offspring_col].astype(str)
    if not ped[offspring_col].is_unique:
        raise ValueError("pedigree offspring IDs must be unique")
    unknown = {str(v).strip() for v in na_ids if not pd.isna(v)} | {"", "0"}
    def normalize_parent(value):
        return np.nan if pd.isna(value) or str(value).strip() in unknown else str(value)
    for col in (parent1_col, parent2_col):
        ped[col] = ped[col].map(normalize_parent)
    known_both = ped[parent1_col].notna() & ped[parent2_col].notna()
    if (ped.loc[known_both, parent1_col] == ped.loc[known_both, parent2_col]).any():
        raise ValueError("each animal must have at most two distinct parents")
    melted = ped.melt(id_vars=offspring_col, value_vars=[parent1_col, parent2_col],
                      value_name="parent", var_name="relatioship").dropna(subset=["parent"])
    melted["distance"] = 0.5
    # Include every recorded offspring, including isolated founders with no links.
    ordered_ids = list(dict.fromkeys(ped[offspring_col].tolist()+melted["parent"].tolist()))
    id2num = {value: i for i, value in enumerate(ordered_ids)}
    melted["offspring_id"] = melted[offspring_col].map(id2num)
    melted["parent_id"] = melted["parent"].map(id2num)
    P = sps.csr_matrix((np.full(len(melted), .5),
        (melted["offspring_id"].to_numpy(dtype=int), melted["parent_id"].to_numpy(dtype=int))),
        shape=(len(id2num), len(id2num)))
    topo_order(P)  # validates cycles, self-parentage, coefficients and parent counts
    return P, melted, id2num


def cutoff_hist_valleys(x, bins=1000, sigma=1.5, rang =  [0,1], total_density = .98):
    """Find candidate relatedness cutoffs from valleys in the relationship histogram.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``smooth[x.start:x.end]``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    bins : int or array-like
        Calibration bin count or ordered bin edges for the statistic/distance being summarized.
        Counts are positive integers; explicit edges must be ordered and cover the intended
        scale. Default: ``1000``. Passed to ``np.histogram`` as ``bins``.

    sigma : float
        Input supplied to ``gaussian_filter1d`` as ``sigma``. See that callable's documented
        ``sigma`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``1.5``. Passed to ``gaussian_filter1d`` as ``sigma``.

    rang : array-like or numeric scalar
        Input supplied to ``np.histogram`` as ``range``. See that callable's documented
        ``range`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``[0, 1]``. Passed to ``np.histogram`` as ``range``.

    total_density : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``0.98``.

    Returns
    -------
    object
        Computed result from ``(cutoffs, centers, smooth)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    hist, edges = np.histogram(x, bins=bins, density=True, range = rang )
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(hist, sigma=sigma)
    mins = argrelextrema(smooth, np.less)[0]
    smooth = pd.Series(smooth, index = np.linspace(rang[0],rang[1],bins))
    cutoffs = pd.DataFrame([[x,y] for x,y in zip(([0]+list(centers[mins])+[1])[:-1], ([0]+list(centers[mins])+[1])[1:])],  columns = ['start', 'end'])
    cutoffs['density'] = cutoffs.apply(lambda x: smooth[x.start: x.end].sum(), axis = 1)
    cutoffs = cutoffs.sort_values('density', ascending = True)
    cutoffs['density_cumsum'] = cutoffs.density.cumsum()
    cutoffs['density_pass'] = cutoffs['density_cumsum'].gt(total_density)
    cutoffs = cutoffs.sort_index()    
    return cutoffs, centers, smooth

def kinship_from_pedigree(pedigree: pd.core.frame.DataFrame, parent1_col='dam', parent2_col='sire', offspring_col='rfid',
                         threshold_approx = None, na_ids = ['NaN', 'NA', 'na', 0, 'nan', 'NAN', 'n/a', 'N/A'], density_bins = 1000, density_sigma = 2):
    """Propagate pedigree relationships through parental transmission, including unknown founders.

    Parameters
    ----------
    pedigree : pd.core.frame.DataFrame
        Pedigree table with offspring and parental identifiers. Parent-before-child/topological
        ordering and unknown-parent markers follow the selected pedigree algorithm; cycles and
        duplicate offspring may be rejected.

    parent1_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'dam'``. Passed to
        ``pedigree2sparse`` as ``parent1_col``.

    parent2_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'sire'``. Passed to
        ``pedigree2sparse`` as ``parent2_col``.

    offspring_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'rfid'``. Passed to
        ``pedigree2sparse`` as ``offspring_col``.

    threshold_approx : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value. Passed to
        ``kinship_from_child_parent_fast`` as ``threshold_approx``.

    na_ids : sequence
        Markers interpreted as unknown/missing parent identifiers. Use the values actually
        present in the pedigree; these do not denote real animals. Default: ``['NaN', 'NA',
        'na', 0, 'nan', 'NAN', 'n/a', 'N/A']``.

    density_bins : int
        Input supplied to ``cutoff_hist_valleys`` as ``bins``. See that callable's documented
        ``bins`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``1000``. Passed to ``cutoff_hist_valleys`` as ``bins``.

    density_sigma : int
        Input supplied to ``cutoff_hist_valleys`` as ``sigma``. See that callable's documented
        ``sigma`` contract for its type, alignment and value domain. This wrapper retains that
        contract. Default: ``2``. Passed to ``cutoff_hist_valleys`` as ``sigma``.

    Returns
    -------
    object
        Computed result from ``(phi, phidf, sped, melted, id2num, cutoffs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    sped, melted, id2num = pedigree2sparse(pedigree, parent1_col=parent1_col,
        parent2_col=parent2_col, offspring_col=offspring_col, na_ids=na_ids)
    if len(set(np.unique(np.array(sped.sum(axis = 1)))) - {0,1, .5}): raise ValueError('some individuals have more than 2 parents')
    phi, _A = kinship_from_child_parent_fast(sped,threshold_approx=threshold_approx)
    phicoo = phi.tocoo()
    phidf = pd.DataFrame(dict(id1 = phicoo.row, id2 = phicoo.col, phi = phicoo.data) )
    del(phicoo)
    id2num_r = pd.Series(data = id2num.keys(), index = id2num.values()).to_dict()
    phidf['iid1'] = phidf.id1.map(id2num_r)
    phidf['iid2'] = phidf.id2.map(id2num_r)
    phidf = phidf[['id1', 'iid1', 'id2', 'iid2', 'phi']]
    phidf['phi_bin']= pd.cut(phidf.phi, bins =[0, 0.0255,0.0535, 0.1125, 0.2065 , .3, .45, .55, 1  ] )
    cutoffs, centers, smooth = cutoff_hist_valleys(phidf.phi, bins =density_bins, sigma= density_sigma)
    bins2 = cutoffs.query('density_pass').melt(value_vars=['start', 'end']).drop_duplicates(subset= 'value').value.to_list()+ [1]
    phidf['phi_autobin']= pd.cut(phidf.phi, bins =sorted(bins2))
    return phi, phidf, sped, melted, id2num, cutoffs

def chrYMT_from_pedigree(ped, parent_female_col='dam', parent_male_col='sire',  offspring_col='rfid',):
    """Construct sex-specific Y/mitochondrial lineage relationships from pedigree inheritance.

    Parameters
    ----------
    ped : pandas.DataFrame or path-like
        Pedigree table with offspring and parental identifiers. Parent-before-child/topological
        ordering and unknown-parent markers follow the selected pedigree algorithm; cycles and
        duplicate offspring may be rejected.

    parent_female_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'dam'``. Passed to
        ``ped.reset_index().assign`` as ``**expanded mapping``.

    parent_male_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'sire'``. Passed to
        ``ped.reset_index().assign`` as ``**expanded mapping``.

    offspring_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'rfid'``.

    Returns
    -------
    object
        Computed result from ``(ytm_version, (spedy, spedy))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    spedy, meltedy, id2numy = pedigree2sparse(ped.reset_index().assign(**{parent_female_col : np.nan}))
    spedmt, meltedmt, id2numt = pedigree2sparse(ped.reset_index().assign(**{parent_male_col : np.nan}))
    spedmt, spedy = spedmt + spedmt.T , spedy + spedy.T
    ccy = sps.csgraph.connected_components(spedy)[1]
    ccmt = sps.csgraph.connected_components(spedmt)[1]
    ytm_version = pd.DataFrame({'rfid': id2numy.keys(), 'Y':ccy[[x for x in id2numy.values() if x<ccy.shape[0]]]}).merge(
                     pd.DataFrame({'rfid': id2numt.keys(), 'MT':ccmt[[x for x in id2numt.values() if x<ccmt.shape[0]]]}),on = 'rfid')\
                      .reset_index(drop = True)
    ytm_version['Y'] = 'connected_component_'+ ytm_version['Y'].astype(str)
    ytm_version['MT'] = 'connected_component_'+ ytm_version['MT'].astype(str)
    return ytm_version, (spedy, spedy)

def sum_powers(P, max_iter = 10):
    """Return exact sparse ancestry S=(I-P)^-1 for a finite pedigree DAG.

    Inputs: square sparse child-parent matrix P with at most two distinct 0.5
    parent links per row. max_iter is retained for compatibility and ignored.
    Outputs: CSR ancestry matrix, including every ancestor path. Cycles and
    invalid coefficients raise ValueError. Time is proportional to propagated
    ancestry entries; worst-case time/storage are O(n²) for n animals.
    """
    # Exact finite-DAG solve; max_iter is retained solely for call compatibility.
    # A generation cap must never silently discard remote common ancestors.
    return compute_S_forward(P)


def parents_of(A):
    """list of parent index lists for each individual (row i has the parents).

    Parameters
    ----------
    A : array-like
        Numerical operand in ``A.shape[0]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    Returns
    -------
    object
        Computed result from ``[A.getrow(i).indices.tolist() for i in range(A.shape[0])]``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return [A.getrow(i).indices.tolist() for i in range(A.shape[0])]


def topo_order(P):
    """Parents before children (DAG required).

    Parameters
    ----------
    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    Returns
    -------
    object
        Computed result from ``(order, founders, par)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from collections import deque
    P = sps.csr_matrix(P, dtype=float, copy=True)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("parent transmission matrix must be square")
    P.sum_duplicates(); P.eliminate_zeros(); P.sort_indices()
    if not np.isfinite(P.data).all() or np.any(P.data != .5):
        raise ValueError("each parent transmission coefficient must equal 0.5")
    if np.any(np.diff(P.indptr) > 2):
        raise ValueError("each animal must have at most two distinct parents")
    if np.any(P.diagonal() != 0):
        raise ValueError("an animal cannot be its own parent")
    n = P.shape[0]
    par = [P.indices[P.indptr[i]:P.indptr[i+1]].tolist() for i in range(n)]
    indegree = np.diff(P.indptr).copy()
    founders = np.flatnonzero(indegree == 0).tolist()
    children = P.T.tocsr()
    ready = deque(founders)
    order = []
    while ready:
        parent = ready.popleft()
        order.append(parent)
        for child in children.indices[children.indptr[parent]:children.indptr[parent+1]]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    if len(order) != n:
        raise ValueError("pedigree contains a directed ancestry cycle")
    return order, founders, par


def kinship_from_child_parent(P, max_iter = 20):
    """Compute exact sparse kinship Phi and additive relationship A=2*Phi.

    Inputs: square sparse child-parent matrix P with at most two distinct 0.5
    parent links per row. max_iter is retained for compatibility and ignored.
    Outputs: (Phi, A), both CSR matrices in P's original animal order. Unknown
    parents are unrelated founders. Cycles and invalid parent links fail closed.
    Storage is O(nnz(S)+nnz(A)); the product can require O(n²) memory.
    """
    # Retained public spelling; both paths use the same exact forward algorithm.
    # max_iter no longer truncates ancestry.
    return kinship_from_child_parent_fast(P)



# ---------- fast S via forward sweep (S = I + P S) ----------
def compute_S_forward(P):
    """P: CSR child->parent with 0.5 entries.
    Returns S in CSR with rows built once in topological order.

    Parameters
    ----------
    P : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training.

    Returns
    -------
    object
        Computed result from ``csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))``; see the summary and the called implementation for its labeled/numerical representation.
    """
    P = sps.csr_matrix(P, dtype=float)
    n = P.shape[0]
    order, founders, par = topo_order(P)

    # We'll build S in LIL (fast row writes), then convert to CSR.
    S = [dict() for _ in range(n)]  # row as {col: val}

    for i in order:
        row = S[i]
        row[i] = 1.0                    # identity contribution
        for p in par[i]:                # add parents' rows scaled by 0.5
            for k, v in S[p].items():
                row[k] = row.get(k, 0.0) + 0.5 * v

    # convert to CSR
    indptr = [0]
    indices = []
    data = []
    for i in range(n):
        # keep sparsity tidy
        items = [(j, v) for j, v in S[i].items() if v != 0.0]
        items.sort()
        indices.extend([j for j, _ in items])
        data.extend([v for _, v in items])
        indptr.append(len(indices))
    return csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))

    
# ---------- weighted row intersection (phi_pq core) ----------
def weighted_row_dot(S, p, q, d):
    """Compute a weighted inner product between sparse row representations.
    S must be CSR. p and q are row indices. d is 1D array.

    Parameters
    ----------
    S : array-like
        Numerical operand in ``S.indptr[p]``. Shape must conform to that matrix/vector operation
        and the enclosing prepared state.

    p : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations.

    q : float
        Quantile probability on [0, 1] selecting a distribution cutoff; larger values select an
        upper-tail bound.

    d : array-like
        Numerical operand in ``val_p[i] * d[a] * val_q[j]``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    Returns
    -------
    object
        Computed result from ``acc``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rp_start, rp_end = S.indptr[p], S.indptr[p+1]
    rq_start, rq_end = S.indptr[q], S.indptr[q+1]
    idx_p = S.indices[rp_start:rp_end]; val_p = S.data[rp_start:rp_end]
    idx_q = S.indices[rq_start:rq_end]; val_q = S.data[rq_start:rq_end]

    # two-pointer merge over sorted column indices
    i = j = 0
    acc = 0.0
    while i < len(idx_p) and j < len(idx_q):
        a, b = idx_p[i], idx_q[j]
        if a == b:
            acc += val_p[i] * d[a] * val_q[j]
            i += 1; j += 1
        elif a < b: i += 1
        else: j += 1
    return acc


def kinship_from_child_parent_fast(P, return_S=False, threshold_approx=None):
    """P: CSR with P[i,j]=0.5 if j is a parent of i, else 0.
    Returns ``(Phi, A)`` using:
      1) S via forward sweep (S = I + P S)
      2) D diagonals via topological pass and weighted row intersections
      3) A = (S * sqrt(d)) @ (S * sqrt(d)).T

    ``threshold_approx`` accepts only None/0; approximate pruning was retired.
    Ancestry is propagated over the full validated DAG, with no generation cap.
    For n animals and a ancestry nonzeros, storage is O(a + nnz(A)); the final
    relationship product can require O(n²) storage even for a sparse pedigree.

    Parameters
    ----------
    P : object (see the cited input contract)
        CSR with P[i,j]=0.5 if j is a parent of i, else 0.

    return_S : bool
        Whether to enforce/use complete observations or include the named
        decomposition/component output. The enabled branch below defines the returned
        representation; disabling skips that optional step. Default: ``False``.

    threshold_approx : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``(Phi, A)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if threshold_approx not in (None, 0):
        raise ValueError("threshold_approx was retired; pedigree kinship is exact")
    P = sps.csr_matrix(P, dtype=float)
    n = P.shape[0]
    order, founders, par = topo_order(P)
    # 1) Build S once (fast)
    S = compute_S_forward(P)   # CSR
    # 2) Build D diagonals (Mendelian sampling variances) and inbreeding
    d = np.zeros(n, dtype=float)
    F = np.zeros(n, dtype=float)
    for i in founders:
        d[i] = 1.0; F[i] = 0.0
    for i in tqdm(order):
        if i in founders: continue
        ps = par[i]
        if len(ps) == 1:
            p = ps[0]
            F[i] = 0.0
            d[i] = 0.75 - 0.25*F[p]
        elif len(ps) == 2:
            p, q = ps
            # φ_pq = 0.5 * sum_k S[p,k]*d[k]*S[q,k]
            phi_pq = 0.5 * weighted_row_dot(S, p, q, d)
            F[i] = phi_pq
            d[i] = 0.5 - 0.25*(F[p] + F[q])
        else:
            d[i] = 1.0; F[i] = 0.0
    # 3) A = S D S^T via column-scaling by sqrt(d)
    rootd = np.sqrt(d, dtype=float)
    Sc = S.tocsc(copy=True)              # scale columns in-place cheaply
    for j in tqdm(range(n)):
        if rootd[j] != 1.0:
            start, end = Sc.indptr[j], Sc.indptr[j+1]
            if end > start: Sc.data[start:end] *= rootd[j]
    if return_S:
        return Sc
    Sc = Sc.tocsr()
    A = (Sc @ Sc.T).tocsr()
    Phi = (0.5 * A).tocsr()
    return Phi, A






def generation_deltas_from_predecessors(E, indices):
    """Returns
    -------
    sp_dist : (m, n) float array
    sp_path : (m, n) int array
    delta   : (m, n) float array
        delta[k, j] = generation(j) - generation(indices[k])
        along the chosen shortest path

    Parameters
    ----------
    E : array-like or covariance specification or None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance.

    indices : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute. Passed to ``shortest_path`` as ``indices``.
    """
    E = E.tocsr()
    U = (E + E.T).tocsr()
    U.sum_duplicates()
    U.eliminate_zeros()
    if U.nnz: U.data[:] = 1
    sp_dist, sp_path = shortest_path( U, directed=False, unweighted=True, indices=indices, return_predecessors=True)
    m, n = sp_path.shape
    delta = np.full((m, n), np.nan, dtype=float)
    # source delta = 0
    delta[np.arange(m), indices] = 0
    finite = np.isfinite(sp_dist)
    if not finite.any(): return sp_dist, sp_path, delta
    maxd = int(np.nanmax(sp_dist[finite]))
    for d in tqdm(range(1, maxd + 1)):
        rows, cols = np.where(sp_dist == d)
        if len(rows) == 0: continue
        pred = sp_path[rows, cols]
        # +1 if pred -> col exists in E, else -1
        forward = np.asarray(E[pred, cols]).ravel() != 0
        step = np.where(forward, 1.0, -1.0)
        delta[rows, cols] = delta[rows, pred] + step
    return sp_dist, sp_path, delta


# Phi, A = kinship_from_child_parent(sped2) 
# sps2fig(Phi, ids).opts(frame_width = 600,frame_height = 600)


# from gwas.kinship import kinship_from_pedigree, generation_deltas_from_predecessors
# E = kinship_from_pedigree(pedigree)
# indices = pedigree[~pedigree[f'generation'].isna()].rfid.map(E[4]).values
# ped_gen = pedigree.loc[~pedigree[f'generation'].isna(), [f'generation']].values
# sp_dist, sp_path, delta = generation_deltas_from_predecessors(E[2], indices)
# new_gen = np.nanmedian(ped_gen - delta, axis = 0)
# pedigree[f'generation_propagation'] = pedigree['rfid'].map(E[4]).apply(lambda x : new_gen[x]) + 1
# pedigree = pedigree[pedigree.coatcolor.isin(['BROWNHOOD', 'BLACKHOOD', 'BROWN', 'BLACK', 'ALBINO'])]

# ---------------------------------------------------------------------------
# Scalable exact inbreeding for Henderson A^-1
# ---------------------------------------------------------------------------
def pedigree_inbreeding_sparse(
    pedigree: pd.DataFrame,
    parent1_col='dam', parent2_col='sire', offspring_col='rfid', *,
    max_pair_cache=200_000,
):
    """Compute pedigree inbreeding with sparse parental relationship recursion.

    The pedigree can arrive in any order; its ancestry graph must be acyclic.
    Named parents absent from the table are retained as unrelated founders.
    Pairwise kinships needed for parental mating pairs are evaluated by the
    standard recursion and memoized with a bounded LRU cache.  Memory is O(n)
    for pedigree/F plus O(max_pair_cache), rather than O(n^2).  The recursion
    is exact under the usual assumption that unknown founders are unrelated.

    ``max_pair_cache=None`` keeps every visited pair; a finite value bounds
    memory for very large pedigrees at the cost of recomputation.

    Parameters
    ----------
    pedigree : pd.DataFrame
        Pedigree table with offspring and parental identifiers. Parent-before-child/topological
        ordering and unknown-parent markers follow the selected pedigree algorithm; cycles and
        duplicate offspring may be rejected.

    parent1_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'dam'``.

    parent2_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'sire'``.

    offspring_col : str
        Column name selecting identifiers/weights in the supplied table. It must exist and
        contain the corresponding animal/node/group values. Default: ``'rfid'``.

    max_pair_cache : int
        Maximum number of pair/group/path work items retained or evaluated by this guard.
        Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
        silently approximate. Default: ``200000``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return _pedigree_inbreeding_exact(pedigree, parent1_col, parent2_col,
        offspring_col, (None, "", "0", 0, np.nan, "NA", "NaN", "nan", "None"),
        max_pair_cache)


# ---------------------------------------------------------------------------
# VIGOR 2.7 pedigree hardening
# ---------------------------------------------------------------------------
def pedigree_inbreeding(pedigree: pd.DataFrame, parent1_col='dam', parent2_col='sire', offspring_col='rfid', *, unknown_values=(None, '', '0', 0, np.nan), return_relationship=False):
    """Compute exact inbreeding coefficients in the original pedigree row order.

    Inputs: DataFrame, offspring/parent column labels, explicit unknown-ID
    markers, and return_relationship=False. IDs must be unique and ancestry
    acyclic; named parents absent from the table are unrelated founders.
    Outputs: Series F, or (F, labeled additive relationship DataFrame) when
    return_relationship=True. The latter materializes O(n²) entries. Cached
    parental recursion normally uses O(n+cache) memory; worst-case pair work
    is O(n²), with additional work possible when bounded-cache entries evict.
    """
    out = _pedigree_inbreeding_exact(pedigree, parent1_col, parent2_col,
        offspring_col, unknown_values, 200_000)
    if not return_relationship:
        return out
    P, _, id2num = pedigree2sparse(pedigree, parent1_col=parent1_col,
        parent2_col=parent2_col, offspring_col=offspring_col, na_ids=unknown_values)
    _, relationship = kinship_from_child_parent_fast(P)
    ids = out.index.astype(str).tolist()
    selected = np.asarray([id2num[x] for x in ids], dtype=np.int64)
    A = relationship[selected][:, selected].toarray()
    return out, pd.DataFrame(A, index=ids, columns=ids)



def _pedigree_inbreeding_exact(pedigree, parent1_col, parent2_col, offspring_col,
                               unknown_values, max_pair_cache):
    """Exact cached parental recursion, including named unrecorded founders.

    Inputs are a pedigree DataFrame, column labels, unknown-ID markers and an
    optional positive LRU cache capacity. Returns F for recorded offspring in
    their original row order. Storage is O(n + cache); time depends on the
    number of ancestral pairs visited (potentially O(n²) without cache eviction).
    Cycles, duplicate offspring and non-distinct parent pairs are rejected.
    """
    from functools import lru_cache
    if max_pair_cache is not None and (isinstance(max_pair_cache, bool) or int(max_pair_cache) != max_pair_cache or max_pair_cache < 1):
        raise ValueError("max_pair_cache must be a positive integer or None")
    P, _, id2num = pedigree2sparse(pedigree, parent1_col=parent1_col,
        parent2_col=parent2_col, offspring_col=offspring_col, na_ids=unknown_values)
    order, _, parents = topo_order(P)
    ranks = {node: rank for rank, node in enumerate(order)}
    parental = [[ranks[p] for p in parents[node]] for node in order]
    F = np.zeros(len(order), dtype=np.float64)
    @lru_cache(maxsize=max_pair_cache)
    def phi(i, j):
        if i < j:
            return phi(j, i)
        if i == j:
            return .5*(1+F[i])
        ps = parental[i]
        return .5*sum(phi(p, j) for p in ps)
    for i, ps in enumerate(parental):
        F[i] = phi(ps[0], ps[1]) if len(ps) == 2 else 0.
    ids = pedigree[offspring_col].astype(str).tolist()
    out = pd.Series([F[ranks[id2num[x]]] for x in ids], index=ids, name="inbreeding")
    out.attrs.update(algorithm="bounded_recursive_kinship", max_pair_cache=max_pair_cache)
    return out
