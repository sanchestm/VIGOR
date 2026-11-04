"""Trait-name alignment and factor-cluster/ontology evaluation.

Clusters use normalized absolute native loadings (Ward linkage). NMI/ARI compare
clusters with ontology labels. Permutations are within project to prevent study
labels alone driving agreement; uninformative nulls are marked explicitly.
"""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def canonical_trait(name):
    """Normalize a project-qualified trait identifier for ontology joins.
    
    Parameters
    ----------
    name : object
        Converted to str. Replaces __regressedlr_ by :regressedlr_; if no
        :regressedlr_ remains, inserts regressedlr_ after the first colon.
        Names without a colon remain unchanged; no case/whitespace normalization.
    
    Returns
    -------
    str
        Canonical identifier; no ontology lookup or existence validation occurs.
    
    Complexity
    ----------
    TIME O(l); SPACE O(l), where l is the input string length.
    """
    name = str(name).replace('__regressedlr_', ':regressedlr_')
    return name if ':regressedlr_' in name else name.replace(':', ':regressedlr_', 1)



def ontology_annotations(path, traits, levels):
    """Read real ontology labels and align them to an ordered trait index.
    
    Parameters
    ----------
    path : str or pathlib.Path
        CSV readable by pandas.read_csv, including inferred gzip compression.
        Key is the first available fulltraitname/full_trait/project_measure_key_raw.
        Optional trait_covariate filters rows to the literal value 'trait'.
    traits : sequence of str or pandas.Index
        Canonical identifiers in desired output order; missing keys yield NaN.
        This is a list of trait names, not an animal-by-trait phenotype matrix.
    levels : list of str
        Requested annotation column names, including project if explicitly stored.
        All requested columns must exist. Conflicting nonmissing duplicate labels
        for one canonical trait are rejected.
    
    Returns
    -------
    pandas.DataFrame, shape (t, l) or (t, l+1)
        Requested labels in traits order, plus project when not requested, inferred
        from each identifier before its first colon. Empty/'nan'/'None' labels
        become NaN. Canonical duplicate rows with agreeing labels collapse first.
    
    Raises
    ------
    ValueError
        No supported key column, requested level absent, or conflicting duplicates.
    OSError
        CSV cannot be opened; pandas parsing errors propagate.
    
    Complexity
    ----------
    TIME O(b+ul+tl); SPACE O(b+ul+tl), with b CSV bytes materialized, u ontology
    rows, t requested traits and l levels; pandas groupby adds implementation-
    dependent hashing/sorting overhead. String key normalization adds key bytes.
    """
    ont=pd.read_csv(path,low_memory=False); ont.columns=ont.columns.astype(str)
    if 'trait_covariate' in ont: ont=ont.loc[ont.trait_covariate.eq('trait')]
    key=next((k for k in ['fulltraitname','full_trait','project_measure_key_raw'] if k in ont),None)
    if key is None: raise ValueError('Ontology needs fulltraitname, full_trait, or project_measure_key_raw')
    ont['trait']=ont[key].map(canonical_trait)
    if set(levels)-set(ont): raise ValueError(f'Missing ontology levels: {set(levels)-set(ont)}')
    if ont.groupby('trait')[levels].nunique().gt(1).any().any(): raise ValueError('Conflicting duplicate ontology annotations')
    ann=ont.drop_duplicates('trait').set_index('trait')[levels].reindex(traits).replace(['nan','None',''],np.nan)
    if 'project' not in ann: ann['project']=pd.Index(traits).str.split(':').str[0]
    return ann


def bh_adjust(p):
    """Apply Benjamini–Hochberg adjustment to finite p-values in original order.
    
    Parameters
    ----------
    p : one-dimensional array-like of float, shape (m,)
        Expected probabilities in [0,1]. NaN and infinities are excluded from the
        family and remain NaN in output. Bounds are assumed, not validated.
    
    Returns
    -------
    numpy.ndarray, shape (m,), float64
        Reverse-cumulative-minimum adjusted values, capped at one, in input order.
        All nonfinite input returns all NaN; empty input returns an empty vector.
    
    Complexity
    ----------
    TIME O(m+f log f); SPACE O(m+f), where f is the number of finite entries.
    """
    p=np.asarray(p,float); out=np.full(len(p),np.nan); ix=np.flatnonzero(np.isfinite(p))
    order=ix[np.argsort(p[ix])]; adjusted=p[order]*len(order)/np.arange(1,len(order)+1)
    out[order]=np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1]);return out


def evaluate_ontology(loadings, ann, seed=42, permutations=199):
    """Cluster absolute native loadings and evaluate real annotation agreement.
    
    Parameters
    ----------
    loadings : pandas.DataFrame of float, shape (t,r)
        Finite trait-by-factor loadings with unique trait index. Rows with norm
        <=1e-8 are excluded. Ward linkage uses normalized absolute loadings; its
        maxclust cut requests min(r, active traits) clusters. Nonfinite values are
        unsupported and may raise from SciPy rather than being imputed.
    ann : pandas.DataFrame, shape (t,l)
        Exactly the same row order/index as loadings, with a project column.
        Each column is evaluated as a categorical level, including project.
        Missing labels are excluded separately per level. Nothing is fabricated.
    seed : int or None
        NumPy Generator seed, default 42, for within-project label permutations.
    permutations : int
        Number of null permutations, default 199, expected nonnegative. No input
        validator enforces this; zero leaves a plus-one p-value of one where movable.
    
    Returns
    -------
    tuple
        membership: Series (t,), cluster labels with -1 for excluded rows;
        scores: DataFrame with level,n,coverage,ari,nmi,within_project_p,
        within_project_null_informative and, when nonempty, within_project_q;
        enrich: DataFrame with level,cluster,label,overlap,cluster_n,label_n,p,q;
        ordered: Index of active traits in Ward leaf order. Levels with fewer than
        four labelled active traits or fewer than two distinct labels are skipped.
        Empty tables may have no columns. Null p-values are NaN when permutations
        cannot change within-project labels. q columns use separate BH families.
    
    Raises
    ------
    KeyError
        project is absent when an eligible level is tested.
    ValueError
        Incompatible positional annotation length or invalid linkage input.
    
    Complexity
    ----------
    TIME O(t²r + l(P t log t + c h t) + e log e); SPACE O(t²+tr+e+P).
    P is permutations, c clusters, h labels per level and e enrichment rows.
    Metric label coding/grouping is conservatively bounded by t log t. Dense
    Ward distances dominate space; this routine is not a streamed clusterer.
    """
    A=np.abs(loadings.to_numpy()); strength=np.linalg.norm(A,axis=1); valid=strength>1e-8
    Z=A[valid]/strength[valid,None]; tree=linkage(Z,method='ward') if len(Z)>1 else None
    membership=pd.Series(-1,index=loadings.index,name='cluster')
    membership.loc[valid]=fcluster(tree,min(loadings.shape[1],len(Z)),criterion='maxclust') if tree is not None else 1
    records=[];enrichment=[];rng=np.random.default_rng(seed)
    for level in ann.columns:
        ok=valid & ann[level].notna().to_numpy();labels=ann.loc[ok,level].astype(str).to_numpy(); clusters=membership.loc[ok].to_numpy()
        if len(labels)<4 or len(set(labels))<2: continue
        observed=normalized_mutual_info_score(labels,clusters); null=[]; movable=False
        projects=ann.loc[ok,'project'].astype(str).to_numpy()
        for _ in range(permutations):
            perm=labels.copy()
            for project in np.unique(projects):
                ii=np.flatnonzero(projects==project); movable |= len(set(labels[ii]))>1;perm[ii]=rng.permutation(perm[ii])
            null.append(normalized_mutual_info_score(perm,clusters))
        records.append(dict(level=level,n=len(labels),coverage=ok.sum()/len(loadings),ari=adjusted_rand_score(labels,clusters),nmi=observed,
                            within_project_p=(1+np.sum(np.asarray(null)>=observed))/(permutations+1) if movable else np.nan,
                            within_project_null_informative=movable))
        for cluster in np.unique(clusters):
            for label in np.unique(labels):
                inside=clusters==cluster; cat=labels==label; k=int(np.sum(inside&cat))
                enrichment.append(dict(level=level,cluster=int(cluster),label=label,overlap=k,cluster_n=int(inside.sum()),label_n=int(cat.sum()),p=hypergeom.sf(k-1,len(labels),cat.sum(),inside.sum())))
    scores=pd.DataFrame(records);enrich=pd.DataFrame(enrichment)
    if len(enrich): enrich['q']=bh_adjust(enrich.p)
    if len(scores): scores['within_project_q']=bh_adjust(scores.within_project_p)
    ordered=loadings.index[valid][leaves_list(tree)] if tree is not None else loadings.index[valid]
    return membership,scores,enrich,ordered


