"""Locus-aware enrichment with tested-gene backgrounds and matched empirical nulls."""
from __future__ import annotations
import numpy as np,pandas as pd
from scipy.stats import hypergeom

def _bh(p):
    """Return the bh result computed by ``np.clip``.

    Parameters
    ----------
    p : array-like
        Probability or p-value quantities transformed or tested by this routine. Finite
        probability inputs are on [0, 1]; preserve alignment with the associated variants or
        observations.
    """
    p=np.asarray(p,float); order=np.argsort(p); q=np.empty_like(p)
    q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.clip(q,0,1)

def _covariate_strata(cov, universe, columns, q=10):
    """Partition the eligible gene universe by covariates for matched enrichment sampling.

    Parameters
    ----------
    cov : array-like, mapping or fitted result
        Covariance quantity or component collection consumed by the stated transformation.
        Square matrices use a common trait order; uncertainty matrices match their corresponding
        covariance entries. See the caller for whether this is genetic, residual or total
        covariance.

    universe : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation.

    columns : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    q : float
        Quantile probability on [0, 1] selecting a distribution cutoff; larger values select an
        upper-tail bound. Default: ``10``. Passed to ``pd.qcut`` as ``q``.
    """
    C=cov.copy(); C.index=C.index.astype(str); C=C.reindex(sorted(universe)); parts=[]
    for col in columns:
        if col not in C: continue
        x=pd.to_numeric(C[col],errors='coerce')
        good=x.notna()
        z=pd.Series('missing',index=C.index,dtype=object)
        if good.sum()>1:
            bins=pd.qcut(x[good].rank(method='first'),q=min(q,max(2,good.sum()//20)),duplicates='drop')
            z.loc[good]=bins.astype(str)
        parts.append(z)
    if not parts: return None
    return pd.Series(list(zip(*[x.tolist() for x in parts])),index=C.index)

def gene_set_enrichment(selected,term_to_genes,*,universe=None,gene_covariates=None,
                        covariate_columns=('length','ld_score'),min_size=3,n_perm=1000,random_state=0):
    """Hypergeometric enrichment plus optional length/LD-matched empirical p-values.

    Parameters
    ----------
    selected : sequence, index or scalar
        Axis/node identifiers or positions selected by the local lookup below. Values must refer
        to the supplied object and preserve the requested output ordering.

    term_to_genes : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation.

    universe : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    gene_covariates : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    covariate_columns : tuple
        Input for the operation ``_covariate_strata(gene_covariates, uni, covariate_columns)``.
        See ``vigor.integration.enrichment.gene_set_enrichment`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``('length', 'ld_score')``.

    min_size : int
        Input for the operation ``K < int(min_size)``. See
        ``vigor.integration.enrichment.gene_set_enrichment`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``3``.

    n_perm : int
        Number of random permutations used to estimate the null/evidence quantity. Nonnegative
        counts control Monte Carlo work; zero disables permutation where supported. Default:
        ``1000``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    sel=set(map(str,selected))
    if universe is None: raise ValueError('gene_universe must be explicit or derived from tested/reachable loci')
    uni=set(map(str,universe)); sel &= uni; rows=[]
    for term,genes in term_to_genes.items():
        gg=set(map(str,genes))&uni; K=len(gg); x=len(sel&gg)
        if K<int(min_size): continue
        rows.append((str(term),x,K,len(sel),len(uni),float(hypergeom.sf(x-1,len(uni),K,len(sel))) if sel else 1.0))
    out=pd.DataFrame(rows,columns=['term','overlap','term_size','selected_size','universe_size','p'])
    if len(out): out['fdr_bh']=_bh(out.p)
    if gene_covariates is not None and len(sel) and len(out):
        strata=_covariate_strata(gene_covariates,uni,covariate_columns)
        if strata is not None:
            rng=np.random.default_rng(random_state); selected_by={}
            for s in sel: selected_by.setdefault(strata.get(s,('missing',)),0); selected_by[strata.get(s,('missing',))]+=1
            pools={k:np.asarray(v.index,dtype=object) for k,v in strata.groupby(strata)}
            null=np.zeros((int(n_perm),len(out)),dtype=np.int32)
            termsets=[set(map(str,term_to_genes[t])) for t in out.term]
            for b in range(int(n_perm)):
                draw=set()
                for st,k in selected_by.items():
                    pool=pools.get(st,np.array([],object)); kk=min(int(k),len(pool))
                    if kk: draw.update(rng.choice(pool,kk,replace=False).tolist())
                null[b]=[len(draw&gg) for gg in termsets]
            obs=out.overlap.to_numpy(); out['matched_empirical_p']=(1+(null>=obs).sum(0))/(int(n_perm)+1)
            out['matched_fdr_bh']=_bh(out.matched_empirical_p); out.attrs['matched_covariates']=list(covariate_columns)
    return out

def locus_preserving_enrichment(selected_by_locus,term_to_genes,*,universe_by_locus,n_perm=1000,random_state=0):
    """Evaluate gene-set enrichment with locus-preserving, covariate-matched null samples.

    Parameters
    ----------
    selected_by_locus : sequence or integer array
        Identifiers or zero-based positions of selected/protected items on the corresponding
        input axis. Entries must exist; paired index arrays define the row/column pairs to
        compute.

    term_to_genes : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation.

    universe_by_locus : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation.

    n_perm : int
        Number of random permutations used to estimate the null/evidence quantity. Nonnegative
        counts control Monte Carlo work; zero disables permutation where supported. Default:
        ``1000``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    object
        Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if universe_by_locus is None: raise ValueError('universe_by_locus is required')
    missing=set(selected_by_locus)-set(universe_by_locus)
    if missing: raise ValueError(f'missing background universe for loci: {sorted(missing)[:5]}')
    selected=set().union(*[set(map(str,v)) for v in selected_by_locus.values()]); rng=np.random.default_rng(random_state); rows=[]
    for term,genes in term_to_genes.items():
        gg=set(map(str,genes)); obs=len(selected&gg); null=np.zeros(int(n_perm),int)
        for b in range(int(n_perm)):
            draw=set()
            for loc,sel in selected_by_locus.items():
                pool=np.array(list(set(map(str,universe_by_locus[loc]))),object); k=min(len(set(sel)),len(pool))
                if k: draw.update(rng.choice(pool,k,replace=False))
            null[b]=len(draw&gg)
        rows.append((term,obs,float(null.mean()),float(null.std(ddof=1)),(1+np.sum(null>=obs))/(len(null)+1)))
    out=pd.DataFrame(rows,columns=['term','overlap','null_mean','null_sd','empirical_p'])
    if len(out): out['fdr_bh']=_bh(out.empirical_p)
    return out
