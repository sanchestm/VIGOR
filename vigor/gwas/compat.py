"""Small-region compatibility table API backed by streaming exact LD."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .ld import _iter_off_diagonal_r2


def off_diagonalR2(snps, snp_dist=1000, min_snp_dist=0, return_square=True, statistic='r2'):
    """Return the historical 200-marker-block LD band for a small region.

    The legacy window was rounded in blocks of 200 markers, not base pairs.
    This wrapper preserves that selection and output schema while using exact
    pairwise-complete R². For genome-wide estimates use ``effective_tests``;
    the requested table/square output here necessarily retains its LD values.

    Parameters
    ----------
    snps : array-like or pandas.DataFrame
        Genotype values with animals in rows and variants in columns, or the genotype source
        accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
        labels and the coding expected by the selected kernel or association model.

    snp_dist : int
        Input for the operation ``int(snp_dist) // 200``. See
        ``vigor.gwas.compat.off_diagonalR2`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``1000``.

    min_snp_dist : int
        Input for the operation ``int(min_snp_dist) // 200``. See
        ``vigor.gwas.compat.off_diagonalR2`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape. Default: ``0``. Source
        comparisons: ``0 <= min_snp_dist <= snp_dist``. See the rejection/fallback branches for
        enforcement.

    return_square : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``True``.

    statistic : {'r2', 'r'}
        Pairwise-complete squared (default) or signed correlation from
        ``vigor.gwas.ld.pairwise_complete``. Default: ``'r2'``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame({'bp1': names.astype(str).to_numpy()[i], 'bp2': names.astype(str).to_numpy()[j], 'value': v, 'distance': np.abs(coords[i] - coords[j])})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    X=np.asarray(snps)
    if X.ndim!=2:
        raise ValueError('snps must be sample by variant')
    if not 0<=min_snp_dist<=snp_dist:
        raise ValueError('require 0 <= min_snp_dist <= snp_dist')
    m=X.shape[1]
    names=pd.Index(snps.columns if isinstance(snps,pd.DataFrame) else [str(j) for j in range(m)])
    if not names.is_unique:
        raise ValueError('variant labels must be unique')
    lo=int(min_snp_dist)//200; hi=int(snp_dist)//200
    ii=[]; jj=[]; vv=[]
    for block in _iter_off_diagonal_r2(X,block_size=200,max_variant_lag=(hi+1)*200-1,statistic=statistic):
        diff=block.j//200-block.i//200
        keep=(diff>=lo)&(diff<=hi)
        i=block.i[keep]; j=block.j[keep]; v=block.r2[keep]
        ii.extend([i,j]); jj.extend([j,i]); vv.extend([v,v])
    if lo==0:
        # Variance-zero and insufficient-data diagonals stay undefined.
        valid=np.sum(np.isfinite(X),axis=0)>=3
        with np.errstate(invalid='ignore'):
            valid &= np.nanvar(X,axis=0)>0
        diag=np.flatnonzero(valid)
        ii.append(diag); jj.append(diag); vv.append(np.ones(len(diag)))
    i=np.concatenate(ii).astype(int) if ii else np.array([],int)
    j=np.concatenate(jj).astype(int) if jj else np.array([],int)
    v=np.concatenate(vv) if vv else np.array([],float)
    if return_square:
        out=np.full((m,m),np.nan); out[i,j]=v
        return pd.DataFrame(out,index=names,columns=names)
    coords=[]
    for k,label in enumerate(names):
        try:coords.append(float(str(label).rsplit(':',1)[-1]))
        except ValueError:coords.append(float(k))
    coords=np.asarray(coords)
    return pd.DataFrame({'bp1':names.astype(str).to_numpy()[i],
                         'bp2':names.astype(str).to_numpy()[j],
                         'value':v,'distance':np.abs(coords[i]-coords[j])})
