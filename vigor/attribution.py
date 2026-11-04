"""Fixed Gaussian-model contributions, distinct from retraining influence."""
import numpy as np
import pandas as pd

from .results import result_dataset, maybe_save, _table_vars, _result_table  # re-exported for callers


def mv_prediction_animal_attribution(kernels, genetic_covariances, residual_covariance,
                                    dual_A, target_rows, *, sample_ids=None, trait_names=None,
                                    trait_mean=None, save_path=None):
    """Exact observation-group contributions to BV and conditional phenotype.

    Value function is the frozen linear posterior map applied to its observed
    dual vector, with zero-dual background. Each training animal groups all its
    observed traits. This is fixed-model contribution, not leave-animal-out
    influence: removing/refitting an animal also changes the inverse covariance.
    E tensor I contributes only the target animal's own observation duals.

    Parameters
    ----------
    kernels : mapping
        Named animal relationship kernels; each matrix/operator must use the same animal order.
        Names identify the matching trait covariance components.

    genetic_covariances : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    residual_covariance : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    dual_A : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    target_rows : array-like or numeric scalar
        Input for the operation ``np.asarray(target_rows, int)``. See
        ``vigor.attribution.mv_prediction_animal_attribution`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities. Default: ``None``. None delegates to the local fallback shown in this body;
        it is not an additional numeric/category value.

    trait_mean : array-like or pandas.DataFrame
        Baseline predicted values in the response animal/trait order, used for conditioning,
        residual construction or evaluation. Supply predictions from the corresponding fit
        without including held-out outcomes in training. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``contributions`` (target, training, trait, component; component is
        ``genetic``/``residual``/``phenotype``, training animals with a nonzero dual only),
        ``breeding_value`` and ``phenotype_prediction`` (target, trait). Attrs:
        ``interpretation`` plus title/software/parameters provenance.
    """
    import xarray as xr
    K=kernels if isinstance(kernels,dict) else {'genetic':kernels}
    G=genetic_covariances if isinstance(genetic_covariances,dict) else {'genetic':genetic_covariances}
    A=np.asarray(dual_A,float);n,t=A.shape;rows=np.asarray(target_rows,int)
    ids=list(map(str,range(n))) if sample_ids is None else list(map(str,sample_ids))
    traits=list(map(str,range(t))) if trait_names is None else list(map(str,trait_names))
    contributions=np.zeros((len(rows),n,t),float)
    for name,k in K.items():contributions+=np.asarray(k)[rows,:,None]*(A@np.asarray(G[name],float).T)[None,:,:]
    residual=np.zeros_like(contributions);E=np.asarray(residual_covariance,float)
    for ii,i in enumerate(rows):residual[ii,i]=A[i]@E.T
    mean=np.zeros(t) if trait_mean is None else np.asarray(trait_mean,float)
    nz=np.flatnonzero(np.any(A!=0,axis=1));tgt=[ids[i] for i in rows]
    cube=np.stack([contributions[:,nz],residual[:,nz],contributions[:,nz]+residual[:,nz]],axis=-1)
    contrib=xr.DataArray(cube,dims=('target','training','trait','component'),
                         coords={'target':tgt,'training':[ids[j] for j in nz],'trait':traits,'component':['genetic','residual','phenotype']})
    ds=result_dataset({'contributions':contrib,
                       'breeding_value':pd.DataFrame(contributions.sum(1),index=tgt,columns=traits),
                       'phenotype_prediction':pd.DataFrame(mean+contributions.sum(1)+residual.sum(1),index=tgt,columns=traits),
                       'interpretation':'fixed linear posterior; zero dual background; grouped by animal; not retraining influence'},
                      dims={'breeding_value':('target','trait'),'phenotype_prediction':('target','trait')},
                      var_attrs={'contributions':{'long_name':'fixed-model contribution of each training animal to each target',
                                                  'description':'genetic: kernel-weighted dual contribution; residual: own-observation E contribution; phenotype: their sum'},
                                 'breeding_value':{'long_name':'reconstructed breeding value','description':'sum of genetic contributions over training animals'},
                                 'phenotype_prediction':{'long_name':'reconstructed conditional phenotype prediction','description':'trait mean plus genetic and residual contributions'}},
                      title='Animal-level prediction attribution',software='vigor.attribution.mv_prediction_animal_attribution',
                      parameters={'kernel_names':','.join(map(str,K)),'n_targets':int(len(rows)),'n_traits':int(t)})
    return maybe_save(ds,save_path)


def linear_snp_attribution(centered_genotypes, dual_A, genetic_covariance, denominator,
                           target_rows, *, block_variants='auto', top_k=64,
                           maxmem=None, memory_fraction=None, save_path=None):
    """Stream exact SNP contributions when K=ZZ'/denominator is the fitted K.

    The caller supplies allele-aligned, frequency-coded Z and its actual GRM
    denominator. No equality to a different cached kernel is assumed. The
    zero-centered genotype background gives additive/interventional linear
    contributions. LD does not turn these contributions into causal evidence.
    Only top contributions are retained; reconstruction includes every SNP.

    Parameters
    ----------
    centered_genotypes : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.

    dual_A : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    genetic_covariance : array-like or numeric scalar
        Input for the operation ``np.asarray(genetic_covariance, float)``. See
        ``vigor.attribution.linear_snp_attribution`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    denominator : array-like or numeric scalar
        Input for the operation ``Z.T @ A @ G.T / float(denominator)``. See
        ``vigor.attribution.linear_snp_attribution`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Source
        comparisons: ``float(denominator) <= 0``. See the rejection/fallback branches for
        enforcement.

    target_rows : array-like or numeric scalar
        Input for the operation ``np.asarray(target_rows, int)``. See
        ``vigor.attribution.linear_snp_attribution`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    block_variants : int or 'auto'
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings). Source comparisons:
        ``block_variants < 1``. See the rejection/fallback branches for enforcement.

    top_k : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``64``. Source comparisons: ``top_k < 1``. See the rejection/fallback branches for
        enforcement.

    maxmem : int, float, unit string or None
        Working-memory budget; see ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: the former ``top_snps`` table along ``candidate`` (``target_row``,
        ``snp``, ``max_abs``; ``contributions`` on (candidate, trait)) and
        ``breeding_value_reconstruction`` (target, trait). Attrs: ``n_variants``,
        ``interpretation`` plus title/software/parameters provenance.
    """
    import xarray as xr
    A=np.asarray(dual_A,float);G=np.asarray(genetic_covariance,float);rows=np.asarray(target_rows,int)
    if not np.isfinite(denominator) or float(denominator)<=0:
        raise ValueError('positive finite actual GRM denominator required')
    if A.ndim!=2 or G.shape!=(A.shape[1],A.shape[1]):
        raise ValueError('aligned dual and trait genetic covariance required')
    if hasattr(centered_genotypes,'dims'):
        sd=next((d for d in centered_genotypes.dims if d in ('iid','sample','animal')),None)
        if sd is None or centered_genotypes.ndim!=2:raise ValueError('labelled sample x variant genotype coding required')
        vd=next(d for d in centered_genotypes.dims if d!=sd);n=centered_genotypes.sizes[sd];m=centered_genotypes.sizes[vd]
        names=centered_genotypes[vd].values.astype(str)
        read=lambda a,b:np.asarray(centered_genotypes.isel({vd:slice(a,b)}).transpose(sd,vd).values,float)
    elif isinstance(centered_genotypes,pd.DataFrame):
        n,m=centered_genotypes.shape;names=list(map(str,centered_genotypes.columns))
        if not centered_genotypes.index.is_unique or not centered_genotypes.columns.is_unique:
            raise ValueError('unique animal/variant labels required')
        read=lambda a,b:centered_genotypes.iloc[:,a:b].to_numpy(float)
    else:
        n,m=centered_genotypes.shape;names=list(map(str,range(m)))
        read=lambda a,b:np.asarray(centered_genotypes[:,a:b],float)
    if n!=A.shape[0] or np.any(rows<0) or np.any(rows>=n):raise ValueError('genotypes/dual/target rows must align')
    if (str(block_variants).lower()!='auto' and int(block_variants)<1) or top_k<1:raise ValueError('positive block_variants and top_k required')
    from .resources import resolve_variant_chunk
    # per call: float64 Z, a source copy and the isfinite mask; per variant: beta and C (+abs) over traits
    block_variants=resolve_variant_chunk(block_variants,n,bytes_per_call=17,bytes_per_variant=8*A.shape[1]*(2+2*len(rows)),
                                         maxmem=maxmem,memory_fraction=memory_fraction,n_variants=m)
    total=np.zeros((len(rows),A.shape[1]));parts=[]
    for a in range(0,m,block_variants):
        b=min(m,a+block_variants);Z=read(a,b)
        if not np.isfinite(Z).all():raise ValueError('supply actual mean-imputed centered GRM coding, no undefined calls')
        beta=(Z.T@A)@G.T/float(denominator)
        C=Z[rows,:,None]*beta[None,:,:];total+=C.sum(1)
        for i,row in enumerate(rows):
            magnitude=np.max(np.abs(C[i]),axis=1);ix=np.argsort(magnitude)[-min(top_k,b-a):]
            for j in ix:parts.append({'target_row':int(row),'snp':str(names[a+j]),'contributions':C[i,j].tolist(),'max_abs':float(magnitude[j])})
    table=pd.DataFrame(parts).sort_values('max_abs',ascending=False).groupby('target_row',sort=False).head(top_k).reset_index(drop=True)
    values=_table_vars(table.drop(columns='contributions'),'top_snps','candidate',
                       long_names={'target_row':'target animal row','snp':'variant identifier','max_abs':'maximum absolute contribution over traits'})
    values['contributions']=xr.DataArray(np.asarray(table['contributions'].tolist(),float).reshape(len(table),A.shape[1]),dims=('candidate','trait'),
                                         attrs={'long_name':'per-trait SNP contribution to the target breeding value'})
    values.update({'breeding_value_reconstruction':total,'n_variants':m,
                   'interpretation':'fixed linear posterior with verified GRM coding; zero centered genotype background; not PIP'})
    ds=result_dataset(values,dims={'breeding_value_reconstruction':('target','trait')},coords={'target':rows},
                      var_attrs={'breeding_value_reconstruction':{'long_name':'breeding value reconstructed from all SNP contributions'}},
                      title='Linear SNP attribution',software='vigor.attribution.linear_snp_attribution',
                      parameters={'denominator':float(denominator),'top_k':int(top_k),'block_variants':int(block_variants)})
    return maybe_save(ds,save_path)
