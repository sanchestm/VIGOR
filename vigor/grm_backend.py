"""Shared bounded-block GRM arithmetic; packed BED stays packed until a block.

npplink uses empirical sample SD (ddof=1); gcta0 uses HWE scaling.
Only observed calls of component-eligible markers enter pair counts.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg.blas import ssyrk, dsyrk

from .grm_methods import validate_grm_method


def _thread_context(threads):
    """Limit BLAS locally and restore the caller's settings on exit.
    
    Parameters
    ----------
    threads : int or str
        Positive local BLAS worker limit; 'auto' and None use every CPU available to the process
        (the per-block SYRK is the only work in flight). The previous thread settings are restored on exit.

    """
    from threadpoolctl import threadpool_limits
    from .resources import usable_cpu_count
    limit = usable_cpu_count() if threads in (None, 'auto', 0) else int(threads)
    if limit < 1:
        raise ValueError('blas_threads must be positive or auto')
    return threadpool_limits(limits=limit, user_api='blas')


def _mirror_lower(matrix):
    """Fill the upper triangle with O(n) scratch, avoiding n² index arrays.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Square writable matrix whose lower triangle contains the accumulated statistic; its upper triangle is filled in place using linear scratch.
    
    """
    for row in range(len(matrix)):
        matrix[row, row+1:] = matrix[row+1:, row]
    return matrix


def _add_pairwise_counts(counts, observed):
    """Add joint observation counts to a lower triangle in place.

    Parameters
    ----------
    counts : writable square ndarray, float32 or float64
        Lower-triangular count accumulator, in sample order.
    observed : bool ndarray, shape (n_samples, n_eligible_variants)
        Observation mask for component-eligible variants only.

    Returns
    -------
    None
        Updates ``counts`` using b - missing_i - missing_j + shared_missing.
        Complete blocks need no second matrix product. Missing blocks use sparse
        row products; no dense n-by-n product or observed-mask SYRK is formed.

    Notes
    -----
    Work is O(n² + nnz(M) + sum_j missing_j²), where M is the missing
    indicator. Scratch is O(nnz(M) + n), in addition to the input mask.
    """
    from scipy.sparse import csr_matrix
    n, width = observed.shape
    if not width:
        return
    if observed.all():
        for col in range(n):
            counts[col:, col] += width
        return
    missing = csr_matrix(~observed, dtype=np.int64)
    missing.sum_duplicates()
    per_sample = np.diff(missing.indptr).astype(np.int64, copy=False)
    transposed = missing.T.tocsc(copy=False)
    for col in range(n):
        counts[col:, col] += width - per_sample[col] - per_sample[col:]
        if per_sample[col]:
            shared = (missing[col:col+1] @ transposed).tocsr()
            keep = shared.indices >= col
            counts[shared.indices[keep], col] += shared.data[keep]


def _coded_components(X, fit, method, components, min_maf, dtype, allele_frequencies=None):
    """Fit per-marker moments, emitting one component and its mask at a time.
    
    Parameters
    ----------
    X : numpy.ndarray
        Animal-by-marker dosage block with observed hardcalls 0/1/2 and NaN for absent calls; the block remains in the requested arithmetic dtype.
    fit : one-dimensional boolean array
        Selected output animals used to estimate marker allele frequencies and component moments; held-out animals never contribute fitting information.
    method : str
        Estimator identifier: 'npplink' uses sample-SD pairwise n-minus-one normalization; 'gcta0' uses per-marker HWE scaling; 'van' and 'vitezica' use global HWE weights.
    components : sequence of str
        Nonempty unique ordered selection of 'additive' and/or 'dominance'; output matrices and eligible marker counts follow this component order.
    min_maf : float
        Minimum fitted minor allele frequency, in [0, 0.5); allele-monomorphic markers are always excluded even when this threshold is zero.
    dtype : NumPy floating dtype
        Working arithmetic precision for component blocks and numerator accumulation; float32 selects ssyrk and float64 selects dsyrk. Count arithmetic remains float32.
    
    """
    valid = np.isfinite(X)
    fitted = X[fit]
    observed = valid[fit]
    count = observed.sum(axis=0)
    sums = np.where(observed, fitted, 0).sum(axis=0, dtype=np.float64)
    p = np.divide(sums, 2*count, out=np.full(X.shape[1], np.nan), where=count > 0)
    if allele_frequencies is not None:
        p = np.asarray(allele_frequencies, dtype=np.float64)
        if p.shape != (X.shape[1],) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('allele_frequencies must be a finite [0,1] vector matching each block')
    keep = np.isfinite(p) & (p > 0) & (p < 1) & (np.minimum(p, 1-p) >= min_maf)
    X = X[:, keep]
    valid = valid[:, keep]
    p = p[keep]
    q = 1-p
    for comp in components:
        if comp == 'additive':
            code = X.copy()
            center = 2*p
            scale = np.sqrt(2*p*q) if method == 'gcta0' else np.ones(len(p))
        else:
            pp = np.asarray(p, dtype=dtype)
            qq = np.asarray(q, dtype=dtype)
            code = np.where(X == 0, -2*pp**2,
                            np.where(X == 1, 2*pp*qq,
                                     np.where(X == 2, -2*qq**2, 0))).astype(dtype, copy=False)
            center = np.zeros(len(p))
            scale = 2*p*q if method == 'gcta0' else np.ones(len(p))
        eligible = np.ones(len(p), bool)
        if method == 'npplink':
            fitted_code = np.where(valid[fit], code[fit], 0)
            cnt = valid[fit].sum(axis=0)
            center = np.divide(fitted_code.sum(axis=0, dtype=np.float64), cnt,
                               out=np.zeros(len(p)), where=cnt > 0)
            # Reduce in double precision; never expand the block to float64.
            fitted_code -= np.asarray(center, dtype=dtype)
            fitted_code[~valid[fit]] = 0
            ss = np.einsum('ij,ij->j', fitted_code, fitted_code, dtype=np.float64)
            scale = np.sqrt(np.divide(ss, cnt-1, out=np.zeros(len(p)), where=cnt > 1))
            eligible = (cnt > 1) & (scale > 1e-12)
        code -= np.asarray(center, dtype=dtype)
        code /= np.asarray(np.where(scale > 0, scale, 1), dtype=dtype)
        mask = valid & eligible[None, :]
        code[~mask] = 0
        # Ineligible columns must not be counted as all-missing admitted markers.
        yield (comp, np.asfortranarray(code[:, eligible]), mask[:, eligible],
               int(keep.sum()), int(eligible.sum()), p[eligible])


def grm_from_blocks(blocks, n_samples, *, method, components, fit_mask=None,
                    min_maf=1e-4, dtype=np.float32, output_dtype=np.float32,
                    blas_threads='auto', return_statistics_only=False,
                    allele_frequencies=None):
    """Compute kernels, weights and raw statistics from bounded dosage blocks.
    
    Complexity: O(n²m) arithmetic and O(n² + n*b) memory. Float32 pair
    counts are exact for at most 2**24 admitted markers; larger runs fail.
    Nonpositive pair denominators produce NaN, with zero normalization weight.
    
    Parameters
    ----------
    blocks : iterable of numpy.ndarray
        Iterator yielding bounded animal-by-marker dosage blocks in identical sample order; fixture or packed reader controls block creation without concatenating the genome.
    n_samples : int
        Number of selected output animals, defining both square matrix dimensions and the required row count of every input block.
    method : str
        Estimator identifier: 'npplink' uses sample-SD pairwise n-minus-one normalization; 'gcta0' uses per-marker HWE scaling; 'van' and 'vitezica' use global HWE weights.
    components : sequence of str
        Nonempty unique ordered selection of 'additive' and/or 'dominance'; output matrices and eligible marker counts follow this component order.
    fit_mask : one-dimensional boolean array or None
        Mask aligned to selected sample order for fitting allele frequencies and empirical moments; None selects all output animals and an empty selection is rejected. Default: ``None``.
    min_maf : float
        Minimum fitted minor allele frequency, in [0, 0.5); allele-monomorphic markers are always excluded even when this threshold is zero. Default: ``0.0001``.
    dtype : NumPy floating dtype
        Working arithmetic precision for component blocks and numerator accumulation; float32 selects ssyrk and float64 selects dsyrk. Count arithmetic remains float32. Default: ``np.float32``.
    output_dtype : NumPy floating dtype
        Output kernel precision, float32 or float64 for the BED interface; widening an ssyrk result does not recover double-precision accumulation accuracy. Default: ``np.float32``.
    blas_threads : int or str
        Positive scoped BLAS worker limit, or 'auto' for every CPU available to the process; the caller's thread settings are restored after computation. Default: ``'auto'``.
    return_statistics_only : bool
        Internal streaming adapters can request raw pairwise numerators/counts
        without allocating normalized kernels or weights. Only pairwise methods
        support this option. The first two returned mappings are empty.
    allele_frequencies : one-dimensional array, optional
        External counted-allele frequencies in complete input variant order.
        Validated in [0,1], sliced per block, filtered before coding. None fits
        frequencies only on fit_mask rows. npplink still uses empirical SDs.
    
    """
    method, components = validate_grm_method(method, components)
    n = int(n_samples)
    if n < 1:
        raise ValueError('n_samples must be positive')
    fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if fit.shape != (n,) or not fit.any():
        raise ValueError('fit_mask must select at least one requested animal')
    if not 0 <= float(min_maf) < .5:
        raise ValueError('min_maf must be in [0, .5)')
    dtype = np.dtype(dtype)
    output_dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')) or output_dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype and output_dtype must be float32 or float64')
    syrk = ssyrk if dtype == np.float32 else dsyrk
    pairwise = method in {'npplink', 'gcta0'}
    if return_statistics_only and not pairwise:
        raise ValueError('raw-statistics mode requires a pairwise GRM method')
    nums = {c: np.zeros((n, n), dtype=dtype, order='F') for c in components}
    counts = {c: np.zeros((n, n), dtype=np.float32, order='F') for c in components} if pairwise else {}
    scalar_weights = dict.fromkeys(components, 0.)
    used_components = dict.fromkeys(components, 0)
    seen = used = 0
    af = None if allele_frequencies is None else np.asarray(allele_frequencies, dtype=np.float64)
    if af is not None and (af.ndim != 1 or not np.isfinite(af).all() or np.any((af < 0) | (af > 1))):
        raise ValueError('allele_frequencies must be a finite [0,1] vector')
    with _thread_context(blas_threads):
        for block in blocks:
            X = np.asarray(block, dtype=dtype)
            if X.ndim != 2 or X.shape[0] != n:
                raise ValueError('genotype blocks must have shape (n_samples, n_variants)')
            if np.isinf(X).any() or np.any((X[np.isfinite(X)] < 0) | (X[np.isfinite(X)] > 2)):
                raise ValueError('dosages must be in [0, 2], with NaN for missing calls')
            if 'dominance' in components and np.any(np.isfinite(X) & (X != np.rint(X))):
                raise ValueError('dominance coding requires hardcalls 0/1/2 or NaN, not fractional dosages')
            block_af = None if af is None else af[seen:seen+X.shape[1]]
            if block_af is not None and len(block_af) != X.shape[1]:
                raise ValueError('allele_frequencies is shorter than the input variant stream')
            seen += X.shape[1]
            if pairwise and seen > 2**24:
                raise ValueError('pairwise float32 counts support at most 2**24 variants')
            kept = 0
            for comp, code, mask, kept, eligible, p in _coded_components(
                    X, fit, method, components, float(min_maf), dtype, block_af):
                if not eligible:
                    continue
                nums[comp] = syrk(1., code, beta=1., c=nums[comp], lower=1, overwrite_c=1)
                used_components[comp] += eligible
                if pairwise:
                    _add_pairwise_counts(counts[comp], mask)
                else:
                    hwe = 2*p*(1-p)
                    scalar_weights[comp] += float(np.sum(hwe if comp == 'additive' else hwe*hwe))
            used += kept
    if af is not None and len(af) != seen:
        raise ValueError('allele_frequencies must match the complete input variant stream')
    if used == 0:
        raise ValueError('no polymorphic variants remain')
    kernels = {}
    weights = {}
    for comp in components:
        num = _mirror_lower(nums[comp])
        if pairwise:
            count = _mirror_lower(counts[comp])
            if return_statistics_only:
                continue
            weight = np.maximum(count-1, 0) if method == 'npplink' else count
            weights[comp] = weight
            kernel = np.full((n, n), np.nan, dtype=output_dtype)
            np.divide(num, weight, out=kernel, where=weight > 0)
        else:
            weights[comp] = scalar_weights[comp]
            if scalar_weights[comp] <= 0:
                raise ValueError(f'no positive normalization weight for {comp}')
            kernel = (num / scalar_weights[comp]).astype(output_dtype)
        kernels[comp] = kernel
    diag = {'n_variants_total': seen, 'n_variants_used': used,
            'n_variants_used_by_component': used_components, 'method': method,
            'normalization': {'npplink': 'empirical_sample_SD_pairwise_n_minus_1',
                              'gcta0': 'HWE_standardized_pairwise_n',
                              'van': 'vanraden_I_global_HWE', 'vitezica': 'global_HWE'}[method],
            'missingness': 'pairwise_component_eligible' if pairwise else 'mean_imputed'}
    if pairwise:
        diag['pairwise_observed_counts_by_component'] = counts
        diag['pairwise_observed_counts'] = counts[components[0]]
        diag['numerators'] = nums
    return kernels, weights, diag


def packed_grm_from_bed(prefix, *, method, components, sample_ids=None, fit_mask=None,
                        chromosome=None, min_maf=1e-4, chunk_variants='auto',
                        memory_fraction=None, maxmem=None, output_dtype=np.float32,
                        blas_threads='auto', progress=False, dosage_scale=None,
                        compute_dtype=np.float32):
    """Read only the current packed BED block, with an incremental RAM ceiling.
    
    The cap reserves all returned n² statistics and conservative block scratch;
    metadata, mapped BED pages, Python and library overhead are additional RSS.
    
    Parameters
    ----------
    prefix : str or pathlib.Path
        PLINK1 fileset prefix with matching BED, BIM and FAM files; BED is mapped read-only and must use SNP-major two-bit encoding.
    method : str
        Estimator identifier: 'npplink' uses sample-SD pairwise n-minus-one normalization; 'gcta0' uses per-marker HWE scaling; 'van' and 'vitezica' use global HWE weights.
    components : sequence of str
        Nonempty unique ordered selection of 'additive' and/or 'dominance'; output matrices and eligible marker counts follow this component order.
    sample_ids : sequence of str or None
        Unique requested FAM individual identifiers in output row order; None retains all source animals. Fitting masks refer to this selected order. Default: ``None``.
    fit_mask : one-dimensional boolean array or None
        Mask aligned to selected sample order for fitting allele frequencies and empirical moments; None selects all output animals and an empty selection is rejected. Default: ``None``.
    chromosome : str or int or None
        Exact BIM chromosome label selected as text; None selects every source variant for the in-memory builder, without interpreting an All pseudo-chromosome. Default: ``None``.
    min_maf : float
        Minimum fitted minor allele frequency, in [0, 0.5); allele-monomorphic markers are always excluded even when this threshold is zero. Default: ``0.0001``.
    chunk_variants : positive int or str
        Requested upper bound on variants decoded per block; 'auto' uses everything the memory budget affords after reserving the GRM matrices; explicit values are ceilings. Default: ``'auto'``.
    memory_fraction : float or None
        Fraction of currently available RAM used when maxmem is absent, strictly above zero and at most 0.95; dense numerical matrices and scratch share the budget. Default: ``None`` (80% of physical RAM).
    maxmem : int, unit string or None
        Incremental numerical working-memory ceiling in bytes or binary units such as 8GiB; None uses memory_fraction. Metadata, mapped pages and library/I/O RSS are additional. Default: ``None``.
    output_dtype : NumPy floating dtype
        Output kernel precision, float32 or float64 for the BED interface; widening an ssyrk result does not recover double-precision accumulation accuracy. Default: ``np.float32``.
    blas_threads : int or str
        Positive scoped BLAS worker limit, or 'auto' for every CPU available to the process; the caller's thread settings are restored after computation. Default: ``'auto'``.
    progress : bool
        Whether to report completed packed variant blocks during computation; disabling progress leaves the numerical results and block-size plan identical. Default: ``False``.
    dosage_scale : one-dimensional array or None
        Optional per-selected-animal dosage multiplier applied to each decoded block for confidently detected male-X recoding; None retains original BED dosages. Default: ``None``.
    compute_dtype : NumPy floating dtype
        Precision of centered blocks and in-place SYRK accumulation. Float32
        is the default; float64 explicitly requests double-precision arithmetic.
    
    """
    from .packed_bed import _metadata, _resolve_sample_indices, _decode_selected_packed
    from .resources import working_memory_bytes
    from .progress import StageProgress
    method, components = validate_grm_method(method, components)
    fam, bim = _metadata(prefix)
    if not fam.iid.astype(str).is_unique:
        raise ValueError('FAM IID entries must be unique')
    sel, ids = _resolve_sample_indices(fam, sample_ids)
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('sample_ids must be nonempty and unique')
    n, n_full = len(sel), len(fam)
    bpv = (n_full+3)//4
    chrom = bim.chrom.astype(str).to_numpy()
    indices = np.arange(len(bim)) if chromosome is None else np.flatnonzero(chrom == str(chromosome))
    if not len(indices):
        raise ValueError(f'no variants found for chromosome {chromosome!r}')
    if len(indices) > 2**24:
        raise ValueError('pairwise float32 counts support at most 2**24 variants')
    if np.dtype(output_dtype) not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('output_dtype must be float32 or float64')
    compute_dtype = np.dtype(compute_dtype)
    if compute_dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('compute_dtype must be float32 or float64')
    budget = working_memory_bytes(maxmem, memory_fraction)
    pairwise = method in {'npplink', 'gcta0'}
    matrix_bytes = n*n*(len(components)*((8 if pairwise else 0)+compute_dtype.itemsize+np.dtype(output_dtype).itemsize)
                        + int(pairwise))  # finalization validity-mask scratch
    # X, component coding/masks, fitting and BLAS layout copies, packed gather.
    per_variant = (80 + 12*max(0, compute_dtype.itemsize-4))*n + 2*bpv + 512
    remaining = budget - matrix_bytes
    affordable = remaining // per_variant
    if affordable < 1:
        raise MemoryError(f'GRM matrices need {matrix_bytes} bytes plus at least {per_variant} block bytes; budget is {budget}')
    # Each block costs one n x n SYRK per component plus O(n^2) fixed overhead, so the
    # block should be as wide as memory allows; explicit requests are ceilings.
    ceiling = affordable if str(chunk_variants).lower() == 'auto' else int(chunk_variants)
    if ceiling < 1:
        raise ValueError('variant chunk must be positive or auto')
    bs = min(ceiling, affordable, len(indices))
    from .io import open_bed_memmap
    mm = open_bed_memmap(prefix, n_samples=n_full, n_variants=len(bim))
    bar = StageProgress(total=(len(indices)+bs-1)//bs, desc=f'Packed BED {method} GRM',
                        enabled=progress, unit='block')
    def blocks():
        """Yield bounded decoded BED blocks and count completed block work."""
        for start in range(0, len(indices), bs):
            gi = indices[start:start+bs]
            # Contiguous genomic runs use a mapped view; sparse selections gather one block.
            P = mm[gi[0]:gi[-1]+1] if gi[-1]-gi[0]+1 == len(gi) else mm[gi]
            X = _decode_selected_packed(P, n_full, sel, np.float32)
            if dosage_scale is not None:
                X *= np.asarray(dosage_scale, np.float32)[:, None]
            yield X
            bar.update(1)
    try:
        kernels, weights, diag = grm_from_blocks(blocks(), n, method=method,
            components=components, fit_mask=fit_mask, min_maf=min_maf,
            dtype=compute_dtype, output_dtype=output_dtype, blas_threads=blas_threads)
    finally:
        bar.close()
        mm._mmap.close()
    import xarray as xr
    coords = {'sample_i': ids, 'sample_j': ids}
    out = {c: xr.DataArray(k, dims=('sample_i', 'sample_j'), coords=coords)
           for c, k in kernels.items()}
    diag.update(chromosome=None if chromosome is None else str(chromosome),
                chunk_variants=int(bs), matrix_bytes=int(matrix_bytes),
                planned_working_bytes=int(matrix_bytes+bs*per_variant),
                memory_budget_bytes=int(budget), storage='in_memory',
                source_representation='plink1_bed_2bit')
    return {'kernels': out, 'weights': weights, 'sample_ids': ids, 'diagnostics': diag}
