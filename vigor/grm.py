"""Additive genomic relatedness matrices with LOCO decomposition.

Exact mode uses a pairwise-missingness-aware GRM definition, while
yielding one chromosome eigensystem at a time.  Randomized mode never forms an
``n x n`` GRM: it applies the mean-imputed standardized genotype operator in
blocks, estimates a retained eigenspace, and represents omitted directions by
a trace-matched bulk eigenvalue.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.linalg import eigh, blas
from sklearn.utils.extmath import randomized_svd

from ._utils import standardize_block

CACHE_FORMAT_VERSION = "vigor-loco-v2-signed-eigen-eligible-counts"


def _eig(G, n_components=None, decomposition="exact", random_state=0):
    """Symmetric PSD eigensystem ordered from largest to smallest.

    Negative eigenvalues are clipped to zero, never reflected with ``abs``.
    Material negative eigenvalues trigger a warning because pairwise missingness
    normalization may make a GRM indefinite. This is an explicit PSD projection.
    Dense exact work is O(n³), with O(n²) storage; randomized work is O(n²r).

    Parameters
    ----------
    G : array-like or covariance specification
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``.

    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver. Default: ``'exact'``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``randomized_svd`` as ``random_state``.
    """
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or not len(G):
        raise ValueError("G must be a nonempty square matrix")
    if not np.isfinite(G).all() or not np.allclose(G, G.T, rtol=1e-5, atol=1e-7):
        raise ValueError("G must be finite and symmetric")
    n = G.shape[0]
    if n_components is not None and (isinstance(n_components, (bool, np.bool_)) or int(n_components) != n_components):
        raise ValueError("n_components must be a positive integer or None")
    rank = n if n_components is None else min(int(n_components), n)
    if rank < 1:
        raise ValueError("n_components must be positive")
    def psd_values(values):
        tolerance = 32*np.finfo(G.dtype if G.dtype.kind == 'f' else float).eps*max(1., float(np.max(np.abs(values))))
        if np.min(values) < -tolerance:
            warnings.warn("GRM has negative eigenvalues; applying a zero-clipped PSD projection",
                          RuntimeWarning, stacklevel=3)
        return np.maximum(values, 0)
    if decomposition == "exact":
        if rank >= n:
            s, U = np.linalg.eigh(G)
        else:
            s, U = eigh(G, subset_by_index=(n - rank, n - 1), check_finite=False)
        order = np.argsort(s)[::-1]
        return U[:, order], psd_values(s[order])
    if decomposition != "randomized":
        raise ValueError("decomposition must be 'exact' or 'randomized'")
    U, _, _ = randomized_svd(
        G, n_components=rank, random_state=random_state, flip_sign=True
    )
    # Singular values cannot encode negative eigenvalues. A Ritz eigenproblem
    # restores signs in the retained subspace before the explicit PSD projection.
    reduced = U.T @ G @ U
    s, rotation = np.linalg.eigh((reduced+reduced.T)*0.5)
    order = np.argsort(s)[::-1]
    return U @ rotation[:, order], psd_values(s[order])


def _resolve_decomposition(decomposition, n, n_components):
    """Validate a decomposition policy and choose the retained animal eigensystem rank.

    Parameters
    ----------
    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver.

    n : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit. Source comparisons: ``n <= 4000``. See the
        rejection/fallback branches for enforcement.

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
    """
    value = decomposition.lower()
    if value not in {"auto", "exact", "randomized"}:
        raise ValueError("decomposition must be 'auto', 'exact', or 'randomized'")
    if value == "auto":
        value = "exact" if n <= 4000 else "randomized"
    if n_components is None:
        n_components = n if value == "exact" else min(2000, max(n - 1, 1))
    rank = min(int(n_components), n)
    if rank < 1:
        raise ValueError("n_components must be positive")
    return value, rank


def _cache_directory(grm_cache, reader):
    """Cache directory.

    Parameters
    ----------
    grm_cache : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported.

    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.
    """
    if not grm_cache:
        return None
    if grm_cache is True:
        return Path(reader.prefix).resolve().parent / ".domino_grm_cache"
    return Path(grm_cache).resolve()


def _cache_key(
    reader,
    sample_index,
    decomposition,
    rank,
    dtype,
    random_state,
    n_iter,
    adaptive_rank,
    max_adaptive_components,
):
    """Cache key.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    n_iter : int
        Number of alternating/projection/relevance/eigenspace refinement iterations performed by
        the selected algorithm. Positive counts trade additional work for a more settled
        approximation.

    adaptive_rank : bool or int
        Adaptive eigenspace expansion control: adaptive_rank enables expansion, while maximum
        component counts bound the retained animal basis.

    max_adaptive_components : bool or int
        Adaptive eigenspace expansion control: adaptive_rank enables expansion, while maximum
        component counts bound the retained animal basis.
    """
    digest = hashlib.sha256()
    digest.update(CACHE_FORMAT_VERSION.encode("ascii"))
    normalization = (
        "pairwise_observed_standardized" if decomposition == "exact"
        else "mean_imputed_standardized_bulk"
    )
    digest.update(normalization.encode("ascii"))
    for extension in ("bed", "bim", "fam"):
        path = Path(f"{reader.prefix}.{extension}").resolve()
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    digest.update(np.asarray(sample_index, dtype=np.int64).tobytes())
    digest.update(
        f"{decomposition}:{rank}:{np.dtype(dtype).name}:{random_state}:{n_iter}:"
        f"{adaptive_rank}:{max_adaptive_components}".encode("ascii")
    )
    return digest.hexdigest()[:24]


def _cache_paths(cache_dir, key, chrom):
    """Cache paths.

    Parameters
    ----------
    cache_dir : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported.

    key : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(chrom))
    stem = cache_dir / f"{key}.chr{safe}"
    return Path(f"{stem}.U.npy"), Path(f"{stem}.s.npy"), Path(f"{stem}.json")


def _atomic_save_json(path, value):
    """Persist JSON atomically by replacing a completed temporary file.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    value : object (see the cited input contract)
        Input for the operation ``temporary.write_text(json.dumps(value, indent=2),
        encoding='utf-8')``. See ``vigor.grm._atomic_save_json`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_cached(cache_dir, key, chrom):
    """Load cached.

    Parameters
    ----------
    cache_dir : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported.

    key : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    if cache_dir is None:
        return None
    upath, spath, mpath = _cache_paths(cache_dir, key, chrom)
    zpath=Path(str(mpath)[:-5]+'.zarr')
    if zpath.exists():
        from .store import VigorStore
        st=VigorStore(zpath,mode='r')
        if st.attrs.get('domino_write_state')!='complete':return None
        with st.open_dataset('eigensystem',lazy=False) as ds:
            metadata=dict(ds.attrs)
            return {'U':ds['U'].values,'s':ds['s'].values,
                    'residual_eigenvalue':float(metadata.get('residual_eigenvalue',0)),
                    'diagnostics':metadata.get('diagnostics',{}),'cache_hit':True,
                    'cache_key':metadata.get('cache_key'),'decomposition':metadata['decomposition']}
    if not (upath.exists() and spath.exists() and mpath.exists()):
        return None
    metadata = json.loads(mpath.read_text(encoding="utf-8"))
    return {
        "U": np.load(upath, mmap_mode="r", allow_pickle=False),
        "s": np.load(spath, mmap_mode="r", allow_pickle=False),
        "residual_eigenvalue": float(metadata.get("residual_eigenvalue", 0.0)),
        "diagnostics": metadata.get("diagnostics", {}),
        "cache_hit": True,
        "cache_key": metadata.get("cache_key"),
        "decomposition": metadata["decomposition"],
    }


def _save_cached(cache_dir, key, chrom, result):
    """Persist cached.

    Parameters
    ----------
    cache_dir : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported.

    key : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required. Passed to
        ``xr.Dataset`` as ``attrs``.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported. Passed to ``xr.Dataset`` as ``attrs``.

    result : mapping or fitted object
        Previously prepared model/result or solver state consumed by this operation. Use the
        object returned by the corresponding fitting/preparation routine with the same animal
        and trait axes. Passed to ``xr.Dataset`` as ``attrs``.
    """
    if cache_dir is None:
        return result
    cache_dir.mkdir(parents=True, exist_ok=True)
    upath, spath, mpath = _cache_paths(cache_dir, key, chrom)
    import xarray as xr
    from .store import VigorStore
    zpath=Path(str(mpath)[:-5]+'.zarr')
    ds=xr.Dataset({'U':(('sample','factor'),np.asarray(result['U'])),
                   's':(('factor',),np.asarray(result['s']))},attrs={
                       'chromosome':str(chrom),'cache_format_version':CACHE_FORMAT_VERSION,
                       'cache_key':result.get('cache_key',key),'decomposition':result['decomposition'],
                       'residual_eigenvalue':float(result.get('residual_eigenvalue',0)),
                       'diagnostics':result.get('diagnostics',{})})
    VigorStore(zpath,mode='w').write_dataset(ds,'eigensystem')
    loaded = _load_cached(cache_dir, key, chrom)
    if loaded is None:
        return result
    loaded["cache_hit"] = False
    return loaded


def _chromosome_seed(random_state, chrom):
    """Return the chromosome seed result computed by ``int``.

    Parameters
    ----------
    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.
    """
    digest = hashlib.sha256(str(chrom).encode("utf-8")).digest()
    return int((int(random_state) + int.from_bytes(digest[:4], "little")) % (2 ** 32))


def _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
    """Iterate over excluding chrom.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Passed to
        ``reader.iter_blocks`` as ``sample_index``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Passed to ``standardize_block`` as ``dtype``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    for bim_slice, genotype in reader.iter_blocks(block_size, sample_index=sample_index):
        keep = bim_slice["chrom"].to_numpy() != str(chrom)
        if not np.any(keep):
            continue
        z, _, _ = standardize_block(genotype[:, keep], dtype=dtype)
        yield z


def _apply_streamed_grm(reader, chrom, sample_index, block_size, X, dtype):
    """Apply streamed genomic relationship matrix.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.

    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Passed to ``np.zeros`` as ``dtype``.
    """
    result = np.zeros((len(sample_index), X.shape[1]), dtype=dtype)
    count = 0
    for z in _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
        result += z @ (z.T @ X)
        count += z.shape[1]
    if count == 0:
        raise ValueError(f"no variants remain after leaving out chromosome {chrom}")
    result /= count
    return result, count


def _randomized_loco(
    reader,
    chrom,
    sample_index,
    block_size,
    rank,
    dtype,
    random_state,
    n_iter,
    operator_error_tolerance,
):
    """Randomized eigensystem from streamed GRM operator products.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked.

    rank : int, str, sequence or mapping
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed.

    n_iter : int
        Number of alternating/projection/relevance/eigenspace refinement iterations performed by
        the selected algorithm. Positive counts trade additional work for a more settled
        approximation.

    operator_error_tolerance : object (see the cited input contract)
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller.
    """
    n = len(sample_index)
    oversample = min(32, max(8, rank // 10), max(n - rank, 0))
    q = min(n, rank + oversample)
    rng = np.random.default_rng(random_state)
    omega = rng.standard_normal((n, q)).astype(dtype, copy=False)
    projected, count = _apply_streamed_grm(
        reader, chrom, sample_index, block_size, omega, dtype
    )
    Q, _ = np.linalg.qr(projected, mode="reduced")
    del omega, projected
    for _ in range(max(int(n_iter), 0)):
        projected, _ = _apply_streamed_grm(
            reader, chrom, sample_index, block_size, Q, dtype
        )
        Q, _ = np.linalg.qr(projected, mode="reduced")
        del projected

    B = np.zeros((q, q), dtype=np.float64)
    trace = 0.0
    observed_count = 0
    for z in _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
        ztq = z.T @ Q
        B += np.asarray(ztq.T @ ztq, dtype=np.float64)
        trace += float(np.sum(z * z, dtype=np.float64))
        observed_count += z.shape[1]
    B /= observed_count
    trace /= observed_count
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1][:rank]
    s = np.maximum(values[order], 0.0)
    U = np.asarray(Q @ vectors[:, order], dtype=dtype)
    omitted = max(n - rank, 0)
    residual = max((trace - float(s.sum())) / omitted, 0.0) if omitted else 0.0

    probes = min(8, n)
    probe = rng.standard_normal((n, probes)).astype(dtype, copy=False)
    exact_probe, _ = _apply_streamed_grm(
        reader, chrom, sample_index, block_size, probe, dtype
    )
    utp = U.T @ probe
    approx_probe = U @ ((s - residual)[:, None] * utp) + residual * probe
    denominator = max(float(np.linalg.norm(exact_probe)), np.finfo(float).tiny)
    operator_error = float(np.linalg.norm(exact_probe - approx_probe) / denominator)
    diagnostics = {
        "operator_relative_error": operator_error,
        "operator_error_tolerance": operator_error_tolerance,
        "rank": int(rank),
        "oversampling": int(q - rank),
        "power_iterations": int(n_iter),
        "variants_in_loco_grm": int(count),
        "trace": float(trace),
        "retained_trace_fraction": float(s.sum() / trace) if trace > 0 else 0.0,
    }
    return {
        "U": U,
        "s": s,
        "residual_eigenvalue": residual,
        "diagnostics": diagnostics,
        "cache_hit": False,
        "decomposition": "randomized",
    }


def iter_loco_grms(
    reader,
    sample_index=None,
    block_size='auto',
    n_components=None,
    chroms=None,
    decomposition="exact",
    compute_dtype="float32",
    grm_cache=None,
    random_state=0,
    randomized_power_iterations=1,
    operator_error_tolerance=0.02,
    adaptive_randomized_rank=True,
    randomized_max_components=None,
    verbose=True,
    maxmem=None,
    memory_fraction=None,
):
    """Yield ``(chromosome, eigensystem)`` without retaining all chromosomes.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Default:
        ``None``.

    block_size : int or 'auto'
        Positive number of genotype variants decoded per block. Exact mode accumulates
        pairwise-observed cross-products/counts; randomized mode applies mean-imputed
        standardized genotype blocks without forming the full animal kernel. Default:
        ``'auto'`` (sized from the memory budget; integers are ceilings).

    n_components : int or None
        Number of retained latent components or feature directions. Must fit the available
        matrix dimensions; None delegates selection to the decomposition helper when supported.
        Default: ``None``.

    chroms : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine. Default: ``None``.

    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver. Default: ``'exact'``.

    compute_dtype : str or NumPy dtype
        Arithmetic dtype: float32 and float64 are the accepted floating precisions; float32 is
        the default. float32 halves dense kernel memory; float64 reduces rounding. iter_loco_grms
        validates this closed set. Default: ``'float32'``.

    grm_cache : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported. Default: ``None``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    randomized_power_iterations : int
        Number of alternating/projection/relevance/eigenspace refinement iterations performed by
        the selected algorithm. Positive counts trade additional work for a more settled
        approximation. Default: ``1``.

    operator_error_tolerance : float
        Nonnegative relative operator-probe error threshold for accepting a randomized retained
        eigenspace. When adaptive_randomized_rank is enabled, exceeding it increases rank up to
        randomized_max_components; inspect diagnostics even when the cap is reached. Default:
        ``0.02``.

    adaptive_randomized_rank : bool
        Allow rank expansion when randomized operator-probe error exceeds
        operator_error_tolerance. False keeps the initial randomized rank; exact decompositions
        bypass this approximation control. Default: ``True``.

    randomized_max_components : bool or int
        Adaptive eigenspace expansion control: adaptive_rank enables expansion, while maximum
        component counts bound the retained animal basis. Default: ``None``. None delegates to
        the local fallback shown in this body; it is not an additional numeric/category value.

    verbose : bool
        Whether to emit progress/status reporting during this operation. False disables the
        corresponding reporting path. Default: ``True``.

    maxmem : int, float, unit string or None
        Working-memory budget for sizing ``block_size='auto'``; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    dtype = np.dtype(compute_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("compute_dtype must be float32 or float64")
    sidx = (
        np.arange(reader.n_samples, dtype=np.int64)
        if sample_index is None
        else np.asarray(sample_index, dtype=np.int64)
    )
    n = len(sidx)
    method, rank = _resolve_decomposition(decomposition, n, n_components)
    chrom_arr = reader.bim["chrom"].to_numpy()
    run_chroms = (
        [str(chrom) for chrom in chroms]
        if chroms is not None
        else list(dict.fromkeys(chrom_arr))
    )
    cache_dir = _cache_directory(grm_cache, reader)
    if randomized_max_components is None:
        randomized_max_components = min(n, max(rank, 4 * rank))
    else:
        randomized_max_components = min(n, int(randomized_max_components))
    if randomized_max_components < rank:
        raise ValueError("randomized_max_components must be at least n_components")
    key = _cache_key(
        reader,
        sidx,
        method,
        rank,
        dtype,
        random_state,
        randomized_power_iterations,
        adaptive_randomized_rank,
        randomized_max_components,
    )
    cached = {chrom: _load_cached(cache_dir, key, chrom) for chrom in run_chroms}
    from .resources import resolve_variant_chunk
    # per call in dtype: genotype, z, mask/keep copy, z*z temporaries + reader decode scratch (~16);
    # exact mode reserves the four n x n accumulators (all/chrom cross-products and counts)
    block_size = resolve_variant_chunk(
        block_size, n, bytes_per_call=4 * dtype.itemsize + 16, maxmem=maxmem,
        memory_fraction=memory_fraction, n_variants=getattr(reader, "n_variants", None),
        reserve_bytes=0 if method == "randomized" or all(v is not None for v in cached.values())
        else 2 * n * n * (dtype.itemsize + 4))

    if method == "randomized":
        for chrom in run_chroms:
            result = cached[chrom]
            if result is None:
                attempted_ranks = []
                attempted_errors = []
                attempted_rank = rank
                while True:
                    result = _randomized_loco(
                        reader,
                        chrom,
                        sidx,
                        block_size,
                        attempted_rank,
                        dtype,
                        _chromosome_seed(random_state, chrom),
                        randomized_power_iterations,
                        operator_error_tolerance,
                    )
                    error = result["diagnostics"]["operator_relative_error"]
                    attempted_ranks.append(int(attempted_rank))
                    attempted_errors.append(float(error))
                    if operator_error_tolerance is None or error <= operator_error_tolerance:
                        break
                    next_rank = min(randomized_max_components, n, attempted_rank * 2)
                    if not adaptive_randomized_rank or next_rank <= attempted_rank:
                        raise RuntimeError(
                            f"randomized LOCO operator error for chromosome {chrom} is "
                            f"{error:.4f}, above tolerance {operator_error_tolerance:.4f}; "
                            f"attempted ranks {attempted_ranks}"
                        )
                    attempted_rank = next_rank
                result["diagnostics"]["attempted_ranks"] = attempted_ranks
                result["diagnostics"]["attempted_operator_errors"] = attempted_errors
                result["cache_key"] = key
                result = _save_cached(cache_dir, key, chrom, result)
            error = result.get("diagnostics", {}).get("operator_relative_error")
            if (
                operator_error_tolerance is not None
                and error is not None
                and error > operator_error_tolerance
            ):
                raise RuntimeError(
                    f"cached randomized LOCO operator error for chromosome {chrom} is "
                    f"{error:.4f}, above tolerance {operator_error_tolerance:.4f}; "
                    "increase n_components or clear the incompatible cache entry"
                )
            if verbose:
                diagnostic = result.get("diagnostics", {})
                error = diagnostic.get("operator_relative_error")
                suffix = "cache hit" if result.get("cache_hit") else f"operator error {error:.4f}"
                print(f"  LOCO GRM chr {chrom}: rank {len(result['s'])}; {suffix}")
            yield chrom, result
        return

    from .grm_backend import grm_from_blocks
    def raw_statistics(variant_index=None):
        _, _, diag = grm_from_blocks(
            (block for _, block in reader.iter_blocks(block_size,
                variant_index=variant_index, sample_index=sidx)),
            n, method="npplink", components=("additive",), min_maf=0,
            dtype=dtype, output_dtype=dtype, return_statistics_only=True)
        return diag["numerators"]["additive"], diag["pairwise_observed_counts_by_component"]["additive"]

    missing = [chrom for chrom in run_chroms if cached[chrom] is None]
    zzt_all = w_all = None
    if missing:
        zzt_all, w_all = raw_statistics()

    for chrom in run_chroms:
        result = cached[chrom]
        idx_chrom = np.where(chrom_arr == chrom)[0]
        if result is None:
            zzt_chrom, w_chrom = raw_statistics(idx_chrom)
            np.subtract(zzt_all, zzt_chrom, out=zzt_chrom)
            np.subtract(w_all, w_chrom, out=w_chrom)
            np.maximum(w_chrom, 1.0, out=w_chrom)
            np.divide(zzt_chrom, w_chrom, out=zzt_chrom)
            U, s = _eig(zzt_chrom, n_components=rank, decomposition="exact")
            result = {
                "U": U,
                "s": s,
                "residual_eigenvalue": 0.0,
                "diagnostics": {"rank": int(rank), "variants_left_out": int(len(idx_chrom))},
                "cache_hit": False,
                "decomposition": "exact",
                "cache_key": key,
            }
            result = _save_cached(cache_dir, key, chrom, result)
        if verbose:
            suffix = "; cache hit" if result.get("cache_hit") else ""
            print(
                f"  LOCO GRM chr {chrom}: {len(idx_chrom)} variants left out, "
                f"rank {len(result['s'])}{suffix}"
            )
        yield chrom, result


class LazyLOCOResults(Mapping):
    """Dict-like LOCO result collection that keeps at most one chromosome resident.

    Indexing behaves like the historical ``dict`` returned by
    :func:`compute_loco_grms`, but each chromosome is computed/read from cache
    on demand. Iterating over ``items()`` is therefore peak-memory bounded by
    one eigensystem unless the caller explicitly wraps the object in ``dict``.
    """
    def __init__(self, reader, kwargs):
        """Initialize LazyLOCOResults with the inputs and resource settings below.

        Parameters
        ----------
        reader : PLINK reader or genotype source
            Opened genotype reader supplying sample metadata, variant metadata and block access. Its
            sample order defines the relationship matrix order.

        kwargs : mapping
            Additional keyword settings forwarded to the selected estimator/writer. See the called
            implementation for its accepted names and value domains; these are not unrestricted
            scientific options.
        """
        self.reader = reader
        self.kwargs = dict(kwargs)
        requested = self.kwargs.pop("chroms", None)
        available = [str(x) for x in getattr(reader, "chroms", reader.bim["chrom"].astype(str).unique())]
        self._keys = available if requested is None else [str(x) for x in requested]
        missing = [x for x in self._keys if x not in set(available)]
        if missing:
            raise KeyError(f"requested LOCO chromosome(s) absent from genotype source: {missing[:5]}")
        self._last_key = None
        self._last_value = None

    def __len__(self):
        """Return the number of entries in this collection.
        """
        return len(self._keys)

    def __iter__(self):
        """Iterate through the collection in its stored order.
        """
        return iter(self._keys)

    def __getitem__(self, chrom):
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        chrom : str or int or None
            Chromosome identifier matched against source metadata as a string. None leaves the
            chromosome unrestricted where supported.

        Returns
        -------
        object
            Computed result from ``value``; see the summary and the called implementation for its labeled/numerical representation.
        """
        key = str(chrom)
        if key not in self._keys:
            raise KeyError(chrom)
        if key == self._last_key:
            return self._last_value
        try:
            _, value = next(iter_loco_grms(self.reader, chroms=[key], **self.kwargs))
        except StopIteration as exc:
            raise KeyError(chrom) from exc
        self._last_key, self._last_value = key, value
        return value

    def materialize(self):
        """Explicitly retain all chromosomes in a normal dictionary.

        Returns
        -------
        object
            Computed result from ``{chrom: self[chrom] for chrom in self._keys}``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return {chrom: self[chrom] for chrom in self._keys}


def compute_loco_grms(reader, *, materialize_all=False, **kwargs):
    """Return dict-like lazy LOCO results; opt in to retaining every chromosome.

    The historical implementation eagerly built ``{chrom: eigensystem}`` and
    therefore multiplied peak memory by chromosome count.  The default now
    returns :class:`LazyLOCOResults`, which preserves ``grms[chrom]`` access
    while retaining only the most recently accessed decomposition. Set
    ``materialize_all=True`` only for small analyses that explicitly need a
    real dictionary.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    materialize_all : bool
        Whether to perform the named persistence/materialization/refitting step. True enables
        it; False follows the bypass path in this routine. Refitting during CV must use training
        outcomes only. Default: ``False``.

    **kwargs : dict
        Options forwarded to iter_loco_grms, including sample_index, block_size, n_components,
        chroms, decomposition, compute_dtype, grm_cache and randomized/error controls. See
        iter_loco_grms for exact defaults and supported values; materializing all chromosomes
        increases resident eigensystem memory.

    Returns
    -------
    LazyLOCOResults or dict
        Mapping chromosome -> eigensystem containing U, s, residual_eigenvalue and diagnostics.
        Lazy mode retains only the most recently accessed chromosome; materialize_all returns an
        eager dictionary.

    """
    lazy = LazyLOCOResults(reader, kwargs)
    return lazy.materialize() if materialize_all else lazy


def compute_grm(reader, sample_index=None, block_size='auto', compute_dtype="float32",
                maxmem=None, memory_fraction=None):
    """Genome-wide additive GRM (dense, pairwise missingness aware).

    Parameters
    ----------
    reader : PlinkReader
        Open PLINK1 reader whose BIM order defines the variant stream and FAM order the sample
        axis. The routine centers/scales each variant and divides pair cross-products by joint
        observed-variant counts, rather than treating missing calls as observed zeros.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Default:
        ``None``.

    block_size : int or 'auto'
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings).

    compute_dtype : str or NumPy dtype
        Arithmetic dtype: float32 and float64 are the accepted floating precisions; float32 is
        the default. float32 halves dense kernel memory; float64 reduces rounding. iter_loco_grms
        validates this closed set. Default: ``'float32'``.

    maxmem : int, float, unit string or None
        Working-memory budget for sizing ``block_size='auto'``; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Returns
    -------
    numpy.ndarray
        Dense sample × sample additive GRM, normalized by each animal pair's jointly observed
        variant count. Sample order follows sample_index or the reader FAM.

    """
    from .grm_backend import grm_from_blocks
    from .resources import resolve_variant_chunk
    sidx = (np.arange(reader.n_samples, dtype=np.int64) if sample_index is None
            else np.asarray(sample_index, dtype=np.int64))
    dtype = np.dtype(compute_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("compute_dtype must be float32 or float64")
    n = len(sidx)
    block_size = resolve_variant_chunk(block_size, n,
        bytes_per_call=80+12*max(0, dtype.itemsize-4), maxmem=maxmem,
        memory_fraction=memory_fraction, n_variants=getattr(reader, "n_variants", None),
        reserve_bytes=n*n*(2*dtype.itemsize+12))
    _, _, diagnostics = grm_from_blocks(
        (block for _, block in reader.iter_blocks(block_size, sample_index=sidx)),
        n, method="npplink", components=("additive",), min_maf=0,
        dtype=dtype, output_dtype=dtype, return_statistics_only=True)
    # Keep this historical API's empirical-SD / n_ij normalization. Public
    # method="npplink" deliberately uses n_ij-1; the raw engine is shared.
    numerator = diagnostics["numerators"]["additive"]
    counts = diagnostics["pairwise_observed_counts_by_component"]["additive"]
    np.divide(numerator, np.maximum(counts, 1), out=numerator)
    return numerator




def _kernel_correlations(A, D):
    """Pearson kernel diagnostics with float64 reductions and O(n) scratch."""
    def correlation(rows):
        count = 0
        sa = sd = saa = sdd = sad = 0.
        for a, d in rows:
            count += a.size
            sa += a.sum(dtype=np.float64); sd += d.sum(dtype=np.float64)
            saa += np.einsum("i,i->", a, a, dtype=np.float64)
            sdd += np.einsum("i,i->", d, d, dtype=np.float64)
            sad += np.einsum("i,i->", a, d, dtype=np.float64)
        if count < 2:
            return np.nan
        va, vd = saa-sa*sa/count, sdd-sd*sd/count
        return (sad-sa*sd/count)/np.sqrt(va*vd) if va > 0 and vd > 0 else np.nan
    return {
        "additive_dominance_kernel_correlation": correlation(zip(A, D)),
        "additive_dominance_offdiag_correlation": correlation((A[i,:i], D[i,:i]) for i in range(len(A))),
        "additive_dominance_diag_correlation": correlation([(np.diag(A), np.diag(D))]),
    }


def compute_genomic_grms_from_plink1(
    prefix,
    *,
    sample_ids=None,
    fit_mask=None,
    include=("additive", "dominance"),
    block_size="auto",
    maxmem=None,
    memory_fraction=None,
    min_maf=1e-4,
    dtype=np.float32,
    accum_dtype=np.float32,
    progress=False,
    return_diagnostics=False,
    use_syrk=True,
):
    """Stream additive and Vitezica dominance GRMs directly from PLINK1.

    The PLINK ``.bed`` file is decoded one variant block at a time; the full
    sample-by-variant genotype matrix is never materialized. Allele frequencies
    are estimated from ``fit_mask`` animals only and then used to code every
    requested animal, which makes the routine suitable for train/test genomic
    prediction. The normalization matches :func:`genomic_grms_from_genotypes`:

    ``K_A = A A' / sum_j 2 p_j q_j`` and
    ``K_D = D D' / sum_j (2 p_j q_j)^2``.

    Parameters
    ----------
    prefix : array-like or numeric scalar
        str or PlinkReader PLINK1 prefix (without extension) or an open :class:`PlinkReader`.

    sample_ids : array-like or numeric scalar
        sequence of str, optional Requested IID order. Defaults to all FAM rows. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    fit_mask : array-like or numeric scalar
        boolean array, optional Rows among ``sample_ids`` used to estimate allele frequencies.
        Defaults to all requested rows. Default: ``None``. None delegates to the local fallback
        shown in this body; it is not an additional numeric/category value.

    include : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default:
        ``('additive', 'dominance')``.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

    maxmem : int, float, unit string or None
        Working-memory ceiling in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
        memory_fraction. The resource helper caps explicit budgets against available RAM and
        shares the limit across workers. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
        < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
        peak per-worker allocation. Default: ``None`` (80% of physical RAM).

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``PlinkReader`` as ``dtype``.

    accum_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float64``. Passed to ``np.zeros`` as ``dtype``.

    progress : bool
        bool Show a variant-block tqdm progress bar when tqdm is installed. Default: ``False``.

    return_diagnostics : bool
        Whether to include the named optional output in the returned/stored result. Enabling
        this materializes or computes the additional table, effects, state or diagnostics
        described by the summary. Default: ``False``.

    use_syrk : bool
        Whether dense symmetric kernel cross-products use BLAS SYRK; False uses the
        matrix-product path. The symmetric mathematical kernel is unchanged. Default: ``True``.

    Returns
    -------
    object
        Computed result from ``(out, diag)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .io import PlinkReader
    from .grm_backend import grm_from_blocks
    from .grm_methods import validate_grm_method
    from .resources import bounded_variant_chunk
    if not use_syrk:
        raise ValueError("use_syrk=False was retired; the shared GRM engine uses in-place SYRK")
    _, components = validate_grm_method("vitezica", include)
    own_reader = not isinstance(prefix, PlinkReader)
    reader = PlinkReader(str(prefix), dtype=dtype) if own_reader else prefix
    try:
        all_ids = reader.fam["iid"].astype(str).to_numpy()
        if not pd.Index(all_ids).is_unique:
            raise ValueError("FAM IID entries must be unique")
        out_ids = all_ids if sample_ids is None else np.asarray(sample_ids, dtype=str)
        if not len(out_ids) or not pd.Index(out_ids).is_unique:
            raise ValueError("sample_ids must be nonempty and unique")
        sample_index = pd.Index(all_ids).get_indexer(out_ids)
        if np.any(sample_index < 0):
            raise KeyError(f"PLINK FAM is missing requested IID(s): {out_ids[sample_index < 0][:5].tolist()}")
        n = len(out_ids)
        fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
        if fit.shape != (n,) or fit.sum() < 2:
            raise ValueError("fit_mask must select at least two requested animals")
        arithmetic = np.dtype(accum_dtype)
        if arithmetic not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("accum_dtype must be float32 or float64")
        reserve = n*n*len(components)*(arithmetic.itemsize+np.dtype(dtype).itemsize)
        bs = bounded_variant_chunk(n, requested="auto" if block_size is None else block_size,
            maxmem=maxmem, memory_fraction=memory_fraction,
            bytes_per_call=80+12*max(0, arithmetic.itemsize-4),
            reserve_bytes=reserve, n_variants=reader.n_variants)
        iterator = reader.iter_blocks(bs, sample_index=sample_index)
        from .progress import StageProgress
        bar = StageProgress(total=(reader.n_variants+bs-1)//bs,
                            desc="PLINK1 A+D GRMs", enabled=progress, unit="block")
        def blocks():
            for _, block in iterator:
                yield block
                bar.update(1)
        try:
            kernels, weights, diagnostics = grm_from_blocks(blocks(), n,
                method="vitezica", components=components, fit_mask=fit,
                min_maf=min_maf, dtype=arithmetic, output_dtype=dtype)
        finally:
            bar.close()
        out = {c: pd.DataFrame(k, index=out_ids, columns=out_ids)
               for c, k in kernels.items()}
        diagnostics.update(n_samples=n, n_fit=int(fit.sum()),
            fit_mask_used=fit_mask is not None, block_size=int(bs),
            block_size_requested=block_size, maxmem=maxmem,
            memory_fraction=memory_fraction, syrk=True,
            normalization="vitezica_global_variant_denominator",
            **{f"weight_{c}": float(w) for c, w in weights.items()})
        if return_diagnostics and "additive" in kernels and "dominance" in kernels:
            correlations = _kernel_correlations(kernels["additive"], kernels["dominance"])
            diagnostics.update(correlations)
            corr = correlations["additive_dominance_diag_correlation"]
            diagnostics["dominance_partition_weakly_identified"] = bool(np.isfinite(corr) and abs(corr) >= .5)
        return (out, diagnostics) if return_diagnostics else out
    finally:
        if own_reader:
            reader.close()




class PlinkGenomicKernelBundle:
    """Lazy A/D genomic kernels backed by PLINK, with dense-RAM default.

    The bundle does not create a GRM file.  ``as_dict()`` returns kernel-like
    objects accepted by VIGOR's multivariate operator layer.  On first
    numerical use, all requested components are built together in one streamed
    PLINK pass and retained only in RAM, provided the dense matrices fit the
    configured memory budget.

    Matrix-free/sketched execution is deliberately *not* selected automatically:
    if the dense kernels do not fit, ``materialize()`` raises with an explicit
    message rather than silently turning every solver matvec into a genome scan.
    """
    def __init__(self,prefix,*,sample_ids=None,fit_mask=None,include=("additive",),
                 block_size="auto",maxmem=None,memory_fraction=None,min_maf=1e-4,
                 dtype=np.float32,accum_dtype=np.float32,progress=False,use_syrk=True):
        """Initialize PlinkGenomicKernelBundle with the inputs and resource settings below.

        Parameters
        ----------
        prefix : str or os.PathLike
            PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
            PLINK2 adapter). Companion metadata and genotype files must describe the same samples
            and variants.

        sample_ids : sequence of str or None
            Animal identifiers in the required row order. IDs must align with the genotype/phenotype
            source; None uses source identifiers or the routine's positional indexing convention.
            Default: ``None``.

        fit_mask : one-dimensional bool array or None
            Mask selecting training animals for fitting frequencies, scaling or model parameters.
            Length must equal the animal axis; None uses the routine's full eligible training set.
            Keep evaluation outcomes outside this mask. Default: ``None``.

        include : bool or sequence of str
            Genetic/environmental components requested for kernel construction. Boolean controls
            include or omit the named term; component sequences select supported
            additive/dominance/environment terms described by the kernel builder. Default:
            ``('additive',)``.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``'auto'``.

        maxmem : int, float, unit string or None
            Working-memory ceiling in bytes; a string such as 8GiB uses binary units
            (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses currently available RAM times
            memory_fraction. The resource helper caps explicit budgets against available RAM and
            shares the limit across workers. Default: ``None``.

        memory_fraction : float or None
            Fraction of physical RAM reserved for temporary blocks; working_memory_bytes enforces 0
            < memory_fraction <= 0.95. Parallel workers share this budget, so lowering it reduces
            peak per-worker allocation. Default: ``None`` (80% of physical RAM).

        min_maf : float
            Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
            below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

        dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``. Passed to ``PlinkReader`` as ``dtype``.

        accum_dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float32``.

        progress : bool
            Whether to emit progress/status reporting during this operation. False disables the
            corresponding reporting path. Default: ``False``.

        use_syrk : bool
            Whether dense symmetric kernel cross-products use BLAS SYRK; False uses the
            matrix-product path. The symmetric mathematical kernel is unchanged. Default: ``True``.
        """
        from .io import PlinkReader
        self.prefix=str(prefix); self.reader=PlinkReader(self.prefix,dtype=dtype)
        all_ids=self.reader.fam["iid"].astype(str).to_numpy()
        if sample_ids is None:
            self.sample_index=np.arange(self.reader.n_samples,dtype=np.int64); self.ids=all_ids
        else:
            self.ids=np.asarray(sample_ids,dtype=str)
            self.sample_index=pd.Index(all_ids).get_indexer(pd.Index(self.ids))
            if np.any(self.sample_index<0): raise KeyError(f"PLINK FAM missing IID(s): {self.ids[self.sample_index<0][:5].tolist()}")
        self.n=len(self.ids); self.fit=np.ones(self.n,bool) if fit_mask is None else np.asarray(fit_mask,bool)
        if self.fit.shape!=(self.n,) or self.fit.sum()<2: raise ValueError("fit_mask must select at least two requested animals")
        self.include=tuple(dict.fromkeys(str(x).lower() for x in include))
        bad=set(self.include)-{"additive","dominance"}
        if bad: raise ValueError(f"unknown GRM component(s): {sorted(bad)}")
        self.block_size=block_size; self.maxmem=maxmem; self.memory_fraction=None if memory_fraction is None else float(memory_fraction)
        self.min_maf=float(min_maf); self.dtype=np.dtype(dtype); self.accum_dtype=np.dtype(accum_dtype)
        self.progress=bool(progress); self.use_syrk=bool(use_syrk); self._dense=None; self._diagnostics=None
        self._operators={name:_PlinkDenseKernelOperator(self,name) for name in self.include}

    @property
    def estimated_dense_bytes(self):
        """Return the estimated dense bytes result computed by ``int``.

        Returns
        -------
        object
            Computed result from ``int(self.n * self.n * self.dtype.itemsize * len(self.include))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return int(self.n*self.n*self.dtype.itemsize*len(self.include))

    def _check_dense_budget(self):
        """Check dense budget.
        """
        from .resources import working_memory_bytes
        ceiling=working_memory_bytes(self.maxmem,self.memory_fraction)
        # Allow output matrices to consume at most 45% of the working budget;
        # the rest is reserved for accumulation + the largest genotype block.
        limit=max(256*1024**2,int(ceiling*.45))
        if self.estimated_dense_bytes>limit:
            raise MemoryError(
                f"dense {self.include} GRM cache needs about {self.estimated_dense_bytes/1024**3:.2f} GiB "
                f"for final float32 matrices (safe active limit {limit/1024**3:.2f} GiB). "
                "VIGOR will not silently switch to matrix-free genotype rescans. Increase maxmem/use a larger machine, "
                "use an explicit low-rank/sketched kernel, or explicitly request an experimental matrix-free path."
            )

    def materialize(self):
        """Load/compute all entries currently represented by this lazy collection.

        Returns
        -------
        object
            Computed result from ``self._dense``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if self._dense is None:
            self._check_dense_budget()
            # Canonical exact PLINK1 implementation lives in packed_bed.py.
            # Keep this bundle as the compatibility/operator facade only.
            from .packed_bed import dense_grm_from_bed
            res=dense_grm_from_bed(
                self.prefix,components=self.include,sample_ids=self.ids,fit_mask=self.fit,
                min_maf=self.min_maf,chunk_variants=self.block_size,memory_fraction=self.memory_fraction,
                maxmem=self.maxmem,accum_dtype=self.accum_dtype,output_dtype=self.dtype,
                blas_threads='auto',progress=self.progress)
            self._dense={k:np.asarray(v,dtype=self.dtype) for k,v in res['kernels'].items()}
            self._diagnostics={**res.get('diagnostics',{}),**{f'weight_{k}':float(v) for k,v in res.get('weights',{}).items()}}
        return self._dense

    @property
    def diagnostics(self):
        """Return approximation/alignment diagnostics for the current kernel representation.

        Returns
        -------
        dict
            Result fields: ``materialized``, ``estimated_dense_bytes``, ``normalization``, ``matrix_free_auto``. Field arrays preserve the corresponding input axis order.
        """
        if self._diagnostics is None: return {"materialized":False,"estimated_dense_bytes":self.estimated_dense_bytes,
                                               "normalization":"vitezica_global_variant_denominator","matrix_free_auto":False}
        return {**self._diagnostics,"materialized":True,"estimated_dense_bytes":self.estimated_dense_bytes,"matrix_free_auto":False}

    def __getitem__(self,name): 
        """Retrieve the requested entry from the indexed collection.

        Parameters
        ----------
        name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        Returns
        -------
        object
            Computed result from ``self._operators[str(name).lower()]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self._operators[str(name).lower()]
    def __iter__(self): 
        """Iterate through the collection in its stored order.
        """
        return iter(self.include)
    def __len__(self): 
        """Return the number of entries in this collection.
        """
        return len(self.include)
    def as_dict(self): 
        """Convert the configuration dataclass to a dictionary of its fields.

        Returns
        -------
        object
            Computed result from ``dict(self._operators)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return dict(self._operators)

    def _direct_cross(self,rows_i,rows_j,component):
        """Compute genotype cross-products directly for the selected animal rows.

        Parameters
        ----------
        rows_i : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        rows_j : sequence or integer array
            Identifiers or zero-based positions of selected/protected items on the corresponding
            input axis. Entries must exist; paired index arrays define the row/column pairs to
            compute.

        component : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
            Source-recognized literal values: ``additive``, ``dominance``. These select the
            corresponding branches below; comparison evidence alone does not establish an exhaustive
            enum.
        """
        ri=np.asarray(rows_i,dtype=np.int64).reshape(-1); rj=np.asarray(rows_j,dtype=np.int64).reshape(-1)
        if self._dense is not None: return self._dense[component][np.ix_(ri,rj)]
        requested=np.unique(np.r_[np.flatnonzero(self.fit),ri,rj])
        reader_rows=self.sample_index[requested]
        inv={int(r):k for k,r in enumerate(requested.tolist())}
        li=np.asarray([inv[int(r)] for r in ri],dtype=np.int64); lj=np.asarray([inv[int(r)] for r in rj],dtype=np.int64)
        lf=np.asarray([inv[int(r)] for r in np.flatnonzero(self.fit)],dtype=np.int64)
        out=np.zeros((len(ri),len(rj)),dtype=np.float64); denom=0.0
        # Small virtual selections favor bounded memory over the dense builder's large blocks.
        from .resources import resolve_variant_chunk
        # float64 per call: X, Xf, X[:,keep], A/D (+3 nested where temps for D), A[li]/A[lj] + masks
        bs=resolve_variant_chunk(self.block_size,len(reader_rows),bytes_per_call=52+(24 if component=="dominance" else 0),
                                 maxmem=self.maxmem,memory_fraction=self.memory_fraction,n_variants=self.reader.n_variants)
        for _meta,X0 in self.reader.iter_blocks(bs,sample_index=reader_rows):
            X=np.asarray(X0,float); Xf=X[lf]; cnt=np.isfinite(Xf).sum(0)
            p=np.divide(np.nansum(Xf,axis=0,dtype=np.float64),2.0*cnt,out=np.full(X.shape[1],np.nan),where=cnt>0)
            maf=np.minimum(p,1.0-p); keep=np.isfinite(p)&(p>0)&(p<1)&(maf>=self.min_maf)
            if not np.any(keep): continue
            X=X[:,keep]; p=np.clip(p[keep],self.min_maf,1.0-self.min_maf); q=1.0-p; valid=np.isfinite(X)
            if component=="additive":
                A=X-2.0*p[None,:]; A[~valid]=0.0
                out += A[li]@A[lj].T; denom += float(np.sum(2.0*p*q))
            elif component=="dominance":
                D=np.where(X==0,-2.0*p[None,:]**2,np.where(X==1,2.0*p[None,:]*q[None,:],np.where(X==2,-2.0*q[None,:]**2,0.0)))
                D[~valid]=0.0; out += D[li]@D[lj].T; denom += float(np.sum((2.0*p*q)**2))
            else: raise KeyError(component)
        return (out/max(denom,1e-30)).astype(self.dtype,copy=False)

    def xarray(self,component="additive",*,max_compute_pairs=4_000_000):
        """Virtual ``DataArray(iid_i,iid_j)``; slice first, then compute.

        Parameters
        ----------
        component : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required. Default:
            ``'additive'``.

        max_compute_pairs : int
            Maximum number of pair/group/path work items retained or evaluated by this guard.
            Positive caps bound runtime/cache memory; exceeding a hard guard may raise rather than
            silently approximate. Default: ``4000000``.

        Returns
        -------
        object
            Computed result from ``arr``; see the summary and the called implementation for its labeled/numerical representation.
        """
        xr=__import__('xarray'); from xarray.backends import BackendArray; from xarray.core import indexing
        comp=str(component).lower()
        if comp not in self.include: raise KeyError(comp)
        parent=self; n=self.n; guard=max(1,int(max_compute_pairs))
        class _Backend(BackendArray):
            """_Backend represents backend.
            """
            shape=(n,n); dtype=parent.dtype
            def __getitem__(self,key):
                """Retrieve the requested entry from the indexed collection.

                Parameters
                ----------
                key : str
                    Identifier used to select or label the relevant component, field or output quantity. It
                    must match the names present in the supplied object where lookup is required.

                Returns
                -------
                object
                    Computed result from ``indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._raw)``; see the summary and the called implementation for its labeled/numerical representation.
                """
                return indexing.explicit_indexing_adapter(key,self.shape,indexing.IndexingSupport.OUTER,self._raw)
            @staticmethod
            def _idx(k):
                """Normalize requested animal positions for the enclosing kernel source.

                Parameters
                ----------
                k : int or index array
                    Local position, index selection or step used by the stated slice/iteration. Indices must
                    refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
                    the enclosing dimensions.
                """
                if isinstance(k,slice): return np.arange(n,dtype=np.int64)[k],False
                a=np.asarray(k)
                if a.ndim==0:return np.asarray([int(a)],np.int64),True
                return a.astype(np.int64,copy=False).reshape(-1),False
            def _raw(self,key):
                """Read the selected raw kernel block before normalization.

                Parameters
                ----------
                key : str
                    Identifier used to select or label the relevant component, field or output quantity. It
                    must match the names present in the supplied object where lookup is required.
                """
                ii,si=self._idx(key[0]); jj,sj=self._idx(key[1]); pairs=len(ii)*len(jj)
                if pairs>guard: raise MemoryError(f"refusing to compute {len(ii):,} x {len(jj):,}={pairs:,} GRM cells; slice .sel/.isel first")
                z=parent._direct_cross(ii,jj,comp)
                if si and sj:return np.asarray(z[0,0],parent.dtype)
                if si:return z[0]
                if sj:return z[:,0]
                return z
        arr=xr.DataArray(indexing.LazilyIndexedArray(_Backend()),dims=("iid_i","iid_j"),
                         coords={"iid_i":self.ids,"iid_j":self.ids},name=f"grm_{comp}")
        arr.attrs.update({"virtual":True,"materialization":"selection_then_compute_only","component":comp,
                          "normalization":"vitezica_global_variant_denominator","source_plink":self.prefix,
                          "full_dense_cached":self._dense is not None,"max_compute_pairs":guard})
        return arr

    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        if getattr(self,'reader',None) is not None:self.reader.close()
    def __del__(self):
        """Release owned resources when the object is finalized.
        """
        try:self.close()
        except Exception:pass


class _PlinkDenseKernelOperator:
    """KernelOperator-compatible view of one shared PLINK A/D dense-RAM cache."""
    def __init__(self,bundle,component):
        """Initialize _PlinkDenseKernelOperator with the inputs and resource settings below.

        Parameters
        ----------
        bundle : mapping or feature/annotation bundle
            Prepared feature/annotation collection indexed by its named groups. Reuse the same
            fitted preprocessing, animal/variant order and metadata when transforming or predicting.

        component : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.
        """
        self.bundle=bundle; self.component=str(component); self.shape=(bundle.n,bundle.n); self.dtype=bundle.dtype
    def _K(self): 
        """Retrieve the represented kernel matrix from the enclosing bundle.
                
        """
        return self.bundle.materialize()[self.component]
    def matmat(self,X): 
        """Apply the enclosing covariance/kernel operator to multiple right-hand sides.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``self._K() @ np.asarray(X)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self._K()@np.asarray(X)
    def diag(self): 
        """Return the diagonal variances of the represented covariance/kernel.

        Returns
        -------
        object
            Computed result from ``np.diag(self._K())``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return np.diag(self._K())
    def row_block(self,start,stop): 
        """Retrieve the requested contiguous animal rows of the represented kernel.

        Parameters
        ----------
        start : int or None
            Boundary index or genomic coordinate for the requested interval. Use the coordinate
            convention of this reader/query; None leaves that boundary unrestricted where supported.

        stop : int or index array
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        Returns
        -------
        object
            Computed result from ``self._K()[int(start):int(stop), :]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self._K()[int(start):int(stop),:]
    def submatrix(self,rows):
        """Retrieve the requested animal-index square/cross kernel block.

        Parameters
        ----------
        rows : sequence, index or scalar
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Returns
        -------
        object
            Computed result from ``self._K()[np.ix_(r, r)]``; see the summary and the called implementation for its labeled/numerical representation.
        """
        r=np.asarray(rows,dtype=np.int64); return self._K()[np.ix_(r,r)]
    def to_xarray(self,**kwargs): 
        """Return the to xarray result computed by ``self.bundle.xarray``.

        Parameters
        ----------
        **kwargs : mapping
            Additional keyword arguments forwarded to ``self.bundle.xarray``. See those callables
            for accepted keys and values. Passed to ``self.bundle.xarray`` as ``**expanded
            mapping``.

        Returns
        -------
        object
            Computed result from ``self.bundle.xarray(self.component, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.bundle.xarray(self.component,**kwargs)
    @property
    def diagnostics(self): 
        """Return approximation/alignment diagnostics for the current kernel representation.

        Returns
        -------
        object
            Computed result from ``self.bundle.diagnostics``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.bundle.diagnostics


def genomic_kernel_operators_from_plink1(prefix,*,include=("additive",),**kwargs):
    """Return operator mapping backed by one shared in-RAM PLINK GRM build.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    include : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default:
        ``('additive',)``. Passed to ``PlinkGenomicKernelBundle`` as ``include``.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``PlinkGenomicKernelBundle``. See those
        callables for accepted keys and values. Passed to ``PlinkGenomicKernelBundle`` as
        ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``ops``; see the summary and the called implementation for its labeled/numerical representation.
    """
    bundle=PlinkGenomicKernelBundle(prefix,include=include,**kwargs)
    ops=bundle.as_dict()
    # Keep the shared bundle discoverable without changing the dict-like API.
    for op in ops.values(): op.bundle=bundle
    return ops



def genomic_features_from_plink1(
    prefix,
    *,
    sample_ids=None,
    fit_mask=None,
    max_variants=2048,
    strategy="equidistant",
    include_dominance=True,
    min_maf=1e-4,
    random_state=0,
    dtype=np.float32,
    screen_targets=None,
    screen_components=8,
    screen_prefilter_multiplier=4,
    screen_bin_bp=250_000,
    screen_max_per_bin=2,
    maxmem=None,
    memory_fraction=None,
):
    """Build a bounded raw-SNP feature panel for residual prediction experts.

    ``equidistant`` and ``random`` inclusion are phenotype-independent.  For
    very large panels, ``strategy='residual_screen'`` performs a bounded-memory
    full-genome scan against low-rank residual targets supplied in
    ``screen_targets`` and then keeps a genome-spaced high-score panel.  The
    screen uses only ``fit_mask`` animals and is intended for an outer training
    split; never include held-out evaluation animals in ``fit_mask``.  Features
    are standardized additive dosage and, optionally, orthogonal Vitezica
    dominance coding.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    sample_ids : sequence of str or None
        Animal identifiers in the required row order. IDs must align with the genotype/phenotype
        source; None uses source identifiers or the routine's positional indexing convention.
        Default: ``None``.

    fit_mask : one-dimensional bool array or None
        Mask selecting training animals for fitting frequencies, scaling or model parameters.
        Length must equal the animal axis; None uses the routine's full eligible training set.
        Keep evaluation outcomes outside this mask. Default: ``None``.

    max_variants : int or None
        Maximum candidates/features retained by the corresponding ranking stage. Positive
        integers bound work/model size; None leaves retention to that stage's policy. Default:
        ``2048``.

    strategy : str
        Candidate/landmark selection policy: 'uniform'/'random' samples broadly; 'even'/'grid'
        distributes candidates along the input order;
        'residual'/'residual_screen'/'supervised_screen' ranks using training residual evidence
        where implemented. See the local selector for its accepted subset. Default:
        ``'equidistant'``.

    include_dominance : bool or sequence of str
        Genetic/environmental components requested for kernel construction. Boolean controls
        include or omit the named term; component sequences select supported
        additive/dominance/environment terms described by the kernel builder. Default: ``True``.

    min_maf : float
        Minor-allele-frequency exclusion threshold on the [0, 0.5] biological scale. Variants
        below the threshold are omitted from the corresponding kernel/scan. Default: ``0.0001``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``randomized_svd`` as ``random_state``.

    dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``np.float32``. Passed to ``PlinkReader`` as ``dtype``.

    screen_targets : object (see the cited input contract)
        Input for the operation ``screen_targets.reindex(pd.Index(out_ids)).to_numpy(float)``.
        See ``vigor.grm.genomic_features_from_plink1`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    screen_components : int
        Input for the operation ``max(1, min(int(screen_components), Tf.shape[0] - 1,
        Tf.shape[1]))``. See ``vigor.grm.genomic_features_from_plink1`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Default: ``8``.

    screen_prefilter_multiplier : int
        Input for the operation ``int(k) * max(1, int(screen_prefilter_multiplier))``. See
        ``vigor.grm.genomic_features_from_plink1`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``4``.

    screen_bin_bp : int
        Input for the operation ``int(row['pos']) // int(screen_bin_bp)``. See
        ``vigor.grm.genomic_features_from_plink1`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``250000``. Source comparisons: ``int(screen_bin_bp) > 0``. See the rejection/fallback
        branches for enforcement.

    screen_max_per_bin : int
        Input for the operation ``c >= max(1, int(screen_max_per_bin))``. See
        ``vigor.grm.genomic_features_from_plink1`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default: ``2``.

    maxmem : int, float, unit string or None
        Working-memory budget for the genotype streaming blocks; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(F, index=out_ids, columns=names)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from .io import PlinkReader
    own_reader = not isinstance(prefix, PlinkReader)
    reader = PlinkReader(str(prefix), dtype=dtype) if own_reader else prefix
    try:
        all_ids = reader.fam["iid"].astype(str).to_numpy()
        if sample_ids is None:
            sidx = np.arange(reader.n_samples, dtype=np.int64); out_ids = all_ids
        else:
            out_ids = np.asarray(sample_ids, dtype=str)
            sidx = pd.Index(all_ids).get_indexer(pd.Index(out_ids))
            if np.any(sidx < 0):
                raise KeyError(f"PLINK FAM is missing requested IID(s): {out_ids[sidx < 0][:5].tolist()}")
        n = len(sidx)
        fit = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
        if fit.shape != (n,) or fit.sum() < 2:
            raise ValueError("fit_mask must select at least two requested animals")
        m = reader.n_variants
        if max_variants is None or int(max_variants) >= m:
            vidx = np.arange(m, dtype=np.int64)
        else:
            k = max(1, int(max_variants))
            mode = str(strategy).lower()
            if mode in {"equidistant", "even", "grid"}:
                vidx = np.unique(np.linspace(0, m-1, k, dtype=np.int64))
            elif mode in {"random", "uniform"}:
                vidx = np.sort(np.random.default_rng(random_state).choice(m, size=k, replace=False))
            elif mode in {"residual_screen", "supervised_screen", "residual"}:
                if screen_targets is None:
                    raise ValueError("strategy='residual_screen' requires screen_targets (BLUP residual traits)")
                if isinstance(screen_targets, (pd.DataFrame, pd.Series)):
                    target = screen_targets.reindex(pd.Index(out_ids)).to_numpy(float)
                else:
                    target = np.asarray(screen_targets, float)
                if target.ndim == 1: target = target[:, None]
                if target.shape[0] != n:
                    raise ValueError("screen_targets must have one row per requested sample")
                Tf = target[fit].copy()
                if Tf.shape[1] == 0:
                    raise ValueError("screen_targets has no columns")
                cm = np.nanmean(Tf, axis=0)
                cm = np.where(np.isfinite(cm), cm, 0.0)
                Tf = np.where(np.isfinite(Tf), Tf, cm[None, :])
                Tf -= Tf.mean(axis=0, keepdims=True)
                qcomp = max(1, min(int(screen_components), Tf.shape[0]-1, Tf.shape[1]))
                if qcomp < min(Tf.shape):
                    U, sv, _ = randomized_svd(Tf, n_components=qcomp, random_state=int(random_state))
                    Q = U * sv[None, :]
                else:
                    Q = Tf
                qnorm = np.sqrt(np.sum(Q*Q, axis=0)); qnorm=np.where(qnorm>1e-12,qnorm,1.0)
                Q = Q / qnorm[None, :]
                pre_k=max(k, min(m, int(k)*max(1,int(screen_prefilter_multiplier))))
                best_i=np.empty(0,dtype=np.int64); best_s=np.empty(0,dtype=float)
                from .resources import resolve_variant_chunk
                # float64 per call: X, Xf, X[:,keep], A, Af, Af*Af (+masks); dominance adds D, 3 where temps, Df, Df*Df
                sbs=resolve_variant_chunk('auto',n,bytes_per_call=58+(64 if include_dominance else 0),bytes_per_variant=16*Q.shape[1],
                                          maxmem=maxmem,memory_fraction=memory_fraction,n_variants=m)
                for meta0, X0 in reader.iter_blocks(block_size=sbs, sample_index=sidx, maxmem=maxmem, memory_fraction=memory_fraction):
                    X=np.asarray(X0,float); Xf=X[fit]
                    cnt=np.isfinite(Xf).sum(0)
                    pp=np.divide(np.nansum(Xf,axis=0,dtype=np.float64),2.0*cnt,out=np.full(X.shape[1],np.nan),where=cnt>0)
                    maf=np.minimum(pp,1.0-pp); keep0=np.isfinite(pp)&(pp>0)&(pp<1)&(maf>=float(min_maf))
                    if not np.any(keep0): continue
                    X=X[:,keep0]; pp=np.clip(pp[keep0],float(min_maf),1.0-float(min_maf)); qq=1.0-pp
                    A=X-2.0*pp[None,:]; A[~np.isfinite(A)]=0.0
                    A/=np.sqrt(np.maximum(2.0*pp*qq,1e-12))[None,:]
                    Af=A[fit]; an=np.sqrt(np.sum(Af*Af,axis=0)); an=np.where(an>1e-12,an,1.0)
                    score=np.max(np.abs((Af.T@Q)/an[:,None]),axis=1)
                    if include_dominance:
                        D=np.where(X==0,-2*pp[None,:]**2,np.where(X==1,2*pp[None,:]*qq[None,:],np.where(X==2,-2*qq[None,:]**2,0.0)))
                        D[~np.isfinite(X)]=0.0; D/=np.maximum(2.0*pp*qq,1e-12)[None,:]
                        Df=D[fit]; dn=np.sqrt(np.sum(Df*Df,axis=0)); dn=np.where(dn>1e-12,dn,1.0)
                        score=np.maximum(score,np.max(np.abs((Df.T@Q)/dn[:,None]),axis=1))
                    idx0=meta0.index.to_numpy(dtype=np.int64)[keep0]
                    best_i=np.r_[best_i,idx0]; best_s=np.r_[best_s,score]
                    if len(best_i)>2*pre_k:
                        jj=np.argpartition(best_s,-pre_k)[-pre_k:]
                        best_i,best_s=best_i[jj],best_s[jj]
                if not len(best_i):
                    raise ValueError("residual screen found no variants passing MAF/QC")
                if len(best_i)>pre_k:
                    jj=np.argpartition(best_s,-pre_k)[-pre_k:]; best_i,best_s=best_i[jj],best_s[jj]
                order=np.argsort(best_s)[::-1]; best_i,best_s=best_i[order],best_s[order]
                # Genome-spaced greedy retention prevents a single long-LD peak
                # from consuming the entire expert feature budget.  This is a
                # scalable proxy for full pairwise LD clumping at 5M variants.
                if int(screen_bin_bp)>0 and 'pos' in reader.bim.columns:
                    chosen=[]; counts_bin={}
                    for vi in best_i:
                        row=reader.bim.iloc[int(vi)]
                        key=(str(row['chrom']), int(row['pos'])//int(screen_bin_bp))
                        c=counts_bin.get(key,0)
                        if c>=max(1,int(screen_max_per_bin)): continue
                        chosen.append(int(vi)); counts_bin[key]=c+1
                        if len(chosen)>=k: break
                    if len(chosen)<k:
                        used=set(chosen)
                        chosen.extend([int(x) for x in best_i if int(x) not in used][:k-len(chosen)])
                    vidx=np.sort(np.asarray(chosen[:k],dtype=np.int64))
                else:
                    vidx=np.sort(best_i[:k].astype(np.int64,copy=False))
            else:
                raise ValueError("strategy must be 'equidistant', 'random', or 'residual_screen'")
        parts=[]; names=[]
        from .resources import resolve_variant_chunk
        # float64 per call: X, two X[fit] copies, X[:,keep], A (+mask); dominance adds D and 3 where temps
        fbs=resolve_variant_chunk('auto',n,bytes_per_call=41+(40 if include_dominance else 0),maxmem=maxmem,
                                  memory_fraction=memory_fraction,n_variants=len(vidx))
        for meta, X0 in reader.iter_blocks(block_size=fbs, variant_index=vidx, sample_index=sidx,
                                         maxmem=maxmem, memory_fraction=memory_fraction):
            X = np.asarray(X0, dtype=float)
            cnt = np.isfinite(X[fit]).sum(0)
            p = np.divide(np.nansum(X[fit], axis=0, dtype=np.float64), 2.0*cnt, out=np.full(X.shape[1], np.nan), where=cnt>0)
            maf = np.minimum(p, 1.0-p)
            keep = np.isfinite(p) & (p > 0) & (p < 1) & (maf >= float(min_maf))
            if not np.any(keep):
                continue
            X=X[:,keep]; p=np.clip(p[keep], float(min_maf), 1.0-float(min_maf)); q=1.0-p
            snp = meta.iloc[np.flatnonzero(keep)]["snp"].astype(str).to_numpy()
            A = X - 2.0*p[None,:]
            A[~np.isfinite(A)] = 0.0
            A /= np.sqrt(np.maximum(2.0*p*q, 1e-12))[None,:]
            parts.append(A.astype(dtype, copy=False)); names.extend([f"snp_{x}" for x in snp])
            if include_dominance:
                D=np.where(X==0,-2*p[None,:]**2,np.where(X==1,2*p[None,:]*q[None,:],np.where(X==2,-2*q[None,:]**2,0.0)))
                D[~np.isfinite(X)] = 0.0
                D /= np.maximum(2.0*p*q, 1e-12)[None,:]
                parts.append(D.astype(dtype, copy=False)); names.extend([f"dom_{x}" for x in snp])
        F=np.concatenate(parts,axis=1) if parts else np.zeros((n,0),dtype=dtype)
        return pd.DataFrame(F,index=out_ids,columns=names)
    finally:
        if own_reader:
            reader.close()
