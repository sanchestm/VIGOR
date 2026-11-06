"""Public association dispatch: exact observed-space score scans, or the fast discovery scan."""
from __future__ import annotations
import os


def _is_plink_prefix(source):
    """True for a PLINK1 fileset prefix (``prefix.bed`` exists), not a Zarr store."""
    if not isinstance(source, (str, os.PathLike)):
        return False
    text = os.fspath(source).rstrip("/\\")
    return not text.endswith(".zarr") and os.path.exists(text + ".bed")


def mvGWAS(*args, engine="fast", out=None, **kwargs):
    """Run a quantitative multivariate association scan.

    Parameters
    ----------
    *args : tuple
        Pipeline: (PLINK prefix, phenotype table). Multivariate: (trait table,
        genotype source). Projected discovery scan: (genotype source,) with
        ``state=``. Keyword forms are bfile/pheno, traitdf/genotypes and
        genotypes/state.
    engine : {'fast','exact','auto','pipeline','multivariate'}
        'fast' (default) runs the same pipeline/multivariate scan as 'exact'
        with the same inputs and output schema, but keeps only the score X'Py
        exact and replaces the per-variant information D'PD by a ratio
        approximation calibrated per chromosome against exact D'PD on a few
        spread variants (:mod:`vigor.gwas.fast`; diagnostics in
        ``attrs['scan_group_metadata']``). Cost is one GEMM per block instead
        of a covariance solve per variant column. With ``state=`` it instead
        runs the projected discovery scan of :mod:`vigor.mvgwas_fast`.
        'exact' evaluates D'PD for every variant (spectral rotation for
        complete phenotypes with one kernel, block PCG otherwise).
        'auto'/'pipeline'/'multivariate' choose the route explicitly and honour
        ``score_engine=`` (default 'fast'). Routing picks pipeline for bfile or
        a path-like first positional argument, else multivariate.
    out : path-like, optional
        Exact engines: stream each block to Zarr v3 and return its lazy Dataset.
        Fast engine: Zarr v3 path for all scanned variants (``zarr_path``).
    **kwargs : mapping
        Options of pipeline.run_gwas, multivariate.mvGWAS or, for 'fast',
        mvgwas_fast.scan_packed_bed (PLINK prefix) / scan_xarray (Zarr store or
        xarray). The fast engine requires ``state``: a FastScoreState, or a
        chromosome -> state mapping/callable for LOCO, built with
        ``vigor.mvgwas_fast.state_from_vigor_null(prepare_score_null(...))``.

    Returns
    -------
    xarray.Dataset or mvgwas_fast.ScanResult
        Exact engines: variant by trait score tests plus variant omnibus fields.
        Fast engine: ScanResult (table, top variants, candidate regions and
        diagnostics); its ``table`` Dataset carries the engine attributes.

    Complexity
    ----------
    Dispatch O(1). See gwas.exact and mvgwas_fast for the scan costs.
    """
    if engine not in {"fast", "exact", "auto", "pipeline", "multivariate"}:
        raise ValueError("engine must be fast, exact, auto, pipeline or multivariate")
    selected = engine
    if selected == "fast" and kwargs.get("state") is not None:
        from . import mvgwas_fast as mf
        state = kwargs.pop("state")
        genotypes = args[0] if args else kwargs.pop("genotypes", None)
        if genotypes is None:
            raise ValueError("engine='fast' requires a genotype source")
        scan = mf.scan_packed_bed if _is_plink_prefix(genotypes) else mf.scan_xarray
        if out is not None:
            kwargs.setdefault("zarr_path", out)
            kwargs.setdefault("return_all", True)
        result = scan(genotypes, state, **kwargs)
        table = getattr(result, "table", None)
        if table is not None and hasattr(table, "attrs"):
            table.attrs["vigor_gwas_engine"] = "fast"
            table.attrs["vigor_score_core"] = "projected_dual_null_trait_calibrated"
        return result
    kwargs.pop("state", None)
    if selected in {"fast", "exact"}:
        if "score_engine" in kwargs and kwargs["score_engine"] != selected:
            raise ValueError(f"engine={selected!r} conflicts with score_engine={kwargs['score_engine']!r}")
        kwargs["score_engine"] = selected
        selected = "auto"
    kwargs.setdefault("score_engine", "fast")
    if selected == "auto":
        selected = "pipeline" if "bfile" in kwargs or (args and isinstance(args[0], (str, os.PathLike))) else "multivariate"
    score_engine = kwargs["score_engine"]
    if selected == "pipeline":
        from .pipeline import run_gwas
        ds = run_gwas(*args, out=out, **kwargs)
    else:
        from .multivariate import mvGWAS as scan
        ds = scan(*args, out=out, **kwargs)
    ds.attrs["vigor_gwas_engine"] = selected
    ds.attrs["vigor_score_core"] = ("observed_space_fast_calibrated" if score_engine == "fast"
                                    else "observed_space_exact")
    return ds


__all__ = ["mvGWAS"]
