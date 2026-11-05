"""Public association dispatch: exact observed-space score scans, or the fast discovery scan."""
from __future__ import annotations
import os


def _is_plink_prefix(source):
    """True for a PLINK1 fileset prefix (``prefix.bed`` exists), not a Zarr store."""
    if not isinstance(source, (str, os.PathLike)):
        return False
    text = os.fspath(source).rstrip("/\\")
    return not text.endswith(".zarr") and os.path.exists(text + ".bed")


def mvGWAS(*args, engine="auto", out=None, **kwargs):
    """Run a quantitative multivariate association scan.

    Parameters
    ----------
    *args : tuple
        Pipeline: (PLINK prefix, phenotype table). Multivariate: (trait table,
        genotype source). Fast: (genotype source,) with ``state=``. Keyword
        forms are bfile/pheno, traitdf/genotypes and genotypes/state.
    engine : {'auto','pipeline','multivariate','fast'}
        'pipeline' and 'multivariate' use the exact observed-space score
        engine (one spectral rotation per null for complete phenotypes with a
        single kernel, block PCG otherwise). 'fast' is the genome-wide
        discovery scan of :mod:`vigor.mvgwas_fast`: one GEMM per genotype block
        against the projected dual, calibrated with simulated null traits, with
        optional exact rescoring of hits (``exact_rescore_fn``). Auto selects
        pipeline for bfile or a path-like first positional argument, else
        multivariate.
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
    if engine not in {"auto", "pipeline", "multivariate", "fast"}:
        raise ValueError("engine must be auto, pipeline, multivariate or fast")
    selected = engine
    if selected == "fast":
        from . import mvgwas_fast as mf
        state = kwargs.pop("state", None)
        if state is None:
            raise ValueError("engine='fast' requires state=; build it per chromosome with "
                             "vigor.mvgwas_fast.state_from_vigor_null(vigor.prepare_score_null(...))")
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
    if selected == "auto":
        selected = "pipeline" if "bfile" in kwargs or (args and isinstance(args[0], (str, os.PathLike))) else "multivariate"
    if selected == "pipeline":
        from .pipeline import run_gwas
        ds = run_gwas(*args, out=out, **kwargs)
    else:
        from .multivariate import mvGWAS as scan
        ds = scan(*args, out=out, **kwargs)
    ds.attrs["vigor_gwas_engine"] = selected
    ds.attrs["vigor_score_core"] = "observed_space_exact"
    return ds


__all__ = ["mvGWAS"]
