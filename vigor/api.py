"""Public association dispatch; both routes use gwas.exact score statistics."""
from __future__ import annotations
import os


def mvGWAS(*args, engine="auto", out=None, **kwargs):
    """Run a quantitative exact-score scan and optionally stream sharded Zarr.

    Parameters
    ----------
    *args : tuple
        Pipeline: (PLINK prefix, phenotype table). Multivariate: (trait table,
        genotype source). Keyword forms are bfile/pheno and traitdf/genotypes.
    engine : {'auto','pipeline','multivariate'}
        Auto selects pipeline for bfile or a path-like first positional argument.
    out : path-like, optional
        Stream each computed block directly to Zarr v3; return its lazy Dataset.
    **kwargs : mapping
        Options documented by pipeline.run_gwas or multivariate.mvGWAS.
        Retired approximation controls raise rather than being silently ignored.

    Returns
    -------
    xarray.Dataset
        Variant by trait coordinate score tests, plus variant omnibus fields.

    Complexity
    ----------
    Dispatch O(1). See gwas.exact and the selected source adapter for dense
    kernel, per-variant information and streaming costs.
    """
    if engine not in {"auto", "pipeline", "multivariate"}:
        raise ValueError("engine must be auto, pipeline or multivariate")
    selected = engine
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
