"""Command-line interface for quantitative-trait VIGOR scans."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import __version__, mvGWAS


def _names(value: str | None) -> list[str]:
    """Parse a comma-separated identifier list, trimming empty entries.

    Parameters
    ----------
    value : str | None
        Input for the operation ``value.split(',')``. See ``vigor.cli._names`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.
    """
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def _integers(value: str | None) -> list[int]:
    """Parse a comma-separated list of integer option values.

    Parameters
    ----------
    value : str | None
        Input for the operation ``_names(value)``. See ``vigor.cli._integers`` and its caller
        for the scientific representation; the shown operation constrains the required
        type/shape.
    """
    return [int(item) for item in _names(value)]


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser with genotype, model and resource options.

    Returns
    -------
    object
        Computed result from ``parser``; see the summary and the called implementation for its labeled/numerical representation.
    """
    parser = argparse.ArgumentParser(
        prog="vigor",
        description="Run a quantitative-trait, dominance-aware GWAS with VIGOR",
    )
    parser.add_argument("--version", action="version", version=f"VIGOR {__version__}")
    parser.add_argument("--bfile", required=True, help="PLINK 1 prefix without .bed/.bim/.fam")
    parser.add_argument("--grm-bfile", help="Optional separate PLINK 1 relationship-panel prefix")
    parser.add_argument("--pheno", required=True, type=Path, help="Tab-separated phenotype/covariate table")
    parser.add_argument("--id-column", default="iid", help="Sample ID column in --pheno")
    parser.add_argument("--traits", required=True, help="Comma-separated quantitative phenotype columns")
    parser.add_argument("--covariates", help="Comma-separated numeric covariate columns")
    parser.add_argument("--categorical-covariates", help="Comma-separated categorical covariate columns")
    parser.add_argument("--chromosomes", help="Comma-separated chromosome labels")
    parser.add_argument("--model", choices=["add-dom", "additive"], default="add-dom")
    parser.add_argument("--trait-mode", choices=["independent", "multivariate"], default="multivariate")
    parser.add_argument(
        "--variance-estimator",
        choices=["auto", "reml", "ml", "profile_reml", "profile_ml", "score"],
        default="auto",
        help="auto selects profile REML for independent mode and SCORE for multivariate mode",
    )
    parser.add_argument("--decomposition", choices=["exact"], default="exact")
    parser.add_argument("--compute-dtype", choices=["float64", "float32"], default="float32")
    parser.add_argument("--block-size", default="128", help="Maximum marker block size, or auto to size from the memory budget")
    parser.add_argument("--memory-budget-mb", type=int)
    parser.add_argument("--n-jobs", type=int, default=1, help="BLAS CPU threads; -1 uses all logical CPUs")
    parser.add_argument("--grm-cache", help="Existing canonical chromosome GRM Zarr store")
    parser.add_argument("--backend", choices=["cpu"], default="cpu")
    parser.add_argument("--complete-case-across-traits", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream sharded Zarr without retaining result blocks")
    parser.add_argument("--out", required=True, type=Path, help="Output prefix")
    return parser


def main(argv=None) -> None:
    """Parse command-line options and execute the requested VIGOR workflow.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line argument tokens. None reads the process argument vector; accepted switches
        and choices are defined by the corresponding parser. Default: ``None``.
    """
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.pheno, sep=None, engine="python", dtype={args.id_column: "string"})
    frame[args.id_column] = frame[args.id_column].str.strip()
    frame = frame.set_index(args.id_column)
    traits = _names(args.traits)
    numeric = _names(args.covariates)
    categorical = _names(args.categorical_covariates)
    requested = traits + numeric + categorical
    missing = sorted(set(requested) - set(frame.columns))
    if missing:
        raise ValueError(f"columns missing from phenotype table: {missing}")

    covar = None
    if numeric or categorical:
        if numeric:
            frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        covar = frame[numeric + categorical]

    variance_estimator = args.variance_estimator
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = mvGWAS(
        args.bfile,
        frame[traits],
        traits=traits,
        covar=covar,
        model=args.model,
        chroms=_names(args.chromosomes) or None,
        block_size="auto" if args.block_size == "auto" else int(args.block_size),
        per_trait_missing=not args.complete_case_across_traits,
        grm_bfile=args.grm_bfile,
        out=str(args.out),
        return_results=not args.stream,
        trait_mode=args.trait_mode,
        variance_estimator=variance_estimator,
        decomposition=args.decomposition,
        compute_dtype=args.compute_dtype,
        memory_budget_mb=args.memory_budget_mb,
        n_jobs=args.n_jobs,
        grm_cache=args.grm_cache,
        backend=args.backend,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "software": "VIGOR",
        "domino_version": __version__,
        "python_namespace": "vigor",
        "bfile": str(Path(args.bfile).resolve()),
        "grm_bfile": str(Path(args.grm_bfile).resolve()) if args.grm_bfile else None,
        "phenotype": str(args.pheno.resolve()),
        "traits": traits,
        "numeric_covariates": numeric,
        "categorical_covariates": categorical,
        "model": args.model,
        "trait_mode": args.trait_mode,
        "variance_estimator": variance_estimator,
        "decomposition": args.decomposition,
        "compute_dtype": args.compute_dtype,
        "block_size_ceiling": args.block_size,
        "memory_budget_mb": args.memory_budget_mb,
        "n_jobs": args.n_jobs,
        "grm_cache": args.grm_cache,
        "backend": args.backend,
        "per_trait_missingness": not args.complete_case_across_traits,
        "streamed_output": True,
        "trait_type": "quantitative_only",
        "runtime_s": elapsed,
        "n_variants": result.sizes["variant"],
        "n_test_coordinates": result.sizes["trait"],
        "execution": result.attrs.get("execution"),
    }
    Path(f"{args.out}.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
