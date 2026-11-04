> VIGOR 2.12.1. Install as described in the [README](../../README.md). Outputs are cleared; the notebooks were checked against the 2.12.1 API but not re-executed for this release. Current association-scan fields and optional projection artifacts are described in [docs/INPUT_OUTPUT.md](../../docs/INPUT_OUTPUT.md).

# VIGOR 2.12.1 tutorials

The 47 retained tutorial notebooks comprise 17 root topics (00–16), 15 synthetic
counterparts and 15 real-HS routes. Synthetic routes demonstrate known-truth or
simulated inputs; real-HS file-fixture execution exercises I/O and workflow, not
the full research cohort. Optional dependencies and unprovided inputs remain
explicit in each route.

Results are labeled `xarray.Dataset`s (see [docs/RESULTS_XARRAY.md](../../docs/RESULTS_XARRAY.md)).
The notebooks read arrays with `res["G"].to_pandas()`, scalars and diagnostics with
`res.attrs[...]`, and long tables with `vigor.result_table(res, ...)` or `vigor.as_frame(...)`.
Pass `save_path=` to keep a result as consolidated, sharded Zarr v3.

The current full-cohort real26 entrypoints are in [real26](../real26/MIGRATION.md).
Their 7,616 × 26 fixture, train-only masking/normalization, genuine comparators,
saved covariance/native factors and optional bounded discovery are distinct from
synthetic fixture checks. The migration table accounts for every retained topic
and names deferred real-data analyses.

Run a notebook top to bottom in a Python kernel where VIGOR is installed. The root
and `synthetic/` routes simulate their own data. The `real_hs/` routes expect your
own project inputs: copy `examples/project_config.py` to `<project>/vigor_run/project_config.py`
and edit its paths; the notebooks find it from their working directory or its parents
(or set `VIGOR_PROJECT_ROOT`). Historical
filenames containing 2_11_4 describe notebook lineage; the code targets VIGOR 2.12.1.
