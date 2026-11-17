> VIGOR 2.12.1. Install as described in the [README](../../README.md). Outputs are cleared; the notebooks were checked against the 2.12.1 API but not re-executed for this release. Association-scan fields are documented in `help(vigor.mvGWAS)` and `help(vigor.factorGWAS)`.

# VIGOR 2.12.1 tutorials

The 47 retained tutorial notebooks comprise 17 root topics (00–16), 15 synthetic
counterparts and 15 real-HS routes. Synthetic routes demonstrate known-truth or
simulated inputs; real-HS file-fixture execution exercises I/O and workflow, not
the full research cohort. Optional dependencies and unprovided inputs remain
explicit in each route.

Results are labeled `xarray.Dataset`s.
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

## Packed synthetic genotypes

The root and `synthetic/` routes simulate genotypes with `vigor.simulate_packed_cohort`.
It writes them once to a 2-bit packed Zarr store (`GT_packed2`, 0.25 byte per call) under
`./vigor_tutorial_data/`. Later notebooks with the same genotype parameters reuse that store.
Traits are simulated from bounded decoded blocks. The returned Dataset has phenotypes,
truth components, GRMs and `variant_info`, but no dense `genotypes` variable.

The notebooks pass the store path (`GENO`) to the APIs that stream genotypes: `vigor.grm`,
`vigor.build_grms`, `vigor.mvGWAS(Y, genotypes=GENO)`, `vigor.factorGWAS` and
`vigor.effective_tests(vigor.open_genotypes(GENO))`. Two routes still need a dense matrix,
so they decode only a bounded window with `vigor.genotype_panel`:

- 05 builds expert SNP features from an evenly spaced panel (`max_variants=1000`).
- 08 fine-maps one 120-variant region.

### Fine mapping → genes → PCSF (08, 10)

08 fine-maps an 80-variant window around a planted effect with the multi-trait
Bayesian fitter (`methods=("bayes",)`):

- the full residual trait covariance is fitted by variational ECM;
- per-effect evidence and LD-purity screens decide which credible sets are kept;
- a synthetic GTF adds positional gene priors and per-gene PIP summaries;
- `pcsf_export=True` writes allele-oriented betas, SNP→gene edges and PIP prizes.

10 solves PCSF on that export with `pcst_fast` (`VIGOR[network]`), falling back to the
greedy connected tree, which has no optimality guarantee. It links genes through
placeholder gene–gene edges, which are not evidence.

The GTF genes are tiled independently of the simulated truth. Gene summaries and the PCSF
selection are positional prioritisation, not causal-gene posteriors. Real analyses need an
assembly-matched GTF and evidence-labelled gene–gene, eQTL or regulatory links.

To try a larger cohort on a small machine, raise `n_samples`/`n_variants` in the setup cell.
Disk grows as `n_samples * n_variants / 4` bytes. Memory is dominated by the n × n GRMs; pass
`compute_grms=False` in `trait_kwargs` if you do not need them. Delete `vigor_tutorial_data/`
to regenerate the store.
