> VIGOR 2.12.1. Outputs are cleared; the notebooks were checked against the 2.12.1 API but not re-executed for this release. These notebooks need the private real26 fixture and `examples/real26_workflow.py`; the root and synthetic tutorials run without private data.

# Real26 entrypoints and source lineage

These are VIGOR 2.12.1 entrypoints for a private cohort. They consume a private,
immutable, SHA256-verified 7,616-animal × 26-trait fixture; no animal-level data
are distributed. Inputs and research caches are opened read-only. Outputs require
a fresh run directory so existing evidence cannot be silently replaced.

| Entry point | Preserved analysis purpose | Main path |
|---|---|---|
| `blup_comparison_real26.ipynb` | Matched heldout model comparison | Current automatic sparse G/E prediction, genuine independently fitted GDEC 3.1.3 and rrBLUP, shared-covariance solver parity, trait/split metrics and comparison plot |
| `mega_gcorr_real26.ipynb` | Covariance, native factors and trait-group exploration | Full real26 prediction, inference/prediction distinction, persisted native factors, G/E correlations, sign-invariant Ward clusters, optional discovery |
| `mega_gcorr_runtime_real26.ipynb` | Read completed results before computation; native animal projections | Named checkpoint opened before phenotype/kernel loading; explicit new fit, embedding and SE toggles; new-fit default only when no completed fit is supplied |

Original attachments are retained by the supervisor, with their source-only
hashes. Their 37, 26 and 29 cells (92 total) have a per-cell source SHA256 and
disposition in the private review evidence. This is a curated migration;
**the historical analyses listed as deferred below are not implemented or
validated by running these entrypoints**. Outputs were not copied from an old run.

## Configuration

Run a notebook from this source tree using the selected scientific Python kernel.
`VIGOR_SOURCE` selects the source root; otherwise the notebook searches its parent
directories for `vigor`. Set these environment variables before the first cell:

| Variable | Meaning/default |
|---|---|
| `VIGOR_REAL26_FIXTURE` | Frozen directory containing fixture.json, train.parquet, truth.parquet, holdout_masks.npz and additive_kernel.npy; default resolves the existing XPS validation layout beneath the home Documents/codex directory |
| `VIGOR_PLINK_PREFIX` | Read-only r11.2.1 BED/BIM/FAM for optional bounded discovery; default is the existing Domino dataset |
| `VIGOR_GDEC` | Supplied genuine genome_decomposition_v3_1_3.py; SHA256 and version must match |
| `VIGOR_RSCRIPT` | Rscript executable whose library contains rrBLUP; default uses the selected Python environment's bin, then PATH |
| `VIGOR_OUTPUT` | Parent of a **new** entrypoint-specific output directory; default vigor_outputs beneath the current directory |
| `VIGOR_THREADS` | Integer 1–20; default 2; the recorded 64 GiB resource target is descriptive, not an OS-enforced allocation limit |
| `VIGOR_RUN_COMPARATORS` | 1 by default in the comparison/mega workflows; 0 records SKIPPED; runtime workflow leaves comparators to the comparison entrypoint |
| `VIGOR_RUN_EXTENDED` | 0 by default; 1 attempts a 512-animal × 128-marker real panel with all 26 traits, direct-score LOCO factorGWAS, 24-marker single-trait Bayesian/ridge fine mapping, covariance mediation and PC |
| `VIGOR_COVARIANCE_FIT` | Runtime only: completed covariance Zarr opened before any input/kernel loading; default absent |
| `VIGOR_RUN_NEW_FIT` | Runtime only: default 0 if a completed fit is supplied, otherwise 1 |
| `VIGOR_RUN_EMBEDDINGS` | Runtime only: default follows RUN_NEW_FIT; 1 requires matching standardized phenotype digest |
| `VIGOR_RUN_SE` | Runtime only: default 0; 1 attempts eight-probe fixed-support analytic SE recovery using exact original standardized traits/kernel |

The frozen fixture pins an r11.2 kernel. **An r11.3.1 BED cannot be substituted**:
create a new paired phenotype/sample/GRM manifest first. The bounded panel checks expected full FAM/BIM hashes, source BED geometry and
all consumed BED byte rows before use. It stores those comparisons, sample-order
fingerprint and exact variant indices in private output. Unconsumed BED rows
are outside this bounded authentication scope. It is not a new whole-genome GRM. The stored GRM is
transductive; upstream phenotype pre-regression and family-independent splitting
have not been revalidated. Only training observations enter normalization and
inner covariance selection; truth is used only for scoring.

## Seventeen tutorial topics

The supplied Domino 2.11.5 tutorial tree actually contains seventeen root topics,
numbered **00–16**, rather than a separate `17_*.ipynb`. Existing synthetic
counterparts retain their known-truth/simulated purpose, and the real-HS routes
retain their distinct external pedigree/founder/genotype requirements. Their file
fixture executions do not count as full-cohort real26 validation.

| Original topic | Current VIGOR route | Real26 status/scope |
|---|---|---|
| 00 storage/data model | Labelled frozen fixture; save/open_covariance_fit | Main path; exact order/NaNs, checksummed inputs and saved native support |
| 01 chromosome GRM/LOCO | bounded_real_panel; factorGWAS background=loco | Small real r11.2 panel only; full chromosome A/D cache reconstruction deferred |
| 02 G/E/A/D/GxE | run_prediction; genetic_varcov_PSD | Real26 G/E; additional A/D/GxE kernels and model ladder deferred |
| 03 factor decomposition | native_factor_loadings; animal_factor_embeddings | Saved native factors, no post-hoc rotation/re-decomposition of sparse support |
| 04 mvBLUP | mvPredict plus exact raw-fixture regression | Whole-animal and within-animal real26 holdouts; genetic and phenotype targets kept separate |
| 05 experts | Existing prediction_experts tutorial/public methods | Nonlinear genomic/matrix expert ablations deferred; no fake expert gain claimed |
| 06 multivariate GWAS/Zarr | Existing streamed GWAS tutorial; bounded factorGWAS | Genome-wide trait scan deferred; default frozen kernel is not LOCO |
| 07 effective tests | Existing effective-tests tutorial and numerical tests | Calibration over all real variants deferred |
| 08 LD/fine mapping/SHAP | bounded fine_map_region, methods bayes/ridge | 24-marker real locus; SHAP and genome-wide LD structure deferred |
| 09 mediation/causal | genetic_covariance_mediation; pc_from_covariance | Exploratory covariance screen, no causal-identification claim |
| 10 network/PCSF/GNN | Existing network tutorial | External annotation/network interpretation and GNN real26 analysis deferred |
| 11 haplotype GWAS | Existing founder-probability tutorial | Suitable phased/founder real26 inputs not supplied to these entrypoints; deferred |
| 12 pedigree | Existing pedigree tutorial | External pedigree route remains separate; real26 pedigree validation deferred |
| 13 uncertainty | add_covariance_se toggle, existing bootstrap tutorial | Conditional on saved rank/support; full support-selection coverage unestablished |
| 14 end to end | Three individually named real26 notebooks | Full real26 core paths; synthetic benchmark remains separately labelled |
| 15 local haplotype graph | Existing bounded graph tutorial | Full real26 phased graph/annotation visualization deferred |
| 16 joint-trait fine mapping | Existing labelled joint-trait tutorial/public wrapper | Current bounded real stage uses one trait with training-only coverage selection; real multi-trait locus validation deferred |

## Execution and failure semantics

Covariance rank failures stay in `prediction.json`; success at full rank does
not convert failed ranks into passes. `extended_stages.json` records attempted
optional stages individually, and `extended.json` is PARTIAL if any fail.
Executing a disabled branch records SKIPPED and supplies no scientific evidence.
The initial posterior-score and direct-score factorGWAS attempts failed null-PCG
residual gates on the sparse panel; the tolerances were not relaxed.

Native animal scores condition on the animal's observed phenotypes and ignore
kinship; they are **not genetic BLUPs**. Frozen-split metrics are descriptive,
not independent confirmation of generalization or estimator superiority. Joint
inclusion does not establish colocalization, and covariance mediation/PC do not
establish causal effects. Full-server scans, ontology enrichment, expert SHAP,
new kernels, graph fine mapping and the original complete model ladder remain
explicit review gaps.
