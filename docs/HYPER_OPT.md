# Compound covariance HPO in VIGOR 2.12.1

This implementation adds `genetic_varcov_PSD(..., hyper_opt=True,
hyper_opt_kwargs={...})`. It has been checked with targeted regression tests, a
48-animal/four-trait sanity run with Optuna 5 (search and deployment) and research
runs on larger cohorts. Windows process spawning, exhaustive resume equivalence and
statistical calibration of the selection rules have not been validated.

Install the existing optional `VIGOR[hpo]` dependency (Optuna >=4,<6) before use.
Ordinary fitting remains the default; the h2 prior is off unless `h2_prior_strength>0` or it is searched.

## Public use

```python
fit = vigor.genetic_varcov_PSD(
    kernels, traits, solver="lbfgs", hyper_opt=True,
    hyper_opt_kwargs={
        "seed": 42, "n_trials": 30,
        "search_timeout_s": 3600, "trial_timeout_s": 300,
        "final_refit_timeout_s": 900, "uncertainty_timeout_s": 600,
        "checkpoint_dir": "vigor_hpo", "resume": False,
        "cv": {
            "trait_projects": trait_projects,  # Series indexed by trait
            "sample_metadata": sample_metadata,  # DataFrame indexed by animal
            "designs": ["unseen_animal", "partial_cell", "few_shot_project",
                        "donor_ablation", "connectivity_stratified"],
            "robustness_blocks": ["family", "generation"],
            "target_calibration_fractions": [0.05, 0.10, 0.25],
            "inner_folds": 2,
            "weights": {"unseen_animal": .25, "partial_cell": .25,
                        "few_shot_project": .50},
            "average": "project_then_trait",
        },
        # Compact preset by default: integer log-range ranks, ridges, the
        # sparse auto fraction and the h2 prior. Add keys to search more.
        "search_space": {"factor_structure_G": ["fa", "sparse_fa"]},
        "n_jobs": 2,                     # parallel fold workers
        "pruning": {"startup_trials": 8},
        "warm_start_from": "previous_checkpoint_dir",
        "prediction_screen": {"solver_tol": 1e-4, "solver_maxiter": 500},
        "prediction_verify": {"solver_tol": 1e-6, "solver_maxiter": 1200},
        "early_stopping": {"min_trials": 12, "patience": 10,
                           "min_verified": 3, "min_relative_gain": .005},
        "progress": True,
    },
    analytic_se="final", n_boot=0, blas_threads=10,
)
if fit.attrs["hpo_final_refit_complete"]:
    predictions = vigor.predict_hyperopt(traits, kernels, fit)
```

The displayed budgets and rank grids are configuration examples, not measured
recommendations. Sparse few-shot targets may not support 5% calibration; the
mask preflight will identify the project/trait/budget that is impossible.
Default minimums are three training, three validation and three calibration
observations per target trait. Raise them for scientifically meaningful studies.

The h2 prior is fitted inside each candidate's L-BFGS objective. Its pilot
per-trait SEs come from Rademacher trace geometry per distinct training mask
(O(c² P n²), no animal-scale factorization), cached per fold and shared by all
candidates on that fold. Its strength and domain are chosen here because training
loss cannot identify a penalty weight. See [H2_PRIOR.md](H2_PRIOR.md).

## CV and leakage contracts

| Design | Behavior |
| --- | --- |
| `unseen_animal` | Withholds every phenotype of selected animals; scores genetic mean. |
| `partial_cell` | Withholds a fraction of observed cells while retaining at least one conditioning phenotype per eligible row; scores the genetic mean (primary) and the total phenotype prediction (secondary). |
| `trait_block` | Withholds an entire declared trait block on test animals, retaining another block. Requires `trait_blocks` or `trait_projects`. |
| `few_shot_project` | Freezes target test animals first, removes their phenotypes from all donor projects, then uses nested target calibration subsets. |
| `donor_ablation` | Adds target-only and each-donor target fits on exactly the same target tests/budgets; diagnostic, not extra objective weight. |
| Genetic-sharing-disabled baseline | Few-shot default diagnostic zeroes cross-project genetic covariances using a PSD block mask. It preserves residual covariance; it is not a complete no-residual-sharing model. Set `disable_cross_project_baseline=False` to omit explicitly. |
| `family_block`, `generation_block` | Withholds complete metadata groups across projects. |
| `batch_or_site_block` | Withholds complete measurement groups; `batch_column` defaults to `batch`. |
| `forward_generation` | Fits earlier generations and evaluates the next; string labels require `generation_order`. An outer design. |
| `unseen_project_same_trait` | Uses sample project/cohort membership to withhold a cohort measuring traits observed elsewhere. Requires `project_column` (default `project`). An outer design. |
| `frozen_embedding_probe` | Independently source-only HPO/representation fitting, frozen source genetic embeddings, then a separate target calibration linear probe. Use the outer evaluator. |
| `frozen_embedding_association` | Reports phenotype associations with source-frozen genetic embedding axes; no zero-shot trait prediction, naive p-values or rG claim. Use the outer evaluator. |
| `connectivity_stratified` | Reporting overlay using a fixed genotype-only reference sample, not another optimization loss. Operators without entry access report unavailable. |
| `project_addition_sensitivity` | Separately compares focal-only versus all-project covariance summaries after selection. It does not select parameters or establish unbiased h2. |

Complete family metadata with enough distinct groups adds family-blocked core
scenarios by default; an explicit robustness_blocks list can override this and
the manifest records a family-blocking opt-out.
`robustness_blocks=['family','generation']` additionally blocks test assignments
in core scenarios. In few-shot blocked scenarios, all donor phenotypes of test
families/generations are withheld. Partial/trait-block scenarios intentionally
retain conditioning phenotypes: they are not unseen-family genetic prediction.

Only finite-observation availability and supplied metadata determine masks.
Masks, seeds, exclusions and hashes are written before candidate generation.
Unlearnable source traits can be excluded according to the prespecified count
rule and are reported; impossible target traits fail preflight. Kernels and
genotypes are transductive fixed inputs; the implementation does not claim
training-frozen allele frequencies or PC mappings. Any outcome-dependent
transformations upstream of this API remain the caller's responsibility.

An intercept is fitted per trait using training cells. Optional numerical
`cv.covariates` are residualized with training-only least squares; their encoding
must be prespecified. No missing phenotype is imputed before fitting. Warm
starts, nested rank CV/HPO, ordinary bootstrap, analytic SE and structural SEM
are disabled within trials. h2-prior pilot SEs and prior variances are derived
only from each fold's training phenotypes. Missing outcomes stay missing in the
final authorized-data refit. `cv.protected_mask` can enforce this explicitly.

**Genetic prediction comes first.** Every design scores the genetic
prediction (intercept plus breeding value) against the held-out phenotypes;
that compound score is the primary objective. Where an animal's other traits
are observed (`partial_cell`, `trait_block`), the total phenotype prediction is
also scored. Total means genetic plus the residual conditioned on the animal's
own observed traits. It enters as a secondary term, `score = genetic +
total_prediction_weight x total`, with default weight 0.1.

Total prediction alone rewards a model that moves between-trait correlation
into E, because the animal's own phenotypes then carry the prediction. A
2-factor G can win that way. The genetic term keeps selection on the genetic
model. Pruning, `by_regime`/`by_project` scores, guardrails and the paired
cluster intervals all use the genetic objective. Set
`total_prediction_weight=0` for genetic only.

The primary objective first averages repeated scenarios/folds/budgets per
trait, then traits per project, then projects per regime, then uses the fixed
regime weights. No project gains weight merely by having more traits/cells.
Correlation, RMSE/MAE, baseline R2 and calibration intercept/slope are secondary
metrics. Predictive intervals/log scores are not fabricated from mean-only
predictions. Connectivity bands and donor baselines never duplicate primary
objective weight.

## Conditional search space

| Controls | Descriptor / condition |
| --- | --- |
| `rank_G`, `rank_E` | Integer range `{'low': 2, 'high': 'full', 'log': True}` (default), which TPE models as one ordered integer; a draw at or above the trait count means `'full'`. A categorical grid such as `[8, 16, 'full']` is still accepted. G supports `{kernel_name: descriptor}`. Rank zero is explicit diagonal FA. |
| `factor_structure_G/E` | `fa`, `pca`, `sparse_fa`, `sparse_pca`; G supports named component descriptors. |
| `factor_lambda_G/E`, `factor_group_lambda_G/E`, rotations | Only active for sparse geometry; positive numeric penalties or declared `auto` policy. |
| `factor_auto_fraction` | Positive log range, active when a selected sparse penalty uses automatic scaling. |
| `factor_adaptive`, `factor_relaxed_refit` | Sparse only; relaxed refit defaults fixed true. |
| `shrink_G/E` | Numeric or supported automatic prediction-retention policy. |
| `E_structure`, `min_E_pair_overlap` | Supported residual structure choices; overlap threshold active for overlap-dependent policies. Unsupported combinations fail explicitly. |
| `ridge_G/E` | Zero is a separate branch from positive log ranges. G supports named descriptors. Ranges are not universal scales. |
| `variance_closure`, strength | Off/auto; positive strength only when on. Closure constrains total variance, not genetic allocation alone. |
| `h2_prior_strength` | Nonnegative multiplier of the empirical-Bayes h2 prior precision inside the L-BFGS objective; 0 is the unpenalized control (default `[0, .5, 1, 2]`). |
| `h2_prior_groups` | `global` or `project` (`cv.trait_projects`), suggested only when the prior is active. Groups with fewer than three traits are pooled. |

**Presets.** `search_preset='compact'` (default) searches only `rank_G`,
`rank_E`, `ridge_G`, `ridge_E`, `factor_auto_fraction` (when a sparse penalty uses
automatic scaling) and the h2 prior. That keeps tens of trials informative. All
other controls are fixed at the `genetic_varcov_PSD` arguments you pass, unless
you name them in `search_space`. `search_preset='extended'` restores the full
historical space.

**Prediction settings are not TPE dimensions.** `shrink_G`, `shrink_E`,
`E_structure` and `min_E_pair_overlap` act after the covariance fit. During
screening, each candidate's fold fit is reused for every entry of
`prediction_grid`. The default tries, strongest shrinkage first:
- per-pair overlap/(overlap+s) with s = 100, 400 and 1600 (`'auto@100'`, `'auto'`
  = the 3.1.3 policy, `'auto@1600'`);
- a global off-diagonal retention of 0.25, 0.5 and 0.75;
- `'fisher_eb_full'`: data-driven per-pair Fisher-z empirical-Bayes shrinkage,
  without rank preservation;
- `'linear_analytic'`: the closed-form Schäfer–Strimmer linear intensity toward
  the diagonal, using each pair's overlap;
- `'model_eb@c'` for c = 0.5, 1, 2 and 4: model-based per-pair empirical Bayes.
  Each pair's sampling variance comes from the fitted model's own moment
  precision, so pairs where G and E are hard to separate shrink more;
  c rescales those variances;
- no shrinkage.

All entries use `shrink_G='auto'` except the unshrunk one.

Two rules keep this cheap:
- **Conditioning gate.** Before a variant's mvBLUP solve, the smallest
  eigenvalue of its residual correlation matrix is checked; it bounds the
  conditioning of V = K⊗G + I⊗E from below. A variant below
  `prediction_condition_floor` (default 1e-3) is skipped without a PCG solve.
- **Dropping settings that never compete** (`prediction_grid_drop=True`). After
  the start-up trials, a setting that has never won, nor come within 2% of a
  candidate's best, is not tried for later candidates.

Per-setting wins and skips are reported in `prediction_variant_stats`. Leave-one-out
shortcuts are not used: they score the total phenotype prediction, not the
genetic-first objective. The candidate keeps the variant with the
best screening score; verification and the final refit use that variant only.
Keys named in `search_space`, or set by an anchor, are left to TPE or the anchor.

Categorical descriptors are nonempty lists. Numeric descriptors use
`{'low': ..., 'high': ..., 'log': True, 'zero': True}`; `zero` is optional.
Mixed retention choices can contain one numeric descriptor, e.g.
`['auto', 1.0, {'low': .25, 'high': 1.0}]`. Named G descriptors must cover all
declared kernels. Separate G/E automatic penalty fractions, alternative h2-prior
estimators, marginal anchor penalties, shared/project-local architectures and
information-aware objectives remain experimental extensions and are rejected
as unsupported search keys. No kernel is silently removed or summed.

Solver stays L-BFGS on the standard MoM/score path. Numerical convergence,
precision, runtime, memory and uncertainty controls are fidelity/resource
settings, not biological search parameters. The exact reference uses float64,
full-kernel prediction. The covariance engine retains its configured float32
GEMMs/float64 reductions by default.

## Screening, verification and selection

All three anchors count against `n_trials`: caller's incumbent, diagonal FA,
and unshrunk prediction. TPE then suggests supported parameters.
- **Sampler:** multivariate/group TPE (`sampler_multivariate=True`) models
  interactions between parameters, such as rank with structure or prior
  strength, and handles conditional branches.
- **Start-up:** `sampler_startup_trials` (default 8) sets the number of initial
  random trials.
- **Seeding from a previous study:** `warm_start_from` takes a previous
  `checkpoint_dir` (its selection plus the best `warm_start_top` screened
  candidates) or a list of configurations. These are enqueued right after the
  anchors as ordinary Optuna trials, so TPE learns from them.
Screening covers one prespecified representative fold per regime/target
project, preferring the largest target calibration budget. Finalists use the
entire objective and paired-diagnostic schedule. Their scores are never ranked
against incomplete-fold or different-fidelity scores.

**Pruning.** While screening, a candidate reports its running score after each
wave of folds: the mean held-out NMSE so far, using its best prediction variant.
Optuna's median rule (`pruning={'enabled': True, 'startup_trials': 8,
'warmup_folds': 0}`) stops it once it is worse than the median of earlier
candidates at the same step. Every candidate runs the same folds in the same
order, so the comparison is like for like. Anchors and verification are never
pruned, and pruned candidates are never verified or selected. Failures and
timeouts are recorded as failed, not pruned.

**Parallel folds.** `n_jobs` runs the folds of one candidate in parallel worker
processes, in waves of `n_jobs`. Integer `blas_threads` settings are divided
between them, so set `blas_threads` to the cores you want in total. Each worker
holds its own copy of the kernels, so memory grows with `n_jobs`. Trials remain
sequential, which keeps the TPE history deterministic. An interrupt or failure
in one worker stops its siblings in the same wave.

A numerical preflight compares 1e-4 and 1e-6 prediction solves on the same
baseline covariance and representative schedule. It records actual residuals,
iterations, normalized prediction differences and NMSE differences. If the
configured thresholds fail, all screening uses verification tolerance. An
unconverged or nonfinite solve cannot complete verification. The default
thresholds are resource/accuracy policies, not a bound on scientific bias.

The incumbent is fully verified first. A fraction of the search wall budget is
reserved for finalist verification, with periodic verification during search.
The best numerical verified model and selected model remain separate. Promotion
defaults require >=0.5% relative improvement and a paired cluster-bootstrap
loss CI upper bound below zero; optional regime deterioration `guardrails` also
apply. If the best numerical candidate fails promotion, the controller considers
other verified candidates before retaining the incumbent. Default bootstrap
units are families if available, otherwise generations if supplied, otherwise
animals. Animal-unit intervals do not account for unprovided family dependence.
All intervals are conditional on inner selection, not untouched-test evidence.

## Monitoring, limits and recovery

The parent emits independent heartbeats while a spawn-safe worker fits moments,
FA/sparse support, predictions or calibration. `events.jsonl` records phases,
accepted covariance iterations, objective deltas, gradient norms, PCG iterations,
final residuals, worker RSS, active parameters and numeric/categorical changes.
Maxiter is a budget, not a completion percentage. Final fit diagnostics record
function evaluations/effective loading-column counts where available.

The supervisor enforces per-job, per-phase, trial and search wall limits, plus
an optional conservative worker/descendant RSS ceiling. It requests cooperative
cancellation, then terminates/kills processes after bounded grace periods.
BLAS threads are limited by the existing fitter; one trial runs at a time.
Independent final point-fit and uncertainty budgets are reserved.
There is at most one explicitly enabled numerical refit recovery, retaining the
same statistical model. Immutable base moments are cached by source/data/mask/
preprocessing/numerical identity, with bounded disk admission. Candidate closure
or ridge never mutates saved moments. Dense inputs are placed once per study
and reopened read-only; operators require trusted spawn-picklable representations.

On first Ctrl-C/stop-file cancellation, search stops and attempts the best fully
verified candidate's final authorized-data fit. An incomplete screening trial
is never promoted. A second interruption preserves settings and labels the
refit interrupted. A converged point fit is committed before uncertainty; a
failure in uncertainty preserves that point fit with explicit unavailable/partial
status. Forced OS termination cannot return a
Python result; atomic state and SQLite transactions support later recovery.

```python
fit = vigor.resume_hyperopt("vigor_hpo", final_refit_only=True,
                           final_refit_timeout_s=1800)
```

The checkpoint directory is trusted local state and contains internal pickles;
never resume one from an untrusted source. Source/data/mask/statistical identity
must match. Sampler seeds are derived per sequential ask. Equivalent completed
schedules are intended to resume reproducibly; killed RUNNING trials become
FAIL and can change subsequent suggestions. SQLite is for a suitable local
filesystem, not an assertion of safe shared-network distributed operation.
Scripts invoking spawn workers must use `if __name__ == '__main__':`.

## Results and durable artifacts

The normal result is an xarray Dataset with `hpo_status`,
`hpo_final_refit_complete`, best/selected/refit settings, selected view,
uncertainty/persistence status, and checkpoint/table paths.
No valid candidate yields a status-only Dataset, not invented zero matrices.
G/E/h2 and their SEs/intervals describe the selected (possibly h2-penalized)
estimator. Use `predict_hyperopt` to reuse the frozen nuisance mapping.

The checkpoint includes mask/input/source identities, Optuna SQLite state,
JSON/JSONL reports, per-trial/fold metrics, grouped losses, donor-gain/learning
curves, covariance diagnostics and trusted recovery pickles. Final labeled
matrices are written through the native consolidated sharded Zarr writer in a
supervised job, then atomically renamed and referenced by a complete manifest.
Interrupted stores keep an incomplete name/write state and are not published
as completed models. Persistence failure does not discard an in-memory fit.

## Independent outer evaluation

```python
outer_metrics = vigor.evaluate_hyperopt_outer(
    kernels, traits,
    outer_cv={"designs": ["family_block"],
              "sample_metadata": sample_metadata,
              "trait_projects": trait_projects, "inner_folds": 3},
    hyper_opt_kwargs={"n_trials": 30, "cv": {"designs": ["unseen_animal", "partial_cell"]}},
    covariance_kwargs={"solver": "lbfgs", "blas_threads": 10},
    out="outer_family_cv",
)
```

Each outer scenario gets independently training-only preprocessing, HPO,
calibration and refit. Source-frozen designs exclude all target trait outcomes
from representation HPO as well as fitting. Target calibration labels enter
only the subsequent frozen-feature probe. Outer scores are never fed back into
the inner optimizer. Forward/project/representation scenarios listed in an
ordinary search manifest are reserved descriptions, not reported generalization
results. Use this explicit, potentially expensive outer evaluator.

Known-truth h2/G/E/rG bias, RMSE and coverage, initialization stability, solver
preflight thresholds, family dependence, Windows/native stall behavior and
database recovery still need runtime validation. Search cannot guarantee an
unbiased covariance estimator or repair scientific confounding by itself.


Nuisance-adjusted covariance results describe residualized traits. This extension
uses fold-specific least-squares preprocessing; it does not replace the original
MoM estimator with arbitrary-covariate REML. Raw covariance uncertainty conditions
on the fitted nuisance transformation and omits its estimation uncertainty.


## h2 prior selection and Optuna 5

Optuna 4.x and 5.x are accepted; the installed version is included in the
checkpoint identity. Use a fresh checkpoint directory after changing the
Optuna version or search space. VIGOR uses the public
create/load/ask/tell API with keyword controls. Optuna 5 changes default TPE
behavior, so identical seeds across major versions do not imply identical
trials. See https://github.com/optuna/optuna/releases/tag/v5.0.0 and
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.study.Study.html.

The empirical-Bayes h2 prior is part of the covariance estimator, not a post-fit
view: every candidate with `h2_prior_strength>0` fits G/E with the prior inside
its L-BFGS objective, and the selected model is deployed as fitted. The
incumbent takes `h2_prior_strength`/`h2_prior_groups` from the
`genetic_varcov_PSD` call; two built-in anchors are unpenalized controls, so
every study compares prior and no-prior models. `report['h2_prior_comparison']`
summarizes verified scores with and without the prior. With `hyper_opt`, pass
`h2_prior_groups` as `'global'` or `'project'` only. Do not pass full-data
GCTA SEs into cross-validation; all prior quantities are training-fold estimates.
See [H2_PRIOR.md](H2_PRIOR.md) for the estimator, SEs and intervals.


## Workflow progress and fitting tolerance

See [PROGRESS.md](PROGRESS.md) for the persistent workflow display and the
h2-prior pilot stage. Screening uses looser prediction tolerance and, by
default, looser covariance-fit tolerance (`fit_screen={'gtol': 1e-5,
'ftol': 1e-7}`). Verified candidates and the final refit use the estimator
defaults (`fit_verify={}`), so selection and reported estimates never rest on
screening-precision fits. Keys you pass are merged into these defaults.

Within a training fold, candidates that differ only in post-fit settings
(`shrink_G`, `shrink_E`, `E_structure`) reuse one covariance fit. The cache key
is the exact moment system, the h2-prior weights, every optimizer argument,
the warm start and the optimizer limits, so any other difference refits.

**Warm starts.** With `warm_start_incumbent=True` (default), every candidate's
fold fit starts from the incumbent's fit (anchor 0) of the same fold. That fit is
re-factored to the candidate's rank (its top eigenvectors), not zero-padded.
Anchor 0 always completes all its folds first, so the starting point never
depends on trial order. The FA fit is nonconvex, so a warm start can reach a
different local optimum than a cold start. The final full-data refit starts cold.

**Iteration-capped fits are scored, not rejected.** A fold fit that stops at
its `maxiter`/`factor_sparse_maxiter`/`maxfun` cap is still a candidate: held-out
prediction scores exactly the matrices it produced, so a poorly converged fit
loses on its score. Only non-finite fits fail. Before 2.12.1 such fits failed the
trial; on a 281-trait panel with sparse FA and a 120-iteration proximal cap every
factor model failed, and the search stopped after five consecutive failures with
only the diagonal (rank 0) control verified. Convergence is still recorded in each
fold's `fit_diagnostics`, and `hpo_final_optimizer_converged` reports the final
refit (with a warning when it is capped). `require_converged=True` restores the
strict rule. Put cheap caps in `fit_screen` and larger budgets in `fit_verify`,
which also governs the final refit.

A search stopped by `max_consecutive_failures` now reports
`hpo_status='repeated_failures'` (it was reported as `'completed'`), warns, and
lists the failure messages in the report's `failure_reasons`.

A coarser coarse-to-fine fitting configuration can be supplied:

```python
hyper_opt_kwargs = {
    'fit_screen': {'gtol': 1e-4, 'ftol': 1e-5, 'maxiter': 300,
                   'factor_sparse_tol': 1e-4},
    'fit_verify': {'gtol': 1e-7, 'ftol': 1e-8, 'maxiter': 2000,
                   'factor_sparse_tol': 1e-7},
    'progress': True,
}
```

These are illustrative numerical budgets, not measured speedups. Preserve the
existing sparse iteration limits unless convergence evidence justifies a change.
Fitting tolerances and `h2_prior_probes` are separate controls. The pilot
geometry avoids animal-scale factorizations; more probes reduce Monte Carlo
noise in the prior weights at additional quadratic cost.
