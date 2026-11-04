# Covariance, CV and h² prior progress

With progress=True, ordinary genetic_varcov_PSD calls own one persistent
fastprogress notebook display. Logical stages are alignment, moments, the
optional h² prior pilot, covariance fit, optional analytic uncertainty and
summaries. Nested optimizer/trace operations report into it instead of leaving
extra bars behind.

The optimizer row reports accepted iterations, the iteration budget, objective,
objective change and gradient norm. The budget is not a predicted stopping time.
The h² prior pilot reports mask number/total and probe count/total during shared
trace-geometry work. The analytic SE stage reports kernel factorization, each
Gaussian draw and the batched tangent projection.

HPO displays search attempts separately from the full workflow. It reports
trial number/budget, screening versus verification, fold number/total, design,
project, component ranks, h² prior strength/domain, objective diagnostics and
scores. Final model fitting and saving remain visible after the trial search.
Durable events.jsonl records retain the full parameters and event history.

Use progress=False for quiet ordinary calls. HPO inherits this setting unless
hyper_opt_kwargs['progress'] overrides it. Worker cancellation is checked
between probe blocks. A progress count describes work completed, not
statistical convergence or a wall-time ETA.

## Runtime

The h² prior pilot computes one rank-512 grm2Us eigensystem per kernel. That
eigensystem is shared with the `'grm2us'` sandwich and every HPO fold. The pilot
then does O(c² P n_m r) work per distinct observation mask, and HPO reuses the
geometry across all candidates of a fold. The prior then adds O(c t) per objective evaluation. The final
analytic sandwich (default `'paired'`) does two O(n²t) kernel products per
probe with no animal-scale factorization, using exactly `analytic_se_probes`
probes. Only with `analytic_se_mc_rtol` set does it add probe blocks, and then
the progress bar's total grows with each block.
`'cholesky'` adds one O(n³/3) factorization per kernel and `'grm2us'` an
O(n² r) randomized eigensystem. See [H2_PRIOR.md](H2_PRIOR.md).
