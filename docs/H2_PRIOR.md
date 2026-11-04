# Empirical-Bayes h² prior inside the PSD fit

`genetic_varcov_PSD(..., h2_prior_strength=s, h2_prior_groups=g)` shrinks
heritabilities across traits **inside** the L-BFGS moment objective. There is no
post-fit calibration step, re-anchoring or separate calibrated view: `G`, `E`,
`h2`, their SEs and intervals all describe one fitted estimator.

```python
res = vigor.genetic_varcov_PSD(G, traits, solver='lbfgs',
    factor_structure_G='sparse_fa', factor_structure_E='sparse_fa', rank_G=150, rank_E=150,
    h2_prior_strength=1.0, h2_prior_groups=None,      # or a trait->domain Series
    analytic_se='final')                               # paired, matrix-free SEs
table = vigor.organize_scores(traits, res)['heritability_table']
# h2, h2_se, h2_ci_low, h2_ci_high, vG/vE with _se/_ci_low/_ci_high,
# h2_prior_mean, h2_prior_weight, h2_pilot, h2_pilot_se
```

## Estimator

The objective is the existing pairwise moment loss (with ridge and soft variance
closure) plus

    sum_j kappa_j (h2_j - mu_g(j))^2,   h2_j = sum_c G_c[jj] / (sum_c G_c[jj] + E[jj]).

h2 has the same definition as the output tables, and the same as GCTA's V(G)/Vp
on the GRM scale. The group mean `mu_g` is profiled exactly at every evaluation,
so it is estimated jointly by L-BFGS. The penalty also enters the sparse-FA
proximal support stage and the support-fixed refit.

Weights: `kappa_j = s * a_j * se_j^2 / (2 tau2_g)`.
- `a_j` is the curvature of trait j's diagonal moment system along h2 at fixed
  total variance (closure does not change it).
- `se_j` is the Gaussian sandwich SE of the closure-augmented per-trait moment h2.
- `tau2_g` is the DerSimonian–Laird between-trait variance of the pilot h2 in
  group g, floored at 1e-4.

For a trait whose diagonal is free (positive specific variance), the fitted
shrinkage weight is `1/(1 + s se_j^2/tau2_g)`. At `s=1` this is the
normal–normal EB weight `tau2/(tau2+se^2)`. Groups with fewer than three traits
are pooled.

Pilot SEs (`h2_prior_pilot`, default `'analytic'`) only set the prior weights.
- **`'analytic'` (default):** the Gaussian null approximation of the moment
  covariance, Cov(B) ≈ 2·Vp²·tr(AₐA_b). That trace matrix is already in the
  moments, so this needs no probes or kernel products and costs O(t c³). It is
  the multivariate form of the standard method-of-moments SE approximation
  (Visscher et al. 2014) and is most accurate at modest h².
- **`'probes'`:** the full Gaussian covariance from Rademacher trace probes
  (`h2_prior_probes`, default 64) per distinct observation mask. By default the pilot uses the same rank-512 grm2Us
approximation as the `'grm2us'` sandwich (`spectral_se_kernel_rank`):
- One randomized eigensystem per kernel is computed and shared by the pilot, the
  final sandwich and every `hyper_opt` fold.
- Each mask then costs O(c² P n_m r), not O(n²) per probe.

The approximation only affects the pilot SEs, which set the prior weights κ.
`spectral_se_kernel_rank='full'` uses the exact kernel instead: the probes of
many masks share wide float32 GEMMs, O(c² P n²) per mask. Neither path forms a
per-mask kernel copy, Cholesky or dense square. The plug-in covariance is the nonnegative
unconstrained per-trait moment solution. In `hyper_opt` the geometry is
cached per training fold.

Strength is a penalty weight, so training loss cannot choose it (the loss is
minimized at 0). `hyper_opt` searches `h2_prior_strength` in `[0, .5, 1, 2]` and
`h2_prior_groups` in `['global', 'project']` by held-out prediction. The
`gamma` correlation shrinkage of the former post-fit EB is removed: correlation
regularization comes from the existing sparse-FA, ridge and prediction-retention
controls.

Supported route: `solver='lbfgs'` with MoM/score moments, without
`structural_refine` and with `n_boot=0` (bootstrap refits would not re-estimate
the prior).

## Standard errors (`analytic_se='final'`)

One sandwich serves every FA geometry. It is the first-order (Newton)
linearization of the fitted estimator, `dphi = (J'(H+P)J + R2)^-1 J' dB_eff`:

- **The fit's own metric.** H is the pairwise moment Hessian: it couples every
  genetic component with E and weights trait pairs by their information. J is
  the joint FA/sparse-FA tangent.
- **Full Hessian, not Gauss–Newton.** R2 adds the residual curvature of the
  loadings (`2 R L` on the support, R the fitted gradient), which matters when
  a reduced-rank fit leaves a moment residual. P is the exact Hessian of the h2
  prior: its Gauss–Newton part plus `kappa_j (h_j - mu_g) d²h_j`, with mu profiled.
- **Gauss–Newton when the observed Hessian is indefinite.** Sparse FA can have
  directions that leave the fitted covariance unchanged to first order. One
  example is a loading with one or two non-zeros trading against its trait's
  specific variance. Along such a direction only the residual curvature acts,
  and a negative residual there makes the observed Hessian indefinite even at a
  converged fit. The sandwich then uses the expected-information (Gauss–Newton)
  Hessian; the residual term has zero expectation under the model. This is
  recorded in `analytic_se['hessian']` and `observed_hessian_indefinite`, with a
  RuntimeWarning.
- **Learned weights move with the data.** `dB_eff` adds to each moment draw the
  movement of every data-dependent weight:
  - the closure moment λ·V̂p and its precision λ = (n-1)/(2 V̂p²);
  - the pilot h2 → DerSimonian–Laird tau2_g → kappa_j → profiled mean chain of
    the prior.

  Pilot SEs and curvatures are held at their data values; they enter only
  through second-order terms.

The moment covariance comes from one of three backends
(`analytic_kernel_factor`):

| Backend | How | Cost per probe | Kernel |
|---|---|---|---|
| `'paired'` (default) | Independent masked Rademacher probes u, w: E[(u'AVw)(u'VBw)] = tr(AVBV) | 2 kernel products of n×t blocks (K u, K w), batched; no factorization, no extra n×n storage | full supplied kernel |
| `'grm2us'` | Gaussian draws and their moments on one `npplink.grm2Us` randomized eigensystem (rank `spectral_se_kernel_rank`, 'auto' = 512) plus a trace-matched isotropic remainder | O(n r t) per draw after one O(n² r) eigensystem; no n×n product per probe | approximate; `analytic_se['kernel_retained_trace_fraction']` |
| `'cholesky'` | Gaussian draws from one exact Cholesky factor per kernel | 2 triangular products (L·Z and L′·Y₀, half the cost of general ones) after one O(n³/3) factorization | exact |

Kernel products run in float32 with float64 reductions.

**Monte Carlo control.** Exactly `analytic_se_probes` probes (default 64) are
used. Each SE's relative Monte Carlo error is reported; it is about 9% at 64. To
target a precision instead, set `analytic_se_mc_rtol`. Probes are then added in
blocks of 32, up to `analytic_se_max_probes` (default 1024), until both hold:
- the 90th percentile of the monitored SEs' errors (G/E marginal variances and
  h2 of traits off the variance boundary) is at most the target;
- no monitored variance is negative.

The result reports:
- `n_probes`, `mc_target_met`, `se_relative_mc_error`,
  `se_relative_mc_error_max` and `se_relative_mc_error_q90`;
- `negative_variance_estimates`, the number of paired estimates that came out
  negative. These are reported as NaN, never clipped to zero.

**Tangent solve.** All draws of a block are solved by batched PCG, stopped on the
energy-norm error of the projected draw (`fa_se_cg_rtol`, default 1e-3). A
Hessian that is not positive definite on the tangent means the fit is not at a
local minimum. Both that and a solve that fails to converge make the SEs
unavailable with a reason.

The SEs condition on rank, sparse support, boundary specific variances and the
prior's strength and grouping.

## Intervals

| Quantity | Interval |
|---|---|
| h2 (prior off) | logit-scale Wald |
| h2 (prior on) | robust EBCI, clipped to [0, 1] |
| vG, vE (prior off) | log-scale Wald |
| vG, vE (prior on) | robust EBCI on the variance scale, clipped at 0 |
| rG | Fisher-z Wald (`g_corr_table.rG_ci_low/high`) |

The robust EBCI (Armstrong, Kolesár & Plagborg-Møller 2022, *Econometrica*)
uses `h ± cva(m2_j)·SE_j` with `m2_j = (1-w_j)² tau2_g / SE_j²`. It reduces to
1.96 when the prior is off or `tau2=0`.

**What the coverage claim covers.** The critical value bounds worst-case
non-coverage over prior shapes with that normalized-bias second moment. The
guarantee is *average* coverage across the traits of a group, under AKP's
assumptions: approximately normal estimates with the reported SEs, and a
consistently estimated bias moment. It is not a per-trait guarantee; a trait far
from its group mean can still under-cover. VIGOR's estimator is not AKP's linear
shrinker: the prior acts inside a coupled FA fit, with weights from a pilot.
Applying their critical values is therefore an approximation.
`validation/coverage_h2_prior.py` checks it by simulation under correlated
traits and missingness; run it before relying on the intervals for a new design.

With the prior on, normal-null h2 p-values are withheld. `significance` means
`h2_ci_low > 0`. Intervals cost O(t) after the sandwich.
