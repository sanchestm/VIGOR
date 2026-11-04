# VIGOR 2.12.1 methods and estimands

## Covariance

Joint moment systems estimate genetic and residual trait covariances from the
supplied animal kernels and observed phenotype masks. FA fits LLᵀ+Psi directly;
PCA omits specific variance. Sparse structures select support and may refit it.
PSD constraints can alter diagonal/off-diagonal estimates. Fitted inference
matrices and prediction-retention policies are distinct. The optional
empirical-Bayes h² prior is part of the fitted objective, not a separate view.
A PSD fit is not proof of unbiased h² or accurate cross-project correlations.

Prediction-rank CV hides only training-scope observed cells and compares the same
eligible validation traits across ranks. It discards full-data warm starts and
disables nested HPO/bootstrap. That inner score does not establish
outer prediction performance or identify genetic correlations. The compound Optuna controller searches independent G/E and named-component
ranks, conditional structure/regularization and the h² prior strength/domain.
Cluster-aware promotion and separate untouched outer evaluation
are described in [HYPER_OPT.md](HYPER_OPT.md).

The h² prior adds κ_j(h²_j − μ_g)² to the L-BFGS moment objective, with
h² = diag(ΣG)/diag(ΣG+E), μ_g profiled jointly and κ_j set from per-trait pilot
moment SEs and the between-trait variance so that strength 1 matches normal–normal
EB shrinkage. The final analytic sandwich is the Newton linearization of the
constrained FA fit in its own moment metric. It includes:
- the loading residual curvature and the exact prior Hessian;
- the data dependence of the closure precision and of the prior's pilot h²,
  τ² and κ.

Moment covariances on the observed masks come, by default, from
factorization-free paired probes that keep the full kernel. `grm2us` and
`cholesky` Gaussian draws are alternatives. The sandwich conditions on rank,
support and prior strength. With the prior on, h², vG and vE use robust EB
intervals with average (not per-trait) coverage. See [H2_PRIOR.md](H2_PRIOR.md).

## Prediction

mvBLUP/mvPredict solve covariance systems on the observed cells and condition on
visible phenotypes under the declared information state. Training means, genotype
frequency fitting, covariance selection and expert feature selection must exclude
held-out outcomes. Whole-animal, hidden-cell and project/relatedness splits test
different generalization questions. No universal improvement over rrBLUP or a
pinned comparator is claimed by this source-only update.

## Association

V = sum_c K_c ⊗ G_c + I ⊗ E is restricted to observed phenotype cells. For the
fixed-effect design C, P = V⁻¹ − V⁻¹C(CᵀV⁻¹C)⁺CᵀV⁻¹. A marker effect design D has
U=DᵀPy and I=DᵀPD. The joint statistic UᵀI⁺U uses rank(I) degrees of freedom.
Per-coordinate score tests use U_j²/I_jj with other tested effects fixed at zero;
conditional coefficient tests use I⁺U and its estimable-coordinate covariance.
They answer different questions when information has off-diagonal structure.

Block PCG retains the full supplied covariance operator. Kronecker eig
preconditioning is an exact full-space inverse only under complete single-kernel
separability. Under masks/multiple kernels it is a solver approximation; numerical
residuals are verified in float64. Statistical exactness is conditional on fitted
covariances/directions and to the solver tolerance, not uncertainty integration.

A+D tests dosage and heterozygote contrasts jointly; founder scans test all F−1
reference-founder contrasts at each locus. Factor scans restrict original-trait
effects to fitted loading directions. Optional Q_SNP is the nested full-trait
minus factor-subspace score under the same null. Fixed genome-wide factor axes
remain a warned sensitivity analysis. Separate posterior factor targets are
exploratory downstream outcomes and do not inherit the mixed-model likelihood.

## Kernels, LD and pedigree

The dense float GRM backend accumulates a lower triangle through in-place SYRK.
MAF eligibility is decided before coding; missingness/count normalization follows
the named method. Pairwise LOCO sums raw numerator and count statistics before
normalization. Global HWE methods use scalar weights. Canonical BED decoding and
pairwise-complete floating LD have one arithmetic implementation per operation.
Explicit packed experiments and matrix-free operators have separate contracts.

Symmetric eigen decomposition zero-clips negative directions with a warning; it
never reflects them through absolute values. Pedigree kinship retains complete
recorded ancestry, handles unknown parents and rejects cycles/pruning requests.
KING uses the conservative between-family robust estimator and actual genotype
and heterozygosity counts. See implementation docstrings for normalization details.

## Downstream interpretation

Fine mapping, radix/LD/SHAP grouping, PC/ANM/SEM and graph integration retain their
specific assumptions. Co-inclusion is not automatically colocalization, unphased
radix branches are not proven recombination events, and covariance-directed
models are not interventions. Interval coverage of the h² prior can be checked by
simulation with `validation/coverage_h2_prior.py`.


HPO nuisance adjustment fits training-only least squares. Its covariance
outputs concern residualized traits; this does not add arbitrary-covariate
REML to the original covariance estimator. Raw covariance SEs condition on
these fitted nuisance transformations and do not propagate their estimation
uncertainty. Selection cannot guarantee unbiased h2/rG or remove confounding.
