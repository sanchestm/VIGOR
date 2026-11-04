# VIGOR 2.12.1 inputs and outputs

## Labeled scientific inputs

Phenotypes are a pandas DataFrame: unique animal rows, unique trait columns,
numeric Gaussian outcomes, NaN for unobserved cells. Quantitative pipeline scans
reject binary outcomes. Covariates are finite sample-by-column values; the CLI
one-hot encodes categorical columns. Intercepts are always projected in scans.
Duplicate labels after string conversion are rejected.

A labeled GRM is square on the animal axis and is explicitly aligned. Unlabeled
arrays/operators are positional. Named kernels require matching named genetic
trait covariance matrices; E is the residual trait covariance. Low-rank kernels
are accepted as supplied representations, not silently substituted into exact
scans. Factor directions are labeled on the phenotype trait axis.

Genotype sources yield sample-by-variant dosages plus unique variant IDs and
source metadata. Built-in sources include BED, biallelic PGEN and canonical
Zarr. Custom sources must preserve requested sample order and enumerate declared
variant IDs exactly once, even if their block DataFrame indices restart at zero.
BED validation checks the header, SNP-major mode and exact byte span. BED dosage
counts the declared effect allele. PGEN counts biallelic ALT. Zarr REF/ALT alone
does not establish counted-allele orientation: supply explicit effect/other allele
coordinates or a recognized counted-allele convention.

## Covariance/prediction Dataset

| Field | Axes | Meaning |
| --- | --- | --- |
| G, E, Rg | trait_i, trait_j | Fitted inference covariance/correlation |
| G_prediction, E_prediction | trait_i, trait_j | Explicit prediction regularization |
| G_components | component, trait_i, trait_j | Named genetic covariance coefficients |
| h2, h2_se, h2_ci95, covariance SE/CI fields | trait or trait-pair | Inference uncertainty of the fitted (possibly h²-penalized) estimator, when requested |
| h2_prior_mean, h2_prior_weight, h2_prior_tau2, h2_prior_kappa, h2_pilot, h2_pilot_se | trait | h² prior diagnostics when `h2_prior_strength>0` |
| h2_ci_critical_value, h2_ci_bias_m2 | trait | Robust EB interval critical values and bias second moments |
| phenotype_prediction, genetic_mean | sample, trait | Requested prediction outputs |
| attrs | metadata | Solver/configuration/uncertainty/provenance diagnostics; `h2_prior` summarizes the prior |

With the h² prior, G/E themselves are the penalized estimates; there is no
separate calibrated view. h² is diag(sum G)/diag(sum G + E) everywhere.
Other terminal functions follow the same array-variable/scalar-attribute policy;
inspect their own docstrings for exact keys. `result_table`/`as_frame` provide
explicit table conversions.

## Exact association Dataset

`variant` is the sequential scan-order coordinate; `variant_index` preserves the
physical source row. Exclusion groups may be interleaved in the input. `trait`
labels are trait names for one contrast, contrast:trait for A+D/founders, or factor
directions for loading-restricted scans.

| Fields | Axes | Definition |
| --- | --- | --- |
| beta, beta_se, z, stat | variant, trait | One-direction null score, other tested effects fixed at zero |
| p, log_p, neglog_p | variant, trait | Chi-square tail, natural log tail, −log10 tail |
| joint_beta, joint_beta_se | variant, trait | Full information-inverse coefficient estimates |
| conditional_stat, conditional_log_p, conditional_neglog_p | variant, trait | Conditional coefficient tests |
| nobs | variant, trait | Actually observed genotype/phenotype support in that direction |
| omnibus_stat, omnibus_dof | variant | Joint score and estimable information rank |
| omnibus_p, omnibus_log_p, omnibus_neglog_p | variant | Joint chi-square tail representations |
| q_snp, q_snp_dof, q_snp_* | variant | Optional nested full-trait minus factor-subspace score |
| variant_id, chromosome, position | variant coordinates | Physical source identifiers |
| effect_allele, other_allele | variant coordinates | Counted orientation when declared by source |

Nonestimable conditional coordinates are NaN; monomorphic/no-information loci
have rank zero and no fabricated p-value. Dense fixed-effect and marker designs
are RHS-blocked. There is no block genomic-control correction or top-hit-only
exactness. Default reductions/small solves/verified residuals are float64 while
large products use float32. An unmet solver tolerance raises.

`attrs['scan_group_metadata']` records factor axes per exclusion group.
`projection_out=` optionally writes posterior factor phenotypes into a separate
artifact under factor_state/<group>; this is exploratory within-animal projection,
not the phenotype used for exact score inference.

## Persistence and exports

`out=` streams a new sharded Zarr v3 store. Existing scan destinations are rejected.
String coordinates are variable-length. Failed/interrupted writes remain marked
writing; only completed artifacts may be opened for analysis. `open_gwas_zarr`
rejects an explicit incomplete marker unless allow_incomplete=True is requested
for inspection. Older unmarked stores remain readable. Zero statistics and zero
physical indices are not treated as missing.

`export_gwas_parquet` is an explicit blockwise long-table export. `select_hits`
and `build_loci` choose canonical neglog_p by default and retain a legacy
neglog_p_additive fallback for older datasets. Specify the desired conditional or
omnibus field explicitly when that is the scientific test of interest.
