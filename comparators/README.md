# VIGOR 2.12.1 comparator assets

This directory restores the comparator layer expected by the consolidated research package.

## Bundled comparator

- `rGblup.R` — GRM-based `rrBLUP` and `BGLR` benchmark driver. It validates animal IDs, aligns GRMs to phenotype order, and supports genetic-mean, breeding-value and conditional-phenotype prediction targets.

## External comparators (not vendored)

The release benchmark specification compares VIGOR against external software when installed by the analyst: GCTA, MTG2, MegaLMM, REGENIE/BOLT-LMM where applicable, official mvSuSiE/susieR for fine mapping, and OmicsIntegrator2 for PCSF/network analysis. VIGOR does not vendor or relabel these projects. Exact commands, versions and hardware must be frozen in the benchmark output before publication.

## 2.12.1 benchmarking rules

1. Use identical train/test animals, phenotype masks and variant sets.
2. Recompute training-dependent allele frequencies and relationship matrices within prediction folds when the comparator requires them.
3. Distinguish genetic-value prediction from phenotype prediction.
4. For multivariate association, use the corrected VIGOR 2.7.0 scorer; results from the audited block-local multivariate genomic-control path are not valid comparators.
5. Report wall time, peak RSS, convergence and accuracy/calibration together.
