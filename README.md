# VIGOR

**Scalable multivariate genetic covariance, breeding-value prediction and genetic factor analysis.**

VIGOR estimates genetic (G) and residual (E) covariance across many traits from a
relationship kernel (GRM), predicts breeding values and missing phenotypes, scans
genotypes for effects on traits and latent genetic factors, and links loci to traits,
genes and networks. It is built for large animal panels with patchy phenotyping:
hundreds of traits measured on different subsets of thousands of animals.

Version 2.12.1 · Python ≥ 3.11 · MIT licence · development status: beta.

## Installation

Install the latest version straight from GitHub:

```bash
pip install "git+https://github.com/sanchestm/VIGOR"
```

With the optional components most analyses use (hyperparameter search, Zarr/Parquet
I/O, notebook progress bars and plotting):

```bash
pip install "VIGOR[hpo,io,progress,plot] @ git+https://github.com/sanchestm/VIGOR"
```

To pin a commit or tag (recommended for a paper or a long-running project), append it
after `@`:

```bash
pip install "VIGOR[hpo,io,progress,plot] @ git+https://github.com/sanchestm/VIGOR@<commit-or-tag>"
```

To upgrade an existing install to the latest commit:

```bash
pip install --upgrade --force-reinstall --no-deps "git+https://github.com/sanchestm/VIGOR"
```

For development, clone and install in editable mode:

```bash
git clone https://github.com/sanchestm/VIGOR.git
```

```bash
pip install -e "./VIGOR[test,hpo,io,progress,plot]"
```

A conda environment with the scientific stack and VIGOR from GitHub:

```bash
conda env create -f environment.yml
```

| Extra | Adds |
|---|---|
| `hpo` | Optuna, for `hyper_opt=True` covariance search |
| `io` | Zarr/Parquet/PLINK2/VCF readers and writers |
| `progress` | notebook progress displays (fastprogress) |
| `plot` | hvPlot / HoloViews / Bokeh / Datashader plotting |
| `scale` | Numba kernels for very large panels |
| `ml` | LightGBM, SHAP and UMAP for the nonlinear prediction experts |
| `network` | PCSF network prioritisation |
| `test` | pytest, ruff, build and notebook execution tools |
| `cuda` | CuPy (optional GPU kernels; the exact score pipeline stays on CPU) |

## Quick start

`Y` is an animals × traits `DataFrame` (NaN = not measured) and `K` a GRM labelled by
animal ID (a `DataFrame`, or a dict of named kernels for A/D/G×E models).

```python
import vigor

fit = vigor.genetic_varcov_PSD(K, Y, rank_G='full', rank_E='full', solver='lbfgs',
                               shrink_E='auto', progress=True)
pred = vigor.mvPredict(Y, K, fit)                    # breeding values / missing phenotypes
scores = vigor.organize_scores(Y, fit)
h2 = scores['heritability_table'].to_pandas()        # h2 per trait
rg = vigor.as_frame(scores['g_corr_table'])          # genetic and residual correlations
```

Results are labelled `xarray.Dataset`s: matrices and vectors are variables, settings
and diagnostics are attributes (`fit['G'].to_pandas()`, `fit.attrs[...]`).
`vigor.as_frame(...)` and `vigor.result_table(...)` give tidy tables, and `save_path=`
writes a result as Zarr.

### Standard errors and the h² prior

```python
fit = vigor.genetic_varcov_PSD(
    K, Y, rank_G='full', rank_E='full', solver='lbfgs',
    analytic_se='final', analytic_se_probes=64,
    analytic_kernel_factor='grm2us', spectral_se_kernel_rank=512,   # low-rank kernel for SEs
    h2_prior_strength=1.0, h2_prior_groups=None,                     # empirical-Bayes h2 prior
    progress=True)
vigor.organize_scores(Y, fit)['heritability_table'].to_pandas()   # h2, h2_se, h2_ci_low / h2_ci_high
```

The h² prior shrinks noisy heritabilities toward their group mean inside the fit;
its intervals target average coverage across a group of traits.

### Sparse factor analysis

```python
fit = vigor.genetic_varcov_PSD(
    K, Y, rank_G=150, rank_E=150, solver='lbfgs',
    factor_structure_G='sparse_fa', factor_structure_E='sparse_fa',
    factor_rotation_G='varimax', factor_rotation_E='varimax',
    factor_lambda_G='auto', factor_lambda_E='auto', factor_relaxed_refit=True)

from vigor.phenome import native_factor_loadings
loadings = native_factor_loadings(fit, 'G')   # traits x factors, exact zeros outside the support
```

Sparse FA uses an adaptive lasso on the loadings followed by an unpenalised refit on
the selected support. It gives interpretable factors; dense FA (or no factor
structure) is usually better for prediction and for reported genetic correlations.

### Residual shrinkage for prediction

`shrink_E` controls how residual correlations are shrunk before mvBLUP:

| Value | Policy |
|---|---|
| `'auto'` | per-pair overlap/(overlap+400) (the GDEC 3.1.3 policy); `'auto@s'` sets the strength |
| `'eb'`, `'fisher_eb_full'` | Fisher-z empirical Bayes |
| `'model_eb'`, `'model_eb@c'` | model-based per-pair EB from the fit's own precision |
| `'linear_analytic'` | closed-form Schäfer–Strimmer linear shrinkage |
| a number in [0, 1] | one global off-diagonal retention |

### Covariance hyperparameter search

```python
fit = vigor.genetic_varcov_PSD(
    K, Y, solver='lbfgs', hyper_opt=True,
    hyper_opt_kwargs={
        'n_trials': 60, 'checkpoint_dir': 'hpo_run', 'resume': True,
        'cv': {'trait_projects': projects, 'inner_folds': 2,
               'designs': ['unseen_animal', 'partial_cell']},
        'n_jobs': 2,
        'fit_screen': {'maxiter': 2000, 'factor_sparse_maxiter': 120},
        'fit_verify': {'maxiter': 20000, 'factor_sparse_maxiter': 2000},
    })
fit.attrs['hpo_status'], fit.attrs['hpo_selected_params']
```

The search tunes ranks, ridges, the sparse penalty and the h² prior by held-out
genetic prediction. It picks residual shrinkage per candidate from a cheap inner
grid and supports pruning, warm starts, time limits and resume. Use
`vigor.predict_hyperopt` for the selected deployment view and
`vigor.resume_hyperopt` to recover an interrupted run.

### Multivariate GWAS

```python
gwas = vigor.mvGWAS('genotypes/genotypes', Y, model='add-dom')   # exact score test, LOCO kernels
```

The exact engine tests every variant in the full observed-space model. With complete
phenotypes and one kernel it diagonalises V once per LOCO kernel, so a block of
variants costs one matrix product rather than one solve per variant. With missing cells
it uses block PCG, with the fixed-effect projection cached.

For genome-wide discovery on large panels, `engine='fast'` scans one GEMM per block
against the projected dual of an exact null. It is calibrated with simulated null traits,
so real signal cannot deflate it:

```python
from vigor import mvgwas_fast
null = vigor.prepare_score_null(Y, K_loco, G, E)        # one per chromosome (LOCO kernel)
state = mvgwas_fast.state_from_vigor_null(null, rank=10)
scan = vigor.mvGWAS('genotypes/genotypes', engine='fast', state={'12': state}, chromosomes=['12'])
scan.top_variants
```

### Factor GWAS

Test variants against the genetic factors of a fitted model, with
leave-one-chromosome-out backgrounds:

```python
gwas = vigor.factorGWAS(
    traits=Y, genotypes='genotypes/genotypes',      # PLINK 1 prefix
    component='G', decomposition=fit, factors=20,   # test the fitted factors themselves
    factor_method='native', background='loco',
    out='factor_gwas.zarr')
```

How the default `method='grammar_gamma'` works:
- **Scores.** Factor scores are exact: they come from the LOCO mvBLUP dual, and missing
  cells are handled.
- **Information.** It is x'x·S, with S rescaled per factor by γ. γ is the median ratio to
  the exact information over about 200 strided variants per chromosome.
- **Hits.** Variants passing `rescore_p` (default 1e-5) are rescored with the exact engine
  and flagged in `exact_rescored`.
- **Covariance.** G, E and the axes are estimated once, or taken from `decomposition`,
  rather than refitted per chromosome.

Other modes:
- `method='exact'` scores every variant with the exact engine.
- `loco_factor='refit'` re-estimates covariance and factors per chromosome group.

## Examples

| Folder | What it shows |
|---|---|
| [`examples/tutorials/`](examples/tutorials/README.md) | 17 topic notebooks (00–16): data model and storage, chromosome GRMs and LOCO, G/E/A/D/G×E covariance, factor decomposition, mvBLUP, prediction experts, multivariate GWAS on Zarr, effective tests, fine mapping with LD-aware SHAP, mediation, networks, haplotype GWAS, pedigree kinship, bootstrap uncertainty, end-to-end runs |
| `examples/tutorials/synthetic/` | the same topics on simulated data (run anywhere, no inputs needed) |
| `examples/tutorials/real_hs/` | the same topics on your own project; copy [`examples/project_config.py`](examples/project_config.py) into `<project>/vigor_run/` and edit the paths |
| [`examples/real26/`](examples/real26/MIGRATION.md) | full-cohort benchmark entrypoints (need a private 7,616-animal × 26-trait fixture) |
| [`examples/quickstart.py`](examples/quickstart.py) | simulate a small PLINK dataset and run VIGOR end to end |
| [`examples/compound_hyperopt.py`](examples/compound_hyperopt.py) | command-line `hyper_opt` run on your own kernel (`.npy`) and trait CSV |
| [`comparators/`](comparators/README.md) | GDEC 3.1.3 and the rrBLUP/BGLR R driver used as benchmarks |

The notebooks were checked against the 2.12.1 API but their outputs are cleared and
they were not re-executed for this release.

## Documentation

Every public function has a NumPy-style docstring covering its estimand, inputs,
outputs and cost, for example `help(vigor.genetic_varcov_PSD)`,
`help(vigor.factorGWAS)` or `help(vigor.mvGWAS)`.

## Command line

Installing VIGOR adds the `vigor` and `vigor-blup` commands (`vigor --help`).

## Issues

Please report bugs and questions at <https://github.com/sanchestm/VIGOR/issues>.
