"""Copy to PROJECT/vigor_run/project_config.py and edit paths/settings."""
from pathlib import Path
import os,numpy as np
ROOT=Path(os.environ.get("VIGOR_PROJECT_ROOT",os.environ.get("DOMINO_PROJECT_ROOT",Path.cwd()))).resolve()
RUN=ROOT/"vigor_run";RUN.mkdir(exist_ok=True,parents=True)
RESULTS=RUN/"results";RESULTS.mkdir(exist_ok=True)
PARQUET=RUN/"parquet";PARQUET.mkdir(exist_ok=True)
ANALYSIS_STORE=RUN/"analysis.zarr"
GWAS_STORE=RUN/"gwas"/"mvgwas.zarr"
PLINK_PREFIX=ROOT/"genotypes"/"genotypes"
PROCESSED_DATA_CSV=ROOT/"processed_data_ready.csv"
TRAIN_CSV=ROOT/"trainset.csv"
TRAIN_RGBLUP_CSV=ROOT/"trainsetRGblup.csv"
TEST_ROWS_CSV=ROOT/"test_rows.csv"
TEST_ELEMS_CSV=ROOT/"test_elems.csv"
SEED=42;ROW_PCT=.30;ELEM_PCT=.10;MIN_R2_N=10
BLAS_THREADS=int(os.environ.get("SLURM_CPUS_PER_TASK",2))
MEMORY_GB=64
GRM_BLOCK_SIZE=4096;GRM_MIN_MAF=1e-4
GRM_COMPUTE_DTYPE=np.float32;GRM_ACCUM_DTYPE=np.float64;GRM_OUTPUT_DTYPE=np.float32
LOW_RANK_G=10;LOW_RANK_E=10;FULL_RANK_G="full";FULL_RANK_E="full";FULL_SOLVER="lbfgs"
VARIANCE_CLOSURE="auto";VARIANCE_CLOSURE_STRENGTH=1.
ANALYTIC_SE="final";ANALYTIC_PROBES=16;ANALYTIC_TRACE_PROBES=8
GXE_ENVIRONMENT_COLUMN=None;GXE_COMPONENTS=("additive",)
MVBLUP_TOL=1e-7;MVBLUP_MAXITER=1000;MVBLUP_PRECONDITIONER="diag";MVBLUP_ACCURACY="exact"
GENOMIC_EXPERT_MAX_VARIANTS=512;GENOMIC_EXPERT_TOP_K=64;GENOMIC_EXPERT_ALPHA=100.
GENOMIC_EXPERT_PRIMARY="screened_ridge";EXPERT_N_SPLITS=2;EXPERT_SELECTION_CV=2;EXPERT_CANONICAL_COMPONENTS=4
MATRIX_METHODS=("iterative_ridge",)
