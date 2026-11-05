"""VIGOR 2.12.1: xarray/Zarr-first scalable multivariate genomic inference.

The 2.8 series keeps the 2.6 low-rank / matrix-free fast paths and 2.7 audit fixes while making
correctness-sensitive fallbacks explicit and fixing audit findings around
calibration, circularity, provenance, leakage and serialization.
"""
from .io import PlinkReader, read_bim, read_fam
from .grm import (compute_loco_grms, compute_grm, iter_loco_grms,
                  compute_genomic_grms_from_plink1, genomic_features_from_plink1,
                  PlinkGenomicKernelBundle, genomic_kernel_operators_from_plink1)
from .vc import (
    estimate_h2, estimate_h2_many, score_variance_components,
    multivariate_score_transform, whiten_y, whiten_matrix, covariance_matrix,
    blup_resid,
)
from .assoc import scan_chromosome, scan_chromosome_gls_eigen, iter_chromosome_gls_eigen
from .classify import degree_of_dominance, classify_da, classify_inheritance, DEFAULT_THRESHOLDS
from .resources import ResourceConfig, ExecutionPlan, plan_execution

from .scalable import (
    StreamingGRMOperator, LowRankGRMOperator, MultiTraitCovarianceOperator,
    PedigreePrecision, candidate_pair_grm, pedigree_relationship_inverse,
    solve_multitrait_pcg, dense_grm_bytes, matrix_free_working_bytes,
)
from .multivariate import (
    genetic_varcov_PSD, prediction_covariance_view, mvBLUP_adaptive, organize_scores, decompose_covariance, decompose_GE, mvBLUP, mvBLUP_complete,
    mvPredict, mvPredict_adaptive, fit_prediction_covariance_cv, mvGWAS, mvGWAS_local, make_gxe_kernels, fit_gxe_mvblup,
    genomic_grms_from_genotypes, genomic_grms_from_npplink,
    multitrait_ridge_finemap, multitrait_elasticnet_finemap,
    multitrait_sparse_group_elasticnet_finemap, lightgbm_multitrait_finemap,
    run_mvsusie_finemap, R2_exact, compute_variant_ld, cluster_variant_ld,
    finemap_model_selection, ld_aware_finemap_shap, ld_group_permutation_shap,
    make_genomic_features,
)
from .multivariate import (
    FactorCovarianceSpec, DenseKernelOperator, LowRankKernelOperator,
    fit_mom_factor_analytic, fit_mom_sem_factor_analytic, select_mom_fa_rank, fit_mom_spectral, fit_mom_projected, fit_mom_spectral_refine, select_mom_spectral_rank, weighted_multi_moments,
    minque_moments_single_kernel, spectral_mom_analytic_se, fa_mom_analytic_se, multi_fa_mom_analytic_se,
    bootstrap_factor_analytic, PreparedMultivariatePlan, prepare_mvBLUP_plan,
)
from .prediction_sparse_cv import fit_sparse_prediction_covariance
from .hyperopt import predict_hyperopt, evaluate_hyperopt_outer, resume_hyperopt
from .factor_gwas_api import factorGWAS
from .latent import factor_gwas
from .finemap import FineMapResult, fine_map_region
from .finemap import (build_haplotype_radix_dag, haplotype_graph_tables,
                      haplotype_graph_variants, haplotype_graph_features,
                      write_haplotype_graph_zarr)
from .multivariate_ad import (
    multitrait_bayes_finemap_ad, multitrait_ridge_finemap_ad,
    multitrait_elasticnet_finemap_ad, lightgbm_multitrait_finemap_ad,
    finemap_colocalization, mixed_model_mediation_ad,
)
from .bootstrap import GeneticVarCovBootstrapCache, bootstrap_genetic_varcov_psd
from .causal import (
    CausalGraph, conditional_covariance, genetic_covariance_mediation,
    partial_correlation_from_cov, anm_pairwise_direction, anm_orient_candidates, pc_from_covariance, discover_ge_network,
    fit_multisource_covariance_sem, structural_sem_tabs, QC_FIGS,
)
from .kinship import pedigree2sparse, kinship_from_pedigree, pedigree_inbreeding
from .blup import (calculate_blups, calculate_component_blups, run_blup,
                   crossfit_nongenetic_predictions, genomic_pcs_from_grm)
from .annotation import *  # narrow re-export list is controlled by annotation.__all__
from .integration import *
from .network import *

from .npplink import (king_robust_kinship, king_robust_kinship_numba,
                       load_plink_packed_xarray, R2_packed, GRM_packed)

from .genotype_source import open_genotypes
from .data_model import open_genotype_dataset, write_genotype_zarr
from .store import VigorStore, DominoStore, open_store
from .results import result_dataset, result_table, as_frame, save_result, open_result
from .sparse_relationship import write_sparse_relationship_zarr, open_sparse_relationship_zarr, SparseZarrCSR
from .grm_store import (build_grms, build_grms_in_memory, stream_loco_grm,
                        merge_chromosome_grms, open_grm, cache_grm_spectral,
                        ZarrArrayKernelOperator, WeightedChromosomeGRMOperator)
from .packed_bed import (open_packed_bed_xarray, virtual_dosage, virtual_ld_matrix, virtual_offset_r2,
                         virtual_grm, dense_grm_from_bed, stream_loco_from_bed,
                         recommended_grm_variant_chunk)
from .gwas_dataset import (as_gwas_dataset, write_gwas_zarr, open_gwas_zarr,
                           export_gwas_parquet, select_hits)
from .haplotypes import founder_kernel, run_haplotype_gwas

from .api import mvGWAS
from .pipeline import run_gwas
from .zarr_io import ingest_genotypes, consolidate_genotypes
from .gwas import effective_tests, EffectiveTestsResult
from .gwas.exact import prepare_score_null, score_block, score_design_block
from .gwas.streaming import write_score_stream
from .covariance_calibration import select_prediction_rank
from .gwas.thresholds import GWASThresholds, EffectiveTestsMixin

from .residual_matrix import fit_residual_matrix_expert, predict_residual_matrix_expert
from .prediction_experts import (
    fit_blup_residual_experts, predict_blup_residual_experts,
    fit_blup_prediction_experts, predict_blup_prediction_experts,
)

from .simulate import (GenomicSimulationResult, simulate_genotypes, gcta_codings,
                       simulate_traits_from_genotypes, simulate_genomic_dataset,
                       simulate_from_plink1, simulation_scenario, write_simulation_plink,
                       simulate_genotypes_to_zarr, simulate_from_zarr)

__version__ = "2.12.1"

# Public exports computed after all imports, including GRM and factor APIs.

from .packed_experimental import packed_countsketch_factor, lowrank_from_countsketch, grm_from_factor
from .packed_experimental import packed_countsketch_factor_direct

from .expert_gates import gate_genomic_kernel_expert, predict_genomic_kernel_expert
from .attribution import mv_prediction_animal_attribution, linear_snp_attribution

from .factor_prediction import mvFactorPredict

from .grm_api import grm, GRMs

__all__ = [name for name in globals() if not name.startswith("_")]
