"""GWAS utilities, including multiple-testing calibration."""

from .effective_tests import (
    EffectiveTestsResult,
    effective_tests,
    empirical_threshold_to_meff,
    meff_to_threshold,
)
from .ld import R2Block, _debiased_r2, _iter_off_diagonal_r2
from .thresholds import EffectiveTestsMixin, GWASThresholds
from .calibration import calibration_table_from_projects, fit_grm_calibration

__all__ = [
    "EffectiveTestsResult", "R2Block", "effective_tests",
    "empirical_threshold_to_meff", "meff_to_threshold",
    "_debiased_r2", "_iter_off_diagonal_r2",
    "EffectiveTestsMixin", "GWASThresholds",
    "calibration_table_from_projects", "fit_grm_calibration",
]

