"""Reference GRM normalization engines for VIGOR.

The output is an additive/dominance covariance and its exact denominator
weights.  Data blocks are sample-by-variant dosages (0/1/2 or NaN).
Only fit_mask animal genotypes set allele frequencies and moments; genomic
prediction test phenotypes are not involved.
"""
from __future__ import annotations


SUPPORTED_GRM_METHODS = ("vitezica", "van", "gcta0", "npplink")

# Canonical standalone chromosome dataset contract. The older scalar-only
# VigorStore envelope remains explicitly versioned as a legacy layout.
# Required coordinates: chromosome, sample_i, sample_j. Components have those
# three dimensions; global methods have scalar chromosome weights. Pairwise
# methods additionally retain numerator_<component> and
# pairwise_observed_counts_<component>, with three-dimensional weights.
GRM_SCHEMA_VERSION = "vigor.chromosome_grm.v1"


def validate_grm_method(method, components=("additive", "dominance")):
    """Inputs: method, components.
        Outputs: relatedness kernels, aligned metadata, model diagnostics or persisted dataset as documented below.
        Big-O: O(n²m) for GRM accumulation or O(n²) for kernel handling; lightweight validation uses O(n+m).
        Allowed inputs: PLINK1 genotypes, aligned IDs, finite dosages 0/1/2/NaN, or documented normalization identifiers.
        
        Compute and validate chromosome relationship matrices with documented inputs.
        
        Parameters
        ----------
        method : object
            GRM normalization choice: npplink is empirical standardized with pairwise missingness; gcta0 per-marker scaled; van VanRaden I global; vitezica legacy A and D.
        components : object
            Requested genetic components, additive and/or dominance, estimated with component-specific valid denominator weights. Default: ``('additive', 'dominance')``.
        
        Returns
        -------
        object
            Scientific matrix, metadata or xarray Dataset, preserving source sample and component coordinate alignment.
    """
    key = str(method).lower()
    if key not in SUPPORTED_GRM_METHODS:
        raise ValueError(f"unknown GRM method {method!r}; choose from {SUPPORTED_GRM_METHODS}")
    comps = tuple(str(c).lower() for c in components)
    if not comps or len(set(comps)) != len(comps) or set(comps) - {"additive", "dominance"}:
        raise ValueError("components must be a nonempty unique sequence of additive/dominance")
    return key, comps

