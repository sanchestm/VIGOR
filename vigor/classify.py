"""Classify variants by inheritance mode from the degree of dominance (d/a).

With additive code {0,1,2} and heterozygote code {0,1,0}, the joint model
    E[y] = mu + a * add + d * het
places the heterozygote at (midpoint of homozygotes) + d, so ``d`` is exactly
the dominance deviation and ``a`` the additive effect. The degree of dominance
|d/a| then maps to the classical categories, matching ADDO's scheme:

    |d/a| < ~0        additive            (A)
    0 < |d/a| < 1     partial dominance   (PD)
    |d/a| ~ 1         complete dominance  (D)
    |d/a| > 1         over-dominance      (OD)
"""
import numpy as np

DEFAULT_THRESHOLDS = (0.25, 0.75, 1.25)


def degree_of_dominance(beta_add_raw, beta_dom_raw):
    """Compute absolute dominance/additive effect ratios on raw genotype scales.
    
    Parameters
    ----------
    beta_add_raw : array-like of float, shape S
        Per-allele additive effects for dosage 0/1/2. Zero/nonfinite values yield NaN.
    beta_dom_raw : array-like of float, shape S
        Dominance deviations for heterozygote code 0/1/0, in the same outcome units.
        Supply the same shape as additive effects; broadcasting is not supported by
        subsequent boolean indexing. Nonfinite values yield NaN.
    
    Returns
    -------
    numpy.ndarray of float64, shape S
        Dimensionless abs(d/a); negative additive effects are valid. Zero-dimensional
        inputs produce zero-dimensional arrays, not Python scalars.
    
    Raises
    ------
    ValueError or IndexError
        Inputs cannot convert to float or shapes cannot share the indexing mask.
    
    Complexity
    ----------
    TIME O(m); SPACE O(m), with m the number of effects.
    """
    a = np.asarray(beta_add_raw, dtype=np.float64)
    d = np.asarray(beta_dom_raw, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(d) & (np.abs(a) > 0)
    out[ok] = np.abs(d[ok] / a[ok])
    return out


def classify_da(da, thresholds=DEFAULT_THRESHOLDS):
    """Assign magnitude-only inheritance categories from absolute d/a ratios.
    
    Parameters
    ----------
    da : array-like of float, shape S
        Dimensionless nonnegative ratios. NaN, infinities and negative values map NA.
    thresholds : sequence of three floats
        Default: ``DEFAULT_THRESHOLDS``. The (0.25, 0.75, 1.25) thresholds place A below first, PD from first to second,
        D from second to third, OD at/above third. Expected finite ordered
        thresholds; ordering is not validated and overlapping masks overwrite.
    
    Returns
    -------
    numpy.ndarray of object strings, shape S
        Each element is A,PD,D,OD or NA. Threshold equality enters the upper bin.
    
    Raises
    ------
    ValueError
        Thresholds do not unpack to three values or inputs cannot convert to float.
    
    Complexity
    ----------
    TIME O(m); SPACE O(m), m input ratios.
    """
    t_a, t_d_lo, t_d_hi = thresholds
    da = np.asarray(da, dtype=np.float64)
    out = np.full(da.shape, "NA", dtype=object)
    ok = np.isfinite(da) & (da >= 0)
    out[ok & (da < t_a)] = "A"
    out[ok & (da >= t_a) & (da < t_d_lo)] = "PD"
    out[ok & (da >= t_d_lo) & (da < t_d_hi)] = "D"
    out[ok & (da >= t_d_hi)] = "OD"
    return out


def classify_inheritance(beta_add_raw, beta_dom_raw, se_add_raw=None,
                         thresholds=DEFAULT_THRESHOLDS, stability_z=1.0):
    """Classify stable raw-scale effects by ratio magnitude and allele direction.
    
    Parameters
    ----------
    beta_add_raw, beta_dom_raw : array-like of float, identical shape S
        Raw per-allele additive effects a and heterozygote deviations d, in the same
        outcome units. Finite nonzero a is required for a stable ratio. Missingness
        remains NA; finite a=0 is UNSTABLE.
    se_add_raw : array-like of float, shape S, or None
        Optional additive SEs, default None. When supplied stability additionally
        requires finite SE and abs(a)>stability_z*SE; SE nonnegativity is assumed.
    thresholds : sequence of three floats
        Default: ``DEFAULT_THRESHOLDS``. The (0.25, 0.75, 1.25) thresholds are passed unchanged to classify_da for ratio bins.
    stability_z : float
        Default 1.0; strict multiplier for additive SE stability. Nonnegative finite
        values are expected but not validated. Ignored when se_add_raw is None.
    
    Returns
    -------
    tuple of four numpy.ndarray, each shape S
        signed float64 d/a, absolute float64 ratio, object coarse category,
        object mode. Stable modes are additive, partial_dominant_A1/A0,
        complete_dominant_A1/A0, overdominant_high or underdominant_low.
        For PD/D, positive signed ratio points toward dosage 2 (A1), negative
        toward dosage 0 (A0). OD high/low follows the sign of d. Finite unstable
        pairs receive coarse UNSTABLE and mode additive_near_zero; nonfinite pairs
        remain NA with NaN ratios. These are descriptive, not significance tests.
    
    Raises
    ------
    ValueError or IndexError
        Inputs cannot convert or elementwise shapes/indexing are incompatible.
    
    Complexity
    ----------
    TIME O(m); SPACE O(m) including classify_da, with m effect pairs.
    """
    a = np.asarray(beta_add_raw, dtype=np.float64)
    d = np.asarray(beta_dom_raw, dtype=np.float64)
    signed = np.full(a.shape, np.nan)
    stable = np.isfinite(a) & np.isfinite(d) & (np.abs(a) > 0)
    if se_add_raw is not None:
        se = np.asarray(se_add_raw, dtype=np.float64)
        stable &= np.isfinite(se) & (np.abs(a) > stability_z * se)
    signed[stable] = d[stable] / a[stable]
    magnitude = np.abs(signed)
    coarse = classify_da(magnitude, thresholds)
    mode = np.full(a.shape, "NA", dtype=object)
    mode[stable & (coarse == "A")] = "additive"
    for code, label in (("PD", "partial_dominant"), ("D", "complete_dominant")):
        select = stable & (coarse == code)
        mode[select & (signed > 0)] = f"{label}_A1"
        mode[select & (signed < 0)] = f"{label}_A0"
    over = stable & (coarse == "OD")
    mode[over & (d > 0)] = "overdominant_high"
    mode[over & (d < 0)] = "underdominant_low"
    unstable = np.isfinite(a) & np.isfinite(d) & ~stable
    coarse[unstable] = "UNSTABLE"
    mode[unstable] = "additive_near_zero"
    return signed, magnitude, coarse, mode
