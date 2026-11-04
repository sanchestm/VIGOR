"""Shared numeric helpers used across the package."""
import numpy as np


def standardize_block(geno, center=True, scale=True, dtype=np.float64):
    """Center and optionally standardize each genotype column over observed calls.
    
    Parameters
    ----------
    geno : array-like of float, shape (n,p)
        Numeric genotypes; NaN is missing. Usually dosage 0/1/2; no dosage bounds
        are enforced. Infinity is treated as observed and propagates nonfiniteness.
    center : bool
        Default True subtracts each observed-column mean. False uses raw codes.
    scale : bool
        Default True divides by sqrt(sum(z²)/max(n_observed-1,1)). Without
        centering this is a raw second-moment divisor, not a standard deviation.
    dtype : NumPy floating dtype or str
        Default: ``np.float64``. Coercion/output dtype; integer dtypes cannot retain NaNs.
    
    Returns
    -------
    tuple
        z (n,p): centered/scaled codes with missing calls replaced by zero;
        sd (p,): divisor, NaN for zero-sum-of-squares columns when scaling, one
        when scale=False; mask (n,p): observed-call indicators in dtype.
        Zero-variance and all-missing columns yield zero z, not arbitrary dosage.
    
    Raises
    ------
    ValueError or TypeError
        Input cannot convert to dtype or does not support the two-dimensional
        axis operations. Shape is assumed rather than explicitly validated.
    
    Complexity
    ----------
    TIME O(np); SPACE O(np+p) for n samples and p variants.
    """
    geno = np.asarray(geno, dtype=dtype)
    mask = ~np.isnan(geno)
    n_obs = mask.sum(0)
    n_obs_safe = np.maximum(n_obs, 1)
    if center:
        mu = np.where(mask, geno, 0.0).sum(0) / n_obs_safe
        z = np.where(mask, geno - mu, 0.0)
    else:
        z = np.where(mask, geno, 0.0)
    if scale:
        ss = (z * z).sum(0)
        sd = np.sqrt(ss / np.maximum(n_obs - 1, 1))
        sd_safe = np.where(sd == 0, 1.0, sd)
        z = z / sd_safe
        z[:, sd == 0] = 0.0
        sd = np.where(sd == 0, np.nan, sd)
    else:
        sd = np.ones(geno.shape[1], dtype=dtype)
    return z.astype(dtype), sd.astype(dtype), mask.astype(dtype)


def dominance_encode(geno, dtype=np.float64):
    """Encode observed heterozygotes while retaining missing genotype calls.
    
    Parameters
    ----------
    geno : array-like of float, shape S
        Biallelic dosage codes: exactly 1 maps to one; every other non-NaN value
        maps to zero. Dosage bounds/hardcall status are assumed, not validated.
    dtype : NumPy floating dtype or str
        Default: ``np.float64``, used for input and returned codes; must represent NaN.
    
    Returns
    -------
    numpy.ndarray, shape S
        One at genotype==1, zero at all other observed entries, NaN at missing.
        This is a heterozygote indicator, not centered Vitezica dominance coding.
    
    Raises
    ------
    ValueError or TypeError
        Coercion fails or requested dtype cannot represent NaN.
    
    Complexity
    ----------
    TIME O(m); SPACE O(m), m genotype entries.
    """
    geno = np.asarray(geno, dtype=dtype)
    out = np.full(geno.shape, np.nan, dtype=dtype)
    m = ~np.isnan(geno)
    out[m] = (geno[m] == 1).astype(dtype)
    return out


def neglog10p(p):
    """Convert probabilities to finite -log10 values using the float64 tiny floor.
    
    Parameters
    ----------
    p : array-like of float, shape S
        Probabilities expected on [0,1]. Values below float64 tiny, including
        negative numbers and zero, are clipped to tiny; values above one to one.
        NaN propagates, positive infinity becomes zero, negative infinity the cap.
    
    Returns
    -------
    numpy.ndarray or numpy.float64
        Elementwise -log10(clip(p,tiny,1)), preserving shape; scalar inputs produce
        a NumPy scalar. Zero maps to about 307.65, not positive infinity.
    
    Raises
    ------
    ValueError or TypeError
        Values cannot convert to float64.
    
    Complexity
    ----------
    TIME O(m); SPACE O(m), with m probability entries.
    """
    p = np.asarray(p, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    return -np.log10(np.clip(p, tiny, 1.0))
