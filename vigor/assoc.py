"""Vectorised association kernels and the streaming per-chromosome scan.

All statistics for a block of variants against all (whitened) traits are
computed with a handful of matrix products, so throughput is set by BLAS, not
by a Python loop over SNPs. The four tests mirror ADDO and then some:

  * additive              additive code vs null            (marginal)
  * dominance (marginal)  heterozygote code vs null        (marginal)
  * add_joint / dom_joint additive & dominance in one model (conditional)
  * avsad                 additive vs additive+dominance    (1-df F test)

plus the degree of dominance and inheritance class per variant.

The genotype math is ported from the (machine-precision-validated) kernels in
npplink; effect sizes are additionally rescaled to the raw allele scale so the
degree of dominance is a genuine d/a rather than a ratio of standardized stats.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as _t, f as _f

from ._utils import standardize_block, dominance_encode, neglog10p
from .resources import resolve_variant_chunk
from .backend import create_projection_backend
from .classify import degree_of_dominance, classify_da, classify_inheritance


def _regress_single(Xz, Y, Xmask, Ymask, center=True, rank_c=1):
    """Single-predictor regression of every column of ``Y`` on every column of
    ``Xz``. Returns beta, se, stat (t), neglog10 p, df; each shape (m, t).

    Parameters
    ----------
    Xz : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    Xmask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    Ymask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``.

    rank_c : int
        Rank of the fixed-covariate design already projected from the regression; nonnegative
        counts determine residual degrees of freedom. Default: ``1``.
    """
    XtY = Xz.T @ Y
    diag = (Xz * Xz).T @ Ymask
    ssr = Xmask.T @ (Y * Ymask) ** 2
    df = Xmask.T @ Ymask - (int(rank_c) + 1 if center else int(rank_c))
    df = np.where(df <= 0, np.nan, df)
    diag = np.where(diag <= 0, np.nan, diag)
    beta = XtY / diag
    ssr = ssr - beta * beta * diag
    se = np.sqrt(ssr / (df * diag))
    stat = beta / se
    p = 2.0 * _t.sf(np.abs(stat), df=df)
    return beta, se, stat, neglog10p(p), df


def _regress_joint(Az, Dz, Y, mask, Ymask, center=True, rank_c=1):
    """Joint additive+dominance regression, plus the additive-vs-AD F test.

    Parameters
    ----------
    Az : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.

    Dz : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    mask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    Ymask : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``. Passed to ``_regress_single`` as
        ``center``.

    rank_c : int
        Rank of the fixed-covariate design already projected from the regression; nonnegative
        counts determine residual degrees of freedom. Default: ``1``. Passed to
        ``_regress_single`` as ``rank_c``.
    """
    x1ty = Az.T @ Y
    x2ty = Dz.T @ Y
    s11 = (Az * Az).T @ Ymask
    s22 = (Dz * Dz).T @ Ymask
    s12 = (Az * Dz).T @ Ymask
    yty = mask.T @ (Y * Ymask) ** 2
    nobs = mask.T @ Ymask

    det = s11 * s22 - s12 * s12
    det = np.where(det <= 0, np.nan, det)
    inv11, inv22, inv12 = s22 / det, s11 / det, -s12 / det

    b_add = inv11 * x1ty + inv12 * x2ty
    b_dom = inv12 * x1ty + inv22 * x2ty
    rss_ad = yty - (b_add * x1ty + b_dom * x2ty)
    rss_ad = np.where(rss_ad < 0, 0.0, rss_ad)
    df_j = nobs - (int(rank_c) + 2 if center else int(rank_c) + 1)
    df_j = np.where(df_j <= 0, np.nan, df_j)
    sig2 = rss_ad / df_j

    se_add = np.sqrt(sig2 * inv11)
    se_dom = np.sqrt(sig2 * inv22)
    stat_add = b_add / se_add
    stat_dom = b_dom / se_dom
    p_add = 2.0 * _t.sf(np.abs(stat_add), df=df_j)
    p_dom = 2.0 * _t.sf(np.abs(stat_dom), df=df_j)

    # additive-only residual SS (for the F test) via the marginal additive fit
    b_a, se_a, _, _, df_a = _regress_single(Az, Y, mask, Ymask, center=center, rank_c=rank_c)
    rss_add = (se_a ** 2) * df_a * np.where(np.isnan(s11), np.nan, s11)
    num = np.where(rss_add - rss_ad < 0, 0.0, rss_add - rss_ad)
    fstat = num / np.where(df_j > 0, rss_ad / df_j, np.nan)
    p_avsad = _f.sf(fstat, 1, df_j)

    return {
        "beta_add_joint": b_add, "se_add_joint": se_add,
        "stat_add_joint": stat_add, "neglog_p_add_joint": neglog10p(p_add),
        "beta_dom_joint": b_dom, "se_dom_joint": se_dom,
        "stat_dom_joint": stat_dom, "neglog_p_dom_joint": neglog10p(p_dom),
        "f_avsad": fstat, "neglog_p_avsad": neglog10p(p_avsad),
        "df_joint": df_j,
    }


def _assemble(metrics, bim_slice, trait_names):
    """Ravel (m, t) metric arrays into a long tidy DataFrame.

    Parameters
    ----------
    metrics : array-like, pandas.DataFrame or mapping
        Association/evaluation quantities used to rank or report candidates. Match identifiers
        to the corresponding variants, traits or feature groups; required fields are selected
        explicitly in the body.

    bim_slice : object (see the cited input contract)
        Input for the operation ``bim_slice['snp']``. See ``vigor.assoc._assemble`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.
    """
    m, t = next(iter(metrics.values())).shape
    base = {
        "variant_index": np.repeat(bim_slice.index.to_numpy(dtype=np.int64), t),
        "snp": np.repeat(bim_slice["snp"].values, t),
        "chrom": np.repeat(bim_slice["chrom"].values, t),
        "pos": np.repeat(bim_slice["pos"].values, t),
        "trait": np.tile(np.asarray(trait_names, dtype=object), m),
    }
    for name, arr in metrics.items():
        base[name] = np.asarray(arr).ravel(order="C")
    return pd.DataFrame(base)


def scan_chromosome(reader, chrom, Ystar, trait_names, sample_index,
                    block_size='auto', model="add-dom", center=True, scale=True, rank_c=1,
                    da_thresholds=(0.25, 0.75, 1.25), maxmem=None, memory_fraction=None):
    """Scan every variant on ``chrom`` against the whitened trait matrix
    ``Ystar`` (shape (n, t), already mean-centred).

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    Ystar : array-like
        Transformed phenotype matrix, animals/eigen-directions × traits, after the
        whitening/eigenbasis step used by this scan. Its axes must match the prepared null
        state.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings).

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Default: ``'add-dom'``. Source-recognized literal values: ``additive``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    center : bool
        Whether to subtract the observed/training mean along the modeled observation axis.
        Disable only when inputs have already been centered using training data or a
        zero-intercept model is intended. Default: ``True``. Passed to ``standardize_block`` as
        ``center``.

    scale : float or array-like
        Numerical scale/retention/step threshold controlling the transformation described above.
        Match vector scales to the affected axis; the exact formula and local comparisons below
        determine its admissible domain. Default: ``True``. Passed to ``standardize_block`` as
        ``scale``.

    rank_c : int
        Rank of the fixed-covariate design already projected from the regression; nonnegative
        counts determine residual degrees of freedom. Default: ``1``. Passed to
        ``_regress_single`` as ``rank_c``.

    da_thresholds : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``(0.25, 0.75, 1.25)``. Passed to ``classify_da``
        as ``thresholds``.

    maxmem : int, float, unit string or None
        Working-memory budget for sizing ``block_size='auto'``; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``pd.concat(parts, ignore_index=True)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    sidx = np.asarray(sample_index)
    n, t = Ystar.shape
    Ymask = np.ones((n, t), dtype=np.float64)
    idx_c = reader.chrom_variant_index(chrom)
    parts = []
    # ~6 float64 n-columns per variant (g, add_z, dom, dom_z, regress scratch) + masks
    block_size = resolve_variant_chunk(block_size, n, bytes_per_call=52, bytes_per_variant=320 * t, maxmem=maxmem,
                                       memory_fraction=memory_fraction, n_variants=len(idx_c))
    for bim_s, g in reader.iter_blocks(block_size, variant_index=idx_c, sample_index=sidx):
        add_z, add_sd, add_mask = standardize_block(g, center=center, scale=scale)
        if model == "additive":
            beta, se, stat, nlp, df = _regress_single(add_z, Ystar, add_mask, Ymask, center, rank_c=rank_c)
            beta_raw = beta / add_sd[:, None]
            metrics = {"beta_additive_raw": beta_raw, "beta_additive": beta,
                       "se_additive": se, "stat_additive": stat,
                       "neglog_p_additive": nlp, "n_obs": add_mask.sum(0)[:, None] + np.zeros((1, t))}
            parts.append(_assemble(metrics, bim_s, trait_names))
            continue

        dom = dominance_encode(g)
        dom_z, dom_sd, dom_mask = standardize_block(dom, center=center, scale=scale)
        mask = np.minimum(add_mask, dom_mask)

        b_add, se_add, stat_add, nlp_add, df_add = _regress_single(add_z, Ystar, mask, Ymask, center, rank_c=rank_c)
        b_dom_m, se_dom_m, stat_dom_m, nlp_dom_m, _ = _regress_single(dom_z, Ystar, mask, Ymask, center, rank_c=rank_c)
        joint = _regress_joint(add_z, dom_z, Ystar, mask, Ymask, center, rank_c=rank_c)

        a_raw = joint["beta_add_joint"] / add_sd[:, None]
        d_raw = joint["beta_dom_joint"] / dom_sd[:, None]
        da = degree_of_dominance(a_raw, d_raw)
        klass = classify_da(da, thresholds=da_thresholds)

        metrics = {
            "neglog_p_additive": nlp_add, "stat_additive": stat_add,
            "beta_additive": b_add, "beta_additive_raw": b_add / add_sd[:, None],
            "neglog_p_dominance_marginal": nlp_dom_m, "stat_dominance_marginal": stat_dom_m,
            "neglog_p_add_joint": joint["neglog_p_add_joint"],
            "neglog_p_dom_joint": joint["neglog_p_dom_joint"],
            "neglog_p_avsad": joint["neglog_p_avsad"], "f_avsad": joint["f_avsad"],
            "beta_add_joint_raw": a_raw, "beta_dom_joint_raw": d_raw,
            "degree_of_dominance": da, "dominance_class": klass,
        }
        parts.append(_assemble(metrics, bim_s, trait_names))
    return pd.concat(parts, ignore_index=True)


def _residualize_matrix(x, design, information_inverse):
    """Remove the fitted fixed-design projection from each input column.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``x - design @ (information_inverse @ (design.T @ x))``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    design : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    information_inverse : array-like
        Inverse/pseudoinverse of the fixed-effect information matrix, covariate × covariate. It
        must correspond to the design in the same whitened observation metric.
    """
    return x - design @ (information_inverse @ (design.T @ x))


def scan_chromosome_gls(reader, chrom, phenotypes, trait_names, sample_index,
                        whitening_matrices, fixed_designs, block_size='auto',
                        model="add-dom", da_thresholds=(0.25, 0.75, 1.25),
                        min_genotype_count=0, stability_z=1.0, maxmem=None, memory_fraction=None):
    """Exact GLS scan after transforming every column of the complete design.

    Missing genotype calls are mean-imputed within a variant after recording
    observed AA/AB/BB counts. Variants failing ``min_genotype_count`` in any
    observed genotype cell receive missing statistics and a ``FILTERED`` mode.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis.

    whitening_matrices : sequence of array-like
        Per-trait observation-space whitening matrices. Each square matrix corresponds to that
        trait's finite observations and variance fit.

    fixed_designs : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings).

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Default: ``'add-dom'``. Source-recognized literal values: ``additive``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    da_thresholds : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``(0.25, 0.75, 1.25)``. Passed to
        ``classify_inheritance`` as ``thresholds``.

    min_genotype_count : int
        Minimum observed calls supporting a variant test; nonnegative counts filter low-support
        variants before reporting effects. Default: ``0``. Source comparisons:
        ``min_genotype_count > 0``. See the rejection/fallback branches for enforcement.

    stability_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``1.0``. Passed to ``classify_inheritance`` as ``stability_z``.

    maxmem : int, float, unit string or None
        Working-memory budget for sizing ``block_size='auto'``; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``pd.concat(parts, ignore_index=True)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    sidx = np.asarray(sample_index)
    y = np.asarray(phenotypes, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, n_traits = y.shape
    if n_traits != len(trait_names):
        raise ValueError("trait_names does not match phenotype columns")

    transformed = []
    for j in range(n_traits):
        L = np.asarray(whitening_matrices[j], dtype=np.float64)
        Cstar = L @ np.asarray(fixed_designs[j], dtype=np.float64)
        info_inv = np.linalg.pinv(Cstar.T @ Cstar)
        rank_c = int(np.linalg.matrix_rank(Cstar))
        ystar = L @ y[:, j]
        yres = _residualize_matrix(ystar[:, None], Cstar, info_inv)[:, 0]
        transformed.append((L, Cstar, info_inv, rank_c, yres, float(yres @ yres)))

    idx_c = reader.chrom_variant_index(chrom)
    parts = []
    # float64 n-columns per variant: g, add_z, dom, dom_z, L@add_z, A, A*A, D, D*D (+ ==0/1/2 masks)
    block_size = resolve_variant_chunk(block_size, n, bytes_per_call=76, bytes_per_variant=320 * n_traits, maxmem=maxmem,
                                       memory_fraction=memory_fraction, n_variants=len(idx_c))
    for bim_s, g in reader.iter_blocks(block_size, variant_index=idx_c, sample_index=sidx):
        g = g.astype(np.float64, copy=False)
        add_z, add_sd, add_mask = standardize_block(g)
        observed = add_mask.sum(axis=0).astype(float)
        count_aa = np.sum(g == 0, axis=0).astype(float)
        count_ab = np.sum(g == 1, axis=0).astype(float)
        count_bb = np.sum(g == 2, axis=0).astype(float)
        allele_frequency = np.nansum(g, axis=0) / np.maximum(2.0 * observed, 1.0)
        maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
        eligible = np.isfinite(add_sd)
        if min_genotype_count > 0:
            eligible &= np.minimum(np.minimum(count_aa, count_ab), count_bb) >= min_genotype_count

        dom = dominance_encode(g)
        dom_z, dom_sd, _ = standardize_block(dom)
        eligible &= np.isfinite(dom_sd)
        m = g.shape[1]
        metric_names = [
            "neglog_p_additive", "stat_additive", "beta_additive",
            "beta_additive_raw", "neglog_p_dominance_marginal",
            "stat_dominance_marginal", "neglog_p_add_joint",
            "neglog_p_dom_joint", "neglog_p_avsad", "f_avsad",
            "beta_add_joint_raw", "se_add_joint_raw", "beta_dom_joint_raw",
            "degree_of_dominance", "degree_of_dominance_abs",
        ]
        metrics = {name: np.full((m, n_traits), np.nan) for name in metric_names}
        metrics.update({
            "n_obs": np.repeat(observed[:, None], n_traits, axis=1),
            "count_AA": np.repeat(count_aa[:, None], n_traits, axis=1),
            "count_AB": np.repeat(count_ab[:, None], n_traits, axis=1),
            "count_BB": np.repeat(count_bb[:, None], n_traits, axis=1),
            "maf": np.repeat(maf[:, None], n_traits, axis=1),
            "genotype_filter_pass": np.repeat(eligible[:, None], n_traits, axis=1),
            "dominance_class": np.full((m, n_traits), "FILTERED", dtype=object),
            "inheritance_mode": np.full((m, n_traits), "FILTERED", dtype=object),
        })

        for j, (L, Cstar, info_inv, rank_c, yres, yty) in enumerate(transformed):
            A = _residualize_matrix(L @ add_z, Cstar, info_inv)
            x1ty = A.T @ yres
            s11 = np.sum(A * A, axis=0)
            df_a = n - rank_c - 1
            valid_a = eligible & (s11 > 0) & (df_a > 0)
            b_add_m = np.divide(x1ty, s11, out=np.full(m, np.nan), where=valid_a)
            rss_a = yty - b_add_m * x1ty
            se_add_m = np.sqrt(np.divide(rss_a, df_a * s11, out=np.full(m, np.nan), where=valid_a))
            stat_add_m = b_add_m / se_add_m
            p_add_m = 2.0 * _t.sf(np.abs(stat_add_m), df=df_a)
            metrics["beta_additive"][:, j] = b_add_m
            metrics["beta_additive_raw"][:, j] = b_add_m / add_sd
            metrics["stat_additive"][:, j] = stat_add_m
            metrics["neglog_p_additive"][:, j] = neglog10p(p_add_m)

            if model == "additive":
                metrics["genotype_filter_pass"][:, j] = valid_a
                metrics["dominance_class"][:, j] = np.where(valid_a, "NA", "FILTERED")
                metrics["inheritance_mode"][:, j] = np.where(valid_a, "NA", "FILTERED")
                continue

            D = _residualize_matrix(L @ dom_z, Cstar, info_inv)
            x2ty = D.T @ yres
            s22 = np.sum(D * D, axis=0)
            valid_d = eligible & (s22 > 0) & (df_a > 0)
            b_dom_m = np.divide(x2ty, s22, out=np.full(m, np.nan), where=valid_d)
            rss_d = yty - b_dom_m * x2ty
            se_dom_m = np.sqrt(np.divide(rss_d, df_a * s22, out=np.full(m, np.nan), where=valid_d))
            stat_dom_m = b_dom_m / se_dom_m
            metrics["stat_dominance_marginal"][:, j] = stat_dom_m
            metrics["neglog_p_dominance_marginal"][:, j] = neglog10p(
                2.0 * _t.sf(np.abs(stat_dom_m), df=df_a)
            )

            s12 = np.sum(A * D, axis=0)
            det = s11 * s22 - s12 * s12
            df_j = n - rank_c - 2
            valid = eligible & (det > np.finfo(float).eps) & (df_j > 0)
            metrics["genotype_filter_pass"][:, j] = valid
            inv11 = np.divide(s22, det, out=np.full(m, np.nan), where=valid)
            inv22 = np.divide(s11, det, out=np.full(m, np.nan), where=valid)
            inv12 = np.divide(-s12, det, out=np.full(m, np.nan), where=valid)
            b_add = inv11 * x1ty + inv12 * x2ty
            b_dom = inv12 * x1ty + inv22 * x2ty
            rss_ad = np.maximum(yty - b_add * x1ty - b_dom * x2ty, 0.0)
            sigma2 = np.divide(rss_ad, df_j, out=np.full(m, np.nan), where=valid)
            se_add = np.sqrt(sigma2 * inv11)
            se_dom = np.sqrt(sigma2 * inv22)
            stat_add = b_add / se_add
            stat_dom = b_dom / se_dom
            fstat = np.divide(np.maximum(rss_a - rss_ad, 0.0), sigma2,
                              out=np.full(m, np.nan), where=valid & (sigma2 > 0))
            a_raw = b_add / add_sd
            se_a_raw = se_add / add_sd
            d_raw = b_dom / dom_sd
            signed, magnitude, coarse, mode = classify_inheritance(
                a_raw, d_raw, se_a_raw, thresholds=da_thresholds, stability_z=stability_z
            )
            coarse[~valid] = "FILTERED"
            mode[~valid] = "FILTERED"
            metrics["neglog_p_add_joint"][:, j] = neglog10p(2.0 * _t.sf(np.abs(stat_add), df=df_j))
            metrics["neglog_p_dom_joint"][:, j] = neglog10p(2.0 * _t.sf(np.abs(stat_dom), df=df_j))
            metrics["neglog_p_avsad"][:, j] = neglog10p(_f.sf(fstat, 1, df_j))
            metrics["f_avsad"][:, j] = fstat
            metrics["beta_add_joint_raw"][:, j] = a_raw
            metrics["se_add_joint_raw"][:, j] = se_a_raw
            metrics["beta_dom_joint_raw"][:, j] = d_raw
            metrics["degree_of_dominance"][:, j] = signed
            metrics["degree_of_dominance_abs"][:, j] = magnitude
            metrics["dominance_class"][:, j] = coarse
            metrics["inheritance_mode"][:, j] = mode

        parts.append(_assemble(metrics, bim_s, trait_names))
    return pd.concat(parts, ignore_index=True)


def _shared_eigen_gls_context(
    phenotypes,
    fixed_design,
    U,
    s,
    variance_fits,
    residual_eigenvalue=0.0,
    tol=1e-10,
):
    """Precompute fixed-design GLS products for all traits at once.

    Parameters
    ----------
    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    fixed_design : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    U : array-like
        Numerical operand in ``U.T @ Y``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    s : array-like
        Numerical operand in ``yvar[None, :] * (h2[None, :] * s[:, None] + 1.0 - h2[None, :])``.
        Shape must conform to that matrix/vector operation and the enclosing prepared state.

    variance_fits : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-10``.
    """
    Y = np.asarray(phenotypes, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    C = np.asarray(fixed_design, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    U = np.asarray(U)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    n, n_traits = Y.shape
    if C.shape[0] != n or U.shape != (n, len(s)):
        raise ValueError("phenotype, design, and eigensystem dimensions do not agree")
    if len(variance_fits) != n_traits:
        raise ValueError("one variance-component fit is required per trait")

    h2 = np.asarray([fit["h2"] for fit in variance_fits], dtype=np.float64)
    yvar = np.asarray([fit["yvar"] for fit in variance_fits], dtype=np.float64)
    top = yvar[None, :] * (h2[None, :] * s[:, None] + 1.0 - h2[None, :])
    residual = yvar * (
        h2 * float(residual_eigenvalue) + 1.0 - h2
    )
    top = np.maximum(top, tol)
    residual = np.maximum(residual, tol)
    w_residual = 1.0 / residual
    delta = 1.0 / top - w_residual[None, :]

    UTY = U.T @ Y
    UTC = U.T @ C
    YTY = np.sum(Y * Y, axis=0)
    CTY = C.T @ Y
    CTC = C.T @ C
    ywy = w_residual * YTY + np.sum(delta * UTY * UTY, axis=0)
    cwy = CTY.T * w_residual[:, None]
    cwy += np.einsum("rp,rt,rt->tp", UTC, delta, UTY, optimize=True)
    cwc = CTC[None, :, :] * w_residual[:, None, None]
    cwc += np.einsum("rp,rt,rq->tpq", UTC, delta, UTC, optimize=True)
    information_inverse = np.linalg.pinv(cwc)
    fixed_beta = np.einsum("tpq,tq->tp", information_inverse, cwy, optimize=True)
    ypy = ywy - np.einsum("tp,tp->t", cwy, fixed_beta, optimize=True)
    return {
        "Y": Y,
        "C": C,
        "U": U,
        "s": s,
        "UTY": UTY,
        "UTC": UTC,
        "w_residual": w_residual,
        "delta": delta,
        "information_inverse": information_inverse,
        "fixed_beta": fixed_beta,
        "ypy": np.maximum(ypy, 0.0),
        "rank_c": int(np.linalg.matrix_rank(C)),
    }


def _marker_terms(X, UTX, raw_XTY, raw_XTC, raw_XTX, context, trait_slice):
    """Return X'P Y and X'P X for every marker and selected trait.

    Parameters
    ----------
    X : array-like
        Predictor matrix with observations in rows and features in columns. Its row order must
        match the response or operator to which it is applied; the selected fitting routine
        determines missing-value support.

    UTX : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    raw_XTY : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    raw_XTC : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    raw_XTX : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    context : float or str
        Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
        descriptive text identifying the operation in an allocation error.

    trait_slice : object (see the cited input contract)
        Input for the operation ``raw_XTY[:, trait_slice] * w[None, :]``. See
        ``vigor.assoc._marker_terms`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.
    """
    w = context["w_residual"][trait_slice]
    delta = context["delta"][:, trait_slice]
    xwy = raw_XTY[:, trait_slice] * w[None, :]
    xwy += UTX.T @ (delta * context["UTY"][:, trait_slice])
    xwc = raw_XTC[:, None, :] * w[None, :, None]
    xwc += np.einsum(
        "rm,rt,rp->mtp", UTX, delta, context["UTC"], optimize=True
    )
    xwx = raw_XTX[:, None] * w[None, :] + (UTX * UTX).T @ delta
    information_inverse = context["information_inverse"][trait_slice]
    fixed_beta = context["fixed_beta"][trait_slice]
    xpy = xwy - np.einsum("mtp,tp->mt", xwc, fixed_beta, optimize=True)
    xpx = xwx - np.einsum(
        "mtp,tpq,mtq->mt", xwc, information_inverse, xwc, optimize=True
    )
    return xpy, xpx, xwc


def _cross_marker_terms(
    UTA,
    UTD,
    raw_ATD,
    AWC,
    DWC,
    context,
    trait_slice,
):
    """Compute additive/dominance cross-products in the whitened fixed-effect metric.

    Parameters
    ----------
    UTA : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    UTD : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    raw_ATD : array-like
        Precomputed projection or cross-product term for the association design. Use the
        eigenbasis, trait tile and genotype coding from the same null/scan preparation; see
        _shared_eigen_gls_context and the calling regression helper.

    AWC : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    DWC : array-like
        Dual/whitened matrix prepared by the corresponding null or projection routine. Animal
        and trait/design axes must match that prepared state; it is not raw phenotype data.

    context : float or str
        Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
        descriptive text identifying the operation in an allocation error.

    trait_slice : array-like or numeric scalar
        Input for the operation ``awd - np.einsum('mtp,tpq,mtq->mt', AWC,
        context['information_inverse'][trait_slice], DWC, optimize=True)``. See
        ``vigor.assoc._cross_marker_terms`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.
    """
    w = context["w_residual"][trait_slice]
    delta = context["delta"][:, trait_slice]
    awd = raw_ATD[:, None] * w[None, :] + (UTA * UTD).T @ delta
    return awd - np.einsum(
        "mtp,tpq,mtq->mt",
        AWC,
        context["information_inverse"][trait_slice],
        DWC,
        optimize=True,
    )


def _empty_eigen_metrics(m, n_traits, observed, counts, maf, eligible, model):
    """Construct empty eigen metrics.

    Parameters
    ----------
    m : int
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension.

    observed : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    counts : array-like or int
        Weights or multiplicities used in this accumulation/resampling step. Align vector
        entries with the indexed sample/variant axis; scalar counts specify the number of
        consecutive items.

    maf : object (see the cited input contract)
        Input for the operation ``maf[:, None]``. See ``vigor.assoc._empty_eigen_metrics`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape.

    eligible : bool or numeric mask array
        Observation/eligibility mask aligned to the corresponding input axis. True/one selects
        observed or eligible entries; false/zero excludes them. A sample mask is
        one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
        axis.

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Source-recognized literal values: ``add-dom``. These select the corresponding
        branches below; comparison evidence alone does not establish an exhaustive enum.
    """
    numeric = [
        "neglog_p_additive",
        "stat_additive",
        "beta_additive",
        "beta_additive_raw",
        "se_additive_raw",
    ]
    if model == "add-dom":
        numeric.extend(
            [
                "neglog_p_dominance_marginal",
                "stat_dominance_marginal",
                "beta_dominance_marginal_raw",
                "se_dominance_marginal_raw",
                "neglog_p_add_joint",
                "stat_add_joint",
                "neglog_p_dom_joint",
                "stat_dom_joint",
                "neglog_p_avsad",
                "f_avsad",
                "beta_add_joint_raw",
                "se_add_joint_raw",
                "beta_dom_joint_raw",
                "se_dom_joint_raw",
                "cov_add_dom_joint_raw",
                "degree_of_dominance",
                "degree_of_dominance_abs",
            ]
        )
    metrics = {name: np.full((m, n_traits), np.nan) for name in numeric}
    metrics.update(
        {
            "n_obs": np.repeat(observed[:, None], n_traits, axis=1),
            "count_AA": np.repeat(counts[0][:, None], n_traits, axis=1),
            "count_AB": np.repeat(counts[1][:, None], n_traits, axis=1),
            "count_BB": np.repeat(counts[2][:, None], n_traits, axis=1),
            "maf": np.repeat(maf[:, None], n_traits, axis=1),
            "genotype_filter_pass": np.repeat(eligible[:, None], n_traits, axis=1),
            "dominance_class": np.full((m, n_traits), "FILTERED", dtype=object),
            "inheritance_mode": np.full((m, n_traits), "FILTERED", dtype=object),
        }
    )
    return metrics


def iter_chromosome_gls_eigen(
    reader,
    chrom,
    phenotypes,
    trait_names,
    sample_index,
    U,
    s,
    variance_fits,
    fixed_design,
    block_size='auto',
    trait_tile_size=None,
    model="add-dom",
    da_thresholds=(0.25, 0.75, 1.25),
    min_genotype_count=0,
    stability_z=1.0,
    residual_eigenvalue=0.0,
    compute_dtype="float32",
    combine_trait_tiles=False,
    backend="cpu",
    gpu_devices=(),
    gpu_memory_budget_mb=None,
    maxmem=None,
    memory_fraction=None,
):
    """Yield bounded-memory GLS result blocks using one shared eigenbasis.

    Genotypes are projected into ``U`` once per block.  Trait-specific inverse
    covariance weights and fixed-effect projections are then applied in tiles,
    avoiding both dense whitening matrices and Python loops over traits.

    Parameters
    ----------
    reader : PLINK reader or genotype source
        Opened genotype reader supplying sample metadata, variant metadata and block access. Its
        sample order defines the relationship matrix order.

    chrom : str or int or None
        Chromosome identifier matched against source metadata as a string. None leaves the
        chromosome unrestricted where supported.

    phenotypes : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    trait_names : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    sample_index : one-dimensional integer array or None
        Zero-based sample positions to retain, in the requested output order. Values must be
        valid positions in the source; None selects the source's full sample axis. Passed to
        ``reader.iter_blocks`` as ``sample_index``.

    U : array-like
        Numerical operand in ``_shared_eigen_gls_context(phenotypes, fixed_design, U, s,
        variance_fits, residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    s : array-like
        Numerical operand in ``_shared_eigen_gls_context(phenotypes, fixed_design, U, s,
        variance_fits, residual_eigenvalue=residual_eigenvalue)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    variance_fits : mapping or sequence
        Trait/component variance estimates used to construct the association or BLUP covariance.
        Match the prepared sample/trait order and include the component fields expected by the
        null-model builder.

    fixed_design : array-like or pandas.DataFrame or None
        Fixed-effect design, animals × covariate columns. Rows must match the phenotype/kernel
        animal order. None uses the routine's intercept-only or no-additional-covariate path;
        rank-deficient designs may be rejected or reduced.

    block_size : int
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings).

    trait_tile_size : int
        Rows/traits per processing or storage tile, along the axis named by the routine.
        Positive counts bound the corresponding intermediate allocation. Default: ``None``. None
        delegates to the local fallback shown in this body; it is not an additional
        numeric/category value.

    model : str or fitted estimator
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Default: ``'add-dom'``. Source-recognized literal values: ``additive``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    da_thresholds : float or mapping
        Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
        supplies named dominance/inheritance thresholds; see the classifier defaults and
        validation for their domain. Default: ``(0.25, 0.75, 1.25)``. Passed to
        ``classify_inheritance`` as ``thresholds``.

    min_genotype_count : int
        Minimum observed calls supporting a variant test; nonnegative counts filter low-support
        variants before reporting effects. Default: ``0``. Source comparisons:
        ``min_genotype_count > 0``. See the rejection/fallback branches for enforcement.

    stability_z : float
        Minimum standardized evidence statistic for promotion or stable effect classification.
        Positive thresholds require stronger support relative to the estimated uncertainty.
        Default: ``1.0``. Passed to ``classify_inheritance`` as ``stability_z``.

    residual_eigenvalue : float
        Eigenvalue assigned to directions outside the retained basis. A nonnegative value
        supplies the bulk covariance contribution; zero discards that omitted-space
        contribution. Default: ``0.0``. Passed to ``_shared_eigen_gls_context`` as
        ``residual_eigenvalue``.

    compute_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``'float32'``.

    combine_trait_tiles : bool
        Context/control value read by the enclosing operation; its contract follows the explicit
        local use and the caller. Default: ``False``.

    backend : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Default: ``'cpu'``.

    gpu_devices : sequence of int
        CUDA device indices assigned to the projection worker pool. Indices must refer to
        available devices; an empty tuple uses the backend's default device selection. Default:
        ``()``. Passed to ``create_projection_backend`` as ``gpu_devices``.

    gpu_memory_budget_mb : float or None
        Memory budget in MiB for the named CPU/GPU resource guard. Positive values bound
        allocations; None uses available memory or the resolved execution plan. Default:
        ``None``. Passed to ``create_projection_backend`` as ``gpu_memory_budget_mb``.

    maxmem : int, float, unit string or None
        Working-memory budget for sizing ``block_size='auto'``; see
        ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    if model not in {"add-dom", "additive"}:
        raise ValueError("model must be 'add-dom' or 'additive'")
    dtype = np.dtype(compute_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("compute_dtype must be float32 or float64")
    names = list(trait_names)
    context = _shared_eigen_gls_context(
        phenotypes,
        fixed_design,
        U,
        s,
        variance_fits,
        residual_eigenvalue=residual_eigenvalue,
    )
    n, n_traits = context["Y"].shape
    if len(names) != n_traits:
        raise ValueError("trait_names does not match phenotype columns")
    tile_size = n_traits if trait_tile_size is None else min(int(trait_tile_size), n_traits)
    if tile_size < 1:
        raise ValueError("trait_tile_size must be positive")
    sample_index = np.asarray(sample_index)
    variant_index = reader.chrom_variant_index(chrom)
    projector = create_projection_backend(
        backend,
        context["U"],
        context["Y"],
        context["C"],
        gpu_devices=gpu_devices,
        gpu_memory_budget_mb=gpu_memory_budget_mb,
    )

    # per call: genotype, add_z, dominance, dom_z, UTA, UTD, two products in dtype + bool masks;
    # per variant: ~40 float64 metric/frame cells for every trait held at once
    block_size = resolve_variant_chunk(
        block_size, n, bytes_per_call=8 * dtype.itemsize + 2,
        bytes_per_variant=320 * (n_traits if combine_trait_tiles else tile_size),
        maxmem=maxmem, memory_fraction=memory_fraction, n_variants=len(variant_index))
    for bim_slice, genotype in reader.iter_blocks(
        block_size, variant_index=variant_index, sample_index=sample_index
    ):
        genotype = genotype.astype(dtype, copy=False)
        add_z, add_sd, add_mask = standardize_block(genotype, dtype=dtype)
        observed = add_mask.sum(axis=0, dtype=np.float64)
        count_aa = np.sum(genotype == 0, axis=0).astype(float)
        count_ab = np.sum(genotype == 1, axis=0).astype(float)
        count_bb = np.sum(genotype == 2, axis=0).astype(float)
        allele_frequency = np.nansum(genotype, axis=0, dtype=np.float64) / np.maximum(
            2.0 * observed, 1.0
        )
        maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
        eligible = np.isfinite(add_sd)
        if min_genotype_count > 0:
            eligible &= np.minimum(np.minimum(count_aa, count_ab), count_bb) >= min_genotype_count

        UTA, raw_ATY, raw_ATC = projector.project(add_z)
        raw_ATA = np.sum(add_z * add_z, axis=0, dtype=np.float64)
        if model == "add-dom":
            dominance = dominance_encode(genotype, dtype=dtype)
            dom_z, dom_sd, _ = standardize_block(dominance, dtype=dtype)
            eligible &= np.isfinite(dom_sd)
            UTD, raw_DTY, raw_DTC = projector.project(dom_z)
            raw_DTD = np.sum(dom_z * dom_z, axis=0, dtype=np.float64)
            raw_ATD = np.sum(add_z * dom_z, axis=0, dtype=np.float64)

        block_frames = []
        m = genotype.shape[1]
        for start in range(0, n_traits, tile_size):
            stop = min(start + tile_size, n_traits)
            trait_slice = slice(start, stop)
            tile_traits = names[start:stop]
            q = stop - start
            metrics = _empty_eigen_metrics(
                m,
                q,
                observed,
                (count_aa, count_ab, count_bb),
                maf,
                eligible,
                model,
            )
            x1ty, s11, AWC = _marker_terms(
                add_z,
                UTA,
                raw_ATY,
                raw_ATC,
                raw_ATA,
                context,
                trait_slice,
            )
            ypy = context["ypy"][trait_slice][None, :]
            df_additive = n - context["rank_c"] - 1
            valid_additive = eligible[:, None] & (s11 > np.finfo(float).eps) & (df_additive > 0)
            beta_additive = np.divide(
                x1ty, s11, out=np.full_like(x1ty, np.nan), where=valid_additive
            )
            rss_additive = np.maximum(ypy - beta_additive * x1ty, 0.0)
            se_additive = np.sqrt(
                np.divide(
                    rss_additive,
                    df_additive * s11,
                    out=np.full_like(x1ty, np.nan),
                    where=valid_additive,
                )
            )
            stat_additive = beta_additive / se_additive
            metrics["beta_additive"] = beta_additive
            metrics["beta_additive_raw"] = beta_additive / add_sd[:, None]
            metrics["se_additive_raw"] = se_additive / add_sd[:, None]
            metrics["stat_additive"] = stat_additive
            metrics["neglog_p_additive"] = neglog10p(
                2.0 * _t.sf(np.abs(stat_additive), df=df_additive)
            )

            if model == "additive":
                metrics["genotype_filter_pass"] = valid_additive
                metrics["dominance_class"] = np.where(valid_additive, "NA", "FILTERED")
                metrics["inheritance_mode"] = np.where(valid_additive, "NA", "FILTERED")
                frame = _assemble(metrics, bim_slice, tile_traits)
                if combine_trait_tiles:
                    block_frames.append(frame)
                else:
                    yield frame
                continue

            x2ty, s22, DWC = _marker_terms(
                dom_z,
                UTD,
                raw_DTY,
                raw_DTC,
                raw_DTD,
                context,
                trait_slice,
            )
            valid_dominance = eligible[:, None] & (s22 > np.finfo(float).eps) & (df_additive > 0)
            beta_dom_marginal = np.divide(
                x2ty, s22, out=np.full_like(x2ty, np.nan), where=valid_dominance
            )
            rss_dominance = np.maximum(ypy - beta_dom_marginal * x2ty, 0.0)
            se_dom_marginal = np.sqrt(
                np.divide(
                    rss_dominance,
                    df_additive * s22,
                    out=np.full_like(x2ty, np.nan),
                    where=valid_dominance,
                )
            )
            stat_dom_marginal = beta_dom_marginal / se_dom_marginal
            metrics["beta_dominance_marginal_raw"] = beta_dom_marginal / dom_sd[:, None]
            metrics["se_dominance_marginal_raw"] = se_dom_marginal / dom_sd[:, None]
            metrics["stat_dominance_marginal"] = stat_dom_marginal
            metrics["neglog_p_dominance_marginal"] = neglog10p(
                2.0 * _t.sf(np.abs(stat_dom_marginal), df=df_additive)
            )

            s12 = _cross_marker_terms(
                UTA, UTD, raw_ATD, AWC, DWC, context, trait_slice
            )
            determinant = s11 * s22 - s12 * s12
            df_joint = n - context["rank_c"] - 2
            valid_joint = (
                eligible[:, None]
                & (determinant > np.finfo(float).eps)
                & (df_joint > 0)
            )
            inv11 = np.divide(
                s22,
                determinant,
                out=np.full_like(determinant, np.nan),
                where=valid_joint,
            )
            inv22 = np.divide(
                s11,
                determinant,
                out=np.full_like(determinant, np.nan),
                where=valid_joint,
            )
            inv12 = np.divide(
                -s12,
                determinant,
                out=np.full_like(determinant, np.nan),
                where=valid_joint,
            )
            beta_add_joint = inv11 * x1ty + inv12 * x2ty
            beta_dom_joint = inv12 * x1ty + inv22 * x2ty
            rss_joint = np.maximum(
                ypy - beta_add_joint * x1ty - beta_dom_joint * x2ty, 0.0
            )
            sigma2 = np.divide(
                rss_joint,
                df_joint,
                out=np.full_like(rss_joint, np.nan),
                where=valid_joint,
            )
            se_add_joint = np.sqrt(sigma2 * inv11)
            se_dom_joint = np.sqrt(sigma2 * inv22)
            stat_add_joint = beta_add_joint / se_add_joint
            stat_dom_joint = beta_dom_joint / se_dom_joint
            # Adding one dominance coefficient makes the nested-model F test
            # algebraically identical to the squared conditional t statistic.
            # This form avoids cancellation between nearly equal residual sums.
            fstat = stat_dom_joint * stat_dom_joint
            add_raw = beta_add_joint / add_sd[:, None]
            add_se_raw = se_add_joint / add_sd[:, None]
            dom_raw = beta_dom_joint / dom_sd[:, None]
            dom_se_raw = se_dom_joint / dom_sd[:, None]
            covariance_raw = sigma2 * inv12 / (add_sd[:, None] * dom_sd[:, None])
            signed, magnitude, coarse, mode = classify_inheritance(
                add_raw,
                dom_raw,
                add_se_raw,
                thresholds=da_thresholds,
                stability_z=stability_z,
            )
            coarse[~valid_joint] = "FILTERED"
            mode[~valid_joint] = "FILTERED"
            metrics["genotype_filter_pass"] = valid_joint
            metrics["neglog_p_add_joint"] = neglog10p(
                2.0 * _t.sf(np.abs(stat_add_joint), df=df_joint)
            )
            metrics["stat_add_joint"] = stat_add_joint
            dominance_joint_score = neglog10p(
                2.0 * _t.sf(np.abs(stat_dom_joint), df=df_joint)
            )
            metrics["neglog_p_dom_joint"] = dominance_joint_score
            metrics["stat_dom_joint"] = stat_dom_joint
            metrics["neglog_p_avsad"] = dominance_joint_score
            metrics["f_avsad"] = fstat
            metrics["beta_add_joint_raw"] = add_raw
            metrics["se_add_joint_raw"] = add_se_raw
            metrics["beta_dom_joint_raw"] = dom_raw
            metrics["se_dom_joint_raw"] = dom_se_raw
            metrics["cov_add_dom_joint_raw"] = covariance_raw
            metrics["degree_of_dominance"] = signed
            metrics["degree_of_dominance_abs"] = magnitude
            metrics["dominance_class"] = coarse
            metrics["inheritance_mode"] = mode
            frame = _assemble(metrics, bim_slice, tile_traits)
            if combine_trait_tiles:
                block_frames.append(frame)
            else:
                yield frame

        if combine_trait_tiles and block_frames:
            combined = pd.concat(block_frames, ignore_index=True)
            trait_order = {name: index for index, name in enumerate(names)}
            combined["_trait_order"] = combined["trait"].map(trait_order)
            combined = combined.sort_values(
                ["variant_index", "_trait_order"], kind="stable"
            ).drop(columns="_trait_order")
            yield combined.reset_index(drop=True)


def scan_chromosome_gls_eigen(*args, **kwargs):
    """Materializing wrapper around :func:`iter_chromosome_gls_eigen`.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments forwarded, in order, to ``iter_chromosome_gls_eigen``.
        See that signature for required scientific inputs and their order.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``iter_chromosome_gls_eigen``. See those
        callables for accepted keys and values. Passed to ``iter_chromosome_gls_eigen`` as
        ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    frames = list(iter_chromosome_gls_eigen(*args, **kwargs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
