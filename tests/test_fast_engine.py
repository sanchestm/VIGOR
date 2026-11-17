"""Fast (calibrated information) score engine vs the exact engine."""
import numpy as np
import pandas as pd
import pytest

import vigor
from vigor.gwas.exact import prepare_score_null, score_block, score_design_block
from vigor.gwas.fast import calibrate_fast_information
from vigor.testutils import simulate

MED = 0.45493642


def _null_fixture(missing=True):
    rng = np.random.default_rng(4242)
    n, t, m = 300, 3, 200
    ids = pd.Index([f"a{i}" for i in range(n)])
    founders = rng.binomial(2, rng.uniform(.1, .5, 400), size=(n, 400)).astype(float)
    Z = founders - founders.mean(0)
    K = pd.DataFrame(Z @ Z.T / 400, index=ids, columns=ids)
    X = pd.DataFrame(rng.binomial(2, rng.uniform(.1, .5, m), size=(n, m)).astype(np.float32),
                     index=ids, columns=[f"v{i}" for i in range(m)])
    L = np.linalg.cholesky(K.to_numpy() + 1e-6 * np.eye(n))
    G = np.array([[.4, .2, .1], [.2, .3, .05], [.1, .05, .5]])
    E = np.array([[.6, .1, 0], [.1, .7, .1], [0, .1, .5]])
    Y = (L @ rng.normal(size=(n, t)) @ np.linalg.cholesky(G).T
         + rng.normal(size=(n, t)) @ np.linalg.cholesky(E).T)
    Y = pd.DataFrame(Y, index=ids, columns=["a", "b", "c"])
    if missing:
        Y.iloc[::5, 1] = np.nan
        Y.iloc[1::7, 2] = np.nan
    names = pd.Index(Y.columns)
    null = prepare_score_null(Y, K, pd.DataFrame(G, index=names, columns=names),
                              pd.DataFrame(E, index=names, columns=names),
                              work_dtype="float64", solver_tol=1e-9)
    return null, X


@pytest.mark.parametrize("missing", [True, False])
def test_fast_scores_are_exact_and_statistics_close(missing):
    null, X = _null_fixture(missing)
    exact = score_block(null, X)
    calibrate_fast_information(null, X.iloc[:, ::6].to_numpy())
    fast = score_block(null, X, engine="fast")
    assert set(fast.data_vars) == set(exact.data_vars)
    np.testing.assert_allclose(fast.nobs, exact.nobs)
    ratio = (fast.stat / exact.stat).values
    ratio = ratio[np.isfinite(ratio) & (exact.stat.values > 1e-6)]
    assert np.median(np.abs(np.log(ratio))) < .1
    orat = (fast.omnibus_stat / exact.omnibus_stat).values
    assert abs(np.median(orat) - 1) < .1
    assert np.corrcoef(fast.omnibus_neglog_p, exact.omnibus_neglog_p)[0, 1] > .98
    assert fast.attrs["score_engine_mode"] == "fast"


def test_fast_add_dom_conditional_identity_and_schema():
    null, X = _null_fixture(True)
    d = X.to_numpy(np.float32)
    designs = np.stack([d, (d == 1).astype(np.float32)], axis=2)
    contrasts = ["additive", "dominance"]
    exact = score_design_block(null, designs, contrast_names=contrasts)
    fast = score_design_block(null, designs, contrast_names=contrasts, engine="fast")
    assert list(fast.trait.values) == list(exact.trait.values)
    c = fast.sel(trait="dominance:a")
    np.testing.assert_allclose(c.conditional_stat, (c.joint_beta / c.joint_beta_se) ** 2,
                               rtol=1e-10, equal_nan=True)
    r = np.corrcoef(np.nan_to_num(fast.omnibus_neglog_p), np.nan_to_num(exact.omnibus_neglog_p))
    assert r[0, 1] > .97


def test_fast_basis_is_rejected():
    null, X = _null_fixture(False)
    with pytest.raises(ValueError, match="identity trait basis"):
        score_block(null, X, basis=np.eye(3), engine="fast")


def test_fast_is_public_default_and_null_calibrated(tmp_path):
    prefix = str(tmp_path / "sim")
    meta = simulate(prefix, n_fam=120, sibs=4, n_chrom=2, snps_per_chrom=400, seed=3)
    res = vigor.mvGWAS(prefix, meta["pheno"], model="add-dom", trait_mode="independent",
                       verbose=False)
    assert res.attrs["vigor_score_core"] == "observed_space_fast_calibrated"
    assert set(res.attrs["scan_group_metadata"]) == {"1", "2"}
    causal = np.r_[meta["causal_add"], meta["causal_dom"], meta["big_dom_snp"]]
    null = np.setdiff1d(np.arange(meta["m"]), causal)
    names = {f"rs{i}" for i in null}
    for contrast in ["additive", "dominance"]:
        d = res.sel(trait=f"{contrast}:y_null")
        keep = np.isin(d.variant_id.values.astype(str), list(names))
        lam = np.nanmedian(d.conditional_stat.values[keep]) / MED
        assert .8 < lam < 1.25, f"{contrast} lambda={lam}"
    with pytest.raises(ValueError, match="conflicts"):
        vigor.mvGWAS(prefix, meta["pheno"], engine="exact", score_engine="fast")


def test_fast_drops_coordinates_monomorphic_within_a_trait():
    # Trait "b" is missing on rows 0, 5, 10, ...; this variant varies only on those rows,
    # so it carries no information for "b". Fast must drop that coordinate like exact.
    null, X = _null_fixture(True)
    calibrate_fast_information(null, X.iloc[:, ::6].to_numpy())
    rng = np.random.default_rng(7)
    x = np.zeros(len(X), np.float32)
    x[::5] = rng.binomial(2, .4, size=len(x[::5]))
    G2 = pd.DataFrame({"mono_b": x, "v0": X.iloc[:, 0].to_numpy()}, index=X.index)
    exact = score_block(null, G2)
    fast = score_block(null, G2, engine="fast")
    np.testing.assert_array_equal(fast.omnibus_dof.values, exact.omnibus_dof.values)
    assert np.isnan(fast.sel(trait="b").z.values[0])
    assert np.isfinite(fast.omnibus_stat.values).all()


def test_fast_add_dom_without_heterozygotes_still_calibrates():
    null, X = _null_fixture(True)
    hom = (X.to_numpy(np.float32) >= 1).astype(np.float32) * 2  # 0/2 only: dominance constant
    designs = np.stack([hom, np.zeros_like(hom)], axis=2)
    from vigor.gwas.fast import calibrate_fast_information as cal
    fast_info = cal(null, designs[:, ::6], attach=True)
    assert not fast_info.exact_fallback and np.isfinite(fast_info.gamma[0]).all()
    contrasts = ["additive", "dominance"]
    exact = score_design_block(null, designs, contrast_names=contrasts)
    fast = score_design_block(null, designs, contrast_names=contrasts, engine="fast")
    np.testing.assert_array_equal(fast.omnibus_dof.values, exact.omnibus_dof.values)
    r = np.corrcoef(fast.omnibus_neglog_p, exact.omnibus_neglog_p)
    assert r[0, 1] > .97
