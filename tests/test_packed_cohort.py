"""simulate_packed_cohort / genotype_panel: packed store, reuse, and lazy == dense simulation."""
import numpy as np
import pandas as pd
import pytest

import vigor
from vigor.simulate import simulate_traits_from_genotypes

TRAITS = dict(n_additive=6, n_dominance=3, n_epistatic_pairs=2, n_gxe=2,
              n_hidden_covariates=2, compute_grms=True)


def _cohort(path, **kw):
    args = dict(n_samples=60, n_variants=96, n_chromosomes=2, n_traits=2, seed=7,
                genotype_kwargs={"generations": 20}, trait_kwargs=TRAITS, variant_chunk=16)
    args.update(kw)
    return vigor.simulate_packed_cohort(str(path), **args)


def test_store_is_packed_and_result_has_no_dense_genotypes(tmp_path):
    path = tmp_path / "geno.zarr"
    sim = _cohort(path)
    assert "genotypes" not in sim and "phenotypes" in sim and "grms" in sim
    assert sim.attrs["genotype_store"] == str(path)
    assert not sim.attrs["metadata"]["genotype_store_reused"]
    import zarr
    root = zarr.open_group(str(path), mode="r")
    assert "GT_packed2" in root["genotype"]
    vi = vigor.result_table(sim, "variant_info")
    assert len(vi) == 96 and set(vi.chrom.astype(str)) == {"1", "2"}
    with pytest.raises(ValueError, match="packed Zarr store"):
        vigor.write_simulation_plink(sim, str(tmp_path / "plink"))


def test_store_reuse_is_deterministic_and_mismatch_is_rejected(tmp_path):
    path = tmp_path / "geno.zarr"
    first = _cohort(path)
    again = _cohort(path)
    assert again.attrs["metadata"]["genotype_store_reused"]
    np.testing.assert_array_equal(first.phenotypes.values, again.phenotypes.values)
    with pytest.raises(FileExistsError):
        _cohort(path, n_samples=61)
    rewritten = _cohort(path, n_samples=61, overwrite=True)
    assert rewritten.sizes["sample"] == 61


def test_lazy_store_simulation_matches_dense_input(tmp_path):
    path = tmp_path / "geno.zarr"
    sim = _cohort(path)
    dense = vigor.genotype_panel(str(path))
    assert dense.shape == (60, 96) and dense.to_numpy().dtype == np.float32
    tseed = int(np.random.SeedSequence(7).spawn(2)[1].generate_state(1)[0])
    ref = simulate_traits_from_genotypes(dense, n_traits=2, seed=tseed, **TRAITS)
    np.testing.assert_allclose(sim.phenotypes.values, ref.phenotypes.to_numpy(),
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sim.grms.sel(kernel="additive").values,
                               ref.grms["additive"].to_numpy(), rtol=1e-5, atol=1e-6)


def test_genotype_panel_windows(tmp_path):
    path = tmp_path / "geno.zarr"
    _cohort(path)
    full = vigor.genotype_panel(str(path))
    chrom2 = vigor.genotype_panel(str(path), chromosome="2")
    assert chrom2.shape[1] == 48 and all(c.startswith("2:") for c in chrom2.columns)
    pd.testing.assert_frame_equal(chrom2, full[chrom2.columns])
    window = vigor.genotype_panel(str(path), chromosome="1", variants=slice(5, 15))
    pd.testing.assert_frame_equal(window, full.iloc[:, 5:15])
    capped = vigor.genotype_panel(str(path), max_variants=10)
    assert capped.shape[1] == 10
    ids = list(full.index[::3])
    sub = vigor.genotype_panel(str(path), variants=list(full.columns[[3, 50]]), sample_ids=ids)
    pd.testing.assert_frame_equal(sub, full.loc[ids, full.columns[[3, 50]]])
