> Applies to VIGOR 2.12.1. For the current association-scan fields and defaults see [INPUT_OUTPUT.md](INPUT_OUTPUT.md).

# Results as labeled xarray Datasets

Terminal VIGOR functions return an `xarray.Dataset`, not a dictionary. The
layout follows `npplink.mvGWAS_xarray`:

| Part | Holds | Access |
|---|---|---|
| Data variables | every array output (covariances, BLUPs, predictions, kernels, statistics) on labeled dimensions | `ds["G"]`, `ds["G"].to_pandas()` |
| Variable attrs | `long_name`, `units`, `description` | `ds["h2"].attrs` |
| Dataset attrs | scalars, diagnostics, nested specifications, fitted objects, `parameters`, provenance (`title`, `software`, `vigor_version`, `created`) | `ds.attrs["parameters"]` |

The key names did not change. An output that used to be `res["G"]` is still
`res["G"]` (now a `DataArray`; call `.to_pandas()` for a DataFrame). A
non-array output that used to be `res["diagnostics"]` is now
`res.attrs["diagnostics"]`.

## Dimension names

| Dimension | Meaning |
|---|---|
| `trait_i`, `trait_j` | trait-by-trait matrices (G, E, rG, SEs) |
| `sample`, `trait` | animal-by-trait tables (BLUPs, predictions, phenotypes) |
| `sample_i`, `sample_j` | genomic relationship matrices |
| `snp` (`snp_i`, `snp_j`) | variant results and pairwise LD |
| `component`, `factor`, `expert`, `fold`, `replicate` | kernels, latent factors, experts, CV folds, bootstrap draws |

## Tables and stacked matrices

- **Long tables:** edge lists, rankings, orientations and metric tables are stored one column per variable along a row dimension (`edge`, `pair`, `snp`, `record`, ...). Rebuild one with `vigor.result_table(ds, "edges")`.
- **Row labels kept as coordinates:** a table such as `organize_scores`' `g_corr_table` keeps trait1/trait2 as coordinates on `trait_pair`. `vigor.as_frame(ds["g_corr_table"])` returns the pandas table with those labels as its index.
- **Stacked matrices:** a dictionary of matrices becomes one variable with a leading label dimension. Select one with `.sel()`: `sim["grms"].sel(kernel="additive")`, `fit["G_components"].sel(component="additive")`.
- **Passing results on:** VIGOR functions accept these DataArrays (and a whole covariance-fit Dataset as `G` with `E=None`) wherever they used to take DataFrames.
- **Simulations:** `simulate_genomic_dataset` returns a Dataset. Write its genotypes to PLINK with `vigor.write_simulation_plink(sim, prefix)`.

## Saving

Every terminal function takes `save_path=None`. When it is set, the result is
written as Zarr v3 with sharded chunks (about 8 MiB chunks, 32 MiB shards) and
consolidated metadata. Dataset and variable attrs go in the store as JSON, and
the call returns the store reopened lazily:

```python
import vigor
fit = vigor.genetic_varcov_PSD(K, Y, save_path="results/covariance.zarr")
fit = vigor.open_result("results/covariance.zarr")   # later, lazily
```

For results you build yourself, or to choose the layout:

```python
from vigor import result_dataset, save_result
ds = result_dataset({"beta": beta_df, "n": 512}, dims={"beta": ("snp", "trait")},
                    var_attrs={"beta": {"long_name": "effect size"}},
                    title="custom scan", software="my_pipeline")
save_result(ds, "results/scan.zarr", chunks={"snp": 25_000}, shards={"snp": 500_000})
# block-wise writers: first block mode="w", later blocks mode="a", append_dim="snp"
```

Fitted estimators and other Python objects stay usable in memory through
`ds.attrs`. In a saved store they are recorded only as their text
representation, so refit or pickle them separately if you need them back.
