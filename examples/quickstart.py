"""Generate a small synthetic PLINK dataset and run VIGOR end to end."""
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vigor
assert tuple(map(int, vigor.__version__.split('.')[:3])) >= (2, 12, 1), vigor.__version__
from vigor.testutils import simulate


def main() -> None:
    output = Path("example_output")
    output.mkdir(exist_ok=True)
    prefix = output / "synthetic"

    simulated = simulate(
        str(prefix),
        n_fam=20,
        sibs=4,
        n_chrom=2,
        snps_per_chrom=200,
        n_causal_add=10,
        n_causal_dom=10,
        seed=2026,
    )
    phenotype = simulated["pheno"][["y_mix"]]
    covariates = phenotype.copy()
    covariates["sex"] = np.where(np.arange(len(phenotype)) % 2, "F", "M")
    covariates["batch"] = np.where(np.arange(len(phenotype)) % 3, "B1", "B2")
    covariates = covariates[["sex", "batch"]]
    analysis = phenotype.join(covariates)
    analysis.index.name = "iid"
    analysis.to_csv(output / "phenotypes.tsv", sep="\t")

    results = vigor.mvGWAS(
        str(prefix),
        phenotype,
        traits=["y_mix"],
        covar=covariates,
        out=str(output / "scan"),
        verbose=True,
    )
    dom = [str(t) for t in results.trait.values if str(t).startswith("dominance:")]
    top = results.conditional_neglog_p.sel(trait=dom).max("trait", skipna=True).argmax("variant")
    print(f"VIGOR {vigor.__version__}: tested {results.sizes['variant']} variants")
    print("top conditional-dominance marker:",results.variant_id.isel(variant=top).compute().item())
    print(f"results: {(output / 'scan.zarr').resolve()}")


if __name__ == "__main__":
    main()
