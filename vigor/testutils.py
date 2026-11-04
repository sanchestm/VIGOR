"""Utilities for tests and demos: write PLINK1 filesets and simulate data
with realistic relatedness plus known additive/dominance causal variants.
"""
import numpy as np
import pandas as pd


def write_plink(prefix, geno, iids, chroms, positions, a0="A", a1="G"):
    """Write a PLINK1 fileset. ``geno`` is (n_samples, n_variants) in {0,1,2,NaN}.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    geno : array-like
        Sample × variant numerical coding used in genotype moments. Raw inputs use biallelic
        dosage; centered/additive/dominance inputs must already follow the coding named by this
        operation. Missing entries are represented by the accompanying mask or NaN policy.
        Source comparisons: ``geno == 0``; ``geno == 1``; ``geno == 2``. See the
        rejection/fallback branches for enforcement.

    iids : object (see the cited input contract)
        Input for the operation ``pd.DataFrame({'fid': iids, 'iid': iids, 'father': 0, 'mother':
        0, 'sex': 0, 'pheno': -9})``. See ``vigor.testutils.write_plink`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.

    chroms : sequence or None
        Chromosome identifiers to process, matched to source metadata. None uses all chromosomes
        available to the routine.

    positions : one-dimensional array-like
        Physical positions in base pairs for each variant, aligned to genotype columns. Sorted
        genomic order is required for searchsorted/banded interval algorithms.

    a0 : array-like or numeric scalar
        Input for the operation ``pd.DataFrame({'chrom': np.asarray(chroms, dtype=str), 'snp':
        [f'rs{i}' for i in range(m)], 'cm': 0, 'pos': np.asarray(positions, dtype=int), 'a0':
        a0, 'a1': a1})``. See ``vigor.testutils.write_plink`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'A'``.

    a1 : array-like or numeric scalar
        Input for the operation ``pd.DataFrame({'chrom': np.asarray(chroms, dtype=str), 'snp':
        [f'rs{i}' for i in range(m)], 'cm': 0, 'pos': np.asarray(positions, dtype=int), 'a0':
        a0, 'a1': a1})``. See ``vigor.testutils.write_plink`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``'G'``.
    """
    geno = np.asarray(geno, dtype=float)
    n, m = geno.shape
    bim = pd.DataFrame({"chrom": np.asarray(chroms, dtype=str),
                        "snp": [f"rs{i}" for i in range(m)],
                        "cm": 0, "pos": np.asarray(positions, dtype=int),
                        "a0": a0, "a1": a1})
    bim.to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)
    fam = pd.DataFrame({"fid": iids, "iid": iids, "father": 0, "mother": 0,
                        "sex": 0, "pheno": -9})
    fam.to_csv(f"{prefix}.fam", sep="\t", header=False, index=False)

    # dosage -> 2-bit code: 0->0b00, 1->0b10, 2->0b11, NaN->0b01
    code = np.select([np.isnan(geno), geno == 0, geno == 1, geno == 2],
                     [0b01, 0b00, 0b10, 0b11], default=0b01).astype(np.uint8)
    bpv = (n + 3) // 4
    out = np.zeros((m, bpv), dtype=np.uint8)
    for v in range(m):
        padded = np.zeros(bpv * 4, dtype=np.uint8)
        padded[:n] = code[:, v]
        out[v] = (padded[0::4] | (padded[1::4] << 2)
                  | (padded[2::4] << 4) | (padded[3::4] << 6))
    with open(f"{prefix}.bed", "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(out.tobytes())


def simulate(prefix, n_fam=200, sibs=4, n_chrom=3, snps_per_chrom=1000,
             Va=0.25, Vd=0.35, Ve=0.40, n_causal_add=60, n_causal_dom=60,
             big_dom_effect=6.0, seed=0):
    """Simulate a family-structured cohort (independent loci) with additive and
    dominance polygenic phenotypes, write it to ``prefix``, and return metadata.

    Injects one large-effect over-dominant variant so classification/power can
    be checked. Returns a dict with the causal indices and a phenotype frame.

    Parameters
    ----------
    prefix : str or os.PathLike
        PLINK fileset prefix without the .bed/.bim/.fam suffix (or .pgen/.pvar/.psam for a
        PLINK2 adapter). Companion metadata and genotype files must describe the same samples
        and variants.

    n_fam : int
        Input for the operation ``n_fam * sibs``. See ``vigor.testutils.simulate`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``200``.

    sibs : int
        Input for the operation ``n_fam * sibs``. See ``vigor.testutils.simulate`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape. Default: ``4``.

    n_chrom : int
        Input for the operation ``n_chrom * snps_per_chrom``. See ``vigor.testutils.simulate``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``3``.

    snps_per_chrom : int
        Input for the operation ``n_chrom * snps_per_chrom``. See ``vigor.testutils.simulate``
        and its caller for the scientific representation; the shown operation constrains the
        required type/shape. Default: ``1000``.

    Va : array-like or numeric scalar
        Input for the operation ``ga / ga.std() * np.sqrt(Va)``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``0.25``.

    Vd : array-like or numeric scalar
        Input for the operation ``gd / gd.std() * np.sqrt(Vd)``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``0.35``.

    Ve : array-like or numeric scalar
        Input for the operation ``ga + gd + rng.normal(size=n) * np.sqrt(Ve)``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``0.4``.

    n_causal_add : int
        Input for the operation ``rng.choice(m, n_causal_add, replace=False)``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``60``.

    n_causal_dom : int
        Input for the operation ``rng.choice(rem, n_causal_dom, replace=False)``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``60``.

    big_dom_effect : float
        Input for the operation ``y_dom + big_dom_effect * (het - het.mean())``. See
        ``vigor.testutils.simulate`` and its caller for the scientific representation; the shown
        operation constrains the required type/shape. Default: ``6.0``.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    Returns
    -------
    dict
        Result fields: ``pheno``, ``causal_add``, ``causal_dom``, ``big_dom_snp``, ``big_dom_name``, ``n``, ``m``, ``maf``. Field arrays preserve the corresponding input axis order.
    """
    rng = np.random.default_rng(seed)
    n = n_fam * sibs
    m = n_chrom * snps_per_chrom
    maf = rng.uniform(0.15, 0.5, size=m)

    G = np.empty((n, m), dtype=np.float32)
    r = 0
    for _ in range(n_fam):
        pa1 = (rng.random((2, m)) < maf).astype(np.int8)
        pa2 = (rng.random((2, m)) < maf).astype(np.int8)
        for _ in range(sibs):
            f0 = np.where(rng.random(m) < 0.5, pa1[0], pa2[0])
            f1 = np.where(rng.random(m) < 0.5, pa1[1], pa2[1])
            G[r] = f0 + f1
            r += 1

    def zcols(X):
        """Return the zcols result computed by ``np.nan_to_num``.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``np.nan_to_num((X - X.mean(0)) / sd)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X = X.astype(np.float64)
        sd = X.std(0)
        sd[sd == 0] = np.nan
        return np.nan_to_num((X - X.mean(0)) / sd)

    Za = zcols(G)
    Zd = zcols((G == 1).astype(float))

    ca = rng.choice(m, n_causal_add, replace=False)
    rem = np.setdiff1d(np.arange(m), ca)
    cd = rng.choice(rem, n_causal_dom, replace=False)

    ga = Za[:, ca] @ rng.normal(size=ca.size)
    ga = ga / ga.std() * np.sqrt(Va)
    gd = Zd[:, cd] @ rng.normal(size=cd.size)
    gd = gd / gd.std() * np.sqrt(Vd)

    y_add = ga + rng.normal(size=n) * np.sqrt(1 - Va)
    y_dom = gd + rng.normal(size=n) * np.sqrt(1 - Vd)
    y_mix = ga + gd + rng.normal(size=n) * np.sqrt(Ve)
    y_null = rng.normal(size=n)

    # a single large over-dominant locus (het very different from both homs)
    big = int(rng.choice(rem))
    het = (G[:, big] == 1).astype(float)
    y_dom = y_dom + big_dom_effect * (het - het.mean())

    iids = [f"ID{i:05d}" for i in range(n)]
    chroms = np.repeat(np.arange(1, n_chrom + 1).astype(str), snps_per_chrom)
    positions = np.tile(np.arange(1, snps_per_chrom + 1) * 1000, n_chrom)
    write_plink(prefix, G, iids, chroms, positions)

    pheno = pd.DataFrame({"y_add": y_add, "y_dom": y_dom,
                          "y_mix": y_mix, "y_null": y_null}, index=iids)
    return {"pheno": pheno, "causal_add": ca, "causal_dom": cd,
            "big_dom_snp": big, "big_dom_name": f"rs{big}",
            "n": n, "m": m, "maf": maf}
