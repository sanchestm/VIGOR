"""Portable synthetic joint-trait VIGOR fine-mapping smoke benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vigor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=215)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    x = pd.DataFrame(rng.binomial(2, .3, (80, 24)), index=[f'i{i}' for i in range(80)],
                     columns=[f'v{i}' for i in range(24)])
    y = pd.DataFrame(x.iloc[:, 0].to_numpy()[:, None] * np.array([[.7, -.5]])
                     + rng.normal(size=(80, 2)), index=x.index, columns=['a', 'b'])
    start = time.perf_counter()
    result = vigor.fine_map_region(x, y, trait_subset=['a', 'b'], heritability=[.5, .5],
                                   methods=('bayes',), L=3, hdbscan=False,
                                   compute_linear_shap=False, use_evidence_priors=False)
    pip = result.ranking.bayes_pip
    assert np.isfinite(pip).all() and pip.between(0, 1).all()
    summary = dict(dataset='synthetic', n_samples=80, n_variants=24,
                   seconds=time.perf_counter() - start, top_variant=str(pip.idxmax()),
                   scope='joint-model smoke; no causal accuracy claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
