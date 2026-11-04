"""Portable synthetic VIGOR graph benchmark; no cohort data required."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vigor.finemap import build_haplotype_radix_dag, haplotype_graph_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=80)
    parser.add_argument('--variants', type=int, default=32)
    parser.add_argument('--seed', type=int, default=215)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    x = pd.DataFrame(rng.binomial(2, .3, (args.samples, args.variants)),
                     index=[f'i{i}' for i in range(args.samples)],
                     columns=[f'v{i}' for i in range(args.variants)])
    variants = pd.DataFrame({'chrom': '1', 'pos': np.arange(args.variants) * 1000 + 1},
                            index=x.columns)
    start = time.perf_counter()
    graph = build_haplotype_radix_dag(x, variants=variants, lead_variant=x.columns[0],
                                     ld_mode='sparse', ld_max_variant_lag=16)
    table = haplotype_graph_variants(graph)
    assert len(table) == args.variants
    result = dict(dataset='synthetic', n_samples=args.samples, n_variants=args.variants,
                  n_nodes=graph.number_of_nodes(), n_edges=graph.number_of_edges(),
                  seconds=time.perf_counter() - start, scope='functional timing; no accuracy claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
