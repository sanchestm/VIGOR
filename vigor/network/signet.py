from __future__ import annotations
import pandas as pd,numpy as np
from .pcsf import diffuse_scores
def rank_genes_signet_like(edges,seeds,*,alpha=.85):
    """Return the rank genes signet like result computed by ``s.sort_values(ascending=False).rename('score').to_frame``.

    Parameters
    ----------
    edges : sequence or mapping or None
        Graph adjacency/path specification or edge-aligned values. Node pairs must match the
        graph's trait/gene identifiers; None imposes no optional constraint/annotation.

    seeds : object (see the cited input contract)
        Input for the operation ``diffuse_scores(edges, seeds, alpha=alpha)``. See
        ``vigor.network.signet.rank_genes_signet_like`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.85``. Passed to ``diffuse_scores`` as ``alpha``.

    Returns
    -------
    object
        Computed result from ``s.sort_values(ascending=False).rename('score').to_frame()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    s=diffuse_scores(edges,seeds,alpha=alpha);return s.sort_values(ascending=False).rename('score').to_frame()
