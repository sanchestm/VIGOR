
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
try:
    import pymc as pm
    import pytensor.tensor as pt
except Exception:  # optional Bayesian layer must not break covariance utilities
    pm = None
    pt = None
import networkx as nx
from itertools import combinations
from scipy.optimize import minimize
from scipy.stats import norm

from .results import result_dataset, maybe_save
from .attribution import _table_vars, _result_table

# Optional plotting deps (only required if you call plot_graph)
try:
    import holoviews as hv
    import hvplot.networkx as hvnx
except ImportError:
    hv = None
    hvnx = None


def softplus(x: pt.TensorVariable) -> pt.TensorVariable:
    # numerically stable softplus
    """Compute log(1+exp(x)) using a numerically stable symbolic form.

    Parameters
    ----------
    x : pt.TensorVariable
        Numerical operand in ``pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0.0)``. Shape must
        conform to that matrix/vector operation and the enclosing prepared state.

    Returns
    -------
    object
        Computed result from ``pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0.0)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0.0)


def _corr_from_cov(Sigma: pt.TensorVariable, eps: float = 1e-12) -> pt.TensorVariable:
    """Normalize symbolic covariance by its floored marginal standard deviations.

    Parameters
    ----------
    Sigma : pt.TensorVariable
        Numerical operand in ``Sigma / (denom + eps)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-12``.
    """
    d = pt.sqrt(pt.clip(pt.diag(Sigma), eps, np.inf))
    denom = d[:, None] * d[None, :]
    return Sigma / (denom + eps)


def _pack_pattern_mask(M: np.ndarray) -> np.ndarray:
    """Pack boolean missing mask (n x t) into bytes per row so we can count unique patterns fast.
    M: True where missing
    Returns: uint8 packed, shape (n, ceil(t/8))

    Parameters
    ----------
    M : np.ndarray
        Numerical operand in ``np.packbits(M, axis=1)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.
    """
    M = np.asarray(M, dtype=np.uint8)
    return np.packbits(M, axis=1)


@dataclass
class _ModelBuildInfo:
    """_ModelBuildInfo represents model build info.

    Attributes
    ----------
    engine : str
        Stored engine; its axis/order follows the construction inputs.
    obs_ids : list[str]
        Stored obs ids; its axis/order follows the construction inputs.
    trait_nodes : list[str]
        Stored trait nodes; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    engine : str
        Stored engine supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    obs_ids : list[str]
        Stored obs ids supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    trait_nodes : list[str]
        Stored trait nodes supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.
    """
    engine: str
    obs_ids: list[str]
    trait_nodes: list[str]


class CausalGraph:
    """
    DAG -> PyMC compiler for multivariate genetic models with:

    - Exogenous nodes: pm.Data, optionally standardized and QR-reduced.
    - GRM nodes: low-rank representation X = U * sqrt(S), where K ≈ X X^T.
    - Latent nodes: per-individual latent variables (e.g., genomicSEM-style factors).
    - Endogenous nodes: observed nodes with Normal/Poisson/Bernoulli likelihoods.

    Key design points for speed + missingness + interpretability
    -----------------------------------------------------------
    1) Missingness:
       - missing='mask' uses a flattened observed-only likelihood (no imputation RV explosion).
       - missing='pattern' is used only for full MVN residual models (lkj/fa) when pattern count is small.
       - missing='impute' uses PyMC masked arrays (can explode dimensionality; keep for small problems).
       - missing='row' is for collapsed marginal-eVD fast path (requires row-complete after filtering).

    2) Residual covariance:
       - residual='diag' is fastest and works naturally with missing='mask'.
       - residual='factor' implements a generative low-rank residual covariance (exact under masking),
         with identifiable loadings (lower-triangular + positive diagonal).
       - residual='lkj'/'fa' are marginal MVN residual models; masking requires missing='pattern' or 'impute'.

    3) Genetic correlations:
       - GRM edges can use grm_trait_cov='lkj' and we store deterministics:
         SigmaG:<edge>, RG:<edge>, V_G_param:<edge>.

    4) State-of-the-art latent spaces:
       - factor edges (latent->traits) use identifiable loadings by default.

    Notes:
      - This is designed to be NUTS-viable for moderate N (e.g., 1k–5k) and r (e.g., 200–800),
        and to have practical MAP/ADVI behavior by avoiding missing-value latent variables by default.
    """

    def __init__(self, df_main: pd.DataFrame):
        """Initialize CausalGraph with the inputs and resource settings below.

        Parameters
        ----------
        df_main : pd.DataFrame
            Main trait observations or selected trait column names for graph construction. Matrix
            rows identify animals and columns traits; named columns must exist in the main frame.
        """
        self.df_main = df_main
        self.obs_ids = [str(x) for x in df_main.index.tolist()]
        self.G = nx.DiGraph()
        self._last_build: _ModelBuildInfo | None = None

    # -----------------------------
    # basic utilities
    # -----------------------------
    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        """Validate/coerce 2d.

        Parameters
        ----------
        x : np.ndarray
            Numerical operand in ``x.ndim == 1``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    @staticmethod
    def _as_str_coords(coords) -> list[str]:
        """Convert inputs to str coords.

        Parameters
        ----------
        coords : object (see the cited input contract)
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller.
        """
        return [str(c) for c in coords]

    @staticmethod
    def _center_scale(arr: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Center scale.

        Parameters
        ----------
        arr : np.ndarray
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        eps : float
            Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
            denominators or nonpositive factor variances. Choose a positive value on the scale of
            the input quantities. Default: ``1e-08``.
        """
        arr = np.asarray(arr, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        std = np.where(std < eps, 1.0, std)
        out = (arr - mean) / std
        return out, mean, std

    @staticmethod
    def decompose_grm(K: np.ndarray, *, r: int, jitter: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Decompose genomic relationship matrix.

        Parameters
        ----------
        K : np.ndarray
            Numerical operand in ``0.5 * (np.asarray(K, dtype=float) + np.asarray(K,
            dtype=float).T)``. Shape must conform to that matrix/vector operation and the enclosing
            prepared state.

        r : int
            Local position, index selection or step used by the stated slice/iteration. Indices must
            refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
            the enclosing dimensions.

        jitter : float
            Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
            denominators or nonpositive factor variances. Choose a positive value on the scale of
            the input quantities. Default: ``1e-06``.

        Returns
        -------
        object
            Computed result from ``(evecs, evals)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
        n = K.shape[0]
        r = max(1, min(int(r), n - 1))
        K = K + float(jitter) * np.eye(n)
        evals, evecs = np.linalg.eigh(K)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, np.inf)
        evecs = evecs[:, order]
        evals, evecs = evals[:r], evecs[:, :r]
        return evecs, evals

    # -----------------------------
    # node creation
    # -----------------------------
    def add_input(
        self,
        name: str,
        data: pd.DataFrame | pd.Series | np.ndarray,
        *,
        fillna_for_exo: float = 0.0,
        standardize: bool = True,
        qr: bool = False,
    ) -> None:
        """Add input.

        Parameters
        ----------
        name : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector. Passed to
            ``self.G.add_node`` as ``dim_name``.

        data : pd.DataFrame | pd.Series | np.ndarray
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        fillna_for_exo : float
            Value replacing missing entries of an exogenous graph input before construction; choose
            a value meaningful on that input's standardized/raw scale. Default: ``0.0``. Passed to
            ``np.nan_to_num`` as ``nan``.

        standardize : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.G.add_node`` as ``standardize``.

        qr : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_node`` as ``qr``.
        """
        data = data.reindex(self.df_main.index) if isinstance(data, (pd.Series, pd.DataFrame)) else data
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float).reshape(-1, 1)
            coords = [data.name if data.name else f"{name}1"]
        elif isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=float)
            coords = list(map(str, data.columns))
        else:
            arr = self._ensure_2d(np.asarray(data, dtype=float))
            coords = [f"{name}{i}" for i in range(1, arr.shape[1] + 1)]

        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=float(fillna_for_exo))

        mean = std = None
        if standardize and arr.shape[1] > 0:
            arr, mean, std = self._center_scale(arr)

        R = None
        if qr and arr.shape[1] > 0:
            Q, R = np.linalg.qr(arr, mode="reduced")
            arr = Q

        self.G.add_node(
            name,
            type="exo",
            data=arr.astype(float),
            shape=arr.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords(coords),
            standardize=bool(standardize),
            mean=None if mean is None else mean.astype(float),
            scale=None if std is None else std.astype(float),
            qr=bool(qr),
            qr_R=None if R is None else R.astype(float),
        )

    def add_grm(
        self,
        name: str,
        grm_matrix: pd.DataFrame | np.ndarray,
        *,
        n_components: int = 300,
        top_eigen2fixed: float | None = None,
        subtract_fixed: str | None = None,
        jitter: float = 1e-6,
        store_full: bool = False,
        align: Literal["strict", "intersection"] = "strict",
    ) -> None:
        """Adds a GRM node using low-rank X = U*sqrt(S) with K ≈ X X^T.

        - align='strict': require GRM to contain all obs_ids (recommended).
        - top_eigen2fixed: move top fraction of eigen-variance into an exo node '<name>_fixed'
          (dense fixed-effect PCs).
        - subtract_fixed: project out an existing exo node from the random-effect X.
        - store_full: store K_full (dense) for exact/collapsed compilers (memory heavy).

        Parameters
        ----------
        name : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector. Passed to
            ``self.G.add_node`` as ``dim_name``.

        grm_matrix : pd.DataFrame | np.ndarray
            Input for the operation ``grm_matrix.loc[self.df_main.index, self.df_main.index]``. See
            ``vigor.causal.CausalGraph.add_grm`` and its caller for the scientific representation;
            the shown operation constrains the required type/shape.

        n_components : int
            Number of retained latent components or feature directions. Must fit the available
            matrix dimensions; None delegates selection to the decomposition helper when supported.
            Default: ``300``. Passed to ``self.decompose_grm`` as ``r``.

        top_eigen2fixed : float | None
            Input for the operation ``float(top_eigen2fixed)``. See
            ``vigor.causal.CausalGraph.add_grm`` and its caller for the scientific representation;
            the shown operation constrains the required type/shape. Default: ``None``. None
            delegates to the local fallback shown in this body; it is not an additional
            numeric/category value. Source comparisons: ``0.0 < frac < 1.0``. See the
            rejection/fallback branches for enforcement.

        subtract_fixed : str | None
            Input for the operation ``self.G.nodes[subtract_fixed]``. See
            ``vigor.causal.CausalGraph.add_grm`` and its caller for the scientific representation;
            the shown operation constrains the required type/shape. Default: ``None``. None
            delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        jitter : float
            Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
            denominators or nonpositive factor variances. Choose a positive value on the scale of
            the input quantities. Default: ``1e-06``. Passed to ``self.decompose_grm`` as
            ``jitter``.

        store_full : bool
            Input for the operation ``bool(store_full)``. See ``vigor.causal.CausalGraph.add_grm``
            and its caller for the scientific representation; the shown operation constrains the
            required type/shape. Default: ``False``.

        align : Literal['strict', 'intersection']
            Input for the operation ``align == 'strict'``. See ``vigor.causal.CausalGraph.add_grm``
            and its caller for the scientific representation; the shown operation constrains the
            required type/shape. Default: ``'strict'``. Source-recognized literal values:
            ``intersection``. These select the corresponding branches below; comparison evidence
            alone does not establish an exhaustive enum.
        """
        if isinstance(grm_matrix, pd.DataFrame):
            if align == "strict":
                if not (set(self.df_main.index).issubset(set(grm_matrix.index)) and set(self.df_main.index).issubset(set(grm_matrix.columns))):
                    raise ValueError("GRM does not contain all obs_ids; use align='intersection' or pre-align.")
                K = grm_matrix.loc[self.df_main.index, self.df_main.index].to_numpy(dtype=float)
            else:
                idx = self.df_main.index.intersection(grm_matrix.index).intersection(grm_matrix.columns)
                self.df_main = self.df_main.loc[idx]
                self.obs_ids = [str(x) for x in self.df_main.index.tolist()]
                K = grm_matrix.loc[idx, idx].to_numpy(dtype=float)
        else:
            K = np.asarray(grm_matrix, dtype=float)

        n = K.shape[0]
        U, S = self.decompose_grm(K, r=int(n_components), jitter=float(jitter))
        X = U * np.sqrt(S)[None, :]

        fixed_node = None
        if top_eigen2fixed is not None:
            frac = float(top_eigen2fixed)
            if not (0.0 < frac < 1.0):
                raise ValueError("top_eigen2fixed must be a float in (0,1).")
            total = float(np.sum(S)) if float(np.sum(S)) > 0 else 1.0
            cum = np.cumsum(S) / total
            cutoff = int(np.searchsorted(cum, frac) + 1)
            cutoff = max(1, min(cutoff, X.shape[1] - 1))
            X_fixed = X[:, :cutoff]
            fixed_node = f"{name}_fixed"
            self.G.add_node(
                fixed_node,
                type="exo",
                data=X_fixed.astype(float),
                shape=X_fixed.shape,
                dim_name=f"{fixed_node}_dim",
                coords=self._as_str_coords([f"Pc{i}" for i in range(1, cutoff + 1)]),
                standardize=False,
                mean=None,
                scale=None,
                qr=False,
                qr_R=None,
            )
            X = X[:, cutoff:]

        if subtract_fixed is not None:
            if subtract_fixed not in self.G.nodes:
                raise ValueError(f"subtract_fixed='{subtract_fixed}' not found.")
            fix = self.G.nodes[subtract_fixed]
            if fix["type"] != "exo":
                raise ValueError("subtract_fixed must be an exo node.")
            C = np.asarray(fix["data"], dtype=float)
            beta, *_ = np.linalg.lstsq(C, X, rcond=None)
            X = X - C @ beta

        kdiag = np.sum(X**2, axis=1)
        K_full = None
        if bool(store_full):
            K_full = 0.5 * (K + K.T)

        self.G.add_node(
            name,
            type="grm",
            data=X.astype(float),
            shape=X.shape,
            dim_name=f"{name}_pc",
            coords=self._as_str_coords([f"Pc{i}" for i in range(1, X.shape[1] + 1)]),
            fixed_node=fixed_node,
            K_full=K_full,
            kdiag_mean=float(np.mean(kdiag)),
        )

    def add_latent(
        self,
        name: str,
        *,
        n_dim: int = 1,
        prior_sigma: float = 1.0,
        track_values: bool = True,
        variance_budget: bool = False,
        budget_w_eps: float = 1e-6,
        budget_concentration: float = 20.0,
        is_factor: bool = False,
    ) -> None:
        """Add latent.

        Parameters
        ----------
        name : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector. Passed to
            ``self.G.add_node`` as ``dim_name``.

        n_dim : int
            Latent/trait factor dimension. Positive integer sizes specify model capacity; None uses
            the graph builder's automatic factor count where supported. Default: ``1``. Passed to
            ``self.G.add_node`` as ``shape``.

        prior_sigma : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``1.0``.
            Passed to ``self.G.add_node`` as ``prior_sigma``.

        track_values : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.G.add_node`` as ``track_values``.

        variance_budget : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_node`` as ``variance_budget``.

        budget_w_eps : float
            Positive floor stabilizing variance-budget allocation weights before normalizing them.
            Default: ``1e-06``. Passed to ``self.G.add_node`` as ``budget_w_eps``.

        budget_concentration : float
            Positive concentration for the prior over variance-budget proportions, controlling
            concentration around the declared allocation. Default: ``20.0``. Passed to
            ``self.G.add_node`` as ``budget_concentration``.

        is_factor : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_node`` as ``is_factor``.
        """
        n_dim = int(n_dim)
        self.G.add_node(
            name,
            type="latent",
            data=None,
            shape=(len(self.obs_ids), n_dim),
            dim_name=f"{name}_dim",
            coords=self._as_str_coords([f"{name}{i}" for i in range(1, n_dim + 1)]),
            prior_sigma=float(prior_sigma),
            track_values=bool(track_values),
            variance_budget=bool(variance_budget),
            budget_w_eps=float(budget_w_eps),
            budget_concentration=float(budget_concentration),
            is_factor=bool(is_factor),
        )

    def add_adjacency(
        self,
        name: str,
        *,
        W: "pd.DataFrame | np.ndarray | Any | None" = None,
        edges: "pd.DataFrame | list[tuple[Any, Any]] | None" = None,
        src_col: str = "iid1",
        dst_col: str = "iid2",
        weight_col: str | None = None,
        binary: bool = True,
        threshold: float = 0.0,
        symmetrize: bool = True,
        drop_diagonal: bool = True,
        align: Literal["strict", "intersection"] = "strict",
    ) -> None:
        """Add an adjacency node for BYM2 edges.

        Provide either:
          - W: adjacency matrix (DataFrame with index/cols, ndarray, or scipy sparse), OR
          - edges: edge list (DataFrame or list of (src, dst)).

        BYM2/ICAR expects symmetric (undirected) adjacency; we symmetrize by default.
        If binary=True, any nonzero becomes 1 (optionally after threshold).

        Parameters
        ----------
        name : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector. Passed to
            ``self.G.add_node`` as ``dim_name``.

        W : 'pd.DataFrame | np.ndarray | Any | None'
            Pairwise weights or jointly observed counts, aligned to the trait/variant-pair axes of
            the associated covariance/LD matrix. Missing-data precision calculations must use the
            actual shared observations. Default: ``None``. None delegates to the local fallback
            shown in this body; it is not an additional numeric/category value.

        edges : 'pd.DataFrame | list[tuple[Any, Any]] | None'
            Graph adjacency/path specification or edge-aligned values. Node pairs must match the
            graph's trait/gene identifiers; None imposes no optional constraint/annotation. Default:
            ``None``.

        src_col : str
            Column name selecting identifiers/weights in the supplied table. It must exist and
            contain the corresponding animal/node/group values. Default: ``'iid1'``.

        dst_col : str
            Column name selecting identifiers/weights in the supplied table. It must exist and
            contain the corresponding animal/node/group values. Default: ``'iid2'``.

        weight_col : str | None
            Column name selecting identifiers/weights in the supplied table. It must exist and
            contain the corresponding animal/node/group values. Default: ``None``. None delegates to
            the local fallback shown in this body; it is not an additional numeric/category value.

        binary : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``.

        threshold : float
            Decision cutoff(s) on the statistic named by this classifier/estimator. A mapping
            supplies named dominance/inheritance thresholds; see the classifier defaults and
            validation for their domain. Default: ``0.0``. Source comparisons: ``float(threshold) >
            0``. See the rejection/fallback branches for enforcement.

        symmetrize : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``.

        drop_diagonal : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``.

        align : Literal['strict', 'intersection']
            Input for the operation ``align == 'strict'``. See
            ``vigor.causal.CausalGraph.add_adjacency`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``'strict'``. Source-recognized literal values: ``intersection``. These select the
            corresponding branches below; comparison evidence alone does not establish an exhaustive
            enum.
        """
        import scipy.sparse as sps
        obs = self.df_main.index
        n = len(obs)
    
        # ---- build sparse adjacency ----
        if W is None:
            if edges is None:
                raise ValueError("Provide W or edges to add_adjacency().")
    
            if isinstance(edges, pd.DataFrame):
                if (src_col not in edges.columns) or (dst_col not in edges.columns):
                    raise ValueError(f"edges must contain columns '{src_col}' and '{dst_col}'.")
                src = edges[src_col].astype(str).to_numpy()
                dst = edges[dst_col].astype(str).to_numpy()
                if (weight_col is not None) and (weight_col in edges.columns) and (not binary):
                    val = edges[weight_col].astype(float).to_numpy()
                else:
                    val = np.ones(len(edges), dtype=float)
            else:
                pairs = [(str(a), str(b)) for a, b in edges]
                src = np.asarray([a for a, _ in pairs], dtype=object)
                dst = np.asarray([b for _, b in pairs], dtype=object)
                val = np.ones(len(pairs), dtype=float)
    
            id2i = {str(x): i for i, x in enumerate(obs)}
            rr, cc, vv = [], [], []
            for a, b, w in zip(src, dst, val, strict=False):
                ia = id2i.get(str(a), None)
                ib = id2i.get(str(b), None)
                if ia is None or ib is None:
                    continue
                rr.append(ia)
                cc.append(ib)
                vv.append(float(w))
    
            Wsp = sps.csr_matrix((vv, (rr, cc)), shape=(n, n))
    
        else:
            if isinstance(W, pd.DataFrame):
                if align == "strict":
                    if not (set(obs).issubset(set(W.index)) and set(obs).issubset(set(W.columns))):
                        raise ValueError("Adjacency DF missing obs_ids; pre-align or use align='intersection'.")
                    A = W.loc[obs, obs].to_numpy()
                else:
                    idx = obs.intersection(W.index).intersection(W.columns)
                    self.df_main = self.df_main.loc[idx]
                    self.obs_ids = [str(x) for x in self.df_main.index.tolist()]
                    obs = self.df_main.index
                    n = len(obs)
                    A = W.loc[obs, obs].to_numpy()
                Wsp = sps.csr_matrix(A)
            else:
                Wsp = sps.csr_matrix(W) if not sps.issparse(W) else W.tocsr()
                if Wsp.shape != (n, n):
                    raise ValueError(f"W must have shape {(n, n)} matching df_main; got {Wsp.shape}.")
    
        # ---- sanitize ----
        if binary:
            if float(threshold) > 0:
                Wsp = (Wsp > float(threshold)).astype(float).tocsr()
            else:
                if Wsp.nnz > 0:
                    Wsp.data[:] = 1.0
    
        if symmetrize:
            Wsp = (Wsp + Wsp.T)
            if binary:
                Wsp = (Wsp > 0).astype(float).tocsr()
            else:
                Wsp = (0.5 * Wsp).tocsr()
    
        if drop_diagonal:
            Wsp.setdiag(0.0)
            Wsp.eliminate_zeros()
    
        # basic stats for QC
        deg = np.asarray(Wsp.sum(axis=1)).ravel()
        mean_deg = float(deg.mean()) if len(deg) else 0.0
        # ---- dummy data so pm.Data exists (compiler needs model_vars[parent]) ----
        dummy = np.zeros((n, 1), dtype=float)
    
        self.G.add_node(
            name,
            type="adj",
            data=dummy,
            shape=dummy.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords([f"{name}1"]),
            standardize=False,
            mean=None,
            scale=None,
            qr=False,
            qr_R=None,
            W=Wsp.toarray(),
            nnz=int(Wsp.nnz),
            mean_degree=mean_deg,
        )

    def add_trait(
        self,
        name: str,
        *,
        data_matrix: pd.DataFrame | np.ndarray | None = None,
        data_cols: list[str] | None = None,
        likelihood: Literal["normal", "poisson", "bernoulli"] = "normal",
        link: Literal["identity", "log", "logit"] = "identity",
        residual: Literal["diag", "lkj", "fa", "factor"] = "diag",
        # residual hyperparams
        eta_E: float = 4.0,
        var_E: float = 1.0,
        n_factors_E: int | None = None,
        # missing handling (NEW)
        missing: Literal["auto", "mask", "pattern", "impute", "row"] = "auto",
        pattern_max_groups: int = 2048,
        pattern_max_t: int = 25,
        # backwards compat
        impute_missing: bool | None = None,
        # tracking
        track_values: bool = True,
        track_values_per_path: bool = False,
        variance_budget: bool = False,
        budget_w_eps: float = 1e-6,
        budget_concentration: float = 20.0,
    ) -> None:
        """Add trait.

        Parameters
        ----------
        name : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector. Passed to
            ``self.G.add_node`` as ``dim_name``.

        data_matrix : pd.DataFrame | np.ndarray | None
            Main trait observations or selected trait column names for graph construction. Matrix
            rows identify animals and columns traits; named columns must exist in the main frame.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        data_cols : list[str] | None
            Main trait observations or selected trait column names for graph construction. Matrix
            rows identify animals and columns traits; named columns must exist in the main frame.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        likelihood : Literal['normal', 'poisson', 'bernoulli']
            Response distribution: 'normal' models continuous observations, 'poisson' counts and
            'bernoulli' binary observations. Match the corresponding link and data support. Default:
            ``'normal'``. Passed to ``self.G.add_node`` as ``likelihood``.

        link : Literal['identity', 'log', 'logit']
            Response link: 'identity' leaves the linear predictor unchanged, 'log' exponentiates it
            for positive mean counts and 'logit' maps it to a probability. Default: ``'identity'``.
            Passed to ``self.G.add_node`` as ``link``.

        residual : Literal['diag', 'lkj', 'fa', 'factor']
            Residual covariance model: 'diag' uses independent trait noise, 'lkj' uses an LKJ
            correlation prior, and 'fa'/'factor' uses low-rank factors plus specific variances.
            Check add_trait/build for scalar-model restrictions. Default: ``'diag'``. Passed to
            ``self.G.add_node`` as ``residual``.

        eta_E : float
            Positive LKJ correlation-prior concentration. Values above one favor weaker
            correlations; one gives the LKJ uniform-correlation prior. Default: ``4.0``. Passed to
            ``self.G.add_node`` as ``eta_E``.

        var_E : float
            Positive working residual variance scale assigned to this trait node; the selected
            residual model determines its prior parameterization. Default: ``1.0``. Passed to
            ``self.G.add_node`` as ``sd_E``.

        n_factors_E : int | None
            Latent/trait factor dimension. Positive integer sizes specify model capacity; None uses
            the graph builder's automatic factor count where supported. Default: ``None``. Passed to
            ``self.G.add_node`` as ``n_factors_E``.

        missing : Literal['auto', 'mask', 'pattern', 'impute', 'row']
            Missing-data handling selected by this entry point: 'auto' resolves a policy from matrix
            size/model; 'mask' models observed cells, 'pattern' groups observation patterns,
            'impute' treats missing values as latent, and 'row' uses complete rows where supported.
            'mean'/'mean_impute' fills with the observed mean only in helpers explicitly supporting
            it. See the local validator for the accepted subset. Default: ``'auto'``. Passed to
            ``self.G.add_node`` as ``missing``.

        pattern_max_groups : int
            Maximum distinct observation patterns or trait count permitted by pattern-group
            missing-data likelihoods; exceeding the guard selects the supported fallback. Default:
            ``2048``. Passed to ``self.G.add_node`` as ``pattern_max_groups``.

        pattern_max_t : int
            Maximum distinct observation patterns or trait count permitted by pattern-group
            missing-data likelihoods; exceeding the guard selects the supported fallback. Default:
            ``25``. Passed to ``self.G.add_node`` as ``pattern_max_t``.

        impute_missing : bool | None
            Input for the operation ``bool(impute_missing)``. See
            ``vigor.causal.CausalGraph.add_trait`` and its caller for the scientific representation;
            the shown operation constrains the required type/shape. Default: ``None``. None
            delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        track_values : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.G.add_node`` as ``track_values``.

        track_values_per_path : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_node`` as ``track_values_per_path``.

        variance_budget : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_node`` as ``variance_budget``.

        budget_w_eps : float
            Positive floor stabilizing variance-budget allocation weights before normalizing them.
            Default: ``1e-06``. Passed to ``self.G.add_node`` as ``budget_w_eps``.

        budget_concentration : float
            Positive concentration for the prior over variance-budget proportions, controlling
            concentration around the declared allocation. Default: ``20.0``. Passed to
            ``self.G.add_node`` as ``budget_concentration``.
        """
        if data_matrix is None and data_cols is None:
            raise ValueError("Provide data_matrix or data_cols.")
        if data_matrix is None:
            data = self.df_main[list(data_cols)]
        else:
            data = data_matrix
        data = data.reindex(self.df_main.index) if isinstance(data, (pd.DataFrame, pd.Series)) else data
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float).reshape(-1, 1)
            coords = [data.name if data.name else f"{name}1"]
        elif isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=float)
            coords = list(map(str, data.columns))
        else:
            arr = self._ensure_2d(np.asarray(data, dtype=float))
            coords = [f"{name}{i}" for i in range(1, arr.shape[1] + 1)]

        likelihood = str(likelihood).lower()
        residual = str(residual).lower()
        link = str(link).lower()

        # Backwards compat: if impute_missing provided, map to missing mode
        if impute_missing is not None:
            missing = "impute" if bool(impute_missing) else "auto"

        self.G.add_node(
            name,
            type="endo",
            data=arr,
            shape=arr.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords(coords),
            likelihood=likelihood,
            link=link,
            residual=residual,
            eta_E=float(eta_E),
            sd_E=float(np.sqrt(max(float(var_E), 1e-12))),
            n_factors_E=None if n_factors_E is None else int(n_factors_E),
            missing=str(missing).lower(),
            pattern_max_groups=int(pattern_max_groups),
            pattern_max_t=int(pattern_max_t),
            track_values=bool(track_values),
            track_values_per_path=bool(track_values_per_path),
            variance_budget=bool(variance_budget),
            budget_w_eps=float(budget_w_eps),
            budget_concentration=float(budget_concentration),
        )

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        kind: Literal["auto", "dense", "grm", "factor", 'bym2'] = "auto",
        prior_variance: float = 1.0,
        eta: float = 2.0,
        grm_trait_cov: Literal["lkj", "diag", "scalar"] = "lkj",
        z_prior: Literal["iid", "ard_pc", "rhs"] = "iid",
        rhs_global_scale: float = 0.3,
        factor_identifiable: bool = True,
        track_variance: bool = False,
        adjust_variance_by_shape: bool = True,
    ) -> None:
        """Add edge.

        Parameters
        ----------
        source : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        target : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        kind : Literal['auto', 'dense', 'grm', 'factor', 'bym2']
            Edge/evidence/input interpretation selected by this constructor/helper. See the local
            branches and the source-derived option descriptions below; each builder admits only its
            own subset, not a package-wide enum. Default: ``'auto'``. Source-recognized literal
            values: ``bym2``, ``dense``, ``factor``, ``grm``. These select the corresponding
            branches below; comparison evidence alone does not establish an exhaustive enum. Passed
            to ``self.G.add_edge`` as ``kind``.

        prior_variance : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``1.0``.

        eta : float
            Positive LKJ correlation-prior concentration. Values above one favor weaker
            correlations; one gives the LKJ uniform-correlation prior. Default: ``2.0``. Passed to
            ``self.G.add_edge`` as ``eta``.

        grm_trait_cov : Literal['lkj', 'diag', 'scalar']
            Genetic trait covariance: 'lkj' uses a correlated trait prior, 'diag' independent trait
            scales, 'scalar' one shared scale and 'fa' a factor covariance only where the builder
            supports it. See the local validator for the accepted subset. Default: ``'lkj'``. Passed
            to ``self.G.add_edge`` as ``grm_trait_cov``.

        z_prior : Literal['iid', 'ard_pc', 'rhs']
            Latent genetic score prior: 'iid' uses independent normal scores, 'ard_pc' allows
            PC-specific automatic relevance shrinkage and 'rhs' uses a regularized horseshoe
            shrinkage prior. Default: ``'iid'``. Passed to ``self.G.add_edge`` as ``z_prior``.

        rhs_global_scale : float
            Input supplied to ``self.G.add_edge`` as ``rhs_global_scale``. See that callable's
            documented ``rhs_global_scale`` contract for its type, alignment and value domain. This
            wrapper retains that contract. Default: ``0.3``. Passed to ``self.G.add_edge`` as
            ``rhs_global_scale``.

        factor_identifiable : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.G.add_edge`` as ``factor_identifiable``.

        track_variance : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``False``. Passed to
            ``self.G.add_edge`` as ``track_variance``.

        adjust_variance_by_shape : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.G.add_edge`` as ``adjust_variance_by_shape``.
        """
        if source not in self.G.nodes or target not in self.G.nodes:
            raise ValueError(f"Missing nodes for edge {source}->{target}")
        src = self.G.nodes[source]
        tgt = self.G.nodes[target]
        if kind == "auto":
            if src["type"] == "grm":
                kind = "grm"
            elif src["type"] == "latent" and bool(src.get("is_factor", False)) and tgt["type"] == "endo":
                kind = "factor"
            else:
                kind = "dense"

        kind = str(kind).lower()
        prior_variance = float(prior_variance)
        if adjust_variance_by_shape and kind != "grm":
            prior_variance = prior_variance / float(max(1, src["shape"][1]))
        prior_sigma = float(np.sqrt(max(prior_variance, 1e-12)))

        self.G.add_edge(
            source,
            target,
            kind=kind,
            prior_sigma=prior_sigma,
            eta=float(eta),
            grm_trait_cov=str(grm_trait_cov).lower(),
            z_prior=str(z_prior).lower(),
            rhs_global_scale=float(rhs_global_scale),
            factor_identifiable=bool(factor_identifiable),
            track_variance=bool(track_variance),
            adjust_variance_by_shape=bool(adjust_variance_by_shape),
        )

    # -----------------------------
    # causal mediation helpers
    # -----------------------------
    def add_mediation_paths(
        self,
        mediator: str,
        outcome: str,
        *,
        exposures: str | list[str] | tuple[str, ...] = (),
        additive_grm: str | None = None,
        dominance_grm: str | None = None,
        include_direct: bool = True,
        exposure_prior_variance: float = 1.0,
        mediation_prior_variance: float = 1.0,
        grm_trait_cov: Literal["lkj", "diag", "scalar"] = "lkj",
        z_prior: Literal["iid", "ard_pc", "rhs"] = "iid",
        track_variance: bool = True,
    ) -> None:
        """Wire a classical genetic mediation subgraph using existing graph nodes.

        The helper does not create data nodes. It connects already-added nodes as

            exposure -> mediator -> outcome
            exposure ----------------> outcome

        and, when supplied, adds additive and dominance polygenic background paths
        from separate GRM nodes to both mediator and outcome. This keeps focal
        genotype effects distinct from genome-wide A/D random effects.

        Parameters
        ----------
        mediator : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        outcome : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        exposures : str | list[str] | tuple[str, ...]
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering. Default: ``()``.

        additive_grm : str | None
            Animal kernel(s) used for the stated background/genetic component. Each matrix is square
            on the same aligned animal axis; operator forms must implement the
            multiplication/diagonal interface used by this routine. Default: ``None``. None
            delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        dominance_grm : str | None
            Animal kernel(s) used for the stated background/genetic component. Each matrix is square
            on the same aligned animal axis; operator forms must implement the
            multiplication/diagonal interface used by this routine. Default: ``None``. None
            delegates to the local fallback shown in this body; it is not an additional
            numeric/category value.

        include_direct : bool
            Whether to enforce/use complete observations or include the named
            decomposition/component output. The enabled branch below defines the returned
            representation; disabling skips that optional step. Default: ``True``.

        exposure_prior_variance : float
            Input supplied to ``self.add_edge`` as ``prior_variance``. See that callable's
            documented ``prior_variance`` contract for its type, alignment and value domain. This
            wrapper retains that contract. Default: ``1.0``. Passed to ``self.add_edge`` as
            ``prior_variance``.

        mediation_prior_variance : float
            Input supplied to ``self.add_edge`` as ``prior_variance``. See that callable's
            documented ``prior_variance`` contract for its type, alignment and value domain. This
            wrapper retains that contract. Default: ``1.0``. Passed to ``self.add_edge`` as
            ``prior_variance``.

        grm_trait_cov : Literal['lkj', 'diag', 'scalar']
            Genetic trait covariance: 'lkj' uses a correlated trait prior, 'diag' independent trait
            scales, 'scalar' one shared scale and 'fa' a factor covariance only where the builder
            supports it. See the local validator for the accepted subset. Default: ``'lkj'``. Passed
            to ``self.add_edge`` as ``grm_trait_cov``.

        z_prior : Literal['iid', 'ard_pc', 'rhs']
            Latent genetic score prior: 'iid' uses independent normal scores, 'ard_pc' allows
            PC-specific automatic relevance shrinkage and 'rhs' uses a regularized horseshoe
            shrinkage prior. Default: ``'iid'``. Passed to ``self.add_edge`` as ``z_prior``.

        track_variance : bool
            Whether to enable the named graph-model transformation, deterministic tracking or
            constraint. True follows its explicit construction branch; False omits it. See that
            branch for how model dimensions/priors are affected. Default: ``True``. Passed to
            ``self.add_edge`` as ``track_variance``.
        """
        if mediator not in self.G or outcome not in self.G:
            raise ValueError("mediator and outcome must be added before add_mediation_paths()")
        exposures = [exposures] if isinstance(exposures, str) else list(exposures)
        for exposure in exposures:
            if exposure not in self.G:
                raise ValueError(f"exposure node '{exposure}' was not found")
            self.add_edge(exposure, mediator, kind="dense", prior_variance=exposure_prior_variance,
                          track_variance=track_variance)
            if include_direct:
                self.add_edge(exposure, outcome, kind="dense", prior_variance=exposure_prior_variance,
                              track_variance=track_variance)
        self.add_edge(mediator, outcome, kind="dense", prior_variance=mediation_prior_variance,
                      track_variance=track_variance)
        for grm in [additive_grm, dominance_grm]:
            if grm is None:
                continue
            if grm not in self.G or self.G.nodes[grm].get("type") != "grm":
                raise ValueError(f"'{grm}' must be an existing GRM node")
            for target in [mediator, outcome]:
                self.add_edge(grm, target, kind="grm", grm_trait_cov=grm_trait_cov,
                              z_prior=z_prior, track_variance=track_variance)
        self.G.graph.setdefault("mediation_models", []).append({
            "mediator": mediator, "outcome": outcome, "exposures": exposures,
            "additive_grm": additive_grm, "dominance_grm": dominance_grm,
            "include_direct": bool(include_direct),
        })

    def mediation_effects(
        self,
        fit: Any,
        mediator: str,
        outcome: str,
        *,
        exposures: str | list[str] | tuple[str, ...],
        hdi_prob: float = 0.95,
        return_draws: bool = False,
    ) -> dict[str, Any]:
        """Summarize direct, indirect and total effects for fitted dense mediation paths.

        For exposure X, mediator M and outcome Y, dense-edge coefficient matrices are
        A=beta[X->M], B=beta[M->Y], C'=beta[X->Y]. The indirect matrix is A@B and
        total=C'+A@B. Posterior draws are propagated exactly by matrix multiplication.

        Parameters
        ----------
        fit : Any
            Previously prepared model/result or solver state consumed by this operation. Use the
            object returned by the corresponding fitting/preparation routine with the same animal
            and trait axes.

        mediator : str
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering.

        outcome : str
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering.

        exposures : str | list[str] | tuple[str, ...]
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering.

        hdi_prob : float
            Posterior/analytical interval coverage probability, strictly between zero and one; the
            routine determines whether intervals use posterior quantiles or a normal approximation.
            Default: ``0.95``.

        return_draws : bool
            Maximum raw covariance/posterior draws included in the result. Zero omits raw draws; a
            positive count caps stored output according to the draw backend. Default: ``False``.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        exposures = [exposures] if isinstance(exposures, str) else list(exposures)
        is_post = hasattr(fit, "posterior")
        post = fit.posterior if is_post else fit

        def get_array(name: str) -> np.ndarray:
            """Extract the named posterior latent array and align it to the graph sample/factor axes.

            Parameters
            ----------
            name : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.

            Returns
            -------
            object
                Computed result from ``a[None, ...]``; see the summary and the called implementation for its labeled/numerical representation.
            """
            if is_post:
                if name not in post:
                    raise KeyError(f"'{name}' not found in posterior")
                da = post[name]
                if "chain" in da.dims and "draw" in da.dims:
                    da = da.stack(sample=("chain", "draw"))
                    rest = [d for d in da.dims if d != "sample"]
                    da = da.transpose("sample", *rest)
                elif "sample" in da.dims:
                    rest = [d for d in da.dims if d != "sample"]
                    da = da.transpose("sample", *rest)
                else:
                    da = da.expand_dims(sample=[0])
                return np.asarray(da.values, dtype=float)
            if name not in post:
                raise KeyError(f"'{name}' not found in MAP fit")
            a = np.asarray(post[name], dtype=float)
            return a[None, ...]

        b_my = get_array(f"beta:{mediator}->{outcome}")
        alpha = (1.0 - float(hdi_prob)) / 2.0
        rows = []
        draws = {}
        for exposure in exposures:
            a_xm = get_array(f"beta:{exposure}->{mediator}")
            indirect = np.einsum("dpm,dmo->dpo", a_xm, b_my, optimize=True)
            direct_name = f"beta:{exposure}->{outcome}"
            direct = get_array(direct_name) if direct_name in post else np.zeros_like(indirect)
            total = direct + indirect
            draws[exposure] = {"direct": direct, "indirect": indirect, "total": total}
            xcoords = self.G.nodes[exposure].get("coords", [f"x{i}" for i in range(indirect.shape[1])])
            ycoords = self.G.nodes[outcome].get("coords", [f"y{i}" for i in range(indirect.shape[2])])
            for kind, arr in draws[exposure].items():
                mean = arr.mean(axis=0)
                lo = np.quantile(arr, alpha, axis=0)
                hi = np.quantile(arr, 1.0-alpha, axis=0)
                for i, xn in enumerate(xcoords):
                    for j, yn in enumerate(ycoords):
                        rows.append((exposure, str(xn), mediator, outcome, str(yn), kind,
                                     float(mean[i,j]), float(lo[i,j]), float(hi[i,j])))
        summary = pd.DataFrame(rows, columns=["exposure_node", "exposure", "mediator", "outcome_node",
                                                  "outcome", "effect", "mean", "hdi_low", "hdi_high"])
        out = {"summary": summary, "hdi_prob": float(hdi_prob)}
        if return_draws:
            out["draws"] = draws
        return out

    # -----------------------------
    # graph QC / plotting
    # -----------------------------
    def graph_qc(self) -> None:
        """Report graph node/edge configuration for inspecting model construction.
                
        """
        print("--- Causal Graph QC ---")
        print(f"Nodes: {self.G.number_of_nodes()} | Edges: {self.G.number_of_edges()}")
        try:
            cycles = list(nx.simple_cycles(self.G))
            print("✅ DAG: acyclic" if not cycles else f"⚠️ cycles detected: {cycles}")
        except Exception:
            print("cycle check skipped")
        isolates = list(nx.isolates(self.G))
        if isolates:
            print(f"⚠️ isolates: {isolates}")

        nodes = pd.DataFrame.from_dict(dict(self.G.nodes(data=True)), orient="index")
        nodes.index.name = "node"
        nodes["n_parents"] = [self.G.in_degree(n) for n in nodes.index]
        nodes["n_children"] = [self.G.out_degree(n) for n in nodes.index]
        if "shape" in nodes.columns:
            nodes["shape_str"] = nodes["shape"].apply(lambda s: "x".join(map(str, s)) if isinstance(s, (tuple, list)) else str(s))
        def misspct(a):
            """Compute the fraction of missing cells in the graph node data.

            Parameters
            ----------
            a : array-like
                Numerical operand in ``a.get('data', None)``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.

            Returns
            -------
            object
                Computed result from ``float(np.isnan(x).mean()) * 100.0``; see the summary and the called implementation for its labeled/numerical representation.
            """
            x = a.get("data", None)
            if x is None:
                return np.nan
            x = np.asarray(x)
            return float(np.isnan(x).mean()) * 100.0
        nodes["missing_pct"] = [misspct(dict(self.G.nodes[n])) for n in nodes.index]
        cols = [c for c in ["type","shape_str","n_parents","n_children","missing_pct","likelihood","residual","missing"] if c in nodes.columns]
        print(nodes.reset_index()[["node"]+cols].to_markdown(index=False))

        edges = nx.to_pandas_edgelist(self.G)
        if len(edges):
            print(edges.sort_values(["source","target"]).to_markdown(index=False))

    def plot_graph(self, *, show_edge_kinds: bool = True, show_edge_param_labels: bool = True):
        """Interactive DAG view (HoloViews/Bokeh).

        - Edge width scales with the *unscaled* prior sigma (undoes variance-by-shape normalization for display).
        - Edge hover shows kind, n_params, and prior variance/sigma.
        - Optional edge midpoint labels with (n_params, variance).

        Parameters
        ----------
        show_edge_kinds : bool
            Input supplied to ``hvnx.draw_networkx_edges`` as ``edge_color``. See that callable's
            documented ``edge_color`` contract for its type, alignment and value domain. This
            wrapper retains that contract. Default: ``True``. Passed to ``hvnx.draw_networkx_edges``
            as ``edge_color``.

        show_edge_param_labels : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``True``.

        Returns
        -------
        object
            Computed result from ``graph.opts(title='GraphView', width=900, height=650, xaxis=None, yaxis=None, tools=['tap', 'box_select', 'hover'])``; see the summary and the called implementation for its labeled/numerical representation.
        """
        P = nx.DiGraph()
        # -----------------
        # nodes
        # -----------------
        for n, attr in self.G.nodes(data=True):
            ntype = str(attr.get("type", "unknown"))
            shape = attr.get("shape", None)
            shape_str = "x".join(map(str, shape)) if shape is not None else "?"
            likelihood = str(attr.get("likelihood", "N/A"))
            residual = str(attr.get("residual", "N/A"))
            missing = str(attr.get("missing", "N/A"))
            P.add_node(  n, type=ntype,  shape_str=shape_str, likelihood=likelihood, residual=residual, missing=missing, )
    
        def _edge_param_count(u: str, v: str, ed: dict[str, Any]) -> int:
            """Count parameters implied by one graph edge and its configured covariance geometry.

            Parameters
            ----------
            u : str
                Input for the operation ``self.G.nodes[u]``. See
                ``vigor.causal.CausalGraph.plot_graph._edge_param_count`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.

            v : str
                Numerical operand in ``self.G.nodes[v]``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.

            ed : dict[str, Any]
                Input for the operation ``str(ed.get('kind', 'dense')).lower()``. See
                ``vigor.causal.CausalGraph.plot_graph._edge_param_count`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.
                Source-recognized literal values: ``ard_pc``, ``bym2``, ``diag``, ``factor``, ``grm``,
                ``rhs``, ``scalar``. These select the corresponding branches below; comparison evidence
                alone does not establish an exhaustive enum.
            """
            kind = str(ed.get("kind", "dense")).lower()
            su = self.G.nodes[u].get("shape", (0, 0))
            sv = self.G.nodes[v].get("shape", (0, 0))
            p = int(su[1]) if (su is not None and len(su) > 1) else 0
            q = int(sv[1]) if (sv is not None and len(sv) > 1) else 0
    
            if kind == "dense": return int(p * q)

            if kind == "bym2":
                n_obs = int(self.G.nodes[v].get("shape", (len(self.obs_ids), 0))[0])
                d = q
                if d <= 0: return 0
                # ICAR_raw + IID_raw per dim => ~2*n_obs*d plus rho + sigma_bym
                n = int(2 * n_obs * d + 2)
                tc = str(ed.get("grm_trait_cov", "diag")).lower()
                if tc == "diag": n += int(d)
                elif tc == "scalar": n += 1
                else: n += int(d * (d + 1) // 2)
                return int(n)
    
            if kind == "factor":
                k = p
                t = q
                if k <= 0 or t <= 0:  return 0
                if bool(ed.get("factor_identifiable", True)): return int(max(0, t * k - (k * (k - 1)) // 2))
                return int(t * k)
    
            if kind == "grm":
                r,d = p, q
                if r <= 0 or d <= 0:  return 0
                n = int(r * d)
                # shrinkage prior extras
                z_prior = str(ed.get("z_prior", "iid")).lower()
                if z_prior == "ard_pc": n += int(r)  # tau_pc per PC
                elif z_prior == "rhs":  n += int(r) + 2  # lam per PC + global tau + c2
                tc = str(ed.get("grm_trait_cov", "lkj")).lower()
                if tc == "diag":n += int(d)
                elif tc == "scalar":   n += 1
                else: n += int(d * (d + 1) // 2)
                return int(n)
            return 0
    
        # -----------------
        # edges (with prior scaling + parameter counts)
        # -----------------
        for u, v, attr in self.G.edges(data=True):
            kind = str(attr.get("kind", "dense")).lower()
    
            # stored prior_sigma may be normalized by source dim (variance budget);
            # for visualization undo that normalization so thickness reflects user prior_variance.
            raw_sigma = float(attr.get("prior_sigma", 1.0))
            if bool(attr.get("adjust_variance_by_shape", False)) and kind != "grm":
                su = self.G.nodes[u].get("shape", (1, 1))
                raw_sigma *= float(np.sqrt(max(int(su[1]), 1)))
    
            variance = float(raw_sigma**2)
            confidence = float(min(4.0, max(raw_sigma, 0.1)))  # clamped width driver
            n_params = int(_edge_param_count(u, v, attr))
    
            P.add_edge(u,v, kind=kind, n_params=n_params, prior_sigma=float(raw_sigma),
                       variance=variance, confidence=confidence)
    
        # left/middle/right layout by in/out degree
        for n in P.nodes():
            out_degree = P.out_degree(n)
            in_degree = P.in_degree(n)
            subset = 0 if in_degree == 0 else (2 if out_degree == 0 else 1)
            P.nodes[n]["subset"] = int(subset)
    
        pos = nx.multipartite_layout(P, subset_key="subset", scale=2.0)
    
        color_map = {"exo": "lightsteelblue", "grm": "salmon", "latent": "palegoldenrod", "endo": "darkseagreen", 'adj': 'orange'}
        marker_map = {"exo": "circle", "grm": "hex", "latent": "square", "endo": "square",  'adj': 'hex'}
        edge_color_map = {"dense": "steelblue", "grm": "firebrick", "factor": "purple", 'bym2': 'orange'}
    
        for n in P.nodes():
            t = P.nodes[n]["type"]
            P.nodes[n]["color"] = str(color_map.get(t, "white"))
            P.nodes[n]["marker"] = str(marker_map.get(t, "circle"))
    
        edge_tooltips = [
            ("Edge", "@start -> @end"),
            ("Kind", "@kind"),
            ("Params", "@n_params"),
            ("Prior sigma", "@prior_sigma"),
            ("Prior var", "@variance") ]
        node_tooltips = [
            ("Node", "@index"),
            ("Type", "@type"),
            ("Shape", "@shape_str"),
            ("Likelihood", "@likelihood"),
            ("Residual", "@residual"),
            ("Missing", "@missing")]
    
        edge_plot = hvnx.draw_networkx_edges(
            P, pos,edge_color="kind" if show_edge_kinds else "black",
            alpha=0.6,arrow_style="-|>", arrow_size=5, edge_width=4 + hv.dim("confidence").norm() * 15,
            cmap=edge_color_map ).opts(tools=["hover"], hover_tooltips=edge_tooltips)
    
        node_plot = hvnx.draw_networkx_nodes(
            P,  pos, node_color="type", node_marker="marker",
            alpha=1.0, node_size=5200, cmap=color_map,
        ).opts(tools=["hover"], hover_tooltips=node_tooltips)
    
        label_df = pd.DataFrame(  {"x": [float(pos[n][0]) for n in P.nodes()],
                "y": [float(pos[n][1]) for n in P.nodes()],
                "text": [f"{n}\n{P.nodes[n]['shape_str']}" for n in P.nodes()]})
        labels = hv.Labels(label_df, kdims=["x", "y"], vdims=["text"]).opts(
                            text_color="black", text_font_size="8pt", yoffset=0.0,
                            text_align="center",  text_baseline="middle" )
    
        edge_param_labels = None
        if show_edge_param_labels and P.number_of_edges() > 0:
            rows = []
            for u, v, ed in P.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                rows.append(
                    {"x": float((x0 + x1) / 2.0),"y": float((y0 + y1) / 2.0),
                    "text": f"n_params: {int(ed['n_params'])}\nvariance: {round(float(ed['variance']), 3)}" } )
            edge_param_labels = hv.Labels(pd.DataFrame(rows), kdims=["x", "y"], vdims=["text"]).opts(
                text_font_size="9pt", text_color="black",
                text_align="left", text_baseline="middle", )
    
        graph = edge_plot * node_plot * labels
        if edge_param_labels is not None: graph = graph * edge_param_labels
    
        return graph.opts( title="GraphView", width=900, height=650, xaxis=None,
                           yaxis=None, tools=["tap", "box_select", "hover"] )

    # -----------------------------
    # build_model: engines & missing resolution
    # -----------------------------
    def _simple_grm_to_traits_spec(self, traits_node: str | None = None) -> dict[str, Any] | None:
        """Identify whether the graph can use the collapsed GRM-to-traits likelihood.

        Parameters
        ----------
        traits_node : str | None
            Input for the operation ``self.G.nodes[traits_node]``. See
            ``vigor.causal.CausalGraph._simple_grm_to_traits_spec`` and its caller for the
            scientific representation; the shown operation constrains the required type/shape.
            Default: ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.
        """
        endos = [n for n, a in self.G.nodes(data=True) if a.get("type") == "endo"]
        if traits_node is None:
            if len(endos) != 1:
                return None
            traits_node = endos[0]
        if traits_node not in self.G.nodes or self.G.nodes[traits_node].get("type") != "endo":
            return None
        # no latents
        if any(a.get("type") == "latent" for _, a in self.G.nodes(data=True)):
            return None
        # traits is leaf
        if len(list(self.G.successors(traits_node))) != 0:
            return None
        parents = list(self.G.predecessors(traits_node))
        grm_parents = [p for p in parents if self.G.nodes[p].get("type") == "grm"]
        if len(grm_parents) != 1:
            return None
        g = grm_parents[0]
        e = self.G.edges[g, traits_node]
        if str(e.get("kind", "dense")).lower() != "grm":
            return None
        for p in parents:
            if p == g:
                continue
            if self.G.nodes[p].get("type") != "exo":
                return None
            if str(self.G.edges[p, traits_node].get("kind", "dense")).lower() != "dense":
                return None
        return {"traits": traits_node, "grm": g, "parents": parents}

    @staticmethod
    def _pattern_groups(Y: np.ndarray) -> tuple[int, dict[bytes, np.ndarray]]:
        # returns (#patterns, {pattern_bytes: row_idx})
        """Group animals by their observed-trait missingness pattern.

        Parameters
        ----------
        Y : np.ndarray
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.
        """
        miss = np.isnan(Y)
        packed = _pack_pattern_mask(miss)
        # unique rows
        uniq, inv = np.unique(packed, axis=0, return_inverse=True)
        groups: dict[bytes, list[int]] = {}
        for i, g in enumerate(inv):
            key = uniq[g].tobytes()
            groups.setdefault(key, []).append(i)
        out = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        return len(out), out

    def _resolve_missing_mode(self, node: str, Y: np.ndarray) -> str:
        """Resolve missing mode.

        Parameters
        ----------
        node : str
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        Y : np.ndarray
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.
        """
        a = self.G.nodes[node]
        residual = str(a.get("residual","diag")).lower()
        mode = str(a.get("missing","auto")).lower()
        if not np.isnan(Y).any():
            return "none"
        if mode != "auto":
            return mode
        # auto
        if residual in ("diag","factor"):
            return "mask"
        if residual in ("lkj","fa"):
            t = Y.shape[1]
            if t <= int(a.get("pattern_max_t", 25)):
                npatt, _ = self._pattern_groups(Y)
                if npatt <= int(a.get("pattern_max_groups", 2048)):
                    return "pattern"
            raise ValueError(
                f"Node '{node}' has residual='{residual}' with elementwise missingness. "
                "Auto mode refuses because pattern MVN would explode. "
                "Use residual='factor' (recommended) or residual='diag', or set missing='impute' for small problems."
            )
        return "mask"

    # -----------------------------
    # Core: generic build_model
    # -----------------------------
    def build_model(
        self,
        *,
        engine: Literal["auto", "generic", "marginal_evd"] = "auto",
        sd_prior_sigma: float = 0.5,
        jitter: float = 1e-6,
        poisson_clip: float = 20.0,
        pattern_jitter: float = 1e-6,
    ) -> pm.Model:
        """engine='auto':
          - if graph matches simple GRM->traits and traits missing in {'none','row'} and traits residual in {'lkj','fa','diag','factor'},
            then use build_model_grm_traits_marginal_evd (collapsed) when possible.
          - otherwise uses generic compiler with missing auto-resolution.

        engine='generic' forces generic compiler.
        engine='marginal_evd' forces collapsed compiler (requires eligibility).

        Parameters
        ----------
        engine : Literal['auto', 'generic', 'marginal_evd']
            Execution/backend selection for this operation. See the source-derived literal
            descriptions below and the called resolver; automatic selection uses the available
            implementation and input representation. Default: ``'auto'``. Source-recognized literal
            values: ``generic``, ``marginal_evd``. These select the corresponding branches below;
            comparison evidence alone does not establish an exhaustive enum.

        sd_prior_sigma : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``0.5``.
            Passed to ``self.build_model_grm_traits_marginal_evd`` as ``sd_prior_sigma``.

        jitter : float
            Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
            denominators or nonpositive factor variances. Choose a positive value on the scale of
            the input quantities. Default: ``1e-06``. Passed to
            ``self.build_model_grm_traits_marginal_evd`` as ``jitter``.

        poisson_clip : float
            Input for the operation ``pt.exp(pt.clip(mu, -float(poisson_clip),
            float(poisson_clip)))``. See ``vigor.causal.CausalGraph.build_model`` and its caller for
            the scientific representation; the shown operation constrains the required type/shape.
            Default: ``20.0``.

        pattern_jitter : float
            Compatibility/callback input accepted but not consumed by this body. Any object is
            locally accepted; see the calling protocol for the supplied type. Default: ``1e-06``.

        Returns
        -------
        object
            Computed result from ``model``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if engine not in ("auto","generic","marginal_evd"):
            raise ValueError("engine must be 'auto'|'generic'|'marginal_evd'")

        # auto fast path
        if engine in ("auto","marginal_evd"):
            spec = self._simple_grm_to_traits_spec()
            if spec is not None:
                tnode = spec["traits"]
                a = self.G.nodes[tnode]
                Y = np.asarray(a["data"], dtype=float)
                miss_mode = str(a.get("missing","auto")).lower()
                if miss_mode == "auto":
                    miss_mode = self._resolve_missing_mode(tnode, Y)
                # collapsed only if no elementwise missing (none) OR user explicitly wants row-complete
                if miss_mode in ("none","row") and str(a.get("likelihood","normal")).lower() == "normal":
                    if engine == "marginal_evd" or (engine == "auto"):
                        model = self.build_model_grm_traits_marginal_evd(
                            target=tnode,
                            row_complete=(miss_mode=="row"),
                            jitter=float(jitter),
                            sd_prior_sigma=float(sd_prior_sigma),
                        )
                        self._last_build = _ModelBuildInfo(engine="marginal_evd", obs_ids=self.obs_ids, trait_nodes=[tnode])
                        return model
            if engine == "marginal_evd":
                raise ValueError("engine='marginal_evd' requested but graph is not eligible.")

        # generic compiler
        order = list(nx.topological_sort(self.G))
        coords: dict[str, list[str]] = {"obs_id": self._as_str_coords(self.obs_ids)}
        for n, a in self.G.nodes(data=True):
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        def _sd_dist(scale: float, shape=None):
            """Return the sd dist result computed by ``pm.LogNormal.dist``.

            Parameters
            ----------
            scale : float
                Numerical scale/retention/step threshold controlling the transformation described above.
                Match vector scales to the affected axis; the exact formula and local comparisons below
                determine its admissible domain. Passed to ``pm.LogNormal.dist`` as ``mu``.

            shape : tuple of int or None
                Requested array/distribution extents in axis order. Nonnegative sizes must agree with
                the associated data; None uses the backend's inferred scalar/default shape. Default:
                ``None``. Passed to ``pm.LogNormal.dist`` as ``shape``.
            """
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        with pm.Model(coords=coords) as model:
            model_vars: dict[str, pt.TensorVariable] = {}
            # pm.Data for exo/grm
            for node in order:
                a = self.G.nodes[node]
                if a["type"] in ("exo","grm", "adj"):
                    model_vars[node] = pm.Data(node, a["data"], dims=("obs_id", a["dim_name"]))

            # compile latents/endos
            for node in order:
                a = self.G.nodes[node]
                if a["type"] in ("exo","grm", "adj"): continue

                n_dim = int(a["shape"][1])
                dim_name = a["dim_name"]
                parents = list(self.G.predecessors(node))

                # variance budget weights (optional; applied as sqrt(w) to each parent's contribution)
                w_par = None
                if bool(a.get("variance_budget", False)) and len(parents) > 1:
                    conc = float(a.get("budget_concentration", 20.0))
                    eps = float(a.get("budget_w_eps", 1e-6))
                    w_raw = pm.Dirichlet(f"{node}:w_parent_raw", a=np.ones(len(parents)) * conc)
                    w_par = pm.Deterministic(f"{node}:w_parent", w_raw * (1.0 - eps * len(parents)) + eps)
                # linear predictor excluding residual factors
                mu = pt.zeros((len(self.obs_ids), n_dim))

                # keep track of parameter-implied genetic variances for this node
                V_G_param_total = None

                for i, parent in enumerate(parents):
                    pa = self.G.nodes[parent]
                    e = self.G.edges[parent, node]
                    kind = str(e.get("kind","dense")).lower()
                    Xp = model_vars[parent]
                    scale_w = pt.sqrt(w_par[i]) if (w_par is not None) else 1.0

                    if kind == "dense":
                        beta = pm.Normal(
                            f"beta:{parent}->{node}",
                            mu=0.0,
                            sigma=float(e.get("prior_sigma", 1.0)),
                            dims=(pa["dim_name"], dim_name),
                        )
                        contrib = pt.dot(Xp, beta) * scale_w

                    elif kind == "bym2":
                        W = pa.get("W", None)
                        if W is None:
                            raise ValueError(f"BYM2 edge needs parent '{parent}' node to have W adjacency.")
                        trait_cov = str(e.get("grm_trait_cov", "diag")).lower()
                        sdG = float(e.get("prior_sigma", 1.0))
                    
                        # build L (same as your GRM logic)
                        if trait_cov == "diag":
                            sd_vec = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(dim_name,),
                            )
                            L = pt.diag(sd_vec)
                        elif trait_cov == "scalar":
                            sd_s = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                            )
                            L = sd_s * pt.eye(n_dim)
                        else:
                            chol_packed = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(e.get("eta", 2.0)),
                                sd_dist=_sd_dist(sdG, shape=n_dim),
                                compute_corr=False,
                            )
                            L = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)
                    
                        w_scale = w_par[i] if (w_par is not None) else 1.0
                        L_eff = L * pt.sqrt(w_scale)
                    
                        # BYM2 mixing
                        rho = pm.Beta(f"rho:{parent}->{node}", alpha=1.0, beta=1.0)
                        sigma_bym = pm.HalfNormal(f"sigma_bym:{parent}->{node}", sigma=1.0)
                    
                        cols = []
                        for j in range(n_dim):
                            u = pm.ICAR(
                                f"ICAR_raw:{parent}->{node}:{j}",
                                W=W,
                                sigma=1.0,
                                zero_sum_stdev=float(e.get("icar_zero_sum_stdev", 0.001)),
                                dims=("obs_id",),
                            )
                            v = pm.Normal(f"IID_raw:{parent}->{node}:{j}", mu=0.0, sigma=1.0, dims=("obs_id",))
                            phi = sigma_bym * (pt.sqrt(rho) * u + pt.sqrt(1.0 - rho) * v)
                            cols.append(phi)
                    
                        Phi = pt.stack(cols, axis=1)           # (N, n_dim)
                        contrib = pt.dot(Phi, L_eff.T)         # (N, n_dim)

                    elif kind == "factor":
                        # latent source must exist
                        k = int(pa["shape"][1])
                        p = int(n_dim)
                        if p < k:
                            raise ValueError(f"factor edge requires target dim >= source dim: {parent}->{node}")
                        # identifiable loadings
                        # build mask for first k rows lower-triangular
                        mask = np.ones((p, k), dtype=float)
                        for r in range(min(p, k)):
                            if r + 1 < k:
                                mask[r, (r+1):] = 0.0
                        mask_t = pt.constant(mask)
                        Lambda_raw = pm.Normal(
                            f"Lambda_raw:{parent}->{node}",
                            mu=0.0,
                            sigma=float(e.get("prior_sigma", 1.0)),
                            dims=(dim_name, pa["dim_name"]),  # (traits_dim, factors_dim)
                        )
                        Lambda = Lambda_raw * mask_t
                        if bool(e.get("factor_identifiable", True)):
                            diag = pm.LogNormal(
                                f"Lambda_diag:{parent}->{node}",
                                mu=np.log(max(float(e.get("prior_sigma", 1.0)), 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(pa["dim_name"],),
                            )
                            idx = np.arange(k)
                            Lambda = pt.set_subtensor(Lambda[idx, idx], diag)
                        pm.Deterministic(f"Lambda:{parent}->{node}", Lambda, dims=(dim_name, pa["dim_name"]))
                        contrib = pt.dot(Xp, Lambda.T) * scale_w

                    elif kind == "grm":
                        # PC-space effects
                        r_dim = int(pa["shape"][1])
                        z_prior = str(e.get("z_prior", "iid")).lower()
                        Z_raw = pm.Normal(
                            f"Z_raw:{parent}->{node}",
                            mu=0.0,
                            sigma=1.0,
                            dims=(pa["dim_name"], dim_name),)
                        if z_prior == "iid": Z = Z_raw
                        elif z_prior == "ard_pc":
                            tau = pm.LogNormal(
                                f"tau_pc:{parent}->{node}",
                                mu=np.log(max(float(e.get("prior_sigma", 1.0)), 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(pa["dim_name"],), )
                            Z = Z_raw * tau[:, None]
                        elif z_prior == "rhs":
                            # regularized horseshoe (per PC)
                            # global shrinkage tau, local lambdas
                            tau = pm.HalfCauchy(f"rhs_tau:{parent}->{node}", beta=float(e.get("rhs_global_scale", 0.3)))
                            lam = pm.HalfCauchy(f"rhs_lam:{parent}->{node}", beta=1.0, dims=(pa["dim_name"],))
                            c2 = pm.InverseGamma(f"rhs_c2:{parent}->{node}", alpha=1.0, beta=1.0)
                            lam_tilde = pt.sqrt(c2) * lam / pt.sqrt(c2 + (tau**2) * (lam**2))
                            Z = Z_raw * (tau * lam_tilde)[:, None]
                        else:
                            raise ValueError("z_prior must be iid|ard_pc|rhs")

                        trait_cov = str(e.get("grm_trait_cov","lkj")).lower()
                        sdG = float(e.get("prior_sigma", 1.0))

                        if trait_cov == "lkj":
                            chol_packed = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(e.get("eta", 2.0)),
                                sd_dist=_sd_dist(sdG, shape=n_dim),
                                compute_corr=False,
                            )
                            L = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)
                        elif trait_cov == "diag":
                            sd_vec = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(dim_name,),
                            )
                            L = pt.diag(sd_vec)
                        elif trait_cov == "scalar":
                            sd_s = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                            )
                            L = sd_s * pt.eye(n_dim)
                        else:
                            raise ValueError("grm_trait_cov must be lkj|diag|scalar")

                        # apply variance budget scaling consistently to reported SigmaG
                        model.add_coords({dim_name + "_t": coords[dim_name]})
                        w_scale = w_par[i] if (w_par is not None) else 1.0
                        L_eff = L * pt.sqrt(w_scale)
                        pm.Deterministic(f"L:{parent}->{node}", L_eff, dims=(dim_name, dim_name+'_t'))
                        SigmaG = pt.dot(L_eff, L_eff.T)
                        pm.Deterministic(f"SigmaG:{parent}->{node}", SigmaG, dims=(dim_name, dim_name+'_t'))
                        RG = _corr_from_cov(SigmaG)
                        pm.Deterministic(f"RG:{parent}->{node}", RG, dims=(dim_name, dim_name+'_t'))

                        # parameter-implied per-trait genetic variance adjusted by mean diag(K)
                        kdiag_mean = float(pa.get("kdiag_mean", 1.0))
                        Vg_param = pt.diag(SigmaG) * float(kdiag_mean)
                        pm.Deterministic(f"V_G_param:{parent}->{node}", Vg_param, dims=(dim_name,))
                        V_G_param_total = Vg_param if (V_G_param_total is None) else (V_G_param_total + Vg_param)

                        contrib = pt.dot(pt.dot(Xp, Z), L_eff.T)  # already includes w_scale via L_eff

                    else:
                        raise ValueError(f"Unknown edge kind '{kind}' for {parent}->{node}")

                    if bool(a.get("track_values_per_path", False)):
                        pm.Deterministic(f"prediction:{parent}->{node}", contrib, dims=("obs_id", dim_name))
                    if bool(e.get("track_variance", False)):
                        pm.Deterministic(f"V_path_realized:{parent}->{node}", pt.var(contrib, axis=0), dims=(dim_name,))
                    mu = mu + contrib

                # store predictor mean
                if bool(a.get("track_values", False)):
                    pm.Deterministic(f"prediction:{node}", mu, dims=("obs_id", dim_name))
                    pm.Deterministic(f"V_pred:{node}", pt.var(mu, axis=0), dims=(dim_name,))

                # latent node
                if a["type"] == "latent":
                    sigma_lat = float(a.get("prior_sigma", 1.0))
                    offset = pm.Normal(f"{node}:offset", mu=0.0, sigma=sigma_lat, dims=("obs_id", dim_name))
                    lat = pm.Deterministic(node, mu + offset, dims=("obs_id", dim_name))
                    model_vars[node] = lat
                    continue

                # endogenous node likelihood
                if a["type"] != "endo":
                    raise ValueError(f"Unknown node type {a['type']}")

                lik = str(a.get("likelihood","normal")).lower()
                residual = str(a.get("residual","diag")).lower()
                Y = np.asarray(a["data"], dtype=float)

                # missing resolution
                miss_mode = self._resolve_missing_mode(node, Y)

                # ---- NORMAL ----
                if lik == "normal":
                    sdE = float(a.get("sd_E", 1.0))

                    # residual='factor': generative low-rank residual covariance
                    Efac = None
                    if residual == "factor":
                        k = int(a.get("n_factors_E") or min(3, n_dim))
                        k = max(1, min(k, n_dim))
                        # F: per-individual factor scores
                        model.add_coords({f"{node}_Efac": [f"F{i}" for i in range(1, k+1)]})
                        # add coords for factor dim
                        F = pm.Normal(f"{node}:Efac_scores", mu=0.0, sigma=1.0, dims=("obs_id", f"{node}_Efac"))
                        # Loadings W: identifiable lower-triangular (first k rows) + positive diagonal
                        W_raw = pm.Normal(f"{node}:Efac_loadings_raw", mu=0.0, sigma=1.0, shape=(n_dim, k))
                        # zero upper triangle in first k rows
                        W = W_raw
                        for r in range(k):
                            for c in range(r+1, k):
                                W = pt.set_subtensor(W[r, c], 0.0)
                        # positive diagonal
                        diag = pm.LogNormal(f"{node}:Efac_diag", mu=np.log(max(sdE, 1e-8)), sigma=float(sd_prior_sigma), shape=(k,))
                        idx = np.arange(k)
                        W = pt.set_subtensor(W[idx, idx], diag)
                        pm.Deterministic(f"{node}:Efac_loadings", W)
                        Efac = pt.dot(F, W.T)  # (N x n_dim)
                        pm.Deterministic(f"{node}:Efac", Efac, dims=("obs_id", dim_name))

                    # residual scale
                    sigma = pm.LogNormal(
                        f"{node}:sigma",
                        mu=np.log(max(sdE, 1e-8)),
                        sigma=float(sd_prior_sigma),
                        dims=(dim_name,) if n_dim > 1 else None,
                    )
                    if n_dim == 1:
                        sigma = sigma if isinstance(sigma, pt.TensorVariable) else sigma
                        mu1 = (mu[:, 0] + (Efac[:, 0] if Efac is not None else 0.0))
                        y1 = Y[:, 0]
                        if miss_mode in ("mask","pattern"):
                            obs = ~np.isnan(y1)
                            idx = np.where(obs)[0].astype(np.int64)
                            pm.Normal(f"{node}:obs", mu=mu1[idx], sigma=sigma, observed=y1[idx])
                            model_vars[node] = mu1
                        elif miss_mode == "impute":
                            pm.Normal(node, mu=mu1, sigma=sigma, observed=np.ma.masked_invalid(y1), dims=("obs_id",))
                            model_vars[node] = mu1
                        elif miss_mode == "none":
                            pm.Normal(node, mu=mu1, sigma=sigma, observed=y1, dims=("obs_id",))
                            model_vars[node] = mu1
                        else:
                            raise ValueError(f"Unsupported missing mode '{miss_mode}' for univariate normal.")
                    else:
                        mean_total = mu + (Efac if Efac is not None else 0.0)
                        # We cannot reference W outside; rebuild diag if needed:
                        if Efac is None:
                            pm.Deterministic(f"V_E_param:{node}", sigma**2, dims=(dim_name,))
                        else:
                            # loadings W stored as Deterministic, read it back
                            W_det = model[f"{node}:Efac_loadings"]
                            pm.Deterministic(f"V_E_param:{node}", pt.sum(W_det**2, axis=1) + sigma**2, dims=(dim_name,))

                        if V_G_param_total is not None:
                            pm.Deterministic(f"V_G_param_total:{node}", V_G_param_total, dims=(dim_name,))
                            pm.Deterministic(f"h2_param:{node}", V_G_param_total / (V_G_param_total + model[f"V_E_param:{node}"]), dims=(dim_name,))
                        pm.Deterministic(f"V_total_param:{node}", (V_G_param_total if V_G_param_total is not None else 0.0) + model[f"V_E_param:{node}"], dims=(dim_name,))

                        # likelihood
                        if miss_mode == "none":
                            pm.Normal(node, mu=mean_total, sigma=sigma, observed=Y, dims=("obs_id", dim_name))
                            model_vars[node] = mean_total
                        elif miss_mode == "impute":
                            pm.Normal(node, mu=mean_total, sigma=sigma, observed=np.ma.masked_invalid(Y), dims=("obs_id", dim_name))
                            model_vars[node] = mean_total
                        elif miss_mode == "mask":
                            obs = ~np.isnan(Y)
                            rows, cols = np.where(obs)
                            idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                            y_obs = Y[rows, cols].astype(float)
                            mu_flat = mean_total.reshape((-1,))
                            idx_t = pt.constant(idx_flat)
                            mu_obs = mu_flat[idx_t]
                            sigma_obs = sigma[pt.constant(cols.astype(np.int64))]
                            pm.Normal(f"{node}:obs", mu=mu_obs, sigma=sigma_obs, observed=y_obs, shape=y_obs.shape[0])
                            model_vars[node] = mean_total
                        elif miss_mode == "pattern":
                            # for diag residual we still prefer mask (pattern is wasted)
                            obs = ~np.isnan(Y)
                            rows, cols = np.where(obs)
                            idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                            y_obs = Y[rows, cols].astype(float)
                            mu_flat = mean_total.reshape((-1,))
                            mu_obs = mu_flat[pt.constant(idx_flat)]
                            sigma_obs = sigma[pt.constant(cols.astype(np.int64))]
                            pm.Normal(f"{node}:obs", mu=mu_obs, sigma=sigma_obs, observed=y_obs, shape=y_obs.shape[0])
                            model_vars[node] = mean_total
                        else:
                            raise ValueError(f"Unsupported missing mode '{miss_mode}' for normal/diag|factor.")
                    # Explained variance (realized)
                    if bool(a.get("track_values", False)):
                        V_pred = model[f"V_pred:{node}"]
                        V_total_real = V_pred + model[f"V_E_param:{node}"]
                        pm.Deterministic(f"V_total_real:{node}", V_total_real, dims=(dim_name,))
                        for parent in parents:
                            key = f"V_path_realized:{parent}->{node}"
                            if key in model.named_vars:
                                pm.Deterministic(f"EV_path:{parent}->{node}", model[key] / V_total_real, dims=(dim_name,))
                    continue

                # ---- POISSON ----
                if lik == "poisson":
                    rate = pt.exp(pt.clip(mu, -float(poisson_clip), float(poisson_clip))) if str(a.get("link","log")).lower() == "log" else pt.maximum(mu, 1e-12)
                    y = np.asarray(a["data"], dtype=float)
                    if np.isnan(y).any() and miss_mode in ("mask","pattern","row"):
                        obs = ~np.isnan(y)
                        rows, cols = np.where(obs)
                        y_obs = y[rows, cols].astype(int)
                        idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                        rate_flat = rate.reshape((-1,))
                        pm.Poisson(f"{node}:obs", mu=rate_flat[pt.constant(idx_flat)], observed=y_obs, shape=y_obs.shape[0])
                        model_vars[node] = rate
                    else:
                        pm.Poisson(node, mu=rate, observed=y.astype(int), dims=("obs_id", dim_name))
                        model_vars[node] = rate
                    continue

                # ---- BERNOULLI ----
                if lik == "bernoulli":
                    if str(a.get("link","logit")).lower() == "logit":
                        p = pm.math.sigmoid(mu)
                    else:
                        p = pt.clip(mu, 0.0, 1.0)
                    y = np.asarray(a["data"], dtype=float)
                    if np.isnan(y).any() and miss_mode in ("mask","pattern","row"):
                        obs = ~np.isnan(y)
                        rows, cols = np.where(obs)
                        y_obs = y[rows, cols].astype(int)
                        idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                        p_flat = p.reshape((-1,))
                        pm.Bernoulli(f"{node}:obs", p=p_flat[pt.constant(idx_flat)], observed=y_obs, shape=y_obs.shape[0])
                        model_vars[node] = p
                    else:
                        pm.Bernoulli(node, p=p, observed=y.astype(int), dims=("obs_id", dim_name))
                        model_vars[node] = p
                    continue

                raise ValueError(f"Unsupported likelihood '{lik}' for node '{node}'")

        trait_nodes = [n for n,a in self.G.nodes(data=True) if a.get("type")=="endo"]
        self._last_build = _ModelBuildInfo(engine="generic", obs_ids=self.obs_ids, trait_nodes=trait_nodes)
        return model

    # -----------------------------
    # Collapsed marginal-eVD compiler (GCTA/MPH correctness for the simple case)
    # -----------------------------
    def build_model_grm_traits_marginal_evd(
        self,
        *,
        target: str = "traits",
        trait_cov_G: Literal["lkj","diag","scalar","fa"] = "lkj",
        residual: Literal["diag","lkj","fa","scalar"] = "diag",
        n_factors_G: int | None = None,
        n_factors_E: int | None = None,
        eta_G: float = 2.0,
        eta_E: float = 4.0,
        sd_G: float = 1.0,
        sd_E: float = 1.0,
        sd_prior_sigma: float = 0.5,
        jitter: float = 1e-6,
        row_complete: bool = False,
    ) -> pm.Model:
        """Collapsed likelihood for simple GRM->traits LMM using eigen-rotation:
          K = U diag(d) U^T
          y*_i ~ MVN(mu*_i, d_i * Sigma_G + Sigma_E)

        This integrates out individual random effects and matches the intended GCTA/MPH LMM math.
        Supports row_complete=True to drop any rows with missing values (approx).

        Parameters
        ----------
        target : str
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering. Default: ``'traits'``.

        trait_cov_G : Literal['lkj', 'diag', 'scalar', 'fa']
            Genetic trait covariance: 'lkj' uses a correlated trait prior, 'diag' independent trait
            scales, 'scalar' one shared scale and 'fa' a factor covariance only where the builder
            supports it. See the local validator for the accepted subset. Default: ``'lkj'``.

        residual : Literal['diag', 'lkj', 'fa', 'scalar']
            Residual covariance model: 'diag' uses independent trait noise, 'lkj' uses an LKJ
            correlation prior, and 'fa'/'factor' uses low-rank factors plus specific variances.
            Check add_trait/build for scalar-model restrictions. Default: ``'diag'``.

        n_factors_G : int | None
            Latent/trait factor dimension. Positive integer sizes specify model capacity; None uses
            the graph builder's automatic factor count where supported. Default: ``None``.

        n_factors_E : int | None
            Latent/trait factor dimension. Positive integer sizes specify model capacity; None uses
            the graph builder's automatic factor count where supported. Default: ``None``.

        eta_G : float
            Positive LKJ correlation-prior concentration. Values above one favor weaker
            correlations; one gives the LKJ uniform-correlation prior. Default: ``2.0``. Passed to
            ``pm.LKJCholeskyCov`` as ``eta``.

        eta_E : float
            Positive LKJ correlation-prior concentration. Values above one favor weaker
            correlations; one gives the LKJ uniform-correlation prior. Default: ``4.0``. Passed to
            ``pm.LKJCholeskyCov`` as ``eta``.

        sd_G : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``1.0``.
            Passed to ``pm.LKJCholeskyCov`` as ``sd_dist``.

        sd_E : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``1.0``.
            Passed to ``pm.LogNormal`` as ``mu``.

        sd_prior_sigma : float
            Variance/scale or trait covariance specified for the selected prior/simulation
            component. Scalar scales must be nonnegative (positive where a prior requires it);
            matrices match the trait axis and must be positive semidefinite. Default: ``0.5``.
            Passed to ``pm.LogNormal`` as ``sigma``.

        jitter : float
            Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
            denominators or nonpositive factor variances. Choose a positive value on the scale of
            the input quantities. Default: ``1e-06``.

        row_complete : bool
            Whether to enforce/use complete observations or include the named
            decomposition/component output. The enabled branch below defines the returned
            representation; disabling skips that optional step. Default: ``False``.

        Returns
        -------
        object
            Computed result from ``model``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if target not in self.G:
            raise ValueError(f"target '{target}' not found.")
        targ = self.G.nodes[target]
        if targ.get("type") != "endo" or str(targ.get("likelihood","normal")).lower() != "normal":
            raise ValueError("marginal_evd requires a Normal endo target.")
        if len(list(self.G.successors(target))) != 0:
            raise ValueError("marginal_evd requires target to be terminal (leaf).")
        Y_full = np.asarray(targ["data"], dtype=float)
        if row_complete:
            keep = ~np.isnan(Y_full).any(axis=1)
            if keep.sum() < 5:
                raise ValueError("Too few complete rows after row_complete filtering.")
        else:
            if np.isnan(Y_full).any():
                raise ValueError("marginal_evd requires no missing values unless row_complete=True.")
            keep = np.ones(Y_full.shape[0], dtype=bool)

        parents = list(self.G.predecessors(target))
        grm_parents = [p for p in parents if self.G.nodes[p].get("type") == "grm"]
        exo_parents = [p for p in parents if self.G.nodes[p].get("type") == "exo"]
        if len(grm_parents) != 1:
            raise ValueError("marginal_evd fast path supports exactly one GRM parent.")
        g = grm_parents[0]
        Kg = self.G.nodes[g].get("K_full")
        if Kg is None:
            raise ValueError(f"GRM node '{g}' must be added with store_full=True.")
        K = np.asarray(Kg, dtype=float)
        K = K[np.ix_(keep, keep)]
        Y = Y_full[keep]
        obs_ids = [oid for oid, k in zip(self.obs_ids, keep) if k]

        # EVD
        K = 0.5*(K+K.T)
        d, U = np.linalg.eigh(K)
        order = np.argsort(d)[::-1]
        d = np.clip(d[order], 0.0, np.inf)
        U = U[:, order]
        U_T = U.T.astype(float)
        d_vec = d.astype(float)

        n, t = Y.shape
        dim_name = targ["dim_name"]

        coords = {"obs_id": self._as_str_coords(obs_ids), dim_name: self._as_str_coords(targ["coords"])}

        def _sd_dist(scale: float, shape=None):
            """Return the sd dist result computed by ``pm.LogNormal.dist``.

            Parameters
            ----------
            scale : float
                Numerical scale/retention/step threshold controlling the transformation described above.
                Match vector scales to the affected axis; the exact formula and local comparisons below
                determine its admissible domain. Passed to ``pm.LogNormal.dist`` as ``mu``.

            shape : tuple of int or None
                Requested array/distribution extents in axis order. Nonnegative sizes must agree with
                the associated data; None uses the backend's inferred scalar/default shape. Default:
                ``None``. Passed to ``pm.LogNormal.dist`` as ``shape``.
            """
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        # exo data subset
        X_exo = {}
        for p in exo_parents:
            a = self.G.nodes[p]
            X_exo[p] = np.asarray(a["data"], dtype=float)[keep]
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        with pm.Model(coords=coords) as model:
            Xp = {p: pm.Data(p, X_exo[p], dims=("obs_id", self.G.nodes[p]["dim_name"])) for p in exo_parents}
            mu = pt.zeros((n, t))
            for p in exo_parents:
                e = self.G.edges[p, target]
                beta = pm.Normal(
                    f"beta:{p}->{target}",
                    mu=0.0,
                    sigma=float(e.get("prior_sigma", 1.0)),
                    dims=(self.G.nodes[p]["dim_name"], dim_name),
                )
                mu = mu + pt.dot(Xp[p], beta)
            pm.Deterministic(f"prediction:{target}", mu, dims=("obs_id", dim_name))

            # Sigma_G
            trait_cov_G = str(trait_cov_G).lower()
            if trait_cov_G == "lkj":
                chol_packed = pm.LKJCholeskyCov(
                    f"cholG:{g}->{target}",
                    n=t,
                    eta=float(eta_G),
                    sd_dist=_sd_dist(sd_G, shape=t),
                    compute_corr=False,
                )
                L_G = pm.expand_packed_triangular(t, chol_packed, lower=True)
            elif trait_cov_G == "diag":
                sdv = pm.LogNormal(f"sdG:{g}->{target}", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=t)
                L_G = pt.diag(sdv)
            elif trait_cov_G == "scalar":
                s = pm.LogNormal(f"sdG:{g}->{target}", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma))
                L_G = s * pt.eye(t)
            elif trait_cov_G == "fa":
                q = int(n_factors_G or min(3, t))
                # simple identifiable FA chol
                W_raw = pm.Normal("GFA_W_raw", 0.0, 1.0, shape=(t, q))
                W = W_raw
                for r in range(min(t, q)):
                    for c in range(r+1, q):
                        W = pt.set_subtensor(W[r, c], 0.0)
                diag = pm.LogNormal("GFA_diag", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=(q,))
                idx = np.arange(q)
                W = pt.set_subtensor(W[idx, idx], diag)
                psi = pm.LogNormal("GFA_psi", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=(t,))
                SigmaG = pt.dot(W, W.T) + pt.diag(psi) + float(jitter)*pt.eye(t)
                L_G = pt.linalg.cholesky(SigmaG)
            else:
                raise ValueError("trait_cov_G must be lkj|diag|scalar|fa")

            L_G_eff = L_G
            pm.Deterministic(f"L:{g}->{target}", L_G_eff, dims=(dim_name, dim_name))
            SigmaG = pt.dot(L_G_eff, L_G_eff.T)
            pm.Deterministic(f"SigmaG:{g}->{target}", SigmaG, dims=(dim_name, dim_name))
            pm.Deterministic(f"RG:{g}->{target}", _corr_from_cov(SigmaG), dims=(dim_name, dim_name))

            kdiag_mean = float(self.G.nodes[g].get("kdiag_mean", 1.0))
            pm.Deterministic(f"V_G_param:{g}->{target}", pt.diag(SigmaG)*kdiag_mean, dims=(dim_name,))

            # Sigma_E
            residual = str(residual).lower()
            if residual == "diag":
                sdv = pm.LogNormal(f"{target}:sdE", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=t)
                SigmaE = pt.diag(sdv**2)
            elif residual == "scalar":
                s = pm.LogNormal(f"{target}:sdE", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma))
                SigmaE = (s**2)*pt.eye(t)
            elif residual == "lkj":
                chol_packed = pm.LKJCholeskyCov(
                    f"{target}:cholE",
                    n=t,
                    eta=float(eta_E),
                    sd_dist=_sd_dist(sd_E, shape=t),
                    compute_corr=False,
                )
                L_E = pm.expand_packed_triangular(t, chol_packed, lower=True)
                SigmaE = pt.dot(L_E, L_E.T)
            elif residual == "fa":
                q = int(n_factors_E or min(3, t))
                W_raw = pm.Normal("EFA_W_raw", 0.0, 1.0, shape=(t, q))
                W = W_raw
                for r in range(min(t, q)):
                    for c in range(r+1, q):
                        W = pt.set_subtensor(W[r, c], 0.0)
                diag = pm.LogNormal("EFA_diag", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=(q,))
                idx = np.arange(q)
                W = pt.set_subtensor(W[idx, idx], diag)
                psi = pm.LogNormal("EFA_psi", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=(t,))
                SigmaE = pt.dot(W, W.T) + pt.diag(psi) + float(jitter)*pt.eye(t)
            else:
                raise ValueError("residual must be diag|scalar|lkj|fa")

            pm.Deterministic(f"SigmaE:{target}", SigmaE, dims=(dim_name, dim_name))
            pm.Deterministic(f"V_E_param:{target}", pt.diag(SigmaE), dims=(dim_name,))
            pm.Deterministic(f"h2_param:{target}", pt.diag(SigmaG)*kdiag_mean/(pt.diag(SigmaG)*kdiag_mean + pt.diag(SigmaE)), dims=(dim_name,))

            # rotated residuals
            Uc = pt.constant(U_T)
            Yc = pt.constant(Y)
            resid_mat = Yc - mu
            resid_rot = pt.dot(Uc, resid_mat)  # (n x t)

            covs = pt.constant(d_vec)[:, None, None] * SigmaG[None, :, :] + SigmaE[None, :, :] + float(jitter) * pt.eye(t)[None, :, :]
            Ls = pt.linalg.cholesky(covs)
            z = pt.linalg.solve_triangular(Ls, resid_rot[:, :, None], lower=True)[:, :, 0]
            quad = pt.sum(z**2, axis=1)
            logdet = 2.0 * pt.sum(pt.log(pt.diagonal(Ls, axis1=1, axis2=2)), axis=1)
            logp = -0.5 * (t*np.log(2*np.pi) + logdet + quad)
            pm.Potential(f"{target}:marginal_logp", pt.sum(logp))
            return model

    # -----------------------------
    # Posterior helper utilities
    # -----------------------------
    def genetic_correlation(self, idata, parent: str, target: str, *, summary: Literal["mean","median","none"]="mean", as_dataframe: bool = True):
        """Summarize posterior genetic correlations for a specified parent-to-trait path.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        parent : str
            Name of the graph parent whose genetic path covariance is summarized. Must identify the
            parent used to fit posterior variable RG:parent->target.

        target : str
            Target trait-node name on that genetic path. Its coords label both trait axes of the
            returned correlation matrix.

        summary : Literal["mean", "median", "none"]
            'mean' averages chain/draw correlations, 'median' takes their median, and 'none' keeps
            posterior draw dimensions. Use as_dataframe=False with unreduced draws because pandas
            conversion expects a matrix. Default: ``'mean'``.

        as_dataframe : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``True``.

        Returns
        -------
        pandas.DataFrame or xarray.DataArray
            Posterior genetic correlations for the selected path, reduced by summary.
            as_dataframe=False preserves xarray coordinates and unreduced draws.

        """
        key = f"RG:{parent}->{target}"
        R = idata.posterior[key]
        if summary == "mean":
            R = R.mean(dim=("chain","draw"))
        elif summary == "median":
            R = R.median(dim=("chain","draw"))
        if as_dataframe:
            cols = self.G.nodes[target]["coords"]
            return R.to_pandas().set_axis(cols).set_axis(cols, axis=1)
        return R

    def latent_draws(self, idata, node: str, *, n_draws: int | None = None):
        """Extract labeled posterior draws for the requested latent graph node.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        node : str
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        n_draws : int | None
            Number of posterior draws retained for this output; positive counts limit storage/work,
            and None uses all available draws. Default: ``None``. Passed to ``X.isel`` as ``draw``.

        Returns
        -------
        object
            Computed result from ``X``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X = idata.posterior[node]
        if n_draws is not None:
            X = X.isel(draw=slice(0, int(n_draws)))
        return X

    def latent_summary(self, idata, node: str, *, summary: Literal["mean","median"]="mean") -> pd.DataFrame:
        """Reduce posterior latent draws by the requested mean or median summary.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        node : str
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        summary : Literal['mean', 'median']
            Posterior reduction: 'mean' averages chain/draw values, 'median' takes their median and
            'none' preserves draws where that method supports unreduced output. Default: ``'mean'``.

        Returns
        -------
        object
            Computed result from ``arr``; see the summary and the called implementation for its labeled/numerical representation.
        """
        X = idata.posterior[node]
        if summary == "mean":
            arr = X.mean(dim=("chain","draw")).to_pandas()
        else:
            arr = X.median(dim=("chain","draw")).to_pandas()
        arr.index = self.obs_ids[:arr.shape[0]]
        return arr

    def blup_draws(self, idata, parent: str, target: str, *, n_draws: int = 100, obs_idx: np.ndarray | None = None) -> np.ndarray:
        """Posterior draws of genetic values g = X @ Z @ L^T for a GRM edge.
        Returns array (n_draws, n_obs, n_traits) or subset if obs_idx provided.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        parent : str
            Parent progress object or graph node used by the enclosing operation. None omits the
            parent relationship; graph methods require an existing node and progress methods a
            compatible bar.

        target : str
            Response/trait identifiers selected for this prediction, mediation or subset operation.
            Names/positions must exist in the supplied response axes; sequences preserve the
            requested ordering.

        n_draws : int
            Number of posterior draws retained for this output; positive counts limit storage/work,
            and None uses all available draws. Default: ``100``. Passed to
            ``Z_eff.stack(sample=('chain', 'draw')).isel`` as ``sample``.

        obs_idx : np.ndarray | None
            Observation/eligibility mask aligned to the corresponding input axis. True/one selects
            observed or eligible entries; false/zero excludes them. A sample mask is
            one-dimensional; cell masks match the matrix and packed masks follow the packed-byte
            axis. Default: ``None``. None delegates to the local fallback shown in this body; it is
            not an additional numeric/category value.

        Returns
        -------
        object
            Computed result from ``g``; see the summary and the called implementation for its labeled/numerical representation.
        """
        pa = self.G.nodes[parent]
        ta = self.G.nodes[target]
        X = np.asarray(pa["data"], dtype=float)
        if obs_idx is not None:
            X = X[np.asarray(obs_idx, dtype=int)]
        Z = idata.posterior[f"Z_raw:{parent}->{target}"]
        # resolve effective Z according to prior; simplest: use posterior of deterministic? (we didn't store)
        # We stored only Z_raw. If z_prior != iid, scale in post-hoc using stored scales if present.
        z_prior = str(self.G.edges[parent, target].get("z_prior","iid")).lower()
        if z_prior == "iid":
            Z_eff = Z
        elif z_prior == "ard_pc":
            tau = idata.posterior[f"tau_pc:{parent}->{target}"]
            Z_eff = Z * tau[..., :, None]
        elif z_prior == "rhs":
            tau = idata.posterior[f"rhs_tau:{parent}->{target}"]
            lam = idata.posterior[f"rhs_lam:{parent}->{target}"]
            c2  = idata.posterior[f"rhs_c2:{parent}->{target}"]
            lam_tilde = np.sqrt(c2) * lam / np.sqrt(c2 + (tau**2) * (lam**2))
            Z_eff = Z * (tau * lam_tilde)[..., :, None]
        else:
            Z_eff = Z
        L = idata.posterior[f"L:{parent}->{target}"]
        # select draws
        Z_eff = Z_eff.stack(sample=("chain","draw")).isel(sample=slice(0,int(n_draws))).values
        L = L.stack(sample=("chain","draw")).isel(sample=slice(0,int(n_draws))).values
        # compute g: (n_draws, n_obs, n_traits)
        g = np.einsum("nr,drt,dtp->dnp", X, Z_eff, np.transpose(L, (0,2,1)), optimize=True)
        return g

    def posterior_impute(
        self,
        idata,
        node: str,
        *,
        n_draws: int = 200,
        include_residual: bool = True,
        random_seed: int = 0,
    ) -> np.ndarray:
        """Sample imputations Y~p(Y|theta) for a node, returning array (n_draws, n_obs, n_dim).
        Works even when missing='mask' because we sample from posterior deterministics + residual params.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        node : str
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        n_draws : int
            Number of posterior draws retained for this output; positive counts limit storage/work,
            and None uses all available draws. Default: ``200``. Passed to
            ``idata.posterior[f'prediction:{node}'].stack(sample=('chain', 'draw')).isel`` as
            ``sample``.

        include_residual : bool
            Whether to enforce/use complete observations or include the named
            decomposition/component output. The enabled branch below defines the returned
            representation; disabling skips that optional step. Default: ``True``.

        random_seed : int
            Input for the operation ``np.random.default_rng(int(random_seed))``. See
            ``vigor.causal.CausalGraph.posterior_impute`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default: ``0``.

        Returns
        -------
        object
            Computed result from ``out``; see the summary and the called implementation for its labeled/numerical representation.
        """
        a = self.G.nodes[node]
        Y = np.asarray(a["data"], dtype=float)
        n, t = Y.shape
        rng = np.random.default_rng(int(random_seed))

        mu = idata.posterior[f"prediction:{node}"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
        # If the likelihood mean included residual factor effects, we stored node: Efac deterministics
        mean_total = mu
        if include_residual and f"{node}:Efac" in idata.posterior:
            efac = idata.posterior[f"{node}:Efac"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
            mean_total = mean_total + efac

        sigma = idata.posterior[f"{node}:sigma"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
        if sigma.ndim == 1:
            sigma = sigma[:, None]  # (draw,1)

        out = mean_total.copy()
        if include_residual:
            eps = rng.normal(size=out.shape) * sigma[:, None, :]
            out = out + eps
        # for observed entries, overwrite with observed values if desired? (we keep posterior predictive draw)
        return out

    def explained_variance(self, idata, node: str, *, summary: Literal["mean","median"]="mean") -> pd.DataFrame:
        """Return per-trait realized variance diagnostics if present:
          V_pred:<node>, V_E_param:<node>, V_total_real:<node>, h2_param:<node>
        and EV_path:* if available.

        Parameters
        ----------
        idata : InferenceData, posterior Dataset or mapping
            Fitted posterior/point-estimate draws with the named variables required by this graph
            method. Trait/node coordinates must match the graph used for fitting.

        node : str
            Axis/node identifiers or positions selected by the local lookup below. Values must refer
            to the supplied object and preserve the requested output ordering.

        summary : Literal['mean', 'median']
            Posterior reduction: 'mean' averages chain/draw values, 'median' takes their median and
            'none' preserves draws where that method supports unreduced output. Default: ``'mean'``.

        Returns
        -------
        object
            Computed result from ``df``; see the summary and the called implementation for its labeled/numerical representation.
        """
        a = self.G.nodes[node]
        cols = a["coords"]
        post = idata.posterior

        def _summ(x):
            """Reduce posterior chain/draw values using the enclosing report summary policy.

            Parameters
            ----------
            x : array-like
                Numerical operand in ``x.mean(dim=('chain', 'draw')).to_pandas()``. Shape must conform
                to that matrix/vector operation and the enclosing prepared state.
            """
            if summary == "mean":
                return x.mean(dim=("chain","draw")).to_pandas()
            return x.median(dim=("chain","draw")).to_pandas()

        rows = {}
        for key in [f"V_pred:{node}", f"V_E_param:{node}", f"V_total_real:{node}", f"h2_param:{node}", f"V_total_param:{node}"]:
            if key in post:
                rows[key] = _summ(post[key]).values
        # EV per path
        for p in self.G.predecessors(node):
            key = f"EV_path:{p}->{node}"
            if key in post:
                rows[key] = _summ(post[key]).values
        df = pd.DataFrame(rows, index=cols)
        return df

    def clone(self) -> "CausalGraph":
        """Deep-copy graph model inputs and configuration into an independent graph object.

        Returns
        -------
        object
            Computed result from ``cg``; see the summary and the called implementation for its labeled/numerical representation.
        """
        cg = CausalGraph(self.df_main)
        cg.obs_ids = list(self.obs_ids)
        cg.G = copy.deepcopy(self.G)
        cg._last_build = copy.deepcopy(self._last_build)
        return cg

    def qc_tabs(
        self,
        fit: Any,
        *,
        hdi_prob: float = 0.95,
        max_text_animals: int = 100,
        max_text_cells: int = 4000,
        top_k_parent: int = 60,
        width: int = 950,
        height: int = 680,
    ):
        """Panel Tabs QC view:
          - Graph tab (edge width ~ explained variance)
          - Per-edge tabs: EV heatmap (+ corr heatmap if LKJ/corr present)
          - Per latent/traits node tabs: mean + CI heatmap over animals x dims

        Supports:
          - posterior InferenceData (fit.posterior exists)
          - MAP dict (single point; CI collapses to the point)

        Parameters
        ----------
        fit : Any
            Previously prepared model/result or solver state consumed by this operation. Use the
            object returned by the corresponding fitting/preparation routine with the same animal
            and trait axes.

        hdi_prob : float
            Posterior/analytical interval coverage probability, strictly between zero and one; the
            routine determines whether intervals use posterior quantiles or a normal approximation.
            Default: ``0.95``.

        max_text_animals : int
            Count limit for the corresponding sampled/retained/displayed items. Positive values
            bound work/output size; None uses all eligible items or the local default where
            supported. Default: ``100``.

        max_text_cells : int
            Count limit for the corresponding sampled/retained/displayed items. Positive values
            bound work/output size; None uses all eligible items or the local default where
            supported. Default: ``4000``. Passed to ``_heatmap_with_text`` as ``add_text``.

        top_k_parent : int
            Maximum candidates/features retained by the corresponding ranking stage. Positive
            integers bound work/model size; None leaves retention to that stage's policy. Default:
            ``60``.

        width : int
            Rendered plot dimension in pixels. Positive values control the corresponding
            horizontal/vertical extent. Default: ``950``. Passed to ``(edge_plot * node_plot *
            labels).opts`` as ``width``.

        height : int
            Rendered plot dimension in pixels. Positive values control the corresponding
            horizontal/vertical extent. Default: ``680``. Passed to ``(edge_plot * node_plot *
            labels).opts`` as ``height``.

        Returns
        -------
        object
            Computed result from ``tabs``; see the summary and the called implementation for its labeled/numerical representation.
        """
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        import xarray as xr
    
        hv.extension("bokeh")
    
        def _is_posterior(obj: Any) -> bool:
            """Test whether the supplied object exposes posterior draws.

            Parameters
            ----------
            obj : Any
                Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
                described in the summary; the local coercion/selection expression below defines the
                accepted representation.
            """
            return hasattr(obj, "posterior")
    
        def _stack_samples(da: xr.DataArray) -> xr.DataArray:
            """Stack posterior chain and draw dimensions into one sample dimension.

            Parameters
            ----------
            da : xr.DataArray
                Labeled data object containing the variables and dimension coordinates read below.
                Scientific axes/metadata must match the corresponding genotype, association or posterior
                schema.
            """
            if ("chain" in da.dims) and ("draw" in da.dims):
                return da.stack(sample=("chain", "draw"))
            if "sample" in da.dims:
                return da
            # already point estimate
            return da.expand_dims(sample=[0])
    
        def _mean_ci(da: xr.DataArray, prob: float) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
            """Summarize posterior means and central quantile intervals for each labeled value.

            Parameters
            ----------
            da : xr.DataArray
                Labeled data object containing the variables and dimension coordinates read below.
                Scientific axes/metadata must match the corresponding genotype, association or posterior
                schema.

            prob : float
                Posterior/analytical interval coverage probability, strictly between zero and one; the
                routine determines whether intervals use posterior quantiles or a normal approximation.
            """
            da = _stack_samples(da)
            q = (1.0 - float(prob)) / 2.0
            mean = da.mean("sample")
            lo = da.quantile(q, dim="sample").drop_vars("quantile", errors="ignore")
            hi = da.quantile(1.0 - q, dim="sample").drop_vars("quantile", errors="ignore")
            return mean, lo, hi
    
        def _heatmap_with_text(
            mean_df: pd.DataFrame,
            lo_df: pd.DataFrame | None,
            hi_df: pd.DataFrame | None,
            *,
            title: str,
            x_name: str,
            y_name: str,
            add_text: bool,
        ):
            """Render a labeled matrix heatmap with optional uncertainty text overlays.

            Parameters
            ----------
            mean_df : pd.DataFrame
                Input for the operation ``mean_df.stack().reset_index()``. See
                ``vigor.causal.CausalGraph.qc_tabs._heatmap_with_text`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.

            lo_df : pd.DataFrame | None
                Input for the operation ``lo_df.stack().reset_index(drop=True)``. See
                ``vigor.causal.CausalGraph.qc_tabs._heatmap_with_text`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.

            hi_df : pd.DataFrame | None
                Input for the operation ``hi_df.stack().reset_index(drop=True)``. See
                ``vigor.causal.CausalGraph.qc_tabs._heatmap_with_text`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.

            title : str
                Displayed title, status or descriptive text attached to the visualization/progress
                output. Passed to ``long.hvplot.heatmap`` as ``title``.

            x_name : str
                Input supplied to ``long.hvplot.heatmap`` as ``x``. See that callable's documented ``x``
                contract for its type, alignment and value domain. This wrapper retains that contract.
                Passed to ``long.hvplot.heatmap`` as ``x``.

            y_name : str
                Input supplied to ``long.hvplot.heatmap`` as ``y``. See that callable's documented ``y``
                contract for its type, alignment and value domain. This wrapper retains that contract.
                Passed to ``long.hvplot.heatmap`` as ``y``.

            add_text : bool
                Context/control value read by the enclosing operation; its contract follows the explicit
                local use and the caller.
            """
            long = mean_df.stack().reset_index()
            long.columns = [y_name, x_name, "mean"]
            if (lo_df is not None) and (hi_df is not None):
                lo_long = lo_df.stack().reset_index(drop=True)
                hi_long = hi_df.stack().reset_index(drop=True)
                long["lo"] = lo_long
                long["hi"] = hi_long
                if add_text:
                    long["text"] = long.apply(
                        lambda r: f"{r['mean']:.3g}\n[{r['lo']:.3g},{r['hi']:.3g}]",
                        axis=1,
                    )
            hm = long.hvplot.heatmap(
                x=x_name,
                y=y_name,
                C="mean",
                title=title,
                frame_width=int(width),
                frame_height=520,
                xlabel="",
                ylabel="",
            )
            if add_text and ("text" in long.columns):
                labels = hv.Labels(long, kdims=[x_name, y_name], vdims=["text"]).opts(
                    text_font_size="7pt",
                    text_align="center",
                    text_baseline="middle",
                )
                return hm * labels
            return hm
    
        def chol2cov(L, return_correlation: bool = False, eps: float = 1e-12):
            # expects xarray DataArray with square last-two dims
            """Reconstruct covariance from Cholesky factors, optionally normalizing to correlation.

            Parameters
            ----------
            L : array-like
                Numerical input passed to ``(L ** 2).sum`` as ``dim``; see that callable for required
                dimensions and matrix role. Passed to ``(L ** 2).sum`` as ``dim``.

            return_correlation : bool
                Whether to normalize reconstructed covariance by marginal standard deviations, returning
                correlation instead of covariance. Default: ``False``.

            eps : float
                Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
                denominators or nonpositive factor variances. Choose a positive value on the scale of
                the input quantities. Default: ``1e-12``.

            Returns
            -------
            object
                Computed result from ``cov / (denom + eps)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            variance = (L**2).sum(dim=L.dims[-1])
            stds = np.sqrt(variance)
            L_t = L.rename({L.dims[-2]: "dim_0", L.dims[-1]: "dim_t"})
            cov = xr.dot(L_t, L_t.rename({"dim_0": "dim_1"}), dims="dim_t")
            if not return_correlation:
                return cov
            s0 = stds.rename({stds.dims[-1]: "dim_0"})
            s1 = stds.rename({stds.dims[-1]: "dim_1"})
            denom = s0 * s1
            return cov / (denom + eps)
    
        def chol2covMAP(map_trace: dict[str, Any], name: str, columns, return_correlation: bool = False, eps: float = 1e-12):
            """Reconstruct labeled covariance or correlation from a point-estimate Cholesky factor.

            Parameters
            ----------
            map_trace : dict[str, Any]
                Fitted posterior/point-estimate draws with the named variables required by this graph
                method. Trait/node coordinates must match the graph used for fitting.

            name : str
                Identifier used to select or label the relevant component, field or output quantity. It
                must match the names present in the supplied object where lookup is required.

            columns : sequence or pandas.Index
                Trait labels in matrix-column order, used to label or align the resulting trait
                quantities.

            return_correlation : bool
                Whether to normalize reconstructed covariance by marginal standard deviations, returning
                correlation instead of covariance. Default: ``False``.

            eps : float
                Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
                denominators or nonpositive factor variances. Choose a positive value on the scale of
                the input quantities. Default: ``1e-12``.

            Returns
            -------
            object
                Computed result from ``pd.DataFrame(cov / (denom + eps), columns=columns, index=columns)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            L = np.asarray(map_trace[name])
            cov = L.dot(L.T)
            if not return_correlation:
                return pd.DataFrame(cov, columns=columns, index=columns)
            sig = np.sqrt(np.diag(cov))
            denom = np.outer(sig, sig)
            return pd.DataFrame(cov / (denom + eps), columns=columns, index=columns)
    
        # ---- pick posterior vs MAP ----
        is_post = _is_posterior(fit)
        post = fit.posterior if is_post else None
        map_trace = fit if (not is_post) else None
    
        # ---- explained variance per edge (diag approximation) ----
        edge_ev_scalar: dict[tuple[str, str], float] = {}
        edge_tabs = []
    
        # Build a plotting graph P with EV-based "confidence"
        P = nx.DiGraph()
        for n, attr in self.G.nodes(data=True):
            ntype = str(attr.get("type", "unknown"))
            shape = attr.get("shape", None)
            shape_str = "x".join(map(str, shape)) if shape is not None else "?"
            likelihood = str(attr.get("likelihood", "N/A"))
            P.add_node(n, type=ntype, shape_str=shape_str, likelihood=likelihood)
    
        def _edge_ev_matrix(u: str, v: str, ed: dict[str, Any]):
            """Summarize edge-specific explained variance in its source and target trait coordinates.

            Parameters
            ----------
            u : str
                Input for the operation ``self.G.nodes[u]``. See
                ``vigor.causal.CausalGraph.qc_tabs._edge_ev_matrix`` and its caller for the scientific
                representation; the shown operation constrains the required type/shape.

            v : str
                Numerical operand in ``self.G.nodes[v]``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.

            ed : dict[str, Any]
                Input for the operation ``str(ed.get('kind', 'dense')).lower()``. See
                ``vigor.causal.CausalGraph.qc_tabs._edge_ev_matrix`` and its caller for the scientific
                representation; the shown operation constrains the required type/shape.
                Source-recognized literal values: ``ard_pc``, ``factor``, ``grm``, ``rhs``. These select
                the corresponding branches below; comparison evidence alone does not establish an
                exhaustive enum.
            """
            kind = str(ed.get("kind", "dense")).lower()
    
            # ------- dense: beta -------
            if kind == "dense":
                bname = f"beta:{u}->{v}"
                X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                if X.ndim != 2:
                    return None, None, None
                varX = X.var(axis=0, ddof=0)
    
                if is_post and (bname in post):
                    beta = post[bname]  # (chain,draw,p,q)
                    # diag approx EV[p,q] = varX[p] * beta[p,q]^2
                    varX_da = xr.DataArray(varX, dims=(beta.dims[-2],), coords={beta.dims[-2]: beta.coords[beta.dims[-2]]})
                    ev = (beta**2) * varX_da[..., None]
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                if (not is_post) and (bname in map_trace):
                    beta = np.asarray(map_trace[bname])
                    ev = (beta**2) * varX[:, None]
                    pcoords = self.G.nodes[u].get("coords", [f"p{i}" for i in range(ev.shape[0])])
                    qcoords = self.G.nodes[v].get("coords", [f"q{i}" for i in range(ev.shape[1])])
                    mean = xr.DataArray(ev, dims=("p", "q"), coords={"p": pcoords, "q": qcoords})
                    lo = mean.copy()
                    hi = mean.copy()
                    return mean, lo, hi
    
                return None, None, None
    
            # ------- factor: Lambda + latent variance -------
            if kind == "factor":
                lname = f"Lambda:{u}->{v}"
                if not is_post:
                    # MAP: just use Lambda^2 * Var(latent) using point estimate (latent may not be stored)
                    if (lname in map_trace) and (u in map_trace):
                        Lam = np.asarray(map_trace[lname])  # (traits, k) by compiler
                        Lat = np.asarray(map_trace[u])      # (obs, k)
                        varLat = Lat.var(axis=0, ddof=0)     # (k,)
                        ev = (Lam**2) * varLat[None, :]
                        # return as (k, traits) for "beta x trait"
                        ev = ev.T
                        kcoords = self.G.nodes[u].get("coords", [f"k{i}" for i in range(ev.shape[0])])
                        tcoords = self.G.nodes[v].get("coords", [f"t{i}" for i in range(ev.shape[1])])
                        mean = xr.DataArray(ev, dims=("k", "t"), coords={"k": kcoords, "t": tcoords})
                        return mean, mean.copy(), mean.copy()
                    return None, None, None
    
                if (lname in post) and (u in post):
                    Lam = post[lname]     # (chain,draw,traits,k)
                    Lat = post[u]         # (chain,draw,obs_id,k)
                    varLat = Lat.var(dim="obs_id")  # (chain,draw,k)
                    # EV[trait,k] = Lam^2 * Var(latent_k)
                    ev_tk = (Lam**2) * varLat[..., None, :]  # broadcast to (chain,draw,traits,k)
                    # to "beta x trait": (k, traits)
                    ev = ev_tk.transpose(*ev_tk.dims[:-2], ev_tk.dims[-1], ev_tk.dims[-2])
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                return None, None, None
    
            # ------- grm: Z (+ shrink) -------
            if kind == "grm":
                zname = f"Z_raw:{u}->{v}"
                z_prior = str(ed.get("z_prior", "iid")).lower()
    
                X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                if X.ndim != 2:
                    return None, None, None
                varX = X.var(axis=0, ddof=0)  # (pc,)
    
                if is_post and (zname in post):
                    Zraw = post[zname]  # (chain,draw,pc,dim)
    
                    Z = Zraw
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in post:
                            tau = post[tname]  # (chain,draw,pc)
                            Z = Zraw * tau[..., :, None]
                    elif z_prior == "rhs":
                        # match compiler: lam_tilde = sqrt(c2)*lam / sqrt(c2 + tau^2*lam^2)
                        tname = f"rhs_tau:{u}->{v}"
                        lname = f"rhs_lam:{u}->{v}"
                        cname = f"rhs_c2:{u}->{v}"
                        if (tname in post) and (lname in post) and (cname in post):
                            tau = post[tname]      # scalar (chain,draw)
                            lam = post[lname]      # (chain,draw,pc)
                            c2 = post[cname]       # scalar (chain,draw)
                            lam_tilde = (np.sqrt(1.0) * pt.sqrt(c2) * lam) / pt.sqrt(c2 + (tau**2) * (lam**2))
                            # xarray-safe: avoid pytensor; do it numerically with xr
                            lam_tilde = (c2**0.5) * lam / (c2 + (tau**2) * (lam**2))**0.5
                            Z = Zraw * (tau * lam_tilde)[..., :, None]
    
                    # EV[pc,dim] = varX[pc] * Z[pc,dim]^2
                    varX_da = xr.DataArray(varX, dims=(Z.dims[-2],), coords={Z.dims[-2]: Z.coords[Z.dims[-2]]})
                    ev = (Z**2) * varX_da[..., None]  # (chain,draw,pc,dim)
    
                    # optionally restrict parent dim for display
                    pc_dim = ev.dims[-2]
                    dim_dim = ev.dims[-1]
                    n_pc = int(ev.sizes[pc_dim])
                    if n_pc > int(top_k_parent):
                        score = _stack_samples(ev).mean("sample").sum(dim_dim)  # (pc,)
                        top_idx = np.argsort(score.values)[::-1][: int(top_k_parent)]
                        ev = ev.isel({pc_dim: top_idx})
    
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                if (not is_post) and (zname in map_trace):
                    Z = np.asarray(map_trace[zname])
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in map_trace:
                            tau = np.asarray(map_trace[tname])
                            Z = Z * tau[:, None]
                    ev = (Z**2) * varX[:, None]
                    # restrict
                    if ev.shape[0] > int(top_k_parent):
                        score = ev.mean(axis=1)
                        top_idx = np.argsort(score)[::-1][: int(top_k_parent)]
                        ev = ev[top_idx, :]
    
                    pcoords = self.G.nodes[u].get("coords", [f"Pc{i}" for i in range(ev.shape[0])])
                    dcoords = self.G.nodes[v].get("coords", [f"d{i}" for i in range(ev.shape[1])])
                    mean = xr.DataArray(ev, dims=("pc", "d"), coords={"pc": pcoords[: ev.shape[0]], "d": dcoords})
                    return mean, mean.copy(), mean.copy()
    
                return None, None, None
    
            return None, None, None
    
        def _edge_corr_matrix(u: str, v: str):
            # prefer RG deterministic if present; else try L -> corr via chol2cov
            """Summarize posterior trait correlations for one graph edge.

            Parameters
            ----------
            u : str
                Context/control value read by the enclosing operation; its contract follows the explicit
                local use and the caller.

            v : str
                Numerical operand in ``self.G.nodes[v]``. Shape must conform to that matrix/vector
                operation and the enclosing prepared state.
            """
            rg = f"RG:{u}->{v}"
            l = f"L:{u}->{v}"
    
            if is_post:
                if rg in post:
                    mean, lo, hi = _mean_ci(post[rg], hdi_prob)
                    return mean, lo, hi
                if l in post:
                    corr = chol2cov(post[l], return_correlation=True)
                    mean, lo, hi = _mean_ci(corr, hdi_prob)
                    return mean, lo, hi
                return None, None, None
    
            # MAP
            if (not is_post) and (rg in map_trace):
                C = np.asarray(map_trace[rg])
                cols = self.G.nodes[v].get("coords", [f"d{i}" for i in range(C.shape[0])])
                mean = xr.DataArray(C, dims=("d0", "d1"), coords={"d0": cols, "d1": cols})
                return mean, mean.copy(), mean.copy()
            if (not is_post) and (l in map_trace):
                cols = self.G.nodes[v].get("coords", [f"d{i}" for i in range(np.asarray(map_trace[l]).shape[0])])
                Cdf = chol2covMAP(map_trace, l, columns=cols, return_correlation=True)
                mean = xr.DataArray(Cdf.values, dims=("d0", "d1"), coords={"d0": cols, "d1": cols})
                return mean, mean.copy(), mean.copy()
    
            return None, None, None
    
        # Compute edge EV scalars and build edge tabs
        for u, v, ed in self.G.edges(data=True):
            mean_ev, lo_ev, hi_ev = _edge_ev_matrix(u, v, ed)
            if mean_ev is None:
                edge_ev_scalar[(u, v)] = 0.0
                continue
    
            # scalar EV for graph thickness: sum over matrix
            ev_scalar = float(np.asarray(mean_ev).sum())
            edge_ev_scalar[(u, v)] = ev_scalar
    
            # edge tab content
            # Format EV matrix as DataFrames
            # The EV matrix is "beta_dim x trait_dim" for factor/grm display; for dense it is parent_dim x child_dim.
            ev_mean_df = mean_ev.to_pandas()
            ev_lo_df = lo_ev.to_pandas() if lo_ev is not None else None
            ev_hi_df = hi_ev.to_pandas() if hi_ev is not None else None
    
            add_text = (ev_mean_df.size <= int(max_text_cells))
            ev_plot = _heatmap_with_text(
                ev_mean_df,
                ev_lo_df,
                ev_hi_df,
                title=f"Explained variance: {u} -> {v}",
                x_name="child_dim",
                y_name="beta_dim",
                add_text=add_text,
            )
    
            col = pn.Column(ev_plot)
    
            # Add correlation heatmap if present
            corr_mean, corr_lo, corr_hi = _edge_corr_matrix(u, v)
            if corr_mean is not None:
                cm = corr_mean.to_pandas()
                cl = corr_lo.to_pandas() if corr_lo is not None else None
                ch = corr_hi.to_pandas() if corr_hi is not None else None
                corr_plot = _heatmap_with_text(
                    cm,
                    cl,
                    ch,
                    title=f"Correlation (LKJ/corr): {u} -> {v}",
                    x_name="dim_1",
                    y_name="dim_0",
                    add_text=(cm.size <= int(max_text_cells)),
                )
                col.append(corr_plot)
    
            edge_tabs.append((f"edge: {u}->{v}", col))
    
        # ---- Graph plot scaled by EV ----
        # Build plotting graph with EV-scaled edge "confidence"
        for u, v, ed in self.G.edges(data=True):
            kind = str(ed.get("kind", "dense")).lower()
            ev = float(edge_ev_scalar.get((u, v), 0.0))
            P.add_edge(u, v, kind=kind, ev=ev)
    
        # subsets for layout
        for n in P.nodes():
            out_degree = P.out_degree(n)
            in_degree = P.in_degree(n)
            subset = 0 if in_degree == 0 else (2 if out_degree == 0 else 1)
            P.nodes[n]["subset"] = int(subset)
    
        pos = nx.multipartite_layout(P, subset_key="subset", scale=2.0)
    
        color_map = {"exo": "lightsteelblue", "grm": "salmon", "latent": "palegoldenrod", "endo": "darkseagreen"}
        marker_map = {"exo": "circle", "grm": "hex", "latent": "square", "endo": "square"}
        edge_color_map = {"dense": "steelblue", "grm": "firebrick", "factor": "purple"}
    
        for n in P.nodes():
            t = P.nodes[n]["type"]
            P.nodes[n]["color"] = str(color_map.get(t, "white"))
            P.nodes[n]["marker"] = str(marker_map.get(t, "circle"))
    
        # normalize EV to a bounded "confidence"
        ev_vals = np.array([P.edges[e].get("ev", 0.0) for e in P.edges()], dtype=float)
        ev_max = float(ev_vals.max()) if len(ev_vals) else 1.0
        for u, v in P.edges():
            ev = float(P.edges[u, v].get("ev", 0.0))
            frac = 0.0 if ev_max <= 0 else (ev / ev_max)
            P.edges[u, v]["confidence"] = float(0.1 + 3.9 * np.sqrt(max(frac, 0.0)))
    
        edge_tooltips = [("Edge", "@start -> @end"), ("Kind", "@kind"), ("EV", "@ev")]
        node_tooltips = [("Node", "@index"), ("Type", "@type"), ("Shape", "@shape_str"), ("Likelihood", "@likelihood")]
    
        edge_plot = hvnx.draw_networkx_edges(
            P,
            pos,
            edge_color="kind",
            alpha=0.65,
            arrow_style="-|>",
            arrow_size=5,
            edge_width=4 + hv.dim("confidence").norm() * 15,
            cmap=edge_color_map,
        ).opts(tools=["hover"], hover_tooltips=edge_tooltips)
    
        node_plot = hvnx.draw_networkx_nodes(
            P,
            pos,
            node_color="type",
            node_marker="marker",
            alpha=1.0,
            node_size=5200,
            cmap=color_map,
        ).opts(tools=["hover"], hover_tooltips=node_tooltips)
    
        label_df = pd.DataFrame(
            {"x": [float(pos[n][0]) for n in P.nodes()],
             "y": [float(pos[n][1]) for n in P.nodes()],
             "text": [f"{n}\n{P.nodes[n]['shape_str']}" for n in P.nodes()]}
        )
        labels = hv.Labels(label_df, kdims=["x", "y"], vdims=["text"]).opts(
            text_color="black",
            text_font_size="8pt",
            text_align="center",
            text_baseline="middle",
        )
    
        graph = (edge_plot * node_plot * labels).opts(
            title="Graph (edge width ~ explained variance)",
            width=int(width),
            height=int(height),
            xaxis=None,
            yaxis=None,
            tools=["tap", "box_select", "hover"],
        )
    
        # ---- Node tabs (latent + endo) ----
        node_tabs = []
        for n, a in self.G.nodes(data=True):
            if str(a.get("type")) not in {"latent", "endo"}:
                continue
    
            if not is_post:
                # MAP: try to show point estimate if present
                if n in map_trace:
                    A = np.asarray(map_trace[n])
                    cols = a.get("coords", [f"d{i}" for i in range(A.shape[1])])
                    idx = self.G.nodes.get("obs_id", None)
                    mean_df = pd.DataFrame(A, columns=cols)
                    lo_df = mean_df.copy()
                    hi_df = mean_df.copy()
                    add_text = (mean_df.shape[0] <= int(max_text_animals)) and (mean_df.size <= int(max_text_cells))
                    plot = _heatmap_with_text(
                        mean_df,
                        lo_df,
                        hi_df,
                        title=f"{n} (MAP point)",
                        x_name="dim",
                        y_name="animal",
                        add_text=add_text,
                    )
                    node_tabs.append((f"node: {n}", pn.Column(plot)))
                continue
    
            # posterior case
            if a["type"] == "endo" and (f"prediction:{n}" in post):
                mu = post[f"prediction:{n}"]
                # if residual factor exists, add it
                efac_name = f"{n}:Efac"
                if efac_name in post:
                    mu = mu + post[efac_name]
                target = mu
            else:
                if n not in post:
                    continue
                target = post[n]
    
            mean, lo, hi = _mean_ci(target, hdi_prob)
            mdf = mean.to_pandas()
            ldf = lo.to_pandas()
            hdf = hi.to_pandas()
    
            add_text = (mdf.shape[0] <= int(max_text_animals)) and (mdf.size <= int(max_text_cells))
            plot = _heatmap_with_text(
                mdf,
                ldf,
                hdf,
                title=f"{n}: mean and CI",
                x_name="dim",
                y_name="animal",
                add_text=add_text,
            )
            node_tabs.append((f"node: {n}", pn.Column(plot)))
    
        tabs = pn.Tabs(("graph", graph), *edge_tabs, *node_tabs)
        return tabs
    
    
    def trace_tabs(
        self,
        fit: Any,
        *,
        hdi_prob: float = 0.95,
        top_k_parent: int = 60,
        width: int = 950,
        frame_width: int = 520,
        frame_height: int = 220,
    ):
        """Same structure as qc_tabs(), but edge/node tabs show trace-style plots
        (KDE + draw lines) similar to plottrace() / Gcorr_trace().

        Only meaningful for posterior InferenceData. For MAP dict, it returns
        a minimal Tabs with a message + graph.

        Parameters
        ----------
        fit : Any
            Previously prepared model/result or solver state consumed by this operation. Use the
            object returned by the corresponding fitting/preparation routine with the same animal
            and trait axes.

        hdi_prob : float
            Posterior/analytical interval coverage probability, strictly between zero and one; the
            routine determines whether intervals use posterior quantiles or a normal approximation.
            Default: ``0.95``.

        top_k_parent : int
            Maximum candidates/features retained by the corresponding ranking stage. Positive
            integers bound work/model size; None leaves retention to that stage's policy. Default:
            ``60``.

        width : int
            Rendered plot dimension in pixels. Positive values control the corresponding
            horizontal/vertical extent. Default: ``950``.

        frame_width : int
            Rendered plot dimension in pixels. Positive values control the corresponding
            horizontal/vertical extent. Default: ``520``.

        frame_height : int
            Rendered plot dimension in pixels. Positive values control the corresponding
            horizontal/vertical extent. Default: ``220``.

        Returns
        -------
        object
            Computed result from ``pn.Tabs(*tabs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        import xarray as xr
    
        hv.extension("bokeh")
    
        def _is_posterior(obj: Any) -> bool:
            """Test whether the supplied object exposes posterior draws.

            Parameters
            ----------
            obj : Any
                Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
                described in the summary; the local coercion/selection expression below defines the
                accepted representation.
            """
            return hasattr(obj, "posterior")
    
        if not _is_posterior(fit):
            return pn.Tabs(("trace", pn.pane.Markdown("MAP fit has no posterior draws to plot traces.")))
    
        post = fit.posterior
    
        def _stack_samples(da: xr.DataArray) -> xr.DataArray:
            """Stack posterior chain and draw dimensions into one sample dimension.

            Parameters
            ----------
            da : xr.DataArray
                Labeled data object containing the variables and dimension coordinates read below.
                Scientific axes/metadata must match the corresponding genotype, association or posterior
                schema.
            """
            if ("chain" in da.dims) and ("draw" in da.dims):
                return da.stack(sample=("chain", "draw"))
            return da
    
        def _kde_plus_line(df: pd.DataFrame, *, title: str, xlim=None, ylim=None):
            # df: rows=draws, cols=series labels
            """Return the kde plus line result computed by ``(kde + line).opts(toolbar='right').opts``.

            Parameters
            ----------
            df : pd.DataFrame
                Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
                described in the summary; the local coercion/selection expression below defines the
                accepted representation.

            title : str
                Displayed title, status or descriptive text attached to the visualization/progress
                output. Passed to ``long.hvplot.kde`` as ``title``.

            xlim : object (see the cited input contract)
                Input supplied to ``long.hvplot.kde`` as ``xlim``. See that callable's documented
                ``xlim`` contract for its type, alignment and value domain. This wrapper retains that
                contract. Default: ``None``. None delegates to the local fallback shown in this body; it
                is not an additional numeric/category value. Passed to ``long.hvplot.kde`` as ``xlim``.

            ylim : object (see the cited input contract)
                Input supplied to ``long.hvplot.line`` as ``ylim``. See that callable's documented
                ``ylim`` contract for its type, alignment and value domain. This wrapper retains that
                contract. Default: ``None``. None delegates to the local fallback shown in this body; it
                is not an additional numeric/category value. Passed to ``long.hvplot.line`` as ``ylim``.
            """
            long = df.reset_index(names="draw").melt(id_vars="draw", var_name="series", value_name="value")
            kde = long.hvplot.kde(
                by="series",
                y="value",
                title=title,
                alpha=0.25,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                cut=0,
                xlabel="",
                xlim=xlim,
            ).opts(show_legend=False)
            line = long.hvplot.line(
                by="series",
                x="draw",
                y="value",
                alpha=0.55,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                xlabel="",
                ylim=ylim,
            ).opts(show_legend=False)
            return (kde + line).opts(toolbar="right").opts(shared_axes=False)
    
        def _corr_trace_plot(corr: xr.DataArray, *, title: str):
            # corr dims: (chain,draw,dim0,dim1) or stacked sample
            """Render posterior correlation densities and draw traces for diagnostic comparison.

            Parameters
            ----------
            corr : xr.DataArray
                Input for the operation ``_stack_samples(corr)``. See
                ``vigor.causal.CausalGraph.trace_tabs._corr_trace_plot`` and its caller for the
                scientific representation; the shown operation constrains the required type/shape.

            title : str
                Displayed title, status or descriptive text attached to the visualization/progress
                output. Passed to ``_kde_plus_line`` as ``title``.
            """
            C = _stack_samples(corr)
            # take upper triangle pairs
            d0, d1 = C.dims[-2], C.dims[-1]
            labels0 = list(map(str, C.coords[d0].values))
            labels1 = list(map(str, C.coords[d1].values))
            n = len(labels0)
            tri = np.triu_indices(n, k=1)
    
            # build DataFrame: sample x pair
            mat = C.values  # (sample,n,n)
            pairs = []
            cols = []
            for i, j in zip(*tri):
                cols.append(f"{labels0[i]}_{labels1[j]}")
                pairs.append(mat[:, i, j])
            if not cols:
                return pn.pane.Markdown("Correlation matrix too small to plot pairs.")
    
            df = pd.DataFrame(np.vstack(pairs).T, columns=cols)
            return _kde_plus_line(df, title=title, xlim=(-1, 1), ylim=(-1, 1))
    
        # We reuse qc_tabs' graph quickly (no EV width here; user can call qc_tabs for that)
        graph = self.plot_graph(show_edge_kinds=True)
    
        tabs = [("graph", graph)]
    
        # ---- edge tabs ----
        for u, v, ed in self.G.edges(data=True):
            kind = str(ed.get("kind", "dense")).lower()
    
            col = pn.Column()
    
            if kind == "dense":
                bname = f"beta:{u}->{v}"
                if bname in post:
                    B = _stack_samples(post[bname])  # (sample,p,q)
                    # summarize per child dim: sum_p VarX[p] * beta^2
                    X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                    varX = X.var(axis=0, ddof=0)
                    # (sample,q)
                    ev_by_child = (B**2 * xr.DataArray(varX, dims=(B.dims[-2],))).sum(dim=B.dims[-2])
                    df = ev_by_child.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by child dim: {u}->{v}"))
    
            elif kind == "factor":
                lname = f"Lambda:{u}->{v}"
                if (lname in post) and (u in post):
                    Lam = _stack_samples(post[lname])  # (sample,traits,k)
                    Lat = _stack_samples(post[u])      # (sample,obs,k)
                    varLat = Lat.var(dim=Lat.dims[-2]) # (sample,k)
                    ev_by_trait = ((Lam**2) * varLat[..., None, :]).sum(dim=Lam.dims[-1])  # (sample,traits)
                    df = ev_by_trait.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by trait: {u}->{v}"))
    
            elif kind == "grm":
                zname = f"Z_raw:{u}->{v}"
                z_prior = str(ed.get("z_prior", "iid")).lower()
                if zname in post:
                    Zraw = _stack_samples(post[zname])  # (sample,pc,dim)
                    Z = Zraw
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in post:
                            tau = _stack_samples(post[tname])  # (sample,pc)
                            Z = Zraw * tau[..., :, None]
    
                    X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                    varX = X.var(axis=0, ddof=0)
    
                    # restrict PCs for plots
                    pc_dim = Z.dims[-2]
                    dim_dim = Z.dims[-1]
                    n_pc = int(Z.sizes[pc_dim])
                    if n_pc > int(top_k_parent):
                        score = (Z**2).mean("sample").sum(dim_dim)
                        top_idx = np.argsort(score.values)[::-1][: int(top_k_parent)]
                        Z = Z.isel({pc_dim: top_idx})
                        varX = varX[top_idx]
    
                    ev_by_dim = ((Z**2) * xr.DataArray(varX, dims=(Z.dims[-2],))[:, None]).sum(dim=Z.dims[-2])  # (sample,dim)
                    df = ev_by_dim.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by latent dim (top PCs): {u}->{v}"))
    
            # correlation traces if present
            rg = f"RG:{u}->{v}"
            if rg in post:
                col.append(_corr_trace_plot(post[rg], title=f"Corr traces: {u}->{v}"))
    
            if len(col) == 0:
                col.append(pn.pane.Markdown(f"No trace plots available for edge {u}->{v}."))
    
            tabs.append((f"edge: {u}->{v}", col))
    
        # ---- node tabs ----
        for n, a in self.G.nodes(data=True):
            if str(a.get("type")) not in {"latent", "endo"}:
                continue
            # prefer V_pred if available (already per-dim)
            vname = f"V_pred:{n}"
            if vname in post:
                V = _stack_samples(post[vname])  # (sample,dim)
                tabs.append((f"node: {n}", pn.Column(_kde_plus_line(V.to_pandas(), title=f"V_pred traces: {n}"))))
                continue
    
            # fallback: var over animals of node value (or mu+Efac for endo)
            if a["type"] == "endo" and (f"prediction:{n}" in post):
                mu = post[f"prediction:{n}"]
                efac = f"{n}:Efac"
                if efac in post:
                    mu = mu + post[efac]
                Z = _stack_samples(mu)  # (sample,obs,dim)
            else:
                if n not in post:
                    continue
                Z = _stack_samples(post[n])
    
            # var over obs -> (sample,dim)
            var_over_obs = Z.var(dim=Z.dims[-2])
            tabs.append((f"node: {n}", pn.Column(_kde_plus_line(var_over_obs.to_pandas(), title=f"Var over animals: {n}"))))
    
        return pn.Tabs(*tabs)


# ============================================================================
# Covariance-component mediation and causal-network utilities
# ============================================================================

def _np_symmetrize(M: np.ndarray) -> np.ndarray:
    """Coerce to NumPy and symmetrize the square matrix.

    Parameters
    ----------
    M : np.ndarray
        Numerical operand in ``0.5 * (M + M.T)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.
    """
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def _np_project_psd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize a NumPy covariance and floor its eigenvalues to restore positive semidefiniteness.

    Parameters
    ----------
    M : np.ndarray
        Numerical operand in ``_np_symmetrize(M)``. Shape must conform to that matrix/vector
        operation and the enclosing prepared state.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-10``.
    """
    M = _np_symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w[None, :]) @ V.T


def conditional_covariance(C: np.ndarray, conditioned, *, target=None, ridge: float = 1e-8) -> np.ndarray:
    """Compute the ridge-stabilized Schur complement for selected covariance variables.
    
    Parameters
    ----------
    C : array-like, shape (p, p)
        Finite covariance matrix in a fixed variable order. The implementation symmetrizes C; it does not establish PSD or statistical identification.
    
    conditioned : int or integer sequence, length m
        Zero-based indices of conditioning variables. Empty sequence returns the unconditioned target block. Names are unsupported; provide distinct valid indices to avoid unintended repeated variables.
    
    target : int, integer sequence or None
        Default: ``None``. None selects the sorted complement of conditioned; explicit indices preserve requested order. Negative NumPy indices are interpreted positionally, not as names.
    
    ridge : float
        Default: ``1e-08``. Scalar diagonal stabilization on the conditioning block; use a nonnegative value, positive for singular blocks.
    
    Returns
    -------
    numpy.ndarray, shape (k, k)
        Symmetric C_TT - C_TM @ solve(C_MM + ridge I, C_MT), where k is the number of targets. Singular unregularized systems raise numpy.linalg.LinAlgError.
    
    Complexity
    ----------
    TIME O(p²+m³+m²k+mk²); SPACE O(p²+m²+mk+k²). p is all variables, m conditioned variables and k target variables. Empty conditioning avoids the solve.
    
    Notes
    -----
    This is a covariance transformation, not evidence of directional causal mediation. Indices are cast to integers; malformed shapes/indices fail through NumPy rather than a comprehensive validator.
    """
    C = _np_symmetrize(np.asarray(C, dtype=float))
    p = C.shape[0]
    cond = np.atleast_1d(conditioned).astype(int)
    targ = np.setdiff1d(np.arange(p), cond) if target is None else np.atleast_1d(target).astype(int)
    if len(cond) == 0:
        return C[np.ix_(targ, targ)]
    Ctt = C[np.ix_(targ, targ)]
    Ctm = C[np.ix_(targ, cond)]
    Cmm = C[np.ix_(cond, cond)] + float(ridge) * np.eye(len(cond))
    return _np_symmetrize(Ctt - Ctm @ np.linalg.solve(Cmm, Ctm.T))


def genetic_covariance_mediation(
    G_additive: np.ndarray,
    mediators,
    *,
    outcomes=None,
    G_dominance: np.ndarray | None = None,
    E: np.ndarray | None = None,
    names: list[str] | None = None,
    ridge: float = 1e-8,
    save_path=None,
) -> "xarray.Dataset":
    """Fast covariance-level mediation screen from A/D/E trait covariance components.

    This is a conditional-covariance screen, not directional causal mediation. For each
    covariance source S, it reports S_YY|M and the fraction of marginal diagonal variance
    statistically accounted for by the mediator set M.

    Parameters
    ----------
    G_additive : np.ndarray
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    mediators : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering.

    outcomes : str, index or sequence
        Response/trait identifiers selected for this prediction, mediation or subset operation.
        Names/positions must exist in the supplied response axes; sequences preserve the
        requested ordering. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    G_dominance : np.ndarray | None
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : np.ndarray | None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    names : list[str] | None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.

    ridge : float
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``1e-08``. Passed to ``_np_project_psd`` as ``eps``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: the former ``summary`` table along ``record`` (``source``,
        ``outcome``, ``marginal_variance``, ``conditional_variance``,
        ``variance_fraction_accounted``) and the former ``details`` mapping stacked as
        ``details_marginal``, ``details_conditional``, ``details_accounted``
        (component, outcome_i, outcome_j). Attrs: ``mediators``, ``outcomes`` plus
        title/software/parameters provenance.
    """
    import xarray as xr
    sources = {"additive": G_additive}
    if G_dominance is not None:
        sources["dominance"] = G_dominance
    if E is not None:
        sources["environmental"] = E
    p = np.asarray(G_additive).shape[0]
    names = [str(i) for i in range(p)] if names is None else list(map(str, names))
    if len(names) != p:
        raise ValueError("names must match covariance dimension")
    name2idx = {n:i for i,n in enumerate(names)}

    def resolve(x):
        """Return the resolve result computed by ``np.array``.

        Parameters
        ----------
        x : array-like
            Numerical operand in ``np.isscalar(x)``. Shape must conform to that matrix/vector
            operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``np.array([name2idx[str(v)] if isinstance(v, str) else int(v) for v in vals], dtype=int)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        vals = [x] if np.isscalar(x) and not isinstance(x, (list, tuple, np.ndarray, pd.Index)) else list(x)
        return np.array([name2idx[str(v)] if isinstance(v, str) else int(v) for v in vals], dtype=int)

    midx = resolve(mediators)
    oidx = np.setdiff1d(np.arange(p), midx) if outcomes is None else resolve(outcomes)
    tables = []
    details = {}
    for source, C0 in sources.items():
        C = _np_project_psd(C0, eps=ridge)
        marginal = C[np.ix_(oidx, oidx)]
        cond = conditional_covariance(C, midx, target=oidx, ridge=ridge)
        explained = _np_symmetrize(marginal - cond)
        frac = 1.0 - np.divide(np.diag(cond), np.diag(marginal),
                               out=np.full(len(oidx), np.nan), where=np.diag(marginal) > ridge)
        tab = pd.DataFrame({"source": source, "outcome": [names[i] for i in oidx],
                            "marginal_variance": np.diag(marginal),
                            "conditional_variance": np.diag(cond),
                            "variance_fraction_accounted": frac})
        tables.append(tab)
        details[source] = {"marginal": marginal, "conditional": cond, "accounted": explained}
    onames = [names[i] for i in oidx]
    coords = {"component": list(details), "outcome_i": onames, "outcome_j": onames}
    values = _table_vars(pd.concat(tables, ignore_index=True), "summary", "record", long_names={
        "source": "covariance component", "outcome": "outcome trait", "marginal_variance": "marginal outcome variance",
        "conditional_variance": "outcome variance conditional on the mediators",
        "variance_fraction_accounted": "fraction of marginal variance accounted for by the mediators"})
    for part, label in [("marginal", "marginal outcome covariance"), ("conditional", "outcome covariance conditional on the mediators"),
                        ("accounted", "outcome covariance accounted for by the mediators")]:
        values[f"details_{part}"] = xr.DataArray(np.stack([details[k][part] for k in details]), dims=("component", "outcome_i", "outcome_j"),
                                                 coords=coords, attrs={"long_name": label})
    values.update({"mediators": [names[i] for i in midx], "outcomes": onames})
    ds = result_dataset(values, title="Genetic covariance mediation screen", software="vigor.causal.genetic_covariance_mediation",
                        parameters={"ridge": float(ridge), "components": ",".join(details)})
    return maybe_save(ds, save_path)


def _rff_hsic_1d(x, y, *, n_features=64, random_state=0):
    """Linear-time random-Fourier approximation to Gaussian-kernel HSIC.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``np.isfinite(x) & np.isfinite(y)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    y : array-like
        Numerical operand in ``np.isfinite(x) & np.isfinite(y)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    n_features : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``64``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.
    """
    x=np.asarray(x,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    n=len(x)
    if n < 8: return np.nan
    def zscore(a):
        """Compute standardized effect magnitudes relative to their robust central value.

        Parameters
        ----------
        a : array-like
            Numerical operand in ``np.median(np.abs(a - med)) * 1.4826``. Shape must conform to that
            matrix/vector operation and the enclosing prepared state.

        Returns
        -------
        object
            Computed result from ``(a - med) / max(float(scale), 1e-10)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        med=np.median(a); mad=np.median(np.abs(a-med))*1.4826
        scale=mad if np.isfinite(mad) and mad>1e-10 else np.std(a)
        return (a-med)/max(float(scale),1e-10)
    x=zscore(x); y=zscore(y)
    rng=np.random.default_rng(int(random_state))
    d=max(8,int(n_features))
    wx=rng.normal(size=d); wy=rng.normal(size=d)
    bx=rng.uniform(0,2*np.pi,size=d); by=rng.uniform(0,2*np.pi,size=d)
    zx=np.sqrt(2.0/d)*np.cos(x[:,None]*wx[None,:]+bx[None,:])
    zy=np.sqrt(2.0/d)*np.cos(y[:,None]*wy[None,:]+by[None,:])
    zx-=zx.mean(axis=0,keepdims=True); zy-=zy.mean(axis=0,keepdims=True)
    C=(zx.T@zy)/max(n-1,1)
    return float(np.sum(C*C))


def _anm_crossfit_residual(cause, effect, *, n_splits=3, n_knots=6, ridge=1e-3, random_state=0):
    """Fast nonlinear cross-fitted residual for an additive-noise direction.

    Parameters
    ----------
    cause : array-like
        Candidate cause/effect observation vector for additive-noise direction testing. Match
        samples across vectors and use finite jointly observed values; inferred direction is
        conditional on the model and observational assumptions.

    effect : array-like
        Candidate cause/effect observation vector for additive-noise direction testing. Match
        samples across vectors and use finite jointly observed values; inferred direction is
        conditional on the model and observational assumptions.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``.

    n_knots : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``6``. Passed to
        ``SplineTransformer`` as ``n_knots``.

    ridge : float or array-like
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``0.001``. Passed to ``Ridge`` as ``alpha``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``KFold`` as ``random_state``.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    x=np.asarray(cause,float).reshape(-1,1); y=np.asarray(effect,float).reshape(-1)
    n=len(y); pred=np.full(n,np.nan,float)
    ns=max(2,min(int(n_splits),n))
    kf=KFold(n_splits=ns,shuffle=True,random_state=int(random_state))
    for tr,te in kf.split(x):
        model=make_pipeline(
            SplineTransformer(n_knots=max(3,int(n_knots)),degree=3,include_bias=False),
            StandardScaler(), Ridge(alpha=float(ridge))
        )
        model.fit(x[tr],y[tr]); pred[te]=model.predict(x[te])
    return y-pred


def anm_pairwise_direction(
    x, y, *, names=("x","y"), min_samples=100, n_splits=3,
    n_knots=6, regression_ridge=1e-3, n_rff=64, min_ratio=1.20,
    random_state=0, save_path=None,
):
    """Orient one candidate pair using a scalable Additive Noise Model screen.

    Both directions are fit out of fold with a univariate spline-ridge model.
    The proposed cause is compared with the cross-fitted residual using a
    linear-time RFF-HSIC dependence score.  Lower residual dependence supports
    that direction.  ANM is an orientation diagnostic, not proof of causality:
    hidden confounding, feedback and linear-Gaussian systems can be unidentifiable.

    Parameters
    ----------
    x : array-like
        Numerical operand in ``np.isfinite(x) & np.isfinite(y)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    y : array-like
        Numerical operand in ``np.isfinite(x) & np.isfinite(y)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``('x', 'y')``.

    min_samples : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``100``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``_anm_crossfit_residual`` as ``n_splits``.

    n_knots : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``6``. Passed to
        ``_anm_crossfit_residual`` as ``n_knots``.

    regression_ridge : float
        Input supplied to ``_anm_crossfit_residual`` as ``ridge``. See that callable's
        documented ``ridge`` contract for its type, alignment and value domain. This wrapper
        retains that contract. Default: ``0.001``. Passed to ``_anm_crossfit_residual`` as
        ``ridge``.

    n_rff : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``64``. Passed to
        ``_rff_hsic_1d`` as ``n_features``.

    min_ratio : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``1.2``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``_anm_crossfit_residual`` as ``random_state``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        No data variables (a single pair yields scalars only). Attrs: ``source``, ``target``
        (None when undetermined), ``direction``, ``n``, ``score_xy``, ``score_yx``, ``ratio``
        and, depending on the branch, ``reason``, ``normality_note``, ``method``,
        ``interpretation``, plus title/software/parameters provenance.
    """
    res=_anm_pairwise_direction(x,y,names=names,min_samples=min_samples,n_splits=n_splits,n_knots=n_knots,
                                regression_ridge=regression_ridge,n_rff=n_rff,min_ratio=min_ratio,random_state=random_state)
    ds=result_dataset({},attrs=res,title="Additive-noise-model pair orientation",software="vigor.causal.anm_pairwise_direction",
                      parameters={"x_name":str(names[0]),"y_name":str(names[1]),"min_samples":int(min_samples),"n_splits":int(n_splits),
                                  "n_knots":int(n_knots),"regression_ridge":float(regression_ridge),"n_rff":int(n_rff),
                                  "min_ratio":float(min_ratio),"random_state":int(random_state)})
    return maybe_save(ds,save_path)


def _anm_pairwise_direction(x, y, *, names=("x","y"), min_samples=100, n_splits=3, n_knots=6,
                            regression_ridge=1e-3, n_rff=64, min_ratio=1.20, random_state=0):
    """Dict-returning core of :func:`anm_pairwise_direction` (reused per candidate/bootstrap)."""
    x=np.asarray(x,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    n=len(x); a,b=str(names[0]),str(names[1])
    if n < int(min_samples):
        return {"source":None,"target":None,"direction":"undetermined","n":n,
                "score_xy":np.nan,"score_yx":np.nan,"ratio":np.nan,"reason":"insufficient_samples"}
    rx_y=_anm_crossfit_residual(x,y,n_splits=n_splits,n_knots=n_knots,ridge=regression_ridge,random_state=random_state)
    ry_x=_anm_crossfit_residual(y,x,n_splits=n_splits,n_knots=n_knots,ridge=regression_ridge,random_state=random_state+17)
    # Linear Gaussian pairs are not direction-identifiable under ANM. Normal
    # diagnostics support abstention; they do not prove Gaussianity or exclude
    # an unmeasured common cause. The nonlinear controls are cross-fitted above.
    from scipy.stats import normaltest
    slope=np.dot(x-x.mean(),y-y.mean())/max(np.dot(x-x.mean(),x-x.mean()),1e-30)
    linear_residual=y-y.mean()-slope*(x-x.mean())
    if n>=20 and all(float(normaltest(v).pvalue)>.01 for v in (x,y,linear_residual)):
        linear_mse=float(np.mean(linear_residual**2))
        nonlinear_gain=1-float(np.mean(rx_y**2))/max(linear_mse,1e-30)
        if nonlinear_gain<.02:
            return {'source':None,'target':None,'direction':'undetermined_linear_gaussian',
                    'n':n,'score_xy':np.nan,'score_yx':np.nan,'ratio':np.nan,
                    'reason':'linear Gaussian ANM direction is nonidentifiable',
                    'normality_note':'failure to reject Gaussianity is an abstention diagnostic, not proof'}
    sxy=_rff_hsic_1d(x,rx_y,n_features=n_rff,random_state=random_state+101)
    syx=_rff_hsic_1d(y,ry_x,n_features=n_rff,random_state=random_state+101)
    lo=min(sxy,syx); hi=max(sxy,syx); ratio=float((hi+1e-15)/(lo+1e-15))
    if not np.isfinite(ratio) or ratio < float(min_ratio):
        src=tgt=None; direction="undetermined"
    elif sxy < syx:
        src,tgt,direction=a,b,f"{a}->{b}"
    else:
        src,tgt,direction=b,a,f"{b}->{a}"
    return {"source":src,"target":tgt,"direction":direction,"n":n,
            "score_xy":float(sxy),"score_yx":float(syx),"ratio":ratio,
            "method":"crossfit_spline_ridge_rff_hsic",
            "interpretation":"lower cause-residual dependence is preferred"}


def anm_orient_candidates(data, candidate_edges, *, names=None, min_samples=100,
                          n_splits=3, n_knots=6, regression_ridge=1e-3,
                          n_rff=64, min_ratio=1.20, random_state=0,
                          n_bootstrap=0, bootstrap_fraction=0.85,
                          direction_stability=0.75, save_path=None):
    """Run ANM only on a sparse candidate skeleton (PC/precision/SEM edges).

    Parameters
    ----------
    data : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    candidate_edges : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.

    min_samples : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``100``. Passed to ``anm_pairwise_direction`` as
        ``min_samples``.

    n_splits : int
        Number of cross-validation folds, ordinarily at least two and no larger than the
        eligible training sample count. Fold-specific fitting must exclude held-out outcomes.
        Default: ``3``. Passed to ``anm_pairwise_direction`` as ``n_splits``.

    n_knots : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``6``. Passed to
        ``anm_pairwise_direction`` as ``n_knots``.

    regression_ridge : float
        Input supplied to ``anm_pairwise_direction`` as ``regression_ridge``. See that
        callable's documented ``regression_ridge`` contract for its type, alignment and value
        domain. This wrapper retains that contract. Default: ``0.001``. Passed to
        ``anm_pairwise_direction`` as ``regression_ridge``.

    n_rff : int
        Number of basis functions/random Fourier features in the selected nonlinear dependence
        model; positive counts set approximation capacity. Default: ``64``. Passed to
        ``anm_pairwise_direction`` as ``n_rff``.

    min_ratio : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``1.2``. Passed to
        ``anm_pairwise_direction`` as ``min_ratio``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``. Passed to
        ``anm_pairwise_direction`` as ``random_state``.

    n_bootstrap : int
        Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
        estimate sampling variation by the resampling scheme used here. Default: ``0``.

    bootstrap_fraction : float
        Fraction of eligible entries/animals selected for the corresponding holdout or
        resampling operation; use a value in (0, 1] compatible with nonempty training/evaluation
        subsets. Default: ``0.85``.

    direction_stability : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``0.75``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: the former ``orientations`` table along ``pair`` (one variable per
        column: ``source``, ``target``, ``direction``, ``n``, ``score_xy``, ``score_yx``,
        ``ratio``, ``orientation_support``, ``forward_support``, ``reverse_support``,
        ``bootstrap_determined``, ``bootstrap_attempted``, ``bootstrap_abstained`` and the
        per-pair ``method``/``interpretation``/``reason``/``normality_note`` when present).
        Attrs: ``graph`` (networkx.DiGraph), ``method``, ``bootstrap``,
        ``direction_stability_threshold``, ``note`` plus title/software/parameters provenance.
        ``candidate_edges`` may also be a PC result Dataset (its ``source``/``target`` edges).
    """
    if hasattr(data,'to_pandas') and not isinstance(data,(pd.DataFrame,pd.Series)): data=data.to_pandas()
    if hasattr(candidate_edges,'data_vars'):
        candidate_edges=(pd.DataFrame({'source':candidate_edges['source'].values,'target':candidate_edges['target'].values})
                         if {'source','target'}<=set(candidate_edges.data_vars) else candidate_edges.attrs['graph'])
    if isinstance(data,pd.DataFrame):
        X=data
        names=list(map(str,X.columns)) if names is None else list(map(str,names))
    else:
        arr=np.asarray(data,float)
        names=[str(i) for i in range(arr.shape[1])] if names is None else list(map(str,names))
        X=pd.DataFrame(arr,columns=names)
    if isinstance(candidate_edges,nx.Graph):
        pairs=list(candidate_edges.to_undirected().edges())
    elif isinstance(candidate_edges,pd.DataFrame):
        if not {"source","target"}.issubset(candidate_edges.columns):
            raise ValueError("candidate edge table must contain source and target columns")
        pairs=list(candidate_edges[["source","target"]].itertuples(index=False,name=None))
    else:
        pairs=list(candidate_edges)
    seen=set(); rows=[]
    for q,(a,b) in enumerate(pairs):
        a,b=str(a),str(b); key=tuple(sorted((a,b)))
        if a==b or key in seen: continue
        seen.add(key)
        if a not in X.columns or b not in X.columns:
            raise KeyError(f"ANM candidate {a!r},{b!r} not found in data columns")
        base=_anm_pairwise_direction(
            X[a].to_numpy(),X[b].to_numpy(),names=(a,b),min_samples=min_samples,
            n_splits=n_splits,n_knots=n_knots,regression_ridge=regression_ridge,
            n_rff=n_rff,min_ratio=min_ratio,random_state=int(random_state)+7919*q)
        nb=max(0,int(n_bootstrap)); votes=[];attempted=0
        base_direction=None if base.get('source') is None else (str(base['source']),str(base['target']))
        if not nb and base_direction is not None:votes.append(base_direction)
        if nb:
            xa=X[a].to_numpy(float);yb=X[b].to_numpy(float);ok=np.isfinite(xa)&np.isfinite(yb);xa=xa[ok];yb=yb[ok]
            rr=np.random.default_rng(int(random_state)+104729*q+17); take=max(20,min(len(xa),int(round(float(bootstrap_fraction)*len(xa)))))
            for bi in range(nb):
                if len(xa)<max(int(min_samples),20):break
                ii=rr.integers(0,len(xa),size=take)
                z=_anm_pairwise_direction(xa[ii],yb[ii],names=(a,b),min_samples=min(int(min_samples),max(20,take//2)),
                    n_splits=n_splits,n_knots=n_knots,regression_ridge=regression_ridge,n_rff=n_rff,
                    min_ratio=min_ratio,random_state=int(random_state)+104729*q+bi+101)
                attempted+=1
                if z.get('source') is not None:votes.append((str(z['source']),str(z['target'])))
        denom=max(1,attempted if nb else len(votes));ab=sum(v==(a,b) for v in votes)/denom;ba=sum(v==(b,a) for v in votes)/denom;support=max(ab,ba)
        base['orientation_support']=float(support);base['forward_support']=float(ab);base['reverse_support']=float(ba);base['bootstrap_determined']=int(len(votes))
        base['bootstrap_attempted']=attempted;base['bootstrap_abstained']=attempted-len(votes) if nb else 0
        if nb and (support<float(direction_stability) or abs(ab-ba)<1e-12):
            base['source']=None;base['target']=None;base['direction']='undetermined_stability'
        elif nb and base_direction != ((a,b) if ab>ba else (b,a)):
            base['source']=None;base['target']=None;base['direction']='undetermined_base_bootstrap_disagreement'
        rows.append(base)
    table=pd.DataFrame(rows)
    G=nx.DiGraph(); G.add_nodes_from(names)
    if len(table):
        for r in table.itertuples(index=False):
            if r.source is not None and r.target is not None:
                G.add_edge(str(r.source),str(r.target),method="ANM",ratio=float(r.ratio))
    values=_table_vars(table,"orientations","pair",long_names=_ANM_LONG_NAMES); values["graph"]=G
    ds=result_dataset(values,attrs={"method":"additive_noise_model",
            "bootstrap":int(n_bootstrap),"direction_stability_threshold":float(direction_stability),
            "note":"ANM requires individual-level observations or component-specific scores; covariance matrices alone are insufficient. Bootstrap-stable orientation is required when n_bootstrap>0."},
        title="Additive-noise-model candidate orientations",software="vigor.causal.anm_orient_candidates",
        parameters={"min_samples":int(min_samples),"n_splits":int(n_splits),"n_knots":int(n_knots),"regression_ridge":float(regression_ridge),
                    "n_rff":int(n_rff),"min_ratio":float(min_ratio),"random_state":int(random_state),"n_bootstrap":int(n_bootstrap),
                    "bootstrap_fraction":float(bootstrap_fraction),"direction_stability":float(direction_stability)})
    return maybe_save(ds,save_path)


def partial_correlation_from_cov(C: np.ndarray, i: int, j: int, conditioned=(), *, ridge: float = 1e-8) -> float:
    """Calculate a Gaussian partial correlation using a stabilized precision submatrix.
    
    Parameters
    ----------
    C : array-like, shape (p, p)
        Covariance matrix in variable order; only the requested submatrix is used and projected to PSD.
    
    i : int
        Zero-based first variable index.
    
    j : int
        Zero-based second variable index; normally distinct from i.
    
    conditioned : iterable of int
        Default: ``()``. Conditioning indices; occurrences of i/j are removed. Other duplicates are not deduplicated.
    
    ridge : float
        Default: ``1e-08``. Eigenvalue floor and added diagonal ridge on the selected covariance; use positive finite stabilization.
    
    Returns
    -------
    float
        Precision-derived partial correlation clipped to [-1,1]. It is descriptive; sampling calibration needs additional assumptions.
    
    Complexity
    ----------
    TIME O(q³); SPACE O(q²), for q=2 plus conditioning entries after removing i/j. The eigendecomposition/inversion are dense.
    """
    idx = [int(i), int(j)] + [int(k) for k in conditioned if int(k) not in {int(i), int(j)}]
    S = _np_project_psd(np.asarray(C, dtype=float)[np.ix_(idx, idx)], eps=ridge)
    P = np.linalg.inv(S + float(ridge) * np.eye(len(idx)))
    den = np.sqrt(max(P[0,0] * P[1,1], ridge))
    return float(np.clip(-P[0,1] / den, -1.0, 1.0))


def _pc_independence_test(C, i, j, conditioned, *, alpha=.01, n_eff=None,
                          bootstrap_covs=None, ridge=1e-8, data=None):
    """PC CI test with optional test-wise complete observations.

    When ``data`` is supplied, every CI test is estimated only from animals
    jointly observed for ``i``, ``j`` and the conditioning set.  This is the
    correct Fisher-z sample size under arbitrary phenotype missingness and
    avoids treating a single global/pairwise overlap as the sample size of a
    higher-order partial correlation.  Insufficient joint observations keep
    the edge (conservative) rather than creating a false conditional
    independence.

    Parameters
    ----------
    C : array-like
        Numerical operand in ``partial_correlation_from_cov(C, i, j, conditioned,
        ridge=ridge)``. Shape must conform to that matrix/vector operation and the enclosing
        prepared state.

    i : int or index array
        Local position, index selection or step used by the stated slice/iteration. Indices must
        refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
        the enclosing dimensions.

    j : int or index array
        Local position, index selection or step used by the stated slice/iteration. Indices must
        refer to the relevant axis; slice steps must be nonzero, and count/rank uses must match
        the enclosing dimensions.

    conditioned : object (see the cited input contract)
        Input for the operation ``[int(i), int(j)] + [int(k) for k in conditioned]``. See
        ``vigor.causal._pc_independence_test`` and its caller for the scientific representation;
        the shown operation constrains the required type/shape.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.01``.

    n_eff : float, array-like or None
        Effective sample count(s) for covariance/conditional-independence calibration. Pair
        matrices match variable-pair order; None delegates to the routine's
        covariance-bootstrap/sample-count policy. Default: ``None``.

    bootstrap_covs : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    ridge : float or array-like
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``1e-08``. Passed to ``partial_correlation_from_cov`` as
        ``ridge``.

    data : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.
    """
    n_local = np.nan
    if data is not None:
        X=np.asarray(data,float); idx=[int(i),int(j)]+[int(k) for k in conditioned]
        ok=np.all(np.isfinite(X[:,idx]),axis=1); n_local=int(ok.sum())
        dof=float(n_local)-len(conditioned)-3.0
        if dof <= 0 or n_local < max(4,len(conditioned)+4):
            return False, np.nan, np.nan, np.nan, np.nan, float(n_local)
        S=np.cov(X[np.ix_(ok,idx)],rowvar=False,ddof=1)
        if np.ndim(S)<2 or not np.all(np.isfinite(S)):
            return False, np.nan, np.nan, np.nan, np.nan, float(n_local)
        r=partial_correlation_from_cov(S,0,1,tuple(range(2,len(idx))),ridge=ridge)
        z=np.arctanh(np.clip(r,-0.999999,0.999999))*np.sqrt(dof)
        p=float(2.0*norm.sf(abs(z)))
        return bool(p>float(alpha)),r,p,np.nan,np.nan,float(n_local)
    r = partial_correlation_from_cov(C, i, j, conditioned, ridge=ridge)
    if bootstrap_covs is not None and len(bootstrap_covs):
        rb = np.array([partial_correlation_from_cov(B, i, j, conditioned, ridge=ridge) for B in bootstrap_covs], float)
        q = float(alpha) / 2.0
        lo, hi = np.quantile(rb, [q, 1.0-q])
        independent = bool(lo <= 0.0 <= hi)
        p = float(min(1.0, 2.0 * min((rb <= 0).mean(), (rb >= 0).mean())))
        return independent, r, p, float(lo), float(hi), float(len(rb))
    if n_eff is None:
        raise ValueError("PC conditional-independence tests require n_eff, bootstrap_covs, or data")
    dof = float(n_eff) - len(conditioned) - 3.0
    if dof <= 0:
        return False, r, np.nan, np.nan, np.nan, float(n_eff)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(dof)
    p = float(2.0 * norm.sf(abs(z)))
    return bool(p > float(alpha)), r, p, np.nan, np.nan, float(n_eff)


_ANM_LONG_NAMES = {
    "source": "inferred cause", "target": "inferred effect", "direction": "orientation decision",
    "n": "jointly observed samples", "score_xy": "RFF-HSIC cause-residual dependence for x->y",
    "score_yx": "RFF-HSIC cause-residual dependence for y->x", "ratio": "dependence-score asymmetry ratio",
    "orientation_support": "bootstrap support of the majority direction", "forward_support": "bootstrap support of x->y",
    "reverse_support": "bootstrap support of y->x", "bootstrap_determined": "bootstrap runs with a determined direction",
    "bootstrap_attempted": "bootstrap runs attempted", "bootstrap_abstained": "bootstrap runs that abstained",
}
_PC_LONG_NAMES = {
    "source": "edge source node", "target": "edge target node", "orientation": "edge orientation (directed/undirected)",
    "x": "first tested variable", "y": "second tested variable", "conditioned": "conditioning set",
    "partial_r": "partial correlation", "p": "conditional-independence p-value",
    "ci_low": "bootstrap partial-correlation lower bound", "ci_high": "bootstrap partial-correlation upper bound",
    "independent": "conditional independence accepted", "alpha_local": "per-test significance level",
    "alpha_effective": "per-test significance level (compatibility alias of alpha_local)",
    "depth": "conditioning-set size", "n_eff_test": "sample size of the CI test",
    "edge_stability": "bootstrap adjacency frequency", "direction_stability": "bootstrap support of the consensus direction",
    "forward_stability": "bootstrap support of source->target", "reverse_stability": "bootstrap support of target->source",
    "consensus_orientation": "consensus orientation", "source_covariance": "covariance component of the network",
    "n_sources": "number of covariance components supporting the pair", "sources": "supporting covariance components",
    "node1": "first node of the pair", "node2": "second node of the pair",
}


def _pc_dataset(res, *, software, parameters, title="PC-stable partially directed causal graph"):
    """Convert an internal PC result dict (``_pc_from_covariance``/stability) into a Dataset.

    ``adjacency`` -> (node_i, node_j); ``edges`` -> ``edge``; ``tests`` -> ``test``;
    ``stability`` -> ``pair`` with ``pair_`` prefix; all other keys -> attrs.
    """
    res = dict(res); names = res.pop("_names")
    tables = {"edges": ("edge", ""), "tests": ("test", ""), "stability": ("pair", "pair_")}
    values = {k: v for k, v in res.items() if k not in tables and k != "adjacency"}
    values["adjacency"] = pd.DataFrame(np.asarray(res["adjacency"], bool), index=names, columns=names)
    for key, (dim, prefix) in tables.items():
        if key in res:
            values.update(_table_vars(res[key], key, dim, prefix=prefix, long_names=_PC_LONG_NAMES))
    return result_dataset(values, dims={"adjacency": ("node_i", "node_j")},
                          var_attrs={"adjacency": {"long_name": "PC skeleton adjacency",
                                                   "description": "True where the conditional-independence skeleton retains the pair"}},
                          title=title, software=software, parameters=parameters)


def pc_from_covariance(
    covariance: np.ndarray, *, names: list[str] | None = None, alpha: float = .01,
    n_eff: float | None = None, bootstrap_covariances=None, max_cond_set: int = 3,
    ridge: float = 1e-8, required_edges=None, forbidden_edges=None,
    multiple_testing: str = "bonferroni", orientation_evidence=None, initial_graph=None,
    data=None, save_path=None,
) -> "xarray.Dataset":
    """PC-stable skeleton + Meek orientation from a covariance matrix.

    Adjacencies are frozen within each conditioning depth (PC-stable), eliminating
    order dependence from in-place edge deletion.  ``multiple_testing`` controls
    family-wise CI-test multiplicity within each depth; ``bonferroni`` is the
    publication-safe default.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance quantity or component collection consumed by the stated transformation.
        Square matrices use a common trait order; uncertainty matrices match their corresponding
        covariance entries. See the caller for whether this is genetic, residual or total
        covariance.

    names : list[str] | None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``. Passed to ``sorted`` as ``key``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.01``.

    n_eff : float | None
        Effective sample count(s) for covariance/conditional-independence calibration. Pair
        matrices match variable-pair order; None delegates to the routine's
        covariance-bootstrap/sample-count policy. Default: ``None``. Passed to
        ``_pc_independence_test`` as ``n_eff``.

    bootstrap_covariances : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    max_cond_set : int
        Largest conditioning-set size examined by PC independence tests; nonnegative integers
        limit combinatorial search. Default: ``3``.

    ridge : float
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``1e-08``. Passed to ``_np_project_psd`` as ``eps``.

    required_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``.

    forbidden_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``.

    multiple_testing : str
        'none' uses alpha for each independence test; 'bonferroni' uses alpha divided by the
        test count at the current conditioning depth. These are the only accepted spellings
        after lowercasing. Default: ``'bonferroni'``.

    orientation_evidence : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    initial_graph : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    data : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``adjacency`` (node_i, node_j); the former ``edges`` table along
        ``edge`` (``source``, ``target``, ``orientation``); the former ``tests`` table along
        ``test`` (``x``, ``y``, ``conditioned``, ``partial_r``, ``p``, ``ci_low``,
        ``ci_high``, ``independent``, ``alpha_local``, ``alpha_effective``, ``depth``,
        ``n_eff_test``). Attrs: ``graph`` (networkx.DiGraph), ``sepsets``, ``alpha``,
        ``multiple_testing``, ``pc_stable``, ``max_cond_set``,
        ``orientation_evidence_applied``, ``initial_graph_screened``,
        ``ambiguous_unshielded_triples``, ``testwise_complete`` plus title/software/parameters
        provenance. ``orientation_evidence`` may be an :func:`anm_orient_candidates` Dataset.
    """
    res=_pc_from_covariance(covariance,names=names,alpha=alpha,n_eff=n_eff,bootstrap_covariances=bootstrap_covariances,
                            max_cond_set=max_cond_set,ridge=ridge,required_edges=required_edges,forbidden_edges=forbidden_edges,
                            multiple_testing=multiple_testing,orientation_evidence=orientation_evidence,
                            initial_graph=initial_graph,data=data)
    return maybe_save(_pc_dataset(res,software="vigor.causal.pc_from_covariance",
                                  parameters={"alpha":float(alpha),"n_eff":None if n_eff is None else float(n_eff),
                                              "max_cond_set":int(max_cond_set),"ridge":float(ridge),
                                              "multiple_testing":str(multiple_testing),"data_supplied":data is not None}),save_path)


def _pc_from_covariance(covariance, *, names=None, alpha=.01, n_eff=None, bootstrap_covariances=None,
                        max_cond_set=3, ridge=1e-8, required_edges=None, forbidden_edges=None,
                        multiple_testing="bonferroni", orientation_evidence=None, initial_graph=None, data=None):
    """Dict-returning core of :func:`pc_from_covariance` (reused by stability/GE wrappers)."""
    C=_np_project_psd(covariance,eps=ridge); p=C.shape[0]
    names=[str(i) for i in range(p)] if names is None else list(map(str,names))
    Xdata=None if data is None else (data.to_numpy(float) if isinstance(data,pd.DataFrame) else np.asarray(data,float))
    if Xdata is not None and (Xdata.ndim!=2 or Xdata.shape[1]!=p):
        raise ValueError("data must be animals x variables and match covariance dimension")
    if len(names)!=p: raise ValueError("names must match covariance dimension")
    boot=None if bootstrap_covariances is None else [np.asarray(x,float) for x in bootstrap_covariances]
    if initial_graph is None:
        adj=np.ones((p,p),dtype=bool); np.fill_diagonal(adj,False)
    elif isinstance(initial_graph,nx.Graph):
        adj=np.zeros((p,p),dtype=bool); idx={n:i for i,n in enumerate(names)}
        for a,b in initial_graph.to_undirected().edges():
            a,b=str(a),str(b)
            if a in idx and b in idx and a!=b: adj[idx[a],idx[b]]=adj[idx[b],idx[a]]=True
    else:
        adj=np.asarray(initial_graph,dtype=bool).copy()
        if adj.shape!=(p,p): raise ValueError("initial_graph adjacency must be p x p")
        adj=np.logical_or(adj,adj.T); np.fill_diagonal(adj,False)
    sepset={}; tests=[]
    forbidden=set()
    for a,b in ([] if forbidden_edges is None else forbidden_edges):
        ia=names.index(str(a)) if isinstance(a,str) else int(a); ib=names.index(str(b)) if isinstance(b,str) else int(b)
        forbidden.add((ia,ib)); adj[ia,ib]=adj[ib,ia]=False
    mt=str(multiple_testing).lower()
    if mt not in {"none","bonferroni"}: raise ValueError("multiple_testing must be 'none' or 'bonferroni'")
    for depth in range(int(max_cond_set)+1):
        frozen=adj.copy(); edges=[(i,j) for i in range(p) for j in range(i+1,p) if frozen[i,j]]
        tasks=[]
        for i,j in edges:
            conditions=set()
            for endpoint,other in [(i,j),(j,i)]:
                candidates=[k for k in np.flatnonzero(frozen[endpoint]).tolist() if k!=other]
                conditions.update(frozenset(cond) for cond in combinations(candidates,depth))
            ordered=sorted(conditions,key=lambda cond:tuple(sorted(names[k] for k in cond)))
            tasks.extend((i,j,tuple(sorted(cond,key=lambda k:names[k]))) for cond in ordered)
        if not tasks:
            if depth>0: break
            continue
        local_alpha=float(alpha)/(len(tasks) if mt=='bonferroni' else 1.0)
        removals={}
        for i,j,cond in tasks:
            if (i,j) in removals: continue
            indep,r,pval,lo,hi,ntest=_pc_independence_test(C,i,j,cond,alpha=local_alpha,n_eff=n_eff,bootstrap_covs=boot,ridge=ridge,data=Xdata)
            tests.append((names[i],names[j],tuple(names[k] for k in cond),r,pval,lo,hi,indep,local_alpha,local_alpha,depth,ntest))
            if indep: removals[(i,j)]=set(cond)
        for (i,j),cond in removals.items(): adj[i,j]=adj[j,i]=False; sepset[(i,j)]=sepset[(j,i)]=cond
        if depth>0 and max(adj.sum(axis=1),default=0)<=depth: break
    P=nx.DiGraph(); P.add_nodes_from(names)
    for i in range(p):
        for j in range(i+1,p):
            if adj[i,j]: P.add_edge(names[i],names[j]); P.add_edge(names[j],names[i])
    def adjacent(a,b): 
        """Test whether two nodes are adjacent in the current partial graph.

        Parameters
        ----------
        a : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        b : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        Returns
        -------
        object
            Computed result from ``P.has_edge(a, b) or P.has_edge(b, a)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return P.has_edge(a,b) or P.has_edge(b,a)
    def undirected(a,b): 
        """Test whether the partial graph retains both directions for an unresolved adjacency.

        Parameters
        ----------
        a : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        b : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        Returns
        -------
        object
            Computed result from ``P.has_edge(a, b) and P.has_edge(b, a)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return P.has_edge(a,b) and P.has_edge(b,a)
    def directed(a,b): 
        """Test whether the partial graph contains the proposed directed adjacency.

        Parameters
        ----------
        a : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        b : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        Returns
        -------
        object
            Computed result from ``P.has_edge(a, b) and (not P.has_edge(b, a))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return P.has_edge(a,b) and not P.has_edge(b,a)
    def orient(a,b):
        """Orient a candidate adjacency in the partial graph using the enclosing orientation rules.

        Parameters
        ----------
        a : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.

        b : hashable node identifier
            Graph node identifier used for adjacency/orientation lookup. It must identify a node in
            the enclosing graph; this input is a label, not a numerical vector.
        """
        if P.has_edge(b,a): P.remove_edge(b,a)
        if not P.has_edge(a,b): P.add_edge(a,b)
    ambiguous_colliders=[]
    for k in range(p):
        nb=np.flatnonzero(adj[k])
        for i,j in combinations(nb,2):
            if adj[i,j]: continue
            if (i,j) not in sepset:
                ambiguous_colliders.append((names[i],names[k],names[j]))
                continue
            if k not in sepset[(i,j)]: orient(names[i],names[k]); orient(names[j],names[k])
    # Meek R1-R3; repeat to closure.
    changed=True
    while changed:
        changed=False
        # R1: a->b, b-c, a !~ c => b->c
        for a,b in list(P.edges()):
            if not directed(a,b): continue
            for c in names:
                if c in {a,b} or not undirected(b,c) or adjacent(a,c): continue
                orient(b,c); changed=True
        # R2: a-b and a->c->b => a->b
        for a,b in [(names[i],names[j]) for i in range(p) for j in range(i+1,p) if undirected(names[i],names[j])]:
            for c in names:
                if c not in {a,b} and directed(a,c) and directed(c,b): orient(a,b); changed=True; break
                if c not in {a,b} and directed(b,c) and directed(c,a): orient(b,a); changed=True; break
        # R3: a-b, a-c, a-d, c->b, d->b and c !~ d => a->b
        for a,b in [(names[i],names[j]) for i in range(p) for j in range(i+1,p) if undirected(names[i],names[j])]:
            cand=[c for c in names if c not in {a,b} and undirected(a,c) and directed(c,b)]
            if any(not adjacent(c,d) for c,d in combinations(cand,2)): orient(a,b); changed=True; continue
            cand=[c for c in names if c not in {a,b} and undirected(b,c) and directed(c,a)]
            if any(not adjacent(c,d) for c,d in combinations(cand,2)): orient(b,a); changed=True
        # R4: a-b, a-c, c->d->b, c !~ b => a->b (and symmetric case).
        for a,b in [(names[i],names[j]) for i in range(p) for j in range(i+1,p) if undirected(names[i],names[j])]:
            done=False
            for c in names:
                if c in {a,b} or not undirected(a,c) or adjacent(c,b): continue
                for d in names:
                    if d not in {a,b,c} and directed(c,d) and directed(d,b):
                        orient(a,b); changed=True; done=True; break
                if done: break
            if done: continue
            for c in names:
                if c in {a,b} or not undirected(b,c) or adjacent(c,a): continue
                for d in names:
                    if d not in {a,b,c} and directed(c,d) and directed(d,a):
                        orient(b,a); changed=True; done=True; break
                if done: break
    # Optional asymmetric orientation evidence (e.g. ANM) is deliberately
    # applied only to edges still undirected after collider + Meek rules; it
    # never overrides conditional-independence orientations.
    anm_oriented=[]
    if orientation_evidence is not None:
        if isinstance(orientation_evidence,dict) and "orientations" in orientation_evidence:
            oe=orientation_evidence["orientations"]
        elif hasattr(orientation_evidence,"data_vars"):
            oe=_result_table(orientation_evidence,"orientations")
        else:
            oe=orientation_evidence
        if isinstance(oe,pd.DataFrame):
            for rr in oe.itertuples(index=False):
                a=getattr(rr,"source",None); b=getattr(rr,"target",None)
                if a is None or b is None: continue
                a,b=str(a),str(b)
                if a in P and b in P and undirected(a,b):
                    orient(a,b); anm_oriented.append((a,b))
        elif isinstance(oe,nx.DiGraph):
            for a,b in oe.edges():
                a,b=str(a),str(b)
                if a in P and b in P and undirected(a,b):
                    orient(a,b); anm_oriented.append((a,b))
    for a,b in ([] if required_edges is None else required_edges): orient(str(a),str(b))
    for a,b in ([] if forbidden_edges is None else forbidden_edges):
        a,b=str(a),str(b)
        if P.has_edge(a,b): P.remove_edge(a,b)
    rows=[];seen=set()
    for a,b in P.edges():
        if (b,a) in seen: continue
        if P.has_edge(b,a): rows.append((a,b,'undirected'));seen|={(a,b),(b,a)}
        else: rows.append((a,b,'directed'));seen.add((a,b))
    tests_df=pd.DataFrame(tests,columns=['x','y','conditioned','partial_r','p','ci_low','ci_high','independent','alpha_local','alpha_effective','depth','n_eff_test'])
    # Compatibility alias retained for older audit code; alpha_local is the canonical name.
    tests_df['alpha_effective']=tests_df['alpha_local']
    return {"graph":P,"adjacency":adj,"edges":pd.DataFrame(rows,columns=['source','target','orientation']),
            "sepsets":{(names[i],names[j]):tuple(names[k] for k in v) for (i,j),v in sepset.items() if i<j},
            "tests":tests_df,
            "alpha":float(alpha),"multiple_testing":mt,"pc_stable":True,"max_cond_set":int(max_cond_set),
            "orientation_evidence_applied":anm_oriented,"initial_graph_screened":initial_graph is not None,
            "ambiguous_unshielded_triples":ambiguous_colliders,
            "testwise_complete":Xdata is not None,"_names":names}


def pc_from_data(data, *, names=None, min_pair_observed=4, save_path=None, **kwargs):
    """PC-stable using raw observations and test-wise complete CI tests.

    Pairwise covariance is used only as a numerical/orientation scaffold; each
    skeleton CI test recomputes the relevant covariance on animals jointly
    observed for the tested pair plus conditioning set.

    Parameters
    ----------
    data : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``. Passed to ``pc_from_covariance`` as ``names``.

    min_pair_observed : int
        Observation/sample-count threshold for the stage described above. Use nonnegative
        counts; the source comparison below defines whether it is a minimum support requirement
        or a full-matrix size cutoff. Default: ``4``. Passed to ``df.cov`` as ``min_periods``.

    **kwargs : mapping
        Additional keyword arguments consumed by the compatibility/callback logic in this body.
        See the explicit keyword lookups below for accepted keys; unused keys are subject to
        that compatibility policy.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Same layout as :func:`pc_from_covariance` (``testwise_complete`` is True).
    """
    res=_pc_from_data(data,names=names,min_pair_observed=min_pair_observed,**kwargs)
    params={"min_pair_observed":int(min_pair_observed),
            **{k:v for k,v in kwargs.items() if isinstance(v,(str,bool,int,float))}}
    return maybe_save(_pc_dataset(res,software="vigor.causal.pc_from_data",parameters=params),save_path)


def _pc_from_data(data, *, names=None, min_pair_observed=4, **kwargs):
    """Dict-returning core of :func:`pc_from_data`."""
    if hasattr(data,'to_pandas') and not isinstance(data,(pd.DataFrame,pd.Series)): data=data.to_pandas()
    if isinstance(data,pd.DataFrame):
        X=data.to_numpy(float); names=list(map(str,data.columns)) if names is None else list(map(str,names))
    else:
        X=np.asarray(data,float); names=[str(i) for i in range(X.shape[1])] if names is None else list(map(str,names))
    if X.ndim!=2 or X.shape[1]!=len(names): raise ValueError('data must be animals x variables')
    df=pd.DataFrame(X,columns=names); C=df.cov(min_periods=max(2,int(min_pair_observed))).to_numpy(float)
    diag=np.nanvar(X,axis=0,ddof=1); C=np.where(np.isfinite(C),C,0.0)
    for j in range(len(names)): C[j,j]=diag[j] if np.isfinite(diag[j]) and diag[j]>0 else 1e-8
    C=.5*(C+C.T)
    kw=dict(kwargs); kw.pop('n_eff',None); kw.pop('bootstrap_covariances',None)
    return _pc_from_covariance(C,names=names,n_eff=max(4,X.shape[0]),data=X,**kw)


def pc_stability_from_data(data, *, names=None, n_boot=16, sample_fraction=0.85,
                           edge_threshold=0.65, direction_threshold=0.75,
                           random_state=0, required_edges=None, forbidden_edges=None,
                           save_path=None, **pc_kwargs):
    """Bootstrap-stability PC graph under arbitrary phenotype missingness.

    Edges are retained by adjacency frequency. A direction is asserted only if
    the same orientation is supported in at least ``direction_threshold`` of
    bootstrap runs where the adjacency survived; otherwise the edge remains
    explicitly unresolved. Directed cycles in the consensus are broken by
    demoting the least-stable direction to an undirected adjacency.

    Parameters
    ----------
    data : array-like, table or labeled object
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    names : sequence of str or None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``. Passed to ``pc_from_data`` as ``names``.

    n_boot : int
        Number of bootstrap replicates. Zero disables bootstrap when supported; positive counts
        estimate sampling variation by the resampling scheme used here. Default: ``16``.

    sample_fraction : float
        Fraction of eligible entries/animals selected for the corresponding holdout or
        resampling operation; use a value in (0, 1] compatible with nonempty training/evaluation
        subsets. Default: ``0.85``.

    edge_threshold : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``0.65``.

    direction_threshold : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``0.75``.

    random_state : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``0``.

    required_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``. Passed to ``pc_from_data`` as ``required_edges``.

    forbidden_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``. Passed to ``pc_from_data`` as ``forbidden_edges``.

    **pc_kwargs : dict
        Input supplied to ``pc_from_data`` as ``**expanded mapping``. See that callable's
        documented ``**expanded mapping`` contract for its type, alignment and value domain.
        This wrapper retains that contract. Passed to ``pc_from_data`` as ``**expanded
        mapping``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        :func:`pc_from_covariance` layout for the full-data run, with the consensus graph in
        ``attrs['graph']``; the ``edge`` table gains ``edge_stability``/``direction_stability``;
        the former ``stability`` table is stored along ``pair`` as ``pair_source``,
        ``pair_target``, ``pair_edge_stability``, ``pair_forward_stability``,
        ``pair_reverse_stability``, ``pair_consensus_orientation``,
        ``pair_direction_stability``. Extra attrs: ``bootstrap_runs``,
        ``edge_stability_threshold``, ``direction_stability_threshold``, ``method``.
    """
    if hasattr(data,'to_pandas') and not isinstance(data,(pd.DataFrame,pd.Series)): data=data.to_pandas()
    if isinstance(data,pd.DataFrame):
        X=data.copy(); names=list(map(str,X.columns)) if names is None else list(map(str,names)); X.columns=names
    else:
        A=np.asarray(data,float); names=[str(i) for i in range(A.shape[1])] if names is None else list(map(str,names)); X=pd.DataFrame(A,columns=names)
    base=_pc_from_data(X,names=names,required_edges=required_edges,forbidden_edges=forbidden_edges,**pc_kwargs)
    nb=max(0,int(n_boot)); rng=np.random.default_rng(int(random_state)); runs=[base]
    n=len(X); take=max(4,min(n,int(round(float(sample_fraction)*n))))
    for b in range(nb):
        ii=rng.integers(0,n,size=take)
        runs.append(_pc_from_data(X.iloc[ii].reset_index(drop=True),names=names,
                                 required_edges=required_edges,forbidden_edges=forbidden_edges,**pc_kwargs))
    p=len(names); idx={x:i for i,x in enumerate(names)}; adj=np.zeros((p,p),int); dirs=np.zeros((p,p),int)
    for r in runs:
        G=r['graph']
        for i in range(p):
            for j in range(i+1,p):
                a,b=names[i],names[j]; ab=G.has_edge(a,b); ba=G.has_edge(b,a)
                if ab or ba:
                    adj[i,j]+=1; adj[j,i]+=1
                    if ab and not ba: dirs[i,j]+=1
                    elif ba and not ab: dirs[j,i]+=1
    denom=float(len(runs)); P=nx.DiGraph();P.add_nodes_from(names); stab=[]
    forbidden={tuple(sorted((str(a),str(b)))) for a,b in ([] if forbidden_edges is None else forbidden_edges)}
    for i in range(p):
        for j in range(i+1,p):
            a,b=names[i],names[j]; ef=adj[i,j]/denom
            if tuple(sorted((a,b))) in forbidden or ef<float(edge_threshold): continue
            dab=dirs[i,j]/max(adj[i,j],1); dba=dirs[j,i]/max(adj[i,j],1)
            if dab>=float(direction_threshold) and dab>dba: P.add_edge(a,b); orient=f'{a}->{b}'; ds=dab
            elif dba>=float(direction_threshold) and dba>dab: P.add_edge(b,a); orient=f'{b}->{a}'; ds=dba
            else: P.add_edge(a,b);P.add_edge(b,a);orient='undirected';ds=max(dab,dba)
            stab.append((a,b,ef,dab,dba,orient,ds))
    for a,b in ([] if required_edges is None else required_edges):
        a,b=str(a),str(b)
        if P.has_edge(b,a):P.remove_edge(b,a)
        P.add_edge(a,b)
    # Preserve a CPDAG-like consensus: if stable directions form a cycle,
    # demote the weakest directed edge(s) rather than inventing a causal order.
    while True:
        D=nx.DiGraph();D.add_nodes_from(names)
        for a,b in P.edges():
            if not P.has_edge(b,a):D.add_edge(a,b)
        cyc=next(nx.simple_cycles(D),None)
        if not cyc: break
        ce=list(zip(cyc,cyc[1:]+cyc[:1])); score={}
        for a,b in ce:
            ia,ib=idx[a],idx[b]; score[(a,b)]=dirs[ia,ib]/max(adj[ia,ib],1)
        a,b=min(ce,key=lambda e:score[e]); P.add_edge(b,a)
    rows=[];seen=set()
    stable_lookup={tuple(sorted((a,b))):(ef,dab,dba,ds) for a,b,ef,dab,dba,_,ds in stab}
    for a,b in P.edges():
        key=tuple(sorted((a,b)))
        if key in seen:continue
        ef,dab,dba,ds=stable_lookup.get(key,(1.0,np.nan,np.nan,1.0))
        if P.has_edge(b,a): rows.append((a,b,'undirected',ef,ds))
        else: rows.append((a,b,'directed',ef,ds))
        seen.add(key)
    out=dict(base); out['graph']=P; out['edges']=pd.DataFrame(rows,columns=['source','target','orientation','edge_stability','direction_stability'])
    out['stability']=pd.DataFrame(stab,columns=['source','target','edge_stability','forward_stability','reverse_stability','consensus_orientation','direction_stability'])
    out['bootstrap_runs']=len(runs); out['edge_stability_threshold']=float(edge_threshold); out['direction_stability_threshold']=float(direction_threshold)
    out['method']='pc_stable_testwise_complete_bootstrap'
    params={"n_boot":int(n_boot),"sample_fraction":float(sample_fraction),"edge_threshold":float(edge_threshold),
            "direction_threshold":float(direction_threshold),"random_state":int(random_state),
            **{k:v for k,v in pc_kwargs.items() if isinstance(v,(str,bool,int,float))}}
    return maybe_save(_pc_dataset(out,software="vigor.causal.pc_stability_from_data",parameters=params,
                                  title="Bootstrap-stable PC causal graph"),save_path)


def discover_ge_network(
    G_additive: np.ndarray,
    *,
    G_dominance: np.ndarray | None = None,
    E: np.ndarray | None = None,
    names: list[str] | None = None,
    bootstrap_G_additive=None,
    bootstrap_G_dominance=None,
    bootstrap_E=None,
    alpha: float = .01,
    n_eff: float | None = None,
    max_cond_set: int = 3,
    required_edges=None,
    forbidden_edges=None,
    save_path=None,
) -> "xarray.Dataset":
    """Discover separate additive-genetic, dominance-genetic and environmental PC networks.

    Parameters
    ----------
    G_additive : np.ndarray
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result.

    G_dominance : np.ndarray | None
        Genetic trait covariance, in the same trait order as the observations; a dense matrix is
        n_traits × n_traits. Multi-kernel prediction accepts matching named components or the
        covariance-fit result. Default: ``None``. None delegates to the local fallback shown in
        this body; it is not an additional numeric/category value.

    E : np.ndarray | None
        Residual trait covariance in observation-column order, including cross-trait residual
        dependence. Prediction can recover this component from a covariance-fit result when None
        is supplied; a separate genetic matrix requires an explicit compatible residual
        covariance. Default: ``None``.

    names : list[str] | None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``. Passed to ``pc_from_covariance`` as ``names``.

    bootstrap_G_additive : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    bootstrap_G_dominance : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    bootstrap_E : graph, mapping, sequence or None
        Initial adjacency/orientation evidence or resampled covariance matrices for graph
        discovery. Matrices share the variable order of the primary covariance; None omits this
        source of constraints/evidence. Default: ``None``.

    alpha : float
        Regularization coefficient, mixing weight or statistical significance threshold selected
        by this estimator; the local validation and objective determine its precise scale.
        Default: ``0.01``. Passed to ``pc_from_covariance`` as ``alpha``.

    n_eff : float | None
        Effective sample count(s) for covariance/conditional-independence calibration. Pair
        matrices match variable-pair order; None delegates to the routine's
        covariance-bootstrap/sample-count policy. Default: ``None``. Passed to
        ``pc_from_covariance`` as ``n_eff``.

    max_cond_set : int
        Largest conditioning-set size examined by PC independence tests; nonnegative integers
        limit combinatorial search. Default: ``3``. Passed to ``pc_from_covariance`` as
        ``max_cond_set``.

    required_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``. Passed to ``pc_from_covariance`` as
        ``required_edges``.

    forbidden_edges : sequence of pairs or None
        Trait-name/index pairs that must be retained or excluded respectively during graph
        refinement. None imposes no such constraints; required directions must remain compatible
        with acyclicity. Default: ``None``. Passed to ``pc_from_covariance`` as
        ``forbidden_edges``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: the former ``edge_support`` table along ``edge`` (``source``,
        ``target``, ``orientation``, ``source_covariance``) and the former ``consensus`` table
        along ``pair`` (``n_sources``, ``sources``, ``node1``, ``node2``). Attrs: ``networks``
        (mapping component -> :func:`pc_from_covariance` Dataset) plus
        title/software/parameters provenance.
    """
    sources = {"additive":(G_additive,bootstrap_G_additive)}
    if G_dominance is not None: sources["dominance"]=(G_dominance,bootstrap_G_dominance)
    if E is not None: sources["environmental"]=(E,bootstrap_E)
    networks={}
    support=[]
    for source,(C,B) in sources.items():
        networks[source]=_pc_from_covariance(C,names=names,alpha=alpha,n_eff=n_eff,
                                            bootstrap_covariances=B,max_cond_set=max_cond_set,
                                            required_edges=required_edges,forbidden_edges=forbidden_edges)
        et=networks[source]['edges'].copy();et['source_covariance']=source;support.append(et)
    support=pd.concat(support,ignore_index=True) if support else pd.DataFrame()
    if len(support):
        key=support.apply(lambda r:tuple(sorted((str(r['source']),str(r['target'])))),axis=1)
        consensus=support.assign(_pair=key).groupby('_pair').agg(n_sources=('source_covariance','nunique'),
                    sources=('source_covariance',lambda x:'|'.join(sorted(set(x))))).reset_index()
        consensus[['node1','node2']]=pd.DataFrame(consensus._pair.tolist(),index=consensus.index)
        consensus=consensus.drop(columns='_pair')
    else: consensus=pd.DataFrame(columns=['n_sources','sources','node1','node2'])
    params={"alpha":float(alpha),"n_eff":None if n_eff is None else float(n_eff),"max_cond_set":int(max_cond_set)}
    values={**_table_vars(support,"edge_support","edge",long_names=_PC_LONG_NAMES),
            **_table_vars(consensus,"consensus","pair",long_names=_PC_LONG_NAMES),
            "networks":{k:_pc_dataset(v,software="vigor.causal.pc_from_covariance",parameters=params) for k,v in networks.items()}}
    ds=result_dataset(values,title="Component-specific G/E causal networks",software="vigor.causal.discover_ge_network",
                      parameters={**params,"components":",".join(networks)})
    return maybe_save(ds,save_path)


def fit_multisource_covariance_sem(
    covariances: dict[str, np.ndarray],
    *,
    edge_mask: np.ndarray | nx.Graph | None = None,
    order=None,
    names: list[str] | None = None,
    source_weights: dict[str,float] | None = None,
    l1: float = 0.0,
    ridge: float = 1e-8,
    maxiter: int = 2000,
    tol: float = 1e-9,
    evaluation_covariances: dict[str, np.ndarray] | None = None,
    save_path=None,
) -> "xarray.Dataset":
    """Fit one structural path matrix B jointly to additive, dominance and E covariances.

    The SEM is Z = B Z + eta_s with a shared B and source-specific diagonal innovation
    covariance Psi_s. For candidate B, Psi_s=(I-B) C_s (I-B)' and the objective penalizes
    its off-diagonal mass across all covariance sources. A topological ``order`` enforces
    acyclicity by allowing only earlier -> later paths. Asymmetric candidates
    preserve their direction and reject cycles or conflicting orders; reciprocal
    candidates are undirected adjacencies. Empty masks fit a fixed B=0 model.
    Evaluation metrics project supplied covariances using training B and refitted
    diagonal innovations; independence of these matrices is not verified.

    Parameters
    ----------
    covariances : mapping of str to square array-like
        Source-specific trait covariances, such as additive, dominance and residual. All sources
        must share one trait dimension/order; each source receives diagonal innovation
        covariance while one acyclic path matrix B is shared. Training covariance fit alone does
        not identify causal direction.

    edge_mask : np.ndarray | nx.Graph | None
        Candidate trait paths, represented by a square trait adjacency mask or graph. Directions
        and node ordering must agree with names; None uses the fitter's candidate policy.
        Default: ``None``.

    order : sequence or None
        Topological trait/node ordering restricting paths to earlier-to-later edges. None lets
        the graph routine choose or infer a compatible acyclic ordering. Default: ``None``.

    names : list[str] | None
        Ordered component identifiers used for lookup and labeling. Length/order must match the
        corresponding matrices; None uses the implementation's inferred/default names. Default:
        ``None``.

    source_weights : dict[str, float] | None
        Weights for the corresponding covariance/evidence/prior components. Match the documented
        component names or variant order; None applies the routine's uniform/default weighting.
        Default: ``None``.

    l1 : float
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``0.0``.

    ridge : float
        Regularization strength added to the fitted objective or linear system. Nonnegative
        values describe a penalty; zero removes that penalty. Componentwise arrays must match
        the component order. Default: ``1e-08``. Passed to ``_np_project_psd`` as ``eps``.

    maxiter : int
        Maximum solver iteration count. A positive integer caps work; convergence diagnostics
        must still be checked rather than assuming the cap guarantees convergence. Default:
        ``2000``. Passed to ``minimize`` as ``options``.

    tol : float
        Convergence tolerance for the corresponding solver criterion. Use a nonnegative finite
        value; relative, gradient and objective criteria follow the underlying solver named in
        this routine. Default: ``1e-09``. Passed to ``minimize`` as ``options``.

    evaluation_covariances : mapping or None
        Optional covariances used only for evaluation under the fitted training B, with
        source-specific diagonal innovations refitted on these matrices. None omits evaluation
        metrics. The routine does not verify independence from training covariances; callers
        must supply held-out estimates to claim validation. Default: ``None``.

    save_path : str, os.PathLike or None
        Optional .zarr path; when given the result is written as consolidated, sharded Zarr v3
        and returned reopened lazily. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        Data variables: ``B``, ``total_effect_matrix`` and ``edge_mask`` (trait_i, trait_j;
        entry [i, j] is the path j -> i); ``innovation_covariances`` and
        ``reconstructed_covariances`` (component, trait_i, trait_j); the former
        ``fit_metrics`` table along ``component`` (``relative_frobenius_error``,
        ``innovation_offdiag_norm``) and the former ``evaluation_metrics`` table along
        ``evaluation_component`` (``evaluation_relative_frobenius_error``). Attrs: ``order``,
        ``success``, ``message``, ``objective``, ``n_iter``, ``evaluation_is_heldout``,
        ``evaluation_covariances_supplied``, ``evaluation_semantics``,
        ``evaluation_independence`` plus title/software/parameters provenance. Path
        coefficients are conditional on the candidate graph and acyclicity constraints.

    """
    if not covariances: raise ValueError('covariances must not be empty')
    keys=list(covariances); mats={k:_np_project_psd(v,eps=ridge) for k,v in covariances.items()};p=next(iter(mats.values())).shape[0]
    if any(v.shape!=(p,p) for v in mats.values()): raise ValueError('all covariance matrices must have the same square shape')
    names=[str(i) for i in range(p)] if names is None else list(map(str,names));
    if len(names)!=p: raise ValueError('names must match covariance dimension')
    if len(set(names)) != p: raise ValueError('names must be unique')
    # Asymmetric candidates specify directions; reciprocal candidates are
    # adjacencies whose orientation is selected by the declared order.
    directed=nx.DiGraph();directed.add_nodes_from(range(p))
    candidates=[]
    if isinstance(edge_mask,nx.Graph):
        if not set(map(str,edge_mask.nodes())).issubset(names):
            raise ValueError('edge_mask references unknown nodes')
        for a,b in edge_mask.edges():
            ia,ib=names.index(str(a)),names.index(str(b))
            if ia==ib: raise ValueError('self edges are not supported')
            strict=isinstance(edge_mask,nx.DiGraph) and not edge_mask.has_edge(b,a)
            candidates.append((ia,ib,strict))
            if strict: directed.add_edge(ia,ib)
    elif edge_mask is not None:
        M=np.asarray(edge_mask,bool)
        if M.shape!=(p,p): raise ValueError('edge_mask must be p x p')
        if np.any(np.diag(M)): raise ValueError('self edges are not supported')
        for dst,src in zip(*np.where(M)):
            strict=not M[src,dst]
            candidates.append((src,dst,strict))
            if strict: directed.add_edge(src,dst)
    if not nx.is_directed_acyclic_graph(directed):
        raise ValueError('cyclic directed candidates do not define a DAG')
    order=list(nx.topological_sort(directed)) if order is None else [names.index(str(x)) if isinstance(x,str) else int(x) for x in order]
    if sorted(order)!=list(range(p)): raise ValueError('order must contain every node exactly once')
    rank={node:i for i,node in enumerate(order)}
    if any(rank[src]>=rank[dst] for src,dst in directed.edges()):
        raise ValueError('order contradicts directed candidates')
    allow=np.zeros((p,p),bool) # B[target, source]
    if edge_mask is None:
        for src in range(p):
            for dst in range(p):allow[dst,src]=rank[src]<rank[dst]
    else:
        for src,dst,strict in candidates:
            if strict or rank[src]<rank[dst]:allow[dst,src]=True
            else:allow[src,dst]=True
    edges=list(zip(*np.where(allow)))
    weights={k:1.0 for k in keys} if source_weights is None else {k:float(source_weights.get(k,1.0)) for k in keys}
    scales={k:max(np.linalg.norm(v,'fro')**2,1e-12) for k,v in mats.items()}

    def unpack(x):
        """Unpack optimizer coordinates into the enclosing matrix parameterization.

        Parameters
        ----------
        x : one-dimensional numpy.ndarray
            Flattened optimization coordinates in the parameter layout captured by this closure.
            Length and factor/path slice order must match that layout; the objective unpacks these
            entries before evaluation.

        Returns
        -------
        object
            Computed result from ``B``; see the summary and the called implementation for its labeled/numerical representation.
        """
        B=np.zeros((p,p),float)
        for val,(dst,src) in zip(x,edges): B[dst,src]=val
        return B
    def objective(x):
        """Evaluate the optimization objective for the current parameter vector.

        Parameters
        ----------
        x : one-dimensional numpy.ndarray
            Flattened optimization coordinates in the parameter layout captured by this closure.
            Length and factor/path slice order must match that layout; the objective unpacks these
            entries before evaluation.

        Returns
        -------
        object
            Computed result from ``loss``; see the summary and the called implementation for its labeled/numerical representation.
        """
        B=unpack(x);A=np.eye(p)-B;loss=0.0
        for k,C in mats.items():
            Psi=_np_symmetrize(A@C@A.T);off=Psi-np.diag(np.diag(Psi));loss += weights[k]*float(np.sum(off*off))/scales[k]
        if l1: loss += float(l1)*float(np.sum(np.sqrt(np.asarray(x)**2+1e-12)))
        return loss
    x0=np.zeros(len(edges),float)
    if edges:
        res=minimize(objective,x0,method='L-BFGS-B',bounds=[(-3.,3.)]*len(edges),
                     options={'maxiter':int(maxiter),'ftol':float(tol),'gtol':float(tol)})
    else:
        from scipy.optimize import OptimizeResult
        res=OptimizeResult(x=x0,fun=objective(x0),success=True,nit=0,
                           message='Fixed empty-edge model; no optimizer parameters')
    B=unpack(res.x);A=np.eye(p)-B;Ctot=np.linalg.inv(A); innovations={};reconstructed={};metrics=[]
    for k,C in mats.items():
        Psi_full=_np_symmetrize(A@C@A.T);Psi=np.diag(np.clip(np.diag(Psi_full),ridge,None));Cre=_np_symmetrize(Ctot@Psi@Ctot.T);innovations[k]=Psi_full;reconstructed[k]=Cre
        metrics.append((k,float(np.linalg.norm(C-Cre,'fro')/max(np.linalg.norm(C,'fro'),1e-12)),
                        float(np.linalg.norm(Psi_full-np.diag(np.diag(Psi_full)),'fro'))))
    Bdf=pd.DataFrame(B,index=names,columns=names);Tdf=pd.DataFrame(Ctot,index=names,columns=names)
    eval_metrics=[]
    if evaluation_covariances is None:
        import warnings
        warnings.warn('fit_multisource_covariance_sem is reporting in-sample fit metrics because evaluation_covariances was not supplied; use held-out covariance matrices for publication-facing model comparison.', RuntimeWarning, stacklevel=2)
    if evaluation_covariances is not None:
        for k,Cev0 in evaluation_covariances.items():
            Cev=_np_project_psd(Cev0,eps=ridge)
            if Cev.shape!=(p,p): raise ValueError('evaluation covariance dimensions must match training')
            Psi_ev=_np_symmetrize(A@Cev@A.T)
            Psi_d=np.diag(np.clip(np.diag(Psi_ev),ridge,None)); Cre=_np_symmetrize(Ctot@Psi_d@Ctot.T)
            eval_metrics.append((k,float(np.linalg.norm(Cev-Cre,'fro')/max(np.linalg.norm(Cev,'fro'),1e-12))))
    import xarray as xr
    comps=[str(k) for k in keys]; tr={'trait_i':names,'trait_j':names}
    stack=lambda d:xr.DataArray(np.stack([d[k] for k in keys]),dims=('component','trait_i','trait_j'),coords={'component':comps,**tr})
    fit_tab=pd.DataFrame(metrics,columns=['source','relative_frobenius_error','innovation_offdiag_norm']).astype({'source':str}).set_index('source')
    ev_tab=pd.DataFrame(eval_metrics,columns=['source','relative_frobenius_error']).astype({'source':str}).set_index('source')
    metric_names={'relative_frobenius_error':'relative Frobenius reconstruction error','innovation_offdiag_norm':'Frobenius norm of off-diagonal innovation covariance'}
    values={'B':Bdf,'total_effect_matrix':Tdf,'innovation_covariances':stack(innovations),
            'reconstructed_covariances':stack(reconstructed),
            **_table_vars(fit_tab,'fit_metrics','component',long_names=metric_names),
            **_table_vars(ev_tab,'evaluation_metrics','evaluation_component',prefix='evaluation_',long_names=metric_names),
            'evaluation_covariances_supplied':evaluation_covariances is not None,
            'evaluation_semantics':'fixed training B; diagonal innovations re-estimated on supplied evaluation covariance: projection diagnostic, not frozen out-of-sample reconstruction',
            'evaluation_independence':'unverified; caller-supplied matrices do not establish held-out independence',
            'edge_mask':pd.DataFrame(allow,index=names,columns=names),'order':[names[i] for i in order],'success':bool(res.success),'message':str(res.message),
            'objective':float(res.fun),'n_iter':int(res.nit)}
    ds=result_dataset(values,dims={'B':('trait_i','trait_j'),'total_effect_matrix':('trait_i','trait_j'),'edge_mask':('trait_i','trait_j')},
        var_attrs={'B':{'long_name':'shared structural path matrix','description':'B[trait_i, trait_j] is the direct path trait_j -> trait_i; Z = B Z + eta'},
                   'total_effect_matrix':{'long_name':'total-effect matrix','description':'(I - B)^-1'},
                   'innovation_covariances':{'long_name':'source-specific innovation covariance','description':'(I - B) C_s (I - B)^T before diagonal projection'},
                   'reconstructed_covariances':{'long_name':'model-implied covariance','description':'(I - B)^-1 diag(Psi_s) (I - B)^-T'},
                   'edge_mask':{'long_name':'allowed structural paths','description':'True where B[trait_i, trait_j] is free'}},
        attrs={'evaluation_is_heldout':None},title='Multisource covariance SEM',software='vigor.causal.fit_multisource_covariance_sem',
        parameters={'l1':float(l1),'ridge':float(ridge),'maxiter':int(maxiter),'tol':float(tol),'components':','.join(comps)})
    return maybe_save(ds,save_path)

def QC_FIGS(imputed_df_fit, imputed_df_g, imputed_df_g_e, df_full, test_rows, test_elems, name):
    """Build comparison plots for fitted/imputed genetic and residual phenotype predictions.

    Parameters
    ----------
    imputed_df_fit : pandas.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.

    imputed_df_g : pandas.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.

    imputed_df_g_e : pandas.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.

    df_full : pandas.DataFrame
        Observed/training or comparison phenotype matrix in shared animal/trait order. Imputed
        frames contain the corresponding posterior/baseline predictions; keep evaluation masks
        separate from fitting outcomes.

    test_rows : object (see the cited input contract)
        Input for the operation ``zip([imputed_df_fit, imputed_df_g, imputed_df_g_e], [df_full,
        test_rows, test_elems], ['steelblue', 'firebrick', 'gray'], ['seen samples', 'unseen
        samples', 'unseen measurement of seen samples'])``. See ``vigor.causal.QC_FIGS`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.

    test_elems : object (see the cited input contract)
        Input for the operation ``zip([imputed_df_fit, imputed_df_g, imputed_df_g_e], [df_full,
        test_rows, test_elems], ['steelblue', 'firebrick', 'gray'], ['seen samples', 'unseen
        samples', 'unseen measurement of seen samples'])``. See ``vigor.causal.QC_FIGS`` and its
        caller for the scientific representation; the shown operation constrains the required
        type/shape.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    Returns
    -------
    object
        Computed result from ``tabs``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from functools import reduce
    try:
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
    except ImportError as exc:
        raise ImportError("QC_FIGS requires the VIGOR 'plot' extra (panel, holoviews, hvplot)") from exc
    tabs = pn.Tabs()
    meta = dict(frame_width = 270, frame_height = 270,alpha = .7, xlabel = '',ylabel = '',  line_width = 1, line_color = 'black', hover = False)
    for df_imputed,df_f,c,names in zip([imputed_df_fit, imputed_df_g, imputed_df_g_e], 
                                       [df_full,test_rows,test_elems],
                                       ['steelblue', 'firebrick', 'gray'], 
                                       ['seen samples', 'unseen samples', 'unseen measurement of seen samples'] ):
        dfppc = pd.concat([df_f,  df_imputed.rename(lambda x: x+'_ppc', axis = 1)], axis = 1)\
                  .rename(lambda x: x.replace('regressedlr_', ''), axis = 1).rename(str)
        ppc_out_of_sample = reduce(lambda x, y : x+y, [dfppc.hvplot.scatter(x = i, y = f'{i}_ppc',color = c,
                                                        title = "r:"+"r2:"+ str(round(dfppc[[i, f'{i}_ppc']].corr().iloc[0, 1]**2, 2))+ ' ' +i, **meta )*\
                                   hv.Slope(1,0).opts(color = 'red') for i in imputed_df_fit.columns.str.replace('regressedlr_', '')]\
                                  ).cols(4)
        tabs.append((f'{name} {names}', ppc_out_of_sample))
    return tabs

def chol2cov(L, return_correlation = False, eps: float = 1e-12):
    """Reconstruct covariance from Cholesky factors, optionally normalizing to correlation.

    Parameters
    ----------
    L : array-like
        Numerical operand in ``L ** 2``. Shape must conform to that matrix/vector operation and
        the enclosing prepared state.

    return_correlation : bool
        Whether to normalize reconstructed covariance by marginal standard deviations, returning
        correlation instead of covariance. Default: ``False``.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-12``.

    Returns
    -------
    object
        Computed result from ``covariance / (denom + eps)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    import xarray as xr
    variance = (L**2).sum(dim="traits_dim_t")
    stds = np.sqrt(variance)
    L_transposed = L.rename({"traits_dim": "traits_dim_2"})
    covariance = xr.dot(L, L_transposed, dims="traits_dim_t")
    stds_2 = stds.rename({"traits_dim": "traits_dim_2"})
    if not return_correlation: return covariance
    denom = (stds * stds_2)
    return covariance / (denom + eps)

def chol2covMAP(trace, name, columns, return_correlation = False, eps: float = 1e-12):
    """Reconstruct labeled covariance or correlation from a point-estimate Cholesky factor.

    Parameters
    ----------
    trace : InferenceData, posterior Dataset or mapping
        Fitted posterior/point-estimate draws with the named variables required by this graph
        method. Trait/node coordinates must match the graph used for fitting.

    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.

    columns : sequence or pandas.Index
        Trait labels in matrix-column order, used to label or align the resulting trait
        quantities.

    return_correlation : bool
        Whether to normalize reconstructed covariance by marginal standard deviations, returning
        correlation instead of covariance. Default: ``False``.

    eps : float
        Small numerical floor or diagonal stabilization used to avoid singular inverses, zero
        denominators or nonpositive factor variances. Choose a positive value on the scale of
        the input quantities. Default: ``1e-12``.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(mapcov / (denom + eps), columns=columns, index=columns)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    mapl = trace[name]
    mapcov = mapl.dot(mapl.T) 
    if not return_correlation: return pd.DataFrame(mapcov, columns = columns, index = columns)
    mapsig = np.sqrt(np.diag(mapcov))
    denom = np.outer(mapsig, mapsig)
    return pd.DataFrame(mapcov/(denom + eps), columns = columns, index = columns)

def create_dual_masking_scheme(df: pd.DataFrame, row_pct=0.10, elem_pct=0.20, seed=42, holdout_row_idx: np.ndarray | None = None):
    """Creates a training dataset with two types of artificial missingness:
    1. Row Mask (Hold-out Set): 'row_pct' of rows are fully masked (all traits hidden).
    2. Element Mask (Imputation Set): For the remaining rows, 'elem_pct' of available data 
       is masked per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
        described in the summary; the local coercion/selection expression below defines the
        accepted representation.

    row_pct : float or pair of float
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``0.1``.

    elem_pct : float or pair of float
        Proportion/probability controlling the named missingness, effect allocation or
        simulation event. Probability values use [0, 1]; ranges supply lower/upper probabilities
        in increasing order. Default: ``0.2``. Source comparisons: ``n_to_drop == 0``. See the
        rejection/fallback branches for enforcement.

    seed : int or None
        Seed controlling randomized sampling, folds or projections for reproducibility. None
        uses the selected random-number backend without a fixed seed. Default: ``42``.

    holdout_row_idx : np.ndarray | None
        Input for the operation ``mask_rows[holdout_row_idx, :]``. See
        ``vigor.causal.create_dual_masking_scheme`` and its caller for the scientific
        representation; the shown operation constrains the required type/shape. Default:
        ``None``. None delegates to the local fallback shown in this body; it is not an
        additional numeric/category value.

    Returns
    -------
    object
        Computed result from ``(mask_rows, mask_elems, combined_mask)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    rng = np.random.default_rng(seed)
    n, t = df.shape
    valid_data = ~df.isna().to_numpy()
    mask_rows, mask_elems = np.zeros((n, t), dtype=bool), np.zeros((n, t), dtype=bool)    
    row_indices = np.arange(n)
    if holdout_row_idx is None: holdout_row_idx = rng.choice(row_indices, round(n*row_pct), replace=False)
    mask_rows[holdout_row_idx, :] = True
    training_row_bool = ~np.isin(row_indices, holdout_row_idx)
    training_row_idx = row_indices[training_row_bool]
    for col_idx in range(t):
        candidates = np.intersect1d(training_row_idx, np.where(valid_data[:, col_idx])[0] )
        if len(candidates) == 0: continue
        n_to_drop = int(len(candidates) * elem_pct)
        if n_to_drop == 0: continue
        drop_idx = rng.choice(candidates, size=n_to_drop, replace=False)
        mask_elems[drop_idx, col_idx] = True
    combined_mask = mask_rows | mask_elems    
    return mask_rows, mask_elems, combined_mask




# -----------------------------
# Example snippets (copy/paste)
# -----------------------------
EXAMPLE_MGREML = r"""
mgreml = genomic_sem_patched5.CausalGraph(df_pheno)
mgreml.add_grm('FullGRM', fgrm, top_eigen2fixed=None, n_components=2000, store_full=True)
mgreml.add_trait('traits', data_cols=df_pheno.columns,
                 track_values=True, track_values_per_path=True,
                 residual='diag', missing='auto')
mgreml.add_edge('FullGRM', 'traits', kind='grm',
                grm_trait_cov='lkj', eta=1.2,
                track_variance=True, prior_variance=0.55,
                adjust_variance_by_shape=False)

mgreml.graph_qc()
mgreml.plot_graph()

# auto chooses generic because missing is elementwise; if you set missing='row' it will choose marginal_evd
model = mgreml.build_model(engine='auto')
"""

EXAMPLE_LATENT_FACTOR = r"""
cg = genomic_sem_patched5.CausalGraph(df_pheno)
cg.add_grm("FullGRM", fgrm, top_eigen2fixed=0.2, n_components=400)
cg.add_grm("TopSNPs", topgrm, subtract_fixed="FullGRM_fixed", n_components=200)
cg.add_latent("OUD_latent", n_dim=6, is_factor=True, prior_sigma=0.2, track_values=True)

# For high missingness, prefer missing='auto' and residual='factor' or 'diag' to avoid pattern explosions
cg.add_trait("traits", data_cols=df_pheno.columns,
             track_values=True, track_values_per_path=False,
             residual="factor", n_factors_E=4,
             missing="auto")

cg.add_edge("FullGRM", "OUD_latent", kind="grm", grm_trait_cov="diag", prior_variance=0.55)
cg.add_edge("TopSNPs", "OUD_latent", kind="grm", grm_trait_cov="diag", prior_variance=0.55)
cg.add_edge("FullGRM_fixed", "traits", kind="dense", prior_variance=0.3)
cg.add_edge("OUD_latent", "traits", kind="factor", prior_variance=0.8)

cg.graph_qc()
cg.plot_graph()

model = cg.build_model(engine="generic")
"""
# Calibrated causal functionality is implemented above; do not shadow current APIs.


def structural_sem_tabs(result, *, title="Structural covariance refinement", edge_threshold=1e-8,
                        width=760, height=560, layout="spring"):
    """Build an hvPlot/NetworkX Panel report for a structural refinement result.

    Returns
    -------
    panel.Tabs
        Tabs contain explanatory text and figures for the fitted SEM network,
        PC/ANM candidate graph, path and total-effect matrices, source-specific
        innovation factors, covariance reconstruction, and outer-loop
        convergence.  Plotting dependencies are imported lazily.

    Parameters
    ----------
    result : object (see the cited input contract)
        mapping or pandas.Series Output of :func:`vigor.genetic_varcov_PSD` with
        ``structural_refine=True``.  A bare structural-result mapping containing
        ``structural_sem``/``structural_pc`` is also accepted.

    title : str
        str Human-facing report title used in the Overview tab. Default: ``'Structural
        covariance refinement'``.

    edge_threshold : float
        Minimum magnitude/asymmetry/stability evidence required for the corresponding graph edge
        or direction. Correlation/stability cutoffs use [0, 1]; directional score ratios use
        positive values, ordinarily greater than one. Default: ``1e-08``.

    width : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``760``. Passed to ``long.hvplot.heatmap`` as
        ``width``.

    height : int
        Rendered plot dimension in pixels. Positive values control the corresponding
        horizontal/vertical extent. Default: ``560``. Passed to ``long.hvplot.heatmap`` as
        ``height``.

    layout : str or sequence
        LD array layout or packed optimizer-coordinate slices. LD spellings 'dense'/'matrix' use a
        square matrix; 'offset'/'band'/'window'/'rolling' use banded offsets where supported.
        Optimization layouts must come from the same parameter-layout helper. Default:
        ``'spring'``.
    """
    try:
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        import hvplot.networkx as hvnx
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("structural_sem_tabs requires VIGOR[plot] (panel, holoviews, hvplot, bokeh)") from exc

    def _get(obj, key, default=None):
        """Read a mapping/attribute value using the supplied fallback when absent.

        Parameters
        ----------
        obj : array-like, table or labeled object
            Input values consumed by this conversion or analysis. Preserve the labels/axis ordering
            described in the summary; the local coercion/selection expression below defines the
            accepted representation.

        key : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required.

        default : object (see the cited input contract)
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``None``. None delegates to the local fallback shown
            in this body; it is not an additional numeric/category value.
        """
        try:
            if hasattr(obj, "data_vars"):  # xarray.Dataset result: arrays, then attrs, then stored tables
                if key in obj.data_vars:
                    return obj[key].to_pandas() if obj[key].ndim <= 2 else obj[key]
                if key in obj.attrs:
                    return obj.attrs[key]
                tab = _result_table(obj, key)
                return tab if len(tab.columns) else default
            return obj[key]
        except Exception:
            return default

    sem = _get(result, "structural_sem")
    if sem is None:
        raise ValueError("result does not contain structural_sem; run genetic_varcov_PSD(..., structural_refine=True)")
    pc = _get(result, "structural_pc", None)
    pc = {} if pc is None else pc
    anm = _get(result, "structural_anm", None)
    hist = _get(result, "structural_history", pd.DataFrame())
    if not isinstance(hist, pd.DataFrame):
        hist = pd.DataFrame(hist)
    B = _get(sem, "B"); B = B if isinstance(B, pd.DataFrame) else pd.DataFrame(B)
    H = _get(sem, "total_effect_matrix"); H = H if isinstance(H, pd.DataFrame) else pd.DataFrame(H, index=B.index, columns=B.columns)
    names = list(map(str, B.index)); B = B.copy(); B.index=B.columns=names; H=H.copy();H.index=H.columns=names

    def _desc(text):
        """Create a Markdown pane containing the supplied report description.

        Parameters
        ----------
        text : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output.
        """
        return pn.pane.Markdown(text, sizing_mode="stretch_width")

    def _network_plot(G, *, edge_values=None, seed=41, directed=True, ttl=""):
        """Render a directed/undirected graph with optional edge-value annotations.

        Parameters
        ----------
        G : networkx.Graph
            Network graph defining the nodes and candidate edges used for this network operation;
            node identifiers must match the associated scores/metadata.

        edge_values : sequence or mapping or None
            Graph adjacency/path specification or edge-aligned values. Node pairs must match the
            graph's trait/gene identifiers; None imposes no optional constraint/annotation. Default:
            ``None``.

        seed : int or None
            Seed controlling randomized sampling, folds or projections for reproducibility. None
            uses the selected random-number backend without a fixed seed. Default: ``41``. Passed to
            ``nx.spring_layout`` as ``seed``.

        directed : bool
            Input supplied to ``hvnx.draw_networkx_edges`` as ``arrow_style``. See that callable's
            documented ``arrow_style`` contract for its type, alignment and value domain. This
            wrapper retains that contract. Default: ``True``. Passed to ``hvnx.draw_networkx_edges``
            as ``arrow_style``.

        ttl : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output. Default: ``''``. Passed to ``(ep * np0 * lp).opts`` as ``title``.
        """
        G = G.copy()
        if not len(G):
            return pn.pane.Markdown("*No edges passed the current threshold.*")
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=int(seed), k=None)
        # hvPlot NetworkX returns HoloViews elements while retaining NetworkX's
        # topology/layout semantics. Path magnitude controls edge width only.
        if edge_values:
            for a,b in G.edges():
                v=float(edge_values.get((a,b),1.0)); G.edges[a,b]["magnitude"]=abs(v); G.edges[a,b]["weight"]=v
        else:
            for a,b in G.edges(): G.edges[a,b]["magnitude"]=1.0;G.edges[a,b]["weight"]=1.0
        ep = hvnx.draw_networkx_edges(G,pos,alpha=.65,edge_width=1.5+hv.dim("magnitude").norm()*6,
                                      arrow_style="-|>" if directed else "-",arrow_size=10)
        np0 = hvnx.draw_networkx_nodes(G,pos,node_size=900,alpha=.95)
        lp = hvnx.draw_networkx_labels(G,pos,font_size="9pt")
        return (ep*np0*lp).opts(width=int(width),height=int(height),title=ttl,xaxis=None,yaxis=None,show_grid=False)

    # ---- Overview / SEM graph -------------------------------------------------
    Gsem=nx.DiGraph();Gsem.add_nodes_from(names); edge_values={}
    for dst in names:
        for src in names:
            v=float(B.loc[dst,src])
            if src!=dst and np.isfinite(v) and abs(v)>float(edge_threshold):
                Gsem.add_edge(src,dst,weight=v,magnitude=abs(v));edge_values[(src,dst)]=v
    sem_plot=_network_plot(Gsem,edge_values=edge_values,ttl="Shared-path SEM")
    edge_rows=[{"source":a,"target":b,"path":float(B.loc[b,a]),"abs_path":abs(float(B.loc[b,a]))} for a,b in Gsem.edges()]
    edge_df=pd.DataFrame(edge_rows).sort_values("abs_path",ascending=False) if edge_rows else pd.DataFrame(columns=["source","target","path","abs_path"])
    overview=pn.Column(
        _desc(f"### {title}\nThe displayed arrows are the non-zero paths in the final shared acyclic SEM. The SEM is fitted to the original MoM tensors; the covariance inference/analytic-SE estimator remains the final support-fixed PSD/FA refit."),
        sem_plot,
        pn.pane.DataFrame(edge_df, sizing_mode="stretch_width", height=min(360,100+28*max(1,len(edge_df))))
    )

    # ---- PC / ANM candidate graph --------------------------------------------
    Gpc=pc.get("graph",nx.DiGraph()) if isinstance(pc,dict) else pc.attrs.get("graph",nx.DiGraph()) if hasattr(pc,"data_vars") else nx.DiGraph()
    pc_plot=_network_plot(Gpc,ttl="PC-stable / ANM-oriented candidate graph") if len(Gpc) else pn.pane.Markdown("*No PC graph available.*")
    anm_table=None
    if isinstance(anm,dict): anm_table=anm.get("orientations")
    elif hasattr(anm,"data_vars"): anm_table=_result_table(anm,"orientations")
    graph_parts=[_desc("### Candidate graph\nPC-stable determines the conditional-dependence skeleton and collider/Meek orientations. ANM is applied only to surviving unresolved candidate edges; it does not override PC orientations.") , pc_plot]
    if isinstance(anm_table,pd.DataFrame) and len(anm_table):
        graph_parts.append(pn.pane.DataFrame(anm_table,sizing_mode="stretch_width",height=min(360,100+28*len(anm_table))))
    graph_tab=pn.Column(*graph_parts)

    def _matrix_heatmap(M, value_name, ttl):
        """Return the matrix heatmap result computed by ``long.hvplot.heatmap(x='source', y='target', C=value_name, cmap='RdBu_r', clim=(-lim, lim), width=int(width), height=int(height), title=ttl).opts``.

        Parameters
        ----------
        M : array-like
            Numerical operand in ``M.copy()``. Shape must conform to that matrix/vector operation
            and the enclosing prepared state.

        value_name : str
            Identifier used to select or label the relevant component, field or output quantity. It
            must match the names present in the supplied object where lookup is required. Passed to
            ``df.rename_axis('target').reset_index().melt`` as ``value_name``.

        ttl : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output. Passed to ``long.hvplot.heatmap`` as ``title``.
        """
        df=M.copy();df.index=df.index.astype(str);df.columns=df.columns.astype(str)
        long=df.rename_axis("target").reset_index().melt(id_vars="target",var_name="source",value_name=value_name)
        lim=float(np.nanmax(np.abs(long[value_name]))) if len(long) else 1.0
        if not np.isfinite(lim) or lim<=0: lim=1.0
        return long.hvplot.heatmap(x="source",y="target",C=value_name,cmap="RdBu_r",clim=(-lim,lim),
                                   width=int(width),height=int(height),title=ttl).opts(invert_yaxis=True,xrotation=90)

    path_tab=pn.Column(_desc("### Direct paths\nEach cell is B[target, source], so a non-zero entry represents source → target. Z = BZ + η."),
                       _matrix_heatmap(B,"path","Direct SEM path coefficients B"))
    total_tab=pn.Column(_desc("### Total effects\nThe matrix H=(I−B)⁻¹ accumulates direct and indirect propagation through the fitted acyclic graph."),
                        _matrix_heatmap(H,"total_effect","Total-effect / propagation matrix"))

    # ---- Innovation factors ---------------------------------------------------
    factor_tabs=pn.Tabs(dynamic=True)
    for source,spec in (_get(sem,"innovation_specs",{}) or {}).items():
        F=np.asarray(spec.get("factor",np.empty((len(names),0))),float)
        if F.ndim!=2 or F.shape[0]!=len(names): continue
        cols=[f"F{i+1}" for i in range(F.shape[1])]
        Fdf=pd.DataFrame(F,index=names,columns=cols)
        if F.shape[1]:
            long=Fdf.rename_axis("trait").reset_index().melt(id_vars="trait",var_name="factor",value_name="loading")
            lim=max(float(np.nanmax(np.abs(long.loading))),1e-12)
            plot=long.hvplot.heatmap(x="factor",y="trait",C="loading",cmap="RdBu_r",clim=(-lim,lim),
                                     width=int(width),height=int(height),title=f"{source}: innovation loadings").opts(invert_yaxis=True)
        else:
            plot=pn.pane.Markdown("*This source has no latent-factor columns (diagonal innovation structure).*" )
        factor_tabs.append((str(source),pn.Column(_desc(f"### {source} innovation factors\nThese loadings describe covariance remaining at the innovation level after directed SEM propagation; sparse zeros are preserved."),plot)))
    factors_tab=pn.Column(_desc("### Source-specific innovation factors\nThe SEM shares B across covariance sources while allowing each source to retain its own PSD factor/diagonal innovation geometry."),factor_tabs)

    # ---- Covariance reconstruction --------------------------------------------
    recon_tabs=pn.Tabs(dynamic=True)
    reconstructed=_get(sem,"reconstructed_covariances",{})
    if hasattr(reconstructed,"dims"):  # stacked (component, trait_i, trait_j) DataArray
        reconstructed={str(c):reconstructed.sel({reconstructed.dims[0]:c}).values for c in reconstructed[reconstructed.dims[0]].values}
    observed={}
    gc=_get(result,"G_components_inference",_get(result,"G_components",{}))
    if hasattr(gc,"dims") and getattr(gc,"ndim",0)==3:  # stacked (component, trait_i, trait_j) DataArray
        gc={str(c):gc.sel({gc.dims[0]:c}).to_pandas() for c in gc[gc.dims[0]].values}
    if isinstance(gc,dict): observed.update(gc)
    eobs=_get(result,"E_inference",_get(result,"E",None))
    if eobs is not None: observed["E"]=eobs
    for source,rec in reconstructed.items():
        if source not in observed: continue
        O=np.asarray(observed[source],float);R=np.asarray(rec,float)
        iu=np.triu_indices_from(O)
        d=pd.DataFrame({"PSD_FA":O[iu],"SEM":R[iu]})
        scatter=d.hvplot.scatter(x="PSD_FA",y="SEM",alpha=.45,width=int(width),height=int(height),
                                 title=f"{source}: SEM reconstruction vs final PSD/FA covariance")
        lo=float(np.nanmin([d.PSD_FA.min(),d.SEM.min()]));hi=float(np.nanmax([d.PSD_FA.max(),d.SEM.max()]))
        if np.isfinite(lo) and np.isfinite(hi) and hi>lo: scatter=scatter*hv.Curve([(lo,lo),(hi,hi)]).opts(line_dash="dashed")
        rel=float(np.linalg.norm(O-R,"fro")/max(np.linalg.norm(O,"fro"),1e-12))
        recon_tabs.append((str(source),pn.Column(_desc(f"### {source} covariance fit\nRelative Frobenius discrepancy between the final PSD/FA covariance and the SEM reconstruction: **{rel:.4g}**. The SEM itself is optimized against the original MoM tensors; this plot shows how its structural restriction compares with the final unconstrained support-fixed covariance estimator."),scatter)))
    recon_tab=pn.Column(_desc("### Covariance reconstruction\nSource-wise comparison of the structural SEM covariance with the final PSD/FA inference geometry."),recon_tabs)

    # ---- Outer-loop convergence -----------------------------------------------
    if len(hist):
        h=hist.copy()
        curves=[]
        for col,label in [("psd_objective","PSD/FA MoM objective"),("sem_objective","SEM MoM objective")]:
            if col in h:
                curves.append(h.hvplot.line(x="outer_iteration",y=col,label=label,width=int(width),height=360))
        conv=curves[0] if curves else pn.pane.Markdown("*No objective history available.*")
        for q in curves[1:]: conv=conv*q
        counts=[c for c in ["pc_edges","anm_oriented","sem_paths"] if c in h]
        count_plot=h.hvplot.line(x="outer_iteration",y=counts,width=int(width),height=320,title="Graph/support updates") if counts else None
        conv_parts=[_desc("### Alternating-fit convergence\nEach outer iteration updates the PC/ANM graph, fits the acyclic SEM directly to the MoM tensors, then feeds the SEM reconstruction back into the sparse support-selecting PSD/FA L-BFGS fit."),conv]
        if count_plot is not None: conv_parts.append(count_plot)
        conv_parts.append(pn.pane.DataFrame(h,sizing_mode="stretch_width",height=min(320,100+28*len(h))))
        convergence_tab=pn.Column(*conv_parts)
    else:
        convergence_tab=pn.Column(_desc("### Alternating-fit convergence\nNo outer-loop history was returned."))

    return pn.Tabs(
        ("SEM network",overview),
        ("PC + ANM graph",graph_tab),
        ("Direct paths",path_tab),
        ("Total effects",total_tab),
        ("Innovation factors",factors_tab),
        ("Covariance fit",recon_tab),
        ("Convergence",convergence_tab),
        dynamic=True,
    )
