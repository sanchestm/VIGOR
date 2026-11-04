"""Reusable exact moment contractions for weighted bootstrap samples."""
from __future__ import annotations

import time

import numpy as np


class PreparedWeightedMoments:
    """Prepare invariant pieces of exact weighted multi-kernel moments.

    ``weights`` passed to :meth:`compute` are sample multiplicities.  Dense
    Hadamard products are cached only while they fit in ``cache_gb``; operator
    kernels are always evaluated in row blocks and are never densified here.
    """

    def __init__(self, kernels, Yraw, block_size=512, cache_gb=.5,
                 accum_dtype=np.float64):
        """Initialize PreparedWeightedMoments with the inputs and resource settings below.

        Parameters
        ----------
        kernels : mapping
            Named animal relationship kernels; each matrix/operator must use the same animal order.
            Names identify the matching trait covariance components.

        Yraw : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        block_size : int
            Number of variants or rows processed together by the local block iterator. Supply a
            positive count; smaller blocks reduce peak memory at the cost of more iteration
            overhead. See the called iterator for which axis is blocked. Default: ``512``.

        cache_gb : float
            Memory budget in GiB for cached moment terms. Zero disables the cache; positive budgets
            permit only cached terms fitting within the limit. Default: ``0.5``.

        accum_dtype : NumPy dtype or str
            Numerical element type for the corresponding output, working block or accumulated
            matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
            numerical precision. Default: ``np.float64``.
        """
        started = time.perf_counter()
        # Lazy import keeps this module usable by multivariate without an
        # import cycle when it becomes the public entry point for this class.
        from .multivariate import as_kernel_operator

        if not kernels:
            raise ValueError("at least one kernel is required")
        self.kernel_names = [str(k) for k in kernels]
        self.ops = {
            str(k): as_kernel_operator(v, dtype=np.float32)
            for k, v in kernels.items()
        }
        self.block_size = max(1, int(block_size))
        self.accum_dtype = np.dtype(accum_dtype)
        self.Yraw = np.asarray(Yraw, dtype=float)
        if self.Yraw.ndim == 1:
            self.Yraw = self.Yraw[:, None]
        if self.Yraw.ndim != 2:
            raise ValueError("Yraw must be a two-dimensional array")
        self.n, self.n_traits = self.Yraw.shape
        for name, op in self.ops.items():
            if tuple(op.shape) != (self.n, self.n):
                raise ValueError(f"kernel {name!r} shape does not match Yraw")

        self._finite = np.isfinite(self.Yraw)
        # np.unique(axis=0) groups equal trait masks, including the common
        # complete-data mask, without retaining one N-vector per trait twice.
        unique, inverse = np.unique(self._finite.T, axis=0, return_inverse=True)
        self._group_masks = np.asarray(unique.T, dtype=self.accum_dtype)
        self._trait_groups = np.asarray(inverse, dtype=np.intp)
        self._diagonals = {
            name: np.asarray(op.diag(), dtype=self.accum_dtype)
            for name, op in self.ops.items()
        }

        budget = max(0, int(float(cache_gb) * (1024 ** 3)))
        self._hadamard = {}
        used = 0
        names = self.kernel_names
        for r, left in enumerate(names):
            for s in range(r, len(names)):
                right = names[s]
                lop, rop = self.ops[left], self.ops[right]
                # The K attribute is the dense operator's existing storage.
                # Requiring it avoids surprising expansion of low-rank/custom
                # operators merely because they implement row_block.
                if not (hasattr(lop, "K") and hasattr(rop, "K")):
                    continue
                needed = self.n * self.n * np.dtype(np.float32).itemsize
                if used + needed > budget:
                    continue
                H = (np.asarray(lop.K, dtype=np.float32)
                     * np.asarray(rop.K, dtype=np.float32))
                H = np.ascontiguousarray(H, dtype=np.float32)
                H.setflags(write=False)
                self._hadamard[(r, s)] = H
                used += H.nbytes

        # Public, plain-data diagnostics are convenient in bootstrap reports
        # and keep the object straightforward to pickle into worker processes.
        self.diagnostics = {
            "preparation_seconds": time.perf_counter() - started,
            "mask_group_count": int(self._group_masks.shape[1]),
            "cache_budget_bytes": budget,
            "cache_bytes": used,
            "cached_hadamard_pairs": len(self._hadamard),
            "no_n_boot_matrices": True,
            "stores_bootstrap_matrices": False,
            "lowrank_hadamard_implicit": True,
            "mask_group_contractions": True,
        }

    @staticmethod
    def _is_lowrank(op):
        """Test whether inputs satisfy lowrank.

        Parameters
        ----------
        op : kernel/linear operator or mapping
            Prepared matrix operator exposing the multiplication methods used below; shape and
            sample order must match the right-hand side or phenotype matrix.
        """
        return all(hasattr(op, name) for name in ("U", "delta", "bulk"))

    def _lowrank_hadamard_matmat(self, left, right, X, pair_chunk=128):
        """Apply ``(K_left ∘ K_right) @ X`` without materializing N x N.

        If K=U diag(d) U' + bI and L=V diag(e) V' + cI, then the
        off-diagonal low-rank part is generated by columns ``u_a * v_b``.
        The identity/bulk cross terms are diagonal and are applied directly.
        For K is L, symmetric (a,b) pairs are collapsed exactly.

        Parameters
        ----------
        left : object (see the cited input contract)
            Input for the operation ``left is right``. See
            ``vigor.bootstrap_moments.PreparedWeightedMoments._lowrank_hadamard_matmat`` and its
            caller for the scientific representation; the shown operation constrains the required
            type/shape.

        right : object (see the cited input contract)
            Input for the operation ``left is right``. See
            ``vigor.bootstrap_moments.PreparedWeightedMoments._lowrank_hadamard_matmat`` and its
            caller for the scientific representation; the shown operation constrains the required
            type/shape.

        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        pair_chunk : int
            Input for the operation ``max(1, int(pair_chunk))``. See
            ``vigor.bootstrap_moments.PreparedWeightedMoments._lowrank_hadamard_matmat`` and its
            caller for the scientific representation; the shown operation constrains the required
            type/shape. Default: ``128``.
        """
        U = np.asarray(left.U)
        V = np.asarray(right.U)
        dl = np.asarray(left.delta, dtype=self.accum_dtype)
        dr = np.asarray(right.delta, dtype=self.accum_dtype)
        X = np.asarray(X, dtype=self.accum_dtype)
        out = np.zeros((self.n, X.shape[1]), dtype=self.accum_dtype)
        same = left is right
        pc = max(1, int(pair_chunk))
        if same:
            # a<=b, with factor 2 on off-diagonal terms.
            ii, jj = np.triu_indices(len(dl))
            coeff = dl[ii] * dl[jj] * np.where(ii == jj, 1.0, 2.0)
        else:
            ii = np.repeat(np.arange(len(dl), dtype=np.intp), len(dr))
            jj = np.tile(np.arange(len(dr), dtype=np.intp), len(dl))
            coeff = dl[ii] * dr[jj]
        for k0 in range(0, len(coeff), pc):
            k1 = min(len(coeff), k0 + pc)
            W = np.asarray(U[:, ii[k0:k1]] * V[:, jj[k0:k1]], dtype=np.float32)
            WX = W.T @ X
            WX *= coeff[k0:k1, None]
            out += W @ WX
        bl = float(left.bulk); br = float(right.bulk)
        if bl or br:
            diag_left_top = (np.asarray(U, dtype=self.accum_dtype) ** 2) @ dl
            diag_right_top = (np.asarray(V, dtype=self.accum_dtype) ** 2) @ dr
            diag_extra = bl * diag_right_top + br * diag_left_top + bl * br
            out += diag_extra[:, None] * X
        return out

    def compute(self, weights):
        """Return exact ``A``/``B`` moments for one multiplicity vector.

        Parameters
        ----------
        weights : array-like or int
            Weights or multiplicities used in this accumulation/resampling step. Align vector
            entries with the indexed sample/variant axis; scalar counts specify the number of
            consecutive items.

        Returns
        -------
        dict
            Result fields: ``A``, ``B``, ``Y0``, ``R``, ``means``, ``kernel_names``. Field arrays preserve the corresponding input axis order.
        """
        w = np.asarray(weights, dtype=self.accum_dtype).reshape(-1)
        if w.size != self.n:
            raise ValueError("weights length does not match Yraw")
        Rbool = self._finite
        # All A matrices depend only on observation masks, not phenotype values.
        # Work in the much smaller space of unique missingness patterns.
        Gmask = self._group_masks
        groups = self._trait_groups
        denom_group = (w[:, None] * Gmask).sum(axis=0)
        denom = denom_group[groups]
        numer = np.sum(np.where(Rbool, self.Yraw * w[:, None], 0.0), axis=0)
        means = np.divide(numer, denom, out=np.zeros(self.n_traits), where=denom > 0)
        Y = np.asarray(np.where(Rbool, self.Yraw - means[None, :], 0.0),
                       dtype=self.accum_dtype)
        wY = w[:, None] * Y

        names = self.kernel_names
        q, p, t = len(names), len(names) + 1, self.n_traits
        A = np.zeros((p, p, t, t), dtype=self.accum_dtype)
        B = np.zeros((p, t, t), dtype=self.accum_dtype)
        for r, name in enumerate(names):
            B[r] = Y.T @ (w[:, None] * self.ops[name].matmat(wY))
            compact = Gmask.T @ ((w * self._diagonals[name])[:, None] * Gmask)
            A[r, q] = A[q, r] = compact[np.ix_(groups, groups)]
        compact_E = Gmask.T @ (w[:, None] * Gmask)
        A[q, q] = compact_E[np.ix_(groups, groups)]
        B[q] = Y.T @ wY

        # Identical observation masks have identical A entries.  Contract G
        # unique masks and expand the resulting GxG matrix to TxT.
        wG = w[:, None] * Gmask
        for r, left in enumerate(names):
            for s in range(r, q):
                H = self._hadamard.get((r, s))
                if H is not None:
                    compact = wG.T @ (H @ wG)
                else:
                    lop, rop = self.ops[left], self.ops[names[s]]
                    if self._is_lowrank(lop) and self._is_lowrank(rop):
                        compact = wG.T @ self._lowrank_hadamard_matmat(lop, rop, wG)
                    else:
                        compact = np.zeros((wG.shape[1], wG.shape[1]),
                                           dtype=self.accum_dtype)
                        for i0 in range(0, self.n, self.block_size):
                            i1 = min(self.n, i0 + self.block_size)
                            block = (np.asarray(lop.row_block(i0, i1), dtype=np.float32)
                                     * np.asarray(rop.row_block(i0, i1), dtype=np.float32))
                            compact += wG[i0:i1].T @ (block @ wG)
                expanded = compact[np.ix_(groups, groups)]
                sym = 0.5 * (expanded + expanded.T)
                A[r, s] = A[s, r] = sym
        for r in range(p):
            B[r] = 0.5 * (B[r] + B[r].T)
        return {"A": A, "B": B, "Y0": Y, "R": Rbool, "means": means,
                "kernel_names": list(names)}


__all__ = ["PreparedWeightedMoments"]
