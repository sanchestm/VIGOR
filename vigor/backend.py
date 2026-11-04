"""Optional CPU/CUDA projection backends."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def resolve_backend_name(requested):
    """Resolve backend name.

    Parameters
    ----------
    requested : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation. Source-recognized literal values: ``auto``,
        ``cpu``, ``cuda``. These select the corresponding branches below; comparison evidence
        alone does not establish an exhaustive enum.

    Returns
    -------
    object
        Computed result from ``'cuda'``; see the summary and the called implementation for its labeled/numerical representation.
    """
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    try:
        import cupy as cp  # type: ignore

        available = cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        available = False
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested != "cuda":
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
    if not available:
        raise RuntimeError(
            "backend='cuda' requires CuPy and a visible CUDA-capable NVIDIA GPU; "
            "install vigor[cuda] or use backend='cpu'"
        )
    return "cuda"


class CPUProjectionBackend:
    """CPUProjectionBackend represents cpu projection backend.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    name = "cpu"

    def __init__(self, U, Y, C, **_):
        """Initialize CPUProjectionBackend with the inputs and resource settings below.

        Parameters
        ----------
        U : array-like
            Numerical operand in the matrix/vector transformation described above. Its dimensions
            must conform to the operations below; see the source expression and upstream preparation
            helper for the precise representation.

        Y : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        C : array-like
            Numerical operand in the matrix/vector transformation described above. Its dimensions
            must conform to the operations below; see the source expression and upstream preparation
            helper for the precise representation.

        **_ : dict
            Additional keyword arguments consumed by the compatibility/callback logic in this body.
            See the explicit keyword lookups below for accepted keys; unused keys are subject to
            that compatibility policy.
        """
        self.U = U
        self.Y = Y
        self.C = C

    def project(self, X):
        """Project the supplied genotype block into the prepared eigen/phenotype/covariate bases.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``(self.U.T @ X, X.T @ self.Y, X.T @ self.C)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self.U.T @ X, X.T @ self.Y, X.T @ self.C

    def close(self):
        """Release resources retained by this reader, operator or progress object.

        Returns
        -------
        object
            Computed result from ``None``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return None


class CUDAProjectionBackend:
    """Replicate fixed arrays and partition marker projections over GPUs."""

    name = "cuda"

    def __init__(self, U, Y, C, devices=(), memory_budget_mb=None):
        """Initialize CUDAProjectionBackend with the inputs and resource settings below.

        Parameters
        ----------
        U : array-like
            Numerical operand in ``np.asarray(U).nbytes + np.asarray(Y).nbytes +
            np.asarray(C).nbytes``. Shape must conform to that matrix/vector operation and the
            enclosing prepared state.

        Y : array-like or pandas.DataFrame
            Phenotype observations, with animals in rows and traits in columns. Preserve the animal
            order used by the relationship kernel; NaN marks unobserved cells where the estimator
            supports missing data.

        C : array-like
            Numerical operand in ``np.asarray(U).nbytes + np.asarray(Y).nbytes +
            np.asarray(C).nbytes``. Shape must conform to that matrix/vector operation and the
            enclosing prepared state.

        devices : sequence of int
            CUDA device indices assigned to the projection worker pool. Indices must refer to
            available devices; an empty tuple uses the backend's default device selection. Default:
            ``()``.

        memory_budget_mb : float or str
            Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
            descriptive text identifying the operation in an allocation error. Default: ``None``.
            None delegates to the local fallback shown in this body; it is not an additional
            numeric/category value. Passed to ``cp.get_default_memory_pool().set_limit`` as
            ``size``.
        """
        import cupy as cp  # type: ignore

        count = cp.cuda.runtime.getDeviceCount()
        self.devices = tuple(devices) if devices else tuple(range(count))
        if not self.devices:
            raise RuntimeError("no CUDA devices are available")
        if any(device < 0 or device >= count for device in self.devices):
            raise ValueError(
                f"gpu_devices must be between 0 and {count - 1}; got {self.devices}"
            )
        self.cp = cp
        self.memory_budget_mb = memory_budget_mb
        required = np.asarray(U).nbytes + np.asarray(Y).nbytes + np.asarray(C).nbytes
        if memory_budget_mb is not None and required > memory_budget_mb * 1024 ** 2:
            raise MemoryError(
                f"CUDA persistent arrays require {required / 1024 ** 2:.0f} MiB per device, "
                f"above gpu_memory_budget_mb={memory_budget_mb}"
            )
        self.arrays = {}
        for device in self.devices:
            with cp.cuda.Device(device):
                if memory_budget_mb is not None:
                    cp.get_default_memory_pool().set_limit(
                        size=int(memory_budget_mb * 1024 ** 2)
                    )
                self.arrays[device] = (
                    cp.asarray(U),
                    cp.asarray(Y),
                    cp.asarray(C),
                )
        self.executor = (
            ThreadPoolExecutor(max_workers=len(self.devices))
            if len(self.devices) > 1
            else None
        )

    def _project_slice(self, device, X):
        """Perform one device-specific slice of the prepared genotype projection.

        Parameters
        ----------
        device : object (see the cited input contract)
            Input for the operation ``self.arrays[device]``. See
            ``vigor.backend.CUDAProjectionBackend._project_slice`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.

        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.
        """
        cp = self.cp
        with cp.cuda.Device(device):
            gpu_U, gpu_Y, gpu_C = self.arrays[device]
            gpu_X = cp.asarray(X)
            return (
                cp.asnumpy(gpu_U.T @ gpu_X),
                cp.asnumpy(gpu_X.T @ gpu_Y),
                cp.asnumpy(gpu_X.T @ gpu_C),
            )

    def project(self, X):
        """Project the supplied genotype block into the prepared eigen/phenotype/covariate bases.

        Parameters
        ----------
        X : array-like
            Predictor matrix with observations in rows and features in columns. Its row order must
            match the response or operator to which it is applied; the selected fitting routine
            determines missing-value support.

        Returns
        -------
        object
            Computed result from ``(np.concatenate([result[0] for result in results], axis=1), np.concatenate([result[1] for result in results], axis=0), np.concatenate([result[2] for result in results], axis=0))``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if len(self.devices) == 1:
            return self._project_slice(self.devices[0], X)
        boundaries = np.linspace(0, X.shape[1], len(self.devices) + 1, dtype=int)
        work = [
            (device, X[:, boundaries[index]:boundaries[index + 1]])
            for index, device in enumerate(self.devices)
            if boundaries[index + 1] > boundaries[index]
        ]
        futures = [
            self.executor.submit(self._project_slice, device, block)
            for device, block in work
        ]
        results = [future.result() for future in futures]
        return (
            np.concatenate([result[0] for result in results], axis=1),
            np.concatenate([result[1] for result in results], axis=0),
            np.concatenate([result[2] for result in results], axis=0),
        )

    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        arrays = getattr(self, "arrays", {})
        arrays.clear()
        if hasattr(self, "cp"):
            for device in getattr(self, "devices", ()):
                with self.cp.cuda.Device(device):
                    self.cp.get_default_memory_pool().free_all_blocks()

    def __del__(self):
        """Release owned resources when the object is finalized.
        """
        try:
            self.close()
        except Exception:
            pass


def create_projection_backend(
    requested, U, Y, C, gpu_devices=(), gpu_memory_budget_mb=None
):
    """Resolve CPU/CUDA execution and construct the genotype projection backend.

    Parameters
    ----------
    requested : str
        Execution/backend selection for this operation. See the source-derived literal
        descriptions below and the called resolver; automatic selection uses the available
        implementation and input representation.

    U : array-like
        Numerical operand in ``CPUProjectionBackend(U, Y, C)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    Y : array-like or pandas.DataFrame
        Phenotype observations, with animals in rows and traits in columns. Preserve the animal
        order used by the relationship kernel; NaN marks unobserved cells where the estimator
        supports missing data.

    C : array-like
        Numerical operand in ``CPUProjectionBackend(U, Y, C)``. Shape must conform to that
        matrix/vector operation and the enclosing prepared state.

    gpu_devices : sequence of int
        CUDA device indices assigned to the projection worker pool. Indices must refer to
        available devices; an empty tuple uses the backend's default device selection. Default:
        ``()``. Passed to ``CUDAProjectionBackend`` as ``devices``.

    gpu_memory_budget_mb : float or None
        Memory budget in MiB for the named CPU/GPU resource guard. Positive values bound
        allocations; None uses available memory or the resolved execution plan. Default:
        ``None``. Passed to ``CUDAProjectionBackend`` as ``memory_budget_mb``.

    Returns
    -------
    object
        Computed result from ``CUDAProjectionBackend(U, Y, C, devices=gpu_devices, memory_budget_mb=gpu_memory_budget_mb)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    name = resolve_backend_name(requested)
    if name == "cpu":
        return CPUProjectionBackend(U, Y, C)
    return CUDAProjectionBackend(
        U,
        Y,
        C,
        devices=gpu_devices,
        memory_budget_mb=gpu_memory_budget_mb,
    )
