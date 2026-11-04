"""Resource planning for bounded-memory vigor execution.

The planner is deliberately conservative.  A requested variant block size is
treated as a ceiling, never a target that is allowed to exceed the process
memory budget.  The same contract is used by the GRM and association stages so
large jobs fail during planning rather than after allocating a dense matrix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Optional, Sequence
import warnings
import threading
import re

import numpy as np

#: Share of physical RAM used as the working budget when no ``maxmem`` is given.
DEFAULT_MEMORY_FRACTION = 0.80
#: Default floating type for heavy genotype/GRM work (half the bytes and ~2x BLAS rate of float64).
DEFAULT_HEAVY_DTYPE = np.float32


def parse_memory_bytes(value):
    """Parse a positive byte count or binary-unit string (for example '8GiB').

    Parameters
    ----------
    value : object (see the cited input contract)
        Input for the operation
        ``re.fullmatch('\\\\s*([0-9]*\\\\.?[0-9]+)\\\\s*(b|kb|kib|mb|mib|gb|gib|tb|tib)?\\\\s*',
        value.lower())``. See ``vigor.resources.parse_memory_bytes`` and its caller for the
        scientific representation; the shown operation constrains the required type/shape.
        Source comparisons: ``result <= 0``. See the rejection/fallback branches for
        enforcement.

    Returns
    -------
    object
        Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if isinstance(value, str):
        match = re.fullmatch(r'\s*([0-9]*\.?[0-9]+)\s*(b|kb|kib|mb|mib|gb|gib|tb|tib)?\s*', value.lower())
        if match is None:
            raise ValueError(f'invalid memory limit: {value!r}')
        units = {'b': 0, 'kb': 1, 'kib': 1, 'mb': 2, 'mib': 2, 'gb': 3, 'gib': 3, 'tb': 4, 'tib': 4}
        value = float(match.group(1)) * 1024 ** units[match.group(2) or 'b']
    result = int(value)
    if result <= 0:
        raise ValueError('memory limit must be positive')
    return result


def available_memory_bytes():
    """Current available RAM, capped by a Linux container's remaining allowance.

    Returns
    -------
    object
        Computed result from ``available``; see the summary and the called implementation for its labeled/numerical representation.
    """
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except ImportError:
        available = _physical_memory_mb() * 1024 ** 2 // 2
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            available = min(available, max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return available


def total_memory_bytes():
    """Physical RAM, capped by a Linux container's memory limit."""
    total = _physical_memory_mb() * 1024 ** 2
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            total = min(total, int(limit))
    except (OSError, ValueError):
        pass
    return int(total)


def usable_cpu_count():
    """CPUs this process may run on (affinity/cgroup aware where the OS exposes it)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def working_memory_bytes(maxmem=None, memory_fraction=None, workers=1):
    """Working-memory budget shared by every chunk planner.

    The budget is ``maxmem`` when given, otherwise ``memory_fraction`` (default
    80%) of physical RAM. Either way it never exceeds 90% of the RAM currently
    available, so a busy machine (or a planner nested inside an already
    allocated block) shrinks blocks instead of swapping. Available memory is
    re-read on each call, and parallel workers share the budget.

    Parameters
    ----------
    maxmem : int, float, unit string or None
        Working-memory budget in bytes; a string such as 8GiB uses binary units
        (B/KB/KiB/MB/MiB/GB/GiB/TB/TiB). None uses memory_fraction of physical RAM.
        Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None, in (0, 0.95]. None uses
        ``DEFAULT_MEMORY_FRACTION`` (0.80). Default: ``None``.

    workers : int
        Number of concurrent workers splitting the budget. Default: ``1``.

    Returns
    -------
    int
        Bytes available to one worker for transient blocks.
    """
    frac = DEFAULT_MEMORY_FRACTION if memory_fraction is None else float(memory_fraction)
    if not 0 < frac <= .95:
        raise ValueError('memory_fraction must be in (0, .95]')
    if int(workers) < 1:
        raise ValueError('workers must be positive')
    available = available_memory_bytes()
    wanted = int(total_memory_bytes() * frac) if maxmem is None else parse_memory_bytes(maxmem)
    return min(wanted, int(available * .9)) // int(workers)


def bounded_variant_chunk(n_samples, *, requested='auto', maxmem=None,
                          memory_fraction=None, workers=1, bytes_per_call=16,
                          bytes_per_variant=0, max_chunk=None, alignment=256,
                          reserve_bytes=0, n_variants=None):
    """Variants per block that fit the working-memory budget.

    ``per_variant = n_samples * bytes_per_call + bytes_per_variant`` is the
    scratch one variant column costs; ``reserve_bytes`` (persistent outputs
    such as an n x n GRM) is subtracted from the budget first. ``'auto'`` takes
    everything affordable; an integer request is a ceiling.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis.

    requested : int or 'auto'
        Requested variants per block; integers are ceilings. Default: ``'auto'``.

    maxmem : int, float, unit string or None
        Working-memory budget; see ``working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM when maxmem is None; None means 0.80. Default: ``None``.

    workers : int
        Concurrent workers sharing the budget. Default: ``1``.

    bytes_per_call : int or float
        Scratch bytes per (sample, variant) cell across all working copies. Default: ``16``.

    bytes_per_variant : int
        Extra scratch bytes per variant independent of n_samples. Default: ``0``.

    max_chunk : int or None
        Optional hard ceiling; None means memory alone decides. Default: ``None``.

    alignment : int
        Blocks at least this wide are rounded down to a multiple of it. Default: ``256``.

    reserve_bytes : int
        Bytes held by persistent outputs, removed from the budget first. Default: ``0``.

    n_variants : int or None
        Total variants to process; the block never exceeds it. Default: ``None``.

    Returns
    -------
    int
        Variants per block.
    """
    per_variant = max(1, int(np.ceil(int(n_samples) * float(bytes_per_call))) + int(bytes_per_variant))
    budget = working_memory_bytes(maxmem, memory_fraction, workers) - int(reserve_bytes)
    affordable = budget // per_variant
    if affordable < 1:
        raise MemoryError(f'one genotype variant needs {per_variant} bytes after reserving '
                          f'{int(reserve_bytes)} bytes; block budget is {budget}')
    chunk = affordable
    if str(requested).lower() != 'auto':
        if int(requested) < 1:
            raise ValueError('variant chunk must be positive or auto')
        chunk = min(chunk, int(requested))
    if max_chunk is not None:
        chunk = min(chunk, int(max_chunk))
    if n_variants is not None:
        chunk = min(chunk, max(1, int(n_variants)))
    if chunk >= int(alignment):
        chunk = chunk // int(alignment) * int(alignment)
    return int(chunk)


def resolve_variant_chunk(requested, n_samples, *, bytes_per_call=12, bytes_per_variant=0,
                          maxmem=None, memory_fraction=None, workers=1, reserve_bytes=0,
                          n_variants=None, alignment=256):
    """Resolve a user-facing ``block_size``/``chunk_variants`` argument to an int.

    ``'auto'`` or None sizes the block from the working-memory budget (default
    80% of RAM); an integer is honored as a ceiling and still clipped to what
    the budget affords. ``bytes_per_call`` defaults to a float32 decode plus
    two working copies.
    """
    return bounded_variant_chunk(
        n_samples, requested='auto' if requested is None else requested, maxmem=maxmem,
        memory_fraction=memory_fraction, workers=workers, bytes_per_call=bytes_per_call,
        bytes_per_variant=bytes_per_variant, alignment=alignment,
        reserve_bytes=reserve_bytes, n_variants=n_variants)


class PeakRSSMonitor:
    """Sample process RSS for run metadata and budget overrun detection."""

    def __init__(self, interval_seconds=0.05):
        """Initialize PeakRSSMonitor with the inputs and resource settings below.

        Parameters
        ----------
        interval_seconds : float
            Minimum interval in seconds between progress refreshes or resident-memory samples.
            Positive values limit monitoring overhead; smaller intervals give more frequent updates.
            Default: ``0.05``.
        """
        try:
            import psutil  # type: ignore

            self._process = psutil.Process()
        except ImportError:
            self._process = None
        self.interval_seconds = interval_seconds
        self.baseline_bytes = self.current_bytes
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = None

    @property
    def current_bytes(self):
        """Return current process resident-memory usage in bytes.

        Returns
        -------
        object
            Computed result from ``None if self._process is None else int(self._process.memory_info().rss)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return None if self._process is None else int(self._process.memory_info().rss)

    def start(self):
        """Start background sampling of resident-memory usage.

        Returns
        -------
        object
            Computed result from ``self``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if self._process is None or self._thread is not None:
            return self

        def sample():
            """Draw/summarize the next value from the enclosing random or resource-monitor process.
                        
            """
            while not self._stop.wait(self.interval_seconds):
                value = self.current_bytes
                if value is not None:
                    self.peak_bytes = max(self.peak_bytes or 0, value)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def snapshot(self):
        """Return the latest peak/current resource-monitor statistics.

        Returns
        -------
        dict
            Result fields: ``rss_mb``, ``peak_rss_mb``, ``peak_increment_mb``. Field arrays preserve the corresponding input axis order.
        """
        value = self.current_bytes
        if value is not None:
            self.peak_bytes = max(self.peak_bytes or 0, value)
        return {
            "rss_mb": None if value is None else value / 1024 ** 2,
            "peak_rss_mb": None if self.peak_bytes is None else self.peak_bytes / 1024 ** 2,
            "peak_increment_mb": (
                None
                if self.peak_bytes is None or self.baseline_bytes is None
                else (self.peak_bytes - self.baseline_bytes) / 1024 ** 2
            ),
        }

    def check_budget(self, memory_budget_mb, context="run"):
        """Check budget.

        Parameters
        ----------
        memory_budget_mb : float or str
            Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
            descriptive text identifying the operation in an allocation error.

        context : float or str
            Resource guard budget/context: memory_budget_mb is a positive MiB limit; context is
            descriptive text identifying the operation in an allocation error. Default: ``'run'``.
        """
        value = self.current_bytes
        if value is not None and value > memory_budget_mb * 1024 ** 2:
            raise MemoryError(
                f"observed RSS exceeded memory_budget_mb during {context}: "
                f"{value / 1024 ** 2:.0f} MiB > {memory_budget_mb} MiB"
            )

    def stop(self):
        """Stop background memory sampling and collect the final usage snapshot.

        Returns
        -------
        object
            Computed result from ``self.snapshot()``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
            self._thread = None
        return self.snapshot()


def _physical_memory_mb() -> int:
    """Best-effort physical-memory detection without a required dependency.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total // (1024 ** 2))
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                """_MemoryStatus represents memory status.
                """
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys // (1024 ** 2))
        except (AttributeError, OSError):
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 ** 2))
    except (AttributeError, ValueError, OSError):
        return 8192


@dataclass(frozen=True)
class ResourceConfig:
    """User-visible execution limits.

    ``memory_budget_mb`` is a hard process budget used for planning.  When it
    is omitted, vigor uses ``DEFAULT_MEMORY_FRACTION`` (80%) of physical memory.
    CUDA fields are recorded here even when a CPU execution path is selected,
    which keeps command lines and metadata stable across hosts.

    Constructor fields
    ------------------
    memory_budget_mb : Optional[int]
        Stored memory budget mb consumed by ``resolved``; see those methods for the
        numerical/label contract. Default: ``None``.

    n_jobs : int
        Stored n jobs consumed by ``resolved``; see those methods for the numerical/label
        contract. Default: ``1``.

    backend : str
        Stored backend consumed by ``resolved``; see those methods for the numerical/label
        contract. Default: ``'cpu'``.

    gpu_devices : Sequence[int]
        Stored gpu devices consumed by ``resolved``; see those methods for the numerical/label
        contract. Default: ``()``.

    gpu_memory_budget_mb : Optional[int]
        Stored gpu memory budget mb consumed by ``resolved``; see those methods for the
        numerical/label contract. Default: ``None``.

    scratch_dir : Optional[str]
        Stored scratch dir consumed by ``resolved``; see those methods for the numerical/label
        contract. Default: ``None``.

    target_runtime_hours : Optional[float]
        Stored target runtime hours consumed by ``resolved``; see those methods for the
        numerical/label contract. Default: ``None``.

    safety_fraction : float
        Fraction of configured memory reserved as a safety margin by the resource planner.
        Default: ``0.9``.
    """

    memory_budget_mb: Optional[int] = None
    n_jobs: int = 1
    backend: str = "cpu"
    gpu_devices: Sequence[int] = ()
    gpu_memory_budget_mb: Optional[int] = None
    scratch_dir: Optional[str] = None
    target_runtime_hours: Optional[float] = None
    safety_fraction: float = 0.90

    def resolved(self) -> "ResourceConfig":
        """Validate resource settings and resolve backend, memory and worker defaults.

        Returns
        -------
        object
            Computed result from ``ResourceConfig(memory_budget_mb=int(budget), n_jobs=int(jobs), backend=backend, gpu_devices=tuple((int(device) for device in self.gpu_devices)), gpu_memory_budget_mb=self.gpu_memory_budget_mb, scra...``; see the summary and the called implementation for its labeled/numerical representation.
        """
        physical = _physical_memory_mb()
        budget = self.memory_budget_mb
        if budget is None:
            budget = max(512, int(physical * DEFAULT_MEMORY_FRACTION))
        if budget <= 0:
            raise ValueError("memory_budget_mb must be positive")
        if self.memory_budget_mb is not None and budget > physical:
            warnings.warn(
                f"memory_budget_mb={budget} exceeds detected physical memory "
                f"({physical} MiB); explicit budget retained for scheduler/container "
                "environments, but the host must actually provide this limit",
                RuntimeWarning,
            )
        jobs = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        if jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        backend = self.backend.lower()
        if backend not in {"cpu", "cuda", "auto"}:
            raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
        if not 0.50 <= self.safety_fraction <= 0.95:
            raise ValueError("safety_fraction must be between 0.50 and 0.95")
        if self.gpu_memory_budget_mb is not None and self.gpu_memory_budget_mb <= 0:
            raise ValueError("gpu_memory_budget_mb must be positive")
        if self.target_runtime_hours is not None and self.target_runtime_hours <= 0:
            raise ValueError("target_runtime_hours must be positive")
        scratch = None if self.scratch_dir is None else str(Path(self.scratch_dir).resolve())
        return ResourceConfig(
            memory_budget_mb=int(budget),
            n_jobs=int(jobs),
            backend=backend,
            gpu_devices=tuple(int(device) for device in self.gpu_devices),
            gpu_memory_budget_mb=self.gpu_memory_budget_mb,
            scratch_dir=scratch,
            target_runtime_hours=self.target_runtime_hours,
            safety_fraction=self.safety_fraction,
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Concrete block/tile choices and their estimated peak memory.

    Constructor fields
    ------------------
    memory_budget_mb : int
        Stored memory budget mb supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    variant_block_size : int
        Stored variant block size supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    trait_tile_size : int
        Stored trait tile size supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    estimated_peak_mb : float
        Stored estimated peak mb supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    persistent_mb : float
        Stored persistent mb supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    working_mb : float
        Stored working mb supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    decomposition_peak_mb : float
        Stored decomposition peak mb supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    output_scratch_gb : Optional[float]
        Stored output scratch gb supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    n_jobs : int
        Stored n jobs supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    backend : str
        Stored backend supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    compute_dtype : str
        Stored compute dtype supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.
    """

    memory_budget_mb: int
    variant_block_size: int
    trait_tile_size: int
    estimated_peak_mb: float
    persistent_mb: float
    working_mb: float
    decomposition_peak_mb: float
    output_scratch_gb: Optional[float]
    n_jobs: int
    backend: str
    compute_dtype: str

    def as_dict(self):
        """Convert the configuration dataclass to a dictionary of its fields.

        Returns
        -------
        object
            Computed result from ``asdict(self)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return asdict(self)


def plan_execution(
    n_samples: int,
    n_traits: int,
    n_covariates: int,
    rank: int,
    requested_block_size="auto",
    model: str = "add-dom",
    compute_dtype="float32",
    decomposition: str = "exact",
    n_variants: Optional[int] = None,
    resource_config: Optional[ResourceConfig] = None,
) -> ExecutionPlan:
    """Choose bounded variant blocks and trait tiles for one analysis group.

    Parameters
    ----------
    n_samples : int
        Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
        length, not an arbitrary truncation limit.

    n_traits : int
        Number of traits in the ordered response/covariance axis; must agree with every
        corresponding matrix dimension.

    n_covariates : int
        Input for the operation ``n_covariates < 1``. See ``vigor.resources.plan_execution`` and
        its caller for the scientific representation; the shown operation constrains the
        required type/shape. Source comparisons: ``n_covariates < 1``. See the
        rejection/fallback branches for enforcement.

    rank : int
        Retained matrix/factor rank. Integer ranks must be compatible with the relevant animal
        or trait dimension. Full and automatic policies depend on the decomposition; see the
        rank resolver used by this routine.

    requested_block_size : int or "auto"
        Input for the operation ``min(n_samples, n_traits, requested_block_size) < 1``. See
        ``vigor.resources.plan_execution`` and its caller for the scientific representation; the
        shown operation constrains the required type/shape.

    model : str
        Model selection or fitted estimator consumed below. String choices select the documented
        genetic/simulation model; estimator objects must implement the fit/predict methods used
        here. Default: ``'add-dom'``. Source-recognized literal values: ``additive``. These
        select the corresponding branches below; comparison evidence alone does not establish an
        exhaustive enum.

    compute_dtype : NumPy dtype or str
        Numerical element type for the corresponding output, working block or accumulated
        matrix. Use a NumPy-compatible dtype; float32 reduces memory while float64 improves
        numerical precision. Default: ``'float32'``.

    decomposition : str
        Kernel eigendecomposition: 'exact' uses a symmetric dense decomposition, 'randomized' a
        retained operator eigenspace with a trace-matched bulk, and 'auto' chooses exact for
        n<=4000 and randomized above it in the LOCO resolver. Default: ``'exact'``.

    n_variants : Optional[int]
        Number of variants in the logical source axis, including the unpadded extent of packed
        arrays. Default: ``None``. None delegates to the local fallback shown in this body; it
        is not an additional numeric/category value.

    resource_config : Optional[ResourceConfig]
        Resolved resource/chunking plan for this workload. None lets the resource planner derive
        a plan; explicit settings must fit the available memory and worker budget. Default:
        ``None``.

    Returns
    -------
    object
        Computed result from ``ExecutionPlan(memory_budget_mb=int(config.memory_budget_mb), variant_block_size=int(block_size), trait_tile_size=int(trait_tile), estimated_peak_mb=peak / 1024 ** 2, persistent_mb=persistent / 1024...``; see the summary and the called implementation for its labeled/numerical representation.
    """
    auto_block = requested_block_size is None or str(requested_block_size).lower() == "auto"
    if min(n_samples, n_traits, 1 if auto_block else int(requested_block_size)) < 1:
        raise ValueError("n_samples, n_traits, and requested_block_size must be positive")
    if n_covariates < 1:
        raise ValueError("n_covariates must include at least the intercept")
    if model not in {"add-dom", "additive"}:
        raise ValueError("model must be 'add-dom' or 'additive'")
    dtype = np.dtype(compute_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("compute_dtype must be float32 or float64")

    config = (resource_config or ResourceConfig()).resolved()
    budget = int(config.memory_budget_mb) * 1024 ** 2
    usable = int(budget * config.safety_fraction)
    n, t, p, r = n_samples, n_traits, n_covariates, min(rank, n_samples)
    item = dtype.itemsize

    # U, Y, C, variance weights, and a fixed writer/allocator reserve.
    persistent = n * r * item + n * t * 8 + n * p * 8
    persistent += r * t * 8 + (t * p * p + t * p) * 8
    reserve = max(256 * 1024 ** 2, int(0.05 * budget))

    decomp = decomposition.lower()
    if decomp not in {"exact", "randomized", "auto"}:
        raise ValueError("decomposition must be 'exact', 'randomized', or 'auto'")
    resolved_decomp = "exact" if decomp == "auto" and n <= 4000 else decomp
    if resolved_decomp == "auto":
        resolved_decomp = "randomized"
    if resolved_decomp == "exact":
        # Genome and chromosome cross-products/counts plus LAPACK workspace.
        decomposition_peak = 6 * n * n * 8 + reserve
    else:
        q = min(n, r + min(32, max(8, r // 10)))
        decomposition_peak = (3 * n * q + q * q) * item + reserve

    if decomposition_peak > usable:
        suggestion = "use decomposition='randomized' with a smaller n_components"
        raise MemoryError(
            f"{resolved_decomp} LOCO decomposition is estimated to require "
            f"{decomposition_peak / 1024 ** 2:.0f} MiB, above the usable "
            f"{usable / 1024 ** 2:.0f} MiB budget; {suggestion}"
        )

    base = persistent + reserve
    if base >= usable:
        raise MemoryError(
            "persistent eigenbasis/phenotype storage exceeds the memory budget; "
            "reduce n_components or the number of simultaneously analyzed traits"
        )

    free = usable - base
    genotype_arrays = 3 if model == "additive" else 6
    projection_arrays = 1 if model == "additive" else 2
    bytes_per_variant = n * item * genotype_arrays + r * item * projection_arrays
    metric_count = 10 if model == "additive" else 31
    bytes_per_variant += max(64, p * 16) + t * metric_count * 8
    block_allocation = int(free * 0.62)
    output_row_cap = 1_000_000
    output_limited_block = max(1, output_row_cap // t)
    block_size = max(
        1,
        min(
            block_allocation if auto_block else int(requested_block_size),
            output_limited_block,
            block_allocation // max(bytes_per_variant, 1),
        ),
    )

    block_base = block_size * bytes_per_variant
    trait_free = max(1, free - block_base)
    bytes_per_tiled_trait = (
        n * 8
        + r * 8
        + block_size * (metric_count * 8 + (2 * p + 8) * 8)
        + (p * p + 4 * p) * 8
    )
    trait_tile = max(1, min(t, trait_free // max(bytes_per_tiled_trait, 1)))
    working = block_base + trait_tile * bytes_per_tiled_trait
    peak = max(base + working, decomposition_peak)
    if peak > budget:
        raise MemoryError("no association block fits within memory_budget_mb")

    scratch_gb = None
    if n_variants is not None:
        bytes_per_row = 70 if model == "add-dom" else 38
        scratch_gb = n_variants * t * bytes_per_row / 1024 ** 3
    return ExecutionPlan(
        memory_budget_mb=int(config.memory_budget_mb),
        variant_block_size=int(block_size),
        trait_tile_size=int(trait_tile),
        estimated_peak_mb=peak / 1024 ** 2,
        persistent_mb=persistent / 1024 ** 2,
        working_mb=working / 1024 ** 2,
        decomposition_peak_mb=decomposition_peak / 1024 ** 2,
        output_scratch_gb=scratch_gb,
        n_jobs=config.n_jobs,
        backend=config.backend,
        compute_dtype=dtype.name,
    )


def factor_scan_chunk(n_samples, rank, *, memory_fraction=None, maxmem=None, max_chunk=None):
    """Resolve a factor feature block using the common memory planner.

    Inputs: n_samples positive animal count; rank nonnegative effect count;
    memory_fraction/maxmem shared budget controls; max_chunk optional ceiling.
    Returns a positive variant count, aligned to 4096 when the budget permits.
    Complexity: O(1) arithmetic plus the common available-memory query.
    The 14 bytes/call + 16 bytes/factor estimate applies to the historical
    feature scorer, not the exact PCG null and its additional kernel workspace.
    """
    return bounded_variant_chunk(n_samples,maxmem=maxmem,memory_fraction=memory_fraction,
        bytes_per_call=14,bytes_per_variant=16*int(rank),max_chunk=max_chunk,alignment=4096)
