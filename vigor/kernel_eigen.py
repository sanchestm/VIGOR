"""Shared randomized kernel eigensystems for the h2 prior pilot and the grm2us sandwich.

One ``npplink.grm2Us`` randomized eigensystem per kernel represents

    K~ = U diag(s) U' + b (I - U U'),   b = (tr K - sum s) / (n - r),

a rank-r approximation with a trace-matched isotropic remainder. Within one
``genetic_varcov_PSD`` call (:func:`shared_kernel_eigensystems`) the pilot
and the final grm2us sandwich reuse the same eigensystem; inside hyper_opt
workers it is also stored under the probe-cache directory, keyed by kernel
content, so every fold and the final refit reuse one decomposition. Cost
O(n² r) once per kernel; storage O(n r).
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import hashlib
import io

import numpy as np

AUTO_RANK = 512
_scope = ContextVar('vigor_kernel_eigensystems', default=None)


def resolve_rank(rank, n, name=None, position=0):
    """Retained rank for one kernel, or None for the full kernel.

    rank is 'auto' (512), 'full'/None/0/False, an integer, or a per-kernel
    mapping (by name) or sequence (by position) of those.
    """
    if isinstance(rank, dict):
        rank = rank.get(name, 'auto')
    elif isinstance(rank, (list, tuple)):
        rank = rank[position]
    if rank is None or rank is False or (isinstance(rank, str) and rank.lower() == 'full'):
        return None
    if isinstance(rank, str) and rank.lower() == 'auto':
        rank = AUTO_RANK
    if int(rank) == 0:
        return None
    return max(1, min(int(rank), int(n) - 1))


def shared_kernel_eigensystems(function):
    """Share eigensystems between the pilot and the sandwich of one covariance fit."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _scope.set(OrderedDict())
        try:
            return function(*args, **kwargs)
        finally:
            _scope.reset(token)
    return wrapped


@contextmanager
def eigensystem_scope():
    """Explicit sharing scope (used by callers outside genetic_varcov_PSD)."""
    token = _scope.set(OrderedDict())
    try:
        yield
    finally:
        _scope.reset(token)


def _cache_directory():
    from .calibration_probes import _scope as probe_scope
    context = probe_scope.get()
    return None if context is None else context[0]


def eigensystem(kernel, dense, rank, seed=0):
    """Randomized eigensystem of one kernel, shared within the fit and the hyper_opt cache.

    kernel is the caller's kernel object (the in-fit sharing key); dense()
    returns its aligned float32 n x n matrix and is called only on a miss.
    Returns a dict with U (float32 n x r), values, bulk, diag (exact kernel
    diagonal), rank and retained_trace_fraction.
    """
    memo = _scope.get()
    key = (id(kernel), rank, int(seed))
    if memo is not None and key in memo:
        return memo[key]
    matrix = np.ascontiguousarray(dense(), dtype=np.float32)
    n = matrix.shape[0]
    path = None
    directory = _cache_directory()
    if directory is not None:
        h = hashlib.sha256()
        h.update(f'{n}:{rank}:{int(seed)}'.encode())
        h.update(matrix.tobytes())
        path = Path(directory) / 'kernel_eigensystems' / f'{h.hexdigest()}.npz'
    out = None
    if path is not None and path.exists():
        with np.load(path) as stored:
            out = {name: stored[name] for name in stored.files}
        out['bulk'] = float(out['bulk']); out['rank'] = int(out['rank'])
        out['retained_trace_fraction'] = float(out['retained_trace_fraction'])
    if out is None:
        from .npplink import grm2Us
        U, values = grm2Us(matrix, rank, decomposition='randomized', random_state=int(seed))
        values = np.maximum(np.asarray(values, float), 0.0)
        diag = np.diag(matrix).astype(np.float64)
        trace = float(diag.sum())
        out = {'U': np.asarray(U, np.float32), 'values': values,
               'bulk': max(trace - float(values.sum()), 0.0) / max(n - rank, 1), 'diag': diag,
               'rank': int(rank), 'retained_trace_fraction': float(values.sum() / trace) if trace > 0 else np.nan}
        if path is not None:
            buffer = io.BytesIO()
            np.savez(buffer, **out)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(path)
    if memo is not None:
        memo[key] = out
        # The memo keeps the kernel object id alive only for this fit; cap it anyway.
        while len(memo) > 8:
            memo.popitem(last=False)
    return out
