"""Reuse of identical covariance fits across hyper_opt candidates of one fold.

Prediction-time settings (shrink_G, shrink_E, E_structure) are applied after
the L-BFGS covariance fit, so candidates that differ only in them share the
same fit. The key is the exact moment system (A, B), the h2-prior weights and
every optimizer argument; any difference refits. Active only inside
:func:`fit_cache` (the hyper_opt worker), never in ordinary calls.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import pickle

import numpy as np

_scope = ContextVar('vigor_fit_cache', default=None)


@contextmanager
def fit_cache(directory=None, key=None, max_bytes=2 * 1024**3):
    """Share covariance fits under the supervisor's immutable data/fold identity."""
    token = _scope.set(None if directory is None else (Path(directory), str(key), int(max_bytes)))
    try:
        yield
    finally:
        _scope.reset(token)


def _digest(moments, h2_penalty, arguments):
    h = hashlib.sha256()
    for name in ('A', 'B'):
        h.update(np.ascontiguousarray(np.asarray(moments[name], float)).tobytes())
    if h2_penalty is not None:
        h.update(np.asarray(h2_penalty['kappa'], float).tobytes())
        h.update(np.asarray(h2_penalty['codes'], np.int64).tobytes())
    h.update(repr(sorted((k, repr(v)) for k, v in arguments.items())).encode())
    return h.hexdigest()


def cached_fit(function, moments, *, h2_penalty=None, **arguments):
    """function(moments, h2_penalty=..., **arguments), reused from the active cache when identical."""
    context = _scope.get()
    if context is None:
        return function(moments, h2_penalty=h2_penalty, **arguments)
    directory, identity, budget = context
    keyed = dict(arguments)
    if keyed.get('warm_start') is not None:
        # A deterministic warm start (the incumbent's fold fit) is part of the key.
        keyed['warm_start'] = hashlib.sha256(pickle.dumps(keyed['warm_start'], protocol=5)).hexdigest()
    path = directory / 'covariance_fits' / f'{_digest(moments, h2_penalty, {**keyed, "_identity": identity})}.pkl'
    if path.exists():
        with path.open('rb') as stream:
            fit = pickle.load(stream)
        fit['fit_cache_hit'] = True
        return fit
    fit = function(moments, h2_penalty=h2_penalty, **arguments)
    try:
        payload = pickle.dumps(fit, protocol=5)
    except Exception:
        return fit
    used = sum(p.stat().st_size for p in directory.rglob('*.pkl')) if directory.exists() else 0
    if used + len(payload) <= budget:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return fit
