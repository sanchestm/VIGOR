"""Spawn-safe, bounded HPO workers and immutable fold-moment caching.

Project functions run only in child processes. The parent owns checkpoints,
heartbeats and cancellation; a native call cannot suppress its watchdog.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import copy
import functools
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import time

_cache = ContextVar('vigor_hpo_moment_cache', default=None)
_limits = ContextVar('vigor_hpo_optimizer_limits', default={})
_point_sink = ContextVar('vigor_hpo_point_sink', default=None)


def acquire_study_lock(path):
    """Acquire a nonblocking OS advisory lock, released automatically on death.

    path is an owned study lock file. Returns its open binary handle. The file
    remains on disk to avoid inode/unlink races. O(1), no waiting or stale-PID
    guessing. Windows locks one byte; POSIX uses flock on the file description.
    """
    handle=open(path,'a+b')
    try:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:handle.write(b' ');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        handle.seek(0);handle.truncate();handle.write(json.dumps({'pid':os.getpid(),'time':time.time()}).encode());handle.flush()
        return handle
    except OSError as exc:
        handle.close();raise RuntimeError('checkpoint already has a live supervisor or file locking is unavailable') from exc


def release_study_lock(handle):
    """Release an acquired OS lock and close the handle; O(1)."""
    try:
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    finally:handle.close()


def atomic_json(path, value):
    """Atomically replace UTF-8 JSON at path; value must contain finite JSON data.

    Time/space O(serialized bytes). fsync precedes same-directory replacement.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_pickle(path, value):
    """Persist an internal trusted-worker result atomically; O(serialized bytes).

    Pickles in checkpoint directories are trusted local state, never user data
    interchange. Do not resume a checkpoint from an untrusted source.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=5); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def optimizer_limits():
    """Return active finite SciPy maxfun/maxls limits, or {} outside HPO; O(1)."""
    return dict(_limits.get())


def checkpoint_point_fit(fit):
    """Commit a converged point fit before optional uncertainty; O(fit bytes).

    Outside supervised final refits this is a no-op. No SE/CI validity is
    claimed for this checkpoint; the caller labels uncertainty separately.
    """
    # Subsequent analytical probes/bootstrap weights are not base moments.
    _cache.set(None)
    sink = _point_sink.get()
    if sink is not None and bool(fit.get('success',False)) and bool(fit.get('optimizer_converged',True)):
        sink(fit)


@contextmanager
def worker_context(cache_dir=None, cache_key=None, limits=None, emit=None, cancel=None, point_sink=None, cache_max_bytes=2*1024**3):
    """Install worker-only cache, optimizer and progress controls; O(1) setup.

    cache_dir/key identify immutable training-only base moments. limits maps
    maxfun/maxls to positive integers. emit receives progress dictionaries;
    cancel is a multiprocessing Event checked at numerical progress callbacks.
    """
    from .progress import progress_events
    token = _cache.set(None if cache_dir is None else
                       {'path': Path(cache_dir), 'key': cache_key, 'emit': emit, 'used': set(), 'max_bytes':cache_max_bytes})
    limit_token = _limits.set(dict(limits or {}))
    point_token = _point_sink.set(point_sink)
    try:
        with progress_events(emit, cancel):
            yield
    finally:
        _cache.reset(token); _limits.reset(limit_token); _point_sink.reset(point_token)


def cached_moments(function):
    """Cache only the first base-moment call per function/fold worker context.

    Inputs/outputs retain the wrapped contract. Cache keys include fold/data,
    preprocessing and numerical settings supplied by the supervisor. Disk
    reads return independent objects; candidate closure/ridge never mutates a
    cached object. Cost is serialization of moments, not another animal kernel.
    """
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        """Return fresh base moments or an isolated trusted cached copy."""
        context = _cache.get()
        if context is None or function.__name__ in context['used']:
            return function(*args, **kwargs)
        context['used'].add(function.__name__)
        digest = hashlib.sha256(str(context['key']).encode()).hexdigest()
        path = context['path'] / digest / (function.__name__ + '.pkl')
        if path.exists():
            with path.open('rb') as stream: value = pickle.load(stream)
            hit = True
        else:
            value = function(*args, **kwargs); hit = False
            estimated=sum(getattr(x,'nbytes',0) for x in value.values()) if isinstance(value,dict) else 0
            used=sum(p.stat().st_size for p in context['path'].rglob('*.pkl')) if context['path'].exists() else 0
            if estimated+used <= context['max_bytes']: atomic_pickle(path, value)
        if context['emit']:
            context['emit']({'phase': 'moment_cache', 'function': function.__name__, 'cache_hit': hit})
        return value
    return wrapped


class WorkerFailure(RuntimeError):
    """Bounded worker failure with its phase and latest progress diagnostics."""
    def __init__(self, reason, latest=None):
        """Store reason and a serializable latest-event dictionary; O(1)."""
        super().__init__(reason); self.latest = latest or {}


def _entry(payload_path, result_path, messages, cancel):
    """Spawn entry point: load trusted payload, run worker, atomically commit.

    Arguments are paths/IPC primitives; no notebook-defined functions are
    pickled as process targets. Exceptions become explicit worker diagnostics.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    last = [0.0]
    def emit(event):
        """Send throttled small diagnostics without blocking numerical work."""
        if cancel.is_set(): raise InterruptedError('cooperative HPO cancellation')
        now = time.monotonic()
        if event.get('iteration') is not None and now-last[0] < 0.2: return
        last[0] = now
        try: messages.put_nowait(dict(event, worker_time=now))
        except queue.Full: pass
    try:
        with open(payload_path, 'rb') as stream: payload = pickle.load(stream)
        from .hyperopt_worker import execute
        result = execute(payload, emit, cancel)
        if cancel.is_set(): raise InterruptedError('cancelled before result commit')
        atomic_pickle(result_path, result)
        messages.put({'phase': 'worker_complete', 'result_committed': True})
    except BaseException as exc:
        atomic_json(str(result_path)+'.error.json',
                    {'type': type(exc).__name__, 'error': str(exc)[:4000]})
        try: messages.put_nowait({'phase': 'worker_failed', 'error': str(exc)[:1000]})
        except queue.Full: pass


def supervised(payload, directory, *, timeout_s, phase_timeout_s, heartbeat_s=5,
               memory_limit_bytes=None, cancel_grace_s=2, emit=None, stop_file=None,
               deadline=None, stage_budgets=None):
    """Execute one spawn child with wall/phase/RSS ceilings and parent heartbeat.

    payload is a trusted internal mapping. directory is a fresh job directory.
    timeout_s bounds the entire job; phase_timeout_s bounds no phase transition
    (iteration events do not reset a phase clock). memory_limit_bytes monitors
    child and descendant RSS, including shared pages conservatively. deadline
    is an optional parent monotonic deadline. Ctrl-C re-raises after cleanup.
    No retries occur here. Space is child working set plus bounded IPC queue.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    payload_path = directory/'payload.pkl'; result_path = directory/'result.pkl'
    atomic_pickle(payload_path, payload)
    context = mp.get_context('spawn'); messages = context.Queue(maxsize=256); cancel = context.Event()
    process = context.Process(target=_entry, args=(str(payload_path), str(result_path), messages, cancel))
    started = time.monotonic(); last_heartbeat = started; phase_started = started
    phase = None; latest = {}; failure = None; process.start()
    milestone='point';milestone_started=started
    monitored=None; psutil=None; monitor_error=None
    try:
        try:
            import psutil
            # Some container runtimes expose the parent /proc mount. Reading a
            # namespace PID there can monitor or kill an unrelated process.
            proc_self=Path('/proc/self')
            mismatched_namespace=proc_self.exists() and os.readlink(proc_self)!=str(os.getpid())
            if mismatched_namespace:
                monitor_error='procfs PID namespace does not match the worker namespace'
            else:
                try:monitored=psutil.Process(process.pid)
                except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:monitor_error=str(exc)
        except ImportError:
            monitor_error='psutil is unavailable'
        if monitor_error:
            latest['rss_monitor_status']='unavailable: '+monitor_error
            if emit:emit(dict(latest))
        # stop_file may be one path or several (user stop file plus the
        # controller's abort flag for parallel sibling workers).
        stop_files = [] if not stop_file else [stop_file] if isinstance(stop_file, (str, os.PathLike)) else [f for f in stop_file if f]
        while process.is_alive():
            now = time.monotonic()
            if any(Path(f).exists() for f in stop_files): raise KeyboardInterrupt
            try:
                while True:
                    event = messages.get_nowait()
                    if event.get('phase') and event.get('phase')!=latest.get('phase'):
                        for key in ('iteration','objective','gradient_norm','objective_delta','objective_relative_delta'):latest.pop(key,None)
                    latest.update(event)
                    if event.get('point_fit_committed') and milestone=='point':
                        milestone='uncertainty';milestone_started=now
                    event_phase = event.get('phase') or event.get('optimizer_phase')
                    if event_phase and event_phase != phase and event_phase != 'heartbeat':
                        phase = event_phase; phase_started = now
                    if emit: emit(event)
            except queue.Empty: pass
            rss = None
            if monitored is not None:
                try:rss=sum(p.memory_info().rss for p in [monitored]+monitored.children(recursive=True) if p.is_running())
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if now-started > timeout_s or (deadline is not None and now >= deadline): failure = 'wall_timeout'
            elif now-phase_started > phase_timeout_s: failure = 'phase_timeout'
            elif memory_limit_bytes is not None and rss is None:failure='memory_monitor_unavailable'
            elif memory_limit_bytes is not None and rss is not None and rss > memory_limit_bytes: failure = 'memory_limit'
            if stage_budgets and now-milestone_started>stage_budgets[milestone]:failure=milestone+'_timeout'
            if now-last_heartbeat >= heartbeat_s:
                if emit: emit(dict(latest, phase='heartbeat', active_phase=phase,
                                   elapsed_s=now-started, worker_rss_bytes=rss,
                                   event_age_s=now-latest.get('worker_time', started)))
                last_heartbeat = now
            if failure: raise WorkerFailure(failure, latest)
            process.join(timeout=min(0.2, heartbeat_s))
        process.join()
        error_path = Path(str(result_path)+'.error.json')
        if process.exitcode != 0 or error_path.exists() or not result_path.exists():
            error = json.loads(error_path.read_text()) if error_path.exists() else {'error': f'exit {process.exitcode}; no committed result'}
            raise WorkerFailure(error['error'], dict(latest, worker_error=error))
        with result_path.open('rb') as stream: return pickle.load(stream)
    finally:
        if process.is_alive():
            descendants=[]
            if monitored is not None:
                try:descendants=monitored.children(recursive=True)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            cancel.set(); process.join(timeout=cancel_grace_s)
            if process.is_alive(): process.terminate(); process.join(timeout=cancel_grace_s)
            if process.is_alive(): process.kill(); process.join(timeout=cancel_grace_s)
            for child in descendants:
                try: child.kill()
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        messages.close(); messages.cancel_join_thread()
        if not process.is_alive(): process.close()
        payload_path.unlink(missing_ok=True)
