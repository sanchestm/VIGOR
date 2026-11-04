"""Low-overhead notebook/terminal stage reporting shared by VIGOR workflows."""
from __future__ import annotations
import time
import sys
from contextvars import ContextVar
from contextlib import contextmanager
from tqdm.auto import tqdm

_active = ContextVar('domino_covariance_progress', default=None)
_work = ContextVar('domino_covariance_work', default=None)
_event_sink = ContextVar('vigor_progress_event_sink', default=None)
_cancellation = ContextVar('vigor_progress_cancellation', default=None)


def report_progress(**event):
    """Emit small structured loop diagnostics to the owning display/supervisor."""
    _emit_progress(event)


@contextmanager
def progress_events(sink=None, cancellation=None):
    """Capture numeric progress without rendering in a supervised HPO worker.

    sink accepts small event dictionaries; cancellation exposes is_set().
    Defaults disable capture. Time/space O(1) per event, excluding the sink.
    """
    event_token = _event_sink.set(sink); cancel_token = _cancellation.set(cancellation)
    try: yield
    finally:
        _event_sink.reset(event_token); _cancellation.reset(cancel_token)


def _emit_progress(event):
    """Forward a progress event and honor cooperative cancellation; O(1)."""
    cancellation = _cancellation.get()
    if cancellation is not None and cancellation.is_set():
        raise InterruptedError('HPO worker cancelled')
    sink = _event_sink.get()
    if sink is not None: sink(event)


def _in_notebook():
    """Detect whether the current process is running in an IPython notebook environment.
        
    """
    try:
        from IPython import get_ipython
        shell=get_ipython()
        return shell is not None and shell.__class__.__name__!='TerminalInteractiveShell'
    except ImportError:
        return False


class _FastBar:
    """fastprogress master/child bars plus replaceable phase statistics.

    Optimizer totals remain unknown: maxiter is a budget, never a completion
    percentage or an ETA. Redirected scripts receive throttled text instead.
    """
    def __init__(self, total, desc, interval, parent=None, style='split+stats'):
        """Initialize _FastBar with the inputs and resource settings below.

        Parameters
        ----------
        total : int or None
            Count limit for the corresponding sampled/retained/displayed items. Positive values
            bound work/output size; None uses all eligible items or the local default where
            supported. Passed to ``cls`` as ``total``.

        desc : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output.

        interval : float
            Minimum interval in seconds between progress refreshes or resident-memory samples.
            Positive values limit monitoring overhead; smaller intervals give more frequent updates.

        parent : progress/graph object, node identifier or None
            Parent progress object or graph node used by the enclosing operation. None omits the
            parent relationship; graph methods require an existing node and progress methods a
            compatible bar. Default: ``None``.

        style : str
            Progress display style: 'split+stats' separates the bar and statistics, 'combined'
            combines them and 'combined+stats' combines with statistics where supported. See the
            local progress validator for accepted styles. Default: ``'split+stats'``.
        """
        import fastprogress.fastprogress as fp
        self.total=total;self.n=0;self.desc=desc;self.detail='';self.closed=False
        self.start=time.monotonic();self.last=-float('inf');self.interval=interval
        self.style=style;self.parent=parent;self.rows={} if parent is None else parent.rows
        self.status_rows=None
        self.key=len(self.rows);self.rows[self.key]=self
        self.interactive=_in_notebook() or sys.stdout.isatty()
        self.native=None;self.master=None
        if self.interactive:
            if parent is None:
                cls=fp.NBMasterBar if _in_notebook() else fp.master_bar
                self.master=cls([],total='noinfer',clean_on_interrupt=False)
                self.native=self.master.main_bar;self.native.update(0)
            else:
                self.master=parent.master
                cls=fp.NBProgressBar if _in_notebook() else fp.progress_bar
                self.native=cls([],total='noinfer' if total is None else total,parent=self.master)
                self.native.update(0)
            self.native.total=total
        self.refresh()

    def set_description(self,value,refresh=False):
        """Set the progress display description and optionally refresh the renderer.

        Parameters
        ----------
        value : object (see the cited input contract)
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller.

        refresh : bool
            Whether to force a display refresh or include memory statistics at this progress stage.
            False leaves the interval/default rendering policy in effect. Default: ``False``.
        """
        self.desc=value
        if refresh:self.refresh()

    def set_postfix_str(self,value,refresh=False):
        """Set the progress display statistics text and optionally refresh it.

        Parameters
        ----------
        value : object (see the cited input contract)
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller.

        refresh : bool
            Whether to force a display refresh or include memory statistics at this progress stage.
            False leaves the interval/default rendering policy in effect. Default: ``False``.
        """
        self.detail=value
        if refresh:self.refresh()

    def _text(self):
        """Format elapsed-time and progress statistics for the text renderer.
        """
        elapsed=max(time.monotonic()-self.start,1e-9)
        count=f'{self.n}/'+('?' if self.total is None else str(self.total))
        if self.status_rows is not None:
            return f'{self.desc} | {count} | elapsed={elapsed:.1f}s | {self.detail}'
        rate=self.n/elapsed
        # Known pair/kernel/block totals justify an ETA; optimizer iterations do not.
        eta='?' if self.total is None or self.n==0 else f'{max(0,self.total-self.n)/rate:.1f}s'
        return f'{self.desc} | {count} | elapsed={elapsed:.1f}s | {rate:.2f}/s | remaining={eta} | {self.detail}'

    def refresh(self,force=False):
        """Refresh the progress renderer if the configured interval has elapsed.

        Parameters
        ----------
        force : bool
            Whether to force a display refresh or include memory statistics at this progress stage.
            False leaves the interval/default rendering policy in effect. Default: ``False``.
        """
        now=time.monotonic()
        if not force and now-self.last<self.interval:return
        self.last=now
        if self.native is None:
            print(self._text(),file=sys.stdout,flush=True)
            return
        self.native.total=self.total;self.native.comment=self.desc+' '+self.detail
        self.native.last_t=time.time()
        self.native.pred_t=None if self.status_rows is not None or self.total is None or not self.n else (self.native.last_t-self.native.start_t)*self.total/self.n
        # No update(total) call: fastprogress would automatically close/reset a
        # task. Lifecycle belongs to the numerical phase, not its last count.
        self.native.update_bar(self.n)
        if '+stats' in self.style and _in_notebook():
            import fastprogress.fastprogress as fp
            rows=self.status_rows
            if rows is None:
                rows=[['Phase / kernel / solver','Completed / total','Elapsed','Rate','Diagnostics / status']]
                selected=list(self.rows.values()) if self.style.startswith('split') else [self]
                for row in selected:
                    seconds=max(now-row.start,1e-9)
                    rows.append([row.desc,f'{row.n}/'+('?' if row.total is None else str(row.total)),f'{seconds:.1f}s',f'{row.n/seconds:.2f}/s',row.detail or ('closed' if row.closed else 'running')])
            self.master.text_parts=[fp.text2html_table(rows)];self.master.show()

    def update(self,n):
        """Advance the progress counter by the requested increment.

        Parameters
        ----------
        n : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit.
        """
        self.n+=int(n);self.refresh()

    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        if self.closed:return
        self.closed=True;self.refresh(force=True)
        if self.native is not None:self.native.on_iter_end()



class StageProgress:
    """Selected fastprogress/tqdm reporting with throttled RSS and a quiet path."""
    def __init__(self, total=None, desc='', enabled=True, unit='step', mininterval=.5, position=None, backend='auto', style='split+stats', parent=None):
        """Initialize StageProgress with the inputs and resource settings below.

        Parameters
        ----------
        total : int or None
            Count limit for the corresponding sampled/retained/displayed items. Positive values
            bound work/output size; None uses all eligible items or the local default where
            supported. Default: ``None``. Passed to ``tqdm`` as ``total``.

        desc : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output. Default: ``''``. Passed to ``tqdm`` as ``desc``.

        enabled : bool
            Whether to emit progress/status reporting during this operation. False disables the
            corresponding reporting path. Default: ``True``.

        unit : str
            Display unit for one progress increment, such as step or iter; it labels counts rather
            than changing numerical computation. Default: ``'step'``. Passed to ``tqdm`` as
            ``unit``.

        mininterval : float
            Minimum interval in seconds between progress refreshes or resident-memory samples.
            Positive values limit monitoring overhead; smaller intervals give more frequent updates.
            Default: ``0.5``. Passed to ``tqdm`` as ``mininterval``.

        position : int or None
            Progress-bar display row. Nonnegative values select a row for nested bars; None lets the
            renderer place it automatically. Default: ``None``.

        backend : str
            Execution/backend selection for this operation. See the source-derived literal
            descriptions below and the called resolver; automatic selection uses the available
            implementation and input representation. Default: ``'auto'``. Source-recognized literal
            values: ``fastprogress``, ``none``, ``tqdm``. These select the corresponding branches
            below; comparison evidence alone does not establish an exhaustive enum.

        style : str
            Progress display style: 'split+stats' separates the bar and statistics, 'combined'
            combines them and 'combined+stats' combines with statistics where supported. See the
            local progress validator for accepted styles. Default: ``'split+stats'``. Passed to
            ``_FastBar`` as ``style``.

        parent : progress/graph object, node identifier or None
            Parent progress object or graph node used by the enclosing operation. None omits the
            parent relationship; graph methods require an existing node and progress methods a
            compatible bar. Default: ``None``. Passed to ``_FastBar`` as ``parent``.
        """
        if backend not in {'auto','fastprogress','tqdm','none'}:raise ValueError('invalid progress_backend')
        if style not in {'split','combined','split+stats','combined+stats'}:raise ValueError('invalid progress_style')
        capture = _event_sink.get() is not None
        self.enabled=(bool(enabled) and backend!='none') or capture
        if backend=='auto' and self.enabled:
            try:import fastprogress;backend='fastprogress'
            except ImportError:backend='tqdm'
        if backend=='auto':backend='none'
        self.backend=backend;self.style=style
        self.mininterval=float(mininterval)
        self._last_memory=-float('inf'); self._memory='n/a'
        self._last_refresh=-float('inf')
        self._process=None
        self.timings={}; self._phase=None; self._phase_start=time.perf_counter(); self._closed=False
        self.completed=0; self.phase_iterations={}; self.events=[]; self.children=[]
        self.operation=str(desc);self.total=total
        options={} if position is None else {'position':position}
        self.bar=None
        if self.enabled and not capture:
            if backend=='fastprogress':
                self.bar=_FastBar(total,desc,mininterval,parent=parent.bar if parent is not None and isinstance(parent.bar,_FastBar) else None,style=style)
            else:self.bar=tqdm(total=total,desc=desc,unit=unit,mininterval=mininterval,dynamic_ncols=True,leave=True,**options)

    def stage(self, text, memory=True, *, phase_key=None):
        """Record a new progress stage with elapsed-time/memory diagnostics.

        Parameters
        ----------
        text : str
            Displayed title, status or descriptive text attached to the visualization/progress
            output.

        memory : bool
            Whether to force a display refresh or include memory statistics at this progress stage.
            False leaves the interval/default rendering policy in effect. Default: ``True``.
        """
        tick=time.perf_counter()
        if self._phase is not None:
            self.timings[self._phase]=self.timings.get(self._phase,0.)+tick-self._phase_start
        self._phase=str(text);self._phase_start=tick
        _emit_progress({'phase': str(text),'phase_key':phase_key,'operation':self.operation})
        if self.bar is None:return
        now=time.monotonic()
        if memory and now-self._last_memory>=1:
            try:
                if self._process is None:
                    import psutil
                    self._process=psutil.Process()
                self._memory=f'{self._process.memory_info().rss / 1024**3:.2f}GiB'
            except Exception:
                self._memory='n/a'
            self._last_memory=now
        self.bar.set_description(f'{text} | MEM:{self._memory}' if memory else str(text),refresh=False)
        if now-self._last_refresh>=self.mininterval:
            self.bar.refresh();self._last_refresh=now

    def update(self,n=1):
        """Advance the progress counter by the requested increment.

        Parameters
        ----------
        n : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit. Default: ``1``.
        """
        self.completed+=int(n)
        _emit_progress({'phase': self._phase, 'completed': self.completed,
                        'total':self.total,'operation':self.operation})
        if self.bar is not None:self.bar.update(n)

    def accepted(self, objective=None, gradient=None, *, limit=None, status='running',phase=None):
        """Count an accepted optimizer step, never a line-search evaluation.

        The bar has an unknown total. The iteration budget is descriptive and
        is not represented as a percentage of a guessed stopping point.
        Diagnostics use already computed objective/gradient values.

        Parameters
        ----------
        objective : object (see the cited input contract)
            Input for the operation ``float(objective)``. See
            ``vigor.progress.StageProgress.accepted`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        gradient : array-like or numeric scalar
            Input for the operation ``np.linalg.norm(gradient)``. See
            ``vigor.progress.StageProgress.accepted`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.

        limit : int or None
            Count limit for the corresponding sampled/retained/displayed items. Positive values
            bound work/output size; None uses all eligible items or the local default where
            supported. Default: ``None``.

        status : str
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``'running'``.

        phase : mapping
            Input for the operation ``self.phase_iterations.get(phase, 0) + 1``. See
            ``vigor.progress.StageProgress.accepted`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape. Default:
            ``None``. None delegates to the local fallback shown in this body; it is not an
            additional numeric/category value.
        """
        if not self.enabled:
            self.update();return
        self.completed+=1
        import numpy as np
        norm=None if gradient is None else float(np.linalg.norm(gradient))
        phase='optimizer' if phase is None else str(phase)
        self.phase_iterations[phase]=self.phase_iterations.get(phase,0)+1
        local=self.phase_iterations[phase]
        event={'iteration':self.completed,'maxiter':limit,'objective':objective,
               'gradient_norm':norm,'status':status,'optimizer_phase':phase,'phase_iteration':local,
               'operation':self.operation}
        previous = next((x.get('objective') for x in reversed(self.events)
                         if x.get('objective') is not None and x.get('optimizer_phase') == phase), None)
        if objective is not None and previous is not None:
            event['objective_delta'] = float(objective)-float(previous)
            event['objective_relative_delta'] = event['objective_delta']/max(abs(float(previous)), 1e-30)
        _emit_progress(dict(event, phase=phase))
        self.events.append(event)
        if _event_sink.get() is not None and len(self.events)>2:
            # The supervisor owns the durable trace. Keep only enough local
            # history for deltas/phase summaries, avoiding huge result attrs.
            del self.events[:-2]
        detail=f'{phase} iter={local}/{limit}, accepted_total={self.completed}, status={status}'
        if objective is not None:detail+=f', f={float(objective):.6g}'
        if norm is not None:detail+=f', |g|={norm:.3g}'
        if hasattr(self.bar,'set_postfix_str'):self.bar.set_postfix_str(detail,refresh=False)
        if self.bar is not None:self.bar.update(1)

    def finish(self, status):
        """Record the final status of the monitored computation.

        Parameters
        ----------
        status : object (see the cited input contract)
            Input for the operation ``'status=' + str(status)``. See
            ``vigor.progress.StageProgress.finish`` and its caller for the scientific
            representation; the shown operation constrains the required type/shape.
        """
        if self.enabled:
            self.events.append({'iteration':self.completed,'status':str(status)})
            if self.bar is not None and hasattr(self.bar,'set_postfix_str'):
                import re
                detail=getattr(self.bar,'detail','')
                detail=re.sub(r'status=[^,]+','status='+str(status),detail) if 'status=' in detail else detail+', status='+str(status)
                self.bar.set_postfix_str(detail.lstrip(', '),refresh=False)
            self.stage(str(status),memory=False)

    def add_total(self, n):
        """Increase a total that is discovered while planning streamed work.

        Parameters
        ----------
        n : int
            Number of animals/rows in the aligned sample axis. Supply the actual nonnegative axis
            length, not an arbitrary truncation limit. Source comparisons: ``increment < 0``. See
            the rejection/fallback branches for enforcement.
        """
        increment = int(n)
        if increment < 0:
            raise ValueError("progress total increment must be non-negative")
        current = 0 if self.total is None else int(self.total)
        self.total=current+increment
        if self.bar is None:return
        self.bar.total = self.total
        # tqdm does not need an immediate repaint for correctness.  Respect the
        # same refresh throttle as stage() when plans are produced rapidly.
        now = time.monotonic()
        if now-self._last_refresh >= self.mininterval:
            self.bar.refresh();self._last_refresh=now

    def close(self):
        """Release resources retained by this reader, operator or progress object.
        """
        if self._closed:return
        if self._phase is not None:
            self.timings[self._phase]=self.timings.get(self._phase,0.)+time.perf_counter()-self._phase_start
        self._closed=True
        if self.bar is not None:self.bar.close()

    def __enter__(self):
        """Enter the managed resource context and return its active object.

        Returns
        -------
        object
            Computed result from ``self``; see the summary and the called implementation for its labeled/numerical representation.
        """
        return self

    def __exit__(self,*args):
        """Close managed resources on context exit and apply the exception policy below.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments supplied by the enclosing callback or context protocol;
            their order and types follow that protocol.
        """
        if args and args[0] is not None:self.finish('error: '+args[0].__name__)
        self.close()


class WorkflowProgress:
    """One persistent fastprogress display for a fit or supervised search.

    Counts cover named stages/trials/folds/traits. Optimizer stopping times are
    unknown; iteration limits are diagnostic budgets, never completion ETAs.
    Only active state is retained; history belongs in the durable HPO event log.
    """
    def __init__(self, *, enabled=True, stages=None, trials=None, backend='auto', style='split+stats'):
        self.stages=list(stages or []);self.trials=trials;self.state={};self.stage_index=0
        self.in_cv=False;self.closed=False;self.last=-float('inf');self.start=time.monotonic()
        self.progress=StageProgress(total=trials if trials is not None else len(self.stages),
            desc='VIGOR search attempts' if trials is not None else 'VIGOR covariance stages',
            enabled=enabled,backend=backend,style=style)
        self.native=self.progress.bar
        if isinstance(self.native,_FastBar):self.native.status_rows=[]

    @staticmethod
    def _number(value):
        try:return f'{float(value):.5g}' if value is not None else '—'
        except (ValueError,TypeError):return str(value)

    def observe(self,event):
        kind=event.get('kind');phase=event.get('phase');force=False
        new_operation=event.get('operation')
        if (new_operation and new_operation!=self.state.get('operation')) or (event.get('phase_key') and event['phase_key']!=self.state.get('phase_key')):
            for key in ('iteration','phase_iteration','objective','objective_delta','gradient_norm','optimizer_phase','completed','total'):
                self.state.pop(key,None)
        if kind in {'fold_start','candidate_start','workflow_stage','calibration'} or phase=='candidate_start':
            for key in ('iteration','phase_iteration','objective','objective_delta','gradient_norm',
                        'optimizer_phase','trait_index','trait_total','trait_completed','calibration_step',
                        'trait','n_observed','operation','completed','total'):
                self.state.pop(key,None)
            force=kind!='calibration'
        if kind in {'candidate_start','workflow_stage'}:
            for key in ('fold_index','fold_total','fold_completed','scenario','design','project'):
                self.state.pop(key,None)
        if kind=='candidate_start':self.state.pop('score',None)
        if kind=='workflow_stage':self.state.pop('trial',None)
        if kind=='rank_cv':self.in_cv=not event.get('finished',False)
        self.state.update({k:v for k,v in event.items() if v is not None})
        key=event.get('phase_key')
        keys=[k for k,_ in self.stages]
        if self.trials is None and key in keys and (not self.in_cv or key=='rank_cv'):
            index=keys.index(key)
            if index!=self.stage_index:force=True
            self.stage_index=index
            if self.native is not None:
                current=getattr(self.native,'n',0)
                if index>current:self.native.update(index-current)
        if phase in {'trial_complete','trial_failed','final_refit','workflow_complete'}:force=True
        self.refresh(force=force)

    def refresh(self,force=False):
        if self.native is None:return
        now=time.monotonic()
        if not force and now-self.last<.5:return
        self.last=now;s=self.state;n=self._number
        rows=[['Scope','Progress','Current work / diagnostics']]
        if self.trials is None:
            label=self.stages[self.stage_index][1] if self.stages else 'Covariance fit'
            rows.append(['Workflow',f'{self.stage_index+1}/{len(self.stages)} stages',label])
            rows.append(['Plan',' → '.join(name for _,name in self.stages),'Stages have unequal cost; no overall ETA'])
        else:
            rows.append(['Workflow',s.get('workflow_label','Preflight'),
                'Preflight → screen/verify trials → selection → final fit → save'])
            rows.append(['Trials',f"{self.progress.completed}/{self.trials} attempts",
                f"completed={s.get('completed_trials',0)} failed={s.get('failed_trials',0)}; search only"])
            if 'trial' in s:
                rows.append(['Current trial',f"{s['trial']+1}/{self.trials}",str(s.get('fidelity','screen'))])
            if 'fold_index' in s:
                rows.append(['CV fold',f"{s['fold_index']}/{s.get('fold_total','?')}",
                    f"{s.get('design','')} | {s.get('project','all projects')} | scenario={s.get('scenario','?')}"])
            if 'parameters' in s:
                p=s['parameters']
                rows.append(['Model','',f"G rank={p.get('rank_G','?')} {p.get('factor_structure_G','')} | E rank={p.get('rank_E','?')} {p.get('factor_structure_E','')} | h2 prior={p.get('h2_prior_strength',0)} ({p.get('h2_prior_groups','global')})"])
            rows.append(['Loss','NMSE',f"current={n(s.get('score'))} | best screen={n(s.get('best_score'))}"])
        if self.in_cv:
            rows.append(['Rank CV',f"{s.get('candidate_index','?')}/{s.get('candidate_total','?')}",f"rank={s.get('rank','?')}"])
        if 'trait_total' in s:
            trait_label=f"trait {s['trait_index']}: {s.get('trait','')}" if 'trait_index' in s else 'all traits'
            rows.append(['h² prior geometry',f"{s.get('trait_completed',0)}/{s['trait_total']} traits complete",
                f"{trait_label} | {s.get('calibration_step','')}"+(f" | n={s['n_observed']}" if 'n_observed' in s else '')])
        if 'phase_iteration' in s:
            rows.append(['Optimizer',f"{s['phase_iteration']} iterations (limit {s.get('maxiter','?')})",
                f"{s.get('optimizer_phase','')} | objective={n(s.get('objective'))} | Δ={n(s.get('objective_delta'))} | gradient={n(s.get('gradient_norm'))}"])
        elif 'completed' in s and 'trait_total' not in s:
            rows.append([s.get('operation','Inner operation'),f"{s['completed']}/{s.get('total','?')}",str(s.get('phase',''))])
        status=s.get('active_phase') if s.get('phase')=='heartbeat' else s.get('phase')
        rows.append(['Status',f'{now-self.start:.1f}s elapsed',str(s.get('status_label') or status or 'starting')])
        self.native.set_description('VIGOR search attempts' if self.trials is not None else f'VIGOR stage {self.stage_index+1}/{len(self.stages)}: {label}',refresh=False)
        detail=' | '.join(str(x) for x in rows[-1][1:])
        if 'trait_total' in s:detail+=f" | h² prior {s.get('trait_completed',0)}/{s['trait_total']}: {s.get('calibration_step','')}"
        elif 'fold_index' in s:detail+=f" | trial {s.get('trial',0)+1}/{self.trials}, fold {s['fold_index']}/{s.get('fold_total','?')} {s.get('design','')}"
        self.native.set_postfix_str(detail,refresh=False)
        if isinstance(self.native,_FastBar):
            self.native.status_rows=rows;self.native.refresh(force=True)
        else:self.native.refresh()

    def close(self,status='completed'):
        if self.closed:return
        self.closed=True;self.state['status_label']=status
        if self.trials is None and status=='completed' and self.native is not None:
            self.stage_index=max(0,len(self.stages)-1)
            remaining=len(self.stages)-getattr(self.native,'n',0)
            if remaining>0:self.native.update(remaining)
        self.refresh(force=True);self.progress.close()


def covariance_workflow(function):
    """Capture nested fit/SE/h2-prior progress into one owning display; no arithmetic changes."""
    from functools import wraps
    import inspect
    signature=inspect.signature(function)
    @wraps(function)
    def wrapped(*args,**kwargs):
        if _event_sink.get() is not None or not kwargs.get('progress',True):
            return function(*args,**kwargs)
        bound=signature.bind_partial(*args,**kwargs);bound.apply_defaults();params=bound.arguments
        stages=[]
        if kwargs.get('rank_selection')=='prediction_cv':stages.append(('rank_cv','Rank CV'))
        stages.extend([('align','Alignment'),('moments','Moments'),('fit','Covariance fit')])
        if params.get('analytic_se') or params.get('n_boot',0):stages.append(('uncertainty','Raw uncertainty'))
        stages.append(('summaries','Summaries'))
        if float(kwargs.get('h2_prior_strength',0) or 0)>0:stages.insert(stages.index(('fit','Covariance fit')),('calibration','h² prior pilot'))
        display=WorkflowProgress(stages=stages,backend=kwargs.get('progress_backend','auto'),style=kwargs.get('progress_style','split+stats'))
        try:
            with progress_events(display.observe):result=function(*args,**kwargs)
        except BaseException as exc:
            display.close('failed: '+type(exc).__name__);raise
        else:
            display.close();return result
    return wrapped


def covariance_runtime(function):
    """Bound BLAS for the entire covariance fit and expose stage timings.

    Parameters
    ----------
    function : callable or None
        Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
        order shown by its call below and return the representation consumed there; None
        disables an optional callback or uses the default implementation.

    Returns
    -------
    object
        Computed result from ``wrapped``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        """Execute the wrapped callable with the enclosing compatibility/progress policy.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments forwarded, in order, to ``function``. See that signature
            for required scientific inputs and their order.

        **kwargs : mapping
            Additional keyword arguments forwarded to ``function``. See those callables for accepted
            keys and values. Passed to ``StageProgress`` as ``enabled``.

        Returns
        -------
        object
            Computed result from ``result``; see the summary and the called implementation for its labeled/numerical representation.
        """
        from contextlib import nullcontext
        from threadpoolctl import threadpool_limits
        requested=kwargs.get('blas_threads','auto')
        # Match the established mvBLUP automatic BLAS budget. Explicit user
        # choices apply to moments, rank selection and fitting, not just one phase.
        threads=4 if requested=='auto' else requested
        if threads not in {None,0,'none'} and (not isinstance(threads,int) or threads<1):
            raise ValueError('blas_threads must be a positive integer, auto, or None')
        context=threadpool_limits(limits=threads,user_api='blas') if threads not in {None,0,'none'} else nullcontext()
        start=time.perf_counter()
        backend=kwargs.pop('progress_backend','auto');style=kwargs.pop('progress_style','split+stats')
        with StageProgress(desc='Covariance',enabled=kwargs.get('progress',True),unit='stage',backend=backend,style=style) as bar:
            kwargs['_runtime_progress']=bar
            token=_active.set(bar)
            try:
                with context:result=function(*args,**kwargs)
            finally:
                _active.reset(token)
        # Covariance engines historically return either a mapping or a pandas
        # Series used as a heterogeneous result container.
        if isinstance(result,dict) or result.__class__.__module__.split('.', 1)[0] == 'pandas':
            result['runtime']={'total_seconds':time.perf_counter()-start,'blas_threads':threads,
                               'stage_seconds':dict(getattr(bar,'timings',{})),
                               'work':bar.children,'progress_backend':bar.backend,'progress_style':bar.style}
        return result
    return wrapped


@contextmanager
def work_progress(desc, *, total=None, unit='step'):
    """A nested work bar; always closed on errors and silent outside a fit.

    Parameters
    ----------
    desc : str
        Displayed title, status or descriptive text attached to the visualization/progress
        output. Passed to ``StageProgress`` as ``desc``.

    total : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``None``. Passed to ``StageProgress`` as ``total``.

    unit : str
        Display unit for one progress increment, such as step or iter; it labels counts rather
        than changing numerical computation. Default: ``'step'``. Passed to ``StageProgress`` as
        ``unit``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    parent=_active.get()
    enabled=parent is not None and parent.enabled
    with StageProgress(total=total,desc=desc,enabled=enabled,unit=unit,position=1,parent=parent,backend=parent.backend if parent else 'none',style=parent.style if parent else 'split+stats') as child:
        token=_work.set(child)
        try:
            yield child
        except BaseException:
            raise
        else:
            if child.enabled and (not child.events or child.events[-1].get('status')=='running'):
                child.finish('completed phase; convergence not certified' if child.events else 'completed work')
        finally:
            _work.reset(token)
            if parent is not None and parent.enabled:
                parent.children.append({'phase':desc,'completed':child.completed,'total':child.bar.total if child.bar is not None else total,
                                        'events':child.events})


def report_work(desc, total=None, unit='step'):
    """Decorate a numerical phase without changing its arguments or arithmetic.

    Parameters
    ----------
    desc : str
        Displayed title, status or descriptive text attached to the visualization/progress
        output.

    total : int or None
        Count limit for the corresponding sampled/retained/displayed items. Positive values
        bound work/output size; None uses all eligible items or the local default where
        supported. Default: ``None``.

    unit : str
        Display unit for one progress increment, such as step or iter; it labels counts rather
        than changing numerical computation. Default: ``'step'``.

    Returns
    -------
    object
        Computed result from ``decorate``; see the summary and the called implementation for its labeled/numerical representation.
    """
    from functools import wraps
    def decorate(function):
        """Attach the enclosing progress/compatibility behavior to a callable.

        Parameters
        ----------
        function : callable or None
            Callback invoked at the corresponding fitting/prediction/scan stage. Accept the argument
            order shown by its call below and return the representation consumed there; None
            disables an optional callback or uses the default implementation.

        Returns
        -------
        object
            Computed result from ``wrapped``; see the summary and the called implementation for its labeled/numerical representation.
        """
        @wraps(function)
        def wrapped(*args,**kwargs):
            """Execute the wrapped callable with the enclosing compatibility/progress policy.

            Parameters
            ----------
            *args : tuple
                Additional positional arguments forwarded, in order, to ``total``, ``function``. See
                that signature for required scientific inputs and their order.

            **kwargs : mapping
                Additional keyword arguments forwarded to ``total``, ``function``. See those callables
                for accepted keys and values. Passed to ``total`` as ``**expanded mapping``.

            Returns
            -------
            object
                Computed result from ``function(*args, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
            """
            count=total(*args,**kwargs) if callable(total) else total
            with work_progress(desc,total=count,unit=unit):
                return function(*args,**kwargs)
        return wrapped
    return decorate


def current_work():
    """Return the current work result computed by ``_work.get``.

    Returns
    -------
    object
        Computed result from ``_work.get()``; see the summary and the called implementation for its labeled/numerical representation.
    """
    return _work.get()


def operation_progress(desc):
    """Wrap an operation in one inherited or explicitly requested progress scope.

    Parameters
    ----------
    desc : str
        Operation title. Wrapped callables accept progress=None to inherit,
        False to disable, and True to enable reporting.

    Returns
    -------
    callable
        Decorator preserving the callable signature and numerical result.
        Context variables and bars are released even when the operation fails.
    """
    from functools import wraps
    import inspect

    def decorate(function):
        """Decorate one numerical operation with a managed reporting scope.

        Parameters
        ----------
        function : callable
            Numerical operation whose progress keyword controls reporting.

        Returns
        -------
        callable
            Wrapped operation with the original signature and result.
        """
        default_progress = inspect.signature(function).parameters['progress'].default

        @wraps(function)
        def wrapped(*args, **kwargs):
            """Call the operation under its explicit or inherited reporting policy.

            Parameters
            ----------
            *args : tuple
                Positional inputs passed unchanged to the wrapped function.
            **kwargs : mapping
                Keyword inputs; progress=None inherits the enclosing scope.

            Returns
            -------
            object
                Unchanged numerical result from the wrapped function.
            """
            requested = kwargs.get('progress', default_progress)
            parent = _active.get()
            enabled = parent.enabled if requested is None and parent else bool(requested)
            if not enabled:
                # Explicit False must silence even an enabled enclosing scope.
                active_token = _active.set(None)
                work_token = _work.set(None)
                try:
                    return function(*args, **kwargs)
                finally:
                    _work.reset(work_token); _active.reset(active_token)
            if current_work() is not None:
                return function(*args, **kwargs)
            if parent is not None and parent.enabled:
                with work_progress(desc) as bar:
                    return function(*args, **kwargs)
            with StageProgress(desc=desc, enabled=True) as bar:
                active_token = _active.set(bar); work_token = _work.set(bar)
                try:
                    return function(*args, **kwargs)
                finally:
                    _work.reset(work_token); _active.reset(active_token)
        return wrapped
    return decorate


def operation_stage(text):
    """Report a genuine phase to the active operation without creating a bar.

    Parameters
    ----------
    text : str
        Name of the phase beginning now.
    """
    bar = _work.get() or _active.get()
    if bar is not None and bar.enabled and bar._phase != str(text):
        bar.stage(text)
