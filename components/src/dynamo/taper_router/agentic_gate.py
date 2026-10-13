# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER, milestone 1: request-start admission with per-task protection.

Implements the first milestone of the TAPER authors' "TAPER for Agentic
Workflows" design (Gandhi, Kozyrakis) on top of the router add-on:

* **One protected request per runnable task.** A task (root session plus its
  subagent sessions) always has at most one protected in-flight request. An
  arriving request takes the slot if it is free; otherwise it is opportunistic.
  When the protected request finishes, the slot passes to the task's oldest
  deferred request (trunk requests first, as the best available estimate of the
  critical path), so a task never sits with runnable work and no protected
  progress -- the starvation the trunk-only ``legacy`` gate produced.
* **Slack-budgeted admission of opportunistic requests**, the paper's rule
  (arXiv 2605.06914, Algorithm 1) evaluated at request start: admit iff
  ``T(n+1, L+prompt) <= T_obs + rho * max(0, tau_min - T_obs)``, where T is
  PACE's linear step-time model, ``T_obs`` the larger of the predicted and the
  recently observed step time, and ``tau_min`` the per-token target of the most
  urgent protected class present (interactive if any interactive request is in
  flight, else agent).
* **Short-lived permits.** An admission counts toward projected (n, L) for
  ``permit_window_seconds`` until it shows up in forward-pass metrics. Deferred
  requests are re-checked every reconcile tick and force-admitted after
  ``max_hold_seconds``. Started requests are never paused or cancelled.

Modes (``TaperModeConfig.mode``): ``eager`` (admit everything; passthrough with
telemetry), ``cap`` (at most ``cap_k`` in-flight requests per task: the fixed-cap
baselines; ``cap_k=1`` serializes a task), ``taper`` (the policy above). The
earlier gate stays available as ``legacy`` (``gate.py``).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from dynamo.taper_router.gate import GateDecision
from dynamo.taper_router.pace_predictor import PaceLatencyPredictor

logger = logging.getLogger(__name__)

MODES = ("eager", "cap", "taper")


@dataclass
class TaperModeConfig:
    mode: str = "taper"
    cap_k: int = 1
    rho: float = 0.8
    tpot_interactive_ms: float = 18.48
    tpot_agent_ms: float = 21.63
    # T(n, L) = a + b*n + c*L in ms (L in tokens), fitted offline.
    pred_a_ms: float = 0.0
    pred_b_ms: float = 0.0
    pred_c_ms: float = 0.0
    online_refit: bool = True
    refit_interval_seconds: float = 600.0
    permit_window_seconds: float = 0.5
    max_hold_seconds: float = 60.0
    reconcile_interval_seconds: float = 0.05
    max_release_per_tick: int = 4
    observed_window_seconds: float = 1.0
    shadow_mode: bool = False


@dataclass
class _Pending:
    task_id: str
    key: str
    is_trunk: bool
    is_interactive: bool
    prompt_tokens: int
    since: float
    event: asyncio.Event = field(default_factory=asyncio.Event)
    protected: bool = False  # set when released as the task's protected request


@dataclass
class _Task:
    protected: Optional[str] = None
    inflight: Set[str] = field(default_factory=set)
    deferred: Deque[_Pending] = field(default_factory=deque)
    n_protected: int = 0
    n_opportunistic: int = 0
    n_deferred: int = 0
    n_forced: int = 0
    n_handoffs: int = 0
    hold_seconds: float = 0.0


class AgenticTaperGate:
    def __init__(self, load, config: TaperModeConfig) -> None:
        if config.mode not in MODES:
            raise ValueError(f"unknown mode {config.mode!r}; expected one of {MODES}")
        self._load = load
        self._cfg = config
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, _Task] = {}
        self._pending: List[_Pending] = []  # all deferred requests, arrival order
        self._interactive_inflight: Set[str] = set()
        self._recent_admits: Deque[tuple[float, int]] = deque()
        self._observed: Deque[tuple[float, float]] = deque()  # (time, step ms)
        self._last_step_key: Dict[int, tuple] = {}
        # PACE refits on its last 200 steps; one of our MiniMax copies runs
        # ~50-70 steps/s, so 200 spans a few seconds of a narrow (n, L) range and
        # extrapolates poorly. 4000 is about a minute of steps.
        self.predictor = PaceLatencyPredictor(
            config.pred_a_ms, config.pred_b_ms, config.pred_c_ms,
            window=4000, refit_interval=config.refit_interval_seconds,
        )
        self._loop_task: Optional[asyncio.Task] = None
        self.stats = dict(protected=0, opportunistic=0, deferred=0, released=0,
                          promoted=0, forced=0, shadow_would_defer=0)
        self._log_interval_s = 30.0
        self._last_log = time.monotonic()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._loop())
            c = self._cfg
            logger.info(
                "AgenticTaperGate started (mode=%s cap_k=%d rho=%.2f tau_interactive=%.2fms "
                "tau_agent=%.2fms T=%.3f+%.4f*n+%.3e*L ms permit=%.2fs max_hold=%.0fs shadow=%s)",
                c.mode, c.cap_k, c.rho, c.tpot_interactive_ms, c.tpot_agent_ms,
                c.pred_a_ms, c.pred_b_ms, c.pred_c_ms, c.permit_window_seconds,
                c.max_hold_seconds, c.shadow_mode)

    async def stop(self) -> None:
        for tid, t in self._tasks.items():
            logger.info(
                "taper.task_summary task=%s protected=%d opportunistic=%d deferred=%d forced=%d "
                "handoffs=%d hold_s=%.2f", tid, t.n_protected, t.n_opportunistic, t.n_deferred,
                t.n_forced, t.n_handoffs, t.hold_seconds)
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    # ------------------------------------------------------------------ load model
    def _task(self, task_id: str) -> _Task:
        t = self._tasks.get(task_id)
        if t is None:
            t = self._tasks[task_id] = _Task()
            logger.info("taper.task created task=%s", task_id)
        return t

    def _projected(self) -> tuple[Optional[float], Optional[float]]:
        """(n, L) of the busiest worker plus admissions inside the permit window."""
        samples = self._load.step_samples() if hasattr(self._load, "step_samples") else {}
        if not samples:
            return None, None
        _, n, ctx, _, _ = max(samples.values(), key=lambda s: self.predictor.predict(s[1], s[2]))
        cutoff = time.monotonic() - self._cfg.permit_window_seconds
        while self._recent_admits and self._recent_admits[0][0] < cutoff:
            self._recent_admits.popleft()
        return (float(n + len(self._recent_admits)),
                float(ctx + sum(p for _, p in self._recent_admits)))

    def _observed_ms(self) -> Optional[float]:
        cutoff = time.monotonic() - self._cfg.observed_window_seconds
        while self._observed and self._observed[0][0] < cutoff:
            self._observed.popleft()
        return statistics.median(v for _, v in self._observed) if self._observed else None

    def _tau_ms(self) -> float:
        return (self._cfg.tpot_interactive_ms if self._interactive_inflight
                else self._cfg.tpot_agent_ms)

    def _fits(self, prompt_tokens: int) -> bool:
        """The paper's slack rule at request start. Fails open without readings."""
        n, l = self._projected()
        if n is None or not self.predictor.is_calibrated:
            return True
        t_now = self.predictor.predict(n, l)
        obs = self._observed_ms()
        t_obs = max(t_now, obs) if obs is not None else t_now
        # The predicted *increment* is charged against the slack left above the
        # current step time, so a predictor that underestimates the absolute step
        # time (observed > predicted) cannot admit work into a step with no slack.
        delta = self.predictor.predict(n + 1, l + prompt_tokens) - t_now
        return delta <= self._cfg.rho * max(0.0, self._tau_ms() - t_obs)

    # ------------------------------------------------------------------ admission
    def _admit_locked(self, t: _Task, key: str, *, protected: bool, is_interactive: bool,
                      prompt_tokens: int) -> None:
        t.inflight.add(key)
        if protected:
            t.protected = key
            t.n_protected += 1
            self.stats["protected"] += 1
        else:
            t.n_opportunistic += 1
            self.stats["opportunistic"] += 1
            self._recent_admits.append((time.monotonic(), prompt_tokens))
        if is_interactive:
            self._interactive_inflight.add(key)

    async def before_request(self, task_id: str, request_id: str, *, is_trunk: bool,
                             prompt_tokens: int = 0, is_interactive: bool = False) -> GateDecision:
        """Admit or hold one request. ``request_id`` must be unique per request."""
        start = time.monotonic()
        async with self._lock:
            t = self._task(task_id)
            mode = self._cfg.mode
            slot_free = t.protected is None and not t.deferred
            if mode == "eager":
                admit, protected = True, slot_free
            elif mode == "cap":
                admit = len(t.inflight) < self._cfg.cap_k and not t.deferred
                protected = slot_free
            else:  # taper
                if slot_free:
                    admit, protected = True, True
                else:
                    protected = False
                    admit = not self._pending and self._fits(prompt_tokens)
            if self._cfg.shadow_mode and not admit:
                self.stats["shadow_would_defer"] += 1
                admit = True
            if admit:
                self._admit_locked(t, request_id, protected=protected,
                                   is_interactive=is_interactive, prompt_tokens=prompt_tokens)
                return GateDecision(task_id=task_id, request_id=request_id, protected=protected)
            p = _Pending(task_id, request_id, is_trunk, is_interactive, prompt_tokens, start)
            t.deferred.append(p)
            t.n_deferred += 1
            self._pending.append(p)
            self.stats["deferred"] += 1
            logger.info("taper.defer task=%s request=%s mode=%s prompt=%d pending=%d",
                        task_id, request_id, mode, prompt_tokens, len(self._pending))
            if mode == "taper":
                self._promote_locked(task_id)
        try:
            await asyncio.wait_for(p.event.wait(), timeout=self._cfg.max_hold_seconds)
        except asyncio.TimeoutError:
            async with self._lock:
                if not p.event.is_set():  # not released concurrently
                    self._drop_pending_locked(p)
                    self._admit_locked(t, request_id, protected=False,
                                       is_interactive=is_interactive, prompt_tokens=prompt_tokens)
                    t.n_forced += 1
                    self.stats["forced"] += 1
                    logger.info("taper.forced_admit task=%s request=%s after %.1fs",
                                task_id, request_id, self._cfg.max_hold_seconds)
        except asyncio.CancelledError:
            # Client went away while held. Never leave it queued (it would later
            # take a protected slot nobody frees); if it was released meanwhile,
            # undo the admission as after_request would.
            async with self._lock:
                if p.event.is_set():
                    self._finish_locked(task_id, request_id)
                else:
                    self._drop_pending_locked(p)
            raise
        waited = time.monotonic() - start
        t.hold_seconds += waited
        return GateDecision(task_id=task_id, request_id=request_id, protected=p.protected,
                            was_deferred=True, waited_seconds=waited)

    def _drop_pending_locked(self, p: _Pending) -> None:
        if p in self._pending:
            self._pending.remove(p)
        t = self._tasks.get(p.task_id)
        if t is not None and p in t.deferred:
            t.deferred.remove(p)

    def _release_locked(self, p: _Pending, *, protected: bool) -> None:
        self._drop_pending_locked(p)
        p.protected = protected
        self._admit_locked(self._tasks[p.task_id], p.key, protected=protected,
                           is_interactive=p.is_interactive, prompt_tokens=p.prompt_tokens)
        p.event.set()

    def _promote_locked(self, task_id: str) -> None:
        """Give a free protected slot to the task's next deferred request."""
        t = self._tasks.get(task_id)
        if t is None or t.protected is not None or not t.deferred:
            return
        nxt = next((p for p in t.deferred if p.is_trunk), t.deferred[0])
        self._release_locked(nxt, protected=True)
        t.n_handoffs += 1
        self.stats["promoted"] += 1
        logger.info("taper.reconcile released=1 promoted task=%s request=%s", task_id, nxt.key)

    async def after_request(self, task_id: str, *, protected: bool,
                            request_id: Optional[str] = None) -> None:
        if request_id is None:
            return
        async with self._lock:
            self._finish_locked(task_id, request_id)

    def _finish_locked(self, task_id: str, request_id: str) -> None:
        t = self._tasks.get(task_id)
        if t is not None:
            t.inflight.discard(request_id)
        self._interactive_inflight.discard(request_id)
        if t is None:
            return
        if t.protected == request_id:
            t.protected = None
            if self._cfg.mode == "taper":
                self._promote_locked(task_id)
        if self._cfg.mode == "cap":
            while t.deferred and len(t.inflight) < self._cfg.cap_k:
                self._release_locked(t.deferred[0], protected=t.protected is None)
                self.stats["released"] += 1

    # ------------------------------------------------------------------ background
    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cfg.reconcile_interval_seconds)
                try:
                    self._observe()
                    async with self._lock:
                        self._reconcile_locked()
                    self._maybe_log()
                except Exception:
                    logger.exception("AgenticTaperGate loop error")
        except asyncio.CancelledError:
            return

    def _observe(self) -> None:
        samples = self._load.step_samples() if hasattr(self._load, "step_samples") else {}
        now = time.monotonic()
        for worker, (key, n, ctx, ms, n_prefill) in samples.items():
            if self._last_step_key.get(worker) == key:
                continue
            self._last_step_key[worker] = key
            self._observed.append((now, ms))
            # The model is fit on decode-only steps, as offline (fit_step_model.py).
            if self._cfg.online_refit and n > 0 and n_prefill == 0:
                self.predictor.record(n, ctx, ms)
                self.predictor.maybe_refit(now)

    def _reconcile_locked(self) -> None:
        if self._cfg.mode != "taper" or not self._pending:
            return
        # Invariant: a task with deferred work and a free slot gets a protected request.
        for task_id in {p.task_id for p in self._pending}:
            self._promote_locked(task_id)
        released = 0
        # Greedy by lowest predicted cost (linear utility): smallest prompt first. By
        # monotonicity of T, once the cheapest candidate does not fit, none does.
        for p in sorted(self._pending, key=lambda q: (q.prompt_tokens, q.since)):
            if released >= self._cfg.max_release_per_tick or not self._fits(p.prompt_tokens):
                break
            self._release_locked(p, protected=False)
            released += 1
        if released:
            self.stats["released"] += released
            logger.info("taper.reconcile released=%d still_deferred=%d", released, len(self._pending))

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < self._log_interval_s:
            return
        self._last_log = now
        n, l = self._projected()
        obs = self._observed_ms()
        logger.info(
            "taper.load mode=%s tasks=%d with_protected=%d inflight=%d pending=%d interactive=%d "
            "n=%s ctx_k=%s obs_ms=%s tau_ms=%.2f coeffs=%s mape=%s stats=%s",
            self._cfg.mode, len(self._tasks),
            sum(1 for t in self._tasks.values() if t.protected is not None),
            sum(len(t.inflight) for t in self._tasks.values()), len(self._pending),
            len(self._interactive_inflight), None if n is None else int(n),
            None if l is None else round(l / 1000), None if obs is None else round(obs, 2),
            self._tau_ms(), tuple(round(c, 6) for c in self.predictor.coeffs),
            None if self.predictor.mape() is None else round(self.predictor.mape(), 3), self.stats)

    # ------------------------------------------------------------------ status
    async def status_snapshot(self) -> dict:
        async with self._lock:
            return {
                "mode": self._cfg.mode,
                "tasks_total": len(self._tasks),
                "pending": len(self._pending),
                "interactive_inflight": len(self._interactive_inflight),
                "tasks": [
                    {"task_id": tid, "protected": t.protected is not None,
                     "inflight": len(t.inflight), "deferred": len(t.deferred),
                     "n_protected": t.n_protected, "n_opportunistic": t.n_opportunistic,
                     "n_deferred": t.n_deferred, "n_forced": t.n_forced,
                     "n_handoffs": t.n_handoffs, "hold_seconds": round(t.hold_seconds, 3)}
                    for tid, t in self._tasks.items()
                ],
            }

    async def metrics_snapshot(self) -> dict:
        async with self._lock:
            return {"counters": dict(self.stats),
                    "gauges": {"tasks_total": len(self._tasks), "pending": len(self._pending),
                               "interactive_inflight": len(self._interactive_inflight)}}
