# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER admission gate: native port of the brief's Appendix C sketch.

Always admit a task's trunk (its own session: root turns, joins); admit
opportunistic branch siblings only
while live decode load is at or below ``load_threshold``; defer the rest and
release them FIFO as load drops. This is the P0-2/P0-3/P0-4/P0-5 sequence from
the brief collapsed into one Python module, following the same
before_request/after_request/background-loop shape
``thunderagent_router/router.py`` uses -- see that file's
``ThunderAgentScheduler`` for the sibling pattern this was built against.

Deliberate v0 simplification: the gate checks load aggregated as the max
across live workers, not per-worker / per-affinity load (brief P1-5,
"task-aware placement", is explicitly later work). A task's branches
co-locate via KV-overlap routing on their shared prefix (brief: "siblings
co-locate via KV-overlap routing on the shared prefix, not via session
affinity"), so in practice load concentrates on one worker per task -- but a
gate keyed on the cluster max will defer more conservatively than one keyed on
the task's actual worker. Tightening this is exactly the kind of thing the
POC's A3 vs. best-tuned-A2 comparison should reveal is or isn't worth doing
before any of this becomes a real PR.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from dynamo.taper_router.load import LoadSnapshotProvider
from dynamo.taper_router.task_table import AgentTaskState, TaskTable

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    task_id: str
    request_id: str
    admitted: bool = True
    protected: bool = False
    was_deferred: bool = False
    waited_seconds: float = 0.0
    # Set when shadow_mode is on: what the gate *would* have decided had it
    # been live. admitted is always True in shadow mode regardless of this.
    shadow_would_defer: bool = False


@dataclass
class TaperConfig:
    # Decode requests per worker at/under which an opportunistic sibling is
    # admitted immediately. Swept per brief Appendix D ("gate load_threshold,
    # the core tuning curve").
    load_threshold: float = 32.0
    # How often the background loop retries releasing deferred siblings.
    # Matches FPM freshness per brief Appendix C (sweep {20, 50, 100} ms).
    reconcile_interval_seconds: float = 0.05
    # Cap on how long an opportunistic sibling waits deferred before being
    # force-admitted, mirroring ThunderAgent's resume_timeout_seconds so a
    # starved sibling can't wait forever if load never drops.
    defer_timeout_seconds: float = 300.0
    # P0-4: compute the real decision and emit it as a counter, but always
    # actually admit (AdmissionDecision::Ready in the brief's shadow mode).
    # Flip once the shadow counters look sane on real traffic.
    shadow_mode: bool = True
    # FPM lags admission: a request only counts as a decode request once its
    # prefill finishes and the next forward pass is published. Each branch
    # admission is counted as extra load for this long, so one stale low
    # reading can't admit or release a whole burst.
    admit_settle_seconds: float = 0.5
    # Most deferred branches released per reconcile tick. Without a cap, one
    # low reading drained the backlog at once (58, then 30, then 29 on a
    # 1-GPU run), recreating the prefill burst the gate exists to prevent.
    max_release_per_tick: int = 4
    # Context budget (tokens): also defer a branch while the busiest worker's
    # live decode context plus the branch's prompt would exceed this. 0
    # disables it. Step time tracks total decode context more than decode
    # count on long-context agents; the budget is the context at which the
    # fitted step time reaches the interactive ITL SLO (NEXT_STEPS 7.3e).
    context_budget_tokens: float = 0.0


class TaperGate:
    def __init__(self, load: LoadSnapshotProvider, config: TaperConfig) -> None:
        self._load = load
        self._cfg = config
        self._table = TaskTable()
        self._lock = asyncio.Lock()
        # (task_id, request_id, event, prompt_tokens)
        self._deferred: list[tuple[str, str, asyncio.Event, int]] = []
        # (monotonic time, prompt tokens) of recent branch admissions, not yet
        # visible in FPM.
        self._recent_admits: deque[tuple[float, int]] = deque()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_tasks_created = 0
        self._stat_protected_admitted = 0
        self._stat_opportunistic_admitted = 0
        self._stat_deferred = 0
        self._stat_released = 0
        self._stat_forced_admits = 0
        self._stat_shadow_would_defer = 0
        # Periodic INFO summary of what the gate sees (decode count and
        # iteration time), so a gate that never acts can be told apart from
        # one that never receives load readings.
        self._load_log_interval_s = 30.0
        self._load_window: list[float] = []
        self._iter_window: list[float] = []
        self._ctx_window: list[float] = []
        self._no_reading = 0
        self._last_load_log = time.monotonic()

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "TaperGate started (load_threshold=%.1f, context_budget_tokens=%.0f, "
            "reconcile=%ss, shadow_mode=%s)",
            self._cfg.load_threshold,
            self._cfg.context_budget_tokens,
            self._cfg.reconcile_interval_seconds,
            self._cfg.shadow_mode,
        )

    async def stop(self) -> None:
        if self._scheduler_task is None:
            return
        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            pass
        self._scheduler_task = None

    def _current_load(self) -> float:
        snapshot = self._load.snapshot()
        if not snapshot:
            # Cold start / no FPM observed yet: fail open, matching
            # ThunderAgent's cold-start behavior in capacity.py's analogous
            # gap ("let the request flow through with no pin").
            return 0.0
        return float(max(snapshot.values()))

    def _current_context(self) -> float:
        # Optional signal: providers without decode_context() (and cold start)
        # contribute no context, so the budget fails open like the count.
        decode_context = getattr(self._load, "decode_context", None)
        snapshot = decode_context() if decode_context is not None else {}
        return float(max(snapshot.values())) if snapshot else 0.0

    def _prune_recent_locked(self) -> None:
        cutoff = time.monotonic() - self._cfg.admit_settle_seconds
        while self._recent_admits and self._recent_admits[0][0] < cutoff:
            self._recent_admits.popleft()

    def _projected_load_locked(self) -> float:
        # Caller holds self._lock. Conservative: an admission can briefly be
        # counted both here and in FPM once FPM catches up.
        self._prune_recent_locked()
        return self._current_load() + len(self._recent_admits)

    def _projected_context_locked(self) -> float:
        self._prune_recent_locked()
        return self._current_context() + sum(t for _, t in self._recent_admits)

    def _has_room_locked(self, prompt_tokens: int) -> tuple[bool, float, float]:
        """(room for this branch, projected count, projected context)."""
        load = self._projected_load_locked()
        ctx = self._projected_context_locked() if self._cfg.context_budget_tokens > 0 else 0.0
        room = load <= self._cfg.load_threshold
        # ctx > 0: with nothing decoding, admit even a branch whose prompt
        # alone exceeds the budget rather than starving it to the timeout.
        if self._cfg.context_budget_tokens > 0 and ctx > 0:
            room = room and ctx + prompt_tokens <= self._cfg.context_budget_tokens
        return room, load, ctx

    def _record_admit_locked(self, prompt_tokens: int = 0) -> None:
        self._recent_admits.append((time.monotonic(), prompt_tokens))

    async def before_request(self, task_id: str, request_id: str, *,
                             is_trunk: bool, prompt_tokens: int = 0) -> GateDecision:
        """Admit or defer one request.

        ``is_trunk``: the request belongs to the task's own session (a root
        turn or a join), not a child branch session. Trunk requests are
        always admitted; only branch requests are gated.
        ``prompt_tokens``: the request's input length, i.e. the decode context
        it adds once admitted (checked against ``context_budget_tokens``).
        """
        wait_started = time.monotonic()
        async with self._lock:
            was_new = task_id not in self._table.tasks
            task = self._table.get_or_create(task_id)
            if was_new:
                self._stat_tasks_created += 1
                logger.info("taper.task created task=%s", task_id)
            if is_trunk:
                task.protected_total += 1
                task.admitted_total += 1
                self._stat_protected_admitted += 1
                logger.debug("taper.admit path=protected task=%s", task_id)
                return GateDecision(task_id=task_id, request_id=request_id,
                                     admitted=True, protected=True)

            room, load, ctx = self._has_room_locked(prompt_tokens)
            # Queue behind already-deferred branches so release stays FIFO.
            would_defer = bool(self._deferred) or not room
            if not would_defer or self._cfg.shadow_mode:
                self._record_admit_locked(prompt_tokens)
                task.inflight_opportunistic += 1
                task.admitted_total += 1
                if would_defer:
                    self._stat_shadow_would_defer += 1
                else:
                    self._stat_opportunistic_admitted += 1
                logger.debug(
                    "taper.admit path=opportunistic task=%s load=%.1f threshold=%.1f "
                    "shadow=%s would_defer=%s",
                    task_id, load, self._cfg.load_threshold, self._cfg.shadow_mode, would_defer,
                )
                return GateDecision(task_id=task_id, request_id=request_id, admitted=True,
                                     protected=False, shadow_would_defer=would_defer)

            # Live gating, no room: defer.
            event = asyncio.Event()
            self._deferred.append((task_id, request_id, event, prompt_tokens))
            task.deferred_request_ids.append(request_id)
            task.deferred_total += 1
            self._stat_deferred += 1
            logger.info(
                "taper.defer task=%s request=%s load=%.1f threshold=%.1f ctx=%.0f "
                "prompt=%d budget=%.0f deferred_total=%d",
                task_id, request_id, load, self._cfg.load_threshold, ctx, prompt_tokens,
                self._cfg.context_budget_tokens, len(self._deferred),
            )

        forced = False
        try:
            await asyncio.wait_for(event.wait(), timeout=self._cfg.defer_timeout_seconds)
        except asyncio.TimeoutError:
            forced = True
            async with self._lock:
                self._remove_deferred_locked(task_id, request_id)
                self._record_admit_locked(prompt_tokens)
                task = self._table.get_or_create(task_id)
                task.inflight_opportunistic += 1
                task.admitted_total += 1
                self._stat_forced_admits += 1
            logger.warning(
                "taper.forced_admit task=%s request=%s after %.1fs",
                task_id, request_id, self._cfg.defer_timeout_seconds,
            )

        waited = time.monotonic() - wait_started
        return GateDecision(task_id=task_id, request_id=request_id, admitted=True,
                             protected=False, was_deferred=True, waited_seconds=waited)

    def _remove_deferred_locked(self, task_id: str, request_id: str) -> None:
        self._deferred = [
            d for d in self._deferred if not (d[0] == task_id and d[1] == request_id)
        ]
        task = self._table.tasks.get(task_id)
        if task is not None and request_id in task.deferred_request_ids:
            task.deferred_request_ids.remove(request_id)

    async def after_request(self, task_id: str, *, protected: bool) -> None:
        """Release the opportunistic-sibling slot this request held.

        Protected (trunk) requests never took a slot, so completing one must
        not decrement the task's in-flight opportunistic count.
        """
        if protected:
            return
        async with self._lock:
            task = self._table.tasks.get(task_id)
            if task is not None and task.inflight_opportunistic > 0:
                task.inflight_opportunistic -= 1

    def end_task(self, task_id: str) -> bool:
        """Drop a task's bookkeeping once the orchestrator signals it's done.

        See ``task_table.TaskTable.release`` docstring: Dynamo has no native
        "whole task finished" signal, so this is opt-in for harnesses that
        emit one, e.g. reusing ThunderAgent's ``x-dynamo-session-final``
        convention on the root/join request.
        """
        return self._table.release(task_id) is not None

    async def _scheduler_loop(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                await asyncio.sleep(self._cfg.reconcile_interval_seconds)
                try:
                    await self._reconcile()
                    self._sample_load()
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.exception("TaperGate reconcile error")
                    if consecutive_failures >= 10:
                        logger.error(
                            "TaperGate reconcile failed %d times in a row; halting loop",
                            consecutive_failures,
                        )
                        return
        except asyncio.CancelledError:
            return

    def _sample_load(self) -> None:
        snap = self._load.snapshot()
        if snap:
            self._load_window.append(float(max(snap.values())))
            self._ctx_window.append(self._current_context())
        else:
            self._no_reading += 1
        iteration_ms = getattr(self._load, "iteration_ms", None)
        if iteration_ms is not None:
            it = iteration_ms()
            if it:
                self._iter_window.append(max(it.values()))
        now = time.monotonic()
        if now - self._last_load_log < self._load_log_interval_s:
            return
        n = len(self._load_window) + self._no_reading
        if n:
            lw, iw = sorted(self._load_window), sorted(self._iter_window)
            cw = sorted(self._ctx_window)

            def q(v, f):
                return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")

            logger.info(
                "taper.load window=%.0fs samples=%d no_reading=%d decode p50=%.0f "
                "p95=%.0f max=%.0f ctx_k p50=%.0f p95=%.0f max=%.0f iteration_ms "
                "p50=%.1f p95=%.1f threshold=%.0f budget_k=%.0f deferred_now=%d",
                now - self._last_load_log, n, self._no_reading, q(lw, 0.5), q(lw, 0.95),
                lw[-1] if lw else float("nan"), q(cw, 0.5) / 1e3, q(cw, 0.95) / 1e3,
                (cw[-1] if cw else float("nan")) / 1e3, q(iw, 0.5), q(iw, 0.95),
                self._cfg.load_threshold, self._cfg.context_budget_tokens / 1e3,
                len(self._deferred))
        self._load_window, self._iter_window, self._ctx_window = [], [], []
        self._no_reading = 0
        self._last_load_log = now

    async def _reconcile(self) -> None:
        """FIFO release of deferred siblings while there's load slack.

        Mirrors the brief's Appendix C ``release_if_slack``, called both on
        ``Completed``/``Aborted`` events and on the periodic ``Reconcile``
        tick -- here folded into one poll loop since ``after_request``
        already frees the slot and the next tick picks it up within
        ``reconcile_interval_seconds``.
        """
        if not self._deferred:
            return
        released = 0
        async with self._lock:
            # Each release is recorded as a recent admit, so projected load
            # rises as we go; the per-tick cap bounds the burst further.
            while (self._deferred
                   and released < self._cfg.max_release_per_tick
                   and self._has_room_locked(self._deferred[0][3])[0]):
                task_id, request_id, event, prompt_tokens = self._deferred.pop(0)
                self._record_admit_locked(prompt_tokens)
                task = self._table.tasks.get(task_id)
                if task is not None and request_id in task.deferred_request_ids:
                    task.deferred_request_ids.remove(request_id)
                if task is not None:
                    task.inflight_opportunistic += 1
                    task.admitted_total += 1
                event.set()
                released += 1
                self._stat_released += 1
        if released:
            logger.info("taper.reconcile released=%d still_deferred=%d", released, len(self._deferred))

    async def status_snapshot(self) -> dict:
        async with self._lock:
            return {
                "tasks_total": len(self._table.tasks),
                "deferred_total": len(self._deferred),
                "load": self._current_load(),
                "load_threshold": self._cfg.load_threshold,
                "context": self._current_context(),
                "context_budget_tokens": self._cfg.context_budget_tokens,
                "shadow_mode": self._cfg.shadow_mode,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "protected_total": t.protected_total,
                        "inflight_opportunistic": t.inflight_opportunistic,
                        "deferred": len(t.deferred_request_ids),
                        "admitted_total": t.admitted_total,
                        "deferred_total": t.deferred_total,
                    }
                    for t in self._table.tasks.values()
                ],
            }

    async def metrics_snapshot(self) -> dict:
        async with self._lock:
            return {
                "counters": {
                    "tasks_created_total": self._stat_tasks_created,
                    "protected_admitted_total": self._stat_protected_admitted,
                    "opportunistic_admitted_total": self._stat_opportunistic_admitted,
                    "deferred_total": self._stat_deferred,
                    "released_total": self._stat_released,
                    "forced_admits_total": self._stat_forced_admits,
                    "shadow_would_defer_total": self._stat_shadow_would_defer,
                },
                "gauges": {
                    "tasks_total": len(self._table.tasks),
                    "currently_deferred": len(self._deferred),
                    "load": self._current_load(),
                },
            }
