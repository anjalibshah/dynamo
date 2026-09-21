# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER admission gate: native port of the brief's Appendix C sketch.

Guarantee one protected request per task; admit opportunistic siblings only
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


class TaperGate:
    def __init__(self, load: LoadSnapshotProvider, config: TaperConfig) -> None:
        self._load = load
        self._cfg = config
        self._table = TaskTable()
        self._lock = asyncio.Lock()
        self._deferred: list[tuple[str, str, asyncio.Event]] = []  # (task_id, request_id, event)
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_tasks_created = 0
        self._stat_protected_admitted = 0
        self._stat_opportunistic_admitted = 0
        self._stat_deferred = 0
        self._stat_released = 0
        self._stat_forced_admits = 0
        self._stat_shadow_would_defer = 0

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "TaperGate started (load_threshold=%.1f, reconcile=%ss, shadow_mode=%s)",
            self._cfg.load_threshold,
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

    async def before_request(self, task_id: str, request_id: str) -> GateDecision:
        wait_started = time.monotonic()
        async with self._lock:
            was_new = task_id not in self._table.tasks
            task = self._table.get_or_create(task_id)
            if was_new:
                self._stat_tasks_created += 1
            if not task.protected_admitted:
                task.protected_admitted = True
                task.admitted_total += 1
                self._stat_protected_admitted += 1
                logger.debug("taper.admit path=protected task=%s", task_id)
                return GateDecision(task_id=task_id, request_id=request_id,
                                     admitted=True, protected=True)

            load = self._current_load()
            would_defer = load > self._cfg.load_threshold
            if not would_defer or self._cfg.shadow_mode:
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

            # Live gating, load over threshold: defer.
            event = asyncio.Event()
            self._deferred.append((task_id, request_id, event))
            task.deferred_request_ids.append(request_id)
            task.deferred_total += 1
            self._stat_deferred += 1
            logger.info(
                "taper.defer task=%s request=%s load=%.1f threshold=%.1f deferred_total=%d",
                task_id, request_id, load, self._cfg.load_threshold, len(self._deferred),
            )

        forced = False
        try:
            await asyncio.wait_for(event.wait(), timeout=self._cfg.defer_timeout_seconds)
        except asyncio.TimeoutError:
            forced = True
            async with self._lock:
                self._remove_deferred_locked(task_id, request_id)
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

    async def after_request(self, task_id: str) -> None:
        """Release the opportunistic-sibling slot this request held.

        Does not release the *task's* protected slot -- ``protected_admitted``
        stays set for the task's whole lifetime so later branches of the same
        task are always treated as opportunistic, matching the brief's "one
        guaranteed branch per task" (not "one guaranteed branch per burst").
        """
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
            while self._deferred and self._current_load() <= self._cfg.load_threshold:
                task_id, request_id, event = self._deferred.pop(0)
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
                "shadow_mode": self._cfg.shadow_mode,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "protected_admitted": t.protected_admitted,
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
