# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side task-aware permit gate for the Agentic TAPER P0 experiment.

This is the actuator for arms A2 and A3 (brief §2, Appendix C). Dynamo's router
holds requests internally but its release condition is host-owned and not
steerable by a live signal, and the worker-selection extension contract
excludes queueing by design (``scheduling/AGENTS.md:100``, "no queue mutation").
So the gate lives in the replay client: it decides *when* each request is
released to the Dynamo frontend.

One gate, two policies, so an A3 win over A2 is attributable to adaptivity and
not to a different code path:

* ``STATIC_CAP`` (A2): admit up to ``k`` concurrent requests per task.
* ``FPM`` (A3): admit opportunistic siblings only while the live load signal
  permits. Two forms of "permits":
    - **budget rule** (faithful to TAPER's ``T(S) <= T0 + rho*B_t``): admit iff
      the *projected* victim ITL after adding this sibling stays within the SLO —
      ``latency_model.itl_at(load + 1) <= slo_ms``. No free threshold; the bar is
      derived from the SLO budget and the calibrated load->latency model.
    - **raw threshold** (legacy / ablation): admit iff ``load <= load_threshold``.
      A hardcoded tau, kept only so we can show the swept value matches the
      budget-derived boundary.

Both policies always admit the one **protected** request per task, so baseline
progress is never blocked (brief §3, H2). Held requests are never rejected; they
drain when slack returns.

This module is transport-agnostic and synchronous at its core so it can be
unit-tested without a GPU, a network, or an event loop. The replay client wires
``send`` to the actual async dispatch and calls ``on_complete`` / ``drain`` from
its completion and timer paths.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


class Policy(enum.Enum):
    EAGER = "eager"             # A1 — admit everything immediately, never hold
    STATIC_CAP = "static_cap"   # A0 (k=1) and A2 (k=K)
    FPM = "fpm"                 # A3


@dataclass
class TaskState:
    protected_admitted: bool = False
    inflight: int = 0                       # currently-running requests for this task
    held: deque = field(default_factory=deque)


# A request is anything with a ``task_id``; the gate only reads that plus an
# opaque payload it hands back to ``send``.
@dataclass
class GateRequest:
    task_id: str
    request_id: str
    payload: object = None
    is_protected: bool = False  # replay client marks the task's baseline request


class PermitGate:
    """Task-aware permit gate. See module docstring.

    Args:
        policy: STATIC_CAP (A2) or FPM (A3).
        send: callback invoked when a request is admitted. Must be cheap /
            non-blocking; the replay client makes it enqueue an async dispatch.
        load_fn: returns the current live load scalar (A3 only). For FPM this is
            ``num_decode_requests`` from the FPM subscriber.
        k: per-task concurrency cap (A2 only).
        load_threshold: admit opportunistic siblings while ``load_fn() <=``
            this (A3 only).
    """

    def __init__(
        self,
        policy: Policy,
        send: Callable[[GateRequest], None],
        *,
        load_fn: Optional[Callable[[], float]] = None,
        k: Optional[int] = None,
        load_threshold: Optional[float] = None,
        latency_model=None,
        slo_ms: Optional[float] = None,
    ) -> None:
        if policy is Policy.STATIC_CAP and (k is None or k < 1):
            raise ValueError("STATIC_CAP requires k >= 1")
        if policy is Policy.FPM:
            if load_fn is None:
                raise ValueError("FPM requires load_fn")
            budget = latency_model is not None and slo_ms is not None
            if not budget and load_threshold is None:
                raise ValueError(
                    "FPM requires either (latency_model + slo_ms) for the budget "
                    "rule, or load_threshold for the legacy threshold")
        # EAGER needs no parameters; it admits everything.
        self.policy = policy
        self._send = send
        self._load_fn = load_fn
        self.k = k
        self.load_threshold = load_threshold
        self._latency_model = latency_model
        self._slo_ms = slo_ms
        self.tasks: dict[str, TaskState] = {}
        # Counters for the "admitted vs. held width" metric (brief §3).
        self.n_admitted_protected = 0
        self.n_admitted_opportunistic = 0
        self.n_held_events = 0

    # -- core decision -------------------------------------------------------

    def _task(self, tid: str) -> TaskState:
        st = self.tasks.get(tid)
        if st is None:
            st = TaskState()
            self.tasks[tid] = st
        return st

    def _has_opportunistic_slack(self, st: TaskState) -> bool:
        if self.policy is Policy.EAGER:
            return True
        if self.policy is Policy.STATIC_CAP:
            # One slot is reserved for the protected request; the remaining
            # k-1 are opportunistic.
            return st.inflight < self.k
        # FPM: gate on live load, independent of per-task count.
        load = self._load_fn()
        if self._latency_model is not None:
            # Budget rule (TAPER T(S) <= T0 + rho*B_t): admitting this sibling
            # takes decode load to load+1; require the projected victim ITL there
            # to stay within the SLO budget. No free threshold.
            return self._latency_model.itl_at(load + 1) <= self._slo_ms
        return load <= self.load_threshold  # legacy raw threshold

    def submit(self, req: GateRequest) -> None:
        """Admit or hold ``req``.

        The task's protected request always runs immediately, even under load,
        but only once per task. The protected slot is filled *only* by a request
        the client explicitly flags ``is_protected`` — the victim's sole request
        or the aggressor's root. This is deliberate: an aggressor's branch may
        arrive before its root, and first-seen would wrongly protect the branch.
        Every non-flagged request is opportunistic, subject to slack.
        """
        st = self._task(req.task_id)
        if req.is_protected and not st.protected_admitted:
            st.protected_admitted = True
            self._admit(st, req, protected=True)
            return
        if self._has_opportunistic_slack(st):
            self._admit(st, req, protected=False)
        else:
            st.held.append(req)
            self.n_held_events += 1

    def _admit(self, st: TaskState, req: GateRequest, *, protected: bool) -> None:
        st.inflight += 1
        if protected:
            self.n_admitted_protected += 1
        else:
            self.n_admitted_opportunistic += 1
        self._send(req)

    # -- release paths -------------------------------------------------------

    def on_complete(self, task_id: str) -> None:
        """Call when an in-flight request for ``task_id`` finishes."""
        st = self._task(task_id)
        st.inflight = max(0, st.inflight - 1)
        self._drain_task(st)

    def drain(self) -> None:
        """Timer-driven drain across all tasks (matches FPM freshness cadence).

        For FPM this is how load *dropping* releases held siblings even when no
        completion fired.
        """
        for st in self.tasks.values():
            self._drain_task(st)

    def _drain_task(self, st: TaskState) -> None:
        while st.held and self._has_opportunistic_slack(st):
            req = st.held.popleft()
            self._admit(st, req, protected=False)

    # -- introspection -------------------------------------------------------

    @property
    def total_held(self) -> int:
        return sum(len(st.held) for st in self.tasks.values())

    def stats(self) -> dict:
        return {
            "admitted_protected": self.n_admitted_protected,
            "admitted_opportunistic": self.n_admitted_opportunistic,
            "held_events": self.n_held_events,
            "still_held": self.total_held,
        }
