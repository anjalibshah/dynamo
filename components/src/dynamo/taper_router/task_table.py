# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task lifecycle data model for the Agentic TAPER admission gate.

Mirrors the shape of ``thunderagent_router/program_state.py`` (same
``session_id``-keyed table pattern, same dataclass-plus-table split) so the two
schedulers stay easy to compare and, eventually, compose. The state tracked is
different because the gate regulates a different axis: ThunderAgent tracks
real token counts against a KV *retention* budget; TAPER tracks admitted vs.
opportunistic *request counts* per task against a decode-load threshold (brief
Appendix C, "Permit gate sketch").

Vocabulary matches the brief (Appendix B) and its "task", not "family"
decision: **task** is the SLO-bearing entity (one user-visible agent run),
**branch** is a concurrent sub-agent execution path within it. We key on
``session_id`` for the task and treat ``parent_session_id`` as the branch
grouping key, per the P0 identity workaround (brief section "PR plan", P0
identity workaround) -- ``AdmissionRequest`` in the native Rust path doesn't
expose ``parent_session_id`` either, but the Python request dict here does
(``agent_context``), so we read it directly instead of encoding it into the
session id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentTaskState:
    task_id: str

    # The one request per task that always runs (brief: "protected width").
    # Set once the first branch for this task is admitted; stays set for the
    # task's lifetime so later branches of the same task are all opportunistic.
    protected_admitted: bool = False

    # Opportunistic siblings currently in flight (admitted but not yet
    # completed). Decremented on completion/abort, not on defer.
    inflight_opportunistic: int = 0

    # Deferred branch request ids waiting on this task, FIFO within the task.
    # The global deferred queue in TaskTable is release order; this list is
    # only for per-task introspection (status/metrics).
    deferred_request_ids: list[str] = field(default_factory=list)

    # Total requests ever admitted/deferred, for metrics.
    admitted_total: int = 0
    deferred_total: int = 0


@dataclass
class TaskTable:
    tasks: dict[str, AgentTaskState] = field(default_factory=dict)

    def get_or_create(self, task_id: str) -> AgentTaskState:
        task = self.tasks.get(task_id)
        if task is None:
            task = AgentTaskState(task_id=task_id)
            self.tasks[task_id] = task
        return task

    def release(self, task_id: str) -> Optional[AgentTaskState]:
        """Drop a task's bookkeeping entirely (all branches finished/aborted).

        Unlike ThunderAgent's ``end_program`` (one terminal request per
        program), Agentic TAPER has no single terminal signal for "the whole
        task is done" -- the join semantics live in the orchestrator (brief:
        "Orch" row, "DAG execution ... join/reduce"), not in Dynamo. Callers
        that *do* have an explicit terminal signal (e.g. a harness emitting
        the same ``x-dynamo-session-final`` convention ThunderAgent uses) can
        call this directly; otherwise tasks are reaped lazily -- see
        ``gate.py``'s idle-task sweep in ``reconcile()``.
        """
        return self.tasks.pop(task_id, None)
