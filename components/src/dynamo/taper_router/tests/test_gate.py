# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TaperGate that don't need a Dynamo runtime.

Mirrors the FakeCapacity-injection pattern in
``thunderagent_router/tests/test_router.py``'s ``FakeCapacity``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.taper_router.gate import TaperConfig, TaperGate

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeLoad:
    """Stand-in for LoadSnapshotProvider with a directly settable snapshot."""

    workers: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[int, int]:
        return dict(self.workers)


def make_gate(
    load_workers: Optional[dict[int, int]] = None,
    config: Optional[TaperConfig] = None,
) -> tuple[TaperGate, FakeLoad]:
    load = FakeLoad(workers=load_workers or {})
    cfg = config or TaperConfig(
        load_threshold=32.0,
        reconcile_interval_seconds=0.02,
        defer_timeout_seconds=2.0,
        shadow_mode=False,
    )
    return TaperGate(load, cfg), load  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_first_branch_of_a_task_is_always_protected():
    gate, _ = make_gate()
    decision = await gate.before_request("t1", "t1-root")
    assert decision.protected is True
    assert decision.admitted is True
    assert decision.was_deferred is False


@pytest.mark.asyncio
async def test_second_branch_is_opportunistic_not_protected():
    gate, _ = make_gate()
    await gate.before_request("t1", "t1-root")
    decision = await gate.before_request("t1", "t1-branch-0")
    assert decision.protected is False


@pytest.mark.asyncio
async def test_opportunistic_branch_admitted_immediately_under_threshold():
    gate, load = make_gate(load_workers={1: 5}, config=TaperConfig(load_threshold=32.0))
    await gate.before_request("t1", "t1-root")
    decision = await gate.before_request("t1", "t1-branch-0")
    assert decision.admitted is True
    assert decision.was_deferred is False


@pytest.mark.asyncio
async def test_opportunistic_branch_deferred_over_threshold_then_released_on_reconcile():
    gate, load = make_gate(
        load_workers={1: 100},
        config=TaperConfig(
            load_threshold=32.0, reconcile_interval_seconds=0.02,
            defer_timeout_seconds=5.0, shadow_mode=False,
        ),
    )
    await gate.before_request("t1", "t1-root")

    task = asyncio.ensure_future(gate.before_request("t1", "t1-branch-0"))
    await asyncio.sleep(0.05)
    assert not task.done()  # still deferred: load is over threshold

    load.workers = {1: 1}  # slack opens up
    await gate._reconcile()

    decision = await asyncio.wait_for(task, timeout=1.0)
    assert decision.admitted is True
    assert decision.was_deferred is True


@pytest.mark.asyncio
async def test_deferred_branch_force_admitted_after_timeout():
    gate, _ = make_gate(
        load_workers={1: 100},
        config=TaperConfig(
            load_threshold=32.0, reconcile_interval_seconds=0.02,
            defer_timeout_seconds=0.05, shadow_mode=False,
        ),
    )
    await gate.before_request("t1", "t1-root")
    decision = await gate.before_request("t1", "t1-branch-0")
    assert decision.admitted is True
    assert decision.was_deferred is True
    assert gate._stat_forced_admits == 1


@pytest.mark.asyncio
async def test_shadow_mode_always_admits_but_records_would_defer():
    gate, _ = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, shadow_mode=True),
    )
    await gate.before_request("t1", "t1-root")
    decision = await gate.before_request("t1", "t1-branch-0")
    assert decision.admitted is True
    assert decision.was_deferred is False
    assert decision.shadow_would_defer is True
    assert gate._stat_shadow_would_defer == 1
    assert gate._stat_deferred == 0


@pytest.mark.asyncio
async def test_after_request_decrements_opportunistic_count_not_protected_slot():
    gate, _ = make_gate(load_workers={1: 5})
    await gate.before_request("t1", "t1-root")
    await gate.before_request("t1", "t1-branch-0")
    task = gate._table.tasks["t1"]
    assert task.inflight_opportunistic == 1

    await gate.after_request("t1")
    assert task.inflight_opportunistic == 0
    # protected_admitted stays set for the task's whole lifetime -- the next
    # branch of the same task is still opportunistic, not re-protected.
    assert task.protected_admitted is True
    decision = await gate.before_request("t1", "t1-branch-1")
    assert decision.protected is False


@pytest.mark.asyncio
async def test_end_task_drops_bookkeeping():
    gate, _ = make_gate()
    await gate.before_request("t1", "t1-root")
    assert gate.end_task("t1") is True
    assert "t1" not in gate._table.tasks
    assert gate.end_task("t1") is False


@pytest.mark.asyncio
async def test_cold_start_with_no_load_signal_fails_open():
    gate, _ = make_gate(load_workers={})  # no FPM observed yet
    await gate.before_request("t1", "t1-root")
    decision = await gate.before_request("t1", "t1-branch-0")
    assert decision.admitted is True
    assert decision.was_deferred is False


@pytest.mark.asyncio
async def test_status_snapshot_reports_tasks_and_load():
    gate, _ = make_gate(load_workers={1: 5, 2: 9})
    await gate.before_request("t1", "t1-root")
    await gate.before_request("t1", "t1-branch-0")

    snapshot = await gate.status_snapshot()
    assert snapshot["tasks_total"] == 1
    assert snapshot["load"] == 9  # max across workers
    assert snapshot["tasks"][0]["task_id"] == "t1"
    assert snapshot["tasks"][0]["protected_admitted"] is True
    assert snapshot["tasks"][0]["inflight_opportunistic"] == 1


@pytest.mark.asyncio
async def test_metrics_snapshot_counters():
    gate, _ = make_gate(load_workers={1: 5})
    await gate.before_request("t1", "t1-root")
    await gate.before_request("t1", "t1-branch-0")

    metrics = await gate.metrics_snapshot()
    assert metrics["counters"]["tasks_created_total"] == 1
    assert metrics["counters"]["protected_admitted_total"] == 1
    assert metrics["counters"]["opportunistic_admitted_total"] == 1
    assert metrics["gauges"]["tasks_total"] == 1


@pytest.mark.asyncio
async def test_start_stop_background_loop_is_idempotent_and_clean():
    gate, load = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, reconcile_interval_seconds=0.02),
    )
    gate.start()
    gate.start()  # no-op second call, mirrors ThunderAgentScheduler.start()

    await gate.before_request("t1", "t1-root")
    deferred = asyncio.ensure_future(gate.before_request("t1", "t1-branch-0"))
    await asyncio.sleep(0.05)
    load.workers = {1: 1}

    decision = await asyncio.wait_for(deferred, timeout=1.0)
    assert decision.was_deferred is True

    await gate.stop()
    await gate.stop()  # no-op second call
