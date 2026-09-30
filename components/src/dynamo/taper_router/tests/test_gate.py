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
async def test_trunk_request_is_protected():
    gate, _ = make_gate()
    decision = await gate.before_request("t1", "t1-root", is_trunk=True)
    assert decision.protected is True
    assert decision.admitted is True
    assert decision.was_deferred is False


@pytest.mark.asyncio
async def test_branch_request_is_opportunistic():
    gate, _ = make_gate()
    await gate.before_request("t1", "t1-root", is_trunk=True)
    decision = await gate.before_request("t1", "t1-branch-0", is_trunk=False)
    assert decision.protected is False


@pytest.mark.asyncio
async def test_branch_arriving_before_root_does_not_take_protected_slot():
    # Regression: protection used to go to whichever request of a task arrived
    # first. With simultaneous fan-out a branch often beat the root, and the
    # root -- the task's critical path -- got deferred under load.
    gate, _ = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, defer_timeout_seconds=5.0,
                            shadow_mode=False),
    )
    branch = asyncio.ensure_future(gate.before_request("t1", "t1-branch-0", is_trunk=False))
    await asyncio.sleep(0.02)
    assert not branch.done()  # branch deferred even though it arrived first

    root = await asyncio.wait_for(
        gate.before_request("t1", "t1-root", is_trunk=True), timeout=1.0)
    assert root.protected is True
    assert root.was_deferred is False
    branch.cancel()


@pytest.mark.asyncio
async def test_every_trunk_turn_is_protected_under_load():
    gate, _ = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, shadow_mode=False),
    )
    for turn in range(3):
        decision = await asyncio.wait_for(
            gate.before_request("t1", "t1-root", is_trunk=True), timeout=1.0)
        assert decision.protected is True
        await gate.after_request("t1", protected=True)
    assert gate._table.tasks["t1"].protected_total == 3


@pytest.mark.asyncio
async def test_opportunistic_branch_admitted_immediately_under_threshold():
    gate, load = make_gate(load_workers={1: 5}, config=TaperConfig(load_threshold=32.0))
    await gate.before_request("t1", "t1-root", is_trunk=True)
    decision = await gate.before_request("t1", "t1-branch-0", is_trunk=False)
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
    await gate.before_request("t1", "t1-root", is_trunk=True)

    task = asyncio.ensure_future(gate.before_request("t1", "t1-branch-0", is_trunk=False))
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
    await gate.before_request("t1", "t1-root", is_trunk=True)
    decision = await gate.before_request("t1", "t1-branch-0", is_trunk=False)
    assert decision.admitted is True
    assert decision.was_deferred is True
    assert gate._stat_forced_admits == 1


@pytest.mark.asyncio
async def test_shadow_mode_always_admits_but_records_would_defer():
    gate, _ = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, shadow_mode=True),
    )
    await gate.before_request("t1", "t1-root", is_trunk=True)
    decision = await gate.before_request("t1", "t1-branch-0", is_trunk=False)
    assert decision.admitted is True
    assert decision.was_deferred is False
    assert decision.shadow_would_defer is True
    assert gate._stat_shadow_would_defer == 1
    assert gate._stat_deferred == 0


@pytest.mark.asyncio
async def test_after_request_decrements_opportunistic_count_not_protected_slot():
    gate, _ = make_gate(load_workers={1: 5})
    await gate.before_request("t1", "t1-root", is_trunk=True)
    await gate.before_request("t1", "t1-branch-0", is_trunk=False)
    task = gate._table.tasks["t1"]
    assert task.inflight_opportunistic == 1

    # A trunk completing must not release a branch's opportunistic slot.
    await gate.after_request("t1", protected=True)
    assert task.inflight_opportunistic == 1

    await gate.after_request("t1", protected=False)
    assert task.inflight_opportunistic == 0
    decision = await gate.before_request("t1", "t1-branch-1", is_trunk=False)
    assert decision.protected is False


@pytest.mark.asyncio
async def test_end_task_drops_bookkeeping():
    gate, _ = make_gate()
    await gate.before_request("t1", "t1-root", is_trunk=True)
    assert gate.end_task("t1") is True
    assert "t1" not in gate._table.tasks
    assert gate.end_task("t1") is False


@pytest.mark.asyncio
async def test_cold_start_with_no_load_signal_fails_open():
    gate, _ = make_gate(load_workers={})  # no FPM observed yet
    await gate.before_request("t1", "t1-root", is_trunk=True)
    decision = await gate.before_request("t1", "t1-branch-0", is_trunk=False)
    assert decision.admitted is True
    assert decision.was_deferred is False


@pytest.mark.asyncio
async def test_status_snapshot_reports_tasks_and_load():
    gate, _ = make_gate(load_workers={1: 5, 2: 9})
    await gate.before_request("t1", "t1-root", is_trunk=True)
    await gate.before_request("t1", "t1-branch-0", is_trunk=False)

    snapshot = await gate.status_snapshot()
    assert snapshot["tasks_total"] == 1
    assert snapshot["load"] == 9  # max across workers
    assert snapshot["tasks"][0]["task_id"] == "t1"
    assert snapshot["tasks"][0]["protected_total"] == 1
    assert snapshot["tasks"][0]["inflight_opportunistic"] == 1


@pytest.mark.asyncio
async def test_metrics_snapshot_counters():
    gate, _ = make_gate(load_workers={1: 5})
    await gate.before_request("t1", "t1-root", is_trunk=True)
    await gate.before_request("t1", "t1-branch-0", is_trunk=False)

    metrics = await gate.metrics_snapshot()
    assert metrics["counters"]["tasks_created_total"] == 1
    assert metrics["counters"]["protected_admitted_total"] == 1
    assert metrics["counters"]["opportunistic_admitted_total"] == 1
    assert metrics["gauges"]["tasks_total"] == 1


@pytest.mark.asyncio
async def test_start_stop_background_loop_is_idempotent_and_clean():
    gate, load = make_gate(
        load_workers={1: 100},
        config=TaperConfig(load_threshold=32.0, reconcile_interval_seconds=0.02,
                            shadow_mode=False),
    )
    gate.start()
    gate.start()  # no-op second call, mirrors ThunderAgentScheduler.start()

    await gate.before_request("t1", "t1-root", is_trunk=True)
    deferred = asyncio.ensure_future(gate.before_request("t1", "t1-branch-0", is_trunk=False))
    await asyncio.sleep(0.05)
    load.workers = {1: 1}

    decision = await asyncio.wait_for(deferred, timeout=1.0)
    assert decision.was_deferred is True

    await gate.stop()
    await gate.stop()  # no-op second call
