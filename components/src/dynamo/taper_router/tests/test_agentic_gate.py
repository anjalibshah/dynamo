# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgenticTaperGate (milestone 1). No runtime needed."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from dynamo.taper_router.agentic_gate import AgenticTaperGate, TaperModeConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

# T(n, L) = 5 + 0.1*n + 1e-4*L ms, so a p-token branch adds 0.1 + 1e-4*p ms. With
# tau_agent = 21.63 and rho = 0.8:
# at n=10, L=50k: T = 11.0, slack 0.8*10.63 = 8.5 -> a 1k-token branch (+0.2) fits.
# at n=10, L=140k: T = 20.0, slack 0.8*1.63 = 1.3 -> a 20k-token branch (+2.1) doesn't.
COEFFS = dict(pred_a_ms=5.0, pred_b_ms=0.1, pred_c_ms=1e-4)


@dataclass
class FakeLoad:
    """Stand-in for LoadSnapshotProvider.step_samples()."""

    samples: dict = field(default_factory=dict)
    _counter: int = 0

    def set(self, n: int, ctx: int, wall_ms: float = 0.0, n_prefill: int = 0) -> None:
        self._counter += 1
        self.samples = {1: ((self._counter,), n, ctx, wall_ms, n_prefill)}

    def step_samples(self) -> dict:
        return dict(self.samples)


def make_gate(mode: str = "taper", **kw) -> tuple[AgenticTaperGate, FakeLoad]:
    load = FakeLoad()
    base = dict(reconcile_interval_seconds=0.01, max_hold_seconds=2.0, online_refit=False)
    cfg = TaperModeConfig(mode=mode, **{**base, **COEFFS, **kw})
    return AgenticTaperGate(load, cfg), load


async def pending(coro) -> asyncio.Future:
    fut = asyncio.ensure_future(coro)
    await asyncio.sleep(0.03)
    return fut


# ---------------------------------------------------------------- taper mode


@pytest.mark.asyncio
async def test_first_request_of_task_is_protected_regardless_of_load():
    gate, load = make_gate()
    load.set(n=200, ctx=500_000)  # far over any budget
    d = await gate.before_request("t1", "t1#0", is_trunk=True, prompt_tokens=50_000)
    assert d.protected and not d.was_deferred


@pytest.mark.asyncio
async def test_branch_is_protected_when_task_has_no_protected_request():
    # The legacy gate's failure: trunk waiting on subagents left the task with
    # no protected progress. Here the first subagent request takes the slot.
    gate, load = make_gate()
    load.set(n=200, ctx=500_000)
    d = await gate.before_request("t1", "sub-a#0", is_trunk=False, prompt_tokens=10_000)
    assert d.protected and not d.was_deferred


@pytest.mark.asyncio
async def test_second_request_admitted_when_slack_allows():
    gate, load = make_gate()
    load.set(n=10, ctx=50_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=1_000)
    assert not d.protected and not d.was_deferred


@pytest.mark.asyncio
async def test_second_request_deferred_without_slack_then_promoted_on_completion():
    gate, load = make_gate()
    load.set(n=10, ctx=140_000)
    p = await gate.before_request("t1", "a#0", is_trunk=False)
    fut = await pending(gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000))
    assert not fut.done()
    await gate.after_request("t1", protected=p.protected, request_id="a#0")
    d = await asyncio.wait_for(fut, 1.0)
    assert d.was_deferred and d.protected
    assert gate.stats["promoted"] == 1


@pytest.mark.asyncio
async def test_promotion_prefers_trunk_over_older_branch():
    gate, load = make_gate()
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "sub-a#0", is_trunk=False)
    branch = await pending(gate.before_request("t1", "sub-b#1", is_trunk=False, prompt_tokens=20_000))
    trunk = await pending(gate.before_request("t1", "t1#2", is_trunk=True, prompt_tokens=20_000))
    await gate.after_request("t1", protected=True, request_id="sub-a#0")
    d = await asyncio.wait_for(trunk, 1.0)
    assert d.protected
    assert not branch.done()


@pytest.mark.asyncio
async def test_deferred_request_released_when_load_drops():
    gate, load = make_gate()
    gate.start()
    try:
        load.set(n=10, ctx=140_000)
        await gate.before_request("t1", "a#0", is_trunk=True)
        fut = await pending(gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000))
        assert not fut.done()
        load.set(n=10, ctx=20_000)
        d = await asyncio.wait_for(fut, 1.0)
        assert d.was_deferred and not d.protected
    finally:
        await gate.stop()


@pytest.mark.asyncio
async def test_release_is_cheapest_first():
    gate, load = make_gate(max_release_per_tick=1)
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    big = await pending(gate.before_request("t1", "big#1", is_trunk=False, prompt_tokens=20_000))
    small = await pending(gate.before_request("t1", "small#2", is_trunk=False, prompt_tokens=2_000))
    load.set(n=10, ctx=100_000)  # T=16, slack 4.5: both fit; one release per tick
    async with gate._lock:
        gate._reconcile_locked()
    await asyncio.sleep(0.02)
    assert small.done() and not big.done()


@pytest.mark.asyncio
async def test_interactive_traffic_tightens_budget():
    # n=10, L=120k: T=18.0. Agent slack 0.8*3.63 = 2.9; interactive slack
    # 0.8*0.48 = 0.38. A 15k branch (+1.6) fits only the agent budget.
    gate, load = make_gate()
    load.set(n=10, ctx=120_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=15_000)
    assert not d.was_deferred
    await gate.before_request("victim-1", "victim-1#2", is_trunk=True, is_interactive=True)
    gate._recent_admits.clear()
    fut = await pending(gate.before_request("t1", "c#3", is_trunk=False, prompt_tokens=15_000))
    assert not fut.done()
    fut.cancel()


@pytest.mark.asyncio
async def test_observed_step_time_overrides_optimistic_prediction():
    gate, load = make_gate()
    load.set(n=10, ctx=50_000, wall_ms=30.0)  # predicted 11 ms, observed 30 ms: no slack
    gate._observe()
    await gate.before_request("t1", "a#0", is_trunk=True)
    fut = await pending(gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=1_000))
    assert not fut.done()
    fut.cancel()


@pytest.mark.asyncio
async def test_permits_count_recent_admits():
    gate, load = make_gate()
    load.set(n=10, ctx=110_000)  # T=17.0, slack 3.7; each 10k branch adds 1.1 ms -> 3 fit
    await gate.before_request("t1", "a#0", is_trunk=True)
    admitted = 0
    for i in range(10):
        f = await pending(gate.before_request("t1", f"b#{i + 1}", is_trunk=False, prompt_tokens=10_000))
        if f.done():
            admitted += 1
        else:
            f.cancel()
            break
    assert admitted == 3


@pytest.mark.asyncio
async def test_fails_open_without_load_readings():
    gate, _ = make_gate()
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=100_000)
    assert not d.was_deferred


@pytest.mark.asyncio
async def test_fails_open_when_uncalibrated():
    gate, load = make_gate(pred_a_ms=0.0, pred_b_ms=0.0, pred_c_ms=0.0)
    load.set(n=200, ctx=500_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=100_000)
    assert not d.was_deferred


@pytest.mark.asyncio
async def test_forced_admit_after_max_hold():
    gate, load = make_gate(max_hold_seconds=0.1)
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000)
    assert d.was_deferred and not d.protected
    assert gate.stats["forced"] == 1
    assert not gate._pending


@pytest.mark.asyncio
async def test_shadow_mode_never_holds():
    gate, load = make_gate(shadow_mode=True)
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    d = await gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000)
    assert not d.was_deferred
    assert gate.stats["shadow_would_defer"] == 1


@pytest.mark.asyncio
async def test_tasks_are_independent():
    gate, load = make_gate()
    load.set(n=200, ctx=500_000)
    for t in ("t1", "t2", "t3"):
        d = await gate.before_request(t, f"{t}#0", is_trunk=True)
        assert d.protected


@pytest.mark.asyncio
async def test_at_most_one_protected_per_task():
    gate, load = make_gate()
    load.set(n=10, ctx=50_000)
    decisions = [await gate.before_request("t1", f"r#{i}", is_trunk=False, prompt_tokens=500)
                 for i in range(4)]
    assert sum(d.protected for d in decisions) == 1


# ---------------------------------------------------------------- baselines


@pytest.mark.asyncio
async def test_eager_admits_everything():
    gate, load = make_gate("eager")
    load.set(n=200, ctx=500_000)
    for i in range(5):
        d = await gate.before_request("t1", f"r#{i}", is_trunk=False, prompt_tokens=50_000)
        assert not d.was_deferred


@pytest.mark.asyncio
async def test_cap_limits_inflight_per_task():
    gate, _ = make_gate("cap", cap_k=2)
    await gate.before_request("t1", "r#0", is_trunk=True)
    await gate.before_request("t1", "r#1", is_trunk=False)
    third = await pending(gate.before_request("t1", "r#2", is_trunk=False))
    assert not third.done()
    other = await gate.before_request("t2", "s#0", is_trunk=True)
    assert not other.was_deferred
    await gate.after_request("t1", protected=False, request_id="r#1")
    d = await asyncio.wait_for(third, 1.0)
    assert d.was_deferred


@pytest.mark.asyncio
async def test_cap1_serializes_task():
    gate, _ = make_gate("cap", cap_k=1)
    await gate.before_request("t1", "r#0", is_trunk=True)
    second = await pending(gate.before_request("t1", "r#1", is_trunk=False))
    assert not second.done()
    await gate.after_request("t1", protected=True, request_id="r#0")
    d = await asyncio.wait_for(second, 1.0)
    assert d.protected


# ---------------------------------------------------------------- telemetry


@pytest.mark.asyncio
async def test_observe_records_each_step_once():
    gate, load = make_gate(online_refit=True)
    load.set(n=10, ctx=50_000, wall_ms=12.0)
    gate._observe()
    gate._observe()  # same step key: ignored
    assert gate.predictor.num_observations == 1
    load.set(n=11, ctx=51_000, wall_ms=12.5)
    gate._observe()
    assert gate.predictor.num_observations == 2
    load.set(n=11, ctx=51_000, wall_ms=40.0, n_prefill=1)  # mixed step: observed only
    gate._observe()
    assert gate.predictor.num_observations == 2
    assert gate._observed_ms() == 12.5


@pytest.mark.asyncio
async def test_status_snapshot_reports_task_counters():
    gate, load = make_gate()
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    fut = await pending(gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000))
    snap = await gate.status_snapshot()
    (task,) = snap["tasks"]
    assert task["protected"] and task["deferred"] == 1 and task["n_deferred"] == 1
    fut.cancel()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        AgenticTaperGate(FakeLoad(), TaperModeConfig(mode="bogus"))


@pytest.mark.asyncio
async def test_cancelled_while_held_leaves_no_queue_entry():
    gate, load = make_gate()
    load.set(n=10, ctx=140_000)
    await gate.before_request("t1", "a#0", is_trunk=True)
    fut = await pending(gate.before_request("t1", "b#1", is_trunk=False, prompt_tokens=20_000))
    fut.cancel()
    await asyncio.sleep(0.02)
    assert not gate._pending and not gate._tasks["t1"].deferred
    # The protected slot is still handed off normally afterwards.
    await gate.after_request("t1", protected=True, request_id="a#0")
    assert gate._tasks["t1"].protected is None and not gate._tasks["t1"].inflight
