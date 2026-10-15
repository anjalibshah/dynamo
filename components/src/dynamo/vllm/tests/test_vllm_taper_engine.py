# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER per-step decode regulation: policy, identity plumbing, and the
skip mixin driven by a fake scheduler with vLLM v0.31.0's running-loop rules."""

from types import SimpleNamespace

import pytest

from dynamo.vllm.taper_engine import (
    EXTRA_ARGS_KEY,
    Cand,
    TaperConfig,
    TaperPolicy,
    TaperSkipMixin,
    attach_identity,
    identity_from_request,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class FakeRequest:
    def __init__(self, rid, ctx, ident=None, prompt=None, decoded=1):
        self.request_id = rid
        self.num_prompt_tokens = ctx if prompt is None else prompt
        self.num_computed_tokens = ctx
        self.num_output_tokens = decoded
        self.is_prefill_chunk = self.num_computed_tokens < self.num_prompt_tokens
        self.next_decode_eligible_step = 0
        extra = {EXTRA_ARGS_KEY: ident} if ident is not None else None
        self.sampling_params = SimpleNamespace(extra_args=extra)


class FakeV031Scheduler:
    """The parts of vLLM v0.31.0 Scheduler.schedule() the skip relies on."""

    def __init__(self, running):
        self.running = list(running)
        self.current_step = 0
        self.preempted = []

    def schedule(self):
        self.current_step += 1  # first statement of the real schedule()
        tokens = {}
        for req in self.running:
            if self.current_step < req.next_decode_eligible_step:
                continue  # the real loop's decode-skip `continue`
            if req.is_prefill_chunk:
                tokens[req.request_id] = req.num_prompt_tokens - req.num_computed_tokens
            else:
                tokens[req.request_id] = 1
        return SimpleNamespace(
            num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values())
        )

    def update_from_output(self, out, _runner_output):
        for req in self.running:
            k = out.num_scheduled_tokens.get(req.request_id)
            if k:
                req.num_computed_tokens += k
                req.is_prefill_chunk = req.num_computed_tokens < req.num_prompt_tokens
                if not req.is_prefill_chunk:
                    req.num_output_tokens += 1

    def shutdown(self):
        pass


class Sched(TaperSkipMixin, FakeV031Scheduler):
    def __init__(self, running, cfg):
        FakeV031Scheduler.__init__(self, running)
        self._taper_init(cfg)


def cfg(mode="role_aware", **kw):
    c = TaperConfig(mode=mode, log_s=1e9)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def agent(task, role):
    return {"task": task, "role": role, "interactive": False}


def overloaded(n_subs=8, ctx=100_000):
    """One interactive chat, task t1's main session, and n_subs subagents of t1."""
    reqs = [FakeRequest("chat", 20_000), FakeRequest("main", ctx, agent("t1", "main"))]
    reqs += [FakeRequest(f"s{i}", ctx, agent("t1", "sub")) for i in range(n_subs)]
    return reqs


def step(s):
    out = s.schedule()
    s.update_from_output(out, None)
    return set(out.num_scheduled_tokens)


# --- the skip mechanism ---------------------------------------------------


def test_off_mode_schedules_everything():
    s = Sched(overloaded(), cfg("off"))
    assert step(s) == {r.request_id for r in s.running}


def test_no_skips_when_batch_fits():
    s = Sched(overloaded(n_subs=2, ctx=10_000), cfg())
    assert step(s) == {r.request_id for r in s.running}


def test_overload_skips_subagents_but_keeps_them_running():
    reqs = overloaded()
    s = Sched(reqs, cfg())
    scheduled = step(s)
    assert {"chat", "main"} <= scheduled
    skipped = {r.request_id for r in reqs} - scheduled
    # base 8.78 ms, limit 16.54 ms, each 100k-token subagent adds ~1.83 ms -> 4 fit
    assert len(skipped) == 4
    assert s.running == reqs  # never removed from running: KV stays resident


def test_skip_lasts_exactly_one_step():
    reqs = overloaded()
    s = Sched(reqs, cfg())
    s._taper_apply()
    skipped = [r for r in reqs if r.next_decode_eligible_step > 0]
    assert skipped
    assert all(r.next_decode_eligible_step == s.current_step + 2 for r in skipped)
    s._taper.decide([])  # forget the decision; the field alone must expire
    out = FakeV031Scheduler.schedule(s)  # the step the skip was for
    assert not {r.request_id for r in skipped} & set(out.num_scheduled_tokens)
    out = FakeV031Scheduler.schedule(s)  # the next step: eligible again
    assert {r.request_id for r in skipped} <= set(out.num_scheduled_tokens)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_progress_bound_every_k_steps(k):
    reqs = overloaded(n_subs=16)
    s = Sched(reqs, cfg(k=k))
    last = {r.request_id: 0 for r in reqs}
    for i in range(1, 60):
        for rid in step(s):
            assert i - last[rid] <= k, (rid, i, last[rid])
            last[rid] = i
    assert all(60 - v <= k for v in last.values())


def test_prefill_requests_are_never_skipped():
    reqs = overloaded() + [
        FakeRequest("p", 100_000, agent("t1", "sub"), prompt=200_000)
    ]
    s = Sched(reqs, cfg())
    assert "p" in step(s)


def test_requests_without_identity_are_never_skipped():
    reqs = [FakeRequest(f"x{i}", 100_000) for i in range(12)]
    s = Sched(reqs, cfg())
    assert step(s) == {r.request_id for r in reqs}


# --- policy ----------------------------------------------------------------


def cands(main=True, n_subs=8, ctx=100_000, task="t1"):
    cs = [Cand("chat", 20_000)]
    if main:
        cs.append(Cand("main", ctx, interactive=False, task=task, role="main"))
    cs += [
        Cand(f"s{i}", ctx, interactive=False, task=task, role="sub")
        for i in range(n_subs)
    ]
    return cs


def test_class_blind_may_skip_the_main_session():
    p = TaperPolicy(cfg("class_blind"))
    skipped = set()
    for _ in range(4):
        skipped |= p.decide(cands(n_subs=12)).skip
    assert "main" in skipped
    assert "chat" not in skipped


def test_role_aware_never_skips_main_session():
    p = TaperPolicy(cfg())
    for _ in range(20):
        d = p.decide(cands(n_subs=12))
        assert d.engaged and "main" not in d.skip and "chat" not in d.skip


def test_join_stragglers_are_protected():
    p = TaperPolicy(cfg())
    p.decide(cands(main=False, n_subs=12))  # fan-out width 12 observed
    # 3 of 12 left (<= ceil(12/4)) -> stragglers protected; pad with another task
    cs = cands(main=False, n_subs=3) + [
        Cand(f"o{i}", 100_000, interactive=False, task="t2", role="sub")
        for i in range(10)
    ]
    d = p.decide(cs)
    assert d.engaged
    assert not {"s0", "s1", "s2"} & d.skip
    assert d.skip  # t2 has 10 > ceil(10/4) running: plain fan-out, opportunistic


def test_no_roles_protects_most_attained_request():
    p = TaperPolicy(cfg())
    cs = [
        Cand(f"r{i}", 100_000, interactive=False, task="t", decoded=i)
        for i in range(12)
    ]
    d = p.decide(cs)
    assert d.engaged and "r11" not in d.skip and d.skip


def test_coarse_guard_caps_opportunistic_context():
    p = TaperPolicy(cfg(max_opp_ctx=150_000))
    d = p.decide(cands())
    assert d.n_opp == 1


def test_bias_tracks_slower_steps_and_is_clamped():
    p = TaperPolicy(cfg())
    for _ in range(200):
        p.observe(10.0 * p.bias, 30.0)
    assert p.bias == 2.0
    before = p.decide(cands()).n_opp
    p.bias = 1.0
    assert p.decide(cands()).n_opp >= before


# --- identity plumbing -------------------------------------------------------


@pytest.mark.parametrize(
    ("ctx", "expected"),
    [
        (None, {"interactive": True}),
        ({"session_id": "victim-3"}, {"interactive": True}),
        (
            {"session_id": "root"},
            {"task": "root", "role": "main", "interactive": False},
        ),
        (
            {"session_id": "child", "parent_session_id": "root"},
            {"task": "root", "role": "sub", "interactive": False},
        ),
    ],
)
def test_identity_from_request(ctx, expected):
    req = {} if ctx is None else {"agent_context": ctx}
    assert identity_from_request(req, "victim-") == expected


@pytest.mark.parametrize(
    ("req", "expected"),
    [
        ({"routing": {"priority": 5}}, {"interactive": True}),
        ({"routing": {"priority": 1}}, {"interactive": True}),
        ({}, {"interactive": False}),  # unset priority is 0
        ({"routing": {"priority": "x"}}, {"interactive": False}),
        (
            {"routing": {"priority": 5}, "agent_context": {"session_id": "root"}},
            {"interactive": True},
        ),
        (
            {"agent_context": {"session_id": "c", "parent_session_id": "root"}},
            {"task": "root", "role": "sub", "interactive": False},
        ),
        # the benchmark prefix rule does not apply in priority mode
        (
            {"agent_context": {"session_id": "victim-3"}},
            {"task": "victim-3", "role": "main", "interactive": False},
        ),
    ],
)
def test_identity_from_priority(req, expected):
    assert identity_from_request(req, "victim-", priority_min=1) == expected


def test_priority_mode_regulates_plain_low_priority_requests():
    lo = {EXTRA_ARGS_KEY: {"interactive": False}}
    reqs = [FakeRequest("chat", 20_000)]
    reqs += [FakeRequest(f"b{i}", 100_000) for i in range(10)]
    for r in reqs[1:]:
        r.sampling_params.extra_args = dict(lo)
    s = Sched(reqs, cfg("class_blind", priority_min=1))
    scheduled = step(s)
    assert "chat" in scheduled
    assert len(scheduled) < len(reqs)  # each plain request is its own task


def test_failsafe_disables_when_bias_stays_pinned():
    s = Sched(overloaded(), cfg(failsafe_steps=3))
    s._taper.bias = 2.0
    for _ in range(2):
        s._taper_failsafe()
    assert s._taper_cfg.enabled
    s._taper_failsafe()
    assert not s._taper_cfg.enabled
    assert step(s) == {r.request_id for r in s.running}  # stock scheduling


def test_failsafe_resets_when_bias_recovers():
    s = Sched(overloaded(), cfg(failsafe_steps=3))
    for b in (2.0, 2.0, 1.2, 2.0, 2.0):
        s._taper.bias = b
        s._taper_failsafe()
    assert s._taper_cfg.enabled


def test_config_priority_from_env():
    c = TaperConfig.from_env(
        {"DYN_TAPER_ENGINE": "class_blind", "DYN_TAPER_PRIORITY_MIN": "1"}
    )
    assert c.priority_min == 1
    assert (
        TaperConfig.from_env({"DYN_TAPER_ENGINE": "class_blind"}).priority_min is None
    )


def test_attach_identity_keeps_other_extra_args():
    sp = SimpleNamespace(extra_args={"kv_transfer_params": {"x": 1}})
    attach_identity(sp, {"agent_context": {"session_id": "root"}}, cfg())
    assert sp.extra_args["kv_transfer_params"] == {"x": 1}
    assert sp.extra_args[EXTRA_ARGS_KEY]["role"] == "main"
    sp = SimpleNamespace(extra_args=None)
    attach_identity(sp, {}, cfg("off"))
    assert sp.extra_args is None


def test_config_from_env():
    c = TaperConfig.from_env(
        {
            "DYN_TAPER_ENGINE": "class_blind",
            "DYN_TAPER_K": "3",
            "DYN_TAPER_COEFFS": "1,2,0.5",
        }
    )
    assert (c.mode, c.k, c.coeffs) == ("class_blind", 3, (1.0, 2.0, 0.5))
    assert not TaperConfig.from_env({}).enabled
    with pytest.raises(ValueError):
        TaperConfig.from_env({"DYN_TAPER_ENGINE": "fastest"})
    with pytest.raises(ValueError):
        TaperConfig.from_env({"DYN_TAPER_ENGINE": "role_aware", "DYN_TAPER_K": "0"})


def test_step_timing_ignores_empty_outputs():
    s = Sched(overloaded(), cfg())
    out = s.schedule()
    s._taper_on_update(SimpleNamespace())  # an empty output never recorded
    assert len(s._taper_pending) == 1
    s.update_from_output(out, None)
    assert not s._taper_pending


def test_fan_out_protects_largest_context_subagent():
    p = TaperPolicy(cfg())
    cs = cands(main=False, n_subs=12)
    cs[5].ctx = 300_000
    skips = {c.req_id: 0 for c in cs}
    for _ in range(40):
        for rid in p.decide(cs).skip:
            skips[rid] += 1
    others = [v for k, v in skips.items() if k not in ("chat", "s4")]
    # cs[5] is s4; forced requests outrank it, so it is demoted only sometimes
    assert skips["s4"] < min(others)


def test_protected_requests_demoted_when_they_break_the_target():
    p = TaperPolicy(cfg())
    # 12 tasks' main sessions alone: 6.4 + 0.227*13 + 1.6e-5*1.22M ~ 29 ms > 18.48
    cs = [Cand("chat", 20_000)] + [
        Cand(f"m{i}", 100_000, interactive=False, task=f"t{i}", role="main")
        for i in range(12)
    ]
    d = p.decide(cs)
    assert d.n_demoted > 0 and d.skip and "chat" not in d.skip
    assert d.pred_kept_ms <= cfg().tau_interactive_ms


def test_prefill_in_the_step_reduces_opportunistic_decodes():
    without = TaperPolicy(cfg()).decide(cands()).n_opp
    p = TaperPolicy(cfg(prefill_coeffs=(1.0, 0.0005)))
    assert p.decide(cands(), prefill_tokens=8192).n_opp < without
