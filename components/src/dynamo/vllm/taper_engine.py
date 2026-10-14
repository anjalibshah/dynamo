# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER milestone 2: per-step decode regulation inside the vLLM scheduler.

Why: under agentic overload, interactive requests miss their target because
every decode step is slow (large batch, ~1M tokens of summed context), not
because their first token is late (ENGINE_TAPER_DESIGN.md section 3.0). The
lever is which *running* requests decode in each step.

Policy, once per ``schedule()`` call, over running requests in the decode phase:

  interactive   never skipped (no agent context, or an interactive session prefix)
  protected     skipped only to keep interactive on target; one task's critical
                path (``role_aware`` only): the main session's request, or during
                a fan-out the subagent with the most context plus the join
                stragglers once the subagents drain to <= max(1, ceil(w / 4)) of
                a fan-out of width w, or with no roles the request with the most
                decoded tokens. If interactive + protected + forced alone break
                the target, protected requests are demoted, most context first
  forced        skipped K - 1 steps in a row; decodes this step (progress bound)
  opportunistic the rest; first-fit while the predicted step time stays inside
                T0 + rho * (tau - T0), most-skipped first, then join stragglers,
                then cheapest context

``class_blind`` is the HyGen/BROS-style ablation: interactive always runs, every
agent request is opportunistic. Nothing is skipped when the whole batch already
fits the tightest per-token target present (the policy is free outside overload).
The prefill chunk expected in the same step (``DYN_TAPER_PREFILL_COEFFS``) is
added to every prediction.

Mechanism (vLLM v0.31.0): a skipped request keeps its KV and its place in
``self.running``; the skip sets ``request.next_decode_eligible_step`` to
``current_step + 2`` before the parent ``schedule()`` increments ``current_step``,
so the parent's running loop passes over it for exactly this step. Requests are
never removed from ``self.running`` (that breaks preemption victim choice and
the running-request cap).

This module has no vLLM import so the policy and the skip mixin are unit-tested
against a fake scheduler; ``taper_scheduler.TaperScheduler`` binds the mixin to
Dynamo's ``InstrumentedScheduler``.

Environment:
  DYN_TAPER_ENGINE            off | class_blind | role_aware (default off)
  DYN_TAPER_COEFFS            a,b,c of T(n, L) = a + b*n + c*L ms (fit_step_model.py)
  DYN_TAPER_PREFILL_COEFFS    q,p: a step with P > 0 prefill tokens adds q + p*P ms
                              (default 0,0: prefill ignored)
  DYN_TAPER_TAU_INTERACTIVE_MS / DYN_TAPER_TAU_AGENT_MS   per-token targets
  DYN_TAPER_RHO               share of the slack opportunistic decodes may use
  DYN_TAPER_K                 a request decodes at least once every K steps
  DYN_TAPER_MAX_OPP_CTX       coarse guard: opportunistic context per step (0 = off)
  DYN_TAPER_INTERACTIVE_PREFIX  session-id prefix marking interactive sessions
  DYN_TAPER_TRACE             JSONL path for per-step records (offline replay)
  DYN_TAPER_LOG_S             seconds between summary log lines
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

MODES = ("off", "class_blind", "role_aware")
EXTRA_ARGS_KEY = "dynamo_taper"
ROLE_MAIN = "main"
ROLE_SUB = "sub"


@dataclass
class TaperConfig:
    mode: str = "off"
    coeffs: tuple = (6.4013, 0.227066, 0.000016055)
    prefill_coeffs: tuple = (0.0, 0.0)
    tau_interactive_ms: float = 18.48
    tau_agent_ms: float = 21.63
    rho: float = 0.8
    k: int = 4
    max_opp_ctx: int = 0
    interactive_prefix: str = "victim-"
    trace_path: str = ""
    log_s: float = 10.0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "TaperConfig":
        env = os.environ if env is None else env
        c = cls()
        mode = env.get("DYN_TAPER_ENGINE", "off").strip().lower() or "off"
        if mode in ("1", "true", "on"):
            mode = "role_aware"
        if mode not in MODES:
            raise ValueError(f"DYN_TAPER_ENGINE={mode!r}: expected one of {MODES}")
        c.mode = mode
        if env.get("DYN_TAPER_COEFFS"):
            parts = [float(x) for x in env["DYN_TAPER_COEFFS"].split(",")]
            if len(parts) != 3:
                raise ValueError("DYN_TAPER_COEFFS needs three numbers a,b,c")
            c.coeffs = tuple(parts)
        if env.get("DYN_TAPER_PREFILL_COEFFS"):
            parts = [float(x) for x in env["DYN_TAPER_PREFILL_COEFFS"].split(",")]
            if len(parts) != 2:
                raise ValueError("DYN_TAPER_PREFILL_COEFFS needs two numbers q,p")
            c.prefill_coeffs = tuple(parts)
        c.tau_interactive_ms = float(
            env.get("DYN_TAPER_TAU_INTERACTIVE_MS", c.tau_interactive_ms)
        )
        c.tau_agent_ms = float(env.get("DYN_TAPER_TAU_AGENT_MS", c.tau_agent_ms))
        c.rho = float(env.get("DYN_TAPER_RHO", c.rho))
        c.k = int(env.get("DYN_TAPER_K", c.k))
        if c.k < 1:
            raise ValueError("DYN_TAPER_K must be >= 1")
        c.max_opp_ctx = int(env.get("DYN_TAPER_MAX_OPP_CTX", c.max_opp_ctx))
        c.interactive_prefix = env.get(
            "DYN_TAPER_INTERACTIVE_PREFIX", c.interactive_prefix
        )
        c.trace_path = env.get("DYN_TAPER_TRACE", "")
        c.log_s = float(env.get("DYN_TAPER_LOG_S", c.log_s))
        return c


# ---------------------------------------------------------------------------
# Identity: handler side (agent context -> SamplingParams.extra_args)
# ---------------------------------------------------------------------------


def identity_from_request(request: Mapping[str, Any], interactive_prefix: str) -> dict:
    """{"task", "role", "interactive"} for the engine, from the request's agent context.

    The task is the root session: ``parent_session_id`` when present, else the
    session itself (the same rule as the taper router). A request without agent
    context is interactive (plain chat traffic is never regulated)."""
    ctx = request.get("agent_context")
    sid = ctx.get("session_id") if isinstance(ctx, Mapping) else None
    if not isinstance(sid, str) or not sid.strip():
        return {"interactive": True}
    sid = sid.strip()
    if interactive_prefix and sid.startswith(interactive_prefix):
        return {"interactive": True}
    parent = ctx.get("parent_session_id")
    task = parent if isinstance(parent, str) and parent else sid
    return {
        "task": task,
        "role": ROLE_MAIN if task == sid else ROLE_SUB,
        "interactive": False,
    }


def attach_identity(
    sampling_params: Any, request: Mapping[str, Any], cfg: TaperConfig
) -> None:
    """Copy the request's TAPER identity into ``sampling_params.extra_args``.

    ``extra_args`` reaches the engine ``Request`` unchanged in vLLM v0.31.0 (the
    KV-connector transfer params ride the same field)."""
    if not cfg.enabled:
        return
    extra = sampling_params.extra_args
    extra = dict(extra) if isinstance(extra, dict) else {}
    extra[EXTRA_ARGS_KEY] = identity_from_request(request, cfg.interactive_prefix)
    sampling_params.extra_args = extra


# ---------------------------------------------------------------------------
# Policy (pure)
# ---------------------------------------------------------------------------


@dataclass
class Cand:
    """A running request in the decode phase, as the policy sees it."""

    req_id: str
    ctx: int
    interactive: bool = True
    task: Optional[str] = None
    role: str = ""
    decoded: int = 0
    skip_run: int = 0


@dataclass
class Decision:
    skip: set = field(default_factory=set)
    engaged: bool = False
    tau_ms: float = 0.0
    pred_all_ms: float = 0.0
    pred_kept_ms: float = 0.0
    n_forced: int = 0
    n_protected: int = 0
    n_demoted: int = 0
    n_opp: int = 0


def _ceil_quarter(w: int) -> int:
    return max(1, math.ceil(w / 4))


class TaperPolicy:
    """Chooses, each step, which decode-phase requests skip; keeps cross-step state."""

    def __init__(self, cfg: TaperConfig) -> None:
        self.cfg = cfg
        self.bias = 1.0  # EWMA of actual / predicted on decode-only steps
        self._skip_run: dict = {}
        self._peak_subs: dict = (
            {}
        )  # task -> fan-out width since its main session last ran
        self._task_seen: dict = (
            {}
        )  # task -> last decision index it had a running request
        self._n = 0

    def predict(self, n: int, ctx: int) -> float:
        a, b, c = self.cfg.coeffs
        return self.bias * max(0.0, a + b * n + c * ctx)

    def predict_prefill(self, tokens: int) -> float:
        """Extra step time for a prefill chunk of ``tokens`` scheduled with the decodes."""
        if tokens <= 0:
            return 0.0
        q, p = self.cfg.prefill_coeffs
        return self.bias * max(0.0, q + p * tokens)

    def observe(
        self, predicted_ms: float, actual_ms: float, alpha: float = 0.05
    ) -> None:
        if predicted_ms <= 0 or actual_ms <= 0:
            return
        r = actual_ms / (predicted_ms / self.bias)
        self.bias = min(2.0, max(0.5, (1 - alpha) * self.bias + alpha * r))

    def _track_widths(self, by_task: dict) -> None:
        """Fan-out width per task: running subagents, reset while the main session runs.

        Agents pause between LLM calls (tool execution), so the width is a peak
        since the main session last ran, forgotten after ~4096 steps unseen."""
        self._n += 1
        for task, cs in by_task.items():
            subs = sum(c.role == ROLE_SUB for c in cs)
            if any(c.role == ROLE_MAIN for c in cs):
                self._peak_subs[task] = subs
            else:
                self._peak_subs[task] = max(self._peak_subs.get(task, 0), subs)
            self._task_seen[task] = self._n
        if self._n % 4096 == 0:
            for task in [t for t, n in self._task_seen.items() if n < self._n - 4096]:
                self._task_seen.pop(task, None)
                self._peak_subs.pop(task, None)

    def _protected(self, by_task: dict) -> set:
        out = set()
        for task, cs in by_task.items():
            mains = [c for c in cs if c.role == ROLE_MAIN]
            subs = [c for c in cs if c.role == ROLE_SUB]
            if mains:
                out.update(c.req_id for c in mains)
            elif subs:
                # critical-path proxy for the join: the subagent with most context
                out.add(max(subs, key=lambda c: (c.ctx, c.req_id)).req_id)
                if len(subs) <= _ceil_quarter(self._peak_subs.get(task, len(subs))):
                    out.update(c.req_id for c in subs)
            else:  # no roles: attained-service proxy for the critical path
                out.add(max(cs, key=lambda c: (c.decoded, c.req_id)).req_id)
        return out

    def _demote_key(self, c: Cand) -> tuple:
        """Order in which protected requests lose protection: most context first."""
        return (-c.ctx, c.req_id)

    def decide(self, cands: Iterable[Cand], prefill_tokens: int = 0) -> Decision:
        cands = list(cands)
        live = {c.req_id for c in cands}
        self._skip_run = {r: v for r, v in self._skip_run.items() if r in live}
        for c in cands:
            c.skip_run = self._skip_run.get(c.req_id, 0)
        d = Decision()
        if not self.cfg.enabled or not cands:
            return self._commit(d, cands)

        managed = [c for c in cands if not c.interactive and c.task is not None]
        by_task: dict = {}
        for c in managed:
            by_task.setdefault(c.task, []).append(c)
        self._track_widths(by_task)
        d.tau_ms = (
            self.cfg.tau_interactive_ms
            if len(managed) < len(cands)
            else self.cfg.tau_agent_ms
        )
        extra = self.predict_prefill(prefill_tokens)
        d.pred_all_ms = self.predict(len(cands), sum(c.ctx for c in cands)) + extra
        if d.pred_all_ms <= d.tau_ms or not managed:
            d.pred_kept_ms = d.pred_all_ms
            return self._commit(d, cands)

        d.engaged = True
        protected = self._protected(by_task) if self.cfg.mode == "role_aware" else set()
        base_ids = {
            c.req_id for c in cands if c.interactive or c.task is None
        } | protected
        forced = {c.req_id for c in managed if c.skip_run >= self.cfg.k - 1}
        base_ids |= forced
        n = len(base_ids)
        ctx = sum(c.ctx for c in cands if c.req_id in base_ids)
        # interactive precedence: demote protected requests, most context first
        demotable = sorted(
            (c for c in managed if c.req_id in protected and c.req_id not in forced),
            key=self._demote_key,
        )
        for c in demotable:
            if self.predict(n, ctx) + extra <= d.tau_ms:
                break
            protected.discard(c.req_id)
            base_ids.discard(c.req_id)
            n, ctx = n - 1, ctx - c.ctx
            d.n_demoted += 1
        d.n_forced, d.n_protected = len(forced), len(protected)

        t0 = self.predict(n, ctx) + extra
        limit = t0 + self.cfg.rho * max(0.0, d.tau_ms - t0)
        opp_ctx = 0
        straggler_tasks = {
            c.task for c in managed if c.req_id in protected and c.role == ROLE_SUB
        }
        order = sorted(
            (c for c in managed if c.req_id not in base_ids),
            key=lambda c: (
                -c.skip_run,
                0 if c.task in straggler_tasks else 1,
                c.ctx,
                c.req_id,
            ),
        )
        for c in order:
            guard_ok = (
                not self.cfg.max_opp_ctx or opp_ctx + c.ctx <= self.cfg.max_opp_ctx
            )
            if guard_ok and self.predict(n + 1, ctx + c.ctx) + extra <= limit:
                n, ctx, opp_ctx = n + 1, ctx + c.ctx, opp_ctx + c.ctx
                d.n_opp += 1
            else:
                d.skip.add(c.req_id)
        d.pred_kept_ms = self.predict(n, ctx) + extra
        return self._commit(d, cands)

    def _commit(self, d: Decision, cands: list) -> Decision:
        for c in cands:
            self._skip_run[c.req_id] = c.skip_run + 1 if c.req_id in d.skip else 0
        return d

    def skip_run(self, req_id: str) -> int:
        return self._skip_run.get(req_id, 0)


# ---------------------------------------------------------------------------
# Scheduler mixin
# ---------------------------------------------------------------------------


def cand_from_request(req: Any) -> Optional[Cand]:
    """A ``Cand`` for a running vLLM request in the decode phase, else None."""
    if (
        getattr(req, "is_prefill_chunk", False)
        or req.num_computed_tokens < req.num_prompt_tokens
    ):
        return None
    sp = getattr(req, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) if sp is not None else None
    ident = extra.get(EXTRA_ARGS_KEY) if isinstance(extra, dict) else None
    c = Cand(
        req_id=req.request_id,
        ctx=req.num_computed_tokens,
        decoded=req.num_output_tokens,
    )
    if (
        isinstance(ident, dict)
        and not ident.get("interactive", True)
        and ident.get("task")
    ):
        c.interactive, c.task, c.role = (
            False,
            str(ident["task"]),
            str(ident.get("role", "")),
        )
    return c


class TaperSkipMixin:
    """Applies ``TaperPolicy`` to a vLLM (Async)Scheduler; mix in before the scheduler.

    Expects the host to provide ``running``, ``current_step``, ``schedule()`` and
    ``update_from_output()`` with vLLM v0.31.0 semantics."""

    def _taper_init(
        self, cfg: Optional[TaperConfig] = None, *, unsupported: str = ""
    ) -> None:
        self._taper_cfg = cfg or TaperConfig.from_env()
        if unsupported and self._taper_cfg.enabled:
            logger.warning("taper.engine disabled: %s", unsupported)
            self._taper_cfg.mode = "off"
        self._taper = TaperPolicy(self._taper_cfg)
        self._taper_pending: deque = (
            deque()
        )  # (output id, t_sched, pred_ms, decode_only, decision)
        self._taper_last_update = 0.0
        self._taper_trace = (
            open(self._taper_cfg.trace_path, "a")
            if (self._taper_cfg.enabled and self._taper_cfg.trace_path)
            else None
        )
        self._taper_stats = _Stats()
        self._taper_next_log = time.monotonic() + self._taper_cfg.log_s
        if self._taper_cfg.enabled:
            logger.info(
                "taper.engine mode=%s coeffs=%s tau=%.2f/%.2f ms rho=%.2f K=%d guard=%d",
                self._taper_cfg.mode,
                self._taper_cfg.coeffs,
                self._taper_cfg.tau_interactive_ms,
                self._taper_cfg.tau_agent_ms,
                self._taper_cfg.rho,
                self._taper_cfg.k,
                self._taper_cfg.max_opp_ctx,
            )

    def _taper_apply(self) -> Optional[Decision]:
        if not self._taper_cfg.enabled or getattr(self, "_bench_active", False):
            return None
        by_id = {}
        cands = []
        for req in self.running:
            c = cand_from_request(req)
            if c is not None:
                cands.append(c)
                by_id[c.req_id] = req
        d = self._taper.decide(cands, self._taper_prefill_estimate())
        if d.skip:
            # schedule() increments current_step first, so this skips exactly one step.
            nxt = self.current_step + 2
            for rid in d.skip:
                req = by_id[rid]
                req.next_decode_eligible_step = max(req.next_decode_eligible_step, nxt)
        return d

    def _taper_prefill_estimate(self) -> int:
        """Prefill tokens the coming step will likely carry: running prefill chunks,
        then waiting prompts, capped at the step's token budget."""
        if not any(self._taper_cfg.prefill_coeffs):
            return 0
        budget = getattr(self, "max_num_scheduled_tokens", 0) or 8192
        tokens = 0
        for req in self.running:
            if req.num_computed_tokens < req.num_prompt_tokens:
                tokens += req.num_prompt_tokens - req.num_computed_tokens
                if tokens >= budget:
                    return budget
        for req in self.waiting:
            tokens += req.num_prompt_tokens - req.num_computed_tokens
            if tokens >= budget:
                return budget
        return tokens

    def _taper_after_schedule(self, output: Any, d: Optional[Decision]) -> None:
        if d is None or output.total_num_scheduled_tokens <= 0:
            return
        running = {r.request_id: r for r in self.running}
        n = ctx = 0
        decode_only = True
        for rid, k in output.num_scheduled_tokens.items():
            if k != 1:
                decode_only = False
                continue
            r = running.get(rid)
            if r is not None:
                n, ctx = n + 1, ctx + r.num_computed_tokens
        pred = self._taper.predict(n, ctx)
        self._taper_pending.append(
            (id(output), time.monotonic(), pred, decode_only, d, n, ctx)
        )
        self._taper_stats.on_decision(d, self._taper)

    def _taper_on_update(self, output: Any) -> None:
        if not self._taper_pending:
            return
        now = time.monotonic()
        if all(p[0] != id(output) for p in self._taper_pending):
            return  # an empty output (never recorded)
        while self._taper_pending[0][0] != id(output):
            self._taper_pending.popleft()
        _, t_sched, pred, decode_only, d, n, ctx = self._taper_pending.popleft()
        actual = (now - max(t_sched, self._taper_last_update)) * 1e3
        self._taper_last_update = now
        if decode_only and n:
            self._taper.observe(pred, actual)
            self._taper_stats.on_step(pred, actual)
        if self._taper_trace is not None:
            self._taper_trace.write(
                json.dumps(
                    {
                        "t": round(now, 4),
                        "n": n,
                        "ctx": ctx,
                        "decode_only": decode_only,
                        "engaged": d.engaged,
                        "skip": len(d.skip),
                        "forced": d.n_forced,
                        "protected": d.n_protected,
                        "opp": d.n_opp,
                        "tau": d.tau_ms,
                        "pred_all": round(d.pred_all_ms, 3),
                        "pred": round(pred, 3),
                        "actual": round(actual, 3),
                        "bias": round(self._taper.bias, 4),
                    }
                )
                + "\n"
            )
        if now >= self._taper_next_log:
            self._taper_next_log = now + self._taper_cfg.log_s
            if self._taper_trace is not None:
                self._taper_trace.flush()
            logger.info(
                "taper.engine %s", self._taper_stats.summary_and_reset(self._taper.bias)
            )

    def _taper_close(self) -> None:
        if self._taper_trace is not None:
            self._taper_trace.close()
            self._taper_trace = None

    # --- overrides -------------------------------------------------------

    def schedule(self, *args, **kwargs):
        d = self._taper_apply()
        output = super().schedule(*args, **kwargs)
        self._taper_after_schedule(output, d)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        if self._taper_cfg.enabled:
            self._taper_on_update(scheduler_output)
        return result

    def shutdown(self) -> None:
        self._taper_close()
        super().shutdown()


class _Stats:
    """Counters between summary log lines (skipped requests are invisible to FPM)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = self.engaged = self.skipped = self.forced = 0
        self.max_skip_run = 0
        self.err = []

    def on_decision(self, d: Decision, policy: TaperPolicy) -> None:
        self.steps += 1
        self.engaged += d.engaged
        self.skipped += len(d.skip)
        self.forced += d.n_forced
        for rid in d.skip:
            self.max_skip_run = max(self.max_skip_run, policy.skip_run(rid))

    def on_step(self, pred: float, actual: float) -> None:
        self.err.append(abs(pred - actual) / actual)

    def summary_and_reset(self, bias: float) -> str:
        e = sorted(self.err)
        mape = f"{e[len(e) // 2]:.1%}" if e else "-"
        s = (
            f"steps={self.steps} engaged={self.engaged} skipped={self.skipped} "
            f"forced={self.forced} max_skip_run={self.max_skip_run} "
            f"pred_err_p50={mape} bias={bias:.3f}"
        )
        self.reset()
        return s
