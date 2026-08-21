# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async DAG-honoring replay client for the Agentic TAPER P0 experiment (P0-3).

Drives one ``(model, arm, sweep-point)`` cell: reads a seeded agentic trace,
respects each row's open-loop arrival ``timestamp`` and ``wait_for`` join
dependencies, stamps task lineage headers, routes every request through the
permit gate (the arm decides the gate's policy), dispatches to an
OpenAI-compatible endpoint, and records per-request client-side latency.

Arm → gate policy (one trace, one gate, one variable changes):

    A0  STATIC_CAP k=1     one request per task at a time (clean baseline)
    A1  EAGER              admit everything as soon as deps clear
    A2  STATIC_CAP k=K     fixed per-task cap
    A3  FPM  threshold=tau task-aware gate on live FPM load

The frontend is pluggable: ``HttpFrontend`` for a real Dynamo+SGLang endpoint
(validated on the box) and ``MockFrontend`` — load-dependent, deterministic —
so the scheduling / gate / DAG logic is unit-tested with no server or GPU.
"""

from __future__ import annotations

import asyncio
import enum
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Optional

from permit_gate import GateRequest, PermitGate, Policy
from prompt_synth import synth_prompt

HEADER_SESSION = "x-dynamo-session-id"
HEADER_PARENT = "x-dynamo-parent-session-id"


class Arm(enum.Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"


@dataclass
class Timing:
    request_id: str
    task_id: str
    role: str
    arm: str
    model: str
    t_arrival: float = 0.0
    t_submit: float = 0.0     # eligible (arrival + deps) and handed to the gate
    t_admit: float = 0.0      # gate released it to dispatch
    t_first_token: float = 0.0
    t_done: float = 0.0
    ttft_ms: float = 0.0
    itls_ms: list = field(default_factory=list)
    ok: bool = True

    @property
    def gate_wait_ms(self) -> float:
        return (self.t_admit - self.t_submit) * 1000.0

    def itl_pct(self, q: float) -> float:
        if not self.itls_ms:
            return 0.0
        s = sorted(self.itls_ms)
        i = min(len(s) - 1, int(q * len(s)))
        return s[i]


# --------------------------------------------------------------------------- #
# Frontends
# --------------------------------------------------------------------------- #

class MockFrontend:
    """Deterministic, load-dependent stand-in for a real engine.

    ITL inflates with the engine's concurrent in-flight count, reproducing the
    "wider shared decode step" the experiment is about — so the offline
    integration test can show A1 hurting the victim tail and the gate helping.
    NOT a performance model; only enough structure to exercise the pipeline.
    """

    needs_prompt = False  # mock ignores prompt text; skip synthesis for speed

    def __init__(self, base_ttft_ms=20.0, base_itl_ms=10.0, alpha=0.06, time_scale=0.001):
        self.base_ttft_ms = base_ttft_ms
        self.base_itl_ms = base_itl_ms
        self.alpha = alpha          # per-in-flight ITL inflation
        self.time_scale = time_scale  # compress simulated ms -> wall seconds for fast tests
        self.engine = None          # set by ReplayEngine for the in-flight view

    async def complete(self, *, prompt, max_tokens, headers, record: Timing, loop):
        # Sample the shared decode-step width once at dispatch. One sleep for the
        # whole request (not per token): asyncio.sleep has a ~1 ms floor, so
        # per-token sleeps would make an offline matrix take minutes. Overlap
        # (hence load-dependence) still comes from open-loop arrivals + request
        # duration.
        inflight = self.engine.inflight if self.engine else 1
        infl = max(0, inflight - 1)
        ttft = self.base_ttft_ms * (1 + self.alpha * infl)
        itl = self.base_itl_ms * (1 + self.alpha * infl)
        n = max(1, min(max_tokens, 8))  # cap simulated tokens for speed
        record.ttft_ms = ttft
        record.itls_ms = [itl] * n
        total_ms = ttft + n * itl
        await asyncio.sleep(total_ms * self.time_scale)
        record.t_first_token = record.t_admit + ttft * self.time_scale
        record.t_done = loop.time()
        record.ok = True


class HttpFrontend:
    """Real OpenAI-compatible frontend (streaming). Validated on the box.

    Uses aiohttp, imported lazily so this module loads without it. Parses SSE
    chunks to time first token and inter-token latencies client-side.
    """

    needs_prompt = True

    def __init__(self, base_url: str, model: str, path: str = "/v1/completions",
                 timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.path = path
        self.timeout_s = timeout_s
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(self, *, prompt, max_tokens, headers, record: Timing, loop):
        session = await self._ensure_session()
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True}
        last = None
        try:
            async with session.post(self.base_url + self.path, json=body, headers=headers,
                                    timeout=self.timeout_s) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    now = loop.time()
                    if record.t_first_token == 0.0:
                        record.t_first_token = now
                        record.ttft_ms = (now - record.t_admit) * 1000.0
                    elif last is not None:
                        record.itls_ms.append((now - last) * 1000.0)
                    last = now
                record.t_done = loop.time()
                record.ok = True
        except Exception:
            record.ok = False
            record.t_done = loop.time()

    async def close(self):
        if self._session is not None:
            await self._session.close()


# --------------------------------------------------------------------------- #
# Replay engine
# --------------------------------------------------------------------------- #

def _is_protected(role: str) -> bool:
    # The task baseline that must always run: the victim's sole request and the
    # aggressor's root. Branches and joins are opportunistic.
    return role in ("victim", "root")


class ReplayEngine:
    def __init__(self, rows: list[dict], arm: Arm, model: str, frontend, *,
                 k: Optional[int] = None, load_source=None, load_threshold=None,
                 drain_interval_ms: float = 50.0, words_per_block: int = 400,
                 clock_scale: float = 1.0):
        self.rows = rows
        self.arm = arm
        self.model = model
        self.frontend = frontend
        if hasattr(frontend, "engine"):
            frontend.engine = self
        self.k = k
        self.load_source = load_source
        self.load_threshold = load_threshold
        self.drain_interval_ms = drain_interval_ms
        self.words_per_block = words_per_block
        # Compress open-loop arrival timing. Keep 1.0 for real runs (true
        # inter-arrival timing matters); use <1 only for fast offline dry-runs.
        self.clock_scale = clock_scale

        self.by_id = {r["request_id"]: r for r in rows}
        self.done_events: dict[str, asyncio.Event] = {}
        self.records: dict[str, Timing] = {}
        self.inflight = 0
        self._loop = None
        self._gate: Optional[PermitGate] = None
        self._pending = 0
        self._all_done: Optional[asyncio.Event] = None

    def _make_gate(self) -> PermitGate:
        send = self._on_admit
        if self.arm is Arm.A1:
            return PermitGate(Policy.EAGER, send)
        if self.arm is Arm.A0:
            return PermitGate(Policy.STATIC_CAP, send, k=1)
        if self.arm is Arm.A2:
            if not self.k:
                raise ValueError("A2 requires k")
            return PermitGate(Policy.STATIC_CAP, send, k=self.k)
        if self.arm is Arm.A3:
            if self.load_source is None or self.load_threshold is None:
                raise ValueError("A3 requires load_source and load_threshold")
            return PermitGate(Policy.FPM, send,
                              load_fn=self.load_source.num_decode_requests,
                              load_threshold=self.load_threshold)
        raise ValueError(self.arm)

    def _headers(self, row: dict) -> dict:
        h = {HEADER_SESSION: row["session_id"]}
        if row.get("parent"):
            h[HEADER_PARENT] = row["parent"]
        return h

    def _on_admit(self, req: GateRequest) -> None:
        # Gate released the request; dispatch it.
        row = req.payload
        rec = self.records[row["request_id"]]
        rec.t_admit = self._loop.time()
        self.inflight += 1
        self._loop.create_task(self._dispatch(row, rec))

    async def _dispatch(self, row: dict, rec: Timing) -> None:
        # Only synthesize the (expensive) shared-prefix prompt when the frontend
        # actually sends it; the mock frontend ignores it.
        prompt = (synth_prompt(row["hash_ids"], self.words_per_block)
                  if getattr(self.frontend, "needs_prompt", True) else "")
        try:
            await self.frontend.complete(
                prompt=prompt, max_tokens=row["output_length"],
                headers=self._headers(row), record=rec, loop=self._loop)
        finally:
            self.inflight -= 1
            self._gate.on_complete(row["task_id"])
            self._gate.drain()
            self.done_events[row["request_id"]].set()
            self._pending -= 1
            if self._pending == 0 and self._all_done:
                self._all_done.set()

    async def _feed(self, row: dict, t0: float) -> None:
        rec = self.records[row["request_id"]]
        # Open-loop arrival.
        arrival = t0 + (row.get("timestamp", 0.0) / 1000.0) * self.clock_scale
        delay = arrival - self._loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        rec.t_arrival = self._loop.time()
        # Join dependencies: wait for all branch completions.
        for dep in row.get("wait_for", []):
            if dep in self.done_events:
                await self.done_events[dep].wait()
        rec.t_submit = self._loop.time()
        self._gate.submit(GateRequest(
            task_id=row["task_id"], request_id=row["request_id"],
            payload=row, is_protected=_is_protected(row.get("role", "")),
        ))

    async def _drain_timer(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.drain_interval_ms / 1000.0)
                self._gate.drain()
        except asyncio.CancelledError:
            pass

    async def run(self) -> list[Timing]:
        self._loop = asyncio.get_event_loop()
        self._gate = self._make_gate()
        self._all_done = asyncio.Event()
        self._pending = len(self.rows)
        for r in self.rows:
            self.done_events[r["request_id"]] = asyncio.Event()
            self.records[r["request_id"]] = Timing(
                request_id=r["request_id"], task_id=r["task_id"],
                role=r.get("role", ""), arm=self.arm.value, model=self.model)
        t0 = self._loop.time()
        drain = self._loop.create_task(self._drain_timer()) if self.arm is Arm.A3 else None
        feeders = [self._loop.create_task(self._feed(r, t0)) for r in self.rows]
        await asyncio.gather(*feeders)
        if self._pending > 0:
            await self._all_done.wait()
        if drain:
            drain.cancel()
            await drain
        return list(self.records.values())


# --------------------------------------------------------------------------- #
# Metrics helpers (a thin summary; full analysis is P0-5)
# --------------------------------------------------------------------------- #

def victim_tail(records: list[Timing], q: float = 0.95) -> Optional[float]:
    """q-quantile of victim per-request mean ITL (ms)."""
    vals = [statistics.mean(r.itls_ms) for r in records
            if r.role == "victim" and r.itls_ms]
    if not vals:
        return None
    vals.sort()
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def task_goodput(records: list[Timing], itl_slo_ms: float) -> float:
    """Fraction of tasks whose every request met the ITL SLO."""
    by_task: dict[str, list[Timing]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    if not by_task:
        return 0.0
    good = 0
    for reqs in by_task.values():
        itls = [x for r in reqs for x in r.itls_ms]
        if reqs and all(r.ok for r in reqs) and (not itls or statistics.mean(itls) <= itl_slo_ms):
            good += 1
    return good / len(by_task)


def write_results(records: list[Timing], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            d = asdict(r)
            d["gate_wait_ms"] = round(r.gate_wait_ms, 3)
            f.write(json.dumps(d) + "\n")


def load_trace(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]
