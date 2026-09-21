# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone interactive-victim traffic generator for the Harbor/Pi A/B gap.

The Harbor/Pi SWE-bench walkthrough (``thunderagent_router/README.md``) runs
only aggressor traffic: 256 concurrent agent tasks, no bystander. That means it
cannot show the thing Agentic TAPER exists to measure -- tail-latency harm to
an unrelated, co-batched interactive task ("private progress, shared cost").
This script fills that gap without touching Harbor: it runs as a second,
independent process against the same ``DYNAMO_BASE_URL`` while
``harbor run -n 256 --n-concurrent-agents 256 ...`` is in flight, firing short
single-turn interactive requests on an open-loop Poisson schedule.

It is deliberately NOT built on ``trace_gen.py`` / ``replay_client.py``'s
synthetic hash-id prompts (``prompt_synth.py``): those exist so KV-block
sharing is controllable in the synthetic sweep, which is irrelevant here --
victims share no prefix with the Harbor SWE-bench containers. Real short
prompts are used instead so this stream stands on its own as a plausible
interactive workload.

Session identity: each victim gets ``x-dynamo-session-id: <prefix>-<n>`` with
no ``x-dynamo-parent-session-id`` -- a distinct namespace from Harbor's
per-trial session ids, so the two streams never collide and are trivially
separable in ``DYN_REQUEST_TRACE`` output by session-id prefix.

Output is one JSON object per line, field-compatible with the ``Timing``
records ``replay_client.py``/``experiment.py`` produce (``role``,
``task_kind``, ``ttft_ms``, ``itls_ms``, ``ok``, ...) so ``analyze.py``'s
victim-side helpers (``_victim_mean_itls``, ``victim_goodput_at_slo``) can be
pointed at this file directly. Wall-clock epoch fields are added on top,
because -- unlike the single-process synthetic sweep -- correlating this
stream against Harbor's own trace sink requires real timestamps, not just
``loop.time()``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

HEADER_SESSION = "x-dynamo-session-id"

# Small built-in pool of short, realistic single-turn interactive prompts.
# Overridable with --prompts-file (one prompt per line) for a first-party
# sample when available -- see brief Appendix D on preferring real
# distributions over synthetic ones once they can be shared.
DEFAULT_PROMPTS = [
    "Summarize the plot of a short story about a lighthouse keeper in two sentences.",
    "What's a good analogy for explaining TCP congestion control to a beginner?",
    "Write a haiku about debugging a race condition.",
    "List three trade-offs between microservices and a monolith.",
    "Explain the difference between a mutex and a semaphore in one paragraph.",
    "What questions should I ask before adopting a new database?",
    "Give me a one-line commit message for a fix to a null-pointer crash.",
    "What's the time complexity of binary search and why?",
    "Suggest a name for a Python utility that retries flaky network calls.",
    "Briefly explain what a bloom filter is used for.",
]


@dataclass
class VictimTiming:
    request_id: str
    task_id: str
    session_id: str
    role: str = "victim"
    task_kind: str = "victim"
    arm: str = ""
    model: str = ""
    prompt: str = ""
    # Monotonic (loop.time()) fields -- comparable within this process only.
    t_arrival: float = 0.0
    t_submit: float = 0.0
    t_first_token: float = 0.0
    t_done: float = 0.0
    ttft_ms: float = 0.0
    itls_ms: list = field(default_factory=list)
    # Wall-clock epoch ms -- for joining against Harbor's own
    # DYN_REQUEST_TRACE jsonl (a separate process/sink) by time window.
    wall_arrival_ms: float = 0.0
    wall_first_token_ms: float = 0.0
    wall_done_ms: float = 0.0
    ok: bool = True
    error: str = ""
    server_request_id: str = ""

    def itl_pct(self, q: float) -> float:
        if not self.itls_ms:
            return 0.0
        s = sorted(self.itls_ms)
        return s[min(len(s) - 1, int(q * len(s)))]


class VictimClient:
    """Fires open-loop Poisson interactive traffic at a chat-completions endpoint.

    Mirrors replay_client.HttpFrontend's SSE parsing so output is directly
    comparable to the synthetic sweep's client-side latencies, but is
    self-contained (no permit_gate / trace_gen / prompt_synth dependency) so it
    can run as a lightweight second process next to Harbor.
    """

    def __init__(self, base_url: str, model: str, *, path: str = "/v1/chat/completions",
                 session_prefix: str = "victim", arm: str = "", max_tokens: int = 64,
                 timeout_s: float = 60.0, max_conns: int = 32,
                 itl_target_ms: Optional[float] = None,
                 ttft_target_ms: Optional[float] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.path = path
        self.session_prefix = session_prefix
        self.arm = arm
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_conns = max_conns
        self.itl_target_ms = itl_target_ms
        self.ttft_target_ms = ttft_target_ms
        self._session = None
        self._sema: Optional[asyncio.Semaphore] = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            connector = aiohttp.TCPConnector(limit=self.max_conns, limit_per_host=self.max_conns)
            self._session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=self.timeout_s))
            self._sema = asyncio.Semaphore(self.max_conns)
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _send_one(self, n: int, prompt: str, loop: asyncio.AbstractEventLoop) -> VictimTiming:
        session_id = f"{self.session_prefix}-{n}"
        record = VictimTiming(
            request_id=f"{session_id}-req", task_id=session_id, session_id=session_id,
            arm=self.arm, model=self.model, prompt=prompt,
        )
        record.t_arrival = loop.time()
        record.wall_arrival_ms = time.time() * 1000.0

        headers = {HEADER_SESSION: session_id}
        nvext: dict = {}
        if self.itl_target_ms is not None or self.ttft_target_ms is not None:
            router: dict = {}
            if self.itl_target_ms is not None:
                router["itl_target"] = self.itl_target_ms
            if self.ttft_target_ms is not None:
                router["ttft_target"] = self.ttft_target_ms
            nvext["router"] = router
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if nvext:
            body["nvext"] = nvext

        session = await self._ensure_session()
        record.t_submit = loop.time()
        last = None
        try:
            async with self._sema:
                async with session.post(self.base_url + self.path, json=body, headers=headers) as resp:
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
                            record.wall_first_token_ms = time.time() * 1000.0
                            record.ttft_ms = (now - record.t_submit) * 1000.0
                            try:
                                cid = json.loads(data).get("id", "")
                                record.server_request_id = (
                                    cid[len("chatcmpl-"):] if cid.startswith("chatcmpl-") else cid)
                            except Exception:
                                pass
                        elif last is not None:
                            record.itls_ms.append((now - last) * 1000.0)
                        last = now
                    record.t_done = loop.time()
                    record.wall_done_ms = time.time() * 1000.0
                    record.ok = True
        except Exception as e:
            record.ok = False
            record.error = f"{type(e).__name__}: {e}"
            record.t_done = loop.time()
            record.wall_done_ms = time.time() * 1000.0
        return record


async def run(args: argparse.Namespace) -> list[VictimTiming]:
    rng = random.Random(args.seed)
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            loaded = [ln.strip() for ln in f if ln.strip()]
        if loaded:
            prompts = loaded

    client = VictimClient(
        base_url=args.base_url, model=args.model, path=args.path,
        session_prefix=args.session_prefix, arm=args.arm, max_tokens=args.max_tokens,
        timeout_s=args.timeout_s, max_conns=args.max_conns,
        itl_target_ms=args.itl_target_ms, ttft_target_ms=args.ttft_target_ms,
    )
    loop = asyncio.get_event_loop()
    step_ms = args.mean_arrival_ms / max(args.burst_multiplier, 1e-9)

    # Same open-loop-Poisson-then-DAG-independent pattern as trace_gen.py, but
    # generated live: arrival schedule is seeded (reproducible across arms run
    # with the same --seed), not gated on prior requests finishing, and
    # independent of Harbor's own wall-clock duration -- pick --n-victims /
    # --mean-arrival-ms to roughly span the SWE-bench run and rerun if it
    # finishes early.
    records: list[VictimTiming] = []
    tasks: list[asyncio.Task] = []
    now = 0.0
    start = loop.time()
    try:
        for n in range(args.n_victims):
            now += rng.expovariate(1.0 / step_ms)
            target = start + now / 1000.0
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            prompt = prompts[n % len(prompts)] if not args.shuffle_prompts else rng.choice(prompts)
            tasks.append(asyncio.ensure_future(client._send_one(n, prompt, loop)))
        records = await asyncio.gather(*tasks)
    finally:
        await client.close()
    return list(records)


def write_jsonl(records: list[VictimTiming], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")


def _print_summary(records: list[VictimTiming]) -> None:
    ok = [r for r in records if r.ok]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    itls = [x for r in ok for x in r.itls_ms]
    print(f"victims: {len(records)} sent, {len(ok)} ok, {len(records) - len(ok)} failed")
    if ttfts:
        print(f"  ttft_ms  p50={statistics.median(ttfts):.1f} "
              f"p95={sorted(ttfts)[int(0.95 * len(ttfts))]:.1f} "
              f"p99={sorted(ttfts)[min(len(ttfts) - 1, int(0.99 * len(ttfts)))]:.1f}")
    if itls:
        print(f"  itl_ms   p50={statistics.median(itls):.1f} "
              f"p95={sorted(itls)[int(0.95 * len(itls))]:.1f} "
              f"p99={sorted(itls)[min(len(itls) - 1, int(0.99 * len(itls)))]:.1f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fire real interactive victim traffic alongside a live Harbor/Pi run.")
    p.add_argument("--base-url", required=True,
                    help="e.g. http://<host>:8100/v1 -- same DYNAMO_BASE_URL Harbor uses")
    p.add_argument("--model", required=True, help="e.g. dynamo/MiniMaxAI/MiniMax-M2")
    p.add_argument("--path", default="/v1/chat/completions")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--arm", default="", help="label written into each record, e.g. ta|kv|taper")
    p.add_argument("--session-prefix", default="victim",
                    help="x-dynamo-session-id namespace; keep distinct from Harbor's trial ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-victims", type=int, default=200)
    p.add_argument("--mean-arrival-ms", type=float, default=2000.0,
                    help="Poisson mean inter-arrival at burst_multiplier=1")
    p.add_argument("--burst-multiplier", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-conns", type=int, default=32)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--itl-target-ms", type=float, default=None,
                    help="stamped as nvext.router.itl_target when set (brief Appendix D)")
    p.add_argument("--ttft-target-ms", type=float, default=None,
                    help="stamped as nvext.router.ttft_target when set")
    p.add_argument("--prompts-file", default=None,
                    help="one prompt per line; overrides the built-in pool")
    p.add_argument("--shuffle-prompts", action="store_true",
                    help="sample prompts with replacement instead of cycling in order")
    args = p.parse_args()

    records = asyncio.run(run(args))
    write_jsonl(records, args.out)
    _print_summary(records)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
