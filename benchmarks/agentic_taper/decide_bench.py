# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU cost of one decode-regulation decision (``TaperSkipMixin._taper_apply``).

Builds a running set shaped like the 24-28 agent MiniMax-M2 runs (agent
requests with 20k-150k tokens of context, a few interactive requests with a
deadline) and times ``_taper_apply`` on a scheduler stub, at a load where the
batch fits the target (decision is free) and at one where it does not.

    python3 decide_bench.py --running 50 100 200 --steps 2000
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from types import SimpleNamespace

from dynamo.vllm.taper_engine import EXTRA_ARGS_KEY, TaperConfig, TaperSkipMixin


class Req:
    """Plain attributes, like vLLM's ``Request``."""

    def __init__(self, rid: str, ctx: int, ident, max_tokens: int, arrival: float):
        self.request_id = rid
        self.num_prompt_tokens = ctx
        self.num_computed_tokens = ctx
        self.num_output_tokens = 1
        self.is_prefill_chunk = False
        self.next_decode_eligible_step = 0
        self.arrival_time = arrival
        extra = {EXTRA_ARGS_KEY: ident, "kv_transfer_params": None}
        self.sampling_params = SimpleNamespace(extra_args=extra, max_tokens=max_tokens)


class Stub(TaperSkipMixin):
    def __init__(self, running, cfg):
        self.running = running
        self.waiting = []
        self.current_step = 0
        self._taper_init(cfg)


def running_set(n: int, interactive: int, seed: int = 0):
    rng = random.Random(seed)
    now = time.time()
    reqs = []
    for i in range(interactive):
        reqs.append(Req(f"i{i}", 2_000, {"interactive": True}, 64, now - 0.3))
    for i in range(n - interactive):
        task = f"t{i // 3}"
        ident = {"task": task, "role": "main" if i % 3 == 0 else "sub"}
        ident["interactive"] = False
        ctx = int(rng.uniform(20_000, 150_000))
        reqs.append(Req(f"a{i}", ctx, ident, 32_768, now - 30))
    return reqs


def bench(n: int, steps: int, overloaded: bool, mode: str) -> tuple:
    cfg = TaperConfig(mode=mode, deadline=(500.0, 18.48), log_s=1e9)
    if not overloaded:
        cfg.coeffs = (1.0, 0.01, 1e-8)  # a fast GPU: the whole batch fits the target
    s = Stub(running_set(n, max(1, n // 25)), cfg)
    times = []
    gc_was = gc.isenabled()
    gc.disable()  # measure the decision, not collector pauses
    try:
        for _ in range(steps):
            t = time.perf_counter()
            d = s._taper_apply()
            times.append((time.perf_counter() - t) * 1e6)
            s.current_step += 1
    finally:
        if gc_was:
            gc.enable()
    times.sort()
    engaged = bool(d and d.engaged)
    return times[len(times) // 2], times[99 * len(times) // 100], engaged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--running", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mode", default="class_blind")
    a = ap.parse_args()
    print(f"{'running':>8} {'load':>11} {'engaged':>8} {'p50 us':>8} {'p99 us':>8}")
    for n in a.running:
        for over in (False, True):
            p50, p99, eng = bench(n, a.steps, over, a.mode)
            load = "overloaded" if over else "fits"
            print(f"{n:>8} {load:>11} {str(eng):>8} {p50:>8.1f} {p99:>8.1f}")


if __name__ == "__main__":
    main()
