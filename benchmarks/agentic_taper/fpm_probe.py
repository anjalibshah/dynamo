# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record and summarize every forward pass of a vLLM worker (NEXT_STEPS 7.3e diagnostics).

The worker's InstrumentedScheduler publishes one ForwardPassMetrics message per
forward pass on a raw ZMQ PUB socket (DYN_FORWARDPASS_METRIC_PORT, 20081 in
run_smoketest_1gpu.sh). This subscribes to that socket directly: read-only,
outside the token path, no Dynamo runtime needed. Unlike the gate's 20 Hz
sampling of the latest message, it sees every step.

    # inside the stack container (needs pyzmq + dynamo):
    python3 fpm_probe.py record --port 20081 --out fpm.jsonl
    # anywhere:
    python3 fpm_probe.py summarize fpm.jsonl [--slo-ms 9.90]

The summary answers two questions:
  1. step time with only trunk turns running (run it on a no-fan-out workload);
  2. whether slow steps are mixed prefill+decode steps or long-context decode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def record(port: int, out: str) -> None:
    import zmq

    from dynamo.common.forward_pass_metrics import decode

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(f"tcp://127.0.0.1:{port}")
    last_flush = time.monotonic()
    with open(out, "a") as f:
        while True:
            frames = sock.recv_multipart()
            m = decode(frames[-1])
            if m is None or not m.wall_time:  # skip idle heartbeats
                continue
            s, q = m.scheduled_requests, m.queued_requests
            f.write(json.dumps({
                "t": time.time(), "worker": m.worker_id, "dp": m.dp_rank,
                "counter": m.counter_id, "wall_ms": m.wall_time * 1000.0,
                "n_prefill": s.num_prefill_requests, "prefill_tok": s.sum_prefill_tokens,
                "n_decode": s.num_decode_requests, "decode_kv_tok": s.sum_decode_kv_tokens,
                "q_prefill": q.num_prefill_requests, "q_decode": q.num_decode_requests,
            }) + "\n")
            if time.monotonic() - last_flush > 1.0:
                f.flush()
                last_flush = time.monotonic()


def _pct(v: list[float], f: float) -> float:
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


def summarize(path: str, slo_ms: float, skip_first_s: float) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        sys.exit(f"no steps in {path}")
    t0 = rows[0]["t"] + skip_first_s
    rows = [r for r in rows if r["t"] >= t0]
    span = rows[-1]["t"] - rows[0]["t"]
    print(f"{path}: {len(rows)} steps over {span / 60:.1f} min "
          f"(first {skip_first_s:g}s skipped); SLO line {slo_ms:g} ms")

    def line(label: str, rs: list[dict]) -> None:
        w = sorted(r["wall_ms"] for r in rs)
        if not w:
            print(f"  {label:<28} -")
            return
        over = sum(x > slo_ms for x in w) / len(w)
        dec = sorted(r["n_decode"] for r in rs)
        print(f"  {label:<28} steps={len(w):>7} ({len(w) / len(rows):5.1%})  "
              f"ms p50={_pct(w, .5):6.1f} p95={_pct(w, .95):6.1f} max={w[-1]:7.1f}  "
              f">SLO={over:5.1%}  decode p50={_pct(dec, .5):.0f}")

    decode_only = [r for r in rows if r["n_prefill"] == 0]
    mixed = [r for r in rows if r["n_prefill"] > 0]
    print("\nby step type (Q2: are slow steps prefill-driven?)")
    line("all", rows)
    line("decode-only", decode_only)
    line("mixed prefill+decode", mixed)
    if mixed:
        pt = sorted(r["prefill_tok"] for r in mixed)
        print(f"  prefill tokens in mixed steps p50={_pct(pt, .5):.0f} p95={_pct(pt, .95):.0f}")
    slow = [r for r in rows if r["wall_ms"] > slo_ms]
    if slow:
        print(f"  steps over SLO that contain prefill: "
              f"{sum(r['n_prefill'] > 0 for r in slow) / len(slow):.1%}")

    print("\ndecode-only steps by total decode context (Q1: long-context cost)")
    for lo, hi in [(0, 50e3), (50e3, 100e3), (100e3, 200e3), (200e3, 400e3), (400e3, float("inf"))]:
        line(f"ctx {lo / 1e3:.0f}k-{hi / 1e3:.0f}k tok",
             [r for r in decode_only if lo <= r["decode_kv_tok"] < hi])
    print("\ndecode-only steps by decode count")
    for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 10**9)]:
        line(f"decode {lo}-{hi - 1}", [r for r in decode_only if lo <= r["n_decode"] < hi])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--port", type=int, default=20081)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("path")
    s.add_argument("--slo-ms", type=float, default=9.90)
    s.add_argument("--skip-first-s", type=float, default=300.0)
    a = p.parse_args()
    if a.cmd == "record":
        record(a.port, a.out)
    else:
        summarize(a.path, a.slo_ms, a.skip_first_s)


if __name__ == "__main__":
    main()
