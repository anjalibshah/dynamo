# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-level goodput from Dynamo request traces (pre-registered rule, NEXT_STEPS 7.3e).

Interactive request (session id starting with --victim-prefix):
    TTFT <= --interactive-ttft-ms  and  ITL <= factor x unloaded interactive ITL
Agent turn (any other request with agent identity):
    TTFT <= --agent-ttft-ms        and  ITL <= factor x unloaded agent ITL
Requests with <= 1 output token are judged on TTFT only.

Unloaded anchors are the median ITL of each class in --unloaded-trace
(measured on the stock kv stack), or can be given explicitly.

    python3 trace_goodput.py --unloaded-trace unloaded/trace.jsonl \\
        kv-r1=kv_r1/trace.jsonl taper-r1=taper_r1/trace.jsonl ...
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from trace_fanout import _pct, _records


def _requests(path: str, victim_prefix: str) -> list[dict]:
    out = []
    for r in _records([path]):
        m, ctx = r["request"], r.get("agent_context") or {}
        start, total, ttft = m.get("request_received_ms"), m.get("total_time_ms"), m.get("ttft_ms")
        if start is None or total is None:
            continue
        ntok = m.get("output_tokens") or 0
        itl = (total - ttft) / (ntok - 1) if ttft is not None and ntok > 1 else None
        sess = ctx.get("session_id") or ""
        cls = ("interactive" if sess.startswith(victim_prefix)
               else "agent" if sess else "unlabeled")
        out.append({"cls": cls, "session": sess, "start": float(start),
                    "end": float(start) + float(total), "ttft": ttft, "itl": itl,
                    "out": ntok})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="label=path/to/trace.jsonl")
    p.add_argument("--unloaded-trace", default=None)
    p.add_argument("--unloaded-interactive-itl-ms", type=float, default=None)
    p.add_argument("--unloaded-agent-itl-ms", type=float, default=None)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--interactive-ttft-ms", type=float, default=500.0)
    p.add_argument("--agent-ttft-ms", type=float, default=2000.0)
    p.add_argument("--victim-prefix", default="victim-")
    a = p.parse_args()

    anchors = {"interactive": a.unloaded_interactive_itl_ms, "agent": a.unloaded_agent_itl_ms}
    if a.unloaded_trace:
        base = _requests(a.unloaded_trace, a.victim_prefix)
        for cls in anchors:
            if anchors[cls] is None:
                itls = [r["itl"] for r in base if r["cls"] == cls and r["itl"] is not None]
                if itls:
                    anchors[cls] = statistics.median(itls)
    missing = [c for c, v in anchors.items() if v is None]
    if missing:
        raise SystemExit(f"no unloaded ITL anchor for {missing}; pass --unloaded-trace "
                         "or --unloaded-<class>-itl-ms")
    slo = {"interactive": (a.interactive_ttft_ms, a.factor * anchors["interactive"]),
           "agent": (a.agent_ttft_ms, a.factor * anchors["agent"])}
    for cls, (ttft_b, itl_b) in slo.items():
        print(f"SLO {cls}: TTFT <= {ttft_b:g} ms, ITL <= {itl_b:.2f} ms "
              f"(unloaded ITL {anchors[cls]:.2f} ms x {a.factor:g})")

    def ok(r: dict) -> bool:
        ttft_b, itl_b = slo[r["cls"]]
        if r["ttft"] is None or r["ttft"] > ttft_b:
            return False
        return r["itl"] is None or r["itl"] <= itl_b

    print()
    head = ["run", "goodput(all)", "interactive", "agent", "int ITL p50/p95",
            "agent ITL p50/p95", "agent TTFT p50/p95", "req/s", "out tok/s", "unlabeled"]
    print(" | ".join(head))
    for spec in a.runs:
        label, _, path = spec.partition("=")
        reqs = _requests(path, a.victim_prefix)
        if not reqs:
            print(f"{label} | no requests in {path}")
            continue
        by = defaultdict(list)
        for r in reqs:
            by[r["cls"]].append(r)
        judged = by["interactive"] + by["agent"]
        span_s = (max(r["end"] for r in reqs) - min(r["start"] for r in reqs)) / 1000.0

        def gp(rs):
            return f"{sum(ok(r) for r in rs) / len(rs):.3f} (n={len(rs)})" if rs else "-"

        def pq(rs, key):
            v = [r[key] for r in rs if r[key] is not None]
            return f"{_pct(v, 0.5):.1f}/{_pct(v, 0.95):.1f}" if v else "-"

        print(" | ".join([
            label, gp(judged), gp(by["interactive"]), gp(by["agent"]),
            pq(by["interactive"], "itl"), pq(by["agent"], "itl"), pq(by["agent"], "ttft"),
            f"{len(reqs) / span_s:.2f}", f"{sum(r['out'] for r in reqs) / span_s:.0f}",
            str(len(by["unlabeled"])),
        ]))


if __name__ == "__main__":
    main()
