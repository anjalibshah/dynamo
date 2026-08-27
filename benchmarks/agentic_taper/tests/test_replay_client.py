# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the replay client, driven by the mock frontend.

Exercises the full DAG + gate + dispatch pipeline for all four arms with no
server and no GPU. Asserts the deterministic structural facts, plus the load
trend the experiment relies on (serialized A0 does not hurt the victim more than
eager A1).
"""

import asyncio
import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_source import MockLoadSource  # noqa: E402
from replay_client import (Arm, MockFrontend, ReplayEngine, HEADER_PARENT,  # noqa: E402
                           HEADER_SESSION, Timing, apply_server_metrics,
                           parse_frontend_metrics, task_goodput, victim_tail)
from trace_gen import WorkloadConfig, generate  # noqa: E402


def _trace(n_tasks=40, k=5, burst=8.0):
    cfg = WorkloadConfig(n_tasks=n_tasks, fanout_k=k, burst_multiplier=burst,
                         shared_prefix_blocks=2, branch_unique_blocks=1)
    return [__import__("json").loads(r.to_json()) for r in generate(cfg, seed=11)]


class HeaderCapturingFrontend(MockFrontend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = {}

    async def complete(self, *, prompt, max_tokens, headers, record, loop):
        self.seen[record.request_id] = dict(headers)
        await super().complete(prompt=prompt, max_tokens=max_tokens,
                               headers=headers, record=record, loop=loop)


def _run(engine):
    return asyncio.run(engine.run())


class TestCompleteness(unittest.TestCase):
    def test_all_arms_complete_every_request(self):
        rows = _trace()
        for arm, kw in [
            (Arm.A0, {}),
            (Arm.A1, {}),
            (Arm.A2, {"k": 2}),
            (Arm.A3, {"load_threshold": 3}),
        ]:
            fe = MockFrontend()
            ls = None
            if arm is Arm.A3:
                eng_holder = {}
                ls = MockLoadSource(lambda: eng_holder["e"].inflight)
            eng = ReplayEngine(rows, arm, "m", fe, load_source=ls,
                               drain_interval_ms=5, **kw)
            if arm is Arm.A3:
                eng_holder["e"] = eng
            recs = _run(eng)
            self.assertEqual(len(recs), len(rows), f"{arm}: record count")
            self.assertTrue(all(r.t_done > 0 for r in recs), f"{arm}: all done")
            self.assertTrue(all(r.ok for r in recs), f"{arm}: all ok")


class TestHeaders(unittest.TestCase):
    def test_lineage_headers_stamped(self):
        rows = _trace(n_tasks=10)
        fe = HeaderCapturingFrontend()
        _run(ReplayEngine(rows, Arm.A1, "m", fe))
        by_id = {r["request_id"]: r for r in rows}
        for rid, hdr in fe.seen.items():
            row = by_id[rid]
            self.assertEqual(hdr[HEADER_SESSION], row["session_id"])
            if row.get("role") in ("branch", "join"):
                self.assertEqual(hdr[HEADER_PARENT], row["parent"])
            if row.get("role") in ("victim", "root"):
                self.assertNotIn(HEADER_PARENT, hdr)


class TestProtectedGuarantee(unittest.TestCase):
    def test_every_task_baseline_admitted(self):
        rows = _trace(n_tasks=30)
        fe = MockFrontend()
        eng = ReplayEngine(rows, Arm.A0, "m", fe)   # k=1: harshest gate
        _run(eng)
        n_tasks = len({r["task_id"] for r in rows})
        # One protected admit per task, even under the k=1 cap.
        self.assertEqual(eng._gate.n_admitted_protected, n_tasks)


class TestStaticCapHolds(unittest.TestCase):
    def test_smaller_cap_holds_more(self):
        rows = _trace(n_tasks=30, k=6)
        held = {}
        for kk in (1, 3):
            fe = MockFrontend()
            eng = ReplayEngine(rows, Arm.A2, "m", fe, k=kk)
            _run(eng)
            held[kk] = eng._gate.n_held_events
        self.assertGreater(held[1], held[3])   # k=1 holds strictly more


class TestLoadTrend(unittest.TestCase):
    def test_eager_not_better_than_serialized_for_victim(self):
        # With a load-dependent engine, eager fan-out (A1) should not give a
        # *lower* victim mean ITL than the serialized baseline (A0).
        rows = _trace(n_tasks=50, k=8, burst=8.0)

        fe0 = MockFrontend(alpha=0.1)
        recs0 = _run(ReplayEngine(rows, Arm.A0, "m", fe0))
        fe1 = MockFrontend(alpha=0.1)
        recs1 = _run(ReplayEngine(rows, Arm.A1, "m", fe1))

        def vmean(recs):
            vals = [statistics.mean(r.itls_ms) for r in recs
                    if r.role == "victim" and r.itls_ms]
            return statistics.mean(vals) if vals else 0.0

        self.assertLessEqual(vmean(recs0), vmean(recs1) + 1e-6)

    def test_metrics_helpers_run(self):
        rows = _trace(n_tasks=10)
        recs = _run(ReplayEngine(rows, Arm.A1, "m", MockFrontend()))
        self.assertIsNotNone(victim_tail(recs, 0.95))
        g = task_goodput(recs, itl_slo_ms=10_000)  # loose SLO => most pass
        self.assertGreaterEqual(g, 0.0)
        self.assertLessEqual(g, 1.0)


class ServerMetricsTest(unittest.TestCase):
    # Real lines carry TWO request_ids; the completion id may be either one.
    LOG = (
        'ts INFO metrics: request received request_id=91690883-aaaa endpoint=completions\n'
        'ts INFO metrics: request completed request_id=91690883-aaaa-4a56-8d00-7521a0ee960d '
        'model=m endpoint=completions status=success elapsed_ms=158 method=POST '
        'uri=/v1/completions request_id=4287a54d-bbbb-41db-92c2-13db4968d449 '
        'input_tokens=11 output_tokens=48 image_count=0 ttft_ms="53.16" '
        'avg_itl_ms="2.24" decode_worker_id=7\n'
        'ts INFO metrics: request completed request_id=3333-4444-single '
        'output_tokens=1 ttft_ms="40.0"\n'  # single token: no avg_itl_ms -> skipped
    )

    def test_parse_frontend_metrics_keys_every_request_id(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(self.LOG)
            path = f.name
        m = parse_frontend_metrics(path)
        os.unlink(path)
        # both request_ids on the completed line map to the same metrics
        self.assertIn("91690883-aaaa-4a56-8d00-7521a0ee960d", m)
        self.assertIn("4287a54d-bbbb-41db-92c2-13db4968d449", m)
        self.assertAlmostEqual(m["4287a54d-bbbb-41db-92c2-13db4968d449"]["avg_itl_ms"], 2.24)
        self.assertEqual(m["91690883-aaaa-4a56-8d00-7521a0ee960d"]["output_tokens"], 48)
        # single-token line (no avg_itl_ms) is skipped
        self.assertNotIn("3333-4444-single", m)

    def test_apply_server_metrics_overwrites_matched_only(self):
        matched = Timing(request_id="r1", task_id="t", role="victim", arm="A0",
                         model="m", server_request_id="1111-2222",
                         itls_ms=[999.0, 999.0], ttft_ms=5000.0)
        unmatched = Timing(request_id="r2", task_id="t", role="victim", arm="A0",
                           model="m", server_request_id="nope", itls_ms=[42.0])
        metrics = {"1111-2222": {"avg_itl_ms": 2.24, "ttft_ms": 53.16,
                                 "output_tokens": 48}}
        n = apply_server_metrics([matched, unmatched], metrics)
        self.assertEqual(n, 1)
        # matched: mean(itls) == server avg_itl, ttft replaced, token count preserved
        self.assertAlmostEqual(statistics.mean(matched.itls_ms), 2.24)
        self.assertEqual(len(matched.itls_ms), 47)   # output_tokens-1
        self.assertAlmostEqual(matched.ttft_ms, 53.16)
        # unmatched untouched
        self.assertEqual(unmatched.itls_ms, [42.0])


if __name__ == "__main__":
    unittest.main()
