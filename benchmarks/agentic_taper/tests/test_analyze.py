# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for analyze.py against a synthetic results directory."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze  # noqa: E402


def _rec(rid, role, itls, ttft=15.0, t0=0.0, t1=1.0):
    return {"request_id": rid, "task_id": rid.split("-")[0], "role": role,
            "arm": "X", "model": "m", "t_arrival": t0, "t_submit": t0,
            "t_admit": t0, "t_first_token": t0 + 0.01, "t_done": t1,
            "ttft_ms": ttft, "itls_ms": itls, "ok": True, "gate_wait_ms": 0.0}


def _write_cell(outdir, cell, records):
    path = os.path.join(outdir, analyze._cell_filename(cell))
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _cell(model, arm, k, burst, rep, param=0, goodput=1.0, pf=8):
    return {"model": model, "arm": arm, "fanout_k": k, "burst": burst,
            "prefix_blocks": pf, "param": float(param), "rep": rep,
            "seed": 1, "goodput": goodput, "victim_p95_itl_ms": 0.0, "n_requests": 0}


class TestCellMetrics(unittest.TestCase):
    def test_victim_percentiles(self):
        recs = [_rec(f"v{i}-req", "victim", [float(i)]) for i in range(1, 101)]
        m = analyze.cell_metrics(recs)
        self.assertEqual(m.n_victim, 100)
        self.assertGreater(m.victim_itl_p95, m.victim_itl_p50)

    def test_ignores_non_victims_for_tail(self):
        recs = [_rec("v0-req", "victim", [10.0]), _rec("a0-b0", "branch", [999.0])]
        m = analyze.cell_metrics(recs)
        self.assertEqual(m.n_victim, 1)
        self.assertEqual(m.victim_itl_p50, 10.0)


class TestChargedExternality(unittest.TestCase):
    def test_paired_diff_same_request_id(self):
        d = tempfile.mkdtemp()
        # Same seed/workload => same victim request ids across A0 and A1.
        a0 = _cell("m", "A0", 5, 3.0, 0)
        a1 = _cell("m", "A1", 5, 3.0, 0)
        # A0: victim ITL 10; A1: victim ITL 30 => externality +20 each.
        _write_cell(d, a0, [_rec("v0-req", "victim", [10.0]),
                            _rec("v1-req", "victim", [10.0])])
        _write_cell(d, a1, [_rec("v0-req", "victim", [30.0]),
                            _rec("v1-req", "victim", [30.0])])
        manifest = [a0, a1]
        ext = analyze.charged_externality(d, manifest)
        key = "m_A1_5_3.0_8_0.0"
        self.assertIn(key, ext)
        self.assertAlmostEqual(ext[key]["median_delta_ms"], 20.0)
        self.assertEqual(ext[key]["n"], 2)


class TestKnee(unittest.TestCase):
    def test_knee_detected_when_goodput_falls_throughput_rises(self):
        curve = [{"burst": 1.0, "goodput": 0.95, "throughput": 100.0},
                 {"burst": 3.0, "goodput": 0.60, "throughput": 180.0},
                 {"burst": 8.0, "goodput": 0.20, "throughput": 240.0}]
        r = analyze.detect_knee(curve)
        self.assertTrue(r["has_knee"])

    def test_no_knee_when_goodput_flat(self):
        curve = [{"burst": 1.0, "goodput": 0.95, "throughput": 100.0},
                 {"burst": 8.0, "goodput": 0.95, "throughput": 240.0}]
        self.assertFalse(analyze.detect_knee(curve)["has_knee"])


class TestH2Verdict(unittest.TestCase):
    def _dir_with_a2_a3(self, a3_goodput):
        d = tempfile.mkdtemp()
        manifest = []
        # A2 caps 2 and 4; A3 taus 4 and 8. One rep each. Victim ITLs encode tail.
        for cap, gp in [(2, 0.5), (4, 0.7)]:
            c = _cell("m", "A2", 5, 3.0, 0, param=cap, goodput=gp)
            _write_cell(d, c, [_rec("v0-req", "victim", [40.0])])
            manifest.append(c)
        for tau, gp in [(4, a3_goodput - 0.05), (8, a3_goodput)]:
            c = _cell("m", "A3", 5, 3.0, 0, param=tau, goodput=gp)
            _write_cell(d, c, [_rec("v0-req", "victim", [20.0])])
            manifest.append(c)
        return d, manifest

    def test_h2_supported_when_a3_beats_best_a2(self):
        d, manifest = self._dir_with_a2_a3(a3_goodput=0.85)
        agg = analyze.aggregate(d, manifest)
        v = analyze.h2_verdict(agg, "m")
        self.assertTrue(v["decidable"])
        self.assertTrue(v["A3_beats_A2_goodput"])   # 0.85 > 0.7
        self.assertTrue(v["H2_supported"])

    def test_h2_negative_when_a3_loses(self):
        d, manifest = self._dir_with_a2_a3(a3_goodput=0.65)  # < best A2 (0.7)
        agg = analyze.aggregate(d, manifest)
        v = analyze.h2_verdict(agg, "m")
        self.assertFalse(v["A3_beats_A2_goodput"])
        self.assertFalse(v["H2_supported"])


class TestEndToEnd(unittest.TestCase):
    def test_summarize_on_real_dry_run_output(self):
        # Run a tiny real dry-run matrix, then analyze it.
        import asyncio
        from experiment import MODELS, Sweep, build_cells, run_matrix
        d = tempfile.mkdtemp()
        sw = Sweep()
        sw.reps = 1
        sw.fanout_k = (5,)
        sw.burst = (1.0, 8.0)
        sw.a2_caps = (2,)
        sw.a3_tau = (8,)
        asyncio.run(run_matrix([MODELS[0]], sw, d, dry_run=True))
        summary = analyze.summarize(d)
        self.assertEqual(summary["models"], [MODELS[0].label])
        self.assertIn(MODELS[0].label, summary["per_model"])
        v = summary["per_model"][MODELS[0].label]
        self.assertIn("H1_call", v)
        self.assertIn("H2", v)
        self.assertTrue(os.path.exists(os.path.join(d, "summary.json")))


if __name__ == "__main__":
    unittest.main()
