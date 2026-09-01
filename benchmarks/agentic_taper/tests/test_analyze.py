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


class TestH1ColocatedVerdict(unittest.TestCase):
    """H1 is judged on the CO-LOCATED victim externality, not the pooled median —
    the externality is localized to victims sharing the fan-out's worker."""

    def _colo(self, a0_p95, a1_p95):
        base = {"victim_p95_isolated_ms": 20.0, "n_colocated": 10, "n_isolated": 20}
        return {"m_A0_5_3.0": {"victim_p95_colocated_ms": a0_p95, **base},
                "m_A1_5_3.0": {"victim_p95_colocated_ms": a1_p95, **base}}

    def test_deltas_pair_a1_minus_a0(self):
        self.assertEqual(analyze._colocated_ext_deltas(self._colo(30, 50), "m"), [20.0])

    def test_supported_when_colocated_victims_hurt(self):
        h1 = analyze.h1_verdict({}, "m", {"has_knee": True}, {}, self._colo(30, 50))
        self.assertEqual(h1["colocated_externality_median_ms"], 20.0)
        self.assertTrue(h1["H1_supported"])

    def test_negative_when_colocated_flat(self):
        # +0.5ms is below the 2ms noise threshold -> not supported.
        h1 = analyze.h1_verdict({}, "m", {"has_knee": True}, {}, self._colo(30, 30.5))
        self.assertFalse(h1["H1_supported"])

    def test_negative_without_knee_even_if_victims_hurt(self):
        h1 = analyze.h1_verdict({}, "m", {"has_knee": False}, {}, self._colo(30, 50))
        self.assertFalse(h1["H1_supported"])

    def test_pooled_median_negative_does_not_veto_colocated_positive(self):
        # The whole point: pooled charged-externality median can be <=0 while the
        # co-located signal is strongly positive; H1 must key on the latter.
        ext = {"m_A1_5_3.0": {"median_delta_ms": -3.0}}
        h1 = analyze.h1_verdict({}, "m", {"has_knee": True}, ext, self._colo(30, 50))
        self.assertLess(h1["charged_externality_median_ms"], 0)
        self.assertTrue(h1["H1_supported"])


class TestSingleWorkerVerdicts(unittest.TestCase):
    """--pin-mode single: no A0 baseline; H1 is A1's victim tail rising with
    fan-out, H2a is a static cap beating eager at the contended cell."""

    def _agg(self):
        agg = {}
        for b in (1.0, 3.0, 8.0):
            for k, (g1, i1) in {2: (0.65, 37.0), 5: (0.46, 64.0), 10: (0.28, 71.0)}.items():
                agg[("m", "A1", k, b, 8, 0)] = analyze.Agg(
                    goodput_median=g1, victim_itl_p95_median=i1, throughput_median=1000)
            for k, (g2, i2) in {2: (0.65, 30.0), 5: (0.67, 35.0), 10: (0.75, 31.0)}.items():
                agg[("m", "A2", k, b, 8, 2)] = analyze.Agg(
                    goodput_median=g2, victim_itl_p95_median=i2, throughput_median=1000)
        return agg

    def test_h1_supported_when_a1_tail_rises_with_fanout_no_a0(self):
        h1 = analyze.h1_verdict(self._agg(), "m", {"has_knee": False}, {}, {})
        self.assertEqual(h1["h1_basis"], "A1_victim_tail_rises_with_fanout")
        self.assertGreater(h1["a1_victim_itl_span_ms"], 30)
        self.assertTrue(h1["H1_supported"])  # no burst-knee required in this design

    def test_h1_negative_when_a1_tail_flat_across_fanout(self):
        agg = {("m", "A1", k, 8.0, 8, 0): analyze.Agg(goodput_median=0.5,
               victim_itl_p95_median=30.0, throughput_median=1000)
               for k in (2, 5, 10)}
        h1 = analyze.h1_verdict(agg, "m", {"has_knee": False}, {}, {})
        self.assertFalse(h1["H1_supported"])

    def test_gate_beats_eager_at_contended_cell(self):
        g = analyze.gate_vs_eager_verdict(self._agg(), "m")
        self.assertTrue(g["decidable"])
        self.assertEqual((g["k"], g["burst"]), (10, 8.0))   # worst fan-out, top load
        self.assertTrue(g["gate_cuts_victim_tail"])          # 31 < 71
        self.assertTrue(g["gate_keeps_goodput"])             # 0.75 >= 0.28
        self.assertTrue(g["H2a_supported"])


class TestSloSweep(unittest.TestCase):
    def test_victim_goodput_at_slo(self):
        recs = [_rec("v0-req", "victim", [10.0]), _rec("v1-req", "victim", [20.0]),
                _rec("v2-req", "victim", [30.0]), _rec("v3-req", "victim", [40.0])]
        self.assertEqual(analyze.victim_goodput_at_slo(recs, 25.0), 0.5)   # 10,20 pass
        self.assertEqual(analyze.victim_goodput_at_slo(recs, 5.0), 0.0)
        self.assertEqual(analyze.victim_goodput_at_slo(recs, 100.0), 1.0)

    def test_slo_sweep_recomputes_goodput_from_records(self):
        d = tempfile.mkdtemp()
        manifest = []
        # A0 victim ITL 12 (fast), A1 victim ITL 40 (loaded). One workload, one rep.
        for arm, itl, gp in [("A0", 12.0, 1.0), ("A1", 40.0, 0.0),
                             ("A2", 30.0, 0.5), ("A3", 18.0, 0.9)]:
            c = _cell("m", arm, 5, 8.0, 0, param=(2 if arm == "A2" else (8 if arm == "A3" else 0)),
                      goodput=gp)
            _write_cell(d, c, [_rec(f"v0-req", "victim", [itl])])
            manifest.append(c)
        # At SLO 15ms: A0 passes (12<=15), A1 fails (40>15) -> externality visible.
        agg15 = analyze.aggregate(d, manifest, slo_ms=15.0)
        a0 = analyze.arm_goodput_at_load(agg15, "m", "A0", 8.0)
        a1 = analyze.arm_goodput_at_load(agg15, "m", "A1", 8.0)
        self.assertEqual(a0, 1.0)
        self.assertEqual(a1, 0.0)
        # At SLO 50ms: everything passes -> no visible externality.
        agg50 = analyze.aggregate(d, manifest, slo_ms=50.0)
        self.assertEqual(analyze.arm_goodput_at_load(agg50, "m", "A1", 8.0), 1.0)

    def test_summarize_includes_slo_sweep(self):
        d = tempfile.mkdtemp()
        manifest = []
        for arm, itl in [("A0", 12.0), ("A1", 40.0), ("A2", 30.0), ("A3", 18.0)]:
            for burst in (1.0, 8.0):
                c = _cell("m", arm, 5, burst, 0,
                          param=(2 if arm == "A2" else (8 if arm == "A3" else 0)))
                _write_cell(d, c, [_rec("v0-req", "victim", [itl if burst == 8.0 else 10.0])])
                manifest.append(c)
        with open(os.path.join(d, "manifest.jsonl"), "w") as f:
            for c in manifest:
                f.write(json.dumps(c) + "\n")
        s = analyze.summarize(d, slo_grid=(15.0, 50.0))
        self.assertIn("slo_sweep", s)
        self.assertIn("15.0", s["slo_sweep"])
        self.assertIn("m", s["slo_sweep"]["15.0"])
        self.assertIn("H2_supported", s["slo_sweep"]["15.0"]["m"])


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
