# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the permit gate. Runs without GPU/network (stdlib unittest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from permit_gate import GateRequest, PermitGate, Policy  # noqa: E402


def _collector():
    sent = []
    return sent, (lambda req: sent.append(req.request_id))


def _req(task, rid, protected=False):
    return GateRequest(task_id=task, request_id=rid, is_protected=protected)


class TestProtectedGuarantee(unittest.TestCase):
    def test_protected_always_admitted_under_static_cap(self):
        sent, send = _collector()
        g = PermitGate(Policy.STATIC_CAP, send, k=1)
        # k=1 => no opportunistic slack at all, yet the protected request runs.
        g.submit(_req("t", "protected", protected=True))
        self.assertEqual(sent, ["protected"])

    def test_protected_always_admitted_under_high_fpm_load(self):
        sent, send = _collector()
        g = PermitGate(Policy.FPM, send, load_fn=lambda: 10_000, load_threshold=1)
        g.submit(_req("t", "protected", protected=True))
        # Load is far over threshold, but the task's baseline still runs.
        self.assertEqual(sent, ["protected"])

    def test_explicit_protected_flag_wins_over_arrival_order(self):
        # A branch arrives before the root; the root is the marked protected one.
        sent, send = _collector()
        g = PermitGate(Policy.FPM, send, load_fn=lambda: 10_000, load_threshold=1)
        g.submit(_req("t", "branch0", protected=False))  # held (over load)
        g.submit(_req("t", "root", protected=True))       # protected -> admitted
        self.assertEqual(sent, ["root"])
        self.assertEqual(g.total_held, 1)


class TestStaticCap(unittest.TestCase):
    def test_cap_limits_concurrency(self):
        sent, send = _collector()
        g = PermitGate(Policy.STATIC_CAP, send, k=3)
        g.submit(_req("t", "r", protected=True))          # protected (1 inflight)
        for i in range(5):
            g.submit(_req("t", f"b{i}"))                   # opportunistic
        # protected + (k-1)=2 opportunistic == 3 inflight; rest held.
        self.assertEqual(len(sent), 3)
        self.assertEqual(g.total_held, 3)

    def test_completion_drains_held(self):
        sent, send = _collector()
        g = PermitGate(Policy.STATIC_CAP, send, k=2)
        g.submit(_req("t", "r", protected=True))
        g.submit(_req("t", "b0"))                          # admitted (inflight=2)
        g.submit(_req("t", "b1"))                          # held
        g.submit(_req("t", "b2"))                          # held
        self.assertEqual(len(sent), 2)
        g.on_complete("t")                                 # frees one slot
        self.assertEqual(len(sent), 3)                     # b1 released
        g.on_complete("t")
        self.assertEqual(len(sent), 4)                     # b2 released
        self.assertEqual(g.total_held, 0)

    def test_cap_is_per_task_not_global(self):
        sent, send = _collector()
        g = PermitGate(Policy.STATIC_CAP, send, k=1)
        # Two tasks, each gets its own protected slot.
        g.submit(_req("t1", "r1", protected=True))
        g.submit(_req("t2", "r2", protected=True))
        self.assertEqual(set(sent), {"r1", "r2"})


class TestFpmGate(unittest.TestCase):
    def test_holds_over_threshold_releases_on_drop(self):
        load = {"v": 100}
        sent, send = _collector()
        g = PermitGate(Policy.FPM, send, load_fn=lambda: load["v"], load_threshold=50)
        g.submit(_req("t", "r", protected=True))           # protected -> admitted
        g.submit(_req("t", "b0"))                           # over load -> held
        g.submit(_req("t", "b1"))                           # held
        self.assertEqual(sent, ["r"])
        self.assertEqual(g.total_held, 2)
        load["v"] = 10                                      # slack returns
        g.drain()                                           # timer path
        self.assertEqual(sent, ["r", "b0", "b1"])

    def test_admits_opportunistic_under_threshold(self):
        sent, send = _collector()
        g = PermitGate(Policy.FPM, send, load_fn=lambda: 5, load_threshold=50)
        g.submit(_req("t", "r", protected=True))
        g.submit(_req("t", "b0"))
        g.submit(_req("t", "b1"))
        self.assertEqual(len(sent), 3)                      # all admitted, load low
        self.assertEqual(g.total_held, 0)


class TestConstruction(unittest.TestCase):
    def test_static_cap_requires_valid_k(self):
        _, send = _collector()
        with self.assertRaises(ValueError):
            PermitGate(Policy.STATIC_CAP, send, k=0)

    def test_fpm_requires_load_fn_and_threshold(self):
        _, send = _collector()
        with self.assertRaises(ValueError):
            PermitGate(Policy.FPM, send, k=3)


if __name__ == "__main__":
    unittest.main()
