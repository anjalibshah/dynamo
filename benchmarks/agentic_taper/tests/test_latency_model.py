# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ITL(load) latency model behind A3's budget rule."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latency_model import LatencyModel  # noqa: E402


class LatencyModelTest(unittest.TestCase):
    def test_fit_recovers_a_known_line(self):
        # y = 10 + 3x, sampled exactly -> fit must recover it
        pts = [(0, 10.0), (2, 16.0), (4, 22.0), (8, 34.0)]
        m = LatencyModel.fit(pts)
        self.assertAlmostEqual(m.t0_ms, 10.0, places=6)
        self.assertAlmostEqual(m.beta_ms_per_req, 3.0, places=6)
        self.assertEqual(m.n_points, 4)

    def test_itl_and_inverse(self):
        m = LatencyModel(t0_ms=20.0, beta_ms_per_req=5.0)
        self.assertEqual(m.itl_at(4), 40.0)
        self.assertEqual(m.load_at(40.0), 4.0)
        # admit boundary for a sibling: f^-1(slo) - 1
        self.assertEqual(m.threshold_for_slo(50.0), 5.0)

    def test_flat_when_no_load_spread(self):
        # all loads equal -> slope unidentifiable -> conservative flat model
        m = LatencyModel.fit([(3, 40.0), (3, 60.0)])
        self.assertEqual(m.beta_ms_per_req, 0.0)
        self.assertAlmostEqual(m.t0_ms, 50.0)
        self.assertEqual(m.load_at(100.0), float("inf"))

    def test_needs_two_points(self):
        with self.assertRaises(ValueError):
            LatencyModel.fit([(1, 10.0)])

    def test_dict_roundtrip(self):
        m = LatencyModel(t0_ms=2.5, beta_ms_per_req=1.25, n_points=7)
        self.assertEqual(LatencyModel.from_dict(m.to_dict()), m)

    def test_noisy_fit_is_positive_and_sane(self):
        # increasing-ish data with noise -> positive slope, intercept near data
        pts = [(1, 22), (2, 28), (4, 39), (8, 61), (16, 105)]
        m = LatencyModel.fit(pts)
        self.assertGreater(m.beta_ms_per_req, 0)
        self.assertGreater(m.itl_at(16), m.itl_at(1))


if __name__ == "__main__":
    unittest.main()
