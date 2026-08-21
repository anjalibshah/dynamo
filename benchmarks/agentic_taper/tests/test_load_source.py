# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the load source. Runs without the dynamo bindings."""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_source import MockLoadSource, decode_kv_frac, decode_load  # noqa: E402


def _fpm(num_decode, kv_tokens=0):
    """Duck-typed stand-in for ForwardPassMetrics."""
    return types.SimpleNamespace(
        scheduled_requests=types.SimpleNamespace(
            num_decode_requests=num_decode, sum_decode_kv_tokens=kv_tokens))


class TestDecodeLoad(unittest.TestCase):
    def test_sums_across_workers(self):
        self.assertEqual(decode_load([_fpm(3), _fpm(5), _fpm(0)]), 8)

    def test_empty_is_zero(self):
        self.assertEqual(decode_load([]), 0)

    def test_malformed_entries_skipped(self):
        bad = types.SimpleNamespace()  # no scheduled_requests
        self.assertEqual(decode_load([_fpm(4), bad, _fpm(2)]), 6)

    def test_kv_frac_sums(self):
        self.assertEqual(decode_kv_frac([_fpm(1, 100), _fpm(1, 250)]), 350.0)


class TestMockLoadSource(unittest.TestCase):
    def test_reflects_dynamic_load(self):
        state = {"v": 0}
        src = MockLoadSource(lambda: state["v"])
        self.assertEqual(src.num_decode_requests(), 0)
        state["v"] = 42
        self.assertEqual(src.num_decode_requests(), 42)

    def test_freshness_is_nonnegative(self):
        src = MockLoadSource(lambda: 1)
        src.num_decode_requests()
        self.assertGreaterEqual(src.freshness_ms(), 0.0)


if __name__ == "__main__":
    unittest.main()
