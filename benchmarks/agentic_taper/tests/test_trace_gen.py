# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the trace generator. Runs without GPU/network."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_gen import WorkloadConfig, generate  # noqa: E402


class TestDeterminism(unittest.TestCase):
    def test_same_seed_byte_identical(self):
        cfg = WorkloadConfig(n_tasks=50)
        a = [r.to_json() for r in generate(cfg, seed=42)]
        b = [r.to_json() for r in generate(cfg, seed=42)]
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        cfg = WorkloadConfig(n_tasks=50)
        a = [r.to_json() for r in generate(cfg, seed=1)]
        b = [r.to_json() for r in generate(cfg, seed=2)]
        self.assertNotEqual(a, b)


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.cfg = WorkloadConfig(n_tasks=300, victim_frac=0.6, fanout_k=5,
                                  shared_prefix_blocks=8, branch_unique_blocks=2)
        self.rows = generate(self.cfg, seed=7)
        self.by_id = {r.request_id: r for r in self.rows}

    def test_sorted_by_timestamp(self):
        ts = [r.timestamp for r in self.rows]
        self.assertEqual(ts, sorted(ts))

    def test_aggressor_has_root_k_branches_and_join(self):
        roots = [r for r in self.rows if r.role == "root"]
        for root in roots:
            self.assertEqual(len(root.branches), self.cfg.fanout_k)
            tid = root.task_id
            branches = [r for r in self.rows if r.task_id == tid and r.role == "branch"]
            joins = [r for r in self.rows if r.task_id == tid and r.role == "join"]
            self.assertEqual(len(branches), self.cfg.fanout_k)
            self.assertEqual(len(joins), 1)

    def test_join_waits_for_all_branches(self):
        for r in self.rows:
            if r.role == "join":
                branch_ids = {b.request_id for b in self.rows
                              if b.task_id == r.task_id and b.role == "branch"}
                self.assertEqual(set(r.wait_for), branch_ids)

    def test_branches_share_root_prefix(self):
        # Each branch's hash_ids must start with the root's full prefix.
        for r in self.rows:
            if r.role == "root":
                prefix = r.hash_ids
                for b in self.rows:
                    if b.task_id == r.task_id and b.role == "branch":
                        self.assertEqual(b.hash_ids[:len(prefix)], prefix,
                                         f"branch {b.request_id} lost the shared prefix")

    def test_branches_carry_parent_lineage(self):
        for r in self.rows:
            if r.role in ("branch", "join"):
                self.assertEqual(r.parent, r.task_id)

    def test_branches_have_distinct_session_ids(self):
        # Distinct ids => co-locate via KV overlap, not session affinity.
        for r in self.rows:
            if r.role == "root":
                sids = [b.session_id for b in self.rows
                        if b.task_id == r.task_id and b.role == "branch"]
                self.assertEqual(len(sids), len(set(sids)))
                self.assertNotIn(r.session_id, sids)

    def test_hash_ids_length_matches_input_length(self):
        for r in self.rows:
            self.assertEqual(len(r.hash_ids), r.input_length // self.cfg.block_size)

    def test_simultaneous_branch_launch_by_default(self):
        # jitter=0 => all siblings share the root's timestamp.
        for r in self.rows:
            if r.role == "root":
                root_ts = r.timestamp
                for b in self.rows:
                    if b.task_id == r.task_id and b.role == "branch":
                        self.assertEqual(b.timestamp, root_ts)


class TestSweeps(unittest.TestCase):
    def test_burst_multiplier_compresses_arrivals(self):
        base = generate(WorkloadConfig(n_tasks=200, burst_multiplier=1.0), seed=3)
        burst = generate(WorkloadConfig(n_tasks=200, burst_multiplier=8.0), seed=3)
        self.assertLess(burst[-1].timestamp, base[-1].timestamp)

    def test_serialized_row_is_valid_json_with_required_fields(self):
        rows = generate(WorkloadConfig(n_tasks=10), seed=0)
        for r in rows:
            d = json.loads(r.to_json())
            for req in ("request_id", "session_id", "input_length",
                        "output_length", "hash_ids", "timestamp"):
                self.assertIn(req, d)


if __name__ == "__main__":
    unittest.main()
