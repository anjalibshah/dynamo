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


class TestSiblingPlacement(unittest.TestCase):
    """Option-D: distributed vs concentrated siblings, paired at matched load."""

    def _cfg(self, distribute):
        return WorkloadConfig(n_tasks=120, victim_frac=0.6, fanout_k=5,
                              shared_prefix_blocks=8, branch_unique_blocks=2,
                              distribute_siblings=distribute)

    def test_concentrated_siblings_share_root_prefix(self):
        rows = generate(self._cfg(False), seed=3)
        by_task = {}
        for r in rows:
            by_task.setdefault(r.task_id, {})[r.role] = by_task.get(r.task_id, {}).get(r.role, [])
        # gather per aggressor task: root prefix and each branch's leading run
        agg = {}
        for r in rows:
            if r.task_kind != "aggressor":
                continue
            agg.setdefault(r.task_id, {"root": None, "branches": []})
            if r.role == "root":
                agg[r.task_id]["root"] = r.hash_ids
            elif r.role == "branch":
                agg[r.task_id]["branches"].append(r.hash_ids)
        checked = 0
        for t, d in agg.items():
            pref = d["root"]
            for bh in d["branches"]:
                # concentrated: branch begins with the root's full prefix
                self.assertEqual(bh[:len(pref)], pref)
                checked += 1
        self.assertGreater(checked, 0)

    def test_distributed_siblings_have_unique_prefixes(self):
        rows = generate(self._cfg(True), seed=3)
        agg = {}
        for r in rows:
            if r.task_kind != "aggressor":
                continue
            agg.setdefault(r.task_id, {"root": None, "branches": []})
            if r.role == "root":
                agg[r.task_id]["root"] = r.hash_ids
            elif r.role == "branch":
                agg[r.task_id]["branches"].append(r.hash_ids)
        for t, d in agg.items():
            pref = d["root"]
            leads = [tuple(bh[:len(pref)]) for bh in d["branches"]]
            # no branch shares the root prefix, and no two branches share a lead run
            for lead in leads:
                self.assertNotEqual(list(lead), pref)
            self.assertEqual(len(set(leads)), len(leads))  # all distinct

    def test_paired_invariant_only_hashids_differ(self):
        # Same seed: request ids, timestamps, and input/output lengths must be
        # identical across the two placement modes — only hash_ids may differ.
        conc = {r.request_id: r for r in generate(self._cfg(False), seed=9)}
        dist = {r.request_id: r for r in generate(self._cfg(True), seed=9)}
        self.assertEqual(set(conc), set(dist))
        for rid, rc in conc.items():
            rd = dist[rid]
            self.assertEqual(rc.timestamp, rd.timestamp)
            self.assertEqual(rc.input_length, rd.input_length)
            self.assertEqual(rc.output_length, rd.output_length)
            self.assertEqual(rc.wait_for, rd.wait_for)
            self.assertEqual(len(rc.hash_ids), len(rd.hash_ids))  # same block count
            if rc.role == "branch":
                self.assertNotEqual(rc.hash_ids, rd.hash_ids)     # placement differs


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
