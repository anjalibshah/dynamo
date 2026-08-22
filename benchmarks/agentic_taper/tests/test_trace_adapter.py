# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native-AgenticMooncakeRow -> replay_client adapter."""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_adapter import adapt, load_agentic_trace  # noqa: E402
from replay_client import Arm, MockFrontend, ReplayEngine, _is_protected  # noqa: E402
from trace_gen import WorkloadConfig, generate  # noqa: E402


def _native_task(tid, k=2, ts=0.0):
    """A native fan-out task with NO role/task_id labels (as an export would be)."""
    root = {"request_id": f"{tid}-root", "session_id": tid, "input_length": 4096,
            "output_length": 32, "hash_ids": [0, 1, 2, 3], "timestamp": ts,
            "branches": [f"{tid}-b{i}" for i in range(k)]}
    branches = [{"request_id": f"{tid}-b{i}", "session_id": f"{tid}-b{i}",
                 "input_length": 5120, "output_length": 128,
                 "hash_ids": [0, 1, 2, 3, 10 + i], "timestamp": ts}
                for i in range(k)]
    join = {"request_id": f"{tid}-join", "session_id": tid, "input_length": 5120,
            "output_length": 64, "hash_ids": [0, 1, 2, 3, 99], "timestamp": ts,
            "wait_for": [f"{tid}-b{i}" for i in range(k)]}
    return [root, *branches, join]


class TestRoleDerivation(unittest.TestCase):
    def setUp(self):
        native = _native_task("t0", k=3) + [
            {"request_id": "solo", "session_id": "solo", "input_length": 2048,
             "output_length": 16, "hash_ids": [200, 201], "timestamp": 1.0}]
        self.rows = adapt(native)
        self.by_id = {r["request_id"]: r for r in self.rows}

    def test_roles(self):
        self.assertEqual(self.by_id["t0-root"]["role"], "root")
        self.assertEqual(self.by_id["t0-b0"]["role"], "branch")
        self.assertEqual(self.by_id["t0-b2"]["role"], "branch")
        self.assertEqual(self.by_id["t0-join"]["role"], "join")
        self.assertEqual(self.by_id["solo"]["role"], "single")

    def test_task_ids_group_the_fanout(self):
        for rid in ("t0-root", "t0-b0", "t0-b1", "t0-b2", "t0-join"):
            self.assertEqual(self.by_id[rid]["task_id"], "t0-root")
        self.assertEqual(self.by_id["solo"]["task_id"], "solo")

    def test_parent_lineage_for_children(self):
        # Branch/join parent should be the root's session id.
        self.assertEqual(self.by_id["t0-b0"]["parent"], "t0")
        self.assertEqual(self.by_id["t0-join"]["parent"], "t0")
        self.assertNotIn("parent", self.by_id["t0-root"])
        self.assertNotIn("parent", self.by_id["solo"])

    def test_protected_roles(self):
        # Root and single are protected; branch/join are not.
        self.assertTrue(_is_protected(self.by_id["t0-root"]["role"]))
        self.assertTrue(_is_protected(self.by_id["solo"]["role"]))
        self.assertFalse(_is_protected(self.by_id["t0-b0"]["role"]))
        self.assertFalse(_is_protected(self.by_id["t0-join"]["role"]))


class TestFieldMapping(unittest.TestCase):
    def test_serde_aliases_and_defaults(self):
        native = [{"request_id": "x", "session_id": "x", "input_tokens": 100,
                   "output_tokens": 50, "created_time": 5.0, "hash_ids": [1]}]
        row = adapt(native)[0]
        self.assertEqual(row["input_length"], 100)   # input_tokens alias
        self.assertEqual(row["output_length"], 50)    # output_tokens alias
        self.assertEqual(row["timestamp"], 5.0)       # created_time alias
        self.assertEqual(row["role"], "single")

    def test_output_length_defaults_to_at_least_one(self):
        native = [{"request_id": "x", "session_id": "x", "hash_ids": [1]}]
        self.assertEqual(adapt(native)[0]["output_length"], 1)

    def test_sorted_by_timestamp(self):
        native = [{"request_id": "b", "session_id": "b", "timestamp": 9.0, "hash_ids": []},
                  {"request_id": "a", "session_id": "a", "timestamp": 1.0, "hash_ids": []}]
        rows = adapt(native)
        self.assertEqual([r["request_id"] for r in rows], ["a", "b"])


class TestIdempotence(unittest.TestCase):
    def test_trace_gen_rows_pass_through_unchanged(self):
        # trace_gen already labels role/task_id; the adapter must not clobber them.
        cfg = WorkloadConfig(n_tasks=20, fanout_k=4)
        native = [json.loads(r.to_json()) for r in generate(cfg, seed=5)]
        adapted = adapt(native)
        src = {r["request_id"]: r for r in native}
        for r in adapted:
            if "role" in src[r["request_id"]]:
                self.assertEqual(r["role"], src[r["request_id"]]["role"])
                self.assertEqual(r["task_id"], src[r["request_id"]]["task_id"])


class TestReplayRoundTrip(unittest.TestCase):
    def test_adapted_trace_replays_through_all_arms(self):
        native = []
        for t in range(15):
            native += _native_task(f"t{t}", k=3, ts=float(t))
        rows = adapt(native)
        for arm, kw in [(Arm.A0, {}), (Arm.A1, {}), (Arm.A2, {"k": 2})]:
            recs = asyncio.run(ReplayEngine(rows, arm, "m", MockFrontend(), **kw).run())
            self.assertEqual(len(recs), len(rows))
            self.assertTrue(all(r.t_done > 0 for r in recs), f"{arm}: all complete")
        # Each fan-out task's root must be admitted as the protected request.
        eng = ReplayEngine(rows, Arm.A0, "m", MockFrontend())
        asyncio.run(eng.run())
        self.assertEqual(eng._gate.n_admitted_protected, 15)  # one root per task


class TestFileIO(unittest.TestCase):
    def test_load_agentic_trace_roundtrip(self):
        import tempfile
        native = _native_task("t0", k=2)
        d = tempfile.mkdtemp()
        path = os.path.join(d, "native.jsonl")
        with open(path, "w") as f:
            for r in native:
                f.write(json.dumps(r) + "\n")
        rows = load_agentic_trace(path)
        self.assertEqual(len(rows), len(native))
        self.assertTrue(any(r["role"] == "root" for r in rows))


if __name__ == "__main__":
    unittest.main()
