# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native AgenticMooncakeRow JSONL into rows the replay client consumes.

This unlocks **Mode A** validation: replay a *real* Claude Code trace through the
same four arms as the synthetic sweep, to check the P0 verdict holds on a real
agent's fan-out shape. Pipeline (all but this step already exists in the tree):

    claude_trace_export                       # real sessions incl. subagents
      -> request_trace_to_mooncake --agentic  # infers branch/join DAG from lineage
      -> AgenticMooncakeRow JSONL             # lib/data-gen/src/mooncake.rs
      -> load_agentic_trace()  (this module)  # -> replay_client rows

The native row expresses lineage *structurally* (``branches`` on a root,
``wait_for`` on a join) but carries no ``role`` / ``task_id`` / ``parent`` — the
labels the gate and analysis need. We derive them from the branch/join graph:

* **root**   — has non-empty ``branches``
* **join**   — has non-empty ``wait_for`` (and is not a root)
* **branch** — appears in some root's ``branches``
* **single** — none of the above (a standalone request = its own whole task)

``task_id`` is the root of the request's fan-out (itself for root/single). The
adapter is idempotent: rows that already carry ``role``/``task_id`` (e.g. from
``trace_gen``) pass through unchanged. Pure stdlib, no GPU.
"""

from __future__ import annotations

import argparse
import json


def _get(row: dict, *keys, default=None):
    """First present, non-None value among ``keys`` (handles serde aliases)."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return default


def _derive(rows: list[dict]) -> list[tuple]:
    """Return (row, role, task_id) for each row, deriving role/task from lineage."""
    branch_to_root: dict[str, str] = {}
    root_ids: set[str] = set()
    join_ids: set[str] = set()
    for r in rows:
        rid = r["request_id"]
        if r.get("branches"):
            root_ids.add(rid)
            for b in r["branches"]:
                branch_to_root[b] = rid
        if r.get("wait_for"):
            join_ids.add(rid)

    # A join maps to the root that owns the branches it waits for.
    join_to_root: dict[str, str] = {}
    for r in rows:
        rid = r["request_id"]
        if rid in join_ids:
            for w in r.get("wait_for", []):
                if w in branch_to_root:
                    join_to_root[rid] = branch_to_root[w]
                    break

    out = []
    for r in rows:
        rid = r["request_id"]
        role = r.get("role")
        if not role:
            if rid in root_ids:
                role = "root"
            elif rid in branch_to_root:
                role = "branch"
            elif rid in join_ids:
                role = "join"
            else:
                role = "single"
        task_id = r.get("task_id")
        if not task_id:
            if role == "branch":
                task_id = branch_to_root.get(rid, rid)
            elif role == "join":
                task_id = join_to_root.get(rid, rid)
            else:  # root or single
                task_id = rid
        out.append((r, role, task_id))
    return out


def adapt(rows: list[dict]) -> list[dict]:
    """Convert native AgenticMooncakeRow dicts to replay_client rows.

    Sorted by open-loop ``timestamp``. ``output_length`` defaults to 1 so no
    request is zero-token; ``hash_ids`` defaults to empty (no shared prefix →
    no co-location, which is correct if the source trace lacks hashes).
    """
    triples = _derive(rows)
    by_id = {r["request_id"]: r for r in rows}
    result = []
    for r, role, task_id in triples:
        rid = r["request_id"]
        session_id = _get(r, "session_id", default=rid)
        parent = r.get("parent")
        if not parent and role in ("branch", "join"):
            root = by_id.get(task_id)
            parent = (root.get("session_id") if root else None) or task_id
        row = {
            "request_id": rid,
            "session_id": session_id,
            "input_length": int(_get(r, "input_length", "input_tokens", default=0) or 0),
            "output_length": int(_get(r, "output_length", "output_tokens", default=1) or 1),
            "hash_ids": r.get("hash_ids") or [],
            "timestamp": float(_get(r, "timestamp", "created_time", default=0.0) or 0.0),
            "wait_for": r.get("wait_for") or [],
            "branches": r.get("branches") or [],
            "policy_class": r.get("policy_class"),
            "task_id": task_id,
            "role": role,
        }
        if parent:
            row["parent"] = parent
        result.append(row)
    result.sort(key=lambda x: (x["timestamp"], x["request_id"]))
    return result


def load_agentic_trace(path: str) -> list[dict]:
    """Load a native AgenticMooncakeRow JSONL and adapt it for the replay client."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return adapt(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adapt native AgenticMooncakeRow JSONL (e.g. from a Claude Code "
                    "export) into replay_client rows.")
    p.add_argument("--in", dest="inp", required=True, help="native AgenticMooncakeRow JSONL")
    p.add_argument("--out", required=True, help="adapted JSONL for replay_client")
    a = p.parse_args()
    rows = load_agentic_trace(a.inp)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_root = sum(1 for r in rows if r["role"] == "root")
    n_single = sum(1 for r in rows if r["role"] == "single")
    print(f"adapted {len(rows)} rows -> {a.out} "
          f"({n_root} fan-out tasks, {n_single} single-request tasks)")


if __name__ == "__main__":
    main()
