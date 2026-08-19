# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded agentic-workload trace generator for the Agentic TAPER P0 experiment.

Emits one JSON object per line in Dynamo's agentic Mooncake schema
(``AgenticMooncakeRow`` in ``lib/data-gen/src/mooncake.rs``). The same seed
produces a byte-identical trace, so all four arms (A0..A3) replay an identical
arrival sequence — the precondition that makes per-request attribution valid
(brief §3, "Four arms").

Two task kinds:

* **victim** — one interactive request with a tight SLO. The bystander whose
  tail latency we measure.
* **aggressor** — one root request that fans out into ``K`` sibling branches
  sharing the root's prefix, then a join request that ``wait_for`` all siblings.

Lineage is explicit: every request carries ``session_id``; branch and join
requests carry the root's ``session_id`` as their parent via the ``parent``
field consumed downstream when the replay client stamps
``x-dynamo-parent-session-id``. Fan-out is expressed structurally through
``branches`` (on the root) and ``wait_for`` (on the join), matching the schema
the native lowering already models and tests.

The generator is deterministic and dependency-free (stdlib ``random`` only), so
it runs and unit-tests without a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

# One KV block = this many tokens. Mirrors the mooncake trace convention
# (hash_ids length == input_length // BLOCK_SIZE). Kept configurable because the
# engine's page size must match at replay time.
DEFAULT_BLOCK_SIZE = 512


@dataclass
class Row:
    """One AgenticMooncakeRow. Field names match the Rust serde contract.

    Optional fields are omitted at serialization time (``skip_serializing_if``
    on the Rust side tolerates their absence), except we always emit the fields
    the replay client needs so the JSONL is self-describing.
    """

    request_id: str
    session_id: str
    input_length: int
    output_length: int
    hash_ids: list[int]
    timestamp: float  # ms since first request (open-loop arrival)
    # Lineage + DAG. `parent` is our extension used by the replay client to
    # stamp x-dynamo-parent-session-id; it is ignored by the Rust row (which
    # infers lineage structurally) so the file remains loadable by both.
    parent: Optional[str] = None
    wait_for: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    policy_class: Optional[str] = None
    # Labels for offline analysis; not interpreted by the engine.
    task_id: str = ""
    task_kind: str = ""  # "victim" | "aggressor"
    role: str = ""       # "victim" | "root" | "branch" | "join"

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items()
             if not (isinstance(v, list) and not v) and v is not None}
        return json.dumps(d, separators=(",", ":"), sort_keys=True)


class HashIdAllocator:
    """Allocates KV-block hash ids so that shared prefixes share ids.

    Two requests whose ``hash_ids`` share a leading run represent a shared KV
    prefix of that many blocks — this is what forces branch co-location via
    KV-overlap routing. Ids are globally unique integers assigned in order, per
    the mooncake convention.
    """

    def __init__(self) -> None:
        self._next = 0

    def fresh(self, n_blocks: int) -> list[int]:
        ids = list(range(self._next, self._next + n_blocks))
        self._next += n_blocks
        return ids

    def extend(self, prefix: list[int], n_blocks: int) -> list[int]:
        """Return ``prefix`` followed by ``n_blocks`` newly allocated ids."""
        return list(prefix) + self.fresh(n_blocks)


@dataclass
class WorkloadConfig:
    n_tasks: int = 200
    victim_frac: float = 0.60          # brief Appendix D default: 60/40
    fanout_k: int = 5                  # {2,5,10} swept
    shared_prefix_blocks: int = 8      # ~4k tokens at block=512; {2..8k} swept
    victim_isl_blocks: int = 4
    victim_osl: int = 64
    branch_unique_blocks: int = 2      # per-branch private prompt beyond prefix
    branch_osl: int = 256
    root_osl: int = 32
    join_osl: int = 128
    mean_arrival_ms: float = 50.0      # Poisson inter-arrival at 1x
    burst_multiplier: float = 1.0      # {1,3,8} — divides inter-arrival
    branch_launch_jitter_ms: float = 0.0  # 0 => siblings launch simultaneously
    victim_policy_class: Optional[str] = None
    agent_policy_class: Optional[str] = None
    block_size: int = DEFAULT_BLOCK_SIZE


def generate(cfg: WorkloadConfig, seed: int) -> list[Row]:
    """Produce a deterministic list of rows sorted by arrival timestamp."""
    rng = random.Random(seed)
    alloc = HashIdAllocator()
    rows: list[Row] = []
    now = 0.0
    step = cfg.mean_arrival_ms / max(cfg.burst_multiplier, 1e-9)

    for t in range(cfg.n_tasks):
        # Open-loop Poisson arrival for the task's first request.
        now += rng.expovariate(1.0 / step)
        is_victim = rng.random() < cfg.victim_frac
        if is_victim:
            rows.append(_victim(cfg, alloc, t, now))
        else:
            rows.extend(_aggressor(cfg, alloc, rng, t, now))

    rows.sort(key=lambda r: (r.timestamp, r.request_id))
    return rows


def _victim(cfg: WorkloadConfig, alloc: HashIdAllocator, t: int, ts: float) -> Row:
    sid = f"victim-{t}"
    return Row(
        request_id=f"{sid}-req",
        session_id=sid,
        input_length=cfg.victim_isl_blocks * cfg.block_size,
        output_length=cfg.victim_osl,
        hash_ids=alloc.fresh(cfg.victim_isl_blocks),
        timestamp=round(ts, 3),
        policy_class=cfg.victim_policy_class,
        task_id=sid,
        task_kind="victim",
        role="victim",
    )


def _aggressor(cfg, alloc, rng, t, ts) -> list[Row]:
    sid = f"agent-{t}"
    prefix = alloc.fresh(cfg.shared_prefix_blocks)
    root = Row(
        request_id=f"{sid}-root",
        session_id=sid,
        input_length=cfg.shared_prefix_blocks * cfg.block_size,
        output_length=cfg.root_osl,
        hash_ids=list(prefix),
        timestamp=round(ts, 3),
        policy_class=cfg.agent_policy_class,
        task_id=sid,
        task_kind="aggressor",
        role="root",
    )
    branch_ids: list[str] = []
    branches: list[Row] = []
    for b in range(cfg.fanout_k):
        bid = f"{sid}-b{b}"
        branch_ids.append(bid)
        jitter = rng.uniform(0, cfg.branch_launch_jitter_ms) if cfg.branch_launch_jitter_ms else 0.0
        branches.append(Row(
            request_id=bid,
            session_id=bid,               # distinct id => co-locate via KV overlap, not affinity
            parent=sid,                    # real lineage for x-dynamo-parent-session-id
            input_length=(cfg.shared_prefix_blocks + cfg.branch_unique_blocks) * cfg.block_size,
            output_length=cfg.branch_osl,
            hash_ids=alloc.extend(prefix, cfg.branch_unique_blocks),
            timestamp=round(ts + jitter, 3),  # jitter=0 => simultaneous launch
            policy_class=cfg.agent_policy_class,
            task_id=sid,
            task_kind="aggressor",
            role="branch",
        ))
    root.branches = list(branch_ids)
    join = Row(
        request_id=f"{sid}-join",
        session_id=sid,
        parent=sid,
        input_length=(cfg.shared_prefix_blocks + cfg.branch_unique_blocks) * cfg.block_size,
        output_length=cfg.join_osl,
        hash_ids=alloc.extend(prefix, cfg.branch_unique_blocks),
        timestamp=round(ts, 3),           # arrival ordering; gated by wait_for at replay
        wait_for=list(branch_ids),        # join runs after all siblings finish
        policy_class=cfg.agent_policy_class,
        task_id=sid,
        task_kind="aggressor",
        role="join",
    )
    return [root, *branches, join]


def write_jsonl(rows: list[Row], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(r.to_json() + "\n")


def _build_cfg(a: argparse.Namespace) -> WorkloadConfig:
    return WorkloadConfig(
        n_tasks=a.n_tasks,
        victim_frac=a.victim_frac,
        fanout_k=a.fanout_k,
        shared_prefix_blocks=a.shared_prefix_blocks,
        burst_multiplier=a.burst_multiplier,
        branch_launch_jitter_ms=a.branch_launch_jitter_ms,
        victim_policy_class=a.victim_policy_class,
        agent_policy_class=a.agent_policy_class,
        block_size=a.block_size,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a seeded agentic trace (Mooncake JSONL).")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=200)
    p.add_argument("--victim-frac", type=float, default=0.60)
    p.add_argument("--fanout-k", type=int, default=5)
    p.add_argument("--shared-prefix-blocks", type=int, default=8)
    p.add_argument("--burst-multiplier", type=float, default=1.0)
    p.add_argument("--branch-launch-jitter-ms", type=float, default=0.0)
    p.add_argument("--victim-policy-class", default=None)
    p.add_argument("--agent-policy-class", default=None)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    a = p.parse_args()
    rows = generate(_build_cfg(a), a.seed)
    write_jsonl(rows, a.out)
    n_v = sum(1 for r in rows if r.task_kind == "victim")
    n_a = sum(1 for r in rows if r.role == "root")
    print(f"wrote {len(rows)} rows to {a.out} ({n_v} victims, {n_a} aggressor tasks, seed={a.seed})")


if __name__ == "__main__":
    main()
