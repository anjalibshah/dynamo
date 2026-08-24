# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment driver — runs the model x arm x sweep matrix (P0-8).

Cross-model generality: the paper measured goodput on a Qwen model; here we
sweep the same four arms across TWO models so H1/H2 are tested for each and we
can report whether the externality and the gate's benefit generalize.

    matrix = MODELS x ARMS x sweep(fanout_k, burst, prefix, A2 cap K, A3 tau) x reps

Each cell:
  1. generates a seeded trace (same seed across arms of the cell -> identical
     arrivals, the paired-comparison precondition),
  2. runs the replay client against a real Dynamo+SGLang endpoint (per model),
  3. writes a per-cell results JSONL and a manifest row.

--dry-run swaps in the MockFrontend so the whole matrix executes offline with no
GPU, to validate the sweep/naming/manifest wiring before spending box time.

IMPORTANT — model ids: ``served_model_name`` is the string sent in the OpenAI
``model`` field and MUST match what the Dynamo frontend serves. The two entries
below are placeholders from the plan; CONFIRM them on the box (they may differ
from these labels) before a real run. Per-model SLOs also differ — a 30B and a
27B decode at different rates — so tune ``itl_slo_ms`` per model.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
from dataclasses import asdict, dataclass, field, replace

from replay_client import (Arm, HttpFrontend, MockFrontend, ReplayEngine,
                           task_goodput, victim_tail, write_results)
from trace_gen import WorkloadConfig, generate


@dataclass
class ModelSpec:
    label: str                 # short id used in filenames
    served_model_name: str     # CONFIRM on the box — the OpenAI `model` field
    base_url: str = "http://localhost:8000"
    itl_slo_ms: float = 50.0   # tune per model
    ttft_slo_ms: float = 500.0


# --- EDIT served_model_name on the box to match the Dynamo `--model` launch -- #
# Backend: Dynamo + vLLM (both models have H100 vLLM recipes; FPM + KV-events
# native). itl_slo_ms=20 is the starting anchor; the true SLO is swept offline in
# analyze.py (--slo-grid) from the saved victim ITLs, so this value only sets the
# run-time goodput column, not the final verdict. Confirm/tune per model after
# the unloaded-ITL calibration (both are ~3B-active hybrid MoE, so a shared 20 ms
# should be fair; verify the two unloaded p50s are within ~20%).
MODELS = [
    # agg-h100-dspark recipe: --served-model-name is the base NVFP4 id; the
    # -DSpark variant is the speculative-decoding DRAFT (num_speculative_tokens=7),
    # not the served model. Confirm with: curl -s http://<frontend>/v1/models | jq
    ModelSpec(label="nemotron-3.5-lightning-30b-a3b",
              served_model_name="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
              itl_slo_ms=20.0),
    ModelSpec(label="qwen3.6-35b-a3b",
              served_model_name="Qwen/Qwen3.6-35B-A3B",
              itl_slo_ms=20.0),
]
# --------------------------------------------------------------------------- #


@dataclass
class Sweep:
    fanout_k: tuple = (2, 5, 10)
    burst: tuple = (1.0, 3.0, 8.0)
    prefix_blocks: tuple = (8,)
    a2_caps: tuple = (2, 4)          # per-task cap K for A2, swept to best
    a3_tau: tuple = (4, 8, 16)       # FPM num_decode_requests threshold for A3
    drain_ms: tuple = (50.0,)
    reps: int = 5
    base_seed: int = 1000


@dataclass
class Cell:
    model: str
    arm: str
    fanout_k: int
    burst: float
    prefix_blocks: int
    param: float          # A2 cap or A3 tau; 0 for A0/A1
    rep: int
    seed: int
    goodput: float = 0.0
    victim_p95_itl_ms: float = 0.0
    n_requests: int = 0

    def name(self) -> str:
        p = f"_p{self.param:g}" if self.arm in ("A2", "A3") else ""
        return (f"{self.model}__{self.arm}{p}__k{self.fanout_k}"
                f"_b{self.burst:g}_pf{self.prefix_blocks}_r{self.rep}")


def _arm_params(arm: Arm, sw: Sweep):
    """Yield the arm-specific swept parameter(s)."""
    if arm is Arm.A2:
        return list(sw.a2_caps)
    if arm is Arm.A3:
        return list(sw.a3_tau)
    return [0]  # A0/A1 have no swept gate parameter


def build_cells(models, sweep: Sweep) -> list[Cell]:
    cells = []
    for m in models:
        for arm in (Arm.A0, Arm.A1, Arm.A2, Arm.A3):
            for k, burst, pf, param, rep in itertools.product(
                    sweep.fanout_k, sweep.burst, sweep.prefix_blocks,
                    _arm_params(arm, sweep), range(sweep.reps)):
                # Same seed for every arm of a (workload, rep) so arrivals match
                # across arms — the paired-comparison precondition.
                seed = sweep.base_seed + hash((k, burst, pf, rep)) % 100000
                cells.append(Cell(model=m.label, arm=arm.value, fanout_k=k,
                                  burst=burst, prefix_blocks=pf, param=float(param),
                                  rep=rep, seed=seed))
    return cells


async def run_cell(cell: Cell, model: ModelSpec, sweep: Sweep, outdir: str,
                   dry_run: bool) -> Cell:
    cfg = WorkloadConfig(n_tasks=60 if dry_run else 200, fanout_k=cell.fanout_k,
                         burst_multiplier=cell.burst,
                         shared_prefix_blocks=cell.prefix_blocks,
                         agent_policy_class="agents", victim_policy_class="latency")
    rows = [json.loads(r.to_json()) for r in generate(cfg, cell.seed)]

    arm = Arm(cell.arm)
    frontend = (MockFrontend(alpha=0.08) if dry_run
                else HttpFrontend(model.base_url, model.served_model_name))

    # Real runs keep true arrival timing; dry-runs compress it so the whole
    # matrix executes offline in seconds.
    kwargs = {"clock_scale": 0.02 if dry_run else 1.0}
    load_source = None
    if arm is Arm.A2:
        kwargs["k"] = int(cell.param)
    elif arm is Arm.A3:
        load_source = _make_load_source(model, dry_run)
        kwargs["load_source"] = load_source
        kwargs["load_threshold"] = cell.param
        kwargs["drain_interval_ms"] = sweep.drain_ms[0]

    eng = ReplayEngine(rows, arm, model.served_model_name, frontend, **kwargs)
    if arm is Arm.A3 and dry_run:
        # Mock load source reads the engine's in-flight count.
        load_source.bind(eng)  # type: ignore[union-attr]

    records = await eng.run()
    if hasattr(frontend, "close"):
        await frontend.close()

    write_results(records, os.path.join(outdir, cell.name() + ".jsonl"))
    cell.n_requests = len(records)
    cell.goodput = round(task_goodput(records, model.itl_slo_ms), 4)
    v = victim_tail(records, 0.95)
    cell.victim_p95_itl_ms = round(v, 3) if v is not None else 0.0
    return cell


# Shared across A3 cells so we build one DistributedRuntime / FPM subscriber for
# the whole matrix, not one per cell.
_RUNTIME = None
_FPM_ENDPOINT = None
_LIVE_FPM_SOURCE = None


def _make_load_source(model: ModelSpec, dry_run: bool):
    if dry_run:
        from load_source import MockLoadSource

        class _Bindable(MockLoadSource):
            def __init__(self):
                super().__init__(lambda: self._e.inflight if hasattr(self, "_e") else 0)
            def bind(self, engine):
                self._e = engine
        return _Bindable()
    # Live: one FpmLoadSource, reused across all A3 cells.
    global _LIVE_FPM_SOURCE
    if _LIVE_FPM_SOURCE is None:
        from load_source import FpmLoadSource
        src = FpmLoadSource(_acquire_fpm_endpoint(model))
        src.start()
        _LIVE_FPM_SOURCE = src
    return _LIVE_FPM_SOURCE


def _acquire_fpm_endpoint(model: ModelSpec):
    """Build (once) the Dynamo Endpoint that anchors FPM discovery.

    ``FpmEventSubscriber`` auto-discovers publishers on the event plane, so this
    only needs a ``DistributedRuntime`` plus the namespace/component/endpoint the
    workers register under. This mirrors the shipped receiver
    ``dynamo.common.recv_forward_pass_metrics`` exactly.

    Configure via env to match your deployment (defaults are Dynamo's):
      DYN_DISCOVERY_BACKEND  (default "etcd")
      DYN_REQUEST_PLANE      (default "nats")
      DYN_NAMESPACE          (default "dynamo")
      DYN_FPM_COMPONENT      (default "backend")   # the worker component
      DYN_FPM_ENDPOINT       (default "generate")

    VERIFY FIRST on the box that FPM is flowing and that these names are right:
      python -m dynamo.common.recv_forward_pass_metrics --mode tracking
    then set the env vars here to whatever made that receiver see messages.
    """
    global _RUNTIME, _FPM_ENDPOINT
    if _FPM_ENDPOINT is not None:
        return _FPM_ENDPOINT
    import asyncio as _asyncio

    from dynamo.runtime import DistributedRuntime

    loop = _asyncio.get_running_loop()
    discovery = os.environ.get("DYN_DISCOVERY_BACKEND", "etcd")
    request_plane = os.environ.get("DYN_REQUEST_PLANE", "nats")
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    component = os.environ.get("DYN_FPM_COMPONENT", "backend")
    endpoint = os.environ.get("DYN_FPM_ENDPOINT", "generate")

    _RUNTIME = DistributedRuntime(loop, discovery, request_plane)
    _FPM_ENDPOINT = _RUNTIME.endpoint(f"{namespace}.{component}.{endpoint}")
    return _FPM_ENDPOINT


async def run_matrix(models, sweep: Sweep, outdir: str, dry_run: bool,
                     limit: int | None = None) -> list[Cell]:
    os.makedirs(outdir, exist_ok=True)
    cells = build_cells(models, sweep)
    if limit:
        cells = cells[:limit]
    done = []
    for i, cell in enumerate(cells):
        model = next(m for m in models if m.label == cell.model)
        res = await run_cell(cell, model, sweep, outdir, dry_run)
        done.append(res)
        print(f"[{i+1}/{len(cells)}] {res.name()}  goodput={res.goodput}  "
              f"victim_p95_itl={res.victim_p95_itl_ms}ms")
    _merge_manifest(outdir, done)
    return done


def _cell_key(d: dict) -> tuple:
    return (d["model"], d["arm"], d["fanout_k"], d["burst"],
            d["prefix_blocks"], d["param"], d["rep"])


def _merge_manifest(outdir: str, done: list) -> None:
    """Merge new cells into manifest.jsonl so per-model runs to the same outdir
    accumulate instead of overwriting (deploy one model, run, swap, run)."""
    path = os.path.join(outdir, "manifest.jsonl")
    rows: dict = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                d = json.loads(line)
                rows[_cell_key(d)] = d
    for c in done:
        d = asdict(c)
        rows[_cell_key(d)] = d
    with open(path, "w") as f:
        for d in rows.values():
            f.write(json.dumps(d) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic TAPER model x arm x sweep runner.")
    p.add_argument("--outdir", default="./results")
    p.add_argument("--dry-run", action="store_true",
                   help="MockFrontend, no GPU — validate matrix wiring offline.")
    p.add_argument("--limit", type=int, default=None, help="run only first N cells")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--models", default=None,
                   help="comma-separated model labels to run (default: all). "
                        "Use one label to run just the model currently deployed; "
                        "manifest.jsonl merges across runs to the same --outdir.")
    p.add_argument("--base-url", default=None,
                   help="override the frontend URL for all selected models "
                        "(e.g. http://localhost:8000)")
    a = p.parse_args()
    sweep = Sweep()
    if a.reps is not None:
        sweep.reps = a.reps

    models = MODELS
    if a.models:
        want = {s.strip() for s in a.models.split(",")}
        models = [m for m in MODELS if m.label in want]
        missing = want - {m.label for m in models}
        if missing:
            raise SystemExit(f"unknown model label(s): {sorted(missing)}; "
                             f"available: {[m.label for m in MODELS]}")
    if a.base_url:
        models = [replace(m, base_url=a.base_url) for m in models]

    cells = build_cells(models, sweep)
    print(f"matrix: {len(models)} model(s) x 4 arms x sweep = {len(cells)} cells "
          f"({'DRY RUN' if a.dry_run else 'LIVE'})")
    asyncio.run(run_matrix(models, sweep, a.outdir, a.dry_run, a.limit))


if __name__ == "__main__":
    main()
