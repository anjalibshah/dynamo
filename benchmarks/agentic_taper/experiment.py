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

import statistics

from calibration import LoadSampler, summarize_calibration
from latency_model import LatencyModel
from replay_client import (Arm, HttpFrontend, MockFrontend, ReplayEngine,
                           apply_server_metrics, concentration_report,
                           discover_worker_ids, parse_frontend_metrics,
                           task_goodput, victim_tail, write_results)
from trace_gen import WorkloadConfig, generate


def _victim_mean_itl(records) -> float:
    """Central victim ITL (mean over victim per-request mean ITLs) — the y for
    calibration points."""
    vals = [statistics.mean(r.itls_ms) for r in records
            if r.role == "victim" and r.itls_ms]
    return round(statistics.mean(vals), 3) if vals else 0.0


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


def build_cells(models, sweep: Sweep, arms=None) -> list[Cell]:
    # arms: optional iterable of Arm to include (default all). Lets a run target
    # just the arms a hypothesis needs (H1 -> A0,A1; H2 -> A0,A2,A3) without doing
    # cell-offset math against --limit.
    selected = tuple(arms) if arms else (Arm.A0, Arm.A1, Arm.A2, Arm.A3)
    cells = []
    for m in models:
        for arm in selected:
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
                   dry_run: bool, n_tasks: int, max_concurrency: int,
                   frontend_log: str | None = None,
                   words_per_block: int = 32, latency_model=None,
                   worker_ids=None, pin_mode: str = "auto",
                   load_source_kind: str = "fpm") -> Cell:
    arm = Arm(cell.arm)
    # A0 is the distributed baseline (siblings spread across workers); A1/A2/A3 use
    # the concentrated trace. Only hash-id placement differs from A1 — matched load.
    cfg = WorkloadConfig(n_tasks=60 if dry_run else n_tasks, fanout_k=cell.fanout_k,
                         burst_multiplier=cell.burst,
                         shared_prefix_blocks=cell.prefix_blocks,
                         distribute_siblings=(arm is Arm.A0),
                         agent_policy_class="agents", victim_policy_class="latency")
    rows = [json.loads(r.to_json()) for r in generate(cfg, cell.seed)]
    frontend = (MockFrontend(alpha=0.08) if dry_run
                else HttpFrontend(model.base_url, model.served_model_name,
                                  max_conns=max_concurrency))

    # Real runs keep true arrival timing; dry-runs compress it so the whole
    # matrix executes offline in seconds.
    kwargs = {"clock_scale": 0.02 if dry_run else 1.0,
              "max_concurrency": max_concurrency,
              "words_per_block": words_per_block,
              "worker_ids": worker_ids, "pin_mode": pin_mode}
    load_source = None
    if arm is Arm.A2:
        kwargs["k"] = int(cell.param)
    elif arm is Arm.A3:
        load_source = _make_load_source(model, dry_run, load_source_kind)
        kwargs["load_source"] = load_source
        kwargs["drain_interval_ms"] = sweep.drain_ms[0]
        if latency_model is not None:
            # Budget rule: admit on projected T(S) <= SLO. The swept cell.param
            # (tau) is ignored — the boundary comes from the model + SLO.
            kwargs["latency_model"] = latency_model
            kwargs["slo_ms"] = model.itl_slo_ms
        else:
            kwargs["load_threshold"] = cell.param   # legacy raw-threshold sweep

    eng = ReplayEngine(rows, arm, model.served_model_name, frontend, **kwargs)
    if arm is Arm.A3 and (dry_run or load_source_kind == "client"):
        # Mock/client load source reads the engine's in-flight count.
        load_source.bind(eng)  # type: ignore[union-attr]

    records = await eng.run()
    if hasattr(frontend, "close"):
        await frontend.close()

    # Prefer server-measured ITL/TTFT over client SSE timing (see replay_client).
    # The client loop can't time tokens accurately under co-batch load; the
    # frontend metrics log is server truth. Give the log a moment to flush first.
    # Fail LOUDLY if --frontend-log was requested but unusable — silently falling
    # back to client timing produces plausible-but-wrong numbers (a whole run's
    # worth), which is exactly the trap that wasted runs during bring-up.
    if frontend_log and not dry_run:
        if not os.path.exists(frontend_log):
            raise SystemExit(
                f"--frontend-log {frontend_log} does not exist. Launch the frontend "
                f"redirected to it: `python3 -m dynamo.frontend … > {frontend_log} 2>&1 &` "
                f"(same container as this run — /tmp is not shared across containers).")
        await asyncio.sleep(1.0)
        matched = apply_server_metrics(records, parse_frontend_metrics(frontend_log))
        print(f"    server-metrics matched {matched}/{len(records)} requests")
        if arm in (Arm.A0, Arm.A1):
            cr = concentration_report(records)
            if cr["tasks"]:
                want = "concentrated ~1" if arm is Arm.A1 else "distributed ~fanout_k"
                print(f"    concentration: {cr['mean_distinct_workers']} distinct "
                      f"workers/task (expect {want})")
        if matched == 0:
            raise SystemExit(
                f"--frontend-log matched 0/{len(records)} requests — the log exists but "
                f"nothing joined (wrong frontend? stale log? format change?). Refusing to "
                f"report client-timed ITL. Check `grep 'request completed' {frontend_log}`.")

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


def _make_load_source(model: ModelSpec, dry_run: bool, kind: str = "fpm"):
    # "client": use the replay client's own in-flight count as the load signal
    # instead of server FPM. Valid when the client is the sole traffic source and
    # everything lands on one worker (--pin-mode single), where client inflight ≈
    # that worker's decode load. Lets A3 run without a verified FPM event path.
    if dry_run or kind == "client":
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
                     limit: int | None = None, n_tasks: int = 120,
                     max_concurrency: int = 64,
                     frontend_log: str | None = None,
                     words_per_block: int = 32, arms=None,
                     latency_model=None, worker_ids=None,
                     pin_mode: str = "auto",
                     load_source_kind: str = "fpm") -> list[Cell]:
    os.makedirs(outdir, exist_ok=True)
    # Worker pinning: the KV router load-balances shared-prefix siblings on this
    # model, so A1 won't concentrate via routing. Pin each request to a named
    # decode worker instead. Discover the pool from the frontend log unless the
    # caller passed an explicit list. No workers found on a live run => pinning is
    # off and A0/A1 collapse, so fail LOUDLY rather than emit a worthless matrix.
    if worker_ids is None and frontend_log and not dry_run:
        worker_ids = discover_worker_ids(frontend_log)
    if not dry_run and not worker_ids:
        raise SystemExit(
            f"No decode worker ids discovered from --frontend-log ({frontend_log}). "
            f"Pinning would be OFF and the load-aware router will spread A1 exactly "
            f"like A0 (A0==A1, invalid). Ensure the frontend log has the startup "
            f"'Adding worker … worker_id: N' lines, or pass --worker-ids.")
    if worker_ids:
        if pin_mode == "single":
            print(f"    pin-mode=single: co-locating ALL requests on worker "
                  f"{worker_ids[0]} (shared decode step; {len(worker_ids)-1} workers idle)")
        else:
            print(f"    pinning across {len(worker_ids)} workers: {worker_ids}")
    cells = build_cells(models, sweep, arms)
    if limit:
        cells = cells[:limit]
    done = []
    for i, cell in enumerate(cells):
        model = next(m for m in models if m.label == cell.model)
        res = await run_cell(cell, model, sweep, outdir, dry_run,
                             n_tasks, max_concurrency, frontend_log,
                             words_per_block, latency_model, worker_ids, pin_mode,
                             load_source_kind)
        done.append(res)
        print(f"[{i+1}/{len(cells)}] {res.name()}  goodput={res.goodput}  "
              f"victim_p95_itl={res.victim_p95_itl_ms}ms")
    _merge_manifest(outdir, done)
    return done


async def run_calibration(model: ModelSpec, sweep: Sweep, outdir: str,
                          dry_run: bool, n_tasks: int, frontend_log: str | None,
                          words_per_block: int, levels: list[int]) -> dict:
    """Sweep offered load (eager A1) across concurrency ``levels``; for each,
    sample the live decode load and pair it with the victim ITL. Fit ITL(load)
    and write ``calibration.json`` — the model + SLO-derived A3 admit boundary.
    """
    os.makedirs(outdir, exist_ok=True)
    load_source = _make_load_source(model, dry_run)
    points = []
    print(f"calibration: {model.label} @ SLO {model.itl_slo_ms} ms, "
          f"levels={levels} ({'DRY RUN' if dry_run else 'LIVE'})")
    for lvl in levels:
        cfg = WorkloadConfig(n_tasks=40 if dry_run else n_tasks, fanout_k=5,
                             burst_multiplier=8.0, shared_prefix_blocks=8,
                             agent_policy_class="agents", victim_policy_class="latency")
        rows = [json.loads(r.to_json()) for r in generate(cfg, sweep.base_seed + lvl)]
        frontend = (MockFrontend(alpha=0.08) if dry_run
                    else HttpFrontend(model.base_url, model.served_model_name, max_conns=lvl))
        eng = ReplayEngine(rows, Arm.A1, model.served_model_name, frontend,
                           clock_scale=0.02 if dry_run else 1.0, max_concurrency=lvl,
                           words_per_block=words_per_block)
        if dry_run and hasattr(load_source, "bind"):
            load_source.bind(eng)
        sampler = LoadSampler(load_source.num_decode_requests,
                              interval_ms=50 if dry_run else 200)
        sampler.start()
        records = await eng.run()
        await sampler.stop()
        if hasattr(frontend, "close"):
            await frontend.close()
        if frontend_log and not dry_run and os.path.exists(frontend_log):
            await asyncio.sleep(1.0)
            apply_server_metrics(records, parse_frontend_metrics(frontend_log))
        load, vitl = sampler.mean(), _victim_mean_itl(records)
        points.append((load, vitl))
        print(f"  conc={lvl:<4} decode_load={load:6.2f}  victim_ITL={vitl:7.2f} ms")

    report = summarize_calibration(points, model.itl_slo_ms)
    path = os.path.join(outdir, "calibration.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    m = report["model"]
    print(f"\nfit: ITL(load) = {m['t0_ms']:.2f} + {m['beta_ms_per_req']:.3f}·load  (n={m['n_points']})")
    print(f"SLO {model.itl_slo_ms} ms  ->  A3 admit boundary tau* = {report['tau_star']:.2f} decode reqs")
    print(f"unloaded floor {report['unloaded_itl_ms']} ms (SLO must exceed this to be feasible)")
    print(f"wrote {path}  — pass it to the H2 run as --latency-model {path}")
    return report


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
    p.add_argument("--n-tasks", type=int, default=120,
                   help="tasks per live cell (default 120; was 200 — lower keeps "
                        "the client loop paced with generation). Ignored in --dry-run.")
    p.add_argument("--max-concurrency", type=int, default=64,
                   help="cap on concurrent HTTP streams read by the client, and the "
                        "aiohttp connection-pool limit. Prevents event-loop "
                        "saturation that otherwise inflates measured ITL.")
    p.add_argument("--frontend-log", default=None,
                   help="path to the dynamo.frontend log (run it with "
                        "`> /tmp/frontend.log 2>&1`). When set, per-request ITL/TTFT "
                        "are read from the server's metrics lines instead of client "
                        "SSE timing — the correct measurement under load.")
    p.add_argument("--words-per-block", type=int, default=32,
                   help="synthetic words per KV block in the prompt (default 32). "
                        "Each word tokenizes to ~4 tokens, so 32 ≈ 128 tokens/block "
                        "— enough to fill KV blocks for co-location without the "
                        "16k-token prompts (words_per_block=400) that overloaded the "
                        "server. Tune live; no re-scp needed.")
    p.add_argument("--arms", default=None,
                   help="comma-separated arms to run (default all). Targets a "
                        "hypothesis without cell-offset math: H1 -> 'A0,A1', "
                        "H2 -> 'A0,A2,A3'. Combine with --reps; skip --limit.")
    p.add_argument("--itl-slo-ms", type=float, default=None,
                   help="override the ITL SLO budget for goodput AND the A3 budget "
                        "rule. Set from calibration's unloaded floor + an interactive "
                        "margin (defaults to each model's built-in itl_slo_ms).")
    p.add_argument("--calibrate", action="store_true",
                   help="calibration mode: sweep offered load, fit ITL(load), write "
                        "calibration.json (the model A3's budget rule needs). Runs "
                        "one model; use --calib-concurrency for the load levels.")
    p.add_argument("--calib-concurrency", default="8,16,32,64,128",
                   help="comma-separated concurrency levels for --calibrate.")
    p.add_argument("--latency-model", default=None,
                   help="path to a calibration.json. When set, A3 uses the budget "
                        "rule (projected T(S) <= SLO) instead of the swept tau — the "
                        "faithful TAPER admission rule.")
    p.add_argument("--models", default=None,
                   help="comma-separated model labels to run (default: all). "
                        "Use one label to run just the model currently deployed; "
                        "manifest.jsonl merges across runs to the same --outdir.")
    p.add_argument("--base-url", default=None,
                   help="override the frontend URL for all selected models "
                        "(e.g. http://localhost:8000)")
    p.add_argument("--worker-ids", default=None,
                   help="comma-separated decode-worker instance ids to pin to "
                        "(A0 spreads a task's branches across them, A1/A2/A3 "
                        "concentrate). Default: auto-discover from --frontend-log. "
                        "Pinning is required because the KV router load-balances "
                        "shared-prefix siblings on this model and won't co-locate them.")
    p.add_argument("--load-source", default="fpm", choices=("fpm", "client"),
                   dest="load_source_kind",
                   help="A3 load signal. fpm (default): server ForwardPassMetrics "
                        "over the event plane (must be verified flowing). client: "
                        "the replay client's in-flight count — valid with --pin-mode "
                        "single (client is sole traffic, one worker), and avoids the "
                        "FPM dependency.")
    p.add_argument("--pin-mode", default="auto", choices=("auto", "single"),
                   help="auto (default): A0 spreads branches, A1/A2/A3 concentrate "
                        "per task. single: co-locate ALL requests on one worker — "
                        "the faithful shared-decode-step test (sweep K within A1 for "
                        "the externality; A1 vs A2/A3 for the gate). Use fewer "
                        "--n-tasks in single mode (one worker carries the whole load).")
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
    if a.itl_slo_ms is not None:
        models = [replace(m, itl_slo_ms=a.itl_slo_ms) for m in models]

    # Calibration mode: one model, sweep load, fit ITL(load), write the model.
    if a.calibrate:
        if len(models) != 1:
            raise SystemExit("--calibrate runs ONE model; pass --models <label>")
        levels = [int(s) for s in a.calib_concurrency.split(",")]
        asyncio.run(run_calibration(models[0], Sweep(), a.outdir, a.dry_run,
                                    a.n_tasks, a.frontend_log, a.words_per_block, levels))
        return

    latency_model = None
    if a.latency_model:
        with open(a.latency_model) as f:
            latency_model = LatencyModel.from_dict(json.load(f)["model"])

    arms = None
    if a.arms:
        want = [s.strip() for s in a.arms.split(",")]
        try:
            arms = [Arm(s) for s in want]
        except ValueError:
            raise SystemExit(f"unknown arm(s) in {want}; valid: A0,A1,A2,A3")

    worker_ids = None
    if a.worker_ids:
        worker_ids = [int(s) for s in a.worker_ids.split(",") if s.strip()]

    cells = build_cells(models, sweep, arms)
    armstr = ",".join(x.value for x in arms) if arms else "A0,A1,A2,A3"
    print(f"matrix: {len(models)} model(s) x [{armstr}] x sweep = {len(cells)} cells "
          f"({'DRY RUN' if a.dry_run else 'LIVE'})")
    asyncio.run(run_matrix(models, sweep, a.outdir, a.dry_run, a.limit,
                           a.n_tasks, a.max_concurrency, a.frontend_log,
                           a.words_per_block, arms, latency_model, worker_ids,
                           a.pin_mode, a.load_source_kind))


if __name__ == "__main__":
    main()
