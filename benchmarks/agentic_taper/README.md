<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic TAPER — P0 measurement harness

Client-side harness for the P0 experiment in *Agentic TAPER on Dynamo* (§3). It
measures whether agentic fan-out imposes a disproportionate externality on
co-batched interactive work (H1), and whether a task-aware permit gate built
only from signals Dynamo already emits recovers goodput and tail latency (H2).

No Dynamo code changes. The four arms run against **stock** Dynamo + vLLM; the
gate lives here in the replay client, because Dynamo's router owns queueing and
its release condition is not steerable by a live signal (brief §1, Appendix C).

## Arms

| Arm | Behavior | Where it's set |
|-----|----------|----------------|
| A0 | Fan-out suppressed — one request per task at a time | trace generator |
| A1 | Eager fan-out (today's default) | trace generator |
| A2 | Fixed per-task cap `K` | `PermitGate(STATIC_CAP, k=K)` |
| A3 | Task-aware gate on live FPM load | `PermitGate(FPM, load_threshold=τ)` |

A2 and A3 are the **same** gate with different admit predicates, so an A3 win is
attributable to adaptivity, not to a different code path.

## Components and status

| # (brief) | Module | Status | Needs GPU? |
|-----------|--------|--------|-----------|
| P0-2 | `trace_gen.py` — seeded agentic Mooncake JSONL | **done, unit-tested** | no |
| P0-4 | `permit_gate.py` — EAGER / STATIC_CAP / FPM gate | **done, unit-tested** | no |
| P0-1 | `load_source.py` — live FPM load signal for A3 | **done**; agg logic tested, network path box-only | bindings + engine |
| P0-3 | `replay_client.py` — async DAG-honoring replay | **done**; full pipeline tested via mock frontend | HTTP path: frontend |
| P0-8 | `experiment.py` — model × arm × sweep driver | **done**; wiring tested via `--dry-run` | live: frontend |
| P0-5/7 | `analyze.py` — cross-cell stats + decision rules | **done, unit-tested** | no |
| —    | `prompt_synth.py` — hash_ids → shared-prefix text | **done, unit-tested** | no |
| —    | `trace_adapter.py` — real-trace (Mode A) adapter | **done, unit-tested** | no |

All modules **import and unit-test on any machine** (stdlib + msgspec). The real
network paths — `HttpFrontend` (aiohttp → Dynamo OpenAI endpoint) and
`FpmLoadSource` (dynamo bindings) — are exercised only on the 8×H100 box; both
have offline stand-ins (`MockFrontend`, `MockLoadSource`) so the DAG, gate, and
sweep logic are fully tested here. 60 tests, `python3 -m unittest discover -s tests`.

### Mode A — validate against a real Claude Code trace
Replay a *real* agent's fan-out through the same four arms to check the verdict
holds (preserves the paired A0–A3 comparison — you replay one captured trace
across all arms). Pipeline reuses existing Dynamo tooling:
```
claude_trace_export → request_trace_to_mooncake --agentic → AgenticMooncakeRow JSONL
python3 trace_adapter.py --in agentic.jsonl --out replay.jsonl   # derive role/task/parent
# then feed replay.jsonl to the replay client / experiment matrix
```
`trace_adapter.py` derives `role`/`task_id`/`parent` from the branch/join graph
(root = has `branches`, join = has `wait_for`, branch = in a root's `branches`,
else single) and is idempotent on already-labeled `trace_gen` rows. Note: a
captured agent trace has no synthetic "victims" — its standalone turns become
`single` tasks; inject synthetic victims (or designate `single`s) for the
interactive SLO measurement.

### Analysis & decision (offline)
```bash
python3 analyze.py --outdir ./results     # prints per-model H1/H2 verdicts, writes summary.json
```
`analyze.py` compares arms **at matched load** (near the knee), not averaged over
the whole sweep — averaging saturates at easy loads and hides the effect. It
reports the A1 throughput-trap curve, paired same-seed charged externality
(A1−A0), and the H2 comparison of A3 vs best-tuned A2 on goodput and victim tail.

### Arms map onto one gate (one variable changes)
| Arm | Gate config |
|-----|-------------|
| A0 | `STATIC_CAP, k=1` — one request per task at a time (baseline) |
| A1 | `EAGER` — admit as soon as DAG deps clear |
| A2 | `STATIC_CAP, k=K` — fixed per-task cap |
| A3 | `FPM, load_threshold=τ` — task-aware on live `num_decode_requests` |

### Two models
`experiment.py`'s `MODELS` list runs the whole matrix per model. **Confirm
`served_model_name` on the box** — the two entries are placeholders and the
labels may not equal what the frontend serves. Tune `itl_slo_ms` per model
(a 30B and a 27B decode at different rates).

### Run the matrix
```bash
python3 experiment.py --dry-run --reps 1        # offline wiring check, no GPU
python3 experiment.py --outdir ./results        # live, on the box
```

## Run the tests (no GPU)

```bash
cd benchmarks/agentic_taper
python3 -m unittest discover -s tests -v
```

## Generate a trace

```bash
python3 trace_gen.py --out trace.jsonl --seed 0 --n-tasks 200 \
    --fanout-k 5 --shared-prefix-blocks 8 --burst-multiplier 1.0 \
    --agent-policy-class agents --victim-policy-class latency
```

Row schema matches `AgenticMooncakeRow` (`lib/data-gen/src/mooncake.rs`): shared
`hash_ids` prefix forces branch co-location via KV-overlap routing; `branches`
(on the root) and `wait_for` (on the join) express fan-out and join structurally;
`parent` carries task lineage for the replay client to stamp as
`x-dynamo-parent-session-id`.

## Arm → trace/gate mapping

* **A0**: generate with `--fanout-k 0` (or run only roots), so each task is one request.
* **A1**: generate with fan-out; replay client sends every request as soon as its
  DAG dependencies clear. No gate.
* **A2**: A1 trace; wrap dispatch in `PermitGate(Policy.STATIC_CAP, k=K)`.
* **A3**: A1 trace; wrap dispatch in `PermitGate(Policy.FPM, load_fn=fpm.num_decode_requests, load_threshold=τ)`.

## Interfaces for the GPU-side components

### `fpm_source.py` (P0-1)
Thin wrapper over the shipped `dynamo.llm.FpmEventSubscriber` (verified present:
`lib/bindings/python/src/dynamo/_core.pyi:1349`). No new Rust needed.

```python
class FpmSource:
    def __init__(self, endpoint): ...      # dynamo Endpoint
    def start(self) -> None: ...           # calls subscriber.start_tracking()
    def num_decode_requests(self) -> int:  # summed across (worker, dp_rank); the A3 scalar
    def kv_frac(self) -> float:            # kv_used_blocks / kv_total_blocks fallback
    def freshness_ms(self) -> float:       # age of newest sample; report alongside results
```
Decode the msgspec payload for `num_decode_requests` / `sum_decode_kv_tokens`
(`components/src/dynamo/common/forward_pass_metrics.py`). **Measure
`freshness_ms` under load first** — if it is ~100 ms not ~ms, τ must absorb the
lag and A3's ceiling drops (brief Appendix E risk).

### `replay_client.py` (P0-3)
Async, open-loop. Honors each row's `timestamp` and `wait_for`; issues one HTTP
request per row to the Dynamo OpenAI endpoint with headers
`x-dynamo-session-id = session_id` and, when `parent` is set,
`x-dynamo-parent-session-id = parent`. For A2/A3 every request passes through
`PermitGate.submit`; `send` performs the dispatch, and the response callback
calls `gate.on_complete(task_id)`. A timer calls `gate.drain()` at the drain
interval (sweep {20,50,100} ms). Marks the victim's request and the aggressor's
root `is_protected=True`.

**Precondition check** (Appendix E): after a warm run, assert siblings actually
co-batched on one worker. If KV-overlap routing spread them, there is no shared
decode step and the experiment measures nothing — fail loudly.

### `analyze.py` (P0-5, P0-7)
Consumes exported OTel spans (`ttft_ms`, `avg_itl_ms`, `itl_p50/p99/max_ms` from
`lib/backend-common/src/adapter.rs`). Joins request → session → task via
`parent`. Emits: task-level goodput (fraction within SLO), victim p50/p95/p99
ITL & TTFT, and **charged externality** as a *paired* per-request A1−A0 (and
A3−A0) difference keyed on the fixed seed. Applies the pre-registered decision
rules (§ below).

## Pre-registered decision rules (write the thresholds before running)

* **H1a (primary)** confirmed if A1's goodput-vs-load curve has a knee — goodput
  turns down while throughput keeps rising — and A0 shows none in range.
* **H1 falsified** if goodput tracks throughput monotonically (no knee) *or*
  victim tail latency is paired-flat between A0 and A1 → report the negative, stop.
* **H2** confirmed if A3 beats best-K A2 on goodput with a paired CI excluding
  zero, at ≥ comparable throughput.
* **H2 falsified** if A3 ≤ best-K A2 → the value isn't in dynamic gating; report, stop.

Run each (arm, sweep-point) ≥5× with a fixed per-config seed; report median and
IQR, not means — the claim is about tails.

## Derisk before building the GPU-side pieces

1. **Co-location spike** — ~20 hand-made requests sharing a prefix; confirm they
   co-batch on one worker. If not, nothing downstream matters.
2. **FPM freshness spike** — `FpmSource`; measure real staleness under load.
3. **Crude A1-vs-A0 spike** — no gate, no sweeps; does *any* tail degradation
   appear? If not at achievable scale, raise K and prefix depth before investing
   in the full harness.
