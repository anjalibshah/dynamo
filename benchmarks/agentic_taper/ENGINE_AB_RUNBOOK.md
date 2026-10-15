<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Engine-level A/B: class-blind decode regulation vs stock and vLLM priority

One GPU session that decides whether class-blind per-step decode regulation
(`TaperScheduler`, `DYN_TAPER_ENGINE=class_blind`) goes forward as a Dynamo
feature. The rule below is fixed before the runs; do not change thresholds
after seeing results.

## Question

Under mixed load (Claude Code agents on SWE-bench Verified plus an
interactive chat client), interactive requests miss their deadline because
every decode step is slow, not because they wait in a queue. Dynamo's
existing priority (`nvext.agent_hints.priority`) orders queues, preemption and
eviction; it does not change which running requests decode in a step. Does
class-blind regulation give the interactive traffic its deadline, at an
acceptable cost to the agents, where the existing priority does not?

## Arms

All arms: MiniMax-M2 TP4 on H200, same image, same backend code
(`MOUNT_VLLM=1`), same workload as sweeps s1/s2 (64 tasks, natural agent
behaviour, interactive client at 2 s mean inter-arrival), and the interactive
requests carry `nvext.agent_hints.priority = 1` (`VICTIM_PRIORITY=1`) so
request bodies are identical across arms.

| Arm | What runs |
|---|---|
| `kv` | Stock: KV router, vLLM default (FCFS) scheduling |
| `prio` | Dynamo's existing priority feature: vLLM `--scheduling-policy priority` |
| `cb` | Class-blind regulation: requests with priority >= 1 get the ITL target, the rest are regulated (`DYN_TAPER_PRIORITY_MIN=1`) |

Agent counts 24 and 28 (the overloaded points of s1/s2).

## Pre-registered rule

Scored by `engine_ab_report.py` with `sweep_report.py`'s definitions
(window: first agent start to the 52nd of 64 task completions, T80;
interactive: end-to-end deadline 500 ms + 3 x 4.95 ms x (tokens - 1)).

| | Criterion | Why this threshold |
|---|---|---|
| R1 | cb interactive >= 0.80 | Stock is 0.12-0.17 (s1/s2); simulation predicts 0.92-0.97 |
| R2 | cb interactive - prio interactive >= 0.30 | The feature must add what existing priority does not |
| R3 | cb task p90 / kv task p90 <= 1.40 | Simulation predicts 1.24-1.40 |
| R4 | cb throughput / kv throughput >= 0.90 | Simulation predicts T80 +5-10% |
| R5 | cb did not fail open; step-time error p50 <= 15%; decision time p99 <= 1 ms | The mechanism worked as designed |

**PASS** if R1-R5 hold at both 24 and 28 agents.

Run-to-run noise: if R3 or R4 misses by less than the kv-vs-kv spread between
s1 and s2 at that agent count (printed by `--noise-sweeps`), run one more
`kv` + `cb` round at that count and decide on the mean of the two. Nothing
else is re-run.

## Session

About 4 hours of 8 H200 GPUs on h200-2: a 30-minute smoke test, then three
rounds of about an hour (two 4-GPU slots per round).

### Setup

```bash
salloc --nodelist=gpu-02 --gres=gpu:8 --time=6:00:00
cd /scratch/$USER/src/dynamo-rebased && git pull && git log -1 --oneline
source ~/mm_env.sh
export REPO=/scratch/$USER/src/dynamo-rebased PY=/scratch/$USER/src/dynamo/.venv/bin/python3
export GPUS_PER_SLOT=4 VICTIM_PRIORITY=1 MOUNT_VLLM=1
cd $REPO/benchmarks/agentic_taper
```

### Step 0: smoke test (plumbing only, no overload)

```bash
N_TASKS=2 ./sweep_parallel.sh e0 "cb:2 kv:2" &
# while it runs (cb is slot 0, so its worker serves /metrics on port 8181):
watch -n 30 "curl -s localhost:8181/metrics | grep '^dynamo_component_decode_regulation'"
# after it finishes:
grep -m1 'taper.engine mode=' $REPO/harbor_ab-mm/swe0-cb-n2/stack.log
grep 'taper.engine steps=' $REPO/harbor_ab-mm/swe0-cb-n2/stack.log | tail -2
```

Go on only if: both stacks became ready, the cb log shows
`taper.engine mode=class_blind`, `/metrics` showed the
`dynamo_component_decode_regulation_*` series with `enabled` 1 and
`interactive_total` and `regulated_total` growing, the summary lines show both
`interactive=` and `regulated=` above 0 (the priority reaches the engine and
classifies both ways), `pred_err_p50` is under about 15%, and there is no
`taper.engine disabled`. If the cb stack
fails to start with an import error, the image predates the rebased branch:
rebuild it from this checkout (`podman_build.sh`) and repeat step 0.

### Steps 1-3: the A/B rounds

```bash
export N_TASKS=64
for r in "kv:24 cb:24" "cb:28 kv:28" "prio:24 prio:28"; do
  ( set -o pipefail; ./sweep_parallel.sh e1 "$r" 2>&1 | tee -a /scratch/$USER/engine_ab.log ) || break
done
```

Stock and cb at the same agent count run side by side, on swapped GPU halves
between the two counts.

### Score

```bash
python3 engine_ab_report.py --sweep e1 --suite=-mm --n-tasks 64 \
    --root $REPO/harbor_ab-mm \
    --noise-sweeps s1 s2 --noise-root /data/$USER/harbor_ab-mm
```

Also record the overhead line (`decide_p50/p99`) and the skipped share
(`skip_share`) from each cb run's last `taper.engine` log line.

## What each outcome means

- **PASS:** write the DEP (per-step decode regulation as a fourth priority
  layer, "decode", next to router queue, engine queue and KV cache) with these
  numbers, and open the PR for `TaperScheduler` behind `DYN_TAPER_ENGINE`
  (default off).
- **R1 fails:** regulation does not reach the deadline on real hardware.
  Compare `pred_err_p50` and `skip_share` against the simulation before
  concluding; a step-time model miss is fixable, a mechanism miss is not.
- **R2 fails:** existing priority already delivers most of it; there is no
  feature to add.
- **R3 or R4 fails (beyond noise):** the cost to agents is larger than
  simulated; report the trade-off curve rather than a default-on feature.
- **R5 fails:** a correctness or calibration problem; fix and re-run the
  smoke test before any conclusion.
