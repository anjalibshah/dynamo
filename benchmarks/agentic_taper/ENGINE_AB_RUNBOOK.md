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

> **Anchor mistake (found 2026-10-15, after e1):** 4.95 ms is the GLM-4.7-Flash
> anchor (NEXT_STEPS 7.3e). MiniMax-M2's unloaded interactive ITL p50 is
> 6.16 ms (7.3j), which s1/s2 were scored with and which the engine's 18.48 ms
> target is built from. e1 is reported as registered (4.95, a stricter
> deadline) and also at 6.16 for comparison with s1/s2. R3-R5 do not depend
> on the anchor. e2 below uses 6.16, fixed before any e2 data.

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

About 4 hours of 8 H200 GPUs on gpu-04 (host h200-r2-1): a 30-minute smoke
test, then three rounds of about an hour (two 4-GPU slots per round). It must
be the node that ran s1/s2: `/scratch` is node-local, and the MiniMax weights
(`/scratch/$USER/hf_local`), the container image and Harbor live there.

### Setup

```bash
salloc --nodelist=gpu-04 --gres=gpu:8 --time=6:00:00
hostname; ls /scratch/$USER/hf_local >/dev/null && echo weights ok
cd /scratch/$USER/src/dynamo && git fetch origin && git status -sb | head -1
# on feat/agentic-taper-rebased: git pull; otherwise add a worktree:
#   git worktree add /scratch/$USER/src/dynamo-rebased origin/feat/agentic-taper-rebased
source ~/mm_env.sh
export REPO=<that checkout> PY=/scratch/$USER/src/dynamo/.venv/bin/python3
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

## Trade-off experiment (e2)

e1 failed the rule at 24 agents on cost (task p90 x1.59, throughput x0.87,
both beyond the s1/s2 kv spread) while interactive reached 0.94 against 0.13
for existing priority. Following "R3 or R4 fails", e2 measures the trade-off
curve at 24 agents instead of re-running e1. It is a new experiment with its
own rule, fixed before the runs; e1's verdict stands whatever e2 shows.

### Arms (24 agents, same setup and backend code as e1)

| Arm | Change from `cb` | Expected |
|---|---|---|
| `cbt1.5` | interactive per-token target x1.5 (`DYN_TAPER_TAU_INTERACTIVE_MS`) | fewer agent decodes skipped; interactive held to a looser deadline |
| `cbt2` | target x2 | further along the curve |
| `cbk2` | `DYN_TAPER_K=2`: every request decodes at least every other step | caps skipping at 50% per request; interactive may drop |
| `cb`, `kv` | none (replicates of e1) | pooled with e1: two runs each, and a measured cb-vs-cb spread |

`prio` is not re-run; e1's `prio-n24` is the existing-priority reference.

### Rule

Scored by `tradeoff_report.py` over e1 and e2 pooled: each metric is the mean
over an arm's runs, against the mean of all `kv` runs at 24 agents. An arm is
an acceptable operating point if:

| | Criterion |
|---|---|
| A1 | interactive share meeting the arm's own deadline (500 ms + 3 x m x 6.16 ms per token, m the target multiplier; 6.16 ms is MiniMax-M2's anchor) >= 0.80 |
| A2 | that share minus `prio`'s at the same deadline >= 0.30 |
| A3 | task p90 / kv <= 1.40 |
| A4 | throughput / kv >= 0.90 |
| A5 | engine health as R5 |

A looser target is only worth offering if the interactive traffic still
meets it (A1) and existing priority does not already meet it (A2).

### Run (about 3 hours on 8 GPUs, gpu-04)

Engine code must be the same as e1 (`taper_engine.py` unchanged since
7c47b3c140); only the benchmark scripts changed.

```bash
export N_TASKS=64
for r in "kv:24 cbt1.5:24" "cbt2:24 cbk2:24" "cb:24 kv:24"; do
  ( set -o pipefail; ./sweep_parallel.sh e2 "$r" 2>&1 | tee -a /scratch/$USER/engine_ab.log ) || break
done
python3 tradeoff_report.py --sweep e1 --sweep e2 --suite=-mm --n-tasks 64 \
    --root $REPO/harbor_ab-mm --agents 24
```

`kv` runs in slot 0 in round 1 and slot 1 in round 3, so a GPU-half effect
averages out of the baseline.

**On 4 GPUs** (one slot, about 6 hours): one arm per round, stock first and
last so drift over the session averages out of the baseline. The second
stock run uses sweep tag `e2b` (a run is skipped if its name already exists).

```bash
for r in "e2 kv:24" "e2 cbt1.5:24" "e2 cbt2:24" "e2 cbk2:24" "e2 cb:24" "e2b kv:24"; do
  ( set -o pipefail; ./sweep_parallel.sh ${r%% *} "${r#* }" 2>&1 | tee -a /scratch/$USER/engine_ab.log ) || break
done
python3 tradeoff_report.py --sweep e1 --sweep e2 --sweep e2b --suite=-mm --n-tasks 64 \
    --root $REPO/harbor_ab-mm --agents 24
```

### What each outcome means

- **One or more acceptable points:** the feature ships default off with the
  best such setting as the documented starting point, and the curve in the DEP.
- **None, but the `cb` replicate lands within the limits:** e1's 24-agent cost
  was mostly noise; report both runs and the measured spread.
- **None:** at 24 agents no tested setting buys interactive latency at an
  acceptable cost; report the curve and the 28-agent pass, and keep the
  feature as an experimental opt-in or stop there.

## Deadline-aware target (e3)

e2 found no acceptable operating point: a looser fixed target lets existing
priority meet it too, and the tight one costs task p90 1.3-1.6x. The
simulator (`decode_sim.py`, s1/s2, 4 runs) then tested BROS's residual-time
budget: each interactive request's target is its remaining end-to-end
deadline over its remaining `max_tokens`, times a margin. `slack0.9` cut task
p90 to 0.93x class-blind (0.90-0.98) at 80-90% interactive, unchanged under
10% step-time noise, with T80 1.06x stock vs 1.10x. Dynamo's per-request
`osl` hint as a fill order added nothing (0.92-0.95); a session-level
predictor would (0.83) but none exists at runtime. e3 tests `slack0.9` on GPU.
It is a new experiment; e1 and e2 verdicts stand.

### Arms (24 agents, 4 GPUs, one at a time; same image and backend code)

`cbs0.9` = `cb` plus `DYN_TAPER_DEADLINE=500,18.48`, `DYN_TAPER_SLACK_MARGIN=0.9`.
Two `cbs0.9` runs, plus one more `kv` and `cb` each so the pooled baselines
include this session.

### Rule (fixed 2026-10-19, before any e3 data)

Scored by `tradeoff_report.py --sweep e1 --sweep e2 --sweep e2b --sweep e3 --sweep e3b
--max-task-p90-ratio 1.20` (6.16 ms anchor; means over each arm's runs; kv = all
five kv runs). `cbs0.9` is an acceptable operating point if A1 (interactive
>= 0.80), A2 (minus prio >= 0.30), A3 (**task p90 / kv <= 1.20**), A4
(throughput / kv >= 0.90) and A5 (engine health) hold. Secondary, reported
either way: `cbs0.9` task p90 / `cb` task p90 (the simulator predicts about 0.93).

A4 caveat recorded in advance: kv throughput spans 208-252 tasks/h across
three identical runs, so A4 with two runs per arm is weak evidence; A3 is the
primary cost criterion.

### Run

```bash
export N_TASKS=64 PORT_BASE=200   # plus PY, GPUS_PER_SLOT=4, VICTIM_PRIORITY=1, MOUNT_VLLM=1, GPU_IDS
for r in "e3 kv:24" "e3 cbs0.9:24" "e3 cb:24" "e3b cbs0.9:24" "e3b kv:24"; do
  ( set -o pipefail; ./sweep_parallel.sh ${r%% *} "${r#* }" 2>&1 | tee -a /scratch/$USER/engine_ab.log ) || break
done
python3 tradeoff_report.py --sweep e1 --sweep e2 --sweep e2b --sweep e3 --sweep e3b \
    --suite=-mm --n-tasks 64 --root $REPO/harbor_ab-mm --agents 24 --max-task-p90-ratio 1.20
```

### What each outcome means

- **cbs0.9 acceptable:** the scheduler PR (default off, deadline-aware target)
  with these numbers; the DEP proposes it.
- **Not acceptable, but task p90 clearly below cb's:** an improvement over
  HyGen-style regulation that still costs too much at 24 agents; report it and
  keep the feature experimental.
- **No better than cb on GPU:** the simulator's gain does not transfer; stop
  the scheduler PR, ship the benchmark PR.

### e3 result (2026-10-19)

`cbs0.9` acceptable: interactive 0.931 (2 runs) vs kv 0.31 (5 runs), task p90 /
kv 0.94, throughput / kv 0.94, skip share 20% (cb: 1.39, 0.77, 35%). Against
the four kv runs before e3b only: 1.01 and 0.91. Secondary: task p90 / cb 0.67
(simulator: 0.93). Full table in NEXT_STEPS.md 7.3s.

## Heavier load (e4): does the deadline budget hold at 28 agents?

e3 tested one load. The budget's slack shrinks as load grows (each step is
slower, so less of every interactive request's deadline is left to spend), so
both the agents' cost and the interactive attainment can change with load. e1
at 28 agents is the reference: there class-blind cost less than at 24
(throughput 0.95, task p90 1.24x).

### Arms (28 agents, 4 GPUs, one at a time; same code as e3)

`kv` and `cbs0.9`, two runs each, in ABBA order so each `cbs0.9` run has a
`kv` run next to it.

### Rule (fixed 2026-10-19, before any e4 data)

Scored by `tradeoff_report.py --sweep e1 --sweep e4 --sweep e4b --agents 28
--max-task-p90-ratio 1.20` (6.16 ms anchor; kv = e1's and e4's kv runs at 28;
prio from e1). Same A1-A5 as e3. Reported either way: each `cbs0.9` run
against the `kv` run of the same sweep, and against e1's `cb` at 28.

### What each outcome means (none of them stops the default-off PR)

- **Acceptable:** the PR claims the result at 24 and 28 agents.
- **Fails A3 or A4 but costs less than e1's cb at 28:** the PR states that the
  agent cost grows with overload, with both numbers.
- **Fails A1 (interactive < 0.80):** the budget runs out under heavier load;
  the PR states the load range where it holds, and the margin is the knob to
  revisit.

### Run

```bash
export N_TASKS=64 PORT_BASE=200   # plus PY, GPUS_PER_SLOT=4, VICTIM_PRIORITY=1, MOUNT_VLLM=1, GPU_IDS
for r in "e4 kv:28" "e4 cbs0.9:28" "e4b cbs0.9:28" "e4b kv:28"; do
  ( set -o pipefail; ./sweep_parallel.sh ${r%% *} "${r#* }" 2>&1 | tee -a /scratch/$USER/engine_ab.log ) || break
done
python3 tradeoff_report.py --sweep e1 --sweep e4 --sweep e4b \
    --suite=-mm --n-tasks 64 --root $REPO/harbor_ab-mm --agents 28 --max-task-p90-ratio 1.20
```
