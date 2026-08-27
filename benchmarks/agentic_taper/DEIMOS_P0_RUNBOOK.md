<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic TAPER P0 — deimos runbook

End-to-end command sequence to validate the P0 experiment on the deimos 8×H100
box (`deimos.pre.nvidia.com`, user `anjshah`). The harness runs **client-side
against stock Dynamo + vLLM** — no Dynamo code changes. See `README.md` for the
design; this file is the operational checklist.

> **Backend = vLLM, not SGLang.** The size-matched MoE pair we want
> (Nemotron-3.5-Lightning-30B-A3B + Qwen3.6-35B-A3B) has no SGLang recipes that
> fit one 8×H100 box — the only SGLang-recipe Qwen MoE is a 2.4T multi-node
> model. vLLM has recipes for both, and FPM (the A3 load signal) is available on
> vLLM. `experiment.py`'s MODELS comment already reflects this.

> Claude's outbound SSH is sandbox-blocked — run these on the box yourself and
> paste output back. Working dir on the box: `/workspace/agentic_taper`.

---

## ✅ Working recipe (2026-08-25) — measure ITL server-side

Clean A0 baseline achieved: `matched 91/91`, goodput=1.0, victim_p95_itl≈19 ms.
The measurement path that works, and the three traps that cost a day of debugging:

**Measure ITL from the server, not the client.** The replay client on one asyncio
loop cannot time tokens under load (60+ concurrent SSE streams starve the loop →
flat ~690 ms artifact, independent of concurrency). Instead the frontend already
logs per-request `avg_itl_ms`/`ttft_ms`/`output_tokens`; the harness reads those:
- Run the frontend logging to a file: `python3 -m dynamo.frontend … > /tmp/frontend.log 2>&1 &`
- Pass `--frontend-log /tmp/frontend.log`. Each cell prints `server-metrics matched N/M`
  (want ~all). The client captures the SSE completion `id` and joins it to the log.

**Right-size the prompt.** Synthetic prompt words tokenize to ~4 tokens each, so the
old `words_per_block=400` made ~16 k-token prompts that genuinely overloaded the
server (279 ms ITL). Use `--words-per-block 32` (≈128 tokens/block, still fills KV
blocks for co-location) → ~640-token prompts, ~17 ms server ITL. Tunable live.

**Three traps (all fixed in code, noted so they're not re-hit):**
1. A "request completed" line carries **two** `request_id`s and the completion id is
   the *second*; `parse_frontend_metrics` keys on **all** ids on the line.
2. The frontend **colorizes logs even to a file** — parser strips ANSI
   (`\x1b\[[0-9;]*m`) before regex. (Or launch with `NO_COLOR=1`.)
3. `--models` takes the short **label**, not the served name (`$SERVED` is only for
   the `/v1/models` curl).

### The run commands
```bash
cd /workspace/agentic_taper && export DYN_NAMESPACE=taper-anjshah
# frontend must be logging to a file (relaunch redirected if it isn't):
#   pkill -f dynamo.frontend; python3 -m dynamo.frontend --trust-remote-code \
#     --http-port $FRONT_PORT > /tmp/frontend.log 2>&1 &

# 1-cell smoke test — expect matched N/N, goodput≈1.0, ITL ~15-30 ms:
python3 experiment.py --models nemotron-3.5-lightning-30b-a3b \
  --base-url "http://localhost:$FRONT_PORT" --frontend-log /tmp/frontend.log \
  --outdir ./results_ok --limit 1 --reps 1 --max-concurrency 64 --n-tasks 40 \
  --words-per-block 32

# H1 comparison — all 9 A0 + first 3 A1 cells (paired A0 k2 b1 = cell 1 vs A1 k2 b1 = cell 10):
python3 experiment.py --models nemotron-3.5-lightning-30b-a3b \
  --base-url "http://localhost:$FRONT_PORT" --frontend-log /tmp/frontend.log \
  --outdir ./results_h1 --limit 12 --reps 1 --max-concurrency 128 --n-tasks 120 \
  --words-per-block 32
```
Flags: `--frontend-log` (server-side ITL), `--words-per-block` (prompt size),
`--max-concurrency` (128 is good now — high concurrency drives the co-batch load
H1 needs and no longer corrupts the measurement), `--n-tasks`. 62 unit tests pass.

Server-side OTel/Tempo (old "step c") is **no longer needed for ITL** — the frontend
log gives server-measured per-request `avg_itl_ms` without the Tempo stack. Only
revisit if per-token p50/p99/max tails (vs per-request mean) are ever required.

<details><summary>Original cheapest-first hypothesis list (superseded by the diagnosis above)</summary>

First limited live run showed **goodput=0.0 with victim_p95_itl ≈ 690 ms** on the
**A0 baseline**, flat across burst b1/b3/b8:

```
[1/6] nemotron-3.5-lightning-30b-a3b__A0__k2_b1_pf8_r0  goodput=0.0  victim_p95_itl=690.776ms
[2/6] ...__A0__k2_b3_pf8_r0  goodput=0.0  victim_p95_itl=714.039ms
[3/6] ...__A0__k2_b8_pf8_r0  goodput=0.0  victim_p95_itl=689.384ms
[4/6] ...__A0__k5_b1_pf8_r0  goodput=0.0  victim_p95_itl=682.026ms
```

**Read before re-running:** goodput=0.0 is *arithmetically forced*, not a result.
`task_goodput` (`replay_client.py:331`) counts a task good only if mean ITL ≤
`itl_slo_ms`, and `itl_slo_ms=20.0` (`experiment.py:64`). Measured ITL ≈ 690 ms is
34× the SLO ⇒ goodput 0 by construction. The real question is **why ITL ≈ 690 ms**.

Why that number is almost certainly an artifact, not the externality effect:
- ~690 ms ITL is ~30–70× the physical expectation for a 3B-active MoE on H100
  (~10–25 ms), and this is **A0** — k=1, the lowest-concurrency arm, i.e. the
  *floor*. A broken floor means nothing downstream is interpretable.
- It is **flat across burst b1/b3/b8**. For A0 that rules out queueing/load and
  points at a *fixed per-request overhead*.

Diagnose in this order (cheapest first):
1. **Single-request calibration.** Replay ONE unloaded victim request; record its
   ITL. If it's still ~690 ms → serving/measurement, not congestion. If it drops
   to ~engine level → client event-loop contention under concurrency (ITL is pure
   client-side wall-time between SSE chunks, `replay_client.py:159-165`; all on one
   asyncio loop, with `synth_prompt` run inline in `_dispatch`, `:254`).
   ```bash
   # crude single-request ITL check, bypassing the harness:
   curl -N -s http://localhost:$FRONT_PORT/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"'"$SERVED"'","prompt":"hello","max_tokens":64,"stream":true,"temperature":0}' \
     | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done
   # eyeball the inter-line deltas — they are your per-token ITL floor.
   ```
2. **SSE chunk ≠ token — and dspark spec-decode is ON.** `(now-last)` in
   `HttpFrontend.complete` assumes one token per `data:` line. The worker runs the
   `-DSpark` speculative draft (`--speculative-config … num_speculative_tokens:7`),
   so accepted tokens arrive in **bursts of up to 8** — one SSE frame can carry
   several tokens (long gap) followed by near-zero gaps, inflating mean/tail ITL as
   a *measurement* artifact. Also skip usage/keepalive/role chunks. Confirm from
   the curl above how many tokens each `data:` frame carries; if >1, either divide
   the gap by the token count or disable spec-decode for an ITL-calibration run.
3. **Wrong served name / path.** Confirm `curl -s localhost:$FRONT_PORT/v1/models | jq`
   lists exactly the base NVFP4 id you pass as `--models`/`SERVED`, and that
   `/v1/completions` (not `/v1/chat/completions`) is the right streaming route.
4. **SLO calibration.** Only after 1–3: measure unloaded p50 ITL and set
   `itl_slo_ms` from it (README "Two models"). 20 ms is a placeholder anchor.

Do NOT run the full matrix until A0 shows a sane, low victim ITL — A0 is the
paired baseline every other arm is measured against.

</details>

---

## 0. Sanity (no GPU — safe anywhere)

```bash
cd /workspace/agentic_taper
python3 -m unittest discover -s tests -v        # 60 tests, stdlib + msgspec
python3 experiment.py --dry-run --reps 1        # offline wiring check, MockFrontend
```

## 1. Deploy the Dynamo + vLLM stack on the box

**Shared-box preflight (deimos is multi-tenant — check first):**
- **Is a Dynamo stack already up?** `docker ps | grep vllm-runtime`; `nvidia-smi`
  (look for a 30B worker, not just the 4 GB `nemotron-embed-server`);
  `curl -s localhost:$FRONT_PORT/v1/models` — a real Dynamo list `{"data":[]}`
  vs `{"detail":"Not Found"}` (something else) tells you if it's yours. The
  container is `docker run --rm`, so a closed SSH session tears it down.
- **Run inside `tmux`** (`tmux new -s taper` → `docker run …` inside it) so an SSH
  drop doesn't `--rm` the whole stack.
- **Ports:** 8000 is a resident `nemotron-embed` NIM and 8001 has been seen
  occupied too. `ss -ltnp | grep -E ':8000|:8001|:8010'` and pick a free
  `FRONT_PORT` (e.g. 8010).
- **etcd/nats:** the `clu-clu-run-*` etcd/nats containers are **bridge-networked,
  not published to host**, so a `--network=host` Dynamo container can't reach them
  at localhost:2379/4222. Verify host reachability: `curl -s localhost:2379/health`
  + `ss -ltnp | grep -E ':2379|:4222'`. If absent, start Dynamo's own:
  `docker compose -f deploy/docker-compose.yml up -d etcd nats`.


Container (model-specific dev image; all GPUs visible so the worker uses
`CUDA_VISIBLE_DEVICES` to pick devices):

```bash
docker run --rm -it \
  --gpus '"device=0"' --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/scratch/anjshah/model-cache:/model-cache \
  -v /mnt/scratch/anjshah/agentic_taper:/workspace/agentic_taper \
  -e HF_HOME=/model-cache -e HF_MODULES_CACHE=/tmp/hf_modules -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1 \
  bash
```
> The trailing `bash` is required — this image has no default CMD, so omitting a
> command errors with `docker: no command specified`. (`--entrypoint /bin/bash`
> before the image name works too.)

**Model-cache write permission.** Docker auto-creates a missing bind-mount source
as **`root:root`**, so `/mnt/scratch/anjshah/model-cache` ends up root-owned and the
container's **non-root** runtime user can't write it — the worker dies with `Failed
to create cache directory "/model-cache/hub": Permission denied (os error 13)` (the
`ModelExpress`/h2 warnings just above it are a harmless fallback to direct HF
download — the perm error is the fatal line). A plain `chmod` fails too (`Operation
not permitted`) because you don't own it. Fix on the HOST with sudo (this is what
worked):
```bash
sudo chown -R $(id -u):$(id -g) /mnt/scratch/anjshah/model-cache
chmod -R 777 /mnt/scratch/anjshah/model-cache
```
Only the **worker** writes the cache, so afterward just re-run the `dynamo.vllm`
block in the existing container — no need to restart the frontend. The model is
downloaded on first run (a few GB of NVFP4 weights), so expect a wait before the
serving line.

No sudo? Two alternatives: relaunch the container with `--user root` (root ignores
the dir's perm bits), or point `HF_HOME` at the already-writable code mount
(`export HF_HOME=/workspace/agentic_taper/hf-cache && mkdir -p $HF_HOME`) — the
latter parks weights next to the code on scratch, so exclude it from scp syncs.

Inside the container:

```bash
cd /workspace/agentic_taper
export DYN_NAMESPACE=taper-anjshah
export FRONT_PORT=8002        # pick a FREE port (8000/8001 have been occupied); see preflight

# 1a. Frontend
python3 -m dynamo.frontend --trust-remote-code --http-port $FRONT_PORT &
sleep 5
curl -s http://localhost:$FRONT_PORT/v1/models   # empty model list, no bind error

# 1b. Nemotron worker (agg; -DSpark is the vLLM speculative DRAFT, num_speculative_tokens=7)
python3 -m dynamo.vllm \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-num-seqs 512 --max-num-batched-tokens 32768 \
  --enable-prefix-caching --async-scheduling --quantization modelopt_fp4 \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode align --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --dyn-tool-call-parser nemotron_nano --dyn-reasoning-parser nemotron_nano \
  --reasoning-parser nemotron_v3 \
  --speculative-config '{"method":"dspark","model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}' \
  --disaggregation-mode agg > /tmp/worker.log 2>&1 &
tail -f /tmp/worker.log                        # wait for "ready"; Ctrl-C the tail once up
```

The second model (Qwen3.6-35B-A3B, recipe `recipes/qwen3.6-35b-a3b/vllm/agg`)
deploys the same way; `manifest.jsonl` merges runs, so deploy one, run, swap, run.
Yesterday's session ran **Nemotron only**.

Confirm what the frontend actually serves (the served name is the **base NVFP4**
id, NOT the `-DSpark` speculative draft):

```bash
curl -s http://localhost:$FRONT_PORT/v1/models | jq
export SERVED=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```

FPM discovery (A3 only; skip for A0/A1/A2) — defaults shown, override to match:
`DYN_FPM_COMPONENT` (default `backend`), `DYN_FPM_ENDPOINT` (default `generate`),
`DYN_DISCOVERY_BACKEND` (etcd), `DYN_REQUEST_PLANE` (nats). Verify FPM flows:

```bash
python -m dynamo.common.recv_forward_pass_metrics --mode tracking
```

## 2. Derisk spikes (before the matrix — README "Derisk")

1. **Co-location spike** — ~20 requests sharing a prefix; confirm they co-batch on
   ONE worker. If KV-overlap routing spreads siblings, there's no shared decode
   step and the experiment measures nothing. Fail loudly.
2. **FPM freshness spike** — `FpmSource`; measure real staleness under load. If
   `freshness_ms` ≈ 100 ms (not ~ms), τ must absorb the lag and A3's ceiling drops.
3. **Crude A1-vs-A0 spike** — no gate, no sweeps; does *any* victim tail
   degradation appear? If not at achievable scale, raise K and prefix depth.

## 3. The run command (as used 2026-08-25)

Single deployed model, limited cells, one rep — the smoke-test shape:

```bash
cd /workspace/agentic_taper
export DYN_NAMESPACE=taper-anjshah
# --models = short LABEL (not $SERVED); base NVFP4 served name is baked into MODELS.
python3 experiment.py --models nemotron-3.5-lightning-30b-a3b \
  --base-url "http://localhost:$FRONT_PORT" \
  --outdir ./results --limit 6 --reps 1 --max-concurrency 64 --n-tasks 120
```

`--models` selects one label; `manifest.jsonl` merges across runs to the same
`--outdir`, so you can deploy one model, run, swap, run. Drop `--limit` for the
full sweep once A0 is sane.

Full matrix (both models, all arms, ≥5 reps — after calibration):

```bash
python3 experiment.py --outdir ./results        # LIVE, all MODELS x 4 arms x sweep
```

## 4. Analyze + decision rules (offline, no GPU)

```bash
python3 analyze.py --outdir ./results           # per-model H1/H2 verdicts, summary.json
```

`analyze.py` compares arms **at matched load** (near the knee), not averaged over
the sweep. Pre-registered rules (write thresholds before running):
- **H1a** confirmed if A1's goodput-vs-load curve has a knee and A0 shows none.
- **H1 falsified** if goodput tracks throughput monotonically, or victim tail is
  paired-flat A0 vs A1 → report the negative, stop.
- **H2** confirmed if A3 beats best-K A2 on goodput, paired CI excluding zero, at
  ≥ comparable throughput.
- **H2 falsified** if A3 ≤ best-K A2 → report, stop.

Run each (arm, sweep-point) ≥5× with a fixed per-config seed; report median + IQR.

## 5. (Optional) Mode A — real Claude Code trace

```bash
claude_trace_export → request_trace_to_mooncake --agentic → AgenticMooncakeRow JSONL
python3 trace_adapter.py --in agentic.jsonl --out replay.jsonl
# then feed replay.jsonl to the replay client / experiment matrix
```

---

## Arm → gate map (one variable changes)

| Arm | Gate config | Set where |
|-----|-------------|-----------|
| A0 | `STATIC_CAP, k=1` — one request per task (baseline) | trace/gate |
| A1 | `EAGER` — admit as deps clear | gate |
| A2 | `STATIC_CAP, k=K` — fixed per-task cap | `--` swept caps |
| A3 | `FPM, load_threshold=τ` on live `num_decode_requests` | swept τ |
