#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pre-registered Harbor comparison (NEXT_STEPS 7.3e): Claude Code on SWE-bench
# Verified against Dynamo, kv vs taper, with interactive victim traffic.
#
#   ./harbor_ab.sh unloaded   # stock kv, nothing else running: 30 sparse
#                             # victims, then one Claude Code task alone
#   ./harbor_ab.sh compare    # kv,taper,taper,kv (2 reps, counterbalanced)
#   ./harbor_ab.sh diag       # not part of the comparison: stock kv + victims,
#                             # every forward pass recorded (fpm_probe.py), for
#                             # trunk-only (no instruction) then fan-out (v2)
#   python3 trace_goodput.py --unloaded-trace $OUT/unloaded/trace.jsonl \
#       kv-r1=$OUT/kv-r1/trace.jsonl taper-r1=... taper-r2=... kv-r2=...
#
# Needs: the podman API socket running (Harbor's compose calls), the model
# already cached, nothing on port 8100. Runs are skipped if their Harbor job
# directory already exists, so rerunning resumes.
set -euo pipefail

MODE="${1:-}"
REPO="${REPO:-/scratch/$USER/src/dynamo}"
HARBOR="${HARBOR:-/scratch/$USER/src/harbor/.venv/bin/harbor}"
PLUGINS="${PLUGINS:-/scratch/$USER/src/agent-plugins}"
HF="${HF:-/scratch/$USER/hf_cache}"
IMAGE="${IMAGE:-localhost/dynamo:latest-vllm-runtime}"
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-4.7-Flash}"
MODEL="${MODEL:-GLM-4.7-Flash}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"
N_TASKS="${N_TASKS:-30}"
CONCURRENT="${CONCURRENT:-10}"
THRESHOLD="${THRESHOLD:-32}"
# ctx arm (second registration, NEXT_STEPS 7.3f): count cap + context budget
CTX_THRESHOLD="${CTX_THRESHOLD:-7}"
CTX_BUDGET="${CTX_BUDGET:-300000}"
# cap arm (third registration, NEXT_STEPS 7.3g): ctx gate with a bounded hold
CAP_TIMEOUT="${CAP_TIMEOUT:-1.0}"
VICTIM_MEAN_MS="${VICTIM_MEAN_MS:-2000}"
ORDER="${ORDER:-kv-r1 taper-r1 taper-r2 kv-r2}"
INSTR="$REPO/benchmarks/agentic_taper/harbor_parallel_subagents_v2.md"
# SUITE separates a model/setup's runs from earlier ones (e.g. SUITE=-mm for
# MiniMax-M2): outputs go to harbor_ab$SUITE/, Harbor jobs to jobs/harbor-ab$SUITE-<run>.
SUITE="${SUITE:-}"
OUT_REL="harbor_ab$SUITE"            # under $REPO, so the container sees it
OUT="$REPO/$OUT_REL"
JP="harbor-ab$SUITE"                 # Harbor job-name prefix
# STACK_GPUS: GPUs for this stack ("all", or e.g. "0,1,2,3" for one TP4 copy);
# PORT_OFFSET shifts every port, so two stacks (e.g. GPUs 0-3 and 4-7, offsets 0
# and 1000) can run side by side. TP and EXTRA_VLLM_ARGS pass to the launcher.
STACK_GPUS="${STACK_GPUS:-all}"
PO="${PORT_OFFSET:-0}"
TP="${TP:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
URL=http://127.0.0.1:$((8100 + PO))
if [[ "$STACK_GPUS" == all ]]; then
    DEV_ARGS=(--device nvidia.com/gpu=all); IN_GPUS="${GPUS:-0}"; SMI_IDS=()
else
    DEV_ARGS=(); IN_GPUS=""; k=0
    IFS=, read -r -a _ids <<<"$STACK_GPUS"
    for g in "${_ids[@]}"; do DEV_ARGS+=(--device "nvidia.com/gpu=$g"); IN_GPUS+="${IN_GPUS:+,}$k"; k=$((k + 1)); done
    SMI_IDS=(-i "$STACK_GPUS")
fi
SITE=/usr/local/lib/python3.12/dist-packages/dynamo/taper_router
PY="${PY:-$REPO/.venv/bin/python3}"   # any python3 with aiohttp
# vLLM TP>1 workers share state via /dev/shm; podman's 64 MB default is too
# small (MiniMax-M2 TP4 needs >160 MB). Harmless for TP1.
SHM_SIZE="${SHM_SIZE:-32g}"

DIAG_TASKS="${DIAG_TASKS:-10}"
DIAG_TAG="${DIAG_TAG:-}"             # e.g. DIAG_TAG=-n5 CONCURRENT=5 ./harbor_ab.sh diag
DIAG_KINDS="${DIAG_KINDS:-trunk fanout}"   # e.g. DIAG_KINDS=fanout for the scale check
# Data-parallel model copies, one per GPU (needs a WORKERS-GPU allocation).
# Scale CONCURRENT and VICTIM_MEAN_MS with it, e.g. WORKERS=8 CONCURRENT=80
# VICTIM_MEAN_MS=250.
WORKERS="${WORKERS:-1}"
FPM_PORTS=$((20081 + PO))
for (( i = 1; i < WORKERS; i++ )); do FPM_PORTS+=",$((20100 + PO + i))"; done
[[ "$MODE" =~ ^(unloaded|compare|diag)$ ]] || { echo "usage: $0 unloaded|compare|diag" >&2; exit 2; }
SOCK="${XDG_RUNTIME_DIR:?source ~/.bashrc first}/podman/podman.sock"
[[ -S "$SOCK" ]] || { echo "podman API socket $SOCK not running (podman system service ...)" >&2; exit 2; }
curl -fsS "$URL/v1/models" >/dev/null 2>&1 && { echo "something already serves $URL" >&2; exit 2; }
mkdir -p "$OUT"

CC_ENV=(--ae "ANTHROPIC_BASE_URL=$URL" --ae ANTHROPIC_API_KEY=dynamo-local
        --ae "ANTHROPIC_MODEL=$MODEL" --ae "ANTHROPIC_SMALL_FAST_MODEL=$MODEL"
        --ae "CLAUDE_CODE_SUBAGENT_MODEL=$MODEL" --ae CLAUDE_CODE_ATTRIBUTION_HEADER=0
        --extra-docker-compose "$PLUGINS/pi-plugin/harbor/host-network.yml")

gpu_idle() { [[ -z "$(nvidia-smi ${SMI_IDS[@]+"${SMI_IDS[@]}"} --query-compute-apps=pid --format=csv,noheader)" ]]; }
# Ready, or stop waiting early on a fatal worker startup error (the frontend
# keeps answering health checks after the worker dies).
stack_ready() {
    if grep -qE "Engine core initialization failed|Insufficient space in /dev/shm|CUDA out of memory|No such file or directory: .*config.json" "$1" 2>/dev/null; then
        echo "fatal error in $1:" >&2; grep -m3 -E "Engine core initialization failed|Insufficient space in /dev/shm|CUDA out of memory|No such file or directory: .*config.json" "$1" >&2; return 2
    fi
    grep -q "smoketest stack ready" "$1" 2>/dev/null
}
wait_for() { local d=$((SECONDS + $1)); shift; local rc; while :; do "$@"; rc=$?; (( rc == 0 )) && return 0; (( rc == 2 )) && return 1; (( SECONDS < d )) || return 1; sleep 5; done; }
port_free() { ! curl -fsS "$URL/v1/models" >/dev/null 2>&1; }

start_stack() {  # start_stack <run> <policy> [extra -e args...]
    local run=$1 policy=$2; shift 2
    mkdir -p "$OUT/$run"; rm -f "$OUT/$run/stack.log" "$OUT/$run/trace.jsonl"
    CID=$(podman run -d --rm --init --network host "${DEV_ARGS[@]}" --shm-size "$SHM_SIZE" \
        -v "$REPO:/workspace" -v "$REPO/components/src/dynamo/taper_router:$SITE" \
        -v "$HF:$HF" -e "HF_HOME=$HF" -e NO_COLOR=1 \
        -e "MODEL_PATH=$MODEL_PATH" -e "MODEL_NAME_ROUTER=$MODEL" \
        -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" -e ANTHROPIC_API=1 \
        -e "WORKERS=$WORKERS" -e "PORT_OFFSET=$PO" -e "TP=$TP" -e "GPUS=$IN_GPUS" \
        -e "EXTRA_VLLM_ARGS=$EXTRA_VLLM_ARGS" \
        -e DYN_REQUEST_TRACE=1 -e DYN_REQUEST_TRACE_SINKS=jsonl \
        -e "DYN_REQUEST_TRACE_OUTPUT_PATH=/workspace/$OUT_REL/$run/trace.jsonl" "$@" "$IMAGE" \
        bash -c "cd /workspace && ./components/src/dynamo/taper_router/run_smoketest_1gpu.sh $policy > /workspace/$OUT_REL/$run/stack.log 2>&1")
    # Several copies, or one large TP model (e.g. MiniMax-M2 from /data), load slowly.
    if ! wait_for $(( WORKERS > 1 || TP > 1 ? 3600 : 900 )) stack_ready "$OUT/$run/stack.log"; then
        echo "stack for $run never became ready; see $OUT/$run/stack.log" >&2
        podman stop -t 10 "$CID" >/dev/null || true; exit 1
    fi
    sleep 5
}

start_probe() {  # start_probe <run>: record every forward pass (read-only)
    local run=$1
    rm -f "$OUT/$run/fpm.jsonl"
    podman exec -d "$CID" bash -c "python3 /workspace/benchmarks/agentic_taper/fpm_probe.py \
        record --port $FPM_PORTS --out /workspace/$OUT_REL/$run/fpm.jsonl \
        > /workspace/$OUT_REL/$run/fpm_probe.log 2>&1"
    sleep 10  # idle steps aren't recorded, so check the process, not the file
    podman exec "$CID" pgrep -f fpm_probe.py >/dev/null 2>&1 \
        || echo "WARNING: fpm_probe may not be running; see $OUT/$run/fpm_probe.log" >&2
}

stop_stack() {
    podman stop -t 10 "$CID" >/dev/null || true
    wait_for 180 gpu_idle || { echo "GPU still busy" >&2; exit 1; }
    wait_for 60 port_free || { echo "port $((8100 + PO)) still bound" >&2; exit 1; }
}

harbor_job() {  # harbor_job <job-name> <extra harbor args...>
    local job=$1; shift
    (cd "$REPO" && "$HARBOR" run -d swebench-verified@1.0 -a claude-code -m "$MODEL" \
        "${CC_ENV[@]}" --job-name "$job" -y "$@")
}

if [[ "$MODE" == "unloaded" ]]; then
    run=unloaded
    if [[ -d "$REPO/jobs/$JP-unloaded" ]]; then echo "skip $run (done)"; exit 0; fi
    echo "=== $run (stock kv) ==="
    start_stack "$run" kv
    "$PY" "$REPO/benchmarks/agentic_taper/victim_client.py" --base-url "$URL" --model "$MODEL" \
        --arm unloaded --out "$OUT/$run/victims.jsonl" --n-victims 30 --mean-arrival-ms 3000
    # One task, alone, natural behavior (no instruction): parallel subagents
    # would not be "unloaded".
    harbor_job "$JP-unloaded" -i astropy__astropy-12907 -n 1
    stop_stack
    "$PY" "$REPO/benchmarks/agentic_taper/trace_fanout.py" "$OUT/$run/trace.jsonl"
    exit 0
fi

if [[ "$MODE" == "diag" ]]; then
    for kind in $DIAG_KINDS; do
        run="diag-$kind$DIAG_TAG"
        if [[ -d "$REPO/jobs/$JP-$run" ]]; then echo "skip $run (done)"; continue; fi
        echo "=== $run ($(date -u +%H:%M:%S)) ==="
        start_stack "$run" kv
        start_probe "$run"
        "$PY" "$REPO/benchmarks/agentic_taper/victim_client.py" --base-url "$URL" --model "$MODEL" \
            --arm "$run" --out "$OUT/$run/victims.jsonl" --n-victims 100000 \
            --mean-arrival-ms "$VICTIM_MEAN_MS" --seed 0 > "$OUT/$run/victim_client.log" 2>&1 &
        VPID=$!
        extra=()
        [[ "$run" == diag-fanout* ]] && extra=(--extra-instruction-path "$INSTR")
        harbor_job "$JP-$run" -l "$DIAG_TASKS" -n "$CONCURRENT" ${extra[@]+"${extra[@]}"} \
            || echo "harbor exited non-zero for $run (see jobs/$JP-$run)" >&2
        kill "$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true
        stop_stack
        "$PY" "$REPO/benchmarks/agentic_taper/trace_fanout.py" "$OUT/$run/trace.jsonl" | head -4
        python3 "$REPO/benchmarks/agentic_taper/fpm_probe.py" summarize "$OUT/$run/fpm.jsonl" \
            | tee "$OUT/$run/fpm_summary.txt"
    done
    exit 0
fi

for run in $ORDER; do
    arm=${run%-*}         # kv-r1 -> kv, ctx-p1 -> ctx, kv-c2 -> kv
    if [[ -d "$REPO/jobs/$JP-$run" ]]; then echo "skip $run (done)"; continue; fi
    echo "=== $run ($(date -u +%H:%M:%S)) ==="
    case "$arm" in
        kv)    start_stack "$run" kv ;;
        taper) start_stack "$run" taper -e "DYN_TAPER_LOAD_THRESHOLD=$THRESHOLD" ;;
        ctx)   start_stack "$run" taper -e "DYN_TAPER_LOAD_THRESHOLD=$CTX_THRESHOLD" \
                   -e "DYN_TAPER_CONTEXT_BUDGET_TOKENS=$CTX_BUDGET" ;;
        cap)   start_stack "$run" taper -e "DYN_TAPER_LOAD_THRESHOLD=$CTX_THRESHOLD" \
                   -e "DYN_TAPER_CONTEXT_BUDGET_TOKENS=$CTX_BUDGET" \
                   -e "DYN_TAPER_DEFER_TIMEOUT_SECONDS=$CAP_TIMEOUT" ;;
        *) echo "unknown arm in $run" >&2; exit 2 ;;
    esac
    start_probe "$run"
    # Interactive traffic for the whole run; analysis uses the server trace,
    # so the client is simply stopped when the agents finish.
    "$PY" "$REPO/benchmarks/agentic_taper/victim_client.py" --base-url "$URL" --model "$MODEL" \
        --arm "$arm" --out "$OUT/$run/victims.jsonl" --n-victims 100000 \
        --mean-arrival-ms "$VICTIM_MEAN_MS" --seed 0 > "$OUT/$run/victim_client.log" 2>&1 &
    VPID=$!
    harbor_job "$JP-$run" -l "$N_TASKS" -n "$CONCURRENT" --extra-instruction-path "$INSTR" \
        || echo "harbor exited non-zero for $run (see jobs/$JP-$run)" >&2
    kill "$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true
    stop_stack
    "$PY" "$REPO/benchmarks/agentic_taper/trace_fanout.py" "$OUT/$run/trace.jsonl" | head -4
    if [[ "$arm" != kv ]]; then
        log=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$run/stack.log")
        echo "gate: defers=$(grep -c 'taper.defer' <<<"$log" || true)" \
             "releases=$(grep -c 'reconcile released' <<<"$log" || true)" \
             "forced=$(grep -c 'taper.forced_admit' <<<"$log" || true)"
    fi
done
echo "done. Next: $PY $REPO/benchmarks/agentic_taper/trace_goodput.py --unloaded-trace $OUT/unloaded/trace.jsonl \\"
echo "  $(for r in $ORDER; do printf '%s=%s ' "$r" "$OUT/$r/trace.jsonl"; done)"
