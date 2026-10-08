#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load sweep (NEXT_STEPS 7.3i): independent single-GPU stacks run in parallel,
# one per slot, each with its own arm and agent concurrency. Slot s uses GPU s
# and PORT_OFFSET=1000*s (HTTP port 8100+1000*s, FPM 20081+1000*s).
#
#   ./sweep_parallel.sh <sweep-tag> "<arm>:<agents> ..."     # one spec per GPU
#   ./sweep_parallel.sh s1 "kv:6 ctx:8 kv:8 ctx:10 kv:10 ctx:12 kv:12 ctx:14"
#   python3 sweep_report.py --sweep s1                          # afterwards
#
# arm: kv (stock KV router) or ctx (taper_router, count cap 7 + 300k context
# budget, unchanged from the 7.3f ctx arm). Every slot gets the same workload as
# the 1-GPU comparisons: Claude Code + v2 subagent instruction on SWE-bench
# Verified, N_TASKS tasks, an interactive client at VICTIM_MEAN_MS. Runs are
# named sw<tag>-<arm>-n<agents>; a slot is skipped if its Harbor job exists.
set -euo pipefail

TAG="${1:?usage: $0 <sweep-tag> \"<arm>:<agents> ...\"}"
SPECS="${2:?usage: $0 <sweep-tag> \"<arm>:<agents> ...\"}"
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
VICTIM_MEAN_MS="${VICTIM_MEAN_MS:-2000}"
CTX_THRESHOLD="${CTX_THRESHOLD:-7}"
CTX_BUDGET="${CTX_BUDGET:-300000}"
INSTR="$REPO/benchmarks/agentic_taper/harbor_parallel_subagents_v2.md"
OUT_REL=harbor_ab
OUT="$REPO/$OUT_REL"
SITE=/usr/local/lib/python3.12/dist-packages/dynamo/taper_router
PY="$REPO/.venv/bin/python3"

SOCK="${XDG_RUNTIME_DIR:?source ~/.bashrc first}/podman/podman.sock"
[[ -S "$SOCK" ]] || { echo "podman API socket $SOCK not running" >&2; exit 2; }
read -r -a SPEC <<<"$SPECS"
NGPU=$(nvidia-smi -L | wc -l)
(( ${#SPEC[@]} <= NGPU )) || { echo "${#SPEC[@]} slots but only $NGPU GPUs" >&2; exit 2; }
mkdir -p "$OUT"

wait_for() { local d=$((SECONDS + $1)); shift; until "$@"; do (( SECONDS < d )) || return 1; sleep 5; done; }

declare -a RUN CID VPID HPID PORT
cleanup() {  # stop everything this sweep started (also on Ctrl-C)
    for s in "${!SPEC[@]}"; do
        [[ -n "${HPID[$s]:-}" ]] && pkill -TERM -P "${HPID[$s]}" 2>/dev/null || true
        [[ -n "${HPID[$s]:-}" ]] && kill "${HPID[$s]}" 2>/dev/null || true
        [[ -n "${VPID[$s]:-}" ]] && kill "${VPID[$s]}" 2>/dev/null || true
        [[ -n "${CID[$s]:-}" ]] && podman stop -t 10 "${CID[$s]}" >/dev/null 2>&1 || true
    done
    # Harbor's task containers outlive a killed Harbor process.
    podman ps -aq --filter name=__env-main | xargs -r podman rm -f >/dev/null 2>&1 || true
}
trap 'cleanup; exit 130' INT TERM

echo "=== sweep $TAG ($(date -u +%H:%M:%S)): $SPECS"
for s in "${!SPEC[@]}"; do
    arm=${SPEC[$s]%%:*}; n=${SPEC[$s]##*:}
    [[ "$arm" == kv || "$arm" == ctx ]] || { echo "bad arm in ${SPEC[$s]}" >&2; exit 2; }
    run="sw$TAG-$arm-n$n"; RUN[$s]=$run; PORT[$s]=$((8100 + 1000 * s))
    if [[ -d "$REPO/jobs/harbor-ab-$run" ]]; then echo "slot $s: skip $run (done)"; RUN[$s]=""; continue; fi
    if curl -fsS "http://127.0.0.1:${PORT[$s]}/v1/models" >/dev/null 2>&1; then
        echo "slot $s: port ${PORT[$s]} already serving" >&2; cleanup; exit 2
    fi
    mkdir -p "$OUT/$run"; rm -f "$OUT/$run/stack.log" "$OUT/$run/trace.jsonl" "$OUT/$run/fpm.jsonl"
    extra=()
    policy=kv
    if [[ "$arm" == ctx ]]; then
        policy=taper
        extra=(-e "DYN_TAPER_LOAD_THRESHOLD=$CTX_THRESHOLD" -e "DYN_TAPER_CONTEXT_BUDGET_TOKENS=$CTX_BUDGET")
    fi
    # One GPU per container (CDI device s); inside it is GPU 0.
    CID[$s]=$(podman run -d --rm --init --network host --device "nvidia.com/gpu=$s" \
        -v "$REPO:/workspace" -v "$REPO/components/src/dynamo/taper_router:$SITE" \
        -v "$HF:$HF" -e "HF_HOME=$HF" -e NO_COLOR=1 \
        -e "MODEL_PATH=$MODEL_PATH" -e "MODEL_NAME_ROUTER=$MODEL" -e GPUS=0 \
        -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" -e ANTHROPIC_API=1 \
        -e "PORT_OFFSET=$((1000 * s))" \
        -e DYN_REQUEST_TRACE=1 -e DYN_REQUEST_TRACE_SINKS=jsonl \
        -e "DYN_REQUEST_TRACE_OUTPUT_PATH=/workspace/$OUT_REL/$run/trace.jsonl" \
        ${extra[@]+"${extra[@]}"} "$IMAGE" \
        bash -c "cd /workspace && ./components/src/dynamo/taper_router/run_smoketest_1gpu.sh $policy > /workspace/$OUT_REL/$run/stack.log 2>&1")
    echo "slot $s: $run on GPU $s, port ${PORT[$s]}"
done

for s in "${!SPEC[@]}"; do
    run=${RUN[$s]}; [[ -z "$run" ]] && continue
    if ! wait_for 1800 grep -q "smoketest stack ready" "$OUT/$run/stack.log"; then
        echo "slot $s ($run) never became ready; see $OUT/$run/stack.log" >&2; cleanup; exit 1
    fi
done
echo "all stacks ready ($(date -u +%H:%M:%S))"
sleep 5

for s in "${!SPEC[@]}"; do
    run=${RUN[$s]}; [[ -z "$run" ]] && continue
    n=${SPEC[$s]##*:}; url="http://127.0.0.1:${PORT[$s]}"
    podman exec -d "${CID[$s]}" bash -c "python3 /workspace/benchmarks/agentic_taper/fpm_probe.py \
        record --port $((20081 + 1000 * s)) --out /workspace/$OUT_REL/$run/fpm.jsonl \
        > /workspace/$OUT_REL/$run/fpm_probe.log 2>&1"
    "$PY" "$REPO/benchmarks/agentic_taper/victim_client.py" --base-url "$url" --model "$MODEL" \
        --arm "$run" --out "$OUT/$run/victims.jsonl" --n-victims 100000 \
        --mean-arrival-ms "$VICTIM_MEAN_MS" --seed 0 > "$OUT/$run/victim_client.log" 2>&1 &
    VPID[$s]=$!
    (cd "$REPO" && "$HARBOR" run -d swebench-verified@1.0 -a claude-code -m "$MODEL" \
        --ae "ANTHROPIC_BASE_URL=$url" --ae ANTHROPIC_API_KEY=dynamo-local \
        --ae "ANTHROPIC_MODEL=$MODEL" --ae "ANTHROPIC_SMALL_FAST_MODEL=$MODEL" \
        --ae "CLAUDE_CODE_SUBAGENT_MODEL=$MODEL" --ae CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
        --extra-docker-compose "$PLUGINS/pi-plugin/harbor/host-network.yml" \
        --job-name "harbor-ab-$run" -y -l "$N_TASKS" -n "$n" --extra-instruction-path "$INSTR" \
        > "$OUT/$run/harbor.log" 2>&1) &
    HPID[$s]=$!
    echo "slot $s: $run started ($n agents)"
done

for s in "${!SPEC[@]}"; do
    [[ -z "${HPID[$s]:-}" ]] && continue
    wait "${HPID[$s]}" || echo "slot $s: harbor exited non-zero (see $OUT/${RUN[$s]}/harbor.log)" >&2
    kill "${VPID[$s]}" 2>/dev/null || true; wait "${VPID[$s]}" 2>/dev/null || true
    podman stop -t 10 "${CID[$s]}" >/dev/null 2>&1 || true
    run=${RUN[$s]}
    gate=""
    if [[ "$run" == *-ctx-* ]]; then
        log=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$run/stack.log")
        gate=" defers=$(grep -c 'taper.defer' <<<"$log" || true) forced=$(grep -c 'taper.forced_admit' <<<"$log" || true)"
    fi
    echo "slot $s: $run finished ($(date -u +%H:%M:%S))$gate"
done
trap - INT TERM
echo "sweep $TAG done. Next: python3 $REPO/benchmarks/agentic_taper/sweep_report.py --sweep $TAG"
