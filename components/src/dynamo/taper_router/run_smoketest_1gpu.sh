#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scaled-down single-GPU smoke test for taper_router: proves the
# session-id -> gate -> KvRouter -> vLLM wiring works end to end without
# needing the 8xH100 MiniMax-M2 scale run_minimax_8xh100.sh uses (that
# script mirrors ThunderAgent's original 2xTP4 production-shaped walkthrough
# for the real §7.4 Harbor/Pi comparison -- overkill for just checking the
# plumbing). One TP1 worker, a small fast-downloading model. Usage:
#   ./run_smoketest_1gpu.sh kv     # stock KV router, no gate
#   ./run_smoketest_1gpu.sh taper  # taper_router gate
#   ./run_smoketest_1gpu.sh ta     # thunderagent_router (same router-hop design; control)
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

POLICY="${1:-}"
if [[ "$POLICY" != "kv" && "$POLICY" != "taper" && "$POLICY" != "ta" ]]; then
    echo "usage: $0 kv|taper|ta" >&2
    exit 2
fi

trap dynamo_exit_trap EXIT

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_NAME_ROUTER="${MODEL_NAME_ROUTER:-Qwen2.5-1.5B-Instruct}"
WORKER_MODEL="$MODEL_NAME_ROUTER"
BLOCK_SIZE=16
HTTP_PORT=8100
# Pilot knobs (defaults keep the original 1-GPU Qwen smoke test):
#   TP=1 GPUS=0 TOOL_PARSER= REASONING_PARSER=  (e.g. glm47 / glm45 for GLM-4.7-Flash)
#   DYN_REQUEST_TRACE=1 DYN_REQUEST_TRACE_OUTPUT_PATH=...  records session/parent ids
#   ANTHROPIC_API=1  serve /v1/messages (Claude Code); off by default so sweep
#                    runs stay configured exactly like earlier ones
TP="${TP:-1}"
FRONTEND_ARGS=()
[[ "${ANTHROPIC_API:-0}" == "1" ]] && FRONTEND_ARGS+=(--enable-anthropic-api)
GPUS="${GPUS:-0}"
PARSER_ARGS=()
[[ -n "${TOOL_PARSER:-}" ]] && PARSER_ARGS+=(--dyn-tool-call-parser "$TOOL_PARSER")
[[ -n "${REASONING_PARSER:-}" ]] && PARSER_ARGS+=(--dyn-reasoning-parser "$REASONING_PARSER")

if [[ "$POLICY" == "taper" || "$POLICY" == "ta" ]]; then
    WORKER_MODEL="dyn-internal-smoketest"
fi

export PYTHONHASHSEED=0
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-/tmp/dynamo-smoketest-${POLICY}-$$}"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
mkdir -p "$DYN_FILE_KV"

# WORKERS=N: N data-parallel copies, worker i on GPU i (TP must be 1). Worker
# 0 keeps the original ports so 1-GPU runs are unchanged; worker i>0 uses
# system 8190+i, FPM 20100+i, NIXL side channel 20200+i, KV events 20300+i.
WORKERS="${WORKERS:-1}"
if (( WORKERS > 1 && TP != 1 )); then echo "WORKERS>1 needs TP=1" >&2; exit 2; fi
worker_ports() {  # worker_ports <i> -> "system fpm nixl kvevents"
    if (( $1 == 0 )); then echo "8181 20081 20097 20080"
    else echo "$((8190 + $1)) $((20100 + $1)) $((20200 + $1)) $((20300 + $1))"; fi
}
for (( i = 0; i < WORKERS; i++ )); do
    read -r SYS FPM NIXL KVE <<<"$(worker_ports $i)"
    gpu=$GPUS; (( WORKERS > 1 )) && gpu=$i
    DYN_SYSTEM_PORT=$SYS DYN_FORWARDPASS_METRIC_PORT=$FPM \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL CUDA_VISIBLE_DEVICES="$gpu" \
    python -m dynamo.vllm \
        --model "$MODEL_PATH" --served-model-name "$WORKER_MODEL" \
        --tensor-parallel-size "$TP" --block-size "$BLOCK_SIZE" \
        --enable-prefix-caching ${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$KVE\",\"enable_kv_cache_events\":true}" &
done

if [[ "$POLICY" == "taper" ]]; then
    DYN_SYSTEM_PORT=8183 python -m dynamo.taper_router \
        --endpoint dynamo.backend.generate \
        --model-name "$MODEL_NAME_ROUTER" \
        --model-path "$MODEL_PATH" ${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} \
        --router-block-size "$BLOCK_SIZE" \
        --load-threshold "${DYN_TAPER_LOAD_THRESHOLD:-32}" \
        --context-budget-tokens "${DYN_TAPER_CONTEXT_BUDGET_TOKENS:-0}" \
        --defer-timeout-seconds "${DYN_TAPER_DEFER_TIMEOUT_SECONDS:-300}" \
        $( [[ "${DYN_TAPER_SHADOW_MODE:-false}" == "true" ]] && echo --shadow-mode || echo --no-shadow-mode ) \
        --shared-cache-type none &
    ROUTER_MODE=round-robin
elif [[ "$POLICY" == "ta" ]]; then
    DYN_SYSTEM_PORT=8183 python -m dynamo.thunderagent_router \
        --endpoint dynamo.backend.generate \
        --model-name "$MODEL_NAME_ROUTER" \
        --model-path "$MODEL_PATH" ${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} \
        --router-block-size "$BLOCK_SIZE" \
        --shared-cache-type none &
    ROUTER_MODE=round-robin
else
    ROUTER_MODE=kv
fi

DYN_SYSTEM_PORT=8184 python -m dynamo.frontend \
    --http-host 0.0.0.0 \
    --http-port "$HTTP_PORT" \
    --router-mode "$ROUTER_MODE" ${FRONTEND_ARGS[@]+"${FRONTEND_ARGS[@]}"} \
    --shared-cache-type none &

# Ready = the worker's own model name is listed (it registers only once the
# engine is up) and the public name is listed (router registered). Polling
# /v1/models/<name>/ready breaks when the Anthropic API is enabled.
models() { curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" 2>/dev/null; }
until models | grep -Fq "\"$WORKER_MODEL\"" && models | grep -Fq "\"$MODEL_NAME_ROUTER\""; do
    sleep 5
done
# With several workers, also wait until every one reports ready on its
# system port (/health: 200 "ready", 503 "notready").
for (( i = 0; i < WORKERS && WORKERS > 1; i++ )); do
    read -r SYS _ <<<"$(worker_ports $i)"
    until curl -fsS "http://127.0.0.1:$SYS/health" >/dev/null 2>&1; do sleep 5; done
done
echo "$POLICY smoketest stack ready at http://127.0.0.1:${HTTP_PORT}/v1 (workers=$WORKERS)"

wait_any_exit
