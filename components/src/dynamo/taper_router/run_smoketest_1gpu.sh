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
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

POLICY="${1:-}"
if [[ "$POLICY" != "kv" && "$POLICY" != "taper" ]]; then
    echo "usage: $0 kv|taper" >&2
    exit 2
fi

trap dynamo_exit_trap EXIT

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_NAME_ROUTER="${MODEL_NAME_ROUTER:-Qwen2.5-1.5B-Instruct}"
WORKER_MODEL="$MODEL_NAME_ROUTER"
BLOCK_SIZE=16
HTTP_PORT=8100

if [[ "$POLICY" == "taper" ]]; then
    WORKER_MODEL="dyn-internal-smoketest"
fi

export PYTHONHASHSEED=0
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-/tmp/dynamo-smoketest-${POLICY}-$$}"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
mkdir -p "$DYN_FILE_KV"

DYN_SYSTEM_PORT=8181 DYN_FORWARDPASS_METRIC_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 \
python -m dynamo.vllm \
    --model "$MODEL_PATH" --served-model-name "$WORKER_MODEL" \
    --tensor-parallel-size 1 --block-size "$BLOCK_SIZE" \
    --enable-prefix-caching \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

if [[ "$POLICY" == "taper" ]]; then
    DYN_SYSTEM_PORT=8183 python -m dynamo.taper_router \
        --endpoint dynamo.backend.generate \
        --model-name "$MODEL_NAME_ROUTER" \
        --model-path "$MODEL_PATH" \
        --router-block-size "$BLOCK_SIZE" \
        --load-threshold "${DYN_TAPER_LOAD_THRESHOLD:-32}" \
        --shadow-mode "${DYN_TAPER_SHADOW_MODE:-false}" \
        --shared-cache-type none &
    ROUTER_MODE=round-robin
else
    ROUTER_MODE=kv
fi

DYN_SYSTEM_PORT=8184 python -m dynamo.frontend \
    --http-host 0.0.0.0 \
    --http-port "$HTTP_PORT" \
    --router-mode "$ROUTER_MODE" \
    --shared-cache-type none &

until curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models/${WORKER_MODEL}/ready" 2>/dev/null \
    | jq -e '([.namespaces[].worker_types.aggregated.workers // 0] | add) == 1' >/dev/null; do
    sleep 5
done
until curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" 2>/dev/null | grep -Fq "$MODEL_NAME_ROUTER"; do
    sleep 5
done
echo "$POLICY smoketest stack ready at http://127.0.0.1:${HTTP_PORT}/v1"

wait_any_exit
