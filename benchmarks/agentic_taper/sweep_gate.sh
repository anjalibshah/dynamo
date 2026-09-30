#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# kv vs taper (shadow / live at several thresholds), several seeds each, on
# one GPU. Each run gets a fresh container: `podman stop` tears down every
# process in it (no orphaned EngineCore), and a fresh engine means no prefix
# cache carried between runs -- trace_gen's hash ids restart at 0 per seed,
# so runs would otherwise share prompt prefixes.
#
# Run on the GPU node from benchmarks/agentic_taper with the dynamo venv
# active, after `source ~/.bashrc`, with nothing else on port 8100:
#   ./sweep_gate.sh
#   CONFIGS="taper-shadow taper-t32" SEEDS="0 1" ./sweep_gate.sh
# Then:
#   python3 server_gate_ab.py table "$OUT"/*.jsonl
set -euo pipefail

OUT="${OUT:-/scratch/$USER/taper_sweep}"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-kv taper-shadow taper-t16 taper-t32 taper-t64}"
REPO="${REPO:-/scratch/$USER/src/dynamo}"
HF="${HF_HOME:-/scratch/$USER/hf_cache}"
IMAGE="${IMAGE:-localhost/dynamo:latest-vllm-runtime}"
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
URL=http://127.0.0.1:8100
SITE=/usr/local/lib/python3.12/dist-packages/dynamo/taper_router

[[ -f server_gate_ab.py ]] || { echo "run from benchmarks/agentic_taper" >&2; exit 2; }
if curl -fsS "$URL/v1/models" >/dev/null 2>&1; then
    echo "something is already serving on $URL; stop it first" >&2; exit 2
fi
mkdir -p "$OUT" "$REPO/sweep_logs" "$HF"
echo "HF cache: $HF (set HF_HOME to override; a new location means a model download)"

wait_for() {  # wait_for <seconds> <command...>
    local deadline=$((SECONDS + $1)); shift
    until "$@"; do
        (( SECONDS < deadline )) || return 1
        sleep 3
    done
}
# The launch script prints this only once the worker is registered AND the
# public model is listed. Polling /v1/models alone is not enough in the taper
# arm: taper_router registers the public name before its worker is up.
stack_ready() { grep -q "smoketest stack ready" "$REPO/sweep_logs/$name.log" 2>/dev/null; }
gpu_idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; }
port_free() { ! curl -fsS "$URL/v1/models" >/dev/null 2>&1; }

for cfg in $CONFIGS; do
    case "$cfg" in
        kv)            policy=kv;    envs=() ;;
        taper-shadow)  policy=taper; envs=(-e DYN_TAPER_SHADOW_MODE=true) ;;
        taper-t*)      policy=taper; envs=(-e "DYN_TAPER_LOAD_THRESHOLD=${cfg#taper-t}") ;;
        *) echo "unknown config $cfg" >&2; exit 2 ;;
    esac
    for seed in $SEEDS; do
        name="${cfg}__s${seed}"
        if [[ -s "$OUT/$name.jsonl" ]]; then echo "skip $name (done)"; continue; fi
        echo "=== $name ==="
        rm -f "$REPO/sweep_logs/$name.log"   # a stale "ready" line would skip the wait
        cid=$(podman run -d --rm --init --network host --device nvidia.com/gpu=all \
            -v "$REPO:/workspace" -v "$REPO/components/src/dynamo/taper_router:$SITE" \
            -v "$HF:$HF" -e "HF_HOME=$HF" -e NO_COLOR=1 "${envs[@]}" "$IMAGE" \
            bash -c "cd /workspace && ./components/src/dynamo/taper_router/run_smoketest_1gpu.sh $policy > /workspace/sweep_logs/$name.log 2>&1")
        if ! wait_for 600 stack_ready; then
            echo "stack for $name never became ready; see $REPO/sweep_logs/$name.log" >&2
            podman stop -t 10 "$cid" >/dev/null || true
            exit 1
        fi
        sleep 5
        python3 server_gate_ab.py run --label "$cfg" --seed "$seed" --base-url "$URL" \
            --model "$MODEL" --out "$OUT/$name.jsonl" \
            --frontend-log "$REPO/sweep_logs/$name.log" || {
            podman stop -t 10 "$cid" >/dev/null || true
            echo "run $name failed; stopping the sweep" >&2; exit 1; }
        podman stop -t 10 "$cid" >/dev/null || true
        wait_for 120 gpu_idle || { echo "GPU still busy after $name" >&2; exit 1; }
        wait_for 60 port_free || { echo "port 8100 still bound after $name" >&2; exit 1; }
    done
done

python3 server_gate_ab.py table "$OUT"/*.jsonl
