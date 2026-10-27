#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end smoke test of decode regulation under the vLLM sidecar, one arm per
# call: frontend + vllm-rs engine + Dynamo sidecar in one container, synthetic
# background load (no priority, regulated) plus interactive probes (priority 1).
#
#   sidecar_smoke.sh observe        # baseline: decides, never skips; writes the trace
#   sidecar_smoke.sh class_blind    # regulates (set COEFFS from the observe fit)
#   sidecar_smoke.sh off            # stock vLLM scheduler
#
# Required env: IMAGE (built from feat/sidecar-decode-regulation-eval), HF, OUT_ROOT, PY
# (a Python with aiohttp). Optional: MODEL, GPU (CDI index), COEFFS, DEADLINE,
# DURATION_S, BG_CONC, BG_PROMPT_TOKENS, BG_MAX_TOKENS, MAX_MODEL_LEN, CONTAINER_USER.
set -euo pipefail

ARM="${1:?usage: sidecar_smoke.sh off|observe|class_blind}"
: "${IMAGE:?}" "${HF:?}" "${OUT_ROOT:?}" "${PY:?}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU="${GPU:-0}"
DEADLINE="${DEADLINE:-500,18.48}"
COEFFS="${COEFFS:-}"
DURATION_S="${DURATION_S:-480}"
BG_CONC="${BG_CONC:-48}"
BG_PROMPT_TOKENS="${BG_PROMPT_TOKENS:-12000}"
BG_MAX_TOKENS="${BG_MAX_TOKENS:-1500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$OUT_ROOT/$ARM"
mkdir -p "$OUT" && chmod 777 "$OUT"

env_args=(-e "HF_HOME=$HF" -e DYN_DISCOVERY_BACKEND=file -e NO_COLOR=1)
sched_args=""
if [[ "$ARM" != off ]]; then
    env_args+=(-e "DYN_DECODE_REGULATION_MODE=$ARM" -e DYN_DECODE_REGULATION_PRIORITY_MIN=1
               -e "DYN_DECODE_REGULATION_DEADLINE=$DEADLINE"
               -e DYN_DECODE_REGULATION_METRICS_PORT=9400
               -e DYN_DECODE_REGULATION_TRACE=/out/trace.jsonl)
    [[ -n "$COEFFS" ]] && env_args+=(-e "DYN_DECODE_REGULATION_COEFFS=$COEFFS")
    sched_args="--scheduler-cls dynamo.vllm.decode_regulation_sidecar.SidecarDecodeRegulationScheduler"
fi
user=(); [[ -n "${CONTAINER_USER:-}" ]] && user=(--user "$CONTAINER_USER")

CID=$(podman run -d --rm --init --network host ${user[@]+"${user[@]}"} \
    --device "nvidia.com/gpu=$GPU" --shm-size 16g \
    -v "$HF:$HF" -v "$OUT:/out" "${env_args[@]}" "$IMAGE" bash -c "
set -x
VLLM_RS=\$(python3 -c 'import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), \"vllm-rs\"))')
python3 -c 'import vllm; print(\"vllm\", vllm.__version__)'
ls -l \"\$VLLM_RS\"
python3 -m dynamo.frontend --http-port 8000 > /out/frontend.log 2>&1 &
\"\$VLLM_RS\" serve $MODEL --host 127.0.0.1 --port 8100 --grpc-port 30001 \
    --max-model-len $MAX_MODEL_LEN --reasoning-parser none -- \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 $sched_args > /out/engine.log 2>&1 &
python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:30001 > /out/sidecar.log 2>&1 &
wait -n
echo 'a process exited'
")
podman logs -f "$CID" > "$OUT/container.log" 2>&1 &
stop() { podman stop -t 10 "$CID" >/dev/null 2>&1 || true; }
trap stop EXIT
echo "$ARM: container ${CID:0:12}, logs in $OUT"

# Wait for the model to be served (first run downloads the weights).
SERVED=""
for _ in $(seq 360); do
    podman container exists "$CID" || { echo "container exited; see $OUT/*.log"; exit 1; }
    SERVED=$(curl -sf localhost:8000/v1/models 2>/dev/null \
        | "$PY" -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "")' 2>/dev/null || true)
    [[ -n "$SERVED" ]] && break
    sleep 5
done
[[ -n "$SERVED" ]] || { echo "model not served after 30 min; see $OUT/*.log"; exit 1; }
echo "$ARM: serving $SERVED"
curl -sf localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}" \
    > "$OUT/warmup.json" || { echo "warm-up request failed"; exit 1; }

"$PY" "$HERE/sidecar_bg_load.py" --model "$SERVED" --out "$OUT/bg.jsonl" \
    --concurrency "$BG_CONC" --prompt-tokens "$BG_PROMPT_TOKENS" \
    --max-tokens "$BG_MAX_TOKENS" --duration-s "$DURATION_S" > "$OUT/bg.log" 2>&1 &
BG=$!
sleep 60  # let the background batch fill before probing
"$PY" "$HERE/victim_client.py" --base-url http://127.0.0.1:8000 --model "$SERVED" \
    --arm "$ARM" --out "$OUT/victims.jsonl" --n-victims $(( (DURATION_S - 90) / 2 )) \
    --mean-arrival-ms 2000 --seed 0 --priority 1 > "$OUT/victim_client.log" 2>&1
[[ "$ARM" != off ]] && curl -sf localhost:9400/metrics > "$OUT/metrics.txt" || true
wait "$BG" || true
echo "$ARM: done"
