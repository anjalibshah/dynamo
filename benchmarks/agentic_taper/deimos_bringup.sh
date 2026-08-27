#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Durable one-command bring-up of the Agentic TAPER P0 stack on deimos.
# Starts a DETACHED, self-restarting container (survives SSH/tmux death), then
# launches the frontend + Nemotron worker as detached execs inside it.
#
# Usage (from the HOST):
#   export HF_TOKEN=<your token>
#   bash deimos_bringup.sh            # port 8002, Nemotron
#   FRONT_PORT=8003 bash deimos_bringup.sh
#
# Idempotent: re-running recreates the container cleanly. Nothing here is tied to
# your shell — exit/disconnect freely; `docker exec -it taper bash` to work inside.
set -euo pipefail

IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1"
NAME="${CONTAINER_NAME:-taper}"
FRONT_PORT="${FRONT_PORT:-8002}"
NS="${DYN_NAMESPACE:-taper-anjshah}"
SERVED="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
: "${HF_TOKEN:?set HF_TOKEN first}"

echo "[1/4] (re)starting detached container '$NAME' ..."
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus '"device=0"' --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/scratch/anjshah/model-cache:/model-cache \
  -v /mnt/scratch/anjshah/agentic_taper:/workspace/agentic_taper \
  -e HF_HOME=/model-cache -e HF_MODULES_CACHE=/tmp/hf_modules -e HF_TOKEN="$HF_TOKEN" \
  "$IMAGE" sleep infinity

echo "[2/4] launching frontend (port $FRONT_PORT) ..."
docker exec -d "$NAME" bash -c "cd /workspace/agentic_taper && \
  DYN_NAMESPACE=$NS NO_COLOR=1 python3 -m dynamo.frontend --trust-remote-code \
  --http-port $FRONT_PORT > /tmp/frontend.log 2>&1"

echo "[3/4] launching Nemotron worker ..."
docker exec -d "$NAME" bash -c "cd /workspace/agentic_taper && DYN_NAMESPACE=$NS \
  python3 -m dynamo.vllm \
    --model $SERVED --served-model-name $SERVED \
    --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 --max-num-batched-tokens 32768 \
    --enable-prefix-caching --async-scheduling --quantization modelopt_fp4 \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
    --mamba-cache-mode align --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --dyn-tool-call-parser nemotron_nano --dyn-reasoning-parser nemotron_nano \
    --reasoning-parser nemotron_v3 \
    --speculative-config '{\"method\":\"dspark\",\"model\":\"$SERVED-DSpark\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}' \
    --disaggregation-mode agg > /tmp/worker.log 2>&1"

echo "[4/4] waiting for worker to register (Ctrl-C to stop waiting) ..."
for i in $(seq 1 60); do
  if curl -s "http://localhost:$FRONT_PORT/v1/models" 2>/dev/null | grep -q NVFP4; then
    echo "  ✅ worker ready — model served on port $FRONT_PORT"
    echo
    echo "Run experiments with:"
    echo "  docker exec -it $NAME bash -c 'cd /workspace/agentic_taper && DYN_NAMESPACE=$NS \\"
    echo "    python3 experiment.py --models nemotron-3.5-lightning-30b-a3b \\"
    echo "      --base-url http://localhost:$FRONT_PORT --frontend-log /tmp/frontend.log \\"
    echo "      --outdir ./results_h1_r5 --limit 18 --reps 5 --max-concurrency 128 \\"
    echo "      --n-tasks 120 --words-per-block 32'"
    exit 0
  fi
  echo "  waiting ($i)... (worker loading; check: docker exec $NAME tail -20 /tmp/worker.log)"
  sleep 10
done
echo "  ⚠️ worker not ready after 10 min — inspect: docker exec $NAME bash -c 'tail -40 /tmp/worker.log'"
exit 1
