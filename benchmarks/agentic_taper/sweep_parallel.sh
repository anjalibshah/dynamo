#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load sweep (NEXT_STEPS 7.3i/7.3j): independent stacks run in parallel, one per
# slot, each with its own arm and agent concurrency. Slot s uses GPUs
# s*G .. s*G+G-1 (G = GPUS_PER_SLOT, the TP size) and PORT_OFFSET=100*s.
# MiniMax-M2 TP4: GPUS_PER_SLOT=4 SUITE=-mm MODEL_PATH=MiniMaxAI/MiniMax-M2.7
#   MODEL=MiniMaxAI/MiniMax-M2 TOOL_PARSER=minimax_m2
#   REASONING_PARSER=minimax_append_think EXTRA_VLLM_ARGS="--kv-cache-dtype fp8"
#
#   ./sweep_parallel.sh <sweep-tag> "<arm>:<agents> ..."     # one spec per GPU
#   ./sweep_parallel.sh s1 "kv:6 ctx:8 kv:8 ctx:10 kv:10 ctx:12 kv:12 ctx:14"
#   python3 sweep_report.py --sweep s1 [--suite=-mm]            # afterwards
#
# arm: kv (stock KV router) or ctx (taper_router, count cap 7 + 300k context
# budget, unchanged from the 7.3f ctx arm), or a milestone-1 arm (NEXT_STEPS
# 7.3m) on taper_router's agentic gate:
#   eager   stock KV router (the pre-registered "eager" baseline; same as kv)
#   pass    router mode eager: router hop + telemetry, no holding
#   cap1    at most 1 in-flight request per task (fan-out serialized)
#   cap2    at most 2 in-flight requests per task
#   taper   one protected request per task + slack admission (needs PRED_COEFFS)
#   shadow  taper decisions counted but never applied (pilot)
# or an engine-level arm (ENGINE_AB_RUNBOOK.md):
#   prio    stock, plus vLLM --scheduling-policy priority (Dynamo's existing
#           priority feature; needs VICTIM_PRIORITY)
#   cb      class-blind per-step decode regulation (TaperScheduler,
#           DYN_TAPER_ENGINE=class_blind) classifying by priority: requests with
#           priority >= PRIORITY_MIN get the ITL target (needs VICTIM_PRIORITY)
# VICTIM_PRIORITY stamps nvext.agent_hints.priority on every interactive
# request (set it for every arm of a comparison so request bodies match).
# MOUNT_VLLM=1 mounts this checkout's components/src/dynamo/vllm over the
# image's, so every arm runs the same backend code (needed for cb on an image
# built before TaperScheduler).
# AGENT selects the coding agent (Harbor -a): claude-code (default), opencode
# (OpenCode, pinned OC_VERSION, default prompt and subagents), or omo (OpenCode
# plus the oh-my-opencode plugin, pinned OMO_VERSION, ultrawork on by default
# via its config so no prompt keyword is needed; every agent/category -> MODEL).
# Every slot gets the same workload as
# the 1-GPU comparisons: Claude Code + v2 subagent instruction on SWE-bench
# Verified, N_TASKS tasks, natural behavior unless INSTRUCTION is set, an
# interactive client at VICTIM_MEAN_MS. Runs are
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
# Step-time model "a,b,c" (ms; T = a + b*n + c*L) for the taper/shadow arms,
# from fit_step_model.py on the calibration runs.
PRED_COEFFS="${PRED_COEFFS:-}"
VICTIM_PRIORITY="${VICTIM_PRIORITY:-}"
PRIORITY_MIN="${PRIORITY_MIN:-1}"
MOUNT_VLLM="${MOUNT_VLLM:-0}"
# Engine step-time model for the cb arm (empty: TaperConfig defaults, fitted on
# MiniMax-M2 TP4 H200): "a,b,c" decode and "q,p" prefill, in ms (p per token).
ENGINE_COEFFS="${ENGINE_COEFFS:-}"
ENGINE_PREFILL_COEFFS="${ENGINE_PREFILL_COEFFS:-17.5,0.035}"
AGENT="${AGENT:-claude-code}"
OC_VERSION="${OC_VERSION:-1.19.0}"     # 1.19.0 sends each subagent's own x-session-id
OMO_VERSION="${OMO_VERSION:-5.1.32}"
case "$AGENT" in claude-code|opencode|omo) ;; *) echo "bad AGENT=$AGENT" >&2; exit 2 ;; esac
# Natural agent behavior by default. Set INSTRUCTION to a file (e.g.
# $REPO/benchmarks/agentic_taper/harbor_parallel_subagents_v2.md) to induce
# parallel subagents, as the GLM runs did.
INSTRUCTION="${INSTRUCTION:-}"
INSTR_ARGS=()
if [[ -n "$INSTRUCTION" ]]; then
    [[ -f "$INSTRUCTION" ]] || { echo "INSTRUCTION file $INSTRUCTION not found" >&2; exit 2; }
    INSTR_ARGS=(--extra-instruction-path "$INSTRUCTION")
fi
GPUS_PER_SLOT="${GPUS_PER_SLOT:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
SUITE="${SUITE:-}"
JP="harbor-ab$SUITE"
OUT_REL="harbor_ab$SUITE"
OUT="$REPO/$OUT_REL"
SITE=/usr/local/lib/python3.12/dist-packages/dynamo/taper_router
SITE_VLLM=/usr/local/lib/python3.12/dist-packages/dynamo/vllm
PY="${PY:-$REPO/.venv/bin/python3}"   # any python3 with aiohttp
"$PY" -c 'import aiohttp' 2>/dev/null || {
    echo "PY=$PY cannot import aiohttp (the interactive client needs it); set PY" >&2; exit 2; }
# vLLM TP>1 workers share state via /dev/shm; podman's 64 MB default is too
# small (MiniMax-M2 TP4 needs >160 MB). Harmless for TP1.
SHM_SIZE="${SHM_SIZE:-32g}"

SOCK="${XDG_RUNTIME_DIR:?source ~/.bashrc first}/podman/podman.sock"
[[ -S "$SOCK" ]] || { echo "podman API socket $SOCK not running" >&2; exit 2; }
read -r -a SPEC <<<"$SPECS"
# GPU_IDS: comma-separated host GPU indices (or UUIDs) the slots use, in
# order; default all GPUs from 0. Set it when another job shares the node,
# e.g. GPU_IDS=4,5,6,7 (check `nvidia-smi` for the GPUs in use).
if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=, read -r -a GPU_LIST <<<"$GPU_IDS"
else
    mapfile -t GPU_LIST < <(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))
fi
NGPU=${#GPU_LIST[@]}
(( ${#SPEC[@]} * GPUS_PER_SLOT <= NGPU )) || { echo "${#SPEC[@]} slots x $GPUS_PER_SLOT GPUs > $NGPU GPUs" >&2; exit 2; }
IN_GPUS=$(seq -s, 0 $((GPUS_PER_SLOT - 1)))
mkdir -p "$OUT"

# Ready, or stop waiting early on a fatal worker startup error.
stack_ready() {
    if grep -qE "Engine core initialization failed|Insufficient space in /dev/shm|CUDA out of memory|No such file or directory: .*config.json" "$1" 2>/dev/null; then
        echo "fatal error in $1:" >&2; grep -m3 -E "Engine core initialization failed|Insufficient space in /dev/shm|CUDA out of memory|No such file or directory: .*config.json" "$1" >&2; return 2
    fi
    grep -q "smoketest stack ready" "$1" 2>/dev/null
}
wait_for() { local d=$((SECONDS + $1)); shift; local rc; while :; do "$@"; rc=$?; (( rc == 0 )) && return 0; (( rc == 2 )) && return 1; (( SECONDS < d )) || return 1; sleep 5; done; }

# Fail fast if any port a stack binds is already taken (e.g. node_exporter owns
# 9100 on the cluster nodes, which is why stacks are offset by 100, not 1000).
ports_free() {  # ports_free <offset>
    local po=$1 p busy=()
    for p in $((8100 + po)) $((8181 + po)) $((8183 + po)) $((8184 + po)) \
             $((20080 + po)) $((20081 + po)) $((20097 + po)); do
        ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && busy+=("$p")
    done
    (( ${#busy[@]} == 0 )) || { echo "ports already in use for offset $po: ${busy[*]}" >&2; return 1; }
}
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

echo "=== sweep $TAG ($(date -u +%H:%M:%S)): $SPECS  [instruction: ${INSTRUCTION:-none (natural)}]"
for s in "${!SPEC[@]}"; do
    arm=${SPEC[$s]%%:*}; n=${SPEC[$s]##*:}
    case "$arm" in kv|ctx|eager|pass|cap1|cap2|taper|shadow|prio|cb) ;;
        *) echo "bad arm in ${SPEC[$s]}" >&2; exit 2 ;; esac
    if [[ ( "$arm" == taper || "$arm" == shadow ) && -z "$PRED_COEFFS" ]]; then
        echo "arm $arm needs PRED_COEFFS=a,b,c" >&2; exit 2
    fi
    if [[ ( "$arm" == prio || "$arm" == cb ) && -z "$VICTIM_PRIORITY" ]]; then
        echo "arm $arm needs VICTIM_PRIORITY (e.g. 1)" >&2; exit 2
    fi
    if [[ "$arm" == cb ]] && (( VICTIM_PRIORITY < PRIORITY_MIN )); then
        echo "arm cb: VICTIM_PRIORITY=$VICTIM_PRIORITY < PRIORITY_MIN=$PRIORITY_MIN would regulate the interactive requests" >&2; exit 2
    fi
    run="sw$TAG-$arm-n$n"; RUN[$s]=$run; PORT[$s]=$((8100 + 100 * s))
    if [[ -d "$REPO/jobs/$JP-$run" ]]; then echo "slot $s: skip $run (done)"; RUN[$s]=""; continue; fi
    ports_free $((100 * s)) || { cleanup; exit 2; }
    mkdir -p "$OUT/$run"; rm -f "$OUT/$run/stack.log" "$OUT/$run/trace.jsonl" "$OUT/$run/fpm.jsonl"
    extra=()
    policy=kv
    vllm_args="$EXTRA_VLLM_ARGS"
    case "$arm" in
        ctx) policy=taper
             extra=(-e "DYN_TAPER_LOAD_THRESHOLD=$CTX_THRESHOLD" -e "DYN_TAPER_CONTEXT_BUDGET_TOKENS=$CTX_BUDGET") ;;
        pass) policy=taper; extra=(-e DYN_TAPER_MODE=eager) ;;
        cap1|cap2) policy=taper; extra=(-e DYN_TAPER_MODE=cap -e "DYN_TAPER_CAP_K=${arm#cap}") ;;
        taper|shadow) policy=taper
             extra=(-e DYN_TAPER_MODE=taper -e "DYN_TAPER_PREDICTOR_COEFFS=$PRED_COEFFS")
             [[ "$arm" == shadow ]] && extra+=(-e DYN_TAPER_SHADOW_MODE=true) ;;
        prio) vllm_args="$vllm_args --scheduling-policy priority" ;;
        cb) extra=(-e DYN_TAPER_ENGINE=class_blind -e "DYN_TAPER_PRIORITY_MIN=$PRIORITY_MIN"
                   -e "DYN_TAPER_PREFILL_COEFFS=$ENGINE_PREFILL_COEFFS"
                   -e "DYN_TAPER_TRACE=/workspace/$OUT_REL/$run/taper_engine.jsonl")
            [[ -n "$ENGINE_COEFFS" ]] && extra+=(-e "DYN_TAPER_COEFFS=$ENGINE_COEFFS") ;;
    esac
    (( MOUNT_VLLM )) && extra+=(-v "$REPO/components/src/dynamo/vllm:$SITE_VLLM")
    # This slot's GPUs as CDI devices; inside the container they are 0..G-1.
    dev=()
    for (( g = s * GPUS_PER_SLOT; g < (s + 1) * GPUS_PER_SLOT; g++ )); do dev+=(--device "nvidia.com/gpu=${GPU_LIST[$g]}"); done
    CID[$s]=$(podman run -d --rm --init --network host "${dev[@]}" --shm-size "$SHM_SIZE" \
        -v "$REPO:/workspace" -v "$REPO/components/src/dynamo/taper_router:$SITE" \
        -v "$HF:$HF" -e "HF_HOME=$HF" -e NO_COLOR=1 \
        -e "MODEL_PATH=$MODEL_PATH" -e "MODEL_NAME_ROUTER=$MODEL" -e "GPUS=$IN_GPUS" -e "TP=$GPUS_PER_SLOT" \
        -e "EXTRA_VLLM_ARGS=$vllm_args" \
        -e "TOOL_PARSER=$TOOL_PARSER" -e "REASONING_PARSER=$REASONING_PARSER" -e ANTHROPIC_API=1 \
        -e "PORT_OFFSET=$((100 * s))" \
        -e DYN_REQUEST_TRACE=1 -e DYN_REQUEST_TRACE_SINKS=jsonl \
        -e "DYN_REQUEST_TRACE_OUTPUT_PATH=/workspace/$OUT_REL/$run/trace.jsonl" \
        ${extra[@]+"${extra[@]}"} "$IMAGE" \
        bash -c "cd /workspace && ./components/src/dynamo/taper_router/run_smoketest_1gpu.sh $policy > /workspace/$OUT_REL/$run/stack.log 2>&1")
    echo "slot $s: $run on GPUs ${GPU_LIST[*]:$((s * GPUS_PER_SLOT)):GPUS_PER_SLOT}, port ${PORT[$s]}"
done

for s in "${!SPEC[@]}"; do
    run=${RUN[$s]}; [[ -z "$run" ]] && continue
    if ! wait_for 3600 stack_ready "$OUT/$run/stack.log"; then
        echo "slot $s ($run) never became ready; see $OUT/$run/stack.log" >&2; cleanup; exit 1
    fi
done
echo "all stacks ready ($(date -u +%H:%M:%S))"
sleep 5

for s in "${!SPEC[@]}"; do
    run=${RUN[$s]}; [[ -z "$run" ]] && continue
    n=${SPEC[$s]##*:}; url="http://127.0.0.1:${PORT[$s]}"
    podman exec -d "${CID[$s]}" bash -c "python3 /workspace/benchmarks/agentic_taper/fpm_probe.py \
        record --port $((20081 + 100 * s)) --out /workspace/$OUT_REL/$run/fpm.jsonl \
        > /workspace/$OUT_REL/$run/fpm_probe.log 2>&1"
    "$PY" "$REPO/benchmarks/agentic_taper/victim_client.py" --base-url "$url" --model "$MODEL" \
        --arm "$run" --out "$OUT/$run/victims.jsonl" --n-victims 100000 \
        --mean-arrival-ms "$VICTIM_MEAN_MS" --seed 0 \
        ${VICTIM_PRIORITY:+--priority "$VICTIM_PRIORITY"} > "$OUT/$run/victim_client.log" 2>&1 &
    VPID[$s]=$!
    if [[ "$AGENT" == claude-code ]]; then
        agent_args=(-a claude-code -m "$MODEL"
            --ae "ANTHROPIC_BASE_URL=$url" --ae ANTHROPIC_API_KEY=dynamo-local
            --ae "ANTHROPIC_MODEL=$MODEL" --ae "ANTHROPIC_SMALL_FAST_MODEL=$MODEL"
            --ae "CLAUDE_CODE_SUBAGENT_MODEL=$MODEL" --ae CLAUDE_CODE_ATTRIBUTION_HEADER=0)
    else
        # OpenCode talks OpenAI chat completions to the frontend; Dynamo maps its
        # x-session-id / x-parent-session-id headers to the task identity.
        oc_cfg=$("$PY" "$REPO/benchmarks/agentic_taper/opencode_config.py" opencode \
            --base-url "$url/v1" --model "$MODEL" \
            $( [[ "$AGENT" == omo ]] && echo --omo-version "$OMO_VERSION" ))
        agent_args=(-a opencode -m "dynamo/$MODEL" --ak "version=$OC_VERSION"
            --ak "opencode_config=$oc_cfg")
        if [[ "$AGENT" == omo ]]; then
            "$PY" "$REPO/benchmarks/agentic_taper/opencode_config.py" omo --model "$MODEL" \
                > "$OUT/$run/oh-my-opencode.json"
            agent_args+=(--mounts "[\"$OUT/$run/oh-my-opencode.json:/root/.config/opencode/oh-my-opencode.json:ro\"]")
        fi
    fi
    (cd "$REPO" && "$HARBOR" run -d swebench-verified@1.0 "${agent_args[@]}" \
        --extra-docker-compose "$PLUGINS/pi-plugin/harbor/host-network.yml" \
        --job-name "$JP-$run" -y -l "$N_TASKS" -n "$n" ${INSTR_ARGS[@]+"${INSTR_ARGS[@]}"} \
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
    if [[ "$run" == *-cb-* ]]; then
        log=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$run/stack.log")
        gate=" engine: $(grep -o 'taper.engine steps=.*' <<<"$log" | tail -1 || true)"
        grep -q 'taper.engine disabled' <<<"$log" && gate+=" [DISABLED: $(grep -m1 -o 'taper.engine disabled.*' <<<"$log")]"
    fi
    echo "slot $s: $run finished ($(date -u +%H:%M:%S))$gate"
done
trap - INT TERM
echo "sweep $TAG done. Next: python3 $REPO/benchmarks/agentic_taper/sweep_report.py --sweep $TAG"
