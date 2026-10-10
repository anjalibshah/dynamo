#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Dynamo vLLM runtime image with rootless podman/buildah, which lacks
# two BuildKit features the rendered Dockerfile uses:
#   * secret mounts exposed as env vars (--mount=type=secret,...,env=VAR): used
#     only for NVIDIA-internal package indexes and the sccache S3 cache. They are
#     removed; the RUN steps then use the public indexes.
#   * file secret mounts must be supplied: each remaining
#     --mount=type=secret,id=X,target=... gets an empty file, which is what an
#     unauthenticated build would see (e.g. an empty netrc).
#
#   PYBIN=/scratch/$USER/pyenv/bin/python ./benchmarks/agentic_taper/podman_build.sh
#
# Env: PYBIN (python3 with pyyaml + jinja2), TAG (default dynamo:latest-vllm-runtime),
# FRAMEWORK (vllm), TARGET (runtime). Extra args are passed to podman build.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYBIN="${PYBIN:-python3}"
TAG="${TAG:-dynamo:latest-vllm-runtime}"
DOCKERFILE=container/rendered.Dockerfile

"$PYBIN" container/render.py --framework "${FRAMEWORK:-vllm}" --target "${TARGET:-runtime}" --output-short-filename
"$PYBIN" - "$DOCKERFILE" <<'EOF'
import re, sys
path = sys.argv[1]
env_secret = re.compile(r"--mount=type=secret,[^\s\\]*\benv=[^\s\\]*\s*")
out, removed = [], 0
for line in open(path).read().split("\n"):
    new, n = env_secret.subn("", line)
    removed += n
    if n:
        body = new.strip()
        if body in ("", "\\"):                # the line held only removed mounts
            continue
        if re.fullmatch(r"RUN\s*\\", body):    # "RUN <mounts> \" -> "RUN \"
            new = "RUN \\"
    out.append(new)
text = "\n".join(out)
if env_secret.search(text):
    sys.exit("env= secret mounts remain")
open(path, "w").write(text)
print(f"removed {removed} env= secret mounts")
EOF

SECRETS=()
SECRET_DIR="${TMPDIR:-/tmp}/dynamo-build-secrets"
mkdir -p "$SECRET_DIR"
for id in $(grep -o -- '--mount=type=secret,id=[^,[:space:]]*' "$DOCKERFILE" | sed 's/.*id=//' | sort -u); do
    : > "$SECRET_DIR/$id"
    SECRETS+=(--secret "id=$id,src=$SECRET_DIR/$id")
done
echo "empty file secrets: ${#SECRETS[@]} args (${SECRETS[*]:-none})"

podman build --network host ${SECRETS[@]+"${SECRETS[@]}"} -t "$TAG" -f "$DOCKERFILE" "$@" .
