# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit OpenCode / oh-my-opencode configs that point every agent at one Dynamo model.

    opencode_config.py opencode --base-url http://127.0.0.1:8100/v1 --model MiniMaxAI/MiniMax-M2 \\
        [--omo-version 5.1.32]          # -> opencode.json (Harbor --ak opencode_config=...)
    opencode_config.py omo --model MiniMaxAI/MiniMax-M2   # -> oh-my-opencode.json

The provider follows Dynamo's agent-harness guide (OpenAI-compatible provider
"dynamo"). For oh-my-opencode every built-in agent and category is mapped to the
same model (its defaults name frontier models we don't serve), and ultrawork is
on by default through its documented ``default_mode`` switch, so the task prompt
stays unchanged. Background concurrency is left at the plugin default (5).
"""

from __future__ import annotations

import argparse
import json

# oh-my-opencode 5.1.32 built-ins (dist/oh-my-opencode.schema.json, CATEGORY_MODEL_REQUIREMENTS).
OMO_AGENTS = ["build", "plan", "sisyphus", "hephaestus", "sisyphus-junior", "OpenCode-Builder",
              "prometheus", "metis", "momus", "oracle", "librarian", "explore",
              "multimodal-looker", "atlas"]
OMO_CATEGORIES = ["visual-engineering", "ultrabrain", "deep-low", "deep-high", "artistry", "quick",
                  "unspecified-low", "unspecified-high", "writing"]


def opencode(base_url: str, model: str, omo_version: str | None, context: int, output: int) -> dict:
    cfg = {
        "provider": {
            "dynamo": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Dynamo",
                "options": {"baseURL": base_url, "apiKey": "dynamo-local"},
                "models": {model: {"id": model, "name": model, "tool_call": True,
                                   "limit": {"context": context, "output": output}}},
            }
        },
        "model": f"dynamo/{model}",
        "small_model": f"dynamo/{model}",
        "permission": {"task": "allow"},
        "autoupdate": False,
        "share": "disabled",
    }
    if omo_version:
        cfg["plugin"] = [f"oh-my-opencode@{omo_version}"]
    return cfg


def omo(model: str) -> dict:
    ref = f"dynamo/{model}"
    return {
        "agents": {a: {"model": ref} for a in OMO_AGENTS},
        "categories": {c: {"model": ref} for c in OMO_CATEGORIES},
        "default_mode": {"ultrawork": True},
        "auto_update": False,
        "telemetry": False,
        "tmux": {"enabled": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["opencode", "omo"])
    ap.add_argument("--base-url")
    ap.add_argument("--model", required=True)
    ap.add_argument("--omo-version")
    ap.add_argument("--context", type=int, default=196608)
    ap.add_argument("--output", type=int, default=32768)
    a = ap.parse_args()
    if a.kind == "opencode":
        if not a.base_url:
            ap.error("--base-url is required for opencode")
        print(json.dumps(opencode(a.base_url, a.model, a.omo_version, a.context, a.output)))
    else:
        print(json.dumps(omo(a.model), indent=1))


if __name__ == "__main__":
    main()
