# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAPER router CLI parsing and config assembly. Mirrors thunderagent_router/args.py."""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.utils import add_argument
from dynamo.router.args import (
    DynamoRouterArgGroup,
    DynamoRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
)
from dynamo.taper_router.gate import TaperConfig


class TaperRouterConfig(DynamoRouterConfig):
    """Extends the standalone-router config with the Agentic TAPER gate params."""

    load_threshold: float
    reconcile_interval_seconds: float
    defer_timeout_seconds: float
    shadow_mode: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None

    def to_taper_config(self) -> TaperConfig:
        return TaperConfig(
            load_threshold=self.load_threshold,
            reconcile_interval_seconds=self.reconcile_interval_seconds,
            defer_timeout_seconds=self.defer_timeout_seconds,
            shadow_mode=self.shadow_mode,
        )

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.load_threshold <= 0:
            raise ValueError("--load-threshold must be > 0")
        if self.reconcile_interval_seconds <= 0:
            raise ValueError("--reconcile-interval-seconds must be > 0")
        if self.defer_timeout_seconds <= 0:
            raise ValueError("--defer-timeout-seconds must be > 0")


class TaperArgGroup(ArgGroup):
    name = "taper-router"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Inherit standard router options (--endpoint, --router-block-size, KV
        # router knobs, AicPerf options).
        DynamoRouterArgGroup().add_arguments(parser)

        g = parser.add_argument_group("Agentic TAPER Gate Options")

        add_argument(
            g,
            flag_name="--load-threshold",
            env_var="DYN_TAPER_LOAD_THRESHOLD",
            default=32.0,
            help="Admit an opportunistic sibling immediately while the busiest "
            "worker's live num_decode_requests is <= this value; defer above "
            "it (default: 32.0). The core tuning curve -- sweep per brief "
            "Appendix D.",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--reconcile-interval-seconds",
            env_var="DYN_TAPER_RECONCILE_INTERVAL_SECONDS",
            default=0.05,
            help="Period of the background loop that retries releasing "
            "deferred siblings. Matches FPM freshness (default: 0.05, i.e. "
            "50ms; sweep {0.02, 0.05, 0.1}).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--defer-timeout-seconds",
            env_var="DYN_TAPER_DEFER_TIMEOUT_SECONDS",
            default=300.0,
            help="Maximum wait on a deferred opportunistic sibling before a "
            "forced admit, so a starved sibling can't wait forever if load "
            "never drops (default: 300.0).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--shadow-mode",
            env_var="DYN_TAPER_SHADOW_MODE",
            default=True,
            help="P0-4: compute the real admit/defer decision and emit it as "
            "a counter, but always actually admit -- proves the decision "
            "logic on real traffic with zero behavioral risk. Set to false "
            "(--no-shadow-mode) to flip on live gating (default: true).",
            arg_type=bool,
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_TAPER_MODEL_NAME",
            default=None,
            help="Model name to register at the Dynamo frontend. When set the "
            "router calls register_model so the frontend dispatches "
            "requests for this model to the router. Leave unset to behave "
            "as a pure utility endpoint with no frontend registration.",
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_TAPER_MODEL_PATH",
            default=None,
            help="Path or HF repo ID to load tokenizer + model card from for "
            "register_model. Defaults to --model-name.",
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--dyn-tool-call-parser",
            dest="tool_call_parser",
            env_var="DYN_TOOL_CALL_PARSER",
            default=None,
            help="Tool-call parser forwarded to register_model. Only applies "
            "when --model-name is set.",
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--dyn-reasoning-parser",
            dest="reasoning_parser",
            env_var="DYN_REASONING_PARSER",
            default=None,
            help="Reasoning parser forwarded to register_model. Only applies "
            "when --model-name is set.",
            arg_type=str,
        )


def parse_args(argv: Optional[list[str]] = None) -> TaperRouterConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo Agentic TAPER Router: task-aware admission gate "
        "(protected + opportunistic width) on top of native KV-aware routing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    TaperArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = TaperRouterConfig.from_cli_args(args)
    config.validate()
    return config


__all__ = [
    "TaperArgGroup",
    "TaperRouterConfig",
    "build_aic_perf_config",
    "build_kv_router_config",
    "parse_args",
]
