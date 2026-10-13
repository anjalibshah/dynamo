# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAPER router CLI parsing and config assembly. Mirrors thunderagent_router/args.py."""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.router.args import (
    DynamoRouterArgGroup,
    DynamoRouterConfig,
    build_ais_perf_config,
    build_kv_router_config,
)
from dynamo.taper_router.agentic_gate import MODES, TaperModeConfig
from dynamo.taper_router.gate import TaperConfig


class TaperRouterConfig(DynamoRouterConfig):
    """Extends the standalone-router config with the Agentic TAPER gate params."""

    load_threshold: float
    reconcile_interval_seconds: float
    defer_timeout_seconds: float
    shadow_mode: bool
    admit_settle_seconds: float
    max_release_per_tick: int
    context_budget_tokens: float = 0.0
    taper_mode: str = "legacy"
    cap_k: int = 1
    rho: float = 0.8
    tpot_interactive_ms: float = 18.48
    tpot_agent_ms: float = 21.63
    predictor_coeffs: str = "0,0,0"
    online_refit: bool = True
    max_hold_seconds: float = 60.0
    interactive_session_prefix: str = "victim-"
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
            admit_settle_seconds=self.admit_settle_seconds,
            max_release_per_tick=self.max_release_per_tick,
            context_budget_tokens=self.context_budget_tokens,
        )

    def parsed_coeffs(self) -> tuple[float, float, float]:
        parts = [float(x) for x in self.predictor_coeffs.split(",")]
        if len(parts) != 3:
            raise ValueError("--predictor-coeffs must be 'a,b,c' (ms, ms/request, ms/token)")
        return parts[0], parts[1], parts[2]

    def to_mode_config(self) -> TaperModeConfig:
        a, b, c = self.parsed_coeffs()
        return TaperModeConfig(
            mode=self.taper_mode,
            cap_k=self.cap_k,
            rho=self.rho,
            tpot_interactive_ms=self.tpot_interactive_ms,
            tpot_agent_ms=self.tpot_agent_ms,
            pred_a_ms=a,
            pred_b_ms=b,
            pred_c_ms=c,
            online_refit=self.online_refit,
            permit_window_seconds=self.admit_settle_seconds,
            max_hold_seconds=self.max_hold_seconds,
            reconcile_interval_seconds=self.reconcile_interval_seconds,
            max_release_per_tick=self.max_release_per_tick,
            shadow_mode=self.shadow_mode,
        )

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.taper_mode not in ("legacy",) + MODES:
            raise ValueError(f"--taper-mode must be one of legacy, {', '.join(MODES)}")
        if self.cap_k < 1:
            raise ValueError("--cap-k must be >= 1")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("--rho must be in [0, 1]")
        if self.max_hold_seconds <= 0:
            raise ValueError("--max-hold-seconds must be > 0")
        self.parsed_coeffs()
        if self.load_threshold <= 0:
            raise ValueError("--load-threshold must be > 0")
        if self.admit_settle_seconds < 0:
            raise ValueError("--admit-settle-seconds must be >= 0")
        if self.context_budget_tokens < 0:
            raise ValueError("--context-budget-tokens must be >= 0")
        if self.max_release_per_tick < 1:
            raise ValueError("--max-release-per-tick must be >= 1")
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
            flag_name="--admit-settle-seconds",
            env_var="DYN_TAPER_ADMIT_SETTLE_SECONDS",
            default=0.5,
            help="Count each branch admission as extra load for this long, "
            "covering the lag before it shows up in forward-pass metrics "
            "(default: 0.5).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--max-release-per-tick",
            env_var="DYN_TAPER_MAX_RELEASE_PER_TICK",
            default=4,
            help="Most deferred branches released per reconcile tick, so a "
            "backlog drains gradually instead of in one burst (default: 4).",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--context-budget-tokens",
            env_var="DYN_TAPER_CONTEXT_BUDGET_TOKENS",
            default=0.0,
            help="Also defer an opportunistic sibling while the busiest "
            "worker's live decode context (sum_decode_kv_tokens) plus the "
            "sibling's prompt length would exceed this many tokens. Set it "
            "to the context at which step time reaches the interactive ITL "
            "SLO (fpm_probe.py summarize reports it). 0 disables (default: 0).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--taper-mode",
            env_var="DYN_TAPER_MODE",
            default="legacy",
            help="Admission policy: legacy (the original trunk-protected count/"
            "context gate, gate.py), eager (admit everything, telemetry only), "
            "cap (at most --cap-k in-flight requests per task), taper (one "
            "protected request per task plus slack-budgeted opportunistic "
            "admission, agentic_gate.py) (default: legacy).",
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--cap-k",
            env_var="DYN_TAPER_CAP_K",
            default=1,
            help="cap mode: in-flight requests allowed per task (default: 1).",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--rho",
            env_var="DYN_TAPER_RHO",
            default=0.8,
            help="taper mode: fraction of the slack budget opportunistic "
            "admissions may consume (paper default: 0.8).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--tpot-interactive-ms",
            env_var="DYN_TAPER_TPOT_INTERACTIVE_MS",
            default=18.48,
            help="taper mode: per-token target while interactive requests "
            "are in flight (default: 18.48).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--tpot-agent-ms",
            env_var="DYN_TAPER_TPOT_AGENT_MS",
            default=21.63,
            help="taper mode: per-token target otherwise (default: 21.63).",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--predictor-coeffs",
            env_var="DYN_TAPER_PREDICTOR_COEFFS",
            default="0,0,0",
            help="taper mode: step-time model T = a + b*n + c*L as 'a,b,c' "
            "(ms; n = decode requests, L = decode context tokens). All zero "
            "means uncalibrated: admit until the online refit has data.",
            arg_type=str,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--online-refit",
            env_var="DYN_TAPER_ONLINE_REFIT",
            default=True,
            help="taper mode: refit the step-time model from live forward-pass "
            "metrics, as PACE does (default: true).",
        )
        add_argument(
            g,
            flag_name="--max-hold-seconds",
            env_var="DYN_TAPER_MAX_HOLD_SECONDS",
            default=60.0,
            help="eager/cap/taper modes: longest a request is held before a "
            "forced admit (default: 60.0). legacy uses --defer-timeout-seconds.",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--interactive-session-prefix",
            env_var="DYN_TAPER_INTERACTIVE_SESSION_PREFIX",
            default="victim-",
            help="Session ids with this prefix are interactive traffic: they "
            "select the tighter per-token target (default: victim-).",
            arg_type=str,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--shadow-mode",
            env_var="DYN_TAPER_SHADOW_MODE",
            default=True,
            help="P0-4: compute the real admit/defer decision and emit it as "
            "a counter, but always actually admit -- proves the decision "
            "logic on real traffic with zero behavioral risk. Use "
            "--no-shadow-mode (or DYN_TAPER_SHADOW_MODE=false) to flip on "
            "live gating (default: true).",
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
    "build_ais_perf_config",
    "build_kv_router_config",
    "parse_args",
]
