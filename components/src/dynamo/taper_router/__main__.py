# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Agentic TAPER router service.

Usage:
    python -m dynamo.taper_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64 \\
        --shadow-mode   # P0-4: decide + count, never actually defer

Serves ``{namespace}.taper_router.generate``. Gating is opt-in per-request via
header-derived task identity; requests without it are routed via plain
KvRouter with no admission control -- the same passthrough contract
``thunderagent_router`` uses, and deliberately so: the two are meant to be
easy to A/B against the same worker pool. Unlike ThunderAgent, TAPER does not
pin requests to a worker -- placement stays native KV-overlap routing's job
(brief: "siblings co-locate via KV-overlap routing on the shared prefix, not
via session affinity"); this router only decides *when* a request is allowed
to dispatch, not *where*.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import uvloop

from dynamo.llm import (
    KvRouter,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.taper_router.args import (
    TaperRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
    parse_args,
)
from dynamo.taper_router.gate import TaperGate
from dynamo.taper_router.load import LoadSnapshotProvider

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _extract_task_and_request_id(request: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(task_id, request_id) from agent_context, or (None, None) if absent.

    task_id is the root task's session_id: ``parent_session_id`` when present
    (a branch/join of a fanned-out task), else the request's own
    ``session_id`` (a root request, or a plain non-fan-out turn -- brief
    Appendix B: task == "no native concept; approximated by the root
    session_id"). request_id is the request's own session_id, since branches
    already carry a distinct id per the brief's P0 identity workaround.
    """
    ctx = request.get("agent_context")
    if not isinstance(ctx, dict):
        return None, None
    session_id = ctx.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None, None
    parent_id = ctx.get("parent_session_id")
    task_id = parent_id if isinstance(parent_id, str) and parent_id else session_id
    return task_id, session_id


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    # Duplicated from dynamo.router/__main__.py (also duplicated in
    # thunderagent_router/__main__.py with the same TODO) since neither
    # package exports it.
    routing = request.get("routing")
    dp_rank = request.get("dp_rank")
    if routing is None and dp_rank is not None:
        routing = {"dp_rank": dp_rank}

    return {
        "model": request.get("model", "unknown"),
        "token_ids": request["token_ids"],
        "stop_conditions": request.get("stop_conditions", {}),
        "sampling_options": request.get("sampling_options", {}),
        "output_options": request.get("output_options", {}),
        "eos_token_ids": request.get("eos_token_ids", []),
        "annotations": request.get("annotations", []),
        "routing": routing,
        "router_config_override": request.get("router_config_override"),
        "prefill_result": request.get("prefill_result"),
        "bootstrap_info": request.get("bootstrap_info"),
        "extra_args": request.get("extra_args"),
        "mm_processor_kwargs": request.get("mm_processor_kwargs"),
        "agent_context": request.get("agent_context"),
        "request_timestamp_ms": request.get("request_timestamp_ms"),
    }


class TaperRouterHandler:
    def __init__(self, runtime: DistributedRuntime, config: TaperRouterConfig) -> None:
        self._runtime = runtime
        self._config = config
        self._kv_router: Optional[KvRouter] = None
        self._load: Optional[LoadSnapshotProvider] = None
        self._gate: Optional[TaperGate] = None
        self._stat_requests_total = 0
        self._stat_gated_requests = 0
        self._stat_passthrough_requests = 0

    async def initialize(self) -> None:
        worker_endpoint = self._runtime.endpoint(self._config.endpoint)

        self._kv_router = KvRouter(
            endpoint=worker_endpoint,
            block_size=self._config.router_block_size,
            kv_router_config=build_kv_router_config(self._config),
            aic_perf_config=build_aic_perf_config(self._config),
        )

        self._load = LoadSnapshotProvider(worker_endpoint)
        self._load.start()

        self._gate = TaperGate(load=self._load, config=self._config.to_taper_config())
        self._gate.start()
        logger.info(
            "TAPER Router initialized (worker_endpoint=%s, block_size=%s, shadow_mode=%s)",
            self._config.endpoint,
            self._config.router_block_size,
            self._config.shadow_mode,
        )

    async def shutdown(self) -> None:
        if self._gate is not None:
            await self._gate.stop()
        if self._load is not None:
            self._load.stop()
        logger.info("TAPER Router shutdown complete")

    async def generate(self, request: dict[str, Any]):
        if self._gate is None or self._kv_router is None:
            raise RuntimeError("TaperRouterHandler used before initialize() was called")

        task_id, request_id = _extract_task_and_request_id(request)
        self._stat_requests_total += 1
        preprocessed = _wrap_preprocessed_request(request)

        # No task identity -> behave like the standalone router (no gating).
        if task_id is None or request_id is None:
            self._stat_passthrough_requests += 1
            logger.debug("taper.route path=passthrough model=%s", request.get("model"))
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                yield chunk
            return

        self._stat_gated_requests += 1
        decision = await self._gate.before_request(task_id, request_id)
        logger.debug(
            "taper.route path=gated task=%s request=%s protected=%s "
            "was_deferred=%s waited=%.4fs",
            task_id, request_id, decision.protected, decision.was_deferred,
            decision.waited_seconds,
        )
        try:
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                yield chunk
        finally:
            await self._gate.after_request(task_id)

    async def status(self, request: Optional[dict[str, Any]] = None):
        gate_status = await self._gate.status_snapshot() if self._gate is not None else None
        yield {
            "status": "ready" if self._gate is not None else "starting",
            "component": "taper_router",
            "namespace": self._config.namespace,
            "worker_endpoint": self._config.endpoint,
            "gate": gate_status,
            "requests": {
                "total": self._stat_requests_total,
                "gated": self._stat_gated_requests,
                "passthrough": self._stat_passthrough_requests,
            },
        }

    async def metrics(self, request: Optional[dict[str, Any]] = None):
        gate_metrics = (
            await self._gate.metrics_snapshot()
            if self._gate is not None
            else {"counters": {}, "gauges": {}}
        )
        counters = {
            **gate_metrics["counters"],
            "requests_total": self._stat_requests_total,
            "gated_requests_total": self._stat_gated_requests,
            "passthrough_requests_total": self._stat_passthrough_requests,
        }
        yield {
            "component": "taper_router",
            "namespace": self._config.namespace,
            "counters": counters,
            "gauges": gate_metrics["gauges"],
        }


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    logger.info("TAPER Router starting (endpoint=%s, namespace=%s)", config.endpoint, config.namespace)

    handler = TaperRouterHandler(runtime, config)
    await handler.initialize()

    generate_endpoint = runtime.endpoint(f"{config.namespace}.taper_router.generate")

    if config.model_name:
        model_path = config.model_path or config.model_name
        runtime_cfg = ModelRuntimeConfig()
        if config.tool_call_parser:
            runtime_cfg.tool_call_parser = config.tool_call_parser
        if config.reasoning_parser:
            runtime_cfg.reasoning_parser = config.reasoning_parser
        await register_model(
            model_input=ModelInput.Tokens,
            model_type=ModelType.Chat | ModelType.Completions,
            endpoint=generate_endpoint,
            model_path=model_path,
            model_name=config.model_name,
            runtime_config=runtime_cfg,
            worker_type=WorkerType.Aggregated,
        )

    status_endpoint = runtime.endpoint(f"{config.namespace}.taper_router.status")
    metrics_endpoint = runtime.endpoint(f"{config.namespace}.taper_router.metrics")

    logger.info(
        "TAPER Router serving endpoints: generate=%s status=%s metrics=%s",
        f"{config.namespace}.taper_router.generate",
        f"{config.namespace}.taper_router.status",
        f"{config.namespace}.taper_router.metrics",
    )

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, graceful_shutdown=True,
                metrics_labels=[("service", "taper_router")],
            ),
            status_endpoint.serve_endpoint(
                handler.status, graceful_shutdown=True,
                metrics_labels=[("service", "taper_router")],
            ),
            metrics_endpoint.serve_endpoint(
                handler.metrics, graceful_shutdown=True,
                metrics_labels=[("service", "taper_router")],
            ),
        )
    finally:
        await handler.shutdown()


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
