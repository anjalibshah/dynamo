# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SidecarDecodeRegulationScheduler -- decode regulation for a vLLM engine run by
Dynamo's sidecar backend (``lib/sidecar/vllm``).

The sidecar runs a stock vLLM engine without Dynamo installed, so this class sits
on vLLM's own ``AsyncScheduler`` (as ``InstrumentedScheduler`` does in-process, which
keeps async scheduling on) and imports nothing from Dynamo but ``decode_regulation``
(standard library only). Three files are enough on the
engine's ``PYTHONPATH``::

    dynamo/vllm/__init__.py
    dynamo/vllm/decode_regulation.py
    dynamo/vllm/decode_regulation_sidecar.py

Select it with the ``DYN_DECODE_REGULATION_*`` variables set on the engine and::

    vllm-rs serve <model> ... -- \\
        --scheduler-cls dynamo.vllm.decode_regulation_sidecar.SidecarDecodeRegulationScheduler

Requests are classified by vLLM's native ``priority``, which the sidecar sets from
Dynamo's priority, so ``DYN_DECODE_REGULATION_PRIORITY_MIN`` is required. The
sidecar's ``/metrics`` does not include the engine's Python metrics: set
``DYN_DECODE_REGULATION_METRICS_PORT`` to serve them, on that port plus the
data-parallel rank.
"""

from __future__ import annotations

import logging
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from dynamo.vllm.decode_regulation import DecodeRegulationConfig, DecodeRegulationMixin

logger = logging.getLogger(__name__)


def _log_to_stderr() -> None:
    """vLLM sets up only its own loggers, so give Dynamo's a handler in the engine."""
    log = logging.getLogger("dynamo.vllm")
    if not log.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s")
        )
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

METRICS_PORT_ENV = "DYN_DECODE_REGULATION_METRICS_PORT"


def _metrics_registry(dp_rank: int):
    """A registry served on the metrics port plus ``dp_rank``, or None."""
    port = os.environ.get(METRICS_PORT_ENV, "").strip()
    if not port:
        return None
    try:
        from prometheus_client import CollectorRegistry, start_http_server

        registry = CollectorRegistry()
        start_http_server(int(port) + dp_rank, registry=registry)
        return registry
    except (ImportError, OSError, ValueError) as e:
        logger.warning("decode_regulation: no metrics endpoint (%s)", e)
        return None


class SidecarDecodeRegulationScheduler(DecodeRegulationMixin, AsyncScheduler):
    def __init__(self, vllm_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        _log_to_stderr()
        cfg = DecodeRegulationConfig.from_env()
        if cfg.enabled and cfg.priority_min is None:
            logger.warning(
                "decode_regulation: DYN_DECODE_REGULATION_PRIORITY_MIN is unset; "
                "under the sidecar every request counts as interactive and none is "
                "regulated"
            )
        dp_rank = getattr(
            getattr(vllm_config, "parallel_config", None), "data_parallel_rank", 0
        )
        registry = _metrics_registry(dp_rank or 0) if cfg.enabled else None
        self._reg_init_vllm(vllm_config, Scheduler, Request, registry)
