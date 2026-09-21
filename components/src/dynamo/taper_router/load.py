# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live per-worker decode-load snapshot for the Agentic TAPER admission gate.

This is the piece the brief's P0-1 (``FpmSnapshotProvider``) scoped as a Rust
PR. It turns out not to need one: ``thunderagent_router/capacity.py`` already
subscribes to ``FpmEventSubscriber`` from Python, but only decodes the model
deployment card (static ``kv_cache_block_size`` / ``total_kv_blocks``) via
``get_model_cards()``. The same subscriber also exposes ``get_recent_stats()``
-- raw per-``(worker_id, dp_rank)`` ``ForwardPassMetrics`` bytes, decodable
with ``dynamo.common.forward_pass_metrics.decode()`` -- which carries the live
``num_decode_requests`` field this gate actually gates on (brief: "start with
a single scalar: ``num_decode_requests``"). Confirmed against
``lib/bindings/python/rust/llm/fpm.rs`` and
``components/src/dynamo/common/forward_pass_metrics.py``.

Two tracking-mode subscribers cannot share one ``FpmEventSubscriber`` per
``start_tracking()``'s docstring ("recv() will raise RuntimeError" after
tracking starts, and tracking is the only multi-read mode) -- but nothing
stops two independent subscriber instances, one per consumer, both
constructed against the same worker endpoint. ThunderAgent and TAPER can
therefore run as separate Python processes against the same workers without
coordination, which is the point: this stays a standalone add-on.
"""

from __future__ import annotations

import logging
from typing import Optional

from dynamo.common.forward_pass_metrics import decode as decode_fpm
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


class LoadSnapshotProvider:
    """Live per-worker decode load, keyed by worker id (dp ranks summed)."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._subscriber: Optional[FpmEventSubscriber] = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = FpmEventSubscriber(self._endpoint)
        self._subscriber.start_tracking()
        logger.info("LoadSnapshotProvider: subscribed to FPM stream")

    def stop(self) -> None:
        if self._subscriber is None:
            return
        try:
            self._subscriber.shutdown()
        except Exception as exc:
            logger.warning("LoadSnapshotProvider shutdown error: %s", exc)
        self._subscriber = None

    def snapshot(self) -> dict[int, int]:
        """Return ``{worker_id: num_decode_requests}``, summed across dp ranks.

        Empty until the first FPM message has been observed for a worker
        (cold start) -- callers must treat a missing worker id as "unknown
        load", not "zero load"; see ``gate.py``'s handling.
        """
        if self._subscriber is None:
            return {}
        try:
            raw = self._subscriber.get_recent_stats()
        except Exception as exc:
            logger.debug("LoadSnapshotProvider snapshot error: %s", exc)
            return {}

        out: dict[int, int] = {}
        for (worker_id_str, _dp_rank), payload in raw.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            metrics = decode_fpm(payload)
            if metrics is None:
                continue
            # scheduled_requests.num_decode_requests: the load actually
            # occupying the shared decode step right now. queued_requests
            # carries preempted/waiting decode requests separately, for a
            # later slack-aware threshold that wants to react before the step
            # widens rather than after.
            out[worker_id] = out.get(worker_id, 0) + metrics.scheduled_requests.num_decode_requests
        return out
