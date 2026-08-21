# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live load signal for the A3 permit gate (P0-1).

The A3 gate keys on a single scalar of engine load. The natural source is
ForwardPassMetrics on Dynamo's event plane: ``scheduled_requests.num_decode_requests``
summed across workers is the count of sequences decoding this iteration — the
width of the shared decode step the brief is about.

Dynamo already ships the subscriber (``dynamo.llm.FpmEventSubscriber``,
``lib/bindings/python/src/dynamo/_core.pyi:1349``) and the payload decoder
(``dynamo.common.forward_pass_metrics.decode``), so this is a thin wrapper, not
new Rust.

This module is import-safe without the ``dynamo`` package: the real
``FpmLoadSource`` imports the bindings lazily inside ``start()``. A
``MockLoadSource`` lets the replay client and its tests run with no engine.
The aggregation logic (``decode_load``) is a pure function unit-tested offline.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional, Protocol


class LoadSource(Protocol):
    """What the gate needs from a load signal."""

    def num_decode_requests(self) -> int: ...
    def kv_frac(self) -> float: ...
    def freshness_ms(self) -> float: ...


def decode_load(fpm_objects: Iterable[object]) -> int:
    """Sum ``scheduled_requests.num_decode_requests`` across per-worker FPM.

    Pure and duck-typed: accepts anything exposing
    ``.scheduled_requests.num_decode_requests`` (the real
    ``ForwardPassMetrics``, or a namespace in tests). Missing / malformed
    entries contribute 0 rather than raising, matching the subscriber's
    skip-bad-message contract.
    """
    total = 0
    for m in fpm_objects:
        try:
            total += int(m.scheduled_requests.num_decode_requests)
        except (AttributeError, TypeError, ValueError):
            continue
    return total


def decode_kv_frac(fpm_objects: Iterable[object]) -> float:
    """KV pressure fallback: sum_decode_kv_tokens / capacity proxy.

    ForwardPassMetrics does not carry total KV capacity, so this returns the
    summed decode KV tokens; callers that want a fraction divide by a known
    ``kv_total`` from ComponentSnapshot. Kept separate so the gate can fall back
    to KV pressure if ``num_decode_requests`` proves too coarse (Appendix E).
    """
    total = 0
    for m in fpm_objects:
        try:
            total += int(m.scheduled_requests.sum_decode_kv_tokens)
        except (AttributeError, TypeError, ValueError):
            continue
    return float(total)


class MockLoadSource:
    """Deterministic load source for offline replay and tests.

    ``load_fn`` is called on every ``num_decode_requests()`` so a test can make
    load rise and fall on a schedule (e.g. proportional to in-flight requests).
    """

    def __init__(self, load_fn: Callable[[], int], kv_fn: Optional[Callable[[], float]] = None):
        self._load_fn = load_fn
        self._kv_fn = kv_fn or (lambda: 0.0)
        self._last_update = time.monotonic()

    def num_decode_requests(self) -> int:
        self._last_update = time.monotonic()
        return int(self._load_fn())

    def kv_frac(self) -> float:
        return float(self._kv_fn())

    def freshness_ms(self) -> float:
        return (time.monotonic() - self._last_update) * 1000.0


class FpmLoadSource:
    """Real load source backed by ``dynamo.llm.FpmEventSubscriber``.

    Network path is validated on the 8xH100 box, not offline. Construction is
    cheap; ``start()`` spawns the subscriber's background tracking tasks.

    Usage:
        src = FpmLoadSource(endpoint)   # a dynamo component Endpoint
        src.start()
        gate = PermitGate(Policy.FPM, send, load_fn=src.num_decode_requests,
                          load_threshold=tau)
    """

    def __init__(self, endpoint: object):
        self._endpoint = endpoint
        self._sub = None
        self._decode = None
        self._last_update = 0.0

    def start(self) -> None:
        # Lazy import so this module loads on machines without the bindings.
        from dynamo.llm import FpmEventSubscriber
        from dynamo.common.forward_pass_metrics import decode as fpm_decode

        self._decode = fpm_decode
        self._sub = FpmEventSubscriber(self._endpoint)
        self._sub.start_tracking()

    def _snapshot(self) -> list[object]:
        if self._sub is None:
            raise RuntimeError("FpmLoadSource.start() not called")
        # get_recent_stats() -> {(worker_id, dp_rank): raw_fpm_bytes}
        raw = self._sub.get_recent_stats()
        objs = []
        for payload in raw.values():
            m = self._decode(payload)
            if m is not None:
                objs.append(m)
        self._last_update = time.monotonic()
        return objs

    def num_decode_requests(self) -> int:
        return decode_load(self._snapshot())

    def kv_frac(self) -> float:
        return decode_kv_frac(self._snapshot())

    def freshness_ms(self) -> float:
        # Age of the most recent poll. Report alongside results: if this is
        # ~100 ms rather than ~ms, the A3 threshold must absorb the lag and the
        # gate's ceiling drops (brief Appendix E risk).
        return (time.monotonic() - self._last_update) * 1000.0 if self._last_update else float("inf")
