# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LoadSnapshotProvider. No runtime needed.

Uses the real ``forward_pass_metrics.encode``/``decode`` round trip (not a
hand-rolled fake payload) so these tests break if the wire schema TaperGate
depends on ever drifts.
"""

from __future__ import annotations

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
    encode,
)
from dynamo.taper_router.load import LoadSnapshotProvider

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _FakeSubscriber:
    def __init__(self, stats: dict[tuple[str, int], bytes]) -> None:
        self._stats = stats

    def get_recent_stats(self) -> dict[tuple[str, int], bytes]:
        return self._stats


def _fpm(worker_id: str, dp_rank: int, num_decode_requests: int) -> bytes:
    metrics = ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        scheduled_requests=ScheduledRequestMetrics(num_decode_requests=num_decode_requests),
    )
    return encode(metrics)


def _make_provider(stats: dict[tuple[str, int], bytes]) -> LoadSnapshotProvider:
    provider = LoadSnapshotProvider(endpoint=None)  # type: ignore[arg-type]
    provider._subscriber = _FakeSubscriber(stats)  # type: ignore[assignment]
    return provider


def test_snapshot_extracts_num_decode_requests():
    stats = {("1", 0): _fpm("1", 0, num_decode_requests=12)}
    provider = _make_provider(stats)
    assert provider.snapshot() == {1: 12}


def test_snapshot_sums_across_dp_ranks_for_same_worker():
    stats = {
        ("1", 0): _fpm("1", 0, num_decode_requests=12),
        ("1", 1): _fpm("1", 1, num_decode_requests=8),
    }
    provider = _make_provider(stats)
    assert provider.snapshot() == {1: 20}


def test_snapshot_keeps_workers_separate():
    stats = {
        ("1", 0): _fpm("1", 0, num_decode_requests=12),
        ("2", 0): _fpm("2", 0, num_decode_requests=5),
    }
    provider = _make_provider(stats)
    assert provider.snapshot() == {1: 12, 2: 5}


def test_snapshot_skips_unparseable_worker_ids():
    stats = {("not-an-int", 0): _fpm("not-an-int", 0, num_decode_requests=12)}
    provider = _make_provider(stats)
    assert provider.snapshot() == {}


def test_snapshot_skips_undecodable_payload():
    stats = {("1", 0): b"not a valid msgpack fpm payload"}
    provider = _make_provider(stats)
    assert provider.snapshot() == {}


def test_snapshot_returns_empty_when_subscriber_unset():
    provider = LoadSnapshotProvider(endpoint=None)  # type: ignore[arg-type]
    assert provider.snapshot() == {}


def test_snapshot_returns_empty_on_subscriber_error():
    class _RaisingSubscriber:
        def get_recent_stats(self):
            raise RuntimeError("boom")

    provider = LoadSnapshotProvider(endpoint=None)  # type: ignore[arg-type]
    provider._subscriber = _RaisingSubscriber()  # type: ignore[assignment]
    assert provider.snapshot() == {}
