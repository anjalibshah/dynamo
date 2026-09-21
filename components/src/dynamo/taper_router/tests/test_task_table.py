# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the task/branch state model."""

from __future__ import annotations

import pytest

from dynamo.taper_router.task_table import AgentTaskState, TaskTable

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_get_or_create_creates_new_task():
    table = TaskTable()
    task = table.get_or_create("t1")
    assert task.task_id == "t1"
    assert task.protected_admitted is False
    assert task.inflight_opportunistic == 0
    assert task.deferred_request_ids == []
    assert table.tasks["t1"] is task


def test_get_or_create_returns_same_instance_on_repeat_calls():
    table = TaskTable()
    first = table.get_or_create("t1")
    second = table.get_or_create("t1")
    assert first is second


def test_get_or_create_tracks_independent_tasks():
    table = TaskTable()
    a = table.get_or_create("a")
    b = table.get_or_create("b")
    a.protected_admitted = True
    assert b.protected_admitted is False
    assert set(table.tasks) == {"a", "b"}


def test_release_removes_task_and_returns_it():
    table = TaskTable()
    task = table.get_or_create("t1")
    removed = table.release("t1")
    assert removed is task
    assert "t1" not in table.tasks


def test_release_unknown_task_returns_none():
    table = TaskTable()
    assert table.release("missing") is None


def test_agent_task_state_defaults_are_independent_across_instances():
    # Regression guard for the classic mutable-default-argument bug: two
    # states must not share the same deferred_request_ids list.
    a = AgentTaskState(task_id="a")
    b = AgentTaskState(task_id="b")
    a.deferred_request_ids.append("req-1")
    assert b.deferred_request_ids == []
