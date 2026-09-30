# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task identity from agent_context headers: which requests are the trunk."""

from __future__ import annotations

import pytest

from dynamo.taper_router.__main__ import _extract_task_identity

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _req(session_id=None, parent_session_id=None) -> dict:
    ctx = {}
    if session_id is not None:
        ctx["session_id"] = session_id
    if parent_session_id is not None:
        ctx["parent_session_id"] = parent_session_id
    return {"agent_context": ctx}


def test_root_turn_is_trunk():
    assert _extract_task_identity(_req("agent-1")) == ("agent-1", "agent-1", True)


def test_branch_is_not_trunk():
    assert _extract_task_identity(_req("agent-1-b0", "agent-1")) == (
        "agent-1", "agent-1-b0", False)


def test_join_reusing_root_session_is_trunk():
    assert _extract_task_identity(_req("agent-1", "agent-1")) == ("agent-1", "agent-1", True)


@pytest.mark.parametrize("request_dict", [{}, {"agent_context": {}}, _req("")])
def test_missing_identity_passes_through(request_dict):
    assert _extract_task_identity(request_dict) == (None, None, False)
