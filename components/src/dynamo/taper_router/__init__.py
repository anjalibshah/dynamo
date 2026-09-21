# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic TAPER admission gate inside a Dynamo router service.

Experimental / POC, not a released component -- see README.md. Standalone
add-on modeled on ``dynamo.thunderagent_router``: wraps the native KvRouter,
zero core Dynamo changes, runs as ``python -m dynamo.taper_router``.
"""

from dynamo.taper_router.gate import GateDecision, TaperConfig, TaperGate

__all__ = ["GateDecision", "TaperConfig", "TaperGate"]
