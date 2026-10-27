# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DecodeRegulationScheduler -- InstrumentedScheduler with per-step decode regulation.

The policy and the skip mechanism live in ``decode_regulation`` (no vLLM import, unit
tested on a fake scheduler); this class binds them to vLLM v0.31.0. Forward
pass metrics keep working: skipped requests are simply not in a step's
``SchedulerOutput``, and ``decode_regulation`` log lines report what was skipped.

Selected automatically when ``DYN_DECODE_REGULATION_MODE`` is not ``off``
(see ``args.py``), or via:
    --scheduler-cls "dynamo.vllm.decode_regulation_scheduler.DecodeRegulationScheduler"
"""

from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from dynamo.vllm.decode_regulation import DecodeRegulationMixin
from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler


class DecodeRegulationScheduler(DecodeRegulationMixin, InstrumentedScheduler):
    def __init__(self, vllm_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._reg_init_vllm(vllm_config, Scheduler, Request)
