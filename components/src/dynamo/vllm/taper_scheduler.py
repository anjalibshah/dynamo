# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TaperScheduler -- InstrumentedScheduler with Agentic TAPER per-step decode regulation.

The policy and the skip mechanism live in ``taper_engine`` (no vLLM import, unit
tested on a fake scheduler); this class binds them to vLLM v0.31.0. Forward
pass metrics keep working: skipped requests are simply not in a step's
``SchedulerOutput``, and ``taper.engine`` log lines report what was skipped.

Selected automatically when ``DYN_TAPER_ENGINE`` is ``class_blind`` or
``role_aware`` (see ``args.py``), or via:
    --scheduler-cls "dynamo.vllm.taper_scheduler.TaperScheduler"
"""

from __future__ import annotations

import logging

from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler
from dynamo.vllm.taper_engine import TaperConfig, TaperSkipMixin

logger = logging.getLogger(__name__)


class TaperScheduler(TaperSkipMixin, InstrumentedScheduler):
    def __init__(self, vllm_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        unsupported = []
        if getattr(self, "use_pp", False):
            unsupported.append(
                "pipeline parallelism (placeholders may be non-zero on rejoin)"
            )
        if getattr(vllm_config, "speculative_config", None) is not None:
            unsupported.append(
                "speculative decoding (skipped requests keep draft placeholders)"
            )
        if getattr(self, "use_v2_model_runner", False):
            # Not verified: the V2 runner's persistent state for unscheduled requests.
            logger.warning(
                "taper.engine: the V2 model runner is not verified with decode skips"
            )
        model = getattr(
            getattr(vllm_config, "model_config", None), "served_model_name", ""
        )
        dp_rank = getattr(
            getattr(vllm_config, "parallel_config", None), "data_parallel_rank", 0
        )
        self._taper_init(
            TaperConfig.from_env(),
            unsupported="; ".join(unsupported),
            model_name=model if isinstance(model, str) else str(model),
            dp_rank=dp_rank or 0,
        )
