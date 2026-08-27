# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration: measure victim ITL vs. decode load, fit the model A3 needs.

To make A3 a real test of TAPER's budget rule (``T(S) <= T0 + rho*B_t``) rather
than a threshold sweep, we need ``ITL(load)`` — how victim latency grows with the
live decode load Dynamo reports. This module drives a load sweep, samples
``num_decode_requests`` while each level runs, pairs it with the victim ITL
measured for that level, and fits a :class:`LatencyModel`.

The heavy lifting (building frontends / the FPM source per level) lives in
``experiment.run_calibration`` so this module stays dependency-light and testable
with a mock load source. Here we own only the sampler and the fit/report.
"""

from __future__ import annotations

import asyncio

from latency_model import LatencyModel


class LoadSampler:
    """Polls ``load_fn`` every ``interval_ms`` into a list, until stopped.

    Used to observe the decode load the server actually ran at during a level,
    so the calibration point is (measured load, measured victim ITL) — both
    empirical, neither assumed.
    """

    def __init__(self, load_fn, interval_ms: float = 200.0):
        self._load_fn = load_fn
        self._interval = interval_ms / 1000.0
        self.samples: list[float] = []
        self._task = None
        self._stop = False

    async def _loop(self):
        while not self._stop:
            try:
                self.samples.append(float(self._load_fn()))
            except Exception:
                pass
            await asyncio.sleep(self._interval)

    def start(self):
        self._stop = False
        self._task = asyncio.get_event_loop().create_task(self._loop())

    async def stop(self):
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def summarize_calibration(points: list, slo_ms: float) -> dict:
    """Fit ITL(load) over (decode_load, victim_itl_ms) points; report A3 params.

    Returns a JSON-able dict with the fitted model, the SLO-derived admit
    boundary (tau* = f^-1(slo) - 1), and a suggested SLO floor (the unloaded
    intercept, below which no gate can ever pass). Raises if < 2 usable points.
    """
    usable = [(x, y) for x, y in points if y and y > 0]
    model = LatencyModel.fit(usable)
    tau_star = model.threshold_for_slo(slo_ms)
    return {
        "model": model.to_dict(),
        "slo_ms": slo_ms,
        "tau_star": tau_star,                       # admit while live load <= this
        "unloaded_itl_ms": round(model.t0_ms, 3),   # SLO must exceed this to be feasible
        "marginal_ms_per_decode_req": round(model.beta_ms_per_req, 4),
        "points": [[round(x, 2), round(y, 2)] for x, y in usable],
    }
