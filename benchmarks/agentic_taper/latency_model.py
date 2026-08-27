# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Victim latency as a function of decode load — the model behind A3's budget rule.

TAPER admits opportunistic work only while projected completion stays within a
slack budget: ``T(S) <= T0 + rho*B_t``. To evaluate that for agentic fan-out we
need ``T(S)`` — the victim's ITL if we admit one more sibling — as a function of
the live decode load Dynamo already reports (``num_decode_requests``). This module
is that function, fit from a calibration sweep of (decode_load, victim_ITL) pairs.

Linear model: ``ITL(load) = t0_ms + beta_ms_per_req * load``. A line is the honest
first model — the calibration has a handful of points and the marginal cost of one
more decode request is ~constant over the useful range; we can swap in a richer
form later without touching the gate (it only calls ``itl_at``). Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyModel:
    t0_ms: float             # unloaded intercept: victim ITL at zero extra load
    beta_ms_per_req: float   # marginal ITL cost per concurrent decode request
    n_points: int = 0        # how many calibration points fit this (provenance)

    def itl_at(self, load: float) -> float:
        """Projected victim ITL at ``load`` concurrent decode requests."""
        return self.t0_ms + self.beta_ms_per_req * max(0.0, load)

    def load_at(self, itl_ms: float) -> float:
        """Inverse: the decode load at which victim ITL reaches ``itl_ms``.

        Infinite if the fit found no positive slope (load doesn't move latency).
        """
        if self.beta_ms_per_req <= 0:
            return float("inf")
        return (itl_ms - self.t0_ms) / self.beta_ms_per_req

    def threshold_for_slo(self, slo_ms: float) -> float:
        """The A3 admit boundary in load units: admit a sibling while current
        load <= this, since admitting adds one (load+1) and we require the
        *post-admit* projected ITL to stay within the SLO. This equals
        ``f^-1(slo) - 1`` — the derived tau* the old hardcoded sweep was groping
        for. Reported for the calibration summary / ablation; the gate evaluates
        the projection directly so it never rounds a boundary."""
        return self.load_at(slo_ms) - 1.0

    def to_dict(self) -> dict:
        return {"t0_ms": self.t0_ms, "beta_ms_per_req": self.beta_ms_per_req,
                "n_points": self.n_points}

    @classmethod
    def from_dict(cls, d: dict) -> "LatencyModel":
        return cls(float(d["t0_ms"]), float(d["beta_ms_per_req"]),
                   int(d.get("n_points", 0)))

    @classmethod
    def fit(cls, pairs: list) -> "LatencyModel":
        """Least-squares fit of ITL = t0 + beta*load over (load, itl_ms) pairs.

        Needs >= 2 points with spread in load. If all loads are equal (no spread),
        slope is unidentifiable -> beta=0, t0=mean(itl) (a flat, conservative
        model that never *lowers* the admit bar as load rises).
        """
        pts = [(float(x), float(y)) for x, y in pairs]
        if len(pts) < 2:
            raise ValueError("LatencyModel.fit needs >= 2 calibration points")
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        if sxx == 0:
            return cls(t0_ms=my, beta_ms_per_req=0.0, n_points=n)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        beta = sxy / sxx
        t0 = my - beta * mx
        return cls(t0_ms=t0, beta_ms_per_req=beta, n_points=n)
