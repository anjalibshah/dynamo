# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline analysis for the Agentic TAPER P0 experiment (P0-5 / P0-7).

Consumes the per-cell ``results/*.jsonl`` and ``manifest.jsonl`` written by
``experiment.py`` and produces the numbers the brief's success criteria and
pre-registered decision rules need, per model:

* **Per-cell metrics** — task goodput (from the manifest, computed at run time
  with the model's SLO), victim p50/p95/p99 ITL & TTFT, a throughput proxy,
  gate admitted/held counts.
* **Aggregation across reps** — median + IQR, because the claim is about tails,
  not means.
* **Charged externality (H1b)** — paired, same-seed victim ITL: A1−A0 and
  A3−A0, matched per victim request id. Valid only because every arm of a
  (workload, rep) shares one seed and therefore one arrival sequence.
* **Throughput-trap / knee (H1a)** — per model, A1 goodput vs. offered load
  (burst) while throughput keeps rising, with A0 as the no-knee control.
* **H2** — A3 vs. best-tuned A2 on goodput and victim tail.
* **Decision verdicts** — applies the pre-registered rules and prints a call.

Pure stdlib. No GPU, no engine — it only reads files. Fully unit-tested against
a synthetic results directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------------- #

def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def _median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _iqr(vals: list[float]) -> float:
    return (_pct(vals, 0.75) - _pct(vals, 0.25)) if vals else 0.0


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _cell_filename(cell: dict) -> str:
    arm = cell["arm"]
    p = f"_p{cell['param']:g}" if arm in ("A2", "A3") else ""
    return (f"{cell['model']}__{arm}{p}__k{cell['fanout_k']}"
            f"_b{cell['burst']:g}_pf{cell['prefix_blocks']}_r{cell['rep']}.jsonl")


def load_manifest(outdir: str) -> list[dict]:
    path = os.path.join(outdir, "manifest.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def load_records(outdir: str, cell: dict) -> list[dict]:
    path = os.path.join(outdir, _cell_filename(cell))
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


# --------------------------------------------------------------------------- #
# per-cell metrics
# --------------------------------------------------------------------------- #

@dataclass
class CellMetrics:
    victim_itl_p50: float = 0.0
    victim_itl_p95: float = 0.0
    victim_itl_p99: float = 0.0
    victim_ttft_p95: float = 0.0
    throughput_tok_s: float = 0.0
    n_victim: int = 0
    gate_wait_p95_ms: float = 0.0


def _victim_mean_itls(records: list[dict]) -> list[float]:
    return [statistics.mean(r["itls_ms"]) for r in records
            if r.get("role") == "victim" and r.get("itls_ms")]


def victim_goodput_at_slo(records: list[dict], slo_ms: float) -> float:
    """Fraction of victim (interactive) requests whose mean ITL meets the SLO.

    Recomputed from the saved per-request ITLs, so any SLO can be evaluated
    offline without re-running the arms — this is what makes the SLO a sweepable
    knob rather than a value baked at run time.
    """
    vitls = _victim_mean_itls(records)
    if not vitls:
        return 0.0
    return sum(1 for v in vitls if v <= slo_ms) / len(vitls)


def cell_metrics(records: list[dict]) -> CellMetrics:
    vitls = _victim_mean_itls(records)
    ttfts = [r["ttft_ms"] for r in records if r.get("role") == "victim" and r.get("ttft_ms")]
    gate_waits = [r.get("gate_wait_ms", 0.0) for r in records]
    # throughput proxy: total generated tokens / makespan
    toks = sum(len(r.get("itls_ms", [])) + (1 if r.get("ttft_ms") else 0) for r in records)
    starts = [r["t_arrival"] for r in records if r.get("t_arrival")]
    ends = [r["t_done"] for r in records if r.get("t_done")]
    makespan = (max(ends) - min(starts)) if starts and ends else 0.0
    return CellMetrics(
        victim_itl_p50=round(_pct(vitls, 0.50), 3),
        victim_itl_p95=round(_pct(vitls, 0.95), 3),
        victim_itl_p99=round(_pct(vitls, 0.99), 3),
        victim_ttft_p95=round(_pct(ttfts, 0.95), 3),
        throughput_tok_s=round(toks / makespan, 2) if makespan > 0 else 0.0,
        n_victim=len(vitls),
        gate_wait_p95_ms=round(_pct(gate_waits, 0.95), 3),
    )


# --------------------------------------------------------------------------- #
# charged externality (paired, same seed)
# --------------------------------------------------------------------------- #

def _victim_itl_by_request(records: list[dict]) -> dict[str, float]:
    return {r["request_id"]: statistics.mean(r["itls_ms"])
            for r in records if r.get("role") == "victim" and r.get("itls_ms")}


def charged_externality(outdir: str, manifest: list[dict]) -> dict:
    """Paired A1−A0 and A3−A0 victim ITL, matched per request id per seed.

    Returns per (model, arm, k, burst, pf): list of per-request deltas across
    reps, plus their median. A0 is the baseline; each rep pairs on its shared
    seed so the same victim request is differenced under the two arms.
    """
    # index A0 records by (model, k, burst, pf, rep)
    def key(c): return (c["model"], c["fanout_k"], c["burst"], c["prefix_blocks"], c["rep"])
    a0 = {}
    for c in manifest:
        if c["arm"] == "A0":
            a0[key(c)] = _victim_itl_by_request(load_records(outdir, c))

    out: dict = defaultdict(list)
    for c in manifest:
        if c["arm"] not in ("A1", "A3"):
            continue
        base = a0.get(key(c))
        if not base:
            continue
        cur = _victim_itl_by_request(load_records(outdir, c))
        gk = (c["model"], c["arm"], c["fanout_k"], c["burst"], c["prefix_blocks"], c.get("param", 0))
        for rid, itl in cur.items():
            if rid in base:
                out[gk].append(itl - base[rid])
    return {"_".join(map(str, k)): {"median_delta_ms": round(_median(v), 3),
                                    "p95_delta_ms": round(_pct(v, 0.95), 3),
                                    "n": len(v)}
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
# aggregation + hypotheses
# --------------------------------------------------------------------------- #

@dataclass
class Agg:
    goodput_median: float = 0.0
    goodput_iqr: float = 0.0
    victim_itl_p95_median: float = 0.0
    throughput_median: float = 0.0
    n_reps: int = 0
    params_seen: list = field(default_factory=list)


def aggregate(outdir: str, manifest: list[dict], slo_ms: Optional[float] = None) -> dict:
    """Group by (model, arm, k, burst, pf, param); median/IQR across reps.

    If ``slo_ms`` is given, goodput is recomputed from each cell's saved victim
    ITLs at that SLO (the offline sweep); otherwise the run-time goodput from the
    manifest is used.
    """
    groups: dict = defaultdict(list)
    for c in manifest:
        gk = (c["model"], c["arm"], c["fanout_k"], c["burst"], c["prefix_blocks"], c.get("param", 0))
        recs = load_records(outdir, c)
        m = cell_metrics(recs)
        good = victim_goodput_at_slo(recs, slo_ms) if slo_ms is not None else c.get("goodput", 0.0)
        groups[gk].append((good, m))
    agg = {}
    for gk, rows in groups.items():
        goods = [g for g, _ in rows]
        itls = [m.victim_itl_p95 for _, m in rows]
        thrus = [m.throughput_tok_s for _, m in rows]
        agg[gk] = Agg(goodput_median=round(_median(goods), 4),
                      goodput_iqr=round(_iqr(goods), 4),
                      victim_itl_p95_median=round(_median(itls), 3),
                      throughput_median=round(_median(thrus), 2),
                      n_reps=len(rows))
    return agg


def knee_curve(agg: dict, model: str, arm: str) -> list[dict]:
    """Goodput & throughput vs. burst for one (model, arm), averaged over k/pf/param."""
    by_burst: dict = defaultdict(lambda: {"good": [], "thru": []})
    for (m, a, k, burst, pf, param), v in agg.items():
        if m == model and a == arm:
            by_burst[burst]["good"].append(v.goodput_median)
            by_burst[burst]["thru"].append(v.throughput_median)
    curve = []
    for burst in sorted(by_burst):
        d = by_burst[burst]
        curve.append({"burst": burst,
                      "goodput": round(_median(d["good"]), 4),
                      "throughput": round(_median(d["thru"]), 2)})
    return curve


def detect_knee(curve: list[dict]) -> dict:
    """H1a: goodput turns down past a knee while throughput keeps rising."""
    if len(curve) < 2:
        return {"has_knee": False, "reason": "need >=2 load points"}
    goods = [p["goodput"] for p in curve]
    thrus = [p["throughput"] for p in curve]
    goodput_declines = goods[-1] < goods[0]
    throughput_rises = thrus[-1] >= thrus[0]
    return {"has_knee": bool(goodput_declines and throughput_rises),
            "goodput_first": goods[0], "goodput_last": goods[-1],
            "throughput_first": thrus[0], "throughput_last": thrus[-1]}


def _loads(agg: dict, model: str) -> list[float]:
    return sorted({burst for (m, a, k, b2, pf, p), _ in agg.items()
                   if m == model for burst in [b2]})


def best_param_at_load(agg: dict, model: str, arm: str, burst: float) -> Optional[dict]:
    """Best swept parameter for an arm at ONE load, by median goodput.

    Comparing at matched load is essential: aggregating goodput across the whole
    burst sweep saturates at 1.0 on the easy points and hides the effect. The
    brief's "A2 swept to its best static point" is evaluated where it matters —
    the contended load near/after the knee.
    """
    cands = [(param, v) for (m, a, k, b2, pf, param), v in agg.items()
             if m == model and a == arm and b2 == burst]
    if not cands:
        return None
    param, v = max(cands, key=lambda kv: kv[1].goodput_median)
    return {"param": param, "burst": burst, "goodput_median": v.goodput_median,
            "victim_itl_p95_median": v.victim_itl_p95_median,
            "throughput_median": v.throughput_median}


def arm_goodput_at_load(agg: dict, model: str, arm: str, burst: float) -> Optional[float]:
    b = best_param_at_load(agg, model, arm, burst)
    return b["goodput_median"] if b else None


def h2_verdict(agg: dict, model: str) -> dict:
    """H2 at matched load: does A3 beat best-tuned A2 at the contended load?

    Evaluated at the highest offered load in the sweep (most contention, where
    dynamic gating should pay off). Reports the full per-load table too.
    """
    loads = _loads(agg, model)
    if not loads:
        return {"decidable": False, "reason": "no cells"}
    contended = loads[-1]
    a2 = best_param_at_load(agg, model, "A2", contended)
    a3 = best_param_at_load(agg, model, "A3", contended)
    if not a2 or not a3:
        return {"decidable": False, "reason": "missing A2 or A3 at contended load"}
    # Goodput is the deciding metric (brief H2 falsifier: "cannot beat a
    # well-tuned fixed cap on goodput"); victim tail is the secondary win.
    # Throughput is REPORTED, not gated on: a gate that holds requests has lower
    # raw throughput by design — that is the throughput trap, not a failure.
    goodput_win = a3["goodput_median"] > a2["goodput_median"]
    tail_win = a3["victim_itl_p95_median"] <= a2["victim_itl_p95_median"]
    a2_thru = a2["throughput_median"] or 1e-9
    per_load = []
    for b in loads:
        g2 = arm_goodput_at_load(agg, model, "A2", b)
        g3 = arm_goodput_at_load(agg, model, "A3", b)
        per_load.append({"burst": b, "best_A2_goodput": g2, "best_A3_goodput": g3})
    return {"decidable": True, "contended_load": contended,
            "best_A2": a2, "best_A3": a3, "per_load": per_load,
            "A3_beats_A2_goodput": bool(goodput_win),
            "A3_le_A2_victim_tail": bool(tail_win),
            "A3_throughput_vs_A2": round(a3["throughput_median"] / a2_thru, 3),
            "H2_supported": bool(goodput_win and tail_win)}


def h1_verdict(agg: dict, model: str, a1_knee: dict, ext: dict) -> dict:
    """H1 evidence: A1's knee + A1 goodput collapse vs A0 at the contended load.

    A0 is not expected to be flat — it degrades under raw load too. The right
    signal is that A1's goodput falls *below A0's* at high load (the fan-out
    externality) and the paired charged externality is positive.
    """
    loads = _loads(agg, model)
    contended = loads[-1] if loads else None
    a0 = arm_goodput_at_load(agg, model, "A0", contended) if contended else None
    a1 = arm_goodput_at_load(agg, model, "A1", contended) if contended else None
    # median charged externality for this model's A1 cells (positive => A1 worse)
    a1_ext = [v["median_delta_ms"] for kstr, v in ext.items()
              if kstr.startswith(f"{model}_A1_")]
    ext_med = _median(a1_ext) if a1_ext else 0.0
    a1_below_a0 = (a0 is not None and a1 is not None and a1 < a0)
    supported = bool(a1_knee.get("has_knee") and a1_below_a0 and ext_med > 0)
    return {"contended_load": contended,
            "A0_goodput_at_load": a0, "A1_goodput_at_load": a1,
            "A1_below_A0": a1_below_a0,
            "A1_has_knee": a1_knee.get("has_knee", False),
            "charged_externality_median_ms": round(ext_med, 3),
            "H1_supported": supported}


# --------------------------------------------------------------------------- #
# summary + decision rules
# --------------------------------------------------------------------------- #

DEFAULT_SLO_GRID = (15.0, 25.0, 50.0)


def _verdicts_for_agg(agg: dict, models: list[str], ext: dict) -> dict:
    per_model = {}
    for model in models:
        knee = detect_knee(knee_curve(agg, model, "A1"))
        h1 = h1_verdict(agg, model, knee, ext)
        h2 = h2_verdict(agg, model)
        per_model[model] = {
            "A1_curve": knee_curve(agg, model, "A1"),
            "A0_curve": knee_curve(agg, model, "A0"),
            "H1": h1,
            "H1_call": ("externality present; proceed to H2"
                        if h1["H1_supported"]
                        else "NEGATIVE — no disproportionate externality; stop"),
            "H2": h2,
            "H2_call": ("SUPPORTED — dynamic gate beats best static cap at the contended load"
                        if h2.get("H2_supported")
                        else "NEGATIVE — value not in dynamic gating; report and stop"),
        }
    return per_model


def summarize(outdir: str, primary_slo: float = 25.0, slo_grid=DEFAULT_SLO_GRID) -> dict:
    manifest = load_manifest(outdir)
    models = sorted({c["model"] for c in manifest})
    ext = charged_externality(outdir, manifest)

    # Primary verdicts use VICTIM goodput at primary_slo, recomputed from saved
    # ITLs — consistent with the sweep. (The manifest's run-time goodput is
    # informational only; it may use an all-task definition or a different SLO.)
    per_model = _verdicts_for_agg(aggregate(outdir, manifest, slo_ms=primary_slo), models, ext)

    # Offline SLO sweep: recompute goodput from saved victim ITLs at each SLO,
    # so we can report whether H1/H2 hold across plausible SLOs (robustness to
    # the SLO choice) — no re-runs, purely from results/*.jsonl.
    slo_grid = tuple(float(s) for s in slo_grid)
    slo_sweep = {}
    for slo in slo_grid:
        agg_s = aggregate(outdir, manifest, slo_ms=slo)
        pm = {}
        for model in models:
            knee = detect_knee(knee_curve(agg_s, model, "A1"))
            h1 = h1_verdict(agg_s, model, knee, ext)
            h2 = h2_verdict(agg_s, model)
            a2 = best_param_at_load(agg_s, model, "A2", _loads(agg_s, model)[-1]) if _loads(agg_s, model) else None
            a3 = best_param_at_load(agg_s, model, "A3", _loads(agg_s, model)[-1]) if _loads(agg_s, model) else None
            pm[model] = {
                "H1_supported": h1["H1_supported"],
                "H2_supported": h2.get("H2_supported", False),
                "best_A2_goodput": a2["goodput_median"] if a2 else None,
                "best_A3_goodput": a3["goodput_median"] if a3 else None,
            }
        slo_sweep[str(slo)] = pm

    summary = {"models": models, "n_cells": len(manifest),
               "per_model": per_model, "charged_externality": ext,
               "slo_grid_ms": list(slo_grid), "slo_sweep": slo_sweep}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _print_summary(summary: dict) -> None:
    print(f"\n=== Agentic TAPER P0 — {summary['n_cells']} cells, "
          f"{len(summary['models'])} models ===")
    for model, v in summary["per_model"].items():
        print(f"\n## {model}")
        print("  A1 goodput vs load (burst → goodput / throughput):")
        for p in v["A1_curve"]:
            print(f"     burst {p['burst']:>4g}:  goodput {p['goodput']:.3f}   "
                  f"throughput {p['throughput']:.1f} tok/s")
        h1 = v["H1"]
        print(f"  H1  {v['H1_call']}")
        print(f"      A1 knee={h1['A1_has_knee']}  "
              f"A1 goodput<{'A0' if h1['A1_below_A0'] else '=A0'} at burst {h1['contended_load']:g} "
              f"(A0={h1['A0_goodput_at_load']}, A1={h1['A1_goodput_at_load']})  "
              f"charged externality median={h1['charged_externality_median_ms']}ms")
        h2 = v["H2"]
        if h2.get("decidable"):
            print(f"  H2  {v['H2_call']}  (at burst {h2['contended_load']:g})")
            print(f"      best A2: goodput {h2['best_A2']['goodput_median']:.3f}, "
                  f"victim p95 ITL {h2['best_A2']['victim_itl_p95_median']:.1f}ms  (cap={h2['best_A2']['param']:g})")
            print(f"      best A3: goodput {h2['best_A3']['goodput_median']:.3f}, "
                  f"victim p95 ITL {h2['best_A3']['victim_itl_p95_median']:.1f}ms  (tau={h2['best_A3']['param']:g})")
        else:
            print(f"  H2  undecidable: {h2.get('reason')}")

    sweep = summary.get("slo_sweep", {})
    if sweep:
        print("\n=== SLO robustness (H1 / H2 supported at each ITL SLO) ===")
        slos = summary.get("slo_grid_ms", [])
        for model in summary["models"]:
            cells = []
            for slo in slos:
                pm = sweep.get(str(float(slo)), {}).get(model, {})
                h1 = "H1✓" if pm.get("H1_supported") else "H1✗"
                h2 = "H2✓" if pm.get("H2_supported") else "H2✗"
                cells.append(f"{slo:>4g}ms:{h1}/{h2}")
            print(f"  {model}:  " + "   ".join(cells))
        print("  (H2✓ across all SLOs = the gate's win is robust to the SLO choice)")
    print(f"\nwrote {os.path.join('<outdir>', 'summary.json')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Agentic TAPER P0 results.")
    p.add_argument("--outdir", default="./results")
    p.add_argument("--slo-grid", default="15,25,50",
                   help="comma-separated ITL SLOs (ms) for the offline robustness sweep")
    p.add_argument("--primary-slo", type=float, default=25.0,
                   help="ITL SLO (ms) for the primary verdict")
    a = p.parse_args()
    grid = tuple(float(x) for x in a.slo_grid.split(","))
    summary = summarize(a.outdir, primary_slo=a.primary_slo, slo_grid=grid)
    _print_summary(summary)


if __name__ == "__main__":
    main()
