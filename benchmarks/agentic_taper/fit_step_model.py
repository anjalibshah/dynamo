# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit the taper gate's step-time model T = a + b*n + c*L from fpm_probe.py records.

PACE's predictor (taper_router/pace_predictor.py): OLS on decode-only steps,
n = decode requests in the step, L = their total KV context (tokens), T in ms.
Reports the coefficients, in-sample R^2 and MAPE, leave-one-run-out MAPE (the
pre-registered pilot check is MAPE < 15%), and error by load bucket, then
prints the PRED_COEFFS line for sweep_parallel.sh. Pure Python: runs anywhere.

    python3 fit_step_model.py /data/$USER/harbor_ab-mm/diag-*/fpm.jsonl \\
        /data/$USER/harbor_ab-mm/sws1-kv-*/fpm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(path: str, skip_first_s: float, max_ms: float) -> list[tuple[float, float, float]]:
    rows = []
    t0 = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # torn last line of a live recording
                continue
            t0 = r["t"] if t0 is None else t0
            if r["t"] - t0 < skip_first_s or r["n_prefill"] or r["n_decode"] < 1:
                continue
            if r["wall_ms"] > max_ms:  # hiccups (GC, allocation): not load-driven
                continue
            rows.append((float(r["n_decode"]), float(r["decode_kv_tok"]), float(r["wall_ms"])))
    return rows


def solve3(m: list[list[float]], v: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a 3x3 system."""
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        if abs(a[col][col]) < 1e-12:
            raise ValueError("singular design: need variation in both n and L")
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for k in range(col, 4):
                a[r][k] -= f * a[col][k]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (a[r][3] - sum(a[r][k] * x[k] for k in range(r + 1, 3))) / a[r][r]
    return x


def fit(rows: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    # Normal equations on [1, n, L/1000] (scaled for conditioning), then unscale.
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for n, l, y in rows:
        x = (1.0, n, l / 1000.0)
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    a, b, ck = solve3(xtx, xty)
    return a, b, ck / 1000.0


def predict(c: tuple[float, float, float], n: float, l: float) -> float:
    return max(0.0, c[0] + c[1] * n + c[2] * l)


def mape(c, rows) -> float:
    return sum(abs(predict(c, n, l) - y) / y for n, l, y in rows) / len(rows)


def r2(c, rows) -> float:
    my = sum(y for _, _, y in rows) / len(rows)
    ss_tot = sum((y - my) ** 2 for _, _, y in rows)
    ss_res = sum((y - predict(c, n, l)) ** 2 for n, l, y in rows)
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="fpm.jsonl files (one per run)")
    ap.add_argument("--skip-first-s", type=float, default=60.0, help="warm-up to drop per run")
    ap.add_argument("--max-ms", type=float, default=100.0, help="drop steps slower than this")
    ap.add_argument("--json", help="also write {a, b, c, ...} here")
    args = ap.parse_args()

    runs = {}
    for p in args.paths:
        rows = load(p, args.skip_first_s, args.max_ms)
        if rows:
            runs[os.path.basename(os.path.dirname(os.path.abspath(p))) or p] = rows
        else:
            print(f"skip {p}: no decode-only steps", file=sys.stderr)
    if not runs:
        sys.exit("no data")
    allrows = [r for rows in runs.values() for r in rows]
    c = fit(allrows)
    print(f"{len(allrows)} decode-only steps from {len(runs)} runs")
    print(f"T(n, L) = {c[0]:.3f} + {c[1]:.4f}*n + {c[2] * 1000:.4f}*(L/1k)  ms")
    print(f"in-sample: R^2={r2(c, allrows):.3f}  MAPE={mape(c, allrows):.1%}")

    if len(runs) > 1:
        print("\nleave-one-run-out (fit on the others, test on this run):")
        worst = 0.0
        for name, rows in runs.items():
            rest = [r for k, rs in runs.items() if k != name for r in rs]
            try:
                m = mape(fit(rest), rows)
            except ValueError:
                continue
            worst = max(worst, m)
            print(f"  {name:<28} steps={len(rows):>7}  MAPE={m:.1%}")
        print(f"  worst held-out MAPE: {worst:.1%}  (pilot check: < 15%)")

    print("\nerror by decode count (in-sample):")
    for lo, hi in [(1, 4), (4, 8), (8, 16), (16, 32), (32, 10**9)]:
        rs = [r for r in allrows if lo <= r[0] < hi]
        if rs:
            bias = sum(predict(c, n, l) - y for n, l, y in rs) / len(rs)
            print(f"  n {lo:>2}-{min(hi - 1, 999):<3} steps={len(rs):>7}  MAPE={mape(c, rs):.1%}  "
                  f"bias={bias:+.2f} ms")
    print("\nerror by decode context (in-sample):")
    for lo, hi in [(0, 100e3), (100e3, 300e3), (300e3, 600e3), (600e3, 1.2e6), (1.2e6, float("inf"))]:
        rs = [r for r in allrows if lo <= r[1] < hi]
        if rs:
            bias = sum(predict(c, n, l) - y for n, l, y in rs) / len(rs)
            label = f"{lo / 1e3:.0f}k-{hi / 1e3:.0f}k"
            print(f"  L {label:<12} steps={len(rs):>7}  "
                  f"MAPE={mape(c, rs):.1%}  bias={bias:+.2f} ms")

    print(f"\nPRED_COEFFS={c[0]:.4f},{c[1]:.6f},{c[2]:.9f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"a": c[0], "b": c[1], "c": c[2], "steps": len(allrows),
                       "runs": sorted(runs), "r2": r2(c, allrows), "mape": mape(c, allrows)}, f, indent=1)


if __name__ == "__main__":
    main()
