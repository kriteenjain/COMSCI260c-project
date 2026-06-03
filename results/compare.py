#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare a baseline result JSON against one or more compressed result JSONs.

Usage:
    # Single comparison
    python results/compare.py results/baseline.json results/wanda-u50.json

    # Multiple compressed runs against one baseline
    python results/compare.py results/baseline.json results/wanda-u50.json results/wanda-2-4.json results/awq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Task name → (primary metric key, display name, higher-is-better)
TASK_METRICS = {
    "gsm8k":     ("accuracy",    "accuracy", True),
    "humaneval": ("pass@1",      "pass@1",   True),
    "musique":   ("exact_match", "EM",       True),
    "nq_open":   ("exact_match", "EM",       True),
    "squad":     ("exact_match", "EM",       True),
}

TASK_ORDER = ["squad", "humaneval", "gsm8k", "musique", "nq_open"]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _score(result: dict, task: str) -> float | None:
    t = result.get("tasks", {}).get(task)
    if not t:
        return None
    key = TASK_METRICS[task][0]
    return t.get(key)


def _f1(result: dict, task: str) -> float | None:
    t = result.get("tasks", {}).get(task)
    if not t:
        return None
    return t.get("f1")


def _tag(result: dict) -> str:
    return result.get("tag", Path(result.get("_path", "?")).stem)


def _model_slug(result: dict) -> str:
    return result.get("model_info", {}).get("model", "?").split("/")[-1]


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "-"


def _delta_str(base: float | None, comp: float | None) -> str:
    if base is None or comp is None:
        return "-"
    d = comp - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _rel_str(base: float | None, comp: float | None) -> str:
    if base is None or comp is None or base == 0:
        return "-"
    rel = (comp - base) / base * 100
    sign = "+" if rel >= 0 else ""
    return f"{sign}{rel:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare baseline vs compressed result JSONs.")
    ap.add_argument("baseline", help="Path to the baseline result JSON.")
    ap.add_argument("compressed", nargs="+", help="One or more compressed result JSONs.")
    ap.add_argument("--f1", action="store_true", help="Also show F1 scores for QA tasks.")
    args = ap.parse_args()

    baseline = _load(args.baseline)
    baseline["_path"] = args.baseline
    compressed_results = []
    for p in args.compressed:
        r = _load(p)
        r["_path"] = p
        compressed_results.append(r)

    print(f"\nBaseline : {_tag(baseline)}  [{_model_slug(baseline)}]")
    for r in compressed_results:
        print(f"Compressed: {_tag(r)}  [{_model_slug(r)}]")
    print()

    # --- Primary metrics table ---
    headers = ["task", "metric", "baseline"] + [_tag(r) for r in compressed_results]
    if len(compressed_results) == 1:
        headers += ["delta", "relative"]

    rows = []
    for task in TASK_ORDER:
        if task not in TASK_METRICS:
            continue
        _, metric_label, _ = TASK_METRICS[task]
        base_score = _score(baseline, task)

        row = [task, metric_label, _fmt(base_score)]
        comp_scores = [_score(r, task) for r in compressed_results]
        row += [_fmt(s) for s in comp_scores]

        if len(compressed_results) == 1:
            row += [_delta_str(base_score, comp_scores[0]),
                    _rel_str(base_score, comp_scores[0])]
        rows.append(row)

        if args.f1 and task in ("musique", "nq_open", "squad"):
            base_f1 = _f1(baseline, task)
            f1_row = ["", "F1", _fmt(base_f1)]
            f1_row += [_fmt(_f1(r, task)) for r in compressed_results]
            if len(compressed_results) == 1:
                f1_row += [_delta_str(base_f1, _f1(compressed_results[0], task)),
                           _rel_str(base_f1, _f1(compressed_results[0], task))]
            rows.append(f1_row)

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        # plain fallback
        col_w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
        fmt_row = lambda r: "  ".join(str(r[i]).ljust(col_w[i]) for i in range(len(r)))
        print(fmt_row(headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt_row(row))

    # --- Summary: overall degradation ---
    if len(compressed_results) == 1:
        print()
        print("Summary:")
        drops = []
        for task in TASK_ORDER:
            b = _score(baseline, task)
            c = _score(compressed_results[0], task)
            if b is not None and c is not None and b > 0:
                drops.append((task, b, c, (c - b) / b * 100))

        drops.sort(key=lambda x: x[3], reverse=True)
        for task, b, c, rel in drops:
            direction = "resilient" if rel >= -10 else ("degraded" if rel >= -40 else "collapsed")
            print(f"  {task:<12} {b:.3f} → {c:.3f}  ({rel:+.1f}%)  [{direction}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
