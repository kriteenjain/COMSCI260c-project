#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare a baseline result JSON against one or more compressed result JSONs.
Saves a styled PNG figure of the comparison table.

Usage:
    python results/compare.py results/baseline.json results/wanda-u50.json results/awq.json
    python results/compare.py results/baseline.json results/wanda-u50.json --f1 --out results/fig.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


TASK_METRICS = {
    "gsm8k":     ("accuracy", "accuracy"),
    "humaneval": ("pass@1",   "pass@1"),
    "musique":   ("exact_match", "EM"),
    "nq_open":   ("exact_match", "EM"),
    "squad":     ("exact_match", "EM"),
}

TASK_LABELS = {
    "squad":     "SQuAD",
    "humaneval": "HumanEval",
    "gsm8k":     "GSM8K",
    "musique":   "MuSiQue",
    "nq_open":   "NQ Open",
}

TASK_ORDER = ["squad", "humaneval", "gsm8k", "musique", "nq_open"]

# Colour palette
COL_HEADER   = "#2C3E50"   # dark slate — header bg
COL_TASK     = "#ECF0F1"   # light grey — task name bg
COL_BASELINE = "#D5E8D4"   # soft green — baseline values
COL_WHITE    = "#FFFFFF"
TXT_HEADER   = "#FFFFFF"
TXT_DARK     = "#2C3E50"

# Delta cell colours
def _delta_colour(rel: float | None) -> str:
    if rel is None:
        return COL_WHITE
    if rel >= -5:
        return "#D5E8D4"   # green  — resilient
    if rel >= -40:
        return "#FFE6CC"   # orange — degraded
    return "#F8CECC"       # red    — collapsed


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _score(result: dict, task: str) -> float | None:
    t = result.get("tasks", {}).get(task)
    return t.get(TASK_METRICS[task][0]) if t else None


def _f1(result: dict, task: str) -> float | None:
    t = result.get("tasks", {}).get(task)
    return t.get("f1") if t else None


def _tag(result: dict) -> str:
    return result.get("tag", Path(result.get("_path", "?")).stem)


def _model(result: dict) -> str:
    return result.get("model_info", {}).get("model", "?")


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _rel(base: float | None, comp: float | None) -> float | None:
    if base is None or comp is None or base == 0:
        return None
    return (comp - base) / base * 100


def _rel_str(rel: float | None) -> str:
    if rel is None:
        return "—"
    return f"{'+' if rel >= 0 else ''}{rel:.1f}%"


def build_figure(baseline, compressed_results, show_f1: bool, out_path: Path) -> None:
    n_comp = len(compressed_results)
    comp_tags = [_tag(r) for r in compressed_results]

    # ---- Build table data ------------------------------------------------
    # Columns: Task | Metric | Baseline | comp1 | comp2 | ... | (rel1 | rel2 | ...)
    col_headers = ["Task", "Metric", "Baseline"] + comp_tags
    show_rel = n_comp == 1   # relative change column only for single comparison
    if show_rel:
        col_headers += ["Change"]

    rows_data   = []   # text
    rows_colour = []   # bg colour per cell

    for task in TASK_ORDER:
        if task not in TASK_METRICS:
            continue
        label = TASK_LABELS.get(task, task)
        metric_label = TASK_METRICS[task][1]
        base = _score(baseline, task)
        comps = [_score(r, task) for r in compressed_results]
        rels = [_rel(base, c) for c in comps]

        row_txt = [label, metric_label, _fmt(base)] + [_fmt(c) for c in comps]
        row_col = [COL_TASK, COL_TASK, COL_BASELINE] + [_delta_colour(rel) for rel in rels]

        if show_rel:
            row_txt.append(_rel_str(rels[0]))
            row_col.append(_delta_colour(rels[0]))

        rows_data.append(row_txt)
        rows_colour.append(row_col)

        if show_f1 and task in ("musique", "nq_open", "squad"):
            base_f1 = _f1(baseline, task)
            comp_f1s = [_f1(r, task) for r in compressed_results]
            rel_f1s = [_rel(base_f1, cf) for cf in comp_f1s]

            f1_txt = ["", "F1", _fmt(base_f1)] + [_fmt(cf) for cf in comp_f1s]
            f1_col = [COL_WHITE, COL_WHITE, COL_BASELINE] + [_delta_colour(r) for r in rel_f1s]

            if show_rel:
                f1_txt.append(_rel_str(rel_f1s[0]))
                f1_col.append(_delta_colour(rel_f1s[0]))

            rows_data.append(f1_txt)
            rows_colour.append(f1_col)

    # ---- Draw ------------------------------------------------------------
    n_rows = len(rows_data)
    n_cols = len(col_headers)

    col_w = 1.0 / n_cols          # equal width for every column
    fig_w = max(7, 1.3 * n_cols)
    fig_h = max(2, 0.42 * (n_rows + 1)) + 0.5   # table rows + legend

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_headers,
        colWidths=[col_w] * n_cols,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Style header row
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor(COL_HEADER)
        cell.set_text_props(color=TXT_HEADER, fontweight="bold")
        cell.set_edgecolor("#FFFFFF")

    # Style data rows
    for row_idx, (row_txt, row_col) in enumerate(zip(rows_data, rows_colour)):
        for col_idx in range(n_cols):
            cell = tbl[row_idx + 1, col_idx]
            cell.set_facecolor(row_col[col_idx])
            cell.set_text_props(color=TXT_DARK)
            cell.set_edgecolor("#CCCCCC")
            # Bold task name
            if col_idx == 0 and row_txt[0]:
                cell.set_text_props(color=TXT_DARK, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color="#D5E8D4", label="Resilient  (< -5%)"),
        mpatches.Patch(color="#FFE6CC", label="Degraded  (-5% to -40%)"),
        mpatches.Patch(color="#F8CECC", label="Collapsed  (> -40%)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out_path}", file=sys.stderr)


METHOD_COLOURS = [
    "#E67E22",   # orange  — wanda u50
    "#E74C3C",   # red     — wanda 2:4
    "#2980B9",   # blue    — awq
    "#27AE60",   # green   — 4th method if needed
]


def build_delta_figure(baseline, compressed_results, out_path: Path) -> None:
    """Bar chart of % change from baseline for each task and compression method."""
    comp_tags = [_tag(r) for r in compressed_results]
    n_methods = len(compressed_results)
    n_tasks = len(TASK_ORDER)

    bar_w = 0.7 / n_methods
    x = list(range(n_tasks))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_tasks), 5))

    for m_idx, (comp, tag) in enumerate(zip(compressed_results, comp_tags)):
        offsets = [xi + (m_idx - (n_methods - 1) / 2) * bar_w for xi in x]
        rels = []
        for task in TASK_ORDER:
            b = _score(baseline, task)
            c = _score(comp, task)
            rels.append(_rel(b, c))

        colour = METHOD_COLOURS[m_idx % len(METHOD_COLOURS)]
        bars = ax.bar(
            offsets,
            [r if r is not None else 0 for r in rels],
            width=bar_w * 0.9,
            color=colour,
            label=tag,
            zorder=3,
        )

        # Value labels on bars
        for bar, rel in zip(bars, rels):
            if rel is None:
                continue
            va = "bottom" if rel >= 0 else "top"
            offset = 0.5 if rel >= 0 else -0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{rel:+.1f}%",
                ha="center", va=va,
                fontsize=7.5, color="#2C3E50",
            )

    # Baseline reference line
    ax.axhline(0, color="#2C3E50", linewidth=1.2, zorder=4)

    # Threshold bands
    ax.axhspan(-5, 0, alpha=0.06, color="#27AE60", zorder=1)
    ax.axhspan(-40, -5, alpha=0.06, color="#E67E22", zorder=1)
    ax.axhspan(ax.get_ylim()[0], -40, alpha=0.06, color="#E74C3C", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in TASK_ORDER], fontsize=10)
    ax.set_ylabel("% change vs. baseline", fontsize=10)
    ax.set_title(
        f"Performance change under compression — {_model(baseline).split('/')[-1]}",
        fontsize=12, fontweight="bold", color="#2C3E50", pad=10,
    )
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PNG comparison figures.")
    ap.add_argument("baseline", help="Baseline result JSON.")
    ap.add_argument("compressed", nargs="+", help="One or more compressed result JSONs.")
    ap.add_argument("--f1", action="store_true", help="Also show F1 for QA tasks (table only).")
    ap.add_argument("--delta", action="store_true",
                    help="Generate %% change bar chart instead of the score table.")
    ap.add_argument("--out", default=None, help="Output PNG path (auto-named if omitted).")
    args = ap.parse_args()

    baseline = _load(args.baseline)
    baseline["_path"] = args.baseline
    compressed_results = []
    for p in args.compressed:
        r = _load(p)
        r["_path"] = p
        compressed_results.append(r)

    tags = "_vs_".join(_tag(r) for r in compressed_results)
    model = _model(baseline).split("/")[-1]
    base_dir = Path(args.baseline).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.delta:
        out_path = Path(args.out) if args.out else base_dir / f"delta_{model}_{tags}.png"
        build_delta_figure(baseline, compressed_results, out_path)
    else:
        out_path = Path(args.out) if args.out else base_dir / f"compare_{model}_{tags}.png"
        build_figure(baseline, compressed_results, args.f1, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
