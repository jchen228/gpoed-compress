#!/usr/bin/env python3
"""
plot_timing_sst.py
==================
Timing figure for the SST dataset.

Per-compression time by method (median across k values and abs_bounds).
Training times were not recorded for SST; a note is included.

Reads from rd_results_SST.csv (produced by sst_rd_comparison.py).
Saves: timing_SST.png
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
ARGONNE  = Path(__file__).resolve().parent.parent
CSV_PATH = ARGONNE / "rd_results_SST.csv"
OUT_PATH = ARGONNE / "timing_SST.png"
DPI      = 150

# ── Display order and labels ──────────────────────────────────────────────────
# Each entry: (method_in_csv, display_label, color)
METHODS = [
    ("SZ2",            "SZ2\n(2-D)",      "#1f77b4"),
    ("SZ2-1D",         "SZ2-1D\n(ocean)", "#aec7e8"),
    ("ZFP",            "ZFP\n(2-D)",      "#ff7f0e"),
    ("ZFP-1D",         "ZFP-1D\n(ocean)", "#ffbb78"),
    ("DEIM-2D+SZ2",    "DEIM\n+SZ2",      "#2ca02c"),
    ("DEIM-2D+ZFP",    "DEIM\n+ZFP",      "#98df8a"),
    ("Kriging-2D+SZ2", "Kriging\n+SZ2",   "#9467bd"),
    ("Kriging-2D+ZFP", "Kriging\n+ZFP",   "#c5b0d5"),
]

# ── Load CSV ──────────────────────────────────────────────────────────────────
def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    by_method = defaultdict(list)   # method → all comp_sec values
    for r in rows:
        try:
            by_method[r["method"]].append(float(r["comp_sec"]))
        except (ValueError, KeyError):
            continue
    return by_method

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(by_method):
    fig, ax = plt.subplots(figsize=(11, 5))

    labels  = []
    heights = []
    colors  = []
    errs    = []

    for method, label, color in METHODS:
        vals = by_method.get(method, [])
        if not vals:
            continue
        labels.append(label)
        heights.append(float(np.median(vals)))
        errs.append(float(np.std(vals)))
        colors.append(color)

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.6,
                  yerr=errs, capsize=4, error_kw={"elinewidth": 1.2,
                                                   "ecolor": "#444"})

    for bar, val in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(heights) * 0.01,
                f"{val:.2f}s", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Median per-compression time (s)", fontsize=12)
    ax.set_title(
        "SST  —  Per-compression time by method\n"
        "(median ± std across all k and accuracy levels; "
        "training times not recorded for SST)",
        fontsize=11)
    ax.set_ylim(0, max(heights) * 1.22)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Group annotations
    yann = max(heights) * 1.16
    def _brace(x0, x1, label, color):
        mid = (x0 + x1) / 2
        ax.annotate("", xy=(x0, yann), xytext=(x1, yann),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.5))
        ax.text(mid, yann + max(heights) * 0.01, label, ha="center",
                va="bottom", fontsize=8, color=color)

    # Baselines: indices 0-3, Hybrids: 4-7
    n_base = sum(1 for m, _, _ in METHODS if m in by_method
                 and m in ("SZ2", "SZ2-1D", "ZFP", "ZFP-1D"))
    n_hyb  = len(labels) - n_base
    if n_base > 0:
        _brace(-0.4, n_base - 0.6, "Baselines", "#555")
    if n_hyb > 0:
        _brace(n_base - 0.4, len(labels) - 0.6, "Hybrid methods", "#555")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    by_method = load()
    print("Median per-compression times:")
    for method, label, _ in METHODS:
        vals = by_method.get(method, [])
        if vals:
            print(f"  {method:30s}  {np.median(vals):.3f}s  (n={len(vals)})")
    plot(by_method)
