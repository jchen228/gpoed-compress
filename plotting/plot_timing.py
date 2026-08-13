#!/usr/bin/env python3
"""
plot_timing.py
==============
Standalone timing figure for the Hurricane dataset.

Two panels:
  Left  — One-time training cost per method (bar chart)
  Right — Median per-compression time by method × k (grouped bars)

Reads from rd_results_Hurricane.csv (produced by hurricane_rd_comparison.py).
Saves: timing_Hurricane.png
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
ARGONNE  = Path(__file__).resolve().parent.parent
CSV_PATH = ARGONNE / "rd_results_Hurricane.csv"
OUT_PATH = ARGONNE / "timing_Hurricane.png"
DPI      = 150

# ── Colours (match hurricane_rd_comparison.py) ────────────────────────────────
COLORS = {
    "SZ2":           "#1f77b4",
    "ZFP":           "#ff7f0e",
    "DEIM-2D+SZ2":   "#2ca02c",
    "Kriging-2D+SZ2":"#9467bd",
}

METHOD_LABELS = {
    "SZ2":           "SZ2",
    "ZFP":           "ZFP",
    "DEIM-2D+SZ2":   "DEIM+SZ2",
    "Kriging-2D+SZ2":"Kriging+SZ2",
}

# ── Load CSV ──────────────────────────────────────────────────────────────────
def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    by_mk = defaultdict(list)       # (method, k) → list of rows
    train_by_method = {}            # method → train_sec (one-time)
    for r in rows:
        method = r["method"]
        k      = int(r["k"])
        try:
            cs = float(r["comp_sec"])
        except (ValueError, KeyError):
            continue
        by_mk[(method, k)].append(cs)
        if method not in train_by_method:
            try:
                train_by_method[method] = float(r["train_sec"])
            except (ValueError, KeyError):
                train_by_method[method] = 0.0
    return by_mk, train_by_method

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(by_mk, train_by_method):
    fig, (ax_train, ax_comp) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: training time ────────────────────────────────────────────────────
    methods_ordered = ["SZ2", "ZFP", "DEIM-2D+SZ2", "Kriging-2D+SZ2"]
    train_vals = [train_by_method.get(m, 0.0) for m in methods_ordered]
    bar_colors = [COLORS[m] for m in methods_ordered]
    xlabels    = [METHOD_LABELS[m] for m in methods_ordered]

    bars = ax_train.bar(xlabels, train_vals, color=bar_colors,
                        edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, train_vals):
        label = f"{val:.0f} s" if val >= 1 else "< 1 s"
        ax_train.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(train_vals) * 0.02,
                      label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_train.set_ylabel("Training time (s)", fontsize=12)
    ax_train.set_title("One-time training cost\n(offline, amortised over all compressions)",
                        fontsize=11)
    ax_train.set_ylim(0, max(train_vals) * 1.18)
    ax_train.grid(axis="y", alpha=0.3, linestyle="--")
    ax_train.spines[["top", "right"]].set_visible(False)
    ax_train.annotate("offline / one-time", xy=(0.97, 0.97),
                       xycoords="axes fraction", ha="right", va="top",
                       fontsize=9, color="gray", style="italic")

    # ── Right: per-compression time ────────────────────────────────────────────
    # Group: SZ2/ZFP (single bar each), DEIM by k, Kriging by k
    groups  = []   # label strings
    heights = []   # median comp_sec
    colors  = []

    for m in ["SZ2", "ZFP"]:
        keys = [(mm, k) for (mm, k) in by_mk if mm == m]
        if not keys:
            continue
        all_cs = [c for key in keys for c in by_mk[key]]
        groups.append(METHOD_LABELS[m])
        heights.append(float(np.median(all_cs)))
        colors.append(COLORS[m])

    for m in ["DEIM-2D+SZ2", "Kriging-2D+SZ2"]:
        ks = sorted({k for (mm, k) in by_mk if mm == m})
        for k in ks:
            cs = by_mk[(m, k)]
            groups.append(f"{METHOD_LABELS[m]}\nk={k}")
            heights.append(float(np.median(cs)))
            colors.append(COLORS[m])

    x = np.arange(len(groups))
    bars2 = ax_comp.bar(x, heights, color=colors, edgecolor="white",
                         linewidth=0.8, width=0.6)
    for bar, val in zip(bars2, heights):
        ax_comp.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.2f}s", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(groups, fontsize=8)
    ax_comp.set_ylabel("Median compression time (s)", fontsize=12)
    ax_comp.set_title("Per-compression time\n(online, per target accuracy level)",
                       fontsize=11)
    ax_comp.set_ylim(0, max(heights) * 1.18)
    ax_comp.grid(axis="y", alpha=0.3, linestyle="--")
    ax_comp.spines[["top", "right"]].set_visible(False)

    # Legend patches for method colours
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
               for m in methods_ordered]
    ax_comp.legend(handles=patches, fontsize=9, loc="upper right",
                   framealpha=0.8)

    fig.suptitle("ISABEL Hurricane TC  —  Timing breakdown", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    by_mk, train_by_method = load()
    print("Training times:")
    for m, t in sorted(train_by_method.items()):
        print(f"  {m:30s}  {t:.1f} s")
    print("\nMedian per-compression times:")
    for (m, k), cs in sorted(by_mk.items()):
        print(f"  {m:30s}  k={k:3d}  {np.median(cs):.3f} s")
    plot(by_mk, train_by_method)
