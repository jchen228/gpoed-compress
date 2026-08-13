#!/usr/bin/env python3
"""
plot_timing_amortized.py
========================
Amortized-cost figure for the Hurricane dataset.

X-axis: number of snapshots compressed
Y-axis: cumulative wall-clock time (training + compression)

Training is ONE-TIME — the model does not need to be retrained for each
new snapshot. Each subsequent compression only requires:
  (1) reading k sensor values (not the full field)
  (2) applying the pre-trained predictor (fast matrix multiply)
  (3) compressing the residual with SZ2

Shows break-even points where hybrid total time falls below SZ2 baseline.

Uses per-compression medians from rd_results_Hurricane.csv.
Training times: DEIM=10.9s, Kriging=339.4s (one-time, from checkpoint).

Saves: timing_amortized_Hurricane.png
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
ARGONNE  = Path(__file__).resolve().parent
CSV_PATH = ARGONNE / "rd_results_Hurricane.csv"
OUT_PATH = ARGONNE / "timing_amortized_Hurricane.png"
DPI      = 150

# ── Which k to show for each hybrid ──────────────────────────────────────────
# Use the k with lowest per-compression time (most favourable amortization)
DEIM_K  = 80    # k=80 has lowest per-comp time (~0.73s)
KRIG_K  = 400   # all Kriging k are ~equal; use largest (best RD)

N_MAX   = 500   # max snapshots on x-axis

# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "SZ2":            "#1f77b4",
    "ZFP":            "#ff7f0e",
    "DEIM-2D+SZ2":    "#2ca02c",
    "Kriging-2D+SZ2": "#9467bd",
}

LABELS = {
    "SZ2":            "SZ2 (no training)",
    "ZFP":            "ZFP (no training)",
    f"DEIM k={DEIM_K}":   f"DEIM+SZ2  k={DEIM_K}  (train once: ~11 s)",
    f"Kriging k={KRIG_K}": f"Kriging+SZ2  k={KRIG_K}  (train once: ~339 s)",
}

# ── Load per-compression medians ──────────────────────────────────────────────
def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    by_mk = defaultdict(list)
    train_by_method = {}
    for r in rows:
        m, k = r["method"], int(r["k"])
        try:
            by_mk[(m, k)].append(float(r["comp_sec"]))
        except (ValueError, KeyError):
            continue
        if m not in train_by_method:
            try:
                train_by_method[m] = float(r["train_sec"])
            except (ValueError, KeyError):
                train_by_method[m] = 0.0
    return by_mk, train_by_method

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(by_mk, train_by_method):
    ns = np.arange(0, N_MAX + 1)

    # Per-compression medians
    sz2_c  = np.median(by_mk[("SZ2", 0)])
    zfp_c  = np.median(by_mk[("ZFP", 0)])
    deim_c = np.median(by_mk[("DEIM-2D+SZ2", DEIM_K)])
    krig_c = np.median(by_mk[("Kriging-2D+SZ2", KRIG_K)])

    deim_train = train_by_method.get("DEIM-2D+SZ2", 10.9)
    krig_train = train_by_method.get("Kriging-2D+SZ2", 339.4)

    curves = {
        "SZ2":             (0.0,        sz2_c,  COLORS["SZ2"],            "-",  2.2),
        "ZFP":             (0.0,        zfp_c,  COLORS["ZFP"],            "--", 2.2),
        f"DEIM k={DEIM_K}":  (deim_train, deim_c, COLORS["DEIM-2D+SZ2"],  "-",  2.0),
        f"Kriging k={KRIG_K}":(krig_train, krig_c, COLORS["Kriging-2D+SZ2"], "-", 2.0),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, (train, comp, color, ls, lw) in curves.items():
        total = train + ns * comp
        ax.plot(ns, total, ls=ls, color=color, lw=lw, label=LABELS.get(key, key))

    # ── Break-even annotations ─────────────────────────────────────────────────
    def breakeven(train, comp_hyb, comp_base):
        """n where train + n*comp_hyb == n*comp_base"""
        if comp_hyb >= comp_base:
            return None
        return train / (comp_base - comp_hyb)

    be_deim = breakeven(deim_train, deim_c, sz2_c)
    be_krig = breakeven(krig_train, krig_c, sz2_c)

    ymax = (krig_train + N_MAX * max(sz2_c, krig_c)) * 1.05

    if be_deim is not None and be_deim <= N_MAX:
        be_y = deim_train + be_deim * deim_c
        ax.axvline(be_deim, color=COLORS["DEIM-2D+SZ2"], lw=1.0,
                   ls=":", alpha=0.7)
        ax.annotate(f"Break-even\nn≈{be_deim:.0f}",
                    xy=(be_deim, be_y),
                    xytext=(be_deim + N_MAX * 0.04, be_y + ymax * 0.04),
                    fontsize=9, color=COLORS["DEIM-2D+SZ2"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["DEIM-2D+SZ2"],
                                   lw=1.0))

    if be_krig is not None and be_krig <= N_MAX:
        be_y = krig_train + be_krig * krig_c
        ax.axvline(be_krig, color=COLORS["Kriging-2D+SZ2"], lw=1.0,
                   ls=":", alpha=0.7)
        ax.annotate(f"Break-even\nn≈{be_krig:.0f}",
                    xy=(be_krig, be_y),
                    xytext=(be_krig + N_MAX * 0.04, be_y + ymax * 0.04),
                    fontsize=9, color=COLORS["Kriging-2D+SZ2"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["Kriging-2D+SZ2"],
                                   lw=1.0))

    # ── Print per-comp rates ───────────────────────────────────────────────────
    print(f"SZ2    per-compression: {sz2_c:.3f}s")
    print(f"ZFP    per-compression: {zfp_c:.3f}s")
    print(f"DEIM   train: {deim_train:.1f}s   per-compression: {deim_c:.3f}s"
          + (f"   break-even with SZ2: n≈{be_deim:.0f}" if be_deim else "  (no break-even in range)"))
    print(f"Kriging train: {krig_train:.1f}s   per-compression: {krig_c:.3f}s"
          + (f"   break-even with SZ2: n≈{be_krig:.0f}" if be_krig else "  (no break-even in range)"))

    ax.set_xlabel("Number of snapshots compressed", fontsize=12)
    ax.set_ylabel("Cumulative time (s)", fontsize=12)
    ax.set_title(
        "Amortized compression cost  —  ISABEL Hurricane TC\n"
        "Training is one-time; subsequent snapshots need only k sensor reads + residual compression",
        fontsize=11)
    ax.set_xlim(0, N_MAX)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    by_mk, train_by_method = load()
    plot(by_mk, train_by_method)
