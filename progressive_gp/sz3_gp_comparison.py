#!/usr/bin/env python3
"""
sz3_gp_comparison.py
====================
Rate-distortion comparison of progressive GP vs SZ3 on the ISABEL hurricane
U-wind field.

Both methods are evaluated at the same set of EB values so the rate-distortion
curves are directly comparable.

For the GP:  sensor positions are fixed (from the checkpoint).  For each EB we
re-quantize the final GP predictions with that EB — no re-run required.

For SZ3:  the binary is called once per EB value.

Output: a single PSNR-vs-bit-rate plot with both curves + a summary table.
"""

from __future__ import annotations
import sys
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ARGONNE = Path(__file__).resolve().parent.parent

# Set USE_2D = True to load the 2D slice-by-slice GP checkpoint.
USE_2D = True

if USE_2D:
    CHECKPOINT_FILE = ARGONNE / "pgp2d_checkpoint_R10_k10_eb0.01_ab10_ds10_zskip0.pkl"
else:
    CHECKPOINT_FILE = ARGONNE / "pgp_checkpoint_R20_k250_eb0.01_ab10_ds10_zskip15.pkl"

# SZ3 binary
SZ3_BIN = Path("/Users/jchen228/spack/opt/spack/darwin-m1/"
               "sz3-3.4.0-cwdwysro3s55ur4mqvlhaar437kjs24w/bin/sz3")

# EB values to sweep for the rate-distortion curve.
# Both GP and SZ3 are evaluated at every value in this list.
EB_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]

# Bits to store one GP sensor: uint32 index (32 b) + float32 value (32 b).
BITS_PER_SENSOR = 64

# Output figure
OUT_FIG = ARGONNE / "sz3_gp_rd_curve.png"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def quantize_sz2(errs: np.ndarray, eb: float):
    """
    SZ2-style quantizer: bin width = 2×eb.
    Returns q_bin (int64) and q_final_err (float, |q_final_err| ≤ eb).
    """
    safe        = np.where(np.isnan(errs), 0.0, errs)
    q_bin       = np.round(safe / (2.0 * eb)).astype(np.int64)
    q_final_err = safe - q_bin * (2.0 * eb)
    return q_bin, q_final_err


def entropy_from_bins(q_bin: np.ndarray) -> float:
    """Shannon entropy (bits/symbol) of integer bin indices."""
    shifted = q_bin - q_bin.min()
    counts  = np.bincount(shifted.ravel())
    total   = counts.sum()
    p       = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def psnr_db(errors: np.ndarray, data_range: float) -> float:
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return 20.0 * np.log10(data_range / rmse) if rmse > 0 else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path):
    if not path.exists():
        sys.exit(f"Checkpoint not found: {path}\nRun the GP script first.")
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    results = payload['results']
    vol     = payload['vol']
    hp      = payload.get('hyperparams', {})
    print(f"Checkpoint : {path.name}")
    print(f"Volume     : {vol.shape}  ({vol.size:,} voxels)")
    print(f"Rounds     : {len(results['rounds'])}")
    if hp:
        print(f"Hyperparams: ls_xy={hp.get('ls_xy','n/a')}  "
              f"sig2={hp.get('sig2','n/a')}")
    return results, vol, hp


# ─────────────────────────────────────────────────────────────────────────────
# GP RATE-DISTORTION CURVE
# ─────────────────────────────────────────────────────────────────────────────

def gp_rd_curve(results: dict, vol: np.ndarray, eb_values: list[float]) -> list[dict]:
    """
    For each EB, re-quantize the GP's final-round predictions and compute
    (bit_rate, PSNR, RMSE).  Sensor positions are fixed — only the quantization
    step changes with EB.
    """
    y_full     = vol.ravel().astype(np.float64)
    N          = y_full.size
    data_range = float(y_full.max() - y_full.min())

    # Total sensors accumulated across all rounds
    k_total = sum(len(rd['sensor_idx']) for rd in results['rounds'])
    print(f"\nGP total sensors: {k_total:,} / {N:,}  "
          f"({100*k_total/N:.2f}% of voxels)\n")

    # Final-round prediction errors (post all rounds)
    last_errs = results['rounds'][-1]['err_vals']

    points = []
    print(f"{'EB':>8}  {'H':>8}  {'bit_rate':>10}  {'PSNR':>8}  {'RMSE':>10}")
    print("-" * 52)
    for eb in eb_values:
        q_bin, q_err = quantize_sz2(last_errs, eb)
        H        = entropy_from_bins(q_bin)
        bit_rate = (k_total * BITS_PER_SENSOR + H * N) / N
        psnr     = psnr_db(q_err, data_range)
        rmse     = float(np.sqrt(np.mean(q_err ** 2)))
        print(f"{eb:>8.4f}  {H:>8.4f}  {bit_rate:>10.4f}  {psnr:>8.2f}  {rmse:>10.6f}")
        points.append(dict(eb=eb, H=H, bit_rate=bit_rate, psnr=psnr, rmse=rmse))
    return points


# ─────────────────────────────────────────────────────────────────────────────
# SZ3 RATE-DISTORTION CURVE
# ─────────────────────────────────────────────────────────────────────────────

def sz3_rd_curve(vol: np.ndarray, eb_values: list[float]) -> list[dict]:
    """Run SZ3 at each EB value and collect (bit_rate, PSNR, RMSE)."""
    if not SZ3_BIN.exists():
        sys.exit(f"SZ3 binary not found: {SZ3_BIN}")

    vol32      = vol.astype(np.float32)
    N          = vol32.size
    data_range = float(vol.max() - vol.min())
    nz, ny, nx = vol32.shape

    points = []
    print(f"\n{'EB':>8}  {'bit_rate':>10}  {'PSNR':>8}  {'RMSE':>10}  {'ratio':>8}")
    print("-" * 52)

    for eb in eb_values:
        with tempfile.TemporaryDirectory() as tmp:
            tmp   = Path(tmp)
            f_in  = tmp / "vol.f32"
            f_sz  = tmp / "vol.sz3"
            f_dec = tmp / "vol_dec.f32"

            vol32.tofile(f_in)
            cmd = [str(SZ3_BIN), "-f",
                   "-i", str(f_in), "-z", str(f_sz), "-o", str(f_dec),
                   "-3", str(nx), str(ny), str(nz),
                   "-M", "ABS", str(eb)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  SZ3 failed at EB={eb}: {result.stderr.strip()}")
                continue
            if not f_sz.exists():
                alt = f_in.with_suffix('.f32.sz3')
                if alt.exists():
                    alt.rename(f_sz)
                else:
                    print(f"  SZ3 produced no output at EB={eb}")
                    continue

            compressed_bytes = f_sz.stat().st_size
            vol_dec = (np.fromfile(f_dec, dtype=np.float32)
                       .reshape(vol32.shape).astype(np.float64))

        errors   = vol.astype(np.float64) - vol_dec
        q_bin, q_err = quantize_sz2(errors, eb)
        H        = entropy_from_bins(q_bin)
        bit_rate = (compressed_bytes * 8) / N
        psnr     = psnr_db(q_err, data_range)
        rmse     = float(np.sqrt(np.mean(q_err ** 2)))
        ratio    = (N * 4) / compressed_bytes   # vs raw float32

        print(f"{eb:>8.4f}  {bit_rate:>10.4f}  {psnr:>8.2f}  {rmse:>10.6f}  {ratio:>8.2f}×")
        points.append(dict(eb=eb, H=H, bit_rate=bit_rate, psnr=psnr, rmse=rmse))

    return points


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_rd_curves(gp_pts: list[dict], sz3_pts: list[dict]) -> None:
    gp_br   = [p['bit_rate'] for p in gp_pts]
    gp_psnr = [p['psnr']     for p in gp_pts]
    gp_rmse = [p['rmse']     for p in gp_pts]
    gp_eb   = [p['eb']       for p in gp_pts]

    sz3_br   = [p['bit_rate'] for p in sz3_pts]
    sz3_psnr = [p['psnr']     for p in sz3_pts]
    sz3_rmse = [p['rmse']     for p in sz3_pts]
    sz3_eb   = [p['eb']       for p in sz3_pts]

    GP_COLOR  = '#1f77b4'
    SZ3_COLOR = '#d62728'

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Rate-distortion: Progressive GP vs SZ3  (U-wind, ISABEL hurricane)\n"
        "Both evaluated at the same EB values — post-quantisation metrics",
        fontsize=12, fontweight='bold')

    for ax, gp_y, sz3_y, ylabel, title in [
        (axes[0], gp_psnr, sz3_psnr,
         'PSNR (dB)', 'PSNR vs Bit Rate  (higher = better)'),
        (axes[1], gp_rmse, sz3_rmse,
         'RMSE (m/s)', 'RMSE vs Bit Rate  (lower = better)'),
    ]:
        ax.plot(gp_br,  gp_y,  'o-', color=GP_COLOR,  lw=2, ms=7, label='Progressive GP')
        ax.plot(sz3_br, sz3_y, 's--', color=SZ3_COLOR, lw=2, ms=7, label='SZ3')

        # Annotate EB values on GP curve
        for br, y, eb in zip(gp_br, gp_y, gp_eb):
            ax.annotate(f'EB={eb}', xy=(br, y),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=GP_COLOR, alpha=0.8)
        for br, y, eb in zip(sz3_br, sz3_y, sz3_eb):
            ax.annotate(f'EB={eb}', xy=(br, y),
                        xytext=(4, -10), textcoords='offset points',
                        fontsize=7, color=SZ3_COLOR, alpha=0.8)

        ax.set_xlabel('Bit rate (bits/sample)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25, ls='--')
        ax.spines[['top', 'right']].set_visible(False)

    # Summary table
    col_labels = ['EB', 'GP bit rate', 'SZ3 bit rate', 'GP PSNR', 'SZ3 PSNR',
                  'GP RMSE', 'SZ3 RMSE']
    sz3_by_eb = {p['eb']: p for p in sz3_pts}
    rows = []
    for gp in gp_pts:
        eb  = gp['eb']
        sz3 = sz3_by_eb.get(eb, {})
        rows.append([
            str(eb),
            f"{gp['bit_rate']:.3f}",
            f"{sz3.get('bit_rate', 'n/a'):.3f}" if sz3 else 'n/a',
            f"{gp['psnr']:.2f} dB",
            f"{sz3.get('psnr', 'n/a'):.2f} dB" if sz3 else 'n/a',
            f"{gp['rmse']:.5f}",
            f"{sz3.get('rmse', 'n/a'):.5f}" if sz3 else 'n/a',
        ])

    fig.subplots_adjust(bottom=0.30, wspace=0.35)
    tbl_ax = fig.add_axes([0.05, 0.01, 0.9, 0.22])
    tbl_ax.axis('off')
    t = tbl_ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for j in range(len(col_labels)):
        t[0, j].set_facecolor('#dce6f1')
        t[0, j].set_text_props(fontweight='bold')

    # Highlight better value per row (GP vs SZ3 at same EB)
    for row_i, gp in enumerate(gp_pts, start=1):
        eb  = gp['eb']
        sz3 = sz3_by_eb.get(eb)
        if not sz3:
            continue
        # bit rate: lower is better
        if gp['bit_rate'] < sz3['bit_rate']:
            t[row_i, 1].set_facecolor('#d5f5e3')
        else:
            t[row_i, 2].set_facecolor('#d5f5e3')
        # PSNR: higher is better
        if gp['psnr'] > sz3['psnr']:
            t[row_i, 3].set_facecolor('#d5f5e3')
        else:
            t[row_i, 4].set_facecolor('#d5f5e3')
        # RMSE: lower is better
        if gp['rmse'] < sz3['rmse']:
            t[row_i, 5].set_facecolor('#d5f5e3')
        else:
            t[row_i, 6].set_facecolor('#d5f5e3')

    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {OUT_FIG}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results, vol, hp = load_checkpoint(CHECKPOINT_FILE)

    print("\n── GP rate-distortion curve ──────────────────────────────────")
    gp_pts = gp_rd_curve(results, vol, EB_VALUES)

    print("\n── SZ3 rate-distortion curve ─────────────────────────────────")
    sz3_pts = sz3_rd_curve(vol, EB_VALUES)

    print("\n── Plotting ──────────────────────────────────────────────────")
    plot_rd_curves(gp_pts, sz3_pts)
    print("Done.")
