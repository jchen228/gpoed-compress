"""
replicate_fig8.py
Replicates Figure 8 from Liang et al. (2018):
  "Data Distortion of Hurricane(CLOUDf:slice 50) with CR=66:1"

Layout: original | SZ | ZFP  at CR≈66:1 on CLOUDf48.bin.f32, slice 50.
Reports CR, PSNR, SSIM for each compressor.

Paper reference values (SZ 1.4.13, ZFP 0.5.2):
  SZ:  PSNR=29.9, SSIM=0.6573
  ZFP: PSNR=22.5, SSIM=0.8893

Usage:
    python3 replicate_fig8.py [--target-cr 66] [--level 50]

Install if needed:
    pip install scikit-image --break-system-packages
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libpressio

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path.home() / "libpressio_tutorial/exercises/datasets/CLOUDf48.bin.f32"
OUT_DIR   = Path(__file__).parent
SHAPE     = (100, 500, 500)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_psnr(orig, recon):
    data_range = float(orig.max() - orig.min())
    rmse = float(np.sqrt(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)))
    if rmse == 0:
        return float('inf')
    return 20.0 * np.log10(data_range / rmse)


def compute_ssim(orig, recon):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(orig, recon,
                                           data_range=float(orig.max() - orig.min())))
    except ImportError:
        pass
    # Global-window fallback
    x, y = orig.astype(np.float64), recon.astype(np.float64)
    dr  = float(orig.max() - orig.min())
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mx, my = x.mean(), y.mean()
    sx, sy = x.var(), y.var()
    sxy = float(np.mean((x - mx) * (y - my)))
    return float(((2*mx*my + C1)*(2*sxy + C2)) / ((mx**2 + my**2 + C1)*(sx + sy + C2)))


# ── Compression helpers ───────────────────────────────────────────────────────

def try_compress(compressor_id, data_3d, abs_bound):
    """Returns (recon_3d, cr) or raises.
    ZFP uses fixed-rate mode (rate = 32/target_cr bits/value) to match paper.
    SZ uses fixed-accuracy mode (abs bound).
    abs_bound is interpreted as a rate (bits/value) for ZFP, abs bound for SZ.
    """
    if compressor_id == "zfp":
        config = {"zfp:rate": abs_bound}   # abs_bound repurposed as rate (bpv) for ZFP
    else:
        config = {"pressio:abs": abs_bound}

    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config": {
            "pressio:metric": "composite",
            "composite:plugins": ["size"],
        },
        "compressor_config": config,
    })
    recon = data_3d.copy()
    compressed = comp.encode(data_3d)
    recon = comp.decode(compressed, recon)
    cr = comp.get_metrics().get("size:compression_ratio", float("nan"))
    return recon, cr


def find_bound_for_cr(compressor_id, data_3d, target_cr, n_probe=12):
    """
    For ZFP: directly compute rate = 32/target_cr (fixed-rate mode gives exact CR).
    For SZ: binary-search abs bound so that CR ≈ target_cr.
    Searches over [1e-8, 1e-1].
    """
    # ZFP fixed-rate: rate = bits_per_value = 32 / CR (exact, no search needed)
    if compressor_id == "zfp":
        rate = 32.0 / target_cr
        recon, cr = try_compress("zfp", data_3d, rate)
        return rate, cr

    lo, hi = 1e-8, 1e-1
    best_bound, best_cr = hi, None

    # First probe log-spaced to bracket the target
    probes = np.logspace(np.log10(lo), np.log10(hi), n_probe)
    crs = []
    for b in probes:
        try:
            _, cr = try_compress(compressor_id, data_3d, b)
            crs.append((b, cr))
        except Exception:
            crs.append((b, float('nan')))

    # Find nearest bracket around target_cr (larger bound → higher CR)
    valid = [(b, cr) for b, cr in crs if not np.isnan(cr)]
    if not valid:
        raise RuntimeError(f"No valid compressions for {compressor_id}")

    # Sort by bound ascending (smaller bound → lower CR)
    valid.sort(key=lambda x: x[0])
    crs_only = [cr for _, cr in valid]

    # Find where target_cr sits
    best = min(valid, key=lambda x: abs(x[1] - target_cr))
    lo_b = max(b for b, cr in valid if cr >= target_cr) if any(cr >= target_cr for _, cr in valid) else valid[0][0]
    hi_b = min(b for b, cr in valid if cr <= target_cr) if any(cr <= target_cr for _, cr in valid) else valid[-1][0]

    # Binary search refinement (5 iterations)
    for _ in range(8):
        mid = np.sqrt(lo_b * hi_b)
        try:
            _, cr = try_compress(compressor_id, data_3d, mid)
            if cr > target_cr:
                lo_b = mid
            else:
                hi_b = mid
            if abs(cr - target_cr) < abs(best[1] - target_cr):
                best = (mid, cr)
        except Exception:
            break

    return best[0], best[1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cr", type=float, default=66.0,
                        help="Target compression ratio (default: 66, matching paper)")
    parser.add_argument("--level", type=int, default=50,
                        help="Which z-level to display (default: 50, matching paper)")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    data = np.fromfile(DATA_PATH, dtype=np.float32).reshape(SHAPE)
    orig_slice = data[args.level].astype(np.float32)

    results = {}
    for cid in ["sz", "zfp"]:
        print(f"\n[{cid.upper()}] Searching for bound giving CR≈{args.target_cr:.0f}× ...")
        try:
            bound, cr = find_bound_for_cr(cid, data, args.target_cr)
            recon, cr = try_compress(cid, data, bound)
            recon_slice = recon[args.level].astype(np.float32)
            p = compute_psnr(orig_slice, recon_slice)
            s = compute_ssim(orig_slice, recon_slice)
            results[cid] = {
                "recon":  recon_slice,
                "bound":  bound,
                "cr":     cr,
                "psnr":   p,
                "ssim":   s,
            }
            param_label = f"rate={bound:.3f} bpv" if cid == "zfp" else f"abs={bound:.2e}"
            print(f"  {param_label}  CR={cr:.1f}×  PSNR={p:.1f} dB  SSIM={s:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not results:
        print("No compressors succeeded. Exiting.")
        sys.exit(1)

    # ── Figure 8 style: original | SZ | ZFP ───────────────────────────────────
    vmin, vmax = orig_slice.min(), orig_slice.max()
    panels = [("Original", orig_slice, None)] + [
        (cid.upper(), res["recon"], res) for cid, res in results.items()
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, arr, res) in zip(axes, panels):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap='jet')
        if res is None:
            ax.set_title(f"(a) Original\nCLOUDf slice {args.level}", fontsize=10)
        else:
            panel_letter = chr(ord('b') + list(results.keys()).index(label.lower()))
            ax.set_title(
                f"({panel_letter}) {label}  "
                f"(CR={res['cr']:.0f}:1)\n"
                f"PSNR={res['psnr']:.1f} dB  SSIM={res['ssim']:.4f}",
                fontsize=9
            )
        ax.axis('off')

    # Paper reference in subtitle
    ref_str = "Paper ref (CR≈66:1):  SZ PSNR=29.9/SSIM=0.6573 | ZFP PSNR=22.5/SSIM=0.8893"
    plt.suptitle(
        f"Data Distortion of Hurricane(CLOUDf:slice {args.level})\n"
        f"[cf. Liang et al. 2018, Fig. 8]  —  {ref_str}",
        fontsize=9, y=1.02
    )
    plt.tight_layout()

    out = OUT_DIR / f"fig8_cloud_slice{args.level}_cr{args.target_cr:.0f}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # ── Error maps ────────────────────────────────────────────────────────────
    if results:
        fig2, axes2 = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5))
        if len(results) == 1:
            axes2 = [axes2]
        dr = float(orig_slice.max() - orig_slice.min())
        for ax, (cid, res) in zip(axes2, results.items()):
            err = np.abs(orig_slice - res["recon"])
            im  = ax.imshow(err, vmin=0, vmax=dr * 0.01, cmap='hot')
            ax.set_title(
                f"{cid.upper()} absolute error  (CR={res['cr']:.0f}:1)\n"
                f"PSNR={res['psnr']:.1f} dB  SSIM={res['ssim']:.4f}",
                fontsize=9
            )
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out2 = OUT_DIR / f"fig8_errormaps_slice{args.level}_cr{args.target_cr:.0f}.png"
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
