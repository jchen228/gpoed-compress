"""
replicate_fig1.py
Replicates the spirit of Figure 1 from Liang et al. (2018):
  "Error-Controlled Lossy Compression Optimized for High Compression Ratios
   of Scientific Datasets"

Using CLOUDf48.bin.f32 (100×500×500) instead of NYX velocity_x.
Shows slice 50 (middle level): original | SZ | ZFP at a target abs error bound.
Reports CR, PSNR, and SSIM for each compressor.

Usage:
    python3 replicate_fig1.py [--bound 5e-4] [--level 50]

Install dependency if needed:
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
DATA_PATH   = Path.home() / "libpressio_tutorial/exercises/datasets/CLOUDf48.bin.f32"
OUT_DIR     = Path(__file__).parent
SHAPE       = (100, 500, 500)


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr(orig, recon):
    """PSNR using the original value range as peak (matches libpressio definition)."""
    data_range = orig.max() - orig.min()
    rmse = float(np.sqrt(np.mean((orig - recon) ** 2)))
    if rmse == 0:
        return float('inf')
    return float(20 * np.log10(data_range / rmse))


def ssim(orig, recon):
    """
    Structural Similarity Index (Wang et al. 2004).
    Uses skimage if available; falls back to a manual implementation.
    """
    try:
        from skimage.metrics import structural_similarity
        data_range = float(orig.max() - orig.min())
        return float(structural_similarity(orig, recon, data_range=data_range))
    except ImportError:
        pass

    # Manual SSIM (single window over full image)
    C1 = (0.01 * (orig.max() - orig.min())) ** 2
    C2 = (0.03 * (orig.max() - orig.min())) ** 2
    mu_x   = orig.mean()
    mu_y   = recon.mean()
    sig_x  = orig.var()
    sig_y  = recon.var()
    sig_xy = float(np.mean((orig - mu_x) * (recon - mu_y)))
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return float(num / den)


# ── Compression ───────────────────────────────────────────────────────────────

def compress_with(compressor_id, data_3d, abs_bound):
    """
    Compress/decompress a 3D array and return (recon_3d, compression_ratio).
    """
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config": {
            "pressio:metric": "composite",
            "composite:plugins": ["size", "error_stat"],
        },
        "compressor_config": {
            "pressio:abs": abs_bound,
        },
    })

    recon = data_3d.copy()
    compressed = comp.encode(data_3d)
    recon = comp.decode(compressed, recon)
    metrics = comp.get_metrics()
    cr = metrics.get("size:compression_ratio", float("nan"))
    return recon, cr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound", type=float, default=5e-4,
                        help="Absolute error bound (default: 5e-4)")
    parser.add_argument("--level", type=int, default=50,
                        help="Which z-level to display (default: 50)")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    data = np.fromfile(DATA_PATH, dtype=np.float32).reshape(SHAPE)
    orig_slice = data[args.level].copy()    # (500, 500)

    results = {}
    for cid in ["sz", "zfp"]:
        print(f"Compressing with {cid} at abs={args.bound:.1e} ...", end=" ", flush=True)
        try:
            recon, cr = compress_with(cid, data, args.bound)
            recon_slice = recon[args.level]
            p = psnr(orig_slice, recon_slice)
            s = ssim(orig_slice, recon_slice)
            results[cid] = {"recon": recon_slice, "cr": cr, "psnr": p, "ssim": s}
            print(f"CR={cr:.1f}×  PSNR={p:.1f} dB  SSIM={s:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    vmin = orig_slice.min()
    vmax = orig_slice.max()
    data_range = vmax - vmin

    n_panels = 1 + len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Original
    axes[0].imshow(orig_slice, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title(f"Original\nCLOUD level {args.level}", fontsize=11)
    axes[0].axis('off')

    # Compressor panels
    for ax, (cid, res) in zip(axes[1:], results.items()):
        ax.imshow(res["recon"], vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(
            f"{cid.upper()}  (abs={args.bound:.0e})\n"
            f"CR={res['cr']:.0f}×  PSNR={res['psnr']:.1f} dB  SSIM={res['ssim']:.4f}",
            fontsize=10
        )
        ax.axis('off')

    plt.suptitle(
        f"CLOUD field comparison — level {args.level} — abs bound = {args.bound:.0e}\n"
        f"(cf. Liang et al. 2018, Fig. 1)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    # ── Error maps (separate figure, matches paper style) ─────────────────────
    fig2, axes2 = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes2 = [axes2]

    for ax, (cid, res) in zip(axes2, results.items()):
        err = np.abs(orig_slice - res["recon"])
        im = ax.imshow(err, vmin=0, vmax=data_range * 0.01, cmap='hot')
        ax.set_title(
            f"{cid.upper()} absolute error\nPSNR={res['psnr']:.1f} dB  SSIM={res['ssim']:.4f}",
            fontsize=10
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Error maps — abs bound = {args.bound:.0e}",
        fontsize=12
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out1 = OUT_DIR / f"fig1_comparison_bound{args.bound:.0e}.png"
    out2 = OUT_DIR / f"fig1_errormaps_bound{args.bound:.0e}.png"
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"\nSaved:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
