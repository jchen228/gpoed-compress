"""
q2_zfp_vs_sz.py  —  libpressio tutorial Exercise 1, Question 2

Compares ZFP (accuracy mode) vs SZ (abs mode) on the CLOUD dataset
at matched error bounds, measuring:
  - compression ratio
  - max absolute error
  - RMSE
  - PSNR

Run from the Argonne directory:
    python3 q2_zfp_vs_sz.py
"""

import numpy as np
import sys

try:
    import libpressio
    from libpressio import PressioCompressor
except ImportError:
    sys.exit("libpressio not found — activate your conda environment first.")

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "/Users/jchen228/Desktop/Argonne/100x500x500/CLOUDf48.bin.f32"
DATA_SHAPE  = (100, 500, 500)
DATA_DTYPE  = np.float32

print(f"Loading {DATA_PATH}...")
data = np.fromfile(DATA_PATH, dtype=DATA_DTYPE).reshape(DATA_SHAPE)
data_range = float(data.max() - data.min())
print(f"  shape={data.shape}  dtype={data.dtype}")
print(f"  min={data.min():.6f}  max={data.max():.6f}  range={data_range:.6f}")

# ── Error bounds to sweep ─────────────────────────────────────────────────────
# Use the same bounds as Q1 so results are directly comparable.
bounds = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

# ── Helper ────────────────────────────────────────────────────────────────────

def run_compressor(name, options, data):
    """Compress+decompress with given options; return (ratio, max_err, rmse, psnr)."""
    comp  = PressioCompressor(name, options)
    compressed   = comp.encode(data)
    decompressed = comp.decode(compressed, data)

    ratio     = data.nbytes / len(compressed)
    diff      = decompressed.astype(np.float64) - data.astype(np.float64)
    max_err   = float(np.abs(diff).max())
    rmse      = float(np.sqrt(np.mean(diff ** 2)))
    data_range_val = float(data.max() - data.min())
    psnr      = float(20 * np.log10(data_range_val / rmse)) if rmse > 0 else np.inf
    return ratio, max_err, rmse, psnr


# ── Run comparison ────────────────────────────────────────────────────────────
print(f"\n{'═'*80}")
print(f"  Exercise 1 Q2 — ZFP (accuracy) vs SZ (abs) on CLOUD dataset")
print(f"  Data range: {data_range:.6f}")
print(f"{'═'*80}")

header = (f"  {'Method':<12} {'Bound':>10} {'Ratio':>8} "
          f"{'Max|err|':>12} {'RMSE':>12} {'PSNR':>8}")
sep    = "  " + "─" * (len(header) - 2)
print(f"\n{header}")
print(sep)

sz_rows  = []
zfp_rows = []

for bound in bounds:
    # SZ — absolute error bound
    sz_opts = {
        "sz:error_bound_mode_str": "abs",
        "sz:absolute_error_bound": bound,
    }
    try:
        ratio, max_err, rmse, psnr = run_compressor("sz", sz_opts, data)
        row = (bound, ratio, max_err, rmse, psnr)
        sz_rows.append(row)
        print(f"  {'SZ abs':<12} {bound:>10.2e} {ratio:>8.1f} "
              f"{max_err:>12.2e} {rmse:>12.2e} {psnr:>8.1f}")
    except Exception as e:
        print(f"  {'SZ abs':<12} {bound:>10.2e}  ERROR: {e}")

print(sep)

for bound in bounds:
    # ZFP — accuracy mode  (guarantees max error ≤ bound per value)
    zfp_opts = {
        "zfp:accuracy": bound,
    }
    try:
        ratio, max_err, rmse, psnr = run_compressor("zfp", zfp_opts, data)
        row = (bound, ratio, max_err, rmse, psnr)
        zfp_rows.append(row)
        print(f"  {'ZFP acc':<12} {bound:>10.2e} {ratio:>8.1f} "
              f"{max_err:>12.2e} {rmse:>12.2e} {psnr:>8.1f}")
    except Exception as e:
        print(f"  {'ZFP acc':<12} {bound:>10.2e}  ERROR: {e}")

print(sep)

# ── Rate-distortion comparison at matched bounds ──────────────────────────────
if sz_rows and zfp_rows:
    print(f"\n  Rate-distortion (higher ratio = better compression at same quality):")
    print(f"  {'Bound':>10}  {'SZ ratio':>10}  {'ZFP ratio':>10}  "
          f"{'SZ RMSE':>12}  {'ZFP RMSE':>12}  {'Winner':>8}")
    print("  " + "─" * 70)
    for sz_r, zfp_r in zip(sz_rows, zfp_rows):
        bound = sz_r[0]
        winner = "SZ" if sz_r[1] > zfp_r[1] else "ZFP"
        print(f"  {bound:>10.2e}  {sz_r[1]:>10.1f}  {zfp_r[1]:>10.1f}  "
              f"{sz_r[3]:>12.2e}  {zfp_r[3]:>12.2e}  {winner:>8}")

print(f"\n{'═'*80}")
print("""
Key observations to look for:
  1. At very tight bounds (1e-4): which compressor achieves better ratio?
  2. At loose bounds (1e-1): does the winner change?
  3. ZFP accuracy mode guarantees per-element max error; SZ abs does too.
     Compare whether their actual max errors respect the requested bound.
  4. CLOUD has a tiny value range (~0.002), so an abs bound of 0.01 is very loose
     relative to the data (same issue as Q1 with SZ abs).
""")
