#!/usr/bin/env python3
"""
hurricane_rd_comparison.py
==========================
Rate-distortion comparison on ISABEL hurricane simulation data.

Methods (4-method comparison on TC variable)
---------------------------------------------
  SZ2              — libpressio SZ2, absolute error mode, 2-D slice-wise
  ZFP              — libpressio ZFP, fixed-accuracy mode, 2-D slice-wise
  DEIM-2D+SZ2      — spatial SVD basis (k spatial modes), SZ2 on residual
  Kriging-2D+SZ2   — spatial Matérn-3/2 GP, SZ2 on residual

  MultiGP (LMC)    — joint U/V/W reconstruction via LMC, plotted SEPARATELY

Data
----
  /Argonne/100x500x500/TCf48.bin.f32  — 100 vertical levels × 500 × 500
  Binary float32, shape (100, 500, 500).  No land mask (all pixels active).
  Variable: TC (temperature [°C]).

Compression pipeline (DEIM / GP)
---------------------------------
  1. Train model on all n_T=100 levels.
  2. Fit model (SVD basis or GP hyperparams + RPCholesky sensors).
  3. For EACH of the n_T=100 vertical levels:
       predict from k sensor observations → compress residual with SZ2.
  4. Store: model + sensor values (float16 + zstd) + compressed residuals.
  5. CR = (n_T × ny × nx × 4) / total_compressed_bytes.

Notes
-----
  • No land mask: all 500×500 = 250 000 pixels are active.  CR denominators
    match between SZ2/ZFP and DEIM/GP (same n_T × ny × nx × 4 denominator).
  • DEIM k_max ≤ n_T = 100 (SVD rank limit).
  • GP sensor placement: RPCholesky (avoids forming 250k×250k kernel matrix).
  • MLE for GP hyperparams fit on a random subsample of 2000 pixels.
  • MultiGP uses U, V, W variables with LMC kernel (from multigp_hurricane.py
    functions) and is plotted separately.

Plots produced
--------------
  poster_rd_psnr_Hurricane.png    — poster-style RD (zoom inset, legend below)
  poster_field_panel_Hurricane.png— true | DEIM | GP | residual panel
  recon_panel_Hurricane.png       — GP prediction panel (level 50)
  multigp_hurricane_rd.png        — MultiGP reconstruction figure
  rd_results_Hurricane.csv        — full results table (all 4 methods)
"""

from __future__ import annotations
from pathlib import Path
import time, struct, csv
from collections import defaultdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import qr as scipy_qr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
ARGONNE   = Path(__file__).resolve().parent
DATA_DIR  = ARGONNE / "100x500x500"
TC_FILE   = DATA_DIR / "TCf48.bin.f32"

# MultiGP variable files (read-only reference data)
MGP_VARS  = {"U": "Uf48.bin.f32", "V": "Vf48.bin.f32", "W": "Wf48.bin.f32"}
MGP_UNITS = {"U": "m/s", "V": "m/s", "W": "m/s"}

# ── Compression parameters ────────────────────────────────────────────────────
N_LEVELS  = 100                              # total vertical levels
N_TRAIN   = 100                              # DEIM: use all levels for SVD
# DEIM k is constrained to k < N_TRAIN; hurricane has only 100 levels
DEIM_K_VALS = [10, 20, 40, 60, 80]          # spatial mode counts
KRIG_K_VALS = [50, 100, 200, 400]           # GP sensor counts

# abs_bound sweep: 1e-4 … 10.0 (TC spans ~60 °C; coarser bounds still useful)
ABS_BOUNDS      = np.logspace(-4, 0, 20)    # 20 log-spaced values
ABS_BOUNDS_SZ   = np.logspace(-4, 0, 20)    # same for SZ2/ZFP

JITTER    = 1e-6   # GP kernel regularisation
SKIP_LVLS = 0      # no near-zero surface levels to skip for TC

# ── Visualisation ─────────────────────────────────────────────────────────────
TIME_IDX  = 50                 # vertical level index for field panels
VIZ_K_DEIM = 60               # k shown in DEIM field panel
VIZ_K_KRIG = 200              # k shown in GP field panel

FIELD_TAG = "Hurricane"
DPI       = 150
CMAP_TC   = "RdYlBu_r"

# ── Colours and markers ────────────────────────────────────────────────────────
COLORS = {
    "SZ2":              "#1f77b4",   # blue
    "ZFP":              "#ff7f0e",   # orange
    "SZ3":              "#d62728",   # red
    "DEIM-2D+SZ2":      "#17becf",   # cyan
    "Kriging-2D+SZ2":   "#e377c2",   # pink
    "DEIM-2D+SZ3":      "#9467bd",   # purple
    "Kriging-2D+SZ3":   "#8c564b",   # brown
    "MultiGP":          "#2ca02c",   # green
}
MARKERS = {
    "SZ2":              "o",
    "ZFP":              "s",
    "SZ3":              "v",
    "DEIM-2D+SZ2":      "^",
    "Kriging-2D+SZ2":   "D",
    "DEIM-2D+SZ3":      "p",
    "Kriging-2D+SZ3":   "h",
    "MultiGP":          "P",
}
METHOD_ORDER = ["SZ2", "ZFP", "SZ3", "DEIM-2D+SZ2", "Kriging-2D+SZ2",
                "DEIM-2D+SZ3", "Kriging-2D+SZ3"]

# ── Flags ─────────────────────────────────────────────────────────────────────
RUN_LIBPRESSIO = True
RUN_ZFP        = True
RUN_SZ3        = True    # SZ3 via subprocess (no libpressio needed)
RUN_HYBRID     = True
PLOTS_ONLY     = False    # True → skip compression, reload CSV, re-plot only
USE_CHECKPOINT = True

# ── zstd / zlib compression backend ──────────────────────────────────────────
try:
    import zstandard as _zstd
    _cctx = _zstd.ZstdCompressor(level=3)
    def _compress(b: bytes) -> bytes: return _cctx.compress(b)
    COMPRESS_BACKEND = "zstd"
except ImportError:
    import zlib
    def _compress(b: bytes) -> bytes: return zlib.compress(b, 6)
    COMPRESS_BACKEND = "zlib"


# ── Quantisation utilities (mirror SZ2 pipeline) ──────────────────────────────
NUM_BINS = 65536

def _huffman_estimate(bins: np.ndarray) -> int:
    _, counts = np.unique(bins.ravel(), return_counts=True)
    probs   = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return int(np.ceil(bins.size * entropy / 8)) + 64

def pack_encode(bins, out_pos, out_vals) -> bytes:
    huff    = _huffman_estimate(bins)
    pos_enc = _compress(out_pos.tobytes()) if len(out_pos) else b''
    return b'\x00' * huff + pos_enc + out_vals.tobytes()

def compress_f16(a): return _compress(a.astype(np.float16).tobytes())
def compress_f32(a): return _compress(a.astype(np.float32).tobytes())
def compress_i32(a): return _compress(a.astype(np.int32).tobytes())


# ── Image quality metrics ─────────────────────────────────────────────────────
def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-15: return 999.9
    dr = float(orig.max() - orig.min())
    if dr < 1e-10:  return 999.9
    return float(20.0 * np.log10(dr / np.sqrt(mse)))

def global_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Global PSNR over all levels and all pixels.
    orig, recon: (n_T, n_full) or (n_T, ny, nx) float arrays.
    """
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = float(np.mean((o - r) ** 2))
    if mse < 1e-15: return 999.9
    dr = float(o.max() - o.min())
    if dr < 1e-10:  return 999.9
    return float(20.0 * np.log10(dr / np.sqrt(mse)))


# ── Pareto / monotone helpers ──────────────────────────────────────────────────
def _trim_sz_monotone(pts):
    """Remove anomalous backward-bending SZ2/ZFP RD points."""
    srt = sorted(pts, key=lambda p: p["abs_bound"], reverse=True)
    clean = [srt[0]]
    for p in srt[1:]:
        if p["cr"] <= clean[-1]["cr"]:
            clean.append(p)
        else:
            break
    return clean

def _valid(pts, key):
    return [p for p in pts
            if np.isfinite(p.get("cr", np.nan)) and np.isfinite(p.get(key, np.nan))]


# ── RPCholesky sensor selection ────────────────────────────────────────────────
def _rpcholesky_sensors(X: np.ndarray, ls: float, k: int, kern_fn, rng=None):
    """
    Select k spatial sensors via Randomly Pivoted Cholesky + column-pivoted QR.
    O(n × rank) memory — handles n = 250 000 without forming full n×n matrix.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n    = len(X)
    rank = min(k + 20, n)
    k    = min(k, n)
    diags = np.ones(n, dtype=np.float64)
    F     = np.zeros((n, rank), dtype=np.float64)
    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total <= 0:
            actual_rank = i; F = F[:, :i]; break
        si  = int(rng.choice(n, p=diags / total))
        col = kern_fn(X, X[[si]], ls).ravel()
        if i > 0:
            col = col - F[:, :i] @ F[si, :i]
        pv = float(col[si])
        if pv <= 0:
            actual_rank = i; F = F[:, :i]; break
        F[:, i] = col / np.sqrt(pv)
        diags = np.maximum(diags - F[:, i] ** 2, 0.0)
    if actual_rank < k:
        return np.array([int(np.argmax(diags))] * k, dtype=np.int32)
    G    = F.T @ F
    _, V = np.linalg.eigh(G)
    u_k  = F @ V[:, -k:]
    norms = np.linalg.norm(u_k, axis=0, keepdims=True)
    u_k  /= np.where(norms > 1e-12, norms, 1.0)
    _, _, p = scipy_qr(u_k.T, pivoting=True)
    return p[:k].astype(np.int32)


# ── Matérn-3/2 kernel ──────────────────────────────────────────────────────────
def matern32(X1: np.ndarray, X2: np.ndarray, ls: float) -> np.ndarray:
    diff = X1[:, None, :] - X2[None, :, :]
    r    = np.sqrt((diff ** 2).sum(-1))
    v    = np.sqrt(3.0) * r / ls
    return (1.0 + v) * np.exp(-v)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_tc() -> tuple[np.ndarray, np.ndarray]:
    """Load TC variable from binary float32 file.

    Returns
    -------
    tc  : (100, 500, 500) float32
    X   : (250000, 2) float64  normalised grid coordinates in [0,1]×[0,1]
    """
    if not TC_FILE.exists():
        raise FileNotFoundError(
            f"TC data not found: {TC_FILE}\n"
            f"Expected binary float32 with shape (100, 500, 500)."
        )
    tc = np.fromfile(str(TC_FILE), dtype=np.float32).reshape(100, 500, 500)
    n_T, ny, nx = tc.shape
    yi = np.linspace(0, 1, ny)
    xi = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(xi, yi, indexing="ij")   # (nx, ny) → ravel to (n,2)
    X = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float64)  # (ny*nx, 2)
    print(f"  TC: shape={tc.shape}  range=[{tc.min():.2f}, {tc.max():.2f}] °C")
    return tc, X

def load_hurricane_var(fname: str) -> np.ndarray:
    """Load a single ISABEL variable from binary float32."""
    path = DATA_DIR / fname
    return np.fromfile(str(path), dtype=np.float32).reshape(100, 500, 500)


# ── CSV I/O ────────────────────────────────────────────────────────────────────
CSV_PATH   = ARGONNE / f"rd_results_{FIELD_TAG}.csv"
CSV_FIELDS = ["method", "k", "abs_bound", "cr", "psnr", "ssim",
              "comp_sec", "decomp_sec", "train_sec", "compressed_MB",
              "model_MB", "sv_MB", "resid_MB", "n_outliers",
              "rmse_mean", "rmse_med", "rel_rmse_mean"]

def save_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {CSV_PATH}")

def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            for fld in ("cr", "psnr", "ssim", "comp_sec", "decomp_sec", "train_sec",
                        "compressed_MB", "model_MB", "sv_MB", "resid_MB", "abs_bound",
                        "rmse_mean", "rmse_med", "rel_rmse_mean"):
                try:   row[fld] = float(row[fld]) if row.get(fld, "") not in ("", "None") else float("nan")
                except: row[fld] = float("nan")
            try:   row["k"] = int(row["k"])
            except: pass
            rows.append(row)
    bm = defaultdict(list)
    for r in rows:
        bm[r["method"]].append(r)
    return bm


# ── libpressio helpers ─────────────────────────────────────────────────────────
def _run_slicewise(data: np.ndarray, compressor_id: str, config: dict, time_idx: int):
    """
    Compress each (ny, nx) level independently with libpressio.
    Matches the GP which also predicts each level independently from 2-D sensors.
    Returns (recon_snap, total_bytes, comp_sec, decomp_sec, psnr).
    """
    import libpressio
    n_T, ny, nx = data.shape
    slice_orig = ny * nx * 4
    global_dr  = float(data.max() - data.min())
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config":  {"pressio:metric": "composite", "composite:plugins": ["size"]},
        "compressor_config": config,
    })
    comp_total = 0.0; decomp_total = 0.0
    total_comp_b = 0; se_total = 0.0
    recon_snap = None
    for t in range(n_T):
        sl  = np.ascontiguousarray(data[t])          # (ny, nx)
        rsl = sl.copy()
        _t0 = time.perf_counter()
        compressed = comp.encode(sl)
        comp_total += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        rsl = comp.decode(compressed, rsl)
        decomp_total += time.perf_counter() - _t0
        cr_t = comp.get_metrics().get("size:compression_ratio", float("nan"))
        total_comp_b += (int(round(slice_orig / cr_t))
                         if np.isfinite(cr_t) and cr_t > 0 else slice_orig)
        if t == time_idx:
            recon_snap = rsl.copy()
        se_total += float(np.sum((data[t].astype(np.float64) -
                                   rsl.astype(np.float64)) ** 2))
    mse  = se_total / data.size
    psnr = (20.0 * np.log10(global_dr / np.sqrt(mse))
            if mse > 1e-15 and global_dr > 1e-10 else 999.9)
    return recon_snap, total_comp_b, comp_total, decomp_total, psnr


def _lp_on_residual(resid_full, recon_full, abs_bounds, model_b, sv_enc,
                    n_T, n_full, k, method_name, train_sec, ny=None, nx=None):
    """
    Compress the prediction residual with SZ2 or ZFP as a 3-D (n_T, ny, nx)
    volume — matching the 3-D mode used by the baselines.
    resid_full: (n_T, n_full) float32.  ny*nx must equal n_full.
    """
    try:
        import libpressio as _lp
    except ImportError:
        print(f"  libpressio not available — skipping {method_name}")
        return []

    cid = "sz" if "+SZ2" in method_name else "zfp"
    if cid == "zfp" and not RUN_ZFP:
        return []

    n_all     = n_T * n_full
    orig_full = recon_full.astype(np.float64) + resid_full.astype(np.float64)
    global_dr = float(orig_full.max() - orig_full.min())

    # Reshape each residual slice to (ny, nx) so SZ2/ZFP use 2-D Lorenzo,
    # matching the baseline which also compresses each level as a 2-D slice.
    use_2d = (ny is not None and nx is not None)
    slice_orig = n_full * 4

    rows = []
    for ab in abs_bounds:
        cfg = ({"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)}
               if cid == "sz" else {"zfp:accuracy": float(ab)})
        comp = _lp.PressioCompressor.from_config({
            "compressor_id": cid,
            "early_config":  {"pressio:metric": "composite", "composite:plugins": ["size"]},
            "compressor_config": cfg,
        })
        total_rb = 0; se_total = 0.0; t0 = time.perf_counter()
        for t in range(n_T):
            if use_2d:
                sl = np.ascontiguousarray(resid_full[t].reshape(ny, nx).astype(np.float32))
            else:
                sl = np.ascontiguousarray(resid_full[t].astype(np.float32))
            rsl = sl.copy()
            rsl = comp.decode(comp.encode(sl), rsl)
            cr_t = comp.get_metrics().get("size:compression_ratio", float("nan"))
            total_rb += (int(round(slice_orig / cr_t))
                         if np.isfinite(cr_t) and cr_t > 0 else slice_orig)
            fin_t = recon_full[t].astype(np.float64) + rsl.ravel().astype(np.float64)
            se_total += float(np.sum((orig_full[t] - fin_t) ** 2))
        cs  = time.perf_counter() - t0
        mse = se_total / n_all
        pv  = (20.0 * np.log10(global_dr / np.sqrt(mse))
               if np.isfinite(mse) and mse > 1e-15 and global_dr > 1e-10
               else float("nan"))
        tb  = model_b + len(sv_enc) + total_rb
        cr  = (n_all * 4) / tb
        rows.append({"method": method_name, "k": k, "abs_bound": ab,
                     "cr": cr, "psnr": pv, "ssim": float("nan"),
                     "comp_sec": cs, "train_sec": train_sec,
                     "compressed_MB": tb / 1e6, "model_MB": model_b / 1e6,
                     "sv_MB": len(sv_enc) / 1e6, "resid_MB": total_rb / 1e6,
                     "n_outliers": 0})
        print(f"       [{method_name}] k={k}  ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return rows


# ── SZ3 on residual (subprocess, 3-D volume) ──────────────────────────────────
def _sz3_on_residual(resid_full, recon_full, abs_bounds, model_b, sv_enc,
                     n_T, n_full, k, method_name, train_sec, ny=None, nx=None):
    """
    Compress the prediction residual with SZ3 via subprocess.
    Mirrors _lp_on_residual but uses SZ3 instead of libpressio SZ2/ZFP.
    resid_full: (n_T, n_full) float32.
    """
    import subprocess, tempfile, os, shutil
    sz3_bin = shutil.which("sz3")
    if sz3_bin is None:
        print(f"  [SZ3] sz3 not found — skipping {method_name}. Run: spack load sz3")
        return []

    n_all     = n_T * n_full
    orig_full = recon_full.astype(np.float64) + resid_full.astype(np.float64)
    global_dr = float(orig_full.max() - orig_full.min())

    if ny is not None and nx is not None:
        data_arr = resid_full.reshape(n_T, ny, nx).astype(np.float32)
        dim_args = ["-3", str(nx), str(ny), str(n_T)]
    else:
        data_arr = resid_full.astype(np.float32)
        dim_args = ["-2", str(n_full), str(n_T)]

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "resid.bin")
        data_arr.tofile(in_file)

        for ab in abs_bounds:
            cmp_file = os.path.join(tmpdir, "compressed.sz3")
            dec_file = os.path.join(tmpdir, "decompressed.bin")

            cmd_cmp = [sz3_bin, "-f", "-i", in_file, "-z", cmp_file] + dim_args + ["-M", "ABS", str(float(ab))]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd_cmp, capture_output=True, text=True)
            cs = time.perf_counter() - t0

            if proc.returncode != 0 or not os.path.exists(cmp_file):
                print(f"  [SZ3] compress FAILED {method_name} ab={ab:.2e}: {proc.stderr.strip()[:120]}")
                continue

            cmd_dec = [sz3_bin, "-f", "-z", cmp_file, "-o", dec_file] + dim_args + ["-M", "ABS", str(float(ab))]
            proc2 = subprocess.run(cmd_dec, capture_output=True, text=True)

            if proc2.returncode != 0 or not os.path.exists(dec_file):
                print(f"  [SZ3] decompress FAILED {method_name} ab={ab:.2e}: {proc2.stderr.strip()[:120]}")
                continue

            resid_dec = np.fromfile(dec_file, dtype=np.float32).reshape(n_T, n_full)
            total_rb  = os.path.getsize(cmp_file)

            fin = recon_full.astype(np.float64) + resid_dec.astype(np.float64)
            mse = float(np.mean((orig_full - fin) ** 2))
            pv  = (20.0 * np.log10(global_dr / np.sqrt(mse))
                   if np.isfinite(mse) and mse > 1e-15 and global_dr > 1e-10
                   else float("nan"))
            tb  = model_b + len(sv_enc) + total_rb
            cr  = (n_all * 4) / tb
            rows.append({"method": method_name, "k": k, "abs_bound": ab,
                         "cr": cr, "psnr": pv, "ssim": float("nan"),
                         "comp_sec": cs, "train_sec": train_sec,
                         "compressed_MB": tb / 1e6, "model_MB": model_b / 1e6,
                         "sv_MB": len(sv_enc) / 1e6, "resid_MB": total_rb / 1e6,
                         "n_outliers": 0})
            print(f"       [{method_name}] k={k}  ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return rows


# ── SZ2 ───────────────────────────────────────────────────────────────────────
def run_sz2(data: np.ndarray, abs_bounds=ABS_BOUNDS_SZ):
    """SZ2 slice-wise on (100, 500, 500) — matches GP which predicts each level independently."""
    n_all = data.size
    results = []
    for ab in abs_bounds:
        config = {"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)}
        recon_snap, total_b, comp_sec, decomp_sec, pv = _run_slicewise(
            data, "sz", config, TIME_IDX)
        cr = (n_all * 4) / total_b if total_b > 0 else float("nan")
        results.append({"method": "SZ2", "k": 0, "abs_bound": ab,
            "cr": cr, "psnr": pv, "ssim": float("nan"),
            "comp_sec": comp_sec, "decomp_sec": decomp_sec, "train_sec": 0.0,
            "compressed_MB": total_b / 1e6,
            "model_MB": 0.0, "sv_MB": 0.0, "resid_MB": 0.0, "n_outliers": 0})
        print(f"  SZ2  abs={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results


def run_zfp(data: np.ndarray, zfp_bounds=ABS_BOUNDS_SZ):
    """ZFP fixed-accuracy slice-wise — matches GP which predicts each level independently."""
    n_all = data.size
    results = []
    for ab in zfp_bounds:
        config = {"zfp:accuracy": float(ab)}
        recon_snap, total_b, comp_sec, decomp_sec, pv = _run_slicewise(
            data, "zfp", config, TIME_IDX)
        cr = (n_all * 4) / total_b if total_b > 0 else float("nan")
        results.append({"method": "ZFP", "k": 0, "abs_bound": ab,
            "cr": cr, "psnr": pv, "ssim": float("nan"),
            "comp_sec": comp_sec, "decomp_sec": decomp_sec, "train_sec": 0.0,
            "compressed_MB": total_b / 1e6,
            "model_MB": 0.0, "sv_MB": 0.0, "resid_MB": 0.0, "n_outliers": 0})
        print(f"  ZFP  acc={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results


# ── SZ3 (subprocess, 3-D mode) ────────────────────────────────────────────────
def run_sz3(data: np.ndarray, abs_bounds=ABS_BOUNDS_SZ):
    """
    SZ3 on full (n_T, ny, nx) volume via command-line subprocess.
    Requires `sz3` on PATH (run `spack load sz3` before this script).
    Uses absolute error mode (-M ABS) on the full 3-D volume.

    SZ3 dimension flag: -3 <nx> <ny> <nz> for data[nz][ny][nx]
    Our data shape: (n_T=100, ny=500, nx=500) → -3 500 500 100
    Compress and decompress are separate calls.
    """
    import subprocess, tempfile, os, shutil

    sz3_bin = shutil.which("sz3")
    if sz3_bin is None:
        print("  [SZ3] sz3 not found on PATH — skipping. Run: spack load sz3")
        return []

    n_T, ny, nx = data.shape
    n_all = data.size
    # SZ3 -3 nx ny nz (C order: data[nz][ny][nx])
    dim_args = ["-3", str(nx), str(ny), str(n_T)]
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        in_file  = os.path.join(tmpdir, "input.bin")
        data.astype(np.float32).tofile(in_file)

        for ab in abs_bounds:
            cmp_file = os.path.join(tmpdir, "compressed.sz3")
            dec_file = os.path.join(tmpdir, "decompressed.bin")

            # Step 1: compress
            cmd_cmp = [sz3_bin, "-f",
                       "-i", in_file, "-z", cmp_file,
                       ] + dim_args + ["-M", "ABS", str(float(ab))]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd_cmp, capture_output=True, text=True)
            comp_sec = time.perf_counter() - t0

            if proc.returncode != 0 or not os.path.exists(cmp_file):
                print(f"  [SZ3] compress FAILED ab={ab:.2e}: {proc.stderr.strip()[:120]}")
                continue

            # Step 2: decompress
            cmd_dec = [sz3_bin, "-f",
                       "-z", cmp_file, "-o", dec_file,
                       ] + dim_args + ["-M", "ABS", str(float(ab))]
            proc2 = subprocess.run(cmd_dec, capture_output=True, text=True)

            if proc2.returncode != 0 or not os.path.exists(dec_file):
                print(f"  [SZ3] decompress FAILED ab={ab:.2e}: {proc2.stderr.strip()[:120]}")
                continue

            recon   = np.fromfile(dec_file, dtype=np.float32).reshape(n_T, ny, nx)
            total_b = os.path.getsize(cmp_file)
            cr  = (n_all * 4) / total_b if total_b > 0 else float("nan")
            pv  = global_psnr(data.astype(np.float64), recon.astype(np.float64))

            results.append({
                "method": "SZ3", "k": 0, "abs_bound": ab,
                "cr": cr, "psnr": pv, "ssim": float("nan"),
                "comp_sec": comp_sec, "decomp_sec": 0.0, "train_sec": 0.0,
                "compressed_MB": total_b / 1e6,
                "model_MB": 0.0, "sv_MB": 0.0, "resid_MB": 0.0, "n_outliers": 0,
            })
            print(f"  SZ3  abs={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")

    return results


# ── DEIM-2D+SZ2 ───────────────────────────────────────────────────────────────
def run_deim_2d(data: np.ndarray, X: np.ndarray, abs_bounds=ABS_BOUNDS):
    """
    Build SVD basis from all N_TRAIN=100 levels on the full 500×500 grid.
    k_max ≤ N_TRAIN = 100 (SVD rank limit with n_T=100 levels).

    CR = (n_T × ny × nx × 4) / (model_b + sv_enc + residual_b)
    No land mask: all ny×nx pixels are compressed.
    """
    n_T, ny, nx = data.shape
    n_full = ny * nx
    n_all  = n_T * n_full

    print(f"\n[DEIM-2D]  n_T={n_T}  ny={ny}  nx={nx}  n_full={n_full}  n_train={N_TRAIN}")

    _deim_ckpt = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
    k_max      = min(max(DEIM_K_VALS), N_TRAIN, n_full)

    if USE_CHECKPOINT and _deim_ckpt.exists():
        print(f"  [checkpoint] Loading SVD basis from {_deim_ckpt.name} ...", flush=True)
        _ck       = np.load(_deim_ckpt)
        Phi_max   = _ck["Phi_max"].astype(np.float64)   # (n_full, k_max)
        mean_full = _ck["mean_full"]
        s_vals    = _ck["s_vals"]
        train_sec = float(_ck["train_sec"]) if "train_sec" in _ck.files else 0.0
        k_max     = min(k_max, Phi_max.shape[1])
        print(f"  [checkpoint] Phi_max {Phi_max.shape}  k_max={k_max}  "
              f"train_sec={train_sec:.1f}s")
    else:
        # Use all n_T levels for training (n_T=100 is small enough)
        train_data = data.reshape(N_TRAIN, n_full).astype(np.float64)
        mean_full  = train_data.mean(axis=0)             # (n_full,)
        F          = train_data - mean_full              # centred
        print(f"  SVD ({N_TRAIN} × {n_full}), k_max={k_max} ...", flush=True)
        t0 = time.perf_counter()
        # Thin SVD: F = U S Vt  shape F=(N_TRAIN, n_full)
        # Vt shape (N_TRAIN, n_full); spatial modes Phi = Vt[:k_max].T (n_full, k_max)
        _, s_vals, Vt = np.linalg.svd(F, full_matrices=False)
        Phi_max    = Vt[:k_max, :].T                     # (n_full, k_max)
        train_sec  = time.perf_counter() - t0
        if USE_CHECKPOINT:
            np.savez_compressed(str(_deim_ckpt),
                                Phi_max=Phi_max.astype(np.float32),
                                mean_full=mean_full.astype(np.float32),
                                s_vals=s_vals,
                                train_sec=np.float64(train_sec))
            print(f"  [checkpoint] SVD basis saved → {_deim_ckpt.name}")

    total_var = float((s_vals ** 2).sum())
    print(f"  SVD done: {train_sec:.1f}s  |  variance explained:")
    for kv in DEIM_K_VALS:
        kc = min(kv, len(s_vals))
        ve = float((s_vals[:kc] ** 2).sum() / total_var)
        print(f"    k={kv:3d} → {100*ve:.2f}%")

    mean_c    = compress_f32(mean_full)
    data_flat = data.reshape(n_T, n_full).astype(np.float32)
    data_cent = data_flat - mean_full.astype(np.float32)   # (n_T, n_full)

    results = []
    pred_snap_deim = None
    _seen_k = set()
    for k in DEIM_K_VALS:
        k = min(k, k_max)
        if k in _seen_k:
            print(f"  [DEIM] Skipping duplicate k={k}")
            continue
        _seen_k.add(k)
        Phi_k = Phi_max[:, :k]                           # (n_full, k)

        # Q-DEIM: column-pivoted QR → k sensor locations
        _, _, p  = scipy_qr(Phi_k.T, pivoting=True)
        sensors  = p[:k]                                  # (k,) indices into [0, n_full)

        phi_c   = compress_f16(Phi_k)
        sens_c  = compress_i32(sensors)
        model_b = len(phi_c) + len(sens_c) + len(mean_c)

        A      = Phi_k[sensors, :]                        # (k, k)
        all_sv = data_cent[:, sensors]                    # (n_T, k)
        sv_enc = _compress(all_sv.astype(np.float16).tobytes())

        print(f"  k={k:3d}  computing DEIM operator & reconstructing {n_T} levels ...",
              flush=True)
        M            = (Phi_k @ np.linalg.inv(A.astype(np.float64))).astype(np.float32)
        recon_full_t = all_sv @ M.T + mean_full.astype(np.float32)   # (n_T, n_full)

        if k == VIZ_K_DEIM:
            pred_snap_deim = recon_full_t[TIME_IDX].reshape(ny, nx).copy()

        resid_full_t = (data_flat - recon_full_t).astype(np.float32)  # (n_T, n_full)

        if RUN_HYBRID and RUN_LIBPRESSIO:
            results += _lp_on_residual(
                resid_full_t, recon_full_t, abs_bounds, model_b, sv_enc,
                n_T, n_full, k, "DEIM-2D+SZ2", train_sec, ny=ny, nx=nx)
        if RUN_HYBRID and RUN_SZ3:
            results += _sz3_on_residual(
                resid_full_t, recon_full_t, abs_bounds, model_b, sv_enc,
                n_T, n_full, k, "DEIM-2D+SZ3", train_sec, ny=ny, nx=nx)

    deim_panel = {"pred": pred_snap_deim, "k": VIZ_K_DEIM}
    return results, deim_panel


# ── Kriging-2D+SZ2 ────────────────────────────────────────────────────────────
def run_kriging_2d(data: np.ndarray, X: np.ndarray, abs_bounds=ABS_BOUNDS):
    """
    Fit Matérn-3/2 GP hyperparams (MLE on 2000 subsampled pixels) then place
    sensors via RPCholesky on the full 250k-pixel grid.

    CR = (n_T × ny × nx × 4) / (model_b + sv_enc + residual_b)
    No land mask: all ny×nx pixels active.
    """
    n_T, ny, nx = data.shape
    n_full = ny * nx
    n_all  = n_T * n_full

    print(f"\n[Kriging-2D]  n_T={n_T}  ny={ny}  nx={nx}  n_full={n_full}")
    t0 = time.perf_counter()

    k_max      = min(max(KRIG_K_VALS), n_full)
    _krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N{k_max}.npz"

    data_flat = data.reshape(n_T, n_full).astype(np.float64)  # (n_T, n_full)

    if USE_CHECKPOINT and _krig_ckpt.exists():
        print(f"  [checkpoint] Loading GP hyperparams + sensors from {_krig_ckpt.name} ...",
              flush=True)
        _ck           = np.load(_krig_ckpt)
        ls            = float(_ck["ls"])
        _var          = float(_ck["var"])
        noise_var     = float(_ck["noise_var"])
        sensors       = _ck["sensors"]          # (k_max,)
        train_mean    = _ck["train_mean"]        # (n_full,)
        train_std_safe= _ck["train_std_safe"]    # (n_full,)
        zero_std      = (train_std_safe == 1.0)
        k_max         = min(k_max, len(sensors))
        train_sec     = float(_ck["train_sec"]) if "train_sec" in _ck.files else 0.0
        print(f"  [checkpoint] ls={ls:.4f}  var={_var:.4f}  noise={noise_var:.2e}  "
              f"k_max={k_max}  train_sec={train_sec:.1f}s")
    else:
        # ── Compute per-pixel mean and std over all levels ──────────────────
        train_mean      = data_flat.mean(axis=0)          # (n_full,)
        train_std       = data_flat.std(axis=0)
        train_std_safe  = np.where(train_std < 1e-10, 1.0, train_std)
        zero_std        = train_std < 1e-10

        # ── MLE on 2000-point subsample ─────────────────────────────────────
        fit_size = min(2000, n_full)
        rng_fit  = np.random.default_rng(0)
        fit_idx  = rng_fit.choice(n_full, size=fit_size, replace=False)
        X_fit    = X[fit_idx]
        # Use first level (after centering) for MLE fitting
        Y_fit    = ((data_flat[0] - train_mean) / train_std_safe)[fit_idx]

        print(f"  Fitting ls + var + noise on {fit_size} points, 3 restarts ...", flush=True)
        from scipy.optimize import minimize as _minimize

        def _neg_lml(log_theta):
            ls_    = float(np.exp(log_theta[0]))
            var_   = float(np.exp(log_theta[1]))
            noise_ = float(np.exp(log_theta[2]))
            K = var_ * matern32(X_fit, X_fit, ls_) + noise_ * np.eye(fit_size)
            try:
                L_ = np.linalg.cholesky(K + 1e-8 * np.eye(fit_size))
            except np.linalg.LinAlgError:
                return 1e10
            log_det = 2.0 * np.sum(np.log(np.diag(L_)))
            alpha   = np.linalg.solve(L_.T, np.linalg.solve(L_, Y_fit))
            return 0.5 * float(Y_fit @ alpha) + 0.5 * log_det

        dists  = np.sqrt(((X_fit[:20, None, :] - X_fit[None, :20, :]) ** 2).sum(-1))
        ls0    = float(np.median(dists[dists > 0])) / 3.0
        var0   = max(float(np.var(Y_fit)), 1e-6)
        noise0 = var0 * 0.01
        bnds   = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
        rng_hp = np.random.default_rng(0)
        starts = [np.log([max(ls0, 0.01), max(var0, 0.01), max(noise0, 1e-6)])] + [
            rng_hp.uniform([b[0] for b in bnds], [b[1] for b in bnds]) for _ in range(2)]

        best_nll, best_ls, best_var, best_noise = np.inf, ls0, var0, noise0
        for x0 in starts:
            try:
                res = _minimize(_neg_lml, x0, method="L-BFGS-B", bounds=bnds,
                                options={"maxiter": 200, "ftol": 1e-9})
                if res.fun < best_nll:
                    best_nll   = res.fun
                    best_ls    = float(np.exp(res.x[0]))
                    best_var   = float(np.exp(res.x[1]))
                    best_noise = float(np.exp(res.x[2]))
            except Exception:
                pass
        ls, noise_var, _var = best_ls, best_noise, best_var
        print(f"  ls={ls:.4f}  var={_var:.4f}  noise={noise_var:.2e}")

        # ── Sensor placement via RPCholesky on full 250k grid ────────────────
        print(f"  RPCholesky: selecting {k_max} sensors from {n_full} pixels ...", flush=True)
        sensors = _rpcholesky_sensors(X, ls, k_max, matern32)

        train_sec = time.perf_counter() - t0
        if USE_CHECKPOINT:
            np.savez_compressed(str(_krig_ckpt),
                                ls=np.float64(ls), var=np.float64(_var),
                                noise_var=np.float64(noise_var),
                                sensors=sensors,
                                train_mean=train_mean.astype(np.float32),
                                train_std_safe=train_std_safe.astype(np.float32),
                                train_sec=np.float64(train_sec))
            print(f"  [checkpoint] Saved → {_krig_ckpt.name}")

    # Build K matrices for all k_max sensors (once; sliced for each k)
    X_sens_max = X[sensors]   # (k_max, 2)
    print(f"  Building K_Xs ({n_full} × {k_max}) and K_ss ({k_max} × {k_max}) ...",
          flush=True)
    K_Xs_max = matern32(X, X_sens_max, ls)                           # (n_full, k_max)
    K_ss_max = matern32(X_sens_max, X_sens_max, ls) + JITTER * np.eye(k_max)

    mean_c  = compress_f32(train_mean.astype(np.float32))
    std_c   = compress_f32(train_std_safe.astype(np.float32))
    print(f"  Training done: {train_sec:.1f}s")

    all_sv_max = data_flat[:, sensors].astype(np.float32)            # (n_T, k_max)

    results = []
    pred_snap_krig = None
    sensors_snap   = None
    _seen_k = set()
    for k in KRIG_K_VALS:
        k = min(k, k_max)
        if k in _seen_k:
            print(f"  [Kriging] Skipping duplicate k={k}")
            continue
        _seen_k.add(k)

        s_k   = sensors[:k]           # (k,)
        K_Xs  = K_Xs_max[:, :k]      # (n_full, k)
        K_ss_k = K_ss_max[:k, :k]

        K_sub = _var * K_ss_k + noise_var * np.eye(k)
        try:    L_k, lower = cho_factor(K_sub, lower=True)
        except: L_k, lower = cho_factor(K_sub + 1e-4 * np.eye(k), lower=True)

        hyperparams_b = struct.pack("<IIddd", ny, nx, float(ls), float(_var), float(noise_var))
        sens_c  = compress_i32(s_k.astype(np.int32))
        model_b = len(hyperparams_b) + len(sens_c) + len(mean_c) + len(std_c)

        all_sv = all_sv_max[:, :k]             # (n_T, k)
        sv_enc = _compress(all_sv.astype(np.float16).tobytes())

        print(f"  k={k:3d}  reconstructing {n_T} levels ...", flush=True)
        ms_k   = train_mean[s_k]
        ss_k   = train_std_safe[s_k]
        y_norm = ((all_sv.astype(np.float64) - ms_k) / ss_k).T   # (k, n_T)
        alpha  = _var * cho_solve((L_k, lower), y_norm)           # (k, n_T)

        # Chunked prediction (n_full is large: 250k)
        CHUNK = 50
        recon_flat_t = np.zeros((n_T, n_full), dtype=np.float32)
        for i in range(0, n_T, CHUNK):
            mu = (K_Xs @ alpha[:, i:i+CHUNK]).T                    # (chunk, n_full)
            mu = mu * train_std_safe + train_mean
            mu[:, zero_std] = train_mean[zero_std]
            recon_flat_t[i:i+CHUNK] = mu.astype(np.float32)

        if k == VIZ_K_KRIG:
            pred_snap_krig = recon_flat_t[TIME_IDX].reshape(ny, nx).copy()
            sensors_snap   = s_k.copy()

        resid_full_t = (data_flat.astype(np.float32) - recon_flat_t)   # (n_T, n_full)

        if RUN_HYBRID and RUN_LIBPRESSIO:
            results += _lp_on_residual(
                resid_full_t, recon_flat_t, abs_bounds, model_b, sv_enc,
                n_T, n_full, k, "Kriging-2D+SZ2", train_sec, ny=ny, nx=nx)
        if RUN_HYBRID and RUN_SZ3:
            results += _sz3_on_residual(
                resid_full_t, recon_flat_t, abs_bounds, model_b, sv_enc,
                n_T, n_full, k, "Kriging-2D+SZ3", train_sec, ny=ny, nx=nx)

    krig_panel = {"pred": pred_snap_krig, "sensors": sensors_snap, "k": VIZ_K_KRIG}
    return results, krig_panel


# ─────────────────────────────────────────────────────────────────────────────
# MultiGP (LMC on U, V, W)  — from multigp_hurricane.py (read-only reference)
# ─────────────────────────────────────────────────────────────────────────────
# We copy the minimal LMC functions needed here rather than importing multigp_hurricane
# (which doesn't do compression, just reconstruction).

def _estimate_B(Y_train_list: list) -> np.ndarray:
    Y_all = np.vstack(Y_train_list)
    B = 0.5 * ((Y_all.T @ Y_all) / len(Y_all))
    B = 0.5 * (B + B.T) + JITTER * np.eye(B.shape[0])
    return B

def _gks_sensors(K_spatial: np.ndarray, k: int) -> np.ndarray:
    from scipy.linalg import qr
    _, _, p = qr(K_spatial, pivoting=True)
    return p[:k]

def _lmc_predict(X_all, X_sensors, Y_sensors_norm, B, ls, noise_var):
    """LMC posterior mean. See multigp_hurricane.py for full derivation."""
    n, d = len(X_all), B.shape[0]
    k    = len(X_sensors)
    K_ss = matern32(X_sensors, X_sensors, ls) + JITTER * np.eye(k)
    K_Xs = matern32(X_all, X_sensors, ls)
    K_sub = np.kron(B, K_ss) + noise_var * np.eye(k * d)
    y_flat = Y_sensors_norm.T.ravel()
    try:
        L, low = cho_factor(K_sub, lower=True)
        alpha  = cho_solve((L, low), y_flat)
    except Exception:
        alpha  = np.linalg.solve(K_sub, y_flat)
    alpha_mat = alpha.reshape(d, k)
    mu_norm   = K_Xs @ (B @ alpha_mat).T    # (n, d)
    return mu_norm

def _fit_ls_lmc(X, Y_train_list, B, noise_var, n_restarts=3, rng=None):
    """Fit lengthscale for LMC via marginal likelihood."""
    from scipy.optimize import minimize as _minimize
    if rng is None:
        rng = np.random.default_rng(42)
    d    = B.shape[0]
    λ_B, Q_B = np.linalg.eigh(B)

    def _nll(log_ls):
        ls_  = float(np.exp(log_ls[0]))
        n    = len(X)
        K_n  = matern32(X, X, ls_) + JITTER * np.eye(n)
        λ_n, Q_n = np.linalg.eigh(K_n)
        Λ    = np.outer(λ_n, λ_B) + noise_var
        log_det = np.sum(np.log(np.maximum(Λ, 1e-300)))
        quad = sum(np.sum((Q_n.T @ Y @ Q_B) ** 2 / Λ) for Y in Y_train_list)
        n_obs = len(Y_train_list)
        return 0.5 * (n_obs * (n * d * np.log(2 * np.pi) + log_det) + quad)

    dists = np.sqrt(((X[::5, None] - X[None, ::5]) ** 2).sum(-1)).ravel()
    ls0   = float(np.median(dists[dists > 0]))
    starts = [np.log(ls0)] + list(np.log(ls0) + rng.uniform(-1, 1, n_restarts - 1))
    best_nll, best_ls = np.inf, ls0
    for x0 in starts:
        try:
            res = _minimize(_nll, [x0], method="L-BFGS-B",
                            bounds=[(-1, np.log(max(X.max(), 10) * 3))],
                            options={"maxiter": 50, "ftol": 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls  = float(np.exp(res.x[0]))
        except Exception:
            pass
    return best_ls


def run_multigp(data_uvw: dict,
                n_sensors: int = 200, downsample: int = 5,
                test_level: int = 50) -> dict:
    """
    Run LMC MultiGP on U, V, W variables (downsampled for tractability).
    Returns a results dict for plotting — does NOT compute compression ratios.
    Saves checkpoint to multigp_ckpt_Hurricane.npz.

    Parameters
    ----------
    data_uvw   : {"U": (100,500,500), "V": ..., "W": ...}
    X_sub      : (n_sub, 2)  grid coordinates for downsampled spatial grid
    n_sensors  : number of GKS sensors
    downsample : spatial downsampling factor (500→167 per axis for ds=3)
    test_level : level index to reconstruct for the figure
    """
    var_names = list(data_uvw.keys())
    d = len(var_names)

    # Downsampled data: (100, ny_s, nx_s)
    data_ds = {v: data_uvw[v][:, ::downsample, ::downsample] for v in var_names}
    n_levels, ny_s, nx_s = data_ds[var_names[0]].shape
    n_sub = ny_s * nx_s
    # Grid for downsampled spatial points
    yi_s = np.linspace(0, 1, ny_s)
    xi_s = np.linspace(0, 1, nx_s)
    XX_s, YY_s = np.meshgrid(xi_s, yi_s, indexing="ij")
    X_all = np.column_stack([XX_s.ravel(), YY_s.ravel()]).astype(np.float64)

    SKIP_LEVELS = 0
    data_tensor = np.stack(
        [data_ds[v].reshape(n_levels, n_sub) for v in var_names], axis=-1
    )   # (100, n_sub, d)

    # Normalise per-location per-variable
    train_mean = data_tensor.mean(axis=0)         # (n_sub, d)
    train_std  = data_tensor.std(axis=0)
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)
    Y_train_list = [(data_tensor[l] - train_mean) / train_std
                    for l in range(n_levels)]

    _mgp_ckpt = ARGONNE / "multigp_ckpt_Hurricane.npz"

    if USE_CHECKPOINT and _mgp_ckpt.exists():
        print(f"  [MultiGP checkpoint] Loading from {_mgp_ckpt.name} ...")
        _ck      = np.load(_mgp_ckpt)
        B        = _ck["B"]
        ls       = float(_ck["ls"])
        sensors  = _ck["sensors"]
        noise_var= float(_ck["noise_var"])
        print(f"  [MultiGP] ls={ls:.4f}  sensors={len(sensors)}  noise={noise_var:.2e}")
    else:
        print(f"\n[MultiGP]  Estimating B matrix ...")
        B = _estimate_B(Y_train_list)
        noise_var = (0.05) ** 2

        print(f"  Fitting lengthscale (n_sub={n_sub}) ...")
        # Subsample for efficiency in LMC MLE (O(n³))
        sub_size = min(500, n_sub)
        rng_sub = np.random.default_rng(0)
        sub_idx = rng_sub.choice(n_sub, size=sub_size, replace=False)
        X_sub_fit = X_all[sub_idx]
        Y_sub_list = [Y[sub_idx] for Y in Y_train_list]
        ls = _fit_ls_lmc(X_sub_fit, Y_sub_list, B, noise_var, n_restarts=3)
        print(f"  Optimal ls = {ls:.4f}")

        print(f"  GKS sensor placement: {n_sensors} sensors from {n_sub} pts ...")
        K_spatial = matern32(X_all, X_all, ls) + JITTER * np.eye(n_sub)
        sensors   = _gks_sensors(K_spatial, n_sensors)

        if USE_CHECKPOINT:
            np.savez_compressed(str(_mgp_ckpt),
                                B=B, ls=np.float64(ls),
                                sensors=sensors,
                                noise_var=np.float64(noise_var),
                                train_mean=train_mean.astype(np.float32),
                                train_std=train_std.astype(np.float32))
            print(f"  [MultiGP checkpoint] Saved → {_mgp_ckpt.name}")

    X_sensors = X_all[sensors]   # (n_sensors, 2)
    lvl = min(test_level, n_levels - 1)
    true_nd  = data_tensor[lvl]                           # (n_sub, d)
    true_norm = (true_nd - train_mean) / train_std
    Y_obs_norm = true_norm[sensors]                       # (n_sensors, d)
    mu_norm = _lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var)
    mu_full = mu_norm * train_std + train_mean            # (n_sub, d)

    # Per-variable RMSE
    print(f"\n  MultiGP reconstruction — level {lvl}")
    for vi, vn in enumerate(var_names):
        diff = mu_full[:, vi] - true_nd[:, vi]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rng  = float(true_nd[:, vi].max() - true_nd[:, vi].min())
        print(f"    {vn}: RMSE={rmse:.4f}  range={rng:.3f} {MGP_UNITS.get(vn,'')}")

    return {
        "var_names":  var_names,
        "true_nd":    true_nd,
        "mu_full":    mu_full,
        "train_mean": train_mean,
        "train_std":  train_std,
        "sensors":    sensors,
        "X_all":      X_all,
        "B":          B,
        "ls":         ls,
        "ny_s":       ny_s,
        "nx_s":       nx_s,
        "lvl":        lvl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_by_k(ax, pts, metric_key, color, marker, x_fn=None, label_prefix="",
               linestyle="--", lw=1.8, ms=5):
    from collections import defaultdict
    if x_fn is None:
        x_fn = lambda p: p["cr"]
    k_groups = defaultdict(list)
    for p in pts:
        k_groups[p["k"]].append(p)
    k_vals = sorted(k_groups.keys())
    n_k = len(k_vals)
    proxy = None
    for ki, kv in enumerate(k_vals):
        kpts = sorted(_valid(k_groups[kv], metric_key), key=x_fn)
        if not kpts:
            continue
        alpha = 0.30 + 0.70 * ki / max(1, n_k - 1)
        xs = [x_fn(p) for p in kpts]
        ys = [p[metric_key] for p in kpts]
        line, = ax.plot(xs, ys, linestyle, color=color, marker=marker,
                        lw=lw, ms=ms, alpha=alpha)
        if ki == n_k // 2:
            proxy = line
        ax.annotate(f"k={kv}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=6, color=color, alpha=min(1.0, alpha + 0.15))
    if proxy is None and k_vals:
        proxy = line
    if proxy is not None and label_prefix:
        proxy.set_label(f"{label_prefix} (k={k_vals[0]}–{k_vals[-1]})")
    return proxy


def _shade_our_wins(ax, by_method, metric_key,
                    our_methods=("DEIM-2D+SZ2", "Kriging-2D+SZ2"),
                    baseline="SZ2", color="#d0d0d0", alpha=0.45, zorder=0):
    base_pts = _valid(by_method.get(baseline, []), metric_key)
    if not base_pts:
        return
    base_pts_trim = sorted(_trim_sz_monotone(base_pts), key=lambda p: p["cr"])
    if len(base_pts_trim) < 2:
        return
    log_base_cr = np.log10(np.array([p["cr"]      for p in base_pts_trim]))
    base_val    =          np.array([p[metric_key] for p in base_pts_trim])
    per_k_curves = []
    for m in our_methods:
        pts = _valid(by_method.get(m, []), metric_key)
        if not pts:
            continue
        k_groups: dict = {}
        for p in pts:
            k_groups.setdefault(p["k"], []).append(p)
        for kv, kpts in k_groups.items():
            kpts_s = sorted(kpts, key=lambda p: p["cr"])
            if len(kpts_s) < 2:
                continue
            per_k_curves.append((
                np.log10(np.array([p["cr"]      for p in kpts_s])),
                np.array([p[metric_key] for p in kpts_s])
            ))
    if not per_k_curves:
        return
    our_log_min = min(lc[0]  for lc, _ in per_k_curves)
    our_log_max = max(lc[-1] for lc, _ in per_k_curves)
    cr_log_min  = max(log_base_cr[0],  our_log_min)
    cr_log_max  = min(log_base_cr[-1], our_log_max)
    if cr_log_min >= cr_log_max:
        return
    log_grid = np.linspace(cr_log_min, cr_log_max, 800)
    cr_grid  = 10 ** log_grid
    our_env  = np.full(len(log_grid), -np.inf)
    for log_crs, psnrs in per_k_curves:
        mask = (log_grid >= log_crs[0]) & (log_grid <= log_crs[-1])
        if not mask.any():
            continue
        our_env[mask] = np.maximum(our_env[mask],
                                   np.interp(log_grid[mask], log_crs, psnrs))
    base_interp = np.interp(log_grid, log_base_cr, base_val)
    wins = np.isfinite(our_env) & (our_env > base_interp)
    if not wins.any():
        return
    diff = np.diff(wins.astype(int), prepend=0, append=0)
    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        ax.axvspan(cr_grid[s], cr_grid[min(e, len(cr_grid) - 1)],
                   color=color, alpha=alpha, zorder=zorder, label="_nolegend_")


def plot_rd_poster(by_method, metric_key, ylabel, fname_suffix):
    """Poster-style RD plot with zoom inset and below-axis legend."""
    if not any(_valid(by_method.get(m, []), metric_key) for m in METHOD_ORDER):
        print(f"  Skipping poster RD ({fname_suffix}) — no valid {metric_key} data")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    _shade_our_wins(ax, by_method, metric_key)

    def _draw_method(target_ax, method, pts, lw=2.0, ms=8):
        c, mk = COLORS.get(method, "#888"), MARKERS.get(method, "o")
        if method in ("SZ2", "ZFP", "SZ3"):
            srt = sorted(pts, key=lambda p: p["cr"])
            target_ax.plot([p["cr"] for p in srt], [p[metric_key] for p in srt],
                           "-", color=c, marker=mk, lw=lw, ms=ms)
        else:
            _plot_by_k(target_ax, pts, metric_key, c, mk, lw=lw, ms=ms // 2)

    all_pts = {}
    for method in METHOD_ORDER:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            continue
        all_pts[method] = pts
        c, mk = COLORS.get(method, "#888"), MARKERS.get(method, "o")
        if method in ("SZ2", "ZFP", "SZ3"):
            srt     = sorted(_trim_sz_monotone(pts), key=lambda p: p["cr"])
            ab_vals = sorted(set(p["abs_bound"] for p in srt))
            lbl     = f"{method} (ε={ab_vals[0]:.0e}–{ab_vals[-1]:.0e})"
            ax.plot([p["cr"] for p in srt], [p[metric_key] for p in srt],
                    "-", color=c, marker=mk, lw=2.2, ms=10, label=lbl)
        else:
            _plot_by_k(ax, pts, metric_key, c, mk, label_prefix=method, lw=2.2, ms=6)

    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_title(
        f"Rate–Distortion  —  ISABEL Hurricane TC  (global PSNR over all levels)",
        fontsize=13)
    ax.legend(fontsize=10, loc="lower left",
              ncol=1, handlelength=1.8, columnspacing=1.0,
              framealpha=0.9, borderpad=0.6)

    # Zoom inset (upper right)
    if metric_key == "psnr" and len(all_pts) >= 2:
        try:
            zoom_pts = []
            for m in ["DEIM-2D+SZ2", "Kriging-2D+SZ2"]:
                zoom_pts += _valid(by_method.get(m, []), metric_key)
            if zoom_pts:
                crs_lo = [p["cr"] for p in zoom_pts]
                val_lo = [p[metric_key] for p in zoom_pts]
                xlim = (max(0.5, min(crs_lo) * 0.7), max(crs_lo) * 1.3)
                ylim = (max(0, min(val_lo) - 3),      min(max(val_lo) + 5, 100))
                axins = ax.inset_axes([0.61, 0.55, 0.36, 0.40])
                for method, pts in all_pts.items():
                    _draw_method(axins, method, pts, lw=1.4, ms=5)
                axins.set_xlim(*xlim); axins.set_ylim(*ylim)
                axins.set_xscale("log")
                axins.tick_params(labelsize=7, pad=1)
                axins.set_xlabel("CR", fontsize=7, labelpad=1)
                axins.set_ylabel("PSNR (dB)", fontsize=7, labelpad=1)
                axins.grid(True, which="both", alpha=0.25, linestyle="--")
                axins.set_title("zoom", fontsize=7, pad=2, color="#333")
                for sp in axins.spines.values():
                    sp.set_linewidth(1.2); sp.set_edgecolor("#555")
                from matplotlib.patches import Rectangle
                rect = Rectangle((xlim[0], ylim[0]),
                                  xlim[1] - xlim[0], ylim[1] - ylim[0],
                                  linewidth=1.2, edgecolor="#555",
                                  facecolor="none", linestyle="--", zorder=5)
                ax.add_patch(rect)
        except Exception:
            pass

    out = ARGONNE / f"poster_rd_{fname_suffix}_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_field_panel(tc: np.ndarray, deim_panel: dict, krig_panel: dict,
                     ny: int, nx: int):
    """4-panel figure: true TC | DEIM pred | GP pred | error histograms."""
    lvl  = TIME_IDX
    true = tc[lvl]                                  # (ny, nx)
    vmin, vmax = np.percentile(true, [2, 98])

    deim_pred = deim_panel.get("pred")
    krig_pred = krig_panel.get("pred")
    if deim_pred is None and krig_pred is None:
        print("  Skipping field panel — no predictions available")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ax_true, ax_deim, ax_gp, ax_hist = axes

    kw = dict(cmap=CMAP_TC, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax_true.imshow(true, **kw)
    ax_true.set_title(f"True TC (level {lvl})", fontsize=11)

    if deim_pred is not None:
        ax_deim.imshow(deim_pred.reshape(ny, nx), **kw)
        ax_deim.set_title(f"DEIM-2D  k={deim_panel['k']}", fontsize=11)
    else:
        ax_deim.axis("off"); ax_deim.set_title("DEIM not run", fontsize=11)

    if krig_pred is not None:
        ax_gp.imshow(krig_pred.reshape(ny, nx), **kw)
        ax_gp.set_title(f"Kriging-2D  k={krig_panel['k']}", fontsize=11)
        sensors = krig_panel.get("sensors")
        if sensors is not None:
            sy, sx = np.unravel_index(sensors, (ny, nx))
            ax_gp.scatter(sx, sy, c="k", s=4, marker="x", linewidths=0.6,
                          zorder=5, alpha=0.5)
    else:
        ax_gp.axis("off"); ax_gp.set_title("GP not run", fontsize=11)

    ax_hist.set_title("Residual histograms", fontsize=11)
    bins = np.linspace(-5, 5, 80)
    if deim_pred is not None:
        resid_d = (true - deim_pred.reshape(ny, nx)).ravel()
        ax_hist.hist(resid_d, bins=bins, density=True, alpha=0.55,
                     label=f"DEIM  k={deim_panel['k']}", color=COLORS["DEIM-2D+SZ2"])
    if krig_pred is not None:
        resid_g = (true - krig_pred.reshape(ny, nx)).ravel()
        ax_hist.hist(resid_g, bins=bins, density=True, alpha=0.55,
                     label=f"GP  k={krig_panel['k']}", color=COLORS["Kriging-2D+SZ2"])
    ax_hist.set_xlabel("Residual (°C)"); ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.25)

    for ax in [ax_true, ax_deim, ax_gp]:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"ISABEL Hurricane TC  —  level {lvl}", fontsize=13, y=1.01)
    fig.tight_layout()
    out = ARGONNE / f"poster_field_panel_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_recon_panel(tc: np.ndarray, krig_panel: dict, ny: int, nx: int):
    """3-panel GP reconstruction: true | predicted | absolute error."""
    if krig_panel.get("pred") is None:
        print("  Skipping recon panel — no GP prediction available")
        return

    lvl  = TIME_IDX
    true = tc[lvl]
    pred = krig_panel["pred"].reshape(ny, nx)
    err  = np.abs(true - pred)

    vmin, vmax = np.percentile(true, [2, 98])
    emax = np.percentile(err, 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kw = dict(cmap=CMAP_TC, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    im0 = axes[0].imshow(true, **kw)
    axes[0].set_title(f"True TC  (level {lvl})", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="°C")

    im1 = axes[1].imshow(pred, **kw)
    axes[1].set_title(f"GP prediction  k={krig_panel['k']}", fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="°C")

    sensors = krig_panel.get("sensors")
    if sensors is not None:
        sy, sx = np.unravel_index(sensors, (ny, nx))
        axes[1].scatter(sx, sy, c="black", s=30, marker="x", linewidths=1.2,
                        zorder=5, alpha=0.9, label=f"{len(sensors)} sensors")

    diff  = true - pred
    dlim  = float(np.percentile(np.abs(diff), 99))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                          aspect="auto", origin="lower")
    axes[2].set_title("True − GP  (signed)", fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="°C")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = compute_psnr(true, pred)
    fig.suptitle(
        f"ISABEL Hurricane TC  —  GP prediction  (RMSE={rmse:.4f} °C, PSNR={psnr:.1f} dB)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    out = ARGONNE / f"recon_panel_GP_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_recon_panel_deim(tc: np.ndarray, deim_panel: dict, ny: int, nx: int):
    """3-panel DEIM reconstruction: true | predicted | absolute error.

    If deim_panel['pred'] is None (PLOTS_ONLY path), loads the checkpoint and
    reconstructs level TIME_IDX on the fly.
    """
    lvl  = TIME_IDX
    true = tc[lvl]

    pred_arr = deim_panel.get("pred")
    k_shown  = deim_panel.get("k", VIZ_K_DEIM)
    sensors_d = None

    if pred_arr is None:
        _ck_path = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
        if not _ck_path.exists():
            print("  Skipping DEIM recon panel — checkpoint not found")
            return
        print(f"  [DEIM recon panel] Loading {_ck_path.name} ...", flush=True)
        _ck       = np.load(_ck_path)
        Phi_max   = _ck["Phi_max"].astype(np.float64)
        mean_full = _ck["mean_full"].astype(np.float64)
        k_shown   = min(VIZ_K_DEIM, Phi_max.shape[1])
        Phi_k     = Phi_max[:, :k_shown]
        _, _, p   = scipy_qr(Phi_k.T, pivoting=True)
        sensors_d = p[:k_shown]
        A         = Phi_k[sensors_d, :]
        M         = Phi_k @ np.linalg.inv(A)
        anom      = true.ravel().astype(np.float64) - mean_full
        pred_arr  = (M @ anom[sensors_d] + mean_full).astype(np.float32).reshape(ny, nx)
        rmse = float(np.sqrt(np.mean((true.ravel() - pred_arr.ravel()) ** 2)))
        print(f"  [DEIM recon panel] k={k_shown}  RMSE={rmse:.4f} °C")

    pred = pred_arr.reshape(ny, nx)
    err  = np.abs(true - pred)

    vmin, vmax = np.percentile(true, [2, 98])
    emax = np.percentile(err, 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kw = dict(cmap=CMAP_TC, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    im0 = axes[0].imshow(true, **kw)
    axes[0].set_title(f"True TC  (level {lvl})", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="°C")

    im1 = axes[1].imshow(pred, **kw)
    axes[1].set_title(f"DEIM prediction  k={k_shown}", fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="°C")

    if sensors_d is not None:
        sy, sx = np.unravel_index(sensors_d, (ny, nx))
        axes[1].scatter(sx, sy, c="black", s=30, marker="x", linewidths=1.2,
                        zorder=5, alpha=0.9)

    diff  = true - pred
    dlim  = float(np.percentile(np.abs(diff), 99))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                          aspect="auto", origin="lower")
    axes[2].set_title("True − DEIM  (signed)", fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="°C")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = compute_psnr(true, pred)
    fig.suptitle(
        f"ISABEL Hurricane TC  —  DEIM prediction  (RMSE={rmse:.4f} °C, PSNR={psnr:.1f} dB)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    out = ARGONNE / f"recon_panel_DEIM_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_multigp(res: dict):
    """3-row × d-col panel: true | GP reconstruction | absolute error for U,V,W."""
    var_names = res["var_names"]
    d         = len(var_names)
    ny_s, nx_s = res["ny_s"], res["nx_s"]
    lvl       = res["lvl"]
    true_nd   = res["true_nd"]
    mu_full   = res["mu_full"]
    sensors   = res["sensors"]

    fig, axes = plt.subplots(3, d, figsize=(5 * d, 13),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    if d == 1:
        axes = axes[:, np.newaxis]

    sensors_rc = (sensors // nx_s, sensors % nx_s)

    for vi, vn in enumerate(var_names):
        true_2d = true_nd[:, vi].reshape(ny_s, nx_s)
        pred_2d = mu_full[:, vi].reshape(ny_s, nx_s)
        err_2d  = np.abs(true_2d - pred_2d)

        vmin, vmax = np.percentile(true_2d, [2, 98])
        emax = np.percentile(err_2d, 99)
        kw   = dict(cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")

        im0 = axes[0, vi].imshow(true_2d, **kw)
        axes[0, vi].set_title(f"{vn} — True  (level {lvl})", fontsize=9)
        axes[0, vi].scatter(sensors_rc[1], sensors_rc[0], c="k", s=8,
                            marker="x", linewidths=0.8, zorder=5,
                            label=f"{len(sensors)} sensors")
        plt.colorbar(im0, ax=axes[0, vi], fraction=0.046, pad=0.04,
                     label=MGP_UNITS.get(vn, ""))

        im1 = axes[1, vi].imshow(pred_2d, **kw)
        axes[1, vi].set_title(f"{vn} — LMC MultiGP reconstruction", fontsize=9)
        plt.colorbar(im1, ax=axes[1, vi], fraction=0.046, pad=0.04,
                     label=MGP_UNITS.get(vn, ""))

        im2 = axes[2, vi].imshow(err_2d, cmap="hot_r", vmin=0, vmax=emax,
                                  aspect="auto", origin="lower")
        diff    = mu_full[:, vi] - true_nd[:, vi]
        rmse_v  = float(np.sqrt(np.mean(diff ** 2)))
        axes[2, vi].set_title(f"{vn} — |error|  RMSE={rmse_v:.4f}", fontsize=9)
        plt.colorbar(im2, ax=axes[2, vi], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"ISABEL Hurricane — LMC MultiGP (GKS sensors)\n"
        f"Variables: {', '.join(var_names)}  |  "
        f"Level {lvl} reconstruction  |  "
        f"Sensors: {len(sensors)}  |  ls = {res['ls']:.2f}",
        fontsize=11, y=0.99
    )
    fig.tight_layout()
    out = ARGONNE / "multigp_hurricane_rd.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ISABEL Hurricane RD comparison  [{FIELD_TAG}]")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  TC file  : {TC_FILE}")
    print(f"  CSV      : {CSV_PATH}")
    print(f"  PLOTS_ONLY={PLOTS_ONLY}  USE_CHECKPOINT={USE_CHECKPOINT}\n")

    # ── Load TC data ──────────────────────────────────────────────────────────
    print("Loading TC data ...")
    tc, X = load_tc()
    n_T, ny, nx = tc.shape
    n_full = ny * nx

    # ── Load or run experiments ───────────────────────────────────────────────
    if PLOTS_ONLY and CSV_PATH.exists():
        print(f"PLOTS_ONLY: Loading results from {CSV_PATH}")
        by_method = load_csv()
        deim_panel = {"pred": None, "k": VIZ_K_DEIM}
        krig_panel = {"pred": None, "sensors": None, "k": VIZ_K_KRIG}
    else:
        all_rows: list = []

        # Load existing CSV for resume guards
        existing: dict = defaultdict(list)
        if CSV_PATH.exists():
            try:
                existing = load_csv()
                n_exist = sum(len(v) for v in existing.values())
                print(f"  Resuming: {n_exist} existing results loaded from CSV")
            except Exception as e:
                print(f"  Warning: could not load existing CSV ({e}); starting fresh")

        # ── SZ2 ───────────────────────────────────────────────────────────────
        if RUN_LIBPRESSIO:
            if existing.get("SZ2"):
                print(f"  [resume] SZ2 already in CSV ({len(existing['SZ2'])} rows) — skipping")
                all_rows.extend(existing["SZ2"])
            else:
                print("\n── SZ2 ────────────────────────────────────────────────")
                all_rows.extend(run_sz2(tc))

        # ── ZFP ───────────────────────────────────────────────────────────────
        if RUN_LIBPRESSIO and RUN_ZFP:
            if existing.get("ZFP"):
                print(f"  [resume] ZFP already in CSV ({len(existing['ZFP'])} rows) — skipping")
                all_rows.extend(existing["ZFP"])
            else:
                print("\n── ZFP ────────────────────────────────────────────────")
                all_rows.extend(run_zfp(tc))

        # ── SZ3 ───────────────────────────────────────────────────────────────────
        if RUN_SZ3:
            if existing.get("SZ3"):
                print(f"  [resume] SZ3 already in CSV ({len(existing['SZ3'])} rows) — skipping")
                all_rows.extend(existing["SZ3"])
            else:
                print("\n── SZ3 ────────────────────────────────────────────────")
                all_rows.extend(run_sz3(tc))

        # ── DEIM-2D hybrids (SZ2 + SZ3) ──────────────────────────────────────
        _deim_done = (existing.get("DEIM-2D+SZ2") and existing.get("DEIM-2D+SZ3"))
        if _deim_done:
            print(f"  [resume] DEIM-2D+SZ2 and DEIM-2D+SZ3 already in CSV — skipping")
            all_rows.extend(existing["DEIM-2D+SZ2"])
            all_rows.extend(existing["DEIM-2D+SZ3"])
            # Still build panel for plots: load checkpoint and reconstruct one level
            _ck_path = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
            if _ck_path.exists():
                _ck = np.load(_ck_path)
                Phi_max   = _ck["Phi_max"].astype(np.float64)
                mean_full = _ck["mean_full"]
                k_use = min(VIZ_K_DEIM, Phi_max.shape[1])
                Phi_k = Phi_max[:, :k_use]
                _, _, p   = scipy_qr(Phi_k.T, pivoting=True)
                sensors_d = p[:k_use]
                A     = Phi_k[sensors_d, :]
                data_flat = tc.reshape(n_T, n_full).astype(np.float64)
                anom  = data_flat[TIME_IDX] - mean_full
                M     = Phi_k @ np.linalg.inv(A)
                pred  = M @ anom[sensors_d] + mean_full
                deim_panel = {"pred": pred.reshape(ny, nx).astype(np.float32), "k": k_use}
            else:
                deim_panel = {"pred": None, "k": VIZ_K_DEIM}
        else:
            print("\n── DEIM-2D (SZ2 + SZ3 hybrids) ────────────────────────────")
            for m in ["DEIM-2D+SZ2", "DEIM-2D+SZ3"]:
                if existing.get(m):
                    all_rows.extend(existing[m])
            deim_rows, deim_panel = run_deim_2d(tc, X)
            _existing_methods = set(existing.keys())
            all_rows.extend([r for r in deim_rows if r["method"] not in _existing_methods])

        # ── Kriging-2D hybrids (SZ2 + SZ3) ───────────────────────────────────
        _krig_done = (existing.get("Kriging-2D+SZ2") and existing.get("Kriging-2D+SZ3"))
        if _krig_done:
            print(f"  [resume] Kriging-2D+SZ2 and Kriging-2D+SZ3 already in CSV — skipping")
            all_rows.extend(existing["Kriging-2D+SZ2"])
            all_rows.extend(existing["Kriging-2D+SZ3"])
            krig_panel = {"pred": None, "sensors": None, "k": VIZ_K_KRIG}
        else:
            print("\n── Kriging-2D (SZ2 + SZ3 hybrids) ─────────────────────────")
            for m in ["Kriging-2D+SZ2", "Kriging-2D+SZ3"]:
                if existing.get(m):
                    all_rows.extend(existing[m])
            krig_rows, krig_panel = run_kriging_2d(tc, X)
            _existing_methods = set(existing.keys())
            all_rows.extend([r for r in krig_rows if r["method"] not in _existing_methods])

        # Save CSV
        save_csv(all_rows)

        by_method = defaultdict(list)
        for r in all_rows:
            by_method[r["method"]].append(r)

    # ── MultiGP (U, V, W) — plotted separately ────────────────────────────────
    print("\n── MultiGP (U, V, W) ────────────────────────────────────────────")
    try:
        data_uvw = {}
        for vn, fname in MGP_VARS.items():
            fpath = DATA_DIR / fname
            if not fpath.exists():
                print(f"  Warning: {fname} not found — skipping MultiGP")
                data_uvw = {}
                break
            data_uvw[vn] = load_hurricane_var(fname)
            print(f"  Loaded {vn}: [{data_uvw[vn].min():.2f}, {data_uvw[vn].max():.2f}] "
                  f"{MGP_UNITS[vn]}")
        if data_uvw:
            mgp_res = run_multigp(data_uvw,
                                  n_sensors=200, downsample=3,
                                  test_level=TIME_IDX)
            plot_multigp(mgp_res)
    except Exception as e:
        print(f"  MultiGP failed: {e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────")
    plot_rd_poster(by_method, "psnr", "PSNR (dB)", "psnr")
    plot_field_panel(tc, deim_panel, krig_panel, ny, nx)
    plot_recon_panel(tc, krig_panel, ny, nx)
    plot_recon_panel_deim(tc, deim_panel, ny, nx)

    print("\nDone.")
    print(f"  poster_rd_psnr_{FIELD_TAG}.png")
    print(f"  poster_field_panel_{FIELD_TAG}.png")
    print(f"  recon_panel_{FIELD_TAG}.png")
    print(f"  multigp_hurricane_rd.png")
    print(f"  rd_results_{FIELD_TAG}.csv")
