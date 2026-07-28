#!/usr/bin/env python3
"""
sst_rd_comparison.py
====================
Rate-distortion comparison on NOAA OI SST V2 weekly mean data.

Methods
-------
  SZ2        — libpressio SZ2 (absolute error mode)
  ZFP        — libpressio ZFP (fixed-accuracy mode)
  DEIM-2D    — spatial SVD modes from N_TRAIN=260 evenly spaced snapshots;
               model applied to all n_T=1727 time steps
  Kriging-2D — spatial Matérn-3/2 GP with RPCholesky sensors;
               applied to all n_T=1727 time steps

Data
----
  sst.wkmean.1990-present.nc  —  1727 weekly snapshots × 180 lat × 360 lon
  Variable: 'sst' (°C, NOAA OI SST V2), no spatial downsampling

Compression pipeline (DEIM / GP)
---------------------------------
  1. Select N_TRAIN=260 evenly spaced training snapshots.
  2. Fit model (SVD basis or GP hyperparams + RPCholesky sensors) from training set.
  3. For EACH of the n_T=1727 time steps:
       predict from k sensor observations → quantize residual
  4. Store: model + sensor values (float16 + zstd) + quantized residuals (Huffman + zstd).
  5. CR = (n_T × ny × nx × 4) / total_compressed_bytes.

DEIM uses a vectorised reconstruction: M = Phi @ A⁻¹ precomputed once per k;
  reconstruction of all n_T snapshots is a single (n_T×k)@(k×n_2D) BLAS call.
GP uses a batched Cholesky solve + chunked matmul to keep peak memory ≤ 1 GB.

Plots produced
--------------
  rd_psnr_cr_SST.png          — PSNR vs compression ratio (mean over all T snapshots)
  rd_psnr_bpv_SST.png         — PSNR vs bits-per-value
  poster_rd_psnr_SST.png      — poster-style RD curve (zoom inset + below-axis legend)
  poster_field_panel_SST.png  — true field | DEIM | GP | residual histogram
  rd_budget_breakdown_SST.png — stacked bar: model / sensor / residual bytes per k
  rd_results_SST.csv          — full results table
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
DATA_PATH = ARGONNE.parent / "gpoed-code-python" / "sst.wkmean.1990-present.nc"
LP_MGP    = ARGONNE / "lp_multigp_compressor.py"   # Matérn kernel helpers (read-only ref)

# ── Compression parameters ────────────────────────────────────────────────────
N_TRAIN     = 1000                        # DEIM training snapshots (evenly spaced in time)
ABS_BOUNDS      = np.logspace(-4, -0.5, 8)   # absolute error tolerance sweep (°C)
ABS_BOUNDS_PRED = np.concatenate([ABS_BOUNDS, [0.5, 1.0, 2.0, 5.0]])  # extended for DEIM/Kriging
NUM_BINS    = 65536                       # quantisation bins (mirrors SZ2)

DEIM_K_VALS = list(range(100, 1001, 100))   # 100,200,...,1000  (≤ N_TRAIN=1000)
KRIG_K_VALS = list(range(100, 1001, 100))   # 100,200,...,1000

HIST_K      = 50    # fixed k used in the residual-histogram panel

# ── Visualisation ─────────────────────────────────────────────────────────────
TIME_IDX   = 864      # snapshot index for field panels (~mid-dataset, July 2006)
VIZ_K_DEIM = 100      # k shown in field panel for DEIM-2D
VIZ_K_KRIG = 100      # k shown in field panel for Kriging-2D
VIZ_AB     = 1e-2     # abs_bound shown in field panel (closest match used)

FIELD_TAG = "SST"
DPI       = 150
CMAP_SST  = "RdYlBu_r"   # standard oceanographic SST colormap

COLORS = {
    "SZ2":              "#1f77b4",
    "ZFP":              "#ff7f0e",
    "DEIM-2D":          "#2ca02c",
    "Kriging-2D":       "#9467bd",
    "DEIM-2D-SO":       "#2ca02c",    # same green, different marker
    "Kriging-2D-SO":    "#9467bd",    # same purple, different marker
    "DEIM-2D+SZ2":      "#17becf",    # cyan  — DEIM prediction + SZ2 residual
    "DEIM-2D+ZFP":      "#bcbd22",    # olive — DEIM prediction + ZFP residual
    "Kriging-2D+SZ2":   "#e377c2",    # pink  — GP prediction + SZ2 residual
    "Kriging-2D+ZFP":   "#8c564b",    # brown — GP prediction + ZFP residual
    "DEIM-2D-TT":       "#d62728",    # red    — TTHRESH-style coefficient quantization (no sensors)
    "DEIM-2D-L2":       "#e6550d",    # orange — sensors kept, L2/Frobenius residual quantization
    "Kriging-2D-L2":    "#756bb1",    # violet — sensors kept, L2/Frobenius residual quantization
}
MARKERS      = {"SZ2": "o", "ZFP": "s", "DEIM-2D": "^", "Kriging-2D": "D",
                "DEIM-2D-SO": "*", "Kriging-2D-SO": "P",
                "DEIM-2D+SZ2": "^", "DEIM-2D+ZFP": "^",
                "Kriging-2D+SZ2": "D", "Kriging-2D+ZFP": "D",
                "DEIM-2D-TT": "v",
                "DEIM-2D-L2": "<", "Kriging-2D-L2": ">"}
METHOD_ORDER = ["SZ2", "ZFP",
                # "DEIM-2D",      # temporarily hidden
                "DEIM-2D-L2",
                # "DEIM-2D-TT",   # temporarily hidden
                # "DEIM-2D+SZ2",  # temporarily hidden
                # "DEIM-2D+ZFP",  # temporarily hidden
                "DEIM-2D-SO",
                # "Kriging-2D",   # temporarily hidden
                "Kriging-2D-L2",
                # "Kriging-2D+SZ2", # temporarily hidden
                # "Kriging-2D+ZFP", # temporarily hidden
                "Kriging-2D-SO"]

# Methods that are single scatter points (one per k), not Pareto curves
_SO_METHODS = {"DEIM-2D-SO", "Kriging-2D-SO"}

# ── Flags ─────────────────────────────────────────────────────────────────────
RUN_LIBPRESSIO = True    # False → skip both SZ2 and ZFP
RUN_ZFP        = True    # fixed: removed invalid zfp:type/zfp:mode keys; use accuracy-only config
RUN_HYBRID     = False   # True  → run DEIM/Kriging+SZ2/ZFP hybrid methods
RUN_DEIM_TT    = False   # True  → run DEIM-2D-TT (TTHRESH-style coefficient quantisation)
PLOTS_ONLY     = False    # True  → skip compression, reload CSV, re-plot only
USE_CHECKPOINT = True     # True  → save/load SVD basis + GP hyperparams to/from disk
                          #         checkpoints are keyed by N_TRAIN; changing N_TRAIN
                          #         produces a new checkpoint automatically

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
def quantize(arr: np.ndarray, abs_bound: float, num_bins: int = NUM_BINS):
    """Uniform quantisation → int16 bins + float32 outliers (SZ2-style)."""
    bw   = 2.0 * abs_bound / num_bins
    flat = arr.ravel().astype(np.float64)
    idx  = np.round(flat / bw).astype(np.int32)
    half = num_bins // 2
    mask = (np.abs(idx) >= half)
    bins = np.clip(idx, -half + 1, half - 1).astype(np.int16)
    return bins, np.where(mask)[0].astype(np.int32), flat[mask].astype(np.float32)

def dequantize(bins, out_pos, out_vals, abs_bound, orig_shape, num_bins=NUM_BINS):
    bw   = 2.0 * abs_bound / num_bins
    flat = bins.ravel().astype(np.float64) * bw
    if len(out_pos):
        flat[out_pos] = out_vals.astype(np.float64)
    return flat.reshape(orig_shape)

def _huffman_estimate(bins: np.ndarray) -> int:
    """Estimate Huffman-coded byte size of a bin-index array."""
    _, counts = np.unique(bins.ravel(), return_counts=True)
    probs     = counts / counts.sum()
    entropy   = -np.sum(probs * np.log2(probs + 1e-12))
    return int(np.ceil(bins.size * entropy / 8)) + 64

def pack_encode(bins, out_pos, out_vals) -> bytes:
    """Huffman-estimate placeholder + zstd(outlier positions) + raw outlier values."""
    huff     = _huffman_estimate(bins)
    pos_enc  = _compress(out_pos.tobytes()) if len(out_pos) else b''
    return b'\x00' * huff + pos_enc + out_vals.tobytes()

def compress_f16(a): return _compress(a.astype(np.float16).tobytes())
def compress_f32(a): return _compress(a.astype(np.float32).tobytes())
def compress_i32(a): return _compress(a.astype(np.int32).tobytes())

# ── Dynamic module loader ──────────────────────────────────────────────────────
import importlib.util, sys

def load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── Image quality metrics ─────────────────────────────────────────────────────
def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-15: return 999.9
    dr = float(orig.max() - orig.min())
    if dr < 1e-10:  return 999.9
    return float(20.0 * np.log10(dr / np.sqrt(mse)))

def compute_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(orig.astype(np.float32), recon.astype(np.float32),
                                           data_range=float(orig.max() - orig.min())))
    except Exception:
        return float("nan")

def metrics(orig, recon):
    o, r = orig.astype(np.float32), recon.astype(np.float32)
    return compute_psnr(o, r), compute_ssim(o, r)

def ocean_metrics(orig2d: np.ndarray, recon2d: np.ndarray,
                  ocean_mask: np.ndarray):
    """PSNR computed only over ocean (True) pixels in a (ny, nx) snapshot.

    SSIM is returned as NaN — it is not meaningful over scattered 1-D pixels.
    """
    o = orig2d[ocean_mask].astype(np.float64)
    r = recon2d[ocean_mask].astype(np.float64)
    mse = float(np.mean((o - r) ** 2))
    if mse < 1e-15:
        return 999.9, float("nan")
    dr = float(o.max() - o.min())
    if dr < 1e-10:
        return 999.9, float("nan")
    return float(20.0 * np.log10(dr / np.sqrt(mse))), float("nan")

# ── Pareto-front helpers ───────────────────────────────────────────────────────
def _pareto_upper(pts, metric_key):
    """Upper-left Pareto front: highest metric at lowest CR.
    Uses >= so ties (common when avg PSNR smooths variation) are not dropped."""
    s = sorted(pts, key=lambda p: p["cr"])
    front, best = [], -np.inf
    for p in s:
        v = p.get(metric_key, float("nan"))
        if np.isfinite(v) and v >= best:
            front.append(p); best = v
    return front

def _pareto_upper_bpv(pts, metric_key):
    s = sorted(pts, key=lambda p: 32.0 / p["cr"])
    front, best = [], -np.inf
    for p in s:
        v = p.get(metric_key, float("nan"))
        if np.isfinite(v) and v >= best:
            front.append(p); best = v
    return front

# ── RPCholesky + RPGKS sensor selection ───────────────────────────────────────
def _rpcholesky_sensors(X: np.ndarray, ls: float, k: int, kern_fn, rng=None):
    """
    Select k spatial sensors via Randomly Pivoted Cholesky + column-pivoted QR.
    O(n × rank) memory — handles n = 64 800 without forming the full n×n kernel.
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
    _, V = np.linalg.eigh(G)               # ascending eigenvalues
    u_k  = F @ V[:, -k:]                   # (n, k) top-k left singular vectors
    norms = np.linalg.norm(u_k, axis=0, keepdims=True)
    u_k  /= np.where(norms > 1e-12, norms, 1.0)
    _, _, p = scipy_qr(u_k.T, pivoting=True)
    return p[:k].astype(np.int32)

# ── Data loading ───────────────────────────────────────────────────────────────
def load_sst():
    """Load SST NetCDF4 + companion land mask.

    Returns
    -------
    sst       : (n_T, ny, nx) float32 — raw field (land pixels filled by NOAA OI)
    lat       : (ny,)  float64
    lon       : (nx,)  float64
    ocean_mask: (ny, nx) bool — True = ocean pixel  (44 219 of 64 800 total)
    """
    import netCDF4 as nc
    ds  = nc.Dataset(str(DATA_PATH))
    sst = ds.variables["sst"][:]    # auto-applies scale_factor=0.01
    lat = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds.variables["lon"][:], dtype=np.float64)
    ds.close()
    if isinstance(sst, np.ma.MaskedArray):
        sst = sst.filled(np.nan)

    # ── Companion land/sea mask (0 = land, 1 = ocean) ─────────────────────────
    mask_path = DATA_PATH.parent / "lsmask.nc"
    ds2       = nc.Dataset(str(mask_path))
    raw_mask  = np.array(ds2.variables["mask"][0], dtype=np.int16)   # (180, 360)
    ds2.close()
    ocean_mask = (raw_mask == 1)   # True = ocean
    n_ocean    = int(ocean_mask.sum())
    n_land     = int((~ocean_mask).sum())
    print(f"  Land mask: {n_ocean} ocean / {n_land} land pixels "
          f"({100*n_ocean/(n_ocean+n_land):.1f}% ocean)")

    return sst.astype(np.float32), lat, lon, ocean_mask

# ── CSV I/O ────────────────────────────────────────────────────────────────────
CSV_PATH   = ARGONNE / f"rd_results_{FIELD_TAG}.csv"
CSV_FIELDS = ["method", "k", "abs_bound", "cr", "psnr", "ssim",
              "comp_sec", "train_sec", "compressed_MB",
              "model_MB", "sv_MB", "resid_MB", "n_outliers"]

def save_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {CSV_PATH}")

def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            for fld in ("cr", "psnr", "ssim", "comp_sec", "train_sec",
                        "compressed_MB", "model_MB", "sv_MB", "resid_MB", "abs_bound"):
                try:   row[fld] = float(row[fld]) if row.get(fld, "") not in ("", "None") else float("nan")
                except: row[fld] = float("nan")
            try:   row["k"] = int(row["k"])
            except: pass
            rows.append(row)
    bm = defaultdict(list)
    for r in rows:
        bm[r["method"]].append(r)
    return bm

# ── libpressio: SZ2 and ZFP ───────────────────────────────────────────────────
def _run_lp(data: np.ndarray, compressor_id: str, config: dict):
    """One libpressio encode→decode cycle. Returns (recon, cr, comp_sec, decomp_sec)."""
    import libpressio
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config":  {"pressio:metric": "composite", "composite:plugins": ["size"]},
        "compressor_config": config,
    })
    recon = data.copy()
    t0         = time.perf_counter()
    compressed = comp.encode(data)
    comp_sec   = time.perf_counter() - t0
    t0         = time.perf_counter()
    recon      = comp.decode(compressed, recon)
    decomp_sec = time.perf_counter() - t0
    cr         = comp.get_metrics().get("size:compression_ratio", float("nan"))
    return recon, cr, comp_sec, decomp_sec

def _run_slicewise(data: np.ndarray, compressor_id: str, config: dict,
                   time_idx: int, ocean_mask: np.ndarray = None):
    """
    Compress each (ny, nx) time slice independently — correct for SZ2/ZFP,
    which are 2-D compressors applied snapshot-by-snapshot (no cross-time
    prediction).  Returns (recon_snap, total_compressed_bytes, comp_sec, avg_psnr).

    The compressor object is created ONCE and reused across all slices to
    avoid repeated C-level alloc/free cycles that cause heap corruption on
    macOS (malloc: Incorrect checksum for freed object).

    CR is computed as:
        CR = (n_T × ny × nx × 4) / sum_over_t(compressed_bytes_per_slice)

    If ocean_mask is provided, avg_psnr is the mean PSNR over all n_T snapshots
    (ocean pixels only).  Otherwise avg_psnr = nan.
    """
    import libpressio
    n_T, ny, nx = data.shape
    slice_orig   = ny * nx * 4          # bytes per uncompressed float32 slice
    total_comp_b = 0
    recon_snap   = None
    psnr_sum     = 0.0
    psnr_count   = 0
    t0           = time.perf_counter()

    # Create compressor once — reuse for every slice
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config":  {"pressio:metric": "composite", "composite:plugins": ["size"]},
        "compressor_config": config,
    })

    for t in range(n_T):
        sl       = np.ascontiguousarray(data[t])   # (ny, nx) float32
        recon_sl = sl.copy()
        compressed = comp.encode(sl)
        recon_sl   = comp.decode(compressed, recon_sl)
        cr_t       = comp.get_metrics().get("size:compression_ratio", float("nan"))
        if np.isfinite(cr_t) and cr_t > 0:
            total_comp_b += int(round(slice_orig / cr_t))
        else:
            total_comp_b += slice_orig        # fallback: assume no compression
        if t == time_idx:
            recon_snap = recon_sl             # (ny, nx) float32
        if ocean_mask is not None:
            pv_t, _ = ocean_metrics(data[t], recon_sl, ocean_mask)
            if np.isfinite(pv_t):
                psnr_sum += pv_t
                psnr_count += 1

    comp_sec = time.perf_counter() - t0
    avg_psnr = psnr_sum / psnr_count if psnr_count > 0 else float("nan")
    return recon_snap, total_comp_b, comp_sec, avg_psnr


def run_sz2(data: np.ndarray, ocean_mask: np.ndarray, abs_bounds=ABS_BOUNDS):
    """
    Compress each (ny, nx) snapshot independently with SZ2 (2-D Lorenzo predictor).
    Total CR aggregated over all 1727 slices. Metrics over ocean pixels only.
    """
    n_T, ny, nx = data.shape
    n_all = data.size
    results = []
    for ab in abs_bounds:
        config = {"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)}
        recon_snap, total_b, comp_sec, pv = _run_slicewise(
            data, "sz", config, TIME_IDX, ocean_mask)
        cr = (n_all * 4) / total_b if total_b > 0 else float("nan")
        sv = float("nan")
        results.append({"method": "SZ2", "k": 0, "abs_bound": ab,
            "cr": cr, "psnr": pv, "ssim": sv,
            "comp_sec": comp_sec, "train_sec": 0.0,
            "compressed_MB": total_b / 1e6,
            "model_MB": 0.0, "sv_MB": 0.0, "resid_MB": 0.0, "n_outliers": 0})
        print(f"  SZ2  ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results


def run_zfp(data: np.ndarray, ocean_mask: np.ndarray, abs_bounds=ABS_BOUNDS):
    """
    Compress each (ny, nx) snapshot independently with ZFP (2-D transform).
    Total CR aggregated over all 1727 slices. Metrics over ocean pixels only.
    """
    n_T, ny, nx = data.shape
    n_all = data.size
    results = []
    for ab in abs_bounds:
        config = {"zfp:accuracy": float(ab)}
        recon_snap, total_b, comp_sec, pv = _run_slicewise(
            data, "zfp", config, TIME_IDX, ocean_mask)
        cr = (n_all * 4) / total_b if total_b > 0 else float("nan")
        sv = float("nan")
        results.append({"method": "ZFP", "k": 0, "abs_bound": ab,
            "cr": cr, "psnr": pv, "ssim": sv,
            "comp_sec": comp_sec, "train_sec": 0.0,
            "compressed_MB": total_b / 1e6,
            "model_MB": 0.0, "sv_MB": 0.0, "resid_MB": 0.0, "n_outliers": 0})
        print(f"  ZFP  ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results

# ── Hybrid helper: compress residual field with SZ2 or ZFP ────────────────────
def _lp_on_residual(resid_full, recon_flat, data, ocean_mask,
                    abs_bounds, model_b, sv_enc,
                    n_all, n_T, ny, nx, k, method_name, train_sec):
    """
    Compress the prediction residual (n_T, ny, nx) with SZ2 or ZFP slice-by-slice.
    Returns a list of result-dict rows (one per abs_bound).

    method_name must contain '+SZ2' or '+ZFP' to select the compressor.
    """
    try:
        import libpressio as _lp
    except ImportError:
        print(f"  libpressio not available — skipping {method_name}")
        return []

    cid = "sz" if "+SZ2" in method_name else "zfp"
    if cid == "zfp" and not RUN_ZFP:
        return []

    slice_orig = ny * nx * 4
    rows = []
    for ab in abs_bounds:
        cfg = ({"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)}
               if cid == "sz" else {"zfp:accuracy": float(ab)})
        comp = _lp.PressioCompressor.from_config({
            "compressor_id": cid,
            "early_config":  {"pressio:metric": "composite", "composite:plugins": ["size"]},
            "compressor_config": cfg,
        })
        total_rb = 0; psnr_vals = []; t0 = time.perf_counter()
        for t in range(n_T):
            sl  = np.ascontiguousarray(resid_full[t])   # (ny, nx) residual slice
            rsl = sl.copy()
            rsl = comp.decode(comp.encode(sl), rsl)
            cr_t = comp.get_metrics().get("size:compression_ratio", float("nan"))
            total_rb += (int(round(slice_orig / cr_t))
                         if np.isfinite(cr_t) and cr_t > 0 else slice_orig)
            # Reconstruction = DEIM/GP prediction + decompressed residual
            fin_t = recon_flat[t].reshape(ny, nx) + rsl
            pv_t, _ = ocean_metrics(data[t].reshape(ny, nx), fin_t, ocean_mask)
            if np.isfinite(pv_t): psnr_vals.append(pv_t)
        cs  = time.perf_counter() - t0
        tb  = model_b + len(sv_enc) + total_rb
        cr  = (n_all * 4) / tb
        pv  = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
        rows.append({"method": method_name, "k": k, "abs_bound": ab,
                     "cr": cr, "psnr": pv, "ssim": float("nan"),
                     "comp_sec": cs, "train_sec": train_sec,
                     "compressed_MB": tb / 1e6, "model_MB": model_b / 1e6,
                     "sv_MB": len(sv_enc) / 1e6, "resid_MB": total_rb / 1e6,
                     "n_outliers": 0})
        print(f"       [{method_name}] k={k}  ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return rows

# ── DEIM-2D: spatial SVD modes from N_TRAIN-snapshot training set ──────────────
def run_deim_2d(data: np.ndarray, ocean_mask: np.ndarray, abs_bounds=ABS_BOUNDS_PRED):
    """
    Build SVD basis from N_TRAIN=260 evenly spaced snapshots.
    Training and sensor selection operate on ocean pixels only (n_ocean = 44 219).
    Land pixels are set to zero in `data` (data_m from main); their residuals
    are identically zero so no bits are spent on land.

    Reconstruction uses the precomputed DEIM interpolation operator
      M = Phi_k @ A⁻¹  (n_ocean × k)
    so the full n_T-snapshot recon is a single BLAS matmul:
      recon_ocean = all_sv @ M.T + mean_ocean      (n_T × n_ocean, float32)

    CR = (n_T × ny × nx × 4) / (model_bytes + sv_float16 + quantised_residuals)
    Metrics computed over ocean pixels only.
    """
    n_T, ny, nx = data.shape
    n_2D  = ny * nx
    n_all = n_T * n_2D

    # ── Ocean pixel bookkeeping ───────────────────────────────────────────────
    ocean_flat = ocean_mask.ravel()                 # (n_2D,) bool
    ocean_idx  = np.where(ocean_flat)[0]            # (n_ocean,) int
    n_ocean    = int(ocean_flat.sum())              # 44 219

    print(f"\n[DEIM-2D]  n_T={n_T}  n_ocean={n_ocean}  n_train={N_TRAIN}")

    # ── SVD basis: load from checkpoint or compute fresh ─────────────────────
    _deim_ckpt = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
    k_max = min(max(DEIM_K_VALS), min(N_TRAIN, n_ocean))

    if USE_CHECKPOINT and _deim_ckpt.exists():
        print(f"  [checkpoint] Loading SVD basis from {_deim_ckpt.name} ...", flush=True)
        _ck       = np.load(_deim_ckpt)
        Phi_max   = _ck["Phi_max"].astype(np.float64)   # saved as float32, upcast for numerics
        mean_ocean = _ck["mean_ocean"]
        s_vals    = _ck["s_vals"]
        train_idx = _ck["train_idx"]
        k_max     = min(k_max, Phi_max.shape[1])        # can't exceed what was saved
        train_sec = 0.0
        print(f"  [checkpoint] Phi_max {Phi_max.shape}  mean_ocean {mean_ocean.shape}  "
              f"k_max={k_max}  (train_sec=0 — skipped SVD)")
    else:
        train_idx   = np.round(np.linspace(0, n_T - 1, N_TRAIN)).astype(int)
        train_ocean = data[train_idx].reshape(N_TRAIN, n_2D)[:, ocean_idx].astype(np.float64)
        mean_ocean  = train_ocean.mean(axis=0)           # (n_ocean,)
        F           = train_ocean - mean_ocean
        print(f"  SVD ({N_TRAIN} × {n_ocean}), k_max={k_max} ...", flush=True)
        t0 = time.perf_counter()
        _, s_vals, Vt = np.linalg.svd(F, full_matrices=False)
        Phi_max   = Vt[:k_max, :].T                     # (n_ocean, k_max)
        train_sec = time.perf_counter() - t0
        if USE_CHECKPOINT:
            np.savez_compressed(str(_deim_ckpt),
                                Phi_max=Phi_max.astype(np.float32),
                                mean_ocean=mean_ocean, s_vals=s_vals, train_idx=train_idx)
            print(f"  [checkpoint] SVD basis saved → {_deim_ckpt.name}")

    total_var = float((s_vals ** 2).sum())
    print(f"  SVD done: {train_sec:.1f}s  |  variance explained by k modes (training set):")
    for kv in DEIM_K_VALS:
        kc = min(kv, len(s_vals))
        ve = float((s_vals[:kc] ** 2).sum() / total_var)
        print(f"    k={kv:4d} → {100*ve:.2f}%")

    mean_c = compress_f32(mean_ocean)               # store ocean mean (not full grid)
    # Pre-centre all n_T ocean observations in float32
    data_ocean = (data.reshape(n_T, n_2D)[:, ocean_idx]
                  - mean_ocean.astype(np.float32))  # (n_T, n_ocean)

    results = []
    pred_snap_deim    = None   # pure DEIM prediction at TIME_IDX (no correction)
    recon_snap_deim   = None   # DEIM + correction at TIME_IDX
    sensors_snap_deim = None   # full-grid sensor indices at VIZ_K_DEIM
    _deim_panel_ab    = np.inf

    for k in DEIM_K_VALS:
        k     = min(k, k_max)
        Phi_k = Phi_max[:, :k]                      # (n_ocean, k)

        # Q-DEIM: column-pivoted QR → k ocean sensor locations (indices in ocean space)
        _, _, p = scipy_qr(Phi_k.T, pivoting=True)
        sensors      = p[:k]                        # (k,) indices into [0, n_ocean)
        sensors_full = ocean_idx[sensors]           # (k,) indices into [0, n_2D) — for storage

        phi_c   = compress_f16(Phi_k)
        sens_c  = compress_i32(sensors_full)        # store full-grid indices
        model_b = len(phi_c) + len(sens_c) + len(mean_c)

        A      = Phi_k[sensors, :]                  # (k, k) DEIM interpolation matrix
        all_sv = data_ocean[:, sensors]             # (n_T, k) centred sensor observations
        sv_enc = _compress(all_sv.astype(np.float16).tobytes())

        # M = Phi_k @ A⁻¹  (n_ocean × k);  recon_ocean = all_sv @ M.T + mean_ocean
        print(f"  k={k:3d}  computing DEIM operator & reconstructing {n_T} snapshots ...",
              flush=True)
        M           = (Phi_k @ np.linalg.inv(A.astype(np.float64))).astype(np.float32)
        recon_ocean = all_sv @ M.T + mean_ocean.astype(np.float32)   # (n_T, n_ocean)

        # Fill full-grid reconstruction (land stays 0 = same as data_m)
        recon_flat = np.zeros((n_T, n_2D), dtype=np.float32)
        recon_flat[:, ocean_idx] = recon_ocean
        # Ocean-only residual: land pixels are 0 in both data_m and recon_flat,
        # so storing them wastes ~32 % of residual space on guaranteed zeros.
        resid = (data.reshape(n_T, n_2D)[:, ocean_idx]
                 - recon_flat[:, ocean_idx]).ravel().astype(np.float32)  # (n_T*n_ocean,)

        # Capture the pure DEIM prediction (no correction) at TIME_IDX for the panel
        if k == VIZ_K_DEIM:
            pred_snap_deim = recon_flat[TIME_IDX].reshape(ny, nx).copy()
            sensors_snap_deim = sensors_full.copy()

        # ── Sensor-only operating point: model + sv, no stored residual ──────
        total_so = model_b + len(sv_enc)
        cr_so    = (n_all * 4) / total_so
        train_set = set(train_idx.tolist())
        psnr_so_train, psnr_so_test, psnr_so_vals = [], [], []
        rmse_so_vals = []
        for t in range(n_T):
            orig_oc  = data[t].reshape(ny, nx)[ocean_mask].astype(np.float64)
            recon_oc = recon_flat[t].reshape(ny, nx)[ocean_mask].astype(np.float64)
            pv_t, _  = ocean_metrics(data[t].reshape(ny, nx),
                                     recon_flat[t].reshape(ny, nx), ocean_mask)
            rmse_t = float(np.sqrt(np.mean((orig_oc - recon_oc) ** 2)))
            if np.isfinite(pv_t):
                psnr_so_vals.append(pv_t)
                rmse_so_vals.append(rmse_t)
                (psnr_so_train if t in train_set else psnr_so_test).append(pv_t)
        psnr_so       = float(np.mean(psnr_so_vals))   if psnr_so_vals   else float("nan")
        psnr_so_tr    = float(np.mean(psnr_so_train))  if psnr_so_train  else float("nan")
        psnr_so_te    = float(np.mean(psnr_so_test))   if psnr_so_test   else float("nan")
        rmse_so_med   = float(np.median(rmse_so_vals)) if rmse_so_vals   else float("nan")
        results.append({"method": "DEIM-2D-SO", "k": k, "abs_bound": 0.0,
                        "cr": cr_so, "psnr": psnr_so, "ssim": float("nan"),
                        "comp_sec": 0.0, "train_sec": train_sec,
                        "compressed_MB": total_so / 1e6,
                        "model_MB": model_b / 1e6,
                        "sv_MB":    len(sv_enc) / 1e6,
                        "resid_MB": 0.0, "n_outliers": 0})
        print(f"       [SO] k={k}  CR={cr_so:.1f}×  avg PSNR={psnr_so:.1f} dB  "
              f"(train={psnr_so_tr:.1f} / test={psnr_so_te:.1f} dB)  "
              f"median RMSE={rmse_so_med:.3f}°C")

        # ── DEIM-2D-TT: TTHRESH-style quantization of SVD coefficients ────────
        if RUN_DEIM_TT:
            # Project full data onto SVD basis, quantise k temporal coefficients.
            # Model: Phi_k + mean (no sensor indices). Decode: C_deq @ Phi_k.T + mean.
            print(f"  [DEIM-TT] k={k}  computing optimal projection ...", flush=True)
            _t0_tt = time.perf_counter()
            C_tt = data_ocean.astype(np.float64) @ Phi_k.astype(np.float64)
            recon_tt_oc   = (C_tt @ Phi_k.T.astype(np.float64)).astype(np.float32)
            recon_tt_full = np.zeros((n_T, n_2D), dtype=np.float32)
            recon_tt_full[:, ocean_idx] = recon_tt_oc + mean_ocean.astype(np.float32)
            _proj_sec = time.perf_counter() - _t0_tt
            model_b_tt = len(phi_c) + len(mean_c)

            for eps_rms in abs_bounds:
                q_tt         = 2.0 * float(eps_rms) * np.sqrt(n_ocean / k)
                abs_bound_tt = q_tt * NUM_BINS / 2

                _t0 = time.perf_counter()
                bins_c, op_c, ov_c = quantize(C_tt.ravel(), abs_bound_tt)
                coeff_enc = pack_encode(bins_c, op_c, ov_c)
                _cs = time.perf_counter() - _t0

                total_b_tt = model_b_tt + len(coeff_enc)
                cr_tt      = (n_all * 4) / total_b_tt

                C_deq = dequantize(bins_c, op_c, ov_c, abs_bound_tt,
                                   (n_T * k,)).reshape(n_T, k)
                delta_oc     = (C_deq - C_tt).astype(np.float32) @ Phi_k.T.astype(np.float32)
                recon_q_full = recon_tt_full.copy()
                recon_q_full[:, ocean_idx] += delta_oc

                psnr_tt = []
                for t in range(n_T):
                    pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                            recon_q_full[t].reshape(ny, nx), ocean_mask)
                    if np.isfinite(pv_t): psnr_tt.append(pv_t)
                pv_tt = float(np.mean(psnr_tt)) if psnr_tt else float("nan")

                results.append({"method": "DEIM-2D-TT", "k": k, "abs_bound": eps_rms,
                                "cr": cr_tt, "psnr": pv_tt, "ssim": float("nan"),
                                "comp_sec": _proj_sec + _cs, "train_sec": train_sec,
                                "compressed_MB": total_b_tt / 1e6,
                                "model_MB": model_b_tt / 1e6,
                                "sv_MB":    0.0,
                                "resid_MB": len(coeff_enc) / 1e6,
                                "n_outliers": int(len(op_c))})
                print(f"       [DEIM-TT] k={k}  eps={eps_rms:.2e}  "
                      f"CR={cr_tt:.1f}×  PSNR={pv_tt:.1f} dB")

        # ── DEIM-2D-L2: same sensors, L2/Frobenius residual quantization ─────────
        # bin width = 2 × eps_rms  (pixels always in-range, no outlier bloat)
        # abs_bound_L2 = eps_rms × NUM_BINS  → full range = 2 × eps_rms × NUM_BINS
        print(f"  [DEIM-L2] k={k}  running L2 residual quantization ...", flush=True)
        for eps_rms in abs_bounds:
            abs_bound_L2 = float(eps_rms) * NUM_BINS   # huge range; bin width = 2*eps_rms

            _t0 = time.perf_counter()
            bins_r2, op_r2, ov_r2 = quantize(resid, abs_bound_L2)
            resid_enc_L2 = pack_encode(bins_r2, op_r2, ov_r2)
            _cs_L2 = time.perf_counter() - _t0

            total_b_L2 = model_b + len(sv_enc) + len(resid_enc_L2)
            cr_L2      = (n_all * 4) / total_b_L2

            resid_rec_L2 = dequantize(bins_r2, op_r2, ov_r2, abs_bound_L2,
                                      (n_T * n_ocean,)).reshape(n_T, n_ocean)
            final_flat_L2 = recon_flat.copy()
            final_flat_L2[:, ocean_idx] += resid_rec_L2.astype(np.float32)

            psnr_L2 = []
            for t in range(n_T):
                pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                        final_flat_L2[t].reshape(ny, nx), ocean_mask)
                if np.isfinite(pv_t): psnr_L2.append(pv_t)
            pv_L2 = float(np.mean(psnr_L2)) if psnr_L2 else float("nan")

            results.append({"method": "DEIM-2D-L2", "k": k, "abs_bound": eps_rms,
                            "cr": cr_L2, "psnr": pv_L2, "ssim": float("nan"),
                            "comp_sec": _cs_L2, "train_sec": train_sec,
                            "compressed_MB": total_b_L2 / 1e6,
                            "model_MB": model_b / 1e6,
                            "sv_MB":    len(sv_enc) / 1e6,
                            "resid_MB": len(resid_enc_L2) / 1e6,
                            "n_outliers": int(len(op_r2))})
            print(f"       [DEIM-L2] k={k}  eps={eps_rms:.2e}  "
                  f"CR={cr_L2:.1f}×  PSNR={pv_L2:.1f} dB  "
                  f"outliers={len(op_r2)}")

        k_results = []
        for ab in abs_bounds:
            t0 = time.perf_counter()
            bins_r, op_r, ov_r = quantize(resid, ab)
            resid_enc = pack_encode(bins_r, op_r, ov_r)
            comp_sec  = time.perf_counter() - t0

            data_b  = len(sv_enc) + len(resid_enc)
            total_b = model_b + data_b
            cr      = (n_all * 4) / total_b

            # Dequantize ocean-only residual and add back to ocean columns
            resid_rec_oc = dequantize(bins_r, op_r, ov_r, ab,
                                      (n_T * n_ocean,)).reshape(n_T, n_ocean)
            final_flat = recon_flat.copy()
            final_flat[:, ocean_idx] += resid_rec_oc.astype(np.float32)
            # Average PSNR over all n_T snapshots (ocean pixels only)
            psnr_vals = []
            for t in range(n_T):
                pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                        final_flat[t].reshape(ny, nx), ocean_mask)
                if np.isfinite(pv_t):
                    psnr_vals.append(pv_t)
            pv   = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
            sv_m = float("nan")

            # Capture the corrected reconstruction at TIME_IDX for the panel
            if k == VIZ_K_DEIM and abs(ab - VIZ_AB) < abs(_deim_panel_ab - VIZ_AB):
                recon_snap_deim = final_flat[TIME_IDX].reshape(ny, nx).copy()
                _deim_panel_ab  = ab

            row = {"method": "DEIM-2D", "k": k, "abs_bound": ab,
                   "cr": cr, "psnr": pv, "ssim": sv_m,
                   "comp_sec": comp_sec, "train_sec": train_sec,
                   "compressed_MB": total_b / 1e6,
                   "model_MB": model_b / 1e6,
                   "sv_MB":    len(sv_enc) / 1e6,
                   "resid_MB": len(resid_enc) / 1e6,
                   "n_outliers": len(op_r)}
            results.append(row); k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"       model={model_b/1e3:.1f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR: {min(psnrs):.1f}–{max(psnrs):.1f} dB")

        # ── Hybrid: SZ2 / ZFP applied to the DEIM residual ───────────────────
        if RUN_HYBRID and RUN_LIBPRESSIO:
            resid_full = (data - recon_flat.reshape(n_T, ny, nx)).astype(np.float32)
            for mname in ("DEIM-2D+SZ2", "DEIM-2D+ZFP"):
                results += _lp_on_residual(
                    resid_full, recon_flat, data, ocean_mask,
                    abs_bounds, model_b, sv_enc,
                    n_all, n_T, ny, nx, k, mname, train_sec)

    deim_panel = {
        "pred":    pred_snap_deim,
        "recon":   recon_snap_deim,
        "sensors": sensors_snap_deim,
        "k":       VIZ_K_DEIM,
    }
    return results, deim_panel

# ── Kriging-2D: spatial Matérn GP ─────────────────────────────────────────────
def run_kriging_2d(data: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   ocean_mask: np.ndarray, abs_bounds=ABS_BOUNDS_PRED):
    """
    Fit Matérn-3/2 GP hyperparams on N_TRAIN=260 training snapshots.
    All spatial operations (coordinates, training, sensor selection, K_Xs)
    use ocean pixels only (n_ocean = 44 219), reducing K_Xs from
    (64 800 × k) to (44 219 × k) — ~32 % smaller, faster, less memory.

    K_Xs and L_k are NOT stored — recomputed at decode time from hyperparams.
    Reconstruct all n_T=1727 snapshots via batched Cholesky + chunked matmul.

    CR = (n_T × ny × nx × 4) / (model_bytes + sv_float16 + quantised_residuals)
    Metrics over ocean pixels only.
    """
    mgp = load_mod(str(LP_MGP), "lp_multigp")

    n_T, ny, nx = data.shape
    n_2D  = ny * nx
    n_all = n_T * n_2D

    # ── Ocean pixel bookkeeping ───────────────────────────────────────────────
    ocean_flat = ocean_mask.ravel()                 # (n_2D,) bool
    ocean_idx  = np.where(ocean_flat)[0]            # (n_ocean,) int
    n_ocean    = int(ocean_flat.sum())              # 44 219

    # Normalise lat/lon to [0,1]×[0,1] for Matérn distance metric
    lat_n = (lat - lat.min()) / (lat.max() - lat.min())
    lon_n = (lon - lon.min()) / (lon.max() - lon.min())
    LON_G, LAT_G = np.meshgrid(lon_n, lat_n)
    X_full = np.column_stack([LAT_G.ravel(), LON_G.ravel()])  # (n_2D, 2)
    X_all  = X_full[ocean_idx]                                 # (n_ocean, 2) — ocean only

    print(f"\n[Kriging-2D]  n_T={n_T}  n_ocean={n_ocean}  n_train={N_TRAIN}")
    t0 = time.perf_counter()

    k_max = min(max(KRIG_K_VALS), n_ocean)

    # ── Hyperparams + sensors: load from checkpoint or fit fresh ─────────────
    _krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"

    if USE_CHECKPOINT and _krig_ckpt.exists():
        print(f"  [checkpoint] Loading GP hyperparams + sensors from {_krig_ckpt.name} ...",
              flush=True)
        _ck        = np.load(_krig_ckpt)
        ls         = float(_ck["ls"])
        _var       = float(_ck["var"])
        noise_var  = float(_ck["noise_var"])
        B          = np.array([[_var]])
        sensors    = _ck["sensors"]           # (k_max,) ocean-pixel indices
        train_mean     = _ck["train_mean"]
        train_std_safe = _ck["train_std_safe"]
        k_max      = min(k_max, len(sensors))
        train_sec  = 0.0
        print(f"  [checkpoint] ls={ls:.4f}  var={_var:.4f}  noise={noise_var:.2e}  "
              f"k_max={k_max}  (skipped MLE + RPCholesky)")
    else:
        train_idx   = np.round(np.linspace(0, n_T - 1, N_TRAIN)).astype(int)
        train_ocean = data[train_idx].reshape(N_TRAIN, n_2D)[:, ocean_idx].astype(np.float64)
        train_mean     = train_ocean.mean(axis=0)
        train_std      = train_ocean.std(axis=0)
        train_std_safe = np.where(train_std < 1e-10, 1.0, train_std)

        fit_size = min(k_max * 4, n_ocean)
        rng_fit  = np.random.default_rng(0)
        fit_idx  = rng_fit.choice(n_ocean, size=fit_size, replace=False)
        X_fit    = X_all[fit_idx]
        Y_fit    = ((train_ocean[0] - train_mean) / train_std_safe)[fit_idx]

        print(f"  Fitting ls + var + noise on {fit_size} ocean pts, 3 restarts ...", flush=True)
        from scipy.optimize import minimize as _minimize
        from scipy.spatial.distance import cdist as _cdist

        def _neg_lml(log_theta):
            ls_    = float(np.exp(log_theta[0]))
            var_   = float(np.exp(log_theta[1]))
            noise_ = float(np.exp(log_theta[2]))
            K = var_ * mgp.matern32(X_fit, X_fit, ls_) + noise_ * np.eye(fit_size)
            try:
                L_ = np.linalg.cholesky(K + 1e-8 * np.eye(fit_size))
            except np.linalg.LinAlgError:
                return 1e10
            log_det = 2.0 * np.sum(np.log(np.diag(L_)))
            alpha   = np.linalg.solve(L_.T, np.linalg.solve(L_, Y_fit))
            return 0.5 * float(Y_fit @ alpha) + 0.5 * log_det

        dists  = _cdist(X_fit[:10], X_fit[:10], "euclidean")
        ls0    = float(np.median(dists[dists > 0])) / 3.0
        var0   = max(float(np.var(Y_fit)), 1e-6)
        noise0 = var0 * 0.01
        bnds   = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
        rng_hp = np.random.default_rng(0)
        starts = [np.log([max(ls0, 0.1), max(var0, 0.01), max(noise0, 1e-6)])] + [
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

        ls, noise_var, B = best_ls, best_noise, np.array([[best_var]])
        print(f"  ls={ls:.4f}  var={B[0,0]:.4f}  noise={noise_var:.2e}")

        print(f"  RPCholesky: selecting {k_max} sensors from {n_ocean} ocean pts ...", flush=True)
        sensors = _rpcholesky_sensors(X_all, ls, k_max, mgp.matern32)  # (k_max,) ocean indices

        if USE_CHECKPOINT:
            np.savez_compressed(str(_krig_ckpt),
                                ls=np.float64(ls), var=np.float64(B[0, 0]),
                                noise_var=np.float64(noise_var),
                                sensors=sensors,
                                train_mean=train_mean,
                                train_std_safe=train_std_safe)
            print(f"  [checkpoint] GP hyperparams + sensors saved → {_krig_ckpt.name}")

        train_sec = time.perf_counter() - t0

    # Save hyperparameters to human-readable text (always)
    hp_path = ARGONNE / f"kriging_hyperparams_{FIELD_TAG}.txt"
    with open(hp_path, "w") as _hpf:
        _hpf.write(f"Matérn-3/2 Kriging hyperparameters (MLE, N_TRAIN={N_TRAIN})\n")
        _hpf.write(f"  length_scale (ls)     = {ls:.6f}  (normalised [0,1]×[0,1])\n")
        _hpf.write(f"  signal_variance (var) = {B[0,0]:.6f}\n")
        _hpf.write(f"  noise_variance        = {noise_var:.2e}\n")
        _hpf.write(f"  signal-to-noise ratio = {B[0,0]/noise_var:.1f}\n")
        _hpf.write(f"  N_TRAIN               = {N_TRAIN}\n")

    # ── Rebuild K_Xs and K_ss from (possibly loaded) ls + sensors ────────────
    sensors_full = ocean_idx[sensors]         # (k_max,) full-grid indices
    X_sens_max   = X_all[sensors]            # (k_max, 2)
    print(f"  Building K_Xs ({n_ocean} × {k_max}) and K_ss ({k_max} × {k_max}) ...", flush=True)
    K_Xs_max = mgp.matern32(X_all, X_sens_max, ls)
    K_ss_max = mgp.matern32(X_sens_max, X_sens_max, ls) + 1e-6 * np.eye(k_max)

    # Model bytes shared across all k: ocean mean + std (stored once)
    mean_c = compress_f32(train_mean)
    std_c  = compress_f32(train_std_safe)

    print(f"  Training done: {train_sec:.1f}s")

    # Pre-collect sensor values for ALL n_T snapshots: (n_T, k_max)
    data_flat  = data.reshape(n_T, n_2D)[:, ocean_idx].astype(np.float64)  # (n_T, n_ocean)
    all_sv_max = data_flat[:, sensors].astype(np.float32)                   # (n_T, k_max)

    results = []
    pred_snap_krig    = None   # pure GP prediction at TIME_IDX (no correction)
    recon_snap_krig   = None   # GP + correction at TIME_IDX
    sensors_snap_krig = None   # full-grid sensor indices at VIZ_K_KRIG
    _krig_panel_ab    = np.inf

    for k in KRIG_K_VALS:
        k        = min(k, k_max)
        s_k      = sensors[:k]                  # (k,) ocean-pixel sensor indices
        s_k_full = sensors_full[:k]             # (k,) full-grid sensor indices
        K_Xs     = K_Xs_max[:, :k]             # (n_ocean, k)
        K_ss_k   = K_ss_max[:k, :k]

        K_sub = B[0, 0] * K_ss_k + noise_var * np.eye(k)
        try:    L_k, lower = cho_factor(K_sub, lower=True)
        except: L_k, lower = cho_factor(K_sub + 1e-4 * np.eye(k), lower=True)

        # Model: hyperparams + full-grid sensor indices (K_Xs recomputed at decode time)
        hyperparams_b = struct.pack("<IIddd", ny, nx, float(ls), float(B[0, 0]), float(noise_var))
        sens_c  = compress_i32(s_k_full)
        model_b = len(hyperparams_b) + len(sens_c) + len(mean_c) + len(std_c)

        all_sv = all_sv_max[:, :k]   # (n_T, k) sensor values
        sv_enc = _compress(all_sv.astype(np.float16).tobytes())

        # ── Reconstruct all n_T snapshots (ocean pixels only) ────────────────
        # y_norm: (k, n_T);  alpha = (B*K_ss + σ²I)⁻¹ y_norm: (k, n_T)
        # K_Xs @ alpha: (n_ocean, n_T) — processed in time chunks to limit RAM
        print(f"  k={k:3d}  reconstructing {n_T} snapshots ...", flush=True)
        ms_k   = train_mean[s_k]
        ss_k   = train_std_safe[s_k]
        y_norm = ((all_sv.astype(np.float64) - ms_k) / ss_k).T   # (k, n_T)
        alpha  = B[0, 0] * cho_solve((L_k, lower), y_norm)        # (k, n_T)

        CHUNK = 200
        recon_flat = np.zeros((n_T, n_2D), dtype=np.float32)
        for i in range(0, n_T, CHUNK):
            mu = (K_Xs @ alpha[:, i:i+CHUNK]).T                   # (chunk, n_ocean)
            mu = mu * train_std_safe + train_mean
            mu[:, zero_std] = train_mean[zero_std]
            recon_flat[i:i+CHUNK, ocean_idx] = mu.astype(np.float32)
        # Land pixels remain 0 — same as data (data_m has land=0)

        # Capture pure GP prediction at TIME_IDX for the panel (before correction)
        if k == VIZ_K_KRIG:
            pred_snap_krig    = recon_flat[TIME_IDX].reshape(ny, nx).copy()
            sensors_snap_krig = s_k_full.copy()

        # ── Sensor-only operating point: model + sv, no stored residual ──────
        total_so = model_b + len(sv_enc)
        cr_so    = (n_all * 4) / total_so
        psnr_so_vals = []
        for t in range(n_T):
            pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                    recon_flat[t].reshape(ny, nx), ocean_mask)
            if np.isfinite(pv_t):
                psnr_so_vals.append(pv_t)
        psnr_so = float(np.mean(psnr_so_vals)) if psnr_so_vals else float("nan")
        results.append({"method": "Kriging-2D-SO", "k": k, "abs_bound": 0.0,
                        "cr": cr_so, "psnr": psnr_so, "ssim": float("nan"),
                        "comp_sec": 0.0, "train_sec": train_sec,
                        "compressed_MB": total_so / 1e6,
                        "model_MB": model_b / 1e6,
                        "sv_MB":    len(sv_enc) / 1e6,
                        "resid_MB": 0.0, "n_outliers": 0})
        print(f"       [SO] k={k}  CR={cr_so:.1f}×  avg PSNR={psnr_so:.1f} dB  "
              f"(model+sv only, no residual)")

        # Ocean-only residual: skip land zeros → ~32% smaller residual array
        resid = (data.reshape(n_T, n_2D)[:, ocean_idx]
                 - recon_flat[:, ocean_idx]).ravel().astype(np.float32)  # (n_T*n_ocean,)

        # ── Kriging-2D-L2: same sensors, L2/Frobenius residual quantization ──────
        # bin width = 2 × eps_rms; abs_bound_L2 = eps_rms × NUM_BINS (no outliers)
        print(f"  [Kriging-L2] k={k}  running L2 residual quantization ...", flush=True)
        for eps_rms in abs_bounds:
            abs_bound_L2 = float(eps_rms) * NUM_BINS

            _t0 = time.perf_counter()
            bins_r2, op_r2, ov_r2 = quantize(resid, abs_bound_L2)
            resid_enc_L2 = pack_encode(bins_r2, op_r2, ov_r2)
            _cs_L2 = time.perf_counter() - _t0

            total_b_L2 = model_b + len(sv_enc) + len(resid_enc_L2)
            cr_L2      = (n_all * 4) / total_b_L2

            resid_rec_L2 = dequantize(bins_r2, op_r2, ov_r2, abs_bound_L2,
                                      (n_T * n_ocean,)).reshape(n_T, n_ocean)
            final_flat_L2 = recon_flat.copy()
            final_flat_L2[:, ocean_idx] += resid_rec_L2.astype(np.float32)

            psnr_L2 = []
            for t in range(n_T):
                pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                        final_flat_L2[t].reshape(ny, nx), ocean_mask)
                if np.isfinite(pv_t): psnr_L2.append(pv_t)
            pv_L2 = float(np.mean(psnr_L2)) if psnr_L2 else float("nan")

            results.append({"method": "Kriging-2D-L2", "k": k, "abs_bound": eps_rms,
                            "cr": cr_L2, "psnr": pv_L2, "ssim": float("nan"),
                            "comp_sec": _cs_L2, "train_sec": train_sec,
                            "compressed_MB": total_b_L2 / 1e6,
                            "model_MB": model_b / 1e6,
                            "sv_MB":    len(sv_enc) / 1e6,
                            "resid_MB": len(resid_enc_L2) / 1e6,
                            "n_outliers": int(len(op_r2))})
            print(f"       [Kriging-L2] k={k}  eps={eps_rms:.2e}  "
                  f"CR={cr_L2:.1f}×  PSNR={pv_L2:.1f} dB  "
                  f"outliers={len(op_r2)}")

        k_results = []
        for ab in abs_bounds:
            t0 = time.perf_counter()
            bins_r, op_r, ov_r = quantize(resid, ab)
            resid_enc = pack_encode(bins_r, op_r, ov_r)
            comp_sec  = time.perf_counter() - t0

            data_b  = len(sv_enc) + len(resid_enc)
            total_b = model_b + data_b
            cr      = (n_all * 4) / total_b

            # Dequantize ocean-only residual and add back to ocean columns
            resid_rec_oc = dequantize(bins_r, op_r, ov_r, ab,
                                      (n_T * n_ocean,)).reshape(n_T, n_ocean)
            final_flat = recon_flat.copy()
            final_flat[:, ocean_idx] += resid_rec_oc.astype(np.float32)
            # Average PSNR over all n_T snapshots (ocean pixels only)
            psnr_vals = []
            for t in range(n_T):
                pv_t, _ = ocean_metrics(data[t].reshape(ny, nx),
                                        final_flat[t].reshape(ny, nx), ocean_mask)
                if np.isfinite(pv_t):
                    psnr_vals.append(pv_t)
            pv   = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
            sv_m = float("nan")

            # Capture corrected reconstruction at TIME_IDX for the panel
            if k == VIZ_K_KRIG and abs(ab - VIZ_AB) < abs(_krig_panel_ab - VIZ_AB):
                recon_snap_krig = final_flat[TIME_IDX].reshape(ny, nx).copy()
                _krig_panel_ab  = ab

            row = {"method": "Kriging-2D", "k": k, "abs_bound": ab,
                   "cr": cr, "psnr": pv, "ssim": sv_m,
                   "comp_sec": comp_sec, "train_sec": train_sec,
                   "compressed_MB": total_b / 1e6,
                   "model_MB": model_b / 1e6,
                   "sv_MB":    len(sv_enc) / 1e6,
                   "resid_MB": len(resid_enc) / 1e6,
                   "n_outliers": len(op_r)}
            results.append(row); k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"       model={model_b/1e3:.0f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR: {min(psnrs):.1f}–{max(psnrs):.1f} dB")

        # ── Hybrid: SZ2 / ZFP applied to the GP residual ─────────────────────
        if RUN_HYBRID and RUN_LIBPRESSIO:
            resid_full = (data - recon_flat.reshape(n_T, ny, nx)).astype(np.float32)
            for mname in ("Kriging-2D+SZ2", "Kriging-2D+ZFP"):
                results += _lp_on_residual(
                    resid_full, recon_flat, data, ocean_mask,
                    abs_bounds, model_b, sv_enc,
                    n_all, n_T, ny, nx, k, mname, train_sec)

    krig_panel = {
        "pred":    pred_snap_krig,
        "recon":   recon_snap_krig,
        "sensors": sensors_snap_krig,
        "k":       VIZ_K_KRIG,
    }
    return results, krig_panel

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _valid(pts, key):
    return [p for p in pts
            if np.isfinite(p.get("cr", np.nan)) and np.isfinite(p.get(key, np.nan))]

# ── Standard RD comparison plots ──────────────────────────────────────────────
def make_rd_plot(by_method, metric_key, ylabel, fname):
    """PSNR or SSIM vs compression ratio."""
    # Skip entirely if no method has valid data for this metric
    if not any(_valid(by_method.get(m, []), metric_key) for m in METHOD_ORDER):
        print(f"  Skipping {fname} — no valid {metric_key} data"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHOD_ORDER:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            continue
        c, mk = COLORS[method], MARKERS[method]
        if method in _SO_METHODS:
            srt = sorted(pts, key=lambda p: p["k"])
            ax.scatter([p["cr"] for p in srt], [p[metric_key] for p in srt],
                       color=c, marker=mk, s=120, zorder=5,
                       label=f"{method} (sensor-only)")
            for p in srt:
                ax.annotate(f"k={p['k']}", (p["cr"], p[metric_key]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7, color=c)
        elif method in ("SZ2", "ZFP"):
            srt = sorted(pts, key=lambda p: p["cr"])
            ab0, ab1 = srt[0]["abs_bound"], srt[-1]["abs_bound"]
            ax.plot([p["cr"] for p in srt], [p[metric_key] for p in srt],
                    "-", color=c, marker=mk, lw=2, ms=7,
                    label=f"{method} (ε={ab0:.0e}–{ab1:.0e})")
        else:
            env = _pareto_upper(pts, metric_key)
            if not env:
                continue
            kv = sorted(set(p["k"] for p in env))
            ax.plot([p["cr"] for p in env], [p[metric_key] for p in env],
                    "--", color=c, marker=mk, lw=2, ms=7,
                    label=f"{method} (k={kv[0]}–{kv[-1]})")
    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio (log scale)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_title(f"Rate–Distortion  —  NOAA OI SST V2  (mean PSNR over all T snapshots)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {ARGONNE / fname}")

def make_bpv_plot(by_method, metric_key, ylabel, fname):
    """PSNR or SSIM vs bits-per-value."""
    if not any(_valid(by_method.get(m, []), metric_key) for m in METHOD_ORDER):
        print(f"  Skipping {fname} — no valid {metric_key} data"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHOD_ORDER:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            continue
        c, mk = COLORS[method], MARKERS[method]
        if method in _SO_METHODS:
            srt = sorted(pts, key=lambda p: p["k"])
            ax.scatter([32.0 / p["cr"] for p in srt], [p[metric_key] for p in srt],
                       color=c, marker=mk, s=120, zorder=5,
                       label=f"{method} (sensor-only)")
            for p in srt:
                ax.annotate(f"k={p['k']}", (32.0 / p["cr"], p[metric_key]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7, color=c)
        elif method in ("SZ2", "ZFP"):
            srt = sorted(pts, key=lambda p: 32.0 / p["cr"])
            ax.plot([32.0 / p["cr"] for p in srt], [p[metric_key] for p in srt],
                    "-", color=c, marker=mk, lw=2, ms=7, label=method)
        else:
            env = _pareto_upper_bpv(pts, metric_key)
            if not env:
                continue
            ax.plot([32.0 / p["cr"] for p in env], [p[metric_key] for p in env],
                    "--", color=c, marker=mk, lw=2, ms=7, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Bit rate (bits per value)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_title(f"Rate–Distortion (BPV)  —  NOAA OI SST V2", fontsize=12)
    fig.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {ARGONNE / fname}")

# ── Poster-style RD curve (zoom inset + legend below axis) ───────────────────
def plot_rd_poster(by_method, metric_key, ylabel, fname_suffix):
    """
    Poster-quality RD plot mirroring poster_plots.py::plot_rd_curves:
      • Solid lines for SZ2/ZFP, dashed Pareto fronts for GP/DEIM.
      • Legend placed below the x-axis label.
      • Zoom inset in the top-right corner highlighting the GP/DEIM low-CR region.
      • Dashed rectangle on the main axes indicating the zoomed region.
    """
    if not any(_valid(by_method.get(m, []), metric_key) for m in METHOD_ORDER):
        print(f"  Skipping poster RD ({fname_suffix}) — no valid {metric_key} data"); return
    fig, ax = plt.subplots(figsize=(9, 7))

    def _draw_method(target_ax, method, pts, lw=2.0, ms=8):
        c, mk = COLORS[method], MARKERS[method]
        if method in _SO_METHODS:
            srt = sorted(pts, key=lambda p: p["k"])
            target_ax.scatter([p["cr"] for p in srt], [p[metric_key] for p in srt],
                              color=c, marker=mk, s=ms * 15, zorder=5)
        elif method in ("SZ2", "ZFP"):
            srt = sorted(pts, key=lambda p: p["cr"])
            target_ax.plot([p["cr"] for p in srt], [p[metric_key] for p in srt],
                           "-", color=c, marker=mk, lw=lw, ms=ms)
        else:
            env = _pareto_upper(pts, metric_key)
            if env:
                target_ax.plot([p["cr"] for p in env], [p[metric_key] for p in env],
                               "--", color=c, marker=mk, lw=lw, ms=ms)

    all_pts = {}
    for method in METHOD_ORDER:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            continue
        all_pts[method] = pts
        c, mk = COLORS[method], MARKERS[method]
        if method in _SO_METHODS:
            srt    = sorted(pts, key=lambda p: p["k"])
            k_vals = [p["k"] for p in srt]
            lbl    = f"{method} (k={k_vals[0]}–{k_vals[-1]})"
            ax.scatter([p["cr"] for p in srt], [p[metric_key] for p in srt],
                       color=c, marker=mk, s=130, zorder=5, label=lbl)
            for p in srt:
                ax.annotate(f"k={p['k']}", (p["cr"], p[metric_key]),
                            textcoords="offset points", xytext=(5, 3), fontsize=8, color=c)
        elif method in ("SZ2", "ZFP"):
            ab_vals = sorted(set(p["abs_bound"] for p in pts))
            lbl = f"{method} (ε={ab_vals[0]:.0e}–{ab_vals[-1]:.0e})"
            srt = sorted(pts, key=lambda p: p["cr"])
            ax.plot([p["cr"] for p in srt], [p[metric_key] for p in srt],
                    "-", color=c, marker=mk, lw=2.2, ms=10, label=lbl)
        else:
            env    = _pareto_upper(pts, metric_key)
            k_vals = sorted(set(p["k"] for p in env)) if env else []
            lbl    = f"{method} (k={k_vals[0]}–{k_vals[-1]})" if k_vals else method
            if env:
                ax.plot([p["cr"] for p in env], [p[metric_key] for p in env],
                        "--", color=c, marker=mk, lw=2.2, ms=10, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_title(f"Rate–Distortion  —  NOAA OI SST V2  (mean PSNR over all T snapshots)", fontsize=13)
    ax.legend(fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              handlelength=1.8, columnspacing=1.0,
              framealpha=0.9, borderpad=0.6)

    # Zoom inset — focus on the hybrid (+SZ2/+ZFP) and SO operating points
    if metric_key == "psnr" and len(all_pts) >= 2:
        try:
            hybrid_methods = ["DEIM-2D+SZ2", "DEIM-2D+ZFP",
                              "Kriging-2D+SZ2", "Kriging-2D+ZFP",
                              "DEIM-2D-SO", "Kriging-2D-SO"]
            zoom_pts = []
            for m in hybrid_methods:
                zoom_pts += _valid(by_method.get(m, []), metric_key)
            # Fall back to all non-SZ2/ZFP methods if hybrids have no data yet
            if not zoom_pts:
                for m in ["DEIM-2D", "Kriging-2D"]:
                    zoom_pts += _valid(by_method.get(m, []), metric_key)
            lo_pts = zoom_pts if zoom_pts else []
            if lo_pts:
                crs_lo = [p["cr"] for p in lo_pts]
                val_lo = [p[metric_key] for p in lo_pts]
                xlim = (max(0.5, min(crs_lo) * 0.7), max(crs_lo) * 1.3)
                ylim = (max(0, min(val_lo) - 3),      min(max(val_lo) + 5, 100))
                axins = ax.inset_axes([0.60, 0.50, 0.37, 0.42])
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

# ── Budget breakdown ───────────────────────────────────────────────────────────
def make_budget_breakdown(by_method, fname):
    """
    2×2 stacked-bar budget figure:
      Row 0: DEIM-2D-L2 and Kriging-2D-L2 — bars grouped by k, stacked model/sv/residuals
      Row 1: SZ2 and ZFP                  — bars grouped by abs_bound, single 'compressed' bar
    """
    FS_TITLE = 13; FS_LABEL = 13; FS_TICK = 10; FS_LEGEND = 10
    COL_MODEL = "#4472C4"; COL_SV = "#ED7D31"; COL_RES = "#70AD47"
    COL_COMP  = "#A9A9A9"   # grey for raw compressor output

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Row 0: DEIM-2D-L2 and Kriging-2D-L2 ─────────────────────────────────
    for ax, method in zip(axes[0], ["DEIM-2D-L2", "Kriging-2D-L2"]):
        pts = by_method.get(method, [])
        if not pts:
            ax.set_title(f"{method} (no data)", fontsize=FS_TITLE); continue
        k_groups = defaultdict(list)
        for p in pts:
            k_groups[p["k"]].append(p)

        ks, model_mb, sv_mb, resid_mb = [], [], [], []
        for kv in sorted(k_groups):
            best = min(k_groups[kv], key=lambda x: x["abs_bound"])
            ks.append(kv)
            model_mb.append(best.get("model_MB", 0.0))
            sv_mb.append(best.get("sv_MB", float("nan")))
            resid_mb.append(best.get("resid_MB", float("nan")))

        have_split = not any(np.isnan(v) for v in sv_mb)
        x         = np.arange(len(ks))
        model_arr = np.array(model_mb)
        sv_arr    = np.array(sv_mb)
        res_arr   = np.array(resid_mb)

        ax.bar(x, model_arr, label="Model overhead", color=COL_MODEL)
        if have_split:
            ax.bar(x, sv_arr,  bottom=model_arr,          label="Sensor values  ↑ k", color=COL_SV)
            ax.bar(x, res_arr, bottom=model_arr + sv_arr, label="Residuals  ↓ k",      color=COL_RES)
        else:
            ax.bar(x, np.zeros(len(ks)), bottom=model_arr, label="Data", color=COL_SV)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={kv}" for kv in ks], fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_ylabel("Total compressed size — all snapshots (MB)", fontsize=FS_LABEL)
        ax.set_title(f"{method} — storage budget at tightest ε\n"
                     "model (amortised) + sensor values + L2 residuals", fontsize=FS_TITLE)
        ax.legend(fontsize=FS_LEGEND)

    # ── Row 1: SZ2 and ZFP ───────────────────────────────────────────────────
    for ax, method in zip(axes[1], ["SZ2", "ZFP"]):
        pts = by_method.get(method, [])
        if not pts:
            ax.set_title(f"{method} (no data)", fontsize=FS_TITLE); continue

        srt = sorted(pts, key=lambda p: p["abs_bound"], reverse=True)   # loose→tight
        abs_vals  = [p["abs_bound"]    for p in srt]
        comp_mb   = [p["compressed_MB"] for p in srt]
        x = np.arange(len(srt))

        ax.bar(x, comp_mb, color=COL_COMP, label="Compressed data")
        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={a:.0e}" for a in abs_vals],
                           fontsize=FS_TICK, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_ylabel("Total compressed size — all snapshots (MB)", fontsize=FS_LABEL)
        ax.set_title(f"{method} — total compressed size vs. error bound\n"
                     "(per-snapshot entropy coding, summed across all snapshots)", fontsize=FS_TITLE)
        ax.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {ARGONNE / fname}")

# ── Compression timing comparison ─────────────────────────────────────────────
def make_timing_plot(by_method, fname):
    """
    Two-panel timing comparison:
      Left:  total encode time per snapshot (ms/snapshot), bar chart per method.
             For SZ2/ZFP: comp_sec / n_T (all training baked in per-snapshot).
             For DEIM/Kriging: (train_sec + comp_sec) / n_T amortised.
      Right: encode throughput (MB raw / s).

    Uses the row with the median abs_bound for each method (representative point).
    """
    FS = 11
    N_T_SNAP = 1727          # total snapshots in dataset
    RAW_MB   = N_T_SNAP * 44219 * 4 / 1e6   # ocean-pixel raw size (305 MB)

    methods_data = {}
    for method in METHOD_ORDER:
        pts = by_method.get(method, [])
        if not pts: continue
        # Pick representative point: median abs_bound (exclude abs_bound=0 for SO)
        pts_ab = [p for p in pts if p.get("abs_bound", 0) > 0]
        if not pts_ab: pts_ab = pts
        pts_ab = sorted(pts_ab, key=lambda p: p.get("abs_bound", 0))
        rep = pts_ab[len(pts_ab) // 2]
        total_sec = float(rep.get("train_sec", 0) or 0) + float(rep.get("comp_sec", 0) or 0)
        if total_sec <= 0: continue
        ms_per_snap = total_sec / N_T_SNAP * 1000
        throughput  = RAW_MB / total_sec          # MB/s
        methods_data[method] = (ms_per_snap, throughput, rep.get("abs_bound", 0))

    if not methods_data:
        print(f"  Skipping {fname} — no timing data"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x     = np.arange(len(methods_data))
    names = list(methods_data.keys())
    cols  = [COLORS.get(m, "#888") for m in names]

    ax1.bar(x, [methods_data[m][0] for m in names], color=cols)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=40, ha="right", fontsize=FS-2)
    ax1.set_ylabel("Amortised encode time (ms / snapshot)", fontsize=FS)
    ax1.set_title("Encode time per snapshot\n(train amortised over all snapshots)", fontsize=FS)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, [methods_data[m][1] for m in names], color=cols)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=40, ha="right", fontsize=FS-2)
    ax2.set_ylabel("Throughput (ocean MB / s)", fontsize=FS)
    ax2.set_title("Encode throughput\n(higher = faster)", fontsize=FS)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Compression timing — NOAA OI SST V2  (train cost amortised)", fontsize=FS+1)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {ARGONNE / fname}")

# ── Poster-style field panel ───────────────────────────────────────────────────
def plot_field_panel(data: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     by_method, ocean_mask: np.ndarray,
                     deim_panel: dict = None, krig_panel: dict = None):
    """
    2×2 poster panel for one time snapshot (TIME_IDX):
      (0,0) True SST  (land shown as gray)
      (0,1) DEIM-2D reconstruction  (VIZ_K_DEIM, VIZ_AB)
      (1,0) Kriging-2D reconstruction (VIZ_K_KRIG, VIZ_AB)
      (1,1) Overlapping residual histogram  (HIST_K, VIZ_AB)

    `data` is data_m (land pixels = 0).  ocean_mask is used to grey out land.
    All DEIM/GP computations train on ocean pixels only.
    """
    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16,
    })

    n_T, ny, nx = data.shape
    ocean_flat = ocean_mask.ravel()
    ocean_idx  = np.where(ocean_flat)[0]
    n_ocean    = int(ocean_flat.sum())

    snap = data[TIME_IDX]   # (ny, nx) — land=0 in data_m
    # For colour scale and display: use ocean pixels only
    ocean_vals = snap[ocean_mask]
    vmin = float(np.nanpercentile(ocean_vals, 2))
    vmax = float(np.nanpercentile(ocean_vals, 98))
    norm   = Normalize(vmin=vmin, vmax=vmax)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    cmap_obj = plt.get_cmap(CMAP_SST).copy()
    cmap_obj.set_bad("lightgray")   # land → gray

    def _to_display(arr2d):
        """Return a masked array with land → NaN (displayed as gray)."""
        out = arr2d.astype(np.float32).copy()
        out[~ocean_mask] = np.nan
        return out

    fig = plt.figure(figsize=(26, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

    def _show(ax_, arr2d, title_, sensor_idx=None):
        """arr2d is raw (land may be 0); _to_display handles the masking."""
        im = ax_.imshow(_to_display(arr2d), cmap=cmap_obj, norm=norm,
                        extent=extent, origin="upper", aspect="auto")
        ax_.set_title(title_, fontsize=16, pad=8)
        ax_.set_xlabel("Longitude (°E)", fontsize=13)
        ax_.set_ylabel("Latitude (°N)", fontsize=13)
        plt.colorbar(im, ax=ax_, fraction=0.025, pad=0.04, label="SST (°C)")
        if sensor_idx is not None and len(sensor_idx):
            # sensor_idx expected in full-grid space
            lons_s = lon[sensor_idx % nx]
            lats_s = lat[sensor_idx // nx]
            ax_.scatter(lons_s, lats_s, s=40, c="k", marker="x", lw=0.8,
                        alpha=0.75, zorder=5)

    # (0,0) True field
    ax00 = fig.add_subplot(gs[0, 0])
    _show(ax00, snap, f"True SST  (snapshot {TIME_IDX})")

    # Helper: find CSV row nearest to (k_target, ab_target) for a method
    def _csv_row(method, k_target, ab_target):
        pts = by_method.get(method, [])
        kpts = [p for p in pts if p.get("k") == k_target]
        if not kpts:
            kpts = sorted(pts, key=lambda p: abs(p.get("k", 0) - k_target))[:1]
        return min(kpts, key=lambda p: abs(p.get("abs_bound", 1e9) - ab_target)) if kpts else None

    # ── DEIM-2D reconstruction for the panel ─────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    recon_d = None; pred_d = None; sensors_d_full = None
    try:
        if deim_panel is not None and deim_panel.get("recon") is not None:
            # Use the reconstruction already computed by run_deim_2d — guaranteed
            # to match the evaluated results exactly.
            recon_d        = deim_panel["recon"]
            pred_d         = deim_panel["pred"]
            sensors_d_full = deim_panel["sensors"]
            k_eff          = deim_panel["k"]
        else:
            # Fallback: recompute inline (e.g. PLOTS_ONLY mode)
            train_idx   = np.round(np.linspace(0, n_T - 1, N_TRAIN)).astype(int)
            train_ocean = data[train_idx].reshape(N_TRAIN, ny * nx)[:, ocean_idx].astype(np.float64)
            mean_ocean  = train_ocean.mean(axis=0)
            F           = train_ocean - mean_ocean
            k_eff       = min(VIZ_K_DEIM, min(N_TRAIN, n_ocean))
            _, _, Vt    = np.linalg.svd(F, full_matrices=False)
            Phi_k       = Vt[:k_eff, :].T
            _, _, p_    = scipy_qr(Phi_k.T, pivoting=True)
            sensors_d      = p_[:k_eff]
            sensors_d_full = ocean_idx[sensors_d]
            A           = Phi_k[sensors_d, :]
            snap_ocean  = snap.ravel()[ocean_idx].astype(np.float64)
            sv_d        = (snap_ocean - mean_ocean)[sensors_d]
            c_d         = np.linalg.solve(A, sv_d)
            recon_ocean_d = (Phi_k @ c_d + mean_ocean).astype(np.float32)
            recon_d     = np.zeros(ny * nx, dtype=np.float32)
            recon_d[ocean_idx] = recon_ocean_d
            recon_d     = recon_d.reshape(ny, nx)
            pred_d      = recon_d.copy()
            resid_d_oc  = snap_ocean.astype(np.float32) - recon_ocean_d
            bins_, op_, ov_ = quantize(resid_d_oc, VIZ_AB)
            corr_ocean  = dequantize(bins_, op_, ov_, VIZ_AB, (n_ocean,)).astype(np.float32)
            recon_d.ravel()[ocean_idx] += corr_ocean

        psnr_d   = ocean_metrics(snap, recon_d, ocean_mask)[0]
        row_d    = _csv_row("DEIM-2D", VIZ_K_DEIM, VIZ_AB)
        cr_d_str = f"{row_d['cr']:.1f}×" if row_d and np.isfinite(row_d.get("cr", np.nan)) else "–"
        _show(ax01, recon_d,
              f"DEIM-2D  k={k_eff},  ε={VIZ_AB:.0e}\nCR={cr_d_str}   PSNR={psnr_d:.1f} dB",
              sensors_d_full)
    except Exception as e:
        ax01.set_title("DEIM-2D (error)", fontsize=14); ax01.axis("off")
        print(f"  [field panel] DEIM-2D error: {e}")

    # ── Kriging-2D reconstruction for the panel ───────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    recon_g = None; pred_g = None
    try:
        if krig_panel is not None and krig_panel.get("recon") is not None:
            # Use the reconstruction already computed by run_kriging_2d with the
            # fitted hyperparameters — matches the evaluated results exactly.
            recon_g     = krig_panel["recon"]
            pred_g      = krig_panel["pred"]
            sens_g_full = krig_panel["sensors"]
            k_eff_g     = krig_panel["k"]
        else:
            # Fallback: recompute inline with fixed ls=0.15 (PLOTS_ONLY mode)
            mgp_p = load_mod(str(LP_MGP), "lp_multigp_panel")
            lat_n  = (lat - lat.min()) / (lat.max() - lat.min())
            lon_n  = (lon - lon.min()) / (lon.max() - lon.min())
            LON_G, LAT_G = np.meshgrid(lon_n, lat_n)
            X_all_p  = np.column_stack([LAT_G.ravel(), LON_G.ravel()])[ocean_idx]
            train_idx   = np.round(np.linspace(0, n_T - 1, N_TRAIN)).astype(int)
            train_ocean = data[train_idx].reshape(N_TRAIN, ny * nx)[:, ocean_idx].astype(np.float64)
            tm_p  = train_ocean.mean(axis=0)
            ts_p  = train_ocean.std(axis=0)
            zs_p  = ts_p < 1e-10
            ts_p  = np.where(zs_p, 1.0, ts_p)
            ls_p = 0.15; var_p = 1.0; noise_p = 0.01
            k_eff_g = min(VIZ_K_KRIG, n_ocean)
            sens_g  = _rpcholesky_sensors(X_all_p, ls_p, k_eff_g, mgp_p.matern32)
            sens_g_full = ocean_idx[sens_g]
            X_s_g   = X_all_p[sens_g]
            K_Xs_g  = mgp_p.matern32(X_all_p, X_s_g, ls_p)
            K_ss_g  = mgp_p.matern32(X_s_g, X_s_g, ls_p) + 1e-6 * np.eye(k_eff_g)
            K_sub_g = var_p * K_ss_g + noise_p * np.eye(k_eff_g)
            L_g, lo_g = cho_factor(K_sub_g, lower=True)
            snap_ocean_g = snap.ravel()[ocean_idx].astype(np.float64)
            sv_g    = snap_ocean_g[sens_g]
            y_n_g   = (sv_g - tm_p[sens_g]) / ts_p[sens_g]
            alp_g   = var_p * cho_solve((L_g, lo_g), y_n_g)
            mu_g    = (K_Xs_g @ alp_g) * ts_p + tm_p
            mu_g[zs_p] = tm_p[zs_p]
            recon_g = np.zeros(ny * nx, dtype=np.float32)
            recon_g[ocean_idx] = mu_g.astype(np.float32)
            recon_g = recon_g.reshape(ny, nx)
            pred_g  = recon_g.copy()
            resid_g_oc = snap_ocean_g.astype(np.float32) - mu_g.astype(np.float32)
            bins_, op_, ov_ = quantize(resid_g_oc, VIZ_AB)
            corr_g_oc = dequantize(bins_, op_, ov_, VIZ_AB, (n_ocean,)).astype(np.float32)
            recon_g.ravel()[ocean_idx] += corr_g_oc

        psnr_g   = ocean_metrics(snap, recon_g, ocean_mask)[0]
        row_g    = _csv_row("Kriging-2D", VIZ_K_KRIG, VIZ_AB)
        cr_g_str = f"{row_g['cr']:.1f}×" if row_g and np.isfinite(row_g.get("cr", np.nan)) else "–"
        _show(ax10, recon_g,
              f"Kriging-2D  k={k_eff_g},  ε={VIZ_AB:.0e}\nCR={cr_g_str}   PSNR={psnr_g:.1f} dB",
              sens_g_full)
    except Exception as e:
        ax10.set_title("Kriging-2D (error)", fontsize=14); ax10.axis("off")
        print(f"  [field panel] Kriging-2D error: {e}")

    # ── Residual histogram  (1,1) — ocean pixels only ────────────────────────
    # Shows the model's raw prediction error (before quantized correction).
    # Bin width is derived from the actual residual scale, NOT the quantizer
    # bin width (which is ~3e-7 °C and would cause everything to clip to ±49).
    ax11 = fig.add_subplot(gs[1, 1])

    snap_ocean_h = snap[ocean_mask].astype(np.float64)  # (n_ocean,) true SST
    all_res = []
    for method, pred_h in [
        ("DEIM-2D",    pred_d),
        ("Kriging-2D", pred_g),
    ]:
        try:
            if pred_h is None:
                raise ValueError("prediction not available")
            res_h = snap_ocean_h - pred_h[ocean_mask].astype(np.float64)
            all_res.append(res_h)
        except Exception:
            all_res.append(None)

    # Compute a shared x-range from all finite residuals (99th-percentile clip)
    combined = np.concatenate([r for r in all_res if r is not None])
    lo = float(np.percentile(combined, 0.5))
    hi = float(np.percentile(combined, 99.5))
    # Expand symmetrically around zero so the plot is centred
    bound = max(abs(lo), abs(hi))
    N_BINS_HIST = 100

    for (method, _), res_h in zip(
        [("DEIM-2D", pred_d), ("Kriging-2D", pred_g)], all_res
    ):
        if res_h is None:
            continue
        counts, edges = np.histogram(res_h, bins=N_BINS_HIST, range=(-bound, bound))
        centres = 0.5 * (edges[:-1] + edges[1:])
        bw      = edges[1] - edges[0]
        ax11.bar(centres, counts, width=bw * 0.9,
                 color=COLORS[method], alpha=0.50, label=method, edgecolor="none")

    ax11.axvline(0, color="k", lw=1.0, linestyle="--", alpha=0.5)
    ax11.set_xlim(-bound, bound)
    ax11.set_xlabel("Prediction error (°C)  [true − predicted]", fontsize=14)
    ax11.set_ylabel("Count", fontsize=14)
    ax11.set_title(f"Prediction residuals before correction\n"
                   f"k={VIZ_K_DEIM}/{VIZ_K_KRIG}  (DEIM/Kriging)", fontsize=16)
    ax11.legend(fontsize=14, loc="upper right", framealpha=0.85)
    ax11.tick_params(labelsize=13)

    out = ARGONNE / f"poster_field_panel_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"SST RD Comparison  |  compress backend: {COMPRESS_BACKEND}")
    print(f"Data: {DATA_PATH}\n")

    data, lat, lon, ocean_mask = load_sst()
    n_T, ny, nx = data.shape
    print(f"Loaded SST: shape={data.shape}  dtype={data.dtype}  "
          f"range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}] °C")

    # Apply land mask: set land pixels to 0 so residuals there are identically
    # zero and no bits are wasted on them.  All compression methods receive
    # data_m.  Metrics are computed over ocean pixels only via ocean_metrics().
    data_m = data.copy()
    data_m[:, ~ocean_mask] = 0.0
    n_ocean = int(ocean_mask.sum())
    print(f"Masked data_m: {n_ocean} ocean pixels active, "
          f"{int((~ocean_mask).sum())} land pixels zeroed")

    all_results = []

    if not PLOTS_ONLY:
        if RUN_LIBPRESSIO:
            try:
                import libpressio   # noqa: F401
                print("\n── SZ2 ──────────────────────────────────────────────")
                all_results += run_sz2(data_m, ocean_mask)
                if RUN_ZFP:
                    print("\n── ZFP ──────────────────────────────────────────────")
                    all_results += run_zfp(data_m, ocean_mask)
                else:
                    print("\n── ZFP skipped (RUN_ZFP=False) ─────────────────────")
            except ImportError:
                print("  libpressio not found — skipping SZ2/ZFP")

        print("\n── DEIM-2D ──────────────────────────────────────────")
        deim_results, deim_panel = run_deim_2d(data_m, ocean_mask)
        all_results += deim_results

        print("\n── Kriging-2D ───────────────────────────────────────")
        krig_results, krig_panel = run_kriging_2d(data_m, lat, lon, ocean_mask)
        all_results += krig_results

        save_csv(all_results)
    else:
        deim_panel = None
        krig_panel = None

    print("\nLoading CSV for plotting ...")
    by_method = load_csv()

    print("\n── Plots ────────────────────────────────────────────────")
    make_rd_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_cr_{FIELD_TAG}.png")
    make_bpv_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_bpv_{FIELD_TAG}.png")
    plot_rd_poster(by_method, "psnr", "PSNR (dB)", "psnr")
    make_budget_breakdown(by_method, f"rd_budget_breakdown_{FIELD_TAG}.png")
    make_timing_plot(by_method, f"rd_timing_{FIELD_TAG}.png")

    print("\n── Poster field panel ───────────────────────────────────")
    plot_field_panel(data_m, lat, lon, by_method, ocean_mask,
                     deim_panel=deim_panel, krig_panel=krig_panel)

    print("\nAll done.")

if __name__ == "__main__":
    main()
