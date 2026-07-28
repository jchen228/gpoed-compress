"""
poster_plots.py
===============
High-resolution figures for posters and presentations.

Reads  rd_results_{FIELD_TAG}.csv  (produced by rate_distortion_comparison.py)
and the raw data files, re-runs each method's reconstruction at a selected
operating point, and saves publication-quality PNG files.

Outputs (all at DPI=300, saved to ARGONNE/):
  poster_reconstruction_{FIELD_TAG}.png   — original + each method side-by-side
                                            with error maps underneath
  poster_sensor_map_{FIELD_TAG}.png       — sensor locations per method
  poster_rd_psnr_{FIELD_TAG}.png          — clean PSNR vs CR curve for poster
  poster_rd_ssim_{FIELD_TAG}.png          — clean SSIM vs CR curve for poster

Usage:
  python poster_plots.py            # uses TARGET_CR defined below
  TARGET_CR=10 python poster_plots.py
"""

import csv, importlib.util, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import qr as scipy_qr, cho_solve
from skimage.metrics import structural_similarity
import libpressio

# ── Config ────────────────────────────────────────────────────────────────────
ARGONNE    = Path(__file__).parent
DATA_DIR   = ARGONNE / "100x500x500"
DATA_PATH  = DATA_DIR / "CLOUDf48.bin.f32"
DATA_PATH2 = DATA_DIR / "QVAPORf48.bin.f32"
SHAPE      = (100, 500, 500)
FIELD_TAG  = DATA_PATH.stem.replace(".bin", "").replace(".f32", "")
LEVEL      = 50          # z-slice to visualise
DS         = 7           # spatial downsampling for all methods (~71×71/level)
NUM_BINS   = 65536       # must match main script
DPI        = 300

CMAP_FIELD = "RdBu_r"   # diverging: blue–white–red, matches pre-libpressio scripts

# Operating-point selection
# For the reconstruction comparison we pick, per method, the CSV row whose
# CR is closest to TARGET_CR (from above).  Adjust as needed.
TARGET_CR  = 5.0

COLORS  = {"SZ2":        "#1f77b4",
           "ZFP":        "#ff7f0e",
           "T-DEIM":     "#2ca02c",
           "DEIM-2D":    "#98df8a",
           "Kriging-2D": "#9467bd",
           "MultiGP":    "#d62728"}
MARKERS = {"SZ2": "o", "ZFP": "s", "T-DEIM": "^",
           "DEIM-2D": "v", "Kriging-2D": "P", "MultiGP": "D"}

METHOD_ORDER = ["SZ2", "ZFP", "DEIM-2D", "T-DEIM", "Kriging-2D", "MultiGP"]

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":         18,
    "axes.titlesize":    20,
    "axes.labelsize":    18,
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "legend.fontsize":   16,
    "legend.title_fontsize": 16,
    "lines.linewidth":   2.5,
    "lines.markersize":  10,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Zstd / zlib ───────────────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    _cctx = _zstd.ZstdCompressor(level=3)
    def _compress(b): return _cctx.compress(b)
except ImportError:
    import zlib
    def _compress(b): return zlib.compress(b, 6)

# ── Shared helpers (duplicated from rate_distortion_comparison.py) ────────────
def quantize(arr, abs_bound, num_bins=NUM_BINS):
    bw   = 2.0 * abs_bound / num_bins
    flat = arr.ravel().astype(np.float64)
    raw  = np.round(flat / bw).astype(np.int32)
    half = num_bins // 2
    mask = np.abs(raw) >= half
    bins = np.where(mask, 0, np.clip(raw, -(half-1), half-1)).astype(np.int16)
    return bins, np.where(mask)[0].astype(np.int32), flat[mask].astype(np.float32)

def dequantize(bins, out_pos, out_vals, abs_bound, shape, num_bins=NUM_BINS):
    bw  = 2.0 * abs_bound / num_bins
    arr = bins.astype(np.float64) * bw
    if len(out_pos):
        arr[out_pos] = out_vals.astype(np.float64)
    return arr.reshape(shape)

def compute_psnr(orig, recon):
    o, r = orig.astype(np.float64), recon.astype(np.float64)
    dr   = float(o.max() - o.min())
    rmse = float(np.sqrt(np.mean((o - r)**2)))
    return 20.0 * np.log10(dr / rmse) if rmse > 0 else float("inf")

def compute_ssim(orig, recon):
    dr = float(orig.max() - orig.min())
    return float(structural_similarity(
        orig.astype(np.float64), recon.astype(np.float64), data_range=dr))

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Data & CSV loaders ────────────────────────────────────────────────────────
def load_data():
    print("Loading data ...")
    data  = np.fromfile(DATA_PATH,  dtype=np.float32).reshape(SHAPE)
    data2 = np.fromfile(DATA_PATH2, dtype=np.float32).reshape(SHAPE) \
            if DATA_PATH2.exists() else None
    ds  = np.ascontiguousarray(data[:, ::DS, ::DS])
    ds2 = np.ascontiguousarray(data2[:, ::DS, ::DS]) if data2 is not None else None
    return data, data2, ds, ds2

def load_csv():
    csv_path = ARGONNE / f"rd_results_{FIELD_TAG}.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}\nRun rate_distortion_comparison.py first.")
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for fld in ("k", "n_outliers"):
                try:   row[fld] = int(float(row[fld])) if row[fld] not in ("","None") else None
                except: row[fld] = None
            for fld in ("abs_bound","cr","cr_no_model","psnr","ssim",
                        "compressed_MB","model_MB","sv_MB","resid_MB",
                        "comp_sec","decomp_sec","train_sec"):
                try:   row[fld] = float(row[fld]) if row[fld] not in ("","None") else float("nan")
                except: row[fld] = float("nan")
            rows.append(row)
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print(f"  Loaded {len(rows)} rows, {len(by_method)} methods.")
    return by_method

def best_row_near_cr(rows, target_cr):
    """Return the row whose CR is closest to target_cr."""
    valid = [r for r in rows if np.isfinite(r["cr"]) and np.isfinite(r["psnr"])]
    if not valid:
        return None
    return min(valid, key=lambda r: abs(r["cr"] - target_cr))

# ── Reconstruction functions (lightweight — no encoding, just predict+dequant) ─

def recon_sz(data_ds, abs_bound, compressor="sz"):
    """SZ2 or ZFP round-trip via libpressio."""
    cfg = {"pressio:abs": abs_bound} if compressor == "sz" \
          else {"zfp:accuracy": abs_bound}
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor,
        "early_config": {"pressio:metric": "composite",
                         "composite:plugins": ["size"]},
        "compressor_config": cfg,
    })
    buf = data_ds.copy()
    enc = comp.encode(data_ds)
    return comp.decode(enc, buf)

def recon_tdeim(data_ds, k, abs_bound):
    """T-DEIM: SVD → Q-DEIM sensors → solve → dequantize residual."""
    tdeim = load_mod(str(ARGONNE / "lp_tdeim_compressor.py"), "lp_tdeim")
    n_L, ny, nz = data_ds.shape
    n_2D = ny * nz
    mean_ds = data_ds.mean(axis=0)
    F = (data_ds - mean_ds).reshape(n_L, n_2D)
    k = min(k, n_L)
    Phi, _ = tdeim.build_3d_basis(F, k)
    Phi_k  = Phi[:, :k]
    _, _, p = scipy_qr(Phi_k.T, pivoting=True)
    sensors = np.sort(p[:k])
    y_s = F.ravel()[sensors].astype(np.float32)
    recon_flat = tdeim.tdeim_reconstruct(Phi_k, sensors, y_s.astype(np.float64))
    recon_ds   = recon_flat.reshape(n_L, ny, nz) + mean_ds
    resid      = (data_ds - recon_ds).ravel().astype(np.float32)
    bins, op, ov = quantize(resid, abs_bound, NUM_BINS)
    resid_rec  = dequantize(bins, op, ov, abs_bound, (n_L, ny, nz), NUM_BINS)
    return recon_ds + resid_rec, sensors, resid   # resid = pre-quant spatial error

def recon_deim2d(data_ds, k, abs_bound):
    """DEIM-2D: per-level SVD → Q-DEIM → solve → dequantize residual."""
    n_L, ny, nz = data_ds.shape
    n_2D = ny * nz
    mean_ds  = data_ds.mean(axis=0)
    F        = (data_ds - mean_ds).reshape(n_L, n_2D)
    k_max    = min(k, min(n_L, n_2D))
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    Phi_k    = Vt[:k_max, :].T
    _, _, p  = scipy_qr(Phi_k.T, pivoting=True)
    sensors  = p[:k_max]
    A        = Phi_k[sensors, :]
    all_sv   = F[:, sensors].astype(np.float32)
    recon_flat = np.zeros((n_L, n_2D), dtype=np.float64)
    for lvl in range(n_L):
        c = np.linalg.solve(A, all_sv[lvl])
        recon_flat[lvl] = Phi_k @ c
    recon_ds = recon_flat.reshape(n_L, ny, nz) + mean_ds
    resid    = (data_ds - recon_ds).ravel().astype(np.float32)
    bins, op, ov = quantize(resid, abs_bound, NUM_BINS)
    resid_rec = dequantize(bins, op, ov, abs_bound, (n_L, ny, nz), NUM_BINS)
    return recon_ds + resid_rec, sensors, resid   # resid = pre-quant spatial error

def _rpcholesky_sensors(X, ls, k, kern_fn, rank=None, rng=None):
    """
    Sensor selection via Randomly Pivoted Cholesky + RPGKS.

    Adapted from gpoed-code-python/pivoted_cholesky.py + rpgks.py.
    Builds low-rank factor F (n × rank) column-by-column without forming
    the full n×n kernel matrix → O(n × rank) memory and time.

    With rank = k+20 this enables DS=1 (n≈250K): F is ~1 GB vs 500 GB for K.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n    = len(X)
    rank = min(rank if rank is not None else k + 20, n)
    k    = min(k, n)

    # ── Step 1: Randomly Pivoted Cholesky ──────────────────────────────────
    diags = np.ones(n, dtype=np.float64)   # K[i,i] = 1 for unit-var Matérn
    F     = np.zeros((n, rank), dtype=np.float64)
    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total <= 0:
            actual_rank = i;  F = F[:, :i];  break
        si  = int(rng.choice(n, p=diags / total))
        col = kern_fn(X, X[[si]], ls).ravel()
        if i > 0:
            col = col - F[:, :i] @ F[si, :i]
        pv = float(col[si])
        if pv <= 0:
            actual_rank = i;  F = F[:, :i];  break
        F[:, i] = col / np.sqrt(pv)
        diags = np.maximum(diags - F[:, i] ** 2, 0.0)

    if actual_rank < k:
        return np.arange(k, dtype=np.int32)   # degenerate fallback

    # ── Step 2: RPGKS via F^T F + eigh (faster than full SVD for large n) ─
    G    = F.T @ F                        # (actual_rank, actual_rank)
    _, V = np.linalg.eigh(G)             # ascending order
    u_k  = F @ V[:, -k:]                 # (n, k) top-k left singular vectors
    norms = np.linalg.norm(u_k, axis=0, keepdims=True)
    u_k  /= np.where(norms > 1e-12, norms, 1.0)
    _, _, p = scipy_qr(u_k.T, pivoting=True)
    return p[:k].astype(np.int32)


def recon_kriging2d(data_ds, k, abs_bound):
    """Kriging-2D (d=1): fast version — joint ls+var+noise HP fitting, greedy sensors."""
    import time
    from scipy.optimize import minimize as _minimize_k
    from scipy.spatial.distance import cdist as _cdist_k
    mgp          = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
    n_L, ny, nz  = data_ds.shape
    n            = ny * nz
    X_all        = mgp.make_grid_coords(ny, nz)

    print(f"    [Kriging] n={n}  k={k}  normalising ...", flush=True)
    data_nd    = data_ds.reshape(n_L, n, 1)
    train_mean = data_nd.mean(axis=0)
    train_std  = data_nd.std(axis=0)
    zero_std_mask = (train_std[:, 0] < 1e-10)   # locations with zero variance (e.g. cloud=0)
    train_std_safe = np.where(train_std < 1e-10, 1.0, train_std)
    Y_train    = [(data_nd[l] - train_mean) / train_std_safe for l in range(n_L)]

    # Joint ls+var+noise fit — mirrors kriging_hurricane.py fit_hyperparams
    fit_size = min(k * 4, n)
    rng_fit  = np.random.default_rng(0)
    fit_idx  = rng_fit.choice(n, size=fit_size, replace=False)
    X_fit    = X_all[fit_idx]
    Y_fit    = Y_train[0][fit_idx, 0]   # level 0, dim 0

    def _neg_lml(log_theta):
        ls_, var_, nv_ = (float(np.exp(log_theta[0])),
                          float(np.exp(log_theta[1])),
                          float(np.exp(log_theta[2])))
        K = var_ * mgp.matern32(X_fit, X_fit, ls_) + nv_ * np.eye(fit_size)
        try:
            L_ = np.linalg.cholesky(K + 1e-8 * np.eye(fit_size))
        except np.linalg.LinAlgError:
            return 1e10
        ld    = 2.0 * np.sum(np.log(np.diag(L_)))
        alpha = np.linalg.solve(L_.T, np.linalg.solve(L_, Y_fit))
        return 0.5 * float(Y_fit @ alpha) + 0.5 * ld

    dists  = _cdist_k(X_fit[:min(10, fit_size)], X_fit[:min(10, fit_size)], 'euclidean')
    ls0    = float(np.median(dists[dists > 0])) / 3.0 if np.any(dists > 0) else 1.0
    var0   = max(float(np.var(Y_fit)), 1e-6)
    noise0 = var0 * 0.01
    bounds = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
    rng_hp = np.random.default_rng(0)
    starts = [np.log([max(ls0, 0.1), max(var0, 0.01), max(noise0, 1e-6)])] + [
        rng_hp.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        for _ in range(2)
    ]
    t0 = time.perf_counter()
    print(f"    [Kriging] fitting ls+var+noise on {fit_size}-pt random subset ...", flush=True)
    best_nll, best_ls, best_var, best_nv = np.inf, ls0, var0, noise0
    for x0 in starts:
        try:
            res = _minimize_k(_neg_lml, x0, method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls  = float(np.exp(res.x[0]))
                best_var = float(np.exp(res.x[1]))
                best_nv  = float(np.exp(res.x[2]))
        except Exception:
            pass
    ls        = best_ls
    noise_var = best_nv
    B         = np.array([[best_var]])
    print(f"    [Kriging] ls={ls:.3f}  var={best_var:.4f}  noise={noise_var:.2e}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"    [Kriging] RPCholesky sensor selection (rank={k+20}, n={n}) ...", flush=True)
    t0 = time.perf_counter()
    sensors = _rpcholesky_sensors(X_all, ls, k, mgp.matern32, rank=k + 20)
    print(f"    [Kriging] sensors done ({time.perf_counter()-t0:.1f}s)", flush=True)
    X_sensors = X_all[sensors]
    print(f"    [Kriging] sensors placed; reconstructing {n_L} levels ...", flush=True)

    recon_flat = np.zeros((n_L, n), dtype=np.float64)
    for lvl in range(n_L):
        Y_obs_norm      = Y_train[lvl][sensors]
        mu_norm, _      = mgp.lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var)
        recon_flat[lvl] = (mu_norm * train_std_safe + train_mean)[:, 0]
        # Zero-variance locations: GP z-score is meaningless — just predict the mean
        recon_flat[lvl][zero_std_mask] = train_mean[zero_std_mask, 0]
    print(f"    [Kriging] reconstruction done.", flush=True)

    recon_ds  = recon_flat.reshape(n_L, ny, nz)
    resid     = (data_ds - recon_ds).ravel().astype(np.float32)
    bins, op, ov = quantize(resid, abs_bound)
    resid_rec = dequantize(bins, op, ov, abs_bound, (n_L, ny, nz))
    return recon_ds + resid_rec, sensors, resid, ls   # 4th: cached ls for histogram reuse

def recon_multigp(data_ds, data2_ds, k, abs_bound):
    """MultiGP (d=2): fast version — joint ls+var+noise HP fitting, greedy sensors."""
    import time
    from scipy.optimize import minimize as _minimize_m
    from scipy.spatial.distance import cdist as _cdist_m
    mgp          = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
    d            = 2
    n_L, ny, nz  = data_ds.shape
    n            = ny * nz
    X_all        = mgp.make_grid_coords(ny, nz)

    print(f"    [MultiGP] n={n}  k={k}  normalising ...", flush=True)
    data_nd    = np.stack([data_ds.reshape(n_L, n),
                           data2_ds.reshape(n_L, n)], axis=-1)
    train_mean = data_nd.mean(axis=0)
    train_std  = data_nd.std(axis=0)
    zero_std_mask_m = (train_std < 1e-10).any(axis=-1)   # (n,) zero-variance in any variable
    train_std_safe  = np.where(train_std < 1e-10, 1.0, train_std)
    Y_train    = [(data_nd[l] - train_mean) / train_std_safe for l in range(n_L)]

    # estimate_B captures cross-variable structure needed for d=2 lmc_predict
    print(f"    [MultiGP] estimating B ...", flush=True)
    B = mgp.estimate_B(Y_train)

    # Joint ls+var+noise fit on variable 0 — mirrors kriging_hurricane.py fit_hyperparams
    fit_size = min(k * 4, n)
    rng_fit  = np.random.default_rng(0)
    fit_idx  = rng_fit.choice(n, size=fit_size, replace=False)
    X_fit    = X_all[fit_idx]
    Y_fit    = Y_train[0][fit_idx, 0]   # level 0, variable 0

    def _neg_lml(log_theta):
        ls_, var_, nv_ = (float(np.exp(log_theta[0])),
                          float(np.exp(log_theta[1])),
                          float(np.exp(log_theta[2])))
        K = var_ * mgp.matern32(X_fit, X_fit, ls_) + nv_ * np.eye(fit_size)
        try:
            L_ = np.linalg.cholesky(K + 1e-8 * np.eye(fit_size))
        except np.linalg.LinAlgError:
            return 1e10
        ld    = 2.0 * np.sum(np.log(np.diag(L_)))
        alpha = np.linalg.solve(L_.T, np.linalg.solve(L_, Y_fit))
        return 0.5 * float(Y_fit @ alpha) + 0.5 * ld

    dists  = _cdist_m(X_fit[:min(10, fit_size)], X_fit[:min(10, fit_size)], 'euclidean')
    ls0    = float(np.median(dists[dists > 0])) / 3.0 if np.any(dists > 0) else 1.0
    var0   = max(float(np.var(Y_fit)), 1e-6)
    noise0 = var0 * 0.01
    bounds = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
    rng_hp = np.random.default_rng(0)
    starts = [np.log([max(ls0, 0.1), max(var0, 0.01), max(noise0, 1e-6)])] + [
        rng_hp.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        for _ in range(2)
    ]
    t0 = time.perf_counter()
    print(f"    [MultiGP] fitting ls+var+noise on {fit_size}-pt random subset ...", flush=True)
    best_nll, best_ls, best_nv = np.inf, ls0, noise0
    for x0 in starts:
        try:
            res = _minimize_m(_neg_lml, x0, method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls  = float(np.exp(res.x[0]))
                best_nv  = float(np.exp(res.x[2]))
        except Exception:
            pass
    ls        = best_ls
    noise_var = best_nv
    print(f"    [MultiGP] ls={ls:.3f}  noise={noise_var:.2e}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"    [MultiGP] RPCholesky sensor selection (rank={k+20}, n={n}) ...", flush=True)
    t0 = time.perf_counter()
    sensors = _rpcholesky_sensors(X_all, ls, k, mgp.matern32, rank=k + 20)
    print(f"    [MultiGP] sensors done ({time.perf_counter()-t0:.1f}s)", flush=True)
    X_sensors = X_all[sensors]
    print(f"    [MultiGP] sensors placed; reconstructing {n_L} levels ...", flush=True)

    recon_nd = np.zeros((n_L, n, d), dtype=np.float64)
    for lvl in range(n_L):
        Y_obs_norm    = Y_train[lvl][sensors]
        mu_norm, _    = mgp.lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var)
        recon_nd[lvl] = mu_norm * train_std_safe + train_mean
        # Zero-variance locations: just predict the mean (GP z-score is meaningless)
        recon_nd[lvl][zero_std_mask_m] = train_mean[zero_std_mask_m]
    print(f"    [MultiGP] reconstruction done.", flush=True)

    recon_ds0 = recon_nd[:, :, 0].reshape(n_L, ny, nz)
    resid     = (data_ds - recon_ds0).ravel().astype(np.float32)
    bins, op, ov = quantize(resid, abs_bound)
    resid_rec = dequantize(bins, op, ov, abs_bound, (n_L, ny, nz))
    return recon_ds0 + resid_rec, sensors, resid, ls   # 4th: cached ls for histogram reuse

# ── Plot 1: Reconstruction comparison grid ────────────────────────────────────
def plot_reconstruction_comparison(data_ds, data2_ds, by_method, target_cr=TARGET_CR):
    """
    Top row:    True field  |  SZ2  |  ZFP  |  DEIM-2D  |  T-DEIM  |  Kriging-2D  |  MultiGP
    Bottom row: (blank)     | error maps for each method
    """
    methods_to_show = [m for m in METHOD_ORDER if m in by_method]
    n_methods = len(methods_to_show)
    n_cols    = 1 + n_methods   # +1 for original
    true_slice = data_ds[LEVEL].astype(np.float32)
    vmin, vmax = float(true_slice.min()), float(true_slice.max())

    print(f"\n[Reconstruction] target CR={target_cr:.1f}× — re-running predictions ...")
    recons  = {}
    sensors = {}
    params  = {}

    for method in methods_to_show:
        row = best_row_near_cr(by_method[method], target_cr)
        if row is None:
            print(f"  {method}: no valid rows — skipped")
            continue
        ab = row["abs_bound"]
        k  = row.get("k")
        params[method] = row
        print(f"  {method}: abs_bound={ab:.2g}  k={k}  CR={row['cr']:.1f}×  PSNR={row['psnr']:.1f} dB")
        try:
            if method == "SZ2":
                recons[method] = recon_sz(data_ds, ab, "sz")[LEVEL]
            elif method == "ZFP":
                recons[method] = recon_sz(data_ds, ab, "zfp")[LEVEL]
            elif method == "T-DEIM":
                r, s, _ = recon_tdeim(data_ds, k, ab)
                recons[method], sensors[method] = r[LEVEL], s
            elif method == "DEIM-2D":
                r, s, _ = recon_deim2d(data_ds, k, ab)
                recons[method], sensors[method] = r[LEVEL], s
            elif method == "Kriging-2D":
                r, s, _ = recon_kriging2d(data_ds, k, ab)
                recons[method], sensors[method] = r[LEVEL], s
            elif method == "MultiGP" and data2_ds is not None:
                r, s, _ = recon_multigp(data_ds, data2_ds, k, ab)
                recons[method], sensors[method] = r[LEVEL], s
        except Exception as e:
            print(f"  {method} reconstruction failed: {e}")

    shown = [m for m in methods_to_show if m in recons]
    n_shown = len(shown)
    n_cols_actual = 1 + n_shown

    fig = plt.figure(figsize=(3.5 * n_cols_actual, 7.5))
    gs  = gridspec.GridSpec(2, n_cols_actual, hspace=0.08, wspace=0.05)

    # ── True field ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(true_slice, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax.set_title("Original", fontweight="bold", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="g kg⁻¹")

    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.axis("off")

    # ── Each method ───────────────────────────────────────────────────────────
    # Determine common error color scale (95th percentile across all methods)
    all_errors = [np.abs(true_slice - recons[m].astype(np.float32)) for m in shown]
    emax = float(np.percentile(np.concatenate([e.ravel() for e in all_errors]), 97)) \
           if all_errors else 1.0

    for col_idx, method in enumerate(shown, start=1):
        recon_slice = recons[method].astype(np.float32)
        err_slice   = np.abs(true_slice - recon_slice)
        p           = params[method]
        psnr_val    = compute_psnr(true_slice, recon_slice)
        ssim_val    = compute_ssim(true_slice, recon_slice)

        # Reconstruction
        ax_r = fig.add_subplot(gs[0, col_idx])
        ax_r.imshow(recon_slice, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        k_str = f", k={p['k']}" if p.get("k") else ""
        ax_r.set_title(f"{method}\nCR={p['cr']:.1f}×{k_str}\nPSNR={psnr_val:.1f} dB  SSIM={ssim_val:.3f}",
                        fontsize=10, fontweight="bold" if method not in ("SZ2","ZFP") else "normal")
        ax_r.axis("off")

        # Error map
        ax_e = fig.add_subplot(gs[1, col_idx])
        im_e = ax_e.imshow(err_slice, vmin=0, vmax=emax, cmap="hot", origin="lower")
        ax_e.set_title(f"error", fontsize=9, color="#555")
        ax_e.axis("off")
        if col_idx == n_shown:
            plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04, label="|error|")

    fig.suptitle(
        f"Reconstruction comparison at matched CR ≈ {target_cr:.0f}×  |  "
        f"CLOUDf48  level {LEVEL}  (downsampled {DS}×)",
        fontsize=13, y=1.01)

    out = ARGONNE / f"poster_reconstruction_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return sensors   # return for sensor map plot

# ── Plot 2: Sensor location maps ──────────────────────────────────────────────
def plot_sensor_maps(data_ds, sensors_by_method, by_method, target_cr=TARGET_CR):
    """Show the k sensor locations overlaid on the true field for each GP/DEIM method."""
    our_methods = [m for m in ["T-DEIM", "DEIM-2D", "Kriging-2D", "MultiGP"]
                   if m in sensors_by_method]
    if not our_methods:
        print("No sensor maps to plot (no sensor data).")
        return

    n_L, ny, nz = data_ds.shape
    true_slice   = data_ds[LEVEL].astype(np.float32)
    n_cols = len(our_methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # Grid coordinates (matching downsampled grid: ny×nz)
    ys = np.arange(ny)
    xs = np.arange(nz)
    XX, YY = np.meshgrid(xs, ys)

    for ax, method in zip(axes, our_methods):
        s_idx = sensors_by_method[method]   # flat indices into n_2D or n_3D_ds
        row = best_row_near_cr(by_method[method], target_cr)
        k_actual = len(s_idx)

        ax.imshow(true_slice, cmap="viridis", origin="lower",
                  extent=[0, nz, 0, ny], aspect="equal", alpha=0.85)

        # For 3D sensors (T-DEIM), take only those in LEVEL's spatial block
        if method == "T-DEIM":
            level_start = LEVEL * (ny * nz)
            level_end   = level_start + ny * nz
            local = s_idx[(s_idx >= level_start) & (s_idx < level_end)] - level_start
            sy = local // nz
            sx = local %  nz
        else:
            sy = s_idx // nz
            sx = s_idx %  nz

        ax.scatter(sx + 0.5, sy + 0.5, c="red", s=max(2, 200 // (k_actual + 1)),
                   marker="x", linewidths=1.5, label=f"k={k_actual} sensors", zorder=5)
        cr_str = f"CR≈{row['cr']:.1f}×" if row else ""
        ax.set_title(f"{method}  {cr_str}\n{k_actual} sensor locations", fontsize=12)
        ax.set_xlabel("x (downsampled)")
        ax.set_ylabel("y (downsampled)")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Sensor placement — CLOUDf48 level {LEVEL}", fontsize=13)
    plt.tight_layout()
    out = ARGONNE / f"poster_sensor_map_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Plot 3: Clean RD curves ───────────────────────────────────────────────────
def _pareto_upper(pts, metric_key):
    """Upper Pareto envelope: best metric at each CR, monotone in both."""
    srt  = sorted(pts, key=lambda p: p["cr"])
    best_val, env = -np.inf, []
    for p in srt:
        v = p[metric_key]
        if np.isfinite(v) and v > best_val:
            best_val = v
            env.append(p)
    return env

def plot_rd_curves(by_method, metric_key, ylabel, fname_suffix):
    fig, ax = plt.subplots(figsize=(8, 6.5))   # taller to accommodate above-legend

    def _plot_method_on(target_ax, method, pts, lw=2.2, ms=10, annotate=False):
        """Plot one method's RD curve on target_ax."""
        c, mk = COLORS[method], MARKERS[method]
        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: x["cr"])
            crs  = [p["cr"]       for p in srt]
            vals = [p[metric_key] for p in srt]
            target_ax.plot(crs, vals, "-", color=c, marker=mk, lw=lw, ms=ms)
            if annotate and srt:
                target_ax.annotate(f"ab={srt[0]['abs_bound']:.1g}",
                                   (crs[0], vals[0]), fontsize=8, color=c,
                                   xytext=(5, 3), textcoords="offset points")
        else:
            env  = _pareto_upper(pts, metric_key)
            crs  = [p["cr"]       for p in env]
            vals = [p[metric_key] for p in env]
            target_ax.plot(crs, vals, "--", color=c, marker=mk, lw=lw, ms=ms)
            if annotate and env:
                target_ax.annotate(f"k={env[0]['k']}",
                                   (crs[0], vals[0]), fontsize=14, color=c,
                                   xytext=(5, 3), textcoords="offset points")

    # ── Main axes ─────────────────────────────────────────────────────────────
    all_pts = {}
    for method in METHOD_ORDER:
        pts = [p for p in by_method.get(method, [])
               if np.isfinite(p.get("cr", np.nan)) and np.isfinite(p.get(metric_key, np.nan))]
        if not pts:
            continue
        all_pts[method] = pts
        c, mk = COLORS[method], MARKERS[method]

        # Build legend label with parameter range for all methods
        if method in ("SZ2", "ZFP"):
            ab_vals = sorted(set(p["abs_bound"] for p in pts if p.get("abs_bound") is not None))
            if ab_vals:
                label = method + f" (ε={ab_vals[0]:.0e}–{ab_vals[-1]:.0e})"
            else:
                label = method
        else:
            k_vals = sorted(set(p["k"] for p in pts if p.get("k") is not None))
            label  = method + (f" (k={k_vals[0]}–{k_vals[-1]})" if k_vals else "")

        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: x["cr"])
            crs  = [p["cr"]       for p in srt]
            vals = [p[metric_key] for p in srt]
            ax.plot(crs, vals, "-", color=c, marker=mk, lw=2.2, label=label)
            if srt:
                ax.annotate(f"ab={srt[0]['abs_bound']:.1g}", (crs[0], vals[0]),
                            fontsize=8, color=c, xytext=(5, 3), textcoords="offset points")
        else:
            env  = _pareto_upper(pts, metric_key)
            crs  = [p["cr"]       for p in env]
            vals = [p[metric_key] for p in env]
            ax.plot(crs, vals, "--", color=c, marker=mk, lw=2.2, label=label)
            if env:
                ax.annotate(f"k={env[0]['k']}", (crs[0], vals[0]),
                            fontsize=14, color=c, xytext=(5, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    # Legend below the x-axis label
    ax.legend(fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.20),
              ncol=3, handlelength=1.8, columnspacing=1.0,
              framealpha=0.9, borderpad=0.6)
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_title(f"Rate–Distortion  —  CLOUDf48  (level {LEVEL})", fontsize=13)

    # ── Zoom inset: bottom-left corner with dashed rectangle on main plot ─────
    if metric_key == "psnr":
        axins = ax.inset_axes([0.65, 0.60, 0.32, 0.37])
        for method, pts in all_pts.items():
            _plot_method_on(axins, method, pts, lw=1.5, ms=5)
        axins.set_xlim(0.05, 7)
        axins.set_ylim(106, 128)
        axins.set_xscale("log")
        axins.tick_params(labelsize=7, pad=1)
        axins.set_xlabel("CR", fontsize=7, labelpad=1)
        axins.set_ylabel("PSNR (dB)", fontsize=7, labelpad=1)
        axins.grid(True, which="both", alpha=0.25, linestyle="--")
        axins.set_title("zoom", fontsize=7, pad=2, color="#333")
        for spine in axins.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#555")
        # Dashed rectangle on main axes showing the zoomed region
        from matplotlib.patches import Rectangle
        rect = Rectangle((0.05, 106), 7 - 0.05, 128 - 106,
                          linewidth=1.2, edgecolor="#555",
                          facecolor="none", linestyle="--", zorder=5)
        ax.add_patch(rect)

    out = ARGONNE / f"poster_rd_{fname_suffix}_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Shared visual helpers ─────────────────────────────────────────────────────
def field_norm(slice_2d):
    """TwoSlopeNorm centred on the field midpoint for RdBu_r."""
    vmin = float(slice_2d.min())
    vmax = float(slice_2d.max())
    vcenter = (vmin + vmax) / 2.0
    return vmin, vmax, TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

def overlay_sensors(ax, sensors, ny, nz, method, k_shown=None):
    """Scatter sensor locations onto an existing imshow axes (pixel coords)."""
    if sensors is None or len(sensors) == 0:
        return
    if method == "T-DEIM":
        level_start = LEVEL * ny * nz
        level_end   = level_start + ny * nz
        local = sensors[(sensors >= level_start) & (sensors < level_end)] - level_start
    else:
        local = sensors
    sy = local // nz
    sx = local %  nz
    n  = k_shown if k_shown is not None else len(sy)
    ax.scatter(sx[:n] + 0.5, sy[:n] + 0.5,
               c="gold", s=max(4, 120 // (len(sy) + 1)),
               marker="x", linewidths=1.4, zorder=5, label=f"{len(sy)} sensors")

def overlay_sensors_norm(ax, sensors, ny, nz, method, k_shown=None):
    """Scatter sensor locations normalised to [0,1] extent (use with extent=[0,1,0,1])."""
    if sensors is None or len(sensors) == 0:
        return
    if method == "T-DEIM":
        level_start = LEVEL * ny * nz
        level_end   = level_start + ny * nz
        local = sensors[(sensors >= level_start) & (sensors < level_end)] - level_start
    else:
        local = sensors
    sy = local // nz
    sx = local %  nz
    n  = k_shown if k_shown is not None else len(sy)
    # +0.5 centres on pixel, then divide by grid size to get [0,1]
    ax.scatter((sx[:n] + 0.5) / nz, (sy[:n] + 0.5) / ny,
               c="gold", s=max(4, 120 // (len(sy) + 1)),
               marker="x", linewidths=1.4, zorder=5, label=f"{len(sy)} sensors")

# ── Helpers for best-PSNR operating point ────────────────────────────────────
def best_row_by_psnr(rows):
    """Return the row with the highest PSNR (lowest reconstruction error)."""
    valid = [r for r in rows if np.isfinite(r.get("psnr", float("nan")))]
    return max(valid, key=lambda r: r["psnr"]) if valid else None

def deim2d_predict(data_ds, k):
    """
    Run DEIM-2D prediction only — no quantization.
    Returns (pred_3d, resid_flat_f32, sensors).
    resid_flat_f32 is the pre-quantization residual, suitable for bin histograms.
    """
    n_L, ny, nz = data_ds.shape
    n_2D        = ny * nz
    mean_ds     = data_ds.mean(axis=0)
    F           = (data_ds - mean_ds).reshape(n_L, n_2D)
    k_eff       = min(k, min(n_L, n_2D))
    _, _, Vt    = np.linalg.svd(F, full_matrices=False)
    Phi_k       = Vt[:k_eff, :].T                        # (n_2D, k_eff)
    _, _, p     = scipy_qr(Phi_k.T, pivoting=True)
    sensors     = p[:k_eff]
    A           = Phi_k[sensors, :]                       # (k_eff, k_eff)
    all_sv      = F[:, sensors].astype(np.float32)
    # Vectorised reconstruction over all levels
    C           = np.linalg.solve(A, all_sv.T)           # (k_eff, n_L)
    recon_flat  = (Phi_k @ C).T                          # (n_L, n_2D)
    pred_3d     = recon_flat.reshape(n_L, ny, nz) + mean_ds
    resid       = (data_ds.astype(np.float32) - pred_3d.astype(np.float32)).ravel()
    return pred_3d, resid, sensors


# ── Figure 1: True | DEIM-2D+sensors | GP+sensors ────────────────────────────
def plot_fig1_triple(data_ds, data2_ds, by_method):
    """
    Three panels — sensor-map style (field as background, sensors overlaid):
      Panel 1 — True field at level LEVEL
      Panel 2 — DEIM-2D reconstruction + sensor locations (best PSNR)
      Panel 3 — Kriging-2D reconstruction + sensor locations (best PSNR)
    """
    true_slice = data_ds[LEVEL].astype(np.float32)
    vmin, vmax, norm = field_norm(true_slice)
    ny, nz = true_slice.shape

    results = {}
    for method, fn in [("DEIM-2D",    lambda r: recon_deim2d(data_ds,    r["k"], r["abs_bound"])),
                       ("Kriging-2D", lambda r: recon_kriging2d(data_ds,  r["k"], r["abs_bound"]))]:
        row = best_row_by_psnr(by_method.get(method, []))
        if row is None:
            print(f"  {method}: no valid CSV rows — skipped"); continue
        print(f"  {method}: k={row['k']}  ab={row['abs_bound']:.2g}  "
              f"CR={row['cr']:.1f}×  PSNR={row['psnr']:.1f} dB")
        try:
            recon_3d, sensors, _ = fn(row)
            results[method] = {"slice": recon_3d[LEVEL].astype(np.float32),
                               "row": row, "sensors": sensors}
        except Exception as e:
            print(f"  {method} FAILED: {e}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    # Panel 0 — true field
    im = axes[0].imshow(true_slice, cmap=CMAP_FIELD, norm=norm, origin="lower")
    axes[0].set_title("True field", fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="g kg⁻¹")

    # Panels 1–2 — reconstructions with sensor overlay
    for ax, method in zip(axes[1:], ["DEIM-2D", "Kriging-2D"]):
        if method not in results:
            ax.axis("off"); continue
        info      = results[method]
        rec       = info["slice"]
        row       = info["row"]
        sensors_m = info["sensors"]
        psnr_val  = compute_psnr(true_slice, rec)
        ssim_val  = compute_ssim(true_slice, rec)
        label     = "GP (Kriging-2D)" if method == "Kriging-2D" else "DEIM-2D"

        ax.imshow(rec, cmap=CMAP_FIELD, norm=norm, origin="lower")
        overlay_sensors(ax, sensors_m, ny, nz, method)
        ax.set_title(f"{label}\nk={row['k']},  ε={row['abs_bound']:.1e}\n"
                     f"CR={row['cr']:.1f}×   PSNR={psnr_val:.1f} dB   SSIM={ssim_val:.3f}")
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=12, markerscale=1.4,
                  framealpha=0.8, handletextpad=0.3)

    out = ARGONNE / f"poster_fig1_triple_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return results


# ── Figure 2: Quantization bin histogram (horizontal, zoomed) ────────────────
def plot_fig2_bin_histogram(data_ds, by_method, fig1_results=None):
    """
    Horizontal bar histogram of DEIM-2D quantization bin indices.
    Zoomed to the occupied bin range; dashed borders mark the full ±32767 extent.
    Bar height = 1 bin unit so the distribution shape is clear without over-thinning.
    """
    row = best_row_by_psnr(by_method.get("DEIM-2D", []))
    if row is None:
        print("  DEIM-2D: no valid rows for histogram — skipped"); return

    k, ab = row["k"], row["abs_bound"]
    print(f"  [Histogram] DEIM-2D  k={k}  ab={ab:.2g}")
    _, resid_flat, _ = deim2d_predict(data_ds, k)

    bins, outlier_pos, _ = quantize(resid_flat, ab, NUM_BINS)
    bins_i32 = bins.astype(np.int32)
    unique_bins, counts = np.unique(bins_i32, return_counts=True)

    # Zoom: show the range that contains 99.9 % of values
    valid = bins_i32[np.setdiff1d(np.arange(len(bins_i32)), outlier_pos)]
    zoom_lo = int(np.percentile(valid, 0.05))  - 2
    zoom_hi = int(np.percentile(valid, 99.95)) + 2

    mask   = (unique_bins >= zoom_lo) & (unique_bins <= zoom_hi)
    y_plot = unique_bins[mask]
    x_plot = counts[mask]

    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.barh(y_plot, x_plot, height=1.0, color=COLORS["DEIM-2D"], edgecolor="none", alpha=0.85)

    ax.set_xlabel("Count", fontsize=18)
    ax.set_ylabel("Bin index  (bin width = 2ε / 65536)", fontsize=16)
    ax.set_ylim(zoom_lo - 0.5, zoom_hi + 0.5)
    ax.set_title(f"DEIM-2D Residual Quantization\nk={k},  ε={ab:.1e}", fontsize=18)

    full_half = NUM_BINS // 2 - 1
    for y_bnd, va, lbl in [(zoom_hi + 0.5, "bottom", f"⋮  full range extends to +{full_half}"),
                            (zoom_lo - 0.5, "top",    f"full range extends to −{full_half}  ⋮")]:
        ax.axhline(y_bnd, color="#888", lw=1.2, ls="--", zorder=3)
        ax.text(x_plot.max() * 0.98, y_bnd, f"  {lbl}", va=va, ha="left",
                fontsize=13, color="#555", style="italic")

    n_out = len(outlier_pos)
    ax.text(0.97, 0.02,
            f"Outliers: {n_out}/{len(bins_i32)} ({100*n_out/len(bins_i32):.3f}%)\n"
            "stored as raw float32",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))

    out = ARGONNE / f"poster_fig2_bin_histogram_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: 2×3 big panel ───────────────────────────────────────────────────
def plot_fig3_big_panel(data_ds, data2_ds, by_method):
    """
    2 rows × 3 columns — all methods at DS=7 (71×71/level):
      Row 0: True data | Kriging-2D recon+sensors | MultiGP recon+sensors
      Row 1: DEIM-2D recon+sensors | T-DEIM recon+sensors | Combined bin histogram
    """
    true_slice = data_ds[LEVEL].astype(np.float32)
    ny, nz     = true_slice.shape
    vmin, vmax, norm = field_norm(true_slice)

    HIST_METHODS = ["DEIM-2D", "T-DEIM", "Kriging-2D", "MultiGP"]
    PANEL_METHODS = [
        ("Kriging-2D", data_ds,  None,     "GP (Kriging-2D)"),
        ("MultiGP",    data_ds,  data2_ds, "MultiGP"),
        ("DEIM-2D",    data_ds,  None,     "DEIM-2D"),
        ("T-DEIM",     data_ds,  None,     "T-DEIM"),
    ]

    # ── Run all reconstructions ───────────────────────────────────────────────
    results = {}
    for method, ds_use, ds2_use, label in PANEL_METHODS:
        row = best_row_by_psnr(by_method.get(method, []))
        if row is None:
            print(f"  {method}: no rows — skipped"); continue
        print(f"  {method}: k={row['k']}  ab={row['abs_bound']:.2g}  "
              f"CR={row['cr']:.1f}×  PSNR={row['psnr']:.1f} dB")
        try:
            if method == "Kriging-2D":
                r3d, sensors, pre_resid, gp_ls = recon_kriging2d(ds_use, row["k"], row["abs_bound"])
            elif method == "MultiGP":
                if data2_ds is None:
                    print("  MultiGP: data2 not found — skipped"); continue
                r3d, sensors, pre_resid, gp_ls = recon_multigp(ds_use, data2_ds, row["k"], row["abs_bound"])
            elif method == "DEIM-2D":
                r3d, sensors, pre_resid = recon_deim2d(ds_use, row["k"], row["abs_bound"])
                gp_ls = None
            elif method == "T-DEIM":
                r3d, sensors, pre_resid = recon_tdeim(ds_use, row["k"], row["abs_bound"])
                gp_ls = None
            resid_cached = (ds_use.astype(np.float32) - r3d.astype(np.float32)).ravel()
            results[method] = {
                "slice":     r3d[LEVEL].astype(np.float32),
                "sensors":   sensors,
                "row":       row,
                "label":     label,
                "ds":        ds_use,
                "resid":     resid_cached,
                "pre_resid": pre_resid,
                "gp_ls":     gp_ls,          # cached for histogram reuse (no refit)
            }
        except Exception as e:
            print(f"  {method} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Build 2×3 figure ─────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size':        22,
        'axes.labelsize':   24,
        'axes.titlesize':   26,
        'xtick.labelsize':  20,
        'ytick.labelsize':  20,
        'legend.fontsize':  20,
    })
    fig = plt.figure(figsize=(24, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.10)

    # Shared display extent — all panels cover the same spatial domain [0,1]×[0,1]
    # so imshow stretches every reconstruction to the same visual size regardless
    # of its native pixel count. bilinear interpolation smooths low-res panels.
    EXTENT = [0, 1, 0, 1]

    # (0,0) True field
    ax00 = fig.add_subplot(gs[0, 0])
    im   = ax00.imshow(true_slice, cmap=CMAP_FIELD, norm=norm, origin="lower")
    ax00.set_title("True field", fontweight="bold", fontsize=30)
    ax00.axis("off")
    plt.colorbar(im, ax=ax00, fraction=0.046, pad=0.04, label="g kg⁻¹")

    # Row 0 panels 1–2 and row 1 panels 0–1
    layout = [
        (0, 1, "Kriging-2D"),
        (0, 2, "MultiGP"),
        (1, 0, "DEIM-2D"),
        (1, 1, "T-DEIM"),
    ]
    for row_i, col_i, method in layout:
        ax = fig.add_subplot(gs[row_i, col_i])
        if method not in results:
            ax.axis("off"); continue
        info     = results[method]
        rec      = info["slice"]
        row_data = info["row"]
        sensors  = info["sensors"]
        label    = info["label"]
        ny_m, nz_m = rec.shape

        ax.imshow(rec, cmap=CMAP_FIELD, norm=norm, origin="lower")
        # Scale sensor coords to [0,1] to match the shared extent
        overlay_sensors(ax, sensors, ny_m, nz_m, method)
        ref = results[method]["ds"][LEVEL].astype(np.float32)
        psnr_val = compute_psnr(ref, rec)
        ssim_val = compute_ssim(ref, rec)
        ax.set_title(
            f"{label}\nk={row_data['k']},  ε={row_data['abs_bound']:.1e}\n"
            f"CR={row_data['cr']:.1f}×   PSNR={psnr_val:.1f} dB",
            fontsize=22)
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=18, framealpha=0.8, markerscale=1.4)

    # (1,2) Combined overlapping histogram — pre-quantization spatial prediction error
    # All methods evaluated at HIST_K=10 so residual scales are comparable across
    # methods (matches rd_residual_histogram_CLOUDf48.png which used k=10 for all).
    ax_hist = fig.add_subplot(gs[1, 2])
    HIST_K  = 10                             # fixed k for histogram (comparable scales)
    HIST_AB = 0.01                           # shared binning scale (matches reference)
    HIST_BW = 2.0 * HIST_AB / NUM_BINS      # ≈ 3.05e-7 per bin

    print("  [histogram] running k=10 reconstructions for each method …", flush=True)
    hist_resids = {}
    try:
        _, _, h_resid = recon_tdeim(data_ds, HIST_K, HIST_AB)
        hist_resids["T-DEIM"] = h_resid
    except Exception as e:
        print(f"    T-DEIM histogram skip: {e}")
    try:
        _, _, h_resid = recon_deim2d(data_ds, HIST_K, HIST_AB)
        hist_resids["DEIM-2D"] = h_resid
    except Exception as e:
        print(f"    DEIM-2D histogram skip: {e}")
    try:
        _, _, h_resid, _ = recon_kriging2d(data_ds, HIST_K, HIST_AB)
        hist_resids["Kriging-2D"] = h_resid
    except Exception as e:
        print(f"    Kriging-2D histogram skip: {e}")
    if data2_ds is not None:
        try:
            _, _, h_resid, _ = recon_multigp(data_ds, data2_ds, HIST_K, HIST_AB)
            hist_resids["MultiGP"] = h_resid
        except Exception as e:
            print(f"    MultiGP histogram skip: {e}")

    zoom_half = 0
    hist_data = {}
    for method in HIST_METHODS:
        if method not in hist_resids:
            continue
        pre_resid = hist_resids[method]
        b32  = np.round(pre_resid.astype(np.float64) / HIST_BW).astype(np.int32)
        half = NUM_BINS // 2
        b32  = np.clip(b32, -(half - 1), half - 1)
        z_half = 50
        zoom_half = max(zoom_half, z_half)
        hist_data[method] = b32

    zoom_lo_all, zoom_hi_all = -zoom_half, zoom_half

    # One bar per integer bin index — all methods overlapping with transparency
    for method in HIST_METHODS:
        if method not in hist_data:
            continue
        b32 = hist_data[method]
        in_range = b32[(b32 >= zoom_lo_all) & (b32 <= zoom_hi_all)]
        unique_bins, counts = np.unique(in_range, return_counts=True)
        ax_hist.bar(unique_bins, counts, width=1.0,
                    color=COLORS[method], alpha=0.35, label=method,
                    edgecolor="none", align="center")

    ax_hist.set_xlim(zoom_lo_all, zoom_hi_all)
    ax_hist.set_xlabel("Bin index", fontsize=22)
    ax_hist.set_ylabel("Count", fontsize=22)
    ax_hist.set_title(f"Prediction Residuals  (k={HIST_K})\n(zoomed, bin width = 2·0.01 / 65536)", fontsize=20)
    ax_hist.legend(fontsize=18, loc="upper right", framealpha=0.85)
    ax_hist.tick_params(labelsize=18)

    out = ARGONNE / f"poster_fig3_big_panel_{FIELD_TAG}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    data, data2, data_ds, data2_ds = load_data()
    by_method = load_csv()

    print("\n[Fig 3] 2×3 big panel (all methods, DS={DS}, {500//DS}×{500//DS} grid) ...")
    plot_fig3_big_panel(data_ds, data2_ds, by_method)

    print("\n[RD curves] ...")
    plot_rd_curves(by_method, "psnr", "PSNR (dB)", "psnr")
    plot_rd_curves(by_method, "ssim", "SSIM",      "ssim")

    print("\nDone.")

if __name__ == "__main__":
    main()
