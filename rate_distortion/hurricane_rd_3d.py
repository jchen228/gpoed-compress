#!/usr/bin/env python3
"""
hurricane_rd_3d.py
==================
3-D Kronecker-GP rate–distortion comparison on ISABEL hurricane TC.

The GP uses a separable kernel
    K_3d((x1,z1),(x2,z2)) = K_spatial(x1,x2) · K_vertical(z1,z2)

exploiting both 2-D horizontal and vertical (level) correlations.
Sensor layout: k_xy RPCholesky spatial sensors × k_z evenly-spaced vertical
levels → k_xy × k_z total sensors in the 3-D volume.

Prediction via Kronecker trick — avoids forming the 25M-point K_Xs matrix:
    MU_norm (n_xy, n_z) = K_xy_Xs @ (K_z_Xs @ alpha_mat).T

SZ2/ZFP baseline and GP residual compression both operate on the full
(100, 500, 500) 3-D volume so the comparison is symmetric.

Existing 2-D spatial checkpoint reused for ls, var, noise, sensors.
ls_z fitted separately from vertical profiles.

Outputs (only)
--------------
  poster_rd_psnr_3D_Hurricane.png
  rd_results_3D_Hurricane.csv
"""
from __future__ import annotations
from pathlib import Path
import time, csv, struct
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize as _minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ARGONNE  = Path(__file__).resolve().parent.parent
TC_FILE  = ARGONNE / "100x500x500" / "TCf48.bin.f32"
# Reuse existing 2-D checkpoint for spatial hyperparams/sensors
CKPT_2D  = ARGONNE / "kriging_ckpt_Hurricane_N400.npz"
CKPT_3D  = ARGONNE / "kriging_3d_ckpt_Hurricane.npz"

# ── Parameters ────────────────────────────────────────────────────────────────
N_T, NY, NX = 100, 500, 500
N_FULL       = NY * NX
TIME_IDX     = 50
DPI          = 150
JITTER       = 1e-6

KRIG_CONFIGS = [(400, 10)]  # single representative config (results insensitive to k_xy/k_z)

ABS_BOUNDS = np.logspace(-4, 0, 20)
RUN_ZFP    = True
PLOTS_ONLY = True

CSV_PATH = ARGONNE / "rd_results_3D_Hurricane.csv"

# ── Colours / markers ─────────────────────────────────────────────────────────
COLORS  = {"SZ2": "#1f77b4", "ZFP": "#ff7f0e",
           "Kriging-3D+SZ2": "#e377c2", "Kriging-3D+ZFP": "#8c564b"}
MARKERS = {"SZ2": "o", "ZFP": "s", "Kriging-3D+SZ2": "D", "Kriging-3D+ZFP": "D"}
METHOD_ORDER = ["SZ2", "ZFP", "Kriging-3D+SZ2"]

# ── Compression helpers ───────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    _cctx = _zstd.ZstdCompressor(level=3)
    def _compress(b): return _cctx.compress(b)
except ImportError:
    import zlib
    def _compress(b): return zlib.compress(b, 6)

def compress_f32(a): return _compress(a.astype(np.float32).tobytes())
def compress_i32(a): return _compress(a.astype(np.int32).tobytes())
def compress_f16(a): return _compress(a.astype(np.float16).tobytes())

# ── Kernel ────────────────────────────────────────────────────────────────────
def matern32(X1, X2, ls):
    d = np.sqrt(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1))
    r = np.sqrt(3) * d / ls
    return (1.0 + r) * np.exp(-r)

def matern32_1d(z1, z2, ls):
    """1-D Matérn-3/2 for vertical coordinates (z1, z2 are 1-D arrays)."""
    d = np.abs(z1[:, None] - z2[None, :])
    r = np.sqrt(3) * d / ls
    return (1.0 + r) * np.exp(-r)

# ── RPCholesky sensor selection ───────────────────────────────────────────────
def _rpcholesky_sensors(X, ls, k, rng=None):
    """Greedy RPCholesky pivoting on the Matérn-3/2 kernel diagonal."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = X.shape[0]
    diag = np.ones(n, dtype=np.float64)
    F    = np.zeros((n, k), dtype=np.float64)
    pivots = []
    for j in range(k):
        probs = np.maximum(diag, 0.0)
        s     = probs.sum()
        if s < 1e-14:
            break
        i = rng.choice(n, p=probs / s)
        pivots.append(i)
        col       = matern32(X, X[[i]], ls).ravel()
        if j > 0:
            col -= F[:, :j] @ F[i, :j]
        denom = max(diag[i], 1e-14)
        col  /= np.sqrt(denom)
        F[:, j] = col
        diag  -= col ** 2
    return np.array(pivots[:len(pivots)], dtype=np.int64)

# ── CSV helpers ───────────────────────────────────────────────────────────────
CSV_FIELDS = ["method", "k_xy", "k_z", "abs_bound", "cr", "psnr",
              "comp_sec", "train_sec", "compressed_MB"]

def save_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def load_csv():
    by_method = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            for fld in ("cr", "psnr", "abs_bound", "comp_sec",
                        "train_sec", "compressed_MB"):
                try:
                    row[fld] = float(row[fld])
                except Exception:
                    row[fld] = float("nan")
            for fld in ("k_xy", "k_z"):
                try:
                    row[fld] = int(row[fld])
                except Exception:
                    row[fld] = 0
            by_method.setdefault(row["method"], []).append(row)
    return by_method

# ── libpressio 3-D helpers ────────────────────────────────────────────────────
def _run_3d(data, compressor_id, config):
    """Compress the full (n_T, ny, nx) volume in one shot."""
    import libpressio
    n_all     = data.size
    global_dr = float(data.max() - data.min())
    vol  = np.ascontiguousarray(data.astype(np.float32))
    reco = vol.copy()
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id":    compressor_id,
        "early_config":     {"pressio:metric": "composite",
                             "composite:plugins": ["size"]},
        "compressor_config": config,
    })
    t0   = time.perf_counter()
    reco = comp.decode(comp.encode(vol), reco)
    cs   = time.perf_counter() - t0
    cr_r = comp.get_metrics().get("size:compression_ratio", float("nan"))
    tb   = (int(round(n_all * 4 / cr_r))
            if np.isfinite(cr_r) and cr_r > 0 else n_all * 4)
    mse  = float(np.mean((data.astype(np.float64) - reco.astype(np.float64)) ** 2))
    psnr = (20.0 * np.log10(global_dr / np.sqrt(mse))
            if mse > 1e-15 and global_dr > 1e-10 else 999.9)
    return tb, cs, psnr

def run_sz2(data, abs_bounds=ABS_BOUNDS):
    """SZ2 on the full (100, 500, 500) 3-D volume."""
    n_all, results = data.size, []
    for ab in abs_bounds:
        tb, cs, pv = _run_3d(data, "sz",
                              {"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)})
        cr = (n_all * 4) / tb if tb > 0 else float("nan")
        results.append({"method": "SZ2", "k_xy": 0, "k_z": 0, "abs_bound": ab,
                         "cr": cr, "psnr": pv, "comp_sec": cs, "train_sec": 0.0,
                         "compressed_MB": tb / 1e6})
        print(f"  SZ2-3D  abs={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results

def run_zfp(data, abs_bounds=ABS_BOUNDS):
    """ZFP on the full (100, 500, 500) 3-D volume."""
    n_all, results = data.size, []
    for ab in abs_bounds:
        tb, cs, pv = _run_3d(data, "zfp", {"zfp:accuracy": float(ab)})
        cr = (n_all * 4) / tb if tb > 0 else float("nan")
        results.append({"method": "ZFP", "k_xy": 0, "k_z": 0, "abs_bound": ab,
                         "cr": cr, "psnr": pv, "comp_sec": cs, "train_sec": 0.0,
                         "compressed_MB": tb / 1e6})
        print(f"  ZFP-3D  acc={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")
    return results

# ── Kronecker-GP ──────────────────────────────────────────────────────────────
def _fit_ls_z(data_flat, train_mean, train_std_safe, z_norm):
    """Fit vertical length scale via 1-D MLE on 200 random pixel profiles."""
    rng     = np.random.default_rng(42)
    pix_idx = rng.choice(N_FULL, size=min(200, N_FULL), replace=False)
    # Normalise each pixel's vertical profile
    Y_sub   = ((data_flat[:, pix_idx] - train_mean[pix_idx])
               / train_std_safe[pix_idx])  # (n_T, 200)

    def _nll(log_ls_z):
        ls_z_ = float(np.exp(log_ls_z))
        K = matern32_1d(z_norm, z_norm, ls_z_) + 1e-4 * np.eye(N_T)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        log_det = 2 * np.sum(np.log(np.diag(L)))
        # Sum NLL over all pixel profiles
        nll = 0.0
        for j in range(Y_sub.shape[1]):
            a = np.linalg.solve(L.T, np.linalg.solve(L, Y_sub[:, j]))
            nll += 0.5 * float(Y_sub[:, j] @ a) + 0.5 * log_det
        return nll

    res = _minimize(_nll, x0=np.log(0.3), method="L-BFGS-B",
                    bounds=[(-4, 2)], options={"maxiter": 100})
    ls_z = float(np.exp(res.x[0]))
    print(f"  ls_z = {ls_z:.4f}  (vertical length scale, normalised units)")
    return ls_z


def run_kriging_3d(data, X_spatial, configs=KRIG_CONFIGS, abs_bounds=ABS_BOUNDS):
    """
    Kronecker-GP: K_3d = K_spatial ⊗ K_vertical.

    Sensor layout: k_xy spatial sensors × k_z vertical levels.
    Prediction via the Kronecker identity (no 25M-row matrix):
        MU_norm (n_xy, n_z) = var * K_xy_Xs @ (K_z_Xs @ alpha_mat).T

    Residuals compressed as a full 3-D (n_T, ny, nx) volume with SZ2.
    """
    try:
        import libpressio as _lp
    except ImportError:
        print("  libpressio not available — skipping Kriging-3D"); return []

    data_flat = data.reshape(N_T, N_FULL).astype(np.float64)  # (n_T, n_full)
    n_all     = N_T * N_FULL

    # Vertical coordinates normalised to [0, 1]
    z_norm   = np.linspace(0.0, 1.0, N_T)              # (100,)

    # ── Load or train spatial hyperparams ─────────────────────────────────────
    t0 = time.perf_counter()
    if CKPT_3D.exists():
        print(f"  [3D ckpt] Loading {CKPT_3D.name} ...")
        ck           = np.load(CKPT_3D)
        ls_xy        = float(ck["ls_xy"])
        ls_z         = float(ck["ls_z"])
        var_gp       = float(ck["var"])
        noise_var    = float(ck["noise_var"])
        sensors_xy   = ck["sensors_xy"]
        train_mean   = ck["train_mean"].astype(np.float64)
        train_std_s  = ck["train_std_safe"].astype(np.float64)
        train_sec    = float(ck.get("train_sec", 0.0))
        print(f"  ls_xy={ls_xy:.4f}  ls_z={ls_z:.4f}  var={var_gp:.4f}  "
              f"noise={noise_var:.2e}  sensors={len(sensors_xy)}")
    elif CKPT_2D.exists():
        print(f"  [2D ckpt] Loading spatial params from {CKPT_2D.name} ...")
        ck           = np.load(CKPT_2D)
        ls_xy        = float(ck["ls"])
        var_gp       = float(ck["var"])
        noise_var    = float(ck["noise_var"])
        sensors_xy   = ck["sensors"]
        train_mean   = ck["train_mean"].astype(np.float64)
        train_std_s  = ck["train_std_safe"].astype(np.float64)
        # Fit vertical length scale
        print("  Fitting vertical length scale ls_z ...")
        ls_z = _fit_ls_z(data_flat, train_mean, train_std_s, z_norm)
        train_sec = time.perf_counter() - t0
        np.savez_compressed(str(CKPT_3D),
                            ls_xy=np.float64(ls_xy), ls_z=np.float64(ls_z),
                            var=np.float64(var_gp), noise_var=np.float64(noise_var),
                            sensors_xy=sensors_xy,
                            train_mean=train_mean.astype(np.float32),
                            train_std_safe=train_std_s.astype(np.float32),
                            train_sec=np.float64(train_sec))
        print(f"  [3D ckpt] Saved → {CKPT_3D.name}")
    else:
        raise FileNotFoundError(
            "No spatial checkpoint found. Run hurricane_rd_comparison.py first "
            "to generate kriging_ckpt_Hurricane_N400.npz.")

    k_max_xy   = min(max(k for k, _ in configs), len(sensors_xy))
    sensors_xy = sensors_xy[:k_max_xy]
    zero_std   = (train_std_s < 1e-10)

    # Spatial kernel matrices built once at k_max_xy, sliced per config
    X_sens_max  = X_spatial[sensors_xy]
    print(f"  Building K_xy_Xs ({N_FULL} × {k_max_xy}) ...", flush=True)
    K_xy_Xs_max = matern32(X_spatial, X_sens_max, ls_xy)   # (n_full, k_max_xy)
    K_xy_ss_max = matern32(X_sens_max, X_sens_max, ls_xy)  # (k_max_xy, k_max_xy)

    ms_xy = train_mean[sensors_xy]
    ss_xy = train_std_s[sensors_xy]

    global_dr = float(data.max() - data.min())
    orig_flat = data_flat
    results   = []

    for (k_xy, k_z) in configs:
        k_xy = min(k_xy, k_max_xy)

        # Vertical sensor levels for this k_z
        z_sens_idx = np.round(np.linspace(0, N_T - 1, k_z)).astype(int)
        z_sens     = z_norm[z_sens_idx]                     # (k_z,)
        K_z_ss     = matern32_1d(z_sens, z_sens, ls_z)      # (k_z, k_z)
        K_z_Xs     = matern32_1d(z_norm, z_sens, ls_z)      # (n_T, k_z)

        s_k      = sensors_xy[:k_xy]
        K_xy_Xs  = K_xy_Xs_max[:, :k_xy]
        K_xy_ss  = K_xy_ss_max[:k_xy, :k_xy]

        k_total  = k_z * k_xy
        K_ss     = var_gp * np.kron(K_z_ss, K_xy_ss) + noise_var * np.eye(k_total)
        try:
            L_k, lo = cho_factor(K_ss, lower=True)
        except np.linalg.LinAlgError:
            K_ss += 1e-3 * np.eye(k_total)
            L_k, lo = cho_factor(K_ss, lower=True)

        Y_obs_raw  = data_flat[z_sens_idx, :]               # (k_z, n_full)
        Y_obs_norm = (Y_obs_raw[:, s_k] - ms_xy[:k_xy]) / ss_xy[:k_xy]
        y_vec      = Y_obs_norm.ravel()

        alpha_vec  = var_gp * cho_solve((L_k, lo), y_vec)
        alpha_mat  = alpha_vec.reshape(k_z, k_xy)

        temp      = K_z_Xs @ alpha_mat                      # (n_T, k_xy)
        MU_norm   = K_xy_Xs @ temp.T                        # (n_full, n_T)

        MU = MU_norm * train_std_s[:, np.newaxis] + train_mean[:, np.newaxis]
        MU[zero_std, :] = train_mean[zero_std, np.newaxis]

        pred_full = MU.T.astype(np.float32)                 # (n_T, n_full)
        pred_3d   = pred_full.reshape(N_T, NY, NX)
        resid_3d  = (data - pred_3d).astype(np.float32)

        model_b = (struct.calcsize("<Iddd") +
                   len(compress_i32(s_k.astype(np.int32))) +
                   len(compress_i32(z_sens_idx.astype(np.int32))) +
                   len(compress_f32(train_mean.astype(np.float32))) +
                   len(compress_f32(train_std_s.astype(np.float32))))
        sv_vals = data_flat[z_sens_idx, :][:, s_k].astype(np.float16)
        sv_enc  = _compress(sv_vals.tobytes())

        print(f"  k_xy={k_xy:3d}  k_z={k_z}  sweeping {len(abs_bounds)} bounds ...",
              flush=True)
        for ab in abs_bounds:
            cfg  = {"sz:error_bound_mode": 0, "sz:abs_err_bound": float(ab)}
            comp = _lp.PressioCompressor.from_config({
                "compressor_id":     "sz",
                "early_config":      {"pressio:metric": "composite",
                                      "composite:plugins": ["size"]},
                "compressor_config": cfg,
            })
            r_vol  = np.ascontiguousarray(resid_3d)
            r_reco = r_vol.copy()
            t0c    = time.perf_counter()
            r_reco = comp.decode(comp.encode(r_vol), r_reco)
            cs     = time.perf_counter() - t0c
            cr_r   = comp.get_metrics().get("size:compression_ratio", float("nan"))
            resid_b = (int(round(N_T * N_FULL * 4 / cr_r))
                       if np.isfinite(cr_r) and cr_r > 0 else N_T * N_FULL * 4)

            final = pred_full + r_reco.reshape(N_T, N_FULL).astype(np.float64)
            mse   = float(np.mean((orig_flat - final) ** 2))
            pv    = (20.0 * np.log10(global_dr / np.sqrt(mse))
                     if mse > 1e-15 and global_dr > 1e-10 else float("nan"))
            tb    = model_b + len(sv_enc) + resid_b
            cr    = (n_all * 4) / tb

            results.append({"method": "Kriging-3D+SZ2", "k_xy": k_xy,
                             "k_z": k_z, "abs_bound": ab,
                             "cr": cr, "psnr": pv, "comp_sec": cs,
                             "train_sec": train_sec,
                             "compressed_MB": tb / 1e6})
            print(f"    ab={ab:.2e}  CR={cr:.1f}×  PSNR={pv:.1f} dB")

    return results

# ── Plotting ──────────────────────────────────────────────────────────────────
def _trim_monotone(pts):
    """Remove non-monotone points: sort by abs_bound ascending (tighter → looser),
    CR should increase; drop any point where CR goes backwards."""
    srt = sorted(pts, key=lambda p: p["abs_bound"])
    out, max_cr = [], -np.inf
    for p in srt:
        if p["cr"] >= max_cr:
            max_cr = p["cr"]; out.append(p)
    return out

def _plot_by_k(ax, pts, c, mk, lw, ms):
    # Group by (k_xy, k_z) pair; only plot configured pairs
    by_cfg = {}
    for p in pts:
        key = (p["k_xy"], p["k_z"])
        by_cfg.setdefault(key, []).append(p)
    # Only show configs listed in KRIG_CONFIGS
    linestyles = ["-", "--", ":", "-."]
    markers    = ["D", "s", "^", "o"]
    for i, cfg in enumerate(KRIG_CONFIGS):
        if cfg not in by_cfg:
            continue
        kpts = by_cfg[cfg]
        srt  = sorted(kpts, key=lambda p: p["cr"])
        k_xy, k_z = cfg
        lbl  = f"Kriging-3D+SZ2  (k_xy={k_xy}, k_z={k_z})"
        ax.plot([p["cr"] for p in srt], [p["psnr"] for p in srt],
                linestyles[i % len(linestyles)], color=c,
                marker=markers[i % len(markers)],
                lw=lw, ms=ms, label=lbl)

def plot_rd_3d(by_method):
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHOD_ORDER:
        pts = [p for p in by_method.get(method, [])
               if np.isfinite(p.get("psnr", float("nan"))) and p["cr"] > 0]
        if not pts:
            continue
        c  = COLORS.get(method, "#888")
        mk = MARKERS.get(method, "o")
        if method in ("SZ2", "ZFP"):
            srt = sorted(_trim_monotone(pts), key=lambda p: p["cr"])
            ab0 = min(p["abs_bound"] for p in srt)
            ab1 = max(p["abs_bound"] for p in srt)
            ax.plot([p["cr"] for p in srt], [p["psnr"] for p in srt],
                    "-", color=c, marker=mk, lw=2.2, ms=9,
                    label=f"{method}-3D (ε={ab0:.0e}–{ab1:.0e})")
        else:
            _plot_by_k(ax, pts, c, mk, lw=2.0, ms=6)

    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    cfg_str = ", ".join(f"k_xy={k},k_z={z}" for k, z in KRIG_CONFIGS)
    ax.set_title(
        "Rate–Distortion  —  ISABEL Hurricane TC  [3-D Kronecker-GP]\n"
        f"SZ2/ZFP: full (100×500×500) volume · GP configs: {cfg_str}",
        fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")

    out = ARGONNE / "poster_rd_psnr_3D_Hurricane.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading TC data ...")
    tc = np.fromfile(str(TC_FILE), dtype=np.float32).reshape(N_T, NY, NX)
    print(f"  TC: shape={tc.shape}  range=[{tc.min():.2f}, {tc.max():.2f}] °C")

    # Spatial coordinates (normalised)
    yi = np.linspace(0, 1, NY); xi = np.linspace(0, 1, NX)
    XX, YY = np.meshgrid(xi, yi, indexing="ij")
    X_spatial = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float64)

    all_results = []
    _done = set()

    if CSV_PATH.exists():
        try:
            by_m = load_csv()
            for m, rows in by_m.items():
                all_results.extend(rows); _done.add(m)
            print(f"  [resume] Loaded {len(all_results)} rows for: {sorted(_done)}")
        except Exception as e:
            print(f"  [resume] Failed ({e}), starting fresh")

    if not PLOTS_ONLY:
        try:
            import libpressio  # noqa: F401

            if "SZ2" not in _done:
                print("\n── SZ2-3D ────────────────────────────────────────")
                all_results += run_sz2(tc)
                save_csv(all_results)
            else:
                print("\n── SZ2-3D (skipped — already in CSV) ────────────")

            if RUN_ZFP and "ZFP" not in _done:
                print("\n── ZFP-3D ────────────────────────────────────────")
                all_results += run_zfp(tc)
                save_csv(all_results)
            elif "ZFP" in _done:
                print("\n── ZFP-3D (skipped — already in CSV) ────────────")

        except ImportError:
            print("  libpressio not found — skipping SZ2/ZFP")

        # Determine which (k_xy, k_z) configs still need to be run
        done_cfgs = {(r["k_xy"], r["k_z"]) for r in all_results
                     if r["method"] == "Kriging-3D+SZ2"}
        missing   = [cfg for cfg in KRIG_CONFIGS if cfg not in done_cfgs]
        if missing:
            print(f"\n── Kriging-3D Kronecker — running {missing} ──────────")
            all_results += run_kriging_3d(tc, X_spatial, configs=missing)
            save_csv(all_results)
        else:
            print("\n── Kriging-3D (skipped — all configs in CSV) ─────────")

    # Plot
    by_method = {}
    for r in all_results:
        by_method.setdefault(r["method"], []).append(r)
    print("\n── Plotting ──────────────────────────────────────────")
    plot_rd_3d(by_method)
    print("Done.")

if __name__ == "__main__":
    main()
