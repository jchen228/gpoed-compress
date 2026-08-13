#!/usr/bin/env python3
"""
hurricane_gp_comparison.py
===========================
Single-script progressive GP compression + SZ3 rate-distortion comparison
on the ISABEL hurricane U-wind field (Uf48.bin.f32).

Set MODE = '2d' for slice-by-slice 2D GP, or '3d' for the full 3D GP.

Both methods are run at every EB in EB_VALUES so the rate-distortion curves
are directly comparable.  GP runs are cached as checkpoints — if a checkpoint
for a given (mode, EB) already exists it is loaded rather than recomputed.
"""
from __future__ import annotations
import sys
import time
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ARGONNE   = Path(__file__).resolve().parent.parent
DATA_DIR  = ARGONNE / "100x500x500"
DATA_FILE = DATA_DIR / "Uf48.bin.f32"

SZ3_BIN = Path("/Users/jchen228/spack/opt/spack/darwin-m1/"
               "sz3-3.4.0-cwdwysro3s55ur4mqvlhaar437kjs24w/bin/sz3")

# ── Mode ─────────────────────────────────────────────────────────────────────
MODE = '2d'   # '2d'  → slice-by-slice 2D GP
              # '3d'  → full 3D GP on entire volume

# ── Run mode ─────────────────────────────────────────────────────────────────
# 'rd'  → sweep EB_VALUES at fixed LS_XY; produce rate-distortion curve vs SZ3
# 'ls'  → sweep LS_XY_VALUES at fixed LS_EB; find best lengthscale first
RUN_MODE = 'rd'

# ── EB sweep (used when RUN_MODE = 'rd') ─────────────────────────────────────
EB_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]

# ── LS sweep (used when RUN_MODE = 'ls') ─────────────────────────────────────
LS_XY_VALUES = [0.15, 0.25, 0.35, 0.50]   # values to try
LS_EB        = 0.01                         # fixed EB for the LS sweep

# ── Shared algorithm knobs ────────────────────────────────────────────────────
# Stopping criterion: run until TARGET_CR or TARGET_PSNR is met, up to MAX_ROUNDS.
# Set both targets to None to use fixed N_ROUNDS instead.
TARGET_CR       = None   # e.g. 5.0  — stop when estimated CR  ≥ this
TARGET_PSNR     = None   # e.g. 85.0 — stop when estimated PSNR ≥ this (dB)
TARGET_BEAT_SZ3 = True   # if True: run SZ3 first, then run GP until its
                          # estimated bit_rate < SZ3 bit_rate at the same EB.
MAX_ROUNDS      = 50     # hard cap when using target-based stopping
N_ROUNDS        = 10     # used only when all targets are None
ACCEPT_BINS     = 10     # ACCEPT_BOUND = ACCEPT_BINS × EB
DOWNSAMPLE   = 10      # spatial downsampling (10 → 50×50 per slice)
Z_SKIP_BOTTOM = 0      # 0 = use all 100 z-levels
ACCUMULATE   = True

SENSOR_STRATEGY  = 'hybrid'
N_RPGKS_ROUNDS   = 1
RESID_PERCENTILE = 25
CHOL_RANK_OVERSAMPLE = 20

KERNEL_TYPE = 'matern52'   # 'matern52' | 'matern32' | 'matern12'
LS_XY  = 0.15              # overridden per-iteration in LS sweep
SIG2   = 1.0
NOISE  = 1e-3
SEED   = 42

# ── 2D-specific ───────────────────────────────────────────────────────────────
K_PER_ROUND_2D = 10
BATCH_PRED_2D  = 10_000

# ── 3D-specific ───────────────────────────────────────────────────────────────
K_PER_ROUND_3D = 250
LS_Z           = 0.25
LOCAL_GP       = True
N_LOCAL        = 50
BATCH_PRED_3D  = 50_000

# ── Sensor storage strategy ───────────────────────────────────────────────────
# Determines how sensor (unpredictable-point) values are stored AND what values
# the GP uses during compression/decompression.  Encoder and decoder must agree,
# so the GP is run with the stored representation — not exact float64.
#
#   'float32'  — 32 bits/sensor; GP uses float32-rounded values (≈ exact for
#                this dataset, matches SZ2/SZ3 unpredictable-point buffer)
#   'float16'  — 16 bits/sensor; GP uses float16-rounded values (cheap halving)
#   'eb_quant' — entropy-coded EB-quantized values (~H_sens bits/sensor);
#                GP uses values quantized to EB bins (error ≤ EB/2 at sensors);
#                NOISE is automatically raised to max(NOISE, (EB/2)²)
SENSOR_STORAGE = 'eb_quant'
OUT_FIG    = ARGONNE / f"rd_curve_{MODE}.png"
OUT_LS_FIG = ARGONNE / f"ls_sweep_{MODE}.png"

# ── Grid (derived) ────────────────────────────────────────────────────────────
NZ_ORIG, NY_ORIG, NX_ORIG = 100, 500, 500
NZ      = NZ_ORIG - Z_SKIP_BOTTOM
NY      = -(-NY_ORIG // DOWNSAMPLE)   # ceiling division → matches numpy ::DS slicing
NX      = -(-NX_ORIG // DOWNSAMPLE)
N_SLICE = NY * NX
N       = NZ * N_SLICE


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & COORDINATES
# ─────────────────────────────────────────────────────────────────────────────

def load_wind() -> np.ndarray:
    raw = np.fromfile(DATA_FILE, dtype=np.float32)
    vol = raw.reshape(NZ_ORIG, NY_ORIG, NX_ORIG)
    return vol[Z_SKIP_BOTTOM:, ::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float64)


def build_coords_2d() -> np.ndarray:
    """(N_SLICE, 2) array of [y, x] in [0,1]²."""
    gy, gx = np.linspace(0, 1, NY), np.linspace(0, 1, NX)
    yy, xx = np.meshgrid(gy, gx, indexing='ij')
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def build_coords_3d() -> np.ndarray:
    """(N, 3) array of [z, y, x] in [0,1]³."""
    gz, gy, gx = np.linspace(0, 1, NZ), np.linspace(0, 1, NY), np.linspace(0, 1, NX)
    zz, yy, xx = np.meshgrid(gz, gy, gx, indexing='ij')
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# KERNELS
# ─────────────────────────────────────────────────────────────────────────────

def _matern52(r2, r, sig2): return sig2*(1+np.sqrt(5)*r+(5/3)*r2)*np.exp(-np.sqrt(5)*r)
def _matern32(r2, r, sig2): return sig2*(1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)
def _matern12(r2, r, sig2): return sig2*np.exp(-r)

def _dispatch(r2, r, sig2):
    if KERNEL_TYPE == 'matern52': return _matern52(r2, r, sig2)
    if KERNEL_TYPE == 'matern32': return _matern32(r2, r, sig2)
    if KERNEL_TYPE == 'matern12': return _matern12(r2, r, sig2)
    raise ValueError(f"Unknown KERNEL_TYPE: {KERNEL_TYPE!r}")


def kernel_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """2D Matérn kernel over (y, x) coords. A:(m,2), B:(n,2) → (m,n)."""
    dy = (A[:, 0:1] - B[:, 0]) / LS_XY
    dx = (A[:, 1:2] - B[:, 1]) / LS_XY
    r2 = dy**2 + dx**2
    return _dispatch(r2, np.sqrt(np.maximum(r2, 0.0)), SIG2)


def kernel_3d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """3D anisotropic Matérn kernel. A:(m,3), B:(n,3) → (m,n)."""
    dz = (A[:, 0:1] - B[:, 0]) / LS_Z
    dy = (A[:, 1:2] - B[:, 1]) / LS_XY
    dx = (A[:, 2:3] - B[:, 2]) / LS_XY
    r2 = dz**2 + dy**2 + dx**2
    return _dispatch(r2, np.sqrt(np.maximum(r2, 0.0)), SIG2)


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SELECTION  (rpgks — works for any coordinate dimension)
# ─────────────────────────────────────────────────────────────────────────────

def select_sensors(coords_cand: np.ndarray, k: int, rng,
                   kern_fn=None) -> np.ndarray:
    """
    Randomly Pivoted Cholesky + GKS (rpgks).
    Returns k indices into coords_cand.
    kern_fn defaults to kernel_3d; pass kernel_2d for 2D mode.
    """
    if kern_fn is None:
        kern_fn = kernel_3d
    n_cand = len(coords_cand)
    k      = min(k, n_cand)
    if k == n_cand:
        return np.arange(k)

    rank  = min(k + CHOL_RANK_OVERSAMPLE, n_cand)
    diags = SIG2 * np.ones(n_cand)
    F     = np.zeros((n_cand, rank))

    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total < 1e-12:
            actual_rank = i; break
        p  = int(rng.choice(n_cand, p=diags / total))
        g  = kern_fn(coords_cand, coords_cand[[p]]).ravel()
        if i > 0:
            g -= F[:, :i] @ F[p, :i]
        piv = g[p]
        if piv <= 0:
            actual_rank = i; break
        F[:, i] = g / np.sqrt(piv)
        diags    = np.maximum(diags - F[:, i]**2, 0.0)

    F = F[:, :actual_rank]
    if actual_rank == 0:
        return rng.choice(n_cand, size=k, replace=False)

    U, _, _ = np.linalg.svd(F, full_matrices=False)
    _, _, p = scipy_qr(U[:, :k].T, pivoting=True)
    return p[:k].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# GP PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def gp_predict_2d(coords_s, y_s, coords_p, batch=None, noise=None):
    """Standard batched GP posterior for 2D kernel."""
    if batch is None: batch = BATCH_PRED_2D
    if noise is None: noise = NOISE
    k_s   = len(coords_s)
    K_ss  = kernel_2d(coords_s, coords_s) + noise * np.eye(k_s)
    L     = np.linalg.cholesky(K_ss)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_s))
    L_inv = np.linalg.solve(L, np.eye(k_s))
    n     = len(coords_p)
    mu    = np.empty(n); var = np.empty(n)
    for s in range(0, n, batch):
        e       = min(s + batch, n)
        K_bs    = kernel_2d(coords_p[s:e], coords_s)
        mu[s:e] = K_bs @ alpha
        v       = L_inv @ K_bs.T
        var[s:e]= np.maximum(SIG2 - np.sum(v**2, axis=0), 0.0)
    return mu, var


def gp_predict_local_3d(coords_s, y_s, coords_p, n_local=None, batch=2_000, noise=None):
    """KDTree local GP for 3D: each voxel predicted from its n_local nearest sensors."""
    from scipy.spatial import KDTree
    if n_local is None: n_local = N_LOCAL
    if noise is None: noise = NOISE
    nl      = min(n_local, len(coords_s))
    n_p     = len(coords_p)
    mu      = np.empty(n_p); var = np.empty(n_p)
    _, nn   = KDTree(coords_s).query(coords_p, k=nl, workers=-1)
    eye_nl  = np.eye(nl)

    for s in range(0, n_p, batch):
        e      = min(s + batch, n_p); B = e - s
        li     = nn[s:e]                         # (B, nl)
        X_s    = coords_s[li]                    # (B, nl, 3)
        y_sl   = y_s[li]                         # (B, nl)
        X_p    = coords_p[s:e]                   # (B, 3)

        # K_ss (B, nl, nl)
        d      = X_s[:, :, None, :] - X_s[:, None, :, :]   # (B,nl,nl,3)
        dz_ss  = d[..., 0] / LS_Z;  dy_ss = d[..., 1] / LS_XY;  dx_ss = d[..., 2] / LS_XY
        r2_ss  = dz_ss**2 + dy_ss**2 + dx_ss**2
        K_ss   = _dispatch(r2_ss, np.sqrt(np.maximum(r2_ss, 0.0)), SIG2) + (noise+1e-6)*eye_nl

        # K_ps (B, nl)
        dp     = X_p[:, None, :] - X_s                       # (B, nl, 3)
        dz_p   = dp[..., 0] / LS_Z;  dy_p = dp[..., 1] / LS_XY;  dx_p = dp[..., 2] / LS_XY
        r2_p   = dz_p**2 + dy_p**2 + dx_p**2
        K_ps   = _dispatch(r2_p, np.sqrt(np.maximum(r2_p, 0.0)), SIG2)

        alpha        = np.linalg.solve(K_ss, y_sl[:, :, None])[:, :, 0]
        mu[s:e]      = np.einsum('bi,bi->b', K_ps, alpha)
        v            = np.linalg.solve(K_ss, K_ps[:, :, None])
        var[s:e]     = np.maximum(SIG2 - np.einsum('bi,bi->b', K_ps, v[:, :, 0]), 0.0)
    return mu, var


def gp_predict_global_3d(coords_s, y_s, coords_p, batch=None, noise=None):
    """Standard batched global GP for 3D kernel."""
    if batch is None: batch = BATCH_PRED_3D
    if noise is None: noise = NOISE
    k_s   = len(coords_s)
    K_ss  = kernel_3d(coords_s, coords_s) + noise * np.eye(k_s)
    L     = np.linalg.cholesky(K_ss)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_s))
    L_inv = np.linalg.solve(L, np.eye(k_s))
    n     = len(coords_p)
    mu    = np.empty(n); var = np.empty(n)
    for s in range(0, n, batch):
        e       = min(s + batch, n)
        K_bs    = kernel_3d(coords_p[s:e], coords_s)
        mu[s:e] = K_bs @ alpha
        v       = L_inv @ K_bs.T
        var[s:e]= np.maximum(SIG2 - np.sum(v**2, axis=0), 0.0)
    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT I/O
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_path(mode: str, eb: float, ls_xy: float | None = None,
                    target_psnr_override: float | None = None) -> Path:
    ls = ls_xy if ls_xy is not None else LS_XY
    # Effective per-EB targets
    eff_cr   = TARGET_CR
    eff_psnr = target_psnr_override if target_psnr_override is not None else TARGET_PSNR
    if eff_cr is not None or eff_psnr is not None or TARGET_BEAT_SZ3:
        stop_tag = f"_Rmax{MAX_ROUNDS}"
        if eff_cr   is not None: stop_tag += f"_tcr{eff_cr}"
        if eff_psnr is not None: stop_tag += f"_tpsnr{eff_psnr:.2f}"
        if TARGET_BEAT_SZ3 and eff_psnr is None: stop_tag += "_beatSZ3"
    else:
        stop_tag = f"_R{N_ROUNDS}"
    tag = f"rd_{mode}_{KERNEL_TYPE}_ls{ls}_eb{eb}{stop_tag}"
    if mode == '2d':
        tag += f"_k{K_PER_ROUND_2D}"
    else:
        tag += f"_k{K_PER_ROUND_3D}"
    tag += f"_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}_zskip{Z_SKIP_BOTTOM}"
    if SENSOR_STORAGE != 'float32':
        tag += f"_sens{SENSOR_STORAGE}"
    tag += ".pkl"
    return ARGONNE / tag


def save_checkpoint(results: dict, vol: np.ndarray, mode: str, eb: float):
    path = checkpoint_path(mode, eb)
    with open(path, 'wb') as f:
        pickle.dump({'results': results, 'vol': vol}, f, protocol=4)
    print(f"  Checkpoint saved: {path.name}  ({path.stat().st_size/1e6:.0f} MB)")


def load_checkpoint(path: Path):
    with open(path, 'rb') as f:
        p = pickle.load(f)
    return p['results'], p['vol']


# ─────────────────────────────────────────────────────────────────────────────
# 2D PROGRESSIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive_2d(vol: np.ndarray, coords_2d: np.ndarray,
                       rng, eb: float,
                       target_psnr:    float | None = None,
                       target_cr:      float | None = None,
                       target_sz3_br:  float | None = None) -> dict:
    eff_target_psnr   = target_psnr   if target_psnr   is not None else TARGET_PSNR
    eff_target_cr     = target_cr     if target_cr     is not None else TARGET_CR
    eff_sz3_bit_rate  = target_sz3_br  # None unless TARGET_BEAT_SZ3
    accept_bound = ACCEPT_BINS * eb
    y_full  = vol.ravel().astype(np.float64)
    y_mean, y_std = y_full.mean(), y_full.std()
    y_norm  = (y_full - y_mean) / y_std
    gp_noise = obs_noise_for_eb(eb)   # raised for 'eb_quant' to (EB/2)²

    print(f"  EB={eb}  accept_bound={accept_bound:.4f}  "
          f"sensor_storage={SENSOR_STORAGE}  gp_noise={gp_noise:.2e}")

    # Per-slice state
    mask_avail_s  = [np.ones(N_SLICE, dtype=bool)  for _ in range(NZ)]
    all_sens_s    = [[]                             for _ in range(NZ)]
    mu_s          = [np.full(N_SLICE, y_mean)       for _ in range(NZ)]
    err_s         = [np.zeros(N_SLICE)              for _ in range(NZ)]
    settled_err_s = [np.full(N_SLICE, np.nan)       for _ in range(NZ)]

    rounds = []
    use_target = (eff_target_psnr is not None or eff_target_cr is not None
                  or eff_sz3_bit_rate is not None)
    n_rounds_to_run = MAX_ROUNDS if use_target else N_ROUNDS
    data_range = float(y_full.max() - y_full.min())

    for r in range(n_rounds_to_run):
        t0 = time.perf_counter()
        print(f"  ── Round {r+1}/{n_rounds_to_run}", end=' ', flush=True)

        round_sensor_idx = []

        for iz in range(NZ):
            sl_off    = iz * N_SLICE
            cand_loc  = np.where(mask_avail_s[iz])[0]
            if len(cand_loc) == 0:
                continue

            y_norm_sl  = y_norm[sl_off : sl_off + N_SLICE]
            y_full_sl  = y_full[sl_off : sl_off + N_SLICE]
            coords_cand = coords_2d[cand_loc]
            err_abs     = np.abs(err_s[iz][cand_loc])
            k_req       = min(K_PER_ROUND_2D, len(cand_loc))

            use_rpgks = (SENSOR_STRATEGY == 'rpgks'
                         or (SENSOR_STRATEGY == 'hybrid' and r < N_RPGKS_ROUNDS))

            if use_rpgks:
                loc_s = select_sensors(coords_cand, k_req, rng, kern_fn=kernel_2d)
            elif SENSOR_STRATEGY == 'max_residual':
                loc_s = np.argsort(err_abs)[::-1][:k_req]
            else:  # rpgks_residual or hybrid exploitation rounds
                thresh   = np.percentile(err_abs, 100.0 - RESID_PERCENTILE)
                hi       = np.where(err_abs >= thresh)[0]
                k_sel    = min(k_req, len(hi))
                if len(hi) >= k_req:
                    sub   = select_sensors(coords_cand[hi], k_sel, rng, kern_fn=kernel_2d)
                    loc_s = hi[sub]
                else:
                    loc_s = hi

            global_s = cand_loc[loc_s]
            round_sensor_idx.extend((global_s + sl_off).tolist())
            all_sens_s[iz].extend(global_s.tolist())

            sens_flat = np.array(all_sens_s[iz]) if ACCUMULATE else global_s
            # Use stored (quantized) sensor values so encoder/decoder agree
            y_stored_norm = ((quantize_sensor_vals(y_full_sl[sens_flat], eb)
                              - y_mean) / y_std)
            mu_norm, _ = gp_predict_2d(coords_2d[sens_flat],
                                       y_stored_norm, coords_2d, noise=gp_noise)
            mu_s[iz]   = mu_norm * y_std + y_mean
            err_s[iz]  = y_full_sl - mu_s[iz]

            mask_avail_s[iz][global_s] = False
            comp_loc = cand_loc[np.abs(err_s[iz][cand_loc]) < accept_bound]
            comp_loc = comp_loc[~np.isin(comp_loc, global_s)]
            mask_avail_s[iz][comp_loc] = False
            settled_err_s[iz][global_s] = 0.0
            settled_err_s[iz][comp_loc] = err_s[iz][comp_loc]

        # Build full-domain error array
        full_err = np.full(N, np.nan)
        for iz in range(NZ):
            sl  = slice(iz * N_SLICE, (iz+1) * N_SLICE)
            full_err[sl] = settled_err_s[iz]
            cand = np.where(mask_avail_s[iz])[0]
            full_err[sl][cand] = err_s[iz][cand]

        n_done     = N - sum(m.sum() for m in mask_avail_s)
        frac_cumul = 100.0 * n_done / N

        rounds.append({
            'round'      : r + 1,
            'sensor_idx' : np.array(round_sensor_idx, dtype=np.int64),
            'err_vals'   : full_err,
            'frac_comp'  : frac_cumul,
        })

        # ── Target-based early stopping ───────────────────────────────────────
        # Estimate is computed correctly:
        #   - H only over non-sensor voxels (sensor zeros must not deflate H)
        #   - bit_rate = (sensor_bits + H * N_resid) / N
        stop = False
        stop_reason = ''
        if use_target:
            all_idx_so_far = np.concatenate([
                np.array(all_sens_s[iz]) + iz * N_SLICE
                for iz in range(NZ) if all_sens_s[iz]
            ])
            k_so_far = len(all_idx_so_far)
            idx_b, val_b = sensor_storage_bits(all_idx_so_far,
                                               y_full[all_idx_so_far], eb)
            # Correct H: exclude sensor locations from residual entropy
            sm_est = np.zeros(N, dtype=bool)
            sm_est[all_idx_so_far] = True
            N_resid_est = N - k_so_far
            resid_est   = full_err[~sm_est]
            q_bin_est, q_err_resid_est = quantize_sz2(resid_est, eb)
            H_est    = entropy_from_bins(q_bin_est)
            # Sensor errors (stored exactly up to eb_quant error)
            sens_q_err = quantize_sz2(full_err[sm_est], eb)[1]
            all_q_err_est = np.empty(N)
            all_q_err_est[~sm_est] = q_err_resid_est
            all_q_err_est[sm_est]  = sens_q_err
            br_est   = (idx_b + val_b + H_est * N_resid_est) / N
            cr_est   = 32.0 / br_est
            psnr_est = psnr_db(all_q_err_est, data_range)
            print(f"done={frac_cumul:.1f}%  CR≈{cr_est:.2f}×  PSNR≈{psnr_est:.1f} dB"
                  f"  BR≈{br_est:.3f}  [{time.perf_counter()-t0:.1f}s]")
            if eff_target_cr is not None and cr_est >= eff_target_cr:
                stop = True; stop_reason = f"CR={cr_est:.2f}≥{eff_target_cr}"
            if eff_target_psnr is not None and psnr_est >= eff_target_psnr:
                stop = True; stop_reason = f"PSNR={psnr_est:.1f}≥{eff_target_psnr:.1f}"
            # TARGET_BEAT_SZ3: stop when our bit_rate < SZ3 bit_rate at this EB
            if eff_sz3_bit_rate is not None and br_est <= eff_sz3_bit_rate:
                stop = True; stop_reason = f"BR={br_est:.3f}≤SZ3={eff_sz3_bit_rate:.3f}"
        else:
            print(f"done={frac_cumul:.1f}%  [{time.perf_counter()-t0:.1f}s]")

        if stop:
            print(f"  Target reached: {stop_reason}  (round {r+1})")
            break

    # Final prediction using ALL sensors
    all_sens_global = np.concatenate([
        np.array(all_sens_s[iz]) + iz * N_SLICE
        for iz in range(NZ) if all_sens_s[iz]
    ])
    mu_decomp = np.empty(N)
    for iz in range(NZ):
        sl = slice(iz * N_SLICE, (iz+1) * N_SLICE)
        if not all_sens_s[iz]:
            mu_decomp[sl] = y_mean; continue
        sf        = np.array(all_sens_s[iz])
        y_full_sl = y_full[sl]
        # Decoder uses stored (quantized) values — same as what was transmitted
        y_stored_norm = ((quantize_sensor_vals(y_full_sl[sf], eb)
                          - y_mean) / y_std)
        mn, _     = gp_predict_2d(coords_2d[sf], y_stored_norm, coords_2d,
                                  noise=gp_noise)
        mu_decomp[sl] = mn * y_std + y_mean
    err_decomp = y_full - mu_decomp

    return dict(rounds=rounds, all_sens=all_sens_global,
                err_decomp=err_decomp, y_full=y_full)


# ─────────────────────────────────────────────────────────────────────────────
# 3D PROGRESSIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive_3d(vol: np.ndarray, coords_3d: np.ndarray,
                       rng, eb: float,
                       target_psnr: float | None = None,
                       target_cr:   float | None = None) -> dict:
    eff_target_psnr = target_psnr if target_psnr is not None else TARGET_PSNR
    eff_target_cr   = target_cr   if target_cr   is not None else TARGET_CR
    accept_bound = ACCEPT_BINS * eb
    y_full  = vol.ravel().astype(np.float64)
    y_mean, y_std = y_full.mean(), y_full.std()
    y_norm  = (y_full - y_mean) / y_std
    gp_noise = obs_noise_for_eb(eb)

    print(f"  EB={eb}  accept_bound={accept_bound:.4f}  "
          f"sensor_storage={SENSOR_STORAGE}  gp_noise={gp_noise:.2e}")

    mask_avail       = np.ones(N, dtype=bool)
    all_sensor_list  = []
    settled_err      = np.full(N, np.nan)
    err_all          = np.zeros(N)
    mu_all           = y_full.copy()
    rounds           = []
    use_target       = eff_target_psnr is not None or eff_target_cr is not None
    n_rounds_to_run  = MAX_ROUNDS if use_target else N_ROUNDS
    data_range       = float(y_full.max() - y_full.min())

    for r in range(n_rounds_to_run):
        t0 = time.perf_counter()
        print(f"  ── Round {r+1}/{n_rounds_to_run}", end=' ', flush=True)

        cand_idx = np.where(mask_avail)[0]
        if len(cand_idx) == 0:
            print("all voxels settled — stopping early")
            break

        coords_cand = coords_3d[cand_idx]
        err_abs     = np.abs(err_all[cand_idx])

        use_rpgks = (SENSOR_STRATEGY == 'rpgks'
                     or (SENSOR_STRATEGY == 'hybrid' and r < N_RPGKS_ROUNDS))
        k_req = min(K_PER_ROUND_3D, len(cand_idx))

        if use_rpgks:
            loc_s = select_sensors(coords_cand, k_req, rng, kern_fn=kernel_3d)
        elif SENSOR_STRATEGY == 'max_residual':
            loc_s = np.argsort(err_abs)[::-1][:k_req]
        else:
            thresh   = np.percentile(err_abs, 100.0 - RESID_PERCENTILE)
            hi       = np.where(err_abs >= thresh)[0]
            k_sel    = min(k_req, len(hi))
            if len(hi) >= k_req:
                sub   = select_sensors(coords_cand[hi], k_sel, rng, kern_fn=kernel_3d)
                loc_s = hi[sub]
            else:
                loc_s = hi

        global_s = cand_idx[loc_s]
        all_sensor_list.append(global_s)

        sens_idx = np.concatenate(all_sensor_list) if ACCUMULATE else global_s
        y_stored_norm = ((quantize_sensor_vals(y_full[sens_idx], eb)
                          - y_mean) / y_std)
        if LOCAL_GP:
            mu_norm, _ = gp_predict_local_3d(
                coords_3d[sens_idx], y_stored_norm, coords_3d, noise=gp_noise)
        else:
            mu_norm, _ = gp_predict_global_3d(
                coords_3d[sens_idx], y_stored_norm, coords_3d, noise=gp_noise)

        mu_all  = mu_norm * y_std + y_mean
        err_all = y_full - mu_all

        # Compression
        comp_mask  = np.abs(err_all[cand_idx]) < accept_bound
        comp_global = cand_idx[comp_mask]
        comp_global = comp_global[~np.isin(comp_global, global_s)]

        # Build histogram error array
        hist_err               = settled_err.copy()
        hist_err[cand_idx]     = err_all[cand_idx]
        hist_err[global_s]     = 0.0

        mask_avail[global_s]   = False
        mask_avail[comp_global] = False
        settled_err[global_s]  = 0.0
        settled_err[comp_global] = err_all[comp_global]

        frac_cumul = 100.0 * (~mask_avail).sum() / N

        rounds.append({
            'round'      : r + 1,
            'sensor_idx' : global_s,
            'err_vals'   : hist_err,
            'frac_comp'  : frac_cumul,
        })

        # ── Target-based early stopping ───────────────────────────────────────
        stop = False
        stop_reason = ''
        if use_target:
            all_sens_so_far = np.concatenate(all_sensor_list)
            idx_b, val_b = sensor_storage_bits(all_sens_so_far,
                                               y_full[all_sens_so_far], eb)
            q_bin, q_err = quantize_sz2(hist_err, eb)
            H_est    = entropy_from_bins(q_bin)
            br_est   = (idx_b + val_b + H_est * N) / N
            cr_est   = 32.0 / br_est
            psnr_est = psnr_db(q_err, data_range)
            print(f"done={frac_cumul:.1f}%  CR≈{cr_est:.2f}×  PSNR≈{psnr_est:.1f} dB"
                  f"  [{time.perf_counter()-t0:.1f}s]")
            if eff_target_cr   is not None and cr_est   >= eff_target_cr:
                stop = True; stop_reason = f"CR={cr_est:.2f}≥{eff_target_cr}"
            if eff_target_psnr is not None and psnr_est >= eff_target_psnr:
                stop = True; stop_reason = f"PSNR={psnr_est:.1f}≥{eff_target_psnr:.1f}"
        else:
            print(f"done={frac_cumul:.1f}%  [{time.perf_counter()-t0:.1f}s]")

        if stop:
            print(f"  Target reached: {stop_reason}  (round {r+1})")
            break

    # Final decompression (decoder uses stored/quantized sensor values)
    all_sens = np.concatenate(all_sensor_list)
    y_stored_norm = ((quantize_sensor_vals(y_full[all_sens], eb)
                      - y_mean) / y_std)
    if LOCAL_GP:
        mu_norm, _ = gp_predict_local_3d(
            coords_3d[all_sens], y_stored_norm, coords_3d, noise=gp_noise)
    else:
        mu_norm, _ = gp_predict_global_3d(
            coords_3d[all_sens], y_stored_norm, coords_3d, noise=gp_noise)
    mu_decomp  = mu_norm * y_std + y_mean
    err_decomp = y_full - mu_decomp

    return dict(rounds=rounds, all_sens=all_sens,
                err_decomp=err_decomp, y_full=y_full)


# ─────────────────────────────────────────────────────────────────────────────
# SZ3 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sz3(vol: np.ndarray, eb: float) -> dict | None:
    if not SZ3_BIN.exists():
        print(f"  SZ3 binary not found: {SZ3_BIN}"); return None
    vol32      = vol.astype(np.float32)
    nz, ny, nx = vol32.shape
    N          = vol32.size
    data_range = float(vol.max() - vol.min())

    with tempfile.TemporaryDirectory() as tmp:
        tmp   = Path(tmp)
        f_in  = tmp / "vol.f32"
        f_sz  = tmp / "vol.sz3"
        f_dec = tmp / "vol_dec.f32"
        vol32.tofile(f_in)
        cmd = [str(SZ3_BIN), "-f",
               "-i", str(f_in), "-z", str(f_sz), "-o", str(f_dec),
               "-3", str(nx), str(ny), str(nz), "-M", "ABS", str(eb)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  SZ3 failed at EB={eb}: {res.stderr.strip()}"); return None
        if not f_sz.exists():
            alt = f_in.with_suffix('.f32.sz3')
            if alt.exists(): alt.rename(f_sz)
            else: print(f"  SZ3 no output at EB={eb}"); return None
        compressed_bytes = f_sz.stat().st_size
        vol_dec = (np.fromfile(f_dec, dtype=np.float32)
                   .reshape(vol32.shape).astype(np.float64))

    errors   = vol.astype(np.float64) - vol_dec
    q_bin, q_err = quantize_sz2(errors, eb)
    H        = entropy_from_bins(q_bin)
    bit_rate = (compressed_bytes * 8) / N
    psnr     = psnr_db(q_err, data_range)
    rmse     = float(np.sqrt(np.mean(q_err**2)))
    cr = N * 4 / compressed_bytes
    print(f"  SZ3 EB={eb}: {bit_rate:.3f} bits  CR={cr:.2f}×  PSNR={psnr:.2f} dB  RMSE={rmse:.5f}")
    return dict(eb=eb, bit_rate=bit_rate, cr=cr, H=H, psnr=psnr, rmse=rmse)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED METRICS
# ─────────────────────────────────────────────────────────────────────────────

def quantize_sz2(errs: np.ndarray, eb: float):
    safe        = np.where(np.isnan(errs), 0.0, errs)
    q_bin       = np.round(safe / (2.0 * eb)).astype(np.int64)
    q_final_err = safe - q_bin * (2.0 * eb)
    return q_bin, q_final_err


def entropy_from_bins(q_bin: np.ndarray) -> float:
    counts = np.bincount((q_bin - q_bin.min()).ravel())
    p      = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def psnr_db(errors: np.ndarray, data_range: float) -> float:
    rmse = float(np.sqrt(np.mean(errors**2)))
    return 20.0 * np.log10(data_range / rmse) if rmse > 0 else float('inf')


def quantize_sensor_vals(vals: np.ndarray, eb: float) -> np.ndarray:
    """
    Return sensor values as they would be stored on disk and used by the decoder.

    'float32'  → round to float32 precision (≈ exact for this dataset)
    'float16'  → round to float16 precision (error ≈ val × 2⁻¹⁰)
    'eb_quant' → quantize to EB bins (error ≤ EB/2)
    """
    if SENSOR_STORAGE == 'float32':
        return vals.astype(np.float32).astype(np.float64)
    elif SENSOR_STORAGE == 'float16':
        return vals.astype(np.float16).astype(np.float64)
    else:  # 'eb_quant'
        return np.round(vals / eb) * eb


def obs_noise_for_eb(eb: float) -> float:
    """
    GP observation-noise variance that accounts for sensor quantization error.
    For 'eb_quant', raise noise to at least (EB/2)² so the GP knows sensors
    are imprecise.  For float32/float16 the quantization is negligible.
    """
    if SENSOR_STORAGE == 'eb_quant':
        return max(NOISE, (eb / 2.0) ** 2)
    return NOISE


def sensor_storage_bits(all_sensor_idx: np.ndarray,
                        all_sensor_vals: np.ndarray,
                        eb: float) -> tuple[float, float]:
    """
    Compute total bits to store sensor positions and values.

    Positions — delta-encoded sorted indices, entropy-coded (all strategies)
    ─────────────────────────────────────────────────────────────────────────
    Sort sensor indices, compute gaps, H(gaps) × k bits.  Mirrors how
    SZ2/SZ3 entropy-codes positions of its unpredictable points.

    Values — depends on SENSOR_STORAGE
    ────────────────────────────────────
    'float32'  : 32 bits/sensor fixed  (SZ2/SZ3 baseline)
    'float16'  : 16 bits/sensor fixed
    'eb_quant' : entropy of EB-quantized bin indices × k bits
                 (GP uses these same quantized values; error ≤ EB/2 ≤ EB)

    Returns (index_bits, value_bits).
    """
    k = len(all_sensor_idx)
    if k == 0:
        return 0.0, 0.0

    # ── Index bits: entropy of delta-encoded sorted indices ───────────────────
    sorted_idx = np.sort(all_sensor_idx).astype(np.int64)
    deltas     = np.diff(np.concatenate([[-1], sorted_idx])).astype(np.int64)
    counts     = np.bincount((deltas - 1).astype(np.int64))
    p          = counts[counts > 0] / counts.sum()
    H_delta    = float(-np.sum(p * np.log2(p)))
    index_bits = H_delta * k

    # ── Value bits ────────────────────────────────────────────────────────────
    if SENSOR_STORAGE == 'float32':
        value_bits = 32.0 * k
    elif SENSOR_STORAGE == 'float16':
        value_bits = 16.0 * k
    else:  # 'eb_quant'
        q_sens   = np.round(all_sensor_vals / eb).astype(np.int64)
        shifted  = q_sens - q_sens.min()
        counts_v = np.bincount(shifted)
        p_v      = counts_v[counts_v > 0] / counts_v.sum()
        H_sens   = float(-np.sum(p_v * np.log2(p_v)))
        value_bits = H_sens * k

    return index_bits, value_bits


def gp_rd_point(results: dict, vol: np.ndarray, eb: float) -> dict:
    """
    Compute (bit_rate, PSNR, RMSE) for a GP checkpoint at a given EB.

    Sensor storage matches SZ2/SZ3 unpredictable-point format:
      - values : 32 bits/sensor (float32)
      - indices: entropy of delta-encoded sorted positions (not fixed 32 bits)
    Residual storage: H(q_bin) bits/sample (entropy of quantised bin indices),
    same as SZ2/SZ3's entropy-coded quantised stream.

    bit_rate = (index_bits + value_bits + H × N) / N
    """
    y_full     = vol.ravel().astype(np.float64)
    data_range = float(y_full.max() - y_full.min())
    N          = y_full.size

    # Collect all sensor indices and their true values
    all_sensor_idx  = np.concatenate([rd['sensor_idx'] for rd in results['rounds']])
    all_sensor_vals = y_full[all_sensor_idx]
    k_total         = len(all_sensor_idx)

    # Sensor storage: delta-coded indices + EB-quantized entropy-coded values
    index_bits, value_bits = sensor_storage_bits(all_sensor_idx, all_sensor_vals, eb)
    sensor_bits = index_bits + value_bits

    # Residual entropy — use full_err from the final round (rounds[-1]['err_vals']).
    # This captures the genuine progressive-coding benefit: voxels that settled
    # early have errors bounded by ACCEPT_BINS×EB (small), which legitimately
    # lowers H.  The err_decomp alternative (final GP refit on all sensors) gives
    # larger errors for early-settled voxels and loses this advantage.
    #
    # The only real bug in the original code was including sensor locations (error=0)
    # in H, which artificially deflated it.  Fix: exclude sensor voxels from H,
    # multiply by N_resid (not N).
    errs = results['rounds'][-1]['err_vals']
    sensor_mask  = np.zeros(N, dtype=bool)
    sensor_mask[all_sensor_idx] = True
    N_resid      = N - k_total          # voxels in residual stream
    resid_errs   = errs[~sensor_mask]   # exclude sensor locations
    q_bin, q_err_resid = quantize_sz2(resid_errs, eb)
    H            = entropy_from_bins(q_bin)

    # PSNR over ALL voxels
    sensor_errs  = errs[sensor_mask]
    all_q_err    = np.empty(N)
    all_q_err[~sensor_mask] = q_err_resid
    all_q_err[sensor_mask]  = quantize_sz2(sensor_errs, eb)[1]

    bit_rate = (sensor_bits + H * N_resid) / N
    psnr     = psnr_db(all_q_err, data_range)
    rmse     = float(np.sqrt(np.mean(all_q_err**2)))

    val_label = (f"val={value_bits/N:.3f} b/s [{SENSOR_STORAGE}]")
    print(f"  GP   EB={eb}: {bit_rate:.3f} bits  PSNR={psnr:.2f} dB  "
          f"RMSE={rmse:.5f}  sensors={k_total:,}  "
          f"idx={index_bits/N:.3f} b/s  {val_label}  H={H:.3f} b/s  "
          f"(H over {N_resid:,} non-sensor voxels)")
    return dict(eb=eb, bit_rate=bit_rate, cr=32.0/bit_rate, H=H,
                psnr=psnr, rmse=rmse, k_total=k_total,
                index_bits_per_sample=index_bits/N,
                value_bits_per_sample=value_bits/N)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_rd_curves(gp_pts: list[dict], sz3_pts: list[dict], mode: str) -> None:
    GP_COLOR  = '#1f77b4'
    SZ3_COLOR = '#d62728'

    # CR = 32 / bit_rate  (original is float32 = 32 bits/sample)
    # Sort by bit_rate so curves connect monotonically (SZ3 may not be in EB order)
    gp_pts_s  = sorted(gp_pts,  key=lambda p: p['bit_rate'])
    sz3_pts_s = sorted(sz3_pts, key=lambda p: p['bit_rate'])

    gp_br    = [p['bit_rate']        for p in gp_pts_s]
    gp_cr    = [32.0 / p['bit_rate'] for p in gp_pts_s]
    gp_psnr  = [p['psnr']            for p in gp_pts_s]
    gp_rmse  = [p['rmse']            for p in gp_pts_s]
    gp_eb    = [p['eb']              for p in gp_pts_s]
    sz3_br   = [p['bit_rate']        for p in sz3_pts_s]
    sz3_cr   = [32.0 / p['bit_rate'] for p in sz3_pts_s]
    sz3_psnr = [p['psnr']            for p in sz3_pts_s]
    sz3_rmse = [p['rmse']            for p in sz3_pts_s]
    sz3_eb   = [p['eb']              for p in sz3_pts_s]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    mode_label = '2D Slice-by-Slice GP' if mode == '2d' else '3D GP'
    k_label = f"k={K_PER_ROUND_2D}/slice" if mode == '2d' else f"k={K_PER_ROUND_3D}"
    fig.suptitle(
        f"Rate-Distortion: {mode_label} vs SZ3  (U-wind, ISABEL)\n"
        f"DS={DOWNSAMPLE}× ({NZ}×{NY}×{NX})  kernel={KERNEL_TYPE}  LS_XY={LS_XY}"
        + (f"  LS_Z={LS_Z}" if mode == '3d' else "")
        + f"  {k_label}  accept_bins={ACCEPT_BINS}  sens={SENSOR_STORAGE}",
        fontsize=10, fontweight='bold')

    panels = [
        (axes[0], gp_cr,   sz3_cr,   gp_psnr, sz3_psnr,
         'Compression Ratio (×)', 'PSNR (dB)', 'CR vs PSNR  (↑↑ better)'),
        (axes[1], gp_br,   sz3_br,   gp_psnr, sz3_psnr,
         'Bit rate (bits/sample)', 'PSNR (dB)', 'Bit Rate vs PSNR  (→↑ better)'),
        (axes[2], gp_br,   sz3_br,   gp_rmse, sz3_rmse,
         'Bit rate (bits/sample)', 'RMSE (m/s)', 'Bit Rate vs RMSE  (→↓ better)'),
    ]

    _lbbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.95)
    # GP labels alternate between two tiers above; SZ3 between two tiers below.
    # Using fixed alternating tiers (not proximity-based) guarantees separation.
    GP_TIERS  = [+22, +42]   # points/sample offset above the marker
    SZ3_TIERS = [-22, -42]   # offset below the marker

    def place_labels(ax, xs, ys, ebs, color, tiers, sign):
        if not xs:
            return
        va = 'bottom' if sign > 0 else 'top'
        for i, (x, y, eb) in enumerate(zip(xs, ys, ebs)):
            offset = tiers[i % 2]
            ax.annotate(
                f'EB={eb}', xy=(x, y),
                xytext=(0, offset), textcoords='offset points',
                fontsize=7.5, color=color, ha='center', va=va, bbox=_lbbox,
                arrowprops=dict(arrowstyle='-', color=color, lw=0.6, alpha=0.5))

    for ax, gp_x, sz3_x, gp_y, sz3_y, xlabel, ylabel, title in panels:
        ax.plot(gp_x,  gp_y,  'o-',  color=GP_COLOR,  lw=2, ms=7, label=mode_label)
        ax.plot(sz3_x, sz3_y, 's--', color=SZ3_COLOR, lw=2, ms=7, label='SZ3')

        place_labels(ax, gp_x,  gp_y,  gp_eb,  GP_COLOR,  GP_TIERS,  +1)
        place_labels(ax, sz3_x, sz3_y, sz3_eb, SZ3_COLOR, SZ3_TIERS, -1)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25, ls='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.25)
        ax.set_box_aspect(1)   # force square axis box

    # Summary table
    sz3_by_eb  = {p['eb']: p for p in sz3_pts}
    col_labels = ['EB', 'GP CR', 'SZ3 CR', 'GP bits', 'SZ3 bits',
                  'GP PSNR', 'SZ3 PSNR', 'GP RMSE', 'SZ3 RMSE']
    rows = []
    for gp in gp_pts:
        eb  = gp['eb']
        sz3 = sz3_by_eb.get(eb, {})
        gp_cr_val  = 32.0 / gp['bit_rate']
        sz3_cr_val = 32.0 / sz3['bit_rate'] if sz3 else None
        rows.append([
            str(eb),
            f"{gp_cr_val:.2f}×",
            f"{sz3_cr_val:.2f}×" if sz3_cr_val else 'n/a',
            f"{gp['bit_rate']:.3f}",
            f"{sz3.get('bit_rate','n/a'):.3f}" if sz3 else 'n/a',
            f"{gp['psnr']:.2f} dB",
            f"{sz3.get('psnr','n/a'):.2f} dB" if sz3 else 'n/a',
            f"{gp['rmse']:.5f}",
            f"{sz3.get('rmse','n/a'):.5f}" if sz3 else 'n/a',
        ])

    fig.subplots_adjust(bottom=0.28, top=0.90, wspace=0.35)
    tbl_ax = fig.add_axes([0.05, 0.01, 0.9, 0.22])
    tbl_ax.axis('off')
    t = tbl_ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center')
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.5)
    for j in range(len(col_labels)):
        t[0, j].set_facecolor('#dce6f1')
        t[0, j].set_text_props(fontweight='bold')
    # Highlight better value per row:
    # CR (higher=better): cols 1,2  | bits (lower=better): cols 3,4
    # PSNR (higher=better): cols 5,6 | RMSE (lower=better): cols 7,8
    for ri, gp in enumerate(gp_pts, 1):
        sz3 = sz3_by_eb.get(gp['eb'])
        if not sz3: continue
        for gi, si, lower_is_better in [(1,2,False),(3,4,True),(5,6,False),(7,8,True)]:
            try:
                gv = float(rows[ri-1][gi].rstrip('×').rstrip(' dB'))
                sv = float(rows[ri-1][si].rstrip('×').rstrip(' dB'))
            except: continue
            win_gp = (gv < sv) if lower_is_better else (gv > sv)
            t[ri, gi if win_gp else si].set_facecolor('#d5f5e3')

    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {OUT_FIG}")


def plot_ls_sweep(ls_pts: list[dict], mode: str, eb: float) -> None:
    """Plot PSNR, RMSE, H, and bit_rate vs LS_XY for the lengthscale sweep."""
    ls_vals  = [p['ls_xy']   for p in ls_pts]
    psnrs    = [p['psnr']    for p in ls_pts]
    rmses    = [p['rmse']    for p in ls_pts]
    hs       = [p['H']       for p in ls_pts]
    brs      = [p['bit_rate']for p in ls_pts]

    mode_label = '2D Slice-by-Slice GP' if mode == '2d' else '3D GP'
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"LS_XY sweep — {mode_label}  |  Kernel={KERNEL_TYPE}  |  EB={eb}\n"
        f"K/slice/round={K_PER_ROUND_2D}  N_ROUNDS={N_ROUNDS}  DS={DOWNSAMPLE}",
        fontsize=11, fontweight='bold')

    COLOR = '#1f77b4'
    for ax, vals, ylabel, title, lower_better in [
        (axes[0], psnrs, 'PSNR (dB)',       'PSNR  (↑ better)',     False),
        (axes[1], rmses, 'RMSE (m/s)',       'RMSE  (↓ better)',     True),
        (axes[2], hs,    'H (bits/sample)',  'Entropy  (↓ better)',  True),
        (axes[3], brs,   'Bit rate (bits)',  'Bit rate  (↓ better)', True),
    ]:
        ax.plot(ls_vals, vals, 'o-', color=COLOR, lw=2, ms=8)
        best_idx = int(np.argmin(vals) if lower_better else np.argmax(vals))
        ax.scatter([ls_vals[best_idx]], [vals[best_idx]],
                   color='gold', s=120, zorder=5, edgecolors='black', lw=1.2,
                   label=f'Best: LS={ls_vals[best_idx]}')
        for x, y in zip(ls_vals, vals):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 6),
                        textcoords='offset points', fontsize=8, ha='center')
        ax.set_xlabel('LS_XY', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, ls='--')
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_LS_FIG, dpi=150, bbox_inches='tight')
    print(f"\nLS sweep figure saved: {OUT_LS_FIG}")

    # Print recommendation
    best_psnr_idx = int(np.argmax(psnrs))
    best_br_idx   = int(np.argmin(brs))
    print(f"\nLS sweep results at EB={eb}, kernel={KERNEL_TYPE}:")
    print(f"  Best PSNR : LS_XY={ls_vals[best_psnr_idx]}  ({psnrs[best_psnr_idx]:.2f} dB)")
    print(f"  Best H    : LS_XY={ls_vals[int(np.argmin(hs))]}  ({min(hs):.4f} bits)")
    print(f"  Best rate : LS_XY={ls_vals[best_br_idx]}  ({brs[best_br_idx]:.4f} bits/sample)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Mode: {MODE}  |  RUN_MODE: {RUN_MODE}  |  "
          f"Grid: {NZ}×{NY}×{NX} = {N:,} voxels")

    print("Loading wind field …")
    vol    = load_wind()
    coords = build_coords_2d() if MODE == '2d' else build_coords_3d()

    # ── LS SWEEP ──────────────────────────────────────────────────────────────
    if RUN_MODE == 'ls':
        ls_pts = []
        for ls in LS_XY_VALUES:
            print(f"\n{'═'*60}")
            print(f"  LS_XY = {ls}  EB = {LS_EB}  kernel = {KERNEL_TYPE}")
            print(f"{'═'*60}")

            # Override the global LS_XY for this iteration
            LS_XY = ls   # noqa: F811

            ckpt = checkpoint_path(MODE, LS_EB, ls_xy=ls)
            if ckpt.exists():
                print(f"  Loading checkpoint: {ckpt.name}")
                results, _ = load_checkpoint(ckpt)
            else:
                print(f"  Running {MODE.upper()} GP …")
                rng_ls = np.random.default_rng(SEED + int(ls * 10000))
                if MODE == '2d':
                    results = run_progressive_2d(vol, coords, rng_ls, LS_EB)
                else:
                    results = run_progressive_3d(vol, coords, rng_ls, LS_EB)
                # save with ls_xy encoded in filename (LS_XY already overridden above)
                with open(ckpt, 'wb') as f:
                    pickle.dump({'results': results, 'vol': vol}, f, protocol=4)
                print(f"  Checkpoint saved: {ckpt.name}")

            pt = gp_rd_point(results, vol, LS_EB)
            pt['ls_xy'] = ls
            ls_pts.append(pt)

        plot_ls_sweep(ls_pts, MODE, LS_EB)
        print("Done.")
        import sys; sys.exit(0)

    # ── RD CURVE ──────────────────────────────────────────────────────────────
    gp_pts  = []
    sz3_pts = []

    # If TARGET_BEAT_SZ3, run all SZ3 points first so we have per-EB PSNR targets
    if TARGET_BEAT_SZ3:
        print("\n── Running SZ3 for all EBs first (needed for per-EB PSNR target) ──")
        for eb in EB_VALUES:
            sz3 = run_sz3(vol, eb)
            if sz3:
                sz3_pts.append(sz3)
        sz3_by_eb = {p['eb']: p for p in sz3_pts}

    for eb in EB_VALUES:
        print(f"\n{'═'*60}")
        print(f"  EB = {eb}")
        print(f"{'═'*60}")

        # Per-EB target: beat SZ3 bit_rate if TARGET_BEAT_SZ3, else global targets
        eb_target_psnr  = None
        eb_target_sz3br = None
        if TARGET_BEAT_SZ3:
            sz3_pt = sz3_by_eb.get(eb, {})
            eb_target_sz3br = sz3_pt.get('bit_rate', None)
            if eb_target_sz3br is not None:
                print(f"  GP target: bit_rate < {eb_target_sz3br:.3f} b/s  "
                      f"(SZ3 CR={sz3_pt.get('cr',0):.2f}×  PSNR={sz3_pt.get('psnr',0):.2f} dB)")
            else:
                print(f"  GP target: none (SZ3 not available at this EB)")
        else:
            eb_target_psnr = TARGET_PSNR

        # ── GP ────────────────────────────────────────────────────────────
        ckpt = checkpoint_path(MODE, eb, target_psnr_override=eb_target_psnr)
        if ckpt.exists():
            print(f"  Loading GP checkpoint: {ckpt.name}")
            results, _ = load_checkpoint(ckpt)
        else:
            print(f"  Running {MODE.upper()} progressive GP …")
            rng_eb = np.random.default_rng(SEED + int(eb * 10000))
            if MODE == '2d':
                results = run_progressive_2d(vol, coords, rng_eb, eb,
                                             target_psnr=eb_target_psnr,
                                             target_cr=TARGET_CR,
                                             target_sz3_br=eb_target_sz3br)
            else:
                results = run_progressive_3d(vol, coords, rng_eb, eb,
                                             target_psnr=eb_target_psnr,
                                             target_cr=TARGET_CR)
            ckpt_path = checkpoint_path(MODE, eb, target_psnr_override=eb_target_psnr)
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'results': results, 'vol': vol}, f, protocol=4)
            print(f"  Checkpoint saved: {ckpt_path.name}")

        gp_pts.append(gp_rd_point(results, vol, eb))

        # ── SZ3 (already done if TARGET_BEAT_SZ3, else run now) ───────────
        if not TARGET_BEAT_SZ3:
            print(f"  Running SZ3 at EB={eb} …")
            sz3 = run_sz3(vol, eb)
            if sz3:
                sz3_pts.append(sz3)

    print(f"\n{'═'*60}")
    print("  Plotting rate-distortion curves …")
    plot_rd_curves(gp_pts, sz3_pts, MODE)
    print("Done.")
