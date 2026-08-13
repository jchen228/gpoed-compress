#!/usr/bin/env python3
"""
hurricane_progressive_gp_2d.py
================================
Slice-by-slice 2D progressive GP compression on the ISABEL hurricane U-wind field.

Each horizontal z-slice is treated as an independent 2D regression problem.
Within each slice, N_ROUNDS rounds of progressive GP sensing + compression are
run using a 2D anisotropic Matérn kernel over (y, x) coordinates only.

WHY 2D INSTEAD OF 3D
---------------------
The 3D GP must simultaneously model correlations across z, y and x.  With a
fixed sensor budget spread over NZ slices, each slice receives very few sensors
(K_3D / NZ ≈ 2–3).  The 2D approach dedicates K_PER_ROUND_2D sensors per slice
per round, so each slice has much denser coverage within its (NY × NX) plane.
Fine horizontal structure (e.g. the eye wall spiral) is easier to capture when
the kernel and sensor selection work in the same 2-D space as the feature.

The trade-off: vertical correlations between adjacent z-levels are ignored.
If the field is strongly correlated across z (which it often is), the 3D GP
can exploit that information for "free" whereas the 2D approach cannot.

CHECKPOINT FORMAT
-----------------
Saved in the same format as hurricane_progressive_gp.py so that
sz3_gp_comparison.py can load and compare both approaches.
"""

from __future__ import annotations
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ARGONNE   = Path(__file__).resolve().parent
DATA_DIR  = ARGONNE / "100x500x500"
DATA_FILE = DATA_DIR / "Uf48.bin.f32"

# ── Algorithm knobs ───────────────────────────────────────────────────────────
N_ROUNDS         = 10    # progressive rounds per slice
K_PER_ROUND_2D   = 10    # sensors per slice per round
                          # total sensors = K_PER_ROUND_2D × NZ × N_ROUNDS
                          # compare to 3D: K_PER_ROUND × N_ROUNDS
ERROR_BOUND      = 0.01  # quantiser bin half-width (m/s)
ACCEPT_BINS      = 10    # voxels with |error| < ACCEPT_BINS×EB are compressed
DOWNSAMPLE       = 10    # spatial downsampling (same as QUICK_TEST in 3D script)
ACCUMULATE       = True  # accumulate sensors across rounds within each slice

# ── Sensor placement strategy ─────────────────────────────────────────────────
# Same options as 3D script: 'rpgks', 'max_residual', 'rpgks_residual', 'hybrid'
SENSOR_STRATEGY  = 'hybrid'
N_RPGKS_ROUNDS   = 1
RESID_PERCENTILE = 25

# ── Z-layer exclusion ─────────────────────────────────────────────────────────
Z_SKIP_BOTTOM = 0    # 0 = use all 100 z-levels

# ── Kernel ────────────────────────────────────────────────────────────────────
# 'matern52', 'matern32', 'matern12'  (2D only — no z component)
KERNEL_TYPE = 'matern12'
LS_XY  = 0.15   # horizontal correlation length in [0,1] coords
SIG2   = 1.0
NOISE  = 1e-3

# ── rpgks ─────────────────────────────────────────────────────────────────────
CHOL_RANK_OVERSAMPLE = 20

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED       = 42
BATCH_PRED = 10_000   # prediction batch size per slice

# ── Derived ───────────────────────────────────────────────────────────────────
ACCEPT_BOUND = ACCEPT_BINS * ERROR_BOUND
SZ2_UNPRED_BOUND = (65536 // 2) * (2 * ERROR_BOUND)

# ── Grid ──────────────────────────────────────────────────────────────────────
NZ_ORIG, NY_ORIG, NX_ORIG = 100, 500, 500
NZ = NZ_ORIG - Z_SKIP_BOTTOM
NY = -(-NY_ORIG // DOWNSAMPLE)   # ceiling division — matches numpy ::DOWNSAMPLE
NX = -(-NX_ORIG // DOWNSAMPLE)
N_SLICE = NY * NX                 # voxels per slice
N       = NZ * N_SLICE            # total voxels

# ── Output ────────────────────────────────────────────────────────────────────
CHECKPOINT_FILE = (ARGONNE /
    f"pgp2d_checkpoint_R{N_ROUNDS}_k{K_PER_ROUND_2D}"
    f"_eb{ERROR_BOUND}_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}_zskip{Z_SKIP_BOTTOM}.pkl")
OUT_HISTS = ARGONNE / "progressive_gp_2d_histograms.png"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_wind() -> np.ndarray:
    raw    = np.fromfile(DATA_FILE, dtype=np.float32)
    vol    = raw.reshape(NZ_ORIG, NY_ORIG, NX_ORIG)
    vol_ds = vol[Z_SKIP_BOTTOM:, ::DOWNSAMPLE, ::DOWNSAMPLE]
    return vol_ds.astype(np.float64)   # (NZ, NY, NX)


def build_coords_2d() -> np.ndarray:
    """(N_SLICE, 2) array of [y, x] coordinates in [0,1]²."""
    gy = np.linspace(0, 1, NY)
    gx = np.linspace(0, 1, NX)
    yy, xx = np.meshgrid(gy, gx, indexing='ij')   # (NY, NX)
    return np.stack([yy.ravel(), xx.ravel()], axis=1)   # (N_SLICE, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2D KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def _r2d(A: np.ndarray, B: np.ndarray):
    """Anisotropic 2D scaled distance components. A:(m,2), B:(n,2) → r2,r."""
    dy = (A[:, 0:1] - B[:, 0]) / LS_XY
    dx = (A[:, 1:2] - B[:, 1]) / LS_XY
    r2 = dy**2 + dx**2
    return r2, np.sqrt(np.maximum(r2, 0.0))


def kernel_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dispatched 2D kernel. A:(m,2), B:(n,2) → (m,n) covariance matrix."""
    r2, r = _r2d(A, B)
    if KERNEL_TYPE == 'matern52':
        return SIG2 * (1 + np.sqrt(5)*r + 5*r2/3) * np.exp(-np.sqrt(5)*r)
    elif KERNEL_TYPE == 'matern32':
        return SIG2 * (1 + np.sqrt(3)*r) * np.exp(-np.sqrt(3)*r)
    elif KERNEL_TYPE == 'matern12':
        return SIG2 * np.exp(-r)
    else:
        raise ValueError(f"Unknown KERNEL_TYPE: {KERNEL_TYPE!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2D GP PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def gp_predict_2d(coords_s, y_s, coords_p):
    """Standard GP posterior mean and variance for 2D slice."""
    k_s   = len(coords_s)
    K_ss  = kernel_2d(coords_s, coords_s) + NOISE * np.eye(k_s)
    L     = np.linalg.cholesky(K_ss)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_s))
    L_inv = np.linalg.solve(L, np.eye(k_s))

    n   = len(coords_p)
    mu  = np.empty(n)
    var = np.empty(n)
    for start in range(0, n, BATCH_PRED):
        end      = min(start + BATCH_PRED, n)
        K_bs     = kernel_2d(coords_p[start:end], coords_s)
        mu[start:end]  = K_bs @ alpha
        v = L_inv @ K_bs.T
        var[start:end] = np.maximum(SIG2 - np.sum(v**2, axis=0), 0.0)
    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# 2D SENSOR SELECTION  (rpgks)
# ─────────────────────────────────────────────────────────────────────────────

def select_sensors_2d(coords_cand: np.ndarray, k: int, rng) -> np.ndarray:
    """rpgks sensor selection in 2D. Returns indices into coords_cand."""
    n_cand = len(coords_cand)
    k      = min(k, n_cand)
    if k == n_cand:
        return np.arange(k)

    rank = min(k + CHOL_RANK_OVERSAMPLE, n_cand)

    # Randomly pivoted Cholesky
    F    = np.zeros((n_cand, rank))
    d    = np.array([kernel_2d(coords_cand[[i]], coords_cand[[i]])[0, 0]
                     for i in range(n_cand)])
    pivots = []
    for j in range(rank):
        p  = int(np.argmax(d))
        pivots.append(p)
        g  = kernel_2d(coords_cand, coords_cand[[p]]).ravel()
        if j > 0:
            g -= F[:, :j] @ F[p, :j]
        pivot_val = max(d[p], 1e-12)
        F[:, j]   = g / np.sqrt(pivot_val)
        d         = np.maximum(d - F[:, j]**2, 0.0)

    # GKS: SVD + pivoted QR to pick k columns
    U, _, _ = np.linalg.svd(F, full_matrices=False)   # (n_cand, rank)
    U_k     = U[:, :k]
    _, _, Pt = scipy_qr(U_k.T, pivoting=True)
    return Pt[:k]


# ─────────────────────────────────────────────────────────────────────────────
# PER-Z RMSE  (for hybrid strategy within each slice)
# ─────────────────────────────────────────────────────────────────────────────

def entropy_bits(errors: np.ndarray) -> float:
    """Shannon entropy (bits/sample) over SZ2-width bins (width=2×EB)."""
    vals = errors[~np.isnan(errors)]
    if len(vals) == 0:
        return 0.0
    half_range = 40.0 * ERROR_BOUND
    n_bins = max(int(np.ceil(2 * half_range / (2 * ERROR_BOUND))), 2)
    edges  = np.linspace(-half_range, half_range, n_bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROGRESSIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive_2d(vol: np.ndarray, coords_2d: np.ndarray, rng) -> dict:
    """
    Run N_ROUNDS of 2D progressive GP on every z-slice independently.

    Each round:
      For each z-slice iz:
        1. Select K_PER_ROUND_2D sensors from remaining candidates in that slice
        2. Fit 2D GP on accumulated sensors, predict full slice
        3. Compress voxels where |pred − true| < ACCEPT_BOUND

    Returns a results dict compatible with the 3D checkpoint format so that
    sz3_gp_comparison.py can load and plot both approaches side-by-side.
    """
    y_full  = vol.ravel()                        # (N,) all voxels
    y_mean  = y_full.mean()
    y_std   = y_full.std()
    y_norm  = (y_full - y_mean) / y_std

    print(f"\nField stats: mean={y_mean:.3f}  std={y_std:.3f}  "
          f"range=[{y_full.min():.2f}, {y_full.max():.2f}] m/s")
    print(f"Grid: NZ={NZ}  NY={NY}  NX={NX}  N_SLICE={N_SLICE}  N={N:,}")
    print(f"Sensors per slice per round: {K_PER_ROUND_2D}")
    print(f"Total sensors per round    : {K_PER_ROUND_2D * NZ}")
    print(f"Total sensors all rounds   : {K_PER_ROUND_2D * NZ * N_ROUNDS}\n")

    # Per-slice state
    # mask_avail_s[iz] : bool array (N_SLICE,) — True = still a candidate
    mask_avail_s = [np.ones(N_SLICE, dtype=bool) for _ in range(NZ)]
    # Accumulated sensors per slice (list of flat indices into slice)
    all_sens_s   = [[] for _ in range(NZ)]
    # Current prediction per slice (denormalised)
    mu_s         = [np.full(N_SLICE, y_mean) for _ in range(NZ)]
    err_s        = [np.full(N_SLICE, 0.0)    for _ in range(NZ)]
    var_s        = [np.full(N_SLICE, SIG2)   for _ in range(NZ)]
    # settled errors (for histogram)
    settled_err_s = [np.full(N_SLICE, np.nan) for _ in range(NZ)]

    rounds = []

    for r in range(N_ROUNDS):
        t_round = time.perf_counter()
        print(f"══ Round {r+1}/{N_ROUNDS} ═════════════════════════════════════")

        round_sensor_idx = []   # global flat indices of sensors this round

        for iz in range(NZ):
            slice_offset  = iz * N_SLICE
            cand_local    = np.where(mask_avail_s[iz])[0]

            if len(cand_local) == 0:
                continue

            coords_cand = coords_2d[cand_local]
            y_norm_slice = y_norm[slice_offset : slice_offset + N_SLICE]
            y_full_slice = y_full[slice_offset : slice_offset + N_SLICE]

            # ── Sensor selection ───────────────────────────────────────────
            err_cand_abs = np.abs(err_s[iz][cand_local])

            use_rpgks_full   = (SENSOR_STRATEGY == 'rpgks'
                or (SENSOR_STRATEGY == 'hybrid' and r < N_RPGKS_ROUNDS))
            use_max_resid    = (SENSOR_STRATEGY == 'max_residual')
            use_rpgks_resid  = (SENSOR_STRATEGY == 'rpgks_residual'
                or (SENSOR_STRATEGY == 'hybrid' and r >= N_RPGKS_ROUNDS))

            k_req = min(K_PER_ROUND_2D, len(cand_local))

            if use_rpgks_full:
                local_s = select_sensors_2d(coords_cand, k_req, rng)
            elif use_max_resid:
                local_s = np.argsort(err_cand_abs)[::-1][:k_req]
            elif use_rpgks_resid:
                thresh    = np.percentile(err_cand_abs, 100.0 - RESID_PERCENTILE)
                hi_local  = np.where(err_cand_abs >= thresh)[0]
                k_sel     = min(k_req, len(hi_local))
                if len(hi_local) >= k_req:
                    sub   = select_sensors_2d(coords_cand[hi_local], k_sel, rng)
                    local_s = hi_local[sub]
                else:
                    local_s = hi_local

            # Global flat indices for this slice's new sensors
            global_s = cand_local[local_s]         # indices into N_SLICE
            round_sensor_idx.extend(global_s + slice_offset)

            # ── Update accumulated sensors ─────────────────────────────────
            all_sens_s[iz].extend(global_s.tolist())
            sens_flat = np.array(all_sens_s[iz]) if ACCUMULATE else global_s

            # ── GP prediction for this slice ───────────────────────────────
            coords_s  = coords_2d[sens_flat]
            y_s_norm  = y_norm_slice[sens_flat]
            mu_norm, var_pred = gp_predict_2d(coords_s, y_s_norm, coords_2d)

            mu_s[iz]  = mu_norm * y_std + y_mean
            err_s[iz] = y_full_slice - mu_s[iz]
            var_s[iz] = var_pred

            # ── Compression ────────────────────────────────────────────────
            compressed_local = (np.abs(err_s[iz][cand_local]) < ACCEPT_BOUND)
            compressed_local[local_s] = False   # sensors not compressed
            comp_idx_local = cand_local[compressed_local]

            mask_avail_s[iz][comp_idx_local] = False
            mask_avail_s[iz][global_s]       = False

            settled_err_s[iz][global_s]      = 0.0
            settled_err_s[iz][comp_idx_local] = err_s[iz][comp_idx_local]

        # ── Round-level metrics ────────────────────────────────────────────
        # Build full-domain error array (NaN for voxels still in candidate pool)
        full_err_vals = np.full(N, np.nan)
        full_pred_vol = np.empty(N)
        full_var_vol  = np.empty(N)

        for iz in range(NZ):
            sl = slice(iz * N_SLICE, (iz+1) * N_SLICE)
            # settled errors
            full_err_vals[sl] = settled_err_s[iz]
            # candidates: fill in current prediction error
            cand  = np.where(mask_avail_s[iz])[0]
            full_err_vals[sl][cand] = err_s[iz][cand]
            full_pred_vol[sl] = mu_s[iz]
            full_var_vol[sl]  = var_s[iz]

        n_done  = N - sum(mask_avail_s[iz].sum() for iz in range(NZ))
        frac_cumul = 100.0 * n_done / N
        H = entropy_bits(full_err_vals)

        # Sensors this round as a flat array (for checkpoint compat)
        round_sensor_arr = np.array(round_sensor_idx, dtype=np.int64)

        # Build comp_idx for this round (all voxels newly compressed)
        # = voxels that are now unavailable minus this round's sensors
        prev_done = N - sum(mask_avail_s[iz].sum() for iz in range(NZ))
        # (already computed above as n_done)
        # Compute comp_idx: unavailable non-sensor voxels added this round
        # We track this via settled_err becoming non-NaN (non-sensor)
        # For checkpoint compat we just store the sensors — comp_idx reconstruction
        # from settled_err works fine in the comparison script.

        print(f"  Done: {n_done:,}/{N:,}  ({frac_cumul:.1f}%)  "
              f"H={H:.3f} bits  "
              f"sensors_this_round={len(round_sensor_arr):,}  "
              f"[{time.perf_counter()-t_round:.1f}s]")

        rounds.append({
            'round'      : r + 1,
            'sensor_idx' : round_sensor_arr,
            'comp_idx'   : np.where(~np.isnan(full_err_vals) &
                               np.isin(np.arange(N), round_sensor_arr,
                                       invert=True))[0],
            'err_vals'   : full_err_vals,
            'pred_vol'   : full_pred_vol,
            'err_vol'    : full_err_vals.copy(),
            'var_vol'    : full_var_vol,
            'n_comp'     : int(np.sum(~np.isnan(full_err_vals))) - len(round_sensor_arr),
            'frac_comp'  : frac_cumul,
            'frac_this'  : 0.0,   # not tracked per-round in 2D mode
        })

    # ── Final decompression test ───────────────────────────────────────────
    print("\n══ Decompression test ══════════════════════════════════════════")
    all_sens_global = np.concatenate([
        np.array(all_sens_s[iz]) + iz * N_SLICE for iz in range(NZ)
        if len(all_sens_s[iz]) > 0
    ])
    print(f"  Total sensors: {len(all_sens_global):,}")

    # Final prediction: run one GP per slice using ALL accumulated sensors
    mu_decomp  = np.empty(N)
    for iz in range(NZ):
        sl = slice(iz * N_SLICE, (iz+1) * N_SLICE)
        if len(all_sens_s[iz]) == 0:
            mu_decomp[sl] = y_mean
            continue
        sens_flat = np.array(all_sens_s[iz])
        y_norm_slice = y_norm[sl]
        mu_norm, _ = gp_predict_2d(
            coords_2d[sens_flat], y_norm_slice[sens_flat], coords_2d)
        mu_decomp[sl] = mu_norm * y_std + y_mean

    err_decomp = y_full - mu_decomp
    frac_within = 100.0 * (np.abs(err_decomp) < ERROR_BOUND).mean()
    rmse_decomp = float(np.sqrt(np.mean(err_decomp**2)))
    data_range  = float(y_full.max() - y_full.min())
    psnr_decomp = 20.0 * np.log10(data_range / rmse_decomp) if rmse_decomp > 0 else float('inf')
    print(f"  Decompression RMSE       : {rmse_decomp:.4f} m/s")
    print(f"  Decompression PSNR       : {psnr_decomp:.2f} dB")
    print(f"  Fraction within ±EB      : {frac_within:.2f}%")
    print(f"  Shannon H (final)        : {entropy_bits(err_decomp):.4f} bits")

    return dict(
        rounds     = rounds,
        y_full     = y_full,
        all_sens   = all_sens_global,
        err_decomp = err_decomp,
        mu_decomp  = mu_decomp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(results: dict) -> None:
    BIN_W  = 2.0 * ERROR_BOUND
    half_r = min(40.0 * ERROR_BOUND,
                 max(abs(r['err_vals'][~np.isnan(r['err_vals'])]).max()
                     for r in results['rounds']))
    n_bins = max(int(np.ceil(2 * half_r / BIN_W)), 2)
    edges  = np.linspace(-half_r, half_r, n_bins + 1)

    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(results['rounds']) - 1, 1))
              for i in range(len(results['rounds']))]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, rd in enumerate(results['rounds']):
        vals   = rd['err_vals'][~np.isnan(rd['err_vals'])]
        counts, _ = np.histogram(vals, bins=edges)
        probs  = counts / counts.sum()
        H      = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        lbl    = (f"Round {rd['round']}  "
                  f"({rd['frac_comp']:.1f}% done  |  H={H:.2f} bits)")
        ax.stairs(probs, edges, color=colors[i], alpha=0.55,
                  linewidth=1.8, fill=True, edgecolor=colors[i], label=lbl)

    ax.axvspan(-ERROR_BOUND, ERROR_BOUND, color='gold', alpha=0.18, zorder=0)
    ax.axvline( ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.8)
    ax.axvline(-ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.8)

    ax.set_xlabel('Prediction error  (m/s)', fontsize=12)
    ax.set_ylabel('Fraction of points', fontsize=12)
    ax.set_title(
        f'2D Slice-by-Slice Progressive GP — Error Histograms\n'
        f'EB={ERROR_BOUND}  ACCEPT_BINS={ACCEPT_BINS}  '
        f'K/slice/round={K_PER_ROUND_2D}  DS={DOWNSAMPLE}  '
        f'Kernel={KERNEL_TYPE}',
        fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.2, ls='--')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_HISTS, dpi=150)
    print(f"Histogram saved: {OUT_HISTS}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print("Loading wind field …")
    vol       = load_wind()
    coords_2d = build_coords_2d()
    print(f"Loaded. vol shape: {vol.shape}  coords_2d: {coords_2d.shape}")

    results = run_progressive_2d(vol, coords_2d, rng)

    # Save checkpoint
    payload = {'results': results, 'vol': vol,
               'hyperparams': dict(ls_z=0.0, ls_xy=LS_XY, sig2=SIG2)}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(payload, f, protocol=4)
    size_mb = CHECKPOINT_FILE.stat().st_size / 1e6
    print(f"\nCheckpoint saved: {CHECKPOINT_FILE.name}  ({size_mb:.0f} MB)")

    plot_histograms(results)
    print("Done.")
