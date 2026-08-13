#!/usr/bin/env python3
"""
sst_progressive_gp.py
=====================
Progressive GP compression on NOAA OI SST V2 weekly mean data.

OVERVIEW
--------
Each weekly snapshot (180 lat × 360 lon) is treated as a 2-D spatial field.
Only ocean pixels (44,219 of 64,800) are compressed; land is stored separately.

The same progressive-sensing loop as hurricane_progressive_gp.py is applied
snapshot by snapshot:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Round r (for one snapshot):                                         │
  │  1. SENSE   — rpgks selects k sensor locations from ocean candidates │
  │  2. PREDICT — Fit 2-D GP at sensor values; predict all ocean pixels  │
  │  3. COMPRESS— Points where |pred − true| < ACCEPT_BOUND are done    │
  │  4. REPEAT  — Remaining high-error points feed into round r+1        │
  └──────────────────────────────────────────────────────────────────────┘

TEMPORAL DESIGN
---------------
Independent per snapshot (current):
  Same kernel hyperparameters for every snapshot (fitted once on a subsample
  drawn across multiple time steps).  Sensor positions are chosen adaptively
  per snapshot by rpgks.

Space-time Kronecker GP (future extension):
  The natural next step is a separable space-time kernel
      k((t,p), (t',p')) = σ² · k_t(t,t') · k_xy(p,p')
  giving K = K_t ⊗ K_xy.  Sensors would sit on a tensor-product grid
  k_t time steps × k_xy spatial positions, and prediction across the whole
  (n_T, n_ocean) cube collapses to two small matrix multiplications — exactly
  the same structure used for (z, xy) in the hurricane Kronecker GP.
  See USE_SPACETIME_KRONECKER placeholder below.
"""

from __future__ import annotations
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ARGONNE   = Path(__file__).resolve().parent
DATA_PATH = ARGONNE.parent / "gpoed-code-python" / "sst.wkmean.1990-present.nc"
MASK_PATH = DATA_PATH.parent / "lsmask.nc"

# ── Algorithm knobs ───────────────────────────────────────────────────────────
N_ROUNDS     = 5       # progressive sensing rounds per snapshot
K_PER_ROUND  = 100     # ocean sensors selected per round
ERROR_BOUND  = 0.5     # quantiser half-width (°C); SZ2 bin width = 2×EB
ACCEPT_BINS  = 1       # bins accepted as "compressed"  (ACCEPT_BOUND = ACCEPT_BINS×EB)
ACCUMULATE   = True    # True: accumulate sensors round-over-round
T_PROCESS    = 100     # number of snapshots to process (None = all 1727)

# ── Space-time Kronecker (placeholder for future extension) ───────────────────
USE_SPACETIME_KRONECKER = False   # not yet implemented; see module docstring

# ── Derived ───────────────────────────────────────────────────────────────────
ACCEPT_BOUND = ACCEPT_BINS * ERROR_BOUND

# ── SZ2/SZ3 quantiser limits ──────────────────────────────────────────────────
SZ2_QUANTIZER_BINS = 65536
SZ2_UNPRED_BOUND   = (SZ2_QUANTIZER_BINS // 2) * (2 * ERROR_BOUND)

# ── GP kernel parameters (fitted via MLE; updated in main) ───────────────────
LS_XY  = 0.10    # horizontal correlation length in [0,1]²
SIG2   = 1.0     # signal variance (data normalised to unit variance)
NOISE  = 1e-3    # GP nugget

# ── rpgks settings ────────────────────────────────────────────────────────────
CHOL_RANK_MUL = 2       # Cholesky rank = CHOL_RANK_MUL × K_PER_ROUND
BATCH_PRED    = 10_000  # batch size for GP prediction

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Checkpoint ────────────────────────────────────────────────────────────────
PLOTS_ONLY = False
CHECKPOINT_FILE = (ARGONNE /
    f"pgp_sst_checkpoint_R{N_ROUNDS}_k{K_PER_ROUND}"
    f"_eb{ERROR_BOUND}_ab{ACCEPT_BINS}_T{T_PROCESS}.pkl")
HP_CACHE = ARGONNE / "pgp_sst_hyperparams_matern52.pkl"

# ── Output files ──────────────────────────────────────────────────────────────
OUT_PANELS = ARGONNE / "sst_progressive_gp_panels.png"
OUT_HISTS  = ARGONNE / "sst_progressive_gp_histograms.png"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_sst():
    """
    Returns
    -------
    sst        : (n_T, ny, nx) float32
    lat        : (ny,) float64
    lon        : (nx,) float64
    ocean_mask : (ny, nx) bool — True = ocean
    """
    import netCDF4 as nc
    ds  = nc.Dataset(str(DATA_PATH))
    sst = ds.variables["sst"][:]
    lat = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds.variables["lon"][:], dtype=np.float64)
    ds.close()
    if isinstance(sst, np.ma.MaskedArray):
        sst = sst.filled(np.nan)

    ds2       = nc.Dataset(str(MASK_PATH))
    raw_mask  = np.array(ds2.variables["mask"][0], dtype=np.int16)
    ds2.close()
    ocean_mask = (raw_mask == 1)
    n_ocean = int(ocean_mask.sum())
    print(f"  Ocean pixels: {n_ocean} / {ocean_mask.size} "
          f"({100*n_ocean/ocean_mask.size:.1f}%)")
    return sst.astype(np.float32), lat, lon, ocean_mask


def build_coords(lat: np.ndarray, lon: np.ndarray,
                 ocean_mask: np.ndarray) -> np.ndarray:
    """
    Build (n_ocean, 2) coordinate array in [0,1]² for all ocean pixels.
    Row order matches ocean_mask.ravel() == True.
    Columns: [lat_norm, lon_norm].
    """
    ny, nx = ocean_mask.shape
    lat_g, lon_g = np.meshgrid(lat, lon, indexing='ij')  # (ny, nx) each
    lat_norm = (lat_g - lat.min()) / (lat.max() - lat.min())
    lon_norm = (lon_g - lon.min()) / (lon.max() - lon.min())
    coords_full = np.stack([lat_norm.ravel(), lon_norm.ravel()], axis=1)
    return coords_full[ocean_mask.ravel()]   # (n_ocean, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D MATÉRN-5/2 KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    2-D anisotropic Matérn-5/2 kernel.

        r  = sqrt( (Δlat/LS_XY)² + (Δlon/LS_XY)² )
        k  = σ² · (1 + √5·r + 5r²/3) · exp(−√5·r)

    A : (m, 2)  [lat_norm, lon_norm]
    B : (n, 2)
    Returns (m, n).
    """
    s5 = np.sqrt(5.0)
    dy = (A[:, 0:1] - B[:, 0]) / LS_XY
    dx = (A[:, 1:2] - B[:, 1]) / LS_XY
    r2 = dy**2 + dx**2
    r  = np.sqrt(np.maximum(r2, 0.0))
    return SIG2 * (1.0 + s5 * r + (5.0 / 3.0) * r2) * np.exp(-s5 * r)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER FITTING (MLE on subsample across multiple snapshots)
# ─────────────────────────────────────────────────────────────────────────────

def fit_hyperparams(coords: np.ndarray, sst_ocean: np.ndarray,
                    n_fit: int = 2000, n_restarts: int = 4,
                    rng=None) -> dict:
    """
    Fit LS_XY, SIG2, noise by maximising the GP log marginal likelihood.

    Parameters
    ----------
    coords     : (n_ocean, 2) normalised coordinates
    sst_ocean  : (n_T, n_ocean) ocean-only SST values (float32)
    n_fit      : total subsample size (points drawn across snapshots)
    n_restarts : L-BFGS-B restarts
    rng        : numpy Generator
    """
    from scipy.optimize import minimize
    global LS_XY, SIG2

    if rng is None:
        rng = np.random.default_rng(0)

    # Draw n_fit points from random (snapshot, ocean_pixel) pairs
    n_T, n_ocean = sst_ocean.shape
    t_idx  = rng.integers(0, n_T,     size=n_fit)
    p_idx  = rng.integers(0, n_ocean, size=n_fit)
    X      = coords[p_idx]             # (n_fit, 2)
    y_raw  = sst_ocean[t_idx, p_idx]  # (n_fit,)

    # Normalise y to z-score
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std())
    y = (y_raw - y_mean) / max(y_std, 1e-6)

    n = len(X)

    def neg_lml(log_params):
        ls_xy, sig2, noise = np.exp(log_params)
        s5 = np.sqrt(5.0)
        dy = (X[:, 0:1] - X[:, 0]) / ls_xy
        dx = (X[:, 1:2] - X[:, 1]) / ls_xy
        r2 = dy**2 + dx**2
        r  = np.sqrt(np.maximum(r2, 0.0))
        K  = sig2 * (1.0 + s5 * r + (5.0 / 3.0) * r2) * np.exp(-s5 * r)
        K += (noise + 1e-6) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml   = (-0.5 * y @ alpha
                 - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2 * np.pi))
        return -lml

    bounds = [(-4.6, 0.69),   # log(ls_xy): [0.01, 2]
              (-2.3, 2.3),    # log(sig2):  [0.1, 10]
              (-9.2, 0.0)]    # log(noise): [1e-4, 1]

    best_nll, best_params = np.inf, None
    x0_list = [np.log([LS_XY, SIG2, NOISE])]
    for _ in range(n_restarts - 1):
        x0_list.append(rng.uniform([b[0] for b in bounds],
                                   [b[1] for b in bounds]))

    for x0 in x0_list:
        res = minimize(neg_lml, x0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=300, ftol=1e-10))
        if res.fun < best_nll:
            best_nll, best_params = res.fun, res.x

    ls_xy, sig2, noise = np.exp(best_params)
    print(f"  Fitted: LS_XY={ls_xy:.4f}  SIG2={sig2:.4f}  "
          f"noise={noise:.2e}  log-LML={-best_nll:.2f}")

    LS_XY = ls_xy
    SIG2  = sig2
    return dict(ls_xy=ls_xy, sig2=sig2, noise=noise,
                log_likelihood=-best_nll)


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SELECTION (2-D rpgks)
# ─────────────────────────────────────────────────────────────────────────────

def select_sensors(coords_cand: np.ndarray, k: int, rng,
                   kern_fn=None) -> np.ndarray:
    """
    Randomly Pivoted Cholesky + GKS sensor selection.

    coords_cand : (n_cand, 2)
    k           : sensors to select
    kern_fn     : kernel(A, B) → (m, n); defaults to module `kernel()`
    Returns (k,) integer indices into coords_cand.
    """
    if kern_fn is None:
        kern_fn = kernel
    n_cand = len(coords_cand)
    rank   = min(CHOL_RANK_MUL * k, n_cand)

    diags = SIG2 * np.ones(n_cand)
    F     = np.zeros((n_cand, rank))

    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total < 1e-12:
            actual_rank = i
            break
        si = int(rng.choice(n_cand, p=diags / total))
        g  = kern_fn(coords_cand, coords_cand[[si]]).ravel()
        if i > 0:
            g = g - F[:, :i] @ F[si, :i]
        piv = g[si]
        if piv <= 0:
            actual_rank = i
            break
        F[:, i] = g / np.sqrt(piv)
        diags    = np.maximum(diags - F[:, i]**2, 0)

    F = F[:, :actual_rank]
    if actual_rank == 0:
        return rng.choice(n_cand, size=k, replace=False)

    U, _, _ = np.linalg.svd(F, full_matrices=False)
    u_k     = U[:, :min(k, actual_rank)]
    _, _, p = scipy_qr(u_k.T, pivoting=True)
    return p[:k].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# GP PREDICTION (batched)
# ─────────────────────────────────────────────────────────────────────────────

def gp_predict(coords_sens: np.ndarray, y_sens: np.ndarray,
               coords_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    GP posterior mean and variance at coords_pred given observations y_sens.

    Returns
    -------
    mu  : (n_pred,) posterior mean
    var : (n_pred,) posterior variance
    """
    n_s = len(coords_sens)
    K_ss = kernel(coords_sens, coords_sens) + NOISE * np.eye(n_s)
    L    = np.linalg.cholesky(K_ss)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sens))

    n_pred = len(coords_pred)
    mu  = np.empty(n_pred)
    var = np.empty(n_pred)

    for start in range(0, n_pred, BATCH_PRED):
        sl = slice(start, start + BATCH_PRED)
        Ks = kernel(coords_pred[sl], coords_sens)   # (batch, n_s)
        mu[sl] = Ks @ alpha
        v      = np.linalg.solve(L, Ks.T)           # (n_s, batch)
        var[sl] = SIG2 - np.sum(v**2, axis=0)

    var = np.maximum(var, 0.0)
    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESSIVE GP — ONE SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive_snapshot(y_ocean: np.ndarray, coords: np.ndarray,
                             rng, snapshot_idx: int) -> dict:
    """
    Run N_ROUNDS of progressive GP on a single SST snapshot.

    Parameters
    ----------
    y_ocean      : (n_ocean,) raw SST values at ocean pixels (°C)
    coords       : (n_ocean, 2) normalised (lat, lon) coordinates
    rng          : numpy Generator
    snapshot_idx : integer, used for print labels

    Returns
    -------
    dict with keys: rounds, y_ocean, y_mean, y_std, coords
    """
    n_ocean  = len(y_ocean)
    y_mean   = float(y_ocean.mean())
    y_std    = float(y_ocean.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_norm   = (y_ocean - y_mean) / y_std
    eb_norm  = ERROR_BOUND / y_std

    # Candidate pool: all ocean pixels
    cand_mask = np.ones(n_ocean, dtype=bool)
    all_sensor_idx: list[np.ndarray] = []
    rounds: list[dict] = []

    # Accumulated sensor set (grows if ACCUMULATE=True)
    acc_sensor_idx: list[int] = []

    mu_all  = np.zeros(n_ocean)
    err_all = y_ocean.copy()   # before any prediction

    for rd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        cand_idx = np.where(cand_mask)[0]
        n_cand   = len(cand_idx)

        if n_cand == 0:
            print(f"    Snapshot {snapshot_idx} Round {rd}: no candidates left.")
            break

        # ── Sensor selection ────────────────────────────────────────────────
        k = min(K_PER_ROUND, n_cand)
        local_idx  = select_sensors(coords[cand_idx], k, rng)
        global_idx = cand_idx[local_idx]
        all_sensor_idx.append(global_idx)

        if ACCUMULATE:
            acc_sensor_idx.extend(global_idx.tolist())
            sens_idx = np.array(acc_sensor_idx, dtype=int)
        else:
            sens_idx = global_idx

        # ── GP prediction ────────────────────────────────────────────────────
        mu_norm, var_norm = gp_predict(
            coords[sens_idx], y_norm[sens_idx], coords)

        mu_all  = mu_norm * y_std + y_mean
        err_all = y_ocean - mu_all

        # Override sensor positions: error = 0 there (known exactly)
        err_all[sens_idx] = 0.0
        mu_all[sens_idx]  = y_ocean[sens_idx]

        # ── Compress candidates within ACCEPT_BOUND ─────────────────────────
        err_cand       = err_all[cand_idx]
        compress_mask  = np.abs(err_cand) < ACCEPT_BOUND
        n_comp         = int(compress_mask.sum())

        # Mark compressed + sensors as done
        newly_done = np.where(compress_mask)[0]
        cand_mask[cand_idx[newly_done]] = False
        cand_mask[sens_idx]              = False

        dt = time.time() - t0
        n_sens_total = len(acc_sensor_idx) if ACCUMULATE else len(sens_idx)
        n_comp_total = int((~cand_mask).sum()) - n_sens_total
        print(f"    Snapshot {snapshot_idx:4d}  Round {rd}  "
              f"({dt:.1f}s)  sens={n_sens_total}  "
              f"comp={n_comp_total}  "
              f"remaining={cand_mask.sum()}")

        rounds.append(dict(
            round      = rd,
            sensor_idx = global_idx,
            mu_all     = mu_all.copy(),
            err_all    = err_all.copy(),
            var_all    = (var_norm * y_std**2).copy(),
            n_comp     = n_comp_total,
            n_sens     = n_sens_total,
            frac_comp  = 100.0 * n_comp_total / n_ocean,
        ))

    return dict(rounds=rounds, y_ocean=y_ocean, y_mean=y_mean,
                y_std=y_std, coords=coords, snapshot_idx=snapshot_idx)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESSIVE GP — ALL SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive_all(sst: np.ndarray, ocean_mask: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray, rng) -> dict:
    """
    Run progressive GP on T_PROCESS snapshots, return aggregated results.
    """
    n_T = sst.shape[0]
    T   = n_T if T_PROCESS is None else min(T_PROCESS, n_T)
    coords = build_coords(lat, lon, ocean_mask)

    print(f"\nRunning progressive GP on {T} snapshots …")
    print(f"  n_ocean={len(coords)}  N_ROUNDS={N_ROUNDS}  "
          f"K_PER_ROUND={K_PER_ROUND}  EB={ERROR_BOUND}°C\n")

    all_results = []
    # Evenly spaced snapshot indices
    t_indices = np.round(np.linspace(0, n_T - 1, T)).astype(int)

    for i, t in enumerate(t_indices):
        y_ocean = sst[t][ocean_mask]
        res = run_progressive_snapshot(y_ocean, coords, rng,
                                       snapshot_idx=int(t))
        res['t_global'] = int(t)
        all_results.append(res)

    return dict(all_results=all_results, coords=coords,
                ocean_mask=ocean_mask, lat=lat, lon=lon,
                sst=sst, t_indices=t_indices)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Checkpoint saved: {CHECKPOINT_FILE.name}")

def load_checkpoint() -> dict:
    with open(CHECKPOINT_FILE, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

CMAP_SST  = "RdYlBu_r"
CMAP_ERR  = "bwr"
BAR_COLOR = "#3a7dbf"


def _ocean_to_grid(values: np.ndarray, ocean_mask: np.ndarray,
                   fill: float = np.nan) -> np.ndarray:
    """Scatter ocean-pixel values back to the full (ny, nx) grid."""
    grid = np.full(ocean_mask.shape, fill, dtype=float)
    grid[ocean_mask] = values
    return grid


def plot_snapshot_panel(data: dict, snapshot_result: dict,
                        round_idx: int = -1) -> None:
    """
    Four-panel plot for one snapshot at the given round index:
      True SST | GP prediction | Error | Std dev
    """
    rd       = snapshot_result['rounds'][round_idx]
    t_global = snapshot_result['t_global']
    omask    = data['ocean_mask']
    lat      = data['lat']
    lon      = data['lon']

    y_true = snapshot_result['y_ocean']
    mu     = rd['mu_all']
    err    = rd['err_all']
    std    = np.sqrt(rd['var_all'])

    vmin_t, vmax_t = float(np.nanmin(y_true)), float(np.nanmax(y_true))
    dlim           = ERROR_BOUND * 5

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]

    for ax, vals, title, cmap, vmin, vmax in [
        (axes[0], y_true, f'True SST  (t={t_global})',
         CMAP_SST, vmin_t, vmax_t),
        (axes[1], mu,     f'GP prediction  (round {rd["round"]})',
         CMAP_SST, vmin_t, vmax_t),
        (axes[2], err,    'Error  (true − pred)',
         CMAP_ERR, -dlim, dlim),
        (axes[3], std,    'GP std dev',
         'viridis', 0, None),
    ]:
        grid = _ocean_to_grid(vals, omask)
        if vmax is None:
            vmax = float(np.nanmax(np.abs(grid[omask])))
        im = ax.imshow(grid, extent=ext, origin='upper',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

        if title.startswith('Error'):
            ax.contour(np.linspace(lon.min(), lon.max(), omask.shape[1]),
                       np.linspace(lat.min(), lat.max(), omask.shape[0]),
                       np.abs(_ocean_to_grid(np.abs(err), omask, 0)),
                       levels=[ERROR_BOUND], colors='k', linewidths=0.8)

    fig.suptitle(
        f'SST Progressive GP — Snapshot t={t_global}  |  '
        f'Round {rd["round"]}/{N_ROUNDS}  |  '
        f'{rd["frac_comp"]:.1f}% compressed  |  '
        f'EB={ERROR_BOUND}°C',
        fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_PANELS, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_PANELS.name}")


def _sz2_bin_edges(all_errs: np.ndarray) -> np.ndarray:
    """Bin edges aligned to SZ2 quantiser (width = 2×EB)."""
    BIN_W   = 2.0 * ERROR_BOUND
    lo_edge = np.floor(np.percentile(all_errs, 0.5)  / BIN_W) * BIN_W - ERROR_BOUND
    hi_edge = np.ceil (np.percentile(all_errs, 99.5) / BIN_W) * BIN_W + ERROR_BOUND
    return np.arange(lo_edge, hi_edge + BIN_W * 0.5, BIN_W)


def plot_histograms(data: dict) -> None:
    """
    Aggregate error histogram over all processed snapshots, all rounds overlaid.
    Each bar = one SZ2 quantiser bin (width 2×EB).
    """
    all_results = data['all_results']
    n_rounds    = max(len(r['rounds']) for r in all_results)

    # Gather errors per round across all snapshots
    errors_by_round: dict[int, list] = {rd: [] for rd in range(1, n_rounds + 1)}
    for snap in all_results:
        for rd_data in snap['rounds']:
            errors_by_round[rd_data['round']].append(rd_data['err_all'])

    all_errs = np.concatenate([
        e for errs in errors_by_round.values() for e in errs])
    edges = _sz2_bin_edges(all_errs)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.cm.plasma(np.linspace(0.1, 0.85, n_rounds))

    for i, rd in enumerate(range(1, n_rounds + 1)):
        vals = np.concatenate(errors_by_round[rd])
        counts, _ = np.histogram(vals, bins=edges)
        probs = counts / max(counts.sum(), 1)
        H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        ax.stairs(probs, edges, color=colors[i], alpha=0.7, linewidth=1.8,
                  fill=True, edgecolor=colors[i],
                  label=f'Round {rd}  H={H:.2f} bits')

    # Gold zone: compressed range
    ax.axvspan(-ERROR_BOUND, ERROR_BOUND, color='gold', alpha=0.18, zorder=0)
    gold_patch = mpatches.Patch(color='gold', alpha=0.5,
                                label=f'Compressed zone (±{ERROR_BOUND}°C)')

    # Red lines: centre-bin boundary (quantiser threshold)
    ax.axvline( ERROR_BOUND, color='crimson', lw=2.0, zorder=4)
    ax.axvline(-ERROR_BOUND, color='crimson', lw=2.0, zorder=4)
    if ACCEPT_BINS > 1:
        ax.axvline( ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.6)
        ax.axvline(-ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.6)
    accept_patch = mpatches.Patch(
        color='crimson', alpha=0.7,
        label=(f'Quantiser bound (±{ERROR_BOUND}°C)'
               + (f' | accept ±{ACCEPT_BOUND:.2f}°C ({ACCEPT_BINS}×EB)'
                  if ACCEPT_BINS > 1 else '')))

    # SZ2 unpredictable boundary (off-chart → arrow at axis edge)
    n_unpred   = int(np.sum(np.abs(all_errs) > SZ2_UNPRED_BOUND))
    pct_unpred = 100.0 * n_unpred / max(len(all_errs), 1)
    xlo, xhi   = edges[0], edges[-1]
    for sign, x_bound, x_edge in [
            (-1, -SZ2_UNPRED_BOUND, xlo),
            (+1,  SZ2_UNPRED_BOUND, xhi)]:
        if sign * x_bound <= sign * x_edge:
            ax.axvline(x_bound, color='purple', lw=1.6, ls='--', zorder=3)
        else:
            ax.annotate(
                '', xy=(x_edge, 0.92), xycoords=('data', 'axes fraction'),
                xytext=(x_edge - sign * (xhi - xlo) * 0.04, 0.92),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.6),
                annotation_clip=False)
            ax.axvline(x_edge, color='purple', lw=1.6, ls='--', zorder=3, alpha=0.7)

    ax.annotate(
        (f'SZ2 unpredictable (|err|>{SZ2_UNPRED_BOUND:.0f}°C): '
         f'{n_unpred:,} pts ({pct_unpred:.2f}%)'),
        xy=(0.99, 0.09), xycoords='axes fraction',
        fontsize=8, ha='right', va='bottom', color='purple',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.annotate(
        f'error range: [{all_errs.min():.3f}, {all_errs.max():.3f}]°C',
        xy=(0.99, 0.03), xycoords='axes fraction',
        fontsize=8.5, ha='right', va='bottom', color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.set_xlabel('Prediction error (°C)', fontsize=12)
    ax.set_ylabel('Fraction of points',    fontsize=12)
    ax.set_title(
        f'SST Progressive GP — Error Histograms  '
        f'({len(all_results)} snapshots, all rounds overlaid)\n'
        f'Each bar = one SZ2/SZ3 quantiser bin (width 2×EB={2*ERROR_BOUND}°C) | '
        f'Red = centre-bin boundary (±EB) | Purple = SZ2 unpredictable limit',
        fontsize=10, pad=8)
    ax.set_xlim(xlo, xhi)
    ax.grid(True, alpha=0.2, ls='--')
    ax.spines[['top', 'right']].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [gold_patch, accept_patch],
              labels  + [gold_patch.get_label(), accept_patch.get_label()],
              fontsize=8.5, loc='upper left', framealpha=0.85)

    plt.tight_layout()
    fig.savefig(OUT_HISTS, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_HISTS.name}")


def plot_compression_summary(data: dict) -> None:
    """
    Line plot: fraction of points compressed vs. round,
    averaged and ±1σ across all processed snapshots.
    """
    all_results = data['all_results']
    n_rounds    = max(len(r['rounds']) for r in all_results)

    frac_mat = np.full((len(all_results), n_rounds), np.nan)
    for i, snap in enumerate(all_results):
        for rd_data in snap['rounds']:
            rd = rd_data['round'] - 1
            frac_mat[i, rd] = rd_data['frac_comp']

    mu_frac  = np.nanmean(frac_mat, axis=0)
    std_frac = np.nanstd(frac_mat,  axis=0)
    rds      = np.arange(1, n_rounds + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rds, mu_frac,  'o-', color='steelblue', lw=2, label='Mean')
    ax.fill_between(rds, mu_frac - std_frac, mu_frac + std_frac,
                    alpha=0.25, color='steelblue', label='±1σ across snapshots')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('% ocean pixels compressed', fontsize=12)
    ax.set_title(
        f'SST Progressive GP — Compression Progress\n'
        f'({len(all_results)} snapshots, K={K_PER_ROUND}/round, EB={ERROR_BOUND}°C)',
        fontsize=11)
    ax.set_xticks(rds)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = ARGONNE / "sst_progressive_gp_summary.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global LS_XY, SIG2, NOISE

    if PLOTS_ONLY:
        print("PLOTS_ONLY mode — loading checkpoint …")
        data = load_checkpoint()
    else:
        rng = np.random.default_rng(SEED)

        print("Loading SST data …")
        sst, lat, lon, ocean_mask = load_sst()
        print(f"  SST shape: {sst.shape}  (n_T, ny, nx)")

        # ── Fit hyperparameters (once, cached) ────────────────────────────────
        if HP_CACHE.exists():
            hp = pickle.load(open(HP_CACHE, 'rb'))
            LS_XY = hp['ls_xy']
            SIG2  = hp['sig2']
            NOISE = hp.get('noise', NOISE)
            print(f"\nHyperparams loaded from cache: "
                  f"LS_XY={LS_XY:.4f}  SIG2={SIG2:.4f}  noise={NOISE:.2e}")
        else:
            sst_ocean = sst.reshape(sst.shape[0], -1)[:, ocean_mask.ravel()]
            coords    = build_coords(lat, lon, ocean_mask)
            print("\nFitting GP hyperparameters on subsample …")
            hp = fit_hyperparams(coords, sst_ocean,
                                 n_fit=2000, n_restarts=4, rng=rng)
            pickle.dump(hp, open(HP_CACHE, 'wb'), protocol=4)
            print(f"  Hyperparams cached: {HP_CACHE.name}")

        data = run_progressive_all(sst, ocean_mask, lat, lon, rng)
        save_checkpoint(data)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    # Panel plot: last round of the first processed snapshot
    plot_snapshot_panel(data, data['all_results'][0], round_idx=-1)
    plot_histograms(data)
    plot_compression_summary(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
