#!/usr/bin/env python3
"""
hurricane_progressive_gp.py
============================
Progressive GP compression on the ISABEL Hurricane wind speed (Uf48, U-component).

CONCEPTUAL OVERVIEW
-------------------
Standard compression treats every point identically. This algorithm exploits
spatial structure through a feedback loop between uncertainty and measurement:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Round r:                                                               │
  │  1. SENSE   — rpgks selects k sensor locations from remaining candidates│
  │               (those not yet compressed, not yet sensed).               │
  │               rpgks maximises mutual information I(sensors; field),     │
  │               equivalently minimising Shannon entropy H(field | sensors)│
  │  2. PREDICT — Fit a GP at the sensor values; predict everywhere.        │
  │  3. COMPRESS— Points where |pred − true| < ERROR_BOUND are "done":     │
  │               the GP nailed them within tolerance. Remove from pool.    │
  │  4. REPEAT  — Feed the remaining high-uncertainty points into round r+1 │
  └─────────────────────────────────────────────────────────────────────────┘

  DECOMPRESSION TEST:
    Collect ALL sensors from all rounds. Run one final GP over the full domain.
    This simulates what a decoder does: given N_ROUNDS × k stored sensor values,
    reconstruct the entire field.

WHY 5× DOWNSAMPLING  (AND WHY rpgks, NOT gks)
----------------------------------------------
gks requires the full n×n kernel matrix:
    n = 500×500×100 = 25M  →  K is 8 TB.  Impossible.

rpgks avoids this by building a low-rank Cholesky factor F of shape (n, rank):
    Step 1 (RPCholesky): O(n × rank) kernel evaluations — one column of K
                         per iteration. No n×n matrix ever formed.
    Step 2 (SVD+QR):     SVD of F gives U (n × rank); QR of U.T selects k pivots.

Memory budget at 5× downsampled (100×100×100 = 1M points):
    F:        1M × 200 × 8 bytes = 1.6 GB
    U (SVD):  1M × 200 × 8 bytes = 1.6 GB
    K_pred:   batched 50k × 200  = 80 MB at a time    ← key memory saving
    Peak:     ~3.5 GB — feasible on a 16 GB laptop.

At native resolution (25M points): F alone = 40 GB. Dead on arrival.
5× spatial downsampling (→ 1M) is the practical ceiling without GPU/cluster.

ACCUMULATED vs INDEPENDENT SENSORS (ACCUMULATE flag)
------------------------------------------------------
ACCUMULATE = False (default): each round's GP uses only that round's k sensors.
  → Shows how much coverage each independent batch provides.
ACCUMULATE = True:            each round adds to the sensor pool.
  → Round 3 uses 600 sensors. Typically better reconstruction but harder to
    attribute improvement to individual rounds.
"""

from __future__ import annotations
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ARGONNE   = Path(__file__).resolve().parent
DATA_DIR  = ARGONNE / "100x500x500"
DATA_FILE = DATA_DIR / "Uf48.bin.f32"

# ── Algorithm knobs ───────────────────────────────────────────────────────────
N_ROUNDS      = 20       # number of progressive sensing rounds
K_PER_ROUND   = 250     # sensors selected per round (total = k_z × k_xy in Kronecker mode)
ERROR_BOUND   = 0.01     # quantisation bin half-width (m/s); SZ2 bin width = 2×EB
ACCEPT_BINS   = 10       # number of central EB-bins that count as "compressed"
                        # ACCEPT_BOUND = ACCEPT_BINS × ERROR_BOUND
                        #   1 → only centre bin (|err| < EB)
                        #   3 → centre 3 bins    (|err| < 3×EB)
DOWNSAMPLE    = 3       # spatial downsampling factor (5 → 100×100 grid)
ACCUMULATE    = True    # True: accumulate sensors; False: fresh k each round

# ── Kronecker GP ──────────────────────────────────────────────────────────────
# USE_KRONECKER = True  switches from arbitrary-3D rpgks to a SEPARABLE GP where
# sensors sit on a tensor-product grid: K_Z_PER_ROUND z-levels × k_xy xy-positions.
#
# Why this is different from the current approach
# ───────────────────────────────────────────────
# Current: sensors at ARBITRARY 3D locations.  The k_s×k_s K_ss matrix has no
#   structure; prediction requires O(N × k_s) batch multiply (N up to 1M, k_s=200).
#   rpgks builds F of shape N × rank = 1M × 200 ≈ 1.6 GB.
#
# Kronecker: sensors on a k_z×k_xy GRID.  The separable kernel
#   k(a,b) = SIG2 × kz(az,bz) × kxy(axy,bxy) gives K_ss = SIG2 × K_z ⊗ K_xy.
#   Eigendecompose K_z (k_z×k_z) and K_xy (k_xy×k_xy) separately (tiny matrices),
#   then prediction reduces to:
#     μ_mat (NZ×N_xy) = SIG2 × K_z_pred @ A_mat @ K_xy_pred.T
#   which is O(NZ·k_z·k_xy + N_xy·k_z·k_xy) instead of O(N·k_s).
#   rpgks runs in 2D only: F shape = N_xy × rank (e.g., 10K × 200 ≈ 16 MB — 100× smaller).
#
# ACCUMULATE strategy with Kronecker: fix k_xy xy-positions in round 1;
# each subsequent round adds K_Z_PER_ROUND NEW z-levels (worst RMSE first).
# The union of all rounds forms a growing complete tensor-product grid, so
# the Kronecker structure is preserved across accumulated rounds.
USE_KRONECKER  = False  # True: Kronecker GP (fast, but sensors constrained to xy-plane grid)
                        # False: standard 3D rpgks — sensors placed freely in full 3D space
K_Z_PER_ROUND  = 4      # z-levels added per round (Kronecker only)
                         # k_xy = K_PER_ROUND // K_Z_PER_ROUND xy-positions (fixed after round 1)

# ── Sensor placement strategy ─────────────────────────────────────────────────
# 'rpgks'      — all rounds use rpgks (mutual-information maximisation)
# 'max_residual'— all rounds pick the k candidates with largest |error|
# 'hybrid'     — first N_RPGKS_ROUNDS rounds use rpgks for broad coverage,
#                then switch to max_residual to collapse the error tails.
#
# Why hybrid?
#   rpgks knows nothing about the field values — it selects sensors based on
#   kernel geometry alone, giving good global coverage in round 1.
#   max_residual uses the current GP error to find exactly where the model is
#   failing, directly targeting the high-|error| voxels that dominate Shannon
#   entropy.  Starting with rpgks then switching combines exploration (round 1)
#   with exploitation (rounds 2+).
SENSOR_STRATEGY  = 'hybrid'  # 'rpgks'          — all rounds: mutual-info rpgks
                              # 'max_residual'   — all rounds: top-k largest |error|
                              # 'rpgks_residual' — all rounds: rpgks on worst-RESID_PERCENTILE% subset
                              # 'hybrid'         — round 1: rpgks; rounds 2+: rpgks_residual
N_RPGKS_ROUNDS   = 1         # (hybrid only) rounds of rpgks before switching to rpgks_residual
RESID_PERCENTILE = 25        # (rpgks_residual / hybrid) run rpgks on the worst X% of candidates
                              # by |error|. Smaller = more aggressive error-targeting but less
                              # exploration. 25 means rpgks picks from the top-25%-error voxels,
                              # giving spatial diversity within the high-error region.

# ── Quick-test mode ───────────────────────────────────────────────────────────
# Set QUICK_TEST = True to use 10× downsampling (250k points instead of 1M).
# Memory drops from ~3.5 GB to ~0.9 GB; each round takes ~20-40s instead of
# 5-15 minutes. Use this to verify plots before running the full 1M experiment.
QUICK_TEST     = True
if QUICK_TEST:
    DOWNSAMPLE = 10

# Derived acceptance bound (do not edit directly — change ACCEPT_BINS above)
ACCEPT_BOUND = ACCEPT_BINS * ERROR_BOUND

# ── SZ2/SZ3 quantizer limits ──────────────────────────────────────────────────
# SZ2 and SZ3 use a linear quantizer with a fixed number of bins.
# The default radius in SZ3 is 32768, giving 65536 usable bins centered at 0.
# Points whose prediction error exceeds the outer boundary are marked
# "unpredictable" and stored losslessly (no compression benefit).
#
#   Boundary = (SZ2_QUANTIZER_BINS // 2) × (2 × ERROR_BOUND)
#            = 32768 × 1.0 = 32768 m/s  (for EB=0.5)
#
# This will almost always be off the histogram x-axis, so we annotate the
# unpredictable count as text rather than drawing lines outside the plot.
SZ2_QUANTIZER_BINS  = 65536                               # SZ2/SZ3 default (2 × radius)
SZ2_UNPRED_BOUND    = (SZ2_QUANTIZER_BINS // 2) * (2 * ERROR_BOUND)  # ±bound in m/s

# ── GP kernel parameters ──────────────────────────────────────────────────────
# Anisotropic kernel with separate length-scales for vertical z and horizontal xy.
# Values are in [0,1]-normalised coordinates.
LS_Z   = 0.25    # vertical correlation length
LS_XY  = 0.15    # horizontal correlation length
SIG2   = 1.0     # signal variance (data normalised to unit variance, so 1.0)
NOISE  = 1e-3    # GP observation noise (regularisation)

# ── Kernel type ───────────────────────────────────────────────────────────────
# 'matern52'   — twice differentiable; smoothest option (original)
# 'matern32'   — once differentiable; moderately rough
# 'matern12'   — not differentiable (Ornstein-Uhlenbeck); roughest standard kernel,
#                follows sharp gradients aggressively near sensors
# 'multiscale' — sum of two Matérn-5/2 kernels at different length scales:
#                a global component (LS_Z, LS_XY, SIG2) captures broad structure,
#                a local component (LS_Z_LOCAL, LS_XY_LOCAL, SIG2_LOCAL) captures
#                fine-scale detail.  Total prior variance = SIG2 + SIG2_LOCAL.
KERNEL_TYPE = 'matern12'

# ── Multi-scale kernel — local component ─────────────────────────────────────
# Only used when KERNEL_TYPE = 'multiscale'.
# Set LS_*_LOCAL much shorter than LS_Z / LS_XY so it resolves fine features.
# SIG2_LOCAL controls how much variance the fine-scale component explains.
# A good starting point: LS_LOCAL ≈ LS / 5,  SIG2_LOCAL ≈ SIG2 * 0.5
LS_Z_LOCAL   = 0.04   # short vertical correlation length for local component
LS_XY_LOCAL  = 0.03   # short horizontal correlation length for local component
SIG2_LOCAL   = 0.5    # variance attributed to fine-scale structure

# ── rpgks / Cholesky settings ─────────────────────────────────────────────────
# CHOL_RANK_OVERSAMPLE: additive oversampling for rpgks.
# Cholesky rank = K_PER_ROUND + CHOL_RANK_OVERSAMPLE.
# The extra columns give the SVD more room to find the best k pivot locations.
# The standard recommendation is 5–20 extra columns; beyond ~20 returns diminish.
#
#   oversample=0  →  rank = K_PER_ROUND      (minimum, noisiest selection)
#   oversample=10 →  rank = K_PER_ROUND + 10
#   oversample=20 →  rank = K_PER_ROUND + 20 (recommended)
#
# Memory for F at 250K candidates: (K_PER_ROUND + oversample) × 250K × 8 bytes
#   oversample=20, K=200 → 220 × 250K × 8 ≈ 440 MB
CHOL_RANK_OVERSAMPLE = 20   # additive oversampling (k + this many extra columns)
BATCH_PRED    = 50_000   # batch size for batched prediction (memory control)

# ── Local GP prediction ───────────────────────────────────────────────────────
# LOCAL_GP = True  switches from global GP (all sensors inform every prediction)
# to local GP (each voxel is predicted from its N_LOCAL nearest sensors only).
#
# Global GP:  smooth reconstruction; distant sensors pull predictions toward
#             the global mean, blurring fine-scale details.
# Local GP:   each voxel uses a tiny independent GP fit on its N_LOCAL nearest
#             sensors.  Predictions are sharper near sensors and better capture
#             local gradients (e.g. the eye wall).  Voxels far from any sensor
#             still fall back to the local mean of their nearest neighbours.
#
# N_LOCAL: number of nearest sensors per voxel.  Smaller = sharper/more local,
#          larger = smoother/more global.  Good starting range: 20–50.
LOCAL_GP = True
N_LOCAL  = 30

# ── Visualisation ─────────────────────────────────────────────────────────────
SLICE_IDX = 50   # which z-level (0–99) to show in the panel plot

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Checkpoint / plots-only ───────────────────────────────────────────────────
# After a full run, results are saved to CHECKPOINT_FILE (pickle).
# Set PLOTS_ONLY = True to skip all computation and re-draw from the saved file.
# The checkpoint name encodes the key parameters so changing N_ROUNDS,
# K_PER_ROUND, ERROR_BOUND, or DOWNSAMPLE automatically invalidates the old file.
PLOTS_ONLY        = False
REFIT_HYPERPARAMS = False   # True: delete cached hyperparams and refit via MLE on next run

# ── Resume from checkpoint ────────────────────────────────────────────────────
# Set RESUME = True to load an existing checkpoint and run N_ROUNDS *additional*
# rounds on top of however many are already stored.
# The prior checkpoint is found automatically (same params, any round count).
# A new checkpoint is saved with the full combined round list.
RESUME = False

# ── Z-layer exclusion (testing) ───────────────────────────────────────────────
# Z_SKIP_BOTTOM: number of bottom z-levels to exclude from the simulation.
# Set to 0 to use all 100 levels.  Set to 10 to skip z-indices 0–9
# (the "trouble layers" that dominate errors in early experiments).
Z_SKIP_BOTTOM = 15

def _checkpoint_path(total_rounds: int) -> Path:
    return (ARGONNE /
        f"pgp_checkpoint_R{total_rounds}_k{K_PER_ROUND}_eb{ERROR_BOUND}"
        f"_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}_zskip{Z_SKIP_BOTTOM}.pkl")

CHECKPOINT_FILE = _checkpoint_path(N_ROUNDS)   # default (non-resume) path

# ── Output files ─────────────────────────────────────────────────────────────
OUT_PANELS = ARGONNE / "progressive_gp_panels.png"
OUT_HISTS  = ARGONNE / "progressive_gp_histograms.png"

# ── Grid dimensions ───────────────────────────────────────────────────────────
NZ_ORIG, NY_ORIG, NX_ORIG = 100, 500, 500
NZ = NZ_ORIG - Z_SKIP_BOTTOM            # active vertical levels (Z_SKIP_BOTTOM set above)
NY = -(-NY_ORIG // DOWNSAMPLE)          # ceiling division — matches numpy ::DOWNSAMPLE slicing
NX = -(-NX_ORIG // DOWNSAMPLE)          # (floor division gives wrong count when DOWNSAMPLE∤500)
N  = NZ * NY * NX

print(f"Grid: {NZ}×{NY}×{NX} = {N:,} points  "
      f"({DOWNSAMPLE}× spatial downsampling)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_wind() -> np.ndarray:
    """
    Load Uf48.bin.f32  (U-component of wind, shape 100×500×500, float32).
    Skip the bottom Z_SKIP_BOTTOM z-levels, then apply DOWNSAMPLE× spatial
    downsampling in x and y.
    Returns float64 array of shape (NZ, NY, NX).
    """
    raw = np.fromfile(DATA_FILE, dtype=np.float32)
    vol = raw.reshape(NZ_ORIG, NY_ORIG, NX_ORIG)
    # Subsample every DOWNSAMPLE-th point in y and x; optionally skip bottom layers
    vol_ds = vol[Z_SKIP_BOTTOM:, ::DOWNSAMPLE, ::DOWNSAMPLE]   # (NZ, NY, NX)
    return vol_ds.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COORDINATE GRID
# ─────────────────────────────────────────────────────────────────────────────

def build_coords() -> np.ndarray:
    """
    Build (N, 3) coordinate array for the 3D grid, normalised to [0, 1]³.

    Layout: axis-0 = z (height level), axis-1 = y (latitude), axis-2 = x (longitude).
    Flattened in C order: z changes slowest, x fastest.
    Separate normalisation per axis so the anisotropic kernel makes sense even
    though z and xy have different physical spacings.
    """
    gz = np.linspace(0, 1, NZ)
    gy = np.linspace(0, 1, NY)
    gx = np.linspace(0, 1, NX)
    zz, yy, xx = np.meshgrid(gz, gy, gx, indexing='ij')  # each (NZ, NY, NX)
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)  # (N, 3)
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def _aniso_r(A: np.ndarray, B: np.ndarray,
             ls_z: float, ls_xy: float):
    """Anisotropic scaled distance components. Returns (r2, r)."""
    dz = (A[:, 0:1] - B[:, 0]) / ls_z
    dy = (A[:, 1:2] - B[:, 1]) / ls_xy
    dx = (A[:, 2:3] - B[:, 2]) / ls_xy
    r2 = dz**2 + dy**2 + dx**2
    return r2, np.sqrt(np.maximum(r2, 0.0))


def _matern52(r2, r, sig2):
    return sig2 * (1.0 + np.sqrt(5.0) * r + (5.0/3.0) * r2) * np.exp(-np.sqrt(5.0) * r)

def _matern32(r2, r, sig2):
    return sig2 * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)

def _matern12(r2, r, sig2):
    return sig2 * np.exp(-r)


def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Anisotropic kernel dispatched by KERNEL_TYPE.

    Kernel options
    ──────────────
    matern52   k = σ²(1+√5r+5r²/3)exp(−√5r)   twice differentiable (smooth)
    matern32   k = σ²(1+√3r)exp(−√3r)          once differentiable
    matern12   k = σ²exp(−r)                    not differentiable (roughest)
    multiscale k = matern52(LS_Z, LS_XY, SIG2)       global smooth component
               + matern52(LS_Z_LOCAL, LS_XY_LOCAL, SIG2_LOCAL)  fine-scale component

    A : (m, 3)  →  rows are points [z, y, x]
    B : (n, 3)
    Returns (m, n) covariance matrix.
    """
    if KERNEL_TYPE == 'matern52':
        r2, r = _aniso_r(A, B, LS_Z, LS_XY)
        return _matern52(r2, r, SIG2)

    elif KERNEL_TYPE == 'matern32':
        r2, r = _aniso_r(A, B, LS_Z, LS_XY)
        return _matern32(r2, r, SIG2)

    elif KERNEL_TYPE == 'matern12':
        r2, r = _aniso_r(A, B, LS_Z, LS_XY)
        return _matern12(r2, r, SIG2)

    elif KERNEL_TYPE == 'multiscale':
        # Global component: broad smooth structure
        r2_g, r_g = _aniso_r(A, B, LS_Z, LS_XY)
        K_global   = _matern52(r2_g, r_g, SIG2)
        # Local component: fine-scale detail
        r2_l, r_l = _aniso_r(A, B, LS_Z_LOCAL, LS_XY_LOCAL)
        K_local    = _matern52(r2_l, r_l, SIG2_LOCAL)
        return K_global + K_local

    else:
        raise ValueError(f"Unknown KERNEL_TYPE: {KERNEL_TYPE!r}. "
                         "Choose 'matern52', 'matern32', 'matern12', or 'multiscale'.")


# ── Unit-variance factor kernels for Kronecker GP ─────────────────────────────
# Matérn-5/2 is NOT algebraically separable in 3D the way RBF is
# (RBF: exp(-r²) = exp(-rz²)·exp(-rxy²), which factorises exactly).
# With Matérn, the r = sqrt(rz² + rxy²) term couples z and xy.
#
# For Kronecker GP we therefore use the PRODUCT of two Matérn-5/2 kernels:
#   k(a,b) = SIG2 · k_z_unit(az, bz) · k_xy_unit(axy, bxy)
# This is still a valid kernel (product of valid kernels), and still gives
# a full Kronecker structure.  It is a slightly different model from the true
# 3D Matérn-5/2 (used in the standard GP path), but in practice the difference
# in fit quality is small compared to the computational benefit.
#
# Concretely: k_z_unit uses |Δz|/LS_Z as its 1-D distance,
#             k_xy_unit uses sqrt((Δy/LS_XY)²+(Δx/LS_XY)²) as its 2-D distance.
# Both apply the same Matérn-5/2 polynomial × exponential form.

_SQRT5 = np.sqrt(5.0)

def _kz_unit(Az: np.ndarray, Bz: np.ndarray) -> np.ndarray:
    """Unit-variance 1-D Matérn-5/2 in z.  Az: (m,) scalar array, Bz: (n,) → (m, n)."""
    r = np.abs(Az[:, None] - Bz[None, :]) / LS_Z
    return (1.0 + _SQRT5 * r + (5.0 / 3.0) * r**2) * np.exp(-_SQRT5 * r)


def _kxy_unit(Axy: np.ndarray, Bxy: np.ndarray) -> np.ndarray:
    """Unit-variance 2-D Matérn-5/2 in (y, x).  Axy: (m, 2), Bxy: (n, 2) → (m, n)."""
    dy = (Axy[:, 0:1] - Bxy[:, 0]) / LS_XY
    dx = (Axy[:, 1:2] - Bxy[:, 1]) / LS_XY
    r  = np.sqrt(np.maximum(dy**2 + dx**2, 0.0))
    return (1.0 + _SQRT5 * r + (5.0 / 3.0) * r**2) * np.exp(-_SQRT5 * r)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HYPERPARAMETER FITTING (3-D anisotropic RBF)
# ─────────────────────────────────────────────────────────────────────────────

def fit_hyperparams_3d(coords: np.ndarray, y_norm: np.ndarray,
                       n_fit: int = 1500, n_restarts: int = 4,
                       rng=None) -> dict:
    """
    Fit LS_Z, LS_XY (and optionally SIG2, noise) by maximising the GP log
    marginal likelihood on a random subsample of the normalised field.

    Because the data is z-score normalised, SIG2 ≈ 1 is a reasonable prior
    and is held fixed.  The noise nugget is also fitted to absorb any
    sub-grid variability that the smooth kernel cannot capture.

    Parameters
    ----------
    coords  : (N, 3) in [0,1]³  [z, y, x]
    y_norm  : (N,)   z-score normalised field values
    n_fit   : number of randomly sampled points to fit on  (keep ≤ 2000
              to avoid O(n³) Cholesky bottleneck)
    n_restarts : number of random initialisations (avoids local optima)
    rng     : np.random.Generator

    Returns
    -------
    dict with keys: ls_z, ls_xy, sig2, noise, log_likelihood
    Updates the module-level globals LS_Z, LS_XY, SIG2 in place.
    """
    from scipy.optimize import minimize
    global LS_Z, LS_XY, SIG2

    if rng is None:
        rng = np.random.default_rng(0)

    # Sub-sample for fitting using Latin Hypercube Sampling (LHS).
    # Pure random sampling can cluster spatially, leaving whole regions of the
    # field unrepresented and biasing the MLE (especially sig2).  LHS stratifies
    # each spatial dimension into n_fit equal-probability bins and draws exactly
    # one sample per bin, guaranteeing coverage across z, y, and x simultaneously.
    from scipy.stats.qmc import LatinHypercube
    n_fit_actual = min(n_fit, len(coords))
    sampler = LatinHypercube(d=3, seed=int(rng.integers(2**31)))
    lhs = sampler.random(n=n_fit_actual)          # (n_fit, 3) in [0, 1)^3
    # Map each LHS point to the nearest voxel on the grid
    iz  = np.clip((lhs[:, 0] * NZ).astype(int), 0, NZ - 1)
    iy  = np.clip((lhs[:, 1] * NY).astype(int), 0, NY - 1)
    ix  = np.clip((lhs[:, 2] * NX).astype(int), 0, NX - 1)
    idx = iz * NY * NX + iy * NX + ix
    idx = np.unique(idx)                           # remove rare duplicates at boundaries
    X   = coords[idx]            # (n_fit, 3)
    y   = y_norm[idx]            # (n_fit,)
    n     = len(X)

    def neg_lml(log_params):
        """Negative log marginal likelihood for anisotropic Matérn-5/2 (minimise this)."""
        ls_z, ls_xy, sig2, noise = np.exp(log_params)
        dz = (X[:, 0:1] - X[:, 0]) / ls_z
        dy = (X[:, 1:2] - X[:, 1]) / ls_xy
        dx = (X[:, 2:3] - X[:, 2]) / ls_xy
        r2 = dz**2 + dy**2 + dx**2
        r  = np.sqrt(np.maximum(r2, 0.0))
        s5 = np.sqrt(5.0)
        K  = sig2 * (1.0 + s5 * r + (5.0 / 3.0) * r2) * np.exp(-s5 * r)
        K += (noise + 1e-6) * np.eye(n)
        try:
            L   = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml   = (-0.5 * y @ alpha
                 - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2 * np.pi))
        return -lml

    # Bounds in log-space: ls ∈ [0.01, 2], sig2 ∈ [0.1, 10], noise ∈ [1e-4, 1]
    bounds = [(-4.6, 0.69),   # log(ls_z)
              (-4.6, 0.69),   # log(ls_xy)
              (-2.3, 2.3),    # log(sig2)
              (-9.2, 0.0)]    # log(noise)

    best_nll  = np.inf
    best_params = None

    # First restart: initialise near current globals
    x0_list = [np.log([LS_Z, LS_XY, SIG2, 1e-3])]
    # Remaining restarts: random initialisations
    for _ in range(n_restarts - 1):
        x0_list.append(rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds]))

    for x0 in x0_list:
        res = minimize(neg_lml, x0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-9))
        if res.fun < best_nll:
            best_nll    = res.fun
            best_params = res.x

    ls_z, ls_xy, sig2, noise = np.exp(best_params)

    print(f"  Fitted hyperparams: LS_Z={ls_z:.4f}  LS_XY={ls_xy:.4f}  "
          f"SIG2={sig2:.4f}  noise={noise:.2e}  "
          f"log-LML={-best_nll:.2f}")

    # Update module globals so kernel() picks them up
    LS_Z  = ls_z
    LS_XY = ls_xy
    SIG2  = sig2

    return dict(ls_z=ls_z, ls_xy=ls_xy, sig2=sig2,
                noise=noise, log_likelihood=-best_nll)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SENSOR SELECTION (rpgks)
# ─────────────────────────────────────────────────────────────────────────────

def select_sensors(coords_cand: np.ndarray, k: int, rng,
                   kern_fn=None) -> np.ndarray:
    """
    Select k sensor locations from candidate points using Randomly Pivoted
    Cholesky + GKS  (rpgks / rp_cssp).

    This greedily maximises the mutual information between the sensor set and
    the Gaussian field — equivalently, it selects the k candidate locations
    whose kernel columns best span the dominant subspace of K, i.e., the
    locations where uncertainty is concentrated.

    Algorithm
    ---------
    Step 1 — Randomly Pivoted Cholesky:
        Build low-rank factor F (n_cand × rank) such that K ≈ F @ F.T.
        Each of the 'rank' iterations picks a random pivot (weighted by
        residual variance) and computes one kernel column: O(n_cand) work.
        Total: O(n_cand × rank) — never forms the full n_cand × n_cand matrix.

    Step 2 — RPGKS (SVD + QR):
        Economy SVD of F → left singular vectors U (n_cand × rank).
        Column-pivoted QR of U.T: the first k pivot columns of U are the
        k points that best represent the GP's dominant uncertainty directions.

    Parameters
    ----------
    coords_cand : (n_cand, d)  candidate point coordinates (d=3 for 3D, d=2 for xy-only)
    k           : int          number of sensors to select
    rng         : numpy RNG    for reproducibility
    kern_fn     : callable or None
                  kernel(A, B) → (m, n) covariance matrix.
                  Defaults to the global 3-D `kernel()`.
                  Pass a 2-D kernel (e.g. _kxy_unit) when coords_cand is (n, 2).

    Returns
    -------
    local_idx : (k,) integer indices into coords_cand
    """
    if kern_fn is None:
        kern_fn = kernel
    n_cand = len(coords_cand)
    rank   = min(k + CHOL_RANK_OVERSAMPLE, n_cand)

    # ── Step 1: Randomly Pivoted Cholesky ────────────────────────────────────
    # diags[i] = K[i,i] − (variance already explained by columns so far)
    # This is the residual self-variance: high = point still uncertain.
    diags = SIG2 * np.ones(n_cand)    # all equal to SIG2 at start (RBF, r=0)
    F     = np.zeros((n_cand, rank))

    # Memory estimate: F alone is n_cand × rank × 8 bytes
    f_gb = n_cand * rank * 8 / 1e9
    print(f"    rpgks: n_cand={n_cand:,}  rank={rank}  F≈{f_gb:.2f} GB")

    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total < 1e-12:
            # Kernel is essentially rank-i; stop early
            actual_rank = i
            break

        # Sample pivot proportional to residual variance (high variance → likely pivot)
        weights = diags / total
        si      = int(rng.choice(n_cand, p=weights))

        # Compute one column of K at the pivot point, then subtract
        # the part already explained by the i columns computed so far.
        g = kern_fn(coords_cand, coords_cand[[si]]).ravel()   # (n_cand,)
        if i > 0:
            g = g - F[:, :i] @ F[si, :i]                    # rank-i update

        piv = g[si]
        if piv <= 0:
            actual_rank = i
            break

        F[:, i] = g / np.sqrt(piv)                          # normalise column
        diags   = np.maximum(diags - F[:, i]**2, 0)         # update residual diag

    F = F[:, :actual_rank]   # trim unused columns

    if actual_rank == 0:
        # Degenerate case: just pick k random candidates
        return rng.choice(n_cand, size=k, replace=False)

    # ── Step 2: RPGKS — SVD of F, then QR pivoting ──────────────────────────
    # U[:,j] is the j-th direction of dominant variation across candidate points.
    # QR pivoting on U.T finds the k candidate points that best "represent" the
    # space spanned by the top-k singular vectors — these are the most
    # informative sensor locations.
    U, _, _ = np.linalg.svd(F, full_matrices=False)    # U: (n_cand, actual_rank)
    u_k     = U[:, :min(k, actual_rank)]                # (n_cand, k)
    _, _, p = scipy_qr(u_k.T, pivoting=True)
    local_idx = p[:k]

    return local_idx.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GP PREDICTION (BATCHED)
# ─────────────────────────────────────────────────────────────────────────────

def gp_predict(
    coords_s: np.ndarray,    # (k_s, 3) sensor locations
    y_s     : np.ndarray,    # (k_s,)   observed values (normalised)
    coords_p: np.ndarray,    # (n, 3)   prediction locations
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard GP posterior mean and variance.

    GP prior: zero-mean,  covariance = kernel(·, ·).
    Likelihood: y = f(x) + ε,   ε ~ N(0, NOISE).

    Posterior (at prediction points X*):
        μ(X*) = K(X*, X_s) · [K(X_s, X_s) + NOISE·I]⁻¹ · y_s
        σ²(X*) = k(X*,X*) − K(X*, X_s) · [K(X_s,X_s) + NOISE·I]⁻¹ · K(X_s, X*)

    Computed in batches of BATCH_PRED rows to keep memory usage under control
    even at n = 1M prediction points.

    Returns
    -------
    mu  : (n,) posterior mean
    var : (n,) posterior variance (clipped to ≥ 0)
    """
    k_s = len(coords_s)

    # Solve once for α = K_ss⁻¹ y_s using Cholesky
    K_ss  = kernel(coords_s, coords_s) + NOISE * np.eye(k_s)  # (k_s, k_s)
    L     = np.linalg.cholesky(K_ss)                           # lower triangular
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_s))     # (k_s,)
    L_inv = np.linalg.solve(L, np.eye(k_s))                   # (k_s, k_s)

    n  = len(coords_p)
    mu  = np.empty(n, dtype=np.float64)
    var = np.empty(n, dtype=np.float64)

    # Batch over prediction points to avoid a single 1M × k_s matrix in memory
    for start in range(0, n, BATCH_PRED):
        end  = min(start + BATCH_PRED, n)
        K_bs = kernel(coords_p[start:end], coords_s)      # (batch, k_s)

        mu[start:end]  = K_bs @ alpha

        # σ²(x) = σ² − ||L⁻¹ K(X_s, x)||²  (one column per prediction point)
        v = L_inv @ K_bs.T                                 # (k_s, batch)
        var[start:end] = SIG2 - np.sum(v**2, axis=0)
        var[start:end] = np.maximum(var[start:end], 0.0)

    return mu, var


def gp_predict_local(
    coords_s: np.ndarray,    # (k_s, 3) sensor locations
    y_s     : np.ndarray,    # (k_s,)   observed values (normalised)
    coords_p: np.ndarray,    # (n, 3)   prediction locations
    n_local : int = 30,      # nearest sensors per voxel
    batch   : int = 2_000,   # voxels processed per batch (memory control)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local GP prediction: each target voxel is predicted from its n_local
    nearest sensors only, rather than all sensors globally.

    Why this is sharper than the global GP
    ───────────────────────────────────────
    In the global GP, predictions are a weighted average of ALL sensor values.
    Distant sensors have small kernel weight individually, but many of them
    together pull every prediction toward the global mean → smooth output.

    Here, each voxel sees only its n_local closest sensors.  Nearby sensors
    dominate completely, so rapid local changes in the field (e.g. the eye
    wall gradient) are captured by the sensors closest to that region without
    being diluted by sensors far away.

    Implementation
    ──────────────
    A scipy KDTree finds the n_local nearest sensors for every target voxel
    in O(N log k_s) time.  Voxels are then processed in batches of size
    `batch`; for each batch we build the (B, n_local, n_local) local kernel
    matrices and solve all B systems simultaneously using numpy's batched
    linear algebra — no Python loop over individual voxels.

    Memory per batch  (B=2000, n_local=30):
        K_ss  : 2000 × 30 × 30 × 8 bytes ≈  14 MB
        K_ps  : 2000 × 30      × 8 bytes ≈   0.5 MB
        diff  : 2000 × 30 × 30 × 3 × 8 bytes ≈ 43 MB   (transient)
    Total: comfortably under 1 GB even for large grids.
    """
    from scipy.spatial import KDTree

    n_s  = len(coords_s)
    n_p  = len(coords_p)
    nl   = min(n_local, n_s)          # can't ask for more neighbours than sensors

    mu  = np.empty(n_p, dtype=np.float64)
    var = np.empty(n_p, dtype=np.float64)

    # ── Build KDTree and query ALL prediction points at once ──────────────
    tree = KDTree(coords_s)
    _, nn_idx = tree.query(coords_p, k=nl, workers=-1)   # (n_p, nl)
    # workers=-1 uses all CPU cores for the query

    eye_nl = np.eye(nl)   # reused every batch

    for start in range(0, n_p, batch):
        end  = min(start + batch, n_p)
        B    = end - start

        local_idx = nn_idx[start:end]          # (B, nl)

        # Local sensor coords and values
        X_s  = coords_s[local_idx]             # (B, nl, 3)
        y_sl = y_s[local_idx]                  # (B, nl)
        X_p  = coords_p[start:end]             # (B, 3)

        # ── K_ss  (B, nl, nl) — respects KERNEL_TYPE ──────────────────────
        diff   = X_s[:, :, np.newaxis, :] - X_s[:, np.newaxis, :, :]  # (B,nl,nl,3)
        dz_ss  = diff[:, :, :, 0] / LS_Z
        dy_ss  = diff[:, :, :, 1] / LS_XY
        dx_ss  = diff[:, :, :, 2] / LS_XY
        r2_ss  = dz_ss**2 + dy_ss**2 + dx_ss**2
        r_ss   = np.sqrt(np.maximum(r2_ss, 0.0))
        if KERNEL_TYPE == 'matern52':
            K_ss = _matern52(r2_ss, r_ss, SIG2)
        elif KERNEL_TYPE == 'matern32':
            K_ss = _matern32(r2_ss, r_ss, SIG2)
        elif KERNEL_TYPE == 'matern12':
            K_ss = _matern12(r2_ss, r_ss, SIG2)
        elif KERNEL_TYPE == 'multiscale':
            dz_sl  = diff[:, :, :, 0] / LS_Z_LOCAL
            dy_sl2 = diff[:, :, :, 1] / LS_XY_LOCAL
            dx_sl2 = diff[:, :, :, 2] / LS_XY_LOCAL
            r2_sl  = dz_sl**2 + dy_sl2**2 + dx_sl2**2
            r_sl   = np.sqrt(np.maximum(r2_sl, 0.0))
            K_ss   = _matern52(r2_ss, r_ss, SIG2) + _matern52(r2_sl, r_sl, SIG2_LOCAL)
        K_ss += (NOISE + 1e-6) * eye_nl        # regularise; broadcasts over batch

        # ── K_ps  (B, nl) — respects KERNEL_TYPE ──────────────────────────
        diff_p = X_p[:, np.newaxis, :] - X_s               # (B, nl, 3)
        dz_p   = diff_p[:, :, 0] / LS_Z
        dy_p   = diff_p[:, :, 1] / LS_XY
        dx_p   = diff_p[:, :, 2] / LS_XY
        r2_p   = dz_p**2 + dy_p**2 + dx_p**2
        r_p    = np.sqrt(np.maximum(r2_p, 0.0))
        if KERNEL_TYPE == 'matern52':
            K_ps = _matern52(r2_p, r_p, SIG2)
        elif KERNEL_TYPE == 'matern32':
            K_ps = _matern32(r2_p, r_p, SIG2)
        elif KERNEL_TYPE == 'matern12':
            K_ps = _matern12(r2_p, r_p, SIG2)
        elif KERNEL_TYPE == 'multiscale':
            dz_pl  = diff_p[:, :, 0] / LS_Z_LOCAL
            dy_pl  = diff_p[:, :, 1] / LS_XY_LOCAL
            dx_pl  = diff_p[:, :, 2] / LS_XY_LOCAL
            r2_pl  = dz_pl**2 + dy_pl**2 + dx_pl**2
            r_pl   = np.sqrt(np.maximum(r2_pl, 0.0))
            K_ps   = _matern52(r2_p, r_p, SIG2) + _matern52(r2_pl, r_pl, SIG2_LOCAL)
        # K_ps shape: (B, nl)

        # ── Batched solve ─────────────────────────────────────────────────
        # np.linalg.solve gufunc signature is (m,m),(m,n)->(m,n).
        # We need an explicit trailing dim so numpy sees (B,nl,1) not (B,nl)
        # — otherwise it mistakes B for the system size m.
        # alpha = K_ss^{-1} y_sl,  shape (B, nl)
        alpha = np.linalg.solve(
            K_ss, y_sl[:, :, np.newaxis])[:, :, 0]  # (B, nl, 1) → (B, nl)
        mu[start:end] = np.einsum('bi,bi->b', K_ps, alpha)

        # var = SIG2 - K_ps @ K_ss^{-1} @ K_ps^T
        # v = K_ss^{-1} K_ps^T,  shape (B, nl, 1)
        v = np.linalg.solve(K_ss, K_ps[:, :, np.newaxis])  # (B, nl, 1)
        var[start:end] = np.maximum(
            SIG2 - np.einsum('bi,bi->b', K_ps, v[:, :, 0]), 0.0)

    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5b — KRONECKER GP PREDICTION  (separable kernel, grid sensors)
# ─────────────────────────────────────────────────────────────────────────────

def gp_predict_kronecker(
    z_sens:  np.ndarray,   # (k_z,)    z-coords of selected levels  [0,1]
    xy_sens: np.ndarray,   # (k_xy, 2) (y,x) coords of sensor positions  [0,1]²
    Y_grid:  np.ndarray,   # (k_z, k_xy) normalised observed values
    z_all:   np.ndarray,   # (NZ,)     z-coords for every prediction level
    xy_all:  np.ndarray,   # (N_xy, 2) (y,x) coords for every prediction point
) -> np.ndarray:            # (NZ * N_xy,) mean prediction in z-first (C) order
    """
    Kronecker GP mean prediction.

    The separable kernel k(a,b) = SIG2 × kz_unit(az,bz) × kxy_unit(axy,bxy)
    gives K_ss = SIG2 × K_z_ss ⊗ K_xy_ss.  Eigendecomposing each factor:

        K_z_ss  = V_z  Λ_z  V_z.T          K_xy_ss = V_xy Λ_xy V_xy.T
        K_ss + NOISE·I  ↔  eigenvalues  SIG2·λ_z,i·λ_xy,j + NOISE

    The alpha vector reshaped as A_mat (k_z × k_xy) is solved in the
    decoupled eigenbasis:

        Y_tilde = V_z.T @ Y_grid @ V_xy
        A_tilde = Y_tilde / (SIG2 · Λ_z ⊗ Λ_xy + NOISE)
        A_mat   = V_z @ A_tilde @ V_xy.T

    Prediction mean at all (NZ × N_xy) grid points then reduces to:

        μ_mat (NZ × N_xy) = SIG2 × K_z_pred @ A_mat @ K_xy_pred.T

    This is O(NZ·k_z·k_xy + N_xy·k_z·k_xy) vs the standard O(N·k_s).
    For NZ=N_xy=100 (quick test), k_z=4, k_xy=50: ~50k vs ~5M ops — 100× faster.
    """
    k_z  = len(z_sens)
    k_xy = len(xy_sens)

    # ── Factor kernel matrices (unit variance) ────────────────────────────────
    K_z_ss  = _kz_unit(z_sens,   z_sens)    # (k_z, k_z)
    K_xy_ss = _kxy_unit(xy_sens, xy_sens)   # (k_xy, k_xy)

    # ── Eigendecompose (symmetric positive-semidefinite, eigh is numerically stable)
    λ_z,  V_z  = np.linalg.eigh(K_z_ss)    # ascending eigenvalues + eigenvectors
    λ_xy, V_xy = np.linalg.eigh(K_xy_ss)
    λ_z  = np.maximum(λ_z,  0.0)           # clip tiny numerical negatives
    λ_xy = np.maximum(λ_xy, 0.0)

    # ── Kronecker eigenvalues and the diagonal of (K_ss + NOISE·I)⁻¹ ─────────
    # λ_kron[i,j] = SIG2 × λ_z[i] × λ_xy[j]  →  +NOISE for the full matrix
    λ_kron = SIG2 * np.outer(λ_z, λ_xy)    # (k_z, k_xy)
    D_inv  = 1.0 / (λ_kron + NOISE)         # (k_z, k_xy) element-wise

    # ── Solve for A_mat = reshape(K_ss_inv @ y_s, k_z, k_xy) ─────────────────
    Y_tilde = V_z.T @ Y_grid @ V_xy         # rotate observations to eigenbasis
    A_mat   = V_z @ (Y_tilde * D_inv) @ V_xy.T   # (k_z, k_xy)

    # ── Predict mean at all NZ × N_xy points ──────────────────────────────────
    K_z_pred  = _kz_unit(z_all,   z_sens)   # (NZ, k_z)
    K_xy_pred = _kxy_unit(xy_all, xy_sens)  # (N_xy, k_xy)

    # μ_mat[iz, ixy] = SIG2 · k_z_pred[iz,:] @ A_mat @ k_xy_pred[ixy,:]
    # Equivalent to the z-first Kronecker-vec product:
    #   (SIG2 · K_z_pred ⊗ K_xy_pred) @ vec(A_mat) = vec(K_xy_pred @ A_mat.T @ K_z_pred.T)
    # but transposed to our row-major (z-slow) convention:
    mu_mat = SIG2 * (K_z_pred @ A_mat @ K_xy_pred.T)   # (NZ, N_xy)

    return mu_mat.ravel()   # (N,) z-first (C order) matches global flat index


def select_z_levels(k_z: int, rmse_z: np.ndarray | None,
                    used_z: set) -> np.ndarray:
    """
    Choose k_z z-levels for this round's Kronecker sensor grid.

    Round 1 (rmse_z=None or used_z empty): evenly spaced across NZ.
    Later rounds: pick the k_z available z-levels with the HIGHEST per-z RMSE
    from the previous round's full-domain prediction (worst-first strategy).
    Already-used z-levels are excluded so we never repeat measurements.

    Returns integer array of shape (k_z,).
    """
    available = np.array([i for i in range(NZ) if i not in used_z])
    if len(available) == 0:
        return np.array([], dtype=int)
    k = min(k_z, len(available))
    if rmse_z is None or len(used_z) == 0:
        # First round: spread evenly
        idx = np.round(np.linspace(0, len(available) - 1, k)).astype(int)
        return available[idx]
    # Subsequent rounds: worst RMSE first
    rmse_avail = rmse_z[available]
    order = np.argsort(rmse_avail)[::-1]
    return available[order[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PROGRESSIVE COMPRESSION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_progressive(vol: np.ndarray, coords: np.ndarray, rng,
                    resume_from: dict | None = None) -> dict:
    """
    Execute N_ROUNDS rounds of progressive GP sensing + compression.

    State between rounds
    --------------------
    mask_avail[i] = True   →  point i is still a candidate:
                               • not yet compressed (error was too large), AND
                               • not yet chosen as a sensor in a prior round.
    When a point is compressed OR sensed, mask_avail is set to False.
    The covariance matrix for rpgks is therefore smaller each round:
    the "hard" unresolved region shrinks as we focus attention on it.

    Returns
    -------
    dict with:
      y_full, y_mean, y_std  —  raw field and normalisation constants
      rounds                 —  list of per-round result dicts
      err_decomp             —  final decompression error (full domain)
      mu_decomp              —  final reconstructed field
    """
    # Flatten volume to 1D (C-order: z slowest, x fastest)
    y_full = vol.ravel()                       # (N,)  true wind speed values

    # Normalise field to zero-mean unit-variance (required for zero-mean GP prior)
    y_mean = y_full.mean()
    y_std  = y_full.std()
    y_norm = (y_full - y_mean) / y_std

    # Error bound converted to normalised units
    eb_norm = ERROR_BOUND / y_std

    print(f"\nField stats:  mean={y_mean:.3f}  std={y_std:.3f}  "
          f"range=[{y_full.min():.2f}, {y_full.max():.2f}] m/s")
    print(f"Error bound: {ERROR_BOUND} m/s  =  {eb_norm:.4f} σ\n")

    # Kronecker coordinate arrays (reused every round regardless of mode)
    z_all  = np.linspace(0, 1, NZ)
    xy_all = coords.reshape(NZ, NY * NX, 3)[0, :, 1:]

    if resume_from is None:
        # ── Fresh start ───────────────────────────────────────────────────
        mask_avail          = np.ones(N, dtype=bool)
        all_sensor_idx_list = []
        settled_err         = np.full(N, np.nan)
        rounds              = []
        xy_sens_coords_fixed: np.ndarray | None = None
        used_z_set:  set    = set()
        rmse_z_prev: np.ndarray | None = None
        err_all = np.zeros(N)
        mu_all  = y_full.copy()
        var_all = np.full(N, SIG2)
        r_start = 0
    else:
        # ── Reconstruct state from a prior checkpoint ─────────────────────
        # We replay every past round to rebuild mask_avail, settled_err, and
        # the accumulated sensor list — then restore prediction arrays from
        # the final round's stored volumes.
        print(f"\nResuming from checkpoint: "
              f"{len(resume_from['rounds'])} rounds already complete.")

        prior_rounds        = resume_from['rounds']
        rounds              = list(prior_rounds)          # will be extended
        r_start             = len(prior_rounds)

        mask_avail          = np.ones(N, dtype=bool)
        all_sensor_idx_list = []
        settled_err         = np.full(N, np.nan)

        for rd in prior_rounds:
            s_idx = rd['sensor_idx']
            c_idx = rd['comp_idx']
            mask_avail[s_idx] = False
            mask_avail[c_idx] = False
            all_sensor_idx_list.append(s_idx)
            settled_err[s_idx] = 0.0                      # sensors stored exactly
            # comp_idx errors come from that round's err_vals
            ev = rd['err_vals']
            settled_err[c_idx] = ev[c_idx]

        # Restore prediction state from the last completed round
        last = prior_rounds[-1]
        mu_all  = last['pred_vol'].copy()                 # original-unit predictions
        err_all = last['err_vol'].copy()                  # signed errors
        var_all = last['var_vol'].copy()

        # Kronecker state (Kronecker disabled, but keep variables valid)
        xy_sens_coords_fixed = None
        used_z_set           = set()
        rmse_z_prev          = _per_z_rmse(err_all)

        remaining = int(mask_avail.sum())
        n_done    = N - remaining
        print(f"  Voxels done (sensed+compressed): {n_done:,} / {N:,} "
              f"({100*n_done/N:.1f}%)")
        print(f"  Candidates remaining           : {remaining:,}")
        print(f"  Running {N_ROUNDS} additional rounds (rounds "
              f"{r_start+1}–{r_start+N_ROUNDS}).\n")

    for r in range(r_start, r_start + N_ROUNDS):
        t_round = time.perf_counter()
        print(f"══ Round {r+1}/{r_start + N_ROUNDS} ══════════════════════════════════════")

        # ── 6a: Define candidate pool ─────────────────────────────────────────
        cand_idx    = np.where(mask_avail)[0]          # (n_cand,) global indices
        coords_cand = coords[cand_idx]                 # (n_cand, 3)
        print(f"  Candidates: {len(cand_idx):,}")

        if USE_KRONECKER:
            # ── 6b (Kronecker): grid sensor selection ─────────────────────────
            # Sensors live on a k_z × k_xy tensor-product grid so that
            # K_ss = SIG2 × K_z_ss ⊗ K_xy_ss and the Kronecker prediction formula
            # applies.  The grid grows each round: SAME k_xy xy-positions (fixed
            # after round 1) + K_Z_PER_ROUND NEW z-levels (worst RMSE first).
            # This ensures the union of all rounds is always a COMPLETE tensor
            # grid — a requirement for the Kronecker eigendecomposition trick.
            k_xy = K_PER_ROUND // K_Z_PER_ROUND    # horizontal positions per grid

            t0 = time.perf_counter()

            # ── z-level selection ─────────────────────────────────────────────
            z_levels = select_z_levels(K_Z_PER_ROUND, rmse_z_prev, used_z_set)
            if len(z_levels) == 0:
                print("  All z-levels exhausted — stopping early.")
                break
            used_z_set.update(z_levels.tolist())

            # ── horizontal (xy) selection via 2D rpgks ────────────────────────
            # Run 2D rpgks on the unique (y,x) positions among current candidates.
            # The candidate pool shrinks each round, so rpgks adapts to where
            # uncertainty remains — but operates in 2D (N_xy << N), making
            # the low-rank F matrix 100× smaller than the 3D version.
            if xy_sens_coords_fixed is None:
                # Round 1: select k_xy xy-positions from candidates' xy projection
                cand_xy_flat   = np.unique(cand_idx % (NY * NX))      # unique xy positions
                cand_xy_coords = np.stack(
                    [cand_xy_flat // NX / max(NY - 1, 1),
                     cand_xy_flat %  NX / max(NX - 1, 1)], axis=1)    # (n_unique, 2) in [0,1]
                local_xy = select_sensors(cand_xy_coords, k_xy, rng,
                                          kern_fn=_kxy_unit)   # 2D rpgks (xy-only kernel)
                xy_sens_coords_fixed = cand_xy_coords[local_xy]        # (k_xy, 2) — fixed forever
                print(f"  Sensor selection (2D rpgks, k_xy={k_xy}): {time.perf_counter()-t0:.1f}s")
            else:
                print(f"  Using fixed {k_xy} xy-positions from round 1; "
                      f"new z-levels: {z_levels.tolist()}")

            sel_xy_flat = np.round(
                xy_sens_coords_fixed[:, 0] * (NY - 1)
            ).astype(int) * NX + np.round(
                xy_sens_coords_fixed[:, 1] * (NX - 1)
            ).astype(int)   # (k_xy,) flat xy-indices

            # ── Build global sensor indices for this round ────────────────────
            global_s_idx = np.array([
                int(iz) * NY * NX + int(ixy)
                for iz in z_levels
                for ixy in sel_xy_flat
            ], dtype=int)   # (K_Z_PER_ROUND × k_xy,)

            all_sensor_idx_list.append(global_s_idx)
            print(f"  Sensor selection total: {time.perf_counter()-t0:.1f}s  "
                  f"({len(z_levels)} z-levels × {k_xy} xy = {len(global_s_idx)} sensors)")

            # ── 6c (Kronecker): accumulated training grid ─────────────────────
            # All accumulated z-levels × same k_xy xy-positions → complete grid.
            # Y_grid rows = z-levels (ascending), cols = xy-positions.
            if ACCUMULATE:
                all_z_acc = np.array(sorted(used_z_set), dtype=int)   # (r×K_Z,)
            else:
                all_z_acc = z_levels
            z_sens_coords = z_all[all_z_acc]                           # (n_z_acc,)
            Y_grid = y_norm[
                np.array([iz * NY * NX + ixy
                          for iz in all_z_acc
                          for ixy in sel_xy_flat])
            ].reshape(len(all_z_acc), k_xy)                            # (n_z_acc, k_xy)
            print(f"  Training grid: {len(all_z_acc)} z-levels × {k_xy} xy "
                  f"= {Y_grid.size} sensors total")

            # ── 6d (Kronecker): fast prediction at ALL N points ───────────────
            t0 = time.perf_counter()
            mu_norm_flat = gp_predict_kronecker(
                z_sens_coords, xy_sens_coords_fixed, Y_grid, z_all, xy_all)
            print(f"  GP prediction (Kronecker): {time.perf_counter()-t0:.1f}s")

            mu_all  = mu_norm_flat * y_std + y_mean   # (N,)
            err_all = y_full - mu_all
            var_all = np.zeros(N)   # variance skipped in Kronecker mode (not needed for compression)

        else:
            # ── 6b (Standard): sensor selection ──────────────────────────────
            t0 = time.perf_counter()
            err_cand_abs = np.abs(err_all[cand_idx])   # (n_cand,) — needed by residual strategies

            use_rpgks_full = (
                SENSOR_STRATEGY == 'rpgks'
                or (SENSOR_STRATEGY == 'hybrid' and r < N_RPGKS_ROUNDS)
            )
            use_max_resid = (SENSOR_STRATEGY == 'max_residual')
            use_rpgks_resid = (
                SENSOR_STRATEGY == 'rpgks_residual'
                or (SENSOR_STRATEGY == 'hybrid' and r >= N_RPGKS_ROUNDS)
            )

            if use_rpgks_full:
                # ── Full rpgks on all candidates ──────────────────────────────
                # Maximises mutual information over the full candidate pool.
                # Good global coverage; ignores current error distribution.
                local_s_idx = select_sensors(coords_cand, K_PER_ROUND, rng)
                print(f"  Sensor selection (rpgks full, round {r+1}): "
                      f"{time.perf_counter()-t0:.1f}s")

            elif use_max_resid:
                # ── Top-k by |error| ──────────────────────────────────────────
                # Greedy: directly picks the K_PER_ROUND worst-error candidates.
                # Risk: all sensors may cluster in one spatial "glob".
                local_s_idx = np.argsort(err_cand_abs)[::-1][:K_PER_ROUND]
                print(f"  Sensor selection (max_residual, round {r+1}): "
                      f"{time.perf_counter()-t0:.4f}s  "
                      f"|err| selected: [{err_cand_abs[local_s_idx[-1]]:.3f}, "
                      f"{err_cand_abs[local_s_idx[0]]:.3f}] m/s")

            elif use_rpgks_resid:
                # ── rpgks on worst-RESID_PERCENTILE% error subset ─────────────
                # Step 1: keep only the top RESID_PERCENTILE% candidates by |error|.
                #         This focuses the budget on high-error regions.
                # Step 2: run rpgks on that subset so sensors are spread across
                #         ALL the high-error globs, not just the single worst one.
                #
                # Compared to pure max_residual: adds spatial diversity within
                # the high-error stratum — a cluster of bad voxels won't consume
                # all K_PER_ROUND sensors; rpgks will spread them.
                # Compared to pure rpgks: ignores well-predicted voxels entirely,
                # so the budget goes where the GP is actually failing.
                threshold   = np.percentile(err_cand_abs,
                                            100.0 - RESID_PERCENTILE)
                hi_err_local = np.where(err_cand_abs >= threshold)[0]   # indices into cand_idx

                # Safety: if subset is smaller than K_PER_ROUND, use all of it
                k_sel = min(K_PER_ROUND, len(hi_err_local))
                if len(hi_err_local) >= K_PER_ROUND:
                    sub_idx  = select_sensors(coords_cand[hi_err_local],
                                              k_sel, rng)
                    local_s_idx = hi_err_local[sub_idx]
                else:
                    local_s_idx = hi_err_local

                err_sel = err_cand_abs[local_s_idx]
                print(f"  Sensor selection (rpgks_residual {RESID_PERCENTILE}%, "
                      f"round {r+1}): {time.perf_counter()-t0:.1f}s  "
                      f"subset={len(hi_err_local):,}  "
                      f"|err| selected: [{err_sel.min():.3f}, {err_sel.max():.3f}] m/s")

            global_s_idx = cand_idx[local_s_idx]
            all_sensor_idx_list.append(global_s_idx)

            # ── 6c (Standard): accumulated training data ──────────────────────
            if ACCUMULATE:
                sens_idx = np.concatenate(all_sensor_idx_list)
            else:
                sens_idx = global_s_idx
            coords_s = coords[sens_idx]
            y_s      = y_norm[sens_idx]
            print(f"  Training sensors: {len(sens_idx):,}")

            # ── 6d (Standard): full-domain batched prediction ─────────────────
            t0 = time.perf_counter()
            if LOCAL_GP:
                mu_norm_all, var_all = gp_predict_local(
                    coords_s, y_s, coords, n_local=N_LOCAL)
                print(f"  GP prediction (local, N_LOCAL={N_LOCAL}): "
                      f"{time.perf_counter()-t0:.1f}s")
            else:
                mu_norm_all, var_all = gp_predict(coords_s, y_s, coords)
                print(f"  GP prediction (global): {time.perf_counter()-t0:.1f}s")
            mu_all  = mu_norm_all * y_std + y_mean
            err_all = y_full - mu_all

        # ── Shared: extract candidate errors ─────────────────────────────────
        err_cand = err_all[cand_idx]
        var_cand = var_all[cand_idx]
        print(f"  Error range (cand): [{err_cand.min():.4f}, {err_cand.max():.4f}] m/s")

        # Update per-z RMSE for next round's z-level selection (Kronecker)
        rmse_z_prev = _per_z_rmse(err_all)

        # ── 6e: Build full-domain histogram error array ───────────────────────
        # For this round's histogram we want an error value for EVERY point:
        #   • Points settled in prior rounds → carry their locked-in error
        #     (0 for prior sensors, ≤ ERROR_BOUND for prior compressed)
        #   • Current candidates (all of cand_idx) → use current GP error
        #   • New sensors within cand_idx → override to 0 (stored exactly)
        # This means the histogram reflects the cumulative picture after r rounds.
        hist_err = settled_err.copy()          # NaN only for current candidates
        hist_err[cand_idx]      = err_cand     # fill current candidates
        hist_err[global_s_idx]  = 0.0          # sensors stored exactly → error = 0

        # ── 6f: Compression — identify points within acceptance band ─────────
        # ACCEPT_BOUND = ACCEPT_BINS × ERROR_BOUND.
        # Points with |err| < ACCEPT_BOUND are "compressed": the GP prediction
        # is good enough that only the quantised bin index needs to be stored.
        # ACCEPT_BINS=1 → centre bin only (classical SZ2-equivalent).
        # ACCEPT_BINS=3 → centre 3 bins, allowing up to ±3×EB error.
        compressed_local  = np.abs(err_cand) < ACCEPT_BOUND  # (n_cand,) bool
        compressed_global = cand_idx[compressed_local]
        n_comp_this_round = int(compressed_local.sum())

        # Cumulative counts after this round
        n_sens_total = int((~mask_avail).sum()) + n_comp_this_round  # sensors so far + new compressed
        # Actually compute from mask after update (done below), so track cumulatively:
        n_comp_cumul  = int(N - mask_avail.sum()) - len(global_s_idx) + n_comp_this_round
        # Simpler: fraction of ALL N voxels that are done (compressed OR sensed) after this round
        n_done_after  = int((~mask_avail).sum()) + n_comp_this_round  # will be done after update
        frac_cumul    = 100.0 * n_done_after / N   # cumulative % of full volume

        # Per-round rate: new compressions out of current pool (diagnostic only)
        frac_this     = 100.0 * n_comp_this_round / max(len(cand_idx), 1)
        print(f"  Compressed this round: {n_comp_this_round:,} / {len(cand_idx):,}"
              f"  ({frac_this:.1f}% of pool)"
              f"  [bound=±{ACCEPT_BOUND:.3f} m/s]")
        print(f"  Cumulative done:       {n_done_after:,} / {N:,}"
              f"  ({frac_cumul:.1f}% of full volume)")

        # ── 6g: Update availability mask and settled errors ───────────────────
        mask_avail[global_s_idx]      = False
        mask_avail[compressed_global] = False
        frac_cumul_actual = 100.0 * (N - mask_avail.sum()) / N   # true cumulative after update
        print(f"  Remaining pool:    {mask_avail.sum():,}  ({100*mask_avail.sum()/N:.1f}% of volume)")

        # Lock in final errors for points leaving the pool this round.
        settled_err[global_s_idx]      = 0.0
        settled_err[compressed_global] = err_cand[compressed_local]

        # ── 6h: Build full-volume arrays for plotting ─────────────────────────
        # Full field — no NaN gaps.  Sensors override to exact truth (error=0).
        pred_vol = mu_all.copy()          # (N,) full reconstruction
        err_vol  = err_all.copy()         # (N,) full signed error
        var_vol  = var_all.copy()         # (N,) posterior variance
        # All sensors placed so far are stored losslessly → error = 0
        sens_so_far = np.concatenate(all_sensor_idx_list)
        err_vol[sens_so_far] = 0.0
        pred_vol[sens_so_far] = y_full[sens_so_far]   # exact at sensors

        rounds.append({
            'round'         : r + 1,
            'sensor_idx'    : global_s_idx,      # this round's sensors only
            'cand_idx'      : cand_idx,           # all candidates this round
            'comp_idx'      : compressed_global,  # compressed this round
            'err_vals'      : hist_err,           # full-domain error (all N points)
            'pred_vol'      : pred_vol,
            'err_vol'       : err_vol,
            'var_vol'       : var_vol,
            'n_cand'        : len(cand_idx),
            'n_comp'        : n_comp_this_round,
            'frac_comp'     : frac_cumul_actual,   # cumulative % of ALL N voxels done
            'frac_this'     : frac_this,           # % of this round's pool compressed
        })
        print(f"  Round total:       {time.perf_counter()-t_round:.1f}s")

    # ── DECOMPRESSION TEST ────────────────────────────────────────────────────
    # Collect ALL sensors from all rounds and regress over the FULL domain.
    # This simulates a decoder: given the stored sensor values (N_ROUNDS × k),
    # reconstruct every point.
    print(f"\n══ Decompression test ══════════════════════════════════════════")
    all_sens = np.concatenate(all_sensor_idx_list)
    print(f"  Total sensors used: {len(all_sens)}")
    t0 = time.perf_counter()
    if LOCAL_GP:
        mu_d_norm, _ = gp_predict_local(
            coords[all_sens], y_norm[all_sens], coords, n_local=N_LOCAL)
        print(f"  GP prediction (local, N_LOCAL={N_LOCAL}): "
              f"{time.perf_counter()-t0:.1f}s")
    else:
        mu_d_norm, _ = gp_predict(coords[all_sens], y_norm[all_sens], coords)
        print(f"  GP prediction (global): {time.perf_counter()-t0:.1f}s")

    mu_decomp  = mu_d_norm * y_std + y_mean
    err_decomp = y_full - mu_decomp      # signed error over entire domain

    frac_within = 100 * (np.abs(err_decomp) < ERROR_BOUND).mean()
    print(f"  Within ±{ERROR_BOUND} m/s:   {frac_within:.1f}% of full domain")

    return {
        'y_full'     : y_full,
        'y_mean'     : y_mean,
        'y_std'      : y_std,
        'rounds'     : rounds,
        'mu_decomp'  : mu_decomp,
        'err_decomp' : err_decomp,
        'all_sens'   : all_sens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — HELPER: EXTRACT SLICE
# ─────────────────────────────────────────────────────────────────────────────

def to_slice(flat: np.ndarray) -> np.ndarray:
    """Extract horizontal slice at SLICE_IDX from flat (N,) array. Returns (NY, NX)."""
    return flat.reshape(NZ, NY, NX)[SLICE_IDX]


def to_slice_z(flat: np.ndarray, z_idx: int) -> np.ndarray:
    """Extract horizontal slice at z_idx from flat (N,) array. Returns (NY, NX)."""
    return flat.reshape(NZ, NY, NX)[z_idx]


def _per_z_rmse(err_vol: np.ndarray) -> np.ndarray:
    """
    Compute RMSE per z-level (height slice) over the full domain.
    Returns (NZ,) array; the argmax is the 'worst' slice to display.
    """
    err_3d = err_vol.reshape(NZ, NY, NX)
    return np.sqrt(np.nanmean(err_3d ** 2, axis=(1, 2)))


def sensor_xy(global_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert global flat indices to (x_pixel, y_pixel) on the horizontal slice.
    ALL sensors are projected onto the slice plane regardless of their z-level,
    so the spatial coverage of sensor placement is visible across the 2-D image.
    """
    # Global index → (z, y, x) in the downsampled grid
    rem    = global_idx % (NY * NX)
    y_pix  = rem // NX
    x_pix  = rem %  NX
    return x_pix, y_pix


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PANEL PLOTS  (static grid  ≤ 3 rounds;  video  > 3 rounds)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_one_round(axes_row, rd: dict, vol_true: np.ndarray,
                    vmin_t: float, vmax_t: float, dlim: float) -> None:
    """
    Fill one row of 5 axes with the panels for a single round.

    axes_row : (ax_true, ax_pred, ax_err, ax_var, ax_rmse)
    rd       : one entry from results['rounds']

    The displayed z-slice is chosen dynamically as the slice with the highest
    per-round RMSE, so the user always sees where the approximation is hardest.
    A 5th panel shows the per-z RMSE profile for the full hurricane volume.
    """
    ax_true, ax_pred, ax_err, ax_var, ax_rmse = axes_row

    # ── Find worst z-slice for this round ────────────────────────────────────
    rmse_z  = _per_z_rmse(rd['err_vol'])
    worst_z = int(np.argmax(rmse_z))

    true_s = to_slice_z(vol_true.ravel(), worst_z)
    pred_s = to_slice_z(rd['pred_vol'],   worst_z)
    err_s  = to_slice_z(rd['err_vol'],    worst_z)
    var_s  = to_slice_z(rd['var_vol'],    worst_z)

    z_label = f'z = {worst_z}  (worst RMSE = {rmse_z[worst_z]:.3f} m/s)'

    # ── Col 0: True field at worst-error slice ────────────────────────────────
    im0 = ax_true.imshow(true_s, cmap='RdYlBu_r',
                          origin='lower', aspect='auto',
                          vmin=vmin_t, vmax=vmax_t)
    ax_true.set_title(f'True wind speed\n{z_label}',
                      fontsize=9, fontweight='bold')
    plt.colorbar(im0, ax=ax_true, fraction=0.046, label='m/s')

    # ── Col 1: GP reconstruction + sensors ───────────────────────────────────
    im1 = ax_pred.imshow(pred_s, cmap='RdYlBu_r',
                          origin='lower', aspect='auto',
                          vmin=vmin_t, vmax=vmax_t)
    ax_pred.set_title(
        f'Round {rd["round"]} reconstruction  [{z_label}]\n'
        f'k={K_PER_ROUND} sensors  |  '
        f'{"accum." if ACCUMULATE else "independent"}',
        fontsize=9, fontweight='bold')
    plt.colorbar(im1, ax=ax_pred, fraction=0.046, label='m/s')

    # Sensor markers — project all this-round sensors onto the chosen slice plane
    sx, sy = sensor_xy(rd['sensor_idx'])
    ax_pred.scatter(sx, sy, c='black', s=25, marker='x',
                    linewidths=1.1, zorder=5, alpha=0.85,
                    label=f'k={K_PER_ROUND} sensors')
    ax_pred.legend(fontsize=7, loc='upper right',
                   framealpha=0.7, markerscale=0.8)

    # ── Col 2: Signed error at worst-error slice ──────────────────────────────
    norm_err = TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)
    im2 = ax_err.imshow(err_s, cmap='RdBu_r',
                         origin='lower', aspect='auto', norm=norm_err)
    ax_err.set_title(
        f'Signed error  (true − pred)  [{z_label}]\n'
        f'{rd["n_comp"]:,}/{rd["n_cand"]:,} compressed  ({rd["frac_comp"]:.1f}%)',
        fontsize=9, fontweight='bold')
    plt.colorbar(im2, ax=ax_err, fraction=0.046, label='m/s')
    ax_err.contour(np.abs(err_s), levels=[ERROR_BOUND],
                   colors='gold', linewidths=1.0, linestyles='--')

    # ── Col 3: GP posterior variance at worst-error slice ─────────────────────
    im3 = ax_var.imshow(var_s, cmap='viridis',
                         origin='lower', aspect='auto', vmin=0)
    ax_var.set_title(
        f'Posterior variance  [{z_label}]\n(next sensors → high-var regions)',
        fontsize=9, fontweight='bold')
    plt.colorbar(im3, ax=ax_var, fraction=0.046, label='σ² (norm.)')

    # ── Col 4: Per-z RMSE profile  (whole-hurricane view) ────────────────────
    z_indices = np.arange(NZ)
    ax_rmse.barh(z_indices, rmse_z, color='steelblue', alpha=0.7, height=0.85)
    ax_rmse.barh(worst_z, rmse_z[worst_z], color='tomato', alpha=0.95,
                 height=0.85, label=f'z={worst_z} (shown)')
    ax_rmse.axvline(ERROR_BOUND, color='gold', lw=1.4, ls='--',
                    label=f'±EB = {ERROR_BOUND} m/s')
    ax_rmse.set_xlabel('RMSE  (m/s)', fontsize=8)
    ax_rmse.set_ylabel('z level', fontsize=8)
    ax_rmse.set_title('Per-z RMSE\n(full hurricane)', fontsize=9, fontweight='bold')
    ax_rmse.set_ylim(-0.5, NZ - 0.5)
    ax_rmse.legend(fontsize=7, loc='lower right')
    ax_rmse.tick_params(labelsize=7)
    ax_rmse.spines[['top', 'right']].set_visible(False)


def _panel_suptitle(fig, extra: str = '') -> None:
    fig.suptitle(
        f'Progressive GP Compression  ─  ISABEL Hurricane  Uf48\n'
        f'{DOWNSAMPLE}× downsampled  |  Error bound ±{ERROR_BOUND} m/s  |  '
        f'{"Accumulated" if ACCUMULATE else "Independent"} sensors  |  '
        f'Worst-RMSE z-slice shown'
        + (f'  |  {extra}' if extra else ''),
        fontsize=11, y=1.01)


def plot_panels(results: dict, vol: np.ndarray) -> None:
    """
    Static N_ROUNDS × 5 grid.  Used when N_ROUNDS ≤ 3.
    Each row: true | reconstruction | error | variance | per-z RMSE profile.
    The displayed z-slice is the worst-RMSE slice for that round.
    """
    rounds = results['rounds']
    vmin_t = float(vol.min())
    vmax_t = float(vol.max())
    dlim   = ERROR_BOUND * 6

    fig, axes = plt.subplots(N_ROUNDS, 5,
                             figsize=(26, 4.8 * N_ROUNDS),
                             constrained_layout=True)
    if N_ROUNDS == 1:
        axes = axes[np.newaxis, :]

    for r_idx, rd in enumerate(rounds):
        _draw_one_round(axes[r_idx], rd, vol, vmin_t, vmax_t, dlim)
        axes[r_idx, 0].set_ylabel(f'Round {rd["round"]}',
                                  fontsize=11, fontweight='bold', labelpad=8)

    _panel_suptitle(fig)
    fig.savefig(OUT_PANELS, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved panel plot: {OUT_PANELS}")


def plot_panels_video(results: dict, vol: np.ndarray) -> None:
    """
    Save one PNG frame per round, then stitch into an MP4 (or GIF fallback).
    Each frame is a 1×5 figure.  The worst-RMSE z-slice is shown every frame.
    Used when N_ROUNDS > 3 (the static grid would be too tall).
    """
    import tempfile, shutil, os
    rounds = results['rounds']
    vmin_t = float(vol.min())
    vmax_t = float(vol.max())
    dlim   = ERROR_BOUND * 6

    frame_dir  = ARGONNE / "frames_panels"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []

    for rd in rounds:
        fig, axes = plt.subplots(1, 5, figsize=(26, 4.8), constrained_layout=True)
        _draw_one_round(axes, rd, vol, vmin_t, vmax_t, dlim)
        _panel_suptitle(fig, extra=f'Round {rd["round"]}/{N_ROUNDS}')

        fpath = frame_dir / f'frame_{rd["round"]:03d}.png'
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(fpath)
        print(f"  Saved frame {rd['round']}: {fpath.name}")

    _frames_to_video(frame_paths, ARGONNE / "progressive_gp_panels.mp4",
                     ARGONNE / "progressive_gp_panels.gif", fps=1)


def _frames_to_video(frame_paths, mp4_path, gif_path, fps=1) -> None:
    """
    Stitch PNG frames into MP4 via system ffmpeg (subprocess), or GIF fallback.

    ffmpeg is called with a concat demuxer so the frames don't have to be
    sequentially numbered — any list of PNG paths works.
    """
    import subprocess, tempfile, os

    # ── MP4 via system ffmpeg (subprocess) ───────────────────────────────────
    try:
        # Write a concat manifest: "file '/abs/path/frame.png'\nduration N\n"
        duration = 1.0 / fps
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            manifest = f.name
            for fp in frame_paths:
                f.write(f"file '{fp}'\nduration {duration}\n")
            # ffmpeg needs the last entry repeated to set its duration
            f.write(f"file '{frame_paths[-1]}'\n")

        # Try codecs in order of preference; stop at first success
        codec_attempts = [
            ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'],
            ['-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-crf', '28'],
            ['-c:v', 'mpeg4',   '-pix_fmt', 'yuv420p', '-q:v', '5'],
            ['-c:v', 'mjpeg',   '-pix_fmt', 'yuvj420p', '-q:v', '3'],
        ]
        succeeded = False
        for codec_args in codec_attempts:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', manifest,
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            ] + codec_args + [str(mp4_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                succeeded = True
                break
            print(f"  codec {codec_args[1]} failed: {result.stderr.splitlines()[-1] if result.stderr else ''}")

        os.unlink(manifest)

        if succeeded:
            print(f"Saved MP4: {mp4_path}")
            return
        else:
            print(f"All ffmpeg codec attempts failed; falling back to GIF …")
    except FileNotFoundError:
        print("ffmpeg not found on PATH; falling back to GIF …")
    except Exception as e:
        print(f"MP4 step failed ({e}); falling back to GIF …")

    # ── GIF via PIL ───────────────────────────────────────────────────────────
    try:
        from PIL import Image
        imgs = [Image.open(str(fp)) for fp in frame_paths]
        imgs[0].save(str(gif_path), save_all=True,
                     append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        print(f"Saved GIF: {gif_path}")
        return
    except Exception as e:
        print(f"PIL GIF also failed ({e})")

    print(f"Individual frames saved in: {frame_paths[0].parent}")
    print("Manual MP4:  "
          f"ffmpeg -framerate {fps} -pattern_type glob -i '*.png' "
          f"-c:v libx264 -pix_fmt yuv420p out.mp4")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PLOT 2: OVERLAPPING ERROR HISTOGRAMS  +  PROPORTION TABLES
# ─────────────────────────────────────────────────────────────────────────────

BAR_COLOR = '#4878cf'   # single fixed colour used in the video frames

def _compute_edges(results: dict) -> np.ndarray:
    """Bin edges shared across all histogram figures.

    Bin width = 2×ERROR_BOUND, matching SZ2/SZ3's linear quantizer bin width
    exactly — each bar in the histogram corresponds to one quantizer bin.
    Range covers 0.5th–99.5th percentile of all error values, snapped to the
    nearest 2×EB boundary so bars stay aligned with SZ2 bins at ±EB, ±3EB, …
    """
    BIN_W = 2.0 * ERROR_BOUND          # SZ2/SZ3 quantizer bin width
    all_vals = np.concatenate(
        [rd['err_vals'][~np.isnan(rd['err_vals'])] for rd in results['rounds']]
        + [results['err_decomp']]
    )
    lo_edge = np.floor(np.percentile(all_vals, 0.5)  / BIN_W) * BIN_W - ERROR_BOUND
    hi_edge = np.ceil (np.percentile(all_vals, 99.5) / BIN_W) * BIN_W + ERROR_BOUND
    return np.arange(lo_edge, hi_edge + BIN_W * 0.5, BIN_W)


def _round_metrics(results: dict) -> list:
    """
    Compute per-round quality metrics from existing results data.
    Returns a list of dicts (one per round) with keys:
        psnr  -- PSNR (dB) over all N points
        H     -- Shannon entropy (bits) of the error histogram
        bpe   -- estimated bit rate (bits per element), "if we stopped here":
                   compressed points → H bits each (Huffman lower bound)
                   sensors           → 32-bit value + 3×32-bit coords = 128 bits each
                   remaining cands   → 32-bit full precision (not yet coded)
    """
    y_full     = results['y_full']
    data_range = float(y_full.max() - y_full.min())
    all_rounds = results['rounds']
    metrics    = []

    cum_comp = cum_sens = 0
    for rd in all_rounds:
        # ── PSNR ──────────────────────────────────────────────────────────────
        err = rd['err_vals']
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        psnr = 20 * np.log10(data_range / rmse) if rmse > 0 else np.inf

        # ── Shannon entropy of this round's error histogram ────────────────
        # Use SZ2-width bins (2×EB) so entropy is measured over quantizer bins.
        BIN_W = 2.0 * ERROR_BOUND
        vals_clean = err[~np.isnan(err)]
        edges_h = np.arange(
            np.floor(vals_clean.min() / BIN_W) * BIN_W - ERROR_BOUND,
            np.ceil (vals_clean.max() / BIN_W) * BIN_W + ERROR_BOUND + BIN_W * 0.5,
            BIN_W)
        counts, _ = np.histogram(vals_clean, bins=edges_h)
        probs = counts / counts.sum()
        H = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

        # ── Estimated bit rate ─────────────────────────────────────────────
        cum_comp += rd['n_comp']
        cum_sens += K_PER_ROUND
        n_remain  = N - cum_comp - cum_sens
        bits = (cum_comp  * H          # compressed: Huffman lower bound
              + cum_sens  * (32 + 96)  # sensors: value + 3D coords at float32
              + n_remain  * 32)        # unhandled: full precision
        bpe = bits / N

        metrics.append(dict(psnr=psnr, H=H, bpe=bpe))

    return metrics


def _proportion_tables(fig, all_rounds: list, current_round: int,
                        metrics: list) -> None:
    """
    Three side-by-side tables:
      Left   — Compressed data counts
      Centre — Sensors placed
      Right  — Quality metrics (PSNR, H, bpe)
    All rounds shown from round 1; current round highlighted; future rows greyed.
    """
    comp_rows = [['Rnd', 'New comp.', 'Cumul.', '% total']]
    sens_rows = [['Rnd', 'Sensors',   'Cumul.', '% total']]
    qual_rows = [['Rnd', 'PSNR (dB)', 'H (bits)', 'bpe (est.)']]
    row_done  = []

    cum_comp = cum_sens = 0
    for i, rd in enumerate(all_rounds):
        done = rd['round'] <= current_round
        row_done.append(done)
        rnd_str = str(rd['round'])
        if done:
            cum_comp += rd['n_comp']
            cum_sens += K_PER_ROUND
            m = metrics[i]
            comp_rows.append([rnd_str, f"{rd['n_comp']:,}",
                               f"{cum_comp:,}", f"{100*cum_comp/N:.1f}%"])
            sens_rows.append([rnd_str, f"{K_PER_ROUND:,}",
                               f"{cum_sens:,}", f"{100*cum_sens/N:.1f}%"])
            qual_rows.append([rnd_str, f"{m['psnr']:.2f}",
                               f"{m['H']:.3f}", f"{m['bpe']:.2f}"])
        else:
            comp_rows.append([rnd_str, '—', '—', '—'])
            sens_rows.append([rnd_str, '—', '—', '—'])
            qual_rows.append([rnd_str, '—', '—', '—'])

    title_y = 0.287
    ax_bot  = 0.01
    ax_h    = 0.265
    # Three equal-width tables across the figure
    w = 0.295
    gaps = [0.01, 0.35, 0.69]
    ax_tc = fig.add_axes([gaps[0], ax_bot, w, ax_h])
    ax_ts = fig.add_axes([gaps[1], ax_bot, w, ax_h])
    ax_tq = fig.add_axes([gaps[2], ax_bot, w, ax_h])

    table_specs = [
        (ax_tc, comp_rows, f'Compressed data  (±{ERROR_BOUND} m/s)'),
        (ax_ts, sens_rows,  'Sensors placed'),
        (ax_tq, qual_rows,  'Quality metrics  (full-domain estimate)'),
    ]
    for (ax_t, rows, title), gx in zip(table_specs, gaps):
        fig.text(gx + w/2, title_y, title,
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_t.axis('off')
        tbl = ax_t.table(cellText=rows[1:], colLabels=rows[0],
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.3)
        n_cols = len(rows[0])
        for col in range(n_cols):
            tbl[(0, col)].set_facecolor('#cccccc')
            tbl[(0, col)].set_text_props(fontweight='bold')
        for row_i, (done, rd) in enumerate(zip(row_done, all_rounds)):
            fc = '#fffbe6' if rd['round'] == current_round else \
                 'white'   if done else '#f0f0f0'
            fw = 'bold' if rd['round'] == current_round else 'normal'
            for col in range(n_cols):
                tbl[(row_i + 1, col)].set_facecolor(fc)
                tbl[(row_i + 1, col)].set_text_props(
                    fontweight=fw, color='black' if done else '#aaaaaa')


def _draw_histogram_fig(results: dict, edges: np.ndarray,
                        current_round: int,
                        overlay: bool = False,
                        include_decomp: bool = False) -> plt.Figure:
    """
    Build one histogram figure.

    overlay=False  (video frames):
        Show only the current round's bars in a single fixed colour.
        Full proportion table visible every frame (future rows greyed out).

    overlay=True   (static summary image):
        All rounds overlaid in distinct colours + decompression curve.
    """
    import matplotlib.patches as mpatches

    all_rounds = results['rounds']
    err_decomp = results['err_decomp']
    cmap       = plt.get_cmap('tab10')

    # Figure: tall enough for histogram + gap + tables
    # Histogram axes: left=8%, bottom=32%, width=89%, height=58%
    # This gives ~8% headroom above ax for suptitle and ~32% below for tables
    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_axes([0.08, 0.34, 0.89, 0.55])

    all_shown_vals = []   # collect for error-range annotation

    if overlay:
        for i, rd in enumerate(all_rounds):
            vals = rd['err_vals'][~np.isnan(rd['err_vals'])]
            all_shown_vals.append(vals)
            counts, _ = np.histogram(vals, bins=edges)
            probs = counts / counts.sum()
            H = -np.sum(probs[probs>0] * np.log2(probs[probs>0]))
            lbl = (f"Round {rd['round']}  ({len(vals):,}/{N:,} pts  |  "
                   f"{rd['frac_comp']:.1f}% compressed)   H={H:.2f} bits")
            ax.stairs(probs, edges, color=cmap(i), alpha=0.55, linewidth=1.8,
                      fill=True, edgecolor=cmap(i), label=lbl)
        if include_decomp:
            counts, _ = np.histogram(err_decomp, bins=edges)
            probs = counts / counts.sum()
            H = -np.sum(probs[probs>0] * np.log2(probs[probs>0]))
            frac_d = 100*(np.abs(err_decomp) < ERROR_BOUND).mean()
            lbl = (f"Decompression  ({len(results['all_sens'])} sensors  |  "
                   f"{frac_d:.1f}% within bound)   H={H:.2f} bits")
            ax.stairs(probs, edges, color=cmap(len(all_rounds)), alpha=0.80,
                      linewidth=2.4, fill=True, edgecolor=cmap(len(all_rounds)),
                      label=lbl)
            all_shown_vals.append(err_decomp)
    else:
        rd   = all_rounds[current_round - 1]
        vals = rd['err_vals'][~np.isnan(rd['err_vals'])]
        all_shown_vals.append(vals)
        counts, _ = np.histogram(vals, bins=edges)
        probs = counts / counts.sum()
        H = -np.sum(probs[probs>0] * np.log2(probs[probs>0]))
        lbl = (f"Round {rd['round']}  ({len(vals):,}/{N:,} pts  |  "
               f"{rd['frac_comp']:.1f}% compressed)   H={H:.2f} bits")
        ax.stairs(probs, edges, color=BAR_COLOR, alpha=0.75, linewidth=1.8,
                  fill=True, edgecolor=BAR_COLOR, label=lbl)

    # ── Gold compressed zone (added to legend as a proxy patch) ───────────────
    ax.axvspan(-ERROR_BOUND, ERROR_BOUND, color='gold', alpha=0.18, zorder=0)
    gold_patch = mpatches.Patch(color='gold', alpha=0.5,
                                label=f'Compressed zone  (±{ERROR_BOUND} m/s)')

    # SZ2 bin boundaries are now the bar edges themselves (bin width = 2×EB).
    # No extra grid lines needed — the bar gaps show every quantizer boundary.

    # ── Accept bound: ±ACCEPT_BOUND dashed lines ─────────────────────────────
    # Show where our acceptance criterion sits (ACCEPT_BINS × ERROR_BOUND).
    ax.axvline( ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.8)
    ax.axvline(-ACCEPT_BOUND, color='crimson', lw=1.4, ls='--', zorder=3, alpha=0.8)
    accept_patch = mpatches.Patch(
        color='crimson', alpha=0.7,
        label=f'Accept bound  (±{ACCEPT_BOUND:.3f} m/s,  {ACCEPT_BINS}×EB)')

    # ── SZ2 unpredictable boundary ────────────────────────────────────────────
    # SZ2_UNPRED_BOUND is usually far outside the histogram x-range (e.g. ±32768 m/s).
    # Always draw the boundary markers: lines if within range, or arrows clamped
    # to the axis edges pointing outward when the bound is off-chart.
    combined   = np.concatenate(all_shown_vals)
    n_unpred   = int(np.sum(np.abs(combined) > SZ2_UNPRED_BOUND))
    pct_unpred = 100.0 * n_unpred / max(len(combined), 1)
    xlo, xhi   = edges[0], edges[-1]

    for sign, x_bound, x_edge, ha_arrow in [
            (-1, -SZ2_UNPRED_BOUND, xlo, 'left'),
            (+1,  SZ2_UNPRED_BOUND, xhi, 'right')]:
        if sign * x_bound <= sign * x_edge:
            # Bound is within the visible range — draw a normal vertical line
            ax.axvline(x_bound, color='purple', lw=1.6, ls='--', zorder=3)
        else:
            # Bound is off-chart — draw an arrow at the axis edge pointing outward
            ax.annotate(
                '',
                xy       =(x_edge, 0.92),
                xycoords =('data', 'axes fraction'),
                xytext   =(x_edge - sign * (xhi - xlo) * 0.04, 0.92),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.6),
                annotation_clip=False)
            ax.axvline(x_edge, color='purple', lw=1.6, ls='--', zorder=3, alpha=0.7)

    # Count annotation just above the error-range box
    ax.annotate(
        (f'SZ2 unpredictable (|err| > {SZ2_UNPRED_BOUND:.0f} m/s):  '
         f'{n_unpred:,} pts  ({pct_unpred:.2f}%)'),
        xy=(0.99, 0.09), xycoords='axes fraction',
        fontsize=8.0, ha='right', va='bottom', color='purple',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))

    # ── Labels / legend / grid ────────────────────────────────────────────────
    ax.set_xlabel('Prediction error  (m/s)', fontsize=12)
    ax.set_ylabel('Fraction of points', fontsize=12)
    title_sfx = 'all rounds overlaid' if overlay else f'Round {current_round} of {N_ROUNDS}'
    ax.set_title(
        f'Progressive GP — Error Histograms  ({title_sfx})\n'
        f'Each bar = one SZ2/SZ3 quantizer bin (width 2×EB={2*ERROR_BOUND} m/s) | '
        f'Red = centre-bin boundary (±EB) | Purple = SZ2 unpredictable limit (±{SZ2_UNPRED_BOUND:.0f} m/s)',
        fontsize=10, pad=8)
    ax.set_xlim(xlo, xhi)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    # Legend in upper-left; add gold-zone and accept-threshold proxies
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [gold_patch, accept_patch],
              labels + [gold_patch.get_label(), accept_patch.get_label()],
              fontsize=8.5, loc='upper left', framealpha=0.85)

    # ── Error range: bottom-right inside axes ─────────────────────────────────
    ax.annotate(
        f'error range:  [{combined.min():.3f},  {combined.max():.3f}] m/s',
        xy=(0.99, 0.03), xycoords='axes fraction',
        fontsize=8.5, ha='right', va='bottom', color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))

    metrics = _round_metrics(results)
    _proportion_tables(fig, all_rounds, current_round, metrics)
    return fig


def plot_histograms(results: dict) -> None:
    """Static overlaid image — all rounds in different colours + decompression."""
    edges = _compute_edges(results)
    fig   = _draw_histogram_fig(results, edges,
                                current_round=N_ROUNDS,
                                overlay=True, include_decomp=True)
    fig.savefig(OUT_HISTS, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram plot: {OUT_HISTS}")


def plot_histograms_video(results: dict) -> None:
    """
    One frame per round: single-colour bars that change each frame.
    Full proportion table shown every frame (future rows greyed out).
    After all frames, also saves the overlaid static image.
    """
    rounds    = results['rounds']
    edges     = _compute_edges(results)
    frame_dir = ARGONNE / "frames_hists"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []

    for rd in rounds:
        fig   = _draw_histogram_fig(results, edges,
                                    current_round=rd['round'],
                                    overlay=False, include_decomp=False)
        fpath = frame_dir / f'hist_{rd["round"]:03d}.png'
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(fpath)
        print(f"  Saved histogram frame {rd['round']}")

    _frames_to_video(frame_paths,
                     ARGONNE / "progressive_gp_histograms.mp4",
                     ARGONNE / "progressive_gp_histograms.gif", fps=1)

    # Save the final overlaid image regardless
    plot_histograms(results)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — 3-D VOLUME COMPARISON  (True | GP reconstruction)
# ─────────────────────────────────────────────────────────────────────────────
#
# Reproduces the same stacked-horizontal-planes style used in tdeim_hurricane.py
# (Figure 4):  for each z-level a flat surface is drawn with facecolors from
# RdYlBu_r, alpha ∝ |field|^0.25 so calm regions are transparent.
# Two side-by-side 3D subplots: left = true data, right = GP reconstruction.
#
# Parameters controlling render quality vs. speed:
VIZ_DS     = 3   # extra spatial downsampling applied to the 3D plot only
                  # (1 = full NX×NY res, 3 = ~3× faster; ≤5 for legibility)
VIZ_STEP   = 2   # render every VIZ_STEP-th z-level (1 = all 100, 2 = 50, …)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_3d_vol(ax, vol3d: np.ndarray, vmin: float, vmax: float,
                 title: str, sensor_idx: np.ndarray | None = None) -> None:
    """
    Render vol3d (NZ, NY, NX) as stacked horizontal planes on a 3D axes.

    Each plane is drawn at its z-level using plot_surface with per-pixel
    RGBA facecolors.  Alpha is proportional to |field| so calm (near-zero)
    regions are semi-transparent and structure stands out.

    Parameters
    ----------
    ax         : Axes3D instance
    vol3d      : (NZ, NY, NX) array in original units
    vmin, vmax : colour limits for the field
    title      : axes title string
    sensor_idx : optional (k,) global flat indices — scatter as black dots
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Spatial downsample for rendering speed
    vol_s = vol3d[:, ::VIZ_DS, ::VIZ_DS]
    nZ_v, nY_v, nX_v = vol_s.shape

    # Build meshgrid for one plane (x=longitude, y=latitude)
    x_plane = np.arange(nX_v) * VIZ_DS   # pixel coords in original grid
    y_plane = np.arange(nY_v) * VIZ_DS
    XX, YY  = np.meshgrid(x_plane, y_plane)   # (nY_v, nX_v)

    norm3d  = plt.Normalize(vmin=vmin, vmax=vmax)
    p99     = max(abs(vmin), abs(vmax), 1e-9)

    for iz in range(0, NZ, VIZ_STEP):
        iz_s   = iz // VIZ_DS           # index in the downsampled array
        if iz_s >= nZ_v:
            continue
        plane  = vol_s[iz_s]             # (nY_v, nX_v)

        rgba   = plt.cm.RdYlBu_r(norm3d(plane))      # (nY_v, nX_v, 4)
        # Alpha: stronger field → more opaque; near-zero → ghost-like
        alpha  = np.clip((np.abs(plane) / p99) ** 0.3, 0.06, 0.92)
        rgba[:, :, 3] = alpha

        ZZ = np.full_like(XX, iz, dtype=float)
        ax.plot_surface(XX, YY, ZZ, facecolors=rgba, shade=False,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)

    # Sensor scatter: project all sensors onto their true (x, y, z) positions
    if sensor_idx is not None and len(sensor_idx) > 0:
        sz_idx = sensor_idx % (NY * NX)
        sy_pix = sz_idx // NX
        sx_pix = sz_idx %  NX
        sz_lev = sensor_idx // (NY * NX)
        ax.scatter(sx_pix, sy_pix, sz_lev, c='k', s=8, marker='o',
                   alpha=0.7, zorder=6, depthshade=True)

    ax.set_xlabel('x  (lon.)',    fontsize=8, labelpad=4)
    ax.set_ylabel('y  (lat.)',    fontsize=8, labelpad=4)
    ax.set_zlabel('z  (alt.)',    fontsize=8, labelpad=4)
    ax.tick_params(labelsize=6)
    ax.set_zlim(0, NZ)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.view_init(elev=22, azim=-55)


def plot_3d_comparison_video(results: dict, vol: np.ndarray) -> None:
    """
    One PNG per round: left = true volume, right = GP reconstruction.
    Saves to frames_3d/vol3d_{r:03d}.png and stitches into an MP4/GIF.
    """
    rounds   = results['rounds']
    frame_dir = ARGONNE / "frames_3d"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []

    # Colour limits fixed to global field range for comparability across frames
    vmin_t = float(np.percentile(vol, 1))
    vmax_t = float(np.percentile(vol, 99))

    # Accumulated sensor list so far (for sensor scatter)
    acc_sensors: list[np.ndarray] = []

    for rd in rounds:
        acc_sensors.append(rd['sensor_idx'])
        all_sens_so_far = np.concatenate(acc_sensors)

        pred3d = rd['pred_vol'].reshape(NZ, NY, NX)
        rmse_r = float(np.sqrt(np.mean((vol.ravel() - rd['pred_vol'])**2)))
        psnr_r = 20 * np.log10((vol.max() - vol.min()) / rmse_r) if rmse_r > 0 else np.inf

        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor('#f7f7f7')

        ax_true = fig.add_subplot(1, 2, 1, projection='3d')
        ax_pred = fig.add_subplot(1, 2, 2, projection='3d')

        _draw_3d_vol(ax_true, vol, vmin_t, vmax_t,
                     f'True wind speed\n(full 3-D volume,  {NZ}×{NY}×{NX})')

        _draw_3d_vol(ax_pred, pred3d, vmin_t, vmax_t,
                     f'GP reconstruction — Round {rd["round"]}\n'
                     f'{len(all_sens_so_far):,} anchors  |  '
                     f'{rd["frac_comp"]:.1f}% compressed  |  '
                     f'PSNR = {psnr_r:.1f} dB',
                     sensor_idx=all_sens_so_far)

        # Shared colorbar
        sm = plt.cm.ScalarMappable(
            cmap='RdYlBu_r',
            norm=plt.Normalize(vmin=vmin_t, vmax=vmax_t))
        sm.set_array([])
        fig.colorbar(sm, ax=[ax_true, ax_pred], shrink=0.45, pad=0.04,
                     label='U-wind  (m/s)', orientation='vertical')

        fig.suptitle(
            f'Progressive GP — ISABEL Hurricane Uf48  |  '
            f'{DOWNSAMPLE}× downsampled  |  EB = ±{ERROR_BOUND} m/s  |  '
            f'Round {rd["round"]}/{N_ROUNDS}\n'
            f'Opacity ∝ |wind|  —  black dots = anchor points',
            fontsize=10)

        fpath = frame_dir / f'vol3d_{rd["round"]:03d}.png'
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(fpath)
        print(f"  Saved 3D frame {rd['round']}: {fpath.name}")

    _frames_to_video(frame_paths,
                     ARGONNE / "progressive_gp_3d.mp4",
                     ARGONNE / "progressive_gp_3d.gif", fps=1)


def plot_3d_comparison(results: dict, vol: np.ndarray) -> None:
    """
    Static version: uses the last round's reconstruction.
    Saves to progressive_gp_3d.png.
    """
    rd = results['rounds'][-1]
    all_sens = results['all_sens']
    pred3d   = rd['pred_vol'].reshape(NZ, NY, NX)

    rmse_r = float(np.sqrt(np.mean((vol.ravel() - rd['pred_vol'])**2)))
    psnr_r = 20 * np.log10((vol.max() - vol.min()) / rmse_r) if rmse_r > 0 else np.inf

    vmin_t = float(np.percentile(vol, 1))
    vmax_t = float(np.percentile(vol, 99))

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#f7f7f7')

    ax_true = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')

    _draw_3d_vol(ax_true, vol, vmin_t, vmax_t,
                 f'True wind speed  (full 3-D volume,  {NZ}×{NY}×{NX})')

    _draw_3d_vol(ax_pred, pred3d, vmin_t, vmax_t,
                 f'GP reconstruction — Round {rd["round"]} (final)\n'
                 f'{len(all_sens):,} anchors  |  '
                 f'{rd["frac_comp"]:.1f}% compressed  |  '
                 f'PSNR = {psnr_r:.1f} dB',
                 sensor_idx=all_sens)

    sm = plt.cm.ScalarMappable(
        cmap='RdYlBu_r',
        norm=plt.Normalize(vmin=vmin_t, vmax=vmax_t))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_true, ax_pred], shrink=0.45, pad=0.04,
                 label='U-wind  (m/s)', orientation='vertical')

    fig.suptitle(
        f'Progressive GP — ISABEL Hurricane Uf48  |  '
        f'{DOWNSAMPLE}× downsampled  |  EB = ±{ERROR_BOUND} m/s\n'
        f'Opacity ∝ |wind|  —  black dots = anchor points',
        fontsize=10)

    out = ARGONNE / "progressive_gp_3d.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D comparison: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT: SAVE & LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(results: dict, vol: np.ndarray) -> None:
    """
    Pickle the results dict and the downsampled volume to CHECKPOINT_FILE.
    The filename encodes N_ROUNDS, K_PER_ROUND, ERROR_BOUND, DOWNSAMPLE so
    that changing any parameter automatically uses a fresh file.
    """
    import pickle
    payload = {'results': results, 'vol': vol,
               'hyperparams': dict(ls_z=LS_Z, ls_xy=LS_XY, sig2=SIG2)}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(payload, f, protocol=4)
    size_mb = CHECKPOINT_FILE.stat().st_size / 1e6
    print(f"Checkpoint saved: {CHECKPOINT_FILE.name}  ({size_mb:.0f} MB)")


def load_checkpoint() -> tuple[dict, np.ndarray]:
    """
    Load results and vol from CHECKPOINT_FILE.
    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    import pickle
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            f"Run with PLOTS_ONLY = False first to generate it.")
    with open(CHECKPOINT_FILE, 'rb') as f:
        payload = pickle.load(f)
    print(f"Checkpoint loaded: {CHECKPOINT_FILE.name}")
    # Restore fitted hyperparameters if present
    if 'hyperparams' in payload:
        global LS_Z, LS_XY, SIG2
        hp = payload['hyperparams']
        LS_Z, LS_XY, SIG2 = hp['ls_z'], hp['ls_xy'], hp['sig2']
        print(f"  Hyperparams restored: LS_Z={LS_Z:.4f}  "
              f"LS_XY={LS_XY:.4f}  SIG2={SIG2:.4f}")
    return payload['results'], payload['vol']


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    if PLOTS_ONLY:
        # ── Skip all computation; load saved results ──────────────────────────
        print("PLOTS_ONLY mode — loading checkpoint …")
        results, vol = load_checkpoint()
    else:
        # ── Run the full progressive GP pipeline ──────────────────────────────
        rng = np.random.default_rng(SEED)
        print("Loading Uf48 wind speed …")
        vol    = load_wind()
        coords = build_coords()
        print(f"Data loaded.  Shape: {vol.shape}")

        # ── Fit 3-D kernel hyperparameters (once; cached to disk) ────────────
        hp_file = ARGONNE / f"pgp_hyperparams_ds{DOWNSAMPLE}_matern52.pkl"
        if REFIT_HYPERPARAMS and hp_file.exists():
            hp_file.unlink()
            print("Deleted hyperparams cache — will refit via MLE.")
        if hp_file.exists():
            import pickle as _pkl
            hp = _pkl.load(open(hp_file, 'rb'))
            LS_Z, LS_XY, SIG2 = hp['ls_z'], hp['ls_xy'], hp['sig2']
            print(f"\nHyperparams loaded from cache: LS_Z={LS_Z:.4f}  "
                  f"LS_XY={LS_XY:.4f}  SIG2={SIG2:.4f}")
        else:
            y_norm_fit = (vol.ravel() - vol.mean()) / vol.std()
            print(f"\nFitting 3-D hyperparameters on subsample …"
                  f"  (defaults: LS_Z={LS_Z}, LS_XY={LS_XY})")
            hp = fit_hyperparams_3d(coords, y_norm_fit,
                                    n_fit=1500, n_restarts=4, rng=rng)
            import pickle as _pkl
            _pkl.dump(hp, open(hp_file, 'wb'), protocol=4)
            print(f"  Hyperparams cached to: {hp_file.name}")

        if RESUME:
            # ── Resume: find prior checkpoint, reconstruct state, run more rounds
            # Look for any checkpoint matching the current params but any round count
            import glob as _glob
            pattern = str(ARGONNE /
                f"pgp_checkpoint_R*_k{K_PER_ROUND}_eb{ERROR_BOUND}"
                f"_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}_zskip{Z_SKIP_BOTTOM}.pkl")
            candidates = sorted(_glob.glob(pattern))
            if not candidates:
                raise FileNotFoundError(
                    f"No prior checkpoint found matching:\n  {pattern}\n"
                    "Run with RESUME=False first to generate one.")
            # Pick the one with the most rounds (largest R value)
            prior_path = max(candidates,
                             key=lambda p: int(Path(p).stem.split('_')[1][1:]))
            print(f"Loading prior checkpoint: {Path(prior_path).name}")
            import pickle as _pkl
            prior_payload = _pkl.load(open(prior_path, 'rb'))
            prior_results = prior_payload['results']
            if 'hyperparams' in prior_payload:
                hp = prior_payload['hyperparams']
                LS_Z, LS_XY, SIG2 = hp['ls_z'], hp['ls_xy'], hp['sig2']
                print(f"  Hyperparams restored: LS_Z={LS_Z:.4f}  "
                      f"LS_XY={LS_XY:.4f}  SIG2={SIG2:.4f}")
            results = run_progressive(vol, coords, rng, resume_from=prior_results)
            # Repoint checkpoint to reflect total rounds completed
            total_rounds = len(results['rounds'])
            CHECKPOINT_FILE = _checkpoint_path(total_rounds)
        else:
            results = run_progressive(vol, coords, rng)
        save_checkpoint(results, vol)

    if N_ROUNDS <= 3:
        # Static grid: all rounds fit on one figure
        plot_panels(results, vol)
        plot_histograms(results)
        # 3D volume comparison (static: last round vs. true)
        print("\nGenerating 3-D volume comparison …")
        plot_3d_comparison(results, vol)
    else:
        # Video: one frame per round for both plots
        print("\nN_ROUNDS > 3 — generating per-round video frames …")
        plot_panels_video(results, vol)
        plot_histograms_video(results)
        # 3D volume comparison — one frame per round, stitched to MP4/GIF
        print("\nGenerating per-round 3-D volume comparison frames …")
        plot_3d_comparison_video(results, vol)

    print("\nDone.")
