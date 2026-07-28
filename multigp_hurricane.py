"""
multigp_hurricane.py

Multi-output Gaussian Process (Linear Model of Coregionalization, LMC) for
joint reconstruction of multiple ISABEL hurricane variables simultaneously
from sparse point-sensor observations.

VARIABLES (default subset)
---------------------------
  U  — East-west (zonal) wind       [m/s]  +ve = blowing eastward
  V  — North-south (meridional) wind [m/s]  +ve = blowing northward
  W  — Vertical wind                 [m/s]  +ve = updraft
  TC — Temperature                   [°C]
  P  — Pressure perturbation         [Pa]   deviation from background profile

WHY MULTI-OUTPUT?
-----------------
U, V, W, TC, and P are physically coupled: the pressure gradient (P) drives
the horizontal circulation (U, V), which organises vertical convection (W),
which releases latent heat that maintains the temperature anomaly (TC) and
sustains the pressure deficit. A single multi-output GP captures these
cross-variable correlations directly, so observing U at a sensor location
simultaneously constrains V, W, TC, and P there — something independent
per-variable GPs cannot do.

THE LMC KERNEL
--------------
For output variables indexed i, j and spatial locations x, x':

    k((x,i), (x',j)) = B[i,j] × k_spatial(x, x')

where B is a d×d positive-definite cross-variable covariance matrix and
k_spatial is a Matérn-3/2 spatial kernel.  The full (nd × nd) kernel matrix
has Kronecker structure:

    K_full = B ⊗ K_spatial

This is exploited for efficient log-likelihood evaluation and prediction.

TRAINING vs TESTING
-------------------
Training (offline):
  A random subset of vertical levels from f48 provides the snapshot ensemble.
  At each training level, all d variables are observed at all n_spatial grid
  points (fully observed output), allowing efficient Kronecker computations.
  From this ensemble we estimate:
    1. B  — sample cross-variable covariance (normalized space)
    2. ls — spatial lengthscale (via marginal likelihood maximization)
  Sensors are then placed by GKS (QR pivoting on K_spatial).

Testing (online):
  A held-out vertical level is reconstructed from observations of all d
  variables at only k sensor locations.  The GP posterior gives the
  predicted mean (and variance) of the full d-variable field everywhere.

References
----------
  Alvarez & Lawrence (2011). Computationally efficient convolved multiple
    output Gaussian processes. JMLR 12, 1459-1500.
  Bonilla et al. (2008). Multi-task Gaussian process prediction. NeurIPS.
  Saatci (2011). Scalable inference for structured GP models. PhD thesis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/Users/jchen228/Desktop/Argonne/100x500x500"

# Variables: display name -> filename
VARIABLES = {
    'U' : 'Uf48.bin.f32',
    'V' : 'Vf48.bin.f32',
    'W' : 'Wf48.bin.f32',
}

UNITS = {'U': 'm/s', 'V': 'm/s', 'W': 'm/s'}

DOWNSAMPLE          = 3    # spatial downsampling: 500/14 ≈ 36 → ~36×36 = ~1296 pts
N_TRAIN_FACTOR      = 4     # n_train = N_TRAIN_FACTOR × n_sensors
SPLIT_SEED          = 42    # reproducible random split
SHOWCASE_TRAIN_IDX  = 50     # which training level to reconstruct (0 = first train level)

# Sensor count: 1% of spatial grid points (set automatically after loading)
# Override with an integer to fix the count.
N_SENSORS      = None

# GP hyperparameters
NOISE_STD      = 0.05   # observation noise std in NORMALISED units (~5% of signal)
N_LS_RESTARTS  = 3      # restarts for lengthscale MLE optimisation
JITTER         = 1e-6   # numerical jitter added to K_spatial diagonal


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_variable(path, downsample=1):
    """Load one ISABEL variable → (100, ny, nz) float32 array."""
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape((100, 500, 500))
    return data[:, ::downsample, ::downsample]


def load_variables(data_dir, var_dict, downsample=1):
    """
    Load all variables in var_dict.

    Returns
    -------
    data : dict  {name: (100, ny, nz) array}
    """
    data = {}
    for name, fname in var_dict.items():
        path = os.path.join(data_dir, fname)
        data[name] = load_variable(path, downsample)
        print(f"  Loaded {name:>3}  ({data[name].min():.2f} – "
              f"{data[name].max():.2f} {UNITS[name]})")
    return data


def make_grid_coords(ny, nz):
    """
    2D spatial coordinates for an ny × nz grid.
    Returns X of shape (ny*nz, 2) with rows [lat_idx, lon_idx].
    """
    yi = np.arange(ny)
    zi = np.arange(nz)
    YY, ZZ = np.meshgrid(yi, zi, indexing='ij')
    return np.column_stack([YY.ravel(), ZZ.ravel()])   # (n, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def matern32(X1, X2, ls):
    """
    Matérn-3/2 kernel:  k(r) = (1 + √3 r/ℓ) exp(-√3 r/ℓ)

    Parameters
    ----------
    X1 : (n1, 2)  X2 : (n2, 2)  ls : float
    Returns K : (n1, n2)
    """
    diff  = X1[:, None, :] - X2[None, :, :]   # (n1, n2, 2)
    r     = np.sqrt((diff**2).sum(axis=-1))    # (n1, n2)
    v     = np.sqrt(3.0) * r / ls
    return (1.0 + v) * np.exp(-v)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VARIABLE COVARIANCE (B matrix)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_B(Y_train_list):
    """
    Estimate the d×d cross-variable covariance matrix B from training data.

    Each element of Y_train_list is an (n, d) array of NORMALISED residuals
    from one training level.  B is the sample covariance averaged over all
    (level, location) pairs — equivalent to the first term of the MLE
    objective when the spatial kernel is fixed to the identity.

    Parameters
    ----------
    Y_train_list : list of (n, d) arrays

    Returns
    -------
    B : (d, d) symmetric positive-definite matrix
    """
    Y_all = np.vstack(Y_train_list)            # (n_train * n, d)
    B     = (Y_all.T @ Y_all) / len(Y_all)    # (d, d)
    # Ensure symmetry and positive-definiteness
    B = 0.5 * (B + B.T)
    B += JITTER * np.eye(B.shape[0])
    return B


# ─────────────────────────────────────────────────────────────────────────────
# LENGTHSCALE FITTING  (Kronecker marginal likelihood)
# ─────────────────────────────────────────────────────────────────────────────

def _neg_log_ml(log_ls, X, Y_train_list, λ_B, Q_B, noise_var):
    """
    Negative log marginal likelihood for the LMC model, computed efficiently
    using the Kronecker structure B ⊗ K_n.

    For fully-observed training data (all n spatial points at each level):

        (B ⊗ K_n + σ²I) = (Q_B ⊗ Q_n)(Λ_B ⊗ Λ_n + σ²I)(Q_B ⊗ Q_n)^T

    The eigendecomposition of B is precomputed (λ_B, Q_B).
    The eigendecomposition of K_n is computed here for each ls.

    Data transform:
        Z_l = Q_n^T @ Y_l @ Q_B   →   z_l[j, i] has eigenvalue λ_n[j] * λ_B[i]
    """
    ls       = np.exp(log_ls)
    n, d     = len(X), len(λ_B)
    K_n      = matern32(X, X, ls) + JITTER * np.eye(n)
    λ_n, Q_n = np.linalg.eigh(K_n)             # O(n³)

    Λ        = np.outer(λ_n, λ_B) + noise_var  # (n, d) eigenvalues of Σ
    log_det  = np.sum(np.log(np.maximum(Λ, 1e-300)))

    quad = 0.0
    for Y in Y_train_list:
        Z     = Q_n.T @ Y @ Q_B               # (n, d)
        quad += np.sum(Z**2 / Λ)

    n_obs = len(Y_train_list)
    nll   = 0.5 * (n_obs * (n * d * np.log(2 * np.pi) + log_det) + quad)
    return nll


def fit_lengthscale(X, Y_train_list, B, noise_var, n_restarts=3, rng=None):
    """
    Maximise the LMC marginal likelihood over the spatial lengthscale.

    Uses the Kronecker identity so cost per evaluation is O(n³ + n_train·n²d)
    instead of O((nd)³).

    Parameters
    ----------
    X            : (n, 2) spatial coordinates
    Y_train_list : list of (n, d) normalised training arrays (one per level)
    B            : (d, d) coregionalization matrix
    noise_var    : float  observation noise variance
    n_restarts   : int    number of random restarts

    Returns
    -------
    ls : float  optimal lengthscale
    """
    if rng is None:
        rng = np.random.default_rng(0)

    λ_B, Q_B = np.linalg.eigh(B)              # precompute once

    # Median pairwise distance heuristic (median trick) as starting point
    dists = np.sqrt(((X[::5, None] - X[None, ::5])**2).sum(-1)).ravel()
    ls0   = float(np.median(dists[dists > 0]))
    print(f"  Lengthscale search start (median trick): ls₀ = {ls0:.3f} grid units")

    starts    = [np.log(ls0)] + list(np.log(ls0) + rng.uniform(-1, 1, n_restarts - 1))
    best_nll  = np.inf
    best_ls   = ls0

    for i, log_ls0 in enumerate(starts):
        try:
            res = minimize(_neg_log_ml, [log_ls0],
                           args=(X, Y_train_list, λ_B, Q_B, noise_var),
                           method='L-BFGS-B',
                           bounds=[(-1, np.log(max(X.max(), 10) * 3))],
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls  = float(np.exp(res.x[0]))
                print(f"  Restart {i+1}: ls = {best_ls:.4f}  nll = {best_nll:.2f}")
        except Exception as e:
            print(f"  Restart {i+1}: failed ({e})")

    return best_ls


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

def gks_sensors(K_spatial, k):
    """
    GKS: QR column pivoting on K_spatial selects the k locations whose kernel
    rows are most linearly independent — maximising spatial information coverage.
    Ignores B; placement is the same regardless of cross-variable coupling.
    """
    from scipy.linalg import qr
    _, _, p = qr(K_spatial, pivoting=True)
    return p[:k]


# ── greedy_lmc_sensors kept below for reference (not currently active) ────────

def greedy_lmc_sensors(K_n, B, noise_var, k):
    """
    Select k sensor locations by greedy reduction of total ICM posterior variance.

    Motivation
    ----------
    GKS (QR pivoting on K_spatial) ignores B entirely — it optimises spatial
    coverage as if all variables were independent.  This greedy approach uses
    the full ICM posterior, so sensors are placed where they reduce uncertainty
    across ALL d output variables jointly, weighted by their cross-variable
    covariance structure.

    Algorithm
    ---------
    Eigendecompose B = Q_B Λ_B Q_B^T.  In the Q_B-rotated basis the ICM
    decomposes into d INDEPENDENT latent GPs:

        g_i ~ GP(0, λ_B[i] * K_n),  observed at sensors with noise σ²

    At each greedy step the score of candidate j is the total posterior
    variance at j in the original output space:

        score(j) = Σ_i λ_B[i] * k_i^post(j, j)

    After selecting j, each latent posterior is updated by the rank-1 rule:

        k_i^{t+1}(x, y) = k_i^t(x, y) - e_i(x) * e_i(y)
        e_i = k_i^t(:, j) / sqrt(k_i^t(j, j) + σ²)

    The update vectors e_i are accumulated so that subsequent posterior columns
    can be evaluated in O(t) per candidate, keeping the total cost O(k² d n).

    Parameters
    ----------
    K_n       : (n, n)  spatial kernel matrix (Matérn-3/2, already jittered)
    B         : (d, d)  cross-variable covariance
    noise_var : float   observation noise variance σ²
    k         : int     number of sensors to select

    Returns
    -------
    sensors : (k,) int array of selected spatial indices (0-based)
    """
    λ_B, _ = np.linalg.eigh(B)
    n, d   = K_n.shape[0], len(λ_B)

    # For each latent process i, maintain:
    #   diag[i] : (n,) current posterior diagonal k_i^t(x, x)
    #   E[i]    : (n, t) matrix of accumulated update vectors
    diag = [λ_B[i] * np.diag(K_n).copy() for i in range(d)]
    E    = [np.zeros((n, 0)) for _  in range(d)]

    sensors = []
    for _ in range(k):
        # Score each candidate: Σ_i λ_B[i] * posterior variance at j
        score = np.zeros(n)
        for i in range(d):
            score += λ_B[i] * diag[i]
        score[sensors] = -np.inf          # exclude already-selected locations
        j = int(np.argmax(score))
        sensors.append(j)

        # Rank-1 posterior update for each latent process
        for i in range(d):
            # Posterior column at j: k_i^t(:, j) = λ_B[i]*K_n[:,j] - E[i] @ E[i][j,:]
            col_j = λ_B[i] * K_n[:, j] - E[i] @ E[i][j, :]
            denom = float(col_j[j]) + noise_var
            e     = col_j / np.sqrt(max(denom, 1e-12))
            diag[i]  -= e ** 2
            diag[i]   = np.maximum(diag[i], 0.0)   # numerical floor
            E[i]      = np.column_stack([E[i], e])

    return np.array(sensors)


# ─────────────────────────────────────────────────────────────────────────────
# LMC PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def lmc_predict(X_all, X_sensors, Y_sensors_norm, B, ls, noise_var):
    """
    LMC posterior mean at all n spatial locations given sensor observations.

    Model:  f(x) ~ GP(0, B ⊗ k_spatial)
    Observations: Y_sensors_norm ∈ R^{k × d} — normalised values at k sensor
                  locations for all d variables simultaneously.

    Prediction formula (standard GP posterior):
        K_sub   = B ⊗ K_ss  +  σ²I_{kd}      (sensor-sensor covariance + noise)
        K_cross = B ⊗ K_Xs                    (all-locations to sensor covariance)
        μ_*     = K_cross @ K_sub^{-1} @ y_flat

    Efficient computation avoids forming the full nd × nd matrix:
        α_mat  = (d, k) matrix of dual variables per variable
        μ_mat  = K_Xs @ (B @ α_mat).T          (n, d) posterior mean

    Parameters
    ----------
    X_all          : (n, 2) all spatial locations
    X_sensors      : (k, 2) sensor spatial locations
    Y_sensors_norm : (k, d) normalised observations at sensors
    B              : (d, d) cross-variable covariance
    ls             : float  spatial lengthscale
    noise_var      : float  observation noise variance

    Returns
    -------
    mu_norm : (n, d) posterior mean in normalised space
    var_norm: (n, d) posterior marginal variance in normalised space
    """
    n, d = len(X_all), B.shape[0]
    k    = len(X_sensors)

    K_ss  = matern32(X_sensors, X_sensors, ls) + JITTER * np.eye(k)  # (k, k)
    K_Xs  = matern32(X_all,    X_sensors, ls)                        # (n, k)

    # Full sensor-sensor covariance (kd × kd) via Kronecker: B ⊗ K_ss + σ²I
    K_sub = np.kron(B, K_ss) + noise_var * np.eye(k * d)             # (kd, kd)

    # Observations in variable-major order: [var0_sensors; var1_sensors; ...]
    y_flat = Y_sensors_norm.T.ravel()                                 # (kd,)

    # Solve K_sub @ alpha = y_flat
    try:
        L, low = cho_factor(K_sub, lower=True)
        alpha  = cho_solve((L, low), y_flat)                          # (kd,)
    except Exception:
        alpha  = np.linalg.solve(K_sub, y_flat)

    # Posterior mean
    # alpha organised as α_mat[var_idx, loc_idx] ∈ R^{d × k}
    alpha_mat = alpha.reshape(d, k)                                   # (d, k)
    mu_norm   = K_Xs @ (B @ alpha_mat).T                             # (n, d)

    # Posterior marginal variance
    # Var[f_i(x)] = B[i,i]*k(x,x) - k_cross_i(x)^T K_sub^{-1} k_cross_i(x)
    #
    # k_cross_i(x) = B[i,:] ⊗ K_Xs[x,:]  is (kd,) — cross-covariance between
    # output i at prediction location x and all kd observations.
    #
    # For all n locations simultaneously:
    #   K_cross_i  = kron(B[i:i+1,:], K_Xs)   shape (n, kd)
    #   V_i        = K_sub^{-1} @ K_cross_i.T  shape (kd, n)
    #   var[:,i]   = diag(K_cross_i @ V_i) = sum(K_cross_i * V_i.T, axis=1)

    var_norm = np.zeros((n, d))
    k_xx = np.ones(n)   # k(x,x) = 1 for Matérn-3/2 (unit variance by construction)
    for i in range(d):
        K_cross_i = np.kron(B[i:i+1, :], K_Xs)   # (n, kd)
        try:
            V_i = cho_solve((L, low), K_cross_i.T) # (kd, n)
        except Exception:
            V_i = np.linalg.solve(K_sub, K_cross_i.T)
        var_norm[:, i] = np.maximum(
            B[i, i] * k_xx - np.sum(K_cross_i * V_i.T, axis=1),
            0.0
        )

    return mu_norm, var_norm


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_panel(ax, fig, field, title, cmap='RdBu_r',
               vmin=None, vmax=None, cbar_label='',
               sensors_rc=None, ny=None, nz=None):
    im = ax.imshow(field, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    if sensors_rc is not None:
        ax.scatter(sensors_rc[1], sensors_rc[0],
                   c='k', s=18, marker='x', linewidths=0.9,
                   zorder=5, label=f'{len(sensors_rc[0])} sensors')
        ax.legend(loc='lower right', fontsize=6, framealpha=0.6)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel('Longitude (grid index)', fontsize=6.5)
    ax.set_ylabel('Latitude (grid index)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    t_start   = time.perf_counter()
    var_names = list(VARIABLES.keys())
    d         = len(var_names)

    # ── Load all variables ────────────────────────────────────────────────────
    print("Loading ISABEL variables...")
    raw = load_variables(DATA_DIR, VARIABLES, DOWNSAMPLE)

    n_levels_raw = 100
    ny, nz       = raw[var_names[0]].shape[1], raw[var_names[0]].shape[2]
    n            = ny * nz
    X_all        = make_grid_coords(ny, nz)   # (n, 2)

    n_sensors = N_SENSORS if N_SENSORS is not None else max(1, n // 66)
    print(f"\n  Grid: {ny}×{nz} = {n} points  |  "
          f"Variables: {var_names}  |  Sensors: {n_sensors} (1% of {n})")

    # Stack into (n_levels_raw, n, d) tensor — full dataset
    data_tensor_full = np.stack(
        [raw[v].reshape(n_levels_raw, n) for v in var_names], axis=-1
    )   # (100, n, d)

    # ── Skip first 10 levels (near-surface, near-zero fields blow up rel. error)
    SKIP_LEVELS = 10
    valid_idx   = np.arange(SKIP_LEVELS, n_levels_raw)   # levels 10..99
    data_tensor = data_tensor_full[valid_idx]             # (90, n, d)
    n_levels    = len(valid_idx)

    # ── Train / test split (mirrors deim_hurricane.py) ───────────────────────
    # Training size = N_TRAIN_FACTOR × n_sensors, capped to leave ≥10 test lvls.
    # All remaining levels become test levels (not just one held-out target).
    n_train = N_TRAIN_FACTOR * n_sensors
    n_train = min(n_train, n_levels - 10)
    n_train = max(n_train, d + 2)
    n_test  = n_levels - n_train

    rng_s     = np.random.default_rng(SPLIT_SEED)
    train_idx = np.sort(rng_s.choice(n_levels, size=n_train, replace=False))
    test_mask = np.ones(n_levels, dtype=bool)
    test_mask[train_idx] = False
    test_idx  = np.where(test_mask)[0]

    train_data = data_tensor[train_idx]    # (n_train, n, d)
    test_data  = data_tensor[test_idx]     # (n_test,  n, d)

    # Map local test indices back to original level numbers for display
    test_levels_global = valid_idx[test_idx]

    print(f"\n  Using levels {SKIP_LEVELS}–{n_levels_raw-1}  "
          f"(first {SKIP_LEVELS} skipped: near-zero fields inflate rel. error)")
    print(f"  n_train={n_train}  n_test={n_test}  "
          f"(split seed={SPLIT_SEED})")

    # ── Normalise: per-location, per-variable z-score ────────────────────────
    # Subtract training mean and divide by training std at each spatial location
    # independently.  This keeps the GP from being dominated by high-variance
    # regions (eyewall) and makes B a spatial-average correlation matrix.
    train_mean = train_data.mean(axis=0)             # (n, d)
    train_std  = train_data.std(axis=0)              # (n, d)
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)

    def normalise(Y_raw):
        return (Y_raw - train_mean) / train_std      # (n, d)

    def denormalise(Y_norm):
        return Y_norm * train_std + train_mean       # (n, d)

    Y_train_list = [normalise(train_data[l]) for l in range(n_train)]

    # ── Estimate cross-variable covariance B ──────────────────────────────────
    print("\n[Offline] Estimating cross-variable covariance B...")
    B    = estimate_B(Y_train_list)
    corr = B / np.sqrt(np.outer(np.diag(B), np.diag(B)))
    print(f"  Correlation matrix:")
    for i, vi in enumerate(var_names):
        row = '  '.join(f'{corr[i,j]:+.3f}' for j in range(d))
        print(f"    {vi:>3}: {row}")

    # ── Fit spatial lengthscale ───────────────────────────────────────────────
    noise_var = NOISE_STD ** 2
    print(f"\n[Offline] Fitting spatial lengthscale (noise_std={NOISE_STD})...")
    t_ls0 = time.perf_counter()
    ls    = fit_lengthscale(X_all, Y_train_list, B, noise_var,
                            n_restarts=N_LS_RESTARTS)
    print(f"  Optimal lengthscale: {ls:.4f} grid units  "
          f"({time.perf_counter()-t_ls0:.1f}s)")

    # ── Sensor placement (GKS on K_spatial) ──────────────────────────────────
    print(f"\n[Offline] Placing {n_sensors} sensors (GKS)...")
    t_p0      = time.perf_counter()
    K_spatial = matern32(X_all, X_all, ls) + JITTER * np.eye(n)
    sensors   = gks_sensors(K_spatial, n_sensors)
    X_sensors = X_all[sensors]
    print(f"  Sensor placement done  ({time.perf_counter()-t_p0:.2f}s)")

    # ── Online: reconstruct one training slice from sensor observations ────────
    # Testing on a training level provides an upper bound on reconstruction
    # quality — the basis was built partly from this level, so if errors are
    # still large here the model or sensor count needs attention.
    si          = SHOWCASE_TRAIN_IDX
    true_nd     = train_data[si]                    # (n, d)
    lvl_global  = valid_idx[train_idx[si]]          # original level number (10–99)

    print(f"\n[Online]  Reconstructing training level {lvl_global} "
          f"(local train idx {si}) from {n_sensors} sensor observations...")
    t_pred0 = time.perf_counter()
    Y_obs_norm      = normalise(true_nd)[sensors]   # (k, d)
    mu_norm, var_norm = lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var)
    mu_full         = denormalise(mu_norm)           # (n, d)
    print(f"  Done  ({time.perf_counter()-t_pred0:.2f}s)")

    # Per-variable pointwise absolute errors (all computed field by field)
    abs_errs = {}   # keyed by varname: dict with 'rmse', 'max', 'mae', 'range'
    for vi, vname in enumerate(var_names):
        diff  = mu_full[:, vi] - true_nd[:, vi]
        rmse  = float(np.sqrt(np.mean(diff ** 2)))
        maxae = float(np.max(np.abs(diff)))
        mae   = float(np.mean(np.abs(diff)))
        frange = float(true_nd[:, vi].max() - true_nd[:, vi].min())
        abs_errs[vname] = {'rmse': rmse, 'max': maxae, 'mae': mae, 'range': frange}

    col = 14
    hdr = (f"  {'Var':<5} {'Units':<6} {'Field range':>{col}} "
           f"{'RMSE':>{col}} {'Max |error|':>{col}} {'MAE':>{col}}")
    sep = '  ' + '─' * (len(hdr) - 2)
    print(f"\n{'═' * len(hdr)}")
    print(f"  Training level {lvl_global}  —  {n_sensors} sensors  —  pointwise absolute error")
    print(sep)
    print(hdr)
    print(sep)
    for vname in var_names:
        e = abs_errs[vname]
        u = UNITS[vname]
        rng_str  = f"{e['range']:.4f} {u}"
        rmse_str = f"{e['rmse']:.4f} {u}"
        max_str  = f"{e['max']:.4f} {u}"
        mae_str  = f"{e['mae']:.4f} {u}"
        print(f"  {vname:<5} {u:<6} {rng_str:>{col}} {rmse_str:>{col}} {max_str:>{col}} {mae_str:>{col}}")
    print(sep)
    print(f"{'═' * len(hdr)}")
    print(f"\n  Total wall time: {time.perf_counter()-t_start:.1f}s")

    # ── Figure: true vs reconstruction for the training slice ─────────────────
    #
    # Layout: 3 rows × d columns
    #   Row 0 — True field       (training level ground truth)
    #   Row 1 — GP reconstruction (posterior mean from k sensor observations)
    #   Row 2 — Absolute error   |reconstruction − truth|
    # ─────────────────────────────────────────────────────────────────────────
    sensors_rc  = (sensors // nz, sensors % nz)
    global_emax = max(
        np.abs(mu_full[:, vi] - true_nd[:, vi]).max() for vi in range(d)
    )

    fig1, axes = plt.subplots(3, d, figsize=(5 * d, 13),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.38})
    if d == 1:
        axes = axes[:, np.newaxis]

    fig1.suptitle(
        'ISABEL Hurricane — ICM Multi-output GP (GKS sensors)\n'
        f'Variables: {", ".join(var_names)}  |  '
        f'Training level {lvl_global} used as test  |  '
        f'Sensors: {n_sensors}  |  ls = {ls:.2f}',
        fontsize=10, y=0.99
    )

    row_labels = ['TRUE FIELD\n(training level ground truth)',
                  f'GP RECONSTRUCTION\n({n_sensors} GKS sensors, all variables jointly)',
                  'ABSOLUTE ERROR\n|reconstruction − truth|']
    row_colors = ['#1a4a8a', '#2a7a2a', '#8a1a1a']

    for row in range(3):
        axes[row, 0].annotate(
            row_labels[row],
            xy=(-0.42, 0.5), xycoords='axes fraction',
            fontsize=8, fontweight='bold', color=row_colors[row],
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=row_colors[row], linewidth=1.5)
        )

    for vi, vname in enumerate(var_names):
        unit     = UNITS[vname]
        true_2d  = true_nd[:, vi].reshape(ny, nz)
        recon_2d = mu_full[:, vi].reshape(ny, nz)
        err_2d   = np.abs(recon_2d - true_2d)
        vmin_v, vmax_v = true_2d.min(), true_2d.max()
        errs_v   = abs_errs[vname]

        plot_panel(axes[0, vi], fig1, true_2d,
                   title=f'{vname} — True\n({vmin_v:.1f} to {vmax_v:.1f} {unit})',
                   vmin=vmin_v, vmax=vmax_v, cbar_label=unit)

        plot_panel(axes[1, vi], fig1, recon_2d,
                   title=(f'{vname} — GP recon.\n'
                          f'RMSE={errs_v["rmse"]:.4f}  max={errs_v["max"]:.4f} {unit}'),
                   vmin=vmin_v, vmax=vmax_v, cbar_label=unit,
                   sensors_rc=sensors_rc)

        plot_panel(axes[2, vi], fig1, err_2d,
                   title=f'{vname} — |Error|\nmax = {err_2d.max():.2f} {unit}',
                   cmap='hot_r', vmin=0, vmax=global_emax, cbar_label=unit)

    fig1.text(
        0.5, 0.005,
        f'Error colorscale shared across all variables (0 – {global_emax:.2f} m/s).  '
        f'Sensors (×) placed by GKS on K_spatial.',
        ha='center', fontsize=7.5, style='italic', color='#555555'
    )
    fig1.savefig('/Users/jchen228/Desktop/Argonne/multigp_results.png',
                 dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: multigp_results.png")

    # ── Figure 2: B correlation heatmap ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(d)); ax2.set_xticklabels(var_names, fontsize=11)
    ax2.set_yticks(range(d)); ax2.set_yticklabels(var_names, fontsize=11)
    for i in range(d):
        for j in range(d):
            ax2.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                     fontsize=10, color='white' if abs(corr[i,j]) > 0.5 else 'black')
    fig2.colorbar(im, ax=ax2, label='Correlation')
    ax2.set_title(
        'Cross-variable correlation matrix B\n'
        f'(z-score space, {n_train} training levels)',
        fontsize=9
    )
    fig2.tight_layout()
    fig2.savefig('/Users/jchen228/Desktop/Argonne/multigp_correlation.png',
                 dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: multigp_correlation.png")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — SZ-style quantization histogram (per variable, GP residuals)
    # ═════════════════════════════════════════════════════════════════════════
    # bin_idx = round(residual / (2 × sz_eb))
    # SZ default: 1024 bins, outlier threshold ±512.
    # sz_eb is set to SZ_REL_TOL × field_range per variable (SZ REL-mode equiv.).
    # Each subplot shows one output variable independently.
    # ─────────────────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches

    SZ_N_BINS  = 1024
    _half      = SZ_N_BINS // 2   # 512
    SZ_REL_TOL = 0.01             # 1% relative error bound — SZ REL-mode comparison

    fig3, axes3 = plt.subplots(1, d, figsize=(5 * d, 5), constrained_layout=True)
    if d == 1:
        axes3 = [axes3]

    fig3.suptitle(
        'SZ-Style Quantization Histogram  —  GP residuals per variable\n'
        f'bin_idx = round(residual / 2·eb)   '
        f'SZ codebook: {SZ_N_BINS} bins  (outlier threshold ±{_half})   '
        f'eb = SZ_REL_TOL × field_range  ({SZ_REL_TOL*100:.0f}% relative)',
        fontsize=10
    )

    for vi, vname in enumerate(var_names):
        ax   = axes3[vi]
        unit = UNITS[vname]
        resid  = mu_full[:, vi] - true_nd[:, vi]          # (n,) original units
        frange = float(true_nd[:, vi].max() - true_nd[:, vi].min())
        sz_eb  = SZ_REL_TOL * frange

        bins_idx = np.round(resid / (2.0 * sz_eb)).astype(np.int64)
        n_total  = bins_idx.size
        n_out    = int(np.sum(np.abs(bins_idx) > _half))
        n_zero   = int(np.sum(bins_idx == 0))
        pct_out  = 100.0 * n_out / n_total
        n_unique = int(np.unique(bins_idx[np.abs(bins_idx) <= _half]).size)

        disp   = max(10, min(200, int(np.percentile(np.abs(bins_idx), 99)) + 5))
        counts, edges = np.histogram(bins_idx.clip(-disp, disp),
                                     bins=range(-disp, disp + 2))
        centers    = 0.5 * (edges[:-1] + edges[1:])
        bar_colors = ['#d62728' if abs(c) >= disp else '#2ca02c' for c in centers]

        ax.bar(centers, counts, width=0.9, color=bar_colors, zorder=2)
        for sign in (-1, 1):
            ax.axvline(sign * min(_half, disp), color='black', ls='--', lw=1.4, zorder=3)
        ax.set_xlabel('Bin index', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.legend(handles=[
            mpatches.Patch(color='#2ca02c', label='In-range  (compressible)'),
            mpatches.Patch(color='#d62728', label=f'Outlier  (|bin| ≥ {disp})'),
            plt.Line2D([0], [0], color='black', ls='--', lw=1.4,
                       label=f'SZ boundary ±{_half}'),
        ], fontsize=7.5, loc='upper right')
        ax.set_title(
            f'{vname}  ({unit})\n'
            f'eb = {SZ_REL_TOL*100:.0f}% × {frange:.3f} {unit} = {sz_eb:.4f} {unit}\n'
            f'outliers: {n_out:,} / {n_total:,} ({pct_out:.1f}%)   '
            f'unique bins: {n_unique}   '
            f'zero-bin: {n_zero:,} ({100*n_zero/n_total:.1f}%)',
            fontsize=8.5
        )

    out3 = '/Users/jchen228/Desktop/Argonne/multigp_quantization_hist.png'
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out3}")
