"""
deim_hurricane.py  [v3]

Reconstructs ISABEL hurricane U-wind fields using two reduced-order methods:

  1. Q-DEIM  — Discrete Empirical Interpolation with QR-based sensor placement.
               Mirrors the Q-DEIM block in script_sst.m (gpoed-code-python).

  2. R-DEIM  — Randomized DEIM (Saibaba 2020, arXiv:1903.00911).
               Replaces the full SVD with a randomized range finder.

SENSOR COUNT FROM ERROR BOUND
------------------------------
k (number of sensors) is set automatically by ERROR_TOL. Specifically:

    adaptive_k finds the smallest k such that:
        sqrt(1 - sum(s[:k]^2) / sum(s^2)) <= ERROR_TOL

This is the relative Frobenius-norm POD projection error over the training
ensemble. By the Eckart-Young theorem, the rank-k truncation minimises this
error over all rank-k approximations, so it is the tightest possible bound
given the training data.

Limitation: the bound is certified only for training snapshots. For test
snapshots, it is a heuristic — the standard approach in DEIM practice.
The DEIM error constant (||U_k[sensors,:]^+||_2) further bounds how much
the interpolation step amplifies the projection error.

TRAINING vs TESTING
--------------------
Step 1 (pilot):  SVD all 100 levels to estimate k.  This sets training
                 set size = N_TRAIN_FACTOR * k.  Slight data leakage
                 (test levels inform set SIZE but not the BASIS).

Step 2 (split):  Randomly select n_train levels for training; the remaining
                 n_test levels are held out completely.

Step 3 (offline): SVD of the (n x n_train) snapshot matrix from training
                  levels only.  adaptive_k -> sensors fixed here.

Step 4 (online):  Each test level is reconstructed from k sensor values alone.
                  Relative error reported for every test level.

SHOWCASE OUTPUT
---------------
Three representative test cases are shown (defined by Q-DEIM error):
  - Lowest error  : best-case reconstruction
  - Median error  : typical reconstruction
  - Highest error : worst-case reconstruction

Median (not mean) is used as the central tendency metric.  Reason: the error
distribution across vertical levels is right-skewed — a few physically
complex levels (e.g., near-surface boundary layer, eye-wall peak) have much
higher error than the smooth mid-atmosphere majority.  Mean is pulled toward
those outliers and overstates typical reconstruction difficulty.  Median
gives the 50th-percentile user experience: what a randomly chosen test level
looks like.  The maximum already captures worst-case behaviour explicitly.

References
----------
  - Drmac & Gugercin (2016). SIAM J. Sci. Comput. 38(2), A631-A648.
  - Saibaba (2020). SIAM J. Sci. Comput. 42(3), A1582-A1608. arXiv:1903.00911
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH        = "/Users/jchen228/Desktop/Argonne/100x500x500/Uf48.bin.f32"
DOWNSAMPLE       = 3       # keep every Nth pixel; 14 -> ~36x36 = 1296 points/level
NORMALIZE        = True     # subtract training mean before decomposing

# ── Adaptive sensor count from POD energy bound ───────────────────────────────
ERROR_TOL        = 1e-2     # relative projection error threshold (5%)
# MAX_SENSORS is set automatically to 1% of the grid point count after loading.
# Override here if you want a fixed cap instead (set to None to use 1% rule).
MAX_SENSORS      = None

# ── Train / test split ────────────────────────────────────────────────────────
N_TRAIN_FACTOR   = 4        # n_train = N_TRAIN_FACTOR * k_pilot (3-5 recommended)
SPLIT_SEED       = 42       # reproducible random split

# ── Method selection ─────────────────────────────────────────────────────────
# Add or remove names to run only the methods you want.
# Options: 'Q-DEIM', 'R-DEIM'
# The figure columns adjust automatically to however many methods are active.
# Showcase rows (best / median / worst) are defined by the first listed method.
METHODS = ['Q-DEIM']

# ── R-DEIM randomization (only used when 'R-DEIM' is in METHODS) ─────────────
RDEIM_OVERSAMPLE = 10       # oversampling p
RDEIM_N_ITER     = 1        # subspace iterations q
RDEIM_SEED       = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_levels(path, downsample=1):
    """Load all 100 vertical levels from an ISABEL .bin.f32 file -> (100, ny, nz)."""
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape((100, 500, 500))
    return data[:, ::downsample, ::downsample]


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_k(singular_values, tol, max_k=None):
    """
    Smallest k such that relative POD projection error <= tol.

    Criterion:  sqrt(1 - sum(s[:k]^2) / sum(s^2)) <= tol
    Mirrors example1.m from rdeim repo:
        tolrank = find(tol >= sqrt(1-cumsum(s.^2)/sum(s.^2)), 1, 'first')

    Returns
    -------
    k        : int      number of modes selected
    proj_err : (T,) array  projection error at each possible k
    """
    s2       = singular_values ** 2
    cumul    = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask     = proj_err <= tol
    k        = int(np.argmax(mask)) + 1 if mask.any() else len(singular_values)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


# ─────────────────────────────────────────────────────────────────────────────
# Q-DEIM  (mirrors script_sst.m + subsetselection.m from rdeim repo)
# ─────────────────────────────────────────────────────────────────────────────

def qdeim_place(U_k):
    """
    Q-DEIM sensor placement: QR with column pivoting on U_k^T.

    Mirrors script_sst.m: [~,~,pivot] = qr(Phi(:,1:r)','vector'); sensors=pivot(1:r)
    Also mirrors subsetselection(V,'pqr') from rdeim:
        [~,~,p] = qr(V',0);  p = p(1:k);  err = norm(pinv(V(p,:)));

    Returns
    -------
    sensors        : (k,) int array  sensor indices (0-based)
    deim_err_const : float  ||U_k[sensors,:]^+||_2  (DEIM error constant)
    """
    k = U_k.shape[1]
    _, _, p        = scipy_qr(U_k.T, pivoting=True)
    sensors        = p[:k]
    deim_err_const = np.linalg.norm(np.linalg.pinv(U_k[sensors, :]))
    return sensors, deim_err_const


def qdeim_reconstruct(basis, sensors, y_meas):
    """
    Reconstruct full field from k sensor measurements.

    Mirrors script_sst.m: xls = Phi(:,1:r) * (Phi(sensors,1:r) \\ x_test(sensors))
    Solves basis[sensors,:] @ c = y_meas  then returns basis @ c.
    """
    c = np.linalg.solve(basis[sensors, :], y_meas)
    return basis @ c


# ─────────────────────────────────────────────────────────────────────────────
# R-DEIM  (mirrors rangefinder.m from rdeim/rand/)
# ─────────────────────────────────────────────────────────────────────────────

def randomized_range_finder(A, k, p=10, q=1, rng=None):
    """
    Randomized range finder: approximate top-k left singular subspace of A.

    Mirrors rangefinder.m (rdeim/rand/rangefinder.m):
        Omega = randn(n, k+p);
        Y = A*Omega; [Q,~] = qr(Y, 0);
        for i = 1:q
            Y = A'*Q; [Q,~] = qr(Y,0);
            Y = A*Q;  [Q,~] = qr(Y,0);
        end
        B = Q'*A; [u,~,~] = svd(B,0); Q = Q*u(:,1:k);

    Returns
    -------
    Q        : (m, k) orthonormal approximate basis
    s_approx : (k,) approximate singular values
    """
    if rng is None:
        rng = np.random.default_rng(0)
    m, n = A.shape
    kp   = min(k + p, n)               # can't exceed number of snapshots
    k    = min(k, n - 1)               # same cap on target rank

    Omega    = rng.standard_normal((n, kp))
    Y        = A @ Omega
    Q, _     = np.linalg.qr(Y, mode='reduced')
    for _ in range(q):
        Z, _ = np.linalg.qr(A.T @ Q, mode='reduced')
        Q, _ = np.linalg.qr(A   @ Z, mode='reduced')
    B            = Q.T @ A
    U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
    Q            = Q @ U_B[:, :k]
    return Q, s_B[:k]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def show_field(ax, fig, field, title, cmap='RdBu_r', vmin=None, vmax=None,
               sensors_rc=None, cbar_label='', xlabel='', ylabel='', fontsize=8):
    """
    Display a 2D field with labeled axes, colorbar, and optional sensor overlay.

    Parameters
    ----------
    sensors_rc : tuple (row_array, col_array) — pixel coordinates of sensors
    cbar_label : units string appended to the colorbar (e.g. 'm/s')
    xlabel/ylabel : axis label strings
    """
    im = ax.imshow(field, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')

    # Sensor locations shown as black X marks
    if sensors_rc is not None:
        ax.scatter(sensors_rc[1], sensors_rc[0],
                   c='k', s=16, marker='x', linewidths=0.8, zorder=5,
                   label=f'{len(sensors_rc[0])} sensors')
        ax.legend(loc='lower right', fontsize=6, framealpha=0.6,
                  markerscale=1.0, handletextpad=0.3)

    ax.set_title(title, fontsize=fontsize, pad=4)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.tick_params(labelsize=6)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load all 100 vertical levels ──────────────────────────────────────────
    print("Loading ISABEL data...")
    t0       = time.perf_counter()
    data_all = load_all_levels(DATA_PATH, DOWNSAMPLE)   # (100, ny, nz)
    ny, nz   = data_all.shape[1], data_all.shape[2]
    n        = ny * nz

    # Skip first 10 near-surface levels: their near-zero values produce
    # artificially large relative errors (small denominator).
    SKIP_LEVELS = 10
    data         = data_all[SKIP_LEVELS:]               # (90, ny, nz)
    n_levels     = data.shape[0]
    print(f"  Loaded 100 levels; using levels {SKIP_LEVELS}–99 "
          f"({n_levels} levels, {ny}x{nz} = {n} pts/level,  "
          f"{time.perf_counter()-t0:.2f}s)")
    print(f"  U-wind range (levels 10–99): {data.min():.2f} to {data.max():.2f} m/s")

    # 1% of total grid points is the sensor cap (overridden by MAX_SENSORS if set), 66:1 is comparable to "error controlled lossy" paper
    _max_sensors = MAX_SENSORS if MAX_SENSORS is not None else max(1, n // 66)
    print(f"  Sensor cap: {_max_sensors} ({_max_sensors/n*100:.1f}% of {n} grid points)")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: Pilot SVD on all levels to estimate k -> set training set size
    #
    # Note: test levels participate in this SVD only to determine HOW MANY
    # training levels to use.  They do NOT influence the basis or sensors.
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[Pilot] SVD of all 100 levels to estimate k...")
    if NORMALIZE:
        pilot_mean = data.mean(axis=0)
        X_pilot    = (data - pilot_mean).reshape(n_levels, n).T
    else:
        X_pilot    = data.reshape(n_levels, n).T

    _, s_pilot, _ = np.linalg.svd(X_pilot, full_matrices=False)
    k_pilot, _    = adaptive_k(s_pilot, ERROR_TOL, _max_sensors)

    # Training set size: N_TRAIN_FACTOR x k, capped to leave >= 10 test levels
    n_train = N_TRAIN_FACTOR * k_pilot
    n_train = min(n_train, n_levels - 10)   # keep at least 10 test levels
    n_train = max(n_train, k_pilot + 2)     # need more snapshots than modes
    n_test  = n_levels - n_train
    print(f"  k_pilot={k_pilot}  =>  n_train={n_train}  n_test={n_test}")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: Random train / test split
    # ═════════════════════════════════════════════════════════════════════════
    rng_split  = np.random.default_rng(SPLIT_SEED)
    train_idx  = np.sort(rng_split.choice(n_levels, size=n_train, replace=False))
    test_mask  = np.ones(n_levels, dtype=bool)
    test_mask[train_idx] = False
    test_idx   = np.where(test_mask)[0]

    train_data = data[train_idx]    # (n_train, ny, nz)
    test_data  = data[test_idx]     # (n_test,  ny, nz)

    print(f"\n  Train levels ({n_train}): {train_idx.tolist()}")
    print(f"  Test  levels ({n_test}):  {test_idx.tolist()}")

    # ── Normalize by training mean only ───────────────────────────────────────
    if NORMALIZE:
        train_mean = train_data.mean(axis=0)   # (ny, nz)
    else:
        train_mean = np.zeros((ny, nz))

    train_c = train_data - train_mean          # (n_train, ny, nz) centred
    test_c  = test_data  - train_mean          # (n_test,  ny, nz) centred

    # Snapshot matrix X: each column is one centred training level flattened
    X = train_c.reshape(n_train, n).T          # (n, n_train)

    # ═════════════════════════════════════════════════════════════════════════
    # RUN SELECTED METHODS
    # results[method_name] = {
    #     'basis'      : (n, k) array  — POD or randomized basis
    #     'sensors'    : (k,) int array
    #     'k'          : int
    #     'deim_const' : float
    #     'errors'     : (n_test,) relative L2 errors across all test levels
    #     'rmse'       : (n_test,) per-level RMSE in original units (m/s)
    #     'maxae'      : (n_test,) per-level max |error| in original units (m/s)
    #     'recons'     : (n_test, n) centred reconstructions
    # }
    # ═════════════════════════════════════════════════════════════════════════
    results = {}

    # ── Q-DEIM ────────────────────────────────────────────────────────────────
    if 'Q-DEIM' in METHODS:
        print(f"\n{'─'*62}\n  Q-DEIM (offline)\n{'─'*62}")
        t0 = time.perf_counter()

        U, s_vals, _ = np.linalg.svd(X, full_matrices=False)
        k_q, perr_q  = adaptive_k(s_vals, ERROR_TOL, _max_sensors)
        energy_q     = (s_vals[:k_q]**2).sum() / (s_vals**2).sum()
        U_k          = U[:, :k_q]
        sensors_q, deim_const_q = qdeim_place(U_k)
        print(f"  k={k_q}  proj_err={perr_q[k_q-1]*100:.3f}%  "
              f"energy={energy_q*100:.2f}%  DEIM_const={deim_const_q:.4f}  "
              f"offline={time.perf_counter()-t0:.3f}s")

        errs_q  = np.zeros(n_test)
        rmse_q  = np.zeros(n_test)
        maxae_q = np.zeros(n_test)
        recons_q = np.zeros((n_test, n))
        for i, lvl_c in enumerate(test_c):
            y_flat        = lvl_c.ravel()
            y_hat_c       = qdeim_reconstruct(U_k, sensors_q, y_flat[sensors_q])
            recons_q[i]   = y_hat_c
            diff          = (y_hat_c.reshape(ny, nz) + train_mean) - test_data[i]
            errs_q[i]     = np.linalg.norm(diff) / np.linalg.norm(test_data[i])
            rmse_q[i]     = float(np.sqrt(np.mean(diff ** 2)))
            maxae_q[i]    = float(np.max(np.abs(diff)))
        print(f"  Online x{n_test} levels done.")

        results['Q-DEIM'] = dict(basis=U_k, sensors=sensors_q, k=k_q,
                                  deim_const=deim_const_q,
                                  errors=errs_q, rmse=rmse_q, maxae=maxae_q,
                                  recons=recons_q)

    # ── R-DEIM ────────────────────────────────────────────────────────────────
    if 'R-DEIM' in METHODS:
        print(f"\n{'─'*62}\n  R-DEIM  "
              f"(p={RDEIM_OVERSAMPLE}, q={RDEIM_N_ITER}) (offline)\n{'─'*62}")
        t0  = time.perf_counter()
        rng = np.random.default_rng(RDEIM_SEED)

        k_cap = min(_max_sensors, n_train - 1)
        Q_rand, s_approx = randomized_range_finder(
            X, k=k_cap, p=RDEIM_OVERSAMPLE, q=RDEIM_N_ITER, rng=rng)
        k_r, _  = adaptive_k(s_approx, ERROR_TOL, k_cap)
        Q_k     = Q_rand[:, :k_r]
        sensors_r, deim_const_r = qdeim_place(Q_k)
        print(f"  k={k_r}  DEIM_const={deim_const_r:.4f}  "
              f"offline={time.perf_counter()-t0:.3f}s")

        errs_r  = np.zeros(n_test)
        rmse_r  = np.zeros(n_test)
        maxae_r = np.zeros(n_test)
        recons_r = np.zeros((n_test, n))
        for i, lvl_c in enumerate(test_c):
            y_flat        = lvl_c.ravel()
            y_hat_c       = qdeim_reconstruct(Q_k, sensors_r, y_flat[sensors_r])
            recons_r[i]   = y_hat_c
            diff          = (y_hat_c.reshape(ny, nz) + train_mean) - test_data[i]
            errs_r[i]     = np.linalg.norm(diff) / np.linalg.norm(test_data[i])
            rmse_r[i]     = float(np.sqrt(np.mean(diff ** 2)))
            maxae_r[i]    = float(np.max(np.abs(diff)))
        print(f"  Online x{n_test} levels done.")

        results['R-DEIM'] = dict(basis=Q_k, sensors=sensors_r, k=k_r,
                                  deim_const=deim_const_r,
                                  errors=errs_r, rmse=rmse_r, maxae=maxae_r,
                                  recons=recons_r)

    if not results:
        raise ValueError("METHODS is empty — add at least one of 'Q-DEIM', 'R-DEIM'.")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════════

    # ── Compression ratio ─────────────────────────────────────────────────────
    # Original: n_levels levels × n grid points, float32 (4 bytes each).
    # Compressed representation (shared basis + per-level payloads):
    #   basis U_k   : n × k   float32   (spatial modes, shared across all levels)
    #   train_mean  : n       float32   (needed for denormalisation)
    #   sensor_idx  : k       int32     (positions, shared across all levels)
    #   observations: n_levels × k float32  (per-level sensor values — the payload)
    #
    # CR = (n_levels × n) / (n×k + n + k + n_levels×k)
    # This counts every level (train + test) as part of what gets compressed,
    # and the basis as the shared overhead amortised across all n_levels.
    _B4 = 4   # float32 / int32 both 4 bytes

    print(f"\n{'═'*74}")
    print(f"  Results over {n_test} test levels  "
          f"(n_train={n_train}, ERROR_TOL={ERROR_TOL}, DOWNSAMPLE={DOWNSAMPLE})")

    # Relative error table
    print(f"\n  Relative L2 error (unitless):")
    print(f"  {'Method':<10} {'k':>4} {'DEIM_const':>11} "
          f"{'Min':>8} {'Median':>8} {'Max':>8} {'Mean':>8}")
    print(f"  {'─'*63}")
    for mname, mres in results.items():
        e = mres['errors']
        print(f"  {mname:<10} {mres['k']:>4} {mres['deim_const']:>11.4f} "
              f"{e.min():>7.4f} {np.median(e):>8.4f} "
              f"{e.max():>8.4f} {e.mean():>8.4f}")

    # Absolute error table (per-level RMSE and max |error|, all in m/s)
    print(f"\n  Pointwise absolute error (m/s, field-by-field over test levels):")
    print(f"  {'Method':<10} {'k':>4}  "
          f"{'RMSE min':>10} {'RMSE med':>10} {'RMSE max':>10}  "
          f"{'Max|e| med':>11} {'Max|e| max':>11}")
    print(f"  {'─'*73}")
    for mname, mres in results.items():
        rm = mres['rmse'];  mx = mres['maxae']
        print(f"  {mname:<10} {mres['k']:>4}  "
              f"{rm.min():>9.4f}  {np.median(rm):>9.4f}  {rm.max():>9.4f}  "
              f"{np.median(mx):>10.4f}  {mx.max():>10.4f}")

    # Compression ratio (same k for all methods; use first method's k)
    k_cr = next(iter(results.values()))['k']
    bytes_orig = n_levels * n * _B4
    bytes_comp = (n * k_cr * _B4          # basis U_k
                  + n * _B4               # train_mean
                  + k_cr * _B4            # sensor indices (int32)
                  + n_levels * k_cr * _B4)# per-level observations
    cr = bytes_orig / bytes_comp
    print(f"\n  Compression ratio  (basis shared across all {n_levels} levels):")
    print(f"    Original : {bytes_orig/1e6:.2f} MB  ({n_levels} levels × {n} pts × float32)")
    print(f"    Compressed: {bytes_comp/1e6:.2f} MB  "
          f"(basis {n}×{k_cr} + mean {n} + idx {k_cr} + {n_levels}×{k_cr} obs)")
    print(f"    CR = {cr:.2f}x")
    print(f"{'═'*74}")

    # ═════════════════════════════════════════════════════════════════════════
    # SHOWCASE SELECTION
    # Three rows: best / median / worst — ranked by the first listed method.
    # ═════════════════════════════════════════════════════════════════════════
    ref_name  = METHODS[0]
    ref_errs  = results[ref_name]['errors']
    idx_min   = int(np.argmin(ref_errs))
    idx_max   = int(np.argmax(ref_errs))
    idx_med   = int(np.argmin(np.abs(ref_errs - np.median(ref_errs))))
    showcase  = [idx_min, idx_med, idx_max]
    show_labels  = ['Lowest error', 'Median error', 'Highest error']
    row_banners  = ['BEST CASE\n(lowest error)', 'TYPICAL CASE\n(median error)',
                    'WORST CASE\n(highest error)']
    row_colors   = ['#1a7a1a', '#b07800', '#b01a1a']

    print(f"\n  Showcase levels (ranked by {ref_name}):")
    for label, si in zip(show_labels, showcase):
        errs_str = '  '.join(
            f"{mname} err={mres['errors'][si]:.4f}"
            for mname, mres in results.items()
        )
        print(f"    {label:<16}: level {test_idx[si]:>3}  {errs_str}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE — dynamic columns based on active methods
    #
    # Layout: 3 rows × (1 + 2 × n_methods) columns
    #   Col 0           : True field (ground truth)
    #   Col 1, 2        : method[0] reconstruction + absolute error
    #   Col 3, 4        : method[1] reconstruction + absolute error  (if present)
    #   ... etc.
    #
    # Rows:
    #   Row 0 — BEST CASE   : test level with lowest error (first method)
    #   Row 1 — TYPICAL CASE: test level closest to median error
    #   Row 2 — WORST CASE  : test level with highest error
    #
    # Error colorscale is per-row (shared across methods within the row) so
    # you can compare methods side-by-side without one dominating. Do NOT
    # compare error colours across rows — use the colorbar values.
    # ═════════════════════════════════════════════════════════════════════════
    n_methods = len(results)
    n_cols    = 1 + 2 * n_methods
    fig_w     = max(14, 5 * n_cols)          # ~5 in per column
    fig1, axes = plt.subplots(3, n_cols, figsize=(fig_w, 14),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.38})
    if n_cols == 1:                           # make axes always 2-D
        axes = axes[:, np.newaxis]

    method_desc = {
        'Q-DEIM': 'exact SVD basis',
        'R-DEIM': f'randomized basis (p={RDEIM_OVERSAMPLE}, q={RDEIM_N_ITER})',
    }

    fig1.suptitle(
        'ISABEL Hurricane — U-wind (east-west) Reconstruction via DEIM\n'
        f'Grid: {ny}×{nz} points (downsampled {DOWNSAMPLE}× from 500×500)  |  '
        f'Training: {n_train} randomly chosen vertical levels  |  '
        f'Test: {n_test} held-out levels  |  ERROR_TOL = {ERROR_TOL}  |  '
        f'Active methods: {", ".join(results.keys())}',
        fontsize=10, y=0.99
    )

    # Column headers
    col_headers = ['TRUE FIELD\n(ground truth, held out)']
    for mname, mres in results.items():
        col_headers.append(
            f'{mname} RECONSTRUCTION\n'
            f'({mres["k"]} sensors, {method_desc.get(mname, "")})'
        )
        col_headers.append(f'{mname} ABSOLUTE ERROR\n|reconstruction − truth|')
    for col, hdr in enumerate(col_headers):
        axes[0, col].set_title(hdr, fontsize=8, fontweight='bold', pad=8)

    ax_kw = dict(xlabel='Longitude (grid index, W→E)',
                 ylabel='Latitude (grid index, S→N)')

    # Global error colorscale: shared across all rows and methods so that
    # colour intensity is directly comparable — the worst-case row will look
    # saturated while the best-case row will show only faint colours, which
    # is the correct visual impression of how much error each case has.
    global_emax = max(
        np.abs(mres['recons'][si].reshape(ny,nz) + train_mean
               - test_data[si]).max()
        for mres in results.values()
        for si in showcase
    )
    row_emax_list = [global_emax] * 3   # same scale for every row

    for row, (si, label, banner, bcolor, emax) in enumerate(
            zip(showcase, show_labels, row_banners, row_colors, row_emax_list)):

        lvl_idx = test_idx[si]
        true_2d = test_data[si]
        vmin, vmax = true_2d.min(), true_2d.max()

        # Row banner
        axes[row, 0].annotate(
            banner,
            xy=(-0.38, 0.5), xycoords='axes fraction',
            fontsize=8, fontweight='bold', color=bcolor,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=bcolor, linewidth=1.5)
        )

        # Per-row subtitle (level + all method errors)
        err_parts = '   '.join(
            f'{mname}: rel={mres["errors"][si]:.3f}  RMSE={mres["rmse"][si]:.4f} m/s'
            for mname, mres in results.items()
        )
        mid_col = n_cols // 2
        fig1.text(0.5,
                  axes[row, mid_col].get_position().y1 + 0.005,
                  f'Vertical level {lvl_idx} / 99   {err_parts}',
                  ha='center', va='bottom', fontsize=7.5,
                  style='italic', color='#333333')

        # Col 0: true field
        show_field(axes[row, 0], fig1, true_2d, title='',
                   vmin=vmin, vmax=vmax, cbar_label='U-wind (m/s)', **ax_kw)

        # Cols 1,2 / 3,4 / ... : one pair per method
        for m_idx, (mname, mres) in enumerate(results.items()):
            recon_2d = mres['recons'][si].reshape(ny, nz) + train_mean
            error_2d = np.abs(recon_2d - true_2d)
            s_rc     = (mres['sensors'] // nz, mres['sensors'] % nz)
            c_recon  = 1 + m_idx * 2
            c_err    = 2 + m_idx * 2

            show_field(axes[row, c_recon], fig1, recon_2d, title='',
                       vmin=vmin, vmax=vmax, sensors_rc=s_rc,
                       cbar_label='U-wind (m/s)', **ax_kw)
            show_field(axes[row, c_err], fig1, error_2d, title='',
                       cmap='hot_r', vmin=0, vmax=emax,
                       cbar_label='|Error| (m/s)', **ax_kw)

    fig1.text(
        0.5, 0.005,
        f'Error colorscale is shared across all rows (0 – {global_emax:.2f} m/s).  '
        'Faint = small absolute error; bright = large error.  '
        'Colour intensity is directly comparable across rows.',
        ha='center', fontsize=7.5, style='italic', color='#555555'
    )

    fig1.savefig('/Users/jchen228/Desktop/Argonne/deim_showcase.png',
                 dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: deim_showcase.png")

