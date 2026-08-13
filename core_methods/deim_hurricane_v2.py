"""
deim_hurricane_v2.py

Reconstructs ISABEL hurricane U-wind fields using Q-DEIM — Discrete
Empirical Interpolation with QR-based sensor placement. Mirrors the
Q-DEIM block in script_sst.m (gpoed-code-python).

This is a trimmed-down version of deim_hurricane.py: the original file
also implemented R-DEIM (randomized DEIM) and a dynamic multi-method
figure layout, but the active configuration only ever ran Q-DEIM
(METHODS = ['Q-DEIM']). That dead code path has been removed here —
this version is Q-DEIM only, with a fixed 3-column figure layout
(true field / reconstruction / absolute error).

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
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH        = "/Users/jchen228/Desktop/Argonne/100x500x500/CLOUDf48.bin.f32"
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
    print(f"  Loaded 100 levels; using levels {SKIP_LEVELS}-99 "
          f"({n_levels} levels, {ny}x{nz} = {n} pts/level,  "
          f"{time.perf_counter()-t0:.2f}s)")
    print(f"  U-wind range (levels 10-99): {data.min():.2f} to {data.max():.2f} m/s")

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
    # Q-DEIM  (offline: basis + sensor placement, online: per-level reconstruction)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'-'*62}\n  Q-DEIM (offline)\n{'-'*62}")
    t0 = time.perf_counter()

    U, s_vals, _ = np.linalg.svd(X, full_matrices=False)
    k, perr      = adaptive_k(s_vals, ERROR_TOL, _max_sensors)
    energy       = (s_vals[:k]**2).sum() / (s_vals**2).sum()
    U_k          = U[:, :k]
    sensors, deim_const = qdeim_place(U_k)
    print(f"  k={k}  proj_err={perr[k-1]*100:.3f}%  "
          f"energy={energy*100:.2f}%  DEIM_const={deim_const:.4f}  "
          f"offline={time.perf_counter()-t0:.3f}s")

    errors = np.zeros(n_test)
    recons = np.zeros((n_test, n))
    for i, lvl_c in enumerate(test_c):
        y_flat     = lvl_c.ravel()
        y_hat_c    = qdeim_reconstruct(U_k, sensors, y_flat[sensors])
        recons[i]  = y_hat_c
        errors[i]  = (np.linalg.norm(y_hat_c.reshape(ny, nz) + train_mean
                                      - test_data[i]) /
                      np.linalg.norm(test_data[i]))
    print(f"  Online x{n_test} levels done.")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Results over {n_test} test levels  "
          f"(n_train={n_train}, ERROR_TOL={ERROR_TOL}, DOWNSAMPLE={DOWNSAMPLE})")
    print(f"  {'k':>4} {'DEIM_const':>11} "
          f"{'Min err':>9} {'Median err':>11} {'Max err':>9} {'Mean err':>9}")
    print(f"  {'-'*54}")
    print(f"  {k:>4} {deim_const:>11.4f} "
          f"{errors.min():>9.4f} {np.median(errors):>11.4f} "
          f"{errors.max():>9.4f} {errors.mean():>9.4f}")
    print(f"{'='*70}")

    # ═════════════════════════════════════════════════════════════════════════
    # SHOWCASE SELECTION
    # Three rows: best / median / worst — ranked by Q-DEIM error.
    # ═════════════════════════════════════════════════════════════════════════
    idx_min   = int(np.argmin(errors))
    idx_max   = int(np.argmax(errors))
    idx_med   = int(np.argmin(np.abs(errors - np.median(errors))))
    showcase  = [idx_min, idx_med, idx_max]
    show_labels  = ['Lowest error', 'Median error', 'Highest error']
    row_banners  = ['BEST CASE\n(lowest error)', 'TYPICAL CASE\n(median error)',
                    'WORST CASE\n(highest error)']
    row_colors   = ['#1a7a1a', '#b07800', '#b01a1a']

    print(f"\n  Showcase levels:")
    for label, si in zip(show_labels, showcase):
        print(f"    {label:<16}: level {test_idx[si]:>3}  err={errors[si]:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE — fixed 3 columns: true field / reconstruction / absolute error
    # ═════════════════════════════════════════════════════════════════════════
    n_cols = 3
    fig1, axes = plt.subplots(3, n_cols, figsize=(15, 14),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.38})

    fig1.suptitle(
        'ISABEL Hurricane - U-wind (east-west) Reconstruction via Q-DEIM\n'
        f'Grid: {ny}x{nz} points (downsampled {DOWNSAMPLE}x from 500x500)  |  '
        f'Training: {n_train} randomly chosen vertical levels  |  '
        f'Test: {n_test} held-out levels  |  ERROR_TOL = {ERROR_TOL}',
        fontsize=10, y=0.99
    )

    col_headers = [
        'TRUE FIELD\n(ground truth, held out)',
        f'Q-DEIM RECONSTRUCTION\n({k} sensors, exact SVD basis)',
        'ABSOLUTE ERROR\n|reconstruction - truth|',
    ]
    for col, hdr in enumerate(col_headers):
        axes[0, col].set_title(hdr, fontsize=8, fontweight='bold', pad=8)

    ax_kw = dict(xlabel='Longitude (grid index, W->E)',
                 ylabel='Latitude (grid index, S->N)')

    # Global error colorscale: shared across all rows so that colour intensity
    # is directly comparable — the worst-case row will look saturated while
    # the best-case row will show only faint colours.
    global_emax = max(
        np.abs(recons[si].reshape(ny, nz) + train_mean - test_data[si]).max()
        for si in showcase
    )

    for row, (si, banner, bcolor) in enumerate(zip(showcase, row_banners, row_colors)):

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

        # Per-row subtitle (level + error)
        fig1.text(0.5,
                  axes[row, 1].get_position().y1 + 0.005,
                  f'Vertical level {lvl_idx} / 99   rel. err = {errors[si]:.3f}',
                  ha='center', va='bottom', fontsize=7.5,
                  style='italic', color='#333333')

        recon_2d = recons[si].reshape(ny, nz) + train_mean
        error_2d = np.abs(recon_2d - true_2d)
        s_rc     = (sensors // nz, sensors % nz)

        show_field(axes[row, 0], fig1, true_2d, title='',
                   vmin=vmin, vmax=vmax, cbar_label='U-wind (m/s)', **ax_kw)
        show_field(axes[row, 1], fig1, recon_2d, title='',
                   vmin=vmin, vmax=vmax, sensors_rc=s_rc,
                   cbar_label='U-wind (m/s)', **ax_kw)
        show_field(axes[row, 2], fig1, error_2d, title='',
                   cmap='hot_r', vmin=0, vmax=global_emax,
                   cbar_label='|Error| (m/s)', **ax_kw)

    fig1.text(
        0.5, 0.005,
        f'Error colorscale is shared across all rows (0 - {global_emax:.2f} m/s).  '
        'Faint = small absolute error; bright = large error.  '
        'Colour intensity is directly comparable across rows.',
        ha='center', fontsize=7.5, style='italic', color='#555555'
    )

    fig1.savefig('/Users/jchen228/Desktop/Argonne/deim_hurricane_v2_showcase.png',
                 dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: deim_hurricane_v2_showcase.png")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — SZ-style quantization histogram (Q-DEIM residuals)
    # ═════════════════════════════════════════════════════════════════════════
    # bin_idx = round(residual / (2 × sz_eb))
    # SZ default: 1024 bins, outlier threshold ±512.
    # Bins outside ±512 are stored losslessly ("uncompressible").
    # The shape of this histogram reveals how well the DEIM prediction
    # reduces the residual entropy — a spike at bin 0 is ideal.
    # ─────────────────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches

    SZ_N_BINS  = 1024
    _half      = SZ_N_BINS // 2   # 512

    field_range = float(data.max() - data.min())
    sz_eb       = ERROR_TOL * field_range   # absolute error bound (m/s)

    # All test residuals in original units across all n_test × n points
    all_resids = (recons.reshape(n_test, ny, nz) + train_mean - test_data).ravel()
    bins_idx   = np.round(all_resids / (2.0 * sz_eb)).astype(np.int64)

    n_total  = bins_idx.size
    n_out    = int(np.sum(np.abs(bins_idx) > _half))
    n_zero   = int(np.sum(bins_idx == 0))
    pct_out  = 100.0 * n_out / n_total
    n_unique = int(np.unique(bins_idx[np.abs(bins_idx) <= _half]).size)

    disp   = max(10, min(200, int(np.percentile(np.abs(bins_idx), 99)) + 5))
    counts, edges = np.histogram(bins_idx.clip(-disp, disp), bins=range(-disp, disp + 2))
    centers       = 0.5 * (edges[:-1] + edges[1:])
    bar_colors    = ['#d62728' if abs(c) >= disp else '#2ca02c' for c in centers]

    fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax2.bar(centers, counts, width=0.9, color=bar_colors, zorder=2)
    for sign in (-1, 1):
        ax2.axvline(sign * min(_half, disp), color='black', ls='--', lw=1.4, zorder=3)
    ax2.set_xlabel('Quantization bin index  [bin_idx = round(residual / 2·eb)]', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, zorder=1)
    ax2.legend(handles=[
        mpatches.Patch(color='#2ca02c', label='In-range  (compressible)'),
        mpatches.Patch(color='#d62728', label=f'Outlier  (|bin| ≥ {disp}, clipped here)'),
        plt.Line2D([0], [0], color='black', ls='--', lw=1.4,
                   label=f'SZ boundary ±{_half}'),
    ], fontsize=9)
    ax2.set_title(
        f'Q-DEIM — SZ-style quantization histogram   '
        f'({n_test} test levels × {n} pts = {n_total:,} residuals)\n'
        f'eb = ERROR_TOL × field_range = {ERROR_TOL} × {field_range:.2f} = {sz_eb:.4f} m/s   '
        f'SZ codebook: {SZ_N_BINS} bins (outlier threshold ±{_half})\n'
        f'Outliers: {n_out:,} / {n_total:,} ({pct_out:.2f}%)   '
        f'Unique bins in range: {n_unique}   '
        f'Zero-bin: {n_zero:,} ({100*n_zero/n_total:.1f}%)',
        fontsize=9
    )
    out2 = '/Users/jchen228/Desktop/Argonne/deim_v2_quantization_hist.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out2}")
