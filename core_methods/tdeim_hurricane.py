"""
tdeim_hurricane.py

Tensor DEIM (T-DEIM): reconstructs the full 3D ISABEL U-wind field from k
sensors scattered throughout the 3D volume — sensors are at arbitrary
(level, y, z) positions, not restricted to a single 2D plane.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLAIN-LANGUAGE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Think of the hurricane wind field as a stack of 100 horizontal maps —
like 100 floors of a building, each showing east-west wind speed at that
altitude.

  deim_hurricane.py (2D, slice-by-slice)
    Picks k fixed (latitude, longitude) positions.
    To reconstruct any floor: measure those k stations ON THAT FLOOR,
    solve for k coefficients, done.  Each floor solved independently.

  tdeim_hurricane.py (T-DEIM, whole building)
    Picks k stations anywhere in the building — each at its own floor
    AND its own (lat, lon).  One single solve using all k readings
    reconstructs ALL 100 floors at once.
    The algorithm spreads sensors to floors where the field varies most,
    because those floors carry the most information about the entire
    3D structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT EACH FIGURE SHOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Figure 1 — Comparison at a single reference slice (SHOW_SLICE)
  Row 1  T-DEIM:  true field | 3D reconstruction | absolute error
  Row 2  2D DEIM: true field | 2D reconstruction | absolute error
  Both methods use exactly k sensors.  Shared error colorscale.
  T-DEIM sensor dots shown only for those that land on the displayed floor.

Figure 2 — Where did T-DEIM put its sensors?
  Left:  histogram of sensor floors (vertical axis = how many sensors
         are on each floor).  Peaks = floors the algorithm judged most
         informative for representing the full 3D structure.
  Right: scatter plot in the 2D (lat, lon) plane, each dot is one sensor
         coloured by the floor it sits on.  Shows spatial spread and
         whether high-altitude sensors (bright) cluster differently from
         low-altitude ones (dark).

Figure 3 — Per-level reconstruction quality (T-DEIM vs 2D DEIM)
  Each row is a floor (level), showing true | T-DEIM | 2D DEIM side by
  side for the best, median, and worst T-DEIM error across all 100 levels.
  Reveals which floors are easy or hard to reconstruct.

Figure 4 — Full 3D volumetric scatter plot of the T-DEIM reconstruction.
  Every point in the volume is drawn with colour ∝ wind speed (RdBu_r) and
  opacity ∝ |wind speed| so calm regions fade to transparent.  The eye
  naturally picks out the high-speed structure floating inside the volume.
  VIZ_DOWNSAMPLE controls spatial density (increase to speed up rendering).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3D BASIS (MODE-1 UNFOLDING + RANK-1 MODES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Let F = normalised field, shape (n_L, n),  n = n_Y * n_Z.
Mode-1 unfold: A = F,  shape (n_L, n).
Thin SVD:      A = U_L diag(s) Vt   [U_L:(n_L,r),  Vt:(r,n)]

Rank-1 3D modes (orthonormal):
  phi_j = flatten( outer(U_L[:,j], Vt[j,:]) ),  shape (n_L*n,)

Proof of orthonormality:
  <phi_i, phi_j> = (U_L[:,i]·U_L[:,j]) * (Vt[i,:]·Vt[j,:]) = delta_ij ✓

Basis matrix: Phi = [phi_1 | ... | phi_k],  shape (n_L*n, k).

Q-DEIM placement: QR column pivoting on Phi.T → k pivot indices in [0, n_L*n).
  index → (level_idx = index // n,  y = (index%n)//nz,  z = (index%n)%nz)

Reconstruction from k sensor observations y_obs:
  Solve  Phi[sensors, :] @ c = y_obs  →  F_hat = Phi @ c  (k×k system)

References
----------
Chaturantabut & Sorensen (2010). SIAM J. Sci. Comput. 32(5), 2737-2764.
Drmac & Gugercin (2016). SIAM J. Sci. Comput. 38(2), A631-A648.
Saibaba (2020). SIAM J. Sci. Comput. 42(3), A1582-A1608.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "/Users/jchen228/Desktop/Argonne/100x500x500/CLOUDf48.bin.f32"
DOWNSAMPLE  = 3          # spatial downsample; 7 → ~72×72 = 5184 pts/level
SKIP_LEVELS          = 0   # 0 = all 100 levels included in basis & errors
SHOWCASE_MIN_LEVEL   = 10  # exclude levels 0–9 from best/median/worst ranking
                           # (near-zero surface values → artificially huge rel. error)
SHOW_SLICE  = 50         # global level index (0–99) for Figure 1
ERROR_TOL   = 1e-2       # Single error tolerance used by all GUARANTEE_MODE options.
                        # Meaning depends on mode — see GUARANTEE_MODE below.
MAX_SENSORS = None       # None → auto: 1% of 3D volume
VIZ_DOWNSAMPLE = 2      # extra spatial downsample for Figure 4 only (1=full res,
                        # 2=half, 3=third …).  Increase if plotting is slow.
LEVEL_STEP     = 2      # render every Nth level in Figure 4 (1=all, 2=every other…)
OVERSAMPLE     = 10     # extra sensors beyond k: p = k + OVERSAMPLE.
                        # 0 → exact square solve (classic DEIM).
                        # >0 → overdetermined system solved by least squares (more robust).
ABS_TOL = 0.001           # Absolute error target in field units (m/s for U-wind).
                        # Used by GUARANTEE_MODE 'abs' only.
GUARANTEE_MODE = 'apriori'
# GUARANTEE_MODE controls how k is chosen; ERROR_TOL is the single tolerance:
#
#   'trunc'   — Smallest k so ε_POD ≤ ERROR_TOL.  Fast.  Does NOT account
#               for sensor interpolation error (Lebesgue constant Λ), so
#               empirical err_full can exceed ERROR_TOL if Λ is large.
#
#   'apriori' — Drmač & Gugercin (2016) a priori bound: find initial k where
#               ε_POD ≤ ERROR_TOL, then raise k until (1+Λ)×ε_POD ≤ ERROR_TOL.
#               Mathematically rigorous but conservative — can over-raise k.
#               Capped at k_orig + n_L//3 to prevent runaway to full rank.
#
#   'abs'     — Absolute error: smallest k so that the estimated RMSE in
#               original field units is ≤ ABS_TOL.  Matches the error-control
#               philosophy of SZ-1.4 (Tao et al. 2017), which bounds each
#               value within a fixed tolerance.  DEIM is an L2 method, so
#               RMSE is guaranteed; pointwise L∞ error is also reported but
#               not formally bounded.  k is chosen by translating ABS_TOL into
#               a relative ε_POD threshold on the normalised basis matrix A.
#               ERROR_TOL is not used in this mode.
#
#   None      — No guarantee; k set by ε_POD ≤ ERROR_TOL (same as 'trunc'
#               but without the guarantee label in output).


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_levels(path, downsample=1):
    """Load all 100 levels from an ISABEL .bin.f32 file → (100, ny, nz)."""
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    return raw.reshape((100, 500, 500))[:, ::downsample, ::downsample]


def adaptive_k(s, tol, max_k=None):
    """Smallest k s.t. relative POD projection error ≤ tol."""
    s2       = s ** 2
    cumul    = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask     = proj_err <= tol
    k        = int(np.argmax(mask)) + 1 if mask.any() else len(s)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


def qdeim_place(Phi, n_sensors=None):
    """Q-DEIM: QR column pivoting on Phi.T → sensor indices, DEIM lebesgue constant.

    Parameters
    ----------
    Phi      : (N, k) orthonormal basis matrix
    n_sensors: number of sensors to select (default k = Phi.shape[1]).
               If n_sensors > k the system is overdetermined; use lstsq to solve.
               If n_sensors == k the system is square; use solve (classic DEIM).

    Returns
    -------
    sensors    : (n_sensors,) int array of row indices
    deim_const : ||pinv(Phi[sensors, :])||_2  (Lebesgue constant; lower = better)
    """
    k = Phi.shape[1]
    if n_sensors is None:
        n_sensors = k
    _, _, p    = scipy_qr(Phi.T, pivoting=True)
    sensors    = p[:n_sensors]
    deim_const = np.linalg.norm(np.linalg.pinv(Phi[sensors, :]))
    return sensors, deim_const


def show_field(ax, fig, field, title, cmap='RdBu_r', vmin=None, vmax=None,
               sx=None, sy=None, sensor_label='', cbar_label='U-wind (m/s)',
               fontsize=11):
    """Imshow panel with optional sensor scatter overlay."""
    im = ax.imshow(field, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    if sx is not None and len(sx) > 0:
        ax.scatter(sy, sx, c='k', s=22, marker='x', linewidths=0.9,
                   zorder=5, label=sensor_label)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.6)
    ax.set_title(title, fontsize=fontsize, pad=4)
    ax.set_xlabel('Longitude index', fontsize=10)
    ax.set_ylabel('Latitude index',  fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading ISABEL U-wind data (all levels)...")
    data_all = load_all_levels(DATA_PATH, DOWNSAMPLE)   # (100, ny, nz)
    ny, nz   = data_all.shape[1], data_all.shape[2]
    n        = ny * nz

    data = data_all[SKIP_LEVELS:]   # (n_L, ny, nz)
    n_L  = data.shape[0]

    print(f"  Grid: {ny}×{nz} = {n} pts/level  |  "
          f"{n_L} levels (skipped first {SKIP_LEVELS})  |  "
          f"3D volume: {n_L}×{n} = {n_L*n} pts")
    print(f"  U-wind range: {data.min():.2f} to {data.max():.2f} m/s")

    _max_sensors = MAX_SENSORS if MAX_SENSORS is not None else max(1, (n_L * n) // 100)
    print(f"  3D sensor cap: {_max_sensors} "
          f"({_max_sensors / (n_L*n) * 100:.3f}% of 3D vol)")

    # ── Normalize: per-(y,z) mean and std across all valid levels ─────────────
    data_flat = data.reshape(n_L, n)           # (n_L, n)
    mean_yz   = data_flat.mean(axis=0)         # (n,)
    std_yz    = data_flat.std(axis=0)          # (n,)
    std_yz    = np.where(std_yz < 1e-10, 1.0, std_yz)
    A         = (data_flat - mean_yz) / std_yz # (n_L, n) normalised

    # ── SVD of mode-1 unfolding — shared between T-DEIM and 2D Q-DEIM ─────────
    print(f"\n[SVD] mode-1 unfolding ({n_L} × {n})...")
    t0         = time.perf_counter()
    U_L, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  Done ({time.perf_counter()-t0:.2f}s)  |  rank={len(s)}")

    # Choose k according to GUARANTEE_MODE
    if GUARANTEE_MODE == 'trunc':
        # Relative truncation: smallest k where ε_POD ≤ ERROR_TOL
        k, proj_err = adaptive_k(s, ERROR_TOL, _max_sensors)
        print(f"  [GUARANTEE_MODE='trunc']  k={k}  "
              f"ε_POD={proj_err[k-1]*100:.3f}% ≤ {ERROR_TOL*100:.1f}%")

    elif GUARANTEE_MODE == 'abs':
        # Absolute RMSE mode: translate ABS_TOL (m/s) into a relative ε_POD
        # threshold on the normalised basis A.
        #
        # Derivation:
        #   y_hat (original units) = y_hat_norm * std_yz + mean_yz
        #   RMSE_orig  ≈  ε_POD × ‖A‖_F × rms(std_yz) / √N
        #     where ‖A‖_F = np.linalg.norm(s)  and  N = n_L × n
        #   Setting RMSE_orig ≤ ABS_TOL gives:
        #     ε_POD ≤ ABS_TOL × √N / (‖A‖_F × rms(std_yz))
        A_frob  = float(np.linalg.norm(s))              # ‖A‖_F in normalised space
        std_rms = float(np.sqrt(np.mean(std_yz ** 2)))  # RMS of per-column stds
        tol_rel = ABS_TOL * np.sqrt(n_L * n) / (A_frob * std_rms)
        tol_rel = float(np.clip(tol_rel, 1e-14, 1.0))   # safety bounds
        k, proj_err = adaptive_k(s, tol_rel, _max_sensors)
        print(f"  [GUARANTEE_MODE='abs']  ABS_TOL={ABS_TOL} m/s  "
              f"→ ε_POD target={tol_rel*100:.4f}%  k={k}  "
              f"ε_POD achieved={proj_err[k-1]*100:.4f}%")

    else:
        # 'apriori' or None: k from ε_POD ≤ ERROR_TOL; Drmač loop handled below
        k, proj_err = adaptive_k(s, ERROR_TOL, _max_sensors)
        print(f"  [GUARANTEE_MODE='{GUARANTEE_MODE}']  "
              f"k={k}  ε_POD={proj_err[k-1]*100:.3f}% ≤ {ERROR_TOL*100:.1f}%")

    energy = (s[:k] ** 2).sum() / (s ** 2).sum()
    print(f"  energy captured={energy*100:.2f}%")

    # ═════════════════════════════════════════════════════════════════════════
    # T-DEIM — 3D sensor placement
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print(f"  T-DEIM  (k={k} sensors scattered in 3D volume)")
    print(f"{'─'*62}")

    # Build 3D basis Phi: (n_L*n, k), orthonormal columns
    # phi_j = flatten(outer(U_L[:,j], Vt[j,:])) — see docstring proof
    print(f"  Building Phi ({n_L*n} × {k})...")
    t0  = time.perf_counter()
    Phi = np.empty((n_L * n, k))
    for j in range(k):
        Phi[:, j] = np.outer(U_L[:, j], Vt[j, :]).ravel()
    print(f"  Done ({time.perf_counter()-t0:.2f}s)")

    p_3d = k + OVERSAMPLE
    print(f"  Q-DEIM sensor placement in 3D  (p={p_3d} = k+{OVERSAMPLE} sensors)...")
    t0 = time.perf_counter()
    sensors_3d, deim_const_3d = qdeim_place(Phi, p_3d)
    print(f"  Done ({time.perf_counter()-t0:.2f}s)  DEIM_const={deim_const_3d:.4f}")

    # ── A priori error guarantee (Drmač & Gugercin 2016) ─────────────────────
    # Only runs when GUARANTEE_MODE='apriori'.
    # Theoretical bound:  ‖f − P_DEIM f‖/‖f‖  ≤  (1 + deim_const) × proj_err
    # Term 1 (deim_const): DEIM Lebesgue constant — controlled by Q-DEIM sensor
    #                       placement; oversampling (OVERSAMPLE > 0) reduces it.
    # Term 2 (proj_err):   POD truncation error — decreases as k increases.
    # If the initial k from ERROR_TOL is not enough, grow k until the bound is met.
    if GUARANTEE_MODE == 'apriori':
        bound = (1.0 + deim_const_3d) * proj_err[k - 1]
        print(f"\n  ── Error guarantee check (target ≤ {ERROR_TOL*100:.1f}%) ──────────────")
        print(f"     (1 + {deim_const_3d:.3f}) × {proj_err[k-1]*100:.3f}%  =  {bound*100:.2f}%")
        if bound > ERROR_TOL:
            print(f"     Bound exceeds target — increasing k to satisfy guarantee ...")
            k_orig = k
            # k_max: cap at k_orig + n_L//3 extra modes so the loop cannot drive k
            # all the way to full rank when the Lebesgue constant is large.  The a
            # priori Drmač bound is conservative; the empirical error (checked after
            # reconstruction) is the ground truth.
            k_max  = min(len(s), k_orig + n_L // 3,
                         max(_max_sensors - OVERSAMPLE, k + 1))
            while bound > ERROR_TOL and k < k_max:
                k      += 1
                new_col = np.outer(U_L[:, k - 1], Vt[k - 1, :]).ravel()[:, None]
                Phi     = np.concatenate([Phi, new_col], axis=1)
                p_3d    = k + OVERSAMPLE
                sensors_3d, deim_const_3d = qdeim_place(Phi, p_3d)
                bound   = (1.0 + deim_const_3d) * proj_err[k - 1]
                # Stop early if proj_err is already at machine precision —
                # the bound can only decrease further with additional modes.
                if proj_err[k - 1] < 1e-13:
                    break
            if bound <= ERROR_TOL:
                energy = (s[:k] ** 2).sum() / (s ** 2).sum()
                print(f"     ✓ k: {k_orig} → {k}  |  "
                      f"bound={bound*100:.2f}%  DEIM_const={deim_const_3d:.4f}  "
                      f"energy={energy*100:.2f}%  p={p_3d} sensors")
            else:
                print(f"     ✗ k={k} (hit rank/cap limit) — "
                      f"bound={bound*100:.2f}% still > {ERROR_TOL*100:.1f}%.")
                print(f"       Consider lowering ERROR_TOL or increasing MAX_SENSORS.")
        else:
            print(f"     ✓ Satisfied at initial k={k}  (bound={bound*100:.2f}%)")

    # Decode 3D positions: (local level, y, z)
    lev_local, yz_flat = np.divmod(sensors_3d, n)
    y_sens             = yz_flat // nz
    z_sens             = yz_flat % nz
    lev_global         = lev_local + SKIP_LEVELS   # global level index (0-based)

    # Reconstruct full 3D field
    # ── Two separate linear algebra steps — do not confuse them: ─────────────
    # Step A  (above) — Q-DEIM sensor PLACEMENT: scipy_qr(Phi.T, pivoting=True)
    #   Pivoted QR on the transposed basis finds the p most linearly independent
    #   rows of Phi.  The column pivots = sensor indices.  This is QR.
    #
    # Step B  (here)  — RECONSTRUCTION SOLVE: np.linalg.lstsq or np.linalg.solve
    #   Given the p sensor observations y_obs, find coefficients c such that
    #   Phi[sensors, :] @ c ≈ y_obs, then reconstruct F_hat = Phi @ c.
    #
    #   • OVERSAMPLE > 0 → overdetermined system (p×k, p > k).
    #     np.linalg.lstsq uses LAPACK dgelsd (divide-and-conquer SVD), NOT QR.
    #     Alternative: scipy.linalg.lstsq(..., lapack_driver='gelsy') uses
    #     column-pivoted QR — faster for small overdetermined systems.
    #
    #   • OVERSAMPLE == 0 → square system (k×k).
    #     np.linalg.solve uses LAPACK dgesv (LU factorisation), NOT QR.
    # ─────────────────────────────────────────────────────────────────────────
    y_3d_flat    = A.ravel()
    y_obs_3d     = y_3d_flat[sensors_3d]
    if OVERSAMPLE > 0:
        c_3d, _, _, _ = np.linalg.lstsq(Phi[sensors_3d, :], y_obs_3d, rcond=None)
    else:
        c_3d = np.linalg.solve(Phi[sensors_3d, :], y_obs_3d)
    y_hat_3d_norm = (Phi @ c_3d).reshape(n_L, n)
    y_hat_3d      = y_hat_3d_norm * std_yz + mean_yz   # (n_L, n) original units

    # Per-level T-DEIM relative L2 errors
    # Returns np.nan for near-zero levels (‖true‖ < 1e-10) to avoid inf%
    # caused by surface levels where wind speed ≈ 0.
    def _rel_err(pred_l, true_l):
        denom = np.linalg.norm(true_l)
        return np.linalg.norm(pred_l - true_l) / denom if denom > 1e-10 else np.nan

    errs_tdeim = np.array([_rel_err(y_hat_3d[l], data_flat[l]) for l in range(n_L)])
    err_full   = np.linalg.norm(y_hat_3d - data_flat) / np.linalg.norm(data_flat)

    # PSNR — Tao et al. (2017) eq. 3: 20·log10(R_X / RMSE)
    # R_X = global value range; RMSE normalises by all N points in the 3D volume.
    R_X    = data_flat.max() - data_flat.min()
    rmse   = np.sqrt(np.mean((y_hat_3d - data_flat) ** 2))
    nrmse  = rmse / R_X
    psnr   = 20.0 * np.log10(R_X / rmse) if rmse > 0 else np.inf

    # ── Compression ratio (paper-style) ──────────────────────────────────────
    # Following the error-controlled lossy compression convention (cf. Tao et al.
    # 2017, IPDPS), CR = original_bytes / compressed_bytes.
    #
    # "Compressed" = everything needed to reconstruct the field from scratch:
    #
    #   Basis factors (offline, can be shared across snapshots):
    #     U_L[:,0:k]  left singular vectors   n_L × k   float32
    #     Vt[0:k,:]   right singular vectors  k  × n    float32
    #     mean_yz     per-(y,z) column mean   n          float32
    #     std_yz      per-(y,z) column std    n          float32
    #
    #   Per-snapshot data (the actual "compressed payload"):
    #     sensors_3d  p sensor index positions  p  int32
    #     y_obs_3d    p sensor observations      p  float32
    #
    # Note: when the basis is amortised over T snapshots, the per-snapshot cost
    # approaches (p_sensors + p_idx) values → CR ≈ N / p_3d.  The single-
    # snapshot CR below is the pessimistic lower bound.
    _B  = 4   # bytes: float32 = 4, int32 = 4
    _n_bytes_orig   = n_L * n   * _B            # original field (float32)
    _n_bytes_UL     = n_L * k   * _B            # U_L[:,0:k]
    _n_bytes_Vt     = k   * n   * _B            # Vt[0:k,:]
    _n_bytes_norm   = 2   * n   * _B            # mean_yz + std_yz
    _n_bytes_sidx   = p_3d      * _B            # sensor index positions (int32)
    _n_bytes_sval   = p_3d      * _B            # sensor observations (float32)
    _n_bytes_basis  = _n_bytes_UL + _n_bytes_Vt + _n_bytes_norm
    _n_bytes_comp   = _n_bytes_basis + _n_bytes_sidx + _n_bytes_sval
    cr_single       = _n_bytes_orig / _n_bytes_comp      # single-snapshot CR
    cr_amortized    = _n_bytes_orig / (_n_bytes_sidx + _n_bytes_sval)  # basis shared

    slice_local = SHOW_SLICE - SKIP_LEVELS
    assert 0 <= slice_local < n_L, \
        f"SHOW_SLICE={SHOW_SLICE} outside valid range [{SKIP_LEVELS},{SKIP_LEVELS+n_L})"

    print(f"  Full 3D rel. L2 error: {err_full*100:.2f}%")
    print(f"  PSNR={psnr:.2f} dB  |  RMSE={rmse:.4f} m/s  |  NRMSE={nrmse:.6f}")
    n_nan_t = np.sum(np.isnan(errs_tdeim))
    print(f"  Per-level error  min={np.nanmin(errs_tdeim)*100:.2f}%  "
          f"median={np.nanmedian(errs_tdeim)*100:.2f}%  "
          f"max={np.nanmax(errs_tdeim)*100:.2f}%  "
          f"({n_nan_t} near-zero levels excluded as NaN)")
    err_at_slice = errs_tdeim[slice_local]
    print(f"  Error at SHOW_SLICE={SHOW_SLICE}: "
          f"{'NaN (near-zero level)' if np.isnan(err_at_slice) else f'{err_at_slice*100:.2f}%'}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2D Q-DEIM — sensors restricted to one (y,z) plane, reconstruct all levels
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print(f"  2D Q-DEIM  (k={k} sensors in the y–z plane, all levels)")
    print(f"{'─'*62}")

    # Spatial modes: U_spatial = Vt.T, shape (n, r); top-k → 2D basis
    U_k_2d             = Vt.T[:, :k]           # (n, k)
    p_2d = k + OVERSAMPLE
    sensors_2d, deim_const_2d = qdeim_place(U_k_2d, p_2d)
    print(f"  DEIM_const_2D={deim_const_2d:.4f}  (p={p_2d} sensors)")

    # Reconstruct every level with the 2D basis
    y_hat_2d_all = np.empty((n_L, n))
    for l in range(n_L):
        if OVERSAMPLE > 0:
            c_l, _, _, _ = np.linalg.lstsq(U_k_2d[sensors_2d, :], A[l, sensors_2d],
                                            rcond=None)
        else:
            c_l = np.linalg.solve(U_k_2d[sensors_2d, :], A[l, sensors_2d])
        y_hat_2d_all[l]  = U_k_2d @ c_l
    y_hat_2d_all = y_hat_2d_all * std_yz + mean_yz   # denormalise

    errs_2d = np.array([_rel_err(y_hat_2d_all[l], data_flat[l]) for l in range(n_L)])
    n_nan_d = np.sum(np.isnan(errs_2d))
    print(f"  Per-level error  min={np.nanmin(errs_2d)*100:.2f}%  "
          f"median={np.nanmedian(errs_2d)*100:.2f}%  "
          f"max={np.nanmax(errs_2d)*100:.2f}%  "
          f"({n_nan_d} near-zero levels excluded as NaN)")
    err2d_at_slice = errs_2d[slice_local]
    print(f"  Error at SHOW_SLICE={SHOW_SLICE}: "
          f"{'NaN (near-zero level)' if np.isnan(err2d_at_slice) else f'{err2d_at_slice*100:.2f}%'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def _fmt(v):
        return 'NaN' if np.isnan(v) else f'{v*100:.2f}%'

    print(f"\n{'═'*62}")
    print(f"  k={k} sensors  |  p={p_3d}  |  Grid {ny}×{nz}  |  "
          f"ERROR_TOL={ERROR_TOL}  |  DS={DOWNSAMPLE}")
    print(f"  PSNR={psnr:.2f} dB  |  RMSE={rmse:.4f} m/s  |  NRMSE={nrmse:.6f}")
    print(f"  ── Compression ratio (paper-style, float32 original) ──")
    print(f"     Original:  {_n_bytes_orig/1e6:.2f} MB  ({n_L*n:,} float32 values)")
    print(f"     Basis:     {_n_bytes_basis/1e6:.2f} MB  "
          f"(UL {n_L}×{k} + Vt {k}×{n} + mean/std {n}×2)")
    print(f"     Payload:   {(_n_bytes_sidx+_n_bytes_sval)/1e3:.1f} KB  "
          f"({p_3d} sensor indices + {p_3d} observations)")
    print(f"     CR (single snapshot, basis included): {cr_single:.3f}x  "
          f"({'< 1 — basis larger than field' if cr_single < 1 else f'{cr_single:.1f}x reduction'})")
    print(f"     CR (basis amortised, payload only):   {cr_amortized:.1f}x  "
          f"(N / p = {n_L*n} / {p_3d})")
    print(f"  {'Method':<20} {'@ lv'+str(SHOW_SLICE):>10} "
          f"{'median':>9} {'max':>9} {'full-3D':>9}")
    print(f"  {'T-DEIM (3D)':<20} {_fmt(errs_tdeim[slice_local]):>10}"
          f" {np.nanmedian(errs_tdeim)*100:>8.2f}%"
          f" {np.nanmax(errs_tdeim)*100:>8.2f}%"
          f" {err_full*100:>8.2f}%")
    print(f"  {'2D Q-DEIM':<20} {_fmt(errs_2d[slice_local]):>10}"
          f" {np.nanmedian(errs_2d)*100:>8.2f}%"
          f" {np.nanmax(errs_2d)*100:>8.2f}%       —")
    print(f"  {'─'*60}")
    if GUARANTEE_MODE == 'abs':
        # Absolute error summary
        max_abs_err = float(np.max(np.abs(y_hat_3d - data_flat)))
        rmse_ok     = rmse        <= ABS_TOL
        linf_ok     = max_abs_err <= ABS_TOL
        print(f"  Error guarantee  (target ≤ {ABS_TOL} m/s, mode='abs')")
        print(f"    RMSE (primary, L2):   {rmse:.4f} m/s  "
              f"{'✓ MEETS TARGET' if rmse_ok else '✗ NOT MET'}")
        print(f"    Max pointwise (L∞):   {max_abs_err:.4f} m/s  "
              f"{'✓' if linf_ok else '✗ (DEIM is L2; L∞ not formally bounded)'}")
        if not rmse_ok:
            print(f"    → Lower ABS_TOL or reduce ERROR_TOL to add more modes.")
    elif GUARANTEE_MODE in ('trunc', 'apriori', None):
        emp_ok      = err_full <= ERROR_TOL
        bound_final = (1.0 + deim_const_3d) * proj_err[k - 1]
        pri_ok      = bound_final <= ERROR_TOL
        print(f"  Error guarantee  (target ≤ {ERROR_TOL*100:.1f}%,  "
              f"mode='{GUARANTEE_MODE}')")
        if GUARANTEE_MODE == 'trunc':
            print(f"    ε_POD (truncation): {proj_err[k-1]*100:.3f}%  "
                  f"{'✓' if proj_err[k-1] <= ERROR_TOL else '✗ NOT MET'}")
            print(f"    A priori bound (info): {bound_final*100:.2f}%  "
                  f"({'✓' if pri_ok else 'conservative — may exceed target'})")
        else:  # 'apriori'
            print(f"    A priori  bound : {bound_final*100:.2f}%  "
                  f"{'✓' if pri_ok else '✗ NOT MET'}")
        print(f"    Empirical full-3D: {err_full*100:.3f}%  "
              f"{'✓ MEETS TARGET' if emp_ok else '✗ NOT MET'}")
        if not emp_ok:
            print(f"    → Lower ERROR_TOL or switch GUARANTEE_MODE to 'apriori'.")
    print(f"  Wall time: {time.perf_counter()-t_start:.1f}s")
    print(f"{'═'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared geometry for figures
    # ─────────────────────────────────────────────────────────────────────────
    on_slice      = (lev_local == slice_local)
    ys_on, zs_on  = y_sens[on_slice], z_sens[on_slice]
    y2d           = sensors_2d // nz
    z2d           = sensors_2d % nz

    def _make_panels(l_idx):
        """Return (true_2d, tdeim_2d, deim2d_2d, err_t, err_d) for level l_idx."""
        true  = data_flat[l_idx].reshape(ny, nz)
        t_rec = y_hat_3d[l_idx].reshape(ny, nz)
        d_rec = y_hat_2d_all[l_idx].reshape(ny, nz)
        return true, t_rec, d_rec, np.abs(t_rec - true), np.abs(d_rec - true)

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — T-DEIM vs 2D Q-DEIM at SHOW_SLICE
    # ═════════════════════════════════════════════════════════════════════════
    true_s, tdeim_s, deim2_s, err_t_s, err_d_s = _make_panels(slice_local)
    emax_1   = max(err_t_s.max(), err_d_s.max())
    vmin_1, vmax_1 = true_s.min(), true_s.max()

    fig1, ax1 = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig1.suptitle(
        f'T-DEIM (3D) vs 2D Q-DEIM — ISABEL U-wind, Level {SHOW_SLICE}   |   '
        f'Grid: {ny}×{nz}  |  k={k} sensors  |  '
        f'T-DEIM err={_fmt(errs_tdeim[slice_local])}  '
        f'2D-DEIM err={_fmt(errs_2d[slice_local])}',
        fontsize=10
    )

    show_field(ax1[0,0], fig1, true_s,   f'True field  (level {SHOW_SLICE})',
               vmin=vmin_1, vmax=vmax_1)
    show_field(ax1[0,1], fig1, tdeim_s,
               f'T-DEIM recon  err={errs_tdeim[slice_local]*100:.2f}%',
               vmin=vmin_1, vmax=vmax_1,
               sx=ys_on, sy=zs_on,
               sensor_label=f'{on_slice.sum()}/{k} sensors on this level')
    show_field(ax1[0,2], fig1, err_t_s,  f'T-DEIM |error|  max={err_t_s.max():.2f} m/s',
               cmap='hot_r', vmin=0, vmax=emax_1, cbar_label='|Error| (m/s)')

    show_field(ax1[1,0], fig1, true_s,   f'True field  (level {SHOW_SLICE})',
               vmin=vmin_1, vmax=vmax_1)
    show_field(ax1[1,1], fig1, deim2_s,
               f'2D Q-DEIM recon  err={errs_2d[slice_local]*100:.2f}%',
               vmin=vmin_1, vmax=vmax_1,
               sx=y2d, sy=z2d,
               sensor_label=f'all {k} sensors in 2D plane')
    show_field(ax1[1,2], fig1, err_d_s,  f'2D Q-DEIM |error|  max={err_d_s.max():.2f} m/s',
               cmap='hot_r', vmin=0, vmax=emax_1, cbar_label='|Error| (m/s)')

    fig1.text(0.5, -0.01,
              f'Shared error colorscale (0 – {emax_1:.2f} m/s).  '
              f'T-DEIM sensors shown only for the {on_slice.sum()} that land '
              f'on level {SHOW_SLICE}; the other {k-on_slice.sum()} are at '
              f'different levels and inform the global solve.',
              ha='center', fontsize=7.5, style='italic', color='#555555')

    out1 = '/Users/jchen228/Desktop/Argonne/tdeim_comparison.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Where did T-DEIM put its sensors?
    #
    # Left panel  — VERTICAL DISTRIBUTION (histogram over levels)
    #   Each bar = number of sensors at that global level.
    #   Tall bars = levels the algorithm judged most informative for the
    #   full 3D structure.  Gaps = levels with no sensors (their structure
    #   is captured indirectly through the global basis).
    #
    # Right panel — SPATIAL SCATTER coloured by level
    #   Each dot = one sensor at its (lat, lon) position, coloured by
    #   which level it lives on (dark=low, bright=high).
    #   Interpretation:
    #     - Dots spread evenly → the algorithm needs information from all
    #       spatial locations to reconstruct the volume.
    #     - Colour clusters → high-altitude sensors prefer different (lat,lon)
    #       regions than low-altitude ones, meaning the spatial structure of
    #       the wind field changes significantly with height.
    # ═════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig2.suptitle(
        f'T-DEIM Sensor Distribution in 3D Space  |  '
        f'k={k} sensors across {n_L} levels  |  DOWNSAMPLE={DOWNSAMPLE}',
        fontsize=10
    )

    # Left: histogram of sensor levels
    bins = np.arange(SKIP_LEVELS - 0.5, SKIP_LEVELS + n_L + 0.5)
    ax2[0].hist(lev_global, bins=bins,
                color='steelblue', edgecolor='white', linewidth=0.5)
    ax2[0].axvline(SHOW_SLICE, color='red', lw=1.5, linestyle='--',
                   label=f'Level {SHOW_SLICE} (Fig 1 reference)')
    ax2[0].set_xlabel('Global level index (0 = surface, 99 = top)', fontsize=9)
    ax2[0].set_ylabel('Number of T-DEIM sensors', fontsize=9)
    ax2[0].set_title(
        'Vertical distribution of 3D sensors\n'
        'Tall bar = that level carries the most information about the full 3D field',
        fontsize=8.5
    )
    ax2[0].legend(fontsize=8)
    ax2[0].tick_params(labelsize=7)

    # Right: 2D scatter coloured by level
    sc  = ax2[1].scatter(z_sens, y_sens, c=lev_global, cmap='viridis',
                         s=45, marker='o', alpha=0.85, edgecolors='none')
    cb2 = fig2.colorbar(sc, ax=ax2[1])
    cb2.set_label('Global level index', fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax2[1].set_xlabel('Longitude index', fontsize=9)
    ax2[1].set_ylabel('Latitude index',  fontsize=9)
    ax2[1].set_title(
        'Spatial (lat, lon) positions of 3D sensors\n'
        'Colour = level  |  Clustering by colour → wind structure changes with altitude',
        fontsize=8.5
    )
    ax2[1].set_xlim(-0.5, nz - 0.5)
    ax2[1].set_ylim(-0.5, ny - 0.5)
    ax2[1].tick_params(labelsize=7)

    out2 = '/Users/jchen228/Desktop/Argonne/tdeim_sensor_distribution.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figures 3 and 4 are suppressed.
    #   Figure 3 (best/median/worst showcase) replaced by Figure 5 below.
    #   Figure 4 (3D volumetric) set to False — re-enable if needed.
    # ─────────────────────────────────────────────────────────────────────────
    _SHOW_FIG4 = False   # ← set True to re-enable 3D volume plot
    if _SHOW_FIG4:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        vol   = y_hat_3d.reshape(n_L, ny, nz)
        vol_s = vol[:, ::VIZ_DOWNSAMPLE, ::VIZ_DOWNSAMPLE]
        nL_v, nY_v, nZ_v = vol_s.shape
        zv_plane = np.arange(nZ_v) * VIZ_DOWNSAMPLE
        yv_plane = np.arange(nY_v) * VIZ_DOWNSAMPLE
        ZZ, YY = np.meshgrid(zv_plane, yv_plane)
        p99   = np.percentile(np.abs(vol_s), 99)
        norm4 = plt.Normalize(vmin=-p99, vmax=p99)
        fig4  = plt.figure(figsize=(13, 9))
        ax4   = fig4.add_subplot(111, projection='3d')
        for l_local in range(0, nL_v, LEVEL_STEP):
            l_global        = l_local + SKIP_LEVELS
            plane           = vol_s[l_local]
            rgba_plane      = plt.cm.RdBu_r(norm4(plane))
            alpha_plane     = np.clip((np.abs(plane) / max(p99, 1e-9)) ** 0.25, 0.08, 1.0)
            rgba_plane[:, :, 3] = alpha_plane
            LL = np.full_like(ZZ, l_global, dtype=float)
            ax4.plot_surface(ZZ, YY, LL, facecolors=rgba_plane, shade=False,
                             rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax4.set_xlabel('Longitude index', fontsize=9, labelpad=6)
        ax4.set_ylabel('Latitude index',  fontsize=9, labelpad=6)
        ax4.set_zlabel('Level (altitude)', fontsize=9, labelpad=6)
        ax4.tick_params(labelsize=7)
        sm4 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm4)
        sm4.set_array([])
        cb4 = fig4.colorbar(sm4, ax=ax4, shrink=0.55, pad=0.08)
        cb4.set_label('U-wind (m/s)', fontsize=9)
        cb4.ax.tick_params(labelsize=7)
        n_planes = len(range(0, nL_v, LEVEL_STEP))
        fig4.suptitle(
            f'T-DEIM 3D Reconstruction — Full Volume  |  k={k} sensors  |  '
            f'{n_planes} planes  |  Grid: {ny}×{nz}  |  '
            f'VIZ_DS={VIZ_DOWNSAMPLE}  LEVEL_STEP={LEVEL_STEP}',
            fontsize=10)
        out4 = '/Users/jchen228/Desktop/Argonne/tdeim_3d_volume.png'
        fig4.savefig(out4, dpi=150, bbox_inches='tight')
        print(f"Saved: {out4}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 5 — Per-level reconstruction error histogram
    #
    # Bar chart of T-DEIM relative L2 error at every level (x = level index,
    # y = error %).  Green bars are below ERROR_TOL; red bars exceed it.
    # Grey bars are near-zero levels (NaN relative error, excluded from
    # the guarantee check).  Three reference lines are drawn:
    #   — dashed black : ERROR_TOL target (1%)
    #   — solid blue   : global full-3D error (Frobenius, all levels combined)
    #   — solid orange : median per-level error (valid levels only)
    # The global error is the metric that ERROR_TOL applies to; per-level
    # bars can exceed 1% while the global error still satisfies the bound.
    # ═════════════════════════════════════════════════════════════════════════
    levels_global = np.arange(SKIP_LEVELS, SKIP_LEVELS + n_L)

    # Check whether all valid errors are at or near machine precision
    # (happens when k = full rank → exact reconstruction up to float64 eps ≈ 1e-16).
    valid_errs  = errs_tdeim[~np.isnan(errs_tdeim)]
    max_valid   = float(np.nanmax(errs_tdeim)) if valid_errs.size > 0 else 0.0
    near_mach   = max_valid < 1e-12      # effectively zero — at machine precision
    thr = (ERROR_TOL * 100) if GUARANTEE_MODE != 'abs' else None

    # Use log scale when all errors are much smaller than the target,
    # so the bars are not invisible on a linear axis.
    use_log = (thr is not None and max_valid * 100 < thr / 20) or near_mach

    # For log-scale bars we need a positive floor; zeros/NaN → small floor value
    LOG_FLOOR = 1e-14  # 10^-14 %
    if use_log:
        bar_pct = np.where(np.isnan(errs_tdeim) | (errs_tdeim == 0),
                           LOG_FLOOR, errs_tdeim * 100)
    else:
        bar_pct = np.where(np.isnan(errs_tdeim), 0.0, errs_tdeim * 100)

    bar_colors = []
    for i in range(n_L):
        if np.isnan(errs_tdeim[i]):
            bar_colors.append('#aaaaaa')   # grey  — near-zero level (NaN)
        elif thr is not None and errs_tdeim[i] * 100 > thr:
            bar_colors.append('#d62728')   # red   — exceeds target
        else:
            bar_colors.append('#2ca02c')   # green — within target

    fig5, ax5 = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax5.bar(levels_global, bar_pct, color=bar_colors, width=0.85, zorder=2)

    if use_log:
        ax5.set_yscale('log')
        ax5.set_ylabel('Relative L2 error (%)  [log scale]', fontsize=10)
    else:
        ax5.set_ylabel('Relative L2 error (%)', fontsize=10)

    if thr is not None:
        ax5.axhline(thr, color='black', linestyle='--', lw=1.5, zorder=3,
                    label=f'Guarantee target: {thr:.1f}%')
    ax5.axhline(err_full * 100, color='steelblue', linestyle='-', lw=1.8,
                zorder=3, label=f'Global 3D error (Frobenius): {err_full*100:.2e}%')
    med_valid = float(np.nanmedian(errs_tdeim)) * 100
    ax5.axhline(max(med_valid, LOG_FLOOR if use_log else 0),
                color='darkorange', linestyle=':', lw=1.5,
                zorder=3, label=f'Median per-level error: {med_valid:.2e}%')

    ax5.set_xlabel('Global level index  (0 = surface,  99 = top of domain)', fontsize=10)
    ax5.set_xlim(SKIP_LEVELS - 0.5, SKIP_LEVELS + n_L - 0.5)
    ax5.tick_params(labelsize=9)
    ax5.grid(axis='y', alpha=0.3, zorder=1)

    n_above = int(np.nansum(errs_tdeim > ERROR_TOL)) if GUARANTEE_MODE != 'abs' else 0
    n_nan   = int(np.sum(np.isnan(errs_tdeim)))
    n_valid = n_L - n_nan
    mach_note = '  ⚠ errors near machine precision — using log scale' if near_mach else (
                '  (log scale: errors much smaller than target)' if use_log else '')
    ax5.set_title(
        f'T-DEIM Per-Level Reconstruction Error  |  k={k} sensors  |  '
        f'Global error={err_full*100:.2e}%  |  PSNR={psnr:.1f} dB\n'
        f'{n_above}/{n_valid} valid levels exceed {thr:.1f}% target  '
        f'({n_nan} near-zero levels shown in grey){mach_note}',
        fontsize=10
    )
    ax5.legend(fontsize=9, loc='upper right')

    # Colour legend patches
    import matplotlib.patches as mpatches
    ax5.legend(handles=[
        mpatches.Patch(color='#2ca02c', label=f'Within target (<{thr:.1f}%)'),
        mpatches.Patch(color='#d62728', label=f'Exceeds target (≥{thr:.1f}%)'),
        mpatches.Patch(color='#aaaaaa', label='Near-zero level (NaN, excluded)'),
        plt.Line2D([0], [0], color='black',      linestyle='--', lw=1.5,
                   label=f'Guarantee target: {thr:.1f}%'),
        plt.Line2D([0], [0], color='steelblue',  linestyle='-',  lw=1.8,
                   label=f'Global 3D error: {err_full*100:.2f}%'),
        plt.Line2D([0], [0], color='darkorange', linestyle=':',  lw=1.5,
                   label=f'Median per-level: {med_valid:.2f}%'),
    ], fontsize=8.5, loc='upper right', framealpha=0.85)

    out5 = '/Users/jchen228/Desktop/Argonne/tdeim_level_errors.png'
    fig5.savefig(out5, dpi=150, bbox_inches='tight')
    print(f"Saved: {out5}")

    plt.show()

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 6 — SZ-style quantization histogram
    # ═════════════════════════════════════════════════════════════════════════
    # Simulates what SZ would see if it used our DEIM reconstruction as its
    # predictor and then tried to entropy-code the residuals:
    #
    #   bin_idx = round(residual / (2 × sz_eb))
    #
    # SZ default codebook: 1024 bins → outlier threshold at ±512.
    # Points outside ±512 are stored losslessly ("uncompressible").
    # Good predictions cluster near bin 0; poor predictions scatter widely.
    # ─────────────────────────────────────────────────────────────────────────
    SZ_N_BINS = 1024
    _half     = SZ_N_BINS // 2   # 512

    # Absolute error bound for SZ comparison (m/s)
    if GUARANTEE_MODE == 'abs':
        sz_eb = ABS_TOL
    else:
        sz_eb = ERROR_TOL * float(data_flat.max() - data_flat.min())

    resid_tdeim = (y_hat_3d     - data_flat).ravel()   # (n_L × n,)
    resid_2d    = (y_hat_2d_all - data_flat).ravel()

    def _quant_hist(ax, residuals, eb, half, title):
        import matplotlib.patches as mpatches
        bins_idx = np.round(residuals / (2.0 * eb)).astype(np.int64)
        n_total  = bins_idx.size
        n_out    = int(np.sum(np.abs(bins_idx) > half))
        n_zero   = int(np.sum(bins_idx == 0))
        pct_out  = 100.0 * n_out / n_total
        n_unique = int(np.unique(bins_idx[np.abs(bins_idx) <= half]).size)
        disp     = max(10, min(200, int(np.percentile(np.abs(bins_idx), 99)) + 5))
        counts, edges = np.histogram(bins_idx.clip(-disp, disp),
                                     bins=range(-disp, disp + 2))
        centers    = 0.5 * (edges[:-1] + edges[1:])
        bar_colors = ['#d62728' if abs(c) >= disp else '#2ca02c' for c in centers]
        ax.bar(centers, counts, width=0.9, color=bar_colors, zorder=2)
        for sign in (-1, 1):
            ax.axvline(sign * min(half, disp), color='black', ls='--', lw=1.4, zorder=3)
        ax.set_xlabel('Quantization bin index', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.legend(handles=[
            mpatches.Patch(color='#2ca02c', label='In-range  (compressible)'),
            mpatches.Patch(color='#d62728', label=f'Outlier  (|bin| ≥ {disp}, clipped here)'),
            plt.Line2D([0], [0], color='black', ls='--', lw=1.4,
                       label=f'SZ boundary ±{half}'),
        ], fontsize=7.5, loc='upper right')
        ax.set_title(
            f'{title}\n'
            f'eb = {eb:.4f} m/s  │  '
            f'outliers: {n_out:,} / {n_total:,} ({pct_out:.2f}%)  │  '
            f'unique bins in range: {n_unique}  │  '
            f'zero-bin: {n_zero:,} ({100*n_zero/n_total:.1f}%)',
            fontsize=8.5
        )

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig6.suptitle(
        'SZ-Style Quantization Histogram  —  T-DEIM vs 2D Q-DEIM residuals\n'
        f'bin_idx = round(residual / 2·eb)   '
        f'SZ codebook: {SZ_N_BINS} bins  (outlier threshold ±{_half})   '
        f"eb = {sz_eb:.4f} m/s  (GUARANTEE_MODE='{GUARANTEE_MODE}')",
        fontsize=10
    )
    _quant_hist(ax6a, resid_tdeim, sz_eb, _half, 'T-DEIM  (3D global solve, all levels)')
    _quant_hist(ax6b, resid_2d,    sz_eb, _half, '2D Q-DEIM  (layer-by-layer, all levels)')
    out6 = '/Users/jchen228/Desktop/Argonne/tdeim_quantization_hist.png'
    fig6.savefig(out6, dpi=150, bbox_inches='tight')
    print(f"Saved: {out6}")
    plt.show()

    plt.show()
