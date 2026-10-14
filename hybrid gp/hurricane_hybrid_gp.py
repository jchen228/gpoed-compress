"""
hurricane_hybrid_gp.py

Hybrid GP: GP-assisted dyadic compression of Hurricane ISABEL W field.

Algorithm
---------
Follows the SZ3 ALGO_INTERP dyadic schedule exactly, but replaces the
predictor for the first GP_LEVELS dyadic levels with a GP-based approach:

  Level 0  (coarse grid, stride INIT_STRIDE):
    1. Treat the ~S coarse-grid values as observations.
    2. Fit HP (ls_xy, ls_z, sig²) via MLE on these S points.
    3. RPCholesky+GKS selects ANCHOR_FRAC × S points → stored as raw float32.
    4. Full GP (no K_LOCAL – condition on the selected anchors only, S is small)
       predicts the remaining (1-ANCHOR_FRAC) × S coarse points.
    5. Quantise residuals; add reconstructed values to conditioning set.

  Levels 1 … GP_LEVELS-1  (first few dyadic refinement rounds):
    Same approach: collect all new midpoint positions for this level,
    condition on ALL previously resolved points (anchors + pseudo-anchors),
    predict ALL new midpoints in one full GP (no K_LOCAL).
    The conditioning set stays small enough for a full Cholesky because we
    only do this for the first GP_LEVELS levels.

  Levels GP_LEVELS … end  (remaining rounds):
    Pure SZ3-style cubic-spline interpolation using 4 nearest known values
    along the prediction axis.  O(N) and very fast.

Why this works
--------------
SZ3's two main storage costs are:
  (a) 100% of the coarse grid stored as raw float32.
  (b) Early rounds have large residuals (sparse grid → poor cubic spline)
      → high-entropy q-bins.
Hybrid GP fixes both by spending a small GP budget on the first few levels.

No K_LOCAL anywhere: the GP only predicts dyadic grid points (not arbitrary
interiors), and the conditioning set is the anchor subset, which is small.

Backburner
----------
1. Adaptive GP_LEVELS: switch to cubic spline when mean |q_bin| < threshold.
2. HP re-fitting per level.
3. RPCholesky selection at levels 1+ as well (currently predict ALL midpoints
   at those levels from the growing anchor set).
"""

import numpy as np
import time
import sys
import subprocess
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# TEE
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, path):
        self._file = open(path, 'w', buffering=1)
        self._stdout = sys.stdout
    def write(self, d):  self._stdout.write(d); self._file.write(d)
    def flush(self):     self._stdout.flush();  self._file.flush()
    def __enter__(self): sys.stdout = self; return self
    def __exit__(self, *_): sys.stdout = self._stdout; self._file.close()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = '/Users/jchen228/Desktop/Argonne/100x500x500/Wf48.bin.f32'
SZ3_BIN    = '/Users/jchen228/Desktop/Argonne/SZ3/build/tools/sz3/sz3'
QOZ_BIN    = '/Users/jchen228/Desktop/Argonne/QoZ/install/bin/qoz'
OUT_DIR    = Path('/Users/jchen228/Desktop/Argonne')

NZ, NY, NX = 100, 500, 500
N          = NZ * NY * NX     # 25,000,000

EB_VALUES   = [0.005, 0.01, 0.02, 0.05, 0.1]
INIT_STRIDE = 32               # coarse-grid stride (level 0)
ANCHOR_FRAC = 0.05             # fraction of level-0 grid stored as raw float32
GP_LEVELS   = 2                # number of dyadic levels handled by GP
                               # level 0 = coarse grid, level 1 = first midpoints
NOISE       = 1e-4             # GP nugget for full GP solve
HP_N_OBS    = None             # None → use all coarse-grid points for HP fitting
SEED        = 42

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> np.ndarray:
    return np.fromfile(DATA_PATH, dtype=np.float32).reshape(NZ, NY, NX).astype(np.float64)


def flat_to_coords(flat_idx: np.ndarray) -> np.ndarray:
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    iz = flat_idx // (NY * NX)
    iy = (flat_idx % (NY * NX)) // NX
    ix = flat_idx % NX
    return np.stack([iz / max(NZ-1,1),
                     iy / max(NY-1,1),
                     ix / max(NX-1,1)], axis=1).astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def quantize_sz2(errs: np.ndarray, eb: float):
    safe  = np.where(np.isnan(errs), 0.0, errs)
    q_bin = np.round(safe / (2.0 * eb)).astype(np.int64)
    q_err = safe - q_bin * (2.0 * eb)
    return q_bin, q_err


def entropy_from_bins(q_bin: np.ndarray) -> float:
    flat   = q_bin.ravel().astype(np.int64)
    flat  -= flat.min()
    counts = np.bincount(flat)
    p      = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def psnr_db(errors: np.ndarray, data_range: float) -> float:
    rmse = float(np.sqrt(np.mean(np.asarray(errors)**2)))
    return 20.0 * np.log10(data_range / rmse) if rmse > 0 else float('inf')

# ─────────────────────────────────────────────────────────────────────────────
# KERNEL  (anisotropic Matérn-5/2)
# ─────────────────────────────────────────────────────────────────────────────
def matern52_3d(X1, X2, ls_xy, ls_z, sig2):
    dz = (X1[:,0:1] - X2[:,0]) / ls_z
    dy = (X1[:,1:2] - X2[:,1]) / ls_xy
    dx = (X1[:,2:3] - X2[:,2]) / ls_xy
    r2 = dz**2 + dy**2 + dx**2
    r  = np.sqrt(np.maximum(r2, 0.0))
    return sig2 * (1.0 + np.sqrt(5)*r + 5*r2/3.0) * np.exp(-np.sqrt(5)*r)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER FITTING  (on level-0 coarse grid)
# ─────────────────────────────────────────────────────────────────────────────
def fit_hyperparams(coords: np.ndarray, vals_norm: np.ndarray,
                    max_obs: int = 800) -> tuple[float, float, float]:
    """
    GP MLE on the provided (coords, vals_norm).
    Uses true data values at the level's dyadic positions so HPs reflect
    the actual correlation structure at that resolution.
    Subsamples to max_obs if needed.
    """
    from scipy.optimize import minimize
    if len(vals_norm) > max_obs:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(vals_norm), max_obs, replace=False)
        coords    = coords[idx]
        vals_norm = vals_norm[idx]
    n = len(vals_norm)

    def neg_log_ml(lp):
        ls_xy = np.exp(lp[0]); ls_z = np.exp(lp[1]); sig2 = np.exp(lp[2])
        K = matern52_3d(coords, coords, ls_xy, ls_z, sig2) + NOISE * np.eye(n)
        try:
            L     = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, vals_norm))
            ldet  = 2.0 * np.sum(np.log(np.diag(L)))
            return float(0.5*vals_norm@alpha + 0.5*ldet + 0.5*n*np.log(2*np.pi))
        except Exception:
            return 1e10

    best = None
    for ls_xy0, ls_z0, s0 in [(0.15,0.15,1.0),(0.05,0.05,0.5),(0.30,0.30,1.5)]:
        res = minimize(neg_log_ml, np.log([ls_xy0, ls_z0, s0]), method='L-BFGS-B',
                       bounds=[(-4,1),(-4,1),(-3,3)],
                       options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        if best is None or res.fun < best.fun:
            best = res
    ls_xy = float(np.exp(best.x[0]))
    ls_z  = float(np.exp(best.x[1]))
    sig2  = float(np.exp(best.x[2]))
    print(f'    HP: ls_xy={ls_xy:.4f}  ls_z={ls_z:.4f}  sig2={sig2:.4f}  '
          f'converged={best.success}  n_obs={n}')
    return ls_xy, ls_z, sig2

# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SELECTION  (RPCholesky + GKS greedy pivot)
# ─────────────────────────────────────────────────────────────────────────────
def select_sensors(cand_coords: np.ndarray, k: int,
                   rng: np.random.Generator,
                   ls_xy: float, ls_z: float, sig2: float) -> np.ndarray:
    n = len(cand_coords)
    if k >= n: return np.arange(n)
    d   = sig2 * np.ones(n)
    sel = []
    for _ in range(k):
        probs = np.maximum(d, 0.0); s = probs.sum()
        if s <= 0: break
        idx = int(rng.choice(n, p=probs/s))
        sel.append(idx)
        k_col = matern52_3d(cand_coords, cand_coords[idx:idx+1],
                            ls_xy, ls_z, sig2).ravel()
        d -= k_col**2 / max(float(d[idx]), 1e-12)
        d  = np.maximum(d, 0.0); d[idx] = 0.0
    return np.array(sel, dtype=np.int64)

# ─────────────────────────────────────────────────────────────────────────────
# FULL GP PREDICTION  (no K_LOCAL – condition on small anchor set)
# ─────────────────────────────────────────────────────────────────────────────
def gp_predict_full(anchor_coords: np.ndarray, anchor_vals_norm: np.ndarray,
                    target_coords: np.ndarray,
                    ls_xy: float, ls_z: float, sig2: float,
                    noise: float = NOISE) -> np.ndarray:
    """
    Predict at target_coords conditioning on all anchor points.
    No K_LOCAL approximation — the anchor set is small by design.
    Returns normalised predictions.
    """
    n_a = len(anchor_coords)
    if n_a == 0:
        return np.zeros(len(target_coords))
    K_aa = matern52_3d(anchor_coords, anchor_coords, ls_xy, ls_z, sig2)
    K_aa += noise * np.eye(n_a)
    try:
        L    = np.linalg.cholesky(K_aa)
    except np.linalg.LinAlgError:
        K_aa += 1e-4 * np.eye(n_a)
        L    = np.linalg.cholesky(K_aa)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, anchor_vals_norm))
    K_ta  = matern52_3d(target_coords, anchor_coords, ls_xy, ls_z, sig2)
    return K_ta @ alpha

# ─────────────────────────────────────────────────────────────────────────────
# DYADIC SCHEDULE  (analytic, mirrors SZ3 ALGO_INTERP)
# ─────────────────────────────────────────────────────────────────────────────
def dyadic_schedule(size: int, stride: int) -> list:
    """
    Returns schedule[k] = np.array of NEW positions first seen at level k.
    Level 0 = coarse grid (stride apart + boundary).
    """
    pos0 = list(range(0, size, stride))
    if pos0[-1] != size - 1:
        pos0.append(size - 1)
    known_sorted = sorted(pos0)
    known_set    = set(known_sorted)
    schedule     = [np.array(known_sorted, dtype=np.int32)]
    while len(known_set) < size:
        new = []
        for a, b in zip(known_sorted[:-1], known_sorted[1:]):
            if b - a > 1:
                m = (a + b) // 2
                if m not in known_set:
                    new.append(m)
        if not new:
            break
        new = sorted(set(new))
        schedule.append(np.array(new, dtype=np.int32))
        known_set.update(new)
        known_sorted = sorted(known_sorted + new)
    return schedule


def midpoint_flat_indices(new_dim_pos, cum_other1, cum_other2, dim):
    """Return flat indices for new positions along `dim`, at all cumulative
    positions in the other two dimensions."""
    if dim == 'x':
        iz, iy, ix = np.meshgrid(cum_other1, cum_other2, new_dim_pos, indexing='ij')
    elif dim == 'y':
        iz, iy, ix = np.meshgrid(cum_other1, new_dim_pos, cum_other2, indexing='ij')
    else:
        iz, iy, ix = np.meshgrid(new_dim_pos, cum_other1, cum_other2, indexing='ij')
    return (iz.ravel().astype(np.int64) * NY * NX
            + iy.ravel().astype(np.int64) * NX
            + ix.ravel().astype(np.int64))

# ─────────────────────────────────────────────────────────────────────────────
# CUBIC-SPLINE INTERPOLATION  (SZ3-style 4-point formula)
# ─────────────────────────────────────────────────────────────────────────────
def cubic_interp_predict(vol_recon: np.ndarray,
                         new_pos: np.ndarray,
                         known_sorted: np.ndarray,
                         cum1: np.ndarray, cum2: np.ndarray,
                         dim: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict new midpoints along `dim` using the SZ3 4-point cubic formula:
        p = (-v_a + 9*v_b + 9*v_c - v_d) / 16
    Falls back to linear at boundaries.

    KEY: preds_3d is built with the SAME axis order as midpoint_flat_indices
    so that flat_all[i] and preds[i] always refer to the same point.

    midpoint_flat_indices axis order (indexing='ij'):
      dim='x': (cum1=z, cum2=y, new_pos=x)  → preds_3d shape (n1, n2, n_new)
      dim='y': (cum1=z, new_pos=y, cum2=x)  → preds_3d shape (n1, n_new, n2)
      dim='z': (new_pos=z, cum1=y, cum2=x)  → preds_3d shape (n_new, n1, n2)
    """
    flat_all = midpoint_flat_indices(new_pos, cum1, cum2, dim)
    vol_flat = vol_recon.ravel()
    ks       = known_sorted
    n1, n2, n_new = len(cum1), len(cum2), len(new_pos)

    # Build preds in the same axis order as flat_all
    if dim == 'x':
        preds_3d = np.zeros((n1, n2, n_new), dtype=np.float64)
    elif dim == 'y':
        preds_3d = np.zeros((n1, n_new, n2), dtype=np.float64)
    else:
        preds_3d = np.zeros((n_new, n1, n2), dtype=np.float64)

    def get_vals(p_fixed: int) -> np.ndarray:
        """
        vol values at all (cum1, cum2) pairs for fixed position along dim.
        Returns shape (n1, n2) in row-major order.
        """
        if dim == 'x':
            IZ, IY = np.meshgrid(cum1, cum2, indexing='ij')
            fi = IZ.ravel().astype(np.int64)*NY*NX + IY.ravel().astype(np.int64)*NX + p_fixed
        elif dim == 'y':
            IZ, IX = np.meshgrid(cum1, cum2, indexing='ij')
            fi = IZ.ravel().astype(np.int64)*NY*NX + p_fixed*NX + IX.ravel().astype(np.int64)
        else:
            IY, IX = np.meshgrid(cum1, cum2, indexing='ij')
            fi = p_fixed*NY*NX + IY.ravel().astype(np.int64)*NX + IX.ravel().astype(np.int64)
        return vol_flat[fi].reshape(n1, n2)

    for i_mid, mp in enumerate(new_pos):
        idx_r = int(np.searchsorted(ks, mp))
        idx_l = idx_r - 1
        idx_l = max(idx_l, 0); idx_r = min(idx_r, len(ks)-1)
        pb = int(ks[idx_l]); pc = int(ks[idx_r])
        boundary = (idx_l == 0) or (idx_r == len(ks)-1)

        vb = get_vals(pb); vc = get_vals(pc)
        if boundary:
            formula = 0.5 * (vb + vc)                      # shape (n1, n2)
        else:
            pa = int(ks[max(idx_l-1, 0)]); pd = int(ks[min(idx_r+1, len(ks)-1)])
            va = get_vals(pa); vd = get_vals(pd)
            formula = (-va + 9*vb + 9*vc - vd) / 16.0      # shape (n1, n2)

        # Write into the slice corresponding to this midpoint position,
        # matching the axis order of flat_all exactly.
        if dim == 'x':
            preds_3d[:, :, i_mid] = formula
        elif dim == 'y':
            preds_3d[:, i_mid, :] = formula
        else:
            preds_3d[i_mid, :, :] = formula

    return flat_all, preds_3d.ravel()

# ─────────────────────────────────────────────────────────────────────────────
# SZ3 BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def _run_sz3(vol: np.ndarray, eb: float, algo: str) -> dict | None:
    """Generic SZ3 runner; algo is e.g. 'ALGO_INTERP' or 'ALGO_LORENZO_REG'."""
    vol32      = vol.astype(np.float32)
    data_range = float(vol.max() - vol.min())
    with tempfile.TemporaryDirectory() as tmp:
        tmp   = Path(tmp)
        cfg   = tmp / 'sz3.cfg'
        cfg.write_text(f'[GlobalSettings]\nCmprAlgo = {algo}\n'
                       f'ErrorBoundMode = ABS\nAbsErrorBound = {eb}\n')
        f_in  = tmp / 'vol.f32'; f_sz = tmp / 'vol.sz3'; f_dec = tmp / 'vol_dec.f32'
        vol32.tofile(f_in)
        cmd = [SZ3_BIN, '-f', '-i', str(f_in), '-z', str(f_sz), '-o', str(f_dec),
               '-3', str(NX), str(NY), str(NZ), '-c', str(cfg)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f'  SZ3 {algo} FAILED EB={eb}: {res.stderr.strip()}'); return None
        if not f_sz.exists():
            alt = Path(str(f_in)+'.sz3')
            if alt.exists(): alt.rename(f_sz)
            else: print(f'  SZ3 {algo} no output EB={eb}'); return None
        nbytes  = f_sz.stat().st_size
        vol_dec = (np.fromfile(f_dec, dtype=np.float32)
                   .reshape(NZ, NY, NX).astype(np.float64))
    errors   = vol - vol_dec
    bit_rate = nbytes * 8 / N
    cr       = (N * 4) / nbytes
    psnr     = psnr_db(errors, data_range)
    label    = 'SZ3-Interp' if algo == 'ALGO_INTERP' else 'SZ3-Lorenzo'
    print(f'  {label} EB={eb}: {bit_rate:.3f} bpv  CR={cr:.2f}×  PSNR={psnr:.2f} dB')
    return dict(eb=eb, bit_rate=bit_rate, cr=cr, psnr=psnr)


def run_sz3_interp(vol: np.ndarray, eb: float) -> dict | None:
    return _run_sz3(vol, eb, 'ALGO_INTERP')




def run_qoz(vol: np.ndarray, eb: float, level: int = 3) -> dict | None:
    """
    QoZ baseline (szcompressor/QoZ).
    CLI: qoz -f -i <in> -z <cmp> -o <dec> -3 NX NY NZ -M ABS <eb> -q <level>

    level 1: basic interpolation
    level 2: + multi-dim interp, natural cubic spline, re-ordering, dim freezing
    level 3: full quality-optimized (default)
    level 4: highest quality

    Skipped gracefully if QOZ_BIN does not exist — set QOZ_BIN in CONFIG.
    """
    if not Path(QOZ_BIN).exists():
        print(f'  QoZ binary not found ({QOZ_BIN}) — skipping EB={eb}')
        return None
    vol32      = vol.astype(np.float32)
    data_range = float(vol.max() - vol.min())
    with tempfile.TemporaryDirectory() as tmp:
        tmp   = Path(tmp)
        f_in  = tmp / 'vol.f32'
        f_cmp = tmp / 'vol.qoz'
        f_dec = tmp / 'vol_dec.f32'
        vol32.tofile(f_in)
        cmd = [QOZ_BIN, '-f',
               '-i', str(f_in), '-z', str(f_cmp), '-o', str(f_dec),
               '-3', str(NX), str(NY), str(NZ),
               '-M', 'ABS', str(eb),
               '-q', str(level)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f'  QoZ-{level} FAILED EB={eb}: {res.stderr.strip()}'); return None
        # QoZ may write <input>.qoz instead of the -z path
        if not f_cmp.exists():
            alt = Path(str(f_in) + '.qoz')
            if alt.exists():
                alt.rename(f_cmp)
            else:
                print(f'  QoZ-{level} no output EB={eb}'); return None
        nbytes  = f_cmp.stat().st_size
        vol_dec = (np.fromfile(f_dec, dtype=np.float32)
                   .reshape(NZ, NY, NX).astype(np.float64))
    errors   = vol - vol_dec
    bit_rate = nbytes * 8 / N
    cr       = (N * 4) / nbytes
    psnr     = psnr_db(errors, data_range)
    print(f'  QoZ-{level} EB={eb}: {bit_rate:.3f} bpv  CR={cr:.2f}×  PSNR={psnr:.2f} dB')
    return dict(eb=eb, bit_rate=bit_rate, cr=cr, psnr=psnr)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC: per-level bit breakdown
# ─────────────────────────────────────────────────────────────────────────────
def print_level_breakdown(level_log: list, n_anchor_bits: int, N: int,
                          eb: float) -> None:
    """
    Print a table showing exactly where bits are spent at each level.
    Columns:
      Level | Method | N_pts | N_anchor | N_pred | H (bits) | Bits_this_level
           | % of total | %|q|≤1 | Cumul BR
    This directly answers:
      - Are GP-level q-bins tighter than cubic-spline levels?
      - Which levels dominate total storage?
      - Are anchor bits negligible?
    """
    total_pred_bits = sum(d['bits'] for d in level_log)
    total_bits      = n_anchor_bits + total_pred_bits
    cum_bits        = n_anchor_bits  # anchors paid upfront

    print(f'\n  {"─"*95}')
    print(f'  {"Lv":>3} {"Method":>6} {"N_pts":>9} {"N_anc":>7} {"N_pred":>8} '
          f'{"H":>6} {"Bits":>12} {"% total":>8} {"% |q|≤1":>9} {"Cum BR":>8}')
    print(f'  {"─"*95}')

    # Anchor row
    pct_anc = 100.0 * n_anchor_bits / total_bits if total_bits else 0.0
    cum_bits_disp = n_anchor_bits
    print(f'  {"—":>3} {"anchors":>6} {"":>9} {level_log[0]["n_anchor"]:>7} {"":>8} '
          f'{"32.0":>6} {n_anchor_bits:>12,} {pct_anc:>7.1f}% {"100.0%":>9} '
          f'{cum_bits_disp/N:>8.4f}')

    for d in level_log:
        cum_bits += d['bits']
        pct_total = 100.0 * d['bits'] / total_bits if total_bits else 0.0
        print(f'  {d["level"]:>3} {d["method"]:>6} {d["n_pts"]:>9,} '
              f'{d["n_anchor"]:>7} {d["n_pred"]:>8,} '
              f'{d["H"]:>6.3f} {d["bits"]:>12,.0f} {pct_total:>7.1f}% '
              f'{d["pct_ok"]:>8.1f}% {cum_bits/N:>8.4f}')

    print(f'  {"─"*95}')
    print(f'  Total bits: {total_bits:,.0f}  →  {total_bits/N:.4f} bpv  '
          f'(anchors = {100*n_anchor_bits/total_bits:.1f}% of total)\n')


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID GP MAIN COMPRESSION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_hybrid_gp(vol: np.ndarray, eb: float) -> dict:
    """
    Returns a metrics dict with bit_rate, cr, psnr and per-level detail.
    """
    t0         = time.perf_counter()
    rng        = np.random.default_rng(SEED)
    y_full     = vol.ravel().astype(np.float64)
    y_mean     = y_full.mean()
    y_std      = max(y_full.std(), 1e-8)
    data_range = float(vol.max() - vol.min())

    NOISE_PSEUDO = eb / y_std  # noise assigned to quantised pseudo-anchors

    # Reconstructed volume (for cubic spline predictor)
    vol_recon = np.full(N, y_mean, dtype=np.float64)

    is_anchor    = np.zeros(N, dtype=bool)
    is_predicted = np.zeros(N, dtype=bool)
    recon_err    = np.zeros(N, dtype=np.float64)
    all_q_bins   = np.zeros(N, dtype=np.int64)
    n_resolved    = 0
    n_anchor_bits = 0

    # Per-level diagnostic log: list of dicts
    level_log = []

    # Build dyadic schedules for each dimension
    sched_z = dyadic_schedule(NZ, INIT_STRIDE)
    sched_y = dyadic_schedule(NY, INIT_STRIDE)
    sched_x = dyadic_schedule(NX, INIT_STRIDE)
    n_levels = max(len(sched_z), len(sched_y), len(sched_x))

    # Cumulative known positions per dimension (grow each level)
    cum_z = sched_z[0].tolist()
    cum_y = sched_y[0].tolist()
    cum_x = sched_x[0].tolist()

    # ── Level 0: GP on coarse grid ────────────────────────────────────────
    print(f'  Level 0: coarse grid  stride={INIT_STRIDE}')
    tL = time.perf_counter()

    # Flat indices of ALL level-0 coarse grid points
    IZ0, IY0, IX0 = np.meshgrid(sched_z[0], sched_y[0], sched_x[0], indexing='ij')
    coarse_flat = (IZ0.ravel().astype(np.int64)*NY*NX
                   + IY0.ravel().astype(np.int64)*NX
                   + IX0.ravel().astype(np.int64))
    S = len(coarse_flat)
    print(f'    Coarse grid size: {S:,} points')

    coarse_coords    = flat_to_coords(coarse_flat)
    coarse_vals_raw  = y_full[coarse_flat]
    coarse_vals_norm = (coarse_vals_raw - y_mean) / y_std

    # Fit HP on coarse grid (subsample if very large)
    hp_idx = (np.random.default_rng(SEED)
              .choice(S, min(S, 800), replace=False) if S > 800
              else np.arange(S))
    print(f'    Fitting HP on {len(hp_idx)} coarse-grid points…')
    ls_xy, ls_z, sig2 = fit_hyperparams(coarse_coords[hp_idx],
                                         coarse_vals_norm[hp_idx])

    # RPCholesky selects ANCHOR_FRAC of coarse grid as true anchors
    K_anc = max(1, int(round(S * ANCHOR_FRAC)))
    sel   = select_sensors(coarse_coords, K_anc, rng, ls_xy, ls_z, sig2)
    print(f'    RPCholesky selected {len(sel)}/{S} anchors ({100*len(sel)/S:.1f}%)')

    # Store anchor values
    anc_flat   = coarse_flat[sel]
    is_anchor[anc_flat] = True
    anc_coords = coarse_coords[sel]
    anc_vals_n = coarse_vals_norm[sel]
    vol_recon.ravel()[anc_flat] = y_full[anc_flat]
    n_anchor_bits += len(anc_flat) * 32
    n_resolved    += len(anc_flat)

    # Predict remaining coarse-grid points with full GP
    pred_mask = np.ones(S, dtype=bool); pred_mask[sel] = False
    pred_flat = coarse_flat[pred_mask]
    if len(pred_flat):
        pred_coords  = coarse_coords[pred_mask]
        pred_norm    = gp_predict_full(anc_coords, anc_vals_n,
                                       pred_coords, ls_xy, ls_z, sig2)
        pred_vals    = pred_norm * y_std + y_mean
        q_bin_0, q_err_0 = quantize_sz2(y_full[pred_flat] - pred_vals, eb)
        recon_err[pred_flat]    = q_err_0
        all_q_bins[pred_flat]   = q_bin_0
        is_predicted[pred_flat] = True
        recon_0 = pred_vals + q_bin_0 * (2.0 * eb)
        vol_recon.ravel()[pred_flat] = recon_0
        n_resolved += len(pred_flat)
        pct_ok = 100.0 * np.mean(np.abs(q_bin_0) <= 1)
        print(f'    Coarse predictions: {pct_ok:.1f}% within ±1 bin')

    # ── Level 0 diagnostic ───────────────────────────────────────────────
    lv0_q = all_q_bins[is_predicted & ~is_anchor]  # GP-predicted coarse pts
    lv0_H = entropy_from_bins(lv0_q) if len(lv0_q) else 0.0
    lv0_bits = n_anchor_bits + lv0_H * len(lv0_q)
    level_log.append(dict(
        level=0, method='GP', n_pts=S,
        n_anchor=len(anc_flat), n_pred=len(lv0_q),
        H=lv0_H, bits=lv0_bits,
        pct_ok=100.0*np.mean(np.abs(lv0_q)<=1) if len(lv0_q) else 100.0,
        cumulative_br=lv0_bits/N,
    ))
    print(f'    Level 0 done  [{time.perf_counter()-tL:.1f}s]')

    # Anchor conditioning set (used if GP_LEVELS > 1)
    # We grow this as pseudo-anchors are added
    all_anc_coords = [anc_coords]
    all_anc_vals_n = [anc_vals_n]
    if len(pred_flat):
        recon_pred_norm = (vol_recon.ravel()[pred_flat] - y_mean) / y_std
        all_anc_coords.append(flat_to_coords(pred_flat))
        all_anc_vals_n.append(recon_pred_norm)

    # ── Levels 1 … GP_LEVELS-1: GP on dyadic midpoints ───────────────────
    # At each level:
    #   1. Collect all new midpoint positions (across X, Y, Z sweeps).
    #   2. Fit HPs on TRUE data values at those positions — captures the
    #      correlation structure at this level's resolution.
    #   3. Full GP: condition on ALL previously resolved points
    #      (reconstructed values bounded to ±EB, same as SZ3).
    #   4. Predict → quantize → reconstruct.  Error bounded to ±EB.
    for level in range(1, GP_LEVELS):
        tL = time.perf_counter()

        # Collect new midpoint flat indices for this level.
        # Process X, Y, Z in order, updating cum lists between sweeps so
        # newly predicted X midpoints feed into Y predictions (mirrors SZ3).
        level_flat_parts = []
        for dim, sched, c1, c2 in [
            ('x', sched_x, cum_z, cum_y),
            ('y', sched_y, cum_z, cum_x),
            ('z', sched_z, cum_y, cum_x),
        ]:
            if level >= len(sched):
                continue
            flat_p = midpoint_flat_indices(sched[level],
                                           np.array(c1, dtype=np.int32),
                                           np.array(c2, dtype=np.int32), dim)
            level_flat_parts.append(flat_p)

        if not level_flat_parts:
            continue
        level_flat  = np.unique(np.concatenate(level_flat_parts))
        tgt_coords  = flat_to_coords(level_flat)
        print(f'  Level {level}: {len(level_flat):,} new midpoints  [GP]')

        # ── Fit HPs on TRUE data values at this level's positions ─────────
        # We have the full data, so reading true values here is valid.
        # These HPs capture the spatial correlation at level-1 resolution,
        # which differs from level-0 (different dominant length scales).
        true_vals_here = y_full[level_flat]
        true_norm_here = (true_vals_here - y_mean) / y_std
        print(f'    Fitting HP on level-{level} dyadic points…')
        ls_xy_l, ls_z_l, sig2_l = fit_hyperparams(tgt_coords, true_norm_here)

        # ── Full GP prediction ────────────────────────────────────────────
        # Condition on all previously resolved points (anchors + pseudo-anchors).
        # Their reconstructed values are within ±EB of truth — same guarantee
        # as SZ3's conditioning set at this level.
        cond_coords = np.vstack(all_anc_coords)
        cond_vals_n = np.concatenate(all_anc_vals_n)

        pred_norm_l = gp_predict_full(cond_coords, cond_vals_n,
                                      tgt_coords,
                                      ls_xy_l, ls_z_l, sig2_l,
                                      noise=NOISE_PSEUDO)
        pred_vals_l = pred_norm_l * y_std + y_mean

        # Quantize: error bounded to ±EB regardless of prediction quality
        q_bin_l, q_err_l = quantize_sz2(y_full[level_flat] - pred_vals_l, eb)
        recon_err[level_flat]    = q_err_l
        all_q_bins[level_flat]   = q_bin_l
        is_predicted[level_flat] = True
        recon_l = pred_vals_l + q_bin_l * (2.0 * eb)
        vol_recon.ravel()[level_flat] = recon_l
        n_resolved += len(level_flat)

        pct_ok = 100.0 * np.mean(np.abs(q_bin_l) <= 1)
        lv_H   = entropy_from_bins(q_bin_l)
        lv_bits = lv_H * len(level_flat)
        cum_bits = (n_anchor_bits
                    + sum(d['bits'] for d in level_log if d['level'] > 0)
                    + lv_bits
                    + entropy_from_bins(all_q_bins[level_log[0]['n_anchor']:
                                                   level_log[0]['n_anchor']+level_log[0]['n_pred']])
                      * level_log[0]['n_pred'])
        level_log.append(dict(
            level=level, method='GP', n_pts=len(level_flat),
            n_anchor=0, n_pred=len(level_flat),
            H=lv_H, bits=lv_bits,
            pct_ok=pct_ok,
            cumulative_br=None,  # filled in print_level_breakdown
        ))
        print(f'    {pct_ok:.1f}% within ±1 bin  H={lv_H:.3f}  [{time.perf_counter()-tL:.1f}s]')

        # Add reconstructed values to conditioning set for next level
        all_anc_coords.append(tgt_coords)
        all_anc_vals_n.append((recon_l - y_mean) / y_std)

        # Update cumulative positions
        for sched, cum_list in [(sched_z, cum_z), (sched_y, cum_y), (sched_x, cum_x)]:
            if level < len(sched):
                cum_list.extend(sched[level].tolist())
                cum_list.sort()

    # ── Levels GP_LEVELS … end: cubic-spline interpolation (SZ3-style) ───
    vol_recon_3d = vol_recon.reshape(NZ, NY, NX)

    for level in range(GP_LEVELS, n_levels):
        tL = time.perf_counter()
        n_new = 0

        for dim, sched, cum_list_self, cum_list_a, cum_list_b in [
            ('x', sched_x, cum_x, cum_z, cum_y),
            ('y', sched_y, cum_y, cum_z, cum_x),
            ('z', sched_z, cum_z, cum_y, cum_x),
        ]:
            if level >= len(sched):
                continue
            new_pos    = sched[level]
            known_arr  = np.array(sorted(cum_list_self), dtype=np.int32)
            cum_a      = np.array(sorted(cum_list_a),    dtype=np.int32)
            cum_b      = np.array(sorted(cum_list_b),    dtype=np.int32)

            flat_p, pred_p = cubic_interp_predict(
                vol_recon_3d, new_pos, known_arr, cum_a, cum_b, dim)

            q_bin_p, q_err_p = quantize_sz2(y_full[flat_p] - pred_p, eb)
            recon_err[flat_p]    = q_err_p
            all_q_bins[flat_p]   = q_bin_p
            is_predicted[flat_p] = True
            recon_p = pred_p + q_bin_p * (2.0 * eb)
            vol_recon.ravel()[flat_p] = recon_p
            vol_recon_3d = vol_recon.reshape(NZ, NY, NX)
            n_resolved += len(flat_p)
            n_new      += len(flat_p)

            # Update cumulative
            for p in new_pos:
                if p not in cum_list_self:
                    cum_list_self.append(int(p))
            cum_list_self.sort()

        if n_new:
            # Diagnostic for this cubic-spline level
            lv_mask = is_predicted.copy()
            for d in level_log:
                pass  # level_log already has prior levels
            # q_bins for points resolved this level = those just added
            # We track by checking which flat_p were set in this level's loop
            # Simple proxy: compute H over all predicted so far minus prior
            prior_pred = sum(d['n_pred'] for d in level_log)
            pred_sorted = np.where(is_predicted)[0]
            this_level_flat = pred_sorted[prior_pred:prior_pred + n_new]
            lv_q    = all_q_bins[this_level_flat]
            lv_H    = entropy_from_bins(lv_q) if len(lv_q) else 0.0
            lv_bits = lv_H * n_new
            level_log.append(dict(
                level=level, method='cubic', n_pts=n_new,
                n_anchor=0, n_pred=n_new,
                H=lv_H, bits=lv_bits,
                pct_ok=100.0*np.mean(np.abs(lv_q)<=1) if len(lv_q) else 100.0,
                cumulative_br=None,
            ))
            print(f'  Level {level}: {n_new:,} pts cubic-spline  '
                  f'H={lv_H:.3f}  {level_log[-1]["pct_ok"]:.1f}%|q|≤1  '
                  f'[{time.perf_counter()-tL:.1f}s]')

    # ── Metrics ───────────────────────────────────────────────────────────
    n_pred     = int(is_predicted.sum())
    H          = entropy_from_bins(all_q_bins[is_predicted]) if n_pred else 0.0
    total_bits = n_anchor_bits + H * n_pred
    bit_rate   = total_bits / N
    cr         = 32.0 / bit_rate if bit_rate > 0 else float('inf')
    psnr       = psnr_db(recon_err, data_range)
    n_anc      = int(is_anchor.sum())
    elapsed    = time.perf_counter() - t0

    print(f'\n  EB={eb}  bit_rate={bit_rate:.3f}  CR={cr:.2f}×  PSNR={psnr:.2f} dB  '
          f'anchors={n_anc}  H={H:.3f}  [{elapsed:.1f}s]')

    # ── Per-level bit breakdown diagnostic ────────────────────────────────
    print_level_breakdown(level_log, n_anchor_bits, N, eb)

    return dict(eb=eb, bit_rate=bit_rate, cr=cr, psnr=psnr, H=H,
                n_anchor=n_anc, n_predicted=n_pred, elapsed=elapsed,
                level_log=level_log)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_rd(hybrid_results: list,
            sz3_interp_results: list,
            qoz_results_by_level: dict) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))

    # Hybrid GP
    brs   = [r['bit_rate'] for r in hybrid_results]
    psnrs = [r['psnr']     for r in hybrid_results]
    ax.plot(brs, psnrs, 'o-', color='steelblue', lw=2, ms=6,
            label=f'Hybrid GP (GP_LEVELS={GP_LEVELS}, α={ANCHOR_FRAC})')
    for r in hybrid_results:
        ax.annotate(f'EB={r["eb"]}', (r['bit_rate'], r['psnr']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)

    # SZ3 Interp
    interp_brs   = [r['bit_rate'] for r in sz3_interp_results if r]
    interp_psnrs = [r['psnr']     for r in sz3_interp_results if r]
    interp_ebs   = [r['eb']       for r in sz3_interp_results if r]
    ax.plot(interp_brs, interp_psnrs, 's--', color='black',
            lw=1.5, ms=7, zorder=5, label='SZ3-Interp')
    for br, ps, e in zip(interp_brs, interp_psnrs, interp_ebs):
        ax.annotate(f'EB={e}', (br, ps), textcoords='offset points',
                    xytext=(4, -9), fontsize=7, color='black')

    # QoZ levels 1–4
    qoz_styles = {
        1: ('P--', 'mediumvioletred'),
        2: ('D--', 'darkorchid'),
        3: ('^--', 'royalblue'),
        4: ('v--', 'darkorange'),
    }
    for level, (marker, color) in qoz_styles.items():
        results = qoz_results_by_level.get(level, [])
        brs_q   = [r['bit_rate'] for r in results if r]
        psnrs_q = [r['psnr']     for r in results if r]
        ebs_q   = [r['eb']       for r in results if r]
        if brs_q:
            ax.plot(brs_q, psnrs_q, marker, color=color,
                    lw=1.5, ms=7, zorder=5, label=f'QoZ-{level}')
            for br, ps, e in zip(brs_q, psnrs_q, ebs_q):
                ax.annotate(f'EB={e}', (br, ps), textcoords='offset points',
                            xytext=(4, -9), fontsize=7, color=color)

    ax.set_xlabel('Bit rate (bits / value)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Hurricane ISABEL W — Hybrid GP vs Baselines')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'rd_hybrid_gp.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'  Saved {out}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with _Tee(OUT_DIR / 'run_log_hybrid_gp.txt'):
        _main()


def _main():
    print('═'*60)
    print('Hurricane ISABEL W  —  Hybrid GP vs SZ3 Baselines')
    print(f'  GP_LEVELS={GP_LEVELS}  ANCHOR_FRAC={ANCHOR_FRAC}  '
          f'INIT_STRIDE={INIT_STRIDE}')
    print('═'*60)

    print('\nLoading data…')
    vol = load_data()
    print(f'  shape={vol.shape}  min={vol.min():.3f}  max={vol.max():.3f}')

    print('\nRunning SZ3-Interp baseline…')
    sz3_interp_results = [run_sz3_interp(vol, eb) for eb in EB_VALUES]

    qoz_results_by_level = {}
    for level in [1, 2, 3, 4]:
        print(f'\nRunning QoZ-{level} baseline…')
        qoz_results_by_level[level] = [run_qoz(vol, eb, level=level) for eb in EB_VALUES]

    hybrid_results = []
    for eb in EB_VALUES:
        print(f'\n{"═"*60}')
        print(f'Hybrid GP  EB={eb}')
        print('═'*60)
        result = run_hybrid_gp(vol, eb)
        hybrid_results.append(result)

    print('\nPlotting RD curves…')
    plot_rd(hybrid_results, sz3_interp_results, qoz_results_by_level)
    print(f'\nDone.  Log → {OUT_DIR}/run_log_hybrid_gp.txt')


if __name__ == '__main__':
    main()
