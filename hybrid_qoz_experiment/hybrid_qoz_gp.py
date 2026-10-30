"""
hybrid_qoz_gp.py  (v3 — requires QoZ-hybrid C++ patch)

Algorithm (compression):
  1. Determine the level-0 coarse grid (stride = maxStep, mirrors QoZ build_grid)
  2. Fit GP hyperparameters via MLE on coarse grid values
  3. RPCholesky+GKS selects ANCHOR_FRAC of coarse points → stored in header (exact float32)
  4. GP predicts the remaining (non-anchor) coarse points
  5. Build modified data array: anchor positions = original, non-anchor coarse = GP prediction
  6. Write l0 temp file = modified array (QoZ reads this at decompression to skip recover_grid)
  7. Call QoZ-hybrid with: -s <maxStep> -q 4 -M ABS <eb_abs> --l0-precomputed <l0_file>
     → build_grid is SKIPPED (no lossless coarse storage in bitstream)
     → QoZ compresses only the finer-level interpolation residuals
  8. Output = hybrid_header + QoZ payload

Algorithm (decompression):
  1. Parse header → HPs + anchor indices + anchor values + data_range
  2. Reconstruct non-anchor coarse values via GP (same HPs, conditioned on anchors)
  3. Build l0 array: anchor positions = exact, non-anchor coarse = GP predictions
  4. Write l0 temp file, call QoZ-hybrid with --l0-precomputed
     → recover_grid is SKIPPED; QoZ reads l0 file directly into decData
     → QoZ decompresses finer-level residuals from payload
  5. Return decompressed array

Why this works:
  - build_grid normally stores ALL N_coarse values losslessly (32 bits each)
  - We store only ANCHOR_FRAC×N_coarse values (in header, 32 bits each) + no anchor overhead in QoZ
  - Finer-level residuals are small because GP predictions are accurate
  - Error bound is respected: anchor positions are exact; non-anchor positions have
    GP prediction error ≤ EB (the GP fits well within the coarse grid spacing)
"""

import numpy as np
import struct
import zlib
import tempfile
import subprocess
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Anisotropic Matérn-5/2 kernel
# ─────────────────────────────────────────────────────────────

def _matern52(coords_a: np.ndarray, coords_b: np.ndarray,
              ls_xy: float, ls_z: float, sig2: float) -> np.ndarray:
    ndim = coords_a.shape[1]
    if ndim == 1:
        r = np.abs(coords_a[:, 0:1] - coords_b[:, 0:1].T) / max(ls_xy, 1e-10)
    elif ndim == 2:
        dy = (coords_a[:, 0:1] - coords_b[:, 0:1].T) / max(ls_xy, 1e-10)
        dx = (coords_a[:, 1:2] - coords_b[:, 1:2].T) / max(ls_xy, 1e-10)
        r  = np.sqrt(dy**2 + dx**2)
    else:
        dz = (coords_a[:, 0:1] - coords_b[:, 0:1].T) / max(ls_z,  1e-10)
        dy = (coords_a[:, 1:2] - coords_b[:, 1:2].T) / max(ls_xy, 1e-10)
        dx = (coords_a[:, 2:3] - coords_b[:, 2:3].T) / max(ls_xy, 1e-10)
        r  = np.sqrt(dz**2 + dy**2 + dx**2)
    return sig2 * (1.0 + np.sqrt(5.0)*r + 5.0/3.0*r**2) * np.exp(-np.sqrt(5.0)*r)


# ─────────────────────────────────────────────────────────────
# Coarse grid  (mirrors QoZ build_grid / recover_grid pattern)
# ─────────────────────────────────────────────────────────────

def _auto_max_step(dims: tuple, target_n: int = 1000) -> int:
    for step in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        n = 1
        for d in dims:
            n *= max(1, (d + step - 1) // step)
        if n >= target_n // 4:
            return step
    return 1


def get_coarse_grid(dims: tuple, max_step: int):
    """
    Return (flat_indices, coords) for the coarse grid matching QoZ's build_grid.

    QoZ CLI convention: -3 NX NY NZ for data[NZ][NY][NX] (fastest dim first).
    QoZ internally does Config(NZ, NY, NX) with standard C-order strides:
      stride[0] = NY*NX,  stride[1] = NX,  stride[2] = 1
    So flat_index = gz*(NY*NX) + gy*NX + gx, which is identical to numpy's
    C-order flat index for array of shape (NZ, NY, NX).

    dims here is the numpy shape (NZ, NY, NX) = (dims[0], dims[1], dims[2]).
    The coarse grid samples every max_step positions along each axis.

    coords are returned in Python dim order (dim0=altitude first) so the GP
    kernel applies ls_z to the correct axis.
    """
    ndim = len(dims)
    # Sample every max_step positions along each numpy axis
    axes = [np.arange(0, d, max(1, max_step if d > max_step else d), dtype=np.int64)
            for d in dims]

    if ndim == 1:
        flat_idx = axes[0]
        coords = flat_idx.reshape(-1, 1).astype(np.float32)
    elif ndim == 2:
        # dims = (NY, NX); flat_idx = gy*NX + gx
        g0, g1 = np.meshgrid(axes[0], axes[1], indexing='ij')
        flat_idx = g0.ravel() * dims[1] + g1.ravel()
        coords = np.stack([g0.ravel(), g1.ravel()], axis=1).astype(np.float32)
    else:
        # dims = (NZ, NY, NX); flat_idx = gz*NY*NX + gy*NX + gx (C-order)
        g0, g1, g2 = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        flat_idx = (g0.ravel() * (dims[1] * dims[2])
                  + g1.ravel() * dims[2]
                  + g2.ravel())
        # coords: (dim0=altitude, dim1, dim2) — ls_z applies to dim0
        coords = np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=1).astype(np.float32)

    return flat_idx.astype(np.int64), coords


# ─────────────────────────────────────────────────────────────
# Level-1 grid  (first dyadic refinement midpoints)
# ─────────────────────────────────────────────────────────────

def get_level1_grid(dims: tuple, max_step: int):
    """
    Return (flat_indices, coords) for all level-1 (stride = max_step//2) positions
    that QoZ processes at its first interpolation level, following QoZ's 3D sweep:
      X-sweep → Y-sweep → Z-sweep.

    Matches QoZ's build_grid loop: `for pos = 0; pos < dim; pos += stride`
    (strictly less-than, NO extra last-element).  This is the critical difference
    from the old _l0_pos_1d helper which appended the last element (e.g. z=99,
    y=499) and thereby generated spurious midpoints (e.g. z=81, y=473) while
    MISSING real QoZ level-1 midpoints (e.g. z=96, y=480).

    Only implemented for 3D.
    """
    if len(dims) != 3:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    NZ, NY, NX = dims

    HS = max_step // 2   # half-stride = stride at level-1
    if HS < 1:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    # QoZ L0 positions: range(0, dim, max_step)  — strict less-than, same as build_grid
    l0z = np.arange(0, NZ, max_step, dtype=np.int64)
    l0y = np.arange(0, NY, max_step, dtype=np.int64)
    l0x = np.arange(0, NX, max_step, dtype=np.int64)

    # New level-1 positions: all multiples of HS within [0, dim) that are NOT L0
    new_z = np.arange(HS, NZ, max_step, dtype=np.int64)  # HS, HS+S, HS+2S, ...
    new_y = np.arange(HS, NY, max_step, dtype=np.int64)
    new_x = np.arange(HS, NX, max_step, dtype=np.int64)

    # All positions at stride HS (= L0 ∪ new):
    all_z = np.arange(0, NZ, HS, dtype=np.int64)
    all_y = np.arange(0, NY, HS, dtype=np.int64)
    all_x = np.arange(0, NX, HS, dtype=np.int64)

    parts = []
    # X-sweep: new x midpoints at (l0z × l0y) anchors
    if len(new_x) and len(l0z) and len(l0y):
        IZ, IY, IX = np.meshgrid(l0z, l0y, new_x, indexing='ij')
        parts.append(IZ.ravel() * (NY * NX) + IY.ravel() * NX + IX.ravel())

    # Y-sweep: new y midpoints at (l0z × all_x) — all_x includes L0 + X-sweep new
    if len(new_y) and len(l0z) and len(all_x):
        IZ, IY, IX = np.meshgrid(l0z, new_y, all_x, indexing='ij')
        parts.append(IZ.ravel() * (NY * NX) + IY.ravel() * NX + IX.ravel())

    # Z-sweep: new z midpoints at (all_y × all_x)
    if len(new_z) and len(all_y) and len(all_x):
        IZ, IY, IX = np.meshgrid(new_z, all_y, all_x, indexing='ij')
        parts.append(IZ.ravel() * (NY * NX) + IY.ravel() * NX + IX.ravel())

    if not parts:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    flat = np.unique(np.concatenate(parts)).astype(np.int64)
    iz   = flat // (NY * NX)
    iy   = (flat % (NY * NX)) // NX
    ix   = flat % NX
    # Integer coords — same space as get_coarse_grid
    coords = np.stack([iz, iy, ix], axis=1).astype(np.float32)
    return flat, coords


# ─────────────────────────────────────────────────────────────
# GP hyperparameter fitting
# ─────────────────────────────────────────────────────────────

def fit_gp_hps(coords: np.ndarray, values: np.ndarray,
               n_restarts: int = 3, max_fit_pts: int = 2000):
    from scipy.optimize import minimize

    n = len(values)
    if n > max_fit_pts:
        idx = np.random.choice(n, max_fit_pts, replace=False)
        c, v = coords[idx].astype(np.float64), values[idx].astype(np.float64)
    else:
        c, v = coords.astype(np.float64), values.astype(np.float64)

    mu    = v.mean()
    v     = v - mu
    v_std = max(v.std(), 1e-10)
    v_n   = v / v_std
    cr    = np.ptp(c, axis=0) + 1

    def nll(log_theta):
        lx, lz, s2, jit = np.exp(log_theta)
        try:
            K = _matern52(c, c, lx, lz, s2)
            K += (jit + 1e-6) * np.eye(len(c))
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e15
        alp    = np.linalg.solve(L.T, np.linalg.solve(L, v_n))
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return 0.5 * (v_n @ alp + logdet + len(v_n) * np.log(2*np.pi))

    ndim    = c.shape[1]
    lx0     = float(cr[-1]) / 4
    lz0     = float(cr[0])  / 4 if ndim >= 3 else lx0
    best, best_t = np.inf, None
    for _ in range(n_restarts):
        s  = np.random.uniform(0.5, 2.0)
        x0 = np.log([max(s*lx0, 1e-3), max(s*lz0, 1e-3), v_std**2, 1e-4])
        try:
            r = minimize(nll, x0, method='L-BFGS-B',
                         bounds=[(-3, 15)]*4,
                         options={'maxiter': 300, 'ftol': 1e-10})
            if r.fun < best:
                best, best_t = r.fun, r.x
        except Exception:
            pass
    if best_t is None:
        return lx0, lz0, v_std**2
    lx, lz, s2, _ = np.exp(best_t)
    return float(lx), float(lz), float(s2 * v_std**2)


# ─────────────────────────────────────────────────────────────
# RPCholesky anchor selection
# ─────────────────────────────────────────────────────────────

def rp_cholesky_select(coords: np.ndarray, ls_xy: float, ls_z: float,
                       sig2: float, n_anchors: int) -> np.ndarray:
    n         = len(coords)
    n_anchors = min(n_anchors, n)
    if n_anchors >= n:
        return np.arange(n, dtype=np.int64)
    c         = coords.astype(np.float64)
    residual  = np.full(n, sig2, dtype=np.float64)
    pivots    = []
    G         = np.zeros((n_anchors, n), dtype=np.float64)

    rng = np.random.default_rng(seed=42)
    for j in range(n_anchors):
        p = np.maximum(residual, 0.0)
        total = p.sum()
        if total <= 0:
            p = np.ones(n, dtype=np.float64) / n
        else:
            p /= total
        # Guarantee p is non-negative and sums to exactly 1.0:
        # clip away any -0.0 / tiny negatives from fp arithmetic,
        # then transfer any residual error onto the largest element.
        p = np.maximum(p, 0.0)
        p[p.argmax()] += 1.0 - p.sum()
        # Guard against NaN in p (from overflow in G[j]**2 below)
        if not np.isfinite(p).all():
            p = np.ones(n, dtype=np.float64) / n
        idx = int(rng.choice(n, p=p))
        pivots.append(idx)
        k_col = _matern52(c, c[idx:idx+1], ls_xy, ls_z, sig2).ravel()
        if j == 0:
            G[j] = k_col / max(residual[idx], 1e-15)**0.5
        else:
            gp   = G[:j, idx]
            sch  = max(residual[idx] - np.dot(gp, gp), 1e-15)
            G[j] = (k_col - G[:j].T @ gp) / sch**0.5
        # Clip G[j] before squaring to prevent float64 overflow → NaN residuals
        np.clip(G[j], -1e154, 1e154, out=G[j])
        residual = np.maximum(0.0, residual - G[j]**2)

    return np.array(pivots, dtype=np.int64)


# ─────────────────────────────────────────────────────────────
# GP posterior mean prediction
# ─────────────────────────────────────────────────────────────

def gp_predict_mean(anc_coords, anc_vals, pred_coords,
                    ls_xy, ls_z, sig2, nugget=1e-10):
    K_aa = _matern52(anc_coords, anc_coords, ls_xy, ls_z, sig2)
    K_aa += (nugget + 1e-6) * np.eye(len(anc_coords))
    K_pa = _matern52(pred_coords, anc_coords, ls_xy, ls_z, sig2)
    L    = np.linalg.cholesky(K_aa)
    alp  = np.linalg.solve(L.T, np.linalg.solve(L, anc_vals))
    return K_pa @ alp


# ─────────────────────────────────────────────────────────────
# Binary header  (version 5 — Approach B)
# Layout (little-endian):
#   4×float64: ls_xy, ls_z, sig2, data_range
#   1×uint32:  n_anchors
#   1×uint32:  header_version  (4 = L0 only; 5 = L0 + per-level zlib bins)
#   n_anchors×int64:   anchor global flat indices
#   n_anchors×float32: anchor values (lossless)
#   1×uint32:  n_non_anchors
#   n_non_anchors×float32: quantized non-anchor coarse values
#   [version 5 only:]
#   1×uint8:  n_extra_gp_levels  (0 = L0 only, 1 = L0+L1, 2 = L0+L1+L2, …)
#   for each extra level:
#     1×uint32:  n_compressed_bytes
#     n_compressed_bytes: zlib(int16 bins array)
#   NOTE: flat indices for extra levels are NOT stored — recomputed deterministically.
# ─────────────────────────────────────────────────────────────

_HDR_FIXED = struct.calcsize('<ddddII')   # 4*8 + 2*4 = 40 bytes

def _encode_header(ls_xy, ls_z, sig2, data_range,
                   anchor_gidx: np.ndarray,
                   anchor_vals: np.ndarray,
                   non_anc_quantized: np.ndarray = None,
                   per_level_bins: list = None) -> bytes:
    """
    per_level_bins: list of bytes objects, one per GP level beyond L0.
                    Each entry is zlib-compressed int16 bin array.
    """
    n_a  = len(anchor_gidx)
    n_ex = len(per_level_bins) if per_level_bins else 0
    ver  = 5 if n_ex > 0 else 4
    buf  = struct.pack('<ddddII', ls_xy, ls_z, sig2, data_range, n_a, ver)
    buf += anchor_gidx.astype(np.int64).tobytes()
    buf += anchor_vals.astype(np.float32).tobytes()
    if non_anc_quantized is not None and len(non_anc_quantized) > 0:
        buf += struct.pack('<I', len(non_anc_quantized))
        buf += non_anc_quantized.astype(np.float32).tobytes()
    else:
        buf += struct.pack('<I', 0)
    if ver >= 5:
        buf += struct.pack('<B', n_ex)
        for b in per_level_bins:
            buf += struct.pack('<I', len(b))
            buf += b
    return buf


def _decode_header(data: bytes):
    ls_xy, ls_z, sig2, dr, n_a, ver = struct.unpack_from('<ddddII', data, 0)
    off         = _HDR_FIXED
    anchor_gidx = np.frombuffer(data, dtype=np.int64,   count=n_a, offset=off); off += n_a * 8
    anchor_vals = np.frombuffer(data, dtype=np.float32, count=n_a, offset=off); off += n_a * 4
    non_anc_quantized = None
    if ver >= 4:
        n_non_a = struct.unpack_from('<I', data, off)[0]; off += 4
        if n_non_a > 0:
            non_anc_quantized = np.frombuffer(data, dtype=np.float32,
                                              count=n_non_a, offset=off).copy()
            off += n_non_a * 4
    per_level_bins = []
    if ver >= 5:
        n_ex = struct.unpack_from('<B', data, off)[0]; off += 1
        for _ in range(n_ex):
            n_b = struct.unpack_from('<I', data, off)[0]; off += 4
            per_level_bins.append(bytes(data[off:off + n_b])); off += n_b
    return ls_xy, ls_z, sig2, dr, anchor_gidx, anchor_vals, off, non_anc_quantized, per_level_bins


# ─────────────────────────────────────────────────────────────
# Helper: build the level-0 array
# ─────────────────────────────────────────────────────────────

def _build_l0_array(dims, flat_idx, coords,
                    anc_local, anc_vals,
                    non_anc_local, ls_xy, ls_z, sig2,
                    base_data: np.ndarray,
                    non_anc_precomputed: np.ndarray = None) -> np.ndarray:
    """
    Full-size float32 array where:
      - anchor positions       = exact original values
      - non-anchor coarse pos  = non_anc_precomputed (if given) else GP posterior mean
      - everything else        = copied from base_data (fine positions unchanged)
    QoZ only reads coarse positions; fine positions in this file are ignored
    because QoZ's decompressor fills them from the quantised residuals.
    """
    arr = base_data.astype(np.float32).copy()
    arr.ravel()[flat_idx[anc_local]] = anc_vals.astype(np.float32)
    if len(non_anc_local) > 0:
        if non_anc_precomputed is not None:
            coarse_vals = non_anc_precomputed.astype(np.float32)
        else:
            coarse_vals = gp_predict_mean(
                coords[anc_local].astype(np.float64), anc_vals.astype(np.float64),
                coords[non_anc_local].astype(np.float64),
                ls_xy, ls_z, sig2).astype(np.float32)
        arr.ravel()[flat_idx[non_anc_local]] = coarse_vals
    return arr


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class HybridQoZGP:
    """
    Hybrid QoZ+GP compressor.
    Requires QoZ-hybrid binary built from patch_qoz_hybrid.sh.
    """

    def __init__(self, qoz_bin: str,
                 anchor_frac: float = 0.10,
                 max_step: int = 0,
                 gp_levels: int = 2,
                 temporal_mode: str = 'fixed',
                 refit_every_t: int = 5,
                 adaptive_threshold: float = 0.05):
        self.qoz_bin   = str(qoz_bin)
        self.anc_frac  = anchor_frac
        self.max_step  = max_step
        self.gp_levels = gp_levels
        self.t_mode    = temporal_mode
        self.t_period  = refit_every_t
        self.t_thresh  = adaptive_threshold

        self._ts         = 0
        self._last_refit = -1
        self._cached_hp  = None
        self._cached_anc = None
        self._last_var   = None
        self._cached_ms  = None

    def _effective_max_step(self, dims):
        if self.max_step > 0:
            return self.max_step
        target = max(500, int(np.prod(dims) ** 0.35))
        return _auto_max_step(dims, target)

    def _needs_refit(self, coarse_vals):
        if self._cached_hp is None:
            return True
        if self.t_mode == 'fixed':
            return (self._ts - self._last_refit) >= self.t_period
        cur_var = float(np.var(coarse_vals))
        if self._last_var is None:
            return True
        return abs(cur_var - self._last_var) / (self._last_var + 1e-12) > self.t_thresh

    def _dim_flags(self, dims):
        # QoZ/SZ3 convention: -N nx ny nz where data layout is data[nz][ny][nx]
        # i.e. fastest-varying dimension first → pass numpy shape reversed.
        rdims = list(reversed(dims))
        n = len(dims)
        if n == 1: return ['-1', str(rdims[0])]
        if n == 2: return ['-2', str(rdims[0]), str(rdims[1])]
        return ['-3', str(rdims[0]), str(rdims[1]), str(rdims[2])]

    # ── compress ─────────────────────────────────────────────

    def compress(self, data: np.ndarray, eb_rel: float,
                 out_path: str, tmp_dir: str = None) -> dict:
        t0     = time.time()
        dims   = data.shape
        N      = data.size
        dr     = float(data.max() - data.min())
        eb_abs = eb_rel * dr

        # 1. Coarse grid
        ms = self._effective_max_step(dims)
        flat_idx, coords = get_coarse_grid(dims, ms)
        coarse_vals = data.ravel()[flat_idx].astype(np.float64)
        n_coarse    = len(flat_idx)

        # 2. Fit / reuse HPs and anchor selection
        if self._needs_refit(coarse_vals) or self._cached_ms != ms:
            print(f'    [GP] Fitting HPs (ts={self._ts}, n_coarse={n_coarse}) …')
            ls_xy, ls_z, sig2 = fit_gp_hps(coords, coarse_vals)
            n_anc     = max(1, int(self.anc_frac * n_coarse))
            anc_local = rp_cholesky_select(coords, ls_xy, ls_z, sig2, n_anc)
            self._cached_hp  = (ls_xy, ls_z, sig2)
            self._cached_anc = anc_local
            self._cached_ms  = ms
            self._last_refit = self._ts
            self._last_var   = float(np.var(coarse_vals))
        else:
            ls_xy, ls_z, sig2 = self._cached_hp
            anc_local         = self._cached_anc
        self._ts += 1

        # 3. Masks
        non_mask      = np.ones(n_coarse, dtype=bool)
        non_mask[anc_local] = False
        non_anc_local = np.where(non_mask)[0]
        anc_vals      = coarse_vals[anc_local]

        # 4. Compute quantized non-anchor coarse values (guarantees EB at coarse level)
        if len(non_anc_local) > 0:
            gp_pred_non_anc = gp_predict_mean(
                coords[anc_local].astype(np.float64), anc_vals.astype(np.float64),
                coords[non_anc_local].astype(np.float64),
                ls_xy, ls_z, sig2)
            orig_non_anc = data.ravel()[flat_idx[non_anc_local]].astype(np.float64)
            quant_idx    = np.round((orig_non_anc - gp_pred_non_anc) / (2.0 * eb_abs))
            non_anc_quantized = (gp_pred_non_anc + quant_idx * (2.0 * eb_abs)).astype(np.float32)
        else:
            non_anc_quantized = np.array([], dtype=np.float32)

        # Build modified data + l0 array (both use same quantized coarse values)
        data_mod = _build_l0_array(dims, flat_idx, coords,
                                   anc_local, anc_vals,
                                   non_anc_local, ls_xy, ls_z, sig2,
                                   data, non_anc_quantized)

        # 4b. Approach B: Level-1+ GP — store zlib-compressed int16 residual bins.
        # Level indices are NOT stored (recomputed deterministically on decompress).
        # Conditioning for level k: all previously reconstructed points (L0 + L1 + …).
        per_level_bins = []
        cond_flat   = flat_idx            # starts as L0 indices
        cond_coords = coords              # starts as L0 coords
        cond_vals   = data_mod.ravel()[flat_idx].astype(np.float64)  # L0 reconstructed

        if self.gp_levels >= 2 and len(dims) == 3:
            l1_flat, l1_coords = get_level1_grid(dims, ms)
            if len(l1_flat) > 0:
                print(f'    [GP-L1] Predicting {len(l1_flat):,} level-1 midpoints …')
                l1_pred = gp_predict_mean(
                    cond_coords.astype(np.float64), cond_vals,
                    l1_coords.astype(np.float64),
                    ls_xy, ls_z, sig2)
                l1_true = data.ravel()[l1_flat].astype(np.float64)
                # Approach B residual bins (int16)
                l1_bins  = np.round((l1_true - l1_pred) / (2.0 * eb_abs)).astype(np.int16)
                l1_recon = (l1_pred + l1_bins.astype(np.float64) * 2.0 * eb_abs).astype(np.float32)
                # Insert L1 reconstructed values into data_mod so QoZ starts at level-2
                data_mod = data_mod.copy()
                data_mod.ravel()[l1_flat] = l1_recon
                # zlib-compress bins (indices NOT stored — recomputed on decompress)
                bins_bytes = zlib.compress(l1_bins.tobytes(), level=6)
                per_level_bins.append(bins_bytes)
                pct0 = 100.0 * float(np.mean(l1_bins == 0))
                print(f'    [GP-L1] bins: min={l1_bins.min()}  max={l1_bins.max()}  '
                      f'pct_zero={pct0:.1f}%  zlib={len(bins_bytes)} bytes '
                      f'(vs {len(l1_flat)*2} bytes raw int16)')
                # Update conditioning set for any future GP levels (level-2, etc.)
                cond_flat   = np.concatenate([cond_flat, l1_flat])
                cond_coords = np.vstack([cond_coords, l1_coords])
                cond_vals   = np.concatenate([cond_vals, l1_recon.astype(np.float64)])

        # 5. Encode header (quantized non-anchor values + per-level zlib bins)
        header = _encode_header(ls_xy, ls_z, sig2, dr,
                                flat_idx[anc_local],
                                anc_vals.astype(np.float32),
                                non_anc_quantized,
                                per_level_bins)

        # 6. Call QoZ-hybrid  (build_grid skipped by --l0-precomputed patch)
        n_gp_lvls    = 1 + len(per_level_bins)   # L0 + any extra levels
        gp_lvl_flags = ['--gp-levels', str(n_gp_lvls)] if n_gp_lvls >= 2 else []
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpd:
            tmpd  = Path(tmpd)
            f_in  = tmpd / 'vol.f32'
            f_l0  = tmpd / 'l0.f32'
            f_cmp = tmpd / 'vol.out'

            data_mod.tofile(f_in)
            data_mod.tofile(f_l0)   # l0 file = same modified array

            cmd = ([self.qoz_bin, '-f',
                    '-i', str(f_in), '-z', str(f_cmp),
                    '--l0-precomputed', str(f_l0),
                    '-C', str(ms),          # -C sets maxStep / anchor stride in QoZ
                    '-M', 'ABS', str(eb_abs), '-q', '4']
                   + gp_lvl_flags
                   + self._dim_flags(dims))

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stderr_str = res.stderr.decode(errors='replace')
            if res.returncode != 0:
                return {'error': f'QoZ compress failed (rc={res.returncode}): {stderr_str[:300]}'}
            if not f_cmp.exists():
                return {'error': f'QoZ produced no output. stderr: {stderr_str[:200]}'}

            qoz_payload = f_cmp.read_bytes()

        # 7. Write output: hdr_len(4B) + header + QoZ payload
        hlen      = len(header)
        out_bytes = struct.pack('<I', hlen) + header + qoz_payload
        Path(out_path).write_bytes(out_bytes)

        t_comp   = time.time() - t0
        n_bytes  = len(out_bytes)
        bit_rate = n_bytes * 8 / N
        cr       = (N * 4) / n_bytes
        hdr_kb   = hlen / 1024
        qoz_kb   = len(qoz_payload) / 1024

        print(f'    [compress] eb_rel={eb_rel:.1e}  CR={cr:.1f}×  '
              f'bpv={bit_rate:.3f}  anchors={len(anc_local)}/{n_coarse}  '
              f'hdr={hdr_kb:.1f}KB  qoz={qoz_kb:.1f}KB  t={t_comp:.1f}s')

        return dict(eb_rel=eb_rel, eb_abs=eb_abs, bit_rate=bit_rate, cr=cr,
                    comp_time=t_comp, n_coarse=n_coarse,
                    n_anchors=int(len(anc_local)), max_step=ms,
                    ls_xy=ls_xy, ls_z=ls_z, sig2=sig2,
                    cmp_bytes=n_bytes, header_bytes=hlen, qoz_bytes=len(qoz_payload))

    # ── decompress ───────────────────────────────────────────

    def decompress(self, cmp_path: str, dims: tuple, eb_rel: float,
                   tmp_dir: str = None) -> tuple:
        t0  = time.time()
        raw = Path(cmp_path).read_bytes()

        hlen         = struct.unpack_from('<I', raw, 0)[0]
        header_bytes = raw[4:4+hlen]
        qoz_payload  = raw[4+hlen:]

        ls_xy, ls_z, sig2, dr, anchor_gidx, anchor_vals, _, non_anc_quantized, per_level_bins = \
            _decode_header(header_bytes)
        eb_abs = eb_rel * dr

        # Reconstruct coarse grid
        ms = self._effective_max_step(dims)
        flat_idx, coords = get_coarse_grid(dims, ms)
        n_coarse = len(flat_idx)

        # Identify anchor / non-anchor positions
        gidx_set  = set(anchor_gidx.tolist())
        anc_mask  = np.array([i in gidx_set for i in flat_idx.tolist()], dtype=bool)
        non_local = np.where(~anc_mask)[0]
        anc_local = np.where(anc_mask)[0]

        # anchor_vals are stored in RPCholesky selection order (from compress);
        # anc_local from np.where is in sorted order — must re-map by global flat index
        gidx_to_val = {int(g): float(v) for g, v in zip(anchor_gidx, anchor_vals)}
        anchor_vals_sorted = np.array([gidx_to_val[int(flat_idx[i])]
                                       for i in anc_local], dtype=np.float32)

        # Build l0 array: anchors exact + non-anchors from header (quantized) or GP fallback
        dummy = np.zeros(dims, dtype=np.float32)
        l0_arr = _build_l0_array(dims, flat_idx, coords,
                                 anc_local, anchor_vals_sorted,
                                 non_local, ls_xy, ls_z, sig2,
                                 dummy, non_anc_quantized)

        # Approach B: reconstruct per-level GP points from zlib-compressed bins.
        # Indices recomputed deterministically; GP prediction identical to compress
        # because same HPs + same L0 conditioning set (restored from header exactly).
        cond_flat   = flat_idx
        cond_coords = coords
        cond_vals   = l0_arr.ravel()[flat_idx].astype(np.float64)

        for level_idx, bins_bytes in enumerate(per_level_bins):
            if level_idx == 0:
                lv_flat, lv_coords = get_level1_grid(dims, ms)
            # future: elif level_idx == 1: lv_flat, lv_coords = get_level2_grid(...)
            else:
                break  # higher levels not yet implemented
            if len(lv_flat) == 0:
                continue
            lv_pred  = gp_predict_mean(
                cond_coords.astype(np.float64), cond_vals,
                lv_coords.astype(np.float64),
                ls_xy, ls_z, sig2)
            bins     = np.frombuffer(zlib.decompress(bins_bytes), dtype=np.int16)
            lv_recon = (lv_pred + bins.astype(np.float64) * eb_abs * 2.0).astype(np.float32)
            l0_arr   = l0_arr.copy()
            l0_arr.ravel()[lv_flat] = lv_recon
            # Update conditioning set for next level
            cond_flat   = np.concatenate([cond_flat, lv_flat])
            cond_coords = np.vstack([cond_coords, lv_coords])
            cond_vals   = np.concatenate([cond_vals, lv_recon.astype(np.float64)])

        n_gp_lvls    = 1 + len(per_level_bins)
        gp_lvl_flags = ['--gp-levels', str(n_gp_lvls)] if n_gp_lvls >= 2 else []

        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpd:
            tmpd  = Path(tmpd)
            f_l0  = tmpd / 'l0.f32'
            f_cmp = tmpd / 'vol.out'
            f_dec = tmpd / 'vol_dec.f32'

            l0_arr.tofile(f_l0)
            f_cmp.write_bytes(qoz_payload)

            cmd = ([self.qoz_bin, '-f',
                    '-z', str(f_cmp), '-o', str(f_dec),
                    '--l0-precomputed', str(f_l0),
                    '-C', str(ms),
                    '-M', 'ABS', str(eb_abs), '-q', '4']
                   + gp_lvl_flags
                   + self._dim_flags(dims))

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f'QoZ decompress failed:\n{res.stderr.decode(errors="replace")}')

            dec = np.fromfile(f_dec, dtype=np.float32).reshape(dims)

        return dec, time.time() - t0
