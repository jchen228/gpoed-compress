"""
rate_distortion_comparison.py  v2
==================================
Fair SZ2-style pipeline comparison for T-DEIM and MultiGP.

Pipeline (T-DEIM / MultiGP):
  1. Predict field from k sensor observations
  2. residual = original − prediction
  3. SZ-style quantize residuals  (bin_width = 2*abs_bound / 65536)
  4. Zstd-compress bin indices + raw float32 outliers

  Sensor values also quantized with the same abs_bound (same treatment
  as SZ's predictable data points), then dequantized before use in
  the prediction step (accurate decompressor simulation).

Model stored losslessly:
  Phi / K_Xs  → float16 + Zstd
  L_kd        → float32 + Zstd  (Cholesky, needs more precision)
  mean / std  → float32 + Zstd
  sensors     → int32  + Zstd

Only the prediction step differs between SZ2 and T-DEIM/MultiGP.
All CR computations include EVERYTHING needed to decompress.

MultiGP fields: d=2 (CLOUDf + QVAPORf)
  Water vapor (QVAPORf) is the direct physical precursor to cloud liquid water
  (CLOUDf): they co-exist in moist/convective regions of the hurricane, giving
  a high off-diagonal B and letting sensors observing one field constrain both.
  CR = (2 × n_3D_full × 4) / compressed_size  (joint compression of 2 fields).
  PSNR/SSIM reported as mean across both fields.

Note on spatial resolution
--------------------------
T-DEIM and MultiGP operate on downsampled data (ds=10 → 50×50/level).
CR denominator = full-resolution file(s).  PSNR/SSIM on 50×50 downsampled slice.
SZ2/ZFP operate on the full 500×500 per level.
"""

import sys, struct, time, importlib.util, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libpressio
from skimage.metrics import structural_similarity
from scipy.linalg import qr as scipy_qr, cho_factor, cho_solve

# ── Config ────────────────────────────────────────────────────────────────────
ARGONNE     = Path(__file__).parent
DATA_DIR    = ARGONNE / "100x500x500"   # full Isabel dataset
DATA_PATH   = DATA_DIR / "Uf48.bin.f32"    # cloud liquid water
DATA_PATH2  = DATA_DIR / "Vf48.bin.f32"   # water vapor (field 2, for MultiGP d=2)
SHAPE       = (100, 500, 500)
LEVEL       = 50          # z-slice used for 2D metrics
DS          = 3           # spatial downsample for all methods except T-DEIM → ~167×167/level
DS_TDEIM    = 10          # T-DEIM builds an (n_3D_ds × k) full 3D basis; pivoted QR on that
                          # matrix is O(k² × n_3D_ds) — infeasibly slow at DS=3 (2.8M pts).
                          # DS=10 → 50×50=2500 pts, QR runs in seconds.
FIELD_TAG   = DATA_PATH.stem.replace(".bin", "").replace(".f32", "")  # e.g. "QRAINf48"

# DEIM-2D, Kriging-2D, MultiGP, SZ2, ZFP all use DS=3 (167×167/level).
# T-DEIM uses DS_TDEIM=10 (50×50/level) — a fair trade-off noted in plots/CSV.
# CR denominators are computed per method using their own effective n_3D_ds.

ABS_BOUNDS       = np.logspace(-4, -0.5, 8)    # shared sweep (controls residual binning)
ZFP_ABS_BOUNDS   = np.logspace(-4, -0.5, 8)   # fixed-accuracy mode, matches SZ2 sweep
TDEIM_K_VALUES   = [1, 2, 5, 10, 25, 50]
DEIM2D_K_VALUES  = [1, 2, 5, 10, 25, 50, 100]   # max useful k = min(n_L, n_2D) = 100
KRIG2D_K_VALUES  = [1, 5, 10, 25, 50, 100, 200, 300, 500]   # GP sensors: no n_L cap
MULTIGP_K_VALUES = [1, 5, 10, 25, 50, 100, 200, 300]

# Quantization bin counts swept only for DEIM-2D residuals.
# Fewer bins → values cluster near 0 → better Zstd ratio.
NUM_BINS_VALUES = [256, 4096, 65536]

# Runtime flags — disable slow methods to speed up iteration
RUN_TDEIM    = True
RUN_MULTIGP  = True
PLOTS_ONLY   = False   # set True to skip compression and re-plot from rd_results.csv

COLORS  = {"SZ2":        "#1f77b4",
           "ZFP":        "#ff7f0e",
           "T-DEIM":     "#2ca02c",
           "DEIM-2D":    "#98df8a",
           "Kriging-2D": "#9467bd",
           "MultiGP":    "#d62728"}
MARKERS = {"SZ2":        "o",
           "ZFP":        "s",
           "T-DEIM":     "^",
           "DEIM-2D":    "v",
           "Kriging-2D": "P",
           "MultiGP":    "D"}

# ── Zstd (zlib fallback) ──────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    _cctx = _zstd.ZstdCompressor(level=3)
    def _compress(b): return _cctx.compress(b)
    COMPRESS_BACKEND = "zstd"
except ImportError:
    import zlib
    def _compress(b): return zlib.compress(b, 6)
    COMPRESS_BACKEND = "zlib"

# ── SZ-style quantization ─────────────────────────────────────────────────────
NUM_BINS = 65536   # default; overridden per-call in T-DEIM / DEIM-2D sweeps

def quantize(arr, abs_bound, num_bins=NUM_BINS):
    """
    bin_width = 2*abs_bound / num_bins
    Returns (bin_indices int16, outlier_pos int32, outlier_vals float32).
    Outliers (|value| > abs_bound) are stored as raw float32, same as SZ.
    """
    bw   = 2.0 * abs_bound / num_bins
    flat = arr.ravel().astype(np.float64)
    raw  = np.round(flat / bw).astype(np.int32)
    half = num_bins // 2
    mask = np.abs(raw) >= half
    bins = np.where(mask, 0, np.clip(raw, -(half - 1), half - 1)).astype(np.int16)
    out_pos  = np.where(mask)[0].astype(np.int32)
    out_vals = flat[mask].astype(np.float32)
    return bins, out_pos, out_vals

def pack_encode(bins, out_pos, out_vals):
    """Estimate compressed size mirroring SZ2: Huffman(bins) + raw outliers.

    SZ2 pipeline: quantize → Huffman-encode bin indices → Zstd → store outliers as raw float32.
    We compute the Huffman size from Shannon entropy (optimal prefix-code lower bound) rather
    than actually encoding, because:
      (a) reconstruction uses bins/out_pos/out_vals directly — no decoding path needed here
      (b) entropy gives the same size as Huffman for large n (difference < n_unique bits)
    Outlier positions are Zstd-compressed; outlier values stored as raw float32 (same as SZ2).

    Returns a dummy bytes object whose len() equals the estimated compressed size.
    """
    flat = bins.ravel()
    if len(flat) == 0:
        huff_bytes = 0
    else:
        _, counts = np.unique(flat, return_counts=True)
        n = len(flat)
        probs = counts / n
        entropy_bps = float(-np.sum(probs * np.log2(probs + 1e-300)))
        huff_bits = int(np.ceil(entropy_bps * n))
        codebook_bytes = len(counts) * 3 + 10   # (sym:i16, codelen:u8) × n_unique + header
        huff_bytes = (huff_bits + 7) // 8 + codebook_bytes

    out_pos_enc  = _compress(out_pos.tobytes()) if len(out_pos) else b''
    out_vals_raw = out_vals.tobytes()   # raw float32, same as SZ2 unpredictables

    return b'\x00' * (huff_bytes + len(out_pos_enc) + len(out_vals_raw))

def dequantize(bins, out_pos, out_vals, abs_bound, orig_shape, num_bins=NUM_BINS):
    """Reconstruct array from quantized representation."""
    bw  = 2.0 * abs_bound / num_bins
    arr = bins.astype(np.float64) * bw
    if len(out_pos):
        arr[out_pos] = out_vals.astype(np.float64)
    return arr.reshape(orig_shape)

# ── Lossless model compression ────────────────────────────────────────────────
def compress_f16(a): return _compress(a.astype(np.float16).tobytes())
def compress_f32(a): return _compress(a.astype(np.float32).tobytes())
def compress_i32(a): return _compress(a.astype(np.int32).tobytes())

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_psnr(orig, recon):
    o, r = orig.astype(np.float64), recon.astype(np.float64)
    dr   = float(o.max() - o.min())
    rmse = float(np.sqrt(np.mean((o - r) ** 2)))
    return 20.0 * np.log10(dr / rmse) if rmse > 0 else float("inf")

def compute_ssim(orig, recon):
    dr = float(orig.max() - orig.min())
    return float(structural_similarity(
        orig.astype(np.float64), recon.astype(np.float64), data_range=dr))

def metrics(a, b):
    return compute_psnr(a, b), compute_ssim(a, b)

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── SZ2 / ZFP via libpressio ─────────────────────────────────────────────────
def run_libpressio(data_3d, compressor_id, config):
    comp = libpressio.PressioCompressor.from_config({
        "compressor_id": compressor_id,
        "early_config": {"pressio:metric": "composite",
                         "composite:plugins": ["size"]},
        "compressor_config": config,
    })
    recon = data_3d.copy()
    t0         = time.perf_counter()
    compressed = comp.encode(data_3d)
    comp_sec   = time.perf_counter() - t0
    t0         = time.perf_counter()
    recon      = comp.decode(compressed, recon)
    decomp_sec = time.perf_counter() - t0
    cr         = comp.get_metrics().get("size:compression_ratio", float("nan"))
    return recon, cr, comp_sec, decomp_sec

# ── T-DEIM + SZ residual ──────────────────────────────────────────────────────
def run_tdeim(data_3d, k_values, abs_bounds=ABS_BOUNDS, ds=DS):
    """
    For each k: build model once (offline).
    For each abs_bound: quantize sensor values → predict → quantize residuals.

    CR = (n_3D_full × 4) / (model_bytes + quantized_sensors + quantized_residuals)

    Model = float16(Phi) + Zstd  +  int32(sensors) + Zstd  +  float32(mean) + Zstd
    """
    tdeim = load_mod(str(ARGONNE / "lp_tdeim_compressor.py"), "lp_tdeim")

    data_ds           = data_3d[:, ::ds, ::ds].astype(np.float64)
    n_L, ny, nz       = data_ds.shape
    n_2D              = ny * nz
    n_3D_ds           = n_L * n_2D            # CR denominator: downsampled data being compressed

    print(f"\n[T-DEIM] Training SVD  ds={ds} → {ny}×{nz}={n_2D}/level ...")
    t0 = time.perf_counter()
    mean_ds  = data_ds.mean(axis=0)
    F        = (data_ds - mean_ds).reshape(n_L, n_2D)
    k_max    = min(max(k_values), n_L)
    Phi, svs = tdeim.build_3d_basis(F, k_max)
    train_sec = time.perf_counter() - t0
    print(f"  SVD done: {train_sec:.1f}s  Phi {Phi.shape}  "
          f"compress backend: {COMPRESS_BACKEND}")

    mean_c    = compress_f32(mean_ds)   # mean compressed once, shared across k
    data_flat = (data_ds - mean_ds).ravel()
    results   = []

    for k in k_values:
        k     = min(k, k_max)
        Phi_k = Phi[:, :k]             # (n_3D_ds, k) — 3D basis

        # Q-DEIM sensor placement in the flat 3D array
        _, _, p = scipy_qr(Phi_k.T, pivoting=True)
        sensors = np.sort(p[:k])       # indices into flat (n_L*ny*nz,)

        # ── Model: float16(Phi) + int32(sensors) + float32(mean), all Zstd ──
        phi_c   = compress_f16(Phi_k)
        sens_c  = compress_i32(sensors)
        model_b = len(phi_c) + len(sens_c) + len(mean_c)

        # Sensor values stored as float16 — same treatment as DEIM-2D/Kriging-2D.
        # Exact centred values at sensor locations.
        y_s_centre = data_flat[sensors].astype(np.float32)   # (k,) centred
        sv_enc     = _compress(y_s_centre.astype(np.float16).tobytes())

        # Reconstruct once per k (prediction doesn't depend on abs_bound)
        recon_flat = tdeim.tdeim_reconstruct(Phi_k, sensors, y_s_centre.astype(np.float64))
        recon_ds   = recon_flat.reshape(n_L, ny, nz) + mean_ds
        resid      = (data_ds - recon_ds).ravel().astype(np.float32)

        k_results = []

        for ab in abs_bounds:
            t0 = time.perf_counter()

            bins_r, op_r, ov_r = quantize(resid, ab, NUM_BINS)
            resid_enc = pack_encode(bins_r, op_r, ov_r)

            comp_sec = time.perf_counter() - t0

            data_b       = len(sv_enc) + len(resid_enc)
            total_b      = model_b + data_b
            cr           = (n_3D_ds * 4) / total_b
            cr_no_model  = (n_3D_ds * 4) / data_b

            resid_rec = dequantize(bins_r, op_r, ov_r, ab, (n_L, ny, nz), NUM_BINS)
            final_ds  = recon_ds + resid_rec
            pv, sv_m  = metrics(data_ds[LEVEL].astype(np.float32),
                                final_ds[LEVEL].astype(np.float32))

            row = {"method": "T-DEIM", "k": k, "abs_bound": ab,
                   "cr": cr, "cr_no_model": cr_no_model,
                   "psnr": pv, "ssim": sv_m,
                   "comp_sec": comp_sec, "decomp_sec": 0.0, "train_sec": train_sec,
                   "compressed_MB": total_b / 1e6, "model_MB": model_b / 1e6,
                   "sv_MB": len(sv_enc) / 1e6, "resid_MB": len(resid_enc) / 1e6,
                   "n_outliers": len(op_r)}
            results.append(row)
            k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"  k={k:3d}  model={model_b/1e3:.1f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR: {min(psnrs):.1f}–{max(psnrs):.1f} dB")

    return results

# ── RPCholesky + RPGKS sensor selection  (from gpoed-code-python) ────────────
def _rpcholesky_sensors(X, ls, k, kern_fn, rank=None, rng=None):
    """
    Select k sensors via Randomly Pivoted Cholesky + RPGKS.

    Adapted from gpoed-code-python/pivoted_cholesky.py and rpgks.py.
    Avoids building the full n×n kernel matrix — only computes one column
    per RPCholesky step → O(n × rank) memory and time.

    Parameters
    ----------
    X       : (n, d) coordinate array
    ls      : Matérn-3/2 lengthscale
    k       : number of sensors to select
    kern_fn : callable(A, B, ls) → (len(A), len(B)) kernel matrix
              (e.g. mgp.matern32)
    rank    : rank of the Cholesky approximation (default: k + 20)
              Higher rank → more accurate K approximation, slower.
    rng     : np.random.Generator for reproducibility

    Returns
    -------
    sensors : (k,) int32 array of selected point indices (0-based)

    Why this enables DS=1
    ---------------------
    At DS=1, n ≈ 250,000.  The full K is 500 GB — infeasible.
    RPCholesky builds F of shape (n, rank) ≈ (250K, k+20) → ~1 GB.
    Sensor selection then takes ~5–20 s instead of being impossible.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n    = len(X)
    rank = min(rank if rank is not None else k + 20, n)
    k    = min(k, n)

    # ── Step 1: Randomly Pivoted Cholesky — build low-rank factor F ────────
    # K[i,i] = 1 for unit-variance Matérn, so initialise diagonal to ones.
    diags = np.ones(n, dtype=np.float64)
    F     = np.zeros((n, rank), dtype=np.float64)
    actual_rank = rank
    for i in range(rank):
        total = diags.sum()
        if total <= 0:
            actual_rank = i
            F = F[:, :i]
            break
        weights = diags / total
        si = int(rng.choice(n, p=weights))          # random pivot

        col = kern_fn(X, X[[si]], ls).ravel()       # (n,) — one kernel column
        if i > 0:
            col = col - F[:, :i] @ F[si, :i]        # subtract accumulated rank

        pivot_val = float(col[si])
        if pivot_val <= 0:
            actual_rank = i
            F = F[:, :i]
            break
        F[:, i] = col / np.sqrt(pivot_val)
        diags = np.maximum(diags - F[:, i] ** 2, 0.0)

    if actual_rank < k:
        # K is nearly rank-deficient; fall back to greedy diagonal picks
        return np.array([int(np.argmax(diags))] * k, dtype=np.int32)

    # ── Step 2: RPGKS — top-k left singular vectors via F^T F + eigh ──────
    # Computing SVD of F (n × rank) via dgesdd is O(n × rank²) and slow for
    # large n. Instead: form G = F^T F (rank × rank), compute eigh(G) to get
    # right singular vectors V, then U_k = F @ V[:, -k:] (normalised).
    # Cost: O(n × rank²) for G, O(rank³) for eigh, O(n × rank × k/rank) for U_k.
    # Same asymptotic cost but better cache behaviour and no large U allocation.
    G       = F.T @ F                                   # (actual_rank, actual_rank)
    _, V    = np.linalg.eigh(G)                         # eigh sorts ascending
    V_k     = V[:, -k:]                                 # (actual_rank, k) top-k
    u_k     = F @ V_k                                   # (n, k) — unnormalised U_k
    norms   = np.linalg.norm(u_k, axis=0, keepdims=True)
    norms   = np.where(norms > 1e-12, norms, 1.0)
    u_k    /= norms
    _, _, p = scipy_qr(u_k.T, pivoting=True)           # column-pivoted QR
    return p[:k].astype(np.int32)


# ── MultiGP + SZ residual  (d=2: CLOUDf + QVAPORf) ──────────────────────────
def run_multigp(data_3d, data2_3d, k_values, abs_bounds=ABS_BOUNDS, ds=DS):
    """
    Two-field (d=2) LMC / ICM GP.  Fields: CLOUDf (field 0) + QVAPORf (field 1).

    Kronecker structure:
      K_sub = kron(B, K_ss_k) + σ²I   shape (dk × dk)
      y_flat[lvl] = [sensors_field0, sensors_field1]  shape (dk,)  — field-major
      alpha = K_sub^{-1} y_flat
      mu_norm[:, i] = Σ_j  B[i,j] * K_Xs @ alpha[j*k : (j+1)*k]

    CR = (d × n_3D_full × 4) / compressed_size   — joint compression of both fields.
    PSNR / SSIM reported as mean across both fields at the LEVEL slice.
    """
    mgp = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")

    d           = 2                                # CLOUDf + QVAPORf
    data_ds0    = data_3d[:, ::ds, ::ds].astype(np.float64)   # (n_L, ny, nz) field 0
    data_ds1    = data2_3d[:, ::ds, ::ds].astype(np.float64)  # (n_L, ny, nz) field 1
    n_L, ny, nz = data_ds0.shape
    n            = ny * nz
    n_3D_ds      = n_L * n              # single-field downsampled size (CR denominator ×d)

    X_all   = mgp.make_grid_coords(ny, nz)                             # (n, 2) grid coords
    data_nd = np.stack([data_ds0.reshape(n_L, n),
                        data_ds1.reshape(n_L, n)], axis=-1)            # (n_L, n, d)

    rng       = np.random.default_rng(42)
    n_train   = max(4, int(0.7 * n_L))
    train_idx = np.sort(rng.choice(n_L, n_train, replace=False))

    print(f"\n[MultiGP] d={d} (CLOUDf + QVAPORf)  ds={ds} ({ny}×{nz}={n}/level)  "
          f"n_train={n_train}")
    t0 = time.perf_counter()

    train_data = data_nd[train_idx]              # (n_train, n, d)
    train_mean = train_data.mean(axis=0)         # (n, d)
    train_std  = train_data.std(axis=0)
    zero_std_mask = train_std < 1e-10            # (n, d) — zero-variance locations
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)

    def norm(Y):   return (Y - train_mean) / train_std   # (n, d) → (n, d) z-scores
    def denorm(Z): return Z * train_std + train_mean

    Y_train_list = [norm(train_data[l]) for l in range(n_train)]  # list of (n, d)
    B = mgp.estimate_B(Y_train_list)   # (d, d)
    print(f"  B =\n    {B}")

    noise_var = 0.05 ** 2
    HP_STEP   = 10   # subsample for HP fitting — avoids O(n³) eigh on full grid
    X_hp      = X_all[::HP_STEP]
    Y_hp      = [Y[::HP_STEP] for Y in Y_train_list]
    print(f"  Fitting lengthscale on {len(X_hp)}-pt subgrid (HP_STEP={HP_STEP}) ...")
    ls = mgp.fit_lengthscale(X_hp, Y_hp, B, noise_var, n_restarts=2)
    print(f"  ls={ls:.3f}")

    k_max   = min(max(k_values), n)
    print(f"  Selecting {k_max} sensors (greedy, O(nk)) ...")
    sensors = _rpcholesky_sensors(X_all, ls, k_max, mgp.matern32)

    X_sens_max = X_all[sensors]
    K_Xs_max   = mgp.matern32(X_all, X_sens_max, ls)            # (n, k_max)
    K_ss_max   = mgp.matern32(X_sens_max, X_sens_max, ls) + 1e-6 * np.eye(k_max)

    # Compress mean/std once (shared across k)  — shape (n, d)
    mean_c = compress_f32(train_mean)
    std_c  = compress_f32(train_std)

    train_sec = time.perf_counter() - t0
    print(f"  Training done: {train_sec:.1f}s")

    # Pre-collect all sensor values in original units: (n_L, k_max, d)
    all_sv_orig_max = data_nd[:, sensors, :]    # (n_L, k_max, d)

    results = []

    for k in k_values:
        k      = min(k, k_max)
        s_k    = sensors[:k]
        K_Xs   = K_Xs_max[:, :k]     # (n, k)
        K_ss_k = K_ss_max[:k, :k]    # (k, k)

        # Kronecker Cholesky: K_sub = kron(B, K_ss_k) + σ²I   shape (dk × dk)
        K_sub = np.kron(B, K_ss_k) + noise_var * np.eye(d * k)
        try:
            L_kd, lower = cho_factor(K_sub, lower=True)
        except Exception:
            L_kd, lower = cho_factor(K_sub + 1e-4 * np.eye(d * k), lower=True)

        # ── Model (lossless, offline) ──────────────────────────────────────────
        # K_Xs and L_kd are NOT stored — decompressor recomputes them from hyperparams.
        # Stored: sensor indices + (ls, B matrix, noise_var, grid dims) + mean + std.
        hyperparams_b = struct.pack('<IId', ny, nz, float(ls))
        b_c     = compress_f32(B)           # (d,d) — small
        sens_c  = compress_i32(s_k)
        model_b = len(hyperparams_b) + len(b_c) + len(sens_c) + len(mean_c) + len(std_c)

        # Sensor values for this k: (n_L, k, d) in original data units
        all_sv_orig = all_sv_orig_max[:, :k, :]    # (n_L, k, d)

        # Stored field-major: [sensors_f0, sensors_f1] per level → (n_L, d*k)
        # For quantization, treat all n_L × d*k values together
        all_sv_flat = np.concatenate(
            [all_sv_orig[:, :, i] for i in range(d)], axis=1
        ).astype(np.float32)   # (n_L, d*k)

        k_results = []

        for ab in abs_bounds:
            t0 = time.perf_counter()

            # ── Quantize sensor values (n_L × dk) ────────────────────────────
            bins_sv, op_sv, ov_sv = quantize(all_sv_flat, ab)
            sv_enc  = pack_encode(bins_sv, op_sv, ov_sv)
            sv_deq  = dequantize(bins_sv, op_sv, ov_sv, ab,
                                 (n_L, d * k))      # (n_L, dk)
            # Split back to per-field: (n_L, k) each
            sv_deq_fields = [sv_deq[:, i*k:(i+1)*k] for i in range(d)]

            # ── GP prediction using dequantized sensors ───────────────────────
            recon_ds = np.zeros((n_L, n, d), dtype=np.float64)
            for lvl in range(n_L):
                # Build y_flat for this level: [f0_sensors, f1_sensors] in z-score
                y_norm_deq = np.concatenate([
                    (sv_deq_fields[i][lvl] - train_mean[s_k, i]) / train_std[s_k, i]
                    for i in range(d)
                ])                                              # (dk,)

                alpha = cho_solve((L_kd, lower), y_norm_deq)   # (dk,)

                # Posterior mean for each output field: Σ_j B[i,j] K_Xs alpha_j
                mu_norm = np.zeros((n, d))
                for i in range(d):
                    for j in range(d):
                        mu_norm[:, i] += B[i, j] * K_Xs @ alpha[j*k:(j+1)*k]

                recon_ds[lvl] = denorm(mu_norm)     # (n, d)
                # Zero-variance locations: GP z-score is meaningless — predict mean
                recon_ds[lvl][zero_std_mask] = train_mean[zero_std_mask]

            recon_3d_nd = recon_ds.reshape(n_L, ny, nz, d)

            # ── Residuals for both fields: (n_L, ny, nz, d) ──────────────────
            orig_3d_nd = np.stack([data_ds0, data_ds1], axis=-1)   # (n_L, ny, nz, d)
            resid = (orig_3d_nd - recon_3d_nd).astype(np.float32)

            # ── Quantize + encode residuals ───────────────────────────────────
            bins_r, op_r, ov_r = quantize(resid, ab)
            resid_enc = pack_encode(bins_r, op_r, ov_r)

            comp_sec = time.perf_counter() - t0

            total_b  = model_b + len(sv_enc) + len(resid_enc)
            # CR: joint compression of d fields vs d × n_3D_ds uncompressed bytes
            cr       = (d * n_3D_ds * 4) / total_b

            # ── Reconstruct for metrics ───────────────────────────────────────
            resid_rec = dequantize(bins_r, op_r, ov_r, ab,
                                   (n_L, ny, nz, d))
            final_nd  = recon_3d_nd + resid_rec                    # (n_L, ny, nz, d)

            # PSNR/SSIM: report CLOUDf (field 0) only for apples-to-apples
            # comparison with SZ2/ZFP/DEIM-2D which all report CLOUDf metrics.
            # QVAPORf metrics are stored separately for reference.
            pv0, sv0 = metrics(data_ds0[LEVEL].astype(np.float32),
                               final_nd[LEVEL, :, :, 0].astype(np.float32))
            pv1, sv1 = metrics(data_ds1[LEVEL].astype(np.float32),
                               final_nd[LEVEL, :, :, 1].astype(np.float32))

            row = {
                "method":        "MultiGP",
                "k":             k,
                "abs_bound":     ab,
                "cr":            cr,
                "psnr":          pv0,   # CLOUDf only
                "ssim":          sv0,   # CLOUDf only
                "psnr_cloud":    pv0,
                "ssim_cloud":    sv0,
                "psnr_qvapor":   pv1,
                "ssim_qvapor":   sv1,
                "comp_sec":      comp_sec,
                "decomp_sec":    0.0,
                "train_sec":     train_sec,
                "compressed_MB": total_b / 1e6,
                "model_MB":      model_b / 1e6,
                "sv_MB":         len(sv_enc) / 1e6,
                "resid_MB":      len(resid_enc) / 1e6,
                "n_outliers":    len(op_r),
            }
            results.append(row)
            k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"  k={k:3d}  model={model_b/1e3:.0f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR(CLOUDf): {min(psnrs):.1f}–{max(psnrs):.1f} dB")

    return results

# ── DEIM-2D (per-level 2D DEIM) ──────────────────────────────────────────────
def run_deim_2d(data_3d, k_values, abs_bounds=ABS_BOUNDS, ds=DS):
    """
    2D DEIM applied independently per level.  Spatial modes from the right
    singular vectors of the (n_L × n_2D) snapshot matrix.

    Model: float16(Phi_2D n_2D×k) + Zstd  — 100× smaller than T-DEIM
           int32(sensors)  + Zstd
           float32(mean)   + Zstd
    CR = (n_3D_ds × 4) / (model + quantized_sensors + quantized_residuals)
    """
    data_ds          = data_3d[:, ::ds, ::ds].astype(np.float64)
    n_L, ny, nz      = data_ds.shape
    n_2D             = ny * nz
    n_3D_ds          = n_L * n_2D

    print(f"\n[DEIM-2D] SVD  ds={ds} → {ny}×{nz}={n_2D}/level ...")
    t0 = time.perf_counter()
    mean_ds  = data_ds.mean(axis=0)
    F        = (data_ds - mean_ds).reshape(n_L, n_2D)
    k_max    = min(max(k_values), min(n_L, n_2D))
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    Phi_max  = Vt[:k_max, :].T           # (n_2D, k_max) — 2D spatial modes
    train_sec = time.perf_counter() - t0
    print(f"  SVD done: {train_sec:.1f}s  Phi_2D {Phi_max.shape}")

    mean_c   = compress_f32(mean_ds)
    data_flat = (data_ds - mean_ds).reshape(n_L, n_2D)
    results  = []

    for k in k_values:
        k      = min(k, k_max)
        Phi_k  = Phi_max[:, :k]          # (n_2D, k)

        # Q-DEIM sensors in the 2D spatial domain
        _, _, p  = scipy_qr(Phi_k.T, pivoting=True)
        sensors  = p[:k]                  # indices in [0, n_2D)

        # Model (tiny: n_2D × k, not n_3D_ds × k)
        phi_c    = compress_f16(Phi_k)
        sens_c   = compress_i32(sensors)
        model_b  = len(phi_c) + len(sens_c) + len(mean_c)

        A        = Phi_k[sensors, :]      # (k, k) — for the DEIM solve

        # Sensor values across all levels: (n_L, k)
        all_sv   = data_flat[:, sensors].astype(np.float32)

        # Sensors stored as exact float32 (no quantization).
        # DEIM interpolates exactly at sensor locations by construction:
        # Phi_k[sensors,:] @ c = sv, so residual at sensors is identically 0.
        # Quantizing sensors first would only add noise into the solve — no benefit.
        sv_enc_exact = _compress(all_sv.astype(np.float16).tobytes())  # float16: ~0.01% rel error, half storage

        k_results = []

        # Reconstruct once per k (prediction doesn't depend on abs_bound)
        recon_flat = np.zeros((n_L, n_2D), dtype=np.float64)
        for lvl in range(n_L):
            c = np.linalg.solve(A, all_sv[lvl])
            recon_flat[lvl] = Phi_k @ c
        recon_ds = recon_flat.reshape(n_L, ny, nz) + mean_ds
        resid = (data_ds - recon_ds).ravel().astype(np.float32)

        for ab in abs_bounds:
            t0 = time.perf_counter()

            # Quantize residuals — bin_width = 2*abs_bound/65536, matching SZ2
            bins_r, op_r, ov_r = quantize(resid, ab, NUM_BINS)
            resid_enc = pack_encode(bins_r, op_r, ov_r)

            comp_sec = time.perf_counter() - t0

            data_b  = len(sv_enc_exact) + len(resid_enc)
            total_b = model_b + data_b
            cr           = (n_3D_ds * 4) / total_b
            cr_no_model  = (n_3D_ds * 4) / data_b

            resid_rec = dequantize(bins_r, op_r, ov_r, ab, (n_L, ny, nz), NUM_BINS)
            final_ds  = recon_ds + resid_rec
            pv, sv_m  = metrics(data_ds[LEVEL].astype(np.float32),
                                final_ds[LEVEL].astype(np.float32))

            row = {"method": "DEIM-2D", "k": k, "abs_bound": ab,
                   "cr": cr, "cr_no_model": cr_no_model,
                   "psnr": pv, "ssim": sv_m,
                   "comp_sec": comp_sec, "decomp_sec": 0.0, "train_sec": train_sec,
                   "compressed_MB": total_b/1e6, "model_MB": model_b/1e6,
                   "sv_MB": len(sv_enc_exact)/1e6, "resid_MB": len(resid_enc)/1e6,
                   "n_outliers": len(op_r)}
            results.append(row)
            k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"  k={k:3d}  model={model_b/1e3:.1f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR: {min(psnrs):.1f}–{max(psnrs):.1f} dB")

    return results

# ── Kriging-2D (per-level GP, d=1) ───────────────────────────────────────────
def run_kriging_2d(data_3d, k_values, abs_bounds=ABS_BOUNDS, ds=DS):
    """
    2D Kriging applied independently per level.  Matérn-3/2 GP with k sensors
    chosen by GKS (greedy k-subset selection on K_spatial).

    Model: float16(K_Xs n_2D×k) + Zstd   — same size class as DEIM-2D
           float32(L_k k×k, mean, std)    + Zstd
           int32(sensors)                 + Zstd
    CR = (n_3D_ds × 4) / (model + quantized_sensors + quantized_residuals)

    Conceptually: Bayesian counterpart to DEIM-2D.  DEIM uses an orthogonal
    basis + linear solve; Kriging uses a kernel-based posterior mean + GKS
    sensors.  Both have O(n_2D × k) model cost.
    """
    mgp = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")

    data_ds          = data_3d[:, ::ds, ::ds].astype(np.float64)
    n_L, ny, nz      = data_ds.shape
    n_2D             = ny * nz
    n_3D_ds          = n_L * n_2D
    data_nd          = data_ds.reshape(n_L, n_2D, 1)   # (n_L, n, 1) for mgp helpers

    X_all = mgp.make_grid_coords(ny, nz)               # (n_2D, 2)

    # Use ALL levels for hyperparameter estimation — consistent with DEIM-2D which builds
    # its SVD from all n_L levels.  These are simultaneous vertical z-levels of one
    # 3D snapshot, so there is no train/test leakage: all levels are available at
    # compression time.
    n_train   = n_L
    train_idx = np.arange(n_L)

    print(f"\n[Kriging-2D] ds={ds} ({ny}×{nz}={n_2D}/level)  n_train={n_train} (all levels)")
    t0 = time.perf_counter()

    train_data = data_nd[train_idx]
    train_mean = train_data.mean(axis=0)          # (n_2D, 1)
    train_std  = train_data.std(axis=0)
    zero_std_mask_k = (train_std[:, 0] < 1e-10)  # (n_2D,) zero-variance locations
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)

    def norm(Y):   return (Y - train_mean) / train_std
    def denorm(Z): return Z * train_std + train_mean

    Y_train_list = [norm(train_data[l]) for l in range(n_train)]
    B = mgp.estimate_B(Y_train_list)              # (1, 1) — scalar for d=1

    # Mirror kriging_hurricane.py fit_hyperparams: random subset, 3 restarts,
    # maxiter=200, ftol=1e-9, fit ls + var + noise jointly.
    k_max    = min(max(k_values), n_2D)
    fit_size = min(k_max * 4, n_2D)
    rng_fit  = np.random.default_rng(0)
    fit_idx  = rng_fit.choice(n_2D, size=fit_size, replace=False)
    X_fit    = X_all[fit_idx]
    Y_fit    = Y_train_list[0][fit_idx].ravel()    # one level (SHOWCASE_TRAIN_IDX=0)

    print(f"  Fitting ls + var + noise on {fit_size} random pts, n_restarts=3 ...")
    from scipy.optimize import minimize as _minimize
    from scipy.spatial.distance import cdist as _cdist

    def _neg_lml_2d(log_theta):
        ls_    = float(np.exp(log_theta[0]))
        var_   = float(np.exp(log_theta[1]))
        noise_ = float(np.exp(log_theta[2]))
        n_fit  = len(X_fit)
        K = var_ * mgp.matern32(X_fit, X_fit, ls_) + noise_ * np.eye(n_fit)
        try:
            L_ = np.linalg.cholesky(K + 1e-8 * np.eye(n_fit))
        except np.linalg.LinAlgError:
            return 1e10
        log_det = 2.0 * np.sum(np.log(np.diag(L_)))
        alpha   = np.linalg.solve(L_.T, np.linalg.solve(L_, Y_fit))
        return 0.5 * float(Y_fit @ alpha) + 0.5 * log_det

    dists  = _cdist(X_fit[:10], X_fit[:10], 'euclidean')
    ls0    = float(np.median(dists[dists > 0])) / 3.0
    var0   = max(float(np.var(Y_fit)), 1e-6)
    noise0 = var0 * 0.01
    bounds = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]   # match prototype
    rng_hp = np.random.default_rng(0)
    starts = [np.log([max(ls0, 0.1), max(var0, 0.01), max(noise0, 1e-6)])] + [
        rng_hp.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        for _ in range(2)
    ]
    best_nll, best_ls, best_var, best_noise = np.inf, ls0, var0, noise0
    for x0 in starts:
        try:
            res = _minimize(_neg_lml_2d, x0, method="L-BFGS-B",
                            bounds=bounds,
                            options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_nll:
                best_nll   = res.fun
                best_ls    = float(np.exp(res.x[0]))
                best_var   = float(np.exp(res.x[1]))
                best_noise = float(np.exp(res.x[2]))
        except Exception:
            pass
    ls        = best_ls
    noise_var = best_noise
    B         = np.array([[best_var]])   # absorb fitted variance into B (d=1)
    print(f"  ls={ls:.3f}  var={B[0,0]:.4f}  noise={noise_var:.2e}")
    print(f"  Selecting {k_max} sensors (greedy, O(nk)) ...")
    sensors = _rpcholesky_sensors(X_all, ls, k_max, mgp.matern32)

    X_sens_max = X_all[sensors]
    K_Xs_max   = mgp.matern32(X_all, X_sens_max, ls)           # (n_2D, k_max)
    K_ss_max   = mgp.matern32(X_sens_max, X_sens_max, ls) + 1e-6 * np.eye(k_max)

    mean_c = compress_f32(train_mean)
    std_c  = compress_f32(train_std)

    train_sec = time.perf_counter() - t0
    print(f"  Training done: {train_sec:.1f}s")

    all_sv_max = data_nd[:, sensors, 0].astype(np.float32)   # (n_L, k_max)

    results = []

    for k in k_values:
        k      = min(k, k_max)
        s_k    = sensors[:k]
        K_Xs   = K_Xs_max[:, :k]
        K_ss_k = K_ss_max[:k, :k]

        # For d=1: K_sub = B[0,0] * K_ss + σ²I
        K_sub = B[0, 0] * K_ss_k + noise_var * np.eye(k)
        try:
            L_k, lower = cho_factor(K_sub, lower=True)
        except Exception:
            L_k, lower = cho_factor(K_sub + 1e-4 * np.eye(k), lower=True)

        # Model: store only sensor indices + hyperparams (ls, B, noise_var, grid dims).
        # K_Xs and L_k are NOT stored — the decompressor recomputes them from the
        # kernel function + stored hyperparams + grid coords. This eliminates the
        # dominant model term (n_2D × k float16) and leaves only ~k×4 + 32 bytes.
        hyperparams_b = struct.pack('<IIddd', ny, nz, float(ls), float(B[0, 0]), float(noise_var))
        sens_c  = compress_i32(s_k)
        model_b = len(hyperparams_b) + len(sens_c) + len(mean_c) + len(std_c)
        # K_Xs and L_k stay in memory for the reconstruction loop below (not stored)

        all_sv  = all_sv_max[:, :k]    # (n_L, k) original units

        # Sensors stored as exact float32 (no quantization), like standalone scripts.
        sv_enc_exact = _compress(all_sv.astype(np.float16).tobytes())  # float16: ~0.01% rel error, half storage

        # Predict once per k — GP posterior mean doesn't depend on abs_bound
        recon_flat = np.zeros((n_L, n_2D), dtype=np.float64)
        for lvl in range(n_L):
            y_norm  = (all_sv[lvl] - train_mean[s_k, 0]) / train_std[s_k, 0]
            alpha   = cho_solve((L_k, lower), y_norm)
            mu_norm = K_Xs @ (B[0, 0] * alpha)
            row_val = denorm(mu_norm.reshape(n_2D, 1))[:, 0]
            row_val[zero_std_mask_k] = train_mean[zero_std_mask_k, 0]
            recon_flat[lvl] = row_val
        recon_ds = recon_flat.reshape(n_L, ny, nz)
        resid    = (data_ds - recon_ds).ravel().astype(np.float32)

        k_results = []

        for ab in abs_bounds:
            t0 = time.perf_counter()

            bins_r, op_r, ov_r = quantize(resid, ab)
            resid_enc = pack_encode(bins_r, op_r, ov_r)

            comp_sec = time.perf_counter() - t0

            data_b  = len(sv_enc_exact) + len(resid_enc)
            total_b = model_b + data_b
            cr          = (n_3D_ds * 4) / total_b
            cr_no_model = (n_3D_ds * 4) / data_b

            resid_rec = dequantize(bins_r, op_r, ov_r, ab, (n_L, ny, nz))
            final_ds  = recon_ds + resid_rec
            pv, sv_m  = metrics(data_ds[LEVEL].astype(np.float32),
                                final_ds[LEVEL].astype(np.float32))

            row = {"method": "Kriging-2D", "k": k, "abs_bound": ab,
                   "cr": cr, "cr_no_model": cr_no_model,
                   "psnr": pv, "ssim": sv_m,
                   "comp_sec": comp_sec, "decomp_sec": 0.0, "train_sec": train_sec,
                   "compressed_MB": total_b/1e6, "model_MB": model_b/1e6,
                   "sv_MB": len(sv_enc_exact)/1e6, "resid_MB": len(resid_enc)/1e6,
                   "n_outliers": len(op_r)}
            results.append(row)
            k_results.append(row)

        crs   = [r["cr"]   for r in k_results]
        psnrs = [r["psnr"] for r in k_results]
        print(f"  k={k:3d}  model={model_b/1e3:.1f} kB  "
              f"CR: {min(crs):.1f}–{max(crs):.1f}×  "
              f"PSNR: {min(psnrs):.1f}–{max(psnrs):.1f} dB")

    return results

# ── Pareto-optimal upper envelope ─────────────────────────────────────────────
def pareto_upper(pts, metric_key="psnr"):
    """Upper envelope: sort by CR ascending, running max of metric."""
    srt = sorted(pts, key=lambda x: x["cr"])
    env, best = [], -np.inf
    for p in srt:
        if p[metric_key] > best:
            best = p[metric_key]
            env.append(p)
    return env

def pareto_upper_bpv(pts, metric_key="psnr"):
    """Pareto upper envelope in (bpv, metric) space.
    bpv = 32/CR — lower bpv = more compressed.  Sort bpv ascending, running max of metric.
    Equivalent to pareto_upper (same CR ordering), just converts x-axis to bpv."""
    srt = sorted(pts, key=lambda x: x["cr"], reverse=True)  # high CR = low bpv first
    env, best = [], -np.inf
    for p in srt:
        if p[metric_key] > best:
            best = p[metric_key]
            env.append(p)
    return env

# ── Plots ─────────────────────────────────────────────────────────────────────
# All methods run on the same ds=10 (50×50/level) downsampled data.
# CR = (n_3D_ds × 4) / compressed_size for single-field methods.
# MultiGP uses d × n_3D_ds since it jointly compresses d=2 fields.
ALL_METHODS = ["SZ2", "ZFP", "DEIM-2D", "T-DEIM", "Kriging-2D", "MultiGP"]
# SZ2/ZFP: solid lines (single parameter sweep)
# Others:  Pareto envelope (thick dashed) — best PSNR achievable at each CR by choosing k

def _valid(pts, metric_key):
    """Filter out points with NaN/inf CR or metric (guards against libpressio returning nan)."""
    return [p for p in pts
            if np.isfinite(p.get("cr", float("nan")))
            and np.isfinite(p.get(metric_key, float("nan")))]

def make_rd_plot(by_method, metric_key, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ALL_METHODS:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            print(f"  WARNING: no valid {metric_key} data for {method} — skipping in {fname}")
            continue
        c, mk = COLORS[method], MARKERS[method]

        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: x["cr"])
            crs  = [p["cr"]       for p in srt]
            vals = [p[metric_key] for p in srt]
            # Show parameter range in legend
            params = sorted(set(p["abs_bound"] for p in srt))
            if False:  # placeholder — both SZ2 and ZFP now use abs_bound
                pass
            else:
                range_str = f" [ab={params[0]:.0e}–{params[-1]:.0e}]"
            ax.plot(crs, vals, "-", color=c, marker=mk, ms=7, lw=2, label=method + range_str)
            # Label only the first (lowest-CR) point
            p0 = srt[0]
            label0 = f"ab={p0['abs_bound']:.2g}"
            ax.annotate(label0, (crs[0], vals[0]),
                        fontsize=6, color=c, alpha=0.85,
                        textcoords="offset points", xytext=(4, 2))
        else:
            env  = pareto_upper(pts, metric_key)
            crs  = [p["cr"]       for p in env]
            vals = [p[metric_key] for p in env]
            # Legend shows parameter ranges explored
            k_vals = sorted(set(p["k"] for p in pts if p.get("k") is not None))
            ab_vals = sorted(set(p["abs_bound"] for p in pts))
            range_str = (f" [k={k_vals[0]}–{k_vals[-1]}, "
                         f"ab={ab_vals[0]:.0e}–{ab_vals[-1]:.0e}]")
            ax.plot(crs, vals, "--", color=c, marker=mk, ms=7, lw=2,
                    label=method + range_str)
            # Label only the first (lowest-CR) Pareto point
            if env:
                p0 = env[0]
                ax.annotate(f"k={p0['k']}, ab={p0['abs_bound']:.1g}",
                            (p0["cr"], p0[metric_key]),
                            fontsize=6, color=c, alpha=0.85,
                            textcoords="offset points", xytext=(4, 2))

    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio (log scale)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        f"Rate-Distortion: {ylabel} vs CR  —  {FIELD_TAG} ds={DS} (50×50/level) slice {LEVEL}\n"
        f"All methods on same downsampled data. CR vs n_3D_ds. "
        f"Dashed = Pareto envelope over k values. Dotted = model bytes free.",
        fontsize=9)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

def make_bpv_plot(by_method, metric_key, ylabel, fname):
    """Bit rate (bpv = bits per value) vs PSNR/SSIM — standard RD plot.

    bpv = 32 / CR  since input is float32 (32 bits/value) and CR = uncompressed/compressed.
    x-axis: bpv (log scale, lower = more compressed = more bits saved).
    For ZFP fixed-accuracy mode, bpv = 32/CR (derived from actual compressed size).
    For MultiGP (d=2): CR = (2 × n_3D_ds × 4) / total_b, so bpv = average bits/value/field.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ALL_METHODS:
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            print(f"  WARNING: no valid {metric_key} data for {method} — skipping in {fname}")
            continue
        c, mk = COLORS[method], MARKERS[method]

        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: 32.0 / x["cr"])   # sort low→high bpv
            bpvs = [32.0 / p["cr"]   for p in srt]
            vals = [p[metric_key]    for p in srt]
            # Show parameter range in legend
            params = sorted(set(p["abs_bound"] for p in srt))
            if False:  # placeholder — both SZ2 and ZFP now use abs_bound
                pass
            else:
                range_str = f" [ab={params[0]:.0e}–{params[-1]:.0e}]"
            ax.plot(bpvs, vals, "-", color=c, marker=mk, ms=7, lw=2, label=method + range_str)
            # Label only the first (lowest-CR = highest-bpv) point
            p0 = srt[0]
            label0 = f"ab={p0['abs_bound']:.2g}"
            ax.annotate(label0, (bpvs[0], vals[0]),
                        fontsize=6, color=c, alpha=0.85,
                        textcoords="offset points", xytext=(4, 2))
        else:
            env  = pareto_upper_bpv(pts, metric_key)
            bpvs = [32.0 / p["cr"]  for p in env]
            vals = [p[metric_key]   for p in env]
            # Legend shows parameter ranges explored
            k_vals  = sorted(set(p["k"] for p in pts if p.get("k") is not None))
            ab_vals = sorted(set(p["abs_bound"] for p in pts))
            range_str = (f" [k={k_vals[0]}–{k_vals[-1]}, "
                         f"ab={ab_vals[0]:.0e}–{ab_vals[-1]:.0e}]")
            ax.plot(bpvs, vals, "--", color=c, marker=mk, ms=7, lw=2,
                    label=method + range_str)
            # Label only the first (highest-bpv = most compressed) Pareto point
            if env:
                p0, bpv0 = env[0], bpvs[0]
                ax.annotate(f"k={p0['k']}, ab={p0['abs_bound']:.1g}",
                            (bpv0, p0[metric_key]),
                            fontsize=6, color=c, alpha=0.85,
                            textcoords="offset points", xytext=(4, 2))

    ax.set_xscale("log")
    ax.set_xlabel("Bit rate (bits per value — lower is more compressed)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        f"Rate-Distortion: {ylabel} vs bit rate  —  {FIELD_TAG} ds={DS} (50×50/level) slice {LEVEL}\n"
        f"bpv = 32/CR  (float32 input). Dashed = Pareto envelope. Dotted = model bytes free.",
        fontsize=9)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

def make_absbound_plot(by_method, metric_key, ylabel, fname):
    """Quality (PSNR or SSIM) vs abs_bound — fair comparison at matched error tolerance.

    X-axis: abs_bound (same value used by all methods).
    For SZ2: one point per abs_bound (single parameter).
    For our methods: best quality across all k at each abs_bound — shows the
    ceiling achievable by sensor-based methods at each error budget.
    ZFP excluded: it uses rate (bpv), not abs_bound.

    This plot answers: "at the same error tolerance, who reconstructs better?"
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ALL_METHODS:
        if False:
            pass  # ZFP now uses abs_bound (fixed-accuracy mode) — include it
        pts = _valid(by_method.get(method, []), metric_key)
        if not pts:
            print(f"  WARNING: no valid {metric_key} data for {method} — skipping in {fname}")
            continue
        c, mk = COLORS[method], MARKERS[method]

        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: x["abs_bound"])
            abs_b = [p["abs_bound"]  for p in srt]
            vals  = [p[metric_key]   for p in srt]
            ax.plot(abs_b, vals, "-", color=c, marker=mk, ms=7, lw=2, label=method)
            for p in srt:
                ax.annotate(f"{p['abs_bound']:.3g}", (p["abs_bound"], p[metric_key]),
                            fontsize=6, color=c, alpha=0.8,
                            textcoords="offset points", xytext=(4, 2))
        else:
            # Best quality at each abs_bound (max over k)
            ab_groups = defaultdict(list)
            for p in pts:
                ab_groups[p["abs_bound"]].append((p[metric_key], p["k"]))
            abs_sorted = sorted(ab_groups.keys())
            best_vals  = [max(ab_groups[ab], key=lambda x: x[0]) for ab in abs_sorted]
            ys   = [v for v, _ in best_vals]
            ks   = [k for _, k in best_vals]
            k_vals  = sorted(set(p["k"] for p in pts if p.get("k") is not None))
            range_str = f" [k={k_vals[0]}–{k_vals[-1]}]" if k_vals else ""
            ax.plot(abs_sorted, ys, "--", color=c, marker=mk, ms=7, lw=2,
                    label=method + range_str)
            # Label each point with the k that achieved the best quality there
            for ab, y, k in zip(abs_sorted, ys, ks):
                lbl = f"k={k}" if k is not None else f"{ab:.2g}"
                ax.annotate(lbl, (ab, y),
                            fontsize=6, color=c, alpha=0.8,
                            textcoords="offset points", xytext=(4, 2))

    ax.set_xscale("log")
    ax.invert_xaxis()   # tighter abs_bound (left) = harder constraint = better quality
    ax.set_xlabel("abs_bound  (← tighter / looser →)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        f"Quality vs Error Tolerance: {ylabel} vs abs_bound  —  {FIELD_TAG}\n"
        f"Our methods: best {ylabel} over all k at each abs_bound.  "
        f"ZFP included (fixed-accuracy mode, same abs_bound sweep as SZ2).",
        fontsize=9)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

def make_sensor_mult_plot(by_method, fname):
    """Per-sensor-mult Pareto envelope for T-DEIM and DEIM-2D.

    Each line = one sensor_mult value; Pareto taken over (k, abs_bound).
    Shows whether loosening the sensor abs_bound (while keeping residual bound fixed)
    shifts the RD curve toward higher CR without hurting PSNR.
    SZ2 and ZFP are included as reference baselines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    mult_colors  = {1: "#333333", 10: "#2ca02c", 100: "#ff7f0e", 1000: "#d62728"}
    mult_styles  = {1: "-",       10: "--",       100: "-.",      1000: ":"}

    for ax, method in zip(axes, ["T-DEIM", "DEIM-2D"]):
        # Baselines as faint reference lines
        for ref in ("SZ2", "SZ3-Lorenzo", "SZ3-Interp", "ZFP"):
            pts = _valid(by_method.get(ref, []), "psnr")
            if pts:
                srt = sorted(pts, key=lambda x: x["cr"])
                ax.plot([p["cr"] for p in srt], [p["psnr"] for p in srt],
                        "-", color=COLORS[ref], lw=1.5, alpha=0.5, label=ref)

        pts_all = _valid(by_method.get(method, []), "psnr")
        if not pts_all:
            ax.set_title(f"{method} (no data)")
            continue

        for mult in SENSOR_MULT_VALUES:
            pts_m = [p for p in pts_all if p.get("sensor_mult", 1) == mult]
            if not pts_m:
                continue
            env  = pareto_upper(pts_m, "psnr")
            crs  = [p["cr"]   for p in env]
            vals = [p["psnr"] for p in env]
            ax.plot(crs, vals,
                    mult_styles[mult], color=mult_colors[mult],
                    marker="o", ms=5, lw=2,
                    label=f"sensor×{mult}")

        ax.set_xscale("log")
        ax.set_xlabel("Compression Ratio (log scale)", fontsize=11)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title(f"{method} — sensor abs_bound multiplier sweep", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Effect of loosening sensor abs_bound (residual abs_bound unchanged)\n"
        f"sensor_ab = abs_bound × mult  |  resid_ab = abs_bound  |  ds={DS}",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

def make_timing_plot(by_method, fname):
    """
    Per-k timing lines for each method.
    x = CR (log),  y = online compression time (log).
    Each dot = one (k, abs_bound) combo at median abs_bound.
    Larger k → lower CR + more time — shows the clear tradeoff.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ALL_METHODS:
        pts = by_method.get(method, [])
        if not pts:
            continue
        c, mk = COLORS[method], MARKERS[method]

        if method in ("SZ2", "ZFP"):
            srt  = sorted(pts, key=lambda x: x["cr"])
            crs  = [p["cr"]       for p in srt]
            comp = [p["comp_sec"] for p in srt]
            ax.plot(crs, comp, "-", color=c, marker=mk, ms=7, lw=2, label=method)
        else:
            # One point per k at median abs_bound; connect with a line
            k_groups = defaultdict(list)
            for p in pts:
                k_groups[p["k"]].append(p)
            # For each k: pick the point closest to median abs_bound
            k_pts = []
            for k in sorted(k_groups.keys()):
                kpts = sorted(k_groups[k], key=lambda x: x["abs_bound"])
                mid  = kpts[len(kpts) // 2]   # median abs_bound point
                k_pts.append(mid)
            k_pts = sorted(k_pts, key=lambda x: x["cr"])
            crs  = [p["cr"]       for p in k_pts]
            comp = [p["comp_sec"] for p in k_pts]
            ks   = [p["k"]        for p in k_pts]
            k_range = sorted(set(p["k"] for p in k_pts))
            range_str = f" [k={k_range[0]}–{k_range[-1]}]"
            ax.plot(crs, comp, "--", color=c, marker=mk, ms=7, lw=1.5,
                    label=method + range_str)
            # Label only the first (lowest k = highest CR = leftmost) point
            ax.annotate(f"k={ks[0]}", (crs[0], comp[0]), fontsize=7, color=c,
                        textcoords="offset points", xytext=(4, 3))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compression Ratio (log scale)", fontsize=12)
    ax.set_ylabel("Online compression time (s, log scale)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        "Online Compression Time vs CR  (training excluded)\n"
        "Each point = one k value at median abs_bound. Larger k → lower CR + more time.",
        fontsize=9)
    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

def make_budget_breakdown(by_method, fname):
    """Stacked bar: model / sensor-values / residuals bytes per k, at tightest abs_bound.

    Splitting sensors and residuals into separate bars reveals why larger k can
    sometimes cost LESS total data bytes: more sensors → better prediction →
    much smaller residuals, even though sensor storage itself grows with k.
    """
    FS_TITLE  = 20   # panel title
    FS_LABEL  = 20   # axis labels
    FS_TICK   = 17   # tick labels
    FS_LEGEND = 16   # legend text

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharey=False)
    axes = axes.ravel()

    for ax, method in zip(axes, ["T-DEIM", "DEIM-2D", "Kriging-2D", "MultiGP"]):
        pts = by_method.get(method, [])
        if not pts:
            ax.set_title(f"{method} (no data)", fontsize=FS_TITLE)
            continue
        k_groups = defaultdict(list)
        for p in pts:
            k_groups[p["k"]].append(p)

        ks, model_mb, sv_mb, resid_mb = [], [], [], []
        for k in sorted(k_groups):
            best = min(k_groups[k], key=lambda x: x["abs_bound"])
            ks.append(k)
            model_mb.append(best.get("model_MB", 0.0))
            sv_mb.append(best.get("sv_MB", float("nan")))
            resid_mb.append(best.get("resid_MB", float("nan")))

        # Fall back to combined data bar if sv_MB / resid_MB not in CSV
        have_split = not any(np.isnan(v) for v in sv_mb)

        x   = np.arange(len(ks))
        # Fixed colors per category (same across all panels for easy cross-panel reading)
        COL_MODEL = "#4472C4"   # blue  — model overhead
        COL_SV    = "#ED7D31"   # orange — sensor values (grows with k)
        COL_RES   = "#70AD47"   # green  — residuals (shrinks with k as prediction improves)

        model_arr = np.array(model_mb)
        sv_arr    = np.array(sv_mb)
        res_arr   = np.array(resid_mb)

        ax.bar(x, model_arr, label="Model (basis + sensor indices + mean)", color=COL_MODEL)
        if have_split:
            ax.bar(x, sv_arr,  bottom=model_arr,          label="Data: sensor values  ↑ with k",
                   color=COL_SV)
            ax.bar(x, res_arr, bottom=model_arr + sv_arr, label="Data: residuals  ↓ with k",
                   color=COL_RES)
        else:
            other = np.array([max(0, best.get("compressed_MB", 0) - m)
                              for best, m in zip(
                                  [min(k_groups[k], key=lambda x: x["abs_bound"])
                                   for k in sorted(k_groups)],
                                  model_mb)])
            ax.bar(x, other, bottom=model_arr, label="Data (sensors + residuals)", color=COL_SV)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=FS_TICK)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_ylabel("Compressed size (MB)", fontsize=FS_LABEL)
        ax.set_title(
            f"{method} — budget at tightest abs_bound\n"
            f"(More sensors → better prediction → smaller residuals)", fontsize=FS_TITLE)
        ax.legend(fontsize=FS_LEGEND, prop={'size': FS_LEGEND})

    plt.tight_layout()
    fig.savefig(ARGONNE / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {ARGONNE / fname}")
    plt.close(fig)

# ── Residual histogram (k=10, one abs_bound) ──────────────────────────────────
def make_residual_histogram(data_3d, data2_3d=None, ds=DS, k=10,
                            ab=1e-2, num_bins=65536):
    """Residual bin-index histogram for ALL methods at a single (k, abs_bound).

    For each method: compute prediction → residual = data - prediction →
    quantize with bin_width = 2*abs_bound/num_bins → plot bin-index distribution.
    A spike at bin 0 means the prediction is accurate (residuals near zero).
    Wide/flat distribution means the prediction leaves large errors.

    SZ2/SZ3/ZFP residuals are computed from the libpressio round-trip reconstruction.
    ZFP uses accuracy=1e-2 as a fixed reference point (fixed-accuracy mode).
    """
    from scipy.linalg import cho_factor, cho_solve

    data_ds = np.ascontiguousarray(data_3d[:, ::ds, ::ds])
    n_L, ny, nz = data_ds.shape
    n_2D    = ny * nz
    n_3D_ds = n_L * n_2D

    bw   = 2.0 * ab / num_bins
    half = num_bins // 2

    def quantize_resid(resid_flat):
        """Returns (bin_indices, n_outliers). Outliers (|resid|>abs_bound) → bin 0."""
        raw  = np.round(resid_flat.astype(np.float64) / bw).astype(np.int32)
        out_mask = np.abs(raw) >= half
        bins = np.clip(raw, -(half-1), half-1)
        bins[out_mask] = 0
        return bins, int(np.sum(out_mask))

    def lorenzo_outliers(data, abs_bound):
        """Count points SZ2 stores as raw float32: |data - Lorenzo_pred| > abs_bound.

        Uses the standard 3D Lorenzo predictor (same as SZ2's default mode):
          pred[i,j,k] = d[i-1,j,k] + d[i,j-1,k] + d[i,j,k-1]
                      - d[i-1,j-1,k] - d[i-1,j,k-1] - d[i,j-1,k-1]
                      + d[i-1,j-1,k-1]
        with 1-D / 2-D fallbacks at boundaries.
        """
        d = data.astype(np.float64)
        pred = np.empty_like(d)
        pred[0, 0, 0] = 0.0
        # edges
        pred[1:, 0, 0] = d[:-1, 0, 0]
        pred[0, 1:, 0] = d[0, :-1, 0]
        pred[0, 0, 1:] = d[0, 0, :-1]
        # faces
        pred[1:, 1:, 0] = d[:-1, 1:, 0] + d[1:, :-1, 0] - d[:-1, :-1, 0]
        pred[1:, 0, 1:] = d[:-1, 0, 1:] + d[1:, 0, :-1] - d[:-1, 0, :-1]
        pred[0, 1:, 1:] = d[0, :-1, 1:] + d[0, 1:, :-1] - d[0, :-1, :-1]
        # interior
        pred[1:, 1:, 1:] = (d[:-1, 1:, 1:] + d[1:, :-1, 1:] + d[1:, 1:, :-1]
                            - d[:-1, :-1, 1:] - d[:-1, 1:, :-1] - d[1:, :-1, :-1]
                            + d[:-1, :-1, :-1])
        return int(np.sum(np.abs(d - pred).ravel() > abs_bound))

    def plot_panel(ax, bins, method, n_outliers=0, subtitle=""):
        n_total = max(len(bins), 1)
        unique, counts = np.unique(bins, return_counts=True)
        c = COLORS.get(method, "gray")
        ax.bar(unique, counts, width=1.0, color=c, edgecolor="none", alpha=0.85)
        ax.set_xlim(-200, 200)
        ax.set_xlabel("Bin index", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        # bin[0] count includes genuine near-zero residuals AND outliers mapped to 0
        n_bin0 = int(counts[unique == 0][0]) if 0 in unique else 0
        n_true_zero = n_bin0 - n_outliers   # genuine near-zero predictions
        frac_zero = n_true_zero / n_total
        frac_out  = n_outliers  / n_total
        out_str = f"  |  {frac_out:.1%} raw outliers" if n_outliers > 0 else "  |  0 outliers"
        ax.set_title(f"{method}\nbin[0]={n_true_zero:,} ({frac_zero:.1%} accurate){out_str}{subtitle}",
                     fontsize=8)
        print(f"[Hist {method:12s}] accurate={n_true_zero:,} ({frac_zero:.1%})  "
              f"outliers={n_outliers:,} ({frac_out:.1%})  total={n_total:,}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    fig.suptitle(
        f"Residual bin-index histogram — k={k}  abs_bound={ab:.2g}  "
        f"bin_width={bw:.2g}  (all 6 methods, same data)\n"
        f"Spike at bin 0 = accurate prediction. Wide distribution = large residuals.",
        fontsize=9)

    # ── SZ2 ─────────────────────────────────────────────────────────────────
    ax = axes[0]
    try:
        recon, _, _, _ = run_libpressio(data_ds, "sz", {"pressio:abs": ab})
        bins, _ = quantize_resid((data_ds - recon).ravel())
        # SZ2 guarantees final errors ≤ abs_bound, so quantize_resid gives 0 outliers.
        # Instead estimate internal raw-storage fraction via the Lorenzo predictor.
        n_out = lorenzo_outliers(data_ds, ab)
        plot_panel(ax, bins, "SZ2", n_outliers=n_out, subtitle=f"\nabs={ab:.2g} (Lorenzo est.)")
    except Exception as e:
        ax.text(0.5, 0.5, f"SZ2 failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        print(f"[Hist SZ2] FAILED: {e}")

    # ── ZFP (accuracy=1e-2 — fixed reference point) ──────────────────────────
    ax = axes[1]
    try:
        recon, _, _, _ = run_libpressio(data_ds, "zfp", {"zfp:accuracy": 1e-2})
        bins, n_out = quantize_resid((data_ds - recon).ravel())
        plot_panel(ax, bins, "ZFP", n_outliers=n_out, subtitle=f"\naccuracy=1e-2")
    except Exception as e:
        ax.text(0.5, 0.5, f"ZFP failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        print(f"[Hist ZFP] FAILED: {e}")

    # ── DEIM-2D ──────────────────────────────────────────────────────────────
    ax = axes[2]
    try:
        mean_ds = data_ds.mean(axis=0)
        X_mat   = (data_ds - mean_ds).reshape(n_L, n_2D)
        _, _, Vt = np.linalg.svd(X_mat, full_matrices=False)
        Phi_k  = Vt[:k].T                             # (n_2D, k)
        _, _, piv = scipy_qr(Phi_k.T, pivoting=True)
        s_k    = np.sort(piv[:k])
        A      = Phi_k[s_k, :]                        # (k, k)
        all_sv = X_mat[:, s_k]                        # (n_L, k)
        recon_flat = np.zeros((n_L, n_2D))
        for lvl in range(n_L):
            recon_flat[lvl] = Phi_k @ np.linalg.solve(A, all_sv[lvl])
        recon_ds = recon_flat.reshape(n_L, ny, nz) + mean_ds
        bins, n_out = quantize_resid((data_ds - recon_ds).ravel())
        plot_panel(ax, bins, "DEIM-2D", n_outliers=n_out, subtitle=f"\nk={k}")
    except Exception as e:
        ax.text(0.5, 0.5, f"DEIM-2D failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        print(f"[Hist DEIM-2D] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Kriging-2D ───────────────────────────────────────────────────────────
    ax = axes[4]
    try:
        mgp_h = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
        X_grid = mgp_h.make_grid_coords(ny, nz)       # (n_2D, 2)

        data_nd   = data_ds.reshape(n_L, n_2D, 1)
        tr_mean   = data_nd.mean(axis=0)              # (n_2D, 1)
        tr_std    = data_nd.std(axis=0)
        tr_std    = np.where(tr_std < 1e-10, 1.0, tr_std)
        Y_list    = [(data_nd[l] - tr_mean) / tr_std for l in range(n_L)]

        # Mirror kriging_hurricane.py: random subset, 3 restarts, maxiter=200
        from scipy.optimize import minimize as _min
        from scipy.spatial.distance import cdist as _cdist_h
        _fit_sz  = min(k * 4, n_2D)
        _rng_fit = np.random.default_rng(0)
        _fit_idx = _rng_fit.choice(n_2D, size=_fit_sz, replace=False)
        _X_fit   = X_grid[_fit_idx]
        _Y_fit   = Y_list[0][_fit_idx].ravel()

        def nlml(lt):
            ls_, v_, n_ = (float(np.exp(lt[0])), float(np.exp(lt[1])),
                           float(np.exp(lt[2])))
            K = v_ * mgp_h.matern32(_X_fit, _X_fit, ls_) + n_*np.eye(_fit_sz)
            try:
                L_ = np.linalg.cholesky(K + 1e-8*np.eye(_fit_sz))
            except Exception:
                return 1e10
            ld    = 2*np.sum(np.log(np.diag(L_)))
            alpha = np.linalg.solve(L_.T, np.linalg.solve(L_, _Y_fit))
            return 0.5*float(_Y_fit @ alpha) + 0.5*ld

        _dists  = _cdist_h(_X_fit[:10], _X_fit[:10], 'euclidean')
        _ls0    = float(np.median(_dists[_dists > 0])) / 3.0
        _var0   = max(float(np.var(_Y_fit)), 1e-6)
        _bds    = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
        _rng_hp = np.random.default_rng(0)
        _sts    = [np.log([max(_ls0, 0.1), max(_var0, 0.01), _var0*0.01])] + [
                   _rng_hp.uniform([b[0] for b in _bds], [b[1] for b in _bds])
                   for _ in range(2)]
        _best_nll, _best = np.inf, [_ls0, _var0, _var0*0.01]
        for _x0 in _sts:
            try:
                _r = _min(nlml, _x0, method="L-BFGS-B", bounds=_bds,
                          options={"maxiter": 200, "ftol": 1e-9})
                if _r.fun < _best_nll:
                    _best_nll = _r.fun; _best = np.exp(_r.x).tolist()
            except Exception:
                pass
        ls, var, noise_var = _best[0], _best[1], _best[2]
        B = np.array([[var]])

        # RPCholesky sensor selection — no full n×n kernel matrix needed
        s_k  = _rpcholesky_sensors(X_grid, ls, k, mgp_h.matern32)
        K_Xs = var * mgp_h.matern32(X_grid, X_grid[s_k], ls)          # (n_2D, k)
        K_ss = var * mgp_h.matern32(X_grid[s_k], X_grid[s_k], ls)     # (k, k)
        Lk, lower = cho_factor(K_ss + noise_var*np.eye(k))

        all_sv = data_nd[:, s_k, 0]                  # (n_L, k)
        recon_flat = np.zeros((n_L, n_2D))
        for lvl in range(n_L):
            y_n = ((all_sv[lvl] - tr_mean[s_k, 0]) / tr_std[s_k, 0])
            alpha = cho_solve((Lk, lower), y_n)
            mu_n  = K_Xs @ alpha
            recon_flat[lvl] = mu_n * tr_std[:, 0] + tr_mean[:, 0]
        recon_ds = recon_flat.reshape(n_L, ny, nz)
        bins, n_out = quantize_resid((data_ds - recon_ds).ravel())
        plot_panel(ax, bins, "Kriging-2D", n_outliers=n_out, subtitle=f"\nk={k} ls={ls:.3f}")
    except Exception as e:
        ax.text(0.5, 0.5, f"Kriging-2D failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        print(f"[Hist Kriging-2D] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── T-DEIM ───────────────────────────────────────────────────────────────
    ax = axes[3]
    try:
        tdeim   = load_mod(str(ARGONNE / "lp_tdeim_compressor.py"), "lp_tdeim")
        mean_ds = data_ds.mean(axis=0)
        F       = (data_ds - mean_ds).reshape(n_L, n_2D)
        Phi, _  = tdeim.build_3d_basis(F, k)          # (n_3D_ds, k)
        sensors, _ = tdeim.qdeim_place(Phi)
        y_s     = (data_ds - mean_ds).ravel()[sensors] # exact sensor values
        recon_flat = tdeim.tdeim_reconstruct(Phi, sensors, y_s)
        recon_ds   = recon_flat.reshape(n_L, ny, nz) + mean_ds
        bins, n_out = quantize_resid((data_ds - recon_ds).ravel())
        plot_panel(ax, bins, "T-DEIM", n_outliers=n_out, subtitle=f"\nk={k}")
    except Exception as e:
        ax.text(0.5, 0.5, f"T-DEIM failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        print(f"[Hist T-DEIM] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── MultiGP ──────────────────────────────────────────────────────────────
    ax = axes[5]
    if data2_3d is None:
        ax.text(0.5, 0.5, "MultiGP: QVAPORf\nnot provided", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_title("MultiGP", fontsize=8)
        print("[Hist MultiGP] skipped — data2 not provided")
    else:
        try:
            mgp2  = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
            data2_ds = np.ascontiguousarray(data2_3d[:, ::ds, ::ds])
            # Stack both fields: (n_L, n_2D, 2)
            both  = np.stack([data_ds, data2_ds], axis=-1).reshape(n_L, n_2D, 2)
            X_g   = mgp2.make_grid_coords(ny, nz)
            tmean = both.mean(axis=0)              # (n_2D, 2)
            tstd  = both.std(axis=0)
            tstd  = np.where(tstd < 1e-10, 1.0, tstd)
            Y_list2 = [(both[l] - tmean) / tstd for l in range(n_L)]
            B2    = mgp2.estimate_B(Y_list2)       # (2, 2)
            noise_var2 = 0.05**2
            _HP   = 10
            ls2   = mgp2.fit_lengthscale(X_g[::_HP], [Y[::_HP] for Y in Y_list2],
                                         B2, noise_var2, n_restarts=1)
            sensors2 = _rpcholesky_sensors(X_g, ls2, k, mgp2.matern32)
            X_s2  = X_g[sensors2]
            # CLOUDf reconstruction only (field 0):
            # lmc_predict expects sensor observations shape (k, d), not (n_2D, d)
            recon_flat = np.zeros((n_L, n_2D))
            for lvl in range(n_L):
                Y_obs = Y_list2[lvl][sensors2]     # (k, 2) — sensor values only
                mu, _ = mgp2.lmc_predict(X_g, X_s2, Y_obs, B2, ls2, noise_var2)
                recon_flat[lvl] = (mu[:, 0] * tstd[:, 0] + tmean[:, 0])
            recon_ds = recon_flat.reshape(n_L, ny, nz)
            bins, n_out = quantize_resid((data_ds - recon_ds).ravel())
            plot_panel(ax, bins, "MultiGP", n_outliers=n_out, subtitle=f"\nk={k} (CLOUDf only)")
        except Exception as e:
            ax.text(0.5, 0.5, f"MultiGP failed:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            print(f"[Hist MultiGP] FAILED: {e}")
            import traceback; traceback.print_exc()

    plt.tight_layout()
    out = ARGONNE / f"rd_residual_histogram_{FIELD_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── CR-matched residual histogram ─────────────────────────────────────────────
def make_cr_matched_histogram(by_method, data_3d, data2_3d=None, ds=DS, num_bins=NUM_BINS):
    """Prediction-residual histogram where every method operates at the SAME compression ratio.

    Why this is fairer than the fixed-abs_bound histogram:
      The fixed-abs_bound histogram compares SZ2 at CR≈50 against our methods at
      CR≈2 — of course SZ2 looks better; it's achieving 25× more compression.
      Here we find the target CR (the highest CR all methods can reach), select
      the closest operating point for each method, and plot the absolute prediction
      errors on a COMMON x-axis.  A narrow, spike-heavy distribution means the
      predictor leaves small residuals at that CR; a wide distribution means the
      predictor is less accurate for the same storage budget.

    Residuals shown are PREDICTION errors only (data - model_reconstruction),
    before the quantized-residual correction term.  This isolates model quality
    from the trivial observation that both models bound final error by abs_bound.

    Panel subtitle shows: operating (k, abs_bound, CR) and CSV-stored PSNR/SSIM.
    """
    from scipy.linalg import cho_factor, cho_solve

    # ── Step 1: determine target CR from CSV ───────────────────────────────────
    peak_cr = {}
    for method in ALL_METHODS:
        pts = [p for p in by_method.get(method, [])
               if np.isfinite(p.get("cr", float("nan")))]
        if pts:
            peak_cr[method] = max(p["cr"] for p in pts)

    if not peak_cr:
        print("  [CR-matched hist] No CR data in CSV — run full compression first.")
        return

    target_cr = min(peak_cr.values())
    bottleneck = min(peak_cr, key=peak_cr.get)
    print(f"\n[CR-matched hist] target_cr = {target_cr:.2f}× "
          f"(bottleneck: {bottleneck}, peak={peak_cr[bottleneck]:.2f}×)")

    # ── Step 2: find closest operating point per method ───────────────────────
    op = {}   # method → best CSV row
    for method in ALL_METHODS:
        pts = [p for p in by_method.get(method, [])
               if np.isfinite(p.get("cr", float("nan")))]
        if not pts:
            print(f"  {method}: no data, skipping")
            continue
        best = min(pts, key=lambda p: abs(p["cr"] - target_cr))
        op[method] = best
        print(f"  {method:12s}: k={str(best.get('k')):>4s}  "
              f"ab={best.get('abs_bound', float('nan')):.2g}  "
              f"CR={best['cr']:.2f}×  "
              f"PSNR={best.get('psnr', float('nan')):.1f} dB  "
              f"SSIM={best.get('ssim', float('nan')):.4f}")

    # ── Step 3: re-run predictions at matched operating points ─────────────────
    data_ds = np.ascontiguousarray(data_3d[:, ::ds, ::ds])
    n_L, ny, nz = data_ds.shape
    n_2D = ny * nz

    # Collect per-method absolute prediction residuals
    residuals = {}   # method → 1-D absolute error array (float32)
    err_msgs  = {}   # method → error string

    # ── SZ2 ────────────────────────────────────────────────────────────────────
    if "SZ2" in op:
        try:
            ab_sz = op["SZ2"]["abs_bound"]
            recon, _, _, _ = run_libpressio(data_ds, "sz", {"pressio:abs": ab_sz})
            residuals["SZ2"] = np.abs((data_ds - recon).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["SZ2"] = str(e); print(f"  SZ2 failed: {e}")

    # ── ZFP ────────────────────────────────────────────────────────────────────
    if "ZFP" in op:
        try:
            rate_zfp = op["ZFP"]["abs_bound"]   # stored as abs_bound in the CSV
            recon, _, _, _ = run_libpressio(data_ds, "zfp", {"zfp:accuracy": rate_zfp})
            residuals["ZFP"] = np.abs((data_ds - recon).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["ZFP"] = str(e); print(f"  ZFP failed: {e}")

    # ── DEIM-2D ────────────────────────────────────────────────────────────────
    if "DEIM-2D" in op:
        try:
            k_d = int(op["DEIM-2D"]["k"])
            mean_ds = data_ds.mean(axis=0)
            X_mat   = (data_ds - mean_ds).reshape(n_L, n_2D)
            _, _, Vt = np.linalg.svd(X_mat, full_matrices=False)
            Phi_k   = Vt[:k_d].T
            _, _, piv = scipy_qr(Phi_k.T, pivoting=True)
            s_k     = np.sort(piv[:k_d])
            A       = Phi_k[s_k, :]
            all_sv  = X_mat[:, s_k]
            recon_flat = np.zeros((n_L, n_2D))
            for lvl in range(n_L):
                recon_flat[lvl] = Phi_k @ np.linalg.solve(A, all_sv[lvl])
            recon_ds = recon_flat.reshape(n_L, ny, nz) + mean_ds
            residuals["DEIM-2D"] = np.abs((data_ds - recon_ds).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["DEIM-2D"] = str(e); print(f"  DEIM-2D failed: {e}")
            import traceback; traceback.print_exc()

    # ── T-DEIM ─────────────────────────────────────────────────────────────────
    if "T-DEIM" in op:
        try:
            k_t    = int(op["T-DEIM"]["k"])
            tdeim  = load_mod(str(ARGONNE / "lp_tdeim_compressor.py"), "lp_tdeim")
            mean_ds = data_ds.mean(axis=0)
            F       = (data_ds - mean_ds).reshape(n_L, n_2D)
            Phi, _  = tdeim.build_3d_basis(F, k_t)
            sensors, _ = tdeim.qdeim_place(Phi)
            y_s     = (data_ds - mean_ds).ravel()[sensors]
            recon_flat = tdeim.tdeim_reconstruct(Phi, sensors, y_s)
            recon_ds   = recon_flat.reshape(n_L, ny, nz) + mean_ds
            residuals["T-DEIM"] = np.abs((data_ds - recon_ds).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["T-DEIM"] = str(e); print(f"  T-DEIM failed: {e}")
            import traceback; traceback.print_exc()

    # ── Kriging-2D ─────────────────────────────────────────────────────────────
    if "Kriging-2D" in op:
        try:
            k_krig = int(op["Kriging-2D"]["k"])
            mgp_h  = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
            X_grid = mgp_h.make_grid_coords(ny, nz)
            data_nd   = data_ds.reshape(n_L, n_2D, 1)
            tr_mean   = data_nd.mean(axis=0)
            tr_std    = data_nd.std(axis=0)
            tr_std    = np.where(tr_std < 1e-10, 1.0, tr_std)
            Y_list    = [(data_nd[l] - tr_mean) / tr_std for l in range(n_L)]
            # Mirror kriging_hurricane.py: random subset, 3 restarts, maxiter=200
            from scipy.optimize import minimize as _min
            from scipy.spatial.distance import cdist as _cdist_k
            _fit_sz2  = min(k_krig * 4, n_2D)
            _rng_fit2 = np.random.default_rng(0)
            _fit_idx2 = _rng_fit2.choice(n_2D, size=_fit_sz2, replace=False)
            _X_fit2   = X_grid[_fit_idx2]
            _Y_fit2   = Y_list[0][_fit_idx2].ravel()

            def nlml_k(lt):
                ls_, v_, n_ = (float(np.exp(lt[0])), float(np.exp(lt[1])),
                               float(np.exp(lt[2])))
                K = v_ * mgp_h.matern32(_X_fit2, _X_fit2, ls_) + n_*np.eye(_fit_sz2)
                try:
                    L_ = np.linalg.cholesky(K + 1e-8*np.eye(_fit_sz2))
                except Exception:
                    return 1e10
                ld    = 2*np.sum(np.log(np.diag(L_)))
                alpha = np.linalg.solve(L_.T, np.linalg.solve(L_, _Y_fit2))
                return 0.5*float(_Y_fit2 @ alpha) + 0.5*ld

            _dists2  = _cdist_k(_X_fit2[:10], _X_fit2[:10], 'euclidean')
            _ls02    = float(np.median(_dists2[_dists2 > 0])) / 3.0
            _var02   = max(float(np.var(_Y_fit2)), 1e-6)
            _bds2    = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]
            _rng_hp2 = np.random.default_rng(0)
            _sts2    = [np.log([max(_ls02, 0.1), max(_var02, 0.01), _var02*0.01])] + [
                        _rng_hp2.uniform([b[0] for b in _bds2], [b[1] for b in _bds2])
                        for _ in range(2)]
            _best_nll2, _best2 = np.inf, [_ls02, _var02, _var02*0.01]
            for _x02 in _sts2:
                try:
                    _r2 = _min(nlml_k, _x02, method="L-BFGS-B", bounds=_bds2,
                               options={"maxiter": 200, "ftol": 1e-9})
                    if _r2.fun < _best_nll2:
                        _best_nll2 = _r2.fun; _best2 = np.exp(_r2.x).tolist()
                except Exception:
                    pass
            ls, var, noise_var = _best2[0], _best2[1], _best2[2]
            s_k  = _rpcholesky_sensors(X_grid, ls, k_krig, mgp_h.matern32)
            K_Xs = var * mgp_h.matern32(X_grid, X_grid[s_k], ls)
            K_ss = var * mgp_h.matern32(X_grid[s_k], X_grid[s_k], ls)
            Lk, lower = cho_factor(K_ss + noise_var*np.eye(k_krig))
            all_sv = data_nd[:, s_k, 0]
            recon_flat = np.zeros((n_L, n_2D))
            for lvl in range(n_L):
                y_n   = (all_sv[lvl] - tr_mean[s_k, 0]) / tr_std[s_k, 0]
                alpha = cho_solve((Lk, lower), y_n)
                mu_n  = K_Xs @ alpha
                recon_flat[lvl] = mu_n * tr_std[:, 0] + tr_mean[:, 0]
            recon_ds = recon_flat.reshape(n_L, ny, nz)
            residuals["Kriging-2D"] = np.abs((data_ds - recon_ds).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["Kriging-2D"] = str(e); print(f"  Kriging-2D failed: {e}")
            import traceback; traceback.print_exc()

    # ── MultiGP ────────────────────────────────────────────────────────────────
    if "MultiGP" in op and data2_3d is not None:
        try:
            k_mg   = int(op["MultiGP"]["k"])
            mgp2   = load_mod(str(ARGONNE / "lp_multigp_compressor.py"), "lp_multigp")
            data2_ds = np.ascontiguousarray(data2_3d[:, ::ds, ::ds])
            both   = np.stack([data_ds, data2_ds], axis=-1).reshape(n_L, n_2D, 2)
            X_g    = mgp2.make_grid_coords(ny, nz)
            tmean  = both.mean(axis=0)
            tstd   = both.std(axis=0)
            tstd   = np.where(tstd < 1e-10, 1.0, tstd)
            Y_list2 = [(both[l] - tmean) / tstd for l in range(n_L)]
            B2     = mgp2.estimate_B(Y_list2)
            noise_var2 = 0.05**2
            _HP    = 10
            ls2    = mgp2.fit_lengthscale(X_g[::_HP], [Y[::_HP] for Y in Y_list2],
                                          B2, noise_var2, n_restarts=1)
            sensors2 = _rpcholesky_sensors(X_g, ls2, k_mg, mgp2.matern32)
            X_s2   = X_g[sensors2]
            recon_flat = np.zeros((n_L, n_2D))
            for lvl in range(n_L):
                Y_obs = Y_list2[lvl][sensors2]
                mu, _ = mgp2.lmc_predict(X_g, X_s2, Y_obs, B2, ls2, noise_var2)
                recon_flat[lvl] = mu[:, 0] * tstd[:, 0] + tmean[:, 0]
            recon_ds = recon_flat.reshape(n_L, ny, nz)
            residuals["MultiGP"] = np.abs((data_ds - recon_ds).ravel()).astype(np.float32)
        except Exception as e:
            err_msgs["MultiGP"] = str(e); print(f"  MultiGP failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 4: plot ───────────────────────────────────────────────────────────
    # Common x-axis: 95th-percentile of all errors (clip extreme outliers for readability)
    all_errs = np.concatenate([r for r in residuals.values()]) if residuals else np.array([1.0])
    x_max    = float(np.percentile(all_errs, 95))
    x_max    = max(x_max, 1e-8)   # guard against degenerate data

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    order = ["SZ2", "ZFP", "DEIM-2D", "T-DEIM", "Kriging-2D", "MultiGP"]

    fig.suptitle(
        f"Prediction residual histogram — MATCHED compression ratio  (target CR ≈ {target_cr:.1f}×)\n"
        f"Each method uses its closest (k, abs_bound) to target CR.  "
        f"X-axis: |data − prediction|, clipped at 95th-pct = {x_max:.3g}.\n"
        f"Narrow distribution = accurate predictor at this compression level.",
        fontsize=9)

    n_bins_plot = 200
    x_edges = np.linspace(0, x_max, n_bins_plot + 1)

    for ax, method in zip(axes, order):
        c = COLORS.get(method, "gray")
        row = op.get(method)

        if method in err_msgs:
            ax.text(0.5, 0.5, f"{method}\nFailed:\n{err_msgs[method]}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_title(method, fontsize=9)
            continue

        if method not in residuals:
            ax.text(0.5, 0.5, f"{method}\n(no data)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.set_title(method, fontsize=9)
            continue

        errs = residuals[method]
        counts, _ = np.histogram(errs, bins=x_edges)
        ax.bar(x_edges[:-1], counts, width=x_max / n_bins_plot,
               color=c, alpha=0.85, align="edge", edgecolor="none")
        ax.axvline(float(np.median(errs)), color="k", lw=1.2, ls="--",
                   label=f"median={np.median(errs):.3g}")
        ax.axvline(float(np.percentile(errs, 95)), color="k", lw=1.0, ls=":",
                   label=f"p95={np.percentile(errs, 95):.3g}")
        ax.set_xlim(0, x_max)
        ax.set_xlabel("|error|", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

        k_str  = str(row.get("k")) if row else "?"
        ab_str = f"{row['abs_bound']:.2g}" if row else "?"
        cr_str = f"{row['cr']:.1f}" if row else "?"
        psnr_v = row.get("psnr", float("nan")) if row else float("nan")
        ssim_v = row.get("ssim", float("nan")) if row else float("nan")
        frac_clipped = float(np.mean(errs > x_max))

        ax.set_title(
            f"{method}  CR={cr_str}×  k={k_str}  ab={ab_str}\n"
            f"PSNR={psnr_v:.1f} dB  SSIM={ssim_v:.4f}"
            + (f"  [{frac_clipped:.1%} clipped]" if frac_clipped > 0 else ""),
            fontsize=8)

    plt.tight_layout()
    out = ARGONNE / f"rd_cr_matched_histogram_{FIELD_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {DATA_PATH} ...")
    data = np.fromfile(DATA_PATH, dtype=np.float32).reshape(SHAPE)
    print(f"  CLOUDf : {data.shape}  {data.nbytes/1e6:.0f} MB  compress={COMPRESS_BACKEND}")

    if DATA_PATH2.exists():
        data2 = np.fromfile(DATA_PATH2, dtype=np.float32).reshape(SHAPE)
        print(f"  QVAPORf: {data2.shape}  {data2.nbytes/1e6:.0f} MB")
    else:
        print(f"  WARNING: {DATA_PATH2} not found — MultiGP will be skipped")
        data2 = None

    # Downsampled data — ALL methods run on this same array for a fair comparison.
    # .copy() is critical: data[:, ::DS, ::DS] is a non-contiguous strided view and
    # libpressio (SZ, ZFP) requires contiguous memory — passing a view causes silent
    # encode failures, which is why SZ2/ZFP would disappear from plots.
    data_ds = np.ascontiguousarray(data[:, ::DS, ::DS])
    n_3D_ds = data_ds.size
    print(f"  Downsampled for all methods: {data_ds.shape}  "
          f"{data_ds.nbytes/1e6:.1f} MB  (CR denominator = {n_3D_ds*4/1e6:.1f} MB)")

    # ── PLOTS_ONLY: skip compression, load existing CSV ───────────────────────
    csv_path = ARGONNE / f"rd_results_{FIELD_TAG}.csv"
    if PLOTS_ONLY:
        if not csv_path.exists():
            print(f"PLOTS_ONLY=True but {csv_path} not found — run without PLOTS_ONLY first.")
            sys.exit(1)
        print(f"\nPLOTS_ONLY mode: loading {csv_path} ...")
        all_results = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                # Cast numeric fields back from strings
                for fld in ("k", "num_bins", "n_outliers"):
                    try: row[fld] = int(float(row[fld])) if row[fld] not in ("", "None") else None
                    except: row[fld] = None
                for fld in ("abs_bound", "cr", "cr_no_model", "psnr", "ssim",
                            "psnr_cloud", "ssim_cloud", "psnr_qvapor", "ssim_qvapor",
                            "comp_sec", "decomp_sec", "train_sec",
                            "compressed_MB", "model_MB", "sv_MB", "resid_MB"):
                    try: row[fld] = float(row[fld]) if row[fld] not in ("", "None") else float("nan")
                    except: row[fld] = float("nan")
                all_results.append(row)
        print(f"  Loaded {len(all_results)} rows.")
        # Skip to plotting
        by_method = defaultdict(list)
        for r in all_results:
            by_method[r["method"]].append(r)
        make_rd_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_cr_{FIELD_TAG}.png")
        make_rd_plot(by_method, "ssim", "SSIM",       f"rd_ssim_cr_{FIELD_TAG}.png")
        make_bpv_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_bpv_{FIELD_TAG}.png")
        make_bpv_plot(by_method, "ssim", "SSIM",       f"rd_ssim_bpv_{FIELD_TAG}.png")
        make_absbound_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_absbound_{FIELD_TAG}.png")
        make_absbound_plot(by_method, "ssim", "SSIM",       f"rd_ssim_absbound_{FIELD_TAG}.png")
        make_timing_plot(by_method,                    f"rd_timing_{FIELD_TAG}.png")
        make_budget_breakdown(by_method,               f"rd_budget_breakdown_{FIELD_TAG}.png")
        print("\n[Histogram] Building residual bin histograms for k=10 ...")
        make_residual_histogram(data, data2_3d=data2, ds=DS, k=10, ab=1e-2, num_bins=65536)
        print("\n[CR-matched histogram] Building CR-matched residual histograms ...")
        make_cr_matched_histogram(by_method, data, data2_3d=data2, ds=DS, num_bins=NUM_BINS)
        print("\nDone (plots-only mode).")
        return

    all_results = []

    # ── SZ2 (on downsampled data) ─────────────────────────────────────────────
    print("\n[SZ2] Sweeping abs bounds (ds=10, 50×50/level) ...")
    for bound in ABS_BOUNDS:
        try:
            recon, cr, cs, dc = run_libpressio(data_ds, "sz", {"pressio:abs": bound})
            pv, sv = metrics(data_ds[LEVEL], recon[LEVEL])
            all_results.append({
                "method": "SZ2", "k": None, "abs_bound": bound,
                "cr": cr, "psnr": pv, "ssim": sv,
                "comp_sec": cs, "decomp_sec": dc, "train_sec": 0.0,
                "compressed_MB": float("nan"), "model_MB": 0.0, "n_outliers": 0
            })
            print(f"  abs={bound:.1e}  CR={cr:.1f}×  PSNR={pv:.1f} dB  SSIM={sv:.4f}")
        except Exception as e:
            print(f"  SKIP abs={bound:.1e}: {e}")

    # ── ZFP (on downsampled data) ─────────────────────────────────────────────
    print("\n[ZFP] Sweeping fixed-accuracy abs bounds (ds=10, 50×50/level) ...")
    for bound in ZFP_ABS_BOUNDS:
        try:
            recon, cr, cs, dc = run_libpressio(data_ds, "zfp", {"zfp:accuracy": bound})
            pv, sv = metrics(data_ds[LEVEL], recon[LEVEL])
            all_results.append({
                "method": "ZFP", "k": None, "abs_bound": bound,
                "cr": cr, "psnr": pv, "ssim": sv,
                "comp_sec": cs, "decomp_sec": dc, "train_sec": 0.0,
                "compressed_MB": float("nan"), "model_MB": 0.0, "n_outliers": 0
            })
            print(f"  ab={bound:.2g}  CR={cr:.1f}×  PSNR={pv:.1f} dB  SSIM={sv:.4f}")
        except Exception as e:
            print(f"  SKIP ab={bound:.2g}: {e}")

    # ── DEIM-2D ──────────────────────────────────────────────────────────────
    try:
        all_results.extend(run_deim_2d(data, DEIM2D_K_VALUES, ABS_BOUNDS, DS))
    except Exception as e:
        print(f"[DEIM-2D] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── T-DEIM ───────────────────────────────────────────────────────────────
    if RUN_TDEIM:
        try:
            all_results.extend(run_tdeim(data, TDEIM_K_VALUES, ABS_BOUNDS, DS_TDEIM))
        except Exception as e:
            print(f"[T-DEIM] FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("[T-DEIM] Skipped (RUN_TDEIM=False)")

    # ── Kriging-2D ───────────────────────────────────────────────────────────
    try:
        all_results.extend(run_kriging_2d(data, KRIG2D_K_VALUES, ABS_BOUNDS, DS))
    except Exception as e:
        print(f"[Kriging-2D] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── MultiGP (d=2: CLOUDf + QVAPORf) ─────────────────────────────────────
    if RUN_MULTIGP and data2 is not None:
        try:
            all_results.extend(run_multigp(data, data2, MULTIGP_K_VALUES, ABS_BOUNDS, DS))
        except Exception as e:
            print(f"[MultiGP] FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        if not RUN_MULTIGP:
            print("[MultiGP] Skipped (RUN_MULTIGP=False)")
        else:
            print("[MultiGP] Skipped (QVAPORf48 not found)")

    if not all_results:
        print("No results."); sys.exit(1)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = ARGONNE / f"rd_results_{FIELD_TAG}.csv"
    fields   = ["method", "k", "abs_bound", "sensor_mult", "num_bins", "cr", "cr_no_model", "psnr", "ssim",
                 "psnr_cloud", "ssim_cloud", "psnr_qvapor", "ssim_qvapor",
                 "comp_sec", "decomp_sec", "train_sec",
                 "compressed_MB", "model_MB", "sv_MB", "resid_MB", "n_outliers"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV → {csv_path}  ({len(all_results)} rows)")

    by_method = defaultdict(list)
    for r in all_results:
        by_method[r["method"]].append(r)

    make_rd_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_cr_{FIELD_TAG}.png")
    make_rd_plot(by_method, "ssim", "SSIM",       f"rd_ssim_cr_{FIELD_TAG}.png")
    make_bpv_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_bpv_{FIELD_TAG}.png")
    make_bpv_plot(by_method, "ssim", "SSIM",       f"rd_ssim_bpv_{FIELD_TAG}.png")
    make_absbound_plot(by_method, "psnr", "PSNR (dB)", f"rd_psnr_absbound_{FIELD_TAG}.png")
    make_absbound_plot(by_method, "ssim", "SSIM",       f"rd_ssim_absbound_{FIELD_TAG}.png")
    make_timing_plot(by_method,                    f"rd_timing_{FIELD_TAG}.png")
    make_budget_breakdown(by_method,               f"rd_budget_breakdown_{FIELD_TAG}.png")

    # Residual histogram: all 6 methods at k=10, one abs_bound
    print("\n[Histogram] Building residual bin histograms for k=10 ...")
    make_residual_histogram(data, data2_3d=data2, ds=DS, k=10, ab=1e-2, num_bins=65536)
    print("\n[CR-matched histogram] Building CR-matched residual histograms ...")
    make_cr_matched_histogram(by_method, data, data2_3d=data2, ds=DS, num_bins=NUM_BINS)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n── Method summary ───────────────────────────────────────────")
    print(f"  {'Method':12}  {'train(s)':>9}  {'model overhead':}")
    for m in ALL_METHODS:
        pts = by_method.get(m, [])
        if not pts:
            continue
        t = pts[0].get("train_sec", 0.0)
        # model overhead at median k and tightest abs_bound
        k_groups = defaultdict(list)
        for p in pts:
            k_groups[p.get("k", 0)].append(p)
        rows = []
        for k in sorted(k_groups):
            best = min(k_groups[k], key=lambda x: x.get("abs_bound", 0))
            rows.append((k, best.get("model_MB", 0), best.get("compressed_MB", float("nan")),
                         best["cr"], best["psnr"]))
        if rows:
            mid = rows[len(rows)//2]
            print(f"  {m:12}  {t:>9.1f}  "
                  f"k={mid[0]} model={mid[1]:.2f}MB total={mid[2]:.2f}MB "
                  f"CR={mid[3]:.1f}× PSNR={mid[4]:.1f}dB")

    print("\nDone.")


if __name__ == "__main__":
    main()
