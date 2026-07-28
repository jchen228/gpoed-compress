"""
lp_multigp_compressor.py  —  libpressio external compressor: Multi-output GP (LMC)

Algorithm: Linear Model of Coregionalization (ICM) with Kronecker structure
           (multigp_hurricane.py).
  Offline (train): Jointly compress d fields (e.g. U, V, W wind components).
                   1. Estimate d×d cross-field covariance matrix B from training
                      slices via sample covariance in z-score space.
                   2. Fit spatial Matérn-3/2 lengthscale via Kronecker marginal
                      likelihood (O(n³) per evaluation rather than O((nd)³)).
                   3. GKS sensor placement on K_spatial → k sensors (same locations
                      observed for ALL d fields simultaneously).
                   k chosen by POD energy criterion on the concatenated field
                   matrix (consistent with DEIM / Kriging).
  Online (compress): store k×d float32 sensor values (k sensors × d fields).
  Online (decompress): LMC posterior mean at all n spatial points for all d fields:
                       μ* = K_{*s}(B ⊗ K_{ss} + σ²I)^{-1} y_flat  (in z-score space)
                       un-normalised with per-location training mean/std.

Why multi-output?
-----------------
  U, V, W are physically coupled (pressure drives circulation, convection, etc.).
  A single multi-output GP captures cross-field correlations: observing U at a
  sensor simultaneously constrains V and W there — independent per-field GPs
  cannot do this.

Error-bound → sensor count
--------------------------
  k is chosen by the POD energy criterion on the concatenated training slices
  (all d fields stacked), consistent with DEIM and Kriging.

CLI
---
  python3 lp_multigp_compressor.py train \\
      --data-dir /path/to/100x500x500 \\
      --variables Uf48.bin.f32 Vf48.bin.f32 Wf48.bin.f32 \\
      --shape 100 500 500 \\
      --model /path/multigp_model.npz \\
      --error-bound 0.01 \\
      [--downsample 14] [--noise-std 0.05]

  python3 lp_multigp_compressor.py compress <input_dir_or_stacked.bin> <output.bin> \\
      --model /path/multigp_model.npz

  python3 lp_multigp_compressor.py decompress <compressed.bin> <output_dir_or_stacked.bin> \\
      --model /path/multigp_model.npz

Input convention for compress/decompress
-----------------------------------------
  The input is a stacked binary file: d consecutive float32 arrays of n values each.
  Layout: [var0_pt0 … var0_pt(n-1)  var1_pt0 … var1_pt(n-1)  …  vard_pt0 … ]
  Total size: d × n float32 values.
  To stack two files before calling compress:
    cat U.bin V.bin W.bin > UVW_stacked.bin

Compressed binary layout:
  4 bytes int32  k   (sensors)
  4 bytes int32  d   (fields)
  4 bytes int32  n   (spatial points)
  k*d*4 bytes float32  sensor obs in z-score space, row-major (k rows, d cols)

libpressio usage
----------------
  comp.set_options({
      "external:command":
          "python3 /path/lp_multigp_compressor.py compress --model /path/model.npz",
      "external:decompressor_command":
          "python3 /path/lp_multigp_compressor.py decompress --model /path/model.npz",
  })

References
----------
  Alvarez & Lawrence (2011). JMLR 12, 1459–1500.  (convolved multi-output GPs)
  Bonilla et al. (2008). NeurIPS. (multi-task GP prediction)
  Saatci (2011). PhD thesis.   (Kronecker GP structure)
"""

import argparse
import os
import struct
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

JITTER = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def matern32(X1, X2, ls):
    """Matérn-3/2 kernel (unit variance): (1 + √3 r/ℓ) exp(−√3 r/ℓ)."""
    r = cdist(X1, X2, 'euclidean')
    v = np.sqrt(3.0) * r / ls
    return (1.0 + v) * np.exp(-v)


def make_grid_coords(ny, nz):
    """(ny*nz, 2) grid coordinate array."""
    YY, ZZ = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    return np.column_stack([YY.ravel(), ZZ.ravel()])


def adaptive_k(singular_values, tol, max_k=None):
    """Smallest k s.t. sqrt(1 − Σs[:k]²/Σs²) ≤ tol."""
    s2    = singular_values ** 2
    cumul = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask  = proj_err <= tol
    k     = int(np.argmax(mask)) + 1 if mask.any() else len(singular_values)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


# ─────────────────────────────────────────────────────────────────────────────
# B-MATRIX ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_B(Y_train_list):
    """
    Estimate d×d cross-field covariance from list of (n, d) normalised arrays.

    B = (1/N) Σ_l Y_l^T Y_l, averaged over all training (level, location) pairs.
    """
    Y_all = np.vstack(Y_train_list)       # (N*n, d)
    B     = (Y_all.T @ Y_all) / len(Y_all)
    B     = 0.5 * (B + B.T) + JITTER * np.eye(B.shape[0])
    return B


# ─────────────────────────────────────────────────────────────────────────────
# LENGTHSCALE FITTING  (Kronecker marginal likelihood)
# ─────────────────────────────────────────────────────────────────────────────

def _neg_log_ml_kron(log_ls, X, Y_train_list, λ_B, Q_B, noise_var):
    """
    Negative log marginal likelihood using Kronecker structure B ⊗ K_n.

    Cost: O(n³) eigendecomp + O(n_train × nd) data transform.
    (vs O((nd)³) for naive full covariance — critical for large n or d.)
    """
    ls  = np.exp(log_ls)
    n   = len(X)
    d   = len(λ_B)
    K_n = matern32(X, X, ls) + JITTER * np.eye(n)
    try:
        λ_n, Q_n = np.linalg.eigh(K_n)
    except Exception:
        return 1e18

    Λ       = np.outer(λ_n, λ_B) + noise_var   # (n, d) eigenvalues
    log_det = np.sum(np.log(np.maximum(Λ, 1e-300)))
    quad    = sum(np.sum((Q_n.T @ Y @ Q_B) ** 2 / Λ) for Y in Y_train_list)
    n_obs   = len(Y_train_list)
    return 0.5 * (n_obs * (n * d * np.log(2 * np.pi) + log_det) + quad)


def fit_lengthscale(X, Y_train_list, B, noise_var, n_restarts=3):
    """
    Fit spatial lengthscale by maximising Kronecker marginal likelihood.
    """
    λ_B, Q_B = np.linalg.eigh(B)
    dists     = np.sqrt(((X[::5, None] - X[None, ::5]) ** 2).sum(-1)).ravel()
    ls0       = float(np.median(dists[dists > 0]))
    rng       = np.random.default_rng(0)
    starts    = [np.log(ls0)] + list(np.log(ls0) + rng.uniform(-1, 1, n_restarts - 1))

    best_nll, best_ls = np.inf, ls0
    for log_ls0 in starts:
        try:
            res = minimize(_neg_log_ml_kron, [log_ls0],
                           args=(X, Y_train_list, λ_B, Q_B, noise_var),
                           method='L-BFGS-B',
                           bounds=[(-1, np.log(max(X.max(), 10) * 3))],
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls  = float(np.exp(res.x[0]))
        except Exception:
            pass
    return best_ls


# ─────────────────────────────────────────────────────────────────────────────
# GKS SENSOR PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

def gks_sensors(K_spatial, k):
    """GKS: QR column-pivoting on K_spatial → k maximally informative locations."""
    from scipy.linalg import qr
    _, _, p = qr(K_spatial, pivoting=True)
    return p[:k]


# ─────────────────────────────────────────────────────────────────────────────
# LMC PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var):
    """
    LMC (ICM) posterior mean and variance.

    K_full = B ⊗ K_ss + σ²I  (kd × kd)
    μ_*    = K_{*s}(B ⊗ K_{ss}^{-1}) α    reshaped appropriately
    α      = K_full^{-1} y_flat

    Returns
    -------
    mu_norm  : (n, d) posterior mean in z-score space
    var_norm : (n, d) posterior marginal variance
    """
    n, d = len(X_all), B.shape[0]
    k    = len(X_sensors)

    K_ss  = matern32(X_sensors, X_sensors, ls) + JITTER * np.eye(k)
    K_Xs  = matern32(X_all,     X_sensors, ls)                        # (n, k)
    K_sub = np.kron(B, K_ss) + noise_var * np.eye(k * d)              # (kd, kd)
    y_flat = Y_obs_norm.T.ravel()                                      # (kd,)

    try:
        L, low = cho_factor(K_sub, lower=True)
        alpha  = cho_solve((L, low), y_flat)                           # (kd,)
    except np.linalg.LinAlgError:
        alpha  = np.linalg.solve(K_sub, y_flat)
        L, low = None, None

    alpha_mat = alpha.reshape(d, k)                                    # (d, k)
    mu_norm   = K_Xs @ (B @ alpha_mat).T                               # (n, d)

    var_norm = np.zeros((n, d))
    for i in range(d):
        K_cross_i = np.kron(B[i:i+1, :], K_Xs)   # (n, kd)
        if L is not None:
            V_i = cho_solve((L, low), K_cross_i.T)
        else:
            V_i = np.linalg.solve(K_sub, K_cross_i.T)
        var_norm[:, i] = np.maximum(B[i, i] - np.sum(K_cross_i * V_i.T, axis=1), 0.0)

    return mu_norm, var_norm


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def load_variable(data_dir, filename, shape, downsample):
    path = os.path.join(data_dir, filename)
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    return raw.reshape(shape)[:, ::downsample, ::downsample]


def train(data_dir, var_files, shape, model_path, error_bound,
          downsample=14, skip_levels=10, n_train_factor=4,
          noise_std=0.05, n_restarts=3):
    """
    Build Multi-output GP model for joint compression of d fields.

    Parameters
    ----------
    data_dir  : str          directory containing the .bin.f32 files
    var_files : list[str]    filenames for each field (e.g. ['Uf48.bin.f32', ...])
    shape     : tuple        full volume shape (n_levels, ny_full, nz_full)
    model_path: str          output .npz file
    error_bound: float       POD energy tolerance for choosing k
    downsample : int         spatial downsampling factor (14 → ~36×36 grid)
    skip_levels: int         skip leading levels (near-surface artefacts)
    """
    n_levels_full = shape[0]
    d             = len(var_files)
    var_names     = [os.path.splitext(v)[0].replace('f48', '') for v in var_files]
    print(f"[MultiGP train]  Fields: {var_names}  d={d}  ds={downsample}")

    # Load and stack all fields
    raw_list = []
    for vf in var_files:
        arr = load_variable(data_dir, vf, shape, downsample)   # (n_L, ny, nz)
        raw_list.append(arr[skip_levels:])
    data_tensor = np.stack([v.reshape(raw_list[0].shape[0], -1)
                             for v in raw_list], axis=-1)   # (n_L, n, d)
    n_L, n, _ = data_tensor.shape
    ny  = raw_list[0].shape[1]
    nz  = raw_list[0].shape[2]
    X_all = make_grid_coords(ny, nz)   # (n, 2)

    # ── Sensor count from POD on concatenated field matrix ─────────────────
    concat = data_tensor.reshape(n_L, n * d).T   # (n*d, n_L)
    _, s_pilot, _ = np.linalg.svd(concat, full_matrices=False)
    max_sensors   = max(1, n // 66)
    k_pilot, _    = adaptive_k(s_pilot, error_bound, max_sensors)

    n_train = n_train_factor * k_pilot
    n_train = min(n_train, n_L - 10)
    n_train = max(n_train, d + 2)

    rng       = np.random.default_rng(42)
    train_idx = np.sort(rng.choice(n_L, size=n_train, replace=False))
    test_mask = np.ones(n_L, dtype=bool)
    test_mask[train_idx] = False
    test_idx  = np.where(test_mask)[0]

    train_data = data_tensor[train_idx]   # (n_train, n, d)
    train_mean = train_data.mean(axis=0)  # (n, d)
    train_std  = train_data.std(axis=0)   # (n, d)
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)

    def normalise(Y): return (Y - train_mean) / train_std
    def denormalise(Z): return Z * train_std + train_mean

    # Final k from training SVD (same concatenated criterion)
    train_concat = train_data.reshape(n_train, n * d).T
    _, s_train, _ = np.linalg.svd(train_concat, full_matrices=False)
    k, proj_err   = adaptive_k(s_train, error_bound, max_sensors)
    print(f"  k={k}  proj_err={proj_err[k-1]*100:.3f}%  n_train={n_train}")

    Y_train_list = [normalise(train_data[l]) for l in range(n_train)]  # (n, d) each

    # ── Estimate B matrix ──────────────────────────────────────────────────────
    B    = estimate_B(Y_train_list)
    corr = B / np.sqrt(np.outer(np.diag(B), np.diag(B)))
    print(f"  B correlation (first 3 vars shown):")
    for i in range(min(d, 3)):
        print("    " + "  ".join(f"{corr[i,j]:+.3f}" for j in range(min(d, 3))))

    # ── Fit lengthscale ────────────────────────────────────────────────────────
    noise_var = noise_std ** 2
    print(f"  Fitting lengthscale (n_restarts={n_restarts})...")
    ls = fit_lengthscale(X_all, Y_train_list, B, noise_var, n_restarts)
    print(f"  Optimal lengthscale: {ls:.4f} grid units")

    # ── GKS sensor placement ───────────────────────────────────────────────────
    K_spatial = matern32(X_all, X_all, ls) + JITTER * np.eye(n)
    sensors   = gks_sensors(K_spatial, k)
    X_sensors = X_all[sensors]
    print(f"  Placed {k} sensors via GKS")

    # ── Precompute for fast online prediction ──────────────────────────────────
    K_ss  = matern32(X_sensors, X_sensors, ls) + JITTER * np.eye(k)
    K_Xs  = matern32(X_all,     X_sensors, ls)    # (n, k)
    K_sub = np.kron(B, K_ss) + noise_var * np.eye(k * d)   # (kd, kd)
    try:
        L_kd, low_kd = cho_factor(K_sub, lower=True)
        L_kd = np.tril(L_kd if low_kd else L_kd.T)
    except Exception:
        L_kd = np.linalg.cholesky(K_sub + 1e-4 * np.eye(k * d))

    # ── Test evaluation ────────────────────────────────────────────────────────
    errs = []
    for lvl in data_tensor[test_idx[:10]]:
        Y_obs_norm = normalise(lvl)[sensors]       # (k, d)
        mu_norm, _ = lmc_predict(X_all, X_sensors, Y_obs_norm, B, ls, noise_var)
        recon      = denormalise(mu_norm)           # (n, d)
        diff       = recon - lvl
        denom      = np.linalg.norm(lvl)
        if denom > 0:
            errs.append(np.linalg.norm(diff) / denom)
    errs = np.array(errs) if errs else np.array([np.nan])
    print(f"  Test rel-L2: median={np.nanmedian(errs):.4f}  max={np.nanmax(errs):.4f}")

    np.savez_compressed(
        model_path,
        sensors=sensors.astype(np.int32),
        train_mean=train_mean.astype(np.float32),    # (n, d)
        train_std=train_std.astype(np.float32),      # (n, d)
        K_Xs=K_Xs.astype(np.float32),               # (n, k) spatial cross-covariance
        L_kd=L_kd.astype(np.float32),               # (kd, kd) lower-triangular Cholesky
        B=B.astype(np.float32),                      # (d, d)
        k=np.int32(k),
        d=np.int32(d),
        n=np.int32(n),
        ny=np.int32(ny),
        nz=np.int32(nz),
        ls=np.float32(ls),
        noise_std=np.float32(noise_std),
        error_bound=np.float32(error_bound),
        var_names=np.array([v.encode() for v in var_names]),
        test_rel_l2_median=np.float32(np.nanmedian(errs)),
    )
    print(f"  Model saved → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def compress(input_path, output_path, model_path):
    """
    Compress a stacked slice (d×n float32 values, fields concatenated)
    to k×d normalised sensor observations.

    Input layout: [var0: n float32]  [var1: n float32]  …  [vard: n float32]
    """
    m          = np.load(model_path, allow_pickle=True)
    sensors    = m['sensors'].astype(np.int64)
    train_mean = m['train_mean'].astype(np.float64)   # (n, d)
    train_std  = m['train_std'].astype(np.float64)    # (n, d)
    k          = int(m['k'])
    d          = int(m['d'])
    n          = int(m['n'])

    raw = np.fromfile(input_path, dtype=np.float32).astype(np.float64)
    if raw.size != n * d:
        raise ValueError(f"Input has {raw.size} values; expected {n*d} (n={n}, d={d}).")

    Y_flat = raw.reshape(d, n).T           # (n, d)  row-major field layout
    Y_norm = (Y_flat - train_mean) / train_std   # normalise
    y_obs  = Y_norm[sensors].astype(np.float32)  # (k, d) sensor obs

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<iii', k, d, n))
        f.write(y_obs.ravel().tobytes())          # k*d float32, row-major

    ratio = (n * d * 4) / (12 + k * d * 4)
    print(f"  Compressed: {n*d}→{k*d} values  ratio={ratio:.1f}×", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def decompress(compressed_path, output_path, model_path):
    """
    Reconstruct full d-field result from compressed k×d sensor observations.

    Output layout: [var0: n float32]  [var1: n float32]  …  same as compress input.
    """
    m          = np.load(model_path, allow_pickle=True)
    K_Xs       = m['K_Xs'].astype(np.float64)        # (n, k)
    L_kd       = m['L_kd'].astype(np.float64)        # (kd, kd) lower-triangular
    B          = m['B'].astype(np.float64)            # (d, d)
    train_mean = m['train_mean'].astype(np.float64)   # (n, d)
    train_std  = m['train_std'].astype(np.float64)    # (n, d)
    n          = int(m['n'])
    d          = int(m['d'])

    with open(compressed_path, 'rb') as f:
        k_s, d_s, n_s = struct.unpack('<iii', f.read(12))
        y_obs = np.frombuffer(f.read(k_s * d_s * 4), dtype=np.float32).astype(np.float64)

    Y_obs_norm = y_obs.reshape(k_s, d_s)    # (k, d)
    kd         = k_s * d_s
    y_flat     = Y_obs_norm.T.ravel()       # (kd,) field-major

    # Posterior mean: μ_* = K_{*s}(B α_mat).T  where α_mat solves K_sub α = y_flat
    alpha    = np.linalg.solve(L_kd.T, np.linalg.solve(L_kd, y_flat))  # (kd,)
    alpha_mat = alpha.reshape(d_s, k_s)   # (d, k)
    mu_norm   = K_Xs @ (B @ alpha_mat).T  # (n, d)

    recon     = mu_norm * train_std + train_mean    # (n, d) de-normalised
    out_flat  = recon.T.ravel().astype(np.float32)  # [var0 vals, var1 vals, …]
    out_flat.tofile(output_path)
    print(f"  Decompressed: {k_s*d_s}→{n*d} values ({d} fields × {n} pts)",
          file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="libpressio external compressor: Multi-output GP (LMC/ICM)")
    sub = parser.add_subparsers(dest='mode', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data-dir',       required=True,
                         help='Directory containing .bin.f32 files')
    p_train.add_argument('--variables',      required=True, nargs='+',
                         metavar='FILE',
                         help='Field filenames, e.g. Uf48.bin.f32 Vf48.bin.f32')
    p_train.add_argument('--shape',          required=True, nargs='+', type=int,
                         metavar='DIM', help='Full volume shape, e.g. 100 500 500')
    p_train.add_argument('--model',          required=True)
    p_train.add_argument('--error-bound',    type=float, default=0.01)
    p_train.add_argument('--downsample',     type=int,   default=14)
    p_train.add_argument('--skip-levels',    type=int,   default=10)
    p_train.add_argument('--n-train-factor', type=int,   default=4)
    p_train.add_argument('--noise-std',      type=float, default=0.05)
    p_train.add_argument('--n-restarts',     type=int,   default=3)

    p_comp = sub.add_parser('compress',
                            help='Compress stacked d-field slice')
    p_comp.add_argument('input',  help='Stacked binary: d × n float32 values')
    p_comp.add_argument('output', help='Output compressed binary')
    p_comp.add_argument('--model', required=True)

    p_dec = sub.add_parser('decompress',
                           help='Reconstruct stacked d-field slice')
    p_dec.add_argument('input',  help='Compressed binary from compress step')
    p_dec.add_argument('output', help='Output stacked binary: d × n float32')
    p_dec.add_argument('--model', required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data_dir, args.variables, tuple(args.shape), args.model,
              args.error_bound, args.downsample, args.skip_levels,
              args.n_train_factor, args.noise_std, args.n_restarts)
    elif args.mode == 'compress':
        compress(args.input, args.output, args.model)
    elif args.mode == 'decompress':
        decompress(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
