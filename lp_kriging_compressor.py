"""
lp_kriging_compressor.py  —  libpressio external compressor: GP/Kriging

Algorithm: Simple Kriging with Matérn-3/2 spatial kernel (kriging_hurricane.py).
  Offline (train): POD energy criterion (same as DEIM) determines k (sensor count).
                   Hyperparameters (lengthscale, variance) estimated via marginal
                   likelihood maximisation on training slices.
                   GKS sensor placement: QR column-pivoting on the top-k right
                   singular vectors of the n×n kernel matrix (= CSSP / rpCSS).
  Online (compress): store k float32 sensor observations per slice.
  Online (decompress): Simple Kriging posterior mean at all n spatial locations:
                       μ* = m0 + K_{*s}(K_{ss} + σ²I)^{-1}(y_s − m0)
                       where m0 = training mean, K_{ss} = k×k sensor covariance,
                       K_{*s} = n×k cross-covariance.

Error-bound → sensor count
--------------------------
  k is chosen identically to DEIM for consistent comparison:
    sqrt(1 − Σs[:k]²/Σs²) ≤ error_bound  on the training slice SVD.
  Smaller error_bound → more modes → more sensors → better reconstruction.

CLI
---
  python3 lp_kriging_compressor.py train \\
      --data /path/Uf48.bin.f32 --shape 100 500 500 \\
      --model /path/kriging_model.npz --error-bound 0.01 \\
      [--downsample 3] [--kernel matern] [--matern-nu 1.5]

  python3 lp_kriging_compressor.py compress <input.bin> <output.bin> \\
      --model /path/kriging_model.npz

  python3 lp_kriging_compressor.py decompress <compressed.bin> <output.bin> \\
      --model /path/kriging_model.npz

libpressio usage
----------------
  comp.set_options({
      "external:command":
          "python3 /path/lp_kriging_compressor.py compress --model /path/model.npz",
      "external:decompressor_command":
          "python3 /path/lp_kriging_compressor.py decompress --model /path/model.npz",
  })

Compressed binary layout:
  4 bytes int32   k   (number of sensors)
  4 bytes int32   n   (number of spatial grid points)
  k*4 bytes float32  sensor observations (mean-subtracted if normalise=True)

References
----------
  Rasmussen & Williams (2006). Gaussian Processes for Machine Learning. MIT Press.
  Csató & Opper (2002). Sparse online Gaussian processes. Neural Computation.
"""

import argparse
import struct
import sys
from functools import partial

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def matern32(Xa, Xb, lengthscale=1.0, variance=1.0):
    """Matérn-3/2 kernel: (1 + √3 r/ℓ) exp(−√3 r/ℓ), scaled by variance."""
    r = cdist(Xa, Xb, 'euclidean')
    v = np.sqrt(3.0) * r / lengthscale
    return variance * (1.0 + v) * np.exp(-v)


def rbf(Xa, Xb, lengthscale=1.0, variance=1.0):
    """Squared-exponential (RBF) kernel: variance * exp(−½ r²/ℓ²)."""
    d2 = cdist(Xa, Xb, 'sqeuclidean')
    return variance * np.exp(-0.5 * d2 / lengthscale ** 2)


KERNELS = {'matern': matern32, 'rbf': rbf}


def get_kernel(name, nu=1.5):
    """Return kernel function by name; nu only used for Matérn."""
    if name == 'matern':
        if nu == 1.5:
            return matern32
        else:
            def _matern_nu(Xa, Xb, lengthscale=1.0, variance=1.0):
                from scipy.spatial.distance import cdist as _cdist
                r  = _cdist(Xa, Xb, 'euclidean')
                if nu == 0.5:
                    return variance * np.exp(-r / lengthscale)
                elif nu == 2.5:
                    v = np.sqrt(5.0) * r / lengthscale
                    return variance * (1.0 + v + v**2 / 3.0) * np.exp(-v)
                else:
                    raise ValueError(f"nu must be 0.5, 1.5, or 2.5; got {nu}")
            return _matern_nu
    elif name == 'rbf':
        return rbf
    else:
        raise ValueError(f"kernel must be 'matern' or 'rbf'; got '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE K (consistent with DEIM)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_k(singular_values, tol, max_k=None):
    """Smallest k s.t. POD projection error ≤ tol."""
    s2    = singular_values ** 2
    cumul = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask  = proj_err <= tol
    k     = int(np.argmax(mask)) + 1 if mask.any() else len(singular_values)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER FITTING  (marginal likelihood)
# ─────────────────────────────────────────────────────────────────────────────

def neg_log_ml(log_theta, X, Y_list, kernel_fn, noise_var):
    """
    Negative log marginal likelihood, averaged over training slices.

    log_theta = [log_lengthscale, log_variance]
    Each slice in Y_list is (n,) centred observations.
    """
    ls, var = np.exp(log_theta[0]), np.exp(log_theta[1])
    n       = len(X)
    K       = kernel_fn(X, X, ls, var) + noise_var * np.eye(n)
    try:
        L, low = cho_factor(K, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(L if low else L.T)))
    except np.linalg.LinAlgError:
        return 1e18

    nll = 0.0
    for y in Y_list:
        alpha  = cho_solve((L, low), y)
        nll   += 0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))
    return nll / len(Y_list)


def fit_hyperparams(X, Y_list, kernel_fn, noise_var, n_restarts=3):
    """
    MLE for lengthscale and variance via L-BFGS-B with random restarts.

    Returns (lengthscale, variance).
    """
    dists = np.sqrt(((X[::5, None] - X[None, ::5]) ** 2).sum(-1)).ravel()
    ls0   = float(np.median(dists[dists > 0]))
    var0  = float(np.mean([y.var() for y in Y_list]))
    rng   = np.random.default_rng(0)

    starts = [[np.log(ls0), np.log(max(var0, 1e-6))]]
    for _ in range(n_restarts - 1):
        starts.append([np.log(ls0) + rng.uniform(-1, 1),
                       np.log(max(var0, 1e-6)) + rng.uniform(-0.5, 0.5)])

    best_nll, best_ls, best_var = np.inf, ls0, var0
    for x0 in starts:
        try:
            res = minimize(neg_log_ml, x0,
                           args=(X, Y_list, kernel_fn, noise_var),
                           method='L-BFGS-B',
                           bounds=[(-1, np.log(max(X.max(), 10) * 3)), (-5, 5)],
                           options={'maxiter': 60, 'ftol': 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls, best_var = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
        except Exception:
            pass
    return best_ls, best_var


# ─────────────────────────────────────────────────────────────────────────────
# GKS SENSOR PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

def gks_sensors(K, k):
    """
    GKS: QR column-pivoting on top-k right singular vectors of K.

    Equivalent to CSSP in kriging_hurricane.py: finds k sensor locations
    whose kernel rows are most linearly independent.
    """
    from scipy.linalg import qr
    n    = K.shape[0]
    rank = min(k, n)
    try:
        from sklearn.utils.extmath import randomized_svd
        _, _, Vh = randomized_svd(K, n_components=rank, random_state=0)
    except ImportError:
        _, _, Vh = np.linalg.svd(K, full_matrices=False)
        Vh = Vh[:rank]
    _, _, p = qr(Vh, pivoting=True)
    return p[:k]


# ─────────────────────────────────────────────────────────────────────────────
# KRIGING PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def kriging_predict(X_all, X_sensors, y_sensors_norm, kernel_fn,
                    lengthscale, variance, noise_var, m0=0.0):
    """
    Simple Kriging posterior mean and variance.

    μ* = m0 + K_{*s}(K_{ss} + σ²I)^{-1}(y_s − m0)

    Returns
    -------
    mu  : (n,) posterior mean in normalised space
    var : (n,) posterior variance in normalised space
    """
    K_ss = kernel_fn(X_sensors, X_sensors, lengthscale, variance) \
           + noise_var * np.eye(len(X_sensors))
    K_xs = kernel_fn(X_all, X_sensors, lengthscale, variance)  # (n, k)

    try:
        L, low = cho_factor(K_ss, lower=True)
        alpha  = cho_solve((L, low), y_sensors_norm - m0)
        V      = cho_solve((L, low), K_xs.T)   # (k, n)
    except np.linalg.LinAlgError:
        K_ss_inv = np.linalg.inv(K_ss)
        alpha    = K_ss_inv @ (y_sensors_norm - m0)
        V        = K_ss_inv @ K_xs.T

    mu  = m0 + K_xs @ alpha
    var = np.maximum(variance - np.sum(K_xs * V.T, axis=1), 0.0)
    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def make_grid_coords(ny, nz):
    """2D spatial coordinates for an ny×nz grid, shape (n, 2)."""
    yi = np.arange(ny)
    zi = np.arange(nz)
    YY, ZZ = np.meshgrid(yi, zi, indexing='ij')
    return np.column_stack([YY.ravel(), ZZ.ravel()])


def train(data_path, shape, model_path, error_bound,
          downsample=1, skip_levels=10, n_train_factor=4,
          kernel='matern', matern_nu=1.5, noise_std=0.05, n_restarts=3):
    """
    Build Kriging model: fit hyperparameters, place GKS sensors.

    Sensor count k is determined by the POD energy criterion (same as DEIM)
    so that error_bound comparisons are apples-to-apples across methods.
    """
    print(f"[Kriging train]  Loading {data_path}  shape={shape}  ds={downsample}")
    with open(data_path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape(shape)[:, ::downsample, ::downsample]   # (n_L, ny, nz)
    data = data[skip_levels:]

    n_L, ny, nz = data.shape
    n = ny * nz
    X_all = make_grid_coords(ny, nz)   # (n, 2)

    max_sensors = max(1, n // 66)

    # ── POD energy criterion to choose k ─────────────────────────────────────
    pilot_mean = data.mean(axis=0)
    X_pilot    = (data - pilot_mean).reshape(n_L, n).T
    _, s_pilot, _ = np.linalg.svd(X_pilot, full_matrices=False)
    k_pilot, _ = adaptive_k(s_pilot, error_bound, max_sensors)

    n_train = n_train_factor * k_pilot
    n_train = min(n_train, n_L - 10)
    n_train = max(n_train, k_pilot + 2)

    rng       = np.random.default_rng(42)
    train_idx = np.sort(rng.choice(n_L, size=n_train, replace=False))
    test_mask = np.ones(n_L, dtype=bool)
    test_mask[train_idx] = False
    test_idx  = np.where(test_mask)[0]

    train_data = data[train_idx]    # (n_train, ny, nz)
    train_mean = train_data.mean(axis=0)  # (ny, nz)
    train_std  = train_data.std(axis=0)   # (ny, nz)
    train_std  = np.where(train_std < 1e-10, 1.0, train_std)

    def normalise(Y): return (Y - train_mean) / train_std
    def denormalise(Z): return Z * train_std + train_mean

    # Final k from training SVD
    train_c = train_data - train_mean
    X_snap  = train_c.reshape(n_train, n).T
    _, s_train, _ = np.linalg.svd(X_snap, full_matrices=False)
    k, proj_err   = adaptive_k(s_train, error_bound, max_sensors)
    print(f"  k={k}  proj_err={proj_err[k-1]*100:.3f}%  n_train={n_train}")

    # ── Fit hyperparameters via MLE ───────────────────────────────────────────
    kernel_fn  = get_kernel(kernel, matern_nu)
    noise_var  = noise_std ** 2
    Y_list     = [normalise(train_data[i]).ravel() for i in range(n_train)]
    print(f"  Fitting hyperparameters ({kernel}, n_restarts={n_restarts})...")
    ls, var    = fit_hyperparams(X_all, Y_list, kernel_fn, noise_var, n_restarts)
    print(f"  Optimal: lengthscale={ls:.4f}  variance={var:.4f}")

    # ── GKS sensor placement ──────────────────────────────────────────────────
    K_full  = kernel_fn(X_all, X_all, ls, var) + 1e-6 * np.eye(n)
    sensors = gks_sensors(K_full, k)
    X_sens  = X_all[sensors]
    print(f"  Placed {k} sensors via GKS")

    # ── Evaluate on test levels ───────────────────────────────────────────────
    test_data = data[test_idx]
    errs      = []
    for lvl in test_data[:20]:          # spot-check 20 test levels
        y_norm   = normalise(lvl).ravel()
        mu_norm, _ = kriging_predict(X_all, X_sens, y_norm[sensors],
                                     kernel_fn, ls, var, noise_var)
        recon    = denormalise(mu_norm.reshape(ny, nz))
        diff     = recon - lvl
        denom    = np.linalg.norm(lvl.ravel())
        if denom > 0:
            errs.append(np.linalg.norm(diff.ravel()) / denom)
    errs = np.array(errs) if errs else np.array([np.nan])
    print(f"  Test rel-L2: median={np.nanmedian(errs):.4f}  max={np.nanmax(errs):.4f}")

    # Pre-compute K_xs (n×k) and L (Cholesky of K_ss) for fast online prediction
    K_ss    = kernel_fn(X_sens, X_sens, ls, var) + noise_var * np.eye(k)
    K_xs    = kernel_fn(X_all,  X_sens, ls, var)   # (n, k)
    L_ss, _ = cho_factor(K_ss, lower=True)
    # Store L as a flat lower-triangular array
    L_lower = np.tril(L_ss)   # (k, k)

    np.savez_compressed(
        model_path,
        sensors=sensors.astype(np.int32),
        train_mean=train_mean.astype(np.float32),
        train_std=train_std.astype(np.float32),
        K_xs=K_xs.astype(np.float32),           # (n, k) cross-covariance
        L_ss=L_lower.astype(np.float32),         # (k, k) Cholesky of K_ss
        k=np.int32(k),
        n=np.int32(n),
        ny=np.int32(ny),
        nz=np.int32(nz),
        lengthscale=np.float32(ls),
        variance=np.float32(var),
        noise_std=np.float32(noise_std),
        kernel=np.bytes_(kernel.encode()),
        matern_nu=np.float32(matern_nu),
        error_bound=np.float32(error_bound),
        test_rel_l2_median=np.float32(np.nanmedian(errs)),
    )
    print(f"  Model saved → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def compress(input_path, output_path, model_path):
    """
    Compress one 2D slice: extract k sensor values (in normalised space).

    Output layout: header (k, n as int32) + k float32 normalised sensor values.
    """
    m          = np.load(model_path, allow_pickle=True)
    sensors    = m['sensors'].astype(np.int64)
    train_mean = m['train_mean'].astype(np.float64)
    train_std  = m['train_std'].astype(np.float64)
    k          = int(m['k'])
    n          = int(m['n'])

    data  = np.fromfile(input_path, dtype=np.float32).astype(np.float64)
    if data.size != n:
        raise ValueError(f"Input has {data.size} elements; model expects {n}.")

    y_norm    = (data - train_mean.ravel()) / train_std.ravel()
    y_sensors = y_norm[sensors].astype(np.float32)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<ii', k, n))
        f.write(y_sensors.tobytes())

    ratio = (n * 4) / (8 + k * 4)
    print(f"  Compressed: {n}→{k} values  ratio={ratio:.1f}×", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def decompress(compressed_path, output_path, model_path):
    """
    Reconstruct full 2D slice via Kriging posterior mean.

    μ* = K_{*s} L^{-T} L^{-1} y_s   (using stored Cholesky factor L)
    """
    m          = np.load(model_path, allow_pickle=True)
    K_xs       = m['K_xs'].astype(np.float64)       # (n, k)
    L_ss       = m['L_ss'].astype(np.float64)       # (k, k) lower triangular
    train_mean = m['train_mean'].astype(np.float64)
    train_std  = m['train_std'].astype(np.float64)
    n          = int(m['n'])
    ny         = int(m['ny'])
    nz         = int(m['nz'])

    with open(compressed_path, 'rb') as f:
        k_stored, n_stored = struct.unpack('<ii', f.read(8))
        y_sensors = np.frombuffer(f.read(k_stored * 4), dtype=np.float32).astype(np.float64)

    # Solve K_ss @ alpha = y_sensors using stored Cholesky
    # L_ss is lower triangular: K_ss = L_ss L_ss^T
    alpha   = np.linalg.solve(L_ss.T, np.linalg.solve(L_ss, y_sensors))
    mu_norm = K_xs @ alpha                          # (n,) normalised posterior mean
    recon   = (mu_norm * train_std.ravel() + train_mean.ravel()).astype(np.float32)
    recon.tofile(output_path)
    print(f"  Decompressed: {k_stored}→{n} values", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="libpressio external compressor: GP Kriging")
    sub = parser.add_subparsers(dest='mode', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data',           required=True)
    p_train.add_argument('--shape',          required=True, nargs='+', type=int)
    p_train.add_argument('--model',          required=True)
    p_train.add_argument('--error-bound',    type=float, default=0.01)
    p_train.add_argument('--downsample',     type=int,   default=1)
    p_train.add_argument('--skip-levels',    type=int,   default=10)
    p_train.add_argument('--n-train-factor', type=int,   default=4)
    p_train.add_argument('--kernel',         default='matern',
                         choices=['matern', 'rbf'])
    p_train.add_argument('--matern-nu',      type=float, default=1.5)
    p_train.add_argument('--noise-std',      type=float, default=0.05)
    p_train.add_argument('--n-restarts',     type=int,   default=3)

    p_comp = sub.add_parser('compress')
    p_comp.add_argument('input')
    p_comp.add_argument('output')
    p_comp.add_argument('--model', required=True)

    p_dec = sub.add_parser('decompress')
    p_dec.add_argument('input')
    p_dec.add_argument('output')
    p_dec.add_argument('--model', required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data, tuple(args.shape), args.model, args.error_bound,
              args.downsample, args.skip_levels, args.n_train_factor,
              args.kernel, args.matern_nu, args.noise_std, args.n_restarts)
    elif args.mode == 'compress':
        compress(args.input, args.output, args.model)
    elif args.mode == 'decompress':
        decompress(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
