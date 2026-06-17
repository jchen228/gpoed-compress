"""
kriging_hurricane.py

GP interpolation and sensor placement applied to the ISABEL hurricane dataset.
Extracted and adapted from Kriging-DEIM.ipynb.

GP methods:
    - Simple Kriging       (known constant mean)
    - Ordinary Kriging     (unknown constant mean)
    - Universal Kriging    (unknown linear trend)

Sensor placement methods:
    - CSSP / GKS           (SVD + QR pivoting on covariance matrix)
    - MaxMin ordering      (farthest-point geometric spreading)
    - Greedy error         (forward selection minimising reconstruction error)
    NOTE: RPCholesky + CSSP to be added separately.

Usage:
    python kriging_hurricane.py

Requirements:
    pip install numpy scipy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ←  only section you need to edit
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "/Users/jchen228/Desktop/Argonne/100x500x500/Uf48.bin.f32"
SLICE_LEVEL = 50    # which vertical level to slice (0–99)
DOWNSAMPLE  = 7    # take every Nth pixel; 10 → 50×50 = 2500 candidate points
N_SENSORS   = 75    # number of sensors to place
LENGTHSCALE = 5.0   # kernel lengthscale in downsampled grid-index units
VARIANCE    = 10.0   # kernel signal variance
NOISE       = 1e-3  # observation noise / regularisation nugget
KERNEL      = 'matern' # 'rbf'  or  'matern'  ← change this to switch kernels
MATERN_NU   = 0.5   # Matérn smoothness — only used when KERNEL='matern'
                    # choices: 0.5 (rough/exponential), 1.5, 2.5 (smooth)
FIT_ON      = 'random' # which sensors to use when fitting hyperparameters:
                    #   'maxmin'  — farthest-point spread; kernel-free (recommended)
                    #   'uniform' — evenly spaced grid subset; kernel-free
                    #   'random'  — random sample; kernel-free
                    #   'greedy'  — uses manual LENGTHSCALE/VARIANCE/NOISE to place
                    #               sensors first, then fits on them
NORMALIZE   = True  # subtract mean and divide by std before Kriging, then
                    # undo afterward. Helps the optimizer and stabilises the
                    # covariance matrix when the data has a large offset or range.


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_isabel_slice(path, slice_level=50, downsample=1):
    """
    Load a single 2D horizontal slice from an ISABEL .bin.f32 file.

    Parameters
    ----------
    path        : str   path to the .bin.f32 file
    slice_level : int   which x-level (0–99) to extract
    downsample  : int   keep every Nth pixel in both y and z directions

    Returns
    -------
    slice_2d : (ny, nz) float32 array
    """
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    # Full volume is (100, 500, 500): 100 vertical levels × 500 lat × 500 lon
    data = data.reshape((100, 500, 500))

    # Extract one horizontal slice and optionally downsample
    slice_2d = data[slice_level, :, :]          # shape (500, 500)
    slice_2d = slice_2d[::downsample, ::downsample]
    return slice_2d


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COVARIANCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rbf_cov(Xa, Xb, lengthscale=1.0, variance=1.0):
    """
    Isotropic RBF (squared-exponential / Gaussian) covariance kernel.

    Works for any spatial dimension d.
    Xa : (n, d) array of n points
    Xb : (m, d) array of m points
    Returns (n, m) kernel matrix K where K[i,j] = variance * exp(-0.5 * ||Xa[i]-Xb[j]||^2 / ls^2)

    Equivalent to MATLAB: gaussKern(Xa, Xb, sqrt(variance), lengthscale)
    Uses scipy cdist instead of MATLAB pdist2 — see MATLAB_to_Python_notes.txt.
    """
    d2 = cdist(Xa, Xb, 'sqeuclidean')
    return variance * np.exp(-0.5 * d2 / lengthscale ** 2)


def matern_cov(Xa, Xb, lengthscale=1.0, variance=1.0, nu=2.5):
    """
    Isotropic Matérn covariance kernel.

    The Matérn kernel controls how smooth the GP is via the parameter nu (ν).
    Unlike RBF (which produces infinitely smooth functions), Matérn allows
    rougher, more realistic fields. A smaller ν means rougher/spikier functions.

    Xa : (n, d) array of n points
    Xb : (m, d) array of m points
    nu : smoothness parameter — must be 0.5, 1.5, or 2.5

    Formulas  (r = Euclidean distance between points):
        ν=0.5:  K = variance * exp(-r / ls)
                  → also called the Ornstein-Uhlenbeck kernel; very rough
        ν=1.5:  K = variance * (1 + √3·r/ls) * exp(-√3·r/ls)
                  → once-differentiable; moderately rough
        ν=2.5:  K = variance * (1 + √5·r/ls + 5r²/(3·ls²)) * exp(-√5·r/ls)
                  → twice-differentiable; nearly as smooth as RBF

    The RBF kernel is the ν→∞ limit.

    MATLAB note: no built-in Matérn — you would write the formula manually,
    same as here. The key difference from RBF is using Euclidean distance (r)
    rather than squared distance (r²).
    """
    r = cdist(Xa, Xb, 'euclidean')   # Euclidean distance (NOT squared)
    if nu == 0.5:
        return variance * np.exp(-r / lengthscale)
    elif nu == 1.5:
        s = np.sqrt(3) * r / lengthscale
        return variance * (1.0 + s) * np.exp(-s)
    elif nu == 2.5:
        s = np.sqrt(5) * r / lengthscale
        return variance * (1.0 + s + s**2 / 3.0) * np.exp(-s)
    else:
        raise ValueError(f"nu must be 0.5, 1.5, or 2.5; got {nu}")


def get_kernel(kernel_name='rbf', nu=2.5):
    """
    Return a covariance function by name.

    Usage:
        cov_fn = get_kernel(KERNEL, MATERN_NU)
        C = cov_fn(Xa, Xb, lengthscale=LENGTHSCALE, variance=VARIANCE)

    Parameters
    ----------
    kernel_name : 'rbf' or 'matern'
    nu          : Matérn smoothness (only used when kernel_name='matern')

    Returns a callable with signature f(Xa, Xb, lengthscale, variance).
    """
    if kernel_name == 'rbf':
        return rbf_cov
    elif kernel_name == 'matern':
        # Wrap matern_cov so nu is baked in; signature matches rbf_cov
        def _matern(Xa, Xb, lengthscale=1.0, variance=1.0):
            return matern_cov(Xa, Xb, lengthscale, variance, nu=nu)
        return _matern
    else:
        raise ValueError(f"kernel_name must be 'rbf' or 'matern'; got '{kernel_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KRIGING / GP METHODS
# ─────────────────────────────────────────────────────────────────────────────

def simple_kriging(X_train, y_train, X_test,
                   m0=0.0, lengthscale=1.0, variance=1.0, noise=1e-6,
                   kernel=None):
    """
    Simple Kriging  —  assumes the mean is a known constant m0 (default 0).

    Parameters
    ----------
    X_train     : (n, d) sensor locations
    y_train     : (n,)   observed values at sensors
    X_test      : (m, d) prediction locations
    m0          : known prior mean
    lengthscale : kernel length scale
    variance    : kernel signal variance
    noise       : observation noise / nugget added to diagonal of C
    kernel      : covariance function f(Xa, Xb, lengthscale, variance) → matrix
                  defaults to rbf_cov; pass matern_cov or get_kernel(...) result

    Returns
    -------
    mean : (m,) posterior mean at X_test
    var  : (m,) posterior variance at X_test
    C    : (n, n) training covariance matrix (returned for reuse in CSSP)
    """
    if kernel is None:
        kernel = rbf_cov
    C     = kernel(X_train, X_train, lengthscale, variance) + noise * np.eye(len(X_train))
    C_inv = np.linalg.inv(C)
    c     = kernel(X_test, X_train, lengthscale, variance)     # (m, n) cross-covariance
    alpha = C_inv @ (y_train - m0)
    mean  = m0 + c @ alpha
    k_star = variance * np.ones(len(X_test))
    var   = k_star - np.sum(c @ C_inv * c, axis=1)
    return mean, np.maximum(var, 1e-8), C


def _build_G(X, basis_funcs):
    """
    Design matrix: each column is one basis function evaluated at all rows of X.
    X           : (n, d) locations
    basis_funcs : list of callables f(X) → (n,)
    Returns (n, p) matrix where p = number of basis functions.
    """
    return np.column_stack([f(X) for f in basis_funcs])


def universal_kriging(X_train, y_train, X_test, basis_funcs,
                      lengthscale=1.0, variance=1.0, noise=1e-6,
                      kernel=None):
    """
    Universal / Ordinary Kriging  —  estimates trend coefficients via GLS.

    For Ordinary Kriging pass:
        basis_funcs = [lambda X: np.ones(len(X))]

    For Universal Kriging with a linear trend in the first spatial dimension:
        basis_funcs = [lambda X: np.ones(len(X)), lambda X: X[:, 0]]

    Parameters
    ----------
    X_train     : (n, d) sensor locations
    y_train     : (n,)   observed values
    X_test      : (m, d) prediction locations
    basis_funcs : list of callables defining the trend basis
    lengthscale, variance, noise : kernel / noise hyperparameters
    kernel      : covariance function — same as in simple_kriging

    Returns
    -------
    mean : (m,) posterior mean
    var  : (m,) posterior variance
    """
    if kernel is None:
        kernel = rbf_cov
    C     = kernel(X_train, X_train, lengthscale, variance) + noise * np.eye(len(X_train))
    c     = kernel(X_test,  X_train, lengthscale, variance)
    C_inv = np.linalg.inv(C)

    G      = _build_G(X_train, basis_funcs)   # (n, p) design matrix at sensors
    g_star = _build_G(X_test,  basis_funcs)   # (m, p) design matrix at test points

    GT_Cinv = G.T @ C_inv
    M       = GT_Cinv @ G                     # (p, p) GLS normal equations matrix
    beta    = np.linalg.solve(M, GT_Cinv @ y_train)  # GLS trend coefficients
    r       = y_train - G @ beta             # residuals after trend removal

    mean     = g_star @ beta + c @ (C_inv @ r)
    k_star   = variance * np.ones(len(X_test))
    base_var = k_star - np.sum(c * (C_inv @ c.T).T, axis=1)
    diff     = g_star - (GT_Cinv @ c.T).T
    Minv     = np.linalg.inv(M)
    extra    = np.einsum('ij,jk,ik->i', diff, Minv, diff)   # GLS variance correction
    return mean, np.maximum(base_var + extra, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SENSOR PLACEMENT METHODS
# ─────────────────────────────────────────────────────────────────────────────

def cssp(C, k):
    """
    Column Subset Selection (CSSP / GKS).

    Selects k sensor locations by finding the k columns of the covariance
    matrix that best span its dominant subspace. Equivalent to gks.m in the
    MATLAB codebase.

    Algorithm:
        1. Randomized SVD → top-k right singular vectors Vh  (k × n)
        2. QR with column pivoting on Vh → pivot order p
        3. First k entries of p are the selected indices

    Parameters
    ----------
    C : (n, n) covariance matrix
    k : number of sensors to select

    Returns
    -------
    pk : (k,) int array of selected indices (0-based)
    """
    _, _, Vh = randomized_svd(C, n_components=k, random_state=0)
    _, _, p  = scipy.linalg.qr(Vh, pivoting=True)
    return p[:k]


def rp_cssp(X, k, cov_fn, lengthscale=1.0, variance=1.0, rank=None, rng=None):
    """
    Scalable sensor placement via Randomly Pivoted Cholesky + RPGKS.

    This is the method from gpoed-code-python (rpgks.py + pivoted_cholesky.py),
    combined here for convenience.

    Why it's better than plain CSSP for large n
    --------------------------------------------
    CSSP (above) needs the full n×n covariance matrix in memory. For n=2500
    that's 50 MB — manageable. For n=250,000 (no downsampling) it's 500 GB —
    impossible. RPCholesky avoids this by building only a low-rank factor
    F (n × rank) that approximates K ≈ F @ F.T, then running CSSP on F.

    Algorithm
    ---------
    1. RPCholesky: randomly pivot through columns of K, building F column by
       column. Each step costs O(n) kernel evaluations (one column of K),
       not O(n²). Total cost: O(n × rank) kernel evaluations.
    2. RPGKS: economy SVD of F → top-k left singular vectors U_k (n × k).
       QR with column pivoting on U_k.T → first k pivots are sensor indices.

    Parameters
    ----------
    X          : (n, d) candidate locations
    k          : number of sensors to select
    cov_fn     : covariance function f(Xa, Xb, lengthscale, variance) → matrix
    lengthscale: kernel lengthscale
    variance   : kernel signal variance
    rank       : rank of the Cholesky approximation (default: min(3*k, n))
                 Higher rank = more accurate approximation of K, but slower.
    rng        : numpy random Generator for reproducibility

    Returns
    -------
    pk : (k,) int array of selected indices (0-based)
    F  : (n, rank) low-rank Cholesky factor (K ≈ F @ F.T)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if rank is None:
        rank = min(3 * k, len(X))

    n = len(X)

    # ── Step 1: Randomly Pivoted Cholesky ─────────────────────────────────────
    # Build F column by column; each column requires one row/column of K.
    # kernel(pts, pts[[si]]) evaluates K(all, pivot) — one kernel column.
    diags = variance * np.ones(n)   # diagonal of K (= variance for RBF/Matérn at r=0)
    F     = np.zeros((n, rank))
    for i in range(rank):
        total = diags.sum()
        if total <= 0:
            rank = i   # early termination: matrix is already well approximated
            F = F[:, :rank]
            break
        weights = diags / total
        si      = int(rng.choice(n, p=weights))           # random pivot

        g = cov_fn(X, X[[si]], lengthscale, variance).ravel()  # (n,) kernel column
        if i > 0:
            g = g - F[:, :i] @ F[si, :i]                 # subtract accumulated rank

        pivot_val = g[si]
        if pivot_val <= 0:
            rank = i
            F = F[:, :rank]
            break
        F[:, i] = g / np.sqrt(pivot_val)

        diags = np.maximum(diags - F[:, i] ** 2, 0)      # update residual diagonal

    # ── Step 2: RPGKS — SVD of F, then QR pivoting ───────────────────────────
    # Economy SVD: U is (n, rank), columns are left singular vectors of F
    U, _, _ = np.linalg.svd(F, full_matrices=False)
    u_k     = U[:, :k]                                    # (n, k) top-k vectors
    _, _, p = scipy.linalg.qr(u_k.T, pivoting=True)       # column-pivoted QR
    pk      = p[:k]

    return pk, F


def maxmin_ordering(points, k):
    """
    Farthest-point (MaxMin) sensor placement.

    Greedily picks the point that is furthest from all already-selected
    points. Purely geometric — no kernel or data required.

    Parameters
    ----------
    points : (n, d) array of candidate locations
    k      : number of sensors to select

    Returns
    -------
    chosen : (k,) int array of selected indices (0-based)
    """
    n         = len(points)
    selected  = np.zeros(n, dtype=bool)
    remaining = np.ones(n,  dtype=bool)

    # Start from the first point
    selected[0]  = True
    remaining[0] = False
    distances    = np.linalg.norm(points - points[0], axis=1)
    chosen       = [0]

    for _ in range(1, k):
        idx              = np.argmax(distances * remaining)
        selected[idx]    = True
        remaining[idx]   = False
        chosen.append(idx)
        # Update min-distances for all remaining points
        diff             = points[remaining] - points[idx]
        distances[remaining] = np.minimum(
            distances[remaining], np.linalg.norm(diff, axis=1)
        )

    return np.array(chosen)


def greedy_error(X_train, y_train, k,
                 lengthscale=1.0, variance=1.0, noise=1e-6,
                 kernel=None):
    """
    Greedy forward selection by reconstruction error.

    At each step, tries every unused candidate point and permanently adds
    the one that most reduces  ||Kriging_prediction - y_train||.

    Oracle method: requires knowing y_train at every candidate location.
    NOTE: slow for large n (O(k * n) Kriging solves). Works best on
    downsampled grids.

    Parameters
    ----------
    X_train     : (n, d) candidate locations (prediction is also over these)
    y_train     : (n,)   true values at all candidate locations
    k           : number of sensors to select
    lengthscale, variance, noise : kernel hyperparameters
    kernel      : covariance function — same as in simple_kriging

    Returns
    -------
    p : (k,) int array of selected indices (0-based)
    """
    n = len(X_train)
    # Initialise with the two endpoints for good boundary coverage
    p = np.array([0, n - 1], dtype=int)

    for _ in range(2, k):
        err = np.full(n, np.inf)
        for j in range(n):
            if j not in p:
                tp      = np.append(p, j)
                mu, _, _ = simple_kriging(
                    X_train[tp], y_train[tp], X_train,
                    lengthscale=lengthscale, variance=variance, noise=noise,
                    kernel=kernel
                )
                err[j] = np.linalg.norm(mu - y_train)
        p = np.append(p, np.argmin(err))

    return p


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER FITTING  (marginal likelihood maximisation)
# ─────────────────────────────────────────────────────────────────────────────

def fit_hyperparams(X_obs, y_obs, kernel_name='rbf', nu=2.5,
                    n_restarts=5, verbose=True):
    """
    Tune lengthscale, variance, and noise by maximising the GP log marginal
    likelihood using scipy.optimize.minimize (L-BFGS-B).

    Works with any kernel — RBF or Matérn — because get_kernel() is called
    internally with whatever kernel_name you pass.

    Background
    ----------
    The log marginal likelihood is:
        log p(y | X, θ) = -0.5 * y^T C^{-1} y
                          -0.5 * log|C|
                          - n/2 * log(2π)

    The first term rewards fit (the model should predict y well).
    The second term penalises complexity (large C = more uncertain model).
    Maximising both together automatically finds the right balance without
    needing a separate validation set.

    We optimise in log-space (log lengthscale, log variance, log noise) so
    the parameters stay positive and the optimizer can move freely.
    n_restarts > 1 runs from multiple random starting points and keeps the
    best result, which helps avoid local optima.

    Parameters
    ----------
    X_obs       : (n, d) locations of the observations used for fitting
                  Tip: use your sensor locations (e.g. XY[pk_cssp]), not all
                  2500 grid points — fitting on ~25–100 points is fast;
                  fitting on 2500 is slow because it inverts an n×n matrix.
    y_obs       : (n,)   observed values at X_obs
    kernel_name : 'rbf' or 'matern'
    nu          : Matérn smoothness (ignored for RBF)
    n_restarts  : number of random restarts (more = less chance of local optima)
    verbose     : print progress and final result

    Returns
    -------
    best : dict with keys 'lengthscale', 'variance', 'noise', 'log_likelihood'
           Pass these directly to simple_kriging / universal_kriging.

    Example
    -------
    result = fit_hyperparams(XY[pk_cssp], vals[pk_cssp],
                             kernel_name=KERNEL, nu=MATERN_NU)
    # Then use result['lengthscale'] etc. in your Kriging calls.
    """
    from scipy.optimize import minimize

    n = len(y_obs)
    cov_fn = get_kernel(kernel_name, nu)

    def neg_log_marginal_likelihood(log_params):
        """Objective: negative log marginal likelihood (we minimise this)."""
        ls, var, noise = np.exp(log_params)   # map back from log-space
        try:
            C = cov_fn(X_obs, X_obs, ls, var) + noise * np.eye(n)
            # Use Cholesky for numerical stability (more reliable than np.inv)
            L = np.linalg.cholesky(C)                    # C = L L^T
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            # log|C| = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * (y_obs @ alpha + log_det + n * np.log(2 * np.pi))
            return nll
        except np.linalg.LinAlgError:
            # Cholesky failed (matrix not PD) — return a large penalty
            return 1e10

    best_nll  = np.inf
    best_params = None

    # Bounds in log-space: keeps parameters in reasonable physical ranges
    # log(ls): [log(0.1), log(1000)]   log(var): [log(0.01), log(10000)]
    # log(noise): [log(1e-6), log(10)]
    bounds = [(-2.3, 6.9), (-4.6, 9.2), (-13.8, 2.3)]

    rng = np.random.default_rng(0)

    for i in range(n_restarts):
        if i == 0:
            # First restart: use sensible data-driven starting point
            # lengthscale ≈ median pairwise distance / 3
            dists  = cdist(X_obs[:10], X_obs[:10], 'euclidean')
            ls0    = float(np.median(dists[dists > 0])) / 3.0
            var0   = float(np.var(y_obs))
            noise0 = float(np.var(y_obs)) * 0.01
            x0 = np.log([max(ls0, 0.1), max(var0, 0.01), max(noise0, 1e-6)])
        else:
            # Subsequent restarts: random initialisation within bounds
            x0 = rng.uniform([b[0] for b in bounds],
                             [b[1] for b in bounds])

        result = minimize(neg_log_marginal_likelihood, x0,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-9})

        if result.fun < best_nll:
            best_nll    = result.fun
            best_params = np.exp(result.x)

        if verbose:
            ls, var, noise = np.exp(result.x)
            print(f"  Restart {i+1}/{n_restarts}: "
                  f"ls={ls:.3f}  var={var:.3f}  noise={noise:.2e}  "
                  f"nll={result.fun:.3f}")

    ls, var, noise = best_params
    if verbose:
        print(f"\n  Best fit → lengthscale={ls:.3f}  variance={var:.3f}  "
              f"noise={noise:.2e}  (log-likelihood={-best_nll:.3f})")

    return {'lengthscale': ls, 'variance': var, 'noise': noise,
            'log_likelihood': -best_nll}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load and prepare data ─────────────────────────────────────────────────
    print("Loading ISABEL data...")
    slice_2d = load_isabel_slice(DATA_PATH, SLICE_LEVEL, DOWNSAMPLE)
    ny, nz   = slice_2d.shape
    print(f"  2D slice shape: {slice_2d.shape}  "
          f"(original 500×500 downsampled by {DOWNSAMPLE})")
    print(f"  Value range:    {slice_2d.min():.2f}  to  {slice_2d.max():.2f} m/s")

    # Build (n, 2) array of grid-index coordinates and flat value array
    yv, zv = np.arange(ny), np.arange(nz)
    Yg, Zg = np.meshgrid(yv, zv, indexing='ij')
    XY   = np.column_stack([Yg.ravel(), Zg.ravel()])   # candidate locations
    vals = slice_2d.ravel()                              # true values everywhere
    n    = len(XY)
    print(f"  Candidate points: {n}")

    # ── Normalisation ─────────────────────────────────────────────────────────
    # Work in normalised space throughout; convert back for errors and plots.
    vals_mean = vals.mean()
    vals_std  = vals.std()
    if NORMALIZE:
        vals_fit = (vals - vals_mean) / vals_std   # zero mean, unit variance
        print(f"  Normalised: mean={vals_mean:.2f}  std={vals_std:.2f}")
    else:
        vals_fit = vals                             # use raw values unchanged

    # ── Select kernel ─────────────────────────────────────────────────────────
    cov_fn = get_kernel(KERNEL, MATERN_NU)
    print(f"Kernel: {KERNEL}" + (f" (ν={MATERN_NU})" if KERNEL == 'matern' else ""))

    # ── Step 1: Place sensors for hyperparameter fitting ─────────────────────
    # FIT_ON controls which method is used (set at top of file).
    print(f"\nPlacing {N_SENSORS} sensors...")
    pk_maxmin = maxmin_ordering(XY, N_SENSORS)
    print("  MaxMin done")

    if FIT_ON == 'maxmin':
        # Kernel-free: no circular dependency. Recommended default.
        pk_for_fit = pk_maxmin
        fit_label  = 'MaxMin'

    elif FIT_ON == 'uniform':
        # Pick N_SENSORS points from the grid at roughly equal spacing.
        # stride = total points / sensors, rounded to the nearest integer.
        # Kernel-free: no circular dependency.
        stride     = max(1, n // N_SENSORS)
        pk_for_fit = np.arange(0, n, stride)[:N_SENSORS]
        fit_label  = 'Uniform'
        print(f"  Uniform grid: every {stride}th point")

    elif FIT_ON == 'random':
        # Draw N_SENSORS points uniformly at random (fixed seed for reproducibility).
        # Kernel-free: no circular dependency.
        rng_fit    = np.random.default_rng(42)
        pk_for_fit = rng_fit.choice(n, size=N_SENSORS, replace=False)
        fit_label  = 'Random'
        print(f"  Random sample: {N_SENSORS} points (seed=42)")

    elif FIT_ON == 'greedy':
        # Greedy needs a kernel, so it uses the manual config values
        # (LENGTHSCALE, VARIANCE, NOISE) for placement. The fitted
        # hyperparameters may differ from these starting values.
        print(f"  Running Greedy with manual hyperparams for fitting "
              f"(ls={LENGTHSCALE}, var={VARIANCE}, noise={NOISE})...")
        pk_for_fit = greedy_error(XY, vals_fit, N_SENSORS,
                                  lengthscale=LENGTHSCALE, variance=VARIANCE,
                                  noise=NOISE, kernel=cov_fn)
        fit_label  = 'Greedy'
        print("  Greedy (for fitting) done")

    else:
        raise ValueError(
            f"FIT_ON must be 'maxmin', 'uniform', 'random', or 'greedy'; got '{FIT_ON}'"
        )

    # ── Step 2: Fit hyperparameters on chosen sensors ─────────────────────────
    print(f"\nFitting hyperparameters via marginal likelihood (on {fit_label} sensors)...")
    hp = fit_hyperparams(XY[pk_for_fit], vals_fit[pk_for_fit],
                         kernel_name=KERNEL, nu=MATERN_NU, n_restarts=5)
    LS        = hp['lengthscale']
    VAR       = hp['variance']
    NOISE_FIT = hp['noise']
    print(f"  Using: lengthscale={LS:.3f}  variance={VAR:.3f}  noise={NOISE_FIT:.2e}")

    # ── Step 3: Build covariance matrix with fitted hyperparameters ───────────
    print("\nBuilding covariance matrix with fitted hyperparameters...")
    C = cov_fn(XY, XY, LS, VAR) + NOISE_FIT * np.eye(n)

    # ── Step 4: CSSP on the well-calibrated covariance matrix ────────────────
    pk_cssp = cssp(C, N_SENSORS)
    print("  CSSP done")

    # RPCholesky + RPGKS (scalable alternative to CSSP — no full n×n matrix needed)
    pk_rpcssp, _ = rp_cssp(XY, N_SENSORS, cov_fn,
                            lengthscale=LS, variance=VAR, rank=min(3*N_SENSORS, n))
    print("  RPCholesky + RPGKS done")

    # ── Step 5: Greedy placement with fitted hyperparameters ─────────────────
    # If FIT_ON='greedy', we reuse the sensors already placed above (avoids
    # running greedy twice). Otherwise run it fresh with the fitted hyperparams.
    # Greedy is slow — comment out the else branch if n is large.
    if FIT_ON == 'greedy':
        pk_greedy = pk_for_fit
        print("  Greedy reused from fitting step")
    else:
        pk_greedy = greedy_error(XY, vals_fit, N_SENSORS,
                                 lengthscale=LS, variance=VAR, noise=NOISE_FIT,
                                 kernel=cov_fn)
        print("  Greedy done")

    # ── Kriging reconstructions ───────────────────────────────────────────────
    # All Kriging runs use vals_fit (normalised if NORMALIZE=True).
    # Outputs are denormalised back to original units before errors and plots.
    print("\nRunning Kriging...")
    kw = dict(lengthscale=LS, variance=VAR, noise=NOISE_FIT, kernel=cov_fn)

    def denorm(mu):
        """Convert normalised predictions back to original wind-speed units."""
        return mu * vals_std + vals_mean if NORMALIZE else mu

    # Simple Kriging with each placement method
    mu_sk_cssp,    _, _ = simple_kriging(XY[pk_cssp],   vals_fit[pk_cssp],   XY, **kw)
    mu_sk_maxmin,  _, _ = simple_kriging(XY[pk_maxmin], vals_fit[pk_maxmin], XY, **kw)
    mu_sk_greedy,  _, _ = simple_kriging(XY[pk_greedy], vals_fit[pk_greedy], XY, **kw)
    mu_sk_rpcssp,  _, _ = simple_kriging(XY[pk_rpcssp], vals_fit[pk_rpcssp], XY, **kw)

    mu_sk_cssp   = denorm(mu_sk_cssp)
    mu_sk_maxmin = denorm(mu_sk_maxmin)
    mu_sk_greedy = denorm(mu_sk_greedy)
    mu_sk_rpcssp = denorm(mu_sk_rpcssp)

    # Universal Kriging with RPCholesky and Greedy sensors
    mu_uk_rpcssp, _ = universal_kriging(
        XY[pk_rpcssp], vals_fit[pk_rpcssp], XY,
        basis_funcs=[lambda X: np.ones(len(X)), lambda X: X[:, 0]], **kw
    )
    mu_uk_greedy, _ = universal_kriging(
        XY[pk_greedy], vals_fit[pk_greedy], XY,
        basis_funcs=[lambda X: np.ones(len(X)), lambda X: X[:, 0]], **kw
    )
    mu_uk_rpcssp = denorm(mu_uk_rpcssp)
    mu_uk_greedy = denorm(mu_uk_greedy)

    # ── Reconstruction errors (in original units) ─────────────────────────────
    print("\nRelative reconstruction errors (lower is better):")
    results = [
        ('Simple  + CSSP',        mu_sk_cssp),
        ('Simple  + MaxMin',      mu_sk_maxmin),
        ('Simple  + Greedy',      mu_sk_greedy),
        ('Simple  + RPCholesky',  mu_sk_rpcssp),
        ('Universal + RPCholesky',mu_uk_rpcssp),
        ('Universal + Greedy',    mu_uk_greedy),
    ]
    for label, mu in results:
        err = np.linalg.norm(mu - vals) / np.linalg.norm(vals)
        print(f"  {label:<26} {err:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def show(ax, field, title, sensors=None):
        """Helper: plot a reconstructed field with optional sensor markers."""
        im = ax.imshow(field.reshape(ny, nz), origin='lower', cmap='RdBu_r',
                       vmin=vals.min(), vmax=vals.max(), aspect='auto')
        if sensors is not None:
            # XY columns are [y-index, z-index]; imshow x-axis = z, y-axis = y
            ax.scatter(XY[sensors, 1], XY[sensors, 0],
                       c='k', s=25, marker='x', linewidths=1, label='sensors')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('z  (altitude index)')
        ax.set_ylabel('y  (latitude index)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='U wind (m/s)')

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f'ISABEL U-wind — vertical level {SLICE_LEVEL},  '
        f'{N_SENSORS} sensors,  {ny}×{nz} grid',
        fontsize=12
    )

    show(axes[0, 0], vals,          'True field  (no sensors)')
    show(axes[0, 1], mu_sk_cssp,   f'Simple Kriging + CSSP',    pk_cssp)
    show(axes[0, 2], mu_sk_maxmin, f'Simple Kriging + MaxMin',  pk_maxmin)
    show(axes[0, 3], mu_sk_greedy, f'Simple Kriging + Greedy',  pk_greedy)
    show(axes[1, 0], vals,          'True field  (no sensors)')
    show(axes[1, 1], mu_uk_rpcssp,   f'Universal Kriging + RPCholesky',  pk_rpcssp)
    show(axes[1, 2], mu_uk_greedy,   f'Universal Kriging + Greedy', pk_greedy)
    show(axes[1, 3], mu_sk_rpcssp, f'Simple Kriging + RPCholesky', pk_rpcssp)

    plt.tight_layout()
    plt.savefig('/Users/jchen228/Desktop/Argonne/kriging_results.png', dpi=150)
    plt.show()
    print("\nPlot saved to kriging_results.png")
