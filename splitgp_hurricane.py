"""
splitgp_hurricane.py

GP interpolation and sensor placement applied to the ISABEL hurricane dataset.
Extends kriging_hurricane.py with optional domain splitting: the spatial domain
is partitioned into subdomains, each fitted with its own GP and hyperparameters.
This captures non-stationary behaviour (e.g. sharp eye-wall gradients vs. calm
outer regions) that a single global GP cannot handle.

Set SPLIT_DOMAIN = True / False to toggle.  All other settings are identical
to kriging_hurricane.py.

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
    python splitgp_hurricane.py

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

# ── Domain splitting ──────────────────────────────────────────────────────────
SPLIT_DOMAIN  = True  # True  → fit a separate GP per subdomain (recommended)
                      # False → single global GP (same as kriging_hurricane.py)
N_DOMAIN_ROWS = 2     # rows of rectangular subdomains  (used when SPLIT_DOMAIN=True)
N_DOMAIN_COLS = 2     # cols of rectangular subdomains  → 2×2 = 4 subdomains
                      # To use non-rectangular domains, set SPLIT_DOMAIN=True and
                      # replace the make_rectangular_domains() call in __main__
                      # with your own list of Domain objects (see section 6 below).
BORDER_BUFFER = 10     # grid-index units.  When SPLIT_DOMAIN=True:
                      #   > 0 → sensors placed GLOBALLY; each domain's GP also
                      #         uses sensors within BORDER_BUFFER units of its
                      #         boundary (shared with neighbours and corners).
                      #   = 0 → sensors placed independently per domain
                      #         (original per-domain behaviour, no sharing).

# ── Performance ───────────────────────────────────────────────────────────────
RUN_GREEDY       = False  # True  → compute greedy sensor placement.
                          #         WARNING: at n=2500, k=75 this runs ~73×2500
                          #         Kriging solves and takes several minutes.
                          # False → skip greedy; MaxMin is used as a fast
                          #         stand-in for the "Greedy" plot panels.
GREEDY_MAX_CANDS = 500    # Only active when RUN_GREEDY=True.
                          # When n > this, randomly subsample this many
                          # candidates per greedy step instead of trying all n.
                          # Reduces cost from O(n) to O(GREEDY_MAX_CANDS)/step.
                          # Lower = faster but slightly less optimal placement.
FIT_RESTARTS     = 3      # L-BFGS-B restarts inside fit_hyperparams.
                          # More restarts → better hyperparameter estimate, but
                          # proportionally slower.  Range: 1 (fast) – 10 (thorough).


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
                 kernel=None, max_candidates=None):
    """
    Greedy forward selection by reconstruction error.

    At each step, tries candidate points and permanently adds the one that
    most reduces  ||Kriging_prediction - y_train||.

    Oracle method: requires knowing y_train at every candidate location.

    Cost: O(k × min(n, max_candidates)) Kriging solves.
      - Without subsampling (max_candidates=None): O(k × n) per step.
        At n=2500, k=75: ~187k solves — several minutes.
      - With max_candidates=500: ~36k solves — seconds.
      Set RUN_GREEDY=False (config) to skip this entirely and use MaxMin
      as a fast stand-in.

    Parameters
    ----------
    X_train        : (n, d) candidate locations
    y_train        : (n,)   true values at all candidate locations
    k              : number of sensors to select
    lengthscale, variance, noise : kernel hyperparameters
    kernel         : covariance function — same as in simple_kriging
    max_candidates : int or None.  When set and n > max_candidates,
                     randomly subsample this many candidates per step.

    Returns
    -------
    p : (k,) int array of selected indices (0-based)
    """
    n     = len(X_train)
    rng_g = np.random.default_rng(0)

    # Track chosen and remaining as lists/sets for O(1) membership checks
    chosen    = [0, n - 1]
    chosen_set = {0, n - 1}
    remaining  = list(range(1, n - 1))   # all points except the two endpoints

    for _ in range(2, k):
        # Optionally subsample to limit cost when n is large
        if max_candidates is not None and len(remaining) > max_candidates:
            pool = rng_g.choice(remaining, size=max_candidates,
                                replace=False).tolist()
        else:
            pool = remaining

        p_arr    = np.array(chosen, dtype=int)
        best_j   = pool[0]
        best_err = np.inf

        for j in pool:
            tp        = np.append(p_arr, j)
            mu, _, _  = simple_kriging(
                X_train[tp], y_train[tp], X_train,
                lengthscale=lengthscale, variance=variance, noise=noise,
                kernel=kernel
            )
            e = np.linalg.norm(mu - y_train)
            if e < best_err:
                best_err = e
                best_j   = j

        chosen.append(best_j)
        chosen_set.add(best_j)
        remaining.remove(best_j)

    return np.array(chosen, dtype=int)


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
# 6.  DOMAIN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

from collections import namedtuple

# A Domain bundles everything the main loop needs to know about one subdomain:
#   name        — label shown in print output
#   mask        — (n,) boolean array; True for points belonging to this domain
#   border_lines — list of ('h', y_pos) or ('v', z_pos) tuples that mark this
#                  domain's interior boundaries on the plot.  For rectangular
#                  domains these are axis-aligned lines; for irregular shapes
#                  you can provide empty lists and draw borders yourself in a
#                  custom show() override.
Domain = namedtuple('Domain', ['name', 'mask', 'border_lines'])


def expand_mask(domain_mask, XY, border_buffer):
    """
    Expand a domain mask to include all points within border_buffer (Euclidean
    distance in grid-index units) of any point already in the domain.

    Works for any domain shape — rectangular or irregular — because it only
    needs the mask and the coordinate array.  The cost is O(n × n_domain)
    distance comparisons, which is fast for typical grid sizes.

    Parameters
    ----------
    domain_mask   : (n,) bool array  — True for points inside the domain
    XY            : (n, 2) float array — grid-index coordinates of all points
    border_buffer : float — expansion radius in grid-index units (0 = no expansion)

    Returns
    -------
    expanded_mask : (n,) bool array  — original domain ∪ buffer zone
    """
    if border_buffer <= 0:
        return domain_mask.copy()
    domain_pts   = XY[domain_mask]                      # (n_d, 2)
    dists        = cdist(XY, domain_pts).min(axis=1)    # (n,) min dist to domain
    return dists < border_buffer                         # includes original domain


def make_rectangular_domains(XY, ny, nz, n_rows=2, n_cols=2):
    """
    Partition the ny×nz grid into n_rows×n_cols rectangular subdomains.

    Each subdomain covers a contiguous rectangular patch of grid indices.
    Points are assigned exclusively to one domain (no overlap).

    Parameters
    ----------
    XY     : (n, 2) array of grid-index coordinates  [y-index, z-index]
    ny, nz : grid dimensions
    n_rows : number of row divisions
    n_cols : number of column divisions

    Returns
    -------
    domains      : list of Domain namedtuples (length n_rows * n_cols)
    border_lines : list of ('h', y_pos) / ('v', z_pos) tuples for ALL
                   interior boundaries — pass to show() to draw them.

    To use irregular/custom domains instead, build your own list of Domain
    objects with appropriate mask arrays and pass them directly to the loop
    in __main__.  The rest of the pipeline is unchanged.
    """
    domains      = []
    border_lines = []

    for r in range(n_rows):
        y_lo = r       * ny // n_rows
        y_hi = (r + 1) * ny // n_rows if r < n_rows - 1 else ny
        # horizontal border above this row (skip the very top edge)
        if r > 0:
            border_lines.append(('h', y_lo - 0.5))

        for c in range(n_cols):
            z_lo = c       * nz // n_cols
            z_hi = (c + 1) * nz // n_cols if c < n_cols - 1 else nz
            # vertical border left of this column (skip the very left edge,
            # and only record each vertical line once — when r==0)
            if r == 0 and c > 0:
                border_lines.append(('v', z_lo - 0.5))

            mask = (
                (XY[:, 0] >= y_lo) & (XY[:, 0] < y_hi) &
                (XY[:, 1] >= z_lo) & (XY[:, 1] < z_hi)
            )
            domains.append(Domain(
                name         = f'D({r},{c})  y[{y_lo}:{y_hi}] z[{z_lo}:{z_hi}]',
                mask         = mask,
                border_lines = border_lines,   # shared reference — same lines for all
            ))

    return domains, border_lines


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load and prepare data ─────────────────────────────────────────────────
    print("Loading ISABEL data...")
    slice_2d = load_isabel_slice(DATA_PATH, SLICE_LEVEL, DOWNSAMPLE)
    ny, nz   = slice_2d.shape
    print(f"  2D slice shape: {slice_2d.shape}  "
          f"(original 500×500 downsampled by {DOWNSAMPLE})")
    print(f"  Value range:    {slice_2d.min():.2f}  to  {slice_2d.max():.2f} m/s")

    yv, zv = np.arange(ny), np.arange(nz)
    Yg, Zg = np.meshgrid(yv, zv, indexing='ij')
    XY   = np.column_stack([Yg.ravel(), Zg.ravel()])
    vals = slice_2d.ravel()
    n    = len(XY)
    print(f"  Candidate points: {n}")

    # ── Normalisation ─────────────────────────────────────────────────────────
    vals_mean = vals.mean()
    vals_std  = vals.std()
    if NORMALIZE:
        vals_fit = (vals - vals_mean) / vals_std
        print(f"  Normalised: mean={vals_mean:.2f}  std={vals_std:.2f}")
    else:
        vals_fit = vals

    def denorm(mu):
        return mu * vals_std + vals_mean if NORMALIZE else mu

    # ── Select kernel ─────────────────────────────────────────────────────────
    cov_fn = get_kernel(KERNEL, MATERN_NU)
    print(f"Kernel: {KERNEL}" + (f" (ν={MATERN_NU})" if KERNEL == 'matern' else ""))

    # ── Helper: fit hyperparams + all sensor placements for one domain ────────
    def fit_and_place(XY_d, vals_d, n_d, ns, label=''):
        """
        Fit hyperparameters and run all sensor placement methods on one domain.

        XY_d   : (n_d, 2) locations for this domain
        vals_d : (n_d,)   normalised values for this domain
        n_d    : number of points in this domain
        ns     : number of sensors to place
        label  : string prefix for print output

        Returns a dict with keys:
            'LS', 'VAR', 'NOISE'  — fitted hyperparameters
            'cssp', 'maxmin', 'greedy', 'rpcssp'  — local sensor indices (into XY_d)
        """
        ns = min(ns, n_d - 1)   # can't place more sensors than points

        # ── FIT_ON placement (kernel-free options work locally too) ──────────
        if FIT_ON == 'maxmin':
            pk_fit = maxmin_ordering(XY_d, ns)
            fl = 'MaxMin'
        elif FIT_ON == 'uniform':
            stride = max(1, n_d // ns)
            pk_fit = np.arange(0, n_d, stride)[:ns]
            fl = 'Uniform'
        elif FIT_ON == 'random':
            rng_fit = np.random.default_rng(42)
            pk_fit  = rng_fit.choice(n_d, size=ns, replace=False)
            fl = 'Random'
        elif FIT_ON == 'greedy':
            pk_fit = greedy_error(XY_d, vals_d, ns,
                                  lengthscale=LENGTHSCALE, variance=VARIANCE,
                                  noise=NOISE, kernel=cov_fn)
            fl = 'Greedy'
        else:
            raise ValueError(f"Unknown FIT_ON='{FIT_ON}'")

        # ── Fit hyperparameters ───────────────────────────────────────────────
        print(f"  {label}Fitting hyperparams on {fl} sensors "
              f"({FIT_RESTARTS} restarts)...")
        hp = fit_hyperparams(XY_d[pk_fit], vals_d[pk_fit],
                             kernel_name=KERNEL, nu=MATERN_NU,
                             n_restarts=FIT_RESTARTS, verbose=False)
        LS_d, VAR_d, NOISE_d = hp['lengthscale'], hp['variance'], hp['noise']
        print(f"  {label}→ ls={LS_d:.3f}  var={VAR_d:.3f}  noise={NOISE_d:.2e}")

        # ── Covariance matrix for this domain ─────────────────────────────────
        C_d = cov_fn(XY_d, XY_d, LS_d, VAR_d) + NOISE_d * np.eye(n_d)

        # ── Sensor placements ─────────────────────────────────────────────────
        pk_cssp_d    = cssp(C_d, ns)
        pk_maxmin_d  = maxmin_ordering(XY_d, ns)
        pk_rpcssp_d, _ = rp_cssp(XY_d, ns, cov_fn,
                                  lengthscale=LS_d, variance=VAR_d,
                                  rank=min(3 * ns, n_d))
        if not RUN_GREEDY:
            # Skip greedy entirely — use MaxMin as a fast stand-in.
            # Set RUN_GREEDY=True in config to re-enable (will be slow).
            pk_greedy_d = pk_maxmin_d
        elif FIT_ON == 'greedy':
            pk_greedy_d = pk_fit   # reuse the sensors already placed for fitting
        else:
            pk_greedy_d = greedy_error(XY_d, vals_d, ns,
                                       lengthscale=LS_d, variance=VAR_d,
                                       noise=NOISE_d, kernel=cov_fn,
                                       max_candidates=GREEDY_MAX_CANDS)

        return dict(LS=LS_d, VAR=VAR_d, NOISE=NOISE_d,
                    cssp=pk_cssp_d, maxmin=pk_maxmin_d,
                    greedy=pk_greedy_d, rpcssp=pk_rpcssp_d)

    # ── Helper: run all Kriging methods for one domain ────────────────────────
    def kriging_domain(XY_d, vals_d, XY_pred, res):
        """
        Run Simple + Universal Kriging for each placement method on one domain.

        XY_d    : sensor candidate locations (domain subset)
        vals_d  : normalised values at those locations
        XY_pred : locations to predict at (same as XY_d for in-domain prediction)
        res     : dict returned by fit_and_place()

        Returns dict of raw (still normalised) prediction arrays.
        """
        kw = dict(lengthscale=res['LS'], variance=res['VAR'],
                  noise=res['NOISE'], kernel=cov_fn)
        basis = [lambda X: np.ones(len(X)), lambda X: X[:, 0]]

        out = {}
        for name, pk in [('cssp',   res['cssp']),
                         ('maxmin', res['maxmin']),
                         ('greedy', res['greedy']),
                         ('rpcssp', res['rpcssp'])]:
            mu, _, _ = simple_kriging(XY_d[pk], vals_d[pk], XY_pred, **kw)
            out[f'sk_{name}'] = mu
            mu_uk, _ = universal_kriging(XY_d[pk], vals_d[pk], XY_pred,
                                         basis_funcs=basis, **kw)
            out[f'uk_{name}'] = mu_uk
        return out

    # ═════════════════════════════════════════════════════════════════════════
    # BRANCH: single global GP  vs.  per-domain split GPs
    # ═════════════════════════════════════════════════════════════════════════

    if not SPLIT_DOMAIN:
        # ── Single global GP (identical to kriging_hurricane.py) ─────────────
        print("\n── Single global GP ──")
        res = fit_and_place(XY, vals_fit, n, N_SENSORS, label='[global] ')
        preds_norm = kriging_domain(XY, vals_fit, XY, res)

        # global sensor arrays for plotting (local == global when no split)
        pk_cssp   = res['cssp'];   pk_maxmin = res['maxmin']
        pk_greedy = res['greedy']; pk_rpcssp = res['rpcssp']

        mu_sk_cssp    = denorm(preds_norm['sk_cssp'])
        mu_sk_maxmin  = denorm(preds_norm['sk_maxmin'])
        mu_sk_greedy  = denorm(preds_norm['sk_greedy'])
        mu_sk_rpcssp  = denorm(preds_norm['sk_rpcssp'])
        mu_uk_rpcssp  = denorm(preds_norm['uk_rpcssp'])
        mu_uk_greedy  = denorm(preds_norm['uk_greedy'])

        border_lines = []   # no domain borders to draw

    else:
        # ── Domain-split GPs ─────────────────────────────────────────────────
        # Build domain partition (rectangular by default).
        # To use custom shapes: replace the next two lines with your own
        # domains list and border_lines list (see make_rectangular_domains docs).
        domains, border_lines = make_rectangular_domains(
            XY, ny, nz, N_DOMAIN_ROWS, N_DOMAIN_COLS
        )
        n_domains = len(domains)

        if BORDER_BUFFER > 0:
            # ── Global placement + border sharing ────────────────────────────
            # Sensors are placed once on the full grid so no sensor is
            # duplicated across the boundary.  Each domain's GP then trains on
            # the sensors that fall inside its expanded region (original mask
            # ∪ BORDER_BUFFER-unit buffer), so a sensor near any boundary is
            # automatically shared by all domains that touch it.

            print(f"\n── Split GP ({N_DOMAIN_ROWS}×{N_DOMAIN_COLS} domains, "
                  f"border_buffer={BORDER_BUFFER}) — GLOBAL placement ──")

            # Step A: global FIT_ON placement + global hyperparams
            res_g = fit_and_place(XY, vals_fit, n, N_SENSORS, label='[global] ')
            pk_cssp   = res_g['cssp'];   pk_maxmin = res_g['maxmin']
            pk_greedy = res_g['greedy']; pk_rpcssp = res_g['rpcssp']
            # Also keep a dict for easy iteration below
            global_pk = {'cssp': pk_cssp, 'maxmin': pk_maxmin,
                         'greedy': pk_greedy, 'rpcssp': pk_rpcssp}

            # Accumulators
            mu_sk_cssp   = np.zeros(n); mu_sk_maxmin  = np.zeros(n)
            mu_sk_greedy = np.zeros(n); mu_sk_rpcssp  = np.zeros(n)
            mu_uk_rpcssp = np.zeros(n); mu_uk_greedy  = np.zeros(n)

            basis = [lambda X: np.ones(len(X)), lambda X: X[:, 0]]

            for dom in domains:
                mask       = dom.mask
                global_idx = np.where(mask)[0]
                n_d        = mask.sum()

                # Expanded mask — includes the buffer zone around this domain
                exp_mask = expand_mask(mask, XY, BORDER_BUFFER)

                # FIT_ON sensors inside the expanded region → local hyperparams
                pk_fit_g      = global_pk[FIT_ON] if FIT_ON in global_pk \
                                else global_pk['maxmin']   # fallback
                fit_in_exp    = pk_fit_g[exp_mask[pk_fit_g]]
                if len(fit_in_exp) < 2:
                    # Too few sensors in buffer; borrow nearest sensors
                    dists = cdist(XY[pk_fit_g], XY[global_idx]).min(axis=1)
                    fit_in_exp = pk_fit_g[np.argsort(dists)[:max(2, N_SENSORS // n_domains)]]

                print(f"\n  Domain '{dom.name}'  "
                      f"({n_d} pts, {exp_mask.sum()} in buffer)")
                print(f"    Fitting hyperparams on {len(fit_in_exp)} FIT_ON sensors...")
                hp_d = fit_hyperparams(XY[fit_in_exp], vals_fit[fit_in_exp],
                                       kernel_name=KERNEL, nu=MATERN_NU,
                                       n_restarts=3, verbose=False)
                LS_d, VAR_d, NOISE_d = (hp_d['lengthscale'],
                                        hp_d['variance'], hp_d['noise'])
                print(f"    → ls={LS_d:.3f}  var={VAR_d:.3f}  noise={NOISE_d:.2e}")

                kw_d = dict(lengthscale=LS_d, variance=VAR_d,
                            noise=NOISE_d, kernel=cov_fn)

                # For each placement method: find its global sensors in the
                # expanded region, then predict at the domain's own points.
                for meth, pk_g in global_pk.items():
                    in_exp = pk_g[exp_mask[pk_g]]   # sensors inside expanded mask
                    if len(in_exp) < 2:
                        # Fallback: closest sensors to domain centre
                        centre = XY[global_idx].mean(axis=0, keepdims=True)
                        dists  = cdist(XY[pk_g], centre).ravel()
                        in_exp = pk_g[np.argsort(dists)[:max(2, N_SENSORS // n_domains)]]

                    mu_sk, _, _ = simple_kriging(
                        XY[in_exp], vals_fit[in_exp], XY[global_idx], **kw_d)
                    mu_uk, _ = universal_kriging(
                        XY[in_exp], vals_fit[in_exp], XY[global_idx],
                        basis_funcs=basis, **kw_d)

                    if meth == 'cssp':
                        mu_sk_cssp[global_idx]   = denorm(mu_sk)
                    elif meth == 'maxmin':
                        mu_sk_maxmin[global_idx] = denorm(mu_sk)
                    elif meth == 'greedy':
                        mu_sk_greedy[global_idx] = denorm(mu_sk)
                        mu_uk_greedy[global_idx] = denorm(mu_uk)
                    elif meth == 'rpcssp':
                        mu_sk_rpcssp[global_idx] = denorm(mu_sk)
                        mu_uk_rpcssp[global_idx] = denorm(mu_uk)

        else:
            # ── Per-domain placement (original behaviour, BORDER_BUFFER=0) ───
            print(f"\n── Split GP ({N_DOMAIN_ROWS}×{N_DOMAIN_COLS} domains, "
                  f"no border sharing) — per-domain placement ──")

            mu_sk_cssp   = np.zeros(n); mu_sk_maxmin  = np.zeros(n)
            mu_sk_greedy = np.zeros(n); mu_sk_rpcssp  = np.zeros(n)
            mu_uk_rpcssp = np.zeros(n); mu_uk_greedy  = np.zeros(n)
            pk_cssp = []; pk_maxmin = []; pk_greedy = []; pk_rpcssp = []

            for dom in domains:
                mask       = dom.mask
                global_idx = np.where(mask)[0]
                XY_d       = XY[mask]
                vals_d     = vals_fit[mask]
                n_d        = len(XY_d)
                ns_d       = max(3, int(N_SENSORS * n_d / n))

                print(f"\n  Domain '{dom.name}'  ({n_d} pts, {ns_d} sensors)")
                res       = fit_and_place(XY_d, vals_d, n_d, ns_d,
                                          label=f'    [{dom.name}] ')
                preds_raw = kriging_domain(XY_d, vals_d, XY_d, res)

                mu_sk_cssp[global_idx]   = denorm(preds_raw['sk_cssp'])
                mu_sk_maxmin[global_idx] = denorm(preds_raw['sk_maxmin'])
                mu_sk_greedy[global_idx] = denorm(preds_raw['sk_greedy'])
                mu_sk_rpcssp[global_idx] = denorm(preds_raw['sk_rpcssp'])
                mu_uk_rpcssp[global_idx] = denorm(preds_raw['uk_rpcssp'])
                mu_uk_greedy[global_idx] = denorm(preds_raw['uk_greedy'])

                pk_cssp.extend(global_idx[res['cssp']].tolist())
                pk_maxmin.extend(global_idx[res['maxmin']].tolist())
                pk_greedy.extend(global_idx[res['greedy']].tolist())
                pk_rpcssp.extend(global_idx[res['rpcssp']].tolist())

            pk_cssp   = np.array(pk_cssp);   pk_maxmin = np.array(pk_maxmin)
            pk_greedy = np.array(pk_greedy); pk_rpcssp = np.array(pk_rpcssp)

    # ── Reconstruction errors ─────────────────────────────────────────────────
    print("\nRelative reconstruction errors (lower is better):")
    for label, mu in [
        ('Simple  + CSSP',         mu_sk_cssp),
        ('Simple  + MaxMin',       mu_sk_maxmin),
        ('Simple  + Greedy',       mu_sk_greedy),
        ('Simple  + RPCholesky',   mu_sk_rpcssp),
        ('Universal + RPCholesky', mu_uk_rpcssp),
        ('Universal + Greedy',     mu_uk_greedy),
    ]:
        err = np.linalg.norm(mu - vals) / np.linalg.norm(vals)
        print(f"  {label:<26} {err:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def show(ax, field, title, sensors=None):
        """Plot one reconstructed field with sensor markers and domain borders."""
        im = ax.imshow(field.reshape(ny, nz), origin='lower', cmap='RdBu_r',
                       vmin=vals.min(), vmax=vals.max(), aspect='auto')
        if sensors is not None:
            ax.scatter(XY[sensors, 1], XY[sensors, 0],
                       c='k', s=20, marker='x', linewidths=0.8, label='sensors')
        # Draw domain borders
        # border_lines contains ('h', y_pos) or ('v', z_pos) tuples.
        # 'h' → horizontal line (constant y in grid = horizontal in imshow).
        # 'v' → vertical   line (constant z in grid = vertical   in imshow).
        for orientation, pos in border_lines:
            if orientation == 'h':
                ax.axhline(pos, color='yellow', linewidth=1.2, linestyle='--')
            else:
                ax.axvline(pos, color='yellow', linewidth=1.2, linestyle='--')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('z  (longitude index)')
        ax.set_ylabel('y  (latitude index)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='U wind (m/s)')

    split_tag = (f'  [{N_DOMAIN_ROWS}×{N_DOMAIN_COLS} domains]'
                 if SPLIT_DOMAIN else '  [single GP]')
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f'ISABEL U-wind — level {SLICE_LEVEL},  {N_SENSORS} sensors,  '
        f'{ny}×{nz} grid{split_tag}',
        fontsize=11
    )

    greedy_tag = 'Greedy' if RUN_GREEDY else 'Greedy→MaxMin'
    show(axes[0, 0], vals,         'True field')
    show(axes[0, 1], mu_sk_cssp,   'Simple + CSSP',                    pk_cssp)
    show(axes[0, 2], mu_sk_maxmin, 'Simple + MaxMin',                   pk_maxmin)
    show(axes[0, 3], mu_sk_greedy, f'Simple + {greedy_tag}',            pk_greedy)
    show(axes[1, 0], vals,         'True field')
    show(axes[1, 1], mu_uk_rpcssp, 'Universal + RPCholesky',            pk_rpcssp)
    show(axes[1, 2], mu_uk_greedy, f'Universal + {greedy_tag}',         pk_greedy)
    show(axes[1, 3], mu_sk_rpcssp, 'Simple + RPCholesky',               pk_rpcssp)

    plt.tight_layout()
    out_path = '/Users/jchen228/Desktop/Argonne/splitgp_results.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {out_path}")
    print("\nPlot saved to kriging_results.png")
