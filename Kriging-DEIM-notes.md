# Kriging-DEIM.ipynb — Reference Notes

> **Purpose of this document:** explain what the notebook does, cell by cell, without requiring you to re-read the code. Written for someone familiar with MATLAB/numerical methods but newer to the Python ML ecosystem.

---

## Big Picture

The notebook is an exploratory workbook with two interleaved threads:

1. **Kriging variants** — understanding the difference between Simple, Ordinary, and Universal Kriging in 1D, 2D, and 3D.
2. **Sensor/Subset Selection** — given a large candidate set, how to pick a small, informative subset of points to observe, using DEIM-style column subset selection and other strategies.

The name "DEIM" refers to the **Discrete Empirical Interpolation Method**, a technique that uses the leading singular vectors of a matrix to select representative interpolation points via QR pivoting with column pivoting. In this notebook that idea is applied to the covariance matrix to choose sensor locations.

The cells progress from simple 1D toy problems toward a real 3D scientific dataset (the ISABEL hurricane simulation).

---

## Core Concepts (read before the cell-by-cell breakdown)

### Kriging in a nutshell

Kriging is Gaussian Process (GP) interpolation. Given noisy observations at a handful of locations, it predicts the full field everywhere and gives a **posterior mean** (best estimate) and **posterior variance** (uncertainty) at every test point.

All three Kriging variants share the same RBF (squared-exponential / Gaussian) covariance:

```
K(x_a, x_b) = variance * exp( -0.5 * ||x_a - x_b||² / lengthscale² )
```

They differ only in how they handle the **mean function**:

| Variant | Mean assumption | How mean is handled |
|---|---|---|
| **Simple Kriging** | Known constant `m0` (default 0) | Fixed; user supplies it |
| **Ordinary Kriging** | Unknown constant | Estimated from data (basis = `[1]`) |
| **Universal Kriging** | Unknown linear trend | Estimated from data (basis = `[1, x]`) |

Universal Kriging is a special case of the general **GLS (Generalized Least Squares)** framework: you supply a set of basis functions `G(x)`, estimate their coefficients `beta` by GLS, then add a GP correction for local structure.

### DEIM / Column Subset Selection (CSSP)

The idea: instead of placing sensors randomly or on a uniform grid, use the **structure of the covariance matrix** to find the most informative locations.

Steps:
1. Compute (or approximate) the covariance matrix `C` of the candidate points.
2. Compute its leading `k` singular vectors via **randomized SVD**: `U, s, Vh = randomized_svd(C, k)`.
3. Apply **QR with column pivoting** to `Vh`: `_, _, p = scipy.linalg.qr(Vh, pivoting=True)`.
4. The first `k` entries of `p` are the selected sensor indices.

This is exactly the DEIM algorithm, applied to the covariance rather than a snapshot matrix.

### MaxMin Ordering

An alternative to CSSP for sensor placement. The algorithm greedily selects points that are **maximally spread** in space:

1. Start with the first point.
2. At each step, pick the remaining point that is **farthest from all already-selected points**.
3. Update the min-distance array incrementally (efficient `O(n)` per step).

This does not use the covariance at all — it is purely geometry-based.

---

## Cell-by-Cell Breakdown

### Cell 0 — 1D Kriging comparison (baseline)

**What it does:**
- Defines the true function `f(x) = 0.6x + sin(1.2x)` on `[0, 10]`.
- Samples 10 training points (no noise, the noise term is multiplied by 0).
- Runs all three Kriging variants and plots their posterior mean + 95% confidence bands.

**Key functions introduced:**
- `rbf_cov(xa, xb, lengthscale, variance)` — the RBF kernel matrix
- `simple_kriging(...)` — posterior mean/variance with fixed mean `m0=0`
- `universal_kriging(...)` — GLS mean estimation + GP correction; handles both Ordinary and Universal depending on which `basis_funcs` you pass
- `build_G(x, funcs)` — constructs the design matrix from basis function list

**Hyperparameters used:** `lengthscale=0.9`, `variance=1.2`

**What to notice:**
- Simple Kriging reverts to 0 away from data (its assumed prior mean).
- Ordinary Kriging extrapolates to a constant.
- Universal Kriging extrapolates along the linear trend.
- All three agree near the training points.

---

### Cell 1 — 2D Kriging comparison

**What it does:**
- Extends Cell 0 to 2D: `f(x,y) = sin(x)*cos(y) + 0.2x` on a 40×40 grid.
- 10 randomly scattered observations (small noise `scale=0.05`).
- Plots a 4-panel heatmap: True field, Simple, Ordinary, Universal.

**Key changes from 1D:**
- `rbf_cov_2d` uses `scipy.spatial.distance.cdist` for pairwise squared distances (efficient for 2D+).
- `build_G_2d` takes functions of `(x, y)` instead of just `x`.
- The linear basis for Universal Kriging is `[1, x]` only (trend only in x-direction).

**What to notice:** With only 10 points on a 40×40 grid the interpolation is quite uncertain far from the observations, but all methods qualitatively recover the structure.

---

### Cell 2 — 3D Kriging on a synthetic Gaussian Random Field

**What it does:**
- Builds a 10×10×10 grid in 3D.
- Generates the true field from a **Gaussian Random Field** (sampled from `rng.multivariate_normal` with the RBF covariance as the covariance matrix), then adds a linear drift `0.15x`.
- 30 randomly chosen observation points.
- Visualizes a 2D slice (middle z-plane) of each method's reconstruction as a 3D surface plot.

**Key point:** Forming the full covariance `K` for even a 1000-point grid is memory-intensive (1000×1000 = 1M entries). This cell still does it explicitly — later cells address the scaling problem.

**What to notice:** The GRF is a random draw so the true field is bumpy and complex. Universal Kriging does best because of the x-drift.

---

### Cell 3 — 1D subset selection with CSSP (first appearance of DEIM)

**What it does:**
- 100 training points, `f(x) = 0.6x + sin(1.2x)`.
- Runs Simple Kriging on **all** 100 points to get the full covariance `C`.
- Applies **CSSP** (randomized SVD + QR pivoting) to `C` to select `nsel=10` points.
- Re-runs Simple Kriging on just those 10 points.
- Plots the full-data Kriging vs the 10-point CSSP Kriging, reporting reconstruction error.

**CSSP pipeline (explicit):**
```python
U, s, Vh = randomized_svd(C, n_components=nsel, random_state=0)
_, _, p = scipy.linalg.qr(Vh, pivoting=True)
p = p[0:nsel]   # selected indices (0-based)
```

**What to notice:** This is the core idea of the notebook — you don't need all 100 points. 10 well-chosen points (selected by the covariance structure) can reconstruct the field almost as well.

---

### Cell 4 — 2D subset selection with CSSP

**What it does:**
- Extends Cell 3 to 2D on the same `sin(x)*cos(y) + 0.2x` field.
- Now uses a **regular grid** of 20×20 = 400 observations (not random).
- Selects `nsel=25` points via CSSP.
- Runs all three Kriging variants on the 25-point subset.
- Notes in a comment that all variants give the same `p` because they all use the same `C` (the kernel matrix depends only on the training points, not which Kriging variant).

**What to notice:** The selected 25 points are spread across the domain, not clustered. CSSP automatically avoids redundant nearby points.

---

### Cell 5 — 1D greedy forward selection

**What it does:**
- An alternative subset selection strategy: pure **greedy forward selection** by reconstruction error.
- Starts with the two endpoints (`p = [0, n_train-1]`).
- At each step, tries every remaining candidate, adds the one that minimizes `||Kriging_prediction - y_train||`.
- Runs until `k=10` points are selected.

**Computational cost:** O(k × n) Kriging solves — slow for large n. This is the brute-force version; CSSP is the fast algebraic approximation to it.

**What to notice:** Compare the selected points from this cell vs Cell 3. CSSP (Cell 3) is much faster but the greedy approach directly minimizes reconstruction error.

---

### Cell 6 — Scratch cell

Prints the array `mu_simple` — this is a leftover debugging cell from interactive development, not part of any workflow.

---

### Cell 7 — 3D CSSP with memory-efficient LinearOperator

**What it does:**
- Returns to 3D but now uses a **deterministic** true field: `sin(x)*cos(y) + 2*sin(2*z) + 0.2x` (avoids the expensive multivariate normal sample from Cell 2).
- Introduces `make_rbf_linear_operator` — a `scipy.sparse.linalg.LinearOperator` that computes **kernel-vector products in batches**, never forming the full N×N matrix.
- Introduces `randomized_svd_linear_operator` — a custom randomized SVD that works with any LinearOperator (since `sklearn.randomized_svd` requires an explicit matrix).
- Selects `n_obs=200` sensor locations via CSSP on the LinearOperator.

**Why this matters:** For a 10×10×10 = 1000-point grid, the full kernel matrix is 1000×1000 (manageable). But for a 100×100×100 grid it would be 10^6 × 10^6 — impossible to store. The LinearOperator approach computes the action of the matrix (i.e., `K @ v`) without storing `K`.

**Key new code:**
```python
def randomized_svd_linear_operator(A, n_components, ...):
    # Randomized range finding
    Omega = rng.standard_normal(size=(m, n_random))
    Y = A @ Omega               # sample range of A
    Q, _ = np.linalg.qr(Y)     # orthonormalize
    # Power iteration for accuracy
    B = Q.T @ (A @ eye(m))     # project to small matrix
    U_tilde, S, Vt = svd(B)    # SVD of small matrix
    U = Q @ U_tilde             # map back to full space
```

---

### Cell 8 — 3D MaxMin ordering

**What it does:**
- Uses a larger 20×20×20 = 8000-point grid.
- Replaces CSSP with **MaxMin ordering** for sensor placement.
- Calls `maxmin_ordering_fast2` (note: this is a typo — the defined function is `maxmin_ordering_fast`; the cell has a bug if run from scratch).
- Selects 150 sensors, then runs all three Kriging variants.

**MaxMin algorithm:**
```
selected = [first point]
distances = ||all_points - first_point||
for i in range(1, n_select):
    next = argmax(distances * remaining_mask)
    update distances = min(old_distances, ||all_points - next||)
```

**What to notice:** MaxMin spreads points geometrically; CSSP spreads them in "covariance space." For smooth kernels these tend to agree, but MaxMin is purely geometric and needs no matrix factorization.

---

### Cell 9 — Real data: ISABEL hurricane simulation

**What it does:**
- Mounts Google Drive and loads a binary file from the **ISABEL dataset** (a simulated category-5 hurricane from NCAR/IEEE Visualization Contest 2004).
- The file `Uf48.bin.f32` contains one field (likely wind velocity component U) at timestep 48, stored as 32-bit floats.
- Shape is `(100, 500, 500)` — a 3D volume (100 vertical levels × 500×500 horizontal).
- Applies MaxMin ordering to select 100 observation points from the 25 million total grid points.

**What to notice:** This is where the notebook transitions from toy problems to real scientific data. The LinearOperator and MaxMin techniques from earlier cells exist precisely because you can't form a 25M×25M matrix.

---

### Cell 10 — pykrige OrdinaryKriging3D on ISABEL

**What it does:**
- Uses the `pykrige` library's `OrdinaryKriging3D` class (a production-quality Kriging implementation) instead of the hand-rolled functions.
- Fits the model to the 100 MaxMin-selected ISABEL observations with a `'spherical'` variogram model.
- Predicts on a 2D slice at `x=50` (one horizontal cross-section through the 3D field).

**Variogram model note:** `pykrige` uses the geostatistics convention of fitting a **variogram** (semi-variance as a function of distance) rather than a covariance directly. "Spherical" is a standard variogram model with a finite range.

---

### Cell 11 — Comparison plot for ISABEL

**What it does:**
- Two-panel plot: pykrige reconstruction of the `x=50` slice vs the true data slice `data[50, :, :]`.
- Quick visual sanity check that the interpolation is reasonable with 100 points.

---

### Cell 12 — Leftover visualization cell

References `F_simple`, `F_ok`, `F_uk` and `F_true` from a prior run (not re-computed in this cell). Essentially a scratch/continuation cell from an interactive session. Will error if run in isolation.

---

### Cell 13 — Full 3D Kriging on ISABEL-scale data

**What it does:**
- Re-applies the hand-rolled `simple_kriging_3d` and `universal_kriging_3d` to the ISABEL observation points.
- Visualizes the same `x=50` slice.

**Note:** This is the hand-rolled version vs Cell 10's pykrige version — a comparison of the two implementations on the same data.

---

## Overall Narrative Arc

```
Cells 0–2:   Understand Kriging variants (Simple / Ordinary / Universal)
             in 1D → 2D → 3D on toy functions.

Cells 3–5:   Add subset selection to the picture.
             Cell 3: CSSP in 1D.
             Cell 4: CSSP in 2D.
             Cell 5: Greedy forward selection in 1D (slower but directly optimizes error).

Cells 6:     Scratch output.

Cell 7:      Scale CSSP to 3D using a memory-efficient LinearOperator.
Cell 8:      Alternative sensor placement: MaxMin geometric ordering.

Cells 9–13:  Apply to real data (ISABEL 500×500×100 hurricane field).
             Pivot from synthetic experiments to scientific application.
```

---

## Connections to gpoed-code-python

If you're familiar with the gpoed-code-python codebase, here's how the pieces map:

| Kriging-DEIM concept | gpoed-code-python equivalent |
|---|---|
| `rbf_cov` | `gauss_kern.py` |
| `simple_kriging` | `krr.py` (Kernel Ridge Regression ≈ Simple Kriging) |
| Randomized SVD + QR pivoting (CSSP) | `rpgks.py` and `gks.py` |
| Full SVD + QR pivoting | `gks.py` |
| `nys` Nyström approximation | `nystrom.py` |
| `make_rbf_linear_operator` | The `K_fun` / `K_matvec` lambda pattern in `setup_sst.py` |
| Greedy forward selection (Cell 5) | `greedy_dopt.py` (same idea, different criterion: D-optimality instead of reconstruction error) |
| MaxMin ordering | Not yet in gpoed-code-python |

**Key conceptual difference:** gpoed uses **D-optimality** (log-determinant of the selected covariance sub-matrix) as the greedy criterion, while this notebook uses **reconstruction error** (`||prediction - truth||`). Both are valid; D-optimality is criterion-based (doesn't need the true field), while reconstruction error is oracle-based (needs the true field).

---

## Potential Issues / Things to Watch

- **Cell 8 bug:** calls `maxmin_ordering_fast2` but the function is defined as `maxmin_ordering_fast`. Will throw `NameError` if run fresh.
- **Cell 12 & 13** rely on variables from earlier cells being in memory. Run top-to-bottom.
- **Cell 9 loads from Google Drive** (`/content/gdrive/...`) — this only works in a Google Colab environment, not locally.
- **No noise** in most experiments: `0 * rng.normal(...)` means the noise term is zeroed out. The `noise=1e-6` parameter in Kriging is just a regularization nugget, not observation noise.
- **`np.linalg.inv` vs `cho_solve`:** The hand-rolled Kriging functions use direct matrix inversion, which is numerically less stable than Cholesky factorization for large/ill-conditioned matrices. The gpoed-code-python implementations use `cho_factor`/`cho_solve` instead.

---

*Document generated from reading Kriging-DEIM.ipynb without executing it. Last updated: 2026-06-11.*
