# Kriging & Sensor Placement — Hurricane Isabel

*NOTE: Readme and code generated through Claude Cowork (Sonnet 4.6)*

Gaussian Process interpolation and optimal sensor placement applied to the
ISABEL hurricane simulation dataset. Extracted and extended from
`Kriging-DEIM.ipynb`.

---

## Files

### `kriging_hurricane.py`

A self-contained Python script implementing GP regression (Kriging) and four
sensor placement methods on a 2D slice of the ISABEL wind field.

**GP methods**

- **Simple Kriging** — assumes a known constant prior mean (default 0)
- **Ordinary Kriging** — estimates an unknown constant mean from data
- **Universal Kriging** — estimates an unknown linear trend from data

**Sensor placement methods**

- **CSSP / GKS** — Column Subset Selection via randomized SVD + QR pivoting on
  the full covariance matrix. Equivalent to `gks.m` in the MATLAB codebase.
- **RPCholesky + RPGKS** — scalable alternative to CSSP; builds a low-rank
  Cholesky approximation of the covariance matrix first, then applies CSSP to
  the factor. Avoids forming the full n×n matrix.
- **MaxMin ordering** — farthest-point geometric spreading; no kernel required.
- **Greedy error** — forward selection minimising reconstruction error at each
  step. Oracle method: requires knowing the true field everywhere.

**Covariance kernels**

- **RBF** (squared-exponential / Gaussian)
- **Matérn** with tunable smoothness ν ∈ {0.5, 1.5, 2.5}

**Hyperparameter fitting**

Lengthscale, variance, and noise are tuned automatically by maximising the GP
log marginal likelihood (`scipy.optimize.minimize`, L-BFGS-B, with multiple
random restarts). The fitting sensors are selected by the method set in
`FIT_ON` before any kernel-dependent placement occurs, avoiding circular
dependency.

**Usage**

```bash
pip install numpy scipy matplotlib scikit-learn
python kriging_hurricane.py
```

Outputs a `kriging_results.png` plot (2×4 grid of reconstructions) and prints
relative reconstruction errors for each method.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `SLICE_LEVEL` | `50` | Vertical level to extract (0–99) |
| `DOWNSAMPLE` | `10` | Keep every Nth pixel; 10 → 50×50 = 2500 points |
| `N_SENSORS` | `75` | Number of sensors to place |
| `KERNEL` | `'matern'` | `'rbf'` or `'matern'` |
| `MATERN_NU` | `0.5` | Matérn smoothness: `0.5`, `1.5`, or `2.5` |
| `FIT_ON` | `'random'` | Sensor set used to fit hyperparameters: `'maxmin'`, `'uniform'`, `'random'`, or `'greedy'` |
| `NORMALIZE` | `True` | Subtract mean and divide by std before Kriging |
| `LENGTHSCALE` | `5.0` | Manual lengthscale (used only when `FIT_ON='greedy'`) |
| `VARIANCE` | `10.0` | Manual variance (used only when `FIT_ON='greedy'`) |
| `NOISE` | `1e-3` | Manual noise (used only when `FIT_ON='greedy'`) |

**Tips for better reconstruction**

- Lower `DOWNSAMPLE` (try 5) for finer spatial resolution before tuning anything else.
- Increase `N_SENSORS` — detail recoverable scales roughly with sensor count.
- Set `NORMALIZE = True` when the field has a large offset or range (always recommended for wind data).
- The hurricane wind field is non-stationary (sharp gradients near the eye wall, smooth elsewhere). A single global lengthscale cannot resolve both regimes simultaneously; this is a fundamental limitation of stationary GP models.

---

## Dataset

**ISABEL Hurricane Simulation**

> M. Shead (ed.), *IEEE Visualization 2004 Contest Dataset: Hurricane Isabel*,
> National Center for Atmospheric Research (NCAR), 2004.
> Data provided by Kelvin K. Droegemeier, Ming Xue, and the Center for
> Analysis and Prediction of Storms (CAPS), University of Oklahoma, and by
> the National Center for Atmospheric Research.
> Available at: https://ieeexplore.ieee.org/document/1566621

The dataset is a numerical simulation of Hurricane Isabel (September 2003)
produced by the Weather Research and Forecast (WRF) model. The full volume is
500 × 500 × 100 grid points over 48 time steps, with 13 atmospheric variables.

This project uses the U (east–west) wind component at timestep 48
(`Uf48.bin.f32`), stored as little-endian 32-bit floats in a flat binary file.
Load with:

```python
import numpy as np
data = np.fromfile("Uf48.bin.f32", dtype=np.float32).reshape((100, 500, 500))
# axis 0: vertical level (0 = bottom, 99 = top)
# axis 1: latitude (y)
# axis 2: longitude (z)
```

---

## Requirements

```
numpy
scipy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy scipy matplotlib scikit-learn
```

---

## Notes

- `Kriging-DEIM.ipynb` — original exploratory notebook (shared, read-only).
  Contains 1D/2D/3D Kriging experiments and the initial CSSP implementation.
- `Kriging-DEIM-fixed.ipynb` — copy of the above with six bugs corrected.
- `Kriging-DEIM-notes.md` — cell-by-cell reference document for the notebook.
