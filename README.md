# Prediction-Based Lossy Compression — Gaussian Process & DEIM Methods

Research code exploring whether global, prediction-based reconstruction
methods (Gaussian Process / Kriging regression and DEIM low-rank bases) can
outperform general-purpose scientific-data compressors (SZ2, SZ3, ZFP) when
used as a predictor ahead of quantization and entropy coding. Applied to two
datasets: the ISABEL hurricane simulation (3D wind/temperature/cloud fields)
and NOAA OI SST V2 (weekly sea-surface temperature). Originally extracted
and extended from `Kriging-DEIM.ipynb`.

Scripts range from standalone reconstruction demos (Kriging, DEIM — no
baseline comparison) to full rate-distortion benchmarks against SZ2/ZFP/SZ3,
to an actively-evolving progressive/iterative GP compressor. See each
script's entry below for what it does, how to run it, and its top-of-file
configuration.

---

## Installation

```bash
pip install -r requirements.txt
```

This covers the pip-installable dependencies (NumPy, SciPy, Matplotlib,
scikit-learn, scikit-image, Pillow, zstandard, netCDF4). A handful of
scripts additionally need things that aren't on PyPI — a `libpressio` build,
a built `sz3` binary, an MPI implementation for `mpi4py`, and the sibling
`gpoed-code-python` package — see the bottom of `requirements.txt` for
details on each, and each script's **Usage** section below for which
scripts need which.

Data files (`*.bin.f32` hurricane volumes under `100x500x500/`, and the SST
NetCDF files) are not included in this repository — see
[Datasets](#datasets) below for where to get them and expected paths.

---

## Files

### `kriging_hurricane.py`

A self-contained Python script implementing GP regression (Simple Kriging)
compared across two scalable sensor-placement strategies on the full 3D
ISABEL U-wind volume. Extracted and adapted from `Kriging-DEIM.ipynb`.

The script loads all 100 vertical levels, skips the first 10 near-surface
levels (near-zero values inflate relative error), and randomly splits the
remaining levels into a training set (`N_TRAIN_FACTOR × n_sensors` levels,
seeded by `SPLIT_SEED`) and a held-out test set. Each spatial location is
normalised using its per-location training mean/std. Kernel hyperparameters
(lengthscale, variance, noise) are fit automatically by maximising the GP log
marginal likelihood (`scipy.optimize.minimize`, L-BFGS-B, multiple restarts)
on a random subsample of the showcase training level.

**GP method**

- **Simple Kriging** — assumes a known constant prior mean (0). Also
  implements Universal/Ordinary Kriging and a greedy oracle sensor selector,
  though the current `__main__` only exercises Simple Kriging.

**Sensor placement methods compared**

- **GKS via Randomly Pivoted Cholesky** — builds a low-rank Cholesky factor of
  the covariance matrix via random pivoting, then applies CSSP (SVD + QR
  column pivoting) to the factor. Scalable to large grids without forming the
  full n×n covariance matrix.
- **GKS via Greedy Pivoted Cholesky** — same RPGKS second stage, but the
  low-rank factor is built by deterministically picking the point with
  largest residual variance at each step (via `pivoted_cholesky` /
  `rpgks` from the external `gpoed-code-python` package).

**Covariance kernels**

- **RBF** (squared-exponential / Gaussian)
- **Matérn** with tunable smoothness ν ∈ {0.5, 1.5, 2.5}

**Usage**

```bash
pip install numpy scipy matplotlib scikit-learn
python kriging_hurricane.py
```

Requires the external `gpoed-code-python` package on `sys.path` (hardcoded to
`/Users/jchen228/Desktop/gpoed-code-python`) for `pivoted_cholesky` and
`rpgks`. Outputs a `kriging_results.png` plot (2×3 grid: true field /
reconstruction / absolute error, for each of the two placement methods) and
prints relative L2 reconstruction error for both.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `DOWNSAMPLE` | `3` | Keep every Nth pixel; 3 → ~167×167 ≈ 27,889 points |
| `N_SENSORS` | `None` | Number of sensors; `None` → auto 1% of grid points |
| `KERNEL` | `'matern'` | `'rbf'` or `'matern'` |
| `MATERN_NU` | `1.5` | Matérn smoothness: `0.5`, `1.5`, or `2.5` |
| `N_TRAIN_FACTOR` | `4` | `n_train = N_TRAIN_FACTOR × n_sensors` |
| `SPLIT_SEED` | `42` | Random seed for the train/test level split |
| `SHOWCASE_TRAIN_IDX` | `49` | Which training level to reconstruct/display |

**Tips for better reconstruction**

- Lower `DOWNSAMPLE` for finer spatial resolution before tuning anything else.
- Increase `N_SENSORS` — detail recoverable scales roughly with sensor count.
- The hurricane wind field is non-stationary (sharp gradients near the eye wall, smooth elsewhere). A single global lengthscale cannot resolve both regimes simultaneously; this is a fundamental limitation of stationary GP models.

---

### `splitgp_hurricane.py`

Extends `kriging_hurricane.py` with optional domain splitting: the spatial
domain is partitioned into rectangular subdomains, each fitted with its own
GP and hyperparameters. This captures non-stationary behaviour (e.g. sharp
eye-wall gradients vs. calm outer regions) that a single global GP cannot
handle. Operates on a single 2D horizontal slice (`SLICE_LEVEL`), unlike
`kriging_hurricane.py`'s full 3D train/test split.

**GP methods**: Simple Kriging, Ordinary Kriging, Universal Kriging (as in
`kriging_hurricane.py`).

**Sensor placement methods**: CSSP/GKS, MaxMin ordering, and Greedy error
(RPCholesky+CSSP not yet wired in here). When `SPLIT_DOMAIN = True`, sensors
can be shared across neighbouring subdomains within `BORDER_BUFFER` grid
units of a subdomain's boundary.

**Usage**

```bash
pip install numpy scipy matplotlib scikit-learn
python splitgp_hurricane.py
```

Saves `splitgp_results.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `SLICE_LEVEL` | `50` | Vertical level to slice (0–99) |
| `DOWNSAMPLE` | `7` | Keep every Nth pixel |
| `N_SENSORS` | `75` | Number of sensors to place |
| `LENGTHSCALE` | `5.0` | Kernel lengthscale (grid-index units) |
| `VARIANCE` | `10.0` | Kernel signal variance |
| `NOISE` | `1e-3` | Observation noise / regularisation nugget |
| `KERNEL` | `'matern'` | `'rbf'` or `'matern'` |
| `MATERN_NU` | `0.5` | Matérn smoothness: `0.5`, `1.5`, or `2.5` |
| `FIT_ON` | `'random'` | Sensor set used to fit hyperparameters: `'maxmin'`, `'uniform'`, `'random'`, or `'greedy'` |
| `NORMALIZE` | `True` | Subtract mean and divide by std before Kriging |
| `SPLIT_DOMAIN` | `True` | `True` → per-subdomain GP; `False` → single global GP |
| `N_DOMAIN_ROWS` | `2` | Rows of rectangular subdomains |
| `N_DOMAIN_COLS` | `2` | Columns of rectangular subdomains (2×2 = 4 subdomains) |
| `BORDER_BUFFER` | `10` | Grid units of sensor sharing across subdomain borders (`0` = no sharing) |
| `RUN_GREEDY` | `False` | `True` → compute greedy placement (slow); `False` → MaxMin stands in |
| `GREEDY_MAX_CANDS` | `500` | Candidate subsample size per greedy step when `RUN_GREEDY=True` |
| `FIT_RESTARTS` | `3` | L-BFGS-B restarts inside `fit_hyperparams` |

---

### `deim_hurricane.py`

Reconstructs ISABEL U-wind fields using two reduced-order sensor-placement
methods: **Q-DEIM** (Discrete Empirical Interpolation with QR-based sensor
placement, mirroring the Q-DEIM block in `script_sst.m` from
`gpoed-code-python`) and **R-DEIM** (Randomized DEIM, Saibaba 2020,
arXiv:1903.00911), which replaces the full SVD with a randomized range
finder.

The number of sensors `k` is set automatically from a POD energy bound:
`adaptive_k` finds the smallest `k` such that the relative Frobenius-norm
projection error over the training ensemble is below `ERROR_TOL`. Levels are
split into a training set (used for the SVD basis) and a held-out test set;
each test level is reconstructed from `k` sensor readings alone. Three
representative test cases (lowest / median / highest error, ranked by the
first method in `METHODS`) are shown in the showcase figure.

**Usage**

```bash
pip install numpy scipy matplotlib
python deim_hurricane.py
```

Saves `deim_showcase.png` (and related diagnostic plots such as
`deim_results.png`, `deim_singular_values.png`, `deim_surface_levels.png`,
`deim_error_distribution.png` from prior runs).

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `DOWNSAMPLE` | `3` | Keep every Nth pixel per level |
| `NORMALIZE` | `True` | Subtract training mean before decomposing |
| `ERROR_TOL` | `1e-2` | Relative POD projection error threshold for `adaptive_k` |
| `MAX_SENSORS` | `None` | Cap on sensor count; `None` → auto 1% of grid points |
| `N_TRAIN_FACTOR` | `4` | `n_train = N_TRAIN_FACTOR × k_pilot` (3–5 recommended) |
| `SPLIT_SEED` | `42` | Reproducible train/test level split |
| `METHODS` | `['Q-DEIM']` | Which methods to run: `'Q-DEIM'`, `'R-DEIM'`, or both |
| `RDEIM_OVERSAMPLE` | `10` | Oversampling `p` for R-DEIM's randomized range finder |
| `RDEIM_N_ITER` | `1` | Subspace iterations `q` for R-DEIM |
| `RDEIM_SEED` | `0` | R-DEIM random seed |

---

### `deim_hurricane_v2.py`

A trimmed-down version of `deim_hurricane.py`. The original also implemented
R-DEIM and a dynamic multi-method figure layout, but since the active
configuration only ever ran Q-DEIM (`METHODS = ['Q-DEIM']`), that dead code
path was removed here. This version is Q-DEIM only, with a fixed 3-column
figure layout (true field / reconstruction / absolute error). Same
adaptive-`k` sensor count, train/test level split, and showcase logic as
`deim_hurricane.py`.

Also produces an SZ-style quantization histogram of the reconstruction
residuals: each residual is binned as `round(residual / (2 × eb))` with
`eb = ERROR_TOL × field_range` (mirroring the SZ-1.4 lossy-compression error
model), and the plot reports the share of out-of-codebook outliers.

**Usage**

```bash
pip install numpy scipy matplotlib
python deim_hurricane_v2.py
```

Saves `deim_hurricane_v2_showcase.png` and `deim_v2_quantization_hist.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/CLOUDf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `DOWNSAMPLE` | `3` | Keep every Nth pixel per level |
| `NORMALIZE` | `True` | Subtract training mean before decomposing |
| `ERROR_TOL` | `1e-2` | Relative POD projection error threshold for `adaptive_k` (also sets the SZ quantization bin width) |
| `MAX_SENSORS` | `None` | Cap on sensor count; `None` → auto 1% of grid points |
| `N_TRAIN_FACTOR` | `4` | `n_train = N_TRAIN_FACTOR × k_pilot` (3–5 recommended) |
| `SPLIT_SEED` | `42` | Reproducible train/test level split |

---

### `tdeim_hurricane.py`

Tensor DEIM (T-DEIM): reconstructs the full 3D ISABEL U-wind field from `k`
sensors scattered anywhere in the 3D volume — each sensor has its own
(level, y, z) position rather than being restricted to a single 2D plane, as
in `deim_hurricane.py`. One single solve using all `k` readings reconstructs
all 100 levels at once, and the algorithm naturally spreads sensors toward
the levels where the field varies most.

The basis is built by mode-1 unfolding the (normalised) field into an
`(n_levels, n_spatial)` matrix, taking its thin SVD, and forming rank-1 3D
modes `phi_j = flatten(outer(U_L[:,j], Vt[j,:]))`. Q-DEIM (QR column
pivoting) places sensors on the stacked basis matrix; sensor indices are then
unravelled back into `(level, y, z)` coordinates.

`k` is chosen by one of three "guarantee modes" (`GUARANTEE_MODE`): `'trunc'`
(fast POD-energy truncation only), `'apriori'` (Drmač & Gugercin 2016 a
priori bound that also accounts for the DEIM interpolation/Lebesgue
constant, capped to avoid runaway rank), or `'abs'` (choose `k` so the
estimated RMSE in physical units meets `ABS_TOL`, in the spirit of the SZ-1.4
error-bounded compressor). `OVERSAMPLE` adds extra sensors beyond `k` and
solves an overdetermined least-squares system instead of the classic square
DEIM solve, for extra robustness.

Produces: (1) a single-slice comparison of T-DEIM vs. 2D DEIM at
`SHOW_SLICE`, (2) sensor distribution across levels and in the (lat, lon)
plane, (3) — currently disabled — a full 3D volumetric scatter plot of the
reconstruction (`_SHOW_FIG4` flag near Figure 4 in `__main__`; set `True` to
re-enable), (4) a per-level relative-error bar chart flagging levels that
exceed the guarantee target, and (5) an SZ-style quantization histogram of
reconstruction residuals (same binning scheme as `deim_hurricane_v2.py`).
The old best/median/worst level-99 showcase figure has been replaced by the
per-level error chart.

**Usage**

```bash
pip install numpy scipy matplotlib
python tdeim_hurricane.py
```

Saves `tdeim_comparison.png`, `tdeim_sensor_distribution.png`,
`tdeim_level_errors.png`, and `tdeim_quantization_hist.png` (plus
`tdeim_3d_volume.png` if `_SHOW_FIG4` is enabled).

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/CLOUDf48.bin.f32` | Path to the ISABEL `.bin.f32` file |
| `DOWNSAMPLE` | `3` | Spatial downsample per level |
| `SKIP_LEVELS` | `0` | Levels excluded from the basis/errors (0 = all 100 included) |
| `SHOWCASE_MIN_LEVEL` | `10` | Excludes levels 0–9 from best/median/worst ranking (near-zero surface values inflate relative error) |
| `SHOW_SLICE` | `50` | Global level index (0–99) shown in Figure 1 |
| `ERROR_TOL` | `1e-2` | Tolerance used by `GUARANTEE_MODE` (meaning depends on mode) |
| `MAX_SENSORS` | `None` | Cap on sensor count; `None` → auto 1% of 3D volume |
| `VIZ_DOWNSAMPLE` | `2` | Extra spatial downsample for the (disabled) 3D volume figure |
| `LEVEL_STEP` | `2` | Render every Nth level in the 3D volume figure |
| `OVERSAMPLE` | `10` | Extra sensors beyond `k`; `0` → exact square DEIM solve, `>0` → least-squares solve |
| `ABS_TOL` | `0.001` | Absolute RMSE target (m/s); used only when `GUARANTEE_MODE='abs'` |
| `GUARANTEE_MODE` | `'apriori'` | `'trunc'`, `'apriori'`, `'abs'`, or `None` — how `k` is chosen |

---

### `multigp_hurricane.py`

Multi-output Gaussian Process (Linear Model of Coregionalization, LMC) for
joint reconstruction of multiple ISABEL hurricane variables (default: U, V,
W wind components) simultaneously from sparse point-sensor observations.
Because the variables are physically coupled (pressure gradients drive
horizontal circulation, which organises vertical convection, etc.), a single
multi-output GP captures cross-variable correlations that independent
per-variable GPs cannot: observing one variable at a sensor also constrains
the others there.

The LMC kernel is `k((x,i),(x',j)) = B[i,j] × k_spatial(x,x')`, where `B` is
a cross-variable covariance matrix and `k_spatial` is a Matérn-3/2 spatial
kernel, giving the full kernel Kronecker structure `K_full = B ⊗ K_spatial`
for efficient likelihood evaluation and prediction. Training estimates `B`
(sample cross-variable covariance) and the spatial lengthscale (via marginal
likelihood maximisation) from a random subset of training levels where all
variables are fully observed; sensors are then placed by GKS (QR pivoting on
`K_spatial`). A held-out test level is reconstructed from all variables
observed at only the placed sensor locations.

**Usage**

```bash
pip install numpy scipy matplotlib
python multigp_hurricane.py
```

Saves `multigp_results.png` (reconstruction panels),
`multigp_correlation.png` (cross-variable correlation), and
`multigp_quantization_hist.png` (SZ-style quantization histogram of GP
residuals per variable, same binning scheme as `deim_hurricane_v2.py`).

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `…/100x500x500` | Directory containing the `.bin.f32` variable files |
| `VARIABLES` | `{'U':'Uf48.bin.f32','V':'Vf48.bin.f32','W':'Wf48.bin.f32'}` | Display name → filename map of variables to reconstruct jointly |
| `UNITS` | `{'U':'m/s','V':'m/s','W':'m/s'}` | Units used in plot labels |
| `DOWNSAMPLE` | `3` | Spatial downsampling; 3 → ~167×167 ≈ 27,889 points per variable |
| `N_TRAIN_FACTOR` | `4` | `n_train = N_TRAIN_FACTOR × n_sensors` |
| `SPLIT_SEED` | `42` | Reproducible random split |
| `SHOWCASE_TRAIN_IDX` | `50` | Which training level to reconstruct/display |
| `N_SENSORS` | `None` | Number of sensors; `None` → auto 1% of grid points |
| `NOISE_STD` | `0.05` | Observation noise std in normalised units (~5% of signal) |
| `N_LS_RESTARTS` | `3` | Restarts for lengthscale MLE optimisation |
| `JITTER` | `1e-6` | Numerical jitter added to `K_spatial` diagonal |

---

### `hurricane_progressive_gp.py`

Progressive/iterative 3D GP compression on the ISABEL U-wind field: instead of
placing all sensors in one shot, the domain is compressed in rounds, each
round sensing only the points that are still poorly predicted. Each round:
(1) **sense** — places `K_PER_ROUND` new sensors from the remaining
candidate pool using `SENSOR_STRATEGY`; (2) **predict** — fit a GP on all
accumulated sensor values and predict everywhere; (3) **compress** — points
where `|pred − true| < ACCEPT_BOUND` (`ACCEPT_BINS × ERROR_BOUND`) are marked
done and removed from the pool; (4) **repeat** for `N_ROUNDS` rounds. A final
decompression test collects all sensors from all rounds and runs one GP over
the full domain to simulate what a decoder reconstructs.

Sensor placement supports four strategies (`SENSOR_STRATEGY`): `'rpgks'`
(randomized-pivoted GKS / mutual-information maximisation, geometry-only);
`'max_residual'` (top-k largest current GP error); `'rpgks_residual'` (rpgks
restricted to the worst `RESID_PERCENTILE`% of candidates by error); and
`'hybrid'` (rpgks for the first `N_RPGKS_ROUNDS` rounds for broad coverage,
then `rpgks_residual` to collapse the error tails). An optional `USE_KRONECKER`
mode places sensors on a tensor-product `(z-levels × xy-positions)` grid with
a separable kernel, trading sensor-placement freedom for much cheaper
prediction at large `k`.

**Usage**

```bash
python3 hurricane_progressive_gp.py
```

Fits/caches 3D kernel hyperparameters to `pgp_hyperparams_ds{DOWNSAMPLE}.pkl`.
Produces panel and histogram figures per round (a static figure if
`N_ROUNDS <= 3`, otherwise per-round video frames stitched to MP4/GIF, plus
standalone `.gif` copies), a 3D volumetric comparison of the final
reconstruction vs. the true field, and a pickled checkpoint
(`pgp_checkpoint_R{N_ROUNDS}_k{K_PER_ROUND}_eb{ERROR_BOUND}_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}...pkl`)
consumed by `sz3_gp_comparison.py` / `hurricane_gp_comparison.py`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_ROUNDS` | `20` | Number of progressive sensing rounds |
| `K_PER_ROUND` | `250` | Sensors selected per round |
| `ERROR_BOUND` | `0.01` | Quantisation bin half-width (m/s); SZ2 bin width = 2×EB |
| `ACCEPT_BINS` | `10` | Central EB-bins counted as "compressed" (`ACCEPT_BOUND = ACCEPT_BINS × ERROR_BOUND`) |
| `DOWNSAMPLE` | `3` (overridden to `10` by `QUICK_TEST`) | Spatial downsampling factor |
| `ACCUMULATE` | `True` | `True` → sensors accumulate across rounds; `False` → fresh `k` sensors each round |
| `SENSOR_STRATEGY` | `'hybrid'` | `'rpgks'`, `'max_residual'`, `'rpgks_residual'`, or `'hybrid'` |
| `N_RPGKS_ROUNDS` | `1` | (hybrid only) rounds of rpgks before switching to `rpgks_residual` |
| `RESID_PERCENTILE` | `25` | (rpgks_residual/hybrid) rpgks draws from the worst X% of candidates by `\|error\|` |
| `USE_KRONECKER` | `False` | `True` → separable tensor-grid GP (fast, sensors constrained to xy-plane grid); `False` → free 3D rpgks |
| `K_Z_PER_ROUND` | `4` | (Kronecker only) z-levels added per round |
| `QUICK_TEST` | `True` | `True` → forces `DOWNSAMPLE = 10` (250k points, ~20–40s/round) for fast iteration |

---

### `hurricane_progressive_gp_2d.py`

A slice-by-slice variant of `hurricane_progressive_gp.py`: instead of one 3D
GP over the whole volume, each horizontal z-slice is treated as an
independent 2D progressive-GP problem (2D anisotropic Matérn kernel over
`(y, x)` only, same sense→predict→compress round loop as the 3D script,
run separately per slice). Rationale (from the module docstring): with a
fixed sensor budget spread across `NZ` slices, the 3D approach gives each
slice only ~2–3 sensors; the 2D approach dedicates `K_PER_ROUND_2D` sensors
*per slice per round*, giving much denser in-plane coverage at the cost of
ignoring vertical (z) correlation entirely. Saves checkpoints in the same
pickle format as `hurricane_progressive_gp.py` so `sz3_gp_comparison.py` can
load and compare either one (`USE_2D` flag).

**Usage**

```bash
python3 hurricane_progressive_gp_2d.py
```

Saves a checkpoint (`pgp2d_checkpoint_R{N_ROUNDS}_k{K_PER_ROUND_2D}_eb{ERROR_BOUND}_ab{ACCEPT_BINS}_ds{DOWNSAMPLE}_zskip{Z_SKIP_BOTTOM}.pkl`)
and `progressive_gp_2d_histograms.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_ROUNDS` | `10` | Progressive rounds per slice |
| `K_PER_ROUND_2D` | `10` | Sensors per slice per round (total sensors = `K_PER_ROUND_2D × NZ × N_ROUNDS`) |
| `ERROR_BOUND` | `0.01` | Quantiser bin half-width (m/s) |
| `ACCEPT_BINS` | `10` | Voxels with `\|error\| < ACCEPT_BINS × EB` are compressed |
| `DOWNSAMPLE` | `10` | Spatial downsampling |
| `SENSOR_STRATEGY` | `'hybrid'` | Same options as the 3D script |
| `Z_SKIP_BOTTOM` | `0` | Number of bottom z-levels excluded (`0` = use all 100) |
| `KERNEL_TYPE` | `'matern12'` | `'matern52'`, `'matern32'`, or `'matern12'` (2D only, no z component) |
| `LS_XY` | `0.15` | Horizontal correlation length in `[0,1]` coords |

---

### `hurricane_gp_comparison.py`

A unified successor to `hurricane_progressive_gp.py` / `hurricane_progressive_gp_2d.py`:
one script that runs *either* mode (`MODE = '2d'` or `'3d'`) and directly
sweeps a full rate-distortion curve against SZ3 (`RUN_MODE = 'rd'` sweeps
`EB_VALUES`; `RUN_MODE = 'ls'` instead sweeps `LS_XY_VALUES` at a fixed EB to
tune the lengthscale first). Adds target-based stopping in place of a fixed
round count: `TARGET_CR` / `TARGET_PSNR` (stop once the estimated
compression ratio or PSNR is reached) or `TARGET_BEAT_SZ3` (run SZ3 first,
then keep adding rounds until GP PSNR meets or beats it), up to `MAX_ROUNDS`.
Also introduces `SENSOR_STORAGE`, which controls how sensor values
themselves are stored/represented (`'float32'`, `'float16'`, or
`'eb_quant'` — entropy-coded EB-quantised, matching encoder/decoder
representations exactly). GP runs are cached per `(mode, EB)` checkpoint and
reloaded rather than recomputed on subsequent runs.

**Usage**

```bash
python3 hurricane_gp_comparison.py
```

Saves `rd_curve_{MODE}.png` (RD sweep) or `ls_sweep_{MODE}.png` (lengthscale
sweep), depending on `RUN_MODE`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `MODE` | `'2d'` | `'2d'` slice-by-slice or `'3d'` full-volume GP |
| `RUN_MODE` | `'rd'` | `'rd'` sweeps `EB_VALUES`; `'ls'` sweeps `LS_XY_VALUES` at fixed `LS_EB` |
| `EB_VALUES` | `[0.005, 0.01, 0.02, 0.05, 0.1]` | Error bounds swept in RD mode |
| `TARGET_CR` / `TARGET_PSNR` | `None` / `None` | Stop rounds early once reached (`None` → use fixed `N_ROUNDS`) |
| `TARGET_BEAT_SZ3` | `False` | `True` → run SZ3 first, use its PSNR as the per-EB stopping target |
| `MAX_ROUNDS` | `30` | Hard cap when using target-based stopping |
| `N_ROUNDS` | `10` | Used only when all targets are `None` |
| `SENSOR_STRATEGY` | `'hybrid'` | Same options as `hurricane_progressive_gp.py` |
| `SENSOR_STORAGE` | `'eb_quant'` | `'float32'`, `'float16'`, or `'eb_quant'` — how sensor values are represented for both encode and decode |
| `DOWNSAMPLE` | `10` | Spatial downsampling |
| `KERNEL_TYPE` | `'matern52'` | `'matern52'`, `'matern32'`, or `'matern12'` |

---

### `sz3_gp_comparison.py`

Loads a progressive-GP checkpoint (from either `hurricane_progressive_gp.py`
or `hurricane_progressive_gp_2d.py`, via the `USE_2D` flag) and builds a
rate-distortion curve against SZ3, run as an external binary on the same
volume. Sensor positions are fixed from the checkpoint; for each `EB` in
`EB_VALUES`, the GP's final-round predictions are re-quantized at that `EB`
(`quantize_sz2`: bin width `2×EB`, unbounded integer bin index so every
point's reconstruction error is bounded by construction — no separate
outlier/raw-storage path is needed since the bin index itself absorbs
arbitrarily large residuals) — no GP re-run needed per `EB`. Reports bit
rate (sensor overhead + entropy of the quantized bins), PSNR, and RMSE for
both methods at each `EB`.

**Usage**

```bash
python3 sz3_gp_comparison.py
```

Requires a checkpoint from `hurricane_progressive_gp.py` /
`hurricane_progressive_gp_2d.py` to already exist, and the `SZ3_BIN` path
(hardcoded near the top of the file) to point at a built SZ3 binary. Saves
`sz3_gp_rd_curve.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `USE_2D` | `True` | `True` → load a 2D slice-by-slice checkpoint; `False` → load a 3D checkpoint |
| `CHECKPOINT_FILE` | *(path, depends on `USE_2D`)* | Which progressive-GP checkpoint to compare |
| `SZ3_BIN` | *(local path)* | Path to a built SZ3 binary |
| `EB_VALUES` | `[0.005, 0.01, 0.02, 0.05, 0.1]` | Error bounds swept for the RD curve |
| `BITS_PER_SENSOR` | `64` | Bits charged per stored sensor (`uint32` index + `float32` value) |

---

### `lp_deim_compressor.py`

A `libpressio` **external compressor** plugin implementing Q-DEIM (Drmač &
Gugercin 2016), extracted from `deim_hurricane.py` into a standalone
train/compress/decompress CLI so it can be driven by `libpressio`'s
`external:command` / `external:decompressor_command` protocol. Offline
`train` does an SVD of training slices to build a POD basis `U_k` (size `k`
chosen by the POD energy criterion `sqrt(1-Σs[:k]²/Σs²) ≤ error_bound`) and
places `k` sensors via QR column-pivoting on `U_k^T`; the model is saved to a
`.npz` file. `compress` stores just the `k` sensor readings per slice;
`decompress` solves the `k×k` system `U_k[sensors,:] c = y_s` and
reconstructs `û = U_k c`.

**Usage**

```bash
# 1. Train once:
python3 lp_deim_compressor.py train \
    --data /path/to/Uf48.bin.f32 --shape 100 500 500 \
    --model /path/to/deim_model.npz --error-bound 0.01 \
    [--downsample 3] [--skip-levels 10] [--n-train-factor 4]

# 2. Compress (libpressio external-compressor protocol):
python3 lp_deim_compressor.py compress <input.bin> <output.bin> --model /path/to/deim_model.npz

# 3. Decompress:
python3 lp_deim_compressor.py decompress <compressed.bin> <output.bin> --model /path/to/deim_model.npz
```

Compressed layout: `int32 k`, `int32 n`, then `k` float32 sensor values.

---

### `lp_kriging_compressor.py`

A `libpressio` external compressor implementing Simple Kriging with a
Matérn-3/2 (or RBF) spatial kernel, extracted from `kriging_hurricane.py`.
`train` chooses sensor count `k` with the same POD energy criterion as DEIM
(for apples-to-apples comparison), fits the lengthscale/variance by marginal
likelihood maximisation, and places sensors via GKS (QR column-pivoting on
the top-`k` right singular vectors of the kernel matrix). `decompress`
computes the Simple Kriging posterior mean at all spatial locations from the
`k` stored sensor readings.

**Usage**

```bash
python3 lp_kriging_compressor.py train \
    --data /path/Uf48.bin.f32 --shape 100 500 500 \
    --model /path/kriging_model.npz --error-bound 0.01 \
    [--downsample 3] [--kernel matern] [--matern-nu 1.5]

python3 lp_kriging_compressor.py compress <input.bin> <output.bin> --model /path/kriging_model.npz
python3 lp_kriging_compressor.py decompress <compressed.bin> <output.bin> --model /path/kriging_model.npz
```

Compressed layout: `int32 k`, `int32 n`, then `k` float32 sensor values
(mean-subtracted if `normalise=True`).

---

### `lp_multigp_compressor.py`

A `libpressio` external compressor implementing the multi-output GP / Linear
Model of Coregionalization from `multigp_hurricane.py`, for jointly
compressing several physically-coupled fields (e.g. U/V/W wind components) at
shared sensor locations. `train` estimates the cross-field covariance matrix
`B`, fits a shared spatial Matérn-3/2 lengthscale via Kronecker-structured
marginal likelihood, and places `k` sensors by GKS on the spatial kernel;
`k` is chosen by the POD energy criterion on the concatenated fields. Input
for `compress`/`decompress` is a stacked binary of `d` consecutive float32
arrays (`cat U.bin V.bin W.bin > UVW_stacked.bin`).

**Usage**

```bash
python3 lp_multigp_compressor.py train \
    --data-dir /path/to/100x500x500 \
    --variables Uf48.bin.f32 Vf48.bin.f32 Wf48.bin.f32 \
    --shape 100 500 500 --model /path/multigp_model.npz --error-bound 0.01 \
    [--downsample 14] [--noise-std 0.05]

python3 lp_multigp_compressor.py compress <stacked.bin> <output.bin> --model /path/multigp_model.npz
python3 lp_multigp_compressor.py decompress <compressed.bin> <output.bin> --model /path/multigp_model.npz
```

Compressed layout: `int32 k`, `int32 d`, `int32 n`, then `k×d` float32 sensor
values (z-score space, row-major).

---

### `lp_tdeim_compressor.py`

A `libpressio` external compressor implementing Tensor DEIM (mode-1 unfolding
+ Q-DEIM), extracted from `tdeim_hurricane.py`. Unlike
`lp_deim_compressor.py`, sensors can sit anywhere in the full 3D volume
(not fixed to a single 2D slice), and one solve reconstructs all levels at
once. `train` builds rank-1 3D POD modes from the mode-1 SVD of training
slices and places `k` sensors via QR pivoting on the stacked 3D basis, with
`k` set by the same POD energy criterion as the other compressors.

**Usage**

```bash
python3 lp_tdeim_compressor.py train \
    --data /path/Uf48.bin.f32 --shape 100 500 500 \
    --model /path/tdeim_model.npz --error-bound 0.01 \
    [--downsample 3] [--oversample 10]

python3 lp_tdeim_compressor.py compress <input.bin> <output.bin> --model /path/tdeim_model.npz
python3 lp_tdeim_compressor.py decompress <compressed.bin> <output.bin> --model /path/tdeim_model.npz
```

Compressed layout: `int32 k`, `int32 n`, then `k` float32 sensor values.

---

### `rate_distortion_comparison.py`

The main head-to-head rate-distortion benchmark: compares SZ2, ZFP,
DEIM-2D, T-DEIM, Kriging-2D, and MultiGP on the ISABEL U-wind (+ V-wind for
MultiGP's second field, `d=2`) fields under a shared, fair SZ2-style
compression pipeline. For the model-based methods
(T-DEIM/Kriging-2D/DEIM-2D/MultiGP): predict the field from `k` sensors,
SZ-style-quantize the residual (`bin_width = 2×abs_bound / num_bins`), then
Zstd-compress the quantized bins plus raw float32 outliers; sensor values
are quantized/dequantized the same way to simulate a realistic decompressor.
Compressed-size accounting includes the trained model itself
(bases/covariance/Cholesky factors, float16/float32 + Zstd) so all methods'
compression ratios are directly comparable. Sweeps a shared error-bound / `k`
grid per method and writes one CSV row per (method, operating point). (Note:
the file's module docstring and some in-file comments still describe an
older CLOUDf/QVAPORf configuration; the config block itself currently points
at Uf48/Vf48.)

**Usage**

```bash
python3 rate_distortion_comparison.py
```

Set `PLOTS_ONLY = True` in the config block to skip re-running compression
and just re-plot from the existing `rd_results_{FIELD_TAG}.csv`. Produces
rate-distortion plots (PSNR/SSIM vs. compression ratio) and
`rd_results_{FIELD_TAG}.csv`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Primary field |
| `DATA_PATH2` | `…/Vf48.bin.f32` | Second field, used only by MultiGP (d=2) |
| `SHAPE` | `(100, 500, 500)` | Full dataset shape |
| `LEVEL` | `50` | Z-slice used for 2D PSNR/SSIM metrics |
| `DS` | `3` | Spatial downsample for SZ2/ZFP/DEIM-2D/Kriging-2D/MultiGP |
| `DS_TDEIM` | `10` | Coarser downsample for T-DEIM (3D basis QR is expensive) |
| `ABS_BOUNDS` / `ZFP_ABS_BOUNDS` | `np.logspace(-4, -0.5, 8)` | Error-bound sweep controlling residual quantization / ZFP accuracy |
| `TDEIM_K_VALUES` | `[1,2,5,10,25,50]` | Sensor counts swept for T-DEIM |
| `DEIM2D_K_VALUES` | `[1,2,5,10,25,50,100]` | Sensor counts swept for DEIM-2D |
| `KRIG2D_K_VALUES` | `[1,5,10,25,50,100,200,300,500]` | Sensor counts swept for Kriging-2D |
| `MULTIGP_K_VALUES` | `[1,5,10,25,50,100,200,300]` | Sensor counts swept for MultiGP |
| `NUM_BINS_VALUES` | `[256, 4096, 65536]` | Quantization bin counts swept for DEIM-2D residuals |
| `RUN_TDEIM` / `RUN_MULTIGP` | `True` | Toggle to skip slower methods |
| `PLOTS_ONLY` | `False` | `True` → skip compression, re-plot from existing CSV |

---

### `hurricane_rd_comparison.py`

A four-method rate-distortion comparison focused on the ISABEL TC
(temperature) field, structured like `rate_distortion_comparison.py` but
with no land mask (all 500×500 pixels active) and no vertical/T-DEIM
dimension exploited: SZ2 and ZFP (2D slice-wise), DEIM-2D+SZ2 (spatial SVD
basis, residual coded with SZ2), and Kriging-2D+SZ2 (spatial Matérn-3/2 GP
with RPCholesky sensor placement, residual coded with SZ2). Each model
trains once on all 100 vertical levels, then is evaluated per-level: predict
from `k` sensors, compress the residual with SZ2. MultiGP (LMC over the
U/V/W wind fields, reusing `multigp_hurricane.py` functions) is run and
plotted separately alongside the four TC methods. GP hyperparameters are
fit by MLE on a random subsample of 2000 pixels.

**Usage**

```bash
python3 hurricane_rd_comparison.py
```

Reads `100x500x500/TCf48.bin.f32` (and `Uf48.bin.f32`/`Vf48.bin.f32`/
`Wf48.bin.f32` for the MultiGP panel). Produces `poster_rd_psnr_Hurricane.png`,
`poster_field_panel_Hurricane.png`, `recon_panel_Hurricane.png`,
`multigp_hurricane_rd.png`, and `rd_results_Hurricane.csv`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_LEVELS` / `N_TRAIN` | `100` / `100` | Total vertical levels; DEIM trains its SVD basis on all of them |
| `DEIM_K_VALS` | `[10, 20, 40, 60, 80]` | Spatial mode counts swept for DEIM-2D |
| `KRIG_K_VALS` | `[50, 100, 200, 400]` | Sensor counts swept for Kriging-2D |
| `ABS_BOUNDS` / `ABS_BOUNDS_SZ` | `np.logspace(-4, 0, 20)` | Error-bound sweep (TC spans ~60 °C) |
| `JITTER` | `1e-6` | GP kernel regularisation |
| `SKIP_LVLS` | `0` | Near-surface levels excluded (none for TC) |
| `TIME_IDX` | `50` | Vertical level shown in field panels |
| `VIZ_K_DEIM` / `VIZ_K_KRIG` | `60` / `200` | `k` shown in the DEIM / GP field panels |

---

### `hurricane_rd_3d.py`

A companion to `hurricane_rd_comparison.py` that instead uses a single
3D Kronecker-structured GP over the full `(100, 500, 500)` TC volume:
`K_3d((x1,z1),(x2,z2)) = K_spatial(x1,x2) · K_vertical(z1,z2)`, with
`k_xy` RPCholesky-selected horizontal sensors combined with `k_z`
evenly-spaced vertical levels. Prediction uses the Kronecker trick
(`MU_norm = K_xy_Xs @ (K_z_Xs @ alpha_mat).T`) so the full 25M-point
`K_Xs` matrix is never formed. Reuses the horizontal lengthscale/variance/
noise/sensor checkpoint from the 2D Kriging runs (`kriging_ckpt_Hurricane_N400.npz`)
and fits the vertical lengthscale separately from vertical profiles;
results are cached to `kriging_3d_ckpt_Hurricane.npz`. Compares against
SZ2/ZFP baselines run on the same full 3D volume.

**Usage**

```bash
python3 hurricane_rd_3d.py
```

Set `PLOTS_ONLY = True` (default) to skip re-running compression and just
re-plot from `rd_results_3D_Hurricane.csv`. Produces
`poster_rd_psnr_3D_Hurricane.png` and `rd_results_3D_Hurricane.csv`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_T, NY, NX` | `100, 500, 500` | Full TC volume shape |
| `TIME_IDX` | `50` | Vertical level index used in field panels |
| `JITTER` | `1e-6` | GP kernel regularisation |
| `KRIG_CONFIGS` | `[(400, 10)]` | `(k_xy, k_z)` pairs to run — results are insensitive to this choice |
| `ABS_BOUNDS` | `np.logspace(-4, 0, 20)` | Error-bound sweep for SZ2/ZFP |
| `RUN_ZFP` | `True` | Toggle ZFP baseline |
| `PLOTS_ONLY` | `True` | `True` → skip compression, reload CSV, re-plot only |

---

### `poster_plots.py`

Generates high-resolution (300 DPI), publication/poster-quality figures from
the results of `rate_distortion_comparison.py`. Reads
`rd_results_{FIELD_TAG}.csv` plus the raw data files, re-runs each method's
reconstruction at the CSV row closest to `TARGET_CR`, and saves clean
side-by-side reconstruction/error-map panels, a sensor-location map, and
PSNR-vs-CR / SSIM-vs-CR curves styled for poster use (larger fonts, thicker
lines).

**Usage**

```bash
python poster_plots.py            # uses TARGET_CR defined in the config block
TARGET_CR=10 python poster_plots.py
```

Saves `poster_reconstruction_{FIELD_TAG}.png`, `poster_sensor_map_{FIELD_TAG}.png`,
`poster_rd_psnr_{FIELD_TAG}.png`, and `poster_rd_ssim_{FIELD_TAG}.png` to the
Argonne directory.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/CLOUDf48.bin.f32` | Primary field (must match the CSV being read) |
| `DATA_PATH2` | `…/QVAPORf48.bin.f32` | Second field, for MultiGP panels |
| `LEVEL` | `50` | Z-slice to visualise |
| `DS` / `DS_GP` / `DS_TDEIM` | `3` / `7` / `10` | Per-method spatial downsample (must match the run that produced the CSV) |
| `NUM_BINS` | `65536` | Must match the value used in `rate_distortion_comparison.py` |
| `DPI` | `300` | Output resolution |
| `TARGET_CR` | `5.0` | Compression ratio to select the CSV row/operating point per method |

---

### `poster_plots_wind.py`

Structurally identical copy of `poster_plots.py` (same functions:
`load_data`, `load_csv`, `recon_sz`/`recon_tdeim`/`recon_deim2d`/
`recon_kriging2d`/`recon_multigp`, `plot_reconstruction_comparison`,
`plot_sensor_maps`, `plot_rd_curves`, `plot_fig1_triple`,
`plot_fig2_bin_histogram`, `plot_fig3_big_panel`), reconfigured to target the
ISABEL U/V wind fields instead of CLOUD/QVAPOR. Reads
`rd_results_{FIELD_TAG}.csv` (produced by `rate_distortion_comparison.py`
run against the wind fields) and regenerates the same set of poster-quality
PNGs for the wind-field comparison.

**Usage**

```bash
python poster_plots_wind.py       # uses TARGET_CR defined in the config block
TARGET_CR=10 python poster_plots_wind.py
```

Saves `poster_reconstruction_{FIELD_TAG}.png`, `poster_sensor_map_{FIELD_TAG}.png`,
`poster_rd_psnr_{FIELD_TAG}.png`, and `poster_rd_ssim_{FIELD_TAG}.png` (`FIELD_TAG`
derived from `DATA_PATH`, e.g. `Uf48`).

**Configuration** — edit only the block at the top of the file (identical
layout to `poster_plots.py`, different data paths):

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/Uf48.bin.f32` | Primary field (must match the CSV being read) |
| `DATA_PATH2` | `…/Vf48.bin.f32` | Second field, for MultiGP panels |
| `LEVEL` | `50` | Z-slice to visualise |
| `DS` | `7` | Spatial downsample for all methods |
| `NUM_BINS` | `65536` | Must match the value used in `rate_distortion_comparison.py` |
| `DPI` | `300` | Output resolution |
| `TARGET_CR` | `5.0` | Compression ratio to select the CSV row/operating point per method |

---

### `run_fig3.py`

Small standalone driver that re-runs only `plot_fig3_big_panel` from
`poster_plots.py` without needing a real `libpressio` install — it stubs
`sys.modules['libpressio']` with `unittest.mock.MagicMock()` before importing
`poster_plots` as a module (the `if __name__ == "__main__"` guard in
`poster_plots.py` means importing it this way runs no compression, just
defines functions). Loads data and the results CSV via `poster_plots.load_data()`/
`load_csv()`, then calls `plot_fig3_big_panel` directly. Useful for quickly
regenerating just the big poster panel figure after tweaking its plotting
code, without re-running the full compression sweep.

**Usage**

```bash
python3 run_fig3.py
```

No command-line arguments or top-of-file config block; behavior follows
whatever `poster_plots.py`'s config and `rd_results_{FIELD_TAG}.csv` currently
contain.

---

### `sst_progressive_gp.py`

The SST analog of `hurricane_progressive_gp.py`: applies the same
progressive sense→predict→compress round loop, but per-snapshot on the NOAA
OI SST weekly-mean data instead of per-volume on the hurricane data. Each of
`T_PROCESS` snapshots (180×360, ocean pixels only — land handled separately)
is treated as an independent 2D field; the same GP kernel hyperparameters
(fit once via MLE across snapshots) are reused for every snapshot, while
sensor positions are chosen adaptively per snapshot via `rpgks`. The module
docstring notes a natural future extension — a separable space-time
Kronecker kernel (`k((t,p),(t',p')) = σ² · k_t(t,t') · k_xy(p,p')`) so
sensors could sit on a joint time×space tensor grid — flagged via the
`USE_SPACETIME_KRONECKER` placeholder but not yet implemented.

**Usage**

```bash
python3 sst_progressive_gp.py
```

Fits/caches kernel hyperparameters to `pgp_sst_hyperparams_matern52.pkl` and
saves a checkpoint (`pgp_sst_checkpoint_R{N_ROUNDS}_k{K_PER_ROUND}_eb{ERROR_BOUND}_ab{ACCEPT_BINS}_T{T_PROCESS}.pkl`).
Produces `sst_progressive_gp_panels.png` and `sst_progressive_gp_histograms.png`.
Set `PLOTS_ONLY = True` to skip recomputation and re-plot from an existing
checkpoint.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_ROUNDS` | `5` | Progressive sensing rounds per snapshot |
| `K_PER_ROUND` | `100` | Ocean sensors selected per round |
| `ERROR_BOUND` | `0.5` | Quantiser half-width (°C); SZ2 bin width = 2×EB |
| `ACCEPT_BINS` | `1` | Bins accepted as "compressed" (`ACCEPT_BOUND = ACCEPT_BINS × EB`) |
| `ACCUMULATE` | `True` | `True` → sensors accumulate round-over-round |
| `T_PROCESS` | `100` | Number of snapshots processed (`None` → all 1727) |
| `LS_XY` | `0.10` | Horizontal correlation length in `[0,1]²` (starting value; refit via MLE) |
| `USE_SPACETIME_KRONECKER` | `False` | Placeholder — space-time Kronecker kernel not yet implemented |
| `PLOTS_ONLY` | `False` | `True` → skip recomputation, reload checkpoint, re-plot only |

---

### `sst_rd_comparison.py`

Rate-distortion comparison analogous to `rate_distortion_comparison.py`, but
on a different dataset entirely: NOAA OI SST V2 weekly-mean sea-surface
temperature (`sst.wkmean.1990-present.nc`, 1727 weekly snapshots × 180 lat ×
360 lon, read from the sibling `gpoed-code-python` directory, not
`100x500x500/`). Compares SZ2, SZ3, ZFP, DEIM-2D, and Kriging-2D (no
T-DEIM/MultiGP, since there's only one field and no vertical-level dimension
to exploit). Land pixels are zeroed out via an ocean mask before compression,
and all PSNR/SSIM/RMSE metrics are computed over ocean pixels only. (Note:
the module docstring still says `N_TRAIN=260`; the config block now trains
on the full `N_TRAIN=1727` snapshots.)

By default `N_TRAIN=1727` (all snapshots) are used to fit each model (SVD
basis for DEIM, or Matérn-3/2 GP hyperparameters + RPCholesky sensor
placement for Kriging, with SVD/GP fits checkpointed to disk when
`USE_CHECKPOINT=True`); the fitted model reconstructs all 1727 time steps
from `k` sensor readings. Beyond the base SZ-style quantized residual (as in
the ISABEL scripts), DEIM-2D and Kriging-2D each have several residual-coding
variants swept via `METHOD_ORDER`: `-L2` (uniform quantization, the default
active variant), `-LP` (Laplacian quantizer), `-SO` (sensors-only, a single
scatter point per `k` with no residual coding), `-TT` (TTHRESH-style
coefficient quantization, off by default via `RUN_DEIM_TT`), and `+SZ2`/`+ZFP`
hybrid variants that residual-code with SZ2/ZFP instead (off by default via
`RUN_HYBRID`). DEIM reconstruction is vectorised across all time steps as a
single BLAS matmul; the GP path uses a batched Cholesky solve with a
memory-capped chunked matmul (≤ 1 GB peak). Compressed-size accounting
includes the trained model (bases/covariance/Cholesky, float16/32 + Zstd) so
all methods are directly comparable. `main()` resumes from `rd_results_SST.csv`
if present, skipping methods already recorded there.

**Usage**

```bash
python3 sst_rd_comparison.py
```

Set `PLOTS_ONLY = True` to skip re-running compression and just reload
`rd_results_SST.csv`. Produces `rd_psnr_cr_SST.png`, `rd_psnr_bpv_SST.png`,
`poster_rd_psnr_SST.png` (zoom-inset poster style), `rd_budget_breakdown_SST.png`
(stacked model/sensor/residual byte breakdown per `k`), `rd_timing_SST.png` and
`rd_decomp_timing_SST.png` (compress/decompress timing), a reconstruction
panel (true / DEIM / error), `poster_field_panel_SST.png` (true field / DEIM /
GP / residual histogram), and `rd_results_SST.csv`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/gpoed-code-python/sst.wkmean.1990-present.nc` | NOAA OI SST V2 NetCDF file (sibling directory to Argonne) |
| `N_TRAIN` | `1727` | Training snapshots used to fit DEIM/Kriging models (all snapshots by default) |
| `ABS_BOUNDS` | `np.logspace(-4, -0.5, 8)` | Absolute error-bound sweep (°C) used by SZ2's own accuracy mode |
| `ABS_BOUNDS_PRED` | `np.logspace(-4, 0, 20)` | Residual-quantization / ZFP sweep, 1e-4–1.0 °C, 20 points |
| `REL_BOUNDS` | `[2**-k for k in range(3, 17)]` | SZ3 relative error-bound sweep (12.5% … 1.5e-5) |
| `ZFP_BOUNDS` | `[2**-k for k in range(1, 14)]` | ZFP fixed-accuracy sweep, aligned to powers of 2 |
| `NUM_BINS` | `65536` | Quantization bin count (mirrors SZ2) |
| `DEIM_K_VALS` | `range(200, 1601, 200)` | Spatial SVD modes swept for DEIM-2D |
| `KRIG_K_VALS` | `range(200, 2001, 200)` | Sensor counts swept for Kriging-2D |
| `HIST_K` | `500` | Fixed `k` used in the residual-histogram panel |
| `TIME_IDX` | `864` | Snapshot index shown in field panels (~mid-dataset, July 2006) |
| `VIZ_K_DEIM` / `VIZ_K_KRIG` | `500` / `500` | `k` shown in field panels for DEIM-2D / Kriging-2D |
| `VIZ_AB` | `1e-2` | Abs-bound shown in field panel (closest match used) |
| `DPI` | `150` | Output resolution for plots |
| `RUN_LIBPRESSIO` | `True` | `False` → skip both SZ2 and ZFP |
| `RUN_ZFP` | `True` | `False` → skip ZFP only |
| `RUN_HYBRID` | `False` | `True` → also run DEIM/Kriging + SZ2/ZFP hybrid residual coding |
| `RUN_DEIM_TT` | `False` | `True` → also run DEIM-2D-TT (TTHRESH-style coefficient quantization) |
| `PLOTS_ONLY` | `False` | `True` → skip compression, reload CSV, re-plot only |
| `USE_CHECKPOINT` | `True` | `True` → cache SVD basis / GP hyperparams to disk, keyed by `N_TRAIN` |

---

### `lorenzo_vs_global_diagnostic.py`

Diagnostic-only script (no compression run) that isolates just the
prediction step and compares residual distributions for three predictors on
the SST dataset: **Lorenzo 2-D** (SZ2's local 2D linear finite-difference
predictor, `pred[i,j] = field[i-1,j] + field[i,j-1] - field[i-1,j-1]`),
**Lorenzo 1-D** (SZ2-1D's predecessor-based predictor over the flattened
ocean-pixel strip), **DEIM** (global SVD predictor loaded from the
checkpoint produced by `sst_rd_comparison.py`), and **GP** (global
Matérn-3/2 GP predictor, also loaded from checkpoint). Requires only the raw
SST NetCDF file plus the `deim_ckpt_SST_N1727.npz` / GP checkpoint files —
does not refit any model. Prints a summary table (RMSE, std, kurtosis,
outlier fractions at 2σ/3σ/5σ of the Lorenzo residual) for each method.

**Usage**

```bash
python3 lorenzo_vs_global_diagnostic.py
```

Requires `sst_rd_comparison.py` to have been run first (to produce the
checkpoint files) and `netCDF4` installed. Saves
`lorenzo_vs_global_residuals.png` (overlaid histograms + violin plots).

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_SNAPSHOTS` | `10` | Number of snapshots analysed (set to `1` for a quick run) |
| `DEIM_K` | `1600` | DEIM sensor/mode count to load from checkpoint |
| `GP_K` | `2000` | GP sensor count to load from checkpoint |
| `N_TRAIN` | `1727` | Must match the `N_TRAIN` used when the checkpoints were built |

---

### `residual_autocorr_diagnostic.py`

Another diagnostic-only script (no recomputation, no `libpressio` needed)
that loads the existing DEIM and GP checkpoints for the SST dataset and
examines residual structure at a fixed sensor count (`K_DIAG`) for one
representative snapshot (`TIME_IDX`): sensor selection is redone cheaply
(Q-DEIM QR pivoting for DEIM; GP sensors read from checkpoint) but the
expensive training step is skipped entirely. Produces a residual histogram
comparing DEIM vs. GP, and a 2D spatial autocorrelation map (plus radial
profile) of each method's residual field, to check whether residuals are
still spatially correlated (and thus further compressible) after
prediction.

**Usage**

```bash
python3 residual_autocorr_diagnostic.py
```

Run from the Argonne directory (or adjust `ARGONNE`/`DATA_PATH` at the top).
Requires the DEIM/GP checkpoints from `sst_rd_comparison.py` to already
exist. Saves `residual_distribution.png` and `residual_autocorr.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `…/sst.nc` | Same SST file used by `sst_rd_comparison.py` |
| `FIELD_TAG` | `'SST'` | Checkpoint filename tag |
| `N_TRAIN` | `1727` | Must match the checkpoint's training set size |
| `K_DIAG` | `800` | Sensor count to inspect |
| `TIME_IDX` | `500` | Snapshot index analysed |

---

### `plot_timing.py`

Standalone timing figure for the Hurricane dataset, reading from
`rd_results_Hurricane.csv` (produced by `hurricane_rd_comparison.py`).
Two panels: one-time training cost per method (bar chart: SZ2, ZFP,
DEIM-2D+SZ2, Kriging-2D+SZ2), and median per-compression time grouped by
method × sensor count `k`.

**Usage**

```bash
python3 plot_timing.py
```

Requires `rd_results_Hurricane.csv` to already exist. Saves
`timing_Hurricane.png`.

No top-of-file config block beyond fixed paths/colours; `CSV_PATH` and
`OUT_PATH` point at the Hurricane results by default.

---

### `plot_timing_amortized.py`

Companion to `plot_timing.py`: shows cumulative wall-clock time (training +
compression) vs. number of snapshots compressed, for the Hurricane dataset,
highlighting the break-even point where a hybrid method's amortized total
time drops below the SZ2/ZFP baseline (since GP/DEIM training is a one-time
cost not repeated per snapshot). Uses per-compression medians from
`rd_results_Hurricane.csv`, with hardcoded fallback training times
(DEIM≈10.9s, Kriging≈339.4s) if not present in the CSV.

**Usage**

```bash
python3 plot_timing_amortized.py
```

Requires `rd_results_Hurricane.csv`. Saves `timing_amortized_Hurricane.png`.

**Configuration** — edit only the block at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DEIM_K` | `80` | DEIM sensor count shown (lowest per-compression time) |
| `KRIG_K` | `400` | Kriging sensor count shown (largest — best RD, all k similar in speed) |
| `N_MAX` | `500` | Max snapshots shown on the x-axis |

---

### `plot_timing_sst.py`

SST-dataset counterpart of `plot_timing.py`: a single bar chart of median
per-compression time (± std), grouped by method, for eight
method/residual-coder combinations (SZ2, SZ2-1D, ZFP, ZFP-1D, DEIM-2D+SZ2,
DEIM-2D+ZFP, Kriging-2D+SZ2, Kriging-2D+ZFP), read from `rd_results_SST.csv`
(produced by `sst_rd_comparison.py`). Training times were not recorded for
the SST runs, so only per-compression time is shown.

**Usage**

```bash
python3 plot_timing_sst.py
```

Requires `rd_results_SST.csv`. Saves `timing_SST.png`.

No top-of-file config block beyond fixed paths/colours.

---

### `mask_hurricane.py`

Tiny inspection script: loads `TCf48.bin.f32` (path currently hardcoded to
a placeholder `/path/to/TCf48.bin.f32` — edit before running) reshaped to
`(100, 500, 500)`, and prints min/max/std/NaN-count for the first 15
vertical levels. Used to spot-check for degenerate or all-NaN near-surface
levels before running the main scripts. No output files.

**Usage**

```bash
python3 mask_hurricane.py
```

---

### `patch_libpressio.py`

One-off build-fix utility, unrelated to the Kriging/DEIM analysis: patches
the `libpressio` 0.94.0 SWIG-generated bindings for macOS compatibility
inside a Spack stage directory. Applies two source-level fixes — (1) in
`pypressio.h`, wraps `vector<uint64_t>` dimension arguments in
`std::vector<size_t>(...)` at each `pressio_data::` call site, and (2) in
the generated `pressioPYTHON_wrap.cxx`, replaces
`_from_numpy_1d/2d/3d/4d<long>` with `<int64_t>` (macOS treats `long` and
`int64_t` as distinct types even though they're the same width). Locates the
Spack stage via a hardcoded glob pattern, prints which replacements were
applied, and prints the follow-up `make`/`spack install` commands to run
manually.

**Usage**

```bash
python3 patch_libpressio.py
```

Run once, after `spack install libpressio@0.94.0 +python` has staged the
source but before the build completes; edit `STAGE_GLOB` at the top if the
Spack staging path differs.

---

### `ex2_comparing_compressors.py`

Working (editable) copy of the `libpressio` tutorial's "Exercise 2" script
(the original tutorial copy is read-only). Sweeps SZ/SZ3/MGARD/ZFP
(auto-detecting which are actually available in the local `libpressio`
build) over a small grid of absolute error bounds on the CLOUD dataset,
using MPI (`mpi4py.futures.MPICommExecutor`) to parallelize the sweep across
ranks, and writes compression ratio / PSNR results to a CSV for later
plotting. Differs from the original tutorial script in that: MGARD is
skipped by default, `PressioException`s are caught and returned as data so
one failed config doesn't abort the whole MPI run, and output paths point
into the tutorial's own `figures/` directory.

**Usage**

```bash
mpirun -n 4 python3 ex2_comparing_compressors.py
# then plot:
python3 ~/libpressio_tutorial/exercises/2_comparing_compressors/rate_distortion.py
```

Writes `results.csv` to the tutorial's `figures/` directory.

---

### `q2_zfp_vs_sz.py`

Answers Exercise 1 Question 2 of the `libpressio` tutorial: compares ZFP
(accuracy mode) against SZ (absolute-error mode) on the CLOUD dataset at
matched error bounds, sweeping `bounds = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]` and
reporting compression ratio, max absolute error, RMSE, and PSNR for each,
followed by a side-by-side rate-distortion comparison table.

**Usage**

```bash
python3 q2_zfp_vs_sz.py
```

Prints results to the console; produces no output files. Reads
`100x500x500/CLOUDf48.bin.f32` (path hardcoded near the top of the file).

---

### `replicate_fig1.py`

Replicates the spirit of Figure 1 from Liang et al. (2018), *"Error-Controlled
Lossy Compression Optimized for High Compression Ratios of Scientific
Datasets,"* using the ISABEL CLOUD field in place of the paper's NYX
velocity_x dataset. Compresses a chosen slice with SZ and ZFP at a target
absolute error bound and displays original/SZ/ZFP side by side, reporting
compression ratio, PSNR, and SSIM (via `scikit-image` if available, else a
manual single-window SSIM fallback) for each.

**Usage**

```bash
python3 replicate_fig1.py [--bound 5e-4] [--level 50]
```

Requires `scikit-image` (`pip install scikit-image --break-system-packages`).
Saves `fig1_comparison_bound{BOUND}.png` and `fig1_errormaps_bound{BOUND}.png`.

---

### `replicate_fig8.py`

Replicates Figure 8 from Liang et al. (2018), *"Data Distortion of
Hurricane (CLOUDf: slice 50) with CR=66:1,"* on the ISABEL CLOUD field. Finds
the compressor settings that hit a target compression ratio (ZFP: exact
fixed-rate computation `rate = 32/target_cr`; SZ: binary search over the
absolute-error bound) and displays original/SZ/ZFP side by side with CR,
PSNR, and SSIM, including a comparison against the paper's own reference
values (SZ: PSNR=29.9/SSIM=0.6573, ZFP: PSNR=22.5/SSIM=0.8893).

**Usage**

```bash
python3 replicate_fig8.py [--target-cr 66] [--level 50]
```

Requires `scikit-image` (`pip install scikit-image --break-system-packages`).
Saves `fig8_cloud_slice{LEVEL}_cr{TARGET_CR}.png` and
`fig8_errormaps_slice{LEVEL}_cr{TARGET_CR}.png`.

---

## Datasets

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

Other variables used by some scripts (same directory, same format):
`Vf48.bin.f32`, `Wf48.bin.f32` (wind components), `CLOUDf48.bin.f32`,
`QVAPORf48.bin.f32`, `TCf48.bin.f32` (temperature).

**NOAA OI SST V2 (weekly means)**

> R. W. Reynolds et al., *NOAA_OI_SST_V2*, NOAA/OAR/ESRL PSL, 1990–present.
> Available at: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html

Used by `sst_rd_comparison.py`, `sst_progressive_gp.py`,
`lorenzo_vs_global_diagnostic.py`, and `residual_autocorr_diagnostic.py`.
1727 weekly snapshots, 180×360 (lat×lon) grid. Scripts expect
`sst.wkmean.1990-present.nc` and a companion `lsmask.nc` land/sea mask, both
read from a sibling `gpoed-code-python/` directory (i.e. one level up from
this repo, in a folder of that name) — update `DATA_PATH`/`MASK_PATH` near
the top of each script if your layout differs. Only ocean pixels are used;
land is masked out before compression and excluded from PSNR/SSIM/RMSE.

---

## Requirements

See [Installation](#installation) above and `requirements.txt` in this
repository for the full, accurate dependency list (including the
non-pip-installable ones: `libpressio`, a built `sz3` binary, `mpi4py` +
MPI, and the sibling `gpoed-code-python` package).

---

## Notes

- `Kriging-DEIM.ipynb` — original exploratory notebook (shared, read-only).
  Contains 1D/2D/3D Kriging experiments and the initial CSSP implementation.
- `Kriging-DEIM-fixed.ipynb` — copy of the above with six bugs corrected.
- `Kriging-DEIM-notes.md` — cell-by-cell reference document for the notebook.

### Generated artifacts

Running these scripts produces checkpoints (`.pkl`, some 100MB–1GB+ for the
progressive-GP scripts), result CSVs, and figures (`.png`/`.gif`/`.mp4`)
directly in this directory. None of these are committed — see `.gitignore`.
Several scripts (`sz3_gp_comparison.py`, `hurricane_gp_comparison.py`,
`lorenzo_vs_global_diagnostic.py`, `residual_autocorr_diagnostic.py`, the
`plot_timing*.py` family, `run_fig3.py`) *read* a checkpoint or CSV produced
by another script rather than generating their own from scratch — check
each entry above for which upstream script needs to run first.

### Hardcoded local paths

A few scripts have machine-specific paths hardcoded near the top that will
need updating for a new environment: `SZ3_BIN` (path to a built SZ3 binary,
in `sz3_gp_comparison.py` and `hurricane_gp_comparison.py`), `STAGE_GLOB`
(Spack stage directory, in `patch_libpressio.py`), and the
`gpoed-code-python` sibling-directory assumption used for SST data and for
`rpgks`/`pivoted_cholesky` imports (`kriging_hurricane.py`,
`sst_rd_comparison.py`, `sst_progressive_gp.py`,
`lorenzo_vs_global_diagnostic.py`, `residual_autocorr_diagnostic.py`).
