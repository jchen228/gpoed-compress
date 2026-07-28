# Parameter Toggles — rate_distortion_comparison.py

---

## Shared across all methods

| Parameter | Location | Current value | Effect of increasing | Effect of decreasing |
|---|---|---|---|---|
| `DS` | Config | `10` (50×50/level) | Finer grid, slower, more data to compress | Coarser grid, faster, fewer values per level |
| `ABS_BOUNDS` | Config | `logspace(-4, -0.5, 8)` | Wider range explores more of the RD curve | Narrower range |
| `NUM_BINS` | Config | `65536` (16-bit) | More quantization precision, larger compressed output | Fewer bins, more lossy quantization, potentially higher CR |
| Zstd level | Config (`_cctx`) | `3` | Slower compression, smaller output | Faster, larger output |

---

## SZ2

| Parameter | Location | Current value | Notes |
|---|---|---|---|
| `ABS_BOUNDS` (abs error bound) | Config | `logspace(-4, -0.5, 8)` | SZ2's primary quality knob. Directly controls max pointwise error. |
| Compressor config | `run_libpressio()` call | `{"pressio:abs": bound}` | Could also try `pressio:rel` (relative error) or `sz:error_bound_mode` |

SZ2 has no model-size toggles — it's entirely Lorenzo predictor + Huffman + Zstd, controlled only by the error bound.

---

## ZFP

| Parameter | Location | Current value | Notes |
|---|---|---|---|
| `ZFP_RATES` (bits per value) | Config | `[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]` | ZFP's primary quality knob. Lower = more compressed, more lossy. |
| Mode | `run_libpressio()` call | `zfp:rate` (fixed-rate) | Could also try `zfp:precision` (fixed-precision) or `zfp:accuracy` (fixed-accuracy, like abs bound) |

---

## T-DEIM, DEIM-2D  *(shared structure)*

| Parameter | Location | Current value | Effect |
|---|---|---|---|
| `TDEIM_K_VALUES` / `DEIM2D_K_VALUES` | Config | `[1, 2, 5, 10, 25, 50]` | Number of SVD modes and sensors. Higher k → better prediction, larger model, lower CR. |
| `ABS_BOUNDS` (sensor quantization) | Config | `logspace(-4, -0.5, 8)` | Controls error in both sensor values and residuals. |
| `NUM_BINS` | Config | `65536` | Shared quantization resolution for sensors + residuals. |
| SVD rank cap | Inside `run_tdeim` / `run_deim_2d` | `min(max(k_values), n_L)` | Implicitly limits k to 50 (= n_L for ds=10). Could increase k sweep beyond n_L if n_2D allows (DEIM-2D only). |
| Sensor placement | Inside each function | Q-DEIM (pivoted QR on Phi.T) | Could swap for random sensors, greedy max-vol, or QDEIM oversampling. |
| Model precision | `compress_f16(Phi_k)` | float16 | Could use float32 for higher fidelity at cost of larger model. |

**T-DEIM only:**

| Parameter | Location | Current value | Effect |
|---|---|---|---|
| Basis dimension | `Phi` shape | `(n_3D_ds, k)` = `(250000, k)` | This is the main reason T-DEIM's model is 100× larger than DEIM-2D. Not easily toggleable without redesign. |

**DEIM-2D only:**

| Parameter | Location | Current value | Effect |
|---|---|---|---|
| Basis dimension | `Phi_2D` shape | `(n_2D, k)` = `(2500, k)` | Fixed by ds and k. Much smaller than T-DEIM. |

---

## Kriging-2D

| Parameter | Location | Current value | Effect |
|---|---|---|---|
| `KRIG2D_K_VALUES` | Config | `[1, 5, 10, 25, 50, 100]` | Number of GP sensors. Higher k → better prediction, larger K_Xs model, slower inference. |
| `ABS_BOUNDS` | Config | `logspace(-4, -0.5, 8)` | Controls sensor + residual quantization error. |
| `noise_var` | `run_kriging_2d()` | `0.05 ** 2 = 0.0025` | GP observation noise. Higher → smoother/more regularized predictions (less variance captured, potentially larger residuals). |
| `n_restarts` (lengthscale MLE) | `run_kriging_2d()` | `2` | More restarts → better lengthscale fit, slower training. |
| Kernel | `run_kriging_2d()` | Matérn-3/2 | Fixed for now. Could try RBF (smoother) or Matérn-1/2 (rougher). |
| Sensor placement | `run_kriging_2d()` | GKS (greedy on K_spatial) | Could try random, pivoted Cholesky (RPCholesky), or Q-DEIM for consistency with DEIM-2D. |
| K_Xs model precision | `compress_f16(K_Xs)` | float16 | Could increase to float32. |
| Jitter (numerical stability) | `K_spatial + 1e-6 * I`, `K_ss + 1e-6 * I` | `1e-6` | Increasing stabilizes Cholesky but changes the effective kernel slightly. |

---

## MultiGP (d=2, CLOUDf + QVAPORf)

All Kriging-2D toggles apply here, plus:

| Parameter | Location | Current value | Effect |
|---|---|---|---|
| `MULTIGP_K_VALUES` | Config | `[1, 5, 10, 25, 50, 100]` | Sensors per field. dk × dk Kronecker system grows as (d×k)². |
| `d` (number of fields) | `run_multigp()` | `2` (CLOUDf + QVAPORf) | Adding more fields increases joint CR denominator and model size proportionally. |
| `noise_var` | `run_multigp()` | `0.05 ** 2` | Same role as in Kriging-2D but affects the (dk × dk) system. |
| `n_restarts` | `run_multigp()` | `2` | Lengthscale MLE restarts. |
| `B` matrix (cross-field covariance) | Estimated via `estimate_B()` | Fitted from data | Could fix B = I (independent fields) as an ablation to measure how much cross-field coupling helps. |
| Metrics reported | `psnr` / `ssim` in results | CLOUDf only (field 0) | QVAPORf stored as `psnr_qvapor`, `ssim_qvapor` in CSV. |

---

## High-priority toggles to try tomorrow

These are most likely to close the CR gap with SZ2 or reveal where our methods win:

1. **`NUM_BINS`**: Try `256`, `1024`, `4096` — coarser quantization means larger bin_width per abs_bound, fewer outliers, smaller compressed residuals. Direct CR lever.
2. **`ABS_BOUNDS` upper end**: Extend to `1e0` or `1e1` to explore the high-CR / low-quality regime.
3. **`KRIG2D_K_VALUES` / `DEIM2D_K_VALUES` lower end**: Add `k=200, 500` to see if very dense sensors help CR (at cost of model size).
4. **Test on QRAIN or QICE**: SZ2's Lorenzo predictor degrades on non-smooth intermittent fields. Our methods may look more competitive there.
5. **MultiGP vs 2×SZ2**: Add a combined SZ2 baseline that compresses CLOUDf + QVAPORf independently and sums the sizes. Compare against MultiGP's joint CR.
6. **`noise_var` in Kriging/MultiGP**: Try `0.1**2` or `0.2**2` — more regularization may smooth residuals enough to compress better.
