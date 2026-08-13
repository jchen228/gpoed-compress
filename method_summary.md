# SST Compression Method Summary

All methods compress the NOAA OI SST V2 dataset (1727 × 180 × 360, float32).
Metrics are averaged over **all 1727 snapshots**, ocean pixels only (44,219 px).
CR = (n_T × ny × nx × 4 bytes) / compressed_bytes.

---

## Current parameter values

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_TRAIN` | 1000 | Training snapshots (evenly spaced in time) |
| `NUM_BINS` | 65,536 | Quantisation bins (uint16; mirrors SZ2) |
| `DEIM_K_VALS` | [100, 200, …, 1000] (step 100) | SVD mode counts tested (≤ N_TRAIN) |
| `KRIG_K_VALS` | [100, 200, …, 1000] (step 100) | GP sensor counts tested |
| `ABS_BOUNDS` | logspace(−4, −0.5, 8) ≈ [1e-4 … 0.32] °C | Absolute bound sweep (DEIM/Kriging L2 quantizer) |
| `ABS_BOUNDS_PRED` | ABS_BOUNDS ∪ [0.5, 1.0, 2.0, 5.0] °C | Extended sweep for DEIM/Kriging variants |
| `REL_BOUNDS` | 2^-10 … 2^-20 (11 pts, ~1e-3 … 1e-6) | SZ2 relative error bound sweep |
| `ZFP_BOUNDS` | 2^-1 … 2^-13 (13 pts, 0.5 … ~1.2e-4 °C) | ZFP fixed-accuracy sweep (power-of-2 aligned) |
| `TIME_IDX` | 864 | Snapshot shown in field panels (~July 2006) |
| `VIZ_K_DEIM` | 100 | k shown in field panel for DEIM methods |
| `VIZ_K_KRIG` | 100 | k shown in field panel for Kriging methods |
| `VIZ_AB` | 0.01 °C | abs_bound shown in field panel (closest match) |

**Active methods (current run):** SZ2, ZFP, DEIM-2D-L2, DEIM-2D-SO, Kriging-2D-L2, Kriging-2D-SO

**Temporarily disabled:** DEIM-2D (L∞), Kriging-2D (L∞), DEIM-2D-TT, DEIM-2D+SZ2, DEIM-2D+ZFP, Kriging-2D+SZ2, Kriging-2D+ZFP
(Re-enable via `RUN_HYBRID = True` / `RUN_DEIM_TT = True` flags and uncomment METHOD_ORDER entries)

---

## Baseline methods

| Method | Storage | Error bound type | Notes |
|--------|---------|-----------------|-------|
| **SZ2** | slice-by-slice compressed field | Relative per pixel (`rel_err_bound = r`, fraction of snapshot range) | Lorenzo predictor + Huffman + zstd; relative mode makes bound dataset-agnostic |
| **ZFP** | slice-by-slice compressed field | Fixed-accuracy (`accuracy = 2^-k`, powers of 2) | Orthogonal transform + fixed-point coding; power-of-2 bounds align with internal exponent grid → monotone CR curve |

---

## DEIM-2D family

SVD on ocean pixels of N_TRAIN snapshots → Phi_k (spatial modes).
Q-DEIM selects k sensor locations; sensor values used to reconstruct via M = Phi_k @ A⁻¹.

| Method | What is stored | Error bound type | CR driver | Notes |
|--------|---------------|-----------------|-----------|-------|
| **DEIM-2D** | model (Phi_k f16 + mean + sensor idx) + k×n_T sensor values (f16) + quantised residual | L∞ (`abs_bound = ε`) | Residual bloat when DEIM errors >> ε | Current standard variant |
| **DEIM-2D-L2** | same as DEIM-2D | L2/Frobenius (`bin_width = 2ε_rms`, `abs_bound = ε_rms × NUM_BINS`) | Residuals always fit in bins (no outlier bytes) | Better CR than DEIM-2D at same ε; error guarantee is RMSE-based |
| **DEIM-2D-TT** | model (Phi_k f16 + mean, **no sensor idx**) + quantised optimal coefficients C (n_T × k) | L2/Frobenius on C (`bin_width q = 2ε_rms√(n_ocean/k)`) | Quantisation of k coefficients per snapshot | No sensors at decode; C = data_ocean @ Phi_k (full-data projection); best k-rank approx |
| **DEIM-2D-SO** | model (Phi_k f16 + mean + sensor idx) + k×n_T sensor values (f16), **no residual** | None (reconstruction error only) | k float16 per snapshot | Single operating point per k; CR improves with larger k; PSNR limited by DEIM approximation quality |
| **DEIM-2D+SZ2** | model + sensor values (f16) + SZ2-compressed residual field | L∞ per pixel on residual | SZ2 on the already-small DEIM residual field | Applied slice-by-slice; CR ≤ standalone SZ2 but PSNR can be higher |
| **DEIM-2D+ZFP** | model + sensor values (f16) + ZFP-compressed residual field | L∞ per pixel on residual (ZFP accuracy) | ZFP on the already-small DEIM residual field | Same pattern as +SZ2 |

---

## Kriging-2D family

Matérn-3/2 GP with RPCholesky sensor selection. Hyperparameters (ls, var, noise) fitted by MLE on ocean pixels of N_TRAIN snapshots. Sensor values used for GP posterior prediction.

| Method | What is stored | Error bound type | CR driver | Notes |
|--------|---------------|-----------------|-----------|-------|
| **Kriging-2D** | model (K_Xs hyperparams + sensor idx + mean/std) + k×n_T sensor values (f16) + quantised residual | L∞ (`abs_bound = ε`) | Residual bloat when GP errors >> ε | Adaptive to spatial correlation; slower to fit than DEIM |
| **Kriging-2D-L2** | same as Kriging-2D | L2/Frobenius (`bin_width = 2ε_rms`, `abs_bound = ε_rms × NUM_BINS`) | Residuals always fit in bins (no outlier bytes) | Same L2 idea as DEIM-2D-L2 applied to GP residuals |
| **Kriging-2D-SO** | model + k×n_T sensor values (f16), **no residual** | None (reconstruction error only) | k float16 per snapshot | Single operating point per k; same trade-off as DEIM-2D-SO |
| **Kriging-2D+SZ2** | model + sensor values (f16) + SZ2-compressed residual field | L∞ per pixel on residual | SZ2 on GP residual field | Applied slice-by-slice |
| **Kriging-2D+ZFP** | model + sensor values (f16) + ZFP-compressed residual field | L∞ per pixel on residual (ZFP accuracy) | ZFP on GP residual field | Applied slice-by-slice |

---

## Quantizer details

| Quantizer | `abs_bound` | bin width | Outlier handling | Characteristic |
|-----------|-------------|-----------|-----------------|---------------|
| **L∞** (standard) | `ε` | `2ε / NUM_BINS` | Values outside `[-ε, ε]` stored as raw float32 — explodes CR | Guarantees max pointwise error ≤ ε |
| **L2/Frobenius** | `ε_rms × NUM_BINS` | `2 × ε_rms` | Range = `[-ε_rms×NUM_BINS, +ε_rms×NUM_BINS]` — residuals always in-range | Guarantees RMSE ≤ ε_rms; no outlier overhead |

NUM_BINS = 65,536 (uint16 indices).

---

## Model storage breakdown (approximate)

| Component | Size formula | Typical size (k=100) |
|-----------|-------------|----------------------|
| Phi_k (float16) | `n_ocean × k × 2 B` (+ zstd) | ~4–6 MB |
| Mean field (float16) | `n_ocean × 2 B` | ~88 kB |
| Sensor indices (int32) | `k × 4 B` | 0.4 kB |
| Sensor values (float16) | `n_T × k × 2 B` | ~0.3 MB |
| Residual (L∞, tight bound) | heavily outlier-laden | ~4–8 MB (or worse) |
| Residual (L2) | zstd of uint16 bins | ~1–3 MB |
| Coefficients C (DEIM-TT, L2) | `n_T × k × 2 B` + zstd | ~0.3–1 MB |

---

## Key observations

- **Why SZ2/ZFP win at tight bounds**: Lorenzo predictor sees every pixel at encode time; adjacent SST pixels differ by ±0.01–0.05 °C → tiny residuals → very high CR.
- **Why DEIM/Kriging struggle at tight L∞ bounds**: DEIM reconstructs from k sparse sensors → fine-scale errors ±0.5–2 °C remain → all residuals become float32 outliers under tight ε.
- **L2 quantization fixes the outlier problem**: Wide bins (2ε_rms wide) guarantee no outliers; the residual array is always fully quantised → predictable, good CR.
- **-SO trades PSNR for CR**: Storing only k float16 sensor values achieves 30–50× CR but PSNR is limited by how well DEIM/GP can reconstruct the full field from sparse sensors.
- **DEIM-TT**: Optimal projection (full data → k coefficients) gives better reconstruction than sensor-based DEIM-SO at same k; L2 quantization on coefficients; no sensors stored.
- **TTHRESH analogy**: Tucker decomposition on full 3D tensor (temporal + spatial modes) with global Frobenius bound and zstd encoding. DEIM-TT is the 2D spatial analog of TTHRESH's spatial factor; DEIM-2D-L2 extends this idea to the residual after sensor-based reconstruction.
