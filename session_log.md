# Session Log — Rate-Distortion Comparison
_Last updated: 2026-07-16_

## Where we are

All six methods are implemented in `rate_distortion_comparison.py` and run on the same downsampled data (ds=10, 50×50/level, CLOUDf48.bin.f32 + QVAPORf48.bin.f32):

| Method | Type | Notes |
|---|---|---|
| SZ2 | Baseline (libpressio) | Lorenzo predictor → quantize → Huffman → Zstd |
| ZFP | Baseline (libpressio) | Fixed-rate wavelet-based |
| T-DEIM | 3D POD basis (n_3D_ds × k) | Large model; fair via SZ-style residual quantization |
| DEIM-2D | Per-level 2D SVD modes (n_2D × k) | 100× smaller model than T-DEIM |
| Kriging-2D | Per-level GP with Matérn-3/2 | GKS sensor placement |
| MultiGP (d=2) | Joint GP on CLOUDf + QVAPORf | CR accounts for both fields; metrics on CLOUDf only |

Output plots: `rd_psnr_cr.png`, `rd_ssim_cr.png`, `rd_psnr_bpv.png`, `rd_ssim_bpv.png`, `rd_timing.png`, `rd_budget_breakdown.png`

## Known issues / comparison problems

- **SZ2 CR is significantly higher than our methods.** Root cause: SZ2's Lorenzo predictor exploits local spatial smoothness at every point using its neighbors, producing tiny prediction errors that Zstd crushes. Our SVD/GP predictors use k global sensors and leave larger per-level residuals. The flat quantize+Zstd on those residuals can't match Lorenzo's local coherence exploitation.
- **libpressio contiguity bug (fixed):** `data[:, ::DS, ::DS]` was a non-contiguous strided view. Changed to `np.ascontiguousarray(...)` — SZ2/ZFP should now appear in all plots.
- **MultiGP metrics (fixed):** Now reports CLOUDf (field 0) PSNR/SSIM only, so it's comparable to the other methods. QVAPORf metrics still stored in CSV as `psnr_qvapor`, `ssim_qvapor`.

## Session 2026-07-15 — Changes and clarifications

### Pipeline changes

- **Residual coder now mirrors SZ2:** `pack_encode` replaced with an entropy-based Huffman size estimate. Bin indices are treated as symbols; compressed size = Shannon entropy × n_symbols (in bytes) + small codebook overhead. Outlier positions Zstd-compressed; outlier values stored as raw float32, matching SZ2's unpredictable-point storage. This is fairer and more accurate than the previous Zstd-of-int16 approach.
- **`NUM_BINS` sweep removed from DEIM-2D:** Fixed at 65536 (matching SZ2's `bin_width = 2×abs_bound/65536`). The `NUM_BINS_VALUES` config constant remains for T-DEIM but is not swept in DEIM-2D or Kriging-2D.
- **Reconstruction hoisted out of abs_bound loop in DEIM-2D:** The DEIM prediction doesn't depend on `abs_bound`, so `recon_ds` and `resid` are now computed once per k, and only quantization is repeated per abs_bound. Speeds up the sweep significantly.
- **Sensor values stored exact (no quantization) in both DEIM-2D and Kriging-2D:** DEIM interpolates exactly at sensor locations by construction (residual at sensors = 0), so quantizing sensors only injects noise into the solve. Kriging similarly.
- **`PLOTS_ONLY` flag added:** Set `PLOTS_ONLY = True` to skip compression and regenerate all plots from the existing `rd_results_{FIELD_TAG}.csv`. Useful after a crash or code fix.
- **`RUN_TDEIM` / `RUN_MULTIGP` flags** now properly gate those blocks in `main()`.
- **`cr_no_model`** added to all DEIM-2D and Kriging-2D result rows and the CSV. Plots now show a second dotted line per method representing CR/bpv if model storage were free.
- **Method-name labels** added to each baseline (SZ2, SZ3, ZFP) curve endpoint in all RD and bpv plots.
- **Output filenames tagged with field name** (`FIELD_TAG` derived from `DATA_PATH.stem`) so CLOUDf and QRAIN results don't overwrite each other.

### Kriging-2D hyperparameter fitting fix

- **Previous:** Used `mgp.fit_lengthscale` (fits only `ls`, with `B` estimated separately via `estimate_B`). This is a sequential fit, not a joint one.
- **Now:** Jointly optimizes `ls` and `var` (kernel output variance) in a single 2D MLE, matching `lp_kriging_compressor.py`. Uses `n_restarts=3`.
- **Coarse-grid MLE:** The joint MLE runs on a subsampled grid (`HP_DS=5` → 10×10 = 100 points) rather than the full 2500-point grid. Since coordinates are normalized to [0,1]², the lengthscale is resolution-invariant and transfers directly. Makes hyperparameter fitting ~15,000× cheaper with negligible accuracy loss.

### k sweep expansion

- `DEIM2D_K_VALUES`: `[1,2,5,10,25,50]` → `[1,2,5,10,25,50,100]` (100 = practical ceiling, capped by n_L=100 SVD modes)
- `KRIG2D_K_VALUES`: `[1,5,10,25,50,100]` → `[1,5,10,25,50,100,200,500]` (GP has no n_L cap; more sensors always helps prediction)

### QRAIN experiment (abandoned)

Tested on `QRAINf48.bin.f32`. SSIM collapsed for DEIM-2D and Kriging-2D because QRAIN is a sparse field (zeros everywhere outside the storm core). Our global SVD/GP methods struggle with sharp zero/nonzero boundaries; SZ2's Lorenzo predictor handles the zero-heavy region trivially. The log10 version (`QRAINf48.log10.bin.f32`) would be the correct preprocessing for rain fields but was not tested. **Reverted to CLOUDf48 as the primary field.**

### Residual histogram bugs (fixed)

- DEIM-2D: dead `np.linalg.qr(..., pivoting=True)` call crashed before reaching the scipy fallback. Removed.
- Kriging-2D: `norm(all_sv[lvl])` passed shape-(k,) to a function expecting shape-(n_2D,). Fixed to use `train_mean[s_k, 0]` for sensor-location normalization.

## Observation: SSIM is very high on CLOUDf

Our methods (DEIM-2D, Kriging-2D) achieve impressively high SSIM relative to their CR on CLOUDf — the global structure is preserved well. The CR gap vs SZ2 remains the main open problem. The `cr_no_model` line now confirms the gap is primarily in the residuals, not the model.

## Session 2026-07-16 — Changes and clarifications

### ZFP mode switched to fixed-accuracy

Changed ZFP from fixed-rate (`zfp:rate`) to fixed-accuracy (`zfp:accuracy`) mode to match the evaluation methodology in *Error-Controlled Lossy Compression Optimized for High Compression Ratios of Scientific Datasets* (Fig. 6). The paper explicitly states absolute error bound mode was used for both SZ and ZFP. `ZFP_RATES` renamed to `ZFP_ABS_BOUNDS = np.logspace(-4, -0.5, 8)` — same sweep as SZ2. ZFP is now included in the abs_bound plot (previously excluded as it had no abs_bound equivalent).

### Prediction-only curves removed from plots

`PRED_ONLY = True` still computes and stores `psnr_pred`, `ssim_pred`, `cr_pred_only` in the CSV, but these curves are no longer drawn in `make_rd_plot` or `make_bpv_plot`.

**Rationale:** For T-DEIM and DEIM-2D, the pred-only computation was incorrectly inside the abs_bound loop, reusing the abs_bound-quantized sensor encoding. This caused two artifacts: (1) cr_pred_only was tied to abs_bound (coarser quantization → smaller sv_enc → higher CR), which is conceptually wrong since pred-only should use exact sensor values; (2) psnr_pred used sensors quantized at the current abs_bound, so loose abs_bound → degraded sensors → poor PSNR. For Kriging-2D and MultiGP, pred-only was computed correctly (float16 exact sensors, outside the abs_bound loop). The inconsistency makes the pred-only curves misleading. The values remain in the CSV for future analysis once the T-DEIM/DEIM-2D pred-only is refactored to use exact sensor encoding outside the abs_bound loop.

### Image resolution increased

All `savefig` calls changed from `dpi=150` to `dpi=300` for sharper rendering in the Beamer slide deck.

### Plot labeling consistency

- `make_timing_plot`: our methods now label only the first (lowest k) point; k range shown in legend.
- `make_rd_plot` / `make_bpv_plot`: no-model dotted curves now show k range in legend and label the first point.

## Open questions / next directions

1. **Two-stage residual compression:** Apply SZ2's Lorenzo predictor to the DEIM/GP residual field (which is itself smooth). This could dramatically improve CR — essentially run libpressio SZ on the residual 3D array at the same abs_bound.
2. **Compress sensor time series:** Each of the k sensor values across 100 levels is a smooth 1D time series — 1D SZ on each sensor trace would compress the sensor storage significantly.
3. **MultiGP vs 2×SZ2:** The fairest comparison is SZ2 run independently on CLOUDf + QVAPORf, sizes summed, vs MultiGP joint compression.
4. **Try log10(QRAIN)** if sparse-field comparison is needed.

## File locations

- Main script: `Argonne/rate_distortion_comparison.py`
- Data: `Argonne/100x500x500/CLOUDf48.bin.f32`, `QVAPORf48.bin.f32`
- Helper compressors (READ-ONLY): `Argonne/lp_tdeim_compressor.py`, `lp_multigp_compressor.py`
- libpressio tutorial (READ-ONLY): `/Users/jchen228/libpressio_tutorial/`
