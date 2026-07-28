# Progress Log — libpressio Custom Compressor Project

## Terminology

- **Field**: one physical quantity over the full domain (e.g. U-wind, 100×500×500).
  Corresponds to one `.bin.f32` file in the ISABEL dataset.
- **Slice**: one 2D horizontal level of a field (e.g. U-wind at level 50, 500×500).
- **Variable** / **snapshot**: not used — use "field" and "slice" instead.

---

## Goal

Translate four custom prediction/reconstruction methods (DEIM, T-DEIM, Kriging,
Multi-output GP) into compressors that the libpressio package can use — so they
can be benchmarked on a level playing field against established lossy compressors
like SZ and ZFP.

---

## Checkpoint: 2026-07-14  09:30 CDT

### What was accomplished

#### Session 1 (previous day) — Getting libpressio working

libpressio was already installed via Spack on the M1 Mac (running x86_64 Anaconda
under Rosetta 2), but two bugs prevented it from working:

**Bug 1 — Wrong shared library path (rpath)**
`_pressio.so` was linked against the wrong Spack prefix (`oekradi`), which was
built without SZ or ZFP. The correct prefix (`nmbr4am`) had both. Fixed with:

    install_name_tool -rpath <old-oekradi-path> <new-nmbr4am-path> \
        /Users/jchen228/Anaconda3/lib/python3.10/site-packages/_pressio.so

**Bug 2 — SWIG / clang type mismatch (uint64_t)**
After fixing the rpath, calling `data_new_empty()` from Python failed with:
    TypeError: argument 2 of type 'std::vector<size_t>'
Root cause: SWIG on macOS x86_64 resolves `uint64_t` → `unsigned long`, but
clang resolves `uint64_t` → `unsigned long long`. They disagreed on the vector
type to pass across the Python/C++ boundary.
Fix: edited `~/libpressio_src/tools/swig/pypressio.h` — changed all 5
`data_new_*` function signatures from `std::vector<uint64_t>` to
`std::vector<unsigned long>`, then rebuilt `_pressio.so` from scratch:

    clang++ -arch x86_64 -std=c++17 -bundle -Wl,-undefined,dynamic_lookup \
      ... pressioPYTHON_wrap.cxx ${LP_INSTALL}/lib/liblibpressio.dylib ...

A helper script `fix_pypressio.py` was written to automate the pypressio.h patch.
After both fixes, `basics.py` (the libpressio tutorial) ran successfully and
produced realistic SZ compression metrics at multiple error bounds.

**Tutorial exercise 1 completed (partial)**
- Q1: compared SZ `vr_rel` vs `abs` modes.
  Key finding: with `abs` bound larger than the data range (~0.00205 for CLOUD),
  SZ outputs only the field mean → 2.27M× "compression" ratio, ~32 dB PSNR.
  `vr_rel` is self-scaling (bound × value range), so it degrades gracefully.
- Q2 (ZFP vs SZ): ZFP options listed, comparison script not yet run.
- Q3 (data shape): not yet attempted.

#### Session 2 (today) — Custom compressors

Four libpressio-compatible external compressor scripts were written, one per method.
All live in `/Users/jchen228/Desktop/Argonne/`:

| File | Method | Source script |
|---|---|---|
| `lp_deim_compressor.py` | Q-DEIM (2D, per-level) | `deim_hurricane.py` |
| `lp_tdeim_compressor.py` | T-DEIM (3D, full volume) | `tdeim_hurricane.py` |
| `lp_kriging_compressor.py` | GP / Simple Kriging | `kriging_hurricane.py` |
| `lp_multigp_compressor.py` | Multi-output GP (LMC/ICM) | `multigp_hurricane.py` |

**How the translation was done**
The original hurricane scripts are monolithic research scripts (load → train → test
→ plot, hardcoded paths). Each was restructured as follows:

1. Core algorithm functions extracted to the top of the file (SVD, sensor placement,
   reconstruct/predict) — logic unchanged from the hurricane scripts.
2. Three callable phases: `train()`, `compress()`, `decompress()`.
   - `train`: builds the model (basis, sensors, hyperparameters), saves to `.npz`.
   - `compress`: loads model, extracts k sensor values from input, writes binary.
   - `decompress`: loads model + compressed binary, reconstructs full field, writes binary.
3. `error_bound` as the single configurable parameter, mapping to sensor count k
   via the same POD energy criterion used in the hurricane scripts:
       sqrt(1 − Σs[:k]² / Σs²) ≤ error_bound
   This is directly analogous to `pressio:abs` in SZ.
4. Binary I/O protocol: raw float32 input/output with a small int32 header,
   so scripts can be called as subprocesses by libpressio's `external` plugin.
5. argparse CLI with `train / compress / decompress` subcommands.

**libpressio `external` compressor wiring (once models are trained):**

    comp = lib.get_compressor("external")
    comp.set_options({
        "external:command":
            "python3 /path/lp_deim_compressor.py compress --model deim_model.npz",
        "external:decompressor_command":
            "python3 /path/lp_deim_compressor.py decompress --model deim_model.npz",
    })

---

## Can these be compared with SZ?

**Yes — that is exactly what libpressio is designed for.**

libpressio provides a unified interface: every compressor (SZ, ZFP, or a custom
`external` script) exposes the same `compress` / `decompress` / `get_options` /
`set_options` API. A single benchmark loop can call all of them identically:

    for name, comp in compressors.items():
        comp.set_options({"pressio:abs": target_bound})
        compressed   = comp.encode(data)
        reconstructed = comp.decode(compressed)
        # compute RMSE, PSNR, compression ratio uniformly

The metrics reported by libpressio (compression ratio, max error, RMSE, PSNR)
are computed the same way for every compressor, so comparisons are fair.

**Meaningful metrics to compare:**
- Compression ratio at a given error bound
- RMSE / max absolute error vs. compression ratio (rate-distortion curve)
- Encode and decode wall time

**Caveats when comparing custom methods to SZ:**
- The custom methods have an offline training cost (building the basis/model).
  This is amortised over many slices and should be reported separately.
- The custom methods are data-specific (trained on ISABEL slices), which
  gives them an advantage over general-purpose compressors on the same dataset.
  A fair comparison requires testing on held-out data (already implemented in
  the train/test split inside each `train()` function).
- Compression ratio for the custom methods includes the model file size if the
  model is counted as part of the compressed payload.

---

## SZ versions supported by libpressio

libpressio wraps multiple SZ generations via separate compressor IDs:

| Compressor ID | SZ version | Notes |
|---|---|---|
| `"sz"` | SZ 1.4 / SZ2 | The original; `sz:error_bound_mode` selects abs / rel / pw_rel |
| `"sz3"` | SZ3 | Rewritten C++ version; better rate-distortion; `sz3:error_bound` |
| `"szx"` | SZx | Ultrafast variant; reduced accuracy; `szx:relative_error_bound` |
| `"mgard"` | MGARD | Multilevel / multigrid approach; different error guarantee |
| `"zfp"` | ZFP | Fixed-rate or accuracy mode; `zfp:accuracy` / `zfp:rate` |

SZ3 is generally the recommended baseline for new comparisons — it has better
compression at equivalent error bounds than SZ2 and is actively maintained.

To check which are available in your build:

    import pressio
    lib = pressio.instance()
    for name in ["sz", "sz3", "szx", "zfp", "mgard"]:
        try:
            c = lib.get_compressor(name)
            print(name, "available:", c.get_options())
        except Exception as e:
            print(name, "NOT available:", e)

---

## Methodological note: offline training vs SZ's per-field prediction (2026-07-14)

A key framing question is whether the offline training step makes our methods
"unfair" compared to SZ/ZFP. The answer is no — and the argument is:

SZ2/SZ3 also perform fitting during compression:
  - SZ2 (regression mode): fits local regression coefficients per block, per field.
  - SZ3 (interpolation): fits a hierarchical interpolation predictor to each field
    at compress time — O(n log n) per field.
Neither is a parameter-free lookup; both do adaptive work proportional to field size.

The difference is granularity of the fitting step:

  | Method              | When        | Scope                      |
  |---------------------|-------------|----------------------------|
  | SZ2 (Lorenzo)       | per field   | local k-neighborhood       |
  | SZ3 (interpolation) | per field   | full field, hierarchical   |
  | DEIM / Kriging / GP | once, offline | full training ensemble   |

When compressing N fields, training cost amortizes across all of them:
  - Amortized compress time  = (training time / N) + per-field online time
  - Effective compression ratio = orig bytes / (compressed bytes + model bytes / N)

For large N, the online step — extracting k sensor values, O(k) — is faster than
SZ3's O(n log n) per-field interpolation. This is a genuine speed advantage.

Important correction (2026-07-14):
  DEIM, Kriging, and T-DEIM are per-field — switching from U-wind to pressure
  requires retraining from scratch. Their amortization argument holds only within
  a fixed field, which is the same scope as SZ (per-field). They do not
  share any structure across variables.

  MultiGP is the exception: it trains jointly on d fields and compresses all d
  simultaneously with one shared set of k sensor locations. This gives two distinct
  advantages over the other methods:
    1. Sensor efficiency: d correlated fields use k shared positions vs k*d for
       d independent compressors. Cross-variable correlations (encoded in B) reduce
       the sensor count needed per field.
    2. Joint reconstruction: all d variable fields are reconstructed in one solve.

Preferred framing by method:

  DEIM / Kriging / T-DEIM:
    "Per-variable ensemble-adaptive prediction. Training cost is paid once per
    field and amortized over N slices of that field. Online
    compression is O(k) sensor extraction, faster than SZ3's O(n log n)
    per-field interpolation for large N."

  MultiGP:
    "Joint multi-field compression. Exploits cross-field physical coupling
    (e.g., pressure drives circulation, which drives convection) that per-field
    compressors — including SZ, ZFP, DEIM, and Kriging — cannot capture. d
    correlated fields are compressed with k shared sensor locations, achieving
    lower total sensor count than d independent single-field compressors."

Metrics to report in benchmark:
  1. Offline training time (reported once, separately)
  2. Per-field online compress time
  3. Per-field online decompress time
  4. Amortized compress time = (training / N) + online (for several N values)
  5. Effective compression ratio (with model size amortized)
  6. Rate-distortion curve: RMSE vs compression ratio

---

## Pending / next steps

- [ ] Q2: run ZFP accuracy vs SZ abs comparison — script written: q2_zfp_vs_sz.py
- [ ] Q3: test changing data shape in basics.py
- [ ] Run `train` step for each custom compressor on ISABEL data
- [ ] Write benchmark loop: DEIM / T-DEIM / Kriging / MultiGP vs SZ / SZ3 / ZFP
        - report all 6 metrics above
        - show amortized time vs N curve
- [ ] Test MultiGP on all 3 wind fields (U, V, W) jointly
