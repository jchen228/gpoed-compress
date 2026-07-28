# Presentation Notes — "Compress to Impress"

---

## PSNR: Simple Explanation

**Verbal version (for the talk):**
PSNR — Peak Signal-to-Noise Ratio — measures how accurately we reconstruct the
original data, in decibels. The formula is 20 times the log of the data range
divided by the RMS error. What that means in practice: every 6 dB is roughly
halving the reconstruction error, and differences of 5–10 dB are meaningful.
Higher is always better.

**On-slide equation (one line):**
PSNR (dB) = 20 log₁₀(data range / RMSE)

**One-sentence version if pressed:**
"It's a log-scaled measure of accuracy — higher means the reconstruction
is closer to the original."

---

## Corrected Framing: Our Methods vs. SZ2/ZFP on CR

Do NOT say: "our methods fall behind in compression ratio."

**Say instead:**
Our methods currently operate at lower compression ratios because we compress
each timestep independently and pay a fixed model overhead (basis storage for
DEIM, hyperparameters for GP). However, at comparable compression ratios,
we outperform SZ2/ZFP in reconstruction accuracy — that is, for the same
number of bytes stored, our reconstructions are more faithful to the original
signal. The low-CR regime is exactly where sparse reconstruction methods
are designed to excel.

The path to higher CR is a hybrid approach: pass the GP/DEIM residuals
through SZ2/ZFP as a second compression stage. This is identified as future work.

---

## SZ2 Weaknesses (for verbal explanation on Slide 2)

1. **Local, generic predictor.**
   SZ2 uses the Lorenzo predictor — polynomial extrapolation from 2–3 already-
   compressed immediate neighbors. It has no concept of global spatial structure.
   It cannot learn that SST follows ocean currents, or that a 3D pressure field
   has coherent vertical structure.

2. **No training phase.**
   SZ2 is purely online — it makes predictions from neighbors with no offline
   learning. Our methods train on historical snapshots and learn field-wide
   dominant patterns before compression begins.

3. **Field-by-field and timestep-by-timestep.**
   SZ2 sees one 2D or 3D array at a time and treats each independently.
   It cannot exploit correlations across variables (e.g., cloud water and
   water vapor co-vary) or across time (e.g., SST at a location follows
   a seasonal cycle).

---

## Future Work (Slide 5 bullets)

1. **Temporal evaluation** — Track reconstruction quality across the full
   time record (seasons, years) rather than a single snapshot. Identify
   regimes where prediction-based methods excel or struggle.

2. **Hybrid residual compression** — Pass GP/DEIM residuals through
   SZ2/ZFP as a second stage. The residuals have much smaller dynamic
   range and tighter distribution than the raw data, so SZ2/ZFP can
   compress them much more aggressively. Goal: increase CR while
   sacrificing minimal PSNR.

---

## Miscellaneous Notes

- Use "we" throughout — mentors are co-authors of this work.
- Tone: moderately casual, but seriously convey the technical content.
- Audience: CS / ML / math — "compression ratio" and "PSNR" can be
  used with brief in-sentence definitions rather than full explanations.
- Dataset for slides: SST (pending code fix) — globally recognizable,
  clean story, no missing values issue after masking.
