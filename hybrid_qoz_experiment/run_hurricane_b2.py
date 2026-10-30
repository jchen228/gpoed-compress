#!/usr/bin/env python3
"""
run_hurricane_b2.py
===================
Run GP-QoZ (Approach B, gp_levels=2, anchor_frac=0.10) on Hurricane ISABEL W
alongside baselines and the arithmetic-coding / raw-int variants.

Compressors tested:
  - QoZ 2.0 / HPEZ    (-q4)
  - SZ3-Interp         (ALGO_INTERP via config)
  - SZ3-Lorenzo+Interp (default SZ3, ALGO_INTERP_LORENZO)
  - GP-QoZ-Huffman     (stride=32, gp_levels=2, default Huffman encoder)
  - GP-QoZ-Arith       (stride=32, gp_levels=2, ARITHMETIC encoder)
  - GP-QoZ-RawInt      (stride=32, gp_levels=2, raw int32 + zstd — no entropy coding)
  - GP-SZ3-sim / GP-SZ3-real / GP-Lorenzo (simulation variants)

Usage (from any directory):
    python /Users/jchen228/Desktop/Argonne/hybrid_qoz_experiment/run_hurricane_b2.py

Outputs:
    results_b2/results_b2.json   — raw numbers
    results_b2/rd_b2.png         — Rate-PSNR (PSNR vs bit-rate)
    results_b2/cr_eb_b2.png      — CR vs relative EB
    results_b2/metrics_b2.txt    — per-EB metrics table
    results_b2/breakdown_b2.png  — header vs QoZ byte breakdown for GP-QoZ variants (Huffman/Arith/RawInt)
"""

import sys, json, math, tempfile, subprocess, os, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/jchen228/Desktop/Argonne/hybrid_qoz_experiment')
from hybrid_qoz_gp import HybridQoZGP

sys.path.insert(0, '/Users/jchen228/Desktop/Argonne')
import hurricane_hybrid_gp as _hgp_mod
from hurricane_hybrid_gp import run_hybrid_gp as _run_gp_sz3

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = '/Users/jchen228/Desktop/Argonne/100x500x500/Wf48.bin.f32'
QOZ_HYBRID   = '/Users/jchen228/Desktop/Argonne/QoZ-hybrid/build/test/qoz'
QOZ_ARITH    = '/Users/jchen228/Desktop/Argonne/QoZ-hybrid/build/test/qoz_arith'
QOZ_RAWINT   = '/Users/jchen228/Desktop/Argonne/QoZ-hybrid/build/test/qoz_rawint'
QOZ_PLAIN    = '/Users/jchen228/Desktop/Argonne/QoZ/build/test/qoz'
SZ3_BIN      = '/Users/jchen228/Desktop/Argonne/SZ3/build/tools/sz3/sz3'
OUT_DIR      = Path('/Users/jchen228/Desktop/Argonne/hybrid_qoz_experiment/results_b2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = (100, 500, 500)        # NZ, NY, NX   (data[NZ][NY][NX])
EBS  = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]   # relative error bounds

# ── load data ────────────────────────────────────────────────────────────────
print('Loading data…')
data = np.fromfile(DATA_PATH, dtype=np.float32).reshape(DIMS)
dr   = float(data.max() - data.min())
n    = data.size
print(f'  shape={data.shape}  dr={dr:.4f}  n={n:,}')

# ── helpers ──────────────────────────────────────────────────────────────────
def psnr(orig, dec):
    mse = float(np.mean((orig.astype(np.float64) - dec.astype(np.float64))**2))
    return 20 * math.log10(dr / math.sqrt(mse)) if mse > 0 else float('inf')

def max_err(orig, dec):
    return float(np.max(np.abs(orig.astype(np.float64) - dec.astype(np.float64))))

def record(alg, eb_rel, dec, n_bytes, extra=None):
    eb_abs = eb_rel * dr
    r = dict(
        algorithm=alg, eb_rel=eb_rel, eb_abs=eb_abs,
        bit_rate=n_bytes * 8 / n,
        cr=(n * 4) / n_bytes,
        psnr=psnr(data, dec),
        max_err=max_err(data, dec),
        cmp_bytes=n_bytes,
        data_range=dr,
    )
    if extra:
        r.update(extra)
    return r

results = {}

# ── 1. QoZ HPEZ (-q4) ────────────────────────────────────────────────────────
print('\n=== QoZ HPEZ (−q4) ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi = os.path.join(td, 'v.f32')
        fc = os.path.join(td, 'v.qoz')
        fd = os.path.join(td, 'v.dec')
        data.tofile(fi)
        r1 = subprocess.run([QOZ_PLAIN, '-f', '-i', fi, '-z', fc,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs), '-q', '4'],
                            capture_output=True)
        if r1.returncode:
            print(f'  FAIL compress eb={eb_rel}: {r1.stderr[-200:]}'); continue
        if not os.path.exists(fc):
            alt = fi + '.qoz'
            if os.path.exists(alt): os.rename(alt, fc)
            else: print(f'  FAIL: no output eb={eb_rel}'); continue
        r2 = subprocess.run([QOZ_PLAIN, '-f', '-z', fc, '-o', fd,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs), '-q', '4'],
                            capture_output=True)
        if r2.returncode:
            print(f'  FAIL decompress eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        r   = record('qoz_hpez', eb_rel, dec, sz)
        results[f'qoz_hpez|{eb_rel:.2e}'] = r
        print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
              f'maxerr={r["max_err"]:.2e}  (bound={eb_abs:.2e})')

# ── 2. SZ3-Interp (ALGO_INTERP) ──────────────────────────────────────────────
print('\n=== SZ3-Interp (ALGO_INTERP) ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi  = os.path.join(td, 'v.f32')
        fc  = os.path.join(td, 'v.sz3')
        fd  = os.path.join(td, 'v.dec')
        cfg = os.path.join(td, 'sz3.cfg')
        with open(cfg, 'w') as f:
            f.write('[GlobalSettings]\nCmprAlgo = ALGO_INTERP\n'
                    f'ErrorBoundMode = ABS\nAbsErrorBound = {eb_abs}\n')
        data.tofile(fi)
        r1 = subprocess.run([SZ3_BIN, '-f', '-i', fi, '-z', fc,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs), '-c', cfg],
                            capture_output=True)
        if r1.returncode:
            print(f'  FAIL compress eb={eb_rel}: {r1.stderr[-200:]}'); continue
        if not os.path.exists(fc):
            alt = fi + '.sz3'
            if os.path.exists(alt): os.rename(alt, fc)
            else: print(f'  FAIL: no output eb={eb_rel}'); continue
        r2 = subprocess.run([SZ3_BIN, '-f', '-z', fc, '-o', fd,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs), '-c', cfg],
                            capture_output=True)
        if r2.returncode:
            print(f'  FAIL decompress eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        r   = record('sz3_interp', eb_rel, dec, sz)
        results[f'sz3_interp|{eb_rel:.2e}'] = r
        print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
              f'maxerr={r["max_err"]:.2e}')

# ── 3. SZ3-Lorenzo+Interp (default, ALGO_INTERP_LORENZO) ─────────────────────
print('\n=== SZ3-Lorenzo+Interp (default) ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi = os.path.join(td, 'v.f32')
        fc = os.path.join(td, 'v.sz3')
        fd = os.path.join(td, 'v.dec')
        data.tofile(fi)
        r1 = subprocess.run([SZ3_BIN, '-f', '-i', fi, '-z', fc,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs)],
                            capture_output=True)
        if r1.returncode:
            print(f'  FAIL compress eb={eb_rel}: {r1.stderr[-200:]}'); continue
        if not os.path.exists(fc):
            alt = fi + '.sz3'
            if os.path.exists(alt): os.rename(alt, fc)
            else: print(f'  FAIL: no output eb={eb_rel}'); continue
        r2 = subprocess.run([SZ3_BIN, '-f', '-z', fc, '-o', fd,
                             '-3', '500', '500', '100',
                             '-M', 'ABS', str(eb_abs)],
                            capture_output=True)
        if r2.returncode:
            print(f'  FAIL decompress eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        r   = record('sz3_lorenzo', eb_rel, dec, sz)
        results[f'sz3_lorenzo|{eb_rel:.2e}'] = r
        print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
              f'maxerr={r["max_err"]:.2e}')

# ── 4. GP-QoZ-Huffman (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ─
# The C++ fix in SZInterpolationCompressor.hpp stores quant_inds.size()
# explicitly in the bitstream so the Huffman decoder doesn't overrun.
print('\n=== GP-QoZ-Huffman (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ===')
comp_huff = HybridQoZGP(qoz_bin=QOZ_HYBRID, anchor_frac=0.10, max_step=32, gp_levels=2)
for eb_rel in EBS:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        cmp_path = os.path.join(td, 'out.hqoz')
        info = comp_huff.compress(data, eb_rel, cmp_path)
        if 'error' in info:
            print(f'  FAIL compress eb={eb_rel}: {info["error"]}'); continue
        dec, t_dec = comp_huff.decompress(cmp_path, DIMS, eb_rel)
        if dec is None:
            print(f'  FAIL decompress eb={eb_rel}'); continue
        sz = info.get('cmp_bytes', os.path.getsize(cmp_path))
        r  = record('gp_qoz_b2', eb_rel, dec, sz, extra=dict(
            n_anchors=info.get('n_anchors', 0),
            n_coarse=info.get('n_coarse', 0),
            header_bytes=info.get('header_bytes', 0),
            qoz_bytes=info.get('qoz_bytes', 0),
            comp_time=info.get('comp_time', 0),
            decomp_time=t_dec,
        ))
        results[f'gp_qoz_b2|{eb_rel:.2e}'] = r
        print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
              f'maxerr={r["max_err"]:.2e}  (bound={eb_rel*dr:.2e})  stride=32  '
              f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
              f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB  '
              f't={time.time()-t0:.1f}s')
        if r['max_err'] > eb_rel * dr * 1.01:
            print(f'  WARNING: max_err {r["max_err"]:.3e} exceeds EB {eb_rel*dr:.3e}!')

# ── 5. GP-QoZ-Arith (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ───
# Same GP pre-pass but the QoZ L2+ residuals are compressed with the arithmetic
# encoder instead of Huffman.  The arith encoder approaches Shannon entropy
# (within ~2 bits per stream total overhead vs Huffman's ≤1 bit/symbol overhead).
_arith_available = os.path.isfile(QOZ_ARITH)
if _arith_available:
    print('\n=== GP-QoZ-Arith (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ===')
    comp_arith = HybridQoZGP(qoz_bin=QOZ_ARITH, anchor_frac=0.10, max_step=32, gp_levels=2)
    for eb_rel in EBS:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            cmp_path = os.path.join(td, 'out.hqoz_arith')
            info = comp_arith.compress(data, eb_rel, cmp_path)
            if 'error' in info:
                print(f'  FAIL compress eb={eb_rel}: {info["error"]}'); continue
            dec, t_dec = comp_arith.decompress(cmp_path, DIMS, eb_rel)
            if dec is None:
                print(f'  FAIL decompress eb={eb_rel}'); continue
            sz = info.get('cmp_bytes', os.path.getsize(cmp_path))
            r  = record('gp_qoz_arith', eb_rel, dec, sz, extra=dict(
                n_anchors=info.get('n_anchors', 0),
                n_coarse=info.get('n_coarse', 0),
                header_bytes=info.get('header_bytes', 0),
                qoz_bytes=info.get('qoz_bytes', 0),
                comp_time=info.get('comp_time', 0),
                decomp_time=t_dec,
            ))
            results[f'gp_qoz_arith|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
                  f'maxerr={r["max_err"]:.2e}  (bound={eb_rel*dr:.2e})  '
                  f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
                  f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB  '
                  f't={time.time()-t0:.1f}s')
            if r['max_err'] > eb_rel * dr * 1.01:
                print(f'  WARNING: max_err {r["max_err"]:.3e} exceeds EB {eb_rel*dr:.3e}!')
else:
    print(f'\n[SKIP] GP-QoZ-Arith: binary not found at {QOZ_ARITH}')
    print('  Rebuild QoZ-hybrid with:  cd QoZ-hybrid/build && make -j$(nproc)')

# ── 5b. GP-QoZ-RawInt (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ─
# Bypasses entropy coding entirely: writes raw int32 residuals so that zstd
# can exploit spatial correlations (LZ77 matches) and run patterns.
# Hypothesis: raw_int + zstd closes the gap with the GP-SZ3 Shannon simulation.
_rawint_available = os.path.isfile(QOZ_RAWINT)
if _rawint_available:
    print('\n=== GP-QoZ-RawInt (Approach B, gp_levels=2, anchor_frac=0.10, stride=32) ===')
    comp_rawint = HybridQoZGP(qoz_bin=QOZ_RAWINT, anchor_frac=0.10, max_step=32, gp_levels=2)
    for eb_rel in EBS:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            cmp_path = os.path.join(td, 'out.hqoz_rawint')
            info = comp_rawint.compress(data, eb_rel, cmp_path)
            if 'error' in info:
                print(f'  FAIL compress eb={eb_rel}: {info["error"]}'); continue
            dec, t_dec = comp_rawint.decompress(cmp_path, DIMS, eb_rel)
            if dec is None:
                print(f'  FAIL decompress eb={eb_rel}'); continue
            sz = info.get('cmp_bytes', os.path.getsize(cmp_path))
            r  = record('gp_qoz_rawint', eb_rel, dec, sz, extra=dict(
                n_anchors=info.get('n_anchors', 0),
                n_coarse=info.get('n_coarse', 0),
                header_bytes=info.get('header_bytes', 0),
                qoz_bytes=info.get('qoz_bytes', 0),
                comp_time=info.get('comp_time', 0),
                decomp_time=t_dec,
            ))
            results[f'gp_qoz_rawint|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
                  f'maxerr={r["max_err"]:.2e}  (bound={eb_rel*dr:.2e})  '
                  f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
                  f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB  '
                  f't={time.time()-t0:.1f}s')
            if r['max_err'] > eb_rel * dr * 1.01:
                print(f'  WARNING: max_err {r["max_err"]:.3e} exceeds EB {eb_rel*dr:.3e}!')
else:
    print(f'\n[SKIP] GP-QoZ-RawInt: binary not found at {QOZ_RAWINT}')
    print('  Rebuild QoZ-hybrid with:  cd QoZ-hybrid/build && make -j$(nproc)')

# ── 6. GP-SZ3 variants and GP-Lorenzo (INIT_STRIDE=32, GP_LEVELS=2) ──────────
_hgp_mod.INIT_STRIDE = 32

print('\n=== GP-SZ3 (sim) / GP-SZ3-real / GP-Lorenzo  [stride=32, GP_LEVELS=2] ===')
data_f64 = data.astype(np.float64)
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    try:
        r_gp      = _run_gp_sz3(data_f64, eb_abs)
        level_log = r_gp.get('level_log', [])

        l0l1_bits = sum(d['bits'] for d in level_log if d['level'] < 2)
        l2plus_n  = sum(d['n_pts'] for d in level_log if d['level'] >= 2)

        # ── GP-SZ3 (sim) ──────────────────────────────────────────────────
        r_gpsz3 = dict(
            algorithm='gp_sz3',
            eb_rel=eb_rel, eb_abs=eb_abs,
            bit_rate=r_gp['bit_rate'],
            cr=r_gp['cr'],
            psnr=r_gp['psnr'],
            max_err=float('nan'),
            cmp_bytes=float('nan'),
            data_range=dr,
            n_anchor=r_gp.get('n_anchor', 0),
            H=r_gp.get('H', float('nan')),
            elapsed=r_gp.get('elapsed', 0),
        )
        results[f'gp_sz3|{eb_rel:.2e}'] = r_gpsz3
        print(f'  GP-SZ3-sim  eb={eb_rel:.0e}  PSNR={r_gp["psnr"]:.2f} dB  '
              f'bit_rate={r_gp["bit_rate"]:.3f}  CR={r_gp["cr"]:.1f}×  (cubic-sim L2+)')

        # ── GP-SZ3-real ───────────────────────────────────────────────────
        key_interp = f'sz3_interp|{eb_rel:.2e}'
        if key_interp in results and l2plus_n > 0:
            interp_br      = results[key_interp]['bit_rate']
            l2plus_bits_sz = interp_br * l2plus_n
            gp_sz3r_br     = (l0l1_bits + l2plus_bits_sz) / n
            gp_sz3r_cr     = 32.0 / gp_sz3r_br if gp_sz3r_br > 0 else float('inf')
            r_gpsz3r = dict(
                algorithm='gp_sz3_real',
                eb_rel=eb_rel, eb_abs=eb_abs,
                bit_rate=gp_sz3r_br, cr=gp_sz3r_cr,
                psnr=r_gp['psnr'], max_err=float('nan'),
                cmp_bytes=float('nan'), data_range=dr,
                n_anchor=r_gp.get('n_anchor', 0), H=float('nan'),
                elapsed=r_gp.get('elapsed', 0),
            )
            results[f'gp_sz3_real|{eb_rel:.2e}'] = r_gpsz3r
            print(f'  GP-SZ3-real eb={eb_rel:.0e}  PSNR={r_gp["psnr"]:.2f} dB  '
                  f'bit_rate={gp_sz3r_br:.3f}  CR={gp_sz3r_cr:.1f}×  '
                  f'(GP L0+L1 entropy + real SZ3-Interp L2+)')
        else:
            print(f'  GP-SZ3-real eb={eb_rel:.0e}  SKIP (sz3_interp result missing)')

        # ── GP-Lorenzo ────────────────────────────────────────────────────
        key_lor = f'sz3_lorenzo|{eb_rel:.2e}'
        if key_lor in results and l2plus_n > 0:
            lor_br        = results[key_lor]['bit_rate']
            l2plus_bits_l = lor_br * l2plus_n
            gp_lor_br     = (l0l1_bits + l2plus_bits_l) / n
            gp_lor_cr     = 32.0 / gp_lor_br if gp_lor_br > 0 else float('inf')
            r_gplor = dict(
                algorithm='gp_lorenzo',
                eb_rel=eb_rel, eb_abs=eb_abs,
                bit_rate=gp_lor_br, cr=gp_lor_cr,
                psnr=r_gp['psnr'], max_err=float('nan'),
                cmp_bytes=float('nan'), data_range=dr,
                n_anchor=r_gp.get('n_anchor', 0), H=float('nan'),
                elapsed=r_gp.get('elapsed', 0),
            )
            results[f'gp_lorenzo|{eb_rel:.2e}'] = r_gplor
            print(f'  GP-Lorenzo  eb={eb_rel:.0e}  PSNR={r_gp["psnr"]:.2f} dB  '
                  f'bit_rate={gp_lor_br:.3f}  CR={gp_lor_cr:.1f}×  '
                  f'(GP L0+L1 entropy + real SZ3-Lorenzo L2+)')
        else:
            print(f'  GP-Lorenzo eb={eb_rel:.0e}  SKIP (sz3_lorenzo result missing)')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  FAIL eb={eb_rel}: {e}')

# ── save results ─────────────────────────────────────────────────────────────
out_json = OUT_DIR / 'results_b2.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved {len(results)} records → {out_json}')

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── Style lookup ─────────────────────────────────────────────────────────
    ALG_STYLE = {
        # ── real compressors ──────────────────────────────────────────────
        'gp_qoz_b2':      dict(label='GP-QoZ-Huffman (real, s=32)',          color='#02C39A', marker='o', ls='-',  lw=2.2, ms=8),
        'gp_qoz_arith':   dict(label='GP-QoZ-Arith (real, s=32)',            color='#007B55', marker='*', ls='-',  lw=2.5, ms=10),
        'gp_qoz_rawint':  dict(label='GP-QoZ-RawInt (real, s=32)',           color='#FF6B35', marker='h', ls='-',  lw=2.5, ms=10),
        'qoz_hpez':     dict(label='QoZ HPEZ (−q4)',                     color='#1A6B8A', marker='s', ls='--', lw=1.6, ms=6),
        'sz3_interp':   dict(label='SZ3-Interp',                         color='#F0A500', marker='^', ls='--', lw=1.6, ms=6),
        'sz3_lorenzo':  dict(label='SZ3-Lorenzo+Interp',                  color='#E05C5C', marker='v', ls='--', lw=1.6, ms=6),
        # ── GP-SZ3 variants (simulation) ─────────────────────────────────
        'gp_sz3':       dict(label='GP-SZ3-sim (cubic sim L2+, s=32)',    color='#9B59B6', marker='D', ls='-',  lw=1.8, ms=7),
        'gp_sz3_real':  dict(label='GP-SZ3-real (SZ3-Interp L2+, s=32)', color='#5B2D8E', marker='X', ls='-',  lw=2.2, ms=8),
        'gp_lorenzo':   dict(label='GP-Lorenzo (SZ3-Lorenzo L2+, s=32)',  color='#E67E22', marker='P', ls='-',  lw=1.8, ms=7),
    }

    by_alg = {}
    for key, r in results.items():
        alg = r['algorithm']
        by_alg.setdefault(alg, []).append(r)

    # ── Plot 1: PSNR vs bit-rate (Rate-Distortion) ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg, recs in sorted(by_alg.items()):
        recs = sorted(recs, key=lambda x: x['bit_rate'])
        br = [r['bit_rate'] for r in recs]
        ps = [r['psnr']     for r in recs]
        st = ALG_STYLE.get(alg, dict(label=alg, color='grey', marker='D', ls=':', lw=1.4, ms=5))
        ax.plot(br, ps, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=st['ms'], label=st['label'])
        for r in recs:
            ax.annotate(f'{r["eb_rel"]:.0e}', (r['bit_rate'], r['psnr']),
                        textcoords='offset points', xytext=(4, 3), fontsize=7,
                        color=st['color'])

    ax.set_xlabel('Bit-rate (bits per value)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Hurricane ISABEL W — PSNR vs Bit-rate', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'rd_b2.png', dpi=150)
    plt.close()
    print(f'RD plot saved → {OUT_DIR}/rd_b2.png')

    # ── Plot 2: Compression Ratio vs Relative EB ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg, recs in sorted(by_alg.items()):
        recs = sorted(recs, key=lambda x: x['eb_rel'])
        eb = [r['eb_rel'] for r in recs]
        cr = [r['cr']     for r in recs]
        st = ALG_STYLE.get(alg, dict(label=alg, color='grey', marker='D', ls=':', lw=1.4, ms=5))
        ax.plot(eb, cr, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=st['ms'], label=st['label'])

    ax.set_xscale('log')
    ax.invert_xaxis()           # tighter EB → right side
    ax.xaxis.set_major_formatter(mticker.LogFormatter())
    ax.set_xlabel('Relative error bound (tighter →)', fontsize=12)
    ax.set_ylabel('Compression Ratio (×)', fontsize=12)
    ax.set_title('Hurricane ISABEL W — CR vs Relative Error Bound', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'cr_eb_b2.png', dpi=150)
    plt.close()
    print(f'CR-EB plot saved → {OUT_DIR}/cr_eb_b2.png')

    # ── Plot 3: Header vs QoZ byte breakdown for GP-QoZ variants ────────────
    gp_algs = ['gp_qoz_b2', 'gp_qoz_arith', 'gp_qoz_rawint']
    gp_alg_labels = {'gp_qoz_b2': 'GP-QoZ-Huffman', 'gp_qoz_arith': 'GP-QoZ-Arith', 'gp_qoz_rawint': 'GP-QoZ-RawInt'}
    gp_colors     = {'gp_qoz_b2': '#02C39A',         'gp_qoz_arith': '#007B55',        'gp_qoz_rawint': '#FF6B35'}

    breakdown_data = {}
    for alg in gp_algs:
        recs = [r for r in results.values() if r['algorithm'] == alg
                and 'header_bytes' in r and 'qoz_bytes' in r]
        if recs:
            recs = sorted(recs, key=lambda x: x['eb_rel'])
            breakdown_data[alg] = recs

    if len(breakdown_data) >= 1:
        fig, axes = plt.subplots(1, len(breakdown_data), figsize=(7 * len(breakdown_data), 5),
                                 sharey=True)
        if len(breakdown_data) == 1:
            axes = [axes]
        for ax, (alg, recs) in zip(axes, breakdown_data.items()):
            eb_labels    = [f'{r["eb_rel"]:.0e}' for r in recs]
            hdr_kb       = [r['header_bytes'] / 1024 for r in recs]
            qoz_kb       = [r['qoz_bytes']    / 1024 for r in recs]
            xs = range(len(recs))
            ax.bar(xs, hdr_kb, label='Header (GP anchors + HPs)', color='#A8DADC')
            ax.bar(xs, qoz_kb, bottom=hdr_kb, label='QoZ payload (L2+ residuals)',
                   color=gp_colors.get(alg, '#888'))
            ax.set_xticks(list(xs)); ax.set_xticklabels(eb_labels, rotation=30)
            ax.set_xlabel('Relative EB')
            ax.set_ylabel('Compressed size (KB)')
            ax.set_title(gp_alg_labels.get(alg, alg))
            ax.legend(fontsize=8)
        fig.suptitle('Hurricane ISABEL W — GP-QoZ byte breakdown', fontsize=13)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'breakdown_b2.png', dpi=150)
        plt.close()
        print(f'Breakdown plot saved → {OUT_DIR}/breakdown_b2.png')

except Exception as e:
    import traceback; traceback.print_exc()
    print(f'Plotting skipped: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(v, fmt='.2f'):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '  n/a  '
    return f'{v:{fmt}}'

alg_order = [
    'gp_qoz_rawint', 'gp_qoz_arith', 'gp_qoz_b2',
    'gp_sz3', 'gp_sz3_real', 'gp_lorenzo',
    'qoz_hpez', 'sz3_interp', 'sz3_lorenzo',
]

lines = []
lines.append('Hurricane ISABEL W  —  per-EB metrics table')
lines.append('=' * 100)
hdr = f"{'Algorithm':<22}  {'EB (rel)':<9}  {'PSNR(dB)':<9}  {'CR':>6}  {'bit/val':>7}  {'max_err':>10}  {'EB satisfied?':>14}  {'hdr_KB':>7}  {'qoz_KB':>7}"
lines.append(hdr)
lines.append('-' * 100)

for alg in alg_order:
    recs = sorted(
        [r for r in results.values() if r['algorithm'] == alg],
        key=lambda x: x['eb_rel']
    )
    for r in recs:
        eb_ok = '—'
        if not (isinstance(r.get('max_err', float('nan')), float)
                and math.isnan(r.get('max_err', float('nan')))):
            eb_ok = 'OK' if r['max_err'] <= r['eb_abs'] * 1.01 else f'FAIL ({r["max_err"]:.2e})'
        hdr_kb = r.get('header_bytes', float('nan'))
        qoz_kb = r.get('qoz_bytes',    float('nan'))
        line = (f"{alg:<22}  {r['eb_rel']:<9.1e}  "
                f"{_fmt(r['psnr']):<9}  "
                f"{_fmt(r['cr'], '.1f'):>6}  "
                f"{_fmt(r['bit_rate'], '.3f'):>7}  "
                f"{_fmt(r.get('max_err'), '.2e'):>10}  "
                f"{eb_ok:>14}  "
                f"{_fmt(hdr_kb/1024 if not (isinstance(hdr_kb,float) and math.isnan(hdr_kb)) else float('nan'), '.1f'):>7}  "
                f"{_fmt(qoz_kb/1024 if not (isinstance(qoz_kb,float) and math.isnan(qoz_kb)) else float('nan'), '.1f'):>7}")
        lines.append(line)
    if recs:
        lines.append('')

out_txt = OUT_DIR / 'metrics_b2.txt'
with open(out_txt, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Metrics table saved → {out_txt}')

print('\nDone.')
