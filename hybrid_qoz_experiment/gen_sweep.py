#!/usr/bin/env python3
"""
Sweep anchor_frac values + add SZ3 and plain QoZ baselines.
Generates results/results.json for plot_results.py.
"""
import sys, json, math, tempfile, subprocess, os
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/jchen228/Desktop/Argonne/hybrid_qoz_experiment')
from hybrid_qoz_gp import HybridQoZGP

DATA_PATH  = '/Users/jchen228/Desktop/Argonne/100x500x500/Wf48.bin.f32'
QOZ_HYBRID = '/Users/jchen228/Desktop/Argonne/QoZ-hybrid/build/test/qoz'
QOZ_PLAIN  = '/Users/jchen228/Desktop/Argonne/QoZ/build/test/qoz'
SZ3_BIN    = '/Users/jchen228/Desktop/Argonne/SZ3/build/tools/sz3/sz3'
RESULTS    = '/Users/jchen228/Desktop/Argonne/hybrid_qoz_experiment/results/results.json'
DIMS       = (100, 500, 500)
EBS        = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
FRACS      = [0.10]      # fixed 10% anchors (Approach B)
STEPS      = [32, 64]   # max_step sweep at frac=1.0 (step=16 crashes QoZ SIGSEGV)

data = np.fromfile(DATA_PATH, dtype=np.float32).reshape(DIMS)
dr   = float(data.max() - data.min())
n    = data.size

def psnr(orig, dec):
    mse = float(np.mean((orig.astype(np.float64) - dec.astype(np.float64))**2))
    return 20*math.log10(dr / math.sqrt(mse)) if mse > 0 else float('inf')

def ssim_approx(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(), b.std()
    s12 = float(np.mean((a-mu1)*(b-mu2)))
    c1, c2 = (0.01*dr)**2, (0.03*dr)**2
    return float((2*mu1*mu2+c1)*(2*s12+c2)/((mu1**2+mu2**2+c1)*(s1**2+s2**2+c2)))

results = {}

# ── Plain QoZ (HPEZ -q4) ─────────────────────────────────────────────────────
print('=== Plain QoZ / HPEZ ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi = os.path.join(td, 'v.f32'); fc = os.path.join(td, 'v.out'); fd = os.path.join(td, 'v.dec')
        data.tofile(fi)
        r = subprocess.run([QOZ_PLAIN,'-f','-i',fi,'-z',fc,'-3','500','500','100',
                            '-M','ABS',str(eb_abs),'-q','4'], capture_output=True)
        if r.returncode: print(f'  FAIL cmp eb={eb_rel}'); continue
        r2 = subprocess.run([QOZ_PLAIN,'-f','-z',fc,'-o',fd,'-3','500','500','100',
                             '-M','ABS',str(eb_abs),'-q','4'], capture_output=True)
        if r2.returncode: print(f'  FAIL dec eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        key = f'hurricane_w|W|qoz_hpez|{eb_rel:.2e}'
        results[key] = dict(dataset='hurricane_w', field='W', algorithm='qoz_hpez',
                            eb_rel=eb_rel, eb_abs=eb_abs,
                            bit_rate=sz*8/n, cr=n*4/sz,
                            psnr=psnr(data,dec), ssim=ssim_approx(data,dec),
                            max_err=float(np.max(np.abs(data-dec))),
                            rmse=float(np.sqrt(np.mean((data-dec)**2))),
                            data_range=dr, cmp_bytes=sz)
        print(f'  eb={eb_rel:.0e}  PSNR={results[key]["psnr"]:.2f}dB  CR={results[key]["cr"]:.1f}x')

# ── SZ3-Interp (pure interpolation, ALGO_INTERP via config) ──────────────────
print('\n=== SZ3-Interp (ALGO_INTERP) ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi = os.path.join(td, 'v.f32'); fc = os.path.join(td, 'v.out'); fd = os.path.join(td, 'v.dec')
        cfg = os.path.join(td, 'sz3.cfg')
        with open(cfg, 'w') as cf:
            cf.write('[GlobalSettings]\nCmprAlgo = ALGO_INTERP\n'
                     f'ErrorBoundMode = ABS\nAbsErrorBound = {eb_abs}\n')
        data.tofile(fi)
        r = subprocess.run([SZ3_BIN,'-f','-i',fi,'-z',fc,'-3','500','500','100',
                            '-M','ABS',str(eb_abs),'-c',cfg], capture_output=True)
        if r.returncode: print(f'  FAIL cmp eb={eb_rel}: {r.stderr[:100]}'); continue
        r2 = subprocess.run([SZ3_BIN,'-f','-z',fc,'-o',fd,'-3','500','500','100',
                             '-M','ABS',str(eb_abs),'-c',cfg], capture_output=True)
        if r2.returncode: print(f'  FAIL dec eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        key = f'hurricane_w|W|sz3_interp|{eb_rel:.2e}'
        results[key] = dict(dataset='hurricane_w', field='W', algorithm='sz3_interp',
                            eb_rel=eb_rel, eb_abs=eb_abs,
                            bit_rate=sz*8/n, cr=n*4/sz,
                            psnr=psnr(data,dec), ssim=ssim_approx(data,dec),
                            max_err=float(np.max(np.abs(data-dec))),
                            rmse=float(np.sqrt(np.mean((data-dec)**2))),
                            data_range=dr, cmp_bytes=sz)
        print(f'  eb={eb_rel:.0e}  PSNR={results[key]["psnr"]:.2f}dB  CR={results[key]["cr"]:.1f}x')

# ── SZ3-Lorenzo+Interp (ALGO_INTERP_LORENZO, SZ3 default) ────────────────────
print('\n=== SZ3-Lorenzo+Interp (ALGO_INTERP_LORENZO) ===')
for eb_rel in EBS:
    eb_abs = eb_rel * dr
    with tempfile.TemporaryDirectory() as td:
        fi = os.path.join(td, 'v.f32'); fc = os.path.join(td, 'v.out'); fd = os.path.join(td, 'v.dec')
        data.tofile(fi)
        # No config file → SZ3 default = ALGO_INTERP_LORENZO
        r = subprocess.run([SZ3_BIN,'-f','-i',fi,'-z',fc,'-3','500','500','100',
                            '-M','ABS',str(eb_abs)], capture_output=True)
        if r.returncode: print(f'  FAIL cmp eb={eb_rel}: {r.stderr[:100]}'); continue
        r2 = subprocess.run([SZ3_BIN,'-f','-z',fc,'-o',fd,'-3','500','500','100',
                             '-M','ABS',str(eb_abs)], capture_output=True)
        if r2.returncode: print(f'  FAIL dec eb={eb_rel}'); continue
        dec = np.fromfile(fd, dtype=np.float32).reshape(DIMS)
        sz  = os.path.getsize(fc)
        key = f'hurricane_w|W|sz3_lorenzo|{eb_rel:.2e}'
        results[key] = dict(dataset='hurricane_w', field='W', algorithm='sz3_lorenzo',
                            eb_rel=eb_rel, eb_abs=eb_abs,
                            bit_rate=sz*8/n, cr=n*4/sz,
                            psnr=psnr(data,dec), ssim=ssim_approx(data,dec),
                            max_err=float(np.max(np.abs(data-dec))),
                            rmse=float(np.sqrt(np.mean((data-dec)**2))),
                            data_range=dr, cmp_bytes=sz)
        print(f'  eb={eb_rel:.0e}  PSNR={results[key]["psnr"]:.2f}dB  CR={results[key]["cr"]:.1f}x')

# ── Hybrid QoZ+GP, anchor_frac sweep ─────────────────────────────────────────
print('\n=== Hybrid QoZ+GP anchor_frac sweep ===')
comp_cache = {}  # reuse GP HPs across EBs for same frac
for frac in FRACS:
    print(f'\n  anchor_frac={frac:.2f}')
    alg_id = f'hybrid_f{int(frac*100):03d}'
    comp = HybridQoZGP(qoz_bin=QOZ_HYBRID, anchor_frac=frac, max_step=64, gp_levels=2)
    for eb_rel in EBS:
        with tempfile.TemporaryDirectory() as td:
            out_p = os.path.join(td, 'out.hqoz')
            info = comp.compress(data, eb_rel, out_p)
            if 'error' in info:
                print(f'    FAIL cmp eb={eb_rel}: {info["error"]}'); continue
            dec, _ = comp.decompress(out_p, DIMS, eb_rel)
            if dec is None:
                print(f'    FAIL dec eb={eb_rel}'); continue
            sz  = info.get('cmp_bytes', os.path.getsize(out_p))
            key = f'hurricane_w|W|{alg_id}|{eb_rel:.2e}'
            results[key] = dict(dataset='hurricane_w', field='W', algorithm=alg_id,
                                eb_rel=eb_rel, eb_abs=eb_rel*dr,
                                bit_rate=sz*8/n, cr=n*4/sz,
                                psnr=psnr(data,dec), ssim=ssim_approx(data,dec),
                                max_err=float(np.max(np.abs(data-dec))),
                                rmse=float(np.sqrt(np.mean((data-dec)**2))),
                                data_range=dr, cmp_bytes=sz,
                                n_anchors=info.get('n_anchors',0),
                                n_coarse=info.get('n_coarse',0),
                                anchor_frac=frac)
            print(f'    eb={eb_rel:.0e}  PSNR={results[key]["psnr"]:.2f}dB  '
                  f'CR={results[key]["cr"]:.1f}x  '
                  f'anchors={info.get("n_anchors",0)}/{info.get("n_coarse",0)}  '
                  f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
                  f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB')

# Save after frac sweep so a crash in the step sweep doesn't lose these results
Path(RESULTS).parent.mkdir(exist_ok=True)
with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n[checkpoint] Saved {len(results)} records → {RESULTS}')

# ── Hybrid QoZ+GP, max_step sweep (frac=1.0 — all coarse exact) ──────────────
print('\n=== Hybrid QoZ+GP max_step sweep (frac=1.0) ===')
for ms in STEPS:
    print(f'\n  max_step={ms}')
    alg_id = f'hybrid_s{ms:03d}'
    comp = HybridQoZGP(qoz_bin=QOZ_HYBRID, anchor_frac=1.0, max_step=ms, gp_levels=2)
    for eb_rel in EBS:
        with tempfile.TemporaryDirectory() as td:
            out_p = os.path.join(td, 'out.hqoz')
            info = comp.compress(data, eb_rel, out_p)
            if 'error' in info:
                print(f'    FAIL cmp eb={eb_rel}: {info["error"]}'); continue
            dec, _ = comp.decompress(out_p, DIMS, eb_rel)
            if dec is None:
                print(f'    FAIL dec eb={eb_rel}'); continue
            sz  = info.get('cmp_bytes', os.path.getsize(out_p))
            key = f'hurricane_w|W|{alg_id}|{eb_rel:.2e}'
            results[key] = dict(dataset='hurricane_w', field='W', algorithm=alg_id,
                                eb_rel=eb_rel, eb_abs=eb_rel*dr,
                                bit_rate=sz*8/n, cr=n*4/sz,
                                psnr=psnr(data,dec), ssim=ssim_approx(data,dec),
                                max_err=float(np.max(np.abs(data-dec))),
                                rmse=float(np.sqrt(np.mean((data-dec)**2))),
                                data_range=dr, cmp_bytes=sz,
                                n_anchors=info.get('n_anchors',0),
                                n_coarse=info.get('n_coarse',0),
                                header_bytes=info.get('header_bytes',0),
                                qoz_bytes=info.get('qoz_bytes',0),
                                anchor_frac=1.0, max_step=ms)
            print(f'    eb={eb_rel:.0e}  PSNR={results[key]["psnr"]:.2f}dB  '
                  f'CR={results[key]["cr"]:.1f}x  '
                  f'anchors={info.get("n_anchors",0)}/{info.get("n_coarse",0)}  '
                  f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
                  f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB')

Path(RESULTS).parent.mkdir(exist_ok=True)
with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved {len(results)} records → {RESULTS}')
