#!/usr/bin/env python3
"""
run_hurricane_w_polaris.py
==========================
Test GP-QoZ (stride=32) vs baselines on 3 timesteps of Hurricane ISABEL W.

Results per timestep are written to results_polaris/,
and a final averaged RD plot is saved there too.

Usage (on Polaris, inside the PBS job):
    python run_hurricane_w_polaris.py

Edit the PATHS block below to match your Polaris environment.
"""

import sys, json, math, tempfile, subprocess, os, time
from pathlib import Path
import numpy as np

# ── EDIT THESE FOR POLARIS ───────────────────────────────────────────────────
DATA_DIR   = Path('/lus/eagle/projects/hp-aaims/hurricane/100x500x500')  # <-- change if needed
TIMESTEPS  = ['Wf46.bin.f32', 'Wf47.bin.f32', 'Wf48.bin.f32']
QOZ_HYBRID = '/home/jchen228/QoZ-hybrid/build/test/qoz'   # <-- path after rebuild on Polaris
QOZ_PLAIN  = '/home/jchen228/QoZ/build/test/qoz'
SZ3_BIN    = '/home/jchen228/SZ3/build/tools/sz3/sz3'
ARGONNE_PY = '/home/jchen228/Argonne'                      # root of your Python code on Polaris
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path('/home/jchen228/Argonne/hybrid_qoz_experiment/results_polaris')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(ARGONNE_PY) / 'hybrid_qoz_experiment'))
sys.path.insert(0, ARGONNE_PY)
from hybrid_qoz_gp import HybridQoZGP
import hurricane_hybrid_gp as _hgp_mod
from hurricane_hybrid_gp import run_hybrid_gp as _run_gp_sz3

DIMS = (100, 500, 500)
N    = int(np.prod(DIMS))
EBS  = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]


# ── helpers ──────────────────────────────────────────────────────────────────
def psnr_db(max_err_sq_mean, data_range):
    if max_err_sq_mean <= 0:
        return float('inf')
    return 20.0 * math.log10(data_range / math.sqrt(max_err_sq_mean))

def record(alg, eb_rel, dec, orig, cmp_bytes, dr, extra=None):
    err   = dec.astype(np.float64) - orig.astype(np.float64)
    mse   = float(np.mean(err**2))
    maxe  = float(np.max(np.abs(err)))
    bits  = cmp_bytes * 8
    br    = bits / N
    cr    = (N * 4) / cmp_bytes
    ps    = psnr_db(mse, dr)
    r = dict(algorithm=alg, eb_rel=eb_rel, bit_rate=br, cr=cr, psnr=ps,
             max_err=maxe, cmp_bytes=cmp_bytes, data_range=dr)
    if extra:
        r.update(extra)
    return r

def run_sz3(binary, data, eb_abs, dims, mode_flags, tmp_dir):
    """Run an SZ3-style binary and return (dec_array, cmp_bytes, elapsed)."""
    rdims = list(reversed(dims))
    dim_flags = ['-3', str(rdims[0]), str(rdims[1]), str(rdims[2])]
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        td = Path(td)
        f_in  = td / 'vol.f32'
        f_cmp = td / 'vol.out'
        f_dec = td / 'vol_dec.f32'
        data.astype(np.float32).tofile(f_in)
        t0 = time.time()
        res = subprocess.run(
            [binary, '-f', '-i', str(f_in), '-z', str(f_cmp),
             '-M', 'ABS', str(eb_abs)] + mode_flags + dim_flags,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode(errors='replace')[:300])
        res2 = subprocess.run(
            [binary, '-f', '-z', str(f_cmp), '-o', str(f_dec),
             '-M', 'ABS', str(eb_abs)] + mode_flags + dim_flags,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res2.returncode != 0:
            raise RuntimeError(res2.stderr.decode(errors='replace')[:300])
        elapsed = time.time() - t0
        dec = np.fromfile(f_dec, dtype=np.float32).reshape(dims)
        cmp_bytes = f_cmp.stat().st_size
    return dec, cmp_bytes, elapsed


def run_one_timestep(ts_file: Path) -> dict:
    """Run all methods on a single timestep. Returns results dict."""
    print(f'\n{"="*70}')
    print(f'  Timestep: {ts_file.name}')
    print(f'{"="*70}')

    data   = np.fromfile(ts_file, dtype=np.float32).reshape(DIMS)
    dr     = float(data.max() - data.min())
    results = {}

    # ── 1. QoZ HPEZ (-q 4) ──────────────────────────────────────────────────
    print('\n=== QoZ HPEZ ===')
    for eb_rel in EBS:
        try:
            dec, cb, _ = run_sz3(QOZ_PLAIN, data, eb_rel*dr, DIMS, ['-q', '4'], '/tmp')
            r = record('qoz_hpez', eb_rel, dec, data, cb, dr)
            results[f'qoz_hpez|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×')
        except Exception as e:
            print(f'  FAIL eb={eb_rel}: {e}')

    # ── 2. SZ3-Interp ────────────────────────────────────────────────────────
    print('\n=== SZ3-Interp ===')
    for eb_rel in EBS:
        try:
            dec, cb, _ = run_sz3(SZ3_BIN, data, eb_rel*dr, DIMS,
                                  ['--config', 'SZ3_config_interp.json'], '/tmp') \
                if False else \
                run_sz3(SZ3_BIN, data, eb_rel*dr, DIMS, ['-a', 'interp'], '/tmp')
            r = record('sz3_interp', eb_rel, dec, data, cb, dr)
            results[f'sz3_interp|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×')
        except Exception as e:
            # Fallback: sz3 default (interp-lorenzo)
            try:
                dec, cb, _ = run_sz3(SZ3_BIN, data, eb_rel*dr, DIMS, [], '/tmp')
                r = record('sz3_interp', eb_rel, dec, data, cb, dr)
                results[f'sz3_interp|{eb_rel:.2e}'] = r
                print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}× (fallback)')
            except Exception as e2:
                print(f'  FAIL eb={eb_rel}: {e2}')

    # ── 3. SZ3-Lorenzo ───────────────────────────────────────────────────────
    print('\n=== SZ3-Lorenzo ===')
    for eb_rel in EBS:
        try:
            dec, cb, _ = run_sz3(SZ3_BIN, data, eb_rel*dr, DIMS, ['-a', 'lorenzo_reg'], '/tmp')
            r = record('sz3_lorenzo', eb_rel, dec, data, cb, dr)
            results[f'sz3_lorenzo|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×')
        except Exception as e:
            print(f'  FAIL eb={eb_rel}: {e}')

    # ── 4. GP-QoZ (stride=32) ────────────────────────────────────────────────
    print('\n=== GP-QoZ (stride=32) ===')
    comp = HybridQoZGP(qoz_bin=QOZ_HYBRID, anchor_frac=0.10, max_step=32, gp_levels=2)
    for eb_rel in EBS:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            cmp_path = os.path.join(td, 'out.hqoz')
            info = comp.compress(data, eb_rel, cmp_path)
            if 'error' in info:
                print(f'  FAIL compress eb={eb_rel}: {info["error"]}'); continue
            dec, t_dec = comp.decompress(cmp_path, DIMS, eb_rel)
            cb  = info.get('cmp_bytes', os.path.getsize(cmp_path))
            r   = record('gp_qoz_b2', eb_rel, dec, data, cb, dr,
                         extra=dict(n_anchors=info.get('n_anchors', 0),
                                    comp_time=info.get('comp_time', 0)))
            results[f'gp_qoz_b2|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
                  f'max_err={r["max_err"]:.2e}  (bound={eb_rel*dr:.2e})  '
                  f't={time.time()-t0:.1f}s')
            if r['max_err'] > eb_rel * dr * 1.01:
                print(f'  WARNING: error bound violated!')

    # ── 5. GP-SZ3 variants (stride=32) ───────────────────────────────────────
    _hgp_mod.INIT_STRIDE = 32
    print('\n=== GP-SZ3 variants (stride=32) ===')
    data_f64 = data.astype(np.float64)
    for eb_rel in EBS:
        eb_abs = eb_rel * dr
        try:
            r_gp      = _run_gp_sz3(data_f64, eb_abs)
            level_log = r_gp.get('level_log', [])
            l0l1_bits = sum(d['bits'] for d in level_log if d['level'] < 2)
            l2plus_n  = sum(d['n_pts'] for d in level_log if d['level'] >= 2)

            # gp_sz3: pure simulation
            results[f'gp_sz3|{eb_rel:.2e}'] = dict(
                algorithm='gp_sz3', eb_rel=eb_rel, eb_abs=eb_abs,
                bit_rate=r_gp['bit_rate'], cr=r_gp['cr'], psnr=r_gp['psnr'],
                max_err=float('nan'), cmp_bytes=float('nan'), data_range=dr,
                n_anchor=r_gp.get('n_anchor', 0), H=r_gp.get('H', float('nan')),
                elapsed=r_gp.get('elapsed', 0))

            # gp_sz3_real: GP L0+L1 entropy sim + real SZ3-Interp L2+
            key_interp = f'sz3_interp|{eb_rel:.2e}'
            if key_interp in results and l2plus_n > 0:
                interp_br     = results[key_interp]['bit_rate']
                l2plus_bits   = interp_br * l2plus_n
                gp_sz3r_br    = (l0l1_bits + l2plus_bits) / N
                gp_sz3r_cr    = 32.0 / gp_sz3r_br if gp_sz3r_br > 0 else float('inf')
                results[f'gp_sz3_real|{eb_rel:.2e}'] = dict(
                    algorithm='gp_sz3_real', eb_rel=eb_rel, eb_abs=eb_abs,
                    bit_rate=gp_sz3r_br, cr=gp_sz3r_cr, psnr=r_gp['psnr'],
                    max_err=float('nan'), cmp_bytes=float('nan'), data_range=dr)

            # gp_lorenzo: GP L0+L1 entropy sim + real SZ3-Lorenzo L2+
            key_lor = f'sz3_lorenzo|{eb_rel:.2e}'
            if key_lor in results and l2plus_n > 0:
                lor_br        = results[key_lor]['bit_rate']
                l2plus_bits_l = lor_br * l2plus_n
                gp_lor_br     = (l0l1_bits + l2plus_bits_l) / N
                gp_lor_cr     = 32.0 / gp_lor_br if gp_lor_br > 0 else float('inf')
                results[f'gp_lorenzo|{eb_rel:.2e}'] = dict(
                    algorithm='gp_lorenzo', eb_rel=eb_rel, eb_abs=eb_abs,
                    bit_rate=gp_lor_br, cr=gp_lor_cr, psnr=r_gp['psnr'],
                    max_err=float('nan'), cmp_bytes=float('nan'), data_range=dr)

            print(f'  eb={eb_rel:.0e}  gp_sz3={r_gp["bit_rate"]:.3f} bpv  '
                  f'PSNR={r_gp["psnr"]:.2f} dB')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  FAIL eb={eb_rel}: {e}')

    return results


def plot_and_save(all_results_by_ts: dict, out_dir: Path):
    """Merge multi-timestep results, plot averaged RD curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot')
        return

    ALG_STYLE = {
        'gp_qoz_b2':   dict(label='GP-QoZ (real, s=32)',              color='#02C39A', marker='o', ls='-',  lw=2.2, ms=8),
        'qoz_hpez':    dict(label='QoZ HPEZ (−q4)',                   color='#1A6B8A', marker='s', ls='--', lw=1.6, ms=6),
        'sz3_interp':  dict(label='SZ3-Interp',                       color='#F0A500', marker='^', ls='--', lw=1.6, ms=6),
        'sz3_lorenzo': dict(label='SZ3-Lorenzo+Interp',               color='#E05C5C', marker='v', ls='--', lw=1.6, ms=6),
        'gp_sz3':      dict(label='GP-SZ3-sim (cubic sim L2+, s=32)', color='#9B59B6', marker='D', ls='-',  lw=1.8, ms=7),
        'gp_sz3_real': dict(label='GP-SZ3-real (SZ3 L2+, s=32)',      color='#5B2D8E', marker='X', ls='-',  lw=2.2, ms=8),
        'gp_lorenzo':  dict(label='GP-Lorenzo (Lorenzo L2+, s=32)',    color='#E67E22', marker='P', ls='-',  lw=1.8, ms=7),
    }

    # Collect per-alg per-eb across timesteps for averaging
    from collections import defaultdict
    alg_eb_vals = defaultdict(lambda: defaultdict(list))  # alg -> eb_rel -> list of (br, psnr)
    for ts, results in all_results_by_ts.items():
        for key, r in results.items():
            alg = r['algorithm']
            eb  = r['eb_rel']
            if not (math.isnan(r['bit_rate']) or math.isinf(r['bit_rate'])):
                alg_eb_vals[alg][eb].append((r['bit_rate'], r['psnr']))

    fig, ax = plt.subplots(figsize=(9, 6))
    for alg, eb_map in sorted(alg_eb_vals.items()):
        rows = sorted(eb_map.items())
        br_avg   = [np.mean([v[0] for v in vals]) for _, vals in rows]
        psnr_avg = [np.mean([v[1] for v in vals]) for _, vals in rows]
        st = ALG_STYLE.get(alg, dict(label=alg, color='grey', marker='D', ls=':', lw=1.4, ms=5))
        ax.plot(br_avg, psnr_avg, color=st['color'], marker=st['marker'],
                linestyle=st['ls'], lw=st['lw'], ms=st['ms'], label=st['label'])

    ts_names = ', '.join(list(all_results_by_ts.keys()))
    ax.set_xlabel('Bit rate (bits/value)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Hurricane W — averaged over {len(all_results_by_ts)} timesteps\n({ts_names})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / 'rd_hurricane_w_polaris.png'
    fig.savefig(plot_path, dpi=150)
    print(f'\nPlot saved: {plot_path}')
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_results = {}

    for ts_name in TIMESTEPS:
        ts_file = DATA_DIR / ts_name
        if not ts_file.exists():
            print(f'WARNING: {ts_file} not found, skipping')
            continue

        results = run_one_timestep(ts_file)
        all_results[ts_name] = results

        # Save per-timestep JSON
        out_json = OUT_DIR / f'results_{Path(ts_name).stem}.json'
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: None if math.isnan(x) else x)
        print(f'\nSaved: {out_json}')

    # Save combined JSON
    combined_path = OUT_DIR / 'results_combined.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f'\nCombined results: {combined_path}')

    plot_and_save(all_results, OUT_DIR)
    print('\nDone.')
