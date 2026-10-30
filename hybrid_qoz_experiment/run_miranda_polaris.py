#!/usr/bin/env python3
"""
run_miranda_polaris.py
======================
Run GP-QoZ (stride=32, gp_levels=2) vs baselines on the Miranda density
dataset (SDRBENCH-Miranda-3072x3072x3072, 7 blocks).

Each block is ~1.2 GB (32×3072×3072 float32).  Results per block are saved
to results_miranda/ as individual JSON files, plus a combined JSON and an
averaged RD plot.

Usage (inside a PBS job on Polaris):
    python run_miranda_polaris.py

Edit the PATHS block below if your Polaris paths differ.
"""

import sys, json, math, glob, tempfile, subprocess, os, time
from pathlib import Path
import numpy as np

# ── EDIT THESE FOR POLARIS ───────────────────────────────────────────────────
DATA_DIR    = Path('/grand/projects/SDR/Miranda/SDRBENCH-Miranda-3072x3072x3072')
N_BLOCKS    = 7              # first N density-*.f32 files (sorted)
QOZ_HYBRID  = '/home/jchen228/QoZ-hybrid/build/test/qoz'
QOZ_PLAIN   = '/home/jchen228/QoZ/build/test/qoz'
SZ3_BIN     = '/home/jchen228/SZ3/build/tools/sz3/sz3'
ARGONNE_PY  = '/home/jchen228/Argonne'
SCRATCH_DIR = '/local/scratch'   # fast local scratch on Polaris compute nodes
# ─────────────────────────────────────────────────────────────────────────────

DIMS = (32, 3072, 3072)      # NZ, NY, NX
N    = int(np.prod(DIMS))    # 301,989,888

EBS  = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

OUT_DIR = Path('/home/jchen228/Argonne/hybrid_qoz_experiment/results_miranda')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(ARGONNE_PY) / 'hybrid_qoz_experiment'))
sys.path.insert(0, ARGONNE_PY)
from hybrid_qoz_gp import HybridQoZGP
import hurricane_hybrid_gp as _hgp_mod
from hurricane_hybrid_gp import run_hybrid_gp as _run_gp_sz3


# ── helpers ──────────────────────────────────────────────────────────────────

def psnr_db(mse, data_range):
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def record(alg, eb_rel, dec, orig, cmp_bytes, dr, extra=None):
    err  = dec.astype(np.float64) - orig.astype(np.float64)
    mse  = float(np.mean(err ** 2))
    maxe = float(np.max(np.abs(err)))
    br   = cmp_bytes * 8 / N
    cr   = (N * 4) / cmp_bytes
    r = dict(
        algorithm=alg, eb_rel=eb_rel, eb_abs=eb_rel * dr,
        bit_rate=br, cr=cr, psnr=psnr_db(mse, dr),
        max_err=maxe, cmp_bytes=cmp_bytes, data_range=dr,
    )
    if extra:
        r.update(extra)
    return r


def run_sz3(binary, data, eb_abs, dims, extra_flags, tmp_dir):
    """Compress + decompress via an SZ3-style CLI binary. Returns (dec, cmp_bytes, elapsed)."""
    rdims     = list(reversed(dims))                      # NX NY NZ order
    dim_flags = ['-3', str(rdims[0]), str(rdims[1]), str(rdims[2])]

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        td     = Path(td)
        f_in   = td / 'vol.f32'
        f_cmp  = td / 'vol.cmp'
        f_dec  = td / 'vol_dec.f32'

        data.astype(np.float32).tofile(f_in)

        t0  = time.time()
        res = subprocess.run(
            [binary, '-f', '-i', str(f_in), '-z', str(f_cmp),
             '-M', 'ABS', str(eb_abs)] + extra_flags + dim_flags,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode(errors='replace')[:400])

        res2 = subprocess.run(
            [binary, '-f', '-z', str(f_cmp), '-o', str(f_dec),
             '-M', 'ABS', str(eb_abs)] + extra_flags + dim_flags,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res2.returncode != 0:
            raise RuntimeError(res2.stderr.decode(errors='replace')[:400])

        elapsed   = time.time() - t0
        dec       = np.fromfile(f_dec, dtype=np.float32).reshape(dims)
        cmp_bytes = f_cmp.stat().st_size

    return dec, cmp_bytes, elapsed


def run_one_block(block_file: Path) -> dict:
    print(f'\n{"="*70}')
    print(f'  Block: {block_file.name}  ({block_file.stat().st_size / 1e9:.2f} GB)')
    print(f'{"="*70}')

    data = np.fromfile(block_file, dtype=np.float32).reshape(DIMS)
    dr   = float(data.max() - data.min())
    print(f'  shape={data.shape}  data_range={dr:.4f}')
    results = {}

    # Use local scratch for temp files to avoid hitting /tmp limits
    tmp = SCRATCH_DIR if os.path.isdir(SCRATCH_DIR) else '/tmp'

    # ── 1. QoZ HPEZ (-q 4) ──────────────────────────────────────────────────
    print('\n=== QoZ HPEZ (-q4) ===')
    for eb_rel in EBS:
        try:
            dec, cb, elapsed = run_sz3(QOZ_PLAIN, data, eb_rel * dr, DIMS, ['-q', '4'], tmp)
            r = record('qoz_hpez', eb_rel, dec, data, cb, dr, extra=dict(elapsed=elapsed))
            results[f'qoz_hpez|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  t={elapsed:.1f}s')
        except Exception as e:
            print(f'  FAIL eb={eb_rel}: {e}')

    # ── 2. SZ3-Interp ────────────────────────────────────────────────────────
    print('\n=== SZ3-Interp ===')
    for eb_rel in EBS:
        try:
            dec, cb, elapsed = run_sz3(SZ3_BIN, data, eb_rel * dr, DIMS, [], tmp)
            r = record('sz3_interp', eb_rel, dec, data, cb, dr, extra=dict(elapsed=elapsed))
            results[f'sz3_interp|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  t={elapsed:.1f}s')
        except Exception as e:
            print(f'  FAIL eb={eb_rel}: {e}')

    # ── 3. SZ3-Lorenzo ───────────────────────────────────────────────────────
    print('\n=== SZ3-Lorenzo+Interp (default) ===')
    for eb_rel in EBS:
        try:
            # SZ3 default mode is interp-lorenzo; use -a lorenzo_reg to isolate
            dec, cb, elapsed = run_sz3(SZ3_BIN, data, eb_rel * dr, DIMS, ['-a', 'lorenzo_reg'], tmp)
            r = record('sz3_lorenzo', eb_rel, dec, data, cb, dr, extra=dict(elapsed=elapsed))
            results[f'sz3_lorenzo|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  t={elapsed:.1f}s')
        except Exception as e:
            # SZ3 on some builds doesn't support -a; fall back to default
            try:
                dec, cb, elapsed = run_sz3(SZ3_BIN, data, eb_rel * dr, DIMS, [], tmp)
                r = record('sz3_lorenzo', eb_rel, dec, data, cb, dr, extra=dict(elapsed=elapsed))
                results[f'sz3_lorenzo|{eb_rel:.2e}'] = r
                print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  t={elapsed:.1f}s (fallback)')
            except Exception as e2:
                print(f'  FAIL eb={eb_rel}: {e2}')

    # ── 4. GP-QoZ-Huffman (stride=32, gp_levels=2) ──────────────────────────
    print('\n=== GP-QoZ-Huffman (stride=32, gp_levels=2) ===')
    comp = HybridQoZGP(qoz_bin=QOZ_HYBRID, anchor_frac=0.10, max_step=32, gp_levels=2)
    for eb_rel in EBS:
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=tmp) as td:
            cmp_path = os.path.join(td, 'out.hqoz')
            info = comp.compress(data, eb_rel, cmp_path)
            if 'error' in info:
                print(f'  FAIL compress eb={eb_rel}: {info["error"]}'); continue
            dec, t_dec = comp.decompress(cmp_path, DIMS, eb_rel)
            if dec is None:
                print(f'  FAIL decompress eb={eb_rel}'); continue
            cb = info.get('cmp_bytes', os.path.getsize(cmp_path))
            r  = record('gp_qoz_b2', eb_rel, dec, data, cb, dr, extra=dict(
                n_anchors=info.get('n_anchors', 0),
                header_bytes=info.get('header_bytes', 0),
                qoz_bytes=info.get('qoz_bytes', 0),
                comp_time=info.get('comp_time', 0),
                decomp_time=t_dec,
            ))
            results[f'gp_qoz_b2|{eb_rel:.2e}'] = r
            print(f'  eb={eb_rel:.0e}  PSNR={r["psnr"]:.2f} dB  CR={r["cr"]:.1f}×  '
                  f'max_err={r["max_err"]:.2e}  (bound={eb_rel*dr:.2e})  '
                  f'hdr={info.get("header_bytes",0)/1024:.1f}KB  '
                  f'qoz={info.get("qoz_bytes",0)/1024:.1f}KB  '
                  f't={time.time()-t0:.1f}s')
            if r['max_err'] > eb_rel * dr * 1.01:
                print(f'  WARNING: error bound violated!')

    # ── 5. GP-SZ3 simulation (stride=32) ────────────────────────────────────
    # Note: GP-SZ3 runs in Python on float64 — this may be slow for 300M-point
    # Miranda blocks.  Skipped if it takes too long.
    _hgp_mod.INIT_STRIDE = 32
    print('\n=== GP-SZ3 sim (stride=32) ===')
    data_f64 = data.astype(np.float64)
    for eb_rel in EBS:
        eb_abs = eb_rel * dr
        try:
            t0   = time.time()
            r_gp = _run_gp_sz3(data_f64, eb_abs)
            level_log = r_gp.get('level_log', [])
            l0l1_bits = sum(d['bits'] for d in level_log if d['level'] < 2)
            l2plus_n  = sum(d['n_pts'] for d in level_log if d['level'] >= 2)

            results[f'gp_sz3|{eb_rel:.2e}'] = dict(
                algorithm='gp_sz3', eb_rel=eb_rel, eb_abs=eb_abs,
                bit_rate=r_gp['bit_rate'], cr=r_gp['cr'], psnr=r_gp['psnr'],
                max_err=float('nan'), cmp_bytes=float('nan'), data_range=dr,
                n_anchor=r_gp.get('n_anchor', 0), H=r_gp.get('H', float('nan')),
                elapsed=time.time() - t0,
            )

            # gp_sz3_real: GP L0+L1 entropy + real SZ3-Interp for L2+
            key_interp = f'sz3_interp|{eb_rel:.2e}'
            if key_interp in results and l2plus_n > 0:
                interp_br  = results[key_interp]['bit_rate']
                gp_sz3r_br = (l0l1_bits + interp_br * l2plus_n) / N
                gp_sz3r_cr = 32.0 / gp_sz3r_br if gp_sz3r_br > 0 else float('inf')
                results[f'gp_sz3_real|{eb_rel:.2e}'] = dict(
                    algorithm='gp_sz3_real', eb_rel=eb_rel, eb_abs=eb_abs,
                    bit_rate=gp_sz3r_br, cr=gp_sz3r_cr, psnr=r_gp['psnr'],
                    max_err=float('nan'), cmp_bytes=float('nan'), data_range=dr)

            print(f'  eb={eb_rel:.0e}  gp_sz3={r_gp["bit_rate"]:.3f} bpv  '
                  f'PSNR={r_gp["psnr"]:.2f} dB  t={time.time()-t0:.1f}s')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  FAIL eb={eb_rel}: {e}')

    return results


def plot_and_save(all_results_by_block: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot')
        return

    ALG_STYLE = {
        'gp_qoz_b2':   dict(label='GP-QoZ-Huffman (real, s=32)',        color='#02C39A', marker='o', ls='-',  lw=2.2, ms=8),
        'qoz_hpez':    dict(label='QoZ HPEZ (−q4)',                     color='#1A6B8A', marker='s', ls='--', lw=1.6, ms=6),
        'sz3_interp':  dict(label='SZ3-Interp',                         color='#F0A500', marker='^', ls='--', lw=1.6, ms=6),
        'sz3_lorenzo': dict(label='SZ3-Lorenzo+Interp',                 color='#E05C5C', marker='v', ls='--', lw=1.6, ms=6),
        'gp_sz3':      dict(label='GP-SZ3-sim (cubic L2+, s=32)',       color='#9B59B6', marker='D', ls='-',  lw=1.8, ms=7),
        'gp_sz3_real': dict(label='GP-SZ3-real (SZ3-Interp L2+, s=32)',color='#5B2D8E', marker='X', ls='-',  lw=2.2, ms=8),
    }

    from collections import defaultdict
    alg_eb_vals = defaultdict(lambda: defaultdict(list))
    for blk, results in all_results_by_block.items():
        for key, r in results.items():
            alg = r['algorithm']
            eb  = r['eb_rel']
            br, ps = r['bit_rate'], r['psnr']
            if not (isinstance(br, float) and math.isnan(br)):
                alg_eb_vals[alg][eb].append((br, ps))

    fig, ax = plt.subplots(figsize=(9, 6))
    for alg, eb_map in sorted(alg_eb_vals.items()):
        rows     = sorted(eb_map.items())
        br_avg   = [np.mean([v[0] for v in vals]) for _, vals in rows]
        psnr_avg = [np.mean([v[1] for v in vals]) for _, vals in rows]
        st = ALG_STYLE.get(alg, dict(label=alg, color='grey', marker='D', ls=':', lw=1.4, ms=5))
        ax.plot(br_avg, psnr_avg, color=st['color'], marker=st['marker'],
                linestyle=st['ls'], lw=st['lw'], ms=st['ms'], label=st['label'])

    ax.set_xlabel('Bit rate (bits/value)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Miranda density — averaged over {len(all_results_by_block)} blocks', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / 'rd_miranda_polaris.png'
    fig.savefig(plot_path, dpi=150)
    print(f'\nRD plot saved: {plot_path}')
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Discover density blocks
    block_files = sorted(DATA_DIR.glob('density-*.f32'))[:N_BLOCKS]

    if not block_files:
        print(f'ERROR: no density-*.f32 files found under {DATA_DIR}')
        sys.exit(1)

    print(f'Found {len(block_files)} Miranda density blocks:')
    for f in block_files:
        print(f'  {f.name}  ({f.stat().st_size/1e9:.2f} GB)')

    all_results = {}

    for block_file in block_files:
        results = run_one_block(block_file)
        all_results[block_file.name] = results

        # Save per-block JSON immediately (in case job gets killed)
        out_json = OUT_DIR / f'results_{block_file.stem}.json'
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(f'\nSaved: {out_json}')

    # Save combined JSON
    combined_path = OUT_DIR / 'results_miranda_combined.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f'\nCombined results: {combined_path}')

    plot_and_save(all_results, OUT_DIR)
    print('\nDone.')
