"""
run_experiments.py

Orchestrator for Hybrid QoZ+GP experiments.
Loops over all (dataset × algorithm × error_bound) combinations,
runs compression/decompression, records metrics, saves resumable JSON results.

Usage (on the cluster):
    python run_experiments.py [--config config.yaml] [--resume] [--datasets rtm cesm]

The script is safe to Ctrl+C and re-run with --resume; already-completed
(dataset, field, algorithm, eb) tuples are skipped.
"""

import argparse
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

# ── lazy import of hybrid module ────────────────────────────────────────────
# (only needed if hybrid algorithm is requested)
_hybrid_module = None

def _get_hybrid_module():
    global _hybrid_module
    if _hybrid_module is None:
        sys.path.insert(0, str(Path(__file__).parent))
        import hybrid_qoz_gp as m
        _hybrid_module = m
    return _hybrid_module


# ────────────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_bin(cfg: dict, key: str) -> str:
    return cfg['binaries'][key]


# ────────────────────────────────────────────────────────────────────────────
# Results I/O  (resumable JSON store)
# ────────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def result_key(dataset_id: str, field_id: str, alg_id: str, eb: float) -> str:
    return f'{dataset_id}|{field_id}|{alg_id}|{eb:.2e}'


# ────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────────────

def load_f32(path: str, shape: tuple) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f'Size mismatch: file has {arr.size} elements, shape {shape} needs {expected}')
    return arr.reshape(shape)


def iter_dataset_files(ds_cfg: dict, ds_id: str):
    """
    Yield (field_id, file_path, shape) tuples for a dataset config block.
    """
    ds_type = ds_cfg.get('type', 'single_file')
    shape   = tuple(ds_cfg.get('shape', []))

    if ds_type == 'single_file':
        yield (ds_cfg.get('field', ds_id),
               ds_cfg['path'],
               shape)

    elif ds_type == 'multi_field_1d':
        for field in ds_cfg.get('fields', []):
            fid  = field['id']
            base = Path(ds_cfg['base_path']) / field.get('subdirectory', '')
            glob = field.get('glob', '*.f32')
            files = sorted(base.glob(glob))
            if not files:
                print(f'  WARNING: no files found for {ds_id}/{fid} in {base}')
                continue
            f = files[0]
            n = f.stat().st_size // 4
            yield (fid, str(f), (n,))

    elif ds_type == 'multi_field_2d':
        glob      = ds_cfg.get('file_glob', '*.f32')
        n_files   = ds_cfg.get('n_files_per_field', 1)
        for field in ds_cfg.get('fields', []):
            fid   = field['id']
            fdir  = Path(ds_cfg['base_path']) / field.get('path', fid)
            files = sorted(fdir.glob(glob))[:n_files]
            if not files:
                print(f'  WARNING: no files found for {ds_id}/{fid} in {fdir}')
                continue
            for fpath in files:
                tag = f'{fid}_{fpath.stem}' if len(files) > 1 else fid
                yield (tag, str(fpath), shape)

    elif ds_type == 'temporal_3d':
        base     = Path(ds_cfg['base_path'])
        glob     = ds_cfg.get('file_glob', '*.f32')
        n_files  = ds_cfg.get('n_files', None)
        sel_mode = ds_cfg.get('file_selection', 'first_n')
        files    = sorted(base.glob(glob))
        if sel_mode == 'first_n' and n_files:
            files = files[:n_files]
        for fpath in files:
            yield (fpath.stem, str(fpath), shape)

    elif ds_type == 'multi_field_3d':
        timesteps = ds_cfg.get('timesteps', [])
        for field in ds_cfg.get('fields', []):
            fid  = field['id']
            fdir = Path(ds_cfg['base_path']) / field.get('subdirectory', fid)
            if timesteps:
                for ts in timesteps:
                    fpath = fdir / ts
                    if fpath.exists():
                        yield (f'{fid}_{ts}', str(fpath), shape)
                    else:
                        print(f'  WARNING: not found: {fpath}')
            else:
                files = sorted(fdir.glob('*.f32'))
                for fpath in files:
                    yield (f'{fid}_{fpath.stem}', str(fpath), shape)

    elif ds_type == 'spatial_blocks_3d':
        base      = Path(ds_cfg['base_path'])
        glob      = ds_cfg.get('block_glob', '*.f32')
        n_blocks  = ds_cfg.get('n_blocks', None)
        sel_mode  = ds_cfg.get('block_selection', 'first_n')
        files     = sorted(base.glob(glob))
        if sel_mode == 'first_n' and n_blocks:
            files = files[:n_blocks]
        for fpath in files:
            yield (fpath.stem, str(fpath), shape)

    else:
        raise ValueError(f'Unknown dataset type: {ds_type}')


# ────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(orig: np.ndarray, dec: np.ndarray,
                    cmp_bytes: int) -> dict:
    n      = orig.size
    err    = dec.astype(np.float64) - orig.astype(np.float64)
    mse    = float(np.mean(err**2))
    max_e  = float(np.max(np.abs(err)))
    vrange = float(orig.max() - orig.min())
    psnr   = (float('inf') if mse == 0
               else 10 * math.log10((vrange**2) / mse))
    ssim   = _ssim_approx(orig, dec)
    bit_rate = cmp_bytes * 8 / n
    cr       = (n * 4) / cmp_bytes
    return dict(bit_rate=bit_rate, cr=cr, psnr=psnr, ssim=ssim,
                max_err=max_e, rmse=math.sqrt(mse),
                data_range=vrange, cmp_bytes=cmp_bytes)


def _ssim_approx(a: np.ndarray, b: np.ndarray,
                 win: int = 8) -> float:
    """Lightweight patch-based SSIM approximation (no scipy needed)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n    = len(a)
    step = max(1, n // (win * win * win))
    idx  = np.arange(0, n, step)
    ap, bp = a[idx], b[idx]
    mu_a, mu_b   = ap.mean(), bp.mean()
    sig_a, sig_b = ap.std() + 1e-8, bp.std() + 1e-8
    sig_ab = np.mean((ap - mu_a) * (bp - mu_b))
    c1, c2 = 0.01**2, 0.03**2
    num = (2*mu_a*mu_b + c1) * (2*sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a**2 + sig_b**2 + c2)
    return float(num / den)


# ────────────────────────────────────────────────────────────────────────────
# Algorithm runners
# ────────────────────────────────────────────────────────────────────────────

def _dim_flags(shape):
    # QoZ/SZ3 convention: -N nx ny nz where data layout is data[nz][ny][nx]
    # i.e. fastest-varying dimension first → pass numpy shape reversed.
    rdims = list(reversed(shape))
    if len(shape) == 1: return ['-1', str(rdims[0])]
    if len(shape) == 2: return ['-2', str(rdims[0]), str(rdims[1])]
    return ['-3', str(rdims[0]), str(rdims[1]), str(rdims[2])]


def run_qoz_hpez(data, shape, eb_rel, qoz_bin, tmp_dir):
    """Run QoZ in HPEZ mode (-q4, REL error bound)."""
    n  = data.size
    dr = float(data.max() - data.min())
    eb_abs = eb_rel * dr

    tmpd  = Path(tmp_dir)
    f_in  = tmpd / 'vol.f32'
    f_cmp = tmpd / 'vol.out'
    f_dec = tmpd / 'vol_dec.f32'

    data.astype(np.float32).tofile(f_in)

    cmd = ([qoz_bin, '-f',
            '-i', str(f_in), '-z', str(f_cmp), '-o', str(f_dec),
            '-M', 'ABS', str(eb_abs), '-q', '4']
           + _dim_flags(shape))

    t0  = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_c = time.time() - t0

    if res.returncode != 0:
        print(f'    QoZ-HPEZ FAILED: {res.stderr.decode()[:200]}')
        return None

    cmp_bytes = f_cmp.stat().st_size
    t0d = time.time()
    dec  = np.fromfile(f_dec, dtype=np.float32).reshape(shape)
    t_d  = time.time() - t0d

    m = compute_metrics(data, dec, cmp_bytes)
    m.update(dict(comp_time=t_c, decomp_time=t_d,
                  eb_rel=eb_rel, eb_abs=eb_abs))
    return m


def run_sz3_interp(data, shape, eb_rel, sz3_bin, tmp_dir):
    """Run SZ3-Interp."""
    n  = data.size
    dr = float(data.max() - data.min())
    eb_abs = eb_rel * dr

    tmpd  = Path(tmp_dir)
    f_in  = tmpd / 'vol.f32'
    f_cmp = tmpd / 'vol.out'
    f_dec = tmpd / 'vol_dec.f32'

    data.astype(np.float32).tofile(f_in)

    # SZ3 syntax: -f (float32), -i <in> -z <cmp> -o <dec> -M ABS <eb> -1/-2/-3 <dims>
    cmd = ([sz3_bin, '-f',
            '-i', str(f_in), '-z', str(f_cmp), '-o', str(f_dec),
            '-M', 'ABS', str(eb_abs)]
           + _dim_flags(shape))

    t0  = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_c = time.time() - t0

    if res.returncode != 0:
        print(f'    SZ3-Interp FAILED: {res.stderr.decode()[:200]}')
        return None

    cmp_bytes = f_cmp.stat().st_size
    t0d = time.time()
    dec  = np.fromfile(f_dec, dtype=np.float32).reshape(shape)
    t_d  = time.time() - t0d

    m = compute_metrics(data, dec, cmp_bytes)
    m.update(dict(comp_time=t_c, decomp_time=t_d,
                  eb_rel=eb_rel, eb_abs=eb_abs))
    return m


def run_hybrid(data, shape, eb_rel, alg_cfg, cfg, tmp_dir, compressor_cache):
    """Run Hybrid QoZ+GP (uses HybridQoZGP class with temporal HP reuse)."""
    m_mod = _get_hybrid_module()
    alg_id = alg_cfg['id']

    if alg_id not in compressor_cache:
        temporal = cfg.get('temporal', {})
        compressor_cache[alg_id] = m_mod.HybridQoZGP(
            qoz_bin           = resolve_bin(cfg, alg_cfg['binary']),
            anchor_frac       = alg_cfg.get('anchor_frac', 0.05),
            max_step          = alg_cfg.get('max_step', 0),
            temporal_mode     = temporal.get('mode', 'fixed'),
            refit_every_t     = temporal.get('refit_every_t', 5),
            adaptive_threshold= temporal.get('adaptive_threshold', 0.05),
        )

    comp = compressor_cache[alg_id]
    tmpd = Path(tmp_dir)
    f_cmp = tmpd / 'hybrid.out'

    try:
        c_meta = comp.compress(data, eb_rel, str(f_cmp), tmp_dir)
    except Exception as e:
        print(f'    Hybrid compress FAILED: {e}')
        return None

    if 'error' in c_meta:
        print(f'    Hybrid compress FAILED: {c_meta["error"]}')
        return None

    try:
        dec, t_d = comp.decompress(str(f_cmp), shape, eb_rel, tmp_dir)
    except Exception as e:
        print(f'    Hybrid decompress FAILED: {e}')
        return None

    cmp_bytes = f_cmp.stat().st_size
    m = compute_metrics(data, dec, cmp_bytes)
    m.update(dict(comp_time=c_meta.get('comp_time', 0),
                  decomp_time=t_d,
                  eb_rel=eb_rel,
                  eb_abs=c_meta.get('eb_abs', 0),
                  n_coarse=c_meta.get('n_coarse', 0),
                  n_anchors=c_meta.get('n_anchors', 0),
                  max_step=c_meta.get('max_step', 0)))
    return m


# ────────────────────────────────────────────────────────────────────────────
# Main orchestration loop
# ────────────────────────────────────────────────────────────────────────────

def run_all(cfg: dict, results: dict, args: argparse.Namespace) -> dict:
    out_cfg   = cfg.get('output', {})
    results_path = Path(out_cfg.get('results_dir', './results')) / out_cfg.get('results_file', 'results.json')
    tmp_root  = Path(out_cfg.get('tmp_dir', '/tmp/hybrid_qoz_exp'))
    tmp_root.mkdir(parents=True, exist_ok=True)

    algs      = cfg['algorithms']
    eb_list   = cfg['error_bounds_rd']
    datasets  = cfg['datasets']

    # Filter datasets if user specified
    ds_filter = set(args.datasets) if args.datasets else None

    # Compressor instances are cached per algorithm (for temporal HP reuse)
    compressor_cache = {}

    total_done = 0
    total_skip = 0
    total_fail = 0

    for ds_id, ds_cfg in datasets.items():
        if ds_filter and ds_id not in ds_filter:
            continue

        print(f'\n{"="*60}')
        print(f'Dataset: {ds_cfg.get("label", ds_id)}')
        print(f'{"="*60}')

        for field_id, fpath, shape in iter_dataset_files(ds_cfg, ds_id):
            print(f'\n  Field: {field_id}  |  path: {fpath}')

            # Load data
            try:
                data = load_f32(fpath, shape)
            except Exception as e:
                print(f'    LOAD ERROR: {e}')
                total_fail += 1
                continue

            print(f'    shape={shape}  range=[{data.min():.3g}, {data.max():.3g}]')

            for alg_cfg in algs:
                alg_id = alg_cfg['id']

                # Per-algorithm HP cache reset between fields
                # (keep compressor instance but effectively each new field
                # starts fresh temporal state if field_id changes)
                if alg_id in compressor_cache and ds_cfg.get('is_temporal', False):
                    pass  # reuse HP across timesteps within same field
                elif alg_id in compressor_cache:
                    # reset temporal state for new independent field
                    c = compressor_cache[alg_id]
                    c._ts, c._last_refit = 0, -1
                    c._cached_hp = None
                    c._cached_anc = None

                for eb in eb_list:
                    key = result_key(ds_id, field_id, alg_id, eb)
                    if key in results:
                        print(f'    [SKIP] {alg_id}  eb={eb:.1e}')
                        total_skip += 1
                        continue

                    print(f'    [RUN]  {alg_id}  eb={eb:.1e} …', end='', flush=True)

                    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpd:
                        try:
                            alg_type = alg_cfg['type']
                            if alg_type == 'qoz':
                                m = run_qoz_hpez(data, shape, eb,
                                                 resolve_bin(cfg, alg_cfg['binary']),
                                                 tmpd)
                            elif alg_type == 'sz3':
                                m = run_sz3_interp(data, shape, eb,
                                                   resolve_bin(cfg, alg_cfg['binary']),
                                                   tmpd)
                            elif alg_type == 'hybrid_python_cpp':
                                m = run_hybrid(data, shape, eb, alg_cfg,
                                               cfg, tmpd, compressor_cache)
                            else:
                                print(f' UNKNOWN TYPE {alg_type}')
                                continue
                        except Exception as e:
                            print(f' ERROR: {e}')
                            total_fail += 1
                            continue

                    if m is None:
                        print(' FAILED')
                        total_fail += 1
                        continue

                    results[key] = {
                        'dataset': ds_id,
                        'field': field_id,
                        'algorithm': alg_id,
                        **m
                    }
                    total_done += 1
                    print(f'  CR={m["cr"]:.1f}×  PSNR={m["psnr"]:.1f} dB  '
                          f'bpv={m["bit_rate"]:.3f}')

                    # Save after every completed run
                    save_results(results, results_path)

    print(f'\n{"="*60}')
    print(f'Done.  completed={total_done}  skipped={total_skip}  failed={total_fail}')
    return results


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run Hybrid QoZ+GP compression experiments')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results.json')
    parser.add_argument('--datasets', nargs='+', metavar='DATASET',
                        help='Only run these dataset IDs (space-separated)')
    parser.add_argument('--algorithms', nargs='+', metavar='ALG',
                        help='Only run these algorithm IDs (space-separated)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Set up logging to file + stdout ──────────────────────────────────────
    out_cfg_early = cfg.get('output', {})
    log_dir = Path(out_cfg_early.get('results_dir', './results'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'run.log'

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    # Redirect print() to also go through logging
    import builtins
    _orig_print = builtins.print
    def _tee_print(*a, **kw):
        _orig_print(*a, **kw)
        msg = ' '.join(str(x) for x in a)
        logging.getLogger('exp').debug(msg)
    builtins.print = _tee_print
    print(f'Logging to {log_path}')

    # Filter algorithms if requested
    if args.algorithms:
        cfg['algorithms'] = [a for a in cfg['algorithms']
                             if a['id'] in set(args.algorithms)]

    out_cfg = cfg.get('output', {})
    results_path = (Path(out_cfg.get('results_dir', './results'))
                    / out_cfg.get('results_file', 'results.json'))

    results = {}
    if args.resume and results_path.exists():
        print(f'Resuming from {results_path}')
        results = load_results(str(results_path))
        print(f'  {len(results)} results already recorded.')

    results = run_all(cfg, results, args)
    save_results(results, str(results_path))
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
