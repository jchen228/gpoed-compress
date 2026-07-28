"""
lp_deim_compressor.py  —  libpressio external compressor: Q-DEIM

Algorithm: Q-DEIM (Drmač & Gugercin 2016).
  Offline (train): SVD of training slices → POD basis U_k → sensor placement
                   via QR-column-pivoting on U_k^T.  k is chosen by the
                   POD energy criterion:  sqrt(1 - Σs[:k]²/Σs²) ≤ error_bound
  Online (compress): store k float32 sensor observations per slice.
  Online (decompress): solve k×k system U_k[sensors,:] c = y_s  →  û = U_k c.

CLI
---
  # 1. Train (offline, once):
  python3 lp_deim_compressor.py train \\
      --data    /path/to/Uf48.bin.f32 \\
      --shape   100 500 500 \\
      --model   /path/to/deim_model.npz \\
      --error-bound 0.01 \\
      [--downsample 3] [--skip-levels 10] [--n-train-factor 4]

  # 2. Compress one slice (libpressio external compressor protocol):
  python3 lp_deim_compressor.py compress <input.bin> <output.bin> \\
      --model /path/to/deim_model.npz

  # 3. Decompress:
  python3 lp_deim_compressor.py decompress <compressed.bin> <output.bin> \\
      --model /path/to/deim_model.npz

libpressio usage
----------------
  lib  = pressio.instance()
  comp = lib.get_compressor("external")
  comp.set_options({
      "external:command":
          "python3 /path/to/lp_deim_compressor.py compress --model /path/model.npz",
      "external:decompressor_command":
          "python3 /path/to/lp_deim_compressor.py decompress --model /path/model.npz",
      "pressio:abs": 0.01,    # passed as LIBPRESSIO_ABS env var (ignored after train)
  })

Compressed binary layout (little-endian):
  4 bytes  int32   k  (number of sensors)
  4 bytes  int32   n  (flattened spatial size of original slice)
  k*4 bytes float32  sensor observations

References
----------
  Drmač & Gugercin (2016). SIAM J. Sci. Comput. 38(2), A631–A648.
"""

import argparse
import os
import struct
import sys

import numpy as np
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHM (extracted from deim_hurricane.py)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_k(singular_values, tol, max_k=None):
    """Smallest k s.t. POD projection error sqrt(1-Σs[:k]²/Σs²) ≤ tol."""
    s2    = singular_values ** 2
    cumul = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask  = proj_err <= tol
    k     = int(np.argmax(mask)) + 1 if mask.any() else len(singular_values)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


def qdeim_place(U_k):
    """QR column-pivoting on U_k^T → k sensor indices and DEIM error constant."""
    k = U_k.shape[1]
    _, _, p        = scipy_qr(U_k.T, pivoting=True)
    sensors        = p[:k]
    deim_err_const = np.linalg.norm(np.linalg.pinv(U_k[sensors, :]), ord=2)
    return sensors, deim_err_const


def qdeim_reconstruct(basis, sensors, y_meas):
    """Reconstruct full field from k sensor measurements: solve then multiply."""
    c = np.linalg.solve(basis[sensors, :], y_meas)
    return basis @ c


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(data_path, shape, model_path, error_bound,
          downsample=1, skip_levels=10, n_train_factor=4):
    """
    Build DEIM model from an ISABEL-style .bin.f32 file and save to model_path.

    Parameters
    ----------
    data_path   : str   path to raw float32 binary file
    shape       : tuple original full shape, e.g. (100, 500, 500)
    model_path  : str   where to save the .npz model
    error_bound : float POD energy truncation tolerance
    downsample  : int   spatial downsampling factor
    skip_levels : int   skip this many leading levels (near-surface artefacts)
    n_train_factor : int  n_train = n_train_factor × k (capped to leave ≥10 test)
    """
    n_levels_full, ny_full, nz_full = shape
    print(f"[DEIM train]  Loading {data_path}  shape={shape}  ds={downsample}")
    with open(data_path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape(shape)[:, ::downsample, ::downsample]   # (n_levels, ny, nz)

    data = data[skip_levels:]                                  # drop near-surface
    n_levels, ny, nz = data.shape
    n = ny * nz

    # ── Pilot SVD to choose k and set n_train ────────────────────────────────
    pilot_mean = data.mean(axis=0)
    X_pilot    = (data - pilot_mean).reshape(n_levels, n).T
    _, s_pilot, _ = np.linalg.svd(X_pilot, full_matrices=False)
    max_sensors = max(1, n // 66)
    k_pilot, _ = adaptive_k(s_pilot, error_bound, max_sensors)

    n_train = n_train_factor * k_pilot
    n_train = min(n_train, n_levels - 10)
    n_train = max(n_train, k_pilot + 2)

    # ── Random train/test split ───────────────────────────────────────────────
    rng        = np.random.default_rng(42)
    train_idx  = np.sort(rng.choice(n_levels, size=n_train, replace=False))
    test_mask  = np.ones(n_levels, dtype=bool)
    test_mask[train_idx] = False
    test_idx   = np.where(test_mask)[0]

    train_data = data[train_idx]
    train_mean = train_data.mean(axis=0)   # (ny, nz)
    train_c    = train_data - train_mean
    X          = train_c.reshape(n_train, n).T   # (n, n_train)

    # ── Offline: SVD → adaptive k → Q-DEIM sensor placement ─────────────────
    U, s_vals, _ = np.linalg.svd(X, full_matrices=False)
    k, proj_err  = adaptive_k(s_vals, error_bound, max_sensors)
    U_k          = U[:, :k]
    sensors, deim_const = qdeim_place(U_k)
    energy = (s_vals[:k] ** 2).sum() / (s_vals ** 2).sum()

    print(f"  k={k}  proj_err={proj_err[k-1]*100:.3f}%  "
          f"energy={energy*100:.2f}%  DEIM_const={deim_const:.4f}")
    print(f"  n={n}  n_train={n_train}  sensors_cap={max_sensors}")

    # ── Evaluate on held-out test levels ─────────────────────────────────────
    test_data = data[test_idx]
    test_c    = test_data - train_mean
    errs      = []
    for lvl_c in test_c:
        y_flat  = lvl_c.ravel()
        y_hat_c = qdeim_reconstruct(U_k, sensors, y_flat[sensors])
        diff    = (y_hat_c.reshape(ny, nz) + train_mean) - (lvl_c.reshape(ny, nz) + train_mean)
        errs.append(float(np.linalg.norm(diff) / np.linalg.norm(lvl_c.reshape(ny, nz) + train_mean)))
    errs = np.array(errs)
    print(f"  Test rel-L2:  min={errs.min():.4f}  median={np.median(errs):.4f}  "
          f"max={errs.max():.4f}")

    np.savez_compressed(model_path,
                        basis=U_k.astype(np.float32),
                        sensors=sensors.astype(np.int32),
                        train_mean=train_mean.astype(np.float32),
                        k=np.int32(k),
                        n=np.int32(n),
                        ny=np.int32(ny),
                        nz=np.int32(nz),
                        error_bound=np.float32(error_bound),
                        deim_const=np.float32(deim_const),
                        test_rel_l2_median=np.float32(np.median(errs)))
    print(f"  Model saved → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def compress(input_path, output_path, model_path):
    """
    Compress one slice (float32 binary) to sensor observations.

    Input : raw float32 binary, n elements.
    Output: binary header (k, n as int32) + k float32 sensor values.
    """
    m = np.load(model_path)
    basis      = m['basis'].astype(np.float64)     # (n, k)
    sensors    = m['sensors'].astype(np.int64)      # (k,)
    train_mean = m['train_mean'].astype(np.float64) # (ny, nz) or flat (n,)
    k  = int(m['k'])
    n  = int(m['n'])
    ny = int(m['ny'])
    nz = int(m['nz'])

    data = np.fromfile(input_path, dtype=np.float32).astype(np.float64)
    if data.size != n:
        raise ValueError(f"Input has {data.size} elements; model expects {n}.")

    # Centre and extract sensor values
    mean_flat   = train_mean.ravel()
    data_c      = data - mean_flat
    y_sensors   = data_c[sensors].astype(np.float32)

    # Write compressed: header (k, n) + sensor values
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<ii', k, n))
        f.write(y_sensors.tobytes())

    ratio = (n * 4) / (8 + k * 4)
    print(f"  Compressed: {n}→{k} values  ratio={ratio:.1f}×", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def decompress(compressed_path, output_path, model_path):
    """
    Reconstruct full slice from compressed sensor observations.

    Reads the header (k, n) + k float32 values, solves the DEIM system,
    adds training mean, writes n float32 values.
    """
    m = np.load(model_path)
    basis      = m['basis'].astype(np.float64)
    sensors    = m['sensors'].astype(np.int64)
    train_mean = m['train_mean'].astype(np.float64)
    ny = int(m['ny'])
    nz = int(m['nz'])

    with open(compressed_path, 'rb') as f:
        k, n = struct.unpack('<ii', f.read(8))
        y_sensors = np.frombuffer(f.read(k * 4), dtype=np.float32).astype(np.float64)

    recon_c  = qdeim_reconstruct(basis, sensors, y_sensors)   # (n,) centred
    recon    = (recon_c + train_mean.ravel()).astype(np.float32)
    recon.tofile(output_path)
    print(f"  Decompressed: {k}→{n} values", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="libpressio external compressor: Q-DEIM")
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Build and save DEIM model')
    p_train.add_argument('--data',           required=True)
    p_train.add_argument('--shape',          required=True, nargs='+', type=int,
                         metavar='DIM', help='Full volume shape, e.g. 100 500 500')
    p_train.add_argument('--model',          required=True)
    p_train.add_argument('--error-bound',    type=float, default=0.01)
    p_train.add_argument('--downsample',     type=int,   default=1)
    p_train.add_argument('--skip-levels',    type=int,   default=10)
    p_train.add_argument('--n-train-factor', type=int,   default=4)

    # ── compress ──────────────────────────────────────────────────────────────
    p_comp = sub.add_parser('compress', help='Compress one slice')
    p_comp.add_argument('input')
    p_comp.add_argument('output')
    p_comp.add_argument('--model', required=True)

    # ── decompress ────────────────────────────────────────────────────────────
    p_dec = sub.add_parser('decompress', help='Decompress one slice')
    p_dec.add_argument('input')
    p_dec.add_argument('output')
    p_dec.add_argument('--model', required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data, tuple(args.shape), args.model, args.error_bound,
              args.downsample, args.skip_levels, args.n_train_factor)
    elif args.mode == 'compress':
        compress(args.input, args.output, args.model)
    elif args.mode == 'decompress':
        decompress(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
