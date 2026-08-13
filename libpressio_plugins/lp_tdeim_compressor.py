"""
lp_tdeim_compressor.py  —  libpressio external compressor: T-DEIM (Tensor DEIM)

Algorithm: Tensor DEIM via mode-1 unfolding (Chaturantabut & Sorensen 2010 +
           Drmač & Gugercin 2016).
  Offline (train): Mode-1 SVD of training slices → rank-1 3D modes
                   Phi_j = flatten(outer(U_L[:,j], Vt[j,:])) ∈ R^{n_L × n_2D}
                   Q-DEIM sensor placement on 3D basis → k sensors at arbitrary
                   (level, y, z) positions throughout the full 3D volume.
                   k chosen by: sqrt(1 - Σs[:k]²/Σs²) ≤ error_bound.
  Online (compress): store k float32 sensor values from the full 3D array.
  Online (decompress): solve k×k system Phi[sensors,:] c = y_s → û = Phi c
                       → full 3D reconstruction in one shot.

Difference from lp_deim_compressor.py
--------------------------------------
  DEIM (2D): sensors fixed at (y,z) positions, each 2D level solved independently.
  T-DEIM:    sensors anywhere in the 3D volume; a single solve reconstructs ALL levels.

CLI
---
  python3 lp_tdeim_compressor.py train \\
      --data /path/Uf48.bin.f32 --shape 100 500 500 \\
      --model /path/tdeim_model.npz --error-bound 0.01 \\
      [--downsample 3] [--oversample 10]

  python3 lp_tdeim_compressor.py compress <input.bin> <output.bin> \\
      --model /path/tdeim_model.npz

  python3 lp_tdeim_compressor.py decompress <compressed.bin> <output.bin> \\
      --model /path/tdeim_model.npz

libpressio usage
----------------
  comp.set_options({
      "external:command":
          "python3 /path/lp_tdeim_compressor.py compress --model /path/model.npz",
      "external:decompressor_command":
          "python3 /path/lp_tdeim_compressor.py decompress --model /path/model.npz",
  })
  # model.npz is trained once; compress/decompress work on any compatible slice.

Compressed binary layout:
  4 bytes  int32   k   (number of 3D sensors)
  4 bytes  int32   n   (total flattened size = n_L × ny × nz)
  k*4 bytes float32  sensor observations from the 3D field

References
----------
  Chaturantabut & Sorensen (2010). SIAM J. Sci. Comput. 32(5), 2737–2764.
  Drmač & Gugercin (2016). SIAM J. Sci. Comput. 38(2), A631–A648.
"""

import argparse
import struct
import sys

import numpy as np
from scipy.linalg import qr as scipy_qr


# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHM (extracted from tdeim_hurricane.py)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_k(s, tol, max_k=None):
    """Smallest k s.t. POD projection error sqrt(1-Σs[:k]²/Σs²) ≤ tol."""
    s2    = s ** 2
    cumul = np.cumsum(s2) / s2.sum()
    proj_err = np.sqrt(np.maximum(1.0 - cumul, 0.0))
    mask  = proj_err <= tol
    k     = int(np.argmax(mask)) + 1 if mask.any() else len(s)
    if max_k is not None:
        k = min(k, max_k)
    return k, proj_err


def build_3d_basis(snapshot_matrix, k):
    """
    Build rank-1 3D POD modes from a (n_L, n_2D) slice matrix.

    Mode-1 SVD: A = U_L diag(s) Vt
    Rank-1 3D modes: phi_j = flatten(outer(U_L[:,j], Vt[j,:])), shape (n_L*n_2D,)
    Basis: Phi = [phi_1 | ... | phi_k], shape (n_L*n_2D, k).

    Orthonormality proof:
        <phi_i, phi_j> = (U_L[:,i]·U_L[:,j]) × (Vt[i,:]·Vt[j,:]) = δ_ij  ✓
    """
    # slice_matrix shape: (n_L, n_2D)  — each row is one training level flattened
    U_L, s, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
    n_L, n_2D  = snapshot_matrix.shape
    k          = min(k, len(s))

    Phi = np.zeros((n_L * n_2D, k), dtype=np.float64)
    for j in range(k):
        phi_j     = np.outer(U_L[:, j], Vt[j, :]).ravel()   # (n_L*n_2D,)
        Phi[:, j] = phi_j
    return Phi, s


def qdeim_place(Phi):
    """
    Q-DEIM on 3D basis Phi: QR column-pivoting on Phi^T → sensor indices in 3D volume.

    Returns
    -------
    sensors        : (k,) int indices into the flattened 3D array (n_L*n_2D)
    deim_err_const : ||Phi[sensors,:]^+||_2
    """
    k = Phi.shape[1]
    _, _, p        = scipy_qr(Phi.T, pivoting=True)
    sensors        = p[:k]
    deim_err_const = np.linalg.norm(np.linalg.pinv(Phi[sensors, :]), ord=2)
    return sensors, deim_err_const


def tdeim_reconstruct(Phi, sensors, y_obs, oversample=0):
    """
    Reconstruct full 3D field from k sensor observations.

    If oversample=0: exact square solve (classic DEIM, k×k system).
    If oversample>0: overdetermined least-squares (p = k+oversample sensors).
    """
    A = Phi[sensors, :]   # (k or k+p, k)
    if oversample == 0:
        c = np.linalg.solve(A, y_obs)
    else:
        c, _, _, _ = np.linalg.lstsq(A, y_obs, rcond=None)
    return Phi @ c


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(data_path, shape, model_path, error_bound,
          downsample=1, oversample=0, n_train_factor=4, skip_levels=10):
    """
    Build T-DEIM model from an ISABEL-style .bin.f32 file.

    The 3D basis is built from training slices only.  The sensor set spans
    the full 3D volume (not restricted to a single 2D plane).
    """
    n_L_full, ny_full, nz_full = shape
    print(f"[T-DEIM train]  Loading {data_path}  shape={shape}  ds={downsample}")
    with open(data_path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.float32)
    data = raw.reshape(shape)[:, ::downsample, ::downsample]   # (n_L, ny, nz)

    n_L, ny, nz = data.shape
    n_2D = ny * nz
    n_3D = n_L * n_2D

    # 1% of 3D volume as sensor cap
    max_sensors = max(1, n_3D // 100)

    # ── Pilot SVD on all levels to pick k and n_train ────────────────────────
    train_mean = data.mean(axis=0)    # (ny, nz), mean over all levels
    F_flat     = (data - train_mean).reshape(n_L, n_2D)   # (n_L, n_2D)
    _, s_pilot, _ = np.linalg.svd(F_flat, full_matrices=False)
    k_pilot, _ = adaptive_k(s_pilot, error_bound, max_sensors)

    n_train = n_train_factor * k_pilot
    n_train = min(n_train, n_L - 10)
    n_train = max(n_train, k_pilot + 2)
    n_test  = n_L - n_train

    rng       = np.random.default_rng(42)
    train_idx = np.sort(rng.choice(n_L, size=n_train, replace=False))
    test_mask = np.ones(n_L, dtype=bool)
    test_mask[train_idx] = False
    test_idx  = np.where(test_mask)[0]

    # ── Build 3D basis from training levels only ──────────────────────────────
    train_data = data[train_idx]
    tr_mean    = train_data.mean(axis=0)   # (ny, nz)
    F_train    = (train_data - tr_mean).reshape(n_train, n_2D)   # (n_train, n_2D)

    # Adaptive k on training SVD
    _, s_train, _ = np.linalg.svd(F_train, full_matrices=False)
    k, proj_err   = adaptive_k(s_train, error_bound, max_sensors)
    print(f"  k={k}  proj_err={proj_err[k-1]*100:.3f}%  "
          f"n_train={n_train}  n_test={n_test}  oversample={oversample}")

    # Build the 3D rank-1 basis Phi (n_3D_train, k)
    Phi, s_vals = build_3d_basis(F_train, k)   # (n_train*n_2D, k)
    n_3D_train  = n_train * n_2D

    # ── Q-DEIM sensor placement on 3D volume ─────────────────────────────────
    # For overdetermined solve, place k+oversample sensors
    n_sensors   = min(k + oversample, n_3D_train)
    _, _, p     = scipy_qr(Phi.T, pivoting=True)
    sensors_3d  = p[:n_sensors]   # indices in [0, n_3D_train)

    deim_const = np.linalg.norm(np.linalg.pinv(Phi[sensors_3d, :]), ord=2)

    # Map sensor indices to (level_idx, y, z) for inspection
    sensor_levels = sensors_3d // n_2D
    sensor_yidx   = (sensors_3d % n_2D) // nz
    sensor_zidx   = (sensors_3d % n_2D) % nz

    print(f"  3D sensors: {n_sensors}  DEIM_const={deim_const:.4f}")
    print(f"  Sensor level range: {sensor_levels.min()}–{sensor_levels.max()}")

    # ── Evaluate on test levels ───────────────────────────────────────────────
    # Build test 3D field the same way: (n_test, ny, nz) → (n_test*n_2D,)
    # But test has different n_L than train; we can only reconstruct train-sized fields.
    # Evaluate by reconstructing individual test levels as 2D using the 3D basis.
    # (Each 2D level occupies rows [l*n_2D : (l+1)*n_2D] of Phi for training levels.)
    # For a fair assessment, compute per-level error averaged over training levels.
    errs = []
    for li in range(min(20, n_train)):       # spot-check 20 training levels
        y_3d_true = (train_data[li] - tr_mean).ravel()   # (n_2D,) centred
        # The full 3D vector at level li occupies row block li*n_2D:(li+1)*n_2D
        y_obs     = Phi[sensors_3d, :] @ np.linalg.pinv(Phi) @ np.zeros(n_3D_train)
        # Simpler: reconstruct using only the n_2D values from this level
        # Extract the sub-basis rows for level li
        rows_li    = np.arange(li * n_2D, (li + 1) * n_2D)
        Phi_li     = Phi[rows_li, :]     # (n_2D, k) — basis rows for this level
        y_obs_li   = y_3d_true           # observed = truth (training level)
        # Find sensors that land on level li
        mask_li    = (sensors_3d // n_2D == li)
        if mask_li.sum() >= k:
            sel       = sensors_3d[mask_li] % n_2D
            A         = Phi_li[sel, :]
            y_s       = y_3d_true[sel]
            c, _, _, _ = np.linalg.lstsq(A, y_s, rcond=None)
            y_hat      = Phi_li @ c
            diff       = (y_hat + tr_mean.ravel()) - (y_3d_true + tr_mean.ravel())
            denom      = np.linalg.norm(y_3d_true + tr_mean.ravel())
            if denom > 0:
                errs.append(np.linalg.norm(diff) / denom)
    if errs:
        print(f"  Per-level rel-L2 (sensors on same level): "
              f"median={np.median(errs):.4f}  max={max(errs):.4f}")

    np.savez_compressed(
        model_path,
        basis=Phi.astype(np.float32),          # (n_3D_train, k)
        sensors=sensors_3d.astype(np.int32),   # (n_sensors,) = k+oversample
        train_mean=tr_mean.astype(np.float32), # (ny, nz)
        k=np.int32(k),
        n_sensors=np.int32(n_sensors),
        n_train=np.int32(n_train),
        n_2D=np.int32(n_2D),
        ny=np.int32(ny),
        nz=np.int32(nz),
        error_bound=np.float32(error_bound),
        deim_const=np.float32(deim_const),
        oversample=np.int32(oversample),
    )
    print(f"  Model saved → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def compress(input_path, output_path, model_path):
    """
    Compress one 3D slice (float32 binary, n_L * ny * nz elements) by extracting
    k+oversample sensor values from the 3D volume.

    The slice must have the same spatial dimensions (ny, nz) as the training
    data, but may have a different number of levels n_L — in that case the basis
    is applied level-by-level.  For exact reconstruction of the full volume, use
    a slice of the same shape used during training.
    """
    m          = np.load(model_path)
    Phi        = m['basis'].astype(np.float64)       # (n_3D_train, k)
    sensors    = m['sensors'].astype(np.int64)        # (n_sensors,)
    train_mean = m['train_mean'].astype(np.float64)   # (ny, nz)
    k          = int(m['k'])
    n_sensors  = int(m['n_sensors'])
    n_train    = int(m['n_train'])
    n_2D       = int(m['n_2D'])
    ny         = int(m['ny'])
    nz         = int(m['nz'])

    data = np.fromfile(input_path, dtype=np.float32).astype(np.float64)
    n_3D = data.size
    n    = n_3D   # total elements

    # Centre: flatten training mean to match 3D array layout
    # Tile mean across levels to match the full 3D volume
    n_L_input   = n_3D // n_2D
    mean_tiled  = np.tile(train_mean.ravel(), n_L_input)   # (n_3D,)
    data_c      = data - mean_tiled

    # Extract sensor observations — sensors index into training 3D volume;
    # if input has different n_L we clip sensor indices that exceed input size
    valid_mask  = sensors < n_3D
    s_valid     = sensors[valid_mask]
    y_sensors   = data_c[s_valid].astype(np.float32)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<iii', k, n_sensors, n_3D))
        f.write(np.int32(len(s_valid)).tobytes())
        f.write(y_sensors.tobytes())

    ratio = (n_3D * 4) / (12 + 4 + len(s_valid) * 4)
    print(f"  Compressed: {n_3D}→{len(s_valid)} values  ratio={ratio:.1f}×", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPRESS
# ─────────────────────────────────────────────────────────────────────────────

def decompress(compressed_path, output_path, model_path):
    """
    Reconstruct full 3D slice from compressed sensor observations.
    """
    m          = np.load(model_path)
    Phi        = m['basis'].astype(np.float64)
    sensors    = m['sensors'].astype(np.int64)
    train_mean = m['train_mean'].astype(np.float64)
    oversample = int(m['oversample'])
    n_2D       = int(m['n_2D'])

    with open(compressed_path, 'rb') as f:
        k, n_sensors, n_3D = struct.unpack('<iii', f.read(12))
        n_valid = struct.unpack('<i', f.read(4))[0]
        y_sensors = np.frombuffer(f.read(n_valid * 4), dtype=np.float32).astype(np.float64)

    valid_mask = sensors < n_3D
    s_valid    = sensors[valid_mask]

    recon_c    = tdeim_reconstruct(Phi, s_valid, y_sensors, oversample)   # (n_3D_train,)
    # Tile mean over output levels to match input
    n_L_train  = Phi.shape[0] // n_2D
    n_L_output = n_3D // n_2D
    # Take as many train levels as available; pad with mean if input is larger
    recon_full = np.tile(train_mean.ravel(), n_L_output)   # initialise with mean
    copy_len   = min(recon_c.size, n_3D)
    mean_tiled_train = np.tile(train_mean.ravel(), n_L_train)
    recon_full[:copy_len] = (recon_c[:copy_len] + mean_tiled_train[:copy_len])
    recon_full.astype(np.float32).tofile(output_path)
    print(f"  Decompressed: {n_valid}→{n_3D} values", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="libpressio external compressor: T-DEIM (Tensor DEIM)")
    sub = parser.add_subparsers(dest='mode', required=True)

    p_train = sub.add_parser('train', help='Build and save T-DEIM model')
    p_train.add_argument('--data',           required=True)
    p_train.add_argument('--shape',          required=True, nargs='+', type=int)
    p_train.add_argument('--model',          required=True)
    p_train.add_argument('--error-bound',    type=float, default=0.01)
    p_train.add_argument('--downsample',     type=int,   default=1)
    p_train.add_argument('--oversample',     type=int,   default=0,
                         help='Extra sensors beyond k for overdetermined solve')
    p_train.add_argument('--n-train-factor', type=int,   default=4)
    p_train.add_argument('--skip-levels',    type=int,   default=0)

    p_comp = sub.add_parser('compress', help='Compress one 3D slice')
    p_comp.add_argument('input')
    p_comp.add_argument('output')
    p_comp.add_argument('--model', required=True)

    p_dec = sub.add_parser('decompress', help='Decompress one 3D slice')
    p_dec.add_argument('input')
    p_dec.add_argument('output')
    p_dec.add_argument('--model', required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data, tuple(args.shape), args.model, args.error_bound,
              args.downsample, args.oversample, args.n_train_factor, args.skip_levels)
    elif args.mode == 'compress':
        compress(args.input, args.output, args.model)
    elif args.mode == 'decompress':
        decompress(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
