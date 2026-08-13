"""
lorenzo_vs_global_diagnostic.py
================================
Isolates the prediction step and compares residual distributions for:

  Lorenzo    — SZ2's local 2-D linear finite-difference predictor
  DEIM       — global SVD-based predictor (from saved checkpoint)
  GP         — global Matérn-3/2 GP predictor (from saved checkpoint)

No compression is run.  Requires only the SST data file and the two
checkpoint files produced by sst_rd_comparison.py.

Output
------
  lorenzo_vs_global_residuals.png  — overlaid histograms + violin plots
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, qr

# ── Configuration ──────────────────────────────────────────────────────────────
ARGONNE   = Path(__file__).resolve().parent.parent
DATA_PATH = ARGONNE.parent / "gpoed-code-python" / "sst.wkmean.1990-present.nc"

N_SNAPSHOTS = 10          # number of snapshots to analyse (set to 1 for quick run)
DEIM_K      = 1600         # DEIM sensor / mode count
GP_K        = 2000         # GP sensor count

FIELD_TAG = "SST"
N_TRAIN   = 1727
DPI       = 150

# ── Load SST data ──────────────────────────────────────────────────────────────
print("Loading SST data ...")
import netCDF4 as nc4
with nc4.Dataset(str(DATA_PATH)) as ds:
    raw = ds.variables["sst"][:]
data = np.ma.filled(raw, fill_value=0.0).astype(np.float32)
n_T, ny, nx = data.shape

# Use the companion land/sea mask — must match what sst_rd_comparison.py used
# when building the DEIM/GP checkpoints.
mask_path = DATA_PATH.parent / "lsmask.nc"
with nc4.Dataset(str(mask_path)) as ds2:
    raw_mask   = np.array(ds2.variables["mask"][0], dtype=np.int16)  # (180, 360)
ocean_mask = (raw_mask == 1)              # True = ocean (44 219 pixels)
ocean_idx  = np.where(ocean_mask.ravel())[0]
n_ocean    = len(ocean_idx)

# Pick N_SNAPSHOTS evenly spaced snapshots from the middle of the series
snap_indices = np.round(np.linspace(n_T // 4, 3 * n_T // 4, N_SNAPSHOTS)).astype(int)
snap_indices = np.unique(np.clip(snap_indices, 0, n_T - 1))
print(f"  Snapshot indices: {snap_indices.tolist()}")
print(f"  n_T={n_T}  ny={ny}  nx={nx}  n_ocean={n_ocean}")

# ── Lorenzo 2-D linear predictor ───────────────────────────────────────────────
def lorenzo_predict_2d(field: np.ndarray) -> np.ndarray:
    """
    2-D linear Lorenzo predictor (the dominant mode used in SZ2).

        pred[i,j] = field[i-1,j] + field[i,j-1] - field[i-1,j-1]

    Boundary rows/columns are left as the original value (residual = 0 there).
    Returns the residual array (field − pred) restricted to interior pixels.
    """
    pred = field.copy()
    pred[1:, 1:] = field[:-1, 1:] + field[1:, :-1] - field[:-1, :-1]
    resid = field - pred
    # Zero out boundary (no prediction there)
    resid[0,  :] = 0.0
    resid[:,  0] = 0.0
    return resid


def lorenzo_predict_1d(strip: np.ndarray) -> np.ndarray:
    """
    1-D Lorenzo predictor applied to the flat ocean-only pixel strip.
    This is exactly what SZ2-1D's internal predictor does: each pixel is
    predicted from its immediate predecessor in the flattened sequence.

        pred[0]   = strip[0]   (no predecessor → residual = 0)
        pred[i]   = strip[i-1] for i > 0

    Returns the residual array (strip − pred).
    """
    resid    = np.empty_like(strip)
    resid[0] = 0.0
    resid[1:] = strip[1:] - strip[:-1]
    return resid

# ── Load DEIM checkpoint ───────────────────────────────────────────────────────
_deim_ckpt = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
if not _deim_ckpt.exists():
    raise FileNotFoundError(f"DEIM checkpoint not found: {_deim_ckpt}\n"
                            "Run sst_rd_comparison.py first.")
print(f"\nLoading DEIM checkpoint: {_deim_ckpt.name} ...")
_ck        = np.load(_deim_ckpt)
Phi_max    = _ck["Phi_max"].astype(np.float64)   # (n_ocean, k_max)
mean_ocean = _ck["mean_ocean"].astype(np.float64)
k_deim     = min(DEIM_K, Phi_max.shape[1])
Phi_k      = Phi_max[:, :k_deim]
_, _, piv  = qr(Phi_k.T, pivoting=True)
sensors_d  = piv[:k_deim]
A_d        = Phi_k[sensors_d, :]                  # (k, k)
M_d        = Phi_k @ np.linalg.inv(A_d)           # (n_ocean, k) DEIM operator
print(f"  DEIM: k={k_deim}  Phi_k {Phi_k.shape}")

# ── Load GP checkpoint ─────────────────────────────────────────────────────────
k_gp_max   = 2000
_krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N{k_gp_max}.npz"
if not _krig_ckpt.exists():
    _krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N1000.npz"
if not _krig_ckpt.exists():
    raise FileNotFoundError(f"GP checkpoint not found — run sst_rd_comparison.py first.")
print(f"Loading GP checkpoint: {_krig_ckpt.name} ...")
_ck2          = np.load(_krig_ckpt)
ls            = float(_ck2["ls"])
var_gp        = float(_ck2["var"])
noise_var     = float(_ck2["noise_var"])
sensors_g     = _ck2["sensors"]
train_mean    = _ck2["train_mean"].astype(np.float64)
train_std_safe= _ck2["train_std_safe"].astype(np.float64)
k_gp          = min(GP_K, len(sensors_g))
sensors_g     = sensors_g[:k_gp]
print(f"  GP: k={k_gp}  ls={ls:.4f}  var={var_gp:.4f}")

# Build GP kernel matrices (once — reused for all snapshots)
print("  Building GP kernel matrices ...")

def matern32(X1, X2, ls):
    d = np.sqrt(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1))
    r = np.sqrt(3) * d / ls
    return (1.0 + r) * np.exp(-r)

# Normalised grid coordinates
lat_n = np.linspace(0, 1, ny)
lon_n = np.linspace(0, 1, nx)
LON_G, LAT_G = np.meshgrid(lon_n, lat_n)
X_full = np.column_stack([LAT_G.ravel(), LON_G.ravel()])
X_all  = X_full[ocean_idx]                        # (n_ocean, 2)
X_sens = X_all[sensors_g]                         # (k_gp, 2)

# Checkpoint stores the BASE kernel (no var_gp factor); var_gp is applied in K_sub and alpha.
# This matches how run_kriging_2d uses B[0,0]*cho_solve with the base-kernel Cholesky.
if "K_Xs_max" in _ck2.files:
    K_Xs = _ck2["K_Xs_max"][:, :k_gp].astype(np.float64)   # base kernel, no var_gp
    K_ss = _ck2["K_ss_max"][:k_gp, :k_gp].astype(np.float64)   # base kernel, no var_gp
else:
    K_Xs = matern32(X_all, X_sens, ls)   # base kernel (no var_gp) to match checkpoint convention
    K_ss = matern32(X_sens, X_sens, ls)

# K_sub = var_gp * K_base_ss + noise I  (correct GP covariance at sensor locations)
K_sub    = var_gp * K_ss + noise_var * np.eye(k_gp)
L_gp, lo = cho_factor(K_sub, lower=True)
ms_k     = train_mean[sensors_g]
ss_k     = train_std_safe[sensors_g]
print("  GP matrices ready.")

# ── Compute residuals for each snapshot ────────────────────────────────────────
all_resid_lor   = []   # list of 1-D ocean residual arrays
all_resid_lor1d = []   # 1-D Lorenzo on ocean-only strip (what SZ2-1D uses)
all_resid_deim  = []
all_resid_gp    = []

print(f"\nComputing residuals over {len(snap_indices)} snapshots ...")
for t in snap_indices:
    snap_full  = data[t]                                    # (ny, nx)
    snap_ocean = snap_full.ravel()[ocean_idx].astype(np.float64)  # (n_ocean,)

    # ── Lorenzo 2-D (full 2-D field, then restrict to ocean) ─────────────────
    resid_lor_2d = lorenzo_predict_2d(snap_full.astype(np.float64))
    resid_lor    = resid_lor_2d.ravel()[ocean_idx]          # ocean pixels only

    # ── Lorenzo 1-D (ocean-only strip — what SZ2-1D's predictor does) ────────
    resid_lor1d  = lorenzo_predict_1d(snap_ocean)

    # ── DEIM ─────────────────────────────────────────────────────────────────
    anom_ocean    = snap_ocean - mean_ocean
    sv_d          = anom_ocean[sensors_d]
    pred_anom     = M_d @ sv_d                              # (n_ocean,)
    resid_deim    = anom_ocean - pred_anom

    # ── GP ───────────────────────────────────────────────────────────────────
    y_norm        = (snap_ocean[sensors_g] - ms_k) / ss_k
    alpha         = var_gp * cho_solve((L_gp, lo), y_norm)
    mu_norm       = K_Xs @ alpha
    mu            = mu_norm * train_std_safe + train_mean
    resid_gp      = snap_ocean - mu

    all_resid_lor.append(resid_lor.astype(np.float32))
    all_resid_lor1d.append(resid_lor1d.astype(np.float32))
    all_resid_deim.append(resid_deim.astype(np.float32))
    all_resid_gp.append(resid_gp.astype(np.float32))

    rmse_l  = np.sqrt(np.mean(resid_lor**2))
    rmse_l1 = np.sqrt(np.mean(resid_lor1d**2))
    rmse_d  = np.sqrt(np.mean(resid_deim**2))
    rmse_g  = np.sqrt(np.mean(resid_gp**2))
    print(f"  t={t:4d}  Lorenzo-2D={rmse_l:.4f}  Lorenzo-1D={rmse_l1:.4f}  "
          f"DEIM(k={k_deim})={rmse_d:.4f}  GP(k={k_gp})={rmse_g:.4f}")

# Pool across all snapshots for aggregate distribution
pool_lor   = np.concatenate(all_resid_lor)
pool_lor1d = np.concatenate(all_resid_lor1d)
pool_deim  = np.concatenate(all_resid_deim)
pool_gp    = np.concatenate(all_resid_gp)

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting ...")

C_LOR   = "#ff7f0e"   # orange  — Lorenzo 2-D
C_LOR1D = "#d62728"   # red     — Lorenzo 1-D (SZ2-1D predictor)
C_DEIM  = "#17becf"   # cyan
C_GP    = "#e377c2"   # pink

fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

ax_hist  = fig.add_subplot(gs[0, :])   # full-width histogram
ax_viol  = fig.add_subplot(gs[1, 0])   # violin per method
ax_rmse  = fig.add_subplot(gs[1, 1])   # per-snapshot RMSE

# ── (1) Overlaid histograms ───────────────────────────────────────────────────
clip = np.percentile(np.abs(pool_lor), 99.5)
bins = np.linspace(-clip, clip, 150)

ax_hist.hist(pool_lor,   bins=bins, density=True, alpha=0.50,
             color=C_LOR,   label="Lorenzo 2-D (SZ2 predictor)")
ax_hist.hist(pool_lor1d, bins=bins, density=True, alpha=0.50,
             color=C_LOR1D, label="Lorenzo 1-D (SZ2-1D predictor)")
ax_hist.hist(pool_deim,  bins=bins, density=True, alpha=0.50,
             color=C_DEIM,  label=f"DEIM  k={k_deim}")
ax_hist.hist(pool_gp,    bins=bins, density=True, alpha=0.50,
             color=C_GP,    label=f"GP    k={k_gp}")

# Overlay std annotations
for arr, col in [(pool_lor,   C_LOR),
                 (pool_lor1d, C_LOR1D),
                 (pool_deim,  C_DEIM),
                 (pool_gp,    C_GP)]:
    s = np.std(arr)
    ax_hist.axvline( s, color=col, lw=1.2, ls="--", alpha=0.7)
    ax_hist.axvline(-s, color=col, lw=1.2, ls="--", alpha=0.7)

ax_hist.set_xlabel("Residual value (°C)", fontsize=12)
ax_hist.set_ylabel("Probability density", fontsize=12)
ax_hist.set_title(
    f"Prediction residual distributions  —  {N_SNAPSHOTS} SST snapshots pooled\n"
    f"(dashed lines = ±1 std)",
    fontsize=12)
ax_hist.legend(fontsize=11)
ax_hist.grid(True, alpha=0.25)

# ── (2) Violin plot ───────────────────────────────────────────────────────────
# Subsample for violin (full pool is large)
rng = np.random.default_rng(0)
n_viol = min(50_000, len(pool_lor))
vdata = [
    rng.choice(pool_lor,   n_viol, replace=False),
    rng.choice(pool_lor1d, n_viol, replace=False),
    rng.choice(pool_deim,  n_viol, replace=False),
    rng.choice(pool_gp,    n_viol, replace=False),
]
parts = ax_viol.violinplot(vdata, positions=[1, 2, 3, 4],
                            showmedians=True, showextrema=False)
for pc, col in zip(parts["bodies"], [C_LOR, C_LOR1D, C_DEIM, C_GP]):
    pc.set_facecolor(col); pc.set_alpha(0.6)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
ax_viol.set_xticks([1, 2, 3, 4])
ax_viol.set_xticklabels(["Lorenzo\n2-D", "Lorenzo\n1-D",
                          f"DEIM\nk={k_deim}", f"GP\nk={k_gp}"],
                         fontsize=10)
ax_viol.set_ylabel("Residual (°C)", fontsize=11)
ax_viol.set_title("Residual distribution shape", fontsize=11)
ax_viol.axhline(0, color="k", lw=0.8, ls="--")
ax_viol.grid(True, axis="y", alpha=0.25)

# ── (3) Per-snapshot RMSE ─────────────────────────────────────────────────────
rmse_lor   = [np.sqrt(np.mean(r**2)) for r in all_resid_lor]
rmse_lor1d = [np.sqrt(np.mean(r**2)) for r in all_resid_lor1d]
rmse_deim  = [np.sqrt(np.mean(r**2)) for r in all_resid_deim]
rmse_gp    = [np.sqrt(np.mean(r**2)) for r in all_resid_gp]

xs = np.arange(len(snap_indices))
w  = 0.20
ax_rmse.bar(xs - 1.5*w, rmse_lor,   width=w, color=C_LOR,   alpha=0.8, label="Lorenzo 2-D")
ax_rmse.bar(xs - 0.5*w, rmse_lor1d, width=w, color=C_LOR1D, alpha=0.8, label="Lorenzo 1-D")
ax_rmse.bar(xs + 0.5*w, rmse_deim,  width=w, color=C_DEIM,  alpha=0.8, label=f"DEIM k={k_deim}")
ax_rmse.bar(xs + 1.5*w, rmse_gp,    width=w, color=C_GP,    alpha=0.8, label=f"GP k={k_gp}")
ax_rmse.set_xticks(xs)
ax_rmse.set_xticklabels([f"t={t}" for t in snap_indices],
                         fontsize=7, rotation=45, ha="right")
ax_rmse.set_ylabel("RMSE (°C)", fontsize=11)
ax_rmse.set_title("Per-snapshot RMSE", fontsize=11)
ax_rmse.legend(fontsize=9)
ax_rmse.grid(True, axis="y", alpha=0.25)

fig.suptitle(
    "Isolating the prediction step: Lorenzo 2-D vs Lorenzo 1-D vs DEIM vs GP\n"
    f"(DEIM k={k_deim}, GP k={k_gp}, SST dataset, {N_SNAPSHOTS} snapshots)",
    fontsize=13, y=1.01)

out = ARGONNE / "lorenzo_vs_global_residuals.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
from scipy.stats import kurtosis

# Outlier threshold: 3σ of the Lorenzo distribution (common reference)
sigma_ref = np.std(pool_lor)
OUTLIER_THRESHOLDS = [2.0 * sigma_ref, 3.0 * sigma_ref, 5.0 * sigma_ref]

print("\n" + "─" * 75)
print(f"  {'Method':<22} {'RMSE':>8}  {'Std':>8}  {'Kurt':>8}  "
      f"{'|r|>2σ_L':>9}  {'|r|>3σ_L':>9}  {'|r|>5σ_L':>9}")
print("─" * 75)
for label, arr in [("Lorenzo 2-D", pool_lor), ("Lorenzo 1-D", pool_lor1d),
                    (f"DEIM k={k_deim}", pool_deim), (f"GP   k={k_gp}", pool_gp)]:
    n    = len(arr)
    rmse = np.sqrt(np.mean(arr**2))
    std  = np.std(arr)
    kurt = kurtosis(arr)
    out2 = np.sum(np.abs(arr) > OUTLIER_THRESHOLDS[0])
    out3 = np.sum(np.abs(arr) > OUTLIER_THRESHOLDS[1])
    out5 = np.sum(np.abs(arr) > OUTLIER_THRESHOLDS[2])
    print(f"  {label:<22} {rmse:8.4f}  {std:8.4f}  {kurt:8.2f}  "
          f"{out2:5d}({100*out2/n:.1f}%)  "
          f"{out3:5d}({100*out3/n:.1f}%)  "
          f"{out5:5d}({100*out5/n:.1f}%)")
print("─" * 75)
print(f"  σ_L = {sigma_ref:.4f} °C  (Lorenzo std, used as outlier reference)")
print("  Outlier thresholds: 2σ, 3σ, 5σ of Lorenzo residuals")
print("  kurtosis > 3 = more peaked than Gaussian = more compressible with entropy coding")
