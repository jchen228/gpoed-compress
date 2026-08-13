"""
residual_autocorr_diagnostic.py
────────────────────────────────
Loads DEIM and GP checkpoints (no recomputation) and produces two figures:

  residual_distribution.png  — histogram of DEIM and GP residuals at k=800
                                for a single representative snapshot
  residual_autocorr.png      — 2-D spatial autocorrelation of the residual
                                field + radial profile, for both methods

Run from the Argonne directory or adjust ARGONNE / DATA_PATH below.
No libpressio needed. Does NOT re-run the compression experiment.
"""

import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.ndimage import uniform_filter

# ── Paths — adjust if running from a different directory ─────────────────────
ARGONNE   = Path(__file__).parent.parent
DATA_PATH = ARGONNE / "sst.nc"          # same file as main script
FIELD_TAG = "SST"
N_TRAIN   = 1727
K_DIAG    = 800                          # sensor count to inspect

# ── Snapshot index to analyse ────────────────────────────────────────────────
TIME_IDX  = 500                          # arbitrary mid-series snapshot

DPI = 150

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load raw data and ocean mask
# ═══════════════════════════════════════════════════════════════════════════
print("Loading SST data …")
try:
    import netCDF4 as nc4
    with nc4.Dataset(DATA_PATH) as ds:
        raw = ds.variables["sst"][:]              # (T, ny, nx) or (T, nx, ny)
        lat = np.array(ds.variables["lat"][:])
        lon = np.array(ds.variables["lon"][:])
except Exception as e:
    raise RuntimeError(f"Could not load {DATA_PATH}: {e}")

data = np.ma.filled(raw, fill_value=0.0).astype(np.float32)
n_T, ny, nx = data.shape
n_2D = ny * nx

ocean_mask = (data != 0).any(axis=0)           # (ny, nx) True = ocean
ocean_idx  = np.where(ocean_mask.ravel())[0]
n_ocean    = len(ocean_idx)
X_all      = np.column_stack(
    np.unravel_index(ocean_idx, (ny, nx))
).astype(np.float64)                            # (n_ocean, 2)

snap_full  = data[TIME_IDX]                     # (ny, nx)
snap_ocean = snap_full.ravel()[ocean_idx].astype(np.float64)

print(f"  n_T={n_T}  ny={ny}  nx={nx}  n_ocean={n_ocean}  snapshot t={TIME_IDX}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. DEIM residual
# ═══════════════════════════════════════════════════════════════════════════
deim_resid_ocean = None
_deim_ckpt = ARGONNE / f"deim_ckpt_{FIELD_TAG}_N{N_TRAIN}.npz"
if _deim_ckpt.exists():
    print(f"Loading DEIM checkpoint: {_deim_ckpt.name} …")
    ck = np.load(_deim_ckpt)
    Phi_max    = ck["Phi_max"].astype(np.float64)   # (n_ocean, k_max)
    mean_ocean = ck["mean_ocean"]
    k_avail    = Phi_max.shape[1]
    k_use      = min(K_DIAG, k_avail)
    Phi_k      = Phi_max[:, :k_use]

    # Q-DEIM sensor selection  (column-pivoted QR on Phi_k^T)
    from scipy.linalg import qr
    _, _, piv = qr(Phi_k.T, pivoting=True)
    sensors_deim = piv[:k_use]

    # Predict snapshot
    anom      = snap_ocean - mean_ocean
    Phi_s     = Phi_k[sensors_deim, :]             # (k, k)
    y_s       = anom[sensors_deim]
    coeffs, *_ = np.linalg.lstsq(Phi_s, y_s, rcond=None)
    pred_anom = Phi_k @ coeffs
    deim_resid_ocean = (anom - pred_anom).astype(np.float32)
    print(f"  DEIM  k={k_use}  residual RMSE={np.sqrt(np.mean(deim_resid_ocean**2)):.4f}")
else:
    print(f"  DEIM checkpoint not found: {_deim_ckpt}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. GP (Kriging) residual
# ═══════════════════════════════════════════════════════════════════════════
gp_resid_ocean = None
k_max_krig = 2000
_krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N{k_max_krig}.npz"
if not _krig_ckpt.exists():
    _krig_ckpt = ARGONNE / f"kriging_ckpt_{FIELD_TAG}_N1000.npz"
if _krig_ckpt.exists():
    print(f"Loading GP checkpoint: {_krig_ckpt.name} …")
    ck = np.load(_krig_ckpt)
    ls          = float(ck["ls"])
    var_gp      = float(ck["var"])
    noise_var   = float(ck["noise_var"])
    sensors_gp  = ck["sensors"][:K_DIAG]
    train_mean  = ck["train_mean"]
    train_std   = ck["train_std"]
    train_std_safe = np.where(train_std < 1e-10, 1.0, train_std)

    # Matérn-3/2 kernel
    def matern32(X1, X2, length_scale):
        d = np.sqrt(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1))
        r = np.sqrt(3) * d / length_scale
        return (1.0 + r) * np.exp(-r)

    k_use_gp = min(K_DIAG, len(sensors_gp))
    sensors_gp = sensors_gp[:k_use_gp]
    X_s = X_all[sensors_gp]

    # Check for pre-computed K matrices
    if "K_Xs_max" in ck.files:
        K_Xs = ck["K_Xs_max"][:, :k_use_gp].astype(np.float64)
        K_ss = ck["K_ss_max"][:k_use_gp, :k_use_gp].astype(np.float64)
    else:
        print("  Recomputing K matrices (not in checkpoint) …")
        K_Xs = var_gp * matern32(X_all, X_s, ls)
        K_ss = var_gp * matern32(X_s, X_s, ls)

    from scipy.linalg import cho_factor, cho_solve
    K_sub = K_ss + noise_var * np.eye(k_use_gp)
    L, lower = cho_factor(K_sub, lower=True)

    # Normalise sensor values
    ms_k = train_mean[sensors_gp]
    ss_k = train_std_safe[sensors_gp]
    y_norm = (snap_ocean[sensors_gp] - ms_k) / ss_k

    alpha   = var_gp * cho_solve((L, lower), y_norm)   # (k,)
    mu_norm = K_Xs @ alpha                               # (n_ocean,)
    mu      = mu_norm * train_std_safe + train_mean      # undo z-score

    gp_resid_ocean = (snap_ocean - mu).astype(np.float32)
    print(f"  GP    k={k_use_gp}  residual RMSE={np.sqrt(np.mean(gp_resid_ocean**2)):.4f}")
else:
    print(f"  GP checkpoint not found: {_krig_ckpt}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Residual histogram
# ═══════════════════════════════════════════════════════════════════════════
print("Plotting residual distributions …")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-2, 2, 120)
if deim_resid_ocean is not None:
    ax.hist(deim_resid_ocean, bins=bins, density=True, alpha=0.55,
            label=f"DEIM  k={K_DIAG}", color="#17becf")
if gp_resid_ocean is not None:
    ax.hist(gp_resid_ocean, bins=bins, density=True, alpha=0.55,
            label=f"GP    k={K_DIAG}", color="#e377c2")

# Overlay fitted Gaussian for reference
xs = np.linspace(-2, 2, 300)
for resid, col in [(deim_resid_ocean, "#17becf"), (gp_resid_ocean, "#e377c2")]:
    if resid is not None:
        s = np.std(resid)
        ax.plot(xs, np.exp(-xs**2 / (2*s**2)) / (s * np.sqrt(2*np.pi)),
                "--", color=col, lw=1.2, alpha=0.8)

ax.set_xlabel("Residual value (°C)", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title(f"Residual distribution at k={K_DIAG}, t={TIME_IDX}  (dashed = Gaussian fit)",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out1 = ARGONNE / "residual_distribution.png"
fig.savefig(out1, dpi=DPI, bbox_inches="tight"); plt.close(fig)
print(f"  Saved: {out1}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Spatial autocorrelation
# ═══════════════════════════════════════════════════════════════════════════
def ocean_to_2d(resid_ocean, ny, nx, ocean_idx):
    """Place ocean residuals back on the 2-D grid (land = NaN)."""
    grid = np.full(ny * nx, np.nan, dtype=np.float32)
    grid[ocean_idx] = resid_ocean
    return grid.reshape(ny, nx)

def spatial_autocorr(field_2d):
    """
    Normalised 2-D autocorrelation via FFT.
    NaN pixels are set to 0 (land mask); result normalised so lag-0 = 1.
    Returns the central patch ±50 pixels in each direction.
    """
    f = np.nan_to_num(field_2d, nan=0.0)
    f = f - f[f != 0].mean()             # demean over ocean only
    F  = np.fft.rfft2(f, s=(2*ny, 2*nx))
    ac = np.fft.irfft2(F * np.conj(F))[:ny, :nx]
    ac = np.fft.fftshift(ac)
    ac /= ac.max()
    return ac

def radial_profile(ac, ny, nx):
    """Average autocorrelation as a function of lag distance (pixels)."""
    cy, cx = ny // 2, nx // 2
    y_idx, x_idx = np.indices(ac.shape)
    r = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2).ravel().astype(int)
    r_max = min(cy, cx)
    radial = np.array([ac.ravel()[r == ri].mean() if (r == ri).any() else np.nan
                       for ri in range(r_max)])
    return np.arange(r_max), radial

print("Computing spatial autocorrelation …")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
HALF = 60   # crop size around centre

for row, (resid_ocean, label, color) in enumerate([
        (deim_resid_ocean, f"DEIM  k={K_DIAG}", "#17becf"),
        (gp_resid_ocean,   f"GP    k={K_DIAG}", "#e377c2")]):
    if resid_ocean is None:
        for ax in axes[row]: ax.axis("off")
        continue

    grid  = ocean_to_2d(resid_ocean, ny, nx, ocean_idx)
    ac    = spatial_autocorr(grid)
    lags, rp = radial_profile(ac, ny, nx)

    cy, cx = ny // 2, nx // 2
    crop   = ac[cy - HALF:cy + HALF, cx - HALF:cx + HALF]

    # (col 0) residual field
    im0 = axes[row, 0].imshow(grid, cmap="RdBu_r", aspect="auto",
                               vmin=-np.nanpercentile(np.abs(grid), 98),
                               vmax= np.nanpercentile(np.abs(grid), 98))
    axes[row, 0].set_title(f"{label} — residual field  t={TIME_IDX}", fontsize=11)
    plt.colorbar(im0, ax=axes[row, 0], fraction=0.03, pad=0.04, label="°C")

    # (col 1) 2-D autocorrelation (centre crop)
    im1 = axes[row, 1].imshow(crop, cmap="coolwarm", vmin=-0.3, vmax=1.0, aspect="auto")
    axes[row, 1].set_title(f"{label} — 2-D autocorr (±{HALF} px)", fontsize=11)
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.03, pad=0.04)

    # (col 2) radial profile
    axes[row, 2].plot(lags[:HALF], rp[:HALF], color=color, lw=2)
    axes[row, 2].axhline(0, color="k", lw=0.8, ls="--")
    axes[row, 2].set_xlabel("Lag (pixels)", fontsize=11)
    axes[row, 2].set_ylabel("Normalised autocorrelation", fontsize=11)
    axes[row, 2].set_title(f"{label} — radial autocorr profile", fontsize=11)
    axes[row, 2].grid(True, alpha=0.25)
    e_folding = np.where(rp[:HALF] < 1/np.e)[0]
    if len(e_folding):
        axes[row, 2].axvline(e_folding[0], color="gray", ls=":", lw=1.2,
                             label=f"e-folding ≈ {e_folding[0]} px")
        axes[row, 2].legend(fontsize=10)

fig.suptitle(f"Residual spatial autocorrelation  —  NOAA OI SST  t={TIME_IDX}  k={K_DIAG}",
             fontsize=13, y=1.01)
fig.tight_layout()
out2 = ARGONNE / "residual_autocorr.png"
fig.savefig(out2, dpi=DPI, bbox_inches="tight"); plt.close(fig)
print(f"  Saved: {out2}")

print("\nDone. Check residual_distribution.png and residual_autocorr.png")
print("Key question: if the radial autocorrelation drops to ~0 within 1-2 pixels,")
print("the residuals are effectively white noise and no spatial quantizer will help.")
print("If the e-folding length is >5 pixels, transform coding (wavelet/DCT) has room to gain.")
