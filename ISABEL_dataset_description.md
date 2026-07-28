# ISABEL Hurricane Dataset — Structure Reference

## Overview

ISABEL is a numerical simulation of Hurricane Isabel (Atlantic, September 2003) produced by the **Weather Research and Forecast (WRF) model** under the IEEE Visualization 2004 Contest. The simulation was run at the National Center for Atmospheric Research (NCAR) by Kelvin K. Droegemeier, Ming Xue, and the Center for Analysis and Prediction of Storms (CAPS) at the University of Oklahoma.

> **Citation:** M. Shead (ed.), *IEEE Visualization 2004 Contest Dataset: Hurricane Isabel*, NCAR, 2004. Available at https://ieeexplore.ieee.org/document/1566621

---

## Grid Structure

Every variable in this dataset shares the same 3D rectilinear grid:

```
Shape:  (100, 500, 500)
Axes:   axis 0 — vertical level   (100 levels, 0 = near-surface, 99 = upper atmosphere)
        axis 1 — latitude / y      (500 points, south → north)
        axis 2 — longitude / z     (500 points, west → east)
```

- **Horizontal resolution:** approximately 4.3 km per grid cell (2139 km east-west ÷ 500 pixels ≈ 4.3 km/pixel; north-south is similar at ~4.0 km/pixel)
- **Vertical resolution:** non-uniform; lower levels are more closely spaced near the surface
- **Total grid points per variable:** 100 × 500 × 500 = **25,000,000**

A single 2D horizontal slice at one vertical level is a **500×500** spatial field covering the full hurricane footprint.

---

## File Format

All data files use the same binary format:

| Property | Value |
|---|---|
| Extension | `.bin.f32` |
| Data type | 32-bit floating point (float32) |
| Byte order | **Little-endian** |
| Layout | Flat binary, C-order (level-major → row-major) |
| File size | ~96 MB per variable |

### Reading in Python

```python
import numpy as np

data = np.fromfile("Uf48.bin.f32", dtype=np.float32)
data = data.reshape((100, 500, 500))
# axis 0: vertical level (0–99)
# axis 1: latitude  y   (0–499, south to north)
# axis 2: longitude z   (0–499, west to east)

# Extract one horizontal slice at level 50
slice_2d = data[50, :, :]           # shape (500, 500)

# Downsample by 7× for faster work
slice_ds  = data[50, ::7, ::7]      # shape (72, 72) = 5184 points
```

> **Note:** Use `dtype=np.float32` with no byte-swapping. The files have already been converted to little-endian format.

---

## Variables

The dataset folder contains **13 physical variables**, each stored as one `.bin.f32` file. Seven of the most skewed (near-zero) variables also have a pre-computed **log₁₀ version** (suffix `.log10.bin.f32`).

### Wind Components

| File | Variable | Units | Min | Max | Mean | Std |
|---|---|---|---|---|---|---|
| `Uf48.bin.f32` | East–west (U) wind | m/s | −53.0 | 39.6 | −2.2 | 9.2 |
| `Vf48.bin.f32` | North–south (V) wind | m/s | −45.6 | 48.1 | 3.6 | 11.1 |
| `Wf48.bin.f32` | Vertical (W) wind | m/s | −3.2 | 13.3 | 0.004 | 0.14 |

U and V are the dominant horizontal wind components driving the hurricane's rotation. W is the vertical updraft/downdraft velocity — much smaller in magnitude but critical near the eye wall where strong updrafts occur. Positive W indicates upward motion.

### Thermodynamic Variables

| File | Variable | Units | Min | Max | Mean | Std |
|---|---|---|---|---|---|---|
| `Pf48.bin.f32` | Pressure perturbation | Pa | −3411.7 | 3224.4 | 375.9 | 504.5 |
| `TCf48.bin.f32` | Temperature | °C | −76.6 | 29.6 | −30.8 | 31.7 |
| `QVAPORf48.bin.f32` | Water vapour mixing ratio | kg/kg | 0.0 | 0.020 | 0.002 | 0.004 |

`Pf48` is the **perturbation pressure** (deviation from a background reference state), not absolute pressure. Negative values near the hurricane eye indicate lower-than-background pressure — this is the pressure deficit that drives the inflow. `TCf48` covers the full atmospheric column: near-surface temperatures can reach ~30°C while upper levels drop below −76°C. `QVAPORf48` is the mass of water vapour per kg of moist air; values peak in the warm moist boundary layer.

### Hydrometeors (Water Species)

| File | Variable | Units | Min | Max | Log10 file available |
|---|---|---|---|---|---|
| `QCLOUDf48.bin.f32` | Cloud liquid water mixing ratio | kg/kg | ≈0 | 0.0020 | ✓ |
| `QRAINf48.bin.f32` | Rain water mixing ratio | kg/kg | ≈0 | 0.0057 | ✓ |
| `QICEf48.bin.f32` | Ice mixing ratio | kg/kg | ≈0 | 0.0012 | ✓ |
| `QSNOWf48.bin.f32` | Snow mixing ratio | kg/kg | ≈0 | 0.0009 | ✓ |
| `QGRAUPf48.bin.f32` | Graupel mixing ratio | kg/kg | ≈0 | 0.0067 | ✓ |

These five fields track the mass of each water phase per kg of air. All are highly sparse — almost zero everywhere except in active precipitation regions (eye wall and spiral rain bands). The log₁₀ versions (`QRAINf48.log10.bin.f32`, etc.) are provided specifically because the raw values span many orders of magnitude and are hard to visualise or analyse linearly. Graupel (pellet-sized ice) has the largest values and is concentrated in the intense convective towers of the eye wall.

### Derived Fields

| File | Variable | Units | Min | Max | Log10 file available |
|---|---|---|---|---|---|
| `CLOUDf48.bin.f32` | Cloud fraction | – | 0.0 | 0.0020 | ✓ |
| `PRECIPf48.bin.f32` | Surface precipitation rate | – | 0.0 | 0.0081 | ✓ |

`CLOUDf48` is the column-integrated cloud cover fraction derived from the hydrometeor fields. `PRECIPf48` is the accumulated surface precipitation rate. Both are sparse outside active convection.

---

## Vertical Structure

The 100 vertical levels span from near the ocean surface (level 0) to the upper troposphere/lower stratosphere (level 99). Below is the U-wind range at selected levels:

| Level | Altitude (approx.) | U min (m/s) | U max (m/s) | U mean (m/s) | U std (m/s) |
|---|---|---|---|---|---|
| 0 | Near surface | −36.6 | 23.5 | −2.5 | 6.4 |
| 10 | Low atmosphere | −52.9 | 30.7 | −2.9 | 11.0 |
| 20 | Mid-low | −48.8 | 32.5 | −2.1 | 11.1 |
| 30 | Mid | −45.4 | 33.8 | −0.9 | 10.1 |
| 40 | Mid | −38.4 | 25.2 | 0.3 | 9.3 |
| 50 | Mid-high | −30.0 | 23.5 | 0.4 | 9.3 |
| 60 | Upper-mid | −24.1 | 27.0 | −1.1 | 8.9 |
| 70 | Upper | −25.5 | 30.0 | −0.7 | 9.1 |
| 80 | High | −20.3 | 13.9 | −3.5 | 6.8 |
| 90 | Very high | −18.1 | 5.7 | −6.7 | 4.5 |

Wind speed is strongest in the low-to-mid atmosphere (levels 10–40), where the rotational winds of the hurricane are most intense. The upper levels (80–99) show weaker winds and a shift toward the ambient east-to-west flow.

---

## Log₁₀ Variants

Seven sparse variables are provided in a log₁₀-transformed version alongside the raw data:

```
CLOUDf48.log10.bin.f32
PRECIPf48.log10.bin.f32
QCLOUDf48.log10.bin.f32
QGRAUPf48.log10.bin.f32
QICEf48.log10.bin.f32
QRAINf48.log10.bin.f32
QSNOWf48.log10.bin.f32
```

These exist because the raw hydrometeor fields span 6–10 orders of magnitude (essentially zero outside clouds, with sharp peaks in the eye wall). Log₁₀ compression makes them easier to visualise and use in regression tasks. The log₁₀ files for CLOUD range from approximately −12.8 to −2.7, corresponding to the full dynamic range of the raw values.

---

## Spatial Structure of the Hurricane

The 2D horizontal fields at any level show:

- **Eye** — a roughly circular calm region (~15–30 km radius) near the domain centre with low wind speed and warm temperatures
- **Eye wall** — the narrow ring immediately surrounding the eye with the highest wind speeds, strongest updrafts (W), and most intense precipitation
- **Spiral rain bands** — curved bands of convection extending outward from the eye wall
- **Outer region** — gradually weakening winds and sparse precipitation

The domain is centred on the hurricane; the storm does not extend to the grid boundary, so edge effects are minimal for most variables.

---

## Usage Notes for Sensor Placement Work

- **Recommended starting level:** 50 (mid-atmosphere, strong spatial structure, moderate dynamic range)
- **Downsampling:** use `data[level, ::N, ::N]` to reduce the 500×500 grid. `N=7` gives a 72×72 = 5184-point grid, a good balance of resolution and speed
- **Normalise before regression:** subtract the field mean and divide by standard deviation before fitting GP hyperparameters or building POD bases
- **U and V together** can reconstruct the full horizontal wind vector field; W adds the vertical component
- **Non-stationarity:** the eye-wall region has very short lengthscales (sharp gradients); the outer region has much longer lengthscales. A single global GP lengthscale is a compromise — domain splitting or non-stationary kernels help
- **Multi-variable snapshot matrix:** treating U, V, W, TC, P as columns of a snapshot matrix gives a richer ensemble for building a POD basis than using vertical levels alone
