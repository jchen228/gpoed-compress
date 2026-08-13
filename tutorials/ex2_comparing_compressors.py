#!/usr/bin/env python
"""
ex2_comparing_compressors.py
Working copy of the Exercise 2 script (tutorial version is read-only).

Changes from original:
  - mgard removed by default (not in this libpressio build); add back if available
  - run_compressor wraps PressioException → RuntimeError so mpi4py can serialize it
  - figures/results.csv written to the tutorial figures dir (same as original)
  - visualize.py path points to the tutorial directory

Run with:
    mpirun -n 4 python3 ex2_comparing_compressors.py
Then plot:
    python3 ~/libpressio_tutorial/exercises/2_comparing_compressors/rate_distortion.py
"""

from pathlib import Path
from pprint import pprint
import csv
import libpressio
import numpy as np
import itertools
from mpi4py.futures import MPICommExecutor

# ── Paths ────────────────────────────────────────────────────────────────────
TUTORIAL_DIR = Path.home() / "libpressio_tutorial/exercises/2_comparing_compressors"
INPUT_PATH   = TUTORIAL_DIR / "../datasets/CLOUDf48.bin.f32"
FIGURES_DIR  = TUTORIAL_DIR / "figures"
VISUALIZE_PY = TUTORIAL_DIR / "visualize.py"

input_data = np.fromfile(INPUT_PATH, dtype=np.float32).reshape(100, 500, 500)

# ── Compressors to test ───────────────────────────────────────────────────────
# Auto-detect which compressors are available in this libpressio build.
def _available(name):
    try:
        libpressio.PressioCompressor.from_config({"compressor_id": name})
        return True
    except Exception:
        return False

ALL_COMPRESSORS = ["sz", "sz3", "mgard", "zfp"]
COMPRESSORS = [c for c in ALL_COMPRESSORS if _available(c)]
print(f"Available compressors: {COMPRESSORS}")

configs = [
    {
        "compressor_id": compressor_id,
        "compressor_config": {"pressio:abs": bound},
        "bound": bound,
    }
    for bound, compressor_id in itertools.product(
        [1e-4, 1e-3],   # quick test; restore np.logspace(-7, -3, num=5) for full run
        COMPRESSORS,
    )
]


def run_compressor(args):
    """Run one (compressor, bound) config; catches PressioException for MPI safety."""
    try:
        compressor = libpressio.PressioCompressor.from_config({
            "compressor_id": args["compressor_id"],
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["size", "error_stat", "time"],
            },
            "compressor_config": args["compressor_config"],
        })

        decomp_data = input_data.copy()
        comp_data   = compressor.encode(input_data)
        decomp_data = compressor.decode(comp_data, decomp_data)
        metrics     = compressor.get_metrics()

        return {
            "compressor_id": args["compressor_id"],
            "bound":         args["bound"],
            "metrics":       metrics,
        }
    except Exception as e:
        # Return error as data so one failure doesn't abort the whole run
        return {
            "compressor_id": args["compressor_id"],
            "bound":         args["bound"],
            "error":         f"{type(e).__name__}: {e}",
            "metrics":       None,
        }


# ── Main ─────────────────────────────────────────────────────────────────────
FIGURES_DIR.mkdir(exist_ok=True)
csv_path = FIGURES_DIR / "results.csv"

with open(csv_path, "w") as csvfile:
    fieldnames = ["compression_ratio", "bound", "psnr", "compressor_id"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with MPICommExecutor() as pool:
        if pool is not None:   # only root rank executes this block
            for result in pool.map(run_compressor, configs, unordered=True):
                if result.get("error"):
                    print(f"  SKIP [{result['compressor_id']} {result['bound']:.1e}]: {result['error']}")
                    continue
                writer.writerow({
                    "compression_ratio": result["metrics"]["size:compression_ratio"],
                    "psnr":              result["metrics"]["error_stat:psnr"],
                    "compressor_id":     result["compressor_id"],
                    "bound":             result["bound"],
                })
                pprint(result)
            print(f"\nResults written to {csv_path}")
            print("Run rate_distortion.py to plot.")
