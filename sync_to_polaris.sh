#!/bin/bash
# sync_to_polaris.sh
# Sync all changed files to Polaris and rebuild QoZ-hybrid there.
#
# Run from your Mac:  bash ~/Desktop/Argonne/sync_to_polaris.sh
#
# Prerequisites: ssh key already set up for polaris.alcf.anl.gov

POLARIS_USER="jchen228"
POLARIS_HOST="polaris.alcf.anl.gov"
REMOTE_BASE="/home/${POLARIS_USER}"

echo "=== 1. Syncing Python files ==="
rsync -avz --progress \
  ~/Desktop/Argonne/hybrid_qoz_experiment/hybrid_qoz_gp.py \
  ~/Desktop/Argonne/hybrid_qoz_experiment/run_hurricane_b2.py \
  ~/Desktop/Argonne/hybrid_qoz_experiment/run_hurricane_w_polaris.py \
  ~/Desktop/Argonne/hybrid_qoz_experiment/run_hurricane_polaris.pbs \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/Argonne/hybrid_qoz_experiment/"

rsync -avz --progress \
  ~/Desktop/Argonne/hurricane_hybrid_gp.py \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/Argonne/"

echo ""
echo "=== 2. Syncing patched C++ headers ==="
rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/compressor/SZInterpolationCompressor.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/compressor/"

# Config.hpp holds gp_levels, g_l0_precomputed_path, g_gp_levels globals.
# Now also has #include <array> to fix the GCC 14 incomplete-type error.
rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/utils/Config.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/utils/"

rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/api/impl/SZInterp.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/api/impl/"

echo ""
echo "=== 2b. Syncing new C++ files (arith + rawint encoder) ==="
rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/encoder/ArithmeticEncoder.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/encoder/"

rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/encoder/RawIntEncoder.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/encoder/"

rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/api/impl/SZInterp_arith.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/api/impl/"

rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/include/QoZ/api/impl/SZInterp_raw.hpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/include/QoZ/api/impl/"

rsync -avz --progress \
  ~/Desktop/Argonne/QoZ-hybrid/test/qoz_arith.cpp \
  ~/Desktop/Argonne/QoZ-hybrid/test/qoz_rawint.cpp \
  "${POLARIS_USER}@${POLARIS_HOST}:${REMOTE_BASE}/QoZ-hybrid/test/"

echo ""
echo "=== 3. Rebuilding QoZ-hybrid on Polaris ==="
ssh "${POLARIS_USER}@${POLARIS_HOST}" bash -l << 'REMOTE'
  set -e
  cd ~/QoZ-hybrid/build
  # NOTE: cmake is NOT needed for a plain make (project was already configured).
  # The build environment (compiler, MPI, etc.) was loaded at configure time and
  # is baked into the Makefiles.  If make fails, run:
  #   module load PrgEnv-gnu cmake/3.x.y  (check: module spider cmake)
  #   cmake .. && make -j8
  make -j8
  echo "Rebuild done."
  echo "  Huffman binary: $(realpath test/qoz)"
  echo "  Arith   binary: $(realpath test/qoz_arith)"
  echo "  RawInt  binary: $(realpath test/qoz_rawint)"
REMOTE

echo ""
echo "=== All done. To submit the job: ==="
echo "  ssh ${POLARIS_USER}@${POLARIS_HOST}"
echo "  qsub ~/Argonne/hybrid_qoz_experiment/run_hurricane_polaris.pbs"
