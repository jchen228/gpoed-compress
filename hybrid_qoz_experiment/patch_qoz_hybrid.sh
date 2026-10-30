#!/usr/bin/env bash
# patch_qoz_hybrid.sh  (v3 — no class-member mutation, uses thread-local global)
#
# Usage:
#   cd /Users/jchen228/Desktop/Argonne    (or cluster equivalent)
#   bash hybrid_qoz_experiment/patch_qoz_hybrid.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARGONNE="$(dirname "$SCRIPT_DIR")"
SRC="$ARGONNE/QoZ"
DST="$ARGONNE/QoZ-hybrid"

echo "==> Copying QoZ → QoZ-hybrid …"
rm -rf "$DST"
cp -r "$SRC" "$DST"
# Remove stale Mac build cache so cmake starts clean
rm -rf "$DST/build"

# ── Locate cmake ─────────────────────────────────────────────────────────────
if command -v cmake &>/dev/null && cmake --version &>/dev/null; then
    CMAKE=cmake
elif python3 -c "import cmake; print(cmake.CMAKE_BIN_DIR)" &>/dev/null; then
    CMAKE="$(python3 -c "import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR,'cmake'))")"
else
    echo "cmake not found. Installing via pip …"
    pip3 install cmake --quiet
    CMAKE="$(python3 -c "import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR,'cmake'))")"
fi
echo "Using cmake: $CMAKE ($($CMAKE --version | head -1))"

# ═════════════════════════════════════════════════════════════════════════════
# Patch 0 — SZInterp.hpp:
#   (a) propagate l0 path via thread-local global before decompress
#   (b) disable freezeDimTest when l0 is set (frozen_dim changes coarse-grid
#       layout in a way the Python l0-builder doesn't account for)
# ═════════════════════════════════════════════════════════════════════════════
INTERP="$DST/include/QoZ/api/impl/SZInterp.hpp"
echo "==> Patching SZInterp.hpp …"
python3 - "$INTERP" <<'PYEOF'
import sys
path = sys.argv[1]
src  = open(path).read()

# ── (a) thread-local propagation ─────────────────────────────────────────────
old_dec = "        sz.decompress(cmpDataPos, cmpSize, decData);"
new_dec = (
    "        QoZ::g_l0_precomputed_path = conf.l0_precomputed_path;\n"
    "        sz.decompress(cmpDataPos, cmpSize, decData);\n"
    "        QoZ::g_l0_precomputed_path.clear();  // reset after use"
)
if old_dec not in src:
    print("  (a) already patched or line not found — skipping")
else:
    src = src.replace(old_dec, new_dec, 1)
    print("  (a) done")

# ── (b) disable freezeDimTest when l0 is set ─────────────────────────────────
# freezeDimTest makes the coarse-grid stride=1 along the chosen frozen dim,
# creating more coarse points than the Python l0-builder expects.
# Insert the override right after the QoZ>=4 block in Tuning().
old_freeze = (
    "        if(conf.QoZ>=4){\n"
    "            conf.crossBlock=1;\n"
    "            conf.blockwiseTuning=1;\n"
    "            if(conf.blockwiseSampleRate<1.0)\n"
    "                conf.blockwiseSampleRate=3.0;\n"
    "        }"
)
new_freeze = (
    old_freeze + "\n"
    "        // When using GP l0, force standard maxStep coarse grid\n"
    "        // (frozen_dim alters coarse-grid layout; Python l0 assumes stride=maxStep)\n"
    "        if (!conf.l0_precomputed_path.empty()) {\n"
    "            conf.freezeDimTest = 0;\n"
    "            conf.frozen_dim = -1;\n"
    "        }"
)
if 'l0_precomputed_path.empty()' in src:
    print("  (b) already patched — skipping")
elif old_freeze not in src:
    print("  (b) QoZ>=4 block not found — skipping")
else:
    src = src.replace(old_freeze, new_freeze, 1)
    print("  (b) done")

open(path,'w').write(src)
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# Patch 1 — Config.hpp: add l0_precomputed_path field + thread-local global
# ═════════════════════════════════════════════════════════════════════════════
CFG="$DST/include/QoZ/utils/Config.hpp"
echo "==> Patching Config.hpp …"
python3 - "$CFG" <<'PYEOF'
import sys
path = sys.argv[1]
src  = open(path).read()
if 'l0_precomputed_path' in src:
    print("  already patched"); sys.exit(0)
# 1a. Add field after:  int sampleBlockSize=0;
marker = "int sampleBlockSize=0;"
if marker not in src:
    print("ERROR: sampleBlockSize marker not found"); sys.exit(1)
src = src.replace(marker,
    marker + "\n        std::string l0_precomputed_path; // set by --l0-precomputed", 1)
# 1b. Add thread-local global before #endif
if 'g_l0_precomputed_path' not in src:
    idx = src.rfind('#endif')
    if idx == -1:
        print("ERROR: #endif not found"); sys.exit(1)
    insert = (
        "namespace QoZ {\n"
        "    inline thread_local std::string g_l0_precomputed_path;\n"
        "}\n\n"
    )
    src = src[:idx] + insert + src[idx:]
open(path,'w').write(src)
print("  done")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# Patch 2 — SZInterpolationCompressor.hpp:
#   (a) skip build_grid when conf.l0_precomputed_path is set
#   (b) skip recover_grid; read l0 file into decData instead
#
#   IMPORTANT: Do NOT add any member variables to this class.
#   The class is a template instantiated in many translation units;
#   adding a std::string member shifts internal offsets and causes
#   SIGSEGV in block_interpolation_1d_crossblock_3d at decompress time.
#   We use the thread-local QoZ::g_l0_precomputed_path instead.
# ═════════════════════════════════════════════════════════════════════════════
COMP="$DST/include/QoZ/compressor/SZInterpolationCompressor.hpp"
echo "==> Patching SZInterpolationCompressor.hpp …"
python3 - "$COMP" <<'PYEOF'
import sys
path = sys.argv[1]
src  = open(path).read()
if 'g_l0_precomputed_path' in src:
    print("  already patched"); sys.exit(0)

# ── (a) wrap build_grid call ─────────────────────────────────────────────────
# conf IS in scope inside compress()
OLD_BUILD = "build_grid(conf,data,maxStep,tuning);"
if OLD_BUILD not in src:
    print("ERROR: build_grid call site not found"); sys.exit(1)
NEW_BUILD = (
    "if (conf.l0_precomputed_path.empty()) {\n"
    "                    build_grid(conf,data,maxStep,tuning);\n"
    "                } else {\n"
    "                    // GP pre-pass owns level-0; skip lossless anchor storage.\n"
    "                    // Force frozen_dim=-1 so the coarse grid is the standard\n"
    "                    // maxStep-aligned grid that the Python l0-builder produces.\n"
    "                    conf.frozen_dim = -1;\n"
    "                }"
)
src = src.replace(OLD_BUILD, NEW_BUILD, 1)

# ── (b) wrap recover_grid call ───────────────────────────────────────────────
# conf is NOT in scope in decompress() — use the thread-local global
# Compute n_elements from global_dimensions (avoids relying on class state)
OLD_RECOVER = "recover_grid(decData,global_dimensions,maxStep,frozen_dim);"
if OLD_RECOVER not in src:
    print("ERROR: recover_grid call site not found"); sys.exit(1)
NEW_RECOVER = (
    "if (QoZ::g_l0_precomputed_path.empty()) {\n"
    "                    recover_grid(decData,global_dimensions,maxStep,frozen_dim);\n"
    "                } else {\n"
    "                    // Read GP pre-computed level-0 (full-size float32 array).\n"
    "                    // Fine positions don't matter — they are overwritten by\n"
    "                    // interpolation in the next step.\n"
    "                    std::ifstream _l0f(QoZ::g_l0_precomputed_path, std::ios::binary);\n"
    "                    if (!_l0f) {\n"
    "                        fprintf(stderr,\n"
    "                                \"ERROR: cannot open l0_precomputed: %s\\n\",\n"
    "                                QoZ::g_l0_precomputed_path.c_str());\n"
    "                        exit(1);\n"
    "                    }\n"
    "                    size_t _n = 1;\n"
    "                    for (auto _d : global_dimensions) _n *= _d;\n"
    "                    std::vector<float> _l0(_n);\n"
    "                    _l0f.read(reinterpret_cast<char*>(_l0.data()), _n * sizeof(float));\n"
    "                    for (size_t _i = 0; _i < _n; _i++)\n"
    "                        decData[_i] = static_cast<T>(_l0[_i]);\n"
    "                }"
)
src = src.replace(OLD_RECOVER, NEW_RECOVER, 1)

# ── add missing includes if needed ──────────────────────────────────────────
if '#include <fstream>' not in src:
    first = src.index('#include')
    eol   = src.index('\n', first)
    src   = src[:eol+1] + '#include <fstream>\n' + src[eol+1:]
if '#include <vector>' not in src:
    first = src.index('#include')
    eol   = src.index('\n', first)
    src   = src[:eol+1] + '#include <vector>\n' + src[eol+1:]

open(path,'w').write(src)
print("  done")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# Patch 3 — qoz.cpp: add --l0-precomputed long option
# ═════════════════════════════════════════════════════════════════════════════
QOZ="$DST/test/qoz.cpp"
echo "==> Patching qoz.cpp …"
python3 - "$QOZ" <<'PYEOF'
import sys
path = sys.argv[1]
src  = open(path).read()
if 'l0-precomputed' in src:
    print("  already patched"); sys.exit(0)

# ── 3a. Declare l0Path variable ──────────────────────────────────────────────
OLD_DECL = "char *decPath = nullptr;"
if OLD_DECL not in src:
    print("ERROR: decPath declaration not found"); sys.exit(1)
src = src.replace(OLD_DECL,
    OLD_DECL + "\n    char *l0Path = nullptr;", 1)

# ── 3b. Parse --l0-precomputed before the single-char switch ────────────────
OLD_GUARD = "if (argv[i][0] != '-' || argv[i][2]) {"
if OLD_GUARD not in src:
    print("ERROR: argv guard not found"); sys.exit(1)
INSERT = (
    "// long options\n"
    "        if (strcmp(argv[i], \"--l0-precomputed\") == 0) {\n"
    "            if (++i == argc) usage();\n"
    "            l0Path = argv[i];\n"
    "            continue;\n"
    "        }\n"
    "        "
)
src = src.replace(OLD_GUARD, INSERT + OLD_GUARD, 1)

# ── 3c. Store into conf after sampleBlockSize ────────────────────────────────
OLD_SAMPLE = "if(sampleBlockSize>0)\n        conf.sampleBlockSize=sampleBlockSize;"
if OLD_SAMPLE not in src:
    print("ERROR: sampleBlockSize assignment not found"); sys.exit(1)
src = src.replace(OLD_SAMPLE,
    OLD_SAMPLE + "\n    if (l0Path != nullptr)\n        conf.l0_precomputed_path = std::string(l0Path);", 1)

# ── 3d. Add <cstring> for strcmp ─────────────────────────────────────────────
if '#include <cstring>' not in src:
    src = '#include <cstring>\n' + src

open(path,'w').write(src)
print("  done")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# Patch 4 — gp_levels: level-1 skip support
#   (a) Config.hpp:                 add gp_levels field + g_gp_levels thread-local
#   (b) SZInterp.hpp:               propagate g_gp_levels before decompress
#   (c) SZInterpolationCompressor:  skip level-1 in compress + decompress loops
#   (d) qoz.cpp:                    add --gp-levels flag
# ═════════════════════════════════════════════════════════════════════════════
echo "==> Patching gp_levels support …"
python3 - "$CFG" "$INTERP" "$COMP" "$QOZ" <<'PYEOF'
import sys
cfg_path, interp_path, comp_path, qoz_path = sys.argv[1:]

# ── (a) Config.hpp ───────────────────────────────────────────────────────────
src = open(cfg_path).read()
if 'gp_levels' not in src:
    src = src.replace(
        "std::string l0_precomputed_path; // set by --l0-precomputed",
        "std::string l0_precomputed_path; // set by --l0-precomputed\n"
        "        uint8_t gp_levels = 0;   // 1=level-0 only, 2=level-0+1", 1)
    src = src.replace(
        "    inline thread_local std::string g_l0_precomputed_path;\n",
        "    inline thread_local std::string g_l0_precomputed_path;\n"
        "    inline thread_local uint8_t g_gp_levels = 1;\n", 1)
    open(cfg_path, 'w').write(src)
    print("  (a) Config.hpp done")
else:
    print("  (a) Config.hpp: already patched")

# ── (b) SZInterp.hpp ─────────────────────────────────────────────────────────
src = open(interp_path).read()
if 'g_gp_levels' not in src:
    src = src.replace(
        "        QoZ::g_l0_precomputed_path = conf.l0_precomputed_path;\n",
        "        QoZ::g_l0_precomputed_path = conf.l0_precomputed_path;\n"
        "        QoZ::g_gp_levels = conf.gp_levels;\n", 1)
    open(interp_path, 'w').write(src)
    print("  (b) SZInterp.hpp done")
else:
    print("  (b) SZInterp.hpp: already patched")

# ── (c) SZInterpolationCompressor.hpp ────────────────────────────────────────
src = open(comp_path).read()
if 'g_gp_levels' not in src:
    # decompress: one occurrence of interpolation_level-- (16 spaces + trailing spaces).
    # Insert level-1 skip immediately after it.
    OLD_DEC = "                interpolation_level--;           \n"
    NEW_DEC = ("                interpolation_level--;           \n"
               "                // Skip level-1 when GP pre-pass covers it\n"
               "                if (QoZ::g_gp_levels >= 2 && interpolation_level > 0)\n"
               "                    interpolation_level--;  // GP pre-pass owns level-1\n")
    if OLD_DEC in src:
        src = src.replace(OLD_DEC, NEW_DEC, 1)
        print("  (c-dec) decompress level-1 skip done")
    else:
        # Try without trailing spaces (editor may have stripped them)
        OLD_DEC2 = "                interpolation_level--;\n"
        NEW_DEC2 = ("                interpolation_level--;\n"
                    "                // Skip level-1 when GP pre-pass covers it\n"
                    "                if (QoZ::g_gp_levels >= 2 && interpolation_level > 0)\n"
                    "                    interpolation_level--;  // GP pre-pass owns level-1\n")
        if OLD_DEC2 in src:
            src = src.replace(OLD_DEC2, NEW_DEC2, 1)
            print("  (c-dec) decompress level-1 skip done (no-trailing-space variant)")
        else:
            print("  (c-dec) WARNING: decompress pattern not found")

    # compress: one occurrence of start_level-- (inside else-if block, 16 spaces).
    # Insert level-1 skip immediately after it.
    OLD_CMP = "                start_level--;\n"
    NEW_CMP = ("                start_level--;\n"
               "                // Skip level-1 when GP pre-pass covers it\n"
               "                if (!conf.l0_precomputed_path.empty() &&\n"
               "                    conf.gp_levels >= 2 && start_level > 0)\n"
               "                    start_level--;  // GP pre-pass owns level-1\n")
    if OLD_CMP in src:
        src = src.replace(OLD_CMP, NEW_CMP, 1)
        print("  (c-cmp) compress level-1 skip done")
    else:
        print("  (c-cmp) WARNING: compress pattern not found")

    open(comp_path, 'w').write(src)
else:
    print("  (c) SZInterpolationCompressor.hpp: already patched")

# ── (d) qoz.cpp ──────────────────────────────────────────────────────────────
src = open(qoz_path).read()
if '--gp-levels' not in src:
    # Add gpLevels variable declaration
    src = src.replace(
        "    char *l0Path = nullptr;",
        "    char *l0Path = nullptr;\n    int gpLevels = 1;", 1)
    # Add --gp-levels parsing next to --l0-precomputed
    src = src.replace(
        '        if (strcmp(argv[i], "--l0-precomputed") == 0) {\n'
        '            if (++i == argc) usage();\n'
        '            l0Path = argv[i];\n'
        '            continue;\n'
        '        }\n',
        '        if (strcmp(argv[i], "--l0-precomputed") == 0) {\n'
        '            if (++i == argc) usage();\n'
        '            l0Path = argv[i];\n'
        '            continue;\n'
        '        }\n'
        '        if (strcmp(argv[i], "--gp-levels") == 0) {\n'
        '            if (++i == argc) usage();\n'
        '            gpLevels = atoi(argv[i]);\n'
        '            continue;\n'
        '        }\n', 1)
    # Store gpLevels into conf
    src = src.replace(
        "    if (l0Path != nullptr)\n        conf.l0_precomputed_path = std::string(l0Path);",
        "    if (l0Path != nullptr)\n        conf.l0_precomputed_path = std::string(l0Path);\n"
        "    conf.gp_levels = static_cast<uint8_t>(gpLevels);", 1)
    open(qoz_path, 'w').write(src)
    print("  (d) qoz.cpp done")
else:
    print("  (d) qoz.cpp: already patched")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# Build
# ═════════════════════════════════════════════════════════════════════════════
echo "==> Building QoZ-hybrid …"
mkdir -p "$DST/build"
cd "$DST/build"

"$CMAKE" .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DST/install" \
    -DCMAKE_CXX_FLAGS="-Wno-error" \
    2>&1 | tail -5

NJOBS=$(python3 -c "import os; print(os.cpu_count() or 4)")
make -j"$NJOBS" 2>&1 | tail -30
make install 2>&1 | tail -5

echo ""
echo "==> Done.  Binary: $DST/install/bin/qoz"
