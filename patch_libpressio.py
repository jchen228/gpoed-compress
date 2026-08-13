#!/usr/bin/env python3
"""
Patch libpressio 0.94.0 SWIG bindings for macOS compatibility.

Two fixes:
1. pypressio.h: vector<uint64_t> → vector<size_t> at each pressio_data:: call site
2. pressioPYTHON_wrap.cxx: _pressio_io_data_from_numpy_1d<long> → <int64_t>
   (on macOS, 'long' != 'int64_t' in the type system even though same width)
"""
import os, sys, glob

STAGE_GLOB = (
    "/var/folders/d6/t6_bmsq93dq2p86mzc13wsjw0000gp/T/jchen228/"
    "spack-stage/spack-stage-libpressio-0.94.0-*/")

stages = glob.glob(STAGE_GLOB)
if not stages:
    sys.exit("ERROR: stage directory not found — re-run spack install first")

STAGE = stages[0].rstrip("/")
SRC   = os.path.join(STAGE, "spack-src")
BUILD = os.path.join(STAGE, "spack-build-" + STAGE.split("-")[-1])

print(f"Stage : {STAGE}")
print(f"Source: {SRC}")
print(f"Build : {BUILD}")

# ── 1. Patch pypressio.h ──────────────────────────────────────────────────────
pypressio = os.path.join(SRC, "tools/swig/pypressio.h")
assert os.path.exists(pypressio), f"Not found: {pypressio}"

with open(pypressio) as f:
    src = f.read()

FIXES = [
    ("pressio_data::empty(dtype, dimensions)",
     "pressio_data::empty(dtype, std::vector<size_t>(dimensions.begin(), dimensions.end()))"),
    ("pressio_data::nonowning(dtype, data, dimensions)",
     "pressio_data::nonowning(dtype, data, std::vector<size_t>(dimensions.begin(), dimensions.end()))"),
    ("pressio_data::copy(dtype, src, dimensions)",
     "pressio_data::copy(dtype, src, std::vector<size_t>(dimensions.begin(), dimensions.end()))"),
    ("pressio_data::owning(dtype, dimensions)",
     "pressio_data::owning(dtype, std::vector<size_t>(dimensions.begin(), dimensions.end()))"),
    ("pressio_data::move(dtype, data, dimensions, deleter, metadata)",
     "pressio_data::move(dtype, data, std::vector<size_t>(dimensions.begin(), dimensions.end()), deleter, metadata)"),
]

patched = 0
for old, new in FIXES:
    if old in src:
        src = src.replace(old, new)
        print(f"  [pypressio.h] patched: {old[:60]}")
        patched += 1
    else:
        print(f"  [pypressio.h] WARNING not found: {old[:60]}")

with open(pypressio, "w") as f:
    f.write(src)
print(f"pypressio.h: {patched}/{len(FIXES)} replacements applied\n")

# ── 2. Patch pressioPYTHON_wrap.cxx ─────────────────────────────────────────
wrap_glob = os.path.join(BUILD, "tools/swig/CMakeFiles/pressio.dir/pressioPYTHON_wrap.cxx")
wraps = glob.glob(wrap_glob)
if not wraps:
    # try alternate location
    wraps = glob.glob(os.path.join(BUILD, "**/pressioPYTHON_wrap.cxx"), recursive=True)

if not wraps:
    print("WARNING: pressioPYTHON_wrap.cxx not found — skipping wrap patch")
    print("         The build may still fail on the 'long' type issue.")
else:
    wrap = wraps[0]
    with open(wrap) as f:
        txt = f.read()

    n_long = txt.count("_from_numpy_1d<long >") + txt.count("_from_numpy_1d<long>")
    txt = txt.replace("_from_numpy_1d<long >", "_from_numpy_1d<int64_t>")
    txt = txt.replace("_from_numpy_1d<long>",  "_from_numpy_1d<int64_t>")

    for dim in ["2d", "3d", "4d"]:
        txt = txt.replace(f"_from_numpy_{dim}<long >", f"_from_numpy_{dim}<int64_t>")
        txt = txt.replace(f"_from_numpy_{dim}<long>",  f"_from_numpy_{dim}<int64_t>")

    # Also fix unsigned long → uint64_t in case it appears
    txt = txt.replace("_from_numpy_1d<unsigned long >", "_from_numpy_1d<uint64_t>")
    txt = txt.replace("_from_numpy_1d<unsigned long>",  "_from_numpy_1d<uint64_t>")

    with open(wrap, "w") as f:
        f.write(txt)
    print(f"pressioPYTHON_wrap.cxx: fixed {n_long} '<long>' occurrences → int64_t")
    print(f"  ({wrap})\n")

# ── 3. Print rebuild instructions ─────────────────────────────────────────────
print("=" * 60)
print("Now rebuild:")
print(f"  cd '{BUILD}'")
print("  make -j8")
print()
print("If make succeeds, finish the spack install with:")
print("  spack install --keep-stage libpressio@0.94.0 +python '~sz3'")
print("(spack will detect the build is done and just run the install phase)")
