#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

METAL_SDK="macosx"
SRC_DIR="gsplat/metal/csrc"
OUT_DIR="gsplat/metal"
MODULE_CACHE_DIR="$OUT_DIR/.clang-module-cache"
MODE="${1:-release}"

mkdir -p "$MODULE_CACHE_DIR"

AIR_FILES=()
for metal_file in "$SRC_DIR"/ops/*/*.metal; do
    air="${metal_file%.metal}.air"
    echo "Compiling $metal_file -> $air"
    args=(-sdk "$METAL_SDK" metal "-fmodules-cache-path=$MODULE_CACHE_DIR" -c "$metal_file" -o "$air")
    if [[ "$MODE" == "debug" ]]; then
        args+=(-gline-tables-only -frecord-sources)
    fi
    xcrun "${args[@]}"
    AIR_FILES+=("$air")
done

echo "Linking -> $OUT_DIR/gsplat_metal.metallib"
xcrun -sdk "$METAL_SDK" metallib "${AIR_FILES[@]}" -o "$OUT_DIR/gsplat_metal.metallib"

echo "Done."
