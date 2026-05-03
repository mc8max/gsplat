# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import os
import subprocess
from types import SimpleNamespace

PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_CACHE_PATH = os.path.join(PATH, ".clang-module-cache")


def compile_metallib(mode: str = "release") -> str:
    """Compile all Metal kernels into a single metallib."""
    os.makedirs(MODULE_CACHE_PATH, exist_ok=True)
    metal_files = sorted(
        glob.glob(os.path.join(PATH, "csrc", "ops", "**", "*.metal"), recursive=True)
    )
    if not metal_files:
        raise FileNotFoundError("No .metal files found under gsplat/metal/csrc/ops")

    air_files = []
    for metal_file in metal_files:
        air_file = metal_file[:-6] + ".air"
        cmd = [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-fmodules-cache-path=" + MODULE_CACHE_PATH,
            "-c",
            metal_file,
            "-o",
            air_file,
        ]
        if mode == "debug":
            cmd.extend(["-gline-tables-only", "-frecord-sources"])
        subprocess.run(cmd, check=True)
        air_files.append(air_file)

    output = os.path.join(PATH, "gsplat_metal.metallib")
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", *air_files, "-o", output],
        check=True,
    )
    return output


def get_build_parameters() -> SimpleNamespace:
    sources = sorted(
        glob.glob(os.path.join(PATH, "csrc", "**", "*.mm"), recursive=True)
    ) + [os.path.join(PATH, "ext.mm")]
    include_dirs = [
        os.path.join(PATH, "include"),
        os.path.join(PATH, "csrc"),
    ]
    return SimpleNamespace(
        name="gsplat.metal_ext",
        sources=sources,
        include_dirs=include_dirs,
        extra_cflags=["-std=c++20", "-arch", "arm64"],
        extra_ldflags=[
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-arch",
            "arm64",
        ],
    )
