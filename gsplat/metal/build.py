# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build helpers for the gsplat Metal extension and kernel library."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_CACHE_PATH = os.path.join(PATH, ".clang-module-cache")


def ensure_xcrun_available() -> None:
    """Fail fast when Apple command-line tooling is not available."""

    if shutil.which("xcrun") is None:
        raise EnvironmentError(
            "xcrun not found. Install Xcode Command Line Tools: "
            "xcode-select --install"
        )


def compile_metallib(mode: str = "release") -> str:
    """Compile all Metal kernels into the bundled ``gsplat_metal.metallib``."""

    ensure_xcrun_available()
    os.makedirs(MODULE_CACHE_PATH, exist_ok=True)
    metal_files = sorted(
        glob.glob(os.path.join(PATH, "csrc", "**", "*.metal"), recursive=True)
    )
    if not metal_files:
        raise FileNotFoundError("No .metal files found under gsplat/metal/csrc")

    output = os.path.join(PATH, "gsplat_metal.metallib")
    with tempfile.TemporaryDirectory() as tmpdir:
        air_files = []
        for metal_file in metal_files:
            air_file = os.path.join(tmpdir, os.path.basename(metal_file)[:-6] + ".air")
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

        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metallib", *air_files, "-o", output],
            check=True,
        )
    return output


def get_build_parameters() -> SimpleNamespace:
    """Return setuptools extension parameters for the Metal bridge module."""

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
