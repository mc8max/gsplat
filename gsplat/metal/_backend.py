# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import torch

_metal_C = None
_loaded = False


def load() -> bool:
    """Load the compiled Metal extension and bundled metallib."""
    global _loaded, _metal_C
    if _loaded:
        return _metal_C is not None

    if sys.platform != "darwin":
        return False
    if not torch.backends.mps.is_available():
        return False

    try:
        from gsplat import metal_ext as _C
    except ImportError:
        return False

    metallib = os.path.join(os.path.dirname(__file__), "gsplat_metal.metallib")
    if not os.path.exists(metallib):
        raise FileNotFoundError(
            f"gsplat Metal library not found at {metallib}. "
            "Re-run: BUILD_NO_CUDA=1 pip install -e . --no-build-isolation"
        )

    _C.load_library(metallib)
    _metal_C = _C
    _loaded = True
    return True


def has_metal() -> bool:
    return load()
