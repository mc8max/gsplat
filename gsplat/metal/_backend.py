# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import threading
import warnings

import torch

_metal_C = None
_loaded = False
_lock = threading.Lock()


def load() -> bool:
    """Load the compiled Metal extension and bundled metallib."""
    global _loaded, _metal_C
    
    with _lock:
        if _loaded:
            return _metal_C is not None
        
        _loaded = True

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
            warnings.warn(f"Unable to find lib file of gsplat_metal.metallib at {metallib}.")
            return False

        try:
            _C.load_library(metallib)
            _metal_C = _C
            return True
        except Exception as e:
            warnings.warn(f"Unable to load gsplat_metal.metallib at {metallib}: {e}")

        return False


def has_metal() -> bool:
    return load()
