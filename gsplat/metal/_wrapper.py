# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from ._backend import load


def metal_null(x: torch.Tensor) -> torch.Tensor:
    """Identity op on MPS backed by a Metal kernel."""
    if not load():
        raise RuntimeError("gsplat Metal extension is not available.")
    return torch.ops.gsplat.metal_null(x)
