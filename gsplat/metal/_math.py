# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch math references shared by the Metal backend and its tests."""

from gsplat.cuda._math import _quat_scale_to_covar_preci
from gsplat.cuda._torch_impl import (
    _fisheye_proj,
    _isect_offset_encode,
    _ortho_proj,
    _persp_proj,
    _spherical_harmonics,
)

__all__ = [
    "_fisheye_proj",
    "_isect_offset_encode",
    "_ortho_proj",
    "_persp_proj",
    "_quat_scale_to_covar_preci",
    "_spherical_harmonics",
]
