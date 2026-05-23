# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch math references shared by the Metal backend and its tests."""

from gsplat.cuda._math import _quat_scale_to_covar_preci
from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut
from gsplat.cuda._torch_impl import (
    _fully_fused_projection,
    _fisheye_proj,
    _isect_tiles,
    _isect_offset_encode,
    _ortho_proj,
    _persp_proj,
    _spherical_harmonics,
)
from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs

__all__ = [
    "_fully_fused_projection",
    "_fully_fused_projection_2dgs",
    "_fully_fused_projection_with_ut",
    "_fisheye_proj",
    "_isect_tiles",
    "_isect_offset_encode",
    "_ortho_proj",
    "_persp_proj",
    "_quat_scale_to_covar_preci",
    "_rasterize_to_pixels_eval3d",
    "_spherical_harmonics",
]
