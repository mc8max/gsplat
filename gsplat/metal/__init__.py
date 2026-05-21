# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Python entry points for the gsplat Metal backend."""

from ._backend import has_metal
from ._wrapper import (
    distort_camera_rays,
    eval_bivariate_poly,
    fully_fused_projection,
    fully_fused_projection_2dgs,
    intersect_offset_encode,
    intersect_tile_count,
    intersect_tile_emit,
    intersect_tiles,
    metal_null,
    projection_ewa_simple,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels,
    rasterize_to_pixels_eval3d,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)

__all__ = [
    "distort_camera_rays",
    "eval_bivariate_poly",
    "fully_fused_projection",
    "fully_fused_projection_2dgs",
    "has_metal",
    "intersect_offset_encode",
    "intersect_tile_count",
    "intersect_tile_emit",
    "intersect_tiles",
    "metal_null",
    "projection_ewa_simple",
    "quat_scale_to_covar_preci",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels",
    "rasterize_to_pixels_eval3d",
    "rasterize_to_pixels_2dgs",
    "spherical_harmonics",
]
