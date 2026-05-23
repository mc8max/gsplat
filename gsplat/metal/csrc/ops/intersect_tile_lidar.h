// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_lidar_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    bool sort,
    bool segmented,
    bool packed,
    int64_t n_bins_azimuth,
    int64_t n_bins_elevation,
    int64_t cdf_resolution_azimuth,
    int64_t cdf_resolution_elevation,
    double angle_to_pixel_scaling_factor,
    double fov_horiz_start,
    double fov_horiz_span,
    double fov_vert_start,
    double fov_vert_span,
    double fov_eps,
    int64_t spinning_direction,
    const at::Tensor& cdf_elevation,
    const at::Tensor& cdf_dense_ray_mask,
    const at::Tensor& tiles_pack_info,
    const at::Tensor& tiles_to_elements_map
);

}  // namespace gsplat::metal
