// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_2dgs_fwd_op(
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    bool packed
);

std::tuple<c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_2dgs_bwd_op(
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const at::Tensor& densify,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_colors,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& median_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const at::Tensor& v_render_normals,
    const at::Tensor& v_render_distort,
    const at::Tensor& v_render_median,
    bool packed,
    bool absgrad
);

}  // namespace gsplat::metal
