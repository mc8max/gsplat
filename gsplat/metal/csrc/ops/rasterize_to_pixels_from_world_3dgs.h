// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_from_world_3dgs_fwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& rays,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const c10::optional<at::Tensor>& sample_counts,
    const c10::optional<at::Tensor>& render_normals,
    bool use_hit_distance
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_bwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& rays,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const c10::optional<at::Tensor>& v_render_normals,
    bool use_hit_distance
);

}  // namespace gsplat::metal
