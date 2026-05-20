// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd_op(
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
);

std::tuple<c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd_op(
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    bool absgrad
);

}  // namespace gsplat::metal
