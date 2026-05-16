// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

at::Tensor intersect_tile_count_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool packed,
    bool segmented
);

std::tuple<at::Tensor, at::Tensor> intersect_tile_emit_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    const at::Tensor& cum_tiles_per_gauss,
    bool packed,
    bool segmented
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool packed,
    bool segmented
);

}  // namespace gsplat::metal
