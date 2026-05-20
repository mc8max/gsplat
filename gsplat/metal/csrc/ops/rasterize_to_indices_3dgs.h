// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs_op(
    int64_t range_start,
    int64_t range_end,
    const at::Tensor& transmittances,
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& opacities,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
);

}  // namespace gsplat::metal
