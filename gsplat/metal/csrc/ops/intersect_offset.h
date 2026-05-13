// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

at::Tensor intersect_offset_op(
    const at::Tensor& isect_ids,
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
);

}  // namespace gsplat::metal
