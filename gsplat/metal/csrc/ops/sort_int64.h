// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <utility>

namespace gsplat::metal {

// Sort (isect_ids, flatten_ids) pairs by isect_ids key in ascending order.
//
// Both tensors must be 1-D, contiguous, and on the same MPS device.
//   isect_ids  : int64   — packed sort key ((image_id | tile_id) << 32 | depth_bits)
//   flatten_ids: int32   — payload co-sorted with the key
//
// v1  mps_lsd_sort — CPU fallback wrapped in the module API.  Correct and
//                    always available; incurs PCIe transfer.  Will be replaced
//                    by radix_sort once v2 is validated.
//
// v2  radix_sort   — custom Metal 8-pass LSD radix sort. O(n) per pass,
//                    entirely on GPU, inherently stable, no PCIe transfer.

// v1: CPU sort behind module API.
std::pair<at::Tensor, at::Tensor> mps_lsd_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool stable = true
);

// v2: custom Metal radix sort.
std::pair<at::Tensor, at::Tensor> radix_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool stable = true
);

}  // namespace gsplat::metal
