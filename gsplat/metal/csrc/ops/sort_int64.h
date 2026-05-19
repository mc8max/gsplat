// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <utility>

namespace gsplat::metal {

// Sort `(isect_ids, flatten_ids)` pairs by `isect_ids` in ascending order.
//
// Both tensors must be 1-D, contiguous, and on the same MPS device.
//   isect_ids  : int64   — packed sort key `((image_id | tile_id) << 32 | depth_bits)`
//   flatten_ids: int32   — payload co-sorted with the key
//
// Current status:
// - `radix_sort(...)` is the active GPU-native path used by `intersect_tile`
// - `mps_lsd_sort(...)` remains as an alternate helper/reference path
//
// The GPU radix path is stable, so sorting by the full 64-bit key preserves the
// same final ordering contract for both global and segmented `intersect_tile`
// outputs.

// Alternate helper path kept for validation / fallback experimentation.
std::pair<at::Tensor, at::Tensor> mps_lsd_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool stable = true
);

// Active GPU-native sort: custom Metal LSD radix sort.
std::pair<at::Tensor, at::Tensor> radix_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool stable = true
);

}  // namespace gsplat::metal
