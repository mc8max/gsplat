// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Implementation note:
//
// MPS (Metal Performance Shaders) internally converts integer tensors to
// floating-point for sort and gather operations. This means:
//
//   - torch.sort(int64, device=mps) loses the low bits (uses float64 → 52-bit mantissa)
//   - torch.sort(int32, device=mps) loses the low bits for values > 2^23 (uses float32)
//   - index_select / gather on int64/int32 MPS tensors suffer the same precision loss
//
// The 4-pass 16-bit LSD approach (described in the plan) is therefore NOT
// viable with existing MPS torch ops: even if the chunk argsort is exact for
// 16-bit values, applying the resulting order via index_select to the original
// int64 tensor produces wrong results.
//
// The only correct GPU-side implementation is a custom Metal radix sort kernel
// (see plan-isect-tile-gpu-sort.md, v2). Until that kernel is implemented,
// this file provides a CPU-side fallback that is wrapped in the module API
// so that intersect_tile.mm can adopt the clean `mps_lsd_sort` call site
// without any further changes when v2 lands.

#include <algorithm>
#include <numeric>
#include <torch/extension.h>
#include <utility>

#include "sort_int64.h"

namespace gsplat::metal {

std::pair<at::Tensor, at::Tensor> mps_lsd_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool stable
) {
    TORCH_CHECK(isect_ids.dim() == 1, "mps_lsd_sort: isect_ids must be 1-D");
    TORCH_CHECK(flatten_ids.dim() == 1, "mps_lsd_sort: flatten_ids must be 1-D");
    TORCH_CHECK(isect_ids.scalar_type() == at::kLong, "mps_lsd_sort: isect_ids must be int64");
    TORCH_CHECK(flatten_ids.scalar_type() == at::kInt, "mps_lsd_sort: flatten_ids must be int32");
    TORCH_CHECK(
        isect_ids.numel() == flatten_ids.numel(),
        "mps_lsd_sort: isect_ids and flatten_ids must have the same length");

    const int64_t n = isect_ids.numel();
    if (n == 0) {
        return {isect_ids, flatten_ids};
    }

    const at::Device device = isect_ids.device();

    // CPU sort: move data off MPS, sort, move back.
    // This is a temporary bridge until the custom Metal radix sort (v2) is
    // implemented. The API is stable so the caller requires no changes.
    at::Tensor keys_cpu = isect_ids.cpu();
    at::Tensor vals_cpu = flatten_ids.cpu();

    auto* keys = keys_cpu.data_ptr<int64_t>();
    auto* vals = vals_cpu.data_ptr<int32_t>();

    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    if (stable) {
        std::stable_sort(order.begin(), order.end(), [keys](int64_t a, int64_t b) {
            return keys[a] < keys[b];
        });
    } else {
        std::sort(order.begin(), order.end(), [keys](int64_t a, int64_t b) {
            return keys[a] < keys[b];
        });
    }

    at::Tensor sorted_keys = at::empty_like(keys_cpu);
    at::Tensor sorted_vals = at::empty_like(vals_cpu);
    auto* out_k = sorted_keys.data_ptr<int64_t>();
    auto* out_v = sorted_vals.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) {
        out_k[i] = keys[order[static_cast<size_t>(i)]];
        out_v[i] = vals[order[static_cast<size_t>(i)]];
    }

    return {sorted_keys.to(device), sorted_vals.to(device)};
}

}  // namespace gsplat::metal
