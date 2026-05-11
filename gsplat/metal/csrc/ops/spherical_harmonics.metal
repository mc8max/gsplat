// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../sh_math.h"

kernel void spherical_harmonics_fwd_kernel(
    device const packed_float3* dirs    [[buffer(0)]],
    device const float*         coeffs  [[buffer(1)]],
    device const uchar*         masks   [[buffer(2)]],
    device       float*         colors  [[buffer(3)]],
    constant uint& n                    [[buffer(4)]],
    constant uint& k                    [[buffer(5)]],
    constant uint& degrees_to_use       [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n * 3u) {
        return;
    }
    uint elem_id = id / 3u;
    uint c = id % 3u;
    if (masks != nullptr && masks[elem_id] == 0u) {
        return;
    }

    float result = 0.0f;
    sh_coeffs_to_color_fast(
        degrees_to_use, c, float3(dirs[elem_id]), coeffs + elem_id * k * 3u, result
    );
    colors[elem_id * 3u + c] = result;
}

kernel void spherical_harmonics_bwd_kernel(
    device const packed_float3* dirs         [[buffer(0)]],
    device const float*         coeffs       [[buffer(1)]],
    device const uchar*         masks        [[buffer(2)]],
    device const float*         v_colors     [[buffer(3)]],
    device       float*         v_coeffs     [[buffer(4)]],
    device       float*         v_dirs       [[buffer(5)]],
    constant uint& n                         [[buffer(6)]],
    constant uint& k                         [[buffer(7)]],
    constant uint& degrees_to_use            [[buffer(8)]],
    constant uint& compute_v_dirs            [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }
    if (masks != nullptr && masks[id] == 0u) {
        return;
    }

    float3 v_dir_acc = float3(0.0f);
    device const float* coeffs_base = coeffs + id * k * 3u;
    device float* v_coeffs_base = v_coeffs + id * k * 3u;
    float3 dir = float3(dirs[id]);
    thread float3* v_dir_ptr = compute_v_dirs != 0u ? &v_dir_acc : nullptr;

    for (uint c = 0; c < 3u; ++c) {
        sh_coeffs_to_color_fast_vjp(
            degrees_to_use,
            c,
            dir,
            coeffs_base,
            v_colors[id * 3u + c],
            v_coeffs_base,
            v_dir_ptr
        );
    }

    if (compute_v_dirs != 0u && v_dirs != nullptr) {
        v_dirs[id * 3u + 0u] = v_dir_acc.x;
        v_dirs[id * 3u + 1u] = v_dir_acc.y;
        v_dirs[id * 3u + 2u] = v_dir_acc.z;
    }
}
