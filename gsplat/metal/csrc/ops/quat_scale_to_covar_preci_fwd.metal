// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

#include "../metal_math.h"

using namespace metal;

kernel void quat_scale_to_covar_preci_fwd_kernel(
    device const float4* quats [[buffer(0)]],
    device const packed_float3* scales [[buffer(1)]],
    device float* covars [[buffer(2)]],
    device float* precis [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& triu [[buffer(5)]],
    constant uint& compute_covar [[buffer(6)]],
    constant uint& compute_preci [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }

    float4 quat = quats[id];
    float3 scale = scales[id];
    float3x3 R = quat_to_rotmat(quat);

    if (compute_covar != 0u && covars != nullptr) {
        float3x3 covar = covar_from_RS(R, scale);
        store_symmetric_matrix(covars, id, covar, triu);
    }
    if (compute_preci != 0u && precis != nullptr) {
        float3x3 preci = covar_from_RS(R, 1.0f / scale);
        store_symmetric_matrix(precis, id, preci, triu);
    }
}
