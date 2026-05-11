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

kernel void quat_scale_to_covar_preci_bwd_kernel(
    device const float4* quats [[buffer(0)]],
    device const packed_float3* scales [[buffer(1)]],
    device const float* v_covars [[buffer(2)]],
    device const float* v_precis [[buffer(3)]],
    device float4* v_quats [[buffer(4)]],
    device packed_float3* v_scales [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& triu [[buffer(7)]],
    constant uint& has_v_covars [[buffer(8)]],
    constant uint& has_v_precis [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }

    float4 quat = quats[id];
    float3 scale = scales[id];
    float3x3 R = quat_to_rotmat(quat);

    float4 v_quat = float4(0.0f);
    float3 v_scale = float3(0.0f);

    if (has_v_covars != 0u && v_covars != nullptr) {
        float3x3 v_covar = load_symmetric_grad(v_covars, id, triu);
        float3x3 v_R = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
        float3 v_s_local = float3(0.0f);
        covar_from_RS_vjp(R, scale, v_covar, v_R, v_s_local);
        v_quat += quat_to_rotmat_vjp(quat, v_R);
        v_scale += v_s_local;
    }

    if (has_v_precis != 0u && v_precis != nullptr) {
        float3 inv_scale = 1.0f / scale;
        float3x3 v_preci = load_symmetric_grad(v_precis, id, triu);
        float3x3 v_R = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
        float3 v_inv_scale = float3(0.0f);
        covar_from_RS_vjp(R, inv_scale, v_preci, v_R, v_inv_scale);
        v_quat += quat_to_rotmat_vjp(quat, v_R);
        v_scale -= inv_scale * inv_scale * v_inv_scale;
    }

    v_quats[id] = v_quat;
    v_scales[id] = v_scale;
}
