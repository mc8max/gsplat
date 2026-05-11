// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../projection_math.h"

constant uint kCameraModelPinhole = 0u;
constant uint kCameraModelOrtho = 1u;
constant uint kCameraModelFisheye = 2u;

kernel void projection_ewa_simple_fwd_kernel(
    device const float* means      [[buffer(0)]],
    device const float* covars     [[buffer(1)]],
    device const float* Ks         [[buffer(2)]],
    device       float* means2d    [[buffer(3)]],
    device       float* covars2d   [[buffer(4)]],
    constant uint& B               [[buffer(5)]],
    constant uint& C               [[buffer(6)]],
    constant uint& N               [[buffer(7)]],
    constant uint& width           [[buffer(8)]],
    constant uint& height          [[buffer(9)]],
    constant uint& camera_model    [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= B * C * N) {
        return;
    }

    uint bid = id / (C * N);
    uint cid = (id / N) % C;
    device const float* mean_ptr = means + id * 3u;
    device const float* covar_ptr = covars + id * 9u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;
    device float* mean2d_ptr = means2d + id * 2u;
    device float* covar2d_ptr = covars2d + id * 4u;

    float fx = K_ptr[0];
    float cx = K_ptr[2];
    float fy = K_ptr[4];
    float cy = K_ptr[5];
    float3 mean3d = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3x3 cov3d = load_mat3_row_major(covar_ptr);
    float2x2 cov2d;
    float2 mean2d_val;

    switch (camera_model) {
    case kCameraModelPinhole:
        persp_proj_metal(mean3d, cov3d, fx, fy, cx, cy, width, height, cov2d, mean2d_val);
        break;
    case kCameraModelOrtho:
        ortho_proj_metal(mean3d, cov3d, fx, fy, cx, cy, width, height, cov2d, mean2d_val);
        break;
    default:  // kCameraModelFisheye
        fisheye_proj_metal(mean3d, cov3d, fx, fy, cx, cy, width, height, cov2d, mean2d_val);
        break;
    }

    mean2d_ptr[0] = mean2d_val.x;
    mean2d_ptr[1] = mean2d_val.y;
    store_mat2_row_major(covar2d_ptr, cov2d);
}

kernel void projection_ewa_simple_bwd_kernel(
    device const float* means        [[buffer(0)]],
    device const float* covars       [[buffer(1)]],
    device const float* Ks           [[buffer(2)]],
    device const float* v_means2d    [[buffer(3)]],
    device const float* v_covars2d   [[buffer(4)]],
    device       float* v_means      [[buffer(5)]],
    device       float* v_covars     [[buffer(6)]],
    constant uint& B                 [[buffer(7)]],
    constant uint& C                 [[buffer(8)]],
    constant uint& N                 [[buffer(9)]],
    constant uint& width             [[buffer(10)]],
    constant uint& height            [[buffer(11)]],
    constant uint& camera_model      [[buffer(12)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= B * C * N) {
        return;
    }

    uint bid = id / (C * N);
    uint cid = (id / N) % C;
    device const float* mean_ptr = means + id * 3u;
    device const float* covar_ptr = covars + id * 9u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;
    device const float* v_mean2d_ptr = v_means2d + id * 2u;
    device const float* v_covar2d_ptr = v_covars2d + id * 4u;
    device float* v_mean_ptr = v_means + id * 3u;
    device float* v_covar_ptr = v_covars + id * 9u;

    float fx = K_ptr[0];
    float cx = K_ptr[2];
    float fy = K_ptr[4];
    float cy = K_ptr[5];
    float3 mean3d = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3x3 cov3d = load_mat3_row_major(covar_ptr);
    float2 v_mean2d_val = float2(v_mean2d_ptr[0], v_mean2d_ptr[1]);
    float2x2 v_cov2d_val = load_mat2_row_major(v_covar2d_ptr);
    float3 v_mean3d = float3(0.0f);
    float3x3 v_cov3d = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));

    switch (camera_model) {
    case kCameraModelPinhole:
        persp_proj_vjp_metal(
            mean3d, cov3d, fx, fy, cx, cy, width, height, v_cov2d_val, v_mean2d_val, v_mean3d, v_cov3d
        );
        break;
    case kCameraModelOrtho:
        ortho_proj_vjp_metal(
            mean3d, cov3d, fx, fy, cx, cy, width, height, v_cov2d_val, v_mean2d_val, v_mean3d, v_cov3d
        );
        break;
    default:
        fisheye_proj_vjp_metal(
            mean3d, cov3d, fx, fy, cx, cy, width, height, v_cov2d_val, v_mean2d_val, v_mean3d, v_cov3d
        );
        break;
    }

    v_mean_ptr[0] = v_mean3d.x;
    v_mean_ptr[1] = v_mean3d.y;
    v_mean_ptr[2] = v_mean3d.z;
    store_mat3_row_major(v_covar_ptr, v_cov3d);
}
