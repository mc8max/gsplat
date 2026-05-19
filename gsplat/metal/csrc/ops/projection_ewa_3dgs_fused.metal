// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../metal_math.h"
#include "../projection_math.h"

using namespace metal;

constant uint kCameraModelPinhole = 0u;
constant uint kCameraModelOrtho = 1u;
constant uint kCameraModelFisheye = 2u;
constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kGaussianExtend = 3.33f;
constant float kMinCompensation = 0.25f;

inline float3x3 load_mat3_triu(device const float* src) {
    return float3x3(
        float3(src[0], src[1], src[2]),
        float3(src[1], src[3], src[4]),
        float3(src[2], src[4], src[5])
    );
}

inline float3 pos_w2c(float3x3 R, float3 t, float3 p_w) {
    return R * p_w + t;
}

inline float3x3 covar_w2c(float3x3 R, float3x3 covar_w) {
    return R * covar_w * transpose(R);
}

inline float add_blur_metal(float eps2d, thread float2x2& covar, thread float& compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(kMinCompensation * kMinCompensation, det_orig / det_blur));
    return det_blur;
}

kernel void projection_ewa_3dgs_fused_fwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* covars [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device int* radii [[buffer(5)]],
    device float* means2d [[buffer(6)]],
    device float* depths [[buffer(7)]],
    device float* conics [[buffer(8)]],
    device float* compensations [[buffer(9)]],
    constant uint& B [[buffer(10)]],
    constant uint& C [[buffer(11)]],
    constant uint& N [[buffer(12)]],
    constant uint& image_width [[buffer(13)]],
    constant uint& image_height [[buffer(14)]],
    constant float& eps2d [[buffer(15)]],
    constant float& near_plane [[buffer(16)]],
    constant float& far_plane [[buffer(17)]],
    constant float& radius_clip [[buffer(18)]],
    constant uint& camera_model [[buffer(19)]],
    constant uint& has_opacities [[buffer(20)]],
    constant uint& has_compensations [[buffer(21)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= B * C * N) {
        return;
    }

    const uint bid = id / (C * N);
    const uint cid = (id / N) % C;
    const uint gid = id % N;

    device const float* mean_ptr = means + (bid * N + gid) * 3u;
    device const float* covar_ptr = covars + (bid * N + gid) * 6u;
    device const float* viewmat_ptr = viewmats + (bid * C + cid) * 16u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;

    float3x3 R = float3x3(
        float3(viewmat_ptr[0], viewmat_ptr[4], viewmat_ptr[8]),
        float3(viewmat_ptr[1], viewmat_ptr[5], viewmat_ptr[9]),
        float3(viewmat_ptr[2], viewmat_ptr[6], viewmat_ptr[10])
    );
    float3 t = float3(viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]);

    float3 mean_c = pos_w2c(R, t, float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]));
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    float3x3 covar_c = covar_w2c(R, load_mat3_triu(covar_ptr));

    float2x2 covar2d;
    float2 mean2d_val;
    float fx = K_ptr[0];
    float fy = K_ptr[4];
    float cx = K_ptr[2];
    float cy = K_ptr[5];
    switch (camera_model) {
    case kCameraModelPinhole:
        persp_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, mean2d_val);
        break;
    case kCameraModelOrtho:
        ortho_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, mean2d_val);
        break;
    default:
        fisheye_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, mean2d_val);
        break;
    }

    float compensation = 1.0f;
    float det = add_blur_metal(eps2d, covar2d, compensation);
    if (det <= 0.0f) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    float inv_det = 1.0f / det;
    float2x2 covar2d_inv = float2x2(
        float2(covar2d[1][1] * inv_det, -covar2d[0][1] * inv_det),
        float2(-covar2d[1][0] * inv_det, covar2d[0][0] * inv_det)
    );
    float extend = kGaussianExtend;
    if (has_opacities != 0u) {
        float opacity = opacities[bid * N + gid];
        if (has_compensations != 0u) {
            opacity *= compensation;
        }
        if (opacity < kAlphaThreshold) {
            radii[id * 2u] = 0;
            radii[id * 2u + 1u] = 0;
            return;
        }
        extend = min(kGaussianExtend, sqrt(max(0.0f, 2.0f * log(opacity / kAlphaThreshold))));
    }

    float radius_x = ceil(extend * sqrt(max(0.0f, covar2d[0][0])));
    float radius_y = ceil(extend * sqrt(max(0.0f, covar2d[1][1])));
    if (radius_x <= radius_clip || radius_y <= radius_clip) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    if (mean2d_val.x + radius_x <= 0.0f || mean2d_val.x - radius_x >= float(image_width) ||
        mean2d_val.y + radius_y <= 0.0f || mean2d_val.y - radius_y >= float(image_height)) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    radii[id * 2u] = int(radius_x);
    radii[id * 2u + 1u] = int(radius_y);
    means2d[id * 2u] = mean2d_val.x;
    means2d[id * 2u + 1u] = mean2d_val.y;
    depths[id] = mean_c.z;
    conics[id * 3u] = covar2d_inv[0][0];
    conics[id * 3u + 1u] = covar2d_inv[0][1];
    conics[id * 3u + 2u] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[id] = compensation;
    }
}
