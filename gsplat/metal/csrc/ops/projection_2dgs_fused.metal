// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../metal_math.h"
#include "../projection_math.h"

using namespace metal;

constant float kAabbExtent = 3.33f;

inline float sign_nonzero(float x) {
    return x >= 0.0f ? 1.0f : -1.0f;
}

inline float sum3(float3 v) {
    return v.x + v.y + v.z;
}

kernel void projection_2dgs_fused_fwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device int* radii [[buffer(5)]],
    device float* means2d [[buffer(6)]],
    device float* depths [[buffer(7)]],
    device float* ray_transforms [[buffer(8)]],
    device float* normals [[buffer(9)]],
    constant uint& B [[buffer(10)]],
    constant uint& C [[buffer(11)]],
    constant uint& N [[buffer(12)]],
    constant uint& image_width [[buffer(13)]],
    constant uint& image_height [[buffer(14)]],
    constant float& near_plane [[buffer(15)]],
    constant float& far_plane [[buffer(16)]],
    constant float& radius_clip [[buffer(17)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= B * C * N) {
        return;
    }

    const uint bid = id / (C * N);
    const uint cid = (id / N) % C;
    const uint gid = id % N;

    device const float* mean_ptr = means + (bid * N + gid) * 3u;
    device const float* quat_ptr = quats + (bid * N + gid) * 4u;
    device const float* scale_ptr = scales + (bid * N + gid) * 3u;
    device const float* viewmat_ptr = viewmats + (bid * C + cid) * 16u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;

    float3x3 R = float3x3(
        float3(viewmat_ptr[0], viewmat_ptr[4], viewmat_ptr[8]),
        float3(viewmat_ptr[1], viewmat_ptr[5], viewmat_ptr[9]),
        float3(viewmat_ptr[2], viewmat_ptr[6], viewmat_ptr[10])
    );
    float3 t = float3(viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]);

    float3 mean_c = R * float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]) + t;
    if (mean_c.z <= near_plane || mean_c.z >= far_plane) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    float3x3 rot = quat_to_rotmat(float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]));
    float3x3 RS_camera = R * (rot * make_diag(float3(scale_ptr[0], scale_ptr[1], scale_ptr[2])));

    float3 normal = RS_camera[2];
    normal *= sign_nonzero(-dot(normal, mean_c));

    // T_cl has columns [RS_camera[:, 0], RS_camera[:, 1], mean_c].
    float3x3 T_cl = float3x3(RS_camera[0], RS_camera[1], mean_c);
    float3x3 K = float3x3(
        float3(K_ptr[0], K_ptr[3], K_ptr[6]),
        float3(K_ptr[1], K_ptr[4], K_ptr[7]),
        float3(K_ptr[2], K_ptr[5], K_ptr[8])
    );
    float3x3 T_sl = K * T_cl;
    float3x3 M = transpose(T_sl);

    const float3 temp_point = float3(1.0f, 1.0f, -1.0f);
    float3 M0 = M[0];
    float3 M1 = M[1];
    float3 M2 = M[2];
    float distance = dot(temp_point * M2, M2);
    if (distance == 0.0f) {  // guard against zero denominator
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    float3 f = temp_point / distance;
    float2 mean2d_val = float2(dot(f * M0, M2), dot(f * M1, M2));
    float2 extents = sqrt(max(
        float2(1.0e-4f),
        mean2d_val * mean2d_val - float2(dot(f * M0, M0), dot(f * M1, M1))
    ));
    float2 radius = ceil(kAabbExtent * extents);
    if (radius.x <= radius_clip || radius.y <= radius_clip) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    bool inside =
        mean2d_val.x + radius.x > 0.0f &&
        mean2d_val.x - radius.x < float(image_width) &&
        mean2d_val.y + radius.y > 0.0f &&
        mean2d_val.y - radius.y < float(image_height);
    if (!inside) {
        radii[id * 2u] = 0;
        radii[id * 2u + 1u] = 0;
        return;
    }

    radii[id * 2u] = int(radius.x);
    radii[id * 2u + 1u] = int(radius.y);
    means2d[id * 2u] = mean2d_val.x;
    means2d[id * 2u + 1u] = mean2d_val.y;
    depths[id] = mean_c.z;
    store_mat3_row_major(ray_transforms + id * 9u, T_sl);
    normals[id * 3u] = normal.x;
    normals[id * 3u + 1u] = normal.y;
    normals[id * 3u + 2u] = normal.z;
}

kernel void projection_2dgs_fused_bwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device const int* radii [[buffer(5)]],
    device const float* ray_transforms [[buffer(6)]],
    device const float* v_means2d [[buffer(7)]],
    device const float* v_depths [[buffer(8)]],
    device const float* v_normals [[buffer(9)]],
    device const float* v_ray_transforms [[buffer(10)]],
    device float* tmp_means [[buffer(11)]],
    device float* tmp_quats [[buffer(12)]],
    device float* tmp_scales [[buffer(13)]],
    device float* tmp_viewmats [[buffer(14)]],
    constant uint& B [[buffer(15)]],
    constant uint& C [[buffer(16)]],
    constant uint& N [[buffer(17)]],
    constant uint& image_width [[buffer(18)]],
    constant uint& image_height [[buffer(19)]],
    constant uint& viewmats_requires_grad [[buffer(20)]],
    uint id [[thread_position_in_grid]]
) {
    (void)image_width;
    (void)image_height;
    if (id >= B * C * N || radii[id * 2u] <= 0 || radii[id * 2u + 1u] <= 0) {
        return;
    }

    const uint bid = id / (C * N);
    const uint cid = (id / N) % C;
    const uint gid = id % N;

    device const float* mean_ptr = means + (bid * N + gid) * 3u;
    device const float* quat_ptr = quats + (bid * N + gid) * 4u;
    device const float* scale_ptr = scales + (bid * N + gid) * 3u;
    device const float* viewmat_ptr = viewmats + (bid * C + cid) * 16u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;
    device const float* ray_transform_ptr = ray_transforms + id * 9u;
    device const float* v_means2d_ptr = v_means2d + id * 2u;
    device const float* v_normals_ptr = v_normals + id * 3u;
    device const float* v_ray_transform_ptr = v_ray_transforms + id * 9u;

    float3x3 R = float3x3(
        float3(viewmat_ptr[0], viewmat_ptr[4], viewmat_ptr[8]),
        float3(viewmat_ptr[1], viewmat_ptr[5], viewmat_ptr[9]),
        float3(viewmat_ptr[2], viewmat_ptr[6], viewmat_ptr[10])
    );
    float3 t = float3(viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]);
    float3 mean_w = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3 mean_c = R * mean_w + t;

    float4 quat = float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
    float2 scale = float2(scale_ptr[0], scale_ptr[1]);
    // Match the CUDA helper exactly: this is not the usual row-major K matrix,
    // but the specific column-major layout used by compute_ray_transforms_aabb_vjp.
    float3x3 P = float3x3(
        float3(K_ptr[0], 0.0f, K_ptr[2]),
        float3(0.0f, K_ptr[4], K_ptr[5]),
        float3(0.0f, 0.0f, 1.0f)
    );

    float3x3 v_rt = load_mat3_row_major(v_ray_transform_ptr);
    v_rt[2][2] += v_depths[id];
    float3 v_normal = float3(v_normals_ptr[0], v_normals_ptr[1], v_normals_ptr[2]);

    float3 v_mean = float3(0.0f);
    float2 v_scale = float2(0.0f);
    float4 v_quat = float4(0.0f);
    float3x3 v_R = float3x3(0.0f);
    float3 v_t = float3(0.0f);
    compute_ray_transforms_aabb_vjp_metal(
        ray_transform_ptr,
        v_means2d_ptr,
        v_normal,
        R,
        P,
        t,
        mean_w,
        mean_c,
        quat,
        scale,
        v_rt,
        v_quat,
        v_scale,
        v_mean,
        v_R,
        v_t
    );

    device float* tmp_mean_ptr = tmp_means + id * 3u;
    tmp_mean_ptr[0] = v_mean.x;
    tmp_mean_ptr[1] = v_mean.y;
    tmp_mean_ptr[2] = v_mean.z;

    device float* tmp_quat_ptr = tmp_quats + id * 4u;
    tmp_quat_ptr[0] = v_quat.x;
    tmp_quat_ptr[1] = v_quat.y;
    tmp_quat_ptr[2] = v_quat.z;
    tmp_quat_ptr[3] = v_quat.w;

    device float* tmp_scale_ptr = tmp_scales + id * 3u;
    tmp_scale_ptr[0] = v_scale.x;
    tmp_scale_ptr[1] = v_scale.y;
    tmp_scale_ptr[2] = 0.0f;

    if (viewmats_requires_grad != 0u && tmp_viewmats != nullptr) {
        device float* tmp_viewmat_ptr = tmp_viewmats + id * 16u;
        tmp_viewmat_ptr[0] = v_R[0][0];
        tmp_viewmat_ptr[1] = v_R[1][0];
        tmp_viewmat_ptr[2] = v_R[2][0];
        tmp_viewmat_ptr[3] = v_t.x;
        tmp_viewmat_ptr[4] = v_R[0][1];
        tmp_viewmat_ptr[5] = v_R[1][1];
        tmp_viewmat_ptr[6] = v_R[2][1];
        tmp_viewmat_ptr[7] = v_t.y;
        tmp_viewmat_ptr[8] = v_R[0][2];
        tmp_viewmat_ptr[9] = v_R[1][2];
        tmp_viewmat_ptr[10] = v_R[2][2];
        tmp_viewmat_ptr[11] = v_t.z;
        tmp_viewmat_ptr[12] = 0.0f;
        tmp_viewmat_ptr[13] = 0.0f;
        tmp_viewmat_ptr[14] = 0.0f;
        tmp_viewmat_ptr[15] = 0.0f;
    }
}
