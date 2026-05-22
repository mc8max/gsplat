// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../metal_math.h"
#include "../projection_math.h"

using namespace metal;

constant float kAabbExtent = 3.33f;
constant uint kThreadsPacked = 256u;

inline float sign_nonzero(float x) {
    return x >= 0.0f ? 1.0f : -1.0f;
}

inline float sum3(float3 v) {
    return v.x + v.y + v.z;
}

// Forward math for a single Gaussian candidate: identical to projection_2dgs_fused.metal
// but returns a struct instead of writing to buffers.
struct Packed2DGSResult {
    int2 radii;
    float2 means2d;
    float depth;
    float3 normal;
    // ray_transforms stored as 9 floats (row-major 3x3)
    float rt00, rt01, rt02;
    float rt10, rt11, rt12;
    float rt20, rt21, rt22;
};

inline bool project_2dgs_candidate(
    uint bid,
    uint cid,
    uint gid,
    device const float* means,
    device const float* quats,
    device const float* scales,
    device const float* viewmats,
    device const float* Ks,
    uint N,
    uint C,
    uint image_width,
    uint image_height,
    float near_plane,
    float far_plane,
    float radius_clip,
    thread Packed2DGSResult& out
) {
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
        return false;
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
    if (distance == 0.0f) {
        return false;
    }

    float3 f = temp_point / distance;
    float2 mean2d_val = float2(dot(f * M0, M2), dot(f * M1, M2));
    float2 extents = sqrt(max(
        float2(1.0e-4f),
        mean2d_val * mean2d_val - float2(dot(f * M0, M0), dot(f * M1, M1))
    ));
    float2 radius = ceil(kAabbExtent * extents);
    if (radius.x <= radius_clip || radius.y <= radius_clip) {
        return false;
    }

    bool inside =
        mean2d_val.x + radius.x > 0.0f &&
        mean2d_val.x - radius.x < float(image_width) &&
        mean2d_val.y + radius.y > 0.0f &&
        mean2d_val.y - radius.y < float(image_height);
    if (!inside) {
        return false;
    }

    out.radii = int2(int(radius.x), int(radius.y));
    out.means2d = mean2d_val;
    out.depth = mean_c.z;
    out.normal = normal;
    out.rt00 = T_sl[0][0];
    out.rt01 = T_sl[0][1];
    out.rt02 = T_sl[0][2];
    out.rt10 = T_sl[1][0];
    out.rt11 = T_sl[1][1];
    out.rt12 = T_sl[1][2];
    out.rt20 = T_sl[2][0];
    out.rt21 = T_sl[2][1];
    out.rt22 = T_sl[2][2];
    return true;
}

// Count kernel: parallel tree reduction within threadgroup
kernel void projection_2dgs_packed_count_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device int* block_cnts [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& image_width [[buffer(9)]],
    constant uint& image_height [[buffer(10)]],
    constant float& near_plane [[buffer(11)]],
    constant float& far_plane [[buffer(12)]],
    constant float& radius_clip [[buffer(13)]],
    constant uint& blocks_per_row [[buffer(14)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = tgp.y;
    const uint block_col = tgp.x;
    const uint gid = block_col * kThreadsPacked + tid;
    const uint block_idx = row * blocks_per_row + block_col;

    threadgroup int flags[256];

    bool valid = false;
    if (row < B * C && gid < N) {
        const uint bid = row / C;
        const uint cid = row % C;
        Packed2DGSResult out;
        valid = project_2dgs_candidate(
            bid, cid, gid,
            means, quats, scales, viewmats, Ks,
            N, C, image_width, image_height,
            near_plane, far_plane, radius_clip,
            out
        );
    }
    flags[tid] = valid ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel tree reduction: 8 barrier steps for 256 elements.
    for (uint s = 128u; s > 0u; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s) flags[tid] += flags[tid + s];
    }
    if (tid == 0u) {
        block_cnts[block_idx] = flags[0];
    }
}

// Emit kernel: exclusive scan + write packed outputs
kernel void projection_2dgs_packed_emit_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device const int* block_accum [[buffer(5)]],
    device long* batch_ids [[buffer(6)]],
    device long* camera_ids [[buffer(7)]],
    device long* gaussian_ids [[buffer(8)]],
    device int* radii [[buffer(9)]],
    device float* means2d [[buffer(10)]],
    device float* depths [[buffer(11)]],
    device float* ray_transforms [[buffer(12)]],
    device float* normals [[buffer(13)]],
    constant uint& B [[buffer(14)]],
    constant uint& C [[buffer(15)]],
    constant uint& N [[buffer(16)]],
    constant uint& image_width [[buffer(17)]],
    constant uint& image_height [[buffer(18)]],
    constant float& near_plane [[buffer(19)]],
    constant float& far_plane [[buffer(20)]],
    constant float& radius_clip [[buffer(21)]],
    constant uint& blocks_per_row [[buffer(22)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = tgp.y;
    const uint block_col = tgp.x;
    const uint gid = block_col * kThreadsPacked + tid;
    const uint block_idx = row * blocks_per_row + block_col;

    threadgroup int flags[256];
    threadgroup int prefix[256];

    Packed2DGSResult out;
    bool valid = false;
    if (row < B * C && gid < N) {
        const uint bid = row / C;
        const uint cid = row % C;
        valid = project_2dgs_candidate(
            bid, cid, gid,
            means, quats, scales, viewmats, Ks,
            N, C, image_width, image_height,
            near_plane, far_plane, radius_clip,
            out
        );
    }
    flags[tid] = valid ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential exclusive scan: for N=256 this is simpler than Blelloch.
    if (tid == 0u) {
        int running = 0;
        for (uint i = 0; i < 256u; ++i) {
            prefix[i] = running;
            running += flags[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!valid) {
        return;
    }

    const int offset = block_idx == 0u ? 0 : block_accum[block_idx - 1u];
    const uint slot = uint(offset + prefix[tid]);
    const uint bid = row / C;
    const uint cid = row % C;

    batch_ids[slot] = long(bid);
    camera_ids[slot] = long(cid);
    gaussian_ids[slot] = long(gid);
    radii[slot * 2u] = out.radii.x;
    radii[slot * 2u + 1u] = out.radii.y;
    means2d[slot * 2u] = out.means2d.x;
    means2d[slot * 2u + 1u] = out.means2d.y;
    depths[slot] = out.depth;
    ray_transforms[slot * 9u] = out.rt00;
    ray_transforms[slot * 9u + 1u] = out.rt10;
    ray_transforms[slot * 9u + 2u] = out.rt20;
    ray_transforms[slot * 9u + 3u] = out.rt01;
    ray_transforms[slot * 9u + 4u] = out.rt11;
    ray_transforms[slot * 9u + 5u] = out.rt21;
    ray_transforms[slot * 9u + 6u] = out.rt02;
    ray_transforms[slot * 9u + 7u] = out.rt12;
    ray_transforms[slot * 9u + 8u] = out.rt22;
    normals[slot * 3u] = out.normal.x;
    normals[slot * 3u + 1u] = out.normal.y;
    normals[slot * 3u + 2u] = out.normal.z;
}

// Backward kernel: one thread per nnz element
kernel void projection_2dgs_packed_bwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device const long* batch_ids [[buffer(5)]],
    device const long* camera_ids [[buffer(6)]],
    device const long* gaussian_ids [[buffer(7)]],
    device const float* ray_transforms [[buffer(8)]],
    device const float* v_means2d [[buffer(9)]],
    device const float* v_depths [[buffer(10)]],
    device const float* v_ray_transforms [[buffer(11)]],
    device const float* v_normals [[buffer(12)]],
    device float* tmp_means [[buffer(13)]],
    device float* tmp_quats [[buffer(14)]],
    device float* tmp_scales [[buffer(15)]],
    device float* tmp_viewmats [[buffer(16)]],
    constant uint& B [[buffer(17)]],
    constant uint& C [[buffer(18)]],
    constant uint& N [[buffer(19)]],
    constant uint& nnz [[buffer(20)]],
    constant uint& image_width [[buffer(21)]],
    constant uint& image_height [[buffer(22)]],
    constant uint& viewmats_requires_grad [[buffer(23)]],
    uint id [[thread_position_in_grid]]
) {
    (void)image_width;
    (void)image_height;
    if (id >= nnz) {
        return;
    }

    const uint bid = uint(batch_ids[id]);
    const uint cid = uint(camera_ids[id]);
    const uint gid = uint(gaussian_ids[id]);

    device const float* mean_ptr = means + (bid * N + gid) * 3u;
    device const float* quat_ptr = quats + (bid * N + gid) * 4u;
    device const float* scale_ptr = scales + (bid * N + gid) * 3u;
    device const float* viewmat_ptr = viewmats + (bid * C + cid) * 16u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;
    device const float* rt_ptr = ray_transforms + id * 9u;
    device const float* v_mean2d_ptr = v_means2d + id * 2u;
    device const float* v_depth_ptr = v_depths + id;
    device const float* v_rt_ptr = v_ray_transforms + id * 9u;
    device const float* v_norm_ptr = v_normals + id * 3u;

    float3x3 R = float3x3(
        float3(viewmat_ptr[0], viewmat_ptr[4], viewmat_ptr[8]),
        float3(viewmat_ptr[1], viewmat_ptr[5], viewmat_ptr[9]),
        float3(viewmat_ptr[2], viewmat_ptr[6], viewmat_ptr[10])
    );
    float3 t = float3(viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]);
    float3 mean_w = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3 mean_c = R * mean_w + t;

    float4 quat = float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
    float3 scale = float3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);

    // P matrix: match CUDA compute_ray_transforms_aabb_vjp layout
    float3x3 P = float3x3(
        float3(K_ptr[0], 0.0f, K_ptr[2]),
        float3(0.0f, K_ptr[4], K_ptr[5]),
        float3(0.0f, 0.0f, 1.0f)
    );

    // Accumulate v_depth into v_ray_transforms[2][2]
    float3x3 v_rt = float3x3(
        float3(v_rt_ptr[0], v_rt_ptr[1], v_rt_ptr[2]),
        float3(v_rt_ptr[3], v_rt_ptr[4], v_rt_ptr[5]),
        float3(v_rt_ptr[6], v_rt_ptr[7], v_rt_ptr[8])
    );
    v_rt[2][2] += v_depth_ptr[0];

    float3 v_normal = float3(v_norm_ptr[0], v_norm_ptr[1], v_norm_ptr[2]);

    float3 v_mean = float3(0.0f);
    float3 v_scale = float3(0.0f);
    float4 v_quat = float4(0.0f);
    float3x3 v_R = float3x3(0.0f);
    float3 v_t = float3(0.0f);

    compute_ray_transforms_aabb_vjp_metal(
        rt_ptr,
        v_mean2d_ptr,
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

    // Write to tmp buffers
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
    tmp_scale_ptr[2] = v_scale.z;

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
