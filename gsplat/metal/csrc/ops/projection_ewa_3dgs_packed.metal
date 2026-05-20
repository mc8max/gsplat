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
// Must match kThreadsPacked in projection_ewa_3dgs_packed.mm.
constant uint kThreadsPacked = 256u;

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

inline void pos_w2c_vjp(
    float3x3 R,
    float3 p_w,
    float3 v_p_c,
    thread float3x3& v_R,
    thread float3& v_t,
    thread float3& v_p_w
) {
    v_R += outer3(v_p_c, p_w);
    v_t += v_p_c;
    v_p_w += transpose(R) * v_p_c;
}

inline void covar_w2c_vjp(
    float3x3 R,
    float3x3 covar_w,
    float3x3 v_covar_c,
    thread float3x3& v_R,
    thread float3x3& v_covar_w
) {
    v_R += v_covar_c * R * transpose(covar_w) + transpose(v_covar_c) * R * covar_w;
    v_covar_w += transpose(R) * v_covar_c * R;
}

inline float add_blur_metal(float eps2d, thread float2x2& covar, thread float& compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(kMinCompensation * kMinCompensation, det_orig / det_blur));
    return det_blur;
}

inline void inverse_vjp_2x2(
    float2x2 Minv,
    float2x2 v_Minv,
    thread float2x2& v_M
) {
    v_M += (-1.0f) * (Minv * v_Minv * Minv);
}

inline void add_blur_vjp_metal(
    float eps2d,
    float2x2 conic_blur,
    float compensation,
    float v_compensation,
    thread float2x2& v_covar
) {
    float det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    float v_sqr_comp = v_compensation * 0.5f / (compensation + 1e-6f);
    float one_minus_sqr_comp = 1.0f - compensation * compensation;
    v_covar[0][0] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

inline void store_triu3(device float* dst, float3x3 M) {
    dst[0] = M[0][0];
    dst[1] = M[1][0] + M[0][1];
    dst[2] = M[2][0] + M[0][2];
    dst[3] = M[1][1];
    dst[4] = M[2][1] + M[1][2];
    dst[5] = M[2][2];
}

struct PackedProjectionResult {
    int2 radii;
    float2 means2d;
    float depth;
    float3 conic;
    float compensation;
};

inline bool project_packed_candidate(
    uint bid,
    uint cid,
    uint gid,
    device const float* means,
    device const float* covars,
    device const float* opacities,
    device const float* viewmats,
    device const float* Ks,
    uint N,
    uint C,
    uint image_width,
    uint image_height,
    float eps2d,
    float near_plane,
    float far_plane,
    float radius_clip,
    uint camera_model,
    uint has_opacities,
    uint has_compensations,
    thread PackedProjectionResult& out
) {
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
        return false;
    }

    float3x3 covar_c = covar_w2c(R, load_mat3_triu(covar_ptr));

    float2x2 covar2d;
    float2 means2d;
    float fx = K_ptr[0];
    float fy = K_ptr[4];
    float cx = K_ptr[2];
    float cy = K_ptr[5];
    switch (camera_model) {
    case kCameraModelPinhole:
        persp_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, means2d);
        break;
    case kCameraModelOrtho:
        ortho_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, means2d);
        break;
    default:
        fisheye_proj_metal(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, covar2d, means2d);
        break;
    }

    float compensation = 1.0f;
    float det = add_blur_metal(eps2d, covar2d, compensation);
    if (det <= 0.0f) {
        return false;
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
            return false;
        }
        extend = min(kGaussianExtend, sqrt(max(0.0f, 2.0f * log(opacity / kAlphaThreshold))));
    }

    float radius_x = ceil(extend * sqrt(max(0.0f, covar2d[0][0])));
    float radius_y = ceil(extend * sqrt(max(0.0f, covar2d[1][1])));
    if (radius_x <= radius_clip || radius_y <= radius_clip) {
        return false;
    }

    if (means2d.x + radius_x <= 0.0f || means2d.x - radius_x >= float(image_width) ||
        means2d.y + radius_y <= 0.0f || means2d.y - radius_y >= float(image_height)) {
        return false;
    }

    out.radii = int2(int(radius_x), int(radius_y));
    out.means2d = means2d;
    out.depth = mean_c.z;
    out.conic = float3(covar2d_inv[0][0], covar2d_inv[0][1], covar2d_inv[1][1]);
    out.compensation = compensation;
    return true;
}

kernel void projection_ewa_3dgs_packed_count_kernel(
    device const float* means [[buffer(0)]],
    device const float* covars [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device int* block_cnts [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& image_width [[buffer(9)]],
    constant uint& image_height [[buffer(10)]],
    constant float& eps2d [[buffer(11)]],
    constant float& near_plane [[buffer(12)]],
    constant float& far_plane [[buffer(13)]],
    constant float& radius_clip [[buffer(14)]],
    constant uint& camera_model [[buffer(15)]],
    constant uint& has_opacities [[buffer(16)]],
    constant uint& has_compensations [[buffer(17)]],
    constant uint& blocks_per_row [[buffer(18)]],
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
        PackedProjectionResult out;
        valid = project_packed_candidate(
            bid,
            cid,
            gid,
            means,
            covars,
            opacities,
            viewmats,
            Ks,
            N,
            C,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            camera_model,
            has_opacities,
            has_compensations,
            out
        );
    }
    flags[tid] = valid ? 1 : 0;
    // Parallel tree reduction: 8 barrier steps for 256 elements.
    for (uint s = 128u; s > 0u; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s) flags[tid] += flags[tid + s];
    }
    if (tid == 0u) {
        block_cnts[block_idx] = flags[0];
    }
}

kernel void projection_ewa_3dgs_packed_emit_kernel(
    device const float* means [[buffer(0)]],
    device const float* covars [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const float* viewmats [[buffer(3)]],
    device const float* Ks [[buffer(4)]],
    device const int* block_accum [[buffer(5)]],
    device long* batch_ids [[buffer(6)]],
    device long* camera_ids [[buffer(7)]],
    device long* gaussian_ids [[buffer(8)]],
    device int* radii [[buffer(9)]],
    device float* means2d [[buffer(10)]],
    device float* depths [[buffer(11)]],
    device float* conics [[buffer(12)]],
    device float* compensations [[buffer(13)]],
    constant uint& B [[buffer(14)]],
    constant uint& C [[buffer(15)]],
    constant uint& N [[buffer(16)]],
    constant uint& image_width [[buffer(17)]],
    constant uint& image_height [[buffer(18)]],
    constant float& eps2d [[buffer(19)]],
    constant float& near_plane [[buffer(20)]],
    constant float& far_plane [[buffer(21)]],
    constant float& radius_clip [[buffer(22)]],
    constant uint& camera_model [[buffer(23)]],
    constant uint& has_opacities [[buffer(24)]],
    constant uint& has_compensations [[buffer(25)]],
    constant uint& blocks_per_row [[buffer(26)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = tgp.y;
    const uint block_col = tgp.x;
    const uint gid = block_col * kThreadsPacked + tid;
    const uint block_idx = row * blocks_per_row + block_col;

    threadgroup int flags[256];
    threadgroup int prefix[256];

    PackedProjectionResult out;
    bool valid = false;
    if (row < B * C && gid < N) {
        const uint bid = row / C;
        const uint cid = row % C;
        valid = project_packed_candidate(
            bid,
            cid,
            gid,
            means,
            covars,
            opacities,
            viewmats,
            Ks,
            N,
            C,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            camera_model,
            has_opacities,
            has_compensations,
            out
        );
    }
    flags[tid] = valid ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential exclusive scan: a parallel Blelloch scan is significantly
    // more complex and not warranted for N=256 (one barrier per two-element
    // merge step vs eight steps for the parallel variant).
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

    // Two-level addressing: `offset` is the global write base for this block
    // (from the CPU-side cumsum of per-block counts), and `prefix[tid]` is the
    // local within-block offset computed by the sequential exclusive scan above.
    // Together they give the unique packed output slot for each valid Gaussian.
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
    conics[slot * 3u] = out.conic.x;
    conics[slot * 3u + 1u] = out.conic.y;
    conics[slot * 3u + 2u] = out.conic.z;
    if (compensations != nullptr) {
        compensations[slot] = out.compensation;
    }
}

kernel void projection_ewa_3dgs_packed_bwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* covars [[buffer(1)]],
    device const float* quats [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* viewmats [[buffer(4)]],
    device const float* Ks [[buffer(5)]],
    device const long* batch_ids [[buffer(6)]],
    device const long* camera_ids [[buffer(7)]],
    device const long* gaussian_ids [[buffer(8)]],
    device const float* conics [[buffer(9)]],
    device const float* compensations [[buffer(10)]],
    device const float* v_means2d [[buffer(11)]],
    device const float* v_depths [[buffer(12)]],
    device const float* v_conics [[buffer(13)]],
    device const float* v_compensations [[buffer(14)]],
    device float* tmp_means [[buffer(15)]],
    device float* tmp_covars [[buffer(16)]],
    device float* tmp_quats [[buffer(17)]],
    device float* tmp_scales [[buffer(18)]],
    device float* tmp_viewmats [[buffer(19)]],
    constant uint& B [[buffer(20)]],
    constant uint& C [[buffer(21)]],
    constant uint& N [[buffer(22)]],
    constant uint& nnz [[buffer(23)]],
    constant uint& image_width [[buffer(24)]],
    constant uint& image_height [[buffer(25)]],
    constant float& eps2d [[buffer(26)]],
    constant uint& camera_model [[buffer(27)]],
    constant uint& use_covars [[buffer(28)]],
    constant uint& has_compensations [[buffer(29)]],
    constant uint& viewmats_requires_grad [[buffer(30)]],
    uint id [[thread_position_in_grid]]
) {
    (void)B;
    if (id >= nnz) {
        return;
    }

    const uint bid = uint(batch_ids[id]);
    const uint cid = uint(camera_ids[id]);
    const uint gid = uint(gaussian_ids[id]);

    device const float* mean_ptr = means + (bid * N + gid) * 3u;
    device const float* viewmat_ptr = viewmats + (bid * C + cid) * 16u;
    device const float* K_ptr = Ks + (bid * C + cid) * 9u;
    device const float* conic_ptr = conics + id * 3u;
    device const float* v_mean2d_ptr = v_means2d + id * 2u;
    device const float* v_depth_ptr = v_depths + id;
    device const float* v_conic_ptr = v_conics + id * 3u;

    float3 mean_w = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3x3 R = float3x3(
        float3(viewmat_ptr[0], viewmat_ptr[4], viewmat_ptr[8]),
        float3(viewmat_ptr[1], viewmat_ptr[5], viewmat_ptr[9]),
        float3(viewmat_ptr[2], viewmat_ptr[6], viewmat_ptr[10])
    );
    float3 t = float3(viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]);
    float fx = K_ptr[0];
    float fy = K_ptr[4];
    float cx = K_ptr[2];
    float cy = K_ptr[5];

    float3x3 covar_w;
    float4 quat = float4(0.0f);
    float3 scale = float3(0.0f);
    float3x3 rotmat = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
    if (use_covars != 0u) {
        device const float* covar_ptr = covars + (bid * N + gid) * 6u;
        covar_w = load_mat3_triu(covar_ptr);
    } else {
        quat = *(reinterpret_cast<device const float4*>(quats) + (bid * N + gid));
        scale = *(reinterpret_cast<device const packed_float3*>(scales) + (bid * N + gid));
        rotmat = quat_to_rotmat(quat);
        covar_w = covar_from_RS(rotmat, scale);
    }

    float3 mean_c = pos_w2c(R, t, mean_w);
    float3x3 covar_c = covar_w2c(R, covar_w);

    float2x2 covar2d_inv = float2x2(
        float2(conic_ptr[0], conic_ptr[1]),
        float2(conic_ptr[1], conic_ptr[2])
    );
    float2x2 v_covar2d_inv = float2x2(
        float2(v_conic_ptr[0], 0.5f * v_conic_ptr[1]),
        float2(0.5f * v_conic_ptr[1], v_conic_ptr[2])
    );
    float2x2 v_covar2d = float2x2(float2(0.0f), float2(0.0f));
    inverse_vjp_2x2(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (has_compensations != 0u && compensations != nullptr && v_compensations != nullptr) {
        add_blur_vjp_metal(
            eps2d,
            covar2d_inv,
            compensations[id],
            v_compensations[id],
            v_covar2d
        );
    }

    float2 v_mean2d_val = float2(v_mean2d_ptr[0], v_mean2d_ptr[1]);
    float3x3 v_covar_c = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
    float3 v_mean_c = float3(0.0f);
    switch (camera_model) {
    case kCameraModelPinhole:
        persp_proj_vjp_metal(
            mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, v_covar2d, v_mean2d_val, v_mean_c, v_covar_c
        );
        break;
    case kCameraModelOrtho:
        ortho_proj_vjp_metal(
            mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, v_covar2d, v_mean2d_val, v_mean_c, v_covar_c
        );
        break;
    default:
        fisheye_proj_vjp_metal(
            mean_c, covar_c, fx, fy, cx, cy, image_width, image_height, v_covar2d, v_mean2d_val, v_mean_c, v_covar_c
        );
        break;
    }

    v_mean_c.z += v_depth_ptr[0];

    float3 v_mean_w = float3(0.0f);
    float3x3 v_covar_w = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
    float3x3 v_R = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
    float3 v_t = float3(0.0f);
    pos_w2c_vjp(R, mean_w, v_mean_c, v_R, v_t, v_mean_w);
    covar_w2c_vjp(R, covar_w, v_covar_c, v_R, v_covar_w);

    device float* tmp_mean_ptr = tmp_means + id * 3u;
    tmp_mean_ptr[0] = v_mean_w.x;
    tmp_mean_ptr[1] = v_mean_w.y;
    tmp_mean_ptr[2] = v_mean_w.z;

    if (use_covars != 0u && tmp_covars != nullptr) {
        store_triu3(tmp_covars + id * 6u, v_covar_w);
    } else if (tmp_quats != nullptr && tmp_scales != nullptr) {
        float3x3 v_rotmat = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
        float3 v_scale = float3(0.0f);
        covar_from_RS_vjp(rotmat, scale, v_covar_w, v_rotmat, v_scale);
        float4 v_quat = quat_to_rotmat_vjp(quat, v_rotmat);
        *(reinterpret_cast<device float4*>(tmp_quats) + id) = v_quat;
        *(reinterpret_cast<device packed_float3*>(tmp_scales) + id) = v_scale;
    }

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
