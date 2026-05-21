// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

#include "../metal_math.h"

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kMaxAlpha = 0.99f;
constant float kTransmittanceThreshold = 1.0e-4f;
constant float kMinOneMinusAlpha = 1.0e-6f;
constant float kMaxKernelDensityCutoff = 0.0113f;
constant uint kMaxBlockSize = 256u;

struct RasterizeFromWorldKernelConfig {
    uint B;
    uint C;
    uint I;
    uint N;
    uint channels;
    uint tile_size;
    uint tile_width;
    uint tile_height;
    uint n_tiles;
    uint total_tiles;
    uint n_isects;
    uint image_width;
    uint image_height;
};

inline float reduce_sum_threadgroup(
    float value,
    threadgroup float* scratch,
    uint local_idx,
    uint block_size
) {
    scratch[local_idx] = value;
    for (uint s = block_size >> 1; s > 0u; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_idx < s) {
            scratch[local_idx] += scratch[local_idx + s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float3 safe_normalize_metal(float3 v) {
    const float l = dot(v, v);
    return l > 0.0f ? v * rsqrt(l) : v;
}

inline float3 safe_normalize_bw_metal(float3 v, float3 d_out) {
    const float l = dot(v, v);
    if (l > 0.0f) {
        const float il = rsqrt(l);
        const float il3 = il * il * il;
        return il * d_out - il3 * dot(d_out, v) * v;
    }
    return d_out;
}

inline float3x3 outer3_metal(float3 a, float3 b) {
    return float3x3(a * b.x, a * b.y, a * b.z);
}

inline void load_or_generate_ray_metal(
    device const float* viewmats,
    device const float* Ks,
    device const float* rays,
    constant RasterizeFromWorldKernelConfig& cfg,
    uint image_id,
    uint pixel_y,
    uint pixel_x,
    thread float3& ray_o,
    thread float3& ray_d
) {
    if (rays != nullptr) {
        const uint pixel_flat = (image_id * cfg.image_height + pixel_y) * cfg.image_width + pixel_x;
        ray_o = float3(
            rays[6 * pixel_flat + 0u],
            rays[6 * pixel_flat + 1u],
            rays[6 * pixel_flat + 2u]
        );
        ray_d = float3(
            rays[6 * pixel_flat + 3u],
            rays[6 * pixel_flat + 4u],
            rays[6 * pixel_flat + 5u]
        );
        return;
    }

    const uint viewmat_base = image_id * 16u;
    const float r00 = viewmats[viewmat_base + 0u];
    const float r01 = viewmats[viewmat_base + 1u];
    const float r02 = viewmats[viewmat_base + 2u];
    const float tx = viewmats[viewmat_base + 3u];
    const float r10 = viewmats[viewmat_base + 4u];
    const float r11 = viewmats[viewmat_base + 5u];
    const float r12 = viewmats[viewmat_base + 6u];
    const float ty = viewmats[viewmat_base + 7u];
    const float r20 = viewmats[viewmat_base + 8u];
    const float r21 = viewmats[viewmat_base + 9u];
    const float r22 = viewmats[viewmat_base + 10u];
    const float tz = viewmats[viewmat_base + 11u];

    const uint K_base = image_id * 9u;
    const float fx = Ks[K_base + 0u];
    const float fy = Ks[K_base + 4u];
    const float cx = Ks[K_base + 2u];
    const float cy = Ks[K_base + 5u];

    const float cam_x = (float(pixel_x) + 0.5f - cx) / fx;
    const float cam_y = (float(pixel_y) + 0.5f - cy) / fy;
    const float3 cam_dir = float3(cam_x, cam_y, 1.0f);

    ray_o = float3(
        -(r00 * tx + r10 * ty + r20 * tz),
        -(r01 * tx + r11 * ty + r21 * tz),
        -(r02 * tx + r12 * ty + r22 * tz)
    );
    ray_d = float3(
        r00 * cam_dir.x + r10 * cam_dir.y + r20 * cam_dir.z,
        r01 * cam_dir.x + r11 * cam_dir.y + r21 * cam_dir.z,
        r02 * cam_dir.x + r12 * cam_dir.y + r22 * cam_dir.z
    );
}

inline void quat_scale_to_preci_half_vjp_metal(
    float4 quat,
    float3 scale,
    float3x3 R,
    float3x3 v_M,
    thread float4& v_quat,
    thread float3& v_scale
) {
    const float sx = 1.0f / scale.x;
    const float sy = 1.0f / scale.y;
    const float sz = 1.0f / scale.z;
    const float3x3 S = float3x3(
        float3(sx, 0.0f, 0.0f),
        float3(0.0f, sy, 0.0f),
        float3(0.0f, 0.0f, sz)
    );
    const float3x3 v_R = v_M * S;

    v_quat += quat_to_rotmat_vjp(quat, v_R);
    v_scale.x += -sx * sx * dot(R[0], v_M[0]);
    v_scale.y += -sy * sy * dot(R[1], v_M[1]);
    v_scale.z += -sz * sz * dot(R[2], v_M[2]);
}

kernel void rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const float* backgrounds [[buffer(5)]],
    device const bool* masks [[buffer(6)]],
    device const float* viewmats [[buffer(7)]],
    device const float* Ks [[buffer(8)]],
    device const float* rays [[buffer(9)]],
    device const int* tile_offsets [[buffer(10)]],
    device const int* flatten_ids [[buffer(11)]],
    device float* render_colors [[buffer(12)]],
    device float* render_alphas [[buffer(13)]],
    device int* last_ids [[buffer(14)]],
    device int* sample_counts [[buffer(15)]],
    device float* render_normals [[buffer(16)]],
    constant RasterizeFromWorldKernelConfig& cfg [[buffer(17)]],
    constant uint& use_hit_distance [[buffer(18)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    if (tgp.x >= cfg.tile_width || tgp.y >= cfg.tile_height || tgp.z >= cfg.I) {
        return;
    }

    const uint image_id = tgp.z;
    const uint tile_x = tgp.x;
    const uint tile_y = tgp.y;
    const uint local_x = tid.x;
    const uint local_y = tid.y;
    const uint local_idx = local_y * cfg.tile_size + local_x;
    const uint block_size = cfg.tile_size * cfg.tile_size;
    if (local_idx >= kMaxBlockSize) {
        return;
    }

    const uint i = tile_y * cfg.tile_size + local_y;
    const uint j = tile_x * cfg.tile_size + local_x;
    const bool inside = i < cfg.image_height && j < cfg.image_width;
    const uint pixel_flat = (image_id * cfg.image_height + i) * cfg.image_width + j;
    const uint tile_id = tile_y * cfg.tile_width + tile_x;
    const uint global_tile_id = image_id * cfg.n_tiles + tile_id;

    if (masks != nullptr && !masks[global_tile_id]) {
        if (inside) {
            const uint color_base = pixel_flat * cfg.channels;
            const uint bg_base = image_id * cfg.channels;
            for (uint k = 0; k < cfg.channels; ++k) {
                render_colors[color_base + k] =
                    backgrounds != nullptr ? backgrounds[bg_base + k] : 0.0f;
            }
            render_alphas[pixel_flat] = 0.0f;
            last_ids[pixel_flat] = -1;
            if (sample_counts != nullptr) {
                sample_counts[pixel_flat] = 0;
            }
            if (render_normals != nullptr) {
                render_normals[pixel_flat * 3u + 0u] = 0.0f;
                render_normals[pixel_flat * 3u + 1u] = 0.0f;
                render_normals[pixel_flat * 3u + 2u] = 0.0f;
            }
        }
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < cfg.total_tiles ? tile_offsets[global_tile_id + 1u] : int(cfg.n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float4 xyz_opacity_batch[kMaxBlockSize];
    threadgroup float3 scale_batch[kMaxBlockSize];
    threadgroup float4 quat_batch[kMaxBlockSize];

    float T = 1.0f;
    int cur_idx = -1;
    int n_accumulated = 0;
    bool done = !inside;
    float pix_out[513];
    for (uint k = 0; k < cfg.channels; ++k) {
        pix_out[k] = 0.0f;
    }
    float3 normal_out = float3(0.0f);
    const bool return_normals = render_normals != nullptr;

    for (int batch_start = range_start; batch_start < range_end; batch_start += int(block_size)) {
        const int idx = batch_start + int(local_idx);
        if (idx < range_end) {
            const int isect_id = flatten_ids[idx];
            const uint image_stride = cfg.C * cfg.N;
            const uint batch_id = uint(isect_id) / image_stride;
            const uint gaussian_id = uint(isect_id) % cfg.N;
            const uint shared_idx = batch_id * cfg.N + gaussian_id;

            id_batch[local_idx] = isect_id;
            xyz_opacity_batch[local_idx] = float4(
                means[3 * shared_idx + 0u],
                means[3 * shared_idx + 1u],
                means[3 * shared_idx + 2u],
                opacities[uint(isect_id)]
            );
            scale_batch[local_idx] = float3(
                scales[3 * shared_idx + 0u],
                scales[3 * shared_idx + 1u],
                scales[3 * shared_idx + 2u]
            );
            quat_batch[local_idx] = float4(
                quats[4 * shared_idx + 0u],
                quats[4 * shared_idx + 1u],
                quats[4 * shared_idx + 2u],
                quats[4 * shared_idx + 3u]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = min(block_size, uint(range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            const float4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac.w;
            const float3 xyz = xyz_opac.xyz;
            const float3 scale = scale_batch[t];
            const float4 quat = quat_batch[t];

            float3 ray_o;
            float3 ray_d;
            load_or_generate_ray_metal(
                viewmats, Ks, rays, cfg, image_id, i, j, ray_o, ray_d
            );

            const float3x3 R = quat_to_rotmat(quat);
            const float3x3 S = float3x3(
                float3(1.0f / scale.x, 0.0f, 0.0f),
                float3(0.0f, 1.0f / scale.y, 0.0f),
                float3(0.0f, 0.0f, 1.0f / scale.z)
            );
            const float3x3 Mt = S * transpose(R);
            const float3 gro = Mt * (ray_o - xyz);
            const float3 grd = safe_normalize_metal(Mt * ray_d);
            const float3 gcrod = cross(grd, gro);
            const float gray_dist = dot(gcrod, gcrod);
            const float power = -0.5f * gray_dist;
            const float max_response = exp(power);
            const float alpha = min(kMaxAlpha, opac * max_response);
            if (alpha < kAlphaThreshold || max_response <= kMaxKernelDensityCutoff) {
                continue;
            }

            float hit_distance = 0.0f;
            if (use_hit_distance != 0u) {
                const float hit_t = dot(grd, -gro);
                const float3 grds = scale * (grd * hit_t);
                hit_distance = length(grds);
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= kTransmittanceThreshold) {
                done = true;
                break;
            }

            const int isect_id = id_batch[t];
            const uint color_base = uint(isect_id) * cfg.channels;
            const float vis = alpha * T;
            for (uint k = 0; k < cfg.channels; ++k) {
                const float value = (use_hit_distance != 0u && k == cfg.channels - 1u)
                    ? hit_distance
                    : colors[color_base + k];
                pix_out[k] += value * vis;
            }

            if (return_normals) {
                const float3 unnormalized_normal = R[2];
                const bool flipped = dot(unnormalized_normal, ray_d) > 0.0f;
                const float3 unnormalized_flipped =
                    flipped ? -unnormalized_normal : unnormalized_normal;
                const float3 normal = safe_normalize_metal(unnormalized_flipped);
                normal_out += normal * vis;
            }

            cur_idx = batch_start + int(t);
            n_accumulated += 1;
            T = next_T;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inside) {
        const uint color_base = pixel_flat * cfg.channels;
        const uint bg_base = image_id * cfg.channels;
        render_alphas[pixel_flat] = 1.0f - T;
        for (uint k = 0; k < cfg.channels; ++k) {
            render_colors[color_base + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[bg_base + k]);
        }
        last_ids[pixel_flat] = cur_idx;
        if (sample_counts != nullptr) {
            sample_counts[pixel_flat] = n_accumulated;
        }
        if (render_normals != nullptr) {
            render_normals[pixel_flat * 3u + 0u] = normal_out.x;
            render_normals[pixel_flat * 3u + 1u] = normal_out.y;
            render_normals[pixel_flat * 3u + 2u] = normal_out.z;
        }
    }
}

kernel void rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const float* backgrounds [[buffer(5)]],
    device const bool* masks [[buffer(6)]],
    device const float* viewmats [[buffer(7)]],
    device const float* Ks [[buffer(8)]],
    device const float* rays [[buffer(9)]],
    device const int* tile_offsets [[buffer(10)]],
    device const int* flatten_ids [[buffer(11)]],
    device const float* render_alphas [[buffer(12)]],
    device const int* last_ids [[buffer(13)]],
    device const float* v_render_colors [[buffer(14)]],
    device const float* v_render_alphas [[buffer(15)]],
    device const float* v_render_normals [[buffer(16)]],
    device float* tmp_means [[buffer(17)]],
    device float* tmp_quats [[buffer(18)]],
    device float* tmp_scales [[buffer(19)]],
    device float* tmp_colors [[buffer(20)]],
    device float* tmp_opacities [[buffer(21)]],
    device float* v_rays [[buffer(22)]],
    constant RasterizeFromWorldKernelConfig& cfg [[buffer(23)]],
    constant uint& use_hit_distance [[buffer(24)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    if (tgp.x >= cfg.tile_width || tgp.y >= cfg.tile_height || tgp.z >= cfg.I) {
        return;
    }

    const uint image_id = tgp.z;
    const uint tile_x = tgp.x;
    const uint tile_y = tgp.y;
    const uint local_x = tid.x;
    const uint local_y = tid.y;
    const uint local_idx = local_y * cfg.tile_size + local_x;
    const uint block_size = cfg.tile_size * cfg.tile_size;
    if (local_idx >= kMaxBlockSize) {
        return;
    }

    const uint i = tile_y * cfg.tile_size + local_y;
    const uint j = tile_x * cfg.tile_size + local_x;
    const bool inside = i < cfg.image_height && j < cfg.image_width;
    const uint pixel_flat = min(
        (image_id * cfg.image_height + i) * cfg.image_width + j,
        cfg.I * cfg.image_height * cfg.image_width - 1u
    );
    const uint tile_id = tile_y * cfg.tile_width + tile_x;
    const uint global_tile_id = image_id * cfg.n_tiles + tile_id;

    if (masks != nullptr && !masks[global_tile_id]) {
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < cfg.total_tiles ? tile_offsets[global_tile_id + 1u] : int(cfg.n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float4 xyz_opacity_batch[kMaxBlockSize];
    threadgroup float3 scale_batch[kMaxBlockSize];
    threadgroup float4 quat_batch[kMaxBlockSize];
    threadgroup float reduce_scratch[kMaxBlockSize];

    const float T_final = inside ? (1.0f - render_alphas[pixel_flat]) : 1.0f;
    float T = T_final;
    float buffer_dot = 0.0f;
    const int bin_final = inside ? last_ids[pixel_flat] : -1;
    float3 normal_buffer = float3(0.0f);
    float3 v_render_n = float3(0.0f);
    if (inside && v_render_normals != nullptr) {
        v_render_n = float3(
            v_render_normals[pixel_flat * 3u + 0u],
            v_render_normals[pixel_flat * 3u + 1u],
            v_render_normals[pixel_flat * 3u + 2u]
        );
    }

    float bg_dot = 0.0f;
    if (inside && backgrounds != nullptr) {
        const uint bg_base = image_id * cfg.channels;
        const uint grad_base = pixel_flat * cfg.channels;
        for (uint k = 0; k < cfg.channels; ++k) {
            bg_dot += backgrounds[bg_base + k] * v_render_colors[grad_base + k];
        }
    }

    const uint clamped_i = min(i, cfg.image_height - 1u);
    const uint clamped_j = min(j, cfg.image_width - 1u);
    float3 ray_o;
    float3 ray_d;
    load_or_generate_ray_metal(
        viewmats, Ks, rays, cfg, image_id, clamped_i, clamped_j, ray_o, ray_d
    );
    float3 v_ray_o = float3(0.0f);
    float3 v_ray_d = float3(0.0f);

    for (uint b = 0u; b < uint((range_end - range_start + int(block_size) - 1) / int(block_size)); ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_end = range_end - 1 - int(block_size) * int(b);
        const int batch_size = min(int(block_size), batch_end + 1 - range_start);
        const int idx = batch_end - int(local_idx);
        if (idx >= range_start) {
            const int isect_id = flatten_ids[idx];
            const uint image_stride = cfg.C * cfg.N;
            const uint batch_id = uint(isect_id) / image_stride;
            const uint gaussian_id = uint(isect_id) % cfg.N;
            const uint shared_idx = batch_id * cfg.N + gaussian_id;

            id_batch[local_idx] = isect_id;
            xyz_opacity_batch[local_idx] = float4(
                means[3 * shared_idx + 0u],
                means[3 * shared_idx + 1u],
                means[3 * shared_idx + 2u],
                opacities[uint(isect_id)]
            );
            scale_batch[local_idx] = float3(
                scales[3 * shared_idx + 0u],
                scales[3 * shared_idx + 1u],
                scales[3 * shared_idx + 2u]
            );
            quat_batch[local_idx] = float4(
                quats[4 * shared_idx + 0u],
                quats[4 * shared_idx + 1u],
                quats[4 * shared_idx + 2u],
                quats[4 * shared_idx + 3u]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < uint(batch_size); ++t) {
            const int isect_idx = batch_end - int(t);
            bool valid = inside && (isect_idx <= bin_final);

            float alpha = 0.0f;
            float opac = 0.0f;
            float vis = 0.0f;
            float dot_gc = 0.0f;
            float3 xyz = float3(0.0f);
            float3 scale = float3(0.0f);
            float4 quat = float4(0.0f);
            float3x3 R;
            float3x3 Mt;
            float3 o_minus_mu = float3(0.0f);
            float3 gro = float3(0.0f);
            float3 grd_raw = float3(0.0f);
            float3 grd = float3(0.0f);
            float3 gcrod = float3(0.0f);
            float local_hit_dist = 0.0f;
            float hit_t = 0.0f;
            bool flipped = false;
            float3 normal = float3(0.0f);

            if (valid) {
                const float4 xyz_opac = xyz_opacity_batch[t];
                opac = xyz_opac.w;
                xyz = xyz_opac.xyz;
                scale = scale_batch[t];
                quat = quat_batch[t];

                R = quat_to_rotmat(quat);
                const float3x3 S = float3x3(
                    float3(1.0f / scale.x, 0.0f, 0.0f),
                    float3(0.0f, 1.0f / scale.y, 0.0f),
                    float3(0.0f, 0.0f, 1.0f / scale.z)
                );
                Mt = S * transpose(R);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd_raw = Mt * ray_d;
                grd = safe_normalize_metal(grd_raw);
                gcrod = cross(grd, gro);
                const float gray_dist = dot(gcrod, gcrod);
                const float power = -0.5f * gray_dist;
                vis = exp(power);
                alpha = min(kMaxAlpha, opac * vis);
                if (power > 0.0f || alpha < kAlphaThreshold || vis <= kMaxKernelDensityCutoff) {
                    valid = false;
                }
                if (valid && use_hit_distance != 0u) {
                    hit_t = dot(grd, -gro);
                    const float3 grds = scale * (grd * hit_t);
                    local_hit_dist = length(grds);
                }
                if (valid && v_render_normals != nullptr) {
                    const float3 unnormalized_normal = R[2];
                    flipped = dot(unnormalized_normal, ray_d) > 0.0f;
                    const float3 unnormalized_flipped =
                        flipped ? -unnormalized_normal : unnormalized_normal;
                    normal = safe_normalize_metal(unnormalized_flipped);
                }
            }

            float3 v_mean_local = float3(0.0f);
            float3 v_scale_local = float3(0.0f);
            float4 v_quat_local = float4(0.0f);
            float v_opacity_local = 0.0f;

            if (valid) {
                const uint grad_base = pixel_flat * cfg.channels;
                const int isect_id = id_batch[t];
                const uint color_base = uint(isect_id) * cfg.channels;
                const float ra = 1.0f / max(kMinOneMinusAlpha, 1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                for (uint k = 0; k < cfg.channels; ++k) {
                    const float rgb_k = (use_hit_distance != 0u && k == cfg.channels - 1u)
                        ? local_hit_dist
                        : colors[color_base + k];
                    dot_gc += rgb_k * v_render_colors[grad_base + k];
                }

                float v_alpha = dot_gc * T - buffer_dot * ra;
                v_alpha += T_final * ra * v_render_alphas[pixel_flat];
                if (backgrounds != nullptr) {
                    v_alpha += -T_final * ra * bg_dot;
                }
                if (v_render_normals != nullptr) {
                    v_alpha += dot(normal * T - normal_buffer * ra, v_render_n);
                }

                float3 v_grd_hit = float3(0.0f);
                float3 v_gro_hit = float3(0.0f);
                if (use_hit_distance != 0u) {
                    const float v_depth = fac * v_render_colors[grad_base + (cfg.channels - 1u)];
                    const float3 grds = scale * (grd * hit_t);
                    const float hit_dist_len = length(grds);
                    float3 v_grds = float3(0.0f);
                    if (hit_dist_len > 1.0e-8f) {
                        v_grds = (grds / hit_dist_len) * v_depth;
                    }
                    const float v_hit_t = dot(scale * grd, v_grds);
                    v_grd_hit = (scale * hit_t) * v_grds;
                    v_scale_local += (grd * hit_t) * v_grds;
                    v_grd_hit += -gro * v_hit_t;
                    v_gro_hit = -grd * v_hit_t;
                }

                if (opac * vis <= kMaxAlpha) {
                    const float v_vis = opac * v_alpha;
                    const float v_gray_dist = -0.5f * vis * v_vis;
                    const float3 v_gcrod = 2.0f * v_gray_dist * gcrod;
                    const float3 v_grd_n = -cross(v_gcrod, gro) + v_grd_hit;
                    const float3 v_gro = cross(v_gcrod, grd) + v_gro_hit;
                    const float3 v_grd = safe_normalize_bw_metal(grd_raw, v_grd_n);
                    const float3x3 v_Mt = outer3_metal(v_grd, ray_d) + outer3_metal(v_gro, o_minus_mu);
                    const float3 v_o_minus_mu = transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    v_ray_o += v_o_minus_mu;
                    v_ray_d += transpose(Mt) * v_grd;

                    quat_scale_to_preci_half_vjp_metal(
                        quat,
                        scale,
                        R,
                        transpose(v_Mt),
                        v_quat_local,
                        v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;

                    if (v_render_normals != nullptr) {
                        const float3 v_normal_local = v_render_n * fac;
                        const float3 unnormalized_normal = R[2];
                        const float3 unnormalized_flipped =
                            flipped ? -unnormalized_normal : unnormalized_normal;
                        const float3 v_unnormalized_flipped =
                            safe_normalize_bw_metal(unnormalized_flipped, v_normal_local);
                        const float3 v_unnormalized =
                            flipped ? -v_unnormalized_flipped : v_unnormalized_flipped;
                        const float3x3 v_R_norm = float3x3(
                            float3(0.0f, 0.0f, 0.0f),
                            float3(0.0f, 0.0f, 0.0f),
                            v_unnormalized
                        );
                        v_quat_local += quat_to_rotmat_vjp(quat, v_R_norm);
                    }
                }

                for (uint k = 0; k < cfg.channels; ++k) {
                    const float rgb_k = (use_hit_distance != 0u && k == cfg.channels - 1u)
                        ? local_hit_dist
                        : colors[color_base + k];
                    buffer_dot += rgb_k * fac * v_render_colors[grad_base + k];
                }
                if (v_render_normals != nullptr) {
                    normal_buffer += normal * fac;
                }
            }

            const float sum_mean_x = reduce_sum_threadgroup(v_mean_local.x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_means[3 * uint(isect_idx) + 0u] = sum_mean_x;
            }
            const float sum_mean_y = reduce_sum_threadgroup(v_mean_local.y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_means[3 * uint(isect_idx) + 1u] = sum_mean_y;
            }
            const float sum_mean_z = reduce_sum_threadgroup(v_mean_local.z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_means[3 * uint(isect_idx) + 2u] = sum_mean_z;
            }

            const float sum_scale_x = reduce_sum_threadgroup(v_scale_local.x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_scales[3 * uint(isect_idx) + 0u] = sum_scale_x;
            }
            const float sum_scale_y = reduce_sum_threadgroup(v_scale_local.y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_scales[3 * uint(isect_idx) + 1u] = sum_scale_y;
            }
            const float sum_scale_z = reduce_sum_threadgroup(v_scale_local.z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_scales[3 * uint(isect_idx) + 2u] = sum_scale_z;
            }

            const float sum_quat_x = reduce_sum_threadgroup(v_quat_local.x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_quats[4 * uint(isect_idx) + 0u] = sum_quat_x;
            }
            const float sum_quat_y = reduce_sum_threadgroup(v_quat_local.y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_quats[4 * uint(isect_idx) + 1u] = sum_quat_y;
            }
            const float sum_quat_z = reduce_sum_threadgroup(v_quat_local.z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_quats[4 * uint(isect_idx) + 2u] = sum_quat_z;
            }
            const float sum_quat_w = reduce_sum_threadgroup(v_quat_local.w, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_quats[4 * uint(isect_idx) + 3u] = sum_quat_w;
            }

            const float sum_opacity = reduce_sum_threadgroup(v_opacity_local, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_opacities[uint(isect_idx)] = sum_opacity;
            }

            const uint grad_base = pixel_flat * cfg.channels;
            for (uint k = 0; k < cfg.channels; ++k) {
                float v_rgb = 0.0f;
                if (valid) {
                    if (use_hit_distance == 0u || k != cfg.channels - 1u) {
                        v_rgb = alpha * T * v_render_colors[grad_base + k];
                    }
                }
                const float sum_rgb = reduce_sum_threadgroup(v_rgb, reduce_scratch, local_idx, block_size);
                if (local_idx == 0u) {
                    tmp_colors[uint(isect_idx) * cfg.channels + k] = sum_rgb;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (inside && v_rays != nullptr) {
        v_rays[6 * pixel_flat + 0u] = v_ray_o.x;
        v_rays[6 * pixel_flat + 1u] = v_ray_o.y;
        v_rays[6 * pixel_flat + 2u] = v_ray_o.z;
        v_rays[6 * pixel_flat + 3u] = v_ray_d.x;
        v_rays[6 * pixel_flat + 4u] = v_ray_d.y;
        v_rays[6 * pixel_flat + 5u] = v_ray_d.z;
    }
}
