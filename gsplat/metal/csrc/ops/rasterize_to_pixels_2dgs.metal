// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kMaxAlpha = 0.99f;
constant float kTransmittanceThreshold = 1.0e-4f;
constant float kMinOneMinusAlpha = 1.0e-6f;
constant uint kMaxBlockSize = 256u;
constant float kFilterInvSquare2DGS = 2.0f;

struct Rasterize2DGSBwdConfig {
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
    uint absgrad;
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

kernel void rasterize_to_pixels_2dgs_fwd_kernel(
    device const float* means2d [[buffer(0)]],
    device const float* ray_transforms [[buffer(1)]],
    device const float* colors [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* normals [[buffer(4)]],
    device const float* backgrounds [[buffer(5)]],
    device const bool* masks [[buffer(6)]],
    device const int* tile_offsets [[buffer(7)]],
    device const int* flatten_ids [[buffer(8)]],
    device float* render_colors [[buffer(9)]],
    device float* render_alphas [[buffer(10)]],
    device float* render_normals [[buffer(11)]],
    device float* render_distort [[buffer(12)]],
    device float* render_median [[buffer(13)]],
    device int* last_ids [[buffer(14)]],
    device int* median_ids [[buffer(15)]],
    constant uint& I [[buffer(16)]],
    constant uint& N [[buffer(17)]],
    constant uint& channels [[buffer(18)]],
    constant uint& tile_size [[buffer(19)]],
    constant uint& tile_width [[buffer(20)]],
    constant uint& tile_height [[buffer(21)]],
    constant uint& n_tiles [[buffer(22)]],
    constant uint& total_tiles [[buffer(23)]],
    constant uint& n_isects [[buffer(24)]],
    constant uint& image_width [[buffer(25)]],
    constant uint& image_height [[buffer(26)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)N;  // Gaussian lookup goes through flatten_ids; N is accepted for API symmetry
    if (tgp.x >= tile_width || tgp.y >= tile_height || tgp.z >= I) {
        return;
    }

    const uint image_id = tgp.z;
    const uint tile_x = tgp.x;
    const uint tile_y = tgp.y;
    const uint local_x = tid.x;
    const uint local_y = tid.y;
    const uint local_idx = local_y * tile_size + local_x;
    const uint block_size = tile_size * tile_size;

    if (local_idx >= kMaxBlockSize) {
        return;
    }

    const uint i = tile_y * tile_size + local_y;
    const uint j = tile_x * tile_size + local_x;
    const bool inside = i < image_height && j < image_width;
    const uint pixel_flat = (image_id * image_height + i) * image_width + j;
    const uint tile_id = tile_y * tile_width + tile_x;
    const uint global_tile_id = image_id * n_tiles + tile_id;

    if (masks != nullptr && !masks[global_tile_id]) {
        if (inside) {
            const uint color_base = pixel_flat * channels;
            for (uint k = 0; k < channels; ++k) {
                render_colors[color_base + k] =
                    backgrounds != nullptr ? backgrounds[image_id * channels + k] : 0.0f;
            }
        }
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < total_tiles ? tile_offsets[global_tile_id + 1u] : int(n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 u_batch[kMaxBlockSize];
    threadgroup float3 v_batch[kMaxBlockSize];
    threadgroup float3 w_batch[kMaxBlockSize];
    threadgroup float3 normal_batch[kMaxBlockSize];

    float T = 1.0f;
    int cur_idx = 0;
    bool done = !inside;
    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    const uint color_base = pixel_flat * channels;

    float pix_out_normals[3] = {0.0f, 0.0f, 0.0f};
    float distort = 0.0f;
    float accum_vis_depth = 0.0f;
    float median_depth = 0.0f;
    uint median_idx = 0u;

    for (int batch_start = range_start; batch_start < range_end; batch_start += int(block_size)) {
        const int idx = batch_start + int(local_idx);
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[local_idx] = g;
            xy_opacity_batch[local_idx] = float3(
                means2d[2 * g],
                means2d[2 * g + 1],
                opacities[g]
            );
            u_batch[local_idx] = float3(
                ray_transforms[9 * g],
                ray_transforms[9 * g + 1],
                ray_transforms[9 * g + 2]
            );
            v_batch[local_idx] = float3(
                ray_transforms[9 * g + 3],
                ray_transforms[9 * g + 4],
                ray_transforms[9 * g + 5]
            );
            w_batch[local_idx] = float3(
                ray_transforms[9 * g + 6],
                ray_transforms[9 * g + 7],
                ray_transforms[9 * g + 8]
            );
            normal_batch[local_idx] = float3(
                normals[3 * g],
                normals[3 * g + 1],
                normals[3 * g + 2]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = min(block_size, uint(range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            const float3 xy_opac = xy_opacity_batch[t];
            const float3 u_M = u_batch[t];
            const float3 v_M = v_batch[t];
            const float3 w_M = w_batch[t];

            const float3 h_u = px * w_M - u_M;
            const float3 h_v = py * w_M - v_M;
            const float3 ray_cross = cross(h_u, h_v);
            if (ray_cross.z == 0.0f) {
                continue;
            }

            const float2 s = float2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);
            const float gauss_weight_3d = s.x * s.x + s.y * s.y;

            const float2 d = float2(xy_opac.x - px, xy_opac.y - py);
            const float gauss_weight_2d = kFilterInvSquare2DGS * (d.x * d.x + d.y * d.y);
            const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

            const float sigma = 0.5f * gauss_weight;
            const float alpha = min(kMaxAlpha, xy_opac.z * exp(-sigma));
            if (sigma < 0.0f || alpha < kAlphaThreshold) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= kTransmittanceThreshold) {
                done = true;
                break;
            }

            const int g = id_batch[t];
            const float vis = alpha * T;
            const uint g_color_base = uint(g) * channels;
            for (uint k = 0; k < channels; ++k) {
                render_colors[color_base + k] += colors[g_color_base + k] * vis;
            }

            const float3 normal = normal_batch[t];
            pix_out_normals[0] += normal.x * vis;
            pix_out_normals[1] += normal.y * vis;
            pix_out_normals[2] += normal.z * vis;

            const float depth = colors[g_color_base + (channels - 1u)];
            const float distort_bi_0 = vis * depth * (1.0f - T);
            const float distort_bi_1 = vis * accum_vis_depth;
            distort += 2.0f * (distort_bi_0 - distort_bi_1);
            accum_vis_depth += vis * depth;

            if (T > 0.5f) {
                median_depth = depth;
                median_idx = batch_start + int(t);
            }

            cur_idx = batch_start + int(t);
            T = next_T;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inside) {
        render_alphas[pixel_flat] = 1.0f - T;
        if (backgrounds != nullptr) {
            const uint bg_base = image_id * channels;
            for (uint k = 0; k < channels; ++k) {
                render_colors[color_base + k] += T * backgrounds[bg_base + k];
            }
        }
        render_normals[pixel_flat * 3u] = pix_out_normals[0];
        render_normals[pixel_flat * 3u + 1u] = pix_out_normals[1];
        render_normals[pixel_flat * 3u + 2u] = pix_out_normals[2];
        render_distort[pixel_flat] = distort;
        render_median[pixel_flat] = median_depth;
        last_ids[pixel_flat] = cur_idx;
        median_ids[pixel_flat] = int(median_idx);
    }
}

kernel void rasterize_to_pixels_2dgs_bwd_kernel(
    device const float* means2d [[buffer(0)]],
    device const float* ray_transforms [[buffer(1)]],
    device const float* colors [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* normals [[buffer(4)]],
    device const float* backgrounds [[buffer(5)]],
    device const bool* masks [[buffer(6)]],
    device const int* tile_offsets [[buffer(7)]],
    device const int* flatten_ids [[buffer(8)]],
    device const float* render_colors [[buffer(9)]],
    device const float* render_alphas [[buffer(10)]],
    device const int* last_ids [[buffer(11)]],
    device const int* median_ids [[buffer(12)]],
    device const float* v_render_colors [[buffer(13)]],
    device const float* v_render_alphas [[buffer(14)]],
    device const float* v_render_normals [[buffer(15)]],
    device const float* v_render_distort [[buffer(16)]],
    device const float* v_render_median [[buffer(17)]],
    device float* tmp_means2d_abs [[buffer(18)]],
    device float* tmp_means2d [[buffer(19)]],
    device float* tmp_ray_transforms [[buffer(20)]],
    device float* tmp_colors [[buffer(21)]],
    device float* tmp_opacities [[buffer(22)]],
    device float* tmp_normals [[buffer(23)]],
    constant Rasterize2DGSBwdConfig& cfg [[buffer(24)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)cfg.N;  // Gaussian lookup goes through flatten_ids; N is accepted for API symmetry
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
        cfg.I * cfg.image_height * cfg.image_width - 1u);
    const uint tile_id = tile_y * cfg.tile_width + tile_x;
    const uint global_tile_id = image_id * cfg.n_tiles + tile_id;

    if (masks != nullptr && !masks[global_tile_id]) {
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < cfg.total_tiles ? tile_offsets[global_tile_id + 1u] : int(cfg.n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 u_batch[kMaxBlockSize];
    threadgroup float3 v_batch[kMaxBlockSize];
    threadgroup float3 w_batch[kMaxBlockSize];
    threadgroup float3 normal_batch[kMaxBlockSize];
    threadgroup float reduce_scratch[kMaxBlockSize];

    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    const float T_final = 1.0f - render_alphas[pixel_flat];
    float T = T_final;
    float buffer_dot = 0.0f;
    float buffer_normal_x = 0.0f;
    float buffer_normal_y = 0.0f;
    float buffer_normal_z = 0.0f;
    const int bin_final = inside ? last_ids[pixel_flat] : 0;
    const int median_idx = inside ? median_ids[pixel_flat] : 0;

    float bg_dot = 0.0f;
    if (inside && backgrounds != nullptr) {
        const uint bg_base = image_id * cfg.channels;
        const uint grad_base = pixel_flat * cfg.channels;
        for (uint k = 0; k < cfg.channels; ++k) {
            bg_dot += backgrounds[bg_base + k] * v_render_colors[grad_base + k];
        }
    }

    const float v_render_a = v_render_alphas[pixel_flat];
    const float v_render_n0 = v_render_normals[pixel_flat * 3u];
    const float v_render_n1 = v_render_normals[pixel_flat * 3u + 1u];
    const float v_render_n2 = v_render_normals[pixel_flat * 3u + 2u];
    const float v_distort = v_render_distort[pixel_flat];
    const float v_median = v_render_median[pixel_flat];
    float accum_d_buffer = render_colors[pixel_flat * cfg.channels + cfg.channels - 1u];
    const float accum_d = accum_d_buffer;
    float accum_w_buffer = render_alphas[pixel_flat];
    const float accum_w = accum_w_buffer;
    float distort_buffer = 0.0f;

    const uint num_batches = uint((range_end - range_start + int(block_size) - 1) / int(block_size));
    for (uint b = 0u; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_end = range_end - 1 - int(block_size) * int(b);
        const int batch_size = min(int(block_size), batch_end + 1 - range_start);
        const int idx = batch_end - int(local_idx);
        if (idx >= range_start) {
            const int g = flatten_ids[idx];
            id_batch[local_idx] = g;
            xy_opacity_batch[local_idx] = float3(
                means2d[2 * g],
                means2d[2 * g + 1],
                opacities[g]
            );
            u_batch[local_idx] = float3(
                ray_transforms[9 * g],
                ray_transforms[9 * g + 1],
                ray_transforms[9 * g + 2]
            );
            v_batch[local_idx] = float3(
                ray_transforms[9 * g + 3],
                ray_transforms[9 * g + 4],
                ray_transforms[9 * g + 5]
            );
            w_batch[local_idx] = float3(
                ray_transforms[9 * g + 6],
                ray_transforms[9 * g + 7],
                ray_transforms[9 * g + 8]
            );
            normal_batch[local_idx] = float3(
                normals[3 * g],
                normals[3 * g + 1],
                normals[3 * g + 2]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < uint(batch_size); ++t) {
            const int isect_idx = batch_end - int(t);
            bool valid = inside && (isect_idx <= bin_final);

            float alpha = 0.0f;
            float opac = 0.0f;
            float vis = 0.0f;
            float gauss_weight_3d = 0.0f;
            float gauss_weight_2d = 0.0f;
            float2 s = float2(0.0f);
            float2 d = float2(0.0f);
            float3 h_u = float3(0.0f);
            float3 h_v = float3(0.0f);
            float3 ray_cross = float3(0.0f);
            float3 w_M = float3(0.0f);

            if (valid) {
                const float3 xy_opac = xy_opacity_batch[t];
                const float3 u_M = u_batch[t];
                const float3 v_M = v_batch[t];
                w_M = w_batch[t];
                opac = xy_opac.z;
                h_u = px * w_M - u_M;
                h_v = py * w_M - v_M;
                ray_cross = cross(h_u, h_v);
                if (ray_cross.z == 0.0f) {
                    valid = false;
                } else {
                    s = float2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);
                    gauss_weight_3d = s.x * s.x + s.y * s.y;
                    d = float2(xy_opac.x - px, xy_opac.y - py);
                    gauss_weight_2d = kFilterInvSquare2DGS * (d.x * d.x + d.y * d.y);
                    const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);
                    const float sigma = 0.5f * gauss_weight;
                    vis = exp(-sigma);
                    alpha = min(kMaxAlpha, opac * vis);
                    if (sigma < 0.0f || alpha < kAlphaThreshold) {
                        valid = false;
                    }
                }
            }

            float v_u_x = 0.0f;
            float v_u_y = 0.0f;
            float v_u_z = 0.0f;
            float v_v_x = 0.0f;
            float v_v_y = 0.0f;
            float v_v_z = 0.0f;
            float v_w_x = 0.0f;
            float v_w_y = 0.0f;
            float v_w_z = 0.0f;
            float v_xy_x = 0.0f;
            float v_xy_y = 0.0f;
            float v_xy_abs_x = 0.0f;
            float v_xy_abs_y = 0.0f;
            float v_opacity = 0.0f;
            float v_normal_x = 0.0f;
            float v_normal_y = 0.0f;
            float v_normal_z = 0.0f;
            float dot_gc = 0.0f;
            float fac = 0.0f;
            float extra_depth_grad = 0.0f;

            if (valid) {
                if (isect_idx == median_idx) {
                    extra_depth_grad += v_median;
                }

                const float ra = 1.0f / max(kMinOneMinusAlpha, 1.0f - alpha);
                T *= ra;
                fac = alpha * T;

                const int g = id_batch[t];
                const uint color_base = uint(g) * cfg.channels;
                const uint grad_base = pixel_flat * cfg.channels;
                for (uint k = 0; k < cfg.channels; ++k) {
                    dot_gc += colors[color_base + k] * v_render_colors[grad_base + k];
                }

                float v_alpha = dot_gc * T - buffer_dot * ra;
                const float3 nrm = normal_batch[t];
                v_normal_x = fac * v_render_n0;
                v_normal_y = fac * v_render_n1;
                v_normal_z = fac * v_render_n2;
                v_alpha += (nrm.x * T - buffer_normal_x * ra) * v_render_n0;
                v_alpha += (nrm.y * T - buffer_normal_y * ra) * v_render_n1;
                v_alpha += (nrm.z * T - buffer_normal_z * ra) * v_render_n2;
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    v_alpha += -T_final * ra * bg_dot;
                }

                const float depth = colors[color_base + cfg.channels - 1u];
                const float dl_dw =
                    2.0f * (2.0f * (depth * accum_w_buffer - accum_d_buffer) + (accum_d - depth * accum_w));
                v_alpha += (dl_dw * T - distort_buffer * ra) * v_distort;
                accum_d_buffer -= fac * depth;
                accum_w_buffer -= fac;
                distort_buffer += dl_dw * fac;
                extra_depth_grad += 2.0f * fac * (2.0f - 2.0f * T - accum_w + fac) * v_distort;

                if (opac * vis <= kMaxAlpha) {
                    // depth gradient flows through extra_depth_grad on the color channel; no additional w_M gradient here
                    const float v_depth = 0.0f;
                    const float v_G = opac * v_alpha;
                    if (gauss_weight_3d <= gauss_weight_2d) {
                        const float2 v_s = float2(
                            v_G * -vis * s.x + v_depth * w_M.x,
                            v_G * -vis * s.y + v_depth * w_M.y);
                        const float3 v_z_w_M = float3(s.x, s.y, 1.0f);
                        const float v_sx_pz = v_s.x / ray_cross.z;
                        const float v_sy_pz = v_s.y / ray_cross.z;
                        const float3 v_ray_cross = float3(
                            v_sx_pz,
                            v_sy_pz,
                            -(v_sx_pz * s.x + v_sy_pz * s.y));
                        const float3 v_h_u = cross(h_v, v_ray_cross);
                        const float3 v_h_v = cross(v_ray_cross, h_u);
                        v_u_x = -v_h_u.x;
                        v_u_y = -v_h_u.y;
                        v_u_z = -v_h_u.z;
                        v_v_x = -v_h_v.x;
                        v_v_y = -v_h_v.y;
                        v_v_z = -v_h_v.z;
                        v_w_x = px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_M.x;
                        v_w_y = px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_M.y;
                        v_w_z = px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_M.z;
                    } else {
                        const float v_G_ddelx = -vis * kFilterInvSquare2DGS * d.x;
                        const float v_G_ddely = -vis * kFilterInvSquare2DGS * d.y;
                        v_xy_x = v_G * v_G_ddelx;
                        v_xy_y = v_G * v_G_ddely;
                        if (cfg.absgrad != 0u) {
                            v_xy_abs_x = abs(v_xy_x);
                            v_xy_abs_y = abs(v_xy_y);
                        }
                    }
                    v_opacity = vis * v_alpha;
                }

                buffer_dot += dot_gc * fac;
                const float3 nrm2 = normal_batch[t];
                buffer_normal_x += nrm2.x * fac;
                buffer_normal_y += nrm2.y * fac;
                buffer_normal_z += nrm2.z * fac;
            }

            const float sum_u_x = reduce_sum_threadgroup(v_u_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx)] = sum_u_x;
            }
            const float sum_u_y = reduce_sum_threadgroup(v_u_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 1u] = sum_u_y;
            }
            const float sum_u_z = reduce_sum_threadgroup(v_u_z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 2u] = sum_u_z;
            }
            const float sum_v_x = reduce_sum_threadgroup(v_v_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 3u] = sum_v_x;
            }
            const float sum_v_y = reduce_sum_threadgroup(v_v_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 4u] = sum_v_y;
            }
            const float sum_v_z = reduce_sum_threadgroup(v_v_z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 5u] = sum_v_z;
            }
            const float sum_w_x = reduce_sum_threadgroup(v_w_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 6u] = sum_w_x;
            }
            const float sum_w_y = reduce_sum_threadgroup(v_w_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 7u] = sum_w_y;
            }
            const float sum_w_z = reduce_sum_threadgroup(v_w_z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_ray_transforms[9 * uint(isect_idx) + 8u] = sum_w_z;
            }

            const float sum_xy_x = reduce_sum_threadgroup(v_xy_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_means2d[2 * uint(isect_idx)] = sum_xy_x;
            }
            const float sum_xy_y = reduce_sum_threadgroup(v_xy_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_means2d[2 * uint(isect_idx) + 1u] = sum_xy_y;
            }
            if (tmp_means2d_abs != nullptr) {
                const float sum_xy_abs_x = reduce_sum_threadgroup(v_xy_abs_x, reduce_scratch, local_idx, block_size);
                if (local_idx == 0u) {
                    tmp_means2d_abs[2 * uint(isect_idx)] = sum_xy_abs_x;
                }
                const float sum_xy_abs_y = reduce_sum_threadgroup(v_xy_abs_y, reduce_scratch, local_idx, block_size);
                if (local_idx == 0u) {
                    tmp_means2d_abs[2 * uint(isect_idx) + 1u] = sum_xy_abs_y;
                }
            }

            const float sum_opacity = reduce_sum_threadgroup(v_opacity, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_opacities[uint(isect_idx)] = sum_opacity;
            }

            const float sum_normal_x = reduce_sum_threadgroup(v_normal_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_normals[3 * uint(isect_idx)] = sum_normal_x;
            }
            const float sum_normal_y = reduce_sum_threadgroup(v_normal_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_normals[3 * uint(isect_idx) + 1u] = sum_normal_y;
            }
            const float sum_normal_z = reduce_sum_threadgroup(v_normal_z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_normals[3 * uint(isect_idx) + 2u] = sum_normal_z;
            }

            const uint grad_base = pixel_flat * cfg.channels;
            for (uint k = 0; k < cfg.channels; ++k) {
                float v_rgb = 0.0f;
                if (valid) {
                    v_rgb = fac * v_render_colors[grad_base + k];
                    if (k == cfg.channels - 1u) {
                        v_rgb += extra_depth_grad;
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
}
