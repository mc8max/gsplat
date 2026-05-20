// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kMaxAlpha = 0.99f;
constant float kTransmittanceThreshold = 1.0e-4f;
constant float kMinOneMinusAlpha = 1.0e-6f;
constant uint kMaxBlockSize = 256u;
// Cache small, common channel buckets in threadgroup memory to avoid
// repeatedly fetching per-Gaussian colors for every pixel in the tile.
constant uint kCachedChannels = 17u;

// Parallel tree reduction over a threadgroup.
// Reduces block_size values in log2(block_size) barrier steps instead of
// the O(block_size) serial loop.  block_size must be a power of two.
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

kernel void rasterize_to_pixels_3dgs_fwd_kernel(
    device const float* means2d [[buffer(0)]],
    device const float* conics [[buffer(1)]],
    device const float* colors [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* backgrounds [[buffer(4)]],
    device const bool* masks [[buffer(5)]],
    device const int* tile_offsets [[buffer(6)]],
    device const int* flatten_ids [[buffer(7)]],
    device float* render_colors [[buffer(8)]],
    device float* render_alphas [[buffer(9)]],
    device int* last_ids [[buffer(10)]],
    constant uint& I [[buffer(11)]],
    constant uint& N [[buffer(12)]],
    constant uint& channels [[buffer(13)]],
    constant uint& tile_size [[buffer(14)]],
    constant uint& tile_width [[buffer(15)]],
    constant uint& tile_height [[buffer(16)]],
    constant uint& n_tiles [[buffer(17)]],
    constant uint& total_tiles [[buffer(18)]],
    constant uint& n_isects [[buffer(19)]],
    constant uint& packed [[buffer(20)]],
    constant uint& image_width [[buffer(21)]],
    constant uint& image_height [[buffer(22)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)N;
    (void)packed;
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
            render_alphas[pixel_flat] = 0.0f;
            last_ids[pixel_flat] = 0;
        }
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < total_tiles ? tile_offsets[global_tile_id + 1u] : int(n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 conic_batch[kMaxBlockSize];
    threadgroup float color_batch[kMaxBlockSize * kCachedChannels];

    float T = 1.0f;
    int cur_idx = 0;
    bool done = !inside;
    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    const uint color_base = pixel_flat * channels;

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
            conic_batch[local_idx] = float3(
                conics[3 * g],
                conics[3 * g + 1],
                conics[3 * g + 2]
            );
            if (channels <= kCachedChannels) {
                const uint g_color_base = uint(g) * channels;
                const uint color_offset = local_idx * kCachedChannels;
                for (uint k = 0; k < channels; ++k) {
                    color_batch[color_offset + k] = colors[g_color_base + k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = min(block_size, uint(range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = float2(xy_opac.x - px, xy_opac.y - py);
            const float3 conic = conic_batch[t];
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
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
            const uint color_offset = t * kCachedChannels;
            const uint g_color_base = uint(g) * channels;
            for (uint k = 0; k < channels; ++k) {
                const float color = channels <= kCachedChannels
                    ? color_batch[color_offset + k]
                    : colors[g_color_base + k];
                render_colors[color_base + k] += color * vis;
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
        last_ids[pixel_flat] = cur_idx;
    }
}

kernel void rasterize_to_pixels_3dgs_bwd_kernel(
    device const float* means2d [[buffer(0)]],
    device const float* conics [[buffer(1)]],
    device const float* colors [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* backgrounds [[buffer(4)]],
    device const bool* masks [[buffer(5)]],
    device const int* tile_offsets [[buffer(6)]],
    device const int* flatten_ids [[buffer(7)]],
    device const float* render_alphas [[buffer(8)]],
    device const int* last_ids [[buffer(9)]],
    device const float* v_render_colors [[buffer(10)]],
    device const float* v_render_alphas [[buffer(11)]],
    device float* tmp_means2d_abs [[buffer(12)]],
    device float* tmp_means2d [[buffer(13)]],
    device float* tmp_conics [[buffer(14)]],
    device float* tmp_colors [[buffer(15)]],
    device float* tmp_opacities [[buffer(16)]],
    constant uint& I [[buffer(17)]],
    constant uint& N [[buffer(18)]],
    constant uint& channels [[buffer(19)]],
    constant uint& tile_size [[buffer(20)]],
    constant uint& tile_width [[buffer(21)]],
    constant uint& tile_height [[buffer(22)]],
    constant uint& n_tiles [[buffer(23)]],
    constant uint& total_tiles [[buffer(24)]],
    constant uint& n_isects [[buffer(25)]],
    constant uint& packed [[buffer(26)]],
    constant uint& image_width [[buffer(27)]],
    constant uint& image_height [[buffer(28)]],
    constant uint& absgrad [[buffer(29)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)N;
    (void)packed;
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
    const uint pixel_flat = min((image_id * image_height + i) * image_width + j, I * image_height * image_width - 1u);
    const uint tile_id = tile_y * tile_width + tile_x;
    const uint global_tile_id = image_id * n_tiles + tile_id;

    if (masks != nullptr && !masks[global_tile_id]) {
        return;
    }

    const int range_start = tile_offsets[global_tile_id];
    const int range_end =
        global_tile_id + 1u < total_tiles ? tile_offsets[global_tile_id + 1u] : int(n_isects);

    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 conic_batch[kMaxBlockSize];
    threadgroup float color_batch[kMaxBlockSize * kCachedChannels];
    threadgroup float reduce_scratch[kMaxBlockSize];

    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    const float T_final = 1.0f - render_alphas[pixel_flat];
    float T = T_final;
    float buffer_dot = 0.0f;
    const int bin_final = inside ? last_ids[pixel_flat] : 0;

    float bg_dot = 0.0f;
    if (inside && backgrounds != nullptr) {
        const uint bg_base = image_id * channels;
        const uint grad_base = pixel_flat * channels;
        for (uint k = 0; k < channels; ++k) {
            bg_dot += backgrounds[bg_base + k] * v_render_colors[grad_base + k];
        }
    }

    for (uint b = 0u; b < uint((range_end - range_start + int(block_size) - 1) / int(block_size)); ++b) {
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
            conic_batch[local_idx] = float3(
                conics[3 * g],
                conics[3 * g + 1],
                conics[3 * g + 2]
            );
            if (channels <= kCachedChannels) {
                const uint g_color_base = uint(g) * channels;
                const uint color_offset = local_idx * kCachedChannels;
                for (uint k = 0; k < channels; ++k) {
                    color_batch[color_offset + k] = colors[g_color_base + k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < uint(batch_size); ++t) {
            // isect_idx is always in [range_start, range_end) because
            // t < batch_size = min(block_size, batch_end + 1 - range_start),
            // so isect_idx = batch_end - t >= range_start >= 0.
            const int isect_idx = batch_end - int(t);
            bool valid = inside && (isect_idx <= bin_final);
            float alpha = 0.0f;
            float opac = 0.0f;
            float2 delta = float2(0.0f);
            float3 conic = float3(0.0f);
            float vis = 0.0f;
            float dot_gc = 0.0f;

            if (valid) {
                conic = conic_batch[t];
                const float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = float2(xy_opac.x - px, xy_opac.y - py);
                const float sigma =
                    0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                vis = exp(-sigma);
                alpha = min(kMaxAlpha, opac * vis);
                if (sigma < 0.0f || alpha < kAlphaThreshold) {
                    valid = false;
                }
            }

            float v_conic_x = 0.0f;
            float v_conic_y = 0.0f;
            float v_conic_z = 0.0f;
            float v_xy_x = 0.0f;
            float v_xy_y = 0.0f;
            float v_xy_abs_x = 0.0f;
            float v_xy_abs_y = 0.0f;
            float v_opacity = 0.0f;

            if (valid) {
                const uint grad_base = pixel_flat * channels;
                const int g = id_batch[t];
                const uint color_offset = t * kCachedChannels;
                const uint color_base = uint(g) * channels;
                const float ra = 1.0f / max(kMinOneMinusAlpha, 1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                for (uint k = 0; k < channels; ++k) {
                    const float color = channels <= kCachedChannels
                        ? color_batch[color_offset + k]
                        : colors[color_base + k];
                    dot_gc += color * v_render_colors[grad_base + k];
                }

                float v_alpha = dot_gc * T - buffer_dot * ra;
                v_alpha += T_final * ra * v_render_alphas[pixel_flat];
                if (backgrounds != nullptr) {
                    v_alpha += -T_final * ra * bg_dot;
                }

                if (opac * vis <= kMaxAlpha) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_x = 0.5f * v_sigma * delta.x * delta.x;
                    v_conic_y = v_sigma * delta.x * delta.y;
                    v_conic_z = 0.5f * v_sigma * delta.y * delta.y;
                    v_xy_x = v_sigma * (conic.x * delta.x + conic.y * delta.y);
                    v_xy_y = v_sigma * (conic.y * delta.x + conic.z * delta.y);
                    if (absgrad != 0u) {
                        v_xy_abs_x = abs(v_xy_x);
                        v_xy_abs_y = abs(v_xy_y);
                    }
                    v_opacity = vis * v_alpha;
                }

                buffer_dot += dot_gc * fac;
            }

            // tmp_* buffers are pre-zeroed by at::zeros in the launcher;
            // no per-slot zeroing is needed here.
            const float sum_conic_x = reduce_sum_threadgroup(v_conic_x, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_conics[3 * uint(isect_idx)] = sum_conic_x;
            }
            const float sum_conic_y = reduce_sum_threadgroup(v_conic_y, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_conics[3 * uint(isect_idx) + 1u] = sum_conic_y;
            }
            const float sum_conic_z = reduce_sum_threadgroup(v_conic_z, reduce_scratch, local_idx, block_size);
            if (local_idx == 0u) {
                tmp_conics[3 * uint(isect_idx) + 2u] = sum_conic_z;
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

            const uint grad_base = pixel_flat * channels;
            for (uint k = 0; k < channels; ++k) {
                float v_rgb = 0.0f;
                if (valid) {
                    v_rgb = alpha * T * v_render_colors[grad_base + k];
                }
                const float sum_rgb = reduce_sum_threadgroup(v_rgb, reduce_scratch, local_idx, block_size);
                if (local_idx == 0u) {
                    tmp_colors[uint(isect_idx) * channels + k] = sum_rgb;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
