// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kMaxAlpha = 0.99f;
constant float kTransmittanceThreshold = 1.0e-4f;
constant uint kMaxBlockSize = 256u;

inline uint batch_count_from_isect_range(
    int isect_range_start,
    int isect_range_end,
    uint block_size
) {
    const int isect_count = max(0, isect_range_end - isect_range_start);
    return uint((isect_count + int(block_size) - 1) / int(block_size));
}

inline bool contributes_to_pixel(
    float px,
    float py,
    float3 xy_opac,
    float3 conic,
    thread float& next_trans
) {
    const float2 delta = float2(xy_opac.x - px, xy_opac.y - py);
    const float sigma =
        0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
        conic.y * delta.x * delta.y;
    const float alpha = min(kMaxAlpha, xy_opac.z * exp(-sigma));
    if (sigma < 0.0f || alpha < kAlphaThreshold) {
        return false;
    }
    next_trans *= (1.0f - alpha);
    return true;
}

kernel void rasterize_to_indices_3dgs_count_kernel(
    device const float* transmittances [[buffer(0)]],
    device const float* means2d [[buffer(1)]],
    device const float* conics [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const int* tile_offsets [[buffer(4)]],
    device const int* flatten_ids [[buffer(5)]],
    device int* chunk_cnts [[buffer(6)]],
    constant uint& range_start [[buffer(7)]],
    constant uint& range_end [[buffer(8)]],
    constant uint& I [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& image_width [[buffer(11)]],
    constant uint& image_height [[buffer(12)]],
    constant uint& tile_size [[buffer(13)]],
    constant uint& tile_width [[buffer(14)]],
    constant uint& tile_height [[buffer(15)]],
    constant uint& n_tiles [[buffer(16)]],
    constant uint& total_tiles [[buffer(17)]],
    constant uint& n_isects [[buffer(18)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)total_tiles;
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
    const uint pix_id = i * image_width + j;
    const uint global_pix_id = image_id * image_height * image_width + pix_id;
    const uint tile_id = tile_y * tile_width + tile_x;
    const uint global_tile_id = image_id * n_tiles + tile_id;

    const int isect_range_start = tile_offsets[global_tile_id];
    const int isect_range_end =
        global_tile_id + 1u < total_tiles ? tile_offsets[global_tile_id + 1u] : int(n_isects);
    const uint num_batches = batch_count_from_isect_range(
        isect_range_start, isect_range_end, block_size
    );
    if (range_start >= num_batches) {
        // chunk_cnts is pre-zeroed by at::zeros in the launcher; no write needed.
        return;
    }

    // Keep the same threadgroup staging layout as rasterize_to_pixels_3dgs so
    // contributor queries and full rasterization walk identical tile-local data.
    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 conic_batch[kMaxBlockSize];

    bool done = !inside;
    float trans = inside ? transmittances[global_pix_id] : 0.0f;
    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    int cnt = 0;

    for (uint b = range_start; b < min(range_end, num_batches); ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_start = isect_range_start + int(block_size * b);
        const int idx = batch_start + int(local_idx);
        if (idx < isect_range_end) {
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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = min(block_size, uint(isect_range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            float next_trans = trans;
            if (!contributes_to_pixel(px, py, xy_opacity_batch[t], conic_batch[t], next_trans)) {
                continue;
            }
            if (next_trans <= kTransmittanceThreshold) {
                done = true;
                break;
            }
            cnt += 1;
            trans = next_trans;
        }
    }

    if (inside) {
        chunk_cnts[global_pix_id] = cnt;
    }
}

kernel void rasterize_to_indices_3dgs_emit_kernel(
    device const float* transmittances [[buffer(0)]],
    device const float* means2d [[buffer(1)]],
    device const float* conics [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const int* tile_offsets [[buffer(4)]],
    device const int* flatten_ids [[buffer(5)]],
    device const int* chunk_starts [[buffer(6)]],
    device long* gaussian_ids [[buffer(7)]],
    device long* pixel_ids [[buffer(8)]],
    constant uint& range_start [[buffer(9)]],
    constant uint& range_end [[buffer(10)]],
    constant uint& I [[buffer(11)]],
    constant uint& N [[buffer(12)]],
    constant uint& image_width [[buffer(13)]],
    constant uint& image_height [[buffer(14)]],
    constant uint& tile_size [[buffer(15)]],
    constant uint& tile_width [[buffer(16)]],
    constant uint& tile_height [[buffer(17)]],
    constant uint& n_tiles [[buffer(18)]],
    constant uint& total_tiles [[buffer(19)]],
    constant uint& n_isects [[buffer(20)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    (void)total_tiles;
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
    const uint pix_id = i * image_width + j;
    const uint global_pix_id = image_id * image_height * image_width + pix_id;
    const uint tile_id = tile_y * tile_width + tile_x;
    const uint global_tile_id = image_id * n_tiles + tile_id;

    const int isect_range_start = tile_offsets[global_tile_id];
    const int isect_range_end =
        global_tile_id + 1u < total_tiles ? tile_offsets[global_tile_id + 1u] : int(n_isects);
    const uint num_batches = batch_count_from_isect_range(
        isect_range_start, isect_range_end, block_size
    );
    if (range_start >= num_batches) {
        return;
    }

    // The emit pass reuses the same staged tile batch structure as the count
    // pass and the full rasterizer to keep traversal behavior aligned.
    threadgroup int id_batch[kMaxBlockSize];
    threadgroup float3 xy_opacity_batch[kMaxBlockSize];
    threadgroup float3 conic_batch[kMaxBlockSize];

    bool done = !inside;
    float trans = inside ? transmittances[global_pix_id] : 0.0f;
    const float px = float(j) + 0.5f;
    const float py = float(i) + 0.5f;
    int cnt = 0;
    const int base = inside ? chunk_starts[global_pix_id] : 0;

    for (uint b = range_start; b < min(range_end, num_batches); ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_start = isect_range_start + int(block_size * b);
        const int idx = batch_start + int(local_idx);
        if (idx < isect_range_end) {
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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = min(block_size, uint(isect_range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            float next_trans = trans;
            if (!contributes_to_pixel(px, py, xy_opacity_batch[t], conic_batch[t], next_trans)) {
                continue;
            }
            if (next_trans <= kTransmittanceThreshold) {
                done = true;
                break;
            }
            const int g = id_batch[t];
            gaussian_ids[base + cnt] = long(g % int(N));
            pixel_ids[base + cnt] = long(global_pix_id);
            cnt += 1;
            trans = next_trans;
        }
    }
}
