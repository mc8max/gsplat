// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

kernel void intersect_tile_count_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device int* tiles_per_gauss [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    constant uint& tile_size [[buffer(4)]],
    constant uint& tile_width [[buffer(5)]],
    constant uint& tile_height [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n_elements) {
        return;
    }

    const int radius_x = radii[idx * 2];
    const int radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        tiles_per_gauss[idx] = 0;
        return;
    }

    // AABB tile overlap count
    const float inv_tile = 1.0f / float(tile_size);
    const float tile_x = means2d[idx * 2] * inv_tile;
    const float tile_y = means2d[idx * 2 + 1] * inv_tile;
    const float tile_radius_x = float(radius_x) * inv_tile;
    const float tile_radius_y = float(radius_y) * inv_tile;

    const int tile_min_x = min(max(0, int(floor(tile_x - tile_radius_x))), int(tile_width));
    const int tile_min_y = min(max(0, int(floor(tile_y - tile_radius_y))), int(tile_height));
    const int tile_max_x = min(max(0, int(ceil(tile_x + tile_radius_x))), int(tile_width));
    const int tile_max_y = min(max(0, int(ceil(tile_y + tile_radius_y))), int(tile_height));

    tiles_per_gauss[idx] = (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x);
}

kernel void intersect_tile_emit_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* depths [[buffer(2)]],
    device const long* image_ids [[buffer(3)]],
    device const long* cum_tiles_per_gauss [[buffer(4)]],
    device long* isect_ids [[buffer(5)]],
    device int* flatten_ids [[buffer(6)]],
    constant uint& n_elements [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& tile_size [[buffer(9)]],
    constant uint& tile_width [[buffer(10)]],
    constant uint& tile_height [[buffer(11)]],
    constant uint& tile_n_bits [[buffer(12)]],
    constant uint& packed [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n_elements) {
        return;
    }

    const int radius_x = radii[idx * 2];
    const int radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        return;
    }

    // AABB tile overlap count
    const float inv_tile = 1.0f / float(tile_size);
    const float tile_x = means2d[idx * 2] * inv_tile;
    const float tile_y = means2d[idx * 2 + 1] * inv_tile;
    const float tile_radius_x = float(radius_x) * inv_tile;
    const float tile_radius_y = float(radius_y) * inv_tile;

    const int tile_min_x = min(max(0, int(floor(tile_x - tile_radius_x))), int(tile_width));
    const int tile_min_y = min(max(0, int(floor(tile_y - tile_radius_y))), int(tile_height));
    const int tile_max_x = min(max(0, int(ceil(tile_x + tile_radius_x))), int(tile_width));
    const int tile_max_y = min(max(0, int(ceil(tile_y + tile_radius_y))), int(tile_height));

    const ulong image_id = packed != 0u ? ulong(image_ids[idx]) : ulong(idx / N);
    const ulong depth_bits = ulong(as_type<uint>(depths[idx]));

    // index to tiles intersection list
    ulong cur_idx = idx == 0u ? 0ul : ulong(cum_tiles_per_gauss[idx - 1u]);

    // iterate over rows and columns of intersected tiles
    for (int y = tile_min_y; y < tile_max_y; ++y) {
        for (int x = tile_min_x; x < tile_max_x; ++x) {
            const ulong tile_id = ulong(y) * ulong(tile_width) + ulong(x);
            const ulong upper = (image_id << tile_n_bits) | tile_id;
            isect_ids[cur_idx] = long((upper << 32) | depth_bits);
            flatten_ids[cur_idx] = int(idx);
            ++cur_idx;
        }
    }
}
