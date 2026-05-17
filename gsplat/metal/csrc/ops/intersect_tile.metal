// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

// Same alpha cutoff used by CUDA when turning opacity into a finite ellipse.
constant float kAlphaThreshold = 1.0f / 255.0f;
// Same hard cap used by gsplat to keep the Gaussian support bounded.
constant float kGaussianExtend = 3.33f;

// Intersect an opacity-thresholded ellipse with one sweep line.
//
// The conic is centered at `p` and described by:
//   A dx^2 + 2B dx dy + C dy^2 + t = 0
//
// `is_y` selects the sweep orientation:
// - `false`: sweep vertical lines, return y-range
// - `true`: sweep horizontal lines, return x-range
//
// The result is the two coordinates where the chosen line cuts the ellipse in
// the orthogonal axis. This mirrors the CUDA helper used by SNUGBOX/AccuTile.
inline float2 accutile_ellipse_intersection(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 p,
    bool is_y,
    float coord
) {
    const float p_u = is_y ? p.y : p.x;
    const float p_v = is_y ? p.x : p.y;
    const float coeff = is_y ? A : C;

    const float h = coord - p_u;
    const float sqrt_term = sqrt(disc * h * h + t * coeff);

    return float2(
        (-B * h - sqrt_term) / coeff + p_v,
        (-B * h + sqrt_term) / coeff + p_v
    );
}

// Sweep a SNUGBOX tile rectangle one line at a time and either count or emit
// tiles touched by the opacity-thresholded ellipse.
//
// The function serves both count and emit passes:
// - if `isect_ids == nullptr`, only the number of touched tiles is returned
// - otherwise one `(isect_id, flatten_id)` record is emitted per touched tile
//
// `upper_prefix` contains the already-shifted image prefix, while `depth_bits`
// are the raw float32 depth bits that remain in the low 32 bits of `isect_id`.
inline uint accutile_process_tiles(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 p,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    uint tile_size,
    uint tile_width,
    bool is_y,
    ulong upper_prefix,
    ulong depth_bits,
    uint flatten_idx,
    device long* isect_ids,
    device int* flatten_ids,
    thread ulong& cur_idx
) {
    const float block = float(tile_size);

    if (is_y) {
        // Reorient the sweep so the main loop always advances along the first
        // rectangle axis and the inner emitted coordinate is the second axis.
        rect_min = int2(rect_min.y, rect_min.x);
        rect_max = int2(rect_max.y, rect_max.x);
        bbox_min = float2(bbox_min.y, bbox_min.x);
        bbox_max = float2(bbox_max.y, bbox_max.x);
        bbox_argmin = float2(bbox_argmin.y, bbox_argmin.x);
        bbox_argmax = float2(bbox_argmax.y, bbox_argmax.x);
    }

    uint tiles_count = 0u;
    float2 intersect_min_line;
    float2 intersect_max_line = float2(bbox_max.y, bbox_min.y);
    float ellipse_min;
    float ellipse_max;
    float min_line = float(rect_min.x) * block;
    float max_line;

    if (bbox_min.x <= min_line) {
        // The first sweep line starts inside the snug bounding box, so it must
        // intersect the ellipse on this boundary.
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, is_y, min_line);
    } else {
        // The first active line starts after the snug minimum; the previous
        // boundary contributes no interior interval.
        intersect_min_line = intersect_max_line;
    }

    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + block;
        if (max_line <= bbox_max.x) {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, is_y, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            // The ellipse reaches its minimum orthogonal extent inside this
            // strip, so clamp to the true snugbox bound rather than the line
            // intersections on the strip edges.
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            // Same handling for the maximum orthogonal extent.
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert the continuous interval to an inclusive/exclusive tile span.
        const int min_tile_v = max(rect_min.y, min(rect_max.y, int(ellipse_min / block)));
        const int max_tile_v = min(rect_max.y, max(rect_min.y, int(ellipse_max / block + 1.0f)));
        tiles_count += uint(max_tile_v - min_tile_v);

        if (isect_ids != nullptr) {
            for (int v = min_tile_v; v < max_tile_v; ++v) {
                // `tile_id` is always row-major in the original image grid even
                // when the sweep orientation has been swapped.
                const ulong tile_id = is_y ? ulong(u) * ulong(tile_width) + ulong(v)
                                           : ulong(v) * ulong(tile_width) + ulong(u);
                isect_ids[cur_idx] = long((upper_prefix | tile_id) << 32 | depth_bits);
                flatten_ids[cur_idx] = int(flatten_idx);
                ++cur_idx;
            }
        }

        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }

    return tiles_count;
}

// Count pass for the ellipse-aware SNUGBOX/AccuTile path.
//
// One thread handles one flattened Gaussian entry and stores the number of
// intersected tiles into `tiles_per_gauss[idx]`.
kernel void intersect_tile_count_accutile_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* conics [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device int* tiles_per_gauss [[buffer(4)]],
    constant uint& n_elements [[buffer(5)]],
    constant uint& tile_size [[buffer(6)]],
    constant uint& tile_width [[buffer(7)]],
    constant uint& tile_height [[buffer(8)]],
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

    const float2 mean2d = float2(means2d[idx * 2], means2d[idx * 2 + 1]);
    const float A = conics[idx * 3];
    const float B = conics[idx * 3 + 1];
    const float C = conics[idx * 3 + 2];
    const float disc = B * B - A * C;
    const float opacity = opacities[idx];
    // Match CUDA: truncate the ellipse at the opacity-aware contour, capped by
    // the same Gaussian extent budget used elsewhere in gsplat.
    const float t = min(
        kGaussianExtend * kGaussianExtend,
        2.0f * log(opacity / kAlphaThreshold)
    );

    // Build the tight axis-aligned SNUGBOX around the thresholded ellipse.
    const float neg_t_over_disc = -t / disc;
    const float x_extent = sqrt(neg_t_over_disc * C);
    const float y_extent = sqrt(neg_t_over_disc * A);
    const float2 bbox_min = float2(mean2d.x - x_extent, mean2d.y - y_extent);
    const float2 bbox_max = float2(mean2d.x + x_extent, mean2d.y + y_extent);
    const float2 bbox_argmin = float2(mean2d.y + B * x_extent / C, mean2d.x + B * y_extent / A);
    const float2 bbox_argmax = float2(mean2d.y - B * x_extent / C, mean2d.x - B * y_extent / A);

    const float tile_size_f = float(tile_size);
    // Clamp the snugbox against the valid tile domain before sweeping.
    const int2 rect_min = int2(
        max(0, min(int(tile_width), int(bbox_min.x / tile_size_f))),
        max(0, min(int(tile_height), int(bbox_min.y / tile_size_f)))
    );
    const int2 rect_max = int2(
        max(0, min(int(tile_width), int(bbox_max.x / tile_size_f + 1.0f))),
        max(0, min(int(tile_height), int(bbox_max.y / tile_size_f + 1.0f)))
    );

    const int y_span = rect_max.y - rect_min.y;
    const int x_span = rect_max.x - rect_min.x;
    if (y_span * x_span == 0) {
        tiles_per_gauss[idx] = 0;
        return;
    }

    // Sweep along the smaller span to reduce the amount of line work.
    const bool is_y = y_span < x_span;
    ulong cur_idx = 0ul;
    tiles_per_gauss[idx] = int(accutile_process_tiles(
        A,
        B,
        C,
        disc,
        t,
        mean2d,
        bbox_min,
        bbox_max,
        bbox_argmin,
        bbox_argmax,
        rect_min,
        rect_max,
        tile_size,
        tile_width,
        is_y,
        0ul,
        0ul,
        idx,
        nullptr,
        nullptr,
        cur_idx
    ));
}

// Emit pass for the ellipse-aware SNUGBOX/AccuTile path.
//
// The write order must stay compatible with CUDA so optional downstream
// sorting and offset encoding observe the same key layout.
kernel void intersect_tile_emit_accutile_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* depths [[buffer(2)]],
    device const float* conics [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const long* image_ids [[buffer(5)]],
    device const long* cum_tiles_per_gauss [[buffer(6)]],
    device long* isect_ids [[buffer(7)]],
    device int* flatten_ids [[buffer(8)]],
    constant uint& n_elements [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& tile_size [[buffer(11)]],
    constant uint& tile_width [[buffer(12)]],
    constant uint& tile_height [[buffer(13)]],
    constant uint& tile_n_bits [[buffer(14)]],
    constant uint& packed [[buffer(15)]],
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

    const float2 mean2d = float2(means2d[idx * 2], means2d[idx * 2 + 1]);
    // In packed mode the image id comes from the explicit packed metadata;
    // otherwise the flattened `(image, gaussian)` index is decoded as `idx / N`.
    const ulong image_id = packed != 0u ? ulong(image_ids[idx]) : ulong(idx / N);
    // Preserve CUDA ordering by packing the raw float bit pattern into the
    // low 32 bits of the final sort key.
    const ulong depth_bits = ulong(as_type<uint>(depths[idx]));
    const ulong upper_prefix = image_id << tile_n_bits;
    ulong cur_idx = idx == 0u ? 0ul : ulong(cum_tiles_per_gauss[idx - 1u]);

    const float A = conics[idx * 3];
    const float B = conics[idx * 3 + 1];
    const float C = conics[idx * 3 + 2];
    const float disc = B * B - A * C;
    const float opacity = opacities[idx];
    // Same opacity-aware threshold as CUDA.
    const float t = min(
        kGaussianExtend * kGaussianExtend,
        2.0f * log(opacity / kAlphaThreshold)
    );

    const float neg_t_over_disc = -t / disc;
    const float x_extent = sqrt(neg_t_over_disc * C);
    const float y_extent = sqrt(neg_t_over_disc * A);
    const float2 bbox_min = float2(mean2d.x - x_extent, mean2d.y - y_extent);
    const float2 bbox_max = float2(mean2d.x + x_extent, mean2d.y + y_extent);
    const float2 bbox_argmin = float2(mean2d.y + B * x_extent / C, mean2d.x + B * y_extent / A);
    const float2 bbox_argmax = float2(mean2d.y - B * x_extent / C, mean2d.x - B * y_extent / A);

    const float tile_size_f = float(tile_size);
    // Clamp the snugbox against the valid tile domain before sweeping.
    const int2 rect_min = int2(
        max(0, min(int(tile_width), int(bbox_min.x / tile_size_f))),
        max(0, min(int(tile_height), int(bbox_min.y / tile_size_f)))
    );
    const int2 rect_max = int2(
        max(0, min(int(tile_width), int(bbox_max.x / tile_size_f + 1.0f))),
        max(0, min(int(tile_height), int(bbox_max.y / tile_size_f + 1.0f)))
    );

    const int y_span = rect_max.y - rect_min.y;
    const int x_span = rect_max.x - rect_min.x;
    if (y_span * x_span == 0) {
        return;
    }

    // Emit one compact `(isect_id, flatten_id)` record per intersected tile.
    const bool is_y = y_span < x_span;
    accutile_process_tiles(
        A,
        B,
        C,
        disc,
        t,
        mean2d,
        bbox_min,
        bbox_max,
        bbox_argmin,
        bbox_argmax,
        rect_min,
        rect_max,
        tile_size,
        tile_width,
        is_y,
        upper_prefix,
        depth_bits,
        idx,
        isect_ids,
        flatten_ids,
        cur_idx
    );
}

// Count pass for the simple axis-aligned tile box fallback.
//
// This is the coarse AABB path used when no conic/opacity data is provided.
kernel void intersect_tile_count_aabb_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* conics [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device int* tiles_per_gauss [[buffer(4)]],
    constant uint& n_elements [[buffer(5)]],
    constant uint& tile_size [[buffer(6)]],
    constant uint& tile_width [[buffer(7)]],
    constant uint& tile_height [[buffer(8)]],
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

    const float2 mean2d = float2(means2d[idx * 2], means2d[idx * 2 + 1]);
    // Convert center and radii from pixel space to tile space.
    const float inv_tile = 1.0f / float(tile_size);
    const float tile_x = mean2d.x * inv_tile;
    const float tile_y = mean2d.y * inv_tile;
    const float tile_radius_x = float(radius_x) * inv_tile;
    const float tile_radius_y = float(radius_y) * inv_tile;

    // Clamp against the valid tile grid. As in CUDA, mins are inclusive and
    // maxes are exclusive.
    const int tile_min_x = min(max(0, int(floor(tile_x - tile_radius_x))), int(tile_width));
    const int tile_min_y = min(max(0, int(floor(tile_y - tile_radius_y))), int(tile_height));
    const int tile_max_x = min(max(0, int(ceil(tile_x + tile_radius_x))), int(tile_width));
    const int tile_max_y = min(max(0, int(ceil(tile_y + tile_radius_y))), int(tile_height));

    // `tile_min` is inclusive and `tile_max` is exclusive, so the count is the
    // rectangle area in tile coordinates.
    tiles_per_gauss[idx] = (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x);
}

// Emit pass for the simple axis-aligned tile box fallback.
//
// Each thread emits one contiguous block of outputs starting at
// `cum_tiles_per_gauss[idx - 1]`.
kernel void intersect_tile_emit_aabb_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* depths [[buffer(2)]],
    device const float* conics [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const long* image_ids [[buffer(5)]],
    device const long* cum_tiles_per_gauss [[buffer(6)]],
    device long* isect_ids [[buffer(7)]],
    device int* flatten_ids [[buffer(8)]],
    constant uint& n_elements [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& tile_size [[buffer(11)]],
    constant uint& tile_width [[buffer(12)]],
    constant uint& tile_height [[buffer(13)]],
    constant uint& tile_n_bits [[buffer(14)]],
    constant uint& packed [[buffer(15)]],
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

    const float2 mean2d = float2(means2d[idx * 2], means2d[idx * 2 + 1]);
    // In packed mode the image id is supplied explicitly. In unpacked mode the
    // flattened index stores `N` gaussians for each image consecutively.
    const ulong image_id = packed != 0u ? ulong(image_ids[idx]) : ulong(idx / N);
    // Keep the raw float depth bits in the low 32 bits for CUDA key parity.
    const ulong depth_bits = ulong(as_type<uint>(depths[idx]));
    const ulong upper_prefix = image_id << tile_n_bits;
    ulong cur_idx = idx == 0u ? 0ul : ulong(cum_tiles_per_gauss[idx - 1u]);

    // Convert center and radii from pixel space to tile space.
    const float inv_tile = 1.0f / float(tile_size);
    const float tile_x = mean2d.x * inv_tile;
    const float tile_y = mean2d.y * inv_tile;
    const float tile_radius_x = float(radius_x) * inv_tile;
    const float tile_radius_y = float(radius_y) * inv_tile;

    const int tile_min_x = min(max(0, int(floor(tile_x - tile_radius_x))), int(tile_width));
    const int tile_min_y = min(max(0, int(floor(tile_y - tile_radius_y))), int(tile_height));
    const int tile_max_x = min(max(0, int(ceil(tile_x + tile_radius_x))), int(tile_width));
    const int tile_max_y = min(max(0, int(ceil(tile_y + tile_radius_y))), int(tile_height));

    // Emit tiles in row-major order to preserve the CUDA traversal order prior
    // to optional sorting.
    for (int y = tile_min_y; y < tile_max_y; ++y) {
        for (int x = tile_min_x; x < tile_max_x; ++x) {
            const ulong tile_id = ulong(y) * ulong(tile_width) + ulong(x);
            isect_ids[cur_idx] = long(((upper_prefix | tile_id) << 32) | depth_bits);
            flatten_ids[cur_idx] = int(idx);
            ++cur_idx;
        }
    }
}
