// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

constant float kPi = 3.14159265358979323846f;

struct LidarMeta {
    uint n_bins_azimuth;
    uint n_bins_elevation;
    uint cdf_resolution_azimuth;
    uint cdf_resolution_elevation;
    float angle_to_pixel_scaling_factor;
    float fov_horiz_start;
    float fov_horiz_span;
    float fov_vert_start;
    float fov_vert_span;
    float fov_eps;
    uint spinning_direction;
};

inline float relative_clock_rotation(float begin, float end, uint direction) {
    return direction == 0u ? (begin - end) : (end - begin);
}

inline float relative_angle(float angle_start, float angle_end, uint direction, float scale) {
    const float period = scale * 2.0f * kPi;
    float rel_angle = relative_clock_rotation(angle_start, angle_end, direction);
    if (-period < rel_angle && rel_angle < 2.0f * period) {
        rel_angle = rel_angle >= period ? rel_angle - period : rel_angle;
        rel_angle = rel_angle < 0.0f ? rel_angle + period : rel_angle;
        return rel_angle;
    }
    rel_angle = fmod(rel_angle, period);
    return rel_angle < 0.0f ? rel_angle + period : rel_angle;
}

inline float2 relative_sensor_angles(float elevation, float azimuth, constant LidarMeta& meta) {
    const float rel_elevation =
        relative_clock_rotation(meta.fov_vert_start * meta.angle_to_pixel_scaling_factor,
                                elevation,
                                0u);
    const float rel_azimuth =
        relative_angle(meta.fov_horiz_start * meta.angle_to_pixel_scaling_factor,
                       azimuth,
                       meta.spinning_direction,
                       meta.angle_to_pixel_scaling_factor);
    return float2(rel_elevation, rel_azimuth);
}

inline int cdf_region_sum(
    device const int* raycdf,
    int raycdf_stride,
    int range_min_el,
    int range_max_el,
    int range_min_az,
    int range_max_az
) {
    return raycdf[range_max_el * raycdf_stride + range_max_az]
           - raycdf[range_min_el * raycdf_stride + range_max_az]
           - raycdf[range_max_el * raycdf_stride + range_min_az]
           + raycdf[range_min_el * raycdf_stride + range_min_az];
}

inline bool has_any_rays_in_tile(
    device const int* cdf_dense_ray_mask,
    constant LidarMeta& meta,
    int range_min_el,
    int range_max_el,
    int range_min_az,
    int range_max_az
) {
    if (range_min_az >= range_max_az) {
        return false;
    }
    if (range_min_az <= 0 && range_max_az >= int(meta.cdf_resolution_azimuth)) {
        return true;
    }
    const int stride = int(meta.cdf_resolution_azimuth) + 1;
    const int num_rays = cdf_region_sum(
        cdf_dense_ray_mask,
        stride,
        range_min_el,
        range_max_el,
        range_min_az,
        range_max_az);
    return num_rays > 0;
}

inline int sample_dense_floor(float pix, float span, uint resolution) {
    return int(floor((pix / span) * float(resolution)));
}

inline int sample_dense_ceil(float pix, float span, uint resolution) {
    return int(ceil((pix / span) * float(resolution)));
}

inline int sample_tile_floor(float pix, float span, uint bins) {
    return int(floor((pix / span) * float(bins)));
}

inline int sample_tile_ceil(float pix, float span, uint bins) {
    return int(ceil((pix / span) * float(bins)));
}

inline void compute_lidar_tile_ranges(
    float2 mean2d,
    int radius_x,
    int radius_y,
    constant LidarMeta& meta,
    device const int* cdf_elevation,
    device const int* cdf_dense_ray_mask,
    thread bool& valid,
    thread int& tile_min_el,
    thread int& tile_max_el,
    thread int2& range_a,
    thread int2& range_b,
    thread bool& periodic_az
) {
    valid = false;
    range_a = int2(0, 0);
    range_b = int2(0, 0);
    tile_min_el = 0;
    tile_max_el = 0;

    if (radius_x <= 0 || radius_y <= 0) {
        return;
    }

    const float azimuth_pix = mean2d.x;
    const float elevation_pix = mean2d.y;
    const float fov_span_pix_el = meta.fov_vert_span * meta.angle_to_pixel_scaling_factor;
    const float fov_span_pix_az = meta.fov_horiz_span * meta.angle_to_pixel_scaling_factor;
    const float full_circle_pix = 2.0f * kPi * meta.angle_to_pixel_scaling_factor;
    const float2 mean_rel = relative_sensor_angles(elevation_pix, azimuth_pix, meta);

    const float beg_az = mean_rel.y - float(radius_x);
    const float end_az_raw = mean_rel.y + float(radius_x);
    const float beg_el = clamp(mean_rel.x - float(radius_y), 0.0f, fov_span_pix_el);
    const float end_el = clamp(mean_rel.x + float(radius_y), 0.0f, fov_span_pix_el);
    const bool full_cover = (beg_az <= 0.0f) && (end_az_raw >= fov_span_pix_az);
    const float end_az = min(end_az_raw, beg_az + full_circle_pix);
    const bool underflows = (beg_az < 0.0f) && !full_cover;
    const bool overflows = (end_az > full_circle_pix) && !full_cover;

    float begA_pix_az;
    float endA_pix_az;
    if (full_cover) {
        begA_pix_az = 0.0f;
        endA_pix_az = fov_span_pix_az;
    } else if (underflows) {
        begA_pix_az = 0.0f;
        endA_pix_az = end_az;
    } else if (overflows) {
        begA_pix_az = beg_az;
        endA_pix_az = full_circle_pix;
    } else {
        begA_pix_az = beg_az;
        endA_pix_az = end_az;
    }
    begA_pix_az = clamp(begA_pix_az, 0.0f, fov_span_pix_az);
    endA_pix_az = clamp(endA_pix_az, 0.0f, fov_span_pix_az);

    const int min_dense_el = sample_dense_floor(beg_el, fov_span_pix_el, meta.cdf_resolution_elevation);
    const int max_dense_el = sample_dense_ceil(end_el, fov_span_pix_el, meta.cdf_resolution_elevation);
    if (min_dense_el >= max_dense_el) {
        return;
    }
    tile_min_el = cdf_elevation[min_dense_el];
    tile_max_el = min(cdf_elevation[max_dense_el - 1] + 1, int(meta.n_bins_elevation));

    const int begA_dense = sample_dense_floor(begA_pix_az, fov_span_pix_az, meta.cdf_resolution_azimuth);
    const int endA_dense = sample_dense_ceil(endA_pix_az, fov_span_pix_az, meta.cdf_resolution_azimuth);
    const bool has_raysA = has_any_rays_in_tile(
        cdf_dense_ray_mask, meta, min_dense_el, max_dense_el, begA_dense, endA_dense);

    int begA_tile = 0;
    int endA_tile = 0;
    if (has_raysA) {
        begA_tile = sample_tile_floor(begA_pix_az, fov_span_pix_az, meta.n_bins_azimuth);
        endA_tile = sample_tile_ceil(endA_pix_az, fov_span_pix_az, meta.n_bins_azimuth);
    }

    bool has_raysB = false;
    int begB_tile = 0;
    int endB_tile = 0;
    if (underflows || overflows) {
        const float begB_pix_az =
            clamp(underflows ? (beg_az + full_circle_pix) : 0.0f, 0.0f, fov_span_pix_az);
        const float endB_pix_az =
            clamp(underflows ? full_circle_pix : (end_az - full_circle_pix), 0.0f, fov_span_pix_az);
        const int begB_dense = sample_dense_floor(begB_pix_az, fov_span_pix_az, meta.cdf_resolution_azimuth);
        const int endB_dense = sample_dense_ceil(endB_pix_az, fov_span_pix_az, meta.cdf_resolution_azimuth);
        has_raysB = has_any_rays_in_tile(
            cdf_dense_ray_mask, meta, min_dense_el, max_dense_el, begB_dense, endB_dense);
        if (has_raysB) {
            begB_tile = sample_tile_floor(begB_pix_az, fov_span_pix_az, meta.n_bins_azimuth);
            endB_tile = sample_tile_ceil(endB_pix_az, fov_span_pix_az, meta.n_bins_azimuth);
        }
    }

    if (!has_raysA && !has_raysB) {
        return;
    }

    periodic_az = fov_span_pix_az >= full_circle_pix;
    range_a = has_raysA ? int2(begA_tile, endA_tile) : int2(0, 0);
    range_b = has_raysB ? int2(begB_tile, endB_tile) : int2(0, 0);

    if (periodic_az) {
        if (has_raysB && underflows) {
            range_a.x = range_b.x - int(meta.n_bins_azimuth);
        }
        if (has_raysB && overflows) {
            range_a.y = range_b.y + int(meta.n_bins_azimuth);
        }
        range_a.x = max(range_a.x, range_a.y - int(meta.n_bins_azimuth));
        range_a.y = min(range_a.y, range_a.x + int(meta.n_bins_azimuth));
        range_b = int2(0, 0);
    } else if (has_raysA && has_raysB &&
               range_b.x < range_a.y && range_a.x < range_b.y) {
        range_a = int2(0, int(meta.n_bins_azimuth));
        range_b = int2(0, 0);
    }

    valid = tile_min_el <= tile_max_el;
}

kernel void intersect_tile_lidar_count_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const int* cdf_elevation [[buffer(2)]],
    device const int* cdf_dense_ray_mask [[buffer(3)]],
    device int* tiles_per_gauss [[buffer(4)]],
    constant LidarMeta& meta [[buffer(5)]],
    constant uint& n_elements [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n_elements) {
        return;
    }

    const int radius_x = radii[idx * 2];
    const int radius_y = radii[idx * 2 + 1];
    bool valid;
    int tile_min_el;
    int tile_max_el;
    int2 range_a;
    int2 range_b;
    bool periodic_az;
    compute_lidar_tile_ranges(
        float2(means2d[idx * 2], means2d[idx * 2 + 1]),
        radius_x,
        radius_y,
        meta,
        cdf_elevation,
        cdf_dense_ray_mask,
        valid,
        tile_min_el,
        tile_max_el,
        range_a,
        range_b,
        periodic_az);
    (void)periodic_az;
    if (!valid) {
        tiles_per_gauss[idx] = 0;
        return;
    }

    const int az_span = max(range_a.y - range_a.x, 0) + max(range_b.y - range_b.x, 0);
    tiles_per_gauss[idx] = (tile_max_el - tile_min_el) * az_span;
}

kernel void intersect_tile_lidar_emit_kernel(
    device const float* means2d [[buffer(0)]],
    device const int* radii [[buffer(1)]],
    device const float* depths [[buffer(2)]],
    device const long* image_ids [[buffer(3)]],
    device const int* cdf_elevation [[buffer(4)]],
    device const int* cdf_dense_ray_mask [[buffer(5)]],
    device const long* cum_tiles_per_gauss [[buffer(6)]],
    device long* isect_ids [[buffer(7)]],
    device int* flatten_ids [[buffer(8)]],
    constant LidarMeta& meta [[buffer(9)]],
    constant uint& n_elements [[buffer(10)]],
    constant uint& N [[buffer(11)]],
    constant uint& tile_n_bits [[buffer(12)]],
    constant uint& packed [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n_elements) {
        return;
    }

    const int radius_x = radii[idx * 2];
    const int radius_y = radii[idx * 2 + 1];
    bool valid;
    int tile_min_el;
    int tile_max_el;
    int2 range_a;
    int2 range_b;
    bool periodic_az;
    compute_lidar_tile_ranges(
        float2(means2d[idx * 2], means2d[idx * 2 + 1]),
        radius_x,
        radius_y,
        meta,
        cdf_elevation,
        cdf_dense_ray_mask,
        valid,
        tile_min_el,
        tile_max_el,
        range_a,
        range_b,
        periodic_az);
    if (!valid) {
        return;
    }

    const ulong image_id = packed != 0u ? ulong(image_ids[idx]) : ulong(idx / N);
    const ulong image_id_enc = image_id << (32u + tile_n_bits);
    const ulong depth_bits = ulong(as_type<uint>(depths[idx]));
    ulong cur_idx = idx == 0u ? 0ul : ulong(cum_tiles_per_gauss[idx - 1u]);

    const int2 ranges[2] = {range_a, range_b};
    for (int el = tile_min_el; el < tile_max_el; ++el) {
        for (uint r = 0; r < 2u; ++r) {
            for (int az = ranges[r].x; az < ranges[r].y; ++az) {
                const int actual_az = periodic_az
                    ? ((az + int(meta.n_bins_azimuth)) % int(meta.n_bins_azimuth))
                    : az;
                const ulong tile_id = ulong(el) * ulong(meta.n_bins_azimuth) + ulong(actual_az);
                isect_ids[cur_idx] = long(image_id_enc | (tile_id << 32u) | depth_bits);
                flatten_ids[cur_idx] = int(idx);
                ++cur_idx;
            }
        }
    }
}
