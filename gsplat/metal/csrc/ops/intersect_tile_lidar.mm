// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "intersect_tile_lidar.h"
#include "sort_int64.h"

namespace gsplat::metal {

namespace {

struct LidarMeta {
    uint32_t n_bins_azimuth;
    uint32_t n_bins_elevation;
    uint32_t cdf_resolution_azimuth;
    uint32_t cdf_resolution_elevation;
    float angle_to_pixel_scaling_factor;
    float fov_horiz_start;
    float fov_horiz_span;
    float fov_vert_start;
    float fov_vert_span;
    float fov_eps;
    uint32_t spinning_direction;
};

struct LidarIntersectConfig {
    bool packed;
    uint32_t I;
    uint32_t N;
    uint32_t n_elements;
    uint32_t tile_n_bits;
    LidarMeta meta;
};

uint32_t tile_n_bits_from_grid(uint32_t n_bins_azimuth, uint32_t n_bins_elevation) {
    const uint64_t n_tiles = static_cast<uint64_t>(n_bins_azimuth) * static_cast<uint64_t>(n_bins_elevation);
    TORCH_CHECK(
        n_tiles > 0 && n_tiles < (uint64_t{1} << 31),
        "n_bins_azimuth * n_bins_elevation must be in [1, 2^31)");
    return 32u - static_cast<uint32_t>(__builtin_clz(static_cast<uint32_t>(n_tiles)));
}

LidarIntersectConfig validate_common(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    bool packed,
    int64_t n_bins_azimuth,
    int64_t n_bins_elevation,
    int64_t cdf_resolution_azimuth,
    int64_t cdf_resolution_elevation,
    double angle_to_pixel_scaling_factor,
    double fov_horiz_start,
    double fov_horiz_span,
    double fov_vert_start,
    double fov_vert_span,
    double fov_eps,
    int64_t spinning_direction,
    const at::Tensor& cdf_elevation,
    const at::Tensor& cdf_dense_ray_mask,
    const at::Tensor& tiles_pack_info,
    const at::Tensor& tiles_to_elements_map
) {
    check_mps_float32(means2d, "means2d");
    check_mps_int32(radii, "radii");
    check_mps_float32(depths, "depths");
    check_mps_int32(cdf_elevation, "cdf_elevation");
    check_mps_int32(cdf_dense_ray_mask, "cdf_dense_ray_mask");
    check_mps_int32(tiles_pack_info, "tiles_pack_info");
    check_mps_int32(tiles_to_elements_map, "tiles_to_elements_map");

    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dimension must be 2");
    TORCH_CHECK(radii.sizes().equals(means2d.sizes()), "radii shape must match means2d");
    TORCH_CHECK(depths.dim() == means2d.dim() - 1, "depths must have shape means2d.shape[:-1]");
    TORCH_CHECK(
        depths.sizes().equals(means2d.sizes().slice(0, means2d.dim() - 1)),
        "depths shape must match means2d.shape[:-1]");

    TORCH_CHECK(n_bins_azimuth > 0, "n_bins_azimuth must be positive");
    TORCH_CHECK(n_bins_elevation > 0, "n_bins_elevation must be positive");
    TORCH_CHECK(cdf_resolution_azimuth > 0, "cdf_resolution_azimuth must be positive");
    TORCH_CHECK(cdf_resolution_elevation > 0, "cdf_resolution_elevation must be positive");
    TORCH_CHECK(angle_to_pixel_scaling_factor > 0.0, "angle_to_pixel_scaling_factor must be positive");
    TORCH_CHECK(fov_horiz_span > 0.0, "fov_horiz_span must be positive");
    TORCH_CHECK(fov_vert_span > 0.0, "fov_vert_span must be positive");
    TORCH_CHECK(fov_eps >= 0.0, "fov_eps must be non-negative");
    TORCH_CHECK(spinning_direction == 0 || spinning_direction == 1, "spinning_direction must be 0 or 1");

    TORCH_CHECK(cdf_elevation.dim() == 1, "cdf_elevation must be 1D");
    TORCH_CHECK(cdf_dense_ray_mask.dim() == 2, "cdf_dense_ray_mask must be 2D");
    TORCH_CHECK(tiles_pack_info.dim() == 2 && tiles_pack_info.size(1) == 2,
        "tiles_pack_info must have shape (n_tiles, 2)");
    TORCH_CHECK(tiles_to_elements_map.dim() == 2 && tiles_to_elements_map.size(1) == 2,
        "tiles_to_elements_map must have shape (n_rays, 2)");
    TORCH_CHECK(cdf_elevation.size(0) == cdf_resolution_elevation + 1,
        "cdf_elevation length must equal cdf_resolution_elevation + 1");
    TORCH_CHECK(
        cdf_dense_ray_mask.size(0) == cdf_resolution_elevation + 1 &&
            cdf_dense_ray_mask.size(1) == cdf_resolution_azimuth + 1,
        "cdf_dense_ray_mask shape must be (cdf_resolution_elevation + 1, cdf_resolution_azimuth + 1)");
    TORCH_CHECK(tiles_pack_info.size(0) == n_bins_azimuth * n_bins_elevation,
        "tiles_pack_info first dimension must equal n_bins_azimuth * n_bins_elevation");

    LidarIntersectConfig cfg{};
    cfg.packed = packed;
    cfg.tile_n_bits = tile_n_bits_from_grid(
        static_cast<uint32_t>(n_bins_azimuth),
        static_cast<uint32_t>(n_bins_elevation));
    cfg.meta = LidarMeta{
        static_cast<uint32_t>(n_bins_azimuth),
        static_cast<uint32_t>(n_bins_elevation),
        static_cast<uint32_t>(cdf_resolution_azimuth),
        static_cast<uint32_t>(cdf_resolution_elevation),
        static_cast<float>(angle_to_pixel_scaling_factor),
        static_cast<float>(fov_horiz_start),
        static_cast<float>(fov_horiz_span),
        static_cast<float>(fov_vert_start),
        static_cast<float>(fov_vert_span),
        static_cast<float>(fov_eps),
        static_cast<uint32_t>(spinning_direction),
    };

    if (packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz, "packed depths must have shape (nnz,)");
        TORCH_CHECK(image_ids.has_value(), "image_ids is required when packed=True");
        TORCH_CHECK(gaussian_ids.has_value(), "gaussian_ids is required when packed=True");
        check_mps_int64(*image_ids, "image_ids");
        check_mps_int64(*gaussian_ids, "gaussian_ids");
        TORCH_CHECK(image_ids->dim() == 1 && image_ids->size(0) == nnz, "packed image_ids must have shape (nnz,)");
        TORCH_CHECK(gaussian_ids->dim() == 1 && gaussian_ids->size(0) == nnz, "packed gaussian_ids must have shape (nnz,)");
        TORCH_CHECK(I >= 0, "I must be non-negative");
        cfg.I = static_cast<uint32_t>(I);
        cfg.N = 0;
        cfg.n_elements = product_i64_to_u32({nnz}, "means2d");
    } else {
        TORCH_CHECK(!image_ids.has_value(), "image_ids must be omitted when packed=False");
        TORCH_CHECK(!gaussian_ids.has_value(), "gaussian_ids must be omitted when packed=False");
        cfg.N = static_cast<uint32_t>(means2d.size(-2));
        const auto image_dims = means2d.sizes().slice(0, means2d.dim() - 2);
        const uint32_t inferred_I = product_i64_to_u32(image_dims, "means2d");
        TORCH_CHECK(static_cast<uint64_t>(I) == static_cast<uint64_t>(inferred_I),
            "I must match the product of means2d image dimensions");
        cfg.I = inferred_I;
        cfg.n_elements = product_i64_to_u32(depths.sizes(), "depths");
    }

    return cfg;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_lidar_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    bool sort,
    bool segmented,
    bool packed,
    int64_t n_bins_azimuth,
    int64_t n_bins_elevation,
    int64_t cdf_resolution_azimuth,
    int64_t cdf_resolution_elevation,
    double angle_to_pixel_scaling_factor,
    double fov_horiz_start,
    double fov_horiz_span,
    double fov_vert_start,
    double fov_vert_span,
    double fov_eps,
    int64_t spinning_direction,
    const at::Tensor& cdf_elevation,
    const at::Tensor& cdf_dense_ray_mask,
    const at::Tensor& tiles_pack_info,
    const at::Tensor& tiles_to_elements_map
) {
    const auto cfg = validate_common(
        means2d,
        radii,
        depths,
        image_ids,
        gaussian_ids,
        I,
        packed,
        n_bins_azimuth,
        n_bins_elevation,
        cdf_resolution_azimuth,
        cdf_resolution_elevation,
        angle_to_pixel_scaling_factor,
        fov_horiz_start,
        fov_horiz_span,
        fov_vert_start,
        fov_vert_span,
        fov_eps,
        spinning_direction,
        cdf_elevation,
        cdf_dense_ray_mask,
        tiles_pack_info,
        tiles_to_elements_map);

    at::Tensor tiles_per_gauss = at::empty_like(depths, depths.options().dtype(at::kInt));
    if (cfg.n_elements == 0u) {
        at::Tensor empty_i64 = at::empty({0}, depths.options().dtype(at::kLong));
        at::Tensor empty_i32 = at::empty({0}, depths.options().dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, empty_i64, empty_i32);
    }

    auto& ctx = MetalContext::instance();
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");
    id<MTLComputePipelineState> count_pso = ctx.pipeline("intersect_tile_lidar_count_kernel");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:count_pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:1];
            [enc setBuffer:to_mtl_buffer(cdf_elevation) offset:byte_offset(cdf_elevation) atIndex:2];
            [enc setBuffer:to_mtl_buffer(cdf_dense_ray_mask) offset:byte_offset(cdf_dense_ray_mask) atIndex:3];
            [enc setBuffer:to_mtl_buffer(tiles_per_gauss) offset:byte_offset(tiles_per_gauss) atIndex:4];
            [enc setBytes:&cfg.meta length:sizeof(cfg.meta) atIndex:5];
            [enc setBytes:&cfg.n_elements length:sizeof(cfg.n_elements) atIndex:6];

            const uint32_t tg = static_cast<uint32_t>(count_pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, cfg.n_elements);
            [enc dispatchThreads:MTLSizeMake(cfg.n_elements, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor flat_counts = tiles_per_gauss.reshape({-1}).to(at::kLong);
    if (flat_counts.numel() == 0) {
        at::Tensor empty_i64 = at::empty({0}, depths.options().dtype(at::kLong));
        at::Tensor empty_i32 = at::empty({0}, depths.options().dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, empty_i64, empty_i32);
    }
    at::Tensor cum_tiles_per_gauss = at::cumsum(flat_counts, 0);
    const int64_t n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    if (n_isects == 0) {
        at::Tensor empty_i64 = at::empty({0}, depths.options().dtype(at::kLong));
        at::Tensor empty_i32 = at::empty({0}, depths.options().dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, empty_i64, empty_i32);
    }

    at::Tensor isect_ids = at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, depths.options().dtype(at::kInt));
    id<MTLComputePipelineState> emit_pso = ctx.pipeline("intersect_tile_lidar_emit_kernel");
    const uint32_t packed_u32 = cfg.packed ? 1u : 0u;
    const at::Tensor image_ids_arg = cfg.packed ? *image_ids : at::Tensor{};

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:emit_pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:1];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:2];
            set_optional_tensor_buffer(enc, image_ids_arg, 3);
            [enc setBuffer:to_mtl_buffer(cdf_elevation) offset:byte_offset(cdf_elevation) atIndex:4];
            [enc setBuffer:to_mtl_buffer(cdf_dense_ray_mask) offset:byte_offset(cdf_dense_ray_mask) atIndex:5];
            [enc setBuffer:to_mtl_buffer(cum_tiles_per_gauss) offset:byte_offset(cum_tiles_per_gauss) atIndex:6];
            [enc setBuffer:to_mtl_buffer(isect_ids) offset:byte_offset(isect_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:8];
            [enc setBytes:&cfg.meta length:sizeof(cfg.meta) atIndex:9];
            [enc setBytes:&cfg.n_elements length:sizeof(cfg.n_elements) atIndex:10];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:11];
            [enc setBytes:&cfg.tile_n_bits length:sizeof(cfg.tile_n_bits) atIndex:12];
            [enc setBytes:&packed_u32 length:sizeof(packed_u32) atIndex:13];

            const uint32_t tg = static_cast<uint32_t>(emit_pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, cfg.n_elements);
            [enc dispatchThreads:MTLSizeMake(cfg.n_elements, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    if (sort) {
        // As in standard intersect_tile, a global stable sort by the full key
        // already yields the correct segmented ordering because image_id lives
        // in the most-significant bits of isect_ids.
        (void)segmented;
        auto sorted = radix_sort(isect_ids, flatten_ids, /*stable=*/true);
        isect_ids = sorted.first;
        flatten_ids = sorted.second;
    } else {
        (void)segmented;
    }

    return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids.to(at::kInt));
}

}  // namespace gsplat::metal
