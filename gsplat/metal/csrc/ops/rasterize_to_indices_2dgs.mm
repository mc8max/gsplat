// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "rasterize_to_indices_common.h"
#include "rasterize_to_indices_2dgs.h"

namespace gsplat::metal {

namespace {

RasterizeIndicesConfig validate_common(
    int64_t range_start,
    int64_t range_end,
    const at::Tensor& transmittances,
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& opacities,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
) {
    check_mps_float32(transmittances, "transmittances");
    check_mps_float32(means2d, "means2d");
    check_mps_float32(ray_transforms, "ray_transforms");
    check_mps_float32(opacities, "opacities");
    check_mps_int32(tile_offsets, "tile_offsets");
    check_mps_int32(flatten_ids, "flatten_ids");

    TORCH_CHECK(range_start >= 0, "range_start must be non-negative");
    TORCH_CHECK(range_end >= range_start, "range_end must be >= range_start");
    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(
        static_cast<uint64_t>(tile_size) * static_cast<uint64_t>(tile_size) <= 256u,
        "Metal rasterize_to_indices_2dgs currently requires tile_size^2 <= 256");
    TORCH_CHECK(means2d.dim() >= 3, "means2d must have shape [..., N, 2]");
    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dimension must be 2");

    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    TORCH_CHECK(
        means2d.sizes().slice(0, means2d.dim() - 2) == image_dims,
        "means2d image dims must match tile_offsets");

    const int64_t N = means2d.size(-2);
    auto expected_rt = image_dims.vec();
    expected_rt.push_back(N);
    expected_rt.push_back(3);
    expected_rt.push_back(3);
    TORCH_CHECK(
        ray_transforms.sizes().vec() == expected_rt,
        "ray_transforms must have shape [..., N, 3, 3]");

    auto expected_opacities = image_dims.vec();
    expected_opacities.push_back(N);
    TORCH_CHECK(opacities.sizes().vec() == expected_opacities, "opacities must have shape [..., N]");

    auto expected_trans = image_dims.vec();
    expected_trans.push_back(image_height);
    expected_trans.push_back(image_width);
    TORCH_CHECK(transmittances.sizes().vec() == expected_trans, "transmittances shape mismatch");
    TORCH_CHECK(
        tile_offsets.dim() >= 2,
        "tile_offsets must have shape [..., tile_height, tile_width]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");

    const int64_t tile_height = tile_offsets.size(-2);
    const int64_t tile_width = tile_offsets.size(-1);
    TORCH_CHECK(tile_height >= 0, "tile_height must be non-negative");
    TORCH_CHECK(tile_width >= 0, "tile_width must be non-negative");
    TORCH_CHECK(tile_height * tile_size >= image_height, "tile grid must cover image_height");
    TORCH_CHECK(tile_width * tile_size >= image_width, "tile grid must cover image_width");

    RasterizeIndicesConfig cfg{};
    cfg.I = product_i64_to_u32(image_dims, "tile_offsets");
    cfg.N = static_cast<uint32_t>(N);
    cfg.tile_size = static_cast<uint32_t>(tile_size);
    cfg.tile_width = static_cast<uint32_t>(tile_width);
    cfg.tile_height = static_cast<uint32_t>(tile_height);
    cfg.n_tiles = cfg.tile_width * cfg.tile_height;
    cfg.total_tiles = cfg.I * cfg.n_tiles;
    cfg.n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    return cfg;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_2dgs_op(
    int64_t range_start,
    int64_t range_end,
    const at::Tensor& transmittances,
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& opacities,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
) {
    const auto cfg = validate_common(
        range_start,
        range_end,
        transmittances,
        means2d,
        ray_transforms,
        opacities,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids
    );

    at::Tensor gaussian_ids = at::empty({0}, means2d.options().dtype(at::kLong));
    at::Tensor pixel_ids = at::empty({0}, means2d.options().dtype(at::kLong));
    if (image_width == 0 || image_height == 0 || cfg.n_isects == 0u) {
        return std::make_tuple(gaussian_ids, pixel_ids);
    }

    at::Tensor chunk_cnts = at::zeros(
        {static_cast<int64_t>(cfg.I) * image_height * image_width},
        means2d.options().dtype(at::kInt)
    );

    auto& ctx = MetalContext::instance();
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t range_start_u32 = static_cast<uint32_t>(range_start);
    const uint32_t range_end_u32 = static_cast<uint32_t>(range_end);
    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);

    id<MTLComputePipelineState> count_pso = ctx.pipeline("rasterize_to_indices_2dgs_count_kernel");
    TORCH_CHECK(
        cfg.tile_size * cfg.tile_size <= static_cast<uint32_t>(count_pso.maxTotalThreadsPerThreadgroup),
        "tile_size^2 exceeds the pipeline threadgroup limit");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");
            [enc setComputePipelineState:count_pso];
            [enc setBuffer:to_mtl_buffer(transmittances) offset:byte_offset(transmittances) atIndex:0];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:1];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:4];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:5];
            [enc setBuffer:to_mtl_buffer(chunk_cnts) offset:byte_offset(chunk_cnts) atIndex:6];
            [enc setBytes:&range_start_u32 length:sizeof(range_start_u32) atIndex:7];
            [enc setBytes:&range_end_u32 length:sizeof(range_end_u32) atIndex:8];
            [enc setBytes:&cfg.I length:sizeof(cfg.I) atIndex:9];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:10];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:11];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:12];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:13];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:14];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:15];
            [enc setBytes:&cfg.n_tiles length:sizeof(cfg.n_tiles) atIndex:16];
            [enc setBytes:&cfg.total_tiles length:sizeof(cfg.total_tiles) atIndex:17];
            [enc setBytes:&cfg.n_isects length:sizeof(cfg.n_isects) atIndex:18];
            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT_AND_WAIT: CPU reads cumsum[-1] immediately to determine n_elems
    // before allocating gaussian_ids and pixel_ids.
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    at::Tensor cumsum = at::cumsum(chunk_cnts, 0, at::kInt);
    const int32_t n_elems = cumsum.numel() > 0 ? cumsum[-1].item<int32_t>() : 0;
    if (n_elems == 0) {
        return std::make_tuple(gaussian_ids, pixel_ids);
    }
    at::Tensor chunk_starts = cumsum - chunk_cnts;

    gaussian_ids = at::empty({n_elems}, means2d.options().dtype(at::kLong));
    pixel_ids = at::empty({n_elems}, means2d.options().dtype(at::kLong));

    id<MTLComputePipelineState> emit_pso = ctx.pipeline("rasterize_to_indices_2dgs_emit_kernel");
    TORCH_CHECK(
        cfg.tile_size * cfg.tile_size <= static_cast<uint32_t>(emit_pso.maxTotalThreadsPerThreadgroup),
        "tile_size^2 exceeds the pipeline threadgroup limit");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");
            [enc setComputePipelineState:emit_pso];
            [enc setBuffer:to_mtl_buffer(transmittances) offset:byte_offset(transmittances) atIndex:0];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:1];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:4];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:5];
            [enc setBuffer:to_mtl_buffer(chunk_starts) offset:byte_offset(chunk_starts) atIndex:6];
            [enc setBuffer:to_mtl_buffer(gaussian_ids) offset:byte_offset(gaussian_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(pixel_ids) offset:byte_offset(pixel_ids) atIndex:8];
            [enc setBytes:&range_start_u32 length:sizeof(range_start_u32) atIndex:9];
            [enc setBytes:&range_end_u32 length:sizeof(range_end_u32) atIndex:10];
            [enc setBytes:&cfg.I length:sizeof(cfg.I) atIndex:11];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:12];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:13];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:14];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:15];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:16];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:17];
            [enc setBytes:&cfg.n_tiles length:sizeof(cfg.n_tiles) atIndex:18];
            [enc setBytes:&cfg.total_tiles length:sizeof(cfg.total_tiles) atIndex:19];
            [enc setBytes:&cfg.n_isects length:sizeof(cfg.n_isects) atIndex:20];
            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient — no immediate CPU read follows; downstream MPS ops
    // on the same stream are automatically serialised.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(gaussian_ids, pixel_ids);
}

}  // namespace gsplat::metal
