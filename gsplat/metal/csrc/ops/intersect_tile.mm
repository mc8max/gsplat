// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <limits>
#include <torch/extension.h>

#include "sort_int64.h"

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "intersect_tile.h"

namespace gsplat::metal {

namespace {

struct IntersectTileConfig {
    bool packed;
    bool use_accutile;
    uint32_t I;
    uint32_t N;
    uint32_t n_elements;
    uint32_t tile_size;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_n_bits;
};

// Single definition of the tile_n_bits formula so there is exactly one place
// to update if the key layout changes.
uint32_t tile_n_bits_from_grid(int64_t tile_width, int64_t tile_height) {
    const uint64_t n_tiles = static_cast<uint64_t>(tile_width) * static_cast<uint64_t>(tile_height);
    TORCH_CHECK(
        n_tiles > 0 && n_tiles < (uint64_t{1} << 31),
        "tile_width * tile_height must be in [1, 2^31)");
    return 32u - static_cast<uint32_t>(__builtin_clz(static_cast<uint32_t>(n_tiles)));
}

IntersectTileConfig validate_common(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool packed,
    bool segmented
) {
    check_mps_float32(means2d, "means2d");
    check_mps_int32(radii, "radii");
    check_mps_float32(depths, "depths");

    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(tile_width > 0, "tile_width must be positive");
    TORCH_CHECK(tile_height > 0, "tile_height must be positive");
    // packed is passed explicitly by the caller; do not infer from tensor rank.
    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dimension must be 2");
    TORCH_CHECK(radii.sizes().equals(means2d.sizes()), "radii shape must match means2d");

    IntersectTileConfig cfg{};
    cfg.packed = packed;
    cfg.use_accutile = conics.has_value() && opacities.has_value();
    cfg.tile_size = static_cast<uint32_t>(tile_size);
    cfg.tile_width = static_cast<uint32_t>(tile_width);
    cfg.tile_height = static_cast<uint32_t>(tile_height);

    cfg.tile_n_bits = tile_n_bits_from_grid(tile_width, tile_height);

    if (packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz, "packed depths must have shape (nnz,)");
        TORCH_CHECK(image_ids.has_value(), "image_ids is required when packed=True");
        TORCH_CHECK(gaussian_ids.has_value(), "gaussian_ids is required when packed=True");
        TORCH_CHECK(I >= 0, "n_images must be non-negative");
        check_mps_int64(*image_ids, "image_ids");
        check_mps_int64(*gaussian_ids, "gaussian_ids");
        TORCH_CHECK(image_ids->dim() == 1 && image_ids->size(0) == nnz, "packed image_ids must have shape (nnz,)");
        TORCH_CHECK(
            gaussian_ids->dim() == 1 && gaussian_ids->size(0) == nnz,
            "packed gaussian_ids must have shape (nnz,)");
        cfg.I = static_cast<uint32_t>(I);
        cfg.N = 0;
        cfg.n_elements = product_i64_to_u32({nnz}, "means2d");
        if (conics.has_value()) {
            check_mps_float32(*conics, "conics");
            TORCH_CHECK(conics->dim() == 2 && conics->size(0) == nnz && conics->size(1) == 3,
                "packed conics must have shape (nnz, 3)");
        }
        if (opacities.has_value()) {
            check_mps_float32(*opacities, "opacities");
            TORCH_CHECK(opacities->dim() == 1 && opacities->size(0) == nnz,
                "packed opacities must have shape (nnz,)");
        }
    } else {
        TORCH_CHECK(depths.dim() == means2d.dim() - 1, "depths must have shape means2d.shape[:-1]");
        TORCH_CHECK(
            depths.sizes().equals(means2d.sizes().slice(0, means2d.dim() - 1)),
            "depths shape must match means2d.shape[:-1]");
        TORCH_CHECK(!image_ids.has_value(), "image_ids must be omitted when packed=False");
        TORCH_CHECK(!gaussian_ids.has_value(), "gaussian_ids must be omitted when packed=False");
        TORCH_CHECK(means2d.dim() >= 2, "unpacked means2d must have shape [..., N, 2]");
        cfg.N = static_cast<uint32_t>(means2d.size(-2));
        const auto image_dims = means2d.sizes().slice(0, means2d.dim() - 2);
        const uint32_t inferred_I = product_i64_to_u32(image_dims, "means2d");
        TORCH_CHECK(
            static_cast<uint64_t>(I) == static_cast<uint64_t>(inferred_I),
            "I must match the product of means2d image dimensions");
        cfg.I = inferred_I;
        cfg.n_elements = product_i64_to_u32(depths.sizes(), "depths");
        if (conics.has_value()) {
            check_mps_float32(*conics, "conics");
            auto expected = means2d.sizes().vec();  // [..., N, 2]
            expected.back() = 3;                    // [..., N, 3]
            TORCH_CHECK(conics->sizes().vec() == expected, "unpacked conics must have shape [..., N, 3]");
        }
        if (opacities.has_value()) {
            check_mps_float32(*opacities, "opacities");
            TORCH_CHECK(opacities->sizes().equals(depths.sizes()), "unpacked opacities must match depths shape");
        }
    }

    return cfg;
}


}  // namespace

at::Tensor intersect_tile_count_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool packed,
    bool segmented
) {
    const auto cfg = validate_common(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        packed,
        segmented);

    at::Tensor tiles_per_gauss = at::empty_like(depths, depths.options().dtype(at::kInt));
    if (cfg.n_elements == 0u) {
        return tiles_per_gauss;
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline(
        cfg.use_accutile ? "intersect_tile_count_accutile_kernel"
                         : "intersect_tile_count_aabb_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:1];
            set_optional_tensor_buffer(enc, conics.has_value() ? *conics : at::Tensor{}, 2);
            set_optional_tensor_buffer(enc, opacities.has_value() ? *opacities : at::Tensor{}, 3);
            [enc setBuffer:to_mtl_buffer(tiles_per_gauss) offset:byte_offset(tiles_per_gauss) atIndex:4];
            [enc setBytes:&cfg.n_elements length:sizeof(cfg.n_elements) atIndex:5];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:6];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:7];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:8];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, cfg.n_elements);
            [enc dispatchThreads:MTLSizeMake(cfg.n_elements, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);
    return tiles_per_gauss;
}

std::tuple<at::Tensor, at::Tensor> intersect_tile_emit_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    const at::Tensor& cum_tiles_per_gauss,
    bool packed,
    bool segmented
) {
    const auto cfg = validate_common(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        packed,
        segmented);
    check_mps_int32(cum_tiles_per_gauss, "cum_tiles_per_gauss");
    TORCH_CHECK(
        cum_tiles_per_gauss.sizes().equals({static_cast<int64_t>(cfg.n_elements)}),
        "cum_tiles_per_gauss must have shape (n_elements,)");

    const int64_t n_isects = cfg.n_elements == 0u ? 0 : int64_t(cum_tiles_per_gauss[-1].item<int>());
    at::Tensor isect_ids = at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, depths.options().dtype(at::kInt));
    if (n_isects == 0) {
        return std::make_tuple(isect_ids, flatten_ids);
    }

    const at::Tensor conics_arg = conics.has_value() ? *conics : at::Tensor{};
    const at::Tensor opacities_arg = opacities.has_value() ? *opacities : at::Tensor{};
    const at::Tensor image_ids_arg = cfg.packed ? *image_ids : at::Tensor{};
    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline(
        cfg.use_accutile ? "intersect_tile_emit_accutile_kernel"
                         : "intersect_tile_emit_aabb_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");
    const uint32_t packed_u32 = cfg.packed ? 1u : 0u;

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:1];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:2];
            set_optional_tensor_buffer(enc, conics_arg, 3);
            set_optional_tensor_buffer(enc, opacities_arg, 4);
            set_optional_tensor_buffer(enc, image_ids_arg, 5);
            [enc setBuffer:to_mtl_buffer(cum_tiles_per_gauss) offset:byte_offset(cum_tiles_per_gauss) atIndex:6];
            [enc setBuffer:to_mtl_buffer(isect_ids) offset:byte_offset(isect_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:8];
            [enc setBytes:&cfg.n_elements length:sizeof(cfg.n_elements) atIndex:9];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:10];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:11];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:12];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:13];
            [enc setBytes:&cfg.tile_n_bits length:sizeof(cfg.tile_n_bits) atIndex:14];
            [enc setBytes:&packed_u32 length:sizeof(packed_u32) atIndex:15];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, cfg.n_elements);
            [enc dispatchThreads:MTLSizeMake(cfg.n_elements, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    // COMMIT_AND_WAIT ensures the emit kernel has finished writing isect_ids
    // before the tensor is returned to the caller. COMMIT alone is insufficient
    // here because the sort=False path returns the tensor directly to Python
    // without any subsequent MPS op that would force completion.
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    return std::make_tuple(isect_ids, flatten_ids);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_op(
    const at::Tensor& means2d,
    const at::Tensor& radii,
    const at::Tensor& depths,
    const c10::optional<at::Tensor>& conics,
    const c10::optional<at::Tensor>& opacities,
    const c10::optional<at::Tensor>& image_ids,
    const c10::optional<at::Tensor>& gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool packed,
    bool segmented
) {
    at::Tensor tiles_per_gauss = intersect_tile_count_op(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        packed,
        segmented);

    at::Tensor flat_counts = tiles_per_gauss.reshape({-1});
    if (flat_counts.numel() == 0) {
        at::Tensor empty_i64 = at::empty({0}, depths.options().dtype(at::kLong));
        at::Tensor empty_i32 = at::empty({0}, depths.options().dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, empty_i64, empty_i32);
    }

    at::Tensor cum_tiles_per_gauss = at::cumsum(flat_counts, 0, at::kInt);
    const int64_t n_isects = int64_t(cum_tiles_per_gauss[-1].item<int>());
    if (n_isects == 0) {
        at::Tensor empty_i64 = at::empty({0}, depths.options().dtype(at::kLong));
        at::Tensor empty_i32 = at::empty({0}, depths.options().dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, empty_i64, empty_i32);
    }

    auto emit_out = intersect_tile_emit_op(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        cum_tiles_per_gauss,
        packed,
        segmented);
    at::Tensor isect_ids = std::get<0>(emit_out);
    at::Tensor flatten_ids = std::get<1>(emit_out);

    if (sort) {
        // 4-pass 16-bit LSD stable sort entirely on MPS — no PCIe transfer.
        // Global sort by the full 64-bit key gives correct per-image ordering
        // because image_id occupies the most significant bits, so the global
        // result is identical to a segmented (per-image) sort.
        auto sorted = radix_sort(isect_ids, flatten_ids, /*stable=*/true);
        isect_ids   = sorted.first;
        flatten_ids = sorted.second;
    }

    return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids.to(at::kInt));
}

}  // namespace gsplat::metal
