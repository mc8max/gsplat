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
#include "rasterize_to_pixels_3dgs.h"

namespace gsplat::metal {

namespace {

struct RasterizeConfig {
    bool packed;
    uint32_t I;
    uint32_t N;
    uint32_t channels;
    uint32_t tile_size;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t n_tiles;
    uint32_t total_tiles;
    uint32_t n_isects;
};

RasterizeConfig validate_common(
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
) {
    check_mps_float32(means2d, "means2d");
    check_mps_float32(conics, "conics");
    check_mps_float32(colors, "colors");
    check_mps_float32(opacities, "opacities");
    check_mps_int32(tile_offsets, "tile_offsets");
    check_mps_int32(flatten_ids, "flatten_ids");

    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(
        static_cast<uint64_t>(tile_size) * static_cast<uint64_t>(tile_size) <= 256u,
        "Metal rasterize_to_pixels currently requires tile_size^2 <= 256");
    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dimension must be 2");
    TORCH_CHECK(conics.size(-1) == 3, "conics last dimension must be 3");
    TORCH_CHECK(colors.dim() >= 2, "colors must have at least 2 dimensions");
    TORCH_CHECK(colors.size(-1) > 0, "colors last dimension must be positive");
    TORCH_CHECK(tile_offsets.dim() >= 2, "tile_offsets must have shape [..., tile_height, tile_width]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");

    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    const uint32_t I = product_i64_to_u32(image_dims, "tile_offsets");
    const int64_t tile_height = tile_offsets.size(-2);
    const int64_t tile_width = tile_offsets.size(-1);

    TORCH_CHECK(tile_height >= 0, "tile_height must be non-negative");
    TORCH_CHECK(tile_width >= 0, "tile_width must be non-negative");
    TORCH_CHECK(
        tile_height * tile_size >= image_height,
        "tile_height * tile_size must cover image_height");
    TORCH_CHECK(
        tile_width * tile_size >= image_width,
        "tile_width * tile_size must cover image_width");

    if (backgrounds.has_value()) {
        check_mps_float32(*backgrounds, "backgrounds");
        auto expected = image_dims.vec();
        expected.push_back(colors.size(-1));
        TORCH_CHECK(backgrounds->sizes().vec() == expected, "backgrounds must have shape [..., channels]");
    }
    if (masks.has_value()) {
        check_mps_bool(*masks, "masks");
        TORCH_CHECK(masks->sizes().equals(tile_offsets.sizes()), "masks shape must match tile_offsets");
    }

    RasterizeConfig cfg{};
    cfg.packed = means2d.dim() == 2;
    cfg.I = I;
    cfg.channels = static_cast<uint32_t>(colors.size(-1));
    cfg.tile_size = static_cast<uint32_t>(tile_size);
    cfg.tile_width = static_cast<uint32_t>(tile_width);
    cfg.tile_height = static_cast<uint32_t>(tile_height);
    cfg.n_tiles = static_cast<uint32_t>(tile_width * tile_height);
    cfg.total_tiles = cfg.I * cfg.n_tiles;
    cfg.n_isects = static_cast<uint32_t>(flatten_ids.size(0));

    if (cfg.packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(means2d.sizes().equals({nnz, 2}), "packed means2d must have shape (nnz, 2)");
        TORCH_CHECK(conics.sizes().equals({nnz, 3}), "packed conics must have shape (nnz, 3)");
        TORCH_CHECK(colors.dim() == 2 && colors.size(0) == nnz, "packed colors must have shape (nnz, channels)");
        TORCH_CHECK(opacities.sizes().equals({nnz}), "packed opacities must have shape (nnz,)");
        cfg.N = 0;
    } else {
        TORCH_CHECK(means2d.dim() >= 3, "unpacked means2d must have shape [..., N, 2]");
        const int64_t N = means2d.size(-2);
        const auto means_image_dims = means2d.sizes().slice(0, means2d.dim() - 2);
        TORCH_CHECK(means_image_dims.vec() == image_dims.vec(), "means2d image dims must match tile_offsets");
        auto expected_conics = means2d.sizes().vec();
        expected_conics.back() = 3;
        TORCH_CHECK(conics.sizes().vec() == expected_conics, "conics must have shape [..., N, 3]");
        auto expected_colors = means_image_dims.vec();
        expected_colors.push_back(N);
        expected_colors.push_back(colors.size(-1));
        TORCH_CHECK(colors.sizes().vec() == expected_colors, "colors must have shape [..., N, channels]");
        auto expected_opacities = means_image_dims.vec();
        expected_opacities.push_back(N);
        TORCH_CHECK(opacities.sizes().vec() == expected_opacities, "opacities must have shape [..., N]");
        cfg.N = static_cast<uint32_t>(N);
    }

    return cfg;
}

void validate_backward_inputs(
    const RasterizeConfig& cfg,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width
) {
    check_mps_float32(render_alphas, "render_alphas");
    check_mps_int32(last_ids, "last_ids");
    check_mps_float32(v_render_colors, "v_render_colors");
    check_mps_float32(v_render_alphas, "v_render_alphas");

    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    auto expected_alphas = image_dims.vec();
    expected_alphas.push_back(image_height);
    expected_alphas.push_back(image_width);
    expected_alphas.push_back(1);
    auto expected_last = image_dims.vec();
    expected_last.push_back(image_height);
    expected_last.push_back(image_width);
    auto expected_colors = image_dims.vec();
    expected_colors.push_back(image_height);
    expected_colors.push_back(image_width);
    expected_colors.push_back(cfg.channels);

    TORCH_CHECK(render_alphas.sizes().vec() == expected_alphas, "render_alphas shape mismatch");
    TORCH_CHECK(v_render_alphas.sizes().vec() == expected_alphas, "v_render_alphas shape mismatch");
    TORCH_CHECK(last_ids.sizes().vec() == expected_last, "last_ids shape mismatch");
    TORCH_CHECK(v_render_colors.sizes().vec() == expected_colors, "v_render_colors shape mismatch");
}

at::DimVector make_render_colors_shape(
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width,
    int64_t channels
) {
    at::DimVector dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    dims.append({image_height, image_width, channels});
    return dims;
}

at::DimVector make_render_alphas_shape(
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width
) {
    at::DimVector dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    dims.append({image_height, image_width, 1});
    return dims;
}

at::DimVector make_last_ids_shape(
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width
) {
    at::DimVector dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    dims.append({image_height, image_width});
    return dims;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd_op(
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
) {
    const auto cfg = validate_common(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids);

    at::Tensor render_colors = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, colors.size(-1)),
        colors.options());
    at::Tensor render_alphas = at::empty(
        make_render_alphas_shape(tile_offsets, image_height, image_width),
        means2d.options().dtype(at::kFloat));
    at::Tensor last_ids = at::empty(
        make_last_ids_shape(tile_offsets, image_height, image_width),
        tile_offsets.options().dtype(at::kInt));

    if (image_width == 0 || image_height == 0) {
        return std::make_tuple(render_colors, render_alphas, last_ids);
    }

    if (cfg.n_isects == 0u) {
        render_alphas.zero_();
        last_ids.zero_();
        render_colors.zero_();
        if (backgrounds.has_value()) {
            render_colors.copy_(backgrounds->unsqueeze(-2).unsqueeze(-2).expand(render_colors.sizes()));
        }
        return std::make_tuple(render_colors, render_alphas, last_ids);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_3dgs_fwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");
    TORCH_CHECK(
        cfg.tile_size * cfg.tile_size <= static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup),
        "tile_size^2 exceeds the pipeline threadgroup limit");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:1];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 4);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 5);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:6];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(render_colors) offset:byte_offset(render_colors) atIndex:8];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:9];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:10];
            [enc setBytes:&cfg.I length:sizeof(cfg.I) atIndex:11];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:12];
            [enc setBytes:&cfg.channels length:sizeof(cfg.channels) atIndex:13];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:14];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:15];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:16];
            [enc setBytes:&cfg.n_tiles length:sizeof(cfg.n_tiles) atIndex:17];
            [enc setBytes:&cfg.total_tiles length:sizeof(cfg.total_tiles) atIndex:18];
            [enc setBytes:&cfg.n_isects length:sizeof(cfg.n_isects) atIndex:19];
            const uint32_t packed_u32 = cfg.packed ? 1u : 0u;
            const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
            const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
            [enc setBytes:&packed_u32 length:sizeof(packed_u32) atIndex:20];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:21];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:22];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient — the caller reads outputs via Python .cpu() or
    // subsequent MPS ops, both of which are serialised on the same stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(render_colors, render_alphas, last_ids);
}

std::tuple<c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd_op(
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    bool absgrad
) {
    const auto cfg = validate_common(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids);
    validate_backward_inputs(
        cfg,
        render_alphas,
        last_ids,
        v_render_colors,
        v_render_alphas,
        tile_offsets,
        image_height,
        image_width);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    c10::optional<at::Tensor> v_means2d_abs = c10::nullopt;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }

    if (cfg.n_isects == 0u || image_width == 0 || image_height == 0) {
        return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
    }

    at::Tensor tmp_means2d = at::zeros({cfg.n_isects, 2}, means2d.options());
    at::Tensor tmp_conics = at::zeros({cfg.n_isects, 3}, conics.options());
    at::Tensor tmp_colors = at::zeros({cfg.n_isects, cfg.channels}, colors.options());
    at::Tensor tmp_opacities = at::zeros({cfg.n_isects}, opacities.options());
    c10::optional<at::Tensor> tmp_means2d_abs = c10::nullopt;
    if (absgrad) {
        tmp_means2d_abs = at::zeros({cfg.n_isects, 2}, means2d.options());
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_3dgs_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");
    TORCH_CHECK(
        cfg.tile_size * cfg.tile_size <= static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup),
        "tile_size^2 exceeds the pipeline threadgroup limit");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:0];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:1];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 4);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 5);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:6];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:8];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:9];
            [enc setBuffer:to_mtl_buffer(v_render_colors) offset:byte_offset(v_render_colors) atIndex:10];
            [enc setBuffer:to_mtl_buffer(v_render_alphas) offset:byte_offset(v_render_alphas) atIndex:11];
            set_optional_tensor_buffer(enc, tmp_means2d_abs.has_value() ? *tmp_means2d_abs : at::Tensor{}, 12);
            [enc setBuffer:to_mtl_buffer(tmp_means2d) offset:byte_offset(tmp_means2d) atIndex:13];
            [enc setBuffer:to_mtl_buffer(tmp_conics) offset:byte_offset(tmp_conics) atIndex:14];
            [enc setBuffer:to_mtl_buffer(tmp_colors) offset:byte_offset(tmp_colors) atIndex:15];
            [enc setBuffer:to_mtl_buffer(tmp_opacities) offset:byte_offset(tmp_opacities) atIndex:16];
            [enc setBytes:&cfg.I length:sizeof(cfg.I) atIndex:17];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:18];
            [enc setBytes:&cfg.channels length:sizeof(cfg.channels) atIndex:19];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:20];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:21];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:22];
            [enc setBytes:&cfg.n_tiles length:sizeof(cfg.n_tiles) atIndex:23];
            [enc setBytes:&cfg.total_tiles length:sizeof(cfg.total_tiles) atIndex:24];
            [enc setBytes:&cfg.n_isects length:sizeof(cfg.n_isects) atIndex:25];
            const uint32_t packed_u32 = cfg.packed ? 1u : 0u;
            const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
            const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
            const uint32_t absgrad_u32 = absgrad ? 1u : 0u;
            [enc setBytes:&packed_u32 length:sizeof(packed_u32) atIndex:26];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:27];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:28];
            [enc setBytes:&absgrad_u32 length:sizeof(absgrad_u32) atIndex:29];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient — index_add_ below is an MPS op on the same stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor gather_idx = flatten_ids.to(at::kLong);

    at::Tensor flat_v_means2d = v_means2d.reshape({-1, 2});
    flat_v_means2d.index_add_(0, gather_idx, tmp_means2d);
    at::Tensor flat_v_conics = v_conics.reshape({-1, 3});
    flat_v_conics.index_add_(0, gather_idx, tmp_conics);
    at::Tensor flat_v_colors = v_colors.reshape({-1, static_cast<int64_t>(cfg.channels)});
    flat_v_colors.index_add_(0, gather_idx, tmp_colors);
    at::Tensor flat_v_opacities = v_opacities.reshape({-1});
    flat_v_opacities.index_add_(0, gather_idx, tmp_opacities);
    if (absgrad) {
        at::Tensor flat_v_means2d_abs = v_means2d_abs->reshape({-1, 2});
        flat_v_means2d_abs.index_add_(0, gather_idx, *tmp_means2d_abs);
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

}  // namespace gsplat::metal
