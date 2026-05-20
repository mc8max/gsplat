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
#include "rasterize_to_pixels_2dgs.h"

namespace gsplat::metal {

namespace {

struct Rasterize2DGSConfig {
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

struct Rasterize2DGSBwdDispatchConfig {
    uint32_t I;
    uint32_t N;
    uint32_t channels;
    uint32_t tile_size;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t n_tiles;
    uint32_t total_tiles;
    uint32_t n_isects;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t absgrad;
};

Rasterize2DGSConfig validate_common(
    bool packed,
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids
) {
    check_mps_float32(means2d, "means2d");
    check_mps_float32(ray_transforms, "ray_transforms");
    check_mps_float32(colors, "colors");
    check_mps_float32(opacities, "opacities");
    check_mps_float32(normals, "normals");
    check_mps_int32(tile_offsets, "tile_offsets");
    check_mps_int32(flatten_ids, "flatten_ids");

    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(
        static_cast<uint64_t>(tile_size) * static_cast<uint64_t>(tile_size) <= 256u,
        "Metal rasterize_to_pixels_2dgs currently requires tile_size^2 <= 256");
    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dimension must be 2");
    TORCH_CHECK(colors.dim() >= 2, "colors must have at least 2 dimensions");
    TORCH_CHECK(colors.size(-1) > 0, "colors last dimension must be positive");
    TORCH_CHECK(tile_offsets.dim() >= 2, "tile_offsets must have shape [..., tile_height, tile_width]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");

    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    const int64_t channels = colors.size(-1);

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
        expected.push_back(channels);
        TORCH_CHECK(backgrounds->sizes().vec() == expected, "backgrounds must have shape [..., channels]");
    }
    if (masks.has_value()) {
        check_mps_bool(*masks, "masks");
        TORCH_CHECK(masks->sizes().equals(tile_offsets.sizes()), "masks shape must match tile_offsets");
    }

    Rasterize2DGSConfig cfg{};
    cfg.packed = packed;
    cfg.I = product_i64_to_u32(image_dims, "tile_offsets");
    cfg.channels = static_cast<uint32_t>(channels);
    cfg.tile_size = static_cast<uint32_t>(tile_size);
    cfg.tile_width = static_cast<uint32_t>(tile_width);
    cfg.tile_height = static_cast<uint32_t>(tile_height);
    cfg.n_tiles = static_cast<uint32_t>(tile_width * tile_height);
    cfg.total_tiles = cfg.I * cfg.n_tiles;
    cfg.n_isects = static_cast<uint32_t>(flatten_ids.size(0));

    if (cfg.packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(means2d.sizes().equals({nnz, 2}), "packed means2d must have shape (nnz, 2)");
        TORCH_CHECK(
            ray_transforms.sizes().equals({nnz, 3, 3}),
            "packed ray_transforms must have shape (nnz, 3, 3)");
        TORCH_CHECK(
            colors.dim() == 2 && colors.size(0) == nnz,
            "packed colors must have shape (nnz, channels)");
        TORCH_CHECK(opacities.sizes().equals({nnz}), "packed opacities must have shape (nnz,)");
        TORCH_CHECK(normals.sizes().equals({nnz, 3}), "packed normals must have shape (nnz, 3)");
        cfg.N = 0;
    } else {
        TORCH_CHECK(means2d.dim() >= 3, "means2d must have shape [..., N, 2]");
        const auto means_image_dims = means2d.sizes().slice(0, means2d.dim() - 2);
        const int64_t N = means2d.size(-2);
        TORCH_CHECK(means_image_dims.vec() == image_dims.vec(), "means2d image dims must match tile_offsets");
        TORCH_CHECK(
            ray_transforms.sizes().slice(0, ray_transforms.dim() - 2) ==
                    means2d.sizes().slice(0, means2d.dim() - 1) &&
                ray_transforms.size(-2) == 3 && ray_transforms.size(-1) == 3,
            "ray_transforms must have shape [..., N, 3, 3]");
        auto expected_colors = means_image_dims.vec();
        expected_colors.push_back(N);
        expected_colors.push_back(channels);
        TORCH_CHECK(colors.sizes().vec() == expected_colors, "colors must have shape [..., N, channels]");
        auto expected_opacities = means2d.sizes().slice(0, means2d.dim() - 1).vec();
        TORCH_CHECK(opacities.sizes().vec() == expected_opacities, "opacities must have shape [..., N]");
        auto expected_normals = means2d.sizes().slice(0, means2d.dim() - 1).vec();
        expected_normals.push_back(3);
        TORCH_CHECK(
            normals.sizes().vec() == expected_normals,
            "normals must have shape [..., N, 3], got ",
            normals.sizes(),
            " expected ",
            expected_normals);
        cfg.N = static_cast<uint32_t>(N);
    }
    return cfg;
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


at::DimVector make_last_ids_shape(
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width
) {
    at::DimVector dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    dims.append({image_height, image_width});
    return dims;
}

void validate_backward_inputs(
    const Rasterize2DGSConfig& cfg,
    const at::Tensor& densify,
    const at::Tensor& render_colors,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& median_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const at::Tensor& v_render_normals,
    const at::Tensor& v_render_distort,
    const at::Tensor& v_render_median,
    const at::Tensor& tile_offsets,
    int64_t image_height,
    int64_t image_width
) {
    check_mps_float32(densify, "densify");
    check_mps_float32(render_colors, "render_colors");
    check_mps_float32(render_alphas, "render_alphas");
    check_mps_int32(last_ids, "last_ids");
    check_mps_int32(median_ids, "median_ids");
    check_mps_float32(v_render_colors, "v_render_colors");
    check_mps_float32(v_render_alphas, "v_render_alphas");
    check_mps_float32(v_render_normals, "v_render_normals");
    check_mps_float32(v_render_distort, "v_render_distort");
    check_mps_float32(v_render_median, "v_render_median");

    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    if (cfg.packed) {
        TORCH_CHECK(
            densify.dim() == 2 && densify.size(1) == 2,
            "packed densify must have shape (nnz, 2)");
    } else {
        at::DimVector expected_densify(image_dims.begin(), image_dims.end());
        expected_densify.push_back(cfg.N);
        expected_densify.push_back(2);
        TORCH_CHECK(densify.sizes().equals(expected_densify), "densify must have shape [..., N, 2]");
    }

    const auto expected_colors =
        make_render_colors_shape(tile_offsets, image_height, image_width, cfg.channels);
    const auto expected_scalar =
        make_render_colors_shape(tile_offsets, image_height, image_width, 1);
    const auto expected_normals =
        make_render_colors_shape(tile_offsets, image_height, image_width, 3);
    const auto expected_ids =
        make_last_ids_shape(tile_offsets, image_height, image_width);

    TORCH_CHECK(render_colors.sizes().equals(expected_colors), "render_colors must match forward output shape");
    TORCH_CHECK(render_alphas.sizes().equals(expected_scalar), "render_alphas must match forward output shape");
    TORCH_CHECK(last_ids.sizes().equals(expected_ids), "last_ids must match forward output shape");
    TORCH_CHECK(median_ids.sizes().equals(expected_ids), "median_ids must match forward output shape");
    TORCH_CHECK(v_render_colors.sizes().equals(expected_colors), "v_render_colors must match render_colors");
    TORCH_CHECK(v_render_alphas.sizes().equals(expected_scalar), "v_render_alphas must match render_alphas");
    TORCH_CHECK(v_render_normals.sizes().equals(expected_normals), "v_render_normals must match normal output shape");
    TORCH_CHECK(v_render_distort.sizes().equals(expected_scalar), "v_render_distort must match distort output shape");
    TORCH_CHECK(v_render_median.sizes().equals(expected_scalar), "v_render_median must match median output shape");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_2dgs_fwd_op(
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    bool packed
) {
    const auto cfg = validate_common(
        packed,
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
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
    at::Tensor render_alphas = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, 1),
        means2d.options());
    at::Tensor render_normals = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, 3),
        means2d.options());
    at::Tensor render_distort = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, 1),
        means2d.options());
    at::Tensor render_median = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, 1),
        means2d.options());
    at::Tensor last_ids = at::zeros(
        make_last_ids_shape(tile_offsets, image_height, image_width),
        tile_offsets.options().dtype(at::kInt));
    at::Tensor median_ids = at::zeros(
        make_last_ids_shape(tile_offsets, image_height, image_width),
        tile_offsets.options().dtype(at::kInt));

    if (image_width == 0 || image_height == 0) {
        return std::make_tuple(
            render_colors, render_alphas, render_normals, render_distort, render_median, last_ids, median_ids);
    }

    if (cfg.n_isects == 0u) {
        if (backgrounds.has_value()) {
            render_colors.copy_(backgrounds->unsqueeze(-2).unsqueeze(-2).expand(render_colors.sizes()));
        }
        return std::make_tuple(
            render_colors, render_alphas, render_normals, render_distort, render_median, last_ids, median_ids);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_2dgs_fwd_kernel");
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
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:1];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            [enc setBuffer:to_mtl_buffer(normals) offset:byte_offset(normals) atIndex:4];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 5);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 6);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:7];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:8];
            [enc setBuffer:to_mtl_buffer(render_colors) offset:byte_offset(render_colors) atIndex:9];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:10];
            [enc setBuffer:to_mtl_buffer(render_normals) offset:byte_offset(render_normals) atIndex:11];
            [enc setBuffer:to_mtl_buffer(render_distort) offset:byte_offset(render_distort) atIndex:12];
            [enc setBuffer:to_mtl_buffer(render_median) offset:byte_offset(render_median) atIndex:13];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:14];
            [enc setBuffer:to_mtl_buffer(median_ids) offset:byte_offset(median_ids) atIndex:15];
            [enc setBytes:&cfg.I length:sizeof(cfg.I) atIndex:16];
            [enc setBytes:&cfg.N length:sizeof(cfg.N) atIndex:17];
            [enc setBytes:&cfg.channels length:sizeof(cfg.channels) atIndex:18];
            [enc setBytes:&cfg.tile_size length:sizeof(cfg.tile_size) atIndex:19];
            [enc setBytes:&cfg.tile_width length:sizeof(cfg.tile_width) atIndex:20];
            [enc setBytes:&cfg.tile_height length:sizeof(cfg.tile_height) atIndex:21];
            [enc setBytes:&cfg.n_tiles length:sizeof(cfg.n_tiles) atIndex:22];
            [enc setBytes:&cfg.total_tiles length:sizeof(cfg.total_tiles) atIndex:23];
            [enc setBytes:&cfg.n_isects length:sizeof(cfg.n_isects) atIndex:24];
            const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
            const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:25];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:26];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient — no immediate CPU read follows; downstream MPS ops on the same stream are automatically serialised.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(
        render_colors, render_alphas, render_normals, render_distort, render_median, last_ids, median_ids);
}

std::tuple<c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_2dgs_bwd_op(
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const at::Tensor& densify,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_colors,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& median_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const at::Tensor& v_render_normals,
    const at::Tensor& v_render_distort,
    const at::Tensor& v_render_median,
    bool packed,
    bool absgrad
) {
    const auto cfg = validate_common(
        packed,
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids);
    validate_backward_inputs(
        cfg,
        densify,
        render_colors,
        render_alphas,
        last_ids,
        median_ids,
        v_render_colors,
        v_render_alphas,
        v_render_normals,
        v_render_distort,
        v_render_median,
        tile_offsets,
        image_height,
        image_width);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_ray_transforms = at::zeros_like(ray_transforms);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_normals = at::zeros_like(normals);
    at::Tensor v_densify = at::zeros_like(densify);
    c10::optional<at::Tensor> v_means2d_abs = c10::nullopt;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }

    if (cfg.n_isects == 0u || image_width == 0 || image_height == 0) {
        return std::make_tuple(
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify);
    }

    at::Tensor tmp_means2d = at::zeros({cfg.n_isects, 2}, means2d.options());
    at::Tensor tmp_ray_transforms = at::zeros({cfg.n_isects, 9}, ray_transforms.options());
    at::Tensor tmp_colors = at::zeros({cfg.n_isects, cfg.channels}, colors.options());
    at::Tensor tmp_opacities = at::zeros({cfg.n_isects}, opacities.options());
    at::Tensor tmp_normals = at::zeros({cfg.n_isects, 3}, normals.options());
    c10::optional<at::Tensor> tmp_means2d_abs = c10::nullopt;
    if (absgrad) {
        tmp_means2d_abs = at::zeros({cfg.n_isects, 2}, means2d.options());
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_2dgs_bwd_kernel");
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
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:1];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:2];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:3];
            [enc setBuffer:to_mtl_buffer(normals) offset:byte_offset(normals) atIndex:4];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 5);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 6);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:7];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:8];
            [enc setBuffer:to_mtl_buffer(render_colors) offset:byte_offset(render_colors) atIndex:9];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:10];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:11];
            [enc setBuffer:to_mtl_buffer(median_ids) offset:byte_offset(median_ids) atIndex:12];
            [enc setBuffer:to_mtl_buffer(v_render_colors) offset:byte_offset(v_render_colors) atIndex:13];
            [enc setBuffer:to_mtl_buffer(v_render_alphas) offset:byte_offset(v_render_alphas) atIndex:14];
            [enc setBuffer:to_mtl_buffer(v_render_normals) offset:byte_offset(v_render_normals) atIndex:15];
            [enc setBuffer:to_mtl_buffer(v_render_distort) offset:byte_offset(v_render_distort) atIndex:16];
            [enc setBuffer:to_mtl_buffer(v_render_median) offset:byte_offset(v_render_median) atIndex:17];
            set_optional_tensor_buffer(enc, tmp_means2d_abs.has_value() ? *tmp_means2d_abs : at::Tensor{}, 18);
            [enc setBuffer:to_mtl_buffer(tmp_means2d) offset:byte_offset(tmp_means2d) atIndex:19];
            [enc setBuffer:to_mtl_buffer(tmp_ray_transforms) offset:byte_offset(tmp_ray_transforms) atIndex:20];
            [enc setBuffer:to_mtl_buffer(tmp_colors) offset:byte_offset(tmp_colors) atIndex:21];
            [enc setBuffer:to_mtl_buffer(tmp_opacities) offset:byte_offset(tmp_opacities) atIndex:22];
            [enc setBuffer:to_mtl_buffer(tmp_normals) offset:byte_offset(tmp_normals) atIndex:23];
            const Rasterize2DGSBwdDispatchConfig dispatch_cfg{
                cfg.I,
                cfg.N,
                cfg.channels,
                cfg.tile_size,
                cfg.tile_width,
                cfg.tile_height,
                cfg.n_tiles,
                cfg.total_tiles,
                cfg.n_isects,
                static_cast<uint32_t>(image_width),
                static_cast<uint32_t>(image_height),
                absgrad ? 1u : 0u,
            };
            [enc setBytes:&dispatch_cfg length:sizeof(dispatch_cfg) atIndex:24];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient — no immediate CPU read follows; downstream MPS ops on the same stream are automatically serialised.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor gather_idx = flatten_ids.to(at::kLong);
    v_means2d.reshape({-1, 2}).index_add_(0, gather_idx, tmp_means2d);
    v_ray_transforms.reshape({-1, 9}).index_add_(0, gather_idx, tmp_ray_transforms);
    v_colors.reshape({-1, static_cast<int64_t>(cfg.channels)}).index_add_(0, gather_idx, tmp_colors);
    v_opacities.reshape({-1}).index_add_(0, gather_idx, tmp_opacities);
    v_normals.reshape({-1, 3}).index_add_(0, gather_idx, tmp_normals);
    if (absgrad) {
        v_means2d_abs->reshape({-1, 2}).index_add_(0, gather_idx, *tmp_means2d_abs);
    }

    at::Tensor depth = ray_transforms.select(-2, 2).select(-1, 2);
    v_densify.select(-1, 0).copy_(v_ray_transforms.select(-2, 0).select(-1, 2) * depth);
    v_densify.select(-1, 1).copy_(v_ray_transforms.select(-2, 1).select(-1, 2) * depth);

    return std::make_tuple(
        v_means2d_abs,
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify);
}

}  // namespace gsplat::metal
