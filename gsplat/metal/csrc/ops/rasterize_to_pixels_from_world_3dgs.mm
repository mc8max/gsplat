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
#include "rasterize_to_pixels_from_world_3dgs.h"

namespace gsplat::metal {

namespace {

struct RasterizeFromWorldConfig {
    uint32_t B;
    uint32_t C;
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
};

RasterizeFromWorldConfig validate_common(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& rays,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const c10::optional<at::Tensor>& sample_counts,
    const c10::optional<at::Tensor>& render_normals
) {
    check_mps_float32(means, "means");
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    check_mps_float32(colors, "colors");
    check_mps_float32(opacities, "opacities");
    check_mps_float32(viewmats, "viewmats");
    check_mps_float32(Ks, "Ks");
    check_mps_int32(tile_offsets, "tile_offsets");
    check_mps_int32(flatten_ids, "flatten_ids");
    if (rays.has_value()) {
        check_mps_float32(*rays, "rays");
    }

    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(
        static_cast<uint64_t>(tile_size) * static_cast<uint64_t>(tile_size) <= 256u,
        "Metal rasterize_to_pixels_from_world_3dgs currently requires tile_size^2 <= 256");
    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3]");
    TORCH_CHECK(quats.sizes().slice(0, quats.dim() - 1).vec() == means.sizes().slice(0, means.dim() - 1).vec(),
        "quats must have shape [..., N, 4] matching means");
    TORCH_CHECK(scales.sizes().equals(means.sizes()), "scales must have shape [..., N, 3] matching means");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
    TORCH_CHECK(quats.size(-1) == 4, "quats last dimension must be 4");
    TORCH_CHECK(scales.size(-1) == 3, "scales last dimension must be 3");
    TORCH_CHECK(colors.dim() >= 3, "colors must have shape [..., C, N, channels]");
    TORCH_CHECK(colors.size(-1) > 0, "colors last dimension must be positive");
    TORCH_CHECK(opacities.dim() >= 2, "opacities must have shape [..., C, N]");
    TORCH_CHECK(viewmats.dim() >= 3, "viewmats must have shape [..., C, 4, 4]");
    TORCH_CHECK(Ks.dim() >= 3, "Ks must have shape [..., C, 3, 3]");
    TORCH_CHECK(tile_offsets.dim() >= 3, "tile_offsets must have shape [..., C, tile_height, tile_width]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");

    const auto batch_dims = means.sizes().slice(0, means.dim() - 2);
    const int64_t N = means.size(-2);
    const int64_t C = colors.size(-3);
    TORCH_CHECK(C > 0, "colors camera dimension must be positive");

    auto expected_image_dims = batch_dims.vec();
    expected_image_dims.push_back(C);
    const auto image_dims = tile_offsets.sizes().slice(0, tile_offsets.dim() - 2);
    TORCH_CHECK(image_dims.vec() == expected_image_dims, "tile_offsets image dims must match batch dims + cameras");

    auto expected_colors = expected_image_dims;
    expected_colors.push_back(N);
    expected_colors.push_back(colors.size(-1));
    TORCH_CHECK(colors.sizes().vec() == expected_colors, "colors must have shape [..., C, N, channels]");

    auto expected_opacities = expected_image_dims;
    expected_opacities.push_back(N);
    TORCH_CHECK(opacities.sizes().vec() == expected_opacities, "opacities must have shape [..., C, N]");

    auto expected_viewmats = expected_image_dims;
    expected_viewmats.push_back(4);
    expected_viewmats.push_back(4);
    TORCH_CHECK(viewmats.sizes().vec() == expected_viewmats, "viewmats must have shape [..., C, 4, 4]");

    auto expected_Ks = expected_image_dims;
    expected_Ks.push_back(3);
    expected_Ks.push_back(3);
    TORCH_CHECK(Ks.sizes().vec() == expected_Ks, "Ks must have shape [..., C, 3, 3]");

    if (rays.has_value()) {
        auto expected_rays = expected_image_dims;
        expected_rays.push_back(image_height);
        expected_rays.push_back(image_width);
        expected_rays.push_back(6);
        TORCH_CHECK(rays->sizes().vec() == expected_rays, "rays must have shape [..., C, image_height, image_width, 6]");
    }

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
        auto expected = expected_image_dims;
        expected.push_back(colors.size(-1));
        TORCH_CHECK(backgrounds->sizes().vec() == expected, "backgrounds must have shape [..., C, channels]");
    }
    if (masks.has_value()) {
        check_mps_bool(*masks, "masks");
        TORCH_CHECK(masks->sizes().equals(tile_offsets.sizes()), "masks shape must match tile_offsets");
    }
    if (sample_counts.has_value()) {
        check_mps_int32(*sample_counts, "sample_counts");
        auto expected = expected_image_dims;
        expected.push_back(image_height);
        expected.push_back(image_width);
        TORCH_CHECK(sample_counts->sizes().vec() == expected, "sample_counts shape mismatch");
    }
    if (render_normals.has_value()) {
        check_mps_float32(*render_normals, "render_normals");
        auto expected = expected_image_dims;
        expected.push_back(image_height);
        expected.push_back(image_width);
        expected.push_back(3);
        TORCH_CHECK(render_normals->sizes().vec() == expected, "render_normals shape mismatch");
    }

    RasterizeFromWorldConfig cfg{};
    cfg.B = product_i64_to_u32(batch_dims, "means");
    cfg.C = static_cast<uint32_t>(C);
    cfg.I = product_i64_to_u32(image_dims, "tile_offsets");
    cfg.N = static_cast<uint32_t>(N);
    cfg.channels = static_cast<uint32_t>(colors.size(-1));
    cfg.tile_size = static_cast<uint32_t>(tile_size);
    cfg.tile_width = static_cast<uint32_t>(tile_width);
    cfg.tile_height = static_cast<uint32_t>(tile_height);
    cfg.n_tiles = static_cast<uint32_t>(tile_width * tile_height);
    cfg.total_tiles = cfg.I * cfg.n_tiles;
    cfg.n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    cfg.image_width = static_cast<uint32_t>(image_width);
    cfg.image_height = static_cast<uint32_t>(image_height);
    return cfg;
}

void validate_backward_inputs(
    const RasterizeFromWorldConfig& cfg,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const c10::optional<at::Tensor>& v_render_normals,
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
    if (v_render_normals.has_value()) {
        check_mps_float32(*v_render_normals, "v_render_normals");
        auto expected_normals = image_dims.vec();
        expected_normals.push_back(image_height);
        expected_normals.push_back(image_width);
        expected_normals.push_back(3);
        TORCH_CHECK(v_render_normals->sizes().vec() == expected_normals, "v_render_normals shape mismatch");
    }
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_from_world_3dgs_fwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& rays,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const c10::optional<at::Tensor>& sample_counts,
    const c10::optional<at::Tensor>& render_normals,
    bool use_hit_distance
) {
    const auto cfg = validate_common(
        means, quats, scales, colors, opacities, backgrounds, masks,
        image_width, image_height, tile_size, viewmats, Ks, rays, tile_offsets, flatten_ids, sample_counts, render_normals);

    at::Tensor render_colors = at::zeros(
        make_render_colors_shape(tile_offsets, image_height, image_width, colors.size(-1)),
        colors.options());
    at::Tensor render_alphas = at::zeros(
        make_render_alphas_shape(tile_offsets, image_height, image_width),
        means.options().dtype(at::kFloat));
    at::Tensor last_ids = at::full(
        make_last_ids_shape(tile_offsets, image_height, image_width),
        -1,
        tile_offsets.options().dtype(at::kInt));

    if (image_width == 0 || image_height == 0) {
        if (sample_counts.has_value()) {
            sample_counts->zero_();
        }
        if (render_normals.has_value()) {
            render_normals->zero_();
        }
        return std::make_tuple(render_colors, render_alphas, last_ids);
    }

    if (cfg.n_isects == 0u) {
        render_alphas.zero_();
        last_ids.zero_();
        render_colors.zero_();
        if (backgrounds.has_value()) {
            render_colors.copy_(backgrounds->unsqueeze(-2).unsqueeze(-2).expand(render_colors.sizes()));
        }
        if (sample_counts.has_value()) {
            sample_counts->zero_();
        }
        if (render_normals.has_value()) {
            render_normals->zero_();
        }
        return std::make_tuple(render_colors, render_alphas, last_ids);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_from_world_3dgs_fwd_kernel");
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
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:3];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:4];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 5);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 6);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:7];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:8];
            set_optional_tensor_buffer(enc, rays.has_value() ? *rays : at::Tensor{}, 9);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:10];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:11];
            [enc setBuffer:to_mtl_buffer(render_colors) offset:byte_offset(render_colors) atIndex:12];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:13];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:14];
            set_optional_tensor_buffer(enc, sample_counts.has_value() ? *sample_counts : at::Tensor{}, 15);
            set_optional_tensor_buffer(enc, render_normals.has_value() ? *render_normals : at::Tensor{}, 16);
            const uint32_t use_hit_distance_u32 = use_hit_distance ? 1u : 0u;
            [enc setBytes:&cfg length:sizeof(cfg) atIndex:17];
            [enc setBytes:&use_hit_distance_u32 length:sizeof(use_hit_distance_u32) atIndex:18];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(render_colors, render_alphas, last_ids);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_bwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const c10::optional<at::Tensor>& backgrounds,
    const c10::optional<at::Tensor>& masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& rays,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    const c10::optional<at::Tensor>& v_render_normals,
    bool use_hit_distance
) {
    const auto cfg = validate_common(
        means, quats, scales, colors, opacities, backgrounds, masks,
        image_width, image_height, tile_size, viewmats, Ks, rays, tile_offsets, flatten_ids, c10::nullopt, c10::nullopt);
    validate_backward_inputs(
        cfg, render_alphas, last_ids, v_render_colors, v_render_alphas, v_render_normals, tile_offsets, image_height, image_width);

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_rays = rays.has_value() ? at::zeros_like(*rays) : at::empty({0}, means.options());

    if (cfg.n_isects == 0u || image_width == 0 || image_height == 0) {
        return std::make_tuple(v_means, v_quats, v_scales, v_colors, v_opacities, v_rays);
    }

    at::Tensor tmp_means = at::zeros({cfg.n_isects, 3}, means.options());
    at::Tensor tmp_quats = at::zeros({cfg.n_isects, 4}, quats.options());
    at::Tensor tmp_scales = at::zeros({cfg.n_isects, 3}, scales.options());
    at::Tensor tmp_colors = at::zeros({cfg.n_isects, cfg.channels}, colors.options());
    at::Tensor tmp_opacities = at::zeros({cfg.n_isects}, opacities.options());

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("rasterize_to_pixels_from_world_3dgs_bwd_kernel");
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
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:3];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:4];
            set_optional_tensor_buffer(enc, backgrounds.has_value() ? *backgrounds : at::Tensor{}, 5);
            set_optional_tensor_buffer(enc, masks.has_value() ? *masks : at::Tensor{}, 6);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:7];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:8];
            set_optional_tensor_buffer(enc, rays.has_value() ? *rays : at::Tensor{}, 9);
            [enc setBuffer:to_mtl_buffer(tile_offsets) offset:byte_offset(tile_offsets) atIndex:10];
            [enc setBuffer:to_mtl_buffer(flatten_ids) offset:byte_offset(flatten_ids) atIndex:11];
            [enc setBuffer:to_mtl_buffer(render_alphas) offset:byte_offset(render_alphas) atIndex:12];
            [enc setBuffer:to_mtl_buffer(last_ids) offset:byte_offset(last_ids) atIndex:13];
            [enc setBuffer:to_mtl_buffer(v_render_colors) offset:byte_offset(v_render_colors) atIndex:14];
            [enc setBuffer:to_mtl_buffer(v_render_alphas) offset:byte_offset(v_render_alphas) atIndex:15];
            set_optional_tensor_buffer(enc, v_render_normals.has_value() ? *v_render_normals : at::Tensor{}, 16);
            [enc setBuffer:to_mtl_buffer(tmp_means) offset:byte_offset(tmp_means) atIndex:17];
            [enc setBuffer:to_mtl_buffer(tmp_quats) offset:byte_offset(tmp_quats) atIndex:18];
            [enc setBuffer:to_mtl_buffer(tmp_scales) offset:byte_offset(tmp_scales) atIndex:19];
            [enc setBuffer:to_mtl_buffer(tmp_colors) offset:byte_offset(tmp_colors) atIndex:20];
            [enc setBuffer:to_mtl_buffer(tmp_opacities) offset:byte_offset(tmp_opacities) atIndex:21];
            set_optional_tensor_buffer(enc, rays.has_value() ? v_rays : at::Tensor{}, 22);
            const uint32_t use_hit_distance_u32 = use_hit_distance ? 1u : 0u;
            [enc setBytes:&cfg length:sizeof(cfg) atIndex:23];
            [enc setBytes:&use_hit_distance_u32 length:sizeof(use_hit_distance_u32) atIndex:24];

            [enc dispatchThreadgroups:MTLSizeMake(cfg.tile_width, cfg.tile_height, cfg.I)
                  threadsPerThreadgroup:MTLSizeMake(cfg.tile_size, cfg.tile_size, 1)];
        }
    });
    // COMMIT is sufficient because the index_add_ reductions below are MPS ops
    // issued on the same stream and therefore observe kernel writes in order.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor gather_image_idx = flatten_ids.to(at::kLong);
    at::Tensor flat_v_colors = v_colors.reshape({-1, static_cast<int64_t>(cfg.channels)});
    flat_v_colors.index_add_(0, gather_image_idx, tmp_colors);
    at::Tensor flat_v_opacities = v_opacities.reshape({-1});
    flat_v_opacities.index_add_(0, gather_image_idx, tmp_opacities);

    const int64_t image_stride = static_cast<int64_t>(cfg.C) * static_cast<int64_t>(cfg.N);
    at::Tensor gather_batch_idx = at::floor_divide(gather_image_idx, image_stride);
    at::Tensor gather_gauss_idx = at::remainder(gather_image_idx, static_cast<int64_t>(cfg.N));
    at::Tensor gather_shared_idx = gather_batch_idx * static_cast<int64_t>(cfg.N) + gather_gauss_idx;

    at::Tensor flat_v_means = v_means.reshape({-1, 3});
    flat_v_means.index_add_(0, gather_shared_idx, tmp_means);
    at::Tensor flat_v_quats = v_quats.reshape({-1, 4});
    flat_v_quats.index_add_(0, gather_shared_idx, tmp_quats);
    at::Tensor flat_v_scales = v_scales.reshape({-1, 3});
    flat_v_scales.index_add_(0, gather_shared_idx, tmp_scales);

    return std::make_tuple(v_means, v_quats, v_scales, v_colors, v_opacities, v_rays);
}

}  // namespace gsplat::metal
