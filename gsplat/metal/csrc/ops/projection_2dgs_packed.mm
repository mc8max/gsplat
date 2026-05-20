// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <tuple>

#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "projection_2dgs_packed.h"

namespace gsplat::metal {

using namespace torch::indexing;

namespace {

constexpr uint32_t kThreadsPacked = 256u;

void validate_forward_common_2dgs(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height
) {
    check_mps_float32(means, "means");
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    check_mps_float32(viewmats, "viewmats");
    check_mps_float32(Ks, "Ks");

    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3]");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
    TORCH_CHECK(quats.dim() == means.dim(), "quats must have same ndim as means");
    TORCH_CHECK(quats.size(-2) == means.size(-2) && quats.size(-1) == 4, "quats must have shape [..., N, 4]");
    TORCH_CHECK(scales.dim() == means.dim(), "scales must have same ndim as means");
    TORCH_CHECK(scales.size(-2) == means.size(-2) && scales.size(-1) == 3, "scales must have shape [..., N, 3]");
    TORCH_CHECK(viewmats.dim() == means.dim() + 1, "viewmats must have shape [..., C, 4, 4]");
    TORCH_CHECK(Ks.dim() == means.dim() + 1, "Ks must have shape [..., C, 3, 3]");
    TORCH_CHECK(viewmats.size(-2) == 4 && viewmats.size(-1) == 4, "viewmats last two dimensions must be 4x4");
    TORCH_CHECK(Ks.size(-2) == 3 && Ks.size(-1) == 3, "Ks last two dimensions must be 3x3");
    TORCH_CHECK(
        viewmats.sizes().slice(0, viewmats.dim() - 3) == means.sizes().slice(0, means.dim() - 2),
        "means and viewmats batch dimensions must match");
    TORCH_CHECK(
        Ks.sizes().slice(0, Ks.dim() - 3) == means.sizes().slice(0, means.dim() - 2),
        "means and Ks batch dimensions must match");
    TORCH_CHECK(viewmats.size(-3) == Ks.size(-3), "viewmats and Ks camera dimension must match");
    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
}

}  // namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_packed_fwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane,
    double radius_clip
) {
    validate_forward_common_2dgs(means, quats, scales, viewmats, Ks, image_width, image_height);

    const uint32_t B = product_i64_to_u32(means.sizes().slice(0, means.dim() - 2), "means");
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t nrows = B * C;
    const uint32_t blocks_per_row = N == 0u ? 0u : round_up(N, kThreadsPacked) / kThreadsPacked;

    at::Tensor indptr = at::zeros({static_cast<int64_t>(nrows) + 1}, means.options().dtype(at::kInt));
    at::Tensor batch_ids = at::empty({0}, means.options().dtype(at::kLong));
    at::Tensor camera_ids = at::empty({0}, means.options().dtype(at::kLong));
    at::Tensor gaussian_ids = at::empty({0}, means.options().dtype(at::kLong));
    at::Tensor radii = at::empty({0, 2}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({0, 2}, means.options());
    at::Tensor depths = at::empty({0}, means.options());
    at::Tensor ray_transforms = at::empty({0, 9}, means.options());
    at::Tensor normals = at::empty({0, 3}, means.options());

    if (B == 0u || C == 0u || N == 0u) {
        return std::make_tuple(
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals
        );
    }

    at::Tensor block_cnts = at::zeros(
        {static_cast<int64_t>(nrows) * static_cast<int64_t>(blocks_per_row)},
        means.options().dtype(at::kInt)
    );

    auto& ctx = MetalContext::instance();
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const float near_plane_f = static_cast<float>(near_plane);
    const float far_plane_f = static_cast<float>(far_plane);
    const float radius_clip_f = static_cast<float>(radius_clip);

    // ---- Pass 1: count ----
    id<MTLComputePipelineState> count_pso = ctx.pipeline("projection_2dgs_packed_count_kernel");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:count_pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(block_cnts) offset:byte_offset(block_cnts) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&C length:sizeof(C) atIndex:7];
            [enc setBytes:&N length:sizeof(N) atIndex:8];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:9];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:10];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:11];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:12];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:13];
            [enc setBytes:&blocks_per_row length:sizeof(blocks_per_row) atIndex:14];

            [enc dispatchThreadgroups:MTLSizeMake(blocks_per_row, nrows, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPacked, 1, 1)];
        }
    });
    // COMMIT_AND_WAIT: CPU reads block_cnts immediately to determine nnz.
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    at::Tensor block_accum = at::cumsum(block_cnts, 0, at::kInt);
    const int32_t nnz = block_accum.numel() > 0 ? block_accum[-1].item<int32_t>() : 0;

    batch_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    camera_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    gaussian_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    radii = at::empty({nnz, 2}, means.options().dtype(at::kInt));
    means2d = at::empty({nnz, 2}, means.options());
    depths = at::empty({nnz}, means.options());
    ray_transforms = at::empty({nnz, 9}, means.options());
    normals = at::empty({nnz, 3}, means.options());

    at::Tensor row_counts = block_cnts.view({static_cast<int64_t>(nrows), static_cast<int64_t>(blocks_per_row)})
                                    .sum(1, false, at::kInt);
    indptr.index_put_({Slice(1, None)}, at::cumsum(row_counts, 0, at::kInt));

    if (nnz == 0) {
        return std::make_tuple(
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals
        );
    }

    // ---- Pass 2: emit ----
    id<MTLComputePipelineState> emit_pso = ctx.pipeline("projection_2dgs_packed_emit_kernel");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:emit_pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(block_accum) offset:byte_offset(block_accum) atIndex:5];
            [enc setBuffer:to_mtl_buffer(batch_ids) offset:byte_offset(batch_ids) atIndex:6];
            [enc setBuffer:to_mtl_buffer(camera_ids) offset:byte_offset(camera_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(gaussian_ids) offset:byte_offset(gaussian_ids) atIndex:8];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:9];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:10];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:11];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:12];
            [enc setBuffer:to_mtl_buffer(normals) offset:byte_offset(normals) atIndex:13];
            [enc setBytes:&B length:sizeof(B) atIndex:14];
            [enc setBytes:&C length:sizeof(C) atIndex:15];
            [enc setBytes:&N length:sizeof(N) atIndex:16];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:17];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:18];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:19];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:20];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:21];
            [enc setBytes:&blocks_per_row length:sizeof(blocks_per_row) atIndex:22];

            [enc dispatchThreadgroups:MTLSizeMake(blocks_per_row, nrows, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPacked, 1, 1)];
        }
    });
    // COMMIT: no immediate CPU read; downstream MPS ops are serialised on stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(
        indptr, batch_ids, camera_ids, gaussian_ids,
        radii, means2d, depths, ray_transforms, normals
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    const at::Tensor& batch_ids,
    const at::Tensor& camera_ids,
    const at::Tensor& gaussian_ids,
    const at::Tensor& ray_transforms,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_ray_transforms,
    const at::Tensor& v_normals,
    bool viewmats_requires_grad,
    bool sparse_grad
) {
    validate_forward_common_2dgs(means, quats, scales, viewmats, Ks, image_width, image_height);
    check_mps_int64(batch_ids, "batch_ids");
    check_mps_int64(camera_ids, "camera_ids");
    check_mps_int64(gaussian_ids, "gaussian_ids");
    check_mps_float32(ray_transforms, "ray_transforms");
    check_mps_float32(v_means2d, "v_means2d");
    check_mps_float32(v_depths, "v_depths");
    check_mps_float32(v_ray_transforms, "v_ray_transforms");
    check_mps_float32(v_normals, "v_normals");

    const int64_t nnz = batch_ids.numel();
    TORCH_CHECK(camera_ids.numel() == nnz, "camera_ids must match batch_ids length");
    TORCH_CHECK(gaussian_ids.numel() == nnz, "gaussian_ids must match batch_ids length");
    TORCH_CHECK(ray_transforms.sizes().equals({nnz, 9}), "ray_transforms must have shape [nnz, 9]");
    TORCH_CHECK(v_means2d.sizes().equals({nnz, 2}), "v_means2d must have shape [nnz, 2]");
    TORCH_CHECK(v_depths.sizes().equals({nnz}), "v_depths must have shape [nnz]");
    TORCH_CHECK(v_ray_transforms.sizes().equals({nnz, 9}), "v_ray_transforms must have shape [nnz, 9]");
    TORCH_CHECK(v_normals.sizes().equals({nnz, 3}), "v_normals must have shape [nnz, 3]");

    const uint32_t B = product_i64_to_u32(means.sizes().slice(0, means.dim() - 2), "means");
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));

    at::Tensor v_means;
    at::Tensor v_quats;
    at::Tensor v_scales;
    at::Tensor v_viewmats = at::zeros_like(viewmats);

    if (sparse_grad) {
        v_means = at::zeros({nnz, 3}, means.options());
        v_quats = at::zeros({nnz, 4}, means.options());
        v_scales = at::zeros({nnz, 3}, means.options());
    } else {
        v_means = at::zeros_like(means);
        v_quats = at::zeros_like(quats);
        v_scales = at::zeros_like(scales);
    }

    if (nnz == 0) {
        return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
    }

    at::Tensor tmp_means = at::zeros({nnz, 3}, means.options());
    at::Tensor tmp_quats = at::zeros({nnz, 4}, means.options());
    at::Tensor tmp_scales = at::zeros({nnz, 3}, means.options());
    c10::optional<at::Tensor> tmp_viewmats = c10::nullopt;
    if (viewmats_requires_grad) {
        tmp_viewmats = at::zeros({nnz, 16}, means.options());
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_2dgs_packed_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const uint32_t viewmats_requires_grad_u32 = viewmats_requires_grad ? 1u : 0u;
    const uint32_t nnz_u32 = static_cast<uint32_t>(nnz);

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(batch_ids) offset:byte_offset(batch_ids) atIndex:5];
            [enc setBuffer:to_mtl_buffer(camera_ids) offset:byte_offset(camera_ids) atIndex:6];
            [enc setBuffer:to_mtl_buffer(gaussian_ids) offset:byte_offset(gaussian_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:8];
            [enc setBuffer:to_mtl_buffer(v_means2d) offset:byte_offset(v_means2d) atIndex:9];
            [enc setBuffer:to_mtl_buffer(v_depths) offset:byte_offset(v_depths) atIndex:10];
            [enc setBuffer:to_mtl_buffer(v_ray_transforms) offset:byte_offset(v_ray_transforms) atIndex:11];
            [enc setBuffer:to_mtl_buffer(v_normals) offset:byte_offset(v_normals) atIndex:12];
            [enc setBuffer:to_mtl_buffer(tmp_means) offset:byte_offset(tmp_means) atIndex:13];
            [enc setBuffer:to_mtl_buffer(tmp_quats) offset:byte_offset(tmp_quats) atIndex:14];
            [enc setBuffer:to_mtl_buffer(tmp_scales) offset:byte_offset(tmp_scales) atIndex:15];
            set_optional_tensor_buffer(enc, tmp_viewmats.value_or(at::Tensor{}), 16);
            [enc setBytes:&B length:sizeof(B) atIndex:17];
            [enc setBytes:&C length:sizeof(C) atIndex:18];
            [enc setBytes:&N length:sizeof(N) atIndex:19];
            [enc setBytes:&nnz_u32 length:sizeof(nnz_u32) atIndex:20];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:21];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:22];
            [enc setBytes:&viewmats_requires_grad_u32 length:sizeof(viewmats_requires_grad_u32) atIndex:23];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, nnz_u32);
            [enc dispatchThreads:MTLSizeMake(nnz_u32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    if (sparse_grad) {
        v_means = tmp_means;
        v_quats = tmp_quats;
        v_scales = tmp_scales;
    } else {
        at::Tensor gather_idx = (batch_ids * static_cast<int64_t>(N) + gaussian_ids).to(at::kLong);
        v_means.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 3}).index_add_(0, gather_idx, tmp_means);
        v_quats.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 4}).index_add_(0, gather_idx, tmp_quats);
        v_scales.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 3}).index_add_(0, gather_idx, tmp_scales);
    }

    if (viewmats_requires_grad) {
        at::Tensor row_idx = (batch_ids * static_cast<int64_t>(C) + camera_ids).to(at::kLong);
        v_viewmats.view({static_cast<int64_t>(B) * static_cast<int64_t>(C), 16}).index_add_(
            0, row_idx, tmp_viewmats.value()
        );
    }

    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

}  // namespace gsplat::metal
