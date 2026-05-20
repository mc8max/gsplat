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
#include "projection_ewa_3dgs_packed.h"
#include "quat_scale_to_covar_preci.h"

namespace gsplat::metal {

using namespace torch::indexing;

namespace {

constexpr int64_t kCameraModelPinhole = 0;
constexpr int64_t kCameraModelOrtho = 1;
constexpr int64_t kCameraModelFisheye = 2;
// Must match kThreadsPacked in projection_ewa_3dgs_packed.metal.
constexpr uint32_t kThreadsPacked = 256u;

void check_camera_model(int64_t camera_model) {
    TORCH_CHECK(
        camera_model == kCameraModelPinhole || camera_model == kCameraModelOrtho ||
            camera_model == kCameraModelFisheye,
        "camera_model must be one of pinhole(0), ortho(1), fisheye(2)");
}

void validate_forward_common(
    const at::Tensor& means,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    int64_t camera_model
) {
    check_mps_float32(means, "means");
    check_mps_float32(viewmats, "viewmats");
    check_mps_float32(Ks, "Ks");

    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3]");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
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
    check_camera_model(camera_model);
}

void validate_covars(const at::Tensor& covars, const at::Tensor& means) {
    check_mps_float32(covars, "covars");
    TORCH_CHECK(
        covars.sizes().slice(0, covars.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
            covars.size(-2) == means.size(-2) && covars.size(-1) == 6,
        "covars must have shape [..., N, 6]");
}

void validate_quats_scales(
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& means
) {
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    TORCH_CHECK(
        quats.sizes().slice(0, quats.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
            quats.size(-2) == means.size(-2) && quats.size(-1) == 4,
        "quats must have shape [..., N, 4]");
    TORCH_CHECK(
        scales.sizes().slice(0, scales.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
            scales.size(-2) == means.size(-2) && scales.size(-1) == 3,
        "scales must have shape [..., N, 3]");
}

void validate_opacities(const at::Tensor& opacities, const at::Tensor& means) {
    check_mps_float32(opacities, "opacities");
    TORCH_CHECK(
        opacities.sizes().slice(0, opacities.dim() - 1) == means.sizes().slice(0, means.dim() - 2) &&
            opacities.size(-1) == means.size(-2),
        "opacities must have shape [..., N]");
}

at::Tensor covars_triu_to_full(const at::Tensor& covars) {
    auto parts = at::unbind(covars, -1);
    at::Tensor row0 = at::stack({parts[0], parts[1], parts[2]}, -1);
    at::Tensor row1 = at::stack({parts[1], parts[3], parts[4]}, -1);
    at::Tensor row2 = at::stack({parts[2], parts[4], parts[5]}, -1);
    return at::stack({row0, row1, row2}, -2);
}

at::Tensor covars_full_to_triu(const at::Tensor& covars) {
    return at::stack(
        {
            covars.index({Ellipsis, 0, 0}),
            covars.index({Ellipsis, 0, 1}),
            covars.index({Ellipsis, 0, 2}),
            covars.index({Ellipsis, 1, 1}),
            covars.index({Ellipsis, 1, 2}),
            covars.index({Ellipsis, 2, 2}),
        },
        -1);
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
    c10::optional<at::Tensor>>
projection_ewa_3dgs_packed_fwd_op(
    const at::Tensor& means,
    const c10::optional<at::Tensor>& covars,
    const c10::optional<at::Tensor>& quats,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& opacities,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model
) {
    validate_forward_common(means, viewmats, Ks, image_width, image_height, camera_model);
    TORCH_CHECK(
        covars.has_value() ^ (quats.has_value() && scales.has_value()),
        "must provide either covars or {quats, scales}");
    if (covars.has_value()) {
        validate_covars(*covars, means);
    } else {
        validate_quats_scales(*quats, *scales, means);
    }
    if (opacities.has_value()) {
        validate_opacities(*opacities, means);
    }

    at::Tensor covars_full;
    if (covars.has_value()) {
        covars_full = covars_triu_to_full(*covars);
    } else {
        auto covar_out = quat_scale_to_covar_preci_fwd_op(*quats, *scales, true, false, false);
        covars_full = std::get<0>(covar_out).value();
    }
    at::Tensor covars_triu = covars_full_to_triu(covars_full).contiguous();

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
    at::Tensor conics = at::empty({0, 3}, means.options());
    c10::optional<at::Tensor> compensations = c10::nullopt;
    if (calc_compensations) {
        compensations = at::empty({0}, means.options());
    }

    if (B == 0u || C == 0u || N == 0u) {
        return std::make_tuple(
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations
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
    const float eps2d_f = static_cast<float>(eps2d);
    const float near_plane_f = static_cast<float>(near_plane);
    const float far_plane_f = static_cast<float>(far_plane);
    const float radius_clip_f = static_cast<float>(radius_clip);
    const uint32_t camera_model_u32 = static_cast<uint32_t>(camera_model);
    const uint32_t has_opacities = opacities.has_value() ? 1u : 0u;
    const uint32_t has_compensations = calc_compensations ? 1u : 0u;

    id<MTLComputePipelineState> count_pso = ctx.pipeline("projection_ewa_3dgs_packed_count_kernel");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:count_pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(covars_triu) offset:byte_offset(covars_triu) atIndex:1];
            set_optional_tensor_buffer(enc, opacities.value_or(at::Tensor{}), 2);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(block_cnts) offset:byte_offset(block_cnts) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&C length:sizeof(C) atIndex:7];
            [enc setBytes:&N length:sizeof(N) atIndex:8];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:9];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:10];
            [enc setBytes:&eps2d_f length:sizeof(eps2d_f) atIndex:11];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:12];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:13];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:14];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:15];
            [enc setBytes:&has_opacities length:sizeof(has_opacities) atIndex:16];
            [enc setBytes:&has_compensations length:sizeof(has_compensations) atIndex:17];
            [enc setBytes:&blocks_per_row length:sizeof(blocks_per_row) atIndex:18];

            [enc dispatchThreadgroups:MTLSizeMake(blocks_per_row, nrows, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPacked, 1, 1)];
        }
    });
    // COMMIT_AND_WAIT: the CPU reads block_cnts immediately via item<int32_t>()
    // to determine nnz before allocating output tensors.
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    at::Tensor block_accum = at::cumsum(block_cnts, 0, at::kInt);
    const int32_t nnz = block_accum.numel() > 0 ? block_accum[-1].item<int32_t>() : 0;

    batch_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    camera_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    gaussian_ids = at::empty({nnz}, means.options().dtype(at::kLong));
    radii = at::empty({nnz, 2}, means.options().dtype(at::kInt));
    means2d = at::empty({nnz, 2}, means.options());
    depths = at::empty({nnz}, means.options());
    conics = at::empty({nnz, 3}, means.options());
    if (calc_compensations) {
        compensations = at::zeros({nnz}, means.options());
    }

    at::Tensor row_counts = block_cnts.view({static_cast<int64_t>(nrows), static_cast<int64_t>(blocks_per_row)})
                                .sum(1, false, at::kInt);
    indptr.index_put_({Slice(1, None)}, at::cumsum(row_counts, 0, at::kInt));

    if (nnz == 0) {
        return std::make_tuple(
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations
        );
    }

    id<MTLComputePipelineState> emit_pso = ctx.pipeline("projection_ewa_3dgs_packed_emit_kernel");
    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:emit_pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(covars_triu) offset:byte_offset(covars_triu) atIndex:1];
            set_optional_tensor_buffer(enc, opacities.value_or(at::Tensor{}), 2);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(block_accum) offset:byte_offset(block_accum) atIndex:5];
            [enc setBuffer:to_mtl_buffer(batch_ids) offset:byte_offset(batch_ids) atIndex:6];
            [enc setBuffer:to_mtl_buffer(camera_ids) offset:byte_offset(camera_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(gaussian_ids) offset:byte_offset(gaussian_ids) atIndex:8];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:9];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:10];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:11];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:12];
            set_optional_tensor_buffer(enc, compensations.value_or(at::Tensor{}), 13);
            [enc setBytes:&B length:sizeof(B) atIndex:14];
            [enc setBytes:&C length:sizeof(C) atIndex:15];
            [enc setBytes:&N length:sizeof(N) atIndex:16];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:17];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:18];
            [enc setBytes:&eps2d_f length:sizeof(eps2d_f) atIndex:19];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:20];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:21];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:22];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:23];
            [enc setBytes:&has_opacities length:sizeof(has_opacities) atIndex:24];
            [enc setBytes:&has_compensations length:sizeof(has_compensations) atIndex:25];
            [enc setBytes:&blocks_per_row length:sizeof(blocks_per_row) atIndex:26];

            [enc dispatchThreadgroups:MTLSizeMake(blocks_per_row, nrows, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPacked, 1, 1)];
        }
    });
    // COMMIT: no immediate CPU read; downstream MPS ops are serialised on stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(
        indptr,
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd_op(
    const at::Tensor& means,
    const c10::optional<at::Tensor>& covars,
    const c10::optional<at::Tensor>& quats,
    const c10::optional<at::Tensor>& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    int64_t camera_model,
    const at::Tensor& batch_ids,
    const at::Tensor& camera_ids,
    const at::Tensor& gaussian_ids,
    const at::Tensor& conics,
    const c10::optional<at::Tensor>& compensations,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_conics,
    const c10::optional<at::Tensor>& v_compensations,
    bool viewmats_requires_grad,
    bool sparse_grad
) {
    validate_forward_common(means, viewmats, Ks, image_width, image_height, camera_model);
    check_mps_int64(batch_ids, "batch_ids");
    check_mps_int64(camera_ids, "camera_ids");
    check_mps_int64(gaussian_ids, "gaussian_ids");
    check_mps_float32(conics, "conics");
    check_mps_float32(v_means2d, "v_means2d");
    check_mps_float32(v_depths, "v_depths");
    check_mps_float32(v_conics, "v_conics");
    TORCH_CHECK(
        covars.has_value() ^ (quats.has_value() && scales.has_value()),
        "must provide either covars or {quats, scales}");
    if (covars.has_value()) {
        validate_covars(*covars, means);
    } else {
        validate_quats_scales(*quats, *scales, means);
    }
    const int64_t nnz = batch_ids.numel();
    TORCH_CHECK(camera_ids.numel() == nnz, "camera_ids must match batch_ids length");
    TORCH_CHECK(gaussian_ids.numel() == nnz, "gaussian_ids must match batch_ids length");
    TORCH_CHECK(conics.sizes().equals({nnz, 3}), "conics must have shape [nnz, 3]");
    TORCH_CHECK(v_means2d.sizes().equals({nnz, 2}), "v_means2d must have shape [nnz, 2]");
    TORCH_CHECK(v_depths.sizes().equals({nnz}), "v_depths must have shape [nnz]");
    TORCH_CHECK(v_conics.sizes().equals({nnz, 3}), "v_conics must have shape [nnz, 3]");
    if (compensations.has_value()) {
        check_mps_float32(*compensations, "compensations");
        TORCH_CHECK(compensations->sizes().equals({nnz}), "compensations must have shape [nnz]");
    }
    if (v_compensations.has_value()) {
        check_mps_float32(*v_compensations, "v_compensations");
        TORCH_CHECK(v_compensations->sizes().equals({nnz}), "v_compensations must have shape [nnz]");
        TORCH_CHECK(compensations.has_value(), "v_compensations requires compensations to be defined");
    }

    const bool use_covars = covars.has_value();
    const uint32_t B = product_i64_to_u32(means.sizes().slice(0, means.dim() - 2), "means");
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));

    at::Tensor v_means;
    at::Tensor v_covars;
    at::Tensor v_quats;
    at::Tensor v_scales;
    at::Tensor v_viewmats = at::zeros_like(viewmats);

    if (sparse_grad) {
        v_means = at::zeros({nnz, 3}, means.options());
        v_covars = use_covars ? at::zeros({nnz, 6}, means.options()) : at::zeros({0, 6}, means.options());
        v_quats = use_covars ? at::zeros({0, 4}, means.options()) : at::zeros({nnz, 4}, means.options());
        v_scales = use_covars ? at::zeros({0, 3}, means.options()) : at::zeros({nnz, 3}, means.options());
    } else {
        v_means = at::zeros_like(means);
        v_covars = use_covars ? at::zeros_like(*covars) : at::zeros({0, 6}, means.options());
        v_quats = use_covars ? at::zeros(at::DimVector{0, 4}, means.options()) : at::zeros_like(*quats);
        v_scales = use_covars ? at::zeros(at::DimVector{0, 3}, means.options()) : at::zeros_like(*scales);
    }

    if (nnz == 0) {
        if (!use_covars && !sparse_grad) {
            v_covars = at::zeros({0, 6}, means.options());
        }
        return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
    }

    at::Tensor tmp_means = at::zeros({nnz, 3}, means.options());
    c10::optional<at::Tensor> tmp_covars = c10::nullopt;
    c10::optional<at::Tensor> tmp_quats = c10::nullopt;
    c10::optional<at::Tensor> tmp_scales = c10::nullopt;
    c10::optional<at::Tensor> tmp_viewmats = c10::nullopt;
    if (use_covars) {
        tmp_covars = sparse_grad ? c10::optional<at::Tensor>(v_covars) : c10::optional<at::Tensor>(at::zeros({nnz, 6}, means.options()));
    } else {
        tmp_quats = sparse_grad ? c10::optional<at::Tensor>(v_quats) : c10::optional<at::Tensor>(at::zeros({nnz, 4}, means.options()));
        tmp_scales = sparse_grad ? c10::optional<at::Tensor>(v_scales) : c10::optional<at::Tensor>(at::zeros({nnz, 3}, means.options()));
    }
    if (viewmats_requires_grad) {
        tmp_viewmats = at::zeros({nnz, 16}, means.options());
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ewa_3dgs_packed_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const float eps2d_f = static_cast<float>(eps2d);
    const uint32_t camera_model_u32 = static_cast<uint32_t>(camera_model);
    const uint32_t use_covars_u32 = use_covars ? 1u : 0u;
    const uint32_t has_compensations_u32 = compensations.has_value() ? 1u : 0u;
    const uint32_t viewmats_requires_grad_u32 = viewmats_requires_grad ? 1u : 0u;
    const uint32_t nnz_u32 = static_cast<uint32_t>(nnz);

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            set_optional_tensor_buffer(enc, covars.value_or(at::Tensor{}), 1);
            set_optional_tensor_buffer(enc, quats.value_or(at::Tensor{}), 2);
            set_optional_tensor_buffer(enc, scales.value_or(at::Tensor{}), 3);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:4];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:5];
            [enc setBuffer:to_mtl_buffer(batch_ids) offset:byte_offset(batch_ids) atIndex:6];
            [enc setBuffer:to_mtl_buffer(camera_ids) offset:byte_offset(camera_ids) atIndex:7];
            [enc setBuffer:to_mtl_buffer(gaussian_ids) offset:byte_offset(gaussian_ids) atIndex:8];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:9];
            set_optional_tensor_buffer(enc, compensations.value_or(at::Tensor{}), 10);
            [enc setBuffer:to_mtl_buffer(v_means2d) offset:byte_offset(v_means2d) atIndex:11];
            [enc setBuffer:to_mtl_buffer(v_depths) offset:byte_offset(v_depths) atIndex:12];
            [enc setBuffer:to_mtl_buffer(v_conics) offset:byte_offset(v_conics) atIndex:13];
            set_optional_tensor_buffer(enc, v_compensations.value_or(at::Tensor{}), 14);
            [enc setBuffer:to_mtl_buffer(tmp_means) offset:byte_offset(tmp_means) atIndex:15];
            set_optional_tensor_buffer(enc, tmp_covars.value_or(at::Tensor{}), 16);
            set_optional_tensor_buffer(enc, tmp_quats.value_or(at::Tensor{}), 17);
            set_optional_tensor_buffer(enc, tmp_scales.value_or(at::Tensor{}), 18);
            set_optional_tensor_buffer(enc, tmp_viewmats.value_or(at::Tensor{}), 19);
            [enc setBytes:&B length:sizeof(B) atIndex:20];
            [enc setBytes:&C length:sizeof(C) atIndex:21];
            [enc setBytes:&N length:sizeof(N) atIndex:22];
            [enc setBytes:&nnz_u32 length:sizeof(nnz_u32) atIndex:23];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:24];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:25];
            [enc setBytes:&eps2d_f length:sizeof(eps2d_f) atIndex:26];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:27];
            [enc setBytes:&use_covars_u32 length:sizeof(use_covars_u32) atIndex:28];
            [enc setBytes:&has_compensations_u32 length:sizeof(has_compensations_u32) atIndex:29];
            [enc setBytes:&viewmats_requires_grad_u32 length:sizeof(viewmats_requires_grad_u32) atIndex:30];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, nnz_u32);
            [enc dispatchThreads:MTLSizeMake(nnz_u32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    if (sparse_grad) {
        v_means = tmp_means;
    } else {
        at::Tensor gather_idx = (batch_ids * static_cast<int64_t>(N) + gaussian_ids).to(at::kLong);
        v_means.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 3}).index_add_(0, gather_idx, tmp_means);
        if (use_covars) {
            v_covars = at::zeros_like(*covars);
            v_covars.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 6}).index_add_(
                0, gather_idx, tmp_covars.value()
            );
        } else {
            v_quats = at::zeros_like(*quats);
            v_scales = at::zeros_like(*scales);
            v_quats.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 4}).index_add_(
                0, gather_idx, tmp_quats.value()
            );
            v_scales.view({static_cast<int64_t>(B) * static_cast<int64_t>(N), 3}).index_add_(
                0, gather_idx, tmp_scales.value()
            );
        }
    }

    if (viewmats_requires_grad) {
        at::Tensor row_idx = (batch_ids * static_cast<int64_t>(C) + camera_ids).to(at::kLong);
        v_viewmats.view({static_cast<int64_t>(B) * static_cast<int64_t>(C), 16}).index_add_(
            0, row_idx, tmp_viewmats.value()
        );
    }

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

}  // namespace gsplat::metal
