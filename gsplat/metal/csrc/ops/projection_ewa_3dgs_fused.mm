// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <algorithm>
#include <tuple>

#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "projection_ewa_3dgs_fused.h"
#include "quat_scale_to_covar_preci.h"

namespace gsplat::metal {

using namespace torch::indexing;

namespace {

constexpr int64_t kCameraModelPinhole = 0;
constexpr int64_t kCameraModelOrtho = 1;
constexpr int64_t kCameraModelFisheye = 2;
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

at::DimVector make_radii_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    shape.push_back(2);
    return shape;
}

at::DimVector make_means2d_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    return make_radii_shape(means, viewmats);
}

at::DimVector make_depths_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    return shape;
}

at::DimVector make_conics_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    shape.push_back(3);
    return shape;
}

at::DimVector make_triu_shape(const at::Tensor& means) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(means.size(-2));
    shape.push_back(6);
    return shape;
}

at::DimVector make_quat_shape(const at::Tensor& means) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(means.size(-2));
    shape.push_back(4);
    return shape;
}

at::DimVector make_scale_shape(const at::Tensor& means) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(means.size(-2));
    shape.push_back(3);
    return shape;
}

// Host-side symmetric-matrix packing helpers matching the Metal kernel layout.
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
projection_ewa_3dgs_fused_fwd_op(
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

    at::Tensor radii = at::empty(make_radii_shape(means, viewmats), means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty(make_means2d_shape(means, viewmats), means.options());
    at::Tensor depths = at::empty(make_depths_shape(means, viewmats), means.options());
    at::Tensor conics = at::empty(make_conics_shape(means, viewmats), means.options());
    c10::optional<at::Tensor> compensations = c10::nullopt;
    if (calc_compensations) {
        compensations = at::empty(make_depths_shape(means, viewmats), means.options());
    }

    const uint32_t B = static_cast<uint32_t>(means.numel() / (means.size(-2) * 3));
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t n = B * C * N;
    if (n == 0u) {
        return std::make_tuple(radii, means2d, depths, conics, compensations);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ewa_3dgs_fused_fwd_kernel");
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

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(covars_triu) offset:byte_offset(covars_triu) atIndex:1];
            set_optional_tensor_buffer(enc, opacities.value_or(at::Tensor{}), 2);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:3];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:4];
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:5];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:6];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:7];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:8];
            set_optional_tensor_buffer(enc, compensations.value_or(at::Tensor{}), 9);
            [enc setBytes:&B length:sizeof(B) atIndex:10];
            [enc setBytes:&C length:sizeof(C) atIndex:11];
            [enc setBytes:&N length:sizeof(N) atIndex:12];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:13];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:14];
            [enc setBytes:&eps2d_f length:sizeof(eps2d_f) atIndex:15];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:16];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:17];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:18];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:19];
            [enc setBytes:&has_opacities length:sizeof(has_opacities) atIndex:20];
            [enc setBytes:&has_compensations length:sizeof(has_compensations) atIndex:21];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd_op(
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
    const at::Tensor& radii,
    const at::Tensor& conics,
    const c10::optional<at::Tensor>& compensations,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_conics,
    const c10::optional<at::Tensor>& v_compensations,
    bool viewmats_requires_grad
) {
    validate_forward_common(means, viewmats, Ks, image_width, image_height, camera_model);
    check_mps_int32(radii, "radii");
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
    TORCH_CHECK(
        radii.sizes().equals(make_radii_shape(means, viewmats)),
        "radii shape must match forward output");
    TORCH_CHECK(
        conics.sizes().equals(make_conics_shape(means, viewmats)),
        "conics shape must match forward output");
    TORCH_CHECK(
        v_means2d.sizes().equals(make_means2d_shape(means, viewmats)),
        "v_means2d shape must match forward output");
    TORCH_CHECK(
        v_depths.sizes().equals(make_depths_shape(means, viewmats)),
        "v_depths shape must match forward output");
    TORCH_CHECK(
        v_conics.sizes().equals(make_conics_shape(means, viewmats)),
        "v_conics shape must match forward output");
    if (compensations.has_value()) {
        check_mps_float32(*compensations, "compensations");
        TORCH_CHECK(
            compensations->sizes().equals(make_depths_shape(means, viewmats)),
            "compensations shape must match forward output");
    }
    if (v_compensations.has_value()) {
        check_mps_float32(*v_compensations, "v_compensations");
        TORCH_CHECK(
            v_compensations->sizes().equals(make_depths_shape(means, viewmats)),
            "v_compensations shape must match forward output");
        TORCH_CHECK(compensations.has_value(), "v_compensations requires compensations to be defined");
    }

    const bool use_covars = covars.has_value();
    const uint32_t B = static_cast<uint32_t>(means.numel() / (means.size(-2) * 3));
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t n = B * C * N;

    at::DimVector tmp_means_shape(means.sizes().slice(0, means.dim() - 2));
    tmp_means_shape.push_back(viewmats.size(-3));
    tmp_means_shape.push_back(means.size(-2));
    tmp_means_shape.push_back(3);
    at::Tensor tmp_means = at::zeros(tmp_means_shape, means.options());

    c10::optional<at::Tensor> tmp_covars = c10::nullopt;
    c10::optional<at::Tensor> tmp_quats = c10::nullopt;
    c10::optional<at::Tensor> tmp_scales = c10::nullopt;
    c10::optional<at::Tensor> tmp_viewmats = c10::nullopt;
    if (use_covars) {
        at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
        shape.push_back(viewmats.size(-3));
        shape.push_back(means.size(-2));
        shape.push_back(6);
        tmp_covars = at::zeros(shape, means.options());
    } else {
        at::DimVector quat_shape(means.sizes().slice(0, means.dim() - 2));
        quat_shape.push_back(viewmats.size(-3));
        quat_shape.push_back(means.size(-2));
        quat_shape.push_back(4);
        tmp_quats = at::zeros(quat_shape, means.options());

        at::DimVector scale_shape(means.sizes().slice(0, means.dim() - 2));
        scale_shape.push_back(viewmats.size(-3));
        scale_shape.push_back(means.size(-2));
        scale_shape.push_back(3);
        tmp_scales = at::zeros(scale_shape, means.options());
    }
    if (viewmats_requires_grad) {
        at::DimVector viewmat_shape(means.sizes().slice(0, means.dim() - 2));
        viewmat_shape.push_back(viewmats.size(-3));
        viewmat_shape.push_back(means.size(-2));
        viewmat_shape.push_back(4);
        viewmat_shape.push_back(4);
        tmp_viewmats = at::zeros(viewmat_shape, means.options());
    }

    if (n == 0u) {
        at::Tensor v_means = at::zeros_like(means);
        at::Tensor v_covars = covars.has_value()
            ? at::zeros_like(*covars)
            : at::zeros(make_triu_shape(means), means.options());
        at::Tensor v_quats = quats.has_value()
            ? at::zeros_like(*quats)
            : at::zeros(make_quat_shape(means), means.options());
        at::Tensor v_scales = scales.has_value()
            ? at::zeros_like(*scales)
            : at::zeros(make_scale_shape(means), means.options());
        at::Tensor v_viewmats = at::zeros_like(viewmats);
        return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ewa_3dgs_fused_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const float eps2d_f = static_cast<float>(eps2d);
    const uint32_t camera_model_u32 = static_cast<uint32_t>(camera_model);
    const uint32_t use_covars_u32 = use_covars ? 1u : 0u;
    const uint32_t has_compensations_u32 = compensations.has_value() ? 1u : 0u;
    const uint32_t viewmats_requires_grad_u32 = viewmats_requires_grad ? 1u : 0u;

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
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:6];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:7];
            set_optional_tensor_buffer(enc, compensations.value_or(at::Tensor{}), 8);
            [enc setBuffer:to_mtl_buffer(v_means2d) offset:byte_offset(v_means2d) atIndex:9];
            [enc setBuffer:to_mtl_buffer(v_depths) offset:byte_offset(v_depths) atIndex:10];
            [enc setBuffer:to_mtl_buffer(v_conics) offset:byte_offset(v_conics) atIndex:11];
            set_optional_tensor_buffer(enc, v_compensations.value_or(at::Tensor{}), 12);
            [enc setBuffer:to_mtl_buffer(tmp_means) offset:byte_offset(tmp_means) atIndex:13];
            set_optional_tensor_buffer(enc, tmp_covars.value_or(at::Tensor{}), 14);
            set_optional_tensor_buffer(enc, tmp_quats.value_or(at::Tensor{}), 15);
            set_optional_tensor_buffer(enc, tmp_scales.value_or(at::Tensor{}), 16);
            set_optional_tensor_buffer(enc, tmp_viewmats.value_or(at::Tensor{}), 17);
            [enc setBytes:&B length:sizeof(B) atIndex:18];
            [enc setBytes:&C length:sizeof(C) atIndex:19];
            [enc setBytes:&N length:sizeof(N) atIndex:20];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:21];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:22];
            [enc setBytes:&eps2d_f length:sizeof(eps2d_f) atIndex:23];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:24];
            [enc setBytes:&use_covars_u32 length:sizeof(use_covars_u32) atIndex:25];
            [enc setBytes:&has_compensations_u32 length:sizeof(has_compensations_u32) atIndex:26];
            [enc setBytes:&viewmats_requires_grad_u32 length:sizeof(viewmats_requires_grad_u32) atIndex:27];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    // COMMIT (not COMMIT_AND_WAIT) is sufficient because all consumers of
    // tmp_* tensors (the sum(-3) calls below) are MPS operations on the same
    // stream and are automatically serialised.  Do not read tmp_* on CPU
    // between here and the sum(-3) calls.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor v_means = tmp_means.sum(-3);
    at::Tensor v_covars = covars.has_value()
        ? tmp_covars.value().sum(-3)
        : at::zeros(make_triu_shape(means), means.options());
    at::Tensor v_quats = quats.has_value()
        ? tmp_quats.value().sum(-3)
        : at::zeros(make_quat_shape(means), means.options());
    at::Tensor v_scales = scales.has_value()
        ? tmp_scales.value().sum(-3)
        : at::zeros(make_scale_shape(means), means.options());
    at::Tensor v_viewmats = viewmats_requires_grad
        ? tmp_viewmats.value().sum(-3)
        : at::zeros_like(viewmats);

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

}  // namespace gsplat::metal
