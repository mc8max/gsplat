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
#include "projection_2dgs_fused.h"

namespace gsplat::metal {

namespace {

void validate_forward_common(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane
) {
    check_mps_float32(means, "means");
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    check_mps_float32(viewmats, "viewmats");
    check_mps_float32(Ks, "Ks");

    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3]");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
    TORCH_CHECK(quats.sizes().slice(0, quats.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
                    quats.size(-2) == means.size(-2) && quats.size(-1) == 4,
                "quats must have shape [..., N, 4]");
    TORCH_CHECK(scales.sizes().slice(0, scales.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
                    scales.size(-2) == means.size(-2) && scales.size(-1) == 3,
                "scales must have shape [..., N, 3]");
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
    TORCH_CHECK(far_plane >= near_plane, "far_plane must be >= near_plane");
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

at::DimVector make_ray_transforms_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    shape.push_back(3);
    shape.push_back(3);
    return shape;
}

at::DimVector make_normals_shape(const at::Tensor& means, const at::Tensor& viewmats) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    shape.push_back(3);
    return shape;
}

// Validate tensor shapes and device without checking near/far plane range
// (not meaningful for backward).
void validate_shapes_common(
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
    TORCH_CHECK(quats.sizes().slice(0, quats.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
                    quats.size(-2) == means.size(-2) && quats.size(-1) == 4,
                "quats must have shape [..., N, 4]");
    TORCH_CHECK(scales.sizes().slice(0, scales.dim() - 2) == means.sizes().slice(0, means.dim() - 2) &&
                    scales.size(-2) == means.size(-2) && scales.size(-1) == 3,
                "scales must have shape [..., N, 3]");
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

void validate_backward_common(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    const at::Tensor& radii,
    const at::Tensor& ray_transforms,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_normals,
    const at::Tensor& v_ray_transforms,
    int64_t image_width,
    int64_t image_height
) {
    validate_shapes_common(means, quats, scales, viewmats, Ks, image_width, image_height);
    check_mps_int32(radii, "radii");
    check_mps_float32(ray_transforms, "ray_transforms");
    check_mps_float32(v_means2d, "v_means2d");
    check_mps_float32(v_depths, "v_depths");
    check_mps_float32(v_normals, "v_normals");
    check_mps_float32(v_ray_transforms, "v_ray_transforms");

    const auto make_std_vec = [](const at::DimVector& shape) {
        return std::vector<int64_t>(shape.begin(), shape.end());
    };
    const auto expected_radii = make_std_vec(make_radii_shape(means, viewmats));
    const auto expected_means2d = make_std_vec(make_means2d_shape(means, viewmats));
    const auto expected_depths = make_std_vec(make_depths_shape(means, viewmats));
    const auto expected_ray_transforms = make_std_vec(make_ray_transforms_shape(means, viewmats));
    const auto expected_normals = make_std_vec(make_normals_shape(means, viewmats));
    TORCH_CHECK(radii.sizes().vec() == expected_radii, "radii shape mismatch");
    TORCH_CHECK(ray_transforms.sizes().vec() == expected_ray_transforms, "ray_transforms shape mismatch");
    TORCH_CHECK(v_means2d.sizes().vec() == expected_means2d, "v_means2d shape mismatch");
    TORCH_CHECK(v_depths.sizes().vec() == expected_depths, "v_depths shape mismatch");
    TORCH_CHECK(v_normals.sizes().vec() == expected_normals, "v_normals shape mismatch");
    TORCH_CHECK(v_ray_transforms.sizes().vec() == expected_ray_transforms, "v_ray_transforms shape mismatch");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_fwd_op(
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
    validate_forward_common(
        means, quats, scales, viewmats, Ks, image_width, image_height, near_plane, far_plane
    );

    at::Tensor radii = at::empty(make_radii_shape(means, viewmats), means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty(make_means2d_shape(means, viewmats), means.options());
    at::Tensor depths = at::empty(make_depths_shape(means, viewmats), means.options());
    at::Tensor ray_transforms = at::empty(make_ray_transforms_shape(means, viewmats), means.options());
    at::Tensor normals = at::empty(make_normals_shape(means, viewmats), means.options());

    const uint32_t B = static_cast<uint32_t>(means.numel() / (means.size(-2) * 3));
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t n = B * C * N;
    if (n == 0u) {
        return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_2dgs_fused_fwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const float near_plane_f = static_cast<float>(near_plane);
    const float far_plane_f = static_cast<float>(far_plane);
    const float radius_clip_f = static_cast<float>(radius_clip);

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
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:5];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:6];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:7];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:8];
            [enc setBuffer:to_mtl_buffer(normals) offset:byte_offset(normals) atIndex:9];
            [enc setBytes:&B length:sizeof(B) atIndex:10];
            [enc setBytes:&C length:sizeof(C) atIndex:11];
            [enc setBytes:&N length:sizeof(N) atIndex:12];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:13];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:14];
            [enc setBytes:&near_plane_f length:sizeof(near_plane_f) atIndex:15];
            [enc setBytes:&far_plane_f length:sizeof(far_plane_f) atIndex:16];
            [enc setBytes:&radius_clip_f length:sizeof(radius_clip_f) atIndex:17];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    // COMMIT is sufficient — downstream MPS ops are serialised on the same stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_bwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    const at::Tensor& radii,
    const at::Tensor& ray_transforms,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_normals,
    const at::Tensor& v_ray_transforms,
    bool viewmats_requires_grad
) {
    validate_backward_common(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        radii,
        ray_transforms,
        v_means2d,
        v_depths,
        v_normals,
        v_ray_transforms,
        image_width,
        image_height
    );

    const uint32_t B = static_cast<uint32_t>(means.numel() / (means.size(-2) * 3));
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t n = B * C * N;

    at::DimVector tmp_shape(means.sizes().slice(0, means.dim() - 2));
    tmp_shape.push_back(viewmats.size(-3));
    tmp_shape.push_back(means.size(-2));

    at::DimVector quat_tmp_shape(tmp_shape);
    quat_tmp_shape.push_back(4);
    at::DimVector vec3_tmp_shape(tmp_shape);
    vec3_tmp_shape.push_back(3);
    at::DimVector viewmat_tmp_shape(tmp_shape);
    viewmat_tmp_shape.push_back(4);
    viewmat_tmp_shape.push_back(4);

    at::Tensor tmp_means = at::zeros(vec3_tmp_shape, means.options());
    at::Tensor tmp_quats = at::zeros(quat_tmp_shape, means.options());
    at::Tensor tmp_scales = at::zeros(vec3_tmp_shape, means.options());
    c10::optional<at::Tensor> tmp_viewmats = c10::nullopt;
    if (viewmats_requires_grad) {
        tmp_viewmats = at::zeros(viewmat_tmp_shape, means.options());
    }

    if (n == 0u) {
        at::Tensor v_means = at::zeros_like(means);
        at::Tensor v_quats = at::zeros_like(quats);
        at::Tensor v_scales = at::zeros_like(scales);
        at::Tensor v_viewmats = at::zeros_like(viewmats);
        return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_2dgs_fused_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
    const uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
    const uint32_t viewmats_requires_grad_u32 = viewmats_requires_grad ? 1u : 0u;

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
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:5];
            [enc setBuffer:to_mtl_buffer(ray_transforms) offset:byte_offset(ray_transforms) atIndex:6];
            [enc setBuffer:to_mtl_buffer(v_means2d) offset:byte_offset(v_means2d) atIndex:7];
            [enc setBuffer:to_mtl_buffer(v_depths) offset:byte_offset(v_depths) atIndex:8];
            [enc setBuffer:to_mtl_buffer(v_normals) offset:byte_offset(v_normals) atIndex:9];
            [enc setBuffer:to_mtl_buffer(v_ray_transforms) offset:byte_offset(v_ray_transforms) atIndex:10];
            [enc setBuffer:to_mtl_buffer(tmp_means) offset:byte_offset(tmp_means) atIndex:11];
            [enc setBuffer:to_mtl_buffer(tmp_quats) offset:byte_offset(tmp_quats) atIndex:12];
            [enc setBuffer:to_mtl_buffer(tmp_scales) offset:byte_offset(tmp_scales) atIndex:13];
            set_optional_tensor_buffer(enc, tmp_viewmats.value_or(at::Tensor{}), 14);
            [enc setBytes:&B length:sizeof(B) atIndex:15];
            [enc setBytes:&C length:sizeof(C) atIndex:16];
            [enc setBytes:&N length:sizeof(N) atIndex:17];
            [enc setBytes:&image_width_u32 length:sizeof(image_width_u32) atIndex:18];
            [enc setBytes:&image_height_u32 length:sizeof(image_height_u32) atIndex:19];
            [enc setBytes:&viewmats_requires_grad_u32 length:sizeof(viewmats_requires_grad_u32) atIndex:20];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    // COMMIT is sufficient — downstream MPS ops are serialised on the same stream.
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    at::Tensor v_means = tmp_means.sum(-3);
    at::Tensor v_quats = tmp_quats.sum(-3);
    at::Tensor v_scales = tmp_scales.sum(-3);
    at::Tensor v_viewmats = viewmats_requires_grad
        ? tmp_viewmats.value().sum(-3)
        : at::zeros_like(viewmats);
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

}  // namespace gsplat::metal
