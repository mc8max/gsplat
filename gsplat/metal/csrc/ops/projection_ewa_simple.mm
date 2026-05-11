// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <algorithm>
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "projection_ewa_simple.h"

namespace gsplat::metal {

namespace {

constexpr int64_t kCameraModelPinhole = 0;
constexpr int64_t kCameraModelOrtho = 1;
constexpr int64_t kCameraModelFisheye = 2;

void check_common_inputs(
    const at::Tensor& means,
    const at::Tensor& covars,
    const at::Tensor& Ks,
    int64_t width,
    int64_t height,
    int64_t camera_model
) {
    check_mps_float32(means, "means");
    check_mps_float32(covars, "covars");
    check_mps_float32(Ks, "Ks");

    TORCH_CHECK(means.dim() >= 3, "means must have shape [..., C, N, 3]");
    TORCH_CHECK(covars.dim() == means.dim() + 1, "covars must have shape [..., C, N, 3, 3]");
    TORCH_CHECK(Ks.dim() == means.dim(), "Ks must have shape [..., C, 3, 3]");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
    TORCH_CHECK(covars.size(-2) == 3 && covars.size(-1) == 3, "covars last two dimensions must be 3x3");
    TORCH_CHECK(Ks.size(-2) == 3 && Ks.size(-1) == 3, "Ks last two dimensions must be 3x3");
    TORCH_CHECK(means.size(-3) == covars.size(-4), "means and covars camera dimension must match");
    TORCH_CHECK(means.size(-3) == Ks.size(-3), "means and Ks camera dimension must match");
    TORCH_CHECK(means.size(-2) == covars.size(-3), "means and covars Gaussian dimension must match");
    TORCH_CHECK(
        means.sizes().slice(0, means.dim() - 3) == covars.sizes().slice(0, covars.dim() - 4),
        "means and covars batch dimensions must match");
    TORCH_CHECK(
        means.sizes().slice(0, means.dim() - 3) == Ks.sizes().slice(0, Ks.dim() - 3),
        "means and Ks batch dimensions must match");
    TORCH_CHECK(width >= 0, "width must be non-negative");
    TORCH_CHECK(height >= 0, "height must be non-negative");
    TORCH_CHECK(
        camera_model == kCameraModelPinhole || camera_model == kCameraModelOrtho ||
            camera_model == kCameraModelFisheye,
        "camera_model must be one of pinhole(0), ortho(1), fisheye(2)");
}

at::DimVector make_means2d_shape(const at::Tensor& means) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 3));
    shape.append({means.size(-3), means.size(-2), 2});
    return shape;
}

at::DimVector make_covars2d_shape(const at::Tensor& means) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 3));
    shape.append({means.size(-3), means.size(-2), 2, 2});
    return shape;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd_op(
    const at::Tensor& means,
    const at::Tensor& covars,
    const at::Tensor& Ks,
    int64_t width,
    int64_t height,
    int64_t camera_model
) {
    check_common_inputs(means, covars, Ks, width, height, camera_model);

    at::Tensor means2d = at::empty(make_means2d_shape(means), means.options());
    at::Tensor covars2d = at::empty(make_covars2d_shape(means), means.options());

    const uint32_t C = static_cast<uint32_t>(means.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t B = static_cast<uint32_t>(means.numel() / (C * N * 3));
    const uint32_t width_u32 = static_cast<uint32_t>(width);
    const uint32_t height_u32 = static_cast<uint32_t>(height);
    const uint32_t camera_model_u32 = static_cast<uint32_t>(camera_model);
    const uint32_t n = B * C * N;
    if (n == 0) {
        return std::make_tuple(means2d, covars2d);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ewa_simple_fwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(covars) offset:byte_offset(covars) atIndex:1];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:2];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:3];
            [enc setBuffer:to_mtl_buffer(covars2d) offset:byte_offset(covars2d) atIndex:4];
            [enc setBytes:&B length:sizeof(B) atIndex:5];
            [enc setBytes:&C length:sizeof(C) atIndex:6];
            [enc setBytes:&N length:sizeof(N) atIndex:7];
            [enc setBytes:&width_u32 length:sizeof(width_u32) atIndex:8];
            [enc setBytes:&height_u32 length:sizeof(height_u32) atIndex:9];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:10];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(means2d, covars2d);
}

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd_op(
    const at::Tensor& means,
    const at::Tensor& covars,
    const at::Tensor& Ks,
    int64_t width,
    int64_t height,
    int64_t camera_model,
    const at::Tensor& v_means2d,
    const at::Tensor& v_covars2d
) {
    check_common_inputs(means, covars, Ks, width, height, camera_model);
    check_mps_float32(v_means2d, "v_means2d");
    check_mps_float32(v_covars2d, "v_covars2d");
    TORCH_CHECK(
        v_means2d.sizes().equals(make_means2d_shape(means)),
        "v_means2d shape must match forward means2d shape");
    TORCH_CHECK(
        v_covars2d.sizes().equals(make_covars2d_shape(means)),
        "v_covars2d shape must match forward covars2d shape");

    at::Tensor v_means = at::empty_like(means);
    at::Tensor v_covars = at::empty_like(covars);

    const uint32_t C = static_cast<uint32_t>(means.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t B = static_cast<uint32_t>(means.numel() / (C * N * 3));
    const uint32_t width_u32 = static_cast<uint32_t>(width);
    const uint32_t height_u32 = static_cast<uint32_t>(height);
    const uint32_t camera_model_u32 = static_cast<uint32_t>(camera_model);
    const uint32_t n = B * C * N;
    if (n == 0) {
        return std::make_tuple(v_means, v_covars);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ewa_simple_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(covars) offset:byte_offset(covars) atIndex:1];
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:2];
            [enc setBuffer:to_mtl_buffer(v_means2d) offset:byte_offset(v_means2d) atIndex:3];
            [enc setBuffer:to_mtl_buffer(v_covars2d) offset:byte_offset(v_covars2d) atIndex:4];
            [enc setBuffer:to_mtl_buffer(v_means) offset:byte_offset(v_means) atIndex:5];
            [enc setBuffer:to_mtl_buffer(v_covars) offset:byte_offset(v_covars) atIndex:6];
            [enc setBytes:&B length:sizeof(B) atIndex:7];
            [enc setBytes:&C length:sizeof(C) atIndex:8];
            [enc setBytes:&N length:sizeof(N) atIndex:9];
            [enc setBytes:&width_u32 length:sizeof(width_u32) atIndex:10];
            [enc setBytes:&height_u32 length:sizeof(height_u32) atIndex:11];
            [enc setBytes:&camera_model_u32 length:sizeof(camera_model_u32) atIndex:12];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(v_means, v_covars);
}

}  // namespace gsplat::metal
