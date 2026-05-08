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
#include "quat_scale_to_covar_preci.h"

namespace gsplat::metal {

namespace {

void check_quats_and_scales(const at::Tensor& quats, const at::Tensor& scales) {
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    TORCH_CHECK(quats.dim() >= 1, "quats must have at least 1 dimension");
    TORCH_CHECK(scales.dim() >= 1, "scales must have at least 1 dimension");
    auto quats_batch = quats.sizes().slice(0, quats.dim() - 1);
    auto scales_batch = scales.sizes().slice(0, scales.dim() - 1);
    TORCH_CHECK(quats_batch == scales_batch, "quats and scales batch dimensions must match");
    TORCH_CHECK(quats.size(-1) == 4, "quats last dimension must be 4");
    TORCH_CHECK(scales.size(-1) == 3, "scales last dimension must be 3");
}

at::DimVector output_shape(const at::Tensor& quats, bool triu) {
    at::DimVector shape(quats.sizes().slice(0, quats.dim() - 1));
    if (triu) {
        shape.push_back(6);
    } else {
        shape.push_back(3);
        shape.push_back(3);
    }
    return shape;
}

void set_optional_tensor_buffer(
    id<MTLComputeCommandEncoder> enc,
    const at::Tensor& tensor,
    NSUInteger index
) {
    if (tensor.defined()) {
        [enc setBuffer:to_mtl_buffer(tensor) offset:byte_offset(tensor) atIndex:index];
    } else {
        [enc setBuffer:nil offset:0 atIndex:index];
    }
}

void check_optional_grad(
    const c10::optional<at::Tensor>& grad,
    const char* name,
    const at::Tensor& quats,
    bool triu
) {
    if (!grad.has_value()) {
        return;
    }
    const at::Tensor& tensor = grad.value();
    check_mps_float32(tensor, name);
    at::DimVector expected = output_shape(quats, triu);
    TORCH_CHECK(
        tensor.sizes().equals(expected),
        name,
        " shape mismatch: expected ",
        expected,
        ", got ",
        tensor.sizes());
}

}  // namespace

std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>> quat_scale_to_covar_preci_fwd_op(
    const at::Tensor& quats,
    const at::Tensor& scales,
    bool compute_covar,
    bool compute_preci,
    bool triu
) {
    check_quats_and_scales(quats, scales);

    at::DimVector shape = output_shape(quats, triu);
    auto opt = quats.options();
    c10::optional<at::Tensor> covars =
        compute_covar ? c10::optional<at::Tensor>(at::empty(shape, opt)) : c10::nullopt;
    c10::optional<at::Tensor> precis =
        compute_preci ? c10::optional<at::Tensor>(at::empty(shape, opt)) : c10::nullopt;

    const uint32_t n = static_cast<uint32_t>(quats.numel() / 4);
    if (n == 0) {
        return std::make_tuple(covars, precis);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso =
        ctx.pipeline("quat_scale_to_covar_preci_fwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t triu_u32 = triu ? 1u : 0u;
    const uint32_t compute_covar_u32 = compute_covar ? 1u : 0u;
    const uint32_t compute_preci_u32 = compute_preci ? 1u : 0u;

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:0];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:1];
            set_optional_tensor_buffer(enc, covars.value_or(at::Tensor{}), 2);
            set_optional_tensor_buffer(enc, precis.value_or(at::Tensor{}), 3);
            [enc setBytes:&n length:sizeof(n) atIndex:4];
            [enc setBytes:&triu_u32 length:sizeof(triu_u32) atIndex:5];
            [enc setBytes:&compute_covar_u32 length:sizeof(compute_covar_u32) atIndex:6];
            [enc setBytes:&compute_preci_u32 length:sizeof(compute_preci_u32) atIndex:7];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(covars, precis);
}

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd_op(
    const at::Tensor& quats,
    const at::Tensor& scales,
    bool triu,
    const c10::optional<at::Tensor>& v_covars,
    const c10::optional<at::Tensor>& v_precis
) {
    check_quats_and_scales(quats, scales);
    check_optional_grad(v_covars, "v_covars", quats, triu);
    check_optional_grad(v_precis, "v_precis", quats, triu);

    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);

    const uint32_t n = static_cast<uint32_t>(quats.numel() / 4);
    if (n == 0 || (!v_covars.has_value() && !v_precis.has_value())) {
        return std::make_tuple(v_quats, v_scales);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso =
        ctx.pipeline("quat_scale_to_covar_preci_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t triu_u32 = triu ? 1u : 0u;
    const uint32_t has_v_covars = v_covars.has_value() ? 1u : 0u;
    const uint32_t has_v_precis = v_precis.has_value() ? 1u : 0u;

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:0];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:1];
            set_optional_tensor_buffer(enc, v_covars.has_value() ? v_covars.value() : at::Tensor(), 2);
            set_optional_tensor_buffer(enc, v_precis.has_value() ? v_precis.value() : at::Tensor(), 3);
            [enc setBuffer:to_mtl_buffer(v_quats) offset:byte_offset(v_quats) atIndex:4];
            [enc setBuffer:to_mtl_buffer(v_scales) offset:byte_offset(v_scales) atIndex:5];
            [enc setBytes:&n length:sizeof(n) atIndex:6];
            [enc setBytes:&triu_u32 length:sizeof(triu_u32) atIndex:7];
            [enc setBytes:&has_v_covars length:sizeof(has_v_covars) atIndex:8];
            [enc setBytes:&has_v_precis length:sizeof(has_v_precis) atIndex:9];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(v_quats, v_scales);
}

}  // namespace gsplat::metal
