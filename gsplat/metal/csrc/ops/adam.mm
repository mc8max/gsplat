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
#include "adam.h"

namespace gsplat::metal {

namespace {

void check_mps_supported_adam_dtype(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf,
        name,
        " must be float32 or float16"
    );
}

}  // namespace

void adam_op(
    at::Tensor& param,
    const at::Tensor& param_grad,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq,
    const c10::optional<at::Tensor>& valid,
    double lr,
    double b1,
    double b2,
    double eps
) {
    check_mps_supported_adam_dtype(param, "param");
    check_mps_supported_adam_dtype(param_grad, "param_grad");
    check_mps_supported_adam_dtype(exp_avg, "exp_avg");
    check_mps_supported_adam_dtype(exp_avg_sq, "exp_avg_sq");

    TORCH_CHECK(param.dim() >= 1, "param must have at least one dimension");
    TORCH_CHECK(
        param.scalar_type() == param_grad.scalar_type(),
        "param and param_grad must have the same dtype"
    );
    TORCH_CHECK(
        param.scalar_type() == exp_avg.scalar_type(),
        "param and exp_avg must have the same dtype"
    );
    TORCH_CHECK(
        param.scalar_type() == exp_avg_sq.scalar_type(),
        "param and exp_avg_sq must have the same dtype"
    );
    TORCH_CHECK(param.sizes() == param_grad.sizes(), "param and param_grad must have the same shape");
    TORCH_CHECK(param.sizes() == exp_avg.sizes(), "param and exp_avg must have the same shape");
    TORCH_CHECK(param.sizes() == exp_avg_sq.sizes(), "param and exp_avg_sq must have the same shape");

    if (valid.has_value()) {
        check_mps_bool(*valid, "valid");
        TORCH_CHECK(valid->dim() == 1, "valid should be 1D tensor");
        TORCH_CHECK(
            valid->size(0) == param.size(0),
            "valid first dimension should match param first dimension"
        );
    }

    const uint32_t n = static_cast<uint32_t>(param.size(0));
    const uint32_t numel = product_i64_to_u32(param.sizes(), "param");
    if (numel == 0) {
        return;
    }
    TORCH_CHECK(n > 0, "param first dimension must be positive when numel > 0");
    const uint32_t d = numel / n;
    TORCH_CHECK(static_cast<uint64_t>(n) * static_cast<uint64_t>(d) == numel, "param shape is not row-major flat");

    const float lr_f = static_cast<float>(lr);
    const float b1_f = static_cast<float>(b1);
    const float b2_f = static_cast<float>(b2);
    const float eps_f = static_cast<float>(eps);

    auto& ctx = MetalContext::instance();
    const char* kernel_name = param.scalar_type() == at::kHalf ? "adam_kernel_half" : "adam_kernel_float";
    id<MTLComputePipelineState> pso = ctx.pipeline(kernel_name);
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(param) offset:byte_offset(param) atIndex:0];
            [enc setBuffer:to_mtl_buffer(param_grad) offset:byte_offset(param_grad) atIndex:1];
            [enc setBuffer:to_mtl_buffer(exp_avg) offset:byte_offset(exp_avg) atIndex:2];
            [enc setBuffer:to_mtl_buffer(exp_avg_sq) offset:byte_offset(exp_avg_sq) atIndex:3];
            if (valid.has_value()) {
                [enc setBuffer:to_mtl_buffer(*valid) offset:byte_offset(*valid) atIndex:4];
            } else {
                [enc setBuffer:nil offset:0 atIndex:4];
            }
            [enc setBytes:&n length:sizeof(n) atIndex:5];
            [enc setBytes:&d length:sizeof(d) atIndex:6];
            [enc setBytes:&lr_f length:sizeof(lr_f) atIndex:7];
            [enc setBytes:&b1_f length:sizeof(b1_f) atIndex:8];
            [enc setBytes:&b2_f length:sizeof(b2_f) atIndex:9];
            [enc setBytes:&eps_f length:sizeof(eps_f) atIndex:10];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, numel);
            [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);
}

}  // namespace gsplat::metal
