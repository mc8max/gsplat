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
#include "eval_bivariate_poly.h"

namespace gsplat::metal {

namespace {

constexpr int64_t kMaxBivariateOrder = 5;

int64_t coeff_count_for_order(int64_t order) {
    return (order + 1) * (order + 2) / 2;
}

void check_eval_bivariate_poly_inputs(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& poly_coeffs,
    int64_t order
) {
    check_mps_float32(x, "x");
    check_mps_float32(y, "y");
    check_mps_float32(poly_coeffs, "poly_coeffs");
    TORCH_CHECK(x.sizes().equals(y.sizes()), "x and y must have the same shape");
    TORCH_CHECK(
        order >= 0 && order <= kMaxBivariateOrder,
        "order must be in [0, ",
        kMaxBivariateOrder,
        "]");
    const int64_t expected = coeff_count_for_order(order);
    TORCH_CHECK(
        poly_coeffs.numel() == expected,
        "poly_coeffs must have ",
        expected,
        " coefficients for order ",
        order,
        ", got ",
        poly_coeffs.numel());
}

}  // namespace

at::Tensor eval_bivariate_poly_op(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& poly_coeffs,
    int64_t order
) {
    check_eval_bivariate_poly_inputs(x, y, poly_coeffs, order);

    at::Tensor result = at::empty_like(x);
    const uint32_t n = static_cast<uint32_t>(x.numel());
    if (n == 0) {
        return result;
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("eval_bivariate_poly_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t order_u32 = static_cast<uint32_t>(order);

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(x) offset:byte_offset(x) atIndex:0];
            [enc setBuffer:to_mtl_buffer(y) offset:byte_offset(y) atIndex:1];
            [enc setBuffer:to_mtl_buffer(poly_coeffs) offset:byte_offset(poly_coeffs) atIndex:2];
            [enc setBuffer:to_mtl_buffer(result) offset:byte_offset(result) atIndex:3];
            [enc setBytes:&n length:sizeof(n) atIndex:4];
            [enc setBytes:&order_u32 length:sizeof(order_u32) atIndex:5];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return result;
}

}  // namespace gsplat::metal
