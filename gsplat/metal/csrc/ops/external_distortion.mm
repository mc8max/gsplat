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
#include "external_distortion.h"

namespace gsplat::metal {

namespace {

constexpr int64_t kMaxBivariateOrder = 5;

int64_t coeff_count_for_order(int64_t order) {
    return (order + 1) * (order + 2) / 2;
}

int64_t order_for_coeff_count(int64_t coeff_count, const char* name) {
    TORCH_CHECK(coeff_count > 0, name, " must have a positive number of coefficients");
    for (int64_t order = 0; order <= kMaxBivariateOrder; ++order) {
        if (coeff_count == coeff_count_for_order(order)) {
            return order;
        }
    }
    TORCH_CHECK(
        false,
        name,
        " must have a valid triangular coefficient count for order in [0, ",
        kMaxBivariateOrder,
        "], got ",
        coeff_count);
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

void check_poly_coeff_count(const at::Tensor& poly, const char* name) {
    check_mps_float32(poly, name);
    (void)order_for_coeff_count(poly.numel(), name);
}

void check_distort_camera_rays_inputs(
    const at::Tensor& rays,
    const at::Tensor& h_poly,
    const at::Tensor& v_poly,
    const at::Tensor& h_inv_poly,
    const at::Tensor& v_inv_poly
) {
    check_mps_float32(rays, "rays");
    TORCH_CHECK(rays.dim() >= 1, "rays must have at least 1 dimension");
    TORCH_CHECK(rays.size(-1) == 3, "rays last dimension must be 3");
    check_poly_coeff_count(h_poly, "h_poly");
    check_poly_coeff_count(v_poly, "v_poly");
    check_poly_coeff_count(h_inv_poly, "h_inv_poly");
    check_poly_coeff_count(v_inv_poly, "v_inv_poly");
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

at::Tensor distort_camera_rays_op(
    const at::Tensor& rays,
    const at::Tensor& h_poly,
    const at::Tensor& v_poly,
    const at::Tensor& h_inv_poly,
    const at::Tensor& v_inv_poly,
    int64_t reference_poly,
    bool inverse
) {
    check_distort_camera_rays_inputs(rays, h_poly, v_poly, h_inv_poly, v_inv_poly);
    
    // reference_poly is carried by BivariateWindshieldModelParameters for use in                                                                                          
    // the full projection pipeline but has no effect on standalone ray distortion.                                                                                          
    // CUDA's distort_camera_rays_torch_op stores it in params but never reads it
    // during kernel execution — same behaviour here.   
    (void)reference_poly;

    const at::Tensor& selected_h_poly = inverse ? h_inv_poly : h_poly;
    const at::Tensor& selected_v_poly = inverse ? v_inv_poly : v_poly;
    const uint32_t h_order = static_cast<uint32_t>(
        order_for_coeff_count(selected_h_poly.numel(), inverse ? "h_inv_poly" : "h_poly")
    );
    const uint32_t v_order = static_cast<uint32_t>(
        order_for_coeff_count(selected_v_poly.numel(), inverse ? "v_inv_poly" : "v_poly")
    );

    at::Tensor result = at::empty_like(rays);
    const uint32_t n = static_cast<uint32_t>(rays.numel() / 3);
    if (n == 0) {
        return result;
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("distort_camera_rays_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(rays) offset:byte_offset(rays) atIndex:0];
            [enc setBuffer:to_mtl_buffer(selected_h_poly) offset:byte_offset(selected_h_poly) atIndex:1];
            [enc setBuffer:to_mtl_buffer(selected_v_poly) offset:byte_offset(selected_v_poly) atIndex:2];
            [enc setBuffer:to_mtl_buffer(result) offset:byte_offset(result) atIndex:3];
            [enc setBytes:&n length:sizeof(n) atIndex:4];
            [enc setBytes:&h_order length:sizeof(h_order) atIndex:5];
            [enc setBytes:&v_order length:sizeof(v_order) atIndex:6];

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
