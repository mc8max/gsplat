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
#include "spherical_harmonics.h"

namespace gsplat::metal {

namespace {

void check_inputs(
    int64_t degrees_to_use,
    const at::Tensor& dirs,
    const at::Tensor& coeffs,
    const c10::optional<at::Tensor>& masks
) {
    check_mps_float32(dirs, "dirs");
    check_mps_float32(coeffs, "coeffs");
    TORCH_CHECK(degrees_to_use >= 0 && degrees_to_use <= 4, "degrees_to_use must be in [0, 4]");
    TORCH_CHECK(dirs.dim() >= 1, "dirs must have at least 1 dimension");
    TORCH_CHECK(coeffs.dim() >= 2, "coeffs must have at least 2 dimensions");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs last dimension must be 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs last dimension must be 3");
    TORCH_CHECK(
        dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
        "dirs and coeffs batch dimensions must match");
    const int64_t num_bases_needed = (degrees_to_use + 1) * (degrees_to_use + 1);
    TORCH_CHECK(
        coeffs.size(-2) >= num_bases_needed,
        "coeffs must have at least ",
        num_bases_needed,
        " basis functions for degree ",
        degrees_to_use);

    if (masks.has_value()) {
        check_mps_bool(masks.value(), "masks");
        TORCH_CHECK(
            masks.value().sizes().equals(dirs.sizes().slice(0, dirs.dim() - 1)),
            "masks shape must match dirs batch dimensions");
    }
}

}  // namespace

at::Tensor spherical_harmonics_fwd_op(
    int64_t degrees_to_use,
    const at::Tensor& dirs,
    const at::Tensor& coeffs,
    const c10::optional<at::Tensor>& masks
) {
    check_inputs(degrees_to_use, dirs, coeffs, masks);

    at::Tensor colors = at::zeros(dirs.sizes(), dirs.options());
    const uint32_t n = static_cast<uint32_t>(dirs.numel() / 3);
    if (n == 0) {
        return colors;
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("spherical_harmonics_fwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t k = static_cast<uint32_t>(coeffs.size(-2));
    const uint32_t degree_u32 = static_cast<uint32_t>(degrees_to_use);
    const uint32_t n_threads = n * 3u;

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(dirs) offset:byte_offset(dirs) atIndex:0];
            [enc setBuffer:to_mtl_buffer(coeffs) offset:byte_offset(coeffs) atIndex:1];
            set_optional_tensor_buffer(enc, masks.value_or(at::Tensor{}), 2);
            [enc setBuffer:to_mtl_buffer(colors) offset:byte_offset(colors) atIndex:3];
            [enc setBytes:&n length:sizeof(n) atIndex:4];
            [enc setBytes:&k length:sizeof(k) atIndex:5];
            [enc setBytes:&degree_u32 length:sizeof(degree_u32) atIndex:6];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n_threads);
            [enc dispatchThreads:MTLSizeMake(n_threads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return colors;
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> spherical_harmonics_bwd_op(
    int64_t degrees_to_use,
    const at::Tensor& dirs,
    const at::Tensor& coeffs,
    const c10::optional<at::Tensor>& masks,
    const at::Tensor& v_colors,
    bool compute_v_dirs
) {
    check_inputs(degrees_to_use, dirs, coeffs, masks);
    check_mps_float32(v_colors, "v_colors");
    TORCH_CHECK(v_colors.sizes().equals(dirs.sizes()), "v_colors shape must match dirs shape");

    at::Tensor v_coeffs = at::zeros_like(coeffs);
    c10::optional<at::Tensor> v_dirs =
        compute_v_dirs ? c10::optional<at::Tensor>(at::zeros_like(dirs)) : c10::nullopt;

    const uint32_t n = static_cast<uint32_t>(dirs.numel() / 3);
    if (n == 0) {
        return std::make_tuple(v_coeffs, v_dirs);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("spherical_harmonics_bwd_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const uint32_t k = static_cast<uint32_t>(coeffs.size(-2));
    const uint32_t degree_u32 = static_cast<uint32_t>(degrees_to_use);
    const uint32_t compute_v_dirs_u32 = compute_v_dirs ? 1u : 0u;

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(dirs) offset:byte_offset(dirs) atIndex:0];
            [enc setBuffer:to_mtl_buffer(coeffs) offset:byte_offset(coeffs) atIndex:1];
            set_optional_tensor_buffer(enc, masks.value_or(at::Tensor{}), 2);
            [enc setBuffer:to_mtl_buffer(v_colors) offset:byte_offset(v_colors) atIndex:3];
            [enc setBuffer:to_mtl_buffer(v_coeffs) offset:byte_offset(v_coeffs) atIndex:4];
            set_optional_tensor_buffer(enc, v_dirs.value_or(at::Tensor{}), 5);
            [enc setBytes:&n length:sizeof(n) atIndex:6];
            [enc setBytes:&k length:sizeof(k) atIndex:7];
            [enc setBytes:&degree_u32 length:sizeof(degree_u32) atIndex:8];
            [enc setBytes:&compute_v_dirs_u32 length:sizeof(compute_v_dirs_u32) atIndex:9];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(v_coeffs, v_dirs);
}

}  // namespace gsplat::metal
