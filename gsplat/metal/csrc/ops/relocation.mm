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
#include "relocation.h"

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor> relocation_op(
    const at::Tensor& opacities,
    const at::Tensor& scales,
    const at::Tensor& ratios,
    const at::Tensor& binoms,
    int64_t n_max
) {
    check_mps_float32(opacities, "opacities");
    check_mps_float32(scales, "scales");
    check_mps_int32(ratios, "ratios");
    check_mps_float32(binoms, "binoms");

    TORCH_CHECK(opacities.dim() == 1, "opacities must have shape [N]");
    const int64_t n = opacities.size(0);
    TORCH_CHECK(scales.sizes() == at::IntArrayRef({n, 3}), "scales must have shape [N, 3]");
    TORCH_CHECK(ratios.sizes() == at::IntArrayRef({n}), "ratios must have shape [N]");
    TORCH_CHECK(n_max >= 0, "n_max must be non-negative");
    TORCH_CHECK(binoms.dim() == 2, "binoms must be rank 2");
    TORCH_CHECK(binoms.size(0) == n_max, "binoms first dimension must match n_max");
    TORCH_CHECK(binoms.size(1) == n_max, "binoms second dimension must match n_max");

    auto new_opacities = at::empty_like(opacities);
    auto new_scales = at::empty_like(scales);
    if (n == 0) {
        return std::make_tuple(new_opacities, new_scales);
    }

    const uint32_t n_u32 = static_cast<uint32_t>(n);
    const uint32_t n_max_u32 = static_cast<uint32_t>(n_max);

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("relocation_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(opacities) offset:byte_offset(opacities) atIndex:0];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:1];
            [enc setBuffer:to_mtl_buffer(ratios) offset:byte_offset(ratios) atIndex:2];
            [enc setBuffer:to_mtl_buffer(binoms) offset:byte_offset(binoms) atIndex:3];
            [enc setBuffer:to_mtl_buffer(new_opacities) offset:byte_offset(new_opacities) atIndex:4];
            [enc setBuffer:to_mtl_buffer(new_scales) offset:byte_offset(new_scales) atIndex:5];
            [enc setBytes:&n_u32 length:sizeof(n_u32) atIndex:6];
            [enc setBytes:&n_max_u32 length:sizeof(n_max_u32) atIndex:7];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n_u32);
            [enc dispatchThreads:MTLSizeMake(n_u32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(new_opacities, new_scales);
}

}  // namespace gsplat::metal
