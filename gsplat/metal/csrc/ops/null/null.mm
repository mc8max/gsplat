// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <ATen/mps/MPSStream.h>
#include <algorithm>
#include <torch/extension.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "null.h"

namespace gsplat::metal {

at::Tensor null_op(const at::Tensor& input) {
    check_mps_float32(input, "input");

    auto output = at::empty_like(input);
    const uint32_t n = static_cast<uint32_t>(input.numel());
    if (n == 0) {
        return output;
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("null_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(input) offset:byte_offset(input) atIndex:0];
            [enc setBuffer:to_mtl_buffer(output) offset:byte_offset(output) atIndex:1];
            [enc useResource:to_mtl_buffer(input) usage:MTLResourceUsageRead];
            [enc useResource:to_mtl_buffer(output) usage:MTLResourceUsageWrite];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    return output;
}

}  // namespace gsplat::metal
