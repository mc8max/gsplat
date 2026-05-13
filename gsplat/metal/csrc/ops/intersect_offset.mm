// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "intersect_offset.h"

namespace gsplat::metal {

namespace {

void check_inputs(
    const at::Tensor& isect_ids,
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
) {
    TORCH_CHECK(isect_ids.defined(), "isect_ids must be defined");
    TORCH_CHECK(isect_ids.is_mps(), "isect_ids must be an MPS tensor");
    TORCH_CHECK(isect_ids.is_contiguous(), "isect_ids must be contiguous");
    TORCH_CHECK(isect_ids.scalar_type() == at::kLong, "isect_ids must be int64");
    TORCH_CHECK(isect_ids.dim() == 1, "isect_ids must be 1D");
    TORCH_CHECK(I >= 0, "I must be non-negative");
    TORCH_CHECK(tile_width > 0, "tile_width must be positive");
    TORCH_CHECK(tile_height > 0, "tile_height must be positive");

    const uint64_t n_tiles_u64 = static_cast<uint64_t>(tile_width) * static_cast<uint64_t>(tile_height);
    TORCH_CHECK(
        n_tiles_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        "tile_width * tile_height must fit in uint32_t");
    TORCH_CHECK(
        n_tiles_u64 < (uint64_t{1} << 31),
        "tile_width * tile_height must be < 2^31 so tile_n_bits < 32");

    const uint64_t total_offsets_u64 = static_cast<uint64_t>(I) * n_tiles_u64;
    TORCH_CHECK(
        total_offsets_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        "I * tile_width * tile_height must fit in uint32_t for kernel loop bounds");
}

}  // namespace

at::Tensor intersect_offset_op(
    const at::Tensor& isect_ids,
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
) {
    check_inputs(isect_ids, I, tile_width, tile_height);

    at::Tensor offsets = at::empty(
        {I, tile_height, tile_width},
        isect_ids.options().dtype(at::kInt)
    );

    const uint32_t n_isects = static_cast<uint32_t>(isect_ids.size(0));
    if (n_isects == 0u) {
        offsets.fill_(0);
        return offsets;
    }

    const uint32_t I_u32 = static_cast<uint32_t>(I);
    const uint32_t n_tiles = static_cast<uint32_t>(tile_width * tile_height);
    const uint32_t tile_n_bits = 32u - static_cast<uint32_t>(__builtin_clz(n_tiles));

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("intersect_offset_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(isect_ids) offset:byte_offset(isect_ids) atIndex:0];
            [enc setBuffer:to_mtl_buffer(offsets) offset:byte_offset(offsets) atIndex:1];
            [enc setBytes:&n_isects length:sizeof(n_isects) atIndex:2];
            [enc setBytes:&I_u32 length:sizeof(I_u32) atIndex:3];
            [enc setBytes:&n_tiles length:sizeof(n_tiles) atIndex:4];
            [enc setBytes:&tile_n_bits length:sizeof(tile_n_bits) atIndex:5];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, n_isects);
            [enc dispatchThreads:MTLSizeMake(n_isects, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return offsets;
}

}  // namespace gsplat::metal
