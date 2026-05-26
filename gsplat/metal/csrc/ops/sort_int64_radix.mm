// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ObjC++ launcher for the 8-pass LSD radix sort.
//
// Per-pass sequence:
//   1. GPU histogram kernel  — fills per-(block,digit) counts into hist_buf
//   2. CPU prefix sum        — converts per-block counts into write positions
//   3. GPU scatter kernel    — places each element at its sorted destination
//
// The CPU step processes (num_blocks × 256) uint32 values, which is trivial
// (~microseconds) even for 10M+ elements.  This avoids the need for a GPU
// parallel scan kernel.
//
// Stability: the scatter kernel processes elements within each block in
// sequential index order, so equal keys are written in the same relative
// order as the input.  LSD stability is required for correctness.

#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "sort_int64.h"

namespace gsplat::metal {

namespace {

constexpr uint32_t kRadixBlockSize = 512u;
constexpr uint32_t kRadixBuckets   = 256u;
constexpr int      kNumPasses      = 8;   // 8 passes × 8 bits = 64-bit key

// ---------------------------------------------------------------------------
// Single histogram pass
// ---------------------------------------------------------------------------
void dispatch_histogram(
    const at::Tensor& keys,   // int64, MPS, shape (n,)
    at::Tensor& hist,         // int32, MPS, shape (num_blocks × 256,)
    uint32_t n,
    uint32_t shift,
    uint32_t num_blocks
) {
    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("radix_histogram_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(keys) offset:byte_offset(keys) atIndex:0];
            [enc setBuffer:to_mtl_buffer(hist) offset:byte_offset(hist) atIndex:1];
            [enc setBytes:&n     length:sizeof(n)     atIndex:2];
            [enc setBytes:&shift length:sizeof(shift) atIndex:3];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, num_blocks);
            [enc dispatchThreads:MTLSizeMake(num_blocks, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    // COMMIT_AND_WAIT so the host can read the histogram immediately.
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
}

// ---------------------------------------------------------------------------
// CPU prefix-sum step
// ---------------------------------------------------------------------------
// Converts per-(block,digit) counts into per-(block,digit) write positions.
//
// block_offsets[b][d] = (global position where block b's digit-d elements
//                        begin writing) = global_prefix[d] + per-digit cumsum over blocks 0..b-1
//
// Layout: hist and block_offsets are both stored row-major as [num_blocks × 256].
void compute_block_offsets(
    const at::Tensor& hist_mps,         // int32, MPS
    at::Tensor& block_offsets_mps,      // int32, MPS
    uint32_t num_blocks
) {
    // Move histogram to CPU for arithmetic.
    at::Tensor hist_cpu = hist_mps.cpu();
    const uint32_t* h = reinterpret_cast<const uint32_t*>(hist_cpu.data_ptr<int32_t>());

    std::vector<uint32_t> offsets(static_cast<size_t>(num_blocks) * kRadixBuckets);

    // For each digit d, compute the running total across blocks.
    // block_offsets[b][d] = (sum of hist[b'][d] for all b' < b)
    //                       + (sum of all hist[b''][d''] for d'' < d, all b'')
    // i.e., = global_prefix[d] + per_digit_cumsum[b][d]

    // Step 1: digit totals (column sums)
    std::vector<uint32_t> digit_total(kRadixBuckets, 0u);
    for (uint32_t b = 0; b < num_blocks; ++b)
        for (uint32_t d = 0; d < kRadixBuckets; ++d)
            digit_total[d] += h[b * kRadixBuckets + d];

    // Step 2: exclusive prefix sum over digits → global_prefix[d]
    std::vector<uint32_t> global_prefix(kRadixBuckets, 0u);
    for (uint32_t d = 1; d < kRadixBuckets; ++d)
        global_prefix[d] = global_prefix[d - 1] + digit_total[d - 1];

    // Step 3: per-digit exclusive prefix sum across blocks.
    // For each digit d: keep a running count across blocks.
    std::vector<uint32_t> per_digit_running(kRadixBuckets, 0u);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        for (uint32_t d = 0; d < kRadixBuckets; ++d) {
            offsets[b * kRadixBuckets + d] = global_prefix[d] + per_digit_running[d];
            per_digit_running[d] += h[b * kRadixBuckets + d];
        }
    }

    // Move back to MPS.
    at::Tensor offsets_cpu = at::from_blob(
        offsets.data(),
        {static_cast<int64_t>(num_blocks * kRadixBuckets)},
        at::TensorOptions().dtype(at::kInt)).clone();
    block_offsets_mps.copy_(offsets_cpu);
}

// ---------------------------------------------------------------------------
// Single scatter pass
// ---------------------------------------------------------------------------
void dispatch_scatter(
    const at::Tensor& keys_in,
    const at::Tensor& vals_in,
    at::Tensor& keys_out,
    at::Tensor& vals_out,
    const at::Tensor& block_offsets,
    uint32_t n,
    uint32_t shift,
    uint32_t num_blocks
) {
    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("radix_scatter_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire MPS stream");

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(keys_in)       offset:byte_offset(keys_in)       atIndex:0];
            [enc setBuffer:to_mtl_buffer(vals_in)       offset:byte_offset(vals_in)       atIndex:1];
            [enc setBuffer:to_mtl_buffer(keys_out)      offset:byte_offset(keys_out)      atIndex:2];
            [enc setBuffer:to_mtl_buffer(vals_out)      offset:byte_offset(vals_out)      atIndex:3];
            [enc setBuffer:to_mtl_buffer(block_offsets) offset:byte_offset(block_offsets) atIndex:4];
            [enc setBytes:&n     length:sizeof(n)     atIndex:5];
            [enc setBytes:&shift length:sizeof(shift) atIndex:6];

            const uint32_t tg = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
            const uint32_t threads = std::min(tg, num_blocks);
            [enc dispatchThreads:MTLSizeMake(num_blocks, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
std::pair<at::Tensor, at::Tensor> radix_sort(
    const at::Tensor& isect_ids,
    const at::Tensor& flatten_ids,
    bool /*stable*/   // radix sort is inherently stable; parameter kept for API symmetry
) {
    TORCH_CHECK(isect_ids.dim() == 1,  "radix_sort: isect_ids must be 1-D");
    TORCH_CHECK(flatten_ids.dim() == 1, "radix_sort: flatten_ids must be 1-D");
    TORCH_CHECK(isect_ids.is_mps(),    "radix_sort: isect_ids must be on MPS");
    TORCH_CHECK(flatten_ids.is_mps(),  "radix_sort: flatten_ids must be on MPS");
    TORCH_CHECK(isect_ids.scalar_type() == at::kLong, "radix_sort: isect_ids must be int64");
    TORCH_CHECK(flatten_ids.scalar_type() == at::kInt, "radix_sort: flatten_ids must be int32");
    TORCH_CHECK(
        isect_ids.numel() == flatten_ids.numel(),
        "radix_sort: isect_ids and flatten_ids must have the same length");

    const int64_t n64 = isect_ids.numel();
    if (n64 == 0) return {isect_ids, flatten_ids};

    TORCH_CHECK(
        n64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "radix_sort: n must fit in uint32_t");
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t num_blocks = (n + kRadixBlockSize - 1) / kRadixBlockSize;

    // Ping-pong buffers.  After 8 passes (even number) the result is in a/a.
    at::Tensor keys_a = isect_ids.contiguous().clone();
    at::Tensor vals_a = flatten_ids.contiguous().clone();
    at::Tensor keys_b = at::empty_like(keys_a);
    at::Tensor vals_b = at::empty_like(vals_a);

    // Per-(block, digit) histogram and write-position buffers.
    const int64_t buf_size = static_cast<int64_t>(num_blocks) * static_cast<int64_t>(kRadixBuckets);
    // Use int32 storage; uint32 values cast via reinterpret in compute_block_offsets.
    at::Tensor hist_buf    = at::empty({buf_size}, isect_ids.options().dtype(at::kInt));
    at::Tensor offsets_buf = at::empty({buf_size}, isect_ids.options().dtype(at::kInt));

    for (int pass = 0; pass < kNumPasses; ++pass) {
        const uint32_t shift = static_cast<uint32_t>(pass * 8);

        at::Tensor& keys_in  = (pass % 2 == 0) ? keys_a : keys_b;
        at::Tensor& vals_in  = (pass % 2 == 0) ? vals_a : vals_b;
        at::Tensor& keys_out = (pass % 2 == 0) ? keys_b : keys_a;
        at::Tensor& vals_out = (pass % 2 == 0) ? vals_b : vals_a;

        dispatch_histogram(keys_in, hist_buf, n, shift, num_blocks);
        compute_block_offsets(hist_buf, offsets_buf, num_blocks);
        dispatch_scatter(keys_in, vals_in, keys_out, vals_out,
                         offsets_buf, n, shift, num_blocks);
    }

    // 8 passes: result is back in keys_a / vals_a.
    return {keys_a, vals_a};
}

}  // namespace gsplat::metal
