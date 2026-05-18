// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// 8-bit LSD radix sort kernels for int64 key / int32 value pairs.
//
// The sort is implemented as two kernels called 8 times (once per 8-bit radix
// pass over the 64-bit key):
//
//   1. radix_histogram_kernel  — count occurrences of each 8-bit digit in
//                                each block of RADIX_BLOCK_SIZE elements.
//   2. radix_scatter_kernel    — scatter each element to its sorted position
//                                using precomputed per-(block, digit) offsets.
//
// Between the two kernel calls the host computes the prefix sums over the
// per-block histograms (256 × num_blocks additions — trivially fast on CPU).
//
// Design: one GPU thread per block of RADIX_BLOCK_SIZE elements.  Within
// each block elements are processed sequentially, which gives a stable sort
// (equal keys preserve their original relative order).

#include <metal_stdlib>
using namespace metal;

// Number of consecutive elements each GPU thread is responsible for.
constant uint kRadixBlockSize = 512u;
// Number of 8-bit digit buckets.
constant uint kRadixBuckets   = 256u;

// ---------------------------------------------------------------------------
// Kernel 1 — per-block histogram
// ---------------------------------------------------------------------------
// Launched with num_blocks = ceil(n / kRadixBlockSize) threads.
// Each thread iterates over its block and fills 256 local counters, then
// writes them to hist_out[block_id × 256 + digit].

kernel void radix_histogram_kernel(
    device const ulong* keys     [[buffer(0)]],
    device       uint*  hist_out [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant uint&      shift    [[buffer(3)]],   // 0, 8, 16, 24, 32, 40, 48, 56
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * kRadixBlockSize;
    if (start >= n) return;
    const uint end = min(start + kRadixBlockSize, n);

    // Private histogram — Metal places small fixed-size arrays in registers.
    uint local_hist[256];
    for (uint d = 0u; d < kRadixBuckets; ++d) local_hist[d] = 0u;

    for (uint i = start; i < end; ++i) {
        const uint digit = uint((keys[i] >> shift) & 0xFFul);
        ++local_hist[digit];
    }

    device uint* dst = hist_out + block_id * kRadixBuckets;
    for (uint d = 0u; d < kRadixBuckets; ++d) dst[d] = local_hist[d];
}

// ---------------------------------------------------------------------------
// Kernel 2 — per-block stable scatter
// ---------------------------------------------------------------------------
// Launched with the same num_blocks threads.
// block_offsets[block_id × 256 + digit] = the global write position for the
// first element of this block that has this digit.  These offsets are computed
// by the host between the two kernel calls.

kernel void radix_scatter_kernel(
    device const ulong* keys_in       [[buffer(0)]],
    device const int*   vals_in       [[buffer(1)]],
    device       ulong* keys_out      [[buffer(2)]],
    device       int*   vals_out      [[buffer(3)]],
    device const uint*  block_offsets [[buffer(4)]],
    constant uint&      n             [[buffer(5)]],
    constant uint&      shift         [[buffer(6)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * kRadixBlockSize;
    if (start >= n) return;
    const uint end = min(start + kRadixBlockSize, n);

    // Load base write positions for this block into private memory.
    const device uint* my_offsets = block_offsets + block_id * kRadixBuckets;
    uint local_pos[256];
    for (uint d = 0u; d < kRadixBuckets; ++d) local_pos[d] = my_offsets[d];

    // Sequential scatter — preserves input order within each digit bucket
    // (stability guarantee required by LSD correctness).
    for (uint i = start; i < end; ++i) {
        const uint digit = uint((keys_in[i] >> shift) & 0xFFul);
        const uint pos   = local_pos[digit]++;
        keys_out[pos] = keys_in[i];
        vals_out[pos] = vals_in[i];
    }
}
