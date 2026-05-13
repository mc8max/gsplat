// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

kernel void intersect_offset_kernel(
    device const ulong* isect_ids   [[buffer(0)]],
    device int*         offsets     [[buffer(1)]],
    constant uint&      n_isects    [[buffer(2)]],
    constant uint&      I           [[buffer(3)]],
    constant uint&      n_tiles     [[buffer(4)]],
    constant uint&      tile_n_bits [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_isects) {
        return;
    }

    ulong isect_id_curr = isect_ids[id] >> 32;
    ulong iid_curr = isect_id_curr >> tile_n_bits;
    ulong tid_curr = isect_id_curr & ((1ul << tile_n_bits) - 1ul);
    uint flat_curr = uint(iid_curr * ulong(n_tiles) + tid_curr);

    if (id == 0u) {
        for (uint i = 0u; i <= flat_curr; ++i) {
            offsets[i] = 0;
        }
    }

    if (id == n_isects - 1u) {
        for (uint i = flat_curr + 1u; i < I * n_tiles; ++i) {
            offsets[i] = int(n_isects);
        }
    }

    if (id > 0u) {
        ulong isect_id_prev = isect_ids[id - 1u] >> 32;
        ulong iid_prev = isect_id_prev >> tile_n_bits;
        ulong tid_prev = isect_id_prev & ((1ul << tile_n_bits) - 1ul);
        uint flat_prev = uint(iid_prev * ulong(n_tiles) + tid_prev);
        if (flat_prev == flat_curr) {
            return;
        }
        for (uint i = flat_prev + 1u; i <= flat_curr; ++i) {
            offsets[i] = int(id);
        }
    }
}
