// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace gsplat::metal {

struct RasterizeIndicesConfig {
    uint32_t I;
    uint32_t N;
    uint32_t tile_size;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t n_tiles;
    uint32_t total_tiles;
    uint32_t n_isects;
};

}  // namespace gsplat::metal
