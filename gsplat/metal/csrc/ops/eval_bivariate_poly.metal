// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../metal_math.h"

kernel void eval_bivariate_poly_kernel(
    device const float* x            [[buffer(0)]],
    device const float* y            [[buffer(1)]],
    device const float* poly_coeffs  [[buffer(2)]],
    device       float* result       [[buffer(3)]],
    constant uint& n                 [[buffer(4)]],
    constant uint& order             [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }
    result[id] = eval_bivariate_poly_metal(poly_coeffs, order, x[id], y[id]);
}
