// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

#include "../metal_math.h"

using namespace metal;

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

kernel void distort_camera_rays_kernel(
    device const packed_float3* rays   [[buffer(0)]],
    device const float* h_poly         [[buffer(1)]],
    device const float* v_poly         [[buffer(2)]],
    device       packed_float3* result [[buffer(3)]],
    constant uint& n                   [[buffer(4)]],
    constant uint& h_order             [[buffer(5)]],
    constant uint& v_order             [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }
    result[id] = distort_camera_ray_metal(float3(rays[id]), h_poly, v_poly, h_order, v_order);
}
