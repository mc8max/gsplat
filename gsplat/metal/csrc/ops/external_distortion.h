// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

at::Tensor eval_bivariate_poly_op(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& poly_coeffs,
    int64_t order
);

at::Tensor distort_camera_rays_op(
    const at::Tensor& rays,
    const at::Tensor& h_poly,
    const at::Tensor& v_poly,
    const at::Tensor& h_inv_poly,
    const at::Tensor& v_inv_poly,
    int64_t reference_poly,
    bool inverse
);

}  // namespace gsplat::metal
