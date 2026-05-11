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

}  // namespace gsplat::metal
