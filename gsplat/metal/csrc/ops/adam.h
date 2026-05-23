// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

void adam_op(
    at::Tensor& param,
    const at::Tensor& param_grad,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq,
    const c10::optional<at::Tensor>& valid,
    double lr,
    double b1,
    double b2,
    double eps
);

}  // namespace gsplat::metal
