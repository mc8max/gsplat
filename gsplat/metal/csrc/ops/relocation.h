// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor> relocation_op(
    const at::Tensor& opacities,
    const at::Tensor& scales,
    const at::Tensor& ratios,
    const at::Tensor& binoms,
    int64_t n_max
);

}  // namespace gsplat::metal
