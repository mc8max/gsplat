// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

at::Tensor spherical_harmonics_fwd_op(
    int64_t degrees_to_use,
    const at::Tensor& dirs,
    const at::Tensor& coeffs,
    const c10::optional<at::Tensor>& masks
);

std::tuple<at::Tensor, c10::optional<at::Tensor>> spherical_harmonics_bwd_op(
    int64_t degrees_to_use,
    const at::Tensor& dirs,
    const at::Tensor& coeffs,
    const c10::optional<at::Tensor>& masks,
    const at::Tensor& v_colors,
    bool compute_v_dirs
);

}  // namespace gsplat::metal
