// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>> quat_scale_to_covar_preci_fwd_op(
    const at::Tensor& quats,
    const at::Tensor& scales,
    bool compute_covar,
    bool compute_preci,
    bool triu
);

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd_op(
    const at::Tensor& quats,
    const at::Tensor& scales,
    bool triu,
    const c10::optional<at::Tensor>& v_covars,
    const c10::optional<at::Tensor>& v_precis
);

}  // namespace gsplat::metal
