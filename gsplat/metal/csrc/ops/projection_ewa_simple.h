// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd_op(
    const at::Tensor& means,
    const at::Tensor& covars,
    const at::Tensor& Ks,
    int64_t width,
    int64_t height,
    int64_t camera_model
);

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd_op(
    const at::Tensor& means,
    const at::Tensor& covars,
    const at::Tensor& Ks,
    int64_t width,
    int64_t height,
    int64_t camera_model,
    const at::Tensor& v_means2d,
    const at::Tensor& v_covars2d
);

}  // namespace gsplat::metal
