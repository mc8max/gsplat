// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>>
projection_ewa_3dgs_packed_fwd_op(
    const at::Tensor& means,
    const c10::optional<at::Tensor>& covars,
    const c10::optional<at::Tensor>& quats,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& opacities,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd_op(
    const at::Tensor& means,
    const c10::optional<at::Tensor>& covars,
    const c10::optional<at::Tensor>& quats,
    const c10::optional<at::Tensor>& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    int64_t camera_model,
    const at::Tensor& batch_ids,
    const at::Tensor& camera_ids,
    const at::Tensor& gaussian_ids,
    const at::Tensor& conics,
    const c10::optional<at::Tensor>& compensations,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_conics,
    const c10::optional<at::Tensor>& v_compensations,
    bool viewmats_requires_grad,
    bool sparse_grad
);

}  // namespace gsplat::metal
