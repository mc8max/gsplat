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
    at::Tensor>
projection_2dgs_packed_fwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane,
    double radius_clip
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& viewmats,
    const at::Tensor& Ks,
    int64_t image_width,
    int64_t image_height,
    const at::Tensor& batch_ids,
    const at::Tensor& camera_ids,
    const at::Tensor& gaussian_ids,
    const at::Tensor& ray_transforms,
    const at::Tensor& v_means2d,
    const at::Tensor& v_depths,
    const at::Tensor& v_ray_transforms,
    const at::Tensor& v_normals,
    bool viewmats_requires_grad,
    bool sparse_grad
);

}  // namespace gsplat::metal
