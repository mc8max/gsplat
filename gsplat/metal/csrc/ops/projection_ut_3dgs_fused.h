// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gsplat::metal {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ut_3dgs_fused_op(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& opacities,
    const at::Tensor& viewmats,
    const c10::optional<at::Tensor>& viewmats_rs,
    const at::Tensor& pose_start,
    const c10::optional<at::Tensor>& pose_end,
    const at::Tensor& Ks,
    const c10::optional<at::Tensor>& radial_coeffs,
    const c10::optional<at::Tensor>& tangential_coeffs,
    const c10::optional<at::Tensor>& thin_prism_coeffs,
    const c10::optional<at::Tensor>& fisheye_max_angle,
    const c10::optional<at::Tensor>& ftheta_pixeldist_to_angle_poly,
    const c10::optional<at::Tensor>& ftheta_angle_to_pixeldist_poly,
    const c10::optional<at::Tensor>& ftheta_dreference_poly,
    const c10::optional<at::Tensor>& ftheta_linear_cde,
    const c10::optional<at::Tensor>& ftheta_max_angle,
    const c10::optional<at::Tensor>& external_h_poly,
    const c10::optional<at::Tensor>& external_v_poly,
    double lidar_fov_horiz_start,
    double lidar_fov_horiz_span,
    double lidar_fov_vert_start,
    double lidar_fov_vert_span,
    double lidar_fov_eps,
    int64_t lidar_spinning_direction,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model,
    bool global_z_order,
    double ut_alpha,
    double ut_beta,
    double ut_kappa,
    double ut_in_image_margin_factor,
    bool ut_require_all_sigma_points_valid,
    int64_t rolling_shutter,
    int64_t ftheta_reference_poly,
    int64_t external_h_order,
    int64_t external_v_order
);

}  // namespace gsplat::metal
