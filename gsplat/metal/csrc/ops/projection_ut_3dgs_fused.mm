// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Metal/Metal.h>

#include <tuple>

#include <torch/extension.h>

#if TORCH_VERSION_MAJOR < 2
#  error "gsplat Metal backend requires PyTorch >= 2.0 (MPS support unavailable)"
#endif
#include <ATen/mps/MPSStream.h>

#include "MetalContext.h"
#include "MPSTensor.h"
#include "projection_ut_3dgs_fused.h"

namespace gsplat::metal {

namespace {

constexpr int64_t kCameraModelPinhole = 0;
constexpr int64_t kCameraModelFisheye = 2;
constexpr int64_t kCameraModelFTheta = 3;
constexpr int64_t kCameraModelLidar = 4;
constexpr int64_t kMaxBivariateOrder = 5;
constexpr int64_t kRollingShutterTopToBottom = 0;
constexpr int64_t kRollingShutterLeftToRight = 1;
constexpr int64_t kRollingShutterBottomToTop = 2;
constexpr int64_t kRollingShutterRightToLeft = 3;
constexpr int64_t kRollingShutterGlobal = 4;

struct ProjectionUT3DGSFusedParams {
    uint32_t B;
    uint32_t C;
    uint32_t N;
    uint32_t image_width;
    uint32_t image_height;
    float eps2d;
    float near_plane;
    float far_plane;
    float radius_clip;
    uint32_t global_z_order;
    float ut_alpha;
    float ut_beta;
    float ut_kappa;
    float ut_in_image_margin_factor;
    uint32_t has_opacities;
    uint32_t has_compensations;
    uint32_t ut_require_all_sigma_points_valid;
    uint32_t camera_model;
    uint32_t has_external_distortion;
    uint32_t has_viewmats_rs;
    uint32_t has_pose_end;
    uint32_t rolling_shutter;
    uint32_t ftheta_reference_poly;
    uint32_t external_h_order;
    uint32_t external_v_order;
    float lidar_fov_horiz_start;
    float lidar_fov_horiz_span;
    float lidar_fov_vert_start;
    float lidar_fov_vert_span;
    float lidar_fov_eps;
    uint32_t lidar_spinning_direction;
};

at::DimVector make_shape(const at::Tensor& means, const at::Tensor& viewmats, int64_t tail0) {
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2));
    shape.push_back(viewmats.size(-3));
    shape.push_back(means.size(-2));
    if (tail0 > 0) {
        shape.push_back(tail0);
    }
    return shape;
}

at::DimVector make_camera_param_shape(const at::Tensor& viewmats, int64_t tail0) {
    at::DimVector shape(viewmats.sizes().slice(0, viewmats.dim() - 3));
    shape.push_back(viewmats.size(-3));
    if (tail0 > 0) {
        shape.push_back(tail0);
    }
    return shape;
}

int64_t coeff_count_for_order(int64_t order) {
    return (order + 1) * (order + 2) / 2;
}

int64_t order_for_coeff_count(int64_t coeff_count, const char* name) {
    TORCH_CHECK(coeff_count > 0, name, " must have a positive number of coefficients");
    for (int64_t order = 0; order <= kMaxBivariateOrder; ++order) {
        if (coeff_count == coeff_count_for_order(order)) {
            return order;
        }
    }
    TORCH_CHECK(
        false,
        name,
        " must have a valid triangular coefficient count for order in [0, ",
        kMaxBivariateOrder,
        "], got ",
        coeff_count);
}

void validate_phase1_inputs(
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
    int64_t camera_model,
    double ut_alpha,
    double ut_beta,
    double ut_kappa,
    double ut_in_image_margin_factor,
    int64_t rolling_shutter,
    int64_t ftheta_reference_poly,
    int64_t external_h_order,
    int64_t external_v_order
) {
    check_mps_float32(means, "means");
    check_mps_float32(quats, "quats");
    check_mps_float32(scales, "scales");
    check_mps_float32(viewmats, "viewmats");
    check_mps_float32(pose_start, "pose_start");
    check_mps_float32(Ks, "Ks");
    if (opacities.has_value()) {
        check_mps_float32(*opacities, "opacities");
    }
    if (viewmats_rs.has_value()) {
        check_mps_float32(*viewmats_rs, "viewmats_rs");
    }
    if (pose_end.has_value()) {
        check_mps_float32(*pose_end, "pose_end");
    }

    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3]");
    TORCH_CHECK(means.size(-1) == 3, "means last dimension must be 3");
    TORCH_CHECK(quats.sizes().slice(0, quats.dim() - 2) == means.sizes().slice(0, means.dim() - 2),
        "means and quats batch dimensions must match");
    TORCH_CHECK(quats.size(-2) == means.size(-2) && quats.size(-1) == 4, "quats must have shape [..., N, 4]");
    TORCH_CHECK(scales.sizes().slice(0, scales.dim() - 2) == means.sizes().slice(0, means.dim() - 2),
        "means and scales batch dimensions must match");
    TORCH_CHECK(scales.size(-2) == means.size(-2) && scales.size(-1) == 3, "scales must have shape [..., N, 3]");
    TORCH_CHECK(viewmats.dim() == means.dim() + 1, "viewmats must have shape [..., C, 4, 4]");
    TORCH_CHECK(Ks.dim() == means.dim() + 1, "Ks must have shape [..., C, 3, 3]");
    TORCH_CHECK(pose_start.dim() == means.dim(), "pose_start must have shape [..., C, 7]");
    TORCH_CHECK(viewmats.size(-2) == 4 && viewmats.size(-1) == 4, "viewmats last two dimensions must be 4x4");
    TORCH_CHECK(Ks.size(-2) == 3 && Ks.size(-1) == 3, "Ks last two dimensions must be 3x3");
    TORCH_CHECK(pose_start.size(-2) == viewmats.size(-3) && pose_start.size(-1) == 7, "pose_start must have shape [..., C, 7]");
    TORCH_CHECK(
        viewmats.sizes().slice(0, viewmats.dim() - 3) == means.sizes().slice(0, means.dim() - 2),
        "means and viewmats batch dimensions must match");
    TORCH_CHECK(
        Ks.sizes().slice(0, Ks.dim() - 3) == means.sizes().slice(0, means.dim() - 2),
        "means and Ks batch dimensions must match");
    TORCH_CHECK(
        pose_start.sizes().slice(0, pose_start.dim() - 2) == means.sizes().slice(0, means.dim() - 2),
        "means and pose_start batch dimensions must match");
    TORCH_CHECK(viewmats.size(-3) == Ks.size(-3), "viewmats and Ks camera dimension must match");
    TORCH_CHECK(image_width >= 0, "image_width must be non-negative");
    TORCH_CHECK(image_height >= 0, "image_height must be non-negative");
    TORCH_CHECK(
        camera_model == kCameraModelPinhole || camera_model == kCameraModelFisheye || camera_model == kCameraModelFTheta ||
            camera_model == kCameraModelLidar,
        "Metal UT projection only supports pinhole(0), fisheye(2), ftheta(3), and lidar(4) camera models"
    );
    TORCH_CHECK(ut_alpha > 0.0, "UT alpha must be positive");
    TORCH_CHECK(3.0 + ut_kappa > 0.0, "UT parameters must satisfy 3 + kappa > 0");
    TORCH_CHECK(ut_in_image_margin_factor >= 0.0, "UT in_image_margin_factor must be non-negative");
    TORCH_CHECK(
        rolling_shutter == kRollingShutterTopToBottom ||
        rolling_shutter == kRollingShutterLeftToRight ||
        rolling_shutter == kRollingShutterBottomToTop ||
        rolling_shutter == kRollingShutterRightToLeft ||
        rolling_shutter == kRollingShutterGlobal,
        "rolling_shutter must be one of {0,1,2,3,4}");
    (void)ut_beta;
    if (rolling_shutter == kRollingShutterGlobal) {
        TORCH_CHECK(!viewmats_rs.has_value(), "viewmats_rs must be omitted for global shutter");
        TORCH_CHECK(!pose_end.has_value(), "pose_end must be omitted for global shutter");
    } else {
        TORCH_CHECK(viewmats_rs.has_value(), "viewmats_rs is required when rolling_shutter is not GLOBAL");
        TORCH_CHECK(pose_end.has_value(), "pose_end is required when rolling_shutter is not GLOBAL");
        TORCH_CHECK(viewmats_rs->sizes() == viewmats.sizes(), "viewmats_rs must match viewmats shape");
        TORCH_CHECK(pose_end->sizes() == pose_start.sizes(), "pose_end must match pose_start shape");
    }
    if (opacities.has_value()) {
        TORCH_CHECK(
            opacities->sizes().slice(0, opacities->dim() - 1) == means.sizes().slice(0, means.dim() - 2) &&
                opacities->size(-1) == means.size(-2),
            "opacities must have shape [..., N]");
    }
    if (radial_coeffs.has_value()) {
        check_mps_float32(*radial_coeffs, "radial_coeffs");
        TORCH_CHECK(
            radial_coeffs->sizes().slice(0, radial_coeffs->dim() - 2) == viewmats.sizes().slice(0, viewmats.dim() - 3) &&
                radial_coeffs->size(-2) == viewmats.size(-3),
            "radial_coeffs must have shape [..., C, K]");
        const int64_t expected_last = camera_model == kCameraModelFisheye ? 4 : 6;
        TORCH_CHECK(
            radial_coeffs->size(-1) == expected_last,
            "radial_coeffs last dimension must be ", expected_last);
    }
    if (camera_model == kCameraModelLidar) {
        TORCH_CHECK(!radial_coeffs.has_value(), "lidar does not support radial_coeffs");
        TORCH_CHECK(!tangential_coeffs.has_value(), "lidar does not support tangential_coeffs");
        TORCH_CHECK(!thin_prism_coeffs.has_value(), "lidar does not support thin_prism_coeffs");
        TORCH_CHECK(!fisheye_max_angle.has_value(), "fisheye_max_angle is only valid for fisheye camera model");
        TORCH_CHECK(!ftheta_pixeldist_to_angle_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_angle_to_pixeldist_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_dreference_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_linear_cde.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_max_angle.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!external_h_poly.has_value() && !external_v_poly.has_value(), "lidar does not support external distortion");
        TORCH_CHECK(rolling_shutter == kRollingShutterGlobal, "lidar only supports global shutter");
        TORCH_CHECK(
            lidar_spinning_direction == 0 || lidar_spinning_direction == 1,
            "lidar_spinning_direction must be 0 (CLOCKWISE) or 1 (COUNTER_CLOCKWISE)");
        TORCH_CHECK(lidar_fov_horiz_span >= 0.0 && lidar_fov_vert_span >= 0.0, "lidar FOV spans must be non-negative");
        TORCH_CHECK(lidar_fov_eps >= 0.0, "lidar_fov_eps must be non-negative");
    } else if (camera_model == kCameraModelFisheye) {
        TORCH_CHECK(!tangential_coeffs.has_value(), "fisheye does not support tangential_coeffs");
        TORCH_CHECK(!thin_prism_coeffs.has_value(), "fisheye does not support thin_prism_coeffs");
        TORCH_CHECK(!ftheta_pixeldist_to_angle_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_angle_to_pixeldist_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_dreference_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_linear_cde.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_max_angle.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(fisheye_max_angle.has_value(), "fisheye_max_angle is required for fisheye UT projection");
        check_mps_float32(*fisheye_max_angle, "fisheye_max_angle");
        TORCH_CHECK(
            fisheye_max_angle->sizes() == viewmats.sizes().slice(0, viewmats.dim() - 2),
            "fisheye_max_angle must have shape [..., C]");
    } else if (camera_model == kCameraModelPinhole) {
        TORCH_CHECK(!fisheye_max_angle.has_value(), "fisheye_max_angle is only valid for fisheye camera model");
        TORCH_CHECK(!ftheta_pixeldist_to_angle_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_angle_to_pixeldist_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_dreference_poly.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_linear_cde.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        TORCH_CHECK(!ftheta_max_angle.has_value(), "ftheta parameters are only valid for the ftheta camera model");
        if (tangential_coeffs.has_value()) {
            check_mps_float32(*tangential_coeffs, "tangential_coeffs");
            TORCH_CHECK(
                tangential_coeffs->sizes() == at::IntArrayRef(make_camera_param_shape(viewmats, 2)),
                "tangential_coeffs must have shape [..., C, 2]");
        }
        if (thin_prism_coeffs.has_value()) {
            check_mps_float32(*thin_prism_coeffs, "thin_prism_coeffs");
            TORCH_CHECK(
                thin_prism_coeffs->sizes() == at::IntArrayRef(make_camera_param_shape(viewmats, 4)),
                "thin_prism_coeffs must have shape [..., C, 4]");
        }
    } else {
        TORCH_CHECK(!radial_coeffs.has_value(), "ftheta does not support radial_coeffs");
        TORCH_CHECK(!tangential_coeffs.has_value(), "ftheta does not support tangential_coeffs");
        TORCH_CHECK(!thin_prism_coeffs.has_value(), "ftheta does not support thin_prism_coeffs");
        TORCH_CHECK(!fisheye_max_angle.has_value(), "fisheye_max_angle is only valid for fisheye camera model");
        TORCH_CHECK(ftheta_pixeldist_to_angle_poly.has_value(), "ftheta_pixeldist_to_angle_poly is required for ftheta UT projection");
        TORCH_CHECK(ftheta_angle_to_pixeldist_poly.has_value(), "ftheta_angle_to_pixeldist_poly is required for ftheta UT projection");
        TORCH_CHECK(ftheta_dreference_poly.has_value(), "ftheta_dreference_poly is required for ftheta UT projection");
        TORCH_CHECK(ftheta_linear_cde.has_value(), "ftheta_linear_cde is required for ftheta UT projection");
        TORCH_CHECK(ftheta_max_angle.has_value(), "ftheta_max_angle is required for ftheta UT projection");
        check_mps_float32(*ftheta_pixeldist_to_angle_poly, "ftheta_pixeldist_to_angle_poly");
        check_mps_float32(*ftheta_angle_to_pixeldist_poly, "ftheta_angle_to_pixeldist_poly");
        check_mps_float32(*ftheta_dreference_poly, "ftheta_dreference_poly");
        check_mps_float32(*ftheta_linear_cde, "ftheta_linear_cde");
        check_mps_float32(*ftheta_max_angle, "ftheta_max_angle");
        TORCH_CHECK(ftheta_pixeldist_to_angle_poly->dim() == 1 && ftheta_pixeldist_to_angle_poly->size(0) == 6,
            "ftheta_pixeldist_to_angle_poly must have shape [6]");
        TORCH_CHECK(ftheta_angle_to_pixeldist_poly->dim() == 1 && ftheta_angle_to_pixeldist_poly->size(0) == 6,
            "ftheta_angle_to_pixeldist_poly must have shape [6]");
        TORCH_CHECK(ftheta_dreference_poly->dim() == 1 && ftheta_dreference_poly->size(0) == 5,
            "ftheta_dreference_poly must have shape [5]");
        TORCH_CHECK(ftheta_linear_cde->dim() == 1 && ftheta_linear_cde->size(0) == 3,
            "ftheta_linear_cde must have shape [3]");
        TORCH_CHECK(ftheta_max_angle->dim() == 1 && ftheta_max_angle->size(0) == 1,
            "ftheta_max_angle must have shape [1]");
        TORCH_CHECK(
            ftheta_reference_poly == 0 || ftheta_reference_poly == 1,
            "ftheta_reference_poly must be 0 (PIXELDIST_TO_ANGLE) or 1 (ANGLE_TO_PIXELDIST)");
    }
    if (external_h_poly.has_value() || external_v_poly.has_value()) {
        TORCH_CHECK(external_h_poly.has_value() && external_v_poly.has_value(),
            "external_h_poly and external_v_poly must be provided together");
        check_mps_float32(*external_h_poly, "external_h_poly");
        check_mps_float32(*external_v_poly, "external_v_poly");
        TORCH_CHECK(external_h_poly->dim() == 1, "external_h_poly must be 1D");
        TORCH_CHECK(external_v_poly->dim() == 1, "external_v_poly must be 1D");
        TORCH_CHECK(external_h_poly->numel() == coeff_count_for_order(external_h_order),
            "external_h_poly has incompatible coefficient count for external_h_order");
        TORCH_CHECK(external_v_poly->numel() == coeff_count_for_order(external_v_order),
            "external_v_poly has incompatible coefficient count for external_v_order");
    }
}

}  // namespace

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
) {
    validate_phase1_inputs(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        viewmats_rs,
        pose_start,
        pose_end,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        fisheye_max_angle,
        ftheta_pixeldist_to_angle_poly,
        ftheta_angle_to_pixeldist_poly,
        ftheta_dreference_poly,
        ftheta_linear_cde,
        ftheta_max_angle,
        external_h_poly,
        external_v_poly,
        lidar_fov_horiz_start,
        lidar_fov_horiz_span,
        lidar_fov_vert_start,
        lidar_fov_vert_span,
        lidar_fov_eps,
        lidar_spinning_direction,
        image_width,
        image_height,
        camera_model,
        ut_alpha,
        ut_beta,
        ut_kappa,
        ut_in_image_margin_factor,
        rolling_shutter,
        ftheta_reference_poly,
        external_h_order,
        external_v_order
    );

    at::Tensor radii = at::zeros(make_shape(means, viewmats, 2), means.options().dtype(at::kInt));
    at::Tensor means2d = at::zeros(make_shape(means, viewmats, 2), means.options());
    at::Tensor depths = at::zeros(make_shape(means, viewmats, -1), means.options());
    at::Tensor conics = at::zeros(make_shape(means, viewmats, 3), means.options());
    at::Tensor compensations = calc_compensations
        ? at::zeros(make_shape(means, viewmats, -1), means.options())
        : at::empty({0}, means.options());

    const uint32_t B = static_cast<uint32_t>(means.numel() / (means.size(-2) * 3));
    const uint32_t C = static_cast<uint32_t>(viewmats.size(-3));
    const uint32_t N = static_cast<uint32_t>(means.size(-2));
    const uint32_t n = B * C * N;
    if (n == 0u) {
        return std::make_tuple(radii, means2d, depths, conics, compensations);
    }

    auto& ctx = MetalContext::instance();
    id<MTLComputePipelineState> pso = ctx.pipeline("projection_ut_3dgs_fused_kernel");
    auto* mps_stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(mps_stream != nullptr, "Failed to acquire current MPS stream");

    const ProjectionUT3DGSFusedParams params = {
        B,
        C,
        N,
        static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height),
        static_cast<float>(eps2d),
        static_cast<float>(near_plane),
        static_cast<float>(far_plane),
        static_cast<float>(radius_clip),
        global_z_order ? 1u : 0u,
        static_cast<float>(ut_alpha),
        static_cast<float>(ut_beta),
        static_cast<float>(ut_kappa),
        static_cast<float>(ut_in_image_margin_factor),
        opacities.has_value() ? 1u : 0u,
        calc_compensations ? 1u : 0u,
        ut_require_all_sigma_points_valid ? 1u : 0u,
        static_cast<uint32_t>(camera_model),
        external_h_poly.has_value() ? 1u : 0u,
        viewmats_rs.has_value() ? 1u : 0u,
        pose_end.has_value() ? 1u : 0u,
        static_cast<uint32_t>(rolling_shutter),
        static_cast<uint32_t>(ftheta_reference_poly),
        static_cast<uint32_t>(external_h_order),
        static_cast<uint32_t>(external_v_order),
        static_cast<float>(lidar_fov_horiz_start),
        static_cast<float>(lidar_fov_horiz_span),
        static_cast<float>(lidar_fov_vert_start),
        static_cast<float>(lidar_fov_vert_span),
        static_cast<float>(lidar_fov_eps),
        static_cast<uint32_t>(lidar_spinning_direction),
    };

    at::mps::dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = mps_stream->commandEncoder();
            TORCH_CHECK(enc != nil, "Failed to create MPS compute encoder");

            [enc setComputePipelineState:pso];
            [enc setBuffer:to_mtl_buffer(means) offset:byte_offset(means) atIndex:0];
            [enc setBuffer:to_mtl_buffer(quats) offset:byte_offset(quats) atIndex:1];
            [enc setBuffer:to_mtl_buffer(scales) offset:byte_offset(scales) atIndex:2];
            set_optional_tensor_buffer(enc, opacities.value_or(at::Tensor{}), 3);
            [enc setBuffer:to_mtl_buffer(viewmats) offset:byte_offset(viewmats) atIndex:4];
            set_optional_tensor_buffer(enc, viewmats_rs.value_or(at::Tensor{}), 5);
            [enc setBuffer:to_mtl_buffer(pose_start) offset:byte_offset(pose_start) atIndex:6];
            set_optional_tensor_buffer(enc, pose_end.value_or(at::Tensor{}), 7);
            [enc setBuffer:to_mtl_buffer(Ks) offset:byte_offset(Ks) atIndex:8];
            set_optional_tensor_buffer(enc, radial_coeffs.value_or(at::Tensor{}), 9);
            set_optional_tensor_buffer(enc, tangential_coeffs.value_or(at::Tensor{}), 10);
            set_optional_tensor_buffer(enc, thin_prism_coeffs.value_or(at::Tensor{}), 11);
            set_optional_tensor_buffer(enc, fisheye_max_angle.value_or(at::Tensor{}), 12);
            set_optional_tensor_buffer(enc, ftheta_pixeldist_to_angle_poly.value_or(at::Tensor{}), 13);
            set_optional_tensor_buffer(enc, ftheta_angle_to_pixeldist_poly.value_or(at::Tensor{}), 14);
            set_optional_tensor_buffer(enc, ftheta_dreference_poly.value_or(at::Tensor{}), 15);
            set_optional_tensor_buffer(enc, ftheta_linear_cde.value_or(at::Tensor{}), 16);
            set_optional_tensor_buffer(enc, ftheta_max_angle.value_or(at::Tensor{}), 17);
            set_optional_tensor_buffer(enc, external_h_poly.value_or(at::Tensor{}), 18);
            set_optional_tensor_buffer(enc, external_v_poly.value_or(at::Tensor{}), 19);
            [enc setBuffer:to_mtl_buffer(radii) offset:byte_offset(radii) atIndex:20];
            [enc setBuffer:to_mtl_buffer(means2d) offset:byte_offset(means2d) atIndex:21];
            [enc setBuffer:to_mtl_buffer(depths) offset:byte_offset(depths) atIndex:22];
            [enc setBuffer:to_mtl_buffer(conics) offset:byte_offset(conics) atIndex:23];
            set_optional_tensor_buffer(enc, calc_compensations ? compensations : at::Tensor{}, 24);
            [enc setBytes:&params length:sizeof(params) atIndex:25];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
            if (w > 256) {
                w = 256;
            }
            if (w == 0) {
                w = 1;
            }
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
    });
    mps_stream->synchronize(at::mps::SyncType::COMMIT);

    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

}  // namespace gsplat::metal
