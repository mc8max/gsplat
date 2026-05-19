// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "MetalContext.h"
#include "ops/external_distortion.h"
#include "ops/intersect_offset.h"
#include "ops/intersect_tile.h"
#include "ops/null.h"
#include "ops/projection_ewa_3dgs_fused.h"
#include "ops/projection_ewa_simple.h"
#include "ops/quat_scale_to_covar_preci.h"
#include "ops/spherical_harmonics.h"
#include "ops/sort_int64.h"

using namespace gsplat::metal;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_library", [](const std::string& path) {
        MetalContext::instance().load_library(path);
    });
    m.def("mps_lsd_sort", [](const at::Tensor& ids, const at::Tensor& vals) {
        return mps_lsd_sort(ids, vals, true);
    });
    m.def("radix_sort", [](const at::Tensor& ids, const at::Tensor& vals) {
        return radix_sort(ids, vals, true);
    });
}

TORCH_LIBRARY_FRAGMENT(gsplat, m) {
    m.def("metal_null(Tensor self) -> Tensor");
    m.def(
        "metal_quat_scale_to_covar_preci_fwd("
        "Tensor quats, Tensor scales, bool compute_covar, bool compute_preci, bool triu"
        ") -> (Tensor?, Tensor?)");
    m.def(
        "metal_quat_scale_to_covar_preci_bwd("
        "Tensor quats, Tensor scales, bool triu, Tensor? v_covars, Tensor? v_precis"
        ") -> (Tensor, Tensor)");
    m.def(
        "metal_spherical_harmonics_fwd("
        "int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks"
        ") -> Tensor");
    m.def(
        "metal_spherical_harmonics_bwd("
        "int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks,"
        " Tensor v_colors, bool compute_v_dirs"
        ") -> (Tensor, Tensor?)");
    m.def(
        "metal_eval_bivariate_poly("
        "Tensor x, Tensor y, Tensor poly_coeffs, int order"
        ") -> Tensor");
    m.def(
        "metal_distort_camera_rays("
        "Tensor rays, Tensor h_poly, Tensor v_poly, Tensor h_inv_poly, "
        "Tensor v_inv_poly, int reference_poly, bool inverse"
        ") -> Tensor");
    m.def(
        "metal_projection_ewa_3dgs_fused_fwd("
        "Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor? opacities, "
        "Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, "
        "float near_plane, float far_plane, float radius_clip, bool calc_compensations, "
        "int camera_model"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor?)");
    m.def(
        "metal_projection_ewa_3dgs_fused_bwd("
        "Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor viewmats, "
        "Tensor Ks, int image_width, int image_height, float eps2d, int camera_model, "
        "Tensor radii, Tensor conics, Tensor? compensations, Tensor v_means2d, "
        "Tensor v_depths, Tensor v_conics, Tensor? v_compensations, bool viewmats_requires_grad"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "metal_projection_ewa_simple_fwd("
        "Tensor means, Tensor covars, Tensor Ks, int width, int height, int camera_model"
        ") -> (Tensor, Tensor)");
    m.def(
        "metal_projection_ewa_simple_bwd("
        "Tensor means, Tensor covars, Tensor Ks, int width, int height, int camera_model, "
        "Tensor v_means2d, Tensor v_covars2d"
        ") -> (Tensor, Tensor)");
    m.def(
        "metal_intersect_offset("
        "Tensor isect_ids, int I, int tile_width, int tile_height"
        ") -> Tensor");
    m.def(
        "metal_intersect_tile_count("
        "Tensor means2d, Tensor radii, Tensor depths, Tensor? conics, Tensor? opacities, "
        "Tensor? image_ids, Tensor? gaussian_ids, int I, int tile_size, int tile_width, "
        "int tile_height, bool packed, bool segmented"
        ") -> Tensor");
    m.def(
        "metal_intersect_tile_emit("
        "Tensor means2d, Tensor radii, Tensor depths, Tensor? conics, Tensor? opacities, "
        "Tensor? image_ids, Tensor? gaussian_ids, int I, int tile_size, int tile_width, "
        "int tile_height, Tensor cum_tiles_per_gauss, bool packed, bool segmented"
        ") -> (Tensor, Tensor)");
    m.def(
        "metal_intersect_tile("
        "Tensor means2d, Tensor radii, Tensor depths, Tensor? conics, Tensor? opacities, "
        "Tensor? image_ids, Tensor? gaussian_ids, int I, int tile_size, int tile_width, "
        "int tile_height, bool sort, bool packed, bool segmented"
        ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat, MPS, m) {
    m.impl("metal_eval_bivariate_poly", &eval_bivariate_poly_op);
    m.impl("metal_distort_camera_rays", &distort_camera_rays_op);
    m.impl("metal_intersect_offset", &intersect_offset_op);
    m.impl("metal_intersect_tile_count", &intersect_tile_count_op);
    m.impl("metal_intersect_tile_emit", &intersect_tile_emit_op);
    m.impl("metal_intersect_tile", &intersect_tile_op);
    m.impl("metal_projection_ewa_3dgs_fused_fwd", &projection_ewa_3dgs_fused_fwd_op);
    m.impl("metal_projection_ewa_3dgs_fused_bwd", &projection_ewa_3dgs_fused_bwd_op);
    m.impl("metal_projection_ewa_simple_fwd", &projection_ewa_simple_fwd_op);
    m.impl("metal_projection_ewa_simple_bwd", &projection_ewa_simple_bwd_op);
    m.impl("metal_null", &null_op);
    m.impl(
        "metal_quat_scale_to_covar_preci_fwd",
        &quat_scale_to_covar_preci_fwd_op);
    m.impl(
        "metal_quat_scale_to_covar_preci_bwd",
        &quat_scale_to_covar_preci_bwd_op);
    m.impl("metal_spherical_harmonics_fwd", &spherical_harmonics_fwd_op);
    m.impl("metal_spherical_harmonics_bwd", &spherical_harmonics_bwd_op);
}
