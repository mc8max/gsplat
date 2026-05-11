// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "MetalContext.h"
#include "ops/null/null.h"
#include "ops/quat_scale_to_covar_preci.h"
#include "ops/spherical_harmonics.h"

using namespace gsplat::metal;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_library", [](const std::string& path) {
        MetalContext::instance().load_library(path);
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
}

TORCH_LIBRARY_IMPL(gsplat, MPS, m) {
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
