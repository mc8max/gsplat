// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "MetalContext.h"
#include "ops/null/null.h"
#include "ops/quat_scale_to_covar_preci.h"

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
}

TORCH_LIBRARY_IMPL(gsplat, MPS, m) {
    m.impl("metal_null", &null_op);
    m.impl(
        "metal_quat_scale_to_covar_preci_fwd",
        &quat_scale_to_covar_preci_fwd_op);
    m.impl(
        "metal_quat_scale_to_covar_preci_bwd",
        &quat_scale_to_covar_preci_bwd_op);
}
