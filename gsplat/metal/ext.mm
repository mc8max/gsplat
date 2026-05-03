// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "MetalContext.h"
#include "ops/null/null.h"

using namespace gsplat::metal;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_library", [](const std::string& path) {
        MetalContext::instance().load_library(path);
    });
}

TORCH_LIBRARY_FRAGMENT(gsplat, m) {
    m.def("metal_null(Tensor self) -> Tensor");
}

TORCH_LIBRARY_IMPL(gsplat, MPS, m) {
    m.impl("metal_null", &null_op);
}
