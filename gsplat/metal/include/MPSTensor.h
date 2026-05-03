// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#import <Metal/Metal.h>

#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>

namespace gsplat::metal {

inline void check_mps_float32(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

inline id<MTLBuffer> to_mtl_buffer(const at::Tensor& t) {
    TORCH_CHECK(t.is_mps(), "Expected MPS tensor");
    return at::native::mps::getMTLBufferStorage(t);
}

inline NSUInteger byte_offset(const at::Tensor& t) {
    return static_cast<NSUInteger>(t.storage_offset() * t.element_size());
}

inline uint32_t round_up(uint32_t n, uint32_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

}  // namespace gsplat::metal
