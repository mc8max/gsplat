// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#import <Metal/Metal.h>

#include <torch/extension.h>

namespace gsplat::metal {

inline void check_mps_float32(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

inline void check_mps_bool(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kBool, name, " must be bool");
    TORCH_CHECK(t.element_size() == 1, name, " must use 1-byte elements");
}

inline id<MTLBuffer> to_mtl_buffer(const at::Tensor& t) {
    TORCH_CHECK(t.is_mps(), "Expected MPS tensor");
    return (__bridge id<MTLBuffer>)(t.storage().data_ptr().get());
}

inline NSUInteger byte_offset(const at::Tensor& t) {
    return static_cast<NSUInteger>(t.storage_offset() * t.element_size());
}

inline void set_optional_tensor_buffer(
    id<MTLComputeCommandEncoder> enc,
    const at::Tensor& tensor,
    NSUInteger index
) {
    if (tensor.defined()) {
        [enc setBuffer:to_mtl_buffer(tensor) offset:byte_offset(tensor) atIndex:index];
    } else {
        [enc setBuffer:nil offset:0 atIndex:index];
    }
}

inline uint32_t round_up(uint32_t n, uint32_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

}  // namespace gsplat::metal
