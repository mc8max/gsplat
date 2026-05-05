// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <string>

namespace gsplat::metal {

class MetalContext {
public:
    static MetalContext& instance();

    void load_library(const std::string& metallib_path);
    id<MTLComputePipelineState> pipeline(const std::string& function_name);

    id<MTLDevice> device() {
        return device_;
    }

private:
    MetalContext();

    std::mutex mutex_;
    id<MTLDevice> device_ = nil;
    id<MTLLibrary> library_ = nil;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines_ = nil;
};

}  // namespace gsplat::metal
