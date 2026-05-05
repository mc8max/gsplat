// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "MetalContext.h"

#include <ATen/mps/MPSDevice.h>
#include <torch/extension.h>

namespace gsplat::metal {

MetalContext::MetalContext() {
    device_ = at::mps::MPSDevice::getInstance()->device();
    TORCH_CHECK(device_ != nil, "Failed to create Metal device");
    pipelines_ = [[NSMutableDictionary alloc] init];
}

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

void MetalContext::load_library(const std::string& metallib_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSError* error = nil;
    library_ = [device_ newLibraryWithFile:path error:&error];
    TORCH_CHECK(
        library_ != nil,
        "Failed to load Metal library at ",
        metallib_path,
        ": ",
        error != nil ? error.localizedDescription.UTF8String : "unknown error");
    [pipelines_ removeAllObjects];
}

id<MTLComputePipelineState> MetalContext::pipeline(const std::string& function_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    TORCH_CHECK(library_ != nil, "Metal library not loaded");

    NSString* key = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLComputePipelineState> cached = [pipelines_ objectForKey:key];
    if (cached != nil) {
        return cached;
    }

    id<MTLFunction> fn = [library_ newFunctionWithName:key];
    TORCH_CHECK(fn != nil, "Missing Metal function: ", function_name);

    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [device_ newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(
        pso != nil,
        "Failed to create pipeline for ",
        function_name,
        ": ",
        error != nil ? error.localizedDescription.UTF8String : "unknown error");

    [pipelines_ setObject:pso forKey:key];
    return pso;
}

}  // namespace gsplat::metal
