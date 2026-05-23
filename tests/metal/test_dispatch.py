# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat._dispatch as gd
import gsplat.metal as gm
from gsplat import _camera_types as camera_types
from gsplat.cuda import _wrapper as cuda_wrapper


def test_dispatch_exports_backend_neutral_types():
    assert gd.RollingShutterType is camera_types.RollingShutterType
    assert gd.FThetaPolynomialType is camera_types.FThetaPolynomialType
    assert gd.FThetaCameraDistortionParameters is camera_types.FThetaCameraDistortionParameters
    assert gd.UnscentedTransformParameters is camera_types.UnscentedTransformParameters
    assert (
        gd.RowOffsetStructuredSpinningLidarModelParametersExt
        is camera_types.RowOffsetStructuredSpinningLidarModelParametersExt
    )
    assert (
        gd.BivariateWindshieldModelParameters
        is camera_types.BivariateWindshieldModelParameters
    )


def test_dispatch_world_to_cam_matches_cuda_wrapper_on_cpu():
    means = torch.randn(4, 3, dtype=torch.float32)
    covars = torch.eye(3, dtype=torch.float32).expand(4, 3, 3).clone()
    viewmats = torch.eye(4, dtype=torch.float32).expand(2, 4, 4).clone()

    actual_means, actual_covars = gd.world_to_cam(means, covars, viewmats)
    expected_means, expected_covars = cuda_wrapper.world_to_cam(means, covars, viewmats)

    torch.testing.assert_close(actual_means, expected_means)
    torch.testing.assert_close(actual_covars, expected_covars)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_dispatch_proj_routes_to_metal_projection_ewa_simple(mps_device):
    means = torch.tensor([[[[0.1, 0.0, 2.0], [0.2, -0.1, 3.0]]]], dtype=torch.float32)
    covars = (
        torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 1, 3, 3)
        .expand(1, 1, 2, 3, 3)
        .contiguous()
    )
    Ks = torch.tensor(
        [[[[160.0, 0.0, 32.0], [0.0, 150.0, 24.0], [0.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    )

    actual = gd.proj(
        means.to(mps_device),
        covars.to(mps_device),
        Ks.to(mps_device),
        width=64,
        height=48,
        camera_model="pinhole",
    )
    expected = gm.projection_ewa_simple(
        means.to(mps_device),
        covars.to(mps_device),
        Ks.to(mps_device),
        width=64,
        height=48,
        camera_model="pinhole",
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_dispatch_quat_scale_routes_to_metal(mps_device):
    quats = torch.randn(5, 4, device=mps_device, dtype=torch.float32)
    scales = torch.rand(5, 3, device=mps_device, dtype=torch.float32) + 0.1

    actual = gd.quat_scale_to_covar_preci(quats, scales, triu=True)
    expected = gm.quat_scale_to_covar_preci(quats, scales, triu=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
