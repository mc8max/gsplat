# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat
import gsplat.metal as gm
from gsplat import _camera_types as camera_types


def test_public_api_exports_backend_neutral_types():
    assert gsplat.CameraModel is camera_types.CameraModel
    assert gsplat.has_camera_wrappers()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_public_api_proj_routes_to_metal(mps_device):
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

    actual = gsplat.proj(
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
