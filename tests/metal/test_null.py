# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def test_has_metal():
    assert gm.has_metal()


def test_null_identity(mps_device):
    x = torch.randn(1024, device=mps_device)
    y = gm.metal_null(x)
    assert y.device.type == "mps"
    torch.testing.assert_close(y.cpu(), x.cpu())


def test_null_via_torch_ops(mps_device):
    assert gm.has_metal()
    x = torch.randn(512, device=mps_device)
    y = torch.ops.gsplat.metal_null(x)
    torch.testing.assert_close(y.cpu(), x.cpu())


def test_null_preserves_shape(mps_device):
    for shape in [(1,), (256,), (32, 64), (4, 128, 128)]:
        x = torch.randn(shape, device=mps_device)
        y = gm.metal_null(x)
        assert y.shape == x.shape
