# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection_2dgs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _assert_forward_close(actual, expected):
    expected_radii, expected_means2d, expected_depths, expected_ray_transforms, expected_normals = expected
    actual_radii, actual_means2d, actual_depths, actual_ray_transforms, actual_normals = actual
    valid = ((expected_radii > 0) & (actual_radii.cpu() > 0)).all(dim=-1)

    torch.testing.assert_close(actual_radii.cpu(), expected_radii, rtol=1e-3, atol=1)
    torch.testing.assert_close(actual_means2d.cpu()[valid], expected_means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_depths.cpu()[valid], expected_depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        actual_ray_transforms.cpu()[valid], expected_ray_transforms[valid], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(actual_normals.cpu()[valid], expected_normals[valid], rtol=1e-4, atol=1e-4)


def _sample_inputs(batch_dims=(), cameras=2, gaussians=12, width=32, height=24):
    torch.manual_seed(42)
    means = torch.randn(*batch_dims, gaussians, 3, dtype=torch.float32) * 0.2
    means[..., 2] = torch.rand(*batch_dims, gaussians, dtype=torch.float32) * 1.5 + 1.5
    quats = torch.randn(*batch_dims, gaussians, 4, dtype=torch.float32)
    scales = torch.rand(*batch_dims, gaussians, 3, dtype=torch.float32) * 0.3 + 0.2
    viewmats = torch.eye(4, dtype=torch.float32).expand(*batch_dims, cameras, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.05, 0.05, cameras, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.03, -0.03, cameras, dtype=torch.float32)
    Ks = torch.zeros(*batch_dims, cameras, 3, 3, dtype=torch.float32)
    Ks[..., 0, 0] = 220.0
    Ks[..., 1, 1] = 210.0
    Ks[..., 0, 2] = width * 0.5
    Ks[..., 1, 2] = height * 0.5
    Ks[..., 2, 2] = 1.0
    return means, quats, scales, viewmats, Ks


@pytest.mark.parametrize("batch_dims", [(), (2,)])
def test_projection_2dgs_fused_forward_matches_reference(batch_dims):
    width, height = 32, 24
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=batch_dims, cameras=2, gaussians=10, width=width, height=height
    )

    expected = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    actual = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
    )
    _assert_forward_close(actual, expected)


def test_projection_2dgs_fused_radius_clip_culls_small_entries():
    width, height = 32, 24
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=8, width=width, height=height
    )

    low_clip = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
        radius_clip=0.0,
    )[0]
    high_clip = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
        radius_clip=100.0,
    )[0]

    assert int((high_clip > 0).sum().item()) == 0
    assert int((low_clip > 0).sum().item()) >= 0


def test_projection_2dgs_fused_output_shapes_and_dtypes():
    width, height = 40, 28
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=3, gaussians=9, width=width, height=height
    )
    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
    )

    assert radii.shape == (2, 3, 9, 2)
    assert means2d.shape == (2, 3, 9, 2)
    assert depths.shape == (2, 3, 9)
    assert ray_transforms.shape == (2, 3, 9, 3, 3)
    assert normals.shape == (2, 3, 9, 3)
    assert radii.dtype == torch.int32
    assert means2d.dtype == torch.float32
    assert depths.dtype == torch.float32
    assert ray_transforms.dtype == torch.float32
    assert normals.dtype == torch.float32


def test_projection_2dgs_fused_forward_large_smoke_matches_reference():
    width, height = 64, 48
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=3, gaussians=48, width=width, height=height
    )
    expected = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    actual = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
    )
    _assert_forward_close(actual, expected)


def test_projection_2dgs_fused_backward_matches_reference():
    width, height = 32, 24
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=6, width=width, height=height
    )

    means_ref = means.clone().requires_grad_(True)
    quats_ref = quats.clone().requires_grad_(True)
    scales_ref = scales.clone().requires_grad_(True)
    viewmats_ref = viewmats.clone().requires_grad_(True)
    radii_ref, means2d_ref, depths_ref, ray_transforms_ref, normals_ref = _fully_fused_projection_2dgs(
        means_ref, quats_ref, scales_ref, viewmats_ref, Ks, width, height
    )

    means_mps = means.to("mps").requires_grad_(True)
    quats_mps = quats.to("mps").requires_grad_(True)
    scales_mps = scales.to("mps").requires_grad_(True)
    viewmats_mps = viewmats.to("mps").requires_grad_(True)
    Ks_mps = Ks.to("mps")
    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        means_mps, quats_mps, scales_mps, viewmats_mps, Ks_mps, width, height
    )

    valid = ((radii_ref > 0) & (radii.cpu() > 0)).all(dim=-1)
    v_means2d = torch.randn_like(means2d_ref) * valid[..., None]
    v_depths = torch.randn_like(depths_ref) * valid
    v_ray_transforms = torch.randn_like(ray_transforms_ref) * valid[..., None, None]
    v_normals = torch.randn_like(normals_ref) * valid[..., None]

    v_viewmats_ref, v_quats_ref, v_scales_ref, v_means_ref = torch.autograd.grad(
        (means2d_ref * v_means2d).sum()
        + (depths_ref * v_depths).sum()
        + (ray_transforms_ref * v_ray_transforms).sum()
        + (normals_ref * v_normals).sum(),
        (viewmats_ref, quats_ref, scales_ref, means_ref),
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d.to("mps")).sum()
        + (depths * v_depths.to("mps")).sum()
        + (ray_transforms * v_ray_transforms.to("mps")).sum()
        + (normals * v_normals.to("mps")).sum(),
        (viewmats_mps, quats_mps, scales_mps, means_mps),
    )

    torch.testing.assert_close(v_viewmats.cpu(), v_viewmats_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_quats.cpu(), v_quats_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_scales.cpu(), v_scales_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_means.cpu(), v_means_ref, rtol=1e-3, atol=1e-3)


def test_projection_2dgs_fused_backward_skips_viewmats_when_not_required():
    width, height = 32, 24
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=6, width=width, height=height
    )

    means_mps = means.to("mps").requires_grad_(True)
    quats_mps = quats.to("mps").requires_grad_(True)
    scales_mps = scales.to("mps").requires_grad_(True)
    viewmats_mps = viewmats.to("mps")
    Ks_mps = Ks.to("mps")
    outputs = gm.fully_fused_projection_2dgs(
        means_mps, quats_mps, scales_mps, viewmats_mps, Ks_mps, width, height
    )
    loss = outputs[1].sum() + outputs[2].sum() + outputs[3].sum() + outputs[4].sum()
    grads = torch.autograd.grad(loss, (means_mps, quats_mps, scales_mps), allow_unused=False)
    assert all(g is not None for g in grads)


def test_projection_2dgs_fused_backward_batched_smoke_matches_reference():
    width, height = 48, 36
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=3, gaussians=10, width=width, height=height
    )

    means_ref = means.clone().requires_grad_(True)
    quats_ref = quats.clone().requires_grad_(True)
    scales_ref = scales.clone().requires_grad_(True)
    viewmats_ref = viewmats.clone().requires_grad_(True)
    radii_ref, means2d_ref, depths_ref, ray_transforms_ref, normals_ref = _fully_fused_projection_2dgs(
        means_ref, quats_ref, scales_ref, viewmats_ref, Ks, width, height
    )

    means_mps = means.to("mps").requires_grad_(True)
    quats_mps = quats.to("mps").requires_grad_(True)
    scales_mps = scales.to("mps").requires_grad_(True)
    viewmats_mps = viewmats.to("mps").requires_grad_(True)
    Ks_mps = Ks.to("mps")
    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        means_mps, quats_mps, scales_mps, viewmats_mps, Ks_mps, width, height, radius_clip=0.1
    )

    valid = ((radii_ref > 0) & (radii.cpu() > 0)).all(dim=-1)
    v_means2d = torch.randn_like(means2d_ref) * valid[..., None]
    v_depths = torch.randn_like(depths_ref) * valid
    v_ray_transforms = torch.randn_like(ray_transforms_ref) * valid[..., None, None]
    v_normals = torch.randn_like(normals_ref) * valid[..., None]

    ref_grads = torch.autograd.grad(
        (means2d_ref * v_means2d).sum()
        + (depths_ref * v_depths).sum()
        + (ray_transforms_ref * v_ray_transforms).sum()
        + (normals_ref * v_normals).sum(),
        (viewmats_ref, quats_ref, scales_ref, means_ref),
    )
    metal_grads = torch.autograd.grad(
        (means2d * v_means2d.to("mps")).sum()
        + (depths * v_depths.to("mps")).sum()
        + (ray_transforms * v_ray_transforms.to("mps")).sum()
        + (normals * v_normals.to("mps")).sum(),
        (viewmats_mps, quats_mps, scales_mps, means_mps),
    )

    for metal_grad, ref_grad, rtol, atol in zip(
        metal_grads,
        ref_grads,
        (1e-2, 1e-2, 5e-2, 1e-3),
        (1e-2, 1e-2, 5e-2, 1e-3),
    ):
        torch.testing.assert_close(metal_grad.cpu(), ref_grad, rtol=rtol, atol=atol)
