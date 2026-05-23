# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat
import gsplat.metal as gm

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_3dgs_inputs(c=2, n=8, channels=4, width=32, height=24):
    torch.manual_seed(7)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.5 + 1.5
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.2 + 0.15
    opacities = torch.rand(n, dtype=torch.float32) * 0.6 + 0.3
    colors = torch.rand(n, channels, dtype=torch.float32)
    backgrounds = torch.rand(c, channels, dtype=torch.float32)
    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.04, -0.04, c, dtype=torch.float32)
    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 220.0
    Ks[:, 1, 1] = 210.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    return means, quats, scales, opacities, colors, backgrounds, viewmats, Ks


def _sample_2dgs_inputs(c=2, n=8, channels=3, width=32, height=24):
    torch.manual_seed(21)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.5 + 1.5
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.2 + 0.15
    opacities = torch.rand(n, dtype=torch.float32) * 0.6 + 0.3
    colors = torch.rand(n, channels, dtype=torch.float32)
    backgrounds = torch.rand(c, channels, dtype=torch.float32)
    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.04, -0.04, c, dtype=torch.float32)
    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 220.0
    Ks[:, 1, 1] = 210.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    return means, quats, scales, opacities, colors, backgrounds, viewmats, Ks


def test_rasterization_matches_direct_metal_pipeline(mps_device):
    width, height = 32, 24
    tile_size = 16
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        backgrounds,
        viewmats,
        Ks,
    ) = _sample_3dgs_inputs(width=width, height=height)

    actual_colors, actual_alphas, meta = gsplat.rasterization(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        colors.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        tile_size=tile_size,
        backgrounds=backgrounds.to(mps_device),
        render_mode="RGB",
    )

    mps_means = means.to(mps_device)
    mps_quats = quats.to(mps_device)
    mps_scales = scales.to(mps_device)
    mps_opacities = opacities.to(mps_device)
    mps_colors = colors.to(mps_device)
    mps_backgrounds = backgrounds.to(mps_device)
    mps_viewmats = viewmats.to(mps_device)
    mps_Ks = Ks.to(mps_device)

    radii, means2d, depths, conics, compensations = gm.fully_fused_projection(
        mps_means,
        None,
        mps_quats,
        mps_scales,
        mps_viewmats,
        mps_Ks,
        width,
        height,
        packed=False,
        sparse_grad=False,
        calc_compensations=False,
    )
    direct_colors = torch.broadcast_to(mps_colors.unsqueeze(0), (viewmats.shape[0], colors.shape[0], colors.shape[1]))
    direct_opacities = torch.broadcast_to(mps_opacities.unsqueeze(0), (viewmats.shape[0], opacities.shape[0]))
    if compensations is not None:
        direct_opacities = direct_opacities * compensations

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = gm.isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=viewmats.shape[0],
        image_ids=None,
        gaussian_ids=None,
        conics=conics,
        opacities=direct_opacities,
    )
    isect_offsets = gm.isect_offset_encode(
        isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    ).reshape(viewmats.shape[0], tile_height, tile_width)
    expected_colors, expected_alphas = gm.rasterize_to_pixels(
        means2d,
        conics,
        direct_colors,
        direct_opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=mps_backgrounds,
        packed=False,
    )

    torch.testing.assert_close(actual_colors, expected_colors)
    torch.testing.assert_close(actual_alphas, expected_alphas)
    torch.testing.assert_close(meta["means2d"], means2d)
    torch.testing.assert_close(meta["isect_offsets"], isect_offsets)


def test_rasterization_eval3d_runs_on_supported_mps_path(mps_device):
    width, height = 32, 24
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        backgrounds,
        viewmats,
        Ks,
    ) = _sample_3dgs_inputs(c=2, n=6, channels=3, width=width, height=height)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        colors.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        backgrounds=backgrounds.to(mps_device),
        render_mode="RGB-d",
        with_eval3d=True,
        return_normals=True,
        camera_model="pinhole",
    )

    assert render_colors.shape == (viewmats.shape[0], height, width, colors.shape[-1] + 1)
    assert render_alphas.shape == (viewmats.shape[0], height, width, 1)
    assert meta["normals"].shape == (viewmats.shape[0], height, width, 3)
    assert torch.isfinite(render_colors).all()
    assert torch.isfinite(render_alphas).all()
    assert torch.isfinite(meta["normals"]).all()


def test_rasterization_eval3d_rejects_ortho_on_mps(mps_device):
    width, height = 32, 24
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        backgrounds,
        viewmats,
        Ks,
    ) = _sample_3dgs_inputs(c=1, n=4, channels=3, width=width, height=height)

    with pytest.raises(NotImplementedError, match="ortho"):
        gsplat.rasterization(
            means.to(mps_device),
            quats.to(mps_device),
            scales.to(mps_device),
            opacities.to(mps_device),
            colors.to(mps_device),
            viewmats.to(mps_device),
            Ks.to(mps_device),
            width,
            height,
            packed=False,
            backgrounds=backgrounds[:1].to(mps_device),
            render_mode="RGB",
            with_eval3d=True,
            camera_model="ortho",
        )


def test_rasterization_2dgs_matches_direct_metal_pipeline(mps_device):
    width, height = 32, 24
    tile_size = 16
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        backgrounds,
        viewmats,
        Ks,
    ) = _sample_2dgs_inputs(width=width, height=height)

    (
        actual_colors,
        actual_alphas,
        actual_normals,
        actual_surf_normals,
        actual_distort,
        actual_median,
        meta,
    ) = gsplat.rasterization_2dgs(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        colors.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        tile_size=tile_size,
        backgrounds=backgrounds.to(mps_device),
        render_mode="RGB",
    )

    mps_means = means.to(mps_device)
    mps_quats = quats.to(mps_device)
    mps_scales = scales.to(mps_device)
    mps_opacities = opacities.to(mps_device)
    mps_colors = colors.to(mps_device)
    mps_backgrounds = backgrounds.to(mps_device)
    mps_viewmats = viewmats.to(mps_device)
    mps_Ks = Ks.to(mps_device)

    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        mps_means,
        mps_quats,
        mps_scales,
        mps_viewmats,
        mps_Ks,
        width,
        height,
        packed=False,
        sparse_grad=False,
    )
    direct_colors = torch.broadcast_to(mps_colors.unsqueeze(0), (viewmats.shape[0], colors.shape[0], colors.shape[1]))
    direct_opacities = torch.broadcast_to(mps_opacities.unsqueeze(0), (viewmats.shape[0], opacities.shape[0]))
    densify = torch.zeros_like(means2d, dtype=mps_means.dtype, device=mps_device)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = gm.isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=viewmats.shape[0],
        image_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = gm.isect_offset_encode(
        isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    ).reshape(viewmats.shape[0], tile_height, tile_width)
    (
        expected_colors,
        expected_alphas,
        expected_normals,
        expected_distort,
        expected_median,
    ) = gm.rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        direct_colors,
        direct_opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=mps_backgrounds,
        packed=False,
    )
    expected_normals = torch.einsum(
        "...ij,...hwj->...hwi",
        torch.linalg.inv(mps_viewmats)[..., :3, :3],
        expected_normals,
    )

    torch.testing.assert_close(actual_colors, expected_colors)
    torch.testing.assert_close(actual_alphas, expected_alphas)
    torch.testing.assert_close(actual_normals, expected_normals)
    torch.testing.assert_close(actual_distort, expected_distort)
    torch.testing.assert_close(actual_median, expected_median)
    assert actual_surf_normals is None
    torch.testing.assert_close(meta["means2d"], means2d)
