# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm
from tests.metal.test_intersect_tile_lidar import _make_test_lidar
from tests.metal.test_rasterize_to_indices_2dgs import _prepare_projected_inputs as _prepare_2dgs_indices
from tests.metal.test_rasterize_to_indices_3dgs import _manual_scene as _manual_3dgs_scene
from tests.metal.test_rasterize_to_pixels_eval3d import _sample_inputs as _sample_eval3d_inputs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _world_inputs(cameras=2, gaussians=4, width=24, height=16):
    torch.manual_seed(7)
    means = torch.randn(gaussians, 3, dtype=torch.float32) * 0.15
    means[:, 2] = torch.rand(gaussians, dtype=torch.float32) * 0.8 + 1.5
    quats = torch.randn(gaussians, 4, dtype=torch.float32)
    scales = torch.rand(gaussians, 3, dtype=torch.float32) * 0.15 + 0.08
    covars = torch.eye(3, dtype=torch.float32).expand(gaussians, 3, 3).clone() * 0.04

    viewmats = torch.eye(4, dtype=torch.float32).expand(cameras, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.04, 0.04, cameras, dtype=torch.float32)
    Ks = torch.zeros(cameras, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 180.0
    Ks[:, 1, 1] = 170.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    opacities = torch.rand(gaussians, dtype=torch.float32) * 0.6 + 0.2
    return means, quats, scales, covars, opacities, viewmats, Ks


def _projected_3dgs_inputs(width=16, height=12, channels=3):
    means, quats, scales, _covars, opacities, viewmats, Ks = _world_inputs(
        cameras=2, gaussians=4, width=width, height=height
    )
    radii, means2d, depths, conics, _ = gm.fully_fused_projection(
        means.to("mps"),
        None,
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = gm.intersect_offset_encode(
        isect_ids, viewmats.shape[0], tile_width, tile_height
    )
    colors = torch.rand(viewmats.shape[0], means.shape[0], channels, dtype=torch.float32, device="mps")
    view_opacities = torch.broadcast_to(opacities.to("mps")[None, :], depths.shape)
    return (
        means2d,
        conics,
        colors,
        view_opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )


def _projected_2dgs_inputs(width=16, height=12, channels=3):
    (
        _transmittances,
        means2d,
        ray_transforms,
        opacities,
        _width,
        _height,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_2dgs_indices(width=width, height=height, tile_size=4)
    normals = torch.zeros(*means2d.shape[:-1], 3, dtype=torch.float32)
    colors = torch.rand(*means2d.shape[:-1], channels, dtype=torch.float32)
    return (
        means2d.to("mps"),
        ray_transforms.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        normals.to("mps"),
        torch.zeros_like(means2d).to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
    )


def test_kernel_smoke_matrix(mps_device):
    assert gm.has_metal()

    means, quats, scales, covars, opacities, viewmats, Ks = _world_inputs()
    tile_size = 4
    tile_width = math.ceil(24 / tile_size)
    tile_height = math.ceil(16 / tile_size)

    x = torch.randn(8, device=mps_device)
    assert gm.metal_null(x).shape == x.shape

    param = torch.randn(4, 3, device=mps_device, dtype=torch.float32)
    param_grad = torch.randn_like(param)
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    gm.adam(param, param_grad, exp_avg, exp_avg_sq, None, 1e-2, 0.9, 0.99, 1e-8)

    binoms = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=mps_device)
    gm.relocation(
        torch.tensor([0.3, 0.6], dtype=torch.float32, device=mps_device),
        torch.ones(2, 3, dtype=torch.float32, device=mps_device),
        torch.tensor([1, 2], dtype=torch.int32, device=mps_device),
        binoms,
        2,
    )

    gm.quat_scale_to_covar_preci(
        quats.to(mps_device), scales.to(mps_device), compute_covar=True, compute_preci=True, triu=True
    )

    dirs = torch.nn.functional.normalize(torch.randn(5, 3, device=mps_device), dim=-1)
    coeffs = torch.randn(5, 4, 3, device=mps_device)
    gm.spherical_harmonics(1, dirs, coeffs)

    gm.eval_bivariate_poly(
        torch.tensor([0.1, -0.2], dtype=torch.float32, device=mps_device),
        torch.tensor([0.3, 0.4], dtype=torch.float32, device=mps_device),
        torch.tensor([1.0, 0.5, -0.25], dtype=torch.float32, device=mps_device),
        1,
    )

    ident = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=mps_device)
    zero = torch.zeros_like(ident)
    gm.distort_camera_rays(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=mps_device),
        ident,
        ident,
        zero,
        zero,
        1,
        False,
    )

    gm.projection_ewa_simple(
        means[None].to(mps_device),
        covars[None].to(mps_device),
        Ks[:1].to(mps_device),
        24,
        16,
        camera_model="pinhole",
    )
    radii, means2d, depths, conics, _ = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        24,
        16,
        calc_compensations=False,
        camera_model="pinhole",
    )
    gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        24,
        16,
        calc_compensations=False,
        camera_model="pinhole",
        packed=True,
    )
    gm.fully_fused_projection_2dgs(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        24,
        16,
    )
    gm.fully_fused_projection_2dgs(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        24,
        16,
        packed=True,
    )
    gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        24,
        16,
        camera_model="pinhole",
    )

    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    gm.intersect_offset_encode(isect_ids, viewmats.shape[0], tile_width, tile_height)

    lidar = _make_test_lidar(mps_device)
    lidar_means2d = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, device=mps_device)
    lidar_radii = torch.tensor([[[32, 24]]], dtype=torch.int32, device=mps_device)
    lidar_depths = torch.tensor([[0.2]], dtype=torch.float32, device=mps_device)
    gm.intersect_tiles_lidar(lidar, lidar_means2d, lidar_radii, lidar_depths, sort=True)

    raster3d_inputs = _projected_3dgs_inputs()
    gm.rasterize_to_pixels(*raster3d_inputs)

    raster2d_inputs = _projected_2dgs_inputs()
    gm.rasterize_to_pixels_2dgs(*raster2d_inputs)

    eval3d_inputs = _sample_eval3d_inputs(channels=3)
    means_w, quats_w, scales_w, colors_w, opac_w, backgrounds_w, viewmats_w, ks_w, rays_w, offsets_w, flat_w, width_w, height_w, tile_w = eval3d_inputs
    gm.rasterize_to_pixels_eval3d(
        means_w.to(mps_device),
        quats_w.to(mps_device),
        scales_w.to(mps_device),
        colors_w.to(mps_device),
        opac_w.to(mps_device),
        viewmats_w.to(mps_device),
        ks_w.to(mps_device),
        width_w,
        height_w,
        tile_w,
        offsets_w.to(mps_device),
        flat_w.to(mps_device),
        backgrounds=backgrounds_w.to(mps_device),
        rays=rays_w.to(mps_device),
    )

    scene3d = _manual_3dgs_scene(image_count=1)
    gm.rasterize_to_indices_in_range(
        0,
        8,
        *[arg.to(mps_device) if isinstance(arg, torch.Tensor) else arg for arg in scene3d],
    )
    inputs2d = _prepare_2dgs_indices()
    gm.rasterize_to_indices_in_range_2dgs(
        0,
        8,
        *[arg.to(mps_device) if isinstance(arg, torch.Tensor) else arg for arg in inputs2d],
    )
