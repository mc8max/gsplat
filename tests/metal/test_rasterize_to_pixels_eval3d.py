# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import sys

import pytest
import torch

import gsplat.metal as gm
from gsplat._camera_types import (
    BivariateWindshieldModelParameters,
    ExternalDistortionReferencePolynomial,
    RollingShutterType,
)
from gsplat.cuda._torch_cameras import (
    _BaseCameraModel,
    _interpolate_shutter_pose,
    _pose_camera_ray_to_world_ray,
    _viewmat_to_pose,
)
from gsplat.metal._math import (
    _fully_fused_projection,
    _isect_offset_encode,
    _isect_tiles,
    _quat_scale_to_covar_preci,
    _rasterize_to_pixels_eval3d,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _pack_info_torch(ray_indices, n_rays=None):
    assert ray_indices.dim() == 1, "ray_indices must be a 1D tensor with shape (n_samples)."
    if n_rays is None:
        n_rays = int(ray_indices.max().item()) + 1 if ray_indices.numel() > 0 else 0
    counts = torch.bincount(ray_indices.to(torch.int64), minlength=n_rays)
    starts = counts.cumsum(dim=0) - counts
    return torch.stack([starts, counts], dim=-1)


def _render_weight_from_alpha_torch(
    alphas,
    packed_info=None,
    ray_indices=None,
    n_rays=None,
    prefix_trans=None,
):
    if packed_info is None:
        if ray_indices is None:
            raise ValueError("packed_info or ray_indices is required")
        packed_info = _pack_info_torch(ray_indices, n_rays)

    starts = packed_info[:, 0].to(torch.int64)
    counts = packed_info[:, 1].to(torch.int64)
    trans = torch.empty_like(alphas)
    weights = torch.empty_like(alphas)
    if prefix_trans is None:
        prefix_vals = torch.ones((packed_info.shape[0],), device=alphas.device, dtype=alphas.dtype)
    else:
        prefix_vals = prefix_trans.reshape(-1).to(device=alphas.device, dtype=alphas.dtype)

    for ray_idx in range(packed_info.shape[0]):
        count = int(counts[ray_idx].item())
        if count == 0:
            continue
        start = int(starts[ray_idx].item())
        alpha_chunk = alphas[start : start + count]
        trans_chunk = torch.cumprod(
            torch.cat(
                [
                    prefix_vals[ray_idx : ray_idx + 1],
                    1.0 - alpha_chunk[:-1],
                ]
            ),
            dim=0,
        )
        trans[start : start + count] = trans_chunk
        weights[start : start + count] = alpha_chunk * trans_chunk

    return weights, trans


def _accumulate_along_rays_torch(weights, values=None, ray_indices=None, n_rays=None):
    if ray_indices is None or n_rays is None:
        raise ValueError("ray_indices and n_rays are required")
    ray_indices = ray_indices.to(torch.int64)
    if values is None:
        out = torch.zeros((n_rays, 1), device=weights.device, dtype=weights.dtype)
        out.index_add_(0, ray_indices, weights[:, None])
        return out
    out = torch.zeros((n_rays, values.shape[-1]), device=values.device, dtype=values.dtype)
    out.index_add_(0, ray_indices, weights[:, None] * values)
    return out


def _ensure_nerfacc_reference_fallbacks():
    nerfacc = pytest.importorskip("nerfacc", reason="eval3d torch reference requires nerfacc")
    if torch.cuda.is_available():
        return
    nerfacc.pack_info = _pack_info_torch
    nerfacc.render_weight_from_alpha = _render_weight_from_alpha_torch
    nerfacc.accumulate_along_rays = _accumulate_along_rays_torch
    sys.modules["nerfacc"].pack_info = _pack_info_torch
    sys.modules["nerfacc"].render_weight_from_alpha = _render_weight_from_alpha_torch
    sys.modules["nerfacc"].accumulate_along_rays = _accumulate_along_rays_torch


def _make_external_distortion_params(
    h_poly,
    v_poly,
    h_inv=None,
    v_inv=None,
    ref_poly=ExternalDistortionReferencePolynomial.FORWARD,
    device=None,
):
    if device is None:
        device = torch.device("cpu")
    if h_inv is None:
        h_inv = [0.0, 1.0, 0.0]
    if v_inv is None:
        v_inv = [0.0, 0.0, 1.0]
    params = BivariateWindshieldModelParameters()
    params.reference_poly = ref_poly
    params.horizontal_poly = torch.tensor(h_poly, dtype=torch.float32, device=device)
    params.vertical_poly = torch.tensor(v_poly, dtype=torch.float32, device=device)
    params.horizontal_poly_inverse = torch.tensor(h_inv, dtype=torch.float32, device=device)
    params.vertical_poly_inverse = torch.tensor(v_inv, dtype=torch.float32, device=device)
    return params


def _sample_inputs(width=24, height=16, tile_size=8, c=2, n=6, channels=5):
    torch.manual_seed(123)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.3 + 1.7
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.15 + 0.12

    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.03, -0.03, c, dtype=torch.float32)

    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 180.0
    Ks[:, 1, 1] = 175.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0

    colors = torch.rand(c, n, channels, dtype=torch.float32)
    opacities = torch.rand(c, n, dtype=torch.float32) * 0.45 + 0.35
    backgrounds = torch.rand(c, channels, dtype=torch.float32)

    covars, _ = _quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )
    radii, means2d, depths, _conics, _comp = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        camera_model="pinhole",
    )
    _, isect_ids, flatten_ids = _isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        math.ceil(width / tile_size),
        math.ceil(height / tile_size),
    )
    isect_offsets = _isect_offset_encode(
        isect_ids,
        means2d.shape[0],
        math.ceil(width / tile_size),
        math.ceil(height / tile_size),
    )
    rays = _make_pinhole_rays(viewmats, Ks, width, height)
    return (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        Ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )


def _make_pinhole_rays(viewmats, ks, width, height):
    c = viewmats.shape[0]
    px = torch.arange(width, dtype=torch.float32) + 0.5
    py = torch.arange(height, dtype=torch.float32) + 0.5
    grid_x, grid_y = torch.meshgrid(px, py, indexing="xy")

    rays = torch.empty(c, height, width, 6, dtype=torch.float32)
    for cam in range(c):
        fx = ks[cam, 0, 0]
        fy = ks[cam, 1, 1]
        cx = ks[cam, 0, 2]
        cy = ks[cam, 1, 2]
        cam_dirs = torch.stack(
            [(grid_x - cx) / fx, (grid_y - cy) / fy, torch.ones_like(grid_x)], dim=-1
        )
        cam_to_world = torch.linalg.inv(viewmats[cam])
        origin = cam_to_world[:3, 3]
        rot = cam_to_world[:3, :3]
        world_dirs = torch.einsum("ij,hwj->hwi", rot, cam_dirs)
        rays[cam, ..., :3] = origin
        rays[cam, ..., 3:] = world_dirs
    return rays


def _make_model_world_rays(
    viewmats,
    ks,
    width,
    height,
    camera_model="pinhole",
    radial_coeffs=None,
    tangential_coeffs=None,
    thin_prism_coeffs=None,
    rolling_shutter=RollingShutterType.GLOBAL,
    viewmats_rs=None,
    external_distortion_coeffs=None,
):
    c = viewmats.shape[0]
    focal_lengths = torch.stack([ks[:, 0, 0], ks[:, 1, 1]], dim=-1)
    principal_points = torch.stack([ks[:, 0, 2], ks[:, 1, 2]], dim=-1)
    camera = _BaseCameraModel.create(
        width=width,
        height=height,
        camera_model=camera_model,
        principal_points=principal_points,
        focal_lengths=focal_lengths if camera_model != "ftheta" else None,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        rs_type=rolling_shutter,
    )

    px = torch.arange(width, dtype=torch.float32) + 0.5
    py = torch.arange(height, dtype=torch.float32) + 0.5
    grid_x, grid_y = torch.meshgrid(px, py, indexing="xy")
    image_points = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
    image_points = image_points.expand(c, -1, -1)

    cam_rays, valid = camera.image_point_to_camera_ray(image_points)
    if external_distortion_coeffs is not None:
        cam_rays = gm.distort_camera_rays(
            cam_rays.reshape(-1, 3).to("mps"),
            external_distortion_coeffs.horizontal_poly.to("mps"),
            external_distortion_coeffs.vertical_poly.to("mps"),
            external_distortion_coeffs.horizontal_poly_inverse.to("mps"),
            external_distortion_coeffs.vertical_poly_inverse.to("mps"),
            int(external_distortion_coeffs.reference_poly),
            True,
        ).cpu().reshape(c, height * width, 3)

    pose_start = _viewmat_to_pose(viewmats)
    pose_end = _viewmat_to_pose(viewmats_rs if viewmats_rs is not None else viewmats)
    rel = camera.shutter_relative_frame_time(image_points)
    pose = _interpolate_shutter_pose(pose_start[:, None, :], pose_end[:, None, :], rel)
    ray_o, ray_d = _pose_camera_ray_to_world_ray(pose, cam_rays)
    ray_o = ray_o * valid[..., None]
    ray_d = ray_d * valid[..., None]
    return torch.cat([ray_o, ray_d], dim=-1).reshape(c, height, width, 6)


def _reference_eval3d(
    means,
    quats,
    scales,
    colors,
    opacities,
    viewmats,
    ks,
    rays,
    image_width,
    image_height,
    tile_size,
    isect_offsets,
    flatten_ids,
    backgrounds=None,
    masks=None,
    use_hit_distance=False,
    return_normals=False,
):
    _ensure_nerfacc_reference_fallbacks()
    return _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        ks,
        image_width,
        image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        return_last_ids=True,
        return_sample_counts=True,
        rays=rays,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
    )


def test_rasterize_to_pixels_eval3d_forward_matches_reference():
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = _sample_inputs()
    expected = _reference_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        ks,
        rays,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        return_normals=True,
    )

    actual = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=rays.to("mps"),
        return_sample_counts=True,
        return_normals=True,
    )

    assert torch.allclose(actual[0].cpu(), expected[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual[1].cpu(), expected[1], atol=1e-4, rtol=1e-4)
    assert torch.equal(actual[2].cpu(), expected[2])
    assert torch.equal(actual[3].cpu(), expected[3])
    assert torch.allclose(actual[4].cpu(), expected[4], atol=1e-4, rtol=1e-4)


def test_rasterize_to_pixels_eval3d_hit_distance_matches_reference():
    inputs = _sample_inputs(channels=4)
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = inputs
    expected = _reference_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        ks,
        rays,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        use_hit_distance=True,
        return_normals=True,
    )

    actual = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=rays.to("mps"),
        return_sample_counts=True,
        use_hit_distance=True,
        return_normals=True,
    )

    assert torch.allclose(actual[0].cpu(), expected[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual[1].cpu(), expected[1], atol=1e-4, rtol=1e-4)
    assert torch.equal(actual[2].cpu(), expected[2])
    assert torch.equal(actual[3].cpu(), expected[3])
    assert torch.allclose(actual[4].cpu(), expected[4], atol=1e-4, rtol=1e-4)


def test_rasterize_to_pixels_eval3d_implicit_pinhole_matches_explicit_rays():
    inputs = _sample_inputs(channels=4)
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = inputs

    explicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=rays.to("mps"),
        return_sample_counts=True,
        use_hit_distance=True,
        return_normals=True,
    )
    implicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=None,
        return_sample_counts=True,
        use_hit_distance=True,
        return_normals=True,
    )

    for idx in range(5):
        lhs = explicit[idx]
        rhs = implicit[idx]
        if lhs.dtype.is_floating_point:
            assert torch.allclose(lhs.cpu(), rhs.cpu(), atol=1e-4, rtol=1e-4)
        else:
            assert torch.equal(lhs.cpu(), rhs.cpu())


def test_rasterize_to_pixels_eval3d_distorted_pinhole_matches_explicit_rays():
    inputs = _sample_inputs(channels=4)
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        _rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = inputs
    radial = torch.tensor(
        [[0.02, -0.01, 0.001, -0.0005, 0.0, 0.0], [-0.015, 0.008, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    tangential = torch.tensor([[0.001, -0.0015], [-0.0007, 0.0012]], dtype=torch.float32)
    thin_prism = torch.tensor(
        [[0.0004, -0.0002, 0.0001, -0.0001], [0.0002, 0.0001, -0.0002, 0.0003]],
        dtype=torch.float32,
    )
    viewmats_rs = viewmats.clone()
    viewmats_rs[:, 0, 3] += 0.02
    viewmats_rs[:, 1, 3] -= 0.01

    explicit_rays = _make_model_world_rays(
        viewmats,
        ks,
        width,
        height,
        camera_model="pinhole",
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs,
    )

    explicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=explicit_rays.to("mps"),
        use_hit_distance=True,
        return_normals=True,
    )
    implicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=None,
        radial_coeffs=radial.to("mps"),
        tangential_coeffs=tangential.to("mps"),
        thin_prism_coeffs=thin_prism.to("mps"),
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs.to("mps"),
        use_hit_distance=True,
        return_normals=True,
    )

    for idx in (0, 1, 2, 4):
        lhs = explicit[idx]
        rhs = implicit[idx]
        if lhs.dtype.is_floating_point:
            assert torch.allclose(lhs.cpu(), rhs.cpu(), atol=1e-4, rtol=1e-4)
        else:
            assert torch.equal(lhs.cpu(), rhs.cpu())


def test_rasterize_to_pixels_eval3d_fisheye_and_external_distortion_match_explicit_rays():
    inputs = _sample_inputs(channels=4)
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        _rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = inputs
    radial = torch.tensor(
        [[0.01, -0.002, 0.0005, -0.0001], [-0.008, 0.001, -0.0003, 0.00005]],
        dtype=torch.float32,
    )
    ext = _make_external_distortion_params(
        h_poly=[0.0, 1.0, 0.02],
        v_poly=[0.0, -0.01, 1.0],
        device=torch.device("mps"),
    )
    explicit_rays = _make_model_world_rays(
        viewmats,
        ks,
        width,
        height,
        camera_model="fisheye",
        radial_coeffs=radial,
        external_distortion_coeffs=ext,
    )

    explicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=explicit_rays.to("mps"),
        return_normals=True,
    )
    implicit = gm.rasterize_to_pixels_eval3d(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=None,
        camera_model="fisheye",
        radial_coeffs=radial.to("mps"),
        external_distortion_coeffs=ext,
        return_normals=True,
    )

    for idx in (0, 1, 2, 4):
        lhs = explicit[idx]
        rhs = implicit[idx]
        if lhs.dtype.is_floating_point:
            assert torch.allclose(lhs.cpu(), rhs.cpu(), atol=1e-4, rtol=1e-4)
        else:
            assert torch.equal(lhs.cpu(), rhs.cpu())


def test_rasterize_to_pixels_eval3d_backward_matches_reference():
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = _sample_inputs(channels=3)

    ref_means = means.clone().requires_grad_(True)
    ref_quats = quats.clone().requires_grad_(True)
    ref_scales = scales.clone().requires_grad_(True)
    ref_colors = colors.clone().requires_grad_(True)
    ref_opacities = opacities.clone().requires_grad_(True)
    ref_rays = rays.clone().requires_grad_(True)
    ref_bg = backgrounds.clone()

    ref_out = _reference_eval3d(
        ref_means,
        ref_quats,
        ref_scales,
        ref_colors,
        ref_opacities,
        viewmats,
        ks,
        ref_rays,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=ref_bg,
        use_hit_distance=True,
        return_normals=True,
    )
    v_colors = torch.randn_like(ref_out[0])
    v_alphas = torch.randn_like(ref_out[1])
    v_normals = torch.randn_like(ref_out[4])
    ref_loss = (
        (ref_out[0] * v_colors).sum()
        + (ref_out[1] * v_alphas).sum()
        + (ref_out[4] * v_normals).sum()
    )
    ref_loss.backward()

    metal_means = means.to("mps").requires_grad_(True)
    metal_quats = quats.to("mps").requires_grad_(True)
    metal_scales = scales.to("mps").requires_grad_(True)
    metal_colors = colors.to("mps").requires_grad_(True)
    metal_opacities = opacities.to("mps").requires_grad_(True)
    metal_rays = rays.to("mps").requires_grad_(True)
    metal_out = gm.rasterize_to_pixels_eval3d(
        metal_means,
        metal_quats,
        metal_scales,
        metal_colors,
        metal_opacities,
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=metal_rays,
        use_hit_distance=True,
        return_normals=True,
    )
    metal_loss = (
        (metal_out[0] * v_colors.to("mps")).sum()
        + (metal_out[1] * v_alphas.to("mps")).sum()
        + (metal_out[4] * v_normals.to("mps")).sum()
    )
    metal_loss.backward()

    assert torch.allclose(metal_means.grad.cpu(), ref_means.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_quats.grad.cpu(), ref_quats.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_scales.grad.cpu(), ref_scales.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_colors.grad.cpu(), ref_colors.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_opacities.grad.cpu(), ref_opacities.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_rays.grad.cpu(), ref_rays.grad, atol=2e-3, rtol=2e-3)


def test_rasterize_to_pixels_eval3d_implicit_pinhole_backward_matches_reference():
    inputs = _sample_inputs(channels=3)
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        viewmats,
        ks,
        rays,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    ) = inputs

    ref_means = means.clone().requires_grad_(True)
    ref_quats = quats.clone().requires_grad_(True)
    ref_scales = scales.clone().requires_grad_(True)
    ref_colors = colors.clone().requires_grad_(True)
    ref_opacities = opacities.clone().requires_grad_(True)

    ref_out = _reference_eval3d(
        ref_means,
        ref_quats,
        ref_scales,
        ref_colors,
        ref_opacities,
        viewmats,
        ks,
        rays,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        use_hit_distance=True,
        return_normals=True,
    )
    v_colors = torch.randn_like(ref_out[0])
    v_alphas = torch.randn_like(ref_out[1])
    v_normals = torch.randn_like(ref_out[4])
    ref_loss = (
        (ref_out[0] * v_colors).sum()
        + (ref_out[1] * v_alphas).sum()
        + (ref_out[4] * v_normals).sum()
    )
    ref_loss.backward()

    metal_means = means.to("mps").requires_grad_(True)
    metal_quats = quats.to("mps").requires_grad_(True)
    metal_scales = scales.to("mps").requires_grad_(True)
    metal_colors = colors.to("mps").requires_grad_(True)
    metal_opacities = opacities.to("mps").requires_grad_(True)
    metal_out = gm.rasterize_to_pixels_eval3d(
        metal_means,
        metal_quats,
        metal_scales,
        metal_colors,
        metal_opacities,
        viewmats.to("mps"),
        ks.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds.to("mps"),
        rays=None,
        use_hit_distance=True,
        return_normals=True,
    )
    metal_loss = (
        (metal_out[0] * v_colors.to("mps")).sum()
        + (metal_out[1] * v_alphas.to("mps")).sum()
        + (metal_out[4] * v_normals.to("mps")).sum()
    )
    metal_loss.backward()

    assert torch.allclose(metal_means.grad.cpu(), ref_means.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_quats.grad.cpu(), ref_quats.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_scales.grad.cpu(), ref_scales.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_colors.grad.cpu(), ref_colors.grad, atol=2e-3, rtol=2e-3)
    assert torch.allclose(metal_opacities.grad.cpu(), ref_opacities.grad, atol=2e-3, rtol=2e-3)


def test_rasterize_to_pixels_eval3d_rejects_remaining_unimplemented_features():
    inputs = _sample_inputs(channels=3)
    means, quats, scales, colors, opacities, backgrounds, viewmats, ks, rays, isect_offsets, flatten_ids, width, height, tile_size = inputs

    with pytest.raises(NotImplementedError):
        gm.rasterize_to_pixels_eval3d(
            means.to("mps"),
            quats.to("mps"),
            scales.to("mps"),
            colors.to("mps"),
            opacities.to("mps"),
            viewmats.to("mps"),
            ks.to("mps"),
            width,
            height,
            tile_size,
            isect_offsets.to("mps"),
            flatten_ids.to("mps"),
            backgrounds=backgrounds.to("mps"),
            rays=rays.to("mps"),
            rolling_shutter=gm.RollingShutterType.LINEAR,
        )
