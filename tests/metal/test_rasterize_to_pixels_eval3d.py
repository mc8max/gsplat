# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm
from gsplat._camera_types import RollingShutterType
from gsplat.cuda._math import _quat_to_rotmat, _safe_normalize
from gsplat.cuda._torch_cameras import (
    _BaseCameraModel,
    _interpolate_shutter_pose,
    _pose_camera_ray_to_world_ray,
    _viewmat_to_pose,
)
from gsplat.cuda._torch_external_distortion import make_params
from gsplat.metal._math import _fully_fused_projection, _isect_offset_encode, _isect_tiles

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


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

    radii, means2d, depths, _conics, _comp = _fully_fused_projection(
        means,
        None,
        quats,
        scales,
        None,
        viewmats,
        Ks,
        width,
        height,
        0.3,
        0.01,
        1e10,
        0.0,
        False,
        "pinhole",
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
    c = colors.shape[0]
    n = means.shape[0]
    channels = colors.shape[-1]
    tile_height, tile_width = isect_offsets.shape[-2:]
    n_isects = int(flatten_ids.numel())
    offsets_flat = isect_offsets.reshape(c * tile_height * tile_width)

    render_colors = torch.zeros(c, image_height, image_width, channels, dtype=means.dtype)
    render_alphas = torch.zeros(c, image_height, image_width, 1, dtype=means.dtype)
    last_ids = torch.full((c, image_height, image_width), -1, dtype=torch.int32)
    sample_counts = torch.zeros(c, image_height, image_width, dtype=torch.int32)
    render_normals = (
        torch.zeros(c, image_height, image_width, 3, dtype=means.dtype)
        if return_normals
        else None
    )

    for image_id in range(c):
        for tile_y in range(tile_height):
            for tile_x in range(tile_width):
                tile_id = tile_y * tile_width + tile_x
                global_tile = image_id * tile_height * tile_width + tile_id
                range_start = int(offsets_flat[global_tile].item())
                range_end = (
                    int(offsets_flat[global_tile + 1].item())
                    if global_tile + 1 < c * tile_height * tile_width
                    else n_isects
                )
                masked = masks is not None and not bool(masks[image_id, tile_y, tile_x].item())
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        py = tile_y * tile_size + local_y
                        px = tile_x * tile_size + local_x
                        if py >= image_height or px >= image_width:
                            continue
                        if masked:
                            if backgrounds is not None:
                                render_colors[image_id, py, px] = backgrounds[image_id]
                            continue

                        ray_o = rays[image_id, py, px, :3]
                        ray_d = rays[image_id, py, px, 3:]
                        T = torch.tensor(1.0, dtype=means.dtype)
                        accum = torch.zeros(channels, dtype=means.dtype)
                        normal_accum = torch.zeros(3, dtype=means.dtype)
                        cur_idx = -1
                        count = 0

                        for idx in range(range_start, range_end):
                            isect_id = int(flatten_ids[idx].item())
                            gaussian_id = isect_id % n
                            xyz = means[gaussian_id]
                            quat = quats[gaussian_id]
                            scale = scales[gaussian_id]
                            R = _quat_to_rotmat(quat)
                            Mt = torch.diag(1.0 / scale) @ R.transpose(0, 1)
                            gro = Mt @ (ray_o - xyz)
                            grd = _safe_normalize(Mt @ ray_d)
                            gcrod = torch.cross(grd, gro, dim=0)
                            gray_dist = torch.sum(gcrod * gcrod)
                            power = -0.5 * gray_dist
                            max_response = torch.exp(power)
                            alpha = torch.minimum(
                                torch.tensor(0.99, dtype=means.dtype),
                                opacities[image_id, gaussian_id] * max_response,
                            )
                            if alpha.detach().item() < (1.0 / 255.0) or max_response.detach().item() <= 0.0113:
                                continue

                            next_T = T * (1.0 - alpha)
                            if next_T.detach().item() <= 1.0e-4:
                                break

                            vis = alpha * T
                            if use_hit_distance:
                                hit_t = torch.sum(grd * (-gro))
                                grds = scale * (grd * hit_t)
                                hit_distance = torch.linalg.norm(grds)
                                sample_value = colors[image_id, gaussian_id].clone()
                                sample_value[-1] = hit_distance
                            else:
                                sample_value = colors[image_id, gaussian_id]

                            accum = accum + sample_value * vis
                            if return_normals:
                                unnormalized_normal = R[:, 2]
                                flipped = torch.sum(unnormalized_normal * ray_d) > 0
                                normal = _safe_normalize(
                                    -unnormalized_normal if flipped else unnormalized_normal
                                )
                                normal_accum = normal_accum + normal * vis
                            T = next_T
                            cur_idx = idx
                            count += 1

                        if backgrounds is not None:
                            accum = accum + backgrounds[image_id] * T
                        render_colors[image_id, py, px] = accum
                        render_alphas[image_id, py, px, 0] = 1.0 - T
                        last_ids[image_id, py, px] = cur_idx
                        sample_counts[image_id, py, px] = count
                        if render_normals is not None:
                            render_normals[image_id, py, px] = normal_accum

    return render_colors, render_alphas, last_ids, sample_counts, render_normals


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
    ext = make_params(
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
