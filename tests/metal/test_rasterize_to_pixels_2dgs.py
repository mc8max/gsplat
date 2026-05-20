# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection_2dgs
from gsplat.metal._math import _isect_offset_encode, _isect_tiles
from gsplat.metal._wrapper import (
    _rasterize_to_pixels_2dgs_absgrad_reference,
    _rasterize_to_pixels_2dgs_median_color_grad,
    _rasterize_to_pixels_2dgs_reference_autograd,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_world_inputs(c=2, n=8, feature_channels=3, width=32, height=24):
    torch.manual_seed(21)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.5 + 1.5
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.2 + 0.15
    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.04, -0.04, c, dtype=torch.float32)
    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 220.0
    Ks[:, 1, 1] = 210.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    features = torch.rand(c, n, feature_channels, dtype=torch.float32)
    opacities = torch.rand(c, n, dtype=torch.float32) * 0.6 + 0.3
    return means, quats, scales, viewmats, Ks, features, opacities


def _prepare_projected_inputs(feature_channels=3, width=32, height=24):
    means, quats, scales, viewmats, Ks, features, opacities = _sample_world_inputs(
        feature_channels=feature_channels, width=width, height=height
    )
    radii, means2d, depths, ray_transforms, normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    colors = torch.cat([features, depths[..., None]], dim=-1)
    backgrounds = torch.zeros(colors.shape[0], colors.shape[-1], dtype=torch.float32)
    tile_size = 8
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    return (
        means2d,
        radii,
        depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    )


def _pack_projected_inputs(
    means2d,
    radii,
    depths,
    ray_transforms,
    normals,
    colors,
    opacities,
):
    image_count, gaussian_count = means2d.shape[:2]
    image_ids = torch.arange(image_count, dtype=torch.int64).repeat_interleave(gaussian_count)
    gaussian_ids = torch.arange(gaussian_count, dtype=torch.int64).repeat(image_count)
    return (
        means2d.reshape(-1, 2),
        radii.reshape(-1, 2),
        depths.reshape(-1),
        ray_transforms.reshape(-1, 3, 3),
        normals.reshape(-1, 3),
        colors.reshape(-1, colors.shape[-1]),
        opacities.reshape(-1),
        image_ids,
        gaussian_ids,
    )


def _rasterize_forward_reference(
    means2d,
    ray_transforms,
    colors,
    opacities,
    normals,
    image_width,
    image_height,
    tile_size,
    isect_offsets,
    flatten_ids,
    backgrounds=None,
    masks=None,
):
    image_dims = tuple(isect_offsets.shape[:-2])
    I = math.prod(image_dims)
    N = means2d.shape[-2]
    channels = colors.shape[-1]
    tile_height, tile_width = isect_offsets.shape[-2:]
    n_isects = int(flatten_ids.numel())

    render_colors = torch.zeros(*image_dims, image_height, image_width, channels, dtype=torch.float32)
    render_alphas = torch.zeros(*image_dims, image_height, image_width, 1, dtype=torch.float32)
    render_normals = torch.zeros(*image_dims, image_height, image_width, 3, dtype=torch.float32)
    render_distort = torch.zeros(*image_dims, image_height, image_width, 1, dtype=torch.float32)
    render_median = torch.zeros(*image_dims, image_height, image_width, 1, dtype=torch.float32)
    last_ids = torch.zeros(*image_dims, image_height, image_width, dtype=torch.int32)
    median_ids = torch.zeros(*image_dims, image_height, image_width, dtype=torch.int32)

    if backgrounds is not None:
        render_colors.copy_(backgrounds.unsqueeze(-2).unsqueeze(-2).expand_as(render_colors))

    offsets_flat = isect_offsets.reshape(I, tile_height * tile_width)
    means_flat = means2d.reshape(-1, 2)
    ray_flat = ray_transforms.reshape(-1, 3, 3)
    color_flat = colors.reshape(-1, channels)
    opac_flat = opacities.reshape(-1)
    normal_flat = normals.reshape(-1, 3)

    for image_id in range(I):
        for tile_y in range(tile_height):
            for tile_x in range(tile_width):
                tile_id = tile_y * tile_width + tile_x
                global_tile = image_id * tile_height * tile_width + tile_id
                range_start = int(offsets_flat[image_id, tile_id].item())
                if global_tile + 1 < I * tile_height * tile_width:
                    next_image = (global_tile + 1) // (tile_height * tile_width)
                    next_tile = (global_tile + 1) % (tile_height * tile_width)
                    range_end = int(offsets_flat[next_image, next_tile].item())
                else:
                    range_end = n_isects
                masked = masks is not None and not bool(
                    masks.reshape(I, tile_height, tile_width)[image_id, tile_y, tile_x].item()
                )
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        i = tile_y * tile_size + local_y
                        j = tile_x * tile_size + local_x
                        if i >= image_height or j >= image_width:
                            continue
                        if masked:
                            continue

                        px = float(j) + 0.5
                        py = float(i) + 0.5
                        T = 1.0
                        accum = torch.zeros(channels, dtype=torch.float32)
                        accum_normal = torch.zeros(3, dtype=torch.float32)
                        distort = 0.0
                        accum_vis_depth = 0.0
                        median_depth = 0.0
                        median_idx = 0
                        cur_idx = 0
                        for idx in range(range_start, range_end):
                            g = int(flatten_ids[idx].item())
                            u_M = ray_flat[g, 0]
                            v_M = ray_flat[g, 1]
                            w_M = ray_flat[g, 2]
                            h_u = px * w_M - u_M
                            h_v = py * w_M - v_M
                            ray_cross = torch.cross(h_u, h_v, dim=-1)
                            if float(ray_cross[2].item()) == 0.0:
                                continue
                            s = ray_cross[:2] / ray_cross[2]
                            gauss_weight_3d = float((s * s).sum().item())
                            d = means_flat[g] - torch.tensor([px, py], dtype=torch.float32)
                            gauss_weight_2d = 2.0 * float((d * d).sum().item())
                            sigma = 0.5 * min(gauss_weight_3d, gauss_weight_2d)
                            alpha = min(0.99, float(opac_flat[g].item()) * math.exp(-sigma))
                            if sigma < 0.0 or alpha < (1.0 / 255.0):
                                continue
                            next_T = T * (1.0 - alpha)
                            if next_T <= 1.0e-4:
                                break

                            vis = alpha * T
                            accum += color_flat[g] * vis
                            accum_normal += normal_flat[g] * vis
                            depth = float(color_flat[g, -1].item())
                            distort += 2.0 * (vis * depth * (1.0 - T) - vis * accum_vis_depth)
                            accum_vis_depth += vis * depth
                            if T > 0.5:
                                median_depth = depth
                                median_idx = idx
                            cur_idx = idx
                            T = next_T

                        if backgrounds is not None:
                            accum += backgrounds.reshape(I, channels)[image_id] * T
                        render_colors.reshape(I, image_height, image_width, channels)[image_id, i, j] = accum
                        render_alphas.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = 1.0 - T
                        render_normals.reshape(I, image_height, image_width, 3)[image_id, i, j] = accum_normal
                        render_distort.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = distort
                        render_median.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = median_depth
                        last_ids.reshape(I, image_height, image_width)[image_id, i, j] = cur_idx
                        median_ids.reshape(I, image_height, image_width)[image_id, i, j] = median_idx

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids,
    )


def test_forward_matches_reference():
    width, height = 32, 24
    (
        means2d,
        _radii,
        _depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=3, width=width, height=height)

    expected = _rasterize_forward_reference(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    (
        expected_colors,
        expected_alphas,
        expected_normals,
        expected_distort,
        expected_median,
        expected_last_ids,
        expected_median_ids,
    ) = expected

    actual = gm.rasterize_to_pixels_2dgs(
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
        backgrounds=backgrounds.to("mps"),
    )
    actual_colors, actual_alphas, actual_normals, actual_distort, actual_median = actual
    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_normals.cpu(), expected_normals, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_distort.cpu(), expected_distort, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_median.cpu(), expected_median, atol=1e-4, rtol=1e-4)

    native = torch.ops.gsplat.metal_rasterize_to_pixels_2dgs_fwd(
        means2d.to("mps"),
        ray_transforms.contiguous().to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        normals.contiguous().to("mps"),
        backgrounds.to("mps"),
        None,
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        False,
    )
    _, _, _, _, _, actual_last_ids, actual_median_ids = native
    torch.testing.assert_close(actual_last_ids.cpu(), expected_last_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_median_ids.cpu(), expected_median_ids, atol=0, rtol=0)


def test_padded_channels_match_reference():
    width, height = 28, 20
    (
        means2d,
        _radii,
        _depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=30, width=width, height=height)

    (
        expected_colors,
        expected_alphas,
        expected_normals,
        _expected_distort,
        _expected_median,
        _expected_last_ids,
        _expected_median_ids,
    ) = _rasterize_forward_reference(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    actual_colors, actual_alphas, actual_normals, _, _ = gm.rasterize_to_pixels_2dgs(
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
        backgrounds=backgrounds.to("mps"),
    )

    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_normals.cpu(), expected_normals, atol=1e-4, rtol=1e-4)


def test_backward_matches_reference():
    width, height = 24, 16
    (
        means2d,
        _radii,
        _depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=3, width=width, height=height)
    (
        _exp_colors,
        _exp_alphas,
        _exp_normals,
        _exp_distort,
        _exp_median,
        _last_ids,
        median_ids,
    ) = _rasterize_forward_reference(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    means2d_ref = means2d.clone().requires_grad_(True)
    ray_transforms_ref = ray_transforms.clone().requires_grad_(True)
    colors_ref = colors.clone().requires_grad_(True)
    opacities_ref = opacities.clone().requires_grad_(True)
    normals_ref = normals.clone().requires_grad_(True)
    backgrounds_ref = backgrounds.clone().requires_grad_(True)
    ref_outputs = _rasterize_to_pixels_2dgs_reference_autograd(
        means2d_ref,
        ray_transforms_ref,
        colors_ref,
        opacities_ref,
        normals_ref,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds_ref,
    )
    ref_colors, ref_alphas, ref_normals, ref_distort = ref_outputs
    v_render_colors = torch.randn_like(ref_colors)
    v_render_alphas = torch.randn_like(ref_alphas)
    v_render_normals = torch.randn_like(ref_normals)
    v_render_distort = torch.randn_like(ref_distort)
    v_render_median = torch.randn_like(ref_distort)

    ref_grads = list(
        torch.autograd.grad(
            (ref_colors * v_render_colors).sum()
            + (ref_alphas * v_render_alphas).sum()
            + (ref_normals * v_render_normals).sum()
            + (ref_distort * v_render_distort).sum(),
            (
                means2d_ref,
                ray_transforms_ref,
                colors_ref,
                opacities_ref,
                normals_ref,
                backgrounds_ref,
            ),
        )
    )
    ref_grads[2] = ref_grads[2] + _rasterize_to_pixels_2dgs_median_color_grad(
        v_render_median,
        median_ids,
        flatten_ids,
        colors_ref.shape,
    )
    ref_densify = torch.zeros_like(means2d)
    ref_densify[..., 0] = ref_grads[1][..., 0, 2] * ray_transforms[..., 2, 2]
    ref_densify[..., 1] = ref_grads[1][..., 1, 2] * ray_transforms[..., 2, 2]

    means2d_mps = means2d.to("mps").requires_grad_(True)
    ray_transforms_mps = ray_transforms.to("mps").requires_grad_(True)
    colors_mps = colors.to("mps").requires_grad_(True)
    opacities_mps = opacities.to("mps").requires_grad_(True)
    normals_mps = normals.to("mps").requires_grad_(True)
    densify_mps = torch.zeros_like(means2d).to("mps").requires_grad_(True)
    backgrounds_mps = backgrounds.to("mps").requires_grad_(True)
    outputs = gm.rasterize_to_pixels_2dgs(
        means2d_mps,
        ray_transforms_mps,
        colors_mps,
        opacities_mps,
        normals_mps,
        densify_mps,
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds_mps,
    )
    grads = torch.autograd.grad(
        (outputs[0] * v_render_colors.to("mps")).sum()
        + (outputs[1] * v_render_alphas.to("mps")).sum()
        + (outputs[2] * v_render_normals.to("mps")).sum()
        + (outputs[3] * v_render_distort.to("mps")).sum()
        + (outputs[4] * v_render_median.to("mps")).sum(),
        (
            means2d_mps,
            ray_transforms_mps,
            colors_mps,
            opacities_mps,
            normals_mps,
            densify_mps,
            backgrounds_mps,
        ),
    )

    torch.testing.assert_close(grads[0].cpu(), ref_grads[0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grads[1].cpu(), ref_grads[1], atol=5e-2, rtol=2e-1)
    torch.testing.assert_close(grads[2].cpu(), ref_grads[2], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grads[3].cpu(), ref_grads[3], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grads[4].cpu(), ref_grads[4], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grads[5].cpu(), ref_densify, atol=5e-2, rtol=2e-1)
    torch.testing.assert_close(grads[6].cpu(), ref_grads[5], atol=1e-5, rtol=1e-5)


def test_absgrad_matches_reference():
    width, height = 8, 8
    (
        means2d,
        _radii,
        _depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=3, width=width, height=height)

    means2d_mps = means2d.to("mps").requires_grad_(True)
    ray_transforms_mps = ray_transforms.to("mps").requires_grad_(True)
    colors_mps = colors.to("mps").requires_grad_(True)
    opacities_mps = opacities.to("mps").requires_grad_(True)
    normals_mps = normals.to("mps").requires_grad_(True)
    densify_mps = torch.zeros_like(means2d).to("mps").requires_grad_(True)
    backgrounds_mps = backgrounds.to("mps").requires_grad_(True)
    outputs = gm.rasterize_to_pixels_2dgs(
        means2d_mps,
        ray_transforms_mps,
        colors_mps,
        opacities_mps,
        normals_mps,
        densify_mps,
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds_mps,
        absgrad=True,
    )
    v_render_colors = torch.randn_like(outputs[0].cpu())
    v_render_alphas = torch.randn_like(outputs[1].cpu())
    v_render_normals = torch.randn_like(outputs[2].cpu())
    v_render_distort = torch.randn_like(outputs[3].cpu())
    loss = (
        (outputs[0] * v_render_colors.to("mps")).sum()
        + (outputs[1] * v_render_alphas.to("mps")).sum()
        + (outputs[2] * v_render_normals.to("mps")).sum()
        + (outputs[3] * v_render_distort.to("mps")).sum()
    )
    loss.backward()

    expected_absgrad = _rasterize_to_pixels_2dgs_absgrad_reference(
        means2d.clone().requires_grad_(True),
        ray_transforms.clone().requires_grad_(True),
        colors.clone().requires_grad_(True),
        opacities.clone().requires_grad_(True),
        normals.clone().requires_grad_(True),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        v_render_colors,
        v_render_alphas,
        v_render_normals,
        v_render_distort,
        backgrounds=backgrounds.clone().requires_grad_(True),
    )
    torch.testing.assert_close(means2d_mps.absgrad.cpu(), expected_absgrad, atol=1e-3, rtol=1e-3)


def test_pipeline_matches_reference():
    width, height = 30, 22
    tile_size = 8
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    (
        means,
        quats,
        scales,
        viewmats,
        Ks,
        features,
        opacities,
    ) = _sample_world_inputs(c=3, n=10, feature_channels=4, width=width, height=height)

    radii_ref, means2d_ref, depths_ref, ray_transforms_ref, normals_ref = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    colors_ref = torch.cat([features, depths_ref[..., None]], dim=-1)
    backgrounds = torch.zeros(colors_ref.shape[0], colors_ref.shape[-1], dtype=torch.float32)
    _, isect_ids_ref, flatten_ids_ref = _isect_tiles(
        means2d_ref, radii_ref, depths_ref, tile_size, tile_width, tile_height
    )
    offsets_ref = _isect_offset_encode(isect_ids_ref, viewmats.shape[0], tile_width, tile_height)
    expected = _rasterize_forward_reference(
        means2d_ref,
        ray_transforms_ref,
        colors_ref,
        opacities,
        normals_ref,
        width,
        height,
        tile_size,
        offsets_ref,
        flatten_ids_ref,
        backgrounds=backgrounds,
    )
    expected_colors, expected_alphas, expected_normals, expected_distort, expected_median, _, _ = expected

    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
    )
    colors = torch.cat([features.to("mps"), depths[..., None]], dim=-1)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
    )
    offsets = gm.intersect_offset_encode(isect_ids, viewmats.shape[0], tile_width, tile_height)
    actual = gm.rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities.to("mps"),
        normals,
        torch.zeros_like(means2d),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds=backgrounds.to("mps"),
    )
    actual_colors, actual_alphas, actual_normals, actual_distort, actual_median = actual

    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_normals.cpu(), expected_normals, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_distort.cpu(), expected_distort, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_median.cpu(), expected_median, atol=2e-4, rtol=2e-4)


def test_large_masked_smoke_matches_reference():
    width, height = 48, 32
    (
        means2d,
        _radii,
        _depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=15, width=width, height=height)

    masks = torch.ones_like(isect_offsets, dtype=torch.bool)
    masks[..., 0, 1] = False
    masks[..., -1, -1] = False

    expected = _rasterize_forward_reference(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
    )
    expected_colors, expected_alphas, expected_normals, expected_distort, expected_median, _, _ = expected

    actual = gm.rasterize_to_pixels_2dgs(
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
        backgrounds=backgrounds.to("mps"),
        masks=masks.to("mps"),
    )
    actual_colors, actual_alphas, actual_normals, actual_distort, actual_median = actual

    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_normals.cpu(), expected_normals, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_distort.cpu(), expected_distort, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_median.cpu(), expected_median, atol=2e-4, rtol=2e-4)


def test_packed_forward_matches_dense_reference():
    width, height = 30, 22
    (
        means2d,
        radii,
        depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=6, width=width, height=height)

    expected = _rasterize_forward_reference(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    expected_colors, expected_alphas, expected_normals, expected_distort, expected_median, _, _ = expected

    (
        means2d_packed,
        radii_packed,
        depths_packed,
        ray_transforms_packed,
        normals_packed,
        colors_packed,
        opacities_packed,
        image_ids,
        gaussian_ids,
    ) = _pack_projected_inputs(
        means2d, radii, depths, ray_transforms, normals, colors, opacities
    )
    tile_width = isect_offsets.shape[-1]
    tile_height = isect_offsets.shape[-2]
    _, isect_ids, packed_flatten_ids = gm.intersect_tiles(
        means2d_packed.to("mps"),
        radii_packed.to("mps"),
        depths_packed.to("mps"),
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=means2d.shape[0],
        image_ids=image_ids.to("mps"),
        gaussian_ids=gaussian_ids.to("mps"),
    )
    packed_offsets = gm.intersect_offset_encode(
        isect_ids, means2d.shape[0], tile_width, tile_height
    )

    actual = gm.rasterize_to_pixels_2dgs(
        means2d_packed.to("mps"),
        ray_transforms_packed.to("mps"),
        colors_packed.to("mps"),
        opacities_packed.to("mps"),
        normals_packed.to("mps"),
        torch.zeros_like(means2d_packed).to("mps"),
        width,
        height,
        tile_size,
        packed_offsets,
        packed_flatten_ids,
        backgrounds=backgrounds.to("mps"),
        packed=True,
    )
    actual_colors, actual_alphas, actual_normals, actual_distort, actual_median = actual

    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_normals.cpu(), expected_normals, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_distort.cpu(), expected_distort, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_median.cpu(), expected_median, atol=2e-4, rtol=2e-4)


def test_packed_backward_matches_dense_reference():
    width, height = 24, 16
    (
        means2d,
        radii,
        depths,
        ray_transforms,
        normals,
        colors,
        opacities,
        backgrounds,
        tile_size,
        isect_offsets,
        flatten_ids,
    ) = _prepare_projected_inputs(feature_channels=3, width=width, height=height)

    means2d_dense = means2d.to("mps").requires_grad_(True)
    ray_dense = ray_transforms.to("mps").requires_grad_(True)
    colors_dense = colors.to("mps").requires_grad_(True)
    opac_dense = opacities.to("mps").requires_grad_(True)
    normals_dense = normals.to("mps").requires_grad_(True)
    densify_dense = torch.zeros_like(means2d).to("mps").requires_grad_(True)
    backgrounds_dense = backgrounds.to("mps").requires_grad_(True)
    dense_outputs = gm.rasterize_to_pixels_2dgs(
        means2d_dense,
        ray_dense,
        colors_dense,
        opac_dense,
        normals_dense,
        densify_dense,
        width,
        height,
        tile_size,
        isect_offsets.to("mps"),
        flatten_ids.to("mps"),
        backgrounds=backgrounds_dense,
    )

    (
        means2d_packed,
        radii_packed,
        depths_packed,
        ray_transforms_packed,
        normals_packed,
        colors_packed,
        opacities_packed,
        image_ids,
        gaussian_ids,
    ) = _pack_projected_inputs(
        means2d, radii, depths, ray_transforms, normals, colors, opacities
    )
    tile_width = isect_offsets.shape[-1]
    tile_height = isect_offsets.shape[-2]
    _, isect_ids, packed_flatten_ids = gm.intersect_tiles(
        means2d_packed.to("mps"),
        radii_packed.to("mps"),
        depths_packed.to("mps"),
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=means2d.shape[0],
        image_ids=image_ids.to("mps"),
        gaussian_ids=gaussian_ids.to("mps"),
    )
    packed_offsets = gm.intersect_offset_encode(
        isect_ids, means2d.shape[0], tile_width, tile_height
    )

    means2d_pack = means2d_packed.to("mps").requires_grad_(True)
    ray_pack = ray_transforms_packed.to("mps").requires_grad_(True)
    colors_pack = colors_packed.to("mps").requires_grad_(True)
    opac_pack = opacities_packed.to("mps").requires_grad_(True)
    normals_pack = normals_packed.to("mps").requires_grad_(True)
    densify_pack = torch.zeros_like(means2d_packed).to("mps").requires_grad_(True)
    backgrounds_pack = backgrounds.to("mps").requires_grad_(True)
    packed_outputs = gm.rasterize_to_pixels_2dgs(
        means2d_pack,
        ray_pack,
        colors_pack,
        opac_pack,
        normals_pack,
        densify_pack,
        width,
        height,
        tile_size,
        packed_offsets,
        packed_flatten_ids,
        backgrounds=backgrounds_pack,
        packed=True,
    )

    torch.manual_seed(123)
    v_render_colors = torch.randn_like(dense_outputs[0])
    v_render_alphas = torch.randn_like(dense_outputs[1])
    v_render_normals = torch.randn_like(dense_outputs[2])
    v_render_distort = torch.randn_like(dense_outputs[3])
    v_render_median = torch.randn_like(dense_outputs[4])

    dense_grads = torch.autograd.grad(
        (dense_outputs[0] * v_render_colors).sum()
        + (dense_outputs[1] * v_render_alphas).sum()
        + (dense_outputs[2] * v_render_normals).sum()
        + (dense_outputs[3] * v_render_distort).sum()
        + (dense_outputs[4] * v_render_median).sum(),
        (
            means2d_dense,
            ray_dense,
            colors_dense,
            opac_dense,
            normals_dense,
            densify_dense,
            backgrounds_dense,
        ),
    )
    packed_grads = torch.autograd.grad(
        (packed_outputs[0] * v_render_colors).sum()
        + (packed_outputs[1] * v_render_alphas).sum()
        + (packed_outputs[2] * v_render_normals).sum()
        + (packed_outputs[3] * v_render_distort).sum()
        + (packed_outputs[4] * v_render_median).sum(),
        (
            means2d_pack,
            ray_pack,
            colors_pack,
            opac_pack,
            normals_pack,
            densify_pack,
            backgrounds_pack,
        ),
    )

    torch.testing.assert_close(packed_grads[0].reshape_as(dense_grads[0]).cpu(), dense_grads[0].cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(packed_grads[1].reshape_as(dense_grads[1]).cpu(), dense_grads[1].cpu(), atol=5e-2, rtol=2e-1)
    torch.testing.assert_close(packed_grads[2].reshape_as(dense_grads[2]).cpu(), dense_grads[2].cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(packed_grads[3].reshape_as(dense_grads[3]).cpu(), dense_grads[3].cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(packed_grads[4].reshape_as(dense_grads[4]).cpu(), dense_grads[4].cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(packed_grads[5].reshape_as(dense_grads[5]).cpu(), dense_grads[5].cpu(), atol=5e-2, rtol=2e-1)
    torch.testing.assert_close(packed_grads[6].cpu(), dense_grads[6].cpu(), atol=1e-5, rtol=1e-5)
