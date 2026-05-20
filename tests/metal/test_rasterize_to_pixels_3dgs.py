# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import (
    _fully_fused_projection,
    _isect_offset_encode,
    _isect_tiles,
    _quat_scale_to_covar_preci,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _rasterize_reference(
    means2d,
    conics,
    colors,
    opacities,
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
    tile_height, tile_width = isect_offsets.shape[-2:]
    channels = colors.shape[-1]
    n_isects = int(flatten_ids.numel())

    if means2d.dim() == 2:
        means_flat = means2d.reshape(-1, 2)
        conics_flat = conics.reshape(-1, 3)
        colors_flat = colors.reshape(-1, channels)
        opacities_flat = opacities.reshape(-1)
    else:
        means_flat = means2d.reshape(-1, 2)
        conics_flat = conics.reshape(-1, 3)
        colors_flat = colors.reshape(-1, channels)
        opacities_flat = opacities.reshape(-1)

    render_colors = torch.zeros(*image_dims, image_height, image_width, channels, dtype=torch.float32)
    render_alphas = torch.zeros(*image_dims, image_height, image_width, 1, dtype=torch.float32)
    last_ids = torch.zeros(*image_dims, image_height, image_width, dtype=torch.int32)

    if backgrounds is not None:
        render_colors.copy_(backgrounds.unsqueeze(-2).unsqueeze(-2).expand_as(render_colors))

    offsets_flat = isect_offsets.reshape(I, tile_height * tile_width)
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

                masked = masks is not None and not bool(masks.reshape(I, tile_height, tile_width)[image_id, tile_y, tile_x].item())
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        i = tile_y * tile_size + local_y
                        j = tile_x * tile_size + local_x
                        if i >= image_height or j >= image_width:
                            continue
                        if masked:
                            render_alphas.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = 0.0
                            last_ids.reshape(I, image_height, image_width)[image_id, i, j] = 0
                            continue

                        px = float(j) + 0.5
                        py = float(i) + 0.5
                        T = 1.0
                        accum = torch.zeros(channels, dtype=torch.float32)
                        cur_idx = 0
                        for idx in range(range_start, range_end):
                            g = int(flatten_ids[idx].item())
                            delta_x = float(means_flat[g, 0].item()) - px
                            delta_y = float(means_flat[g, 1].item()) - py
                            conic = conics_flat[g]
                            sigma = (
                                0.5 * (float(conic[0].item()) * delta_x * delta_x + float(conic[2].item()) * delta_y * delta_y)
                                + float(conic[1].item()) * delta_x * delta_y
                            )
                            alpha = min(0.99, float(opacities_flat[g].item()) * math.exp(-sigma))
                            if sigma < 0.0 or alpha < (1.0 / 255.0):
                                continue
                            next_T = T * (1.0 - alpha)
                            if next_T <= 1.0e-4:
                                break
                            accum += colors_flat[g] * (alpha * T)
                            cur_idx = idx
                            T = next_T

                        if backgrounds is not None and not masked:
                            accum += (
                                backgrounds.reshape(I, channels)[image_id] * T
                            )
                        render_colors.reshape(I, image_height, image_width, channels)[image_id, i, j] = accum
                        render_alphas.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = 1.0 - T
                        last_ids.reshape(I, image_height, image_width)[image_id, i, j] = cur_idx

    return render_colors, render_alphas, last_ids


def _rasterize_reference_autograd(
    means2d,
    conics,
    colors,
    opacities,
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
    tile_height, tile_width = isect_offsets.shape[-2:]
    channels = colors.shape[-1]
    n_isects = int(flatten_ids.numel())
    device = means2d.device
    dtype = means2d.dtype

    means_flat = means2d.reshape(-1, 2)
    conics_flat = conics.reshape(-1, 3)
    colors_flat = colors.reshape(-1, channels)
    opacities_flat = opacities.reshape(-1)

    render_colors = torch.zeros(*image_dims, image_height, image_width, channels, dtype=dtype, device=device)
    render_alphas = torch.zeros(*image_dims, image_height, image_width, 1, dtype=dtype, device=device)
    if backgrounds is not None:
        render_colors = render_colors + backgrounds.unsqueeze(-2).unsqueeze(-2)

    offsets_flat = isect_offsets.reshape(I, tile_height * tile_width)
    alpha_threshold = torch.tensor(1.0 / 255.0, dtype=dtype, device=device)
    max_alpha = torch.tensor(0.99, dtype=dtype, device=device)
    trans_thresh = 1.0e-4
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

                        px = torch.tensor(float(j) + 0.5, dtype=dtype, device=device)
                        py = torch.tensor(float(i) + 0.5, dtype=dtype, device=device)
                        T = torch.tensor(1.0, dtype=dtype, device=device)
                        accum = torch.zeros(channels, dtype=dtype, device=device)
                        for idx in range(range_start, range_end):
                            g = int(flatten_ids[idx].item())
                            delta_x = means_flat[g, 0] - px
                            delta_y = means_flat[g, 1] - py
                            conic = conics_flat[g]
                            sigma = (
                                0.5 * (conic[0] * delta_x * delta_x + conic[2] * delta_y * delta_y)
                                + conic[1] * delta_x * delta_y
                            )
                            alpha = torch.minimum(max_alpha, opacities_flat[g] * torch.exp(-sigma))
                            if sigma.detach().item() < 0.0 or alpha.detach().item() < alpha_threshold.item():
                                continue
                            next_T = T * (1.0 - alpha)
                            if next_T.detach().item() <= trans_thresh:
                                break
                            accum = accum + colors_flat[g] * (alpha * T)
                            T = next_T

                        if backgrounds is not None:
                            accum = accum + backgrounds.reshape(I, channels)[image_id] * T
                        render_colors.reshape(I, image_height, image_width, channels)[image_id, i, j] = accum
                        render_alphas.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = 1.0 - T

    return render_colors, render_alphas


def _sample_inputs(image_count=2, n=10, channels=7, width=32, height=24):
    torch.manual_seed(42)
    means2d = torch.rand(image_count, n, 2, dtype=torch.float32)
    means2d[..., 0] *= width - 1
    means2d[..., 1] *= height - 1

    radii = torch.randint(1, 4, (image_count, n, 2), dtype=torch.int32)
    depths = torch.rand(image_count, n, dtype=torch.float32)
    conics = torch.zeros(image_count, n, 3, dtype=torch.float32)
    conics[..., 0] = torch.rand(image_count, n) * 0.25 + 0.15
    conics[..., 1] = (torch.rand(image_count, n) - 0.5) * 0.05
    conics[..., 2] = torch.rand(image_count, n) * 0.25 + 0.15
    colors = torch.rand(image_count, n, channels, dtype=torch.float32)
    opacities = torch.rand(image_count, n, dtype=torch.float32) * 0.7 + 0.2
    backgrounds = torch.rand(image_count, channels, dtype=torch.float32)
    return means2d, radii, depths, conics, colors, opacities, backgrounds


def _sample_world_inputs(c=2, n=8, channels=5, width=32, height=24):
    torch.manual_seed(7)
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
    colors = torch.rand(c, n, channels, dtype=torch.float32)
    opacities = torch.rand(n, dtype=torch.float32) * 0.6 + 0.3
    backgrounds = torch.rand(c, channels, dtype=torch.float32)
    return means, quats, scales, viewmats, Ks, colors, opacities, backgrounds


def _absgrad_reference(
    means2d,
    conics,
    colors,
    opacities,
    image_width,
    image_height,
    tile_size,
    isect_offsets,
    flatten_ids,
    v_render_colors,
    v_render_alphas,
    backgrounds=None,
    masks=None,
):
    means2d_ref = means2d.clone().requires_grad_(True)
    conics_ref = conics.clone().requires_grad_(True)
    colors_ref = colors.clone().requires_grad_(True)
    opacities_ref = opacities.clone().requires_grad_(True)
    backgrounds_ref = backgrounds.clone().requires_grad_(True) if backgrounds is not None else None
    render_colors, render_alphas = _rasterize_reference_autograd(
        means2d_ref,
        conics_ref,
        colors_ref,
        opacities_ref,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds_ref,
        masks=masks,
    )

    absgrad = torch.zeros_like(means2d_ref)
    image_dims = render_alphas.shape[:-3]
    shape = (*image_dims, image_height, image_width)
    for flat_idx in range(math.prod(shape)):
        unravel = []
        rem = flat_idx
        for size in reversed(shape):
            unravel.append(rem % size)
            rem //= size
        index_tuple = tuple(reversed(unravel))
        color_scalar = (render_colors[index_tuple] * v_render_colors[index_tuple]).sum()
        alpha_scalar = (
            render_alphas[index_tuple + (0,)] * v_render_alphas[index_tuple + (0,)]
        ).sum()
        pixel_grad = torch.autograd.grad(
            color_scalar + alpha_scalar,
            means2d_ref,
            retain_graph=True,
        )[0]
        absgrad = absgrad + pixel_grad.abs()
    return absgrad


def test_unpacked_forward_matches_reference(mps_device):
    width, height = 32, 24
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        channels=7, width=width, height=height
    )

    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    expected_colors, expected_alphas, expected_last_ids = _rasterize_reference(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    actual_colors, actual_alphas = gm.rasterize_to_pixels(
        means2d.to(mps_device),
        conics.to(mps_device),
        colors.to(mps_device),
        opacities.to(mps_device),
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds.to(mps_device),
    )
    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=1e-5, rtol=1e-5)

    assert gm.has_metal()
    _, _, actual_last_ids = torch.ops.gsplat.metal_rasterize_to_pixels_3dgs_fwd(
        means2d.to(mps_device),
        conics.to(mps_device),
        torch.cat(
            [colors, torch.zeros(*colors.shape[:-1], 1, dtype=colors.dtype)], dim=-1
        ).to(mps_device),
        opacities.to(mps_device),
        torch.cat(
            [backgrounds, torch.zeros(*backgrounds.shape[:-1], 1, dtype=backgrounds.dtype)], dim=-1
        ).to(mps_device),
        None,
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
    )
    torch.testing.assert_close(actual_last_ids.cpu(), expected_last_ids, atol=0, rtol=0)


def test_packed_forward_matches_unpacked_reference(mps_device):
    width, height = 28, 20
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        channels=5, width=width, height=height
    )
    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)

    expected_colors, expected_alphas, _ = _rasterize_reference(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    actual_colors, actual_alphas = gm.rasterize_to_pixels(
        means2d.reshape(-1, 2).to(mps_device),
        conics.reshape(-1, 3).to(mps_device),
        colors.reshape(-1, colors.shape[-1]).to(mps_device),
        opacities.reshape(-1).to(mps_device),
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds.to(mps_device),
        packed=True,
    )
    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=1e-5, rtol=1e-5)


def test_masked_tiles_render_background_only(mps_device):
    width, height = 24, 16
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        image_count=1, channels=3, width=width, height=height
    )
    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    masks = torch.zeros_like(isect_offsets, dtype=torch.bool)

    actual_colors, actual_alphas = gm.rasterize_to_pixels(
        means2d.to(mps_device),
        conics.to(mps_device),
        colors.to(mps_device),
        opacities.to(mps_device),
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds.to(mps_device),
        masks=masks.to(mps_device),
    )

    expected_colors = backgrounds[:, None, None, :].expand_as(actual_colors.cpu())
    expected_alphas = torch.zeros_like(actual_alphas.cpu())
    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=0, rtol=0)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=0, rtol=0)


def test_empty_intersections_return_background_or_zero(mps_device):
    width, height = 16, 12
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    means2d = torch.zeros(2, 3, 2, dtype=torch.float32)
    conics = torch.zeros(2, 3, 3, dtype=torch.float32)
    colors = torch.rand(2, 3, 3, dtype=torch.float32)
    opacities = torch.zeros(2, 3, dtype=torch.float32)
    isect_offsets = torch.zeros(2, tile_height, tile_width, dtype=torch.int32)
    flatten_ids = torch.zeros(0, dtype=torch.int32)
    backgrounds = torch.rand(2, 3, dtype=torch.float32)

    actual_colors, actual_alphas = gm.rasterize_to_pixels(
        means2d.to(mps_device),
        conics.to(mps_device),
        colors.to(mps_device),
        opacities.to(mps_device),
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds.to(mps_device),
    )
    torch.testing.assert_close(
        actual_colors.cpu(),
        backgrounds[:, None, None, :].expand_as(actual_colors.cpu()),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(actual_alphas.cpu(), torch.zeros_like(actual_alphas.cpu()), atol=0, rtol=0)


def test_backward_matches_reference_unpacked(mps_device):
    width, height = 24, 16
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        image_count=2, n=6, channels=3, width=width, height=height
    )
    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)

    means2d_ref = means2d.clone().requires_grad_(True)
    conics_ref = conics.clone().requires_grad_(True)
    colors_ref = colors.clone().requires_grad_(True)
    opacities_ref = opacities.clone().requires_grad_(True)
    backgrounds_ref = backgrounds.clone().requires_grad_(True)
    ref_colors, ref_alphas = _rasterize_reference_autograd(
        means2d_ref,
        conics_ref,
        colors_ref,
        opacities_ref,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds_ref,
    )
    v_render_colors = torch.randn_like(ref_colors)
    v_render_alphas = torch.randn_like(ref_alphas)
    ref_grads = torch.autograd.grad(
        (ref_colors * v_render_colors).sum() + (ref_alphas * v_render_alphas).sum(),
        (means2d_ref, conics_ref, colors_ref, opacities_ref, backgrounds_ref),
    )

    means2d_mps = means2d.clone().to(mps_device).requires_grad_(True)
    conics_mps = conics.clone().to(mps_device).requires_grad_(True)
    colors_mps = colors.clone().to(mps_device).requires_grad_(True)
    opacities_mps = opacities.clone().to(mps_device).requires_grad_(True)
    backgrounds_mps = backgrounds.clone().to(mps_device).requires_grad_(True)
    out_colors, out_alphas = gm.rasterize_to_pixels(
        means2d_mps,
        conics_mps,
        colors_mps,
        opacities_mps,
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds_mps,
    )
    grads = torch.autograd.grad(
        (out_colors * v_render_colors.to(mps_device)).sum()
        + (out_alphas * v_render_alphas.to(mps_device)).sum(),
        (means2d_mps, conics_mps, colors_mps, opacities_mps, backgrounds_mps),
    )
    for actual, expected, atol, rtol in (
        (grads[0], ref_grads[0], 2e-4, 2e-4),
        (grads[1], ref_grads[1], 2e-4, 2e-4),
        (grads[2], ref_grads[2], 2e-4, 2e-4),
        (grads[3], ref_grads[3], 2e-4, 2e-4),
        (grads[4], ref_grads[4], 2e-4, 2e-4),
    ):
        torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


def test_backward_matches_reference_packed(mps_device):
    width, height = 20, 16
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        image_count=2, n=5, channels=4, width=width, height=height
    )
    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)

    means2d_ref = means2d.reshape(-1, 2).clone().requires_grad_(True)
    conics_ref = conics.reshape(-1, 3).clone().requires_grad_(True)
    colors_ref = colors.reshape(-1, colors.shape[-1]).clone().requires_grad_(True)
    opacities_ref = opacities.reshape(-1).clone().requires_grad_(True)
    backgrounds_ref = backgrounds.clone().requires_grad_(True)
    ref_colors, ref_alphas = _rasterize_reference_autograd(
        means2d_ref,
        conics_ref,
        colors_ref,
        opacities_ref,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds_ref,
    )
    v_render_colors = torch.randn_like(ref_colors)
    v_render_alphas = torch.randn_like(ref_alphas)
    ref_grads = torch.autograd.grad(
        (ref_colors * v_render_colors).sum() + (ref_alphas * v_render_alphas).sum(),
        (means2d_ref, conics_ref, colors_ref, opacities_ref, backgrounds_ref),
    )

    means2d_mps = means2d.reshape(-1, 2).clone().to(mps_device).requires_grad_(True)
    conics_mps = conics.reshape(-1, 3).clone().to(mps_device).requires_grad_(True)
    colors_mps = colors.reshape(-1, colors.shape[-1]).clone().to(mps_device).requires_grad_(True)
    opacities_mps = opacities.reshape(-1).clone().to(mps_device).requires_grad_(True)
    backgrounds_mps = backgrounds.clone().to(mps_device).requires_grad_(True)
    out_colors, out_alphas = gm.rasterize_to_pixels(
        means2d_mps,
        conics_mps,
        colors_mps,
        opacities_mps,
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds_mps,
        packed=True,
    )
    grads = torch.autograd.grad(
        (out_colors * v_render_colors.to(mps_device)).sum()
        + (out_alphas * v_render_alphas.to(mps_device)).sum(),
        (means2d_mps, conics_mps, colors_mps, opacities_mps, backgrounds_mps),
    )
    for actual, expected, atol, rtol in (
        (grads[0], ref_grads[0], 2e-4, 2e-4),
        (grads[1], ref_grads[1], 2e-4, 2e-4),
        (grads[2], ref_grads[2], 2e-4, 2e-4),
        (grads[3], ref_grads[3], 2e-4, 2e-4),
        (grads[4], ref_grads[4], 2e-4, 2e-4),
    ):
        torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


def test_absgrad_matches_reference(mps_device):
    width, height = 12, 8
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means2d, radii, depths, conics, colors, opacities, backgrounds = _sample_inputs(
        image_count=1, n=4, channels=3, width=width, height=height
    )
    _, isect_ids, flatten_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = _isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)

    v_render_colors = torch.randn(1, height, width, 3, dtype=torch.float32)
    v_render_alphas = torch.randn(1, height, width, 1, dtype=torch.float32)
    expected_absgrad = _absgrad_reference(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        v_render_colors,
        v_render_alphas,
        backgrounds=backgrounds,
    )

    means2d_mps = means2d.clone().to(mps_device).requires_grad_(True)
    conics_mps = conics.clone().to(mps_device).requires_grad_(True)
    colors_mps = colors.clone().to(mps_device).requires_grad_(True)
    opacities_mps = opacities.clone().to(mps_device).requires_grad_(True)
    backgrounds_mps = backgrounds.clone().to(mps_device).requires_grad_(True)
    out_colors, out_alphas = gm.rasterize_to_pixels(
        means2d_mps,
        conics_mps,
        colors_mps,
        opacities_mps,
        width,
        height,
        tile_size,
        isect_offsets.to(mps_device),
        flatten_ids.to(mps_device),
        backgrounds=backgrounds_mps,
        absgrad=True,
    )
    loss = (
        (out_colors * v_render_colors.to(mps_device)).sum()
        + (out_alphas * v_render_alphas.to(mps_device)).sum()
    )
    loss.backward()
    assert hasattr(means2d_mps, "absgrad")
    torch.testing.assert_close(
        means2d_mps.absgrad.cpu(),
        expected_absgrad,
        atol=2e-4,
        rtol=2e-4,
    )


def test_end_to_end_pipeline_matches_reference(mps_device):
    width, height = 32, 24
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means, quats, scales, viewmats, Ks, colors, opacities, backgrounds = _sample_world_inputs(
        c=2, n=8, channels=5, width=width, height=height
    )

    covars_full, _ = _quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )
    expected_radii, expected_means2d, expected_depths, expected_conics, _ = _fully_fused_projection(
        means,
        covars_full,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    expected_opacities = torch.broadcast_to(opacities[None, :], expected_depths.shape)
    _, expected_isect_ids, expected_flatten_ids = _isect_tiles(
        expected_means2d,
        expected_radii,
        expected_depths,
        tile_size,
        tile_width,
        tile_height,
    )
    expected_offsets = _isect_offset_encode(
        expected_isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    )
    expected_colors, expected_alphas, _ = _rasterize_reference(
        expected_means2d,
        expected_conics,
        colors,
        expected_opacities,
        width,
        height,
        tile_size,
        expected_offsets,
        expected_flatten_ids,
        backgrounds=backgrounds,
    )

    actual_radii, actual_means2d, actual_depths, actual_conics, _ = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    actual_opacities = torch.broadcast_to(opacities.to(mps_device)[None, :], actual_depths.shape)
    _, actual_isect_ids, actual_flatten_ids = gm.intersect_tiles(
        actual_means2d,
        actual_radii,
        actual_depths,
        tile_size,
        tile_width,
        tile_height,
    )
    actual_offsets = gm.intersect_offset_encode(
        actual_isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    )
    actual_colors, actual_alphas = gm.rasterize_to_pixels(
        actual_means2d,
        actual_conics,
        colors.to(mps_device),
        actual_opacities,
        width,
        height,
        tile_size,
        actual_offsets,
        actual_flatten_ids,
        backgrounds=backgrounds.to(mps_device),
    )
    torch.testing.assert_close(actual_colors.cpu(), expected_colors, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_alphas.cpu(), expected_alphas, atol=2e-4, rtol=2e-4)
