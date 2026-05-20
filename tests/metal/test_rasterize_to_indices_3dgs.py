# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _manual_scene(image_count=2, n=6, width=4, height=4, tile_size=2):
    tile_height = math.ceil(height / tile_size)
    tile_width = math.ceil(width / tile_size)
    base_points = torch.tensor(
        [
            [0.75, 0.75],
            [2.25, 0.75],
            [0.75, 2.25],
            [2.25, 2.25],
            [1.50, 1.50],
            [1.75, 1.25],
            [0.75, 1.50],
            [2.25, 1.50],
        ],
        dtype=torch.float32,
    )
    repeats = math.ceil(n / base_points.size(0))
    tiled = base_points.repeat(repeats, 1)[:n].clone()
    scale_x = max(width / 4.0, 1.0)
    scale_y = max(height / 4.0, 1.0)
    tiled[:, 0] = tiled[:, 0] * scale_x
    tiled[:, 1] = tiled[:, 1] * scale_y
    tiled[:, 0].clamp_(0.5, width - 0.5)
    tiled[:, 1].clamp_(0.5, height - 0.5)
    means2d = torch.stack(
        [tiled + torch.tensor([0.0, 0.0], dtype=torch.float32) for _ in range(image_count)],
        dim=0,
    )
    conics = torch.tensor([0.18, 0.0, 0.18], dtype=torch.float32).repeat(image_count, n, 1)
    opacities = torch.full((image_count, n), 0.25, dtype=torch.float32)
    transmittances = torch.ones(image_count, height, width, dtype=torch.float32)

    offsets = []
    flatten = []
    running = 0
    for image_id in range(image_count):
        for _tile_y in range(tile_height):
            for _tile_x in range(tile_width):
                offsets.append(running)
                flatten.extend(image_id * n + gauss_id for gauss_id in range(n))
                running += n
    isect_offsets = torch.tensor(offsets, dtype=torch.int32).reshape(
        image_count, tile_height, tile_width
    )
    flatten_ids = torch.tensor(flatten, dtype=torch.int32)

    return (
        transmittances,
        means2d,
        conics,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )


def _sample_world_inputs(c=2, n=12, channels=5, width=32, height=24):
    torch.manual_seed(11)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.3 + 1.6
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.2 + 0.14
    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.03, -0.03, c, dtype=torch.float32)
    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 215.0
    Ks[:, 1, 1] = 205.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    colors = torch.rand(c, n, channels, dtype=torch.float32)
    opacities = torch.rand(n, dtype=torch.float32) * 0.5 + 0.35
    backgrounds = torch.rand(c, channels, dtype=torch.float32)
    return means, quats, scales, viewmats, Ks, colors, opacities, backgrounds


def _accumulate_from_indices(
    means2d,
    conics,
    colors,
    opacities,
    gaussian_ids,
    pixel_ids,
    image_ids,
    image_width,
    image_height,
    initial_transmittances=None,
):
    image_dims = tuple(means2d.shape[:-2])
    i_count = math.prod(image_dims)
    channels = colors.shape[-1]
    means_flat = means2d.reshape(i_count, means2d.shape[-2], 2)
    conics_flat = conics.reshape(i_count, conics.shape[-2], 3)
    colors_flat = colors.reshape(i_count, colors.shape[-2], channels)
    opacities_flat = opacities.reshape(i_count, opacities.shape[-1])

    render_colors = torch.zeros(
        *image_dims, image_height, image_width, channels, dtype=torch.float32
    )
    render_alphas = torch.zeros(
        *image_dims, image_height, image_width, 1, dtype=torch.float32
    )
    if initial_transmittances is None:
        trans = torch.ones(*image_dims, image_height, image_width, dtype=torch.float32)
    else:
        trans = initial_transmittances.clone()

    for g, pix, image_id in zip(gaussian_ids.tolist(), pixel_ids.tolist(), image_ids.tolist()):
        pixel_y = pix // image_width
        pixel_x = pix % image_width
        px = float(pixel_x) + 0.5
        py = float(pixel_y) + 0.5
        mean = means_flat[image_id, g]
        conic = conics_flat[image_id, g]
        delta_x = float(mean[0].item()) - px
        delta_y = float(mean[1].item()) - py
        sigma = (
            0.5
            * (
                float(conic[0].item()) * delta_x * delta_x
                + float(conic[2].item()) * delta_y * delta_y
            )
            + float(conic[1].item()) * delta_x * delta_y
        )
        alpha = min(0.99, float(opacities_flat[image_id, g].item()) * math.exp(-sigma))
        visibility = float(trans[image_id, pixel_y, pixel_x].item()) * alpha
        render_colors[image_id, pixel_y, pixel_x] += (
            colors_flat[image_id, g] * visibility
        )
        trans[image_id, pixel_y, pixel_x] *= 1.0 - alpha
        render_alphas[image_id, pixel_y, pixel_x, 0] = 1.0 - trans[image_id, pixel_y, pixel_x]

    return render_colors, render_alphas, trans


def _reference_rasterize_to_indices(
    range_start,
    range_end,
    transmittances,
    means2d,
    conics,
    opacities,
    image_width,
    image_height,
    tile_size,
    isect_offsets,
    flatten_ids,
):
    image_dims = tuple(means2d.shape[:-2])
    i_count = math.prod(image_dims)
    n = means2d.shape[-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    block_size = tile_size * tile_size
    n_isects = int(flatten_ids.numel())

    means_flat = means2d.reshape(i_count * n, 2)
    conics_flat = conics.reshape(i_count * n, 3)
    opacities_flat = opacities.reshape(i_count * n)
    trans_flat = transmittances.reshape(i_count, image_height, image_width)
    offsets_flat = isect_offsets.reshape(i_count * tile_height * tile_width)

    gaussian_ids = []
    pixel_ids = []
    image_ids = []
    for image_id in range(i_count):
        for pixel_y in range(image_height):
            tile_y = pixel_y // tile_size
            for pixel_x in range(image_width):
                tile_x = pixel_x // tile_size
                tile_id = tile_y * tile_width + tile_x
                global_tile_id = image_id * tile_height * tile_width + tile_id
                isect_start = int(offsets_flat[global_tile_id].item())
                if global_tile_id + 1 < i_count * tile_height * tile_width:
                    isect_end = int(offsets_flat[global_tile_id + 1].item())
                else:
                    isect_end = n_isects
                isect_count = max(0, isect_end - isect_start)
                num_batches = (isect_count + block_size - 1) // block_size
                if range_start >= num_batches:
                    continue

                px = float(pixel_x) + 0.5
                py = float(pixel_y) + 0.5
                trans = float(trans_flat[image_id, pixel_y, pixel_x].item())
                done = False
                for batch in range(range_start, min(range_end, num_batches)):
                    batch_start = isect_start + batch * block_size
                    batch_end = min(isect_end, batch_start + block_size)
                    for idx in range(batch_start, batch_end):
                        g = int(flatten_ids[idx].item())
                        delta_x = float(means_flat[g, 0].item()) - px
                        delta_y = float(means_flat[g, 1].item()) - py
                        conic = conics_flat[g]
                        sigma = (
                            0.5
                            * (
                                float(conic[0].item()) * delta_x * delta_x
                                + float(conic[2].item()) * delta_y * delta_y
                            )
                            + float(conic[1].item()) * delta_x * delta_y
                        )
                        alpha = min(
                            0.99,
                            float(opacities_flat[g].item()) * math.exp(-sigma),
                        )
                        if sigma < 0.0 or alpha < (1.0 / 255.0):
                            continue
                        next_trans = trans * (1.0 - alpha)
                        if next_trans <= 1.0e-4:
                            done = True
                            break
                        gaussian_ids.append(g % n)
                        pixel_ids.append(pixel_y * image_width + pixel_x)
                        image_ids.append(image_id)
                        trans = next_trans
                    if done:
                        break

    return (
        torch.tensor(gaussian_ids, dtype=torch.int64),
        torch.tensor(pixel_ids, dtype=torch.int64),
        torch.tensor(image_ids, dtype=torch.int64),
    )


def test_rasterize_to_indices_full_range_matches_reference():
    inputs = _manual_scene(image_count=2)
    expected = _reference_rasterize_to_indices(0, 1_000_000, *inputs)

    actual = gm.rasterize_to_indices_in_range(
        0,
        1_000_000,
        *[arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in inputs],
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_partial_range_matches_reference():
    inputs = _manual_scene(image_count=1)
    expected = _reference_rasterize_to_indices(1, 2, *inputs)

    actual = gm.rasterize_to_indices_in_range(
        1,
        2,
        *[arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in inputs],
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_empty_intersections_returns_empty_outputs():
    image_width = 4
    image_height = 4
    tile_size = 2
    means2d = torch.zeros(1, 0, 2, dtype=torch.float32, device="mps")
    conics = torch.zeros(1, 0, 3, dtype=torch.float32, device="mps")
    opacities = torch.zeros(1, 0, dtype=torch.float32, device="mps")
    transmittances = torch.ones(1, image_height, image_width, dtype=torch.float32, device="mps")
    isect_offsets = torch.zeros(1, 2, 2, dtype=torch.int32, device="mps")
    flatten_ids = torch.empty(0, dtype=torch.int32, device="mps")

    gaussian_ids, pixel_ids, image_ids = gm.rasterize_to_indices_in_range(
        0,
        8,
        transmittances,
        means2d,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    assert gaussian_ids.dtype == torch.int64
    assert pixel_ids.dtype == torch.int64
    assert image_ids.dtype == torch.int64
    assert gaussian_ids.numel() == 0
    assert pixel_ids.numel() == 0
    assert image_ids.numel() == 0


def test_rasterize_to_indices_large_smoke_matches_reference():
    inputs = _manual_scene(image_count=2, n=40, width=16, height=16, tile_size=4)
    expected = _reference_rasterize_to_indices(1, 3, *inputs)

    actual = gm.rasterize_to_indices_in_range(
        1,
        3,
        *[arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in inputs],
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_end_to_end_pipeline_matches_reference():
    width, height = 32, 24
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means, quats, scales, viewmats, Ks, _, opacities, _ = _sample_world_inputs(
        c=2, n=12, width=width, height=height
    )
    mps = torch.device("mps")

    radii, means2d, depths, conics, _ = gm.fully_fused_projection(
        means.to(mps),
        None,
        quats.to(mps),
        scales.to(mps),
        viewmats.to(mps),
        Ks.to(mps),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    per_view_opacities = torch.broadcast_to(opacities.to(mps)[None, :], depths.shape)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
    )
    offsets = gm.intersect_offset_encode(
        isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    )
    transmittances = torch.ones(viewmats.shape[0], height, width, dtype=torch.float32, device=mps)

    expected = _reference_rasterize_to_indices(
        0,
        1_000_000,
        transmittances.cpu(),
        means2d.cpu(),
        conics.cpu(),
        per_view_opacities.cpu(),
        width,
        height,
        tile_size,
        offsets.cpu(),
        flatten_ids.cpu(),
    )
    actual = gm.rasterize_to_indices_in_range(
        0,
        1_000_000,
        transmittances,
        means2d,
        conics,
        per_view_opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_iterative_batches_reconstruct_full_rasterizer():
    width, height = 32, 24
    tile_size = 4
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    means, quats, scales, viewmats, Ks, colors, opacities, backgrounds = _sample_world_inputs(
        c=2, n=14, channels=5, width=width, height=height
    )
    mps = torch.device("mps")

    radii, means2d, depths, conics, _ = gm.fully_fused_projection(
        means.to(mps),
        None,
        quats.to(mps),
        scales.to(mps),
        viewmats.to(mps),
        Ks.to(mps),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    per_view_opacities = torch.broadcast_to(opacities.to(mps)[None, :], depths.shape)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
    )
    offsets = gm.intersect_offset_encode(
        isect_ids,
        viewmats.shape[0],
        tile_width,
        tile_height,
    )

    expected_colors, expected_alphas = gm.rasterize_to_pixels(
        means2d,
        conics,
        colors.to(mps),
        per_view_opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds=backgrounds.to(mps),
    )

    block_size = tile_size * tile_size
    offsets_flat = torch.cat(
        [offsets.reshape(-1).cpu(), torch.tensor([flatten_ids.numel()], dtype=torch.int32)]
    )
    max_range = int((offsets_flat[1:] - offsets_flat[:-1]).max().item()) if offsets_flat.numel() > 1 else 0
    num_batches = (max_range + block_size - 1) // block_size

    gathered_gaussians = []
    gathered_pixels = []
    gathered_images = []
    transmittances = torch.ones(viewmats.shape[0], height, width, dtype=torch.float32, device=mps)
    for step in range(num_batches):
        gauss_ids, pixel_ids, image_ids = gm.rasterize_to_indices_in_range(
            step,
            step + 1,
            transmittances,
            means2d,
            conics,
            per_view_opacities,
            width,
            height,
            tile_size,
            offsets,
            flatten_ids,
        )
        if gauss_ids.numel() == 0:
            continue
        gathered_gaussians.append(gauss_ids.cpu())
        gathered_pixels.append(pixel_ids.cpu())
        gathered_images.append(image_ids.cpu())
        _, _, trans_cpu = _accumulate_from_indices(
            means2d.cpu(),
            conics.cpu(),
            colors,
            per_view_opacities.cpu(),
            gauss_ids.cpu(),
            pixel_ids.cpu(),
            image_ids.cpu(),
            width,
            height,
            initial_transmittances=transmittances.cpu(),
        )
        transmittances = trans_cpu.to(mps)

    all_gaussians = torch.cat(gathered_gaussians) if gathered_gaussians else torch.empty(0, dtype=torch.int64)
    all_pixels = torch.cat(gathered_pixels) if gathered_pixels else torch.empty(0, dtype=torch.int64)
    all_images = torch.cat(gathered_images) if gathered_images else torch.empty(0, dtype=torch.int64)
    rebuilt_colors, rebuilt_alphas, rebuilt_trans = _accumulate_from_indices(
        means2d.cpu(),
        conics.cpu(),
        colors,
        per_view_opacities.cpu(),
        all_gaussians,
        all_pixels,
        all_images,
        width,
        height,
    )
    rebuilt_colors = rebuilt_colors + backgrounds[:, None, None, :] * rebuilt_trans[..., None]

    torch.testing.assert_close(rebuilt_colors, expected_colors.cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(rebuilt_alphas, expected_alphas.cpu(), atol=2e-4, rtol=2e-4)
