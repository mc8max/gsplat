# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection_2dgs
from gsplat.metal._math import _isect_offset_encode, _isect_tiles

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_world_inputs(c=2, n=10, width=40, height=28):
    torch.manual_seed(31)
    means = torch.randn(n, 3, dtype=torch.float32) * 0.18
    means[:, 2] = torch.rand(n, dtype=torch.float32) * 1.2 + 1.6
    quats = torch.randn(n, 4, dtype=torch.float32)
    scales = torch.rand(n, 3, dtype=torch.float32) * 0.18 + 0.12

    viewmats = torch.eye(4, dtype=torch.float32).expand(c, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.06, 0.06, c, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.04, -0.04, c, dtype=torch.float32)

    Ks = torch.zeros(c, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 215.0
    Ks[:, 1, 1] = 205.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0

    opacities = torch.rand(c, n, dtype=torch.float32) * 0.55 + 0.3
    return means, quats, scales, viewmats, Ks, opacities


def _prepare_projected_inputs(width=40, height=28, tile_size=8):
    means, quats, scales, viewmats, Ks, opacities = _sample_world_inputs(
        width=width, height=height
    )
    radii, means2d, depths, ray_transforms, _normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
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
    transmittances = torch.ones(means2d.shape[0], height, width, dtype=torch.float32)
    return (
        transmittances,
        means2d,
        ray_transforms,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )


def _reference_rasterize_to_indices_2dgs(
    range_start,
    range_end,
    transmittances,
    means2d,
    ray_transforms,
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
    rt_flat = ray_transforms.reshape(i_count * n, 3, 3)
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
                        u_m = rt_flat[g, 0]
                        v_m = rt_flat[g, 1]
                        w_m = rt_flat[g, 2]

                        h_u = px * w_m - u_m
                        h_v = py * w_m - v_m
                        ray_cross = torch.cross(h_u, h_v, dim=0)
                        if float(ray_cross[2].item()) == 0.0:
                            continue

                        s = ray_cross[:2] / ray_cross[2]
                        gauss_weight_3d = float((s * s).sum().item())
                        delta = means_flat[g] - torch.tensor([px, py], dtype=torch.float32)
                        gauss_weight_2d = 2.0 * float((delta * delta).sum().item())
                        sigma = 0.5 * min(gauss_weight_3d, gauss_weight_2d)
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


def test_rasterize_to_indices_2dgs_full_range_matches_reference():
    inputs = _prepare_projected_inputs()
    expected = _reference_rasterize_to_indices_2dgs(0, 1_000_000, *inputs)

    actual = gm.rasterize_to_indices_in_range_2dgs(
        0,
        1_000_000,
        *[arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in inputs],
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_2dgs_partial_range_matches_reference():
    inputs = _prepare_projected_inputs()
    expected = _reference_rasterize_to_indices_2dgs(1, 3, *inputs)

    actual = gm.rasterize_to_indices_in_range_2dgs(
        1,
        3,
        *[arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in inputs],
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_rasterize_to_indices_2dgs_accepts_native_projection_outputs():
    width = 36
    height = 24
    tile_size = 8
    means, quats, scales, viewmats, Ks, opacities = _sample_world_inputs(
        width=width, height=height
    )

    radii, means2d, depths, ray_transforms, _normals = gm.fully_fused_projection_2dgs(
        means.to("mps"),
        quats.to("mps"),
        scales.to("mps"),
        viewmats.to("mps"),
        Ks.to("mps"),
        width,
        height,
    )
    _tiles_per_gauss, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        math.ceil(width / tile_size),
        math.ceil(height / tile_size),
    )
    isect_offsets = gm.intersect_offset_encode(
        isect_ids.contiguous(),
        means2d.shape[0],
        math.ceil(width / tile_size),
        math.ceil(height / tile_size),
    )
    transmittances = torch.ones(means2d.shape[0], height, width, device="mps", dtype=torch.float32)

    gaussian_ids, pixel_ids, image_ids = gm.rasterize_to_indices_in_range_2dgs(
        0,
        1_000_000,
        transmittances,
        means2d,
        ray_transforms,
        opacities.to("mps"),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    assert gaussian_ids.dtype == torch.int64
    assert pixel_ids.dtype == torch.int64
    assert image_ids.dtype == torch.int64
    assert gaussian_ids.shape == pixel_ids.shape == image_ids.shape
