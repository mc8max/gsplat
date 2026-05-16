# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _isect_offset_encode, _isect_tiles

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)

def _assert_equal(actual, expected):
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), e.cpu(), atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


def test_unpacked_matches_reference(mps_device):
    tile_size = 4
    tile_width = 6
    tile_height = 5

    means2d = torch.tensor(
        [
            [[3.0, 3.0], [8.0, 6.0], [15.0, 9.0]],
            [[2.0, 14.0], [18.0, 10.0], [7.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[1, 1], [3, 2], [0, 0]],
            [[2, 1], [4, 3], [1, 2]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [[0.1, 0.4, 0.7], [0.3, 0.2, 0.5]],
        dtype=torch.float32,
    )

    expected = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    actual = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
    )
    _assert_equal(actual, expected)


def test_packed_matches_manual_reference(mps_device):
    tile_size = 4
    tile_width = 5
    tile_height = 4

    means2d = torch.tensor(
        [[3.0, 3.0], [7.0, 6.0], [11.0, 5.0], [15.0, 8.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [[1, 1], [2, 1], [0, 0], [2, 2]],
        dtype=torch.int32,
    )
    depths = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    image_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    tile_means2d = means2d / tile_size
    tile_radii = radii.to(torch.float32) / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[:, 0] = torch.clamp(tile_mins[:, 0], 0, tile_width)
    tile_mins[:, 1] = torch.clamp(tile_mins[:, 1], 0, tile_height)
    tile_maxs[:, 0] = torch.clamp(tile_maxs[:, 0], 0, tile_width)
    tile_maxs[:, 1] = torch.clamp(tile_maxs[:, 1], 0, tile_height)
    expected_tiles = ((tile_maxs - tile_mins).prod(dim=-1) * (radii > 0).all(dim=-1)).to(torch.int32)

    entries = []
    tile_n_bits = (tile_width * tile_height).bit_length()
    for idx in range(means2d.size(0)):
        if not bool((radii[idx] > 0).all()):
            continue
        depth_bits = depths[idx].view(torch.int32).item() & 0xFFFFFFFF
        for y in range(tile_mins[idx, 1].item(), tile_maxs[idx, 1].item()):
            for x in range(tile_mins[idx, 0].item(), tile_maxs[idx, 0].item()):
                tile_id = y * tile_width + x
                upper = (int(image_ids[idx].item()) << tile_n_bits) | tile_id
                entries.append(((upper << 32) | depth_bits, idx))
    entries.sort()
    expected_isect_ids = torch.tensor([k for k, _ in entries], dtype=torch.int64)
    expected_flatten_ids = torch.tensor([v for _, v in entries], dtype=torch.int32)

    actual_tiles, actual_isect_ids, actual_flatten_ids = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=2,
        image_ids=image_ids.to(mps_device),
        gaussian_ids=torch.arange(means2d.size(0), dtype=torch.int64, device=mps_device),
    )

    _assert_equal(actual_tiles, expected_tiles)
    _assert_equal(actual_isect_ids, expected_isect_ids)
    _assert_equal(actual_flatten_ids, expected_flatten_ids)


def test_explicit_count_emit_match_wrapper(mps_device):
    means2d = torch.tensor([[[3.0, 3.0], [8.0, 6.0]]], dtype=torch.float32, device=mps_device)
    radii = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.int32, device=mps_device)
    depths = torch.tensor([[0.1, 0.2]], dtype=torch.float32, device=mps_device)

    counts = gm.intersect_tile_count(means2d, radii, depths, 4, 4, 4)
    cum = torch.cumsum(counts.reshape(-1).to(torch.int64), dim=0)
    isect_ids, flatten_ids = gm.intersect_tile_emit(
        means2d, radii, depths, cum, 4, 4, 4
    )
    expected = gm.intersect_tiles(means2d, radii, depths, 4, 4, 4)
    _assert_equal((counts, isect_ids, flatten_ids), expected)


def test_sort_false_preserves_emit_order(mps_device):
    means2d = torch.tensor([[[7.0, 7.0], [3.0, 3.0]]], dtype=torch.float32, device=mps_device)
    radii = torch.tensor([[[2, 2], [1, 1]]], dtype=torch.int32, device=mps_device)
    depths = torch.tensor([[0.2, 0.1]], dtype=torch.float32, device=mps_device)

    counts = gm.intersect_tile_count(means2d, radii, depths, 4, 4, 4)
    cum = torch.cumsum(counts.reshape(-1).to(torch.int64), dim=0)
    expected_isect_ids, expected_flatten_ids = gm.intersect_tile_emit(
        means2d, radii, depths, cum, 4, 4, 4
    )

    actual_counts, actual_isect_ids, actual_flatten_ids = gm.intersect_tiles(
        means2d, radii, depths, 4, 4, 4, sort=False
    )
    _assert_equal(actual_counts, counts)
    _assert_equal(actual_isect_ids, expected_isect_ids)
    _assert_equal(actual_flatten_ids, expected_flatten_ids)


def test_duplicate_keys_preserve_multiplicity(mps_device):
    tile_size = 4
    tile_width = 4
    tile_height = 4
    means2d = torch.tensor([[[6.0, 6.0], [6.0, 6.0]]], dtype=torch.float32)
    radii = torch.tensor([[[2, 2], [2, 2]]], dtype=torch.int32)
    depths = torch.tensor([[0.25, 0.25]], dtype=torch.float32)

    expected = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    actual = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
    )
    _assert_equal(actual, expected)

    _, isect_ids, flatten_ids = actual
    assert isect_ids.numel() > 1
    assert torch.unique(isect_ids.cpu()).numel() < isect_ids.numel()
    assert torch.unique(flatten_ids.cpu()).numel() == 2


def test_empty_inputs_return_empty_outputs(mps_device):
    means2d = torch.empty((2, 0, 2), dtype=torch.float32, device=mps_device)
    radii = torch.empty((2, 0, 2), dtype=torch.int32, device=mps_device)
    depths = torch.empty((2, 0), dtype=torch.float32, device=mps_device)

    counts, isect_ids, flatten_ids = gm.intersect_tiles(means2d, radii, depths, 4, 4, 4)
    assert counts.shape == (2, 0)
    assert counts.dtype == torch.int32
    assert isect_ids.shape == (0,)
    assert isect_ids.dtype == torch.int64
    assert flatten_ids.shape == (0,)
    assert flatten_ids.dtype == torch.int32


def test_all_culled_returns_zero_isects(mps_device):
    means2d = torch.tensor([[[2.0, 2.0], [6.0, 6.0], [10.0, 10.0]]], dtype=torch.float32)
    radii = torch.zeros((1, 3, 2), dtype=torch.int32)
    depths = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    counts, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        4,
        4,
        4,
    )
    _assert_equal(counts, torch.zeros((1, 3), dtype=torch.int32))
    assert isect_ids.shape == (0,)
    assert isect_ids.dtype == torch.int64
    assert flatten_ids.shape == (0,)
    assert flatten_ids.dtype == torch.int32


def test_unpacked_chain_matches_reference(mps_device):
    tile_size = 4
    tile_width = 6
    tile_height = 5
    means2d = torch.tensor(
        [
            [[3.0, 3.0], [8.0, 6.0], [15.0, 9.0]],
            [[2.0, 14.0], [18.0, 10.0], [7.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[1, 1], [3, 2], [0, 0]],
            [[2, 1], [4, 3], [1, 2]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [[0.1, 0.4, 0.7], [0.3, 0.2, 0.5]],
        dtype=torch.float32,
    )

    expected_tiles, expected_isect_ids, expected_flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    expected_offsets = _isect_offset_encode(
        expected_isect_ids, means2d.shape[0], tile_width, tile_height
    )

    actual_tiles, actual_isect_ids, actual_flatten_ids = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
    )
    actual_offsets = gm.intersect_offset_encode(
        actual_isect_ids, means2d.shape[0], tile_width, tile_height
    )

    _assert_equal(actual_tiles, expected_tiles)
    _assert_equal(actual_isect_ids, expected_isect_ids)
    _assert_equal(actual_flatten_ids, expected_flatten_ids)
    _assert_equal(actual_offsets, expected_offsets)


def test_packed_chain_matches_manual_reference(mps_device):
    tile_size = 4
    tile_width = 5
    tile_height = 4
    n_images = 2

    means2d = torch.tensor(
        [[3.0, 3.0], [7.0, 6.0], [11.0, 5.0], [15.0, 8.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [[1, 1], [2, 1], [0, 0], [2, 2]],
        dtype=torch.int32,
    )
    depths = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    image_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    tile_means2d = means2d / tile_size
    tile_radii = radii.to(torch.float32) / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[:, 0] = torch.clamp(tile_mins[:, 0], 0, tile_width)
    tile_mins[:, 1] = torch.clamp(tile_mins[:, 1], 0, tile_height)
    tile_maxs[:, 0] = torch.clamp(tile_maxs[:, 0], 0, tile_width)
    tile_maxs[:, 1] = torch.clamp(tile_maxs[:, 1], 0, tile_height)

    entries = []
    tile_n_bits = (tile_width * tile_height).bit_length()
    for idx in range(means2d.size(0)):
        if not bool((radii[idx] > 0).all()):
            continue
        depth_bits = depths[idx].view(torch.int32).item() & 0xFFFFFFFF
        for y in range(tile_mins[idx, 1].item(), tile_maxs[idx, 1].item()):
            for x in range(tile_mins[idx, 0].item(), tile_maxs[idx, 0].item()):
                tile_id = y * tile_width + x
                upper = (int(image_ids[idx].item()) << tile_n_bits) | tile_id
                entries.append(((upper << 32) | depth_bits, idx))
    entries.sort()
    expected_isect_ids = torch.tensor([k for k, _ in entries], dtype=torch.int64)
    expected_offsets = _isect_offset_encode(
        expected_isect_ids, n_images, tile_width, tile_height
    )

    _, actual_isect_ids, _ = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=n_images,
        image_ids=image_ids.to(mps_device),
        gaussian_ids=torch.arange(means2d.size(0), dtype=torch.int64, device=mps_device),
    )
    actual_offsets = gm.intersect_offset_encode(
        actual_isect_ids, n_images, tile_width, tile_height
    )

    _assert_equal(actual_isect_ids, expected_isect_ids)
    _assert_equal(actual_offsets, expected_offsets)


def test_segmented_rejected(mps_device):
    means2d = torch.tensor([[[3.0, 3.0]]], dtype=torch.float32, device=mps_device)
    radii = torch.tensor([[[1, 1]]], dtype=torch.int32, device=mps_device)
    depths = torch.tensor([[0.1]], dtype=torch.float32, device=mps_device)

    with pytest.raises(NotImplementedError, match="segmented=True"):
        gm.intersect_tiles(means2d, radii, depths, 4, 4, 4, segmented=True)


def test_accutile_inputs_rejected(mps_device):
    means2d = torch.tensor([[[3.0, 3.0]]], dtype=torch.float32, device=mps_device)
    radii = torch.tensor([[[1, 1]]], dtype=torch.int32, device=mps_device)
    depths = torch.tensor([[0.1]], dtype=torch.float32, device=mps_device)
    conics = torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float32, device=mps_device)
    opacities = torch.tensor([[0.5]], dtype=torch.float32, device=mps_device)

    with pytest.raises(NotImplementedError, match="AABB path"):
        gm.intersect_tiles(
            means2d,
            radii,
            depths,
            4,
            4,
            4,
            conics=conics,
            opacities=opacities,
        )
