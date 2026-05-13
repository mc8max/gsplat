# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Metal intersect_offset helper op."""

import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _isect_offset_encode

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _mps_sync():
    torch.mps.synchronize()


def _pack_isect_ids(entries, tile_width, tile_height):
    """Pack [(image_id, tile_id, depth_key), ...] into sorted int64 isect ids."""
    tile_n_bits = (tile_width * tile_height).bit_length()
    packed = []
    for image_id, tile_id, depth_key in entries:
        upper = (image_id << tile_n_bits) | tile_id
        packed.append((upper << 32) | (depth_key & 0xFFFFFFFF))
    packed.sort()
    return torch.tensor(packed, dtype=torch.int64)


def _assert_offsets_close(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


def _make_large_sorted_isect_ids(n_isects, I, tile_width, tile_height):
    n_tiles = tile_width * tile_height
    entries = []
    for idx in range(n_isects):
        flat_tile = idx % (I * n_tiles)
        image_id = flat_tile // n_tiles
        tile_id = flat_tile % n_tiles
        depth_key = idx & 0xFFFFFFFF
        entries.append((image_id, tile_id, depth_key))
    return _pack_isect_ids(entries, tile_width, tile_height)


def test_empty(mps_device):
    isect_ids = torch.empty((0,), dtype=torch.int64, device=mps_device)
    actual = gm.intersect_offset_encode(isect_ids, 2, 4, 3)
    expected = torch.zeros((2, 3, 4), dtype=torch.int32)
    assert actual.dtype == torch.int32
    assert actual.shape == (2, 3, 4)
    _assert_offsets_close(actual, expected)


def test_single_intersection(mps_device):
    I, tile_width, tile_height = 1, 4, 3
    isect_ids_cpu = _pack_isect_ids([(0, 6, 17)], tile_width, tile_height)
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    assert actual.dtype == torch.int32
    assert actual.shape == (I, tile_height, tile_width)
    _assert_offsets_close(actual, expected)


def test_multiple_intersections_same_tile(mps_device):
    I, tile_width, tile_height = 1, 4, 3
    isect_ids_cpu = _pack_isect_ids(
        [(0, 5, 10), (0, 5, 11), (0, 5, 12), (0, 5, 13)],
        tile_width,
        tile_height,
    )
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    _assert_offsets_close(actual, expected)


def test_gaps_between_tiles(mps_device):
    I, tile_width, tile_height = 1, 5, 2
    isect_ids_cpu = _pack_isect_ids(
        [(0, 1, 7), (0, 1, 8), (0, 7, 20), (0, 9, 30)],
        tile_width,
        tile_height,
    )
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    _assert_offsets_close(actual, expected)


def test_multiple_images(mps_device):
    I, tile_width, tile_height = 3, 4, 2
    isect_ids_cpu = _pack_isect_ids(
        [(0, 0, 1), (0, 3, 9), (1, 0, 2), (1, 7, 10), (2, 4, 3)],
        tile_width,
        tile_height,
    )
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    _assert_offsets_close(actual, expected)


def test_full_dense_tile_occupancy(mps_device):
    I, tile_width, tile_height = 2, 3, 2
    entries = []
    depth = 0
    for image_id in range(I):
        for tile_id in range(tile_width * tile_height):
            entries.append((image_id, tile_id, depth))
            depth += 1
    isect_ids_cpu = _pack_isect_ids(entries, tile_width, tile_height)
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    _assert_offsets_close(actual, expected)


@pytest.mark.parametrize("n", [1_000, 10_000])
def test_large_exact(mps_device, n):
    I, tile_width, tile_height = 4, 16, 12
    isect_ids_cpu = _make_large_sorted_isect_ids(n, I, tile_width, tile_height)
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height)
    _assert_offsets_close(actual, expected)


def test_100k_spotcheck(mps_device):
    I, tile_width, tile_height = 8, 32, 16
    isect_ids_cpu = _make_large_sorted_isect_ids(100_000, I, tile_width, tile_height)
    actual = gm.intersect_offset_encode(isect_ids_cpu.to(mps_device), I, tile_width, tile_height).cpu()
    expected = _isect_offset_encode(isect_ids_cpu, I, tile_width, tile_height)

    _assert_offsets_close(actual[:2, :4, :8], expected[:2, :4, :8])
    _assert_offsets_close(actual[3:5, 5:9, 8:16], expected[3:5, 5:9, 8:16])
    _assert_offsets_close(actual[-2:, -4:, -8:], expected[-2:, -4:, -8:])


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf(mps_device, n, capsys):
    I, tile_width, tile_height = 8, 32, 16
    isect_ids = _make_large_sorted_isect_ids(n, I, tile_width, tile_height).to(mps_device)

    for _ in range(5):
        gm.intersect_offset_encode(isect_ids, I, tile_width, tile_height)
    _mps_sync()

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.intersect_offset_encode(isect_ids, I, tile_width, tile_height)
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] intersect_offset  n={n:>7,}  {elapsed_ms:.3f} ms/iter")
