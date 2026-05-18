# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

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


def _accutile_ellipse_intersection(A, B, C, disc, t, p, is_y, coord):
    p_u = p[1] if is_y else p[0]
    p_v = p[0] if is_y else p[1]
    coeff = A if is_y else C
    h = coord - p_u
    sqrt_term = math.sqrt(disc * h * h + t * coeff)
    return ((-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v)


def _accutile_process_tiles(
    A,
    B,
    C,
    disc,
    t,
    p,
    bbox_min,
    bbox_max,
    bbox_argmin,
    bbox_argmax,
    rect_min,
    rect_max,
    tile_size,
    tile_width,
    is_y,
    depth_bits,
    upper_prefix,
    flatten_idx,
    entries,
):
    block = float(tile_size)

    if is_y:
        rect_min = (rect_min[1], rect_min[0])
        rect_max = (rect_max[1], rect_max[0])
        bbox_min = (bbox_min[1], bbox_min[0])
        bbox_max = (bbox_max[1], bbox_max[0])
        bbox_argmin = (bbox_argmin[1], bbox_argmin[0])
        bbox_argmax = (bbox_argmax[1], bbox_argmax[0])

    tiles_count = 0
    intersect_max_line = (bbox_max[1], bbox_min[1])
    min_line = rect_min[0] * block
    if bbox_min[0] <= min_line:
        intersect_min_line = _accutile_ellipse_intersection(A, B, C, disc, t, p, is_y, min_line)
    else:
        intersect_min_line = intersect_max_line

    for u in range(rect_min[0], rect_max[0]):
        max_line = min_line + block
        if max_line <= bbox_max[0]:
            intersect_max_line = _accutile_ellipse_intersection(A, B, C, disc, t, p, is_y, max_line)

        if min_line <= bbox_argmin[1] < max_line:
            ellipse_min = bbox_min[1]
        else:
            ellipse_min = min(intersect_min_line[0], intersect_max_line[0])

        if min_line <= bbox_argmax[1] < max_line:
            ellipse_max = bbox_max[1]
        else:
            ellipse_max = max(intersect_min_line[1], intersect_max_line[1])

        min_tile_v = max(rect_min[1], min(rect_max[1], int(ellipse_min / block)))
        max_tile_v = min(rect_max[1], max(rect_min[1], int(ellipse_max / block + 1.0)))
        tiles_count += max_tile_v - min_tile_v

        if entries is not None:
            for v in range(min_tile_v, max_tile_v):
                tile_id = u * tile_width + v if is_y else v * tile_width + u
                entries.append((((upper_prefix | tile_id) << 32) | depth_bits, flatten_idx))

        intersect_min_line = intersect_max_line
        min_line = max_line

    return tiles_count


def _accutile_reference(
    means2d,
    radii,
    depths,
    conics,
    opacities,
    tile_size,
    tile_width,
    tile_height,
    *,
    packed=False,
    image_ids=None,
    n_images=None,
    sort=True,
):
    alpha_threshold = 1.0 / 255.0
    gaussian_extend = 3.33
    tile_n_bits = (tile_width * tile_height).bit_length()

    if packed:
        nnz = means2d.size(0)
        means_flat = means2d.reshape(nnz, 2)
        radii_flat = radii.reshape(nnz, 2)
        depths_flat = depths.reshape(nnz)
        conics_flat = conics.reshape(nnz, 3)
        opacities_flat = opacities.reshape(nnz)
        image_ids_flat = image_ids.reshape(nnz)
        out_shape = (nnz,)
        N = None
    else:
        image_dims = means2d.shape[:-2]
        N = means2d.shape[-2]
        I = math.prod(image_dims)
        means_flat = means2d.reshape(I * N, 2)
        radii_flat = radii.reshape(I * N, 2)
        depths_flat = depths.reshape(I * N)
        conics_flat = conics.reshape(I * N, 3)
        opacities_flat = opacities.reshape(I * N)
        image_ids_flat = None
        out_shape = image_dims + (N,)

    counts = torch.zeros((means_flat.size(0),), dtype=torch.int32)
    entries = []
    for idx in range(means_flat.size(0)):
        radius_x = int(radii_flat[idx, 0].item())
        radius_y = int(radii_flat[idx, 1].item())
        if radius_x <= 0 or radius_y <= 0:
            continue

        mean = (float(means_flat[idx, 0].item()), float(means_flat[idx, 1].item()))
        A = float(conics_flat[idx, 0].item())
        B = float(conics_flat[idx, 1].item())
        C = float(conics_flat[idx, 2].item())
        disc = B * B - A * C
        opacity = float(opacities_flat[idx].item())
        t = min(gaussian_extend * gaussian_extend, 2.0 * math.log(opacity / alpha_threshold))
        neg_t_over_disc = -t / disc
        x_extent = math.sqrt(neg_t_over_disc * C)
        y_extent = math.sqrt(neg_t_over_disc * A)
        bbox_min = (mean[0] - x_extent, mean[1] - y_extent)
        bbox_max = (mean[0] + x_extent, mean[1] + y_extent)
        bbox_argmin = (mean[1] + B * x_extent / C, mean[0] + B * y_extent / A)
        bbox_argmax = (mean[1] - B * x_extent / C, mean[0] - B * y_extent / A)

        rect_min = (
            max(0, min(tile_width, int(bbox_min[0] / tile_size))),
            max(0, min(tile_height, int(bbox_min[1] / tile_size))),
        )
        rect_max = (
            max(0, min(tile_width, int(bbox_max[0] / tile_size + 1.0))),
            max(0, min(tile_height, int(bbox_max[1] / tile_size + 1.0))),
        )
        y_span = rect_max[1] - rect_min[1]
        x_span = rect_max[0] - rect_min[0]
        if y_span * x_span == 0:
            continue

        depth_bits = depths_flat[idx : idx + 1].view(torch.int32)[0].item() & 0xFFFFFFFF
        image_id = int(image_ids_flat[idx].item()) if packed else idx // N
        upper_prefix = image_id << tile_n_bits
        counts[idx] = _accutile_process_tiles(
            A,
            B,
            C,
            disc,
            t,
            mean,
            bbox_min,
            bbox_max,
            bbox_argmin,
            bbox_argmax,
            rect_min,
            rect_max,
            tile_size,
            tile_width,
            y_span < x_span,
            depth_bits,
            upper_prefix,
            idx,
            entries,
        )

    if sort:
        entries.sort()
    isect_ids = torch.tensor([k for k, _ in entries], dtype=torch.int64)
    flatten_ids = torch.tensor([v for _, v in entries], dtype=torch.int32)
    return counts.reshape(out_shape), isect_ids, flatten_ids


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


def test_segmented_matches_default_sort_unpacked(mps_device):
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

    expected = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
        segmented=False,
    )
    actual = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
        segmented=True,
    )
    _assert_equal(actual, expected)


def test_segmented_matches_default_sort_packed(mps_device):
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
    gaussian_ids = torch.arange(means2d.size(0), dtype=torch.int64)

    expected = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
        segmented=False,
        packed=True,
        n_images=2,
        image_ids=image_ids.to(mps_device),
        gaussian_ids=gaussian_ids.to(mps_device),
    )
    actual = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
        segmented=True,
        packed=True,
        n_images=2,
        image_ids=image_ids.to(mps_device),
        gaussian_ids=gaussian_ids.to(mps_device),
    )
    _assert_equal(actual, expected)


def test_segmented_sort_false_matches_unsorted_emit(mps_device):
    means2d = torch.tensor([[[7.0, 7.0], [3.0, 3.0]]], dtype=torch.float32, device=mps_device)
    radii = torch.tensor([[[2, 2], [1, 1]]], dtype=torch.int32, device=mps_device)
    depths = torch.tensor([[0.2, 0.1]], dtype=torch.float32, device=mps_device)

    expected = gm.intersect_tiles(means2d, radii, depths, 4, 4, 4, sort=False, segmented=False)
    actual = gm.intersect_tiles(means2d, radii, depths, 4, 4, 4, sort=False, segmented=True)
    _assert_equal(actual, expected)


def test_accutile_unpacked_matches_reference(mps_device):
    tile_size = 4
    tile_width = 8
    tile_height = 6
    means2d = torch.tensor(
        [[[10.0, 10.0], [18.0, 9.0], [7.0, 14.0]]],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [[[3, 3], [4, 3], [3, 2]]],
        dtype=torch.int32,
    )
    depths = torch.tensor([[0.1, 0.4, 0.2]], dtype=torch.float32)
    conics = torch.tensor(
        [[[0.18, 0.02, 0.12], [0.11, -0.03, 0.20], [0.25, 0.04, 0.30]]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.9, 0.8, 0.75]], dtype=torch.float32)

    expected = _accutile_reference(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        tile_size,
        tile_width,
        tile_height,
    )
    actual = gm.intersect_tiles(
        means2d.to(mps_device),
        radii.to(mps_device),
        depths.to(mps_device),
        tile_size,
        tile_width,
        tile_height,
        conics=conics.to(mps_device),
        opacities=opacities.to(mps_device),
    )
    _assert_equal(actual, expected)


def test_accutile_packed_matches_reference(mps_device):
    tile_size = 4
    tile_width = 8
    tile_height = 6
    means2d = torch.tensor(
        [[10.0, 10.0], [18.0, 9.0], [7.0, 14.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [[3, 3], [4, 3], [3, 2]],
        dtype=torch.int32,
    )
    depths = torch.tensor([0.1, 0.4, 0.2], dtype=torch.float32)
    conics = torch.tensor(
        [[0.18, 0.02, 0.12], [0.11, -0.03, 0.20], [0.25, 0.04, 0.30]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.9, 0.8, 0.75], dtype=torch.float32)
    image_ids = torch.tensor([0, 1, 1], dtype=torch.int64)

    expected = _accutile_reference(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        image_ids=image_ids,
        n_images=2,
    )
    actual = gm.intersect_tiles(
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
        conics=conics.to(mps_device),
        opacities=opacities.to(mps_device),
    )
    _assert_equal(actual, expected)


# ---------------------------------------------------------------------------
# M1 — AccuTile output is a subset of AABB output
# ---------------------------------------------------------------------------

def test_accutile_self_consistency(mps_device):
    """AccuTile output is internally consistent: isect count matches tiles_per_gauss sum,
    isect_ids are sorted, and flatten_ids are valid Gaussian indices."""
    tile_size, tile_width, tile_height = 4, 12, 10

    means2d = torch.tensor(
        [[[20.0, 16.0], [40.0, 24.0], [8.0, 8.0]]],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [[[4, 4], [6, 4], [3, 3]]],
        dtype=torch.int32,
    )
    depths = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    conics = torch.tensor(
        [[[0.20, 0.05, 0.15], [0.12, -0.04, 0.18], [0.25, 0.03, 0.22]]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.95, 0.85, 0.70]], dtype=torch.float32)

    counts, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
    )

    n_isects = isect_ids.numel()
    # tiles_per_gauss.sum() must equal the total number of intersection records.
    assert int(counts.sum().item()) == n_isects
    # isect_ids must be sorted (sort=True by default).
    if n_isects > 1:
        assert (isect_ids[1:] >= isect_ids[:-1]).all().item()
    # flatten_ids must be valid Gaussian indices.
    n_gaussians = means2d.numel() // 2
    assert flatten_ids.min().item() >= 0
    assert flatten_ids.max().item() < n_gaussians
    # All outputs must be finite.
    assert torch.isfinite(counts.float()).all()
    assert torch.isfinite(isect_ids.float()).all()


# ---------------------------------------------------------------------------
# M2 — AccuTile edge cases
# ---------------------------------------------------------------------------

def test_accutile_opacity_below_threshold_emits_no_tiles(mps_device):
    """A Gaussian with opacity << threshold should produce zero tile records."""
    tile_size, tile_width, tile_height = 4, 8, 8
    means2d = torch.tensor([[[16.0, 16.0]]], dtype=torch.float32)
    radii = torch.tensor([[[3, 3]]], dtype=torch.int32)
    depths = torch.tensor([[0.5]], dtype=torch.float32)
    # A tiny positive conic (positive-definite)
    conics = torch.tensor([[[0.20, 0.0, 0.20]]], dtype=torch.float32)
    # Opacity well below 1/255 (~0.004) means the thresholded ellipse has zero area.
    opacities = torch.tensor([[1e-6]], dtype=torch.float32)

    counts, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
    )
    assert torch.isfinite(counts.float()).all(), "tiles_per_gauss must be finite"
    assert isect_ids.numel() == 0, (
        f"Expected 0 isects for sub-threshold opacity, got {isect_ids.numel()}"
    )


def test_accutile_ellipse_outside_tile_grid_emits_no_tiles(mps_device):
    """An ellipse whose SNUGBOX lies entirely outside the tile grid emits nothing."""
    tile_size, tile_width, tile_height = 4, 8, 6
    # Center is far outside the 32×24 pixel grid.
    means2d = torch.tensor([[[-200.0, -200.0]]], dtype=torch.float32)
    radii = torch.tensor([[[3, 3]]], dtype=torch.int32)
    depths = torch.tensor([[0.1]], dtype=torch.float32)
    conics = torch.tensor([[[0.20, 0.0, 0.20]]], dtype=torch.float32)
    opacities = torch.tensor([[0.9]], dtype=torch.float32)

    counts, isect_ids, _ = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
    )
    assert int(counts.sum()) == 0
    assert isect_ids.numel() == 0


def test_accutile_fallback_when_only_conics_provided(mps_device):
    """Providing conics without opacities falls back silently to AABB."""
    tile_size, tile_width, tile_height = 4, 6, 5
    means2d = torch.tensor([[[10.0, 10.0]]], dtype=torch.float32)
    radii = torch.tensor([[[2, 2]]], dtype=torch.int32)
    depths = torch.tensor([[0.1]], dtype=torch.float32)
    conics = torch.tensor([[[0.20, 0.0, 0.20]]], dtype=torch.float32)

    aabb_result = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
    )
    conics_only_result = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device),  # no opacities
    )
    _assert_equal(aabb_result, conics_only_result)


def test_accutile_sort_false(mps_device):
    """AccuTile with sort=False emits records in Gaussian-index order."""
    tile_size, tile_width, tile_height = 4, 8, 6
    means2d = torch.tensor(
        [[[20.0, 10.0], [10.0, 20.0]]],
        dtype=torch.float32,
    )
    radii = torch.tensor([[[3, 3], [3, 3]]], dtype=torch.int32)
    depths = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
    conics = torch.tensor(
        [[[0.20, 0.0, 0.20], [0.15, 0.0, 0.15]]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.9, 0.8]], dtype=torch.float32)

    counts, isect_sorted, flatten_sorted = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
        sort=True,
    )
    _, isect_unsorted, flatten_unsorted = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
        sort=False,
    )

    # Both have same number of records.
    assert isect_sorted.numel() == isect_unsorted.numel()
    # Sorted output is monotonically non-decreasing (verified on CPU to avoid
    # MPS int64 sort precision loss).
    sorted_cpu = isect_sorted.cpu()
    if sorted_cpu.numel() > 1:
        assert (sorted_cpu[1:] >= sorted_cpu[:-1]).all().item()
    # Unsorted output sorted on CPU must equal the sorted output (also on CPU).
    _assert_equal(
        torch.sort(isect_unsorted.cpu())[0],
        sorted_cpu,
    )


# ---------------------------------------------------------------------------
# M3 — Large-N AccuTile correctness
# ---------------------------------------------------------------------------

def _make_accutile_inputs(n: int, tile_width: int, tile_height: int, tile_size: int):
    """Generate n random Gaussians with valid conics and opacities."""
    torch.manual_seed(42)
    total_pixels = tile_width * tile_size
    means2d = torch.rand(1, n, 2) * float(total_pixels)
    radii = (torch.rand(1, n, 2) * 4 + 1).to(torch.int32)
    depths = torch.rand(1, n) * 10.0

    # Build positive-definite conics: A, C > 0, B²-AC < 0.
    a = torch.rand(1, n) * 0.2 + 0.1
    c = torch.rand(1, n) * 0.2 + 0.1
    b = (torch.rand(1, n) - 0.5) * 0.05   # keep |B| small so AC > B²
    conics = torch.stack([a, b, c], dim=-1)
    opacities = torch.rand(1, n) * 0.5 + 0.5  # [0.5, 1.0]
    return means2d, radii, depths, conics, opacities


@pytest.mark.parametrize("n", [500, 2_000])
def test_accutile_large(mps_device, n):
    tile_size, tile_width, tile_height = 4, 16, 12
    means2d, radii, depths, conics, opacities = _make_accutile_inputs(
        n, tile_width, tile_height, tile_size
    )

    expected = _accutile_reference(
        means2d, radii, depths, conics, opacities, tile_size, tile_width, tile_height
    )
    actual = gm.intersect_tiles(
        means2d.to(mps_device), radii.to(mps_device), depths.to(mps_device),
        tile_size, tile_width, tile_height,
        conics=conics.to(mps_device), opacities=opacities.to(mps_device),
    )
    _assert_equal(actual, expected)
