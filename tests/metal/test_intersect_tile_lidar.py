# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import gsplat.metal as gm
from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt, SpinningDirection
from gsplat.cuda._lidar import LidarTiling, RowOffsetStructuredSpinningLidarModelParameters
from gsplat.cuda._torch_impl_lidar import ANGLE_TO_PIXEL_SCALING_FACTOR, _isect_tiles_lidar
from gsplat.metal._math import _isect_offset_encode


def _integral_mask(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        (mask.shape[0] + 1, mask.shape[1] + 1),
        dtype=torch.int32,
        device=mask.device,
    )
    out[1:, 1:] = mask.to(torch.int32)
    return out.cumsum(0).cumsum(1).to(torch.int32)


def _make_test_lidar(device: torch.device) -> RowOffsetStructuredSpinningLidarModelParametersExt:
    params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.tensor([0.4, -0.4], dtype=torch.float32),
        column_azimuths_rad=torch.tensor([-0.6, -0.2, 0.2, 0.6], dtype=torch.float32),
        row_azimuth_offsets_rad=torch.zeros((2,), dtype=torch.float32),
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.COUNTER_CLOCKWISE,
    )
    tiling = LidarTiling(
        n_bins_azimuth=4,
        n_bins_elevation=2,
        cdf_elevation=torch.tensor([0, 1, 2], dtype=torch.int32),
        cdf_dense_ray_mask=_integral_mask(torch.ones((2, 4), dtype=torch.int32)),
        tiles_pack_info=torch.zeros((8, 2), dtype=torch.int32),
        tiles_to_elements_map=torch.zeros((0, 2), dtype=torch.int32),
    )
    angles_to_columns_map = torch.zeros((2, 4), dtype=torch.int32)
    lidar = RowOffsetStructuredSpinningLidarModelParametersExt(
        params,
        angles_to_columns_map,
        tiling,
    )

    # The Metal wrapper moves these tensors to MPS on dispatch, but the
    # reference path expects them on the same device as the inputs.
    lidar.tiling.cdf_elevation = lidar.tiling.cdf_elevation.to(device)
    lidar.tiling.cdf_dense_ray_mask = lidar.tiling.cdf_dense_ray_mask.to(device)
    lidar.tiling.tiles_pack_info = lidar.tiling.tiles_pack_info.to(device)
    lidar.tiling.tiles_to_elements_map = lidar.tiling.tiles_to_elements_map.to(device)
    return lidar


def _reference_isect_tiles_lidar(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    *,
    sort: bool,
):
    lidar = _make_test_lidar(torch.device("cpu"))
    return _isect_tiles_lidar(
        lidar,
        means2d.cpu(),
        radii.cpu(),
        depths.cpu(),
        sort=sort,
    )


def _assert_equal(actual, expected):
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), e.cpu(), atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


def test_intersect_tiles_lidar_matches_reference_unpacked(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15]]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[[220, 120], [160, 90], [80, 60]]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float32, device=mps_device)

    actual = gm.intersect_tiles_lidar(lidar, means2d, radii, depths, sort=True)
    expected = _reference_isect_tiles_lidar(means2d, radii, depths, sort=True)

    _assert_equal(actual, expected)


def test_intersect_tiles_lidar_matches_reference_packed_unsorted(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15], [0.5, 0.0]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[220, 120], [160, 90], [80, 60], [140, 100]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([0.2, 0.5, 0.1, 0.3], dtype=torch.float32, device=mps_device)
    image_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=mps_device)
    gaussian_ids = torch.tensor([2, 7, 1, 4], dtype=torch.int64, device=mps_device)

    actual = gm.intersect_tiles_lidar(
        lidar,
        means2d,
        radii,
        depths,
        sort=False,
        packed=True,
        n_images=2,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )

    expected_tiles, expected_ids, expected_flat = _reference_isect_tiles_lidar(
        means2d.reshape(2, 2, 2),
        radii.reshape(2, 2, 2),
        depths.reshape(2, 2),
        sort=False,
    )
    expected = (expected_tiles.reshape(-1), expected_ids, expected_flat)

    torch.testing.assert_close(actual[0].cpu(), expected[0].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(actual[1].cpu(), expected[1].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(actual[2].cpu(), expected[2].cpu(), atol=0, rtol=0)


def test_segmented_matches_default_sort_unpacked(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15]]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[[220, 120], [160, 90], [80, 60]]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float32, device=mps_device)

    expected = gm.intersect_tiles_lidar(
        lidar, means2d, radii, depths, sort=True, segmented=False
    )
    actual = gm.intersect_tiles_lidar(
        lidar, means2d, radii, depths, sort=True, segmented=True
    )
    _assert_equal(actual, expected)


def test_segmented_matches_default_sort_packed(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15], [0.5, 0.0]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[220, 120], [160, 90], [80, 60], [140, 100]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([0.2, 0.5, 0.1, 0.3], dtype=torch.float32, device=mps_device)
    image_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=mps_device)
    gaussian_ids = torch.tensor([2, 7, 1, 4], dtype=torch.int64, device=mps_device)

    expected = gm.intersect_tiles_lidar(
        lidar,
        means2d,
        radii,
        depths,
        sort=True,
        segmented=False,
        packed=True,
        n_images=2,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    actual = gm.intersect_tiles_lidar(
        lidar,
        means2d,
        radii,
        depths,
        sort=True,
        segmented=True,
        packed=True,
        n_images=2,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    _assert_equal(actual, expected)


def test_segmented_sort_false_matches_unsorted_emit(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15]]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[[220, 120], [160, 90], [80, 60]]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float32, device=mps_device)

    expected = gm.intersect_tiles_lidar(
        lidar, means2d, radii, depths, sort=False, segmented=False
    )
    actual = gm.intersect_tiles_lidar(
        lidar, means2d, radii, depths, sort=False, segmented=True
    )
    _assert_equal(actual, expected)


def test_intersect_tiles_lidar_offset_chain_matches_reference(mps_device):
    lidar = _make_test_lidar(mps_device)
    means2d = torch.tensor(
        [[[0.0, 0.0], [0.35, 0.1], [-0.45, -0.15]]],
        dtype=torch.float32,
        device=mps_device,
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.tensor(
        [[[220, 120], [160, 90], [80, 60]]],
        dtype=torch.int32,
        device=mps_device,
    )
    depths = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float32, device=mps_device)

    _, actual_ids, _ = gm.intersect_tiles_lidar(
        lidar, means2d, radii, depths, sort=True, segmented=False
    )
    _, expected_ids, _ = _reference_isect_tiles_lidar(means2d, radii, depths, sort=True)

    actual_offsets = gm.intersect_offset_encode(
        actual_ids,
        1,
        lidar.tiling.n_bins_azimuth,
        lidar.tiling.n_bins_elevation,
    )
    expected_offsets = _isect_offset_encode(
        expected_ids.cpu(),
        1,
        lidar.tiling.n_bins_azimuth,
        lidar.tiling.n_bins_elevation,
    )
    _assert_equal(actual_offsets, expected_offsets)
