# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat
import gsplat.metal as gm
from gsplat._camera_types import (
    BivariateWindshieldModelParameters,
    ExternalDistortionReferencePolynomial,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    UnscentedTransformParameters,
)
from gsplat.metal._math import _fully_fused_projection_with_ut

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_inputs(batch_dims=(), cameras=2, gaussians=10, width=64, height=48):
    means = torch.randn(*batch_dims, gaussians, 3, dtype=torch.float32) * 0.15
    means[..., 2] = torch.rand(*batch_dims, gaussians, dtype=torch.float32) * 1.5 + 1.25
    quats = torch.randn(*batch_dims, gaussians, 4, dtype=torch.float32)
    scales = torch.rand(*batch_dims, gaussians, 3, dtype=torch.float32) * 0.12 + 0.05
    viewmats = torch.eye(4, dtype=torch.float32).expand(*batch_dims, cameras, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.08, 0.08, cameras, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.05, -0.05, cameras, dtype=torch.float32)
    Ks = torch.zeros(*batch_dims, cameras, 3, 3, dtype=torch.float32)
    Ks[..., 0, 0] = 180.0
    Ks[..., 1, 1] = 170.0
    Ks[..., 0, 2] = width * 0.5
    Ks[..., 1, 2] = height * 0.5
    Ks[..., 2, 2] = 1.0
    opacities = torch.rand(*batch_dims, gaussians, dtype=torch.float32) * 0.7 + 0.2
    return means, quats, scales, opacities, viewmats, Ks


def _make_lidar(device: torch.device, seed: int = 42):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n_rows = 128
    n_columns = 1200
    elevation_start = 0.2256710722828668
    elevation_end = -0.2176425577236929
    azimuth_base_start = 1.0471975511965976
    azimuth_base_end = -1.0471975511965976
    elevation_span = abs(elevation_end - elevation_start)
    azimuth_base_span = abs(azimuth_base_end - azimuth_base_start)

    row_elevations_rad = (
        torch.linspace(elevation_start, elevation_end, n_rows, dtype=torch.float32, device=device)
        + (torch.rand(n_rows, dtype=torch.float32, device=device, generator=generator) - 0.5)
        * (elevation_span / (n_rows - 1))
        * 0.01
    )
    column_azimuths_rad = (
        torch.linspace(azimuth_base_start, azimuth_base_end, n_columns, dtype=torch.float32, device=device)
        + (torch.rand(n_columns, dtype=torch.float32, device=device, generator=generator) - 0.5)
        * (azimuth_base_span / (n_columns - 1))
        * 0.01
    )
    row_azimuth_offsets_rad = (
        torch.rand(n_rows, dtype=torch.float32, device=device, generator=generator) - 0.5
    ) * 0.2

    lidar_params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        spinning_frequency_hz=10,
        spinning_direction=gsplat.SpinningDirection.CLOCKWISE,
    )
    angles_to_columns_map = torch.zeros(
        (4 * n_rows, 4 * n_columns), dtype=torch.int32, device=device
    )
    tiling = gsplat.LidarTiling(
        n_bins_azimuth=1,
        n_bins_elevation=1,
        cdf_elevation=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cdf_dense_ray_mask=torch.ones((2, 2), dtype=torch.int32, device=device),
        tiles_pack_info=torch.zeros((1, 2), dtype=torch.int32, device=device),
        tiles_to_elements_map=torch.zeros((0, 2), dtype=torch.int32, device=device),
    )
    return gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )


def _assert_projection_close(actual, expected):
    actual_radii, actual_means2d, actual_depths, actual_conics, actual_compensations = actual
    expected_radii, expected_means2d, expected_depths, expected_conics, expected_compensations = expected

    valid = ((actual_radii.cpu() > 0) & (expected_radii > 0)).all(dim=-1)
    torch.testing.assert_close(actual_radii.cpu(), expected_radii, atol=1, rtol=0)
    torch.testing.assert_close(actual_means2d.cpu()[valid], expected_means2d[valid], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_depths.cpu()[valid], expected_depths[valid], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_conics.cpu()[valid], expected_conics[valid], atol=2e-3, rtol=2e-3)
    if expected_compensations is None:
        assert actual_compensations is None
    else:
        torch.testing.assert_close(
            actual_compensations.cpu()[valid],
            expected_compensations[valid],
            atol=5e-4,
            rtol=5e-4,
        )


def test_projection_ut_3dgs_fused_matches_reference_phase1(mps_device):
    width, height = 64, 48
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=3, gaussians=12, width=width, height=height
    )
    ut_params = UnscentedTransformParameters(
        alpha=0.2,
        beta=2.0,
        kappa=0.0,
        in_image_margin_factor=0.15,
        require_all_sigma_points_valid=True,
    )

    expected = _fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        rolling_shutter=RollingShutterType.GLOBAL,
        global_z_order=False,
        eps2d=0.3,
        near_plane=0.05,
        far_plane=10.0,
        radius_clip=0.0,
    )
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        eps2d=0.3,
        near_plane=0.05,
        far_plane=10.0,
        radius_clip=0.0,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        rolling_shutter=RollingShutterType.GLOBAL,
        global_z_order=False,
    )
    _assert_projection_close(actual, expected)


def test_projection_ut_3dgs_fused_matches_reference_pinhole_distortion(mps_device):
    width, height = 72, 52
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=10, width=width, height=height
    )
    radial = torch.tensor(
        [[[0.03, -0.01, 0.002, 0.0, 0.0, 0.0], [0.01, 0.005, -0.001, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )[0]
    tangential = torch.tensor(
        [[[0.002, -0.001], [-0.0015, 0.0007]]],
        dtype=torch.float32,
    )[0]
    thin_prism = torch.tensor(
        [[[0.0003, -0.0001, 0.0002, 0.0001], [0.0001, 0.0002, -0.0002, 0.00015]]],
        dtype=torch.float32,
    )[0]
    ut_params = UnscentedTransformParameters()

    expected = _fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
    )
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        radial_coeffs=radial.to(mps_device),
        tangential_coeffs=tangential.to(mps_device),
        thin_prism_coeffs=thin_prism.to(mps_device),
    )
    _assert_projection_close(actual, expected)


def test_projection_ut_3dgs_fused_matches_reference_fisheye(mps_device):
    width, height = 72, 52
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=2, gaussians=10, width=width, height=height
    )
    radial = torch.tensor(
        [
            [[0.02, -0.005, 0.0005, -0.00005], [0.01, 0.002, -0.0003, 0.00002]],
            [[0.015, -0.004, 0.0004, -0.00004], [0.009, 0.0015, -0.0002, 0.00001]],
        ],
        dtype=torch.float32,
    )
    ut_params = UnscentedTransformParameters()

    expected = _fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="fisheye",
        ut_params=ut_params,
        radial_coeffs=radial,
    )
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="fisheye",
        ut_params=ut_params,
        radial_coeffs=radial.to(mps_device),
    )
    _assert_projection_close(actual, expected)


def test_projection_ut_3dgs_fused_matches_reference_rolling_shutter(mps_device):
    width, height = 68, 50
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(1,), cameras=2, gaussians=8, width=width, height=height
    )
    viewmats_rs = viewmats.clone()
    viewmats_rs[..., 0, 3] += torch.tensor([0.03, -0.02], dtype=torch.float32)
    viewmats_rs[..., 1, 3] += torch.tensor([-0.015, 0.025], dtype=torch.float32)
    ut_params = UnscentedTransformParameters(
        alpha=0.15,
        beta=2.0,
        kappa=0.0,
        in_image_margin_factor=0.1,
        require_all_sigma_points_valid=False,
    )

    expected = _fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs,
    )
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs.to(mps_device),
    )
    _assert_projection_close(actual, expected)


def test_projection_ut_3dgs_fused_matches_reference_lidar(mps_device):
    batch_dims = ()
    lidar_cpu = _make_lidar(torch.device("cpu"), seed=42)
    lidar_mps = _make_lidar(mps_device, seed=42)

    width = lidar_cpu.n_columns
    height = lidar_cpu.n_rows
    means, quats, scales, opacities, viewmats, _ = _sample_inputs(
        batch_dims=batch_dims, cameras=1, gaussians=8, width=width, height=height
    )
    means[..., 2] = torch.rand(8, dtype=torch.float32) * 2.0 + 2.0
    Ks = torch.tensor(
        [[[float(width), 0.0, width / 2.0], [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    ut_params = UnscentedTransformParameters()

    expected = _fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="lidar",
        ut_params=ut_params,
        lidar_coeffs=lidar_cpu,
    )
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="lidar",
        ut_params=ut_params,
        lidar_coeffs=lidar_mps,
    )
    _assert_projection_close(actual, expected)


def test_projection_ut_3dgs_fused_matches_reference_ftheta(mps_device):
    width, height = 72, 52
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(2,), cameras=2, gaussians=8, width=width, height=height
    )
    ftheta_coeffs = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        angle_to_pixeldist_poly=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        max_angle=1.25,
        linear_cde=(1.0, 0.015, -0.02),
    )
    ut_params = UnscentedTransformParameters()
    actual = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="ftheta",
        ut_params=ut_params,
        ftheta_coeffs=ftheta_coeffs,
    )
    baseline = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        ut_params=ut_params,
    )

    radii, means2d, depths, conics, compensations = actual
    assert radii.shape == (2, 2, 8, 2)
    assert means2d.shape == (2, 2, 8, 2)
    assert depths.shape == (2, 2, 8)
    assert conics.shape == (2, 2, 8, 3)
    assert compensations is not None and compensations.shape == (2, 2, 8)
    assert torch.isfinite(means2d).all()
    assert torch.isfinite(depths).all()
    assert torch.isfinite(conics).all()
    assert torch.any((radii > 0).all(dim=-1))
    valid = ((radii.cpu() > 0) & (baseline[0].cpu() > 0)).all(dim=-1)
    assert torch.any((means2d.cpu()[valid] - baseline[1].cpu()[valid]).abs() > 1e-4)


def test_projection_ut_3dgs_fused_external_distortion_identity_and_effect(mps_device):
    width, height = 64, 48
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=8, width=width, height=height
    )
    identity_distortion = BivariateWindshieldModelParameters(
        reference_poly=ExternalDistortionReferencePolynomial.FORWARD,
        horizontal_poly=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        vertical_poly=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        horizontal_poly_inverse=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        vertical_poly_inverse=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    )
    skewed_distortion = BivariateWindshieldModelParameters(
        reference_poly=ExternalDistortionReferencePolynomial.FORWARD,
        horizontal_poly=torch.tensor([0.0, 1.05, 0.03], dtype=torch.float32),
        vertical_poly=torch.tensor([0.0, -0.02, 0.95], dtype=torch.float32),
        horizontal_poly_inverse=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        vertical_poly_inverse=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    )

    baseline = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
    )
    identity = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        external_distortion_coeffs=identity_distortion,
    )
    _assert_projection_close(identity, tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in baseline))

    distorted = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        external_distortion_coeffs=skewed_distortion,
    )
    assert torch.isfinite(distorted[1]).all()
    valid = ((distorted[0].cpu() > 0) & (baseline[0].cpu() > 0)).all(dim=-1)
    assert torch.any((distorted[1].cpu()[valid] - baseline[1].cpu()[valid]).abs() > 1e-4)


def test_projection_ut_3dgs_fused_culls_invalid_gaussians(mps_device):
    width, height = 32, 24
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=2, gaussians=4, width=width, height=height
    )
    means[0] = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float32)
    quats[1] = 0.0
    scales[2, 0] = 0.0
    means[3] = torch.tensor([50.0, 0.0, 1.0], dtype=torch.float32)

    radii, means2d, depths, conics, compensations = gm.fully_fused_projection_with_ut(
        means.to(mps_device),
        quats.to(mps_device),
        scales.to(mps_device),
        opacities.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )

    assert radii.shape == (2, 4, 2)
    assert means2d.shape == (2, 4, 2)
    assert depths.shape == (2, 4)
    assert conics.shape == (2, 4, 3)
    assert compensations is None
    assert int((radii[0, 1:] > 0).sum().item()) == 0
    assert int((radii[1, 1:] > 0).sum().item()) == 0


def test_projection_ut_3dgs_fused_rejects_unsupported_features(mps_device):
    width, height = 32, 24
    means, quats, scales, opacities, viewmats, Ks = _sample_inputs(
        batch_dims=(), cameras=1, gaussians=3, width=width, height=height
    )

    with pytest.raises(ValueError, match="viewmats_rs is required"):
        gm.fully_fused_projection_with_ut(
            means.to(mps_device),
            quats.to(mps_device),
            scales.to(mps_device),
            opacities.to(mps_device),
            viewmats.to(mps_device),
            Ks.to(mps_device),
            width,
            height,
            camera_model="pinhole",
            rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        )

    with pytest.raises(ValueError, match="ftheta requires ftheta_coeffs"):
        gm.fully_fused_projection_with_ut(
            means.to(mps_device),
            quats.to(mps_device),
            scales.to(mps_device),
            opacities.to(mps_device),
            viewmats.to(mps_device),
            Ks.to(mps_device),
            width,
            height,
            camera_model="ftheta",
        )

    lidar = _make_lidar(mps_device, seed=42)
    with pytest.raises(NotImplementedError, match="lidar UT projection only supports global shutter"):
        gm.fully_fused_projection_with_ut(
            means.to(mps_device),
            quats.to(mps_device),
            scales.to(mps_device),
            opacities.to(mps_device),
            viewmats.to(mps_device),
            Ks.to(mps_device),
            lidar.n_columns,
            lidar.n_rows,
            camera_model="lidar",
            lidar_coeffs=lidar,
            rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
            viewmats_rs=viewmats.to(mps_device),
        )

    with pytest.raises(NotImplementedError, match="tangential or thin-prism"):
        gm.fully_fused_projection_with_ut(
            means.to(mps_device),
            quats.to(mps_device),
            scales.to(mps_device),
            opacities.to(mps_device),
            viewmats.to(mps_device),
            Ks.to(mps_device),
            width,
            height,
            camera_model="fisheye",
            tangential_coeffs=torch.zeros((1, 1, 2), dtype=torch.float32, device=mps_device),
        )
