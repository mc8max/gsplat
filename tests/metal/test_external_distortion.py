# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for Metal external-distortion helper ops."""

import math
import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.cuda._wrapper import ExternalDistortionReferencePolynomial
from gsplat.cuda._torch_external_distortion import (
    make_identity_horizontal_poly,
    make_identity_vertical_poly,
    make_zero_poly,
    num_coeffs_for_order,
    ref_eval_bivariate_poly,
    ref_distort_camera_ray,
    ref_compute_order,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _eval_reference(coeffs, order, x_vals, y_vals):
    return torch.tensor(
        [
            ref_eval_bivariate_poly(coeffs.tolist(), order, float(xv), float(yv))
            for xv, yv in zip(x_vals.tolist(), y_vals.tolist())
        ],
        dtype=torch.float32,
    )


def _assert_eval_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


def _distort_reference(rays_cpu, h_poly, v_poly):
    h_order = ref_compute_order(len(h_poly))
    v_order = ref_compute_order(len(v_poly))
    return torch.tensor(
        [
            ref_distort_camera_ray(tuple(ray.tolist()), h_poly, v_poly, h_order, v_order)
            for ray in rays_cpu
        ],
        dtype=torch.float32,
    )


def _assert_ray_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


def test_eval_bivariate_poly_constant(mps_device):
    x = torch.tensor([0.0, 1.0], dtype=torch.float32, device=mps_device)
    y = torch.tensor([0.0, 2.0], dtype=torch.float32, device=mps_device)
    coeffs = torch.tensor([3.5], dtype=torch.float32, device=mps_device)
    result = gm.eval_bivariate_poly(x, y, coeffs, 0)
    expected = torch.tensor([3.5, 3.5], dtype=torch.float32)
    _assert_eval_close(result, expected, atol=1e-6, rtol=1e-6)


def test_eval_bivariate_poly_linear_identity_horizontal(mps_device):
    x = torch.tensor([0.5], dtype=torch.float32, device=mps_device)
    y = torch.tensor([0.7], dtype=torch.float32, device=mps_device)
    coeffs = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=mps_device)
    result = gm.eval_bivariate_poly(x, y, coeffs, 1)
    expected = torch.tensor([0.5], dtype=torch.float32)
    _assert_eval_close(result, expected, atol=1e-6, rtol=1e-6)


def test_eval_bivariate_poly_linear_identity_vertical(mps_device):
    x = torch.tensor([0.5], dtype=torch.float32, device=mps_device)
    y = torch.tensor([0.7], dtype=torch.float32, device=mps_device)
    coeffs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=mps_device)
    result = gm.eval_bivariate_poly(x, y, coeffs, 1)
    expected = torch.tensor([0.7], dtype=torch.float32)
    _assert_eval_close(result, expected, atol=1e-6, rtol=1e-6)


def test_eval_bivariate_poly_linear_general(mps_device):
    x = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float32, device=mps_device)
    y = torch.tensor([1.0, 0.0, 0.0, 3.0], dtype=torch.float32, device=mps_device)
    coeffs = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32, device=mps_device)
    result = gm.eval_bivariate_poly(x, y, coeffs, 1)
    expected = torch.tensor([10.0, 2.0, 8.0, 17.0], dtype=torch.float32)
    _assert_eval_close(result, expected, atol=1e-5, rtol=1e-5)


def test_eval_bivariate_poly_quadratic(mps_device):
    x = torch.tensor([3.0], dtype=torch.float32, device=mps_device)
    y = torch.tensor([4.0], dtype=torch.float32, device=mps_device)

    result = gm.eval_bivariate_poly(
        x,
        y,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=mps_device),
        2,
    )
    _assert_eval_close(result, torch.tensor([1.0], dtype=torch.float32), atol=1e-5, rtol=1e-5)

    result = gm.eval_bivariate_poly(
        x,
        y,
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=mps_device),
        2,
    )
    _assert_eval_close(result, torch.tensor([25.0], dtype=torch.float32), atol=1e-4, rtol=1e-4)

    result = gm.eval_bivariate_poly(
        x,
        y,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=torch.float32, device=mps_device),
        2,
    )
    _assert_eval_close(result, torch.tensor([24.0], dtype=torch.float32), atol=1e-4, rtol=1e-4)


def test_eval_bivariate_poly_at_zero(mps_device):
    x = torch.tensor([0.0], dtype=torch.float32, device=mps_device)
    y = torch.tensor([0.0], dtype=torch.float32, device=mps_device)
    for order in range(6):
        coeffs = torch.tensor(
            [float(i + 1) for i in range(num_coeffs_for_order(order))],
            dtype=torch.float32,
            device=mps_device,
        )
        result = gm.eval_bivariate_poly(x, y, coeffs, order)
        _assert_eval_close(
            result,
            torch.tensor([coeffs[0].item()], dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )


def test_eval_bivariate_poly_matches_reference(mps_device):
    torch.manual_seed(42)
    for order in range(6):
        coeffs_cpu = torch.randn(num_coeffs_for_order(order), dtype=torch.float32)
        x_cpu = torch.rand(50, dtype=torch.float32) * 2.0 - 1.0
        y_cpu = torch.rand(50, dtype=torch.float32) * 2.0 - 1.0
        expected = _eval_reference(coeffs_cpu, order, x_cpu, y_cpu)
        actual = gm.eval_bivariate_poly(
            x_cpu.to(mps_device),
            y_cpu.to(mps_device),
            coeffs_cpu.to(mps_device),
            order,
        )
        _assert_eval_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_eval_bivariate_poly_empty(mps_device):
    x = torch.empty((0,), dtype=torch.float32, device=mps_device)
    y = torch.empty((0,), dtype=torch.float32, device=mps_device)
    coeffs = torch.tensor([3.5], dtype=torch.float32, device=mps_device)
    result = gm.eval_bivariate_poly(x, y, coeffs, 0)
    assert result.shape == (0,)


def test_eval_bivariate_poly_2d_input(mps_device):
    """Output shape follows x.shape for non-1D inputs."""
    torch.manual_seed(7)
    order = 3
    rows, cols = 4, 8
    coeffs_cpu = torch.randn(num_coeffs_for_order(order), dtype=torch.float32)
    x_cpu = torch.rand(rows, cols, dtype=torch.float32) * 2.0 - 1.0
    y_cpu = torch.rand(rows, cols, dtype=torch.float32) * 2.0 - 1.0

    actual = gm.eval_bivariate_poly(
        x_cpu.to(mps_device),
        y_cpu.to(mps_device),
        coeffs_cpu.to(mps_device),
        order,
    )

    assert actual.shape == (rows, cols)
    expected = _eval_reference(coeffs_cpu, order, x_cpu.flatten(), y_cpu.flatten())
    _assert_eval_close(actual.cpu().flatten(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [1_000, 10_000])
def test_eval_bivariate_poly_large(mps_device, n):
    torch.manual_seed(0)
    order = 5
    coeffs_cpu = torch.randn(num_coeffs_for_order(order), dtype=torch.float32)
    x_cpu = torch.rand(n, dtype=torch.float32) * 2.0 - 1.0
    y_cpu = torch.rand(n, dtype=torch.float32) * 2.0 - 1.0
    expected = _eval_reference(coeffs_cpu, order, x_cpu, y_cpu)
    actual = gm.eval_bivariate_poly(
        x_cpu.to(mps_device),
        y_cpu.to(mps_device),
        coeffs_cpu.to(mps_device),
        order,
    )
    _assert_eval_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_eval_bivariate_poly_100k_spot_check(mps_device):
    """100k dispatch correctness without running the pure-Python reference on all points."""
    torch.manual_seed(1)
    order = 5
    spot = 500
    coeffs_cpu = torch.randn(num_coeffs_for_order(order), dtype=torch.float32)
    x_cpu = torch.rand(100_000, dtype=torch.float32) * 2.0 - 1.0
    y_cpu = torch.rand(100_000, dtype=torch.float32) * 2.0 - 1.0

    actual = gm.eval_bivariate_poly(
        x_cpu.to(mps_device),
        y_cpu.to(mps_device),
        coeffs_cpu.to(mps_device),
        order,
    ).cpu()

    expected = _eval_reference(coeffs_cpu, order, x_cpu[:spot], y_cpu[:spot])
    _assert_eval_close(actual[:spot], expected, atol=1e-4, rtol=1e-4)


def _mps_sync():
    torch.mps.synchronize()


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_eval_bivariate_poly(mps_device, n, capsys):
    torch.manual_seed(123)
    order = 5
    coeffs = torch.randn(num_coeffs_for_order(order), dtype=torch.float32, device=mps_device)
    x = (torch.rand(n, dtype=torch.float32, device=mps_device) * 2.0) - 1.0
    y = (torch.rand(n, dtype=torch.float32, device=mps_device) * 2.0) - 1.0

    for _ in range(3):
        gm.eval_bivariate_poly(x, y, coeffs, order)
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.eval_bivariate_poly(x, y, coeffs, order)
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] eval_bivariate_poly  n={n:>7,}  {elapsed_ms:.3f} ms/iter")


def test_distort_camera_rays_identity_on_axis_ray(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    rays = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    _assert_ray_close(result, expected, atol=1e-6, rtol=1e-6)


def test_distort_camera_rays_identity_on_negative_z_ray(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    rays = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    expected = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    _assert_ray_close(result, expected, atol=1e-6, rtol=1e-6)


def test_distort_camera_rays_identity_preserves_direction(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.tensor(
        [
            [0.3, 0.4, 0.8],
            [0.1, 0.0, 1.0],
            [0.0, 0.2, 0.9],
            [-0.3, 0.2, 0.7],
            [0.5, -0.3, 0.6],
        ],
        dtype=torch.float32,
    )
    expected = rays_cpu / torch.linalg.norm(rays_cpu, dim=-1, keepdim=True)
    result = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    _assert_ray_close(result, expected, atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_zero_poly_maps_to_z_axis(mps_device):
    h = make_zero_poly(1)
    v = make_zero_poly(1)
    rays = torch.tensor([[0.5, 0.3, 0.8]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    _assert_ray_close(result, expected, atol=1e-6, rtol=1e-6)


def test_distort_camera_rays_zero_poly_negative_z(mps_device):
    h = make_zero_poly(1)
    v = make_zero_poly(1)
    rays = torch.tensor([[0.5, 0.3, -0.8]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    expected = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    _assert_ray_close(result, expected, atol=1e-6, rtol=1e-6)


def test_distort_camera_rays_output_is_unit_length(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays = torch.tensor(
        [
            [0.3, 0.4, 0.8],
            [0.1, 0.0, 1.0],
            [0.0, 0.2, 0.9],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.7],
        ],
        dtype=torch.float32,
        device=mps_device,
    )
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    lengths = torch.linalg.norm(result, dim=-1)
    _assert_ray_close(lengths, torch.ones_like(lengths), atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_clamp_prevents_nan(mps_device):
    h = [math.pi / 2]
    v = [math.pi / 2]
    rays = torch.tensor([[0.3, 0.3, 0.8]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    assert torch.isfinite(result).all()
    _assert_ray_close(result[:, 2], torch.tensor([0.0], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    # When sin(h)² + sin(v)² > 1 the clamp forces z=0 and the output is NOT a
    # unit vector — this is the documented contract, matching the Python reference.
    lengths = torch.linalg.norm(result, dim=-1)
    assert (lengths > 1.0).all(), "expected non-unit output when poly > π/2"


def test_distort_camera_rays_known_constant_offset(mps_device):
    h = [0.1, 1.0, 0.0]
    v = [0.0, 0.0, 1.0]
    rays = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    expected = torch.tensor([[math.sin(0.1), 0.0, math.sqrt(1.0 - math.sin(0.1) ** 2)]], dtype=torch.float32)
    _assert_ray_close(result, expected, atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_horizontal_vertical_swap(mps_device):
    h = [0.1, 0.9, 0.05]
    v = [0.0, 0.0, 1.0]
    rays = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=mps_device)
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    result_normal = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    result_swapped = gm.distort_camera_rays(
        rays,
        v_t,
        h_t,
        v_t,
        h_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    _assert_ray_close(result_normal[:, 0], result_swapped[:, 1], atol=1e-5, rtol=1e-5)
    _assert_ray_close(result_normal[:, 1], result_swapped[:, 0], atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_inverse_flag_uses_inverse_polynomials(mps_device):
    h_fwd = [0.1, 1.0, 0.0]
    v_fwd = [0.0, 0.0, 1.0]
    h_inv = [-0.1, 1.0, 0.0]
    v_inv = [0.0, 0.0, 1.0]
    rays = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=mps_device)
    h_fwd_t = torch.tensor(h_fwd, dtype=torch.float32, device=mps_device)
    v_fwd_t = torch.tensor(v_fwd, dtype=torch.float32, device=mps_device)
    h_inv_t = torch.tensor(h_inv, dtype=torch.float32, device=mps_device)
    v_inv_t = torch.tensor(v_inv, dtype=torch.float32, device=mps_device)
    result_fwd = gm.distort_camera_rays(
        rays,
        h_fwd_t,
        v_fwd_t,
        h_inv_t,
        v_inv_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    result_inv = gm.distort_camera_rays(
        rays,
        h_fwd_t,
        v_fwd_t,
        h_inv_t,
        v_inv_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        True,
    )
    _assert_ray_close(result_fwd[:, 0], torch.tensor([math.sin(0.1)], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    _assert_ray_close(result_inv[:, 0], torch.tensor([math.sin(-0.1)], dtype=torch.float32), atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_matches_reference_identity(mps_device):
    torch.manual_seed(42)
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.randn(20, 3, dtype=torch.float32)
    rays_cpu[:, 2] = rays_cpu[:, 2].abs() + 0.5
    expected = _distort_reference(rays_cpu, h, v)
    actual = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    _assert_ray_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_matches_reference_nonidentity(mps_device):
    torch.manual_seed(123)
    h = [0.05, 0.95, 0.02]
    v = [-0.02, 0.01, 0.98]
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.randn(20, 3, dtype=torch.float32)
    rays_cpu[:, 2] = rays_cpu[:, 2].abs() + 0.5
    expected = _distort_reference(rays_cpu, h, v)
    actual = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    _assert_ray_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_distort_camera_rays_batch_consistency(mps_device):
    h = [0.05, 1.0, 0.0]
    v = [0.0, 0.0, 1.0]
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    ray = torch.tensor([[0.3, 0.2, 0.9]], dtype=torch.float32, device=mps_device)
    result_single = gm.distort_camera_rays(
        ray,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    result_batch = gm.distort_camera_rays(
        ray.repeat(10, 1),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    for i in range(10):
        _assert_ray_close(result_single[0], result_batch[i], atol=1e-7, rtol=1e-7)


def test_distort_camera_rays_boundary_rays_asin_clamp(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1e-30],
            [0.0, 1.0, 1e-30],
            [-1.0, 0.0, 1e-30],
            [0.0, -1.0, 1e-30],
        ],
        dtype=torch.float32,
    )
    actual = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    assert torch.isfinite(actual).all()
    expected = _distort_reference(rays_cpu, h, v)
    _assert_ray_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_distort_camera_rays_empty(mps_device):
    h = make_identity_horizontal_poly()
    v = make_identity_vertical_poly()
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays = torch.empty((0, 3), dtype=torch.float32, device=mps_device)
    result = gm.distort_camera_rays(
        rays,
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    assert result.shape == (0, 3)


@pytest.mark.parametrize("n", [1_000, 10_000])
def test_distort_camera_rays_large(mps_device, n):
    torch.manual_seed(9)
    h = [0.05, 0.95, 0.02]
    v = [-0.02, 0.01, 0.98]
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.randn(n, 3, dtype=torch.float32)
    rays_cpu[:, 2] = rays_cpu[:, 2].abs() + 0.5
    expected = _distort_reference(rays_cpu, h, v)
    actual = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    )
    _assert_ray_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_distort_camera_rays_100k_spot_check(mps_device):
    torch.manual_seed(11)
    spot = 500
    h = [0.05, 0.95, 0.02]
    v = [-0.02, 0.01, 0.98]
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays_cpu = torch.randn(100_000, 3, dtype=torch.float32)
    rays_cpu[:, 2] = rays_cpu[:, 2].abs() + 0.5
    actual = gm.distort_camera_rays(
        rays_cpu.to(mps_device),
        h_t,
        v_t,
        h_t,
        v_t,
        int(ExternalDistortionReferencePolynomial.FORWARD),
        False,
    ).cpu()
    expected = _distort_reference(rays_cpu[:spot], h, v)
    _assert_ray_close(actual[:spot], expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_distort_camera_rays(mps_device, n, capsys):
    torch.manual_seed(1234)
    h = [0.05, 0.95, 0.02]
    v = [-0.02, 0.01, 0.98]
    h_t = torch.tensor(h, dtype=torch.float32, device=mps_device)
    v_t = torch.tensor(v, dtype=torch.float32, device=mps_device)
    rays = torch.randn(n, 3, dtype=torch.float32, device=mps_device)
    rays[:, 2] = rays[:, 2].abs() + 0.5

    for _ in range(3):
        gm.distort_camera_rays(
            rays,
            h_t,
            v_t,
            h_t,
            v_t,
            int(ExternalDistortionReferencePolynomial.FORWARD),
            False,
        )
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.distort_camera_rays(
            rays,
            h_t,
            v_t,
            h_t,
            v_t,
            int(ExternalDistortionReferencePolynomial.FORWARD),
            False,
        )
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] distort_camera_rays  n={n:>7,}  {elapsed_ms:.3f} ms/iter")
