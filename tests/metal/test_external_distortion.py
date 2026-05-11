# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for Metal external-distortion helper ops."""

import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.cuda._torch_external_distortion import (
    num_coeffs_for_order,
    ref_eval_bivariate_poly,
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
