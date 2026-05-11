# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Metal spherical harmonics custom op."""

import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _spherical_harmonics

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_inputs(shape, degree):
    dirs = torch.randn(*shape, 3, dtype=torch.float32)
    coeffs = torch.randn(*shape, (degree + 1) ** 2, 3, dtype=torch.float32)
    return dirs, coeffs


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_fwd_matches_reference(mps_device, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((16,), degree)
    expected = _spherical_harmonics(degree, dirs_cpu, coeffs_cpu)

    actual = gm.spherical_harmonics(
        degree, dirs_cpu.to(mps_device), coeffs_cpu.to(mps_device)
    )

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


def test_fwd_masked(mps_device):
    degree = 4
    dirs_cpu, coeffs_cpu = _sample_inputs((12,), degree)
    masks_cpu = torch.tensor(
        [True, False, True, False, True, True, False, True, False, True, True, False]
    )
    expected = _spherical_harmonics(degree, dirs_cpu, coeffs_cpu) * masks_cpu[:, None]

    actual = gm.spherical_harmonics(
        degree,
        dirs_cpu.to(mps_device),
        coeffs_cpu.to(mps_device),
        masks_cpu.to(mps_device),
    )

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


def test_fwd_empty(mps_device):
    dirs = torch.empty((0, 3), device=mps_device, dtype=torch.float32)
    coeffs = torch.empty((0, 25, 3), device=mps_device, dtype=torch.float32)
    colors = gm.spherical_harmonics(4, dirs, coeffs)
    assert colors.shape == (0, 3)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_bwd_v_coeffs(mps_device, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((8,), degree)
    dirs_ref = dirs_cpu.clone().requires_grad_(True)
    coeffs_ref = coeffs_cpu.clone().requires_grad_(True)
    colors_ref = _spherical_harmonics(degree, dirs_ref, coeffs_ref)
    (colors_ref.sum()).backward()

    dirs_mps = dirs_cpu.to(mps_device).requires_grad_(True)
    coeffs_mps = coeffs_cpu.to(mps_device).requires_grad_(True)
    colors_mps = gm.spherical_harmonics(degree, dirs_mps, coeffs_mps)
    (colors_mps.sum()).backward()

    assert coeffs_mps.grad is not None
    torch.testing.assert_close(coeffs_mps.grad.cpu(), coeffs_ref.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_bwd_v_dirs(mps_device, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((8,), degree)
    dirs_ref = dirs_cpu.clone().requires_grad_(True)
    coeffs_ref = coeffs_cpu.clone().requires_grad_(True)
    colors_ref = _spherical_harmonics(degree, dirs_ref, coeffs_ref)
    (colors_ref.sum()).backward()

    dirs_mps = dirs_cpu.to(mps_device).requires_grad_(True)
    coeffs_mps = coeffs_cpu.to(mps_device).requires_grad_(True)
    colors_mps = gm.spherical_harmonics(degree, dirs_mps, coeffs_mps)
    (colors_mps.sum()).backward()

    assert dirs_mps.grad is not None
    torch.testing.assert_close(dirs_mps.grad.cpu(), dirs_ref.grad, atol=1e-3, rtol=1e-3)


def test_bwd_no_v_dirs(mps_device):
    """Calling the op directly: compute_v_dirs=False returns None for v_dirs."""
    assert gm.has_metal()
    degree = 4
    dirs_cpu, coeffs_cpu = _sample_inputs((6,), degree)
    v_colors = torch.randn(6, 3, dtype=torch.float32, device=mps_device)
    v_coeffs, v_dirs = torch.ops.gsplat.metal_spherical_harmonics_bwd(
        degree,
        dirs_cpu.to(mps_device),
        coeffs_cpu.to(mps_device),
        None,
        v_colors,
        False,
    )
    assert v_coeffs.shape == (6, 25, 3)
    assert v_dirs is None


def test_bwd_no_v_dirs_via_autograd(mps_device):
    """Autograd path: when dirs has no requires_grad, its gradient is None."""
    degree = 3
    dirs_cpu, coeffs_cpu = _sample_inputs((6,), degree)
    dirs_mps = dirs_cpu.to(mps_device)           # no requires_grad
    coeffs_mps = coeffs_cpu.to(mps_device).requires_grad_(True)
    colors = gm.spherical_harmonics(degree, dirs_mps, coeffs_mps)
    colors.sum().backward()
    assert dirs_mps.grad is None
    assert coeffs_mps.grad is not None


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_gradcheck(mps_device, degree):
    dirs = (torch.randn(2, 3, device=mps_device, dtype=torch.float32) * 0.2).requires_grad_(True)
    coeffs = (
        torch.randn(2, (degree + 1) ** 2, 3, device=mps_device, dtype=torch.float32) * 0.2
    ).requires_grad_(True)

    def fn(dirs_, coeffs_):
        return gm.spherical_harmonics(degree, dirs_, coeffs_)

    assert torch.autograd.gradcheck(
        fn,
        (dirs, coeffs),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True,
    )


# ---------------------------------------------------------------------------
# L2 — batch dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [0, 2, 4])
def test_fwd_batch_dims(mps_device, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((3, 8), degree)
    expected = _spherical_harmonics(degree, dirs_cpu, coeffs_cpu)

    actual = gm.spherical_harmonics(
        degree, dirs_cpu.to(mps_device), coeffs_cpu.to(mps_device)
    )

    assert actual.shape == (3, 8, 3)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("degree", [1, 3])
def test_bwd_batch_dims(mps_device, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((3, 8), degree)
    dirs_ref = dirs_cpu.clone().requires_grad_(True)
    coeffs_ref = coeffs_cpu.clone().requires_grad_(True)
    _spherical_harmonics(degree, dirs_ref, coeffs_ref).sum().backward()

    dirs_mps = dirs_cpu.to(mps_device).requires_grad_(True)
    coeffs_mps = coeffs_cpu.to(mps_device).requires_grad_(True)
    gm.spherical_harmonics(degree, dirs_mps, coeffs_mps).sum().backward()

    assert dirs_mps.grad is not None and coeffs_mps.grad is not None
    torch.testing.assert_close(dirs_mps.grad.cpu(), dirs_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(coeffs_mps.grad.cpu(), coeffs_ref.grad, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# L1 — large-N correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("degree", [1, 4])
def test_fwd_large(mps_device, n, degree):
    dirs_cpu, coeffs_cpu = _sample_inputs((n,), degree)
    expected = _spherical_harmonics(degree, dirs_cpu, coeffs_cpu)

    actual = gm.spherical_harmonics(
        degree, dirs_cpu.to(mps_device), coeffs_cpu.to(mps_device)
    )

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_bwd_large(mps_device, n):
    degree = 4
    dirs_cpu, coeffs_cpu = _sample_inputs((n,), degree)
    dirs_ref = dirs_cpu.clone().requires_grad_(True)
    coeffs_ref = coeffs_cpu.clone().requires_grad_(True)
    _spherical_harmonics(degree, dirs_ref, coeffs_ref).sum().backward()

    dirs_mps = dirs_cpu.to(mps_device).requires_grad_(True)
    coeffs_mps = coeffs_cpu.to(mps_device).requires_grad_(True)
    gm.spherical_harmonics(degree, dirs_mps, coeffs_mps).sum().backward()

    assert dirs_mps.grad is not None and coeffs_mps.grad is not None
    torch.testing.assert_close(dirs_mps.grad.cpu(), dirs_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(coeffs_mps.grad.cpu(), coeffs_ref.grad, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# L3 — performance benchmarks (pytest -s to see output)
# ---------------------------------------------------------------------------

def _mps_sync():
    torch.mps.synchronize()


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd(mps_device, n, capsys):
    degree = 3
    dirs, coeffs = _sample_inputs((n,), degree)
    dirs = dirs.to(mps_device)
    coeffs = coeffs.to(mps_device)

    for _ in range(3):  # warmup
        gm.spherical_harmonics(degree, dirs, coeffs)
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.spherical_harmonics(degree, dirs, coeffs)
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] sh fwd      degree={degree}  n={n:>7,}  {elapsed_ms:.3f} ms/iter")


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd_bwd(mps_device, n, capsys):
    degree = 3
    dirs, coeffs = _sample_inputs((n,), degree)
    dirs = dirs.to(mps_device).requires_grad_(True)
    coeffs = coeffs.to(mps_device).requires_grad_(True)

    for _ in range(3):  # warmup
        gm.spherical_harmonics(degree, dirs, coeffs).sum().backward()
        dirs.grad = None
        coeffs.grad = None
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.spherical_harmonics(degree, dirs, coeffs).sum().backward()
        dirs.grad = None
        coeffs.grad = None
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] sh fwd+bwd  degree={degree}  n={n:>7,}  {elapsed_ms:.3f} ms/iter")
