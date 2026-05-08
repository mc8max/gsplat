# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Metal quat-scale-to-covariance/precision custom op."""

import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _quat_scale_to_covar_preci

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_inputs(shape):
    """Generate numerically stable test inputs for the Metal op."""

    quats = torch.randn(*shape, 4, dtype=torch.float32)
    scales = torch.rand(*shape, 3, dtype=torch.float32) + 0.25
    return quats, scales


def _assert_optional_close(actual, expected, atol=1e-4, rtol=1e-4):
    """Assert closeness while allowing an expected ``None`` result."""

    if expected is None:
        assert actual is None
        return
    assert actual is not None
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("triu", [False, True])
def test_fwd_matches_reference(mps_device, triu):
    quats_cpu, scales_cpu = _sample_inputs((8,))
    expected_covars, expected_precis = _quat_scale_to_covar_preci(
        quats_cpu, scales_cpu, compute_covar=True, compute_preci=True, triu=triu
    )

    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device), scales_cpu.to(mps_device), triu=triu
    )

    _assert_optional_close(covars, expected_covars)
    _assert_optional_close(precis, expected_precis)


def test_fwd_triu_shape(mps_device):
    quats_cpu, scales_cpu = _sample_inputs((5,))
    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device), scales_cpu.to(mps_device), triu=True
    )
    assert covars is not None and precis is not None
    assert covars.shape == (5, 6)
    assert precis.shape == (5, 6)


def test_fwd_compute_covar_only(mps_device):
    quats_cpu, scales_cpu = _sample_inputs((6,))
    expected_covars, _ = _quat_scale_to_covar_preci(
        quats_cpu, scales_cpu, compute_covar=True, compute_preci=False, triu=False
    )

    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device),
        scales_cpu.to(mps_device),
        compute_covar=True,
        compute_preci=False,
    )

    _assert_optional_close(covars, expected_covars)
    assert precis is None


def test_fwd_compute_preci_only(mps_device):
    quats_cpu, scales_cpu = _sample_inputs((6,))
    _, expected_precis = _quat_scale_to_covar_preci(
        quats_cpu, scales_cpu, compute_covar=False, compute_preci=True, triu=False
    )

    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device),
        scales_cpu.to(mps_device),
        compute_covar=False,
        compute_preci=True,
    )

    assert covars is None
    _assert_optional_close(precis, expected_precis)


def test_fwd_empty_input(mps_device):
    quats = torch.empty((0, 4), device=mps_device)
    scales = torch.empty((0, 3), device=mps_device)
    covars, precis = gm.quat_scale_to_covar_preci(quats, scales, triu=False)
    assert covars is not None and precis is not None
    assert covars.shape == (0, 3, 3)
    assert precis.shape == (0, 3, 3)


@pytest.mark.parametrize("triu", [False, True])
def test_fwd_batch_dims(mps_device, triu):
    quats_cpu, scales_cpu = _sample_inputs((3, 8))
    expected_covars, expected_precis = _quat_scale_to_covar_preci(
        quats_cpu, scales_cpu, compute_covar=True, compute_preci=True, triu=triu
    )

    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device), scales_cpu.to(mps_device), triu=triu
    )

    assert covars is not None and precis is not None
    expected_shape = (3, 8, 6) if triu else (3, 8, 3, 3)
    assert covars.shape == expected_shape
    assert precis.shape == expected_shape
    _assert_optional_close(covars, expected_covars)
    _assert_optional_close(precis, expected_precis)


@pytest.mark.parametrize("triu", [False, True])
def test_bwd_matches_reference(mps_device, triu):
    quats_cpu, scales_cpu = _sample_inputs((4,))
    quats_ref = quats_cpu.clone().requires_grad_(True)
    scales_ref = scales_cpu.clone().requires_grad_(True)
    covars_ref, precis_ref = _quat_scale_to_covar_preci(
        quats_ref, scales_ref, compute_covar=True, compute_preci=True, triu=triu
    )
    assert covars_ref is not None and precis_ref is not None
    loss_ref = covars_ref.sum() + 0.25 * precis_ref.sum()
    loss_ref.backward()

    quats_mps = quats_cpu.to(mps_device).requires_grad_(True)
    scales_mps = scales_cpu.to(mps_device).requires_grad_(True)
    covars_mps, precis_mps = gm.quat_scale_to_covar_preci(
        quats_mps, scales_mps, compute_covar=True, compute_preci=True, triu=triu
    )
    assert covars_mps is not None and precis_mps is not None
    loss_mps = covars_mps.sum() + 0.25 * precis_mps.sum()
    loss_mps.backward()

    assert quats_mps.grad is not None and scales_mps.grad is not None
    torch.testing.assert_close(
        quats_mps.grad.cpu(), quats_ref.grad, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        scales_mps.grad.cpu(), scales_ref.grad, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("triu", [False, True])
def test_gradcheck(mps_device, triu):
    quats = (torch.randn(2, 4, device=mps_device, dtype=torch.float32) * 0.2).requires_grad_(True)
    scales = (torch.rand(2, 3, device=mps_device, dtype=torch.float32) + 0.5).requires_grad_(True)

    def fn(quats_, scales_):
        covars, precis = gm.quat_scale_to_covar_preci(
            quats_, scales_, compute_covar=True, compute_preci=True, triu=triu
        )
        assert covars is not None and precis is not None
        return covars, precis

    assert torch.autograd.gradcheck(
        fn,
        (quats, scales),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True,
    )


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("triu", [False, True])
def test_fwd_large(mps_device, n, triu):
    quats_cpu, scales_cpu = _sample_inputs((n,))
    expected_covars, expected_precis = _quat_scale_to_covar_preci(
        quats_cpu, scales_cpu, compute_covar=True, compute_preci=True, triu=triu
    )

    covars, precis = gm.quat_scale_to_covar_preci(
        quats_cpu.to(mps_device), scales_cpu.to(mps_device), triu=triu
    )

    _assert_optional_close(covars, expected_covars, atol=1e-4, rtol=1e-4)
    _assert_optional_close(precis, expected_precis, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("triu", [False, True])
def test_bwd_large(mps_device, n, triu):
    quats_cpu, scales_cpu = _sample_inputs((n,))
    quats_ref = quats_cpu.clone().requires_grad_(True)
    scales_ref = scales_cpu.clone().requires_grad_(True)
    covars_ref, precis_ref = _quat_scale_to_covar_preci(
        quats_ref, scales_ref, compute_covar=True, compute_preci=True, triu=triu
    )
    assert covars_ref is not None and precis_ref is not None
    (covars_ref.sum() + 0.25 * precis_ref.sum()).backward()

    quats_mps = quats_cpu.to(mps_device).requires_grad_(True)
    scales_mps = scales_cpu.to(mps_device).requires_grad_(True)
    covars_mps, precis_mps = gm.quat_scale_to_covar_preci(
        quats_mps, scales_mps, compute_covar=True, compute_preci=True, triu=triu
    )
    assert covars_mps is not None and precis_mps is not None
    (covars_mps.sum() + 0.25 * precis_mps.sum()).backward()

    assert quats_mps.grad is not None and scales_mps.grad is not None
    torch.testing.assert_close(
        quats_mps.grad.cpu(), quats_ref.grad, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        scales_mps.grad.cpu(), scales_ref.grad, atol=1e-3, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# Performance benchmarks — run with pytest -s to see timing output
# ---------------------------------------------------------------------------

def _mps_sync():
    torch.mps.synchronize()


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd(mps_device, n, capsys):
    quats, scales = _sample_inputs((n,))
    quats = quats.to(mps_device)
    scales = scales.to(mps_device)

    for _ in range(3):  # warmup
        gm.quat_scale_to_covar_preci(quats, scales, triu=False)
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.quat_scale_to_covar_preci(quats, scales, triu=False)
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] fwd        n={n:>7,}  {elapsed_ms:.3f} ms/iter")


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd_bwd(mps_device, n, capsys):
    quats, scales = _sample_inputs((n,))
    quats = quats.to(mps_device).requires_grad_(True)
    scales = scales.to(mps_device).requires_grad_(True)

    for _ in range(3):  # warmup
        covars, precis = gm.quat_scale_to_covar_preci(quats, scales, triu=False)
        assert covars is not None and precis is not None
        (covars.sum() + precis.sum()).backward()
        quats.grad = None
        scales.grad = None
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        covars, precis = gm.quat_scale_to_covar_preci(quats, scales, triu=False)
        assert covars is not None and precis is not None
        (covars.sum() + precis.sum()).backward()
        quats.grad = None
        scales.grad = None
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] fwd+bwd    n={n:>7,}  {elapsed_ms:.3f} ms/iter")
