# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Metal projection_ewa_simple custom op."""

import time

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fisheye_proj, _ortho_proj, _persp_proj

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _mps_sync():
    torch.mps.synchronize()


def _sample_inputs(batch_shape=(), c=2, n=8, width=640, height=480):
    means = torch.randn(*batch_shape, c, n, 3, dtype=torch.float32) * 0.25
    means[..., 2] = torch.rand(*batch_shape, c, n, dtype=torch.float32) * 2.0 + 1.5

    A = torch.randn(*batch_shape, c, n, 3, 3, dtype=torch.float32) * 0.05
    eye = torch.eye(3, dtype=torch.float32).expand(*batch_shape, c, n, 3, 3)
    covars = A @ A.transpose(-1, -2) + 0.05 * eye

    Ks = torch.zeros(*batch_shape, c, 3, 3, dtype=torch.float32)
    Ks[..., 0, 0] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 1, 1] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0
    return means, covars, Ks


def _reference_projection(camera_model, means, covars, Ks, width, height):
    if camera_model == "pinhole":
        return _persp_proj(means, covars, Ks, width, height)
    if camera_model == "ortho":
        return _ortho_proj(means, covars, Ks, width, height)
    if camera_model == "fisheye":
        return _fisheye_proj(means, covars, Ks, width, height)
    raise AssertionError(f"unsupported camera_model {camera_model}")


def _assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_fwd_matches_reference(mps_device, camera_model):
    width, height = 640, 480
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(c=3, n=16, width=width, height=height)
    expected_means2d, expected_covars2d = _reference_projection(
        camera_model, means_cpu, covars_cpu, Ks_cpu, width, height
    )

    actual_means2d, actual_covars2d = gm.projection_ewa_simple(
        means_cpu.to(mps_device),
        covars_cpu.to(mps_device),
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model=camera_model,
    )

    _assert_close(actual_means2d, expected_means2d, atol=2e-4, rtol=2e-4)
    _assert_close(actual_covars2d, expected_covars2d, atol=2e-4, rtol=2e-4)


def test_fwd_batch_dims(mps_device):
    width, height = 800, 600
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(batch_shape=(2,), c=3, n=7, width=width, height=height)
    expected_means2d, expected_covars2d = _persp_proj(means_cpu, covars_cpu, Ks_cpu, width, height)

    actual_means2d, actual_covars2d = gm.projection_ewa_simple(
        means_cpu.to(mps_device),
        covars_cpu.to(mps_device),
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model="pinhole",
    )

    assert actual_means2d.shape == (2, 3, 7, 2)
    assert actual_covars2d.shape == (2, 3, 7, 2, 2)
    _assert_close(actual_means2d, expected_means2d, atol=2e-4, rtol=2e-4)
    _assert_close(actual_covars2d, expected_covars2d, atol=2e-4, rtol=2e-4)


def test_fwd_empty(mps_device):
    means = torch.empty((2, 0, 3), device=mps_device, dtype=torch.float32)
    covars = torch.empty((2, 0, 3, 3), device=mps_device, dtype=torch.float32)
    Ks = torch.eye(3, device=mps_device, dtype=torch.float32).repeat(2, 1, 1)
    means2d, covars2d = gm.projection_ewa_simple(means, covars, Ks, 640, 480, camera_model="pinhole")
    assert means2d.shape == (2, 0, 2)
    assert covars2d.shape == (2, 0, 2, 2)


def test_pinhole_clipping_matches_reference(mps_device):
    width, height = 640, 480
    means = torch.tensor([[[[20.0, -15.0, 1.0]]]], dtype=torch.float32)
    covars = torch.eye(3, dtype=torch.float32).reshape(1, 1, 1, 3, 3) * 0.1
    Ks = torch.tensor(
        [[[[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    )
    expected_means2d, expected_covars2d = _persp_proj(means, covars, Ks, width, height)

    actual_means2d, actual_covars2d = gm.projection_ewa_simple(
        means.to(mps_device),
        covars.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        camera_model="pinhole",
    )

    _assert_close(actual_means2d, expected_means2d, atol=1e-4, rtol=1e-4)
    _assert_close(actual_covars2d, expected_covars2d, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_bwd_matches_reference(mps_device, camera_model):
    width, height = 640, 480
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(c=2, n=5, width=width, height=height)

    means_ref = means_cpu.clone().requires_grad_(True)
    covars_ref = covars_cpu.clone().requires_grad_(True)
    means2d_ref, covars2d_ref = _reference_projection(
        camera_model, means_ref, covars_ref, Ks_cpu, width, height
    )
    v_means2d = torch.randn_like(means2d_ref)
    v_covars2d = torch.randn_like(covars2d_ref)
    ref_grads = torch.autograd.grad(
        (means2d_ref, covars2d_ref),
        (means_ref, covars_ref),
        grad_outputs=(v_means2d, v_covars2d),
    )

    means_mps = means_cpu.to(mps_device).requires_grad_(True)
    covars_mps = covars_cpu.to(mps_device).requires_grad_(True)
    means2d_mps, covars2d_mps = gm.projection_ewa_simple(
        means_mps,
        covars_mps,
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model=camera_model,
    )
    mps_grads = torch.autograd.grad(
        (means2d_mps, covars2d_mps),
        (means_mps, covars_mps),
        grad_outputs=(v_means2d.to(mps_device), v_covars2d.to(mps_device)),
    )

    _assert_close(mps_grads[0], ref_grads[0], atol=3e-3, rtol=3e-3)
    _assert_close(mps_grads[1], ref_grads[1], atol=3e-3, rtol=3e-3)


def test_bwd_batch_dims(mps_device):
    width, height = 800, 600
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(batch_shape=(2,), c=2, n=4, width=width, height=height)

    means_ref = means_cpu.clone().requires_grad_(True)
    covars_ref = covars_cpu.clone().requires_grad_(True)
    means2d_ref, covars2d_ref = _fisheye_proj(means_ref, covars_ref, Ks_cpu, width, height)
    v_means2d = torch.randn_like(means2d_ref)
    v_covars2d = torch.randn_like(covars2d_ref)
    ref_grads = torch.autograd.grad(
        (means2d_ref, covars2d_ref),
        (means_ref, covars_ref),
        grad_outputs=(v_means2d, v_covars2d),
    )

    means_mps = means_cpu.to(mps_device).requires_grad_(True)
    covars_mps = covars_cpu.to(mps_device).requires_grad_(True)
    means2d_mps, covars2d_mps = gm.projection_ewa_simple(
        means_mps,
        covars_mps,
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model="fisheye",
    )
    mps_grads = torch.autograd.grad(
        (means2d_mps, covars2d_mps),
        (means_mps, covars_mps),
        grad_outputs=(v_means2d.to(mps_device), v_covars2d.to(mps_device)),
    )

    _assert_close(mps_grads[0], ref_grads[0], atol=3e-3, rtol=3e-3)
    _assert_close(mps_grads[1], ref_grads[1], atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("n", [1_000, 10_000])
def test_fwd_large(mps_device, n):
    width, height = 640, 480
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(c=1, n=n, width=width, height=height)
    expected_means2d, expected_covars2d = _persp_proj(means_cpu, covars_cpu, Ks_cpu, width, height)

    actual_means2d, actual_covars2d = gm.projection_ewa_simple(
        means_cpu.to(mps_device),
        covars_cpu.to(mps_device),
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model="pinhole",
    )

    _assert_close(actual_means2d, expected_means2d, atol=2e-4, rtol=2e-4)
    _assert_close(actual_covars2d, expected_covars2d, atol=2e-4, rtol=2e-4)


def test_fwd_100k_spotcheck(mps_device):
    width, height = 640, 480
    means_cpu, covars_cpu, Ks_cpu = _sample_inputs(c=1, n=100_000, width=width, height=height)
    actual_means2d, actual_covars2d = gm.projection_ewa_simple(
        means_cpu.to(mps_device),
        covars_cpu.to(mps_device),
        Ks_cpu.to(mps_device),
        width,
        height,
        camera_model="pinhole",
    )
    expected_means2d, expected_covars2d = _persp_proj(means_cpu[:1, :512], covars_cpu[:1, :512], Ks_cpu, width, height)

    _assert_close(actual_means2d.cpu()[:1, :512], expected_means2d, atol=2e-4, rtol=2e-4)
    _assert_close(actual_covars2d.cpu()[:1, :512], expected_covars2d, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd(mps_device, n, capsys):
    width, height = 640, 480
    means, covars, Ks = _sample_inputs(c=1, n=n, width=width, height=height)
    means = means.to(mps_device)
    covars = covars.to(mps_device)
    Ks = Ks.to(mps_device)

    for _ in range(5):
        gm.projection_ewa_simple(means, covars, Ks, width, height, camera_model="pinhole")
    _mps_sync()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gm.projection_ewa_simple(means, covars, Ks, width, height, camera_model="pinhole")
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] projection_ewa_simple fwd      n={n:>7,}  {elapsed_ms:.3f} ms/iter")


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_perf_fwd_bwd(mps_device, n, capsys):
    width, height = 640, 480
    means, covars, Ks = _sample_inputs(c=1, n=n, width=width, height=height)
    means = means.to(mps_device).requires_grad_(True)
    covars = covars.to(mps_device).requires_grad_(True)
    Ks = Ks.to(mps_device)

    for _ in range(3):
        means2d, covars2d = gm.projection_ewa_simple(
            means, covars, Ks, width, height, camera_model="pinhole"
        )
        (means2d.sum() + covars2d.sum()).backward()
        means.grad = None
        covars.grad = None
    _mps_sync()

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        means2d, covars2d = gm.projection_ewa_simple(
            means, covars, Ks, width, height, camera_model="pinhole"
        )
        (means2d.sum() + covars2d.sum()).backward()
        means.grad = None
        covars.grad = None
    _mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    with capsys.disabled():
        print(f"\n[perf] projection_ewa_simple fwd+bwd  n={n:>7,}  {elapsed_ms:.3f} ms/iter")
