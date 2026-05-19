# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Metal fully fused 3DGS projection path."""

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection, _quat_scale_to_covar_preci

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    if expected is None:
        assert actual is None
        return
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


def _sample_inputs(batch_shape=(), c=2, n=8, width=640, height=480):
    torch.manual_seed(42)
    means = torch.randn(*batch_shape, n, 3, dtype=torch.float32) * 0.25
    means[..., 2] = torch.rand(*batch_shape, n, dtype=torch.float32) * 2.0 + 1.5

    quats = torch.randn(*batch_shape, n, 4, dtype=torch.float32)
    scales = torch.rand(*batch_shape, n, 3, dtype=torch.float32) * 0.3 + 0.2

    viewmats = torch.eye(4, dtype=torch.float32).expand(*batch_shape, c, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.1, 0.1, c, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.05, -0.05, c, dtype=torch.float32)

    Ks = torch.zeros(*batch_shape, c, 3, 3, dtype=torch.float32)
    Ks[..., 0, 0] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 1, 1] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0
    return means, quats, scales, viewmats, Ks


@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_forward_matches_reference_quat_scale(mps_device, camera_model):
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=3, n=16, width=width, height=height)

    covars_full, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=False)
    expected = _fully_fused_projection(
        means,
        covars_full,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model=camera_model,
    )
    actual = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model=camera_model,
    )

    expected_radii, expected_means2d, expected_depths, expected_conics, expected_comp = expected
    actual_radii, actual_means2d, actual_depths, actual_conics, actual_comp = actual
    valid = (expected_radii > 0).all(dim=-1) & (actual_radii.cpu() > 0).all(dim=-1)

    torch.testing.assert_close(actual_radii.cpu(), expected_radii.cpu(), atol=1, rtol=0)
    _assert_close(actual_means2d[valid.to(actual_means2d.device)], expected_means2d[valid], atol=2e-4, rtol=2e-4)
    _assert_close(actual_depths[valid.to(actual_depths.device)], expected_depths[valid], atol=2e-4, rtol=2e-4)
    _assert_close(actual_conics[valid.to(actual_conics.device)], expected_conics[valid], atol=2e-4, rtol=2e-4)
    _assert_close(actual_comp[valid.to(actual_comp.device)], expected_comp[valid], atol=2e-4, rtol=2e-4)


def test_forward_matches_reference_covars(mps_device):
    width, height = 800, 600
    means, quats, scales, viewmats, Ks = _sample_inputs(batch_shape=(2,), c=2, n=10, width=width, height=height)

    covars_full, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=False)
    covars_triu, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    expected = _fully_fused_projection(
        means,
        covars_full,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    actual = gm.fully_fused_projection(
        means.to(mps_device),
        covars_triu.to(mps_device),
        None,
        None,
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )

    for a, e in zip(actual[:4], expected[:4]):
        if a.dtype == torch.int32:
            torch.testing.assert_close(a.cpu(), e.cpu(), atol=1, rtol=0)
        else:
            valid = (expected[0] > 0).all(dim=-1) & (actual[0].cpu() > 0).all(dim=-1)
            _assert_close(a[valid.to(a.device)], e[valid], atol=2e-4, rtol=2e-4)
    assert actual[4] is None


def test_backward_matches_reference_quat_scale(mps_device):
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=2, n=6, width=width, height=height)

    means_ref = means.clone().requires_grad_(True)
    quats_ref = quats.clone().requires_grad_(True)
    scales_ref = scales.clone().requires_grad_(True)
    viewmats_ref = viewmats.clone().requires_grad_(True)

    covars_ref, _ = _quat_scale_to_covar_preci(quats_ref, scales_ref, compute_preci=False, triu=False)
    ref_radii, ref_means2d, ref_depths, ref_conics, ref_comp = _fully_fused_projection(
        means_ref,
        covars_ref,
        viewmats_ref,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
    )
    valid_ref = (ref_radii > 0).all(dim=-1)
    v_means2d = torch.randn_like(ref_means2d) * valid_ref[..., None]
    v_depths = torch.randn_like(ref_depths) * valid_ref
    v_conics = torch.randn_like(ref_conics) * valid_ref[..., None]
    v_comp = torch.randn_like(ref_comp) * valid_ref
    ref_loss = (
        (ref_means2d * v_means2d).sum()
        + (ref_depths * v_depths).sum()
        + (ref_conics * v_conics).sum()
        + (ref_comp * v_comp).sum()
    )
    ref_grads = torch.autograd.grad(ref_loss, (means_ref, quats_ref, scales_ref, viewmats_ref))

    means_mps = means.clone().to(mps_device).requires_grad_(True)
    quats_mps = quats.clone().to(mps_device).requires_grad_(True)
    scales_mps = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_mps = viewmats.clone().to(mps_device).requires_grad_(True)
    Ks_mps = Ks.to(mps_device)
    out = gm.fully_fused_projection(
        means_mps,
        None,
        quats_mps,
        scales_mps,
        viewmats_mps,
        Ks_mps,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
    )
    loss = (
        (out[1] * v_means2d.to(mps_device)).sum()
        + (out[2] * v_depths.to(mps_device)).sum()
        + (out[3] * v_conics.to(mps_device)).sum()
        + (out[4] * v_comp.to(mps_device)).sum()
    )
    grads = torch.autograd.grad(loss, (means_mps, quats_mps, scales_mps, viewmats_mps))

    for actual, expected in zip(grads, ref_grads):
        _assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_opacity_threshold_culls_gaussian(mps_device):
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=1, n=2, width=width, height=height)
    opacities = torch.tensor([1e-6, 0.9], dtype=torch.float32)

    radii, means2d, depths, conics, compensations = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.to(mps_device),
    )

    assert radii.shape == (1, 2, 2)
    assert torch.equal(radii[:, 0], torch.zeros_like(radii[:, 0]))
    assert (radii[:, 1] >= 0).all()
    assert means2d.shape[-1] == 2
    assert depths.shape[-1] == 2
    assert conics.shape[-1] == 3
    assert compensations is not None


# ---------------------------------------------------------------------------
# L2 — backward coverage: covars input, fisheye/ortho, compensation gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_backward_camera_models(mps_device, camera_model):
    """Backward matches reference for all three camera models (quat/scale path)."""
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=2, n=6, width=width, height=height)

    means_ref = means.clone().requires_grad_(True)
    quats_ref = quats.clone().requires_grad_(True)
    scales_ref = scales.clone().requires_grad_(True)

    covars_ref, _ = _quat_scale_to_covar_preci(quats_ref, scales_ref, compute_preci=False, triu=False)
    ref_radii, ref_means2d, ref_depths, ref_conics, _ = _fully_fused_projection(
        means_ref, covars_ref, viewmats, Ks, width, height,
        calc_compensations=False, camera_model=camera_model,
    )
    valid = (ref_radii > 0).all(dim=-1)
    v_means2d = torch.randn_like(ref_means2d) * valid[..., None]
    v_depths   = torch.randn_like(ref_depths)  * valid
    v_conics   = torch.randn_like(ref_conics)  * valid[..., None]
    ref_loss = (ref_means2d * v_means2d).sum() + (ref_depths * v_depths).sum() + (ref_conics * v_conics).sum()
    ref_grads = torch.autograd.grad(ref_loss, (means_ref, quats_ref, scales_ref))

    means_mps  = means.clone().to(mps_device).requires_grad_(True)
    quats_mps  = quats.clone().to(mps_device).requires_grad_(True)
    scales_mps = scales.clone().to(mps_device).requires_grad_(True)
    out = gm.fully_fused_projection(
        means_mps, None, quats_mps, scales_mps, viewmats.to(mps_device), Ks.to(mps_device),
        width, height, calc_compensations=False, camera_model=camera_model,
    )
    loss = (out[1] * v_means2d.to(mps_device)).sum() + (out[2] * v_depths.to(mps_device)).sum() + (out[3] * v_conics.to(mps_device)).sum()
    grads = torch.autograd.grad(loss, (means_mps, quats_mps, scales_mps))

    for actual, expected in zip(grads, ref_grads):
        _assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_backward_covars_input(mps_device):
    """Backward computes correct v_covars gradient when covars triu is the input."""
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=2, n=6, width=width, height=height)

    covars_full_cpu, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=False)
    covars_triu_cpu, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    means_ref   = means.clone().requires_grad_(True)
    covars_ref  = covars_full_cpu.clone().requires_grad_(True)
    ref_radii, ref_means2d, ref_depths, ref_conics, _ = _fully_fused_projection(
        means_ref, covars_ref, viewmats, Ks, width, height,
        calc_compensations=False, camera_model="pinhole",
    )
    valid = (ref_radii > 0).all(dim=-1)
    v_means2d = torch.randn_like(ref_means2d) * valid[..., None]
    v_depths   = torch.randn_like(ref_depths)  * valid
    v_conics   = torch.randn_like(ref_conics)  * valid[..., None]
    ref_loss = (ref_means2d * v_means2d).sum() + (ref_depths * v_depths).sum() + (ref_conics * v_conics).sum()
    ref_g_means, ref_g_covars_full = torch.autograd.grad(ref_loss, (means_ref, covars_ref))
    # Project full covar gradient down to triu for comparison
    ref_g_covars_triu = torch.stack([
        ref_g_covars_full[..., 0, 0],
        ref_g_covars_full[..., 0, 1] + ref_g_covars_full[..., 1, 0],
        ref_g_covars_full[..., 0, 2] + ref_g_covars_full[..., 2, 0],
        ref_g_covars_full[..., 1, 1],
        ref_g_covars_full[..., 1, 2] + ref_g_covars_full[..., 2, 1],
        ref_g_covars_full[..., 2, 2],
    ], dim=-1)

    means_mps  = means.clone().to(mps_device).requires_grad_(True)
    covars_mps = covars_triu_cpu.clone().to(mps_device).requires_grad_(True)
    out = gm.fully_fused_projection(
        means_mps, covars_mps, None, None, viewmats.to(mps_device), Ks.to(mps_device),
        width, height, calc_compensations=False, camera_model="pinhole",
    )
    loss = (out[1] * v_means2d.to(mps_device)).sum() + (out[2] * v_depths.to(mps_device)).sum() + (out[3] * v_conics.to(mps_device)).sum()
    g_means, g_covars = torch.autograd.grad(loss, (means_mps, covars_mps))

    _assert_close(g_means,  ref_g_means,       atol=3e-4, rtol=3e-4)
    _assert_close(g_covars, ref_g_covars_triu,  atol=3e-4, rtol=3e-4)


def test_backward_compensation_gradient(mps_device):
    """Gradient through compensation is correctly propagated."""
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(c=2, n=6, width=width, height=height)

    means_ref  = means.clone().requires_grad_(True)
    quats_ref  = quats.clone().requires_grad_(True)
    scales_ref = scales.clone().requires_grad_(True)

    covars_ref, _ = _quat_scale_to_covar_preci(quats_ref, scales_ref, compute_preci=False, triu=False)
    ref_radii, ref_means2d, ref_depths, ref_conics, ref_comp = _fully_fused_projection(
        means_ref, covars_ref, viewmats, Ks, width, height,
        calc_compensations=True, camera_model="pinhole",
    )
    assert ref_comp is not None
    valid = (ref_radii > 0).all(dim=-1)
    v_comp = torch.randn_like(ref_comp) * valid
    ref_grads = torch.autograd.grad((ref_comp * v_comp).sum(), (means_ref, quats_ref, scales_ref))

    means_mps  = means.clone().to(mps_device).requires_grad_(True)
    quats_mps  = quats.clone().to(mps_device).requires_grad_(True)
    scales_mps = scales.clone().to(mps_device).requires_grad_(True)
    out = gm.fully_fused_projection(
        means_mps, None, quats_mps, scales_mps, viewmats.to(mps_device), Ks.to(mps_device),
        width, height, calc_compensations=True, camera_model="pinhole",
    )
    assert out[4] is not None
    grads = torch.autograd.grad((out[4] * v_comp.to(mps_device)).sum(), (means_mps, quats_mps, scales_mps))

    for actual, expected in zip(grads, ref_grads):
        _assert_close(actual, expected, atol=3e-4, rtol=3e-4)


# ---------------------------------------------------------------------------
# L3 — batch-dims backward
# ---------------------------------------------------------------------------

def test_backward_batch_dims(mps_device):
    """Backward produces correct gradients for batched input [..., C, N, ...]."""
    width, height = 640, 480
    means, quats, scales, viewmats, Ks = _sample_inputs(
        batch_shape=(2,), c=2, n=4, width=width, height=height
    )

    means_ref   = means.clone().requires_grad_(True)
    quats_ref   = quats.clone().requires_grad_(True)
    scales_ref  = scales.clone().requires_grad_(True)
    viewmats_ref = viewmats.clone().requires_grad_(True)

    covars_ref, _ = _quat_scale_to_covar_preci(quats_ref, scales_ref, compute_preci=False, triu=False)
    ref_radii, ref_means2d, ref_depths, ref_conics, _ = _fully_fused_projection(
        means_ref, covars_ref, viewmats_ref, Ks, width, height,
        calc_compensations=False, camera_model="pinhole",
    )
    valid = (ref_radii > 0).all(dim=-1)
    v_means2d = torch.randn_like(ref_means2d) * valid[..., None]
    v_depths   = torch.randn_like(ref_depths)  * valid
    v_conics   = torch.randn_like(ref_conics)  * valid[..., None]
    ref_loss = (ref_means2d * v_means2d).sum() + (ref_depths * v_depths).sum() + (ref_conics * v_conics).sum()
    ref_grads = torch.autograd.grad(ref_loss, (means_ref, quats_ref, scales_ref, viewmats_ref))

    means_mps    = means.clone().to(mps_device).requires_grad_(True)
    quats_mps    = quats.clone().to(mps_device).requires_grad_(True)
    scales_mps   = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_mps = viewmats.clone().to(mps_device).requires_grad_(True)
    out = gm.fully_fused_projection(
        means_mps, None, quats_mps, scales_mps, viewmats_mps, Ks.to(mps_device),
        width, height, calc_compensations=False, camera_model="pinhole",
    )
    loss = (out[1] * v_means2d.to(mps_device)).sum() + (out[2] * v_depths.to(mps_device)).sum() + (out[3] * v_conics.to(mps_device)).sum()
    grads = torch.autograd.grad(loss, (means_mps, quats_mps, scales_mps, viewmats_mps))

    for actual, expected in zip(grads, ref_grads):
        assert actual.shape == expected.shape
        _assert_close(actual, expected, atol=3e-4, rtol=3e-4)
