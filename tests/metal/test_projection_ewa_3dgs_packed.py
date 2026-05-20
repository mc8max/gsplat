# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _quat_scale_to_covar_preci

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_inputs(batch_shape=(), c=2, n=8, width=640, height=480):
    torch.manual_seed(123)
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

    opacities = torch.rand(*batch_shape, n, dtype=torch.float32) * 0.7 + 0.2
    return means, quats, scales, viewmats, Ks, opacities


def _pack_dense_reference(radii, means2d, depths, conics, compensations=None):
    batch_shape = tuple(radii.shape[:-3])
    c = radii.shape[-3]
    n = radii.shape[-2]
    B = 1
    for dim in batch_shape:
        B *= dim
    radii_flat = radii.reshape(B, c, n, 2)
    means2d_flat = means2d.reshape(B, c, n, 2)
    depths_flat = depths.reshape(B, c, n)
    conics_flat = conics.reshape(B, c, n, 3)
    comp_flat = compensations.reshape(B, c, n) if compensations is not None else None

    batch_ids = []
    camera_ids = []
    gaussian_ids = []
    out_radii = []
    out_means2d = []
    out_depths = []
    out_conics = []
    out_comp = []
    indptr = [0]
    for b in range(B):
        for cam in range(c):
            valid = (radii_flat[b, cam] > 0).all(dim=-1)
            idxs = torch.nonzero(valid, as_tuple=False).flatten()
            for gid in idxs.tolist():
                batch_ids.append(b)
                camera_ids.append(cam)
                gaussian_ids.append(gid)
                out_radii.append(radii_flat[b, cam, gid])
                out_means2d.append(means2d_flat[b, cam, gid])
                out_depths.append(depths_flat[b, cam, gid])
                out_conics.append(conics_flat[b, cam, gid])
                if comp_flat is not None:
                    out_comp.append(comp_flat[b, cam, gid])
            indptr.append(len(batch_ids))

    device = radii.device
    expected = {
        "batch_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "camera_ids": torch.tensor(camera_ids, dtype=torch.long, device=device),
        "gaussian_ids": torch.tensor(gaussian_ids, dtype=torch.long, device=device),
        "indptr": torch.tensor(indptr, dtype=torch.int32, device=device),
        "radii": torch.stack(out_radii, dim=0).to(torch.int32) if out_radii else torch.empty(0, 2, dtype=torch.int32, device=device),
        "means2d": torch.stack(out_means2d, dim=0) if out_means2d else torch.empty(0, 2, dtype=torch.float32, device=device),
        "depths": torch.stack(out_depths, dim=0) if out_depths else torch.empty(0, dtype=torch.float32, device=device),
        "conics": torch.stack(out_conics, dim=0) if out_conics else torch.empty(0, 3, dtype=torch.float32, device=device),
        "compensations": torch.stack(out_comp, dim=0) if out_comp else (torch.empty(0, dtype=torch.float32, device=device) if compensations is not None else None),
    }
    return expected


def _scatter_packed_grads_to_dense(
    batch_ids,
    camera_ids,
    gaussian_ids,
    dense_radii,
    v_means2d,
    v_depths,
    v_conics,
    v_compensations=None,
):
    if dense_radii.dim() == 3:
        grad_means2d = torch.zeros(*dense_radii.shape, dtype=v_means2d.dtype)
        grad_depths = torch.zeros(dense_radii.shape[:-1], dtype=v_depths.dtype)
        grad_conics = torch.zeros(*dense_radii.shape[:-1], 3, dtype=v_conics.dtype)
        grad_comp = (
            torch.zeros(dense_radii.shape[:-1], dtype=v_compensations.dtype)
            if v_compensations is not None
            else None
        )
        for idx in range(batch_ids.numel()):
            c = int(camera_ids[idx].item())
            g = int(gaussian_ids[idx].item())
            grad_means2d[c, g] = v_means2d[idx].cpu()
            grad_depths[c, g] = v_depths[idx].cpu()
            grad_conics[c, g] = v_conics[idx].cpu()
            if grad_comp is not None:
                grad_comp[c, g] = v_compensations[idx].cpu()
        return grad_means2d, grad_depths, grad_conics, grad_comp

    dense_shape = dense_radii.shape[:-1]
    grad_means2d = torch.zeros(*dense_shape, 2, dtype=v_means2d.dtype)
    grad_depths = torch.zeros(dense_shape, dtype=v_depths.dtype)
    grad_conics = torch.zeros(*dense_shape, 3, dtype=v_conics.dtype)
    grad_comp = (
        torch.zeros(dense_shape, dtype=v_compensations.dtype)
        if v_compensations is not None
        else None
    )
    for idx in range(batch_ids.numel()):
        b = int(batch_ids[idx].item())
        c = int(camera_ids[idx].item())
        g = int(gaussian_ids[idx].item())
        grad_means2d[b, c, g] = v_means2d[idx].cpu()
        grad_depths[b, c, g] = v_depths[idx].cpu()
        grad_conics[b, c, g] = v_conics[idx].cpu()
        if grad_comp is not None:
            grad_comp[b, c, g] = v_compensations[idx].cpu()
    return grad_means2d, grad_depths, grad_conics, grad_comp


def test_packed_forward_matches_dense_reference_quat_scale(mps_device):
    width, height = 640, 480
    means, quats, scales, viewmats, Ks, _ = _sample_inputs(batch_shape=(2,), c=3, n=12, width=width, height=height)

    dense = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    packed = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=False,
        camera_model="pinhole",
    )

    expected = _pack_dense_reference(*dense)
    batch_ids, camera_ids, gaussian_ids, indptr, radii, means2d, depths, conics, compensations = packed
    torch.testing.assert_close(batch_ids.cpu(), expected["batch_ids"].cpu())
    torch.testing.assert_close(camera_ids.cpu(), expected["camera_ids"].cpu())
    torch.testing.assert_close(gaussian_ids.cpu(), expected["gaussian_ids"].cpu())
    torch.testing.assert_close(indptr.cpu(), expected["indptr"].cpu())
    torch.testing.assert_close(radii.cpu(), expected["radii"].cpu(), atol=1, rtol=0)
    torch.testing.assert_close(means2d.cpu(), expected["means2d"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(depths.cpu(), expected["depths"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(conics.cpu(), expected["conics"].cpu(), atol=2e-4, rtol=2e-4)
    assert compensations is None


def test_packed_forward_matches_dense_reference_opacity_compensation(mps_device):
    width, height = 320, 240
    means, quats, scales, viewmats, Ks, opacities = _sample_inputs(c=2, n=10, width=width, height=height)

    dense = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        calc_compensations=True,
        camera_model="ortho",
        opacities=opacities.to(mps_device),
    )
    packed = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=True,
        camera_model="ortho",
        opacities=opacities.to(mps_device),
    )

    expected = _pack_dense_reference(*dense)
    _, _, _, indptr, radii, means2d, depths, conics, compensations = packed
    torch.testing.assert_close(indptr.cpu(), expected["indptr"].cpu())
    torch.testing.assert_close(radii.cpu(), expected["radii"].cpu(), atol=1, rtol=0)
    torch.testing.assert_close(means2d.cpu(), expected["means2d"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(depths.cpu(), expected["depths"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(conics.cpu(), expected["conics"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(compensations.cpu(), expected["compensations"].cpu(), atol=2e-4, rtol=2e-4)


def test_packed_forward_empty_outputs(mps_device):
    width, height = 64, 48
    means, quats, scales, viewmats, Ks, _ = _sample_inputs(c=2, n=6, width=width, height=height)
    means[..., 2] = 100.0

    batch_ids, camera_ids, gaussian_ids, indptr, radii, means2d, depths, conics, compensations = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=True,
        camera_model="fisheye",
        far_plane=1.0,
    )

    assert batch_ids.numel() == 0
    assert camera_ids.numel() == 0
    assert gaussian_ids.numel() == 0
    assert radii.shape == (0, 2)
    assert means2d.shape == (0, 2)
    assert depths.shape == (0,)
    assert conics.shape == (0, 3)
    assert compensations.shape == (0,)
    torch.testing.assert_close(indptr.cpu(), torch.zeros(viewmats.shape[-3] + 1, dtype=torch.int32))


def test_packed_backward_matches_dense_reference_quat_scale(mps_device):
    width, height = 320, 240
    means, quats, scales, viewmats, Ks, opacities = _sample_inputs(c=2, n=8, width=width, height=height)

    means_dense = means.clone().to(mps_device).requires_grad_(True)
    quats_dense = quats.clone().to(mps_device).requires_grad_(True)
    scales_dense = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_dense = viewmats.clone().to(mps_device).requires_grad_(True)
    dense = gm.fully_fused_projection(
        means_dense,
        None,
        quats_dense,
        scales_dense,
        viewmats_dense,
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.to(mps_device),
    )

    means_packed = means.clone().to(mps_device).requires_grad_(True)
    quats_packed = quats.clone().to(mps_device).requires_grad_(True)
    scales_packed = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_packed = viewmats.clone().to(mps_device).requires_grad_(True)
    packed = gm.fully_fused_projection(
        means_packed,
        None,
        quats_packed,
        scales_packed,
        viewmats_packed,
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.to(mps_device),
    )

    batch_ids, camera_ids, gaussian_ids, _, _, p_means2d, p_depths, p_conics, p_comp = packed
    torch.manual_seed(99)
    v_means2d = torch.randn_like(p_means2d)
    v_depths = torch.randn_like(p_depths)
    v_conics = torch.randn_like(p_conics)
    v_comp = torch.randn_like(p_comp)
    d_v_means2d, d_v_depths, d_v_conics, d_v_comp = _scatter_packed_grads_to_dense(
        batch_ids, camera_ids, gaussian_ids, dense[0].cpu(), v_means2d, v_depths, v_conics, v_comp
    )

    torch.autograd.backward(
        (dense[1], dense[2], dense[3], dense[4]),
        (
            d_v_means2d.to(mps_device),
            d_v_depths.to(mps_device),
            d_v_conics.to(mps_device),
            d_v_comp.to(mps_device),
        ),
    )
    torch.autograd.backward(
        (p_means2d, p_depths, p_conics, p_comp),
        (v_means2d, v_depths, v_conics, v_comp),
    )

    torch.testing.assert_close(means_packed.grad.cpu(), means_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(quats_packed.grad.cpu(), quats_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(scales_packed.grad.cpu(), scales_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(viewmats_packed.grad.cpu(), viewmats_dense.grad.cpu(), atol=3e-4, rtol=3e-4)


def test_packed_backward_sparse_grad_matches_dense_reference(mps_device):
    width, height = 256, 192
    means, quats, scales, viewmats, Ks, _ = _sample_inputs(c=2, n=7, width=width, height=height)

    means_dense = means.clone().to(mps_device).requires_grad_(True)
    quats_dense = quats.clone().to(mps_device).requires_grad_(True)
    scales_dense = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_dense = viewmats.clone().to(mps_device).requires_grad_(True)
    dense = gm.fully_fused_projection(
        means_dense,
        None,
        quats_dense,
        scales_dense,
        viewmats_dense,
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        calc_compensations=False,
        camera_model="ortho",
    )

    means_packed = means.clone().to(mps_device).requires_grad_(True)
    quats_packed = quats.clone().to(mps_device).requires_grad_(True)
    scales_packed = scales.clone().to(mps_device).requires_grad_(True)
    viewmats_packed = viewmats.clone().to(mps_device).requires_grad_(True)
    packed = gm.fully_fused_projection(
        means_packed,
        None,
        quats_packed,
        scales_packed,
        viewmats_packed,
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        sparse_grad=True,
        calc_compensations=False,
        camera_model="ortho",
    )

    batch_ids, camera_ids, gaussian_ids, _, _, p_means2d, p_depths, p_conics, _ = packed
    torch.manual_seed(1234)
    v_means2d = torch.randn_like(p_means2d)
    v_depths = torch.randn_like(p_depths)
    v_conics = torch.randn_like(p_conics)
    d_v_means2d, d_v_depths, d_v_conics, _ = _scatter_packed_grads_to_dense(
        batch_ids, camera_ids, gaussian_ids, dense[0].cpu(), v_means2d, v_depths, v_conics
    )

    torch.autograd.backward(
        (dense[1], dense[2], dense[3]),
        (
            d_v_means2d.to(mps_device),
            d_v_depths.to(mps_device),
            d_v_conics.to(mps_device),
        ),
    )
    torch.autograd.backward(
        (p_means2d, p_depths, p_conics),
        (v_means2d, v_depths, v_conics),
    )

    assert means_packed.grad.is_sparse
    assert quats_packed.grad.is_sparse
    assert scales_packed.grad.is_sparse
    torch.testing.assert_close(means_packed.grad.to_dense().cpu(), means_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(quats_packed.grad.to_dense().cpu(), quats_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(scales_packed.grad.to_dense().cpu(), scales_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(viewmats_packed.grad.cpu(), viewmats_dense.grad.cpu(), atol=3e-4, rtol=3e-4)


def test_packed_backward_matches_dense_reference_covars_no_viewmat_grad(mps_device):
    width, height = 300, 220
    means, quats, scales, viewmats, Ks, _ = _sample_inputs(c=3, n=9, width=width, height=height)
    covars, _ = _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False, triu=True)

    means_dense = means.clone().to(mps_device).requires_grad_(True)
    covars_dense = covars.clone().to(mps_device).requires_grad_(True)
    dense = gm.fully_fused_projection(
        means_dense,
        covars_dense,
        None,
        None,
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        calc_compensations=False,
        camera_model="fisheye",
    )

    means_packed = means.clone().to(mps_device).requires_grad_(True)
    covars_packed = covars.clone().to(mps_device).requires_grad_(True)
    packed = gm.fully_fused_projection(
        means_packed,
        covars_packed,
        None,
        None,
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=False,
        camera_model="fisheye",
    )

    batch_ids, camera_ids, gaussian_ids, _, _, p_means2d, p_depths, p_conics, _ = packed
    torch.manual_seed(314)
    v_means2d = torch.randn_like(p_means2d)
    v_depths = torch.randn_like(p_depths)
    v_conics = torch.randn_like(p_conics)
    d_v_means2d, d_v_depths, d_v_conics, _ = _scatter_packed_grads_to_dense(
        batch_ids, camera_ids, gaussian_ids, dense[0].cpu(), v_means2d, v_depths, v_conics
    )

    torch.autograd.backward(
        (dense[1], dense[2], dense[3]),
        (
            d_v_means2d.to(mps_device),
            d_v_depths.to(mps_device),
            d_v_conics.to(mps_device),
        ),
    )
    torch.autograd.backward(
        (p_means2d, p_depths, p_conics),
        (v_means2d, v_depths, v_conics),
    )

    torch.testing.assert_close(means_packed.grad.cpu(), means_dense.grad.cpu(), atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(covars_packed.grad.cpu(), covars_dense.grad.cpu(), atol=3e-4, rtol=3e-4)


def test_packed_forward_large_smoke_row_ranges_match_dense_reference(mps_device):
    width, height = 512, 384
    means, quats, scales, viewmats, Ks, opacities = _sample_inputs(batch_shape=(2,), c=4, n=48, width=width, height=height)

    dense = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=False,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.to(mps_device),
    )
    packed = gm.fully_fused_projection(
        means.to(mps_device),
        None,
        quats.to(mps_device),
        scales.to(mps_device),
        viewmats.to(mps_device),
        Ks.to(mps_device),
        width,
        height,
        packed=True,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.to(mps_device),
    )

    expected = _pack_dense_reference(*dense)
    batch_ids, camera_ids, gaussian_ids, indptr, radii, means2d, depths, conics, compensations = packed
    assert batch_ids.numel() == indptr[-1].item()
    assert camera_ids.numel() == batch_ids.numel()
    assert gaussian_ids.numel() == batch_ids.numel()
    torch.testing.assert_close(indptr.cpu(), expected["indptr"].cpu())
    torch.testing.assert_close(batch_ids.cpu(), expected["batch_ids"].cpu())
    torch.testing.assert_close(camera_ids.cpu(), expected["camera_ids"].cpu())
    torch.testing.assert_close(gaussian_ids.cpu(), expected["gaussian_ids"].cpu())
    torch.testing.assert_close(radii.cpu(), expected["radii"].cpu(), atol=1, rtol=0)
    torch.testing.assert_close(means2d.cpu(), expected["means2d"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(depths.cpu(), expected["depths"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(conics.cpu(), expected["conics"].cpu(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(compensations.cpu(), expected["compensations"].cpu(), atol=2e-4, rtol=2e-4)
