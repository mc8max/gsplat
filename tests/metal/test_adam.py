# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.optimizers import SelectiveAdam

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _adam_reference(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    valid: torch.Tensor | None,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    param = param.clone()
    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()
    for row in range(param.shape[0]):
        if valid is not None and not bool(valid[row]):
            continue
        row_grad = grad[row]
        next_exp_avg = b1 * exp_avg[row] + (1.0 - b1) * row_grad
        next_exp_avg_sq = b2 * exp_avg_sq[row] + (1.0 - b2) * row_grad * row_grad
        param[row] += -lr * next_exp_avg / (torch.sqrt(next_exp_avg_sq) + eps)
        exp_avg[row] = next_exp_avg
        exp_avg_sq[row] = next_exp_avg_sq
    return param, exp_avg, exp_avg_sq


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_adam_matches_reference(mps_device, dtype):
    torch.manual_seed(0)
    param = torch.randn(7, 3, device=mps_device, dtype=dtype)
    grad = torch.randn_like(param)
    exp_avg = torch.randn_like(param)
    exp_avg_sq = torch.rand_like(param) + torch.tensor(0.1, device=mps_device, dtype=dtype)

    expected = _adam_reference(
        param.float().cpu(),
        grad.float().cpu(),
        exp_avg.float().cpu(),
        exp_avg_sq.float().cpu(),
        None,
        lr=1e-2,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )

    gm.adam(param, grad, exp_avg, exp_avg_sq, None, 1e-2, 0.9, 0.999, 1e-8)

    tol = dict(rtol=1e-6, atol=1e-6) if dtype == torch.float32 else dict(rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(param.float().cpu(), expected[0], **tol)
    torch.testing.assert_close(exp_avg.float().cpu(), expected[1], **tol)
    torch.testing.assert_close(exp_avg_sq.float().cpu(), expected[2], **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_adam_valid_mask_skips_rows(mps_device, dtype):
    torch.manual_seed(1)
    param = torch.randn(6, 4, device=mps_device, dtype=dtype)
    grad = torch.randn_like(param)
    exp_avg = torch.randn_like(param)
    exp_avg_sq = torch.rand_like(param) + torch.tensor(0.05, device=mps_device, dtype=dtype)
    valid = torch.tensor([True, False, True, False, True, False], device=mps_device)

    param_before = param.clone()
    exp_avg_before = exp_avg.clone()
    exp_avg_sq_before = exp_avg_sq.clone()

    expected = _adam_reference(
        param.float().cpu(),
        grad.float().cpu(),
        exp_avg.float().cpu(),
        exp_avg_sq.float().cpu(),
        valid.cpu(),
        lr=5e-3,
        b1=0.8,
        b2=0.95,
        eps=1e-6,
    )

    gm.adam(param, grad, exp_avg, exp_avg_sq, valid, 5e-3, 0.8, 0.95, 1e-6)

    tol = dict(rtol=1e-6, atol=1e-6) if dtype == torch.float32 else dict(rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(param.float().cpu(), expected[0], **tol)
    torch.testing.assert_close(exp_avg.float().cpu(), expected[1], **tol)
    torch.testing.assert_close(exp_avg_sq.float().cpu(), expected[2], **tol)
    torch.testing.assert_close(param[~valid].cpu(), param_before[~valid].cpu())
    torch.testing.assert_close(exp_avg[~valid].cpu(), exp_avg_before[~valid].cpu())
    torch.testing.assert_close(exp_avg_sq[~valid].cpu(), exp_avg_sq_before[~valid].cpu())


def test_adam_validates_shapes(mps_device):
    param = torch.randn(4, 3, device=mps_device)
    grad = torch.randn(4, 3, device=mps_device)
    exp_avg = torch.randn(4, 3, device=mps_device)
    exp_avg_sq = torch.randn(4, 3, device=mps_device)
    valid = torch.ones(5, dtype=torch.bool, device=mps_device)

    with pytest.raises(ValueError, match="valid first dimension must match param first dimension"):
        gm.adam(param, grad, exp_avg, exp_avg_sq, valid, 1e-2, 0.9, 0.999, 1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_adam_supports_noncontiguous_tensors(mps_device, dtype):
    torch.manual_seed(3)
    param = torch.randn(3, 5, device=mps_device, dtype=dtype).transpose(0, 1)
    grad = torch.randn_like(param).transpose(0, 1).transpose(0, 1)
    exp_avg = torch.randn_like(param)
    exp_avg_sq = torch.rand_like(param) + torch.tensor(0.2, device=mps_device, dtype=dtype)
    valid = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=mps_device)

    assert not param.is_contiguous()
    assert not exp_avg.is_contiguous()
    assert not exp_avg_sq.is_contiguous()

    expected = _adam_reference(
        param.float().cpu(),
        grad.float().cpu(),
        exp_avg.float().cpu(),
        exp_avg_sq.float().cpu(),
        valid.cpu(),
        lr=2e-3,
        b1=0.85,
        b2=0.97,
        eps=1e-6,
    )

    gm.adam(param, grad, exp_avg, exp_avg_sq, valid, 2e-3, 0.85, 0.97, 1e-6)

    tol = dict(rtol=1e-6, atol=1e-6) if dtype == torch.float32 else dict(rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(param.float().cpu(), expected[0], **tol)
    torch.testing.assert_close(exp_avg.float().cpu(), expected[1], **tol)
    torch.testing.assert_close(exp_avg_sq.float().cpu(), expected[2], **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_adam_matches_reference_over_multiple_steps(mps_device, dtype):
    torch.manual_seed(4)
    param = torch.randn(4, 3, device=mps_device, dtype=dtype)
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    param_ref = param.float().cpu()
    exp_avg_ref = exp_avg.float().cpu()
    exp_avg_sq_ref = exp_avg_sq.float().cpu()

    grads = [
        torch.randn_like(param),
        torch.randn_like(param),
        torch.randn_like(param),
    ]
    valids = [
        torch.tensor([True, True, False, True], dtype=torch.bool, device=mps_device),
        None,
        torch.tensor([False, True, True, True], dtype=torch.bool, device=mps_device),
    ]

    for grad, valid in zip(grads, valids):
        param_ref, exp_avg_ref, exp_avg_sq_ref = _adam_reference(
            param_ref,
            grad.float().cpu(),
            exp_avg_ref,
            exp_avg_sq_ref,
            None if valid is None else valid.cpu(),
            lr=1e-2,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        )
        gm.adam(param, grad, exp_avg, exp_avg_sq, valid, 1e-2, 0.9, 0.999, 1e-8)

    tol = dict(rtol=1e-6, atol=1e-6) if dtype == torch.float32 else dict(rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(param.float().cpu(), param_ref, **tol)
    torch.testing.assert_close(exp_avg.float().cpu(), exp_avg_ref, **tol)
    torch.testing.assert_close(exp_avg_sq.float().cpu(), exp_avg_sq_ref, **tol)


def test_selective_adam_routes_to_metal(mps_device):
    torch.manual_seed(2)
    lr = 5e-3
    param = torch.randn(5, 3, device=mps_device, requires_grad=True)
    visibility = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=mps_device)
    optimizer = SelectiveAdam([param], eps=1e-6, betas=(0.8, 0.95))
    optimizer.param_groups[0]["lr"] = lr

    loss = (param.square().sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0], device=mps_device)).sum()
    loss.backward()

    grad = param.grad.detach().clone()
    before = param.detach().clone()
    expected = _adam_reference(
        before.cpu(),
        grad.cpu(),
        torch.zeros_like(before).cpu(),
        torch.zeros_like(before).cpu(),
        visibility.cpu(),
        lr=lr,
        b1=0.8,
        b2=0.95,
        eps=1e-6,
    )

    optimizer.step(visibility=visibility)

    state = optimizer.state[param]
    assert state["exp_avg"].device.type == "mps"
    assert state["exp_avg_sq"].device.type == "mps"
    torch.testing.assert_close(param.detach().cpu(), expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state["exp_avg"].cpu(), expected[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state["exp_avg_sq"].cpu(), expected[2], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(param.detach()[~visibility].cpu(), before[~visibility].cpu())
