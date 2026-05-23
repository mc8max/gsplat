# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import relocate

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _relocation_reference(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    binoms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = opacities.shape[0]
    n_max = binoms.shape[0]
    new_opacities = torch.empty_like(opacities)
    new_scales = torch.empty_like(scales)
    for idx in range(n):
        n_idx = int(ratios[idx].item())
        opacity = float(opacities[idx].item())
        new_opacity = 1.0 - (1.0 - opacity) ** (1.0 / n_idx)
        new_opacities[idx] = new_opacity
        denom_sum = 0.0
        for i in range(1, n_idx + 1):
            for k in range(i):
                bin_coeff = float(binoms[i - 1, k].item())
                term = (((-1.0) ** k) / ((k + 1) ** 0.5)) * (new_opacity ** (k + 1))
                denom_sum += bin_coeff * term
        coeff = opacity / denom_sum
        new_scales[idx] = coeff * scales[idx]
    return new_opacities, new_scales


def test_relocation_matches_reference(mps_device):
    opacities = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32)
    scales = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, 0.75, 1.0], [1.25, 1.5, 2.0]],
        dtype=torch.float32,
    )
    ratios = torch.tensor([1, 2, 4], dtype=torch.int32)
    binoms = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [1.0, 3.0, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_opacities, expected_scales = _relocation_reference(opacities, scales, ratios, binoms)

    got_opacities, got_scales = gm.relocation(
        opacities.to(mps_device),
        scales.to(mps_device),
        ratios.to(mps_device),
        binoms.to(mps_device),
        4,
    )

    torch.testing.assert_close(got_opacities.cpu(), expected_opacities, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_scales.cpu(), expected_scales, rtol=1e-6, atol=1e-6)


def test_relocation_empty_inputs(mps_device):
    opacities = torch.empty((0,), dtype=torch.float32, device=mps_device)
    scales = torch.empty((0, 3), dtype=torch.float32, device=mps_device)
    ratios = torch.empty((0,), dtype=torch.int32, device=mps_device)
    binoms = torch.empty((0, 0), dtype=torch.float32, device=mps_device)

    new_opacities, new_scales = gm.relocation(opacities, scales, ratios, binoms, 0)

    assert tuple(new_opacities.shape) == (0,)
    assert tuple(new_scales.shape) == (0, 3)


def test_compute_relocation_routes_to_metal(mps_device):
    opacities = torch.tensor([0.25, 0.6], dtype=torch.float32, device=mps_device)
    scales = torch.tensor([[1.0, 1.5, 2.0], [0.5, 0.75, 1.0]], dtype=torch.float32, device=mps_device)
    ratios = torch.tensor([2.0, 3.0], dtype=torch.float32, device=mps_device)
    binoms = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device=mps_device,
    )

    expected_opacities, expected_scales = _relocation_reference(
        opacities.cpu(),
        scales.cpu(),
        ratios.int().cpu(),
        binoms.cpu(),
    )
    got_opacities, got_scales = compute_relocation(opacities, scales, ratios, binoms)

    assert got_opacities.device.type == "mps"
    assert got_scales.device.type == "mps"
    torch.testing.assert_close(got_opacities.cpu(), expected_opacities, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_scales.cpu(), expected_scales, rtol=1e-6, atol=1e-6)


def test_compute_relocation_supports_noncontiguous_inputs(mps_device):
    opacities = torch.tensor(
        [[0.2, 9.0], [0.4, 8.0], [0.8, 7.0]], dtype=torch.float32, device=mps_device
    )[:, 0]
    scales_base = torch.tensor(
        [
            [1.0, 2.0, 3.0, 9.0],
            [0.5, 0.75, 1.0, 8.0],
            [1.25, 1.5, 2.0, 7.0],
        ],
        dtype=torch.float32,
        device=mps_device,
    )
    scales = scales_base[:, :3].transpose(0, 1).transpose(0, 1)
    ratios = torch.tensor(
        [[1.0, 9.0], [2.0, 8.0], [4.0, 7.0]], dtype=torch.float32, device=mps_device
    )[:, 0]
    binoms = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [1.0, 3.0, 3.0, 1.0],
        ],
        dtype=torch.float32,
        device=mps_device,
    ).transpose(0, 1)

    assert not opacities.is_contiguous()
    assert not binoms.is_contiguous()

    expected_opacities, expected_scales = _relocation_reference(
        opacities.cpu(),
        scales.cpu(),
        ratios.int().cpu(),
        binoms.cpu(),
    )
    got_opacities, got_scales = compute_relocation(opacities, scales, ratios, binoms)

    torch.testing.assert_close(got_opacities.cpu(), expected_opacities, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_scales.cpu(), expected_scales, rtol=1e-6, atol=1e-6)


def test_strategy_relocate_smoke_on_mps(mps_device):
    binoms = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device=mps_device,
    )
    params = {
        "opacities": torch.nn.Parameter(
            torch.logit(torch.tensor([0.8, 0.5, 0.001], dtype=torch.float32, device=mps_device))
        ),
        "scales": torch.nn.Parameter(
            torch.log(torch.tensor(
                [
                    [1.0, 1.2, 1.4],
                    [0.8, 0.9, 1.0],
                    [0.7, 0.75, 0.8],
                ],
                dtype=torch.float32,
                device=mps_device,
            ))
        ),
    }
    optimizers = {
        name: torch.optim.Adam([param], lr=1e-3)
        for name, param in params.items()
    }
    dead_mask = torch.tensor([False, False, True], dtype=torch.bool, device=mps_device)

    before_opacities = params["opacities"].detach().clone()
    before_scales = params["scales"].detach().clone()

    relocate(
        params=params,
        optimizers=optimizers,
        state={},
        mask=dead_mask,
        binoms=binoms,
        min_opacity=0.005,
    )

    assert params["opacities"].device.type == "mps"
    assert params["scales"].device.type == "mps"
    assert tuple(params["opacities"].shape) == (3,)
    assert tuple(params["scales"].shape) == (3, 3)
    assert optimizers["opacities"].param_groups[0]["params"][0] is params["opacities"]
    assert optimizers["scales"].param_groups[0]["params"][0] is params["scales"]
    assert not torch.equal(params["opacities"].detach().cpu(), before_opacities.cpu())
    assert not torch.equal(params["scales"].detach().cpu(), before_scales.cpu())


def test_relocation_validates_shapes(mps_device):
    opacities = torch.ones((2,), dtype=torch.float32, device=mps_device)
    scales = torch.ones((2, 3), dtype=torch.float32, device=mps_device)
    ratios = torch.ones((2,), dtype=torch.int32, device=mps_device)
    binoms = torch.ones((3, 2), dtype=torch.float32, device=mps_device)

    with pytest.raises(ValueError, match="binoms must have shape"):
        gm.relocation(opacities, scales, ratios, binoms, 3)
