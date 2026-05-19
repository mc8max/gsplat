#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Metal fused 3DGS projection against the old wrapper-style fallback.

The fallback path here is a reconstructed approximation of the earlier
implementation:

1. native quat/scale -> covar op
2. pure PyTorch/MPS fully fused projection math

That gives a useful before/after comparison for the main change in Stage B:
native Metal backward kernel + temp-buffer reductions versus autograd through a
composed tensor graph.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection


@dataclass
class BenchCase:
    name: str
    batch: int
    cameras: int
    gaussians: int


def _sample_inputs(
    device: torch.device,
    batch: int,
    cameras: int,
    gaussians: int,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.randn(batch, gaussians, 3, device=device, dtype=torch.float32) * 0.25
    means[..., 2] = torch.rand(batch, gaussians, device=device, dtype=torch.float32) * 2.0 + 1.5

    quats = torch.randn(batch, gaussians, 4, device=device, dtype=torch.float32)
    scales = torch.rand(batch, gaussians, 3, device=device, dtype=torch.float32) * 0.3 + 0.2

    viewmats = torch.eye(4, device=device, dtype=torch.float32).expand(batch, cameras, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.1, 0.1, cameras, device=device, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.05, -0.05, cameras, device=device, dtype=torch.float32)

    Ks = torch.zeros(batch, cameras, 3, 3, device=device, dtype=torch.float32)
    Ks[..., 0, 0] = torch.rand(batch, cameras, device=device, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 1, 1] = torch.rand(batch, cameras, device=device, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0
    return means, quats, scales, viewmats, Ks


def _fallback_fully_fused_projection(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
):
    covars, _ = gm.quat_scale_to_covar_preci(
        quats,
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )
    return _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=True,
        camera_model="pinhole",
    )


def _make_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    radii, means2d, depths, conics, compensations = outputs
    valid = (radii > 0).all(dim=-1)
    loss = (
        (means2d * valid[..., None]).sum()
        + (depths * valid).sum()
        + (conics * valid[..., None]).sum()
    )
    if compensations is not None:
        loss = loss + (compensations * valid).sum()
    return loss


def _clone_requires_grad(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    viewmats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        means.detach().clone().requires_grad_(True),
        quats.detach().clone().requires_grad_(True),
        scales.detach().clone().requires_grad_(True),
        viewmats.detach().clone().requires_grad_(True),
    )


def _sync():
    torch.mps.synchronize()


def _time_fn(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def benchmark_case(
    case: BenchCase,
    width: int,
    height: int,
    warmup: int,
    iters: int,
    *,
    verbose: bool = True,
) -> dict[str, tuple[float, float]]:
    device = torch.device("mps")
    if verbose:
        print(f"\n[{case.name}] B={case.batch} C={case.cameras} N={case.gaussians}")

    with torch.random.fork_rng():
        torch.manual_seed(42)
        base = _sample_inputs(device, case.batch, case.cameras, case.gaussians, width, height)

    means, quats, scales, viewmats, Ks = base

    def native_forward():
        gm.fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=True,
            camera_model="pinhole",
        )

    def fallback_forward():
        _fallback_fully_fused_projection(means, quats, scales, viewmats, Ks, width, height)

    def native_backward():
        m, q, s, v = _clone_requires_grad(means, quats, scales, viewmats)
        out = gm.fully_fused_projection(
            m,
            None,
            q,
            s,
            v,
            Ks,
            width,
            height,
            calc_compensations=True,
            camera_model="pinhole",
        )
        _make_loss(out).backward()

    def fallback_backward():
        m, q, s, v = _clone_requires_grad(means, quats, scales, viewmats)
        out = _fallback_fully_fused_projection(m, q, s, v, Ks, width, height)
        _make_loss(out).backward()

    results = {
        "forward_native": _time_fn(native_forward, warmup, iters),
        "forward_fallback": _time_fn(fallback_forward, warmup, iters),
        "backward_native": _time_fn(native_backward, warmup, iters),
        "backward_fallback": _time_fn(fallback_backward, warmup, iters),
    }

    fwd_native = results["forward_native"][0]
    fwd_fallback = results["forward_fallback"][0]
    bwd_native = results["backward_native"][0]
    bwd_fallback = results["backward_fallback"][0]

    if verbose:
        print(
            f"forward  native   {fwd_native:8.3f} ms   "
            f"fallback {fwd_fallback:8.3f} ms   "
            f"speedup {fwd_fallback / fwd_native:6.2f}x"
        )
        print(
            f"backward native   {bwd_native:8.3f} ms   "
            f"fallback {bwd_fallback:8.3f} ms   "
            f"speedup {bwd_fallback / bwd_native:6.2f}x"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for this benchmark")

    cases = [
        BenchCase("medium", batch=1, cameras=4, gaussians=2048),
        BenchCase("large", batch=1, cameras=8, gaussians=8192),
    ]
    for case in cases:
        benchmark_case(case, args.width, args.height, args.warmup, args.iters)


if __name__ == "__main__":
    main()
