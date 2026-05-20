#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Metal fused 2DGS projection against the PyTorch reference path."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

import gsplat.metal as gm
from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs


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
    means = torch.randn(batch, gaussians, 3, device=device, dtype=torch.float32) * 0.2
    means[..., 2] = torch.rand(batch, gaussians, device=device, dtype=torch.float32) * 2.0 + 1.5
    quats = torch.randn(batch, gaussians, 4, device=device, dtype=torch.float32)
    scales = torch.rand(batch, gaussians, 3, device=device, dtype=torch.float32) * 0.3 + 0.2
    viewmats = torch.eye(4, device=device, dtype=torch.float32).expand(batch, cameras, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.08, 0.08, cameras, device=device, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.04, -0.04, cameras, device=device, dtype=torch.float32)
    Ks = torch.zeros(batch, cameras, 3, 3, device=device, dtype=torch.float32)
    Ks[..., 0, 0] = 220.0
    Ks[..., 1, 1] = 210.0
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0
    return means, quats, scales, viewmats, Ks


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


def _make_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    radii, means2d, depths, ray_transforms, normals = outputs
    valid = (radii > 0).all(dim=-1)
    return (
        (means2d * valid[..., None]).sum()
        + (depths * valid).sum()
        + (ray_transforms * valid[..., None, None]).sum()
        + (normals * valid[..., None]).sum()
    )


def _sync() -> None:
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


def benchmark_case(case: BenchCase, width: int, height: int, warmup: int, iters: int) -> None:
    device = torch.device("mps")
    print(f"\n[{case.name}] B={case.batch} C={case.cameras} N={case.gaussians}")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        means, quats, scales, viewmats, Ks = _sample_inputs(
            device, case.batch, case.cameras, case.gaussians, width, height
        )

    def native_forward():
        gm.fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width, height)

    def reference_forward():
        _fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width, height)

    def native_backward():
        m, q, s, v = _clone_requires_grad(means, quats, scales, viewmats)
        out = gm.fully_fused_projection_2dgs(m, q, s, v, Ks, width, height)
        _make_loss(out).backward()

    def reference_backward():
        m, q, s, v = _clone_requires_grad(means, quats, scales, viewmats)
        out = _fully_fused_projection_2dgs(m, q, s, v, Ks, width, height)
        _make_loss(out).backward()

    fwd_native = _time_fn(native_forward, warmup, iters)
    fwd_reference = _time_fn(reference_forward, warmup, iters)
    bwd_native = _time_fn(native_backward, warmup, iters)
    bwd_reference = _time_fn(reference_backward, warmup, iters)

    print(
        f"forward  native {fwd_native[0]:8.3f} ms   "
        f"reference {fwd_reference[0]:8.3f} ms   "
        f"speedup {fwd_reference[0] / fwd_native[0]:6.2f}x"
    )
    print(
        f"backward native {bwd_native[0]:8.3f} ms   "
        f"reference {bwd_reference[0]:8.3f} ms   "
        f"speedup {bwd_reference[0] / bwd_native[0]:6.2f}x"
    )


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
