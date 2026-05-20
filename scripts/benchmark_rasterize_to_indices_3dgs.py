#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native Metal rasterize_to_indices_3dgs on realistic tile batches."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import torch

import gsplat.metal as gm


@dataclass
class BenchCase:
    name: str
    images: int
    gaussians: int
    width: int
    height: int
    tile_size: int


def _sample_world_inputs(
    device: torch.device,
    *,
    images: int,
    gaussians: int,
    width: int,
    height: int,
):
    means = torch.randn(gaussians, 3, device=device, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(gaussians, device=device, dtype=torch.float32) * 1.3 + 1.6
    quats = torch.randn(gaussians, 4, device=device, dtype=torch.float32)
    scales = torch.rand(gaussians, 3, device=device, dtype=torch.float32) * 0.2 + 0.14
    viewmats = torch.eye(4, device=device, dtype=torch.float32).expand(images, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, images, device=device, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.03, -0.03, images, device=device, dtype=torch.float32)
    Ks = torch.zeros(images, 3, 3, device=device, dtype=torch.float32)
    Ks[:, 0, 0] = 215.0
    Ks[:, 1, 1] = 205.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0
    opacities = torch.rand(gaussians, device=device, dtype=torch.float32) * 0.5 + 0.35
    return means, quats, scales, viewmats, Ks, opacities


def _make_inputs(case: BenchCase, device: torch.device):
    means, quats, scales, viewmats, Ks, opacities = _sample_world_inputs(
        device,
        images=case.images,
        gaussians=case.gaussians,
        width=case.width,
        height=case.height,
    )
    tile_width = math.ceil(case.width / case.tile_size)
    tile_height = math.ceil(case.height / case.tile_size)
    radii, means2d, depths, conics, _ = gm.fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        case.width,
        case.height,
        calc_compensations=False,
        camera_model="pinhole",
    )
    per_view_opacities = torch.broadcast_to(opacities[None, :], depths.shape)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        case.tile_size,
        tile_width,
        tile_height,
    )
    isect_offsets = gm.intersect_offset_encode(
        isect_ids,
        case.images,
        tile_width,
        tile_height,
    )
    transmittances = torch.ones(
        case.images, case.height, case.width, device=device, dtype=torch.float32
    )
    max_isects_per_tile = 0
    if isect_offsets.numel() > 0:
        offsets_flat = torch.cat(
            [isect_offsets.reshape(-1), torch.tensor([flatten_ids.numel()], device=device, dtype=torch.int32)]
        )
        max_isects_per_tile = int((offsets_flat[1:] - offsets_flat[:-1]).max().item())
    block_size = case.tile_size * case.tile_size
    num_batches = (max_isects_per_tile + block_size - 1) // block_size if block_size > 0 else 0
    return (
        transmittances,
        means2d,
        conics,
        per_view_opacities,
        isect_offsets,
        flatten_ids,
        num_batches,
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


def benchmark_case(case: BenchCase, *, warmup: int, iters: int) -> None:
    device = torch.device("mps")
    print(
        f"\n[{case.name}] I={case.images} N={case.gaussians} "
        f"size={case.width}x{case.height} tile={case.tile_size}"
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        (
            transmittances,
            means2d,
            conics,
            opacities,
            isect_offsets,
            flatten_ids,
            num_batches,
        ) = _make_inputs(case, device)

    def full_range() -> None:
        gm.rasterize_to_indices_in_range(
            0,
            1_000_000,
            transmittances,
            means2d,
            conics,
            opacities,
            case.width,
            case.height,
            case.tile_size,
            isect_offsets,
            flatten_ids,
        )

    def first_batch() -> None:
        gm.rasterize_to_indices_in_range(
            0,
            1,
            transmittances,
            means2d,
            conics,
            opacities,
            case.width,
            case.height,
            case.tile_size,
            isect_offsets,
            flatten_ids,
        )

    full_mean, full_std = _time_fn(full_range, warmup, iters)
    print(
        f"full-range   {full_mean:8.3f} ms  stdev {full_std:6.3f} ms  "
        f"batches/tile max {num_batches}"
    )
    if num_batches > 0:
        batch_mean, batch_std = _time_fn(first_batch, warmup, iters)
        print(f"first-batch  {batch_mean:8.3f} ms  stdev {batch_std:6.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for this benchmark")

    cases = [
        BenchCase("medium", images=2, gaussians=2048, width=320, height=240, tile_size=8),
        BenchCase("large", images=4, gaussians=8192, width=640, height=480, tile_size=16),
    ]
    for case in cases:
        benchmark_case(case, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
