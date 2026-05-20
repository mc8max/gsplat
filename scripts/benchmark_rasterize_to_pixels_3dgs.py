#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native Metal 3DGS rasterization for common channel buckets."""

from __future__ import annotations

import argparse
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
    channels: int


def _sample_inputs(
    device: torch.device,
    *,
    images: int,
    gaussians: int,
    channels: int,
    width: int,
    height: int,
    tile_size: int,
):
    means2d = torch.rand(images, gaussians, 2, device=device, dtype=torch.float32)
    means2d[..., 0] *= width - 1
    means2d[..., 1] *= height - 1

    radii = torch.randint(1, 5, (images, gaussians, 2), device=device, dtype=torch.int32)
    depths = torch.rand(images, gaussians, device=device, dtype=torch.float32)
    conics = torch.zeros(images, gaussians, 3, device=device, dtype=torch.float32)
    conics[..., 0] = torch.rand(images, gaussians, device=device, dtype=torch.float32) * 0.25 + 0.15
    conics[..., 1] = (torch.rand(images, gaussians, device=device, dtype=torch.float32) - 0.5) * 0.05
    conics[..., 2] = torch.rand(images, gaussians, device=device, dtype=torch.float32) * 0.25 + 0.15
    colors = torch.rand(images, gaussians, channels, device=device, dtype=torch.float32)
    opacities = torch.rand(images, gaussians, device=device, dtype=torch.float32) * 0.7 + 0.2
    backgrounds = torch.rand(images, channels, device=device, dtype=torch.float32)

    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    isect_offsets = gm.intersect_offset_encode(isect_ids, images, tile_width, tile_height)
    return means2d, conics, colors, opacities, backgrounds, isect_offsets, flatten_ids


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


def benchmark_case(
    case: BenchCase,
    *,
    width: int,
    height: int,
    tile_size: int,
    warmup: int,
    iters: int,
) -> None:
    device = torch.device("mps")
    print(
        f"\n[{case.name}] I={case.images} N={case.gaussians} "
        f"C={case.channels} size={width}x{height}"
    )

    with torch.random.fork_rng():
        torch.manual_seed(42)
        base = _sample_inputs(
            device,
            images=case.images,
            gaussians=case.gaussians,
            channels=case.channels,
            width=width,
            height=height,
            tile_size=tile_size,
        )

    means2d, conics, colors, opacities, backgrounds, isect_offsets, flatten_ids = base

    def forward_only() -> None:
        gm.rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )

    def backward_full() -> None:
        m = means2d.detach().clone().requires_grad_(True)
        c = conics.detach().clone().requires_grad_(True)
        rgb = colors.detach().clone().requires_grad_(True)
        a = opacities.detach().clone().requires_grad_(True)
        render_colors, render_alphas = gm.rasterize_to_pixels(
            m,
            c,
            rgb,
            a,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            absgrad=True,
        )
        (render_colors.sum() + render_alphas.sum()).backward()

    fwd_mean, fwd_std = _time_fn(forward_only, warmup, iters)
    bwd_mean, bwd_std = _time_fn(backward_full, warmup, iters)
    print(f"forward  {fwd_mean:8.3f} ms  stdev {fwd_std:6.3f} ms")
    print(f"backward {bwd_mean:8.3f} ms  stdev {bwd_std:6.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for this benchmark")

    cases = [
        BenchCase("rgb", images=2, gaussians=1024, channels=3),
        BenchCase("cached-17", images=2, gaussians=1024, channels=17),
        BenchCase("uncached-33", images=2, gaussians=1024, channels=33),
    ]
    for case in cases:
        benchmark_case(
            case,
            width=args.width,
            height=args.height,
            tile_size=args.tile_size,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
