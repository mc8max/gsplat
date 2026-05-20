#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native Metal 2DGS rasterization."""

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
    packed: bool


def _sample_world_inputs(*, images: int, gaussians: int, channels: int):
    means = torch.randn(gaussians, 3, dtype=torch.float32) * 0.2
    means[:, 2] = torch.rand(gaussians, dtype=torch.float32) * 1.5 + 1.5
    quats = torch.randn(gaussians, 4, dtype=torch.float32)
    scales = torch.rand(gaussians, 3, dtype=torch.float32) * 0.2 + 0.15
    viewmats = torch.eye(4, dtype=torch.float32).expand(images, 4, 4).clone()
    viewmats[:, 0, 3] = torch.linspace(-0.05, 0.05, images, dtype=torch.float32)
    viewmats[:, 1, 3] = torch.linspace(0.04, -0.04, images, dtype=torch.float32)
    features = torch.rand(images, gaussians, channels, dtype=torch.float32)
    opacities = torch.rand(images, gaussians, dtype=torch.float32) * 0.6 + 0.3
    return means, quats, scales, viewmats, features, opacities


def _prepare_case(
    device: torch.device,
    *,
    images: int,
    gaussians: int,
    channels: int,
    width: int,
    height: int,
    tile_size: int,
    packed: bool,
):
    means, quats, scales, viewmats, features, opacities = _sample_world_inputs(
        images=images, gaussians=gaussians, channels=channels
    )
    Ks = torch.zeros(images, 3, 3, dtype=torch.float32)
    Ks[:, 0, 0] = 220.0
    Ks[:, 1, 1] = 210.0
    Ks[:, 0, 2] = width * 0.5
    Ks[:, 1, 2] = height * 0.5
    Ks[:, 2, 2] = 1.0

    radii, means2d, depths, ray_transforms, normals = gm.fully_fused_projection_2dgs(
        means.to(device),
        quats.to(device),
        scales.to(device),
        viewmats.to(device),
        Ks.to(device),
        width,
        height,
    )
    colors = torch.cat([features.to(device), depths[..., None]], dim=-1)
    backgrounds = torch.zeros(images, colors.shape[-1], device=device, dtype=torch.float32)
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    if not packed:
        _, isect_ids, flatten_ids = gm.intersect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height
        )
        isect_offsets = gm.intersect_offset_encode(isect_ids, images, tile_width, tile_height)
        densify = torch.zeros_like(means2d)
        return means2d, ray_transforms, colors, opacities.to(device), normals, densify, backgrounds, isect_offsets, flatten_ids

    image_ids = torch.arange(images, device=device, dtype=torch.int64).repeat_interleave(gaussians)
    gaussian_ids = torch.arange(gaussians, device=device, dtype=torch.int64).repeat(images)
    means2d_p = means2d.reshape(-1, 2)
    radii_p = radii.reshape(-1, 2)
    depths_p = depths.reshape(-1)
    ray_p = ray_transforms.reshape(-1, 3, 3)
    normals_p = normals.reshape(-1, 3)
    colors_p = colors.reshape(-1, colors.shape[-1])
    opac_p = opacities.to(device).reshape(-1)
    _, isect_ids, flatten_ids = gm.intersect_tiles(
        means2d_p,
        radii_p,
        depths_p,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=images,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = gm.intersect_offset_encode(isect_ids, images, tile_width, tile_height)
    densify = torch.zeros_like(means2d_p)
    return means2d_p, ray_p, colors_p, opac_p, normals_p, densify, backgrounds, isect_offsets, flatten_ids


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


def benchmark_case(case: BenchCase, *, width: int, height: int, tile_size: int, warmup: int, iters: int):
    device = torch.device("mps")
    print(
        f"\n[{case.name}] I={case.images} N={case.gaussians} "
        f"C={case.channels} packed={case.packed} size={width}x{height}"
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        base = _prepare_case(
            device,
            images=case.images,
            gaussians=case.gaussians,
            channels=case.channels,
            width=width,
            height=height,
            tile_size=tile_size,
            packed=case.packed,
        )
    means2d, ray_transforms, colors, opacities, normals, densify, backgrounds, isect_offsets, flatten_ids = base

    def forward_only() -> None:
        gm.rasterize_to_pixels_2dgs(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=case.packed,
        )

    def backward_full() -> None:
        m = means2d.detach().clone().requires_grad_(True)
        r = ray_transforms.detach().clone().requires_grad_(True)
        c = colors.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        n = normals.detach().clone().requires_grad_(True)
        d = densify.detach().clone().requires_grad_(True)
        outputs = gm.rasterize_to_pixels_2dgs(
            m,
            r,
            c,
            o,
            n,
            d,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=case.packed,
            absgrad=True,
        )
        sum(out.sum() for out in outputs).backward()

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
        BenchCase("dense-rgbd", images=2, gaussians=1024, channels=3, packed=False),
        BenchCase("dense-padded", images=2, gaussians=1024, channels=15, packed=False),
        BenchCase("packed-rgbd", images=2, gaussians=1024, channels=3, packed=True),
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
