#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the public gsplat MPS render/backward path on a real COLMAP scene.

This script is intended for before/after performance checks while optimizing
the Metal backend. It exercises the same top-level rasterization path used by
the example trainer rather than benchmarking only native kernels in isolation.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from datasets.colmap import Dataset, Parser
from simple_trainer import create_splats_with_optimizers

from gsplat.rendering import rasterization


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _time_fn(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def _prepare_scene(
    *,
    data_dir: str,
    data_factor: int,
    device: torch.device,
    camera_index: int,
    max_gaussians: int | None,
) -> tuple[torch.nn.ParameterDict, dict, int, int]:
    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=True,
        test_every=8,
        load_exposure=False,
    )
    dataset = Dataset(parser, split="train", patch_size=None, load_depths=False)
    sample = dataset[camera_index % len(dataset)]

    splats, _ = create_splats_with_optimizers(
        parser=parser,
        init_type="sfm",
        init_opacity=0.1,
        init_scale=1.0,
        sh_degree=3,
        sparse_grad=False,
        visible_adam=False,
        batch_size=1,
        device=str(device),
        world_rank=0,
        world_size=1,
    )
    if max_gaussians is not None:
        for key, value in list(splats.items()):
            splats[key] = torch.nn.Parameter(value[:max_gaussians].detach().clone())
        splats = splats.to(device)

    image = sample["image"].to(device=device, dtype=torch.float32) / 255.0
    camtoworld = sample["camtoworld"].to(device)
    K = sample["K"].to(device)

    batch = {
        "image": image.unsqueeze(0),
        "viewmats": torch.linalg.inv(camtoworld).unsqueeze(0),
        "Ks": K.unsqueeze(0),
    }
    height, width = image.shape[:2]
    return splats, batch, width, height


def _render(
    splats: torch.nn.ParameterDict,
    batch: dict,
    *,
    width: int,
    height: int,
):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=batch["viewmats"],
        Ks=batch["Ks"],
        width=width,
        height=height,
        sh_degree=3,
        packed=False,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=False,
        with_eval3d=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark public gsplat MPS render/backward path")
    parser.add_argument("--data-dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for this benchmark")
    device = torch.device("mps")

    with torch.random.fork_rng():
        torch.manual_seed(42)
        splats, batch, width, height = _prepare_scene(
            data_dir=args.data_dir,
            data_factor=args.data_factor,
            device=device,
            camera_index=args.camera_index,
            max_gaussians=args.max_gaussians,
        )

    pixels = batch["image"]
    print(
        f"scene={args.data_dir} factor={args.data_factor} camera={args.camera_index} "
        f"gaussians={splats['means'].shape[0]} size={width}x{height}"
    )

    def forward_only() -> None:
        _render(splats, batch, width=width, height=height)

    def forward_backward() -> None:
        render_colors, render_alphas, _ = _render(splats, batch, width=width, height=height)
        loss = (render_colors[..., :3] - pixels).abs().mean() + 0.01 * render_alphas.mean()
        loss.backward()
        for param in splats.values():
            param.grad = None

    fwd_mean, fwd_std = _time_fn(forward_only, device=device, warmup=args.warmup, iters=args.iters)
    bwd_mean, bwd_std = _time_fn(forward_backward, device=device, warmup=args.warmup, iters=args.iters)

    print(f"forward           {fwd_mean:8.3f} ms  stdev {fwd_std:6.3f} ms")
    print(f"forward+backward  {bwd_mean:8.3f} ms  stdev {bwd_std:6.3f} ms")


if __name__ == "__main__":
    main()
