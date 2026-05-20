#!/usr/bin/env python3
"""Benchmark script for Metal projection_2dgs_packed_fwd/bwd."""

import argparse
import time

import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection_2dgs


def make_inputs(n, c, width, height, device="mps"):
    torch.manual_seed(42)
    means = torch.randn(1, n, 3, dtype=torch.float32, device=device) * 0.5
    means[..., 2] = torch.rand(1, n, dtype=torch.float32, device=device) * 3.0 + 2.0

    quats = torch.randn(1, n, 4, dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    scales = torch.rand(1, n, 3, dtype=torch.float32, device=device) * 0.5 + 0.1

    viewmats = torch.eye(4, dtype=torch.float32).expand(1, c, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.5, 0.5, c, dtype=torch.float32)

    Ks = torch.zeros(1, c, 3, 3, dtype=torch.float32, device=device)
    Ks[..., 0, 0] = float(width) * 0.6
    Ks[..., 1, 1] = float(height) * 0.6
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0

    return means, quats, scales, viewmats, Ks


def benchmark_packed_fwd(means, quats, scales, viewmats, Ks, width, height, iters=100):
    """Benchmark packed forward."""
    # Warmup
    for _ in range(10):
        gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, width, height, packed=True
        )
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, width, height, packed=True
        )
    torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms


def benchmark_dense_fwd(means, quats, scales, viewmats, Ks, width, height, iters=100):
    """Benchmark dense forward."""
    # Warmup
    for _ in range(10):
        _fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width, height)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width, height)
    torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms


def benchmark_packed_bwd(means, quats, scales, viewmats, Ks, width, height, iters=100):
    """Benchmark packed backward."""
    # Forward first
    result = gm.fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height, packed=True
    )
    (
        indptr, batch_ids, camera_ids, gaussian_ids,
        radii, means2d, depths, ray_transforms, normals,
    ) = result
    nnz = batch_ids.numel()

    v_means2d = torch.ones_like(means2d)
    v_depths = torch.ones_like(depths)
    v_ray_transforms = torch.ones_like(ray_transforms)
    v_normals = torch.ones_like(normals)

    # Warmup
    for _ in range(10):
        torch.autograd.grad(
            (means2d, depths, ray_transforms, normals),
            (means, quats, scales, viewmats),
            (v_means2d, v_depths, v_ray_transforms, v_normals),
            allow_unused=True,
        )
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.autograd.grad(
            (means2d, depths, ray_transforms, normals),
            (means, quats, scales, viewmats),
            (v_means2d, v_depths, v_ray_transforms, v_normals),
            allow_unused=True,
        )
    torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark Metal 2DGS packed projection")
    parser.add_argument("--n", type=int, default=1000, help="Number of Gaussians")
    parser.add_argument("--c", type=int, default=4, help="Number of cameras")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--device", type=str, default="mps", help="Device")
    args = parser.parse_args()

    if args.device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"

    means, quats, scales, viewmats, Ks = make_inputs(args.n, args.c, args.width, args.height, args.device)

    print(f"Configuration: N={args.n}, C={args.c}, {args.width}x{args.height}, device={args.device}")
    print(f"Iterations: {args.iters}")
    print()

    # Forward benchmarks
    dense_ms = benchmark_dense_fwd(means, quats, scales, viewmats, Ks, args.width, args.height, args.iters)
    packed_ms = benchmark_packed_fwd(means, quats, scales, viewmats, Ks, args.width, args.height, args.iters)

    print(f"Dense forward:  {dense_ms:.3f} ms")
    print(f"Packed forward: {packed_ms:.3f} ms")
    print(f"Speedup:        {dense_ms / packed_ms:.2f}x")
    print()

    # Backward benchmark
    bwd_ms = benchmark_packed_bwd(means, quats, scales, viewmats, Ks, args.width, args.height, args.iters)
    print(f"Packed backward: {bwd_ms:.3f} ms")


if __name__ == "__main__":
    main()
