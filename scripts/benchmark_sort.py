#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Performance comparison: CPU sort vs Metal GPU radix sort for isect_ids.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_sort.py
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_sort.py --n 10000000
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_sort.py --sizes 100000 1000000 10000000
"""

import argparse
import struct
import sys
import time

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync():
    torch.mps.synchronize()


def _make_isect_ids(n: int, I: int, tile_width: int, tile_height: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n random but valid isect_ids (non-negative int64) and flatten_ids."""
    n_tiles = tile_width * tile_height
    tile_n_bits = (n_tiles).bit_length()

    torch.manual_seed(42)
    image_ids = torch.randint(0, I, (n,), dtype=torch.int64)
    tile_ids  = torch.randint(0, n_tiles, (n,), dtype=torch.int64)
    # Depth values as float32, bit-cast to uint32 for encoding.
    depths_f  = torch.rand(n, dtype=torch.float32) * 10.0
    depth_bits = depths_f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    upper     = (image_ids << tile_n_bits | tile_ids) << 32
    isect_ids = upper | depth_bits
    flatten_ids = torch.arange(n, dtype=torch.int32)

    return isect_ids.to(device), flatten_ids.to(device)


def _load_metal():
    """Load and return the gsplat Metal extension module."""
    import gsplat.metal as gm
    if not gm.has_metal():
        print("ERROR: Metal extension not available.", file=sys.stderr)
        sys.exit(1)
    # Access the pybind11 module directly.
    from gsplat.metal._backend import _metal_C
    return _metal_C


def _bench(fn, warmup: int, iters: int, sync: bool = True) -> float:
    """Run fn() warmup+iters times; return mean time in ms over iters runs."""
    for _ in range(warmup):
        fn()
    if sync:
        _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(C, mps: torch.device) -> bool:
    """Verify that radix_sort and mps_lsd_sort (CPU) agree for a small input."""
    n = 1000
    ids, vals = _make_isect_ids(n, 4, 16, 12, mps)

    r_ids, r_vals = C.radix_sort(ids, vals)
    c_ids, c_vals = C.mps_lsd_sort(ids, vals)

    if not r_ids.cpu().equal(c_ids.cpu()):
        print("  [FAIL] radix_sort keys differ from CPU sort reference")
        return False
    if not r_vals.cpu().equal(c_vals.cpu()):
        print("  [FAIL] radix_sort values differ from CPU sort reference")
        return False

    # Verify sorted order is non-decreasing.
    s = r_ids.cpu()
    if not (s[1:] >= s[:-1]).all():
        print("  [FAIL] radix_sort output is not sorted")
        return False

    print("  [OK]   radix_sort output matches CPU reference and is sorted")
    return True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    sizes: list[int],
    I: int,
    tile_width: int,
    tile_height: int,
    warmup: int,
    iters: int,
    mps: torch.device,
    C,
) -> None:
    header = (
        f"\n{'n':>12}  {'cpu_ms':>10}  {'gpu_ms':>10}  "
        f"{'speedup':>9}  {'cpu_MB/s':>10}  {'gpu_MB/s':>10}"
    )
    print(header)
    print("-" * len(header.lstrip("\n")))

    for n in sizes:
        ids, vals = _make_isect_ids(n, I, tile_width, tile_height, mps)

        # --- CPU sort (mps_lsd_sort = CPU fallback, PCIe transfer in + out) ---
        cpu_ms = _bench(lambda: C.mps_lsd_sort(ids, vals), warmup, iters)

        # --- GPU radix sort (entirely on MPS, no PCIe) ---
        gpu_ms = _bench(lambda: C.radix_sort(ids, vals), warmup, iters)

        speedup  = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
        # Throughput: n × 12 bytes (int64 key + int32 val) in + out = × 2
        data_mb  = n * (8 + 4) * 2 / 1e6
        cpu_gbps = data_mb / cpu_ms
        gpu_gbps = data_mb / gpu_ms

        print(
            f"{n:>12,}  {cpu_ms:>10.3f}  {gpu_ms:>10.3f}  "
            f"{speedup:>8.2f}x  {cpu_gbps:>10.1f}  {gpu_gbps:>10.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sort benchmark: CPU vs GPU radix")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000],
        help="List of n_isects values to benchmark",
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Single n value (overrides --sizes)")
    parser.add_argument("--I", type=int, default=8,
                        help="Number of images (default: 8)")
    parser.add_argument("--tile-width",  type=int, default=32)
    parser.add_argument("--tile-height", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--iters",  type=int, default=20,
                        help="Timed iterations (default: 20)")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available on this machine.", file=sys.stderr)
        sys.exit(1)

    mps = torch.device("mps")
    C   = _load_metal()

    print("=" * 70)
    print(f"  gsplat Metal sort benchmark")
    print(f"  I={args.I}  tile={args.tile_width}×{args.tile_height}"
          f"  warmup={args.warmup}  iters={args.iters}")
    print("=" * 70)

    print("\nCorrectness check (n=1000):")
    ok = check_correctness(C, mps)
    if not ok:
        sys.exit(1)

    sizes = [args.n] if args.n is not None else args.sizes
    print(f"\nBenchmark  (times in ms, throughput in MB/s = n×12B×2 / ms):")

    run_benchmark(
        sizes       = sizes,
        I           = args.I,
        tile_width  = args.tile_width,
        tile_height = args.tile_height,
        warmup      = args.warmup,
        iters       = args.iters,
        mps         = mps,
        C           = C,
    )

    print()


if __name__ == "__main__":
    main()
