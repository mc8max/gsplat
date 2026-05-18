# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mps_lsd_sort (v1, CPU fallback) and radix_sort (v2, GPU kernel).

Both sort functions are exposed via pybind11 on the gsplat Metal extension module
and are accessed through gsplat.metal._backend._metal_C.

Test structure:
  - Basic correctness    : empty, single, small known-answer, random large
  - Stability            : equal keys preserve original flatten_id order
  - Depth-bit precision  : large int64 values (real float32 bit patterns) are exact
  - Cross-pass stability : each of the 8 radix passes is independently stable
  - Agreement            : v1 and v2 produce identical output on the same input
"""

import struct

import pytest
import torch

import gsplat.metal as gm

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture(scope="session", autouse=True)
def _load_metal():
    """Ensure the Metal extension is loaded before any test accesses C."""
    assert gm.has_metal(), "Metal extension failed to load"


def _get_C():
    """Return the loaded pybind11 module (must be called after has_metal())."""
    from gsplat.metal._backend import _metal_C
    assert _metal_C is not None, "Metal extension not loaded"
    return _metal_C


C = None  # populated in session fixture below

MPS = torch.device("mps")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack(image_id: int, tile_id: int, depth: float, tile_n_bits: int = 5) -> int:
    """Pack one (image_id, tile_id, depth) triple into an isect_id int64."""
    depth_bits = struct.unpack(">I", struct.pack(">f", depth))[0]
    upper = (image_id << tile_n_bits | tile_id) << 32
    return upper | depth_bits


def _ids(*triples, tile_n_bits: int = 5) -> torch.Tensor:
    """Build an int64 MPS tensor from a list of (image_id, tile_id, depth) triples."""
    return torch.tensor(
        [_pack(i, t, d, tile_n_bits) for i, t, d in triples],
        dtype=torch.int64,
        device=MPS,
    )


def _vals(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32, device=MPS)


def _both(sort_fn, ids, vals):
    """Call sort_fn and return (sorted_ids_cpu, sorted_vals_cpu)."""
    s_ids, s_vals = sort_fn(ids, vals)
    return s_ids.cpu(), s_vals.cpu()


def _assert_sorted(ids_cpu: torch.Tensor) -> None:
    if ids_cpu.numel() > 1:
        assert (ids_cpu[1:] >= ids_cpu[:-1]).all().item(), \
            "keys are not sorted in non-decreasing order"


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Parametrize over both implementations
# ---------------------------------------------------------------------------

@pytest.fixture(params=["mps_lsd_sort", "radix_sort"])
def sort_fn(request):
    return getattr(_get_C(), request.param)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_empty(sort_fn):
    ids  = torch.empty(0, dtype=torch.int64,  device=MPS)
    vals = torch.empty(0, dtype=torch.int32, device=MPS)
    s_ids, s_vals = sort_fn(ids, vals)
    assert s_ids.shape  == (0,)
    assert s_ids.dtype  == torch.int64
    assert s_vals.shape == (0,)
    assert s_vals.dtype == torch.int32


def test_single(sort_fn):
    ids  = _ids((0, 3, 0.5))
    vals = _vals(1)
    s_ids, s_vals = _both(sort_fn, ids, vals)
    _assert_exact(s_ids, ids.cpu())
    _assert_exact(s_vals, vals.cpu())


def test_small_known_answer(sort_fn):
    """Three records with a known sorted order."""
    # image=0, tile=3, depth=0.3  >  image=0, tile=1, depth=0.1  (same image)
    # image=1, tile=0, depth=0.2  — different image, comes last
    ids = _ids((0, 3, 0.3), (0, 1, 0.1), (1, 0, 0.2))
    vals = _vals(3)
    s_ids, s_vals = _both(sort_fn, ids, vals)

    cpu_ref = torch.sort(ids.cpu())[0]
    _assert_exact(s_ids, cpu_ref)
    _assert_sorted(s_ids)


def test_random_large(sort_fn):
    """10k random records match CPU torch.sort reference."""
    torch.manual_seed(7)
    tile_n_bits = 5
    n = 10_000
    raw_ids = (
        torch.randint(0, 8, (n,), dtype=torch.int64) << (tile_n_bits + 32)
        | torch.randint(0, 32, (n,), dtype=torch.int64) << 32
        | (torch.rand(n).view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    ).to(MPS)
    vals = _vals(n)

    s_ids, _ = _both(sort_fn, raw_ids, vals)
    cpu_ref  = torch.sort(raw_ids.cpu())[0]
    _assert_exact(s_ids, cpu_ref)


def test_already_sorted(sort_fn):
    ids  = _ids((0, 0, 0.1), (0, 1, 0.1), (0, 2, 0.1), (1, 0, 0.2))
    vals = _vals(4)
    s_ids, s_vals = _both(sort_fn, ids, vals)
    _assert_exact(s_ids,  ids.cpu())
    _assert_exact(s_vals, vals.cpu())


def test_reverse_sorted(sort_fn):
    ids = _ids((1, 3, 0.9), (1, 2, 0.5), (0, 3, 0.5), (0, 0, 0.1))
    vals = _vals(4)
    s_ids, s_vals = _both(sort_fn, ids, vals)
    cpu_ref = torch.sort(ids.cpu())[0]
    _assert_exact(s_ids, cpu_ref)
    _assert_sorted(s_ids)


def test_all_same_keys(sort_fn):
    """All keys identical — output is a permutation of the input."""
    ids  = torch.full((8,), _pack(0, 5, 0.25), dtype=torch.int64, device=MPS)
    vals = _vals(8)
    s_ids, s_vals = _both(sort_fn, ids, vals)
    _assert_sorted(s_ids)
    # All keys equal, so vals must be a permutation of [0..7]
    assert sorted(s_vals.tolist()) == list(range(8))


# ---------------------------------------------------------------------------
# Stability: equal keys must preserve original flatten_id order
# ---------------------------------------------------------------------------

def test_stability_equal_keys(sort_fn):
    """Three records with equal keys must come out in emission (val) order."""
    key = _pack(0, 4, 0.3)
    ids  = torch.tensor([key, key, key], dtype=torch.int64,  device=MPS)
    vals = torch.tensor([7, 2, 5],       dtype=torch.int32, device=MPS)
    _, s_vals = _both(sort_fn, ids, vals)
    # Stable sort must preserve [7, 2, 5] for equal keys.
    _assert_exact(s_vals, torch.tensor([7, 2, 5], dtype=torch.int32))


def test_stability_mixed_keys(sort_fn):
    """Equal keys in a mixed sequence must preserve relative order."""
    k0 = _pack(0, 0, 0.1)
    k1 = _pack(0, 1, 0.2)
    ids = torch.tensor([k1, k0, k1, k0, k1], dtype=torch.int64, device=MPS)
    vals = torch.arange(5, dtype=torch.int32, device=MPS)
    _, s_vals = _both(sort_fn, ids, vals)
    # k0 records: original positions 1, 3  → vals 1, 3 (in order)
    # k1 records: original positions 0, 2, 4 → vals 0, 2, 4 (in order)
    expected = torch.tensor([1, 3, 0, 2, 4], dtype=torch.int32)
    _assert_exact(s_vals, expected)


# ---------------------------------------------------------------------------
# Depth-bit precision: large int64 values must be sorted exactly
# ---------------------------------------------------------------------------

def test_depth_bits_precision(sort_fn):
    """Adjacent float32 bit patterns must be sorted correctly."""
    d0 = struct.unpack(">I", struct.pack(">f", 0.1))[0]  # 0x3DCCCCCD
    d1 = struct.unpack(">I", struct.pack(">f", 0.2))[0]  # 0x3E4CCCCD

    # Same tile, different depths (one ULP apart for each pair)
    k_base = (0 << 5 | 3) << 32  # image=0, tile=3
    ids = torch.tensor(
        [k_base | d1, k_base | (d0 + 1), k_base | d0, k_base | (d1 - 1)],
        dtype=torch.int64, device=MPS,
    )
    vals = _vals(4)
    s_ids, s_vals = _both(sort_fn, ids, vals)

    # CPU reference is authoritative
    cpu_ref = torch.sort(ids.cpu())[0]
    _assert_exact(s_ids, cpu_ref)


# ---------------------------------------------------------------------------
# Cross-pass stability (radix_sort specific)
# ---------------------------------------------------------------------------
# These tests verify that each 8-bit radix pass is independently stable.
# An LSD radix sort is correct ONLY if every pass preserves the relative
# order of records with equal digit values.  These tests would FAIL for an
# unstable scatter even if the final key order happened to be correct.

def _chunk(ids_cpu: torch.Tensor, shift: int) -> torch.Tensor:
    """Extract one 8-bit chunk from the keys."""
    return ((ids_cpu >> shift) & 0xFF).to(torch.int32)


@pytest.mark.parametrize("pass_idx", [0, 1, 2, 3, 4, 5, 6, 7])
def test_stable_within_pass(pass_idx):
    """All records share the same bits for passes 0..pass_idx-1, differing only
    in the pass_idx-th 8-bit chunk.  After a stable sort by that chunk, records
    with equal chunk values must appear in original emission order."""
    shift = pass_idx * 8
    n = 64

    # Build n keys where ALL lower passes (0..pass_idx-1) are equal (0),
    # and the current pass has values cycling through [0, 3] to create ties.
    base = torch.zeros(n, dtype=torch.int64, device=MPS)
    # Current-pass digit: 0 for the first half, 1 for the second half.
    # Within each group, emission order should be preserved.
    digit = torch.tensor([i % 2 for i in range(n)], dtype=torch.int64, device=MPS)
    ids = base | (digit << shift)
    vals = _vals(n)

    s_ids, s_vals = _get_C().radix_sort(ids, vals)
    s_ids_cpu  = s_ids.cpu()
    s_vals_cpu = s_vals.cpu()

    # Keys must be non-decreasing.
    _assert_sorted(s_ids_cpu)

    # Within each digit group, vals must appear in ascending order
    # (i.e., original emission order is preserved).
    for d in range(2):
        mask = (s_ids_cpu >> shift) & 0xFF == d
        group_vals = s_vals_cpu[mask]
        assert (group_vals[1:] > group_vals[:-1]).all().item(), \
            f"pass {pass_idx}: digit {d} group not in emission order: {group_vals.tolist()}"


def test_stable_same_tile_different_depths():
    """Same tile, adjacent float32 depths — radix_sort must resolve by depth bits."""
    tile_n_bits = 5
    k_base = (0 << tile_n_bits | 7) << 32  # image=0, tile=7

    d_ref = struct.unpack(">I", struct.pack(">f", 0.1))[0]  # 0x3DCCCCCD
    # Three adjacent bit-patterns (differ only in the low byte)
    ids = torch.tensor(
        [k_base | (d_ref + 2), k_base | d_ref, k_base | (d_ref + 1)],
        dtype=torch.int64, device=MPS,
    )
    vals = torch.tensor([10, 20, 30], dtype=torch.int32, device=MPS)

    s_ids, s_vals = _get_C().radix_sort(ids, vals)
    s_ids_cpu  = s_ids.cpu()
    s_vals_cpu = s_vals.cpu()

    # Expected order: d_ref, d_ref+1, d_ref+2 → vals [20, 30, 10]
    assert s_ids_cpu[0].item() == k_base | d_ref
    assert s_ids_cpu[1].item() == k_base | (d_ref + 1)
    assert s_ids_cpu[2].item() == k_base | (d_ref + 2)
    _assert_exact(s_vals_cpu, torch.tensor([20, 30, 10], dtype=torch.int32))


def test_stable_same_image_different_tiles():
    """Same image, tiles 0–7 each with two records at different depths.
    Tiles must be in ascending order; within each tile, depths are ascending."""
    tile_n_bits = 4  # 4×4 grid
    image_id = 1
    d_lo = struct.unpack(">I", struct.pack(">f", 0.1))[0]
    d_hi = struct.unpack(">I", struct.pack(">f", 0.9))[0]

    entries = []
    for tile in range(8):
        base = (image_id << tile_n_bits | tile) << 32
        entries.extend([base | d_hi, base | d_lo])  # emit high-depth first

    ids  = torch.tensor(entries, dtype=torch.int64,  device=MPS)
    vals = torch.arange(len(entries), dtype=torch.int32, device=MPS)

    s_ids, s_vals = _get_C().radix_sort(ids, vals)
    s_ids_cpu  = s_ids.cpu()
    s_vals_cpu = s_vals.cpu()

    _assert_sorted(s_ids_cpu)

    # For each tile, the low-depth record (originally at odd positions) must
    # precede the high-depth record (originally at even positions).
    for i, tile in enumerate(range(8)):
        base = (image_id << tile_n_bits | tile) << 32
        lo_pos = (s_ids_cpu == (base | d_lo)).nonzero(as_tuple=True)[0].item()
        hi_pos = (s_ids_cpu == (base | d_hi)).nonzero(as_tuple=True)[0].item()
        assert lo_pos < hi_pos, f"tile {tile}: lo-depth not before hi-depth"
        # Emission order: hi was emitted first (val = tile*2), lo second (val = tile*2+1)
        assert s_vals_cpu[lo_pos].item() == tile * 2 + 1  # original index of lo
        assert s_vals_cpu[hi_pos].item() == tile * 2      # original index of hi


def test_stable_tiebreak_flatten_ids():
    """Two records with bit-identical keys — original emission order must be kept."""
    key = _pack(0, 0, 0.5)
    ids  = torch.tensor([key, key], dtype=torch.int64,  device=MPS)
    vals = torch.tensor([99, 42],   dtype=torch.int32, device=MPS)
    _, s_vals = _get_C().radix_sort(ids, vals)
    _assert_exact(s_vals.cpu(), torch.tensor([99, 42], dtype=torch.int32))


def test_stable_multipass_composition():
    """Four-level tiebreak chain: correctness requires all 4 lowest passes stable.

    Key layout (tile_n_bits=8, 1 image):
      bits 0–7  : depth low byte  — distinguishes within byte-0 ties
      bits 8–15 : depth high byte — distinguishes within byte-1 ties
      bits 32–39: tile low byte   — distinguishes within byte-4 ties
      bits 40–47: tile high byte  — distinguishes within byte-5 ties
      bits 48+  : image id = 0

    We construct 16 records that form 4 levels of tiebreaks, verifiable only
    if passes 0, 1, 4, and 5 are all individually stable.
    """
    tile_n_bits = 16  # 256×256 tile grid

    def make_key(tile_hi, tile_lo, depth_hi, depth_lo):
        tile_id = (tile_hi << 8) | tile_lo
        depth_bits = (depth_hi << 8) | depth_lo
        return (0 << tile_n_bits | tile_id) << 32 | depth_bits

    # 16 records: all combinations of (tile_hi, tile_lo, depth_hi, depth_lo) ∈ {0,1}^4
    # Emit in reverse order so any stable pass must move records.
    from itertools import product as iproduct
    combos = list(iproduct([1, 0], repeat=4))  # descending order
    keys = [make_key(*c) for c in combos]
    ids  = torch.tensor(keys, dtype=torch.int64,  device=MPS)
    vals = torch.arange(len(keys), dtype=torch.int32, device=MPS)

    s_ids, s_vals = _get_C().radix_sort(ids, vals)
    s_ids_cpu  = s_ids.cpu()
    s_vals_cpu = s_vals.cpu()

    # Keys must be sorted.
    _assert_sorted(s_ids_cpu)

    # The expected order is the ascending order of the 4-tuple (tile_hi, tile_lo, depth_hi, depth_lo).
    expected_order = sorted(range(len(combos)), key=lambda i: keys[i])
    expected_vals  = torch.tensor(expected_order, dtype=torch.int32)
    _assert_exact(s_vals_cpu, expected_vals)


# ---------------------------------------------------------------------------
# v1 vs v2 agreement
# ---------------------------------------------------------------------------

def test_v1_v2_agree():
    """mps_lsd_sort (CPU) and radix_sort (GPU) must produce identical output."""
    torch.manual_seed(13)
    n = 5_000
    tile_n_bits = 5
    ids = (
        torch.randint(0, 4, (n,), dtype=torch.int64) << (tile_n_bits + 32)
        | torch.randint(0, 32, (n,), dtype=torch.int64) << 32
        | (torch.rand(n).view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    ).to(MPS)
    vals = _vals(n)

    c_ids, c_vals = _both(_get_C().mps_lsd_sort, ids, vals)
    r_ids, r_vals = _both(_get_C().radix_sort,   ids, vals)

    _assert_exact(r_ids,  c_ids)
    _assert_exact(r_vals, c_vals)
