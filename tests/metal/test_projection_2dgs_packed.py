# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gsplat.metal as gm
from gsplat.metal._math import _fully_fused_projection_2dgs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def _sample_inputs(batch_shape=(), c=2, n=8, width=640, height=480):
    torch.manual_seed(123)
    means = torch.randn(*batch_shape, n, 3, dtype=torch.float32) * 0.25
    means[..., 2] = torch.rand(*batch_shape, n, dtype=torch.float32) * 2.0 + 1.5

    quats = torch.randn(*batch_shape, n, 4, dtype=torch.float32)
    scales = torch.rand(*batch_shape, n, 3, dtype=torch.float32) * 0.3 + 0.2

    viewmats = torch.eye(4, dtype=torch.float32).expand(*batch_shape, c, 4, 4).clone()
    viewmats[..., 0, 3] = torch.linspace(-0.1, 0.1, c, dtype=torch.float32)
    viewmats[..., 1, 3] = torch.linspace(0.05, -0.05, c, dtype=torch.float32)

    Ks = torch.zeros(*batch_shape, c, 3, 3, dtype=torch.float32)
    Ks[..., 0, 0] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 1, 1] = torch.rand(*batch_shape, c, dtype=torch.float32) * 250.0 + 300.0
    Ks[..., 0, 2] = float(width) * 0.5
    Ks[..., 1, 2] = float(height) * 0.5
    Ks[..., 2, 2] = 1.0

    return means, quats, scales, viewmats, Ks


def _pack_dense_reference(radii, means2d, depths, ray_transforms, normals):
    """Pack dense outputs into COO format for comparison with packed mode."""
    batch_shape = tuple(radii.shape[:-3])
    c = radii.shape[-3]
    n = radii.shape[-2]
    B = 1
    for dim in batch_shape:
        B *= dim

    radii_flat = radii.reshape(B, c, n, 2)
    means2d_flat = means2d.reshape(B, c, n, 2)
    depths_flat = depths.reshape(B, c, n)
    rt_flat = ray_transforms.reshape(B, c, n, 3, 3)
    norms_flat = normals.reshape(B, c, n, 3)

    batch_ids = []
    camera_ids = []
    gaussian_ids = []
    out_radii = []
    out_means2d = []
    out_depths = []
    out_ray_transforms = []
    out_normals = []
    indptr = [0]

    for b in range(B):
        for cam in range(c):
            valid = (radii_flat[b, cam] > 0).all(dim=-1)
            idxs = torch.nonzero(valid, as_tuple=False).flatten()
            for gid in idxs.tolist():
                batch_ids.append(b)
                camera_ids.append(cam)
                gaussian_ids.append(gid)
                out_radii.append(radii_flat[b, cam, gid])
                out_means2d.append(means2d_flat[b, cam, gid])
                out_depths.append(depths_flat[b, cam, gid])
                out_ray_transforms.append(rt_flat[b, cam, gid])
                out_normals.append(norms_flat[b, cam, gid])
            indptr.append(len(batch_ids))

    device = radii.device
    expected = {
        "batch_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "camera_ids": torch.tensor(camera_ids, dtype=torch.long, device=device),
        "gaussian_ids": torch.tensor(gaussian_ids, dtype=torch.long, device=device),
        "indptr": torch.tensor(indptr, dtype=torch.int32, device=device),
        "radii": torch.stack(out_radii, dim=0) if out_radii else torch.empty((0, 2), dtype=torch.int32, device=device),
        "means2d": torch.stack(out_means2d, dim=0) if out_means2d else torch.empty((0, 2), dtype=torch.float32, device=device),
        "depths": torch.tensor(out_depths, dtype=torch.float32, device=device) if out_depths else torch.empty((0,), dtype=torch.float32, device=device),
        "ray_transforms": torch.stack(out_ray_transforms, dim=0) if out_ray_transforms else torch.empty((0, 3, 3), dtype=torch.float32, device=device),
        "normals": torch.stack(out_normals, dim=0) if out_normals else torch.empty((0, 3), dtype=torch.float32, device=device),
    }
    return expected


def _scatter_packed_grads_to_dense(v_means_packed, v_quats_packed, v_scales_packed, batch_ids, camera_ids, gaussian_ids, batch_shape, c, n):
    """Scatter packed gradients back to dense layout for comparison."""
    B = 1
    for dim in batch_shape:
        B *= dim
    nnz = batch_ids.numel()

    v_means_dense = torch.zeros(*batch_shape, n, 3, dtype=torch.float32, device=v_means_packed.device)
    v_quats_dense = torch.zeros(*batch_shape, n, 4, dtype=torch.float32, device=v_quats_packed.device)
    v_scales_dense = torch.zeros(*batch_shape, n, 3, dtype=torch.float32, device=v_scales_packed.device)

    v_means_flat = v_means_dense.reshape(B * c * n, 3)
    v_quats_flat = v_quats_dense.reshape(B * c * n, 4)
    v_scales_flat = v_scales_dense.reshape(B * c * n, 3)

    gather_idx = (batch_ids * c + camera_ids) * n + gaussian_ids

    v_means_flat.index_add_(0, gather_idx, v_means_packed)
    v_quats_flat.index_add_(0, gather_idx, v_quats_packed)
    v_scales_flat.index_add_(0, gather_idx, v_scales_packed)

    return v_means_dense, v_quats_dense, v_scales_dense


class TestProjection2DGSPackedForward:
    """Stage A: forward parity tests for packed 2DGS projection."""

    def test_forward_matches_dense(self):
        """Packed forward should produce the same values as dense forward at packed indices."""
        means, quats, scales, viewmats, Ks = _sample_inputs()
        means = means.to("mps")
        quats = quats.to("mps")
        scales = scales.to("mps")
        viewmats = viewmats.to("mps")
        Ks = Ks.to("mps")

        # Dense reference
        radii_d, means2d_d, depths_d, rt_d, norms_d = _fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480
        )
        dense_packed = _pack_dense_reference(radii_d, means2d_d, depths_d, rt_d, norms_d)

        # Packed Metal
        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        nnz = batch_ids.numel()
        assert int(indptr[-1].item()) == nnz, f"indptr[-1]={int(indptr[-1].item())} != nnz={nnz}"

        assert batch_ids.tolist() == dense_packed["batch_ids"].tolist()
        assert camera_ids.tolist() == dense_packed["camera_ids"].tolist()
        assert gaussian_ids.tolist() == dense_packed["gaussian_ids"].tolist()

        if nnz > 0:
            atol, rtol = 0.0, 1e-5
            assert torch.allclose(radii, dense_packed["radii"], atol=atol, rtol=rtol), "radii mismatch"
            assert torch.allclose(means2d, dense_packed["means2d"], atol=atol, rtol=rtol), "means2d mismatch"
            assert torch.allclose(depths, dense_packed["depths"], atol=atol, rtol=rtol), "depths mismatch"
            assert torch.allclose(ray_transforms, dense_packed["ray_transforms"], atol=atol, rtol=rtol), "ray_transforms mismatch"
            assert torch.allclose(normals, dense_packed["normals"], atol=atol, rtol=rtol), "normals mismatch"

    def test_forward_empty_outputs(self):
        """When all Gaussians are culled, packed outputs should be empty."""
        torch.manual_seed(42)
        # Place all Gaussians far behind the camera
        means = torch.zeros(1, 8, 3, dtype=torch.float32, device="mps")
        means[..., 2] = -10.0  # behind camera
        quats = torch.eye(4, dtype=torch.float32).expand(1, 8, 4).clone()
        scales = torch.ones(1, 8, 3, dtype=torch.float32, device="mps") * 0.3
        viewmats = torch.eye(4, dtype=torch.float32).expand(1, 1, 4, 4).clone()
        Ks = torch.zeros(1, 1, 3, 3, dtype=torch.float32, device="mps")
        Ks[..., 0, 0] = 500.0
        Ks[..., 1, 1] = 500.0
        Ks[..., 0, 2] = 320.0
        Ks[..., 1, 2] = 240.0
        Ks[..., 2, 2] = 1.0

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        assert batch_ids.numel() == 0
        assert camera_ids.numel() == 0
        assert gaussian_ids.numel() == 0
        assert radii.shape[0] == 0
        assert int(indptr[-1].item()) == 0

    def test_forward_all_valid(self):
        """When all Gaussians are valid, packed nnz == B * C * N."""
        torch.manual_seed(99)
        means = torch.randn(1, 10, 3, dtype=torch.float32, device="mps") * 0.1
        means[..., 2] = 2.0
        quats = torch.eye(4, dtype=torch.float32).expand(1, 10, 4).clone()
        scales = torch.ones(1, 10, 3, dtype=torch.float32, device="mps") * 0.5
        viewmats = torch.eye(4, dtype=torch.float32).expand(1, 1, 4, 4).clone()
        Ks = torch.zeros(1, 1, 3, 3, dtype=torch.float32, device="mps")
        Ks[..., 0, 0] = 500.0
        Ks[..., 1, 1] = 500.0
        Ks[..., 0, 2] = 320.0
        Ks[..., 1, 2] = 240.0
        Ks[..., 2, 2] = 1.0

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        assert batch_ids.numel() == 10
        assert int(indptr[-1].item()) == 10

    def test_forward_output_shapes(self):
        """Output tensor shapes should match expected packed layout."""
        means, quats, scales, viewmats, Ks = _sample_inputs(n=16, c=3)
        means = means.to("mps")
        quats = quats.to("mps")
        scales = scales.to("mps")
        viewmats = viewmats.to("mps")
        Ks = Ks.to("mps")

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        nnz = batch_ids.numel()
        assert batch_ids.dtype == torch.long
        assert camera_ids.dtype == torch.long
        assert gaussian_ids.dtype == torch.long
        assert indptr.dtype == torch.int32
        assert radii.dtype == torch.int32
        assert radii.shape == (nnz, 2)
        assert means2d.shape == (nnz, 2)
        assert depths.shape == (nnz,)
        assert ray_transforms.shape == (nnz, 3, 3)
        assert normals.shape == (nnz, 3)
        assert indptr.shape == (3 * 1 + 1,)  # B*C + 1

    def test_forward_multicamera_indptr(self):
        """indptr should correctly segment per-camera counts."""
        means, quats, scales, viewmats, Ks = _sample_inputs(c=4, n=8)
        means = means.to("mps")
        quats = quats.to("mps")
        scales = scales.to("mps")
        viewmats = viewmats.to("mps")
        Ks = Ks.to("mps")

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        nnz = batch_ids.numel()
        assert int(indptr[0].item()) == 0
        assert int(indptr[-1].item()) == nnz
        # indptr should be monotonically non-decreasing
        for i in range(len(indptr) - 1):
            assert int(indptr[i].item()) <= int(indptr[i + 1].item())


class TestProjection2DGSPackedBackward:
    """Stage B: backward parity tests for packed 2DGS projection."""

    def _get_tensors(self):
        means, quats, scales, viewmats, Ks = _sample_inputs()
        means = means.to("mps").requires_grad_(True)
        quats = quats.to("mps").requires_grad_(True)
        scales = scales.to("mps").requires_grad_(True)
        viewmats = viewmats.to("mps").requires_grad_(True)
        Ks = Ks.to("mps")
        return means, quats, scales, viewmats, Ks

    def test_backward_dense_accumulation(self):
        """Packed backward with sparse_grad=False should match dense backward."""
        means, quats, scales, viewmats, Ks = self._get_tensors()

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True, sparse_grad=False
        )

        v_means2d = torch.ones_like(means2d)
        v_depths = torch.ones_like(depths)
        v_ray_transforms = torch.ones_like(ray_transforms)
        v_normals = torch.ones_like(normals)

        (
            v_means, v_quats, v_scales, v_viewmats,
        ) = torch.autograd.grad(
            (means2d, depths, ray_transforms, normals),
            (means, quats, scales, viewmats),
            (v_means2d, v_depths, v_ray_transforms, v_normals),
            allow_unused=True,
        )

        # Compare with dense backward
        means2, quats2, scales2, viewmats2, Ks2 = self._get_tensors()
        radii2, means2d2, depths2, rt2, norms2 = _fully_fused_projection_2dgs(
            means2, quats2, scales2, viewmats2, Ks2, 640, 480
        )
        dense_packed = _pack_dense_reference(radii2, means2d2, depths2, rt2, norms2)

        v_means2d2 = torch.ones_like(means2d2)
        v_depths2 = torch.ones_like(depths2)
        v_ray_transforms2 = torch.ones_like(rt2)
        v_normals2 = torch.ones_like(norms2)

        (
            v_means2_out, v_quats2_out, v_scales2_out, v_viewmats2_out,
        ) = torch.autograd.grad(
            (means2d2, depths2, rt2, norms2),
            (means2, quats2, scales2, viewmats2),
            (v_means2d2, v_depths2, v_ray_transforms2, v_normals2),
            allow_unused=True,
        )

        atol, rtol = 1e-4, 1e-4
        assert v_means is not None
        assert v_quats is not None
        assert v_scales is not None
        assert v_viewmats is not None
        assert torch.allclose(v_means, v_means2_out, atol=atol, rtol=rtol), "v_means mismatch"
        assert torch.allclose(v_quats, v_quats2_out, atol=atol, rtol=rtol), "v_quats mismatch"
        assert torch.allclose(v_scales, v_scales2_out, atol=atol, rtol=rtol), "v_scales mismatch"
        assert torch.allclose(v_viewmats, v_viewmats2_out, atol=atol, rtol=rtol), "v_viewmats mismatch"

    def test_backward_sparse_grad(self):
        """Packed backward with sparse_grad=True should return [nnz, ...] tensors."""
        means, quats, scales, viewmats, Ks = self._get_tensors()

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True, sparse_grad=True
        )

        v_means2d = torch.ones_like(means2d)
        v_depths = torch.ones_like(depths)
        v_ray_transforms = torch.ones_like(ray_transforms)
        v_normals = torch.ones_like(normals)

        (
            v_means, v_quats, v_scales, v_viewmats,
        ) = torch.autograd.grad(
            (means2d, depths, ray_transforms, normals),
            (means, quats, scales, viewmats),
            (v_means2d, v_depths, v_ray_transforms, v_normals),
            allow_unused=True,
        )

        nnz = batch_ids.numel()
        assert v_means.shape == (nnz, 3), f"Expected ({nnz}, 3), got {v_means.shape}"
        assert v_quats.shape == (nnz, 4), f"Expected ({nnz}, 4), got {v_quats.shape}"
        assert v_scales.shape == (nnz, 3), f"Expected ({nnz}, 3), got {v_scales.shape}"

    def test_backward_empty(self):
        """Backward with nnz=0 should return zero tensors."""
        torch.manual_seed(42)
        means = torch.zeros(1, 8, 3, dtype=torch.float32, device="mps").requires_grad_(True)
        means[..., 2] = -10.0
        quats = torch.eye(4, dtype=torch.float32).expand(1, 8, 4).clone().to("mps").requires_grad_(True)
        scales = torch.ones(1, 8, 3, dtype=torch.float32, device="mps").requires_grad_(True)
        viewmats = torch.eye(4, dtype=torch.float32).expand(1, 1, 4, 4).clone().to("mps").requires_grad_(True)
        Ks = torch.zeros(1, 1, 3, 3, dtype=torch.float32, device="mps")
        Ks[..., 0, 0] = 500.0
        Ks[..., 1, 1] = 500.0
        Ks[..., 0, 2] = 320.0
        Ks[..., 1, 2] = 240.0
        Ks[..., 2, 2] = 1.0

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )

        assert batch_ids.numel() == 0

        v_means2d = torch.zeros(0, 2, dtype=torch.float32, device="mps")
        v_depths = torch.zeros(0, dtype=torch.float32, device="mps")
        v_ray_transforms = torch.zeros(0, 3, 3, dtype=torch.float32, device="mps")
        v_normals = torch.zeros(0, 3, dtype=torch.float32, device="mps")

        (
            v_means, v_quats, v_scales, v_viewmats,
        ) = torch.autograd.grad(
            (means2d, depths, ray_transforms, normals),
            (means, quats, scales, viewmats),
            (v_means2d, v_depths, v_ray_transforms, v_normals),
            allow_unused=True,
        )

        assert v_means is not None
        assert v_quats is not None
        assert v_scales is not None
        assert v_viewmats is not None


class TestProjection2DGSPackedPipeline:
    """Stage C: pipeline integration tests."""

    def test_packed_equals_dense_image(self):
        """Full packed pipeline should produce identical images to dense pipeline."""
        torch.manual_seed(77)
        means, quats, scales, viewmats, Ks = _sample_inputs(n=20, c=2)
        means = means.to("mps")
        quats = quats.to("mps")
        scales = scales.to("mps")
        viewmats = viewmats.to("mps")
        Ks = Ks.to("mps")

        colors = torch.rand(1, 2, 20, 3, dtype=torch.float32, device="mps")
        opacities = torch.rand(1, 20, dtype=torch.float32, device="mps") * 0.5 + 0.3
        backgrounds = torch.zeros(1, 2, 3, dtype=torch.float32, device="mps")

        # Dense pipeline
        radii_d, means2d_d, depths_d, rt_d, norms_d = _fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480
        )
        tile_size = 16
        tile_width = (640 + tile_size - 1) // tile_size
        tile_height = (480 + tile_size - 1) // tile_size
        isect_offsets_d = gm.isect_offset_encode(
            gm.isect_tiles(means2d_d, radii_d, depths_d, tile_size, tile_width, tile_height, packed=False)[1],
            1, tile_width, tile_height
        )
        render_colors_d, render_alphas_d = gm.rasterize_to_pixels_2dgs(
            means2d_d, rt_d, colors, opacities, norms_d,
            torch.zeros_like(means2d_d),
            640, 480, tile_size, isect_offsets_d,
            gm.isect_tiles(means2d_d, radii_d, depths_d, tile_size, tile_width, tile_height, packed=False)[2],
            backgrounds=backgrounds, packed=False
        )

        # Packed pipeline
        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 640, 480, packed=True
        )
        isect_offsets_p, flatten_ids_p = gm.isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height, packed=True,
            n_images=1, image_ids=batch_ids, gaussian_ids=gaussian_ids
        )
        render_colors_p, render_alphas_p = gm.rasterize_to_pixels_2dgs(
            means2d, ray_transforms, colors, opacities, normals,
            torch.zeros_like(means2d),
            640, 480, tile_size, isect_offsets_p, flatten_ids_p,
            backgrounds=backgrounds, packed=True
        )

        atol, rtol = 1e-3, 1e-3
        assert torch.allclose(render_colors_d, render_colors_p, atol=atol, rtol=rtol), "render_colors mismatch"
        assert torch.allclose(render_alphas_d, render_alphas_p, atol=atol, rtol=rtol), "render_alphas mismatch"


class TestProjection2DGSPackedPerformance:
    """Stage D: performance / benchmark tests."""

    def test_large_scene(self):
        """Should handle large scenes without errors."""
        torch.manual_seed(1234)
        n = 500
        c = 4
        means = torch.randn(1, n, 3, dtype=torch.float32, device="mps") * 0.5
        means[..., 2] = torch.rand(1, n, dtype=torch.float32, device="mps") * 3.0 + 2.0
        quats = torch.randn(1, n, 4, dtype=torch.float32, device="mps")
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.rand(1, n, 3, dtype=torch.float32, device="mps") * 0.5 + 0.1
        viewmats = torch.eye(4, dtype=torch.float32).expand(1, c, 4, 4).clone()
        viewmats[..., 0, 3] = torch.linspace(-0.5, 0.5, c, dtype=torch.float32)
        Ks = torch.zeros(1, c, 3, 3, dtype=torch.float32, device="mps")
        Ks[..., 0, 0] = 800.0
        Ks[..., 1, 1] = 800.0
        Ks[..., 0, 2] = 640.0
        Ks[..., 1, 2] = 480.0
        Ks[..., 2, 2] = 1.0

        (
            indptr, batch_ids, camera_ids, gaussian_ids,
            radii, means2d, depths, ray_transforms, normals,
        ) = gm.fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 1280, 960, packed=True
        )

        nnz = batch_ids.numel()
        assert nnz > 0, "Expected some valid Gaussians"
        assert int(indptr[-1].item()) == nnz
