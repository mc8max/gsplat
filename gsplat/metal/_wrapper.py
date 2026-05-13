# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python wrappers for Metal-backed gsplat custom operators."""

from __future__ import annotations

from typing import Optional

import torch

from ._backend import load


def _make_lazy_metal_func(name: str):
    """Return a callable that loads the Metal extension on first use."""

    def call(*args, **kwargs):
        if not load():
            raise RuntimeError("gsplat Metal extension is not available.")
        return getattr(torch.ops.gsplat, name)(*args, **kwargs)

    return call


def metal_null(x: torch.Tensor) -> torch.Tensor:
    """Identity op on MPS backed by a Metal kernel."""
    return _make_lazy_metal_func("metal_null")(x)


def eval_bivariate_poly(
    x: torch.Tensor,
    y: torch.Tensor,
    poly_coeffs: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Evaluate the external-distortion bivariate polynomial on MPS."""

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if order < 0 or order > 5:
        raise ValueError(f"order must be in [0, 5], got {order}")
    expected = (order + 1) * (order + 2) // 2
    if poly_coeffs.numel() != expected:
        raise ValueError(
            f"poly_coeffs must have {expected} coefficients for order {order}, "
            f"got {poly_coeffs.numel()}"
        )
    return _make_lazy_metal_func("metal_eval_bivariate_poly")(
        x.contiguous(),
        y.contiguous(),
        poly_coeffs.contiguous(),
        order,
    )


def distort_camera_rays(
    rays: torch.Tensor,
    h_poly: torch.Tensor,
    v_poly: torch.Tensor,
    h_inv_poly: torch.Tensor,
    v_inv_poly: torch.Tensor,
    reference_poly: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Distort rays using the external-distortion bivariate windshield model on MPS."""

    if rays.shape[-1] != 3:
        raise ValueError(f"rays last dimension must be 3, got {rays.shape[-1]}")
    return _make_lazy_metal_func("metal_distort_camera_rays")(
        rays.contiguous(),
        h_poly.contiguous(),
        v_poly.contiguous(),
        h_inv_poly.contiguous(),
        v_inv_poly.contiguous(),
        reference_poly,
        inverse,
    )


def intersect_offset_encode(
    isect_ids: torch.Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> torch.Tensor:
    """Encode sorted intersection ids into dense per-tile start offsets on MPS."""

    if isect_ids.dim() != 1:
        raise ValueError(f"isect_ids must be 1D, got shape {tuple(isect_ids.shape)}")
    if isect_ids.dtype != torch.int64:
        raise ValueError(f"isect_ids must be int64, got {isect_ids.dtype}")
    if n_images < 0:
        raise ValueError(f"n_images must be non-negative, got {n_images}")
    if tile_width <= 0:
        raise ValueError(f"tile_width must be positive, got {tile_width}")
    if tile_height <= 0:
        raise ValueError(f"tile_height must be positive, got {tile_height}")
    return _make_lazy_metal_func("metal_intersect_offset")(
        isect_ids.contiguous(),
        n_images,
        tile_width,
        tile_height,
    )


_CAMERA_MODEL_TO_INT = {
    "pinhole": 0,
    "ortho": 1,
    "fisheye": 2,
}


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Autograd bridge for the Metal quat-scale-to-covariance/precision op."""
    
    @staticmethod
    def forward(ctx, quats, scales, compute_covar=True, compute_preci=True, triu=False):
        """Run the forward Metal kernel and stash inputs for backward."""

        ctx.set_materialize_grads(False)
        covars, precis = _make_lazy_metal_func("metal_quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars, v_precis):
        """Propagate gradients through the Metal backward kernel."""

        quats, scales = ctx.saved_tensors
        if ctx.compute_covar and v_covars is not None and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if ctx.compute_preci and v_precis is not None and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_metal_func("metal_quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            ctx.triu,
            v_covars.contiguous() if ctx.compute_covar and v_covars is not None else None,
            v_precis.contiguous() if ctx.compute_preci and v_precis is not None else None,
        )
        return v_quats, v_scales, None, None, None


class _SphericalHarmonics(torch.autograd.Function):
    """Autograd bridge for the Metal spherical harmonics op."""

    @staticmethod
    def forward(ctx, sh_degree, dirs, coeffs, masks):
        """Run the forward Metal kernel and stash inputs for backward."""

        ctx.set_materialize_grads(False)
        colors = _make_lazy_metal_func("metal_spherical_harmonics_fwd")(
            sh_degree, dirs, coeffs, masks
        )
        # save_for_backward requires tensors; use an empty bool tensor as a
        # None sentinel since save_for_backward does not accept None directly.
        ctx.save_for_backward(
            dirs,
            coeffs,
            masks if masks is not None else torch.empty(0, device=dirs.device, dtype=torch.bool),
        )
        ctx.sh_degree = sh_degree
        ctx.has_masks = masks is not None
        return colors

    @staticmethod
    def backward(ctx, v_colors):
        """Propagate gradients through the Metal backward kernel."""

        if v_colors is None:
            return None, None, None, None
        dirs, coeffs, masks = ctx.saved_tensors
        masks = masks if ctx.has_masks else None
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_metal_func("metal_spherical_harmonics_bwd")(
            ctx.sh_degree,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        return None, v_dirs, v_coeffs, None


class _ProjectionEWASimple(torch.autograd.Function):
    """Autograd bridge for the Metal EWA projection op."""

    @staticmethod
    def forward(ctx, means, covars, Ks, width, height, camera_model="pinhole"):
        if camera_model == "ftheta":
            raise ValueError(
                "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
            )
        if camera_model not in _CAMERA_MODEL_TO_INT:
            raise ValueError(
                f"camera_model must be one of {tuple(_CAMERA_MODEL_TO_INT)}, got {camera_model}"
            )

        camera_model_type = _CAMERA_MODEL_TO_INT[camera_model]
        means2d, covars2d = _make_lazy_metal_func("metal_projection_ewa_simple_fwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        ctx.camera_model_type = camera_model_type
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d, v_covars2d):
        means, covars, Ks = ctx.saved_tensors
        v_means, v_covars = _make_lazy_metal_func("metal_projection_ewa_simple_bwd")(
            means,
            covars,
            Ks,
            ctx.width,
            ctx.height,
            ctx.camera_model_type,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None, None


def quat_scale_to_covar_preci(
    quats: torch.Tensor,
    scales: torch.Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
):
    """Compute covariance and precision matrices from quaternions and scales.

    Args:
        quats: Tensor with shape ``[..., 4]`` containing quaternion parameters.
        scales: Tensor with shape ``[..., 3]`` containing axis-aligned scales.
        compute_covar: Whether to return covariance matrices.
        compute_preci: Whether to return precision matrices.
        triu: Whether to pack symmetric outputs as upper-triangular vectors.

    Returns:
        A pair ``(covars, precis)`` where disabled outputs are returned as ``None``.
        Each computed output has shape ``[..., 3, 3]`` when ``triu=False`` and
        ``[..., 6]`` when ``triu=True``.
    """

    batch_dims = quats.shape[:-1]
    if quats.shape != batch_dims + (4,):
        raise ValueError(f"quats must have shape [..., 4], got {quats.shape}")
    if scales.shape != batch_dims + (3,):
        raise ValueError(f"scales must have shape [..., 3], got {scales.shape}")
    covars, precis = _QuatScaleToCovarPreci.apply(
        quats.contiguous(),
        scales.contiguous(),
        compute_covar,
        compute_preci,
        triu,
    )
    return covars if compute_covar else None, precis if compute_preci else None


def spherical_harmonics(
    degrees_to_use: int,
    dirs: torch.Tensor,
    coeffs: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute spherical harmonics colors from directions and SH coefficients."""

    num_bases_needed = (degrees_to_use + 1) ** 2
    if degrees_to_use < 0 or degrees_to_use > 4:
        raise ValueError(f"degrees_to_use must be in [0, 4], got {degrees_to_use}")
    if coeffs.shape[-2] < num_bases_needed:
        raise ValueError(
            f"coeffs must have at least {num_bases_needed} basis functions "
            f"(degrees_to_use={degrees_to_use}), got {coeffs.shape[-2]}"
        )
    if dirs.shape[-1] != 3:
        raise ValueError(f"dirs last dimension must be 3, got {dirs.shape[-1]}")
    if coeffs.shape[-1] != 3:
        raise ValueError(f"coeffs last dimension must be 3, got {coeffs.shape[-1]}")
    if dirs.shape[:-1] != coeffs.shape[:-2]:
        raise ValueError(
            f"dirs and coeffs batch dimensions must match, got dirs {dirs.shape} "
            f"and coeffs {coeffs.shape}"
        )
    if masks is not None and masks.shape != dirs.shape[:-1]:
        raise ValueError(
            f"masks shape must match batch dims {dirs.shape[:-1]}, got {masks.shape}"
        )
    return _SphericalHarmonics.apply(
        degrees_to_use,
        dirs.contiguous(),
        coeffs.contiguous(),
        masks.contiguous() if masks is not None else None,
    )


def projection_ewa_simple(
    means: torch.Tensor,
    covars: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    camera_model: str = "pinhole",
):
    """Project camera-space 3D Gaussian means and covariances to 2D on MPS."""

    if means.shape[-1] != 3:
        raise ValueError(f"means last dimension must be 3, got {means.shape[-1]}")
    if covars.shape[-2:] != (3, 3):
        raise ValueError(f"covars last two dimensions must be (3, 3), got {covars.shape[-2:]}")
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Ks last two dimensions must be (3, 3), got {Ks.shape[-2:]}")
    if means.shape[:-3] != covars.shape[:-4]:
        raise ValueError(
            f"means and covars batch dimensions must match, got {means.shape} and {covars.shape}"
        )
    if means.shape[:-3] != Ks.shape[:-3]:
        raise ValueError(
            f"means and Ks batch dimensions must match, got {means.shape} and {Ks.shape}"
        )
    if means.shape[-3] != covars.shape[-4] or means.shape[-3] != Ks.shape[-3]:
        raise ValueError("camera dimension must match across means, covars, and Ks")
    if means.shape[-2] != covars.shape[-3]:
        raise ValueError("Gaussian dimension must match between means and covars")

    return _ProjectionEWASimple.apply(
        means.contiguous(),
        covars.contiguous(),
        Ks.contiguous(),
        width,
        height,
        camera_model,
    )
