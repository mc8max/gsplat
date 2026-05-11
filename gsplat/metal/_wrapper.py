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
