# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python wrappers for Metal-backed gsplat custom operators."""

from __future__ import annotations

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
