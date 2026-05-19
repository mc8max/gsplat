# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python wrappers for Metal-backed gsplat custom operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ._backend import load
@dataclass(frozen=True)
class _IntersectTileInputs:
    I: int
    n_elements: int

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


def _prepare_intersect_tile_inputs(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    packed: bool,
    n_images: Optional[int],
    image_ids: Optional[torch.Tensor],
    gaussian_ids: Optional[torch.Tensor],
    conics: Optional[torch.Tensor],
    opacities: Optional[torch.Tensor],
    segmented: bool,
) -> _IntersectTileInputs:
    if means2d.device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {means2d.device}")
    if radii.device != means2d.device or depths.device != means2d.device:
        raise ValueError("means2d, radii, and depths must be on the same device")
    if tile_size <= 0 or tile_width <= 0 or tile_height <= 0:
        raise ValueError(
            f"tile_size/tile_width/tile_height must be positive, got "
            f"{tile_size}/{tile_width}/{tile_height}"
        )
    if packed:
        nnz = means2d.size(0)
        if means2d.shape != (nnz, 2):
            raise ValueError(f"packed means2d must have shape (nnz, 2), got {means2d.shape}")
        if radii.shape != (nnz, 2):
            raise ValueError(f"packed radii must have shape (nnz, 2), got {radii.shape}")
        if depths.shape != (nnz,):
            raise ValueError(f"packed depths must have shape (nnz,), got {depths.shape}")
        if image_ids is None or gaussian_ids is None or n_images is None:
            raise ValueError(
                "image_ids, gaussian_ids, and n_images are required when packed=True"
            )
        if image_ids.shape != (nnz,):
            raise ValueError(f"packed image_ids must have shape (nnz,), got {image_ids.shape}")
        if gaussian_ids.shape != (nnz,):
            raise ValueError(
                f"packed gaussian_ids must have shape (nnz,), got {gaussian_ids.shape}"
            )
        if image_ids.device != means2d.device or gaussian_ids.device != means2d.device:
            raise ValueError("image_ids and gaussian_ids must be on the same device as means2d")
        if image_ids.dtype != torch.int64:
            raise ValueError(f"image_ids must be int64, got {image_ids.dtype}")
        if gaussian_ids.dtype != torch.int64:
            raise ValueError(f"gaussian_ids must be int64, got {gaussian_ids.dtype}")
        if n_images < 0:
            raise ValueError(f"n_images must be non-negative, got {n_images}")
        if conics is not None:
            if conics.shape != (nnz, 3):
                raise ValueError(f"packed conics must have shape (nnz, 3), got {conics.shape}")
            if conics.device != means2d.device or conics.dtype != torch.float32:
                raise ValueError("packed conics must be float32 on the same device as means2d")
        if opacities is not None:
            if opacities.shape != (nnz,):
                raise ValueError(
                    f"packed opacities must have shape (nnz,), got {opacities.shape}"
                )
            if opacities.device != means2d.device or opacities.dtype != torch.float32:
                raise ValueError(
                    "packed opacities must be float32 on the same device as means2d"
                )
        I = n_images
        N = None
        n_elements = nnz
    else:
        image_dims = means2d.shape[:-2]
        N = means2d.shape[-2]
        if radii.shape != image_dims + (N, 2):
            raise ValueError(f"radii must have shape {image_dims + (N, 2)}, got {radii.shape}")
        if depths.shape != image_dims + (N,):
            raise ValueError(f"depths must have shape {image_dims + (N,)}, got {depths.shape}")
        if image_ids is not None or gaussian_ids is not None:
            raise ValueError("image_ids and gaussian_ids must be omitted when packed=False")
        if conics is not None:
            if conics.shape != image_dims + (N, 3):
                raise ValueError(
                    f"conics must have shape {image_dims + (N, 3)}, got {conics.shape}"
                )
            if conics.device != means2d.device or conics.dtype != torch.float32:
                raise ValueError("conics must be float32 on the same device as means2d")
        if opacities is not None:
            if opacities.shape != image_dims + (N,):
                raise ValueError(
                    f"opacities must have shape {image_dims + (N,)}, got {opacities.shape}"
                )
            if opacities.device != means2d.device or opacities.dtype != torch.float32:
                raise ValueError("opacities must be float32 on the same device as means2d")
        I = math.prod(image_dims)
        n_elements = I * N

    if depths.dtype != torch.float32:
        raise ValueError(
            f"Metal intersect_tiles currently requires float32 depths, got {depths.dtype}"
        )
    if means2d.dtype != torch.float32:
        raise ValueError(
            f"Metal intersect_tiles currently requires float32 means2d, got {means2d.dtype}"
        )
    if radii.dtype != torch.int32:
        raise ValueError(
            f"Metal intersect_tiles currently requires int32 radii, got {radii.dtype}"
        )

    return _IntersectTileInputs(
        I=I,
        n_elements=n_elements,
    )

def intersect_tile_count(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[torch.Tensor] = None,
    gaussian_ids: Optional[torch.Tensor] = None,
    conics: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
    segmented: bool = False,
) -> torch.Tensor:
    """Count per-Gaussian tile overlaps on MPS."""

    inputs = _prepare_intersect_tile_inputs(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=n_images,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics,
        opacities=opacities,
        segmented=segmented,
    )
    return _make_lazy_metal_func("metal_intersect_tile_count")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous() if conics is not None else None,
        opacities.contiguous() if opacities is not None else None,
        image_ids.contiguous() if image_ids is not None else None,
        gaussian_ids.contiguous() if gaussian_ids is not None else None,
        inputs.I,
        tile_size,
        tile_width,
        tile_height,
        packed,
        segmented,
    )


def intersect_tile_emit(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    cum_tiles_per_gauss: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[torch.Tensor] = None,
    gaussian_ids: Optional[torch.Tensor] = None,
    conics: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
    segmented: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit compact tile-intersection records on MPS using prefix-sum offsets."""

    inputs = _prepare_intersect_tile_inputs(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=n_images,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics,
        opacities=opacities,
        segmented=segmented,
    )

    if cum_tiles_per_gauss.device != means2d.device:
        raise ValueError("cum_tiles_per_gauss must be on the same device as means2d")
    if cum_tiles_per_gauss.dtype != torch.int64:
        raise ValueError(f"cum_tiles_per_gauss must be int64, got {cum_tiles_per_gauss.dtype}")
    if cum_tiles_per_gauss.shape != (inputs.n_elements,):
        raise ValueError(
            "cum_tiles_per_gauss must have shape "
            f"({inputs.n_elements},), got {tuple(cum_tiles_per_gauss.shape)}"
        )
    return _make_lazy_metal_func("metal_intersect_tile_emit")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous() if conics is not None else None,
        opacities.contiguous() if opacities is not None else None,
        image_ids.contiguous() if image_ids is not None else None,
        gaussian_ids.contiguous() if gaussian_ids is not None else None,
        inputs.I,
        tile_size,
        tile_width,
        tile_height,
        cum_tiles_per_gauss.contiguous(),
        packed,
        segmented,
    )


def intersect_tiles(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[torch.Tensor] = None,
    gaussian_ids: Optional[torch.Tensor] = None,
    conics: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
):
    """Map projected Gaussians to intersecting tiles on MPS.

    This Metal backend implementation uses the CUDA-style AABB path by default
    and switches to the AccuTile ellipse refinement when both ``conics`` and
    ``opacities`` are provided.
    """
    inputs = _prepare_intersect_tile_inputs(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=n_images,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics,
        opacities=opacities,
        segmented=segmented,
    )
    return _make_lazy_metal_func("metal_intersect_tile")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous() if conics is not None else None,
        opacities.contiguous() if opacities is not None else None,
        image_ids.contiguous() if image_ids is not None else None,
        gaussian_ids.contiguous() if gaussian_ids is not None else None,
        inputs.I,
        tile_size,
        tile_width,
        tile_height,
        sort,
        packed,
        segmented,
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


class _FullyFusedProjection(torch.autograd.Function):
    """Autograd bridge for the Metal fused 3DGS projection op."""

    @staticmethod
    def forward(
        ctx,
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model="pinhole",
        opacities=None,
    ):
        if camera_model == "ftheta":
            raise ValueError(
                "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
            )
        if camera_model not in _CAMERA_MODEL_TO_INT:
            raise ValueError(
                f"camera_model must be one of {tuple(_CAMERA_MODEL_TO_INT)}, got {camera_model}"
            )

        camera_model_type = _CAMERA_MODEL_TO_INT[camera_model]
        radii, means2d, depths, conics, compensations = _make_lazy_metal_func(
            "metal_projection_ewa_3dgs_fused_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        ctx.save_for_backward(
            means,
            covars if covars is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            quats if quats is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            scales if scales is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            viewmats,
            Ks,
            radii,
            conics,
            compensations
            if compensations is not None
            else torch.empty(0, device=means.device, dtype=means.dtype),
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model_type = camera_model_type
        ctx.has_covars = covars is not None
        ctx.has_quats = quats is not None
        ctx.has_scales = scales is not None
        ctx.has_compensations = compensations is not None
        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        means, covars, quats, scales, viewmats, Ks, radii, conics, compensations = ctx.saved_tensors
        covars = covars if ctx.has_covars else None
        quats = quats if ctx.has_quats else None
        scales = scales if ctx.has_scales else None
        compensations = compensations if ctx.has_compensations else None
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()

        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_metal_func(
            "metal_projection_ewa_3dgs_fused_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            ctx.camera_model_type,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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


def fully_fused_projection(
    means: torch.Tensor,
    covars: Optional[torch.Tensor],
    quats: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: Optional[torch.Tensor] = None,
):
    """Project world-space 3DGS Gaussians to screen-space on MPS."""

    if packed:
        raise NotImplementedError("Metal fully_fused_projection currently supports packed=False only")
    if sparse_grad:
        raise NotImplementedError("Metal fully_fused_projection does not support sparse_grad")
    if camera_model == "ftheta":
        raise ValueError(
            "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
        )
    if camera_model not in _CAMERA_MODEL_TO_INT:
        raise ValueError(
            f"camera_model must be one of {tuple(_CAMERA_MODEL_TO_INT)}, got {camera_model}"
        )
    if means.device.type != "mps":
        raise ValueError(f"means must be on MPS, got {means.device}")
    if means.dtype != torch.float32:
        raise ValueError(f"means must be float32, got {means.dtype}")
    if viewmats.device != means.device or Ks.device != means.device:
        raise ValueError("viewmats and Ks must be on the same device as means")
    if viewmats.dtype != torch.float32 or Ks.dtype != torch.float32:
        raise ValueError("viewmats and Ks must be float32")

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    if means.shape != batch_dims + (N, 3):
        raise ValueError(f"means must have shape [..., N, 3], got {means.shape}")
    if viewmats.shape != batch_dims + (C, 4, 4):
        raise ValueError(f"viewmats must have shape {batch_dims + (C, 4, 4)}, got {viewmats.shape}")
    if Ks.shape != batch_dims + (C, 3, 3):
        raise ValueError(f"Ks must have shape {batch_dims + (C, 3, 3)}, got {Ks.shape}")

    if covars is not None:
        if quats is not None or scales is not None:
            raise ValueError("covars and {quats, scales} are mutually exclusive")
        if covars.device != means.device or covars.dtype != torch.float32:
            raise ValueError("covars must be float32 on the same device as means")
        if covars.shape != batch_dims + (N, 6):
            raise ValueError(f"covars must have shape {batch_dims + (N, 6)}, got {covars.shape}")
    else:
        if quats is None or scales is None:
            raise ValueError("either covars or {quats, scales} must be provided")
        if quats.device != means.device or scales.device != means.device:
            raise ValueError("quats and scales must be on the same device as means")
        if quats.dtype != torch.float32 or scales.dtype != torch.float32:
            raise ValueError("quats and scales must be float32")
        if quats.shape != batch_dims + (N, 4):
            raise ValueError(f"quats must have shape {batch_dims + (N, 4)}, got {quats.shape}")
        if scales.shape != batch_dims + (N, 3):
            raise ValueError(f"scales must have shape {batch_dims + (N, 3)}, got {scales.shape}")

    if opacities is not None:
        if opacities.device != means.device or opacities.dtype != torch.float32:
            raise ValueError("opacities must be float32 on the same device as means")
        if opacities.shape != batch_dims + (N,):
            raise ValueError(f"opacities must have shape {batch_dims + (N,)}, got {opacities.shape}")
        opacities = opacities.contiguous()
    return _FullyFusedProjection.apply(
        means.contiguous(),
        covars.contiguous() if covars is not None else None,
        quats.contiguous() if quats is not None else None,
        scales.contiguous() if scales is not None else None,
        viewmats.contiguous(),
        Ks.contiguous(),
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model,
        opacities,
    )
