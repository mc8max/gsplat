# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python wrappers for Metal-backed gsplat custom operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from gsplat._camera_types import (
    BivariateWindshieldModelParameters,
    CameraModel,
    FThetaCameraDistortionParameters,
    RollingShutterType,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    UnscentedTransformParameters,
    viewmat_to_pose,
)

from ._backend import load


@dataclass(frozen=True)
class _IntersectTileInputs:
    I: int
    n_elements: int


@dataclass(frozen=True)
class _LidarTilePayload:
    n_bins_azimuth: int
    n_bins_elevation: int
    cdf_resolution_azimuth: int
    cdf_resolution_elevation: int
    angle_to_pixel_scaling_factor: float
    fov_horiz_start: float
    fov_horiz_span: float
    fov_vert_start: float
    fov_vert_span: float
    fov_eps: float
    spinning_direction: int
    cdf_elevation: torch.Tensor
    cdf_dense_ray_mask: torch.Tensor
    tiles_pack_info: torch.Tensor
    tiles_to_elements_map: torch.Tensor


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


def adam(
    param: torch.Tensor,
    param_grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    valid: Optional[torch.Tensor],
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    """Fused Adam update on MPS for float32/float16 tensors."""

    def _prepare_mutable_tensor(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
        if tensor.device != param.device:
            raise ValueError(f"{name} must be on the same device as param, got {tensor.device} and {param.device}")
        if tensor.dtype != param.dtype:
            raise ValueError(f"{name} must have the same dtype as param, got {tensor.dtype} and {param.dtype}")
        needs_copy_back = not tensor.is_contiguous()
        return (tensor.contiguous() if needs_copy_back else tensor, needs_copy_back)

    if param.device.type != "mps":
        raise ValueError(f"param must be on MPS, got {param.device}")
    if param.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"param must be float32 or float16, got {param.dtype}")
    if param.dim() < 1:
        raise ValueError(f"param must have at least one dimension, got {tuple(param.shape)}")
    if param.shape != param_grad.shape:
        raise ValueError(f"param and param_grad must have the same shape, got {param.shape} and {param_grad.shape}")
    if param.shape != exp_avg.shape:
        raise ValueError(f"param and exp_avg must have the same shape, got {param.shape} and {exp_avg.shape}")
    if param.shape != exp_avg_sq.shape:
        raise ValueError(
            f"param and exp_avg_sq must have the same shape, got {param.shape} and {exp_avg_sq.shape}"
        )

    param_arg, copy_param_back = _prepare_mutable_tensor(param, "param")
    param_grad_arg, _ = _prepare_mutable_tensor(param_grad, "param_grad")
    exp_avg_arg, copy_exp_avg_back = _prepare_mutable_tensor(exp_avg, "exp_avg")
    exp_avg_sq_arg, copy_exp_avg_sq_back = _prepare_mutable_tensor(exp_avg_sq, "exp_avg_sq")

    valid_arg = None
    if valid is not None:
        if valid.device.type != "mps":
            raise ValueError(f"valid must be on MPS, got {valid.device}")
        if valid.dtype != torch.bool:
            raise ValueError(f"valid must be bool, got {valid.dtype}")
        if valid.dim() != 1:
            raise ValueError(f"valid must be 1D, got {tuple(valid.shape)}")
        if valid.shape[0] != param.shape[0]:
            raise ValueError(
                f"valid first dimension must match param first dimension, got {valid.shape[0]} and {param.shape[0]}"
            )
        valid_arg = valid.contiguous()

    _make_lazy_metal_func("metal_adam")(
        param_arg,
        param_grad_arg,
        exp_avg_arg,
        exp_avg_sq_arg,
        valid_arg,
        float(lr),
        float(b1),
        float(b2),
        float(eps),
    )

    if copy_param_back:
        param.copy_(param_arg)
    if copy_exp_avg_back:
        exp_avg.copy_(exp_avg_arg)
    if copy_exp_avg_sq_back:
        exp_avg_sq.copy_(exp_avg_sq_arg)


def relocation(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    binoms: torch.Tensor,
    n_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute relocated opacity/scale values on MPS."""

    if opacities.device.type != "mps":
        raise ValueError(f"opacities must be on MPS, got {opacities.device}")
    if opacities.dtype != torch.float32:
        raise ValueError(f"opacities must be float32, got {opacities.dtype}")
    if opacities.dim() != 1:
        raise ValueError(f"opacities must have shape [N], got {tuple(opacities.shape)}")
    n = opacities.shape[0]

    if scales.device != opacities.device:
        raise ValueError(f"scales must be on the same device as opacities, got {scales.device} and {opacities.device}")
    if scales.dtype != torch.float32:
        raise ValueError(f"scales must be float32, got {scales.dtype}")
    if tuple(scales.shape) != (n, 3):
        raise ValueError(f"scales must have shape {(n, 3)}, got {tuple(scales.shape)}")

    if ratios.device != opacities.device:
        raise ValueError(f"ratios must be on the same device as opacities, got {ratios.device} and {opacities.device}")
    if ratios.dtype != torch.int32:
        raise ValueError(f"ratios must be int32, got {ratios.dtype}")
    if tuple(ratios.shape) != (n,):
        raise ValueError(f"ratios must have shape {(n,)}, got {tuple(ratios.shape)}")

    if binoms.device != opacities.device:
        raise ValueError(f"binoms must be on the same device as opacities, got {binoms.device} and {opacities.device}")
    if binoms.dtype != torch.float32:
        raise ValueError(f"binoms must be float32, got {binoms.dtype}")
    if binoms.dim() != 2:
        raise ValueError(f"binoms must be rank 2, got {binoms.dim()}")
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}")
    if tuple(binoms.shape) != (n_max, n_max):
        raise ValueError(f"binoms must have shape {(n_max, n_max)}, got {tuple(binoms.shape)}")

    return _make_lazy_metal_func("metal_relocation")(
        opacities.contiguous(),
        scales.contiguous(),
        ratios.contiguous(),
        binoms.contiguous(),
        n_max,
    )


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


def _synthesize_eval3d_world_rays(
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    camera_model: CameraModel,
    radial_coeffs: Optional[torch.Tensor] = None,
    tangential_coeffs: Optional[torch.Tensor] = None,
    thin_prism_coeffs: Optional[torch.Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from gsplat.cuda._torch_cameras import (
        _BaseCameraModel,
        _interpolate_shutter_pose,
        _pose_camera_ray_to_world_ray,
        _viewmat_to_pose,
    )

    batch_dims = viewmats.shape[:-3]
    c = viewmats.shape[-3]
    flat_count = math.prod(batch_dims) * c
    device = viewmats.device
    dtype = viewmats.dtype

    viewmats_flat = viewmats.reshape(flat_count, 4, 4)
    Ks_flat = Ks.reshape(flat_count, 3, 3)
    focal_lengths = torch.stack([Ks_flat[:, 0, 0], Ks_flat[:, 1, 1]], dim=-1)
    principal_points = torch.stack([Ks_flat[:, 0, 2], Ks_flat[:, 1, 2]], dim=-1)

    grid_x = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y = torch.arange(height, device=device, dtype=dtype) + 0.5
    px, py = torch.meshgrid(grid_x, grid_y, indexing="xy")
    image_points = torch.stack([px, py], dim=-1).reshape(1, height * width, 2)
    image_points = image_points.expand(flat_count, -1, -1)

    camera = _BaseCameraModel.create(
        width=width,
        height=height,
        camera_model=camera_model,
        principal_points=principal_points,
        focal_lengths=None if camera_model == "ftheta" else focal_lengths,
        radial_coeffs=radial_coeffs.reshape(flat_count, -1) if radial_coeffs is not None else None,
        tangential_coeffs=tangential_coeffs.reshape(flat_count, 2) if tangential_coeffs is not None else None,
        thin_prism_coeffs=thin_prism_coeffs.reshape(flat_count, 4) if thin_prism_coeffs is not None else None,
        ftheta_coeffs=ftheta_coeffs,
        rs_type=rolling_shutter,
    )
    camera_rays, valid = camera.image_point_to_camera_ray(image_points)

    if external_distortion_coeffs is not None:
        camera_rays = distort_camera_rays(
            camera_rays.reshape(-1, 3),
            external_distortion_coeffs.horizontal_poly.to(device=device, dtype=dtype),
            external_distortion_coeffs.vertical_poly.to(device=device, dtype=dtype),
            external_distortion_coeffs.horizontal_poly_inverse.to(device=device, dtype=dtype),
            external_distortion_coeffs.vertical_poly_inverse.to(device=device, dtype=dtype),
            int(external_distortion_coeffs.reference_poly),
            True,
        ).reshape(flat_count, height * width, 3)

    pose_start = _viewmat_to_pose(viewmats_flat)
    pose_end = _viewmat_to_pose(
        (viewmats_rs if viewmats_rs is not None else viewmats).reshape(flat_count, 4, 4)
    )
    relative_time = camera.shutter_relative_frame_time(image_points)
    pose = _interpolate_shutter_pose(
        pose_start[:, None, :],
        pose_end[:, None, :],
        relative_time,
    )
    ray_o, ray_d = _pose_camera_ray_to_world_ray(pose, camera_rays)
    ray_o = ray_o * valid[..., None]
    ray_d = ray_d * valid[..., None]
    rays = torch.cat([ray_o, ray_d], dim=-1)
    return rays.reshape(batch_dims + (c, height, width, 6))


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


def _prepare_intersect_tile_lidar_inputs(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    *,
    packed: bool,
    n_images: Optional[int],
    image_ids: Optional[torch.Tensor],
    gaussian_ids: Optional[torch.Tensor],
) -> _IntersectTileInputs:
    if means2d.device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {means2d.device}")
    if radii.device != means2d.device or depths.device != means2d.device:
        raise ValueError("means2d, radii, and depths must be on the same device")
    if means2d.dtype != torch.float32:
        raise ValueError(
            f"Metal intersect_tiles_lidar currently requires float32 means2d, got {means2d.dtype}"
        )
    if depths.dtype != torch.float32:
        raise ValueError(
            f"Metal intersect_tiles_lidar currently requires float32 depths, got {depths.dtype}"
        )
    if radii.dtype != torch.int32:
        raise ValueError(
            f"Metal intersect_tiles_lidar currently requires int32 radii, got {radii.dtype}"
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
        if image_ids.device != means2d.device or gaussian_ids.device != means2d.device:
            raise ValueError("image_ids and gaussian_ids must be on the same device as means2d")
        if image_ids.dtype != torch.int64:
            raise ValueError(f"image_ids must be int64, got {image_ids.dtype}")
        if gaussian_ids.dtype != torch.int64:
            raise ValueError(f"gaussian_ids must be int64, got {gaussian_ids.dtype}")
        if image_ids.shape != (nnz,):
            raise ValueError(f"packed image_ids must have shape (nnz,), got {image_ids.shape}")
        if gaussian_ids.shape != (nnz,):
            raise ValueError(
                f"packed gaussian_ids must have shape (nnz,), got {gaussian_ids.shape}"
            )
        if n_images < 0:
            raise ValueError(f"n_images must be non-negative, got {n_images}")
        I = n_images
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
        I = math.prod(image_dims)
        n_elements = I * N

    return _IntersectTileInputs(I=I, n_elements=n_elements)


def _pack_lidar_tile_payload(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    device: torch.device,
) -> _LidarTilePayload:
    from gsplat.cuda._torch_impl_lidar import ANGLE_TO_PIXEL_SCALING_FACTOR

    tiling = lidar.tiling

    if tiling.cdf_elevation.dtype != torch.int32 or tiling.cdf_elevation.ndim != 1:
        raise ValueError("lidar.tiling.cdf_elevation must be int32 with shape (R+1,)")
    if tiling.cdf_dense_ray_mask.dtype != torch.int32 or tiling.cdf_dense_ray_mask.ndim != 2:
        raise ValueError(
            "lidar.tiling.cdf_dense_ray_mask must be int32 with shape (Re+1, Ra+1)"
        )
    if tiling.tiles_pack_info.dtype != torch.int32 or tiling.tiles_pack_info.ndim != 2:
        raise ValueError(
            "lidar.tiling.tiles_pack_info must be int32 with shape (n_tiles, 2)"
        )
    if tiling.tiles_to_elements_map.dtype != torch.int32 or tiling.tiles_to_elements_map.ndim != 2:
        raise ValueError(
            "lidar.tiling.tiles_to_elements_map must be int32 with shape (n_rays, 2)"
        )
    if tiling.tiles_pack_info.shape[1] != 2:
        raise ValueError(
            "lidar.tiling.tiles_pack_info must have shape (n_tiles, 2), got "
            f"{tuple(tiling.tiles_pack_info.shape)}"
        )
    if tiling.tiles_to_elements_map.shape[1] != 2:
        raise ValueError(
            "lidar.tiling.tiles_to_elements_map must have shape (n_rays, 2), got "
            f"{tuple(tiling.tiles_to_elements_map.shape)}"
        )

    cdf_elevation = tiling.cdf_elevation.to(device=device).contiguous()
    cdf_dense_ray_mask = tiling.cdf_dense_ray_mask.to(device=device).contiguous()
    tiles_pack_info = tiling.tiles_pack_info.to(device=device).contiguous()
    tiles_to_elements_map = tiling.tiles_to_elements_map.to(device=device).contiguous()

    expected_tiles = tiling.n_bins_azimuth * tiling.n_bins_elevation
    if tiles_pack_info.shape != (expected_tiles, 2):
        raise ValueError(
            "lidar.tiling.tiles_pack_info must have shape "
            f"({expected_tiles}, 2), got {tuple(tiles_pack_info.shape)}"
        )
    if cdf_elevation[-1].item() != tiling.n_bins_elevation:
        raise ValueError(
            "lidar.tiling.cdf_elevation[-1] must equal n_bins_elevation, got "
            f"{cdf_elevation[-1].item()} and {tiling.n_bins_elevation}"
        )

    return _LidarTilePayload(
        n_bins_azimuth=tiling.n_bins_azimuth,
        n_bins_elevation=tiling.n_bins_elevation,
        cdf_resolution_azimuth=tiling.cdf_resolution_azimuth,
        cdf_resolution_elevation=tiling.cdf_resolution_elevation,
        angle_to_pixel_scaling_factor=float(ANGLE_TO_PIXEL_SCALING_FACTOR),
        fov_horiz_start=float(lidar.fov_horiz_rad.start),
        fov_horiz_span=float(lidar.fov_horiz_rad.span),
        fov_vert_start=float(lidar.fov_vert_rad.start),
        fov_vert_span=float(lidar.fov_vert_rad.span),
        fov_eps=float(lidar.fov_eps_rad),
        spinning_direction=int(lidar.spinning_direction.value),
        cdf_elevation=cdf_elevation,
        cdf_dense_ray_mask=cdf_dense_ray_mask,
        tiles_pack_info=tiles_pack_info,
        tiles_to_elements_map=tiles_to_elements_map,
    )


def intersect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[torch.Tensor] = None,
    gaussian_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map projected lidar Gaussians to intersecting tiles on MPS.

    Phase 1 provides the Metal API surface and explicit lidar-state packing.
    Native count/emit kernels land in later phases.
    """

    inputs = _prepare_intersect_tile_lidar_inputs(
        lidar,
        means2d,
        radii,
        depths,
        packed=packed,
        n_images=n_images,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    payload = _pack_lidar_tile_payload(lidar, means2d.device)
    return _make_lazy_metal_func("metal_intersect_tile_lidar")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids.contiguous() if image_ids is not None else None,
        gaussian_ids.contiguous() if gaussian_ids is not None else None,
        inputs.I,
        sort,
        segmented,
        packed,
        payload.n_bins_azimuth,
        payload.n_bins_elevation,
        payload.cdf_resolution_azimuth,
        payload.cdf_resolution_elevation,
        payload.angle_to_pixel_scaling_factor,
        payload.fov_horiz_start,
        payload.fov_horiz_span,
        payload.fov_vert_start,
        payload.fov_vert_span,
        payload.fov_eps,
        payload.spinning_direction,
        payload.cdf_elevation,
        payload.cdf_dense_ray_mask,
        payload.tiles_pack_info,
        payload.tiles_to_elements_map,
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

_UT_CAMERA_MODELS = {"pinhole": 0, "fisheye": 2, "ftheta": 3, "lidar": 4}


def _sparse_coo_grad(indices: torch.Tensor, values: torch.Tensor, size, *, is_coalesced: bool):
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        is_coalesced=is_coalesced,
        check_invariants=False,
    )


def _coerce_projection_2dgs_inputs(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move auxiliary projection inputs onto the means device for Metal execution."""

    if means.device.type != "mps":
        raise ValueError(f"means must be on MPS, got {means.device}")
    if means.dtype != torch.float32:
        raise ValueError(f"means must be float32, got {means.dtype}")

    device = means.device
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    viewmats = viewmats.to(device=device, dtype=torch.float32)
    Ks = Ks.to(device=device, dtype=torch.float32)
    return means, quats, scales, viewmats, Ks


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


class _FullyFusedProjection2DGS(torch.autograd.Function):
    """Autograd bridge for the Metal fused 2DGS projection op."""

    @staticmethod
    def forward(
        ctx,
        means,
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
    ):
        del eps2d  # CUDA keeps this in the Python signature, but the native fused op does not use it.
        radii, means2d, depths, ray_transforms, normals = _make_lazy_metal_func(
            "metal_projection_2dgs_fused_fwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(means, quats, scales, viewmats, Ks, radii, ray_transforms)
        ctx.width = width
        ctx.height = height
        return radii, means2d, depths, ray_transforms, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):
        means, quats, scales, viewmats, Ks, radii, ray_transforms = ctx.saved_tensors
        del v_radii
        v_means, v_quats, v_scales, v_viewmats = _make_lazy_metal_func(
            "metal_projection_2dgs_fused_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_normals.contiguous(),
            v_ray_transforms.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None
        return (
            v_means,
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
        )


class _FullyFusedProjectionPacked2DGS(torch.autograd.Function):
    """Autograd bridge for packed Metal 2DGS projection op."""

    @staticmethod
    def forward(
        ctx,
        means,
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
        sparse_grad,
    ):
        del eps2d
        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms_flat,
            normals,
        ) = _make_lazy_metal_func("metal_projection_2dgs_packed_fwd")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
        )
        ray_transforms = ray_transforms_flat.reshape(-1, 3, 3)
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms_flat,
        )
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad
        return (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        )

    @staticmethod
    def backward(
        ctx,
        v_indptr,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
    ):
        del v_indptr, v_batch_ids, v_camera_ids, v_gaussian_ids, v_radii
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms_flat,
        ) = ctx.saved_tensors
        (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
        ) = _make_lazy_metal_func("metal_projection_2dgs_packed_bwd")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms_flat,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_ray_transforms.contiguous().reshape(-1, 9),
            v_normals.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
            ctx.sparse_grad,
        )

        if ctx.sparse_grad:
            if ctx.needs_input_grad[0]:
                v_means = _sparse_coo_grad(
                    gaussian_ids[None],
                    v_means,
                    means.shape,
                    is_coalesced=len(viewmats) == 1,
                )
            else:
                v_means = None
            if ctx.needs_input_grad[1]:
                v_quats = _sparse_coo_grad(
                    gaussian_ids[None],
                    v_quats,
                    quats.shape,
                    is_coalesced=len(viewmats) == 1,
                )
            else:
                v_quats = None
            if ctx.needs_input_grad[2]:
                v_scales = _sparse_coo_grad(
                    gaussian_ids[None],
                    v_scales,
                    scales.shape,
                    is_coalesced=len(viewmats) == 1,
                )
            else:
                v_scales = None
        else:
            if not ctx.needs_input_grad[0]:
                v_means = None
            if not ctx.needs_input_grad[1]:
                v_quats = None
            if not ctx.needs_input_grad[2]:
                v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # sparse_grad
        )


class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Autograd bridge for packed Metal 3DGS projection."""

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
        sparse_grad,
        calc_compensations,
        camera_model,
        opacities=None,
    ):
        ctx.set_materialize_grads(False)
        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = _make_lazy_metal_func("metal_projection_ewa_3dgs_packed_fwd")(
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
            _CAMERA_MODEL_TO_INT[camera_model],
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars if covars is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            quats if quats is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            scales if scales is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            viewmats,
            Ks,
            conics,
            compensations if compensations is not None else torch.empty(0, device=means.device, dtype=means.dtype),
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad
        ctx.camera_model = camera_model
        ctx.has_covars = covars is not None
        ctx.has_quats = quats is not None
        ctx.has_scales = scales is not None
        ctx.has_compensations = compensations is not None
        return (
            batch_ids,
            camera_ids,
            gaussian_ids,
            indptr,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        )

    @staticmethod
    def backward(
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_indptr,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
    ):
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        covars = covars if ctx.has_covars else None
        quats = quats if ctx.has_quats else None
        scales = scales if ctx.has_scales else None
        compensations = compensations if ctx.has_compensations else None
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_metal_func(
            "metal_projection_ewa_3dgs_packed_bwd"
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
            _CAMERA_MODEL_TO_INT[ctx.camera_model],
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],
            ctx.sparse_grad,
        )

        if ctx.sparse_grad:
            if ctx.needs_input_grad[0]:
                v_means = _sparse_coo_grad(
                    gaussian_ids[None], v_means, means.shape, is_coalesced=len(viewmats) == 1
                )
            else:
                v_means = None
            if ctx.needs_input_grad[1]:
                v_covars = _sparse_coo_grad(
                    gaussian_ids[None], v_covars, covars.shape, is_coalesced=len(viewmats) == 1
                )
            else:
                v_covars = None
            if ctx.needs_input_grad[2]:
                v_quats = _sparse_coo_grad(
                    gaussian_ids[None], v_quats, quats.shape, is_coalesced=len(viewmats) == 1
                )
            else:
                v_quats = None
            if ctx.needs_input_grad[3]:
                v_scales = _sparse_coo_grad(
                    gaussian_ids[None], v_scales, scales.shape, is_coalesced=len(viewmats) == 1
                )
            else:
                v_scales = None
        else:
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
    if packed:
        if sparse_grad and batch_dims != ():
            raise ValueError("sparse_grad does not support batch dimensions when packed=True")
        return _FullyFusedProjectionPacked.apply(
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
            sparse_grad,
            calc_compensations,
            camera_model,
            opacities,
        )
    if sparse_grad:
        raise NotImplementedError("Metal fully_fused_projection does not support sparse_grad")
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


def fully_fused_projection_with_ut(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: Optional[torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    radial_coeffs: Optional[torch.Tensor] = None,
    tangential_coeffs: Optional[torch.Tensor] = None,
    thin_prism_coeffs: Optional[torch.Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[torch.Tensor] = None,
    global_z_order: bool = True,
):
    """Project world-space 3DGS Gaussians with UT support on MPS."""

    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    if means.device.type != "mps":
        raise ValueError(f"means must be on MPS, got {means.device}")
    if means.dtype != torch.float32:
        raise ValueError(f"means must be float32, got {means.dtype}")
    if camera_model not in _UT_CAMERA_MODELS:
        raise ValueError(
            f"camera_model must be one of {tuple(_UT_CAMERA_MODELS)}, got {camera_model}"
        )
    if camera_model not in ("pinhole", "fisheye", "ftheta", "lidar"):
        raise NotImplementedError(
            "Metal UT projection only supports camera_model in {'pinhole', 'fisheye', 'ftheta', 'lidar'}"
        )

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    if means.shape != batch_dims + (N, 3):
        raise ValueError(f"means must have shape {batch_dims + (N, 3)}, got {means.shape}")
    if quats.shape != batch_dims + (N, 4):
        raise ValueError(f"quats must have shape {batch_dims + (N, 4)}, got {quats.shape}")
    if scales.shape != batch_dims + (N, 3):
        raise ValueError(f"scales must have shape {batch_dims + (N, 3)}, got {scales.shape}")
    if viewmats.shape != batch_dims + (C, 4, 4):
        raise ValueError(f"viewmats must have shape {batch_dims + (C, 4, 4)}, got {viewmats.shape}")
    if Ks.shape != batch_dims + (C, 3, 3):
        raise ValueError(f"Ks must have shape {batch_dims + (C, 3, 3)}, got {Ks.shape}")
    if quats.device != means.device or scales.device != means.device:
        raise ValueError("quats and scales must be on the same device as means")
    if viewmats.device != means.device or Ks.device != means.device:
        raise ValueError("viewmats and Ks must be on the same device as means")
    if quats.dtype != torch.float32 or scales.dtype != torch.float32:
        raise ValueError("quats and scales must be float32")
    if viewmats.dtype != torch.float32 or Ks.dtype != torch.float32:
        raise ValueError("viewmats and Ks must be float32")
    if rolling_shutter != RollingShutterType.GLOBAL and viewmats_rs is None:
        raise ValueError("viewmats_rs is required when rolling_shutter is not GLOBAL")
    if viewmats_rs is not None:
        if viewmats_rs.device != means.device or viewmats_rs.dtype != torch.float32:
            raise ValueError("viewmats_rs must be float32 on the same device as means")
        if viewmats_rs.shape != batch_dims + (C, 4, 4):
            raise ValueError(f"viewmats_rs must have shape {batch_dims + (C, 4, 4)}, got {viewmats_rs.shape}")
        viewmats_rs = viewmats_rs.contiguous()
    if opacities is not None:
        if opacities.device != means.device or opacities.dtype != torch.float32:
            raise ValueError("opacities must be float32 on the same device as means")
        if opacities.shape != batch_dims + (N,):
            raise ValueError(f"opacities must have shape {batch_dims + (N,)}, got {opacities.shape}")
        opacities = opacities.contiguous()

    radial_coeffs_prepared = None
    tangential_coeffs_prepared = None
    thin_prism_coeffs_prepared = None
    fisheye_max_angle = None
    ftheta_pixeldist_to_angle_poly = None
    ftheta_angle_to_pixeldist_poly = None
    ftheta_dreference_poly = None
    ftheta_linear_cde = None
    ftheta_max_angle = None
    ftheta_reference_poly = 0
    external_h_poly = None
    external_v_poly = None
    external_h_order = 0
    external_v_order = 0
    lidar_fov_horiz_start = 0.0
    lidar_fov_horiz_span = 0.0
    lidar_fov_vert_start = 0.0
    lidar_fov_vert_span = 0.0
    lidar_fov_eps = 0.0
    lidar_spinning_direction = 0
    pose_start = None
    pose_end = None
    image_dims = batch_dims + (C,)
    if camera_model == "pinhole":
        if lidar_coeffs is not None:
            raise NotImplementedError("Metal pinhole UT projection does not support lidar coefficients")
        if ftheta_coeffs is not None:
            raise NotImplementedError("Metal pinhole UT projection does not support ftheta coefficients")
        if radial_coeffs is not None:
            if radial_coeffs.device != means.device or radial_coeffs.dtype != torch.float32:
                raise ValueError("radial_coeffs must be float32 on the same device as means")
            if radial_coeffs.shape[:-1] != image_dims or radial_coeffs.shape[-1] not in (4, 6):
                raise ValueError(
                    f"radial_coeffs must have shape {image_dims + (4,)} or {image_dims + (6,)}, got {radial_coeffs.shape}"
                )
            if radial_coeffs.shape[-1] == 4:
                radial_coeffs_prepared = torch.nn.functional.pad(radial_coeffs, (0, 2)).contiguous()
            else:
                radial_coeffs_prepared = radial_coeffs.contiguous()
        if tangential_coeffs is not None:
            if tangential_coeffs.device != means.device or tangential_coeffs.dtype != torch.float32:
                raise ValueError("tangential_coeffs must be float32 on the same device as means")
            if tangential_coeffs.shape != image_dims + (2,):
                raise ValueError(
                    f"tangential_coeffs must have shape {image_dims + (2,)}, got {tangential_coeffs.shape}"
                )
            tangential_coeffs_prepared = tangential_coeffs.contiguous()
        if thin_prism_coeffs is not None:
            if thin_prism_coeffs.device != means.device or thin_prism_coeffs.dtype != torch.float32:
                raise ValueError("thin_prism_coeffs must be float32 on the same device as means")
            if thin_prism_coeffs.shape != image_dims + (4,):
                raise ValueError(
                    f"thin_prism_coeffs must have shape {image_dims + (4,)}, got {thin_prism_coeffs.shape}"
                )
            thin_prism_coeffs_prepared = thin_prism_coeffs.contiguous()
    elif camera_model == "fisheye":
        if lidar_coeffs is not None:
            raise NotImplementedError("Metal fisheye UT projection does not support lidar coefficients")
        if tangential_coeffs is not None or thin_prism_coeffs is not None:
            raise NotImplementedError("Metal Phase 2 fisheye UT projection does not support tangential or thin-prism coefficients")
        if radial_coeffs is not None:
            if radial_coeffs.device != means.device or radial_coeffs.dtype != torch.float32:
                raise ValueError("radial_coeffs must be float32 on the same device as means")
            if radial_coeffs.shape != image_dims + (4,):
                raise ValueError(
                    f"radial_coeffs must have shape {image_dims + (4,)}, got {radial_coeffs.shape}"
                )
            radial_coeffs_prepared = radial_coeffs.contiguous()
        from gsplat.cuda._torch_cameras import _BaseCameraModel

        focal_lengths = torch.stack([Ks[..., 0, 0], Ks[..., 1, 1]], dim=-1)
        principal_points = Ks[..., :2, 2]
        fisheye_camera = _BaseCameraModel.create(
            width=width,
            height=height,
            camera_model="fisheye",
            principal_points=principal_points,
            focal_lengths=focal_lengths,
            radial_coeffs=radial_coeffs_prepared,
            rs_type=RollingShutterType.GLOBAL,
        )
        fisheye_max_angle = fisheye_camera.max_angle.contiguous()
    elif camera_model == "ftheta":
        if lidar_coeffs is not None:
            raise NotImplementedError("Metal ftheta UT projection does not support lidar coefficients")
        if radial_coeffs is not None or tangential_coeffs is not None or thin_prism_coeffs is not None:
            raise NotImplementedError("Metal ftheta UT projection does not support radial, tangential, or thin-prism coefficients")
        if ftheta_coeffs is None:
            raise ValueError("ftheta requires ftheta_coeffs")
        from gsplat._camera_types import FThetaPolynomialType

        pixeldist_to_angle_poly = torch.tensor(
            ftheta_coeffs.pixeldist_to_angle_poly,
            device=means.device,
            dtype=means.dtype,
        )
        angle_to_pixeldist_poly = torch.tensor(
            ftheta_coeffs.angle_to_pixeldist_poly,
            device=means.device,
            dtype=means.dtype,
        )
        if ftheta_coeffs.reference_poly == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            dreference_coeffs = torch.tensor(
                [
                    1.0 * ftheta_coeffs.pixeldist_to_angle_poly[1],
                    2.0 * ftheta_coeffs.pixeldist_to_angle_poly[2],
                    3.0 * ftheta_coeffs.pixeldist_to_angle_poly[3],
                    4.0 * ftheta_coeffs.pixeldist_to_angle_poly[4],
                    5.0 * ftheta_coeffs.pixeldist_to_angle_poly[5],
                ],
                device=means.device,
                dtype=means.dtype,
            )
        else:
            dreference_coeffs = torch.tensor(
                [
                    1.0 * ftheta_coeffs.angle_to_pixeldist_poly[1],
                    2.0 * ftheta_coeffs.angle_to_pixeldist_poly[2],
                    3.0 * ftheta_coeffs.angle_to_pixeldist_poly[3],
                    4.0 * ftheta_coeffs.angle_to_pixeldist_poly[4],
                    5.0 * ftheta_coeffs.angle_to_pixeldist_poly[5],
                ],
                device=means.device,
                dtype=means.dtype,
            )
        ftheta_pixeldist_to_angle_poly = pixeldist_to_angle_poly.contiguous()
        ftheta_angle_to_pixeldist_poly = angle_to_pixeldist_poly.contiguous()
        ftheta_dreference_poly = dreference_coeffs.contiguous()
        ftheta_linear_cde = torch.tensor(
            ftheta_coeffs.linear_cde,
            device=means.device,
            dtype=means.dtype,
        ).contiguous()
        ftheta_max_angle = torch.tensor(
            [ftheta_coeffs.max_angle],
            device=means.device,
            dtype=means.dtype,
        ).contiguous()
        ftheta_reference_poly = int(ftheta_coeffs.reference_poly)
    else:
        if lidar_coeffs is None:
            raise ValueError("lidar requires lidar_coeffs")
        if ftheta_coeffs is not None:
            raise NotImplementedError("Metal lidar UT projection does not support ftheta coefficients")
        if radial_coeffs is not None or tangential_coeffs is not None or thin_prism_coeffs is not None:
            raise NotImplementedError("Metal lidar UT projection does not support camera distortion coefficients")
        if external_distortion_coeffs is not None:
            raise NotImplementedError("Metal lidar UT projection does not support external distortion")
        if rolling_shutter != RollingShutterType.GLOBAL or viewmats_rs is not None:
            raise NotImplementedError("Metal lidar UT projection only supports global shutter")
        if width != lidar_coeffs.n_columns or height != lidar_coeffs.n_rows:
            raise ValueError(
                f"lidar width/height must match lidar_coeffs ({lidar_coeffs.n_columns}, {lidar_coeffs.n_rows}), got {(width, height)}"
            )
        if lidar_coeffs.row_elevations_rad.device != means.device or lidar_coeffs.row_elevations_rad.dtype != torch.float32:
            raise ValueError("lidar_coeffs tensors must be float32 on the same device as means")
        if lidar_coeffs.column_azimuths_rad.device != means.device or lidar_coeffs.column_azimuths_rad.dtype != torch.float32:
            raise ValueError("lidar_coeffs tensors must be float32 on the same device as means")
        if lidar_coeffs.row_azimuth_offsets_rad.device != means.device or lidar_coeffs.row_azimuth_offsets_rad.dtype != torch.float32:
            raise ValueError("lidar_coeffs tensors must be float32 on the same device as means")
        lidar_fov_horiz_start = float(lidar_coeffs.fov_horiz_rad.start)
        lidar_fov_horiz_span = float(lidar_coeffs.fov_horiz_rad.span)
        lidar_fov_vert_start = float(lidar_coeffs.fov_vert_rad.start)
        lidar_fov_vert_span = float(lidar_coeffs.fov_vert_rad.span)
        lidar_fov_eps = float(lidar_coeffs.fov_eps_rad)
        lidar_spinning_direction = int(lidar_coeffs.spinning_direction.value)

    if external_distortion_coeffs is not None:
        horizontal_poly = external_distortion_coeffs.horizontal_poly
        vertical_poly = external_distortion_coeffs.vertical_poly
        if horizontal_poly is None or vertical_poly is None:
            raise ValueError("external_distortion_coeffs requires horizontal_poly and vertical_poly")
        external_h_poly = horizontal_poly.to(device=means.device, dtype=means.dtype).contiguous()
        external_v_poly = vertical_poly.to(device=means.device, dtype=means.dtype).contiguous()
        external_h_order = int((math.isqrt(1 + 8 * external_h_poly.numel()) - 3) // 2)
        external_v_order = int((math.isqrt(1 + 8 * external_v_poly.numel()) - 3) // 2)
        if (external_h_order + 1) * (external_h_order + 2) // 2 != external_h_poly.numel():
            raise ValueError("external_distortion horizontal_poly has invalid triangular coefficient count")
        if (external_v_order + 1) * (external_v_order + 2) // 2 != external_v_poly.numel():
            raise ValueError("external_distortion vertical_poly has invalid triangular coefficient count")

    pose_start = viewmat_to_pose(viewmats).contiguous()
    pose_end = viewmat_to_pose(viewmats_rs).contiguous() if viewmats_rs is not None else None

    radii, means2d, depths, conics, compensations = _make_lazy_metal_func(
        "metal_projection_ut_3dgs_fused"
    )(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        opacities,
        viewmats.contiguous(),
        viewmats_rs,
        pose_start,
        pose_end,
        Ks.contiguous(),
        radial_coeffs_prepared,
        tangential_coeffs_prepared,
        thin_prism_coeffs_prepared,
        fisheye_max_angle,
        ftheta_pixeldist_to_angle_poly,
        ftheta_angle_to_pixeldist_poly,
        ftheta_dreference_poly,
        ftheta_linear_cde,
        ftheta_max_angle,
        external_h_poly,
        external_v_poly,
        lidar_fov_horiz_start,
        lidar_fov_horiz_span,
        lidar_fov_vert_start,
        lidar_fov_vert_span,
        lidar_fov_eps,
        lidar_spinning_direction,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        _UT_CAMERA_MODELS[camera_model],
        global_z_order,
        ut_params.alpha,
        ut_params.beta,
        ut_params.kappa,
        ut_params.in_image_margin_factor,
        ut_params.require_all_sigma_points_valid,
        int(rolling_shutter),
        ftheta_reference_poly,
        external_h_order,
        external_v_order,
    )
    if not calc_compensations:
        compensations = None
    return radii, means2d, depths, conics, compensations


def fully_fused_projection_2dgs(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
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
):
    """Project world-space 2DGS Gaussians to screen-space on MPS."""

    means, quats, scales, viewmats, Ks = _coerce_projection_2dgs_inputs(
        means, quats, scales, viewmats, Ks
    )
    batch_dims = means.shape[:-2]
    if packed:
        if sparse_grad and batch_dims != ():
            raise ValueError("sparse_grad does not support batch dimensions when packed=True")
        return _FullyFusedProjectionPacked2DGS.apply(
            means.contiguous(),
            quats.contiguous(),
            scales.contiguous(),
            viewmats.contiguous(),
            Ks.contiguous(),
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
        )
    if sparse_grad:
        raise ValueError("sparse_grad is only supported when packed=True")
    N = means.shape[-2]
    C = viewmats.shape[-3]
    if means.shape != batch_dims + (N, 3):
        raise ValueError(f"means must have shape [..., N, 3], got {means.shape}")
    if quats.shape != batch_dims + (N, 4):
        raise ValueError(f"quats must have shape {batch_dims + (N, 4)}, got {quats.shape}")
    if scales.shape != batch_dims + (N, 3):
        raise ValueError(f"scales must have shape {batch_dims + (N, 3)}, got {scales.shape}")
    if viewmats.shape != batch_dims + (C, 4, 4):
        raise ValueError(
            f"viewmats must have shape {batch_dims + (C, 4, 4)}, got {viewmats.shape}"
        )
    if Ks.shape != batch_dims + (C, 3, 3):
        raise ValueError(f"Ks must have shape {batch_dims + (C, 3, 3)}, got {Ks.shape}")

    return _FullyFusedProjection2DGS.apply(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        viewmats.contiguous(),
        Ks.contiguous(),
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
    )


def _rasterize_to_pixels_2dgs_reference_autograd(
    means2d: torch.Tensor,
    ray_transforms: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    normals: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_dims = tuple(isect_offsets.shape[:-2])
    I = math.prod(image_dims)
    channels = colors.shape[-1]
    tile_height, tile_width = isect_offsets.shape[-2:]
    n_isects = int(flatten_ids.numel())
    device = means2d.device
    dtype = means2d.dtype

    render_colors = torch.zeros(*image_dims, image_height, image_width, channels, dtype=dtype, device=device)
    render_alphas = torch.zeros(*image_dims, image_height, image_width, 1, dtype=dtype, device=device)
    render_normals = torch.zeros(*image_dims, image_height, image_width, 3, dtype=dtype, device=device)
    render_distort = torch.zeros(*image_dims, image_height, image_width, 1, dtype=dtype, device=device)
    if backgrounds is not None:
        render_colors = render_colors + backgrounds.unsqueeze(-2).unsqueeze(-2)

    offsets_flat = isect_offsets.reshape(I, tile_height * tile_width)
    means_flat = means2d.reshape(-1, 2)
    ray_flat = ray_transforms.reshape(-1, 3, 3)
    colors_flat = colors.reshape(-1, channels)
    opacities_flat = opacities.reshape(-1)
    normals_flat = normals.reshape(-1, 3)
    alpha_threshold = torch.tensor(1.0 / 255.0, dtype=dtype, device=device)
    max_alpha = torch.tensor(0.99, dtype=dtype, device=device)
    trans_thresh = 1.0e-4
    filter_inv_square = torch.tensor(2.0, dtype=dtype, device=device)

    for image_id in range(I):
        for tile_y in range(tile_height):
            for tile_x in range(tile_width):
                tile_id = tile_y * tile_width + tile_x
                global_tile = image_id * tile_height * tile_width + tile_id
                range_start = int(offsets_flat[image_id, tile_id].item())
                if global_tile + 1 < I * tile_height * tile_width:
                    next_image = (global_tile + 1) // (tile_height * tile_width)
                    next_tile = (global_tile + 1) % (tile_height * tile_width)
                    range_end = int(offsets_flat[next_image, next_tile].item())
                else:
                    range_end = n_isects

                masked = masks is not None and not bool(
                    masks.reshape(I, tile_height, tile_width)[image_id, tile_y, tile_x].item()
                )
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        i = tile_y * tile_size + local_y
                        j = tile_x * tile_size + local_x
                        if i >= image_height or j >= image_width or masked:
                            continue

                        px = torch.tensor(float(j) + 0.5, dtype=dtype, device=device)
                        py = torch.tensor(float(i) + 0.5, dtype=dtype, device=device)
                        T = torch.tensor(1.0, dtype=dtype, device=device)
                        accum = torch.zeros(channels, dtype=dtype, device=device)
                        accum_normal = torch.zeros(3, dtype=dtype, device=device)
                        distort = torch.tensor(0.0, dtype=dtype, device=device)
                        accum_vis_depth = torch.tensor(0.0, dtype=dtype, device=device)
                        for idx in range(range_start, range_end):
                            g = int(flatten_ids[idx].item())
                            u_M = ray_flat[g, 0]
                            v_M = ray_flat[g, 1]
                            w_M = ray_flat[g, 2]
                            h_u = px * w_M - u_M
                            h_v = py * w_M - v_M
                            ray_cross = torch.cross(h_u, h_v, dim=-1)
                            if float(ray_cross[2].detach().item()) == 0.0:
                                continue
                            s = ray_cross[:2] / ray_cross[2]
                            gauss_weight_3d = (s * s).sum()
                            d = means_flat[g] - torch.stack([px, py])
                            gauss_weight_2d = filter_inv_square * (d * d).sum()
                            sigma = 0.5 * torch.minimum(gauss_weight_3d, gauss_weight_2d)
                            alpha = torch.minimum(max_alpha, opacities_flat[g] * torch.exp(-sigma))
                            if float(sigma.detach().item()) < 0.0 or float(alpha.detach().item()) < float(alpha_threshold.item()):
                                continue
                            next_T = T * (1.0 - alpha)
                            if float(next_T.detach().item()) <= trans_thresh:
                                break

                            vis = alpha * T
                            accum = accum + colors_flat[g] * vis
                            accum_normal = accum_normal + normals_flat[g] * vis
                            depth = colors_flat[g, -1]
                            distort = distort + 2.0 * (vis * depth * (1.0 - T) - vis * accum_vis_depth)
                            accum_vis_depth = accum_vis_depth + vis * depth
                            T = next_T

                        if backgrounds is not None:
                            accum = accum + backgrounds.reshape(I, channels)[image_id] * T
                        render_colors.reshape(I, image_height, image_width, channels)[image_id, i, j] = accum
                        render_alphas.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = 1.0 - T
                        render_normals.reshape(I, image_height, image_width, 3)[image_id, i, j] = accum_normal
                        render_distort.reshape(I, image_height, image_width, 1)[image_id, i, j, 0] = distort

    return render_colors, render_alphas, render_normals, render_distort


def _rasterize_to_pixels_2dgs_absgrad_reference(
    means2d: torch.Tensor,
    ray_transforms: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    normals: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    v_render_colors: torch.Tensor,
    v_render_alphas: torch.Tensor,
    v_render_normals: torch.Tensor,
    v_render_distort: torch.Tensor,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
    ) = _rasterize_to_pixels_2dgs_reference_autograd(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
    )
    absgrad = torch.zeros_like(means2d)
    image_dims = render_alphas.shape[:-3]
    shape = (*image_dims, image_height, image_width)
    for flat_idx in range(math.prod(shape)):
        unravel = []
        rem = flat_idx
        for size in reversed(shape):
            unravel.append(rem % size)
            rem //= size
        index_tuple = tuple(reversed(unravel))
        scalar = (
            (render_colors[index_tuple] * v_render_colors[index_tuple]).sum()
            + (render_alphas[index_tuple + (0,)] * v_render_alphas[index_tuple + (0,)]).sum()
            + (render_normals[index_tuple] * v_render_normals[index_tuple]).sum()
            + (render_distort[index_tuple + (0,)] * v_render_distort[index_tuple + (0,)]).sum()
        )
        pixel_grad = torch.autograd.grad(scalar, means2d, retain_graph=True)[0]
        absgrad = absgrad + pixel_grad.abs()
    return absgrad


def _rasterize_to_pixels_2dgs_median_color_grad(
    v_render_median: Optional[torch.Tensor],
    median_ids: torch.Tensor,
    flatten_ids: torch.Tensor,
    colors_shape: torch.Size,
) -> torch.Tensor:
    if v_render_median is None:
        return torch.zeros(colors_shape, device=median_ids.device, dtype=torch.float32)
    grad = torch.zeros(colors_shape, device=v_render_median.device, dtype=v_render_median.dtype)
    grad_flat = grad.reshape(-1, colors_shape[-1])
    median_vals = v_render_median.reshape(-1)
    median_ids_flat = median_ids.reshape(-1)
    for pixel_idx in range(median_ids_flat.numel()):
        isect_idx = int(median_ids_flat[pixel_idx].item())
        if isect_idx < 0 or isect_idx >= flatten_ids.numel():
            continue
        g = int(flatten_ids[isect_idx].item())
        grad_flat[g, -1] += median_vals[pixel_idx]
    return grad


class _RasterizeToPixels2DGS(torch.autograd.Function):
    """Autograd bridge for the Metal 2DGS rasterization op.

    Forward and backward both use native Metal ops.
    """

    @staticmethod
    def forward(
        ctx,
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        packed,
        absgrad,
        distloss,
    ):
        ctx.set_materialize_grads(False)
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
        ) = _make_lazy_metal_func("metal_rasterize_to_pixels_2dgs_fwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            packed,
        )
        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds if backgrounds is not None else torch.empty(0, device=means2d.device, dtype=means2d.dtype),
            masks if masks is not None else torch.empty(0, device=means2d.device, dtype=torch.bool),
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.packed = packed
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.has_backgrounds = backgrounds is not None
        ctx.has_masks = masks is not None
        return render_colors, render_alphas, render_normals, render_distort, render_median

    @staticmethod
    def backward(
        ctx,
        v_render_colors,
        v_render_alphas,
        v_render_normals,
        v_render_distort,
        v_render_median,
    ):
        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        backgrounds = backgrounds if ctx.has_backgrounds else None
        masks = masks if ctx.has_masks else None

        if v_render_colors is None:
            v_render_colors = torch.zeros_like(render_colors)
        if v_render_alphas is None:
            v_render_alphas = torch.zeros_like(render_alphas)
        if v_render_normals is None:
            v_render_normals = torch.zeros(
                *render_alphas.shape[:-1], 3, device=render_alphas.device, dtype=render_alphas.dtype
            )
        if v_render_distort is None:
            v_render_distort = torch.zeros_like(render_alphas)
        if v_render_median is None:
            v_render_median = torch.zeros_like(render_alphas)

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_metal_func("metal_rasterize_to_pixels_2dgs_bwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            ctx.packed,
            ctx.absgrad,
        )

        if ctx.absgrad and v_means2d_abs is not None:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas)).sum(dim=(-3, -2))
        else:
            v_backgrounds = None
        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
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


class _RasterizeToPixels(torch.autograd.Function):
    """Autograd bridge for the Metal 3DGS rasterization op."""

    @staticmethod
    def forward(
        ctx,
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        absgrad,
    ):
        ctx.set_materialize_grads(False)
        render_colors, render_alphas, last_ids = _make_lazy_metal_func(
            "metal_rasterize_to_pixels_3dgs_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )
        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds
            if backgrounds is not None
            else torch.empty(0, device=means2d.device, dtype=means2d.dtype),
            masks
            if masks is not None
            else torch.empty(0, device=means2d.device, dtype=torch.bool),
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.has_backgrounds = backgrounds is not None
        ctx.has_masks = masks is not None
        return render_colors, render_alphas

    @staticmethod
    def backward(ctx, v_render_colors, v_render_alphas):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        backgrounds = backgrounds if ctx.has_backgrounds else None
        masks = masks if ctx.has_masks else None

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_metal_func("metal_rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            ctx.absgrad,
        )

        if ctx.absgrad and v_means2d_abs is not None:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas)).sum(dim=(-3, -2))
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPixelsEval3D(torch.autograd.Function):
    """Autograd bridge for the Metal explicit-rays world-space 3DGS rasterizer."""

    @staticmethod
    def forward(
        ctx,
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        camera_model,
        ut_params,
        rays,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        rolling_shutter,
        viewmats_rs,
        return_sample_counts,
        use_hit_distance,
        return_normals,
    ):
        ctx.set_materialize_grads(False)

        batch_dims = means.shape[:-2]
        c = colors.shape[-3]
        sample_counts = None
        render_normals = None
        if return_sample_counts:
            sample_counts = torch.empty(
                batch_dims + (c, height, width), dtype=torch.int32, device=means.device
            )
        if return_normals:
            render_normals = torch.empty(
                batch_dims + (c, height, width, 3),
                dtype=means.dtype,
                device=means.device,
            )

        render_colors, render_alphas, last_ids = _make_lazy_metal_func(
            "metal_rasterize_to_pixels_from_world_3dgs_fwd"
        )(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            viewmats,
            Ks,
            rays,
            isect_offsets,
            flatten_ids,
            sample_counts,
            render_normals,
            use_hit_distance,
        )
        ctx.save_for_backward(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds
            if backgrounds is not None
            else torch.empty(0, device=means.device, dtype=means.dtype),
            masks
            if masks is not None
            else torch.empty(0, device=means.device, dtype=torch.bool),
            viewmats,
            Ks,
            rays if rays is not None else torch.empty(0, device=means.device, dtype=means.dtype),
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.has_backgrounds = backgrounds is not None
        ctx.has_masks = masks is not None
        ctx.has_rays = rays is not None
        ctx.return_normals = return_normals
        ctx.use_hit_distance = use_hit_distance
        return render_colors, render_alphas, last_ids, sample_counts, render_normals

    @staticmethod
    def backward(
        ctx,
        v_render_colors,
        v_render_alphas,
        v_last_ids,
        v_sample_counts,
        v_render_normals,
    ):
        del v_last_ids, v_sample_counts
        (
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            viewmats,
            Ks,
            rays,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        backgrounds = backgrounds if ctx.has_backgrounds else None
        masks = masks if ctx.has_masks else None
        rays = rays if ctx.has_rays else None

        if v_render_colors is None:
            v_render_colors = torch.zeros(
                render_alphas.shape[:-1] + (colors.shape[-1],),
                device=render_alphas.device,
                dtype=render_alphas.dtype,
            )
        if v_render_alphas is None:
            v_render_alphas = torch.zeros_like(render_alphas)

        (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_rays,
        ) = _make_lazy_metal_func("metal_rasterize_to_pixels_from_world_3dgs_bwd")(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            viewmats,
            Ks,
            rays,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous() if v_render_normals is not None else None,
            ctx.use_hit_distance,
        )
        if not ctx.has_rays:
            v_rays = None

        if ctx.needs_input_grad[5]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas)).sum(dim=(-3, -2))
        else:
            v_backgrounds = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_backgrounds,
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
            v_rays,
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
            None,
        )


def rasterize_to_pixels_eval3d(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[torch.Tensor] = None,
    radial_coeffs: Optional[torch.Tensor] = None,
    tangential_coeffs: Optional[torch.Tensor] = None,
    thin_prism_coeffs: Optional[torch.Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[torch.Tensor] = None,
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
):
    """Rasterize 3D Gaussians in world space on MPS using explicit per-pixel rays.

    Current Metal support is intentionally narrow:
    - unpacked inputs only
    - supports explicit `rays` or wrapper-synthesized world rays
    - supports pinhole, fisheye, and ftheta ray synthesis
    - supports OpenCV pinhole/fisheye distortion coefficients
    - supports rolling shutter via `viewmats_rs`
    - supports external distortion
    - no lidar
    - rendered normals supported
    - hit-distance mode supported
    """

    if means.device.type != "mps":
        raise ValueError(f"means must be on MPS, got {means.device}")
    if camera_model == "ortho":
        raise NotImplementedError("Metal currently does not support the ortho camera model")
    if camera_model == "lidar" or lidar_coeffs is not None:
        raise NotImplementedError("Metal currently does not support lidar camera model")

    batch_dims = means.shape[:-2]
    n = means.shape[-2]
    c = viewmats.shape[-3]
    channels = colors.shape[-1]
    image_dims = batch_dims + (c,)

    if means.shape != batch_dims + (n, 3):
        raise ValueError(f"means must have shape {batch_dims + (n, 3)}, got {means.shape}")
    if quats.shape != batch_dims + (n, 4):
        raise ValueError(f"quats must have shape {batch_dims + (n, 4)}, got {quats.shape}")
    if scales.shape != batch_dims + (n, 3):
        raise ValueError(f"scales must have shape {batch_dims + (n, 3)}, got {scales.shape}")
    if colors.shape != image_dims + (n, channels):
        raise ValueError(f"colors must have shape {image_dims + (n, channels)}, got {colors.shape}")
    if opacities.shape != image_dims + (n,):
        raise ValueError(f"opacities must have shape {image_dims + (n,)}, got {opacities.shape}")
    if viewmats.shape != image_dims + (4, 4):
        raise ValueError(f"viewmats must have shape {image_dims + (4, 4)}, got {viewmats.shape}")
    if Ks.shape != image_dims + (3, 3):
        raise ValueError(f"Ks must have shape {image_dims + (3, 3)}, got {Ks.shape}")
    if rays is not None and rays.shape != image_dims + (height, width, 6):
        raise ValueError(f"rays must have shape {image_dims + (height, width, 6)}, got {rays.shape}")
    if isect_offsets.shape[:-2] != image_dims:
        raise ValueError(
            f"isect_offsets image dims must match {image_dims}, got {isect_offsets.shape[:-2]}"
        )
    if means.dtype != torch.float32 or quats.dtype != torch.float32 or scales.dtype != torch.float32:
        raise ValueError("means, quats, and scales must be float32")
    if colors.dtype != torch.float32 or opacities.dtype != torch.float32:
        raise ValueError("colors and opacities must be float32")
    if viewmats.dtype != torch.float32 or Ks.dtype != torch.float32:
        raise ValueError("viewmats and Ks must be float32")
    if rays is not None and rays.dtype != torch.float32:
        raise ValueError("rays must be float32")
    if isect_offsets.device != means.device or flatten_ids.device != means.device:
        raise ValueError("isect_offsets and flatten_ids must be on the same device as means")
    if isect_offsets.dtype != torch.int32 or flatten_ids.dtype != torch.int32:
        raise ValueError("isect_offsets and flatten_ids must be int32")
    if rays is not None and rays.device != means.device:
        raise ValueError("rays must be on the same device as means")
    if backgrounds is not None:
        if backgrounds.device != means.device or backgrounds.dtype != torch.float32:
            raise ValueError("backgrounds must be float32 on the same device as means")
        if backgrounds.shape != image_dims + (channels,):
            raise ValueError(
                f"backgrounds must have shape {image_dims + (channels,)}, got {backgrounds.shape}"
            )
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        if masks.device != means.device or masks.dtype != torch.bool:
            raise ValueError("masks must be bool on the same device as means")
        if masks.shape != isect_offsets.shape:
            raise ValueError(f"masks must have shape {isect_offsets.shape}, got {masks.shape}")
        masks = masks.contiguous()
    if radial_coeffs is not None:
        expected_last = 4 if camera_model == "fisheye" else 6
        if radial_coeffs.device != means.device or radial_coeffs.dtype != torch.float32:
            raise ValueError("radial_coeffs must be float32 on the same device as means")
        if radial_coeffs.shape != image_dims + (expected_last,):
            raise ValueError(
                f"radial_coeffs must have shape {image_dims + (expected_last,)}, got {radial_coeffs.shape}"
            )
        radial_coeffs = radial_coeffs.contiguous()
    if tangential_coeffs is not None:
        if tangential_coeffs.device != means.device or tangential_coeffs.dtype != torch.float32:
            raise ValueError("tangential_coeffs must be float32 on the same device as means")
        if tangential_coeffs.shape != image_dims + (2,):
            raise ValueError(
                f"tangential_coeffs must have shape {image_dims + (2,)}, got {tangential_coeffs.shape}"
            )
        tangential_coeffs = tangential_coeffs.contiguous()
    if thin_prism_coeffs is not None:
        if thin_prism_coeffs.device != means.device or thin_prism_coeffs.dtype != torch.float32:
            raise ValueError("thin_prism_coeffs must be float32 on the same device as means")
        if thin_prism_coeffs.shape != image_dims + (4,):
            raise ValueError(
                f"thin_prism_coeffs must have shape {image_dims + (4,)}, got {thin_prism_coeffs.shape}"
            )
        thin_prism_coeffs = thin_prism_coeffs.contiguous()
    if viewmats_rs is not None:
        if viewmats_rs.device != means.device or viewmats_rs.dtype != torch.float32:
            raise ValueError("viewmats_rs must be float32 on the same device as means")
        if viewmats_rs.shape != image_dims + (4, 4):
            raise ValueError(f"viewmats_rs must have shape {image_dims + (4, 4)}, got {viewmats_rs.shape}")
        viewmats_rs = viewmats_rs.contiguous()
    if rays is not None and rolling_shutter != RollingShutterType.GLOBAL:
        raise NotImplementedError(
            "Metal rasterize_to_pixels_eval3d does not support rolling_shutter with explicit rays"
        )
    if rolling_shutter != RollingShutterType.GLOBAL and viewmats_rs is None:
        raise ValueError("viewmats_rs is required when rolling_shutter is not GLOBAL")

    if rays is None:
        rays = _synthesize_eval3d_world_rays(
            viewmats=viewmats.contiguous(),
            Ks=Ks.contiguous(),
            width=width,
            height=height,
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            external_distortion_coeffs=external_distortion_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
        )

    if channels > 513 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.zeros(*colors.shape[:-1], padded_channels, device=means.device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(*backgrounds.shape[:-1], padded_channels, device=means.device),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    if tile_height * tile_size < height:
        raise ValueError(
            f"tile_height * tile_size must cover image_height, got {tile_height} * {tile_size} < {height}"
        )
    if tile_width * tile_size < width:
        raise ValueError(
            f"tile_width * tile_size must cover image_width, got {tile_width} * {tile_size} < {width}"
        )

    render_colors, render_alphas, last_ids, sample_counts, render_normals = (
        _RasterizeToPixelsEval3D.apply(
            means.contiguous(),
            quats.contiguous(),
            scales.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            backgrounds,
            masks,
            viewmats.contiguous(),
            Ks.contiguous(),
            width,
            height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            camera_model,
            ut_params,
            rays.contiguous(),
            radial_coeffs.contiguous() if radial_coeffs is not None else None,
            tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
            thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
            ftheta_coeffs,
            lidar_coeffs,
            external_distortion_coeffs,
            rolling_shutter,
            viewmats_rs,
            return_sample_counts,
            use_hit_distance,
            return_normals,
        )
    )
    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )
    return render_colors, render_alphas, last_ids, sample_counts, render_normals


def rasterize_to_pixels(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize projected 3DGS Gaussians to pixels on MPS."""

    image_dims = isect_offsets.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {device}")
    if packed:
        nnz = means2d.size(0)
        if means2d.shape != (nnz, 2):
            raise ValueError(f"packed means2d must have shape (nnz, 2), got {means2d.shape}")
        if conics.shape != (nnz, 3):
            raise ValueError(f"packed conics must have shape (nnz, 3), got {conics.shape}")
        if colors.shape[0] != nnz:
            raise ValueError(f"packed colors must have nnz leading dim, got {colors.shape}")
        if opacities.shape != (nnz,):
            raise ValueError(f"packed opacities must have shape (nnz,), got {opacities.shape}")
    else:
        means_image_dims = means2d.shape[:-2]
        N = means2d.size(-2)
        if means2d.shape != means_image_dims + (N, 2):
            raise ValueError(f"means2d must have shape {means_image_dims + (N, 2)}, got {means2d.shape}")
        if means_image_dims != image_dims:
            raise ValueError(
                f"means2d image dims must match isect_offsets image dims, got {means_image_dims} vs {image_dims}"
            )
        if conics.shape != means_image_dims + (N, 3):
            raise ValueError(f"conics must have shape {means_image_dims + (N, 3)}, got {conics.shape}")
        if colors.shape != means_image_dims + (N, channels):
            raise ValueError(f"colors must have shape {means_image_dims + (N, channels)}, got {colors.shape}")
        if opacities.shape != means_image_dims + (N,):
            raise ValueError(f"opacities must have shape {means_image_dims + (N,)}, got {opacities.shape}")
    if means2d.dtype != torch.float32 or conics.dtype != torch.float32:
        raise ValueError("means2d and conics must be float32")
    if colors.dtype != torch.float32 or opacities.dtype != torch.float32:
        raise ValueError("colors and opacities must be float32")
    if isect_offsets.device != device or flatten_ids.device != device:
        raise ValueError("isect_offsets and flatten_ids must be on the same device as means2d")
    if isect_offsets.dtype != torch.int32:
        raise ValueError(f"isect_offsets must be int32, got {isect_offsets.dtype}")
    if flatten_ids.dtype != torch.int32:
        raise ValueError(f"flatten_ids must be int32, got {flatten_ids.dtype}")
    if backgrounds is not None:
        if backgrounds.device != device or backgrounds.dtype != torch.float32:
            raise ValueError("backgrounds must be float32 on the same device as means2d")
        if backgrounds.shape != image_dims + (channels,):
            raise ValueError(f"backgrounds must have shape {image_dims + (channels,)}, got {backgrounds.shape}")
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        if masks.device != device or masks.dtype != torch.bool:
            raise ValueError("masks must be bool on the same device as means2d")
        if masks.shape != isect_offsets.shape:
            raise ValueError(f"masks must have shape {isect_offsets.shape}, got {masks.shape}")
        masks = masks.contiguous()

    if channels > 513 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.zeros(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(*backgrounds.shape[:-1], padded_channels, device=device),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    if tile_height * tile_size < image_height:
        raise ValueError(
            f"tile_height * tile_size must cover image_height, got {tile_height} * {tile_size} < {image_height}"
        )
    if tile_width * tile_size < image_width:
        raise ValueError(
            f"tile_width * tile_size must cover image_width, got {tile_width} * {tile_size} < {image_width}"
        )

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )
    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


def rasterize_to_pixels_2dgs(
    means2d: torch.Tensor,
    ray_transforms: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    normals: torch.Tensor,
    densify: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize projected 2DGS Gaussians to pixels on MPS."""
    channels = colors.shape[-1]
    device = means2d.device
    if device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {device}")

    image_dims = isect_offsets.shape[:-2]
    if packed:
        nnz = means2d.size(0)
        if means2d.shape != (nnz, 2):
            raise ValueError(f"means2d must have shape {(nnz, 2)}, got {means2d.shape}")
        if ray_transforms.shape != (nnz, 3, 3):
            raise ValueError(
                f"ray_transforms must have shape {(nnz, 3, 3)}, got {ray_transforms.shape}"
            )
        if colors.shape != (nnz, channels):
            raise ValueError(f"colors must have shape {(nnz, channels)}, got {colors.shape}")
        if opacities.shape != (nnz,):
            raise ValueError(f"opacities must have shape {(nnz,)}, got {opacities.shape}")
        if normals.shape != (nnz, 3):
            raise ValueError(f"normals must have shape {(nnz, 3)}, got {normals.shape}")
        if densify.shape != (nnz, 2):
            raise ValueError(f"densify must have shape {(nnz, 2)}, got {densify.shape}")
    else:
        N = means2d.size(-2)
        if means2d.shape != image_dims + (N, 2):
            raise ValueError(f"means2d must have shape {image_dims + (N, 2)}, got {means2d.shape}")
        if ray_transforms.shape != image_dims + (N, 3, 3):
            raise ValueError(
                f"ray_transforms must have shape {image_dims + (N, 3, 3)}, got {ray_transforms.shape}"
            )
        if colors.shape != image_dims + (N, channels):
            raise ValueError(f"colors must have shape {image_dims + (N, channels)}, got {colors.shape}")
        if opacities.shape != image_dims + (N,):
            raise ValueError(f"opacities must have shape {image_dims + (N,)}, got {opacities.shape}")
        if normals.shape != image_dims + (N, 3):
            raise ValueError(f"normals must have shape {image_dims + (N, 3)}, got {normals.shape}")
        if densify.shape != image_dims + (N, 2):
            raise ValueError(f"densify must have shape {image_dims + (N, 2)}, got {densify.shape}")
    if means2d.dtype != torch.float32 or ray_transforms.dtype != torch.float32:
        raise ValueError("means2d and ray_transforms must be float32")
    if colors.dtype != torch.float32 or opacities.dtype != torch.float32:
        raise ValueError("colors and opacities must be float32")
    if normals.dtype != torch.float32:
        raise ValueError(f"normals must be float32, got {normals.dtype}")
    if densify.dtype != torch.float32:
        raise ValueError(f"densify must be float32, got {densify.dtype}")

    tile_height, tile_width = isect_offsets.shape[-2:]
    if isect_offsets.device != device or flatten_ids.device != device:
        raise ValueError("isect_offsets and flatten_ids must be on the same device as means2d")
    if isect_offsets.dtype != torch.int32:
        raise ValueError(f"isect_offsets must be int32, got {isect_offsets.dtype}")
    if flatten_ids.dtype != torch.int32:
        raise ValueError(f"flatten_ids must be int32, got {flatten_ids.dtype}")

    if backgrounds is not None:
        if backgrounds.device != device or backgrounds.dtype != torch.float32:
            raise ValueError("backgrounds must be float32 on the same device as means2d")
        if backgrounds.shape != image_dims + (channels,):
            raise ValueError(f"backgrounds must have shape {image_dims + (channels,)}, got {backgrounds.shape}")
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        if masks.device != device or masks.dtype != torch.bool:
            raise ValueError("masks must be bool on the same device as means2d")
        if masks.shape != isect_offsets.shape:
            raise ValueError(f"masks must have shape {isect_offsets.shape}, got {masks.shape}")
        masks = masks.contiguous()

    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        # Keep the final depth-like channel at the end after padding.
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.zeros(*colors.shape[:-1], padded_channels, device=device, dtype=colors.dtype),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds[..., :-1],
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device, dtype=backgrounds.dtype
                    ),
                    backgrounds[..., -1:],
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    if tile_height * tile_size < image_height:
        raise ValueError(
            f"tile_height * tile_size must cover image_height, got {tile_height} * {tile_size} < {image_height}"
        )
    if tile_width * tile_size < image_width:
        raise ValueError(
            f"tile_width * tile_size must cover image_width, got {tile_width} * {tile_size} < {image_width}"
        )

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = _RasterizeToPixels2DGS.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        packed,
        absgrad,
        distloss,
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, render_normals, render_distort, render_median


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize a Gaussian batch range and return only hit indices on MPS."""

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]

    if means2d.device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {means2d.device}")
    if transmittances.shape != image_dims + (image_height, image_width):
        raise ValueError(
            f"transmittances must have shape {image_dims + (image_height, image_width)}, "
            f"got {transmittances.shape}"
        )
    if means2d.shape != image_dims + (N, 2):
        raise ValueError(f"means2d must have shape {image_dims + (N, 2)}, got {means2d.shape}")
    if conics.shape != image_dims + (N, 3):
        raise ValueError(f"conics must have shape {image_dims + (N, 3)}, got {conics.shape}")
    if opacities.shape != image_dims + (N,):
        raise ValueError(f"opacities must have shape {image_dims + (N,)}, got {opacities.shape}")
    if isect_offsets.shape != image_dims + (tile_height, tile_width):
        raise ValueError(
            f"isect_offsets must have shape {image_dims + (tile_height, tile_width)}, "
            f"got {isect_offsets.shape}"
        )
    if transmittances.device != means2d.device:
        raise ValueError("transmittances must be on the same device as means2d")
    if conics.device != means2d.device or opacities.device != means2d.device:
        raise ValueError("conics and opacities must be on the same device as means2d")
    if isect_offsets.device != means2d.device or flatten_ids.device != means2d.device:
        raise ValueError("isect_offsets and flatten_ids must be on the same device as means2d")
    if transmittances.dtype != torch.float32:
        raise ValueError(f"transmittances must be float32, got {transmittances.dtype}")
    if means2d.dtype != torch.float32 or conics.dtype != torch.float32:
        raise ValueError("means2d and conics must be float32")
    if opacities.dtype != torch.float32:
        raise ValueError(f"opacities must be float32, got {opacities.dtype}")
    if isect_offsets.dtype != torch.int32:
        raise ValueError(f"isect_offsets must be int32, got {isect_offsets.dtype}")
    if flatten_ids.dtype != torch.int32:
        raise ValueError(f"flatten_ids must be int32, got {flatten_ids.dtype}")
    if tile_height * tile_size < image_height:
        raise ValueError(
            f"tile_height * tile_size must cover image_height, got "
            f"{tile_height} * {tile_size} < {image_height}"
        )
    if tile_width * tile_size < image_width:
        raise ValueError(
            f"tile_width * tile_size must cover image_width, got "
            f"{tile_width} * {tile_size} < {image_width}"
        )

    out_gauss_ids, out_indices = _make_lazy_metal_func("metal_rasterize_to_indices_3dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    pixels_per_image = image_width * image_height
    out_pixel_ids = out_indices % pixels_per_image
    out_image_ids = out_indices // pixels_per_image
    return out_gauss_ids, out_pixel_ids, out_image_ids


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: torch.Tensor,
    means2d: torch.Tensor,
    ray_transforms: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize a 2DGS batch range and return only hit indices on MPS."""

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]

    if means2d.device.type != "mps":
        raise ValueError(f"means2d must be on MPS, got {means2d.device}")
    if transmittances.shape != image_dims + (image_height, image_width):
        raise ValueError(
            f"transmittances must have shape {image_dims + (image_height, image_width)}, "
            f"got {transmittances.shape}"
        )
    if means2d.shape != image_dims + (N, 2):
        raise ValueError(f"means2d must have shape {image_dims + (N, 2)}, got {means2d.shape}")
    if ray_transforms.shape != image_dims + (N, 3, 3):
        raise ValueError(
            f"ray_transforms must have shape {image_dims + (N, 3, 3)}, got {ray_transforms.shape}"
        )
    if opacities.shape != image_dims + (N,):
        raise ValueError(f"opacities must have shape {image_dims + (N,)}, got {opacities.shape}")
    if isect_offsets.shape != image_dims + (tile_height, tile_width):
        raise ValueError(
            f"isect_offsets must have shape {image_dims + (tile_height, tile_width)}, "
            f"got {isect_offsets.shape}"
        )
    if transmittances.device != means2d.device:
        raise ValueError("transmittances must be on the same device as means2d")
    if ray_transforms.device != means2d.device or opacities.device != means2d.device:
        raise ValueError("ray_transforms and opacities must be on the same device as means2d")
    if isect_offsets.device != means2d.device or flatten_ids.device != means2d.device:
        raise ValueError("isect_offsets and flatten_ids must be on the same device as means2d")
    if transmittances.dtype != torch.float32:
        raise ValueError(f"transmittances must be float32, got {transmittances.dtype}")
    if means2d.dtype != torch.float32 or ray_transforms.dtype != torch.float32:
        raise ValueError("means2d and ray_transforms must be float32")
    if opacities.dtype != torch.float32:
        raise ValueError(f"opacities must be float32, got {opacities.dtype}")
    if isect_offsets.dtype != torch.int32:
        raise ValueError(f"isect_offsets must be int32, got {isect_offsets.dtype}")
    if flatten_ids.dtype != torch.int32:
        raise ValueError(f"flatten_ids must be int32, got {flatten_ids.dtype}")
    if tile_height * tile_size < image_height:
        raise ValueError(
            f"tile_height * tile_size must cover image_height, got "
            f"{tile_height} * {tile_size} < {image_height}"
        )
    if tile_width * tile_size < image_width:
        raise ValueError(
            f"tile_width * tile_size must cover image_width, got "
            f"{tile_width} * {tile_size} < {image_width}"
        )

    out_gauss_ids, out_indices = _make_lazy_metal_func("metal_rasterize_to_indices_2dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    pixels_per_image = image_width * image_height
    out_pixel_ids = out_indices % pixels_per_image
    out_image_ids = out_indices // pixels_per_image
    return out_gauss_ids, out_pixel_ids, out_image_ids
