# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral dispatch helpers for gsplat custom op wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ._camera_types import (
    BivariateWindshieldModelParameters,
    CameraModel,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    UnscentedTransformParameters,
)
from .cuda._lidar import RowOffsetStructuredSpinningLidarModelParameters


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_tensors(item)


def _infer_device(*args: Any, **kwargs: Any) -> torch.device | None:
    tensors = [tensor for value in (*args, *kwargs.values()) for tensor in _iter_tensors(value)]
    if not tensors:
        return None

    first_device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.device != first_device:
            raise ValueError(
                "All tensor inputs must be on the same device, got "
                f"{first_device} and {tensor.device}."
            )
    return first_device


def _cuda_wrapper():
    from .cuda import _wrapper as cuda_wrapper

    return cuda_wrapper


def _metal_wrapper():
    from .metal import _wrapper as metal_wrapper

    return metal_wrapper


def _has_metal_op(name: str) -> bool:
    from .metal._backend import has_metal

    if not has_metal():
        return False
    return hasattr(torch.ops.gsplat, name)


def _convert_shared_type_for_cuda(value: Any) -> Any:
    cuda_wrapper = _cuda_wrapper()

    if type(value) is UnscentedTransformParameters:
        return cuda_wrapper.UnscentedTransformParameters(
            alpha=value.alpha,
            beta=value.beta,
            kappa=value.kappa,
            in_image_margin_factor=value.in_image_margin_factor,
            require_all_sigma_points_valid=value.require_all_sigma_points_valid,
        )

    if type(value) is FThetaCameraDistortionParameters:
        return cuda_wrapper.FThetaCameraDistortionParameters(
            reference_poly=int(value.reference_poly),
            pixeldist_to_angle_poly=tuple(float(x) for x in value.pixeldist_to_angle_poly),
            angle_to_pixeldist_poly=tuple(float(x) for x in value.angle_to_pixeldist_poly),
            max_angle=float(value.max_angle),
            linear_cde=tuple(float(x) for x in value.linear_cde),
        )

    if type(value) is BivariateWindshieldModelParameters:
        params = cuda_wrapper.BivariateWindshieldModelParameters()
        params.reference_poly = value.reference_poly
        params.horizontal_poly = value.horizontal_poly
        params.vertical_poly = value.vertical_poly
        params.horizontal_poly_inverse = value.horizontal_poly_inverse
        params.vertical_poly_inverse = value.vertical_poly_inverse
        return params

    if type(value) is RowOffsetStructuredSpinningLidarModelParametersExt:
        return cuda_wrapper.RowOffsetStructuredSpinningLidarModelParametersExt(
            value,
            value.angles_to_columns_map,
            value.tiling,
        )

    return value


def _prepare_for_cuda(value: Any) -> Any:
    value = _convert_shared_type_for_cuda(value)
    if isinstance(value, Mapping):
        return type(value)((key, _prepare_for_cuda(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_prepare_for_cuda(item) for item in value)
    if isinstance(value, list):
        return [_prepare_for_cuda(item) for item in value]
    return value


def _select_wrapper(device: torch.device | None):
    if device is not None and device.type == "mps":
        return _metal_wrapper(), "mps"
    return _cuda_wrapper(), "cuda"


def _dispatch(name: str, *args: Any, metal_name: str | None = None, **kwargs: Any):
    device = _infer_device(*args, **kwargs)
    wrapper, backend = _select_wrapper(device)
    if backend == "cuda":
        args = tuple(_prepare_for_cuda(arg) for arg in args)
        kwargs = {key: _prepare_for_cuda(value) for key, value in kwargs.items()}
    target_name = metal_name if backend == "mps" and metal_name is not None else name
    return getattr(wrapper, target_name)(*args, **kwargs)


def _normalize_eval3d_kwargs_for_mps(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if "image_width" in kwargs:
        kwargs["width"] = kwargs.pop("image_width")
    if "image_height" in kwargs:
        kwargs["height"] = kwargs.pop("image_height")
    return kwargs


def has_camera_wrappers() -> bool:
    return True


def has_2dgs() -> bool:
    if _cuda_wrapper().has_2dgs():
        return True
    return _has_metal_op("metal_projection_2dgs_fused_fwd")


def has_3dgs() -> bool:
    if _cuda_wrapper().has_3dgs():
        return True
    return _has_metal_op("metal_projection_ewa_simple_fwd")


def has_3dgut() -> bool:
    if _cuda_wrapper().has_3dgut():
        return True
    return _has_metal_op("metal_projection_ut_3dgs_fused")


def has_adam() -> bool:
    if _cuda_wrapper().has_adam():
        return True
    return _has_metal_op("metal_adam")


def has_reloc() -> bool:
    if _cuda_wrapper().has_reloc():
        return True
    return _has_metal_op("metal_relocation")


def world_to_cam(*args: Any, **kwargs: Any):
    return _cuda_wrapper().world_to_cam(*args, **kwargs)


def adam(*args: Any, **kwargs: Any):
    return _dispatch("adam", *args, **kwargs)


def relocation(*args: Any, **kwargs: Any):
    return _dispatch("relocation", *args, metal_name="relocation", **kwargs)


def spherical_harmonics(*args: Any, **kwargs: Any):
    return _dispatch("spherical_harmonics", *args, **kwargs)


def quat_scale_to_covar_preci(*args: Any, **kwargs: Any):
    return _dispatch("quat_scale_to_covar_preci", *args, **kwargs)


def proj(*args: Any, **kwargs: Any):
    return _dispatch("proj", *args, metal_name="projection_ewa_simple", **kwargs)


def fully_fused_projection(*args: Any, **kwargs: Any):
    return _dispatch("fully_fused_projection", *args, **kwargs)


def fully_fused_projection_with_ut(*args: Any, **kwargs: Any):
    return _dispatch("fully_fused_projection_with_ut", *args, **kwargs)


def fully_fused_projection_2dgs(*args: Any, **kwargs: Any):
    return _dispatch("fully_fused_projection_2dgs", *args, **kwargs)


def isect_tiles(*args: Any, **kwargs: Any):
    return _dispatch("isect_tiles", *args, metal_name="intersect_tiles", **kwargs)


def isect_tiles_lidar(*args: Any, **kwargs: Any):
    return _dispatch("isect_tiles_lidar", *args, metal_name="intersect_tiles_lidar", **kwargs)


def isect_offset_encode(*args: Any, **kwargs: Any):
    return _dispatch("isect_offset_encode", *args, metal_name="intersect_offset_encode", **kwargs)


def rasterize_to_pixels(*args: Any, **kwargs: Any):
    return _dispatch("rasterize_to_pixels", *args, **kwargs)


def rasterize_to_pixels_eval3d(*args: Any, **kwargs: Any):
    device = _infer_device(*args, **kwargs)
    if device is not None and device.type == "mps":
        return _metal_wrapper().rasterize_to_pixels_eval3d(
            *args, **_normalize_eval3d_kwargs_for_mps(kwargs)
        )
    args = tuple(_prepare_for_cuda(arg) for arg in args)
    kwargs = {key: _prepare_for_cuda(value) for key, value in kwargs.items()}
    return _cuda_wrapper().rasterize_to_pixels_eval3d(*args, **kwargs)


def rasterize_to_pixels_eval3d_extra(*args: Any, **kwargs: Any):
    device = _infer_device(*args, **kwargs)
    if device is not None and device.type == "mps":
        return _metal_wrapper().rasterize_to_pixels_eval3d(
            *args, **_normalize_eval3d_kwargs_for_mps(kwargs)
        )
    args = tuple(_prepare_for_cuda(arg) for arg in args)
    kwargs = {key: _prepare_for_cuda(value) for key, value in kwargs.items()}
    return _cuda_wrapper().rasterize_to_pixels_eval3d_extra(*args, **kwargs)


def rasterize_to_indices_in_range(*args: Any, **kwargs: Any):
    return _dispatch("rasterize_to_indices_in_range", *args, **kwargs)


def rasterize_to_pixels_2dgs(*args: Any, **kwargs: Any):
    return _dispatch("rasterize_to_pixels_2dgs", *args, **kwargs)


def rasterize_to_indices_in_range_2dgs(*args: Any, **kwargs: Any):
    return _dispatch("rasterize_to_indices_in_range_2dgs", *args, **kwargs)


__all__ = [
    "BivariateWindshieldModelParameters",
    "CameraModel",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "RollingShutterType",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "UnscentedTransformParameters",
    "adam",
    "fully_fused_projection",
    "fully_fused_projection_2dgs",
    "fully_fused_projection_with_ut",
    "has_2dgs",
    "has_3dgs",
    "has_3dgut",
    "has_adam",
    "has_camera_wrappers",
    "has_reloc",
    "isect_offset_encode",
    "isect_tiles",
    "isect_tiles_lidar",
    "proj",
    "quat_scale_to_covar_preci",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_pixels_eval3d",
    "rasterize_to_pixels_eval3d_extra",
    "relocation",
    "spherical_harmonics",
    "world_to_cam",
]
