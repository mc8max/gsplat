# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from enum import IntEnum
from typing import Any

import torch
from typing_extensions import Literal

from gsplat.cuda._lidar import (
    FOV as FOVBase,
    RowOffsetStructuredSpinningLidarModelParametersExt as RowOffsetStructuredSpinningLidarModelParametersExtBase,
)

ExternalDistortionModelMeta = Literal["bivariate-windshield"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]


def _unavailable_cuda_cls(name: str) -> Any:
    class _UnavailableCudaCls:
        __name__ = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "gsplat CUDA extension is not available (not built or failed to load). "
                f"Cannot instantiate '{name}'."
            )

    return _UnavailableCudaCls


def _make_lazy_cuda_cls(name: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from gsplat.cuda._backend import _C

    if _C is None:
        return _unavailable_cuda_cls(name)

    try:
        return getattr(torch.classes.gsplat, name)
    except RuntimeError as e:
        if "does not exist" in str(e) or "torch::class_" in str(e):
            return _unavailable_cuda_cls(name)
        raise


class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


UnscentedTransformParameters = _make_lazy_cuda_cls("UnscentedTransformParameters")
FThetaCameraDistortionParameters = _make_lazy_cuda_cls(
    "FThetaCameraDistortionParameters"
)


class ExternalDistortionModelParameters(ABC):
    """Base class for external distortion model parameters."""


class ExternalDistortionReferencePolynomial(IntEnum):
    FORWARD = 1
    BACKWARD = 2


class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Thin wrapper around the CUDA BivariateWindshieldModelParameters class."""

    _cuda_cls = None
    MAX_ORDER: int = 5
    MAX_COEFFS: int = 21

    @classmethod
    def _ensure_cuda_cls(cls):
        if cls._cuda_cls is None:
            cls._cuda_cls = _make_lazy_cuda_cls("BivariateWindshieldModelParameters")
            cls.MAX_ORDER = cls._cuda_cls.get_max_order()
            cls.MAX_COEFFS = cls._cuda_cls.get_max_coeffs()

    def __new__(cls):
        cls._ensure_cuda_cls()
        return cls._cuda_cls()


class FOV(FOVBase):
    @classmethod
    def from_base(cls, base: FOVBase) -> "FOV":
        return cls(start=base.start, span=base.span, direction=base.direction)

    def to_cpp(self):
        fov_cuda = _make_lazy_cuda_cls("FOV")
        return fov_cuda(start=self.start, span=self.span)


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParametersExtBase
):
    """Lidar camera parameters extended with acceleration structures."""

    def to_cpp(self) -> Any:
        lidar_params_cuda = _make_lazy_cuda_cls(
            "RowOffsetStructuredSpinningLidarModelParametersExt"
        )
        return lidar_params_cuda(
            row_elevations_rad=self.row_elevations_rad.contiguous(),
            column_azimuths_rad=self.column_azimuths_rad.contiguous(),
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad.contiguous(),
            spinning_direction=self.spinning_direction.value,
            spinning_frequency_hz=self.spinning_frequency_hz,
            fov_vert_rad=FOV.from_base(self.fov_vert_rad).to_cpp(),
            fov_horiz_rad=FOV.from_base(self.fov_horiz_rad).to_cpp(),
            fov_eps_rad=self.fov_eps_rad,
            angles_to_columns_map=self.angles_to_columns_map,
            n_bins_azimuth=self.tiling.n_bins_azimuth,
            n_bins_elevation=self.tiling.n_bins_elevation,
            cdf_elevation=self.tiling.cdf_elevation.contiguous(),
            cdf_dense_ray_mask=self.tiling.cdf_dense_ray_mask.contiguous(),
            tiles_to_elements_map=self.tiling.tiles_to_elements_map.contiguous(),
            tiles_pack_info=self.tiling.tiles_pack_info.contiguous(),
        )


__all__ = [
    "BivariateWindshieldModelParameters",
    "CameraModel",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "FOV",
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "RollingShutterType",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "UnscentedTransformParameters",
]
