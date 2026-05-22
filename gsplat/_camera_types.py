# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Sequence

import torch
from typing_extensions import Literal

from gsplat.cuda._lidar import (
    FOV as FOVBase,
    RowOffsetStructuredSpinningLidarModelParametersExt as RowOffsetStructuredSpinningLidarModelParametersExtBase,
)

ExternalDistortionModelMeta = Literal["bivariate-windshield"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]


class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    LINEAR = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


@dataclass
class UnscentedTransformParameters:
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise RuntimeError("alpha must be positive")
        if 3.0 + self.kappa <= 0.0:
            raise RuntimeError("alpha and kappa must satisfy D + kappa > 0 for D=3")


@dataclass
class FThetaCameraDistortionParameters:
    reference_poly: FThetaPolynomialType
    pixeldist_to_angle_poly: Sequence[float]
    angle_to_pixeldist_poly: Sequence[float]
    max_angle: float
    linear_cde: Sequence[float]

    def __post_init__(self) -> None:
        self.pixeldist_to_angle_poly = tuple(self.pixeldist_to_angle_poly)
        self.angle_to_pixeldist_poly = tuple(self.angle_to_pixeldist_poly)
        self.linear_cde = tuple(self.linear_cde)
        if len(self.pixeldist_to_angle_poly) != 6:
            raise RuntimeError("pixeldist_to_angle_poly must have 6 coefficients")
        if len(self.angle_to_pixeldist_poly) != 6:
            raise RuntimeError("angle_to_pixeldist_poly must have 6 coefficients")
        if len(self.linear_cde) != 3:
            raise RuntimeError("linear_cde must have 3 coefficients")
        if self.max_angle <= 0.0:
            raise RuntimeError("max_angle must be positive")


class ExternalDistortionModelParameters(ABC):
    """Base class for external distortion model parameters."""


class ExternalDistortionReferencePolynomial(IntEnum):
    FORWARD = 1
    BACKWARD = 2


@dataclass
class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Backend-neutral external distortion parameters."""

    MAX_ORDER: ClassVar[int] = 5
    MAX_COEFFS: ClassVar[int] = 21

    reference_poly: ExternalDistortionReferencePolynomial = (
        ExternalDistortionReferencePolynomial.FORWARD
    )
    horizontal_poly: torch.Tensor | None = None
    vertical_poly: torch.Tensor | None = None
    horizontal_poly_inverse: torch.Tensor | None = None
    vertical_poly_inverse: torch.Tensor | None = None


class FOV(FOVBase):
    @classmethod
    def from_base(cls, base: FOVBase) -> "FOV":
        return cls(start=base.start, span=base.span, direction=base.direction)


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParametersExtBase
):
    """Lidar camera parameters extended with acceleration structures."""


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
