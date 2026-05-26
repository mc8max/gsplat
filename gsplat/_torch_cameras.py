# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral accessors for shared torch camera helpers."""

from gsplat.cuda._torch_cameras import (
    _BaseCameraModel,
    _interpolate_shutter_pose,
    _pose_camera_ray_to_world_ray,
)

__all__ = [
    "_BaseCameraModel",
    "_interpolate_shutter_pose",
    "_pose_camera_ray_to_world_ray",
]
