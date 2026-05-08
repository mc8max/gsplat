# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch math references shared by the Metal backend and its tests."""

from gsplat.cuda._math import _quat_scale_to_covar_preci

__all__ = ["_quat_scale_to_covar_preci"]
