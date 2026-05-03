# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ._backend import has_metal
from ._wrapper import metal_null

__all__ = ["has_metal", "metal_null"]
