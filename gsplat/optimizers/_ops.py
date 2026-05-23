# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch


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
    """Dispatch fused Adam to the backend that matches the parameter device."""

    device_type = param.device.type
    if device_type == "mps":
        from ..metal._wrapper import adam as metal_adam

        metal_adam(param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps)
        return
    if device_type == "cuda":
        from ..cuda._wrapper import adam as cuda_adam

        cuda_adam(param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps)
        return
    raise RuntimeError(f"SelectiveAdam only supports CUDA and MPS tensors, got {param.device}")
