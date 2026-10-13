# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AudioTensor:
    """Normalized float audio with its sample rate."""

    samples: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class VisualTensor:
    """Pixels with their RGB channel axis."""

    pixels: torch.Tensor
    channel_axis: int
