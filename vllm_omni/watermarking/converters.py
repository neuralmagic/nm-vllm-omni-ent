# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple, TypeAlias, cast

import numpy as np
import torch
from PIL import Image

from vllm_omni.outputs.output_modality import OutputModalityNames

VisualOutput: TypeAlias = torch.Tensor | np.ndarray | Image.Image

MediaToTensor = Callable[[object], torch.Tensor]
TensorToMedia = Callable[[torch.Tensor, object], object]


class ConverterPair(NamedTuple):
    """Utils for converting a given modality to tensors & back."""

    to_tensor: MediaToTensor
    restore: TensorToMedia


def media_to_tensor(data: object, converter: MediaToTensor) -> torch.Tensor:
    """Convert nested media values to one tensor."""
    if isinstance(data, list):
        return torch.stack([media_to_tensor(item, converter) for item in data])
    return converter(data)


def restore_media(data: torch.Tensor, source: object, converter: TensorToMedia) -> object:
    """Restore a tensor to its nested media value types."""
    if isinstance(source, list):
        return [restore_media(item, template, converter) for item, template in zip(data, source, strict=True)]
    return converter(data, source)


def array_to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Try to convert raw multimodal data to a tensor (modality agnostic)."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    raise TypeError(f"unsupported output type: {type(data).__name__}")


def restore_array(data: torch.Tensor, source: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Try to restore a data tensor to its original type (modality agnostic)."""
    if isinstance(source, torch.Tensor):
        return data
    if isinstance(source, np.ndarray):
        return data.numpy()
    raise TypeError(f"unsupported output type: {type(source).__name__}")


def image_to_tensor(data: VisualOutput) -> torch.Tensor:
    """Convert an image output to a tensor."""
    if isinstance(data, Image.Image):
        return torch.from_numpy(np.array(data, copy=True))
    return array_to_tensor(data)


def restore_image(data: torch.Tensor, source: VisualOutput) -> VisualOutput:
    """Restore a tensor to its image output type."""
    if isinstance(source, Image.Image):
        return Image.fromarray(data.cpu().numpy())
    return restore_array(data, source)


def video_to_tensor(data: VisualOutput) -> torch.Tensor:
    """Convert a video output to a tensor."""
    return image_to_tensor(data)


def restore_video(data: torch.Tensor, source: VisualOutput) -> VisualOutput:
    """Restore a tensor to its video output type."""
    return restore_image(data, source)


def infer_visual_channel_axis(
    source: object,
    pixels: torch.Tensor,
    modality: OutputModalityNames,
) -> int:
    """Best effort resolve the RGB channel axis for an image or video output."""
    valid_ranks = {3, 4} if modality is OutputModalityNames.IMAGE else {4, 5}
    if pixels.ndim not in valid_ranks:
        raise ValueError(f"invalid {modality.value} shape {tuple(pixels.shape)}")

    list_depth = 0
    while isinstance(source, list):
        if not source:
            raise ValueError(f"empty {modality.value} output")
        source = source[0]
        list_depth += 1

    if isinstance(source, np.ndarray | Image.Image):
        channel_axis = pixels.ndim - 1
    elif isinstance(source, torch.Tensor):
        first_media_axis = list_depth
        if not list_depth and pixels.ndim == max(valid_ranks):
            first_media_axis = 1
        candidates = [axis for axis in range(first_media_axis, pixels.ndim) if pixels.shape[axis] == 3]
        if len(candidates) != 1:
            raise ValueError(
                f"ambiguous {modality.value} channel axis for shape {tuple(pixels.shape)}; provide channel_axis"
            )
        channel_axis = candidates[0]
    else:
        raise TypeError(f"unsupported {modality.value} output type: {type(source).__name__}")
    if pixels.shape[channel_axis] != 3:
        raise ValueError(f"{modality.value} pixels have no RGB channel at axis {channel_axis}")
    return channel_axis


MEDIA_CONVERTERS: Mapping[OutputModalityNames, ConverterPair] = {
    OutputModalityNames.AUDIO: ConverterPair(cast(MediaToTensor, array_to_tensor), cast(TensorToMedia, restore_array)),
    OutputModalityNames.IMAGE: ConverterPair(cast(MediaToTensor, image_to_tensor), cast(TensorToMedia, restore_image)),
    OutputModalityNames.VIDEO: ConverterPair(cast(MediaToTensor, video_to_tensor), cast(TensorToMedia, restore_video)),
}
