# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.watermarking.base import Watermarker
from vllm_omni.watermarking.converters import infer_visual_channel_axis
from vllm_omni.watermarking.types import VisualTensor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from videoseal.models.videoseal import Videoseal
    from videoseal.utils.cfg import VideosealConfig

try:
    import videoseal
except ImportError:
    videoseal = PlaceholderModule("videoseal")  # type: ignore[assignment]


@dataclass(frozen=True)
class _VideoSealState:
    message: torch.Tensor


def _to_float_pixels(pixels: torch.Tensor, media: str) -> torch.Tensor:
    """Normalize pixel tensor to float in the range [0, 1]."""
    if pixels.dtype == torch.uint8:
        return pixels.to(torch.float32).div_(255)
    if not pixels.is_floating_point():
        raise TypeError(f"{media} pixels must use uint8 or a floating-point dtype")
    pixels = pixels.to(torch.float32)
    if not torch.isfinite(pixels).all():
        raise ValueError(f"{media} pixels must be finite")
    if pixels.amin().item() < 0 or pixels.amax().item() > 1:
        raise ValueError(f"floating-point {media} pixels must be in [0, 1]")
    return pixels


def _restore_dtype(pixels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore the original dtype of the pixel tensor."""
    if dtype == torch.uint8:
        return pixels.mul(255).round().clamp_(0, 255).to(torch.uint8)
    return pixels.to(dtype)


def _canonicalize_pixels(
    data: VisualTensor,
    *,
    is_video: bool,
) -> torch.Tensor:
    media = "video" if is_video else "image"
    unbatched_rank = 4 if is_video else 3
    canonical_channel_axis = 2 if is_video else 1
    pixels = data.pixels
    if pixels.ndim not in {unbatched_rank, unbatched_rank + 1}:
        raise ValueError(f"{media} pixels must have rank {unbatched_rank} or {unbatched_rank + 1}")
    if pixels.ndim == unbatched_rank:
        pixels = pixels.unsqueeze(0)
        channel_axis = data.channel_axis + 1
    else:
        channel_axis = data.channel_axis
    pixels = pixels.movedim(channel_axis, canonical_channel_axis)
    pixels = _to_float_pixels(pixels, media).contiguous()
    return pixels


def _restore_pixels(
    pixels: torch.Tensor,
    source: VisualTensor,
    *,
    is_video: bool,
) -> VisualTensor:
    unbatched_rank = 4 if is_video else 3
    canonical_channel_axis = 2 if is_video else 1
    if source.pixels.ndim == unbatched_rank:
        pixels = pixels.squeeze(0)
        canonical_channel_axis -= 1
    pixels = pixels.movedim(canonical_channel_axis, source.channel_axis)
    pixels = _restore_dtype(pixels, source.pixels.dtype)
    return VisualTensor(pixels, source.channel_axis)


def _visual_tensor_from_output(
    data: torch.Tensor,
    metadata: Mapping[str, object],
    modality: OutputModalityNames,
) -> VisualTensor:
    channel_axis = metadata.get("channel_axis")
    if channel_axis is None:
        channel_axis = infer_visual_channel_axis(data, data, modality)
    if isinstance(channel_axis, bool) or not isinstance(channel_axis, int):
        raise TypeError("channel_axis must be an integer")
    normalized_axis = channel_axis % data.ndim
    if data.shape[normalized_axis] != 3:
        raise ValueError(f"invalid channel axis {channel_axis} for shape {tuple(data.shape)}")
    return VisualTensor(data, normalized_axis)


class VideoSealImageWatermarker(Watermarker[VisualTensor, _VideoSealState]):
    """Watermark images with VideoSeal."""

    supported_types = (VisualTensor,)

    def __init__(self, *, detection_accuracy_threshold: float = 0.6) -> None:
        super().__init__()
        if isinstance(videoseal, PlaceholderModule):
            raise ImportError("VideoSeal requires `pip install 'vllm-omni[watermarking]'`")
        self._detection_accuracy_threshold = detection_accuracy_threshold
        logger.info("Loading VideoSeal watermark model on CPU")
        self._model = self._load_model()

    def _from_output(self, data: torch.Tensor, metadata: Mapping[str, object]) -> VisualTensor:
        return _visual_tensor_from_output(data, metadata, OutputModalityNames.IMAGE)

    def _to_output(self, data: VisualTensor) -> torch.Tensor:
        return data.pixels

    def _new_state(self, data: VisualTensor) -> _VideoSealState:
        pixels = self._canonicalize(data)
        return _VideoSealState(self._messages(pixels.shape[0]))

    def _messages(self, batch_size: int) -> torch.Tensor:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return self._model.get_random_msg(batch_size)

    def _watermark(self, data: VisualTensor, state: _VideoSealState) -> VisualTensor:
        pixels = self._canonicalize(data)
        watermarked = self._embed(pixels, state.message)
        return self._restore(watermarked, data)

    @staticmethod
    def _canonicalize(data: VisualTensor) -> torch.Tensor:
        return _canonicalize_pixels(data, is_video=False)

    @staticmethod
    def _restore(pixels: torch.Tensor, source: VisualTensor) -> VisualTensor:
        return _restore_pixels(pixels, source, is_video=False)

    def _embed(self, pixels: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        outputs = self._model.embed(
            pixels.to(device="cpu"),
            msgs=message,
            is_video=False,
        )
        return self._embedded_pixels(outputs, pixels)

    @staticmethod
    def _embedded_pixels(outputs: Mapping[str, object], source: torch.Tensor) -> torch.Tensor:
        watermarked = outputs.get("imgs_w")
        if not isinstance(watermarked, torch.Tensor) or watermarked.shape != source.shape:
            raise TypeError("VideoSeal embed output must contain an 'imgs_w' tensor matching the input shape")
        return watermarked.to(device=source.device, dtype=source.dtype)

    def _is_watermarked(self, data: VisualTensor) -> bool:
        pixels = self._canonicalize(data)
        decoded = self._extract_messages(pixels)
        accuracies = self._message_accuracies(decoded, self._messages(pixels.shape[0]))
        return accuracies.amin().item() > self._detection_accuracy_threshold

    def _extract_messages(self, pixels: torch.Tensor) -> torch.Tensor:
        outputs = self._model.detect(pixels.to(device="cpu"), is_video=False)
        predictions = outputs.get("preds")
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 2 or predictions.shape[1] < 2:
            raise TypeError("VideoSeal detect output must contain a rank-2 'preds' tensor")
        return predictions[:, 1:] > 0

    @staticmethod
    def _message_accuracies(decoded: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        if decoded.shape != messages.shape:
            raise ValueError("VideoSeal detect output does not match the watermark message shape")
        return (decoded == messages.to(device=decoded.device).bool()).float().mean(dim=1)

    def _validate_output(self, source: VisualTensor, watermarked: VisualTensor) -> None:
        super()._validate_output(source, watermarked)
        if (
            watermarked.pixels.shape != source.pixels.shape
            or watermarked.pixels.dtype != source.pixels.dtype
            or watermarked.pixels.device != source.pixels.device
            or watermarked.channel_axis != source.channel_axis
        ):
            raise ValueError("VideoSeal changed the output tensor or layout")

    @staticmethod
    def _load_model() -> Videoseal:
        """Loads a Videoseal (1.0) model; this is needed because Videoseal does not
        resolve file paths correctly when installed through pip.

        See: https://github.com/facebookresearch/videoseal/issues/73; once this issue is resolved,
        we can remove this workaround.
        """
        from videoseal.augmentation.augmenter import get_dummy_augmenter
        from videoseal.models import Videoseal, build_embedder, build_extractor
        from videoseal.modules.jnd import JND

        cfg_path = Path(videoseal.__file__).parent / "cards" / "videoseal_1.0.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"videoseal config path {cfg_path} does not exist!")

        card = cast("VideosealConfig", OmegaConf.load(cfg_path))
        args = card.args
        embedder = build_embedder(card.embedder.model, card.embedder.params, args.nbits, args.hidden_size_multiplier)
        extractor = build_extractor(card.extractor.model, card.extractor.params, args.img_size_proc, args.nbits)
        augmenter = get_dummy_augmenter()
        # This is the underlying attenuation object that gets initialized from jnd_1_1.
        # https://github.com/facebookresearch/videoseal/blob/main/videoseal/cards/videoseal_1.0.yaml
        attenuation = JND(in_channels=1, out_channels=1)

        with torch.random.fork_rng(devices=[]):
            model = Videoseal(
                embedder,
                extractor,
                augmenter,
                attenuation=attenuation,
                scaling_w=args.scaling_w,
                scaling_i=args.scaling_i,
                img_size=args.img_size_proc,
                chunk_size=args.videoseal_chunk_size,
                step_size=args.videoseal_step_size,
            )
            checkpoint = torch.hub.load_state_dict_from_url(
                card.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "VideoSeal checkpoint is incompatible with the packaged model card: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        model.eval().to(torch.device("cpu"))
        return model


class VideoSealVideoWatermarker(VideoSealImageWatermarker):
    """Watermark videos with VideoSeal."""

    def _from_output(self, data: torch.Tensor, metadata: Mapping[str, object]) -> VisualTensor:
        return _visual_tensor_from_output(data, metadata, OutputModalityNames.VIDEO)

    @staticmethod
    def _canonicalize(data: VisualTensor) -> torch.Tensor:
        return _canonicalize_pixels(data, is_video=True)

    @staticmethod
    def _restore(pixels: torch.Tensor, source: VisualTensor) -> VisualTensor:
        return _restore_pixels(pixels, source, is_video=True)

    def _embed(self, pixels: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        watermarked = []
        for index, frames in enumerate(pixels):
            outputs = self._model.embed(
                frames.to(device="cpu"),
                msgs=message[index : index + 1],
                is_video=True,
            )
            watermarked.append(self._embedded_pixels(outputs, frames))
        return torch.stack(watermarked)

    def _extract_messages(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._model.extract_message(frames.to(device="cpu"), aggregation="avg") for frames in pixels])
