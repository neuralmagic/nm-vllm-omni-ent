# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.watermarking.audio_seal import AudioSealWatermarker
from vllm_omni.watermarking.base import AudioWatermarkerBase, Watermarker, WatermarkFailureError
from vllm_omni.watermarking.types import AudioTensor, VisualTensor
from vllm_omni.watermarking.video_seal import VideoSealImageWatermarker, VideoSealVideoWatermarker

WATERMARKER_REGISTRY: dict[str, dict[str, type[Watermarker]]] = {
    "audio": {"audioseal": AudioSealWatermarker},
    "image": {"videoseal": VideoSealImageWatermarker},
    "video": {"videoseal": VideoSealVideoWatermarker},
}

__all__ = [
    "AudioSealWatermarker",
    "AudioTensor",
    "AudioWatermarkerBase",
    "VideoSealImageWatermarker",
    "VideoSealVideoWatermarker",
    "VisualTensor",
    "WATERMARKER_REGISTRY",
    "WatermarkFailureError",
    "Watermarker",
]
