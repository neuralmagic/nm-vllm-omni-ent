# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.config.watermarking import WatermarkConfig
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.watermarking import VideoSealVideoWatermarker, VisualTensor
from vllm_omni.watermarking.utils import watermark_outputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_videoseal_video_survives_streaming_chunks_and_mp4_round_trip() -> None:
    pytest.importorskip("videoseal")
    source_path = Path(__file__).parents[1] / "assets/hunyuan/hunyuan_image_ref.png"
    source = Image.open(source_path).convert("RGB").resize((320, 192))
    pixels = torch.from_numpy(np.array(source, copy=True)).movedim(-1, 0)
    frames = torch.stack([pixels.roll(frame, dims=1) for frame in range(4)])
    watermarker = VideoSealVideoWatermarker()

    try:
        assert not watermarker.is_watermarked(VisualTensor(frames, 1))
        watermarked_chunks = []
        chunks = frames.split(2)
        for index, chunk in enumerate(chunks):
            output = OmniRequestOutput.from_diffusion(
                request_id="request",
                images=[chunk],
                final_output_type="video",
                finished=index == len(chunks) - 1,
            )
            watermark_outputs([output], {"video": watermarker})
            watermarked_chunks.append(output.images[0])
        watermarked = torch.cat(watermarked_chunks)

        assert watermarked.shape == frames.shape
        assert watermarked.dtype == torch.uint8
        encoded = mux_video_audio_bytes(watermarked.movedim(1, -1).numpy(), fps=4)
        with av.open(io.BytesIO(encoded), format="mp4") as container:
            decoded = torch.stack(
                [
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).movedim(-1, 0)
                    for frame in container.decode(video=0)
                ]
            )

        assert decoded.shape == frames.shape
        assert watermarker.is_watermarked(VisualTensor(decoded, 1))
    finally:
        watermarker.close()


@pytest.mark.parametrize("nested", [False, True])
def test_watermark_outputs_watermarks_pil_video_frames(nested: bool) -> None:
    """Ensure diffusion video frames remain PIL images after watermarking."""
    watermarker = _FakeOutputWatermarker()
    frames = [Image.new("RGB", (10, 8)), Image.new("RGB", (10, 8))]
    output = OmniRequestOutput.from_diffusion(
        request_id="request",
        images=[frames] if nested else frames,
        final_output_type="video",
    )

    watermark_outputs([output], {"video": watermarker})

    watermarked = output.images[0] if nested else output.images
    assert all(isinstance(frame, Image.Image) for frame in watermarked)
    assert watermarked[0].getpixel((0, 0)) == (1, 1, 1)
    assert watermarker.calls == 1
    assert watermarker.input_shapes == [(1, 2, 8, 10, 3) if nested else (2, 8, 10, 3)]
    assert watermarker.discarded == ["request"]


@pytest.mark.parametrize("frame_type", [np.ndarray, torch.Tensor])
def test_watermark_outputs_watermarks_array_frame_lists(frame_type: type[object]) -> None:
    watermarker = _FakeOutputWatermarker()
    arrays = [np.zeros((8, 10, 3), dtype=np.uint8) for _ in range(2)]
    frames = arrays if frame_type is np.ndarray else [torch.from_numpy(frame) for frame in arrays]
    output = OmniRequestOutput.from_diffusion(
        request_id="request",
        images=[frames],
        final_output_type="video",
    )

    watermark_outputs([output], {"video": watermarker})

    assert all(isinstance(frame, frame_type) for frame in output.images[0])
    assert int(output.images[0][0][0, 0, 0]) == 1
    assert watermarker.calls == 1
    assert watermarker.input_shapes == [(1, 2, 8, 10, 3)]


def test_watermark_outputs_preserves_batched_video_array() -> None:
    watermarker = _FakeOutputWatermarker()
    video = np.zeros((2, 3, 8, 10, 3), dtype=np.uint8)
    output = OmniRequestOutput.from_diffusion(
        request_id="request",
        images=[video],
        final_output_type="video",
    )

    watermark_outputs([output], {"video": watermarker})

    assert len(output.images) == 1
    assert isinstance(output.images[0], np.ndarray)
    assert output.images[0].shape == video.shape
    assert watermarker.input_shapes == [video.shape]


class _FakeOutputWatermarker:
    def __init__(self) -> None:
        self.calls = 0
        self.input_shapes: list[tuple[int, ...]] = []
        self.discarded: list[str] = []

    def watermark_output(self, request_id: str, pixels: torch.Tensor, metadata: object) -> torch.Tensor:
        self.calls += 1
        self.input_shapes.append(tuple(pixels.shape))
        return pixels + 1

    def discard_request_state(self, request_id: str) -> None:
        self.discarded.append(request_id)


def test_stage_pool_initializes_video_watermarker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure video stage initialization recognizes the video modality."""
    watermarker = object()
    monkeypatch.setitem(StagePool._watermarker_registry["video"], "videoseal", lambda: watermarker)

    result = StagePool.initialize_watermarkers(
        "video",
        WatermarkConfig({"video": {"algorithm": "videoseal"}}),
    )

    assert result == {"video": watermarker}
