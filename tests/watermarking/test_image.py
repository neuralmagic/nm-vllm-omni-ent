# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.watermarking import VideoSealImageWatermarker, VisualTensor
from vllm_omni.watermarking.utils import watermark_outputs, watermark_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_videoseal_survives_png_round_trip(
    tmp_path: Path,
) -> None:
    pytest.importorskip("videoseal")
    source_path = Path(__file__).parents[1] / "assets/hunyuan/hunyuan_image_ref.png"
    source = Image.open(source_path).convert("RGB").resize((320, 192))
    pixels = torch.from_numpy(np.array(source, copy=True)).movedim(-1, 0)
    watermarker = VideoSealImageWatermarker()
    rng_state = torch.random.get_rng_state()

    try:
        assert not watermarker.is_watermarked(VisualTensor(pixels, 0))
        output = OmniRequestOutput.from_diffusion(request_id="request", images=[source])
        watermark_outputs([output], {"image": watermarker})
        watermarked = output.images[0]
        assert isinstance(watermarked, Image.Image)
        output_path = tmp_path / "watermarked.png"
        watermarked.save(output_path)
        reloaded = torch.from_numpy(np.array(Image.open(output_path).convert("RGB"), copy=True)).movedim(-1, 0)

        assert watermarker.is_watermarked(VisualTensor(reloaded, 0))
        assert not watermarker.is_watermarked(VisualTensor(torch.stack([reloaded, pixels]), 1))
        assert torch.equal(torch.random.get_rng_state(), rng_state)
    finally:
        watermarker.close()


def test_watermark_outputs_watermarks_pil_diffusion_images() -> None:
    """Ensure diffusion images use the image watermarker and remain PIL images."""
    watermarker = MagicMock()
    watermarker.watermark_output.side_effect = lambda _request_id, pixels, _metadata: pixels + 1
    output = OmniRequestOutput.from_diffusion(
        request_id="request",
        images=[Image.new("RGB", (10, 8))],
    )

    watermark_outputs([output], {"image": watermarker})

    assert isinstance(output.images[0], Image.Image)
    assert output.images[0].getpixel((0, 0)) == (1, 1, 1)
    watermarker.watermark_output.assert_called_once()
    watermarker.discard_request_state.assert_called_once_with("request")


def test_watermark_outputs_preserves_batched_image_array() -> None:
    watermarker = MagicMock()
    watermarker.watermark_output.side_effect = lambda _request_id, pixels, _metadata: pixels + 1
    images = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    output = OmniRequestOutput.from_diffusion(request_id="request", images=[images])

    watermark_outputs([output], {"image": watermarker})

    assert len(output.images) == 1
    assert isinstance(output.images[0], np.ndarray)
    assert output.images[0].shape == images.shape


def test_watermark_outputs_returns_raw_invalid_output() -> None:
    watermarker = MagicMock()
    watermarker.watermark_output.side_effect = ValueError("invalid output")
    image = Image.new("RGB", (10, 8))
    output = OmniRequestOutput.from_diffusion(request_id="request", images=[image])

    watermark_outputs([output], {"image": watermarker})

    assert output.images == [image]
    watermarker.discard_request_state.assert_called_once_with("request")


def test_watermark_payload_preserves_payload_partition() -> None:
    """Ensure metadata media is not shadowed by a new tensor entry."""
    watermarker = MagicMock()
    watermarker.watermark_output.side_effect = lambda _request_id, pixels, _metadata: pixels + 1
    payload = MultimodalPayload(metadata={"image": Image.new("RGB", (10, 8))})

    watermark_payload("request", OutputModalityNames.IMAGE, watermarker, payload)

    assert "image" not in payload.tensors
    assert payload.to_dict()["image"].getpixel((0, 0)) == (1, 1, 1)
