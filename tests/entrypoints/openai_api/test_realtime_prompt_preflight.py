# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types import realtime as types

from vllm_omni.entrypoints.openai.realtime.connection import (
    FullDuplexRealtimeConnection,
    _ResolvedResponse,
)
from vllm_omni.entrypoints.openai.realtime.session import (
    ActiveResponse,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


def _websocket_events(websocket: _FakeWebSocket) -> list[dict[str, Any]]:
    return [json.loads(message) for message in websocket.messages]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_updates", "expected_wm_vals", "expected_error_codes", "expected_num_updated"),
    [
        ([], [], [], 0),
        ([{"watermarking": False}, {"instructions": "Be brief."}], [False], [], 2),
        ([{"watermarking": None}], [], ["invalid_request_error"], 0),
    ],
)
async def test_realtime_watermarking_follows_session_updates(
    session_updates,
    expected_wm_vals,
    expected_error_codes,
    expected_num_updated,
) -> None:
    """Ensure responses use the session's latest valid watermarking setting and invalid updates are not applied."""
    submitted_wm_vals: list[bool] = []

    async def generate(**kwargs: Any):
        if "watermarking" in kwargs:
            submitted_wm_vals.append(kwargs["watermarking"])
        yield SimpleNamespace(final_output_type="text", outputs=[])

    engine = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=100, multimodal_config=None),
        default_sampling_params_list=[],
        generate=generate,
    )
    websocket = _FakeWebSocket()
    connection = FullDuplexRealtimeConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
    )

    # Patch build_full_prompt on the connection so that we don't need a tokenizer
    async def build_full_prompt(**_kwargs: Any) -> dict[str, Any]:
        return {"prompt_token_ids": []}

    connection._build_full_prompt = build_full_prompt

    for session in session_updates:
        event = types.SessionUpdateEvent.model_validate(
            {"type": "session.update", "session": {"type": "realtime", **session}}
        )
        await connection._dispatch_event(event)
    response = _ResolvedResponse(
        input=[],
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )
    active = ActiveResponse(response_id="resp_test", request_id="req_test")
    connection.session.active_response = active

    await connection._run_response(active.response_id, response)
    events = _websocket_events(websocket)
    assert events and events[-1]["response"]["status"] == "completed"

    actual_error_codes = [event["error"]["code"] for event in events if event["type"] == "error"]
    actual_num_updated = sum(event["type"] == "session.updated" for event in events)

    # Ensure the watermark value submitted to generate is correct
    assert submitted_wm_vals == expected_wm_vals
    # Ensure that we have the right numbers of errors vs successful updates
    assert actual_error_codes == expected_error_codes
    assert actual_num_updated == expected_num_updated
