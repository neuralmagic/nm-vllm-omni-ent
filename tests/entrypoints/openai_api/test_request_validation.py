# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from vllm_omni.entrypoints.openai.request_validation import validate_json_object_request


@pytest.mark.asyncio
async def test_validate_json_object_request_accepts_object():
    request = MagicMock()
    request.json = AsyncMock(return_value={"model": "m", "messages": []})

    with patch(
        "vllm_omni.entrypoints.openai.request_validation.validate_json_request",
        new_callable=AsyncMock,
    ):
        await validate_json_object_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "this is not valid json{{{",
        ["not", "an", "object"],
        42,
        None,
    ],
)
async def test_validate_json_object_request_rejects_non_object(body):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)

    with patch(
        "vllm_omni.entrypoints.openai.request_validation.validate_json_request",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await validate_json_object_request(request)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
    assert "JSON object" in str(exc_info.value.detail)
