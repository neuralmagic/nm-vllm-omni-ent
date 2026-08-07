# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request body validation helpers for Omni OpenAI-compatible routes."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import HTTPException, Request
from vllm.entrypoints.serve.utils.api_utils import validate_json_request


async def validate_json_object_request(raw_request: Request) -> None:
    """Require ``application/json`` whose top-level value is a JSON object.

    vLLM ``ChatCompletionRequest`` ``mode="before"`` validators (e.g.
    ``check_cache_salt_support``) call ``data.get(...)`` and assume a dict.
    A JSON string/array body therefore raises ``AttributeError`` and becomes
    HTTP 500. Reject non-objects here with 400 before those validators run.

    See: https://github.com/neuralmagic/nm-cicd/issues/849
    """
    await validate_json_request(raw_request)
    try:
        body = await raw_request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON body",
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
