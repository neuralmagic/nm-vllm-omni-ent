# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for Qwen3-Omni-specific conversation.item.truncate
transcript reconstruction (FullDuplexRealtimeConnection.
_qwen3_omni_truncate_transcript).

Pure logic, no engine/tokenizer/websocket dependencies -- constructs the
connection object via object.__new__ and stubs only the two attributes the
method under test actually reads (self.session, self._tokenizer), matching
this file's existing narrow-scope model-specific helpers
(_qwen3_omni_audio_token_count) rather than spinning up a full connection.
"""

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.realtime.connection import (
    FullDuplexRealtimeConnection,
)
from vllm_omni.entrypoints.openai.realtime.session import AudioFullDuplexSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    """decode() just renders the token id list so tests can assert on
    exactly which prefix of ids was decoded, without a real vocab."""

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i in token_ids)


def _make_connection(item_token_ids: dict[str, list[int]]) -> FullDuplexRealtimeConnection:
    conn = object.__new__(FullDuplexRealtimeConnection)
    session = AudioFullDuplexSessionState()
    session.item_token_ids = item_token_ids
    conn.session = session
    conn._tokenizer = _FakeTokenizer()
    return conn


@pytest.mark.parametrize(
    ("audio_end_ms", "expected_tokens_heard"),
    [
        (0, 0),
        (191, 0),  # just under half of QWEN3_OMNI_MS_PER_TOKEN (383) -> rounds down
        (192, 1),  # just over half -> rounds up to the next token
        (383, 1),
        (574, 1),
        (575, 2),
        (766, 2),
        # The actual calibration data point this constant was derived from:
        # a real production truncate at audio_end_ms=8808 whose heard audio
        # was independently confirmed (Whisper transcription + tokenizer
        # alignment against the stored transcript) to stop at exactly token
        # index 22 (23 tokens) -- see QWEN3_OMNI_MS_PER_TOKEN's own comment.
        (8808, 23),
    ],
)
def test_truncate_transcript_ms_per_token_math(audio_end_ms, expected_tokens_heard) -> None:
    token_ids = list(range(30))
    conn = _make_connection({"item_1": token_ids})

    result = conn._qwen3_omni_truncate_transcript("item_1", audio_end_ms)

    expected = " ".join(f"tok{i}" for i in token_ids[:expected_tokens_heard]) if expected_tokens_heard else ""
    assert result == expected


def test_truncate_transcript_clamps_to_available_tokens() -> None:
    """A short response (fewer real tokens than the audio_end_ms implies,
    e.g. trailing EOS/pad frames after the last real token) must not index
    past the captured token list."""
    conn = _make_connection({"item_1": [11, 22, 33]})

    result = conn._qwen3_omni_truncate_transcript("item_1", audio_end_ms=100_000)

    assert result == "tok11 tok22 tok33"


def test_truncate_transcript_falls_back_to_empty_when_untracked() -> None:
    """Responses that included a tool call intentionally don't populate
    item_token_ids (see connection.py) -- must fall back to today's
    spec-minimum blank-out rather than raising or guessing."""
    conn = _make_connection({})

    result = conn._qwen3_omni_truncate_transcript("item_missing", audio_end_ms=1000)

    assert result == ""
