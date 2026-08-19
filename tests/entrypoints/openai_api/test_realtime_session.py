# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for the /v1/realtime duplex session state
(vllm_omni.entrypoints.openai.realtime.session): AudioFullDuplexSessionState's
item bookkeeping (insert_item upsert/positioning, remove_item side-table
cleanup, find_item/find_item_index) and merge_session_config/_deep_merge.

Pure Python/pydantic, no engine/tokenizer/websocket dependencies.
"""

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.realtime import types
from vllm_omni.entrypoints.openai.realtime.session import (
    AudioFullDuplexSessionState,
    _deep_merge,
    merge_session_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _assistant_item(item_id: str | None = None) -> types.RealtimeConversationItemAssistantMessage:
    return types.RealtimeConversationItemAssistantMessage(
        type="message",
        role="assistant",
        id=item_id,
        content=[{"type": "output_text", "text": "hi"}],  # type: ignore[list-item]
    )


# ---------------------------------------------------------------------- #
#  insert_item                                                            #
# ---------------------------------------------------------------------- #


def test_insert_item_appends_when_no_previous_item_id() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))
    s.insert_item(_assistant_item("item_b"))

    assert [item.id for item in s.items] == ["item_a", "item_b"]
    assert s.next_item_index == 2


def test_insert_item_inserts_after_previous_item_id() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))
    s.insert_item(_assistant_item("item_c"))

    s.insert_item(_assistant_item("item_b"), previous_item_id="item_a")

    assert [item.id for item in s.items] == ["item_a", "item_b", "item_c"]


def test_insert_item_root_inserts_at_start() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))

    s.insert_item(_assistant_item("item_root"), previous_item_id="root")

    assert [item.id for item in s.items] == ["item_root", "item_a"]


def test_insert_item_missing_previous_item_id_raises() -> None:
    s = AudioFullDuplexSessionState()

    with pytest.raises(ValueError, match="not found"):
        s.insert_item(_assistant_item("item_a"), previous_item_id="item_does_not_exist")


def test_insert_item_generates_id_when_none() -> None:
    s = AudioFullDuplexSessionState()

    pos = s.insert_item(_assistant_item(None))

    assert s.items[pos].id is not None
    assert s.items[pos].id.startswith("item_")


def test_insert_item_sets_object_and_status_defaults() -> None:
    s = AudioFullDuplexSessionState()

    s.insert_item(_assistant_item("item_a"))

    item = s.items[0]
    assert item.object == "realtime.item"
    assert item.status == "completed"


def test_insert_item_upserts_existing_id_in_place() -> None:
    """Re-inserting an item under an id already in history replaces it at
    its existing position instead of appending a duplicate -- this is what
    lets a client (or the server's own truncate handler) correct an item it
    already knows about without producing a second, conflicting entry."""
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))
    s.insert_item(_assistant_item("item_b"))
    s.insert_item(_assistant_item("item_c"))

    replacement = types.RealtimeConversationItemAssistantMessage(
        type="message",
        role="assistant",
        id="item_b",
        content=[{"type": "output_text", "text": "replaced"}],  # type: ignore[list-item]
    )
    idx = s.insert_item(replacement)

    assert idx == 1
    assert len(s.items) == 3
    assert [item.id for item in s.items] == ["item_a", "item_b", "item_c"]
    assert s.items[1].content[0].text == "replaced"


# ---------------------------------------------------------------------- #
#  remove_item                                                            #
# ---------------------------------------------------------------------- #


def test_remove_item_cleans_up_all_side_tables() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))
    s.item_duration_ms["item_a"] = 1234.0
    s.item_token_ids["item_a"] = [1, 2, 3]
    s.item_in_progress["item_a"] = True
    s.pending_truncations_ms["item_a"] = 500

    removed = s.remove_item("item_a")

    assert removed is not None
    assert removed.id == "item_a"
    assert s.items == []
    assert s.next_item_index == 0
    assert "item_a" not in s.item_duration_ms
    assert "item_a" not in s.item_token_ids
    assert "item_a" not in s.item_in_progress
    assert "item_a" not in s.pending_truncations_ms


def test_remove_item_missing_id_returns_none_and_is_a_noop() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))

    removed = s.remove_item("item_does_not_exist")

    assert removed is None
    assert [item.id for item in s.items] == ["item_a"]


# ---------------------------------------------------------------------- #
#  find_item / find_item_index                                            #
# ---------------------------------------------------------------------- #


def test_find_item_and_find_item_index_found() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))
    s.insert_item(_assistant_item("item_b"))

    assert s.find_item_index("item_b") == 1
    found = s.find_item("item_b")
    assert found is not None
    assert found.id == "item_b"


def test_find_item_and_find_item_index_not_found() -> None:
    s = AudioFullDuplexSessionState()
    s.insert_item(_assistant_item("item_a"))

    assert s.find_item_index("item_does_not_exist") is None
    assert s.find_item("item_does_not_exist") is None


# ---------------------------------------------------------------------- #
#  turn_detection / is_semantic_vad / is_manual_mode                      #
# ---------------------------------------------------------------------- #


def test_turn_detection_defaults_to_manual_mode() -> None:
    s = AudioFullDuplexSessionState()

    assert s.turn_detection is None
    assert s.is_semantic_vad is False
    assert s.is_manual_mode is True


def test_is_semantic_vad_true_when_configured() -> None:
    config = types.RealtimeSessionCreateRequest(
        type="realtime",
        audio={
            "input": {
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "medium",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
        },
    )
    s = AudioFullDuplexSessionState(config=config)

    assert s.is_semantic_vad is True
    assert s.is_manual_mode is False


def test_is_manual_mode_true_when_audio_configured_without_turn_detection() -> None:
    """audio.input present but turn_detection unset (None) must still count
    as manual mode -- exercises the turn_detection property's second guard
    (audio.input is not None) separately from the "no audio at all" case."""
    config = types.RealtimeSessionCreateRequest(
        type="realtime",
        audio={"input": {}},
    )
    s = AudioFullDuplexSessionState(config=config)

    assert s.turn_detection is None
    assert s.is_manual_mode is True
    assert s.is_semantic_vad is False


# ---------------------------------------------------------------------- #
#  _deep_merge / merge_session_config                                     #
# ---------------------------------------------------------------------- #


def test_deep_merge_recurses_into_nested_dicts() -> None:
    base = {"audio": {"input": {"a": 1, "b": 2}}}
    update = {"audio": {"input": {"b": 99}}}

    merged = _deep_merge(base, update)

    assert merged == {"audio": {"input": {"a": 1, "b": 99}}}


def test_deep_merge_overwrites_non_dict_values() -> None:
    base = {"instructions": "old", "count": 1}
    update = {"instructions": "new"}

    merged = _deep_merge(base, update)

    assert merged == {"instructions": "new", "count": 1}


def test_merge_session_config_preserves_unset_fields() -> None:
    current = types.RealtimeSessionCreateRequest(
        type="realtime",
        instructions="be nice",
        output_modalities=["audio"],
    )
    update = types.RealtimeSessionCreateRequest(type="realtime", instructions="be nicer")

    merged = merge_session_config(current, update)

    assert merged.instructions == "be nicer"
    assert merged.output_modalities == ["audio"]


def test_merge_session_config_deep_merges_nested_audio_config() -> None:
    """A session.update patching only turn_detection must not clobber a
    sibling nested field (e.g. format) already set under audio.input."""
    current = types.RealtimeSessionCreateRequest(
        type="realtime",
        audio={"input": {"format": {"type": "audio/pcm", "rate": 24000}}},
    )
    update = types.RealtimeSessionCreateRequest(
        type="realtime",
        audio={"input": {"turn_detection": {"type": "server_vad"}}},
    )

    merged = merge_session_config(current, update)

    assert merged.audio.input.turn_detection.type == "server_vad"
    assert merged.audio.input.format.rate == 24000
