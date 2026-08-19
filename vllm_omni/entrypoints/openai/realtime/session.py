from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from vllm_omni.entrypoints.openai.realtime import types


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:24]}"


@dataclass(slots=True)
class ResponseUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ActiveResponse:
    """Handle for the response currently being generated.

    ``request_id`` is the engine request backing this response -- each
    response.create submits a brand-new, independent request (see
    FullDuplexRealtimeConnection._run_response_inner), so cancelling one is
    just ``engine.abort(request_id)``.
    """

    response_id: str
    request_id: str


def _default_config() -> types.RealtimeSessionCreateRequest:
    return types.RealtimeSessionCreateRequest(
        type="realtime",
        output_modalities=["audio"],
        max_output_tokens="inf",
        truncation="auto",
    )


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_session_config(
    current: types.RealtimeSessionCreateRequest,
    update: types.RealtimeSessionCreateRequest,
) -> types.RealtimeSessionCreateRequest:
    base = current.model_dump(exclude_none=True)
    patch = update.model_dump(exclude_none=True)
    return types.RealtimeSessionCreateRequest.model_validate(_deep_merge(base, patch))


@dataclass
class AudioFullDuplexSessionState:
    session_id: str = field(default_factory=lambda: _gen_id("sess"))
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 1800)

    config: types.RealtimeSessionCreateRequest = field(default_factory=_default_config)

    conversation_id: str = field(default_factory=lambda: _gen_id("conv"))
    items: list[types.ConversationItemModel] = field(default_factory=list)
    next_item_index: int = 0

    item_duration_ms: dict[str, float] = field(default_factory=dict)
    """Actual audio duration of each assistant message item (total samples
    generated / SAMPLE_RATE_HZ), used to validate conversation.item.truncate's
    audio_end_ms (spec: "If the audio_end_ms is greater than the actual audio
    duration, the server will respond with an error")."""

    item_token_ids: dict[str, list[int]] = field(default_factory=dict)
    """Raw thinker output token ids for each assistant message item, in the
    same order the talker consumed them to synthesize audio. Lets
    conversation.item.truncate reconstruct the portion of the transcript
    whose audio had actually finished playing, instead of blanking it
    entirely -- see FullDuplexRealtimeConnection._qwen3_omni_truncate_transcript.
    Only populated for responses with no tool calls (see that method for why);
    absent for others, which fall back to today's spec-minimum blank-out."""

    item_in_progress: dict[str, bool] = field(default_factory=dict)
    """True from the moment an assistant message item's placeholder is
    inserted (response.output_item.added) until FullDuplexRealtimeConnection
    ._run_response_inner finalizes it. A conversation.item.truncate that
    arrives while this is True has nothing to act on yet (item_duration_ms/
    item_token_ids aren't populated, and the placeholder's content is still
    empty) -- see pending_truncations_ms."""

    pending_truncations_ms: dict[str, int] = field(default_factory=dict)
    """audio_end_ms from a conversation.item.truncate that arrived for an
    item still in item_in_progress. _run_response_inner checks this at
    finalization time (once item_token_ids/item_duration_ms are finally
    populated) and resolves it then, instead of the truncate silently doing
    nothing and finalization's own write clobbering it afterward -- a race
    that goes the other way too (an uncancelled response finishing normally
    would otherwise overwrite an already-applied truncate with the full,
    untruncated content)."""

    input_audio_buffer: bytearray = field(default_factory=bytearray)
    input_audio_speech_active: bool = False
    input_audio_speech_start_ms: int | None = None

    output_audio_buffer: bytes = b""
    output_audio_response_id: str | None = None

    active_response: ActiveResponse | None = None

    has_output_audio: bool = False
    instructions_locked: bool = False

    @property
    def turn_detection(self) -> Any:
        audio = self.config.audio
        if audio is None or audio.input is None:
            return None
        return audio.input.turn_detection

    @property
    def is_semantic_vad(self) -> bool:
        td = self.turn_detection
        if td is None:
            return False
        return getattr(td, "type", None) == "semantic_vad"

    @property
    def is_manual_mode(self) -> bool:
        return self.turn_detection is None

    def find_item_index(self, item_id: str) -> int | None:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return i
        return None

    def find_item(self, item_id: str) -> types.ConversationItemModel | None:
        idx = self.find_item_index(item_id)
        return self.items[idx] if idx is not None else None

    def insert_item(self, item: types.ConversationItemModel, previous_item_id: str | None = None) -> int:
        if item.id is not None:
            existing_idx = self.find_item_index(item.id)
            if existing_idx is not None:
                # Upsert: replace in place at its existing position rather
                # than inserting a second entry. This is what lets a client
                # correct an item it already knows about -- e.g. LiveKit's
                # update_chat_ctx pushing its own better-informed view of an
                # interrupted turn (it tracks real playback position; we
                # don't) -- instead of producing a duplicate, conflicting
                # entry under the same id.
                if item.object is None:
                    item.object = "realtime.item"
                if item.status is None:
                    item.status = "completed"
                self.items[existing_idx] = item
                return existing_idx

        if item.id is None:
            item.id = _gen_id("item")
        if item.object is None:
            item.object = "realtime.item"
        if item.status is None:
            item.status = "completed"

        if previous_item_id is None:
            pos = len(self.items)
        elif previous_item_id == "root":
            pos = 0
        else:
            idx = self.find_item_index(previous_item_id)
            if idx is None:
                raise ValueError(f"previous_item_id '{previous_item_id}' not found")
            pos = idx + 1

        self.items.insert(pos, item)
        self.next_item_index = len(self.items)
        return pos

    def remove_item(self, item_id: str) -> types.ConversationItemModel | None:
        idx = self.find_item_index(item_id)
        if idx is None:
            return None
        item = self.items.pop(idx)
        if item.id:
            self.item_duration_ms.pop(item.id, None)
            self.item_token_ids.pop(item.id, None)
            self.item_in_progress.pop(item.id, None)
            self.pending_truncations_ms.pop(item.id, None)
        self.next_item_index = len(self.items)
        return item
