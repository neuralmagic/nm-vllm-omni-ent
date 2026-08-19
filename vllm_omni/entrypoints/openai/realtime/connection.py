from __future__ import annotations

import asyncio
import base64
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams
from vllm.tool_parsers import ToolParserManager

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime import types
from vllm_omni.entrypoints.openai.realtime.session import (
    ActiveResponse,
    AudioFullDuplexSessionState,
    ResponseUsage,
    _gen_id,
    merge_session_config,
)

logger = init_logger(__name__)

SAMPLE_RATE_HZ = 24000
BYTES_PER_SAMPLE_PCM16 = 2

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"

# Qwen3-Omni talker/code2wav constant for correlating an assistant item's
# audio_end_ms (conversation.item.truncate) back to how many real thinker
# output tokens had actually been heard -- see
# FullDuplexRealtimeConnection._qwen3_omni_truncate_transcript for how it's
# used. MODEL-SPECIFIC, NOT PORTABLE, same caveat as
# _qwen3_omni_audio_token_count.
#
# Empirically calibrated, NOT derived from source: an earlier version of
# this assumed a fixed 80ms codec frame with exactly 1 real token consumed
# per talker decode step (read directly out of qwen3_omni.py's
# _get_talker_assistant_parts / _thinker_decode_to_talker_decode), which
# looked solid from the code but was wrong by ~4.8x in practice. Confirmed
# by direct measurement: a real production truncate at audio_end_ms=8808
# produced audio whose actual spoken content -- verified two independent
# ways, Whisper transcription of the truncated audio clip AND manual
# listening -- stopped exactly at token index 22 (23 tokens) of the
# stored (untruncated) transcript when tokenized with this model's own
# tokenizer. 8808 / 23 = 382.96ms/token, vs. the old formula's assumed
# 80ms/frame (which produced 105 tokens for that same audio_end_ms -- a
# ~4.7x overshoot, confirmed bit-for-bit: 8808 // 80 - 5 = 105). Based on
# a single calibration point; revisit if truncated transcripts still look
# off by a roughly constant factor.
QWEN3_OMNI_MS_PER_TOKEN = 383.0

# Default (truncation="auto") history-trimming thresholds, as a fraction of
# the input token limit (max_model_len minus any reserved output budget).
# Trigger early rather than waiting for the hard limit -- max_output_tokens
# defaults to "inf" (reserving nothing), so waiting until 100% full would
# leave zero guaranteed room for the response itself.
AUTO_TRUNCATION_TRIGGER_RATIO = 0.8
AUTO_TRUNCATION_TARGET_RATIO = 0.5

# A long response can stream audio deltas continuously for well over a
# minute while the client sends nothing back (it's just listening) --
# observed in production causing a network-level idle-connection drop
# (NAT/firewall between client and server, ~60s of client-side silence,
# no application error on either side) that looks like it was "caused" by
# an interruption but is really just whatever pause happened to follow one
# crossing an idle threshold. Starlette's WebSocket has no exposed
# protocol-level ping, so this sends a harmless, unrecognized-by-design
# JSON message purely to keep bytes flowing in both directions; real
# clients (confirmed against livekit's realtime plugin) silently ignore
# unknown event types.
KEEPALIVE_INTERVAL_SECONDS = 20


@dataclass
class _ToolParserRequest:
    """Duck-typed stand-in for the ChatCompletionRequest/ResponsesRequest
    that vLLM's ToolParser engines are normally driven by.

    Confirmed by inspection of vllm/parser/engine/parser_engine.py: the
    engine only ever reads `.skip_special_tokens` and `.include_reasoning`
    off this object (both used to gate output shape, not to constrain
    generation). We're only reusing the parser for text extraction, not
    vLLM's guided-decoding/structural-tag machinery, so a real request
    object isn't needed.
    """

    tools: list[dict] = field(default_factory=list)
    tool_choice: str = "auto"
    include_reasoning: bool = False
    skip_special_tokens: bool = True


class FullDuplexRealtimeConnection:
    """OpenAI /v1/realtime full-duplex WebSocket connection handler.

    One instance per WebSocket connection. This is a thin layer: it keeps
    conversation history (``session.items``) and in-flight staging buffers,
    and on every response.create it builds one complete prompt from that
    history plus whatever's newly buffered, and submits it as a brand-new,
    independent engine request. Cancelling a response is just aborting that
    request -- there is no persistent, continuously-edited request to
    surgically patch, so barge-in needs no KV-cache surgery.
    """

    def __init__(
        self,
        websocket: WebSocket,
        engine: AsyncOmni,
        model_name: str,
        supports_native_vad: bool = False,
        tool_call_parser: str | None = None,
        enable_auto_tool_choice: bool = False,
    ):
        self.ws = websocket
        self.engine = engine
        self.model_name = model_name
        self.supports_native_vad = supports_native_vad
        # Tool-call detection is opt-in via the same --tool-call-parser /
        # --enable-auto-tool-choice flags the other serving handlers already
        # use (see api_server.py) -- without both, tools are never parsed
        # out of the model's text even if a client declares them.
        self._tool_call_parser_name = tool_call_parser if enable_auto_tool_choice else None

        if supports_native_vad:
            initial_config = types.RealtimeSessionCreateRequest(
                type="realtime",
                model=model_name,
                output_modalities=["audio"],
                max_output_tokens="inf",
                truncation="auto",
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
        else:
            initial_config = types.RealtimeSessionCreateRequest(
                type="realtime",
                model=model_name,
                output_modalities=["audio"],
                max_output_tokens="inf",
                truncation="auto",
            )
        self.session = AudioFullDuplexSessionState(config=initial_config)

        self._connected = True
        self._response_task: asyncio.Task | None = None
        self._response_cancel_event = asyncio.Event()
        self._keepalive_task: asyncio.Task | None = None

        self._tokenizer: Any = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def handle_connection(self):
        await self.ws.accept()
        logger.info("[realtime] connection opened, session_id=%s", self.session.session_id)
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        try:
            await self._send_session_created()
            await self._send_conversation_created()
            self._tokenizer = await self._resolve_tokenizer()

            while self._connected:
                try:
                    text = await self.ws.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    event = types.client_event_adapter.validate_json(text)
                except Exception:
                    await self._send_error(
                        "Invalid or unrecognized client event",
                        "invalid_event",
                    )
                    continue
                await self._dispatch_event(event)
        except Exception:
            logger.exception("Unhandled error in realtime connection")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        self._connected = False
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        await self._cancel_active_response()
        logger.info("[realtime] connection closed, session_id=%s", self.session.session_id)

    async def _keepalive_loop(self) -> None:
        """Periodically send a harmless message to keep bytes flowing in
        both directions -- see KEEPALIVE_INTERVAL_SECONDS for why."""
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
                if not self._connected:
                    break
                await self._send_json({"type": "realtime.keepalive"})
        except asyncio.CancelledError:
            pass

    async def _resolve_tokenizer(self) -> Any:
        tokenizer = await self.engine.get_tokenizer()
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
        try:
            from vllm.transformers_utils.processor import cached_processor_from_config

            preprocessor = await self.engine.get_input_preprocessor()
            model_config = preprocessor.model_config
            processor = cached_processor_from_config(model_config)
            if getattr(processor, "apply_chat_template", None):
                return processor
        except Exception:
            logger.warning("Could not load processor for chat templating")
        return tokenizer

    # ------------------------------------------------------------------ #
    #  Event dispatch                                                     #
    # ------------------------------------------------------------------ #

    async def _dispatch_event(self, event: types.ClientEvent):
        handlers = {
            types.SessionUpdateEvent: self._handle_session_update,
            types.InputAudioBufferAppendEvent: self._handle_audio_append,
            types.InputAudioBufferCommitEvent: self._handle_audio_commit,
            types.InputAudioBufferClearEvent: self._handle_audio_clear,
            types.ResponseCreateEvent: self._handle_response_create,
            types.ResponseCancelEvent: self._handle_response_cancel,
            types.ConversationItemCreateEvent: self._handle_item_create,
            types.ConversationItemDeleteEvent: self._handle_item_delete,
            types.ConversationItemRetrieveEvent: self._handle_item_retrieve,
            types.ConversationItemTruncateEvent: self._handle_item_truncate,
        }
        handler = handlers.get(type(event))
        if handler is None:
            await self._send_error(
                f"Unknown event type: {event.type}",
                "invalid_event",
                event_id=event.event_id,
            )
            return
        try:
            await handler(event)
        except Exception as e:
            logger.exception("Error handling event %s", event.type)
            await self._send_error(str(e), "processing_error", event_id=event.event_id)

    # ------------------------------------------------------------------ #
    #  session.update                                                     #
    # ------------------------------------------------------------------ #

    async def _handle_session_update(self, event: types.SessionUpdateEvent):
        cfg = event.session
        s = self.session
        event_id = event.event_id

        if cfg.instructions is not None and s.instructions_locked:
            await self._send_error(
                "Cannot update instructions after audio output",
                "invalid_request_error",
                event_id=event_id,
            )
            return

        if cfg.audio is not None:
            inp = cfg.audio.input
            if inp is not None and inp.turn_detection is not None:
                td = inp.turn_detection
                if td.type == "server_vad":
                    await self._send_error(
                        "server_vad is not supported; use semantic_vad or null",
                        "invalid_request_error",
                        event_id=event_id,
                    )
                    return
                if td.type == "semantic_vad" and not self.supports_native_vad:
                    await self._send_error(
                        f"Model {s.config.model} does not support semantic_vad; use null",
                        "invalid_request_error",
                        event_id=event_id,
                    )
                    return

            out = cfg.audio.output
            if out is not None and out.voice is not None and s.has_output_audio:
                await self._send_error(
                    "Cannot change voice after audio output",
                    "invalid_request_error",
                    event_id=event_id,
                )
                return

        s.config = merge_session_config(s.config, cfg)
        await self._send_session_updated()

    # ------------------------------------------------------------------ #
    #  input_audio_buffer.append / .commit / .clear                       #
    # ------------------------------------------------------------------ #

    async def _handle_audio_append(self, event: types.InputAudioBufferAppendEvent):
        # Diagnostic: input_audio_buffer.append arrives at a roughly fixed
        # cadence (~every 100ms) while the client is streaming mic audio. A
        # gap much larger than that means our receive loop fell behind
        # draining the socket (e.g. blocked on something else), which is
        # exactly the precondition for uvicorn's ws_max_queue (default 32)
        # to overflow and force-close the connection with no application
        # error on either side.
        now = time.monotonic()
        last = getattr(self, "_last_append_wall_time", None)
        if last is not None:
            gap = now - last
            if gap > 0.5:
                logger.warning(
                    "[realtime] input_audio_buffer.append gap of %.2fs (receive loop may have stalled)",
                    gap,
                )
        self._last_append_wall_time = now

        if not event.audio:
            return
        try:
            audio_bytes = base64.b64decode(event.audio)
        except Exception:
            await self._send_error("Invalid base64 audio data", "invalid_request_error", event_id=event.event_id)
            return
        self.session.input_audio_buffer.extend(audio_bytes)

    def _commit_audio_buffer(self) -> types.RealtimeConversationItemUserMessage | None:
        """Turn whatever's in input_audio_buffer into a new ConversationItem.

        Per spec, committing always creates a distinct user message item --
        there's no deferral tied to whether some later response actually
        uses it. Returns None (no-op) if the buffer is empty.
        """
        s = self.session
        if len(s.input_audio_buffer) == 0:
            return None
        pcm16 = np.frombuffer(bytes(s.input_audio_buffer), dtype=np.int16)
        audio_f32 = pcm16.astype(np.float32) / 32768.0
        item = types.RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            status="completed",
            content=[{"type": "input_audio", "audio": self._pcm16_b64(audio_f32)}],
        )
        s.insert_item(item)
        s.input_audio_buffer.clear()
        return item

    async def _commit_audio_buffer_and_announce(
        self,
    ) -> types.RealtimeConversationItemUserMessage | None:
        """Commit input_audio_buffer and emit the resulting events.

        Shared by explicit input_audio_buffer.commit (manual turn detection)
        and the implicit commit at response.create time in semantic_vad mode.
        Returns None (nothing announced) if there was nothing buffered.
        """
        item = self._commit_audio_buffer()
        if item is None:
            return None
        idx = self.session.find_item_index(item.id)
        previous_item_id = self.session.items[idx - 1].id if idx else None
        await self._send_event(
            types.InputAudioBufferCommittedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.committed",
                item_id=item.id,
                previous_item_id=previous_item_id,
            )
        )
        # Per spec (conversation.item.created's own example for exactly this
        # "input audio buffer committed" case), the item on the wire has
        # content=[] -- the client already has the audio it just streamed,
        # so echoing the full buffer back is both unnecessary and, for a
        # long-uncommitted buffer, can produce a single WS message larger
        # than the client's own max_msg_size (confirmed in production: a
        # >60s buffer produced a ~4.66MB frame and the client killed the
        # connection). The real content stays in s.items (already inserted
        # by _commit_audio_buffer above) for prompt-building; only the wire
        # copy is stripped.
        wire_item = item.model_copy(update={"content": []})
        await self._send_event(
            types.ConversationItemCreatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.created",
                previous_item_id=previous_item_id,
                item=wire_item,  # type: ignore[arg-type]
            )
        )
        return item

    async def _handle_audio_commit(self, event: types.InputAudioBufferCommitEvent):
        s = self.session
        if s.is_semantic_vad:
            await self._send_error(
                "input_audio_buffer.commit is not allowed in semantic_vad mode",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return
        if len(s.input_audio_buffer) == 0:
            await self._send_error(
                "Input audio buffer is empty",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._commit_audio_buffer_and_announce()

    async def _handle_audio_clear(self, event: types.InputAudioBufferClearEvent):
        self.session.input_audio_buffer.clear()
        await self._send_event(
            types.InputAudioBufferClearedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.cleared",
            )
        )

    # ------------------------------------------------------------------ #
    #  response.create                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_tools_and_choice(s: AudioFullDuplexSessionState, response_cfg: Any):
        """Per-response overrides win over session.config, same pattern as
        output_modalities/max_output_tokens below."""
        tools = s.config.tools
        if response_cfg is not None and getattr(response_cfg, "tools", None) is not None:
            tools = response_cfg.tools
        tool_choice = s.config.tool_choice
        if response_cfg is not None and getattr(response_cfg, "tool_choice", None) is not None:
            tool_choice = response_cfg.tool_choice
        return tools, tool_choice

    async def _estimate_total_tokens(self, tools: list | None) -> int:
        """Real prompt token count for the current history, including audio
        expansion -- _build_full_prompt's own prompt_token_ids only count
        the literal <|audio_pad|> placeholder text (a handful of tokens),
        not what vLLM's multimodal processor actually expands it to at
        generation time (see _qwen3_omni_audio_token_count).

        Diagnostic: this runs apply_chat_template + tokenizer.encode over
        the *entire* history synchronously on the event loop (HF tokenizers
        aren't async-native) -- if this is slow and _maybe_truncate_history
        calls it repeatedly in its trim loop, that's exactly the kind of
        blocking that could stall the receive loop long enough to overflow
        uvicorn's ws_max_queue while the client keeps streaming audio.
        """
        t0 = time.monotonic()
        prompt = await self._build_full_prompt(tools=tools)
        total = len(prompt["prompt_token_ids"])
        audio_arrays = prompt.get("multi_modal_data", {}).get("audio", [])
        if audio_arrays:
            raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
            placeholder_len = len(raw_tok.encode(AUDIO_PLACEHOLDER, add_special_tokens=False))
            for arr, sr in audio_arrays:
                total += self._qwen3_omni_audio_token_count(arr.shape[0], sr) - placeholder_len
        elapsed = time.monotonic() - t0
        if elapsed > 0.05:
            logger.warning(
                "[realtime] _estimate_total_tokens took %.3fs (history=%d items, %d audio segments)",
                elapsed,
                len(self.session.items),
                len(audio_arrays),
            )
        return total

    async def _maybe_truncate_history(self, tools: list | None) -> bool:
        """Enforce session.config.truncation (spec: RealtimeTruncation).

        Returns False if truncation is `disabled` and history is already
        over the input token limit -- the caller must error out rather than
        generate. Otherwise trims s.items in place (oldest non-system item
        first -- system items fold into config.instructions and aren't part
        of the trimmable "post-instruction" budget) until under the
        retention target for whichever strategy is configured, emitting
        conversation.item.deleted for each removed item so client-side
        history (e.g. LiveKit's local chat_ctx) doesn't silently drift out
        of sync with ours -- exactly the kind of divergence that caused the
        update_chat_ctx "root" insertion bug earlier.

        Interpretation note: the spec describes truncated messages as "not
        included in the model's context", which could mean either (a) still
        retrievable but excluded from generation, or (b) actually removed.
        This implements (b) -- simplest given s.items doubles as both our
        history store and what _build_full_prompt renders -- so a
        conversation.item.retrieve for a trimmed item returns "not found".
        """
        s = self.session
        max_model_len = getattr(self.engine.model_config, "max_model_len", None)
        if not max_model_len:
            return True  # can't determine a budget; don't enforce blindly

        truncation = s.config.truncation or "auto"
        ratio = 1.0
        custom_limit = None
        if truncation == "disabled":
            mode = "disabled"
        elif truncation == "auto":
            mode = "auto"
        else:
            mode = "retention_ratio"
            ratio = truncation.retention_ratio
            if truncation.token_limits is not None:
                custom_limit = truncation.token_limits.post_instructions

        max_output_tokens = s.config.max_output_tokens
        reserved_output = max_output_tokens if isinstance(max_output_tokens, int) else 0
        limit = custom_limit if custom_limit is not None else max(0, max_model_len - reserved_output)

        if mode == "auto":
            # The client isn't guaranteed to set max_output_tokens (default
            # is "inf", reserving nothing), so "auto" can't just wait until
            # the hard limit to act -- that would leave zero room for the
            # response itself. Trigger early and trim down further, so
            # there's always real headroom left after trimming rather than
            # cutting it exactly at the wire every time.
            trigger = int(limit * AUTO_TRUNCATION_TRIGGER_RATIO)
            target = int(limit * AUTO_TRUNCATION_TARGET_RATIO)
        elif mode == "retention_ratio":
            trigger = limit
            target = int(limit * ratio)
        else:  # disabled
            trigger = limit
            target = limit

        total = await self._estimate_total_tokens(tools)
        if total <= trigger:
            return True
        if mode == "disabled":
            logger.warning(
                "[realtime] token budget exceeded (%d/%d) and truncation is disabled -- rejecting response.create",
                total,
                limit,
            )
            return False

        removed_ids: list[str] = []
        idx = 0
        while total > target and idx < len(s.items):
            item = s.items[idx]
            if getattr(item, "role", None) == "system":
                idx += 1
                continue

            # function_call/function_call_output must be removed together --
            # dropping just one leaves a dangling tool_calls reference with
            # no result, or a "tool" role message with no preceding call,
            # either of which is a malformed prompt. The pair isn't
            # necessarily adjacent or even in call-then-output order: a
            # client's update_chat_ctx push can insert a function_call_output
            # earlier in session.items than the function_call it answers
            # (observed in production -- see the earlier tool-calling
            # ordering investigation), so search the whole list by call_id
            # rather than assuming a fixed relative position.
            ids_to_remove = [item.id]
            if item.type in ("function_call", "function_call_output"):
                call_id = getattr(item, "call_id", None)
                pair = next(
                    (other for other in s.items if other.id != item.id and getattr(other, "call_id", None) == call_id),
                    None,
                )
                if pair is not None and pair.id is not None:
                    ids_to_remove.append(pair.id)

            for item_id in ids_to_remove:
                removed = s.remove_item(item_id)
                if removed is not None:
                    removed_ids.append(item_id)
                    await self._send_event(
                        types.ConversationItemDeletedEvent(
                            event_id=_gen_id("evt"),
                            type="conversation.item.deleted",
                            item_id=item_id,
                        )
                    )
            total = await self._estimate_total_tokens(tools)

        return True

    async def _handle_response_create(self, event: types.ResponseCreateEvent):
        s = self.session

        if s.is_semantic_vad:
            # There's no explicit commit in this mode (spec forbids it) --
            # the client's own VAD decided a turn just ended by calling
            # response.create, so whatever's been appended since the last
            # turn becomes its own committed item now.
            await self._commit_audio_buffer_and_announce()

        response_cfg = event.response
        response_id = _gen_id("resp")

        # Enforce session.config.truncation before committing to this
        # response at all -- disabled-mode overflow must reject the
        # response.create outright (no response.created), matching spec:
        # "the server will never truncate but would instead return an
        # error if the conversation exceeds the model's input token limit."
        precheck_tools, _ = self._resolve_tools_and_choice(s, response_cfg)
        if not await self._maybe_truncate_history(precheck_tools):
            await self._send_error(
                "The conversation exceeds the model's input token limit and truncation is disabled",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        output_modalities = s.config.output_modalities
        max_output_tokens = s.config.max_output_tokens
        if response_cfg is not None:
            if response_cfg.output_modalities is not None:
                output_modalities = response_cfg.output_modalities
            if response_cfg.max_output_tokens is not None:
                max_output_tokens = response_cfg.max_output_tokens

        response_obj = types.RealtimeResponse(
            id=response_id,
            object="realtime.response",
            status="in_progress",
            output=[],
            conversation_id=s.conversation_id,
            output_modalities=output_modalities,
            max_output_tokens=max_output_tokens,
            # Echo the client's response.metadata (e.g. client_event_id) back
            # on response.created -- clients like livekit-agents key a
            # per-call future off of it and time out if it never comes back.
            metadata=response_cfg.metadata if response_cfg is not None else None,
        )

        # Acknowledge every response.create immediately, before any
        # cancellation work below -- the client must always see
        # response.created for a response.create it sends, even if
        # cancelling the previous response fails.
        await self._send_event(
            types.ResponseCreatedEvent(
                event_id=_gen_id("evt"),
                type="response.created",
                response=response_obj,
            )
        )

        if s.active_response is not None:
            # Cancelling just aborts the interrupted response's own isolated
            # request. Every input_audio_buffer.commit already became a
            # permanent ConversationItem the moment it happened (see
            # _commit_audio_buffer), independent of whether any response
            # ever used it -- so the user never loses anything they said,
            # and the next response.create's _build_full_prompt naturally
            # picks it back up from session.items.
            await self._cancel_active_response()

        s.active_response = ActiveResponse(response_id=response_id, request_id=f"rt-{response_id}")

        self._response_cancel_event.clear()
        self._response_task = asyncio.create_task(self._run_response(response_id, response_cfg))

    async def _run_response(self, response_id: str, response_cfg: Any):
        s = self.session
        active = s.active_response
        if active is None:
            return

        try:
            await self._run_response_inner(response_id, response_cfg, s, active)
        except Exception:
            logger.exception("_run_response failed for %s", response_id)
        finally:
            s.active_response = None

    async def _run_response_inner(self, response_id, response_cfg, s, active):
        # Captured now, before anything else can mutate session.items (e.g.
        # a new input_audio_buffer.commit landing while this response is
        # still generating) -- this is where the item this response produces
        # actually belongs chronologically, not wherever s.items happens to
        # end at completion time.
        previous_item_id = s.items[-1].id if s.items else None

        modalities = s.config.output_modalities
        if response_cfg is not None and getattr(response_cfg, "output_modalities", None) is not None:
            modalities = response_cfg.output_modalities
        is_audio = "audio" in modalities

        tools, tool_choice = self._resolve_tools_and_choice(s, response_cfg)

        prompt = await self._build_full_prompt(tools=tools)

        item_id = _gen_id("item")
        output_index = 0

        converted_tools = self._convert_tools(tools) if tools else []
        tool_parser = None
        structural_tag_json = None
        if converted_tools and tool_choice != "none" and self._tool_call_parser_name:
            tool_parser_cls = ToolParserManager.get_tool_parser(self._tool_call_parser_name)
            strict_tools = [ChatCompletionToolsParam(**t) for t in self._convert_tools(tools, strict=True)]
            tool_parser = tool_parser_cls(self._tokenizer, tools=strict_tools)
            # Guided decoding: constrains the arguments JSON to each tool's
            # schema once the model emits <tool_call> (structural_tag_model
            # = "hermes" for this parser, matching this model's actual
            # <tool_call>{"name":..,"arguments":..}</tool_call> format --
            # see get_hermes_structural_tag in vLLM's
            # structural_tag_registry.py). Requires VLLM_ENFORCE_STRICT_TOOL_CALLING
            # (defaults True) and at least one strict tool for tool_choice="auto"
            # (get_model_structural_tag/_any_tool_strict) -- hence strict=True
            # above, since RealtimeFunctionTool has no strict field of its own.
            # Known gap: a named/forced tool_choice (RealtimeFunctionTool's
            # own ToolChoiceFunction/ToolChoiceMcp) isn't the same class as
            # the openai.types.responses ones xgrammar's dumper isinstance-
            # checks against, so it'd fall through to a generic dump that
            # may not match xgrammar's expected shape. Untested -- "auto"/
            # "none"/"required" (plain string comparisons) are unaffected.
            structure_tag = tool_parser.get_structural_tag(
                _ToolParserRequest(tools=strict_tools, tool_choice=tool_choice or "auto"),
                reasoning=False,
            )
            if structure_tag is not None:
                structural_tag_json = json.dumps(structure_tag.model_dump())

        item_obj = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status="in_progress",
            content=[],
        )

        # Inserted before the item's id is ever announced to the client
        # (silently -- no conversation.item.created yet, that still only
        # fires once at real completion below, matching today's
        # client-visible behavior) so a conversation.item.truncate/delete
        # arriving while this response is still generating has something to
        # act on. This must happen before the send below, not after: await
        # yields control to the event loop, so a client that reacts to
        # response.output_item.added fast enough can get its truncate
        # processed before the next line runs, and "not found" again --
        # confirmed happening in production logs with insert-after-send.
        s.insert_item(item_obj, previous_item_id=previous_item_id or "root")
        s.item_in_progress[item_id] = True

        await self._send_event(
            types.ResponseOutputItemAddedEvent(
                event_id=_gen_id("evt"),
                type="response.output_item.added",
                response_id=response_id,
                output_index=output_index,
                item=item_obj,  # type: ignore[arg-type]
            )
        )

        content_index = 0
        part_type = "audio" if is_audio else "text"
        part_obj = {"type": part_type, "text": "", "audio": "", "transcript": ""}

        await self._send_event(
            types.ResponseContentPartAddedEvent(
                event_id=_gen_id("evt"),
                type="response.content_part.added",
                response_id=response_id,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                part=part_obj,  # type: ignore[arg-type]
            )
        )

        full_text = ""
        full_transcript = ""
        full_token_ids: list[int] = []
        full_audio_chunks: list[np.ndarray] = []
        cancelled = False
        usage = ResponseUsage()
        total_audio_samples = 0

        text_finished = False
        audio_finished = False

        # Tool-call streaming state. previous_text/previous_token_ids track
        # the cumulative text/tokens the tool parser has seen so far --
        # extract_tool_calls_streaming needs both the previous and current
        # cumulative view to detect e.g. a <tool_call> tag boundary that
        # straddles two deltas. pending_tool_calls is keyed by the parser's
        # own per-call index (supports multiple simultaneous tool calls);
        # dict insertion order is used as emission/history order.
        previous_text = ""
        previous_token_ids: list[int] = []
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        next_output_index = 1  # 0 is the message item, reserved above
        # The talker has no concept of tool-call spans and will synthesize
        # audio for <tool_call>...</tool_call> text same as any other
        # content (confirmed: nothing in _thinker_to_talker_prefill excludes
        # it). We can't suppress that at the source without threading
        # tool-call boundaries down into the model's decode loop, so instead
        # stop forwarding audio deltas to the client once a call is
        # detected -- imperfect (some audio for the tag/name may already be
        # in flight) but eliminates the bulk of it, especially the
        # arguments, which stream in later deltas after detection.
        tool_call_seen = False

        async def emit_content_delta(piece: str) -> None:
            nonlocal full_text, full_transcript
            if not piece:
                return
            full_transcript += piece
            if is_audio:
                await self._send_event(
                    types.ResponseAudioTranscriptDeltaEvent(
                        event_id=_gen_id("evt"),
                        type="response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=piece,
                    )
                )
            else:
                full_text += piece
                await self._send_event(
                    types.ResponseTextDeltaEvent(
                        event_id=_gen_id("evt"),
                        type="response.output_text.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=piece,
                    )
                )

        async def handle_tool_parser_delta(delta_msg) -> None:
            nonlocal next_output_index, tool_call_seen
            if delta_msg is None:
                return
            if delta_msg.content:
                await emit_content_delta(delta_msg.content)
            for tc in delta_msg.tool_calls:
                entry = pending_tool_calls.get(tc.index)
                if entry is None:
                    tool_call_seen = True
                    entry = {
                        "item_id": _gen_id("item"),
                        "call_id": tc.id or _gen_id("call"),
                        "name": tc.function.name if tc.function else None,
                        "arguments": "",
                        "output_index": next_output_index,
                    }
                    pending_tool_calls[tc.index] = entry
                    next_output_index += 1
                    await self._send_event(
                        types.ResponseOutputItemAddedEvent(
                            event_id=_gen_id("evt"),
                            type="response.output_item.added",
                            response_id=response_id,
                            output_index=entry["output_index"],
                            item=types.RealtimeConversationItemFunctionCall(
                                type="function_call",
                                id=entry["item_id"],
                                call_id=entry["call_id"],
                                name=entry["name"] or "",
                                arguments="",
                                status="in_progress",
                            ),  # type: ignore[arg-type]
                        )
                    )
                elif not entry["name"] and tc.function and tc.function.name:
                    entry["name"] = tc.function.name

                if tc.function and tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
                    await self._send_event(
                        types.ResponseFunctionCallArgumentsDeltaEvent(
                            event_id=_gen_id("evt"),
                            type="response.function_call_arguments.delta",
                            response_id=response_id,
                            item_id=entry["item_id"],
                            output_index=entry["output_index"],
                            call_id=entry["call_id"],
                            delta=tc.function.arguments,
                        )
                    )

        sampling_params_list = list(self.engine.default_sampling_params_list)
        structural_tag_applied = False
        for sp in sampling_params_list:
            if isinstance(sp, SamplingParams):
                sp.output_kind = RequestOutputKind.DELTA
                # Only the first real (text/thinker) stage generates the
                # <tool_call> JSON that needs constraining -- later stages
                # (talker, codec) get their own SamplingParams untouched.
                if not structural_tag_applied and structural_tag_json is not None:
                    sp.structured_outputs = StructuredOutputsParams(structural_tag=structural_tag_json)
                    structural_tag_applied = True

        gen = self.engine.generate(
            prompt=prompt,
            request_id=active.request_id,
            sampling_params_list=sampling_params_list,
        )

        try:
            async for output in gen:
                if not self._connected:
                    cancelled = True
                    break

                output_type = getattr(output, "final_output_type", "text")
                first_out_dbg = output.outputs[0] if output.outputs else None

                if output_type == "audio":
                    audio_chunks = self._extract_audio_deltas(output)
                    for chunk in audio_chunks:
                        s.has_output_audio = True
                        s.instructions_locked = True
                        total_audio_samples += chunk.shape[0]
                        # Keep the full raw audio for the server-side history
                        # copy only (see stored_item below) -- this is for
                        # observability/debugging (e.g. replaying exactly
                        # what the model produced for a given item), never
                        # sent to the client as part of item history, since
                        # the client already received it live via the
                        # response.output_audio.delta events below and
                        # echoing a whole response's audio back in
                        # conversation.item.created would reproduce the
                        # oversized-message bug fixed for
                        # input_audio_buffer.commit.
                        full_audio_chunks.append(chunk)
                        # Keep counting samples (item_duration_ms/drop_message_item
                        # bookkeeping still needs to reflect that audio was
                        # really generated) but stop forwarding it to the
                        # client once a tool call has started -- see
                        # tool_call_seen's definition above for why.
                        if tool_call_seen:
                            continue
                        b64 = self._pcm16_b64(chunk)
                        await self._send_event(
                            types.ResponseAudioDeltaEvent(
                                event_id=_gen_id("evt"),
                                type="response.output_audio.delta",
                                response_id=response_id,
                                item_id=item_id,
                                output_index=output_index,
                                content_index=content_index,
                                delta=b64,
                            )
                        )
                    if first_out_dbg and first_out_dbg.finish_reason is not None:
                        audio_finished = True
                        if text_finished:
                            break
                    continue

                if output.outputs:
                    first_out = output.outputs[0]
                    delta_text = first_out.text or ""
                    delta_token_ids = list(first_out.token_ids)
                    usage.output_tokens += len(delta_token_ids)
                    # Raw thinker token stream, in talker-consumption order --
                    # this is what _qwen3_omni_truncate_transcript correlates
                    # against codec frames, independent of any tool-parser
                    # stripping applied to full_text/full_transcript below
                    # (the talker speaks the raw stream, tool tags included).
                    full_token_ids.extend(delta_token_ids)

                    if output.prompt_token_ids:
                        usage.input_tokens = max(usage.input_tokens, len(output.prompt_token_ids))

                    if tool_parser is not None:
                        # Additive branch: when no tools are configured for
                        # this response, tool_parser is None and this whole
                        # block is skipped -- the plain-text path below is
                        # untouched.
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + delta_token_ids
                        delta_msg = None
                        if delta_text or delta_token_ids:
                            delta_msg = tool_parser.extract_tool_calls_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                delta_token_ids,
                                request=_ToolParserRequest(tools=converted_tools, tool_choice=tool_choice or "auto"),
                            )
                        previous_text = current_text
                        previous_token_ids = current_token_ids
                        await handle_tool_parser_delta(delta_msg)
                    elif delta_text:
                        await emit_content_delta(delta_text)

                    finish = first_out.finish_reason
                    if finish is not None:
                        text_finished = True
                        if not is_audio or audio_finished:
                            break
        except asyncio.CancelledError:
            cancelled = True
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.debug("Error closing generator for %s", active.request_id, exc_info=True)

        if tool_parser is not None and getattr(tool_parser, "engine_based_streaming", False):
            # finish_streaming() only exists on the newer ParserEngine-based
            # parsers (engine_based_streaming=True, e.g. Qwen3EngineToolParser)
            # -- the base ToolParser class legacy regex-based parsers extend
            # (e.g. Hermes2ProToolParser) don't declare it at all and would
            # raise AttributeError here (confirmed in production logs).
            await handle_tool_parser_delta(tool_parser.finish_streaming())

        if self._response_cancel_event.is_set():
            cancelled = True

        status = "cancelled" if cancelled else "completed"

        if is_audio:
            await self._send_event(
                types.ResponseAudioDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_audio.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                )
            )
            await self._send_event(
                types.ResponseAudioTranscriptDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_audio_transcript.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    transcript=full_transcript,
                )
            )
        else:
            await self._send_event(
                types.ResponseTextDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_text.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    text=full_text,
                )
            )

        # Reconstruct rather than mutate item_obj.status/.content in place:
        # pydantic does not validate/coerce plain attribute assignment after
        # construction, so `item_obj.content = [{...}]` would silently leave
        # item_obj.content holding raw dicts instead of Content objects --
        # _assistant_item_text's attribute-based access (part.transcript)
        # then always returns None for those, so no assistant turn's text
        # ever actually made it into a later response's prompt.
        item_obj = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status="completed" if not cancelled else "incomplete",
            content=(
                [{"type": "output_audio", "transcript": full_transcript}]  # type: ignore[list-item]
                if is_audio
                else [{"type": "output_text", "text": full_text}]  # type: ignore[list-item]
            ),
        )

        done_part = {"type": part_type, "text": full_text, "transcript": full_transcript}
        await self._send_event(
            types.ResponseContentPartDoneEvent(
                event_id=_gen_id("evt"),
                type="response.content_part.done",
                response_id=response_id,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                part=done_part,  # type: ignore[arg-type]
            )
        )

        await self._send_event(
            types.ResponseOutputItemDoneEvent(
                event_id=_gen_id("evt"),
                type="response.output_item.done",
                response_id=response_id,
                output_index=output_index,
                item=item_obj,  # type: ignore[arg-type]
            )
        )

        # Per spec, response.done always includes every output item that was
        # generated, regardless of final status -- so the item exists in
        # history either way. For a cancelled response we can't know how
        # much of it the user actually heard (no alignment between audio
        # timing and text/token position -- see conversation.item.truncate),
        # so rather than guess, the history copy gets empty content; the
        # wire events above already carried the real accumulated content.
        history_item = item_obj
        if cancelled:
            history_item = types.RealtimeConversationItemAssistantMessage(
                type="message",
                role="assistant",
                id=item_obj.id,
                status="incomplete",
                content=(
                    [{"type": "output_audio", "transcript": ""}]  # type: ignore[list-item]
                    if is_audio
                    else [{"type": "output_text", "text": ""}]  # type: ignore[list-item]
                ),
            )

        # A response that only called tools has no message item in `output`
        # (matches real OpenAI behavior) -- drop the placeholder rather than
        # keep an empty message. Scoped to a response that actually
        # completed: a cancelled response keeps today's empty-incomplete
        # message-item behavior regardless of any in-flight tool call, since
        # cancellation-mid-tool-call isn't handled specially here.
        #
        # total_audio_samples == 0 is required, not just empty text: the
        # tool parser classifies raw <tool_call>...</tool_call> text as
        # "not content", but the talker has no concept of that span and
        # synthesizes audio for the whole segment regardless (nothing in
        # _thinker_to_talker_prefill special-cases tool-call text) -- so a
        # "no content" tool-call response can still have real audio that was
        # actually streamed to and played by the client. Dropping the
        # message item in that case orphans that audio: a later
        # conversation.item.truncate against it fails with "not found"
        # (confirmed in production logs, audio_end_ms=6201 on an item that
        # had already been dropped here).
        drop_message_item = (
            not cancelled
            and total_audio_samples == 0
            and not (full_text or full_transcript)
            and bool(pending_tool_calls)
        )

        # The placeholder inserted at response.output_item.added time is
        # still there unless the client explicitly removed it mid-stream
        # (conversation.item.truncate/delete racing ahead of our own
        # cancellation, which is normal -- see _handle_response_create). If
        # so, respect that instead of resurrecting it with the same id.
        # chain_after tracks the most recently inserted history item so
        # function-call items below chain onto the right predecessor whether
        # the message item was kept, dropped, or never made it in at all.
        # Server-side-only copy of history_item carrying the full raw audio
        # the model generated for this item, for observability/debugging
        # (e.g. replaying exactly what was produced for a given turn) --
        # never sent over the wire (see full_audio_chunks.append above for
        # why): conversation.item.created below still uses the audio-less
        # history_item, only s.items/the /tmp dump gets this richer copy.
        stored_item = history_item
        if is_audio and full_audio_chunks:
            # Constructing via model_copy(update=...) here (like item_obj
            # above) would NOT validate/coerce stored_content's raw dicts
            # into Content objects -- _assistant_item_text's attribute-based
            # access (part.transcript) would then silently return None for
            # this item on every later prompt build, exactly the bug class
            # called out at item_obj's construction above. Use the
            # validating constructor instead.
            full_audio_b64 = self._pcm16_b64(np.concatenate(full_audio_chunks))
            stored_content = [
                part.model_dump() | ({"audio": full_audio_b64} if getattr(part, "type", None) == "output_audio" else {})
                for part in history_item.content
            ]
            stored_item = types.RealtimeConversationItemAssistantMessage(
                type="message",
                role="assistant",
                id=history_item.id,
                status=history_item.status,
                content=stored_content,  # type: ignore[arg-type]
            )

        chain_after = previous_item_id
        if s.find_item_index(item_id) is not None:
            if drop_message_item:
                s.remove_item(item_id)
            else:
                s.insert_item(stored_item, previous_item_id=previous_item_id or "root")
                if item_obj.id:
                    s.item_duration_ms[item_obj.id] = total_audio_samples / SAMPLE_RATE_HZ * 1000
                    # Skip storing for tool-call responses: full_token_ids is
                    # the raw thinker stream (tool-call tags included), but
                    # this item's transcript/text is the tool-parser-stripped
                    # content -- the two no longer line up token-for-token,
                    # so _qwen3_omni_truncate_transcript falls back to
                    # blanking (today's behavior) rather than risk splicing
                    # raw <tool_call> text into a truncated transcript.
                    if not pending_tool_calls:
                        s.item_token_ids[item_obj.id] = full_token_ids
                await self._send_event(
                    types.ConversationItemCreatedEvent(
                        event_id=_gen_id("evt"),
                        type="conversation.item.created",
                        previous_item_id=previous_item_id,
                        item=history_item,  # type: ignore[arg-type]
                    )
                )
                chain_after = history_item.id

            # Only now -- after item_duration_ms/item_token_ids are finally
            # populated (or the item is gone, for drop_message_item) -- is
            # it safe to resolve a conversation.item.truncate that arrived
            # while this response was still item_in_progress (see
            # _handle_item_truncate). Clearing item_in_progress first means
            # a truncate arriving from here on goes straight through
            # _handle_item_truncate's normal, non-deferred path.
            s.item_in_progress.pop(item_id, None)
            pending_ms = s.pending_truncations_ms.get(item_id)
            if pending_ms is not None:
                await self._do_item_truncate(
                    types.ConversationItemTruncateEvent(
                        event_id=_gen_id("evt"),
                        type="conversation.item.truncate",
                        item_id=item_id,
                        content_index=0,
                        audio_end_ms=pending_ms,
                    )
                )

        # Function calls aren't subject to the truncate-race protection the
        # message item needed above -- per spec only assistant message items
        # can ever be truncated, so there's no client action that could race
        # ahead and remove one of these before we get here.
        function_call_items: list[types.RealtimeConversationItemFunctionCall] = []
        for entry in pending_tool_calls.values():
            await self._send_event(
                types.ResponseFunctionCallArgumentsDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.function_call_arguments.done",
                    response_id=response_id,
                    item_id=entry["item_id"],
                    output_index=entry["output_index"],
                    call_id=entry["call_id"],
                    name=entry["name"] or "",
                    arguments=entry["arguments"],
                )
            )
            fc_item = types.RealtimeConversationItemFunctionCall(
                type="function_call",
                id=entry["item_id"],
                call_id=entry["call_id"],
                name=entry["name"] or "",
                arguments=entry["arguments"],
                status="completed" if not cancelled else "incomplete",
            )
            await self._send_event(
                types.ResponseOutputItemDoneEvent(
                    event_id=_gen_id("evt"),
                    type="response.output_item.done",
                    response_id=response_id,
                    output_index=entry["output_index"],
                    item=fc_item,  # type: ignore[arg-type]
                )
            )
            s.insert_item(fc_item, previous_item_id=chain_after or "root")
            await self._send_event(
                types.ConversationItemCreatedEvent(
                    event_id=_gen_id("evt"),
                    type="conversation.item.created",
                    previous_item_id=chain_after,
                    item=fc_item,  # type: ignore[arg-type]
                )
            )
            chain_after = fc_item.id
            function_call_items.append(fc_item)

        usage.total_tokens = usage.input_tokens + usage.output_tokens

        status_details = None
        if cancelled:
            status_details = {
                "type": "cancelled",
                "reason": "client_cancelled",
            }

        output_items: list[Any] = [] if drop_message_item else [item_obj]
        output_items.extend(function_call_items)

        done_response = types.RealtimeResponse(
            id=response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,  # type: ignore[arg-type]
            output=output_items,  # type: ignore[arg-type]
            conversation_id=s.conversation_id,
            output_modalities=s.config.output_modalities,
            max_output_tokens=s.config.max_output_tokens,
            usage={  # type: ignore[arg-type]
                "total_tokens": usage.total_tokens,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )

        await self._send_event(
            types.ResponseDoneEvent(
                event_id=_gen_id("evt"),
                type="response.done",
                response=done_response,
            )
        )

    # ------------------------------------------------------------------ #
    #  response.cancel                                                    #
    # ------------------------------------------------------------------ #

    async def _handle_response_cancel(self, event: types.ResponseCancelEvent):
        if self.session.active_response is None:
            await self._send_error(
                "No response is in progress",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return
        await self._cancel_active_response()

    async def _cancel_active_response(self) -> None:
        active = self.session.active_response
        if active is None:
            return
        self._response_cancel_event.set()
        # engine.abort() tears down orchestrator/stage-pool bookkeeping for
        # the request, but never pushes a completion sentinel through the
        # per-request output queue -- so it does NOT by itself unblock
        # _run_response_inner's `async for output in gen:` loop, which would
        # otherwise wait on that queue forever. Cancelling the task is what
        # actually stops it (injects CancelledError at the current await),
        # which _run_response_inner catches to still emit response.done.
        try:
            await self.engine.abort(active.request_id)
        except Exception:
            logger.exception("Failed to abort request %s", active.request_id)
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ #
    #  conversation.item.create                                           #
    # ------------------------------------------------------------------ #

    async def _handle_item_create(self, event: types.ConversationItemCreateEvent):
        item = event.item

        try:
            s = self.session
            pos = s.insert_item(item, event.previous_item_id)
        except ValueError as e:
            await self._send_error(str(e), "invalid_request_error", event_id=event.event_id)
            return
        # Reflects the item's actual final position -- correct whether this
        # was a fresh insert (per previous_item_id) or an upsert-in-place
        # (previous_item_id ignored, item already existed elsewhere).
        prev_id = s.items[pos - 1].id if pos > 0 else None

        if isinstance(item, types.RealtimeConversationItemSystemMessage):
            for part in item.content:
                if part.type == "input_text" and part.text:
                    self.session.config = merge_session_config(
                        self.session.config,
                        types.RealtimeSessionCreateRequest(
                            type="realtime",
                            instructions=part.text,
                        ),
                    )

        await self._send_event(
            types.ConversationItemCreatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.created",
                previous_item_id=prev_id,
                item=item,  # type: ignore[arg-type]
            )
        )

    # ------------------------------------------------------------------ #
    #  conversation.item.delete                                           #
    # ------------------------------------------------------------------ #

    async def _handle_item_delete(self, event: types.ConversationItemDeleteEvent):
        item_id = event.item_id

        # Per spec: "Send this event when you want to remove any item from
        # the conversation history" -- no positional restriction, the only
        # failure case is the item not existing.
        removed = self.session.remove_item(item_id)
        if removed is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        await self._send_event(
            types.ConversationItemDeletedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.deleted",
                item_id=item_id,
            )
        )

    # ------------------------------------------------------------------ #
    #  conversation.item.retrieve                                         #
    # ------------------------------------------------------------------ #

    async def _handle_item_retrieve(self, event: types.ConversationItemRetrieveEvent):
        item_id = event.item_id

        item = self.session.find_item(item_id)
        if item is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        wire_item = item
        if isinstance(item, types.RealtimeConversationItemAssistantMessage):
            # The stored item may carry a server-side-only "audio" field on
            # its output_audio content (see stored_item in
            # _run_response_inner) -- never send that back over the wire:
            # it exists only for observability/the /tmp dump, and echoing a
            # whole response's audio back here would reproduce the
            # oversized-message bug fixed for input_audio_buffer.commit.
            wire_item = item.model_copy(
                update={
                    "content": [
                        part.model_copy(update={"audio": None}) if getattr(part, "audio", None) else part
                        for part in item.content
                    ]
                }
            )

        await self._send_json(
            {
                "event_id": _gen_id("evt"),
                "type": "conversation.item.retrieved",
                "item": wire_item.model_dump(exclude_none=True),
            }
        )

    # ------------------------------------------------------------------ #
    #  conversation.item.truncate                                         #
    # ------------------------------------------------------------------ #

    async def _handle_item_truncate(self, event: types.ConversationItemTruncateEvent):
        item_id = event.item_id
        s = self.session

        item = s.find_item(item_id)
        if item is None:
            await self._send_error(
                f"Item '{item_id}' not found",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        if not isinstance(item, types.RealtimeConversationItemAssistantMessage):
            await self._send_error(
                "Can only truncate assistant messages",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        # Record intent unconditionally. If the item's own response is still
        # generating (item_in_progress), item_duration_ms/item_token_ids
        # don't exist yet and the placeholder's content is still empty --
        # applying the truncation now would be a silent no-op, and
        # _run_response_inner's own finalization write would clobber it
        # afterward regardless (blank-on-cancel, or the full untruncated
        # content if the response isn't cancelled and keeps generating past
        # this point). Deferring to finalization -- the one place
        # item_token_ids is finally complete -- closes that race regardless
        # of which order the two arrive in; see _run_response_inner's own
        # check of this dict at finalization time.
        s.pending_truncations_ms[item_id] = event.audio_end_ms

        if not s.item_in_progress.get(item_id, False):
            await self._do_item_truncate(event)

    async def _do_item_truncate(self, event: types.ConversationItemTruncateEvent) -> None:
        """Apply a conversation.item.truncate once the target item is no
        longer in progress -- called either directly from
        _handle_item_truncate (item already finalized) or from
        _run_response_inner's finalization (item was still in progress when
        the truncate first arrived)."""
        s = self.session
        item_id = event.item_id
        content_index = event.content_index
        # A second truncate may have arrived (and overwritten
        # pending_truncations_ms) while the first was still waiting on this
        # same in-progress item -- always honor the latest recorded value
        # rather than the one on the event that happened to trigger this call.
        audio_end_ms = s.pending_truncations_ms.pop(item_id, event.audio_end_ms)

        item = s.find_item(item_id)
        if item is None:
            # Deleted/removed between the truncate request and its
            # resolution (e.g. conversation.item.delete raced ahead, or
            # drop_message_item removed it in _run_response_inner) --
            # nothing left to truncate.
            return
        if not isinstance(item, types.RealtimeConversationItemAssistantMessage):
            return

        duration_ms = s.item_duration_ms.get(item_id)
        if duration_ms is not None and audio_end_ms > duration_ms:
            await self._send_error(
                f"audio_end_ms ({audio_end_ms}) is greater than the actual audio duration",
                "invalid_request_error",
                event_id=event.event_id,
            )
            return

        # Per spec, truncating audio must not leave text in context the user
        # never heard -- but rather than blanking the transcript entirely
        # (the previous, spec-minimum behavior), reconstruct the prefix that
        # *was* actually heard using Qwen3-Omni's fixed talker text/frame
        # correlation (see _qwen3_omni_truncate_transcript). This keeps the
        # model's own memory of what it said in sync with what the user
        # actually heard, instead of wiping it and confusing later turns.
        # Falls back to "" (today's behavior) when we don't have a captured
        # token stream for this item (e.g. it included a tool call).
        truncated_text = self._qwen3_omni_truncate_transcript(item_id, audio_end_ms)
        new_content = list(item.content)
        if 0 <= content_index < len(new_content):
            part = new_content[content_index]
            if hasattr(part, "transcript"):
                # Preserve the stored observability audio (see stored_item in
                # _run_response_inner) across truncation -- truncate only
                # concerns the transcript's context-consistency with what was
                # heard, not the debugging record of what was generated.
                stored_audio = getattr(part, "audio", None)
                new_content[content_index] = {
                    "type": "output_audio",
                    "transcript": truncated_text,
                    **({"audio": stored_audio} if stored_audio else {}),
                }
            elif hasattr(part, "text"):
                new_content[content_index] = {"type": "output_text", "text": truncated_text}
        truncated_item = types.RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            id=item_id,
            status=item.status,
            content=new_content,  # type: ignore[arg-type]
        )
        s.insert_item(truncated_item)

        await self._send_event(
            types.ConversationItemTruncatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.truncated",
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        )

    # ------------------------------------------------------------------ #
    #  History -> prompt serialization                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_tools(tools: list, *, strict: bool = False) -> list[dict]:
        """RealtimeFunctionTool is flat (type/name/description/parameters);
        apply_chat_template(tools=...) and ChatCompletionToolsParam both
        expect the chat-completions nested shape. MCP tools
        (RealtimeResponseCreateMcpTool) are not handled -- skipped.

        strict=True marks every tool as OpenAI chat-completions "strict"
        function calling -- RealtimeFunctionTool has no such field of its
        own, but vLLM's structural-tag builder
        (get_model_structural_tag/_any_tool_strict) only builds a
        tool_choice="auto" grammar when at least one tool is strict, so
        callers building guided-decoding tools need this set. Left False
        for the copy handed to apply_chat_template, which shouldn't show
        clients an OpenAI-specific field their own tool declaration never
        asked for.
        """
        converted = []
        for tool in tools:
            if getattr(tool, "type", None) != "function":
                continue
            function: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            if strict:
                function["strict"] = True
            converted.append({"type": "function", "function": function})
        return converted

    def _assistant_item_text(self, item: types.RealtimeConversationItemAssistantMessage) -> str:
        """Return this item's transcript/text as currently stored.

        conversation.item.truncate rewrites the transcript/text content in
        place (see _handle_item_truncate/_qwen3_omni_truncate_transcript), so
        a truncated item's stored text already reflects only what the client
        actually heard -- nothing to trim here.
        """
        for part in item.content:
            text = getattr(part, "transcript", None) or getattr(part, "text", None)
            if text:
                return text
        return ""

    def _qwen3_omni_truncate_transcript(self, item_id: str, audio_end_ms: float) -> str:
        """Reconstruct the portion of an assistant item's transcript whose
        audio had actually finished playing by audio_end_ms.

        MODEL-SPECIFIC, NOT PORTABLE (see _qwen3_omni_audio_token_count for
        the equivalent input-side caveat). Uses a flat empirically-
        calibrated ms/token rate (QWEN3_OMNI_MS_PER_TOKEN) -- see that
        constant's own comment for how it was measured and why the earlier,
        more "principled"-looking frame-count derivation from reading
        qwen3_omni.py's talker/code2wav code was actually wrong by ~4.8x.

        Falls back to "" (today's spec-minimum blank-out) if we never
        captured this item's raw token stream -- currently true for any
        response that included a tool call, where the talker's raw text
        (tool-call tags included) no longer lines up token-for-token with
        the tool-parser-stripped transcript we'd be reconstructing from.
        """
        token_ids = self.session.item_token_ids.get(item_id)
        if not token_ids:
            return ""
        # round(), not int()/floor(): QWEN3_OMNI_MS_PER_TOKEN is itself
        # rounded from the measured 382.96, so floor-dividing would
        # silently undercount by 1 right at the calibration point itself
        # (int(8808 / 383.0) == 22, not the actual 23) -- confirmed while
        # writing this.
        tokens_heard = round(audio_end_ms / QWEN3_OMNI_MS_PER_TOKEN)
        tokens_heard = min(tokens_heard, len(token_ids))
        if tokens_heard <= 0:
            return ""
        raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        return raw_tok.decode(token_ids[:tokens_heard], skip_special_tokens=True)

    @staticmethod
    def _qwen3_omni_audio_token_count(num_samples: int, sample_rate: int) -> int:
        """Exact number of prompt tokens a single audio segment expands to.

        MODEL-SPECIFIC, NOT PORTABLE: this mirrors the formula vLLM's own
        Qwen3-Omni multimodal processor uses to expand the <|audio_pad|>
        placeholder to its real token count
        (vllm.model_executor.models.qwen3_omni_moe_thinker
        ._get_feat_extract_output_lengths, confirmed against
        WhisperFeatureExtractor: sampling_rate=16000, hop_length=160 -> 100
        mel-spectrogram frames/sec, feature_extraction_whisper.py's own
        comment confirms frame count is exactly samples_16k // hop_length,
        no rounding ambiguity). It encodes THIS model's specific audio
        encoder downsampling ratios -- there is no shared vLLM-wide formula
        for "how many tokens will this audio become". A different omni
        model needs its own version of this function; look for that
        model's multimodal processor's own audio-placeholder-expansion
        logic (grep for get_num_audio_tokens / feat_extract_output_lengths
        under vllm_omni/model_executor/models/<model>/) rather than reusing
        this one.
        """
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            _get_feat_extract_output_lengths,
        )

        num_samples_16k = round(num_samples * 16000 / sample_rate)
        input_lengths = num_samples_16k // 160
        output_lengths = _get_feat_extract_output_lengths(input_lengths)
        return int(output_lengths) + 2  # + <|audio_start|> + <|audio_end|>

    async def _build_full_prompt(self, tools: list | None = None) -> TokensPrompt:
        """Render system instructions + conversation history (session.items)
        into one prompt for a brand-new engine request.

        Every response.create rebuilds this from scratch -- there's no
        persistent request whose KV cache carries history forward, so this
        function *is* the connection's memory of the conversation.
        """
        from vllm.inputs import TokensPrompt

        s = self.session
        messages: list[dict[str, Any]] = []
        audio_arrays: list[tuple[np.ndarray, int]] = []
        converted_tools = self._convert_tools(tools) if tools else None

        if s.config.instructions:
            messages.append({"role": "system", "content": s.config.instructions})

        for item in s.items:
            if item.type == "function_call":
                # "" not None: this model's chat template only handles
                # message.content as a string or a list (line 55/59 of its
                # Jinja template unconditionally iterates non-string content
                # assuming it's a list) -- None crashes with "'NoneType'
                # object is not iterable" (confirmed in production logs).
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {"name": item.name, "arguments": item.arguments},
                            }
                        ],
                    }
                )
                continue
            if item.type == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item.call_id, "content": item.output})
                continue

            role = getattr(item, "role", None)
            if role == "user":
                parts_text = []
                for part in item.content:
                    if part.type == "input_audio" and part.audio:
                        parts_text.append(AUDIO_PLACEHOLDER)
                        audio_bytes = base64.b64decode(part.audio)
                        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_arrays.append((pcm16.astype(np.float32) / 32768.0, SAMPLE_RATE_HZ))
                    elif part.type == "input_text" and part.text:
                        parts_text.append(part.text)
                if parts_text:
                    messages.append({"role": "user", "content": "".join(parts_text)})
            elif role == "assistant":
                text = self._assistant_item_text(item)
                if text:
                    messages.append({"role": "assistant", "content": text})
            # System-role items are tracked in s.items for spec-correct
            # retrieve/list ordering, but _handle_item_create already folds
            # their content into s.config.instructions (the leading system
            # message above), so they're intentionally skipped here to avoid
            # emitting the same instructions twice.

        chat_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
        if converted_tools:
            chat_template_kwargs["tools"] = converted_tools
        text = self._tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        raw_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        token_ids = raw_tok.encode(text, add_special_tokens=False)

        prompt_data = TokensPrompt(prompt_token_ids=token_ids)
        if audio_arrays:
            prompt_data["multi_modal_data"] = {"audio": audio_arrays}
        return prompt_data

    # ------------------------------------------------------------------ #
    #  Audio output processing                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    def _extract_audio_deltas(self, output) -> list[np.ndarray]:
        """Return this engine step's new audio samples, as-is.

        Qwen3-Omni's code2wav already returns non-overlapping increments per
        step -- both chunked_decode and chunked_decode_streaming
        (qwen3_omni_code2wav.py) explicitly slice the left-context overlap
        *out* of their own output before returning it (see
        `wav_chunk[..., context_size * self.total_upsample:]` and the
        `start = left_context_size * total_upsample - tail` slice
        respectively), specifically so nothing downstream has to re-derive
        what's new. This used to also diff each array against a stored
        reference of the previous one (np.allclose with a loose tolerance)
        as a defensive guard -- removed because it was solving a problem
        the model already solves, and had a real failure mode: two
        independent (already-correct) increments landing within tolerance
        of each other by coincidence -- most likely on quiet/near-silent
        passages -- would make it wrongly treat the second one as
        "old prefix + new tail" and silently drop the portion it mistook
        for overlap, corrupting playback for the rest of that response.
        """
        from collections.abc import Mapping

        mm = getattr(output, "multimodal_output", None)
        if mm is None or not isinstance(mm, Mapping):
            return []

        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return []

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []

        if isinstance(raw_audio, (list, tuple)):
            if raw_audio:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.append(arr)
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.append(arr)
        return chunks

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode()

    # ------------------------------------------------------------------ #
    #  Server event emission                                              #
    # ------------------------------------------------------------------ #

    async def _send_event(self, event) -> None:
        if not self._connected:
            return
        try:
            if hasattr(event, "model_dump"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Pydantic serializer warnings",
                        category=UserWarning,
                    )
                    data = event.model_dump(mode="json", exclude_none=True)
            else:
                data = event
            await self.ws.send_text(json.dumps(data))
        except Exception:
            logger.warning("[realtime] send failed, marking connection dead: %s", data.get("type"), exc_info=True)
            self._connected = False

    async def _send_json(self, payload: dict) -> None:
        if not self._connected:
            return
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception:
            logger.warning("[realtime] send failed, marking connection dead: %s", payload.get("type"), exc_info=True)
            self._connected = False

    async def _send_error(
        self,
        message: str,
        code: str = "server_error",
        event_id: str | None = None,
    ) -> None:
        error_data: dict[str, Any] = {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "param": None,
            "event_id": event_id,
        }
        await self._send_event(
            types.RealtimeErrorEvent(
                event_id=_gen_id("evt"),
                type="error",
                error=error_data,  # type: ignore[arg-type]
            )
        )

    async def _send_session_created(self) -> None:
        session_obj = self._build_session_object()
        await self._send_event(
            types.SessionCreatedEvent(
                event_id=_gen_id("evt"),
                type="session.created",
                session=session_obj,  # type: ignore[arg-type]
            )
        )

    async def _send_session_updated(self) -> None:
        session_obj = self._build_session_object()
        await self._send_event(
            types.SessionUpdatedEvent(
                event_id=_gen_id("evt"),
                type="session.updated",
                session=session_obj,  # type: ignore[arg-type]
            )
        )

    async def _send_conversation_created(self) -> None:
        await self._send_event(
            types.ConversationCreatedEvent(
                event_id=_gen_id("evt"),
                type="conversation.created",
                conversation={  # type: ignore[arg-type]
                    "id": self.session.conversation_id,
                    "object": "realtime.conversation",
                },
            )
        )

    def _build_session_object(self) -> dict[str, Any]:
        s = self.session
        obj = s.config.model_dump(exclude_none=True)
        obj["object"] = "realtime.session"
        obj["id"] = s.session_id
        obj["expires_at"] = int(s.expires_at)
        return obj
