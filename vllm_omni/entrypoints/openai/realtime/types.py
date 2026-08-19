# ruff: noqa: F401
from __future__ import annotations

from typing import Annotated, Union

from openai.types.realtime import (
    ConversationCreatedEvent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemRetrieveEvent,
    ConversationItemTruncatedEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
    RealtimeErrorEvent,
    RealtimeResponse,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from pydantic import Field, TypeAdapter

# MCP tools (RealtimeResponseCreateMcpTool / ToolChoiceMcp) are out of scope
# for this handler -- only RealtimeFunctionTool function calling is wired up.
ConversationItemModel = (
    RealtimeConversationItemSystemMessage
    | RealtimeConversationItemUserMessage
    | RealtimeConversationItemAssistantMessage
    | RealtimeConversationItemFunctionCall
    | RealtimeConversationItemFunctionCallOutput
)

ClientEvent = Annotated[
    SessionUpdateEvent
    | InputAudioBufferAppendEvent
    | InputAudioBufferCommitEvent
    | InputAudioBufferClearEvent
    | ResponseCreateEvent
    | ResponseCancelEvent
    | ConversationItemCreateEvent
    | ConversationItemDeleteEvent
    | ConversationItemRetrieveEvent
    | ConversationItemTruncateEvent,
    Field(discriminator="type"),
]

client_event_adapter: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)
