# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 Thinker-to-Talker and Talker-to-Code2Wav bridges."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)
_MINICPMO45_ASYNC_STATE = "_minicpmo45_async_codec_state"
_MINICPMO45_STREAM_RECORD = "_minicpmo45_async_stream_record"
_MINICPMO45_SILENCE_CODE = 4218


class _MiniCPMO45MetaStruct(MetaStruct):
    """Model-owned metadata for the split Talker-to-Code2Wav bridge."""

    ref_audio_sr: int | None = None
    llm_output_text_utf8: torch.Tensor | None = None
    segment_end: bool | None = None
    turn_end: bool | None = None
    tts_is_last_chunk: bool | None = None


def _extract_first_audio_ref(multi_modal_data):
    if not isinstance(multi_modal_data, dict):
        return None
    audio_data = multi_modal_data.get("audio")
    if audio_data is None:
        return None
    if isinstance(audio_data, list):
        if not audio_data:
            return None
        audio_data = audio_data[0]

    samples = None
    sample_rate = None
    if isinstance(audio_data, tuple) and len(audio_data) >= 2:
        samples, sample_rate = audio_data[0], audio_data[1]
    elif isinstance(audio_data, dict):
        sample_rate = audio_data.get("sample_rate") or audio_data.get("sampling_rate") or audio_data.get("sr")
        for key in ("audio", "wav", "samples", "array", "waveform"):
            if key in audio_data:
                samples = audio_data[key]
                break
    if samples is None or sample_rate is None:
        return None

    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.ndim > 1:
        if waveform.shape[0] <= 2 and waveform.shape[-1] > waveform.shape[0]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=-1)
    return waveform.reshape(-1).cpu(), int(sample_rate)


def _coerce_token_id_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _to_transport_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _coerce_int(value):
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        value = flat[0].item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _codec_config(transfer_manager: Any) -> tuple[int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    config = raw_config.get("extra", raw_config) if isinstance(raw_config, dict) else {}
    config = config if isinstance(config, dict) else {}
    chunk_frames = int(config.get("codec_chunk_frames", 25))
    left_context_frames = int(config.get("codec_left_context_frames", 3))
    if chunk_frames <= 0 or left_context_frames < 0:
        raise ValueError(
            "Invalid MiniCPM-o codec chunk config: "
            f"codec_chunk_frames={chunk_frames}, "
            f"codec_left_context_frames={left_context_frames}"
        )
    return chunk_frames, left_context_frames


def _codec_scalars(value: Any) -> list[int]:
    """Normalize one request-routed codec delta to CPU scalar token IDs."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return []
        return value.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scalars: list[int] = []
        for item in value:
            scalars.extend(_codec_scalars(item))
        return scalars
    if isinstance(value, (int, bool)):
        return [int(value)]
    raise TypeError(f"Unsupported MiniCPM-o codec delta type: {type(value).__name__}")


def _extract_codec_delta(pooling_output: Any, request_id: str) -> list[int]:
    if pooling_output is None:
        return []
    if isinstance(pooling_output, Mapping):
        meta = pooling_output.get("meta")
        routed_id = meta.get("request_id") if isinstance(meta, Mapping) else None
        routed_id = routed_id or pooling_output.get("request_id")
        if routed_id is not None and str(routed_id) != request_id:
            return []
        codes = pooling_output.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else pooling_output.get("codes.audio")
        return _codec_scalars(audio)
    if isinstance(pooling_output, Sequence) and not isinstance(
        pooling_output,
        (str, bytes, bytearray),
    ):
        delta: list[int] = []
        for item in pooling_output:
            values = _extract_codec_delta(item, request_id) if isinstance(item, Mapping) else _codec_scalars(item)
            delta.extend(values)
        return delta
    return _codec_scalars(pooling_output)


def _drop_codec_state(transfer_manager: Any, request_id: str) -> None:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if isinstance(request_payload, dict):
        container = request_payload.get(request_id)
        if isinstance(container, dict):
            container.pop(_MINICPMO45_ASYNC_STATE, None)
            if not container:
                request_payload.pop(request_id, None)
        else:
            request_payload.pop(request_id, None)
    code_accumulators = getattr(transfer_manager, "code_prompt_token_ids", None)
    if hasattr(code_accumulators, "pop"):
        code_accumulators.pop(request_id, None)


def _is_aborted(request: Any) -> bool:
    status_name = getattr(getattr(request, "status", None), "name", "")
    return any(marker in status_name for marker in ("ABORT", "CANCEL", "IGNORED", "ERROR"))


def tts2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Stream request-owned MiniCPM-o codec windows to Code2Wav."""
    external_id = getattr(request, "external_req_id", None)
    internal_id = getattr(request, "request_id", None)
    request_id = str(external_id if external_id is not None else internal_id)
    internal_id = str(internal_id if internal_id is not None else request_id)
    output_meta = multimodal_output.get("meta") if isinstance(multimodal_output, Mapping) else None
    output_meta = output_meta if isinstance(output_meta, Mapping) else {}
    segment_text_utf8 = output_meta.get("llm_output_text_utf8")
    if not isinstance(segment_text_utf8, torch.Tensor):
        segment_text_utf8 = None
    turn_end = bool(_coerce_int(output_meta.get("turn_end")))

    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    container = request_payload.get(request_id)
    if not isinstance(container, dict):
        container = {}
        request_payload[request_id] = container

    record = container.get(_MINICPMO45_STREAM_RECORD)
    if not isinstance(record, dict):
        record = {
            "internal_id": internal_id,
            "cache_epoch": 0,
            "chunk_seq": 0,
            "retired_internal_ids": set(),
        }
        container[_MINICPMO45_STREAM_RECORD] = record
    elif internal_id in record["retired_internal_ids"]:
        return None
    elif record["internal_id"] != internal_id:
        record["retired_internal_ids"].add(record["internal_id"])
        record["internal_id"] = internal_id
        record["cache_epoch"] = int(record["cache_epoch"]) + 1
        record["chunk_seq"] = 0
        _drop_codec_state(transfer_manager, request_id)

    if _is_aborted(request):
        record["retired_internal_ids"].add(internal_id)
        _drop_codec_state(transfer_manager, request_id)
        return None

    state = container.get(_MINICPMO45_ASYNC_STATE)
    if not isinstance(state, dict):
        state = {
            "internal_id": internal_id,
            "pending": [],
            "left_context": [],
            "codec_end": 0,
        }
        container[_MINICPMO45_ASYNC_STATE] = state

    pending = state["pending"]
    pending.extend(_extract_codec_delta(multimodal_output, request_id))
    request_finished = getattr(request, "is_finished", None)
    finished = bool(is_finished or (callable(request_finished) and request_finished()))
    chunk_frames, left_context_frames = _codec_config(transfer_manager)
    flush_pending = finished
    last_chunk = bool(flush_pending)
    if not flush_pending and len(pending) < chunk_frames:
        return None

    new_token_count = len(pending) if flush_pending else chunk_frames
    new_codes = pending[:new_token_count]
    del pending[:new_token_count]
    codec_start = int(state["codec_end"])
    codec_end = codec_start + new_token_count

    if new_token_count:
        if codec_start == 0:
            context = [_MINICPMO45_SILENCE_CODE] * left_context_frames
        else:
            context = list(state["left_context"])
        output_codes = [*context, *new_codes]
        history = output_codes
        state["left_context"] = history[-left_context_frames:] if left_context_frames else []
    elif last_chunk and codec_start > 0 and state["left_context"]:
        context = list(state["left_context"])
        output_codes = context
    else:
        context = []
        output_codes = []
    state["codec_end"] = codec_end
    code_flat_numel = len(output_codes)
    if flush_pending and not last_chunk and code_flat_numel == 0:
        # Keep the generic generation connector model-agnostic: a real token
        # makes this control-only TTS boundary schedulable, while the explicit
        # zero length tells Code2Wav to discard the placeholder.
        output_codes = [0]

    if last_chunk:
        record["retired_internal_ids"].add(internal_id)
        _drop_codec_state(transfer_manager, request_id)

    chunk_seq = int(record["chunk_seq"])
    record["chunk_seq"] = chunk_seq + 1
    ref_audio = None
    ref_audio_sr = None
    if int(record["cache_epoch"]) == 0 and chunk_seq == 0:
        request_info = getattr(request, "additional_information", None)
        if isinstance(request_info, Mapping):
            codes_info = request_info.get("codes")
            meta_info = request_info.get("meta")
            raw_ref_audio = codes_info.get("ref") if isinstance(codes_info, Mapping) else None
            raw_ref_audio_sr = meta_info.get("ref_audio_sr") if isinstance(meta_info, Mapping) else None
            ref_audio_sr = _coerce_int(raw_ref_audio_sr)
            if raw_ref_audio is not None:
                ref_audio = torch.as_tensor(raw_ref_audio, dtype=torch.float32).reshape(-1).cpu()
    finished_tensor = torch.tensor(last_chunk, dtype=torch.bool)
    payload = OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(output_codes, dtype=torch.long),
            ref=ref_audio,
        ),
        meta=_MiniCPMO45MetaStruct(
            request_id=request_id,
            chunk_seq=chunk_seq,
            cache_epoch=int(record["cache_epoch"]),
            code_flat_numel=code_flat_numel,
            codec_chunk_frames=new_token_count,
            codec_left_context_frames=len(context),
            left_context_size=len(context),
            last_chunk=last_chunk,
            stream_finished=finished_tensor,
            finished=finished_tensor,
            is_segment_finished=finished_tensor,
            req_id=[request_id],
            llm_output_text_utf8=segment_text_utf8,
            tts_is_last_chunk=flush_pending,
            turn_end=turn_end and last_chunk,
            ref_audio_sr=ref_audio_sr,
        ),
        request_id=request_id,
    )
    return payload


def tts2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
) -> OmniPayloadStruct:
    """Build one terminal Code2Wav payload when async chunks are disabled."""
    external_id = getattr(request, "external_req_id", None)
    internal_id = getattr(request, "request_id", None)
    request_id = str(external_id if external_id is not None else internal_id)
    codes = _extract_codec_delta(pooling_output, request_id)
    _, left_context_frames = _codec_config(transfer_manager)
    context = [_MINICPMO45_SILENCE_CODE] * left_context_frames if codes else []
    output_codes = [*context, *codes]

    request_info = getattr(request, "additional_information", None)
    if not isinstance(request_info, Mapping):
        request_info = {}
    codes_info = request_info.get("codes")
    if not isinstance(codes_info, Mapping):
        codes_info = {}
    meta_info = request_info.get("meta")
    if not isinstance(meta_info, Mapping):
        meta_info = {}
    ref_audio = codes_info.get("ref")
    finished = torch.tensor(True, dtype=torch.bool)
    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(output_codes, dtype=torch.long),
            ref=torch.as_tensor(ref_audio, dtype=torch.float32).reshape(-1) if ref_audio is not None else None,
        ),
        meta=_MiniCPMO45MetaStruct(
            request_id=request_id,
            chunk_seq=0,
            cache_epoch=0,
            code_flat_numel=len(output_codes),
            codec_chunk_frames=len(codes),
            codec_left_context_frames=len(context),
            left_context_size=len(context),
            last_chunk=True,
            stream_finished=finished,
            finished=finished,
            req_id=[request_id],
            ref_audio_sr=_coerce_int(meta_info.get("ref_audio_sr")),
            segment_end=bool(meta_info.get("segment_end", False)),
            turn_end=bool(meta_info.get("turn_end", False)),
            tts_is_last_chunk=True,
        ),
        request_id=request_id,
    )


def tts2code2wav_token_only(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build the sync Code2Wav placeholder; codec data arrives by connector."""
    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        codes = multimodal_output.get("codes") if isinstance(multimodal_output, Mapping) else None
        audio = (
            codes.get("audio")
            if isinstance(codes, Mapping)
            else multimodal_output.get("codes.audio")
            if isinstance(multimodal_output, Mapping)
            else None
        )
        codec_codes = _codec_scalars(audio)
        if not codec_codes:
            continue
        prompt_len = len(codec_codes) + 3
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def _special_token_ids_from_mm_output(mm_output):
    if not isinstance(mm_output, Mapping):
        return {}
    meta = mm_output.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    special_token_ids = mm_output.get("special_token_ids")
    if not isinstance(special_token_ids, dict):
        special_token_ids = {}
    flat_meta = {
        key.removeprefix("meta."): value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.")
    }
    return {
        key: value
        for key, value in (
            (key, _coerce_int(value))
            for source in (special_token_ids, meta, flat_meta)
            for key, value in source.items()
        )
        if value is not None and value >= 0
    }


def _build_tts_scheduler_prompt_token_ids(
    tts_token_ids: torch.Tensor | None,
    llm_output_ids: list[int],
    prompt_token_ids: list[int],
) -> list[int]:
    if tts_token_ids is not None:
        ids = _coerce_token_id_list(tts_token_ids)
        if ids:
            return ids
    if llm_output_ids:
        return llm_output_ids
    if prompt_token_ids:
        return prompt_token_ids[-1:]
    raise ValueError("MiniCPM-o TTS stage requires at least one scheduler prompt token")


def llm2tts(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    _streaming_context=None,
):
    """Build Talker conditioning for ordinary streaming."""
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {}
    for llm_output, p in zip(llm_outputs, prompt):
        if isinstance(p, dict):
            multi_modal_data[llm_output.request_id] = p.get("multi_modal_data", None)
        else:
            multi_modal_data[llm_output.request_id] = getattr(p, "multi_modal_data", None)

    for llm_output in llm_outputs:
        output = llm_output.outputs[0]
        request_mm_output = getattr(llm_output, "multimodal_output", None)
        completion_mm_output = getattr(output, "multimodal_output", None)
        if isinstance(request_mm_output, Mapping):
            mm_output = request_mm_output
        elif isinstance(completion_mm_output, Mapping):
            mm_output = completion_mm_output
        else:
            mm_output = {}
        special_token_ids = _special_token_ids_from_mm_output(mm_output)
        prompt_token_ids = list(llm_output.prompt_token_ids)
        llm_output_ids = getattr(output, "token_ids", None)
        cumulative_output_ids = getattr(output, "cumulative_token_ids", None)
        if llm_output_ids is None:
            llm_output_ids = cumulative_output_ids or []
        elif cumulative_output_ids is not None and len(cumulative_output_ids) > len(llm_output_ids):
            # DELTA outputs (any streaming client) carry only the newest tokens,
            # while the forwarded hidden states span prompt + whole reply. With
            # the delta alone, tts_bos lands at the end of full_token_ids, the
            # tts_bos..tts_eos slice collapses to nothing and the Talker is
            # conditioned on an empty span, so the reply is silent.
            llm_output_ids = cumulative_output_ids
        # Always copy: CompletionOutput.token_ids can alias the upstream
        # detokenizer's live token list. Forwarding that exact object as the
        # talker prompt makes the stage-1 streaming update extend the list
        # with itself (state and update bind the same object), doubling the
        # thinker's recorded output every segment until the TTS engine input
        # buffer overflows.
        llm_output_ids = list(llm_output_ids)
        thinker_text = getattr(output, "text", "") or ""
        prompt_token_ids_len = len(prompt_token_ids)

        latent = mm_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.detach()
        if thinker_hidden_states.ndim == 3 and thinker_hidden_states.shape[0] == 1:
            thinker_hidden_states = thinker_hidden_states.squeeze(0)

        # Build full token sequence and extract TTS region
        full_token_ids = prompt_token_ids + llm_output_ids

        tts_bos_id = special_token_ids.get("tts_bos_token_id")
        tts_end_ids: set[int] = set()

        # Plain-chat (use_tts_template) fallback: requests do not
        # surface special_token_ids, so use MiniCPM-o 4.5's fixed boundaries.
        if tts_bos_id is None:
            tts_bos_id = 151703
            tts_end_ids = {151704, 151645}

        tts_bos_idx = None
        for idx_t in range(len(full_token_ids)):
            if full_token_ids[idx_t] == tts_bos_id:
                tts_bos_idx = idx_t + 1
        if tts_bos_idx is None and llm_output_ids:
            # Audio routing is a model-stage concern, not an OpenAI serving
            # default. Plain chat templates do not include <|tts_bos|>; in
            # that case condition the Talker on the generated assistant span.
            tts_bos_idx = prompt_token_ids_len

        tts_eos_idx = None
        if tts_bos_idx is not None:
            for idx_t in range(tts_bos_idx, len(full_token_ids)):
                if full_token_ids[idx_t] in tts_end_ids:
                    tts_eos_idx = idx_t
                    break

        tts_token_ids_slice = tts_hidden_slice = None
        if tts_bos_idx is not None and thinker_hidden_states.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else thinker_hidden_states.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = thinker_hidden_states[tts_bos_idx:end_idx].to(torch.float32).contiguous()
        handoff_ids = _coerce_token_id_list(tts_token_ids_slice) if tts_token_ids_slice is not None else None
        model_intermediate_buffer: dict[str, object] = {
            "request_id": str(llm_output.request_id),
            "ids": {"prompt": prompt_token_ids, "output": llm_output_ids},
            "llm_output_text": thinker_text,
            "meta": {},
        }
        req_mm_data = multi_modal_data.get(llm_output.request_id)
        ref_audio = _extract_first_audio_ref(req_mm_data)
        if ref_audio is not None:
            ref_waveform, ref_sr = ref_audio
            model_intermediate_buffer.setdefault("codes", {})["ref"] = _to_transport_list(ref_waveform)
            model_intermediate_buffer.setdefault("meta", {})["ref_audio_sr"] = ref_sr
        handoff_hidden = _to_transport_list(tts_hidden_slice) if tts_hidden_slice is not None else None
        model_intermediate_buffer.setdefault("ids", {})["tts"] = handoff_ids
        model_intermediate_buffer.setdefault("hidden", {})["tts"] = handoff_hidden

        if handoff_ids is not None and handoff_hidden is not None:
            condition_length = max(len(handoff_ids), len(handoff_hidden)) + 2
            scheduler_prompt_token_ids = [0] * condition_length
            handoff_meta = model_intermediate_buffer.setdefault("meta", {})
            handoff_meta["next_stage_prompt_len"] = condition_length
            handoff_meta["replace_streaming_prompt"] = True
        else:
            scheduler_prompt_token_ids = _build_tts_scheduler_prompt_token_ids(
                tts_token_ids_slice,
                llm_output_ids,
                prompt_token_ids,
            )
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=scheduler_prompt_token_ids,
                model_intermediate_buffer=model_intermediate_buffer,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return tts_inputs
