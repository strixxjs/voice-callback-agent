"""LiveKit voice agent worker entrypoint."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from typing import Any

from google.protobuf.duration_pb2 import Duration
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    AutoSubscribe,
    CloseEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    llm,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import RoomOptions

from agent.extraction import (
    build_transcript_text,
    extract_answers,
    save_call_summary,
)
from agent.metrics import ActiveModels, SessionMetricsCollector
from agent.prompts import ScriptConfig, build_system_prompt, load_script
from agent.telephony import (
    extract_job_phone_number,
    mask_phone_number,
    validate_e164,
)
from agent.tools import SessionState, end_call, terminate_call
from app.config import settings

logger = logging.getLogger("agent.worker")


def prewarm(proc: JobProcess) -> None:
    """Pre-warm worker process: load VAD and import heavy AI provider packages.

    This runs during worker startup, keeping heavy imports off the real-time event loop
    and out of the first conversational turn.
    """
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()

    provider = settings.llm_provider.lower()
    if provider == "google":
        import google.genai  # noqa: F401
        from livekit.plugins import google  # noqa: F401
    elif provider == "openai":
        from livekit.plugins import openai  # noqa: F401

    if settings.stt_provider.lower() == "deepgram":
        from livekit.plugins import deepgram  # noqa: F401

    if settings.tts_provider.lower() == "elevenlabs":
        from livekit.plugins import elevenlabs  # noqa: F401


def _create_llm_instance(provider: str, model: str) -> Any:
    """Create a single LLM instance for a given provider and model."""
    if provider == "google":
        from livekit.plugins import google

        api_key = settings.google_api_key or os.environ.get("GOOGLE_API_KEY")
        kwargs: dict[str, Any] = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return google.LLM(**kwargs)
    elif provider == "openai":
        from livekit.plugins import openai

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return openai.LLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


DEFAULT_ELEVENLABS_VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "llm": {
        "google": "gemini-3.5-flash-lite",
        "openai": "gpt-4o-mini",
    },
    "stt": {
        "deepgram": "nova-3",
        "openai": "whisper-1",
    },
    "tts_model": {
        "elevenlabs": "eleven_turbo_v2_5",
        "openai": "tts-1",
    },
    "tts_voice": {
        "elevenlabs": DEFAULT_ELEVENLABS_VOICE_ID,
        "openai": "alloy",
    },
}


def resolve_llm_model(provider: str | None = None, model: str | None = None) -> str:
    """Resolve active LLM model name from explicit config or provider default."""
    if model:
        return model
    prov = (provider or settings.llm_provider).lower()
    return PROVIDER_DEFAULTS["llm"].get(prov, "gemini-3.5-flash-lite")


def resolve_stt_model(provider: str | None = None, model: str | None = None) -> str:
    """Resolve active STT model name from explicit config or provider default."""
    if model:
        return model
    prov = (provider or settings.stt_provider).lower()
    return PROVIDER_DEFAULTS["stt"].get(prov, "nova-3")


def resolve_tts_model(provider: str | None = None, model: str | None = None) -> str:
    """Resolve active TTS model name from explicit config or provider default."""
    if model:
        return model
    prov = (provider or settings.tts_provider).lower()
    return PROVIDER_DEFAULTS["tts_model"].get(prov, "eleven_turbo_v2_5")


def resolve_tts_voice(provider: str | None = None, voice_id: str | None = None) -> str:
    """Resolve active TTS voice ID from explicit config or provider default."""
    if voice_id:
        return voice_id
    prov = (provider or settings.tts_provider).lower()
    return PROVIDER_DEFAULTS["tts_voice"].get(prov, DEFAULT_ELEVENLABS_VOICE_ID)


def _sync_google_thought_signatures(primary_llm: Any, fallback_llm: Any) -> None:
    """Share thought signatures dictionary between Google LLM instances.

    Why:
        Gemini 2.5+ and 3+ models require thought signatures on function call parts
        in multi-turn history (Google API rejects missing signatures with HTTP 400).
        In livekit-plugins-google (livekit/plugins/google/llm.py:318, 574, 881),
        thought signatures are stored in `self._thought_signatures: dict[str, bytes]`.
        When FallbackAdapter switches to fallback_llm, fallback_llm starts with an
        empty dict and fails with HTTP 400 on any turn following previous tool calls.
        Sharing the dictionary object ensures the fallback retains all thought
        signatures from earlier turns.

    Warning:
        Relies on private attribute `_thought_signatures` in
        livekit.plugins.google.llm.LLM. Must be rechecked on plugin upgrades.
    """
    if not hasattr(primary_llm, "_thought_signatures") or not hasattr(
        fallback_llm, "_thought_signatures"
    ):
        logger.warning(
            "Cannot sync Google thought signatures: '_thought_signatures' attribute "
            "missing on primary (%s) or fallback (%s) LLM. "
            "Check livekit-plugins-google.",
            type(primary_llm).__name__,
            type(fallback_llm).__name__,
        )
        return

    fallback_llm._thought_signatures = primary_llm._thought_signatures


def get_llm() -> Any:
    """Lazily instantiate configured LLM, optionally wrapped in FallbackAdapter."""
    provider = settings.llm_provider.lower()
    primary_model = resolve_llm_model(provider, settings.llm_model)
    primary_llm = _create_llm_instance(provider, primary_model)

    if settings.llm_fallback_model:
        from livekit.agents.llm import FallbackAdapter

        fallback_llm = _create_llm_instance(provider, settings.llm_fallback_model)
        if provider == "google":
            _sync_google_thought_signatures(primary_llm, fallback_llm)

        logger.info(
            "Configuring LLM with FallbackAdapter: primary=%s, fallback=%s",
            primary_model,
            settings.llm_fallback_model,
        )
        # Gemini API rejects request deadlines below 10s (returns HTTP 400)
        return FallbackAdapter(
            [primary_llm, fallback_llm],
            attempt_timeout=10.0,
            max_retry_per_llm=0,
            retry_interval=0.5,
            retry_on_chunk_sent=False,
        )

    return primary_llm


def get_stt(language: str) -> Any:
    """Lazily instantiate the configured STT provider."""
    provider = settings.stt_provider.lower()
    model = resolve_stt_model(provider, settings.stt_model)
    if provider == "deepgram":
        from livekit.plugins import deepgram

        api_key = settings.deepgram_api_key or os.environ.get("DEEPGRAM_API_KEY")
        kwargs: dict[str, Any] = {
            "language": language,
            "model": model,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return deepgram.STT(**kwargs)
    elif provider == "openai":
        from livekit.plugins import openai

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = {
            "language": language,
            "model": model,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return openai.STT(**kwargs)
    else:
        raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")


def get_tts() -> Any:
    """Lazily instantiate the configured TTS provider."""
    provider = settings.tts_provider.lower()
    model = resolve_tts_model(provider, settings.tts_model)
    voice = resolve_tts_voice(provider, settings.tts_voice_id)
    if provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        api_key = (
            settings.elevenlabs_api_key
            or os.environ.get("ELEVENLABS_API_KEY")
            or os.environ.get("ELEVEN_API_KEY")
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "voice_id": voice,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return elevenlabs.TTS(**kwargs)
    elif provider == "openai":
        from livekit.plugins import openai

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = {
            "model": model,
            "voice": voice,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return openai.TTS(**kwargs)
    else:
        raise ValueError(f"Unsupported TTS provider: {settings.tts_provider}")


def get_vad(ctx: JobContext | None = None) -> Any:
    """Retrieve pre-warmed VAD model or lazily load it."""
    if ctx and hasattr(ctx, "proc") and "vad" in ctx.proc.userdata:
        return ctx.proc.userdata["vad"]
    from livekit.plugins import silero

    return silero.VAD.load()


def get_turn_handling() -> TurnHandlingOptions:
    """Configure turn handling, endpointing, and interruption options."""
    mode = settings.turn_detection_mode.lower()
    if mode == "turn_detector":
        from livekit.agents.inference import TurnDetector

        turn_detector = TurnDetector()
    else:
        # Default VAD-only turn detection (free, language-independent, 0 cloud credits)
        turn_detector = "vad"

    interruption_mode = settings.interruption_mode.lower()
    return TurnHandlingOptions(
        turn_detection=turn_detector,
        interruption={"enabled": True, "mode": interruption_mode},
        endpointing={
            "min_delay": settings.endpointing_min_delay,
            "max_delay": settings.endpointing_max_delay,
        },
        preemptive_generation={
            "enabled": settings.preemptive_generation,
        },
    )


class VoiceCallbackAgent(Agent):
    """Voice agent initialized with script instructions and tools."""

    def __init__(self, script: ScriptConfig) -> None:
        instructions = build_system_prompt(script)
        super().__init__(
            instructions=instructions,
            tools=[end_call],
        )


def extract_transcript(chat_ctx: llm.ChatContext) -> list[dict[str, str]]:
    """Extracts conversational user and assistant turns into structured JSON items."""
    transcript: list[dict[str, str]] = []
    for item in chat_ctx.items:
        if isinstance(item, llm.ChatMessage):
            role = str(item.role.value if hasattr(item.role, "value") else item.role)
            # Skip internal system prompt
            if role == "system":
                continue

            text_parts: list[str] = []
            if isinstance(item.content, list):
                for part in item.content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif hasattr(part, "text") and part.text:
                        text_parts.append(str(part.text))
            elif isinstance(item.content, str):
                text_parts.append(item.content)

            text = " ".join(text_parts).strip()
            if text:
                transcript.append({"role": role, "content": text})
    return transcript


def log_session_summary(
    room_name: str,
    history: llm.ChatContext,
    state: SessionState,
    *,
    extraction_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Logs the final call transcript and collected answers as JSON."""
    transcript = extract_transcript(history)
    summary: dict[str, Any] = {
        "room": room_name,
        "transcript": transcript,
    }

    if extraction_result and extraction_result.get("status") == "extraction_failed":
        summary["status"] = "extraction_failed"
        summary["error"] = extraction_result.get("error", "Unknown extraction error")
        summary["answers"] = state.answers
        summary["missing"] = extraction_result.get("missing", [])
    elif extraction_result:
        summary["answers"] = extraction_result.get("answers", state.answers)
        summary["missing"] = extraction_result.get("missing", [])
    else:
        summary["answers"] = state.answers

    logger.info(
        "Call session ended. Summary:\n%s",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return summary


async def post_call_extraction_and_summary(
    room_name: str,
    history: llm.ChatContext,
    state: SessionState,
    metrics_collector: SessionMetricsCollector,
    script: ScriptConfig,
) -> dict[str, Any]:
    """Post-call extraction from transcript, writes call log, and metrics."""
    transcript_text = build_transcript_text(history)
    question_keys = script.question_fields

    extraction_res = await extract_answers(
        transcript=transcript_text,
        question_keys=question_keys,
        model=settings.extraction_model,
    )

    if extraction_res.get("status") == "ok":
        state.answers = extraction_res.get("answers", {})
    else:
        state.answers = dict.fromkeys(question_keys)

    summary = log_session_summary(
        room_name=room_name,
        history=history,
        state=state,
        extraction_result=extraction_res,
    )

    try:
        save_call_summary(room_name, summary)
    except Exception as e:
        logger.warning("Failed to save call summary to logs/calls: %s", e)

    metrics_collector.on_session_close()
    return summary


async def _cleanup_and_shutdown(
    ctx: JobContext,
    reason: str,
    session: AgentSession | None = None,
) -> None:
    """Closes session (if open), deletes room, and shuts down job context."""
    if session is not None:
        await session.aclose()
    with contextlib.suppress(Exception):
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    ctx.shutdown(reason=reason)


server = AgentServer(
    ws_url=settings.livekit_url,
    api_key=settings.livekit_api_key,
    api_secret=settings.livekit_api_secret,
    setup_fnc=prewarm,
    shutdown_process_timeout=30.0,
)


@server.rtc_session(agent_name=settings.agent_dispatch_name)
async def entrypoint(ctx: JobContext) -> None:
    """RTC session entrypoint for voice callback agent."""
    has_phone_key, raw_phone = extract_job_phone_number(
        getattr(ctx.job, "metadata", None)
    )

    if has_phone_key:
        if not raw_phone or not validate_e164(raw_phone):
            logger.error(
                "Invalid E.164 phone number in job metadata: '%s'. Aborting call.",
                mask_phone_number(raw_phone or ""),
            )
            await _cleanup_and_shutdown(ctx, reason="invalid phone_number in metadata")
            return

        phone_number = raw_phone
        phone_mode = True
        logger.info(
            "Initializing voice agent session for room: %s (phone mode, callee: %s)",
            ctx.room.name,
            mask_phone_number(phone_number),
        )
    else:
        phone_number = None
        phone_mode = False
        logger.info(
            "Initializing voice agent session for room: %s (browser mode)",
            ctx.room.name,
        )

    script = load_script()
    session_state = SessionState(
        closing_phrase=script.closing_phrase,
        error_phrase=script.error_phrase,
        phone_mode=phone_mode,
        room_name=ctx.room.name,
        api=ctx.api,
    )

    # script.yaml is the single source of truth for language;
    # STT_LANGUAGE is an optional override.
    stt_language = settings.stt_language or script.language
    stt = get_stt(stt_language)
    llm_instance = get_llm()
    tts = get_tts()
    vad = get_vad(ctx)
    turn_handling = get_turn_handling()

    # Faster failure for voice: max 1 retry with short 0.5s interval.
    # Note: timeout must be at least 10s as Gemini API rejects deadlines below 10s.
    conn_options = SessionConnectOptions(
        llm_conn_options=APIConnectOptions(
            max_retry=1,
            retry_interval=0.5,
            timeout=10.0,
        ),
    )

    session = AgentSession(
        stt=stt,
        llm=llm_instance,
        tts=tts,
        vad=vad,
        turn_handling=turn_handling,
        conn_options=conn_options,
        userdata=session_state,
    )

    # Note: llm_model records the primary model; fallback-LLM switches are not tracked
    active_models = ActiveModels(
        llm_provider=settings.llm_provider,
        llm_model=resolve_llm_model(settings.llm_provider, settings.llm_model),
        stt_provider=settings.stt_provider,
        stt_model=resolve_stt_model(settings.stt_provider, settings.stt_model),
        tts_provider=settings.tts_provider,
        tts_model=resolve_tts_model(settings.tts_provider, settings.tts_model),
        tts_voice_id=resolve_tts_voice(settings.tts_provider, settings.tts_voice_id),
    )
    metrics_collector = SessionMetricsCollector(
        room_name=ctx.room.name,
        models=active_models,
    )
    metrics_collector.attach_to_session(session)

    agent = VoiceCallbackAgent(script)

    summary_processed = False

    async def _on_shutdown() -> None:
        nonlocal summary_processed
        if not summary_processed:
            summary_processed = True
            try:
                await post_call_extraction_and_summary(
                    room_name=ctx.room.name,
                    history=session.history,
                    state=session_state,
                    metrics_collector=metrics_collector,
                    script=script,
                )
            except Exception as e:
                logger.error(
                    "Error in post-call extraction: %s",
                    mask_phone_number(str(e)),
                )
            finally:
                metrics_collector.on_session_close()

    @session.on("error")
    def _on_error(event: ErrorEvent) -> None:
        err = getattr(event, "error", None)
        if isinstance(err, llm.LLMError):
            if not err.recoverable:
                logger.error(
                    "Unrecoverable LLM error after retries/fallbacks: %s. "
                    "Speaking fixed error phrase and terminating call.",
                    err.error,
                )
                terminate_call(session, session_state, session_state.error_phrase)
            else:
                logger.warning(
                    "Recoverable LLM error (retrying/falling back): %s",
                    err.error,
                )
        else:
            logger.warning("Session error event: %s", event)

    @session.on("close")
    def _on_close(event: CloseEvent) -> None:
        logger.info("Session close event: reason=%s", event.reason)

    ctx.add_shutdown_callback(_on_shutdown)

    if phone_mode and phone_number is not None:
        trunk_id = settings.sip_outbound_trunk_id
        if not trunk_id:
            logger.error(
                "SIP_OUTBOUND_TRUNK_ID is not configured in settings. "
                "Cannot place phone call."
            )
            await _cleanup_and_shutdown(ctx, reason="missing SIP_OUTBOUND_TRUNK_ID")
            return

        # Connect to room first
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to room '%s' in phone mode.", ctx.room.name)

        opaque_identity = f"sip_{uuid.uuid4().hex[:8]}"
        room_options = RoomOptions(
            participant_identity=opaque_identity,
            delete_room_on_close=True,
        )

        # Start the session so no audio is lost once answered
        await session.start(agent=agent, room=ctx.room, room_options=room_options)

        # Dial out via SIP
        logger.info(
            "Dialing %s via trunk %s...",
            mask_phone_number(phone_number),
            trunk_id,
        )
        try:
            sip_req = api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                participant_identity=opaque_identity,
                wait_until_answered=True,
                ringing_timeout=Duration(seconds=30),
                max_call_duration=Duration(seconds=600),
            )
            sip_info = await ctx.api.sip.create_sip_participant(sip_req)
            logger.info(
                "Call answered by %s (participant identity: %s)",
                mask_phone_number(phone_number),
                sip_info.participant_identity or opaque_identity,
            )
        except api.SipCallError as e:
            logger.error(
                "SIP call to %s failed: status_code=%s, status=%s, reason=%s",
                mask_phone_number(phone_number),
                e.sip_status_code,
                e.sip_status,
                e.message,
            )
            await _cleanup_and_shutdown(
                ctx,
                reason=f"sip call failed: {e.sip_status or e.message}",
                session=session,
            )
            return
        except Exception as e:
            logger.error(
                "Dialing %s failed: %s",
                mask_phone_number(phone_number),
                e,
            )
            await _cleanup_and_shutdown(ctx, reason="dial failed", session=session)
            return

        # Greet only after the call is answered
        logger.info("Agent greeting participant with: '%s'", script.greeting)
        await session.say(script.greeting, allow_interruptions=False)

    else:
        # Browser mode (M1 behaviour)
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to room '%s', waiting for participant...", ctx.room.name)
        try:
            participant = await ctx.wait_for_participant()
        except RuntimeError:
            logger.info("room closed before participant joined")
            return

        logger.info(
            "Participant joined: identity=%s, name=%s",
            participant.identity,
            participant.name,
        )

        await session.start(agent=agent, room=ctx.room)

        logger.info("Agent greeting participant with: '%s'", script.greeting)
        await session.say(script.greeting, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
