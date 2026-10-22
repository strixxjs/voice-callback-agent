"""LiveKit voice agent worker entrypoint."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

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

from agent.prompts import ScriptConfig, build_system_prompt, load_script
from agent.tools import SessionState, end_call, save_answer, terminate_call
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


def get_llm() -> Any:
    """Lazily instantiate configured LLM, optionally wrapped in FallbackAdapter."""
    provider = settings.llm_provider.lower()
    primary_model = settings.llm_model or (
        "gemini-3.5-flash-lite" if provider == "google" else "gpt-4o-mini"
    )
    primary_llm = _create_llm_instance(provider, primary_model)

    if settings.llm_fallback_model:
        from livekit.agents.llm import FallbackAdapter

        fallback_llm = _create_llm_instance(provider, settings.llm_fallback_model)
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
    if provider == "deepgram":
        from livekit.plugins import deepgram

        api_key = settings.deepgram_api_key or os.environ.get("DEEPGRAM_API_KEY")
        kwargs: dict[str, Any] = {"language": language}
        if settings.stt_model:
            kwargs["model"] = settings.stt_model
        if api_key:
            kwargs["api_key"] = api_key
        return deepgram.STT(**kwargs)
    elif provider == "openai":
        from livekit.plugins import openai

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = {"language": language}
        if settings.stt_model:
            kwargs["model"] = settings.stt_model
        if api_key:
            kwargs["api_key"] = api_key
        return openai.STT(**kwargs)
    else:
        raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")


def get_tts() -> Any:
    """Lazily instantiate the configured TTS provider."""
    provider = settings.tts_provider.lower()
    if provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        api_key = (
            settings.elevenlabs_api_key
            or os.environ.get("ELEVENLABS_API_KEY")
            or os.environ.get("ELEVEN_API_KEY")
        )
        kwargs: dict[str, Any] = {}
        if settings.tts_model:
            kwargs["model"] = settings.tts_model
        if settings.tts_voice_id:
            kwargs["voice_id"] = settings.tts_voice_id
        if api_key:
            kwargs["api_key"] = api_key
        return elevenlabs.TTS(**kwargs)
    elif provider == "openai":
        from livekit.plugins import openai

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = {}
        if settings.tts_model:
            kwargs["model"] = settings.tts_model
        if settings.tts_voice_id:
            kwargs["voice"] = settings.tts_voice_id
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
    """Configure turn handling and interruption options."""
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
    )


class VoiceCallbackAgent(Agent):
    """Voice agent initialized with script instructions and tools."""

    def __init__(self, script: ScriptConfig) -> None:
        instructions = build_system_prompt(script)
        super().__init__(
            instructions=instructions,
            tools=[save_answer, end_call],
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
) -> dict[str, Any]:
    """Logs the final call transcript and collected answers as JSON."""
    transcript = extract_transcript(history)
    summary: dict[str, Any] = {
        "room": room_name,
        "answers": state.answers,
        "transcript": transcript,
    }
    logger.info(
        "Call session ended. Summary:\n%s",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return summary


server = AgentServer(
    ws_url=settings.livekit_url,
    api_key=settings.livekit_api_key,
    api_secret=settings.livekit_api_secret,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name=settings.agent_dispatch_name)
async def entrypoint(ctx: JobContext) -> None:
    """RTC session entrypoint for voice callback agent."""
    logger.info("Initializing voice agent session for room: %s", ctx.room.name)

    script = load_script()
    session_state = SessionState(
        closing_phrase=script.closing_phrase,
        error_phrase=script.error_phrase,
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

    agent = VoiceCallbackAgent(script)

    summary_logged = False

    def on_session_done() -> None:
        nonlocal summary_logged
        if not summary_logged:
            summary_logged = True
            log_session_summary(ctx.room.name, session.history, session_state)

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
        on_session_done()

    ctx.add_shutdown_callback(on_session_done)

    # Connect to room and wait for participant
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to room '%s', waiting for participant...", ctx.room.name)
    participant = await ctx.wait_for_participant()
    logger.info(
        "Participant joined: identity=%s, name=%s",
        participant.identity,
        participant.name,
    )

    await session.start(agent=agent, room=ctx.room)

    # Agent speaks first with the greeting.
    # Note: Interruptions are disabled as it contains the call recording notice.
    logger.info("Agent greeting participant with: '%s'", script.greeting)
    await session.say(script.greeting, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
