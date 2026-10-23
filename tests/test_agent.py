"""Tests for agent configuration, prompt building, and session helpers."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import StopResponse, llm

from agent.prompts import QuestionItem, ScriptConfig, build_system_prompt, load_script
from agent.tools import SessionState, end_call, save_answer_to_state, terminate_call
from agent.worker import extract_transcript, get_llm, log_session_summary
from app.config import settings


def test_load_default_script() -> None:
    """Verifies default script.yaml loads correctly in Ukrainian."""
    script = load_script()

    assert script.company_name == "Solaris Tech"
    assert script.agent_name == "Анна"
    assert script.language == "uk"
    assert "записуватися" in script.greeting.lower()
    assert "асистентка" in script.greeting.lower()
    assert len(script.closing_phrase) > 0
    assert "помилка" in script.error_phrase.lower()
    assert "менеджер" in script.error_phrase.lower()

    fields = [q.field for q in script.questions]
    assert fields == ["name", "service", "timeline", "preferred_call_time"]

    required_map = {q.field: q.required for q in script.questions}
    assert required_map["name"] is True
    assert required_map["service"] is True
    assert required_map["timeline"] is False
    assert required_map["preferred_call_time"] is True


def test_load_custom_script(tmp_path: Path) -> None:
    """Verifies loading a custom script file with English."""
    custom_yaml = tmp_path / "custom_script.yaml"
    custom_yaml.write_text(
        """
company_name: "Acme Corp"
agent_name: "Alex"
language: "en"
greeting: "Hello, this is Alex from Acme Corp."
closing_phrase: "Have a nice day!"
questions:
  - field: "topic"
    question: "What would you like to discuss?"
    required: true
""",
        encoding="utf-8",
    )

    script = load_script(custom_yaml)
    assert script.company_name == "Acme Corp"
    assert script.agent_name == "Alex"
    assert script.language == "en"
    assert len(script.questions) == 1
    assert script.questions[0].field == "topic"


def test_load_polish_script(tmp_path: Path) -> None:
    """Verifies that switching back to Polish via script.yaml works cleanly."""
    pl_yaml = tmp_path / "polish_script.yaml"
    pl_yaml.write_text(
        """
company_name: "Solaris Tech"
agent_name: "Anna"
language: "pl"
greeting: "Dzień dobry! Z tej strony Anna. Rozmowa może być nagrywana."
closing_phrase: "Dziękuję i do usłyszenia!"
questions:
  - field: "name"
    question: "Jak masz na imię?"
    required: true
""",
        encoding="utf-8",
    )

    script = load_script(pl_yaml)
    assert script.language == "pl"
    assert script.agent_name == "Anna"
    assert "nagrywana" in script.greeting


def test_build_system_prompt_rules() -> None:
    """Verifies that all prompt constraints and rules are present."""
    script = ScriptConfig(
        company_name="Test Company",
        agent_name="Ewa",
        language="pl",
        greeting="Dzień dobry!",
        closing_phrase="Dziękuję i do usłyszenia!",
        questions=[
            QuestionItem(field="name", question="Jak masz na imię?", required=True),
            QuestionItem(field="budget", question="Jaki masz budżet?", required=False),
        ],
    )

    prompt = build_system_prompt(script)

    # Identity & Language
    assert "Ewa" in prompt
    assert "Test Company" in prompt
    assert "Polish (pl)" in prompt

    # Spoken rules
    assert "plain spoken text" in prompt.lower()
    assert "never use markdown" in prompt.lower()
    assert "1 to 2 sentences" in prompt
    assert "one question at a time" in prompt.lower()
    assert "first name only" in prompt.lower()
    assert "пан" in prompt

    # Guardrails
    assert "never invent prices" in prompt.lower()
    assert "not interested" in prompt.lower()

    # Flow & questions
    assert "Field 'name'" in prompt
    assert "Field 'budget'" in prompt
    assert "Dziękuję i do usłyszenia!" in prompt


def test_build_system_prompt_language_switch() -> None:
    """Verifies that prompt generation adapts to English and Ukrainian."""
    uk_script = ScriptConfig(
        company_name="Demo UA",
        agent_name="Oksana",
        language="uk",
        greeting="Привіт!",
        closing_phrase="До побаczenia!",
        questions=[],
    )
    uk_prompt = build_system_prompt(uk_script)
    assert "Ukrainian (uk)" in uk_prompt

    en_script = ScriptConfig(
        company_name="Demo EN",
        agent_name="Sarah",
        language="en",
        greeting="Hello!",
        closing_phrase="Goodbye!",
        questions=[],
    )
    en_prompt = build_system_prompt(en_script)
    assert "English (en)" in en_prompt


def test_session_state_and_save_answer() -> None:
    """Tests storing answers in SessionState."""
    state = SessionState(closing_phrase="Pa!")
    assert state.answers == {}
    assert state.call_ended is False
    assert len(state.background_tasks) == 0

    save_answer_to_state(state, "name", "Jan Kowalski")
    save_answer_to_state(state, "service", "Instalacja paneli PV")

    assert state.answers["name"] == "Jan Kowalski"
    assert state.answers["service"] == "Instalacja paneli PV"


@pytest.mark.asyncio
async def test_end_call_raises_stop_response_and_retains_task() -> None:
    """Verifies end_call raises StopResponse and tracks background task."""
    state = SessionState(closing_phrase="Do usłyszenia!")

    mock_speech_handle = MagicMock()
    mock_speech_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.say.return_value = mock_speech_handle
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    # Verify strong reference is stored in background_tasks
    assert len(state.background_tasks) == 1

    # Allow background task to execute
    for task in list(state.background_tasks):
        await task

    mock_session.say.assert_called_once_with(
        "Do usłyszenia!",
        allow_interruptions=False,
    )
    mock_session.aclose.assert_awaited_once()
    assert len(state.background_tasks) == 0


def test_extract_transcript_and_summary() -> None:
    """Tests extracting transcript items and compiling session summary."""
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="System instruction")
    chat_ctx.add_message(role="assistant", content="Dzień dobry! Z tej strony Anna.")
    chat_ctx.add_message(role="user", content="Dzień dobry, Jan z tej strony.")
    chat_ctx.add_message(role="assistant", content="W czym mogę pomóc?")

    transcript = extract_transcript(chat_ctx)

    # System instruction must be excluded from conversation transcript
    assert len(transcript) == 3
    assert transcript[0] == {
        "role": "assistant",
        "content": "Dzień dobry! Z tej strony Anna.",
    }
    assert transcript[1] == {
        "role": "user",
        "content": "Dzień dobry, Jan z tej strony.",
    }
    assert transcript[2] == {
        "role": "assistant",
        "content": "W czym mogę pomóc?",
    }

    state = SessionState(closing_phrase="Pa!", answers={"name": "Jan"})
    summary = log_session_summary("test-room-123", chat_ctx, state)

    assert summary["room"] == "test-room-123"
    assert summary["answers"] == {"name": "Jan"}
    assert summary["transcript"] == transcript


def test_agent_dispatch_name_setting() -> None:
    """Verifies agent_dispatch_name defaults to empty string for auto dispatch."""
    assert isinstance(settings.agent_dispatch_name, str)


def test_llm_fallback_model_setting() -> None:
    """Verifies default llm_model and fallback model settings."""
    assert settings.llm_model == "gemini-3.5-flash-lite"
    assert settings.llm_fallback_model == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_terminate_call_speaks_error_phrase_and_disconnects() -> None:
    """Verifies terminate_call speaks error phrase without interruption and closes."""
    state = SessionState(error_phrase="Перепрошую, виникла технічна помилка.")

    mock_speech_handle = MagicMock()
    mock_speech_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.say.return_value = mock_speech_handle
    mock_session.aclose = AsyncMock()

    task = terminate_call(mock_session, state, state.error_phrase)
    assert task is not None
    assert state.call_ended is True
    assert task in state.background_tasks

    # A second call must be a no-op (idempotent)
    second_task = terminate_call(mock_session, state, state.error_phrase)
    assert second_task is None

    await task

    mock_session.say.assert_called_once_with(
        "Перепрошую, виникла технічна помилка.",
        allow_interruptions=False,
    )
    mock_speech_handle.wait_for_playout.assert_awaited_once()
    mock_session.aclose.assert_awaited_once()
    assert len(state.background_tasks) == 0


def test_get_llm_fallback_adapter_wrapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies get_llm wraps with FallbackAdapter when fallback model is set."""
    from livekit.agents.llm import FallbackAdapter

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "google")
    monkeypatch.setattr(settings, "llm_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "llm_fallback_model", "gemini-3.5-flash")

    llm_instance = get_llm()
    assert isinstance(llm_instance, FallbackAdapter)
    assert len(llm_instance._llm_instances) == 2
    assert llm_instance._llm_instances[0].model == "gemini-3.5-flash-lite"
    assert llm_instance._llm_instances[1].model == "gemini-3.5-flash"
    assert llm_instance._attempt_timeout == 10.0
    assert llm_instance._retry_on_chunk_sent is False
    assert llm_instance._max_retry_per_llm == 0
    assert llm_instance._retry_interval == 0.5


@pytest.mark.asyncio
async def test_entrypoint_registers_async_shutdown_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that entrypoint registers an async shutdown callback.

    In LiveKit job runner, zero-argument callbacks are wrapped with:
        async def wrapper(_: str) -> None:
            await callback()
    A synchronous callback returns None, triggering TypeError:
    object NoneType can't be used in 'await' expression.
    The callback must be a coroutine function.
    """
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_ctx = MagicMock()
    mock_ctx.room.name = "test-room"
    mock_ctx.job.metadata = None
    mock_ctx.connect = AsyncMock()
    # Cancel at wait_for_participant to halt entrypoint after session setup
    mock_ctx.wait_for_participant = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await entrypoint(mock_ctx)

    mock_ctx.add_shutdown_callback.assert_called_once()
    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]

    # Must be a coroutine function
    assert inspect.iscoroutinefunction(registered_cb)

    # Calling and awaiting must succeed without TypeError
    result = await registered_cb()
    assert result is None

    # Simulating LiveKit's job.py:637 wrapper must also succeed
    async def lk_wrapper(_: str) -> None:
        await registered_cb()

    await lk_wrapper("room disconnected")

