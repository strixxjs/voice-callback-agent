"""Tests for agent configuration, prompt building, and session helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from livekit.agents import StopResponse, llm

from agent.extraction import (
    build_transcript_text,
    extract_answers,
    format_short_error,
    save_call_summary,
)
from agent.metrics import SessionMetricsCollector
from agent.prompts import QuestionItem, ScriptConfig, build_system_prompt, load_script
from agent.telephony import mask_phone_number
from agent.tools import (
    SessionState,
    end_call,
    has_pending_agent_question,
    is_question_text,
    terminate_call,
)
from agent.worker import (
    VoiceCallbackAgent,
    extract_transcript,
    get_llm,
    get_stt,
    get_turn_handling,
    log_session_summary,
    post_call_extraction_and_summary,
)
from app.config import settings


def test_load_default_script() -> None:
    """Verifies default script.yaml loads correctly in Ukrainian."""
    script = load_script()

    assert script.company_name == "Solaris Tech"
    assert script.agent_name == "Саня"
    assert script.language == "uk"
    assert "асистентка" in script.greeting.lower()
    assert "solaris tech" in script.greeting.lower()
    assert len(script.closing_phrase) > 0
    assert "помилка" in script.error_phrase.lower()
    assert "менеджер" in script.error_phrase.lower()

    fields = [q.field for q in script.questions]
    assert fields == ["name", "service", "timeline", "preferred_call_time"]
    assert script.question_fields == fields

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
    assert "Question 'name'" in prompt
    assert "Question 'budget'" in prompt
    assert "end_call" in prompt
    assert "farewell" in prompt.lower()
    assert "only after the user has answered the last question" in prompt.lower()
    assert "never ask a question and call `end_call()`" in prompt.lower()
    assert (
        "ask every question from the list in order, including optional ones"
        in prompt.lower()
    )
    assert '"optional" only means the caller may decline to answer it' in prompt.lower()
    assert "never skip asking it" in prompt.lower()
    assert "new information, or a correction" in prompt.lower()
    assert "first reply in one or two short" in prompt.lower()
    assert "sentences in the call language" in prompt.lower()
    assert "explicitly acknowledge a correction" in prompt.lower()
    assert "merely thanks you or says goodbye" in prompt.lower()
    assert "right away without extra words" in prompt.lower()


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


def test_save_answer_not_in_tools_and_end_call_is() -> None:
    """Verifies save_answer is removed from agent tools and end_call is present."""
    import agent.tools

    assert not hasattr(agent.tools, "save_answer")
    assert not hasattr(agent.tools, "save_answer_to_state")

    script = load_script()
    agent_instance = VoiceCallbackAgent(script)
    tool_names = [getattr(t, "__name__", str(t)) for t in agent_instance.tools]

    assert "end_call" in tool_names
    assert "save_answer" not in tool_names


def test_session_state_initialization() -> None:
    """Tests initializing SessionState without save_answer."""
    state = SessionState(closing_phrase="Pa!")
    assert state.answers == {}
    assert state.call_ended is False
    assert len(state.background_tasks) == 0


@pytest.mark.asyncio
async def test_end_call_waits_for_playout_and_disconnects() -> None:
    """Verifies end_call speaks closing phrase, waits playout, and terminates."""
    closing = "Дякую за розмову! До побачення!"
    state = SessionState(closing_phrase=closing)

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    mock_context.wait_for_playout.assert_awaited_once()
    assert len(state.background_tasks) == 1

    for task in list(state.background_tasks):
        await task

    # Closing phrase was passed to terminate_call -> spoken via session.say
    mock_session.say.assert_called_once_with(closing, allow_interruptions=True)
    mock_handle.wait_for_playout.assert_awaited_once()
    mock_session.aclose.assert_awaited_once()
    assert len(state.background_tasks) == 0


@pytest.mark.asyncio
async def test_end_call_does_not_close_when_user_speaking() -> None:
    """Verifies end_call aborts and returns None if caller is currently speaking."""
    state = SessionState(closing_phrase="До побачення!")

    mock_session = MagicMock()
    mock_session.user_state = "speaking"
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    # Must return None and NOT raise StopResponse
    result = await end_call(mock_context)
    assert result is None

    assert state.call_ended is False
    assert len(state.background_tasks) == 0
    mock_context.wait_for_playout.assert_not_called()
    mock_session.say.assert_not_called()
    mock_session.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_end_call_does_not_close_if_user_starts_speaking_during_playout() -> None:
    """Verifies end_call aborts if user starts speaking while waiting for playout."""
    state = SessionState(closing_phrase="До побачення!")

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session

    async def simulate_user_starts_speaking() -> None:
        mock_session.user_state = "speaking"

    mock_context.wait_for_playout = AsyncMock(side_effect=simulate_user_starts_speaking)

    result = await end_call(mock_context)
    assert result is None

    assert state.call_ended is False
    assert len(state.background_tasks) == 0
    mock_session.say.assert_not_called()
    mock_session.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_end_call_logs_warning_on_playout_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies playout error is logged at warning level and call still terminates."""
    closing = "До побачення!"
    state = SessionState(closing_phrase=closing)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.say = MagicMock(return_value=None)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock(
        side_effect=RuntimeError("simulated playout failure")
    )

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    # Verify warning log was emitted
    warning_records = [
        r
        for r in caplog.records
        if "wait_for_playout in end_call" in r.message and r.levelname == "WARNING"
    ]
    assert len(warning_records) == 1


def test_is_question_text_detection() -> None:
    """Verifies question mark detection with various punctuations and quotes."""
    # Positive cases
    assert (
        is_question_text(
            "О котрій годині нашому менеджеру буде найзручніше вам зателефонувати?"
        )
        is True
    )
    assert is_question_text("Зручно зараз говорити? ") is True
    assert is_question_text("Чи можу я уточнити ваше ім'я?") is True
    assert is_question_text("What time is best for a callback?") is True
    assert is_question_text("Jak masz na imię?") is True
    assert is_question_text("«Зручно говорити?»") is True
    assert is_question_text('"Ви впевнені?"') is True
    assert is_question_text("Справді?!") is True
    assert is_question_text("Коли саме?...") is True
    assert is_question_text("Зручно?.)") is True

    # Negative cases
    assert is_question_text("Дякую за розмову!") is False
    assert is_question_text("Менеджер зв'яжеться з вами.") is False
    assert is_question_text("це все, до побачення") is False
    assert is_question_text("не хочу говорити") is False
    assert is_question_text("") is False
    assert is_question_text(None) is False


def test_has_pending_agent_question_decision() -> None:
    """Verifies decision: question pending -> reject;
    user answered / requested finish -> allow.
    """
    last_q = "О котрій годині нашому менеджеру буде найзручніше вам зателефонувати?"

    # 1. Question pending in current generation -> reject (True)
    assert has_pending_agent_question(current_generation_text=last_q) is True

    # 2. Question pending in history (no subsequent user message) -> reject (True)
    messages_pending_history = [
        {"role": "user", "content": "Привіт"},
        {"role": "assistant", "content": last_q},
    ]
    assert has_pending_agent_question(messages=messages_pending_history) is True

    # ChatContext variant of pending question in history
    ctx_pending = llm.ChatContext()
    ctx_pending.add_message(role="assistant", content="Як вас звати?")
    ctx_pending.add_message(role="user", content="Олексій")
    ctx_pending.add_message(role="assistant", content=last_q)
    assert has_pending_agent_question(messages=ctx_pending.messages()) is True

    # 3. User answered after agent question -> allow (False)
    messages_answered = [
        {"role": "assistant", "content": last_q},
        {"role": "user", "content": "О 15:00"},
    ]
    assert has_pending_agent_question(messages=messages_answered) is False

    ctx_answered = llm.ChatContext()
    ctx_answered.add_message(role="assistant", content=last_q)
    ctx_answered.add_message(role="user", content="Завтра після 14:00")
    assert has_pending_agent_question(messages=ctx_answered.messages()) is False

    # 4. User asked to finish after agent question -> allow (False)
    messages_finish_phrase_1 = [
        {"role": "assistant", "content": last_q},
        {"role": "user", "content": "це все"},
    ]
    assert has_pending_agent_question(messages=messages_finish_phrase_1) is False

    messages_finish_phrase_2 = [
        {"role": "assistant", "content": last_q},
        {"role": "user", "content": "до побачення"},
    ]
    assert has_pending_agent_question(messages=messages_finish_phrase_2) is False

    messages_refuse = [
        {"role": "assistant", "content": last_q},
        {"role": "user", "content": "не хочу говорити"},
    ]
    assert has_pending_agent_question(messages=messages_refuse) is False

    # 5. No question asked (statement / empty) -> allow (False)
    messages_no_question = [
        {"role": "assistant", "content": "Дякую, всі дані зафіксовано."},
    ]
    assert has_pending_agent_question(messages=messages_no_question) is False
    assert has_pending_agent_question(messages=[]) is False
    assert has_pending_agent_question(current_generation_text="Дякую!") is False


@pytest.mark.asyncio
async def test_end_call_rejected_when_question_pending_in_current_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies end_call returns None and logs warning if current speech
    generation asked a question.
    """
    question = "О котрій годині нашому менеджеру буде найзручніше вам зателефонувати?"
    state = SessionState(closing_phrase="Дякую за розмову! До побачення!")

    chat_ctx = llm.ChatContext()
    agent_msg = chat_ctx.add_message(role="assistant", content=question)

    mock_speech_handle = MagicMock()
    mock_speech_handle.chat_items = [agent_msg]

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.speech_handle = mock_speech_handle
    mock_context.wait_for_playout = AsyncMock()

    result = await end_call(mock_context)

    # Must return None so agent waits without generating another LLM turn
    assert result is None
    assert state.call_ended is False
    assert len(state.background_tasks) == 0
    mock_session.say.assert_not_called()
    mock_session.aclose.assert_not_called()

    # Guard warning must be logged
    warning_logs = [
        r.message
        for r in caplog.records
        if "end_call rejected: pending question" in r.message
        and r.levelname == "WARNING"
    ]
    assert len(warning_logs) == 1


@pytest.mark.asyncio
async def test_end_call_rejected_when_question_pending_in_history(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies end_call returns None and logs warning when last agent turn
    in history was a question.
    """
    question = "Чи можу я уточнити ваше ім'я та прізвище?"
    state = SessionState(closing_phrase="До побачення!")

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Доброго дня")
    chat_ctx.add_message(role="assistant", content=question)

    mock_speech_handle = MagicMock()
    mock_speech_handle.chat_items = []

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.speech_handle = mock_speech_handle
    mock_context.wait_for_playout = AsyncMock()

    result = await end_call(mock_context)

    assert result is None
    assert state.call_ended is False
    assert len(state.background_tasks) == 0
    mock_session.say.assert_not_called()
    mock_session.aclose.assert_not_called()

    warning_logs = [
        r.message
        for r in caplog.records
        if "end_call rejected: pending question" in r.message
        and r.levelname == "WARNING"
    ]
    assert len(warning_logs) == 1


@pytest.mark.asyncio
async def test_end_call_allowed_when_user_answered() -> None:
    """Verifies end_call terminates when user responded to the last agent question."""
    closing = "Дякую за розмову! До побачення!"
    state = SessionState(closing_phrase=closing)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(
        role="assistant",
        content="О котрій годині нашому менеджеру буде найзручніше вам зателефонувати?",
    )
    chat_ctx.add_message(role="user", content="О 15:00")

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_speech_handle = MagicMock()
    mock_speech_handle.chat_items = []

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.speech_handle = mock_speech_handle
    mock_context.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    assert len(state.background_tasks) == 1
    for task in list(state.background_tasks):
        await task
    mock_session.say.assert_called_once_with(closing, allow_interruptions=True)
    mock_session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_end_call_allowed_when_user_requested_to_finish() -> None:
    """Verifies end_call terminates when caller explicitly asks to finish."""
    closing = "Дякую за розмову! До побачення!"
    state = SessionState(closing_phrase=closing)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="assistant", content="Яка послуга вас цікавить?")
    chat_ctx.add_message(role="user", content="це все, до побачення")

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    for task in list(state.background_tasks):
        await task
    mock_session.say.assert_called_once_with(closing, allow_interruptions=True)


@pytest.mark.asyncio
async def test_end_call_allowed_when_no_question_asked() -> None:
    """Verifies end_call terminates when last agent message was a statement
    rather than a question.
    """
    closing = "Дякую за розмову! До побачення!"
    state = SessionState(closing_phrase=closing)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="assistant", content="Дякую, вся інформація записана.")

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    for task in list(state.background_tasks):
        await task
    mock_session.say.assert_called_once_with(closing, allow_interruptions=True)


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


def test_get_llm_google_shares_thought_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that primary and fallback Google LLMs share _thought_signatures."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "google")
    monkeypatch.setattr(settings, "llm_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "llm_fallback_model", "gemini-3.5-flash")

    llm_instance = get_llm()
    primary, fallback = llm_instance._llm_instances

    # 1. Both instances must share the exact same dictionary object
    assert hasattr(primary, "_thought_signatures")
    assert hasattr(fallback, "_thought_signatures")
    assert primary._thought_signatures is fallback._thought_signatures

    # 2. A signature written through primary must be visible through fallback
    primary._thought_signatures["test_call_id"] = b"sig_bytes"
    assert fallback._thought_signatures["test_call_id"] == b"sig_bytes"


def test_sync_google_thought_signatures_missing_attribute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that missing _thought_signatures logs a warning and does not crash."""
    from agent.worker import _sync_google_thought_signatures

    class DummyLLM:
        pass

    primary = DummyLLM()
    fallback = DummyLLM()

    # Must not raise an exception
    with caplog.at_level("WARNING"):
        _sync_google_thought_signatures(primary, fallback)

    assert "Cannot sync Google thought signatures" in caplog.text
    assert "attribute missing" in caplog.text


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


def test_build_transcript_text_chat_context() -> None:
    """Verifies build_transcript_text converts ChatContext to plain text turns."""
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="System instruction to ignore")
    chat_ctx.add_message(role="assistant", content="Доброго дня! Чим можу допомогти?")
    chat_ctx.add_message(
        role="user", content="Доброго дня, мене цікавить консультація."
    )
    chat_ctx.add_message(role="assistant", content="Як я можу до вас звертатися?")
    chat_ctx.add_message(role="user", content="Я Дмитро.")

    transcript = build_transcript_text(chat_ctx)
    lines = transcript.splitlines()

    assert len(lines) == 4
    assert lines[0] == "Agent: Доброго дня! Чим можу допомогти?"
    assert lines[1] == "User: Доброго дня, мене цікавить консультація."
    assert lines[2] == "Agent: Як я можу до вас звертатися?"
    assert lines[3] == "User: Я Дмитро."
    assert "System instruction" not in transcript


def test_build_transcript_text_list_items() -> None:
    """Verifies build_transcript_text handles list-of-dicts representation."""
    items = [
        {"role": "system", "content": "Ignore this"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Hi there."},
        {"role": "assistant", "content": "   "},  # empty turn skipped
    ]
    transcript = build_transcript_text(items)
    assert transcript == "Agent: Hello!\nUser: Hi there."


@pytest.mark.asyncio
async def test_extract_answers_full_mocked() -> None:
    """Verifies extract_answers parses complete answers from a mocked LLM response."""
    question_keys = ["name", "service", "timeline", "preferred_call_time"]
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "answers": {
                "name": "Тарас Шевченко",
                "service": "Встановлення сонячних панелей",
                "timeline": "через місяць",
                "preferred_call_time": "завтра о другій годині",
            },
            "missing": [],
        }
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    transcript = "Agent: Як вас звати?\nUser: Тарас Шевченко."
    result = await extract_answers(
        transcript=transcript,
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "ok"
    assert result["answers"]["name"] == "Тарас Шевченко"
    assert result["answers"]["service"] == "Встановлення сонячних панелей"
    assert result["answers"]["timeline"] == "через місяць"
    assert result["answers"]["preferred_call_time"] == "завтра о другій годині"
    assert result["missing"] == []


@pytest.mark.asyncio
async def test_extract_answers_partial_mocked() -> None:
    """Verifies extract_answers identifies missing fields and raw wording."""
    question_keys = ["name", "service", "timeline", "preferred_call_time"]
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "answers": {
                "name": "Оксана",
                "service": None,
                "preferred_call_time": "anytime tomorrow afternoon after 3pm",
            },
            "missing": ["service"],
        }
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await extract_answers(
        transcript="Agent: Привіт\nUser: Я Оксана",
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "ok"
    assert result["answers"]["name"] == "Оксана"
    assert result["answers"]["service"] is None
    assert result["answers"]["timeline"] is None
    assert (
        result["answers"]["preferred_call_time"]
        == "anytime tomorrow afternoon after 3pm"
    )
    assert sorted(result["missing"]) == ["service", "timeline"]


@pytest.mark.asyncio
async def test_extract_answers_invalid_json() -> None:
    """Verifies invalid JSON does not crash worker and marks extraction_failed."""
    question_keys = ["name", "service"]
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON at all: {foo: bar"

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await extract_answers(
        transcript="Agent: Привіт\nUser: Привіт",
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "extraction_failed"
    assert result["answers"] == {"name": None, "service": None}
    assert result["missing"] == ["name", "service"]
    assert "error" in result
    assert len(result["error"]) > 0


@pytest.mark.asyncio
async def test_extract_answers_timeout() -> None:
    """Verifies timeout from LLM does not crash worker and marks extraction_failed."""
    question_keys = ["name", "service"]
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=TimeoutError())

    result = await extract_answers(
        transcript="Agent: Привіт\nUser: Привіт",
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "extraction_failed"
    assert result["answers"] == {"name": None, "service": None}
    assert result["missing"] == ["name", "service"]
    assert result["error"].startswith("TimeoutError:")
    assert "timed out" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_answers_api_error() -> None:
    """Verifies API error does not crash, masks phones, and truncates error."""
    question_keys = ["name", "service"]
    mock_client = MagicMock()
    long_msg = "Connection failed for +380501234567: " + ("detail " * 40)
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=RuntimeError(long_msg)
    )

    result = await extract_answers(
        transcript="Agent: Привіт\nUser: Привіт",
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "extraction_failed"
    assert result["answers"] == {"name": None, "service": None}
    assert result["missing"] == ["name", "service"]
    assert result["error"].startswith("RuntimeError:")
    # Verify phone number was masked in error
    assert "+380501234567" not in result["error"]
    assert mask_phone_number("+380501234567") in result["error"]
    # Verify error message is truncated to <= 200 chars after prefix
    prefix_len = len("RuntimeError: ")
    assert len(result["error"]) <= prefix_len + 200


def test_format_short_error_with_status_code() -> None:
    """Verifies format_short_error includes status code and truncates long body."""

    class CustomAPIError(Exception):
        status_code = 503

    err = CustomAPIError("X" * 300)
    formatted = format_short_error(err)
    assert formatted.startswith("CustomAPIError: 503 ")
    assert len(formatted) == len("CustomAPIError: 503 ") + 200


@pytest.mark.asyncio
async def test_extract_answers_empty_transcript() -> None:
    """Verifies empty transcript returns null answers without calling LLM."""
    question_keys = ["name", "service", "timeline"]
    mock_client = MagicMock()

    result = await extract_answers(
        transcript="",
        question_keys=question_keys,
        client=mock_client,
    )

    assert result["status"] == "ok"
    assert result["answers"] == {"name": None, "service": None, "timeline": None}
    assert result["missing"] == ["name", "service", "timeline"]
    mock_client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_entrypoint_room_closed_before_participant_joined(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies entrypoint logs info, reports no_answer event and status,
    and returns cleanly on empty room RuntimeError.
    """
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps(
        {"call_id": "test-call-room-closed", "mode": "browser"}
    )
    mock_ctx.room.name = "test-empty-room"
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(
        side_effect=RuntimeError("room closed before participant joined")
    )

    # Must return cleanly without raising
    with caplog.at_level("INFO"):
        await entrypoint(mock_ctx)

    mock_ctx.connect.assert_awaited_once()
    mock_ctx.wait_for_participant.assert_awaited_once()
    assert "room closed before participant joined" in caplog.text
    mock_reporter.report_event.assert_called_once_with(
        "no_answer",
        f"Participant did not join within {settings.participant_join_timeout} s",
        {"timeout_seconds": settings.participant_join_timeout},
    )
    mock_reporter.report_ended.assert_called_once_with(
        status="no_answer", duration_seconds=0, latency_summary=ANY
    )
    mock_reporter.report_extraction.assert_not_called()
    mock_reporter.aclose.assert_awaited_once()


def test_save_call_summary(tmp_path: Path) -> None:
    """Verifies save_call_summary creates dir and writes pretty JSON."""
    summary = {
        "room": "test-room-456",
        "answers": {"name": "Іван"},
        "missing": ["service"],
        "transcript": [{"role": "user", "content": "Іван"}],
    }
    calls_dir = tmp_path / "calls"
    file_path = save_call_summary("test-room-456", summary, base_dir=calls_dir)

    assert file_path.exists()
    content = json.loads(file_path.read_text(encoding="utf-8"))
    assert content["room"] == "test-room-456"
    assert content["answers"]["name"] == "Іван"
    assert content["missing"] == ["service"]


@pytest.mark.asyncio
async def test_post_call_extraction_and_summary_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies post_call_extraction_and_summary writes summary and logs correctly."""
    from agent.metrics import ActiveModels, SessionMetricsCollector

    script = load_script()
    state = SessionState()
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="assistant", content="Доброго дня!")
    chat_ctx.add_message(role="user", content="Доброго дня, мене звати Тарас.")

    metrics_collector = SessionMetricsCollector(
        room_name="test-room-extraction",
        models=ActiveModels(
            llm_provider="google",
            llm_model="gemini-3.5-flash-lite",
            stt_provider="deepgram",
            stt_model="nova-3",
            tts_provider="elevenlabs",
            tts_model="eleven_turbo_v2_5",
            tts_voice_id="voice-id",
        ),
    )

    calls_dir = tmp_path / "calls"

    # Patch save_call_summary to write into tmp_path
    def mock_save(room: str, summary: dict[str, Any], base_dir: Any = None) -> Path:
        return save_call_summary(room, summary, base_dir=calls_dir)

    monkeypatch.setattr("agent.worker.save_call_summary", mock_save)

    # Patch extract_answers to return mock extraction
    async def mock_extract(
        transcript: str, question_keys: list[str], **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "answers": {
                "name": "Тарас",
                "service": "IT",
                "timeline": None,
                "preferred_call_time": None,
            },
            "missing": ["timeline", "preferred_call_time"],
        }

    monkeypatch.setattr("agent.worker.extract_answers", mock_extract)

    summary = await post_call_extraction_and_summary(
        room_name="test-room-extraction",
        history=chat_ctx,
        state=state,
        metrics_collector=metrics_collector,
        script=script,
    )

    assert summary["room"] == "test-room-extraction"
    assert summary["answers"]["name"] == "Тарас"
    assert summary["missing"] == ["timeline", "preferred_call_time"]
    assert state.answers["name"] == "Тарас"
    assert (calls_dir / "test-room-extraction.json").exists()


@pytest.mark.asyncio
async def test_entrypoint_join_timeout_reports_no_answer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies join-timeout reports ended=no_answer once, no extraction, and closes."""
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-123", "mode": "browser"})
    mock_ctx.room.name = "test-timeout-room"
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(side_effect=TimeoutError())
    mock_ctx.api.room.delete_room = AsyncMock()

    await entrypoint(mock_ctx)

    mock_reporter.report_event.assert_called_once_with(
        "no_answer",
        f"Participant did not join within {settings.participant_join_timeout} s",
        {"timeout_seconds": settings.participant_join_timeout},
    )
    mock_reporter.report_ended.assert_called_once_with(
        status="no_answer", duration_seconds=0, latency_summary=ANY
    )
    mock_reporter.report_extraction.assert_not_called()
    mock_reporter.aclose.assert_awaited_once()

    # Call registered shutdown callback; must not invoke report_ended again
    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()
    mock_reporter.report_ended.assert_called_once()
    mock_reporter.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_entrypoint_user_hangup_reports_completed_and_ended_by_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies user hangup records ended=completed and call_ended with
    ended_by=user.
    """
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "answers": {"name": "Bob"},
            "missing": [],
        }

    monkeypatch.setattr("agent.worker.post_call_extraction_and_summary", mock_extract)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-user-hangup"})
    mock_ctx.room.name = "test-user-hangup-room"
    mock_ctx.connect = AsyncMock()

    mock_participant = MagicMock()
    mock_participant.identity = "user-1"
    mock_participant.name = "Bob"
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)

    with (
        patch("agent.worker.AgentSession.start", new_callable=AsyncMock),
        patch("agent.worker.AgentSession.say", new_callable=AsyncMock),
    ):
        await entrypoint(mock_ctx)

    # Trigger shutdown callback (simulating user disconnect: call_ended=False)
    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()

    mock_reporter.report_event.assert_any_call(
        "call_ended", "Call ended by user", {"ended_by": "user"}
    )
    mock_reporter.report_ended.assert_called_once_with(
        status="completed", duration_seconds=ANY, latency_summary=ANY
    )
    mock_reporter.report_extraction.assert_called_once_with(
        extracted_answers={"name": "Bob"},
        extracted_missing=[],
        extraction_status="ok",
    )
    mock_reporter.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_entrypoint_agent_hangup_reports_ended_by_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies agent termination records call_ended event with ended_by=agent."""
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "answers": {}, "missing": []}

    monkeypatch.setattr("agent.worker.post_call_extraction_and_summary", mock_extract)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-agent-hangup"})
    mock_ctx.room.name = "test-agent-hangup-room"
    mock_ctx.connect = AsyncMock()
    mock_participant = MagicMock(identity="user-2", name="Alice")
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)

    captured_session = None
    orig_attach = SessionMetricsCollector.attach_to_session

    def spy_attach(self: Any, sess: Any) -> None:
        nonlocal captured_session
        captured_session = sess
        orig_attach(self, sess)

    monkeypatch.setattr(SessionMetricsCollector, "attach_to_session", spy_attach)

    with (
        patch("agent.worker.AgentSession.start", new_callable=AsyncMock),
        patch("agent.worker.AgentSession.say", new_callable=AsyncMock),
    ):
        await entrypoint(mock_ctx)

    assert captured_session is not None
    # Simulate end_call tool having terminated the call
    captured_session.userdata.call_ended = True

    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()

    mock_reporter.report_event.assert_any_call(
        "call_ended", "Call ended by agent", {"ended_by": "agent"}
    )
    mock_reporter.report_ended.assert_called_once_with(
        status="completed", duration_seconds=ANY, latency_summary=ANY
    )


@pytest.mark.asyncio
async def test_entrypoint_extraction_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies post-call extraction error reports extraction_failed with all
    question keys.
    """
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    async def mock_extract_fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("LLM rate limit reached")

    monkeypatch.setattr(
        "agent.worker.post_call_extraction_and_summary", mock_extract_fail
    )

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-fail"})
    mock_ctx.room.name = "test-fail-room"
    mock_ctx.connect = AsyncMock()
    mock_participant = MagicMock(identity="user-3", name="Charlie")
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)

    with (
        patch("agent.worker.AgentSession.start", new_callable=AsyncMock),
        patch("agent.worker.AgentSession.say", new_callable=AsyncMock),
    ):
        await entrypoint(mock_ctx)

    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()

    script = load_script()
    mock_reporter.report_extraction.assert_called_once_with(
        extracted_answers={},
        extracted_missing=list(script.question_fields),
        extraction_status="extraction_failed",
    )


@pytest.mark.asyncio
async def test_entrypoint_on_session_close_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies metrics_collector.on_session_close() is called exactly once per
    session.
    """
    from agent.worker import entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    close_calls = 0
    orig_close = SessionMetricsCollector.on_session_close

    def spy_close(self: Any) -> dict[str, Any]:
        nonlocal close_calls
        close_calls += 1
        return orig_close(self)

    monkeypatch.setattr(SessionMetricsCollector, "on_session_close", spy_close)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-close-once"})
    mock_ctx.room.name = "test-close-room"
    mock_ctx.connect = AsyncMock()
    mock_participant = MagicMock(identity="user-4", name="Dana")
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "answers": {}, "missing": []}

    monkeypatch.setattr("agent.worker.post_call_extraction_and_summary", mock_extract)

    with (
        patch("agent.worker.AgentSession.start", new_callable=AsyncMock),
        patch("agent.worker.AgentSession.say", new_callable=AsyncMock),
    ):
        await entrypoint(mock_ctx)

    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()

    assert close_calls == 1


@pytest.mark.asyncio
async def test_entrypoint_session_close_event_triggers_shutdown_and_single_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies session 'close' event deletes room, triggers ctx.shutdown,
    and finalizes once.
    """
    from livekit.agents.voice import CloseEvent

    from agent.worker import _close_tasks, entrypoint

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-key")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")

    mock_reporter = MagicMock()
    mock_reporter.is_noop = False
    mock_reporter.aclose = AsyncMock()
    monkeypatch.setattr("agent.worker.CallReporter", lambda **kw: mock_reporter)

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "answers": {"name": "Eve"}, "missing": []}

    monkeypatch.setattr("agent.worker.post_call_extraction_and_summary", mock_extract)

    mock_ctx = MagicMock()
    mock_ctx.job.metadata = json.dumps({"call_id": "test-call-close-event"})
    mock_ctx.room.name = "test-close-room"
    mock_ctx.connect = AsyncMock()
    mock_ctx.api.room.delete_room = AsyncMock()
    mock_participant = MagicMock(identity="user-5", name="Eve")
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)

    captured_session = None
    orig_attach = SessionMetricsCollector.attach_to_session

    def spy_attach(self: Any, sess: Any) -> None:
        nonlocal captured_session
        captured_session = sess
        orig_attach(self, sess)

    monkeypatch.setattr(SessionMetricsCollector, "attach_to_session", spy_attach)

    start_kwargs: dict[str, Any] = {}

    async def spy_start(*args: Any, **kwargs: Any) -> None:
        start_kwargs.update(kwargs)

    with (
        patch("agent.worker.AgentSession.start", side_effect=spy_start),
        patch("agent.worker.AgentSession.say", new_callable=AsyncMock),
    ):
        await entrypoint(mock_ctx)

    assert captured_session is not None
    # 1. Verify delete_room_on_close was passed in browser mode
    assert "room_options" in start_kwargs
    assert start_kwargs["room_options"].delete_room_on_close is True

    # 2. Emit the 'close' event on the session
    close_event = CloseEvent(reason="user_initiated")
    captured_session.emit("close", close_event)

    # Allow the scheduled background task to run
    if _close_tasks:
        await asyncio.gather(*list(_close_tasks))

    # 3. Verify room deletion and ctx.shutdown were called
    mock_ctx.api.room.delete_room.assert_awaited_once()
    assert mock_ctx.api.room.delete_room.call_args[0][0].room == "test-close-room"
    mock_ctx.shutdown.assert_called_once_with(
        reason=f"session closed: {close_event.reason}"
    )

    # 4. Trigger the shutdown callback (simulating framework shutdown lifecycle)
    registered_cb = mock_ctx.add_shutdown_callback.call_args[0][0]
    await registered_cb()

    # 5. Verify call_ended event, /ended, extraction, and aclose
    mock_reporter.report_event.assert_any_call(
        "call_ended", "Call ended by user", {"ended_by": "user"}
    )
    mock_reporter.report_ended.assert_called_once_with(
        status="completed", duration_seconds=ANY, latency_summary=ANY
    )
    mock_reporter.report_extraction.assert_called_once_with(
        extracted_answers={"name": "Eve"},
        extracted_missing=[],
        extraction_status="ok",
    )
    mock_reporter.aclose.assert_awaited_once()

    # 6. Verify calling registered_cb again does not double-finalize
    await registered_cb()
    mock_reporter.report_ended.assert_called_once()
    mock_reporter.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_end_call_guard(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that when ending_in_progress is True,
    a concurrent end_call returns None with a log line.
    """
    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.aclose = AsyncMock()

    state = SessionState(closing_phrase="Bye", ending_in_progress=True)
    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session

    with caplog.at_level("INFO"):
        res = await end_call(mock_context)

    assert res is None
    assert any(
        "end_call rejected: ending already in progress" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_end_call_grace_aborts_on_user_transcript(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that user speech during grace period aborts end_call
    and returns None.
    """
    monkeypatch.setattr(settings, "end_call_grace_period", 0.05)
    state = SessionState(closing_phrase="Bye")

    listeners: dict[str, list[Any]] = {}

    def mock_on(event: str, cb: Any) -> None:
        listeners.setdefault(event, []).append(cb)

    def mock_off(event: str, cb: Any) -> None:
        if event in listeners and cb in listeners[event]:
            listeners[event].remove(cb)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.on = MagicMock(side_effect=mock_on)
    mock_session.off = MagicMock(side_effect=mock_off)
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    async def emit_speech_during_grace() -> None:
        await asyncio.sleep(0.01)
        mock_ev = MagicMock()
        mock_ev.transcript = "Зачекайте, будь ласка!"
        mock_ev.is_final = True
        for cb in list(listeners.get("user_input_transcribed", [])):
            cb(mock_ev)

    asyncio.create_task(emit_speech_during_grace())

    with caplog.at_level("INFO"):
        res = await end_call(mock_context)

    assert res is None
    assert state.call_ended is False
    assert mock_session.off.called
    assert any(
        "end_call aborted: user spoke during grace period" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_end_call_grace_ignores_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that noise (user_state speaking with no transcript)
    during grace window is ignored and call terminates.
    """
    monkeypatch.setattr(settings, "end_call_grace_period", 0.02)
    monkeypatch.setattr(settings, "end_call_noise_timeout_cap", 0.03)
    state = SessionState(closing_phrase="Bye")

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    listeners: dict[str, list[Any]] = {}

    def mock_on(event: str, cb: Any) -> None:
        listeners.setdefault(event, []).append(cb)

    def mock_off(event: str, cb: Any) -> None:
        if event in listeners and cb in listeners[event]:
            listeners[event].remove(cb)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.on = MagicMock(side_effect=mock_on)
    mock_session.off = MagicMock(side_effect=mock_off)
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    async def simulate_noise_during_grace() -> None:
        await asyncio.sleep(0.01)
        # Background noise triggers user_state = speaking briefly
        mock_session.user_state = "speaking"
        # STT emits empty/whitespace transcript
        mock_ev = MagicMock()
        mock_ev.transcript = "   "
        mock_ev.is_final = True
        for cb in list(listeners.get("user_input_transcribed", [])):
            cb(mock_ev)
        await asyncio.sleep(0.01)
        mock_session.user_state = "listening"

    asyncio.create_task(simulate_noise_during_grace())

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    assert mock_session.off.called


@pytest.mark.asyncio
async def test_closing_phrase_interrupted_resets_call_ended() -> None:
    """Verifies that when closing phrase is interrupted by caller speech,
    call_ended is reset to False and goodbye_count increments.
    """
    state = SessionState(closing_phrase="Дякую за розмову! До побачення!")

    listeners: dict[str, list[Any]] = {}

    def mock_on(event: str, cb: Any) -> None:
        listeners.setdefault(event, []).append(cb)

    def mock_off(event: str, cb: Any) -> None:
        if event in listeners and cb in listeners[event]:
            listeners[event].remove(cb)

    mock_handle = MagicMock()
    mock_handle.interrupted = True

    async def mock_wait_playout() -> None:
        # Simulate user speech transcript arriving during playout
        mock_ev = MagicMock()
        mock_ev.transcript = "А яка ціна?"
        mock_ev.is_final = True
        for cb in list(listeners.get("user_input_transcribed", [])):
            cb(mock_ev)

    mock_handle.wait_for_playout = AsyncMock(side_effect=mock_wait_playout)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.on = MagicMock(side_effect=mock_on)
    mock_session.off = MagicMock(side_effect=mock_off)
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    task = terminate_call(
        mock_session, state, phrase=state.closing_phrase, allow_interruptions=True
    )
    assert task is not None
    await task

    # Real user speech interrupted the closing phrase
    assert state.call_ended is False
    assert state.goodbye_count == 1
    mock_session.aclose.assert_not_called()
    assert mock_session.off.called


@pytest.mark.asyncio
async def test_end_call_grace_ignores_interim_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that interim transcripts (is_final=False) during grace
    window are ignored and call terminates.
    """
    monkeypatch.setattr(settings, "end_call_grace_period", 0.02)
    state = SessionState(closing_phrase="Bye")

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    listeners: dict[str, list[Any]] = {}

    def mock_on(event: str, cb: Any) -> None:
        listeners.setdefault(event, []).append(cb)

    def mock_off(event: str, cb: Any) -> None:
        if event in listeners and cb in listeners[event]:
            listeners[event].remove(cb)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.on = MagicMock(side_effect=mock_on)
    mock_session.off = MagicMock(side_effect=mock_off)
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    async def simulate_interim_during_grace() -> None:
        await asyncio.sleep(0.01)
        mock_ev = MagicMock()
        mock_ev.transcript = "Зачекайте"
        mock_ev.is_final = False  # Interim event only!
        for cb in list(listeners.get("user_input_transcribed", [])):
            cb(mock_ev)

    asyncio.create_task(simulate_interim_during_grace())

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    assert mock_session.off.called


@pytest.mark.asyncio
async def test_closing_phrase_interim_transcript_does_not_abort() -> None:
    """Verifies that interim transcripts (is_final=False) during closing
    phrase playout do not abort termination.
    """
    state = SessionState(closing_phrase="Дякую за розмову! До побачення!")

    listeners: dict[str, list[Any]] = {}

    def mock_on(event: str, cb: Any) -> None:
        listeners.setdefault(event, []).append(cb)

    def mock_off(event: str, cb: Any) -> None:
        if event in listeners and cb in listeners[event]:
            listeners[event].remove(cb)

    mock_handle = MagicMock()
    mock_handle.interrupted = True

    async def mock_wait_playout() -> None:
        mock_ev = MagicMock()
        mock_ev.transcript = "Але..."
        mock_ev.is_final = False  # Interim only
        for cb in list(listeners.get("user_input_transcribed", [])):
            cb(mock_ev)

    mock_handle.wait_for_playout = AsyncMock(side_effect=mock_wait_playout)

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.on = MagicMock(side_effect=mock_on)
    mock_session.off = MagicMock(side_effect=mock_off)
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    task = terminate_call(
        mock_session, state, phrase=state.closing_phrase, allow_interruptions=True
    )
    assert task is not None
    await task

    # Session terminates because only interim transcripts were received
    assert state.call_ended is True
    assert state.goodbye_count == 0
    mock_session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminate_call_default_uninterruptible_and_end_call_interruptible() -> (
    None
):
    """Verifies terminate_call defaults to allow_interruptions=False
    and end_call uses True.
    """
    state1 = SessionState(error_phrase="Помилка")
    mock_handle1 = MagicMock()
    mock_handle1.wait_for_playout = AsyncMock()
    mock_session1 = MagicMock()
    mock_session1.say = MagicMock(return_value=mock_handle1)
    mock_session1.aclose = AsyncMock()

    # Default terminate_call call
    task = terminate_call(mock_session1, state1, state1.error_phrase)
    assert task is not None
    await task
    mock_session1.say.assert_called_once_with("Помилка", allow_interruptions=False)

    # end_call tool call
    state2 = SessionState(closing_phrase="Бувайте")
    mock_handle2 = MagicMock()
    mock_handle2.wait_for_playout = AsyncMock()
    mock_session2 = MagicMock()
    mock_session2.user_state = "listening"
    mock_session2.say = MagicMock(return_value=mock_handle2)
    mock_session2.aclose = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.userdata = state2
    mock_ctx.session = mock_session2
    mock_ctx.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_ctx)

    for bg_task in list(state2.background_tasks):
        await bg_task
    mock_session2.say.assert_called_once_with("Бувайте", allow_interruptions=True)


@pytest.mark.asyncio
async def test_second_end_call_uses_short_goodbye() -> None:
    """Verifies that a subsequent end_call after interruption uses short_goodbye."""
    state = SessionState(
        closing_phrase="Дякую за розмову! Менеджер зв'яжеться з вами найближчим часом.",
        short_goodbye="До побачення!",
        goodbye_count=1,
    )

    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    with pytest.raises(StopResponse):
        await end_call(mock_context)

    assert state.call_ended is True
    for task in list(state.background_tasks):
        await task

    mock_session.say.assert_called_once_with("До побачення!", allow_interruptions=True)


@pytest.mark.asyncio
async def test_pending_question_guard_on_second_end_call() -> None:
    """Verifies pending question guard still blocks second end_call
    if question was asked.
    """
    state = SessionState(
        closing_phrase="Дякую!",
        short_goodbye="До побачення!",
        goodbye_count=1,
    )

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="assistant", content="О котрій годині зателефонувати?")

    mock_session = MagicMock()
    mock_session.user_state = "listening"
    mock_session.history = chat_ctx
    mock_session.say = MagicMock()
    mock_session.aclose = AsyncMock()

    mock_context = MagicMock()
    mock_context.userdata = state
    mock_context.session = mock_session
    mock_context.wait_for_playout = AsyncMock()

    res = await end_call(mock_context)
    assert res is None
    assert state.call_ended is False
    mock_session.say.assert_not_called()


def test_endpointing_mode_and_keyterms_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies endpointing mode is propagated and STT keyterms are
    passed to Deepgram nova-3.
    """
    monkeypatch.setattr(settings, "endpointing_mode", "dynamic")
    opts = get_turn_handling()
    assert opts["endpointing"]["mode"] == "dynamic"

    monkeypatch.setattr(settings, "stt_provider", "deepgram")
    monkeypatch.setattr(settings, "stt_model", "nova-3")
    monkeypatch.setattr(settings, "stt_keyterms", "Solaris, CRM, автоматизація")

    with patch("livekit.plugins.deepgram.STT") as mock_stt_cls:
        get_stt("uk")
        mock_stt_cls.assert_called_once()
        _, kwargs = mock_stt_cls.call_args
        assert kwargs.get("keyterm") == ["Solaris", "CRM", "автоматизація"]


def test_eval_call_markdown_output() -> None:
    """Verifies eval_call row formatting, agent role, extraction deduplication,
    and '-' for empty latency.
    """
    from app.models import Call, CallEvent, CallTurn
    from scripts.eval_call import (
        count_error_events,
        count_split_turns,
        format_eval_row,
        format_extraction_status,
    )

    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)
    turns = [
        CallTurn(turn_index=0, role="agent", content="Вітаю!", timestamp=now),
        CallTurn(turn_index=1, role="user", content="Добрий день", timestamp=now),
        CallTurn(turn_index=2, role="user", content="Я щодо CRM", timestamp=now),
        CallTurn(turn_index=3, role="agent", content="Яка послуга?", timestamp=now),
    ]
    events = [
        CallEvent(
            event_type="stt_error", message="Deepgram socket timeout", timestamp=now
        ),
    ]
    call = Call(
        id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        room_name="call_+380501234567",
        created_at=now,
        latency_summary={
            "turns": 2,
            "p50_measured": 1.120,
            "max_measured": 1.450,
            "p50_computed": 1.250,
        },
        extracted_answers={"name": "Іван", "service": "CRM", "empty_field": ""},
        extracted_missing=["timeline", "service", "preferred_call_time"],
        extraction_status="completed",
        turns=turns,
        events=events,
        status="completed",
    )

    assert count_split_turns(turns) == 1
    assert count_error_events(events) == 1

    # extraction: answered = 2 ("name", "service" - "empty_field" excluded)
    # missing = 2 ("timeline", "preferred_call_time" - "service" already in answered)
    # total = 2 + 2 = 4 -> "2/4"
    assert format_extraction_status(call) == "2/4"

    row = format_eval_row(call, variant="baseline")
    assert "| 11111111 |" in row  # Run label = first 8 chars of call id
    assert "| 2026-10-24 |" in row
    assert "| baseline |" in row
    assert "| 2 |" in row
    assert "| 1.120s |" in row
    assert "| 1.450s |" in row
    assert "| 1.250s |" in row
    assert "| 1 |" in row  # split turns
    assert "| - | - |" in row  # Lost words = "-", STT errors = "-"
    assert "| 2/4 |" in row  # extraction
    assert "errors=1, ended_by=completed" in row  # Notes

    # Test default variant when not given
    row_no_var = format_eval_row(call)
    assert "| 11111111 | 2026-10-24 | - | 2 |" in row_no_var

    # Test empty latency summary prints "-"
    empty_call = Call(
        id=uuid.UUID("22222222-2222-3333-4444-555555555555"),
        created_at=now,
        latency_summary={"turns": 0, "avg": 0.0, "p50": 0.0, "max": 0.0},
        status="completed",
    )
    row_empty = format_eval_row(empty_call)
    expected_empty = (
        "| 22222222 | 2026-10-24 | - | 0 | - | - | - | 0 | - | - | - | "
        "errors=0, ended_by=completed |"
    )
    assert expected_empty in row_empty


def test_eval_call_agent_role_and_extraction_edge_cases() -> None:
    """Tests agent role fallback turn count and extraction deduplication edge cases."""
    from app.models import Call, CallTurn
    from scripts.eval_call import format_eval_row, format_extraction_status

    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)

    # 1. Agent role is "agent" - fallback when latency_summary has no turns count
    call_no_lat = Call(
        id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        created_at=now,
        latency_summary=None,
        turns=[
            CallTurn(turn_index=0, role="agent", content="Hello", timestamp=now),
            CallTurn(turn_index=1, role="user", content="Hi", timestamp=now),
            CallTurn(turn_index=2, role="agent", content="Help?", timestamp=now),
            CallTurn(turn_index=3, role="user", content="Bye", timestamp=now),
        ],
        status="completed",
    )
    row = format_eval_row(call_no_lat)
    # Turns column should be 2 (counting only role="agent")
    assert "| 33333333 | 2026-10-24 | - | 2 |" in row

    # 2. Extraction deduplication and empty value filtering
    call_overlap = Call(
        id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
        created_at=now,
        extracted_answers={
            "f1": "val1",
            "f2": None,
            "f3": "",
            "f4": "   ",
            "f5": "val5",
        },
        extracted_missing=["f1", "f2", "f5", "f6"],
    )
    # answered: f1, f5 -> 2
    # missing: f2, f6 (f1 and f5 are already in answered so not double counted) -> 2
    # total = 2 + 2 = 4 -> "2/4"
    assert format_extraction_status(call_overlap) == "2/4"
