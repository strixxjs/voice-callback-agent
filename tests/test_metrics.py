"""Tests for latency metrics aggregation, SessionMetricsCollector, and settings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from livekit.agents import llm
from livekit.agents.voice.events import ConversationItemAddedEvent

from agent.metrics import (
    ActiveModels,
    SessionMetricsCollector,
    compute_latency_summary,
)
from agent.prompts import load_script
from agent.worker import (
    DEFAULT_ELEVENLABS_VOICE_ID,
    resolve_stt_model,
    resolve_tts_model,
    resolve_tts_voice,
)
from app.config import Settings
from app.services.calls import compute_call_latency_summary


def _create_test_models() -> ActiveModels:
    return ActiveModels(
        llm_provider="google",
        llm_model="gemini-3.5-flash-lite",
        stt_provider="deepgram",
        stt_model="nova-3",
        tts_provider="elevenlabs",
        tts_model="eleven_turbo_v2_5",
        tts_voice_id="hpp4J3VqNfWAUOO0d1Us",
    )


def test_compute_latency_summary_standard() -> None:
    """Verifies calculation of avg, p50, and max for typical response latencies."""
    latencies = [1.0, 2.0, 3.0]
    summary = compute_latency_summary(latencies)
    assert summary == {"turns": 3, "avg": 2.0, "p50": 2.0, "max": 3.0}


def test_compute_latency_summary_even_p50() -> None:
    """Verifies median calculation when number of elements is even."""
    latencies = [1.0, 2.0, 4.0, 5.0]
    summary = compute_latency_summary(latencies)
    assert summary == {"turns": 4, "avg": 3.0, "p50": 3.0, "max": 5.0}


def test_compute_latency_summary_single_turn() -> None:
    """Verifies summary for a single turn."""
    summary = compute_latency_summary([1.234])
    assert summary == {"turns": 1, "avg": 1.234, "p50": 1.234, "max": 1.234}


def test_compute_latency_summary_empty() -> None:
    """Verifies that empty latency list returns zeroed summary metrics."""
    summary = compute_latency_summary([])
    assert summary == {"turns": 0, "avg": 0.0, "p50": 0.0, "max": 0.0}


def test_compute_latency_summary_filters_invalid() -> None:
    """Verifies that non-numeric or None values are gracefully filtered out."""
    summary = compute_latency_summary([1.0, None, 3.0, "invalid"])  # type: ignore[list-item]
    assert summary == {"turns": 2, "avg": 2.0, "p50": 2.0, "max": 3.0}


def test_settings_m3_defaults() -> None:
    """Verifies Settings has None for provider models, keeping defaults decoupled."""
    test_settings = Settings(_env_file=None)
    assert test_settings.stt_provider == "deepgram"
    assert test_settings.stt_model is None
    assert test_settings.tts_provider == "elevenlabs"
    assert test_settings.tts_model is None
    assert test_settings.tts_voice_id is None
    assert test_settings.endpointing_mode == "fixed"
    assert test_settings.endpointing_min_delay == 1.0
    assert test_settings.endpointing_max_delay == 3.0
    assert test_settings.preemptive_generation is True
    assert test_settings.end_call_grace_period == 1.2
    assert test_settings.end_call_noise_timeout_cap == 3.0
    assert test_settings.stt_keyterms is None


def test_provider_model_and_voice_resolution() -> None:
    """Verifies that provider defaults resolve for OpenAI, Deepgram, ElevenLabs."""
    # OpenAI defaults when settings models/voices are None
    assert resolve_stt_model("openai", None) == "whisper-1"
    assert resolve_tts_model("openai", None) == "tts-1"
    assert resolve_tts_voice("openai", None) == "alloy"

    # Deepgram + ElevenLabs defaults when settings models/voices are None
    assert resolve_stt_model("deepgram", None) == "nova-3"
    assert resolve_tts_model("elevenlabs", None) == "eleven_turbo_v2_5"
    assert resolve_tts_voice("elevenlabs", None) == DEFAULT_ELEVENLABS_VOICE_ID

    # Explicit values override defaults
    assert resolve_stt_model("openai", "whisper-custom") == "whisper-custom"
    assert resolve_tts_model("openai", "tts-1-hd") == "tts-1-hd"
    assert resolve_tts_voice("openai", "nova") == "nova"
    assert resolve_stt_model("deepgram", "nova-2") == "nova-2"
    assert resolve_tts_model("elevenlabs", "eleven_flash_v2_5") == "eleven_flash_v2_5"
    assert resolve_tts_voice("elevenlabs", "custom-voice-id") == "custom-voice-id"


def test_session_metrics_collector_sdk_source(tmp_path: Path) -> None:
    """Verifies that e2e_latency from SDK is recorded with e2e_source='sdk'."""
    collector = SessionMetricsCollector(
        room_name="test-room-101",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # 1. Initial greeting from assistant (no prior user speech) -> must be skipped
    greeting_msg = llm.ChatMessage(
        role="assistant",
        content=["Привіт, я Анна."],
    )
    greeting_msg.metrics = {"tts_node_ttfb": 0.25}
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=greeting_msg))
    assert collector.turn_index == 0
    assert len(collector.turn_latencies) == 0

    # 2. User turn
    user_msg = llm.ChatMessage(
        role="user",
        content=["Мене звати Олександр."],
    )
    user_msg.metrics = {
        "end_of_turn_delay": 0.521,
        "transcription_delay": 0.315,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg))
    assert collector._last_user_metrics is not None

    # 3. Assistant response with SDK e2e_latency
    assistant_msg = llm.ChatMessage(
        role="assistant",
        content=["Дуже приємно, Олександре."],
    )
    assistant_msg.metrics = {
        "llm_node_ttft": 0.412,
        "tts_node_ttfb": 0.284,
        "e2e_latency": 1.217,
    }
    collector._on_conversation_item_added(
        ConversationItemAddedEvent(item=assistant_msg)
    )

    assert collector.turn_index == 1
    assert collector.turn_latencies == [1.217]

    # Verify JSONL content
    jsonl_file = tmp_path / "test-room-101.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["room"] == "test-room-101"
    assert record["turn_index"] == 1
    assert record["end_of_utterance_delay"] == 0.521
    assert record["stt_duration"] == 0.315
    assert record["llm_ttft"] == 0.412
    assert record["tts_ttfb"] == 0.284
    assert record["total_response_latency"] == 1.217
    assert record["e2e_source"] == "sdk"
    assert record["stt_model"] == "nova-3"
    assert record["tts_model"] == "eleven_turbo_v2_5"
    assert record["tts_voice_id"] == "hpp4J3VqNfWAUOO0d1Us"
    # Ensure no transcript or participant identity leaks
    assert "content" not in record
    assert "transcript" not in record
    assert "participant" not in record

    # Verify session close summary
    summary = collector.on_session_close()
    assert summary["turns"] == 1
    assert summary["missing_latency_turns"] == 0
    assert summary["avg"] == 1.217
    assert summary["p50"] == 1.217
    assert summary["max"] == 1.217


def test_session_metrics_collector_fallback_computed(tmp_path: Path) -> None:
    """Verifies that missing e2e_latency with all components triggers fallback sum."""
    collector = SessionMetricsCollector(
        room_name="test-room-fallback",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg = llm.ChatMessage(role="user", content=["Так."])
    user_msg.metrics = {
        "end_of_turn_delay": 0.500,
        "transcription_delay": 0.200,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg))

    assistant_msg = llm.ChatMessage(role="assistant", content=["Зрозуміло."])
    # Omit e2e_latency to test fallback formula
    assistant_msg.metrics = {
        "llm_node_ttft": 0.350,
        "tts_node_ttfb": 0.150,
    }
    collector._on_conversation_item_added(
        ConversationItemAddedEvent(item=assistant_msg)
    )

    assert collector.turn_index == 1
    # Formula: 0.500 + 0.350 + 0.150 = 1.000
    assert collector.turn_latencies == [1.0]

    jsonl_file = tmp_path / "test-room-fallback.jsonl"
    record = json.loads(jsonl_file.read_text(encoding="utf-8").strip())
    assert record["total_response_latency"] == 1.0
    assert record["e2e_source"] == "computed"


def test_session_metrics_collector_partial_missing(tmp_path: Path) -> None:
    """Verifies that missing e2e_latency with partial sub-latencies is 'missing'."""
    collector = SessionMetricsCollector(
        room_name="test-room-missing",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg = llm.ChatMessage(role="user", content=["Алло"])
    # user_msg missing end_of_turn_delay
    user_msg.metrics = {
        "transcription_delay": 0.250,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg))

    assistant_msg = llm.ChatMessage(role="assistant", content=["Доброго дня."])
    assistant_msg.metrics = {
        "llm_node_ttft": 0.300,
        # missing tts_node_ttfb and e2e_latency
    }
    collector._on_conversation_item_added(
        ConversationItemAddedEvent(item=assistant_msg)
    )

    assert collector.turn_index == 1
    # Must NOT invent zeros: total is None and not added to turn_latencies
    assert collector.turn_latencies == []
    assert collector.missing_latency_turns == 1

    jsonl_file = tmp_path / "test-room-missing.jsonl"
    record = json.loads(jsonl_file.read_text(encoding="utf-8").strip())
    assert record["total_response_latency"] is None
    assert record["e2e_source"] == "missing"

    summary = collector.on_session_close()
    assert summary["turns"] == 0
    assert summary["missing_latency_turns"] == 1
    assert summary["avg"] == 0.0


def test_metrics_collector_never_breaks_call(tmp_path: Path) -> None:
    """Verifies that any error in metrics collection or writing is swallowed."""
    collector = SessionMetricsCollector(
        room_name="test-error-room",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # 1. Simulate an unhandled file system / OS error during JSONL write
    with patch("builtins.open", side_effect=OSError("Disk full or read-only")):
        user_msg = llm.ChatMessage(role="user", content=["Тест"])
        user_msg.metrics = {"end_of_turn_delay": 0.5}
        collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg))

        assistant_msg = llm.ChatMessage(role="assistant", content=["Відповідь"])
        assistant_msg.metrics = {"llm_node_ttft": 0.3}

        # Must NOT raise exception despite OSError
        collector._on_conversation_item_added(
            ConversationItemAddedEvent(item=assistant_msg)
        )

    # 2. Simulate completely malformed / unexpected object passed to event listener
    corrupt_event = MagicMock()
    corrupt_event.item = 12345  # Not a ChatMessage

    # Must NOT raise exception
    collector._on_conversation_item_added(corrupt_event)

    # 3. Session close with internal error simulated
    with patch(
        "agent.metrics.compute_latency_summary",
        side_effect=RuntimeError("Math crash"),
    ):
        summary = collector.on_session_close()
        assert summary["turns"] == 0
        assert summary["avg"] == 0.0


def test_session_metrics_collector_split_user_turn(tmp_path: Path) -> None:
    """Verifies that split user utterances preserve earlier non-empty metrics."""
    collector = SessionMetricsCollector(
        room_name="test-room-split",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # 1. First part of split user utterance (has metrics from STT)
    user_msg_1 = llm.ChatMessage(
        role="user",
        content=["Приблизно знов впродовж трьох."],
    )
    user_msg_1.metrics = {
        "end_of_turn_delay": 0.794,
        "transcription_delay": 0.792,
        "speech_start_time": 1792747960.51,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_1))

    # 2. Second part of split user utterance (STT split, metrics empty or None values)
    user_msg_2 = llm.ChatMessage(
        role="user",
        content=["Днів."],
    )
    user_msg_2.metrics = {
        "speech_start_time": None,
        "end_of_turn_delay": None,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_2))

    # 3. Assistant response
    assistant_msg = llm.ChatMessage(
        role="assistant",
        content=["Чудово. О котрій годині зателефонувати?"],
    )
    assistant_msg.metrics = {
        "llm_node_ttft": 0.427,
        "tts_node_ttfb": 0.214,
        "e2e_latency": 2.191,
    }
    collector._on_conversation_item_added(
        ConversationItemAddedEvent(item=assistant_msg)
    )

    assert collector.turn_index == 1
    assert collector.turn_latencies == [2.191]

    # Verify JSONL content preserved user_msg_1 metrics
    jsonl_file = tmp_path / "test-room-split.jsonl"
    record = json.loads(jsonl_file.read_text(encoding="utf-8").strip())
    assert record["turn_index"] == 1
    assert record["end_of_utterance_delay"] == 0.794
    assert record["stt_duration"] == 0.792
    assert record["llm_ttft"] == 0.427
    assert record["tts_ttfb"] == 0.214
    assert record["total_response_latency"] == 2.191
    assert record["e2e_source"] == "sdk"


def test_script_short_goodbye_loading() -> None:
    """Verifies script.yaml defines short_goodbye and load_script populates it."""
    script = load_script()
    assert script.short_goodbye == "До побачення!"
    assert script.closing_phrase != ""


def test_metrics_measured_and_computed_latency(tmp_path: Path) -> None:
    """Verifies measured latency tracking, computed latency fallback,
    and summary metrics.
    """
    collector = SessionMetricsCollector(
        room_name="test-room-meas",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # Turn 1: Measured latency from assistant ChatMessage.metrics["e2e_latency"]
    user_msg_1 = llm.ChatMessage(role="user", content=["Доброго дня"])
    user_msg_1.metrics = {
        "end_of_turn_delay": 0.500,
        "transcription_delay": 0.200,
        "stopped_speaking_at": 100.0,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_1))

    asst_msg_1 = llm.ChatMessage(role="assistant", content=["Вітаю!"])
    asst_msg_1.metrics = {
        "llm_node_ttft": 0.300,
        "tts_node_ttfb": 0.200,
        "started_speaking_at": 100.850,
        "e2e_latency": 0.850,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_1))

    assert collector.turn_index == 1
    assert collector.measured_latencies == [0.850]
    # computed would be: 0.500 + 0.300 + 0.200 = 1.000
    assert collector.computed_latencies == [1.000]
    assert collector.turn_latencies == [0.850]

    # Turn 2: Measured latency from started_speaking_at - stopped_speaking_at
    user_msg_2 = llm.ChatMessage(role="user", content=["Мене цікавить CRM"])
    user_msg_2.metrics = {
        "end_of_turn_delay": 0.520,
        "transcription_delay": 0.250,
        "stopped_speaking_at": 110.0,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_2))

    asst_msg_2 = llm.ChatMessage(role="assistant", content=["Чудово."])
    # Omit e2e_latency, but provide timestamps
    asst_msg_2.metrics = {
        "llm_node_ttft": 0.350,
        "tts_node_ttfb": 0.220,
        "started_speaking_at": 111.150,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_2))

    assert collector.turn_index == 2
    # 111.150 - 110.0 = 1.150
    assert collector.measured_latencies == [0.850, 1.150]
    # computed: 0.520 + 0.350 + 0.220 = 1.090
    assert collector.computed_latencies == [1.000, 1.090]

    # Turn 3: Computed fallback only (no measured latency or timestamps)
    user_msg_3 = llm.ChatMessage(role="user", content=["Так, дякую"])
    user_msg_3.metrics = {
        "end_of_turn_delay": 0.510,
        "transcription_delay": 0.180,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_3))

    asst_msg_3 = llm.ChatMessage(role="assistant", content=["До побачення!"])
    asst_msg_3.metrics = {
        "llm_node_ttft": 0.280,
        "tts_node_ttfb": 0.160,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_3))

    assert collector.turn_index == 3
    # Measured latencies remains length 2
    assert len(collector.measured_latencies) == 2
    # computed: 0.510 + 0.280 + 0.160 = 0.950
    assert collector.computed_latencies == [1.000, 1.090, 0.950]
    # Turn latencies has all 3 turns (measured when present, fallback to computed)
    assert collector.turn_latencies == [0.850, 1.150, 0.950]

    # Check summary metrics
    summary = collector.on_session_close()
    assert summary["turns"] == 2  # Primary is measured (2 measured turns)
    assert summary["p50_measured"] == 1.0  # median(0.850, 1.150) = 1.0
    assert summary["max_measured"] == 1.150
    assert summary["p50_computed"] == 1.000  # median(1.000, 1.090, 0.950) = 1.000
    assert summary["max_computed"] == 1.090
    assert summary["p50"] == summary["p50_measured"]

    # Verify JSONL lines have measured_latency and computed_latency keys
    jsonl_file = tmp_path / "test-room-meas.jsonl"
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    rec1 = json.loads(lines[0])
    assert rec1["measured_latency"] == 0.850
    assert rec1["sdk_e2e_latency"] == 0.850
    assert rec1["computed_latency"] == 1.000
    assert rec1["e2e_source"] == "measured"

    rec2 = json.loads(lines[1])
    assert rec2["measured_latency"] == 1.150
    assert rec2["sdk_e2e_latency"] is None
    assert rec2["computed_latency"] == 1.090
    assert rec2["e2e_source"] == "measured"

    rec3 = json.loads(lines[2])
    assert rec3["measured_latency"] is None
    assert rec3["computed_latency"] == 0.950
    assert rec3["e2e_source"] == "computed"


def test_metrics_scripted_turn_excluded_from_latencies(tmp_path: Path) -> None:
    """Verifies that an assistant turn with no llm_node_ttft is treated as scripted,
    logged with e2e_source='scripted', excluded from p50/max latency tracking,
    and does not increment missing_latency_turns.
    """
    collector = SessionMetricsCollector(
        room_name="test-room-scripted",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # Turn 1: Normal measured LLM turn
    user_msg_1 = llm.ChatMessage(role="user", content=["Доброго дня"])
    user_msg_1.metrics = {
        "end_of_turn_delay": 0.500,
        "transcription_delay": 0.200,
        "stopped_speaking_at": 100.0,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_1))

    asst_msg_1 = llm.ChatMessage(role="assistant", content=["Вітаю!"])
    asst_msg_1.metrics = {
        "llm_node_ttft": 0.300,
        "tts_node_ttfb": 0.200,
        "e2e_latency": 0.900,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_1))

    # Turn 2: Normal measured LLM turn
    user_msg_2 = llm.ChatMessage(role="user", content=["Цікавить CRM"])
    user_msg_2.metrics = {
        "end_of_turn_delay": 0.520,
        "transcription_delay": 0.250,
        "stopped_speaking_at": 110.0,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_2))

    asst_msg_2 = llm.ChatMessage(role="assistant", content=["Зрозумів."])
    asst_msg_2.metrics = {
        "llm_node_ttft": 0.320,
        "tts_node_ttfb": 0.210,
        "e2e_latency": 1.100,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_2))

    # Turn 3: Scripted assistant turn (closing phrase via session.say)
    # Has measured e2e inflated by grace window (2.500s), but NO llm_node_ttft
    user_msg_3 = llm.ChatMessage(role="user", content=["До побачення"])
    user_msg_3.metrics = {
        "end_of_turn_delay": 0.500,
        "stopped_speaking_at": 120.0,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=user_msg_3))

    asst_msg_3 = llm.ChatMessage(role="assistant", content=["Дякую за дзвінок!"])
    asst_msg_3.metrics = {
        "tts_node_ttfb": 0.150,
        "started_speaking_at": 122.500,
        "e2e_latency": 2.500,
    }
    collector._on_conversation_item_added(ConversationItemAddedEvent(item=asst_msg_3))

    assert collector.turn_index == 3

    # Turn 3 must NOT be in measured_latencies, computed_latencies, or turn_latencies
    assert collector.measured_latencies == [0.900, 1.100]
    assert len(collector.measured_latencies) == 2
    assert collector.turn_latencies == [0.900, 1.100]
    assert collector.missing_latency_turns == 0

    # Summary "turns" counts ONLY measured LLM turns (2), and p50/max excludes 2.500s
    summary = collector.on_session_close()
    assert summary["turns"] == 2
    assert summary["p50_measured"] == 1.000  # median(0.900, 1.100) = 1.000
    assert summary["max_measured"] == 1.100  # NOT 2.500
    assert summary["p50"] == 1.000
    assert summary["max"] == 1.100
    assert summary["missing_latency_turns"] == 0

    # Verify JSONL lines: turn 3 has e2e_source="scripted"
    jsonl_file = tmp_path / "test-room-scripted.jsonl"
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    rec3 = json.loads(lines[2])
    assert rec3["e2e_source"] == "scripted"
    assert rec3["llm_ttft"] is None
    assert rec3["measured_latency"] == 2.500
    assert rec3["computed_latency"] is None


def test_session_metrics_collector_merge_last_wins(tmp_path: Path) -> None:
    """Verifies that subsequent non-empty user metrics overwrite earlier ones,
    but None/'' does not overwrite existing values.
    """
    collector = SessionMetricsCollector(
        room_name="test-room-merge",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg_1 = llm.ChatMessage(role="user", content=["Перша фраза"])
    user_msg_1.metrics = {
        "stopped_speaking_at": 100.0,
        "end_of_turn_delay": 0.500,
        "speech_start_time": 95.0,
    }
    collector.process_chat_message(user_msg_1)

    user_msg_2 = llm.ChatMessage(role="user", content=["Друга фраза"])
    user_msg_2.metrics = {
        "stopped_speaking_at": 103.0,
        "end_of_turn_delay": 0.600,
        "speech_start_time": None,  # None must not overwrite 95.0
    }
    collector.process_chat_message(user_msg_2)

    assert collector._last_user_metrics is not None
    assert collector._last_user_metrics["stopped_speaking_at"] == 103.0
    assert collector._last_user_metrics["end_of_turn_delay"] == 0.600
    assert collector._last_user_metrics["speech_start_time"] == 95.0


def test_session_metrics_collector_grace_continuation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies grace continuation latency anchor correction:
    user_msg_1 stopped 100.0, user_msg_2 stopped 103.0, assistant started 104.2,
    SDK e2e 4.2 -> e2e 1.2, source 'measured', sdk_e2e 4.2.
    """
    import logging

    collector = SessionMetricsCollector(
        room_name="test-room-grace",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg_1 = llm.ChatMessage(role="user", content=["Алло, я тут"])
    user_msg_1.metrics = {
        "stopped_speaking_at": 100.0,
        "end_of_turn_delay": 0.500,
        "transcription_delay": 0.300,
    }
    collector.process_chat_message(user_msg_1)

    user_msg_2 = llm.ChatMessage(role="user", content=["Продовжую говорити"])
    user_msg_2.metrics = {
        "stopped_speaking_at": 103.0,
    }
    collector.process_chat_message(user_msg_2)

    asst_msg = llm.ChatMessage(role="assistant", content=["Слухаю вас."])
    asst_msg.metrics = {
        "started_speaking_at": 104.2,
        "e2e_latency": 4.2,
        "llm_node_ttft": 0.400,
        "tts_node_ttfb": 0.200,
    }

    with caplog.at_level(logging.INFO, logger="agent.metrics"):
        metrics = collector.process_chat_message(asst_msg)

    assert metrics is not None
    assert metrics["e2e"] == 1.2
    assert metrics["e2e_source"] == "measured"
    assert metrics["sdk_e2e"] == 4.2
    assert metrics["scripted"] is False
    assert metrics["llm_fallback"] is False

    # Check INFO log message
    assert "e2e anchor corrected: sdk=4.200 measured=1.200 (turn 1)" in caplog.text


def test_session_metrics_collector_fallback_to_sdk_when_stopped_speaking_missing(
    tmp_path: Path,
) -> None:
    """Verifies fallback to SDK e2e_latency when user stopped_speaking_at is missing."""
    collector = SessionMetricsCollector(
        room_name="test-room-sdk-fallback",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg = llm.ChatMessage(role="user", content=["Привіт"])
    user_msg.metrics = {
        "end_of_turn_delay": 0.500,
        "transcription_delay": 0.300,
    }
    collector.process_chat_message(user_msg)

    asst_msg = llm.ChatMessage(role="assistant", content=["Вітаю!"])
    asst_msg.metrics = {
        "started_speaking_at": 104.2,
        "e2e_latency": 1.5,
        "llm_node_ttft": 0.400,
        "tts_node_ttfb": 0.200,
    }
    metrics = collector.process_chat_message(asst_msg)

    assert metrics is not None
    assert metrics["e2e"] == 1.5
    assert metrics["e2e_source"] == "sdk"
    assert metrics["sdk_e2e"] == 1.5


def test_split_user_turn_missing_anchor_computed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that split user turn whose last segment has no stop time ignores
    stale anchor and SDK value, falling back to computed latency with INFO log:
    user1 stopped 100.0 (eou 1.0, transcription 0.5), user2 without stop time,
    assistant started 104.5, llm_ttft 0.66, tts_ttfb 0.13, no SDK e2e -> e2e 1.79.
    """
    import logging

    collector = SessionMetricsCollector(
        room_name="test-room-split-no-stop",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    user_msg_1 = llm.ChatMessage(role="user", content=["Перший сегмент"])
    user_msg_1.metrics = {
        "stopped_speaking_at": 100.0,
        "end_of_turn_delay": 1.0,
        "transcription_delay": 0.5,
    }
    collector.process_chat_message(user_msg_1)

    user_msg_2 = llm.ChatMessage(role="user", content=["Другий сегмент без стопу"])
    user_msg_2.metrics = {
        "speech_start_time": 101.0,
    }
    collector.process_chat_message(user_msg_2)

    asst_msg = llm.ChatMessage(role="assistant", content=["Відповідь асистента"])
    asst_msg.metrics = {
        "started_speaking_at": 104.5,
        "llm_node_ttft": 0.66,
        "tts_node_ttfb": 0.13,
        "e2e_latency": 4.5,
    }

    with caplog.at_level(logging.INFO, logger="agent.metrics"):
        metrics = collector.process_chat_message(asst_msg)

    assert metrics is not None
    assert metrics["e2e"] == 1.79
    assert metrics["e2e_source"] == "computed"
    assert metrics["sdk_e2e"] == 4.5
    assert metrics["scripted"] is False
    assert (
        "e2e anchor unavailable: last user segment has no stop time "
        "(turn 1), using computed" in caplog.text
    )

    # Also verify scripted turn with missing anchor produces e2e null
    user_msg_3 = llm.ChatMessage(role="user", content=["Сегмент зі стопом"])
    user_msg_3.metrics = {"stopped_speaking_at": 200.0}
    collector.process_chat_message(user_msg_3)

    user_msg_4 = llm.ChatMessage(role="user", content=["Сегмент без стопу"])
    user_msg_4.metrics = {}
    collector.process_chat_message(user_msg_4)

    scripted_msg = llm.ChatMessage(role="assistant", content=["Скриптове прощання"])
    scripted_msg.metrics = {
        "started_speaking_at": 205.0,
        "tts_node_ttfb": 0.12,
    }
    m_scripted = collector.process_chat_message(scripted_msg)
    assert m_scripted is not None
    assert m_scripted["scripted"] is True
    assert m_scripted["e2e"] is None
    assert m_scripted["e2e_source"] == "scripted"


def test_compute_call_latency_summary_scripted_excluded() -> None:
    """Verifies that compute_call_latency_summary excludes scripted turns from p50/max,
    counts llm and scripted turns, and collects fallback turns.
    """
    turns = [
        {"role": "agent", "turn_index": 1, "metrics": {"scripted": True, "e2e": None}},
        {"role": "user", "turn_index": 2, "metrics": None},
        {
            "role": "agent",
            "turn_index": 3,
            "metrics": {"scripted": False, "e2e": 1.2, "llm_fallback": False},
        },
        {"role": "user", "turn_index": 4, "metrics": None},
        {
            "role": "agent",
            "turn_index": 5,
            "metrics": {"scripted": False, "e2e": 1.8, "llm_fallback": True},
        },
        {"role": "agent", "turn_index": 6, "metrics": {"scripted": True, "e2e": 0.2}},
    ]
    summary = compute_call_latency_summary(turns)
    assert summary["p50"] == 1.5
    assert summary["max"] == 1.8
    assert summary["llm_turns"] == 2
    assert summary["scripted_turns"] == 2
    assert summary["fallback_turns"] == [5]


def test_compute_call_latency_summary_old_calls_without_metrics() -> None:
    """Verifies that calls without metrics return clean null/zero values."""
    turns = [
        {"role": "agent", "turn_index": 1, "metrics": None},
        {"role": "user", "turn_index": 2, "metrics": None},
        {"role": "agent", "turn_index": 3, "metrics": None},
    ]
    summary = compute_call_latency_summary(turns)
    assert summary == {
        "p50": None,
        "max": None,
        "llm_turns": 0,
        "scripted_turns": 0,
        "fallback_turns": [],
    }


def test_assistant_turn_with_and_without_llm_ttft(tmp_path: Path) -> None:
    """Verifies that assistant messages with llm_node_ttft are treated as LLM turns
    (even with no prior user metrics: scripted=False, e2e=None, e2e_source='missing'),
    and only messages without llm_node_ttft are scripted.
    """
    collector = SessionMetricsCollector(
        room_name="test-room-llm-without-user",
        models=_create_test_models(),
        log_dir=tmp_path,
    )

    # 1. Assistant message with no prior user metrics but WITH llm_node_ttft -> LLM turn
    llm_greeting = llm.ChatMessage(role="assistant", content=["Доброго дня!"])
    llm_greeting.metrics = {
        "llm_node_ttft": 0.350,
        "tts_node_ttfb": 0.220,
        "e2e_latency": 1.200,
        "llm_metadata": {"model_name": "gemini-3.5-flash-lite"},
    }
    m1 = collector.process_chat_message(llm_greeting)
    assert m1 is not None
    assert m1["scripted"] is False
    assert m1["e2e"] is None
    assert m1["e2e_source"] == "missing"
    assert m1["sdk_e2e"] == 1.2
    assert m1["llm_ttft"] == 0.350
    assert m1["tts_ttfb"] == 0.220
    assert m1["llm_model"] == "gemini-3.5-flash-lite"
    assert m1["llm_fallback"] is False

    # 2. Assistant message with no prior user metrics and no llm_node_ttft -> scripted
    collector2 = SessionMetricsCollector(
        room_name="test-room-scripted-no-user",
        models=_create_test_models(),
        log_dir=tmp_path,
    )
    scripted_greeting = llm.ChatMessage(
        role="assistant", content=["Вітаємо у компанії!"]
    )
    scripted_greeting.metrics = {
        "tts_node_ttfb": 0.180,
    }
    m2 = collector2.process_chat_message(scripted_greeting)
    assert m2 is not None
    assert m2["scripted"] is True
    assert m2["e2e"] is None
    assert m2["e2e_source"] == "scripted"
    assert m2["llm_ttft"] is None
    assert m2["llm_model"] is None

    # 3. Assistant message with prior user metrics and no llm_node_ttft -> scripted
    user_msg = llm.ChatMessage(role="user", content=["Дякую, до побачення"])
    user_msg.metrics = {"stopped_speaking_at": 50.0}
    collector2.process_chat_message(user_msg)

    scripted_bye = llm.ChatMessage(role="assistant", content=["Гарного дня!"])
    scripted_bye.metrics = {
        "started_speaking_at": 50.8,
        "tts_node_ttfb": 0.150,
    }
    m3 = collector2.process_chat_message(scripted_bye)
    assert m3 is not None
    assert m3["scripted"] is True
    assert m3["e2e"] == 0.8
    assert m3["e2e_source"] == "scripted"
    assert m3["llm_ttft"] is None
