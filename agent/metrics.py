"""Turn latency metrics collection and aggregation for LiveKit voice agents."""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from livekit.agents import llm
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import ConversationItemAddedEvent

logger = logging.getLogger("agent.metrics")


@dataclass(frozen=True)
class ActiveModels:
    """Resolved model and provider metadata active for the session."""

    llm_provider: str
    llm_model: str
    stt_provider: str
    stt_model: str
    tts_provider: str
    tts_model: str
    tts_voice_id: str | None


def compute_latency_summary(latencies: list[float]) -> dict[str, float]:
    """Pure aggregation function computing turns, avg, p50, and max response latency.

    Args:
        latencies: List of response latencies (in seconds). Non-numeric or None values
            are filtered out.

    Returns:
        Dictionary with keys:
            - turns: int (count of valid latency values)
            - avg: float (mean latency rounded to 3 decimal places)
            - p50: float (median latency rounded to 3 decimal places)
            - max: float (maximum latency rounded to 3 decimal places)
    """
    valid: list[float] = [
        float(x)
        for x in latencies
        if x is not None and isinstance(x, (int, float))
    ]
    if not valid:
        return {"turns": 0, "avg": 0.0, "p50": 0.0, "max": 0.0}

    turns = len(valid)
    avg = round(sum(valid) / turns, 3)
    p50 = round(float(statistics.median(valid)), 3)
    max_val = round(float(max(valid)), 3)
    return {"turns": turns, "avg": avg, "p50": p50, "max": max_val}


class SessionMetricsCollector:
    """Collects per-turn latency metrics from ChatMessage.metrics and logs summary.

    Primary metrics source: ChatMessage.metrics attached to conversation items:
      - User ChatMessage:
          - end_of_turn_delay: delay between VAD speech end and end-of-turn decision
          - transcription_delay: STT processing duration for user speech
      - Assistant ChatMessage:
          - llm_node_ttft: LLM time to first generated token
          - tts_node_ttfb: TTS time to first audio chunk byte
          - e2e_latency: user stops speaking -> agent audio playback starts

    Fallback total latency formula:
      If e2e_latency is missing and all three sub-latencies are present:
        total_response_latency = end_of_turn_delay + llm_node_ttft + tts_node_ttfb
        e2e_source = "computed"
      If any sub-latency is also missing:
        total_response_latency = None
        e2e_source = "missing"
        (recorded in JSONL as null, not added to latency averages).

    Never breaks a call: all errors are logged as warnings and swallowed.
    """

    def __init__(
        self,
        room_name: str,
        models: ActiveModels,
        log_dir: str | Path = "logs/metrics",
    ) -> None:
        self.room_name = room_name
        self.models = models
        safe_room = re.sub(r"[^\w\-.]", "_", room_name)
        self.log_file = Path(log_dir) / f"{safe_room}.jsonl"
        self.turn_index = 0
        self.missing_latency_turns = 0
        self.turn_latencies: list[float] = []
        self._last_user_metrics: dict[str, Any] | None = None
        self._closed = False

    def attach_to_session(self, session: AgentSession) -> None:
        """Subscribes to session conversation items without deprecated events."""
        try:
            session.on("conversation_item_added", self._on_conversation_item_added)
        except Exception as e:
            logger.warning("Failed to attach metrics listener to session: %s", e)

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        """Processes conversation items, pairing user metrics with assistant replies."""
        try:
            item = getattr(event, "item", None)
            if not isinstance(item, llm.ChatMessage):
                return

            if item.role == "user":
                raw_metrics = getattr(item, "metrics", {}) or {}
                if self._last_user_metrics is None:
                    self._last_user_metrics = {}
                for k, v in raw_metrics.items():
                    if v is not None and v != "":
                        current_val = self._last_user_metrics.get(k)
                        if current_val is None or current_val == "":
                            self._last_user_metrics[k] = v
            elif item.role == "assistant":
                user_metrics = self._last_user_metrics
                self._last_user_metrics = None

                # Skip assistant messages not answering user speech (e.g. greeting)
                if user_metrics is None:
                    return

                raw_assistant_metrics = getattr(item, "metrics", {}) or {}
                self._record_turn(user_metrics, dict(raw_assistant_metrics))
        except Exception as e:
            logger.warning("Metrics collector error handling conversation item: %s", e)

    def _record_turn(
        self, user_metrics: dict[str, Any], assistant_metrics: dict[str, Any]
    ) -> None:
        """Extracts latency fields, logs one summary line, and writes JSONL."""
        try:
            self.turn_index += 1

            eou = user_metrics.get("end_of_turn_delay")
            stt_dur = user_metrics.get("transcription_delay")
            llm_ttft = assistant_metrics.get("llm_node_ttft")
            tts_ttfb = assistant_metrics.get("tts_node_ttfb")
            raw_total = assistant_metrics.get("e2e_latency")

            total_latency: float | None = None
            if raw_total is not None and isinstance(raw_total, (int, float)):
                total_latency = float(raw_total)
                e2e_source = "sdk"
                self.turn_latencies.append(round(total_latency, 3))
            elif (
                eou is not None
                and isinstance(eou, (int, float))
                and llm_ttft is not None
                and isinstance(llm_ttft, (int, float))
                and tts_ttfb is not None
                and isinstance(tts_ttfb, (int, float))
            ):
                # Fallback: sum of all three pipeline delays
                total_latency = float(eou) + float(llm_ttft) + float(tts_ttfb)
                e2e_source = "computed"
                self.turn_latencies.append(round(total_latency, 3))
            else:
                total_latency = None
                e2e_source = "missing"
                self.missing_latency_turns += 1

            eou_str = f"{eou:.3f}s" if isinstance(eou, (int, float)) else "N/A"
            stt_str = f"{stt_dur:.3f}s" if isinstance(stt_dur, (int, float)) else "N/A"
            llm_str = (
                f"{llm_ttft:.3f}s" if isinstance(llm_ttft, (int, float)) else "N/A"
            )
            tts_str = (
                f"{tts_ttfb:.3f}s" if isinstance(tts_ttfb, (int, float)) else "N/A"
            )
            total_str = (
                f"{total_latency:.3f}s" if total_latency is not None else "N/A"
            )

            logger.info(
                "Turn %d latency: EOU=%s | STT=%s | LLM TTFT=%s | TTS TTFB=%s | "
                "Total=%s (e2e_source=%s)",
                self.turn_index,
                eou_str,
                stt_str,
                llm_str,
                tts_str,
                total_str,
                e2e_source,
            )

            models = self.models
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "room": self.room_name,
                "turn_index": self.turn_index,
                "end_of_utterance_delay": (
                    round(float(eou), 3) if isinstance(eou, (int, float)) else None
                ),
                "stt_duration": (
                    round(float(stt_dur), 3)
                    if isinstance(stt_dur, (int, float))
                    else None
                ),
                "llm_ttft": (
                    round(float(llm_ttft), 3)
                    if isinstance(llm_ttft, (int, float))
                    else None
                ),
                "tts_ttfb": (
                    round(float(tts_ttfb), 3)
                    if isinstance(tts_ttfb, (int, float))
                    else None
                ),
                "total_response_latency": (
                    round(total_latency, 3) if total_latency is not None else None
                ),
                "e2e_source": e2e_source,
                "llm_provider": models.llm_provider,
                "llm_model": models.llm_model,
                "stt_provider": models.stt_provider,
                "stt_model": models.stt_model,
                "tts_provider": models.tts_provider,
                "tts_model": models.tts_model,
                "tts_voice_id": models.tts_voice_id,
            }
            self._write_jsonl_record(record)
        except Exception as e:
            logger.warning(
                "Error recording turn %d metrics: %s", self.turn_index, e, exc_info=True
            )

    def _write_jsonl_record(self, record: dict[str, Any]) -> None:
        """Appends turn record to JSONL file safely."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Error appending metrics record to %s: %s", self.log_file, e)

    def on_session_close(self) -> dict[str, Any]:
        """Calculates final session latency summary and logs it. Idempotent."""
        try:
            if self._closed:
                res = compute_latency_summary(self.turn_latencies)
                res["missing_latency_turns"] = self.missing_latency_turns
                return res
            self._closed = True

            summary = compute_latency_summary(self.turn_latencies)
            summary["missing_latency_turns"] = self.missing_latency_turns
            models = self.models
            logger.info(
                "Session metrics summary: room=%s, turns=%d, "
                "missing_latency_turns=%d, avg=%.3fs, p50=%.3fs, "
                "max=%.3fs | LLM=%s/%s, STT=%s/%s, TTS=%s/%s (voice=%s)",
                self.room_name,
                summary["turns"],
                self.missing_latency_turns,
                summary["avg"],
                summary["p50"],
                summary["max"],
                models.llm_provider,
                models.llm_model,
                models.stt_provider,
                models.stt_model,
                models.tts_provider,
                models.tts_model,
                models.tts_voice_id,
            )
            return summary
        except Exception as e:
            logger.warning("Error finalizing session metrics summary: %s", e)
            return {
                "turns": 0,
                "missing_latency_turns": self.missing_latency_turns,
                "avg": 0.0,
                "p50": 0.0,
                "max": 0.0,
            }
