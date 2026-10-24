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
        float(x) for x in latencies if x is not None and isinstance(x, (int, float))
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
        self.measured_latencies: list[float] = []
        self.computed_latencies: list[float] = []
        self._last_user_metrics: dict[str, Any] | None = None
        self._user_anchor: float | None = None
        self._any_user_had_stop_time: bool = False
        self._closed = False

    def attach_to_session(self, session: AgentSession) -> None:
        """Stores session reference; metrics are processed via worker reporting."""
        self._session = session

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        """Processes conversation items, pairing user metrics with assistant replies."""
        try:
            item = getattr(event, "item", None)
            self.process_chat_message(item)
        except Exception as e:
            logger.warning("Metrics collector error handling conversation item: %s", e)

    def process_chat_message(self, item: Any) -> dict[str, Any] | None:
        """Processes a ChatMessage; returns turn metrics for assistant turns."""
        try:
            if not isinstance(item, llm.ChatMessage):
                return None

            role_val = str(
                item.role.value if hasattr(item.role, "value") else item.role
            ).lower()

            if role_val == "user":
                raw_metrics = getattr(item, "metrics", {}) or {}
                if self._last_user_metrics is None:
                    self._last_user_metrics = {}
                for k, v in raw_metrics.items():
                    if v is not None and v != "":
                        self._last_user_metrics[k] = v

                # Track anchor separately from merged metrics:
                # anchor = numeric stopped_speaking_at of the LAST user message
                # since previous assistant message, else None.
                # Also remember whether ANY user message in that span had a stop time.
                val = raw_metrics.get("stopped_speaking_at")
                if val is not None and isinstance(val, (int, float)):
                    self._user_anchor = float(val)
                    self._any_user_had_stop_time = True
                else:
                    self._user_anchor = None
                return None

            if role_val in ("assistant", "agent"):
                user_metrics = self._last_user_metrics
                anchor = self._user_anchor
                any_user_had_stop_time = self._any_user_had_stop_time

                self._last_user_metrics = None
                self._user_anchor = None
                self._any_user_had_stop_time = False

                raw_assistant_metrics = getattr(item, "metrics", {}) or {}
                return self._record_turn(
                    user_metrics,
                    dict(raw_assistant_metrics),
                    anchor=anchor,
                    any_user_had_stop_time=any_user_had_stop_time,
                )

            return None
        except Exception as e:
            logger.warning("Metrics collector error processing chat message: %s", e)
            return None

    def _record_turn(
        self,
        user_metrics: dict[str, Any] | None,
        assistant_metrics: dict[str, Any],
        *,
        anchor: float | None = None,
        any_user_had_stop_time: bool = False,
    ) -> dict[str, Any]:
        """Extracts latency fields, logs summary, writes JSONL, returns metrics."""
        try:
            # Fallback if _record_turn is invoked directly without anchor kwargs
            if anchor is None and not any_user_had_stop_time and user_metrics:
                raw_stop = user_metrics.get("stopped_speaking_at")
                if raw_stop is not None and isinstance(raw_stop, (int, float)):
                    anchor = float(raw_stop)
                    any_user_had_stop_time = True

            llm_ttft = assistant_metrics.get("llm_node_ttft")
            has_llm_ttft = llm_ttft is not None and isinstance(llm_ttft, (int, float))

            if user_metrics is None and not has_llm_ttft:
                # Scripted turn without prior user speech (e.g. greeting)
                tts_ttfb = assistant_metrics.get("tts_node_ttfb")
                return {
                    "e2e": None,
                    "e2e_source": "scripted",
                    "sdk_e2e": None,
                    "scripted": True,
                    "endpointing_delay": None,
                    "transcript_delay": None,
                    "llm_ttft": None,
                    "tts_ttfb": (
                        round(float(tts_ttfb), 3)
                        if tts_ttfb is not None and isinstance(tts_ttfb, (int, float))
                        else None
                    ),
                    "llm_model": None,
                    "llm_fallback": False,
                }

            has_prior_user = user_metrics is not None
            if user_metrics is None:
                user_metrics = {}

            self.turn_index += 1

            eou = user_metrics.get("end_of_turn_delay")
            stt_dur = user_metrics.get("transcription_delay")
            tts_ttfb = assistant_metrics.get("tts_node_ttfb")
            raw_sdk_e2e = assistant_metrics.get("e2e_latency")
            started_speaking = assistant_metrics.get("started_speaking_at")

            # Measured end-to-end diff: assistant started - last user stopped
            diff_measured: float | None = None
            if (
                started_speaking is not None
                and isinstance(started_speaking, (int, float))
                and anchor is not None
                and isinstance(anchor, (int, float))
            ):
                diff_measured = float(started_speaking) - float(anchor)

            sdk_e2e_val: float | None = None
            if raw_sdk_e2e is not None and isinstance(raw_sdk_e2e, (int, float)):
                sdk_e2e_val = float(raw_sdk_e2e)

            # Check for scripted assistant turn (no LLM TTFT, e.g. session.say)
            is_scripted = not has_llm_ttft

            # Log correction when both exist and differ by > 0.3s
            if (
                sdk_e2e_val is not None
                and diff_measured is not None
                and abs(sdk_e2e_val - diff_measured) > 0.3
            ):
                logger.info(
                    "e2e anchor corrected: sdk=%.3f measured=%.3f (turn %d)",
                    sdk_e2e_val,
                    diff_measured,
                    self.turn_index,
                )

            computed_latency: float | None = None
            if (
                eou is not None
                and isinstance(eou, (int, float))
                and llm_ttft is not None
                and isinstance(llm_ttft, (int, float))
                and tts_ttfb is not None
                and isinstance(tts_ttfb, (int, float))
            ):
                computed_latency = float(eou) + float(llm_ttft) + float(tts_ttfb)

            total_latency: float | None = None

            if is_scripted:
                # Scripted turns: same anchor rule; if anchor is missing -> e2e null
                e2e_source = "scripted"
                total_latency = diff_measured
            elif not has_prior_user:
                # LLM turn with no preceding user metrics (e.g. LLM greeting)
                total_latency = None
                e2e_source = "missing"
                self.missing_latency_turns += 1
            elif anchor is not None:
                # Anchor present: started_speaking_at - anchor, source "measured"
                if diff_measured is not None:
                    total_latency = diff_measured
                    e2e_source = "measured"
                elif sdk_e2e_val is not None:
                    total_latency = sdk_e2e_val
                    e2e_source = "sdk"
                elif computed_latency is not None:
                    total_latency = computed_latency
                    e2e_source = "computed"
                else:
                    total_latency = None
                    e2e_source = "missing"
                    self.missing_latency_turns += 1
            elif any_user_had_stop_time:
                # Anchor missing but earlier segment in span had stop time (split turn):
                # computed if available, else missing. Do NOT use SDK value here.
                if computed_latency is not None:
                    total_latency = computed_latency
                    e2e_source = "computed"
                    logger.info(
                        "e2e anchor unavailable: last user segment has no stop time "
                        "(turn %d), using computed",
                        self.turn_index,
                    )
                else:
                    total_latency = None
                    e2e_source = "missing"
                    self.missing_latency_turns += 1
            else:
                # No stop time in span: current behavior (SDK, then computed, missing)
                if sdk_e2e_val is not None:
                    total_latency = sdk_e2e_val
                    e2e_source = "sdk"
                elif computed_latency is not None:
                    total_latency = computed_latency
                    e2e_source = "computed"
                else:
                    total_latency = None
                    e2e_source = "missing"
                    self.missing_latency_turns += 1

            if not is_scripted:
                if total_latency is not None and e2e_source in ("measured", "sdk"):
                    self.measured_latencies.append(round(total_latency, 3))
                if computed_latency is not None:
                    self.computed_latencies.append(round(computed_latency, 3))
                if total_latency is not None:
                    self.turn_latencies.append(round(total_latency, 3))

            llm_model: str | None = None
            llm_fallback = False
            if not is_scripted:
                llm_meta = assistant_metrics.get("llm_metadata")
                if isinstance(llm_meta, dict):
                    llm_model = llm_meta.get("model_name")
                elif hasattr(llm_meta, "model_name"):
                    llm_model = llm_meta.model_name
                llm_fallback = bool(llm_model and llm_model != self.models.llm_model)

            meas_val = (
                diff_measured
                if diff_measured is not None
                else (sdk_e2e_val if e2e_source == "sdk" else None)
            )
            meas_str = f"{meas_val:.3f}s" if meas_val is not None else "N/A"
            comp_str = (
                f"{computed_latency:.3f}s" if computed_latency is not None else "N/A"
            )
            total_str = f"{total_latency:.3f}s" if total_latency is not None else "N/A"
            eou_str = f"{eou:.3f}s" if isinstance(eou, (int, float)) else "N/A"
            stt_str = f"{stt_dur:.3f}s" if isinstance(stt_dur, (int, float)) else "N/A"
            llm_str = (
                f"{llm_ttft:.3f}s" if isinstance(llm_ttft, (int, float)) else "N/A"
            )
            tts_str = (
                f"{tts_ttfb:.3f}s" if isinstance(tts_ttfb, (int, float)) else "N/A"
            )

            logger.info(
                "Turn %d latency: EOU=%s | STT=%s | LLM TTFT=%s | TTS TTFB=%s | "
                "Meas=%s | Comp=%s | Total=%s (e2e_source=%s)",
                self.turn_index,
                eou_str,
                stt_str,
                llm_str,
                tts_str,
                meas_str,
                comp_str,
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
                "measured_latency": (
                    round(meas_val, 3) if meas_val is not None else None
                ),
                "sdk_e2e_latency": (
                    round(sdk_e2e_val, 3) if sdk_e2e_val is not None else None
                ),
                "computed_latency": (
                    round(computed_latency, 3) if computed_latency is not None else None
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

            return {
                "e2e": round(total_latency, 3) if total_latency is not None else None,
                "e2e_source": e2e_source,
                "sdk_e2e": round(sdk_e2e_val, 3) if sdk_e2e_val is not None else None,
                "scripted": is_scripted,
                "endpointing_delay": (
                    round(float(eou), 3) if isinstance(eou, (int, float)) else None
                ),
                "transcript_delay": (
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
                "llm_model": llm_model,
                "llm_fallback": llm_fallback,
            }
        except Exception as e:
            logger.warning(
                "Error recording turn %d metrics: %s", self.turn_index, e, exc_info=True
            )
            return {
                "e2e": None,
                "e2e_source": "missing",
                "sdk_e2e": None,
                "scripted": False,
                "endpointing_delay": None,
                "transcript_delay": None,
                "llm_ttft": None,
                "tts_ttfb": None,
                "llm_model": None,
                "llm_fallback": False,
            }

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
            meas_sum = compute_latency_summary(self.measured_latencies)
            comp_sum = compute_latency_summary(self.computed_latencies)

            # Primary is measured; fallback to turn_latencies if no measured
            primary = (
                meas_sum
                if self.measured_latencies
                else compute_latency_summary(self.turn_latencies)
            )

            summary = {
                "turns": primary["turns"],
                "missing_latency_turns": self.missing_latency_turns,
                "avg": primary["avg"],
                "p50": primary["p50"],
                "max": primary["max"],
                "p50_measured": meas_sum["p50"],
                "max_measured": meas_sum["max"],
                "p50_computed": comp_sum["p50"],
                "max_computed": comp_sum["max"],
            }
            if self._closed:
                return summary
            self._closed = True

            models = self.models
            logger.info(
                "Session metrics summary: room=%s, turns=%d, "
                "missing_latency_turns=%d, avg=%.3fs, p50=%.3fs, "
                "max=%.3fs (measured p50=%.3fs max=%.3fs | "
                "computed p50=%.3fs max=%.3fs) | "
                "LLM=%s/%s, STT=%s/%s, TTS=%s/%s (voice=%s)",
                self.room_name,
                summary["turns"],
                self.missing_latency_turns,
                summary["avg"],
                summary["p50"],
                summary["max"],
                summary["p50_measured"],
                summary["max_measured"],
                summary["p50_computed"],
                summary["max_computed"],
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
                "p50_measured": 0.0,
                "max_measured": 0.0,
                "p50_computed": 0.0,
                "max_computed": 0.0,
            }
