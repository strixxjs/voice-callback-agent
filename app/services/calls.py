"""Call lifecycle service: the single place where call state is mutated.

Every call event, status change, transcript turn and extraction result goes
through this module (used by POST /leads and by the internal worker API).
Each function commits its own unit of work and then calls ``_publish`` so a
live broadcaster (M4 PR 3, SSE) can hook in here without touching callers.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.telephony import mask_phone_number
from app.live import broadcaster
from app.models import Call, CallEvent, CallTurn, Lead

logger = logging.getLogger("app.services.calls")

# Call statuses already used by the worker / model (PR 1). No new values here.
CALL_STATUS_DIALING = "dialing"
CALL_STATUS_IN_PROGRESS = "in_progress"
CALL_STATUS_FAILED = "failed"


def room_name_for_call(call_id: uuid.UUID) -> str:
    """Derives the LiveKit room name from the call id."""
    return f"call-{call_id}"


def compute_call_latency_summary(turns: list[Any]) -> dict[str, Any]:
    """Pure helper function computing latency statistics from call turns.

    Calculates:
      - p50 and max over non-scripted agent turns with non-null e2e (in seconds).
      - count of LLM turns and scripted turns.
      - list of turn indexes where llm_fallback is true.
    """
    valid_e2es: list[float] = []
    llm_turns = 0
    scripted_turns = 0
    fallback_turns: list[int] = []

    for t in turns:
        role = getattr(t, "role", None) if not isinstance(t, dict) else t.get("role")
        if str(role).lower() not in ("agent", "assistant"):
            continue

        metrics = (
            getattr(t, "metrics", None) if not isinstance(t, dict) else t.get("metrics")
        )
        turn_idx = (
            getattr(t, "turn_index", None)
            if not isinstance(t, dict)
            else t.get("turn_index")
        )

        if not isinstance(metrics, dict):
            continue

        is_scripted = bool(metrics.get("scripted", False))
        if is_scripted:
            scripted_turns += 1
        else:
            llm_turns += 1
            e2e = metrics.get("e2e")
            if e2e is not None and isinstance(e2e, (int, float)):
                valid_e2es.append(float(e2e))

        if metrics.get("llm_fallback") is True and turn_idx is not None:
            fallback_turns.append(int(turn_idx))

    p50: float | None = None
    max_val: float | None = None
    if valid_e2es:
        p50 = round(float(statistics.median(valid_e2es)), 3)
        max_val = round(float(max(valid_e2es)), 3)

    return {
        "p50": p50,
        "max": max_val,
        "llm_turns": llm_turns,
        "scripted_turns": scripted_turns,
        "fallback_turns": fallback_turns,
    }


def _publish(kind: str, call: Call, data: dict[str, Any]) -> None:
    """Post-commit hook for live updates fanned out via SSE broadcaster."""
    try:
        event = {
            "kind": kind,
            "call_id": str(call.id),
            "status": call.status,
            "data": data,
            "ts": datetime.now(UTC).isoformat(),
        }
        broadcaster.publish(event)
    except Exception as exc:
        logger.warning(
            "Failed to publish live event kind=%s call_id=%s: %s",
            kind,
            call.id,
            exc,
        )


async def create_lead_and_call(
    db: AsyncSession,
    *,
    name: str,
    phone: str,
    mode: str,
) -> tuple[Lead, Call]:
    """Persists a lead and its first call in the initial (model default) status."""
    now = datetime.now(UTC)
    lead = Lead(id=uuid.uuid4(), name=name, phone=phone, created_at=now)
    call_id = uuid.uuid4()
    call = Call(
        id=call_id,
        lead_id=lead.id,
        room_name=room_name_for_call(call_id),
        mode=mode,
        status=CALL_STATUS_DIALING,
        created_at=now,
        updated_at=now,
    )
    db.add_all([lead, call])
    await db.commit()
    logger.info(
        "Lead created lead_id=%s call_id=%s status=%s mode=%s phone=%s",
        lead.id,
        call.id,
        call.status,
        mode,
        mask_phone_number(phone),
    )
    _publish("call_created", call, {"status": call.status})
    return lead, call


async def append_call_event(
    db: AsyncSession,
    call: Call,
    event_type: str,
    message: str,
    payload: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> CallEvent:
    """Appends an operational event to a call and commits it."""
    event = CallEvent(
        call_id=call.id,
        event_type=event_type,
        message=message,
        payload=payload,
        timestamp=timestamp or datetime.now(UTC),
    )
    db.add(event)
    await db.commit()
    logger.info(
        "Call event lead_id=%s call_id=%s type=%s status=%s",
        call.lead_id,
        call.id,
        event_type,
        call.status,
    )
    _publish("event", call, {"event_id": event.id, "event_type": event_type})
    return event


async def update_call_status(
    db: AsyncSession,
    call: Call,
    status: str,
    *,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    duration_seconds: int | None = None,
    latency_summary: dict[str, Any] | None = None,
    call_config: dict[str, Any] | None = None,
) -> Call:
    """Changes call status (and optional lifecycle fields) and commits it."""
    previous = call.status
    call.status = status
    if started_at is not None:
        call.started_at = started_at
    if ended_at is not None:
        call.ended_at = ended_at
    if duration_seconds is not None:
        call.duration_seconds = duration_seconds
    if latency_summary is not None:
        call.latency_summary = latency_summary
    if call_config is not None:
        call.call_config = call_config
    await db.commit()
    logger.info(
        "Call status lead_id=%s call_id=%s %s -> %s",
        call.lead_id,
        call.id,
        previous,
        status,
    )
    _publish("status", call, {"previous": previous, "status": status})
    return call


async def set_call_config(
    db: AsyncSession,
    call: Call,
    call_config: dict[str, Any],
) -> Call:
    """Sets call configuration directly and commits without emitting status events."""
    call.call_config = call_config
    await db.commit()
    return call


async def record_call_turn(
    db: AsyncSession,
    call: Call,
    *,
    turn_index: int,
    role: str,
    content: str,
    timestamp: datetime,
    metrics: dict[str, Any] | None = None,
) -> CallTurn | None:
    """Stores a transcript turn; idempotent on (call_id, turn_index).

    Returns None when the turn already exists.
    """
    existing = await db.scalar(
        select(CallTurn).where(
            CallTurn.call_id == call.id,
            CallTurn.turn_index == turn_index,
        )
    )
    if existing is not None:
        return None

    turn = CallTurn(
        call_id=call.id,
        turn_index=turn_index,
        role=role,
        content=content,
        timestamp=timestamp,
        metrics=metrics,
    )
    db.add(turn)
    await db.commit()
    _publish("turn", call, {"turn_index": turn_index, "role": role})
    return turn


async def set_call_extraction(
    db: AsyncSession,
    call: Call,
    *,
    answers: dict[str, Any] | None,
    missing: list[str] | None,
    status: str | None,
) -> Call:
    """Stores post-call extraction results and commits them."""
    call.extracted_answers = answers
    call.extracted_missing = missing
    call.extraction_status = status
    await db.commit()
    logger.info(
        "Call extraction lead_id=%s call_id=%s extraction_status=%s",
        call.lead_id,
        call.id,
        status,
    )
    _publish("extraction", call, {"extraction_status": status})
    return call
