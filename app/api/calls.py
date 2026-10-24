"""Read-only demo API: call list, call detail and project overview (no auth)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.telephony import mask_phone_number
from app.config import Settings, settings
from app.db import get_db
from app.models import Call, CallEvent, CallTurn, Lead
from app.schemas import (
    CallDetail,
    CallEventOut,
    CallSummary,
    ComponentOut,
    ComponentStatus,
    ExtractionOut,
    OverviewResponse,
    StageOut,
    TranscriptTurn,
)

router = APIRouter(prefix="/api", tags=["demo"])
DbSession = Annotated[AsyncSession, Depends(get_db)]

# Single source of truth for the milestone board shown by the demo console.
PROJECT_STAGES: tuple[StageOut, ...] = (
    StageOut(id="M0", title="Налаштування", status="done"),
    StageOut(id="M1", title="Агент у браузері", status="done"),
    StageOut(id="M2", title="Код телефонного дзвінка", status="done"),
    StageOut(id="M3", title="Якість розмови", status="done"),
    StageOut(id="M4", title="Бекенд і консоль", status="done"),
    StageOut(id="M5", title="Резюме і доставка", status="planned"),
    StageOut(id="M6", title="Готовність до демо", status="planned"),
    StageOut(id="phone_go_live", title="Живий телефонний дзвінок", status="deferred"),
)

CALL_ENDED_EVENT = "call_ended"


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat them as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _duration_seconds(call: Call) -> int | None:
    if call.ended_at is None:
        return None
    if call.duration_seconds is not None:
        return call.duration_seconds
    started, ended = _as_utc(call.started_at), _as_utc(call.ended_at)
    if started is None or ended is None:
        return None
    return max(0, int((ended - started).total_seconds()))


def _ended_by(payload: dict[str, Any] | None) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("ended_by"), str):
        return payload["ended_by"]
    return None


async def _ended_by_map(
    db: AsyncSession, call_ids: list[uuid.UUID]
) -> dict[uuid.UUID, str | None]:
    """Latest call_ended event's ended_by per call (one query)."""
    if not call_ids:
        return {}
    rows = await db.execute(
        select(CallEvent.call_id, CallEvent.payload)
        .where(
            CallEvent.call_id.in_(call_ids),
            CallEvent.event_type == CALL_ENDED_EVENT,
        )
        .order_by(CallEvent.timestamp, CallEvent.id)
    )
    result: dict[uuid.UUID, str | None] = {}
    for call_id, payload in rows:
        value = _ended_by(payload)
        if value is not None:
            result[call_id] = value
    return result


def _summary_fields(
    call: Call, lead: Lead | None, ended_by: str | None
) -> dict[str, Any]:
    return {
        "call_id": call.id,
        "lead_name": lead.name if lead else None,
        "phone_masked": mask_phone_number(lead.phone) if lead else None,
        "mode": call.mode,
        "status": call.status,
        "room": call.room_name,
        "created_at": _as_utc(call.created_at),
        "started_at": _as_utc(call.started_at),
        "ended_at": _as_utc(call.ended_at),
        "duration_seconds": _duration_seconds(call),
        "ended_by": ended_by,
    }


@router.get("/calls", response_model=list[CallSummary])
async def list_calls(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[CallSummary]:
    """Lists calls, newest first."""
    rows = (
        await db.execute(
            select(Call, Lead)
            .outerjoin(Lead, Call.lead_id == Lead.id)
            .order_by(Call.created_at.desc(), Call.id.desc())
            .limit(limit)
        )
    ).all()
    ended_by = await _ended_by_map(db, [call.id for call, _ in rows])
    return [
        CallSummary(**_summary_fields(call, lead, ended_by.get(call.id)))
        for call, lead in rows
    ]


@router.get("/calls/{call_id}", response_model=CallDetail)
async def get_call(call_id: uuid.UUID, db: DbSession) -> CallDetail:
    """Call header + transcript + events + extraction."""
    row = (
        await db.execute(
            select(Call, Lead)
            .outerjoin(Lead, Call.lead_id == Lead.id)
            .where(Call.id == call_id)
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Call not found"
        )
    call, lead = row

    turns = (
        await db.scalars(
            select(CallTurn)
            .where(CallTurn.call_id == call_id)
            .order_by(CallTurn.turn_index)
        )
    ).all()
    events = (
        await db.scalars(
            select(CallEvent)
            .where(CallEvent.call_id == call_id)
            .order_by(CallEvent.timestamp, CallEvent.id)
        )
    ).all()

    ended_by: str | None = None
    for event in events:
        if event.event_type == CALL_ENDED_EVENT:
            ended_by = _ended_by(event.payload) or ended_by

    has_extraction = (
        call.extraction_status is not None
        or call.extracted_answers is not None
        or call.extracted_missing is not None
    )
    extraction = (
        ExtractionOut(
            answers=call.extracted_answers,
            missing=call.extracted_missing,
            status=call.extraction_status,
        )
        if has_extraction
        else None
    )

    return CallDetail(
        **_summary_fields(call, lead, ended_by),
        transcript=[
            TranscriptTurn(
                turn_index=t.turn_index,
                role=t.role,
                text=t.content,
                timestamp=_as_utc(t.timestamp),
            )
            for t in turns
        ],
        events=[
            CallEventOut(
                id=e.id,
                event_type=e.event_type,
                message=e.message,
                payload=e.payload,
                timestamp=_as_utc(e.timestamp),
            )
            for e in events
        ],
        extraction=extraction,
    )


def _configured(*values: str | None) -> ComponentStatus:
    return "configured" if all(values) else "not_configured"


def _provider_key(cfg: Settings, provider: str) -> str | None:
    """API key used by a given provider name, if known."""
    keys: dict[str, str | None] = {
        "google": cfg.google_api_key,
        "openai": cfg.openai_api_key,
        "deepgram": cfg.deepgram_api_key,
        "elevenlabs": cfg.elevenlabs_api_key,
    }
    return keys.get(provider.lower())


def build_components(cfg: Settings) -> list[ComponentOut]:
    """Component statuses from config presence only (no network calls)."""
    return [
        ComponentOut(
            name="livekit",
            provider="livekit",
            status=_configured(
                cfg.livekit_url, cfg.livekit_api_key, cfg.livekit_api_secret
            ),
        ),
        ComponentOut(
            name="llm",
            provider=cfg.llm_provider,
            status=_configured(_provider_key(cfg, cfg.llm_provider)),
        ),
        ComponentOut(
            name="stt",
            provider=cfg.stt_provider,
            status=_configured(_provider_key(cfg, cfg.stt_provider)),
        ),
        ComponentOut(
            name="tts",
            provider=cfg.tts_provider,
            status=_configured(_provider_key(cfg, cfg.tts_provider)),
        ),
        ComponentOut(name="telephony", provider=None, status="not_connected"),
    ]


@router.get("/overview", response_model=OverviewResponse)
async def overview() -> OverviewResponse:
    """Static milestone board + component configuration status."""
    return OverviewResponse(
        stages=list(PROJECT_STAGES),
        components=build_components(settings),
    )
