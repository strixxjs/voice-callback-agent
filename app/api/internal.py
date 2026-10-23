"""Internal API endpoints for worker reporting."""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models import Call
from app.services import calls as call_service

router = APIRouter(prefix="/internal/calls", tags=["internal"])
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def _get_call_or_404(db: AsyncSession, call_id: uuid.UUID) -> Call:
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found",
        )
    return call


async def verify_internal_secret(
    x_internal_secret: str | None = Header(None, alias="X-Internal-Secret"),
) -> None:
    """Validates X-Internal-Secret header using constant-time comparison."""
    expected_secret = settings.internal_api_secret
    if not expected_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API secret is not configured on server",
        )
    if not x_internal_secret or not secrets.compare_digest(
        x_internal_secret, expected_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API secret",
        )


class CallEventRequest(BaseModel):
    event_type: str
    message: str
    payload: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CallTurnRequest(BaseModel):
    turn_index: int
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CallEndedRequest(BaseModel):
    status: str
    ended_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: int | None = None
    latency_summary: dict[str, Any] | None = None


class CallExtractionRequest(BaseModel):
    extracted_answers: dict[str, Any] | None = None
    extracted_missing: list[str] | None = None
    extraction_status: str | None = None


@router.post(
    "/{call_id}/event",
    dependencies=[Depends(verify_internal_secret)],
)
async def record_call_event(
    call_id: uuid.UUID,
    request: CallEventRequest,
    db: DbSession,
) -> dict[str, str]:
    """Records an operational lifecycle event for a call."""
    call = await _get_call_or_404(db, call_id)

    if request.event_type == "participant_joined" and (
        call.status == call_service.CALL_STATUS_DIALING or call.started_at is None
    ):
        new_status = (
            call_service.CALL_STATUS_IN_PROGRESS
            if call.status == call_service.CALL_STATUS_DIALING
            else call.status
        )
        await call_service.update_call_status(
            db,
            call,
            new_status,
            started_at=call.started_at or request.timestamp,
        )

    await call_service.append_call_event(
        db,
        call,
        request.event_type,
        request.message,
        payload=request.payload,
        timestamp=request.timestamp,
    )
    return {"status": "ok"}


@router.post(
    "/{call_id}/turn",
    dependencies=[Depends(verify_internal_secret)],
)
async def record_call_turn(
    call_id: uuid.UUID,
    request: CallTurnRequest,
    db: DbSession,
) -> dict[str, str]:
    """Records a conversational turn with idempotent deduplication."""
    call = await _get_call_or_404(db, call_id)

    # Idempotent deduplication on (call_id, turn_index)
    await call_service.record_call_turn(
        db,
        call,
        turn_index=request.turn_index,
        role=request.role,
        content=request.content,
        timestamp=request.timestamp,
    )
    return {"status": "ok"}


@router.post(
    "/{call_id}/ended",
    dependencies=[Depends(verify_internal_secret)],
)
async def record_call_ended(
    call_id: uuid.UUID,
    request: CallEndedRequest,
    db: DbSession,
) -> dict[str, str]:
    """Records session termination status, duration, and latency before extraction."""
    call = await _get_call_or_404(db, call_id)

    await call_service.update_call_status(
        db,
        call,
        request.status,
        ended_at=request.ended_at,
        duration_seconds=request.duration_seconds,
        latency_summary=request.latency_summary,
    )
    return {"status": "ok"}


@router.post(
    "/{call_id}/extraction",
    dependencies=[Depends(verify_internal_secret)],
)
async def record_call_extraction(
    call_id: uuid.UUID,
    request: CallExtractionRequest,
    db: DbSession,
) -> dict[str, str]:
    """Records post-call LLM extraction answers, missing fields, and status."""
    call = await _get_call_or_404(db, call_id)

    await call_service.set_call_extraction(
        db,
        call,
        answers=request.extracted_answers,
        missing=request.extracted_missing,
        status=request.extraction_status,
    )
    return {"status": "ok"}
