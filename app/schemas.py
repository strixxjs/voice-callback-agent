"""Pydantic request/response schemas for the public and demo API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

E164_PATTERN = r"^\+[1-9]\d{7,14}$"

LeadName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]
E164Phone = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=E164_PATTERN)
]
CallMode = Literal["browser", "phone"]


class LeadCreateRequest(BaseModel):
    name: LeadName
    phone: E164Phone
    mode: CallMode


class LeadCreateResponse(BaseModel):
    lead_id: uuid.UUID
    call_id: uuid.UUID
    room: str
    mode: CallMode
    status: str
    livekit_url: str
    token: str


class CallSummary(BaseModel):
    call_id: uuid.UUID
    lead_name: str | None
    phone_masked: str | None
    mode: str
    status: str
    room: str
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: int | None
    ended_by: str | None


class TranscriptTurn(BaseModel):
    turn_index: int
    role: str
    text: str
    timestamp: datetime
    metrics: dict[str, Any] | None = None


class CallEventOut(BaseModel):
    id: int
    event_type: str
    message: str
    payload: dict[str, Any] | None
    timestamp: datetime


class ExtractionOut(BaseModel):
    answers: dict[str, Any] | None
    missing: list[str] | None
    status: str | None


class LatencySummaryOut(BaseModel):
    p50: float | None = None
    max: float | None = None
    llm_turns: int = 0
    scripted_turns: int = 0
    fallback_turns: list[int] = Field(default_factory=list)


class CallDetail(CallSummary):
    transcript: list[TranscriptTurn]
    events: list[CallEventOut]
    extraction: ExtractionOut | None
    call_config: dict[str, Any] | None = None
    latency: LatencySummaryOut | None = None


StageStatus = Literal["done", "in_progress", "planned", "deferred"]
ComponentStatus = Literal["configured", "not_configured", "not_connected"]


class StageOut(BaseModel):
    id: str
    title: str
    status: StageStatus


class ComponentOut(BaseModel):
    name: str
    provider: str | None
    status: ComponentStatus


class OverviewResponse(BaseModel):
    stages: list[StageOut]
    components: list[ComponentOut]
