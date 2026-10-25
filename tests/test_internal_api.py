"""Tests for worker internal reporting API endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.live import broadcaster
from app.models import Call, CallEvent, CallTurn


@pytest.fixture
async def sample_call(db_session: AsyncSession) -> Call:
    """Fixture providing a persisted Call instance in 'dialing' status."""
    call = Call(
        room_name=f"test-room-{uuid.uuid4().hex[:8]}",
        mode="browser",
        status="dialing",
    )
    db_session.add(call)
    await db_session.commit()
    await db_session.refresh(call)
    return call


@pytest.mark.asyncio
async def test_auth_missing_secret(
    async_client: httpx.AsyncClient, sample_call: Call
) -> None:
    """Verifies 401 Unauthorized when X-Internal-Secret header is omitted."""
    resp = await async_client.post(
        f"/internal/calls/{sample_call.id}/event",
        json={"event_type": "test", "message": "hello"},
    )
    assert resp.status_code == 401
    assert "Invalid or missing internal API secret" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_wrong_secret(
    async_client: httpx.AsyncClient, sample_call: Call
) -> None:
    """Verifies 401 Unauthorized when X-Internal-Secret header does not match."""
    resp = await async_client.post(
        f"/internal/calls/{sample_call.id}/event",
        headers={"X-Internal-Secret": "wrong-secret"},
        json={"event_type": "test", "message": "hello"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_server_secret_empty(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies 503 Service Unavailable when server has no secret configured."""
    import app.api.internal

    monkeypatch.setattr(app.api.internal.settings, "internal_api_secret", "")
    resp = await async_client.post(
        f"/internal/calls/{sample_call.id}/event",
        headers={"X-Internal-Secret": "any-secret"},
        json={"event_type": "test", "message": "hello"},
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_unknown_call_id_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    """Verifies 404 Not Found on all internal endpoints for non-existent call_id."""
    fake_id = uuid.uuid4()
    headers = {"X-Internal-Secret": settings.internal_api_secret}

    resp_event = await async_client.post(
        f"/internal/calls/{fake_id}/event",
        headers=headers,
        json={"event_type": "test", "message": "msg"},
    )
    assert resp_event.status_code == 404

    resp_turn = await async_client.post(
        f"/internal/calls/{fake_id}/turn",
        headers=headers,
        json={"turn_index": 1, "role": "agent", "content": "hi"},
    )
    assert resp_turn.status_code == 404

    resp_ended = await async_client.post(
        f"/internal/calls/{fake_id}/ended",
        headers=headers,
        json={"status": "completed"},
    )
    assert resp_ended.status_code == 404

    resp_extract = await async_client.post(
        f"/internal/calls/{fake_id}/extraction",
        headers=headers,
        json={"extraction_status": "ok"},
    )
    assert resp_extract.status_code == 404


@pytest.mark.asyncio
async def test_record_call_event_and_participant_joined(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    db_session: AsyncSession,
) -> None:
    """Verifies event recording and status transition on participant_joined."""
    headers = {"X-Internal-Secret": settings.internal_api_secret}
    ts = datetime.now(UTC)

    resp = await async_client.post(
        f"/internal/calls/{sample_call.id}/event",
        headers=headers,
        json={
            "event_type": "participant_joined",
            "message": "User connected",
            "payload": {"identity": "user-123"},
            "timestamp": ts.isoformat(),
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    await db_session.refresh(sample_call)
    assert sample_call.status == "in_progress"
    assert sample_call.started_at is not None

    events = (
        await db_session.scalars(
            select(CallEvent).where(CallEvent.call_id == sample_call.id)
        )
    ).all()
    assert len(events) == 1
    assert events[0].event_type == "participant_joined"
    assert events[0].payload == {"identity": "user-123"}


@pytest.mark.asyncio
async def test_record_call_turn_idempotency(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    db_session: AsyncSession,
) -> None:
    """Verifies recording turns is idempotent for identical (call_id, turn_index)."""
    headers = {"X-Internal-Secret": settings.internal_api_secret}

    # First attempt: creates row
    resp1 = await async_client.post(
        f"/internal/calls/{sample_call.id}/turn",
        headers=headers,
        json={"turn_index": 1, "role": "agent", "content": "Доброго дня!"},
    )
    assert resp1.status_code == 200

    # Duplicate attempt: must succeed idempotently without creating a second row
    resp2 = await async_client.post(
        f"/internal/calls/{sample_call.id}/turn",
        headers=headers,
        json={"turn_index": 1, "role": "agent", "content": "Доброго дня!"},
    )
    assert resp2.status_code == 200

    turns = (
        await db_session.scalars(
            select(CallTurn).where(CallTurn.call_id == sample_call.id)
        )
    ).all()
    assert len(turns) == 1
    assert turns[0].turn_index == 1
    assert turns[0].content == "Доброго дня!"


@pytest.mark.asyncio
async def test_record_call_ended_and_extraction(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    db_session: AsyncSession,
) -> None:
    """Verifies two-stage completion: /ended before extraction, /extraction after."""
    headers = {"X-Internal-Secret": settings.internal_api_secret}
    ended_at = datetime.now(UTC)

    # Stage 1: /ended
    resp_ended = await async_client.post(
        f"/internal/calls/{sample_call.id}/ended",
        headers=headers,
        json={
            "status": "completed",
            "ended_at": ended_at.isoformat(),
            "duration_seconds": 45,
            "latency_summary": {"turns": 3, "avg": 0.75, "p50": 0.7, "max": 0.9},
        },
    )
    assert resp_ended.status_code == 200

    await db_session.refresh(sample_call)
    assert sample_call.status == "completed"
    assert sample_call.duration_seconds == 45
    assert sample_call.latency_summary == {
        "turns": 3,
        "avg": 0.75,
        "p50": 0.7,
        "max": 0.9,
    }

    # Stage 2: /extraction
    resp_extract = await async_client.post(
        f"/internal/calls/{sample_call.id}/extraction",
        headers=headers,
        json={
            "extracted_answers": {
                "name": "Jane",
                "service": "App",
                "timeline": "Soon",
                "preferred_call_time": "14:00",
            },
            "extracted_missing": [],
            "extraction_status": "ok",
        },
    )
    assert resp_extract.status_code == 200

    await db_session.refresh(sample_call)
    assert sample_call.extraction_status == "ok"
    assert sample_call.extracted_answers["name"] == "Jane"
    assert sample_call.extracted_missing == []


@pytest.mark.asyncio
async def test_internal_endpoint_stores_turn_metrics_and_call_config(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    db_session: AsyncSession,
) -> None:
    """Verifies that internal endpoints store turn metrics and call_config."""
    headers = {"X-Internal-Secret": settings.internal_api_secret}

    # 1. Store call_config via event moving call to in_progress
    config_payload = {
        "llm_provider": "google",
        "llm_model": "gemini-3.5-flash-lite",
        "stt_provider": "deepgram",
        "stt_model": "nova-3",
    }
    resp_event = await async_client.post(
        f"/internal/calls/{sample_call.id}/event",
        headers=headers,
        json={
            "event_type": "participant_joined",
            "message": "User connected",
            "payload": {
                "identity": "user-123",
                "call_config": config_payload,
            },
        },
    )
    assert resp_event.status_code == 200
    await db_session.refresh(sample_call)
    assert sample_call.status == "in_progress"
    assert sample_call.call_config == config_payload

    # Verify event payload in DB does not keep call_config
    events = (
        await db_session.scalars(
            select(CallEvent).where(
                CallEvent.call_id == sample_call.id,
                CallEvent.event_type == "participant_joined",
            )
        )
    ).all()
    assert len(events) == 1
    assert events[0].payload == {"identity": "user-123"}

    # 2. Store turn metrics via /turn
    metrics_payload = {
        "e2e": 1.152,
        "e2e_source": "measured",
        "sdk_e2e": 1.152,
        "scripted": False,
        "endpointing_delay": 0.52,
        "transcript_delay": 0.25,
        "llm_ttft": 0.32,
        "tts_ttfb": 0.21,
        "llm_model": "gemini-3.5-flash-lite",
        "llm_fallback": False,
    }
    resp_turn = await async_client.post(
        f"/internal/calls/{sample_call.id}/turn",
        headers=headers,
        json={
            "turn_index": 1,
            "role": "agent",
            "content": "Доброго дня!",
            "metrics": metrics_payload,
        },
    )
    assert resp_turn.status_code == 200

    turns = (
        await db_session.scalars(
            select(CallTurn).where(CallTurn.call_id == sample_call.id)
        )
    ).all()
    assert len(turns) == 1
    assert turns[0].turn_index == 1
    assert turns[0].metrics == metrics_payload


@pytest.mark.asyncio
async def test_record_call_event_call_config_when_status_unchanged_no_status_sse(
    async_client: httpx.AsyncClient,
    sample_call: Call,
    db_session: AsyncSession,
) -> None:
    """Verifies that when call is already in_progress, an event with call_config
    updates call.call_config without calling update_call_status (no status SSE event).
    """
    headers = {"X-Internal-Secret": settings.internal_api_secret}

    # Move call to in_progress first
    sample_call.status = "in_progress"
    sample_call.call_config = {"initial": "config"}
    await db_session.commit()

    queue = broadcaster.subscribe()
    try:
        updated_config = {
            "llm_provider": "google",
            "llm_model": "gemini-2.5-flash",
            "turn_detection_mode": "vad",
        }
        resp = await async_client.post(
            f"/internal/calls/{sample_call.id}/event",
            headers=headers,
            json={
                "event_type": "diagnostic_event",
                "message": "Routine health event",
                "payload": {
                    "reason": "periodic_check",
                    "call_config": updated_config,
                },
            },
        )
        assert resp.status_code == 200

        await db_session.refresh(sample_call)
        assert sample_call.status == "in_progress"
        assert sample_call.call_config == updated_config

        # Check DB event payload does not include call_config
        events = (
            await db_session.scalars(
                select(CallEvent).where(
                    CallEvent.call_id == sample_call.id,
                    CallEvent.event_type == "diagnostic_event",
                )
            )
        ).all()
        assert len(events) == 1
        assert events[0].payload == {"reason": "periodic_check"}

        # Drain broadcaster events: only 'event' kind must be emitted, NO 'status' kind
        kinds = []
        while not queue.empty():
            msg = queue.get_nowait()
            kinds.append(msg.get("kind"))
        assert "event" in kinds
        assert "status" not in kinds
    finally:
        broadcaster.unsubscribe(queue)
