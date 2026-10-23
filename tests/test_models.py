"""Tests for SQLAlchemy database models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Call, CallEvent, CallTurn, Lead


@pytest.mark.asyncio
async def test_create_lead_and_call(db_session: AsyncSession) -> None:
    """Verifies creating a lead and call record with UUIDs and JSON fields."""
    lead = Lead(name="Test Lead", phone="+48123456789")
    db_session.add(lead)
    await db_session.commit()
    await db_session.refresh(lead)

    assert isinstance(lead.id, uuid.UUID)
    assert lead.name == "Test Lead"
    assert lead.phone == "+48123456789"
    assert lead.created_at is not None

    call = Call(
        lead_id=lead.id,
        room_name="test-room-123",
        mode="browser",
        status="dialing",
        extracted_answers={"name": "Test"},
        extracted_missing=["service"],
        latency_summary={"turns": 2, "avg": 0.5},
    )
    db_session.add(call)
    await db_session.commit()
    await db_session.refresh(call)

    assert isinstance(call.id, uuid.UUID)
    assert call.lead_id == lead.id
    assert call.room_name == "test-room-123"
    assert call.extracted_answers == {"name": "Test"}
    assert call.extracted_missing == ["service"]
    assert call.latency_summary == {"turns": 2, "avg": 0.5}


@pytest.mark.asyncio
async def test_call_turn_unique_constraint(db_session: AsyncSession) -> None:
    """Verifies that (call_id, turn_index) is strictly unique at DB level."""
    call = Call(room_name="test-turn-room", mode="browser", status="in_progress")
    db_session.add(call)
    await db_session.commit()
    await db_session.refresh(call)

    turn1 = CallTurn(
        call_id=call.id,
        turn_index=1,
        role="agent",
        content="Hello!",
        timestamp=datetime.now(UTC),
    )
    db_session.add(turn1)
    await db_session.commit()

    # Attempting to add same turn_index for same call must raise IntegrityError
    turn2 = CallTurn(
        call_id=call.id,
        turn_index=1,
        role="user",
        content="Hi duplicate",
        timestamp=datetime.now(UTC),
    )
    db_session.add(turn2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_call_events_cascade_delete(db_session: AsyncSession) -> None:
    """Verifies that deleting a call cascades to its turns and events."""
    call = Call(room_name="test-cascade-room", mode="browser")
    db_session.add(call)
    await db_session.commit()
    await db_session.refresh(call)

    turn = CallTurn(
        call_id=call.id,
        turn_index=1,
        role="agent",
        content="Hello",
        timestamp=datetime.now(UTC),
    )
    event = CallEvent(
        call_id=call.id,
        event_type="test_event",
        message="Event message",
        payload={"key": "val"},
        timestamp=datetime.now(UTC),
    )
    db_session.add_all([turn, event])
    await db_session.commit()

    # Delete call
    await db_session.delete(call)
    await db_session.commit()

    turns = (
        await db_session.scalars(select(CallTurn).where(CallTurn.call_id == call.id))
    ).all()
    events = (
        await db_session.scalars(select(CallEvent).where(CallEvent.call_id == call.id))
    ).all()
    assert len(turns) == 0
    assert len(events) == 0
