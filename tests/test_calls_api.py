"""Tests for GET /api/calls, GET /api/calls/{id} and GET /api/overview."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.calls import PROJECT_STAGES
from app.config import Settings
from app.models import Call, CallEvent, CallTurn, Lead

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


async def _seed_call(
    db: AsyncSession,
    *,
    created_at: datetime,
    name: str = "Jane",
    phone: str = "+48123456789",
    with_lead: bool = True,
    **call_fields: Any,
) -> Call:
    lead = None
    if with_lead:
        lead = Lead(name=name, phone=phone, created_at=created_at)
        db.add(lead)
        await db.flush()
    call_id = uuid.uuid4()
    call = Call(
        id=call_id,
        lead_id=lead.id if lead else None,
        room_name=f"call-{call_id}",
        mode="browser",
        status=call_fields.pop("status", "dialing"),
        created_at=created_at,
        **call_fields,
    )
    db.add(call)
    await db.commit()
    return call


async def test_list_calls_newest_first_with_limit(
    async_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    calls = [
        await _seed_call(db_session, created_at=T0 + timedelta(minutes=i), name=f"L{i}")
        for i in range(3)
    ]

    resp = await async_client.get("/api/calls")
    assert resp.status_code == 200
    items = resp.json()
    assert [i["call_id"] for i in items] == [str(c.id) for c in reversed(calls)]
    assert [i["lead_name"] for i in items] == ["L2", "L1", "L0"]

    limited = (await async_client.get("/api/calls?limit=2")).json()
    assert [i["call_id"] for i in limited] == [str(calls[2].id), str(calls[1].id)]


@pytest.mark.parametrize("limit", [0, 201, -1])
async def test_list_calls_limit_bounds(
    async_client: httpx.AsyncClient, limit: int
) -> None:
    resp = await async_client.get(f"/api/calls?limit={limit}")
    assert resp.status_code == 422


async def test_list_calls_item_shape_masking_and_duration(
    async_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    ended = await _seed_call(
        db_session,
        created_at=T0,
        status="completed",
        started_at=T0,
        ended_at=T0 + timedelta(seconds=45),
        duration_seconds=45,
    )
    db_session.add(
        CallEvent(
            call_id=ended.id,
            event_type="call_ended",
            message="Call ended by agent",
            payload={"ended_by": "agent"},
            timestamp=T0 + timedelta(seconds=45),
        )
    )
    await db_session.commit()
    computed = await _seed_call(
        db_session,
        created_at=T0 + timedelta(minutes=1),
        status="failed",
        started_at=T0,
        ended_at=T0 + timedelta(seconds=30),
    )
    active = await _seed_call(
        db_session,
        created_at=T0 + timedelta(minutes=2),
        status="in_progress",
        started_at=T0,
    )
    orphan = await _seed_call(
        db_session, created_at=T0 + timedelta(minutes=3), with_lead=False
    )

    resp = await async_client.get("/api/calls")
    assert "+48123456789" not in resp.text
    by_id = {i["call_id"]: i for i in resp.json()}

    item = by_id[str(ended.id)]
    assert set(item) == {
        "call_id",
        "lead_name",
        "phone_masked",
        "mode",
        "status",
        "room",
        "created_at",
        "started_at",
        "ended_at",
        "duration_seconds",
        "ended_by",
    }
    assert item["phone_masked"] == "*********789"
    assert item["status"] == "completed"
    assert item["room"] == ended.room_name
    assert item["duration_seconds"] == 45
    assert item["ended_by"] == "agent"

    assert by_id[str(computed.id)]["duration_seconds"] == 30
    assert by_id[str(computed.id)]["ended_by"] is None
    assert by_id[str(active.id)]["duration_seconds"] is None
    assert by_id[str(active.id)]["ended_at"] is None
    assert by_id[str(orphan.id)]["lead_name"] is None
    assert by_id[str(orphan.id)]["phone_masked"] is None


async def test_call_detail_transcript_events_extraction(
    async_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    call = await _seed_call(db_session, created_at=T0)
    for idx in (3, 1, 2):
        db_session.add(
            CallTurn(
                call_id=call.id,
                turn_index=idx,
                role="agent" if idx % 2 else "user",
                content=f"turn {idx}",
                timestamp=T0 + timedelta(seconds=idx),
            )
        )
    same_ts = T0 + timedelta(seconds=5)
    db_session.add_all(
        [
            CallEvent(
                call_id=call.id,
                event_type="late",
                message="late",
                timestamp=T0 + timedelta(seconds=9),
            ),
            CallEvent(call_id=call.id, event_type="a", message="a", timestamp=same_ts),
            CallEvent(call_id=call.id, event_type="b", message="b", timestamp=same_ts),
        ]
    )
    await db_session.commit()

    resp = await async_client.get(f"/api/calls/{call.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "+48123456789" not in resp.text
    assert data["phone_masked"] == "*********789"
    assert data["lead_name"] == "Jane"
    assert [t["turn_index"] for t in data["transcript"]] == [1, 2, 3]
    assert data["transcript"][0] == {
        "turn_index": 1,
        "role": "agent",
        "text": "turn 1",
        "timestamp": data["transcript"][0]["timestamp"],
    }
    assert [e["event_type"] for e in data["events"]] == ["a", "b", "late"]
    assert data["extraction"] is None

    call.extracted_answers = {"name": "Jane"}
    call.extracted_missing = ["timeline"]
    call.extraction_status = "ok"
    await db_session.commit()

    data = (await async_client.get(f"/api/calls/{call.id}")).json()
    assert data["extraction"] == {
        "answers": {"name": "Jane"},
        "missing": ["timeline"],
        "status": "ok",
    }


async def test_call_detail_404(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.get(f"/api/calls/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_overview_not_configured(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()

    assert data["stages"] == [s.model_dump() for s in PROJECT_STAGES]
    titles = {s["id"]: s["title"] for s in data["stages"]}
    assert titles == {
        "M0": "Налаштування",
        "M1": "Агент у браузері",
        "M2": "Код телефонного дзвінка",
        "M3": "Якість розмови",
        "M4": "Бекенд і консоль",
        "M5": "Резюме і доставка",
        "M6": "Готовність до демо",
        "phone_go_live": "Живий телефонний дзвінок",
    }
    statuses = {s["id"]: s["status"] for s in data["stages"]}
    assert statuses == {
        "M0": "done",
        "M1": "done",
        "M2": "done",
        "M3": "in_progress",
        "M4": "done",
        "M5": "planned",
        "M6": "planned",
        "phone_go_live": "deferred",
    }
    components = {c["name"]: c for c in data["components"]}
    assert set(components) == {"livekit", "llm", "stt", "tts", "telephony"}
    assert components["livekit"]["status"] == "not_configured"
    assert components["llm"]["status"] == "not_configured"
    assert components["stt"]["status"] == "not_configured"
    assert components["tts"]["status"] == "not_configured"
    assert components["telephony"]["status"] == "not_connected"


async def test_overview_configured(
    async_client: httpx.AsyncClient,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for attr, value in {
        "livekit_api_key": "k",
        "livekit_api_secret": "s",
        "google_api_key": "g",
        "deepgram_api_key": "d",
        "elevenlabs_api_key": "e",
    }.items():
        monkeypatch.setattr(isolate_env, attr, value)

    data = (await async_client.get("/api/overview")).json()
    components = {c["name"]: c for c in data["components"]}
    assert components["livekit"]["status"] == "configured"
    assert components["llm"] == {
        "name": "llm",
        "provider": "google",
        "status": "configured",
    }
    assert components["stt"]["status"] == "configured"
    assert components["tts"]["status"] == "configured"
    assert components["telephony"]["status"] == "not_connected"
    assert "provider" in components["livekit"]


async def test_overview_llm_follows_provider(
    async_client: httpx.AsyncClient,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(isolate_env, "llm_provider", "openai")
    monkeypatch.setattr(isolate_env, "google_api_key", "g")

    components = {
        c["name"]: c
        for c in (await async_client.get("/api/overview")).json()["components"]
    }
    assert components["llm"]["status"] == "not_configured"

    monkeypatch.setattr(isolate_env, "openai_api_key", "o")
    components = {
        c["name"]: c
        for c in (await async_client.get("/api/overview")).json()["components"]
    }
    assert components["llm"]["status"] == "configured"
