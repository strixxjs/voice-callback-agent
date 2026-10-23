"""Tests for POST /leads and the LiveKit wrapper (no network)."""

from __future__ import annotations

import base64
import json
import logging
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from livekit.api import TokenVerifier
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.livekit_rooms import LiveKitRooms, get_livekit_rooms
from app.main import app as fastapi_app
from app.models import Call, CallEvent, Lead

API_KEY = "test-api-key"
API_SECRET = "test-api-secret-0123456789abcdef0123456789abcdef"
RAW_PHONE = "+48123456789"


class FakeLiveKitAPI:
    """Stands in for livekit.api.LiveKitAPI; records requests, never hits network."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.fail_on: str | None = None
        self.closed = False
        self.init_kwargs: dict[str, Any] | None = None
        self.room = SimpleNamespace(create_room=self._create_room)
        self.agent_dispatch = SimpleNamespace(create_dispatch=self._create_dispatch)

    async def _create_room(self, req: Any) -> Any:
        self.requests.append(("create_room", req))
        if self.fail_on == "create_room":
            raise RuntimeError(f"boom {API_SECRET} {RAW_PHONE}")
        return SimpleNamespace(name=req.name)

    async def _create_dispatch(self, req: Any) -> Any:
        self.requests.append(("create_dispatch", req))
        if self.fail_on == "create_dispatch":
            raise RuntimeError(f"boom {API_SECRET}")
        return SimpleNamespace(id="AD_test123")

    async def aclose(self) -> None:
        self.closed = True

    def factory(self, **kwargs: Any) -> FakeLiveKitAPI:
        self.init_kwargs = kwargs
        return self


@pytest.fixture
def fake_lk(
    async_client: httpx.AsyncClient,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> FakeLiveKitAPI:
    """Overrides the LiveKit dependency with the real wrapper on a fake client."""
    monkeypatch.setattr(isolate_env, "livekit_url", "wss://test.livekit.cloud")
    monkeypatch.setattr(isolate_env, "livekit_api_key", API_KEY)
    monkeypatch.setattr(isolate_env, "livekit_api_secret", API_SECRET)
    fake = FakeLiveKitAPI()

    def _override() -> LiveKitRooms:
        return LiveKitRooms(
            url=isolate_env.livekit_url,
            api_key=isolate_env.livekit_api_key,
            api_secret=isolate_env.livekit_api_secret,
            agent_name=isolate_env.agent_dispatch_name,
            api_factory=fake.factory,
        )

    fastapi_app.dependency_overrides[get_livekit_rooms] = _override
    return fake


async def _count(db: AsyncSession, model: type) -> int:
    return int(await db.scalar(select(func.count()).select_from(model)) or 0)


async def _events(db: AsyncSession, call_id: uuid.UUID) -> list[CallEvent]:
    return list(
        (
            await db.scalars(
                select(CallEvent)
                .where(CallEvent.call_id == call_id)
                .order_by(CallEvent.timestamp, CallEvent.id)
            )
        ).all()
    )


def _app_logs(caplog: pytest.LogCaptureFixture) -> str:
    """Application log output only (the test-only aiosqlite driver logs SQL params)."""
    return "\n".join(
        r.getMessage() for r in caplog.records if r.name.split(".")[0] == "app"
    )


def _lead_body(**overrides: Any) -> dict[str, Any]:
    body = {"name": "Jane Doe", "phone": RAW_PHONE, "mode": "browser"}
    body.update(overrides)
    return body


async def test_phone_mode_rejected_without_side_effects(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
) -> None:
    resp = await async_client.post("/leads", json=_lead_body(mode="phone"))

    assert resp.status_code == 400
    assert resp.json() == {"detail": "Phone mode is disabled until Phone go-live"}
    assert await _count(db_session, Lead) == 0
    assert await _count(db_session, Call) == 0
    assert await _count(db_session, CallEvent) == 0
    assert fake_lk.init_kwargs is None
    assert fake_lk.requests == []


@pytest.mark.parametrize(
    "phone",
    [
        "48123456789",  # missing +
        "+0123456789",  # leading zero country code
        "+1234567",  # too short (7 digits)
        "+1234567890123456",  # too long (16 digits)
        "+48 123 456 789",  # spaces
        "+48abc456789",
    ],
)
async def test_invalid_phone_returns_422_without_echo(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    phone: str,
) -> None:
    resp = await async_client.post("/leads", json=_lead_body(phone=phone))

    assert resp.status_code == 422
    assert phone not in resp.text
    assert await _count(db_session, Lead) == 0
    assert fake_lk.requests == []


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
async def test_invalid_name_returns_422(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    name: str,
) -> None:
    resp = await async_client.post("/leads", json=_lead_body(name=name))

    assert resp.status_code == 422
    assert RAW_PHONE not in resp.text
    assert await _count(db_session, Lead) == 0


async def test_invalid_mode_returns_422(
    async_client: httpx.AsyncClient, fake_lk: FakeLiveKitAPI
) -> None:
    resp = await async_client.post("/leads", json=_lead_body(mode="sip"))
    assert resp.status_code == 422


async def test_browser_happy_path_auto_join(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    isolate_env: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AGENT_DISPATCH_NAME empty: room created, no explicit dispatch."""
    caplog.set_level(logging.DEBUG, logger="app")
    resp = await async_client.post("/leads", json=_lead_body(name="  Jane Doe  "))

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert set(data) == {
        "lead_id",
        "call_id",
        "room",
        "mode",
        "status",
        "livekit_url",
        "token",
    }
    assert data["mode"] == "browser"
    assert data["status"] == "dialing"
    assert data["room"] == f"call-{data['call_id']}"
    assert data["livekit_url"] == "wss://test.livekit.cloud"
    assert data["token"]
    assert RAW_PHONE not in resp.text

    lead = await db_session.get(Lead, uuid.UUID(data["lead_id"]))
    call = await db_session.get(Call, uuid.UUID(data["call_id"]))
    assert lead is not None and call is not None
    assert lead.name == "Jane Doe"
    assert lead.phone == RAW_PHONE  # raw phone lives in leads only
    assert call.lead_id == lead.id
    assert call.room_name == data["room"]
    assert call.mode == "browser"
    assert call.status == "dialing"

    # Explicit credentials passed (pydantic-settings does not populate os.environ)
    assert fake_lk.init_kwargs == {
        "url": "wss://test.livekit.cloud",
        "api_key": API_KEY,
        "api_secret": API_SECRET,
    }
    assert [name for name, _ in fake_lk.requests] == ["create_room"]
    room_req = fake_lk.requests[0][1]
    assert room_req.name == data["room"]
    assert room_req.empty_timeout > 0
    metadata = json.loads(room_req.metadata)
    assert metadata == {
        "call_id": data["call_id"],
        "lead_id": data["lead_id"],
        "mode": "browser",
    }
    assert RAW_PHONE not in room_req.metadata
    assert fake_lk.closed is True

    events = await _events(db_session, call.id)
    assert [e.event_type for e in events] == ["lead_created", "room_created"]
    for e in events:
        assert RAW_PHONE not in e.message
        assert RAW_PHONE not in json.dumps(e.payload or {})

    assert RAW_PHONE not in _app_logs(caplog)
    assert "*********789" in _app_logs(caplog)


async def test_browser_happy_path_explicit_dispatch(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENT_DISPATCH_NAME set: explicit dispatch with the same metadata."""
    monkeypatch.setattr(isolate_env, "agent_dispatch_name", "callback-agent")

    resp = await async_client.post("/leads", json=_lead_body())

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert [name for name, _ in fake_lk.requests] == ["create_room", "create_dispatch"]
    room_req = fake_lk.requests[0][1]
    dispatch_req = fake_lk.requests[1][1]
    assert dispatch_req.agent_name == "callback-agent"
    assert dispatch_req.room == data["room"]
    assert json.loads(dispatch_req.metadata) == json.loads(room_req.metadata)
    assert RAW_PHONE not in dispatch_req.metadata
    assert fake_lk.closed is True

    events = await _events(db_session, uuid.UUID(data["call_id"]))
    assert [e.event_type for e in events] == [
        "lead_created",
        "room_created",
        "dispatched",
    ]
    assert events[-1].payload == {
        "agent_name": "callback-agent",
        "dispatch_id": "AD_test123",
    }


def _jwt_payload(token: str) -> dict[str, Any]:
    part = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


async def test_participant_token_is_join_only_for_room(
    async_client: httpx.AsyncClient, fake_lk: FakeLiveKitAPI
) -> None:
    resp = await async_client.post("/leads", json=_lead_body())
    data = resp.json()

    claims = TokenVerifier(API_KEY, API_SECRET).verify(data["token"])
    assert claims.identity == f"lead-{data['lead_id']}"
    assert claims.name == "Jane Doe"
    assert claims.video is not None
    assert claims.video.room_join is True
    assert claims.video.room == data["room"]
    assert not claims.video.room_create
    assert not claims.video.room_admin
    assert not claims.video.room_list

    payload = _jwt_payload(data["token"])
    assert 3500 <= payload["exp"] - payload["nbf"] <= 3700
    assert RAW_PHONE not in json.dumps(payload)

    with pytest.raises(Exception):  # noqa: B017 - wrong secret must not verify
        TokenVerifier(API_KEY, "x" * 48).verify(data["token"])


@pytest.mark.parametrize(
    ("fail_on", "expected_events"),
    [
        ("create_room", ["lead_created", "error"]),
        ("create_dispatch", ["lead_created", "room_created", "error"]),
    ],
)
async def test_livekit_failure_marks_call_failed(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fail_on: str,
    expected_events: list[str],
) -> None:
    monkeypatch.setattr(isolate_env, "agent_dispatch_name", "callback-agent")
    fake_lk.fail_on = fail_on
    caplog.set_level(logging.DEBUG, logger="app")

    resp = await async_client.post("/leads", json=_lead_body())

    assert resp.status_code == 502
    assert resp.json() == {"detail": "Failed to start LiveKit session"}
    assert API_SECRET not in resp.text
    assert fake_lk.closed is True

    call = await db_session.scalar(select(Call))
    assert call is not None
    assert call.status == "failed"
    events = await _events(db_session, call.id)
    assert [e.event_type for e in events] == expected_events
    error = events[-1]
    assert error.payload == {"stage": fail_on, "error_type": "RuntimeError"}
    assert API_SECRET not in error.message
    assert RAW_PHONE not in error.message
    assert API_SECRET not in _app_logs(caplog)
    assert RAW_PHONE not in _app_logs(caplog)


async def test_missing_livekit_credentials_returns_502(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
    fake_lk: FakeLiveKitAPI,
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(isolate_env, "livekit_api_secret", None)

    resp = await async_client.post("/leads", json=_lead_body())

    assert resp.status_code == 502
    assert fake_lk.init_kwargs is None  # client never constructed
    call = await db_session.scalar(select(Call))
    assert call is not None and call.status == "failed"
    events = await _events(db_session, call.id)
    assert events[-1].payload == {
        "stage": "create_room",
        "error_type": "LiveKitNotConfiguredError",
    }


async def test_lead_flow_then_worker_join_reaches_detail(
    async_client: httpx.AsyncClient,
    fake_lk: FakeLiveKitAPI,
    isolate_env: Settings,
) -> None:
    """POST /leads → worker reports participant_joined → visible in /api/calls."""
    data = (await async_client.post("/leads", json=_lead_body())).json()
    resp = await async_client.post(
        f"/internal/calls/{data['call_id']}/event",
        headers={"X-Internal-Secret": isolate_env.internal_api_secret},
        json={"event_type": "participant_joined", "message": "joined"},
    )
    assert resp.status_code == 200

    detail = (await async_client.get(f"/api/calls/{data['call_id']}")).json()
    assert detail["status"] == "in_progress"
    assert detail["started_at"] is not None
    assert detail["phone_masked"] == "*********789"
    assert [e["event_type"] for e in detail["events"]] == [
        "lead_created",
        "room_created",
        "participant_joined",
    ]


async def test_wrapper_aclose_without_client_is_noop() -> None:
    factory_calls: list[dict[str, Any]] = []
    make: Callable[..., Any] = lambda **kw: factory_calls.append(kw)  # noqa: E731
    rooms = LiveKitRooms(
        url="wss://x", api_key=API_KEY, api_secret=API_SECRET, api_factory=make
    )
    await rooms.aclose()
    assert factory_calls == []
