"""Tests for the live SSE broadcaster, streaming generator, and console pages."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.api.events import sse_event_stream
from app.config import Settings
from app.live import Broadcaster, broadcaster
from app.livekit_rooms import LiveKitRooms, get_livekit_rooms
from app.main import app as fastapi_app

API_KEY = "test-api-key"
API_SECRET = "test-api-secret-0123456789abcdef0123456789abcdef"
INTERNAL_SECRET = "test-internal-secret-xyz"


class FakeLiveKitAPI:
    """Fake LiveKit client recording calls without network."""

    def __init__(self) -> None:
        self.room = SimpleNamespace(create_room=self._create_room)
        self.agent_dispatch = SimpleNamespace(create_dispatch=self._create_dispatch)

    async def _create_room(self, req: Any) -> Any:
        return SimpleNamespace(name=req.name)

    async def _create_dispatch(self, req: Any) -> Any:
        return SimpleNamespace(id="AD_test123")

    async def aclose(self) -> None:
        pass

    def factory(self, **kwargs: Any) -> FakeLiveKitAPI:
        return self


@pytest.fixture
def fake_lk(
    isolate_env: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> FakeLiveKitAPI:
    monkeypatch.setattr(isolate_env, "livekit_url", "wss://test.livekit.cloud")
    monkeypatch.setattr(isolate_env, "livekit_api_key", API_KEY)
    monkeypatch.setattr(isolate_env, "livekit_api_secret", API_SECRET)
    monkeypatch.setattr(isolate_env, "internal_api_secret", INTERNAL_SECRET)
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


def test_broadcaster_subscribe_publish_unsubscribe() -> None:
    """Verifies basic subscription, publishing, and unsubscription."""
    b = Broadcaster()
    q = b.subscribe()
    assert len(b._subscribers) == 1

    event = {"kind": "test", "call_id": "123"}
    b.publish(event)
    assert not q.empty()
    assert q.get_nowait() == event

    b.unsubscribe(q)
    assert len(b._subscribers) == 0

    b.publish({"kind": "after_unsub"})
    assert q.empty()


def test_broadcaster_full_queue_drops_without_raising() -> None:
    """Verifies that publishing to a full queue drops the event without error."""
    b = Broadcaster(maxsize=1)
    q = b.subscribe()

    ev1 = {"kind": "first"}
    ev2 = {"kind": "dropped"}

    b.publish(ev1)
    # Queue is now at capacity (1/1); second event should drop cleanly
    b.publish(ev2)

    assert q.get_nowait() == ev1
    assert q.empty()
    b.unsubscribe(q)


def test_broadcaster_multiple_subscribers() -> None:
    """Verifies fan-out to multiple subscribers."""
    b = Broadcaster()
    q1 = b.subscribe()
    q2 = b.subscribe()

    ev = {"kind": "broadcast", "msg": "hello"}
    b.publish(ev)

    assert q1.get_nowait() == ev
    assert q2.get_nowait() == ev

    b.unsubscribe(q1)
    b.unsubscribe(q2)


@pytest.mark.asyncio
async def test_service_publishes_on_lead_and_event(
    async_client: httpx.AsyncClient,
    fake_lk: FakeLiveKitAPI,
) -> None:
    """Verifies POST /leads and /internal/calls/{id}/event trigger broadcaster events

    without leaking phone numbers.
    """
    raw_phone = "+380501234567"
    queue = broadcaster.subscribe()

    try:
        # 1. Create lead
        res = await async_client.post(
            "/leads",
            json={"name": "Olek", "phone": raw_phone, "mode": "browser"},
        )
        assert res.status_code == 201
        data = res.json()
        call_id = data["call_id"]

        # Collect events emitted during lead creation
        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        assert len(events) >= 2
        kinds = [e["kind"] for e in events]
        assert "call_created" in kinds
        assert "event" in kinds

        for e in events:
            assert e["call_id"] == call_id
            payload_str = json.dumps(e)
            assert raw_phone not in payload_str

        # 2. Add an internal event (participant_joined transitions status
        # to in_progress and appends event)
        event_res = await async_client.post(
            f"/internal/calls/{call_id}/event",
            headers={"X-Internal-Secret": INTERNAL_SECRET},
            json={
                "event_type": "participant_joined",
                "message": "User joined room",
                "payload": {"participant": "user-1"},
            },
        )
        assert event_res.status_code == 200

        internal_events: list[dict[str, Any]] = []
        while not queue.empty():
            internal_events.append(queue.get_nowait())

        internal_kinds = [e["kind"] for e in internal_events]
        assert "status" in internal_kinds
        assert "event" in internal_kinds

        event_item = next(e for e in internal_events if e["kind"] == "event")
        assert event_item["call_id"] == call_id
        assert event_item["data"]["event_type"] == "participant_joined"
        assert raw_phone not in json.dumps(internal_events)
    finally:
        broadcaster.unsubscribe(queue)


@pytest.mark.asyncio
async def test_sse_generator_ready_and_formatting() -> None:
    """Verifies initial ready event and proper SSE message formatting."""
    b = Broadcaster()
    gen = sse_event_stream(b, ping_interval=10.0)

    # 1. Initial event must be ready
    ready = await anext(gen)
    assert ready == "event: ready\ndata: {}\n\n"

    # 2. Published event is formatted as SSE
    ev = {
        "kind": "status",
        "call_id": "c1",
        "status": "in_progress",
        "data": {"status": "in_progress"},
        "ts": "2026-10-23T18:00:00Z",
    }
    b.publish(ev)

    formatted = await anext(gen)
    expected_data = json.dumps(ev)
    assert formatted == f"event: status\ndata: {expected_data}\n\n"

    await gen.aclose()


@pytest.mark.asyncio
async def test_sse_generator_call_id_filter() -> None:
    """Verifies that sse_event_stream filters out events for other call_ids."""
    b = Broadcaster()
    gen = sse_event_stream(b, call_id="target-123", ping_interval=10.0)

    # Ready event
    assert (await anext(gen)) == "event: ready\ndata: {}\n\n"

    # Event for another call should be skipped
    b.publish({"kind": "event", "call_id": "other-456", "data": {}})
    # Event for target call
    target_ev = {"kind": "event", "call_id": "target-123", "data": {"match": True}}
    b.publish(target_ev)

    item = await anext(gen)
    assert "target-123" in item
    assert "other-456" not in item

    await gen.aclose()


@pytest.mark.asyncio
async def test_sse_generator_heartbeat_and_unsubscribe_on_cancel() -> None:
    """Verifies periodic heartbeat pings and unsubscription on generator close."""
    b = Broadcaster()
    assert len(b._subscribers) == 0

    gen = sse_event_stream(b, ping_interval=0.01)
    assert (await anext(gen)) == "event: ready\ndata: {}\n\n"
    assert len(b._subscribers) == 1

    # Next item after 0.01 s timeout should be heartbeat ping
    ping = await anext(gen)
    assert ping == ": ping\n\n"

    # Canceling / closing generator must unsubscribe
    await gen.aclose()
    assert len(b._subscribers) == 0


@pytest.mark.asyncio
async def test_html_pages_and_static_assets(async_client: httpx.AsyncClient) -> None:
    """Verifies GET / and GET /calls return HTML and static assets return 200."""
    # 1. Overview page
    res_index = await async_client.get("/")
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers.get("content-type", "")
    assert "Solaris Voice Agent" in res_index.text
    assert "Тестовий лід" in res_index.text

    # 2. Calls page
    res_calls = await async_client.get("/calls")
    assert res_calls.status_code == 200
    assert "text/html" in res_calls.headers.get("content-type", "")
    assert "Solaris Voice Agent" in res_calls.text
    assert "Список дзвінків" in res_calls.text

    # 3. Static assets
    res_css = await async_client.get("/static/app.css")
    assert res_css.status_code == 200
    assert "--bg:" in res_css.text

    res_common = await async_client.get("/static/common.js")
    assert res_common.status_code == 200
    assert "STATUS_LABELS" in res_common.text

    res_overview = await async_client.get("/static/overview.js")
    assert res_overview.status_code == 200
    assert "connectLiveKit" in res_overview.text

    res_calls_js = await async_client.get("/static/calls.js")
    assert res_calls_js.status_code == 200
    assert "renderTranscript" in res_calls_js.text
