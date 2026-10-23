"""Server-Sent Events endpoint for real-time console updates."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.live import Broadcaster, broadcaster

logger = logging.getLogger("app.api.events")

router = APIRouter(prefix="/api", tags=["events"])

DEFAULT_PING_INTERVAL_SECONDS = 15.0


async def sse_event_stream(
    broadcaster_inst: Broadcaster = broadcaster,
    *,
    call_id: str | None = None,
    ping_interval: float = DEFAULT_PING_INTERVAL_SECONDS,
) -> AsyncGenerator[str, None]:
    """Generates SSE formatted events from the broadcaster.

    Emits an initial ready event, filters events by call_id if supplied,
    sends heartbeat comments on timeout, and always unsubscribes on exit.
    """
    queue = broadcaster_inst.subscribe()
    try:
        yield "event: ready\ndata: {}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=ping_interval)
                if call_id is not None and event.get("call_id") != call_id:
                    continue
                kind = event.get("kind", "message")
                payload = json.dumps(event)
                yield f"event: {kind}\ndata: {payload}\n\n"
            except TimeoutError:
                yield ": ping\n\n"
    finally:
        broadcaster_inst.unsubscribe(queue)


@router.get("/events")
async def get_events(
    call_id: Annotated[
        str | None, Query(description="Filter events by call_id")
    ] = None,
) -> StreamingResponse:
    """Streams live call updates over Server-Sent Events."""
    return StreamingResponse(
        sse_event_stream(broadcaster, call_id=call_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
