"""In-process SSE event broadcaster for the demo console.

Maintains bounded asyncio queues for active subscribers in a single uvicorn process.
Publishing is non-blocking and safe: full queues drop the event for that subscriber
without raising or blocking the caller.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("app.live")

DEFAULT_QUEUE_MAXSIZE = 100


class Broadcaster:
    """Manages active subscriber queues and fan-out publishing."""

    def __init__(self, maxsize: int = DEFAULT_QUEUE_MAXSIZE) -> None:
        self.maxsize = maxsize
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Registers a new subscriber and returns its bounded queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.maxsize)
        self._subscribers.add(queue)
        logger.debug("Subscriber connected; active count: %d", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Removes a subscriber queue."""
        self._subscribers.discard(queue)
        logger.debug(
            "Subscriber disconnected; active count: %d", len(self._subscribers)
        )

    def publish(self, event: dict[str, Any]) -> None:
        """Publishes an event to all subscribers non-blocking.

        If a subscriber's queue is full, the event is dropped for that subscriber
        and logged at debug level. Never blocks or raises into the caller.
        """
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug(
                    "Subscriber queue full; dropping event kind=%s call_id=%s",
                    event.get("kind"),
                    event.get("call_id"),
                )


# Global in-process broadcaster instance
broadcaster = Broadcaster()
