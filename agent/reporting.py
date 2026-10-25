"""CallReporter client for sending non-blocking call events to internal API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from agent.telephony import mask_phone_number
from app.config import settings

logger = logging.getLogger("agent.reporting")


def extract_call_id(metadata: Any) -> str | None:
    """Extracts call_id string from JSON metadata if present."""
    if not metadata or not isinstance(metadata, str):
        return None
    try:
        data = json.loads(metadata)
        if isinstance(data, dict) and "call_id" in data and data["call_id"]:
            return str(data["call_id"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def extract_chat_message_text(item: Any) -> str:
    """Extracts plain text content from a LiveKit ChatMessage or similar object."""
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text") and part.text:
                parts.append(str(part.text))
        return " ".join(parts).strip()
    return ""


class CallReporter:
    """Non-blocking reporter queuing call events and sending them sequentially."""

    def __init__(
        self,
        call_id: str | None,
        api_url: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.call_id = call_id
        self.is_noop = not bool(call_id)
        self.api_url = (api_url or settings.internal_api_url).rstrip("/")
        self.api_secret = (
            api_secret if api_secret is not None else settings.internal_api_secret
        )

        self._queue: asyncio.Queue[tuple[str, str, dict[str, Any]]] = asyncio.Queue()
        self._client: httpx.AsyncClient | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

        if not self.is_noop:
            self._ensure_worker_running()

    def set_call_id(self, call_id: str) -> None:
        """Upgrades reporter from no-op mode once call_id becomes available."""
        if self.is_noop and call_id:
            self.call_id = call_id
            self.is_noop = False
            self._ensure_worker_running()

    def _ensure_worker_running(self) -> None:
        """Starts the background worker task if not already running."""
        if self._worker_task is None or self._worker_task.done():
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=2.0)
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Background task processing queue sequentially to preserve order."""
        while True:
            try:
                method, path, payload = await self._queue.get()
                await self._send_with_retry(method, path, payload)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Unexpected error in CallReporter worker loop: %s",
                    mask_phone_number(str(e)),
                )

    async def _send_with_retry(
        self, method: str, path: str, payload: dict[str, Any]
    ) -> None:
        """Sends HTTP request to internal endpoint with at most 1 retry."""
        if self._client is None or self._client.is_closed:
            return

        url = f"{self.api_url}{path}"
        headers = {"X-Internal-Secret": self.api_secret}

        for attempt in range(2):
            try:
                resp = await self._client.request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 404:
                    logger.warning(
                        "Internal API returned 404 for %s: call_id %s not found",
                        path,
                        self.call_id,
                    )
                    break
                if resp.is_error:
                    safe_body = mask_phone_number(resp.text)
                    logger.warning(
                        "Internal API error %s on %s (attempt %d): %s",
                        resp.status_code,
                        path,
                        attempt + 1,
                        safe_body,
                    )
                    if attempt == 0:
                        await asyncio.sleep(0.1)
                        continue
                else:
                    break
            except Exception as e:
                safe_err = mask_phone_number(str(e))
                logger.warning(
                    "Failed to report to internal API %s (attempt %d): %s",
                    path,
                    attempt + 1,
                    safe_err,
                )
                if attempt == 0:
                    await asyncio.sleep(0.1)
                    continue

    def _enqueue(self, method: str, path: str, payload: dict[str, Any]) -> None:
        """Enqueues an item synchronously. Public methods are non-blocking."""
        if self.is_noop or self._closed:
            return
        self._ensure_worker_running()
        self._queue.put_nowait((method, path, payload))

    def report_event(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Enqueues an operational event."""
        if self.is_noop:
            return
        self._enqueue(
            "POST",
            f"/internal/calls/{self.call_id}/event",
            {
                "event_type": event_type,
                "message": message,
                "payload": payload,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def report_turn(
        self,
        role: str,
        content: str,
        turn_index: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Enqueues a conversational turn with worker-assigned turn_index."""
        if self.is_noop:
            return
        payload: dict[str, Any] = {
            "turn_index": turn_index,
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if metrics is not None:
            payload["metrics"] = metrics
        self._enqueue(
            "POST",
            f"/internal/calls/{self.call_id}/turn",
            payload,
        )

    def report_ended(
        self,
        status: str,
        duration_seconds: int | None = None,
        latency_summary: dict[str, Any] | None = None,
    ) -> None:
        """Enqueues call ended status before extraction."""
        if self.is_noop:
            return
        payload: dict[str, Any] = {
            "status": status,
            "ended_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration_seconds,
            "latency_summary": latency_summary,
        }
        self._enqueue(
            "POST",
            f"/internal/calls/{self.call_id}/ended",
            payload,
        )

    def report_extraction(
        self,
        extracted_answers: dict[str, Any] | None = None,
        extracted_missing: list[str] | None = None,
        extraction_status: str | None = None,
    ) -> None:
        """Enqueues post-call extraction answers and missing fields."""
        if self.is_noop:
            return
        self._enqueue(
            "POST",
            f"/internal/calls/{self.call_id}/extraction",
            {
                "extracted_answers": extracted_answers,
                "extracted_missing": extracted_missing,
                "extraction_status": extraction_status,
            },
        )

    async def aclose(self, flush_timeout: float = 5.0) -> None:
        """Flushes pending queue up to flush_timeout, then closes client."""
        if self.is_noop or self._closed:
            return
        self._closed = True

        try:
            # Wait for all queued items to be processed
            async with asyncio.timeout(flush_timeout):
                await self._queue.join()
        except TimeoutError:
            logger.warning(
                "Timed out flushing CallReporter queue after %.1fs",
                flush_timeout,
            )
        finally:
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task
            if self._client and not self._client.is_closed:
                await self._client.aclose()
