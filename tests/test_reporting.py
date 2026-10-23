"""Tests for CallReporter client, queue ordering, and metadata parsing."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent.reporting import (
    CallReporter,
    extract_call_id,
    extract_chat_message_text,
)
from agent.telephony import extract_job_phone_number


@pytest.mark.asyncio
async def test_extract_call_id() -> None:
    """Verifies parsing call_id from JSON metadata."""
    cid = str(uuid.uuid4())
    assert extract_call_id(json.dumps({"call_id": cid})) == cid
    assert (
        extract_call_id(
            json.dumps({"call_id": cid, "lead_id": "123", "mode": "browser"})
        )
        == cid
    )
    assert extract_call_id(None) is None
    assert extract_call_id("") is None
    assert extract_call_id("invalid-json") is None
    assert extract_call_id(json.dumps({"mode": "browser"})) is None
    assert extract_call_id(MagicMock()) is None


def test_metadata_never_yields_sip_mode_for_browser_lead() -> None:
    """Critical safety check: browser lead metadata NEVER triggers phone mode."""
    cid = str(uuid.uuid4())

    # Standard browser metadata: only call_id, lead_id, mode
    browser_meta = json.dumps({"call_id": cid, "lead_id": "123", "mode": "browser"})
    has_phone, phone = extract_job_phone_number(browser_meta)
    assert has_phone is False
    assert phone is None

    # Even if an errant phone key exists with mode="browser", must NOT yield phone mode
    errant_meta = json.dumps(
        {
            "call_id": cid,
            "mode": "browser",
            "phone_number": "+48123456789",
        }
    )
    has_phone2, phone2 = extract_job_phone_number(errant_meta)
    assert has_phone2 is False
    assert phone2 is None

    # Standard phone metadata: mode is not browser and phone_number is present
    phone_meta = json.dumps({"phone_number": "+48123456789"})
    has_phone3, phone3 = extract_job_phone_number(phone_meta)
    assert has_phone3 is True
    assert phone3 == "+48123456789"


def test_extract_chat_message_text() -> None:
    """Verifies extracting text from various ChatMessage content structures."""

    class MockItem:
        def __init__(self, content: Any) -> None:
            self.content = content

    class TextPart:
        def __init__(self, text: str) -> None:
            self.text = text

    assert extract_chat_message_text(MockItem("Hello world")) == "Hello world"
    assert extract_chat_message_text(MockItem(["Hello", "world"])) == "Hello world"
    assert (
        extract_chat_message_text(MockItem([TextPart("Part 1"), "Part 2"]))
        == "Part 1 Part 2"
    )
    assert extract_chat_message_text(MockItem(None)) == ""


@pytest.mark.asyncio
async def test_reporter_noop_mode() -> None:
    """Verifies that CallReporter with call_id=None acts as a complete no-op."""
    reporter = CallReporter(call_id=None)
    assert reporter.is_noop is True

    # None of these should raise or queue anything
    reporter.report_event("test", "msg")
    reporter.report_turn("agent", "hello", 1)
    reporter.report_ended("completed")
    reporter.report_extraction(extracted_answers={"name": "test"})

    await reporter.aclose()


@pytest.mark.asyncio
async def test_reporter_set_call_id_upgrades_from_noop() -> None:
    """Verifies that set_call_id transitions reporter from no-op to active."""
    reporter = CallReporter(call_id=None)
    assert reporter.is_noop is True

    reporter.set_call_id("new-call-id")
    assert reporter.is_noop is False
    assert reporter.call_id == "new-call-id"
    assert reporter._worker_task is not None

    await reporter.aclose()


@pytest.mark.asyncio
async def test_reporter_sequential_ordering_and_flush() -> None:
    """Verifies that CallReporter queues items and sends them in exact order."""
    sent_requests: list[tuple[str, str, dict[str, Any]]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        sent_requests.append((request.method, request.url.path, body))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    reporter = CallReporter(
        call_id="call-order-123",
        api_url="http://test",
        api_secret="test-secret",
    )
    # Replace internal client with our mock client
    reporter._client = mock_client

    reporter.report_event("agent_started", "Agent dispatched")
    reporter.report_turn("agent", "Turn 1", turn_index=1)
    reporter.report_turn("user", "Turn 2", turn_index=2)
    reporter.report_ended(status="completed", duration_seconds=30)
    reporter.report_extraction(
        extracted_answers={"name": "Alice"},
        extraction_status="ok",
    )

    # Flush queue on shutdown
    await reporter.aclose(flush_timeout=2.0)

    assert len(sent_requests) == 5
    assert sent_requests[0][1] == "/internal/calls/call-order-123/event"
    assert sent_requests[0][2]["event_type"] == "agent_started"

    assert sent_requests[1][1] == "/internal/calls/call-order-123/turn"
    assert sent_requests[1][2]["turn_index"] == 1

    assert sent_requests[2][1] == "/internal/calls/call-order-123/turn"
    assert sent_requests[2][2]["turn_index"] == 2

    assert sent_requests[3][1] == "/internal/calls/call-order-123/ended"
    assert sent_requests[3][2]["status"] == "completed"

    assert sent_requests[4][1] == "/internal/calls/call-order-123/extraction"
    assert sent_requests[4][2]["extracted_answers"] == {"name": "Alice"}


@pytest.mark.asyncio
async def test_reporter_swallows_errors_and_masks_phone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that network errors, 500 responses, and 404s never raise exceptions."""

    def mock_error_handler(request: httpx.Request) -> httpx.Response:
        # Return 500 error containing an unmasked phone number
        return httpx.Response(
            500,
            text="Internal error processing call for +48123456789",
        )

    transport = httpx.MockTransport(mock_error_handler)
    mock_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    reporter = CallReporter(
        call_id="call-err-123",
        api_url="http://test",
        api_secret="test-secret",
    )
    reporter._client = mock_client

    with caplog.at_level(logging.WARNING):
        reporter.report_event("error_test", "trigger error")
        await reporter.aclose(flush_timeout=2.0)

    # Must log warning without raising
    assert "Internal API error 500" in caplog.text
    # Phone number in error message must be masked
    assert "+48123456789" not in caplog.text
    assert "*********789" in caplog.text


@pytest.mark.asyncio
async def test_reporter_handles_404_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that 404 response logs a warning and does not retry."""
    attempts = 0

    def mock_404_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, json={"detail": "Call not found"})

    transport = httpx.MockTransport(mock_404_handler)
    mock_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    reporter = CallReporter(
        call_id="call-missing-123",
        api_url="http://test",
        api_secret="test-secret",
    )
    reporter._client = mock_client

    with caplog.at_level(logging.WARNING):
        reporter.report_event("test_404", "trigger 404")
        await reporter.aclose(flush_timeout=2.0)

    # Should not retry 404
    assert attempts == 1
    assert "Internal API returned 404" in caplog.text
