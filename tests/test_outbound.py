"""Tests for Milestone M2: outbound telephony, phone masking, and call termination."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.api import (
    CreateAgentDispatchRequest,
    CreateRoomRequest,
    DeleteRoomRequest,
    TwirpError,
    TwirpErrorCode,
)

from agent.telephony import (
    extract_job_phone_number,
    mask_phone_number,
    validate_e164,
)
from agent.tools import SessionState, terminate_call
from app.config import Settings
from scripts.call_me import dispatch_phone_call


def test_validate_e164() -> None:
    """Verifies strict ITU-T E.164 phone number validation."""
    valid_numbers = [
        "+48123456789",
        "+48791234567",
        "+14155552671",
        "+380501234567",
        "+442071838750",
    ]
    for num in valid_numbers:
        assert validate_e164(num) is True, f"Expected {num} to be valid E.164"

    invalid_numbers = [
        "48123456789",  # missing leading +
        "+0123456789",  # country code cannot start with 0
        "+48 123 456 789",  # spaces not allowed
        "+48-123-456-789",  # hyphens not allowed
        "+(48)123456789",  # parentheses not allowed
        "+1234567890123456",  # 16 digits (max allowed is 15)
        "+",  # just +
        "",  # empty
        "invalid",  # letters
    ]
    for num in invalid_numbers:
        assert validate_e164(num) is False, f"Expected {num} to be invalid E.164"


def test_mask_phone_number() -> None:
    """Verifies phone number masking preserves only the last 3 digits."""
    assert mask_phone_number("+48123456789") == "*********789"
    assert mask_phone_number("+14155552671") == "*********671"
    assert mask_phone_number("123") == "***"
    assert mask_phone_number("12") == "**"
    assert mask_phone_number("") == ""

    # Sensitive parts of number should not be visible
    masked = mask_phone_number("+48123456789")
    assert "+48123" not in masked
    assert masked.endswith("789")


def test_extract_job_phone_number() -> None:
    """Verifies extraction of phone_number from job metadata JSON."""
    valid_meta = json.dumps({"phone_number": "+48123456789"})
    assert extract_job_phone_number(valid_meta) == (True, "+48123456789")

    # Invalid phone in metadata (key is present, but phone is not valid E.164)
    invalid_meta = json.dumps({"phone_number": "not-e164"})
    assert extract_job_phone_number(invalid_meta) == (True, "not-e164")

    # Browser mode cases (no phone_number key or invalid JSON)
    assert extract_job_phone_number(None) == (False, None)
    assert extract_job_phone_number("") == (False, None)
    assert extract_job_phone_number("{}") == (False, None)
    assert extract_job_phone_number(json.dumps({"unrelated": "field"})) == (
        False,
        None,
    )
    assert extract_job_phone_number("not-a-json-string") == (False, None)


@pytest.mark.asyncio
async def test_dispatch_phone_call_validation_failures() -> None:
    """Verifies that dispatch_phone_call fails early with clear error messages."""
    settings = Settings(
        livekit_url="wss://test.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        agent_dispatch_name="",
        sip_outbound_trunk_id="",
    )

    # 1. Invalid E.164 - must mask the phone number in error message
    with pytest.raises(ValueError) as exc_info:
        await dispatch_phone_call("48123456789", settings)
    assert "Invalid E.164 phone number" in str(exc_info.value)
    assert mask_phone_number("48123456789") in str(exc_info.value)
    assert "48123456" not in str(exc_info.value)

    # 2. Missing AGENT_DISPATCH_NAME
    with pytest.raises(ValueError, match="AGENT_DISPATCH_NAME is not set"):
        await dispatch_phone_call("+48123456789", settings)

    # 3. Missing SIP_OUTBOUND_TRUNK_ID
    settings.agent_dispatch_name = "test-agent"
    with pytest.raises(ValueError, match="SIP_OUTBOUND_TRUNK_ID is not set"):
        await dispatch_phone_call("+48123456789", settings)


@pytest.mark.asyncio
async def test_dispatch_phone_call_mocked_api() -> None:
    """Verifies room creation and agent dispatch payload with mocked LiveKitAPI."""
    settings = Settings(
        livekit_url="wss://test.livekit.cloud",
        livekit_api_key="test-key",
        livekit_api_secret="test-secret",
        agent_dispatch_name="callback-agent",
        sip_outbound_trunk_id="ST_test123",
    )

    mock_lkapi = MagicMock()
    mock_lkapi.room.create_room = AsyncMock()
    mock_dispatch = MagicMock()
    mock_dispatch.id = "disp_xyz987"
    mock_lkapi.agent_dispatch.create_dispatch = AsyncMock(return_value=mock_dispatch)

    room_name, dispatch_id = await dispatch_phone_call(
        phone_number="+48123456789",
        app_settings=settings,
        lkapi=mock_lkapi,
    )

    assert room_name.startswith("call-")
    assert dispatch_id == "disp_xyz987"

    # Verify create_room called with proper request
    mock_lkapi.room.create_room.assert_awaited_once()
    create_room_arg: CreateRoomRequest = mock_lkapi.room.create_room.call_args[0][0]
    assert create_room_arg.name == room_name
    assert create_room_arg.empty_timeout == 300

    # Verify create_dispatch called with expected metadata
    mock_lkapi.agent_dispatch.create_dispatch.assert_awaited_once()
    dispatch_arg: CreateAgentDispatchRequest = (
        mock_lkapi.agent_dispatch.create_dispatch.call_args[0][0]
    )
    assert dispatch_arg.agent_name == "callback-agent"
    assert dispatch_arg.room == room_name
    assert json.loads(dispatch_arg.metadata) == {"phone_number": "+48123456789"}


@pytest.mark.asyncio
async def test_worker_invalid_phone_metadata_aborts() -> None:
    """Verifies that an invalid phone number in metadata aborts.

    Must not fall back to browser mode.
    """
    from agent.worker import entrypoint

    mock_ctx = MagicMock()
    mock_ctx.room.name = "test-call-room"
    mock_ctx.job.metadata = json.dumps({"phone_number": "invalid-phone-12345"})
    mock_ctx.api.room.delete_room = AsyncMock()
    mock_ctx.shutdown = MagicMock()
    mock_ctx.wait_for_participant = AsyncMock()

    await entrypoint(mock_ctx)

    mock_ctx.api.room.delete_room.assert_awaited_once()
    req = mock_ctx.api.room.delete_room.call_args[0][0]
    assert req.room == "test-call-room"
    mock_ctx.shutdown.assert_called_once_with(reason="invalid phone_number in metadata")
    # Must NOT fall back to browser mode (wait_for_participant)
    mock_ctx.wait_for_participant.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_call_phone_mode_order() -> None:
    """Verifies that speech playout finishes BEFORE room deletion in phone mode."""
    events: list[str] = []

    mock_handle = MagicMock()

    async def mock_wait_for_playout() -> None:
        events.append("say_finished")

    mock_handle.wait_for_playout = AsyncMock(side_effect=mock_wait_for_playout)

    mock_session = MagicMock()
    mock_session.say = MagicMock(return_value=mock_handle)

    async def mock_aclose() -> None:
        events.append("session_aclose")

    mock_session.aclose = AsyncMock(side_effect=mock_aclose)

    mock_api = MagicMock()

    async def mock_delete_room(_req: DeleteRoomRequest) -> None:
        events.append("delete_room")

    mock_api.room.delete_room = AsyncMock(side_effect=mock_delete_room)

    state = SessionState(
        closing_phrase="Dziękuję za rozmowę, do usłyszenia!",
        phone_mode=True,
        room_name="call-test-room-123",
        api=mock_api,
    )

    task = terminate_call(mock_session, state, state.closing_phrase)
    assert task is not None
    await task

    # Assert exact order: say finished -> session aclose -> delete room
    assert events == ["say_finished", "session_aclose", "delete_room"]

    # Verify delete_room was called with correct room
    mock_api.room.delete_room.assert_awaited_once()
    req = mock_api.room.delete_room.call_args[0][0]
    assert req.room == "call-test-room-123"


@pytest.mark.asyncio
async def test_terminate_call_phone_mode_not_found_handled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that NOT_FOUND TwirpError is handled gracefully.

    Treated as success and logged at DEBUG.
    """
    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_api = MagicMock()
    mock_api.room.delete_room = AsyncMock(
        side_effect=TwirpError(TwirpErrorCode.NOT_FOUND, "room not found", status=404)
    )

    state = SessionState(
        closing_phrase="Do widzenia!",
        phone_mode=True,
        room_name="already-deleted-room",
        api=mock_api,
    )

    with caplog.at_level(logging.DEBUG, logger="agent.tools"):
        task = terminate_call(mock_session, state, state.closing_phrase)
        assert task is not None
        await task

    # Assert delete_room was attempted
    mock_api.room.delete_room.assert_awaited_once()
    # Assert it was logged at DEBUG, not WARNING
    assert any(
        "already deleted" in record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


@pytest.mark.asyncio
async def test_terminate_call_browser_mode_no_delete_room() -> None:
    """Verifies that browser mode does NOT delete the room upon termination."""
    mock_handle = MagicMock()
    mock_handle.wait_for_playout = AsyncMock()

    mock_session = MagicMock()
    mock_session.say = MagicMock(return_value=mock_handle)
    mock_session.aclose = AsyncMock()

    mock_api = MagicMock()
    mock_api.room.delete_room = AsyncMock()

    state = SessionState(
        closing_phrase="Do widzenia!",
        phone_mode=False,
        room_name="playground-room",
        api=mock_api,
    )

    task = terminate_call(mock_session, state, state.closing_phrase)
    assert task is not None
    await task

    mock_session.say.assert_called_once_with("Do widzenia!", allow_interruptions=False)
    mock_handle.wait_for_playout.assert_awaited_once()
    mock_session.aclose.assert_awaited_once()
    # In browser mode, delete_room must NOT be called
    mock_api.room.delete_room.assert_not_called()
