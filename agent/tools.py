"""Voice agent tools for collecting answers and terminating sessions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from livekit.agents import RunContext, StopResponse, function_tool
from livekit.api import DeleteRoomRequest, TwirpError, TwirpErrorCode

if TYPE_CHECKING:
    from livekit.api import LiveKitAPI

logger = logging.getLogger("agent.tools")


@dataclass
class SessionState:
    """Mutable session state passed across tool invocations."""

    closing_phrase: str = ""
    error_phrase: str = ""
    answers: dict[str, str] = field(default_factory=dict)
    call_ended: bool = False
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    phone_mode: bool = False
    room_name: str | None = None
    api: LiveKitAPI | None = None


def save_answer_to_state(state: SessionState, field_name: str, value: str) -> None:
    """Helper to record an answer into session state."""
    state.answers[field_name] = value
    logger.info("Saved answer for field '%s': %s", field_name, value)


@function_tool
async def save_answer(context: RunContext, field: str, value: str) -> str:
    """Stores a collected answer in the session state and logs it.

    Args:
        field: Name of the field being answered (e.g. 'name', 'service',
            'timeline', 'preferred_call_time').
        value: The answer value provided by the user.
    """
    state: SessionState = context.userdata
    save_answer_to_state(state, field, value)
    return f"Saved answer for {field}."


async def _say_and_disconnect(
    session: Any,
    phrase: str,
    state: SessionState | None = None,
) -> None:
    """Speaks the termination phrase, waits for playout, and closes the RTC session.

    In phone mode, room deletion is only executed after speech playout completes
    so speech is never cut off.
    """
    try:
        # Yield briefly so any pending events finalize in the runtime
        await asyncio.sleep(0.1)
        if phrase:
            handle = session.say(
                phrase,
                allow_interruptions=False,
            )
            if handle is not None:
                await handle.wait_for_playout()
    except Exception as e:
        logger.warning("Error during speech playout: %s", e)
    finally:
        await session.aclose()
        if state and state.phone_mode and state.api and state.room_name:
            try:
                await state.api.room.delete_room(
                    DeleteRoomRequest(room=state.room_name)
                )
                logger.info(
                    "Deleted room '%s' to terminate phone call.", state.room_name
                )
            except TwirpError as e:
                if e.code == TwirpErrorCode.NOT_FOUND:
                    logger.debug(
                        "Room '%s' was already deleted (not found).",
                        state.room_name,
                    )
                else:
                    logger.warning(
                        "Twirp error deleting room '%s': %s",
                        state.room_name,
                        e,
                    )
            except Exception as e:
                logger.warning("Error deleting room '%s': %s", state.room_name, e)


def terminate_call(
    session: Any,
    state: SessionState,
    phrase: str,
) -> asyncio.Task[None] | None:
    """Speaks the given phrase without interruption and closes the session.

    Guarantees idempotency (runs only once per session) and retains a strong reference
    to the background task on state.background_tasks so it cannot be garbage-collected.
    """
    if state.call_ended:
        return None
    state.call_ended = True

    task = asyncio.create_task(_say_and_disconnect(session, phrase, state))
    state.background_tasks.add(task)
    task.add_done_callback(state.background_tasks.discard)
    return task


@function_tool
async def end_call(context: RunContext) -> None:
    """Speaks the closing phrase and ends the call session.

    Raises:
        StopResponse: Instructs the LiveKit agent runtime not to generate
            any follow-up LLM response (preventing a duplicate goodbye).
    """
    state: SessionState = context.userdata
    logger.info("end_call tool triggered.")
    terminate_call(context.session, state, state.closing_phrase)

    # StopResponse asks for silence (no reply_required) so the model says nothing more
    raise StopResponse()
