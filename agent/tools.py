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
    answers: dict[str, Any] = field(default_factory=dict)
    call_ended: bool = False
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    phone_mode: bool = False
    room_name: str | None = None
    api: LiveKitAPI | None = None


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
    """Terminates call session when questions are done or caller wishes to finish.

    Call this tool when all questions have been asked/answered or when the user
    wants to finish the call. This tool speaks the goodbye phrase itself, so do
    NOT say goodbye or speak a farewell in your own reply.

    Raises:
        StopResponse: Instructs the LiveKit agent runtime not to generate
            any follow-up LLM response.
    """
    session = context.session
    state: SessionState = context.userdata

    # Check if caller is speaking via LiveKit AgentSession.user_state
    if session.user_state == "speaking":
        logger.info("end_call invoked while user is speaking; aborting termination.")
        return None

    # Wait until the agent's current speech generation/playout finishes
    try:
        await context.wait_for_playout()
    except Exception as e:
        logger.warning("wait_for_playout in end_call: %s", e)

    # Re-check user state in case caller spoke during playout
    if session.user_state == "speaking":
        logger.info("User began speaking during speech playout; aborting termination.")
        return None

    logger.info("end_call tool triggered, closing call session.")
    terminate_call(session, state, phrase=state.closing_phrase)
    raise StopResponse()
