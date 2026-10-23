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


def is_question_text(text: str | None) -> bool:
    """Returns True if the text ends with a question mark.

    Handles trailing whitespace, quotes, brackets, and secondary
    punctuation (e.g. ?!, ?..).
    """
    if not text:
        return False
    cleaned = text.strip().rstrip(" \t\n\r\"'`»”’)]}")
    cleaned = cleaned.rstrip(".!")
    cleaned = cleaned.rstrip(" \t\n\r\"'`»”’)]}")
    return cleaned.endswith("?")


def _get_message_role(msg: Any) -> str:
    """Extracts role in lowercase from a ChatMessage, dict, or object."""
    if isinstance(msg, dict):
        return str(msg.get("role", "")).lower()
    role = getattr(msg, "role", None)
    if role is None:
        return ""
    if hasattr(role, "value") and isinstance(role.value, str):
        return role.value.lower()
    if isinstance(role, str):
        return role.lower()
    return ""


def _extract_message_text(msg: Any) -> str:
    """Extracts plain text content from a ChatMessage, dict, or object."""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        return str(msg.get("content") or msg.get("text") or "")
    if hasattr(msg, "text_content") and isinstance(msg.text_content, str):
        return msg.text_content
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif hasattr(p, "text") and isinstance(p.text, str):
                    parts.append(p.text)
            return " ".join(parts).strip()
    return ""


def _extract_history_messages(history: Any) -> list[Any]:
    """Safely extracts message list from a ChatContext, list, or object."""
    if history is None:
        return []
    if isinstance(history, (list, tuple)):
        return list(history)
    if hasattr(history, "messages"):
        msgs = history.messages
        if callable(msgs):
            try:
                res = msgs()
                if isinstance(res, (list, tuple)):
                    return list(res)
            except Exception:
                pass
        elif isinstance(msgs, (list, tuple)):
            return list(msgs)
    if hasattr(history, "items"):
        items = history.items
        if callable(items):
            try:
                res = items()
                if isinstance(res, (list, tuple)):
                    return list(res)
            except Exception:
                pass
        elif isinstance(items, (list, tuple)):
            return list(items)
    return []


def _extract_speech_text(speech_handle: Any) -> str:
    """Safely extracts spoken assistant text from SpeechHandle.chat_items."""
    if speech_handle is None:
        return ""
    chat_items = getattr(speech_handle, "chat_items", None)
    if not isinstance(chat_items, (list, tuple)):
        return ""
    texts: list[str] = []
    for item in chat_items:
        role = _get_message_role(item)
        if role == "assistant":
            t = _extract_message_text(item)
            if t:
                texts.append(t)
    return " ".join(texts).strip()


def has_pending_agent_question(
    messages: list[Any] | None = None,
    current_generation_text: str | None = None,
) -> bool:
    """Determines whether an agent question is pending without a user response.

    Returns True if:
    1. The agent's text in the current generation ends with a question.
    2. OR the last agent message in history ends with a question and there was
       no user message after it.
    """
    if current_generation_text and is_question_text(current_generation_text):
        return True

    if not messages:
        return False

    last_agent_idx: int | None = None
    last_agent_text: str = ""
    last_user_idx: int | None = None

    for idx, item in enumerate(messages):
        role = _get_message_role(item)
        if role == "assistant":
            last_agent_idx = idx
            last_agent_text = _extract_message_text(item)
        elif role == "user":
            last_user_idx = idx

    if last_agent_idx is not None and is_question_text(last_agent_text):
        if last_user_idx is None or last_user_idx < last_agent_idx:
            return True

    return False


@function_tool
async def end_call(context: RunContext) -> None:
    """Terminates call session when questions are done or caller wishes to finish.

    Call this tool ONLY AFTER the user has answered the last question or
    wants to finish the call. NEVER call this tool in the same response as
    asking a question. This tool speaks the goodbye phrase itself, so do
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

    # Guard: do not end call if agent just asked a question awaiting user's answer
    current_text = _extract_speech_text(getattr(context, "speech_handle", None))
    history_messages = _extract_history_messages(getattr(session, "history", None))

    if has_pending_agent_question(
        messages=history_messages,
        current_generation_text=current_text,
    ):
        logger.warning("end_call rejected: pending question")
        return None

    logger.info("end_call tool triggered, closing call session.")
    terminate_call(session, state, phrase=state.closing_phrase)
    raise StopResponse()
