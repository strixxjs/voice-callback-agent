"""Script to dispatch an outbound phone call agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from livekit.api import (  # noqa: E402
    CreateAgentDispatchRequest,
    CreateRoomRequest,
    LiveKitAPI,
)

from agent.telephony import mask_phone_number, validate_e164  # noqa: E402
from app.config import Settings, settings  # noqa: E402

logger = logging.getLogger("scripts.call_me")


async def dispatch_phone_call(
    phone_number: str,
    app_settings: Settings,
    lkapi: LiveKitAPI | None = None,
) -> tuple[str, str]:
    """Validates parameters, creates room and agent dispatch for a phone call.

    Returns:
        tuple[room_name, dispatch_id]
    """
    if not validate_e164(phone_number):
        raise ValueError(
            f"Invalid E.164 phone number: '{mask_phone_number(phone_number)}'. "
            "Must match format '+<country_code><number>' (max 15 digits)."
        )

    if not app_settings.agent_dispatch_name:
        raise ValueError(
            "AGENT_DISPATCH_NAME is not set. Outbound phone calls require "
            "an explicit agent dispatch name. "
            "Please configure AGENT_DISPATCH_NAME in .env."
        )

    if not app_settings.sip_outbound_trunk_id:
        raise ValueError(
            "SIP_OUTBOUND_TRUNK_ID is not set. Outbound phone calls require "
            "a configured SIP trunk. "
            "Please configure SIP_OUTBOUND_TRUNK_ID in .env."
        )

    if not app_settings.livekit_url:
        raise ValueError("LIVEKIT_URL is not set.")
    if not app_settings.livekit_api_key or not app_settings.livekit_api_secret:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set.")

    room_name = f"call-{uuid.uuid4().hex[:12]}"

    close_api = False
    if lkapi is None:
        lkapi = LiveKitAPI(
            url=app_settings.livekit_url,
            api_key=app_settings.livekit_api_key,
            api_secret=app_settings.livekit_api_secret,
        )
        close_api = True

    try:
        await lkapi.room.create_room(
            CreateRoomRequest(
                name=room_name,
                empty_timeout=300,
            )
        )
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=app_settings.agent_dispatch_name,
                room=room_name,
                metadata=json.dumps({"phone_number": phone_number}),
            )
        )
        return room_name, dispatch.id
    finally:
        if close_api:
            await lkapi.aclose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Dispatch outbound voice agent call to a phone number."
    )
    parser.add_argument(
        "phone_number",
        help="Phone number in E.164 format, e.g. +48XXXXXXXXX",
    )
    args = parser.parse_args()

    phone_number = args.phone_number.strip()
    logger.info("Initiating call dispatch to %s...", mask_phone_number(phone_number))

    try:
        room_name, dispatch_id = asyncio.run(
            dispatch_phone_call(phone_number, settings)
        )
        print(f"Room name: {room_name}")
        print(f"Dispatch ID: {dispatch_id}")
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to dispatch call: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
