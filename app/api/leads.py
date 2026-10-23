"""Public lead intake: POST /leads starts a browser-mode call via LiveKit."""

from __future__ import annotations

import logging
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.livekit_rooms import LiveKitRooms, get_livekit_rooms
from app.models import Call
from app.schemas import LeadCreateRequest, LeadCreateResponse
from app.services import calls as call_service

logger = logging.getLogger("app.api.leads")

router = APIRouter(tags=["leads"])
DbSession = Annotated[AsyncSession, Depends(get_db)]
LiveKitDep = Annotated[LiveKitRooms, Depends(get_livekit_rooms)]

PHONE_MODE_DISABLED_DETAIL = "Phone mode is disabled until Phone go-live"
LIVEKIT_FAILED_DETAIL = "Failed to start LiveKit session"


async def _fail_call(
    db: AsyncSession, call: Call, stage: str, exc: Exception
) -> NoReturn:
    """Marks the call failed with a short, secret-free error event; raises 502."""
    error_type = type(exc).__name__
    logger.warning(
        "LiveKit %s failed lead_id=%s call_id=%s error_type=%s",
        stage,
        call.lead_id,
        call.id,
        error_type,
    )
    await call_service.update_call_status(db, call, call_service.CALL_STATUS_FAILED)
    await call_service.append_call_event(
        db,
        call,
        "error",
        f"LiveKit {stage} failed ({error_type})",
        payload={"stage": stage, "error_type": error_type},
    )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=LIVEKIT_FAILED_DETAIL,
    )


@router.post(
    "/leads",
    status_code=status.HTTP_201_CREATED,
    response_model=LeadCreateResponse,
)
async def create_lead(
    body: LeadCreateRequest,
    db: DbSession,
    livekit: LiveKitDep,
) -> LeadCreateResponse:
    """Creates a lead + call and a LiveKit room the lead joins from the browser."""
    if body.mode == "phone":
        # Nothing is created: no rows, no LiveKit calls, no SIP code touched.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=PHONE_MODE_DISABLED_DETAIL,
        )

    lead, call = await call_service.create_lead_and_call(
        db, name=body.name, phone=body.phone, mode=body.mode
    )
    await call_service.append_call_event(
        db,
        call,
        "lead_created",
        "Lead created from browser demo",
        payload={"lead_id": str(lead.id), "mode": body.mode},
    )

    room = call.room_name
    # Never put the phone number into room/dispatch metadata.
    metadata = {"call_id": str(call.id), "lead_id": str(lead.id), "mode": body.mode}

    try:
        try:
            await livekit.create_room(room, metadata)
        except Exception as exc:
            await _fail_call(db, call, "create_room", exc)
        await call_service.append_call_event(
            db, call, "room_created", f"Room {room} created", payload={"room": room}
        )

        agent_name = settings.agent_dispatch_name
        if agent_name:
            try:
                dispatch_id = await livekit.create_dispatch(room, metadata)
            except Exception as exc:
                await _fail_call(db, call, "create_dispatch", exc)
            await call_service.append_call_event(
                db,
                call,
                "dispatched",
                f"Agent '{agent_name}' dispatched",
                payload={"agent_name": agent_name, "dispatch_id": dispatch_id},
            )

        try:
            token = livekit.make_participant_token(
                room, identity=f"lead-{lead.id}", name=lead.name
            )
        except Exception as exc:
            await _fail_call(db, call, "create_token", exc)
    finally:
        await livekit.aclose()

    return LeadCreateResponse(
        lead_id=lead.id,
        call_id=call.id,
        room=room,
        mode=body.mode,
        status=call.status,
        livekit_url=settings.livekit_url,
        token=token,
    )
