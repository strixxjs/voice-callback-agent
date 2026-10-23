"""Thin LiveKit server-API wrapper used by the backend (rooms, dispatch, tokens).

Exposed to routes via the ``get_livekit_rooms`` FastAPI dependency so tests can
replace it with ``app.dependency_overrides`` (no network in tests).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from livekit.api import (
    AccessToken,
    CreateAgentDispatchRequest,
    CreateRoomRequest,
    LiveKitAPI,
    VideoGrants,
)

from app.config import settings

# Seconds an empty room stays alive, so the browser user has time to join.
ROOM_EMPTY_TIMEOUT_SECONDS = 300
PARTICIPANT_TOKEN_TTL = timedelta(hours=1)


class LiveKitNotConfiguredError(RuntimeError):
    """Raised when LiveKit url/key/secret are missing."""


class LiveKitRooms:
    """Creates rooms and dispatches and signs participant tokens.

    The underlying ``LiveKitAPI`` client is created lazily on first use and
    must be released with ``aclose()``.
    """

    def __init__(
        self,
        *,
        url: str | None,
        api_key: str | None,
        api_secret: str | None,
        agent_name: str = "",
        api_factory: Callable[..., Any] = LiveKitAPI,
        empty_timeout: int = ROOM_EMPTY_TIMEOUT_SECONDS,
        token_ttl: timedelta = PARTICIPANT_TOKEN_TTL,
    ) -> None:
        self.url = url
        self.agent_name = agent_name
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_factory = api_factory
        self._empty_timeout = empty_timeout
        self._token_ttl = token_ttl
        self._api: Any | None = None

    def _ensure_configured(self) -> tuple[str, str]:
        if not self._api_key or not self._api_secret:
            raise LiveKitNotConfiguredError("LiveKit credentials are not configured")
        return self._api_key, self._api_secret

    def _client(self) -> Any:
        if self._api is None:
            api_key, api_secret = self._ensure_configured()
            if not self.url:
                raise LiveKitNotConfiguredError("LiveKit URL is not configured")
            # Credentials are passed explicitly: pydantic-settings does not
            # export .env values into os.environ.
            self._api = self._api_factory(
                url=self.url,
                api_key=api_key,
                api_secret=api_secret,
            )
        return self._api

    async def create_room(self, name: str, metadata: dict[str, str]) -> None:
        """Creates a room with JSON metadata."""
        await self._client().room.create_room(
            CreateRoomRequest(
                name=name,
                metadata=json.dumps(metadata),
                empty_timeout=self._empty_timeout,
            )
        )

    async def create_dispatch(self, room: str, metadata: dict[str, str]) -> str:
        """Explicitly dispatches the named agent into a room; returns dispatch id."""
        if not self.agent_name:
            raise LiveKitNotConfiguredError("Agent dispatch name is not configured")
        dispatch = await self._client().agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=self.agent_name,
                room=room,
                metadata=json.dumps(metadata),
            )
        )
        return str(dispatch.id)

    def make_participant_token(self, room: str, identity: str, name: str) -> str:
        """Signs a join-only token for a single room."""
        api_key, api_secret = self._ensure_configured()
        return (
            AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(VideoGrants(room_join=True, room=room))
            .with_ttl(self._token_ttl)
            .to_jwt()
        )

    async def aclose(self) -> None:
        """Closes the underlying LiveKitAPI client if it was created."""
        api, self._api = self._api, None
        if api is not None:
            await api.aclose()


def get_livekit_rooms() -> LiveKitRooms:
    """FastAPI dependency building the wrapper from settings."""
    return LiveKitRooms(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        agent_name=settings.agent_dispatch_name,
    )
