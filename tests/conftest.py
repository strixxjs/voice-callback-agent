"""Pytest fixtures for isolating test execution from local .env and credentials."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import agent.extraction
import agent.reporting
import agent.worker
import app.api.internal
import app.config
import tests.test_agent
from app.config import Settings
from app.db import get_db
from app.main import app as fastapi_app
from app.models import Base


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Provides isolated test settings without reading local .env, matching CI."""
    clean_settings = Settings(_env_file=None)
    monkeypatch.setattr(app.config, "settings", clean_settings)
    monkeypatch.setattr(agent.worker, "settings", clean_settings)
    monkeypatch.setattr(agent.extraction, "settings", clean_settings)
    monkeypatch.setattr(agent.reporting, "settings", clean_settings)
    monkeypatch.setattr(app.api.internal, "settings", clean_settings)
    monkeypatch.setattr(tests.test_agent, "settings", clean_settings)
    try:
        import scripts.call_me

        monkeypatch.setattr(scripts.call_me, "settings", clean_settings)
    except ImportError:
        pass
    return clean_settings


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory async SQLite session with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def async_client(
    db_session: AsyncSession,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async test client wired to test DB session."""

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    fastapi_app.dependency_overrides.clear()
