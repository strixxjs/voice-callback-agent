"""Database configuration and async session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

async_session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with async_session_factory() as session:
        yield session
