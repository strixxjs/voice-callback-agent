"""SQLAlchemy declarative models for leads, calls, turns, and events."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

# Portable JSON type supporting PostgreSQL JSONB and SQLite JSON
PortableJSON = sa.JSON().with_variant(JSONB, "postgresql")
# Portable BigInteger supporting PostgreSQL BIGSERIAL and SQLite AUTOINCREMENT
PortableBigInt = sa.BigInteger().with_variant(sa.Integer, "sqlite")


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""


class Lead(Base):
    """Lead intake record from website/demo form."""

    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    phone: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    calls: Mapped[list[Call]] = relationship("Call", back_populates="lead")


class Call(Base):
    """Voice agent call session lifecycle and results."""

    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    lead_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("leads.id", ondelete="SET NULL"),
        nullable=True,
    )
    room_name: Mapped[str] = mapped_column(
        sa.String(128),
        unique=True,
        index=True,
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(
        sa.String(16),
        nullable=False,
        default="browser",
    )
    status: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        index=True,
        default="dialing",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    duration_seconds: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    extracted_answers: Mapped[dict[str, Any] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )
    extracted_missing: Mapped[list[str] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )
    extraction_status: Mapped[str | None] = mapped_column(
        sa.String(32),
        nullable=True,
    )
    latency_summary: Mapped[dict[str, Any] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )
    call_config: Mapped[dict[str, Any] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    lead: Mapped[Lead | None] = relationship("Lead", back_populates="calls")
    turns: Mapped[list[CallTurn]] = relationship(
        "CallTurn",
        back_populates="call",
        cascade="all, delete-orphan",
        order_by="CallTurn.turn_index",
    )
    events: Mapped[list[CallEvent]] = relationship(
        "CallEvent",
        back_populates="call",
        cascade="all, delete-orphan",
        order_by="CallEvent.timestamp",
    )


class CallTurn(Base):
    """Individual conversational speech turn within a call."""

    __tablename__ = "call_turns"
    __table_args__ = (
        sa.UniqueConstraint("call_id", "turn_index", name="uq_call_turns_call_turn"),
    )

    id: Mapped[int] = mapped_column(
        PortableBigInt, primary_key=True, autoincrement=True
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    role: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )

    call: Mapped[Call] = relationship("Call", back_populates="turns")


class CallEvent(Base):
    """Operational event or log entry during call execution."""

    __tablename__ = "call_events"

    id: Mapped[int] = mapped_column(
        PortableBigInt, primary_key=True, autoincrement=True
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    message: Mapped[str] = mapped_column(sa.Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(
        PortableJSON,
        nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
    )

    call: Mapped[Call] = relationship("Call", back_populates="events")
