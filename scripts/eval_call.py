"""Script to evaluate call quality metrics and generate markdown row for test log."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when script is executed directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import selectinload  # noqa: E402

from app.db import async_session_factory  # noqa: E402
from app.models import Call  # noqa: E402

logger = logging.getLogger("scripts.eval_call")


def count_split_turns(turns: list[Any]) -> int:
    """Calculates number of split user turns (consecutive turns by user role).

    Expects turns ordered by turn_index from SQL.
    """
    split_count = 0
    prev_role: str | None = None
    for turn in turns:
        role = str(getattr(turn, "role", "")).lower()
        if role == "user" and prev_role == "user":
            split_count += 1
        prev_role = role
    return split_count


def count_error_events(events: list[Any]) -> int:
    """Counts error events in call events."""
    count = 0
    for ev in events:
        ev_type = str(getattr(ev, "event_type", "")).lower()
        msg = str(getattr(ev, "message", "")).lower()
        if "error" in ev_type or "error" in msg:
            count += 1
    return count


def format_extraction_status(call: Call) -> str:
    """Formats answered vs total extraction status (e.g. 4/4 or 3/4).

    answered = number of non-empty (not None, not "") values in extracted_answers.
    total = answered + len(extracted_missing) (no double counting if a key is in both).
    """
    raw_answers = call.extracted_answers or {}
    raw_missing = call.extracted_missing or []

    answered_keys: set[str] = set()
    if isinstance(raw_answers, dict):
        for k, v in raw_answers.items():
            if v is not None and str(v).strip() != "":
                answered_keys.add(k)

    missing_keys: set[str] = set()
    if isinstance(raw_missing, (list, tuple, set)):
        for item in raw_missing:
            if isinstance(item, str) and item.strip():
                k = item.strip()
                if k not in answered_keys:
                    missing_keys.add(k)

    answered_count = len(answered_keys)
    total = answered_count + len(missing_keys)
    if total > 0:
        return f"{answered_count}/{total}"
    if call.extraction_status:
        return str(call.extraction_status)
    return "-"


def format_eval_row(
    call: Call,
    run_label: str | None = None,
    variant: str | None = None,
) -> str:
    """Generates one markdown row conforming to docs/m3-test-log.md columns:

    Run | Date | Variant (.env) | Turns | p50 meas. | max meas. | p50 comp. |
    Split turns | Lost words | STT errors | Extraction 4/4 | Notes
    """
    date_val = call.created_at or datetime.now(UTC)
    date_str = date_val.strftime("%Y-%m-%d")

    # Run label is first 8 chars of call id (or explicitly passed run_label)
    run_str = run_label if run_label else (str(call.id)[:8] if call.id else "-")
    variant_str = variant or "-"

    lat_sum = call.latency_summary or {}
    turns_count = (
        lat_sum.get("turns", 0) if isinstance(lat_sum.get("turns"), int) else 0
    )

    p50_m = lat_sum.get("p50_measured", lat_sum.get("p50"))
    max_m = lat_sum.get("max_measured", lat_sum.get("max"))
    p50_c = lat_sum.get("p50_computed")

    # If turns == 0 or measured list was empty, print "-" instead of 0.000s
    has_measured = turns_count > 0 and isinstance(p50_m, (int, float)) and p50_m > 0
    p50_meas_str = f"{p50_m:.3f}s" if has_measured else "-"
    max_meas_str = (
        f"{max_m:.3f}s" if has_measured and isinstance(max_m, (int, float)) else "-"
    )

    has_computed = turns_count > 0 and isinstance(p50_c, (int, float)) and p50_c > 0
    p50_comp_str = f"{p50_c:.3f}s" if has_computed else "-"

    turns_val = lat_sum.get("turns")
    if turns_val is None or not isinstance(turns_val, int):
        agent_turns = [
            t for t in (call.turns or []) if getattr(t, "role", "") == "agent"
        ]
        turns_val = len(agent_turns) if agent_turns else len(call.turns or [])
    turns_str = str(turns_val)

    split_turns = count_split_turns(call.turns or [])
    stt_errors_col = "-"  # Manual evaluation column
    lost_words = "-"  # Manual evaluation column
    extraction_str = format_extraction_status(call)

    ended_by = None
    for ev in call.events or []:
        if getattr(ev, "event_type", "") == "call_ended" and isinstance(
            getattr(ev, "payload", None), dict
        ):
            ended_by = ev.payload.get("ended_by")
            if ended_by:
                break
    ended_by_str = ended_by or getattr(call, "status", None) or "unknown"
    err_count = count_error_events(call.events or [])
    notes_str = f"errors={err_count}, ended_by={ended_by_str}"

    return (
        f"| {run_str} | {date_str} | {variant_str} | {turns_str} | {p50_meas_str} | "
        f"{max_meas_str} | {p50_comp_str} | {split_turns} | {lost_words} | "
        f"{stt_errors_col} | {extraction_str} | {notes_str} |"
    )


async def evaluate_call(target: str, variant: str | None = None) -> str:
    """Loads call by UUID or room_name from database and formats evaluation row."""
    target_uuid: uuid.UUID | None = None
    try:
        target_uuid = uuid.UUID(target)
    except (ValueError, TypeError):
        pass

    async with async_session_factory() as session:
        # call.turns relationship has order_by="CallTurn.turn_index"
        stmt = select(Call).options(
            selectinload(Call.turns),
            selectinload(Call.events),
        )
        if target_uuid:
            stmt = stmt.where(Call.id == target_uuid)
        else:
            stmt = stmt.where(Call.room_name == target)

        result = await session.execute(stmt)
        call = result.scalar_one_or_none()

    if call is None:
        raise ValueError(f"Call not found for identifier: {target}")

    return format_eval_row(call, variant=variant)


def main() -> None:
    """CLI entry point for evaluating a call."""
    parser = argparse.ArgumentParser(
        description="Print markdown evaluation row for a completed call."
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help="Call ID (UUID) or room name",
    )
    parser.add_argument(
        "--call-id",
        dest="call_id",
        help="Call UUID to evaluate",
    )
    parser.add_argument(
        "--room",
        dest="room",
        help="Room name to evaluate",
    )
    parser.add_argument(
        "--variant",
        dest="variant",
        default=None,
        help="Variant description (printed in Variant column, '-' if omitted)",
    )
    args = parser.parse_args()

    target = args.call_id or args.room or args.identifier
    if not target:
        parser.error("Must provide call_id or room identifier.")

    try:
        row = asyncio.run(evaluate_call(target, variant=args.variant))
        print(row)
    except Exception as e:
        print(
            f"Error evaluating call '{target}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
