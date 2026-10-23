"""Post-call transcript extraction module using Gemini."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from google.genai import types
from livekit.agents import llm
from pydantic import BaseModel, Field, create_model

from agent.telephony import mask_phone_number
from app.config import settings

logger = logging.getLogger("agent.extraction")


class ExtractionResult(BaseModel):
    """Structured extraction result containing extracted answers and missing fields."""

    answers: dict[str, str | None] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


def build_transcript_text(
    chat_ctx_or_items: llm.ChatContext | list[dict[str, str]],
) -> str:
    """Builds a plain-text conversational transcript from session chat history.

    Includes only assistant ('Agent:') and user ('User:') conversational turns,
    skipping system prompts, empty turns, and tool execution messages.
    """
    lines: list[str] = []

    if isinstance(chat_ctx_or_items, llm.ChatContext):
        for item in chat_ctx_or_items.items:
            if isinstance(item, llm.ChatMessage):
                role = str(
                    item.role.value if hasattr(item.role, "value") else item.role
                ).lower()
                if role == "system":
                    continue

                text_parts: list[str] = []
                if isinstance(item.content, list):
                    for part in item.content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif hasattr(part, "text") and part.text:
                            text_parts.append(str(part.text))
                elif isinstance(item.content, str):
                    text_parts.append(item.content)

                text = " ".join(text_parts).strip()
                if not text:
                    continue

                if role in ("assistant", "agent"):
                    lines.append(f"Agent: {text}")
                elif role in ("user", "caller"):
                    lines.append(f"User: {text}")
    elif isinstance(chat_ctx_or_items, list):
        for item in chat_ctx_or_items:
            role = str(item.get("role", "")).lower()
            if role == "system":
                continue
            text = str(item.get("content", "")).strip()
            if not text:
                continue
            if role in ("assistant", "agent"):
                lines.append(f"Agent: {text}")
            elif role in ("user", "caller"):
                lines.append(f"User: {text}")

    return "\n".join(lines)


def _build_extraction_prompt(transcript: str, question_keys: list[str]) -> str:
    """Creates the LLM extraction prompt for structured JSON answer extraction."""
    keys_formatted = "\n".join(f"- {k}" for k in question_keys)
    return (
        "You are an expert post-call data extraction assistant. "
        "Extract structured answers from the phone call transcript between an AI "
        "voice assistant ('Agent') and a lead ('User').\n\n"
        f"Target fields to extract:\n{keys_formatted}\n\n"
        "Extraction Rules:\n"
        "1. For each target field, extract the specific answer given by the user.\n"
        "2. If a field was not asked, not answered, declined, or the user did not "
        "know, set its value to null.\n"
        "3. An answer provided earlier in the call or out of order must still "
        "be extracted.\n"
        "4. CRITICAL for 'preferred_call_time': preserve the exact raw wording "
        "spoken by the user (e.g. 'tomorrow afternoon', 'after 5pm', "
        "'завтра о другій', 'десь о 14:00'). "
        "Do NOT convert relative times to absolute dates or ISO timestamps.\n"
        "5. 'missing': an array containing all field names that were not answered "
        "(where value is null).\n"
        "6. Return ONLY a valid JSON object matching the schema:\n"
        "{\n"
        '  "answers": {\n'
        '    "<field_name>": "<extracted string value or null>"\n'
        "  },\n"
        '  "missing": ["<unanswered_field_name>", ...]\n'
        "}\n\n"
        f"Call Transcript:\n{transcript}"
    )


def _create_response_schema(question_keys: list[str]) -> Any:
    """Builds a Gemini response schema using LiveKit Google plugin's transformer."""
    try:
        from livekit.plugins.google.utils import to_response_format

        field_defs = dict.fromkeys(question_keys, (str | None, None))
        AnswersModel = create_model("AnswersModel", **field_defs)

        class StructuredExtraction(BaseModel):
            answers: AnswersModel  # type: ignore
            missing: list[str]

        return to_response_format(StructuredExtraction)
    except Exception as e:
        logger.debug("Failed to build structured response_schema: %s", e)
        return None


def parse_extraction_json(
    raw_text: str,
    question_keys: list[str],
) -> ExtractionResult:
    """Parses and sanitizes LLM JSON output into an ExtractionResult model."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    data = json.loads(clean)
    if not isinstance(data, dict):
        raise ValueError("Extraction response root must be a JSON object")

    raw_answers = data.get("answers")
    if not isinstance(raw_answers, dict):
        raise ValueError("Field 'answers' must be a JSON object")

    normalized_answers: dict[str, str | None] = {}
    missing_fields: list[str] = []

    for k in question_keys:
        val = raw_answers.get(k)
        if val is None or (isinstance(val, str) and not val.strip()):
            normalized_answers[k] = None
            missing_fields.append(k)
        else:
            normalized_answers[k] = str(val).strip()

    return ExtractionResult(answers=normalized_answers, missing=missing_fields)


def format_short_error(e: Exception) -> str:
    """Formats concise error: <ExceptionClass>: <code> <first 200 chars>."""
    cls_name = type(e).__name__
    code = getattr(e, "code", None)
    if code is None:
        code = getattr(e, "status_code", None)

    raw_msg = str(e).strip()
    safe_msg = re.sub(
        r"\+?[1-9]\d{6,14}",
        lambda m: mask_phone_number(m.group(0)),
        raw_msg,
    )[:200]

    if code is not None:
        return f"{cls_name}: {code} {safe_msg}".strip()
    return f"{cls_name}: {safe_msg}".strip()


async def extract_answers(
    transcript: str,
    question_keys: list[str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    request_timeout: float = 15.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Extracts structured answers from a transcript using Gemini.

    Guarantees:
    - Never crashes or raises unhandled exceptions.
    - Timeout is at least 10 seconds (Gemini requirement).
    - Phone numbers in logs/errors are masked.
    - Returns either a successful extraction dict with 'answers' and 'missing',
      or a failure dict {'status': 'extraction_failed', 'error': ...}.
    """
    if not question_keys:
        return {"status": "ok", "answers": {}, "missing": []}

    # If transcript is empty, return empty answers without LLM call
    if not transcript.strip():
        logger.info("Empty transcript provided; skipping LLM extraction.")
        return {
            "status": "ok",
            "answers": dict.fromkeys(question_keys),
            "missing": list(question_keys),
        }

    # Ensure timeout is >= 10s per Gemini requirement
    effective_timeout = max(10.0, request_timeout)
    target_model = model or settings.extraction_model or "gemini-3.5-flash"
    prompt = _build_extraction_prompt(transcript, question_keys)

    try:
        if client is None:
            from google.genai import Client

            effective_api_key = (
                api_key or settings.google_api_key or os.environ.get("GOOGLE_API_KEY")
            )
            client = Client(api_key=effective_api_key)

        schema = _create_response_schema(question_keys)
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
        }
        if schema:
            config_kwargs["response_schema"] = schema

        config = types.GenerateContentConfig(**config_kwargs)

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            ),
            timeout=effective_timeout,
        )

        raw_text = getattr(response, "text", "") or ""
        result = parse_extraction_json(raw_text, question_keys)
        logger.info(
            "Extraction successful for model '%s': answers=%s, missing=%s",
            target_model,
            result.answers,
            result.missing,
        )
        return {
            "status": "ok",
            "answers": result.answers,
            "missing": result.missing,
        }

    except TimeoutError as e:
        error_msg = format_short_error(e)
        if not str(e).strip():
            error_msg = (
                f"TimeoutError: Extraction timed out after {effective_timeout:.1f}s"
            )
        logger.error("Extraction failed: %s", error_msg)
        return {
            "status": "extraction_failed",
            "answers": dict.fromkeys(question_keys),
            "missing": list(question_keys),
            "error": error_msg,
        }
    except Exception as e:
        safe_error = format_short_error(e)
        logger.error("Extraction failed: %s", safe_error)
        return {
            "status": "extraction_failed",
            "answers": dict.fromkeys(question_keys),
            "missing": list(question_keys),
            "error": safe_error,
        }


def save_call_summary(
    room_name: str,
    summary: dict[str, Any],
    base_dir: Path | str = "logs/calls",
) -> Path:
    """Writes the final call summary JSON to logs/calls/<room>.json."""
    dir_path = Path(base_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{room_name}.json"
    file_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path
