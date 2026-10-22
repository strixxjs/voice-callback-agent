"""Telephony helpers for phone validation, masking, and metadata parsing."""

from __future__ import annotations

import json
import re

E164_REGEX = re.compile(r"^\+[1-9]\d{1,14}$")


def validate_e164(phone: str) -> bool:
    """Validates if a phone number complies with the E.164 standard.

    Must start with '+' followed by country code (1-9) and digits,
    up to 15 digits total.
    """
    if not isinstance(phone, str):
        return False
    return bool(E164_REGEX.match(phone.strip()))


def mask_phone_number(phone: str) -> str:
    """Masks a phone number, preserving only the last 3 digits for PII protection.

    Examples:
        '+48123456789' -> '*********789'
        '123'          -> '***'
        ''             -> ''
    """
    if not phone:
        return ""
    phone = phone.strip()
    if len(phone) <= 3:
        return "*" * len(phone)
    return f"{'*' * (len(phone) - 3)}{phone[-3:]}"


def extract_job_phone_number(job_metadata: str | None) -> tuple[bool, str | None]:
    """Extracts phone_number from job metadata JSON.

    Returns:
        (has_phone_key, phone_value)
        - has_phone_key: True if 'phone_number' key was present in metadata dictionary.
        - phone_value: The string value if present, else None.
    """
    if not job_metadata:
        return False, None
    try:
        data = json.loads(job_metadata)
        if isinstance(data, dict) and "phone_number" in data:
            val = data["phone_number"]
            return True, str(val).strip() if val is not None else ""
    except (json.JSONDecodeError, TypeError):
        pass
    return False, None
