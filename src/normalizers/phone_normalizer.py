"""
Phone number normalizer — converts messy phone strings to E.164 format.

Uses Google's libphonenumber (phonenumbers library) for robust parsing.
Falls back to US/IN region if no country hint is available.
Returns None for unparseable inputs — never invents data.
"""

from __future__ import annotations

import logging
from typing import Optional

import phonenumbers

logger = logging.getLogger(__name__)

# Default regions to try when no country hint is available
_DEFAULT_REGIONS = ["IN", "US", "GB"]


def normalize_phone(
    raw: Optional[str],
    country_hint: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize a raw phone string to E.164 format.

    Args:
        raw: The raw phone string (e.g. "(555) 123-4567", "+91-98765-43210")
        country_hint: ISO-3166 alpha-2 country code to help parsing

    Returns:
        E.164 formatted string (e.g. "+15551234567") or None if unparseable
    """
    if not raw or not isinstance(raw, str):
        return None

    # Strip common noise
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("n/a", "na", "none", "null", "-", "0", "0000000000"):
        return None

    # Build the list of regions to try
    regions = []
    if country_hint:
        regions.append(country_hint.upper())
    regions.extend(r for r in _DEFAULT_REGIONS if r not in regions)

    for region in regions:
        try:
            parsed = phonenumbers.parse(cleaned, region)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164
                )
        except phonenumbers.NumberParseException:
            continue

    # Last resort: try parsing without any region (works if number has +XX prefix)
    try:
        parsed = phonenumbers.parse(cleaned, None)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except phonenumbers.NumberParseException:
        pass

    logger.debug("Could not parse phone number: %r", raw)
    return None


def normalize_phones(
    phones: list[str],
    country_hint: Optional[str] = None,
) -> list[str]:
    """
    Normalize a list of phone strings, dropping unparseable ones.

    Returns deduplicated list of E.164 phone numbers.
    """
    seen: set[str] = set()
    result: list[str] = []
    for p in phones:
        normalized = normalize_phone(p, country_hint)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
