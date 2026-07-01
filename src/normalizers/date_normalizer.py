"""
Date normalizer — converts messy date strings to YYYY-MM format.

Handles a wide range of formats: "Jan 2020", "2020-01-15", "January 2020",
"01/2020", "2020", etc.  Returns None for unparseable dates — never invents.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from dateutil import parser as date_parser

logger = logging.getLogger(__name__)

# Pre-compiled patterns for quick extraction
_YEAR_ONLY = re.compile(r"^\d{4}$")
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_MONTH_YEAR_SLASH = re.compile(r"^(\d{1,2})/(\d{4})$")


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a raw date string to YYYY-MM format.

    Args:
        raw: Date string in any reasonable format

    Returns:
        "YYYY-MM" string or None if unparseable
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("n/a", "na", "none", "null", "-", "present", "current"):
        return None

    # --- fast-path patterns ---

    # "2020" → "2020-01" (year only — assume January)
    if _YEAR_ONLY.match(cleaned):
        return f"{cleaned}-01"

    # "2020-01" or "2020-1" — already in target format
    m = _YEAR_MONTH.match(cleaned)
    if m:
        year, month = m.group(1), m.group(2).zfill(2)
        return f"{year}-{month}"

    # "01/2020" — month/year with slash
    m = _MONTH_YEAR_SLASH.match(cleaned)
    if m:
        month, year = m.group(1).zfill(2), m.group(2)
        return f"{year}-{month}"

    # --- general parsing via dateutil ---
    try:
        dt = date_parser.parse(cleaned, fuzzy=True)
        return dt.strftime("%Y-%m")
    except (ValueError, OverflowError, TypeError):
        pass

    logger.debug("Could not parse date: %r", raw)
    return None


def normalize_dates_in_experience(
    experience: list[dict],
) -> list[dict]:
    """
    Normalize start/end dates in a list of experience dicts.

    Modifies dicts in-place and returns the list.
    """
    for entry in experience:
        if "start" in entry:
            entry["start"] = normalize_date(entry.get("start"))
        if "end" in entry:
            entry["end"] = normalize_date(entry.get("end"))
    return experience
