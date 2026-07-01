"""
Location normalizer — parses free-text locations into structured
{city, state, country} with ISO-3166 alpha-2 country codes.

Uses pycountry for country lookup plus a manual map of common abbreviations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import pycountry

logger = logging.getLogger(__name__)

# Common abbreviations and aliases → ISO-3166 alpha-2
_COUNTRY_ALIASES: dict[str, str] = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "america": "US",
    "uk": "GB", "united kingdom": "GB", "great britain": "GB", "england": "GB",
    "in": "IN", "india": "IN",
    "ca": "CA", "canada": "CA",
    "au": "AU", "australia": "AU",
    "de": "DE", "germany": "DE",
    "fr": "FR", "france": "FR",
    "sg": "SG", "singapore": "SG",
    "jp": "JP", "japan": "JP",
    "cn": "CN", "china": "CN",
    "br": "BR", "brazil": "BR",
    "il": "IL", "israel": "IL",
    "nl": "NL", "netherlands": "NL",
    "se": "SE", "sweden": "SE",
    "ie": "IE", "ireland": "IE",
    "nz": "NZ", "new zealand": "NZ",
    "kr": "KR", "south korea": "KR",
}

# US state abbreviations (for detecting US locations)
_US_STATES: set[str] = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

# Indian states / UTs (common abbreviations)
_INDIAN_STATES: set[str] = {
    "AP", "AR", "AS", "BR", "CT", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB", "DL",
}


def _lookup_country(raw: str) -> Optional[str]:
    """Try to resolve a raw string to an ISO-3166 alpha-2 country code."""
    normalized = raw.strip().lower()

    # 1. Check our alias table
    if normalized in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[normalized]

    # 2. Already a valid alpha-2?
    if len(normalized) == 2:
        upper = normalized.upper()
        try:
            pycountry.countries.get(alpha_2=upper)
            return upper
        except (KeyError, LookupError):
            pass

    # 3. Try pycountry fuzzy search
    try:
        results = pycountry.countries.search_fuzzy(raw.strip())
        if results:
            return results[0].alpha_2
    except LookupError:
        pass

    return None


def normalize_location(raw: Optional[str]) -> Optional[dict[str, Optional[str]]]:
    """
    Parse a free-text location into {city, state, country}.

    Examples:
        "San Francisco, CA, USA"  → {city: "San Francisco", state: "CA", country: "US"}
        "Bangalore, India"        → {city: "Bangalore", state: null, country: "IN"}
        "New York"                → {city: "New York", state: null, country: null}

    Returns None if input is empty/garbage.
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("n/a", "na", "none", "null", "-", "remote"):
        return None

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]

    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    if len(parts) == 1:
        # Could be just a city, a country, or a state
        country_code = _lookup_country(parts[0])
        if country_code:
            country = country_code
        else:
            city = parts[0]

    elif len(parts) == 2:
        # "City, Country" or "City, State"
        upper_part1 = parts[1].upper()
        if upper_part1 in _INDIAN_STATES:
            city = parts[0]
            state = upper_part1
            country = "IN"
        elif upper_part1 in _US_STATES:
            city = parts[0]
            state = upper_part1
            country = "US"
        else:
            country_code = _lookup_country(parts[1])
            if country_code:
                city = parts[0]
                country = country_code
            else:
                city = parts[0]
                state = parts[1]

    elif len(parts) >= 3:
        # "City, State, Country"
        city = parts[0]
        state = parts[1].strip()
        country_code = _lookup_country(parts[2])
        country = country_code if country_code else parts[2].strip()

    return {
        "city": city,
        "state": state,
        "country": country,
    }


def normalize_location_to_string(location_dict: Optional[dict]) -> Optional[str]:
    """
    Format a location dict back to a readable string.

    Useful for display: {city: "SF", state: "CA", country: "US"} → "SF, CA, US"
    """
    if not location_dict:
        return None
    parts = [
        location_dict.get("city"),
        location_dict.get("state"),
        location_dict.get("country"),
    ]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else None
