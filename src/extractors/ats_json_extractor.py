"""
ATS JSON Extractor — parses applicant tracking system JSON records.

ATS systems use their own field naming conventions (e.g. "applicant_name"
instead of "full_name").  This extractor maps those to our canonical fields.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.extractors.base import BaseExtractor
from src.models.canonical import (
    ExtractionMethod,
    ProvenanceEntry,
    RawRecord,
    SourceType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ATS field name mapping
# ---------------------------------------------------------------------------

_ATS_FIELD_MAP: dict[str, str] = {
    # Name
    "applicant_name": "full_name",
    "candidate_name": "full_name",
    "name": "full_name",
    "full_name": "full_name",
    "first_name": "_first_name",
    "last_name": "_last_name",

    # Contact
    "contact_email": "emails",
    "email": "emails",
    "emails": "emails",
    "email_addresses": "emails",
    "mobile": "phones",
    "phone": "phones",
    "phones": "phones",
    "phone_number": "phones",
    "contact_phone": "phones",
    "mobile_number": "phones",

    # Location
    "location": "location_raw",
    "address": "location_raw",
    "city": "_city",
    "state": "_state",
    "country": "_country",
    "region": "_state",

    # Professional
    "headline": "headline",
    "title": "headline",
    "current_title": "headline",
    "job_title": "headline",
    "position": "headline",
    "summary": "headline",

    # Skills
    "skills": "skills_raw",
    "skill_set": "skills_raw",
    "technologies": "skills_raw",
    "competencies": "skills_raw",

    # Experience
    "experience": "experience_raw",
    "work_experience": "experience_raw",
    "employment_history": "experience_raw",
    "work_history": "experience_raw",
    "positions": "experience_raw",

    # Education
    "education": "education_raw",
    "education_history": "education_raw",
    "degrees": "education_raw",
    "qualifications": "education_raw",

    # Links
    "linkedin_url": "_linkedin",
    "linkedin": "_linkedin",
    "github_url": "_github",
    "github": "_github",
    "portfolio_url": "_portfolio",
    "portfolio": "_portfolio",
    "website": "_portfolio",

    # Experience
    "years_of_experience": "years_experience",
    "years_experience": "years_experience",
    "yoe": "years_experience",
    "total_experience": "years_experience",
}


class ATSJSONExtractor(BaseExtractor):
    """Extracts candidate records from ATS JSON data."""

    source_type = SourceType.ATS_JSON

    def extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        path = Path(source_path)
        if not path.exists():
            logger.warning("ATS JSON file not found: %s", source_path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to parse ATS JSON %s: %s", source_path, e)
            return []

        # Handle both single record and array of records
        if isinstance(data, dict):
            # Check if it's wrapped in a top-level key like "candidates" or "records"
            for key in ("candidates", "records", "applicants", "data", "results"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data]  # Single record

        if not isinstance(data, list):
            logger.warning("ATS JSON has unexpected structure in %s", source_path)
            return []

        records = []
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                logger.warning("Skipping non-dict entry at index %d in %s", idx, source_path)
                continue
            try:
                record = self._entry_to_record(entry, source_path, idx)
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "Skipping ATS entry %d in %s: %s", idx, source_path, e
                )

        return records

    def _entry_to_record(
        self,
        entry: dict[str, Any],
        source_path: str,
        idx: int,
    ) -> RawRecord | None:
        """Convert a single ATS JSON entry to a RawRecord."""
        data: dict[str, Any] = {
            "emails": [],
            "phones": [],
            "skills_raw": [],
            "links": {},
            "experience_raw": [],
            "education_raw": [],
            "extra": {},
        }
        provenance_entries: list[ProvenanceEntry] = []

        first_name = None
        last_name = None
        city = None
        state = None
        country = None

        for key, value in entry.items():
            if value is None:
                continue

            canonical = _ATS_FIELD_MAP.get(key.lower().strip())

            if canonical is None:
                # Unknown field — stash in extra
                data["extra"][key] = value
                continue

            if canonical == "full_name":
                data["full_name"] = str(value).strip()
                provenance_entries.append(ProvenanceEntry(
                    field="full_name",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.9,
                    raw_value=str(value),
                ))

            elif canonical == "_first_name":
                first_name = str(value).strip()

            elif canonical == "_last_name":
                last_name = str(value).strip()

            elif canonical == "emails":
                emails = self._to_list(value)
                data["emails"].extend(emails)
                for e in emails:
                    provenance_entries.append(ProvenanceEntry(
                        field="emails",
                        source=SourceType.ATS_JSON,
                        method=ExtractionMethod.DIRECT_MAPPING,
                        confidence=0.9,
                        raw_value=e,
                    ))

            elif canonical == "phones":
                phones = self._to_list(value)
                data["phones"].extend(phones)
                for p in phones:
                    provenance_entries.append(ProvenanceEntry(
                        field="phones",
                        source=SourceType.ATS_JSON,
                        method=ExtractionMethod.DIRECT_MAPPING,
                        confidence=0.9,
                        raw_value=p,
                    ))

            elif canonical == "location_raw":
                if isinstance(value, dict):
                    # Structured location: {"city": ..., "state": ..., "country": ...}
                    parts = [
                        str(value.get("city", "")),
                        str(value.get("state", "")),
                        str(value.get("country", "")),
                    ]
                    data["location_raw"] = ", ".join(p for p in parts if p)
                else:
                    data["location_raw"] = str(value).strip()
                provenance_entries.append(ProvenanceEntry(
                    field="location",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.85,
                    raw_value=str(value),
                ))

            elif canonical == "headline":
                data["headline"] = str(value).strip()
                provenance_entries.append(ProvenanceEntry(
                    field="headline",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.9,
                    raw_value=str(value),
                ))

            elif canonical == "skills_raw":
                skills = self._to_list(value)
                data["skills_raw"].extend(skills)
                provenance_entries.append(ProvenanceEntry(
                    field="skills",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.85,
                    raw_value=str(value),
                ))

            elif canonical == "experience_raw":
                if isinstance(value, list):
                    data["experience_raw"].extend(value)
                elif isinstance(value, dict):
                    data["experience_raw"].append(value)
                provenance_entries.append(ProvenanceEntry(
                    field="experience",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.85,
                    raw_value=str(value)[:200],
                ))

            elif canonical == "education_raw":
                if isinstance(value, list):
                    data["education_raw"].extend(value)
                elif isinstance(value, dict):
                    data["education_raw"].append(value)
                provenance_entries.append(ProvenanceEntry(
                    field="education",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.85,
                    raw_value=str(value)[:200],
                ))

            elif canonical == "_linkedin":
                data["links"]["linkedin"] = str(value).strip()

            elif canonical == "_github":
                data["links"]["github"] = str(value).strip()

            elif canonical == "_portfolio":
                data["links"]["portfolio"] = str(value).strip()

            elif canonical == "years_experience":
                try:
                    data["years_experience"] = float(value)
                except (ValueError, TypeError):
                    data["extra"]["years_experience_raw"] = value

            elif canonical == "_city":
                city = str(value).strip()
            elif canonical == "_state":
                state = str(value).strip()
            elif canonical == "_country":
                country = str(value).strip()

        # Assemble name from parts if needed
        if not data.get("full_name") and (first_name or last_name):
            parts = [p for p in [first_name, last_name] if p]
            data["full_name"] = " ".join(parts)
            provenance_entries.append(ProvenanceEntry(
                field="full_name",
                source=SourceType.ATS_JSON,
                method=ExtractionMethod.DIRECT_MAPPING,
                confidence=0.9,
                raw_value=f"{first_name}|{last_name}",
            ))

        # Assemble location from parts if needed
        if not data.get("location_raw") and any([city, state, country]):
            parts = [p for p in [city, state, country] if p]
            data["location_raw"] = ", ".join(parts)

        # Skip entries with no useful data
        if not data.get("full_name") and not data.get("emails"):
            logger.debug("Skipping ATS entry %d — no name or email", idx)
            return None

        data["provenance"] = provenance_entries
        data["source_type"] = SourceType.ATS_JSON
        data["source_file"] = source_path
        data["extraction_confidence"] = 0.9

        return RawRecord(**data)

    @staticmethod
    def _to_list(value: Any) -> list[str]:
        """Convert a value to a list of strings."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if v is not None and str(v).strip()]
        elif isinstance(value, str):
            # Try splitting on common delimiters
            if "," in value:
                return [s.strip() for s in value.split(",") if s.strip()]
            elif ";" in value:
                return [s.strip() for s in value.split(";") if s.strip()]
            return [value.strip()] if value.strip() else []
        else:
            return [str(value)] if value else []
