"""
CSV Extractor — parses recruiter CSV exports into RawRecords.

Handles missing columns, empty cells, and malformed rows gracefully.
Maps CSV column headers to canonical field names via a configurable mapping.
"""

from __future__ import annotations

import csv
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
# Column name mapping — maps common CSV headers to canonical fields
# ---------------------------------------------------------------------------

_COLUMN_MAP: dict[str, str] = {
    # Name variations
    "name": "full_name",
    "full_name": "full_name",
    "full name": "full_name",
    "candidate_name": "full_name",
    "candidate name": "full_name",
    "first_name": "_first_name",
    "first name": "_first_name",
    "last_name": "_last_name",
    "last name": "_last_name",

    # Email variations
    "email": "emails",
    "emails": "emails",
    "email_address": "emails",
    "email address": "emails",
    "contact_email": "emails",
    "contact email": "emails",

    # Phone variations
    "phone": "phones",
    "phones": "phones",
    "phone_number": "phones",
    "phone number": "phones",
    "mobile": "phones",
    "mobile_number": "phones",
    "contact_phone": "phones",

    # Location variations
    "location": "location_raw",
    "city": "_city",
    "state": "_state",
    "country": "_country",
    "address": "location_raw",

    # Professional
    "title": "headline",
    "headline": "headline",
    "job_title": "headline",
    "job title": "headline",
    "current_title": "headline",
    "position": "headline",

    # Skills
    "skills": "skills_raw",
    "skill": "skills_raw",
    "technologies": "skills_raw",
    "tech_stack": "skills_raw",

    # Links
    "linkedin": "_linkedin",
    "linkedin_url": "_linkedin",
    "github": "_github",
    "github_url": "_github",
    "portfolio": "_portfolio",
    "website": "_portfolio",

    # Experience
    "years_experience": "years_experience",
    "years_of_experience": "years_experience",
    "experience_years": "years_experience",
    "yoe": "years_experience",

    # Company / current employer
    "company": "_company",
    "current_company": "_company",
    "employer": "_company",
}


class CSVExtractor(BaseExtractor):
    """Extracts candidate records from a recruiter CSV export."""

    source_type = SourceType.RECRUITER_CSV

    def extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        path = Path(source_path)
        if not path.exists():
            logger.warning("CSV file not found: %s", source_path)
            return []

        records: list[RawRecord] = []
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    logger.warning("CSV has no headers: %s", source_path)
                    return []

                # Map actual headers to canonical fields
                header_map = self._build_header_map(reader.fieldnames)
                logger.debug("CSV header mapping: %s", header_map)

                for row_num, row in enumerate(reader, start=2):  # row 1 is header
                    try:
                        record = self._row_to_record(row, header_map, source_path, row_num)
                        if record:
                            records.append(record)
                    except Exception as e:
                        logger.warning(
                            "Skipping malformed row %d in %s: %s",
                            row_num, source_path, e,
                        )
                        continue

        except Exception as e:
            logger.error("Failed to read CSV %s: %s", source_path, e)

        return records

    def _build_header_map(self, fieldnames: list[str]) -> dict[str, str]:
        """Map actual CSV column names to canonical field names."""
        mapping = {}
        for col in fieldnames:
            normalized = col.strip().lower()
            if normalized in _COLUMN_MAP:
                mapping[col] = _COLUMN_MAP[normalized]
            else:
                mapping[col] = None  # Unknown column — stash in extra
        return mapping

    def _row_to_record(
        self,
        row: dict[str, str],
        header_map: dict[str, str | None],
        source_path: str,
        row_num: int,
    ) -> RawRecord | None:
        """Convert a single CSV row to a RawRecord."""
        data: dict[str, Any] = {
            "emails": [],
            "phones": [],
            "skills_raw": [],
            "links": {},
            "experience_raw": [],
            "extra": {},
        }
        provenance_entries: list[ProvenanceEntry] = []

        first_name = None
        last_name = None
        city = None
        state = None
        country = None
        company = None

        for col, value in row.items():
            if not value or not value.strip():
                continue

            value = value.strip()
            canonical = header_map.get(col)

            if canonical is None:
                # Unknown column — stash in extra
                data["extra"][col] = value
                continue

            if canonical == "full_name":
                data["full_name"] = value
                provenance_entries.append(ProvenanceEntry(
                    field="full_name",
                    source=SourceType.RECRUITER_CSV,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.8,
                    raw_value=value,
                ))

            elif canonical == "_first_name":
                first_name = value

            elif canonical == "_last_name":
                last_name = value

            elif canonical == "emails":
                # Could be comma-separated
                emails = [e.strip() for e in value.split(",") if e.strip()]
                data["emails"].extend(emails)
                for e in emails:
                    provenance_entries.append(ProvenanceEntry(
                        field="emails",
                        source=SourceType.RECRUITER_CSV,
                        method=ExtractionMethod.DIRECT_MAPPING,
                        confidence=0.85,
                        raw_value=e,
                    ))

            elif canonical == "phones":
                phones = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
                data["phones"].extend(phones)
                for p in phones:
                    provenance_entries.append(ProvenanceEntry(
                        field="phones",
                        source=SourceType.RECRUITER_CSV,
                        method=ExtractionMethod.DIRECT_MAPPING,
                        confidence=0.8,
                        raw_value=p,
                    ))

            elif canonical == "location_raw":
                data["location_raw"] = value
                provenance_entries.append(ProvenanceEntry(
                    field="location",
                    source=SourceType.RECRUITER_CSV,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.7,
                    raw_value=value,
                ))

            elif canonical == "headline":
                data["headline"] = value
                provenance_entries.append(ProvenanceEntry(
                    field="headline",
                    source=SourceType.RECRUITER_CSV,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.8,
                    raw_value=value,
                ))

            elif canonical == "skills_raw":
                skills = [s.strip() for s in value.replace(";", ",").split(",") if s.strip()]
                data["skills_raw"].extend(skills)
                provenance_entries.append(ProvenanceEntry(
                    field="skills",
                    source=SourceType.RECRUITER_CSV,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.7,
                    raw_value=value,
                ))

            elif canonical == "_linkedin":
                data["links"]["linkedin"] = value

            elif canonical == "_github":
                data["links"]["github"] = value

            elif canonical == "_portfolio":
                data["links"]["portfolio"] = value

            elif canonical == "years_experience":
                try:
                    data["years_experience"] = float(value)
                except ValueError:
                    data["extra"]["years_experience_raw"] = value

            elif canonical == "_city":
                city = value
            elif canonical == "_state":
                state = value
            elif canonical == "_country":
                country = value
            elif canonical == "_company":
                company = value

        # Assemble name from first + last if full_name wasn't set
        if not data.get("full_name") and (first_name or last_name):
            parts = [p for p in [first_name, last_name] if p]
            data["full_name"] = " ".join(parts)
            provenance_entries.append(ProvenanceEntry(
                field="full_name",
                source=SourceType.RECRUITER_CSV,
                method=ExtractionMethod.DIRECT_MAPPING,
                confidence=0.8,
                raw_value=f"{first_name}|{last_name}",
            ))

        # Assemble location from city/state/country if location_raw wasn't set
        if not data.get("location_raw") and any([city, state, country]):
            parts = [p for p in [city, state, country] if p]
            data["location_raw"] = ", ".join(parts)

        # Add current company as experience if available
        if company:
            data["experience_raw"].append({
                "company": company,
                "title": data.get("headline"),
                "start": None,
                "end": None,
            })

        # Skip rows with no useful data
        if not data.get("full_name") and not data.get("emails"):
            logger.debug("Skipping row %d — no name or email", row_num)
            return None

        data["provenance"] = provenance_entries
        data["source_type"] = SourceType.RECRUITER_CSV
        data["source_file"] = source_path
        data["extraction_confidence"] = 0.8

        return RawRecord(**data)
