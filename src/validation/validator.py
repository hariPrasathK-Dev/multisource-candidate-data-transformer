"""
Validator — validates pipeline output against JSON schemas.

Uses jsonschema for validation. Reports all errors (doesn't fail on first).
Also validates runtime config at load time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import jsonschema
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default canonical output schema
# ---------------------------------------------------------------------------

_CANONICAL_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CanonicalCandidateProfile",
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "full_name": {"type": ["string", "null"]},
        "emails": {
            "type": "array",
            "items": {"type": "string", "format": "email"},
        },
        "phones": {
            "type": "array",
            "items": {"type": "string"},
            "description": "E.164 formatted phone numbers",
        },
        "location": {
            "type": ["object", "null"],
            "properties": {
                "city": {"type": ["string", "null"]},
                "state": {"type": ["string", "null"]},
                "country": {
                    "type": ["string", "null"],
                    "description": "ISO-3166 alpha-2 code",
                },
            },
        },
        "headline": {"type": ["string", "null"]},
        "links": {
            "type": ["object", "null"],
            "properties": {
                "linkedin": {"type": ["string", "null"]},
                "github": {"type": ["string", "null"]},
                "portfolio": {"type": ["string", "null"]},
                "other": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "years_experience": {"type": ["number", "null"]},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": ["string", "null"]},
                },
                "required": ["name"],
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "start": {
                        "type": ["string", "null"],
                        "description": "YYYY-MM format",
                    },
                    "end": {"type": ["string", "null"]},
                    "summary": {"type": ["string", "null"]},
                },
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": ["string", "null"]},
                    "degree": {"type": ["string", "null"]},
                    "field_of_study": {"type": ["string", "null"]},
                    "end_year": {"type": ["integer", "null"]},
                },
            },
        },
        "provenance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "source": {"type": "string"},
                    "method": {"type": "string"},
                    "confidence": {"type": "number"},
                    "raw_value": {"type": ["string", "null"]},
                    "text_span": {"type": ["string", "null"]},
                },
                "required": ["field", "source", "method"],
            },
        },
        "overall_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": ["candidate_id"],
}


class OutputValidator:
    """Validates pipeline output against JSON schemas."""

    def __init__(self, schema: Optional[dict[str, Any]] = None):
        self.schema = schema or _CANONICAL_SCHEMA
        self._validator = Draft7Validator(self.schema)

    @classmethod
    def from_schema_file(cls, path: str) -> "OutputValidator":
        """Load a custom schema from a JSON file."""
        schema_path = Path(path)
        if not schema_path.exists():
            logger.warning("Schema file not found: %s. Using default.", path)
            return cls()

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            return cls(schema=schema)
        except Exception as e:
            logger.warning("Failed to load schema from %s: %s. Using default.", path, e)
            return cls()

    def validate(self, data: dict[str, Any]) -> list[str]:
        """
        Validate a single output record.

        Returns a list of error messages (empty = valid).
        """
        errors = []
        for error in self._validator.iter_errors(data):
            error_path = " → ".join(str(p) for p in error.absolute_path) or "(root)"
            errors.append(f"[{error_path}] {error.message}")

        if errors:
            logger.warning("Validation errors: %s", errors)
        else:
            logger.debug("Validation passed for record")

        return errors

    def validate_batch(self, records: list[dict[str, Any]]) -> dict[int, list[str]]:
        """
        Validate a batch of output records.

        Returns a dict mapping record index → list of error messages.
        Only records with errors are included.
        """
        all_errors: dict[int, list[str]] = {}
        for idx, record in enumerate(records):
            errors = self.validate(record)
            if errors:
                all_errors[idx] = errors
        return all_errors

    def is_valid(self, data: dict[str, Any]) -> bool:
        """Quick check — returns True if the record is valid."""
        return self._validator.is_valid(data)
