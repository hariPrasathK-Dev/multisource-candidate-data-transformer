"""
Canonical data models for the Multi-Source Candidate Data Transformer.

Defines the golden-record schema and intermediate representations used
throughout the pipeline.  Every field on the canonical profile carries
provenance (which source, which method) so the output is fully explainable.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """Known source types that feed into the pipeline."""
    RECRUITER_CSV = "recruiter_csv"
    ATS_JSON = "ats_json"
    GITHUB_API = "github_api"
    RESUME_LLM = "resume_llm"
    RESUME_REGEX_FALLBACK = "resume_regex_fallback"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """How a field value was obtained."""
    DIRECT_MAPPING = "direct_mapping"
    LLM_EXTRACTION = "llm_extraction"
    REGEX_EXTRACTION = "regex_extraction"
    API_FETCH = "api_fetch"
    INFERRED = "inferred"
    MERGED = "merged"


# ---------------------------------------------------------------------------
# Provenance — tracks *where* every value came from
# ---------------------------------------------------------------------------

class ProvenanceEntry(BaseModel):
    """Records the origin of a single field value."""
    field: str = Field(..., description="Canonical field name")
    source: SourceType = Field(..., description="Which source provided this value")
    method: ExtractionMethod = Field(..., description="How the value was extracted")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Extraction confidence [0, 1]",
    )
    raw_value: Optional[str] = Field(
        default=None,
        description="Original value before normalization (for auditability)",
    )
    text_span: Optional[str] = Field(
        default=None,
        description="For LLM extractions — the source text span cited",
    )


# ---------------------------------------------------------------------------
# Nested value objects
# ---------------------------------------------------------------------------

class Location(BaseModel):
    """Normalized location — ISO-3166 alpha-2 country code."""
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = Field(
        default=None,
        description="ISO-3166 alpha-2 country code, e.g. 'US'",
    )

    @field_validator("country", mode="before")
    @classmethod
    def uppercase_country(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.upper().strip()
        return v


class Links(BaseModel):
    """Candidate web links — LinkedIn, GitHub, portfolio, etc."""
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    other: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    """A single skill with optional confidence and source provenance."""
    name: str = Field(..., description="Canonical skill name")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Optional[SourceType] = None


class ExperienceEntry(BaseModel):
    """One work-experience block."""
    company: Optional[str] = None
    title: Optional[str] = None
    start: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM format",
    )
    end: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM format, or null if current",
    )
    summary: Optional[str] = None

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        # Accept YYYY, YYYY-MM, or YYYY-MM-DD — we'll normalize later
        return v


class EducationEntry(BaseModel):
    """One education block."""
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    end_year: Optional[int] = None


# ---------------------------------------------------------------------------
# Raw Record — intermediate representation from any extractor
# ---------------------------------------------------------------------------

class RawRecord(BaseModel):
    """
    Intermediate representation that every extractor produces.

    This is a loose bag-of-fields — not yet normalized or validated.
    The pipeline normalizes these into canonical fields before merging.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType = SourceType.UNKNOWN
    source_file: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    # --- raw fields (all optional — each source may only fill a subset) ---
    full_name: Optional[str] = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    location_raw: Optional[str] = None
    headline: Optional[str] = None
    links: dict[str, Any] = Field(default_factory=dict)
    skills_raw: list[str] = Field(default_factory=list)
    experience_raw: list[dict[str, Any]] = Field(default_factory=list)
    education_raw: list[dict[str, Any]] = Field(default_factory=list)
    years_experience: Optional[float] = None
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all for source-specific fields not yet mapped",
    )

    # Provenance entries generated during extraction
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        # Allow arbitrary extra fields so extractors can stash source-specific data
        extra = "allow"


# ---------------------------------------------------------------------------
# Canonical Profile — the golden record
# ---------------------------------------------------------------------------

class CanonicalProfile(BaseModel):
    """
    The single, clean, canonical profile for one candidate.

    This is the output of the merge stage — all fields are normalized,
    provenance is attached, and confidence is scored.  The projection
    layer may reshape this before final emission.
    """
    candidate_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this canonical profile",
    )
    full_name: Optional[str] = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(
        default_factory=list,
        description="E.164 formatted phone numbers",
    )
    location: Optional[Location] = None
    headline: Optional[str] = None
    links: Optional[Links] = None
    years_experience: Optional[float] = None

    skills: list[Skill] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)

    # --- metadata ---
    provenance: list[ProvenanceEntry] = Field(
        default_factory=list,
        description="Full audit trail: one entry per field-source pair",
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score across all fields",
    )

    # ---- helpers ----

    def to_output_dict(self, include_confidence: bool = True) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output."""
        data = self.model_dump(mode="json", exclude_none=False)
        if not include_confidence:
            data.pop("provenance", None)
            data.pop("overall_confidence", None)
            # Also strip per-skill confidence
            for skill in data.get("skills", []):
                skill.pop("confidence", None)
                skill.pop("source", None)
        return data

    @field_validator("phones", mode="before")
    @classmethod
    def filter_none_phones(cls, v: list) -> list:
        """Drop None entries that may have been appended during merge."""
        return [p for p in v if p is not None]

    @field_validator("emails", mode="before")
    @classmethod
    def normalize_emails(cls, v: list) -> list:
        """Lowercase all emails, drop empties."""
        return list({e.lower().strip() for e in v if e and e.strip()})
