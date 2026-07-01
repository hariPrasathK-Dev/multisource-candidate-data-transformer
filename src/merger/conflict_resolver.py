"""
Conflict Resolver — merges multiple RawRecords into one CanonicalProfile.

Survivorship rules:
  - Scalar fields: pick highest-trust source; if tied, pick most recent
  - Array fields: union-merge, deduplicate
  - Nested arrays: union-merge, deduplicate by key fields

Source trust weights are configurable via source_trust.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.models.canonical import (
    CanonicalProfile,
    EducationEntry,
    ExperienceEntry,
    ExtractionMethod,
    Links,
    Location,
    ProvenanceEntry,
    RawRecord,
    Skill,
    SourceType,
)
from src.normalizers.date_normalizer import normalize_date
from src.normalizers.location_normalizer import normalize_location
from src.normalizers.phone_normalizer import normalize_phones
from src.normalizers.skill_normalizer import normalize_skills

logger = logging.getLogger(__name__)

# Default source trust weights
_DEFAULT_TRUST: dict[str, float] = {
    "ats_json": 0.9,
    "recruiter_csv": 0.8,
    "github_api": 0.75,
    "resume_llm": 0.7,
    "resume_regex_fallback": 0.5,
    "unknown": 0.3,
}


class ConflictResolver:
    """Merges a cluster of RawRecords into a single CanonicalProfile."""

    def __init__(self, trust_weights: Optional[dict[str, float]] = None):
        self.trust_weights = trust_weights or _DEFAULT_TRUST

    @classmethod
    def from_config(cls, config_path: str) -> "ConflictResolver":
        """Load trust weights from a JSON config file."""
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                logger.info("Loaded source trust weights from %s", config_path)
                return cls(trust_weights=weights)
            except Exception as e:
                logger.warning("Failed to load trust weights from %s: %s", config_path, e)
        return cls()

    def merge(self, cluster: list[RawRecord]) -> CanonicalProfile:
        """
        Merge a cluster of records into one CanonicalProfile.

        Each record in the cluster is believed to refer to the same candidate.
        """
        if not cluster:
            return CanonicalProfile()

        if len(cluster) == 1:
            return self._single_record_to_profile(cluster[0])

        # Sort by trust weight (highest first) for scalar field precedence
        sorted_records = sorted(
            cluster,
            key=lambda r: self._get_trust(r.source_type),
            reverse=True,
        )

        all_provenance: list[ProvenanceEntry] = []

        # --- Scalar fields: pick from highest-trust source ---
        full_name = self._pick_scalar(sorted_records, "full_name")
        headline = self._pick_scalar(sorted_records, "headline")
        years_exp = self._pick_numeric(sorted_records, "years_experience")

        # --- Array fields: union-merge ---
        all_emails = self._union_emails(sorted_records)
        all_phones = self._union_phones(sorted_records)
        all_skills = self._union_skills(sorted_records)

        # --- Nested: location (pick highest trust) ---
        location = self._pick_location(sorted_records)

        # --- Links: merge all ---
        links = self._merge_links(sorted_records)

        # --- Experience: union-merge, dedupe by company+title ---
        experience = self._merge_experience(sorted_records)

        # --- Education: union-merge, dedupe by institution+degree ---
        education = self._merge_education(sorted_records)

        # --- Collect all provenance entries ---
        for record in cluster:
            all_provenance.extend(record.provenance)

        # Add merge provenance
        for field_name in ["full_name", "emails", "phones", "location", "headline",
                           "skills", "experience", "education"]:
            sources = [r.source_type.value for r in cluster if self._has_field(r, field_name)]
            if len(sources) > 1:
                all_provenance.append(ProvenanceEntry(
                    field=field_name,
                    source=SourceType.UNKNOWN,
                    method=ExtractionMethod.MERGED,
                    confidence=0.0,  # Will be computed by confidence scorer
                    raw_value=f"Merged from: {', '.join(sources)}",
                ))

        profile = CanonicalProfile(
            full_name=full_name,
            emails=all_emails,
            phones=all_phones,
            location=location,
            headline=headline,
            links=links,
            years_experience=years_exp,
            skills=all_skills,
            experience=experience,
            education=education,
            provenance=all_provenance,
        )

        return profile

    def _get_trust(self, source_type: SourceType) -> float:
        return self.trust_weights.get(source_type.value, 0.3)

    def _pick_scalar(
        self, sorted_records: list[RawRecord], field: str
    ) -> Optional[str]:
        """Pick a scalar field from the highest-trust source that has it."""
        for record in sorted_records:
            value = getattr(record, field, None)
            if value and str(value).strip():
                return str(value).strip()
        return None

    def _pick_numeric(
        self, sorted_records: list[RawRecord], field: str
    ) -> Optional[float]:
        """Pick a numeric field from the highest-trust source."""
        for record in sorted_records:
            value = getattr(record, field, None)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        return None

    def _union_emails(self, records: list[RawRecord]) -> list[str]:
        """Union-merge and deduplicate emails across all records."""
        seen: set[str] = set()
        result: list[str] = []
        for r in records:
            for email in r.emails:
                email_lower = email.lower().strip()
                if email_lower and email_lower not in seen:
                    seen.add(email_lower)
                    result.append(email_lower)
        return result

    def _union_phones(self, records: list[RawRecord]) -> list[str]:
        """Union-merge phones, normalize to E.164, deduplicate."""
        all_raw: list[str] = []
        for r in records:
            all_raw.extend(r.phones)
        # Get country hint from location if available
        country_hint = None
        for r in records:
            if r.location_raw:
                loc = normalize_location(r.location_raw)
                if loc and loc.get("country"):
                    country_hint = loc["country"]
                    break
        return normalize_phones(all_raw, country_hint)

    def _union_skills(self, records: list[RawRecord]) -> list[Skill]:
        """Union-merge skills, normalize, deduplicate."""
        all_raw: list[str] = []
        skill_sources: dict[str, SourceType] = {}
        for r in records:
            for s in r.skills_raw:
                all_raw.append(s)
                if s not in skill_sources:
                    skill_sources[s] = r.source_type

        # Normalize
        canonical = normalize_skills(all_raw)

        # Build Skill objects with source info
        return [
            Skill(
                name=name,
                confidence=0.5,  # Will be refined by confidence scorer
                source=skill_sources.get(name),
            )
            for name in canonical
        ]

    def _pick_location(self, sorted_records: list[RawRecord]) -> Optional[Location]:
        """Pick location from highest-trust source and normalize."""
        for record in sorted_records:
            if record.location_raw:
                loc = normalize_location(record.location_raw)
                if loc:
                    return Location(**loc)
        return None

    def _merge_links(self, records: list[RawRecord]) -> Links:
        """Merge links from all sources — first non-None wins for each type."""
        linkedin = None
        github = None
        portfolio = None
        other: list[str] = []

        for r in records:
            links = r.links
            if not links:
                continue

            if not linkedin and links.get("linkedin"):
                linkedin = links["linkedin"]
            if not github and links.get("github"):
                github = links["github"]
            if not portfolio and links.get("portfolio"):
                portfolio = links["portfolio"]

            for key, value in links.items():
                if key not in ("linkedin", "github", "portfolio") and value:
                    if value not in other:
                        other.append(str(value))

        return Links(
            linkedin=linkedin,
            github=github,
            portfolio=portfolio,
            other=other,
        )

    def _merge_experience(self, records: list[RawRecord]) -> list[ExperienceEntry]:
        """Union-merge experience, deduplicate by company+title."""
        seen_keys: set[str] = set()
        result: list[ExperienceEntry] = []

        for r in records:
            for exp_dict in r.experience_raw:
                if not isinstance(exp_dict, dict):
                    continue

                company = str(exp_dict.get("company", "")).strip().lower()
                title = str(exp_dict.get("title", "")).strip().lower()
                key = f"{company}|{title}"

                if key in seen_keys and key != "|":
                    continue
                seen_keys.add(key)

                # Normalize dates
                start = normalize_date(exp_dict.get("start"))
                end = normalize_date(exp_dict.get("end"))

                result.append(ExperienceEntry(
                    company=exp_dict.get("company"),
                    title=exp_dict.get("title"),
                    start=start,
                    end=end,
                    summary=exp_dict.get("summary"),
                ))

        return result

    def _merge_education(self, records: list[RawRecord]) -> list[EducationEntry]:
        """Union-merge education, deduplicate by institution+degree."""
        seen_keys: set[str] = set()
        result: list[EducationEntry] = []

        for r in records:
            for edu_dict in r.education_raw:
                if not isinstance(edu_dict, dict):
                    continue

                institution = str(edu_dict.get("institution", "")).strip().lower()
                degree = str(edu_dict.get("degree", "")).strip().lower()
                key = f"{institution}|{degree}"

                if key in seen_keys and key != "|":
                    continue
                seen_keys.add(key)

                # Parse end_year
                end_year = edu_dict.get("end_year")
                if end_year is not None:
                    try:
                        end_year = int(end_year)
                    except (ValueError, TypeError):
                        end_year = None

                result.append(EducationEntry(
                    institution=edu_dict.get("institution"),
                    degree=edu_dict.get("degree"),
                    field_of_study=edu_dict.get("field_of_study") or edu_dict.get("field"),
                    end_year=end_year,
                ))

        return result

    def _single_record_to_profile(self, record: RawRecord) -> CanonicalProfile:
        """Convert a single RawRecord to a CanonicalProfile (no merge needed)."""
        # Normalize location
        location = None
        if record.location_raw:
            loc = normalize_location(record.location_raw)
            if loc:
                location = Location(**loc)

        # Normalize phones
        country_hint = location.country if location else None
        phones = normalize_phones(record.phones, country_hint)

        # Normalize skills
        skill_names = normalize_skills(record.skills_raw)
        skills = [
            Skill(name=name, confidence=0.5, source=record.source_type)
            for name in skill_names
        ]

        # Build links
        links_data = record.links or {}
        links = Links(
            linkedin=links_data.get("linkedin"),
            github=links_data.get("github"),
            portfolio=links_data.get("portfolio"),
        )

        # Normalize experience dates
        experience = []
        for exp_dict in record.experience_raw:
            if isinstance(exp_dict, dict):
                experience.append(ExperienceEntry(
                    company=exp_dict.get("company"),
                    title=exp_dict.get("title"),
                    start=normalize_date(exp_dict.get("start")),
                    end=normalize_date(exp_dict.get("end")),
                    summary=exp_dict.get("summary"),
                ))

        # Parse education
        education = []
        for edu_dict in record.education_raw:
            if isinstance(edu_dict, dict):
                end_year = edu_dict.get("end_year")
                if end_year is not None:
                    try:
                        end_year = int(end_year)
                    except (ValueError, TypeError):
                        end_year = None
                education.append(EducationEntry(
                    institution=edu_dict.get("institution"),
                    degree=edu_dict.get("degree"),
                    field_of_study=edu_dict.get("field_of_study") or edu_dict.get("field"),
                    end_year=end_year,
                ))

        return CanonicalProfile(
            full_name=record.full_name,
            emails=list({e.lower() for e in record.emails if e}),
            phones=phones,
            location=location,
            headline=record.headline,
            links=links,
            years_experience=record.years_experience,
            skills=skills,
            experience=experience,
            education=education,
            provenance=record.provenance,
        )

    @staticmethod
    def _has_field(record: RawRecord, field: str) -> bool:
        """Check if a record has a non-empty value for a field."""
        if field == "emails":
            return bool(record.emails)
        elif field == "phones":
            return bool(record.phones)
        elif field == "skills":
            return bool(record.skills_raw)
        elif field == "experience":
            return bool(record.experience_raw)
        elif field == "education":
            return bool(record.education_raw)
        elif field == "location":
            return bool(record.location_raw)
        else:
            val = getattr(record, field, None)
            return val is not None and str(val).strip() != ""
