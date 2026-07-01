"""
Confidence Scorer — computes per-field and overall confidence scores.

Scoring formula:
  Per-field: source_trust × extraction_confidence × agreement_bonus
  Overall:   Weighted average of per-field scores

Agreement bonus: if 2+ sources agree on the same value → ×1.2 (capped at 1.0)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from src.models.canonical import (
    CanonicalProfile,
    ProvenanceEntry,
    SourceType,
)

logger = logging.getLogger(__name__)

# Source trust weights (defaults — should match conflict_resolver)
_SOURCE_TRUST: dict[str, float] = {
    "ats_json": 0.9,
    "recruiter_csv": 0.8,
    "github_api": 0.75,
    "resume_llm": 0.7,
    "resume_regex_fallback": 0.5,
    "unknown": 0.3,
}

# Field importance weights for overall score calculation
_FIELD_WEIGHTS: dict[str, float] = {
    "full_name": 2.0,
    "emails": 1.5,
    "phones": 1.0,
    "location": 0.8,
    "headline": 0.7,
    "skills": 1.0,
    "experience": 1.2,
    "education": 0.8,
    "links": 0.5,
    "years_experience": 0.5,
}

# Agreement bonus multiplier
_AGREEMENT_BONUS = 1.2
_MAX_CONFIDENCE = 1.0


class ConfidenceScorer:
    """Computes confidence scores for a CanonicalProfile."""

    def __init__(
        self,
        source_trust: Optional[dict[str, float]] = None,
        field_weights: Optional[dict[str, float]] = None,
    ):
        self.source_trust = source_trust or _SOURCE_TRUST
        self.field_weights = field_weights or _FIELD_WEIGHTS

    def score(self, profile: CanonicalProfile) -> CanonicalProfile:
        """
        Score a CanonicalProfile in-place and return it.

        Updates:
          - Per-skill confidence
          - Provenance entry confidences
          - Overall confidence
        """
        # Group provenance entries by field
        field_provenance: dict[str, list[ProvenanceEntry]] = defaultdict(list)
        for entry in profile.provenance:
            field_provenance[entry.field].append(entry)

        # Compute per-field confidence
        field_confidences: dict[str, float] = {}

        for field_name, entries in field_provenance.items():
            field_conf = self._compute_field_confidence(field_name, entries, profile)
            field_confidences[field_name] = field_conf

        # Add confidence for fields that exist but have no provenance
        for field_name in ["full_name", "emails", "phones", "location", "headline",
                           "skills", "experience", "education", "links"]:
            if field_name not in field_confidences:
                value = getattr(profile, field_name, None)
                if value and (not isinstance(value, (list,)) or len(value) > 0):
                    field_confidences[field_name] = 0.3  # Low confidence — no provenance

        # Update per-skill confidence
        for skill in profile.skills:
            skill_conf = field_confidences.get("skills", 0.5)
            # Boost if skill came from a high-trust source
            if skill.source:
                source_trust = self.source_trust.get(skill.source.value, 0.5)
                skill.confidence = min(skill_conf * source_trust * 1.5, _MAX_CONFIDENCE)
            else:
                skill.confidence = round(skill_conf, 2)

        # Compute overall confidence
        profile.overall_confidence = self._compute_overall(field_confidences)

        logger.debug(
            "Confidence scored for %s: overall=%.2f, fields=%s",
            profile.candidate_id[:8],
            profile.overall_confidence,
            {k: round(v, 2) for k, v in field_confidences.items()},
        )

        return profile

    def _compute_field_confidence(
        self,
        field_name: str,
        entries: list[ProvenanceEntry],
        profile: CanonicalProfile,
    ) -> float:
        """
        Compute confidence for a single field.

        Formula: max(source_trust × extraction_confidence) × agreement_bonus
        """
        if not entries:
            return 0.0

        # Find the best individual confidence
        best_confidence = 0.0
        source_values: dict[str, set[str]] = defaultdict(set)

        for entry in entries:
            source_trust = self.source_trust.get(entry.source.value, 0.3)
            individual = source_trust * entry.confidence
            best_confidence = max(best_confidence, individual)

            # Track values by source for agreement check
            if entry.raw_value:
                source_values[entry.source.value].add(
                    entry.raw_value.lower().strip()
                )

        # Agreement bonus: if 2+ distinct sources provided values
        distinct_sources = len([s for s in source_values if source_values[s]])
        if distinct_sources >= 2:
            # Check if any values actually agree across sources
            all_values = set()
            for values in source_values.values():
                all_values.update(values)

            # Count how many sources agree on at least one value
            agreement_count = 0
            for value in all_values:
                sources_with_value = sum(
                    1 for s_values in source_values.values()
                    if value in s_values
                )
                if sources_with_value >= 2:
                    agreement_count = sources_with_value
                    break

            if agreement_count >= 2:
                best_confidence *= _AGREEMENT_BONUS

        # Also boost if field has multiple provenance entries (more coverage)
        coverage_bonus = min(1.0 + (len(entries) - 1) * 0.05, 1.15)
        best_confidence *= coverage_bonus

        return min(round(best_confidence, 4), _MAX_CONFIDENCE)

    def _compute_overall(self, field_confidences: dict[str, float]) -> float:
        """
        Compute overall confidence as a weighted average.
        """
        if not field_confidences:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for field, confidence in field_confidences.items():
            weight = self.field_weights.get(field, 0.5)
            weighted_sum += confidence * weight
            weight_total += weight

        if weight_total == 0:
            return 0.0

        overall = weighted_sum / weight_total
        return round(min(overall, _MAX_CONFIDENCE), 2)
