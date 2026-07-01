"""
Unit tests for entity resolution and conflict resolution (merger).
"""

import pytest

from src.models.canonical import (
    ExtractionMethod,
    ProvenanceEntry,
    RawRecord,
    SourceType,
)
from src.merger.entity_resolver import EntityResolver
from src.merger.conflict_resolver import ConflictResolver


class TestEntityResolver:
    """Tests for entity resolution — matching records across sources."""

    def test_email_match(self):
        """Records with same email should be clustered together."""
        r1 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="John Doe",
            emails=["john@example.com"],
        )
        r2 = RawRecord(
            source_type=SourceType.ATS_JSON,
            full_name="Jonathan Doe",
            emails=["john@example.com"],
        )
        r3 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Jane Smith",
            emails=["jane@example.com"],
        )

        resolver = EntityResolver()
        clusters = resolver.resolve([r1, r2, r3])

        assert len(clusters) == 2
        # Find the cluster with John
        john_cluster = [c for c in clusters if any(r.full_name == "John Doe" for r in c)][0]
        assert len(john_cluster) == 2

    def test_phone_match(self):
        """Records with same phone should be clustered."""
        r1 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Priya Sharma",
            phones=["+919876543210"],
        )
        r2 = RawRecord(
            source_type=SourceType.RESUME_LLM,
            full_name="Priya S.",
            phones=["+919876543210"],
        )

        resolver = EntityResolver()
        clusters = resolver.resolve([r1, r2])

        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_no_match_different_people(self):
        """Completely different people should stay separate."""
        r1 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Alice",
            emails=["alice@a.com"],
        )
        r2 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Bob",
            emails=["bob@b.com"],
        )

        resolver = EntityResolver()
        clusters = resolver.resolve([r1, r2])

        assert len(clusters) == 2

    def test_empty_input(self):
        """Empty input returns empty output."""
        resolver = EntityResolver()
        assert resolver.resolve([]) == []


class TestConflictResolver:
    """Tests for conflict resolution — merging records."""

    def test_single_record_merge(self):
        """Single record should produce a valid profile."""
        record = RawRecord(
            source_type=SourceType.ATS_JSON,
            full_name="Test User",
            emails=["test@example.com"],
            phones=["+14155550000"],
            location_raw="San Francisco, CA, US",
            skills_raw=["Python", "JS"],
            provenance=[
                ProvenanceEntry(
                    field="full_name",
                    source=SourceType.ATS_JSON,
                    method=ExtractionMethod.DIRECT_MAPPING,
                    confidence=0.9,
                )
            ],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([record])

        assert profile.full_name == "Test User"
        assert "test@example.com" in profile.emails
        assert profile.location is not None
        assert profile.location.country == "US"
        assert len(profile.skills) >= 2

    def test_multi_source_merge_scalar(self):
        """Scalar fields should pick from highest-trust source (ATS > CSV)."""
        csv_record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="John Doe",
            headline="Engineer",
            provenance=[],
        )
        ats_record = RawRecord(
            source_type=SourceType.ATS_JSON,
            full_name="Jonathan Doe",
            headline="Staff Engineer",
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([csv_record, ats_record])

        # ATS has higher trust (0.9 vs 0.8) so its name wins
        assert profile.full_name == "Jonathan Doe"
        assert profile.headline == "Staff Engineer"

    def test_multi_source_merge_arrays(self):
        """Array fields should be union-merged and deduplicated."""
        r1 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            emails=["a@test.com"],
            skills_raw=["Python", "JS"],
            provenance=[],
        )
        r2 = RawRecord(
            source_type=SourceType.ATS_JSON,
            emails=["a@test.com", "b@test.com"],
            skills_raw=["JavaScript", "React"],
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([r1, r2])

        # Emails should be deduped
        assert len(profile.emails) == 2
        assert "a@test.com" in profile.emails
        assert "b@test.com" in profile.emails

    def test_experience_dedup(self):
        """Experience with same company+title should be deduped."""
        r1 = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            experience_raw=[
                {"company": "Google", "title": "SWE", "start": "2020-01"},
            ],
            provenance=[],
        )
        r2 = RawRecord(
            source_type=SourceType.ATS_JSON,
            experience_raw=[
                {"company": "Google", "title": "SWE", "start": "Jan 2020"},
                {"company": "Meta", "title": "SDE", "start": "2022-06"},
            ],
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([r1, r2])

        # Google+SWE should be deduped, Meta+SDE should be added
        assert len(profile.experience) == 2
        companies = {e.company for e in profile.experience}
        assert "Google" in companies
        assert "Meta" in companies
