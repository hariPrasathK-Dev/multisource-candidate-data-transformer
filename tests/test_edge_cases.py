"""
Edge case tests — verifying the pipeline handles garbage/missing data gracefully.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.extractors.csv_extractor import CSVExtractor
from src.extractors.ats_json_extractor import ATSJSONExtractor
from src.extractors.resume_extractor import ResumeExtractor
from src.merger.conflict_resolver import ConflictResolver
from src.models.canonical import RawRecord, SourceType


class TestGarbageInputs:
    """Tests that garbage inputs don't crash the pipeline."""

    def test_empty_csv(self, tmp_path):
        """Empty CSV produces no records."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        extractor = CSVExtractor()
        records = extractor.safe_extract(str(csv_file))
        assert records == []

    def test_malformed_csv(self, tmp_path):
        """CSV with only headers and no data produces no records."""
        csv_file = tmp_path / "headers_only.csv"
        csv_file.write_text("name,email,phone\n")

        extractor = CSVExtractor()
        records = extractor.safe_extract(str(csv_file))
        assert records == []

    def test_csv_with_some_bad_rows(self, tmp_path):
        """CSV with mixed good/bad rows should extract the good ones."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "name,email,phone\n"
            "Good Person,good@test.com,(555)123-4567\n"
            ",,\n"  # Empty row — should be skipped
            "Another Person,another@test.com,N/A\n"
        )

        extractor = CSVExtractor()
        records = extractor.safe_extract(str(csv_file))
        assert len(records) == 2

    def test_invalid_json(self, tmp_path):
        """Invalid JSON should not crash."""
        json_file = tmp_path / "bad.json"
        json_file.write_text("{ this is not valid json }")

        extractor = ATSJSONExtractor()
        records = extractor.safe_extract(str(json_file))
        assert records == []

    def test_empty_json_array(self, tmp_path):
        """Empty JSON array produces no records."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")

        extractor = ATSJSONExtractor()
        records = extractor.safe_extract(str(json_file))
        assert records == []

    def test_nonexistent_file(self):
        """Non-existent file should not crash."""
        extractor = CSVExtractor()
        records = extractor.safe_extract("/nonexistent/path.csv")
        assert records == []

    def test_resume_empty_text(self, tmp_path):
        """Empty resume file produces no records."""
        resume = tmp_path / "empty.txt"
        resume.write_text("")

        extractor = ResumeExtractor(bedrock_client=None)
        records = extractor.safe_extract(str(resume))
        assert records == []


class TestMissingFields:
    """Tests that missing fields result in null values, not crashes."""

    def test_profile_with_only_name(self):
        """A record with only a name should produce a valid (sparse) profile."""
        record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Lone Person",
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([record])

        assert profile.full_name == "Lone Person"
        assert profile.emails == []
        assert profile.phones == []
        assert profile.location is None
        assert profile.skills == []
        assert profile.overall_confidence == 0.0  # Not yet scored

    def test_profile_with_only_email(self):
        """A record with only email should still produce a profile."""
        record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            emails=["orphan@test.com"],
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([record])

        assert profile.full_name is None
        assert "orphan@test.com" in profile.emails


class TestConflictingData:
    """Tests for conflicting data across sources."""

    def test_name_conflict_highest_trust_wins(self):
        """When two sources disagree on name, highest trust wins."""
        csv_record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="John Doe",
            provenance=[],
        )
        ats_record = RawRecord(
            source_type=SourceType.ATS_JSON,
            full_name="Jonathan Doe",
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([csv_record, ats_record])

        # ATS (0.9) > CSV (0.8)
        assert profile.full_name == "Jonathan Doe"

    def test_conflicting_locations_highest_trust_wins(self):
        """When locations conflict, highest trust wins."""
        csv_record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            location_raw="Bangalore, India",
            provenance=[],
        )
        ats_record = RawRecord(
            source_type=SourceType.ATS_JSON,
            location_raw="Bengaluru, Karnataka, India",
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([csv_record, ats_record])

        # ATS wins — location should be from ATS
        assert profile.location is not None
        assert profile.location.country == "IN"

    def test_never_invents_data(self):
        """Fields not present in any source must be null, not invented."""
        record = RawRecord(
            source_type=SourceType.RECRUITER_CSV,
            full_name="Simple Person",
            emails=["simple@test.com"],
            provenance=[],
        )

        resolver = ConflictResolver()
        profile = resolver.merge([record])

        # These were never provided — must be null/empty, not invented
        assert profile.headline is None
        assert profile.years_experience is None
        assert profile.experience == []
        assert profile.education == []
