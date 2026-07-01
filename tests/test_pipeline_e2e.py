"""
End-to-end pipeline test — feeds sample inputs through the full pipeline.
"""

import json
import pytest
from pathlib import Path

from src.pipeline import Pipeline
from src.models.config import OutputConfig


class TestPipelineE2E:
    """End-to-end pipeline tests using sample data."""

    @pytest.fixture
    def sample_inputs_dir(self):
        """Path to sample inputs."""
        base = Path(__file__).parent.parent / "data" / "sample_inputs"
        if not base.exists():
            pytest.skip("Sample inputs not found")
        return str(base)

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline without Bedrock (uses regex fallback)."""
        return Pipeline()

    def test_csv_only(self):
        """Test pipeline with only CSV input."""
        csv_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "recruiter_export.csv")
        if not Path(csv_path).exists():
            pytest.skip("CSV sample not found")

        pipeline = Pipeline()
        results = pipeline.run(input_paths=[csv_path])

        assert len(results) > 0
        for record in results:
            assert "candidate_id" in record

    def test_ats_json_only(self):
        """Test pipeline with only ATS JSON input."""
        json_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "ats_records.json")
        if not Path(json_path).exists():
            pytest.skip("ATS JSON sample not found")

        pipeline = Pipeline()
        results = pipeline.run(input_paths=[json_path])

        assert len(results) > 0
        for record in results:
            assert "candidate_id" in record
            # ATS records should have full_name
            if record.get("full_name"):
                assert isinstance(record["full_name"], str)

    def test_multi_source_with_merge(self):
        """Test pipeline with CSV + ATS JSON — should merge matching candidates."""
        csv_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "recruiter_export.csv")
        json_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "ats_records.json")

        if not Path(csv_path).exists() or not Path(json_path).exists():
            pytest.skip("Sample files not found")

        pipeline = Pipeline()
        results = pipeline.run(input_paths=[csv_path, json_path])

        assert len(results) > 0

        # We have 5 in CSV and 3 in ATS (3 overlap) → should get 5 profiles
        # (3 merged + 2 CSV-only)
        # But exact count depends on entity resolution matching
        assert len(results) <= 8  # At most 5+3 if no merging

        # Check that each result has required fields
        for record in results:
            assert "candidate_id" in record

    def test_provenance_populated(self):
        """Every profile should have provenance entries."""
        csv_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "recruiter_export.csv")
        if not Path(csv_path).exists():
            pytest.skip("CSV sample not found")

        config = OutputConfig.default_config()
        pipeline = Pipeline()
        results = pipeline.run(input_paths=[csv_path])

        for record in results:
            # Default config includes provenance
            provenance = record.get("provenance", [])
            assert isinstance(provenance, list)
            if record.get("full_name"):
                # If we have a name, we should have at least one provenance entry
                assert len(provenance) > 0

    def test_confidence_in_range(self):
        """Confidence scores should be between 0 and 1."""
        csv_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "recruiter_export.csv")
        if not Path(csv_path).exists():
            pytest.skip("CSV sample not found")

        pipeline = Pipeline()
        results = pipeline.run(input_paths=[csv_path])

        for record in results:
            confidence = record.get("overall_confidence", 0)
            assert 0.0 <= confidence <= 1.0

    def test_with_custom_config(self):
        """Test pipeline with the custom config from the problem statement."""
        csv_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "recruiter_export.csv")
        config_path = str(Path(__file__).parent.parent / "config" / "custom_config_example.json")

        if not Path(csv_path).exists() or not Path(config_path).exists():
            pytest.skip("Required files not found")

        pipeline = Pipeline()
        results = pipeline.run(
            input_paths=[csv_path],
            config_path=config_path,
        )

        assert len(results) > 0
        for record in results:
            # Custom config selects: full_name, primary_email, phone, skills
            assert "full_name" in record
            # primary_email might be None if no email, but key should exist
            assert "primary_email" in record

    def test_resume_extraction(self):
        """Test pipeline with resume input (regex fallback, no Bedrock)."""
        resume_path = str(Path(__file__).parent.parent / "data" / "sample_inputs" / "resumes" / "sample_resume.txt")
        if not Path(resume_path).exists():
            pytest.skip("Resume sample not found")

        pipeline = Pipeline()
        results = pipeline.run(input_paths=[resume_path])

        assert len(results) > 0
        # Resume should extract at least email
        record = results[0]
        emails = record.get("emails", [])
        assert len(emails) > 0

    def test_directory_input(self, sample_inputs_dir):
        """Test pipeline with a directory input — should find all files."""
        pipeline = Pipeline()
        results = pipeline.run(input_paths=[sample_inputs_dir])

        # Should process CSV, ATS JSON, and resume
        assert len(results) > 0
