"""
Unit tests for the output projection layer.
"""

import pytest

from src.models.canonical import (
    CanonicalProfile,
    Links,
    Location,
    Skill,
    ExperienceEntry,
)
from src.models.config import FieldSpec, OnMissing, OutputConfig
from src.projection.projector import Projector


@pytest.fixture
def sample_profile():
    """Create a sample CanonicalProfile for testing."""
    return CanonicalProfile(
        candidate_id="test-123",
        full_name="John Doe",
        emails=["john@example.com", "john.doe@corp.com"],
        phones=["+14155550123"],
        location=Location(city="San Francisco", state="CA", country="US"),
        headline="Staff Engineer",
        links=Links(
            linkedin="https://linkedin.com/in/johndoe",
            github="https://github.com/johndoe",
        ),
        years_experience=12.0,
        skills=[
            Skill(name="Python", confidence=0.9),
            Skill(name="Go", confidence=0.85),
            Skill(name="React", confidence=0.7),
        ],
        experience=[
            ExperienceEntry(
                company="TechCorp",
                title="Staff Engineer",
                start="2019-06",
                end=None,
            ),
        ],
        overall_confidence=0.85,
    )


class TestProjector:
    """Tests for the output projector."""

    def test_simple_field_selection(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(path="full_name", type="string", required=True),
                FieldSpec(path="emails", type="string[]"),
            ],
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert result["full_name"] == "John Doe"
        assert "john@example.com" in result["emails"]
        assert "phones" not in result  # Not in config

    def test_field_renaming(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(
                    path="primary_email",
                    **{"from": "emails[0]"},
                    type="string",
                ),
            ],
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert "primary_email" in result
        assert result["primary_email"] in sample_profile.emails

    def test_array_index_access(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(
                    path="phone",
                    **{"from": "phones[0]"},
                    type="string",
                ),
            ],
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert result["phone"] == "+14155550123"

    def test_array_map_access(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(
                    path="skill_names",
                    **{"from": "skills[].name"},
                    type="string[]",
                ),
            ],
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert "Python" in result["skill_names"]
        assert "Go" in result["skill_names"]
        assert len(result["skill_names"]) == 3

    def test_dot_path_access(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(
                    path="country",
                    **{"from": "location.country"},
                    type="string",
                ),
            ],
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert result["country"] == "US"

    def test_on_missing_null(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(path="nonexistent_field", type="string"),
            ],
            on_missing=OnMissing.NULL,
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert "nonexistent_field" in result
        assert result["nonexistent_field"] is None

    def test_on_missing_omit(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(path="nonexistent_field", type="string"),
            ],
            on_missing=OnMissing.OMIT,
            include_confidence=False,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert "nonexistent_field" not in result

    def test_on_missing_error_required(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(path="nonexistent_field", type="string", required=True),
            ],
            on_missing=OnMissing.ERROR,
            include_confidence=False,
        )

        projector = Projector()
        with pytest.raises(ValueError, match="Required field"):
            projector.project(sample_profile, config)

    def test_include_confidence(self, sample_profile):
        config = OutputConfig(
            fields=[
                FieldSpec(path="full_name", type="string"),
            ],
            include_confidence=True,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert "overall_confidence" in result

    def test_custom_config_from_problem_statement(self, sample_profile):
        """Test the exact config from the Eightfold problem statement."""
        config = OutputConfig(
            fields=[
                FieldSpec(path="full_name", type="string", required=True),
                FieldSpec(
                    path="primary_email",
                    **{"from": "emails[0]"},
                    type="string",
                    required=True,
                ),
                FieldSpec(
                    path="phone",
                    **{"from": "phones[0]"},
                    type="string",
                    normalize="E.164",
                ),
                FieldSpec(
                    path="skills",
                    **{"from": "skills[].name"},
                    type="string[]",
                    normalize="canonical",
                ),
            ],
            include_confidence=True,
            on_missing=OnMissing.NULL,
        )

        projector = Projector()
        result = projector.project(sample_profile, config)

        assert result["full_name"] == "John Doe"
        assert result["primary_email"] in sample_profile.emails
        assert result["phone"] == "+14155550123"
        assert "Python" in result["skills"]
