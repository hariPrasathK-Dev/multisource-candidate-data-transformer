"""
Unit tests for all normalizers: phone, date, location, skill.
"""

import pytest

from src.normalizers.phone_normalizer import normalize_phone, normalize_phones
from src.normalizers.date_normalizer import normalize_date
from src.normalizers.location_normalizer import normalize_location
from src.normalizers.skill_normalizer import normalize_skill, normalize_skills


# =========================================================================
# Phone Normalizer Tests
# =========================================================================

class TestPhoneNormalizer:
    """Tests for E.164 phone normalization."""

    def test_indian_phone_spaced(self):
        result = normalize_phone("98765 43210", country_hint="IN")
        assert result == "+919876543210"

    def test_indian_phone_dashes(self):
        result = normalize_phone("987-654-3210", country_hint="IN")
        assert result == "+919876543210"

    def test_us_phone_with_plus_prefix(self):
        result = normalize_phone("+1-415-555-0123")
        assert result == "+14155550123"

    def test_indian_phone(self):
        result = normalize_phone("+91-98765-43210")
        assert result == "+919876543210"

    def test_indian_phone_no_prefix(self):
        result = normalize_phone("9876543210", country_hint="IN")
        assert result == "+919876543210"

    def test_garbage_returns_none(self):
        assert normalize_phone("not a phone") is None
        assert normalize_phone("N/A") is None
        assert normalize_phone("") is None
        assert normalize_phone(None) is None
        assert normalize_phone("0000000000") is None

    def test_dedup_phones(self):
        result = normalize_phones(
            ["+91-98765-43210", "98765 43210", "9876543210"],
            country_hint="IN",
        )
        # All should normalize to the same E.164 number
        assert len(result) == 1
        assert result[0] == "+919876543210"


# =========================================================================
# Date Normalizer Tests
# =========================================================================

class TestDateNormalizer:
    """Tests for YYYY-MM date normalization."""

    def test_month_name_year(self):
        assert normalize_date("Jan 2020") == "2020-01"
        assert normalize_date("January 2020") == "2020-01"

    def test_full_date(self):
        assert normalize_date("2020-01-15") == "2020-01"

    def test_year_month_format(self):
        assert normalize_date("2020-01") == "2020-01"
        assert normalize_date("2020-1") == "2020-01"

    def test_month_year_slash(self):
        assert normalize_date("01/2020") == "2020-01"

    def test_year_only(self):
        assert normalize_date("2020") == "2020-01"

    def test_garbage_returns_none(self):
        assert normalize_date("N/A") is None
        assert normalize_date("") is None
        assert normalize_date(None) is None
        assert normalize_date("present") is None
        assert normalize_date("current") is None

    def test_feb_2020(self):
        assert normalize_date("Feb 2020") == "2020-02"


# =========================================================================
# Location Normalizer Tests
# =========================================================================

class TestLocationNormalizer:
    """Tests for location normalization with ISO-3166 codes."""

    def test_city_state_country(self):
        result = normalize_location("San Francisco, CA, USA")
        assert result is not None
        assert result["city"] == "San Francisco"
        assert result["state"] == "CA"
        assert result["country"] == "US"

    def test_city_country(self):
        result = normalize_location("Bangalore, India")
        assert result is not None
        assert result["city"] == "Bangalore"
        assert result["country"] == "IN"

    def test_city_indian_state(self):
        result = normalize_location("Pune, MH")
        assert result is not None
        assert result["city"] == "Pune"
        assert result["state"] == "MH"
        assert result["country"] == "IN"

    def test_country_only(self):
        result = normalize_location("India")
        assert result is not None
        assert result["country"] == "IN"

    def test_garbage_returns_none(self):
        assert normalize_location("") is None
        assert normalize_location(None) is None
        assert normalize_location("N/A") is None
        assert normalize_location("remote") is None


# =========================================================================
# Skill Normalizer Tests
# =========================================================================

class TestSkillNormalizer:
    """Tests for canonical skill name mapping."""

    def test_common_aliases(self):
        assert normalize_skill("JS") == "JavaScript"
        assert normalize_skill("js") == "JavaScript"
        assert normalize_skill("k8s") == "Kubernetes"
        assert normalize_skill("ML") == "Machine Learning"
        assert normalize_skill("golang") == "Go"
        assert normalize_skill("react") == "React"
        assert normalize_skill("postgres") == "PostgreSQL"

    def test_already_canonical(self):
        assert normalize_skill("Python") == "Python"
        assert normalize_skill("Docker") == "Docker"

    def test_title_case_fallback(self):
        assert normalize_skill("data engineering") == "Data Engineering"

    def test_dedup_skills(self):
        result = normalize_skills(["JS", "javascript", "JavaScript", "Python", "py"])
        assert "JavaScript" in result
        assert "Python" in result
        assert len(result) == 2

    def test_preserves_casing_if_mixed(self):
        assert normalize_skill("PostgreSQL") == "PostgreSQL"  # Already proper case
