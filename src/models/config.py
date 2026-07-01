"""
Runtime configuration models for the configurable output projection layer.

The output config lets consumers reshape the canonical profile at runtime
without code changes — selecting fields, renaming paths, setting normalization
overrides, and choosing what happens when a field is missing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class OnMissing(str, Enum):
    """Behaviour when a projected field is missing from the canonical record."""
    NULL = "null"    # include the key with a null value
    OMIT = "omit"    # silently skip the key
    ERROR = "error"  # raise a validation error


class FieldSpec(BaseModel):
    """
    Specification for one field in the projected output.

    Example (from the problem statement):
        {
            "path": "full_name",
            "type": "string",
            "required": true
        }
    or with renaming:
        {
            "path": "primary_email",
            "from": "emails[0]",
            "type": "string",
            "required": true
        }
    """
    path: str = Field(
        ...,
        description="Output field name (the key in the emitted JSON)",
    )
    from_field: Optional[str] = Field(
        default=None,
        alias="from",
        description=(
            "Source path in the canonical record.  Supports dot-notation "
            "and bracket indexing, e.g. 'emails[0]' or 'skills[].name'.  "
            "If omitted, defaults to `path`."
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="Expected type hint (string, number, array, object). Informational.",
    )
    required: bool = Field(
        default=False,
        description="If true and the field is missing, an error is raised regardless of on_missing.",
    )
    normalize: Optional[str] = Field(
        default=None,
        description="Override normalization format, e.g. 'E.164', 'canonical', 'ISO-3166'.",
    )

    class Config:
        populate_by_name = True  # accept both 'from' and 'from_field'

    @property
    def source_path(self) -> str:
        """The canonical-record path to read from."""
        return self.from_field if self.from_field else self.path


class OutputConfig(BaseModel):
    """
    Top-level runtime configuration for output projection.

    Example (from the Eightfold problem statement):
        {
            "fields": [
                {"path": "full_name", "type": "string", "required": true},
                {"path": "primary_email", "from": "emails[0]", "type": "string", "required": true},
                {"path": "phone", "from": "phones[0]", "type": "string", "normalize": "E.164"},
                {"path": "skills", "from": "skills[].name", "type": "string[]", "normalize": "canonical"}
            ],
            "include_confidence": true,
            "on_missing": "null"
        }
    """
    fields: list[FieldSpec] = Field(
        default_factory=list,
        description="Ordered list of fields to include in the output.",
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include provenance and confidence in the output.",
    )
    on_missing: OnMissing = Field(
        default=OnMissing.NULL,
        description="Default behaviour when a projected field is absent.",
    )

    @field_validator("fields", mode="after")
    @classmethod
    def check_unique_paths(cls, v: list[FieldSpec]) -> list[FieldSpec]:
        paths = [f.path for f in v]
        dupes = [p for p in paths if paths.count(p) > 1]
        if dupes:
            raise ValueError(f"Duplicate output paths in config: {set(dupes)}")
        return v

    @classmethod
    def load_from_file(cls, path: str) -> "OutputConfig":
        """Load and validate a config from a JSON file."""
        import json
        from pathlib import Path

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return cls.model_validate(raw)

    @classmethod
    def default_config(cls) -> "OutputConfig":
        """Return a config that passes through all canonical fields."""
        return cls(
            fields=[
                FieldSpec(path="candidate_id", type="string", required=True),
                FieldSpec(path="full_name", type="string", required=True),
                FieldSpec(path="emails", type="string[]"),
                FieldSpec(path="phones", type="string[]"),
                FieldSpec(path="location", type="object"),
                FieldSpec(path="headline", type="string"),
                FieldSpec(path="links", type="object"),
                FieldSpec(path="years_experience", type="number"),
                FieldSpec(path="skills", type="object[]"),
                FieldSpec(path="experience", type="object[]"),
                FieldSpec(path="education", type="object[]"),
                FieldSpec(path="provenance", type="object[]"),
                FieldSpec(path="overall_confidence", type="number"),
            ],
            include_confidence=True,
            on_missing=OnMissing.NULL,
        )
