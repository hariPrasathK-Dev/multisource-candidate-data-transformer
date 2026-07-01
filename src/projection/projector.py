"""
Projector — applies runtime config to reshape canonical profiles.

Takes a CanonicalProfile + OutputConfig → produces a projected output dict.
Supports field selection, renaming, array indexing, and on_missing behavior.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.models.canonical import CanonicalProfile
from src.models.config import FieldSpec, OnMissing, OutputConfig

logger = logging.getLogger(__name__)

# Pattern for bracket-indexed paths like "emails[0]" or "skills[].name"
_INDEX_PATTERN = re.compile(r"^(\w+)\[(\d*)\](?:\.(\w+))?$")
# Pattern for dot-paths like "location.city"
_DOT_PATTERN = re.compile(r"^(\w+)\.(\w+)$")


class Projector:
    """Applies output configuration to reshape canonical profiles."""

    def project(
        self,
        profile: CanonicalProfile,
        config: OutputConfig,
    ) -> dict[str, Any]:
        """
        Project a canonical profile according to the output config.

        Returns a plain dict suitable for JSON serialization.
        """
        # Get the full profile as a dict
        full_data = profile.to_output_dict(
            include_confidence=config.include_confidence
        )

        # If no fields specified, return everything
        if not config.fields:
            return full_data

        result: dict[str, Any] = {}

        for field_spec in config.fields:
            value = self._resolve_path(full_data, field_spec.source_path)

            if value is None:
                # Handle missing value
                if field_spec.required:
                    logger.warning(
                        "Required field '%s' is missing (source: '%s')",
                        field_spec.path,
                        field_spec.source_path,
                    )
                    if config.on_missing == OnMissing.ERROR:
                        raise ValueError(
                            f"Required field '{field_spec.path}' is missing"
                        )

                if config.on_missing == OnMissing.OMIT and not field_spec.required:
                    continue
                else:
                    result[field_spec.path] = None
            else:
                result[field_spec.path] = value

        # Optionally include confidence metadata
        if config.include_confidence:
            if "provenance" not in result and "provenance" in full_data:
                result["provenance"] = full_data["provenance"]
            if "overall_confidence" not in result and "overall_confidence" in full_data:
                result["overall_confidence"] = full_data["overall_confidence"]

        return result

    def _resolve_path(self, data: dict[str, Any], path: str) -> Any:
        """
        Resolve a dot-notation / bracket-notation path against a data dict.

        Supported patterns:
          - "full_name"           → data["full_name"]
          - "emails[0]"           → data["emails"][0]
          - "skills[].name"       → [s["name"] for s in data["skills"]]
          - "location.city"       → data["location"]["city"]
          - "skills[]"            → data["skills"] (full array)
        """
        if not path:
            return None

        # Simple key
        if path in data:
            return data[path]

        # Bracket-indexed: emails[0], skills[].name, skills[]
        m = _INDEX_PATTERN.match(path)
        if m:
            array_key = m.group(1)
            index_str = m.group(2)
            sub_key = m.group(3)

            arr = data.get(array_key)
            if not isinstance(arr, list):
                return None

            if index_str:
                # Specific index: emails[0]
                idx = int(index_str)
                if idx < len(arr):
                    item = arr[idx]
                    if sub_key and isinstance(item, dict):
                        return item.get(sub_key)
                    return item
                return None
            else:
                # All items: skills[].name or skills[]
                if sub_key:
                    return [
                        item.get(sub_key)
                        for item in arr
                        if isinstance(item, dict) and item.get(sub_key) is not None
                    ]
                return arr

        # Dot-path: location.city
        m = _DOT_PATTERN.match(path)
        if m:
            obj_key = m.group(1)
            sub_key = m.group(2)
            obj = data.get(obj_key)
            if isinstance(obj, dict):
                return obj.get(sub_key)
            return None

        return None


def project_profiles(
    profiles: list[CanonicalProfile],
    config: OutputConfig,
) -> list[dict[str, Any]]:
    """
    Project a list of canonical profiles according to the output config.
    """
    projector = Projector()
    return [projector.project(p, config) for p in profiles]
