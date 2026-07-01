"""
Abstract base extractor — defines the interface every source extractor must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.canonical import RawRecord, SourceType

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Base class for all source extractors.

    Each extractor reads from one source type and produces a list of RawRecords.
    The contract:
      - extract() NEVER raises — it logs errors and returns partial/empty results
      - Every returned RawRecord has source_type and source_file set
      - Provenance entries are populated during extraction
    """

    source_type: SourceType = SourceType.UNKNOWN

    @abstractmethod
    def extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        """
        Extract raw records from the given source.

        Args:
            source_path: Path to the source file or identifier
            **kwargs: Additional options (e.g. API tokens)

        Returns:
            List of RawRecord instances (may be empty on failure)
        """
        ...

    def safe_extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        """
        Wrapper that catches all exceptions — the pipeline never crashes.
        """
        try:
            records = self.extract(source_path, **kwargs)
            logger.info(
                "Extracted %d records from %s (%s)",
                len(records), source_path, self.source_type.value,
            )
            return records
        except Exception as e:
            logger.error(
                "Extraction failed for %s (%s): %s",
                source_path, self.source_type.value, e,
                exc_info=True,
            )
            return []

    @staticmethod
    def detect_source_type(path: str) -> SourceType:
        """
        Auto-detect source type from file extension and content.
        """
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == ".csv":
            return SourceType.RECRUITER_CSV
        elif suffix == ".json":
            # Read first few bytes to distinguish ATS JSON from GitHub profiles
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read(500)
                if "applicant" in content.lower() or "candidate" in content.lower():
                    return SourceType.ATS_JSON
                elif "github" in content.lower() or "username" in content.lower():
                    return SourceType.GITHUB_API
                else:
                    return SourceType.ATS_JSON  # default for JSON
            except Exception:
                return SourceType.ATS_JSON
        elif suffix in (".pdf", ".txt"):
            return SourceType.RESUME_LLM
        else:
            return SourceType.UNKNOWN
