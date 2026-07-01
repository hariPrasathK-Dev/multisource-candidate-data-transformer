"""
Resume Extractor — extracts candidate data from PDF/TXT resumes.

Primary path: LLM extraction via AWS Bedrock Claude Sonnet
Fallback path: Regex-based extraction for emails, phones, names

The LLM prompt requires the model to cite text spans (provenance).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.extractors.base import BaseExtractor
from src.models.canonical import (
    ExtractionMethod,
    ProvenanceEntry,
    RawRecord,
    SourceType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for fallback extraction
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)
_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}",
)
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
)
_LINKEDIN_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?",
    re.IGNORECASE,
)
_GITHUB_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?",
    re.IGNORECASE,
)


class ResumeExtractor(BaseExtractor):
    """Extracts candidate data from resume files (PDF/TXT)."""

    source_type = SourceType.RESUME_LLM

    def __init__(self, bedrock_client: Optional[Any] = None):
        self.bedrock_client = bedrock_client

    def extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        path = Path(source_path)
        if not path.exists():
            logger.warning("Resume file not found: %s", source_path)
            return []

        # Extract text from the file
        text = self._extract_text(path)
        if not text or not text.strip():
            logger.warning("No text extracted from resume: %s", source_path)
            return []

        # Try LLM extraction first, fall back to regex
        record = self._extract_with_llm(text, source_path)
        if record is None:
            logger.info("Falling back to regex extraction for %s", source_path)
            record = self._extract_with_regex(text, source_path)

        return [record] if record else []

    def _extract_text(self, path: Path) -> Optional[str]:
        """Extract text content from PDF or TXT file."""
        suffix = path.suffix.lower()

        if suffix == ".txt":
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    return path.read_text(encoding="latin-1")
                except Exception as e:
                    logger.error("Failed to read TXT file %s: %s", path, e)
                    return None

        elif suffix == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    pages = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    return "\n\n".join(pages)
            except ImportError:
                logger.warning("pdfplumber not installed — cannot parse PDF %s", path)
                return None
            except Exception as e:
                logger.error("Failed to parse PDF %s: %s", path, e)
                return None
        else:
            logger.warning("Unsupported resume format: %s", suffix)
            return None

    def _extract_with_llm(self, text: str, source_path: str) -> Optional[RawRecord]:
        """Use Bedrock LLM for intelligent extraction."""
        if self.bedrock_client is None:
            return None

        from src.llm.bedrock_client import BedrockClient, RESUME_EXTRACTION_SCHEMA

        if not isinstance(self.bedrock_client, BedrockClient):
            return None

        if not self.bedrock_client.is_available:
            return None

        try:
            result = self.bedrock_client.extract_from_text(
                text=text,
                extraction_schema=RESUME_EXTRACTION_SCHEMA,
            )
            if not result:
                return None

            # Build RawRecord from LLM extraction
            provenance_entries: list[ProvenanceEntry] = []
            data: dict[str, Any] = {
                "source_type": SourceType.RESUME_LLM,
                "source_file": source_path,
                "extraction_confidence": 0.7,
                "emails": result.get("emails", []),
                "phones": result.get("phones", []),
                "skills_raw": result.get("skills", []),
                "links": {},
                "experience_raw": result.get("experience", []),
                "education_raw": result.get("education", []),
                "extra": {},
            }

            if result.get("full_name"):
                data["full_name"] = result["full_name"]
                provenance_entries.append(ProvenanceEntry(
                    field="full_name",
                    source=SourceType.RESUME_LLM,
                    method=ExtractionMethod.LLM_EXTRACTION,
                    confidence=0.75,
                    raw_value=result["full_name"],
                    text_span=result.get("full_name_span"),
                ))

            if result.get("location"):
                data["location_raw"] = result["location"]
                provenance_entries.append(ProvenanceEntry(
                    field="location",
                    source=SourceType.RESUME_LLM,
                    method=ExtractionMethod.LLM_EXTRACTION,
                    confidence=0.65,
                    raw_value=result["location"],
                ))

            if result.get("headline"):
                data["headline"] = result["headline"]
                provenance_entries.append(ProvenanceEntry(
                    field="headline",
                    source=SourceType.RESUME_LLM,
                    method=ExtractionMethod.LLM_EXTRACTION,
                    confidence=0.7,
                    raw_value=result["headline"],
                ))

            if result.get("years_experience"):
                data["years_experience"] = result["years_experience"]

            # Links
            if result.get("linkedin_url"):
                data["links"]["linkedin"] = result["linkedin_url"]
            if result.get("github_url"):
                data["links"]["github"] = result["github_url"]
            if result.get("portfolio_url"):
                data["links"]["portfolio"] = result["portfolio_url"]

            # Provenance for array fields
            for email in data["emails"]:
                provenance_entries.append(ProvenanceEntry(
                    field="emails",
                    source=SourceType.RESUME_LLM,
                    method=ExtractionMethod.LLM_EXTRACTION,
                    confidence=0.8,
                    raw_value=email,
                ))

            if data["skills_raw"]:
                provenance_entries.append(ProvenanceEntry(
                    field="skills",
                    source=SourceType.RESUME_LLM,
                    method=ExtractionMethod.LLM_EXTRACTION,
                    confidence=0.7,
                    raw_value=", ".join(data["skills_raw"][:10]),
                ))

            data["provenance"] = provenance_entries
            return RawRecord(**data)

        except Exception as e:
            logger.warning("LLM extraction failed for %s: %s", source_path, e)
            return None

    def _extract_with_regex(self, text: str, source_path: str) -> Optional[RawRecord]:
        """Fallback: regex-based extraction for basic fields."""
        provenance_entries: list[ProvenanceEntry] = []
        data: dict[str, Any] = {
            "source_type": SourceType.RESUME_REGEX_FALLBACK,
            "source_file": source_path,
            "extraction_confidence": 0.5,
            "emails": [],
            "phones": [],
            "skills_raw": [],
            "links": {},
            "experience_raw": [],
            "education_raw": [],
            "extra": {},
        }

        # Extract emails
        emails = list(set(_EMAIL_PATTERN.findall(text)))
        if emails:
            data["emails"] = emails
            for e in emails:
                provenance_entries.append(ProvenanceEntry(
                    field="emails",
                    source=SourceType.RESUME_REGEX_FALLBACK,
                    method=ExtractionMethod.REGEX_EXTRACTION,
                    confidence=0.9,  # Emails are high-confidence regex matches
                    raw_value=e,
                ))

        # Extract phones
        phones = list(set(_PHONE_PATTERN.findall(text)))
        # Filter out unlikely phone numbers (too short or too long)
        phones = [p.strip() for p in phones if 7 <= len(re.sub(r"\D", "", p)) <= 15]
        if phones:
            data["phones"] = phones
            for p in phones:
                provenance_entries.append(ProvenanceEntry(
                    field="phones",
                    source=SourceType.RESUME_REGEX_FALLBACK,
                    method=ExtractionMethod.REGEX_EXTRACTION,
                    confidence=0.6,
                    raw_value=p,
                ))

        # Extract LinkedIn URL
        linkedin_matches = _LINKEDIN_PATTERN.findall(text)
        if linkedin_matches:
            data["links"]["linkedin"] = linkedin_matches[0]

        # Extract GitHub URL
        github_matches = _GITHUB_PATTERN.findall(text)
        if github_matches:
            data["links"]["github"] = github_matches[0]

        # Try to extract name from first line(s)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            # Heuristic: the first non-empty line that looks like a name
            for line in lines[:5]:
                # A name is typically 2-4 words, no special chars
                words = line.split()
                if 2 <= len(words) <= 4 and all(w.isalpha() or w == "." for w in words):
                    data["full_name"] = line
                    provenance_entries.append(ProvenanceEntry(
                        field="full_name",
                        source=SourceType.RESUME_REGEX_FALLBACK,
                        method=ExtractionMethod.REGEX_EXTRACTION,
                        confidence=0.4,
                        raw_value=line,
                    ))
                    break

        # Skip if we got nothing useful
        if not data.get("full_name") and not data.get("emails") and not data.get("phones"):
            return None

        data["provenance"] = provenance_entries
        return RawRecord(**data)
