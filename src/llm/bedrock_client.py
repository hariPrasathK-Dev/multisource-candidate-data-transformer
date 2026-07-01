"""
AWS Bedrock Client — wraps the Bedrock Runtime for Claude Sonnet.

Key features:
  - Temperature=0 for deterministic output
  - Tool-use pattern to force structured JSON responses
  - Configurable via environment variables
  - Timeout + retry with exponential backoff
  - Returns None on failure (never crashes the pipeline)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default model ID — can be overridden via env var
_DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class BedrockClient:
    """
    Client for AWS Bedrock's Claude models.

    Gracefully degrades if AWS credentials are not configured or
    the service is unavailable.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", _DEFAULT_MODEL_ID
        )
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._available = None  # None = not yet checked

    @property
    def is_available(self) -> bool:
        """Check if Bedrock is available (credentials configured, service reachable)."""
        if self._available is not None:
            return self._available

        try:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
            # Quick check — don't actually call the model
            self._available = True
            logger.info("Bedrock client initialized (model=%s, region=%s)", self.model_id, self.region)
        except Exception as e:
            logger.warning("Bedrock not available: %s. Will use fallback extraction.", e)
            self._available = False

        return self._available

    def extract_from_text(
        self,
        text: str,
        extraction_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Send text to Claude and extract structured data using tool-use pattern.

        Args:
            text: The source text (resume, profile, etc.)
            extraction_schema: JSON schema describing the expected output fields
            system_prompt: Optional system prompt to guide extraction

        Returns:
            Extracted data as a dict, or None on failure
        """
        if not self.is_available:
            return None

        if not system_prompt:
            system_prompt = (
                "You are a precise data extraction assistant. Extract candidate "
                "information from the provided text. Only extract information that "
                "is explicitly present — never invent or hallucinate data. For each "
                "extracted field, note the exact text span you extracted it from."
            )

        # Build the tool-use request
        tool_definition = {
            "name": "extract_candidate_data",
            "description": "Extract structured candidate data from text",
            "input_schema": extraction_schema,
        }

        messages = [
            {
                "role": "user",
                "content": f"Extract all candidate information from this text. Use the extract_candidate_data tool to return structured data.\n\nTEXT:\n{text}",
            }
        ]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0,
            "system": system_prompt,
            "messages": messages,
            "tools": [tool_definition],
            "tool_choice": {"type": "tool", "name": "extract_candidate_data"},
        }

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(response["body"].read())

                # Extract tool use result
                for content_block in response_body.get("content", []):
                    if content_block.get("type") == "tool_use":
                        return content_block.get("input", {})

                logger.warning("No tool_use block in Bedrock response")
                return None

            except Exception as e:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Bedrock call failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, self.max_retries + 1, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Bedrock call failed after %d attempts: %s", self.max_retries + 1, e)
                    return None

    def normalize_skill(self, raw_skill: str) -> Optional[str]:
        """
        Ask Claude to map a raw skill string to its canonical form.

        Returns the canonical skill name or None if unavailable.
        """
        if not self.is_available:
            return None

        schema = {
            "type": "object",
            "properties": {
                "canonical_name": {
                    "type": "string",
                    "description": "The standard, canonical name for this skill",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the mapping (0-1)",
                },
            },
            "required": ["canonical_name", "confidence"],
        }

        result = self.extract_from_text(
            text=f"Skill: {raw_skill}",
            extraction_schema=schema,
            system_prompt=(
                "Map the given skill to its canonical, industry-standard name. "
                "For example: 'k8s' → 'Kubernetes', 'JS' → 'JavaScript'. "
                "If it's already canonical, return it as-is."
            ),
        )

        if result and result.get("canonical_name"):
            return result["canonical_name"]
        return None


# ---------------------------------------------------------------------------
# Resume extraction schema — used by the resume extractor
# ---------------------------------------------------------------------------

RESUME_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "full_name": {
            "type": ["string", "null"],
            "description": "Candidate's full name",
        },
        "full_name_span": {
            "type": ["string", "null"],
            "description": "Exact text span where the name was found",
        },
        "emails": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All email addresses found",
        },
        "phones": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All phone numbers found (raw format)",
        },
        "location": {
            "type": ["string", "null"],
            "description": "Location as a single string (e.g. 'San Francisco, CA')",
        },
        "headline": {
            "type": ["string", "null"],
            "description": "Professional headline or title",
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All skills, technologies, and competencies mentioned",
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"], "description": "Start date"},
                    "end": {"type": ["string", "null"], "description": "End date or 'Present'"},
                    "summary": {"type": ["string", "null"]},
                },
            },
            "description": "Work experience entries",
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": ["string", "null"]},
                    "degree": {"type": ["string", "null"]},
                    "field_of_study": {"type": ["string", "null"]},
                    "end_year": {"type": ["integer", "null"]},
                },
            },
            "description": "Education entries",
        },
        "linkedin_url": {
            "type": ["string", "null"],
            "description": "LinkedIn profile URL",
        },
        "github_url": {
            "type": ["string", "null"],
            "description": "GitHub profile URL",
        },
        "portfolio_url": {
            "type": ["string", "null"],
            "description": "Portfolio or personal website URL",
        },
        "years_experience": {
            "type": ["number", "null"],
            "description": "Total years of professional experience",
        },
    },
    "required": ["full_name", "emails", "phones", "skills"],
}
