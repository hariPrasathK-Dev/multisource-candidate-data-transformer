"""
GitHub Extractor — fetches public profiles via GitHub REST API.

Extracts: name, bio (headline), location, email, repos (skills from languages).
Rate-limit aware: 60 req/hr unauthenticated, handles 403 gracefully.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import requests

from src.extractors.base import BaseExtractor
from src.models.canonical import (
    ExtractionMethod,
    ProvenanceEntry,
    RawRecord,
    SourceType,
)

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


class GitHubExtractor(BaseExtractor):
    """Extracts candidate data from GitHub public profiles."""

    source_type = SourceType.GITHUB_API

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: Optional GitHub personal access token for higher rate limits
        """
        self.token = token
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "EightFold-Candidate-Transformer/1.0",
        })
        if token:
            self._session.headers["Authorization"] = f"token {token}"

    def extract(self, source_path: str, **kwargs: Any) -> list[RawRecord]:
        """
        Extract records from a JSON file containing GitHub usernames.

        The file should be a JSON array of usernames or objects with a "username" field.
        """
        path = Path(source_path)
        if not path.exists():
            logger.warning("GitHub profiles file not found: %s", source_path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to parse GitHub profiles JSON: %s", e)
            return []

        # Handle different formats
        usernames: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    usernames.append(item)
                elif isinstance(item, dict) and "username" in item:
                    usernames.append(item["username"])
        elif isinstance(data, dict) and "usernames" in data:
            usernames = data["usernames"]

        records = []
        for username in usernames:
            record = self._fetch_profile(username, source_path)
            if record:
                records.append(record)

        return records

    def _fetch_profile(self, username: str, source_path: str) -> Optional[RawRecord]:
        """Fetch a single GitHub user profile and their repos."""
        provenance_entries: list[ProvenanceEntry] = []

        # Fetch user profile
        try:
            resp = self._session.get(
                f"{_GITHUB_API_BASE}/users/{username}",
                timeout=10,
            )
            if resp.status_code == 403:
                logger.warning("GitHub rate limit hit for user %s", username)
                return None
            if resp.status_code == 404:
                logger.warning("GitHub user not found: %s", username)
                return None
            resp.raise_for_status()
            profile = resp.json()
        except requests.RequestException as e:
            logger.warning("Failed to fetch GitHub profile for %s: %s", username, e)
            return None

        # Fetch repos for language/skill extraction
        skills_raw: list[str] = []
        try:
            resp = self._session.get(
                f"{_GITHUB_API_BASE}/users/{username}/repos",
                params={"per_page": 100, "sort": "updated"},
                timeout=10,
            )
            if resp.status_code == 200:
                repos = resp.json()
                # Extract languages from repos
                language_counts: dict[str, int] = {}
                for repo in repos:
                    lang = repo.get("language")
                    if lang:
                        language_counts[lang] = language_counts.get(lang, 0) + 1

                # Sort by frequency and take as skills
                skills_raw = [
                    lang for lang, _ in sorted(
                        language_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]

                # Also extract topics from repos
                for repo in repos:
                    for topic in repo.get("topics", []):
                        if topic not in skills_raw:
                            skills_raw.append(topic)
        except requests.RequestException as e:
            logger.warning("Failed to fetch repos for %s: %s", username, e)

        # Build the record
        data: dict[str, Any] = {
            "source_type": SourceType.GITHUB_API,
            "source_file": source_path,
            "extraction_confidence": 0.75,
            "emails": [],
            "phones": [],
            "skills_raw": skills_raw,
            "links": {"github": profile.get("html_url", f"https://github.com/{username}")},
            "experience_raw": [],
            "education_raw": [],
            "extra": {
                "github_username": username,
                "github_bio": profile.get("bio"),
                "github_repos_count": profile.get("public_repos", 0),
                "github_followers": profile.get("followers", 0),
            },
        }

        # Name
        name = profile.get("name")
        if name:
            data["full_name"] = name
            provenance_entries.append(ProvenanceEntry(
                field="full_name",
                source=SourceType.GITHUB_API,
                method=ExtractionMethod.API_FETCH,
                confidence=0.75,
                raw_value=name,
            ))

        # Email
        email = profile.get("email")
        if email:
            data["emails"] = [email]
            provenance_entries.append(ProvenanceEntry(
                field="emails",
                source=SourceType.GITHUB_API,
                method=ExtractionMethod.API_FETCH,
                confidence=0.8,
                raw_value=email,
            ))

        # Location
        location = profile.get("location")
        if location:
            data["location_raw"] = location
            provenance_entries.append(ProvenanceEntry(
                field="location",
                source=SourceType.GITHUB_API,
                method=ExtractionMethod.API_FETCH,
                confidence=0.7,
                raw_value=location,
            ))

        # Headline from bio
        bio = profile.get("bio")
        if bio:
            data["headline"] = bio
            provenance_entries.append(ProvenanceEntry(
                field="headline",
                source=SourceType.GITHUB_API,
                method=ExtractionMethod.API_FETCH,
                confidence=0.6,
                raw_value=bio,
            ))

        # Blog/portfolio
        blog = profile.get("blog")
        if blog:
            if "linkedin.com" in blog.lower():
                data["links"]["linkedin"] = blog
            else:
                data["links"]["portfolio"] = blog

        # Skills provenance
        if skills_raw:
            provenance_entries.append(ProvenanceEntry(
                field="skills",
                source=SourceType.GITHUB_API,
                method=ExtractionMethod.API_FETCH,
                confidence=0.7,
                raw_value=", ".join(skills_raw[:10]),
            ))

        data["provenance"] = provenance_entries
        return RawRecord(**data)
