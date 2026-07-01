"""
Entity Resolver — matches candidate records across sources.

Deterministic match-key hierarchy:
  1. Email match (exact) → same person (highest confidence)
  2. Phone match (E.164 exact) → same person (high confidence)
  3. Name + Location fuzzy (Jaro-Winkler > 0.85) → possible match

Groups matched records into candidate clusters for merging.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import jellyfish

from src.models.canonical import RawRecord

logger = logging.getLogger(__name__)

# Similarity thresholds
_NAME_SIMILARITY_THRESHOLD = 0.85
_NAME_EXACT_BONUS = 0.1  # Bonus if names match exactly (case-insensitive)


class EntityResolver:
    """Groups RawRecords that refer to the same candidate."""

    def resolve(self, records: list[RawRecord]) -> list[list[RawRecord]]:
        """
        Cluster records by candidate identity.

        Returns a list of clusters — each cluster is a list of RawRecords
        that (we believe) refer to the same person.
        """
        if not records:
            return []

        # Union-Find for clustering
        parent: dict[str, str] = {}  # record_id → root_id

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Initialize each record as its own cluster
        for r in records:
            parent[r.id] = r.id

        # Build indices for matching
        email_to_records: dict[str, list[str]] = defaultdict(list)
        phone_to_records: dict[str, list[str]] = defaultdict(list)

        for r in records:
            for email in r.emails:
                email_lower = email.lower().strip()
                if email_lower:
                    email_to_records[email_lower].append(r.id)
            for phone in r.phones:
                phone_clean = phone.strip()
                if phone_clean:
                    phone_to_records[phone_clean].append(r.id)

        # --- Pass 1: Exact email match ---
        for email, record_ids in email_to_records.items():
            if len(record_ids) > 1:
                base = record_ids[0]
                for other in record_ids[1:]:
                    union(base, other)
                    logger.debug(
                        "Email match: %s links records %s and %s",
                        email, base[:8], other[:8],
                    )

        # --- Pass 2: Exact phone match (E.164) ---
        for phone, record_ids in phone_to_records.items():
            if len(record_ids) > 1:
                base = record_ids[0]
                for other in record_ids[1:]:
                    union(base, other)
                    logger.debug(
                        "Phone match: %s links records %s and %s",
                        phone, base[:8], other[:8],
                    )

        # --- Pass 3: Fuzzy name + location match ---
        # Only compare records not yet in the same cluster
        record_map = {r.id: r for r in records}
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ri, rj = records[i], records[j]
                if find(ri.id) == find(rj.id):
                    continue  # Already in same cluster

                if self._fuzzy_match(ri, rj):
                    union(ri.id, rj.id)
                    logger.debug(
                        "Fuzzy match: records %s and %s (name+location)",
                        ri.id[:8], rj.id[:8],
                    )

        # Build clusters
        clusters: dict[str, list[RawRecord]] = defaultdict(list)
        for r in records:
            root = find(r.id)
            clusters[root].append(r)

        result = list(clusters.values())
        logger.info(
            "Entity resolution: %d records → %d candidates",
            len(records), len(result),
        )
        return result

    def _fuzzy_match(self, a: RawRecord, b: RawRecord) -> bool:
        """
        Check if two records might be the same person via fuzzy name matching.

        Requires both records to have a name, and at least one of:
        - Same location (city or country)
        - Very high name similarity (> 0.92)
        """
        if not a.full_name or not b.full_name:
            return False

        name_a = a.full_name.lower().strip()
        name_b = b.full_name.lower().strip()

        # Exact match (case-insensitive)
        if name_a == name_b:
            # Still need some location overlap or other signal
            if self._locations_overlap(a.location_raw, b.location_raw):
                return True
            # Exact name match alone is risky (common names) — require email domain match
            return self._email_domains_overlap(a.emails, b.emails)

        # Jaro-Winkler similarity
        similarity = jellyfish.jaro_winkler_similarity(name_a, name_b)

        if similarity >= 0.92:
            # Very high similarity — likely same person even without location
            return True

        if similarity >= _NAME_SIMILARITY_THRESHOLD:
            # Moderate similarity — require location overlap
            return self._locations_overlap(a.location_raw, b.location_raw)

        return False

    @staticmethod
    def _locations_overlap(loc_a: Optional[str], loc_b: Optional[str]) -> bool:
        """Check if two location strings share a city or country."""
        if not loc_a or not loc_b:
            return False

        # Simple overlap check on location parts
        parts_a = {p.strip().lower() for p in loc_a.split(",") if p.strip()}
        parts_b = {p.strip().lower() for p in loc_b.split(",") if p.strip()}

        return bool(parts_a & parts_b)

    @staticmethod
    def _email_domains_overlap(emails_a: list[str], emails_b: list[str]) -> bool:
        """Check if any email addresses share a domain."""
        domains_a = set()
        for e in emails_a:
            parts = e.lower().split("@")
            if len(parts) == 2:
                domains_a.add(parts[1])

        for e in emails_b:
            parts = e.lower().split("@")
            if len(parts) == 2 and parts[1] in domains_a:
                return True

        return False
