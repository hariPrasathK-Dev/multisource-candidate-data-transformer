"""
Pipeline Orchestrator — runs all 7 stages of the candidate data transformer.

Stages:
  1. DETECT & INGEST   — auto-detect source types, load files
  2. EXTRACT           — run appropriate extractors per source
  3. NORMALIZE         — normalize all fields (phones, dates, locations, skills)
  4. ENTITY RESOLUTION — match records across sources to identify same candidates
  5. MERGE             — merge matched records using conflict resolution
  6. CONFIDENCE SCORE  — compute per-field and overall confidence
  7. PROJECT & VALIDATE — apply output config, validate, emit

Never crashes: each stage wraps errors and continues with partial results.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.extractors.ats_json_extractor import ATSJSONExtractor
from src.extractors.base import BaseExtractor
from src.extractors.csv_extractor import CSVExtractor
from src.extractors.github_extractor import GitHubExtractor
from src.extractors.resume_extractor import ResumeExtractor
from src.llm.bedrock_client import BedrockClient
from src.merger.conflict_resolver import ConflictResolver
from src.merger.entity_resolver import EntityResolver
from src.models.canonical import CanonicalProfile, RawRecord, SourceType
from src.models.config import OutputConfig
from src.projection.projector import Projector
from src.scoring.confidence import ConfidenceScorer
from src.validation.validator import OutputValidator

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full 7-stage candidate data transformation pipeline.

    Usage:
        pipeline = Pipeline()
        results = pipeline.run(
            input_paths=["data/sample_inputs/"],
            config_path="config/custom_config_example.json",
        )
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        trust_config_path: Optional[str] = None,
        github_token: Optional[str] = None,
    ):
        # Initialize components
        self.bedrock_client = bedrock_client or BedrockClient()
        self.entity_resolver = EntityResolver()
        self.conflict_resolver = (
            ConflictResolver.from_config(trust_config_path)
            if trust_config_path
            else ConflictResolver()
        )
        self.confidence_scorer = ConfidenceScorer()
        self.projector = Projector()
        self.validator = OutputValidator()
        self.github_token = github_token

        # Extractors
        self._extractors: dict[SourceType, BaseExtractor] = {
            SourceType.RECRUITER_CSV: CSVExtractor(),
            SourceType.ATS_JSON: ATSJSONExtractor(),
            SourceType.GITHUB_API: GitHubExtractor(token=github_token),
            SourceType.RESUME_LLM: ResumeExtractor(bedrock_client=self.bedrock_client),
        }

    def run(
        self,
        input_paths: list[str],
        config_path: Optional[str] = None,
        schema_path: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Run the full pipeline.

        Args:
            input_paths: List of input file/directory paths
            config_path: Optional path to output config JSON
            schema_path: Optional path to custom JSON schema

        Returns:
            List of projected, validated output records
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("PIPELINE START")
        logger.info("=" * 60)

        # Load output config
        config = self._load_config(config_path)

        # Load custom schema if provided
        if schema_path:
            self.validator = OutputValidator.from_schema_file(schema_path)

        # --- Stage 1 & 2: Detect, Ingest, Extract ---
        raw_records = self._stage_extract(input_paths)
        logger.info("Stage 1-2 complete: %d raw records extracted", len(raw_records))

        if not raw_records:
            logger.warning("No records extracted from any source. Pipeline ending.")
            return []

        # --- Stage 3: Normalize ---
        # (Normalization happens during merge in conflict_resolver,
        #  but we log it as a stage for clarity)
        logger.info("Stage 3: Normalization will be applied during merge")

        # --- Stage 4: Entity Resolution ---
        clusters = self._stage_resolve(raw_records)
        logger.info("Stage 4 complete: %d candidate clusters", len(clusters))

        # --- Stage 5: Merge (with normalization) ---
        profiles = self._stage_merge(clusters)
        logger.info("Stage 5 complete: %d canonical profiles", len(profiles))

        # --- Stage 6: Confidence Scoring ---
        profiles = self._stage_score(profiles)
        logger.info("Stage 6 complete: confidence scores computed")

        # --- Stage 7: Project & Validate ---
        output = self._stage_project_and_validate(profiles, config)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            "PIPELINE COMPLETE: %d profiles in %.2fs",
            len(output), elapsed,
        )
        logger.info("=" * 60)

        return output

    def _load_config(self, config_path: Optional[str]) -> OutputConfig:
        """Load output config from file or use defaults."""
        if config_path:
            try:
                config = OutputConfig.load_from_file(config_path)
                logger.info("Loaded output config from %s", config_path)
                return config
            except Exception as e:
                logger.warning(
                    "Failed to load config from %s: %s. Using defaults.",
                    config_path, e,
                )
        return OutputConfig.default_config()

    def _stage_extract(self, input_paths: list[str]) -> list[RawRecord]:
        """Stage 1 & 2: Auto-detect sources and extract raw records."""
        all_records: list[RawRecord] = []

        # Resolve all input paths (expand directories)
        source_files = self._resolve_input_paths(input_paths)
        logger.info("Found %d source files to process", len(source_files))

        for source_path in source_files:
            try:
                source_type = BaseExtractor.detect_source_type(source_path)
                extractor = self._extractors.get(source_type)

                if extractor is None:
                    logger.warning(
                        "No extractor for source type %s (file: %s)",
                        source_type.value, source_path,
                    )
                    continue

                records = extractor.safe_extract(source_path)
                all_records.extend(records)

            except Exception as e:
                logger.error(
                    "Failed to process source %s: %s",
                    source_path, e,
                    exc_info=True,
                )

        return all_records

    def _stage_resolve(self, records: list[RawRecord]) -> list[list[RawRecord]]:
        """Stage 4: Entity resolution — group records by candidate."""
        try:
            return self.entity_resolver.resolve(records)
        except Exception as e:
            logger.error("Entity resolution failed: %s", e, exc_info=True)
            # Fallback: treat each record as a separate candidate
            return [[r] for r in records]

    def _stage_merge(self, clusters: list[list[RawRecord]]) -> list[CanonicalProfile]:
        """Stage 5: Merge each cluster into a canonical profile."""
        profiles: list[CanonicalProfile] = []

        for cluster in clusters:
            try:
                profile = self.conflict_resolver.merge(cluster)
                profiles.append(profile)
            except Exception as e:
                logger.error(
                    "Merge failed for cluster of %d records: %s",
                    len(cluster), e,
                    exc_info=True,
                )

        return profiles

    def _stage_score(self, profiles: list[CanonicalProfile]) -> list[CanonicalProfile]:
        """Stage 6: Compute confidence scores."""
        scored: list[CanonicalProfile] = []
        for profile in profiles:
            try:
                scored_profile = self.confidence_scorer.score(profile)
                scored.append(scored_profile)
            except Exception as e:
                logger.error(
                    "Scoring failed for profile %s: %s",
                    profile.candidate_id[:8], e,
                )
                scored.append(profile)  # Use unscored version
        return scored

    def _stage_project_and_validate(
        self,
        profiles: list[CanonicalProfile],
        config: OutputConfig,
    ) -> list[dict[str, Any]]:
        """Stage 7: Project output according to config and validate."""
        output: list[dict[str, Any]] = []

        for profile in profiles:
            try:
                # Project
                projected = self.projector.project(profile, config)

                # Validate
                errors = self.validator.validate(projected)
                if errors:
                    logger.warning(
                        "Validation warnings for %s: %s",
                        profile.candidate_id[:8],
                        errors,
                    )
                    # Still include the record — validation is advisory

                output.append(projected)

            except Exception as e:
                logger.error(
                    "Projection/validation failed for %s: %s",
                    profile.candidate_id[:8], e,
                )
                # Fallback: include raw profile data
                try:
                    output.append(profile.to_output_dict())
                except Exception:
                    pass

        return output

    def _resolve_input_paths(self, paths: list[str]) -> list[str]:
        """Expand directories to individual files, skip unsupported types."""
        source_files: list[str] = []
        supported_extensions = {".csv", ".json", ".pdf", ".txt"}

        for path_str in paths:
            path = Path(path_str)

            if path.is_file():
                if path.suffix.lower() in supported_extensions:
                    source_files.append(str(path))
                else:
                    logger.debug("Skipping unsupported file: %s", path)

            elif path.is_dir():
                # Recursively find all supported files
                for ext in supported_extensions:
                    for file_path in path.rglob(f"*{ext}"):
                        source_files.append(str(file_path))
            else:
                logger.warning("Path not found: %s", path)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for f in source_files:
            normalized = str(Path(f).resolve())
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(f)

        return sorted(deduped)
