"""
CLI entry point for the Multi-Source Candidate Data Transformer.

Commands:
  run              — Run the pipeline on input data
  validate-config  — Validate a runtime config file without running
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from src.pipeline import Pipeline
from src.models.config import OutputConfig


def _setup_logging(verbose: bool) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)


@click.group()
@click.version_option(version="1.0.0", prog_name="eightfold-transformer")
def cli():
    """
    🔄 Eightfold Multi-Source Candidate Data Transformer

    Merges messy candidate data from CSV, ATS JSON, GitHub, and resumes
    into one clean, canonical profile with provenance and confidence scoring.
    """
    pass


@cli.command()
@click.option(
    "--inputs", "-i",
    required=True,
    multiple=True,
    help="Input file or directory paths (can specify multiple)",
)
@click.option(
    "--config", "-c",
    default=None,
    help="Path to output configuration JSON (optional — uses defaults if omitted)",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path for the JSON results (optional — prints to stdout if omitted)",
)
@click.option(
    "--schema",
    default=None,
    help="Path to custom JSON schema for validation (optional)",
)
@click.option(
    "--trust-config",
    default=None,
    help="Path to source trust weights JSON (optional)",
)
@click.option(
    "--github-token",
    default=None,
    envvar="GITHUB_TOKEN",
    help="GitHub personal access token for higher API rate limits",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Pretty-print JSON output (default: pretty)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose/debug logging",
)
def run(
    inputs: tuple[str, ...],
    config: Optional[str],
    output: Optional[str],
    schema: Optional[str],
    trust_config: Optional[str],
    github_token: Optional[str],
    pretty: bool,
    verbose: bool,
):
    """
    Run the candidate data transformer pipeline.

    Examples:

      # Run with defaults on a directory
      python -m src.main run --inputs data/sample_inputs/

      # Run with custom config
      python -m src.main run -i data/sample_inputs/ -c config/custom_config_example.json -o output.json

      # Multiple input paths
      python -m src.main run -i data/sample_inputs/recruiter_export.csv -i data/sample_inputs/ats_records.json
    """
    _setup_logging(verbose)
    logger = logging.getLogger("cli")

    # Auto-detect trust config if not specified
    if not trust_config:
        default_trust = Path("config/source_trust.json")
        if default_trust.exists():
            trust_config = str(default_trust)

    # Initialize pipeline
    pipeline = Pipeline(
        trust_config_path=trust_config,
        github_token=github_token,
    )

    # Run the pipeline
    try:
        results = pipeline.run(
            input_paths=list(inputs),
            config_path=config,
            schema_path=schema,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)

    if not results:
        click.echo("⚠️  No candidates found in the input data.", err=True)
        sys.exit(0)

    # Format output
    indent = 2 if pretty else None
    json_output = json.dumps(results, indent=indent, ensure_ascii=False, default=str)

    # Write to file and/or stdout
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        click.echo(f"✅ {len(results)} candidate(s) written to {output}")
    else:
        click.echo(json_output)

    click.echo(f"\n📊 Summary: {len(results)} canonical profile(s) generated", err=True)


@cli.command("validate-config")
@click.option(
    "--config", "-c",
    required=True,
    help="Path to the output configuration JSON to validate",
)
def validate_config(config: str):
    """
    Validate a runtime configuration file without running the pipeline.

    Checks: file exists, valid JSON, valid OutputConfig schema,
    no duplicate field paths, valid on_missing values.
    """
    _setup_logging(False)

    try:
        output_config = OutputConfig.load_from_file(config)
        click.echo(f"✅ Config is valid: {config}")
        click.echo(f"   Fields: {len(output_config.fields)}")
        click.echo(f"   Include confidence: {output_config.include_confidence}")
        click.echo(f"   On missing: {output_config.on_missing.value}")

        if output_config.fields:
            click.echo("   Field list:")
            for f in output_config.fields:
                src = f" ← {f.source_path}" if f.source_path != f.path else ""
                req = " (required)" if f.required else ""
                click.echo(f"     • {f.path}{src}{req}")

    except FileNotFoundError as e:
        click.echo(f"❌ Config file not found: {e}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Config validation failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
