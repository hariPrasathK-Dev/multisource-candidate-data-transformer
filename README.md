# Multi-Source Candidate Data Transformer

> **Eightfold Engineering Intern Assignment (Jul–Dec 2026)**
>
> A production-grade data pipeline that ingests candidate information from multiple messy sources and produces one clean, canonical profile per candidate — with provenance tracking, confidence scoring, and configurable output projection.

---

## Demo Video

🎥 **[Watch the 2-minute Demo Video Here](#)** *(Placeholder for YouTube/Loom link)*

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                MULTI-SOURCE CANDIDATE DATA TRANSFORMER                │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  CSV ────┐                                                            │
│  JSON ───┼──> DETECT ─> EXTRACT ─> NORMALIZE ─> RESOLVE ─> MERGE ──┐  │
│  GitHub ─┤              │                       │                  │  │
│  Resume ─┘              ├─> LLM                 ├─> Fuzzy          │  │
│                         └─> Regex Fallback      └─> Exact          │  │
│                                                                    v  │
│                                                                  SCORE│
│                                                                    │  │
│  OUTPUT <── EMIT <── VALIDATE <── PROJECT <────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 7-Stage Pipeline

| Stage | What it does |
|-------|-------------|
| **1. Detect & Ingest** | Auto-detect source type (CSV, JSON, resume, GitHub) from file extension/content |
| **2. Extract** | Run appropriate extractor — structured (direct mapping) or unstructured (LLM/regex) |
| **3. Normalize** | Phones → E.164, Dates → YYYY-MM, Countries → ISO-3166, Skills → canonical names |
| **4. Entity Resolution** | Match records across sources via email > phone > fuzzy name+location |
| **5. Merge** | Conflict resolution using source-trust-weighted survivorship rules |
| **6. Confidence Score** | Per-field + overall Bayesian confidence with multi-source agreement bonus |
| **7. Project & Validate** | Apply runtime config to reshape output, validate against JSON schema |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- (Optional) AWS credentials for Bedrock LLM extraction

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd EightFoldAI

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Run with default schema on all sample inputs
python -m src.main run --inputs data/sample_inputs/

# Run with custom config
python -m src.main run \
  --inputs data/sample_inputs/ \
  --config config/custom_config_example.json \
  --output output.json

# Run on specific files
python -m src.main run \
  -i data/sample_inputs/recruiter_export.csv \
  -i data/sample_inputs/ats_records.json \
  -o output.json

# Validate a config file
python -m src.main validate-config --config config/custom_config_example.json

# Verbose mode (debug logging)
python -m src.main run --inputs data/sample_inputs/ -v
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

### Produced Output

The pipeline outputs a consolidated, schema-validated JSON file. Each field includes a full provenance trail. Example output files can be found in `data/sample_outputs/`.

---

## Project Structure

```
EightFoldAI/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project metadata
├── config/
│   ├── default_schema.json             # Canonical output JSON schema
│   ├── custom_config_example.json      # Example runtime config
│   └── source_trust.json              # Source reliability weights
├── data/
│   ├── sample_inputs/                  # Test data
│   │   ├── recruiter_export.csv        # 5 candidates (structured)
│   │   ├── ats_records.json            # 3 candidates (structured, overlapping)
│   │   ├── github_profiles.json        # GitHub usernames to fetch
│   │   └── resumes/sample_resume.txt   # Resume (unstructured)
│   └── sample_outputs/                 # Generated outputs
├── src/
│   ├── main.py                         # CLI entry point
│   ├── pipeline.py                     # 7-stage orchestrator
│   ├── models/
│   │   ├── canonical.py                # Pydantic models (CanonicalProfile, RawRecord)
│   │   └── config.py                   # Runtime config models (OutputConfig)
│   ├── extractors/
│   │   ├── base.py                     # Abstract extractor interface
│   │   ├── csv_extractor.py            # Recruiter CSV parser
│   │   ├── ats_json_extractor.py       # ATS JSON parser
│   │   ├── github_extractor.py         # GitHub REST API fetcher
│   │   └── resume_extractor.py         # Resume parser (LLM + regex fallback)
│   ├── normalizers/
│   │   ├── phone_normalizer.py         # → E.164 (phonenumbers lib)
│   │   ├── date_normalizer.py          # → YYYY-MM (dateutil)
│   │   ├── location_normalizer.py      # → {city, state, country: ISO-3166}
│   │   └── skill_normalizer.py         # → canonical names (100+ aliases)
│   ├── merger/
│   │   ├── entity_resolver.py          # Union-Find entity matching
│   │   └── conflict_resolver.py        # Trust-weighted merge
│   ├── scoring/confidence.py           # Bayesian confidence scorer
│   ├── projection/projector.py         # Runtime config → output reshaping
│   ├── validation/validator.py         # JSON Schema validation
│   └── llm/bedrock_client.py           # AWS Bedrock Claude client
└── tests/
    ├── test_normalizers.py             # Normalizer unit tests
    ├── test_merger.py                  # Entity resolution + merge tests
    ├── test_projection.py              # Output projection tests
    ├── test_edge_cases.py              # Garbage input, missing field tests
    └── test_pipeline_e2e.py            # End-to-end pipeline tests
```

---

## Configurable Output

The pipeline accepts a runtime config that reshapes the output **without code changes**:

```json
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
```

### Path Syntax

| Pattern | Meaning | Example |
|---------|---------|---------|
| `field_name` | Direct field access | `full_name` |
| `array[0]` | First element | `emails[0]` |
| `array[].key` | Map over array | `skills[].name` |
| `object.key` | Nested access | `location.city` |

### `on_missing` Behavior

| Value | Effect |
|-------|--------|
| `"null"` | Include the key with a `null` value |
| `"omit"` | Silently skip the key |
| `"error"` | Raise a validation error |

---

## Key Design Decisions

### 1. Hybrid Extraction (Rule-based + LLM)
- **Structured sources** (CSV, JSON): Direct field mapping — fast, deterministic
- **Unstructured sources** (resumes): AWS Bedrock Claude Sonnet with tool-use pattern for structured extraction
- **Fallback**: Regex extraction if Bedrock is unavailable — graceful degradation, never crashes

### 2. Provenance at Every Level
Every field carries a full audit trail: `{field, source, method, confidence, raw_value, text_span}`. For LLM-extracted fields, `method = "llm_extraction"` with the cited text span.

### 3. Bayesian Confidence Scoring
```
Per-field:  source_trust × extraction_confidence × agreement_bonus
Overall:    Weighted average (name: ×2, emails: ×1.5, skills: ×1, links: ×0.5)
```

### 4. Entity Resolution via Union-Find
Three-pass matching: exact email → exact phone (E.164) → fuzzy name (Jaro-Winkler > 0.85) + location overlap.

### 5. Source Trust Hierarchy
```
ATS JSON (0.9) > Recruiter CSV (0.8) > GitHub API (0.75) > Resume LLM (0.7) > Regex Fallback (0.5)
```
Configurable via `config/source_trust.json`.

---

## Edge Cases Handled

| Scenario | Behavior |
|----------|----------|
| Garbage phone (`"not a phone"`) | Normalized to `null`, not included |
| Missing source entirely | Partial profile with reduced confidence |
| Invalid CSV row | Skipped with warning, continues processing |
| Conflicting names across sources | Highest-trust source wins, both in provenance |
| All sources missing a field | Field is `null`, never invented |
| GitHub rate limit (403) | Gracefully skipped, logged |
| Bedrock unavailable | Falls back to regex extraction |

---

## AWS Bedrock Setup (Optional)

The LLM extraction layer is optional. Without it, resumes use regex fallback (lower confidence).

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Optionally override the model
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Assumptions & Descoped

### Assumptions
- **Trust Hierarchy**: We assume ATS records are the most reliable, followed by Recruiter exports, GitHub profiles, and lastly unstructured Resumes. (This is fully configurable).
- **Primary Keys**: We assume emails and E.164-normalized phone numbers are strong enough to uniquely match candidates across sources.
- **LLM Capabilities**: We assume Claude Sonnet is capable of robust structured extraction for English resumes.

### Descoped
- **Persistent Storage**: Saving to a database (e.g., PostgreSQL/MongoDB) is descoped to keep this a focused CLI data pipeline.
- **Web Interface**: A frontend UI was descoped as a CLI was stated as acceptable and allows for better automation testing.
- **Deep GitHub Code Analysis**: We infer candidate skills by looking at their top repository languages rather than deeply cloning and parsing the AST of their code.

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | Best ecosystem for data pipelines + LLM |
| Data Models | Pydantic v2 | Schema validation, type safety, JSON serialization |
| CLI | Click | Clean CLI with help text and options |
| LLM | AWS Bedrock (boto3) | Claude Sonnet for unstructured extraction |
| Phone Normalization | phonenumbers | Google's libphonenumber port |
| Date Parsing | python-dateutil | Handles every date format |
| Country Codes | pycountry | ISO-3166 alpha-2 mapping |
| Fuzzy Matching | jellyfish | Jaro-Winkler similarity |
| JSON Schema | jsonschema | Standard validation |
| Testing | pytest | Industry standard |

---
