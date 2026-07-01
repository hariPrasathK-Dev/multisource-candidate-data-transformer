"""
Skill normalizer — maps raw skill strings to canonical names.

Two-tier approach:
  1. Local dictionary for common variations (fast, deterministic)
  2. LLM fallback for unknown skills (via Bedrock client)

Deduplicates after canonicalization.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical skill mapping — common aliases → canonical name
# ---------------------------------------------------------------------------

_SKILL_ALIASES: dict[str, str] = {
    # Programming Languages
    "js": "JavaScript",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "py": "Python",
    "python": "Python",
    "python3": "Python",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",
    "golang": "Go",
    "go": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "rb": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "kt": "Kotlin",
    "r": "R",
    "scala": "Scala",
    "perl": "Perl",
    "lua": "Lua",
    "dart": "Dart",
    "sql": "SQL",

    # Web Frameworks & Libraries
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "angular": "Angular",
    "angularjs": "Angular",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "rails": "Ruby on Rails",
    "ruby on rails": "Ruby on Rails",
    "ror": "Ruby on Rails",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",

    # Cloud & DevOps
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "microsoft azure": "Microsoft Azure",
    "docker": "Docker",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",

    # Data & ML
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "hadoop": "Hadoop",
    "kafka": "Apache Kafka",
    "apache kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    "apache airflow": "Apache Airflow",

    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "es": "Elasticsearch",
    "dynamodb": "DynamoDB",
    "cassandra": "Apache Cassandra",
    "sqlite": "SQLite",
    "oracle": "Oracle Database",
    "mssql": "Microsoft SQL Server",
    "sql server": "Microsoft SQL Server",

    # Tools & Practices
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "jira": "Jira",
    "agile": "Agile",
    "scrum": "Scrum",
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "restful": "REST APIs",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "microservices": "Microservices",
    "linux": "Linux",
    "bash": "Bash",
    "shell": "Shell Scripting",
    "shell scripting": "Shell Scripting",

    # Frontend
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    "sass": "Sass",
    "scss": "Sass",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "figma": "Figma",
    "webpack": "Webpack",
}


def normalize_skill(raw: str) -> str:
    """
    Map a raw skill string to its canonical form.

    Uses the local alias dictionary first.  If no match is found,
    returns the original string with title-case normalization.
    """
    if not raw or not isinstance(raw, str):
        return raw

    cleaned = raw.strip()
    lookup_key = cleaned.lower()

    if lookup_key in _SKILL_ALIASES:
        return _SKILL_ALIASES[lookup_key]

    # No alias found — apply title-case heuristic but preserve known casing
    # e.g. "data engineering" → "Data Engineering"
    # but don't break "PostgreSQL" into "Postgresql"
    if cleaned == cleaned.lower():
        return cleaned.title()

    return cleaned


def normalize_skills(raw_skills: list[str]) -> list[str]:
    """
    Normalize and deduplicate a list of raw skill strings.

    Returns a deduplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for s in raw_skills:
        canonical = normalize_skill(s)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


async def normalize_skill_with_llm(
    raw: str,
    bedrock_client: Optional[object] = None,
) -> tuple[str, float]:
    """
    Attempt LLM-assisted skill normalization for unknown skills.

    Returns (canonical_name, confidence).
    Falls back to local normalization if Bedrock is unavailable.
    """
    # First try local
    local = normalize_skill(raw)
    if local.lower() != raw.strip().lower():
        return local, 0.9  # High confidence from alias match

    if bedrock_client is None:
        return local, 0.3  # Low confidence — just title-cased

    # LLM path would go here — for now return local with lower confidence
    # This will be wired up when bedrock_client is implemented
    try:
        from src.llm.bedrock_client import BedrockClient
        if isinstance(bedrock_client, BedrockClient):
            result = await bedrock_client.normalize_skill(raw)
            if result:
                return result, 0.8
    except Exception as e:
        logger.warning("LLM skill normalization failed for %r: %s", raw, e)

    return local, 0.3
