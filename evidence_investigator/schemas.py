"""Pydantic schemas for every message exchanged between agents.

Why Pydantic: the proposal commits to "a fixed JSON schema" for intermediate
outputs so they are inspectable and testable (section 5). Validating LLM output
against these models is also the first line of defence against malformed or
hallucinated structure: anything that does not validate is rejected and the
calling agent falls back to its deterministic method.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

StudyDesign = Literal[
    "meta-analysis", "review", "experimental", "quasi-experimental",
    "observational", "survey", "qualitative", "theoretical", "other",
]
Stance = Literal["supports", "opposes", "mixed", "neutral"]


class SearchPlan(BaseModel):
    """Planner output: how the question is decomposed and searched."""

    sub_questions: list[str] = Field(min_length=1, max_length=5)
    queries: list[str] = Field(min_length=1, max_length=5)
    rationale: str = ""

    @field_validator("queries")
    @classmethod
    def _short_queries(cls, queries: list[str]) -> list[str]:
        # Academic search APIs perform poorly on long natural-language strings,
        # so overly long LLM queries are trimmed rather than rejected outright.
        cleaned = [" ".join(q.split()[:10]) for q in queries if q.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty query is required")
        return cleaned


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    abstract: str = ""
    doi: str | None = None
    url: str = ""
    venue: str | None = None
    source: str
    relevance: float = 0.0


class Appraisal(BaseModel):
    """Critic's reading of one paper. Linked to its source by ``paper_id``."""

    paper_id: str
    finding: str = Field(max_length=600)
    study_design: StudyDesign = "other"
    stance: Stance = "neutral"
    limitations: str = Field(default="", max_length=400)
    method: Literal["llm", "heuristic"] = "heuristic"


class CriticVerdict(BaseModel):
    relevant_count: int
    designs: list[str]
    stances: dict[str, int]
    sufficient: bool
    relevant: bool
    balanced: bool
    reasons: list[str]
    suggested_queries: list[str] = []


class KeyFinding(BaseModel):
    statement: str
    paper_ids: list[str] = Field(min_length=1)


class Briefing(BaseModel):
    """The final structured research briefing (proposal, Figure 2)."""

    question: str
    summary: str
    key_findings: list[KeyFinding]
    conflicting_evidence: list[str] = []
    limitations: list[str] = []
    research_opportunities: list[str] = []
    evidence_status: str = ""
    generated_by: Literal["llm", "deterministic"] = "deterministic"


class AuditEvent(BaseModel):
    """One entry in the audit trail of searches, retained papers and re-plans."""

    cycle: int
    agent: str
    action: str
    detail: str
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
