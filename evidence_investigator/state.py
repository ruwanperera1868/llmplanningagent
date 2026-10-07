"""Shared research state passed through the LangGraph workflow.

This is the "Shared Research State" box in the proposal's architecture
(Figure 1). In BDI terms (Archibald et al., 2024) it holds the system's
*beliefs* (papers, appraisals, critic verdict), while ``pending_queries`` is the
current *intention* the Planner has committed to.

Fields that several cycles add to (papers retrieved, audit events) use an
``operator.add`` reducer so each node returns only what it produced and
LangGraph appends it. Nodes therefore never overwrite each other's work, which
keeps the audit trail complete.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .schemas import Appraisal, AuditEvent, Briefing, CriticVerdict, Paper, SearchPlan


class ResearchState(TypedDict, total=False):
    question: str
    cycle: int
    plan: SearchPlan
    pending_queries: list[str]
    queries_tried: Annotated[list[str], operator.add]
    raw_papers: Annotated[list[Paper], operator.add]
    papers: list[Paper]
    appraisals: dict[str, Appraisal]
    verdict: CriticVerdict
    briefing: Briefing
    run_id: int
    audit: Annotated[list[AuditEvent], operator.add]
