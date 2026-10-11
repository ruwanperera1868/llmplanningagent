"""Composition root: builds the agents from settings and runs one investigation.

All wiring happens here so every other module receives its collaborators
instead of constructing them. That is what lets tests swap in fake LLMs and
recorded sources without patching (dependency injection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .agents import CriticAgent, PlannerAgent, ReportingAgent, RetrievalAgent
from .config import Settings
from .graph import build_graph
from .llm import HF_BASE_URL, ChatCompletionsClient, LLMClient
from .rendering import render_markdown
from .schemas import Appraisal, AuditEvent, Briefing, CriticVerdict, Paper
from .sources import CrossrefSource, OfflineSource, SearchSource, SemanticScholarSource
from .storage import Storage

log = logging.getLogger(__name__)
MAX_QUESTION_CHARS = 300


@dataclass
class InvestigationResult:
    run_id: int | None
    briefing: Briefing
    papers: list[Paper]
    appraisals: dict[str, Appraisal]
    verdict: CriticVerdict | None
    audit: list[AuditEvent]
    markdown: str


def validate_question(question: str) -> str:
    question = " ".join((question or "").split())
    # Rejecting empty or very long input protects the LLM context budget and
    # limits prompt-injection surface from pasted text.
    if len(question) < 10:
        raise ValueError("Please enter a research question of at least 10 characters.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Research questions are limited to {MAX_QUESTION_CHARS} characters.")
    return question


def default_sources(settings: Settings) -> list[SearchSource]:
    if settings.offline:
        return [OfflineSource()]
    available = {"semantic_scholar": lambda: SemanticScholarSource(),
                 "crossref": lambda: CrossrefSource(mailto=settings.crossref_mailto)}
    return [available[name]() for name in settings.sources if name in available]


def default_llm(settings: Settings) -> LLMClient | None:
    # Offline mode is fully deterministic: no LLM, so results are reproducible.
    if settings.offline:
        return None
    # Without a token the hosted router cannot be used, so fall back to the
    # deterministic agents; a local endpoint (Ollama) needs no token.
    if not settings.hf_token and settings.llm_base_url.rstrip("/") == HF_BASE_URL:
        return None
    return ChatCompletionsClient(settings.hf_token, settings.hf_model,
                                 settings.llm_base_url, settings.llm_timeout)


class Investigator:
    def __init__(self, settings: Settings, *, llm: LLMClient | None = None,
                 sources: list[SearchSource] | None = None, storage: Storage | None = None,
                 use_default_llm: bool = True):
        self.settings = settings
        self.llm = llm if llm is not None or not use_default_llm else default_llm(settings)
        self.sources = sources if sources is not None else default_sources(settings)
        self.storage = storage
        self.graph = build_graph(
            PlannerAgent(self.llm),
            RetrievalAgent(self.sources, settings.results_per_query),
            CriticAgent(self.llm, settings.min_relevant_papers, settings.min_relevance),
            ReportingAgent(self.llm, storage, settings.max_cycles),
            max_cycles=settings.max_cycles, min_relevance=settings.min_relevance,
            analysis_limit=settings.max_papers_for_analysis,
        )

    @property
    def mode(self) -> str:
        sources = "+".join(s.name for s in self.sources)
        return f"{sources}; llm={self.llm.name if self.llm else 'none'}"

    def run(self, question: str) -> InvestigationResult:
        question = validate_question(question)
        run_id = self.storage.start_run(question, self.mode) if self.storage else None
        log.info("Run %s started (%s): %s", run_id, self.mode, question)
        # Each cycle visits 4 nodes; the limit is a backstop above max_cycles.
        state = self.graph.invoke(
            {"question": question, "cycle": 0, "run_id": run_id},
            {"recursion_limit": 4 * self.settings.max_cycles + 5})
        markdown = render_markdown(state["briefing"], state.get("papers", []),
                                   state.get("appraisals", {}), state.get("verdict"),
                                   state.get("audit", []))
        return InvestigationResult(run_id, state["briefing"], state.get("papers", []),
                                   state.get("appraisals", {}), state.get("verdict"),
                                   state.get("audit", []), markdown)
