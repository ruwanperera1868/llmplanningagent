"""Reporting Agent: turns appraised evidence into a structured briefing.

Grounding rules (the proposal's main hallucination mitigation, section 6):
* the LLM only sees appraisals of retrieved papers, never an open prompt;
* every key finding must cite at least one retrieved ``paper_id``; citations
  to unknown ids are removed, and findings left with none are dropped;
* the "evidence status" line is written by Python from the Critic's verdict,
  so the system cannot overstate how complete its evidence is.

Retrieval-grounded generation of this kind follows Lewis et al. (2020).
Grounding is not a guarantee: retrieval-augmented systems still fail when the
retrieved evidence is incomplete or misleading (Huang et al., 2025), which is
why the evidence status is reported alongside every briefing.
"""

from __future__ import annotations

from collections import Counter

from ..llm import LLMClient, LLMError, structured_call
from ..schemas import Appraisal, AuditEvent, Briefing, CriticVerdict, KeyFinding, Paper
from ..state import ResearchState
from ..storage import Storage

SYSTEM = (
    "You are the reporting agent of an academic evidence investigator. Using ONLY the "
    "appraised papers provided, write a briefing. Every key finding must cite the "
    "paper_id(s) it comes from. Point out conflicting evidence. Research opportunities "
    "must be framed as gaps within the retrieved papers only, not the whole literature. "
    'Reply with JSON only: {"question": ..., "summary": ..., "key_findings": '
    '[{"statement": ..., "paper_ids": [...]}], "conflicting_evidence": [...], '
    '"limitations": [...], "research_opportunities": [...]}'
)
ALL_DESIGNS = ("meta-analysis", "experimental", "quasi-experimental", "observational",
               "survey", "qualitative")


def evidence_status(verdict: CriticVerdict | None, cycle: int, max_cycles: int) -> str:
    if verdict is None:
        return "No evidence was assessed."
    if verdict.sufficient:
        return f"Judged sufficient and balanced after {cycle} search cycle(s)."
    return (f"Stopped at the {max_cycles}-cycle limit with evidence judged insufficient or "
            f"unbalanced ({'; '.join(verdict.reasons)}). Treat conclusions as provisional.")


def counts(counter: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in counter.most_common()) or "none"


def deterministic_briefing(question: str, papers: list[Paper],
                           appraisals: dict[str, Appraisal]) -> Briefing:
    used = [(p, appraisals[p.paper_id]) for p in papers if p.paper_id in appraisals]
    stances = Counter(a.stance for _, a in used)
    designs = Counter(a.study_design for _, a in used)
    findings = [KeyFinding(statement=a.finding, paper_ids=[p.paper_id]) for p, a in used[:6]]
    conflicting = [f"{p.title}: {a.finding}" for p, a in used if a.stance in ("opposes", "mixed")]
    limitations = list(dict.fromkeys(a.limitations for _, a in used))[:5]
    missing = [d for d in ALL_DESIGNS if d not in designs]
    opportunities = ([f"No {d} studies were found among the retrieved papers." for d in missing[:3]]
                     or ["Retrieved designs are varied; check recency and context coverage."])
    summary = (f"{len(used)} papers were appraised. Stances: {counts(stances)}. Study designs: "
               f"{counts(designs)}. Findings below are the most question-relevant sentence of each "
               "abstract, so read the sources before relying on them.")
    return Briefing(question=question, summary=summary, key_findings=findings or [
        KeyFinding(statement="No relevant evidence was retrieved.", paper_ids=["none"])],
        conflicting_evidence=conflicting, limitations=limitations,
        research_opportunities=opportunities, generated_by="deterministic")


def ground(briefing: Briefing, valid_ids: set[str]) -> tuple[Briefing, int]:
    """Remove citations to papers that were never retrieved."""
    kept, removed = [], 0
    for finding in briefing.key_findings:
        ids = [i for i in finding.paper_ids if i in valid_ids]
        removed += len(finding.paper_ids) - len(ids)
        if ids:
            kept.append(finding.model_copy(update={"paper_ids": ids}))
    return briefing.model_copy(update={"key_findings": kept}), removed


class ReportingAgent:
    name = "reporter"

    def __init__(self, llm: LLMClient | None, storage: Storage | None, max_cycles: int):
        self.llm, self.storage, self.max_cycles = llm, storage, max_cycles

    def __call__(self, state: ResearchState) -> dict:
        question, cycle = state["question"], state.get("cycle", 0)
        papers = state.get("papers", [])
        appraisals = state.get("appraisals", {})
        briefing, how = self._write(question, papers, appraisals)
        briefing = briefing.model_copy(update={
            "question": question,
            "evidence_status": evidence_status(state.get("verdict"), cycle, self.max_cycles)})

        events = [AuditEvent(cycle=cycle, agent=self.name, action="report", detail=how)]
        if self.storage and state.get("run_id"):
            self.storage.save_result(state["run_id"], briefing, papers, appraisals,
                                     state.get("audit", []) + events)
            events.append(AuditEvent(cycle=cycle, agent=self.name, action="store",
                                     detail=f"saved run {state['run_id']} to SQLite"))
        return {"briefing": briefing, "audit": events}

    def _write(self, question: str, papers: list[Paper],
               appraisals: dict[str, Appraisal]) -> tuple[Briefing, str]:
        fallback = deterministic_briefing(question, papers, appraisals)
        if not self.llm or not papers:
            return fallback, "deterministic briefing"
        evidence = "\n".join(
            f"- paper_id: {p.paper_id} | {p.title} ({p.year}) | design: {a.study_design} | "
            f"stance: {a.stance} | finding: {a.finding} | limitations: {a.limitations}"
            for p in papers if (a := appraisals.get(p.paper_id)))
        try:
            draft = structured_call(self.llm, SYSTEM, f"Research question: {question}\n{evidence}",
                                    Briefing, max_tokens=1200)
        except LLMError as exc:
            return fallback, f"deterministic briefing after LLM failure ({exc})"
        grounded, removed = ground(draft, {p.paper_id for p in papers})
        if not grounded.key_findings:
            return fallback, "deterministic briefing: LLM findings had no valid citations"
        return grounded.model_copy(update={"generated_by": "llm"}), \
            f"LLM briefing ({self.llm.name}); {removed} invalid citations removed"
