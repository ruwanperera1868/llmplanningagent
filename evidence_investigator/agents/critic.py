"""Evidence Analysis / Critic Agent.

Two jobs, deliberately split between the LLM and Python:

1. *Appraise* each retained paper: main finding, study design, stance towards
   the question and limitations. This needs language understanding, so the
   LLM does it when available, with a transparent lexicon heuristic otherwise.
2. *Judge* whether the evidence is sufficient, relevant and balanced (the
   decision diamond in the proposal's Figure 2). This is done with explicit
   thresholds in Python, not by the LLM. A model asked "is this enough?" tends
   to agree with itself, and self-reflection reduces but does not remove
   hallucination (Ji et al., 2023). Rule-based stopping criteria are also
   testable and explainable to a user.

The resulting feedback loop to the Planner follows Reflexion's idea of using
verbal feedback to improve the next attempt (Shinn et al., 2023).
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ValidationError

from ..llm import LLMClient, LLMError, extract_json
from ..schemas import Appraisal, AuditEvent, CriticVerdict, Paper
from ..state import ResearchState
from ..text import keywords, overlap_score, split_sentences
from .planner import core_phrase

SYSTEM = (
    "You are the evidence-appraisal agent of an academic evidence investigator. "
    "For each paper, using ONLY its abstract, give: finding (one sentence), "
    "study_design (one of meta-analysis, review, experimental, quasi-experimental, "
    "observational, survey, qualitative, theoretical, other), stance towards the "
    "research question (supports, opposes, mixed, neutral) and limitations "
    "(one sentence; say 'not stated in abstract' if unknown). Do not invent numbers. "
    'Reply with JSON only: {"appraisals": [{"paper_id": ..., "finding": ..., '
    '"study_design": ..., "stance": ..., "limitations": ...}]}'
)

# Ordered: the first matching design wins, so specific designs are listed
# before general ones (a "systematic review and meta-analysis" is a meta-analysis).
DESIGN_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("meta-analysis", ("meta-analysis", "meta analysis")),
    ("review", ("systematic review", "scoping review", "literature review", "this survey",
                "survey reviews", "survey gives", "we review", "this review")),
    ("qualitative", ("interview", "focus group", "thematic analysis", "qualitative")),
    ("survey", ("questionnaire", "respondents", "survey of students", "survey data")),
    ("quasi-experimental", ("quasi-experiment", "pre-test", "pretest", "post-test")),
    ("experimental", ("randomised", "randomized", "controlled trial", "experiment",
                      "benchmark", "control group")),
    ("observational", ("cohort", "correlation", "observational", "longitudinal")),
    ("theoretical", ("framework", "architecture", "we propose", "introduces", "conceptual")),
]
POSITIVE = ("improve", "increase", "enhance", "benefit", "effective", "outperform",
            "better", "gain", "positive", "significantly higher", "reduces hallucination")
NEGATIVE = ("no significant", "did not", "does not", "decline", "negative", "harm", "worse",
            "struggle", "poor at", "concern", "risk", "limitation", "fail", "obstacle", "suffer")
TYPICAL_LIMITATION = {
    "meta-analysis": "Depends on the quality and heterogeneity of the included studies.",
    "review": "Synthesises others' work; no new primary data.",
    "qualitative": "Small samples limit generalisability.",
    "survey": "Self-reported data may be biased.",
    "quasi-experimental": "Without randomisation, confounding cannot be ruled out.",
    "experimental": "Controlled or benchmark settings may not reflect real-world use.",
    "observational": "Shows association rather than causation.",
    "theoretical": "Proposed design may lack independent empirical validation.",
    "other": "Design unclear from the abstract.",
}


class _AppraisalBatch(BaseModel):
    appraisals: list[Appraisal]


def heuristic_appraisal(paper: Paper, question: str) -> Appraisal:
    text = paper.abstract.lower()
    design = next((d for d, cues in DESIGN_CUES if any(c in text for c in cues)), "other")
    pos = any(c in text for c in POSITIVE)
    neg = any(c in text for c in NEGATIVE)
    stance = "mixed" if pos and neg else "supports" if pos else "opposes" if neg else "neutral"

    # The most question-relevant sentence is used as the finding; ties go to
    # later sentences because abstracts usually end with their results.
    terms = keywords(question)
    sentences = split_sentences(paper.abstract) or [paper.abstract]
    finding = max(reversed(sentences), key=lambda s: overlap_score(terms, s))
    stated = next((s for s in sentences if "limitation" in s.lower()), None)
    return Appraisal(
        paper_id=paper.paper_id, finding=finding[:600], study_design=design, stance=stance,
        limitations=(stated or f"Typical for this design: {TYPICAL_LIMITATION[design]}")[:400],
        method="heuristic",
    )


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLMClient | None, min_relevant: int = 5, min_relevance: float = 0.3):
        self.llm = llm
        self.min_relevant = min_relevant
        self.min_relevance = min_relevance

    def __call__(self, state: ResearchState) -> dict:
        cycle, question = state["cycle"], state["question"]
        papers = state.get("papers", [])
        appraisals = dict(state.get("appraisals", {}))
        events = []

        # Only papers not appraised in an earlier cycle are sent to the LLM,
        # which keeps token cost proportional to new evidence.
        new = [p for p in papers if p.paper_id not in appraisals]
        if new:
            fresh, how = self._appraise(new, question)
            appraisals.update(fresh)
            events.append(AuditEvent(cycle=cycle, agent=self.name, action="appraise",
                                     detail=f"{len(new)} papers appraised ({how})"))

        verdict = self.judge(question, papers, appraisals, state.get("queries_tried", []))
        events.append(AuditEvent(
            cycle=cycle, agent=self.name, action="verdict",
            detail=("sufficient" if verdict.sufficient else "insufficient")
            + f": {'; '.join(verdict.reasons)}"))
        return {"appraisals": appraisals, "verdict": verdict, "audit": events}

    def _appraise(self, papers: list[Paper], question: str) -> tuple[dict[str, Appraisal], str]:
        result: dict[str, Appraisal] = {}
        how = "heuristic"
        if self.llm:
            listing = "\n\n".join(
                f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract[:1200]}"
                for p in papers)
            try:
                raw = self.llm.complete(SYSTEM, f"Research question: {question}\n\n{listing}",
                                        max_tokens=250 * len(papers) + 200)
                batch = _AppraisalBatch.model_validate(extract_json(raw))
                wanted = {p.paper_id for p in papers}
                # Appraisals for ids we never sent are discarded: an LLM
                # "finding" must always trace back to a retrieved source.
                result = {a.paper_id: a.model_copy(update={"method": "llm"})
                          for a in batch.appraisals if a.paper_id in wanted}
                how = f"LLM {self.llm.name}"
            except (LLMError, ValidationError) as exc:
                how = f"heuristic after LLM failure: {type(exc).__name__}"
        for p in papers:
            if p.paper_id not in result:
                result[p.paper_id] = heuristic_appraisal(p, question)
        return result, how

    def judge(self, question: str, papers: list[Paper], appraisals: dict[str, Appraisal],
              tried: list[str]) -> CriticVerdict:
        relevant = [p for p in papers if p.relevance >= self.min_relevance]
        used = [appraisals[p.paper_id] for p in relevant if p.paper_id in appraisals]
        designs = Counter(a.study_design for a in used)
        stances = Counter(a.stance for a in used)

        enough = len(relevant) >= self.min_relevant
        # "Balanced" means the evidence is not all one kind: at least two study
        # designs, and not every paper pointing the same way. One-sided evidence
        # is exactly where a briefing could mislead, so it triggers a targeted
        # search for counter-evidence rather than being reported as settled.
        varied_design = len([d for d in designs if d != "other"]) >= 2
        # Only a uniformly *directional* stance counts as one-sided. All-neutral
        # evidence (common for descriptive papers) is not a bias to correct,
        # and treating it as one caused needless re-plans (docs/TESTING.md, R2).
        one_sided = len(used) > 1 and len(stances) == 1 and next(iter(stances)) in ("supports", "opposes")
        balanced = varied_design and not one_sided

        reasons, suggestions = [], []
        core = core_phrase(question, 4)
        if not enough:
            reasons.append(f"only {len(relevant)} relevant papers (need {self.min_relevant})")
            terms = keywords(question)
            suggestions.append(" ".join(terms[:3]))
            suggestions.append(f"{core} evidence")
        if not varied_design:
            reasons.append(f"study designs not varied: {dict(designs) or 'none'}")
            suggestions += [f"{core} randomised controlled trial", f"{core} meta-analysis"]
        if one_sided:
            reasons.append(f"all evidence has stance '{next(iter(stances))}'")
            suggestions += [f"{core} limitations challenges", f"{core} negative effects"]
        if not reasons:
            reasons.append(f"{len(relevant)} relevant papers across {len(designs)} designs "
                           f"with stances {dict(stances)}")

        tried_l = {q.lower() for q in tried}
        return CriticVerdict(
            relevant_count=len(relevant), designs=sorted(designs), stances=dict(stances),
            sufficient=enough and balanced, relevant=enough, balanced=balanced, reasons=reasons,
            suggested_queries=[s for s in dict.fromkeys(suggestions) if s.lower() not in tried_l],
        )
