"""Planner / Supervisor Agent.

Responsibilities (proposal, section 3): interpret the research question,
decompose it into sub-questions and generate search queries; on later cycles,
revise the strategy using the Critic's feedback. This is the agent that holds
the *intention* in the BDI sense and revises it when beliefs change
(Archibald et al., 2024; Wooldridge, 2009).

The LLM proposes a plan, but Python validates it (schema, query length,
de-duplication against queries already tried). If the LLM is unavailable or
returns an invalid plan, a transparent keyword-based planner is used, so the
agent stays autonomous without a model.
"""

from __future__ import annotations

from ..llm import LLMClient, LLMError, structured_call
from ..schemas import AuditEvent, SearchPlan
from ..state import ResearchState
from ..text import keywords

SYSTEM = (
    "You are the planning agent of an academic evidence investigator. "
    "Break the research question into 2-4 sub-questions and write 2-4 short "
    "academic database search queries (3-8 words, no boolean operators). "
    "Reply with JSON only: {\"sub_questions\": [...], \"queries\": [...], \"rationale\": \"...\"}"
)


def core_phrase(question: str, max_terms: int = 6) -> str:
    return " ".join(keywords(question)[:max_terms])


def fallback_plan(question: str) -> SearchPlan:
    """Rule-based plan: the core topic, plus design-targeted variants.

    Adding "systematic review" and "empirical study" deliberately pulls in
    both secondary and primary evidence from the first cycle, which makes a
    balanced evidence base (one of the Critic's checks) more likely early.
    """
    core = core_phrase(question)
    return SearchPlan(
        sub_questions=[
            f"What empirical evidence exists on {core}?",
            f"Which study designs have examined {core}?",
            f"What limitations or contrary findings are reported on {core}?",
        ],
        queries=[core, f"{core} systematic review", f"{core} empirical study"],
        rationale="Keyword-based plan (LLM unavailable): core topic plus review and primary-study variants.",
    )


def novel(queries: list[str], tried: list[str]) -> list[str]:
    seen = {q.lower().strip() for q in tried}
    out = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def __call__(self, state: ResearchState) -> dict:
        cycle = state.get("cycle", 0) + 1
        question = state["question"]
        tried = state.get("queries_tried", [])
        verdict = state.get("verdict")

        if verdict is None:
            plan, how = self._initial_plan(question)
            queries = plan.queries
        else:
            plan = state["plan"]
            queries, how = self._replan(question, verdict.reasons, verdict.suggested_queries, tried)

        queries = novel(queries, tried)
        if not queries:
            # Every idea has been tried already: broaden by dropping the most
            # specific trailing term rather than repeating a search.
            terms = keywords(question)
            queries = novel([" ".join(terms[: max(2, len(terms) - cycle)])], tried) or [question]

        event = AuditEvent(cycle=cycle, agent=self.name,
                           action="plan" if verdict is None else "re-plan",
                           detail=f"{how}; queries: {queries}")
        return {"cycle": cycle, "plan": plan, "pending_queries": queries, "audit": [event]}

    def _initial_plan(self, question: str) -> tuple[SearchPlan, str]:
        if self.llm:
            try:
                return structured_call(self.llm, SYSTEM, f"Research question: {question}", SearchPlan), \
                    f"LLM plan ({self.llm.name})"
            except LLMError as exc:
                return fallback_plan(question), f"LLM plan rejected ({exc}); used keyword plan"
        return fallback_plan(question), "keyword plan (offline)"

    def _replan(self, question: str, reasons: list[str], suggested: list[str],
                tried: list[str]) -> tuple[list[str], str]:
        if self.llm:
            prompt = (f"Research question: {question}\nQueries already tried: {tried}\n"
                      f"Critic feedback: {reasons}\nWrite NEW queries that address the feedback.")
            try:
                plan = structured_call(self.llm, SYSTEM, prompt, SearchPlan)
                fresh = novel(plan.queries, tried)
                if fresh:
                    return fresh, f"LLM re-plan from critic feedback ({len(reasons)} issues)"
            except LLMError as exc:
                return suggested, f"LLM re-plan failed ({exc}); used critic's suggested queries"
        return suggested, "used critic's suggested queries"
