"""LangGraph workflow wiring the agents together (proposal, Figures 1 and 2).

Why LangGraph: the workflow needs shared state, a conditional edge (critic ->
planner or reporter) and a bounded loop. LangGraph makes each of those an
explicit, inspectable part of the graph rather than control flow hidden inside
prompts, which is why it was preferred over CrewAI in the proposal. The
structure is LangGraph's evaluator-optimiser pattern (LangChain, no date):
the Critic evaluates and the Planner optimises the search. Alternating
reasoning nodes (planner, critic) with acting nodes (retrieval) is the ReAct
pattern of interleaving reasoning with external actions (Yao et al., 2023).

The loop is bounded twice: ``route_after_critic`` stops at ``max_cycles``,
and LangGraph's ``recursion_limit`` is a hard backstop in case of a bug.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import CriticAgent, PlannerAgent, ReportingAgent, RetrievalAgent
from .processing import process
from .schemas import AuditEvent
from .state import ResearchState


def make_processing_node(min_relevance: float, limit: int):
    def processing(state: ResearchState) -> dict:
        # Re-processes the whole accumulated pool each cycle, so a paper found
        # in cycle 2 can outrank one from cycle 1 and duplicates across cycles
        # are caught.
        papers, stats = process(state.get("raw_papers", []), state["question"], min_relevance, limit)
        return {"papers": papers,
                "audit": [AuditEvent(cycle=state["cycle"], agent="processing",
                                     action="dedupe-filter-rank", detail=stats.describe())]}
    return processing


def make_router(max_cycles: int):
    def route_after_critic(state: ResearchState) -> str:
        # The stop decision is a rule, not the model's opinion of its own work:
        # self-reflection reduces but does not remove hallucination (Ji et al., 2023).
        verdict = state["verdict"]
        if verdict.sufficient or state["cycle"] >= max_cycles:
            return "report"
        return "replan"
    return route_after_critic


def build_graph(planner: PlannerAgent, retrieval: RetrievalAgent, critic: CriticAgent,
                reporter: ReportingAgent, *, max_cycles: int, min_relevance: float,
                analysis_limit: int):
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("retrieval", retrieval)
    graph.add_node("processing", make_processing_node(min_relevance, analysis_limit))
    graph.add_node("critic", critic)
    graph.add_node("reporter", reporter)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "processing")
    graph.add_edge("processing", "critic")
    graph.add_conditional_edges("critic", make_router(max_cycles),
                                {"replan": "planner", "report": "reporter"})
    graph.add_edge("reporter", END)
    return graph.compile()
