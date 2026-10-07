"""Retrieval Agent: runs the planned queries against every configured source.

Why each source is queried independently with errors caught per call: one
provider being down or rate-limited should reduce coverage, not end the run.
Every failure is written to the audit trail so the briefing can be read with
knowledge of what was actually searched.
"""

from __future__ import annotations

from ..schemas import AuditEvent
from ..sources import SearchSource, SourceError
from ..state import ResearchState


class RetrievalAgent:
    name = "retrieval"

    def __init__(self, sources: list[SearchSource], per_query: int = 10):
        self.sources = sources
        self.per_query = per_query

    def __call__(self, state: ResearchState) -> dict:
        cycle = state["cycle"]
        queries = state.get("pending_queries", [])
        found, events = [], []
        for query in queries:
            for source in self.sources:
                try:
                    papers = source.search(query, self.per_query)
                    found.extend(papers)
                    events.append(AuditEvent(cycle=cycle, agent=self.name, action="search",
                                             detail=f"{source.name} '{query}' -> {len(papers)} records"))
                except SourceError as exc:
                    events.append(AuditEvent(cycle=cycle, agent=self.name, action="search-failed",
                                             detail=f"{source.name} '{query}': {exc}"))
        return {"raw_papers": found, "queries_tried": queries, "pending_queries": [], "audit": events}
