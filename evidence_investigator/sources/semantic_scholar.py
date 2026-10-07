"""Semantic Scholar Graph API (https://api.semanticscholar.org/api-docs/graph).

Chosen in the proposal as the primary source because it is free, covers
peer-reviewed venues across disciplines and returns abstracts, which the
Critic needs to appraise findings and methods.
"""

from __future__ import annotations

from ..schemas import Paper
from .base import Fetcher, RateLimiter, SourceError, http_json

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,url,venue,externalIds"


def parse(data: dict) -> list[Paper]:
    if not isinstance(data, dict) or "data" not in data:
        raise SourceError("unexpected Semantic Scholar payload")
    papers = []
    for item in data["data"] or []:
        if not item.get("title") or not item.get("paperId"):
            continue
        doi = (item.get("externalIds") or {}).get("DOI")
        papers.append(Paper(
            paper_id=f"doi:{doi.lower()}" if doi else f"s2:{item['paperId']}",
            title=" ".join(item["title"].split()),
            authors=[a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
            year=item.get("year"),
            # Missing abstracts are kept as "" here; processing decides what to
            # do with them so the decision is made, and logged, in one place.
            abstract=" ".join((item.get("abstract") or "").split()),
            doi=doi.lower() if doi else None,
            url=item.get("url") or "",
            venue=item.get("venue") or None,
            source="semantic_scholar",
        ))
    return papers


class SemanticScholarSource:
    name = "semantic_scholar"

    def __init__(self, fetch: Fetcher = http_json, limiter: RateLimiter | None = None):
        self._fetch = fetch
        self._limiter = limiter or RateLimiter(1.1)

    def search(self, query: str, limit: int) -> list[Paper]:
        self._limiter.wait()
        return parse(self._fetch(API_URL, {"query": query, "limit": limit, "fields": FIELDS}))
