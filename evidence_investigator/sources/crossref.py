"""Crossref REST API (https://api.crossref.org).

Why Crossref as well: it is the DOI registration agency for most journals, so
its metadata (DOI, venue, year) is authoritative. Using a second source is the
proposal's mitigation for biased or incomplete retrieval, and shared DOIs let
processing merge records across the two sources reliably.
"""

from __future__ import annotations

import re

from ..schemas import Paper
from .base import Fetcher, RateLimiter, SourceError, http_json

API_URL = "https://api.crossref.org/works"
SELECT = "DOI,title,author,issued,abstract,URL,container-title,type"
_TAG = re.compile(r"<[^>]+>")


def _clean_abstract(raw: str | None) -> str:
    # Crossref abstracts are JATS XML fragments (<jats:p> ...). Stripping tags is
    # enough for appraisal; a full XML parse adds nothing for plain text.
    text = _TAG.sub(" ", raw or "")
    text = re.sub(r"^\s*abstract\s*", "", " ".join(text.split()), flags=re.I)
    return text


def parse(data: dict) -> list[Paper]:
    try:
        items = data["message"]["items"]
    except (KeyError, TypeError) as exc:
        raise SourceError("unexpected Crossref payload") from exc
    papers = []
    for item in items:
        title = " ".join((item.get("title") or [""])[0].split())
        doi = (item.get("DOI") or "").lower()
        # Crossref also indexes peer reviews, components and datasets; only
        # scholarly works are useful evidence.
        if not title or not doi or item.get("type") not in {
            "journal-article", "proceedings-article", "book-chapter", "posted-content", "report"
        }:
            continue
        date_parts = (item.get("issued") or {}).get("date-parts") or [[None]]
        authors = [
            " ".join(p for p in (a.get("given"), a.get("family")) if p)
            for a in item.get("author") or []
        ]
        papers.append(Paper(
            paper_id=f"doi:{doi}",
            title=title,
            authors=[a for a in authors if a],
            year=date_parts[0][0] if date_parts and date_parts[0] else None,
            abstract=_clean_abstract(item.get("abstract")),
            doi=doi,
            url=f"https://doi.org/{doi}",
            venue=(item.get("container-title") or [None])[0],
            source="crossref",
        ))
    return papers


class CrossrefSource:
    name = "crossref"

    def __init__(self, fetch: Fetcher = http_json, mailto: str = "",
                 limiter: RateLimiter | None = None):
        self._fetch = fetch
        self._mailto = mailto
        self._limiter = limiter or RateLimiter(1.0)

    def search(self, query: str, limit: int) -> list[Paper]:
        params = {"query.bibliographic": query, "rows": limit, "select": SELECT,
                  # Restrict to items that have an abstract to appraise.
                  "filter": "has-abstract:true"}
        if self._mailto:
            # Identifies us to Crossref's "polite pool", as their etiquette asks.
            params["mailto"] = self._mailto
        self._limiter.wait()
        return parse(self._fetch(API_URL, params))
