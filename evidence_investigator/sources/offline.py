"""Offline source backed by a small bundled corpus.

Why it exists: unit and functional tests must not depend on the network, and a
live demonstration can fail on a restricted network or a rate limit. Offline
mode is *only* used when explicitly requested (``--offline``). It is never a
silent fallback for a failed live search, because mixing demo data into a real
briefing would misrepresent the evidence base.
"""

from __future__ import annotations

import json
from importlib import resources

from ..schemas import Paper
from ..rendering import ET_AL
from ..text import keywords, overlap_score


def load_corpus() -> list[Paper]:
    raw = json.loads(resources.files("evidence_investigator.data")
                     .joinpath("offline_corpus.json").read_text("utf-8"))
    return [
        # The bundled author lists are abbreviated, so an explicit "et al."
        # marker stops the reference formatter presenting them as complete.
        Paper(paper_id=p["id"].lower(), title=p["title"], authors=[*p["authors"], ET_AL], year=p["year"],
              abstract=p["abstract"], url=f"https://arxiv.org/abs/{p['id'].split(':', 1)[1]}",
              venue="arXiv", source="offline")
        for p in raw["papers"]
    ]


class OfflineSource:
    name = "offline"

    def __init__(self, corpus: list[Paper] | None = None, min_score: float = 0.34):
        self._corpus = corpus if corpus is not None else load_corpus()
        # A threshold mimics a real engine returning few results for a poor
        # query; without it every search would return the whole corpus and the
        # critic's re-plan path could never be exercised offline.
        self._min_score = min_score

    def search(self, query: str, limit: int) -> list[Paper]:
        terms = keywords(query)
        scored = [(overlap_score(terms, f"{p.title} {p.title} {p.abstract}"), p) for p in self._corpus]
        hits = sorted((s for s in scored if s[0] >= self._min_score), key=lambda s: s[0], reverse=True)
        return [p for _, p in hits[:limit]]
