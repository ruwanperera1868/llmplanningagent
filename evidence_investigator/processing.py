"""Deterministic "Python Processing" stage: deduplicate, clean, filter, rank.

The proposal deliberately assigns these steps to conventional code rather than
the LLM (section 3 and Conclusion): they have a single correct answer, so an
LLM would add cost, latency and the risk of silently dropping or inventing
records, while giving nothing in return. pandas is used because duplicate
removal across two sources is naturally a table operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .schemas import Paper
from .text import keywords, overlap_score


@dataclass
class ProcessingStats:
    received: int
    duplicates_removed: int
    missing_abstract: int
    below_relevance: int
    retained: int

    def describe(self) -> str:
        return (f"received {self.received}, removed {self.duplicates_removed} duplicates, "
                f"{self.missing_abstract} without abstract, {self.below_relevance} below relevance "
                f"threshold; retained {self.retained}")


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def first_author_surname(authors: list[str]) -> str:
    return authors[0].split()[-1].lower() if authors and authors[0].split() else ""


def process(papers: list[Paper], question: str, min_relevance: float,
            limit: int) -> tuple[list[Paper], ProcessingStats]:
    if not papers:
        return [], ProcessingStats(0, 0, 0, 0, 0)

    df = pd.DataFrame([p.model_dump() for p in papers])
    df["has_abstract"] = df["abstract"].str.len() > 0
    # Sorting so the most complete record of each paper comes first means
    # drop_duplicates keeps the best version (with abstract, from the richer source).
    df = df.sort_values(["has_abstract", "source"], ascending=[False, False])
    received = len(df)

    # Match on DOI first (exact identity), then on title plus first-author
    # surname for records with no DOI, as specified in the proposal (section 6).
    # Title alone is not enough: short generic titles occasionally collide.
    with_doi = df[df["doi"].notna()].drop_duplicates(subset="doi")
    without_doi = df[df["doi"].isna()]
    df = pd.concat([with_doi, without_doi])
    df["match_key"] = df["title"].map(title_key) + "|" + df["authors"].map(first_author_surname)
    df = df.drop_duplicates(subset="match_key")
    duplicates = received - len(df)

    missing = int((~df["has_abstract"]).sum())
    df = df[df["has_abstract"]].copy()

    terms = keywords(question)
    df["relevance"] = [overlap_score(terms, f"{t} {a}") for t, a in zip(df["title"], df["abstract"])]
    below = int((df["relevance"] < min_relevance).sum())
    df = df[df["relevance"] >= min_relevance]

    # Ties on relevance are broken by recency: newer work is more likely to
    # reflect the current state of a fast-moving field such as generative AI.
    df = df.sort_values(["relevance", "year"], ascending=[False, False], na_position="last")
    df = df.head(limit)

    columns = list(Paper.model_fields)
    kept = [Paper(**{k: (None if pd.isna(v) else v) if not isinstance(v, list) else v
                     for k, v in row.items()})
            for row in df[columns].to_dict("records")]
    return kept, ProcessingStats(received, duplicates, missing, below, len(kept))
