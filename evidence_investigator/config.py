"""Runtime settings.

Settings come from environment variables (optionally via a local ``.env`` that
is git-ignored) so that API credentials are never committed to the public
GitHub repository, addressing the credential-protection risk identified in the
Unit 6 proposal (section 6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    hf_token: str = ""
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    crossref_mailto: str = ""
    db_path: Path = Path("data/investigator.db")
    offline: bool = False

    # The proposal requires "a maximum number of search cycles" so the
    # critic-to-planner loop cannot run forever (section 5). Three cycles lets
    # the critic ask for more evidence twice, which in testing was enough to
    # fix most weak first searches without long waits.
    max_cycles: int = 3

    # Sufficiency thresholds used by the Critic. Kept here rather than hard-coded
    # so they can be tuned and their effect demonstrated.
    min_relevant_papers: int = 5
    min_relevance: float = 0.3
    results_per_query: int = 10

    # Only the top-ranked papers are sent to the LLM. This is the
    # "filter before LLM analysis" mitigation for context and cost limits.
    max_papers_for_analysis: int = 12

    sources: tuple[str, ...] = field(default=("semantic_scholar", "crossref"))

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values = dict(
            hf_token=os.environ.get("HF_TOKEN", "").strip(),
            hf_model=os.environ.get("HF_MODEL", cls.hf_model),
            crossref_mailto=os.environ.get("CROSSREF_MAILTO", "").strip(),
            db_path=Path(os.environ.get("INVESTIGATOR_DB", cls.db_path)),
            offline=os.environ.get("INVESTIGATOR_OFFLINE", "") == "1",
            max_cycles=_int("MAX_CYCLES", cls.max_cycles),
            min_relevant_papers=_int("MIN_RELEVANT_PAPERS", cls.min_relevant_papers),
        )
        values.update(overrides)
        return cls(**values)
