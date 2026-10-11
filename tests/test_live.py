"""Live smoke tests against the real APIs. Skipped unless RUN_LIVE_TESTS=1.

Kept out of the default run because network calls are slow, rate-limited and
non-deterministic. They confirm the parsers still match the real services.
"""

import os

import pytest

from evidence_investigator.config import Settings
from evidence_investigator.investigator import Investigator
from evidence_investigator.investigator import default_llm
from evidence_investigator.sources import CrossrefSource, SemanticScholarSource

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1")]
Q = "generative AI university student academic performance"


def test_semantic_scholar_live():
    assert SemanticScholarSource().search(Q, 5)


def test_crossref_live():
    papers = CrossrefSource(mailto=os.environ.get("CROSSREF_MAILTO", "")).search(Q, 5)
    assert papers and all(p.doi for p in papers)


@pytest.mark.skipif(not (os.environ.get("HF_TOKEN") or os.environ.get("LLM_BASE_URL")),
                    reason="set HF_TOKEN or LLM_BASE_URL")
def test_llm_live():
    # Uses whichever endpoint is configured: Hugging Face, or a local Ollama.
    client = default_llm(Settings.from_env())
    assert client.complete("Reply with the word OK.", "Say OK", max_tokens=5)


def test_full_live_run(tmp_path):
    result = Investigator(Settings.from_env(db_path=tmp_path / "db")).run(
        "Does generative AI improve university students' academic performance?")
    assert result.papers and result.briefing.key_findings
