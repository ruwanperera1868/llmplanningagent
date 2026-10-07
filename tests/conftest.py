"""Shared test doubles.

The fakes implement the same small protocols as the real LLM client and
sources, so tests exercise real agent code with no network and no token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from evidence_investigator.config import Settings
from evidence_investigator.schemas import Paper
from evidence_investigator.sources import SourceError

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text("utf-8"))


def make_paper(pid: str, title: str = "Generative AI and student performance",
               abstract: str = "Generative AI improved student performance in a randomised trial.",
               **kw) -> Paper:
    defaults = dict(authors=["Alex Example"], year=2024, url=f"https://example.org/{pid}", source="fake")
    defaults.update(kw)
    return Paper(paper_id=pid, title=title, abstract=abstract, **defaults)


class FakeLLM:
    """Routes each call to a handler chosen by the agent's system prompt."""

    name = "fake-llm"

    def __init__(self, planner: Callable | str | None = None, critic: Callable | str | None = None,
                 reporter: Callable | str | None = None):
        self.handlers = {"planning agent": planner, "evidence-appraisal": critic,
                         "reporting agent": reporter}
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        for marker, handler in self.handlers.items():
            if marker in system:
                self.calls.append((marker, prompt))
                if handler is None:
                    from evidence_investigator.llm import LLMError
                    raise LLMError("no handler configured")
                return handler(prompt) if callable(handler) else handler
        raise AssertionError(f"unexpected system prompt: {system[:60]}")


class FakeSource:
    """Returns scripted results per call; can be told to fail."""

    def __init__(self, batches: list[list[Paper]] | Callable[[str], list[Paper]], name: str = "fake",
                 fail: bool = False):
        self.name = name
        self._batches = batches
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[Paper]:
        self.queries.append(query)
        if self.fail:
            raise SourceError("simulated outage")
        if callable(self._batches):
            return self._batches(query)[:limit]
        return self._batches.pop(0)[:limit] if self._batches else []


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_path=tmp_path / "test.db", offline=True)
