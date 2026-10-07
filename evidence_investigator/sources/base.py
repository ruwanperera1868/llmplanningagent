"""Shared plumbing for academic search sources."""

from __future__ import annotations

import time
from typing import Callable, Protocol

import requests

from ..schemas import Paper

USER_AGENT = "AcademicEvidenceInvestigator/1.0 (MSc coursework prototype)"

# A fetcher takes a URL and query parameters and returns decoded JSON. Sources
# receive one by injection so unit tests can replay recorded API responses.
Fetcher = Callable[[str, dict], dict]


class SourceError(RuntimeError):
    """A source could not be reached or returned unusable data."""


class SearchSource(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[Paper]: ...


class RateLimiter:
    """Minimum interval between requests to one provider.

    Semantic Scholar's unauthenticated pool is shared and throttled, and
    Crossref asks clients to be "polite". Respecting provider limits is part of
    the professional and legal responsibilities the module asks us to consider,
    and it also avoids HTTP 429 errors that would weaken retrieval.
    """

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def http_json(url: str, params: dict, timeout: float = 20.0, retries: int = 2) -> dict:
    """GET JSON with a short back-off on 429/5xx, which free tiers return often."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
            if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise SourceError(f"{url} failed: {exc}") from exc
            time.sleep(2 ** (attempt + 1))
    raise SourceError(f"{url} failed after retries")
