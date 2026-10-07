import pytest
import requests

from conftest import load_fixture
from evidence_investigator.sources import (CrossrefSource, OfflineSource, RateLimiter,
                                           SemanticScholarSource, SourceError)
from evidence_investigator.sources import base, crossref, semantic_scholar


class NoWait(RateLimiter):
    def __init__(self):
        super().__init__(0)


def test_semantic_scholar_parse_maps_fields_and_doi_ids():
    papers = semantic_scholar.parse(load_fixture("semantic_scholar_search.json"))
    assert [p.paper_id for p in papers] == ["doi:10.9999/example.a", "s2:s2bbb222", "s2:s2ccc333"]
    a = papers[0]
    assert a.doi == "10.9999/example.a" and a.year == 2024 and a.authors == ["Alex Example", "Sam Placeholder"]
    assert papers[1].abstract == ""  # kept, so processing can count and log it


def test_semantic_scholar_rejects_unexpected_payload():
    with pytest.raises(SourceError):
        semantic_scholar.parse({"error": "rate limited"})


def test_crossref_parse_strips_jats_and_skips_non_articles():
    papers = crossref.parse(load_fixture("crossref_works.json"))
    assert [p.paper_id for p in papers] == ["doi:10.9999/example.a", "doi:10.9999/example.d"]
    assert papers[0].abstract == "We ran a randomised controlled trial with 300 undergraduates."
    assert papers[0].url == "https://doi.org/10.9999/example.a"
    assert papers[1].authors == ["Robin Template"] and papers[1].venue == "Example Teaching Quarterly"


def test_sources_send_expected_parameters():
    seen = {}

    def fetch(url, params):
        seen[url] = params
        return load_fixture("crossref_works.json") if "crossref" in url else load_fixture(
            "semantic_scholar_search.json")

    CrossrefSource(fetch, mailto="me@example.org", limiter=NoWait()).search("ai tutoring", 5)
    SemanticScholarSource(fetch, limiter=NoWait()).search("ai tutoring", 5)
    cr = seen["https://api.crossref.org/works"]
    assert cr["rows"] == 5 and cr["mailto"] == "me@example.org" and cr["filter"] == "has-abstract:true"
    assert seen[semantic_scholar.API_URL]["limit"] == 5


def test_rate_limiter_sleeps_for_remaining_interval():
    now, slept = [100.0], []
    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=slept.append)
    limiter.wait()
    now[0] = 101.0
    limiter.wait()
    assert slept == [2.0]


def test_offline_source_ranks_and_thresholds():
    source = OfflineSource()
    hits = source.search("reflection verbal reinforcement language agents", 3)
    assert hits[0].title.startswith("Reflexion")
    assert source.search("marine biology coral reefs", 5) == []
    assert hits[0].authors[-1] == "et al."  # abbreviated author lists are marked


class FakeHTTP:
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), 0

    def __call__(self, url, params, timeout, headers):
        self.calls += 1
        status = self.statuses.pop(0)
        if status == "timeout":
            raise requests.Timeout("slow")
        resp = requests.Response()
        resp.status_code, resp._content = status, b'{"ok": true}'
        return resp


@pytest.mark.parametrize("statuses,ok", [([200], True), ([429, 200], True), (["timeout", 503, 200], True),
                                         ([429, 429, 429], False), ([404], False)])
def test_http_json_retries_transient_errors(monkeypatch, statuses, ok):
    fake = FakeHTTP(statuses)
    monkeypatch.setattr(base.requests, "get", fake)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    if ok:
        assert base.http_json("https://x", {}) == {"ok": True}
    else:
        with pytest.raises(SourceError):
            base.http_json("https://x", {})
