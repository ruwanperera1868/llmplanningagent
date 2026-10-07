from datetime import date

from conftest import make_paper
from evidence_investigator.rendering import ET_AL, harvard_author, in_text, reference
from evidence_investigator.schemas import AuditEvent, Briefing, KeyFinding
from evidence_investigator.storage import Storage


def test_harvard_author_initials():
    assert harvard_author("Joon Sung Park") == "Park, J.S."
    assert harvard_author("Jean-Luc Picard") == "Picard, J.L."


def test_in_text_variants():
    assert in_text(make_paper("a", authors=["Ann Smith"], year=2020)) == "Smith, 2020"
    assert in_text(make_paper("a", authors=["Ann Smith", "Bo Jones"], year=2020)) == "Smith and Jones, 2020"
    assert in_text(make_paper("a", authors=["A B", "C D", "E F"], year=None)) == "B et al., no date"
    assert in_text(make_paper("a", authors=["Ann Smith", ET_AL], year=2021)) == "Smith et al., 2021"


def test_reference_follows_cite_them_right_harvard():
    p = make_paper("doi:10.1/x", title="A study", authors=["Ann Smith", "Bo Jones"], year=2024,
                   doi="10.1/x", venue="Journal of Things")
    assert reference(p, date(2026, 10, 7)) == (
        "Smith, A. and Jones, B. (2024) 'A study', Journal of Things. Available at: "
        "https://doi.org/10.1/x (Accessed: 7 October 2026).")


def test_storage_round_trip_and_listing(tmp_path):
    storage = Storage(tmp_path / "sub" / "db.sqlite")
    run_id = storage.start_run("q?", "offline")
    assert storage.list_runs()[0]["finished_at"] is None
    b = Briefing(question="q?", summary="s", key_findings=[KeyFinding(statement="k", paper_ids=["p"])])
    storage.save_result(run_id, b, [make_paper("p", relevance=0.5)], {}, [AuditEvent(cycle=1, agent="a",
                                                                                     action="x", detail="d")])
    loaded = storage.load_run(run_id)
    assert loaded["briefing"] == b and loaded["papers"][0].relevance == 0.5 and loaded["audit"][0].agent == "a"
    assert storage.load_run(999) is None
    storage.close()
