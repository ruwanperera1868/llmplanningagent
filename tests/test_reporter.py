import json

from conftest import FakeLLM, make_paper
from evidence_investigator.agents.reporter import (ReportingAgent, deterministic_briefing, evidence_status,
                                                   ground)
from evidence_investigator.schemas import Appraisal, Briefing, CriticVerdict, KeyFinding
from evidence_investigator.storage import Storage

Q = "Does generative AI improve student performance?"
PAPERS = [make_paper("p0"), make_paper("p1", abstract="No significant effect was found.")]
APPRAISALS = {"p0": Appraisal(paper_id="p0", finding="AI improved grades", study_design="experimental",
                              stance="supports", limitations="one course"),
              "p1": Appraisal(paper_id="p1", finding="No effect", study_design="survey", stance="opposes",
                              limitations="self-report")}


def verdict(ok):
    return CriticVerdict(relevant_count=2, designs=["experimental"], stances={}, sufficient=ok, relevant=ok,
                         balanced=ok, reasons=["only 2 relevant papers (need 5)"])


def test_ground_removes_unknown_citations():
    b = Briefing(question=Q, summary="s", key_findings=[
        KeyFinding(statement="real", paper_ids=["p0", "ghost"]), KeyFinding(statement="made up", paper_ids=["ghost"])])
    grounded, removed = ground(b, {"p0", "p1"})
    assert [k.statement for k in grounded.key_findings] == ["real"] and removed == 2
    assert grounded.key_findings[0].paper_ids == ["p0"]


def test_deterministic_briefing_reports_conflicts_and_missing_designs():
    b = deterministic_briefing(Q, PAPERS, APPRAISALS)
    assert b.generated_by == "deterministic" and len(b.key_findings) == 2
    assert any("No effect" in c for c in b.conflicting_evidence)
    assert any("meta-analysis" in o for o in b.research_opportunities)


def test_evidence_status_is_honest_when_cycle_limit_reached():
    assert "sufficient" in evidence_status(verdict(True), 1, 3)
    assert "provisional" in evidence_status(verdict(False), 3, 3)


def test_llm_briefing_used_when_grounded():
    reply = {"question": "ignored", "summary": "Mixed evidence.",
             "key_findings": [{"statement": "Grades improved in one trial", "paper_ids": ["p0"]}],
             "conflicting_evidence": ["p1 found no effect"], "limitations": [], "research_opportunities": []}
    out = ReportingAgent(FakeLLM(reporter=json.dumps(reply)), None, 3)(
        {"question": Q, "cycle": 1, "papers": PAPERS, "appraisals": APPRAISALS, "verdict": verdict(True)})
    b = out["briefing"]
    assert b.generated_by == "llm" and b.question == Q and b.summary == "Mixed evidence."


def test_llm_briefing_with_no_valid_citations_is_replaced():
    reply = {"question": Q, "summary": "s", "key_findings": [{"statement": "x", "paper_ids": ["fake"]}]}
    out = ReportingAgent(FakeLLM(reporter=json.dumps(reply)), None, 3)(
        {"question": Q, "cycle": 1, "papers": PAPERS, "appraisals": APPRAISALS, "verdict": verdict(True)})
    assert out["briefing"].generated_by == "deterministic"
    assert "no valid citations" in out["audit"][0].detail


def test_reporter_saves_to_storage():
    storage = Storage(":memory:")
    run_id = storage.start_run(Q, "test")
    out = ReportingAgent(None, storage, 3)({"question": Q, "cycle": 1, "papers": PAPERS, "appraisals": APPRAISALS,
                                            "verdict": verdict(True), "run_id": run_id, "audit": []})
    assert out["audit"][-1].action == "store"
    saved = storage.load_run(run_id)
    assert saved["briefing"] == out["briefing"] and len(saved["papers"]) == 2
