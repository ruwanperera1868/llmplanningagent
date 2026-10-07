"""Functional tests: the whole LangGraph workflow, end to end, with no network.

Each test states the behaviour from the Unit 6 proposal it demonstrates.
"""

import json

import pytest

from conftest import FakeLLM, FakeSource, make_paper
from evidence_investigator.config import Settings
from evidence_investigator.investigator import Investigator, validate_question
from evidence_investigator.storage import Storage

Q = "Does generative AI improve university students' academic performance?"


def good_batch(prefix=""):
    designs = [("randomised controlled trial improved exam scores", 2024),
               ("survey found no significant difference in grades", 2023),
               ("systematic review reports mixed benefit and concern", 2025),
               ("interview study raised integrity concerns", 2024),
               ("quasi-experiment with pre-test showed improved performance", 2022),
               ("cohort correlation between use and performance", 2021)]
    return [make_paper(f"{prefix}g{i}", title=f"Generative AI and university student performance {prefix}{i}",
                       abstract=f"Generative AI and university students' academic performance: a {d}.", year=y)
            for i, (d, y) in enumerate(designs)]


def run(settings, sources, llm=None, storage=None):
    return Investigator(settings, llm=llm, sources=sources, storage=storage, use_default_llm=False).run(Q)


def test_offline_end_to_end_produces_grounded_briefing(tmp_path):
    """Goal in, plan, retrieve, process, critique, report and store: the core autonomous loop."""
    settings = Settings(db_path=tmp_path / "db", offline=True)
    with Storage(settings.db_path) as storage:
        result = Investigator(settings, storage=storage).run("How do LLM-based agents plan, act and reflect?")
        saved = storage.load_run(result.run_id)
    agents = [e.agent for e in result.audit]
    assert agents[0] == "planner" and "retrieval" in agents and "critic" in agents
    assert agents[-2:] == ["reporter", "reporter"]
    ids = {p.paper_id for p in result.papers}
    assert all(set(k.paper_ids) <= ids for k in result.briefing.key_findings)
    assert "## References" in result.markdown and "Accessed:" in result.markdown
    assert saved["briefing"] == result.briefing


def test_critic_triggers_replan_when_first_search_is_weak(settings):
    """Critic -> Planner feedback loop: weak evidence leads to a second, different search."""
    source = FakeSource([[make_paper("w1", title="Generative AI students performance")], [], [],
                         good_batch(), [], [], []])
    result = run(settings, [source])
    actions = [(e.cycle, e.action) for e in result.audit if e.agent == "planner"]
    assert actions == [(1, "plan"), (2, "re-plan")]
    assert result.verdict.sufficient
    assert "sufficient" in result.briefing.evidence_status
    assert len(set(source.queries)) == len(source.queries)  # never repeats a query


def test_loop_stops_at_max_cycles_and_flags_provisional(settings):
    """Bounded autonomy: the proposal's maximum number of search cycles."""
    source = FakeSource(lambda q: [])
    result = run(settings, [source])
    assert max(e.cycle for e in result.audit) == settings.max_cycles
    assert "provisional" in result.briefing.evidence_status


def test_one_source_failing_does_not_stop_the_run(settings):
    """Resilience: an outage is logged in the audit trail and the other source still contributes."""
    result = run(settings, [FakeSource([], name="down", fail=True), FakeSource([good_batch()], name="up")])
    assert any(e.action == "search-failed" and "down" in e.detail for e in result.audit)
    assert result.papers


def test_duplicates_across_sources_are_merged(settings):
    batch = good_batch()
    result = run(settings, [FakeSource([batch], name="a"), FakeSource([list(batch)], name="b")])
    assert len(result.papers) == len(batch)


def test_full_llm_path_with_scripted_model(settings):
    """All three LLM roles used, with outputs validated and citations grounded."""
    papers = good_batch()

    def critic(prompt):
        return json.dumps({"appraisals": [
            {"paper_id": p.paper_id, "finding": "f", "study_design": d, "stance": s, "limitations": "l"}
            for p, d, s in zip(papers, ["experimental", "survey", "review", "qualitative", "quasi-experimental",
                                        "observational"],
                               ["supports", "opposes", "mixed", "opposes", "supports", "neutral"])
            if f"paper_id: {p.paper_id}" in prompt]})

    reporter = json.dumps({"question": Q, "summary": "Evidence is mixed.", "key_findings": [
        {"statement": "An RCT found gains", "paper_ids": ["g0", "hallucinated"]}],
        "conflicting_evidence": ["Survey found no difference"], "limitations": ["Short studies"],
        "research_opportunities": ["Few long-term studies"]})
    llm = FakeLLM(planner=json.dumps({"sub_questions": ["s"], "queries": ["genai student grades"]}),
                  critic=critic, reporter=reporter)
    result = run(settings, [FakeSource([papers])], llm=llm)
    assert result.briefing.generated_by == "llm"
    assert result.briefing.key_findings[0].paper_ids == ["g0"]
    assert {a.method for a in result.appraisals.values()} == {"llm"}
    assert [m for m, _ in llm.calls] == ["planning agent", "evidence-appraisal", "reporting agent"]


def test_llm_outage_degrades_gracefully(settings):
    """Every LLM call fails, yet the agent still completes with deterministic fallbacks."""
    result = run(settings, [FakeSource([good_batch()])], llm=FakeLLM())
    assert result.briefing.generated_by == "deterministic"
    assert result.briefing.key_findings


@pytest.mark.parametrize("bad", ["", "   ", "short", "x" * 301])
def test_question_validation(bad):
    with pytest.raises(ValueError):
        validate_question(bad)
