import json

import pytest

from conftest import FakeLLM, make_paper
from evidence_investigator.agents.critic import CriticAgent, heuristic_appraisal
from evidence_investigator.schemas import Appraisal

Q = "Does generative AI improve student performance?"


@pytest.mark.parametrize("abstract,design,stance", [
    ("A systematic review and meta-analysis of 40 trials showed AI improved scores.", "meta-analysis", "supports"),
    ("We conducted a randomised controlled trial; AI tutoring improved grades.", "experimental", "supports"),
    ("Interviews with lecturers raised concerns about integrity.", "qualitative", "opposes"),
    ("A questionnaire of 500 respondents found no significant difference but some benefit.", "survey", "mixed"),
    ("Students used chatbots in class.", "other", "neutral"),
])
def test_heuristic_appraisal(abstract, design, stance):
    a = heuristic_appraisal(make_paper("p", abstract=abstract), Q)
    assert (a.study_design, a.stance, a.method) == (design, stance, "heuristic")
    assert a.limitations


def test_heuristic_prefers_stated_limitation():
    a = heuristic_appraisal(make_paper("p", abstract="AI helped. A limitation is the small sample."), Q)
    assert a.limitations == "A limitation is the small sample."


def appraisal(pid, design, stance):
    return Appraisal(paper_id=pid, finding="f", study_design=design, stance=stance)


def papers(n, relevance=0.9):
    return [make_paper(f"p{i}", relevance=relevance) for i in range(n)]


def test_judge_sufficient_when_enough_varied_and_not_one_sided():
    ps = papers(5)
    ap = {p.paper_id: appraisal(p.paper_id, d, s) for p, d, s in zip(
        ps, ["experimental", "survey", "review", "experimental", "qualitative"],
        ["supports", "opposes", "mixed", "supports", "neutral"])}
    v = CriticAgent(None, min_relevant=5).judge(Q, ps, ap, [])
    assert v.sufficient and v.relevant and v.balanced


def test_judge_flags_too_few_papers_and_suggests_queries():
    v = CriticAgent(None, min_relevant=5).judge(Q, papers(2), {}, [])
    assert not v.sufficient and not v.relevant and v.suggested_queries


def test_judge_flags_one_sided_evidence():
    ps = papers(5)
    ap = {p.paper_id: appraisal(p.paper_id, d, "supports") for p, d in zip(
        ps, ["experimental", "survey", "review", "experimental", "survey"])}
    v = CriticAgent(None, min_relevant=5).judge(Q, ps, ap, [])
    assert not v.balanced and any("negative effects" in q for q in v.suggested_queries)


def test_all_neutral_is_not_one_sided():
    # Regression for remediation R2.
    ps = papers(5)
    ap = {p.paper_id: appraisal(p.paper_id, d, "neutral") for p, d in zip(
        ps, ["experimental", "survey", "review", "experimental", "survey"])}
    assert CriticAgent(None, min_relevant=5).judge(Q, ps, ap, []).balanced


def test_judge_ignores_low_relevance_and_does_not_resuggest_tried_queries():
    v = CriticAgent(None, min_relevant=1, min_relevance=0.5).judge(Q, papers(3, relevance=0.1), {}, [])
    assert v.relevant_count == 0
    v2 = CriticAgent(None, min_relevant=1).judge(Q, [], {}, v.suggested_queries)
    assert not set(v.suggested_queries) & set(v2.suggested_queries)


def test_llm_appraisals_are_grounded_and_gaps_filled_by_heuristic():
    ps = papers(2)
    reply = {"appraisals": [
        {"paper_id": "p0", "finding": "AI raised scores", "study_design": "experimental", "stance": "supports",
         "limitations": "single site"},
        {"paper_id": "invented-id", "finding": "x", "study_design": "review", "stance": "opposes",
         "limitations": ""}]}
    out = CriticAgent(FakeLLM(critic=json.dumps(reply)))({"question": Q, "cycle": 1, "papers": ps})
    ap = out["appraisals"]
    assert set(ap) == {"p0", "p1"}
    assert ap["p0"].method == "llm" and ap["p1"].method == "heuristic"


def test_only_new_papers_are_sent_to_llm():
    llm = FakeLLM(critic=json.dumps({"appraisals": []}))
    ps = papers(3)
    existing = {"p0": appraisal("p0", "survey", "neutral")}
    CriticAgent(llm)({"question": Q, "cycle": 2, "papers": ps, "appraisals": existing})
    prompt = llm.calls[0][1]
    assert "paper_id: p0" not in prompt and "paper_id: p1" in prompt


def test_llm_failure_falls_back_to_heuristic():
    out = CriticAgent(FakeLLM(critic="not json"))({"question": Q, "cycle": 1, "papers": papers(1)})
    assert out["appraisals"]["p0"].method == "heuristic"
    assert "heuristic after LLM failure" in out["audit"][0].detail
