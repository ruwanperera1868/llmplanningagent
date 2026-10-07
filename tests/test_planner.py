import json

from conftest import FakeLLM
from evidence_investigator.agents.planner import PlannerAgent, fallback_plan
from evidence_investigator.schemas import CriticVerdict

Q = "Does generative AI improve university students' academic performance?"


def verdict(suggested):
    return CriticVerdict(relevant_count=1, designs=[], stances={}, sufficient=False, relevant=False,
                         balanced=False, reasons=["too few"], suggested_queries=suggested)


def test_offline_initial_plan_uses_keyword_fallback():
    out = PlannerAgent(None)({"question": Q, "cycle": 0})
    assert out["cycle"] == 1
    assert out["pending_queries"][0] == "generative ai improve university students academic"
    assert any("systematic review" in q for q in out["pending_queries"])
    assert out["audit"][0].action == "plan"


def test_llm_plan_is_used_when_valid():
    llm = FakeLLM(planner=json.dumps({"sub_questions": ["s"], "queries": ["genai exam scores", "ai tutoring rct"],
                                      "rationale": "r"}))
    out = PlannerAgent(llm)({"question": Q, "cycle": 0})
    assert out["pending_queries"] == ["genai exam scores", "ai tutoring rct"]
    assert "LLM plan" in out["audit"][0].detail


def test_invalid_llm_plan_falls_back_and_is_logged():
    out = PlannerAgent(FakeLLM(planner="I think you should search Google."))({"question": Q, "cycle": 0})
    assert out["plan"] == fallback_plan(Q)
    assert "rejected" in out["audit"][0].detail


def test_replan_uses_critic_suggestions_and_skips_tried_queries():
    state = {"question": Q, "cycle": 1, "plan": fallback_plan(Q), "queries_tried": ["a", "b"],
             "verdict": verdict(["b", "c"])}
    out = PlannerAgent(None)(state)
    assert out["pending_queries"] == ["c"] and out["cycle"] == 2 and out["audit"][0].action == "re-plan"


def test_replan_llm_must_produce_new_queries():
    llm = FakeLLM(planner=json.dumps({"sub_questions": ["s"], "queries": ["a"]}))  # only a tried query
    state = {"question": Q, "cycle": 1, "plan": fallback_plan(Q), "queries_tried": ["a"],
             "verdict": verdict(["fresh one"])}
    assert PlannerAgent(llm)(state)["pending_queries"] == ["fresh one"]


def test_replan_broadens_when_every_idea_was_tried():
    state = {"question": Q, "cycle": 1, "plan": fallback_plan(Q), "queries_tried": ["x"],
             "verdict": verdict(["x"])}
    queries = PlannerAgent(None)(state)["pending_queries"]
    assert queries and queries[0] != "x"
