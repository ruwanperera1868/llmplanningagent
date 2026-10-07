import pytest
import requests

from evidence_investigator.llm import HuggingFaceClient, LLMError, extract_json, structured_call
from evidence_investigator.schemas import SearchPlan


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]


@pytest.mark.parametrize("bad", ["no json here", '{"a": ', "{'single': 'quotes'}"])
def test_extract_json_rejects_bad_output(bad):
    with pytest.raises(LLMError):
        extract_json(bad)


class Echo:
    name = "echo"

    def __init__(self, text):
        self.text = text

    def complete(self, system, prompt, max_tokens=800):
        return self.text


def test_structured_call_validates_schema():
    plan = structured_call(Echo('{"sub_questions": ["q"], "queries": ["  a  b  "]}'), "s", "p", SearchPlan)
    assert plan.queries == ["a b"]
    with pytest.raises(LLMError):
        structured_call(Echo('{"queries": []}'), "s", "p", SearchPlan)


def test_search_plan_trims_long_queries():
    plan = SearchPlan(sub_questions=["q"], queries=[" ".join(["word"] * 20)])
    assert len(plan.queries[0].split()) == 10


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response, self.sent = response, None

    def post(self, url, headers, json, timeout):
        self.sent = (url, headers, json)
        return self.response


def test_hf_client_success_sends_model_and_auth():
    session = FakeSession(FakeResponse(payload={"choices": [{"message": {"content": " hi "}}]}))
    client = HuggingFaceClient("secret-token", "Qwen/Qwen2.5-7B-Instruct", session=session)
    assert client.complete("sys", "prompt") == "hi"
    url, headers, body = session.sent
    assert body["model"] == "Qwen/Qwen2.5-7B-Instruct" and headers["Authorization"] == "Bearer secret-token"


def test_hf_client_errors_become_llm_error_without_leaking_token():
    client = HuggingFaceClient("secret-token", "m", session=FakeSession(FakeResponse(status=429)))
    with pytest.raises(LLMError) as info:
        client.complete("s", "p")
    assert "secret-token" not in str(info.value)


def test_hf_client_requires_token():
    with pytest.raises(LLMError):
        HuggingFaceClient("", "m")
