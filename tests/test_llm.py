import pytest
import requests

from evidence_investigator.config import Settings
from evidence_investigator.investigator import default_llm
from evidence_investigator.llm import (ChatCompletionsClient, HuggingFaceClient, LLMError, extract_json,
                                      structured_call)
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


def test_local_endpoint_needs_no_token_and_sends_no_auth_header():
    # Remediation R6: the same client must reach a keyless local server (Ollama).
    session = FakeSession(FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
    client = ChatCompletionsClient("", "qwen2.5:7b", "http://localhost:11434/v1/", session=session)
    assert client.complete("s", "p") == "ok"
    url, headers, _ = session.sent
    assert url == "http://localhost:11434/v1/chat/completions" and headers == {}
    assert client.name == "localhost:qwen2.5:7b"


def test_provider_error_message_is_kept():
    # The first live run logged only "400 Bad Request"; the cause was in the body.
    response = FakeResponse(status=400)
    response.text = '{"error": {"message": "model not supported by any provider"}}'
    client = HuggingFaceClient("secret-token", "m", session=FakeSession(response))
    with pytest.raises(LLMError, match="HTTP 400.*model not supported") as info:
        client.complete("s", "p")
    assert "secret-token" not in str(info.value)


def test_default_llm_uses_configured_endpoint():
    assert default_llm(Settings()) is None                       # no token, hosted router
    local = default_llm(Settings(llm_base_url="http://localhost:11434/v1", hf_model="qwen2.5:7b"))
    assert local.url == "http://localhost:11434/v1/chat/completions"
    assert default_llm(Settings(hf_token="t", offline=True)) is None
