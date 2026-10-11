"""LLM access layer for Qwen2.5-7B-Instruct via Hugging Face.

Design decisions
----------------
* Agents depend on the small ``LLMClient`` protocol, not on a vendor SDK
  (dependency inversion). The proposal keeps the model "configurable", and this
  also lets tests inject a scripted fake so they never need a token or network.
* The client calls an OpenAI-compatible ``/chat/completions`` endpoint with
  ``requests``, which is listed in the proposal's stack. Pulling in a full
  LangChain model wrapper was rejected because it would add a large dependency
  for one HTTP call.
* The endpoint defaults to the Hugging Face Inference Providers router but is
  configurable (``LLM_BASE_URL``). Remediation R6: the live run showed that
  provider support for a model changes over time (Qwen2.5-7B-Instruct stopped
  being served on the router) and that free-tier credit runs out. Because
  Hugging Face, Ollama and Groq all speak the same protocol, one client covers
  a hosted model or the same Qwen2.5-7B running locally, with no code change.
* Every failure becomes ``LLMError``. Callers must catch it and fall back to a
  deterministic method, so a missing token, free-tier rate limit or invalid
  JSON lowers quality but never stops the agent from producing a briefing.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

HF_BASE_URL = "https://router.huggingface.co/v1"
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to obtain a usable, schema-valid completion."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str: ...


class ChatCompletionsClient:
    def __init__(self, token: str, model: str, base_url: str = HF_BASE_URL,
                 timeout: float = 90.0, session=None):
        base_url = base_url.rstrip("/")
        # A hosted router always needs a token; a local server such as Ollama
        # does not, so the check applies only to the Hugging Face default.
        if not token and base_url == HF_BASE_URL:
            raise LLMError("HF_TOKEN is not set")
        self._token = token
        self.model = model
        self.url = f"{base_url}/chat/completions"
        self.timeout = timeout
        self._session = session or requests.Session()
        host = "huggingface" if base_url == HF_BASE_URL else urlparse(base_url).hostname
        self.name = f"{host}:{model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            resp = self._session.post(
                self.url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    # Low temperature favours consistent, schema-following
                    # output over creativity and makes runs more repeatable.
                    "temperature": 0.2,
                },
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                # Remediation R6: the first live run logged only "400 Bad
                # Request", which hid the real cause ("model not supported by
                # any provider"). The provider's own message is now kept,
                # truncated, because it says what the user has to change.
                raise LLMError(f"LLM call failed: HTTP {resp.status_code}: {_body(resp)}")
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            # The token is never echoed, so logs and the audit trail are safe to share.
            raise LLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc


# Kept so existing callers and the proposal's terminology still work.
HuggingFaceClient = ChatCompletionsClient


def _body(resp) -> str:
    try:
        text = resp.text
    except Exception:  # noqa: BLE001 - an unreadable body must not mask the HTTP error
        return ""
    return " ".join(str(text).split())[:300]


def extract_json(text: str) -> object:
    """Parse the first JSON object or array in a model response.

    Instruction-tuned 7B models often wrap JSON in Markdown fences or add a
    sentence of prose even when told not to, so strict parsing of the raw text
    fails far more often than the content itself is wrong.
    """
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise LLMError("no JSON found in model response")
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end <= start:
        raise LLMError("unterminated JSON in model response")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON in model response: {exc}") from exc


def structured_call(llm: LLMClient, system: str, prompt: str, schema: type[T],
                    max_tokens: int = 800) -> T:
    """Ask for JSON and validate it against a Pydantic schema in one step."""
    raw = llm.complete(system, prompt, max_tokens)
    try:
        return schema.model_validate(extract_json(raw))
    except ValidationError as exc:
        raise LLMError(f"response did not match {schema.__name__}: {exc.error_count()} errors") from exc
