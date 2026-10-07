"""LLM access layer for Qwen2.5-7B-Instruct via Hugging Face.

Design decisions
----------------
* Agents depend on the small ``LLMClient`` protocol, not on a vendor SDK
  (dependency inversion). The proposal keeps the model "configurable", and this
  also lets tests inject a scripted fake so they never need a token or network.
* The client calls the OpenAI-compatible chat endpoint of the Hugging Face
  Inference Providers router with ``requests``, which is listed in the
  proposal's stack. Pulling in a full LangChain model wrapper was rejected
  because it would add a large dependency for one HTTP call.
* Every failure becomes ``LLMError``. Callers must catch it and fall back to a
  deterministic method, so a missing token, free-tier rate limit or invalid
  JSON lowers quality but never stops the agent from producing a briefing.
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to obtain a usable, schema-valid completion."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str: ...


class HuggingFaceClient:
    def __init__(self, token: str, model: str, timeout: float = 90.0, session=None):
        if not token:
            raise LLMError("HF_TOKEN is not set")
        self._token = token
        self.model = model
        self.timeout = timeout
        self._session = session or requests.Session()
        self.name = f"huggingface:{model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        try:
            resp = self._session.post(
                HF_CHAT_URL,
                headers={"Authorization": f"Bearer {self._token}"},
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
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            # The token is never echoed, so logs and the audit trail are safe to share.
            raise LLMError(f"Hugging Face call failed: {type(exc).__name__}: {exc}") from exc


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
