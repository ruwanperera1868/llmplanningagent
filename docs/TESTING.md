# Testing and remediation

## Strategy

Testing follows the proposal's plan (section 5): each component is tested on its own, then the integrated workflow, using a fixed schema so intermediate outputs can be checked. Three choices shape the suite:

1. **No network in the default run.** Sources and the LLM are injected through small protocols, so tests use scripted fakes (`tests/conftest.py`) and synthetic API responses (`tests/fixtures/`). Results are deterministic and the suite runs in about three seconds.
2. **Behaviour from the proposal is tested end to end.** `test_functional_workflow.py` runs the real LangGraph graph and checks the behaviours promised in Unit 6: the critic-to-planner loop, the cycle limit, resilience to an outage, deduplication across sources, citation grounding, and graceful degradation.
3. **Live checks are opt-in.** `tests/test_live.py` calls the real Semantic Scholar and Crossref services and the configured LLM endpoint when `RUN_LIVE_TESTS=1`. They are kept out of the default run because the services are rate-limited and their results change.

| Level | Files | What is verified |
|---|---|---|
| Unit | `test_text.py`, `test_sources.py`, `test_llm.py`, `test_processing.py`, `test_planner.py`, `test_critic.py`, `test_reporter.py`, `test_rendering_storage.py` | Tokenising and stemming; API parsing and edge cases; rate limiting and retries; JSON extraction and schema validation; deduplication, filtering and ranking; plan, re-plan and fallback; appraisal heuristics and the sufficiency rules; citation grounding; Harvard formatting; SQLite round trip |
| Functional | `test_functional_workflow.py` | The whole agent loop with fakes, including a full scripted LLM path |
| Interface | `test_interfaces.py` | CLI exit codes and output file; Flask form, briefing page, JSON export, 400/404 handling, HTML escaping |
| Live (opt-in) | `test_live.py` | Real APIs still match the parsers; a full live run completes |

## How to run

```bash
python -m pytest -v --cov=evidence_investigator --cov-report=term-missing
python -m pytest -W error::ResourceWarning     # as used for the final report
RUN_LIVE_TESTS=1 python -m pytest tests/test_live.py -v
```

## Results

Final run (`evidence/test-report.txt`): **82 passed, 4 skipped (live), 0 failed; 96% line coverage.**

| File | Tests |
|---|---|
| test_critic.py | 15 |
| test_functional_workflow.py | 11 |
| test_sources.py | 13 |
| test_llm.py | 12 |
| test_planner.py | 6 |
| test_reporter.py | 6 |
| test_interfaces.py | 5 |
| test_rendering_storage.py | 5 |
| test_text.py | 5 |
| test_processing.py | 4 |
| test_live.py (skipped unless enabled) | 4 |

Lines not covered are mainly `__main__.py` (exercised through the CLI tests by calling `main()` directly) and the network branch of `http_json` that sleeps between real retries.

## Remediation log

Issues found during development and testing, in the order they were found.

| ID | Found by | Problem | Fix | Regression test |
|---|---|---|---|---|
| R1 | First offline smoke run of *"How do LLM-based agents plan, act and reflect?"* | Only 2 papers were judged relevant and the agent used all 3 cycles. "LLM-based" never matched abstracts that say "large language models", and "plan" did not match "planning". | `text.py` folds synonyms (LLM, large language model, GenAI, higher education) and applies light stemming before overlap scoring. The same question then retrieved 10 relevant papers in 1 cycle. | `test_synonyms_fold_llm_variants`, `test_stem_matches_inflections` |
| R2 | Same run | The critic called all-neutral evidence "one-sided" and triggered needless re-plans for descriptive topics. | Only a uniformly *supporting* or *opposing* set counts as one-sided. | `test_all_neutral_is_not_one_sided` |
| R3 | First coverage run (`evidence/test-run-1-before-remediation.txt`) | `ResourceWarning: unclosed database`. `create_app` opened its own SQLite connection in addition to the caller's, and some tests never closed theirs. | `create_app` accepts a shared `Storage`; `Storage` became a context manager; the tests close their connections. The suite now runs with `-W error::ResourceWarning`, so any new leak fails the build. | Whole suite under `-W error::ResourceWarning` |
| R4 | New retry test | A permanent HTTP 404 was retried like a 429 until the fake ran out of responses (`IndexError`), so a bad request would waste the shared rate limit. | `http_json` retries only timeouts, 429 and 5xx; other 4xx errors fail immediately. | `test_http_json_retries_transient_errors` (5 cases) |
| R5 | Reviewing the briefing screenshot | References were ordered by title, not alphabetically by first author's surname as Harvard style requires, and the summary printed raw Python dictionaries. | `rendering.by_first_author` is used by both the Markdown and HTML outputs; counts are printed as readable text. | `test_references_sorted_by_first_author_surname` |
| R6 | First live run on Windows | Every LLM call failed and the agents fell back to their heuristics. The log said only "400 Bad Request". Calling the router directly showed `model_not_supported`: no provider served Qwen2.5-7B-Instruct any more, and the free account's credit for larger models was used up (HTTP 402). | `llm.py` now keeps the provider's error message in the `LLMError`, and the endpoint is configurable (`LLM_BASE_URL`, `LLM_TIMEOUT`). The same client reaches the Hugging Face router, a hosted alternative, or Qwen2.5-7B running locally under Ollama, which needs no token. The graceful fallback worked as designed: the run still produced a briefing and the audit trail recorded each failure. | `test_provider_error_message_is_kept`, `test_local_endpoint_needs_no_token_and_sends_no_auth_header`, `test_default_llm_uses_configured_endpoint` |
| R7 | Same live run | Semantic Scholar answered every query with HTTP 429 (its shared unauthenticated pool was throttled), so only Crossref contributed papers. Two short retries were not enough. | `http_json` now retries up to four times with doubling back-off (2 to 16 s), honours `Retry-After` (capped at 30 s), and an optional `SEMANTIC_SCHOLAR_API_KEY` moves requests to a dedicated rate limit. Retrieval still continues with the other source if one fails, as before. | `test_backoff_honours_retry_after_and_is_capped`, `test_semantic_scholar_sends_api_key_only_when_set`, `test_http_json_retries_transient_errors` |
| R8 | First live run with Qwen2.5-7B under Ollama | The Planner and Reporter used the LLM successfully, but the Critic logged `heuristic after LLM failure: ValidationError`: the batch of 12 appraisals was validated all-or-nothing, so one malformed item discarded the rest. | `coerce_appraisal` validates each item separately, maps common label synonyms (e.g. "systematic review" to review, "positive" to supports), trims over-long text and rejects only unusable items, which fall back to the heuristic individually. The audit trail reports how many were appraised by heuristic. | `test_llm_labels_are_normalised_item_by_item` |

One failure was in the test, not the code: an early processing test used two fixtures with the same title and author, which the code correctly treated as duplicates. The test data was corrected.

## Known limitations of the testing

- The live tests have not yet been run from the development environment, which blocks the external APIs. Run `RUN_LIVE_TESTS=1 python -m pytest tests/test_live.py -v` with network access and add the output to `evidence/`.
- The LLM path is tested with a scripted model. How reliably Qwen2.5-7B follows the JSON schemas is checked only by the live run; every LLM failure falls back to the deterministic path, and the audit trail records when that happens.
- The heuristic appraisal lexicon is tested on short, hand-written abstracts, not against a labelled benchmark.
