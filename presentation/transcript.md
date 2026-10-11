# Audio narration transcript

Academic Evidence Investigator: Unit 11 individual presentation, Ruwan Perera

About 1300 words, roughly 9 minutes at a natural speaking pace.

## Slide 1: Academic Evidence Investigator

Hello, I'm Ruwan Perera. This presentation demonstrates my implementation of the Academic Evidence Investigator, the LLM-powered multi-agent system my group designed in Unit 6. I'll explain what the agent does, how the code realises the design, the main implementation decisions and why I made them, show it running, and finish with how I tested it, including the problems testing uncovered and how I fixed them.

## Slide 2: One goal in, a referenced briefing out

The intended users are postgraduate researchers, who spend a lot of time turning a question into searches, reading abstracts and judging how strong the evidence is. The agent takes one goal: a research question. It then works through five stages by itself. It plans, breaking the question into sub-questions and search queries. It retrieves papers from Semantic Scholar and Crossref. It appraises each paper's main finding, study design, stance and limitations. And it reports a structured briefing with Harvard references. What makes it an agent rather than a pipeline is the decision in the middle: if the evidence is thin or one-sided, it chooses to search again with a revised plan, up to a fixed limit.

## Slide 3: LangGraph realises the proposed architecture

This is the architecture from the proposal, implemented as a LangGraph state graph. The Planner, Retrieval, Processing and Critic nodes run in sequence and all read and write one shared ResearchState. In belief-desire-intention terms (Archibald et al., 2024), the papers, appraisals and verdict are the agent's beliefs, and the Planner's current queries are its intention, which it revises when the Critic reports a problem. The Critic's output goes to a conditional edge: if the evidence is sufficient, or the cycle limit is reached, control passes to the Reporter; otherwise it loops back to the Planner with feedback. LangGraph was chosen over CrewAI because this edge and the bounded loop are explicit in code, which is LangGraph's evaluator-optimiser pattern (LangChain, no date). Alternating reasoning nodes with the acting retrieval node follows ReAct (Yao et al., 2023).

## Slide 4: The LLM proposes, Python verifies

The guiding principle, taken from the proposal, is that the LLM only does work that needs language understanding. Qwen2.5-7B-Instruct, running locally under Ollama for the live demonstration, proposes a plan, appraises abstracts and drafts the briefing. Python handles everything with one correct answer: deduplication, ranking, rate limiting and retries, and every LLM output crosses a Pydantic schema first. Four decisions along the bottom are where most of the critical thinking sits. Whether to search again is decided by thresholds, not by asking the model, because self-reflection reduces but does not remove hallucination (Ji et al., 2023). Retrieval-augmented generation helps (Lewis et al., 2020) but can still fail (Huang et al., 2025), so citations to papers that were never retrieved are stripped. Every LLM step has a deterministic fallback, so a rate limit lowers quality but never stops the run. And offline mode is explicit, never a silent fallback, because mixing demo data into a live briefing would misrepresent the evidence.

## Slide 5: Code and executables

Here is how the code is organised and how it is run. graph.py wires the agents together, state.py holds the shared research state, each agent has its own module, and the sources, the LLM client, processing and storage are kept separate, so each can be tested on its own. There are three ways to run it. The command line takes a question and prints the briefing with the audit trail of every agent step. The web interface offers the same run in a browser, with the run history. And the test suite runs 83 tests, all passing, with 96 percent coverage. Live mode uses Qwen running locally under Ollama; offline mode runs the same code on a bundled corpus. The README gives the full setup.

## Slide 6: Demonstration: from question to briefing

Here is the Flask interface after a live run on my own computer, asking the question from our Unit 6 proposal: does generative AI improve university students' academic performance? The agent searched Semantic Scholar and Crossref and used Qwen running locally. The status line at the top is written by Python from the Critic's verdict, so the system cannot overstate how complete its evidence is. Here the Critic judged twelve papers sufficient and balanced after one cycle. Each key finding carries in-text citations to the papers it came from. Below that are conflicting evidence, limitations, and possible research opportunities, framed only as gaps within the retrieved papers. The evidence table shows each paper's design, stance and relevance, and the audit trail lists every search and decision, including which steps used the model and which fell back. The whole run is stored in SQLite and can be reopened or downloaded as JSON. Offline mode runs the same code on built-in data, which keeps the next demonstration reproducible.

## Slide 7: Demonstration: the Critic drives re-planning

This is the code that makes the system an agent rather than a fixed pipeline. After every cycle, LangGraph calls route_after_critic. If the Critic's verdict is sufficient, or the cycle limit is reached, the graph goes to the Reporter; otherwise it loops back to the Planner. The comment explains why: the stop decision is a rule, not the model judging its own work, because self-reflection reduces but does not remove hallucination (Ji et al., 2023). Below is what that code did in an offline run of "Does retrieval reduce hallucination?". The Critic found only two relevant papers, so the Planner re-planned with queries for trials and meta-analyses, then broadened the search. At the cycle limit the Reporter wrote the briefing anyway, but labelled it provisional. By contrast, the live generative AI question was judged sufficient after one cycle. So the agent's behaviour depends on what it finds, not on a fixed script.

## Slide 8: Testing: 83 tests, no network needed

The test suite has 83 automated tests, all passing, with 96 percent line coverage, and it needs no network or API key. That is possible because the sources and the LLM are injected through small interfaces, so tests use scripted fakes and synthetic API responses. Sixty-seven unit tests cover each agent, parser and rule. Eleven functional tests run the real graph end to end and prove the behaviours the proposal promised, listed on the right: the re-plan, the cycle limit, an outage, deduplication, an invented citation being removed, and a complete LLM outage still producing a briefing. Interface tests drive the command line and the web app, including escaping a malicious paper title. Four live tests call the real services and the model when enabled.

## Slide 9: How it developed: build, test, refine, run live

This slide shows how the system developed over the module. It started from our Unit 6 group design. I then built it offline-first in incremental commits: the schemas and shared state, the sources and LLM client, the agents in a LangGraph graph, the Flask interface, and the tests. Testing drove five refinements. For example, "LLM-based" never matched "large language models", and synonym folding took the same question from two relevant papers to ten. Running live on Windows found four more. The hosted Qwen model was withdrawn, so I made the endpoint configurable and ran Qwen locally. Semantic Scholar throttled requests, so retries back off longer. One bad appraisal discarded all twelve, so each is now validated on its own. And the model once wrote invalid JSON, so JSON mode now constrains its output. The suite grew from 76 to 83 tests, and every step is a commit or pull request on GitHub.

## Slide 10: Responsible use, limits and next steps

To finish: the agent respects provider rate limits, keeps credentials out of code and storage, frames research gaps only within what it retrieved, and states that it supports rather than replaces the researcher. Its limits are real: without the LLM the appraisal is a keyword heuristic, it reads abstracts only, relevance is term overlap rather than semantic similarity, and the web app handles one user at a time. The next steps would be embedding-based relevance, PostgreSQL, and evaluating appraisals against a hand-labelled set. Overall, the implementation delivers the proposal's planning, delegation, critique and re-planning, while keeping every finding traceable to its evidence. Thank you.

## References cited in the narration

Full references in UoEO Harvard format are in README.md: Archibald et al. (2024); Huang et al. (2025); Ji et al. (2023); LangChain (no date); Lewis et al. (2020); Yao et al. (2023).
