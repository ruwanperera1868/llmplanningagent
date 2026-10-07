# Audio narration transcript

Academic Evidence Investigator: Unit 11 individual presentation, Ruwan Perera

About 1139 words, roughly 8 minutes at a natural speaking pace.

## Slide 1: Academic Evidence Investigator

Hello, I'm Ruwan Perera. This presentation demonstrates my implementation of the Academic Evidence Investigator, the LLM-powered multi-agent system my group designed in Unit 6. I'll explain what the agent does, how the code realises the design, the main implementation decisions and why I made them, show it running, and finish with how I tested it, including the problems testing uncovered and how I fixed them.

## Slide 2: One goal in, a referenced briefing out

The intended users are postgraduate researchers, who spend a lot of time turning a question into searches, reading abstracts and judging how strong the evidence is. The agent takes one goal: a research question. It then works through five stages by itself. It plans, breaking the question into sub-questions and search queries. It retrieves papers from Semantic Scholar and Crossref. It appraises each paper's main finding, study design, stance and limitations. And it reports a structured briefing with Harvard references. What makes it an agent rather than a pipeline is the decision in the middle: if the evidence is thin or one-sided, it chooses to search again with a revised plan, up to a fixed limit.

## Slide 3: LangGraph realises the proposed architecture

This is the architecture from the proposal, implemented as a LangGraph state graph. The Planner, Retrieval, Processing and Critic nodes run in sequence and all read and write one shared ResearchState. In belief-desire-intention terms (Archibald et al., 2024), the papers, appraisals and verdict are the agent's beliefs, and the Planner's current queries are its intention, which it revises when the Critic reports a problem. The Critic's output goes to a conditional edge: if the evidence is sufficient, or the cycle limit is reached, control passes to the Reporter; otherwise it loops back to the Planner with feedback. LangGraph was chosen over CrewAI because this edge and the bounded loop are explicit in code, which is LangGraph's evaluator-optimiser pattern (LangChain, no date). Alternating reasoning nodes with the acting retrieval node follows ReAct (Yao et al., 2023).

## Slide 4: The LLM proposes, Python verifies

The guiding principle, taken from the proposal, is that the LLM only does work that needs language understanding. Qwen2.5-7B-Instruct, through Hugging Face, proposes a plan, appraises abstracts and drafts the briefing. Python handles everything with one correct answer: deduplication by DOI and then by title and first author, ranking, rate limiting, and retries. Every LLM output crosses a Pydantic schema before anything else uses it. The Critic sends all new papers to the model in one batched call, and only new papers, which keeps token cost in line with new evidence.

## Slide 5: Four decisions that make it trustworthy

These four decisions are where most of the critical thinking in the code sits. First, the decision to stop or keep searching is made by explicit thresholds, not by asking the model whether it is satisfied. Self-reflection reduces hallucination but does not remove it (Ji et al., 2023), and a rule can be tested and explained. Second, reporting is grounded: retrieval-augmented generation helps (Lewis et al., 2020), but it can still fail (Huang et al., 2025), so any citation to a paper that was never retrieved is stripped. Third, every LLM step has a deterministic fallback, so a free-tier rate limit lowers quality but never stops the run, and the audit trail records which path was used. Fourth, offline mode is explicit and never a silent fallback, because mixing demo data into a live briefing would misrepresent the evidence.

## Slide 6: Demonstration: from question to briefing

Here is the Flask interface after an offline run. The researcher types a question and presses Investigate. The status line at the top is written by Python from the Critic's verdict, so the system cannot overstate how complete its evidence is. Each key finding carries in-text citations to the papers it came from. Below that are conflicting evidence, limitations, and possible research opportunities, framed only as gaps within the retrieved papers. The evidence table shows each paper's design, stance and relevance, and the audit trail lists every search and decision. The whole run is stored in SQLite and can be reopened or downloaded as JSON. With a Hugging Face token set, the same code searches Semantic Scholar and Crossref live and uses Qwen for each LLM step; offline mode makes this demonstration reproducible.

## Slide 7: Demonstration: the Critic drives re-planning

This log shows the autonomous loop. For the question "Does retrieval reduce hallucination?" the first cycle retrieves only two relevant papers, both experimental. The Critic judges this insufficient and unbalanced, and suggests targeted searches. The Planner re-plans with queries for randomised controlled trials and meta-analyses. When that still finds little, it broadens the query in the third cycle. At the cycle limit the Reporter writes the briefing anyway, but labels it provisional and says why. By contrast, a question about how LLM agents plan, act and reflect was judged sufficient and balanced after one cycle. So the agent's behaviour depends on what it finds, not on a fixed script.

## Slide 8: Testing: 76 tests, no network needed

The test suite has 76 automated tests, all passing, with 97 percent line coverage, and it needs no network or API key. That is possible because the sources and the LLM are injected through small interfaces, so tests use scripted fakes and synthetic API responses. Sixty unit tests cover each agent, parser and rule. Eleven functional tests run the real graph end to end and prove the behaviours the proposal promised, listed on the right: the re-plan, the cycle limit, an outage, deduplication, an invented citation being removed, and a complete LLM outage still producing a briefing. Interface tests drive the command line and the web app, including escaping a malicious paper title. Four live tests call the real services when enabled.

## Slide 9: What testing found, and what changed

Testing changed the code. The first smoke run showed that "LLM-based" never matched abstracts that say "large language models", so the agent wasted all three cycles. Adding synonym folding and light stemming took the same question from two papers in three cycles to ten papers in one. The same run showed that the Critic treated all-neutral evidence as biased. The first coverage report showed leaked database connections, and the suite now fails on any leak. A new retry test showed a 404 being retried like a rate limit. Reviewing a screenshot showed references in the wrong order for Harvard style. Each fix has a regression test, the full log is in docs/TESTING.md, and the Git history shows the system being built up step by step.

## Slide 10: Responsible use, limits and next steps

To finish: the agent respects provider rate limits, keeps credentials out of code and storage, frames research gaps only within what it retrieved, and states that it supports rather than replaces the researcher. Its limits are real: without the LLM the appraisal is a keyword heuristic, it reads abstracts only, relevance is term overlap rather than semantic similarity, and the web app handles one user at a time. The next steps would be embedding-based relevance, PostgreSQL, and evaluating appraisals against a hand-labelled set. Overall, the implementation delivers the proposal's planning, delegation, critique and re-planning, while keeping every finding traceable to its evidence. Thank you.

## References cited in the narration

Full references in UoEO Harvard format are in README.md: Archibald et al. (2024); Huang et al. (2025); Ji et al. (2023); LangChain (no date); Lewis et al. (2020); Yao et al. (2023).
