from evidence_investigator.text import keywords, overlap_score, split_sentences, stem


def test_keywords_drop_stop_words_and_keep_order():
    assert keywords("Does generative AI improve university students' academic performance?") == [
        "generative", "ai", "improve", "university", "students", "academic", "performance"]


def test_synonyms_fold_llm_variants():
    # Regression for remediation R1: "LLM-based" must match "large language models".
    assert keywords("LLM-based agents") == ["llm", "agents"]
    assert overlap_score(["llm", "agents"], "Agents built on large language models") > 0.8


def test_stem_matches_inflections():
    assert stem("planning") == stem("plans") == stem("plan")
    assert stem("improves") == stem("improve") == stem("improved")


def test_overlap_score_bounds():
    assert overlap_score([], "anything") == 0.0
    assert overlap_score(["quantum"], "nothing relevant here") == 0.0
    assert 0 < overlap_score(["agent", "memory"], "agent with no recall") < 1


def test_split_sentences():
    assert split_sentences("First one. Second one! 3rd? no") == ["First one.", "Second one!", "3rd? no"]
