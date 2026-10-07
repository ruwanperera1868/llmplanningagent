"""Small text utilities for keyword extraction and relevance scoring.

Why hand-rolled rather than NLTK or spaCy: relevance ranking and the heuristic
fallbacks only need tokenising, stop-word removal and light stemming. A large
NLP dependency (and its model downloads) would make installation fragile for
little gain on short abstracts.
"""

from __future__ import annotations

import re
from collections import Counter

STOP_WORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers how i if in into is it its
    itself just me more most my no nor not now of off on once only or other our out over
    own same she should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where which while who
    whom why will with would you your find recent research papers paper study studies
    review literature overview summarise summarize explain using use used based approach
    approaches new show shows propose proposed method methods work works also""".split()
)

# Abbreviations are folded into one canonical token before tokenising.
# Remediation (see docs/TESTING.md, issue R1): without this, the question "LLM
# agents" scored zero against abstracts that say "large language models".
SYNONYMS = [
    (re.compile(r"\blarge language models?\b", re.I), "llm"),
    (re.compile(r"\bllms\b", re.I), "llm"),
    (re.compile(r"\bgenerative artificial intelligence\b|\bgenai\b|\bgen ai\b", re.I), "generative ai"),
    (re.compile(r"\bartificial intelligence\b", re.I), "ai"),
    (re.compile(r"\bhigher education\b", re.I), "university"),
]

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ers", "er", "ed", "es", "s", "ly")


def normalise(text: str) -> str:
    for pattern, canonical in SYNONYMS:
        text = pattern.sub(canonical, text)
    return text


def tokenize(text: str) -> list[str]:
    # Hyphens split tokens so "LLM-based" yields "llm" (and the stop word "based").
    return [t.lower() for t in _TOKEN.findall(normalise(text))]


def stem(word: str) -> str:
    """Very light suffix stripping so "planning", "plans" and "plan" match.

    A full Porter stemmer is unnecessary for overlap scoring; this handles the
    plural and verb forms that caused most missed matches in testing.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def keywords(text: str) -> list[str]:
    """Content words in first-seen order, duplicates removed.

    Order is kept because people usually put the main concept first, which the
    planner relies on when it shortens queries.
    """
    seen: dict[str, None] = {}
    for tok in tokenize(text):
        if tok not in STOP_WORDS and (len(tok) > 2 or tok == "ai"):
            seen.setdefault(tok, None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def overlap_score(query_terms: list[str], text: str) -> float:
    """Share of query terms found in the text, with a small frequency bonus.

    Deliberately simpler than BM25 (Robertson and Zaragoza, 2009): the papers
    being ranked were already selected by the search APIs, so the score only
    needs to order a few dozen abstracts in a way a user can understand.
    """
    stems = list(dict.fromkeys(stem(t) for t in query_terms))
    if not stems:
        return 0.0
    counts = Counter(stem(t) for t in tokenize(text))
    hits = sum(1 for s in stems if counts[s])
    bonus = sum(min(counts[s], 3) for s in stems) / (3 * len(stems))
    return round(hits / len(stems) * 0.8 + bonus * 0.2, 4)
