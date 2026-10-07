# Test fixtures

`semantic_scholar_search.json` and `crossref_works.json` are **synthetic** responses written
by hand in the response formats documented by the Semantic Scholar Graph API and the
Crossref REST API. The papers in them are fictional, so they cannot be mistaken for real
citations. They exercise the parsers' edge cases (missing abstracts, JATS markup, non-article
record types, DOIs shared across sources) without network access.
