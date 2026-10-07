from conftest import make_paper
from evidence_investigator.processing import process

Q = "Does generative AI improve student performance?"


def test_dedupes_by_doi_across_sources_keeping_record_with_abstract():
    a1 = make_paper("doi:10.1/x", doi="10.1/x", abstract="", source="semantic_scholar")
    a2 = make_paper("doi:10.1/x", doi="10.1/x", source="crossref")
    kept, stats = process([a1, a2], Q, 0.0, 10)
    assert len(kept) == 1 and kept[0].abstract and stats.duplicates_removed == 1


def test_dedupes_by_title_and_first_author_when_no_doi():
    a = make_paper("s2:1", title="Generative AI: A Study!")
    b = make_paper("s2:2", title="generative ai a study")
    c = make_paper("s2:3", title="generative ai a study", authors=["Different Person"])
    kept, stats = process([a, b, c], Q, 0.0, 10)
    assert len(kept) == 2 and stats.duplicates_removed == 1


def test_filters_missing_abstract_and_low_relevance_then_ranks():
    papers = [
        make_paper("p1", title="Generative AI without abstract", abstract=""),
        make_paper("p2", title="Coral reef ecology", abstract="Fish populations in reefs."),
        make_paper("p3", title="AI and performance", abstract="Generative AI and student outcomes.", year=2020),
        make_paper("p4"),
    ]
    kept, stats = process(papers, Q, 0.3, 10)
    assert [p.paper_id for p in kept] == ["p4", "p3"]
    assert (stats.missing_abstract, stats.below_relevance, stats.retained) == (1, 1, 2)
    assert kept[0].relevance >= kept[1].relevance


def test_limit_and_empty_input():
    kept, _ = process([make_paper(f"p{i}", title=f"Generative AI study {i}") for i in range(5)], Q, 0, 2)
    assert len(kept) == 2
    assert process([], Q, 0.3, 5)[0] == []
