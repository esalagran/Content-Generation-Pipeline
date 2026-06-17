"""Grounding-check unit tests — the planted-hallucination cases that prove the
deterministic checks have teeth (in isolation; the meta-eval proves it end-to-end).
"""

from content_pipeline.fixtures import (
    BAD_BLAND,
    BAD_EMPTY_CITATION,
    BAD_UNLISTED_AMENITY,
    COTTAGE,
    GOOD_COTTAGE,
    GOOD_VILLA,
    VILLA,
)
from content_pipeline.grounding import (
    CitationValidityCheck,
    QualityCheck,
    build_corpus,
    salient_facts,
    strip_html,
)


def test_corpus_strips_html_and_includes_reviews():
    corpus = build_corpus(VILLA)
    assert "<p>" not in corpus and "<strong>" not in corpus
    assert "marina" in corpus  # from the (HTML) description
    assert "old town" in corpus  # from a review -> reviews ground facts


def test_strip_html():
    assert strip_html("<p>hi <strong>there</strong></p>").split() == ["hi", "there"]


def test_citation_passes_on_good_copy():
    assert CitationValidityCheck(VILLA).run(GOOD_VILLA).passed
    assert CitationValidityCheck(COTTAGE).run(GOOD_COTTAGE).passed


def test_citation_catches_unlisted_amenity():
    result = CitationValidityCheck(VILLA).run(BAD_UNLISTED_AMENITY)
    assert not result.passed
    assert any("Helipad" in f for f in result.failures)


def test_citation_catches_empty_field_citation():
    # cottage has no reviews; a highlight citing `reviews` is an empty citation
    result = CitationValidityCheck(COTTAGE).run(BAD_EMPTY_CITATION)
    assert not result.passed
    assert any("reviews" in f for f in result.failures)


def test_quality_passes_on_good_copy():
    assert QualityCheck(VILLA).run(GOOD_VILLA).passed


def test_quality_catches_bland_copy():
    result = QualityCheck(VILLA).run(BAD_BLAND)
    assert not result.passed
    # bland sample trips multiple sub-rules: generic phrase, too few highlights, repetition
    assert len(result.failures) >= 2


def test_salient_facts_villa_includes_premium_and_social_proof():
    kinds = {f["kind"] for f in salient_facts(VILLA)}
    assert {"capacity", "location", "social_proof", "amenity"} <= kinds
    assert any("pool" in f["fact"] for f in salient_facts(VILLA))  # PrivatePool is premium


def test_salient_facts_cottage_drops_social_proof_when_no_reviews():
    facts = salient_facts(COTTAGE)
    assert "social_proof" not in {f["kind"] for f in facts}  # 0 reviews
    assert any("fireplace" in f["fact"] for f in facts)
