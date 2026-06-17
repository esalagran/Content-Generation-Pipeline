"""Validates atomic_claims — the decomposition the faithfulness metric rests on.
A bad split means every downstream verdict is computed on garbage, and the gold set
only checks verdicts, not the split. These pin the splitter's behaviour. Offline."""

from content_pipeline.models import (
    AmenityDescription,
    Claim,
    GeneratedContent,
    SourceField,
)
from content_pipeline.scorers import atomic_claims


def _content(about: str) -> GeneratedContent:
    return GeneratedContent(
        hero_headline="Seafront villa",
        highlights=[Claim(text="Sleeps six", source_field=SourceField.rental_info)],
        about_section=about,
        amenity_descriptions=[AmenityDescription(code="PrivatePool", text="A private pool.")],
    )


def test_includes_every_section_and_no_empty_claims():
    claims = atomic_claims(_content("First sentence. Second sentence."))
    texts = [c["text"] for c in claims]
    assert "Seafront villa" in texts           # hero
    assert "Sleeps six" in texts               # highlight
    assert "A private pool." in texts          # amenity
    assert all(c["text"].strip() for c in claims)  # no empty fragments


def test_decimals_are_not_split_mid_number():
    # "4.96 out" has no space after the dot, so the score stays in one claim.
    about = [c["text"] for c in atomic_claims(_content("Rated 4.96 out of 5 by guests."))
             if c["field"] == "about"]
    assert any("4.96" in t for t in about)
    assert "4" not in [t.strip() for t in about]  # never split into a bare "4"


def test_trailing_and_repeated_whitespace_produce_no_empty_claims():
    about = [c for c in atomic_claims(_content("One sentence.   \n  Two sentence.  "))
             if c["field"] == "about"]
    assert [c["text"] for c in about] == ["One sentence.", "Two sentence."]


def test_known_abbreviation_overspit_is_a_documented_limitation():
    # ponytail: the naive splitter breaks on "St." -> this WILL over-split. Pinned so
    # a future segmenter swap is a conscious change, not a silent behaviour shift.
    about = [c["text"] for c in atomic_claims(_content("Near St. Mary church."))
             if c["field"] == "about"]
    assert about == ["Near St.", "Mary church."]  # known wrong; upgrade = real segmenter
