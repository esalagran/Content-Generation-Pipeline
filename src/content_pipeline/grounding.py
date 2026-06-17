"""Deterministic grounding checks — the cheap, robust pre-filter tier.

Pure logic, no Inspect dependency, so it is trivially unit-testable. The Inspect
@scorer wrappers live in scorers.py.

These checks deliberately cover only what is genuinely deterministic:
- citation validity (does each cited source field actually hold data?)
- structural quality (length, count, repetition, generic boilerplate)

Numeric and geographic grounding are NOT here: detecting "sleeps eight" or
"moments from the beach" by regex is brittle and misses the plausible
hallucinations that matter, so the LLM judge owns semantic grounding.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import unescape

from .models import GeneratedContent, PropertyData, SourceField


def strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", text))


def _decamel(code: str) -> str:
    """InternetBroadband -> 'Internet Broadband' so decoded amenity claims match."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", code)


def build_corpus(property: PropertyData) -> str:
    """Normalized union of ALL input: description (HTML-stripped), headline, every
    review, AND the structured fields (guest counts, review score, amenities decoded
    from their codes, policies, house rules). A landmark named in a review is
    grounded. (Reviews are weaker evidence than structured fields — see README.)"""
    p = property
    parts = [
        p.property_name,
        p.property_type,
        p.description.name,
        p.description.headline,
        strip_html(p.description.description),
        p.location.city,
        p.location.country,
        *p.amenities,
        *[_decamel(a) for a in p.amenities],
        *p.reviews,
        f"sleeps up to {p.rental_info.max_guests} guests",
        f"{p.rental_info.bedrooms} bedrooms",
        f"{p.rental_info.bathrooms} bathrooms",
        f"check-in {p.house_rules.check_in_time}, check-out {p.house_rules.check_out_time}",
    ]
    if p.num_of_reviews > 0:
        parts.append(
            f"average review score {p.average_review_score} from {p.num_of_reviews} reviews"
        )
    for label, value in [
        ("cancellation policy", p.policies.cancellation_policy),
        ("payment schedule", p.policies.payment_schedule),
        ("damage deposit", p.policies.damage_deposit),
    ]:
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts).lower()


@dataclass
class CheckResult:
    passed: bool
    failures: list[str]


class GroundingCheck(ABC):
    """Base check: holds the property + its grounding corpus, returns a
    CheckResult. Subclasses implement one concrete grounding dimension."""

    name: str

    def __init__(self, property: PropertyData):
        self.property = property
        self.corpus = build_corpus(property)

    @abstractmethod
    def run(self, content: GeneratedContent) -> CheckResult: ...


class CitationValidityCheck(GroundingCheck):
    """Accuracy: every highlight cites a source field that actually holds data,
    and every described amenity code is one of the property's input codes."""

    name = "citation_validity"

    def run(self, content: GeneratedContent) -> CheckResult:
        failures: list[str] = []
        for claim in content.highlights:
            if self._field_is_empty(claim.source_field):
                failures.append(
                    f"highlight cites empty field '{claim.source_field.value}': {claim.text!r}"
                )
        valid_codes = set(self.property.amenities)
        for amenity in content.amenity_descriptions:
            if amenity.code not in valid_codes:
                failures.append(f"amenity code not in input: {amenity.code!r}")
        return CheckResult(not failures, failures)

    def _field_is_empty(self, field: SourceField) -> bool:
        p = self.property
        present = {
            SourceField.description: bool(strip_html(p.description.description).strip()),
            SourceField.reviews: bool(p.reviews),
            SourceField.amenities: bool(p.amenities),
            SourceField.rental_info: True,
            SourceField.location: True,
            SourceField.policies: any(
                [p.policies.cancellation_policy, p.policies.payment_schedule,
                 p.policies.damage_deposit]
            ),
            SourceField.house_rules: True,
            SourceField.average_review_score: p.num_of_reviews > 0,
            SourceField.num_of_reviews: p.num_of_reviews > 0,
            SourceField.property_type: bool(p.property_type),
        }
        return not present[field]


class QualityCheck(GroundingCheck):
    """Quality: catches accurate-but-lifeless copy — bad lengths/counts, repeated
    phrasing, and generic boilerplate. A faithful but bland listing fails here."""

    name = "quality"

    GENERIC_PHRASES = [
        "nestled in the heart of",
        "home away from home",
        "something for everyone",
        "look no further",
        "a stone's throw",
        "whether you're looking",
        "the perfect getaway",
    ]

    def run(self, content: GeneratedContent) -> CheckResult:
        failures: list[str] = []

        headline_len = len(content.hero_headline)
        if not (10 <= headline_len <= 90):
            failures.append(f"hero_headline length {headline_len} outside 10-90")

        n_highlights = len(content.highlights)
        if not (3 <= n_highlights <= 5):
            failures.append(f"{n_highlights} highlights, expected 3-5")

        blob = " ".join(
            [content.hero_headline, content.about_section]
            + [c.text for c in content.highlights]
            + [a.text for a in content.amenity_descriptions]
        ).lower()

        for phrase in self.GENERIC_PHRASES:
            if phrase in blob:
                failures.append(f"generic phrase: {phrase!r}")

        repeated = self._repeated_trigram(blob)
        if repeated:
            failures.append(f"repeated phrasing: {repeated!r}")

        return CheckResult(not failures, failures)

    @staticmethod
    def _repeated_trigram(blob: str) -> str | None:
        # ponytail: naive trigram-frequency heuristic; flags the worst offender.
        words = re.findall(r"[a-z0-9']+", blob)
        counts: dict[str, int] = {}
        for i in range(len(words) - 2):
            tri = " ".join(words[i : i + 3])
            counts[tri] = counts.get(tri, 0) + 1
        worst = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
        return worst[0] if worst[1] >= 3 else None


DETERMINISTIC_CHECKS: list[type[GroundingCheck]] = [CitationValidityCheck, QualityCheck]
