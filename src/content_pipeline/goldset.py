"""Human-labeled gold set for judge-vs-human agreement.

Each entry is a claim drawn from a committed run, labeled by a human (me) as
grounded (`supported=True`) or not. The notebook looks up the judge's verdict for
the same claim in the committed .eval log and builds a confusion matrix.

Agreement is reported as Cohen's kappa + TPR/TNR (not raw agreement, which is
misleading under class imbalance) in report.py.

HONEST LIMITATIONS (do not over-read these numbers):
- n is tiny (a demonstration, not a significant estimate). With ~12 claims the
  confidence interval is very wide; treat this as methodology that scales to
  hundreds, not a trustworthy point estimate (kappa included — its CI is huge here).
- Single annotator (me, who also wrote the judge prompt) => no inter-annotator
  agreement and built-in bias.
- The labels validate the judge's *verdict*, not the upstream claim
  *decomposition* (validated separately in tests/test_decomposition.py).

`claim` strings must match the logged claim text exactly (lookup is by text).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldLabel:
    source: str  # "generation" or "meta" — which committed task's log to read
    sample_id: str
    claim: str
    supported: bool  # human judgment
    note: str = ""


GOLD: list[GoldLabel] = [
    # --- supported claims (human=True) from real generations + good fixtures ---
    # NOTE: the "generation"-source claim texts below were re-derived from the
    # committed claude-sonnet-4-6 run (they must match the logged text verbatim).
    GoldLabel("generation", "villa",
              "Sleeps up to 6 guests across 3 bedrooms and 2 bathrooms.", True),
    GoldLabel("generation", "villa",
              "Rated 4.96 out of 5 across 87 reviews — guests consistently praise the sea views and spotless pool.",
              True),
    GoldLabel("generation", "cottage",
              "Stay connected with broadband internet throughout your stay.", True),
    GoldLabel("meta", "good_villa", "Three bedrooms sleeping up to six guests", True),
    GoldLabel("meta", "good_villa", "Rated 4.96 across 87 guest reviews", True),
    # judge over-flags these legitimate claims -> expected human/judge disagreement.
    # The amenity IS in the input list; the judge marks the descriptive flourish
    # 'unsupported' because the embellishment isn't literally in the corpus.
    GoldLabel("generation", "villa",
              "Free on-site parking means one less thing to worry about on arrival.",
              True, "FreeParking is in amenities; judge over-flags the descriptive flourish"),
    GoldLabel("generation", "cottage",
              "Check in from 4 PM and check out by 10 AM, giving you plenty of time to enjoy the tranquillity of the area.",
              True, "check-in/out times are in house_rules; judge over-flags the embellishment"),
    GoldLabel("meta", "good_villa", "Air conditioning throughout for warm summers.", True,
              "AC is in amenities and praised in a review"),

    # --- unsupported / hallucinated claims (human=False) ---
    GoldLabel("meta", "bad_fabricated_landmark",
              "A five-minute walk to the Sagrada Familia", False),
    GoldLabel("meta", "bad_fabricated_landmark",
              "Steps from the marina and a short walk to Barcelona's Sagrada Familia.", False),
    GoldLabel("meta", "bad_wrong_number",
              "Five bedrooms sleeping up to twelve guests", False, "contradicts rental_info"),
    GoldLabel("meta", "bad_unlisted_amenity", "A private helipad on site.", False,
              "injection attempt in a review; not a real fact"),
]
