"""Human-labeled gold set for judge-vs-human agreement.

Each entry is a claim drawn from a committed run, labeled by a human (me) as
grounded (`supported=True`) or not. The notebook looks up the judge's verdict for
the same claim in the committed .eval log and builds a confusion matrix.

HONEST LIMITATIONS (do not over-read these numbers):
- n is tiny (a demonstration, not a significant estimate). With ~13 claims the
  confidence interval is very wide; treat this as methodology that scales to
  hundreds, not a trustworthy point estimate.
- Single annotator (me, who also wrote the judge prompt) => no inter-annotator
  agreement and built-in bias.
- The labels validate the judge's *verdict*, not the upstream claim
  *decomposition* (assumed correct; spot-checked only).

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
    GoldLabel("generation", "villa",
              "Three-bedroom villa sleeping up to 6 guests with 2 bathrooms", True),
    GoldLabel("generation", "villa", "Rated 4.96 out of 5 by 87 guests", True),
    GoldLabel("generation", "villa", "Ten-minute walk to the old town", True,
              "grounded in a guest review"),
    GoldLabel("generation", "cottage",
              "Broadband internet access keeps you connected throughout your stay.", True),
    GoldLabel("meta", "good_villa", "Three bedrooms sleeping up to six guests", True),
    GoldLabel("meta", "good_villa", "Rated 4.96 across 87 guest reviews", True),
    # judge over-flags these legitimate claims -> expected human/judge disagreement:
    GoldLabel("generation", "villa",
              "Stay connected with high-speed broadband internet access throughout the property.",
              True, "broadband IS in amenities; 'high-speed' is mild puffery"),
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
    GoldLabel("meta", "good_cottage", "A wood-burning fireplace.", False,
              "'wood-burning' detail not in data"),
]
