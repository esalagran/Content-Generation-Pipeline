"""Pydantic models for property input and generated marketing copy.

The key design move is `Claim.source_field`: every structured claim cites the
input field it came from, which turns grounding from an LLM guess into a
deterministic citation-validity check (see grounding.py).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


# --- Input: property data (mirrors the assignment spec) ---


class Description(BaseModel):
    name: str
    headline: str
    description: str  # may contain HTML


class RentalInfo(BaseModel):
    max_guests: int
    bedrooms: int
    bathrooms: int


class Location(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float


class Policies(BaseModel):
    cancellation_policy: str | None = None
    payment_schedule: str | None = None
    damage_deposit: str | None = None


class HouseRules(BaseModel):
    check_in_time: str
    check_out_time: str


class PropertyData(BaseModel):
    property_id: int
    property_name: str
    property_type: str
    description: Description
    amenities: list[str]
    image_urls: list[str]
    reviews: list[str]
    num_of_reviews: int
    average_review_score: float
    rental_info: RentalInfo
    location: Location
    policies: Policies
    house_rules: HouseRules


# --- Output: generated marketing copy ---


class SourceField(str, Enum):
    """Input fields a structured claim is allowed to cite. `image_urls` is
    deliberately absent: visual claims are not groundable against text."""

    description = "description"
    reviews = "reviews"
    amenities = "amenities"
    rental_info = "rental_info"
    location = "location"
    policies = "policies"
    house_rules = "house_rules"
    average_review_score = "average_review_score"
    num_of_reviews = "num_of_reviews"
    property_type = "property_type"


class Claim(BaseModel):
    """A short marketing statement bound to the input field it derives from."""

    text: str
    source_field: SourceField


class AmenityDescription(BaseModel):
    """A natural-language description of one amenity. `code` must be one of the
    property's input amenity codes (checked deterministically)."""

    code: str
    text: str


class GeneratedContent(BaseModel):
    hero_headline: str
    highlights: list[Claim]
    about_section: str  # free prose -> grounded by the judge, not by citation
    amenity_descriptions: list[AmenityDescription]
