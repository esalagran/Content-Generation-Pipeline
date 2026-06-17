"""Generator unit tests — dependency injection + mocking, no API calls.

A fake MessageClient is injected into ContentGenerator, so the generation logic
(prompt -> parse) is tested in isolation from any real model.
"""

import pytest

from content_pipeline.fixtures import GOOD_VILLA, VILLA
from content_pipeline.generator import ContentGenerator
from content_pipeline.models import GeneratedContent
from content_pipeline.prompt import ContentParseError


class FakeClient:
    """Records the call and returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    def create_message(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response


def test_generate_parses_valid_output():
    client = FakeClient(GOOD_VILLA.model_dump_json())
    content = ContentGenerator(client).generate(VILLA)

    assert isinstance(content, GeneratedContent)
    assert content.hero_headline == GOOD_VILLA.hero_headline
    assert len(content.highlights) == len(GOOD_VILLA.highlights)
    # DI seam exercised: client was called at temperature 0 with the property in the prompt
    assert client.calls[0]["temperature"] == 0.0
    assert str(VILLA.property_id) in client.calls[0]["user"]


def test_generate_tolerates_code_fences():
    fenced = "```json\n" + GOOD_VILLA.model_dump_json() + "\n```"
    content = ContentGenerator(FakeClient(fenced)).generate(VILLA)
    assert isinstance(content, GeneratedContent)


def test_generate_raises_on_unparseable_output():
    with pytest.raises(ContentParseError):
        ContentGenerator(FakeClient("sorry, I can't help with that")).generate(VILLA)
