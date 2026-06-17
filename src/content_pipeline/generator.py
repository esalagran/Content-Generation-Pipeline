"""Content generator with an injected model client (the DI seam).

The Inspect solver generates through Inspect's own model layer (eval_task.py);
this class exists for the non-Inspect path and is what the unit tests exercise
with a fake client — satisfying the dependency-injection / mocking requirement
without coupling tests to a real API.
"""

from __future__ import annotations

from typing import Protocol

from .models import GeneratedContent, PropertyData
from .prompt import build_system, build_user, parse_output

DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2048


class MessageClient(Protocol):
    """Minimal slice of the Anthropic client surface the generator depends on.
    Anything matching this shape (real SDK or a test fake) can be injected."""

    def create_message(self, *, model: str, max_tokens: int, temperature: float,
                       system: str, user: str) -> str:
        ...


class ContentGenerator:
    def __init__(self, client: MessageClient, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def generate(self, property: PropertyData) -> GeneratedContent:
        raw = self.client.create_message(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            system=build_system(),
            user=build_user(property),
        )
        return parse_output(raw)


class AnthropicClient:
    """Adapts the real Anthropic SDK to MessageClient. Not used in tests or in
    the offline replay path, so it is never imported without a key."""

    def __init__(self, sdk_client):
        self._sdk = sdk_client

    def create_message(self, *, model, max_tokens, temperature, system, user) -> str:
        resp = self._sdk.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
