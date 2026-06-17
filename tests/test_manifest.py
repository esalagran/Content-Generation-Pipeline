"""Guards the committed generations against silent prompt drift: if the prompt
changes, the recorded prompt_hash no longer matches and the committed .eval logs
are stale. Reads the manifest only — no API."""

import json

from content_pipeline.fixtures import PROPERTIES
from content_pipeline.prompt import prompt_hash


def test_manifest_prompt_hashes_match_current_prompt():
    manifest = json.load(open("fixtures/manifest.json"))
    drifted = [
        k for k, p in PROPERTIES.items()
        if prompt_hash(p) != manifest["generations"][k]["prompt_hash"]
    ]
    assert not drifted, f"prompt changed; committed generations stale for: {drifted}"
