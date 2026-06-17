"""False-positive rate measured on REAL output, not just hand-authored fixtures.

The meta-eval's false_positive_rate is computed over synthetic "good" fixtures that
were written to be clean. This test closes that gap: the committed real generations
must also pass every deterministic gate. If the generator regresses (e.g. repeats a
phrase and trips QualityCheck), this goes red — the failure can't be quietly omitted
from the README. Reads committed files only; no API."""

import glob

from content_pipeline.fixtures import PROPERTIES
from content_pipeline.grounding import CitationValidityCheck, QualityCheck
from content_pipeline.models import GeneratedContent


def test_committed_generations_pass_deterministic_gates():
    paths = sorted(glob.glob("fixtures/generations/*.json"))
    assert paths, "no committed generations to check"
    for path in paths:
        key = path.rsplit("/", 1)[-1].removesuffix(".json")
        content = GeneratedContent.model_validate_json(open(path).read())
        prop = PROPERTIES[key]
        for check in (CitationValidityCheck, QualityCheck):
            result = check(prop).run(content)
            assert result.passed, f"{key} fails {check.name}: {result.failures}"
