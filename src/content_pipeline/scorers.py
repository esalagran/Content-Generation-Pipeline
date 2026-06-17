"""Inspect @scorer adapters.

Deterministic scorers wrap the pure checks in grounding.py and need no model, so
they re-execute fully offline via the replay solver. The judge scorer calls a
model and emits per-claim verdicts into Score.metadata (round-trips through the
.eval log — verified in the step-0 spike) so judge-vs-gold is reconstructable
offline from the committed log.
"""

from __future__ import annotations

import json
import re

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    mean,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .grounding import CitationValidityCheck, GroundingCheck, QualityCheck, build_corpus
from .models import GeneratedContent, PropertyData
from .prompt import ContentParseError, parse_output


def _property(state: TaskState) -> PropertyData:
    return PropertyData.model_validate(state.metadata["property"])


def check_score_fn(check_cls: type[GroundingCheck]):
    async def score(state: TaskState, target: Target) -> Score:
        prop = _property(state)
        try:
            content = parse_output(state.output.completion)
        except ContentParseError as e:
            return Score(value=INCORRECT, explanation=f"parse failure: {e}")
        result = check_cls(prop).run(content)
        return Score(
            value=CORRECT if result.passed else INCORRECT,
            explanation="; ".join(result.failures) or f"{check_cls.name}: pass",
            metadata={"failures": result.failures},
        )

    return score


@scorer(name="citation_validity", metrics=[accuracy(), stderr()])
def citation_validity_scorer():
    return check_score_fn(CitationValidityCheck)


@scorer(name="quality", metrics=[accuracy(), stderr()])
def quality_scorer():
    return check_score_fn(QualityCheck)


# --- LLM judge: atomic-claim faithfulness (the grounding centerpiece) ---


def atomic_claims(content: GeneratedContent) -> list[dict]:
    """Decompose generated copy into atomic, individually-checkable statements.

    NOTE (validation gap, stated in README): the gold set validates the judge's
    *verdict*, not this decomposition. Decomposition is assumed-correct here and
    only spot-checked manually."""
    claims = [{"field": "hero_headline", "text": content.hero_headline}]
    for c in content.highlights:
        claims.append({"field": f"highlight:{c.source_field.value}", "text": c.text})
    for sentence in re.split(r"(?<=[.!?])\s+", content.about_section.strip()):
        if sentence.strip():
            claims.append({"field": "about", "text": sentence.strip()})
    for a in content.amenity_descriptions:
        claims.append({"field": f"amenity:{a.code}", "text": a.text})
    return claims


_JUDGE_SYSTEM = """\
You verify whether vacation-rental marketing claims are grounded in source facts.
For each numbered claim decide exactly one verdict:
- "entailed": directly supported by the source facts.
- "contradicted": conflicts with the source facts.
- "unsupported": no basis in the source facts (neither supported nor contradicted).
Approximations consistent with the data are entailed (e.g. "nearly 5 stars" for 4.96).
Output ONLY a JSON array, one object per claim:
[{"id": 0, "verdict": "entailed", "reason": "..."}]"""


def _judge_user(corpus: str, claims: list[dict]) -> str:
    listed = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims))
    return f"SOURCE FACTS:\n{corpus}\n\nCLAIMS:\n{listed}"


def _parse_verdicts(raw: str, n: int) -> tuple[list[str], int]:
    """Map judge output to a verdict per claim id; missing -> 'unsupported'.
    Also returns the number of claims left at the default (judge gave no parseable
    verdict) so a format regression is visible, not silently scored as unsupported."""
    verdicts = ["unsupported"] * n
    seen: set[int] = set()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return verdicts, n
    try:
        for item in json.loads(match.group(0)):
            i = item.get("id")
            v = item.get("verdict")
            if isinstance(i, int) and 0 <= i < n and v in {"entailed", "contradicted", "unsupported"}:
                verdicts[i] = v
                seen.add(i)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return verdicts, n - len(seen)


@scorer(name="faithfulness", metrics=[mean(), stderr()])
def faithfulness_scorer(grader_model: str | None = None):
    """One judge call per generation returns a verdict for every atomic claim.
    value = fraction entailed. Per-claim verdicts go to Score.metadata so the
    committed log alone reconstructs judge-vs-gold offline."""

    async def score(state: TaskState, target: Target) -> Score:
        prop = _property(state)
        try:
            content = parse_output(state.output.completion)
        except ContentParseError as e:
            return Score(value=0.0, explanation=f"parse failure: {e}", metadata={"verdicts": []})

        claims = atomic_claims(content)
        grader = get_model(grader_model) if grader_model else get_model()
        out = await grader.generate(
            [
                ChatMessageSystem(content=_JUDGE_SYSTEM),
                ChatMessageUser(content=_judge_user(build_corpus(prop), claims)),
            ],
            config=GenerateConfig(temperature=0.0, max_tokens=1500),
        )
        verdicts, unparsed = _parse_verdicts(out.completion, len(claims))

        per_claim = [
            {"claim": c["text"], "field": c["field"], "verdict": v}
            for c, v in zip(claims, verdicts)
        ]
        entailed = verdicts.count("entailed")
        contradicted = verdicts.count("contradicted")
        faithfulness = entailed / len(claims) if claims else 1.0
        return Score(
            value=faithfulness,
            answer=f"{entailed}/{len(claims)} entailed",
            explanation=(
                f"{entailed} entailed, {contradicted} contradicted, "
                f"{len(claims) - entailed - contradicted} unsupported"
            ),
            metadata={"verdicts": per_claim, "unparsed": unparsed},
        )

    return score


# --- LLM judge: brand voice (idiomatic use of the built-in model_graded_qa) ---

_BRAND_TEMPLATE = """\
You are grading the brand voice of vacation-rental listing copy.

Property data:
{question}

Generated copy:
{answer}

{instructions}"""

_BRAND_INSTRUCTIONS = """\
Grade whether the copy is specific, vivid, and free of generic filler (not bland
boilerplate). Reply with 'GRADE: C' (good), 'GRADE: P' (mediocre/partial), or
'GRADE: I' (generic/poor), then a one-line justification."""


def brand_voice_scorer(grader_model: str | None = None):
    return model_graded_qa(
        template=_BRAND_TEMPLATE,
        instructions=_BRAND_INSTRUCTIONS,
        partial_credit=True,
        model=grader_model,
    )
