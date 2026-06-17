"""Inspect @scorer adapters.

Deterministic scorers wrap the pure checks in grounding.py and need no model, so
they re-execute fully offline via the replay solver. The judge scorers call a model
and write per-claim verdicts into Score.metadata, so judge-vs-human agreement is
reconstructable offline from the committed .eval log.
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
    SampleScore,
    Score,
    Target,
    accuracy,
    metric,
    model_graded_qa,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import TaskState

from .grounding import (
    CitationValidityCheck,
    GroundingCheck,
    QualityCheck,
    build_corpus,
    salient_facts,
)
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
    """Split generated copy into atomic, individually-checkable statements: the
    headline, each highlight, each about-section sentence, each amenity description."""
    claims = [{"field": "hero_headline", "text": content.hero_headline}]
    for c in content.highlights:
        claims.append({"field": f"highlight:{c.source_field.value}", "text": c.text})
    # Naive sentence split: decimals are safe ("4.96 by" has no space after the dot),
    # but abbreviations like "St. Mary" over-split (a known limit, pinned in tests).
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


_FAITHFULNESS_VERDICTS = frozenset({"entailed", "contradicted", "unsupported"})


def _parse_verdicts(
    raw: str,
    n: int,
    allowed: frozenset[str] = _FAITHFULNESS_VERDICTS,
    default: str = "unsupported",
) -> tuple[list[str], int]:
    """Map judge output to a verdict per item id; missing -> `default`.
    Also returns the number of items left at the default (judge gave no parseable
    verdict) so a format regression is visible, not silently scored as default."""
    verdicts = [default] * n
    seen: set[int] = set()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return verdicts, n
    try:
        for item in json.loads(match.group(0)):
            i = item.get("id")
            v = item.get("verdict")
            if isinstance(i, int) and 0 <= i < n and v in allowed:
                verdicts[i] = v
                seen.add(i)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return verdicts, n - len(seen)


# --- judge metrics: read per-claim verdicts + sample metadata from the .eval log ---


_to_float = value_to_float()


def _judge_errored(s: SampleScore) -> bool:
    return bool((s.score.metadata or {}).get("judge_error"))


def _verdicts(s: SampleScore) -> list[dict]:
    return (s.score.metadata or {}).get("verdicts") or []


@metric
def judge_value_mean():
    """Mean judge score (faithfulness or coverage), EXCLUDING samples whose judge
    output was unparseable — those are infra failures, not content failures."""

    def compute(scores: list[SampleScore]) -> float:
        vals = [_to_float(s.score.value) for s in scores if not _judge_errored(s)]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@metric
def judge_error_rate():
    """Fraction of samples whose judge output was unparseable — a SEPARATE signal so
    a flaky judge never masquerades as a content score of 0."""

    def compute(scores: list[SampleScore]) -> float:
        return sum(_judge_errored(s) for s in scores) / len(scores) if scores else 0.0

    return compute


@metric
def judge_detection_rate():
    """Per-claim recall: of the claims each sample's `expect_judge_flag` marks as
    planted hallucinations, the fraction the judge actually flagged (not 'entailed')."""

    def compute(scores: list[SampleScore]) -> float:
        caught = total = 0
        for s in scores:
            if _judge_errored(s):
                continue
            expected = (s.sample_metadata or {}).get("expect_judge_flag") or []
            if not expected:
                continue
            vmap = {v["claim"]: v["verdict"] for v in _verdicts(s)}
            for claim in expected:
                total += 1
                if vmap.get(claim, "unsupported") != "entailed":
                    caught += 1
        return caught / total if total else 0.0

    return compute


@metric
def judge_false_positive_rate():
    """Per-claim over-flagging: of all claims on genuinely-clean `good` samples, the
    fraction the judge wrongly flagged (the honest over-flag rate)."""

    def compute(scores: list[SampleScore]) -> float:
        flagged = total = 0
        for s in scores:
            if _judge_errored(s):
                continue
            if (s.sample_metadata or {}).get("label") != "good":
                continue
            for v in _verdicts(s):
                total += 1
                if v["verdict"] != "entailed":
                    flagged += 1
        return flagged / total if total else 0.0

    return compute


# --- judge coroutine, shared by the production and meta faithfulness scorers ---


def _faithfulness_score_fn(grader_model: str | None):
    async def score(state: TaskState, target: Target) -> Score:
        prop = _property(state)
        try:
            content = parse_output(state.output.completion)
        except ContentParseError as e:
            # generation parse failure: a real content failure (faithfulness 0),
            # NOT a judge-infra failure.
            return Score(
                value=0.0,
                explanation=f"parse failure: {e}",
                metadata={"verdicts": [], "judge_error": False},
            )

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
        judge_error = bool(claims) and unparsed == len(claims)

        per_claim = [
            {"claim": c["text"], "field": c["field"], "verdict": v}
            for c, v in zip(claims, verdicts)
        ]
        entailed = verdicts.count("entailed")
        contradicted = verdicts.count("contradicted")
        explanation = (
            f"{entailed} entailed, {contradicted} contradicted, "
            f"{len(claims) - entailed - contradicted} unsupported"
        )
        if judge_error:
            explanation = "JUDGE PARSE FAILURE — not a content signal; " + explanation
        return Score(
            value=entailed / len(claims) if claims else 1.0,
            answer=f"{entailed}/{len(claims)} entailed",
            explanation=explanation,
            metadata={"verdicts": per_claim, "unparsed": unparsed, "judge_error": judge_error},
        )

    return score


@scorer(name="faithfulness", metrics=[judge_value_mean(), judge_error_rate(), stderr()])
def faithfulness_scorer(grader_model: str | None = None):
    """Production judge: one call per generation, value = fraction of claims entailed.
    Pass grader_model to judge with a different model than the generator."""
    return _faithfulness_score_fn(grader_model)


@scorer(
    name="faithfulness",
    metrics=[judge_detection_rate(), judge_false_positive_rate(), judge_error_rate()],
)
def meta_faithfulness_scorer(grader_model: str | None = None):
    """Same judge logic, GATED by the metrics that validate it: per-claim detection
    on planted hallucinations and false-positives on clean copy."""
    return _faithfulness_score_fn(grader_model)


# --- LLM judge: coverage (recall complement of faithfulness) ---

_COVERAGE_SYSTEM = """\
You judge whether vacation-rental listing copy USES the property's strongest
selling points. You are given a numbered list of salient facts and the generated
copy. For each fact decide exactly one verdict:
- "used": the copy clearly conveys this fact or its benefit. Paraphrase is fine;
  approximations consistent with the fact count (e.g. "nearly 5 stars" for 4.96).
- "missed": the copy omits this selling point entirely.
Output ONLY a JSON array, one object per fact:
[{"id": 0, "verdict": "used", "reason": "..."}]"""


def _copy_text(content: GeneratedContent) -> str:
    parts = [content.hero_headline, content.about_section]
    parts += [c.text for c in content.highlights]
    parts += [f"{a.code}: {a.text}" for a in content.amenity_descriptions]
    return "\n".join(parts)


def _coverage_user(facts: list[dict], copy_text: str) -> str:
    listed = "\n".join(f"{f['id']}. {f['fact']}" for f in facts)
    return f"SALIENT FACTS:\n{listed}\n\nGENERATED COPY:\n{copy_text}"


@scorer(name="coverage", metrics=[judge_value_mean(), judge_error_rate(), stderr()])
def coverage_scorer(grader_model: str | None = None):
    """Of the property's strongest selling points (salient_facts), the fraction
    the copy surfaces. value = used / total. Per-fact verdicts go to
    Score.metadata so the committed log alone reconstructs the judgement offline.
    Shares the judge-error separation: an unparseable judge response is flagged
    judge_error (excluded from judge_value_mean), not silently scored 'all missed'."""

    async def score(state: TaskState, target: Target) -> Score:
        prop = _property(state)
        try:
            content = parse_output(state.output.completion)
        except ContentParseError as e:
            return Score(value=0.0, explanation=f"parse failure: {e}",
                         metadata={"verdicts": [], "judge_error": False})

        facts = salient_facts(prop)
        if not facts:
            return Score(value=1.0, explanation="no salient facts",
                         metadata={"verdicts": [], "judge_error": False})

        grader = get_model(grader_model) if grader_model else get_model()
        out = await grader.generate(
            [
                ChatMessageSystem(content=_COVERAGE_SYSTEM),
                ChatMessageUser(content=_coverage_user(facts, _copy_text(content))),
            ],
            config=GenerateConfig(temperature=0.0, max_tokens=1000),
        )
        verdicts, unparsed = _parse_verdicts(
            out.completion, len(facts), allowed=frozenset({"used", "missed"}), default="missed"
        )
        judge_error = unparsed == len(facts)

        per_fact = [
            {"fact": f["fact"], "kind": f["kind"], "verdict": v}
            for f, v in zip(facts, verdicts)
        ]
        used = verdicts.count("used")
        explanation = f"{used} of {len(facts)} salient facts surfaced"
        if judge_error:
            explanation = "JUDGE PARSE FAILURE — not a content signal; " + explanation
        return Score(
            value=used / len(facts),
            answer=f"{used}/{len(facts)} used",
            explanation=explanation,
            metadata={"verdicts": per_fact, "unparsed": unparsed, "judge_error": judge_error},
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
