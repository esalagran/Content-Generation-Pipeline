"""Inspect tasks.

Task 1 (generation_eval): generate copy from a property via Inspect's model layer,
then score it. Run with a real model; logs are committed.

Task 2 (meta_eval): the point of the submission — validate the evaluator. A replay
solver returns committed/labeled generations (zero API), the same scorers run, and
custom metrics report detection rate (recall) and false-positive rate (precision).
The deterministic scorers re-execute fully offline; pass include_judge=True (with a
key) to also exercise the judge, whose verdicts are then read from the committed log.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.scorer import SampleScore, metric, scorer, value_to_float
from inspect_ai.solver import Generate, TaskState, generate, solver

from .fixtures import META_GENERATIONS, PROPERTIES
from .grounding import CitationValidityCheck, QualityCheck
from .prompt import build_system, build_user
from .scorers import (
    check_score_fn,
    brand_voice_scorer,
    citation_validity_scorer,
    coverage_scorer,
    faithfulness_scorer,
    meta_faithfulness_scorer,
    quality_scorer,
)

# system_message is set via config (NOT the system_message() solver) because the
# prompt embeds a JSON schema full of literal braces, which the solver would try
# to treat as template variables.
GENERATE_CONFIG = GenerateConfig(
    temperature=0.0, max_tokens=2048, system_message=build_system()
)
DETERMINISTIC_SCORERS = [citation_validity_scorer(), quality_scorer()]


@solver
def replay_generation():
    """Return the generation stored on the sample — no model call."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(
            model="replay", content=state.metadata["generation"]
        )
        return state

    return solve


# --- Task 1: generation eval ---


@task
def generation_eval() -> Task:
    samples = [
        Sample(
            input=build_user(prop),
            id=key,
            metadata={"property": prop.model_dump(), "property_key": key},
        )
        for key, prop in PROPERTIES.items()
    ]
    return Task(
        dataset=samples,
        solver=[generate()],
        scorer=DETERMINISTIC_SCORERS
        + [faithfulness_scorer(), coverage_scorer(), brand_voice_scorer()],
        config=GENERATE_CONFIG,
        name="generation_eval",
    )


# --- Meta-eval metrics: read each sample's label/expectation from metadata ---


_to_float = value_to_float()  # CORRECT -> 1.0, INCORRECT -> 0.0


def _fired(s: SampleScore) -> bool:
    """A scorer 'fired' (flagged a defect) when it did NOT return the pass value.
    Metrics receive values already coerced to float, so compare numerically."""
    return _to_float(s.score.value) < 0.5


@metric
def detection_rate(check_name: str):
    """Among samples this check SHOULD catch, the fraction it caught (recall)."""

    def compute(scores: list[SampleScore]) -> float:
        relevant = [
            s for s in scores
            if check_name in ((s.sample_metadata or {}).get("expect_deterministic") or [])
        ]
        return sum(_fired(s) for s in relevant) / len(relevant) if relevant else 0.0

    return compute


@metric
def false_positive_rate():
    """Among good samples, the fraction this check wrongly fired on (1 - precision)."""

    def compute(scores: list[SampleScore]) -> float:
        good = [s for s in scores if (s.sample_metadata or {}).get("label") == "good"]
        return sum(_fired(s) for s in good) / len(good) if good else 0.0

    return compute


def _meta_scorer(check_cls, name: str):
    """A deterministic check scored for the meta-eval, with recall + FP metrics.
    Reuses the same check logic as the production scorers (scorers.py)."""

    @scorer(name=f"meta_{name}", metrics=[detection_rate(name), false_positive_rate()])
    def s():
        return check_score_fn(check_cls)

    return s()


@task
def meta_eval(include_judge: bool = False) -> Task:
    samples = []
    for g in META_GENERATIONS:
        prop = PROPERTIES[g.property_key]
        samples.append(
            Sample(
                input=g.key,
                id=g.key,
                metadata={
                    "property": prop.model_dump(),
                    "generation": g.raw,
                    "label": g.label,
                    "expect_deterministic": list(g.expect_deterministic),
                    "expect_judge_flag": list(g.expect_judge_flag),
                    "judge_failure": g.judge_failure,
                    "note": g.note,
                },
            )
        )

    scorers = [
        _meta_scorer(CitationValidityCheck, "citation_validity"),
        _meta_scorer(QualityCheck, "quality"),
    ]
    if include_judge:
        # meta_faithfulness_scorer = same judge call, but GATED by per-claim
        # detection/false-positive metrics (not just reported mean).
        scorers.append(meta_faithfulness_scorer())
        scorers.append(coverage_scorer())

    return Task(
        dataset=samples,
        solver=[replay_generation()],
        scorer=scorers,
        config=GENERATE_CONFIG,
        name="meta_eval",
    )
