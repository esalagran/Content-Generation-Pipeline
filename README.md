# Property Listing Content Generator — Eval-First

Generates structured marketing copy (hero headline, highlights, "about", amenity
descriptions) from vacation-rental property data — built so that **the evaluation
suite is the primary artifact**. The generator is the thing under test.

The whole evaluation runs **fully offline with zero API calls**: deterministic
scorers re-execute live; LLM-judge results are read from committed `inspect-ai`
logs.

## Quickstart

```bash
uv sync
uv run pytest                          # unit tests, offline
uv run inspect view --log-dir logs     # browse the committed real runs
```

Open `evals.ipynb` to see the full story (already executed; outputs committed).
Re-run it offline with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace evals.ipynb
```

To regenerate the committed logs (needs an Anthropic key in `.env`):

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
uv run python -c "from inspect_ai import eval; from content_pipeline.eval_task import generation_eval, meta_eval; \
  eval(generation_eval(), model='anthropic/claude-sonnet-4-5', log_dir='logs'); \
  eval(meta_eval(include_judge=True), model='anthropic/claude-sonnet-4-5', log_dir='logs')"
```

## How to read the results

- **`uv run inspect view --log-dir logs`** opens the Inspect log viewer. There are
  two committed real runs:
  - `generation_eval` — real Anthropic generations scored by all four scorers.
  - `meta_eval` — the meta-evaluation (below), including judge verdicts.
- **`evals.ipynb`** is the narrative: a sample generation, production scores, the
  meta-eval, and the judge-vs-human confusion matrix — each cell labeled as either
  a *committed real run* or a *live offline recompute*.

## Approach

### Generation
Strict-JSON prompting validated by Pydantic (`models.py`, `prompt.py`), shared by
the Inspect solver and the DI `ContentGenerator` so there is one code path. Each
structured highlight carries a `source_field` citing the input it derives from —
this binding is what makes grounding mechanically checkable.

> Trade-off: I used strict-JSON prompting rather than provider tool-forcing.
> Inspect's `ToolDef` builds its schema from a callable signature, which is awkward
> for a nested schema; JSON-prompting keeps a single provider-agnostic path and
> makes parse failure an explicit, observable outcome (`ContentParseError`) instead
> of something silently retried away.

### Evaluation — three tiers
1. **Deterministic pre-filter** (`grounding.py`, no model, recomputable offline):
   - `CitationValidityCheck` — every highlight cites a field that holds data; every
     amenity code is one of the input codes.
   - `QualityCheck` — length/format bounds, n-gram repetition, generic-phrase
     blocklist. Catches accurate-but-bland boilerplate.
2. **LLM judge** (`scorers.py`): atomic-claim **faithfulness** (each statement
   judged entailed / contradicted / unsupported against the full input corpus,
   RAGAS-style) plus a `model_graded_qa` brand-voice grader. Per-claim verdicts are
   written to `Score.metadata`, so the committed log alone reconstructs
   judge-vs-human agreement offline.
   - Numeric and geographic grounding live here, not in the deterministic tier:
     detecting "sleeps eight" or "moments from the beach" by regex is brittle and
     misses the plausible hallucinations that matter.
3. **Meta-eval** (`eval_task.py: meta_eval`): **validates the evaluator itself.**
   A replay solver feeds hand-labeled good/bad generations through the scorers and
   custom `@metric`s report **detection rate** (recall) and **false-positive rate**
   (precision). The judge-vs-human confusion matrix compares committed verdicts to a
   gold set (`goldset.py`).

### Results (this committed run)
- Deterministic: **detection 100% / false-positive 0%** on the labeled set.
- Judge faithfulness on real generations: **0.96**.
- Judge on planted hallucinations: catches every one (fabricated landmark, wrong
  guest count, and a **prompt-injection attempt** — "private helipad" injected via a
  review — all marked *contradicted*).
- Judge vs human gold set (n=13): **recall 100%, precision 71%** — it over-flags two
  legitimate claims ("high-speed broadband", "air conditioning throughout").

The eval also caught **a bug in my own code**: the grounding corpus initially
omitted `rental_info` and review counts, so the judge marked valid claims
unsupported (good-copy faithfulness 0.77). Completing the corpus moved it to 0.96.
That is the eval-first loop doing its job.

## Reproducibility
Single mechanism: a **replay solver** returns committed generations by sample id, so
deterministic scorers re-run inside Inspect with zero API and no second scoring path
to drift. The judge needs a key, so its verdicts are read from the committed log.
`inspect-ai` is pinned exactly (`==0.3.240`) because `inspect view` is
version-sensitive; `fixtures/manifest.json` records model id, prompt hash, and
timestamp for each committed generation.

## Engineering notes
- **Dependency injection / mocking**: `ContentGenerator` takes a `MessageClient`;
  tests inject a fake (`tests/test_generator.py`). No test touches the network.
- **Inheritance**: `GroundingCheck` ABC + two concrete checks sharing corpus access
  and result type. Used where it earns its place — not manufactured.
- **Inspect-native**: `Dataset`/`Sample`, `@scorer`/`@metric`, `model_graded_qa`,
  `value_to_float`, per-claim `Score.metadata`.

## How I used AI
Built with Claude Code (the agent) in plan mode. I drove several adversarial
critique rounds against my own plan before writing code — each round is reflected in
the architecture (replay solver instead of `mockllm`; cutting a brittle numeric
regex check; making false-positive rate first-class; the meta-eval as the
centerpiece). The eval-first loop above (corpus bug found by the judge) was likewise
caught by running the suite, not by inspection.

## What I deliberately deferred (and why)
3–4h forces choices; I spent depth on eval *validation* over generation breadth.

- **Omission / coverage** — *a primary dimension I ran out of time for*, not a niche.
  A faithful listing that forgets the 4.96 score or the standout amenity is a
  business failure. Faithfulness is necessary, not sufficient.
- **Grounding-source limits** — `image_urls` excluded (visual claims aren't
  groundable against text without vision). Reviews are *weaker* grounding than
  structured fields (complaints, contradictions, PII); a review mentioning a landmark
  grounds a fact, but UGC in the corpus is also a poisoning vector (see the injection
  fixture).
- **Multilingual** — Barcelona property → EN/ES? mixed-language reviews? Unaddressed.
- **Failure modes** — parse failure is defined (fail the sample); refusal /
  truncation / rate-limit handling is not built out.
- **Regression gating** — the suite's production *purpose* is to gate prompt/model
  changes with pass thresholds over two `.eval` runs; not wired into CI here.
- **Judge cost** — atomic-claim judging is the scaling bottleneck (one judge call per
  generation here; verdict quality vs. cost is the real production lever).
- **Statistical power** — n is a *demonstration* size. Per-check rates and the
  confusion matrix are methodology that scales to hundreds, not significant point
  estimates; the gold set has a single annotator (me, who wrote the judge prompt) so
  there is no inter-annotator agreement; and the gold set validates judge *verdicts*,
  not the upstream claim *decomposition*.
```
