# Property Listing Content Generator: an Eval-First Pipeline

This system turns structured vacation-rental data into marketing copy (hero
headline, highlights, an "about" section, amenity descriptions). But the copy
generator is the easy half. The **deliverable is the evaluation suite**, and,
specifically, a suite that has been turned on *itself* to show it actually measures
what it claims to.

The guiding principle: **the generator is the thing under test; the evaluator is the
thing being built.** Everything below follows from that inversion.

---

## 1. Run it

```bash
uv sync
uv run pytest                          # unit tests, fully offline
uv run inspect view --log-dir logs     # browse the two committed real runs
```

Open `evals.ipynb` for the full narrative (already executed, outputs committed).
Re-run it offline (**no API key needed**) with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace evals.ipynb
```

To regenerate the committed `.eval` logs from scratch (this *does* need a key):

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
uv run python -c "from inspect_ai import eval; from content_pipeline.eval_task import generation_eval, meta_eval; \
  eval(generation_eval(), model='anthropic/claude-sonnet-4-6', log_dir='logs'); \
  eval(meta_eval(include_judge=True), model='anthropic/claude-sonnet-4-6', log_dir='logs')"
```

**Where to look:** `evals.ipynb` is the guided tour; `uv run inspect view` opens the
raw logs. Two committed runs: `generation_eval` (real generations + all scorers) and
`meta_eval` (the meta-evaluation, including the gated judge metrics).

---

## 2. The solution at a glance

```
 PropertyData ──▶ prompt (strict JSON) ──▶ LLM ──▶ parse/validate ──▶ GeneratedContent
   (models.py)        (prompt.py)                  (Pydantic)          (4 sections)
                                                                            │
                                                ┌───────────────────────────┘
                                                ▼
                          ┌──────────────────  EVALUATION  ──────────────────┐
                          │ Tier 1  deterministic   (no model, runs in CI)    │
                          │ Tier 2  LLM judge       (faithfulness/coverage/…) │
                          │ Tier 3  meta-eval       (validates Tiers 1 & 2)   │
                          └────────────────────────────────────────────────────┘
```

| Concern | Where | Note |
|---|---|---|
| Data model | `models.py` | `PropertyData` in; `GeneratedContent` out |
| Generation | `prompt.py`, `generator.py` | one prompt path, shared by Inspect + the DI generator |
| Deterministic checks | `grounding.py` | pure functions, no model, unit-tested |
| LLM-judge scorers | `scorers.py` | faithfulness, coverage, brand voice + their metrics |
| Inspect tasks | `eval_task.py` | `generation_eval` and `meta_eval` |
| Fixtures / gold set | `fixtures.py`, `goldset.py` | labelled generations + human judgements |
| Offline reporting | `report.py` | reads committed logs; computes κ, confusion matrix |

### Generation
Each highlight is not free text: it is a `Claim` carrying a `source_field` naming the
input it derives from, and each `AmenityDescription` carries the exact input amenity
`code`. **That binding is the single most important design choice**: it converts
"is this grounded?" from an LLM guess into a mechanical lookup (Tier 1). The model is
prompted for strict JSON, which Pydantic validates (`parse_output`); a parse failure
is a loud, observable `ContentParseError`, never a silent retry.

---

## 3. How the evaluation pipeline works

The pipeline is three tiers, cheapest first: the cost hierarchy the production-eval
literature converges on (code assertions → LLM judge → human calibration).

### Tier 1: deterministic pre-filter (`grounding.py`, no model)
Runs in CI on every change, recomputes offline bit-for-bit.
- **`CitationValidityCheck`**: every highlight cites a field that actually holds data;
  every amenity code is one of the property's input codes.
- **`QualityCheck`**: length/count bounds, n-gram repetition, a generic-phrase
  blocklist. Catches accurate-but-lifeless boilerplate.

### Tier 2: LLM judge (`scorers.py`)
Each dimension is **one judge call** emitting **per-claim binary verdicts** into
`Score.metadata` (so the committed log reconstructs everything offline):
- **faithfulness** (*precision*). Decompose copy into atomic claims, judge each
  `entailed / contradicted / unsupported` against the input corpus (the RAGAS
  extract→verify→score pattern). Value = fraction entailed.
- **coverage** (*recall*). Of the property's marquee selling points (`salient_facts`),
  the fraction the copy actually surfaces. A perfectly faithful listing that buries the
  4.96/87 social proof fails here and **nowhere else**.
- **brand voice**: a `model_graded_qa`-style grader for tone/specificity.
- A judge response that won't parse is flagged `judge_error` and *excluded* from the
  score mean, so infra failure never masquerades as a content score of 0.

### Tier 3: meta-eval, validating the evaluator (`eval_task.py: meta_eval`)
This is the centerpiece. A **replay solver** feeds hand-labelled good/bad generations
(`fixtures.py`) back through the scorers with zero API, and custom `@metric`s ask "is
the scorer any good?":
- *Deterministic*: `detection_rate` (recall on planted defects) + `false_positive_rate`.
- *Judge, gated not just reported*: `judge_detection_rate` (per-claim recall: did it
  flag the **specific** planted claim?) and `judge_false_positive_rate` (over-flagging
  on genuinely-clean copy).
- *Judge vs human*: the confusion matrix in `report.py` reports **Cohen's κ** + TPR/TNR
  against the gold set (`goldset.py`).

Some negatives are deliberately **held-out** (not co-designed with the checks), so
detection rate measures *generalization*, not construction (see Tradeoffs).

---

## 4. Tradeoffs (the interesting part)

**Eval-first, generator second.** I spent the budget on validating the evaluator, not
on generation breadth. A suite that rubber-stamps everything is worthless; proving it
catches real failures *and* doesn't cry wolf is the harder, more valuable thing.

**Gating the judge, not just the cheap tier.** The easy version validates only the
deterministic checks (which can't really be wrong) and merely *reports* judge scores.
I made the judge a first-class meta-eval citizen with its own detection/false-positive
metrics, because the judge is where the real grounding risk lives, so it's the part
that most needs validating.

**Citation binding over free-text grounding.** Making each claim cite its
`source_field` costs prompt complexity and constrains the model, but it buys a
deterministic, model-free grounding check. Worth it.

**Strict-JSON prompting over provider tool-forcing.** Inspect's `ToolDef` derives its
schema from a callable signature, awkward for a nested object. JSON-prompting keeps one
provider-agnostic path and makes parse failure explicit (`ContentParseError`) rather
than something silently retried away.

**Held-out negatives over self-fitted ones.** It's tempting to write negatives that
exactly trip your checks (detection = 100%, looks great, proves nothing). I included a
bland sample whose phrasing is *not* on the denylist, and `QualityCheck` misses it, so
`meta_quality` detection honestly drops to **50%**. That number is the point: it
measures whether a 7-phrase blocklist generalizes (it doesn't), and shows the judge
backstopping it. An honest 50% beats a constructed 100%.

**Binary verdicts over Likert; κ over raw agreement.** Per-claim judgements are binary
with a written reason (not a 1–5 scale that clusters on the middle). Agreement is
reported as Cohen's κ, because raw agreement flatters under class imbalance.

**Faithfulness *and* coverage.** Faithfulness alone is necessary but not sufficient: a
listing can be 100% grounded and still bury every selling point. Coverage is the recall
complement; together they bracket the quality of the copy.

**Offline reproducibility: two mechanisms, by necessity.** Deterministic scorers
re-run live via the replay solver (truly reproducible, no key). The judge needs a key,
so its verdicts are read from the **committed `.eval` logs**. `inspect-ai` is pinned
exactly (`==0.3.240`) because the log format / viewer are version-sensitive, and
`fixtures/manifest.json` records model id + prompt hash so stale logs are detectable.

**Judge == generator model (a conceded tradeoff).** Both are `claude-sonnet-4-6` to fit
the budget; a distinct judge model would avoid self-enhancement bias. `faithfulness_scorer`
already takes a `grader_model`, so it's a one-line change.

---

## 5. Results (committed `claude-sonnet-4-6` run)

| Signal | Value | Reading |
|---|---|---|
| `citation_validity` detection / FP | **100% / 0%** | catches every designed citation defect |
| `quality` detection | **50%** | *by design*: held-out bland negative slips the denylist |
| `judge_detection_rate` | **100%** | flags the specific planted claim (incl. a prompt-injection attempt) |
| `judge_false_positive_rate` | **~13%** | over-flags grounded amenity flourishes (real calibration debt) |
| `judge_error_rate` | **0%** | no infra failures this run |
| judge-vs-human **Cohen's κ** | **≈ 0.53** | moderate; raw agreement (75%) would have flattered |
| faithfulness (real gens) | villa **0.85** / cottage **0.90** | |
| coverage (real gens) | **100%** | `BAD_INCOMPLETE_VILLA` is the low-coverage case only this catches |
| deterministic gates on real gens | **pass** | `tests/test_real_generations.py`: FP measured on real output, not just fixtures |


---

## 6. Engineering notes
- **Dependency injection / mocking**: `ContentGenerator` takes a `MessageClient`;
  `tests/test_generator.py` injects a fake. No test touches the network.
- **Inheritance**: `GroundingCheck` ABC + concrete checks sharing corpus access and
  result type, used where it earns its place.
- **Inspect-native**: `Dataset`/`Sample`, `@scorer`/`@metric` (custom
  `judge_detection_rate` / `judge_false_positive_rate` / `judge_error_rate`),
  `model_graded_qa`, per-claim `Score.metadata`, replay solver.
- **Tested invariants**: decomposition correctness (`test_decomposition.py`),
  gold-set↔log join integrity (`test_report.py`), prompt-hash drift (`test_manifest.py`).

---

## 7. Documented, not built (and why)
Scoped out deliberately under a 3–4h budget; I can speak to each:
- **Judge model ≠ generator** (self-enhancement bias): one-line change, left equal for budget.
- **Gating coverage & brand voice**: only **faithfulness** is gated (per-claim
  detection / false-positive metrics). Coverage and brand voice are *reported*
  (`judge_value_mean`), not gated: no metric yet asserts coverage fired on
  `BAD_INCOMPLETE_VILLA`. Mirroring the faithfulness pattern is the next step.
- **Injection hardening**: the judge corpus concatenates review text as source facts,
  so injection-resistance currently relies on the judge's intelligence (it does catch the
  planted "helipad"); structurally separating trusted fields from untrusted UGC is the fix.
- **CI regression gating** over `.eval` thresholds, **multilingual**, **image grounding**,
  and **refusal/truncation handling** are all unaddressed.
- **Statistical power**: n is a *demonstration* size; κ and every rate have very wide CIs,
  and a single annotator (who wrote the judge prompt) means no inter-annotator agreement.

### The production picture (what this is a scoped instance of)
Three layers used together, code assertions (CI) → LLM judge → human calibration, with
evals emerging from *observed failures* (error analysis), not a generic metric menu
(ROUGE/BERTScore are deliberately avoided). In production this closes into a flywheel:
the offline golden set gates every change, sampled live traces are judged reference-free
and clustered into new fixtures, and the judge prompt is re-tuned when κ against humans
drifts. This suite is the offline half, built to scale into that loop.

---

## 8. How I used AI
Built with Claude Code. I ran several adversarial critique rounds against my own design
before and during implementation, each reflected in the architecture: the replay
solver instead of `mockllm`, cutting a brittle numeric-regex check in favour of the
judge, making false-positive rate first-class, gating the judge (not just the
deterministic tier), and adding held-out negatives so detection measures generalization.
The corpus bug above was caught by *running* the suite, not by reading the code.
