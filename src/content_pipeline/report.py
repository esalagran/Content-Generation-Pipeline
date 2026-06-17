"""Read committed .eval logs and compute reportable numbers for the notebook.

Kept out of the notebook so the logic is importable and testable. All functions
are read-only over the committed logs (no API).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass

from inspect_ai.log import EvalLog, read_eval_log

from .goldset import GOLD


def load_logs(log_dir: str = "logs") -> list[EvalLog]:
    return [read_eval_log(f) for f in sorted(glob.glob(f"{log_dir}/*.eval"))]


def find_log(task: str, log_dir: str = "logs") -> EvalLog:
    """Return the committed log for a task, preferring one with judge verdicts."""
    candidates = [l for l in load_logs(log_dir) if l.eval.task == task]
    with_judge = [
        l for l in candidates
        if any("faithfulness" in (s.scores or {}) for s in (l.samples or []))
    ]
    candidates = with_judge or candidates
    if not candidates:
        raise FileNotFoundError(f"no committed log for task '{task}' in {log_dir}/")
    return candidates[-1]


def verdict_map(log: EvalLog) -> dict[tuple[str, str], str]:
    """(sample_id, claim_text) -> judge verdict, from the committed faithfulness scores."""
    out: dict[tuple[str, str], str] = {}
    for sample in log.samples or []:
        score = (sample.scores or {}).get("faithfulness")
        if not score or not score.metadata:
            continue
        if score.metadata.get("judge_error"):
            continue  # unparseable judge output — not a real verdict, drop the sample
        for v in score.metadata["verdicts"]:
            out[(str(sample.id), v["claim"])] = v["verdict"]
    return out


@dataclass
class GoldRow:
    sample_id: str
    claim: str
    human_supported: bool
    judge_verdict: str | None
    judge_supported: bool
    agree: bool


@dataclass
class GoldReport:
    rows: list[GoldRow]
    tp: int  # human-unsupported AND judge-flagged (caught hallucination)
    fp: int  # human-supported BUT judge-flagged (over-flag)
    tn: int
    fn: int

    @property
    def agreement(self) -> float:
        n = len(self.rows)
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:  # == TPR
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def tpr(self) -> float:
        return self.recall

    @property
    def tnr(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def cohen_kappa(self) -> float:
        """Chance-corrected agreement — the honest number under class imbalance,
        where raw agreement flatters. kappa = (po - pe) / (1 - pe)."""
        n = len(self.rows)
        if not n:
            return 0.0
        po = (self.tp + self.tn) / n
        pe = (
            (self.tp + self.fp) * (self.tp + self.fn)
            + (self.tn + self.fn) * (self.tn + self.fp)
        ) / (n * n)
        return (po - pe) / (1 - pe) if pe != 1 else 0.0


def gold_report(log_dir: str = "logs") -> GoldReport:
    """Compare committed judge verdicts to human gold labels. Positive class =
    'not grounded' (a hallucination the judge should flag)."""
    maps = {
        "generation": verdict_map(find_log("generation_eval", log_dir)),
        "meta": verdict_map(find_log("meta_eval", log_dir)),
    }
    rows, tp = [], 0
    fp = tn = fn = 0
    for g in GOLD:
        verdict = maps[g.source].get((g.sample_id, g.claim))
        judge_supported = verdict == "entailed"
        agree = judge_supported == g.supported
        rows.append(GoldRow(g.sample_id, g.claim, g.supported, verdict, judge_supported, agree))
        if not g.supported and not judge_supported:
            tp += 1
        elif g.supported and not judge_supported:
            fp += 1
        elif g.supported and judge_supported:
            tn += 1
        else:
            fn += 1
    return GoldReport(rows, tp, fp, tn, fn)
