"""Guards the gold-set <-> committed-log join: if a gold claim's text drifts from
the logged verdict text, the lookup silently returns None and the confusion matrix
is wrong. This catches that. Reads committed logs only — no API."""

from content_pipeline.report import gold_report


def test_every_gold_claim_matches_a_logged_verdict():
    report = gold_report()
    unmatched = [r.claim for r in report.rows if r.judge_verdict is None]
    assert not unmatched, f"gold claims with no matching logged verdict: {unmatched}"


def test_confusion_matrix_totals_consistent():
    report = gold_report()
    assert report.tp + report.fp + report.tn + report.fn == len(report.rows)
    assert report.recall == 1.0  # judge catches every hallucination in the gold set
