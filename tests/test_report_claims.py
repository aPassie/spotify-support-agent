"""Every number quoted in REPORT.md and README.md must match the generated metrics.

A report that drifts from its own metrics is worse than no report, and this is the kind of
drift that happens silently after a re-run. Run `make reproduce` first, then this test.
"""
import json
from pathlib import Path

import pytest

METRICS = Path("outputs/metrics/metrics.json")
PAIRWISE = Path("outputs/metrics/pairwise.json")

pytestmark = pytest.mark.skipif(not METRICS.exists(), reason="run `make reproduce` first")


@pytest.fixture(scope="module")
def M():
    return json.load(open(METRICS))


# (label, quoted value, path into metrics.json, tolerance)
CLAIMS = [
    ("agent auto-handle rate", 0.419, ("systems", "agent", "random", "escalation", "coverage_auto_rate"), 0.006),
    ("agent auto precision", 0.970, ("systems", "agent", "random", "escalation", "auto_precision"), 0.006),
    ("agent unsafe replies", 2, ("systems", "agent", "random", "escalation", "n_unsafe_auto"), 0),
    ("agent intent accuracy", 0.681, ("systems", "agent", "random", "intent", "accuracy"), 0.006),
    ("simple auto-handle rate", 0.625, ("systems", "simple", "random", "escalation", "coverage_auto_rate"), 0.006),
    ("simple auto precision", 0.810, ("systems", "simple", "random", "escalation", "auto_precision"), 0.006),
    ("simple unsafe replies", 19, ("systems", "simple", "random", "escalation", "n_unsafe_auto"), 0),
    ("simple intent accuracy", 0.494, ("systems", "simple", "random", "intent", "accuracy"), 0.006),
    ("no_policy unsafe replies", 59, ("systems", "abl:no_policy", "random", "escalation", "n_unsafe_auto"), 0),
    ("no_policy judge mean", 3.67, ("systems", "abl:no_policy", "all", "judge", "mean_overall"), 0.006),
    ("rules_clf auto-handle rate", 0.263, ("systems", "abl:rules_clf", "random", "escalation", "coverage_auto_rate"), 0.006),
    ("no_exemplars judge mean", 3.24, ("systems", "abl:no_exemplars", "all", "judge", "mean_overall"), 0.006),
    ("embed judge mean", 3.46, ("systems", "abl:embed", "all", "judge", "mean_overall"), 0.006),
    ("agent judge mean", 3.52, ("systems", "agent", "all", "judge", "mean_overall"), 0.006),
    ("simple judge mean", 3.61, ("systems", "simple", "all", "judge", "mean_overall"), 0.006),
    ("simple wrong-name rate", 0.39, ("systems", "simple", "all", "reply_auto", "pct_wrong_name"), 0.006),
    ("agent wrong-name rate", 0.00, ("systems", "agent", "all", "reply_auto", "pct_wrong_name"), 0.0),
    ("agent unknown-link rate", 0.00, ("systems", "agent", "all", "reply_auto", "pct_unknown_url"), 0.0),
    ("judge-human Spearman", 0.49, ("judge_vs_human", "all", "spearman"), 0.006),
    ("judge-human kappa", 0.32, ("judge_vs_human", "all", "kappa_quadratic"), 0.006),
    ("judge generosity", 1.02, ("judge_vs_human", "all", "judge_minus_human_mean"), 0.006),
]


@pytest.mark.parametrize("label,quoted,path,tol", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_report_claim_matches_metrics(M, label, quoted, path, tol):
    node = M
    for key in path:
        node = node[key]
    assert abs(node - quoted) <= tol, f"REPORT.md quotes {label}={quoted}, metrics say {node}"


def test_paired_significance_is_reported_correctly(M):
    sig = M["significance_random"]["agent_vs_simple"]
    assert sig["unsafe_auto_mcnemar"]["p_value"] < 1e-4
    assert sig["intent_mcnemar"]["p_value"] < 1e-3
    # the direction must be the one the report claims
    assert sig["unsafe_auto_mcnemar"]["only_b_wrong"] > sig["unsafe_auto_mcnemar"]["only_a_wrong"]


@pytest.mark.skipif(not PAIRWISE.exists(), reason="pairwise sheet not scored")
def test_pairwise_is_reported_as_underpowered():
    p = json.load(open(PAIRWISE))
    assert abs(p["win_rate_b_of_decided"] - 0.619) <= 0.006
    assert p["p_value"] > 0.05, "report calls this non-significant"
    assert p["power"]["n_decided_for_80pct_power"] > p["power"]["eligible_items"], (
        "report claims the golden set can never settle this"
    )
