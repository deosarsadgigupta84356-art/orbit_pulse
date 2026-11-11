"""
Automated Decision Support Policy & Safety Wording Tests
========================================================
Verifies:
1. High predicted Pc (>= 1e-4) escalates to MANOEUVRE REVIEW.
2. Low predicted Pc with stale/unstable evidence escalates to UNCERTAIN (never all-clear).
3. Strictly forbids dangerous terms ("execute manoeuvre", "collision certain", "safe to ignore").
"""

import pytest
from src.decision_engine import evaluate_decision_policy

def test_high_risk_escalation():
    """Verify Pc >= 1e-4 escalates to HIGH / MANOEUVRE REVIEW."""
    pred_pc = 2.5e-4
    evidence = {
        "freshness": {"category": "FRESH", "age_hours": 4.0},
        "completeness": {"category": "HIGH", "completeness_ratio": 0.95},
        "stability": {"stability_status": "STABLE"}
    }

    decision = evaluate_decision_policy(pred_pc, evidence)
    assert decision["risk_band"] == "HIGH"
    assert "HIGH" in decision["advisory_label"]
    assert "MANOEUVRE REVIEW" in decision["advisory_label"]
    assert "EVIDENCE STRONGER" in decision["evidence_qualifier"]

def test_uncertain_on_stale_or_limited_evidence():
    """Verify low Pc with stale/incomplete evidence leads to UNCERTAIN review."""
    pred_pc = 1.0e-7  # Very low Pc
    evidence = {
        "freshness": {"category": "STALE", "age_hours": 36.0}, # Very stale
        "completeness": {"category": "LIMITED", "completeness_ratio": 0.50},
        "stability": {"stability_status": "SENSITIVE"}
    }

    decision = evaluate_decision_policy(pred_pc, evidence)
    # Must NOT be LOW / all-clear
    assert decision["risk_band"] == "UNCERTAIN"
    assert "UNCERTAIN" in decision["advisory_label"]

def test_strict_guardrail_forbidden_wording():
    """Verify forbidden autonomous wording never appears in decision output."""
    forbidden_terms = [
        "execute manoeuvre",
        "execute maneuver",
        "collision certain",
        "safe to ignore",
        "autonomous burn"
    ]

    test_cases = [
        (1e-2, {"freshness": {"category": "FRESH"}, "completeness": {"category": "HIGH"}, "stability": {"stability_status": "STABLE"}}),
        (1e-7, {"freshness": {"category": "STALE"}, "completeness": {"category": "LIMITED"}, "stability": {"stability_status": "SENSITIVE"}}),
        (1e-8, {"freshness": {"category": "FRESH"}, "completeness": {"category": "HIGH"}, "stability": {"stability_status": "STABLE"}})
    ]

    for pc, ev in test_cases:
        res = evaluate_decision_policy(pc, ev)
        full_text = (
            f"{res['advisory_label']} {res['evidence_qualifier']} {res['rationale']} "
            f"{' '.join(res['actionable_next_steps'])} {res['disclaimer']}"
        ).lower()

        for term in forbidden_terms:
            assert term not in full_text, f"Forbidden term '{term}' found in decision output!"



def test_ageing_evidence_does_not_produce_routine_low_status():
    """AGEING is not the same as FRESH and must not become an all-clear."""
    evidence = {
        "freshness": {"category": "AGEING", "age_hours": 18.0},
        "completeness": {"category": "HIGH", "completeness_ratio": 0.95},
        "stability": {"stability_status": "STABLE"}
    }

    decision = evaluate_decision_policy(1.0e-7, evidence)
    assert decision["risk_band"] == "UNCERTAIN"
    assert "ageing observation" in decision["rationale"]


def test_unestimable_stability_does_not_produce_routine_low_status():
    """A one-message history cannot be described as a stable forecast."""
    evidence = {
        "freshness": {"category": "FRESH", "age_hours": 4.0},
        "completeness": {"category": "HIGH", "completeness_ratio": 0.95},
        "stability": {"stability_status": "NOT_ESTIMABLE"}
    }

    decision = evaluate_decision_policy(1.0e-7, evidence)
    assert decision["risk_band"] == "UNCERTAIN"
    assert "could not be estimated" in decision["rationale"]
