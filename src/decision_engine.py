"""
ORBIT PULSE - Advisory Decision Support Policy
=============================================
Translates risk predictions and multi-dimensional evidence indicators
into auditable, human-in-the-loop operational review recommendations.

SAFETY RULES:
- Never say "execute manoeuvre", "collision certain", or "safe to ignore".
- Always use transparent advisory wording:
    LOW — ROUTINE MONITORING
    UNCERTAIN — FURTHER ASSESSMENT
    HIGH — ESCALATE FOR MANOEUVRE REVIEW
- A low forecast with stale/unstable evidence must NEVER look like an all-clear.
"""

from typing import Dict, Any, Optional
from src.data_loader import get_thresholds_config

OPERATIONAL_HIGH_PC = 1e-4  # Standard conjunction review threshold: 1 in 10,000

def evaluate_decision_policy(
    predicted_pc: float,
    evidence_intel: Dict[str, Any],
    threshold_override: Optional[float] = None
) -> Dict[str, Any]:
    """
    Applies the transparent advisory decision policy to predicted Pc and evidence.

    Returns:
        dict containing:
            risk_band: "LOW", "UNCERTAIN", or "HIGH"
            advisory_label: Official UI text
            evidence_qualifier: Contextual qualifier
            rationale: Step-by-step audit rationale
            actionable_next_steps: Recommended operational steps
            competition_binary_flag: Advisory flag if required for competition scoring
    """
    config = get_thresholds_config()
    high_threshold = (
        threshold_override
        if threshold_override is not None
        else config["risk_thresholds"]["high_pc_threshold"]
    )

    freshness = evidence_intel.get("freshness", {})
    completeness = evidence_intel.get("completeness", {})
    stability = evidence_intel.get("stability", {})

    is_fresh = freshness.get("category") == "FRESH"
    # Any category other than FRESH is not sufficient for a routine all-clear.
    is_stale = not is_fresh
    is_incomplete = completeness.get("category") not in ["HIGH", "MODERATE"]
    stability_status = stability.get("stability_status")
    is_sensitive = stability_status != "STABLE"

    predicted_high = predicted_pc >= high_threshold

    # 1. High Risk Category (Forecast >= Threshold)
    if predicted_high:
        risk_band = "HIGH"
        advisory_label = config["labels"]["high"]

        if is_stale or is_incomplete or is_sensitive:
            evidence_qualifier = "HIGH RISK — REVIEW URGENT; EVIDENCE LIMITED / UNVERIFIED"
            rationale = (
                f"Predicted final Pc ({predicted_pc:.2e}) meets or exceeds operational threshold ({high_threshold:.2e}). "
                "At least one evidence dimension is not fully verified "
                f"(freshness={freshness.get('category', 'UNKNOWN')}, "
                f"completeness={completeness.get('category', 'UNKNOWN')}, "
                f"stability={stability.get('stability_status', 'UNKNOWN')}). "
                "Urgent review and additional tracking evidence are recommended."
            )
            actionable_next_steps = [
                "Escalate to Conjunction Assessment Officer immediately.",
                "Request urgent high-priority radar/optical tracking pass.",
                "Prepare candidate manoeuvre profile options in scenario sandbox."
            ]
        else:
            evidence_qualifier = "HIGH RISK — REVIEW RECOMMENDED; EVIDENCE STRONGER"
            rationale = (
                f"Predicted final Pc ({predicted_pc:.2e}) meets or exceeds operational threshold ({high_threshold:.2e}) "
                f"with fresh, high-completeness tracking history and stable model forecast."
            )
            actionable_next_steps = [
                "Escalate to Spacecraft Operations Directorate.",
                "Commence secondary object screening and ephemeris refinement.",
                "Evaluate delta-V burn windows and propellant budget."
            ]
        competition_binary_flag = "MANOEUVRE_REVIEW_REQUIRED"

    # 2. Low Numerical Risk BUT Weak/Uncertain Evidence
    elif is_stale or is_incomplete or is_sensitive:
        risk_band = "UNCERTAIN"
        advisory_label = config["labels"]["uncertain"]
        evidence_qualifier = "UNCERTAIN — LOW RISK FORECAST BUT WEAK EVIDENCE"

        reasons = []
        if is_stale:
            freshness_category = freshness.get("category", "UNKNOWN")
            if freshness_category == "AGEING":
                reasons.append(f"ageing observation ({freshness.get('age_hours')}h old)")
            elif freshness_category == "STALE":
                reasons.append(f"stale observation ({freshness.get('age_hours')}h old)")
            else:
                reasons.append("no verified fresh observation")
        if is_incomplete:
            reasons.append("incomplete covariance descriptors")
        if stability_status == "SENSITIVE":
            reasons.append("high model sensitivity to historical updates")
        elif stability_status == "NOT_ESTIMABLE":
            reasons.append("forecast stability could not be estimated from the available history")
        elif stability_status != "STABLE":
            reasons.append("forecast stability status is not verified")

        rationale = (
            f"Predicted final Pc ({predicted_pc:.2e}) is nominally below the review threshold ({high_threshold:.2e}), "
            f"but evidence reliability is compromised due to: {', '.join(reasons)}. "
            "This event cannot be safely signed off as routine."
        )
        actionable_next_steps = [
            "Maintain on active watch list.",
            "Verify tracking update availability before next orbit pass.",
            "Re-screen event when next CDM arrives."
        ]
        competition_binary_flag = "FURTHER_MONITORING"

    # 3. Low Risk and Adequate Evidence
    else:
        risk_band = "LOW"
        advisory_label = config["labels"]["low"]
        evidence_qualifier = "LOW RISK — ROUTINE MONITORING; EVIDENCE ADEQUATE"
        rationale = (
            f"Predicted final Pc ({predicted_pc:.2e}) is well below operational review threshold ({high_threshold:.2e}). "
            "Tracking data is fresh, complete, and model forecast is stable under perturbations."
        )
        actionable_next_steps = [
            "Log conjunction summary to automated flight register.",
            "Continue standard uncrewed / routine monitoring.",
            "No manoeuvre planning required at this epoch."
        ]
        competition_binary_flag = "NO_ACTION_REQUIRED"

    return {
        "risk_band": risk_band,
        "advisory_label": advisory_label,
        "evidence_qualifier": evidence_qualifier,
        "rationale": rationale,
        "actionable_next_steps": actionable_next_steps,
        "competition_binary_flag": competition_binary_flag,
        "operational_threshold_applied": high_threshold,
        "disclaimer": config.get("disclaimer", "Advisory prototype only.")
    }
