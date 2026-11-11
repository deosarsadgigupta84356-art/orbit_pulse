"""
ORBIT PULSE - Evidence Intelligence Layer
=========================================
Computes three distinct, uncollapsed evidence dimensions:
1. Data Freshness (age of latest admitted message)
2. Evidence Completeness (coverage of expected CDM schema fields)
3. Forecast Stability (model sensitivity under eligible history perturbations)

SAFETY RULE: Never collapse these into an arbitrary 'confidence score'.
Always state that forecast sensitivity is model sensitivity to input history,
NOT physical orbit covariance or real-world uncertainty bounds.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.snapshot_48h import EventSnapshot48h
from src.feature_engineering import extract_snapshot_features, safe_log10
from src.data_loader import get_thresholds_config

def evaluate_data_freshness(snapshot: EventSnapshot48h) -> Dict[str, Any]:
    """
    Evaluates freshness of the latest eligible observation relative to cutoff.
    """
    if not snapshot.eligible or snapshot.latest_eligible_message_time is None:
        return {
            "age_hours": None,
            "category": "NO_DATA",
            "description": "No eligible observation available before TCA-48h."
        }

    age_seconds = (snapshot.cutoff - snapshot.latest_eligible_message_time).total_seconds()
    age_hours = max(0.0, float(age_seconds / 3600.0))

    freshness_cfg = get_thresholds_config()["evidence_thresholds"]["freshness"]
    fresh_limit = float(freshness_cfg["fresh_hours"])
    ageing_limit = float(freshness_cfg["ageing_hours"])
    stale_limit = float(freshness_cfg.get("stale_hours", ageing_limit))

    if age_hours <= fresh_limit:
        category = "FRESH"
        description = f"Observation is {age_hours:.1f}h old (≤{fresh_limit:g}h). Fresh tracking data."
    elif age_hours <= ageing_limit:
        category = "AGEING"
        description = f"Observation is {age_hours:.1f}h old ({fresh_limit:g}-{ageing_limit:g}h). Moderately aged state solution."
    elif age_hours >= stale_limit:
        category = "STALE"
        description = f"Observation is {age_hours:.1f}h old (>{stale_limit:g}h). Significant latency in orbit update."
    else:
        category = "AGEING"
        description = f"Observation is {age_hours:.1f}h old. Tracking evidence is ageing."

    return {
        "age_hours": round(age_hours, 2),
        "category": category,
        "description": description
    }

def evaluate_evidence_completeness(snapshot: EventSnapshot48h) -> Dict[str, Any]:
    """
    Evaluates schema completeness of the eligible messages.
    Separates required fields from optional geometric/uncertainty fields.
    """
    if not snapshot.eligible or snapshot.eligible_message_count == 0:
        return {
            "completeness_ratio": 0.0,
            "category": "INCOMPLETE",
            "required_present": False,
            "optional_checklist": {},
            "description": "No messages available to assess completeness."
        }

    msgs = snapshot.eligible_messages

    required_cols = ["message_id", "message_time", "tca", "collision_probability"]
    optional_cols = ["miss_distance", "relative_speed", "c_r_r", "c_t_t", "c_n_n", "sat_1_id", "sat_2_id"]

    req_present = all(col in msgs.columns and msgs[col].notnull().all() for col in required_cols)

    optional_checklist = {}
    present_optional_count = 0
    for col in optional_cols:
        present = bool(col in msgs.columns and msgs[col].notnull().any())
        optional_checklist[col] = present
        if present:
            present_optional_count += 1

    total_fields = len(required_cols) + len(optional_cols)
    total_present = (len(required_cols) if req_present else 0) + present_optional_count
    ratio = float(total_present / total_fields)

    completeness_cfg = get_thresholds_config()["evidence_thresholds"]["completeness"]
    high_ratio = float(completeness_cfg["high_ratio"])
    acceptable_ratio = float(completeness_cfg["acceptable_ratio"])

    if ratio >= high_ratio:
        category = "HIGH"
        description = "High completeness. Comprehensive orbital uncertainty & state fields present."
    elif ratio >= acceptable_ratio:
        category = "MODERATE"
        description = "Moderate completeness. Core fields present, minor optional covariance fields absent."
    else:
        category = "LIMITED"
        description = "Limited completeness. Critical orbital descriptor fields missing."

    return {
        "completeness_ratio": round(ratio, 3),
        "category": category,
        "required_present": req_present,
        "optional_checklist": optional_checklist,
        "description": description
    }

def evaluate_forecast_stability(
    snapshot: EventSnapshot48h,
    model,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Runs perturbation experiments on eligible messages to evaluate model forecast stability.
    Leave-one-message-out experiment across eligible history.
    """
    disclaimer = "Model sensitivity to eligible input history — NOT physical orbit covariance or real-world uncertainty bound."
    stability_cfg = get_thresholds_config()["evidence_thresholds"]["stability"]
    min_samples = int(stability_cfg["min_perturbation_samples"])
    sensitive_limit = float(stability_cfg["max_log_shift_sensitive"])

    if not snapshot.eligible or snapshot.eligible_message_count == 0:
        return {
            "stability_status": "NOT_ESTIMABLE",
            "base_log_pc": None,
            "min_log_pc": None,
            "max_log_pc": None,
            "log_spread": None,
            "perturbed_forecasts": [],
            "disclaimer": disclaimer,
            "description": "Cannot evaluate stability: No eligible messages."
        }

    if snapshot.eligible_message_count < 2:
        # Base prediction only
        feat_dict = extract_snapshot_features(snapshot)
        x_base = pd.DataFrame([[feat_dict[k] for k in feature_names]], columns=feature_names)
        base_log_pc = float(model.predict(x_base)[0])
        return {
            "stability_status": "NOT_ESTIMABLE",
            "base_log_pc": round(base_log_pc, 3),
            "min_log_pc": None,
            "max_log_pc": None,
            "log_spread": None,
            "perturbed_forecasts": [],
            "disclaimer": disclaimer,
            "description": "Single eligible message available; leave-one-out perturbation not estimable."
        }

    # Complete eligible base prediction
    feat_base = extract_snapshot_features(snapshot)
    x_base = pd.DataFrame([[feat_base[k] for k in feature_names]], columns=feature_names)
    base_log_pc = float(model.predict(x_base)[0])

    # Leave-one-out perturbation
    perturbed_preds = []
    msgs = snapshot.eligible_messages

    for drop_idx in range(len(msgs)):
        perturbed_msgs = msgs.drop(index=drop_idx).reset_index(drop=True)
        # Create temporary snapshot
        temp_snap = EventSnapshot48h(
            event_id=snapshot.event_id,
            tca=snapshot.tca,
            cutoff_utc=snapshot.cutoff,
            eligible=True,
            exclusion_reason=None,
            eligible_messages=perturbed_msgs,
            excluded_messages=snapshot.excluded_messages,
            target_final_pc=snapshot.target_final_pc
        )
        temp_feat = extract_snapshot_features(temp_snap)
        if temp_feat is not None:
            x_pert = pd.DataFrame([[temp_feat[k] for k in feature_names]], columns=feature_names)
            p_pred = float(model.predict(x_pert)[0])
            perturbed_preds.append(p_pred)

    if not perturbed_preds:
        perturbed_preds = [base_log_pc]

    min_log_pc = float(np.min(perturbed_preds))
    max_log_pc = float(np.max(perturbed_preds))
    spread = abs(max_log_pc - min_log_pc)

    # Threshold: > 1.0 log units (1 order of magnitude in Pc) = SENSITIVE
    if spread > sensitive_limit:
        status = "SENSITIVE"
        description = f"Forecast shows sensitivity to historical sequence (spread = {spread:.2f} log10 units)."
    else:
        status = "STABLE"
        description = f"Forecast is stable across history perturbations (spread = {spread:.2f} log10 units)."

    return {
        "stability_status": status,
        "base_log_pc": round(base_log_pc, 3),
        "min_log_pc": round(min_log_pc, 3),
        "max_log_pc": round(max_log_pc, 3),
        "log_spread": round(spread, 3),
        "perturbed_forecasts": [round(p, 3) for p in perturbed_preds],
        "disclaimer": disclaimer,
        "description": description
    }

def get_complete_evidence_intelligence(
    snapshot: EventSnapshot48h,
    model,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Consolidates all three independent evidence layers."""
    return {
        "freshness": evaluate_data_freshness(snapshot),
        "completeness": evaluate_evidence_completeness(snapshot),
        "stability": evaluate_forecast_stability(snapshot, model, feature_names)
    }
