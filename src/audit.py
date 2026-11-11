"""
ORBIT PULSE - Provenance & Audit Logging
========================================
Maintains an append-only audit trail of every inference, snapshot cutoff,
and decision recommendation for complete operational traceability.
"""

import os
import json
from pathlib import Path
import hashlib
from typing import Dict, Any, List
import pandas as pd
from src.snapshot_48h import EventSnapshot48h

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_LOG_PATH = str(PROJECT_ROOT / "reports" / "audit_log.jsonl")

def hash_snapshot_inputs(snapshot: EventSnapshot48h) -> str:
    """Computes a deterministic SHA-256 hash of every eligible input field."""
    if not snapshot.eligible or snapshot.eligible_message_count == 0:
        return "INELIGIBLE_EMPTY_HASH"

    ordered = snapshot.eligible_messages.sort_values("message_time").reset_index(drop=True)
    canonical = ordered.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        float_format="%.12g"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def record_audit_entry(prediction_output: Dict[str, Any], snapshot: EventSnapshot48h, log_path: str = AUDIT_LOG_PATH):
    """Appends an auditable record of the prediction and input lineage."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    input_hash = hash_snapshot_inputs(snapshot)

    entry = {
        "timestamp_utc": prediction_output.get("prediction_timestamp_utc"),
        "event_id": prediction_output.get("event_id"),
        "eligible": prediction_output.get("eligible"),
        "exclusion_reason": prediction_output.get("exclusion_reason"),
        "cutoff_utc": prediction_output.get("cutoff_utc"),
        "tca_utc": prediction_output.get("tca_utc"),
        "input_hash": input_hash,
        "eligible_message_count": prediction_output.get("eligible_message_count"),
        "predicted_value": prediction_output.get("predicted_value"),
        "predicted_log_value": prediction_output.get("predicted_log_value"),
        "risk_band": prediction_output.get("risk_band"),
        "recommendation": prediction_output.get("recommendation"),
        "advisory_label": prediction_output.get("advisory_label"),
        "data_freshness_hours": prediction_output.get("data_freshness_hours"),
        "evidence_completeness": prediction_output.get("evidence_completeness"),
        "forecast_stability": prediction_output.get("forecast_stability"),
        "dataset_version": prediction_output.get("dataset_version"),
        "model_version": prediction_output.get("model_version")
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry

def load_audit_trail(log_path: str = AUDIT_LOG_PATH) -> pd.DataFrame:
    """Reads the audit trail log into a pandas DataFrame."""
    if not os.path.exists(log_path):
        return pd.DataFrame()

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass
    return pd.DataFrame(records)
