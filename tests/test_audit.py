import json
import pandas as pd

from src.audit import hash_snapshot_inputs, record_audit_entry
from src.event_builder import ConjunctionEventHistory
from src.snapshot_48h import build_48h_snapshot


def make_snapshot():
    tca = pd.Timestamp("2026-07-15T12:00:00Z")
    messages = pd.DataFrame([
        {
            "message_id": "M1",
            "event_id": "EVT-AUDIT",
            "message_time": tca - pd.Timedelta(hours=72),
            "tca": tca,
            "collision_probability": 1e-6,
            "miss_distance": 5000.0,
            "relative_speed": 10000.0,
        },
        {
            "message_id": "M2",
            "event_id": "EVT-AUDIT",
            "message_time": tca - pd.Timedelta(hours=12),
            "tca": tca,
            "collision_probability": 2e-6,
            "miss_distance": 4000.0,
            "relative_speed": 10000.0,
        },
    ])
    return build_48h_snapshot(ConjunctionEventHistory("EVT-AUDIT", messages))


def test_snapshot_hash_covers_optional_input_fields():
    snapshot = make_snapshot()
    original_hash = hash_snapshot_inputs(snapshot)

    snapshot.eligible_messages.loc[0, "miss_distance"] = 1234.0
    assert hash_snapshot_inputs(snapshot) != original_hash


def test_audit_entry_is_written_with_snapshot_lineage(tmp_path):
    snapshot = make_snapshot()
    output_path = tmp_path / "audit.jsonl"
    prediction = {
        "prediction_timestamp_utc": "2026-07-13T12:00:00Z",
        "event_id": snapshot.event_id,
        "eligible": snapshot.eligible,
        "cutoff_utc": snapshot.cutoff_utc,
        "tca_utc": snapshot.tca_utc,
        "eligible_message_count": snapshot.eligible_message_count,
        "predicted_value": 1e-6,
        "risk_band": "LOW",
        "dataset_version": "TEST",
        "model_version": "TEST_MODEL",
    }

    record_audit_entry(prediction, snapshot, str(output_path))
    entry = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert entry["event_id"] == "EVT-AUDIT"
    assert entry["input_hash"] == hash_snapshot_inputs(snapshot)
    assert entry["eligible_message_count"] == 1
