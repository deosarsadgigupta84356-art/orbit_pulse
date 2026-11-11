"""
Automated Event Grouping & Ordering Tests
========================================
Verifies:
1. Different event IDs never mix.
2. Messages are strictly ordered chronologically by message_time.
3. Duplicates are detected and audited.
"""

import pytest
import pandas as pd
from src.validation import validate_cdm_records
from src.event_builder import build_event_histories

def test_event_isolation_and_chronological_ordering():
    """Verify different event IDs never mix and timestamps are ordered."""
    raw_data = pd.DataFrame([
        {
            "event_id": "EVT_A",
            "message_id": "A2",
            "message_time": "2026-06-02T10:00:00Z",
            "tca": "2026-06-05T12:00:00Z",
            "collision_probability": 1e-5
        },
        {
            "event_id": "EVT_B",
            "message_id": "B1",
            "message_time": "2026-06-01T08:00:00Z",
            "tca": "2026-06-06T12:00:00Z",
            "collision_probability": 2e-5
        },
        {
            "event_id": "EVT_A",
            "message_id": "A1",
            "message_time": "2026-06-01T12:00:00Z", # Earlier than A2
            "tca": "2026-06-05T12:00:00Z",
            "collision_probability": 5e-6
        }
    ])

    valid_df, _, audit = validate_cdm_records(raw_data)
    events = build_event_histories(valid_df)

    assert len(events) == 2
    assert "EVT_A" in events
    assert "EVT_B" in events

    evt_a = events["EVT_A"]
    # Check chronological ordering: A1 (June 1) must be before A2 (June 2)
    assert evt_a.messages.iloc[0]["message_id"] == "A1"
    assert evt_a.messages.iloc[1]["message_id"] == "A2"
    assert evt_a.messages.iloc[0]["message_time"] < evt_a.messages.iloc[1]["message_time"]

def test_duplicate_handling():
    """Verify exact duplicates are identified and filtered."""
    raw_data = pd.DataFrame([
        {
            "event_id": "EVT_DUP",
            "message_id": "M1",
            "message_time": "2026-06-01T10:00:00Z",
            "tca": "2026-06-05T12:00:00Z",
            "collision_probability": 1e-5
        },
        {
            "event_id": "EVT_DUP",
            "message_id": "M1", # Duplicate
            "message_time": "2026-06-01T10:00:00Z",
            "tca": "2026-06-05T12:00:00Z",
            "collision_probability": 1e-5
        }
    ])

    valid_df, _, audit = validate_cdm_records(raw_data)
    assert audit["duplicate_records_count"] == 1
    assert len(valid_df) == 1
