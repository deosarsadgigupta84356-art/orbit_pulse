"""
Automated Timestamp & Ingestion Tests
====================================
Verifies:
1. ISO-8601 UTC timestamp parsing.
2. Inconsistent TCA detection and quarantine.
3. Out-of-bounds collision probability handling.
"""

import pytest
import pandas as pd
from src.validation import validate_cdm_records

def test_timestamp_parsing_and_utc_normalization():
    """Verify timestamps are parsed into timezone-aware UTC."""
    df_raw = pd.DataFrame([
        {
            "event_id": "EVT_TS_1",
            "message_id": "M1",
            "message_time": "2026-08-01T12:00:00Z",
            "tca": "2026-08-04T12:00:00Z",
            "collision_probability": 1e-5
        }
    ])

    valid_df, _, audit = validate_cdm_records(df_raw)
    assert audit["timestamp_parse_errors"] == 0
    assert valid_df["message_time"].iloc[0].tz is not None

def test_inconsistent_tca_quarantine():
    """Verify events with drifting/inconsistent TCAs (> 1 hour) are quarantined."""
    df_raw = pd.DataFrame([
        {
            "event_id": "EVT_BAD_TCA",
            "message_id": "M1",
            "message_time": "2026-08-01T12:00:00Z",
            "tca": "2026-08-04T12:00:00Z",
            "collision_probability": 1e-5
        },
        {
            "event_id": "EVT_BAD_TCA",
            "message_id": "M2",
            "message_time": "2026-08-02T12:00:00Z",
            "tca": "2026-08-10T12:00:00Z", # 6 days different TCA for same event!
            "collision_probability": 1e-5
        }
    ])

    valid_df, quarantine_df, audit = validate_cdm_records(df_raw)
    assert audit["tca_inconsistencies"] > 0
    assert len(quarantine_df) == 2
    assert len(valid_df) == 0

def test_invalid_probability_quarantine():
    """Verify Pc < 0 or Pc > 1 is quarantined."""
    df_raw = pd.DataFrame([
        {
            "event_id": "EVT_BAD_PC",
            "message_id": "M1",
            "message_time": "2026-08-01T12:00:00Z",
            "tca": "2026-08-04T12:00:00Z",
            "collision_probability": 1.5 # Invalid Pc > 1
        }
    ])

    valid_df, quarantine_df, audit = validate_cdm_records(df_raw)
    assert audit["pc_range_violations"] == 1
    assert len(quarantine_df) == 1
    assert len(valid_df) == 0
