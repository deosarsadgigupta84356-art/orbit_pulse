"""
Automated Cutoff Boundary Tests
==============================
Verifies:
1. Messages exactly at TCA-48h boundary are included.
2. Messages after TCA-48h boundary are strictly excluded from snapshot.
3. Events with no messages before TCA-48h are marked ineligible without substituting later messages.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.event_builder import ConjunctionEventHistory
from src.snapshot_48h import build_48h_snapshot

def test_cutoff_boundary_exact_inclusion():
    """Verify message exactly at TCA-48h is eligible."""
    tca = pd.Timestamp("2026-05-10T12:00:00Z")
    cutoff = tca - timedelta(hours=48)

    msgs = pd.DataFrame([
        {
            "message_id": "M1",
            "message_time": cutoff - timedelta(hours=10),
            "tca": tca,
            "collision_probability": 1e-5
        },
        {
            "message_id": "M2_EXACT_BOUNDARY",
            "message_time": cutoff,  # Exactly at boundary
            "tca": tca,
            "collision_probability": 2e-5
        },
        {
            "message_id": "M3_POST_CUTOFF",
            "message_time": cutoff + timedelta(minutes=1),  # 1 min post-cutoff
            "tca": tca,
            "collision_probability": 5e-5
        }
    ])

    event = ConjunctionEventHistory(event_id="EVT_BOUNDARY_TEST", messages_df=msgs)
    snapshot = build_48h_snapshot(event)

    assert snapshot.eligible is True
    assert snapshot.eligible_message_count == 2
    assert "M2_EXACT_BOUNDARY" in snapshot.eligible_messages["message_id"].values
    assert "M3_POST_CUTOFF" not in snapshot.eligible_messages["message_id"].values
    assert "M3_POST_CUTOFF" in snapshot.excluded_messages["message_id"].values

def test_ineligible_event_no_pre_cutoff_observation():
    """Verify events with no observation before TCA-48h are marked ineligible."""
    tca = pd.Timestamp("2026-05-10T12:00:00Z")
    cutoff = tca - timedelta(hours=48)

    # Only messages post-cutoff
    msgs = pd.DataFrame([
        {
            "message_id": "M_LATE_1",
            "message_time": cutoff + timedelta(hours=12),  # TCA - 36h
            "tca": tca,
            "collision_probability": 3e-4
        },
        {
            "message_id": "M_LATE_2",
            "message_time": cutoff + timedelta(hours=36),  # TCA - 12h
            "tca": tca,
            "collision_probability": 8e-4
        }
    ])

    event = ConjunctionEventHistory(event_id="EVT_INELIGIBLE_TEST", messages_df=msgs)
    snapshot = build_48h_snapshot(event)

    assert snapshot.eligible is False
    assert snapshot.exclusion_reason == "NO_OBSERVATION_BEFORE_TCA_MINUS_48H"
    assert snapshot.eligible_message_count == 0
    assert snapshot.excluded_message_count == 2
    # Never substitute later message
    assert snapshot.last_eligible_pc is None
