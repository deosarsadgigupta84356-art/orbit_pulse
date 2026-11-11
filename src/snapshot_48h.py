"""
ORBIT PULSE - 48-Hour Leakage-Safe Snapshot Builder
===================================================
Enforces strict TCA-48h cutoff logic.
Constructs an auditable, leakage-safe snapshot for each event,
isolating post-cutoff observations and keeping final targets as evaluation labels only.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import timedelta
from src.event_builder import ConjunctionEventHistory

CUTOFF_HOURS_BEFORE_TCA = 48.0

class EventSnapshot48h:
    """Represents a frozen, leakage-safe snapshot of an event at TCA-48h."""
    def __init__(
        self,
        event_id: str,
        tca: pd.Timestamp,
        cutoff_utc: pd.Timestamp,
        eligible: bool,
        exclusion_reason: Optional[str],
        eligible_messages: pd.DataFrame,
        excluded_messages: pd.DataFrame,
        target_final_pc: float,
        sat_1_id: str = "SAT_1",
        sat_2_id: str = "SAT_2"
    ):
        self.event_id = event_id
        self.tca = tca
        self.tca_utc = tca.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.cutoff = cutoff_utc
        self.cutoff_utc = cutoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.eligible = eligible
        self.exclusion_reason = exclusion_reason
        self.eligible_messages = eligible_messages.reset_index(drop=True)
        self.excluded_messages = excluded_messages.reset_index(drop=True)
        self.target_final_pc = target_final_pc
        self.sat_1_id = sat_1_id
        self.sat_2_id = sat_2_id
        self.eligible_message_count = len(eligible_messages)
        self.excluded_message_count = len(excluded_messages)

        if self.eligible and self.eligible_message_count > 0:
            latest_eligible = self.eligible_messages.iloc[-1]
            self.latest_eligible_message_time = latest_eligible["message_time"]
            self.latest_eligible_message_utc = self.latest_eligible_message_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            self.last_eligible_pc = float(latest_eligible["collision_probability"])
        else:
            self.latest_eligible_message_time = None
            self.latest_eligible_message_utc = None
            self.last_eligible_pc = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert snapshot metadata to dictionary following Appendix A contract."""
        return {
            "event_id": self.event_id,
            "tca_utc": self.tca_utc,
            "cutoff_utc": self.cutoff_utc,
            "latest_eligible_message_utc": self.latest_eligible_message_utc,
            "eligible_message_count": self.eligible_message_count,
            "excluded_message_count": self.excluded_message_count,
            "target_final_pc": self.target_final_pc,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
            "last_eligible_pc": self.last_eligible_pc,
            "sat_1_id": self.sat_1_id,
            "sat_2_id": self.sat_2_id
        }

def build_48h_snapshot(event: ConjunctionEventHistory) -> EventSnapshot48h:
    """
    Enforce TCA - 48h boundary rule:
    eligible message_time <= TCA - 48 hours.
    Boundary message exactly at TCA - 48h is included.
    Messages with message_time > TCA - 48h are strictly excluded from model inputs.
    If no message exists at or before cutoff, event is marked ineligible.
    """
    cutoff_time = event.tca - timedelta(hours=CUTOFF_HOURS_BEFORE_TCA)

    # Boundary logic: <= cutoff_time is eligible
    eligible_mask = event.messages["message_time"] <= cutoff_time
    eligible_msgs = event.messages[eligible_mask].copy()
    excluded_msgs = event.messages[~eligible_mask].copy()

    if len(eligible_msgs) == 0:
        return EventSnapshot48h(
            event_id=event.event_id,
            tca=event.tca,
            cutoff_utc=cutoff_time,
            eligible=False,
            exclusion_reason="NO_OBSERVATION_BEFORE_TCA_MINUS_48H",
            eligible_messages=eligible_msgs,
            excluded_messages=excluded_msgs,
            target_final_pc=event.final_target_pc,
            sat_1_id=event.sat_1_id,
            sat_2_id=event.sat_2_id
        )

    return EventSnapshot48h(
        event_id=event.event_id,
        tca=event.tca,
        cutoff_utc=cutoff_time,
        eligible=True,
        exclusion_reason=None,
        eligible_messages=eligible_msgs,
        excluded_messages=excluded_msgs,
        target_final_pc=event.final_target_pc,
        sat_1_id=event.sat_1_id,
        sat_2_id=event.sat_2_id
    )

def build_all_snapshots(events: Dict[str, ConjunctionEventHistory]) -> Dict[str, EventSnapshot48h]:
    """Build snapshots for all events in the dataset."""
    return {event_id: build_48h_snapshot(event) for event_id, event in events.items()}
