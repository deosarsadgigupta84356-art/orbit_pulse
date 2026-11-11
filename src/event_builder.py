"""
ORBIT PULSE - Conjunction Event Builder
=======================================
Groups validated CDM records by stable event_id and chronologically
sorts message histories without mixing different conjunction events.
"""

from typing import Dict, List, Any
import pandas as pd

class ConjunctionEventHistory:
    """Represents the complete chronological CDM history of a single encounter."""
    def __init__(self, event_id: str, messages_df: pd.DataFrame):
        self.event_id = str(event_id)
        # Ensure strict chronological sorting by message_time
        self.messages = messages_df.sort_values("message_time", ascending=True).reset_index(drop=True)
        self.tca = self.messages["tca"].iloc[0]
        self.sat_1_id = self.messages["sat_1_id"].iloc[0] if "sat_1_id" in self.messages.columns else "SAT_1"
        self.sat_2_id = self.messages["sat_2_id"].iloc[0] if "sat_2_id" in self.messages.columns else "SAT_2"
        self.total_messages = len(self.messages)
        self.first_message_time = self.messages["message_time"].iloc[0]
        self.last_message_time = self.messages["message_time"].iloc[-1]
        # Final reported target from the latest known message in the complete history
        self.final_target_pc = float(self.messages["collision_probability"].iloc[-1])

    def __repr__(self) -> str:
        return f"<ConjunctionEventHistory {self.event_id}: {self.total_messages} CDMs, TCA={self.tca}>"

def build_event_histories(valid_df: pd.DataFrame) -> Dict[str, ConjunctionEventHistory]:
    """
    Groups validated CDM records by event_id.
    Ensures different event IDs never mix and maintains chronological sorting.
    """
    if valid_df.empty:
        return {}

    events: Dict[str, ConjunctionEventHistory] = {}
    grouped = valid_df.groupby("event_id")

    for event_id, group in grouped:
        event_obj = ConjunctionEventHistory(event_id=str(event_id), messages_df=group)
        events[str(event_id)] = event_obj

    return events
