"""
ORBIT PULSE - Synthetic CDM Fixture Generator
=============================================
WARNING & DATA PROVENANCE NOTICE:
This dataset is a SYNTHETIC TEST FIXTURE designed strictly for software engineering,
pipeline validation, automated testing, and UI demonstration of the ORBIT PULSE prototype.
It does NOT represent actual on-orbit tracking data or real space collisions.
In accordance with Blueprint Section 1, 7, and 16:
NEVER report synthetic predictive performance as real validation.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

def generate_synthetic_cdm_fixture(
    output_path: str = "data/raw/synthetic_cdm_fixture.csv",
    num_events: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)

    # Chronological timeline starting from Jan 1, 2026
    start_sim_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    records = []

    # Generate 40 events spread over 90 days
    # Event types:
    # 32 standard eligible events (messages spread across TCA-7d to TCA-12h)
    # 4 ineligible events (first message arrives after TCA-48h, e.g. at TCA-36h)
    # 2 boundary test events (message exactly at TCA-48h)
    # 2 duplicate test events (contains exact duplicate rows)

    for i in range(num_events):
        event_id = f"EVT-2026-{i+1:04d}"
        sat1_id = f"SAT-{1000 + (i % 12)}"
        sat2_id = f"DEB-{8000 + i}"

        # TCA spaced out chronologically
        days_offset = i * 2.2 + np.random.uniform(0.1, 1.0)
        tca = start_sim_time + timedelta(days=days_offset)

        # Determine event risk character
        # 0: Low risk throughout (~60%)
        # 1: Moderate risk (~25%)
        # 2: High risk / escalating (~15%)
        risk_profile = np.random.choice([0, 1, 2], p=[0.60, 0.25, 0.15])

        # Base physical parameters
        rel_speed = float(np.random.uniform(7000.0, 14500.0)) # m/s

        if risk_profile == 0:
            final_miss = float(np.random.uniform(2500.0, 15000.0)) # m
            final_log_pc = float(np.random.uniform(-10.0, -6.5))
        elif risk_profile == 1:
            final_miss = float(np.random.uniform(800.0, 2500.0))
            final_log_pc = float(np.random.uniform(-6.0, -4.2))
        else: # High risk
            final_miss = float(np.random.uniform(120.0, 600.0))
            final_log_pc = float(np.random.uniform(-3.8, -2.1))

        final_pc = 10 ** final_log_pc

        # Message timing strategy
        # Is this an ineligible event? (event i in [28, 29, 30, 31])
        is_ineligible = (i in [28, 29, 30, 31])
        has_boundary = (i in [5, 15])

        if is_ineligible:
            # First message only arrives at TCA - 36h, then TCA - 20h, TCA - 8h
            offsets_hours = [-36.0, -20.0, -8.0]
        else:
            # Messages start 5-7 days before TCA
            num_msgs = np.random.randint(4, 9)
            earliest_h = np.random.uniform(-168.0, -120.0)
            offsets_hours = sorted(np.random.uniform(earliest_h, -10.0, size=num_msgs))

            # Ensure at least 2 messages before TCA-48h
            offsets_hours[0] = min(offsets_hours[0], -96.0)
            offsets_hours[1] = min(offsets_hours[1], -52.0)

            if has_boundary:
                # Add a message EXACTLY at TCA - 48h
                offsets_hours.append(-48.0)
                offsets_hours = sorted(list(set(offsets_hours)))

        # Build individual CDMs for this event
        for m_idx, offset_h in enumerate(offsets_hours):
            msg_time = tca + timedelta(hours=offset_h)
            msg_id = f"CDM-{event_id}-{m_idx+1:02d}"

            # Noise decreases as time approaches TCA
            # Covariance shrinks as encounter nears
            progress = (offset_h + 168.0) / 168.0  # 0 at -168h, 1 at 0h
            progress = max(0.01, min(1.0, progress))

            # Miss distance evolves toward final
            curr_miss = max(50.0, final_miss + (1.0 - progress) * np.random.normal(0, 1000.0))

            # Pc evolves toward final reported Pc
            curr_log_pc = final_log_pc + (1.0 - progress) * np.random.normal(0, 0.9)
            curr_pc = max(1e-15, min(1.0, 10 ** curr_log_pc))

            # Covariance diagonals (m^2)
            c_r_r = float(max(10.0, (1.0 - progress) * 20000.0 + 500.0 + np.random.normal(0, 50.0)))
            c_t_t = float(max(20.0, (1.0 - progress) * 80000.0 + 2000.0 + np.random.normal(0, 200.0)))
            c_n_n = float(max(15.0, (1.0 - progress) * 40000.0 + 1000.0 + np.random.normal(0, 100.0)))

            # Introduce occasional missing values in optional fields for realism (test completeness)
            if np.random.rand() < 0.05 and not has_boundary:
                c_n_n = np.nan

            records.append({
                "message_id": msg_id,
                "event_id": event_id,
                "message_time": msg_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tca": tca.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sat_1_id": sat1_id,
                "sat_2_id": sat2_id,
                "miss_distance": round(curr_miss, 2),
                "relative_speed": round(rel_speed, 2),
                "collision_probability": curr_pc,
                "c_r_r": round(c_r_r, 2) if not np.isnan(c_r_r) else np.nan,
                "c_t_t": round(c_t_t, 2) if not np.isnan(c_t_t) else np.nan,
                "c_n_n": round(c_n_n, 2) if not np.isnan(c_n_n) else np.nan,
                "is_synthetic_fixture": True
            })

            # Duplicate test: inject exact duplicate for event 10
            if i == 10 and m_idx == 0:
                records.append(records[-1].copy())

    df = pd.DataFrame(records)

    # Shuffle slightly to test robust chronological sorting
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_shuffled.to_csv(output_path, index=False)
    print(f"Generated synthetic CDM fixture with {len(df_shuffled)} rows across {num_events} events at {output_path}")
    return df_shuffled

if __name__ == "__main__":
    generate_synthetic_cdm_fixture()
