"""
ORBIT PULSE - Feature Engineering
=================================
Transforms 48h event snapshots into feature vectors.
Extracts Pc history, encounter geometry, uncertainty, and temporal indicators.
Strictly ensures zero feature leakage: features use ONLY eligible messages.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.snapshot_48h import EventSnapshot48h

EPSILON = 1e-12

def safe_log10(val: float, eps: float = EPSILON) -> float:
    """Safely compute log10 with an epsilon offset to avoid -inf."""
    if pd.isna(val) or val is None:
        return np.nan
    val_clean = max(float(val), 0.0)
    return float(np.log10(val_clean + eps))

def extract_snapshot_features(snapshot: EventSnapshot48h) -> Optional[Dict[str, float]]:
    """
    Extracts features for a single eligible event snapshot.
    Returns None if the event is ineligible.
    """
    if not snapshot.eligible or snapshot.eligible_message_count == 0:
        return None

    msgs = snapshot.eligible_messages
    n_msgs = len(msgs)

    # 1. Collision Probability (Pc) history features
    pcs = msgs["collision_probability"].values
    log_pcs = [safe_log10(p) for p in pcs]

    last_pc_log = log_pcs[-1]
    min_pc_log = float(np.min(log_pcs))
    max_pc_log = float(np.max(log_pcs))
    mean_pc_log = float(np.mean(log_pcs))
    std_pc_log = float(np.std(log_pcs)) if n_msgs > 1 else 0.0

    # Trend slope: change in log10(Pc) per hour of update
    if n_msgs > 1:
        time_deltas = (msgs["message_time"] - msgs["message_time"].iloc[0]).dt.total_seconds() / 3600.0
        if time_deltas.iloc[-1] > 0.01:
            # Linear slope
            slope = float(np.polyfit(time_deltas.values, log_pcs, deg=1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0

    # 2. Geometry features
    if "miss_distance" in msgs.columns and not msgs["miss_distance"].dropna().empty:
        latest_miss_dist = float(msgs["miss_distance"].dropna().iloc[-1])
        min_miss_dist = float(msgs["miss_distance"].dropna().min())
    else:
        latest_miss_dist = 5000.0
        min_miss_dist = 5000.0

    if "relative_speed" in msgs.columns and not msgs["relative_speed"].dropna().empty:
        latest_rel_speed = float(msgs["relative_speed"].dropna().iloc[-1])
    else:
        latest_rel_speed = 10000.0

    # 3. Covariance uncertainty features
    latest_log_radial_cov = safe_log10(msgs["c_r_r"].dropna().iloc[-1]) if ("c_r_r" in msgs.columns and not msgs["c_r_r"].dropna().empty) else 3.0
    latest_log_alongtrack_cov = safe_log10(msgs["c_t_t"].dropna().iloc[-1]) if ("c_t_t" in msgs.columns and not msgs["c_t_t"].dropna().empty) else 4.0
    latest_log_crosstrack_cov = safe_log10(msgs["c_n_n"].dropna().iloc[-1]) if ("c_n_n" in msgs.columns and not msgs["c_n_n"].dropna().empty) else 3.5

    # 4. Temporal & data quality features
    latest_msg_time = snapshot.latest_eligible_message_time
    latest_age_hours = (snapshot.cutoff - latest_msg_time).total_seconds() / 3600.0
    latest_age_hours = max(0.0, float(latest_age_hours))

    # Completeness ratio across expected fields
    expected_cols = ["message_id", "message_time", "tca", "collision_probability", "miss_distance", "relative_speed", "c_r_r", "c_t_t", "c_n_n"]
    present_elements = 0
    total_elements = 0
    for col in expected_cols:
        if col in msgs.columns:
            present_elements += int(msgs[col].notnull().sum())
            total_elements += len(msgs)
        else:
            total_elements += len(msgs)
    completeness_ratio = float(present_elements / total_elements) if total_elements > 0 else 0.0

    features = {
        "last_eligible_pc_log": last_pc_log,
        "min_eligible_pc_log": min_pc_log,
        "max_eligible_pc_log": max_pc_log,
        "mean_eligible_pc_log": mean_pc_log,
        "std_eligible_pc_log": std_pc_log,
        "eligible_message_count": float(n_msgs),
        "pc_trend_slope": slope,
        "latest_miss_distance": latest_miss_dist,
        "min_miss_distance": min_miss_dist,
        "latest_relative_speed": latest_rel_speed,
        "latest_log_radial_cov": latest_log_radial_cov,
        "latest_log_alongtrack_cov": latest_log_alongtrack_cov,
        "latest_log_crosstrack_cov": latest_log_crosstrack_cov,
        "time_to_tca_at_cutoff_hours": 48.0,
        "latest_message_age_hours": latest_age_hours,
        "completeness_ratio": completeness_ratio
    }

    return features

def build_dataset_feature_matrix(snapshots: Dict[str, EventSnapshot48h]) -> pd.DataFrame:
    """
    Builds a single feature table from all event snapshots.
    Keeps final target isolated as training/evaluation label only.
    """
    rows = []
    for event_id, snap in snapshots.items():
        if not snap.eligible:
            continue
        feat = extract_snapshot_features(snap)
        if feat is not None:
            feat_row = feat.copy()
            feat_row["event_id"] = event_id
            feat_row["tca"] = snap.tca
            # Isolated labels
            feat_row["target_final_pc"] = snap.target_final_pc
            feat_row["target_final_log_pc"] = safe_log10(snap.target_final_pc)
            # Binary dangerous target (Pc >= 1e-4)
            feat_row["target_is_dangerous"] = int(snap.target_final_pc >= 1e-4)
            rows.append(feat_row)

    df_feat = pd.DataFrame(rows)
    if not df_feat.empty:
        # Sort chronologically by TCA
        df_feat = df_feat.sort_values("tca").reset_index(drop=True)
    return df_feat
