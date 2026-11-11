"""
ORBIT PULSE - Ingestion & Schema Validation
===========================================
Validates raw CDM records, parses timestamps into standard UTC,
checks plausible ranges and TCA consistency, and isolates quarantined records.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["event_id", "message_time", "tca", "collision_probability"]

def validate_cdm_records(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Validates raw CDM records.
    Returns:
        valid_df: Cleaned and validated dataframe
        quarantine_df: Quarantined records that failed validation
        audit_summary: Dictionary of validation counts, exclusions, and issues
    """
    audit = {
        "total_records_ingested": len(df_raw),
        "valid_records_count": 0,
        "quarantined_records_count": 0,
        "duplicate_records_count": 0,
        "missing_required_fields": {},
        "timestamp_parse_errors": 0,
        "tca_inconsistencies": 0,
        "pc_range_violations": 0,
        "pc_parse_errors": 0,
        "pc_non_finite_values": 0,
        "quarantine_reasons": []
    }

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), audit

    df = df_raw.copy()
    quarantine_indices = set()

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation: Missing required columns: {missing_cols}")

    # 2. Check for missing required values
    for col in REQUIRED_COLUMNS:
        null_count = int(df[col].isnull().sum())
        audit["missing_required_fields"][col] = null_count
        if null_count > 0:
            null_idx = df[df[col].isnull()].index
            quarantine_indices.update(null_idx)
            audit["quarantine_reasons"].append(f"Null value in required column: {col}")
            #

    # 3. Parse and validate timestamps (message_time and tca)
    for ts_col in ["message_time", "tca"]:
        try:
            df[ts_col] = pd.to_datetime(
                df[ts_col],
                utc=True,
                errors="coerce",
                format="mixed"
            )

            ts_nulls = df[df[ts_col].isnull()].index

            if len(ts_nulls) > 0:
                quarantine_indices.update(ts_nulls)
                audit["timestamp_parse_errors"] += len(ts_nulls)
                audit["quarantine_reasons"].append(
                    f"Unparseable timestamp in: {ts_col}"
                )

        except Exception as e:
            raise ValueError(
                f"Failed parsing timestamps for {ts_col}: {str(e)}"
            )

    # 4. Quarantine messages occurring more than 2 hours after TCA.
    POST_TCA_TOLERANCE_HOURS = 2.0

    post_tca_violations = df[
        df["message_time"] >
        (df["tca"] + pd.Timedelta(hours=POST_TCA_TOLERANCE_HOURS))
    ].index

    if len(post_tca_violations) > 0:
        quarantine_indices.update(post_tca_violations)
        audit["quarantine_reasons"].append(
            "Message time occurs more than 2 hours after TCA"
        )

    # 5. Check TCA consistency within each event.
    # Allow at most 1 second of timestamp variation.
    TCA_TOLERANCE_SECONDS = 1.0

    valid_tca_rows = df[
        df["event_id"].notna() & df["tca"].notna()
    ]

    tca_range_per_event = valid_tca_rows.groupby("event_id")["tca"].agg(
        lambda values: (values.max() - values.min()).total_seconds()
    )

    inconsistent_events = tca_range_per_event[
        tca_range_per_event > TCA_TOLERANCE_SECONDS
    ].index

    if len(inconsistent_events) > 0:
        inconsistent_idx = df[
            df["event_id"].isin(inconsistent_events)
        ].index

        quarantine_indices.update(inconsistent_idx)
        audit["tca_inconsistencies"] += len(inconsistent_idx)
        audit["quarantine_reasons"].append(
            f"Inconsistent TCA for events: {list(inconsistent_events)}"
        )

        # 6. Validate numerical collision probability values.
    # Convert safely so malformed strings cannot crash validation.
    pc_original = df["collision_probability"]
    pc_numeric = pd.to_numeric(pc_original, errors="coerce")

    # Convert safely so malformed strings cannot crash validation.
    pc_original = df["collision_probability"]
    pc_numeric = pd.to_numeric(pc_original, errors="coerce")

    # Detect non-null values that cannot be converted to numbers.
    pc_parse_invalid = pc_original.notna() & pc_numeric.isna()
    pc_parse_indices = df.index[pc_parse_invalid]

    if len(pc_parse_indices) > 0:
        quarantine_indices.update(pc_parse_indices)
        audit["pc_parse_errors"] = len(pc_parse_indices)
        audit["quarantine_reasons"].append(
            "Non-numeric collision probability values"
        )

    # Store normalized numeric values for downstream processing.
    df["collision_probability"] = pc_numeric

    # Detect infinity and other non-finite numeric values.
    pc_non_finite = pc_numeric.notna() & ~np.isfinite(pc_numeric)
    pc_non_finite_indices = df.index[pc_non_finite]

    if len(pc_non_finite_indices) > 0:
        quarantine_indices.update(pc_non_finite_indices)
        audit["pc_non_finite_values"] = len(pc_non_finite_indices)
        audit["quarantine_reasons"].append(
            "Non-finite collision probability values"
        )

    # Validate finite values: 0.0 <= Pc <= 1.0.
    pc_range_invalid = (
        pc_numeric.notna()
        & np.isfinite(pc_numeric)
        & ((pc_numeric < 0.0) | (pc_numeric > 1.0))
    )

    pc_range_indices = df.index[pc_range_invalid]

    if len(pc_range_indices) > 0:
        quarantine_indices.update(pc_range_indices)
        audit["pc_range_violations"] = len(pc_range_indices)
        audit["quarantine_reasons"].append(
            "Collision probability out of valid bounds [0, 1]"
        )
    # 7. Check for exact duplicate rows or message_id collisions
    dedup_subset = ["event_id", "message_time"]
    if "message_id" in df.columns:
        dedup_subset = ["event_id", "message_id"]
    duplicate_mask = df.duplicated(subset=dedup_subset, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    audit["duplicate_records_count"] = duplicate_count

    # Split into valid and quarantine
    quarantine_mask = df.index.isin(quarantine_indices)
    quarantine_df = df[quarantine_mask].copy()

    # Exclude quarantined and duplicates from valid set
    valid_df = df[~quarantine_mask & ~duplicate_mask].copy()

    audit["valid_records_count"] = len(valid_df)
    audit["quarantined_records_count"] = len(quarantine_df)

    # Preserve metadata
    valid_df.attrs = df_raw.attrs.copy() if hasattr(df_raw, "attrs") else {}

    return valid_df, quarantine_df, audit
