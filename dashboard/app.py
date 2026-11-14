"""
ORBIT PULSE - Mission-Control Analytics Console
================================================
Satellite Conjunction Risk Intelligence & Decision Support
Follows the ORBIT PULSE Master Blueprint as exact source of truth.
"""

import os
import sys
from pathlib import Path
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import html
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import streamlit as st

from src.data_loader import load_raw_cdm_data, get_thresholds_config, get_feature_schema_config
from src.validation import validate_cdm_records
from src.event_builder import build_event_histories
from src.snapshot_48h import build_all_snapshots, build_48h_snapshot
from src.feature_engineering import build_dataset_feature_matrix, extract_snapshot_features
from src.predict import OrbitPulsePredictor
from dashboard.esa_view import render_esa_screen
from src.audit import record_audit_entry, load_audit_trail, hash_snapshot_inputs
from dashboard.theme import (
    CUSTOM_CSS, COLOR_BG, COLOR_PANEL, COLOR_ACCENT_CYAN,
    COLOR_LOW, COLOR_UNCERTAIN, COLOR_HIGH, COLOR_TEXT_MUTED
)
from dashboard.charts import (
    create_event_pc_history_chart,
    create_risk_monitor_timeline_chart,
    create_forecast_sensitivity_chart,
    create_ablation_comparison_chart,
    create_confusion_matrix_chart,
    create_risk_distribution_chart,
    create_precision_recall_chart,
    create_error_distribution_chart
)
from dashboard.orbital_view import create_orbital_scene

st.set_page_config(
    page_title="ORBIT PULSE | Conjunction Risk Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Aerospace Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "REPLAY"
if "replay_index" not in st.session_state:
    st.session_state.replay_index = 60  # Initial sequence count for replay demo
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None
if "audit_keys" not in st.session_state:
    st.session_state.audit_keys = set()
# Top Navigation Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
st.sidebar.title("ORBIT PULSE")
st.sidebar.caption("Satellite Conjunction Risk Intelligence & Decision Support")
st.sidebar.markdown("### Data Source")
selected_data_source = st.sidebar.radio(
    "Choose intelligence dataset:",
    ["Synthetic Demonstration", "Historical ESA Dataset"],
    index=0,
    key="orbit_pulse_data_source"
)
# Shared Navigation — Available for Both Datasets
screens = [
    "Mission Overview",
    "Live Risk Monitor",
    "Event Explorer",
    "Event Detail",
    "Evidence & Explanation",
    "Model Performance & Ablations",
    "Audit & Data Quality",
    "3D Orbital Intelligence"
]

active_screen = st.sidebar.radio("Console View:", screens, index=0)

# Sidebar Safety Disclaimer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small><b>Historical Data Decision Support</b><br>"
    "Advisory output only. Not for autonomous spacecraft manoeuvre decisions.<br>"
    "Qualified operators retain final operational authority.</small>",
    unsafe_allow_html=True
)

# Historical ESA Dataset Routing
if selected_data_source == "Historical ESA Dataset":
    st.sidebar.info("Historical ESA Data · Offline Inference")
    render_esa_screen(active_screen)
    st.stop()

# Initialize the original synthetic predictor only in synthetic mode.
if "predictor" not in st.session_state:
    try:
        st.session_state.predictor = OrbitPulsePredictor()
        st.session_state.model_load_error = None
    except Exception as e:
        st.session_state.predictor = None
        st.session_state.model_load_error = f"{type(e).__name__}: {e}"
if st.session_state.predictor is None:
    st.sidebar.error("MODEL OFFLINE — predictions are disabled")
    if st.sidebar.button("Retry model loading", use_container_width=True):
        try:
            st.session_state.predictor = OrbitPulsePredictor()
            st.session_state.model_load_error = None
            st.rerun()
        except Exception as exc:
            st.session_state.model_load_error = f"{type(exc).__name__}: {exc}"
    if st.session_state.get("model_load_error"):
        with st.sidebar.expander("Model loading diagnostics"):
            st.code(st.session_state.model_load_error)
else:
    st.sidebar.success(f"MODEL READY · {st.session_state.predictor.model_version}")

# Operational Mode Selector
st.sidebar.markdown("### 📡 Operational Mode")
mode_selection = st.sidebar.radio(
    "Data Source Mode:",
    ["REPLAY", "LIVE FEED", "MANUAL / BATCH"],
    index=["REPLAY", "LIVE FEED", "MANUAL / BATCH"].index(st.session_state.data_mode)
)
st.session_state.data_mode = mode_selection

# Mode-Specific Sidebar Controls
if st.session_state.data_mode == "REPLAY":
    st.sidebar.markdown("**Replay Controls**")
    speed = st.sidebar.slider("Playback Increment (messages/step):", 1, 20, 5)
    col_r1, col_r2 = st.sidebar.columns(2)
    with col_r1:
        if st.button("Step Forward ⏩"):
            st.session_state.replay_index += speed
    with col_r2:
        if st.button("Reset Stream 🔄"):
            st.session_state.replay_index = 40

elif st.session_state.data_mode == "MANUAL / BATCH":
    st.sidebar.markdown("**Manual File Load**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV CDM File", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.manual_df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom CSV Loaded")
        except Exception as exc:
            st.session_state.manual_df = None
            st.sidebar.error(f"Could not read CSV: {exc}")
    else:
        st.session_state.manual_df = None

# -------------------------------------------------------------
# Data Loading & Processing Pipeline
# -------------------------------------------------------------
def load_and_validate_pipeline(mode: str, replay_idx: int, manual_data=None):
    raw_df = load_raw_cdm_data()

    if mode == "REPLAY":
        # Sort chronologically by message_time and take first replay_idx
        df_sorted = raw_df.sort_values("message_time").reset_index(drop=True)
        active_raw = df_sorted.iloc[:max(0, min(replay_idx, len(df_sorted)))].copy()
        feed_time = active_raw["message_time"].iloc[-1] if not active_raw.empty else "NO RECORDS"
    elif mode == "LIVE FEED":
        active_raw = raw_df.copy()
        feed_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:  # MANUAL
        active_raw = manual_data
        if active_raw is None:
            active_raw = raw_df.copy()

        else:
            active_raw = active_raw.copy()

            # Conservatively classify provenance for manual uploads.
            provenance_status = "UNKNOWN_UNVERIFIED"

            if "is_synthetic_fixture" in active_raw.columns:
                raw_flags = active_raw["is_synthetic_fixture"].dropna()
                normalized_flags = (
                    raw_flags.astype(str).str.strip().str.lower()
                )

                true_values = {"true", "1", "yes", "y"}
                recognized_flags = set(normalized_flags.unique())

                if (
                    recognized_flags
                    and recognized_flags.issubset(true_values)
                ):
                    provenance_status = "SYNTHETIC"

            is_synthetic = provenance_status == "SYNTHETIC"

            active_raw.attrs["dataset_version"] = "MANUAL_UPLOAD"
            active_raw.attrs["is_synthetic_fixture"] = is_synthetic
            active_raw.attrs["provenance_status"] = provenance_status

            if is_synthetic:
                active_raw.attrs["license_notice"] = (
                    "SYNTHETIC TEST FIXTURE: UI & Pipeline Validation Only. "
                    "Not Real-World Operational Validation."
                )
            else:
                active_raw.attrs["license_notice"] = (
                    "DATA PROVENANCE UNVERIFIED: Official source and usage "
                    "authorization have not been established."
                )

        feed_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if active_raw.empty:
        active_raw = raw_df.iloc[0:0].copy()
    try:
        valid_df, quarantine_df, audit = validate_cdm_records(active_raw)
    except (ValueError, KeyError, TypeError) as exc:
        # Keep the dashboard usable when a manual CSV has an unsupported schema.
        valid_df = raw_df.iloc[0:0].copy()
        quarantine_df = active_raw.copy()
        audit = {
            "total_records_ingested": len(active_raw),
            "valid_records_count": 0,
            "quarantined_records_count": len(active_raw),
            "duplicate_records_count": 0,
            "missing_required_fields": {},
            "timestamp_parse_errors": 0,
            "tca_inconsistencies": 0,
            "pc_range_violations": 0,
            "quarantine_reasons": [f"Input schema or type validation failed: {exc}"],
            "schema_error": str(exc)
        }
    events = build_event_histories(valid_df)
    snapshots = build_all_snapshots(events)
    return active_raw, valid_df, quarantine_df, audit, events, snapshots, feed_time

# Compute summaries from current snapshots without caching global/session state.
def get_all_event_predictions(current_snapshots, predictor):
    summaries = []
    for evt_id, snap in current_snapshots.items():
        pred_out = predictor.predict_snapshot(snap) if predictor else {}
        if predictor:
            audit_key = (
                f"{evt_id}|{snap.cutoff_utc}|{hash_snapshot_inputs(snap)}|"
                f"{predictor.model_version}"
            )
            if audit_key not in st.session_state.audit_keys:
                record_audit_entry(pred_out, snap)
                st.session_state.audit_keys.add(audit_key)
        summaries.append({
            "event_id": evt_id,
            "sat_1_id": snap.sat_1_id,
            "sat_2_id": snap.sat_2_id,
            "tca": snap.tca,
            "tca_utc": snap.tca_utc,
            "cutoff_utc": snap.cutoff_utc,
            "eligible": snap.eligible,
            "eligible_message_count": snap.eligible_message_count,
            "excluded_message_count": snap.excluded_message_count,
            "predicted_pc": pred_out.get("predicted_value"),
            "predicted_log": pred_out.get("predicted_log_value"),
            "risk_band": pred_out.get("risk_band", "MODEL_UNAVAILABLE"),
            "recommendation": pred_out.get("recommendation"),
            "advisory_label": pred_out.get("advisory_label", "MODEL UNAVAILABLE"),
            "freshness_hours": pred_out.get("data_freshness_hours"),
            "completeness": pred_out.get("evidence_completeness"),
            "stability": pred_out.get("forecast_stability"),
            "target_final_pc": snap.target_final_pc
        })
    summary_columns = [
        "event_id", "sat_1_id", "sat_2_id", "tca", "tca_utc", "cutoff_utc", "eligible",
        "eligible_message_count", "excluded_message_count", "predicted_pc",
        "predicted_log", "risk_band", "recommendation", "advisory_label",
        "freshness_hours", "completeness", "stability", "target_final_pc"
    ]
    return pd.DataFrame(summaries, columns=summary_columns)

raw_active, valid_df, quarantine_df, val_audit, events, snapshots, feed_time = load_and_validate_pipeline(
    st.session_state.data_mode,
    st.session_state.replay_index,
    getattr(st.session_state, "manual_df", None)
)
df_event_summaries = get_all_event_predictions(snapshots, st.session_state.predictor)

# Set default selected event if none
if (
    st.session_state.selected_event_id is None
    or st.session_state.selected_event_id not in snapshots
) and not df_event_summaries.empty:
    # Prefer an eligible event with high or uncertain risk for demonstration
    high_evts = df_event_summaries[df_event_summaries["risk_band"] == "HIGH"]
    if not high_evts.empty:
        st.session_state.selected_event_id = high_evts["event_id"].iloc[0]
    else:
        st.session_state.selected_event_id = df_event_summaries["event_id"].iloc[0]

# -------------------------------------------------------------
# Mission Control Masthead
# -------------------------------------------------------------
st.markdown("""
<div class="mission-masthead">
  <div class="masthead-kicker">ORBITAL SAFETY · CONJUNCTION INTELLIGENCE</div>
  <div class="masthead-title">MISSION CONTROL <span> / ORBIT PULSE</span></div>
  <div class="masthead-subtitle">Time-aware collision-probability forecasting · Evidence intelligence · Human-reviewed decisions</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Top Status Bar
# -------------------------------------------------------------
if val_audit.get("schema_error"):
    st.error(f"Uploaded dataset could not be processed: {val_audit['schema_error']}")

badge_class = "badge-replay" if st.session_state.data_mode == "REPLAY" else (
    "badge-replay" if st.session_state.data_mode == "LIVE FEED" else "badge-manual"
)
badge_text = f"REPLAY • historical data • simulated clock: {feed_time}" if st.session_state.data_mode == "REPLAY" else (
    f"SIMULATED FEED • not operational live data • clock: {feed_time}" if st.session_state.data_mode == "LIVE FEED" else
    f"MANUAL / BATCH • dataset loaded: {feed_time}"
)

st.markdown(f"""
<div class="status-bar">
    <div>
        <span class="{badge_class}">● {html.escape(str(badge_text))}</span>
        <span style="margin-left: 15px; color: {COLOR_TEXT_MUTED}; font-size: 0.85rem;">
                       Dataset: <code>{html.escape(str(raw_active.attrs.get('dataset_version', 'UNKNOWN_UNVERIFIED')))}</code>
            &nbsp; | &nbsp; Provenance: <b>{html.escape(str(raw_active.attrs.get('provenance_status', 'UNKNOWN_UNVERIFIED')))}</b>  </span>
    </div>
    <div style="font-size: 0.82rem; color: {COLOR_ACCENT_CYAN};">
        Cutoff Rule: <b>message_time ≤ TCA − 48h</b> | Model: <b>HistGradientBoosting + Baseline</b>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.predictor is None:
    st.error("Model unavailable: predictions are disabled. Open the sidebar's 'Model loading diagnostics' panel for the exact cause. Data ingestion and audit views remain available.")
    if st.session_state.get("model_load_error"):
        with st.expander("Model loading error details"):
            st.code(st.session_state.model_load_error)

# -------------------------------------------------------------
# SCREEN 1: MISSION OVERVIEW
# -------------------------------------------------------------
if active_screen == "Mission Overview":
    st.subheader("Mission Operations Overview")

    # KPI Grid
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CDMs Ingested</div>
            <div class="metric-value">{len(raw_active)}</div>
            <div class="metric-subtitle">Valid: {len(valid_df)} | Quarantined: {len(quarantine_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Conjunction Events</div>
            <div class="metric-value">{len(events)}</div>
            <div class="metric-subtitle">Active Space Encounters</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        n_eligible = int(df_event_summaries["eligible"].sum()) if not df_event_summaries.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Eligible (≤ TCA-48h)</div>
            <div class="metric-value" style="color: {COLOR_ACCENT_CYAN};">{n_eligible}</div>
            <div class="metric-subtitle">Admitted to Predictor</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        n_ineligible = len(events) - n_eligible
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ineligible Events</div>
            <div class="metric-value" style="color: {COLOR_UNCERTAIN};">{n_ineligible}</div>
            <div class="metric-subtitle">No CDMs pre-cutoff</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        n_high = int((df_event_summaries["risk_band"] == "HIGH").sum()) if not df_event_summaries.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Risk / Review</div>
            <div class="metric-value" style="color: {COLOR_HIGH};">{n_high}</div>
            <div class="metric-subtitle">Predicted Pc ≥ 1e-4</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        n_uncertain = int((df_event_summaries["risk_band"] == "UNCERTAIN").sum()) if not df_event_summaries.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Uncertain / Stale</div>
            <div class="metric-value" style="color: {COLOR_UNCERTAIN};">{n_uncertain}</div>
            <div class="metric-subtitle">Evidence Limited</div>
        </div>
        """, unsafe_allow_html=True)

    # Timeline of active conjunction events
    if not df_event_summaries.empty:
        st.markdown("### 🌌 Conjunction Risk Horizon")
        chart_col, dist_col = st.columns([2, 1])
        with chart_col:
            fig_timeline = create_risk_monitor_timeline_chart(df_event_summaries)
            st.plotly_chart(fig_timeline, use_container_width=True)
        with dist_col:
            st.plotly_chart(create_risk_distribution_chart(df_event_summaries), use_container_width=True)

    # High Priority Action Items Table
    st.markdown("### ⚠️ Priority Review Queue")
    urgent_df = df_event_summaries[df_event_summaries["risk_band"].isin(["HIGH", "UNCERTAIN"])].copy()
    if not urgent_df.empty:
        st.dataframe(
            urgent_df[["event_id", "tca_utc", "risk_band", "advisory_label", "predicted_pc", "freshness_hours", "stability"]],
            column_config={
                "predicted_pc": st.column_config.NumberColumn("Predicted Final Pc", format="%.2e"),
                "freshness_hours": st.column_config.NumberColumn("Data Latency (h)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("All current eligible conjunction events are currently within routine monitoring parameters.")

# -------------------------------------------------------------
# SCREEN 2: LIVE RISK MONITOR
# -------------------------------------------------------------
elif active_screen == "Live Risk Monitor":
    st.subheader("Live Stream & Stream Health Monitor")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### ⚡ Incoming CDM Ingestion Feed")
        recent_cdms = raw_active.sort_values("message_time", ascending=False).head(15)
        st.dataframe(
            recent_cdms[["message_id", "event_id", "message_time", "tca", "collision_probability", "miss_distance", "relative_speed"]],
            column_config={
                "collision_probability": st.column_config.NumberColumn("Reported Pc", format="%.2e"),
                "miss_distance": st.column_config.NumberColumn("Miss Dist (m)", format="%.1f"),
                "relative_speed": st.column_config.NumberColumn("Rel Speed (m/s)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )
    with col2:
        st.markdown("#### 🛡️ Feed Governance")
        st.markdown(f"""
        - **Feed Mode**: `{st.session_state.data_mode}`
        - **Last Ingested Timestamp**: `{feed_time}`
        - **Total Records Ingested**: `{len(raw_active)}`
        - **Valid CDM Records**: `{len(valid_df)}`
        - **Quarantine Pool**: `{len(quarantine_df)}`
        - **Duplicate Messages**: `{val_audit['duplicate_records_count']}`
        - **Status**: <span class="badge-live">SIMULATED / VALIDATED</span>
        """, unsafe_allow_html=True)
        if st.session_state.data_mode == "REPLAY":
            st.info(f"Replay cursor at record {st.session_state.replay_index} of {len(load_raw_cdm_data())}. Use sidebar to step forward.")

    if not df_event_summaries.empty:
        st.markdown("#### 📈 Dynamic Conjunction Risk Graph")
        fig_timeline = create_risk_monitor_timeline_chart(df_event_summaries)
        st.plotly_chart(fig_timeline, use_container_width=True)

# -------------------------------------------------------------
# SCREEN 3: EVENT EXPLORER
# -------------------------------------------------------------
elif active_screen == "Event Explorer":
    st.subheader("Conjunction Event Explorer")

    # Filter controls
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        search_query = st.text_input("🔍 Search Event ID:", "")
    with fcol2:
        band_filter = st.multiselect("Filter Risk Band:", ["HIGH", "UNCERTAIN", "LOW"], default=["HIGH", "UNCERTAIN", "LOW"])
    with fcol3:
        eligibility_filter = st.selectbox("Eligibility Filter:", ["All Events", "Eligible Only (≤ TCA-48h)", "Ineligible Only"])
    with fcol4:
        object_query = st.text_input("Search Object ID:", "")

    filtered_df = df_event_summaries.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["event_id"].astype(str).str.contains(search_query, case=False, regex=False, na=False)]
    if band_filter:
        filtered_df = filtered_df[filtered_df["risk_band"].isin(band_filter)]
    if object_query:
        object_mask = (
            filtered_df["sat_1_id"].astype(str).str.contains(object_query, case=False, regex=False, na=False)
            | filtered_df["sat_2_id"].astype(str).str.contains(object_query, case=False, regex=False, na=False)
        )
        filtered_df = filtered_df[object_mask]
    if eligibility_filter == "Eligible Only (≤ TCA-48h)":
        filtered_df = filtered_df[filtered_df["eligible"] == True]
    elif eligibility_filter == "Ineligible Only":
        filtered_df = filtered_df[filtered_df["eligible"] == False]


    if filtered_df.empty:
        st.warning(
            "No conjunction events match the selected filters. "
            "Try changing your search or filter selections."
        )
    else:
        st.download_button(
            "Download filtered event register (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="orbit_pulse_filtered_events.csv",
            mime="text/csv"
        )

        st.dataframe(
            filtered_df[[
                "event_id", "sat_1_id", "sat_2_id", "tca_utc",
                "eligible", "eligible_message_count",
                "excluded_message_count", "risk_band",
                "predicted_pc", "freshness_hours",
                "stability", "advisory_label"
            ]],
            column_config={
                "predicted_pc": st.column_config.NumberColumn(
                    "Predicted Final Pc", format="%.2e"
                ),
                "freshness_hours": st.column_config.NumberColumn(
                    "Freshness (h)", format="%.1f"
                )
            },
            hide_index=True,
            use_container_width=True
        )

        compare_ids = st.multiselect(
            "Compare events (select up to 4)",
            filtered_df["event_id"].tolist(),
            key="event_compare_ids"
        )

        if len(compare_ids) > 4:
            st.warning(
                "Please select no more than four events for side-by-side comparison."
            )
            compare_ids = compare_ids[:4]

        if compare_ids:
            compare_df = filtered_df[
                filtered_df["event_id"].isin(compare_ids)
            ].copy()

            st.markdown("#### Selected Event Comparison")
            st.dataframe(
                compare_df[[
                    "event_id", "tca_utc", "risk_band",
                    "predicted_pc", "eligible_message_count",
                    "freshness_hours", "completeness",
                    "stability", "advisory_label"
                ]],
                hide_index=True,
                use_container_width=True
            )

        event_ids = filtered_df["event_id"].tolist()
        selected = st.selectbox(
            "Inspect Conjunction Event in Detail:",
            event_ids,
            index=0
        )

        if st.button("Open Selected Event in Detail View"):
            st.session_state.selected_event_id = selected
            st.rerun()


# -------------------------------------------------------------
# SCREEN 4: EVENT DETAIL
# -------------------------------------------------------------
elif active_screen == "Event Detail":
    st.subheader("Conjunction Encounter Deep Dive")

    all_evt_ids = list(events.keys())
    if not all_evt_ids:
        st.info("No events are available in the current dataset or replay window. Load data or advance the replay stream.")
    else:
        sel_idx = all_evt_ids.index(st.session_state.selected_event_id) if st.session_state.selected_event_id in all_evt_ids else 0
        current_evt_id = st.selectbox("Select Active Event:", all_evt_ids, index=sel_idx)
        st.session_state.selected_event_id = current_evt_id

        event_obj = events[current_evt_id]
        snapshot = snapshots[current_evt_id]
        pred_res = st.session_state.predictor.predict_snapshot(snapshot) if st.session_state.predictor else {}

        # Header Card
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Event Identifier**: `{snapshot.event_id}`")
            st.markdown(f"**Primary Object**: `{snapshot.sat_1_id}`")
            st.markdown(f"**Secondary Object**: `{snapshot.sat_2_id}`")
        with c2:
            st.markdown(f"**TCA (UTC)**: `{snapshot.tca_utc}`")
            st.markdown(f"**Cutoff (TCA-48h)**: `{snapshot.cutoff_utc}`")
        with c3:
            st.markdown(f"**Eligible CDMs**: `{snapshot.eligible_message_count}`")
            st.markdown(f"**Post-Cutoff CDMs**: `{snapshot.excluded_message_count}`")
        with c4:
            band = pred_res.get("risk_band", "UNKNOWN")
            b_class = "badge-high" if band == "HIGH" else ("badge-uncertain" if band == "UNCERTAIN" else "badge-low")
            st.markdown(f"**Advisory Status**: <span class='{b_class}'>{html.escape(str(pred_res.get('advisory_label', 'N/A')))}</span>", unsafe_allow_html=True)
            predicted_value = pred_res.get("predicted_value")
            predicted_label = f"{predicted_value:.2e}" if predicted_value is not None else "N/A — ineligible / unavailable"
            st.markdown(f"**Predicted Final Pc**: `{predicted_label}`")

        # Interactive Chart
        st.markdown("### 📊 Chronological CDM History & Cutoff Marker")
        fig_hist = create_event_pc_history_chart(
            eligible_df=snapshot.eligible_messages,
            excluded_df=snapshot.excluded_messages,
            tca_utc=snapshot.tca_utc,
            cutoff_utc=snapshot.cutoff_utc,
            predicted_pc=pred_res.get("predicted_value"),
            final_actual_pc=snapshot.target_final_pc
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Message Breakdown Table
        t1, t2 = st.tabs(["Admitted Pre-Cutoff CDMs (Features)", "Excluded Post-Cutoff CDMs (Audit Only)"])
        with t1:
            if not snapshot.eligible_messages.empty:
                st.dataframe(snapshot.eligible_messages, use_container_width=True)
            else:
                st.warning("No observations were received on or before TCA-48h. This event is ineligible.")
        with t2:
            if not snapshot.excluded_messages.empty:
                st.dataframe(snapshot.excluded_messages, use_container_width=True)
            else:
                st.info("No post-cutoff messages received yet.")
        event_export = pd.concat([
            snapshot.eligible_messages.assign(cutoff_status="ELIGIBLE"),
            snapshot.excluded_messages.assign(cutoff_status="EXCLUDED_POST_CUTOFF")
        ], ignore_index=True).sort_values("message_time")
        st.download_button(
            "Download this event's CDM history (CSV)",
            data=event_export.to_csv(index=False).encode("utf-8"),
            file_name=f"{current_evt_id}_cdm_history.csv",
            mime="text/csv"
        )
# -------------------------------------------------------------
# SCREEN 5: EVIDENCE & EXPLANATION
# -------------------------------------------------------------
elif active_screen == "Evidence & Explanation":
    st.subheader("Evidence Intelligence & Forecast Sensitivity")
    st.markdown("*(Signature Architecture Feature: Three Independent Dimensions — Never collapsed into an arbitrary score)*")

    evidence_event_ids = list(events.keys())
    if not evidence_event_ids:
        st.info("No events are available for evidence analysis in the current dataset or replay window.")
    else:
        current_evt_id = st.session_state.selected_event_id if st.session_state.selected_event_id in evidence_event_ids else evidence_event_ids[0]
        if st.session_state.get("ev_select") not in evidence_event_ids:
            st.session_state.pop("ev_select", None)
        current_evt_id = st.selectbox(
            "Active Conjunction Event:",
            evidence_event_ids,
            index=evidence_event_ids.index(current_evt_id),
            key="ev_select"
        )
        st.session_state.selected_event_id = current_evt_id
        snapshot = snapshots[current_evt_id]
        pred_res = st.session_state.predictor.predict_snapshot(snapshot) if st.session_state.predictor else {}
        evidence = pred_res.get("evidence_intelligence", {})
        decision = pred_res.get("decision", {})

        # 3 Separate Panels
        p1, p2, p3 = st.columns(3)

        # Panel 1: Freshness
        with p1:
            fresh = evidence.get("freshness", {})
            f_cat = fresh.get("category", "UNKNOWN")
            f_class = "badge-live" if f_cat == "FRESH" else ("badge-uncertain" if f_cat == "AGEING" else "badge-high")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">1. Data Freshness</div>
                <div class="metric-value"><span class="{f_class}">{html.escape(str(f_cat))}</span></div>
                <div class="metric-subtitle">Latest Admitted Message Age: {fresh.get('age_hours', 'N/A')} h</div>
                <hr style="border-color: {COLOR_PANEL};">
                <small>{html.escape(str(fresh.get('description', '')))}</small>
            </div>
            """, unsafe_allow_html=True)

        # Panel 2: Completeness
        with p2:
            comp = evidence.get("completeness", {})
            c_cat = comp.get("category", "UNKNOWN")
            c_class = "badge-live" if c_cat == "HIGH" else ("badge-uncertain" if c_cat == "MODERATE" else "badge-high")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">2. Evidence Completeness</div>
                <div class="metric-value"><span class="{c_class}">{html.escape(str(c_cat))} ({int(comp.get('completeness_ratio', 0)*100)}%)</span></div>
                <div class="metric-subtitle">Required & Covariance Fields Present</div>
                <hr style="border-color: {COLOR_PANEL};">
                <small>{html.escape(str(comp.get('description', '')))}</small>
            </div>
            """, unsafe_allow_html=True)

        # Panel 3: Forecast Stability
        with p3:
            stab = evidence.get("stability", {})
            s_cat = stab.get("stability_status", "UNKNOWN")
            s_class = "badge-live" if s_cat == "STABLE" else ("badge-high" if s_cat == "SENSITIVE" else "badge-uncertain")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">3. Forecast Stability</div>
                <div class="metric-value"><span class="{s_class}">{html.escape(str(s_cat))}</span></div>
                <div class="metric-subtitle">Perturbation Spread: {stab.get('log_spread') if stab.get('log_spread') is not None else 'N/A'} log10 units</div>
                <hr style="border-color: {COLOR_PANEL};">
                <small>{html.escape(str(stab.get('description', '')))}</small>
            </div>
            """, unsafe_allow_html=True)

        # Sensitivity Plot
        st.markdown("### 🔬 Perturbation Experiment (Leave-One-Message-Out)")
        fig_sens = create_forecast_sensitivity_chart(evidence.get("stability", {}))
        st.plotly_chart(fig_sens, use_container_width=True)

        st.markdown("### 🧩 Model Input Snapshot")
        st.caption("Values below are derived only from CDMs admitted at or before TCA − 48h. They are input values, not causal feature-attribution scores.")
        feature_values = extract_snapshot_features(snapshot)
        if feature_values:
            feature_df = pd.DataFrame(
                [{"Feature": name, "Value": value} for name, value in feature_values.items()]
            )
            st.dataframe(feature_df, hide_index=True, use_container_width=True)
        else:
            st.info("No model feature vector is available because this event is ineligible.")

        # Advisory Review Status & Policy
        st.markdown("### 📋 Advisory Decision Support Recommendation")
        st.markdown(f"""
        <div style="background-color: {COLOR_PANEL}; border-left: 4px solid {COLOR_ACCENT_CYAN}; padding: 18px; border-radius: 6px;">
            <h4 style="margin: 0; color: {COLOR_TEXT_MUTED}; font-size: 0.9rem;">POLICY STATUS</h4>
            <h2 style="margin: 8px 0; color: {COLOR_ACCENT_CYAN};">{html.escape(str(decision.get('advisory_label', 'N/A')))}</h2>
            <p><b>Evidence Context:</b> <i>{html.escape(str(decision.get('evidence_qualifier', 'N/A')))}</i></p>
            <p><b>Audit Rationale:</b> {html.escape(str(decision.get('rationale', 'N/A')))}</p>
            <p><b>Recommended Operational Steps:</b></p>
            <ul>
                {''.join(f'<li>{html.escape(str(step))}</li>' for step in decision.get('actionable_next_steps', []))}
            </ul>
            <small style="color: {COLOR_TEXT_MUTED};">Operational Escalation Threshold Applied: Pc ≥ 1.00e-04 (configured review threshold)</small>
        </div>
        """, unsafe_allow_html=True)
# -------------------------------------------------------------
# SCREEN 6: MODEL PERFORMANCE & ABLATIONS
# -------------------------------------------------------------
elif active_screen == "Model Performance & Ablations":
    st.subheader("Model Validation & Mandatory Ablation Analysis")
    st.markdown("*(Strict chronological, event-separated held-out test evaluation)*")
    st.warning("The bundled evaluation uses a synthetic test fixture. These metrics demonstrate pipeline behaviour only; they are not real-world operational validation.")

    # Load evaluation metrics from reports/metrics.json
    metrics_path = str(PROJECT_ROOT / "reports" / "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    else:
        st.warning("Metrics report not generated yet. Please run evaluation.")
        metrics_data = None

    if metrics_data:
        st.download_button(
            "Download evaluation metrics (JSON)",
            data=json.dumps(metrics_data, indent=2),
            file_name="orbit_pulse_metrics.json",
            mime="application/json"
        )
        baseline_data = metrics_data["baseline_model"]
        model_data = metrics_data["trained_ml_model"]
        base_m = baseline_data["metrics"]
        ml_m = model_data["metrics"]

        if ml_m["mae_log10"] > base_m["mae_log10"]:
            st.warning(
                f"The trained ML model currently has higher MAE ({ml_m['mae_log10']:.4f}) than the last-eligible-Pc baseline ({base_m['mae_log10']:.4f}). "
                "The baseline is the stronger regression result on this held-out synthetic test split."
            )
        split = metrics_data.get("chronological_split", {})
        st.caption(
            f"Chronological event split — train: {split.get('train_events', 'N/A')}, "
            f"validation: {split.get('val_events', 'N/A')}, held-out test: {split.get('test_events', 'N/A')}. "
            "The operational classification threshold shown below is the configured Pc ≥ 1e-4 rule."
        )
        # Benchmark comparison cards
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.metric("Baseline MAE (log10)", f"{base_m['mae_log10']:.4f}")
        with bc2:
            st.metric("ML Model MAE (log10)", f"{ml_m['mae_log10']:.4f}", delta=f"{metrics_data['trained_ml_model']['relative_mae_improvement_pct']:.1f}%")
        with bc3:
            st.metric("Operational Recall (Pc ≥ 1e-4)", f"{ml_m['classification']['recall']:.2f}")
        with bc4:
            st.metric("False Negatives (Missed Danger)", f"{ml_m['classification']['fn']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📊 Mandatory Ablation Study (Experiments A, B, C, D)")
            fig_abl = create_ablation_comparison_chart(metrics_data["mandatory_ablations"])
            st.plotly_chart(fig_abl, use_container_width=True)

        with col_b:
            st.markdown("#### 🎯 Confusion Matrix (Held-out Test Events)")
            fig_cm = create_confusion_matrix_chart(ml_m["classification"])
            st.plotly_chart(fig_cm, use_container_width=True)

        pr_col, error_col = st.columns(2)
        with pr_col:
            st.markdown("#### Precision–Recall Comparison")
            st.plotly_chart(
                create_precision_recall_chart(
                    baseline_data.get("pr_curve", {}),
                    model_data.get("pr_curve", {})
                ),
                use_container_width=True
            )
        with error_col:
            st.markdown("#### Absolute Error Comparison")
            st.plotly_chart(
                create_error_distribution_chart(
                    baseline_data.get("absolute_error_log10", []),
                    model_data.get("absolute_error_log10", [])
                ),
                use_container_width=True
            )

        # Full ablation table
        st.markdown("#### 📑 Detailed Ablation Progression")
        abl_records = []
        for exp_key, data in metrics_data["mandatory_ablations"].items():
            abl_records.append({
                "Experiment": exp_key,
                "Feature Count": data["feature_count"],
                "MAE (log10)": data["metrics"]["mae_log10"],
                "RMSE (log10)": data["metrics"]["rmse_log10"],
                "Recall (Pc ≥ 1e-4)": data["metrics"]["classification"]["recall"],
                "Precision": data["metrics"]["classification"]["precision"]
            })
        st.dataframe(pd.DataFrame(abl_records), hide_index=True, use_container_width=True)

# -------------------------------------------------------------
# SCREEN 7: AUDIT & DATA QUALITY
# -------------------------------------------------------------
elif active_screen == "Audit & Data Quality":
    st.subheader("Data Quality, Provenance & Audit Trail")

    t1, t2, t3 = st.tabs(["Audit Trail Log", "Ingestion & Validation Audit", "Reproducibility Metadata"])

    with t1:
        st.markdown("#### 📜 Inference & Cutoff Audit Trail")
        audit_trail_df = load_audit_trail()
        if not audit_trail_df.empty:
            st.download_button(
                "Download complete audit trail (CSV)",
                data=audit_trail_df.to_csv(index=False).encode("utf-8"),
                file_name="orbit_pulse_audit_trail.csv",
                mime="text/csv"
            )
            st.dataframe(audit_trail_df.tail(20), use_container_width=True)
        else:
            st.info("No audit entries recorded yet in this session.")

    with t2:
        st.markdown("#### 🔍 Schema & Ingestion Health")
        st.json(val_audit)
        st.download_button(
            "Download validation audit (JSON)",
            data=json.dumps(val_audit, indent=2, default=str),
            file_name="orbit_pulse_validation_audit.json",
            mime="application/json"
        )

    with t3:
        st.markdown("#### ⚙️ Persisted Model Metadata")
        meta_path = str(PROJECT_ROOT / "models" / "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                st.json(json.load(f))

# -------------------------------------------------------------
# SCREEN 8: 3D ORBITAL INTELLIGENCE
# -------------------------------------------------------------
elif active_screen == "3D Orbital Intelligence":
    st.subheader("3D Orbital Intelligence")
    st.caption(
        "Interactive Earth-centered visualization with illustrative "
        "orbital trajectories and selected conjunction intelligence."
    )

    if df_event_summaries.empty:
        st.warning("No conjunction events are currently available.")
    else:
        event_options = df_event_summaries["event_id"].astype(str).tolist()

        default_event = st.session_state.selected_event_id
        default_index = (
            event_options.index(str(default_event))
            if str(default_event) in event_options
            else 0
        )

        selected_orbital_event = st.selectbox(
            "Select conjunction event",
            event_options,
            index=default_index,
            key="orbital_intelligence_event"
        )

        selected_rows = df_event_summaries[
            df_event_summaries["event_id"].astype(str)
            == selected_orbital_event
        ]

        if not selected_rows.empty:
            event_row = selected_rows.iloc[0].to_dict()

            # Retrieve latest validated CDM measurements for this event.
            event_records = valid_df[
                valid_df["event_id"].astype(str)
                == selected_orbital_event
            ].copy()

            if not event_records.empty:
                event_records = event_records.sort_values(
                    "message_time"
                )
                latest_record = event_records.iloc[-1]

                event_row["miss_distance"] = latest_record.get(
                    "miss_distance"
                )
                event_row["relative_speed"] = latest_record.get(
                    "relative_speed"
                )

            st.session_state.selected_event_id = selected_orbital_event

            # Event intelligence metrics.
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            with metric_1:
                st.metric(
                    "Risk Classification",
                    str(event_row.get("risk_band", "UNKNOWN"))
                )

            with metric_2:
                pc_value = event_row.get("predicted_pc")
                st.metric(
                    "Predicted Collision Probability",
                    f"{pc_value:.3e}"
                    if pd.notna(pc_value) else "Unavailable"
                )

            with metric_3:
                miss_value = event_row.get("miss_distance")
                st.metric(
                    "Miss Distance",
                    f"{miss_value:,.2f} m"
                    if pd.notna(miss_value) else "Unavailable"
                )

            with metric_4:
                speed_value = event_row.get("relative_speed")
                st.metric(
                    "Relative Speed",
                    f"{speed_value:,.2f} m/s"
                    if pd.notna(speed_value) else "Unavailable"
                )

            st.markdown("---")

                       # Interactive 3D visualization controls.
            st.markdown("### Visualization Controls")

            control_1, control_2, control_3, control_4 = st.columns(4)

            with control_1:
                show_satellites = st.checkbox(
                    "Satellite Objects",
                    value=True,
                    key="orbital_show_satellites"
                )

            with control_2:
                show_debris = st.checkbox(
                    "Debris Objects",
                    value=True,
                    key="orbital_show_debris"
                )

            with control_3:
                show_orbits = st.checkbox(
                    "Orbital Trajectories",
                    value=True,
                    key="orbital_show_orbits"
                )

            with control_4:
                show_labels = st.checkbox(
                    "Object Labels",
                    value=True,
                    key="orbital_show_labels"
                )

            orbital_figure = create_orbital_scene(
                event_row,
                show_satellites=show_satellites,
                show_debris=show_debris,
                show_orbits=show_orbits,
                show_labels=show_labels,
            )
            st.plotly_chart(
                orbital_figure,
                use_container_width=True,
                key="orbital_intelligence_3d_scene"
            )

            st.markdown("### Selected Event Intelligence")

            detail_col_1, detail_col_2 = st.columns(2)

            with detail_col_1:
                st.markdown(
                    f"**Event ID:** `{selected_orbital_event}`"
                )
                st.markdown(
                    f"**Object 1:** `{event_row.get('sat_1_id', 'N/A')}`"
                )
                st.markdown(
                    f"**Object 2:** `{event_row.get('sat_2_id', 'N/A')}`"
                )

            with detail_col_2:
                st.markdown(
                    f"**Time of Closest Approach (UTC):** "
                    f"`{event_row.get('tca_utc', 'N/A')}`"
                )
                st.markdown(
                    f"**Risk Band:** `{event_row.get('risk_band', 'N/A')}`"
                )
                st.markdown(
                    f"**Recommendation:** "
                    f"{event_row.get('recommendation') or 'Unavailable'}"
                )

            st.info(
                "Scientific transparency: the Earth sphere, orbital paths, "
                "object positions and connector are illustrative only. "
                "The current synthetic CDM fixture does not contain actual "
                "orbital elements or position vectors. Event measurements "
                "and model outputs are displayed from the available dataset."
            )


# Bottom Operational Disclaimer (Every screen)
st.markdown("""
<div class="disclaimer-box">
    <b>OPERATIONAL & SCIENTIFIC SAFETY DISCLAIMER</b><br>
    ORBIT PULSE is a satellite conjunction-risk intelligence and engineering decision-support system using historical and demonstration datasets.
    1. A final reported CDM Pc is a dataset-defined risk estimate, not proof that a physical collision occurred.<br>
    2. Model sensitivity to eligible historical inputs reflects algorithmic stability, not physical orbit covariance or real-world uncertainty bounds.<br>
    3. All recommendations are advisory. Never issue autonomous manoeuvre burn commands without qualified human operator review.
</div>
""", unsafe_allow_html=True)
