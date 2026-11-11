"""
ORBIT PULSE - Interactive Plotly Visualizations
===============================================
Generates mission-control grade charts adhering to Blueprint Section 5:
- Event Pc History with strict TCA-48h cutoff marker
- Risk Monitor Timeline
- Forecast Sensitivity perturbation plot (labeled model sensitivity, not orbit uncertainty)
- PR Curve & Confusion Matrix
- Ablation Comparison
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dashboard.theme import (
    PLOTLY_LAYOUT, COLOR_ACCENT_CYAN, COLOR_ACCENT_BLUE,
    COLOR_LOW, COLOR_UNCERTAIN, COLOR_HIGH, COLOR_CUTOFF, COLOR_TEXT_MUTED
)

def create_event_pc_history_chart(
    eligible_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    tca_utc: str,
    cutoff_utc: str,
    predicted_pc: float = None,
    final_actual_pc: float = None
) -> go.Figure:
    """
    Renders chronological Pc trajectory with TCA-48h cutoff marker.
    Differentiates eligible inputs from excluded post-cutoff observations.
    """
    fig = go.Figure()

    # 1. Eligible observations (admitted to model)
    eligible_plot = eligible_df[
        pd.to_numeric(eligible_df["collision_probability"], errors="coerce") > 0
    ] if not eligible_df.empty else eligible_df
    excluded_plot = excluded_df[
        pd.to_numeric(excluded_df["collision_probability"], errors="coerce") > 0
    ] if not excluded_df.empty else excluded_df

    if not eligible_plot.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(eligible_plot["message_time"]),
            y=eligible_plot["collision_probability"],
            mode="lines+markers",
            name="Eligible CDMs (Pre-Cutoff)",
            line=dict(color=COLOR_ACCENT_CYAN, width=2.5),
            marker=dict(size=9, color=COLOR_ACCENT_CYAN, symbol="circle"),
            hovertemplate="<b>Eligible CDM</b><br>Time: %{x|%Y-%m-%d %H:%M:%S UTC}<br>Pc: %{y:.2e}<extra></extra>"
        ))

    # 2. Excluded observations (post-cutoff, strictly for audit/timeline display)
    if not excluded_plot.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(excluded_plot["message_time"]),
            y=excluded_plot["collision_probability"],
            mode="lines+markers",
            name="Post-Cutoff CDMs (Excluded from Inputs)",
            line=dict(color=COLOR_TEXT_MUTED, width=1.5, dash="dot"),
            marker=dict(size=7, color=COLOR_TEXT_MUTED, symbol="circle-open"),
            hovertemplate="<b>Post-Cutoff CDM (Excluded)</b><br>Time: %{x|%Y-%m-%d %H:%M:%S UTC}<br>Pc: %{y:.2e}<extra></extra>"
        ))

    # 3. Model Predicted Final Pc marker
    if predicted_pc is not None and float(predicted_pc) > 0:
        tca_dt = pd.to_datetime(tca_utc)
        fig.add_trace(go.Scatter(
            x=[tca_dt],
            y=[predicted_pc],
            mode="markers",
            name="ORBIT PULSE Predicted Pc",
            marker=dict(size=14, color="#E040FB", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="<b>Forecasted Final Pc</b><br>Epoch: TCA<br>Predicted Pc: %{y:.2e}<extra></extra>"
        ))

    # 4. Final Reported Actual CDM Pc marker (ground truth label)
    if final_actual_pc is not None and float(final_actual_pc) > 0:
        tca_dt = pd.to_datetime(tca_utc)
        fig.add_trace(go.Scatter(
            x=[tca_dt],
            y=[final_actual_pc],
            mode="markers",
            name="Dataset Final Reported Pc",
            marker=dict(size=13, color="#00E676", symbol="square-open", line=dict(color="#00E676", width=2)),
            hovertemplate="<b>Final Reported Pc (Label)</b><br>Epoch: TCA<br>Final Pc: %{y:.2e}<extra></extra>"
        ))

    # Cutoff Vertical Line (TCA - 48h)
    cutoff_dt = pd.to_datetime(cutoff_utc)
    fig.add_vline(
        x=cutoff_dt,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_CUTOFF,
        annotation_text="Strict Cutoff: TCA - 48h",
        annotation_position="top left",
        annotation_font_color=COLOR_CUTOFF
    )

    # Operational review threshold line (Pc = 1e-4)
    fig.add_hline(
        y=1e-4,
        line_width=1,
        line_dash="dot",
        line_color=COLOR_UNCERTAIN,
        annotation_text="Operational Escalation Threshold (1e-4)",
        annotation_position="bottom right",
        annotation_font_color=COLOR_UNCERTAIN
    )

    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Event Collision Probability (Pc) Evolution & Cutoff Marker",
        xaxis_title="CDM Availability Timestamp (UTC)",
        yaxis_title="Reported Collision Probability (log scale)",
        yaxis_type="log"
    )
    fig.update_layout(layout)
    return fig

def create_risk_monitor_timeline_chart(events_summary_df: pd.DataFrame) -> go.Figure:
    """
    Scatter timeline of upcoming conjunction events colored by risk status.
    """
    fig = go.Figure()

    color_map = {
        "LOW": COLOR_LOW,
        "UNCERTAIN": COLOR_UNCERTAIN,
        "HIGH": COLOR_HIGH
    }

    for band in ["LOW", "UNCERTAIN", "HIGH"]:
        subset = events_summary_df[
            (events_summary_df["risk_band"] == band)
            & pd.to_numeric(events_summary_df["predicted_pc"], errors="coerce").gt(0)
        ]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(subset["tca"]),
                y=subset["predicted_pc"],
                mode="markers+text",
                name=f"{band} Risk Events",
                text=subset["event_id"],
                textposition="top center",
                textfont=dict(size=9, color=COLOR_TEXT_MUTED),
                marker=dict(
                    size=11,
                    color=color_map.get(band, COLOR_ACCENT_BLUE),
                    line=dict(color="#FFFFFF", width=1)
                ),
                hovertemplate="<b>Event %{text}</b><br>TCA: %{x|%Y-%m-%d %H:%M UTC}<br>Predicted Pc: %{y:.2e}<extra></extra>"
            ))

    fig.add_hline(
        y=1e-4,
        line_width=1.5,
        line_dash="dash",
        line_color=COLOR_HIGH,
        annotation_text="Escalation Threshold (1e-4)",
        annotation_position="top left"
    )

    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Active Conjunction Events Timeline (by TCA)",
        xaxis_title="Time of Closest Approach (TCA, UTC)",
        yaxis_title="Predicted Conjunction Risk (Pc, log scale)",
        yaxis_type="log"
    )
    fig.update_layout(layout)
    return fig

def create_forecast_sensitivity_chart(stability_data: Dict[str, Any]) -> go.Figure:
    """
    Visualizes leave-one-out perturbation predictions.
    Displays base forecast and perturbation range.
    """
    fig = go.Figure()

    base_log = stability_data.get("base_log_pc")
    min_log = stability_data.get("min_log_pc")
    max_log = stability_data.get("max_log_pc")
    perturbed = stability_data.get("perturbed_forecasts", [])

    if base_log is None or not perturbed:
        fig.add_annotation(text="Sensitivity experiment not estimable (<2 messages)", showarrow=False)
        fig.update_layout(PLOTLY_LAYOUT)
        return fig

    x_indices = list(range(len(perturbed)))
    labels = [f"Drop CDM {i+1}" for i in range(len(perturbed))]

    # Observed leave-one-out range; this is model sensitivity, not physical uncertainty.
    if min_log is not None and max_log is not None:
        fig.add_hrect(
            y0=min_log,
            y1=max_log,
            fillcolor=COLOR_ACCENT_CYAN,
            opacity=0.10,
            line_width=0,
            annotation_text="LOO sensitivity range",
            annotation_position="top left"
        )

    # Perturbation points
    fig.add_trace(go.Scatter(
        x=labels,
        y=perturbed,
        mode="markers",
        name="Perturbation Runs (Leave-One-Out)",
        marker=dict(size=10, color=COLOR_ACCENT_CYAN, symbol="circle")
    ))

    # Base forecast line
    fig.add_hline(
        y=base_log,
        line_width=2,
        line_color=COLOR_ACCENT_BLUE,
        line_dash="dash",
        annotation_text=f"Base Forecast: {base_log:.2f} log10(Pc)",
        annotation_position="top right"
    )

    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Model Sensitivity to Eligible Input History (Perturbation Test)",
        xaxis_title="Perturbation Run",
        yaxis_title="Predicted log10(Pc)",
        annotations=[
            dict(
                text="Note: Measures algorithmic model stability to message history; NOT physical orbit uncertainty.",
                xref="paper", yref="paper",
                x=0.5, y=-0.25,
                showarrow=False,
                font=dict(size=10, color=COLOR_TEXT_MUTED)
            )
        ]
    )
    fig.update_layout(layout)
    return fig

def create_ablation_comparison_chart(ablations: Dict[str, Any]) -> go.Figure:
    """Compare reported MAE values without fabricating zeros for missing experiments."""
    exp_names = [
        "A: Last-Pc Baseline",
        "B: + Pc History",
        "C: + Geometry & Cov",
        "D: + Evidence Features"
    ]
    keys = [
        "A_baseline_last_pc",
        "B_pc_history",
        "C_history_geometry_cov",
        "D_full_evidence"
    ]

    maes = []
    labels = []
    for key in keys:
        value = ablations.get(key, {}).get("metrics", {}).get("mae_log10")
        try:
            value = float(value) if value is not None and np.isfinite(float(value)) else None
        except (TypeError, ValueError):
            value = None
        maes.append(value)
        labels.append(f"{value:.4f}" if value is not None else "N/A")

    fig = go.Figure(data=[
        go.Bar(
            x=exp_names,
            y=maes,
            marker_color=[COLOR_ACCENT_BLUE, COLOR_ACCENT_CYAN, "#6366F1", "#8B5CF6"],
            text=labels,
            textposition="auto",
            hovertemplate="%{x}<br>MAE: %{y:.4f}<extra></extra>"
        )
    ])

    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Mandatory Ablation Study: MAE in log10(Pc) Space",
        xaxis_title="Ablation Experiment",
        yaxis_title="MAE (log10 space, lower is better)"
    )
    fig.update_layout(layout)
    return fig


def create_risk_distribution_chart(events_summary_df: pd.DataFrame) -> go.Figure:
    """Show event classifications without hiding ineligible/uncertain events."""
    if events_summary_df.empty or "risk_band" not in events_summary_df.columns:
        counts = pd.Series(dtype=int)
    else:
        counts = events_summary_df["risk_band"].fillna("UNCLASSIFIED").value_counts()

    ordered_labels = ["HIGH", "UNCERTAIN", "LOW", "MODEL_UNAVAILABLE"]
    labels = [label for label in ordered_labels if counts.get(label, 0) > 0]
    values = [int(counts[label]) for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        textinfo="label+value",
        hovertemplate="%{label}: %{value} events (%{percent})<extra></extra>"
    )])
    layout = PLOTLY_LAYOUT.copy()
    layout.update(title="Event Risk Distribution", showlegend=True)
    fig.update_layout(layout)
    if not labels:
        fig.add_annotation(text="No event classifications available", showarrow=False)
    return fig



def create_precision_recall_chart(baseline_curve: Dict[str, Any], model_curve: Dict[str, Any]) -> go.Figure:
    """Compare baseline and ML precision-recall curves on the held-out test split."""
    fig = go.Figure()
    for label, curve, color in [
        ("Last-Pc Baseline", baseline_curve, COLOR_ACCENT_BLUE),
        ("HistGradientBoosting", model_curve, COLOR_ACCENT_CYAN)
    ]:
        precision = curve.get("precision", [])
        recall = curve.get("recall", [])
        if precision and recall and len(precision) == len(recall):
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
            ))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Precision–Recall Curves (Held-out Test Events)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1], gridcolor="#243356"),
        yaxis=dict(range=[0, 1.05], gridcolor="#243356")
    )
    fig.update_layout(layout)
    if not fig.data:
        fig.add_annotation(text="Precision–recall curve unavailable", showarrow=False)
    return fig


def create_error_distribution_chart(baseline_errors: List[float], model_errors: List[float]) -> go.Figure:
    """Compare absolute regression errors in log10(Pc) space."""
    fig = go.Figure()
    if baseline_errors:
        fig.add_trace(go.Histogram(
            x=baseline_errors,
            name="Last-Pc Baseline",
            opacity=0.65,
            marker_color=COLOR_ACCENT_BLUE
        ))
    if model_errors:
        fig.add_trace(go.Histogram(
            x=model_errors,
            name="HistGradientBoosting",
            opacity=0.65,
            marker_color=COLOR_ACCENT_CYAN
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Absolute Error Distribution (Held-out Test Events)",
        xaxis_title="Absolute error in log10(Pc)",
        yaxis_title="Event count",
        barmode="overlay"
    )
    if not fig.data:
        fig.add_annotation(text="Error distribution unavailable", showarrow=False)
    return fig


def create_confusion_matrix_chart(cm_data: Dict[str, Any]) -> go.Figure:
    """Confusion matrix heatmap for operational threshold (Pc >= 1e-4)."""
    tp = cm_data.get("tp", 0)
    fp = cm_data.get("fp", 0)
    tn = cm_data.get("tn", 0)
    fn = cm_data.get("fn", 0)

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Routine (<1e-4)", "Predicted Escalate (>=1e-4)"]
    y = ["Actual Routine (<1e-4)", "Actual Dangerous (>=1e-4)"]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, "#1C2541"], [1, "#00E5FF"]],
        text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
        texttemplate="%{text}",
        showscale=False
    ))

    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title="Operational Confusion Matrix (Threshold Pc = 1e-4)",
        xaxis_title="Prediction",
        yaxis_title="Dataset Ground Truth Label"
    )
    fig.update_layout(layout)
    return fig
